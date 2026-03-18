import unittest
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import openpyxl
except ImportError:  # pragma: no cover
    openpyxl = None

try:
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError):  # pragma: no cover
    TestClient = None

if TestClient is not None:
    from app import app


@unittest.skipIf(openpyxl is None or TestClient is None, "测试依赖未安装")
class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("model", response.json())

    def test_single_classify(self):
        response = self.client.post("/api/classify", json={"text": "消防喷淋管网维修"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["level1"], "消防")
        self.assertEqual(data["level2"], "消防管网维修")
        self.assertIn("is_composite", data)
        self.assertIn("needs_review", data)
        self.assertIn("composite_reason", data)
        self.assertIn("secondary_candidates", data)

    def test_boundary_driven_single_classify(self):
        response = self.client.post("/api/classify", json={"text": "外墙渗漏水维修"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["level1"], "防水工程")
        self.assertEqual(data["level2"], "外墙防水")

    @patch("services.llm_client.request_llm_classification", side_effect=RuntimeError("offline"))
    def test_excel_classify(self, _mock_request):
        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        worksheet.cell(row=1, column=1, value="工程名称")
        worksheet.cell(row=2, column=1, value="消防喷淋管网维修")
        worksheet.cell(row=3, column=1, value="某小区公共区域综合整治提升项目")

        output = BytesIO()
        workbook.save(output)
        output.seek(0)

        response = self.client.post(
            "/api/classify-excel",
            files={
                "file": (
                    "sample.xlsx",
                    output.getvalue(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["content-type"],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        workbook = openpyxl.load_workbook(BytesIO(response.content))
        worksheet = workbook.active
        headers = [worksheet.cell(row=1, column=i).value for i in range(1, worksheet.max_column + 1)]
        self.assertIn("是否复合工程", headers)
        self.assertIn("是否建议复核", headers)
        self.assertIn("复合原因", headers)
        self.assertIn("候选分类", headers)


if __name__ == "__main__":
    unittest.main()
