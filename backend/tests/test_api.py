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
except ImportError:  # pragma: no cover
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


if __name__ == "__main__":
    unittest.main()
