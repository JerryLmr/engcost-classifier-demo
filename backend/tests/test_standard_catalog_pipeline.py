import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import openpyxl

from classifier.candidate_retriever import candidate_label, retrieve_candidates
from classifier.llm_client import llm_select_catalog_item, llm_select_repair_status
from classifier.standard_catalog_loader import (
    OUT_OF_SCOPE_ID,
    get_standard_catalog_by_id,
    load_standard_catalog,
)
from services.standard_classifier import classify_project_standard


NEW_HEADERS = [
    "工程名称",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "是否复合工程",
    "复合候选目录",
    "是否紧急维修",
    "是否白蚁相关",
    "是否建议复核",
    "候选目录",
    "分类依据",
]

OLD_HEADERS = {"置信度", "匹配类型", "分类方式"}


def make_workbook(*project_names: str) -> bytes:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.cell(row=1, column=1, value="工程名称")
    for row, value in enumerate(project_names, start=2):
        worksheet.cell(row=row, column=1, value=value)
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


class StandardCatalogPipelineTestCase(unittest.TestCase):
    def test_standard_catalog_loads_required_items(self):
        catalog = load_standard_catalog()
        catalog_by_id = get_standard_catalog_by_id()
        self.assertGreater(len(catalog), 0)
        self.assertNotIn(OUT_OF_SCOPE_ID, catalog_by_id)
        self.assertEqual(len(catalog_by_id), len(catalog))
        self.assertIn("CF-015-05", catalog_by_id)
        self.assertIn("减压阀", catalog_by_id["CF-015-05"].item)
        self.assertIn("TERMITE-001", catalog_by_id)
        for item in catalog:
            self.assertTrue(item.status_basis)
            self.assertTrue(all(key and value for key, value in item.status_basis.items()))

    def test_candidate_retriever_top5_contains_expected_items(self):
        samples = {
            "减压阀更换": "CF-015-05",
            "地下车库防水层渗漏维修": "CP-005-03",
            "电梯钢丝绳更换": "CF-017-05",
            "消防喷淋泵维修": "CF-022-01",
            "道闸系统维修": "CF-018-07",
            "视频监控摄像机更换": "CF-018-02",
            "屋面防水维修": "CP-002-03",
            "外墙面脱落维修": "CP-003-01",
            "白蚁灭治": "TERMITE-001",
        }
        for text, expected_id in samples.items():
            with self.subTest(text=text):
                ids = [candidate.item.id for candidate in retrieve_candidates(text)]
                self.assertIn(expected_id, ids)
                self.assertLessEqual(len(ids), 5)

    @patch(
        "classifier.llm_client.request_llm_json",
        return_value={
            "catalog_id": "CF-015-05",
            "secondary_catalog_ids": [],
            "is_composite": False,
            "needs_review": False,
            "reason": "对象为减压阀",
        },
    )
    def test_llm_item_selection_accepts_valid_catalog_id(self, _mock_request):
        candidates = [get_standard_catalog_by_id()["CF-015-05"]]
        result = llm_select_catalog_item("减压阀更换", candidates)
        self.assertEqual(result.catalog_id, "CF-015-05")
        self.assertFalse(result.needs_review)

    @patch(
        "classifier.llm_client.request_llm_json",
        return_value={
            "catalog_id": "BAD-ID",
            "secondary_catalog_ids": [],
            "is_composite": False,
            "needs_review": False,
            "reason": "bad",
        },
    )
    def test_llm_item_selection_falls_back_after_invalid_id_retry(self, mock_request):
        candidates = [get_standard_catalog_by_id()["CF-015-05"]]
        result = llm_select_catalog_item("减压阀更换", candidates)
        self.assertEqual(result.catalog_id, OUT_OF_SCOPE_ID)
        self.assertTrue(result.needs_review)
        self.assertTrue(result.invalid_after_retry)
        self.assertEqual(mock_request.call_count, 2)

    @patch("classifier.llm_client.request_llm_json", side_effect=ValueError("not json"))
    def test_llm_item_selection_falls_back_after_non_json_retry(self, mock_request):
        candidates = [get_standard_catalog_by_id()["CF-015-05"]]
        result = llm_select_catalog_item("减压阀更换", candidates)
        self.assertEqual(result.catalog_id, OUT_OF_SCOPE_ID)
        self.assertTrue(result.invalid_after_retry)
        self.assertEqual(mock_request.call_count, 2)

    @patch(
        "classifier.llm_client.request_llm_json",
        return_value={
            "catalog_id": OUT_OF_SCOPE_ID,
            "secondary_catalog_ids": [],
            "is_composite": False,
            "needs_review": True,
            "reason": "候选目录中没有合适项",
        },
    )
    def test_llm_item_selection_allows_out_of_scope(self, _mock_request):
        candidates = [get_standard_catalog_by_id()["CF-015-05"]]
        result = llm_select_catalog_item("未知项目", candidates)
        self.assertEqual(result.catalog_id, OUT_OF_SCOPE_ID)
        self.assertTrue(result.needs_review)

    @patch(
        "classifier.llm_client.request_llm_json",
        return_value={
            "catalog_id": "CP-002-03",
            "secondary_catalog_ids": ["CP-003-01"],
            "is_composite": True,
            "needs_review": True,
            "reason": "屋面及外墙复合工程",
        },
    )
    def test_llm_item_selection_allows_secondary_ids(self, _mock_request):
        catalog = get_standard_catalog_by_id()
        candidates = [catalog["CP-002-03"], catalog["CP-003-01"]]
        result = llm_select_catalog_item("屋面及外墙渗漏维修", candidates)
        self.assertEqual(result.catalog_id, "CP-002-03")
        self.assertEqual(result.secondary_catalog_ids, ("CP-003-01",))
        self.assertTrue(result.is_composite)

    @patch(
        "classifier.llm_client.request_llm_json",
        return_value={
            "repair_status": "非法状态",
            "needs_review": False,
            "reason": "bad",
        },
    )
    def test_llm_status_selection_falls_back_after_invalid_status_retry(self, mock_request):
        item = get_standard_catalog_by_id()["CF-015-05"]
        result = llm_select_repair_status("减压阀更换", item)
        self.assertEqual(result.repair_status, "不确定")
        self.assertTrue(result.needs_review)
        self.assertTrue(result.invalid_after_retry)
        self.assertEqual(mock_request.call_count, 2)

    @patch(
        "classifier.llm_client.llm_select_catalog_item",
        return_value=type(
            "Selection",
            (),
            {
                "catalog_id": "CF-015-05",
                "secondary_catalog_ids": (),
                "is_composite": False,
                "needs_review": False,
                "reason": "对象为减压阀",
                "invalid_after_retry": False,
            },
        )(),
    )
    @patch(
        "classifier.llm_client.llm_select_repair_status",
        return_value=type(
            "Status",
            (),
            {
                "repair_status": "更新",
                "needs_review": False,
                "reason": "更换对应更新",
                "invalid_after_retry": False,
            },
        )(),
    )
    def test_standard_classifier_result_shape(self, _mock_status, _mock_item):
        result = classify_project_standard("减压阀更换")
        self.assertEqual(result["catalog_id"], "CF-015-05")
        self.assertEqual(result["repair_status"], "更新")
        self.assertIn(candidate_label(get_standard_catalog_by_id()["CF-015-05"]), result["candidate_labels"])
        self.assertFalse(result["needs_review"])

    def test_batch_script_outputs_new_headers_only(self):
        script = ROOT / "scripts" / "batch_classify_excel.py"
        env = os.environ.copy()
        env["LLM_PROVIDER"] = "disabled"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "input.xlsx"
            output_file = tmp_path / "output.xlsx"
            input_file.write_bytes(make_workbook("减压阀更换", "屋面及外墙渗漏维修"))
            run = subprocess.run(
                [sys.executable, str(script), str(input_file), "-o", str(output_file), "--overwrite"],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            workbook = openpyxl.load_workbook(output_file)
            worksheet = workbook.active
            headers = [worksheet.cell(row=1, column=i).value for i in range(1, worksheet.max_column + 1)]
            self.assertEqual(headers, NEW_HEADERS)
            self.assertTrue(OLD_HEADERS.isdisjoint(set(headers)))
            self.assertEqual(worksheet.cell(row=2, column=1).value, "减压阀更换")
            self.assertEqual(worksheet.cell(row=2, column=2).value, OUT_OF_SCOPE_ID)

    def test_batch_script_accepts_multiple_files(self):
        script = ROOT / "scripts" / "batch_classify_excel.py"
        env = os.environ.copy()
        env["LLM_PROVIDER"] = "disabled"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "outputs"
            first = tmp_path / "first.xlsx"
            second = tmp_path / "second.xlsx"
            first.write_bytes(make_workbook("屋面防水维修"))
            second.write_bytes(make_workbook("白蚁灭治"))
            run = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(first),
                    str(second),
                    "-o",
                    str(output_dir),
                    "--overwrite",
                ],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            self.assertTrue((output_dir / "first_classified.xlsx").exists())
            self.assertTrue((output_dir / "second_classified.xlsx").exists())

    def test_batch_script_importable(self):
        script = ROOT / "scripts" / "batch_classify_excel.py"
        spec = importlib.util.spec_from_file_location("batch_classify_excel", script)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        self.assertEqual(module.RESULT_HEADERS, NEW_HEADERS)


if __name__ == "__main__":
    unittest.main()
