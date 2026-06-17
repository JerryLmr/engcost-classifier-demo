import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from classifier.alias_matcher import load_text_aliases, match_aliases
from classifier.llm_client import (
    ItemSelection,
    StatusSelection,
    build_full_catalog_item_selection_prompt,
    llm_select_catalog_item_from_full_catalog,
    llm_select_repair_status,
)
from classifier.standard_catalog_loader import (
    OUT_OF_SCOPE_ID,
    catalog_label,
    get_standard_catalog_by_id,
    load_standard_catalog,
)
from services.analysis_service import load_records_from_path
from services.standard_classifier import classify_project_standard


NEW_HEADERS = [
    "工程名称",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "是否复合工程",
    "复合目录",
    "是否紧急维修",
    "是否白蚁相关",
    "是否建议复核",
    "分类依据",
]

OLD_HEADERS = {"置信度", "匹配类型", "分类方式", "候选目录"}


def make_workbook(*project_names: str) -> bytes:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.cell(row=1, column=1, value="工程名称")
    for row, value in enumerate(project_names, start=2):
        worksheet.cell(row=row, column=1, value=value)
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def make_classified_workbook(path: Path) -> None:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    for column, header in enumerate(NEW_HEADERS, start=1):
        worksheet.cell(row=1, column=column, value=header)
    values = [
        "屋面及外墙渗漏维修",
        "CP-002-03",
        "屋面",
        "防水层",
        "维修",
        "共用部位",
        "是",
        "CP-003-01 | 外墙面 | 面层",
        "否",
        "否",
        "是",
        "疑似复合工程",
    ]
    for column, value in enumerate(values, start=1):
        worksheet.cell(row=2, column=column, value=value)
    workbook.save(path)


class StandardCatalogPipelineTestCase(unittest.TestCase):
    def test_standard_catalog_loads_required_items(self):
        catalog = load_standard_catalog()
        catalog_by_id = get_standard_catalog_by_id()
        self.assertGreater(len(catalog), 0)
        self.assertNotIn(OUT_OF_SCOPE_ID, catalog_by_id)
        self.assertEqual(len(catalog_by_id), len(catalog))
        for item_id in ("CF-017-00", "CF-018-00", "CF-028-00", "TERMITE-001"):
            self.assertIn(item_id, catalog_by_id)
        for item in catalog:
            self.assertTrue(item.status_basis)
            self.assertEqual(catalog_label(item), f"{item.id} | {item.category} | {item.item}")

    def test_text_aliases_load_and_only_expand_terms(self):
        entries = load_text_aliases()
        self.assertGreater(len(entries), 0)
        result = match_aliases("致远大厦外立面修缮工程")
        self.assertIn("外墙面", result.expanded_terms)
        self.assertFalse(hasattr(result, "catalog_hits"))
        self.assertFalse(hasattr(result, "negative_hints"))
        self.assertFalse(hasattr(result, "review_hints"))

    def test_full_catalog_prompt_uses_compact_lines_without_status_basis(self):
        catalog = [get_standard_catalog_by_id()["CF-015-05"]]
        prompt = build_full_catalog_item_selection_prompt("减压阀更换", catalog, ["alias辅助扩展词（不能直接决定分类）：阀门"])
        self.assertIn("catalog_id | 标准对象 | 一级分类 | 二级分类 | 可选状态", prompt)
        self.assertIn("CF-015-05 | 共用设施设备 | 给排水系统", prompt)
        self.assertIn("alias辅助扩展词", prompt)
        self.assertNotIn("状态依据", prompt)
        self.assertNotIn("候选目录：", prompt)

    @patch(
        "classifier.llm_client.request_llm_json",
        return_value={
            "catalog_id": "CF-015-05",
            "secondary_catalog_ids": ["BAD-ID", "CF-015-04"],
            "is_composite": True,
            "needs_review": False,
            "reason": "对象为减压阀和管道",
        },
    )
    def test_full_catalog_selection_filters_invalid_secondary_ids(self, _mock_request):
        result = llm_select_catalog_item_from_full_catalog("减压阀及管道维修", load_standard_catalog())
        self.assertEqual(result.catalog_id, "CF-015-05")
        self.assertEqual(result.secondary_catalog_ids, ("CF-015-04",))
        self.assertTrue(result.needs_review)
        self.assertIn("已丢弃标准外 secondary id", result.reason)

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
    def test_full_catalog_selection_falls_back_after_invalid_primary_id(self, mock_request):
        result = llm_select_catalog_item_from_full_catalog("未知项目", load_standard_catalog())
        self.assertEqual(result.catalog_id, OUT_OF_SCOPE_ID)
        self.assertTrue(result.needs_review)
        self.assertTrue(result.invalid_after_retry)
        self.assertIn("LLM returned invalid catalog_id: BAD-ID", result.reason)
        self.assertEqual(mock_request.call_count, 2)

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
        "classifier.llm_client.llm_select_repair_status",
        return_value=StatusSelection(repair_status="更新", needs_review=False, reason="更换对应更新"),
    )
    @patch(
        "classifier.llm_client.llm_select_catalog_item_from_full_catalog",
        return_value=ItemSelection(
            catalog_id="CF-015-05",
            secondary_catalog_ids=(),
            is_composite=False,
            needs_review=False,
            reason="对象为减压阀",
        ),
    )
    def test_standard_classifier_result_shape_has_no_candidate_labels(self, _mock_item, _mock_status):
        result = classify_project_standard("减压阀更换")
        self.assertEqual(result["catalog_id"], "CF-015-05")
        self.assertEqual(result["repair_status"], "更新")
        self.assertNotIn("candidate_labels", result)
        self.assertEqual(result["secondary_catalog_ids"], [])
        self.assertEqual(result["secondary_catalog_labels"], [])
        self.assertFalse(result["needs_review"])

    @patch(
        "classifier.llm_client.llm_select_repair_status",
        return_value=StatusSelection(repair_status="维修", needs_review=False, reason="按工程名称判断为维修"),
    )
    @patch("classifier.llm_client.llm_select_catalog_item_from_full_catalog")
    def test_full_catalog_regression_cases_use_llm_selection_without_candidates(self, mock_item, _mock_status):
        expected_by_text = {
            "篮球场和网球场改造工程 篮球场和网球场地改造、更新": ("CF-011-02", (), True),
            "楼道粉刷": ("CP-004-02", (), False),
            "楼道整新": ("CP-004-02", (), False),
            "楼道整改": ("CP-004-02", (), False),
            "安防、人脸识别、监控、门禁": ("CF-018-00", ("CF-018-02", "CF-018-03"), True),
            "车牌识别系统更新": ("CF-018-07", (), False),
            "君临天下花园人行道维修": ("CF-007-02", (), False),
            "山鑫康城二次供水维修更新 二次供水改直供水": ("CF-015-02", ("CF-015-06",), True),
            "岛语树雅苑电梯钢带维修工程 电梯钢带维修": ("CF-017-05", (), False),
            "电梯维修": ("CF-017-00", (), True),
            "消防改造": ("CF-028-00", (), True),
            "弱电智能化改造": ("CF-018-00", (), True),
            "电梯监控维修": ("CF-018-02", (), False),
            "轿厢监控系统更新": ("CF-018-02", (), False),
            "消防检测维修": ("CF-028-00", (), True),
            "电梯维保维修": ("CF-017-00", (), True),
            "美丽家园改造：道路、绿化、监控": ("CF-007-02", ("CF-018-02",), True),
        }

        def select_item(project_name, _catalog, _context_hints=None):
            catalog_id, secondary_ids, needs_review = expected_by_text[project_name]
            return ItemSelection(
                catalog_id=catalog_id,
                secondary_catalog_ids=secondary_ids,
                is_composite=bool(secondary_ids),
                needs_review=needs_review,
                reason="测试固定目录",
            )

        mock_item.side_effect = select_item

        for text, (expected_id, secondary_ids, expected_review) in expected_by_text.items():
            with self.subTest(text=text):
                result = classify_project_standard(text)
                self.assertEqual(result["catalog_id"], expected_id)
                self.assertNotEqual(result["catalog_id"], OUT_OF_SCOPE_ID)
                self.assertEqual(result["secondary_catalog_ids"], list(secondary_ids))
                self.assertNotIn("candidate_labels", result)
                if expected_review or secondary_ids:
                    self.assertTrue(result["needs_review"])

    @patch(
        "classifier.llm_client.llm_select_catalog_item_from_full_catalog",
        side_effect=[
            ItemSelection(
                catalog_id=OUT_OF_SCOPE_ID,
                secondary_catalog_ids=(),
                is_composite=False,
                needs_review=True,
                reason="宏观项目未明确具体维修对象",
            ),
            ItemSelection(
                catalog_id=OUT_OF_SCOPE_ID,
                secondary_catalog_ids=(),
                is_composite=False,
                needs_review=True,
                reason="综合整治未明确具体维修对象",
            ),
        ],
    )
    def test_macro_projects_can_stay_out_of_scope_without_candidate_fallback(self, _mock_item):
        for text in ("美丽家园改造", "小区综合整治工程"):
            with self.subTest(text=text):
                result = classify_project_standard(text)
                self.assertEqual(result["catalog_id"], OUT_OF_SCOPE_ID)
                self.assertTrue(result["needs_review"])
                self.assertNotIn("candidate_labels", result)

    @patch(
        "classifier.llm_client.llm_select_repair_status",
        return_value=StatusSelection(repair_status="维修", needs_review=False, reason="按工程名称判断为维修"),
    )
    @patch(
        "classifier.llm_client.llm_select_catalog_item_from_full_catalog",
        return_value=ItemSelection(
            catalog_id="CP-002-03",
            secondary_catalog_ids=("CP-003-01",),
            is_composite=True,
            needs_review=True,
            reason="屋面及外墙复合工程",
        ),
    )
    def test_secondary_catalog_labels_are_preserved(self, _mock_item, _mock_status):
        result = classify_project_standard("屋面及外墙渗漏维修")
        self.assertEqual(result["secondary_catalog_ids"], ["CP-003-01"])
        self.assertEqual(result["secondary_catalog_labels"], [catalog_label(get_standard_catalog_by_id()["CP-003-01"])])
        self.assertNotIn("candidate_labels", result)

    def test_analysis_service_accepts_outputs_without_candidate_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "classified.xlsx"
            make_classified_workbook(path)
            records = load_records_from_path(path)
        self.assertEqual(len(records), 1)
        self.assertNotIn("candidate_labels", records[0])
        self.assertEqual(records[0]["secondary_candidates"], ["CP-003-01", "外墙面", "面层"])

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
