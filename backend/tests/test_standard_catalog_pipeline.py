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

from classifier.alias_matcher import load_text_aliases, match_aliases
from classifier.candidate_retriever import candidate_label, retrieve_candidates
from classifier.llm_client import ItemSelection, StatusSelection, llm_select_catalog_item, llm_select_repair_status
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
        self.assertIn("CF-017-00", catalog_by_id)
        self.assertIn("CF-018-00", catalog_by_id)
        self.assertIn("CF-028-00", catalog_by_id)
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

    def test_text_aliases_load_and_expand_terms(self):
        entries = load_text_aliases()
        self.assertGreater(len(entries), 0)
        result = match_aliases("致远大厦外立面修缮工程")
        self.assertIn("外墙面", result.expanded_terms)
        self.assertTrue(
            any(
                hit.canonical_term == "外墙面" and "外立面" in hit.matched_patterns
                for hit in result.hits
            )
        )

    def test_text_aliases_do_not_return_catalog_or_hint_fields(self):
        result = match_aliases("消防技术咨询服务合同 消防泵故障维修")
        self.assertFalse(hasattr(result, "catalog_hits"))
        self.assertFalse(hasattr(result, "negative_hints"))
        self.assertFalse(hasattr(result, "review_hints"))

    def test_text_alias_expansion_uses_natural_retrieval_without_alias_source(self):
        candidates = retrieve_candidates("外立面翻新")
        by_id = {candidate.item.id: candidate for candidate in candidates}
        self.assertIn("CP-003-01", by_id)
        self.assertFalse(any("alias" in candidate.source for candidate in candidates))
        self.assertFalse(any("alias" in candidate.reason for candidate in candidates))

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

    @patch(
        "classifier.llm_client.request_llm_json",
        return_value={
            "catalog_id": "CF-028-00",
            "secondary_catalog_ids": [],
            "is_composite": False,
            "needs_review": False,
            "reason": "消防一级兜底",
        },
    )
    def test_llm_item_selection_rejects_catalog_id_outside_candidates(self, mock_request):
        candidates = [get_standard_catalog_by_id()["CF-022-01"]]
        result = llm_select_catalog_item("消防泵维修", candidates)
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

    @patch(
        "classifier.llm_client.llm_select_repair_status",
        return_value=StatusSelection(repair_status="维修", needs_review=False, reason="按工程名称判断为维修"),
    )
    @patch("classifier.llm_client.llm_select_catalog_item")
    def test_generic_elevator_candidates_include_internal_item_and_still_call_llm(self, mock_item, _mock_status):
        samples = [
            "志成花苑46台电梯维修",
            "电梯维修合同",
            "长江路366弄老旧电梯更新工程",
        ]

        def select_item(_project_name, candidates, _context_hints=None):
            candidate_ids = {item.id for item in candidates}
            self.assertIn("CF-017-00", candidate_ids)
            return ItemSelection(
                catalog_id="CF-017-00",
                secondary_catalog_ids=(),
                is_composite=False,
                needs_review=True,
                reason="普通电梯未明确具体部件",
            )

        mock_item.side_effect = select_item

        for text in samples:
            with self.subTest(text=text):
                ids = [candidate.item.id for candidate in retrieve_candidates(text)]
                self.assertIn("CF-017-00", ids)
                result = classify_project_standard(text)
                self.assertEqual(result["catalog_id"], "CF-017-00")
                self.assertTrue(result["needs_review"])
                for forbidden in ("保护规则", "forced_catalog_id", "domain_guard", "内部扩展项"):
                    self.assertNotIn(forbidden, result["reason"])
        self.assertEqual(mock_item.call_count, len(samples))

    @patch(
        "classifier.llm_client.llm_select_repair_status",
        return_value=StatusSelection(repair_status="维修", needs_review=False, reason="按工程名称判断为维修"),
    )
    @patch("classifier.llm_client.llm_select_catalog_item")
    def test_specific_elevator_and_system_cases_use_llm_candidates(self, mock_item, _mock_status):
        expected_by_text = {
            "商场自动扶梯扶手带更换": "CF-017-13",
            "自动人行道梯级链维修": "CF-017-13",
            "电梯控制柜维修": "CF-017-07",
            "电梯三方通话维修": "CF-017-10",
            "女儿墙外侧粉刷损坏修补": "CP-003-01",
            "楼道墙砖维修": "CP-004-02",
            "地下车库墙面维修": "CP-005-01",
            "视频监控系统改造": "CF-018-02",
            "弱电智能化工程": "CF-018-16",
            "安防监控全覆盖改造": "CF-018-02",
        }

        def select_item(project_name, candidates, _context_hints=None):
            expected_id = expected_by_text[project_name]
            candidate_ids = {item.id for item in candidates}
            self.assertIn(expected_id, candidate_ids)
            return ItemSelection(
                catalog_id=expected_id,
                secondary_catalog_ids=(),
                is_composite=False,
                needs_review=False,
                reason="测试固定目录",
            )

        mock_item.side_effect = select_item

        for text, expected_id in expected_by_text.items():
            with self.subTest(text=text):
                result = classify_project_standard(text)
                self.assertEqual(result["catalog_id"], expected_id)
                self.assertNotEqual(result["catalog_id"], OUT_OF_SCOPE_ID)
                for forbidden in ("保护规则", "forced_catalog_id", "domain_guard", "内部扩展项"):
                    self.assertNotIn(forbidden, result["reason"])

    @patch(
        "classifier.llm_client.llm_select_repair_status",
        return_value=StatusSelection(repair_status="维修", needs_review=False, reason="按工程名称判断为维修"),
    )
    @patch("classifier.llm_client.llm_select_catalog_item")
    def test_elevator_regression_final_selection_uses_specific_items(self, mock_item, _mock_status):
        expected_by_text = {
            "更换限速器、限速器钢丝绳": ("CF-017-06", ("CF-017-05",), False),
            "更换钢丝绳、制动器、电磁铁、导向轮、变频器主板": (
                "CF-017-05",
                ("CF-017-02", "CF-017-04"),
                False,
            ),
            "制动器、安全钳、门挂轮、导向轮等更换": (
                "CF-017-02",
                ("CF-017-06", "CF-017-04"),
                False,
            ),
            "电梯三方通话": ("CF-017-10", (), False),
            "电梯监控、部分共用光收发器维修更新": ("CF-018-02", ("CF-018-12",), False),
            "电梯钢丝绳更换": ("CF-017-05", (), False),
            "外立面翻新": ("CP-003-01", (), False),
            "墙砖翻新": ("CP-004-02", (), True),
        }

        def select_item(project_name, candidates, _context_hints=None):
            catalog_id, secondary_ids, needs_review = expected_by_text[project_name]
            candidate_ids = {item.id for item in candidates}
            self.assertIn(catalog_id, candidate_ids)
            self.assertTrue(set(secondary_ids).issubset(candidate_ids))
            if project_name.startswith("电梯监控"):
                self.assertFalse(any(item.category == "电梯" for item in candidates))
                self.assertNotIn(OUT_OF_SCOPE_ID, candidate_ids)
            else:
                self.assertNotIn("CF-017-00", candidate_ids)
            return ItemSelection(
                catalog_id=catalog_id,
                secondary_catalog_ids=secondary_ids,
                is_composite=bool(secondary_ids),
                needs_review=needs_review,
                reason="测试固定具体目录",
            )

        mock_item.side_effect = select_item

        for text, (expected_id, secondary_ids, expected_review) in expected_by_text.items():
            with self.subTest(text=text):
                result = classify_project_standard(text)
                self.assertEqual(result["catalog_id"], expected_id)
                self.assertEqual(result["secondary_catalog_ids"], list(secondary_ids))
                self.assertEqual(result["needs_review"], expected_review or bool(secondary_ids))
                self.assertNotIn("一级明确但二级未明确", result["reason"])

    @patch(
        "classifier.llm_client.llm_select_repair_status",
        return_value=StatusSelection(repair_status="维修", needs_review=False, reason="按工程名称判断为维修"),
    )
    @patch("classifier.llm_client.llm_select_catalog_item")
    def test_generic_elevator_final_selection_can_use_reviewed_fallback(self, mock_item, _mock_status):
        samples = ("普通电梯维修", "电梯更新")

        def select_item(_project_name, candidates, _context_hints=None):
            candidate_ids = {item.id for item in candidates}
            self.assertIn("CF-017-00", candidate_ids)
            return ItemSelection(
                catalog_id="CF-017-00",
                secondary_catalog_ids=(),
                is_composite=False,
                needs_review=True,
                reason="普通电梯未明确具体子项",
            )

        mock_item.side_effect = select_item

        for text in samples:
            with self.subTest(text=text):
                result = classify_project_standard(text)
                self.assertEqual(result["catalog_id"], "CF-017-00")
                self.assertTrue(result["needs_review"])

    def test_specific_elevator_parts_rank_before_generic_elevator_fallback(self):
        samples = {
            "电梯钢丝绳更换": "CF-017-05",
            "电梯制动器维修": "CF-017-02",
            "电梯控制柜维修": "CF-017-07",
            "自动扶梯扶手带更换": "CF-017-13",
        }
        for text, expected_id in samples.items():
            with self.subTest(text=text):
                candidates = retrieve_candidates(text)
                ids = [candidate.item.id for candidate in candidates]
                self.assertIn(expected_id, ids)
                self.assertNotIn("CF-017-00", ids)

    def test_specific_elevator_terms_do_not_recall_generic_elevator_fallback(self):
        samples = {
            "更换限速器、限速器钢丝绳": {"CF-017-06", "CF-017-05"},
            "更换钢丝绳、制动器、电磁铁、导向轮、变频器主板": {
                "CF-017-05",
                "CF-017-02",
                "CF-017-04",
            },
            "制动器、安全钳、门挂轮、导向轮等更换": {
                "CF-017-02",
                "CF-017-06",
                "CF-017-04",
            },
            "电梯三方通话": {"CF-017-10"},
        }
        for text, expected_ids in samples.items():
            with self.subTest(text=text):
                ids = {candidate.item.id for candidate in retrieve_candidates(text)}
                self.assertNotIn("CF-017-00", ids)
                self.assertTrue(expected_ids.issubset(ids))

    def test_elevator_monitoring_recalls_weak_current_not_elevator(self):
        candidates = retrieve_candidates("电梯监控、部分共用光收发器维修更新")
        ids = {candidate.item.id for candidate in candidates}
        categories = {candidate.item.category for candidate in candidates}
        self.assertIn("CF-018-02", ids)
        self.assertIn("CF-018-12", ids)
        self.assertNotIn("电梯", categories)

    def test_wall_aliases_recall_expected_candidates_without_alias_scoring(self):
        alias_result = match_aliases("外立面翻新")
        self.assertIn("外墙面", alias_result.expanded_terms)

        exterior = retrieve_candidates("外立面翻新")
        exterior_by_id = {candidate.item.id: candidate for candidate in exterior}
        self.assertIn("CP-003-01", exterior_by_id)
        self.assertFalse(any("alias" in candidate.source for candidate in exterior))

        interior = retrieve_candidates("墙砖翻新")
        interior_by_id = {candidate.item.id: candidate for candidate in interior}
        self.assertIn("CP-004-02", interior_by_id)
        self.assertFalse(any("alias" in candidate.source for candidate in interior))

    @patch("classifier.llm_client.llm_select_catalog_item")
    def test_family_out_of_scope_is_converted_to_fallback_catalog(self, mock_item):
        samples = {
            "电梯大修": "CF-017-00",
            "货梯维修": "CF-017-00",
            "电梯更新工程": "CF-017-00",
            "消防改造": "CF-028-00",
            "消防设施维修": "CF-028-00",
            "景泰大厦消防设施维修工程": "CF-028-00",
            "弱电智能化改造": "CF-018-00",
            "新梅淞南苑小区弱电智能化改造工程": "CF-018-00",
        }

        mock_item.return_value = ItemSelection(
            catalog_id=OUT_OF_SCOPE_ID,
            secondary_catalog_ids=(),
            is_composite=False,
            needs_review=True,
            reason="候选目录中没有合适项",
        )

        for text, expected_id in samples.items():
            with self.subTest(text=text):
                ids = [candidate.item.id for candidate in retrieve_candidates(text)]
                self.assertIn(expected_id, ids)
                result = classify_project_standard(text)
                self.assertEqual(result["catalog_id"], expected_id)
                self.assertTrue(result["needs_review"])
                self.assertEqual(result["repair_status"], "不确定")
                self.assertIn("LLM returned OUT_OF_SCOPE", result["reason"])

    @patch(
        "classifier.llm_client.llm_select_repair_status",
        return_value=StatusSelection(repair_status="维修", needs_review=False, reason="按工程名称判断为维修"),
    )
    @patch(
        "classifier.llm_client.llm_select_catalog_item",
        return_value=ItemSelection(
            catalog_id="CF-021-01",
            secondary_catalog_ids=(),
            is_composite=False,
            needs_review=False,
            reason="消防报警系统",
        ),
    )
    def test_catalog_id_outside_candidates_is_rejected_even_when_family_related(self, _mock_item, _mock_status):
        result = classify_project_standard("消防泵维修")
        self.assertEqual(result["catalog_id"], OUT_OF_SCOPE_ID)
        self.assertTrue(result["needs_review"])
        self.assertIn("outside retrieved candidates", result["reason"])

    def test_specific_fire_catalog_recalled_without_alias_source(self):
        candidates = retrieve_candidates("消防喷淋泵维修")
        by_id = {candidate.item.id: candidate for candidate in candidates}
        self.assertIn("CF-022-01", by_id)
        self.assertNotIn("alias", by_id["CF-022-01"].source)

    def test_family_fallback_appends_without_alias_source(self):
        candidates = retrieve_candidates("消防改造")
        ids = [candidate.item.id for candidate in candidates]
        self.assertIn("CF-028-00", ids)
        self.assertEqual(candidates[-1].item.id, "CF-028-00")
        self.assertEqual(candidates[-1].source, "family_recall")
        self.assertEqual(candidates[-1].retrieval_score, 0.0)

    def test_generic_terms_are_not_forced_to_specific_catalog_items(self):
        for text in ("线路维修", "设备改造", "门维修"):
            with self.subTest(text=text):
                candidates = retrieve_candidates(text)
                self.assertFalse(any("alias" in candidate.source for candidate in candidates))

    def test_elevator_as_location_or_weak_current_modifier_does_not_recall_generic_elevator_item(self):
        samples = ["电梯厅吊顶维修", "电梯厅墙面粉刷", "电梯监控维修", "梯控系统改造"]
        for text in samples:
            with self.subTest(text=text):
                ids = [candidate.item.id for candidate in retrieve_candidates(text)]
                self.assertNotIn("CF-017-00", ids)

    def test_water_supply_aliases_recall_expected_candidates(self):
        samples = {
            "污水总管改造": "CF-015-04",
            "落水管更换": "CF-015-04",
            "窨井维修": "CF-015-01",
            "地下泵房水泵更换": "CF-015-02",
            "生化池维修": "CF-015-03",
        }
        for text, expected_id in samples.items():
            with self.subTest(text=text):
                ids = [candidate.item.id for candidate in retrieve_candidates(text)]
                self.assertIn(expected_id, ids)
                self.assertNotIn("CF-015-00", ids)

    def test_water_supply_aliases_expand_multiple_specific_terms(self):
        result = match_aliases("生化池出口总管更换、疏通、更换窨井")
        self.assertTrue({"处理池", "给排水管道及附件", "井道"}.issubset(result.expanded_terms))
        ids = {candidate.item.id for candidate in retrieve_candidates("生化池出口总管更换、疏通、更换窨井")}
        self.assertTrue({"CF-015-01", "CF-015-03", "CF-015-04"}.issubset(ids))
        self.assertNotIn("CF-015-00", ids)

    def test_water_supply_aliases_expand_single_specific_terms(self):
        pipe_result = match_aliases("污水总管改造")
        self.assertIn("给排水管道及附件", pipe_result.expanded_terms)
        pipe_ids = {candidate.item.id for candidate in retrieve_candidates("污水总管改造")}
        self.assertIn("CF-015-04", pipe_ids)
        self.assertNotIn("CF-015-00", pipe_ids)

        pump_result = match_aliases("地下泵房水泵更换")
        self.assertIn("给排水机电设备及控制部分", pump_result.expanded_terms)
        pump_ids = {candidate.item.id for candidate in retrieve_candidates("地下泵房水泵更换")}
        self.assertIn("CF-015-02", pump_ids)
        self.assertNotIn("CF-015-00", pump_ids)

    def test_termite_candidate_is_limited_to_termite_text(self):
        for text in ["外墙防水维修", "电梯维修", "监控系统改造"]:
            with self.subTest(text=text):
                ids = [candidate.item.id for candidate in retrieve_candidates(text)]
                self.assertNotIn("TERMITE-001", ids)
        termite_ids = [candidate.item.id for candidate in retrieve_candidates("白蚁防治工程")]
        self.assertIn("TERMITE-001", termite_ids)

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
