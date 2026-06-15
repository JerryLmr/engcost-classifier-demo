import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.catalog_loader import load_catalog
from services.classifier import classify_text


class CatalogClassifierTestCase(unittest.TestCase):
    def test_catalog_loads_fixed_items(self):
        catalog = load_catalog()
        self.assertGreater(len(catalog), 0)
        self.assertEqual(catalog[0].id, "001")
        self.assertTrue(all(item.level1 and item.level2 and item.level3_items for item in catalog))

    def test_direct_catalog_hits_return_three_levels(self):
        samples = [
            ("高压柜", "高压柜"),
            ("燃气炉", "燃气炉"),
            ("消防栓", "消防栓、箱"),
            ("污水泵", "污水泵、排水泵"),
            ("生活用泵", "生活用泵"),
            ("屋面防水", "平屋面"),
            ("电视监控控制台", "监控室设备"),
        ]
        for text, expected_level2 in samples:
            with self.subTest(text=text):
                result = classify_text(text)
                self.assertEqual(result["method"], "规则优先")
                self.assertEqual(result["level2"], expected_level2)
                self.assertIn("level3_item", result)
                self.assertIn(result["candidate_ids"][0], result["candidate_ids"])

    def test_same_domain_multi_item_needs_review(self):
        result = classify_text("屋面防水坡屋面维修")
        self.assertEqual(result["level1"], "屋面工程")
        self.assertEqual(result["level2"], "坡屋面")
        self.assertEqual(result["level3_item"], "维修坡屋面")
        self.assertIn("维修坡屋面", result["matched_level3_items"])

    def test_cross_domain_needs_review(self):
        result = classify_text("屋面防水消防栓更换")
        self.assertEqual(result["match_type"], "cross_domain")
        self.assertTrue(result["needs_review"])
        self.assertGreaterEqual(len(result["candidate_ids"]), 2)

    @patch("classifier.llm_client.request_llm_classification", side_effect=RuntimeError("ollama offline"))
    def test_low_confidence_falls_back_when_llm_unavailable(self, _mock_request):
        result = classify_text("某小区综合整治提升项目")
        self.assertEqual(result["method"], "默认兜底")
        self.assertEqual(result["confidence"], "低")
        self.assertEqual(result["match_type"], "fallback")
        self.assertTrue(result["needs_review"])
        self.assertIn("LLM 不可用", result["reason"])

    @patch("classifier.llm_client.request_llm_classification", side_effect=RuntimeError("ollama offline"))
    def test_generic_actions_do_not_create_high_confidence_rule_results(self, _mock_request):
        for text in ["维修", "更换", "改造"]:
            with self.subTest(text=text):
                result = classify_text(text)
                self.assertNotEqual(result["method"], "规则优先")
                self.assertNotEqual(result["confidence"], "高")

    @patch("classifier.llm_client.request_llm_classification", side_effect=RuntimeError("ollama offline"))
    def test_regression_samples_do_not_hit_monitor_room_or_garbage_by_action_only(self, _mock_request):
        samples = [
            ("外墙渗水维修", {"059"}),
            ("生活水泵维修", {"059"}),
            ("小区消火栓损坏维修", {"059"}),
            ("屋面维修工程", {"059"}),
            ("出入口改造车辆识别系统", {"027"}),
        ]
        for text, forbidden_ids in samples:
            with self.subTest(text=text):
                result = classify_text(text)
                self.assertNotIn(result["candidate_ids"][0] if result["candidate_ids"] else "", forbidden_ids)

    def test_real_objects_still_match_target_catalog_items(self):
        monitor = classify_text("电视监控控制台维修")
        self.assertEqual(monitor["candidate_ids"][0], "059")
        self.assertEqual(monitor["method"], "规则优先")
        self.assertIn("电视监控控制台", monitor["matched_level3_items"])

        garbage = classify_text("垃圾房维修")
        self.assertEqual(garbage["candidate_ids"][0], "027")
        self.assertEqual(garbage["method"], "规则优先")

    def test_derived_level3_item_hits_fill_specific_items(self):
        samples = [
            ("防盗门更新", "防盗门及附属设施"),
            ("外墙渗水维修", "修补外墙粉刷"),
            ("消火栓维修", "更换消防栓、箱"),
            ("生活水泵维修", "水泵维修(门)"),
        ]
        for text, expected_level3_item in samples:
            with self.subTest(text=text):
                result = classify_text(text)
                self.assertEqual(result["method"], "规则优先")
                self.assertEqual(result["level3_item"], expected_level3_item)
                self.assertIn(expected_level3_item, result["matched_level3_items"])
                self.assertNotEqual(result["level3_item"], "未明确具体细项")

    def test_object_only_rule_result_requires_review_and_no_high_confidence(self):
        result = classify_text("高压柜")
        self.assertEqual(result["method"], "规则优先")
        self.assertEqual(result["level3_item"], "未明确具体细项")
        self.assertEqual(result["matched_level3_items"], [])
        self.assertTrue(result["needs_review"])
        self.assertNotEqual(result["confidence"], "高")
        self.assertIn("未命中具体三级细项", result["reason"])


if __name__ == "__main__":
    unittest.main()
