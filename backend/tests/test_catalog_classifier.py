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
        self.assertTrue(all(item.level1 and item.level2 and item.level3 for item in catalog))

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
                self.assertTrue(result["level3"])
                self.assertEqual(result["confidence"], "高")
                self.assertIn(result["candidate_ids"][0], result["candidate_ids"])

    def test_same_domain_multi_item_needs_review(self):
        result = classify_text("屋面防水坡屋面维修")
        self.assertEqual(result["level1"], "屋面工程")
        self.assertEqual(result["match_type"], "same_domain_multi_item")
        self.assertTrue(result["needs_review"])
        self.assertGreaterEqual(len(result["candidate_ids"]), 2)

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


if __name__ == "__main__":
    unittest.main()
