import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier import llm_client
from classifier.catalog_loader import load_catalog
from services.classifier import classify_text, classify_text_llm_only


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

    @patch(
        "classifier.llm_client.request_llm_classification",
        return_value={
            "id": "046",
            "level3_item": "更换消防栓、箱",
            "reason": "主工程对象为消防栓",
            "needs_review": False,
        },
    )
    def test_llm_only_returns_valid_catalog_item(self, _mock_request):
        result = classify_text_llm_only("消防栓更换")
        self.assertEqual(result["method"], "LLM-only")
        self.assertEqual(result["match_type"], "llm_only")
        self.assertEqual(result["candidate_ids"], ["046"])
        self.assertEqual(result["level2"], "消防栓、箱")
        self.assertEqual(result["level3_item"], "更换消防栓、箱")
        self.assertEqual(result["matched_level3_items"], ["更换消防栓、箱"])
        self.assertFalse(result["needs_review"])

    @patch(
        "classifier.llm_client.request_llm_classification",
        return_value={
            "id": "046",
            "level3_item": "自造消防细项",
            "reason": "主工程对象为消防栓",
            "needs_review": False,
        },
    )
    def test_llm_only_invalid_level3_item_requires_review(self, _mock_request):
        result = classify_text_llm_only("消防栓更换")
        self.assertEqual(result["method"], "LLM-only")
        self.assertEqual(result["level3_item"], "未明确具体细项")
        self.assertEqual(result["matched_level3_items"], [])
        self.assertTrue(result["needs_review"])
        self.assertIn("不在标准目录中", result["reason"])

    @patch(
        "classifier.llm_client.request_llm_classification",
        return_value={
            "id": "049",
            "level3_item": " 更换￠150以上（含￠150）管道 ",
            "reason": "消防管道更换",
            "needs_review": False,
        },
    )
    def test_llm_only_normalizes_level3_item_format_only(self, _mock_request):
        result = classify_text_llm_only("消防管道更换")
        self.assertEqual(result["candidate_ids"], ["049"])
        self.assertEqual(result["level3_item"], "更换￠150以上(含￠150)管道")
        self.assertEqual(result["matched_level3_items"], ["更换￠150以上(含￠150)管道"])
        self.assertFalse(result["needs_review"])

    @patch(
        "classifier.llm_client.request_llm_classification",
        return_value={
            "id": "046",
            "level3_item": "更换￠150以上(含￠150)管道",
            "reason": "消防管道更换",
            "needs_review": False,
        },
    )
    def test_llm_only_does_not_correct_level3_item_across_ids(self, _mock_request):
        result = classify_text_llm_only("消防管道更换")
        self.assertEqual(result["candidate_ids"], ["046"])
        self.assertEqual(result["level3_item"], "未明确具体细项")
        self.assertEqual(result["matched_level3_items"], [])
        self.assertTrue(result["needs_review"])
        self.assertIn("不在标准目录中", result["reason"])

    @patch("classifier.llm_client.request_llm_classification", side_effect=RuntimeError("offline"))
    def test_llm_only_falls_back_when_llm_unavailable(self, _mock_request):
        result = classify_text_llm_only("某小区综合整治提升项目")
        self.assertEqual(result["method"], "默认兜底")
        self.assertEqual(result["confidence"], "低")
        self.assertEqual(result["match_type"], "fallback")
        self.assertTrue(result["needs_review"])
        self.assertIn("LLM-only 模式不可用", result["reason"])

    @patch("classifier.llm_client.requests.post")
    @patch("classifier.llm_client.LLM_PROVIDER", "lmstudio")
    def test_lmstudio_classification_parses_json_content(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "{\"id\":\"049\",\"level3_item\":\"更换￠150以上(含￠150)管道\","
                            "\"reason\":\"消防管道\",\"needs_review\":true}"
                        )
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = llm_client.request_llm_classification("消防管道更换", [])

        self.assertEqual(result["id"], "049")
        self.assertEqual(result["level3_item"], "更换￠150以上(含￠150)管道")
        self.assertEqual(result["reason"], "消防管道")
        self.assertTrue(result["needs_review"])
        mock_response.raise_for_status.assert_called_once()
        mock_post.assert_called_once()
        _args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["model"], "qwen/qwen3.6-35b-a3b")
        self.assertEqual(kwargs["json"]["temperature"], 0)
        self.assertEqual(kwargs["json"]["max_tokens"], 256)
        self.assertNotIn("response_format", kwargs["json"])

    @patch("classifier.llm_client.requests.post")
    @patch("classifier.llm_client.LLM_PROVIDER", "lmstudio")
    def test_lmstudio_json_omits_response_format_by_default(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "{\"catalog_id\":\"001\"}"}}]
        }
        mock_post.return_value = mock_response

        result = llm_client.request_llm_json("选择目录")

        self.assertEqual(result, {"catalog_id": "001"})
        _args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["max_tokens"], 256)
        self.assertNotIn("response_format", kwargs["json"])

    @patch("classifier.llm_client.requests.post")
    @patch("classifier.llm_client.LMSTUDIO_RESPONSE_FORMAT", "text")
    @patch("classifier.llm_client.LLM_PROVIDER", "lmstudio")
    def test_lmstudio_json_supports_text_response_format(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "{\"catalog_id\":\"001\"}"}}]
        }
        mock_post.return_value = mock_response

        result = llm_client.request_llm_json("选择目录")

        self.assertEqual(result, {"catalog_id": "001"})
        _args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["response_format"], {"type": "text"})

    @patch("classifier.llm_client.requests.post")
    @patch("classifier.llm_client.LMSTUDIO_RESPONSE_FORMAT", "json_schema")
    @patch("classifier.llm_client.LLM_PROVIDER", "lmstudio")
    def test_lmstudio_json_supports_json_schema_response_format(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "{\"catalog_id\":\"001\"}"}}]
        }
        mock_post.return_value = mock_response

        result = llm_client.request_llm_json("选择目录")

        self.assertEqual(result, {"catalog_id": "001"})
        _args, kwargs = mock_post.call_args
        response_format = kwargs["json"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "classification_result")
        self.assertEqual(response_format["json_schema"]["schema"]["type"], "object")
        self.assertTrue(response_format["json_schema"]["schema"]["additionalProperties"])

    @patch("classifier.llm_client.requests.post")
    @patch("classifier.llm_client.LMSTUDIO_RESPONSE_FORMAT", "json_object")
    @patch("classifier.llm_client.LLM_PROVIDER", "lmstudio")
    def test_lmstudio_json_rejects_unsupported_response_format(self, mock_post):
        with self.assertRaisesRegex(ValueError, "Unsupported LMSTUDIO_RESPONSE_FORMAT"):
            llm_client.request_llm_json("选择目录")
        mock_post.assert_not_called()

    def test_extract_json_object_accepts_direct_json(self):
        self.assertEqual(llm_client._extract_json_object("{\"catalog_id\":\"001\"}"), {"catalog_id": "001"})

    def test_extract_json_object_strips_reasoning(self):
        text = "<think>先分析候选</think>{\"catalog_id\":\"001\"}"
        self.assertEqual(llm_client._extract_json_object(text), {"catalog_id": "001"})

    def test_extract_json_object_unwraps_markdown_fence(self):
        text = "```json\n{\"catalog_id\":\"001\"}\n```"
        self.assertEqual(llm_client._extract_json_object(text), {"catalog_id": "001"})

    def test_extract_json_object_accepts_prose_around_json(self):
        text = "结果如下：{\"catalog_id\":\"001\",\"reason\":\"匹配\"}。"
        self.assertEqual(
            llm_client._extract_json_object(text),
            {"catalog_id": "001", "reason": "匹配"},
        )

    def test_extract_json_object_returns_first_valid_object(self):
        text = "无效片段 {not json} 有效 {\"catalog_id\":\"001\",\"nested\":{\"ok\":true}} 结束"
        self.assertEqual(
            llm_client._extract_json_object(text),
            {"catalog_id": "001", "nested": {"ok": True}},
        )


if __name__ == "__main__":
    unittest.main()
