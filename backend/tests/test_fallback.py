import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.classifier import classify_text


class FallbackTestCase(unittest.TestCase):
    @patch("services.llm_client.request_llm_classification", side_effect=RuntimeError("ollama offline"))
    def test_fallback_when_llm_unavailable(self, _mock_request):
        result = classify_text("某小区公共区域综合整治提升项目")
        self.assertEqual(result["level1"], "公共设施")
        self.assertEqual(result["level2"], "公共区域维修")
        self.assertEqual(result["method"], "降级兜底")
        self.assertIn("LLM 不可用", result["reason"])


if __name__ == "__main__":
    unittest.main()
