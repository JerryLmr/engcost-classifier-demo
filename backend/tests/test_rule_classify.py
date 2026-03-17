import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.classifier import rule_classify


class RuleClassifyTestCase(unittest.TestCase):
    def test_fire_pipeline_repair(self):
        result = rule_classify("消防喷淋管网维修")
        self.assertEqual(result["level1"], "消防")
        self.assertEqual(result["level2"], "消防管网维修")
        self.assertEqual(result["method"], "规则优先")

    def test_sewage_cleanup(self):
        result = rule_classify("化粪池清理维修")
        self.assertEqual(result["level1"], "污水")
        self.assertEqual(result["level2"], "化粪池清理维修")

    def test_wall_refurbish(self):
        result = rule_classify("外墙粉刷翻新工程")
        self.assertEqual(result["level1"], "外立面修缮")
        self.assertEqual(result["level2"], "外墙粉刷翻新")


if __name__ == "__main__":
    unittest.main()
