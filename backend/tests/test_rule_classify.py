import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.classifier import classify_text, rule_classify


class RuleClassifyTestCase(unittest.TestCase):
    def test_wall_leak_prefers_waterproof(self):
        result = rule_classify("外墙渗漏水维修")
        self.assertEqual(result["level1"], "防水工程")
        self.assertEqual(result["level2"], "外墙防水")

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

    def test_facade_leak_stays_consistent(self):
        result = rule_classify("外立面漏水维修")
        self.assertEqual(result["level1"], "防水工程")
        self.assertEqual(result["level2"], "外墙防水")

    def test_fire_door_update_prefers_equipment_replacement(self):
        result = rule_classify("防火门更新维修")
        self.assertEqual(result["level1"], "消防")
        self.assertEqual(result["level2"], "消防设备更换")

    def test_monitor_upgrade_prefers_system_upgrade(self):
        result = rule_classify("监控设备改造工程")
        self.assertEqual(result["level1"], "监控")
        self.assertEqual(result["level2"], "监控系统升级")

    def test_elevator_part_replacement(self):
        result = rule_classify("电梯钢丝绳更换")
        self.assertEqual(result["level1"], "电梯")
        self.assertEqual(result["level2"], "电梯部件更换")

    def test_elevator_update_prefers_upgrade(self):
        result = rule_classify("电梯更新项目")
        self.assertEqual(result["level1"], "电梯")
        self.assertEqual(result["level2"], "电梯改造升级")

    def test_composite_result_has_flags(self):
        result = classify_text("小区道路改造及绿化补种施工合同 道路拓宽及绿化补种")
        self.assertTrue(result["is_composite"])
        self.assertTrue(result["needs_review"])
        self.assertIsNotNone(result["composite_reason"])
        self.assertIn("绿化景观", result["secondary_candidates"])

    def test_single_result_has_default_flags(self):
        result = classify_text("灭火器过期更换")
        self.assertFalse(result["is_composite"])
        self.assertFalse(result["needs_review"])
        self.assertEqual(result["secondary_candidates"], [])

    def test_same_domain_multi_system_is_not_composite(self):
        result = classify_text("消防栓以及自动报警系统维修")
        self.assertFalse(result["is_composite"])
        self.assertTrue(result["needs_review"])
        self.assertIsNone(result["composite_reason"])
        self.assertEqual(result["secondary_candidates"], [])

    def test_same_project_multi_building_is_not_composite(self):
        result = classify_text("A楼和B楼电梯维修")
        self.assertFalse(result["is_composite"])
        self.assertFalse(result["needs_review"])

    def test_same_project_multi_part_is_not_composite(self):
        result = classify_text("外墙及屋顶渗漏水维修")
        self.assertFalse(result["is_composite"])

    def test_cross_domain_project_is_composite(self):
        result = classify_text("电梯更新及门禁更换")
        self.assertTrue(result["is_composite"])
        self.assertTrue(result["needs_review"])
        self.assertIn("门禁设施", result["secondary_candidates"])


if __name__ == "__main__":
    unittest.main()
