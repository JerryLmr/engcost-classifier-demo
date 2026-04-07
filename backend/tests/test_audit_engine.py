import json
import sys
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError):  # pragma: no cover
    TestClient = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.audit_service import audit_project
from services.mapping_service import map_project_name

if TestClient is not None:
    from app import app


def find_catalog_id(full_path: str) -> int:
    catalog_path = Path(__file__).resolve().parents[1] / "config" / "object_catalog.json"
    with catalog_path.open("r", encoding="utf-8") as fp:
        catalog = json.load(fp)
    for item in catalog["items"]:
        if item["full_path"] == full_path:
            return item["id"]
    raise AssertionError(f"未找到目录对象: {full_path}")


@unittest.skipIf(TestClient is None, "测试依赖未安装")
class AuditApiTestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_legacy_classify_endpoint_is_still_available(self):
        response = self.client.post("/api/classify", json={"text": "消防喷淋管网维修"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["level1"], "消防")
        self.assertEqual(data["level2"], "消防管网维修")


class AuditServiceTestCase(unittest.TestCase):
    def _audit(self, payload):
        mapping_result = map_project_name(payload["project_name"])
        return audit_project(payload, mapping_result)

    def test_audit_maps_catalog_object_dynamically(self):
        data = self._audit({"project_name": "3号楼电梯曳引机维修"})

        expected_id = find_catalog_id("电梯/曳引系统/曳引机")
        self.assertIn(expected_id, data["matched_object_ids"])
        self.assertTrue(any(item["full_path"] == "电梯/曳引系统/曳引机" for item in data["mapped_objects"]))
        self.assertIn("repairable_object", data["normalized_tags"])
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertIn("建议补充表决、公示及审价等流程材料", data["reasons"])
        self.assertNotIn("当前未命中明确结论规则，需补充进一步审计材料", data["reasons"])
        self.assertEqual(data["reason_codes"], [])
        self.assertEqual(data["missing_items"], [])

    def test_process_reason_requires_explicit_false_input(self):
        data = self._audit({"project_name": "3号楼电梯曳引机维修", "has_vote": False})
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertIn("MISSING_VOTE", data["reason_codes"])
        self.assertIn("has_vote", data["missing_items"])
        self.assertNotIn("建议补充表决、公示及审价等流程材料", data["reasons"])

    def test_generic_input_returns_insufficient_info(self):
        data = self._audit({"project_name": "维修工程"})
        self.assertEqual(data["overall_result"], "manual_review")
        self.assertEqual(data["reason_codes"], ["INSUFFICIENT_INFO"])

    def test_nonempty_outside_catalog_input_returns_outside_catalog(self):
        data = self._audit({"project_name": "小区设施优化工程"})
        self.assertEqual(data["overall_result"], "manual_review")
        self.assertEqual(data["reason_codes"], ["OUTSIDE_CATALOG"])

    def test_exclusion_flow_still_wins_without_catalog_hit(self):
        data = self._audit({"project_name": "垃圾桶更换"})
        self.assertEqual(data["overall_result"], "non_compliant")
        self.assertIn("CLEANING_SANITATION", data["reason_codes"])

    def test_weak_gray_case_defaults_to_need_supplement(self):
        data = self._audit({"project_name": "楼道窗户玻璃维修"})
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertIn("MISSING_GRAY_CASE_EVIDENCE", data["reason_codes"])

    def test_weak_gray_case_can_transition_to_specific_normal_flow_prompt(self):
        data = self._audit(
            {
                "project_name": "楼道窗户玻璃维修",
                "gray_case_evidence_complete": True,
                "is_property_service_scope": False,
                "is_common_part": True,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], [])
        self.assertIn("建议补充表决、公示及审价等流程材料", data["reasons"])
        self.assertNotIn("当前未命中明确结论规则，需补充进一步审计材料", data["reasons"])
        self.assertEqual(data["missing_items"], [])

    def test_multi_project_cross_domain_adds_reason_code(self):
        data = self._audit(
            {"project_name": "防盗门油漆，人行道闸、绿化补种、电子门禁"},
        )
        self.assertEqual(data["overall_result"], "manual_review")
        self.assertIn("MULTI_PROJECT", data["reason_codes"])
        self.assertIn("CROSS_DOMAIN_PROJECT", data["reason_codes"])


if __name__ == "__main__":
    unittest.main()
