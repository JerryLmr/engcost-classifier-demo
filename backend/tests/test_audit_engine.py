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
from services.audit_pipeline_service import run_audit_pipeline
from services.mapping_service import map_project_name

if TestClient is not None:
    from app import app


def find_catalog_id(full_path: str) -> int:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "rules" / "repairable_object_catalog.json",
        root / "config" / "object_catalog.json",
    ]
    catalog = None
    for catalog_path in candidates:
        if not catalog_path.exists():
            continue
        with catalog_path.open("r", encoding="utf-8") as fp:
            catalog = json.load(fp)
        break
    if catalog is None:
        raise AssertionError("未找到维修对象目录配置文件")
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

    def _audit_pipeline(self, payload):
        return run_audit_pipeline(payload)

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
        self.assertIn("sub_audits", data)
        self.assertIn("process_audit", data["sub_audits"])
        self.assertEqual(data["sub_audits"]["process_audit"]["result"], "need_supplement")
        self.assertEqual(data["sub_audits"]["process_audit"]["display_result"], "需补充材料")
        self.assertIn("document_extraction_targets", data)
        self.assertIn("vote_documents", data["document_extraction_targets"])

    def test_process_reason_requires_explicit_false_input(self):
        data = self._audit({"project_name": "3号楼电梯曳引机维修", "has_vote": False})
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertIn("MISSING_VOTE", data["reason_codes"])
        self.assertIn("has_vote", data["missing_items"])
        self.assertNotIn("建议补充表决、公示及审价等流程材料", data["reasons"])
        self.assertEqual(data["sub_audits"]["process_audit"]["reason_codes"], ["MISSING_VOTE"])
        self.assertEqual(data["sub_audits"]["document_completeness_audit"]["reason_codes"], [])

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

    def test_scope_audit_only_returns_compliant_when_scope_fact_is_explicit(self):
        data = self._audit(
            {
                "project_name": "外墙渗漏维修",
                "is_common_part": True,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["sub_audits"]["scope_audit"]["result"], "compliant")
        self.assertEqual(data["sub_audits"]["scope_audit"]["reason_codes"], ["IN_SCOPE_COMMON_PART"])
        self.assertEqual(data["sub_audits"]["scope_audit"]["facts_used"], ["is_common_part"])

    def test_scope_audit_can_mark_common_facility_in_scope(self):
        data = self._audit(
            {
                "project_name": "3号楼电梯曳引机维修",
                "scope_facts": {
                    "is_common_facility": True,
                },
            }
        )
        self.assertEqual(data["sub_audits"]["scope_audit"]["result"], "compliant")
        self.assertEqual(data["sub_audits"]["scope_audit"]["reason_codes"], ["IN_SCOPE_COMMON_FACILITY"])
        self.assertEqual(data["overall_result"], "need_supplement")

    def test_scope_audit_property_service_scope_changes_overall_result(self):
        data = self._audit(
            {
                "project_name": "3号楼电梯曳引机维修",
                "is_property_service_scope": True,
            }
        )
        self.assertEqual(data["overall_result"], "non_compliant")
        self.assertEqual(data["reason_codes"], ["PROPERTY_SERVICE_SCOPE"])
        self.assertEqual(data["sub_audits"]["scope_audit"]["reason_codes"], ["PROPERTY_SERVICE_SCOPE"])

    def test_process_audit_can_require_contract_without_hitting_document_audit(self):
        data = self._audit(
            {
                "project_name": "消防喷淋系统维修",
                "has_contract": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_CONTRACT"])
        self.assertEqual(data["sub_audits"]["process_audit"]["reason_codes"], ["MISSING_CONTRACT"])
        self.assertEqual(data["sub_audits"]["document_completeness_audit"]["reason_codes"], [])

    def test_document_completeness_can_require_invoice(self):
        data = self._audit(
            {
                "project_name": "3号楼电梯曳引机维修",
                "has_invoice": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_INVOICE"])
        self.assertEqual(data["sub_audits"]["document_completeness_audit"]["reason_codes"], ["MISSING_INVOICE"])
        self.assertEqual(data["sub_audits"]["process_audit"]["reason_codes"], [])

    def test_document_completeness_can_require_completion_materials(self):
        data = self._audit(
            {
                "project_name": "屋面防水维修",
                "has_completion_report": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_COMPLETION_REPORT"])
        self.assertEqual(data["sub_audits"]["document_completeness_audit"]["reason_codes"], ["MISSING_COMPLETION_REPORT"])

    def test_document_completeness_can_require_settlement_report(self):
        data = self._audit(
            {
                "project_name": "屋面防水维修",
                "has_settlement_report": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_SETTLEMENT_REPORT"])
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["reason_codes"],
            ["MISSING_SETTLEMENT_REPORT"],
        )
        self.assertEqual(data["sub_audits"]["process_audit"]["reason_codes"], [])

    def test_document_completeness_can_require_payment_proof(self):
        data = self._audit(
            {
                "project_name": "3号楼电梯曳引机维修",
                "has_payment_proof": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_PAYMENT_PROOF"])
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["reason_codes"],
            ["MISSING_PAYMENT_PROOF"],
        )
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["facts_used"],
            ["has_payment_proof"],
        )
        self.assertEqual(data["sub_audits"]["process_audit"]["reason_codes"], [])

    def test_document_completeness_can_require_site_photos_for_non_emergency_case(self):
        data = self._audit(
            {
                "project_name": "电梯曳引机维修",
                "has_site_photos": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_SITE_PHOTOS"])
        self.assertEqual(data["missing_items"], ["has_site_photos"])
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["reason_codes"],
            ["MISSING_SITE_PHOTOS"],
        )
        self.assertEqual(data["sub_audits"]["process_audit"]["reason_codes"], [])

    def test_document_completeness_can_require_rectification_notice_for_non_emergency_case(self):
        data = self._audit(
            {
                "project_name": "屋面防水维修",
                "has_rectification_notice": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_RECTIFICATION_NOTICE"])
        self.assertEqual(data["missing_items"], ["has_rectification_notice"])
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["reason_codes"],
            ["MISSING_RECTIFICATION_NOTICE"],
        )
        self.assertEqual(data["sub_audits"]["process_audit"]["reason_codes"], [])

    def test_document_completeness_can_require_emergency_proof(self):
        data = self._audit(
            {
                "project_name": "排水管爆裂维修",
                "is_emergency": True,
                "has_emergency_proof": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_EMERGENCY_DOC"])
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["reason_codes"],
            ["MISSING_EMERGENCY_DOC"],
        )

    def test_document_completeness_can_require_acceptance_record(self):
        data = self._audit(
            {
                "project_name": "屋面防水维修",
                "has_acceptance_record": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_COMPLETION_REPORT"])
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["reason_codes"],
            ["MISSING_COMPLETION_REPORT"],
        )

    def test_document_completeness_prioritizes_missing_completion_and_acceptance_together(self):
        data = self._audit(
            {
                "project_name": "屋面防水维修",
                "has_completion_report": False,
                "has_acceptance_record": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_COMPLETION_REPORT"])
        self.assertEqual(data["missing_items"], ["has_completion_report", "has_acceptance_record"])
        self.assertEqual(data["reasons"], ["缺少完工报告及验收记录"])
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["missing_items"],
            ["has_completion_report", "has_acceptance_record"],
        )
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["reasons"],
            ["缺少完工报告及验收记录"],
        )

    def test_document_completeness_can_require_damage_assessment_for_gray_case(self):
        data = self._audit(
            {
                "project_name": "楼道窗户玻璃维修",
                "has_damage_assessment": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_GRAY_CASE_EVIDENCE"])
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["reason_codes"],
            ["MISSING_GRAY_CASE_EVIDENCE"],
        )

    def test_scope_unknown_object_boundary_only_applies_to_weak_gray_case_with_complete_evidence(self):
        data = self._audit(
            {
                "project_name": "楼道窗户玻璃维修",
                "gray_case_evidence_complete": True,
            }
        )
        self.assertEqual(data["overall_result"], "manual_review")
        self.assertEqual(data["reason_codes"], ["OUTSIDE_SCOPE_UNKNOWN_OBJECT"])
        self.assertEqual(data["sub_audits"]["scope_audit"]["result"], "manual_review")
        self.assertEqual(data["sub_audits"]["scope_audit"]["reason_codes"], ["OUTSIDE_SCOPE_UNKNOWN_OBJECT"])
        self.assertEqual(
            data["sub_audits"]["scope_audit"]["facts_used"],
            [
                "gray_case_evidence_complete",
                "is_common_part",
                "is_common_facility",
                "is_private_part",
                "is_property_service_scope",
            ],
        )

    def test_scope_unknown_object_boundary_does_not_apply_to_normal_repairable_object(self):
        data = self._audit(
            {
                "project_name": "外墙渗漏维修",
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["sub_audits"]["scope_audit"]["reason_codes"], [])

    def test_document_completeness_can_require_all_ticket_chain_items_in_fixed_order(self):
        data = self._audit(
            {
                "project_name": "电梯曳引机维修",
                "has_invoice": False,
                "has_settlement_report": False,
                "has_payment_proof": False,
            }
        )
        expected_reason_codes = [
            "MISSING_INVOICE",
            "MISSING_SETTLEMENT_REPORT",
            "MISSING_PAYMENT_PROOF",
        ]
        expected_missing_items = [
            "has_invoice",
            "has_settlement_report",
            "has_payment_proof",
        ]
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], expected_reason_codes)
        self.assertEqual(data["missing_items"], expected_missing_items)
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["reason_codes"],
            expected_reason_codes,
        )
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["missing_items"],
            expected_missing_items,
        )
        self.assertEqual(data["sub_audits"]["process_audit"]["reason_codes"], [])

    def test_document_completeness_can_require_invoice_and_settlement_in_fixed_order(self):
        data = self._audit(
            {
                "project_name": "电梯曳引机维修",
                "has_invoice": False,
                "has_settlement_report": False,
            }
        )
        self.assertEqual(
            data["reason_codes"],
            ["MISSING_INVOICE", "MISSING_SETTLEMENT_REPORT"],
        )
        self.assertEqual(
            data["missing_items"],
            ["has_invoice", "has_settlement_report"],
        )

    def test_document_completeness_keeps_emergency_rule_priority_over_missing_site_photos(self):
        data = self._audit(
            {
                "project_name": "排水管爆裂维修",
                "is_emergency": True,
                "has_site_photos": False,
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertEqual(data["reason_codes"], ["MISSING_EMERGENCY_DOC"])
        self.assertNotIn("MISSING_SITE_PHOTOS", data["reason_codes"])
        self.assertEqual(
            data["sub_audits"]["document_completeness_audit"]["reason_codes"],
            ["MISSING_EMERGENCY_DOC"],
        )

    def test_structured_input_groups_are_merged_into_rule_context(self):
        data = self._audit(
            {
                "project_name": "3号楼电梯曳引机维修",
                "process_facts": {
                    "has_vote": False,
                },
                "document_facts": {
                    "has_invoice": True,
                },
                "timeline_facts": {
                    "application_date": "2026-04-01",
                },
            }
        )
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertIn("MISSING_VOTE", data["reason_codes"])
        self.assertEqual(data["sub_audits"]["process_audit"]["reason_codes"], ["MISSING_VOTE"])
        self.assertIn("has_invoice", data["sub_audits"]["document_completeness_audit"]["facts_used"])
        self.assertIn("application_date", data["sub_audits"]["timeline_audit"]["facts_used"])

    def test_multi_project_cross_domain_adds_reason_code(self):
        data = self._audit(
            {"project_name": "防盗门油漆，人行道闸、绿化补种、电子门禁"},
        )
        self.assertEqual(data["overall_result"], "manual_review")
        self.assertIn("MULTI_PROJECT", data["reason_codes"])
        self.assertIn("CROSS_DOMAIN_PROJECT", data["reason_codes"])

    def test_high_freq_fire_extinguisher_refill_direct_reject(self):
        data = self._audit_pipeline({"project_name": "上海锦绣逸庭园区灭火器充装二氧化碳"})
        self.assertEqual(data["overall_result"], "non_compliant")
        self.assertIn("high_freq_mapping", data["audit_path"])
        self.assertIn("direct_reject", data["audit_path"])

    def test_high_freq_elevator_brake_testing_direct_reject(self):
        data = self._audit_pipeline({"project_name": "电梯125%制动试验"})
        self.assertEqual(data["overall_result"], "non_compliant")
        self.assertIn("direct_reject", data["audit_path"])

    def test_high_freq_trash_bin_replacement_direct_reject(self):
        data = self._audit_pipeline({"project_name": "垃圾桶更换"})
        self.assertEqual(data["overall_result"], "non_compliant")
        self.assertIn("direct_reject", data["audit_path"])

    def test_high_freq_tree_pruning_direct_reject(self):
        data = self._audit_pipeline({"project_name": "小区树木修剪"})
        self.assertEqual(data["overall_result"], "non_compliant")
        self.assertIn("direct_reject", data["audit_path"])

    def test_high_freq_camera_new_install_direct_reject(self):
        data = self._audit_pipeline({"project_name": "新增摄像头安装工程"})
        self.assertEqual(data["overall_result"], "non_compliant")
        self.assertIn("direct_reject", data["audit_path"])

    def test_high_freq_route_to_full_audit_for_elevator_mainframe_repair(self):
        data = self._audit_pipeline({"project_name": "3号楼电梯主机维修"})
        self.assertIn("route_to_full_audit", data["audit_path"])
        self.assertIn("mapping", data["audit_path"])
        self.assertNotIn("direct_reject", data["audit_path"])
        self.assertNotEqual(data["overall_result"], "compliant")

    def test_high_freq_route_to_full_audit_for_exterior_wall_repair(self):
        data = self._audit_pipeline({"project_name": "12号楼外墙渗漏维修"})
        self.assertIn("route_to_full_audit", data["audit_path"])
        self.assertIn("mapping", data["audit_path"])
        self.assertNotIn("direct_reject", data["audit_path"])
        self.assertNotEqual(data["overall_result"], "compliant")

    def test_high_freq_window_glass_goes_to_need_supplement_route(self):
        data = self._audit_pipeline({"project_name": "2号楼楼道窗户玻璃维修"})
        self.assertEqual(data["overall_result"], "need_supplement")
        self.assertIn("route_to_need_supplement", data["audit_path"])

    def test_high_freq_roof_repair_routes_to_full_audit(self):
        data = self._audit_pipeline({"project_name": "屋面防水维修"})
        self.assertIn("route_to_full_audit", data["audit_path"])
        self.assertIn("mapping", data["audit_path"])

    def test_high_freq_fire_system_repair_routes_to_full_audit(self):
        data = self._audit_pipeline({"project_name": "消防喷淋维修"})
        self.assertIn("route_to_full_audit", data["audit_path"])
        self.assertIn("mapping", data["audit_path"])

    def test_high_freq_greening_renovation_routes_to_manual_review(self):
        data = self._audit_pipeline({"project_name": "公共景观绿化整体翻新"})
        self.assertEqual(data["overall_result"], "manual_review")
        self.assertIn("route_to_manual_review", data["audit_path"])

    def test_high_freq_weak_current_optimization_routes_to_manual_review(self):
        data = self._audit_pipeline({"project_name": "门禁线路迁改优化升级"})
        self.assertEqual(data["overall_result"], "manual_review")
        self.assertIn("route_to_manual_review", data["audit_path"])

    def test_mapping_prefers_elevator_domain_for_elevator_mainframe_repair(self):
        mapping = map_project_name("3号楼电梯主机维修")
        self.assertGreater(len(mapping["mapped_objects"]), 0)
        self.assertTrue(mapping["mapped_objects"][0]["full_path"].startswith("电梯/"))
        self.assertNotIn("暖通系统", mapping["mapped_objects"][0]["full_path"])

        data = self._audit_pipeline({"project_name": "3号楼电梯主机维修"})
        self.assertNotIn("CROSS_DOMAIN_PROJECT", data["reason_codes"])

    def test_mapping_keeps_hvac_mainframe_for_split_ac_mainframe_repair(self):
        mapping = map_project_name("分体式空调主机维修")
        self.assertGreater(len(mapping["mapped_objects"]), 0)
        self.assertTrue(mapping["mapped_objects"][0]["full_path"].startswith("暖通系统/分体式空调/主机"))

    def test_mapping_keeps_cross_domain_for_real_multi_project_input(self):
        data = self._audit_pipeline({"project_name": "电梯主机维修，空调主机维修"})
        self.assertIn("MULTI_PROJECT", data["reason_codes"])


if __name__ == "__main__":
    unittest.main()
