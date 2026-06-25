from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import openpyxl

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None


ROOT = Path(__file__).resolve().parents[2]


def load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_samples_script = load_script_module("build_cost_item_samples", "scripts/build_cost_item_samples.py")

if np is not None and pd is not None:
    build_index = load_script_module("build_cost_item_embedding_index", "scripts/build_cost_item_embedding_index.py")
    query_estimate = load_script_module("query_cost_item_estimate", "scripts/query_cost_item_estimate.py")
    query_estimate_llm = load_script_module("query_cost_estimate_llm", "scripts/query_cost_estimate_llm.py")
else:
    build_index = None
    query_estimate = None
    query_estimate_llm = None


@unittest.skipIf(np is None or pd is None, "cost item estimate dependencies are not installed")
class CostItemEstimateScriptTestCase(unittest.TestCase):
    def sample_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source_row_id": 2,
                    "item_row_id": "2-1",
                    "工程名称": "屋面漏水维修工程",
                    "sub_project_id": "SP1",
                    "catalog_id": "CP-002-03",
                    "一级分类": "屋面",
                    "二级分类": "防水层",
                    "维修状态": "维修",
                    "标准对象": "共用部位",
                    "是否复合工程": "否",
                    "复合目录": "",
                    "seq": 1,
                    "cost_item_name": "屋面卷材防水",
                    "project_description": "3.0mm SBS 沥青防水卷材",
                    "unit_normalized": "m²",
                    "quantity": 100,
                    "unit_price": 80,
                    "labor_unit_price": 20,
                    "machinery_unit_price": 5,
                    "item_similarity_text": "屋面卷材防水；3.0mm SBS 沥青防水卷材",
                    "item_context_text": "屋面漏水维修工程 / 屋面卷材防水",
                },
                {
                    "source_row_id": 3,
                    "item_row_id": "3-1",
                    "工程名称": "外墙维修工程",
                    "sub_project_id": "SP2",
                    "catalog_id": "CP-003-01",
                    "一级分类": "外墙面",
                    "二级分类": "面层",
                    "维修状态": "维修",
                    "标准对象": "共用部位",
                    "是否复合工程": "是",
                    "复合目录": "CP-002-03 | 屋面 | 防水层",
                    "seq": 1,
                    "cost_item_name": "外墙防水",
                    "project_description": "外墙渗漏处理",
                    "unit_normalized": "m²",
                    "quantity": 200,
                    "unit_price": 100,
                    "labor_unit_price": 30,
                    "machinery_unit_price": 8,
                    "item_similarity_text": "外墙防水；外墙渗漏处理",
                    "item_context_text": "外墙维修工程 / 外墙防水",
                },
                {
                    "source_row_id": 4,
                    "item_row_id": "4-1",
                    "工程名称": "管道维修工程",
                    "sub_project_id": "SP3",
                    "catalog_id": "CF-015-04",
                    "一级分类": "给排水系统",
                    "二级分类": "管道",
                    "维修状态": "维修",
                    "标准对象": "共用设施设备",
                    "是否复合工程": "否",
                    "复合目录": "",
                    "seq": 1,
                    "cost_item_name": "管道更换",
                    "project_description": "给水管道更换",
                    "unit_normalized": "m",
                    "quantity": 50,
                    "unit_price": None,
                    "labor_unit_price": 12,
                    "machinery_unit_price": None,
                    "item_similarity_text": "管道更换；给水管道更换",
                    "item_context_text": "管道维修工程 / 管道更换",
                },
            ]
        )

    def test_normalize_embeddings_handles_zero_vector(self):
        embeddings = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        normalized = build_index.normalize_embeddings(embeddings)
        self.assertAlmostEqual(float(np.linalg.norm(normalized[0])), 1.0)
        self.assertEqual(normalized[1].tolist(), [0.0, 0.0])

    def test_shared_normalize_unit_handles_square_and_cubic_units(self):
        self.assertEqual(build_samples_script.normalize_unit("m^2"), "m²")
        self.assertEqual(build_samples_script.normalize_unit("平方米"), "m²")
        self.assertEqual(build_samples_script.normalize_unit("平"), "m²")
        self.assertEqual(build_samples_script.normalize_unit("m^{3}"), "m³")
        self.assertEqual(build_samples_script.normalize_unit(" 台 "), "台")

    def test_build_index_keeps_samples_without_item_embedding_text_columns(self):
        samples = self.sample_frame()
        samples.loc[1, "工程名称"] = ""
        samples.loc[1, "consultation_project_name"] = "外墙咨询"
        samples.loc[1, "renovation_content"] = "渗漏维修"

        indexed = build_index.add_embedding_text_columns(samples)

        self.assertEqual(indexed.columns.tolist(), samples.columns.tolist())
        self.assertNotIn("item_text", indexed.columns)
        self.assertNotIn("project_text", indexed.columns)
        self.assertNotIn("full_text", indexed.columns)

    def test_build_index_meta_only_tracks_project_group_embedding_files(self):
        meta = build_index.build_index_meta(Path("samples.xlsx"), "demo-model", 3, 2)

        self.assertEqual(
            meta["files"],
            {
                "samples": "samples.parquet",
                "project_groups": "project_groups.parquet",
                "project_group_embeddings": "project_group_embeddings.npy",
            },
        )
        self.assertNotIn("item_text", meta["field_descriptions"])
        self.assertNotIn("item_embeddings", meta["files"])

    def test_build_project_groups_aggregates_source_rows(self):
        samples = self.sample_frame()
        duplicate = samples.iloc[0].copy()
        duplicate["item_row_id"] = "2-2"
        duplicate["cost_item_name"] = "防水层拆除"
        duplicate["project_description"] = "拆除原屋面防水层"
        samples = pd.concat([samples, pd.DataFrame([duplicate])], ignore_index=True)

        project_groups = build_index.build_project_groups(samples)

        self.assertEqual(project_groups["source_row_id"].tolist(), [2, 3, 4])
        roof = project_groups[project_groups["source_row_id"] == 2].iloc[0]
        self.assertEqual(roof["工程名称"], "屋面漏水维修工程")
        self.assertEqual(roof["catalog_id"], "CP-002-03")
        self.assertEqual(roof["item_count"], 2)
        self.assertIn("屋面卷材防水 3.0mm SBS 沥青防水卷材", roof["group_text"])
        self.assertIn("防水层拆除 拆除原屋面防水层", roof["group_text"])

    def test_build_samples_validate_paths_requires_overwrite_for_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "classified.xlsx"
            output_path = tmp_path / "samples.xlsx"
            workbook = openpyxl.Workbook()
            workbook.save(input_path)
            output_path.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "输出已存在，请加 --overwrite"):
                build_samples_script.validate_paths(input_path, output_path, overwrite=False)
            build_samples_script.validate_paths(input_path, output_path, overwrite=True)

    def test_build_index_validate_output_dir_requires_overwrite_for_managed_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "index"
            output_dir.mkdir()
            (output_dir / "samples.parquet").write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "输出已存在，请加 --overwrite"):
                build_index.validate_output_dir(output_dir, overwrite=False)
            build_index.validate_output_dir(output_dir, overwrite=True)

    def test_query_validate_output_path_requires_overwrite_for_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "query.xlsx"
            output_path.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "输出已存在，请加 --overwrite"):
                query_estimate.validate_output_path(output_path, overwrite=False)
            query_estimate.validate_output_path(output_path, overwrite=True)

    def test_candidate_mask_relaxes_missing_unit_and_supplements_composite_catalog(self):
        samples = self.sample_frame()
        mask, notes = query_estimate.build_candidate_mask(
            samples,
            unit="kg",
            catalog_id="CP-002-03",
            min_candidates=2,
        )
        self.assertEqual(mask.tolist(), [True, True, False])
        self.assertIn("单位过滤无匹配，已放宽", notes)
        self.assertIn("复合目录已补充相似样本候选", notes)

    def test_score_and_price_summary_use_top_matches(self):
        samples = self.sample_frame()
        item_embeddings = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        context_embeddings = np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        matches = query_estimate.score_candidates(
            samples=samples,
            item_embeddings=item_embeddings,
            context_embeddings=context_embeddings,
            query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            context_embedding=np.array([1.0, 0.0], dtype=np.float32),
            candidate_mask=np.array([True, True, False]),
            top_k=2,
            item_weight=0.75,
            context_weight=0.25,
        )
        self.assertEqual(matches["cost_item_name"].tolist(), ["屋面卷材防水", "外墙防水"])

        summary = query_estimate.summarize_price_ranges(
            matches=matches,
            query="屋面卷材防水",
            context="屋面漏水维修工程",
            unit="m²",
            catalog_id="CP-002-03",
            quantity=10,
            top_k=2,
            filter_notes=[],
        )
        self.assertEqual(summary["unit_price_count"], 2)
        self.assertEqual(summary["unit_price_min"], 80.0)
        self.assertEqual(summary["unit_price_median"], 90.0)
        self.assertEqual(summary["unit_price_max"], 100.0)
        self.assertEqual(summary["estimated_total_median"], 900.0)
        self.assertIn("样本数不足，区间仅供参考", summary["filter_notes"])

    def test_write_query_result_workbook_has_expected_sheets(self):
        samples = self.sample_frame()
        matches = samples.iloc[:1].copy()
        matches.insert(0, "context_score", [0.9])
        matches.insert(0, "item_score", [0.95])
        matches.insert(0, "final_score", [0.94])
        matches.insert(0, "rank", [1])
        matches = matches[query_estimate.MATCH_COLUMNS]
        summary = query_estimate.summarize_price_ranges(
            matches=matches,
            query="屋面卷材防水",
            context="屋面漏水维修工程",
            unit="m²",
            catalog_id="CP-002-03",
            quantity=10,
            top_k=1,
            filter_notes=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "query_result.xlsx"
            query_estimate.write_query_result_workbook(output_path, summary, matches)
            workbook = openpyxl.load_workbook(output_path, data_only=True)
            self.assertEqual(workbook.sheetnames, ["summary", "matches"])
            summary_headers = [workbook["summary"].cell(row=1, column=column).value for column in range(1, 5)]
            match_headers = [workbook["matches"].cell(row=1, column=column).value for column in range(1, 5)]
            workbook.close()

        self.assertEqual(summary_headers, ["query", "context", "unit", "catalog_id"])
        self.assertEqual(match_headers, ["rank", "final_score", "item_score", "context_score"])

    def test_llm_query_load_embedding_model_forces_cpu(self):
        fake_module = types.ModuleType("sentence_transformers")
        calls = []

        class FakeSentenceTransformer:
            def __init__(self, model_name, device=None):
                calls.append((model_name, device))

        fake_module.SentenceTransformer = FakeSentenceTransformer

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            model = query_estimate_llm.load_embedding_model("demo-model")

        self.assertIsInstance(model, FakeSentenceTransformer)
        self.assertEqual(calls, [("demo-model", "cpu")])

    def test_select_project_groups_uses_exact_catalog_only(self):
        project_groups = pd.DataFrame(
            [
                {"source_row_id": 2, "catalog_id": "CP-002-03", "工程名称": "屋面漏水维修工程", "item_count": 3},
                {"source_row_id": 3, "catalog_id": "CP-003-01", "工程名称": "外墙维修工程", "item_count": 2},
                {"source_row_id": 4, "catalog_id": "CF-015-04", "工程名称": "管道维修工程", "item_count": 1},
            ]
        )
        embeddings = np.array([[0.7, 0.0], [1.0, 0.0], [0.9, 0.0]], dtype=np.float32)

        selected, exact_available, exact_count = query_estimate_llm.select_project_groups(
            project_groups,
            embeddings,
            raw_query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            predicted_catalog_id="CP-002-03",
            top_k=10,
        )
        self.assertTrue(exact_available)
        self.assertEqual(exact_count, 1)
        self.assertEqual(selected["source_row_id"].tolist(), [2])

        selected, exact_available, exact_count = query_estimate_llm.select_project_groups(
            project_groups,
            embeddings,
            raw_query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            predicted_catalog_id="CP-999-00",
            top_k=10,
        )
        self.assertFalse(exact_available)
        self.assertEqual(exact_count, 0)
        self.assertTrue(selected.empty)

    def test_recommended_items_aggregate_by_signature_and_price_fields(self):
        samples = self.sample_frame()
        duplicate = samples.iloc[0].copy()
        duplicate["source_row_id"] = 5
        duplicate["item_row_id"] = "5-1"
        duplicate["unit_price"] = 120
        duplicate["labor_unit_price"] = 30
        duplicate["machinery_unit_price"] = 7
        samples = pd.concat([samples, pd.DataFrame([duplicate])], ignore_index=True)
        matched_projects = pd.DataFrame(
            [
                {"source_row_id": 2, "project_score": 0.9},
                {"source_row_id": 5, "project_score": 0.8},
            ]
        )

        recommended = query_estimate_llm.aggregate_recommended_items(samples, matched_projects, top_k=10)

        self.assertEqual(len(recommended), 1)
        row = recommended.iloc[0]
        self.assertEqual(row["item_id"], "rec_001")
        self.assertEqual(row["cost_item_name"], "屋面卷材防水")
        self.assertEqual(row["source_project_count"], 2)
        self.assertEqual(row["occurrence_count"], 2)
        self.assertEqual(row["support_ratio"], 1.0)
        self.assertEqual(row["unit_price_count"], 2)
        self.assertEqual(row["unit_price_median"], 100.0)
        self.assertEqual(row["labor_unit_price_count"], 2)
        self.assertEqual(row["machinery_unit_price_count"], 2)
        self.assertEqual(row["unit_price_coverage"], 1.0)
        self.assertEqual(row["labor_unit_price_coverage"], 1.0)
        self.assertEqual(row["machinery_unit_price_coverage"], 1.0)
        self.assertEqual(row["price_breakdown_status"], "有人工和机械费用拆分")
        self.assertIn("2", str(row["example_source_row_ids"]))
        self.assertIn("5", str(row["example_source_row_ids"]))

    def test_extract_simple_quantity_reuses_unit_normalizer(self):
        cases = [
            ("面积大概500平", "平"),
            ("面积大概500 平方米", "平方米"),
            ("面积大概500m2", "m2"),
            ("面积大概500 ㎡", "㎡"),
            ("面积大概500 m²", "m²"),
        ]
        for text, raw_unit in cases:
            with self.subTest(text=text):
                quantity = query_estimate_llm.extract_simple_quantity(text)
                self.assertEqual(quantity["quantity"], 500.0)
                self.assertEqual(quantity["raw_unit"].replace(" ", ""), raw_unit.replace(" ", ""))
                self.assertEqual(quantity["unit"], "m²")
                self.assertEqual(quantity["source"], "regex")

        self.assertIsNone(query_estimate_llm.extract_simple_quantity("屋面漏水维修工程"))

        with patch.object(query_estimate_llm, "normalize_unit", return_value=""):
            quantity = query_estimate_llm.extract_simple_quantity("面积500平")
        self.assertEqual(quantity["source"], "regex_unrecognized_unit")
        self.assertEqual(quantity["unit"], "")

    def test_apply_simple_amount_estimates_handles_quantity_price_and_unit_cases(self):
        recommended = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "cost_item_name": "完整区间",
                    "unit_normalized": "m²",
                    "unit_price_p25": 30,
                    "unit_price_median": 36,
                    "unit_price_p75": 42,
                },
                {
                    "rank": 2,
                    "cost_item_name": "仅中位数",
                    "unit_normalized": "m²",
                    "unit_price_p25": None,
                    "unit_price_median": 20,
                    "unit_price_p75": None,
                },
                {
                    "rank": 3,
                    "cost_item_name": "缺少综合单价",
                    "unit_normalized": "m²",
                    "unit_price_p25": None,
                    "unit_price_median": None,
                    "unit_price_p75": None,
                },
                {
                    "rank": 4,
                    "cost_item_name": "单位不一致",
                    "unit_normalized": "项",
                    "unit_price_p25": 100,
                    "unit_price_median": 120,
                    "unit_price_p75": 140,
                },
            ]
        )
        for column in query_estimate_llm.RECOMMENDED_ITEM_COLUMNS:
            if column not in recommended.columns:
                recommended[column] = None
        recommended = recommended[query_estimate_llm.RECOMMENDED_ITEM_COLUMNS]

        no_quantity = query_estimate_llm.apply_simple_amount_estimates(recommended, None)
        self.assertEqual(no_quantity["estimated_amount_note"].iloc[0], "未识别工程量，暂不计算参考金额")

        estimated = query_estimate_llm.apply_simple_amount_estimates(
            recommended,
            {"quantity": 500.0, "raw_unit": "平", "unit": "m²", "source": "regex"},
        )
        self.assertEqual(estimated.loc[0, "estimated_amount_p25"], 15000)
        self.assertEqual(estimated.loc[0, "estimated_amount_median"], 18000)
        self.assertEqual(estimated.loc[0, "estimated_amount_p75"], 21000)
        self.assertEqual(estimated.loc[0, "estimated_amount_note"], "按输入工程量和综合单价历史区间简单估算")
        self.assertTrue(pd.isna(estimated.loc[1, "estimated_amount_p25"]))
        self.assertEqual(estimated.loc[1, "estimated_amount_median"], 10000)
        self.assertIn("缺少完整 P25-P75 区间", estimated.loc[1, "estimated_amount_note"])
        self.assertTrue(pd.isna(estimated.loc[2, "estimated_amount_median"]))
        self.assertEqual(estimated.loc[2, "estimated_amount_note"], "缺少综合单价样本，无法计算参考金额")
        self.assertTrue(pd.isna(estimated.loc[3, "estimated_amount_median"]))
        self.assertEqual(estimated.loc[3, "estimated_amount_note"], "单位不一致，未按输入工程量计算")
        self.assertEqual(query_estimate_llm.count_calculable_amount_items(estimated), 1)
        self.assertEqual(query_estimate_llm.simple_total_amounts(estimated), {"p25": 15000.0, "median": 18000.0, "p75": 21000.0})

    def test_llm_query_warnings_for_missing_exact_catalog_samples(self):
        warnings = query_estimate_llm.build_warnings(
            classify_warnings=["工程分类未能稳定匹配标准目录：fallback"],
            exact_catalog_available=False,
            exact_catalog_total_count=0,
            matched_projects=pd.DataFrame(),
            recommended_count=0,
        )

        self.assertIn("工程分类未能稳定匹配标准目录：fallback", warnings)
        self.assertIn(query_estimate_llm.NO_EXACT_CATALOG_WARNING, warnings)
        self.assertIn("未形成推荐清单项，请补充材料、做法、面积或设备规格。", warnings)

    def test_build_answer_uses_fallback_template_for_empty_items(self):
        summary = {"提示": ""}
        recommended = pd.DataFrame(columns=query_estimate_llm.RECOMMENDED_ITEM_COLUMNS)

        result = query_estimate_llm.build_answer("屋面漏水", summary, recommended)
        self.assertEqual(result["answer_source"], "fallback_template")
        self.assertEqual(result["answer_error"], "")
        self.assertEqual(result["answer_plan"]["sections"], [])
        self.assertIn("样本不足", result["answer"])

    def test_answer_plan_payload_hides_price_and_amount_numbers(self):
        recommended = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "item_id": "rec_001",
                    "cost_item_name": "屋面卷材防水",
                    "project_description": "3mm SBS",
                    "unit_normalized": "m²",
                    "source_project_count": 1,
                    "occurrence_count": 1,
                    "support_ratio": 1.0,
                    "unit_price_count": 1,
                    "unit_price_p25": 80,
                    "unit_price_median": 90,
                    "unit_price_p75": 100,
                    "estimated_amount_median": 45000,
                    "estimated_amount_note": "按输入工程量和综合单价历史区间简单估算",
                }
            ]
        )

        payload = query_estimate_llm.build_answer_plan_payload("屋面漏水", {"提示": ""}, recommended)
        item = payload["recommended_items"][0]

        self.assertEqual(item["item_id"], "rec_001")
        self.assertTrue(item["has_unit_price"])
        self.assertTrue(item["has_amount_estimate"])
        self.assertNotIn("unit_price_p25", item)
        self.assertNotIn("unit_price_median", item)
        self.assertNotIn("unit_price_p75", item)
        self.assertNotIn("estimated_amount_median", item)

    def test_validate_answer_plan_filters_invalid_duplicates_and_notes(self):
        recommended = pd.DataFrame(
            [
                {"rank": 1, "item_id": "rec_001", "cost_item_name": "主项"},
                {"rank": 2, "item_id": "rec_002", "cost_item_name": "同类项"},
            ]
        )
        plan = {
            "mode": "sectioned",
            "sections": [
                {"title": "主项", "item_ids": ["rec_001", "missing", "rec_001"]},
                {"title": "重复分组", "item_ids": ["rec_001", "rec_002"]},
            ],
            "similar_groups": [
                {
                    "title": "相近做法",
                    "item_ids": ["rec_002", "missing"],
                    "display_item_id": "missing",
                    "reason": "同类",
                }
            ],
            "conditional_item_ids": ["rec_002", "missing"],
            "excluded_item_ids": ["missing", "rec_001"],
            "notes": ["一", "二", "三", "四"],
        }

        validated = query_estimate_llm.validate_answer_plan(plan, recommended)

        self.assertEqual(validated["sections"][0]["item_ids"], ["rec_001"])
        self.assertEqual(validated["sections"][1]["item_ids"], ["rec_002"])
        self.assertEqual(validated["similar_groups"][0]["display_item_id"], "rec_002")
        self.assertEqual(validated["conditional_item_ids"], ["rec_002"])
        self.assertEqual(validated["excluded_item_ids"], ["rec_001"])
        self.assertEqual(validated["notes"], ["一", "二", "三"])

    def test_answer_plan_for_output_uses_row_types_and_plan_action(self):
        recommended = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "item_id": "rec_001",
                    "cost_item_name": "屋面卷材防水",
                    "project_description": "3mm SBS",
                    "unit_normalized": "m²",
                },
                {
                    "rank": 2,
                    "item_id": "rec_002",
                    "cost_item_name": "屋面卷材防水加强层",
                    "project_description": "SBS 附加层",
                    "unit_normalized": "m²",
                },
                {
                    "rank": 3,
                    "item_id": "rec_003",
                    "cost_item_name": "拆除原防水层",
                    "project_description": "旧防水拆除",
                    "unit_normalized": "m²",
                },
            ]
        )
        answer_plan = {
            "planner_source": "planner_template",
            "planner_error": "",
            "sections": [{"title": "核心维修工程（SBS防水）", "item_ids": ["rec_001", "rec_002"]}],
            "similar_groups": [
                {
                    "title": "SBS 卷材做法",
                    "item_ids": ["rec_001", "rec_002"],
                    "display_item_id": "rec_001",
                    "reason": "同类卷材做法",
                }
            ],
            "conditional_item_ids": ["rec_001"],
            "excluded_item_ids": ["rec_003"],
            "notes": ["需确认基层状态"],
        }

        output = query_estimate_llm.answer_plan_for_output(answer_plan, recommended)

        self.assertEqual(output.columns.tolist(), query_estimate_llm.ANSWER_PLAN_COLUMNS)
        self.assertEqual(output["row_type"].tolist(), ["shown_item", "hidden_similar_item", "excluded_item", "note"])
        self.assertEqual(output.loc[0, "section_title"], "核心维修工程（SBS防水）")
        self.assertEqual(output.loc[0, "plan_action"], "展示，代表相近做法；需现场确认")
        self.assertEqual(output.loc[0, "representative_item_id"], "rec_001")
        self.assertEqual(output.loc[1, "section_title"], "核心维修工程（SBS防水）")
        self.assertEqual(output.loc[1, "plan_action"], "作为相近做法隐藏")
        self.assertEqual(output.loc[1, "representative_item_id"], "rec_001")
        self.assertEqual(output.loc[1, "similar_group_title"], "SBS 卷材做法")
        self.assertEqual(output.loc[1, "reason"], "同类卷材做法")
        self.assertEqual(output.loc[2, "section_title"], "")
        self.assertEqual(output.loc[2, "plan_action"], "排除")
        self.assertEqual(output.loc[3, "plan_action"], "全局提示")
        self.assertEqual(output.loc[3, "note"], "需确认基层状态")
        self.assertNotIn("planner_source", output.columns)
        self.assertNotIn("planner_error", output.columns)

    def test_build_answer_uses_planner_template_and_hides_similar_detail_items(self):
        recommended = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "item_id": "rec_001",
                    "cost_item_name": "屋面卷材防水",
                    "project_description": "3mm SBS",
                    "unit_normalized": "m²",
                    "support_ratio": 1.0,
                    "unit_price_count": 1,
                    "unit_price_p25": 80,
                    "unit_price_median": 90,
                    "unit_price_p75": 100,
                    "labor_unit_price_count": 0,
                    "machinery_unit_price_count": 0,
                    "input_quantity": 500,
                    "input_quantity_unit": "m²",
                    "estimated_amount_p25": 40000,
                    "estimated_amount_median": 45000,
                    "estimated_amount_p75": 50000,
                },
                {
                    "rank": 2,
                    "item_id": "rec_002",
                    "cost_item_name": "屋面卷材防水加强层",
                    "project_description": "SBS 附加层",
                    "unit_normalized": "m²",
                    "support_ratio": 0.5,
                    "unit_price_count": 1,
                    "unit_price_median": 30,
                    "labor_unit_price_count": 0,
                    "machinery_unit_price_count": 0,
                },
                {
                    "rank": 3,
                    "item_id": "rec_003",
                    "cost_item_name": "基层处理",
                    "project_description": "基层清理",
                    "unit_normalized": "m²",
                    "support_ratio": 0.4,
                    "unit_price_count": 0,
                    "labor_unit_price_count": 0,
                    "machinery_unit_price_count": 0,
                },
            ]
        )
        planner_plan = {
            "mode": "sectioned",
            "sections": [{"title": "防水做法", "item_ids": ["rec_001", "rec_002", "rec_003"]}],
            "similar_groups": [
                {
                    "title": "SBS 卷材做法",
                    "item_ids": ["rec_001", "rec_002"],
                    "display_item_id": "rec_001",
                    "reason": "同类卷材做法",
                }
            ],
            "conditional_item_ids": ["rec_003"],
            "excluded_item_ids": [],
            "notes": ["需确认基层状态"],
        }

        with patch.object(query_estimate_llm, "request_llm_json", return_value=planner_plan):
            result = query_estimate_llm.build_answer("屋面漏水", {"提示": "", "工程量单位": "m²"}, recommended)

        answer = result["answer"]
        self.assertEqual(result["answer_source"], "planner_template")
        self.assertEqual(result["answer_error"], "")
        self.assertIn("【防水做法】", answer)
        self.assertIn("1. 屋面卷材防水", answer)
        self.assertIn("历史样本中另有相近做法/同类项：屋面卷材防水加强层，详见 recommended_items。", answer)
        self.assertNotIn("2. 屋面卷材防水加强层", answer)
        self.assertIn("基层处理（需现场确认）", answer)
        self.assertIn("当前不自动汇总为总价", answer)

    def test_build_answer_falls_back_when_planner_fails(self):
        recommended = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "item_id": "rec_001",
                    "cost_item_name": "消防主机",
                    "project_description": "主机更换",
                    "unit_normalized": "台",
                    "support_ratio": 1.0,
                    "unit_price_count": 0,
                    "labor_unit_price_count": 0,
                    "machinery_unit_price_count": 0,
                }
            ]
        )

        with patch.object(query_estimate_llm, "request_llm_json", side_effect=RuntimeError("service down")):
            result = query_estimate_llm.build_answer("消防主机更换", {"提示": ""}, recommended)

        self.assertEqual(result["answer_source"], "fallback_template")
        self.assertIn("service down", result["answer_error"])
        self.assertEqual(result["answer_plan"]["sections"][0]["title"], "历史样本候选清单项")
        self.assertIn("消防主机", result["answer"])

    def test_answer_price_text_formatters_do_not_infer_missing_labor_or_machinery(self):
        base = {
            "cost_item_name": "屋面卷材防水",
            "project_description": "3.0mm SBS",
            "unit_normalized": "m²",
            "support_ratio": 1.0,
            "unit_price_count": 1,
            "unit_price_p25": 30,
            "unit_price_median": 36.77,
            "unit_price_p75": 42,
            "labor_unit_price_count": 0,
            "machinery_unit_price_count": 0,
        }

        row = pd.Series({**base, "labor_unit_price_count": 1, "labor_unit_price_median": 2.63})
        self.assertEqual(
            query_estimate_llm.format_unit_price_text(row),
            "参考综合单价：约 36.77 元/m²，历史样本区间约 30-42 元/m²",
        )
        self.assertEqual(
            query_estimate_llm.format_price_breakdown_text(row),
            "价格拆分：其中包含人工约 2.63 元/m²，历史样本未见机械费用拆分",
        )

        row = pd.Series({**base, "machinery_unit_price_count": 1, "machinery_unit_price_median": 0.25})
        self.assertEqual(
            query_estimate_llm.format_price_breakdown_text(row),
            "价格拆分：其中包含机械约 0.25 元/m²，历史样本未见人工费用拆分",
        )

        row = pd.Series(
            {
                **base,
                "labor_unit_price_count": 1,
                "labor_unit_price_median": 2.63,
                "machinery_unit_price_count": 1,
                "machinery_unit_price_median": 0.25,
            }
        )
        self.assertEqual(
            query_estimate_llm.format_price_breakdown_text(row),
            "价格拆分：其中包含人工约 2.63 元/m²、机械约 0.25 元/m²",
        )

        row = pd.Series(base)
        self.assertEqual(
            query_estimate_llm.format_price_breakdown_text(row),
            "价格拆分：历史样本未见人工、机械费用拆分",
        )

        row = pd.Series({**base, "unit_price_count": 0})
        self.assertEqual(
            query_estimate_llm.format_unit_price_text(row),
            "参考综合单价：暂无可靠综合单价参考",
        )
        self.assertEqual(
            query_estimate_llm.format_price_breakdown_text(row),
            "价格拆分：缺少可用综合单价，暂不展开费用拆分",
        )

    def test_template_answer_uses_preformatted_customer_fields_and_amount_ranges(self):
        recommended = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "cost_item_name": "屋面卷材防水",
                    "project_description": "3mm SBS",
                    "unit_normalized": "m²",
                    "source_project_count": 1,
                    "occurrence_count": 1,
                    "support_ratio": 1.0,
                    "unit_price_count": 1,
                    "unit_price_p25": 80,
                    "unit_price_median": 90,
                    "unit_price_p75": 100,
                    "labor_unit_price_count": 0,
                    "machinery_unit_price_count": 0,
                    "input_quantity": 500,
                    "input_quantity_unit": "m²",
                    "estimated_amount_p25": 40000,
                    "estimated_amount_median": 45000,
                    "estimated_amount_p75": 50000,
                    "estimated_amount_note": "按输入工程量和综合单价历史区间简单估算",
                },
            ]
        )

        answer = query_estimate_llm.build_answer_fallback(
            "屋面漏水",
            {"提示": "", "工程量单位": "m²"},
            recommended,
        )

        self.assertIn("历史样本类似工程常见程度：高", answer)
        self.assertIn("参考综合单价：约 90 元/m²，历史样本区间约 80-100 元/m²", answer)
        self.assertIn("价格拆分：历史样本未见人工、机械费用拆分", answer)
        self.assertIn("施工工艺/项目特征：3mm SBS", answer)
        self.assertIn("简单估算：按 500 m² 计算，参考金额约 40,000-50,000 元", answer)
        self.assertNotIn("按已能匹配单位的清单项简单合计", answer)
        self.assertIn("当前不自动汇总为总价", answer)
        self.assertNotIn("预计总价", answer)
        self.assertNotIn("报价", answer)

    def test_simple_estimate_text_uses_concentrated_price_when_range_rounds_equal(self):
        row = pd.Series(
            {
                "input_quantity": 500,
                "input_quantity_unit": "m²",
                "estimated_amount_p25": 18385,
                "estimated_amount_median": 18385,
                "estimated_amount_p75": 18385,
            }
        )

        text = query_estimate_llm.format_simple_estimate_text(row)

        self.assertEqual(text, "简单估算：按 500 m² 计算，历史样本价格集中，参考金额约 18,385 元")
        self.assertNotIn("18,385-18,385 元", text)

    def test_fallback_answer_keeps_recommended_order_without_keyword_roles(self):
        recommended = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "cost_item_name": "垂直运输",
                    "project_description": "材料垂直运输",
                    "unit_normalized": "项",
                    "source_project_count": 1,
                    "occurrence_count": 1,
                    "support_ratio": 1.0,
                    "unit_price_count": 0,
                    "labor_unit_price_count": 0,
                    "machinery_unit_price_count": 0,
                },
                {
                    "rank": 2,
                    "cost_item_name": "屋面卷材防水",
                    "project_description": "3mm SBS",
                    "unit_normalized": "m²",
                    "source_project_count": 1,
                    "occurrence_count": 1,
                    "support_ratio": 1.0,
                    "unit_price_count": 1,
                    "unit_price_p25": 80,
                    "unit_price_median": 80,
                    "unit_price_p75": 80,
                    "labor_unit_price_count": 0,
                    "machinery_unit_price_count": 0,
                },
            ]
        )

        answer = query_estimate_llm.build_answer_fallback("屋面漏水", {"提示": ""}, recommended)

        self.assertEqual(recommended["cost_item_name"].tolist(), ["垂直运输", "屋面卷材防水"])
        self.assertLess(answer.index("1. 垂直运输"), answer.index("2. 屋面卷材防水"))
        self.assertNotIn("垂直运输（措施/辅助项）", answer)
        self.assertIn("历史样本类似工程常见程度：高", answer)
        self.assertIn("参考综合单价：80 元/m²", answer)
        self.assertIn("价格拆分：历史样本未见人工、机械费用拆分", answer)
        self.assertIn("施工工艺/项目特征：3mm SBS", answer)
        for forbidden in [
            "不需要人工费",
            "不需要机械费",
            "未单列人工单价",
            "未单列机械单价",
            "人工/机械单价暂无可靠价格样本",
            "已含",
        ]:
            self.assertNotIn(forbidden, answer)

    def test_write_llm_query_result_workbook_has_expected_sheets_and_style(self):
        summary = {
            "原始输入": "屋面漏水",
            "标准目录ID": "CP-002-03",
            "一级分类": "屋面",
            "二级分类": "防水层",
            "维修状态": "维修",
            "标准对象": "共用部位",
            "最高工程相似度": 0.9,
            "相似工程数": 1,
            "推荐清单项数": 1,
            "识别工程量": 500,
            "工程量单位": "m²",
            "可计算清单项数": 1,
            "简单合计参考金额": "未自动合计",
            "总结来源": "planner_template",
            "提示": "",
        }
        answer_result = {
            "answer": "自动总结",
            "answer_source": "planner_template",
            "answer_error": "",
            "answer_plan": {
                "planner_source": "planner_template",
                "planner_error": "",
                "sections": [{"title": "防水做法", "item_ids": ["rec_001"]}],
                "similar_groups": [],
                "conditional_item_ids": [],
                "excluded_item_ids": [],
                "notes": ["单项参考"],
            },
        }
        matched_projects = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "project_score": 0.9,
                    "source_row_id": 2,
                    "工程名称": "屋面漏水维修工程",
                    "catalog_id": "CP-002-03",
                    "一级分类": "屋面",
                    "二级分类": "防水层",
                    "维修状态": "维修",
                    "标准对象": "共用部位",
                    "item_count": 1,
                }
            ]
        )
        recommended = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "item_id": "rec_001",
                    "cost_item_name": "屋面卷材防水",
                    "project_description": "3mm SBS",
                    "unit_normalized": "m²",
                    "source_project_count": 1,
                    "occurrence_count": 1,
                    "support_ratio": 1.0,
                    "unit_price_count": 1,
                    "unit_price_p25": 80,
                    "unit_price_median": 80,
                    "unit_price_p75": 80,
                    "labor_unit_price_count": 1,
                    "labor_unit_price_p25": 20,
                    "labor_unit_price_median": 20,
                    "labor_unit_price_p75": 20,
                    "machinery_unit_price_count": 1,
                    "machinery_unit_price_p25": 5,
                    "machinery_unit_price_median": 5,
                    "machinery_unit_price_p75": 5,
                    "input_quantity": 500,
                    "input_quantity_unit": "m²",
                    "estimated_amount_p25": 40000,
                    "estimated_amount_median": 40000,
                    "estimated_amount_p75": 40000,
                    "estimated_amount_note": "按输入工程量和综合单价历史区间简单估算",
                    "example_source_row_ids": "2",
                    "example_item_row_ids": "2-1",
                }
            ],
            columns=query_estimate_llm.RECOMMENDED_ITEM_COLUMNS,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "query_result.xlsx"
            query_estimate_llm.write_query_result_workbook(output_path, answer_result, summary, matched_projects, recommended)
            workbook = openpyxl.load_workbook(output_path, data_only=True)
            self.assertEqual(workbook.sheetnames, ["answer", "summary", "answer_plan", "matched_projects", "recommended_items"])
            self.assertEqual(workbook["answer"].cell(row=2, column=1).value, "总结来源")
            self.assertEqual(workbook["answer"].cell(row=2, column=2).value, "planner_template")
            self.assertEqual(workbook["answer"].cell(row=3, column=1).value, "错误信息")
            self.assertIsNone(workbook["answer"].cell(row=3, column=2).value)
            self.assertEqual(workbook["answer"].cell(row=4, column=1).value, "总结")
            self.assertEqual(workbook["answer"].cell(row=4, column=2).value, "自动总结")
            summary_headers = [workbook["summary"].cell(row=1, column=column).value for column in range(1, 16)]
            self.assertIn("总结来源", summary_headers)
            self.assertIn("识别工程量", summary_headers)
            self.assertEqual(workbook["summary"].cell(row=2, column=13).value, "未自动合计")
            answer_plan_headers = [workbook["answer_plan"].cell(row=1, column=column).value for column in range(1, 13)]
            self.assertEqual(answer_plan_headers, query_estimate_llm.ANSWER_PLAN_COLUMNS)
            self.assertNotIn("planner_source", answer_plan_headers)
            self.assertNotIn("planner_error", answer_plan_headers)
            self.assertEqual(workbook["answer_plan"].cell(row=2, column=1).value, "shown_item")
            self.assertEqual(workbook["answer_plan"].cell(row=2, column=2).value, "防水做法")
            self.assertEqual(workbook["answer_plan"].cell(row=2, column=4).value, "rec_001")
            self.assertEqual(workbook["answer_plan"].cell(row=2, column=5).value, "屋面卷材防水")
            self.assertEqual(workbook["answer_plan"].cell(row=2, column=8).value, "展示")
            self.assertEqual(workbook["recommended_items"].cell(row=1, column=2).value, "item_id")
            self.assertEqual(workbook["recommended_items"].cell(row=2, column=2).value, "rec_001")
            self.assertEqual(workbook["matched_projects"].cell(row=1, column=1).value, "rank")
            self.assertEqual(workbook["summary"].freeze_panes, "A2")
            self.assertTrue(workbook["summary"]["A1"].font.bold)
            self.assertTrue(workbook["summary"]["A1"].alignment.wrap_text)
            self.assertEqual(workbook["recommended_items"]["H2"].number_format, "0.0000")
            self.assertEqual(workbook["recommended_items"]["J2"].number_format, "0.00")
            self.assertEqual(workbook["recommended_items"]["U2"].number_format, "0.0000")
            workbook.close()


if __name__ == "__main__":
    unittest.main()
