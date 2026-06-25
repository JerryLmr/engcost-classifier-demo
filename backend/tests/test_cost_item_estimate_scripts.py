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

    def test_build_index_adds_three_embedding_text_columns(self):
        samples = self.sample_frame()
        samples.loc[1, "工程名称"] = ""
        samples.loc[1, "consultation_project_name"] = "外墙咨询"
        samples.loc[1, "renovation_content"] = "渗漏维修"

        indexed = build_index.add_embedding_text_columns(samples)

        self.assertEqual(indexed.loc[0, "item_text"], "屋面卷材防水 3.0mm SBS 沥青防水卷材")
        self.assertEqual(indexed.loc[0, "project_text"], "屋面漏水维修工程")
        self.assertEqual(indexed.loc[0, "full_text"], "屋面漏水维修工程 屋面卷材防水 3.0mm SBS 沥青防水卷材 单位：m²")
        self.assertEqual(indexed.loc[1, "project_text"], "外墙咨询 渗漏维修")

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
        self.assertEqual(row["cost_item_name"], "屋面卷材防水")
        self.assertEqual(row["source_project_count"], 2)
        self.assertEqual(row["occurrence_count"], 2)
        self.assertEqual(row["support_ratio"], 1.0)
        self.assertEqual(row["unit_price_count"], 2)
        self.assertEqual(row["unit_price_median"], 100.0)
        self.assertEqual(row["labor_unit_price_count"], 2)
        self.assertEqual(row["machinery_unit_price_count"], 2)
        self.assertIn("2", str(row["example_source_row_ids"]))
        self.assertIn("5", str(row["example_source_row_ids"]))

    def test_debug_item_matches_has_no_prefix_catalog_bonus(self):
        samples = self.sample_frame()
        item_embeddings = np.array([[0.8, 0.0], [0.99, 0.0], [0.0, 1.0]], dtype=np.float32)
        project_embeddings = np.array([[0.8, 0.0], [0.99, 0.0], [0.0, 1.0]], dtype=np.float32)
        full_embeddings = np.array([[0.8, 0.0], [0.99, 0.0], [0.0, 1.0]], dtype=np.float32)

        matches = query_estimate_llm.build_debug_item_matches(
            samples=samples,
            item_embeddings=item_embeddings,
            project_embeddings=project_embeddings,
            full_embeddings=full_embeddings,
            item_query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            project_query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            full_query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            predicted_catalog_id="CP-002-03",
            top_k=2,
            min_score=0.6,
        )

        self.assertEqual(matches["cost_item_name"].tolist(), ["外墙防水", "屋面卷材防水"])
        self.assertEqual(matches["catalog_match"].tolist(), [False, True])
        self.assertEqual(matches["catalog_mismatch"].tolist(), [True, False])
        self.assertAlmostEqual(float(matches.loc[matches["cost_item_name"] == "外墙防水", "final_score"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(matches.loc[matches["cost_item_name"] == "屋面卷材防水", "final_score"].iloc[0]), 0.87)

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

    def test_build_answer_reports_llm_success_error_and_empty_answer_sources(self):
        summary = {"提示": ""}
        recommended = pd.DataFrame(columns=query_estimate_llm.RECOMMENDED_ITEM_COLUMNS)

        with patch.object(query_estimate_llm, "request_llm_json", return_value={"answer": "自然语言总结"}):
            result = query_estimate_llm.build_answer("屋面漏水", summary, recommended)
        self.assertEqual(result, {"answer": "自然语言总结", "answer_source": "llm", "answer_error": ""})

        with patch.object(query_estimate_llm, "request_llm_json", return_value={"answer": ""}):
            result = query_estimate_llm.build_answer("屋面漏水", summary, recommended)
        self.assertEqual(result["answer_source"], "fallback")
        self.assertEqual(result["answer_error"], "LLM 返回空 answer")
        self.assertIn("样本不足", result["answer"])

        with patch.object(query_estimate_llm, "request_llm_json", side_effect=RuntimeError("service down")):
            result = query_estimate_llm.build_answer("屋面漏水", summary, recommended)
        self.assertEqual(result["answer_source"], "fallback")
        self.assertEqual(result["answer_error"], "service down")
        self.assertIn("样本不足", result["answer"])

    def test_fallback_answer_orders_primary_items_before_auxiliary_items_only_for_answer(self):
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
        self.assertLess(answer.index("1. 屋面卷材防水"), answer.index("2. 垂直运输"))

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
            "总结来源": "fallback",
            "提示": "",
        }
        answer_result = {"answer": "自动总结", "answer_source": "fallback", "answer_error": "service down"}
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
                    "example_source_row_ids": "2",
                    "example_item_row_ids": "2-1",
                }
            ],
            columns=query_estimate_llm.RECOMMENDED_ITEM_COLUMNS,
        )
        debug = pd.DataFrame([{column: "" for column in query_estimate_llm.DEBUG_ITEM_MATCH_COLUMNS}])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "query_result.xlsx"
            query_estimate_llm.write_query_result_workbook(output_path, answer_result, summary, recommended, debug)
            workbook = openpyxl.load_workbook(output_path, data_only=True)
            self.assertEqual(workbook.sheetnames, ["answer", "summary", "recommended_items", "debug_item_matches"])
            self.assertEqual(workbook["answer"].cell(row=2, column=1).value, "总结来源")
            self.assertEqual(workbook["answer"].cell(row=2, column=2).value, "fallback")
            self.assertEqual(workbook["answer"].cell(row=3, column=1).value, "错误信息")
            self.assertEqual(workbook["answer"].cell(row=3, column=2).value, "service down")
            self.assertEqual(workbook["answer"].cell(row=4, column=1).value, "总结")
            self.assertEqual(workbook["answer"].cell(row=4, column=2).value, "自动总结")
            summary_headers = [workbook["summary"].cell(row=1, column=column).value for column in range(1, 12)]
            self.assertIn("总结来源", summary_headers)
            self.assertEqual(workbook["summary"].freeze_panes, "A2")
            self.assertTrue(workbook["summary"]["A1"].font.bold)
            self.assertTrue(workbook["summary"]["A1"].alignment.wrap_text)
            self.assertEqual(workbook["recommended_items"]["G2"].number_format, "0.0000")
            self.assertEqual(workbook["recommended_items"]["J2"].number_format, "0.00")
            workbook.close()


if __name__ == "__main__":
    unittest.main()
