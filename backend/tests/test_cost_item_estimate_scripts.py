from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

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
            workbook = openpyxl.load_workbook(output_path, read_only=True, data_only=True)
            self.assertEqual(workbook.sheetnames, ["summary", "matches"])
            summary_headers = [workbook["summary"].cell(row=1, column=column).value for column in range(1, 5)]
            match_headers = [workbook["matches"].cell(row=1, column=column).value for column in range(1, 5)]
            workbook.close()

        self.assertEqual(summary_headers, ["query", "context", "unit", "catalog_id"])
        self.assertEqual(match_headers, ["rank", "final_score", "item_score", "context_score"])

    def test_llm_query_rerank_uses_catalog_and_unit_bonuses(self):
        samples = self.sample_frame()
        item_embeddings = np.array(
            [
                [1.0, 0.0],
                [0.96, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        project_embeddings = np.array(
            [
                [1.0, 0.0],
                [0.95, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        full_embeddings = np.array(
            [
                [1.0, 0.0],
                [0.94, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        matches = query_estimate_llm.score_and_rerank_candidates(
            samples=samples,
            item_embeddings=item_embeddings,
            project_embeddings=project_embeddings,
            full_embeddings=full_embeddings,
            item_query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            project_query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            full_query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            predicted_catalog_id="CP-002-03",
            unit="m²",
            top_k=2,
        )

        self.assertEqual(matches["cost_item_name"].tolist(), ["屋面卷材防水", "外墙防水"])
        self.assertEqual(matches["catalog_match"].tolist(), ["exact", "prefix"])
        self.assertEqual(matches["unit_match"].tolist(), [True, True])
        self.assertAlmostEqual(float(matches["final_score"].iloc[0]), 1.10, places=5)

    def test_llm_query_price_summary_groups_by_unit_and_totals_only_with_quantity(self):
        matches = pd.DataFrame(
            [
                {"final_score": 0.8, "unit_normalized": "m²", "unit_price": 80},
                {"final_score": 0.7, "unit_normalized": "m²", "unit_price": 100},
                {"final_score": 0.9, "unit_normalized": "台", "unit_price": 500},
                {"final_score": 0.4, "unit_normalized": "m²", "unit_price": 120},
                {"final_score": 0.95, "unit_normalized": "m", "unit_price": None},
            ]
        )

        price_pool = query_estimate_llm.price_sample_pool(matches, min_score=0.6)
        grouped = query_estimate_llm.summarize_unit_prices(price_pool, quantity=10)

        square = grouped[grouped["unit"] == "m²"].iloc[0]
        self.assertEqual(square["priced_count"], 2)
        self.assertEqual(square["median"], 90.0)
        self.assertEqual(square["estimated_total_median"], 900.0)
        self.assertEqual(query_estimate_llm.main_priced_count(price_pool, "m²"), 2)
        self.assertEqual(query_estimate_llm.main_priced_count(price_pool, ""), 3)

        grouped_without_quantity = query_estimate_llm.summarize_unit_prices(price_pool, quantity=None)
        self.assertIsNone(grouped_without_quantity[grouped_without_quantity["unit"] == "m²"].iloc[0]["estimated_total_median"])

    def test_llm_query_warnings_follow_profile_and_score_rules(self):
        warnings = query_estimate_llm.build_warnings(
            profile={"quantity": None, "unit": "", "confidence": "low"},
            classify_warnings=["工程分类未能稳定匹配标准目录：fallback"],
            priced_count=1,
            top_score=0.5,
            min_score=0.6,
        )

        self.assertIn("工程分类未能稳定匹配标准目录：fallback", warnings)
        self.assertIn("未识别工程量，只能给单价参考，不能估算总价。", warnings)
        self.assertIn("未识别计量单位，已按历史样本单位分组展示，不能混合不同单位直接估价。", warnings)
        self.assertIn("高相似可计价样本不足，建议补充材料、做法、面积或设备规格。", warnings)
        self.assertIn("相似度偏低，结果仅供线索参考。", warnings)
        self.assertIn("输入信息较模糊，系统解析置信度低。", warnings)

    def test_write_llm_query_result_workbook_has_expected_sheets(self):
        summary = {
            "raw_text": "屋面漏水",
            "project_text": "屋面漏水维修工程",
            "item_text": "屋面防水",
            "feature_text": "3mm SBS",
            "inferred_quantity": 10,
            "inferred_unit": "m²",
            "predicted_catalog_id": "CP-002-03",
            "predicted_category": "屋面",
            "predicted_item": "防水层",
            "min_score": 0.6,
            "top_score": 0.9,
            "priced_count": 1,
            "warnings": "",
        }
        unit_price_by_unit = pd.DataFrame(
            [
                {
                    "unit": "m²",
                    "sample_count": 1,
                    "priced_count": 1,
                    "min": 80,
                    "p25": 80,
                    "median": 80,
                    "p75": 80,
                    "max": 80,
                    "estimated_total_p25": 800,
                    "estimated_total_median": 800,
                    "estimated_total_p75": 800,
                }
            ],
            columns=query_estimate_llm.UNIT_PRICE_COLUMNS,
        )
        matches = pd.DataFrame([{column: "" for column in query_estimate_llm.MATCH_COLUMNS}])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "query_result.xlsx"
            query_estimate_llm.write_query_result_workbook(output_path, summary, unit_price_by_unit, matches)
            workbook = openpyxl.load_workbook(output_path, read_only=True, data_only=True)
            self.assertEqual(workbook.sheetnames, ["summary", "unit_price_by_unit", "matches"])
            summary_headers = [workbook["summary"].cell(row=1, column=column).value for column in range(1, 5)]
            match_headers = [workbook["matches"].cell(row=1, column=column).value for column in range(1, 5)]
            workbook.close()

        self.assertEqual(summary_headers, ["raw_text", "project_text", "item_text", "feature_text"])
        self.assertEqual(match_headers, ["rank", "final_score", "item_score", "project_score"])


if __name__ == "__main__":
    unittest.main()
