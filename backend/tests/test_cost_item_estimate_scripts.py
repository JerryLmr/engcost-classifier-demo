from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from datetime import date
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
    query_estimate_llm = load_script_module("query_cost_estimate_llm", "scripts/query_cost_estimate_llm.py")
else:
    build_index = None
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
                    "unit": "平方米",
                    "unit_normalized": "m²",
                    "quantity": 100,
                    "unit_price": 80,
                    "total_price": 8000,
                    "labor_unit_price": 20,
                    "machinery_unit_price": 5,
                    "item_similarity_text": "屋面卷材防水；3.0mm SBS 沥青防水卷材",
                    "item_context_text": "屋面漏水维修工程 / 屋面卷材防水",
                    "consultation_time": "2026-03-01",
                    "location": "浙江省嘉兴市",
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
                    "unit": "平方米",
                    "unit_normalized": "m²",
                    "quantity": 200,
                    "unit_price": 100,
                    "total_price": 20000,
                    "labor_unit_price": 30,
                    "machinery_unit_price": 8,
                    "item_similarity_text": "外墙防水；外墙渗漏处理",
                    "item_context_text": "外墙维修工程 / 外墙防水",
                    "consultation_time": "2024-01-01",
                    "location": "浙江省杭州市",
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
                    "unit": "米",
                    "unit_normalized": "m",
                    "quantity": 50,
                    "unit_price": None,
                    "total_price": 3000,
                    "labor_unit_price": 12,
                    "machinery_unit_price": None,
                    "item_similarity_text": "管道更换；给水管道更换",
                    "item_context_text": "管道维修工程 / 管道更换",
                    "consultation_time": "2026-02-01",
                    "location": "浙江省嘉兴市",
                },
            ]
        )

    def project_groups_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source_row_id": 2,
                    "工程名称": "屋面漏水维修工程",
                    "catalog_id": "CP-002-03",
                    "一级分类": "屋面",
                    "二级分类": "防水层",
                    "维修状态": "维修",
                    "标准对象": "共用部位",
                    "consultation_time": "2026-03-01",
                    "location": "浙江省嘉兴市",
                    "group_text": "屋面漏水维修工程 屋面卷材防水 3mm SBS",
                    "item_count": 1,
                },
                {
                    "source_row_id": 3,
                    "工程名称": "外墙维修工程",
                    "catalog_id": "CP-003-01",
                    "一级分类": "外墙面",
                    "二级分类": "面层",
                    "维修状态": "维修",
                    "标准对象": "共用部位",
                    "consultation_time": "2024-01-01",
                    "location": "浙江省杭州市",
                    "group_text": "外墙维修工程 外墙防水",
                    "item_count": 1,
                },
                {
                    "source_row_id": 4,
                    "工程名称": "管道维修工程",
                    "catalog_id": "CF-015-04",
                    "一级分类": "给排水系统",
                    "二级分类": "管道",
                    "维修状态": "维修",
                    "标准对象": "共用设施设备",
                    "consultation_time": "2026-02-01",
                    "location": "浙江省嘉兴市",
                    "group_text": "管道维修工程 管道更换",
                    "item_count": 1,
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
                query_estimate_llm.validate_output_path(output_path, overwrite=False)
            query_estimate_llm.validate_output_path(output_path, overwrite=True)

    def test_query_load_index_reads_project_group_index_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            self.sample_frame().to_parquet(index_dir / "samples.parquet", index=False)
            self.project_groups_frame().to_parquet(index_dir / "project_groups.parquet", index=False)
            np.save(index_dir / "project_group_embeddings.npy", np.ones((3, 2), dtype=np.float32))
            (index_dir / "index_meta.json").write_text('{"model": "demo-model"}', encoding="utf-8")

            samples, project_groups, embeddings, meta = query_estimate_llm.load_index(index_dir)

        self.assertEqual(len(samples), 3)
        self.assertEqual(len(project_groups), 3)
        self.assertEqual(embeddings.shape, (3, 2))
        self.assertEqual(meta["model"], "demo-model")

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

    def test_parse_query_requirements_uses_llm_then_normalizes_fields(self):
        llm_result = {
            "semantic_query_text": "屋面漏水，3mm SBS防水",
            "quantity": "500",
            "unit": "平",
            "location_hint": "嘉兴",
            "time_range_type": "last_year",
            "catalog_id": "SHOULD_IGNORE",
        }

        with patch.object(query_estimate_llm, "request_llm_json", return_value=llm_result):
            parsed = query_estimate_llm.parse_query_requirements("屋面漏水，面积500平，参考嘉兴一年内", self.project_groups_frame())

        self.assertEqual(parsed.semantic_query_text, "屋面漏水，3mm SBS防水")
        self.assertEqual(parsed.quantity, 500.0)
        self.assertEqual(parsed.unit, "m²")
        self.assertEqual(parsed.location, "浙江省嘉兴市")
        self.assertEqual((parsed.consultation_time_to - parsed.consultation_time_from).days, 365)

    def test_parse_query_requirements_falls_back_on_llm_failure(self):
        with patch.object(query_estimate_llm, "request_llm_json", side_effect=query_estimate_llm.LLMServiceError("down")):
            parsed = query_estimate_llm.parse_query_requirements("屋面漏水", self.project_groups_frame())

        self.assertEqual(parsed.semantic_query_text, "屋面漏水")
        self.assertIsNone(parsed.quantity)
        self.assertEqual(parsed.unit, "")
        self.assertEqual(parsed.location, "")
        self.assertIn("LLM 解析失败", parsed.parse_notes[0])

    def test_time_range_to_dates_uses_base_date(self):
        time_from, time_to = query_estimate_llm.time_range_to_dates("last_3_months", date(2026, 6, 26))
        self.assertEqual(time_from, date(2026, 3, 28))
        self.assertEqual(time_to, date(2026, 6, 26))
        self.assertEqual(query_estimate_llm.time_range_to_dates("none", date(2026, 6, 26)), (None, None))

    def test_filter_project_groups_applies_location_and_time_as_hard_filters(self):
        project_groups = self.project_groups_frame()
        embeddings = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
        parsed = query_estimate_llm.ParsedQuery(
            raw_query="屋面",
            semantic_query_text="屋面",
            quantity=None,
            unit="",
            location="浙江省嘉兴市",
            consultation_time_from=date(2026, 1, 1),
            consultation_time_to=date(2026, 12, 31),
            parse_notes=[],
        )

        filtered, filtered_embeddings = query_estimate_llm.filter_project_groups(project_groups, embeddings, parsed)

        self.assertEqual(filtered["source_row_id"].tolist(), [2, 4])
        self.assertEqual(filtered_embeddings.tolist(), [[1.0, 0.0], [0.0, 1.0]])

        empty = query_estimate_llm.ParsedQuery("x", "x", None, "", "上海市", None, None, [])
        filtered, filtered_embeddings = query_estimate_llm.filter_project_groups(project_groups, embeddings, empty)
        self.assertTrue(filtered.empty)
        self.assertEqual(filtered_embeddings.shape[0], 0)

    def test_score_project_groups_and_expand_samples_by_source_row_id(self):
        samples = self.sample_frame()
        duplicate = samples.iloc[0].copy()
        duplicate["item_row_id"] = "2-2"
        duplicate["seq"] = 2
        duplicate["cost_item_name"] = "基层处理"
        samples = pd.concat([samples, pd.DataFrame([duplicate])], ignore_index=True)
        project_groups = self.project_groups_frame()
        embeddings = np.array([[1.0, 0.0], [0.6, 0.4], [0.0, 1.0]], dtype=np.float32)

        selected = query_estimate_llm.score_project_groups(project_groups, embeddings, np.array([1.0, 0.0], dtype=np.float32), 1)
        matches = query_estimate_llm.expand_matched_samples(samples, selected)

        self.assertEqual(selected["source_row_id"].tolist(), [2])
        self.assertEqual(selected["project_rank"].tolist(), [1])
        self.assertEqual(matches["item_row_id"].tolist(), ["2-1", "2-2"])
        self.assertIn("group_text", matches.columns)

    def test_recommend_items_aggregate_and_estimate_totals(self):
        samples = self.sample_frame()
        duplicate = samples.iloc[0].copy()
        duplicate["source_row_id"] = 5
        duplicate["item_row_id"] = "5-1"
        duplicate["unit_price"] = 120
        duplicate["total_price"] = 12000
        samples = pd.concat([samples, pd.DataFrame([duplicate])], ignore_index=True)
        matched_projects = pd.DataFrame(
            [
                {"source_row_id": 2, "project_rank": 1, "project_score": 0.9, "group_text": "屋面"},
                {"source_row_id": 5, "project_rank": 2, "project_score": 0.8, "group_text": "屋面2"},
                {"source_row_id": 4, "project_rank": 3, "project_score": 0.7, "group_text": "管道"},
            ]
        )
        matches = query_estimate_llm.expand_matched_samples(samples, matched_projects)
        parsed = query_estimate_llm.ParsedQuery("屋面", "屋面", 500.0, "m²", "", None, None, [])

        recommended = query_estimate_llm.aggregate_recommend_items(matches, parsed)

        self.assertEqual(recommended["cost_item_name"].tolist()[:2], ["屋面卷材防水", "管道更换"])
        roof = recommended.iloc[0]
        self.assertEqual(roof["recommend_rank"], 1)
        self.assertEqual(roof["sample_count"], 2)
        self.assertEqual(roof["unit_price_min"], 80.0)
        self.assertEqual(roof["unit_price_median"], 100.0)
        self.assertEqual(roof["unit_price_max"], 120.0)
        self.assertEqual(roof["estimated_total_median"], 50000.0)
        self.assertEqual(roof["estimate_basis"], "按用户工程量 × 历史综合单价估算")
        pipe = recommended[recommended["cost_item_name"] == "管道更换"].iloc[0]
        self.assertEqual(pipe["estimated_total_median"], 3000.0)
        self.assertEqual(pipe["estimate_basis"], "单位不一致，展示历史样本总价范围")

    def test_write_query_result_workbook_has_two_sheets_and_debug_toggle(self):
        samples = self.sample_frame()
        matched_projects = pd.DataFrame(
            [{"source_row_id": 2, "project_rank": 1, "project_score": 0.9, "group_text": "debug group text"}]
        )
        matches = query_estimate_llm.expand_matched_samples(samples, matched_projects)
        parsed = query_estimate_llm.ParsedQuery("屋面", "屋面", None, "", "", None, None, [])
        recommended = query_estimate_llm.aggregate_recommend_items(matches, parsed)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "query_result.xlsx"
            query_estimate_llm.write_query_result_workbook(output_path, recommended, matches, include_debug_text=False)
            workbook = openpyxl.load_workbook(output_path, data_only=True)
            self.assertEqual(workbook.sheetnames, ["recommend_items", "matches"])
            match_headers = [workbook["matches"].cell(row=1, column=column).value for column in range(1, workbook["matches"].max_column + 1)]
            self.assertNotIn("工程名称", match_headers)
            self.assertNotIn("project_description", match_headers)
            self.assertNotIn("group_text", match_headers)
            workbook.close()

            debug_path = Path(tmpdir) / "query_result_debug.xlsx"
            query_estimate_llm.write_query_result_workbook(debug_path, recommended, matches, include_debug_text=True)
            workbook = openpyxl.load_workbook(debug_path, data_only=True)
            debug_headers = [workbook["matches"].cell(row=1, column=column).value for column in range(1, workbook["matches"].max_column + 1)]
            workbook.close()

        self.assertIn("工程名称", debug_headers)
        self.assertIn("project_description", debug_headers)
        self.assertIn("group_text", debug_headers)

    def test_legacy_query_entrypoint_removed_and_readme_points_to_official_script(self):
        legacy_name = "query_cost_" "item_estimate.py"
        self.assertFalse((ROOT / "scripts" / legacy_name).exists())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("scripts/query_cost_estimate_llm.py", readme)
        self.assertNotIn(f"scripts/{legacy_name}", readme)


if __name__ == "__main__":
    unittest.main()
