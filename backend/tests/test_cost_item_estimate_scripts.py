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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_samples_script = load_script_module("build_cost_item_samples", "scripts/build_cost_item_samples.py")
run_ingest_batch = load_script_module("run_ingest_batch", "scripts/run_ingest_batch.py")
merge_samples = load_script_module("merge_cost_item_sample_batches", "scripts/merge_cost_item_sample_batches.py")

if np is not None and pd is not None:
    build_index = load_script_module("build_cost_item_embedding_index", "scripts/build_cost_item_embedding_index.py")
    query_estimate_llm = load_script_module("query_cost_estimate_llm", "scripts/query_cost_estimate_llm.py")
else:
    build_index = None
    query_estimate_llm = None


@unittest.skipIf(np is None or pd is None, "cost item estimate dependencies are not installed")
class CostItemEstimateScriptTestCase(unittest.TestCase):
    def raw_sample_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "project_key": "batch-a::2",
                    "batch_id": "batch-a",
                    "source_row_id": 2,
                    "item_row_id": "2-1",
                    "工程名称": "屋面漏水维修工程",
                    "project_name_text": "屋面漏水维修",
                    "catalog_id": "CP-002-03",
                    "一级分类": "屋面",
                    "二级分类": "防水层",
                    "维修状态": "维修",
                    "标准对象": "共用部位",
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
                    "consultation_time": "2026-03-01",
                    "location": "浙江省嘉兴市",
                },
                {
                    "project_key": "batch-a::2",
                    "batch_id": "batch-a",
                    "source_row_id": 2,
                    "item_row_id": "2-2",
                    "工程名称": "屋面漏水维修工程",
                    "project_name_text": "屋面漏水维修",
                    "catalog_id": "CP-002-03",
                    "一级分类": "屋面",
                    "二级分类": "防水层",
                    "维修状态": "维修",
                    "标准对象": "共用部位",
                    "seq": 2,
                    "cost_item_name": "防水层拆除",
                    "project_description": "拆除原屋面防水层",
                    "unit": "平方米",
                    "unit_normalized": "m²",
                    "quantity": 100,
                    "unit_price": 20,
                    "total_price": 2000,
                    "labor_unit_price": 12,
                    "machinery_unit_price": 1,
                    "consultation_time": "2026-03-01",
                    "location": "浙江省嘉兴市",
                },
                {
                    "project_key": "batch-a::3",
                    "batch_id": "batch-a",
                    "source_row_id": 3,
                    "item_row_id": "3-1",
                    "工程名称": "外墙维修工程",
                    "project_name_text": "外墙维修",
                    "catalog_id": "CP-003-01",
                    "一级分类": "外墙面",
                    "二级分类": "面层",
                    "维修状态": "维修",
                    "标准对象": "共用部位",
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
                    "consultation_time": "2024-01-01",
                    "location": "浙江省杭州市",
                },
                {
                    "project_key": "batch-a::4",
                    "batch_id": "batch-a",
                    "source_row_id": 4,
                    "item_row_id": "4-1",
                    "工程名称": "管道维修工程",
                    "project_name_text": "管道维修",
                    "catalog_id": "CF-015-04",
                    "一级分类": "给排水系统",
                    "二级分类": "管道",
                    "维修状态": "维修",
                    "标准对象": "共用设施设备",
                    "seq": 1,
                    "cost_item_name": "管道更换",
                    "project_description": "DN100 镀锌钢管更换",
                    "unit": "米",
                    "unit_normalized": "m",
                    "quantity": 30,
                    "unit_price": 100,
                    "total_price": 3000,
                    "labor_unit_price": "",
                    "machinery_unit_price": "",
                    "consultation_time": "2026-02-01",
                    "location": "浙江省嘉兴市",
                },
            ]
        )

    def prepared_samples(self) -> pd.DataFrame:
        samples = build_index.normalize_numeric_columns(
            build_index.ensure_optional_columns(build_index.ensure_project_key(self.raw_sample_frame()))
        )
        samples.insert(0, "sample_index", range(len(samples)))
        samples["project_package_id"] = samples["project_key"]
        samples["item_retrieval_text"] = samples.apply(build_index.build_item_retrieval_text, axis=1)
        samples["fine_signature"] = samples.apply(build_index.build_fine_signature, axis=1)
        samples["family_signature"] = samples.apply(build_index.build_family_signature, axis=1)
        return samples

    def test_normalize_embeddings_handles_zero_vector(self):
        embeddings = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        normalized = build_index.normalize_embeddings(embeddings)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(normalized[0])), 1.0)
        self.assertEqual(normalized[1].tolist(), [0.0, 0.0])

    def test_shared_normalize_unit_handles_square_and_cubic_units(self):
        self.assertEqual(build_samples_script.normalize_unit("m^2"), "m²")
        self.assertEqual(build_samples_script.normalize_unit("平方米"), "m²")
        self.assertEqual(build_samples_script.normalize_unit("平"), "m²")
        self.assertEqual(build_samples_script.normalize_unit("m^{3}"), "m³")
        self.assertEqual(build_samples_script.normalize_unit(" 台 "), "台")

    def test_item_retrieval_text_contains_only_item_description_and_unit(self):
        row = self.prepared_samples().iloc[0]

        self.assertEqual(
            row["item_retrieval_text"],
            "清单项：屋面卷材防水\n项目特征：3.0mm SBS 沥青防水卷材\n单位：m²",
        )
        self.assertNotIn("一级分类", row["item_retrieval_text"])
        self.assertNotIn("工程语义", row["item_retrieval_text"])
        self.assertEqual(row["fine_signature"], "屋面卷材防水 | 3.0mm sbs 沥青防水卷材 | m²")
        self.assertEqual(row["family_signature"], "屋面卷材防水 | m²")

    def test_project_packages_use_cost_item_names_summary_for_embedding(self):
        packages = build_index.build_project_packages(self.prepared_samples())

        roof = packages[packages["project_package_id"] == "batch-a::2"].iloc[0]
        self.assertEqual(roof["cost_item_names_summary"], "屋面卷材防水；防水层拆除")
        self.assertEqual(roof["item_summary"], "屋面卷材防水；防水层拆除")
        self.assertEqual(
            roof["package_text"],
            "工程名称：屋面漏水维修工程\n工程语义：屋面漏水维修\n包含清单项：屋面卷材防水；防水层拆除",
        )
        self.assertNotIn("CP-002-03", roof["package_text"])
        self.assertNotIn("3.0mm SBS", roof["package_text"])
        self.assertNotIn("单位", roof["package_text"])
        self.assertIn("catalog_summary", packages.columns)

    def test_build_index_meta_describes_new_retrieval_fields(self):
        meta = build_index.build_index_meta(Path("samples.xlsx"), "demo-model", 4, 3, 2)

        self.assertEqual(meta["files"]["project_package_embeddings"], "project_package_embeddings.npy")
        self.assertEqual(meta["files"]["item_embeddings"], "item_embeddings.npy")
        self.assertIn("cost_item_name、project_description、unit_normalized", meta["field_descriptions"]["item_retrieval_text"])
        self.assertIn("cost_item_names_summary", meta["field_descriptions"]["package_text"])

    def test_write_index_outputs_new_files_and_removes_legacy_files(self):
        samples = self.prepared_samples()
        packages = build_index.build_project_packages(samples)
        package_embeddings = np.ones((len(packages), 2), dtype=np.float32)
        item_embeddings = np.ones((len(samples), 2), dtype=np.float32)
        meta = build_index.build_index_meta(Path("samples.xlsx"), "demo-model", len(samples), len(packages), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "index"
            output_dir.mkdir()
            for name in build_index.LEGACY_OUTPUT_FILES:
                (output_dir / name).write_text("legacy", encoding="utf-8")

            build_index.write_index(samples, packages, package_embeddings, item_embeddings, output_dir, meta)

            self.assertTrue((output_dir / "samples.parquet").exists())
            self.assertTrue((output_dir / "project_packages.parquet").exists())
            self.assertTrue((output_dir / "project_package_embeddings.npy").exists())
            self.assertTrue((output_dir / "item_embeddings.npy").exists())
            for name in build_index.LEGACY_OUTPUT_FILES:
                self.assertFalse((output_dir / name).exists())

    def test_build_index_validate_output_dir_requires_overwrite_for_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "index"
            output_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "输出已存在，请加 --overwrite"):
                build_index.validate_output_dir(output_dir, overwrite=False)
            build_index.validate_output_dir(output_dir, overwrite=True)

    def test_query_parse_args_uses_new_top_package_and_item_options(self):
        with patch.object(sys, "argv", ["query_cost_estimate_llm.py", "--text", "屋面漏水"]):
            args = query_estimate_llm.parse_args()

        self.assertEqual(args.index_dir, "embeddings")
        self.assertEqual(args.top_packages, 20)
        self.assertEqual(args.top_items, 300)
        self.assertEqual(args.llm_check_timeout, 3.0)
        self.assertFalse(hasattr(args, "top_k"))
        self.assertFalse(hasattr(args, "project_name_weight"))

    def test_query_main_checks_llm_service_before_run_query(self):
        call_order: list[str] = []

        def check_service(timeout_seconds: float):
            call_order.append(f"check:{timeout_seconds}")

        def run_query(**_kwargs):
            call_order.append("run_query")
            return object()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "query.xlsx"
            with patch.object(
                sys,
                "argv",
                [
                    "query_cost_estimate_llm.py",
                    "--text",
                    "屋面漏水",
                    "--output",
                    str(output_path),
                    "--llm-check-timeout",
                    "1.5",
                ],
            ), patch.object(query_estimate_llm, "check_lmstudio_service", side_effect=check_service) as check_mock, patch.object(
                query_estimate_llm,
                "run_query",
                side_effect=run_query,
            ) as run_mock, patch.object(query_estimate_llm, "print_terminal_summary"):
                exit_code = query_estimate_llm.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(call_order, ["check:1.5", "run_query"])
        check_mock.assert_called_once_with(timeout_seconds=1.5)
        run_mock.assert_called_once()

    def test_query_main_returns_before_run_query_when_llm_service_unavailable(self):
        with patch.object(sys, "argv", ["query_cost_estimate_llm.py", "--text", "屋面漏水"]), patch.object(
            query_estimate_llm,
            "check_lmstudio_service",
            side_effect=RuntimeError("LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1"),
        ) as check_mock, patch.object(query_estimate_llm, "run_query") as run_mock, patch("builtins.print") as print_mock:
            exit_code = query_estimate_llm.main()

        self.assertEqual(exit_code, 1)
        check_mock.assert_called_once_with(timeout_seconds=3.0)
        run_mock.assert_not_called()
        print_mock.assert_called_once_with(
            "[ERROR] LLM 服务不可用，请先启动 LM Studio Server，并检查 "
            "LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1"
        )

    def test_query_load_index_reads_new_files_and_validates_shapes(self):
        samples = self.prepared_samples()
        packages = build_index.build_project_packages(samples)
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir)
            samples.to_parquet(index_dir / "samples.parquet", index=False)
            packages.to_parquet(index_dir / "project_packages.parquet", index=False)
            np.save(index_dir / "project_package_embeddings.npy", np.ones((len(packages), 2), dtype=np.float32))
            np.save(index_dir / "item_embeddings.npy", np.zeros((len(samples), 2), dtype=np.float32))
            (index_dir / "index_meta.json").write_text('{"model": "demo-model"}', encoding="utf-8")

            loaded_samples, loaded_packages, package_embeddings, item_embeddings, meta = query_estimate_llm.load_index(index_dir)

        self.assertEqual(len(loaded_samples), 4)
        self.assertEqual(len(loaded_packages), 3)
        self.assertEqual(package_embeddings.shape, (3, 2))
        self.assertEqual(item_embeddings.shape, (4, 2))
        self.assertEqual(meta["model"], "demo-model")

    def test_query_rewrite_outputs_two_queries_and_fallbacks_empty_item_query(self):
        llm_result = {
            "project_package_query_text": "屋面漏水维修工程 屋面卷材防水",
            "item_query_text": "",
            "parsed_quantities": [{"raw_text": "500平", "value": 500, "unit": "m²"}],
            "materials_or_specs": ["3mm SBS"],
            "repair_object": "屋面防水层",
            "uncertainties": ["是否拆除旧防水层未知"],
            "likely_catalog": {"SHOULD": "IGNORE"},
        }
        with patch.object(query_estimate_llm, "request_llm_json", return_value=llm_result):
            rewrite, trace = query_estimate_llm.query_rewrite_for_embedding("屋面漏水")

        self.assertTrue(rewrite.success)
        self.assertEqual(rewrite.item_query_text, "屋面漏水维修工程 屋面卷材防水")
        self.assertEqual(rewrite.parsed_quantities[0]["value"], 500)
        self.assertEqual(rewrite.materials_or_specs, ["3mm SBS"])
        self.assertIn("item_query_text 为空", rewrite.notes[0])
        self.assertEqual(trace["step"], "query_rewrite_for_embedding")

        with patch.object(query_estimate_llm, "request_llm_json", side_effect=query_estimate_llm.LLMServiceError("down")):
            fallback, trace = query_estimate_llm.query_rewrite_for_embedding("屋面漏水")

        self.assertFalse(fallback.success)
        self.assertEqual(fallback.project_package_query_text, "屋面漏水")
        self.assertEqual(fallback.item_query_text, "屋面漏水")
        self.assertEqual(fallback.parsed_quantities, [])
        self.assertEqual(trace["success"], "否")

    def test_classify_query_catalog_reuses_standard_classifier(self):
        result = {
            "catalog_id": "CP-002-03",
            "category": "屋面",
            "item": "防水层",
            "repair_status": "维修",
            "standard_group": "共用部位",
            "pipeline_status": "classified",
        }
        with patch.object(query_estimate_llm, "classify_project_standard", return_value=result) as classifier:
            catalog, trace = query_estimate_llm.classify_query_catalog(
                "屋面漏水",
                "屋面漏水维修工程 屋面卷材防水",
                "屋面卷材防水 3mm SBS",
            )

        classifier.assert_called_once()
        self.assertTrue(catalog.success)
        self.assertEqual(catalog.catalog_id, "CP-002-03")
        self.assertEqual(catalog.一级分类, "屋面")
        self.assertEqual(catalog.二级分类, "防水层")
        self.assertEqual(trace["step"], "query_catalog_classification")

    def test_catalog_score_uses_query_catalog_and_neutral_on_failure(self):
        row = self.prepared_samples().iloc[0]
        catalog = query_estimate_llm.QueryCatalog("CP-002-03", "屋面", "防水层", "维修", "共用部位", None, {}, True, [])
        failed = query_estimate_llm.QueryCatalog("", "", "", "", "", None, {}, False, ["fail"])

        self.assertEqual(query_estimate_llm.catalog_score(row, catalog), 1.0)
        self.assertEqual(query_estimate_llm.catalog_score(row, failed), 0.5)
        changed = row.copy()
        changed["catalog_id"] = "DIFFERENT"
        self.assertEqual(query_estimate_llm.catalog_score(changed, catalog), 0.8)

    def test_package_recall_dedupes_repeated_packages_before_top_k(self):
        packages = build_index.build_project_packages(self.prepared_samples())
        duplicate = packages.iloc[0].copy()
        duplicate["project_package_id"] = "batch-z::99"
        packages = pd.concat([packages, pd.DataFrame([duplicate])], ignore_index=True)
        embeddings = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [1.0, 0.0]], dtype=np.float32)

        matched = query_estimate_llm.score_project_packages(
            packages,
            embeddings,
            np.array([1.0, 0.0], dtype=np.float32),
            top_packages=3,
        )

        self.assertEqual(len(matched), 3)
        self.assertEqual(matched["rank"].tolist(), [1, 2, 3])
        self.assertEqual(matched["工程名称"].tolist().count("屋面漏水维修工程"), 1)

    def test_candidate_pool_uses_item_scores_and_query_catalog(self):
        samples = self.prepared_samples()
        packages = build_index.build_project_packages(samples)
        matched = packages[packages["project_package_id"].isin(["batch-a::2", "batch-a::3"])].copy()
        matched.insert(0, "package_score", [0.9, 0.6])
        matched.insert(0, "rank", [1, 2])
        direct = samples[samples["sample_index"].isin([0, 3])].copy()
        item_scores = np.array([0.95, 0.2, 0.3, 0.7], dtype=np.float32)
        catalog = query_estimate_llm.QueryCatalog("CP-002-03", "屋面", "防水层", "维修", "共用部位", None, {}, True, [])

        candidates = query_estimate_llm.candidate_pool(samples, matched, direct, item_scores, catalog)

        self.assertEqual(sorted(candidates["sample_index"].tolist()), [0, 1, 2, 3])
        roof = candidates[candidates["sample_index"] == 0].iloc[0]
        pipe = candidates[candidates["sample_index"] == 3].iloc[0]
        self.assertAlmostEqual(float(roof["package_score"]), 0.9)
        self.assertAlmostEqual(float(roof["item_score"]), 0.95)
        self.assertAlmostEqual(float(roof["cooccur_score"]), 0.5)
        self.assertEqual(float(roof["catalog_score"]), 1.0)
        self.assertEqual(roof["source_ref"], "batch-a::2::2-1")
        self.assertEqual(pipe["package_score"], 0.0)
        self.assertEqual(pipe["direct_hit"], True)
        self.assertEqual(pipe["catalog_score"], 0.55)

    def test_candidate_item_stats_include_labor_and_machinery_stats(self):
        candidates = self.prepared_samples().head(2).copy()
        candidates["package_score"] = 0.8
        candidates["package_rank"] = 1
        candidates["item_score"] = [0.9, 0.2]
        candidates["cooccur_score"] = 1.0
        candidates["catalog_score"] = 1.0
        candidates["unit_score"] = 0.5
        candidates["final_score"] = [0.85, 0.6]
        candidates["source_ref"] = ["batch-a::2::2-1", "batch-a::2::2-2"]

        stats = query_estimate_llm.build_candidate_item_stats(candidates)

        self.assertIn("candidate_id", stats.columns)
        self.assertIn("source_refs", stats.columns)
        self.assertIn("历史人工单价中位数", stats.columns)
        self.assertIn("历史机械单价中位数", stats.columns)
        roof = stats[stats["cost_item_name"] == "屋面卷材防水"].iloc[0]
        self.assertEqual(roof["candidate_id"], "C001")
        self.assertEqual(roof["历史人工单价中位数"], 20.0)
        self.assertEqual(roof["历史机械单价中位数"], 5.0)
        self.assertEqual(roof["source_refs"], "batch-a::2::2-1")

    def test_candidate_families_group_by_family_signature(self):
        candidates = self.prepared_samples().head(1).copy()
        duplicate = candidates.iloc[0].copy()
        duplicate["project_description"] = "4.0mm SBS 沥青防水卷材"
        duplicate["fine_signature"] = "屋面卷材防水 | 4.0mm sbs 沥青防水卷材 | m²"
        duplicate["quantity"] = 300
        duplicate["unit_price"] = 120
        duplicate["total_price"] = 36000
        duplicate["labor_unit_price"] = 30
        duplicate["machinery_unit_price"] = 7
        duplicate["project_key"] = "batch-a::5"
        duplicate["project_package_id"] = "batch-a::5"
        duplicate["item_row_id"] = "5-1"
        candidates = pd.concat([candidates, duplicate.to_frame().T], ignore_index=True)
        candidates["package_score"] = [0.8, 0.7]
        candidates["package_rank"] = [1, 2]
        candidates["item_score"] = [0.9, 0.8]
        candidates["cooccur_score"] = 1.0
        candidates["catalog_score"] = 1.0
        candidates["unit_score"] = 0.5
        candidates["final_score"] = [0.85, 0.75]
        candidates["source_ref"] = ["batch-a::2::2-1", "batch-a::5::5-1"]

        stats = query_estimate_llm.build_candidate_item_stats(candidates)
        families = query_estimate_llm.build_candidate_families(candidates, stats)

        self.assertEqual(families.columns.tolist(), query_estimate_llm.CANDIDATE_FAMILY_COLUMNS)
        self.assertEqual(len(families), 1)
        family = families.iloc[0]
        self.assertEqual(family["family_id"], "F001")
        self.assertEqual(family["覆盖candidate_ids"], "C001, C002")
        self.assertEqual(family["历史样本数"], 2)
        self.assertEqual(family["来源工程包数"], 2)
        self.assertEqual(family["历史综合单价最低值"], 80.0)
        self.assertEqual(family["历史综合单价中位数"], 100.0)
        self.assertEqual(family["历史综合单价最高值"], 120.0)
        self.assertEqual(family["representative_project_description"], "3.0mm SBS 沥青防水卷材")

    def test_source_ref_recovers_from_batch_source_row_and_seq(self):
        rows = self.prepared_samples().head(1).copy()
        rows["project_key"] = ""
        rows["item_row_id"] = ""
        warnings: list[str] = []

        recovered = query_estimate_llm.attach_source_refs(rows, warnings)

        self.assertEqual(recovered.iloc[0]["project_key"], "batch-a::2")
        self.assertEqual(recovered.iloc[0]["item_row_id"], "2-1")
        self.assertEqual(recovered.iloc[0]["source_ref"], "batch-a::2::2-1")
        self.assertIn("source_ref_recovered_from_batch_source_row", warnings)
        self.assertIn("source_ref_recovered_from_source_row_seq", warnings)

    def test_compressed_llm_payload_limits_rows_and_refs(self):
        candidates = pd.DataFrame(
            [
                {
                    "fine_signature": f"sig-{index}",
                    "family_signature": f"fam-{index}",
                    "cost_item_name": f"项{index}",
                    "project_description": "做法",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "历史样本数": 1,
                    "来源工程包数": 1,
                    "final_score": 1 - index / 100,
                    "candidate_id": f"C{index + 1:03d}",
                    "source_refs": "batch-a::1::1-1, batch-a::1::1-2, batch-a::1::1-3, batch-a::1::1-4, batch-a::1::1-5, batch-a::1::1-6",
                }
                for index in range(40)
            ]
        )

        compressed = query_estimate_llm.compressed_candidates_for_llm(candidates, 30)

        self.assertEqual(len(compressed), 30)
        self.assertEqual(
            compressed.iloc[0]["source_refs_sample"],
            ["batch-a::1::1-1", "batch-a::1::1-2", "batch-a::1::1-3", "batch-a::1::1-4", "batch-a::1::1-5"],
        )
        self.assertNotIn("source_refs", compressed.columns)

        families = pd.DataFrame(
            [
                {
                    "family_id": f"F{index + 1:03d}",
                    "family_signature": f"fam-{index}",
                    "representative_cost_item_name": f"族{index}",
                    "representative_project_description": "代表做法",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "覆盖candidate_ids": f"C{index + 1:03d}",
                    "历史样本数": 1,
                    "来源工程包数": 1,
                    "final_score": 1 - index / 100,
                    "source_refs": "batch-a::1::1-1, batch-a::1::1-2, batch-a::1::1-3, batch-a::1::1-4, batch-a::1::1-5, batch-a::1::1-6",
                }
                for index in range(25)
            ]
        )

        compressed_families = query_estimate_llm.compressed_families_for_llm(families, 20)

        self.assertEqual(len(compressed_families), 20)
        self.assertEqual(
            compressed_families.iloc[0]["source_refs_sample"],
            ["batch-a::1::1-1", "batch-a::1::1-2", "batch-a::1::1-3", "batch-a::1::1-4", "batch-a::1::1-5"],
        )
        self.assertNotIn("source_refs", compressed_families.columns)

    def test_suggested_bill_success_and_fallback_columns(self):
        result = {
            "suggested_bill": [
                {
                    "seq": 1,
                    "family_id": "F001",
                    "candidate_id": "C001",
                    "recommend_type": "直接匹配项",
                    "item_role": "核心施工项",
                    "estimate_method": "按面积线性估算",
                    "cost_item_name": "屋面卷材防水",
                    "project_description": "3mm SBS",
                    "unit": "m²",
                    "suggested_quantity": 500,
                    "unit_price_low": 80,
                    "labor_unit_price_mid": 20,
                    "estimated_labor_amount_mid": 10000,
                    "source_refs": ["SHOULD_IGNORE"],
                }
            ]
        }
        warnings: list[str] = []

        bill = query_estimate_llm.suggested_bill_from_llm_result(result, warnings)

        self.assertEqual(bill.columns.tolist(), query_estimate_llm.SUGGESTED_BILL_COLUMNS)
        self.assertEqual(bill.loc[0, "family_id"], "F001")
        self.assertEqual(bill.loc[0, "candidate_id"], "C001")
        self.assertEqual(bill.loc[0, "项目角色"], "核心施工项")
        self.assertEqual(bill.loc[0, "其中包含人工费单价中位数"], 20)
        self.assertEqual(bill.loc[0, "估算人工费中位数"], 10000)
        self.assertEqual(bill.loc[0, "来源样本"], "")
        self.assertIn("llm_source_fields_ignored", warnings)

        fallback = query_estimate_llm.fallback_suggested_bill(
            pd.DataFrame(
                [
                    {
                        "family_id": "F001",
                        "candidate_id": "C001",
                        "cost_item_name": "屋面卷材防水",
                        "project_description": "3mm SBS",
                        "unit": "m²",
                        "unit_normalized": "m²",
                        "历史人工单价中位数": 20,
                        "source_refs": "batch-a::2::2-1",
                    }
                ]
            )
        )
        self.assertEqual(fallback.loc[0, "推荐类型"], "fallback_candidate")
        self.assertIn("不代表最终建议清单", fallback.loc[0, "采用理由"])
        self.assertEqual(fallback.loc[0, "不确定性说明"], "需修复 LLM 上下文或降低候选规模后重新生成")
        self.assertEqual(fallback.loc[0, "来源样本"], "batch-a::2::2-1")

    def test_postprocess_overrides_prices_amounts_and_marks_adoption(self):
        stats = pd.DataFrame(
            [
                {
                    "candidate_id": "C001",
                    "fine_signature": "sig",
                    "family_signature": "fam",
                    "cost_item_name": "屋面卷材防水",
                    "project_description": "3mm SBS",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "历史样本数": 1,
                    "来源工程包数": 1,
                    "历史综合单价最低值": 80,
                    "历史综合单价中位数": 90,
                    "历史综合单价最高值": 100,
                    "历史人工单价中位数": 20,
                    "历史机械单价中位数": None,
                    "source_refs": "batch-a::2::2-1",
                    "是否被LLM采用": "",
                }
            ],
            columns=query_estimate_llm.CANDIDATE_ITEM_STATS_COLUMNS,
        )
        bill = pd.DataFrame(
            [
                {
                    "序号": 1,
                    "candidate_id": "C001",
                    "推荐类型": "直接匹配项",
                    "项目角色": "核心施工项",
                    "计量/估算口径": "按面积估算",
                    "清单项名称": "屋面卷材防水",
                    "项目特征/施工工艺": "3mm SBS",
                    "单位": "m²",
                    "建议工程量": 10,
                    "工程量依据": "用户给定",
                    "综合单价最低值": 1,
                    "综合单价中位数": 2,
                    "综合单价最高值": 3,
                    "估算金额最低值": 1,
                    "估算金额中位数": 2,
                    "估算金额最高值": 3,
                    "采用理由": "",
                    "不确定性说明": "",
                }
            ],
            columns=query_estimate_llm.SUGGESTED_BILL_COLUMNS,
        )
        warnings: list[str] = []

        processed_bill, processed_stats = query_estimate_llm.postprocess_suggested_bill(bill, stats, warnings)

        self.assertEqual(processed_bill.loc[0, "综合单价最低值"], 80)
        self.assertEqual(processed_bill.loc[0, "估算金额中位数"], 900)
        self.assertEqual(processed_bill.loc[0, "其中包含机械费单价中位数"], "")
        self.assertEqual(processed_bill.loc[0, "来源样本"], "batch-a::2::2-1")
        self.assertEqual(processed_stats.loc[0, "是否被LLM采用"], "是")
        self.assertIn("llm_unit_price_overridden_by_candidate_stats", warnings)
        self.assertIn("llm_amount_overridden_by_program_calculation", warnings)

    def test_postprocess_prefers_family_and_drops_duplicate_family_id(self):
        stats = pd.DataFrame(
            [
                {
                    "candidate_id": "C001",
                    "fine_signature": "sig-a",
                    "family_signature": "fam",
                    "cost_item_name": "屋面卷材防水",
                    "project_description": "3mm SBS",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "历史样本数": 1,
                    "来源工程包数": 1,
                    "历史综合单价最低值": 80,
                    "历史综合单价中位数": 90,
                    "历史综合单价最高值": 100,
                    "source_refs": "batch-a::2::2-1",
                    "是否被LLM采用": "",
                },
                {
                    "candidate_id": "C002",
                    "fine_signature": "sig-b",
                    "family_signature": "fam",
                    "cost_item_name": "屋面卷材防水",
                    "project_description": "4mm SBS",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "历史样本数": 1,
                    "来源工程包数": 1,
                    "历史综合单价最低值": 110,
                    "历史综合单价中位数": 120,
                    "历史综合单价最高值": 130,
                    "source_refs": "batch-a::5::5-1",
                    "是否被LLM采用": "",
                },
            ],
            columns=query_estimate_llm.CANDIDATE_ITEM_STATS_COLUMNS,
        )
        families = pd.DataFrame(
            [
                {
                    "family_id": "F001",
                    "family_signature": "fam",
                    "representative_cost_item_name": "屋面卷材防水",
                    "representative_project_description": "3mm SBS",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "覆盖candidate_ids": "C001, C002",
                    "历史样本数": 2,
                    "来源工程包数": 2,
                    "历史综合单价最低值": 80,
                    "历史综合单价中位数": 100,
                    "历史综合单价最高值": 130,
                    "历史人工单价最低值": 20,
                    "历史人工单价中位数": 25,
                    "历史人工单价最高值": 30,
                    "历史机械单价中位数": None,
                    "source_refs": "batch-a::2::2-1, batch-a::5::5-1",
                }
            ],
            columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS,
        )
        bill = pd.DataFrame(
            [
                {
                    "序号": 1,
                    "family_id": "F001",
                    "candidate_id": "C002",
                    "推荐类型": "直接匹配项",
                    "项目角色": "核心施工项",
                    "计量/估算口径": "按面积估算",
                    "清单项名称": "",
                    "项目特征/施工工艺": "",
                    "单位": "",
                    "建议工程量": 10,
                    "工程量依据": "用户给定",
                    "综合单价最低值": 1,
                    "综合单价中位数": 2,
                    "综合单价最高值": 3,
                    "估算金额最低值": 1,
                    "估算金额中位数": 2,
                    "估算金额最高值": 3,
                    "采用理由": "",
                    "不确定性说明": "",
                },
                {
                    "序号": 2,
                    "family_id": "F001",
                    "candidate_id": "C001",
                    "推荐类型": "重复项",
                    "建议工程量": 10,
                },
            ],
            columns=query_estimate_llm.SUGGESTED_BILL_COLUMNS,
        )
        warnings: list[str] = []

        processed_bill, processed_stats = query_estimate_llm.postprocess_suggested_bill(bill, stats, families, warnings)

        self.assertEqual(len(processed_bill), 1)
        self.assertEqual(processed_bill.loc[0, "序号"], 1)
        self.assertEqual(processed_bill.loc[0, "family_id"], "F001")
        self.assertEqual(processed_bill.loc[0, "清单项名称"], "屋面卷材防水")
        self.assertEqual(processed_bill.loc[0, "综合单价中位数"], 100)
        self.assertEqual(processed_bill.loc[0, "估算金额中位数"], 1000)
        self.assertEqual(processed_bill.loc[0, "来源样本"], "batch-a::2::2-1, batch-a::5::5-1")
        self.assertEqual(processed_stats["是否被LLM采用"].tolist(), ["是", "是"])
        self.assertIn("duplicate_family_id_dropped", warnings)

    def test_write_query_result_workbook_has_new_eight_sheets(self):
        rewrite = query_estimate_llm.QueryRewrite("屋面", "屋面工程", "屋面防水", [], [], "", [], [], True)
        catalog = query_estimate_llm.QueryCatalog("CP-002-03", "屋面", "防水层", "维修", "共用部位", None, {}, True, [])
        result = query_estimate_llm.QueryResult(
            rewrite=rewrite,
            query_catalog=catalog,
            estimate_summary=pd.DataFrame([{"字段": "原始需求", "值": "屋面"}]),
            suggested_bill=pd.DataFrame(
                [
                    {
                        "序号": 1,
                        "family_id": "F001",
                        "candidate_id": "C001",
                        "推荐类型": "直接匹配项",
                        "项目角色": "核心施工项",
                        "计量/估算口径": "按面积估算",
                        "清单项名称": "屋面卷材防水",
                        "项目特征/施工工艺": "3mm SBS",
                        "单位": "m²",
                        "建议工程量": 500,
                        "来源样本": "batch-a::2::2-1",
                    }
                ],
                columns=query_estimate_llm.SUGGESTED_BILL_COLUMNS,
            ),
            matched_project_packages=pd.DataFrame(columns=query_estimate_llm.MATCHED_PROJECT_PACKAGE_COLUMNS),
            candidate_families=pd.DataFrame(columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS),
            candidate_item_stats=pd.DataFrame(columns=query_estimate_llm.CANDIDATE_ITEM_STATS_COLUMNS),
            evidence_items=pd.DataFrame(columns=query_estimate_llm.EVIDENCE_ITEM_COLUMNS),
            parse_info=pd.DataFrame([{"字段": "project_package_query_text", "值": "屋面工程"}]),
            llm_trace=pd.DataFrame(
                [
                    {"step": "query_rewrite_for_embedding"},
                    {"step": "query_catalog_classification"},
                    {"step": "suggested_bill_generation"},
                ],
                columns=query_estimate_llm.LLM_TRACE_COLUMNS,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "query_result.xlsx"
            query_estimate_llm.write_query_result_workbook(output_path, result)
            workbook = openpyxl.load_workbook(output_path, data_only=True)
            self.assertEqual(
                workbook.sheetnames,
                [
                    "estimate_summary",
                    "suggested_bill",
                    "matched_project_packages",
                    "candidate_families",
                    "candidate_item_stats",
                    "evidence_items",
                    "parse_info",
                    "llm_trace",
                ],
            )
            matched_headers = [
                workbook["matched_project_packages"].cell(row=1, column=column).value
                for column in range(1, workbook["matched_project_packages"].max_column + 1)
            ]
            trace_steps = [
                workbook["llm_trace"].cell(row=row, column=1).value
                for row in range(2, workbook["llm_trace"].max_row + 1)
            ]
            workbook.close()

        self.assertIn("cost_item_names_summary", matched_headers)
        self.assertNotIn("分类摘要", matched_headers)
        self.assertEqual(
            trace_steps,
            ["query_rewrite_for_embedding", "query_catalog_classification", "suggested_bill_generation"],
        )

    def test_query_validate_output_path_requires_overwrite_for_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "query.xlsx"
            output_path.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "输出已存在，请加 --overwrite"):
                query_estimate_llm.validate_output_path(output_path, overwrite=False)
            query_estimate_llm.validate_output_path(output_path, overwrite=True)

    def test_run_ingest_batch_parses_batch_id_from_filename(self):
        input_path = Path("excel_inputs/audit_ocr_export_20260630_001.xlsx")

        self.assertEqual(run_ingest_batch.infer_batch_id(input_path), "20260630_001")
        self.assertEqual(run_ingest_batch.batch_id_from_args(input_path, "manual_001"), "manual_001")

        with self.assertRaisesRegex(ValueError, "无法从文件名解析 batch_id"):
            run_ingest_batch.infer_batch_id(Path("excel_inputs/audit_ocr_export.xlsx"))

    def test_merge_samples_validate_output_paths_requires_overwrite_for_output_or_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_path = tmp_path / "samples" / "cost_item_samples_all.xlsx"
            report_path = merge_samples.dedup_report_path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "输出已存在，请加 --overwrite"):
                merge_samples.validate_output_paths(output_path, report_path, overwrite=False)
            merge_samples.validate_output_paths(output_path, report_path, overwrite=True)

    def test_legacy_query_entrypoint_removed_and_readme_points_to_official_script(self):
        legacy_name = "query_cost_" "item_estimate.py"
        self.assertFalse((ROOT / "scripts" / legacy_name).exists())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("scripts/query_cost_estimate_llm.py", readme)
        self.assertNotIn(f"scripts/{legacy_name}", readme)
        self.assertNotIn("project_name_embeddings.npy", readme)
        self.assertNotIn("project_detail_embeddings.npy", readme)
        self.assertNotIn("item_text_embeddings.npy", readme)


if __name__ == "__main__":
    unittest.main()
