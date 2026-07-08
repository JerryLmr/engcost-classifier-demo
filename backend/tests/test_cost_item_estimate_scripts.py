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
        self.assertEqual(row["fine_signature"], "屋面卷材防水 | 3.0mm弹性体改性沥青防水卷材 | m²")

    def test_fine_signature_normalizes_spacing_and_units(self):
        normalize = build_index.normalize_fine_signature_text

        self.assertEqual(normalize("外墙脚手架 高度13m以内"), normalize("外墙脚手架 高度 13m 以内"))
        self.assertEqual(normalize("外墙脚手架 高度20m以内"), normalize("外墙脚手架 高度 20m 以内"))
        self.assertNotEqual(normalize("外墙脚手架 高度13m以内"), normalize("外墙脚手架 高度20m以内"))
        self.assertEqual(normalize("㎡"), normalize("m2"))
        self.assertEqual(normalize("平方米"), normalize("m²"))

    def test_fine_signature_normalizes_waterproof_thickness(self):
        normalize = build_index.normalize_fine_signature_text

        self.assertEqual(normalize("立面聚合物水泥防水涂料 ~1.2mm厚"), normalize("立面聚合物水泥防水涂料 ~1.2mm 厚"))
        self.assertEqual(normalize("平面聚氨酯防水涂料 ~1.5mm厚"), normalize("平面聚氨酯防水涂料~1.5mm 厚"))
        self.assertEqual(normalize("厚 1.5 mm 防水涂料"), normalize("1.5毫米厚防水涂料"))
        self.assertNotEqual(normalize("平面聚氨酯防水涂料 ~1.5mm厚"), normalize("平面聚氨酯防水涂料 ~1.2mm厚"))
        self.assertNotEqual(normalize("3mm SBS"), normalize("4mm SBS"))
        self.assertNotEqual(normalize("自粘卷材"), normalize("热熔卷材"))
        self.assertNotEqual(normalize("一层防水"), normalize("两层防水"))

    def test_fine_signature_normalizes_sbs_waterproof_aliases(self):
        normalize = build_index.normalize_fine_signature_text
        expected = "3.0mm弹性体改性沥青防水卷材"

        equivalent_values = [
            "1.3.0mm厚弹性体改性沥青防水卷材",
            "1.3.0厚弹性体改性沥青防水卷材",
            "1.3.0mm弹性体改性沥青防水卷材",
            "3.0mm厚弹性体改性沥青防水卷材",
            "厚 3.0 mm 弹性体改性沥青防水卷材",
            "3.0mm弹性改性沥青防水卷材",
            "3.0mm SBS防水卷材",
            "3.0mm SBS改性沥青防水卷材",
            "3.0mm SBS弹性体改性沥青防水卷材",
            "卷材品种、规格、厚度：3.0mm SBS防水卷材",
        ]

        for value in equivalent_values:
            with self.subTest(value=value):
                self.assertEqual(normalize(value), expected)

    def test_fine_signature_keeps_price_sensitive_waterproof_differences(self):
        normalize = build_index.normalize_fine_signature_text
        base = normalize("3.0mm SBS防水卷材")

        different_values = [
            "4.0mm SBS防水卷材",
            "3.0mm自粘SBS防水卷材",
            "3.0mm热熔SBS防水卷材",
            "3.0mm单层SBS防水卷材",
            "3.0mm双层SBS防水卷材",
            "3.0mm一道SBS防水卷材",
            "3.0mm两道SBS防水卷材",
            "3.0mm耐根穿刺SBS防水卷材",
            "3.0mm SBS防水卷材附加层",
            "3.0mm SBS防水卷材含基层处理",
            "拆除3.0mm SBS防水卷材",
            "新做3.0mm SBS防水卷材",
            "平面3.0mm SBS防水卷材",
            "立面3.0mm SBS防水卷材",
            "砂面3.0mm SBS防水卷材",
        ]

        for value in different_values:
            with self.subTest(value=value):
                self.assertNotEqual(normalize(value), base)

    def test_build_fine_signature_normalizes_sbs_waterproof_aliases(self):
        row = pd.Series(
            {
                "cost_item_name": "屋面卷材防水",
                "project_description": "卷材品种、规格、厚度：1.3.0mm厚SBS防水卷材",
                "unit": "平方米",
                "unit_normalized": "m²",
            }
        )

        self.assertEqual(
            build_index.build_fine_signature(row),
            "屋面卷材防水 | 3.0mm弹性体改性沥青防水卷材 | m²",
        )

    def test_fine_signature_normalizes_tilde_between_chinese(self):
        normalize = build_index.normalize_fine_signature_text

        self.assertEqual(normalize("抹灰面铲除 抹灰面 只拆除面层时"), normalize("抹灰面铲除 抹灰面~只拆除面层时"))

    def test_fine_signature_normalizes_thickness_order_without_merging_meaning(self):
        normalize = build_index.normalize_fine_signature_text

        self.assertEqual(normalize("厚 1.5（mm）聚氨酯防水涂料"), normalize("1.5mm厚聚氨酯防水涂料"))
        self.assertEqual(normalize("1.1.5mm厚聚氨酯防水涂料"), normalize("1.5mm聚氨酯防水涂料"))
        self.assertEqual(normalize("m^{2}"), "m²")
        self.assertEqual(normalize("原有面层铲除及垃圾外运"), normalize("原面层拆除及垃圾清运"))
        self.assertNotEqual(normalize("屋面卷材防水 1.5mm厚"), normalize("墙面卷材防水 1.5mm厚"))

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
        self.assertEqual(args.family_selection_limit, 50)
        self.assertEqual(args.family_exploration_limit, 5)
        self.assertFalse(hasattr(args, "family_final_score_limit"))
        self.assertFalse(hasattr(args, "family_item_score_limit"))
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
        self.assertEqual(query_estimate_llm.catalog_score(changed, catalog), 0.95)

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
        self.assertAlmostEqual(float(roof["package_path_score"]), 0.8)
        self.assertAlmostEqual(float(roof["item_path_score"]), 0.915)
        self.assertAlmostEqual(float(roof["final_score"]), 0.915)
        self.assertEqual(roof["source_ref"], "batch-a::2::2-1")
        self.assertEqual(pipe["package_score"], 0.0)
        self.assertEqual(pipe["direct_hit"], True)
        self.assertEqual(pipe["catalog_score"], 0.1625)
        self.assertAlmostEqual(float(pipe["item_path_score"]), 0.5725)
        self.assertAlmostEqual(float(pipe["final_score"]), float(pipe["item_path_score"]))

    def test_candidate_families_group_by_fine_signature_only(self):
        candidates = self.prepared_samples().head(1).copy()
        same = candidates.iloc[0].copy()
        same["project_description"] = "3.0mm   SBS 沥青防水卷材"
        same["fine_signature"] = build_index.build_fine_signature(same)
        same["quantity"] = 120
        same["unit_price"] = 90
        same["total_price"] = 10800
        same["labor_unit_price"] = 22
        same["machinery_unit_price"] = 6
        same["project_key"] = "batch-a::5"
        same["project_package_id"] = "batch-a::5"
        same["item_row_id"] = "5-1"
        different = candidates.iloc[0].copy()
        different["project_description"] = "4.0mm SBS 沥青防水卷材"
        different["fine_signature"] = build_index.build_fine_signature(different)
        different["quantity"] = 300
        different["unit_price"] = 120
        different["total_price"] = 36000
        different["labor_unit_price"] = 30
        different["machinery_unit_price"] = 7
        different["project_key"] = "batch-a::6"
        different["project_package_id"] = "batch-a::6"
        different["item_row_id"] = "6-1"
        candidates = pd.concat([candidates, same.to_frame().T, different.to_frame().T], ignore_index=True)
        candidates["package_score"] = [0.8, 0.7, 0.6]
        candidates["package_rank"] = [1, 2, 3]
        candidates["item_score"] = [0.9, 0.8, 0.7]
        candidates["cooccur_score"] = 1.0
        candidates["catalog_score"] = 1.0
        candidates["unit_score"] = 0.5
        candidates["package_path_score"] = [0.8, 0.7, 0.6]
        candidates["item_path_score"] = [0.9, 0.85, 0.75]
        candidates["final_score"] = [0.85, 0.75, 0.65]
        candidates["source_ref"] = ["batch-a::2::2-1", "batch-a::5::5-1", "batch-a::6::6-1"]

        families = query_estimate_llm.build_candidate_families(candidates)

        self.assertEqual(families.columns.tolist(), query_estimate_llm.CANDIDATE_FAMILY_COLUMNS)
        self.assertEqual(len(families), 2)
        roof3 = families[families["representative_project_description"].str.contains("3.0mm")].iloc[0]
        roof4 = families[families["representative_project_description"].str.contains("4.0mm")].iloc[0]
        self.assertEqual(roof3["fine_signature"], build_index.build_fine_signature(candidates.iloc[0]))
        self.assertEqual(roof3["历史样本数"], 2)
        self.assertEqual(roof3["来源工程包数"], 2)
        self.assertEqual(roof3["历史综合单价最低值"], 80.0)
        self.assertEqual(roof3["历史综合单价中位数"], 85.0)
        self.assertEqual(roof3["历史综合单价最高值"], 90.0)
        self.assertEqual(roof3["历史人工单价中位数"], 21.0)
        self.assertEqual(roof3["历史机械单价中位数"], 5.5)
        self.assertEqual(roof3["package_path_score最大值"], 0.8)
        self.assertEqual(roof3["item_path_score最大值"], 0.9)
        self.assertEqual(roof3["representative_project_description"], "3.0mm SBS 沥青防水卷材")
        self.assertEqual(roof4["历史综合单价最低值"], 120.0)

        evidence = query_estimate_llm.build_evidence_items(candidates, families)
        family_by_signature = dict(zip(families["fine_signature"], families["family_id"], strict=False))
        self.assertEqual(len(evidence), len(candidates))
        self.assertIn("family_id", evidence.columns)
        self.assertIn("fine_signature", evidence.columns)
        self.assertFalse(evidence["family_id"].map(query_estimate_llm.cell_text).eq("").any())
        for _index, row in evidence.iterrows():
            self.assertEqual(row["family_id"], family_by_signature[row["fine_signature"]])

    def test_select_families_for_llm_refills_overlapping_rankings(self):
        families = pd.DataFrame(
            [
                {
                    "family_id": f"F{index:03d}",
                    "final_score": 1.0 - index / 100.0,
                    "item_score最大值": 1.0 - index / 100.0 if index <= 30 else 0.01,
                    "cooccur_score": 1.0 - index / 200.0,
                    "来源工程包数": 1,
                    "历史样本数": 1,
                }
                for index in range(1, 61)
            ]
        )

        selected = query_estimate_llm.select_families_for_llm(
            families,
            family_selection_limit=50,
            exploration_limit=5,
        )

        selected_ids = selected["family_id"].tolist()
        self.assertEqual(len(selected_ids), 50)
        self.assertEqual(len(selected_ids), len(set(selected_ids)))
        self.assertIn("F001", selected_ids)
        self.assertIn("F060", selected_ids)
        self.assertEqual(selected["selection_rank"].tolist(), list(range(1, 51)))
        self.assertTrue(selected["candidate_source"].map(bool).all())

    def test_select_families_for_llm_returns_all_when_under_limit(self):
        families = pd.DataFrame(
            [
                {
                    "family_id": f"F{index:03d}",
                    "final_score": 1.0 - index / 100.0,
                    "item_score最大值": 0.5,
                    "cooccur_score": 0.2,
                    "来源工程包数": 1,
                    "历史样本数": 1,
                }
                for index in range(1, 19)
            ]
        )

        selected = query_estimate_llm.select_families_for_llm(families, family_selection_limit=50, exploration_limit=5)

        self.assertEqual(len(selected), 18)
        self.assertEqual(len(selected["family_id"].tolist()), len(set(selected["family_id"].tolist())))

    def test_select_families_for_llm_adds_deterministic_exploration(self):
        families = pd.DataFrame(
            [
                {
                    "family_id": f"F{index:03d}",
                    "final_score": 1.0 - index / 1000.0,
                    "item_score最大值": 1.0 - index / 1000.0,
                    "cooccur_score": 1.0 - index / 1000.0,
                    "来源工程包数": 1,
                    "历史样本数": 1,
                }
                for index in range(1, 66)
            ]
        )

        first = query_estimate_llm.select_families_for_llm(families, family_selection_limit=50, exploration_limit=5)
        second = query_estimate_llm.select_families_for_llm(families, family_selection_limit=50, exploration_limit=5)
        exploration = first[first["candidate_source"].eq("exploration")]

        self.assertEqual(first["family_id"].tolist(), second["family_id"].tolist())
        self.assertEqual(len(exploration), 5)
        self.assertEqual(len(exploration["family_id"].tolist()), len(set(exploration["family_id"].tolist())))
        self.assertTrue(set(exploration["family_id"]).issubset({f"F{index:03d}" for index in range(46, 66)}))

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

    def test_family_selection_prompt_sends_only_allowed_fields(self):
        rewrite = query_estimate_llm.QueryRewrite("屋面漏水", "屋面漏水", "屋面防水", [], [], "屋面", [], [], True)
        catalog = query_estimate_llm.QueryCatalog("CP-002-03", "屋面", "防水层", "维修", "共用部位", None, {}, True, [])
        families = pd.DataFrame(
            [
                {
                    "family_id": "F001",
                    "representative_cost_item_name": "屋面卷材防水",
                    "representative_project_description": "3mm SBS",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "历史样本数": 2,
                    "来源工程包数": 2,
                    "final_score": 0.9,
                    "source_refs": "SHOULD_NOT_SEND",
                    "历史综合单价中位数": 90,
                }
            ]
        )

        prompt, records = query_estimate_llm.build_family_selection_prompt(rewrite, catalog, families)

        self.assertEqual(records[0]["id"], "F001")
        self.assertEqual(records[0]["samples"], 2)
        self.assertEqual(records[0]["packages"], 2)
        self.assertIn("item_score", records[0])
        self.assertIn("cooccur_score", records[0])
        self.assertNotIn("final_score", records[0])
        self.assertNotIn("source_refs", prompt)
        self.assertNotIn("final_score", prompt)
        self.assertNotIn("package_path_score", prompt)
        self.assertNotIn("item_path_score", prompt)
        self.assertNotIn("历史综合单价", prompt)
        self.assertIn("selected_families", prompt)

    def test_family_selection_validation_rejects_invalid_rows(self):
        result = {
            "selected_families": [
                {"family_id": "F001", "item_type": "核心施工项", "selection_reason": "直接对应", "cost_item_name": "污染"},
                {"family_id": "BAD", "item_type": "核心施工项", "selection_reason": "无效"},
                {"family_id": "F001", "item_type": "核心施工项", "selection_reason": "重复"},
                {"family_id": "F002", "item_type": "补充候选项", "selection_reason": "非法类型"},
                {"family_id": "F003", "item_type": "施工措施/现场条件项", "selection_reason": "新类型"},
                {"family_id": "F004", "item_type": "措施/条件项", "selection_reason": "旧类型非法"},
            ]
        }
        warnings: list[str] = []

        selected, meta = query_estimate_llm.parse_family_selection_result(result, {"F001", "F002", "F003", "F004"}, warnings)

        self.assertEqual(selected["family_id"].tolist(), ["F001", "F003"])
        self.assertEqual(meta["invalid_family_ids"], ["BAD"])
        self.assertEqual(meta["duplicate_family_ids"], ["F001"])
        self.assertEqual(meta["invalid_item_types"], ["补充候选项", "措施/条件项"])
        self.assertIn("invalid_family_ids", warnings)

    def test_family_selection_trace_records_sent_and_selected_families(self):
        families = pd.DataFrame(
            [
                {
                    "selection_rank": 1,
                    "candidate_source": "final_score,item_score",
                    "family_id": "F001",
                    "fine_signature": "sig-1",
                    "representative_cost_item_name": "屋面卷材防水",
                    "representative_project_description": "3mm SBS",
                    "unit": "平方米",
                    "unit_normalized": "m²",
                    "历史样本数": 2,
                    "来源工程包数": 2,
                    "final_score": 0.9,
                    "item_score最大值": 0.8,
                    "cooccur_score": 0.7,
                },
                {
                    "selection_rank": 2,
                    "candidate_source": "exploration",
                    "family_id": "F002",
                    "fine_signature": "sig-2",
                    "representative_cost_item_name": "脚手架",
                    "representative_project_description": "现场措施",
                    "unit": "项",
                    "unit_normalized": "项",
                    "历史样本数": 1,
                    "来源工程包数": 1,
                    "final_score": 0.2,
                    "item_score最大值": 0.1,
                    "cooccur_score": 0.3,
                },
            ]
        )
        selected = pd.DataFrame(
            [{"family_id": "F001", "item_type": "核心施工项", "selection_reason": "直接对应"}]
        )

        trace = query_estimate_llm.build_family_selection_trace_frame(families)
        trace = query_estimate_llm.apply_family_selection_trace_result(trace, selected, "llm")

        self.assertEqual(trace["selection_rank"].tolist(), [1, 2])
        self.assertEqual(trace["selected_by_llm"].tolist(), ["是", "否"])
        self.assertEqual(trace.loc[0, "item_type"], "核心施工项")
        self.assertEqual(trace.loc[0, "selection_reason"], "直接对应")
        self.assertEqual(trace.loc[1, "selection_source"], "not_selected")

    def test_dedup_selection_suppresses_display_duplicates_without_merging_prices(self):
        families = pd.DataFrame(
            [
                {
                    "family_id": "F001",
                    "fine_signature": "sig-1",
                    "representative_cost_item_name": "屋面卷材防水",
                    "representative_project_description": "3.0mm SBS",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "历史样本数": 1,
                    "来源工程包数": 1,
                    "历史综合单价最低值": 80,
                    "历史综合单价中位数": 80,
                    "历史综合单价最高值": 80,
                    "final_score": 0.95,
                    "source_refs": "batch-a::1::1-1",
                },
                {
                    "family_id": "F002",
                    "fine_signature": "sig-2",
                    "representative_cost_item_name": "屋面卷材防水",
                    "representative_project_description": "3.0mm SBS",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "历史样本数": 3,
                    "来源工程包数": 2,
                    "历史综合单价最低值": 120,
                    "历史综合单价中位数": 120,
                    "历史综合单价最高值": 120,
                    "final_score": 0.9,
                    "source_refs": "batch-a::2::2-1",
                },
            ],
            columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS,
        )
        selected = pd.DataFrame(
            [
                {"family_id": "F001", "item_type": "核心施工项", "selection_reason": "重复项"},
                {"family_id": "F002", "item_type": "核心施工项", "selection_reason": "代表项"},
            ]
        )

        deduped, suppressed_by = query_estimate_llm.deterministic_dedup_selection(selected, families)
        decisions = pd.DataFrame(
            [
                {
                    "family_id": "F002",
                    "quantity_source": "用户明确给定",
                    "suggested_quantity_low": 1,
                    "suggested_quantity_mid": 1,
                    "suggested_quantity_high": 1,
                    "include_in_amount": True,
                    "quantity_reason": "用户给定",
                    "confirmation_note": "",
                }
            ]
        )
        bill = query_estimate_llm.build_final_suggested_bill(deduped, decisions, families)

        self.assertEqual(suppressed_by, {"F001": "F002"})
        self.assertEqual(deduped["family_id"].tolist(), ["F002"])
        self.assertEqual(bill["family_id"].tolist(), ["F002"])
        self.assertEqual(bill.loc[0, "综合单价中位数"], 120)
        self.assertEqual(bill.loc[0, "来源样本"], "batch-a::2::2-1")

    def test_dedup_selection_validation_rejects_cycles_and_invalid_ids(self):
        with self.assertRaisesRegex(ValueError, "形成环"):
            query_estimate_llm.parse_dedup_selection_result(
                {
                    "keep": [],
                    "suppress": [
                        {"family_id": "F001", "representative_id": "F002", "reason": "重复"},
                        {"family_id": "F002", "representative_id": "F001", "reason": "重复"},
                    ],
                },
                {"F001", "F002"},
            )
        with self.assertRaisesRegex(ValueError, "非法 family_id"):
            query_estimate_llm.parse_dedup_selection_result({"keep": ["BAD"], "suppress": []}, {"F001"})

        keep_ids, suppress_map, suppress_detail, keep_reasons = query_estimate_llm.parse_dedup_selection_result(
            {
                "keep": ["F002"],
                "suppress": [{"family_id": "F001", "representative_id": "F002", "reason": "F002覆盖范围更完整"}],
                "keep_reasons": [{"family_id": "F002", "reason": "保留代表项"}],
            },
            {"F001", "F002"},
        )

        self.assertEqual(keep_ids, {"F002"})
        self.assertEqual(suppress_map, {"F001": "F002"})
        self.assertEqual(suppress_detail, [{"family_id": "F001", "representative_id": "F002", "reason": "F002覆盖范围更完整"}])
        self.assertEqual(keep_reasons, [{"family_id": "F002", "reason": "保留代表项"}])

    def test_quantity_decision_validation_and_missing_defaults(self):
        selected = pd.DataFrame(
            [
                {"family_id": "F001", "item_type": "核心施工项", "selection_reason": ""},
                {"family_id": "F002", "item_type": "可选/替代工艺", "selection_reason": ""},
                {"family_id": "F003", "item_type": "施工措施/现场条件项", "selection_reason": ""},
            ]
        )
        result = {
            "family_quantities": [
                {
                    "family_id": "F001",
                    "quantity_source": "用户明确给定",
                    "suggested_quantity_low": 100,
                    "suggested_quantity_mid": 90,
                    "suggested_quantity_high": 120,
                    "include_in_amount": True,
                },
                {
                    "family_id": "F002",
                    "quantity_source": "历史样本估算",
                    "suggested_quantity_low": 10,
                    "suggested_quantity_mid": 20,
                    "suggested_quantity_high": 30,
                    "include_in_amount": True,
                },
                {"family_id": "BAD", "quantity_source": "用户明确给定"},
                {"family_id": "F003", "quantity_source": "非法来源"},
            ]
        }
        warnings: list[str] = []

        decisions, meta = query_estimate_llm.parse_quantity_decision_result(result, selected, warnings)

        f001 = decisions[decisions["family_id"] == "F001"].iloc[0]
        f002 = decisions[decisions["family_id"] == "F002"].iloc[0]
        f003 = decisions[decisions["family_id"] == "F003"].iloc[0]
        self.assertTrue(pd.isna(f001["suggested_quantity_low"]))
        self.assertFalse(f001["include_in_amount"])
        self.assertFalse(f002["include_in_amount"])
        self.assertEqual(f003["quantity_source"], "需现场确认")
        self.assertEqual(f003["confirmation_note"], "工程量及计价范围需确认")
        self.assertEqual(meta["invalid_family_ids"], ["BAD"])
        self.assertEqual(meta["invalid_quantity_sources"], ["非法来源"])
        self.assertEqual(meta["invalid_quantity_ranges"], ["F001"])

    def test_final_suggested_bill_backfills_prices_and_calculates_amounts(self):
        families = pd.DataFrame(
            [
                {
                    "family_id": "F001",
                    "fine_signature": "sig",
                    "representative_cost_item_name": "屋面卷材防水",
                    "representative_project_description": "3mm SBS",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "历史样本数": 2,
                    "来源工程包数": 2,
                    "历史综合单价最低值": 80,
                    "历史综合单价中位数": 100,
                    "历史综合单价最高值": 130,
                    "历史人工单价最低值": 20,
                    "历史人工单价中位数": 25,
                    "历史人工单价最高值": 30,
                    "历史机械单价最低值": 5,
                    "历史机械单价中位数": None,
                    "历史机械单价最高值": 7,
                    "source_refs": "batch-a::2::2-1, batch-a::5::5-1",
                }
            ],
            columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS,
        )
        selected = pd.DataFrame(
            [
                {
                    "family_id": "F001",
                    "item_type": "核心施工项",
                    "selection_reason": "直接解决屋面防水需求",
                }
            ]
        )
        decisions = pd.DataFrame(
            [
                {
                    "family_id": "F001",
                    "quantity_source": "用户明确给定",
                    "suggested_quantity_low": 10,
                    "suggested_quantity_mid": 20,
                    "suggested_quantity_high": 30,
                    "include_in_amount": True,
                    "quantity_reason": "用户给定",
                    "confirmation_note": "确认边界",
                }
            ]
        )

        bill = query_estimate_llm.build_final_suggested_bill(selected, decisions, families)

        self.assertEqual(bill.columns.tolist(), query_estimate_llm.SUGGESTED_BILL_COLUMNS)
        self.assertNotIn("candidate_id", bill.columns)
        self.assertEqual(bill.loc[0, "清单项名称"], "屋面卷材防水")
        self.assertEqual(bill.loc[0, "综合单价中位数"], 100)
        self.assertEqual(bill.loc[0, "估算金额最低值"], 800)
        self.assertEqual(bill.loc[0, "估算金额中位数"], 2000)
        self.assertEqual(bill.loc[0, "估算金额最高值"], 3900)
        self.assertEqual(bill.loc[0, "估算金额中包含人工费最低值"], 200)
        self.assertEqual(bill.loc[0, "估算金额中包含人工费中位数"], 500)
        self.assertEqual(bill.loc[0, "估算金额中包含人工费最高值"], 900)
        self.assertEqual(bill.loc[0, "估算金额中包含机械费最低值"], 50)
        self.assertEqual(bill.loc[0, "估算金额中包含机械费最高值"], 210)
        self.assertIn("F001", bill.loc[0, "金额计算口径"])
        self.assertEqual(bill.loc[0, "来源样本"], "batch-a::2::2-1, batch-a::5::5-1")

    def test_estimate_summary_sums_only_included_amount_rows(self):
        rewrite = query_estimate_llm.QueryRewrite(
            raw_query="维修",
            project_package_query_text="维修",
            item_query_text="维修",
            parsed_quantities=[],
            materials_or_specs=[],
            repair_object="屋面",
            uncertainties=[],
            notes=[],
            success=True,
        )
        catalog = query_estimate_llm.QueryCatalog("CP-002-03", "屋面", "防水层", "维修", "共用部位", None, {}, True, [])
        suggested_bill = pd.DataFrame(
            [
                {
                    "序号": 1,
                    "family_id": "F001",
                    "建议项类型": "核心施工项",
                    "清单项名称": "核心做法",
                    "项目特征/施工工艺": "3mm SBS",
                    "工程量来源": "用户明确给定",
                    "建议工程量中位数": 10,
                    "是否计入参考金额区间": "是",
                    "估算金额最低值": 100,
                    "估算金额中位数": 100,
                    "估算金额最高值": 100,
                    "需确认事项": "核心范围需确认",
                },
                {
                    "序号": 2,
                    "family_id": "F002",
                    "建议项类型": "可选/替代工艺",
                    "清单项名称": "替代做法",
                    "项目特征/施工工艺": "4mm SBS",
                    "工程量来源": "用户明确给定",
                    "建议工程量中位数": 10,
                    "是否计入参考金额区间": "否",
                    "估算金额最低值": 1000,
                    "估算金额中位数": 1000,
                    "估算金额最高值": 1000,
                    "需确认事项": "可选/替代工艺需确认",
                },
            ],
            columns=query_estimate_llm.SUGGESTED_BILL_COLUMNS,
        )

        summary = query_estimate_llm.build_estimate_summary(rewrite, catalog, suggested_bill)
        values = dict(summary.values.tolist())

        self.assertEqual(summary["字段"].tolist(), ["需求理解", "匹配分类", "建议方案概览", "计入金额项目", "参考金额区间", "需现场确认"])
        self.assertIn("catalog_id=CP-002-03", values["匹配分类"])
        self.assertIn("核心施工项：核心做法（3mm SBS）", values["建议方案概览"])
        self.assertIn("可选/替代工艺：替代做法（4mm SBS）", values["建议方案概览"])
        self.assertIn("核心做法", values["计入金额项目"])
        self.assertNotIn("替代做法", values["计入金额项目"])
        self.assertIn("100.00 - 100.00", values["参考金额区间"])
        self.assertNotIn("1,100.00", values["参考金额区间"])
        self.assertIn("可选/替代工艺", values["需现场确认"])

    def test_build_parse_info_includes_llm_metrics(self):
        rewrite = query_estimate_llm.QueryRewrite("屋面", "屋面工程", "屋面防水", [], [], "", [], [], True)
        catalog = query_estimate_llm.QueryCatalog("CP-002-03", "屋面", "防水层", "维修", "共用部位", None, {}, True, [])

        parse_info = query_estimate_llm.build_parse_info(
            rewrite=rewrite,
            query_catalog=catalog,
            top_packages=10,
            top_items=20,
            max_packages_per_cache_subject=3,
            meta={},
            sample_count=100,
            package_count=10,
            candidate_pool_row_count=50,
            candidate_family_count=12,
            family_selection_input_count=12,
            family_selection_selected_count=2,
            family_selection_trace={"prompt_chars": 100, "prompt_tokens": 40, "completion_tokens": 8},
            family_selection_fallback=False,
            family_selection_error="",
            family_selection_meta={
                "invalid_family_ids": ["BAD"],
                "duplicate_family_ids": [],
                "invalid_item_types": [],
                "candidate_ids": ["F001", "F002"],
                "selected_ids": ["F001"],
                "selected_detail": [{"family_id": "F001", "item_type": "核心施工项", "selection_reason": "直接"}],
                "candidate_source_counts": {"final_score": 1, "exploration": 1},
                "exploration_count": 1,
            },
            dedup_selection_input_count=2,
            dedup_selection_output_count=1,
            dedup_selection_trace={"prompt_chars": 50, "estimated_tokens": 20, "completion_tokens": 4},
            dedup_selection_fallback=False,
            dedup_selection_error="",
            dedup_selection_meta={
                "auto_suppressed": ["F001->F002"],
                "llm_suppressed": [],
                "input_ids": ["F001", "F002"],
                "keep_ids": ["F002"],
                "suppress_detail": [{"family_id": "F001", "representative_id": "F002", "reason": "重复"}],
                "keep_reasons": [{"family_id": "F002", "reason": "代表项"}],
            },
            quantity_decision_input_count=2,
            quantity_relation_count=3,
            quantity_decision_trace={"prompt_chars": 200, "estimated_tokens": 100, "completion_tokens": ""},
            quantity_decision_fallback=True,
            quantity_decision_error="down",
            quantity_decision_meta={"invalid_family_ids": [], "duplicate_family_ids": [], "invalid_quantity_sources": ["bad"], "invalid_quantity_ranges": ["F001"]},
            output_path=None,
            started_at=query_estimate_llm.datetime.now(),
            index_dir=Path("query_index"),
            include_debug_text=False,
            family_selection_prompt="family prompt",
            dedup_selection_prompt="dedup prompt",
            quantity_decision_prompt="quantity prompt",
            warnings=[],
        )
        values = dict(parse_info.values.tolist())

        self.assertEqual(values["candidate_pool_row_count"], 50)
        self.assertEqual(values["candidate_family_count"], 12)
        self.assertEqual(values["family_selection_prompt_tokens"], 40)
        self.assertEqual(values["family_selection_candidate_ids"], '["F001", "F002"]')
        self.assertEqual(values["family_selection_selected_ids"], '["F001"]')
        self.assertEqual(values["family_selection_exploration_count"], 1)
        self.assertEqual(values["dedup_selection_prompt_tokens"], 20)
        self.assertEqual(values["dedup_input_ids"], '["F001", "F002"]')
        self.assertEqual(values["dedup_keep_ids"], '["F002"]')
        self.assertEqual(values["dedup_auto_suppressed"], "F001->F002")
        self.assertEqual(values["quantity_decision_prompt_tokens"], 100)
        self.assertEqual(values["invalid_family_ids"], "BAD")
        self.assertEqual(values["invalid_quantity_sources"], "bad")
        self.assertEqual(values["是否 quantity_decision fallback"], "是")

    def test_write_query_result_workbook_has_new_seven_sheets(self):
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
                        "建议项类型": "核心施工项",
                        "清单项名称": "屋面卷材防水",
                        "项目特征/施工工艺": "3mm SBS",
                        "单位": "m²",
                        "工程量来源": "用户明确给定",
                        "建议工程量中位数": 500,
                        "工程量依据": "按用户明确面积作为工程量",
                        "来源样本": "batch-a::2::2-1",
                    }
                ],
                columns=query_estimate_llm.SUGGESTED_BILL_COLUMNS,
            ),
            matched_project_packages=pd.DataFrame(columns=query_estimate_llm.MATCHED_PROJECT_PACKAGE_COLUMNS),
            candidate_families=pd.DataFrame(columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS),
            family_selection_trace=pd.DataFrame(columns=query_estimate_llm.FAMILY_SELECTION_TRACE_COLUMNS),
            evidence_items=pd.DataFrame(columns=query_estimate_llm.EVIDENCE_ITEM_COLUMNS),
            parse_info=pd.DataFrame([{"字段": "project_package_query_text", "值": "屋面工程"}]),
            llm_trace=pd.DataFrame(
                [
                    {"step": "query_rewrite_for_embedding"},
                    {"step": "query_catalog_classification"},
                    {"step": "family_selection"},
                    {"step": "dedup_selection"},
                    {"step": "quantity_decision"},
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
                    "family_selection_trace",
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
            ["query_rewrite_for_embedding", "query_catalog_classification", "family_selection", "dedup_selection", "quantity_decision"],
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
