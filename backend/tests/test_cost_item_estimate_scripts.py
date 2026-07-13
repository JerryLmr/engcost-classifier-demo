from __future__ import annotations

import csv
import importlib.util
import json
import re
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


class MergeCostItemSampleBatchesTestCase(unittest.TestCase):
    SAMPLE_HEADERS = [
        "source_row_id",
        "file_name",
        "consultation_project_name",
        "sub_project_id",
        "seq",
        "project_name",
        "project_description",
        "unit",
        "quantity",
    ]

    def write_batch(self, input_dir: Path, batch_id: str, rows: list[dict[str, object]]) -> None:
        batch_dir = input_dir / batch_id
        batch_dir.mkdir(parents=True)
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "samples"
        sheet.append(self.SAMPLE_HEADERS)
        for row in rows:
            sheet.append([row.get(header) for header in self.SAMPLE_HEADERS])
        workbook.save(batch_dir / merge_samples.BATCH_SAMPLE_NAME)

    def sample_row(
        self,
        source_row_id: int,
        file_name: str,
        sub_project_id: str,
        project_name: str,
        *,
        consultation_project_name: str = "咨询项目",
        seq: int = 1,
    ) -> dict[str, object]:
        return {
            "source_row_id": source_row_id,
            "file_name": file_name,
            "consultation_project_name": consultation_project_name,
            "sub_project_id": sub_project_id,
            "seq": seq,
            "project_name": project_name,
            "project_description": f"{project_name}特征",
            "unit": "项",
            "quantity": 1,
        }

    def test_merge_keeps_latest_project_groups_then_latest_stable_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "samples"
            output_path = tmp_path / "cost_item_samples_all.xlsx"
            report_path = merge_samples.dedup_report_path(output_path)

            old_blank_duplicate = self.sample_row(6, "", "空键子项目", "空键重复")
            latest_blank_duplicate = dict(old_blank_duplicate, source_row_id=16)
            same_batch_duplicate = self.sample_row(17, "同批次文件", "同批次子项目", "同批次重复")

            self.write_batch(
                input_dir,
                "20260618_001",
                [
                    self.sample_row(1, "楼幢项目", "7幢", "7幢旧数据"),
                    self.sample_row(2, "楼幢项目", "8幢", "8幢旧数据"),
                    self.sample_row(3, "楼幢项目", "9幢", "9幢旧数据一"),
                    self.sample_row(4, "楼幢项目", "9幢", "9幢旧数据二", seq=2),
                    self.sample_row(5, "另一个文件", "9幢", "不同文件保留"),
                    old_blank_duplicate,
                ],
            )
            self.write_batch(
                input_dir,
                "20260630_001",
                [
                    self.sample_row(10, "楼幢项目", "9幢", "9幢中间批次"),
                ],
            )
            self.write_batch(
                input_dir,
                "20260701_001",
                [
                    self.sample_row(15, "楼幢项目", "9幢", "9幢最新完整数据"),
                    latest_blank_duplicate,
                    same_batch_duplicate,
                    dict(same_batch_duplicate, source_row_id=18),
                ],
            )

            stats = merge_samples.merge_batches(input_dir, output_path, report_path)

            self.assertEqual(stats, (11, 6, 2, 3, 2))

            workbook = openpyxl.load_workbook(output_path, read_only=True, data_only=True)
            sheet = workbook["samples"]
            headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
            output_rows = [dict(zip(headers, values)) for values in sheet.iter_rows(min_row=2, values_only=True)]
            workbook.close()

            self.assertEqual(
                headers,
                [*self.SAMPLE_HEADERS, "batch_id", "project_key", "stable_sample_id"],
            )
            output_names = [row["project_name"] for row in output_rows]
            self.assertEqual(
                output_names,
                [
                    "7幢旧数据",
                    "8幢旧数据",
                    "不同文件保留",
                    "9幢最新完整数据",
                    "空键重复",
                    "同批次重复",
                ],
            )
            self.assertNotIn("9幢旧数据一", output_names)
            self.assertNotIn("9幢旧数据二", output_names)
            self.assertNotIn("9幢中间批次", output_names)

            blank_row = next(row for row in output_rows if row["project_name"] == "空键重复")
            tie_row = next(row for row in output_rows if row["project_name"] == "同批次重复")
            self.assertEqual(blank_row["batch_id"], "20260701_001")
            self.assertEqual(blank_row["project_key"], "20260701_001::16")
            self.assertEqual(tie_row["project_key"], "20260701_001::17")
            for output_row in output_rows:
                source_row = {header: output_row[header] for header in self.SAMPLE_HEADERS}
                self.assertEqual(output_row["stable_sample_id"], merge_samples.stable_sample_id(source_row))
                self.assertEqual(
                    output_row["project_key"],
                    merge_samples.build_project_key(output_row["batch_id"], output_row["source_row_id"]),
                )

            with report_path.open("r", encoding="utf-8-sig", newline="") as report_file:
                report_rows = list(csv.DictReader(report_file))

            self.assertEqual(list(report_rows[0]), merge_samples.DEDUP_REPORT_HEADERS)
            group_reports = [row for row in report_rows if row["dedup_type"] == "project_group_replaced"]
            stable_reports = [row for row in report_rows if row["dedup_type"] == "stable_sample_duplicate"]
            self.assertEqual(len(group_reports), 2)
            self.assertEqual(
                {(row["duplicate_batch_id"], row["duplicate_row_count"]) for row in group_reports},
                {("20260618_001", "2"), ("20260630_001", "1")},
            )
            self.assertTrue(all(row["kept_batch_id"] == "20260701_001" for row in group_reports))
            self.assertTrue(all(row["dedup_key"] == "楼幢项目::9幢" for row in group_reports))
            self.assertTrue(all(row["stable_sample_id"] == "" for row in group_reports))
            self.assertTrue(all(row["cost_item_name"] == "" for row in group_reports))

            self.assertEqual(len(stable_reports), 2)
            self.assertTrue(all(row["duplicate_row_count"] == "1" for row in stable_reports))
            self.assertTrue(all(row["kept_batch_id"] == "20260701_001" for row in stable_reports))
            self.assertEqual(
                {row["duplicate_batch_id"] for row in stable_reports},
                {"20260618_001", "20260701_001"},
            )
            self.assertTrue(all(row["dedup_key"] == row["stable_sample_id"] for row in stable_reports))


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
        self.assertFalse(hasattr(args, "display_" + "selection_limit"))
        self.assertFalse(hasattr(args, "display_" + "exploration_limit"))
        self.assertEqual(args.package_weight_temperature, 0.1)
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

    def test_query_main_returns_nonzero_after_final_explanation_failure(self):
        result = types.SimpleNamespace(success=False, error_message="invalid explanation")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "query.xlsx"
            with patch.object(
                sys,
                "argv",
                ["query_cost_estimate_llm.py", "--text", "屋面漏水", "--output", str(output_path)],
            ), patch.object(query_estimate_llm, "check_lmstudio_service"), patch.object(
                query_estimate_llm, "run_query", return_value=result
            ), patch.object(query_estimate_llm, "print_terminal_summary"), patch("builtins.print") as print_mock:
                exit_code = query_estimate_llm.main()

        self.assertEqual(exit_code, 1)
        print_mock.assert_called_once_with(
            "[ERROR] final_explanation 失败，已输出无说明估价: invalid explanation"
        )

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
            "extra_analysis": [{"raw_text": "500平", "value": 500, "unit": "m²"}],
            "extra_specs": ["3mm SBS"],
            "likely_catalog": {"SHOULD": "IGNORE"},
        }
        with patch.object(query_estimate_llm, "request_llm_json", return_value=llm_result):
            rewrite, trace = query_estimate_llm.query_rewrite_for_embedding("屋面漏水")

        self.assertTrue(rewrite.success)
        self.assertEqual(rewrite.project_package_query_text, "屋面漏水维修工程 屋面卷材防水")
        self.assertEqual(rewrite.item_query_text, "屋面漏水维修工程 屋面卷材防水")
        self.assertIn("item_query_text 为空", rewrite.notes[0])
        self.assertEqual(trace["stage"], "query_rewrite_for_embedding")

        with patch.object(query_estimate_llm, "request_llm_json", side_effect=query_estimate_llm.LLMServiceError("down")):
            fallback, trace = query_estimate_llm.query_rewrite_for_embedding("屋面漏水")

        self.assertFalse(fallback.success)
        self.assertEqual(fallback.project_package_query_text, "屋面漏水")
        self.assertEqual(fallback.item_query_text, "屋面漏水")
        self.assertEqual(trace["parsed_status"], "failed")

    def test_package_evidence_weights_use_continuous_softmax(self):
        weights = query_estimate_llm.build_package_evidence_weights(
            ["p1", "p2", "p3"],
            {"p1": 0.9, "p2": 0.6, "p3": 0.7},
            temperature=0.1,
        )

        self.assertEqual(weights.columns.tolist(), query_estimate_llm.PACKAGE_EVIDENCE_WEIGHT_COLUMNS)
        self.assertAlmostEqual(float(weights["package_evidence_weight"].sum()), 1.0)
        p1 = weights[weights["project_package_id"].eq("p1")].iloc[0]
        p2 = weights[weights["project_package_id"].eq("p2")].iloc[0]
        self.assertGreater(float(p1["package_evidence_weight"]), float(p2["package_evidence_weight"]))
        with self.assertRaisesRegex(ValueError, "必须大于 0"):
            query_estimate_llm.build_package_evidence_weights(["p1"], {"p1": 0.9}, temperature=0)
        empty = query_estimate_llm.build_package_evidence_weights([], {}, temperature=0.1)
        self.assertTrue(empty.empty)

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

    def test_build_retrieved_evidence_items_uses_base_similarities_without_composite_scores(self):
        samples = self.prepared_samples()
        packages = build_index.build_project_packages(samples)
        matched = packages[packages["project_package_id"].isin(["batch-a::2", "batch-a::3"])].copy()
        matched.insert(0, "package_query_similarity", [0.9, 0.6])
        matched.insert(0, "rank", [1, 2])
        direct = samples[samples["sample_index"].isin([0, 3])].copy()
        item_query_similarities = np.array([0.95, 0.2, 0.3, 0.7], dtype=np.float32)
        package_similarity_by_id = {"batch-a::2": 0.9, "batch-a::3": 0.6, "batch-a::4": 0.4}

        retrieved_evidence_items = query_estimate_llm.build_retrieved_evidence_items(
            samples,
            matched,
            direct,
            item_query_similarities,
            package_query_similarity_by_id=package_similarity_by_id,
        )

        self.assertEqual(sorted(retrieved_evidence_items["sample_index"].tolist()), [0, 1, 2, 3])
        roof = retrieved_evidence_items[retrieved_evidence_items["sample_index"] == 0].iloc[0]
        pipe = retrieved_evidence_items[retrieved_evidence_items["sample_index"] == 3].iloc[0]
        self.assertAlmostEqual(float(roof["package_query_similarity"]), 0.9)
        self.assertAlmostEqual(float(roof["item_query_similarity"]), 0.95)
        self.assertEqual(roof["source_ref"], "batch-a::2::2-1")
        self.assertEqual(pipe["package_query_similarity"], 0.4)
        self.assertEqual(pipe["direct_hit"], True)
        self.assertNotIn("final_score", retrieved_evidence_items.columns)
        self.assertNotIn("cooccur_score", retrieved_evidence_items.columns)

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
        candidates["package_query_similarity"] = [0.8, 0.7, 0.6]
        candidates["package_rank"] = [1, 2, 3]
        candidates["item_query_similarity"] = [0.9, 0.8, 0.7]
        candidates["source_ref"] = ["batch-a::2::2-1", "batch-a::5::5-1", "batch-a::6::6-1"]

        families = query_estimate_llm.build_candidate_families(candidates)

        self.assertEqual(families.columns.tolist(), query_estimate_llm.CANDIDATE_FAMILY_COLUMNS)
        self.assertEqual(len(families), 2)
        roof3 = families[families["representative_project_description"].str.contains("3.0mm")].iloc[0]
        roof4 = families[families["representative_project_description"].str.contains("4.0mm")].iloc[0]
        self.assertEqual(roof3["fine_signature"], build_index.build_fine_signature(candidates.iloc[0]))
        self.assertEqual(roof3["本次召回样本数"], 2)
        self.assertEqual(roof3["本次召回工程包数"], 2)
        self.assertEqual(roof3["本次召回综合单价最低值"], 80.0)
        self.assertEqual(roof3["本次召回综合单价中位数"], 85.0)
        self.assertEqual(roof3["本次召回综合单价最高值"], 90.0)
        self.assertEqual(roof3["本次召回人工费单价中位数"], 21.0)
        self.assertEqual(roof3["本次召回机械费单价中位数"], 5.5)
        self.assertEqual(roof3["package_query_similarity最大值"], 0.8)
        self.assertEqual(roof3["item_query_similarity最大值"], 0.9)
        self.assertEqual(roof3["representative_project_description"], "3.0mm SBS 沥青防水卷材")
        self.assertEqual(roof4["本次召回综合单价最低值"], 120.0)

        evidence = query_estimate_llm.attach_family_ids_to_evidence_items(candidates, families)
        family_by_signature = dict(zip(families["fine_signature"], families["family_id"], strict=False))
        self.assertEqual(len(evidence), len(candidates))
        self.assertIn("family_id", evidence.columns)
        self.assertIn("fine_signature", evidence.columns)
        self.assertFalse(evidence["family_id"].map(query_estimate_llm.cell_text).eq("").any())
        for _index, row in evidence.iterrows():
            self.assertEqual(row["family_id"], family_by_signature[row["fine_signature"]])

    def test_display_groups_group_by_display_name_and_unit_without_merging_prices(self):
        families = pd.DataFrame(
            [
                {
                    "family_id": "F001",
                    "fine_signature": "sig-1",
                    "representative_cost_item_name": "屋面卷材防水",
                    "representative_project_description": "3.0mm SBS",
                    "unit": "平方米",
                    "unit_normalized": "m²",
                    "本次召回样本数": 2,
                    "本次召回工程包数": 2,
                    "本次召回综合单价中位数": 80,
                    "item_query_similarity最大值": 0.8,
                },
                {
                    "family_id": "F002",
                    "fine_signature": "sig-2",
                    "representative_cost_item_name": "屋面卷材防水",
                    "representative_project_description": "4.0mm SBS",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "本次召回样本数": 3,
                    "本次召回工程包数": 3,
                    "本次召回综合单价中位数": 120,
                    "item_query_similarity最大值": 0.7,
                },
                {
                    "family_id": "F003",
                    "fine_signature": "sig-3",
                    "representative_cost_item_name": "屋面卷材防水",
                    "representative_project_description": "含基层处理",
                    "unit": "m²",
                    "unit_normalized": "m²",
                    "本次召回样本数": 1,
                    "本次召回工程包数": 1,
                    "本次召回综合单价中位数": 140,
                    "item_query_similarity最大值": 0.6,
                },
                {
                    "family_id": "F004",
                    "fine_signature": "sig-4",
                    "representative_cost_item_name": "垃圾外运",
                    "representative_project_description": "建筑垃圾外运",
                    "unit": "项",
                    "unit_normalized": "项",
                    "本次召回样本数": 1,
                    "本次召回工程包数": 1,
                    "本次召回综合单价中位数": 500,
                    "item_query_similarity最大值": 0.5,
                },
            ],
            columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS,
        )
        evidence = pd.DataFrame(
            [
                {"family_id": "F001", "project_package_id": "p1"},
                {"family_id": "F002", "project_package_id": "p2"},
                {"family_id": "F003", "project_package_id": "p2"},
                {"family_id": "F004", "project_package_id": "p3"},
            ]
        )

        groups, group_families = query_estimate_llm.build_candidate_display_groups(families, evidence)

        self.assertEqual(len(groups), 2)
        roof = groups[groups["display_name"].eq("屋面卷材防水")].iloc[0]
        self.assertEqual(roof["family_count"], 3)
        self.assertEqual(roof["family_ids"], "F001,F002,F003")
        self.assertEqual(roof["retrieval_package_count"], 2)
        self.assertEqual(roof["direct_item_similarity_max"], 0.8)
        self.assertEqual(set(group_families[group_families["display_id"].eq(roof["display_id"])]["family_id"]), {"F001", "F002", "F003"})
        self.assertEqual(families[families["family_id"].eq("F001")].iloc[0]["本次召回综合单价中位数"], 80)
        self.assertEqual(families[families["family_id"].eq("F002")].iloc[0]["本次召回综合单价中位数"], 120)

    def test_display_groups_do_not_merge_different_units(self):
        families = pd.DataFrame(
            [
                {"family_id": "F001", "representative_cost_item_name": "垃圾外运", "representative_project_description": "按项", "unit": "项", "unit_normalized": "项", "本次召回样本数": 1, "本次召回工程包数": 1, "item_query_similarity最大值": 0.9},
                {"family_id": "F002", "representative_cost_item_name": "垃圾外运", "representative_project_description": "按方", "unit": "m³", "unit_normalized": "m³", "本次召回样本数": 1, "本次召回工程包数": 1, "item_query_similarity最大值": 0.8},
            ],
            columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS,
        )
        groups, _families = query_estimate_llm.build_candidate_display_groups(families, pd.DataFrame([{"family_id": "F001", "project_package_id": "p1"}, {"family_id": "F002", "project_package_id": "p2"}]))

        self.assertEqual(len(groups), 2)
        self.assertEqual(set(groups["unit"]), {"项", "m³"})

    def test_display_source_package_count_uses_evidence_distinct_packages(self):
        families = pd.DataFrame(
            [
                {"family_id": "F001", "representative_cost_item_name": "屋面卷材防水", "representative_project_description": "3mm", "unit": "m²", "unit_normalized": "m²", "本次召回样本数": 2, "本次召回工程包数": 2, "item_query_similarity最大值": 0.9},
                {"family_id": "F002", "representative_cost_item_name": "屋面卷材防水", "representative_project_description": "4mm", "unit": "m²", "unit_normalized": "m²", "本次召回样本数": 2, "本次召回工程包数": 2, "item_query_similarity最大值": 0.8},
            ],
            columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS,
        )
        evidence = pd.DataFrame(
            [
                {"family_id": "F001", "project_package_id": "p1"},
                {"family_id": "F001", "project_package_id": "p2"},
                {"family_id": "F002", "project_package_id": "p2"},
                {"family_id": "F002", "project_package_id": "p3"},
            ]
        )

        groups, _families = query_estimate_llm.build_candidate_display_groups(families, evidence)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups.iloc[0]["retrieval_package_count"], 3)

    def test_display_support_ratio_counts_each_package_once(self):
        groups = pd.DataFrame(
            [
                {"display_id": "D001", "display_key": "a|m²", "display_name": "A", "unit": "m²", "family_count": 2, "family_ids": "F001,F002", "top_family_examples": "[]"},
                {"display_id": "D002", "display_key": "b|m²", "display_name": "B", "unit": "m²", "family_count": 1, "family_ids": "F003", "top_family_examples": "[]"},
            ],
            columns=query_estimate_llm.CANDIDATE_DISPLAY_GROUP_COLUMNS,
        )
        display_families = pd.DataFrame(
            [
                {"display_id": "D001", "family_id": "F001"},
                {"display_id": "D001", "family_id": "F002"},
                {"display_id": "D002", "family_id": "F003"},
            ]
        )
        evidence = pd.DataFrame(
            [
                {"family_id": "F001", "project_package_id": "p1", "item_query_similarity": 0.8},
                {"family_id": "F002", "project_package_id": "p1", "item_query_similarity": 0.7},
                {"family_id": "F002", "project_package_id": "p2", "item_query_similarity": 0.9},
                {"family_id": "F003", "project_package_id": "p3", "item_query_similarity": 0.6},
            ]
        )
        weights = pd.DataFrame(
            [
                {"project_package_id": "p1", "package_query_similarity": 0.9, "package_evidence_weight": 0.5},
                {"project_package_id": "p2", "package_query_similarity": 0.8, "package_evidence_weight": 0.3},
                {"project_package_id": "p3", "package_query_similarity": 0.7, "package_evidence_weight": 0.2},
            ],
            columns=query_estimate_llm.PACKAGE_EVIDENCE_WEIGHT_COLUMNS,
        )

        output = query_estimate_llm.attach_display_support_ratios(groups, display_families, evidence, weights)
        d001 = output[output["display_id"].eq("D001")].iloc[0]

        self.assertAlmostEqual(float(d001["retrieval_package_support_ratio"]), 0.8)
        self.assertEqual(d001["retrieval_item_count"], 3)
        self.assertEqual(d001["retrieval_package_count"], 2)
        self.assertAlmostEqual(float(d001["direct_item_similarity_max"]), 0.9)

    def test_normalize_display_description_repairs_only_leading_ocr_numbering(self):
        normalize = query_estimate_llm.normalize_display_description

        self.assertEqual(normalize("1.3.0mmSBS防水卷材"), "3.0mmSBS防水卷材")
        self.assertEqual(normalize("1.4.0mmSBS防水卷材"), "4.0mmSBS防水卷材")
        self.assertEqual(normalize("1.1.5mm聚氨酯"), "1.5mm聚氨酯")
        self.assertEqual(normalize("1.5mm聚氨酯"), "1.5mm聚氨酯")

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

    def test_matched_project_examples_use_top_three_complete_packages_and_source_order(self):
        matched = pd.DataFrame(
            [
                {"rank": 3, "project_package_id": "P3", "工程名称": "三"},
                {"rank": 1, "project_package_id": "P1", "工程名称": "一"},
                {"rank": 4, "project_package_id": "P4", "工程名称": "四"},
                {"rank": 2, "project_package_id": "P2", "工程名称": "二"},
            ]
        )
        samples = pd.DataFrame(
            [
                {"project_package_id": "P1", "item_row_id": "7-10", "seq": 10, "cost_item_name": "第十项"},
                {"project_package_id": "P1", "item_row_id": "7-2", "seq": 2, "cost_item_name": "第二项"},
                {"project_package_id": "P1", "item_row_id": "7-1", "seq": 1, "cost_item_name": "第一项"},
                {"project_package_id": "P2", "item_row_id": "x", "seq": 2, "cost_item_name": "P2-2"},
                {"project_package_id": "P2", "item_row_id": "y", "seq": 1, "cost_item_name": "P2-1"},
                {"project_package_id": "P3", "item_row_id": "1", "seq": 1, "cost_item_name": "P3-1"},
                {"project_package_id": "P4", "item_row_id": "1", "seq": 1, "cost_item_name": "P4-1"},
            ]
        )
        samples["stable_sample_id"] = [f"sid-{index}" for index in range(len(samples))]
        sample_lookup = {
            row["stable_sample_id"]: {
                "source_ref": "duplicate-ref" if index < 2 else f"ref-{index}",
                "family_id": f"F{index:03d}",
                "display_id": f"D{index:03d}",
                "practice_option_id": f"D{index:03d}-O01",
            }
            for index, row in samples.iterrows()
        }

        examples = query_estimate_llm.build_matched_project_examples(matched, samples, sample_lookup)

        self.assertEqual([example["project_package_id"] for example in examples], ["P1", "P2", "P3"])
        self.assertEqual([item["cost_item_name"] for item in examples[0]["items"]], ["第一项", "第二项", "第十项"])
        self.assertEqual([item["cost_item_name"] for item in examples[1]["items"]], ["P2-1", "P2-2"])
        self.assertEqual(sum(len(example["items"]) for example in examples), 6)
        self.assertTrue(all(item["stable_sample_id"] for example in examples for item in example["items"]))
        self.assertEqual(examples[0]["items"][0]["source_ref"], "ref-2")

    def test_display_option_grouping_prompt_includes_family_coverage_contract(self):
        display_groups = pd.DataFrame([{"display_id": "D001", "display_name": "屋面卷材防水", "unit": "m²", "family_count": 2}])
        display_families = pd.DataFrame(
            [
                {
                    "display_id": "D001",
                    "family_id": "F001",
                    "representative_cost_item_name": "屋面卷材防水",
                    "representative_project_description": "A" * 130,
                    "unit": "m²",
                    "本次召回样本数": 1,
                    "本次召回工程包数": 1,
                    "item_query_similarity最大值": 0.8,
                },
                {
                    "display_id": "D001",
                    "family_id": "F002",
                    "representative_cost_item_name": "屋面卷材防水",
                    "representative_project_description": "4mm SBS",
                    "unit": "m²",
                    "本次召回样本数": 2,
                    "本次召回工程包数": 2,
                    "item_query_similarity最大值": 0.9,
                },
            ]
        )
        candidate_families = pd.DataFrame(
            [
                {"family_id": "F001", "本次召回综合单价最低值": 80, "本次召回综合单价中位数": 90, "本次召回综合单价最高值": 100},
                {"family_id": "F002", "本次召回综合单价最低值": 100, "本次召回综合单价中位数": 110, "本次召回综合单价最高值": 120},
            ],
            columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS,
        )

        prompt, records = query_estimate_llm.build_display_option_grouping_prompt(
            display_groups, display_families, candidate_families
        )

        self.assertEqual(set(records[0]), {"display_id", "display_name", "candidate_families"})
        self.assertEqual(
            set(records[0]["candidate_families"][0]),
            {"family_id", "name", "spec", "unit"},
        )
        self.assertEqual(records[0]["candidate_families"][1]["spec"], "A" * 120)
        self.assertNotIn("candidate_family_ids", prompt)
        self.assertNotIn("candidate_family_count", prompt)
        for removed_field in ["samples", "packages", "item_query_similarity", "unit_price_min", "unit_price_median", "unit_price_max"]:
            self.assertNotIn(f'"{removed_field}"', prompt)
        self.assertIn("仅存在 OCR、标点、文字顺序、同义表达", prompt)
        self.assertIn("关键规格、厚度", prompt)
        for removed_text in ["raw_query", "project_package_query_text", "item_query_text", "selection_reason", "default_representative_family_id", "family_assignments", "group_reason"]:
            self.assertNotIn(removed_text, prompt)

    def test_display_option_grouping_groups_practice_options_without_legacy_fields(self):
        display_groups = pd.DataFrame([{"display_id": "D001", "display_name": "屋面卷材防水", "unit": "m²", "family_count": 3}])
        display_families = pd.DataFrame(
            [
                {"display_id": "D001", "family_id": "F004", "unit": "m²", "本次召回样本数": 2},
                {"display_id": "D001", "family_id": "F007", "unit": "m²", "本次召回样本数": 3},
                {"display_id": "D001", "family_id": "F010", "unit": "m²", "本次召回样本数": 1},
            ]
        )

        displays_with_options, meta = query_estimate_llm.parse_display_option_grouping_result(
            {
                "display_results": [
                    {
                        "display_id": "D001",
                        "practice_options": [
                            {
                                "representative_family_id": "F004",
                                "practice_description": "3mm SBS卷材防水",
                                "family_ids": ["F004", "F007"],
                            },
                            {
                                "representative_family_id": "F010",
                                "practice_description": "聚氨酯涂膜防水",
                                "family_ids": ["F010"],
                            },
                        ],
                    }
                ]
            },
            display_groups,
            display_families,
            [],
        )

        self.assertEqual(displays_with_options.loc[0, "practice_options"][0]["practice_option_id"], "D001-O01")
        self.assertEqual(displays_with_options.loc[0, "practice_options"][0]["sample_count"], 5)
        self.assertEqual(displays_with_options.loc[0, "practice_options"][1]["practice_option_id"], "D001-O02")
        self.assertNotIn("selected_family_id", displays_with_options.columns)
        self.assertNotIn("default_practice_option_id", displays_with_options.columns)
        self.assertNotIn("default_practice", displays_with_options.columns)
        self.assertNotIn("family_relations", displays_with_options.columns)
        self.assertNotIn("selection_reason", displays_with_options.columns)
        self.assertNotIn("group_reason", displays_with_options.loc[0, "practice_options"][0])
        self.assertEqual(meta["display_ids"], ["D001"])
        self.assertNotIn("default_practice_option_ids", meta)
        self.assertEqual(meta["practice_option_count"], 2)
        self.assertEqual(meta["families_grouped_count"], 3)
        self.assertEqual(meta["option_count_by_display"], {"D001": 2})
        self.assertEqual(meta["max_options_per_display"], 2)

    def test_display_option_grouping_covers_each_of_four_families_once(self):
        display_groups = pd.DataFrame([{"display_id": "D001", "display_name": "屋面防水", "unit": "m²", "family_count": 4}])
        display_families = pd.DataFrame(
            [{"display_id": "D001", "family_id": family_id, "unit": "m²"} for family_id in ["F001", "F002", "F003", "F004"]]
        )
        displays_with_options, meta = query_estimate_llm.parse_display_option_grouping_result(
            {
                "display_results": [
                    {
                        "display_id": "D001",
                        "practice_options": [
                            {"representative_family_id": "F001", "practice_description": "1.5mm涂膜", "family_ids": ["F001", "F002"]},
                            {"representative_family_id": "F003", "practice_description": "2.0mm涂膜", "family_ids": ["F003", "F004"]},
                        ],
                    }
                ]
            },
            display_groups,
            display_families,
            [],
        )

        flattened = [family_id for option in displays_with_options.loc[0, "practice_options"] for family_id in option["family_ids"]]
        self.assertEqual(flattened, ["F001", "F002", "F003", "F004"])
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(meta["families_grouped_count"], 4)

    def test_display_option_grouping_strict_validation_rejects_invalid_options(self):
        display_groups = pd.DataFrame([{"display_id": "D001", "display_name": "屋面防水", "unit": "m²", "family_count": 2}])
        display_families = pd.DataFrame(
            [
                {"display_id": "D001", "family_id": "F001", "unit": "m²"},
                {"display_id": "D001", "family_id": "F002", "unit": "m²"},
            ]
        )
        valid_result = {
            "display_results": [
                {
                    "display_id": "D001",
                    "practice_options": [
                        {
                            "representative_family_id": "F001",
                            "practice_description": "3mm SBS",
                            "family_ids": ["F001", "F002"],
                        }
                    ],
                }
            ]
        }

        def assert_invalid(result, message):
            with self.assertRaisesRegex(ValueError, message):
                query_estimate_llm.parse_display_option_grouping_result(
                    result,
                    display_groups,
                    display_families,
                    [],
                )

        assert_invalid({"display_results": [{**valid_result["display_results"][0], "display_id": "D999"}]}, "未知 display")
        assert_invalid(
            {"display_results": [valid_result["display_results"][0], valid_result["display_results"][0]]},
            "重复返回 display",
        )
        assert_invalid(
            {
                "display_results": [
                    {
                        "display_id": "D001",
                        "practice_options": [
                            {
                                "representative_family_id": "F001",
                                "practice_description": "3mm SBS",
                                "family_ids": ["F001"],
                            }
                        ],
                    }
                ],
            },
            "遗漏 candidate family",
        )
        assert_invalid(
            {
                "display_results": [
                    {
                        "display_id": "D001",
                        "practice_options": [
                            {
                                "representative_family_id": "F001",
                                "practice_description": "3mm SBS",
                                "family_ids": ["F001", "F001", "F002"],
                            }
                        ],
                    }
                ],
            },
            "family 重复",
        )
        assert_invalid(
            {
                "display_results": [
                    {
                        "display_id": "D001",
                        "practice_options": [
                            {
                                "representative_family_id": "F001",
                                "practice_description": "3mm SBS",
                                "family_ids": ["F001", "BAD"],
                            }
                        ],
                    }
                ],
            },
            "不属于当前 display",
        )
        for legacy_field in ["default_representative_family_id", "selection_reason", "family_assignments"]:
            assert_invalid(
                {"display_results": [{**valid_result["display_results"][0], legacy_field: "legacy"}]},
                "display_result 只允许包含",
            )
        assert_invalid(
            {"display_results": [{"display_id": "D001", "practice_options": [{**valid_result["display_results"][0]["practice_options"][0], "group_reason": "legacy"}]}]},
            "practice_option 只允许包含",
        )
        assert_invalid(
            {"display_results": [{"display_id": "D001", "practice_options": [{"representative_family_id": "BAD", "practice_description": "3mm SBS", "family_ids": ["F001", "F002"]}]}]},
            "representative_family_id 必须位于",
        )
        assert_invalid(
            {"display_results": [{"display_id": "D001", "practice_options": [
                {"representative_family_id": "F001", "practice_description": "做法一", "family_ids": ["F001", "F002"]},
                {"representative_family_id": "F002", "practice_description": "做法二", "family_ids": ["F002", "F001"]},
            ]}]},
            "完全相同的 family_ids 分组",
        )

    def test_display_option_grouping_rejects_unit_mismatch_inside_option(self):
        display_groups = pd.DataFrame([{"display_id": "D001", "display_name": "屋面防水", "unit": "m²", "family_count": 2}])
        display_families = pd.DataFrame(
            [
                {"display_id": "D001", "family_id": "F001", "unit": "m²"},
                {"display_id": "D001", "family_id": "F002", "unit": "项"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "单位必须一致"):
            query_estimate_llm.parse_display_option_grouping_result(
                {
                    "display_results": [
                        {
                            "display_id": "D001",
                            "practice_options": [
                                {
                                    "representative_family_id": "F001",
                                    "practice_description": "屋面防水",
                                    "family_ids": ["F001", "F002"],
                                }
                            ],
                        }
                    ]
                },
                display_groups,
                display_families,
                [],
            )

    def test_generate_display_option_grouping_handles_all_displays_with_one_llm_call(self):
        display_groups = pd.DataFrame(
            [
                {"display_id": "D001", "display_name": "屋面防水", "unit": "m²", "family_count": 1},
                {"display_id": "D002", "display_name": "墙面防水", "unit": "m²", "family_count": 2},
            ]
        )
        display_families = pd.DataFrame(
            [
                {"display_id": "D001", "family_id": "F001", "unit": "m²", "本次召回样本数": 1, "本次召回工程包数": 1, "item_query_similarity最大值": 0.9},
                {"display_id": "D002", "family_id": "F002", "unit": "m²", "本次召回样本数": 2, "本次召回工程包数": 2, "item_query_similarity最大值": 0.8},
                {"display_id": "D002", "family_id": "F003", "unit": "m²", "本次召回样本数": 3, "本次召回工程包数": 3, "item_query_similarity最大值": 0.7},
            ]
        )
        candidate_families = pd.DataFrame(
            [
                {"family_id": "F001", "representative_cost_item_name": "屋面防水", "representative_project_description": "3mm SBS", "unit": "m²", "unit_normalized": "m²", "本次召回样本数": 1},
                {"family_id": "F002", "representative_cost_item_name": "墙面防水", "representative_project_description": "1.5mm涂膜", "unit": "m²", "unit_normalized": "m²", "本次召回样本数": 2},
                {"family_id": "F003", "representative_cost_item_name": "墙面防水", "representative_project_description": "2.0mm涂膜", "unit": "m²", "unit_normalized": "m²", "本次召回样本数": 3},
            ],
            columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS,
        )
        response = types.SimpleNamespace(
            content={
                "display_results": [
                    {
                        "display_id": "D002",
                        "practice_options": [
                            {"representative_family_id": "F002", "practice_description": "1.5mm涂膜", "family_ids": ["F002"]},
                            {"representative_family_id": "F003", "practice_description": "2.0mm涂膜", "family_ids": ["F003"]},
                        ],
                    }
                ]
            },
            usage={},
            raw_content="{}",
        )

        with patch.object(query_estimate_llm, "request_llm_json_with_usage", return_value=response) as llm_mock:
            displays_with_options, *_rest, meta, trace_frame = query_estimate_llm.generate_display_option_grouping(
                display_groups, display_families, candidate_families, []
            )

        llm_mock.assert_called_once()
        self.assertEqual(displays_with_options["display_id"].tolist(), ["D001", "D002"])
        self.assertEqual(displays_with_options.iloc[0]["practice_options"][0]["practice_option_id"], "D001-O01")
        self.assertEqual(displays_with_options.iloc[1]["practice_options"][1]["practice_option_id"], "D002-O02")
        self.assertEqual(meta["programmatic_single_family_display_count"], 1)
        self.assertEqual(meta["llm_display_count"], 1)
        self.assertEqual(set(trace_frame["display_id"]), {"D001", "D002"})

    def displays_with_options_fixture(self):
        return pd.DataFrame(
            [
                {
                    "display_id": "D001",
                    "display_name": "屋面防水",
                    "unit": "m²",
                    "practice_options": [
                        {
                            "practice_option_id": "D001-O01",
                            "representative_family_id": "F001",
                            "practice_description": "3mm SBS卷材防水",
                            "sample_count": 3,
                            "family_ids": ["F001", "F002"],
                        },
                        {
                            "practice_option_id": "D001-O02",
                            "representative_family_id": "F003",
                            "practice_description": "聚氨酯涂膜防水",
                            "sample_count": 1,
                            "family_ids": ["F003"],
                        },
                    ],
                },
                {
                    "display_id": "D002",
                    "display_name": "防水层拆除",
                    "unit": "m²",
                    "practice_options": [
                        {
                            "practice_option_id": "D002-O01",
                            "representative_family_id": "F004",
                            "practice_description": "拆除原防水层",
                            "sample_count": 2,
                            "family_ids": ["F004"],
                        }
                    ],
                },
            ]
        )

    def test_display_option_grouping_trace_marks_practice_options_without_default_fields(self):
        displays_with_options = self.displays_with_options_fixture().iloc[[0]].copy()
        display_families = pd.DataFrame(
            [
                {"display_id": "D001", "display_name": "屋面防水", "family_id": family_id}
                for family_id in ["F001", "F002", "F003"]
            ]
        )
        families = pd.DataFrame(
            [
                {"family_id": "F001", "unit": "m²", "unit_normalized": "m²"},
                {"family_id": "F002", "unit": "m²", "unit_normalized": "m²"},
                {"family_id": "F003", "unit": "m²", "unit_normalized": "m²"},
            ],
            columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS,
        )

        trace = query_estimate_llm.build_display_option_grouping_trace_frame(displays_with_options, display_families, families).set_index("family_id")

        self.assertIn("practice_option_id", query_estimate_llm.DISPLAY_OPTION_GROUPING_TRACE_COLUMNS)
        self.assertNotIn("is_current_default_option", query_estimate_llm.DISPLAY_OPTION_GROUPING_TRACE_COLUMNS)
        self.assertNotIn("included_in_current_price_scope", query_estimate_llm.DISPLAY_OPTION_GROUPING_TRACE_COLUMNS)
        self.assertNotIn("group_reason", query_estimate_llm.DISPLAY_OPTION_GROUPING_TRACE_COLUMNS)
        self.assertEqual(trace.index.tolist(), ["F001", "F002", "F003"])
        self.assertEqual(trace.loc["F001", "practice_option_id"], "D001-O01")
        self.assertEqual(trace.loc["F003", "practice_option_id"], "D001-O02")

    def historical_plan_fixtures(self):
        displays = pd.DataFrame(
            [
                {
                    "display_id": "D001",
                    "display_name": "屋面卷材防水",
                    "unit": "m²",
                    "practice_options": [
                        {
                            "practice_option_id": "D001-O01", "practice_description": "3mm SBS",
                            "representative_family_id": "F001", "family_ids": ["F001"], "sample_count": 3,
                        },
                        {
                            "practice_option_id": "D001-O02", "practice_description": "4mm SBS",
                            "representative_family_id": "F002", "family_ids": ["F002"], "sample_count": 2,
                        },
                    ],
                },
                {
                    "display_id": "D002",
                    "display_name": "防水层拆除",
                    "unit": "m²",
                    "practice_options": [
                        {"practice_option_id": "D002-O01", "practice_description": "拆除原防水层", "family_ids": ["F003"]},
                    ],
                },
            ]
        )
        examples = [
            {
                "project_package_id": "PKG-1",
                "project_name": "历史屋面工程",
                "items": [
                    {"stable_sample_id": "sid-main", "display_id": "D001"},
                    {"stable_sample_id": "sid-remove", "display_id": "D002"},
                ],
            },
            {"project_package_id": "PKG-2", "project_name": "其他工程", "items": [{"stable_sample_id": "sid-other"}]},
            {"project_package_id": "PKG-3", "project_name": "第三个工程", "items": [{"stable_sample_id": "sid-third"}]},
        ]
        lookup = {
            "sid-main": {"project_package_id": "PKG-1", "source_ref": "same-ref", "family_id": "F002", "display_id": "D001", "practice_option_id": "D001-O02"},
            "sid-remove": {"project_package_id": "PKG-1", "source_ref": "same-ref", "family_id": "F003", "display_id": "D002", "practice_option_id": "D002-O01"},
            "sid-other": {"project_package_id": "PKG-2", "source_ref": "ref-other", "family_id": "F001", "display_id": "D001", "practice_option_id": "D001-O01"},
            "sid-third": {"project_package_id": "PKG-3", "source_ref": "ref-third", "family_id": "F002", "display_id": "D001", "practice_option_id": "D001-O02"},
            "sid-not-in-input": {"project_package_id": "PKG-1", "source_ref": "ref-hidden", "family_id": "F001", "display_id": "D001", "practice_option_id": "D001-O01"},
        }
        valid = {
            "project_selections": [True, False, False],
            "selected_items": [
                {
                    "keep": True,
                    "practice_option_id": "D001-O01",
                    "quantity": {"type": "exact", "value": 500},
                    "quantity_reason": "用户明确屋面维修面积约500㎡并指定3mm SBS",
                },
                {
                    "keep": True,
                    "practice_option_id": "D002-O01",
                    "quantity": {"type": "range", "min": 0, "max": 500},
                    "quantity_reason": "拆除范围需现场确认，不机械复制主体工程量",
                },
            ],
        }
        return displays, examples, lookup, valid

    def historical_option_model(self):
        class FakeModel:
            def encode(self, texts, **_kwargs):
                vectors = []
                for text in texts:
                    value = 10.0 if "3mm" in text else 1.0
                    match = re.search(r"score-(\d+)", text)
                    if match:
                        value = float(match.group(1))
                    vectors.append([value, 1.0])
                return np.asarray(vectors, dtype=np.float32)

        return FakeModel()

    def historical_plan_context(self, raw_text, examples, displays, lookup):
        return query_estimate_llm.build_historical_plan_determination_prompt(
            raw_text,
            examples,
            displays,
            lookup,
            self.historical_option_model(),
            np.asarray([1.0, 0.0], dtype=np.float32),
        )

    def test_historical_plan_parser_allows_same_display_practice_selection(self):
        displays, examples, lookup, valid = self.historical_plan_fixtures()
        _prompt, projects, display_options = self.historical_plan_context(
            "屋面漏水，想做3mm SBS防水，面积大概500平", examples, displays, lookup
        )

        plan = query_estimate_llm.parse_historical_plan_determination_result(
            valid, projects, display_options, displays, lookup
        )

        self.assertEqual(plan.project_package_id, "PKG-1")
        self.assertEqual(plan.items[0].stable_sample_id, "sid-main")
        self.assertEqual(plan.items[0].display_id, "D001")
        self.assertEqual(plan.items[0].source_ref, "same-ref")
        self.assertEqual(plan.items[0].practice_option_id, "D001-O01")
        self.assertEqual(plan.items[0].quantity, {"type": "exact", "value": 500.0})
        self.assertEqual(plan.items[1].quantity, {"type": "range", "min": 0.0, "max": 500.0})

    def test_historical_plan_prompt_contains_only_public_project_and_item_fields(self):
        displays, examples, lookup, _valid = self.historical_plan_fixtures()
        examples[0]["items"][0].update(
            {
                "source_ref": "source-a",
                "family_id": "F002",
                "display_id": "D001",
                "practice_option_id": "D001-O02",
                "cost_item_name": "屋面卷材防水",
                "project_description": "4mm SBS改性沥青防水卷材",
                "unit": "m²",
                "quantity": 500,
                "unit_price": 9999,
                "package_query_similarity": 0.9,
            }
        )

        prompt, projects, display_options = self.historical_plan_context(
            "屋面漏水，面积大概500平", examples, displays, lookup
        )

        self.assertEqual(len(projects), 3)
        self.assertEqual(projects[0]["project_package_id"], "PKG-1")
        self.assertEqual(projects[0]["items"][0]["stable_sample_id"], "sid-main")
        self.assertEqual(projects[0]["items"][0]["display_id"], "D001")
        self.assertEqual(projects[0]["items"][0]["practice_option_id"], "D001-O02")
        self.assertEqual(projects[1]["items"][0]["practice_option_id"], "D001-O01")
        for project in projects:
            self.assertEqual(set(project), {"project_package_id", "project_name", "items"})
            for item in project["items"]:
                self.assertEqual(
                    set(item),
                    {
                        "stable_sample_id", "source_ref", "family_id", "display_id",
                        "practice_option_id", "cost_item_name",
                        "project_description", "unit", "quantity",
                    },
                )
        prompt_projects = query_estimate_llm.historical_project_prompt_records(projects)
        for project in prompt_projects:
            self.assertEqual(set(project), {"project_name", "items"})
            for item in project["items"]:
                self.assertEqual(
                    set(item),
                    {
                        "display_id", "practice_option_id", "cost_item_name",
                        "project_description", "unit", "quantity",
                    },
                )
        self.assertEqual(
            display_options,
            {
                "D001": [
                    {"practice_option_id": "D001-O02", "practice_description": "4mm SBS"},
                    {"practice_option_id": "D001-O01", "practice_description": "3mm SBS"},
                ]
            },
        )
        self.assertNotIn("D002", display_options)
        for option in display_options["D001"]:
            self.assertEqual(set(option), {"practice_option_id", "practice_description"})
        self.assertEqual(prompt.count('"practice_description": "4mm SBS"'), 1)
        self.assertEqual(prompt.count('"practice_description": "3mm SBS"'), 1)
        for forbidden_field in [
            "project_package_id", "stable_sample_id", "family_id", "family_ids",
            "representative_family_id", "source_ref", "core_display_id",
            "displays_with_options", "package_query_similarity", "unit_price", "total_price", "sample_count",
            "option_group_id", "original_practice_option_id", "is_current", "G01",
            "project_no", "item_no", "option_no",
            "单价", "合价", "相似度数值", "样本统计",
        ]:
            self.assertNotIn(forbidden_field, prompt)
        for required_rule in [
            "直接维修对象匹配优先于工程整体纯度",
            "对象层级、维修范围和维修动作",
            "不得用下一级部件替代完整对象",
            "不得用局部维修替代整体更换",
            "只有对象、层级、范围和动作基本匹配时，才比较工程整体纯度",
            "用户没有逐项提到某个施工层，不构成删除理由",
            "无法确认是否相关时优先保留",
            "不得为缩短输出压缩清单",
        ]:
            self.assertIn(required_rule, prompt)
        prompt_instructions = prompt.split("输入：", 1)[0]
        for forbidden_case in ["消防报警主机", "控制盘", "主板", "屋面防水"]:
            self.assertNotIn(forbidden_case, prompt_instructions)
        self.assertIn('"historical_projects"', prompt)
        self.assertIn('"display_options"', prompt)

        legacy_projects = json.loads(json.dumps(prompt_projects, ensure_ascii=False))
        for project in legacy_projects:
            for item in project["items"]:
                item["practice_options"] = display_options.get(item["display_id"], [])
        unified_tokens = query_estimate_llm.estimated_tokens(
            query_estimate_llm.json_text(
                {"historical_projects": prompt_projects, "display_options": display_options}
            )
        )
        duplicated_tokens = query_estimate_llm.estimated_tokens(
            query_estimate_llm.json_text({"historical_projects": legacy_projects})
        )
        self.assertLess(unified_tokens, duplicated_tokens)

    def test_historical_display_options_keep_originals_and_top_five_alternatives(self):
        projects = [
            {
                "project_package_id": "P1",
                "items": [
                    {"stable_sample_id": "s1", "display_id": "D900", "practice_option_id": "D900-O00"}
                ],
            }
        ]
        displays = pd.DataFrame(
            [
                {
                    "display_id": "D900",
                    "practice_options": [
                        {"practice_option_id": "D900-O00", "practice_description": "原工艺"},
                        *[
                            {
                                "practice_option_id": f"D900-O{index:02d}",
                                "practice_description": f"score-{index}",
                            }
                            for index in range(1, 7)
                        ],
                    ],
                }
            ]
        )

        display_options = query_estimate_llm.historical_display_options(
            projects,
            displays,
            self.historical_option_model(),
            np.asarray([1.0, 0.0], dtype=np.float32),
            max_alternatives=99,
        )

        self.assertEqual(
            [option["practice_option_id"] for option in display_options["D900"]],
            ["D900-O00", "D900-O06", "D900-O05", "D900-O04", "D900-O03", "D900-O02"],
        )

    def test_complete_object_replacement_can_select_non_first_project_by_position(self):
        examples = [
            {
                "project_package_id": "PART",
                "project_name": "内部部件工程",
                "items": [{
                    "stable_sample_id": "part-1", "cost_item_name": "内部部件维修",
                    "project_description": "维修内部部件", "unit": "块", "quantity": 1,
                }],
            },
            {
                "project_package_id": "WHOLE",
                "project_name": "完整设备工程",
                "items": [{
                    "stable_sample_id": "whole-1", "cost_item_name": "完整设备更换",
                    "project_description": "拆除并更换完整设备", "unit": "台", "quantity": 1,
                }],
            },
            {
                "project_package_id": "OTHER",
                "project_name": "其他工程",
                "items": [{
                    "stable_sample_id": "other-1", "cost_item_name": "附属构件维修",
                    "project_description": "局部维修附属构件", "unit": "项", "quantity": 1,
                }],
            },
        ]
        lookup = {
            "part-1": {"project_package_id": "PART", "source_ref": "r1", "family_id": "F101", "display_id": "D101", "practice_option_id": "D101-O01"},
            "whole-1": {"project_package_id": "WHOLE", "source_ref": "r2", "family_id": "F102", "display_id": "D102", "practice_option_id": "D102-O01"},
            "other-1": {"project_package_id": "OTHER", "source_ref": "r3", "family_id": "F103", "display_id": "D103", "practice_option_id": "D103-O01"},
        }
        displays = pd.DataFrame([
            {"display_id": f"D{index}", "practice_options": [{"practice_option_id": f"D{index}-O01", "practice_description": description}]}
            for index, description in [(101, "内部部件维修"), (102, "完整设备更换"), (103, "附属构件维修")]
        ])
        llm_result = {
            "project_selections": [False, True, False],
            "selected_items": [{
                "keep": True,
                "practice_option_id": "D102-O01",
                "quantity": {"type": "exact", "value": 1},
                "quantity_reason": "完整对象和更换动作直接匹配",
            }],
        }
        response = types.SimpleNamespace(
            content=llm_result, usage={}, raw_content=json.dumps(llm_result, ensure_ascii=False)
        )

        with patch.object(query_estimate_llm, "request_llm_json_with_usage", return_value=response):
            plan, success, error, prompt, trace = query_estimate_llm.generate_historical_plan_determination(
                "完整设备故障，需要更换",
                examples,
                displays,
                lookup,
                self.historical_option_model(),
                np.asarray([1.0, 0.0], dtype=np.float32),
            )

        self.assertTrue(success, error)
        self.assertEqual(plan.project_package_id, "WHOLE")
        self.assertEqual(plan.items[0].stable_sample_id, "whole-1")
        self.assertNotIn("project_package_id", prompt)
        self.assertNotIn("stable_sample_id", prompt)
        self.assertNotIn("project_package_id", trace["raw_response"])
        trace_summary = json.loads(trace["input_summary"])
        self.assertEqual(trace_summary["selected_project_package_id"], "WHOLE")
        self.assertEqual(trace_summary["selected_stable_sample_ids"], ["whole-1"])

    def test_historical_plan_prompt_rejects_missing_original_practice_option(self):
        displays, examples, lookup, _valid = self.historical_plan_fixtures()
        invalid_lookup = {
            **lookup,
            "sid-main": {**lookup["sid-main"], "practice_option_id": "D001-BAD"},
        }

        with self.assertRaisesRegex(ValueError, "原 practice option 无法回查"):
            self.historical_plan_context("屋面维修", examples, displays, invalid_lookup)

    def test_historical_plan_result_schema_contains_only_selection_practice_and_quantity_fields(self):
        _displays, _examples, _lookup, valid = self.historical_plan_fixtures()

        self.assertEqual(set(valid), {"project_selections", "selected_items"})
        self.assertEqual(valid["project_selections"], [True, False, False])
        self.assertTrue(valid["selected_items"])
        for item in valid["selected_items"]:
            self.assertEqual(set(item), {"keep", "practice_option_id", "quantity", "quantity_reason"})
        serialized = json.dumps(valid, ensure_ascii=False)
        for forbidden in ["project_package_id", "stable_sample_id", "source_ref", "family_id", "display_id"]:
            self.assertNotIn(forbidden, serialized)

    def test_historical_plan_generation_uses_original_first_option_when_unspecified(self):
        displays, examples, lookup, valid = self.historical_plan_fixtures()
        unspecified_result = {
            **valid,
            "selected_items": [
                {**valid["selected_items"][0], "practice_option_id": "D001-O02"},
                valid["selected_items"][1],
            ],
        }
        response = types.SimpleNamespace(
            content=unspecified_result, usage={}, raw_content=json.dumps(unspecified_result)
        )

        with patch.object(query_estimate_llm, "request_llm_json_with_usage", return_value=response) as request:
            plan, success, error, _prompt, trace = query_estimate_llm.generate_historical_plan_determination(
                "屋面维修500㎡，未指定厚度",
                examples,
                displays,
                lookup,
                self.historical_option_model(),
                np.asarray([1.0, 0.0], dtype=np.float32),
            )

        self.assertTrue(success)
        self.assertEqual(error, "")
        self.assertEqual(plan.project_package_id, "PKG-1")
        self.assertEqual(plan.items[0].practice_option_id, "D001-O02")
        self.assertEqual(request.call_args.kwargs["max_tokens"], 4096)
        self.assertEqual(trace["max_tokens"], 4096)
        self.assertEqual(json.loads(trace["input_summary"])["historical_display_option_count"], 2)

    def test_historical_plan_parser_rejects_invalid_references_and_shape(self):
        displays, examples, lookup, valid = self.historical_plan_fixtures()
        _prompt, projects, display_options = self.historical_plan_context(
            "屋面维修", examples, displays, lookup
        )
        first = valid["selected_items"][0]
        second = valid["selected_items"][1]
        cases = [
            ({**valid, "extra": True}, "顶层只允许"),
            ({**valid, "project_selections": [True, False]}, "工程数量一致"),
            ({**valid, "project_selections": [1, False, False]}, "仅包含 boolean"),
            ({**valid, "project_selections": [False, False, False]}, "只能选中一个"),
            ({**valid, "project_selections": [True, True, False]}, "只能选中一个"),
            ({**valid, "selected_items": []}, "数量必须等于"),
            ({**valid, "selected_items": [first, second, first]}, "数量必须等于"),
            ({**valid, "selected_items": [{**first, "keep": "yes"}, second]}, "keep 必须为 boolean"),
            ({**valid, "selected_items": [{**first, "practice_option_id": "D002-O01"}, second]}, "不属于该位置"),
            ({**valid, "selected_items": [{**first, "practice_option_id": "D001-NOT-SENT"}, second]}, "不属于该位置"),
            ({**valid, "selected_items": [{**first, "quantity_reason": ""}, second]}, "quantity_reason"),
            ({**valid, "selected_items": [{**first, "extra": True}, second]}, "只允许"),
            ({**valid, "selected_items": [{**first, "quantity": {"type": "exact", "value": -1}}, second]}, "非负"),
            ({**valid, "selected_items": [{"keep": False, "practice_option_id": "D001-O01", "quantity": None, "quantity_reason": ""}, second]}, "删除项必须"),
            ({**valid, "selected_items": [
                {"keep": False, "practice_option_id": "", "quantity": None, "quantity_reason": ""},
                {"keep": False, "practice_option_id": "", "quantity": None, "quantity_reason": ""},
            ]}, "至少必须保留"),
            ({**valid, "selected_items": [second, first]}, "不属于该位置"),
        ]
        for result, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    query_estimate_llm.parse_historical_plan_determination_result(
                        result, projects, display_options, displays, lookup
                    )

        tampered_display_options = {
            **display_options,
            "D001": [
                *display_options["D001"],
                {"practice_option_id": "D002-O01", "practice_description": "错误跨 display 候选"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "不属于该位置"):
            query_estimate_llm.parse_historical_plan_determination_result(
                {**valid, "selected_items": [{**first, "practice_option_id": "D002-O01"}, second]},
                projects,
                tampered_display_options,
                displays,
                lookup,
            )

        displays_with_unsent_option = displays.copy(deep=True)
        displays_with_unsent_option.at[0, "practice_options"] = [
            *displays.iloc[0]["practice_options"],
            {"practice_option_id": "D001-O99", "practice_description": "存在但未传入 prompt 的工艺"},
        ]
        with self.assertRaisesRegex(ValueError, "不属于该位置"):
            query_estimate_llm.parse_historical_plan_determination_result(
                {**valid, "selected_items": [{**first, "practice_option_id": "D001-O99"}, second]},
                projects,
                display_options,
                displays_with_unsent_option,
                lookup,
            )

        second_project_result = {
            "project_selections": [False, True, False],
            "selected_items": [
                {
                    "keep": True,
                    "practice_option_id": "D001-O01",
                    "quantity": {"type": "exact", "value": 1},
                    "quantity_reason": "按输入数组第二个工程及其第一条清单回查",
                }
            ],
        }
        second_project_plan = query_estimate_llm.parse_historical_plan_determination_result(
            second_project_result, projects, display_options, displays, lookup
        )
        self.assertEqual(second_project_plan.project_package_id, "PKG-2")
        self.assertEqual(second_project_plan.items[0].stable_sample_id, "sid-other")

    def test_historical_plan_conservative_selection_preserves_same_object_chain(self):
        item_names = ["防水层拆除", "屋面卷材防水", "屋面涂膜防水", "垂直运输", "措施费", "外墙修补"]
        historical_projects = [
            {
                "project_package_id": "ROOF",
                "project_name": "完整屋面维修",
                "items": [
                    {
                        "stable_sample_id": f"roof-{index}",
                        "display_id": f"D{index:03d}",
                        "practice_option_id": f"D{index:03d}-O01",
                        "cost_item_name": name,
                        "project_description": name,
                        "unit": "项",
                        "quantity": 1,
                    }
                    for index, name in enumerate(item_names, start=1)
                ],
            },
            {"project_package_id": "OTHER-1", "project_name": "其他一", "items": []},
            {"project_package_id": "OTHER-2", "project_name": "其他二", "items": []},
        ]
        lookup = {
            f"roof-{index}": {
                "project_package_id": "ROOF",
                "source_ref": f"ref-{index}",
                "display_id": f"D{index:03d}",
                "practice_option_id": f"D{index:03d}-O01",
            }
            for index in range(1, 7)
        }
        result = {
            "project_selections": [True, False, False],
            "selected_items": [
                {
                    "keep": True,
                    "practice_option_id": f"D{index:03d}-O01",
                    "quantity": {"type": "exact", "value": 1},
                    "quantity_reason": "同一屋面维修对象的完整施工链",
                }
                for index in range(1, 6)
            ] + [
                {
                    "keep": False,
                    "practice_option_id": "",
                    "quantity": None,
                    "quantity_reason": "",
                }
            ],
        }
        displays = pd.DataFrame(
            [
                {
                    "display_id": f"D{index:03d}",
                    "practice_options": [
                        {"practice_option_id": f"D{index:03d}-O01", "practice_description": name}
                    ],
                }
                for index, name in enumerate(item_names, start=1)
            ]
        )

        plan = query_estimate_llm.parse_historical_plan_determination_result(
            result, historical_projects, {}, displays, lookup
        )

        self.assertEqual(
            [item.stable_sample_id for item in plan.items],
            ["roof-1", "roof-2", "roof-3", "roof-4", "roof-5"],
        )
        self.assertNotIn("roof-6", {item.stable_sample_id for item in plan.items})

    def test_stable_sample_lookup_is_unique_and_allows_duplicate_source_ref(self):
        samples = pd.DataFrame([{"stable_sample_id": "sid-1"}, {"stable_sample_id": "sid-2"}])
        evidence = pd.DataFrame(
            [
                {"stable_sample_id": "sid-1", "source_ref": "duplicate", "family_id": "F001", "project_package_id": "P1"},
                {"stable_sample_id": "sid-2", "source_ref": "duplicate", "family_id": "F002", "project_package_id": "P1"},
            ]
        )
        display_families = pd.DataFrame(
            [{"display_id": "D001", "family_id": "F001"}, {"display_id": "D001", "family_id": "F002"}]
        )
        displays = pd.DataFrame(
            [{"display_id": "D001", "practice_options": [{"practice_option_id": "D001-O01", "family_ids": ["F001", "F002"]}]}]
        )

        lookup = query_estimate_llm.build_stable_sample_lookup(samples, evidence, display_families, displays)

        self.assertEqual(set(lookup), {"sid-1", "sid-2"})
        self.assertEqual(lookup["sid-1"]["source_ref"], lookup["sid-2"]["source_ref"])
        duplicate_samples = pd.concat([samples, samples.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "全局唯一"):
            query_estimate_llm.build_stable_sample_lookup(duplicate_samples, evidence, display_families, displays)

    def test_scenario_outputs_use_full_option_evidence_and_preserve_trace_ids(self):
        displays, examples, lookup, valid = self.historical_plan_fixtures()
        families = pd.DataFrame(
            [{"family_id": "F001"}, {"family_id": "F002"}, {"family_id": "F003"}],
            columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS,
        )
        evidence = pd.DataFrame(
            [
                {"family_id": "F001", "source_ref": "skeleton", "project_package_id": "PKG-1", "unit_price": 120, "labor_unit_price": 20, "machinery_unit_price": 2},
                {"family_id": "F001", "source_ref": "outside-top3", "project_package_id": "PKG-9", "unit_price": 80, "labor_unit_price": 10, "machinery_unit_price": 1},
            ],
            columns=query_estimate_llm.EVIDENCE_ITEM_COLUMNS,
        )
        _prompt, projects, display_options = self.historical_plan_context(
            "屋面漏水，想做3mm SBS防水，面积大概500平", examples, displays, lookup
        )
        plan = query_estimate_llm.parse_historical_plan_determination_result(
            valid, projects, display_options, displays, lookup
        )
        scenario = query_estimate_llm.EstimateScenario("S001", 1, "屋面方案", "说明", [plan.items[0]])

        output = query_estimate_llm.build_scenario_outputs([scenario], displays, families, evidence)

        row = output.iloc[0]
        self.assertEqual(row["project_package_id"], "PKG-1")
        self.assertEqual(row["stable_sample_id"], "sid-main")
        self.assertEqual(row["source_ref"], "same-ref")
        self.assertEqual(row["工程量依据"], "用户明确屋面维修面积约500㎡并指定3mm SBS")
        self.assertEqual(row["综合单价最低值"], 80.0)
        self.assertEqual(row["综合单价最高值"], 120.0)
        self.assertEqual(row["价格证据样本数"], 2)
        self.assertEqual(row["合价最低值"], 40000.0)

    def test_scenario_outputs_reject_missing_main_price_evidence(self):
        displays, _examples, _lookup, _valid = self.historical_plan_fixtures()
        families = pd.DataFrame([{"family_id": "F001"}], columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS)
        item = query_estimate_llm.ScenarioItem(
            "PKG-1", "sid-main", "ref", "D001", "D001-O01", "", {"type": "exact", "value": 1}, "依据"
        )
        scenario = query_estimate_llm.EstimateScenario("S001", 1, "", "", [item])
        with self.assertRaisesRegex(ValueError, "价格回查失败"):
            query_estimate_llm.build_scenario_outputs(
                [scenario], displays, families, pd.DataFrame(columns=query_estimate_llm.EVIDENCE_ITEM_COLUMNS)
            )

    def test_final_explanation_uses_order_and_requires_exact_item_count(self):
        displays, examples, lookup, valid = self.historical_plan_fixtures()
        _prompt, projects, display_options = self.historical_plan_context(
            "屋面维修", examples, displays, lookup
        )
        plan = query_estimate_llm.parse_historical_plan_determination_result(
            valid, projects, display_options, displays, lookup
        )
        scenario = query_estimate_llm.scenario_from_historical_plan(plan)
        result = {
            "scenario_name": "屋面维修方案",
            "scenario_summary": "以真实历史屋面工程为骨架。",
            "item_explanations": [
                {"item_explanation": "主防水层。"},
                {"item_explanation": "按现场确定拆除范围。"},
            ],
        }

        explained = query_estimate_llm.parse_final_explanation_result(result, scenario)

        self.assertEqual(explained.scenario_id, "S001")
        self.assertEqual(explained.items[0].selection_reason, "主防水层。")
        self.assertEqual(explained.items[1].selection_reason, "按现场确定拆除范围。")
        invalid = [
            ({**result, "item_explanations": result["item_explanations"][:1]}, "数量必须等于"),
            ({**result, "item_explanations": result["item_explanations"] + [{"item_explanation": "x"}]}, "数量必须等于"),
            ({**result, "item_explanations": [{"item_explanation": "第一项", "extra": True}, result["item_explanations"][1]]}, "只允许"),
            ({**result, "item_explanations": [{"item_explanation": ""}, result["item_explanations"][1]]}, "不得为空"),
            ({**result, "extra": True}, "顶层字段非法"),
        ]
        for payload, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    query_estimate_llm.parse_final_explanation_result(payload, scenario)

    def test_final_explanation_prompt_contains_only_final_public_items(self):
        displays, examples, lookup, valid = self.historical_plan_fixtures()
        examples[0]["items"][1]["cost_item_name"] = "已删除的其他维修对象"
        _prompt, projects, display_options = self.historical_plan_context(
            "当前对象维修", examples, displays, lookup
        )
        selected_result = {
            **valid,
            "selected_items": [
                valid["selected_items"][0],
                {"keep": False, "practice_option_id": "", "quantity": None, "quantity_reason": ""},
            ],
        }
        plan = query_estimate_llm.parse_historical_plan_determination_result(
            selected_result, projects, display_options, displays, lookup
        )
        scenario = query_estimate_llm.scenario_from_historical_plan(plan)
        priced = pd.DataFrame(
            [{column: "" for column in query_estimate_llm.ESTIMATE_SCENARIO_COLUMNS}]
        )
        priced.loc[0, ["清单名称", "选用工艺", "单位", "工程量预估", "工程量依据"]] = [
            "最终保留清单", "3mm SBS", "m²", "500", "用户明确面积",
        ]
        prompt = query_estimate_llm.build_final_explanation_prompt(
            "当前对象维修", scenario, priced, examples
        )

        self.assertIn('"selected_project_name": "历史屋面工程"', prompt)
        self.assertIn("最终保留清单", prompt)
        self.assertNotIn("已删除的其他维修对象", prompt)
        self.assertNotIn("selected_historical_project", prompt)
        for forbidden in [
            "project_package_id", "stable_sample_id", "source_ref", "family_id", "display_id",
            "practice_option_id",
        ]:
            self.assertNotIn(forbidden, prompt)
        current_payload = prompt.split("输入：\n", 1)[1]
        legacy_payload = query_estimate_llm.json_text({
            "user_query": "当前对象维修",
            "selected_historical_project": examples[0],
            "final_items": [{
                **priced.iloc[0].to_dict(),
                "stable_sample_id": "sid-main-with-a-deliberately-long-hash",
                "source_ref": "source-with-a-deliberately-long-reference",
                "family_id": "F001",
                "display_id": "D001",
            }],
        })
        self.assertLess(
            query_estimate_llm.estimated_tokens(current_payload),
            query_estimate_llm.estimated_tokens(legacy_payload),
        )

    def test_final_explanation_failure_keeps_blank_scenario_and_records_trace(self):
        displays, examples, lookup, valid = self.historical_plan_fixtures()
        _prompt, projects, display_options = self.historical_plan_context(
            "屋面维修", examples, displays, lookup
        )
        plan = query_estimate_llm.parse_historical_plan_determination_result(
            valid, projects, display_options, displays, lookup
        )
        scenario = query_estimate_llm.scenario_from_historical_plan(plan)
        priced = pd.DataFrame(
            [
                {column: ("sid-main" if column == "stable_sample_id" else "") for column in query_estimate_llm.ESTIMATE_SCENARIO_COLUMNS},
                {column: ("sid-remove" if column == "stable_sample_id" else "") for column in query_estimate_llm.ESTIMATE_SCENARIO_COLUMNS},
            ]
        )
        with patch.object(query_estimate_llm, "request_llm_json_with_usage", side_effect=RuntimeError("LLM down")):
            output, success, error, _prompt, trace = query_estimate_llm.generate_final_explanation(
                "屋面维修", scenario, priced, examples
            )

        self.assertFalse(success)
        self.assertIn("LLM down", error)
        self.assertEqual(output.scenario_name, "")
        self.assertTrue(all(item.selection_reason == "" for item in output.items))
        self.assertEqual(trace["stage"], "final_explanation")
        self.assertEqual(trace["parsed_status"], "failed")

    def test_quantity_validation(self):
        self.assertEqual(query_estimate_llm.validate_quantity({"type": "exact", "value": 10}), {"type": "exact", "value": 10.0})
        self.assertEqual(query_estimate_llm.validate_quantity({"type": "range", "min": 0, "max": 10}), {"type": "range", "min": 0.0, "max": 10.0})
        for quantity in [
            {"type": "exact", "value": -1},
            {"type": "range", "min": 20, "max": 10},
            {"type": "unknown"},
        ]:
            with self.assertRaises(ValueError):
                query_estimate_llm.validate_quantity(quantity)

    def test_parse_info_records_two_stage_status_and_selected_project(self):
        trace = {"prompt_chars": 10, "prompt_tokens": 4, "completion_tokens": 2}
        parse_info = query_estimate_llm.build_parse_info(
            rewrite=query_estimate_llm.QueryRewrite("屋面", "屋面工程", "屋面防水", [], True),
            top_packages=20,
            top_items=300,
            max_packages_per_cache_subject=1,
            package_weight_temperature=0.1,
            evidence_package_universe_count=20,
            package_evidence_weight_count=20,
            package_evidence_weight_sum=1.0,
            meta={},
            sample_count=100,
            package_count=10,
            retrieved_evidence_item_row_count=50,
            evidence_item_row_count=50,
            candidate_family_count=8,
            candidate_display_group_count=4,
            matched_project_example_count=3,
            matched_project_example_item_count=30,
            display_option_grouping_display_count=4,
            display_option_grouping_trace=trace,
            display_option_grouping_fallback=False,
            display_option_grouping_error="",
            display_option_grouping_meta={},
            selected_project_package_id="PKG-1",
            selected_item_count=2,
            selected_exact_quantity_count=1,
            selected_range_quantity_count=1,
            historical_plan_trace=trace,
            historical_plan_error="",
            final_explanation_trace=trace,
            final_explanation_error="invalid explanation",
            output_path=Path("query.xlsx"),
            started_at=query_estimate_llm.datetime.now(),
            index_dir=Path("embeddings"),
            include_debug_text=True,
            display_option_grouping_prompt="grouping",
            historical_plan_prompt="historical",
            final_explanation_prompt="explanation",
            warnings=["final_explanation_failed"],
        )
        values = dict(parse_info.values.tolist())

        self.assertEqual(values["historical_plan_input_project_count"], 3)
        self.assertEqual(values["historical_plan_input_item_count"], 30)
        self.assertEqual(values["selected_project_package_id"], "PKG-1")
        self.assertEqual(values["historical_plan_determination_status"], "success")
        self.assertEqual(values["final_explanation_status"], "failed")
        self.assertIn("invalid explanation", values["final_explanation LLM error"])
        self.assertEqual(values["historical_plan_determination_prompt_preview"], "historical")

    def test_workbook_includes_new_columns_and_two_stage_trace(self):
        result = query_estimate_llm.QueryResult(
            rewrite=query_estimate_llm.QueryRewrite("屋面", "屋面工程", "屋面防水", [], True),
            estimate_summary=pd.DataFrame(columns=query_estimate_llm.ESTIMATE_SUMMARY_COLUMNS),
            estimate_scenarios=pd.DataFrame(columns=query_estimate_llm.ESTIMATE_SCENARIO_COLUMNS),
            matched_project_packages=pd.DataFrame(columns=query_estimate_llm.MATCHED_PROJECT_PACKAGE_COLUMNS),
            candidate_families=pd.DataFrame(columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS),
            candidate_display_groups=pd.DataFrame(columns=query_estimate_llm.CANDIDATE_DISPLAY_GROUP_COLUMNS),
            package_evidence_weights=pd.DataFrame(columns=query_estimate_llm.PACKAGE_EVIDENCE_WEIGHT_COLUMNS),
            display_group_families=pd.DataFrame(columns=query_estimate_llm.DISPLAY_GROUP_FAMILY_COLUMNS),
            display_option_grouping_trace=pd.DataFrame(columns=query_estimate_llm.DISPLAY_OPTION_GROUPING_TRACE_COLUMNS),
            matched_project_examples=pd.DataFrame(columns=query_estimate_llm.MATCHED_PROJECT_EXAMPLE_COLUMNS),
            evidence_items=pd.DataFrame(columns=query_estimate_llm.EVIDENCE_ITEM_COLUMNS),
            parse_info=pd.DataFrame([{"字段": "final_explanation_status", "值": "failed"}]),
            llm_trace=pd.DataFrame(
                [{"stage": stage} for stage in ["query_rewrite_for_embedding", "display_option_grouping", "historical_plan_determination", "final_explanation"]],
                columns=query_estimate_llm.LLM_TRACE_COLUMNS,
            ),
            success=False,
            error_message="explanation failed",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.xlsx"
            query_estimate_llm.write_query_result_workbook(path, result)
            workbook = openpyxl.load_workbook(path, data_only=True)
            scenario_headers = [cell.value for cell in workbook["estimate_scenarios"][1]]
            trace_stages = [workbook["llm_trace"].cell(row=row, column=1).value for row in range(2, 6)]
            workbook.close()

        self.assertEqual(
            scenario_headers,
            [
                "方案顺序", "方案编号", "方案名称", "project_package_id", "stable_sample_id", "source_ref",
                "display_id", "清单名称", "选用工艺", "其他可选工艺", "单位", "项目说明", "工程量预估",
                "工程量依据", "合价最低值", "合价中位数", "合价最高值", "综合单价最低值", "综合单价中位数",
                "综合单价最高值", "其中包含人工费单价最低值", "其中包含人工费单价中位数",
                "其中包含人工费单价最高值", "其中包含机械费单价最低值", "其中包含机械费单价中位数",
                "其中包含机械费单价最高值", "价格证据样本数", "来源样本", "practice_option_id", "价格证据family",
            ],
        )
        self.assertEqual(trace_stages, ["query_rewrite_for_embedding", "display_option_grouping", "historical_plan_determination", "final_explanation"])

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
