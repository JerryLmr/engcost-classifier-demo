from __future__ import annotations

import csv
import importlib.util
import json
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

        examples = query_estimate_llm.build_matched_project_examples(matched, samples)

        self.assertEqual([example["project_package_id"] for example in examples], ["P1", "P2", "P3"])
        self.assertEqual([item["cost_item_name"] for item in examples[0]["items"]], ["第一项", "第二项", "第十项"])
        self.assertEqual([item["cost_item_name"] for item in examples[1]["items"]], ["P2-1", "P2-2"])
        self.assertEqual(sum(len(example["items"]) for example in examples), 6)

    def test_compact_matched_project_examples_for_scenario_preserves_full_debug_data(self):
        examples = [
            {
                "rank": 1,
                "project_package_id": "P1",
                "project_name": "",
                "project_name_text": "历史消防工程",
                "consultation_time": "2025-01",
                "location": "上海",
                "items": [
                    {
                        "cost_item_name": "报警主机",
                        "project_description": "更换主机",
                        "unit": "台",
                        "quantity": 1,
                        "unit_price": 1000,
                        "total_price": 1000,
                        "project_code": "X1",
                        "family_id": "F001",
                        "display_id": "D001",
                    },
                    {
                        "cost_item_name": "系统调试",
                        "project_description": "联动调试",
                        "unit": "项",
                        "quantity": 1,
                        "unit_price": 200,
                        "total_price": 200,
                    },
                ],
            }
        ]

        compact = query_estimate_llm.compact_matched_project_examples_for_scenario(examples)
        debug_frame = query_estimate_llm.matched_project_examples_frame(examples)

        self.assertEqual(set(compact[0]), {"project_name", "items"})
        self.assertEqual(compact[0]["project_name"], "历史消防工程")
        self.assertEqual(len(compact[0]["items"]), 2)
        self.assertEqual(
            set(compact[0]["items"][0]),
            {"cost_item_name", "project_description", "unit", "quantity"},
        )
        self.assertEqual(examples[0]["rank"], 1)
        self.assertEqual(examples[0]["items"][0]["unit_price"], 1000)
        self.assertEqual(debug_frame.loc[0, "unit_price"], 1000)
        self.assertEqual(debug_frame.loc[0, "total_price"], 1000)

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

    def test_scenario_generation_prompt_uses_only_display_options(self):
        examples = [{"project_name": "历史屋面维修", "items": [{"cost_item_name": "基层处理", "project_description": "清理基层", "unit": "m²", "quantity": 100}]}]
        prompt, records = query_estimate_llm.build_scenario_generation_prompt(
            "屋面漏水 100平", self.displays_with_options_fixture(), examples
        )

        self.assertEqual(set(records[0]), {"display_id", "display_name", "unit", "practice_options"})
        self.assertEqual(set(records[0]["practice_options"][0]), {"practice_option_id", "practice_description"})
        self.assertNotIn("sample_count", prompt)
        self.assertIn("scenario_summary", prompt)
        self.assertIn("item_explanation", prompt)
        self.assertNotIn("include", prompt)
        self.assertNotIn("amount", prompt)
        self.assertNotIn("unknown", prompt)
        self.assertIn("存在同一施工范围继承关系时，应在 item_explanation 中说明", prompt)
        self.assertNotIn("project_package_query_text", prompt)
        self.assertNotIn("item_query_text", prompt)
        self.assertNotIn("family_ids", prompt)
        self.assertNotIn("unit_price", prompt)
        self.assertNotIn("suggested_quantity", prompt)
        self.assertIn("matched_project_examples", prompt)
        self.assertIn("历史屋面维修", prompt)
        self.assertNotIn("已选 display", prompt)

    def test_select_scenario_displays_applies_limit_and_support_rank_order(self):
        displays = pd.DataFrame(
            [
                {
                    "display_id": f"D{index:03d}",
                    "display_name": f"清单{index}",
                    "practice_options": [],
                }
                for index in range(1, 66)
            ]
        )
        groups = pd.DataFrame(
            [
                {"display_id": f"D{index:03d}", "support_rank": 66 - index}
                for index in range(1, 66)
            ]
        )

        selected = query_estimate_llm.select_scenario_displays(displays, groups)

        self.assertEqual(len(selected), query_estimate_llm.SCENARIO_DISPLAY_LIMIT)
        self.assertEqual(selected["display_id"].tolist(), [f"D{index:03d}" for index in range(65, 5, -1)])

    def test_select_scenario_displays_keeps_all_when_below_limit_and_stable_ties(self):
        displays = pd.DataFrame(
            [
                {"display_id": "D001", "practice_options": []},
                {"display_id": "D002", "practice_options": []},
                {"display_id": "D003", "practice_options": []},
            ]
        )
        groups = pd.DataFrame(
            [
                {"display_id": "D003", "support_rank": 1},
                {"display_id": "D001", "support_rank": 2},
                {"display_id": "D002", "support_rank": 2},
            ]
        )

        selected = query_estimate_llm.select_scenario_displays(displays, groups)

        self.assertEqual(selected["display_id"].tolist(), ["D003", "D001", "D002"])

    def valid_scenario_result(self):
        return {
            "scenarios": [
                {
                    "scenario_id": "S001",
                    "scenario_order": 1,
                    "scenario_name": "卷材方案",
                    "scenario_summary": "采用卷材防水并包含拆除。",
                    "items": [
                        {
                            "display_id": "D001",
                            "practice_option_id": "D001-O01",
                            "item_explanation": "该项用于屋面防水层施工，位于基层处理后的主防水环节；按用户给出的同一施工面积暂估，最终以现场核定为准；纳入方案并计入金额。",
                            "quantity": {"type": "exact", "value": 100},
                        },
                        {
                            "display_id": "D002",
                            "practice_option_id": "D002-O01",
                            "item_explanation": "该项用于拆除原防水层，属于新做防水前置工序；因原层范围和厚度待现场确认，纳入方案但暂不计价。",
                            "quantity": {"type": "range", "min": 0, "max": 100},
                        },
                    ],
                }
            ]
        }

    def test_scenario_parser_accepts_valid_result(self):
        scenarios = query_estimate_llm.parse_scenario_generation_result(self.valid_scenario_result(), self.displays_with_options_fixture())

        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0].scenario_id, "S001")
        self.assertEqual(scenarios[0].scenario_order, 1)
        self.assertEqual(scenarios[0].scenario_summary, "采用卷材防水并包含拆除。")
        self.assertEqual(scenarios[0].items[0].quantity, {"type": "exact", "value": 100.0})

    def test_scenario_parser_only_accepts_selected_scenario_displays(self):
        scenario_displays = self.displays_with_options_fixture().iloc[[0]].copy()
        result = self.valid_scenario_result()

        with self.assertRaisesRegex(ValueError, "无效 display_id: D002"):
            query_estimate_llm.parse_scenario_generation_result(result, scenario_displays)

    def test_scenario_generation_trace_records_compact_input_counts_without_display_ids(self):
        examples = [
            {
                "project_name": "历史项目",
                "items": [
                    {"cost_item_name": "基层处理", "project_description": "处理", "unit": "m²", "quantity": 100},
                    {"cost_item_name": "防水施工", "project_description": "施工", "unit": "m²", "quantity": 100},
                ],
            }
        ]
        response = types.SimpleNamespace(content=self.valid_scenario_result(), usage={}, raw_content="{}")

        with patch.object(query_estimate_llm, "request_llm_json_with_usage", return_value=response):
            scenarios, success, *_rest, trace = query_estimate_llm.generate_estimate_scenarios(
                "屋面漏水 100平",
                self.displays_with_options_fixture(),
                examples,
                all_candidate_display_count=75,
            )

        input_summary = json.loads(trace["input_summary"])
        self.assertTrue(success)
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(input_summary["all_candidate_display_count"], 75)
        self.assertEqual(input_summary["scenario_display_limit"], 60)
        self.assertEqual(input_summary["scenario_display_count"], 2)
        self.assertEqual(input_summary["scenario_practice_option_count"], 3)
        self.assertEqual(input_summary["matched_project_example_count"], 1)
        self.assertEqual(input_summary["matched_project_item_count"], 2)
        self.assertNotIn("display_ids", input_summary)

    def test_scenario_parser_rejects_invalid_references_and_fields(self):
        displays_with_options = self.displays_with_options_fixture()

        invalid_cases = [
            ({"scenarios": [{**self.valid_scenario_result()["scenarios"][0], "scenario_id": "S001"}, {**self.valid_scenario_result()["scenarios"][0], "scenario_order": 2}]}, "scenario_id 重复"),
            ({"scenarios": [{**self.valid_scenario_result()["scenarios"][0], "scenario_order": 2}]}, "scenario_order"),
            ({"scenarios": [{**self.valid_scenario_result()["scenarios"][0], "scenario_summary": ""}]}, "scenario_summary"),
            ({"scenarios": [{**self.valid_scenario_result()["scenarios"][0], "items": []}]}, "至少包含一个 item"),
            ({"scenarios": [{**self.valid_scenario_result()["scenarios"][0], "items": [{**self.valid_scenario_result()["scenarios"][0]["items"][0], "display_id": "BAD"}]}]}, "无效 display_id"),
            ({"scenarios": [{**self.valid_scenario_result()["scenarios"][0], "items": [{**self.valid_scenario_result()["scenarios"][0]["items"][0], "practice_option_id": "D001-BAD"}]}]}, "不属于对应 display"),
            ({"scenarios": [{**self.valid_scenario_result()["scenarios"][0], "items": [{**self.valid_scenario_result()["scenarios"][0]["items"][0], "include": True}]}]}, "未要求字段"),
            ({"scenarios": [{**self.valid_scenario_result()["scenarios"][0], "items": [{**self.valid_scenario_result()["scenarios"][0]["items"][0], "amount": True}]}]}, "未要求字段"),
            ({"scenarios": [{**self.valid_scenario_result()["scenarios"][0], "items": [{**self.valid_scenario_result()["scenarios"][0]["items"][0], "item_explanation": ""}]}]}, "item_explanation"),
            ({"scenarios": [{**self.valid_scenario_result()["scenarios"][0], "items": [{**self.valid_scenario_result()["scenarios"][0]["items"][0], "unit_price": 100}]}]}, "未要求字段"),
        ]
        for result, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    query_estimate_llm.parse_scenario_generation_result(result, displays_with_options)

    def test_quantity_validation(self):
        self.assertEqual(query_estimate_llm.validate_quantity({"type": "exact", "value": 10}), {"type": "exact", "value": 10.0})
        self.assertEqual(query_estimate_llm.validate_quantity({"type": "range", "min": 5, "max": 10}), {"type": "range", "min": 5.0, "max": 10.0})
        self.assertEqual(query_estimate_llm.validate_quantity({"type": "range", "min": 0, "max": 10}), {"type": "range", "min": 0.0, "max": 10.0})

        invalid_quantities = [
            ({"type": "exact"}, "exact"),
            ({"type": "range", "min": 10}, "range"),
            ({"type": "range", "min": 20, "max": 10}, "不得大于"),
            ({"type": "unknown"}, "exact 或 range"),
            ({"type": "include"}, "exact 或 range"),
            ({"type": "amount"}, "exact 或 range"),
            ({"type": "exact", "value": -1}, "非负"),
            ({"type": "range", "min": -1, "max": 10}, "非负"),
        ]
        for quantity, message in invalid_quantities:
            with self.subTest(quantity=quantity):
                with self.assertRaisesRegex(ValueError, message):
                    query_estimate_llm.validate_quantity(quantity)

    def test_scenario_outputs_backfill_prices_and_calculate_amounts(self):
        displays_with_options = self.displays_with_options_fixture()
        families = pd.DataFrame(
            [
                {"family_id": "F001", "本次召回综合单价最低值": 80, "本次召回综合单价中位数": 100, "本次召回综合单价最高值": 120},
                {"family_id": "F002", "本次召回综合单价最低值": 90, "本次召回综合单价中位数": 110, "本次召回综合单价最高值": 140},
                {"family_id": "F003", "本次召回综合单价最低值": 50, "本次召回综合单价中位数": 60, "本次召回综合单价最高值": 70},
                {
                    "family_id": "F004",
                    "本次召回综合单价最低值": 10,
                    "本次召回综合单价中位数": 20,
                    "本次召回综合单价最高值": 30,
                    "source_refs": ["fallback-ref"],
                },
            ],
            columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS,
        )
        evidence_items = pd.DataFrame(
            [
                {"family_id": "F001", "source_ref": "a", "unit_price": 80, "labor_unit_price": 10, "machinery_unit_price": 1},
                {"family_id": "F002", "source_ref": "b", "unit_price": 140, "labor_unit_price": 30, "machinery_unit_price": 5},
                {"family_id": "F004", "source_ref": "c", "unit_price": 20},
            ],
            columns=query_estimate_llm.EVIDENCE_ITEM_COLUMNS,
        )
        scenarios = query_estimate_llm.parse_scenario_generation_result(
            {
                "scenarios": [
                    {
                        "scenario_id": "S001",
                        "scenario_order": 1,
                        "scenario_name": "卷材方案",
                        "scenario_summary": "卷材方案包含拆除确认和3mm SBS主防水施工，适用于屋面基层可处理后铺贴卷材的现场；与涂膜方案差异在主材做法，防水已计价，拆除仅展示待确认。",
                        "items": [
                            {
                                "display_id": "D001",
                                "practice_option_id": "D001-O01",
                                "item_explanation": "用于屋面主防水层施工，按用户给出的同一施工面积暂估，最终以现场核定为准；纳入方案并计入金额。",
                                "quantity": {"type": "exact", "value": 10},
                            },
                            {
                                "display_id": "D002",
                                "practice_option_id": "D002-O01",
                                "item_explanation": "用于拆除原防水层，属于前置工序；原防水层范围和厚度待现场确认，纳入方案但暂不计价。",
                                "quantity": {"type": "range", "min": 0, "max": 10},
                            },
                        ],
                    },
                    {
                        "scenario_id": "S002",
                        "scenario_order": 2,
                        "scenario_name": "涂膜方案",
                        "scenario_summary": "涂膜方案不设置拆除项，采用聚氨酯涂膜作为主防水，适用于细部节点多且基层具备涂刷条件的现场；与卷材方案差异在材料做法和施工方式，主防水已计价。",
                        "items": [
                            {
                                "display_id": "D001",
                                "practice_option_id": "D001-O02",
                                "item_explanation": "用于屋面主防水层施工，作为卷材之外的涂膜做法；工程量按区间暂估，纳入方案并计入金额。",
                                "quantity": {"type": "range", "min": 5, "max": 15},
                            }
                        ],
                    },
                ]
            },
            displays_with_options,
        )

        estimate_scenarios = query_estimate_llm.build_scenario_outputs(scenarios, displays_with_options, families, evidence_items)
        summary = query_estimate_llm.build_estimate_summary(scenarios, estimate_scenarios)

        first = estimate_scenarios.iloc[0]
        self.assertEqual(estimate_scenarios.columns.tolist(), query_estimate_llm.ESTIMATE_SCENARIO_COLUMNS)
        self.assertEqual(first["方案编号"], "S001")
        self.assertEqual(first["清单名称"], "屋面防水")
        self.assertEqual(first["项目说明"], "用于屋面主防水层施工，按用户给出的同一施工面积暂估，最终以现场核定为准；纳入方案并计入金额。")
        self.assertEqual(first["工程量预估"], 10.0)
        self.assertEqual(first["综合单价最低值"], 80.0)
        self.assertEqual(first["综合单价最高值"], 140.0)
        self.assertEqual(first["其中包含人工费单价最低值"], 10.0)
        self.assertEqual(first["其中包含人工费单价中位数"], 20.0)
        self.assertEqual(first["其中包含人工费单价最高值"], 30.0)
        self.assertEqual(first["其中包含机械费单价最低值"], 1.0)
        self.assertEqual(first["其中包含机械费单价中位数"], 3.0)
        self.assertEqual(first["其中包含机械费单价最高值"], 5.0)
        self.assertEqual(first["价格证据样本数"], 2)
        self.assertEqual(first["来源样本"], "a, b")
        self.assertEqual(first["价格证据family"], "F001,F002")
        self.assertEqual(first["合价最低值"], 800.0)
        self.assertEqual(first["合价中位数"], 1100.0)
        self.assertEqual(first["合价最高值"], 1400.0)
        self.assertEqual(estimate_scenarios.iloc[1]["合价最低值"], 0.0)
        self.assertEqual(estimate_scenarios.iloc[1]["合价中位数"], 100.0)
        self.assertEqual(estimate_scenarios.iloc[1]["合价最高值"], 200.0)
        self.assertEqual(estimate_scenarios.iloc[1]["综合单价中位数"], 20.0)
        self.assertEqual(estimate_scenarios.iloc[1]["其中包含人工费单价最低值"], "")
        self.assertEqual(estimate_scenarios.iloc[1]["其中包含机械费单价最低值"], "")
        range_row = estimate_scenarios[estimate_scenarios["方案编号"] == "S002"].iloc[0]
        self.assertEqual(estimate_scenarios.iloc[1]["工程量预估"], "0～10")
        self.assertEqual(range_row["工程量预估"], "5～15")
        self.assertEqual(range_row["合价最低值"], 250.0)
        self.assertEqual(range_row["合价中位数"], 600.0)
        self.assertEqual(range_row["合价最高值"], 1050.0)
        self.assertEqual(summary["方案编号"].tolist(), ["S001", "S002"])
        self.assertEqual(summary.loc[0, "是否推荐方案"], "是")
        self.assertIn("屋面防水（3mm SBS卷材防水）", summary.loc[0, "主要施工内容"])
        self.assertIn("防水层拆除（拆除原防水层）", summary.loc[0, "主要施工内容"])
        self.assertEqual(summary.loc[0, "计价项目数"], 2)
        self.assertNotIn("展示但未计价项目数", summary.columns)
        self.assertNotIn("是否纳入方案", estimate_scenarios.columns)
        self.assertNotIn("是否计入金额", estimate_scenarios.columns)
        self.assertEqual(summary.loc[0, "合价最低值"], 800.0)
        self.assertIn("原防水层范围和厚度待现场确认", summary.loc[0, "待现场确认事项"])
        for removed_column in ["工程量类型", "工程量最低值", "工程量中位数", "工程量最高值", "工程量依据"]:
            self.assertNotIn(removed_column, estimate_scenarios.columns)

    def test_quantity_values_keeps_internal_three_value_calculation(self):
        self.assertEqual(query_estimate_llm.quantity_values({"type": "exact", "value": 10}), (10, 10, 10))
        self.assertEqual(query_estimate_llm.quantity_values({"type": "range", "min": 5, "max": 15}), (5.0, 10.0, 15.0))

    def test_build_parse_info_includes_scenario_metrics(self):
        rewrite = query_estimate_llm.QueryRewrite("屋面", "屋面工程", "屋面防水", [], True)

        parse_info = query_estimate_llm.build_parse_info(
            rewrite=rewrite,
            top_packages=10,
            top_items=20,
            max_packages_per_cache_subject=3,
            package_weight_temperature=0.1,
            evidence_package_universe_count=5,
            package_evidence_weight_count=5,
            package_evidence_weight_sum=1.0,
            meta={},
            sample_count=100,
            package_count=10,
            retrieved_evidence_item_row_count=50,
            evidence_item_row_count=50,
            candidate_family_count=12,
            candidate_display_group_count=7,
            matched_project_example_count=3,
            matched_project_example_item_count=79,
            display_option_grouping_display_count=7,
            display_option_grouping_trace={"prompt_chars": 80, "prompt_tokens": 30, "completion_tokens": 6},
            display_option_grouping_fallback=False,
            display_option_grouping_error="",
            display_option_grouping_meta={
                "display_ids": ["D001", "D002"],
                "llm_display_count": 1,
                "programmatic_single_family_display_count": 1,
                "practice_option_count": 2,
                "families_grouped_count": 4,
                "option_count_by_display": {"D001": 2},
                "max_options_per_display": 2,
            },
            scenario_count=2,
            scenario_item_count=3,
            scenario_exact_quantity_count=1,
            scenario_range_quantity_count=2,
            scenario_input_display_count=7,
            scenario_input_practice_option_count=12,
            scenario_input_historical_project_count=3,
            scenario_input_historical_item_count=79,
            scenario_generation_trace={"prompt_chars": 120, "prompt_tokens": 50, "completion_tokens": 10},
            scenario_generation_fallback=False,
            scenario_generation_error="",
            output_path=None,
            started_at=query_estimate_llm.datetime.now(),
            index_dir=Path("query_index"),
            include_debug_text=False,
            display_option_grouping_prompt="display option prompt",
            scenario_generation_prompt="scenario prompt",
            warnings=[],
        )
        values = dict(parse_info.values.tolist())

        self.assertEqual(values["display_option_grouping_display_ids"], '["D001", "D002"]')
        self.assertEqual(values["matched_project_example_item_count"], 79)
        self.assertEqual(values["scenario_count"], 2)
        self.assertEqual(values["scenario_item_count"], 3)
        self.assertEqual(values["scenario_input_display_count"], 7)
        self.assertEqual(values["scenario_input_practice_option_count"], 12)
        self.assertEqual(values["scenario_input_historical_project_count"], 3)
        self.assertEqual(values["scenario_input_historical_item_count"], 79)
        self.assertNotIn("scenario_included_item_count", values)
        self.assertNotIn("scenario_amount_item_count", values)
        self.assertNotIn("scenario_unknown_quantity_count", values)
        self.assertEqual(values["scenario_generation_status"], "success")
        self.assertEqual(values["scenario_generation_prompt_tokens"], 50)
        self.assertFalse(any("quantity_decision" in cell_text for cell_text in values))

    def test_write_query_result_workbook_has_expected_sheets(self):
        rewrite = query_estimate_llm.QueryRewrite("屋面", "屋面工程", "屋面防水", [], True)
        result = query_estimate_llm.QueryResult(
            rewrite=rewrite,
            estimate_summary=pd.DataFrame(columns=query_estimate_llm.ESTIMATE_SUMMARY_COLUMNS),
            estimate_scenarios=pd.DataFrame(columns=query_estimate_llm.ESTIMATE_SCENARIO_COLUMNS),
            matched_project_packages=pd.DataFrame(columns=query_estimate_llm.MATCHED_PROJECT_PACKAGE_COLUMNS),
            candidate_families=pd.DataFrame(columns=query_estimate_llm.CANDIDATE_FAMILY_COLUMNS),
            candidate_display_groups=pd.DataFrame(
                [
                    {
                        "display_id": "D001",
                        "display_key": "屋面卷材防水|m²",
                        "display_name": "屋面卷材防水",
                        "unit": "m²",
                        "family_count": 1,
                        "family_ids": "F001",
                        "retrieval_package_support_ratio": 0.8,
                        "support_rank": 1,
                        "retrieval_item_count": 3,
                        "retrieval_package_count": 2,
                        "top_family_examples": "[]",
                        "direct_item_similarity_max": 0.9,
                    }
                ],
                columns=query_estimate_llm.CANDIDATE_DISPLAY_GROUP_COLUMNS,
            ),
            package_evidence_weights=pd.DataFrame(columns=query_estimate_llm.PACKAGE_EVIDENCE_WEIGHT_COLUMNS),
            display_group_families=pd.DataFrame(columns=query_estimate_llm.DISPLAY_GROUP_FAMILY_COLUMNS),
            display_option_grouping_trace=pd.DataFrame(columns=query_estimate_llm.DISPLAY_OPTION_GROUPING_TRACE_COLUMNS),
            matched_project_examples=pd.DataFrame(columns=query_estimate_llm.MATCHED_PROJECT_EXAMPLE_COLUMNS),
            evidence_items=pd.DataFrame(columns=query_estimate_llm.EVIDENCE_ITEM_COLUMNS),
            parse_info=pd.DataFrame([{"字段": "project_package_query_text", "值": "屋面工程"}]),
            llm_trace=pd.DataFrame(
                [
                    {"stage": "query_rewrite_for_embedding"},
                    {"stage": "display_option_grouping"},
                    {"stage": "scenario_generation"},
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
                    "estimate_scenarios",
                    "candidate_display_groups",
                    "candidate_families",
                    "display_option_grouping_trace",
                    "matched_project_packages",
                    "matched_project_examples",
                    "package_evidence_weights",
                    "evidence_items",
                    "parse_info",
                    "llm_trace",
                ],
            )
            trace_stages = [
                workbook["llm_trace"].cell(row=row, column=1).value
                for row in range(2, workbook["llm_trace"].max_row + 1)
            ]
            workbook.close()

        self.assertEqual(
            trace_stages,
            [
                "query_rewrite_for_embedding",
                "display_option_grouping",
                "scenario_generation",
            ],
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
