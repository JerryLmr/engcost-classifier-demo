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
    def write_sample_batch(self, path: Path, rows: list[dict[str, object]], sheet_name: str = "samples") -> None:
        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        worksheet.title = sheet_name
        headers = [
            "source_row_id",
            "item_row_id",
            "file_name",
            "consultation_project_name",
            "consultation_time",
            "location",
            "renovation_content",
            "sub_project_id",
            "seq",
            "cost_item_name",
            "project_description",
            "unit",
            "unit_normalized",
            "quantity",
        ]
        worksheet.append(headers)
        for row in rows:
            worksheet.append([row.get(header) for header in headers])
        path.parent.mkdir(parents=True, exist_ok=True)
        workbook.save(path)

    def sample_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source_row_id": 2,
                    "item_row_id": "2-1",
                    "工程名称": "屋面漏水维修工程",
                    "project_name_text": "屋面漏水维修",
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
                    "project_name_text": "外墙维修",
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
                    "project_name_text": "管道维修",
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
                    "project_name_text": "屋面漏水维修",
                    "project_detail_text": "屋面卷材防水 3mm SBS",
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
                    "project_name_text": "外墙维修",
                    "project_detail_text": "外墙防水",
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
                    "project_name_text": "管道维修",
                    "project_detail_text": "管道更换",
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
                "project_name_embeddings": "project_name_embeddings.npy",
                "project_detail_embeddings": "project_detail_embeddings.npy",
            },
        )
        self.assertNotIn("group_text", meta["field_descriptions"])
        self.assertNotIn("item_text", meta["field_descriptions"])
        self.assertNotIn("item_embeddings", meta["files"])

    def test_build_project_groups_aggregates_source_rows(self):
        samples = self.sample_frame()
        duplicate = samples.iloc[0].copy()
        duplicate["item_row_id"] = "2-2"
        duplicate["cost_item_name"] = "防水层拆除"
        duplicate["project_description"] = "拆除原屋面防水层"
        duplicate["consultation_project_name"] = "不应拼接的咨询项目名"
        duplicate["renovation_content"] = "不应拼接的维修内容"
        samples = pd.concat([samples, pd.DataFrame([duplicate])], ignore_index=True)

        project_groups = build_index.build_project_groups(samples)

        self.assertEqual(project_groups["source_row_id"].tolist(), [2, 3, 4])
        self.assertEqual(project_groups.columns.tolist(), build_index.PROJECT_GROUP_COLUMNS)
        self.assertNotIn("group_text", project_groups.columns)
        roof = project_groups[project_groups["source_row_id"] == 2].iloc[0]
        self.assertEqual(roof["工程名称"], "屋面漏水维修工程")
        self.assertEqual(roof["project_name_text"], "屋面漏水维修")
        self.assertEqual(roof["catalog_id"], "CP-002-03")
        self.assertEqual(roof["item_count"], 2)
        self.assertIn("屋面卷材防水 3.0mm SBS 沥青防水卷材", roof["project_detail_text"])
        self.assertIn("防水层拆除 拆除原屋面防水层", roof["project_detail_text"])
        self.assertNotIn("屋面漏水维修工程", roof["project_detail_text"])

    def test_build_project_groups_falls_back_when_project_name_text_empty(self):
        samples = self.sample_frame()
        samples.loc[samples["source_row_id"] == 2, "project_name_text"] = ""

        project_groups = build_index.build_project_groups(samples)

        roof = project_groups[project_groups["source_row_id"] == 2].iloc[0]
        self.assertEqual(roof["工程名称"], "屋面漏水维修工程")
        self.assertEqual(roof["project_name_text"], "屋面漏水维修工程")

    def test_build_samples_propagates_project_name_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "classified.xlsx"
            output_path = tmp_path / "samples.xlsx"
            workbook = openpyxl.Workbook()
            worksheet = workbook.active
            headers = [
                "file_name",
                "工程名称",
                "project_name_text",
                "consultation_project_name",
                "renovation_content",
                "catalog_id",
                "一级分类",
                "二级分类",
                "维修状态",
                "标准对象",
                "是否复合工程",
                "复合目录",
                "是否紧急维修",
                "是否白蚁相关",
                "是否建议复核",
                "分类依据",
                "consultation_time",
                "location",
                "sub_item_project_rows",
            ]
            worksheet.append(headers)
            worksheet.append(
                [
                    "source.pdf",
                    "嘉兴某小区12幢屋面渗漏维修工程",
                    "屋面渗漏维修",
                    "嘉兴某小区12幢屋面",
                    "渗漏维修工程",
                    "CP-002-03",
                    "屋面",
                    "防水层",
                    "维修",
                    "共用部位",
                    "否",
                    "",
                    "否",
                    "否",
                    "否",
                    "测试",
                    "2026-01-01",
                    "浙江省嘉兴市",
                    '[{"seq": 1, "cost_item_name": "屋面卷材防水", "project_description": "3mm SBS", "unit": "平方米", "quantity": 10, "unit_price": 80}]',
                ]
            )
            workbook.save(input_path)

            samples_count, errors_count = build_samples_script.build_and_write_samples(input_path, output_path)

            output_workbook = openpyxl.load_workbook(output_path, data_only=True)
            self.assertEqual(output_workbook.sheetnames, ["samples", "parse_errors"])
            samples_sheet = output_workbook["samples"]
            errors_sheet = output_workbook["parse_errors"]
            sample_headers = [
                samples_sheet.cell(row=1, column=column).value
                for column in range(1, samples_sheet.max_column + 1)
            ]
            error_headers = [
                errors_sheet.cell(row=1, column=column).value
                for column in range(1, errors_sheet.max_column + 1)
            ]
            sample = {
                header: samples_sheet.cell(row=2, column=index).value
                for index, header in enumerate(sample_headers, start=1)
            }
            output_workbook.close()

        self.assertEqual(samples_count, 1)
        self.assertEqual(errors_count, 0)
        self.assertEqual(sample_headers, build_samples_script.SAMPLE_HEADERS)
        self.assertEqual(error_headers, build_samples_script.PARSE_ERROR_HEADERS)
        self.assertEqual(sample["工程名称"], "嘉兴某小区12幢屋面渗漏维修工程")
        self.assertEqual(sample["project_name_text"], "屋面渗漏维修")
        self.assertEqual(sample["cost_item_name"], "屋面卷材防水")
        self.assertEqual(sample["project_description"], "3mm SBS")
        self.assertEqual(sample["unit"], "平方米")
        self.assertEqual(sample["unit_normalized"], "m²")
        self.assertEqual(sample["quantity"], 10)
        self.assertEqual(sample["unit_price"], 80)
        self.assertEqual(sample["item_similarity_text"], "屋面卷材防水；3mm SBS")
        self.assertEqual(sample["item_context_text"], "嘉兴某小区12幢屋面渗漏维修工程 / 屋面卷材防水 / 3mm SBS")
        self.assertIn('"cost_item_name": "屋面卷材防水"', sample["source_json"])

    def test_build_samples_streaming_helpers_use_row_values(self):
        header_map = build_samples_script.load_header_map_from_values(
            ("file_name", "project_name_text", "sub_item_project_rows")
        )

        self.assertEqual(header_map, {"file_name": 0, "project_name_text": 1, "sub_item_project_rows": 2})
        self.assertEqual(
            build_samples_script.get_cell_from_row(("a.pdf", "屋面", "[]"), header_map, "file_name"),
            "a.pdf",
        )
        self.assertIsNone(build_samples_script.get_cell_from_row(("a.pdf",), header_map, "sub_item_project_rows"))

    def test_build_samples_streams_parse_errors_to_output_sheet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "classified.xlsx"
            output_path = tmp_path / "samples.xlsx"
            workbook = openpyxl.Workbook()
            worksheet = workbook.active
            headers = [
                "file_name",
                "工程名称",
                "project_name_text",
                "consultation_project_name",
                "renovation_content",
                "catalog_id",
                "一级分类",
                "二级分类",
                "consultation_time",
                "location",
                "sub_item_project_rows",
            ]
            worksheet.append(headers)
            worksheet.append(
                [
                    "bad-json.pdf",
                    "坏 JSON 工程",
                    "坏 JSON",
                    "",
                    "",
                    "CP-000",
                    "一级",
                    "二级",
                    "2026-01-01",
                    "浙江省嘉兴市",
                    "[not-json]",
                ]
            )
            worksheet.append(
                [
                    "bad-number.pdf",
                    "数值异常工程",
                    "数值异常",
                    "",
                    "",
                    "CP-001",
                    "一级",
                    "二级",
                    "2026-01-02",
                    "浙江省嘉兴市",
                    '[{"seq": 1, "cost_item_name": "管道维修", "unit": "米", "quantity": 2, "unit_price": "abc"}]',
                ]
            )
            workbook.save(input_path)

            samples_count, errors_count = build_samples_script.build_and_write_samples(input_path, output_path)

            output_workbook = openpyxl.load_workbook(output_path, data_only=True)
            samples_sheet = output_workbook["samples"]
            errors_sheet = output_workbook["parse_errors"]
            error_headers = [
                errors_sheet.cell(row=1, column=column).value
                for column in range(1, errors_sheet.max_column + 1)
            ]
            error_rows = [
                {
                    header: errors_sheet.cell(row=row, column=index).value
                    for index, header in enumerate(error_headers, start=1)
                }
                for row in range(2, errors_sheet.max_row + 1)
            ]
            samples_max_row = samples_sheet.max_row
            output_workbook.close()

        self.assertEqual(samples_count, 1)
        self.assertEqual(errors_count, 3)
        self.assertEqual(samples_max_row, 2)
        self.assertEqual(error_headers, build_samples_script.PARSE_ERROR_HEADERS)
        self.assertEqual(
            [row["error_type"] for row in error_rows],
            ["invalid_sub_item_project_rows", "invalid_numeric_field", "missing_unit_price"],
        )
        self.assertIn("JSON 解析失败", error_rows[0]["error_message"])
        self.assertIn("field=unit_price", error_rows[1]["error_message"])

    def test_build_samples_streaming_missing_required_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "classified.xlsx"
            output_path = tmp_path / "samples.xlsx"
            workbook = openpyxl.Workbook()
            worksheet = workbook.active
            worksheet.append(["file_name", "sub_item_project_rows"])
            worksheet.append(["source.pdf", "[]"])
            workbook.save(input_path)

            with self.assertRaisesRegex(ValueError, "输入 Excel 缺少必要列: project_name_text"):
                build_samples_script.build_and_write_samples(input_path, output_path)

    def test_write_index_outputs_only_core_index_files(self):
        samples = self.sample_frame()
        project_groups = build_index.build_project_groups(samples)
        embeddings = np.ones((len(project_groups), 2), dtype=np.float32)
        meta = build_index.build_index_meta(Path("samples.xlsx"), "demo-model", len(samples), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "index"
            build_index.write_index(samples, project_groups, embeddings, embeddings, output_dir, meta)

            debug_path = Path(tmpdir) / ("cost_item_" + "project_groups.xlsx")
            self.assertFalse(debug_path.exists())
            self.assertTrue((output_dir / "samples.parquet").exists())
            self.assertTrue((output_dir / "project_groups.parquet").exists())
            self.assertTrue((output_dir / "project_name_embeddings.npy").exists())
            self.assertTrue((output_dir / "project_detail_embeddings.npy").exists())
            self.assertTrue((output_dir / "index_meta.json").exists())

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

    def test_build_index_parse_args_defaults_to_merged_samples(self):
        with patch.object(
            sys,
            "argv",
            ["build_cost_item_embedding_index.py", "--output-dir", "outputs/cost_item_index"],
        ):
            args = build_index.parse_args()

        self.assertEqual(args.samples, "samples/cost_item_samples_all.xlsx")
        self.assertEqual(args.output_dir, "outputs/cost_item_index")

    def test_build_index_validate_output_dir_requires_overwrite_for_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "index"
            output_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "输出已存在，请加 --overwrite"):
                build_index.validate_output_dir(output_dir, overwrite=False)
            build_index.validate_output_dir(output_dir, overwrite=True)

    def test_run_ingest_batch_parses_batch_id_from_filename(self):
        input_path = Path("excel_inputs/audit_ocr_export_20260630_001.xlsx")

        self.assertEqual(run_ingest_batch.infer_batch_id(input_path), "20260630_001")
        self.assertEqual(run_ingest_batch.batch_id_from_args(input_path, "manual_001"), "manual_001")

        with self.assertRaisesRegex(ValueError, "无法从文件名解析 batch_id"):
            run_ingest_batch.infer_batch_id(Path("excel_inputs/audit_ocr_export.xlsx"))

    def test_run_ingest_batch_rejects_existing_outputs_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cleaned.xlsx"
            output_path.write_text("existing", encoding="utf-8")
            outputs = {"cleaned": output_path}

            with self.assertRaisesRegex(ValueError, "换一个唯一的 --batch-id"):
                run_ingest_batch.validate_output_conflicts(outputs, overwrite=False)
            run_ingest_batch.validate_output_conflicts(outputs, overwrite=True)

    def test_run_ingest_batch_passes_overwrite_to_child_commands(self):
        outputs = {
            "cleaned": Path("cleaned_inputs/20260630_001/ocr_required_cleaned.xlsx"),
            "removed": Path("removed_inputs/20260630_001/ocr_required_removed.xlsx"),
            "classified": Path("classified_outputs/20260630_001/classified_projects.xlsx"),
            "samples": Path("samples/20260630_001/cost_item_samples.xlsx"),
        }

        commands_without_overwrite = run_ingest_batch.command_steps(Path("input.xlsx"), outputs, overwrite=False)
        commands_with_overwrite = run_ingest_batch.command_steps(Path("input.xlsx"), outputs, overwrite=True)

        self.assertFalse(any("--overwrite" in command for command in commands_without_overwrite))
        self.assertTrue(all(command[-1] == "--overwrite" for command in commands_with_overwrite))
        self.assertIn("--clean-output", commands_with_overwrite[0])
        self.assertIn("-o", commands_with_overwrite[1])
        self.assertIn("-o", commands_with_overwrite[2])

    def test_merge_samples_deduplicates_batches_and_writes_report(self):
        row = {
            "source_row_id": 2,
            "item_row_id": "2-1",
            "file_name": "source.pdf",
            "consultation_project_name": "嘉兴某小区屋面",
            "consultation_time": "2026-06-30",
            "location": "浙江省嘉兴市",
            "renovation_content": "渗漏维修",
            "sub_project_id": "屋面",
            "seq": 1,
            "cost_item_name": "屋面卷材防水",
            "project_description": "3mm SBS",
            "unit": "平方米",
            "unit_normalized": "m²",
            "quantity": 10,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            samples_dir = tmp_path / "samples"
            self.write_sample_batch(samples_dir / "20260630_001" / "cost_item_samples.xlsx", [row])
            self.write_sample_batch(samples_dir / "20260630_002" / "cost_item_samples.xlsx", [row])
            self.write_sample_batch(
                samples_dir / "20260701_001" / "cost_item_samples.xlsx",
                [{**row, "seq": 2, "quantity": 20}],
            )
            (samples_dir / "cost_item_samples_all.xlsx").write_text("not an input", encoding="utf-8")

            output_path = samples_dir / "cost_item_samples_all_output.xlsx"
            report_path = merge_samples.dedup_report_path(output_path)
            input_rows, output_rows, duplicate_rows = merge_samples.merge_batches(
                samples_dir,
                output_path,
                report_path,
            )

            workbook = openpyxl.load_workbook(output_path, data_only=True)
            worksheet = workbook["samples"]
            headers = [worksheet.cell(row=1, column=column).value for column in range(1, worksheet.max_column + 1)]
            rows = [
                {
                    header: worksheet.cell(row=row_index, column=column).value
                    for column, header in enumerate(headers, start=1)
                }
                for row_index in range(2, worksheet.max_row + 1)
            ]
            workbook.close()
            report_text = report_path.read_text(encoding="utf-8-sig")

        self.assertEqual((input_rows, output_rows, duplicate_rows), (3, 2, 1))
        self.assertEqual(headers[-2:], ["batch_id", "stable_sample_id"])
        self.assertEqual([row["batch_id"] for row in rows], ["20260630_001", "20260701_001"])
        self.assertIn("20260630_001", report_text)
        self.assertIn("20260630_002", report_text)

    def test_merge_samples_rejects_missing_samples_sheet_and_header_mismatch(self):
        row = {"file_name": "source.pdf", "cost_item_name": "屋面卷材防水"}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            samples_dir = tmp_path / "samples"
            self.write_sample_batch(samples_dir / "20260630_001" / "cost_item_samples.xlsx", [row])
            self.write_sample_batch(
                samples_dir / "20260630_002" / "cost_item_samples.xlsx",
                [row],
                sheet_name="not_samples",
            )

            with self.assertRaisesRegex(ValueError, "缺少 samples sheet"):
                merge_samples.merge_batches(
                    samples_dir,
                    samples_dir / "cost_item_samples_all.xlsx",
                    samples_dir / "cost_item_samples_all_dedup_report.csv",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            samples_dir = tmp_path / "samples"
            self.write_sample_batch(samples_dir / "20260630_001" / "cost_item_samples.xlsx", [row])
            workbook = openpyxl.Workbook()
            worksheet = workbook.active
            worksheet.title = "samples"
            worksheet.append(["different_header"])
            worksheet.append(["value"])
            batch_path = samples_dir / "20260630_002" / "cost_item_samples.xlsx"
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            workbook.save(batch_path)

            with self.assertRaisesRegex(ValueError, "样本表头不一致"):
                merge_samples.merge_batches(
                    samples_dir,
                    samples_dir / "cost_item_samples_all.xlsx",
                    samples_dir / "cost_item_samples_all_dedup_report.csv",
                )

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
            np.save(index_dir / "project_name_embeddings.npy", np.ones((3, 2), dtype=np.float32))
            np.save(index_dir / "project_detail_embeddings.npy", np.zeros((3, 2), dtype=np.float32))
            (index_dir / "index_meta.json").write_text('{"model": "demo-model"}', encoding="utf-8")

            samples, project_groups, name_embeddings, detail_embeddings, meta = query_estimate_llm.load_index(index_dir)

        self.assertEqual(len(samples), 3)
        self.assertEqual(len(project_groups), 3)
        self.assertEqual(name_embeddings.shape, (3, 2))
        self.assertEqual(detail_embeddings.shape, (3, 2))
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

    def test_query_prompt_uses_shared_semantic_extraction_rules(self):
        prompt = query_estimate_llm.build_parse_query_prompt("屋面漏水，500平，参考嘉兴一年内")

        self.assertIn("semantic_query_text", prompt)
        self.assertIn("用于相似项目检索的工程语义文本", prompt)
        self.assertIn("不要简单复制一级分类、二级分类、维修状态", prompt)
        self.assertNotIn("project_query_text", prompt)
        self.assertNotIn("detail_query_text", prompt)

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
        name_embeddings = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
        detail_embeddings = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]], dtype=np.float32)
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

        filtered, filtered_name_embeddings, filtered_detail_embeddings = query_estimate_llm.filter_project_groups(
            project_groups,
            name_embeddings,
            detail_embeddings,
            parsed,
        )

        self.assertEqual(filtered["source_row_id"].tolist(), [2, 4])
        self.assertEqual(filtered_name_embeddings.tolist(), [[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(filtered_detail_embeddings, np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32))

        empty = query_estimate_llm.ParsedQuery("x", "x", None, "", "上海市", None, None, [])
        filtered, filtered_name_embeddings, filtered_detail_embeddings = query_estimate_llm.filter_project_groups(
            project_groups,
            name_embeddings,
            detail_embeddings,
            empty,
        )
        self.assertTrue(filtered.empty)
        self.assertEqual(filtered_name_embeddings.shape[0], 0)
        self.assertEqual(filtered_detail_embeddings.shape[0], 0)

    def test_project_weights_validate_and_normalize(self):
        self.assertEqual(query_estimate_llm.normalize_project_weights(2.0, 1.0), (2.0 / 3.0, 1.0 / 3.0))
        self.assertEqual(query_estimate_llm.normalize_project_weights(0.85, 0.15), (0.85, 0.15))
        with self.assertRaisesRegex(ValueError, "必须大于等于 0"):
            query_estimate_llm.normalize_project_weights(-0.1, 1.0)
        with self.assertRaisesRegex(ValueError, "不能同时为 0"):
            query_estimate_llm.normalize_project_weights(0.0, 0.0)

    def test_score_project_groups_and_expand_samples_by_source_row_id(self):
        samples = self.sample_frame()
        duplicate = samples.iloc[0].copy()
        duplicate["item_row_id"] = "2-2"
        duplicate["seq"] = 2
        duplicate["cost_item_name"] = "基层处理"
        samples = pd.concat([samples, pd.DataFrame([duplicate])], ignore_index=True)
        project_groups = self.project_groups_frame()
        name_embeddings = np.array([[0.8, 0.2], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        detail_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)

        selected = query_estimate_llm.score_project_groups(
            project_groups,
            name_embeddings,
            detail_embeddings,
            np.array([1.0, 0.0], dtype=np.float32),
            1,
            0.5,
            0.5,
        )
        matches = query_estimate_llm.expand_matched_samples(samples, selected)

        self.assertEqual(selected["source_row_id"].tolist(), [2])
        self.assertEqual(selected["project_rank"].tolist(), [1])
        self.assertAlmostEqual(float(selected["project_name_score"].iloc[0]), 0.8)
        self.assertAlmostEqual(float(selected["project_detail_score"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(selected["project_score"].iloc[0]), 0.9)
        self.assertEqual(matches["item_row_id"].tolist(), ["2-1", "2-2"])
        self.assertIn("project_name_score", matches.columns)
        self.assertIn("project_detail_score", matches.columns)
        self.assertNotIn("group_text", matches.columns)

    def test_recommend_items_aggregate_and_estimate_totals(self):
        samples = self.sample_frame()
        samples.loc[0, "unit"] = "$ m^{2} $"
        duplicate = samples.iloc[0].copy()
        duplicate["source_row_id"] = 5
        duplicate["item_row_id"] = "5-1"
        duplicate["unit_price"] = 120
        duplicate["total_price"] = 12000
        samples = pd.concat([samples, pd.DataFrame([duplicate])], ignore_index=True)
        matched_projects = pd.DataFrame(
            [
                {"source_row_id": 2, "project_rank": 1, "project_score": 0.9, "project_name_score": 0.95, "project_detail_score": 0.6},
                {"source_row_id": 5, "project_rank": 2, "project_score": 0.8, "project_name_score": 0.85, "project_detail_score": 0.5},
                {"source_row_id": 4, "project_rank": 3, "project_score": 0.7, "project_name_score": 0.4, "project_detail_score": 0.9},
            ]
        )
        matches = query_estimate_llm.expand_matched_samples(samples, matched_projects)
        parsed = query_estimate_llm.ParsedQuery("屋面", "屋面", 500.0, "m²", "", None, None, [])

        recommended = query_estimate_llm.aggregate_recommend_items(matches, parsed)

        self.assertEqual(recommended.columns.tolist(), query_estimate_llm.RECOMMENDED_ITEM_COLUMNS)
        self.assertEqual(
            recommended.columns.tolist(),
            [
                "序号",
                "一级分类",
                "二级分类",
                "维修状态",
                "清单项名称",
                "项目特征/施工工艺",
                "单位",
                "历史工程量最小值",
                "历史工程量中位数",
                "历史工程量最大值",
                "本次估算金额最小值",
                "本次估算金额中位数",
                "本次估算金额最大值",
                "历史综合单价最小值",
                "历史综合单价中位数",
                "历史综合单价最大值",
                "历史总价最小值",
                "历史总价中位数",
                "历史总价最大值",
                "其中包含人工单价最小值",
                "其中包含人工单价中位数",
                "其中包含人工单价最大值",
                "其中包含机械单价最小值",
                "其中包含机械单价中位数",
                "其中包含机械单价最大值",
                "来源清单行",
                "历史样本数",
            ],
        )
        self.assertEqual(recommended["清单项名称"].tolist()[:2], ["屋面卷材防水", "管道更换"])
        self.assertEqual(recommended.columns.get_loc("项目特征/施工工艺"), recommended.columns.get_loc("清单项名称") + 1)
        self.assertEqual(recommended.columns.get_loc("单位"), recommended.columns.get_loc("项目特征/施工工艺") + 1)
        self.assertNotIn("project_description", recommended.columns)
        roof = recommended.iloc[0]
        self.assertEqual(roof["序号"], 1)
        self.assertEqual(roof["项目特征/施工工艺"], "3.0mm SBS 沥青防水卷材")
        self.assertEqual(roof["单位"], "m²")
        self.assertEqual(roof["历史样本数"], 2)
        self.assertEqual(roof["历史综合单价最小值"], 80.0)
        self.assertEqual(roof["历史综合单价中位数"], 100.0)
        self.assertEqual(roof["历史综合单价最大值"], 120.0)
        self.assertEqual(roof["本次估算金额中位数"], 50000.0)
        pipe = recommended[recommended["清单项名称"] == "管道更换"].iloc[0]
        self.assertEqual(pipe["本次估算金额中位数"], 3000.0)
        self.assertEqual(pipe["来源清单行"], "4-1")

    def test_recommend_items_display_formats_empty_unit_without_trailing_slash(self):
        recommend_items = pd.DataFrame(
            [
                {
                    "序号": 1,
                    "一级分类": "其他",
                    "二级分类": "",
                    "维修状态": "维修",
                    "清单项名称": "零星维修",
                    "项目特征/施工工艺": "",
                    "单位": "",
                    "历史工程量最小值": 12.0,
                    "历史工程量中位数": 12.0,
                    "历史工程量最大值": 12.0,
                    "本次估算金额最小值": 13590.0,
                    "本次估算金额中位数": 13590.0,
                    "本次估算金额最大值": 13590.0,
                    "历史综合单价最小值": 27.18,
                    "历史综合单价中位数": 27.18,
                    "历史综合单价最大值": 27.18,
                    "历史总价最小值": 6678.97,
                    "历史总价中位数": 6678.97,
                    "历史总价最大值": 6678.97,
                    "其中包含人工单价最小值": 3.58,
                    "其中包含人工单价中位数": 3.58,
                    "其中包含人工单价最大值": 3.58,
                    "其中包含机械单价最小值": 0.06,
                    "其中包含机械单价中位数": 0.06,
                    "其中包含机械单价最大值": 0.06,
                    "来源清单行": "1-1",
                    "历史样本数": 1,
                }
            ]
        )

        structured = query_estimate_llm.recommend_items_for_output(recommend_items, display=False)
        display = query_estimate_llm.recommend_items_for_output(recommend_items, display=True)

        self.assertEqual(structured.loc[0, "历史工程量最小值"], 12.0)
        self.assertEqual(structured.loc[0, "历史综合单价最小值"], 27.18)
        self.assertEqual(display.loc[0, "历史工程量最小值"], "12.00")
        self.assertEqual(display.loc[0, "历史综合单价最小值"], "27.18 元")
        self.assertEqual(display.loc[0, "其中包含人工单价最小值"], "3.58 元")
        self.assertEqual(display.loc[0, "其中包含机械单价最小值"], "0.06 元")
        self.assertEqual(display.loc[0, "历史总价最小值"], "6678.97 元")
        self.assertEqual(display.loc[0, "本次估算金额中位数"], "13590.00 元")

    def test_write_query_result_workbook_has_two_sheets_and_debug_toggle(self):
        samples = self.sample_frame()
        matched_projects = pd.DataFrame(
            [
                {
                    "source_row_id": 2,
                    "project_rank": 1,
                    "project_score": 0.9,
                    "project_name_score": 0.95,
                    "project_detail_score": 0.6,
                    "工程名称": "屋面漏水维修工程",
                    "project_name_text": "屋面漏水维修",
                    "project_detail_text": "屋面卷材防水 3mm SBS",
                },
                {
                    "source_row_id": 4,
                    "project_rank": 2,
                    "project_score": 0.8,
                    "project_name_score": 0.4,
                    "project_detail_score": 0.9,
                    "工程名称": "管道维修工程",
                    "project_name_text": "管道维修",
                    "project_detail_text": "管道更换",
                },
            ]
        )
        matches = query_estimate_llm.expand_matched_samples(samples, matched_projects)
        parsed = query_estimate_llm.ParsedQuery(
            "500平米屋面防水，浙江省嘉兴市，一年内",
            "屋面防水",
            500.0,
            "m²",
            "浙江省嘉兴市",
            date(2025, 1, 1),
            date(2026, 1, 1),
            [],
        )
        recommended = query_estimate_llm.aggregate_recommend_items(matches, parsed)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "query_result.xlsx"
            query_estimate_llm.write_query_result_workbook(output_path, parsed, recommended, matches, include_debug_text=False)
            workbook = openpyxl.load_workbook(output_path, data_only=True)
            self.assertEqual(workbook.sheetnames, ["recommend_items", "matches"])
            recommend_sheet = workbook["recommend_items"]
            self.assertIsNone(recommend_sheet.freeze_panes)
            self.assertEqual(recommend_sheet.cell(row=1, column=1).value, "查询内容")
            self.assertEqual(recommend_sheet.cell(row=1, column=2).value, "500平米屋面防水，浙江省嘉兴市，一年内")
            self.assertEqual(recommend_sheet.cell(row=2, column=1).value, "解析结果")
            self.assertEqual(recommend_sheet.cell(row=2, column=2).value, "识别工程量：500 m²；地点：浙江省嘉兴市；时间：一年内")
            recommend_headers = [
                recommend_sheet.cell(row=3, column=column).value
                for column in range(1, recommend_sheet.max_column + 1)
            ]
            match_headers = [workbook["matches"].cell(row=1, column=column).value for column in range(1, workbook["matches"].max_column + 1)]
            self.assertEqual(recommend_headers, query_estimate_llm.RECOMMENDED_ITEM_COLUMNS)
            self.assertIn("项目特征/施工工艺", recommend_headers)
            self.assertNotIn("project_description", recommend_headers)
            self.assertNotIn("unit", recommend_headers)
            self.assertNotIn("工程名称", match_headers)
            self.assertNotIn("project_description", match_headers)
            self.assertNotIn("group_text", match_headers)
            self.assertIn("project_name_score", match_headers)
            self.assertIn("project_detail_score", match_headers)
            recommend_column = {header: index for index, header in enumerate(recommend_headers, start=1)}
            self.assertEqual(recommend_sheet.cell(row=4, column=recommend_column["单位"]).value, "m²")
            self.assertEqual(recommend_sheet.cell(row=4, column=recommend_column["历史工程量最小值"]).value, 100)
            self.assertEqual(recommend_sheet.cell(row=4, column=recommend_column["历史综合单价最小值"]).value, 80)
            self.assertEqual(recommend_sheet.cell(row=4, column=recommend_column["本次估算金额中位数"]).value, 40000)
            self.assertIn(recommend_sheet.cell(row=5, column=recommend_column["历史综合单价中位数"]).value, ("", None))
            self.assertIn(recommend_sheet.cell(row=5, column=recommend_column["其中包含机械单价中位数"]).value, ("", None))
            workbook.close()

            display_path = Path(tmpdir) / "query_result_display.xlsx"
            query_estimate_llm.write_query_result_workbook(
                display_path,
                parsed,
                recommended,
                matches,
                include_debug_text=False,
                display=True,
            )
            workbook = openpyxl.load_workbook(display_path, data_only=True)
            display_sheet = workbook["recommend_items"]
            self.assertEqual(display_sheet.cell(row=4, column=recommend_column["历史工程量最小值"]).value, "100.00 m²")
            self.assertEqual(display_sheet.cell(row=4, column=recommend_column["历史综合单价最小值"]).value, "80.00 元/m²")
            self.assertEqual(display_sheet.cell(row=4, column=recommend_column["本次估算金额中位数"]).value, "40000.00 元")
            workbook.close()

            debug_path = Path(tmpdir) / "query_result_debug.xlsx"
            query_estimate_llm.write_query_result_workbook(debug_path, parsed, recommended, matches, include_debug_text=True)
            workbook = openpyxl.load_workbook(debug_path, data_only=True)
            debug_headers = [workbook["matches"].cell(row=1, column=column).value for column in range(1, workbook["matches"].max_column + 1)]
            workbook.close()

        self.assertIn("工程名称", debug_headers)
        self.assertIn("project_name_text", debug_headers)
        self.assertIn("project_detail_text", debug_headers)
        self.assertNotIn("group_text", debug_headers)
        self.assertNotIn("project_description", debug_headers)

    def test_legacy_query_entrypoint_removed_and_readme_points_to_official_script(self):
        legacy_name = "query_cost_" "item_estimate.py"
        self.assertFalse((ROOT / "scripts" / legacy_name).exists())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("scripts/query_cost_estimate_llm.py", readme)
        self.assertNotIn(f"scripts/{legacy_name}", readme)


if __name__ == "__main__":
    unittest.main()
