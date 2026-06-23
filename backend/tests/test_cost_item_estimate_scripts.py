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


if np is not None and pd is not None:
    build_index = load_script_module("build_cost_item_embedding_index", "scripts/build_cost_item_embedding_index.py")
    query_estimate = load_script_module("query_cost_item_estimate", "scripts/query_cost_item_estimate.py")
else:
    build_index = None
    query_estimate = None


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
        self.assertEqual(matches["item_row_id"].tolist(), ["2-1", "3-1"])

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


if __name__ == "__main__":
    unittest.main()
