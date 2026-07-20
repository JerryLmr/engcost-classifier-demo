from __future__ import annotations

import importlib
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from estimator import paths
from estimator.indexing.index_loader import load_index
from estimator.ingestion.pipeline import batch_outputs, command_steps
from estimator.query_models import QueryRewrite
from estimator.retrieval.query_rewrite import query_rewrite_for_embedding


MODULES = [
    "estimator.paths",
    "estimator.query_models",
    "estimator.ingestion.ocr_filter",
    "estimator.ingestion.project_classification",
    "estimator.ingestion.sample_builder",
    "estimator.ingestion.batch_merge",
    "estimator.ingestion.pipeline",
    "estimator.indexing.embedding_model",
    "estimator.indexing.index_builder",
    "estimator.indexing.index_loader",
    "estimator.retrieval.query_rewrite",
    "estimator.retrieval.constraints",
    "estimator.retrieval.package_retrieval",
    "estimator.retrieval.item_retrieval",
    "estimator.retrieval.evidence_pool",
    "estimator.retrieval.weights",
    "estimator.candidates.signatures",
    "estimator.candidates.families",
    "estimator.candidates.display_groups",
    "estimator.candidates.practice_options",
    "estimator.planning.package_selection",
    "estimator.planning.range_selection",
    "estimator.planning.option_selection",
    "estimator.planning.quantity_determination",
    "estimator.planning.scenarios",
    "estimator.pricing.evidence_expansion",
    "estimator.pricing.quantity_statistics",
    "estimator.pricing.price_statistics",
    "estimator.pricing.estimate_calculator",
    "estimator.pricing.summary",
    "estimator.reporting.columns",
    "estimator.reporting.frames",
    "estimator.reporting.formatting",
    "estimator.reporting.excel_writer",
    "estimator.reporting.explanation",
    "estimator.query_pipeline",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_estimator_module_import_smoke(module_name: str) -> None:
    assert importlib.import_module(module_name)


def test_paths_resolve_from_repo_root_and_keep_absolute_paths() -> None:
    assert paths.resolve_repo_path("samples/example.xlsx") == (
        paths.REPO_ROOT / "samples" / "example.xlsx"
    ).resolve()
    absolute = Path("/tmp/estimator-example.xlsx")
    assert paths.resolve_repo_path(absolute) == absolute

    frozen_now = Mock()
    frozen_now.strftime.return_value = "202607201305"
    with patch("estimator.paths.datetime") as datetime_mock:
        datetime_mock.now.return_value = frozen_now
        assert paths.default_query_output_path() == paths.REPO_ROOT / "query/202607201305.xlsx"


def test_query_rewrite_uses_fixed_payload_and_request_contract() -> None:
    request = Mock(
        return_value={
            "project_package_query_text": "屋面维修工程",
            "item_query_text": "3mm SBS 防水卷材",
            "location": "浙江省嘉兴市",
            "start_date": "2025-07-20",
            "end_date": "2026-07-20",
        }
    )
    rewrite, trace = query_rewrite_for_embedding(
        "嘉兴一年内屋面漏水",
        current_date=date(2026, 7, 20),
        request_json=request,
    )

    assert rewrite == QueryRewrite(
        raw_query="嘉兴一年内屋面漏水",
        project_package_query_text="屋面维修工程",
        item_query_text="3mm SBS 防水卷材",
        location="浙江省嘉兴市",
        start_date="2025-07-20",
        end_date="2026-07-20",
        notes=[],
        success=True,
    )
    assert trace["stage"] == "query_rewrite_for_embedding"
    assert trace["max_tokens"] == 512
    assert request.call_args.kwargs == {
        "max_tokens": 512,
        "system_prompt": "你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
    }


def test_query_rewrite_fixed_failure_uses_original_query() -> None:
    rewrite, trace = query_rewrite_for_embedding(
        "屋面漏水",
        current_date=date(2026, 7, 20),
        request_json=Mock(side_effect=RuntimeError("fixed failure")),
    )
    assert rewrite.project_package_query_text == "屋面漏水"
    assert rewrite.item_query_text == "屋面漏水"
    assert rewrite.success is False
    assert trace["parsed_status"] == "failed"
    assert trace["error_message"] == "fixed failure"


def test_ingestion_paths_and_commands_stay_under_repo_root() -> None:
    outputs = batch_outputs("20260720_001")
    assert outputs == {
        "cleaned": paths.REPO_ROOT / "ingestion_data/cleaned_inputs/20260720_001/ocr_required_cleaned.xlsx",
        "classified": paths.REPO_ROOT / "ingestion_data/classified_outputs/20260720_001/classified_projects.xlsx",
        "samples": paths.REPO_ROOT / "samples/20260720_001/cost_item_samples.xlsx",
    }
    commands = command_steps(paths.REPO_ROOT / "ingestion_data/excel_inputs/input.xlsx", outputs, True)
    assert [Path(command[1]).name for command in commands] == [
        "filter_required_ocr_rows.py",
        "batch_classify_excel.py",
        "build_cost_item_samples.py",
    ]
    assert all(command[-1] == "--overwrite" for command in commands)


def test_index_loader_validates_schema_and_embedding_shapes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir)
        for name in [
            "samples.parquet",
            "project_packages.parquet",
            "project_package_embeddings.npy",
            "item_embeddings.npy",
            "index_meta.json",
        ]:
            (index_dir / name).touch()
        (index_dir / "index_meta.json").write_text('{"model": "fixed"}', encoding="utf-8")

        samples = pd.DataFrame({"sample_index": [0, 1], "project_package_id": ["a", "b"]})
        packages = pd.DataFrame({"project_package_id": ["a", "b"]})
        with patch("estimator.indexing.index_loader.pd.read_parquet", side_effect=[samples, packages]), patch(
            "estimator.indexing.index_loader.np.load",
            side_effect=[np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32)],
        ):
            loaded = load_index(index_dir)
        assert loaded[0].equals(samples)
        assert loaded[1].equals(packages)
        assert loaded[2].shape == (2, 3)
        assert loaded[3].shape == (2, 3)


def test_index_loader_rejects_missing_files_and_shape_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="索引目录缺少文件"):
            load_index(Path(tmpdir))

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir)
        for name in [
            "samples.parquet",
            "project_packages.parquet",
            "project_package_embeddings.npy",
            "item_embeddings.npy",
            "index_meta.json",
        ]:
            (index_dir / name).touch()
        (index_dir / "index_meta.json").write_text("{}", encoding="utf-8")
        samples = pd.DataFrame({"sample_index": [0], "project_package_id": ["a"]})
        packages = pd.DataFrame({"project_package_id": ["a"]})
        with patch("estimator.indexing.index_loader.pd.read_parquet", side_effect=[samples, packages]), patch(
            "estimator.indexing.index_loader.np.load",
            side_effect=[np.ones((1, 3), dtype=np.float32), np.ones((1, 4), dtype=np.float32)],
        ), pytest.raises(ValueError, match="维度不一致"):
            load_index(index_dir)


def test_thin_entrypoints_keep_legacy_symbols() -> None:
    classification_entry = importlib.import_module("estimator.scripts.batch_classify_excel")
    ocr_entry = importlib.import_module("estimator.scripts.filter_required_ocr_rows")
    index_entry = importlib.import_module("estimator.scripts.build_cost_item_embedding_index")
    assert callable(classification_entry.classify_workbook)
    assert callable(classification_entry._classification_cache_key)
    assert callable(ocr_entry.filter_required_ocr_rows)
    assert callable(index_entry.build_cost_item_embedding_index)
