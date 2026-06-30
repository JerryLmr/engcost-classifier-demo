## 已审定清单样本库与初案估价流程

本流程用于将 OCR 识别出的已审定维修项目资料转为清单级样本库，并基于相似历史工程召回给新维修初案提供参考价格区间。

当前流程不用于判断“异常 / 违规 / 不合理”，只输出相似样本和参考区间。

### 1. 批次导入 OCR Excel

后续 OCR 文件统一放在：

```text
excel_inputs/
```

文件名格式：

```text
audit_ocr_export_YYYYMMDD_NNN.xlsx
```

示例：

```text
excel_inputs/audit_ocr_export_20260630_001.xlsx
excel_inputs/audit_ocr_export_20260630_002.xlsx
```

日常导入单批次：

```bash
backend/.venv/bin/python scripts/run_ingest_batch.py \
  --input excel_inputs/audit_ocr_export_20260630_001.xlsx
```

不传 `--batch-id` 时，脚本会从文件名解析 `batch_id`：

```text
audit_ocr_export_20260630_001.xlsx -> 20260630_001
```

如需手动指定批次：

```bash
backend/.venv/bin/python scripts/run_ingest_batch.py \
  --input excel_inputs/audit_ocr_export_20260630_001.xlsx \
  --batch-id 20260630_001
```

单批次产物固定为：

```text
cleaned_inputs/{batch_id}/ocr_required_cleaned.xlsx
removed_inputs/{batch_id}/ocr_required_removed.xlsx
classified_outputs/{batch_id}/classified_projects.xlsx
samples/{batch_id}/cost_item_samples.xlsx
```

如果任一批次产物已存在，默认报错。确认要重跑并覆盖该批次时显式传：

```bash
backend/.venv/bin/python scripts/run_ingest_batch.py \
  --input excel_inputs/audit_ocr_export_20260630_001.xlsx \
  --overwrite
```

`run_ingest_batch.py` 内部依次执行：

```text
filter_required_ocr_rows.py
batch_classify_excel.py
build_cost_item_samples.py
```

它不会构建 embedding index。

### 2. Active Sheet 约定

当前 OCR 清洗、批量分类和样本展开都只读取输入 xlsx 的 active sheet，不遍历多个 sheet。

提供 OCR 文件时建议只保留一张数据 sheet，或者保存时确保数据 sheet 是当前 active sheet。

输入 Excel 需要包含：

```text
file_name
consultation_project_name
consultation_time
renovation_content
sub_item_project_rows
location
```

### 3. 合并所有批次样本

导入一个或多个批次后，合并历史样本：

```bash
backend/.venv/bin/python scripts/merge_cost_item_sample_batches.py
```

默认等价于：

```bash
backend/.venv/bin/python scripts/merge_cost_item_sample_batches.py \
  --input-dir samples \
  --output samples/cost_item_samples_all.xlsx
```

脚本只读取：

```text
samples/*/cost_item_samples.xlsx
```

不会读取 `samples/cost_item_samples_all.xlsx`，因为它是合并产物。

输出：

```text
samples/cost_item_samples_all.xlsx
samples/cost_item_samples_all_dedup_report.csv
```

如果总样本文件或去重报告已存在，默认报错。确认要覆盖时显式传：

```bash
backend/.venv/bin/python scripts/merge_cost_item_sample_batches.py \
  --overwrite
```

合并时会给每条样本追加：

```text
batch_id
stable_sample_id
```

`batch_id` 负责来源追踪；`stable_sample_id` 负责样本去重。`stable_sample_id` 不使用 `project_code`，也不使用 `batch_id`，因此同一份 OCR 重复导入到不同批次时仍可去重。

### 4. 构建 embedding 索引

合并总样本后重建本地 embedding index：

```bash
backend/.venv/bin/python scripts/build_cost_item_embedding_index.py \
  --samples samples/cost_item_samples_all.xlsx \
  --output-dir outputs/cost_item_index \
  --model BAAI/bge-m3 \
  --overwrite
```

`--samples` 默认就是 `samples/cost_item_samples_all.xlsx`，也可以省略：

```bash
backend/.venv/bin/python scripts/build_cost_item_embedding_index.py \
  --output-dir outputs/cost_item_index \
  --model BAAI/bge-m3 \
  --overwrite
```

输出目录：

```text
outputs/cost_item_index/
```

包含：

```text
samples.parquet
project_groups.parquet
project_name_embeddings.npy
project_detail_embeddings.npy
index_meta.json
```

同时生成便于人工检查的工程组调试表：

```text
outputs/cost_item_project_groups.xlsx
```

索引阶段不再调用 LLM 清洗工程名称，只读取样本中的 `project_name_text`。如果该字段为空，会打印 warning 并回退为原始 `工程名称`。`project_detail_text` 由同一历史工程下的 `cost_item_name + project_description` 去重拼接，不拼入工程名称。

当前阶段不使用 Milvus，不使用 LangChain；每次合并后允许重建整个本地 parquet + npy + `index_meta.json`。

### 5. 自然语言造价查询

示例：用户只输入口语化维修需求，系统调用本地 LLM 将 `--text` 解析为结构化 `ParsedQuery`，再由程序执行 project group 双路 embedding 检索、location / consultation_time 硬过滤、历史工程下样本展开和推荐清单项聚合。

```bash
backend/.venv/bin/python scripts/query_cost_estimate_llm.py \
  --index-dir outputs/cost_item_index \
  --text "屋面漏水，想做3mm SBS防水，面积大概500平，参考嘉兴一年内的造价" \
  --output outputs/cost_estimate_result.xlsx \
  --overwrite
```

输出文件：

```text
outputs/cost_estimate_result.xlsx
```

其中：

- `recommend_items` sheet：领导展示主表，按相似工程展开后的历史清单项聚合，默认使用结构化数值列展示一级/二级分类、维修状态、清单项名称、项目特征/施工工艺、单位、样本数、历史工程量、估算金额、综合单价、历史总价、人工单价、机械单价和来源清单行；需要把单位和金额单位拼入数值单元格时加 `--display`。
- `matches` sheet：内部追溯表，展示相似工程展开后的清单样本行，并保留 `project_score`、`project_name_score`、`project_detail_score`。默认不输出历史工程名称和召回文本；需要调试明文时加 `--include-debug-text`。

### 6. 日常流程

第一批：

```bash
backend/.venv/bin/python scripts/run_ingest_batch.py \
  --input excel_inputs/audit_ocr_export_20260630_001.xlsx
```

当天第二批：

```bash
backend/.venv/bin/python scripts/run_ingest_batch.py \
  --input excel_inputs/audit_ocr_export_20260630_002.xlsx
```

合并所有历史样本：

```bash
backend/.venv/bin/python scripts/merge_cost_item_sample_batches.py
```

重建 embedding：

```bash
backend/.venv/bin/python scripts/build_cost_item_embedding_index.py \
  --output-dir outputs/cost_item_index \
  --model BAAI/bge-m3 \
  --overwrite
```

新增 OCR 文件时，不需要人工合并 Excel。每批中间结果都会独立保留，便于检查、回滚和重跑。

所有会覆盖已有产物的操作都必须显式传 `--overwrite`。

### 7. 文件提交说明

以下目录和文件为运行产物，不提交 Git：

```text
excel_outputs/
cleaned_inputs/
removed_inputs/
classified_outputs/
samples/
outputs/
*.xlsx
*.csv
*.npy
*.parquet
```

如需提交示例数据，应使用脱敏的小样本文件。
