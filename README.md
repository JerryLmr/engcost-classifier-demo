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
classified_outputs/{batch_id}/classified_projects.xlsx
samples/{batch_id}/cost_item_samples.xlsx
```

`ocr_required_cleaned.xlsx` 包含两个 sheet：`cleaned` 保存必填字段完整行，`removed` 保存被过滤行和缺失字段原因。

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
project_key
stable_sample_id
```

`batch_id` 负责来源追踪；`stable_sample_id` 负责样本去重。`stable_sample_id` 不使用 `project_code`，也不使用 `batch_id`，因此同一份 OCR 重复导入到不同批次时仍可去重。
`source_row_id` 只在单个 batch 内有意义；跨 batch 后 query 统一使用 `project_key` 追溯和展开。

### 4. 构建 embedding 索引

合并总样本后重建本地 embedding index：

```bash
backend/.venv/bin/python scripts/build_cost_item_embedding_index.py --overwrite
```

默认读取：

```text
samples/cost_item_samples_all.xlsx
```

默认输出到：

```text
embeddings/
```

默认模型：

```text
BAAI/bge-m3
```

如果目标索引目录已存在，脚本默认不覆盖；确认要重建时必须显式加：

```bash
backend/.venv/bin/python scripts/build_cost_item_embedding_index.py --overwrite
```

也可以按需传参覆盖默认值，例如：

```bash
backend/.venv/bin/python scripts/build_cost_item_embedding_index.py \
  --samples samples/cost_item_samples_all.xlsx \
  --output-dir embeddings \
  --model BAAI/bge-m3 \
  --overwrite
```

`embeddings`中包含：

```text
samples.parquet
project_packages.parquet
project_package_embeddings.npy
item_embeddings.npy
index_meta.json
```

索引包含两层 embedding：

- `project_package_embeddings.npy`：每个 `project_key` 一个历史工程包 embedding，文本只基于 `工程名称 + project_name_text + cost_item_names_summary`，用于召回相似历史工程包。
- `item_embeddings.npy`：每条历史清单行一个 embedding，文本只基于 `cost_item_name + project_description + unit_normalized`，用于召回相似清单行证据。

目录分类字段仍保留在 `samples.parquet` / `project_packages.parquet` 中用于追溯和 `catalog_score`，但不进入 package/item embedding 文本。

当前阶段不使用 Milvus，不使用 LangChain；每次合并后允许重建整个本地 parquet + npy + `index_meta.json`。

### 5. 自然语言造价查询

本系统不是通用联网报价工具，而是基于内部历史审价样本和相似工程包的离线估价辅助工具。LLM 只负责识别需求、选择真实存在的候选施工做法、判断工程量区间和是否计入金额；清单名称、项目特征、单位、历史单价、来源样本和估算金额全部由程序按 `fine_signature` 回填或计算。

运行 `query_cost_estimate_llm.py` 前需先启动 LM Studio Server 或兼容 OpenAI `chat/completions` 的本地 LLM 服务；脚本启动时会先检查 `LMSTUDIO_BASE_URL/models`，服务不可用会快速退出。

```bash
backend/.venv/bin/python scripts/query_cost_estimate_llm.py \
  --text "屋面漏水，想做3mm SBS防水，面积大概500平"
```

默认读取索引目录：

```text
embeddings/
```

默认输出文件：

```text
query/YYYYMMDDHHMM.xlsx
```

如果要指定其它索引目录或输出文件：

```bash
backend/.venv/bin/python scripts/query_cost_estimate_llm.py \
  --index-dir embeddings \
  --text "屋面漏水，想做3mm SBS防水，面积大概500平" \
  --output query/test_result.xlsx \
  --overwrite
```

如果输出文件已存在，需要显式传 `--overwrite`。

display_selection 默认最多发送 50 个 display，其中主排序候选由 `final_score最大值`、`item_score最大值`、`cooccur_score最大值`、`来源工程包数去重` 四个完整排行榜交替补足，重复 display 不占配额；候选超过 50 时末尾保留 5 个确定性等距探索候选。当前 CLI 参数名沿用：

```text
--family-selection-limit
--family-exploration-limit
```

新查询流程：

```text
用户查询
→ LLM query rewrite
→ 工程包与清单行 embedding 召回
→ candidate pool
→ fine_signature family 聚合
→ 按 cost_item_name + unit 形成 display groups
→ LLM 选择最小完整施工链中的 display
→ 对已选 display 选择默认 family 和其他历史做法
→ display 级重复/包含关系处理
→ 工程量与金额处理
→ suggested_bill
```

输出 xlsx 固定包含：

- `estimate_summary`：面向用户/领导的估价摘要，展示原始需求、ParsedQuery、核心分类、估算金额区间、核心项目、共现措施项、未计入或需现场确认项目、主要不确定因素和来源说明。
- `suggested_bill`：估价主表，一行一个最终保留的 display。清单名称来自 display，默认参考做法和价格来源来自 `selected_family_id`；同组其他 family 只作为“其他历史做法”展示，不单独生成金额行。
- `matched_project_packages`：相似历史工程包摘要，包括 `package_score`、`project_package_id`、工程名称、`project_name_text`、`cost_item_names_summary`、`consultation_time`、`location`、`cache_subject` 和 `item_count`。
- `candidate_families`：按规范化 `fine_signature = cost_item_name + project_description + unit` 聚合的候选施工做法统计。当前没有新增第二套 family 规范化规则；每个 family 是唯一可报价边界，3mm/4mm、自粘/热熔、单层/双层等不同规格不会合并；表内保留 `package_path_score最大值`、`item_path_score最大值` 和 `final_score` 便于排查召回排序。
- `candidate_display_groups`：按 normalized `cost_item_name + unit` 组织出来的客户展示候选。display 只用于减少同名清单项重复展示，不合并同组 family 的价格样本。
- `display_group_families`：display 到内部 family 的映射表，用于从 display 回查 `fine_signature`、默认 family 和 evidence。
- `display_selection_trace`：记录实际发送给 display_selection LLM 的 display、`selection_rank`、`candidate_source`、是否被选中、display 层 `item_type` 和选择原因。
- `display_family_selection_trace`：记录每个已选 display 内部的 family、默认 family、其他历史做法及差异说明。
- `evidence_items`：来源样本明细，保存本次查询进入候选池的历史清单行。`source_ref = project_key + "::" + item_row_id`，`family_id` 和 `fine_signature` 可用于从历史样本回查所属 family。
- `parse_info`：本次查询解析结果和检索参数，包括原始需求、ParsedQuery、分类结果、候选池行数、family 数、display 数、LLM 输入/输出规模、token、fallback、错误、dedup 抑制摘要和 warnings。
- `llm_trace`：记录 query rewrite、目录分类、display selection、display family selection、dedup selection、quantity decision 是否成功、prompt 长度、真实 token（服务返回时）或估算 token、输入摘要和错误。

`fine_signature` 会对已确认的等价表达做受控归一化，例如：

- `1.3.0厚` / `厚3.0mm` / `3.0mm厚` / `3.0mm` 统一为 `3.0mm`。
- `SBS防水卷材` / `SBS沥青防水卷材` / `SBS改性沥青防水卷材` / `弹性改性沥青防水卷材` / `弹性体改性沥青防水卷材` 统一为正式名称 `弹性体改性沥青防水卷材`。
- `m2` / `m²` / `m^{2}` / `平方米` 统一为 `m²`。
- `原有` / `原`、`铲除` / `拆除`、`垃圾外运` / `垃圾清运` 只作为已确认等价文本做受控统一。

不同厚度、施工方式、层数、耐根穿刺、附加层、基层处理、平面/立面、砂面等仍保持独立 family，不会因为材料同义词归一化而合并。

LLM 职责边界：

- display_selection：只从输入的 `candidate_displays` 中选择真实存在的 `display_id`，并在 display 层判断 `核心施工项`、`常见前置项`、`恢复/收尾项`、`施工措施/现场条件项`、`可选/替代工艺`。
- display_family_selection：只在已选 display 内选择一个 `selected_family_id` 作为默认做法和价格来源，并选择 0～3 个同组其他历史做法。
- dedup_selection：只对已选 display 做重复或包含关系抑制，不创建新 display，不合并价格样本、来源、工程量或单价区间。
- quantity_decision：只为 dedup 后保留的 display 判断工程量低/中/高区间、工程量来源和是否计入参考金额。
- LLM 不生成单价、来源、清单名称、单位或金额。综合单价必须来自 `selected_family_id` 对应的同一 `fine_signature` 历史样本统计。

来源样本统一使用：

```text
source_ref = project_key + "::" + item_row_id
```

示例：

```text
20260630_001::27::27-1
```

回查时在 `samples/cost_item_samples_all.xlsx` 的 `samples` sheet 中筛选：

```text
project_key == "20260630_001::27"
item_row_id == "27-1"
```

查询输出不再使用临时证据编号。

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
backend/.venv/bin/python scripts/build_cost_item_embedding_index.py --overwrite
```

新增 OCR 文件时，不需要人工合并 Excel。每批中间结果都会独立保留，便于检查、回滚和重跑。

所有会覆盖已有产物的操作都必须显式传 `--overwrite`。

### 7. 文件提交说明

以下目录和文件为运行产物，不提交 Git：

```text
excel_outputs/
cleaned_inputs/
classified_outputs/
samples/
embeddings/
query/
outputs/
*.xlsx
*.csv
*.npy
*.parquet
```

如需提交示例数据，应使用脱敏的小样本文件。
