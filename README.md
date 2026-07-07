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

新查询流程：

```text
用户自然语言需求
→ LLM 生成 ParsedQuery、project_package_query_text 和 item_query_text
→ 标准目录分类器选择一个主目录
→ package query 检索 project_package_embeddings
→ item query 检索 item_embeddings
→ candidate_pool 展开历史清单行
→ 按规范化 fine_signature 精确聚合 candidate_families
→ 第一次 LLM 只选择 family_id 和建议项类型
→ 程序构建已选 family 的历史数量关系
→ 第二次 LLM 判断工程量低/中/高区间和是否计入金额
→ 程序按 family_id 回填名称、项目特征、单位、历史单价和来源样本
→ 程序计算金额并生成 estimate_summary
→ 输出 estimate_summary、suggested_bill、matched_project_packages、candidate_families、evidence_items、parse_info、llm_trace
```

输出 xlsx 固定包含：

- `estimate_summary`：面向用户/领导的估价摘要，展示原始需求、ParsedQuery、核心分类、估算金额区间、核心项目、共现措施项、未计入或需现场确认项目、主要不确定因素和来源说明。
- `suggested_bill`：估价主表，展示最终采用的推荐清单项、项目角色、工程量来源、建议工程量低/中/高、历史综合单价最低值/中位数/最高值、人工费和机械费单价区间、估算金额区间、金额计算说明、采用理由、不确定性说明和来源样本。历史价格由程序从同一 `fine_signature` 的 `candidate_families` 回填。
- `matched_project_packages`：相似历史工程包摘要，包括 `package_score`、`project_package_id`、工程名称、`project_name_text`、`cost_item_names_summary`、`consultation_time`、`location`、`cache_subject` 和 `item_count`。
- `candidate_families`：按规范化 `fine_signature = cost_item_name + project_description + unit` 聚合的候选施工做法统计。每个 family 是唯一可报价边界，3mm/4mm、自粘/热熔、单层/双层等不同规格不会合并。
- `evidence_items`：来源样本明细，保存本次查询进入候选池的历史清单行。`source_ref = project_key + "::" + item_row_id`，可用于回查 `samples/cost_item_samples_all.xlsx` / `samples.parquet`。
- `parse_info`：本次查询解析结果和检索参数，包括原始需求、ParsedQuery、分类结果、候选池行数、family 数、两次 LLM 输入/输出规模、token、fallback、错误和 warnings。
- `llm_trace`：记录 query rewrite、目录分类、family selection、quantity decision 是否成功、prompt 长度、真实 token（服务返回时）或估算 token、输入摘要和错误。

`fine_signature` 会对已确认的等价表达做受控归一化，例如：

- `厚3.0mm` / `3.0mm厚` / `3.0mm` 统一为 `3.0mm`。
- `SBS防水卷材` / `SBS改性沥青防水卷材` / `弹性改性沥青防水卷材` / `弹性体改性沥青防水卷材` 统一为正式名称 `弹性体改性沥青防水卷材`。

不同厚度、施工方式、层数、耐根穿刺、附加层、基层处理、平面/立面、砂面等仍保持独立 family，不会因为材料同义词归一化而合并。

两次 LLM 职责边界：

- 第一次 LLM：只从输入的 `candidate_families` 中选择真实存在的 `family_id`，并判断 `核心施工项`、`常见前置项`、`恢复/收尾项`、`措施/条件项`、`可选/替代工艺`。
- 第二次 LLM：只为第一次已选 family 判断工程量低/中/高区间、工程量来源和是否计入参考金额。
- LLM 不生成单价、来源、清单名称、项目特征、单位或金额。综合单价必须来自同一 `fine_signature` 下的历史样本统计。

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
