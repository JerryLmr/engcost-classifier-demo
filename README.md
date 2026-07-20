## 已审定清单样本库与初案估价流程

本流程用于将 OCR 识别出的已审定维修项目资料转为清单级样本库，并基于相似历史工程召回给新维修初案提供参考价格区间。

当前流程不用于判断“异常 / 违规 / 不合理”，只输出相似样本和参考区间。

代码目录分为：

```text
classifier/backend/    分类后端与测试
classifier/frontend/   分类前端
estimator/ingestion/   OCR 清洗、分类接入与样本构建
estimator/indexing/    embedding 模型、索引构建与加载
estimator/retrieval/   query rewrite、约束、召回与证据权重
estimator/candidates/  Family、Display 与 Practice Option 候选
estimator/planning/    工程包、区间、Option 与工程量决策
estimator/pricing/     证据扩展、统计与金额计算
estimator/reporting/   QueryResult 到 workbook frames、Excel 写入与最终说明
estimator/query_pipeline.py  造价查询编排，返回既有 QueryResult
estimator/scripts/     保留命令入口及 ingestion/indexing 脚本
```

以下命令均从仓库根目录执行。估价脚本中的相对数据路径统一相对仓库根目录解析，导入阶段数据写入根目录的 `ingestion_data/excel_inputs/`、`ingestion_data/cleaned_inputs/`、`ingestion_data/classified_outputs/`，其余运行数据继续写入 `samples/`、`embeddings/`、`query/`、`outputs/` 等目录。

### 1. 批次导入 OCR Excel

后续 OCR 文件统一放在：

```text
ingestion_data/excel_inputs/
```

文件名格式：

```text
audit_ocr_export_YYYYMMDD_NNN.xlsx
```

示例：

```text
ingestion_data/excel_inputs/audit_ocr_export_20260630_001.xlsx
ingestion_data/excel_inputs/audit_ocr_export_20260630_002.xlsx
```

日常导入单批次：

```bash
.venv/bin/python estimator/scripts/run_ingest_batch.py \
  --input ingestion_data/excel_inputs/audit_ocr_export_20260630_001.xlsx
```

不传 `--batch-id` 时，脚本会从文件名解析 `batch_id`：

```text
audit_ocr_export_20260630_001.xlsx -> 20260630_001
```

如需手动指定批次：

```bash
.venv/bin/python estimator/scripts/run_ingest_batch.py \
  --input ingestion_data/excel_inputs/audit_ocr_export_20260630_001.xlsx \
  --batch-id 20260630_001
```

单批次产物固定为：

```text
ingestion_data/cleaned_inputs/{batch_id}/ocr_required_cleaned.xlsx
ingestion_data/classified_outputs/{batch_id}/classified_projects.xlsx
samples/{batch_id}/cost_item_samples.xlsx
```

`ocr_required_cleaned.xlsx` 包含两个 sheet：`cleaned` 保存必填字段完整行，`removed` 保存被过滤行和缺失字段原因。

如果任一批次产物已存在，默认报错。确认要重跑并覆盖该批次时显式传：

```bash
.venv/bin/python estimator/scripts/run_ingest_batch.py \
  --input ingestion_data/excel_inputs/audit_ocr_export_20260630_001.xlsx \
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
.venv/bin/python estimator/scripts/merge_cost_item_sample_batches.py
```

默认等价于：

```bash
.venv/bin/python estimator/scripts/merge_cost_item_sample_batches.py \
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
.venv/bin/python estimator/scripts/merge_cost_item_sample_batches.py \
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
.venv/bin/python estimator/scripts/build_cost_item_embedding_index.py --overwrite
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
.venv/bin/python estimator/scripts/build_cost_item_embedding_index.py --overwrite
```

也可以按需传参覆盖默认值，例如：

```bash
.venv/bin/python estimator/scripts/build_cost_item_embedding_index.py \
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

目录分类字段仍保留在 `samples.parquet` / `project_packages.parquet` 中用于追溯和 LLM 输入上下文，但不进入 package/item embedding 文本，也不参与候选综合评分。

当前阶段不使用 Milvus，不使用 LangChain；每次合并后允许重建整个本地 parquet + npy + `index_meta.json`。

### 5. 自然语言造价查询

本系统不是通用联网报价工具，而是基于内部历史审价样本和相似工程包的离线估价辅助工具。程序确定性选择历史工程包；LLM 只裁剪一个连续清单区间，并判断用户明确数量能否直接对应最终清单。清单名称、项目特征、单位、工艺、历史单价、来源样本和估算金额全部由程序按原记录和 `fine_signature` 回填或计算。

系统对全部召回工程包稳定排序并计算平均清单数，只在相似度前 5 中选择清单数距平均值最近的工程；距离相同时选择相似度排名更高者。选中工程完整展开后，区间 LLM 只返回连续的 0-based 起止位置，失败时使用完整工程。最终项沿用历史样本原 `practice_option_id`，价格从 top 20 工程包与 top 300 直接清单形成的完整证据池回填。

运行 `query_cost_estimate_llm.py` 前需先启动 LM Studio Server 或兼容 OpenAI `chat/completions` 的本地 LLM 服务；脚本启动时会先检查 `LMSTUDIO_BASE_URL/models`，服务不可用会快速退出。

```bash
.venv/bin/python estimator/scripts/query_cost_estimate_llm.py \
  --text "屋面墙面漏水，想做3mm SBS防水，面积大概500平"
```

正式查询调用链为：

```text
estimator/scripts/query_cost_estimate_llm.py
→ estimator.query_pipeline.run_estimate_query(...)
→ estimator.query_models.QueryResult
→ estimator.reporting.frames.build_workbook_frames(result)
→ estimator.reporting.excel_writer.write_estimate_workbook(...)
```

查询脚本只负责参数、路径、覆盖校验、LM Studio 可用性检查、写文件和终端摘要；业务编排只存在于 `estimator/query_pipeline.py`。workbook sheet 顺序和 `QueryResult` 字段映射统一定义在 `estimator/reporting/columns.py`，frames 只在写 Excel 前临时构造，不存入 `QueryResult`。

方案生成默认关闭；需要生成说明时显式增加 `--with-explanations`。

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
.venv/bin/python estimator/scripts/query_cost_estimate_llm.py \
  --index-dir embeddings \
  --text "屋面漏水，想做3mm SBS防水，面积大概500平" \
  --output query/test_result.xlsx \
  --overwrite
```

如果输出文件已存在，需要显式传 `--overwrite`。

`--package-weight-temperature` 控制工程包证据权重 softmax 的平滑程度，默认 `0.10`，必须大于 0。

新查询流程：

```text
用户查询
→ LLM query rewrite
→ 标准地级行政区域与绝对日期约束校验
→ 按 location / consultation_time 同步过滤工程包、清单行及对应 embedding
→ 受约束范围内的工程包与清单行 embedding 召回
→ retrieved_evidence_items
→ evidence package universe 与 package_evidence_weight
→ fine_signature family 聚合
→ 按 cost_item_name + unit 形成 display groups
→ 按 display 重新计算 retrieval_package_support_ratio
→ display_option_grouping 为全部候选 display 形成完整 practice_options
→ 程序按相似度与平均 item_count 确定性选择一个历史工程包
→ 从完整 samples 按原始顺序展开所选工程全部清单
→ range_selection LLM 只返回一个连续的 0-based 起止区间，失败回退完整工程
→ 最终项沿用历史 practice_option_id
→ quantity_determination 只绑定能直接匹配最终清单的用户明确数量
→ 其余清单按选用 practice option 的全库同类样本工程量中位数暂估
→ 程序按选用的 practice option 回填单价并计算合价
→ 客户展示后处理过滤价格证据样本数小于 3 的最终项
→ 仅按保留项重新汇总方案金额、人工费、机械费和参考证据
→ 可选 final_explanation 生成基于历史清单的组合维修参考方案说明
→ estimate_summary / estimate_scenarios
```

召回数据层次：

```text
project retrieval ─┐
                   ├─ retrieved_evidence_items
item retrieval ────┘
          ↓ fine_signature
candidate_families
          ↓ cost_item_name + unit
candidate_display_groups
```

查询阶段不使用 LLM 选择工程包。连续区间 LLM 只接收用户原始需求、所选工程名称和按原顺序排列的必要清单字段，不接收内部 ID、工艺候选或价格。

Query Rewrite 固定输出 `project_package_query_text`、`item_query_text`、`location`、`start_date`、`end_date`。程序将运行当天以 `当前日期：YYYY-MM-DD` 显式传给 LLM，相对时间由 LLM 转为绝对日期。普通地级市使用“省级行政区 + 地级市”（例如 `浙江省嘉兴市`），直辖市使用 `北京市`、`上海市`、`天津市` 或 `重庆市`。程序不缩写地域、不推断县级行政区归属，也不自动放宽地域或时间范围。

`consultation_time` 查询时严格按 `%Y-%m-%d` 解析，但不覆盖索引中的原始字符串。无时间约束时非法日期行仍可参与召回；有时间约束时非法日期行会被排除。约束后没有工程包时查询明确失败，不静默回退全库。

最终价格仍从 top 20 工程包与 top 300 直接清单构成的完整证据池回填。

输出 xlsx 固定包含：

- `estimate_summary`：面向客户的单方案估价摘要，保留用户原始问题，展示历史清单组合说明、主要施工内容、计价项目数、参考项目数、参考样本数、合价区间以及方案级人工费和机械费金额汇总。
- `estimate_scenarios`：面向客户的项目级主表，展示清单名称、项目特征、工程量、历史工程量统计、价格证据样本数、综合单价、暂估合价、价格区间、人工费/机械费单价组成参考和来源样本；不再展示工程量来源和工程量说明。
- 价格证据样本数小于 3 的最终项不进入上述两个客户表，也不参与方案金额或 final explanation；其完整计算结果和价格证据仍保留在调试 sheet 中。
- `option_evidence_expansion`：展示每个最终清单的价格证据 family 和样本数在扩充前后的变化。扩充只对原 practice option 去重价格证据少于 10 条的最终项执行；LLM 结合清单名称、项目特征和辅助标签综合判断最多 20 个同单位候选是否与代表 family 属于同一价格统计口径。程序严格校验候选白名单，并对明确的单位、厚度、材料和楼层/高度冲突进行兜底拒绝，不再以四项标签字符串完全一致作为最终判断。原证据不少于 10 条或没有候选时不调用 LLM，仍保留 sheet 与 trace 记录。
- `price_evidence_items`：保留所有最终计算项的完整去重价格证据，包括在客户展示中因样本数不足而被过滤的项；同时保留 `project_key` 和 `project_package_id` 用于参考工程去重与调试回查。
- `matched_project_packages`：工程包级召回结果，包括 `package_query_similarity`、`project_package_id`、工程名称、`project_name_text`、`cost_item_names_summary`、`consultation_time`、`location`、`cache_subject` 和 `item_count`。
- `direct_item_hits`：清单行级直接召回结果，参与生成 `retrieved_evidence_items`，不单独输出为 sheet。
- `retrieved_evidence_items`：工程包召回与清单行召回合并后的逐行结果；输出时体现为回填 `family_id` 后的 `evidence_items`。
- `package_evidence_weights`：本次查询证据工程包全集的权重表，包括 `project_package_id`、`package_query_similarity`、`package_evidence_weight`；权重和为 1，direct item 引入但未进入 package top-k 的工程包也会按相似度获得连续权重。
- `candidate_families`：按规范化 `fine_signature = cost_item_name + project_description + unit` 聚合的候选施工做法统计。当前没有新增第二套 family 规范化规则；每个 family 是唯一可报价边界，3mm/4mm、自粘/热熔、单层/双层等不同规格不会合并；表内保留 `package_query_similarity最大值`、`item_query_similarity最大值` 和价格样本统计用于追溯。
- `candidate_display_groups`：按 normalized `cost_item_name + unit` 组织出来的客户展示候选。display 只用于减少同名清单项重复展示，不合并同组 family 的价格样本；`retrieval_package_support_ratio` 表示本次召回工程包证据对该 display 的加权支持比例。
- `display_group_families`：display 到内部 family 的映射表，用于从 display 回查 `fine_signature` 和 evidence。
- `display_option_grouping_trace`：记录全部候选 display 内部的 practice options 及其覆盖的 family；单 family display 由程序直接生成唯一 option，多 family display 合并为一次 LLM 分组调用。
- `matched_project_examples`：按召回 rank 展开前五个历史工程包的完整清单行，保持原工程 item 顺序，作为调试信息。
- `evidence_items`：来源样本明细，保存本次查询进入候选池的历史清单行。`source_ref = project_key + "::" + item_row_id`，`family_id` 和 `fine_signature` 可用于从历史样本回查所属 family。
- `parse_info`：本次查询解析结果和检索参数，包括 Query Rewrite 地域与绝对日期、约束校验 notes、工程包/样本过滤前后数量、无效 consultation_time 数量、retrieved evidence 行数、family 数、display 数、LLM 输入/输出规模、token、fallback、错误和 warnings。
- `llm_trace`：记录 query rewrite、display option grouping、range selection、quantity determination 和可选 final explanation 的 prompt、原始响应、解析状态、token、输入摘要和错误；quantity determination 另以 `quantity_items` JSON 保存每项来源、LLM 数量、最终数量、历史统计和 fallback 状态。

`fine_signature` 会对已确认的等价表达做受控归一化，例如：

- `1.3.0厚` / `厚3.0mm` / `3.0mm厚` / `3.0mm` 统一为 `3.0mm`。
- `SBS防水卷材` / `SBS沥青防水卷材` / `SBS改性沥青防水卷材` / `弹性改性沥青防水卷材` / `弹性体改性沥青防水卷材` 统一为正式名称 `弹性体改性沥青防水卷材`。
- `m2` / `m²` / `m^{2}` / `平方米` 统一为 `m²`。
- `原有` / `原`、`铲除` / `拆除`、`垃圾外运` / `垃圾清运` 只作为已确认等价文本做受控统一。

不同厚度、施工方式、层数、耐根穿刺、附加层、基层处理、平面/立面、砂面等仍保持独立 family，不会因为材料同义词归一化而合并。

LLM 职责边界：

- query rewrite Prompt：`estimator/retrieval/query_rewrite.py`。
- display_option_grouping Prompt：`estimator/candidates/practice_options.py`，把候选 display 内的 family 按具体工艺和价格统计口径完整整理为 practice_options，不选择默认 option。
- range_selection Prompt：`estimator/planning/range_selection.py`，只在程序选定的完整历史工程中返回一个连续起止区间；调用、解析或校验失败时回退完整工程。
- option_selection Prompt：`estimator/planning/option_selection.py`，只能选择当前 display 已有 Option。
- quantity_determination Prompt：`estimator/planning/quantity_determination.py`，只为最终区间的每个绝对 `item_position` 判断 `user_explicit` 或 `historical_median`。用户明确数量只绑定部位、项目名称、材料、规格和单位直接匹配的清单，其余项目由程序采用全库同类历史样本工程量中位数，不在项目之间推导或复制数量。
- evidence expansion Prompt：`estimator/pricing/evidence_expansion.py`。
- final_explanation Prompt：`estimator/reporting/explanation.py`，仅在传入 `--with-explanations` 且存在达到最低证据要求的展示清单时调用，只读取过滤后的清单、工程量、价格和汇总金额，生成“历史清单驱动的组合维修参考方案”。未开启或调用失败时使用程序 fallback，不影响主查询成功。
- LLM 不生成单价、来源、清单名称、单位或合价。综合单价和合价由程序根据 scenario 选用的 practice option 回填和计算。

历史工程量中位数只用于初步估算，不代表现场确认工程量。综合单价始终来自检索样本；若同类证据没有有效工程量，程序仅可明确标记后采用最佳召回样本的有效工程量，最佳样本也无有效值时直接失败。

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
.venv/bin/python estimator/scripts/run_ingest_batch.py \
  --input ingestion_data/excel_inputs/audit_ocr_export_20260630_001.xlsx
```

当天第二批：

```bash
.venv/bin/python estimator/scripts/run_ingest_batch.py \
  --input ingestion_data/excel_inputs/audit_ocr_export_20260630_002.xlsx
```

合并所有历史样本：

```bash
.venv/bin/python estimator/scripts/merge_cost_item_sample_batches.py
```

重建 embedding：

```bash
.venv/bin/python estimator/scripts/build_cost_item_embedding_index.py --overwrite
```

新增 OCR 文件时，不需要人工合并 Excel。每批中间结果都会独立保留，便于检查、回滚和重跑。

所有会覆盖已有产物的操作都必须显式传 `--overwrite`。

### 7. 文件提交说明

以下目录和文件为运行产物，不提交 Git：

```text
excel_outputs/
ingestion_data/cleaned_inputs/
ingestion_data/classified_outputs/
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

### 8. 启动分类服务

启动后端：

```bash
.venv/bin/python -m uvicorn app:app --app-dir classifier/backend --reload
```

启动前端：

```bash
cd classifier/frontend
npm run dev
```
