# PROJECT STATE

## Current Milestone
- Demo 已进入第二阶段：基于已审定清单样本表构建相似历史工程检索索引，为维修项目初案估算提供价格参考区间；当前重点是支持 OCR 批次增量导入、样本合并去重和本地索引重建。

## System Capabilities
- 支持单条工程名称分类，返回一级分类、二级分类、维修状态、分类依据、复合工程、紧急维修、白蚁相关和建议复核字段。
- 支持 OCR Excel 批量分类，只读取 active sheet，并保留 OCR 追溯字段。
- 支持按批次导入 OCR Excel，独立生成 cleaned、removed、classified 和单批次 cost item samples。
- 支持自动合并 `samples/*/cost_item_samples.xlsx` 为总样本，并使用不含 `project_code` / `batch_id` 的 `stable_sample_id` 去重。
- 支持从总样本 `samples/cost_item_samples_all.xlsx` 构建项目组双路 embedding 索引，保留样本明细、项目组明细、`project_name_text` 向量、`project_detail_text` 向量和索引元数据。
- 支持自然语言造价查询：按 location / consultation_time 硬过滤后召回相似历史工程，展开工程下清单样本并聚合 recommend_items，计算综合单价、人工单价、机械单价的参考区间和估算金额。

## Recent Changes
- 新增批次入口 `scripts/run_ingest_batch.py`，统一执行 OCR 必填字段过滤、项目级分类和单批次样本生成。
- 新增 `scripts/merge_cost_item_sample_batches.py`，自动合并历史批次样本、追加 `batch_id` / `stable_sample_id`，并输出去重报告。
- `build_cost_item_embedding_index.py` 默认读取 `samples/cost_item_samples_all.xlsx`，继续输出当前项目组索引格式。
- README 已改为增量批次导入、样本合并、索引重建的日常流程。
- 批量分类与查询继续使用 `project_name_text + project_detail_text` 双路项目组召回。

## Decisions
- 当前阶段不引入数据库、Milvus 或 LangChain；样本合并后重建本地 parquet + npy 索引。
- OCR xlsx 继续只处理 active sheet，不支持多 sheet 遍历。
- 批次产物和索引产物不允许静默覆盖，覆盖必须显式传 `--overwrite`。
- `batch_id` 只负责来源追踪；`stable_sample_id` 负责样本去重，且不包含 `project_code` 或 `batch_id`。
- 索引构建阶段不再调用 LLM 清洗工程名称，只读取 batch 分类产出的 `project_name_text`；为空时 warning 并回退原始工程名称。
- 查询阶段只让 LLM 解析 `semantic_query_text`、工程量、单位、地点和时间范围，不生成价格结论。

## Known Limitations
- 当前分类体系仍是项目内自行定义，个别样本是否属于“体系外”依赖业务口径。
- 完整分类质量仍依赖本地 LLM 回归，自动测试主要验证链路、字段结构和标准目录 id 校验。
- OCR 多 sheet 文件需要人工确保数据 sheet 是 active sheet。
- 当前合并总样本只合并 `samples` sheet，不合并各批次 `parse_errors`。
- 首次构建或查询 embedding 索引时可能需要下载 `BAAI/bge-m3` 或用户指定的 sentence-transformers 模型。
- 当前造价查询不加一级分类过滤，价格区间只作为初案估算参考。

## Next Steps
- 使用真实新增 OCR 批次验证批次导入、样本去重报告和索引重建流程。
- 根据样本规模增长情况，再评估是否引入 FAISS 或其它向量索引。
- 与业务方确认剩余分类边界后，再决定是否调整分类体系或继续细化目录。
