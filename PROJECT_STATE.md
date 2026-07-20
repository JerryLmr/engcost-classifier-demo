# PROJECT STATE

## Current Milestone
- Stage 3 code restructuring completed.
- Manual fixed-query regression pending.
- 查询编排、输出 frame、Excel 写入与最终说明已有正式模块；真实固定 query 仍待用户人工回归。

## System Capabilities
- 支持 OCR Excel 批次清洗、分类、样本展开与跨批次稳定去重。
- 支持从总样本构建 project package / item embedding 索引，并保留样本、工程包、向量和索引元数据。
- 支持自然语言造价查询：约束过滤、工程包与清单召回、Family / Display / Option 候选、连续区间选择、工程量决策、历史数量与价格统计及金额计算。
- 支持生成固定 sheet 顺序的 XLSX；项目级说明默认关闭，可通过 `--with-explanations` 启用。
- 支持通过 source_ref、stable_sample_id、family / display / option ID 和 LLM trace 回查估价证据与决策。

## Recent Changes
- Estimator 第一阶段拆分 `ingestion / indexing / retrieval`，相对数据路径统一按仓库根目录解析。
- Estimator 第二阶段拆分 `candidates / planning / pricing`，quantity decision 与 pricing calculation 保持独立。
- Estimator Stage 3 新增 `query_pipeline.py` 和 `output/`，正式查询 pipeline 返回既有 `QueryResult`，不写文件。
- Workbook frames 仅在写 Excel 前由 `build_workbook_frames(result)` 临时生成；sheet 顺序与 `QueryResult` 字段映射只定义一份。
- `estimator/scripts/query_cost_estimate_llm.py` 收敛为正式命令入口，不再 re-export 业务函数或维护旧 facade。
- 现有 estimator 测试已改为直接 import 和 patch 正式模块。

## Decisions
- 代码按职责分为 `classifier/` 与 estimator 的 `ingestion / indexing / retrieval / candidates / planning / pricing / output`；估价查询入口仍为 `estimator/scripts/query_cost_estimate_llm.py`。
- `QueryResult` 是 pipeline 的唯一结果容器，不新增 frames、traces、metadata 或重复 DataFrame 副本。
- Writer 只接收已准备好的 workbook frames；业务模块和 query pipeline 不依赖旧查询脚本。
- 当前阶段不引入数据库、Milvus、LangChain、`estimator/cli` 或共享 `estimator/llm` 包。
- 批次产物、索引产物和查询输出不允许静默覆盖，覆盖必须显式传 `--overwrite`。
- LLM 不生成单价、来源、清单名称、单位或金额；价格、工程量统计和金额由程序确定性处理。
- Option 是同一展示项下业务等价 Family 的集合；最终价格证据按 Option Family 的 normalized signature 从全量样本精确展开。

## Known Limitations
- 真实固定 query 尚未在本次职责迁移后人工回归；当前结论只覆盖 mock、import、CLI help 和自动测试。
- 完整分类质量仍依赖本地 LLM 与真实数据回归；自动测试主要验证链路、字段结构和标准目录 ID 校验。
- OCR 仍只读取 active sheet；总样本合并只合并 `samples` sheet。
- 首次构建或查询索引时可能需要下载 `BAAI/bge-m3` 或用户指定模型。
- 当前造价查询不执行额外目录分类或一级分类硬过滤，价格区间只作为初案参考。
- 初版不允许跨历史工程补项或创建新清单。

## Next Steps
- 用户使用固定屋面 3mm SBS query 人工比较 Stage 3 前后的 sheet、列、ID、选择、统计、金额、fallback 和 trace。
- 人工回归通过后再准备 Stage 3 提交。
- 使用真实新增 OCR 批次验证导入、样本去重和索引重建流程。
- 根据样本规模增长情况评估是否引入 FAISS 或其它向量索引。
