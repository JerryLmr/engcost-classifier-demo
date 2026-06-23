# PROJECT STATE

## Current Milestone
- Demo 已完成 CP/CF 标准目录分类闭环，并进入第二阶段：基于已审定清单样本表构建相似清单检索索引，为维修项目初案估算提供价格参考区间。

## System Capabilities
- 支持单条工程名称分类，返回一级分类、二级分类、维修状态、分类依据、复合工程、紧急维修、白蚁相关和建议复核字段。
- 支持 Excel 批量分类，自动输出分类结果、复合工程标记、复合目录和复核相关列并下载结果文件。
- 支持 OCR 原始 Excel 直连批量分类，可由 `consultation_project_name` 和 `renovation_content` 自动生成工程名称，并保留 OCR 追溯字段。
- CP/CF 标准目录管线采用完整 compact 标准目录 LLM 选择：normalizer、alias、动作词和复核提示只提供上下文，最终 catalog_id 经标准目录 id 校验后进入维修状态判断。
- 支持复合工程的其它标准目录输出：`secondary_catalog_ids` / `secondary_catalog_labels`。
- 支持基于目录的批量结果分析，并可导出 Excel 汇总报表。
- 前端支持上传已分类结果文件并展示摘要统计、分类分布和重点样本。
- 支持从已审定清单 `samples` sheet 构建双路 embedding 索引，保留样本明细、清单项语义向量、上下文向量和索引元数据。
- 支持按单位和标准目录硬过滤相似清单样本，输出 top-k 相似样本，并计算综合单价、人工单价、机械单价的参考区间和估算金额。

## Recent Changes
- 新增 `build_cost_item_embedding_index.py`，可从已审定清单样本 Excel 构建 parquet + numpy embedding 索引。
- 新增 `query_cost_item_estimate.py`，可对输入清单项进行相似样本检索，并输出终端摘要或 Excel 查询结果。
- 已为第二阶段补充 `pandas`、`numpy`、`sentence-transformers`、`pyarrow` 依赖。
- 已更新 `build_cost_item_samples.py` 的 samples 输出结构，保留 `file_name` 追溯字段。
- 已更新批量分类脚本，可直接读取 OCR 原始四字段并输出 `final_target.xlsx`。
- 已更新批量分类、上传分类、结果分析和前端展示以适配 full catalog only 输出。

## Decisions
- 当前阶段以“LLM 直接理解完整标准目录、规则只作提示、输出结构不保留旧候选兼容字段”为优先目标。
- CP/CF 标准目录不再维护旧候选召回对比路径；分类入口始终走 full catalog 主链路。
- alias、normalizer、动作词和复核提示只作为文本理解上下文，不直接决定最终 catalog_id，也不早期直接输出 OUT_OF_SCOPE。
- 完整目录链路中，只有 LLM 返回标准目录外 id 或明确 OUT_OF_SCOPE 时才进入体系外 fallback；标准目录内 id 直接接受并继续判断维修状态。
- `secondary_catalog_ids` / `secondary_catalog_labels` 仅表示复合工程的其它标准目录，不是候选召回结果。
- 第二阶段只做 embedding 相似样本检索和参考区间估算，不接入 LLM、LM Studio 或 chat/completions API。
- 第二阶段不做关键词加分、施工子目目录推断或价格评判；输出只表达相似样本、参考区间、估算金额、样本数量和注意事项。

## Known Limitations
- 当前分类体系仍是项目内自行定义，个别样本是否属于“体系外”依赖业务口径，不代表分类体系最终定稿。
- 完整分类质量仍依赖本地 LLM 回归，自动测试主要验证链路、字段结构和标准目录 id 校验。
- 仍存在少量需要业务确认的边界，例如 `楼宇对讲 / 门禁 / 弱电系统`、`道路工程 / 停车交通`。
- 当前全量样本中仍有少量 `体系外默认分类`，例如幕墙玻璃补漏打胶类样本，暂未单独扩类。
- 前端 `API_BASE` 仍默认指向本地 `http://127.0.0.1:8000`，跨环境部署时需要显式配置。
- API 自动化测试在缺少 `httpx` 时会跳过，测试环境依赖仍需补齐。
- 首次构建或查询 embedding 索引时可能需要下载 `BAAI/bge-m3` 或用户指定的 sentence-transformers 模型。
- 当 top-k 有效价格样本少于 3 条时，价格区间仍可展示，但只作为初案估算参考。

## Next Steps
- 使用真实已审定样本持续验证相似清单召回质量和价格参考区间稳定性。
- 根据样本规模增长情况，再评估是否引入 FAISS 或其它向量索引。
- 与领导确认剩余业务边界后，再决定是否调整分类体系或继续细化目录。
