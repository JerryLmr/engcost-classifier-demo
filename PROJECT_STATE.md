# PROJECT STATE

## Current Milestone
- Demo 已完成从“物业工程名称输入”到 CP/CF 标准目录输出的稳定闭环；当前标准目录管线只保留完整 compact 目录交给 LLM 判断，一二级目录由 LLM 直接在标准目录内选择。

## System Capabilities
- 支持单条工程名称分类，返回一级分类、二级分类、维修状态、分类依据、复合工程、紧急维修、白蚁相关和建议复核字段。
- 支持 Excel 批量分类，自动输出分类结果、复合工程标记、复合目录和复核相关列并下载结果文件。
- CP/CF 标准目录管线采用完整 compact 标准目录 LLM 选择：normalizer、alias、动作词和复核提示只提供上下文，最终 catalog_id 经标准目录 id 校验后进入维修状态判断。
- 支持复合工程的其它标准目录输出：`secondary_catalog_ids` / `secondary_catalog_labels`。
- 支持基于目录的批量结果分析，并可导出 Excel 汇总报表。
- 前端支持上传已分类结果文件并展示摘要统计、分类分布、异常/复杂样本。

## Recent Changes
- 已删除旧 n-gram 候选召回链路、family fallback 配置和 `CLASSIFIER_USE_FULL_CATALOG` 开关。
- 已删除旧候选目录输出字段；当前 API、Excel 和分析结果不再包含“候选目录”或 `candidate_labels`。
- 已保留 alias 作为 full catalog prompt 的辅助扩展词，但 alias 不绑定、不直接决定 catalog_id。
- 已将目录展示格式迁移为 `catalog_label(item)`，用于复合工程的 `secondary_catalog_labels`。
- 已更新批量分类、上传分类、结果分析和前端展示以适配 full catalog only 输出。

## Decisions
- 当前阶段以“LLM 直接理解完整标准目录、规则只作提示、输出结构不保留旧候选兼容字段”为优先目标。
- CP/CF 标准目录不再维护旧候选召回对比路径；分类入口始终走 full catalog 主链路。
- alias、normalizer、动作词和复核提示只作为文本理解上下文，不直接决定最终 catalog_id，也不早期直接输出 OUT_OF_SCOPE。
- 完整目录链路中，只有 LLM 返回标准目录外 id 或明确 OUT_OF_SCOPE 时才进入体系外 fallback；标准目录内 id 直接接受并继续判断维修状态。
- `secondary_catalog_ids` / `secondary_catalog_labels` 仅表示复合工程的其它标准目录，不是候选召回结果。

## Known Limitations
- 当前分类体系仍是项目内自行定义，个别样本是否属于“体系外”依赖业务口径，不代表分类体系最终定稿。
- 完整分类质量仍依赖本地 LLM 回归，自动测试主要验证链路、字段结构和标准目录 id 校验。
- 仍存在少量需要业务确认的边界，例如 `楼宇对讲 / 门禁 / 弱电系统`、`道路工程 / 停车交通`。
- 当前全量样本中仍有少量 `体系外默认分类`，例如幕墙玻璃补漏打胶类样本，暂未单独扩类。
- 前端 `API_BASE` 仍默认指向本地 `http://127.0.0.1:8000`，跨环境部署时需要显式配置。
- API 自动化测试在缺少 `httpx` 时会跳过，测试环境依赖仍需补齐。

## Next Steps
- 用 OUT41 和后续新增真实样本持续验证 full catalog 标准目录管线。
- 与领导确认剩余业务边界后，再决定是否调整分类体系或继续细化目录。
- 若分类体系稳定，下一阶段可考虑提升 alias 词典维护体验，例如配置校验、可编辑流程或简单管理界面。
- 若前端继续增强，可在分析区增加排序、筛选或详情展开，而不是继续堆叠更多规则说明。
