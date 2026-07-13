# PROJECT STATE

## Current Milestone
- Demo 已进入第二阶段：基于已审定清单样本表构建 project package / cost item embedding 索引，为维修项目初案估算提供可追溯的历史单价和金额参考区间。

## System Capabilities
- 支持单条工程名称分类，返回一级分类、二级分类、维修状态、分类依据、复合工程、紧急维修、白蚁相关和建议复核字段。
- 支持 OCR Excel 批量分类，只读取 active sheet，并保留 OCR 追溯字段。
- 支持按批次导入 OCR Excel，独立生成 cleaned、removed、classified 和单批次 cost item samples。
- 支持自动合并 `samples/*/cost_item_samples.xlsx` 为总样本，并使用不含 `project_code` / `batch_id` 的 `stable_sample_id` 去重。
- 支持从总样本 `samples/cost_item_samples_all.xlsx` 构建 project package / item embedding 索引，保留样本明细、工程包明细、工程包向量、清单行向量和索引元数据。
- 支持自然语言造价查询：从前三个完整真实历史工程中选择一个 `project_package_id` 作为唯一方案骨架，按 `stable_sample_id` 裁剪清单、在同 display 内确定工艺，并从完整召回证据池回填价格和金额。

## Recent Changes
- 新增 `scripts/merge_cost_item_sample_batches.py`，自动合并历史批次样本、追加 `batch_id` / `stable_sample_id`，并输出去重报告。
- `build_cost_item_embedding_index.py` 默认读取 `samples/cost_item_samples_all.xlsx`，输出 project package / item embedding 索引。
- 自然语言造价查询的召回证据层次统一为 `matched_project_packages` / `direct_item_hits` → `retrieved_evidence_items` → `candidate_families` → `candidate_display_groups`。
- 删除 display 预筛选 LLM；`display_option_grouping` 全量处理 candidate displays，单 family 由程序生成 option，多 family 合并为一次 LLM 调用。
- 估价方案生成重构为 `historical_plan_determination` 与 `final_explanation` 两阶段：程序只生成 `S001`，清单、工艺、工程量和价格在说明生成前锁定。

## Decisions
- 当前阶段不引入数据库、Milvus 或 LangChain；样本合并后重建本地 parquet + npy 索引。
- OCR xlsx 继续只处理 active sheet，不支持多 sheet 遍历。
- 批次产物和索引产物不允许静默覆盖，覆盖必须显式传 `--overwrite`。
- `batch_id` 只负责来源追踪；`stable_sample_id` 负责样本去重，且不包含 `project_code` 或 `batch_id`。
- 索引构建阶段不再调用 LLM 清洗工程名称，只读取 batch 分类产出的 `project_name_text`；为空时 warning 并回退原始工程名称。
- 查询阶段 LLM 不生成清单名称、项目特征、单位、单价、来源或金额；价格和金额由程序按同一 `fine_signature` 历史样本确定性回填和计算。
- 前三个历史工程只用于选择真实方案骨架；最终 option 价格仍使用 top 20 工程包与 top 300 直接清单形成的完整证据池。
- `stable_sample_id` 是历史清单唯一回查键；允许 `source_ref` 重复，后者只用于展示、调试和人工追溯。
- dedup_selection 只抑制最终展示项，不创建新 family，不合并 source_refs、本次召回样本、工程量或价格区间。

## Known Limitations
- 当前分类体系仍是项目内自行定义，个别样本是否属于“体系外”依赖业务口径。
- 完整分类质量仍依赖本地 LLM 回归，自动测试主要验证链路、字段结构和标准目录 id 校验。
- OCR 多 sheet 文件需要人工确保数据 sheet 是 active sheet。
- 当前合并总样本只合并 `samples` sheet，不合并各批次 `parse_errors`。
- 首次构建或查询 embedding 索引时可能需要下载 `BAAI/bge-m3` 或用户指定的 sentence-transformers 模型。
- 当前造价查询不对用户 query 额外执行目录分类，也不加一级分类硬过滤；价格区间只作为初案估算参考。
- 新 fine_signature 聚合偏保守，宁可拆细施工做法，也不跨规格、材料或工艺混合价格。
- 初版不允许跨历史工程补项；核心维修对象按 direct item 相似度最高的 display 校验，骨架不覆盖时整链失败。

## Next Steps
- 使用真实新增 OCR 批次验证批次导入、样本去重报告和索引重建流程。
- 根据样本规模增长情况，再评估是否引入 FAISS 或其它向量索引。
- 与业务方确认剩余分类边界后，再决定是否调整分类体系或继续细化目录。
- 使用屋面 3mm SBS、消防报警主机等真实查询回归验证历史骨架裁剪、同 display 工艺替换和说明失败降级语义。
