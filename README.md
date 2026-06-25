## 已审定清单样本库与初案估价流程

本流程用于将 OCR 识别出的已审定维修项目资料转为清单级样本库，并基于相似清单检索给新维修初案提供参考价格区间。

当前流程不用于判断“异常 / 违规 / 不合理”，只输出相似样本和参考区间。

### 1. 输入文件

OCR 原始结果放在：

```bash
excel_inputs/audit_ocr_export.xlsx
```

输入 Excel 需要包含 4 列：

```text
file_name
consultation_project_name
renovation_content
sub_item_project_rows
```

其中系统会自动生成：

```text
工程名称 = consultation_project_name + " " + renovation_content
```

不需要人工复制粘贴分类结果和 OCR 字段。

### 2. 工程项目级分类

LLM 后端仅支持 LM Studio。启动 LM Studio Server 后，可手动检查：

```bash
curl --max-time 3 "$LMSTUDIO_BASE_URL/models"
```

批量分类脚本会在正式处理前自动执行 LM Studio preflight；服务不可用时会快速失败。

```bash
backend/.venv/bin/python scripts/batch_classify_excel.py \
  excel_inputs/audit_ocr_export.xlsx \
  -o excel_outputs/classified_projects.xlsx \
  --overwrite
```

输出文件：

```bash
excel_outputs/classified_projects.xlsx
```

输出字段顺序为：

```text
工程名称
catalog_id
一级分类
二级分类
维修状态
标准对象
是否复合工程
复合目录
是否紧急维修
是否白蚁相关
是否建议复核
分类依据
file_name
consultation_project_name
renovation_content
sub_item_project_rows
```

### 3. 展开清单级样本

```bash
backend/.venv/bin/python scripts/build_cost_item_samples.py \
  excel_outputs/classified_projects.xlsx \
  -o outputs/cost_item_samples.xlsx \
  --overwrite
```

输出文件：

```bash
outputs/cost_item_samples.xlsx
```

其中 `samples` sheet 每一行代表一条清单样本：

```text
source_row_id + sub_item_project_rows 中的 seq
```

关键字段包括：

```text
file_name
工程名称
sub_project_id
catalog_id
一级分类
二级分类
维修状态
标准对象
复合目录
cost_item_name
project_description
unit_normalized
quantity
unit_price
labor_unit_price
machinery_unit_price
item_similarity_text
item_context_text
```

### 4. 构建 embedding 索引

```bash
backend/.venv/bin/python scripts/build_cost_item_embedding_index.py \
  --samples outputs/cost_item_samples.xlsx \
  --output-dir outputs/cost_item_index \
  --model BAAI/bge-m3 \
  --overwrite
```

输出目录：

```bash
outputs/cost_item_index/
```

包含：

```text
samples.parquet
project_groups.parquet
project_group_embeddings.npy
index_meta.json
```

这些文件是运行产物，不需要提交 Git。

构建索引阶段可使用 sentence-transformers 默认设备策略；如果本机环境可用 GPU，可让构建过程使用 GPU 加速。

### 5. LLM 自然语言估价实验

示例：用户只输入口语化维修需求，系统先用 `classify_project_standard(raw_text)` 做标准目录分类，再用原始输入 `raw_text` 检索同目录历史工程，聚合这些工程中的清单项组合，再由 answer planner 基于 `recommended_items` 规划展示分组，最后用程序模板渲染自然语言总结。

```bash
backend/.venv/bin/python scripts/query_cost_estimate_llm.py \
  --index-dir outputs/cost_item_index \
  --text "屋面漏水维修工程" \
  --output outputs/estimate_llm_roof_leak.xlsx \
  --overwrite
```

输出文件：

```bash
outputs/estimate_llm_roof_leak.xlsx
```

其中：

- `answer` sheet：面向最终问答形态的自然语言总结，由程序模板按 `answer_plan` 渲染，价格和估算金额只来自 `recommended_items`。
- `summary` sheet：原始输入、标准目录分类结果、同目录工程数量、推荐清单项数量、识别工程量、可计算清单项数、是否自动合计和 warning。
- `answer_plan` sheet：answer planner 的展示计划，记录每个候选项的展示分组、处理方式、相近做法归并和排除/提示信息，便于复核最终 answer 结构。
- `matched_projects` sheet：召回的相似历史工程，用于复核工程级证据来源。
- `recommended_items` sheet：主结果，按同目录历史工程中的清单项签名聚合，展示出现次数、支持工程数、支持率和综合/人工/机械单价分位数。

所有输出默认不覆盖；如需覆盖已有结果，请显式传入 `--overwrite`。

说明：

- 用户不需要提供 `catalog_id`；`catalog_id` 来自 `classify_project_standard(raw_text)`。
- 当前脚本是结构化 RAG 单轮查询；检索对象是历史工程和清单项样本，不是普通文档。
- 主链路为“工程组召回 → 展开 source_row_id 下清单项 → 聚合 recommended_items → answer_plan → 程序模板 answer”。
- `matched_projects` 是召回的相似历史工程，`recommended_items` 是从相似工程展开并聚合得到的候选清单项。
- `answer_plan` 是基于 `recommended_items` 的展示规划，不等于完整候选池或最终报价清单。
- `answer_plan.section_title` 是 LLM planner 根据本次 `recommended_items` 生成的 answer 展示分组标题，不是固定枚举，也不能作为稳定业务分类字段使用。
- `answer_plan.plan_action` 是 planner 决策的核心字段；判断一个 item 最终如何处理，应优先看 `plan_action`。
- 判断一个 item 在最终 answer 中放在哪一段，看 `answer_plan.section_title`。
- `debug_item_matches` 已移出默认结果，因为它属于全库单项 embedding 调试，不是当前主链路证据。
- item/project/full 三路清单级 embedding 已从 query 主链路移除。
- 查询阶段 embedding 模型固定使用 CPU，并在生成模板 answer 前释放 embedding model；这样可以避免和 LM Studio 抢 GPU/显存。
- LLM answer planner 只做语义归并、业务分组和展示选择；不重新检索向量库，不新增清单项，不输出价格，不直接生成最终 answer 正文。
- 当前 answer 只负责基于 `answer_plan` 和 `recommended_items` 总结，不负责标准目录分类，不新增未出现的清单项，不编造价格。
- 如果样本库缺少同标准目录历史工程，系统会提示样本不足，不强行用 CP/CF 前缀相同的其它目录给价格。
- 价格参考来自 `recommended_items` 中的同类历史工程清单项聚合结果，仅用于维修项目初案估算参考。
- 如果用户输入含简单工程量，系统会对单位一致且有综合单价分位数的清单项做“综合单价 × 工程量”的 P25-P75 简单参考金额计算。
- 单项金额估算可以展示，但当前不自动合计总价；部分清单项可能属于替代做法、重复候选或条件措施项，需要用户确认适用项后再汇总。
- 后续多轮状态中，LLM 可用于理解用户补充信息和选择适用清单项；用户确认后，再由程序基于 selected_items 计算合计参考金额。


### 6. 文件提交说明

以下目录和文件为运行产物，不提交 Git：

```text
excel_outputs/
outputs/
*.xlsx
*.npy
*.parquet
```

如需提交示例数据，应使用脱敏的小样本文件。
