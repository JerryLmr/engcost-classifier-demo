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
item_embeddings.npy
project_embeddings.npy
full_embeddings.npy
project_groups.parquet
project_group_embeddings.npy
index_meta.json
```

这些文件是运行产物，不需要提交 Git。

构建索引阶段可使用 sentence-transformers 默认设备策略；如果本机环境可用 GPU，可让构建过程使用 GPU 加速。

### 5. LLM 自然语言估价实验

示例：用户只输入口语化维修需求，系统先用 `classify_project_standard(raw_text)` 做标准目录分类，再用原始输入 `raw_text` 检索同目录历史工程，聚合这些工程中的清单项组合，最后由 LLM 基于结构化推荐结果生成自然语言总结。

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

- `answer` sheet：面向最终问答形态的自然语言总结，只基于 `recommended_items` 前若干条结构化证据生成。
- `summary` sheet：原始输入、标准目录分类结果、同目录工程数量、推荐清单项数量和 warning。
- `recommended_items` sheet：主结果，按同目录历史工程中的清单项签名聚合，展示出现次数、支持工程数、支持率和综合/人工/机械单价分位数。
- `debug_item_matches` sheet：全库 item embedding 调试召回结果，查询文本同样使用 `raw_text`，只用于观察语义召回，不作为业务结论。

所有输出默认不覆盖；如需覆盖已有结果，请显式传入 `--overwrite`。

说明：

- 用户不需要提供 `catalog_id`；`catalog_id` 来自 `classify_project_standard(raw_text)`。
- 本脚本不使用 LLM profile 影响主检索；工程召回和调试召回都直接使用 `raw_text`。
- 查询阶段 embedding 模型固定使用 CPU，并在生成 LLM answer 前释放 embedding model；这样可以避免和 LM Studio 抢 GPU/显存。
- 本脚本新增的 LLM answer 只负责基于 `recommended_items` 总结，不负责标准目录分类，不新增未出现的清单项，不编造价格。
- 如果样本库缺少同标准目录历史工程，系统会提示样本不足，不强行用 CP/CF 前缀相同的其它目录给价格。
- 价格参考来自 `recommended_items` 中的同类历史工程清单项聚合结果，仅用于维修项目初案估算参考。


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
