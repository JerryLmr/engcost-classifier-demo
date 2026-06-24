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
index_meta.json
```

这些文件是运行产物，不需要提交 Git。

### 5. LLM 自然语言估价实验

示例：用户只输入口语化维修需求，由 LLM 抽取检索 profile，再用相似清单样本统计参考价格。

```bash
backend/.venv/bin/python scripts/query_cost_estimate_llm.py \
  --index-dir outputs/cost_item_index \
  --text "屋面漏水，想做3mm SBS防水，面积大概500平" \
  --output outputs/estimate_llm.xlsx \
  --overwrite
```

输出文件：

```bash
outputs/estimate_llm.xlsx
```

其中：

- `summary` sheet：LLM 解析出的工程场景、清单对象、做法特征、内部预测目录、相似度和 warning。
- `unit_price_by_unit` sheet：按历史样本单位分组的综合单价统计；缺少用户单位时不同单位不会混合估价。
- `matches` sheet：top-k 重排后的相似清单样本明细，包含三路语义分数、目录匹配和单位匹配。

所有输出默认不覆盖；如需覆盖已有结果，请显式传入 `--overwrite`。

说明：

- 用户不需要提供 `catalog_id`；系统会用工程分类结果生成内部 `catalog_id`，仅用于同类样本强化和解释。
- 用户不一定提供单位或数量；缺少单位时按历史样本单位分组，缺少数量时只给单价参考，不估算总价。
- `--min-score` 默认 `0.6`，用于筛选进入价格统计的可信样本。
- LLM 只负责解析输入，不直接估价；价格区间来自已审定清单样本的相似项统计，仅用于维修项目初案估算参考。


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
