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
item_similarity_embeddings.npy
item_context_embeddings.npy
index_meta.json
```

这些文件是运行产物，不需要提交 Git。

### 5. 查询相似清单并估算价格区间

示例：估算 500 平方米屋面卷材防水的参考价格。

```bash
backend/.venv/bin/python scripts/query_cost_item_estimate.py \
  --index-dir outputs/cost_item_index \
  --query "屋面卷材防水；3.0mm SBS沥青防水卷材；含基层清理" \
  --context "屋面漏水维修工程" \
  --unit "m²" \
  --quantity 500 \
  --top-k 10 \
  --output outputs/cost_estimate_result.xlsx \
  --overwrite
```

输出文件：

```bash
outputs/cost_estimate_result.xlsx
```

其中：

- `summary` sheet：参考单价区间、估算总价区间、样本数量、说明。
- `matches` sheet：top-k 相似清单样本明细。

所有输出默认不覆盖；如需覆盖已有结果，请显式传入 `--overwrite`。

说明：

```text
参考区间来自已审定清单样本的相似项检索结果，仅用于维修项目初案估算参考，不替代正式造价审核。
```


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
