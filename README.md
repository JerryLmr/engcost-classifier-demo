## 已审定清单样本库与初案估价流程

本流程用于将 OCR 识别出的已审定维修项目资料转为清单级样本库，并基于相似清单检索给新维修初案提供参考价格区间。

当前流程不用于判断“异常 / 违规 / 不合理”，只输出相似样本和参考区间。

### 1. 输入文件

OCR 原始结果放在：

```bash
excel_inputs/audit_ocr_export.xlsx
```

输入 Excel 需要包含 6 列：

```text
file_name
consultation_project_name
consultation_time
renovation_content
sub_item_project_rows
location
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
consultation_time
location
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
consultation_time
location
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

### 5. 自然语言造价查询

示例：用户只输入口语化维修需求，系统调用本地 LLM 将 `--text` 解析为结构化 `ParsedQuery`，再由程序执行 project group embedding 检索、location / consultation_time 硬过滤、历史样本展开和推荐清单项聚合。

```bash
backend/.venv/bin/python scripts/query_cost_estimate_llm.py \
  --index-dir outputs/cost_item_index \
  --text "屋面漏水，想做3mm SBS防水，面积大概500平，参考嘉兴一年内的造价" \
  --output outputs/cost_estimate_result.xlsx \
  --overwrite
```

输出文件：

```bash
outputs/cost_estimate_result.xlsx
```

其中：

- `recommend_items` sheet：领导展示主表，按相似工程展开后的历史清单项聚合，展示工程量、综合单价、历史总价、人工单价、机械单价和估算总价区间。
- `matches` sheet：内部追溯表，展示相似工程展开后的清单样本行。默认不输出历史工程名称、历史项目描述和 `group_text`；需要调试明文时加 `--include-debug-text`。

所有输出默认不覆盖；如需覆盖已有结果，请显式传入 `--overwrite`。

说明：

- LLM 只负责解析 `semantic_query_text`、`quantity`、`unit`、`location_hint`、`time_range_type`，不参与价格估算，不生成 answer，不推荐清单。
- 程序会校验并归一化 LLM 输出；LLM 失败时回退为原始 `--text` 检索，不中断查询。
- `location` 和 `consultation_time` 是结构化硬过滤条件，不参与 embedding；过滤后无结果时会输出空的 `recommend_items` 和 `matches`。
- 主链路为“ParsedQuery → project group 召回 → 展开 source_row_id 下清单样本 → 聚合 recommend_items”。
- 如果解析出用户工程量且单位与推荐项 `unit_normalized` 一致，估算总价按“用户工程量 × 历史综合单价区间”计算；否则展示历史样本总价范围。
- 查询阶段 embedding 模型固定使用 CPU，避免和 LM Studio 抢 GPU/显存。

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
