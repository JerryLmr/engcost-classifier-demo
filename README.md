## 已审定清单样本库与初案估价流程

本流程用于将 OCR 识别出的已审定维修项目资料转为清单级样本库，并基于相似历史工程召回给新维修初案提供参考价格区间。

当前流程不用于判断“异常 / 违规 / 不合理”，只输出相似样本和参考区间。

### 1. 过滤 OCR 必填字段

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

先过滤缺少必填 OCR 字段的行：

```bash
backend/.venv/bin/python scripts/filter_required_ocr_rows.py \
  excel_inputs/audit_ocr_export.xlsx \
  --clean-output cleaned_inputs/ocr_required_cleaned.xlsx \
  --removed-output outputs/ocr_required_removed.xlsx \
  --overwrite
```

输出文件：

```bash
cleaned_inputs/ocr_required_cleaned.xlsx
outputs/ocr_required_removed.xlsx
```

`cleaned_inputs/ocr_required_cleaned.xlsx` 只包含 6 个必填字段都不为空的行，后续 `batch_classify` 只处理该文件。`outputs/ocr_required_removed.xlsx` 保留被移除行和缺失字段原因，便于回查原始 OCR 数据。

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
  cleaned_inputs/ocr_required_cleaned.xlsx \
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
project_name_text
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

其中 `project_name_text` 是 batch 分类阶段从原始工程名称抽取出的项目级语义文本，用于后续相似工程主召回。抽取时会去掉地点、时间、面积、金额、小区名、业主单位、地址、楼栋号、编号和参考造价等非工程语义，保留维修对象、部位、故障/病害、维修动作、材料、工艺和规格。

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
project_name_text
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
project_name_embeddings.npy
project_detail_embeddings.npy
index_meta.json
```

同时生成便于人工检查的工程组调试表：

```bash
outputs/cost_item_project_groups.xlsx
```

索引阶段不再调用 LLM 清洗工程名称，只读取样本中的 `project_name_text`。如果该字段为空，会打印 warning 并回退为原始 `工程名称`。`project_detail_text` 由同一历史工程下的 `cost_item_name + project_description` 去重拼接，不拼入工程名称。

这些文件是运行产物，不需要提交 Git。

构建索引阶段可使用 sentence-transformers 默认设备策略；如果本机环境可用 GPU，可让构建过程使用 GPU 加速。

### 5. 自然语言造价查询

示例：用户只输入口语化维修需求，系统调用本地 LLM 将 `--text` 解析为结构化 `ParsedQuery`，再由程序执行 project group 双路 embedding 检索、location / consultation_time 硬过滤、历史工程下样本展开和推荐清单项聚合。

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

- `recommend_items` sheet：领导展示主表，按相似工程展开后的历史清单项聚合，默认使用结构化数值列展示一级/二级分类、维修状态、清单项名称、项目特征/施工工艺、单位、样本数、历史工程量、估算金额、综合单价、历史总价、人工单价、机械单价和来源清单行；需要把单位和金额单位拼入数值单元格时加 `--display`。
- `matches` sheet：内部追溯表，展示相似工程展开后的清单样本行，并保留 `project_score`、`project_name_score`、`project_detail_score`。默认不输出历史工程名称和召回文本；需要调试明文时加 `--include-debug-text`，会输出 `工程名称`、`project_name_text`、`project_detail_text`。

所有输出默认不覆盖；如需覆盖已有结果，请显式传入 `--overwrite`。

说明：

- LLM 只负责解析 `semantic_query_text`、`quantity`、`unit`、`location_hint`、`time_range_type`，不参与价格估算，不生成 answer，不推荐清单。`semantic_query_text` 与分类阶段 `project_name_text` 使用同一套工程语义抽取口径。
- 程序会校验并归一化 LLM 输出；LLM 失败时回退为原始 `--text` 检索，不中断查询。
- `location` 和 `consultation_time` 是结构化硬过滤条件，不参与 embedding；过滤后无结果时会输出空的 `recommend_items` 和 `matches`。
- 主链路为“ParsedQuery → 双路 project group embedding 加权召回 → 展开 source_row_id 下清单样本 → 聚合 recommend_items”，不加一级分类过滤。
- 查询召回只使用 LLM 解析出的单一 `semantic_query_text`，同时计算工程名称和工程明细两路 embedding 相似度；默认权重为 `--project-name-weight 0.85`、`--project-detail-weight 0.15`，权重和不为 1 时会自动归一化。
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
