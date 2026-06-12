# Codex 任务：将工程分类 Demo 重构为固定三级目录分类器

## 项目目标

当前项目用于把工程项目文本稳定识别到标准工程目录：

```text
一级目录 level1
二级目录 level2
三级目录 level3
```

这一步不是最终目标，而是为后续工程量识别、单价合理性判断、历史均价/区间比对做数据整理基础。

后续场景包括：

- 只有一句项目名称，例如“屋面维修工程15幢”；
- 有项目名称 + 工程概况，例如“消防主机更换，多线盘更换，感烟感温更换，手动报警按钮更换，模块更换等”；
- 有施工合同、工程量清单 PDF，后续会从中抽取更细的施工项目和工程量。

本轮先完成目录和分类器重构，不做复杂造价计算。

## 固定目录

新增唯一目录配置文件：

```text
backend/config/catalog.json
```

格式如下：

```json
{
  "items": [
    {
      "id": "001",
      "level1": "屋面工程",
      "level2": "坡屋面",
      "level3": "维修坡屋面",
      "rules": {
        "object_keywords": ["坡屋面", "屋面工程"],
        "action_keywords": ["维修", "维修坡屋面"],
        "weak_keywords": [],
        "min_score": 2
      }
    }
  ]
}
```

注意：

- 不要添加旧目录兼容逻辑。
- 不要保留 `RULE_SOURCE=python/json` 双轨。
- `id` 是字符串，格式为 `001`, `002`, `003`。
- `catalog.json` 是唯一目录来源。
- 不要让 LLM 创造目录，只能从 catalog items 中选择。

## 必须进行一次性目录搬迁和旧代码清理

这次不做“先保留旧文件、以后再清理”的兼容式改造。请一次性完成结构整理。

建议新结构：

```text
backend/
  app.py

  api/
    routes.py

  classifier/
    __init__.py
    catalog_loader.py
    rule_engine.py
    llm_client.py
    text_normalizer.py

  config/
    catalog.json

  services/
    excel_service.py
    analysis_service.py

  models/
    schemas.py
```

请删除或停止使用旧分类链路相关文件：

```text
backend/core/rule_loader.py
backend/core/rule_validator.py
backend/data/categories.py
backend/data/rules.py
backend/data/boundaries.py
backend/data/structure_rules.py
backend/config/taxonomy.json
backend/config/level1_rules.json
backend/config/level2_rules.json
backend/config/detailed_level2_rules.json
backend/config/boundary_rules.json
backend/config/structure_rules.json
```

如果某些文件因 import 依赖暂时不能物理删除，必须确保运行链路已经完全不依赖它们，并在本轮最后删除无用 import。

`core/config.py` 也请重命名或收敛。这里的配置主要是运行时环境变量，不应再混放规则目录逻辑。建议改成：

```text
backend/classifier/settings.py
```

或者：

```text
backend/settings.py
```

只保留：

```text
OLLAMA_BASE_URL
OLLAMA_MODEL
LLM_TIMEOUT_SECONDS
DEFAULT_FALLBACK_LEVEL1/2/3 如仍需要
```

## 分类逻辑

旧逻辑是先判一级再判二级：

```text
level1 -> level2
```

现在改为直接遍历三级目录候选：

```text
输入文本
↓
normalize
↓
遍历 catalog.items
↓
对每个 item 打分
↓
选择最高分三级目录
↓
由该 item 直接返回 level1 / level2 / level3
```

不要先死判一级，因为很多输入直接命中三级对象，例如：

```text
高压柜
燃气炉
消防栓
污水泵
生活用泵
屋面防水
电视监控控制台
```

## 规则打分建议

对每条 catalog item：

- `level2` 完整命中：+4
- `level3` 里任意短语完整命中：+4
- `object_keywords` 每个命中：+3
- `action_keywords` 每个命中：+2
- `weak_keywords` 每个命中：+1

返回最高分 item。

置信度建议：

```text
score >= 6: 高
3 <= score < 6: 中，needs_review=true
score < 3: 调用 LLM 兜底
```

规则结果字段：

```json
{
  "project_name": "...",
  "level1": "...",
  "level2": "...",
  "level3": "...",
  "method": "规则优先",
  "confidence": "高",
  "reason": "命中对象词：...；命中动作词：...",
  "needs_review": false,
  "candidate_ids": ["001", "002"]
}
```

## LLM 兜底

当规则低置信或无命中时，调用 Ollama。

要求：

- LLM prompt 只能给 catalog items。
- LLM 只能返回一个已有 item id。
- 不允许创造 level1/level2/level3。
- 如果输入是混合工程，可以返回主分类，同时 `needs_review=true`，并在 reason 里说明还有其他候选。
- 如果多个候选接近，返回最主要工程对象对应的 item，同时把候选 id 放入 `candidate_ids`。

建议 LLM 返回：

```json
{
  "id": "001",
  "level1": "...",
  "level2": "...",
  "level3": "...",
  "reason": "...",
  "needs_review": true
}
```

后端收到后必须校验 id 是否存在于 catalog；不存在则降级为默认分类或复核项。

## API / Schema

请更新 `models/schemas.py`：

- `ClassifyResponse` 增加 `level3`
- 增加 `confidence`
- 增加 `needs_review`
- 增加 `candidate_ids`

如果已有字段如 `is_composite`, `structure_type`, `secondary_candidates` 和新目标冲突，可以先删除或停用。当前阶段不需要继续维护旧的复合工程展示逻辑。

## Excel 批量分类

`services/excel_service.py` 输出列改为：

```text
一级分类
二级分类
三级分类
分类方式
置信度
分类依据
是否建议复核
候选目录ID
```

不要再输出旧版复合工程字段，除非后端仍然明确支持。

## 前端

前端单条分类展示增加：

```text
三级分类
置信度
是否建议复核
候选目录ID
```

Excel 上传下载保持可用。

分析页面如果暂时复杂，可以先保证它不会因新增 `三级分类` 报错；后续再做三级统计。

## 测试建议

至少添加或更新几个测试样例：

```text
高压柜更换断路器 -> 强电 / 高压柜 / ...
生活水泵维修 -> 生活用泵 / 生活用泵 / ...
屋面防水维修 -> 屋面工程 / 平屋面 / ...
消防主机更换感烟感温手动报警按钮模块 -> 消防设备 或 弱电火灾报警相关目录，needs_review 可为 true
屋面墙面渗水维修 -> 主分类屋面工程或外立面工程，needs_review=true，候选中保留另一个方向
```

## 重要约束

- 不做旧目录兼容。
- 不做旧 Python 规则兜底。
- 不引入数据库。
- catalog 规模小，直接遍历即可。
- 先保证单条分类、Excel 批量分类、前端展示跑通。
- 不要过度设计插件化、版本化、迁移层。
