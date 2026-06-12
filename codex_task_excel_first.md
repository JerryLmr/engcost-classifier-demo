# Codex 任务补充：先用 Python 脚本跑通 Excel 三级分类，再改前端

## 当前目标

先不要优先改前端。当前第一目标是：

```text
输入 Excel 第一列：工程名称
其中“工程名称”可以由用户提前把项目名称 + 工程概况拼接得到
↓
后端 Python 脚本批量分类
↓
输出新的 Excel
↓
确认 Ollama 恢复后，规则 + LLM 兜底至少能跑出 level1 / level2 / level3
```

前端改造放到后面。

## 输入 Excel 约定

沿用原来的输入方式：

```text
第一列：工程名称
```

用户可能会把名称和概况提前拼起来，例如：

```text
屋面维修工程15幢。工程概况：屋面、墙面渗水维修
```

脚本只需要读取第一列，不要强制要求新增“工程概况”列。

## 输出 Excel 表头

请在原始数据后追加以下列：

```text
一级分类
二级分类
三级分类
分类方式
置信度
匹配类型
是否建议复核
候选目录ID
候选目录
分类依据
```

字段含义：

- `一级分类`：catalog item.level1
- `二级分类`：catalog item.level2
- `三级分类`：catalog item.level3
- `分类方式`：规则优先 / LLM兜底 / 默认兜底
- `置信度`：高 / 中 / 低
- `匹配类型`：
  - single
  - cross_domain
  - same_domain_multi_item
  - low_confidence
  - llm_fallback
  - fallback
- `是否建议复核`：是 / 否
- `候选目录ID`：candidate_ids 用 ` | ` 拼接
- `候选目录`：候选目录文本，例如 `001 屋面工程 > 平屋面 > 屋面防水...`
- `分类依据`：命中词、LLM原因或兜底原因

## 后端结果结构

`ClassifyResponse` 建议返回：

```json
{
  "project_name": "...",
  "level1": "...",
  "level2": "...",
  "level3": "...",
  "method": "规则优先",
  "confidence": "高",
  "match_type": "single",
  "needs_review": false,
  "candidate_ids": ["001"],
  "reason": "命中对象词：屋面；命中动作词：维修"
}
```

## 先实现 CLI 脚本

请优先实现或改造：

```text
scripts/batch_classify_excel.py
```

要求：

```bash
cd ~/githubRepository/engcost-classifier-demo
source backend/.venv/bin/activate
python scripts/batch_classify_excel.py excel_inputs -o excel_outputs_new --overwrite
```

或者支持单文件：

```bash
python scripts/batch_classify_excel.py input.xlsx -o output.xlsx --overwrite
```

脚本必须直接调用后端分类函数，而不是请求前端。

## 分类链路

分类函数优先基于 `backend/config/catalog.json`：

1. normalize 输入文本
2. 遍历 catalog.items
3. 对每个 item 打分
4. 得到 top candidates
5. 如果 top score 足够高，返回规则结果
6. 如果多个高分候选跨一级目录，`match_type=cross_domain`, `needs_review=true`
7. 如果多个高分候选同一级但不同二/三级，`match_type=same_domain_multi_item`, `needs_review=true`
8. 如果低置信，调用 Ollama qwen3:8b 兜底
9. LLM 只能返回已有 catalog id

## 规则低置信与 LLM

当 score 较低或候选很接近时，可以调用 LLM。

LLM prompt 必须要求：

- 只能从 catalog id 中选；
- 不允许创造目录；
- 返回 JSON；
- 如果是复合工程或混合内容，返回主目录 id，并在 reason 说明候选；
- 后端必须校验 LLM 返回的 id 是否存在于 catalog。

## 暂时不要做的事

本轮不要做：

- PDF解析
- 工程量清单抽取
- 单价比对
- 历史价格区间
- 数据库
- 复杂前端分析页重构

本轮验收标准：

```text
1. catalog.json 被正确加载；
2. 单条分类函数返回 level1 / level2 / level3；
3. scripts/batch_classify_excel.py 能读取第一列工程名称并输出结果 Excel；
4. 输出 Excel 有上述新表头；
5. 至少能用 Ollama qwen3:8b 跑低置信兜底；
6. 旧目录、旧 Python 规则、RULE_SOURCE 双轨不再参与分类。
```
