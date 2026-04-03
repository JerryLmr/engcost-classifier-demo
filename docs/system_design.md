# 系统设计说明（system_design.md）

## 1. 目标

在现有 `engcost-classifier-demo` 项目上新增“维修资金审计链路”，不重开项目，不推翻现有 FastAPI 和前端骨架，不破坏现有分类接口的基本可用性。

当前阶段目标不是做完整 LLM 审计系统，而是先打通一个**可解释、可维护、可测试**的规则型审计最小闭环：

单条文本输入 → 对象识别 → 标签映射 → 规则裁决 → 统一输出

---

## 2. 总体原则

1. 现有 `taxonomy.json` 不再作为正式分类标准，只作为 legacy 参考。
2. 新系统正式采用四层规则资产：
   - `object_catalog.json`：维修对象目录事实层
   - `rule_mapping.json`：对象到审计语义标签映射层
   - `rule_engine.json`：审计规则裁决层
   - `output_schema.json`：统一输出层
3. 现有 `classify` 接口保留，但仅作为粗识别入口，不再作为最终审计依据。
4. 新增 `audit` 接口，负责完整审计链路。
5. 第一轮不接 LLM，不做复杂前端，不做知识图谱，只先把规则链打通。
6. 输出口径柔性化：
   - `compliant` -> 初步符合
   - `non_compliant` -> 疑似违规
   - `need_supplement` -> 需补充材料
   - `manual_review` -> 建议人工复核

---

## 3. 架构分层

### 3.1 输入层
输入可以先只支持一条项目名称：

- `project_name`
- 可选：`project_desc`

后续可以扩展到 PDF / Excel / 图片，但不属于第一轮范围。

### 3.2 对象目录层（object_catalog）
职责：

- 存放维修对象目录事实
- 不承担公开法规 source
- 不输出最终结论
- 只作为标准对象匹配基础

当前建议字段：

- `id`
- `level_1`
- `level_2`
- `level_3`
- `full_path`
- `status`

### 3.3 标签映射层（rule_mapping）
职责：

- 将对象或原始文本映射为有限审计语义标签
- 决定进入哪条规则路径
- 不直接输出最终结论

当前标签集合：

- `repairable_object`
- `shared_part`
- `shared_facility`
- `daily_service_exclusion`
- `daily_greening`
- `cleaning_sanitation`
- `inspection_testing`
- `new_construction`
- `emergency_scope`
- `gray_case`
- `multi_project`
- `unknown`

### 3.4 规则裁决层（rule_engine）
职责：

- 根据标签和结构化字段裁决
- 输出内部结果：
  - `compliant`
  - `non_compliant`
  - `need_supplement`
  - `manual_review`

支持的路径：

- `input_check_flow`
- `exclusion_flow`
- `manual_review_flow`
- `gray_case_review_flow`
- `emergency_flow`
- `normal_flow`

### 3.5 输出层（output_schema）
职责：

- 统一后端返回结构
- 支撑前端展示、报告、测试

至少包含：

- `project_name`
- `matched_object_ids`
- `normalized_tags`
- `overall_result`
- `display_result`
- `reason_codes`
- `reasons`
- `basis_documents`
- `missing_items`
- `audit_path`
- `manual_review_required`

---

## 4. 灰区机制

### 4.1 基本原则
- `gray_case` 是标签，不是最终输出结果
- `manual_review` 是输出结果，不是标签

### 4.2 灰区分层
- **强灰区**：直接进入 `manual_review_flow`
- **弱灰区**：进入 `gray_case_review_flow`

### 4.3 弱灰区处理逻辑
弱灰区不直接人工复核，而是：

- 证据不足 -> `need_supplement`
- 证据充分且属于物业维保范围 -> `non_compliant`
- 证据充分、不属于物业维保、且属于共用部位 -> 转 `normal_flow`
- 证据充分但仍边界不清 -> `manual_review`

---

## 5. 输入异常与复合工程

### 5.1 复合工程
若输入中出现多个不同对象混在一起，例如：

- 防盗门油漆
- 人行道闸
- 绿化补种
- 电子门禁

则应识别为：

- `multi_project`
- 或 `CROSS_DOMAIN_PROJECT`

处理方式：

- 不强行合并审计
- 输出 `manual_review`
- 提示建议拆分后分别审计

### 5.2 目录外对象
若未命中目录：

- 保留 reason code：`OUTSIDE_CATALOG`
- 输出：`manual_review`

---

## 6. 当前高频业务规则

1. 绿化修剪 / 补种 / 回缩
   - 标签：`daily_greening`
   - 路径：`exclusion_flow`
   - 结果：`non_compliant`

2. 垃圾清运 / 垃圾桶更换
   - 标签：`cleaning_sanitation`
   - 路径：`exclusion_flow`
   - 结果：`non_compliant`

3. 电梯年检 / 制动试验
   - 标签：`inspection_testing`
   - 路径：`exclusion_flow`
   - 结果：`non_compliant`

4. 新增摄像头 / 加装设备
   - 标签：`new_construction`
   - 路径：`exclusion_flow`
   - 结果：`non_compliant`

5. 电梯维修 / 外墙渗漏 / 屋面防水 / 给排水维修 / 消防维修
   - 标签：`repairable_object`
   - 路径：`normal_flow` 或 `emergency_flow`

6. 绿化翻新
   - 标签：`gray_case`
   - 强灰区
   - 结果：`manual_review`

7. 楼道窗户玻璃维修
   - 标签：`gray_case`
   - 弱灰区
   - 路径：`gray_case_review_flow`

8. 零星维修
   - 标签：`gray_case`
   - 强灰区
   - 结果：`manual_review`

---

## 7. 服务层职责

### 7.1 mapping_service.py
职责：

- 读取 `object_catalog.json`
- 根据 `project_name` 做对象匹配
- 输出标准对象候选
- 不做审计判断

输入：

```json
{
  "project_name": "3号楼电梯曳引机维修"
}
```

输出：

```json
{
  "mapped_objects": [
    {
      "id": 313,
      "full_path": "电梯/曳引系统/曳引机",
      "match_score": 0.92,
      "match_method": "keyword"
    }
  ]
}
```

### 7.2 audit_service.py
职责：

- 读取 `rule_mapping.json` 和 `rule_engine.json`
- 将对象映射为标签
- 根据路径和规则裁决
- 输出统一结果

不负责：

- LLM 推理
- 前端渲染
- 文档解析

### 7.3 explanation_service.py
第一轮可以不实现。
后续若接 LLM，仅负责：
- 将规则结果翻译成说明
- 引用公开法规依据
- 生成补正建议

---

## 8. API 设计

### 8.1 保留旧接口
保留原有 `/api/classify`，但只作为粗识别入口。

### 8.2 新增审计接口
新增：

`POST /api/audit`

输入：

```json
{
  "project_name": "楼道窗户玻璃维修"
}
```

输出：

```json
{
  "project_name": "楼道窗户玻璃维修",
  "matched_object_ids": [251],
  "normalized_tags": ["gray_case"],
  "overall_result": "need_supplement",
  "display_result": "需补充材料",
  "reason_codes": ["MISSING_GRAY_CASE_EVIDENCE"],
  "reasons": ["灰区对象缺少关键补充材料，需先补证"],
  "basis_documents": [],
  "missing_items": ["维修部位说明", "现场照片", "责任边界说明"],
  "audit_path": ["INPUT_CHECK", "CATALOG_CHECK", "GRAY_CASE_REVIEW_CHECK"],
  "manual_review_required": false
}
```

---

## 9. 测试要求

第一轮必须新增最小测试文件：

- `tests/test_audit_engine.py`

至少覆盖：

- 绿化修剪
- 垃圾清运
- 电梯制动试验
- 新增摄像头
- 电梯曳引机维修
- 外墙渗漏维修
- 楼道窗户玻璃维修
- 绿化翻新
- 零星维修
- 复合工程输入

---

## 10. 迁移原则

1. 现有项目继续使用，不重开。
2. 原 taxonomy 降级为 legacy，不再作为正式主标准。
3. 新系统正式主标准改为：
   - `object_catalog.json`
   - `rule_mapping.json`
   - `rule_engine.json`
   - `output_schema.json`
4. 第一轮目标只是“打通规则最小闭环”，不是做完整产品。

---

## 11. 当前阶段的完成标准

当以下链路跑通时，视为第一轮完成：

单条文本输入
-> 对象识别
-> 标签映射
-> 规则裁决
-> 输出统一 JSON
-> 最小测试集通过
