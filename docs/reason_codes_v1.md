# reason_code 标准表 V1

用途：
- 统一规则输出原因
- 统一前端展示和报告说明
- 避免同一问题多种表述

## A. 范围类（Scope）

| code | 含义 |
|---|---|
| IN_SCOPE_COMMON_PART | 属于共用部位维修 |
| IN_SCOPE_COMMON_FACILITY | 属于共用设施设备维修 |
| OUTSIDE_SCOPE_PRIVATE_PART | 属于专有部分 |
| OUTSIDE_SCOPE_UNKNOWN_OBJECT | 无法确认维修对象 |

## B. 排除类（Exclusion）

| code | 含义 |
|---|---|
| DAILY_SERVICE | 属于物业日常服务 |
| GREENING_MAINTENANCE | 属于绿化养护 |
| CLEANING_SANITATION | 属于清洁卫生服务 |
| INSPECTION_TESTING | 属于检测/年检/试验 |
| NEW_CONSTRUCTION | 属于新增/加装/扩建 |
| WITHIN_WARRANTY | 属于质保期内 |
| PROPERTY_SERVICE_SCOPE | 属于物业服务合同维保范围 |

## C. 灰区类（Gray Case）

| code | 含义 |
|---|---|
| GRAY_CASE_GENERAL | 一般灰区 |
| GRAY_CASE_OBJECT_BOUNDARY | 对象边界不清 |
| GRAY_CASE_CONTRACT_BOUNDARY | 物业维保边界不清 |
| GRAY_CASE_UPGRADE_VS_REPAIR | 修复与升级边界不清 |
| GRAY_CASE_MINOR_REPAIR | 零星维修 |

## D. 流程类（Process）

| code | 含义 |
|---|---|
| MISSING_VOTE | 缺少表决信息 |
| MISSING_ANNOUNCEMENT | 缺少公示信息 |
| MISSING_BUDGET_REVIEW | 缺少审价材料 |
| MISSING_CONTRACT | 缺少施工合同 |
| MISSING_INVOICE | 缺少发票 |
| MISSING_COMPLETION_REPORT | 缺少完工/验收材料 |
| MISSING_EMERGENCY_DOC | 缺少紧急维修证明材料 |

## E. 输入类（Input）

| code | 含义 |
|---|---|
| OUTSIDE_CATALOG | 未命中对象目录 |
| INSUFFICIENT_INFO | 输入信息不足 |
| MISSING_GRAY_CASE_EVIDENCE | 灰区缺关键补充材料 |
| MULTI_PROJECT | 输入中包含多个不同工程 |
| CROSS_DOMAIN_PROJECT | 跨域复合工程 |

## 输出原则

- 一个结果可以对应多个 reason_code
- gray_case 是标签，不是 reason_code
- gray_case 最终会转成：
  - need_supplement（如缺灰区证据）
  - manual_review（如强灰区或补证后仍不清）