# 输入字段清单 V1

用途：
- 约束后端 API
- 指导前端表单
- 作为 LLM/解析层的抽取目标
- 支撑灰区补证

## 一、基础输入字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| project_name | string | 是 | 项目名称 |
| project_desc | string | 否 | 补充描述 |

## 二、对象识别字段（系统生成）

| 字段 | 类型 | 说明 |
|---|---|---|
| matched_object_ids | array[string] | 命中的对象 ID |
| normalized_tags | array[string] | 命中的语义标签 |
| mapping_confidence | string | high / medium / low |
| gray_case_type | string/null | strong / weak / null |

## 三、审计判断字段

| 字段 | 类型 | 说明 |
|---|---|---|
| is_common_part | boolean/null | 是否共用部位 |
| is_common_facility | boolean/null | 是否共用设施设备 |
| is_emergency | boolean/null | 是否紧急维修 |
| is_out_of_warranty | boolean/null | 是否过保修期 |
| amount | number/null | 金额 |
| has_vote | boolean/null | 是否有表决 |
| has_announcement | boolean/null | 是否有公示 |
| has_budget_review | boolean/null | 是否有审价 |
| has_site_photos | boolean/null | 是否有现场照片 |
| has_rectification_notice | boolean/null | 是否有整改通知/鉴定报告 |
| has_construction_plan | boolean/null | 是否有施工方案 |
| has_completion_report | boolean/null | 是否有完工/验收报告 |
| has_invoice | boolean/null | 是否有发票 |

## 四、灰区补充字段

| 字段 | 类型 | 说明 |
|---|---|---|
| repair_scope_description | string/null | 维修部位说明 |
| damage_description | string/null | 损坏情况说明 |
| repair_extent | string/null | 维修范围（单点/多点/大面积） |
| site_photos | array/null | 灰区证据照片 |
| is_property_service_scope | boolean/null | 是否属于物业维保范围 |
| gray_case_evidence_complete | boolean | 灰区证据是否完整 |

## 五、楼道玻璃专用字段

| 字段 | 类型 | 说明 |
|---|---|---|
| is_public_window | boolean/null | 是否公共窗户 |
| location_detail | string/null | 位置明细（楼道/门厅/单元等） |
| damage_count | number/null | 损坏数量 |

## 六、弱电类专用字段

| 字段 | 类型 | 说明 |
|---|---|---|
| is_function_restoration | boolean/null | 是否恢复原功能 |
| is_upgrade | boolean/null | 是否升级 |
| is_relocation | boolean/null | 是否位置调整 |

## 七、输入问题字段

| 字段 | 类型 | 说明 |
|---|---|---|
| split_projects | array[string] | 拆分后的子工程列表 |