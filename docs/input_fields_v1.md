# 输入字段清单 V1

用途：
- 约束后端 `/api/audit` 输入 schema
- 指导后续前端补件表单设计
- 作为 OCR / 文档解析的抽取目标
- 支撑完整审计链的分项审计输出

## 一、基础输入字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| project_name | string | 是 | 项目名称 |
| project_desc | string | 否 | 补充描述 |

## 二、对象识别字段（系统生成或可回填）

| 字段 | 类型 | 说明 |
|---|---|---|
| matched_object_ids | array[string] | 命中的对象 ID |
| normalized_tags | array[string] | 命中的语义标签 |
| mapping_confidence | string | high / medium / low |
| gray_case_type | string/null | strong / weak / null |
| split_projects | array[string] | 拆分后的子工程列表 |
| catalog_domains | array[string] | 命中的对象域 |

## 三、结构化输入分组

后端现已支持在保留原顶层兼容字段的同时，按分组传入结构化事实：

- `scope_facts`
- `process_facts`
- `document_facts`
- `timeline_facts`
- `amount_facts`
- `emergency_facts`
- `gray_case_facts`
- `document_parse_context`

说明：
- 顶层旧字段继续兼容
- 若同时传顶层字段和结构化分组，默认以顶层显式值优先

## 四、使用范围审计字段（scope_facts）

| 字段 | 类型 | 说明 |
|---|---|---|
| is_common_part | boolean/null | 是否共用部位 |
| is_common_facility | boolean/null | 是否共用设施设备 |
| is_private_part | boolean/null | 是否专有部分 |
| is_property_service_scope | boolean/null | 是否属于物业服务合同维保范围 |
| is_public_window | boolean/null | 是否公共窗户 |
| is_original_standard_restoration | boolean/null | 是否恢复原标准 |
| is_function_restoration | boolean/null | 是否恢复原功能 |
| is_upgrade | boolean/null | 是否属于升级 |
| is_relocation | boolean/null | 是否属于移位 |
| is_function_improvement | boolean/null | 是否属于功能改善 |
| is_capacity_expansion | boolean/null | 是否属于扩容 |

## 五、流程合规审计字段（process_facts）

| 字段 | 类型 | 说明 |
|---|---|---|
| has_vote | boolean/null | 是否有表决材料 |
| vote_passed | boolean/null | 表决是否通过 |
| has_announcement | boolean/null | 是否有公示材料 |
| announcement_completed | boolean/null | 公示是否完成 |
| has_budget_review | boolean/null | 是否有审价材料 |
| budget_review_required | boolean/null | 是否达到审价要求 |
| has_contract | boolean/null | 是否有施工合同 |
| procurement_method | string/null | 采购方式 |
| selected_vendor_count | number/null | 参与比选/采购的单位数量 |

## 六、资料完整性审计字段（document_facts）

| 字段 | 类型 | 说明 |
|---|---|---|
| has_site_photos | boolean/null | 是否有现场照片 |
| has_rectification_notice | boolean/null | 是否有整改通知/鉴定报告 |
| has_damage_assessment | boolean/null | 是否有损坏评估/检测结论 |
| has_construction_plan | boolean/null | 是否有施工方案 |
| has_completion_report | boolean/null | 是否有完工报告 |
| has_acceptance_record | boolean/null | 是否有验收记录 |
| has_settlement_report | boolean/null | 是否有结算材料 |
| has_invoice | boolean/null | 是否有发票 |
| has_payment_proof | boolean/null | 是否有付款凭证 |
| has_owner_signature_sheet | boolean/null | 是否有业主签字页 |
| has_emergency_proof | boolean/null | 是否有应急维修证明材料 |

## 七、时序合规审计字段（timeline_facts）

| 字段 | 类型 | 说明 |
|---|---|---|
| application_date | string/null | 申请日期 |
| vote_date | string/null | 表决日期 |
| announcement_date | string/null | 公示日期 |
| budget_review_date | string/null | 审价日期 |
| contract_sign_date | string/null | 合同签订日期 |
| construction_start_date | string/null | 开工日期 |
| construction_end_date | string/null | 完工日期 |
| acceptance_date | string/null | 验收日期 |
| invoice_date | string/null | 发票日期 |
| payment_date | string/null | 付款日期 |
| emergency_report_date | string/null | 应急备案/上报日期 |

## 八、金额合理性审计字段（amount_facts）

| 字段 | 类型 | 说明 |
|---|---|---|
| amount | number/null | 当前项目金额 |
| budget_amount | number/null | 预算金额 |
| approved_amount | number/null | 审定金额 |
| settlement_amount | number/null | 结算金额 |
| invoice_amount | number/null | 发票金额 |
| quoted_vendor_count | number/null | 报价单位数量 |
| unit_price_reference_available | boolean/null | 是否有单价参考依据 |
| amount_deviation_ratio | number/null | 偏差比例 |

## 九、应急维修审计字段（emergency_facts）

| 字段 | 类型 | 说明 |
|---|---|---|
| is_emergency | boolean/null | 是否紧急维修 |
| emergency_reason | string/null | 紧急原因 |
| emergency_hazard_type | string/null | 险情类型 |
| temporary_measure_taken | boolean/null | 是否已采取临时处置 |
| post_emergency_vote_required | boolean/null | 应急后是否要求补表决 |
| post_emergency_vote_completed | boolean/null | 应急后补表决是否完成 |

## 十、灰区与现场事实字段（gray_case_facts）

| 字段 | 类型 | 说明 |
|---|---|---|
| repair_scope_description | string/null | 维修部位说明 |
| damage_description | string/null | 损坏情况说明 |
| repair_extent | string/null | 维修范围（单点/多点/大面积） |
| location_detail | string/null | 位置明细（楼道/门厅/单元等） |
| damage_count | number/null | 损坏数量 |
| gray_case_evidence_complete | boolean/null | 灰区证据是否完整 |

## 十一、文档解析预留字段（document_parse_context）

| 字段 | 类型 | 说明 |
|---|---|---|
| source_documents | array | 原始文档列表（仅元信息） |
| extracted_document_fields | object | 文档抽取得到的结构化字段 |
| field_confidence_map | object | 字段抽取置信度 |

## 十二、OCR / 文档解析目标字段

按文档类型建议优先抽取：

- 表决材料：`has_vote`、`vote_date`、`vote_passed`、`owner_count_approved`、`owner_area_approved`
- 公示材料：`has_announcement`、`announcement_date`、`announcement_duration_days`、`announcement_completed`
- 审价/预算材料：`has_budget_review`、`budget_review_date`、`budget_amount`、`approved_amount`、`review_agency_name`
- 合同材料：`has_contract`、`contract_sign_date`、`contract_amount`、`contract_vendor_name`
- 完工/验收材料：`has_completion_report`、`has_acceptance_record`、`construction_start_date`、`construction_end_date`、`acceptance_date`
- 发票/结算/支付材料：`has_invoice`、`invoice_date`、`invoice_amount`、`has_settlement_report`、`settlement_amount`、`has_payment_proof`、`payment_date`
- 应急证明材料：`is_emergency`、`emergency_reason`、`emergency_hazard_type`、`emergency_report_date`、`temporary_measure_taken`、`has_emergency_proof`
- 灰区现场证据：`has_site_photos`、`has_damage_assessment`、`damage_description`、`repair_extent`、`location_detail`、`damage_count`、`is_property_service_scope`
