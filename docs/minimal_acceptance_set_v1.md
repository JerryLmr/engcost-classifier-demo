# 最小样本验收集 V1

用途：
- 作为系统最小验收基线
- 每次改规则后必须跑这一组样本
- 保证系统稳定性，不因调整一条规则影响全局

## 样本结构

| 字段 | 含义 |
|---|---|
| id | 测试 ID |
| project_name | 输入项目名 |
| expected_tags | 预期标签 |
| expected_flow | 预期路径 |
| expected_result | 预期内部结果 |
| expected_reason_codes | 预期原因码 |
| notes | 说明 |

## 样本（20条）

### 排除类
- T001 小区树木修剪 -> daily_greening -> exclusion_flow -> non_compliant -> GREENING_MAINTENANCE
- T002 小区垃圾清运 -> cleaning_sanitation -> exclusion_flow -> non_compliant -> CLEANING_SANITATION
- T003 电梯125%制动试验 -> inspection_testing -> exclusion_flow -> non_compliant -> INSPECTION_TESTING
- T004 新增摄像头监控系统工程 -> new_construction -> exclusion_flow -> non_compliant -> NEW_CONSTRUCTION
- T005 灭火器换粉 -> daily_service_exclusion -> exclusion_flow -> non_compliant -> DAILY_SERVICE
- T006 垃圾桶更换 -> cleaning_sanitation -> exclusion_flow -> non_compliant -> CLEANING_SANITATION

### 正向类
- T007 电梯曳引机更换 -> repairable_object -> normal_flow -> need_supplement -> MISSING_BUDGET_REVIEW
- T008 屋面防水维修 -> repairable_object -> normal_flow -> need_supplement -> MISSING_VOTE
- T009 外墙渗漏维修 -> repairable_object -> normal_flow -> need_supplement -> MISSING_ANNOUNCEMENT
- T010 消防喷淋系统维修 -> repairable_object -> normal_flow -> need_supplement -> MISSING_CONTRACT
- T011 排水管爆裂维修 -> repairable_object + emergency_scope -> emergency_flow -> need_supplement -> MISSING_EMERGENCY_DOC

### 灰区类
- T012 楼道窗户玻璃维修 -> gray_case -> gray_case_review_flow -> need_supplement -> MISSING_GRAY_CASE_EVIDENCE
- T013 绿化翻新工程 -> gray_case -> manual_review_flow -> manual_review -> GRAY_CASE_GENERAL
- T014 弱电线路优化调整 -> gray_case -> manual_review_flow -> manual_review -> GRAY_CASE_UPGRADE_VS_REPAIR
- T015 小区外墙粉刷翻新 -> gray_case -> manual_review_flow -> manual_review -> GRAY_CASE_GENERAL
- T016 楼道门窗维修 -> gray_case -> gray_case_review_flow -> need_supplement -> MISSING_GRAY_CASE_EVIDENCE
- T017 外墙零星维修 -> gray_case -> manual_review_flow -> manual_review -> GRAY_CASE_MINOR_REPAIR

### 输入问题类
- T018 华丰苑小区美丽家园工程合同 防盗门油漆，人行道闸、绿化补种、电子门禁 -> multi_project -> input_check_flow -> manual_review -> MULTI_PROJECT
- T019 维修工程 -> unknown -> input_check_flow -> manual_review -> INSUFFICIENT_INFO
- T020 小区设施优化工程 -> unknown -> input_check_flow -> manual_review -> OUTSIDE_CATALOG