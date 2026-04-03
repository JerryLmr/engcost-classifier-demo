# 高频对象映射表 V1.1

用途：
- 作为 rule_mapping.json 的人工底稿
- 先稳定高频对象，不追求一次覆盖全量目录

| 高频类别 | 常见原始表述 | 推荐标准对象 | 标签 | 默认路径 | 默认结果 | 置信度 | 说明 |
|---|---|---|---|---|---|---|---|
| 电梯维修 | 曳引机更换、主机维修、缓冲器维修 | 电梯维修对象 | repairable_object, shared_facility | normal_flow | 继续审计 | high | 不直接判合规，要继续走范围/流程判断 |
| 电梯检测 | 年检、制动试验、125%制动试验 | 电梯检测事项 | inspection_testing | exclusion_flow | non_compliant | high | 检测检查类排除 |
| 电梯整改类加装 | 加装急停开关、加装限位装置 | 电梯新增/加装事项 | new_construction | exclusion_flow | non_compliant | high | 有加装即归新增 |
| 外墙维修 | 外墙渗漏、空鼓、脱落、开裂 | 外墙维修对象 | repairable_object, shared_part | normal_flow | 继续审计 | high | 属于正向对象 |
| 外墙美化/清洗 | 清洗、粉刷、美化翻新 | 外墙表面处理 | gray_case | manual_review_flow | manual_review | medium | 维修与美化边界不稳 |
| 屋面维修 | 屋面防水、屋顶渗漏、屋面翻修 | 屋面维修对象 | repairable_object, shared_part | normal_flow | 继续审计 | high | 正向对象 |
| 给排水维修 | 水泵故障、排水管爆裂、污水管堵塞 | 给排水设施维修对象 | repairable_object, shared_facility, emergency_scope | emergency_flow/normal_flow | 继续审计 | high | 是否紧急由字段决定 |
| 消防维修 | 报警系统维修、喷淋维修、消火栓维修 | 消防设施维修对象 | repairable_object, shared_facility, emergency_scope | emergency_flow/normal_flow | 继续审计 | high | 整改通知书类更明确 |
| 灭火器充装/换粉 | 充装二氧化碳、换粉 | 消防日常维保事项 | daily_service_exclusion | exclusion_flow | non_compliant | high | 日常维保 |
| 绿化养护 | 树木修剪、回缩、补种、移除 | 绿化养护事项 | daily_greening, daily_service_exclusion | exclusion_flow | non_compliant | high | 最稳定排除类 |
| 绿化翻新 | 景观提升、整体翻新、公共景观绿化 | 公共绿化改造事项 | gray_case | manual_review_flow | manual_review | medium | 强灰区 |
| 垃圾清运 | 大件垃圾清运、临时垃圾清运 | 清洁卫生事项 | cleaning_sanitation, daily_service_exclusion | exclusion_flow | non_compliant | high | 明确排除 |
| 垃圾桶更换 | 垃圾桶更换、干湿垃圾桶 | 清洁卫生事项 | cleaning_sanitation, daily_service_exclusion | exclusion_flow | non_compliant | high | 明确排除 |
| 摄像头/监控新增 | 增设摄像头、监控系统工程 | 新增安防设施 | new_construction | exclusion_flow | non_compliant | high | 新增类 |
| 弱电维修 | 门禁维修、电子防盗门维修 | 弱电维修对象 | repairable_object, shared_facility | normal_flow | 继续审计 | medium | 修复类先正向进入 |
| 弱电优化/调整 | 位置调整、线路迁改、优化升级 | 弱电新增/调整事项 | gray_case | manual_review_flow | manual_review | medium | 强灰区 |
| 楼道窗户玻璃维修 | 楼道窗玻璃、公共窗户玻璃 | 共用部位玻璃事项 | gray_case | gray_case_review_flow | need_supplement | medium | 弱灰区，需要补证 |
| 零星维修 | 外墙零星维修、电梯零星修理 | 零星维修事项 | gray_case | manual_review_flow | manual_review | medium | 强灰区 |