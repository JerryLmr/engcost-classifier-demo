# 审计语义标签说明表 V1

用途：
- 统一分类口径
- 约束 rule_mapping.json
- 让规则引擎只吃标签，不直接吃原始文本
- 给前后端、规则和解释层一个共同语言

## 标签总览

| 标签 | 分类性质 | 是否自动判定 | 说明 |
|---|---|---|---|
| repairable_object | 正向 | 是 | 可进入维修资金审计范围的维修对象 |
| shared_part | 属性 | 否 | 共用部位 |
| shared_facility | 属性 | 否 | 共用设施设备 |
| daily_service_exclusion | 排除 | 是 | 属于物业日常服务 |
| daily_greening | 排除（细分） | 是 | 绿化养护类 |
| cleaning_sanitation | 排除（细分） | 是 | 清洁卫生类 |
| inspection_testing | 排除 | 是 | 检测/年检/试验类 |
| new_construction | 排除 | 是 | 新增建设类 |
| emergency_scope | 特殊通道 | 否 | 紧急维修范围 |
| gray_case | 灰区 | 否 | 需要人工复核或补证判断 |
| multi_project | 输入问题 | 是 | 输入中包含多个不同工程 |
| unknown | 输入问题 | 是 | 无法识别对象 |

## 标签定义

### repairable_object
属于共用部位或共用设施设备的维修、更新、改造对象。
注意：这个标签不等于最终合规，只表示可进入维修资金审计流程。

典型对象：
- 电梯曳引机更换
- 外墙渗漏维修
- 屋面防水翻修
- 给排水维修
- 消防系统维修

### shared_part
共用部位属性标签。

典型对象：
- 外墙
- 屋面
- 楼道
- 公共门窗

### shared_facility
共用设施设备属性标签。

典型对象：
- 电梯
- 消防系统
- 水泵
- 给排水系统
- 弱电系统

### daily_service_exclusion
属于物业日常管理、维护、养护或运行维保范围，应排除。

### daily_greening
绿化日常养护行为，应直接排除。

典型对象：
- 树木修剪
- 回缩修剪
- 补种
- 日常绿化维护

### cleaning_sanitation
清洁卫生与垃圾处理相关行为，应直接排除。

典型对象：
- 垃圾清运
- 大件垃圾清运
- 垃圾桶更换
- 公共区域清洁

### inspection_testing
检测、年检、试验类行为，应直接排除。

典型对象：
- 电梯年检
- 电梯制动试验
- 消防检测
- 系统测试

### new_construction
新增、扩建、功能提升类工程，应直接排除。

典型对象：
- 新增摄像头
- 新增车位
- 加装装置
- 系统扩容

### emergency_scope
符合紧急维修范围的对象，进入应急维修分支，而不是普通流程。

### gray_case
边界不清、证据不足或规则无法稳定自动裁决的对象。
注意：gray_case 是标签，不是最终输出结果。

### multi_project
输入中混有多个工程对象，需要拆分或人工复核。

### unknown
无法识别对象，或描述过于模糊。

## 标签使用规则

### 优先级（高到低）
1. 排除类标签（daily_service_exclusion / daily_greening / cleaning_sanitation / inspection_testing / new_construction）
2. gray_case
3. repairable_object
4. 属性标签（shared_part / shared_facility）

### 固定映射关系
- gray_case -> 不直接输出 gray_case
- gray_case -> 根据灰区强弱进入：
  - strong -> manual_review_flow -> manual_review
  - weak -> gray_case_review_flow -> need_supplement / manual_review / 转 normal_flow

### 结果层允许的值
- compliant
- non_compliant
- need_supplement
- manual_review