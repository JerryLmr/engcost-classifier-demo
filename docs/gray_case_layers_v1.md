# 灰区对象分层表 V1

目的：
不是所有 gray_case 都直接人工复核。
需要分成：
- 强灰区：直接人工复核
- 弱灰区：先补材料，再决定是否转正常流程

## 一、强灰区（strong）

| 对象类型 | 常见表述 | 默认路径 | 默认结果 | 原因 |
|---|---|---|---|---|
| 绿化翻新/景观提升 | 绿化翻新、景观绿化提升 | manual_review_flow | manual_review | 日常养护与整体更新边界不稳 |
| 外墙美化/立面翻新 | 外墙粉刷、立面翻新、清洗美化 | manual_review_flow | manual_review | 维修与美化提升边界不稳 |
| 弱电优化/系统调整 | 位置调整、线路优化、迁改提升 | manual_review_flow | manual_review | 维修、优化、新增边界不清 |
| 零星维修 | 零星修理、零星维修 | manual_review_flow | manual_review | 业务口径倾向排除，但公开依据不够稳 |

## 二、弱灰区（weak）

| 对象类型 | 常见表述 | 默认路径 | 缺材料时 | 补证后 |
|---|---|---|---|---|
| 楼道窗户玻璃维修 | 楼道窗玻璃、公共窗户玻璃 | gray_case_review_flow | need_supplement | 转 normal_flow 或 manual_review |
| 共用门窗维修 | 楼道门窗、公共门窗、门厅玻璃门 | gray_case_review_flow | need_supplement | 转 normal_flow 或 manual_review |
| 部分弱电修复 | 门禁修复、对讲门修理 | gray_case_review_flow | need_supplement | 转 normal_flow 或 manual_review |
| 部分附属设施修复 | 园椅、栏杆、小设施损坏修复 | gray_case_review_flow | need_supplement | 转 normal_flow 或 manual_review |

## 三、统一规则

- gray_case 是标签
- strong gray_case -> manual_review_flow -> manual_review
- weak gray_case -> gray_case_review_flow
  - 材料不足 -> need_supplement
  - 材料足够且仍边界不清 -> manual_review
  - 材料足够且边界清楚 -> 转 normal_flow