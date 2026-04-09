# 高频入口验收基线（Phase 1）

目的：固定高频前置分流的回归标准，后续规则调整必须先过这组基线。

## 分组 A：`direct_reject`

| 输入 | 期望结果 | 关键断言 |
|---|---|---|
| 上海锦绣逸庭园区灭火器充装二氧化碳 | `non_compliant` | `audit_path` 包含 `high_freq_mapping` + `direct_reject` |
| 电梯125%制动试验 | `non_compliant` | 命中检测检查类排除 |
| 垃圾桶更换 | `non_compliant` | 命中清洁卫生排除 |
| 小区树木修剪 | `non_compliant` | 命中绿化养护排除 |
| 新增摄像头安装工程 | `non_compliant` | 命中新增加装排除 |

## 分组 B：灰区分流

| 输入 | 期望结果 | 关键断言 |
|---|---|---|
| 2号楼楼道窗户玻璃维修 | `need_supplement` | `audit_path` 包含 `route_to_need_supplement` |
| 公共景观绿化整体翻新 | `manual_review` | `audit_path` 包含 `route_to_manual_review` |
| 门禁线路迁改优化升级 | `manual_review` | `audit_path` 包含 `route_to_manual_review` |

## 分组 C：`route_to_full_audit`

| 输入 | 期望结果 | 关键断言 |
|---|---|---|
| 3号楼电梯主机维修 | 进入完整审计链 | `audit_path` 包含 `route_to_full_audit` + `mapping`，不含 `direct_reject` |
| 12号楼外墙渗漏维修 | 进入完整审计链 | `audit_path` 包含 `route_to_full_audit` + `mapping` |
| 屋面防水维修 | 进入完整审计链 | `audit_path` 包含 `route_to_full_audit` + `mapping` |
| 消防喷淋维修 | 进入完整审计链 | `audit_path` 包含 `route_to_full_audit` + `mapping` |

## 可解释性口径（当前实现）

- `audit_path` 语义段统一为：
  - `input_normalization`
  - `high_freq_mapping`
  - `high_freq_matched` 或 `high_freq_no_match`
  - 分流动作（如 `direct_reject` / `route_to_full_audit`）
  - 后续审计链路径（如 `mapping`、`normal_flow`）
- 对 `route_to_full_audit` 场景，`reasons` 优先包含高频业务口径说明；规则引擎输出说明保留在后。

