# Phase 2 设计稿：`timeline_audit` / `amount_audit` / `emergency_audit`

目标：在不改 `/api/audit` 路径、不改输出骨架的前提下，补实三项分项审计，并将“强异常强处理、证据缺口弱结论”落成统一规则语义。

## 核心判定原则

- 强异常（矛盾、逆序、超阈值异常）：
  - 进入 `manual_review`
  - 作为顶层强信号参与汇总
- 明确违规/明确排除：
  - 进入 `non_compliant`
  - 作为顶层最高优先级强信号
- 字段缺失、材料不足、证据不充分：
  - 优先写入 `missing_items` + `reasons`
  - 规则标记为 `top_level_effect=advisory`
  - 不作为顶层强阻断信号

## 三项分项规则（最小可执行集）

### 1) `timeline_audit`（时序合规审计）

- 强异常：
  - `vote_date < application_date` -> `manual_review`
  - `announcement_date < vote_date` -> `manual_review`
- 弱结论：
  - 关键节点日期缺失 -> `need_supplement`
  - 文案口径：“时序暂无法充分核验，需补充关键时间节点”
  - 标记 `top_level_effect=advisory`

边界：
- 只处理时序关系与节点完整性
- 不处理流程是否必须（归 `process_audit`）
- 不处理材料是否齐全（归 `document_completeness_audit`）

### 2) `amount_audit`（金额合理性审计）

- 强异常：
  - `amount_deviation_ratio >= 阈值` -> `manual_review`
- 弱结论：
  - 金额/预算/审批依据不足 -> `need_supplement`
  - 文案口径：“金额依据不足，建议补充预算、审批及金额依据材料”
  - 标记 `top_level_effect=advisory`

边界：
- 只处理金额异常与金额依据
- 票据链缺件继续归 `document_completeness_audit`
- 流程门槛（如应审价）由 `process_audit` 承担主语义

### 3) `emergency_audit`（应急维修审计）

- 强异常：
  - `is_emergency=true` 且 `post_emergency_vote_required=true` 且 `post_emergency_vote_completed=false`
  - -> `manual_review`
- 弱结论：
  - 应急证明不足 -> `need_supplement`
  - 文案口径：“紧急维修证据材料不足，暂无法充分核验”
  - 标记 `top_level_effect=advisory`

边界：
- 应急后补流程是否完成由 `emergency_audit` 主判
- 证明材料缺口可同步体现在 `document_completeness_audit`

## 顶层汇总策略（替代“先命中先返回”）

规则触发后按信号分层汇总：

1. `direct` 强信号（默认）：
   - 优先级：`non_compliant > manual_review > need_supplement > compliant`
2. `advisory` 弱信号：
   - 不参与强阻断排序
   - 用于补充顶层 `missing_items/reasons`
3. 仅当无强信号时：
   - 使用 `advisory` 或 `OUTPUT_BUILD` 兜底结果

`manual_review_required`：
- `overall_result=manual_review` 时为 `true`
- 命中 `advisory` 弱信号时也可置 `true`，提示仍需人工关注

## 验收样例（设计口径）

1. `{"project_name":"屋面防水维修","application_date":"2026-01-10","vote_date":"2026-01-08"}`
   - 预期：`timeline_audit=manual_review`，顶层可 `manual_review`

2. `{"project_name":"电梯维修","amount":120000,"has_budget_review":false}`
   - 预期：金额依据不足为弱结论，进入缺口提示，不默认强阻断顶层

3. `{"project_name":"排水管爆裂维修","is_emergency":true,"has_emergency_proof":false}`
   - 预期：应急证明不足为弱结论，提示缺口，不直接按强异常处理

4. `{"project_name":"排水管爆裂维修","is_emergency":true,"post_emergency_vote_required":true,"post_emergency_vote_completed":false}`
   - 预期：`emergency_audit=manual_review`，顶层可 `manual_review`
