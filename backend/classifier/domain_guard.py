from dataclasses import dataclass

from classifier.standard_normalizer import normalize_project_text


@dataclass(frozen=True)
class GuardDecision:
    forced_catalog_id: str | None = None
    blocked_catalog_ids: tuple[str, ...] = ()
    needs_review: bool = False
    reason: str = ""
    context_hints: tuple[str, ...] = ()


GENERIC_ELEVATOR_TERMS = (
    "电梯",
    "客梯",
    "乘客电梯",
    "住宅电梯",
    "垂直电梯",
    "老旧电梯",
)

SPECIFIC_ELEVATOR_TERMS = (
    "曳引机",
    "制动器",
    "电动机",
    "导向轮",
    "曳引轮",
    "钢丝绳",
    "限速器",
    "限速系统",
    "控制柜",
    "励磁柜",
    "层门",
    "轿门",
    "轿厢门",
    "门机板",
    "导靴",
    "吊门轮",
    "缓冲器",
    "紧急报警",
    "呼叫电话",
    "呼梯",
    "按钮",
    "液压梯",
    "液压泵站",
    "自动扶梯",
    "自动人行道",
    "扶手带",
    "梯级",
    "踏板",
    "梯级链",
    "滚轮",
)

WALL_TERMS = (
    "墙砖",
    "墙面砖",
    "墙面",
    "粉刷",
    "瓷砖",
    "块料面层",
    "面砖",
)

EXTERIOR_WALL_TERMS = (
    "外墙",
    "外立面",
    "外墙面",
    "外墙砖",
    "外墙面砖",
    "女儿墙外侧",
    "幕墙",
    "玻璃幕墙",
)

INTERIOR_PUBLIC_WALL_TERMS = (
    "楼道",
    "门厅",
    "走廊",
    "过道",
    "楼梯间",
    "电梯厅",
    "单元厅",
    "公共走道",
    "公共部位内墙",
)

BASEMENT_WALL_TERMS = (
    "地下室",
    "地库",
    "地下车库",
    "车库",
    "地下空间",
)

WEAK_CURRENT_SYSTEM_TERMS = (
    "弱电",
    "智能化",
    "安防",
    "监控系统",
    "视频监控",
    "门禁系统",
    "对讲系统",
    "停车场系统",
    "道闸系统",
    "周界防范",
    "公共广播",
    "信息发布",
)

WEAK_CURRENT_COMPONENT_TERMS = (
    "摄像机",
    "拾音器",
    "读卡器",
    "电锁",
    "出门按钮",
    "红外对射",
    "电子围栏",
    "扬声器",
    "入口设备",
    "出口设备",
    "道闸",
    "车辆检测",
    "线缆",
    "线管",
    "桥架",
    "支吊架",
    "录像机",
    "矩阵",
    "编解码",
    "交换机",
    "服务器",
    "存储器",
    "软件",
    "监视器",
    "控制柜",
    "UPS",
    "不间断供电",
)

ELEVATOR_DETAIL_REVIEW_TERMS = (
    "钢带",
    "曳引带",
    "控制面板",
    "主板",
    "主控板",
    "电路板",
    "三方通话",
    "五方通话",
    "紧急通话",
    "轿厢对讲",
    "电梯对讲",
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _merge_decisions(decisions: list[GuardDecision]) -> GuardDecision:
    forced_catalog_id = next((decision.forced_catalog_id for decision in decisions if decision.forced_catalog_id), None)
    blocked_ids: list[str] = []
    context_hints: list[str] = []
    reasons: list[str] = []
    needs_review = False

    for decision in decisions:
        needs_review = needs_review or decision.needs_review
        if decision.reason:
            reasons.append(decision.reason)
        for catalog_id in decision.blocked_catalog_ids:
            if catalog_id not in blocked_ids:
                blocked_ids.append(catalog_id)
        for hint in decision.context_hints:
            if hint not in context_hints:
                context_hints.append(hint)

    return GuardDecision(
        forced_catalog_id=forced_catalog_id,
        blocked_catalog_ids=tuple(blocked_ids),
        needs_review=needs_review,
        reason="；".join(reasons),
        context_hints=tuple(context_hints),
    )


def _generic_elevator_guard(text: str) -> GuardDecision:
    has_specific_term = _contains_any(text, SPECIFIC_ELEVATOR_TERMS) or _contains_any(text, ELEVATOR_DETAIL_REVIEW_TERMS)
    if _contains_any(text, GENERIC_ELEVATOR_TERMS) and not has_specific_term:
        return GuardDecision(
            forced_catalog_id="CF-017-00",
            blocked_catalog_ids=("CF-017-13",),
            needs_review=True,
            reason="工程仅明确为电梯类维修/更新/改造，未明确四川表17中的具体部件或子系统，使用内部扩展项",
            context_hints=("普通电梯项目未明确具体部件，优先使用 CF-017-00，不要误选自动扶梯及自动人行道",),
        )
    return GuardDecision()


def _wall_location_guard(text: str) -> GuardDecision:
    hints: list[str] = []
    needs_review = False
    reason = ""
    has_wall = _contains_any(text, WALL_TERMS)
    has_exterior = _contains_any(text, EXTERIOR_WALL_TERMS)
    has_public_interior = _contains_any(text, INTERIOR_PUBLIC_WALL_TERMS)
    has_basement = _contains_any(text, BASEMENT_WALL_TERMS)

    if has_exterior:
        hints.append("外墙/外立面/女儿墙外侧/幕墙应优先 CP-003-01 外墙面面层")
    if has_public_interior and has_wall:
        hints.append("楼道/门厅/走廊/楼梯间/电梯厅墙面块材应优先 CP-004-02")
    if has_basement and has_wall:
        hints.append("地下室/地库/车库墙面应优先 CP-005-01")
    if has_wall and not (has_exterior or has_public_interior or has_basement):
        needs_review = True
        reason = "墙面/墙砖位置不明，需在外墙面、室内公共墙面、地下室墙面之间复核"
        hints.append("墙面/墙砖位置不明，不能直接 OUT，需在外墙面、室内公共墙面、地下室墙面之间复核")

    return GuardDecision(needs_review=needs_review, reason=reason, context_hints=tuple(hints))


def _weak_current_system_guard(text: str) -> GuardDecision:
    if _contains_any(text, WEAK_CURRENT_SYSTEM_TERMS) and not _contains_any(text, WEAK_CURRENT_COMPONENT_TERMS):
        return GuardDecision(
            needs_review=True,
            reason="弱电系统级项目未明确前端设备、传输系统或中央处理单元",
            context_hints=("弱电/监控/安防系统级项目未明确具体组成部分时，不要无复核地只归摄像机等前端设备",),
        )
    return GuardDecision()


def _elevator_detail_review_guard(text: str) -> GuardDecision:
    if "电梯" in text and _contains_any(text, ELEVATOR_DETAIL_REVIEW_TERMS):
        return GuardDecision(
            needs_review=True,
            reason="电梯细部件为近似目录映射，需复核四川表17对应关系",
            context_hints=("电梯钢带、主板、控制面板、三方/五方通话等仅在电梯上下文中近似映射，需复核",),
        )
    return GuardDecision()


def evaluate_domain_guards(project_name: str) -> GuardDecision:
    normalized = normalize_project_text(project_name)
    text = normalized.normalized_text
    return _merge_decisions(
        [
            _generic_elevator_guard(text),
            _wall_location_guard(text),
            _weak_current_system_guard(text),
            _elevator_detail_review_guard(text),
        ]
    )
