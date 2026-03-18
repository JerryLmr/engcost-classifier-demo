from typing import Dict, List, Optional, TypedDict


class BoundaryDecision(TypedDict, total=False):
    level1: str
    allowed_level2: List[str]
    reason: str


class BoundaryRule(TypedDict, total=False):
    level1: str
    reason: str
    any_keywords: List[str]
    all_keywords: List[str]
    none_keywords: List[str]
    allowed_level2: List[str]


BOUNDARY_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "防水工程": {
        "definition": "以防渗、防漏、防水层修复为主要目的的工程",
        "strong_keywords": ["防水", "渗漏", "漏水", "防水层", "屋面", "屋顶", "地下室"],
        "weak_keywords": ["维修", "修复"],
        "conflict_keywords": ["外墙", "外立面"],
    },
    "外立面修缮": {
        "definition": "以外墙或立面表层修复、翻新、美观恢复为主要目的的工程",
        "strong_keywords": ["粉刷", "空鼓", "脱落", "裂缝", "翻新", "外立面"],
        "weak_keywords": ["维修", "修补"],
        "conflict_keywords": ["渗漏", "防水", "防水层"],
    },
    "消防": {
        "definition": "围绕消火栓、喷淋、报警、灭火器、防火门等消防设施的维修、更换和改造",
        "strong_keywords": ["消防", "消火栓", "喷淋", "报警", "灭火器", "防火门"],
        "weak_keywords": ["设备", "系统", "维修", "更换", "改造", "更新"],
    },
    "电梯": {
        "definition": "围绕电梯整梯或核心部件的维修、更换、改造升级",
        "strong_keywords": ["电梯", "扶梯", "轿厢", "层门", "主机", "钢丝绳", "抱闸"],
        "weak_keywords": ["维修", "更换", "改造", "更新", "升级"],
    },
}


BOUNDARY_RULES: List[BoundaryRule] = [
    {
        "level1": "防水工程",
        "reason": "外墙或屋面场景中出现渗漏/防水词，优先归入防水工程",
        "all_keywords": ["外墙"],
        "any_keywords": ["渗漏", "漏水", "渗水", "防水", "防水层"],
        "allowed_level2": ["外墙防水"],
    },
    {
        "level1": "防水工程",
        "reason": "外立面场景中出现渗漏/防水词，优先归入防水工程",
        "all_keywords": ["外立面"],
        "any_keywords": ["渗漏", "漏水", "渗水", "防水", "防水层"],
        "allowed_level2": ["外墙防水"],
    },
    {
        "level1": "防水工程",
        "reason": "地下室防水属于防水工程的明确场景",
        "any_keywords": ["地下室"],
        "allowed_level2": ["地下室防水"],
    },
    {
        "level1": "防水工程",
        "reason": "屋面或屋顶的防水/渗漏治理优先归入防水工程",
        "any_keywords": ["屋面", "屋顶"],
        "allowed_level2": ["屋面防水维修"],
    },
    {
        "level1": "外立面修缮",
        "reason": "外墙表层修复、粉刷、空鼓和脱落优先归入外立面修缮",
        "all_keywords": ["外墙"],
        "any_keywords": ["粉刷", "空鼓", "脱落", "裂缝", "翻新", "修补", "涂料"],
        "none_keywords": ["渗漏", "漏水", "渗水", "防水", "防水层"],
    },
    {
        "level1": "外立面修缮",
        "reason": "外立面表层翻新或修补优先归入外立面修缮",
        "any_keywords": ["外立面"],
        "none_keywords": ["渗漏", "漏水", "渗水", "防水", "防水层"],
    },
    {
        "level1": "消防",
        "reason": "消防对象词出现时优先限定在消防分类内",
        "any_keywords": ["消防", "消火栓", "消防栓", "喷淋", "报警", "灭火器", "防火门"],
        "allowed_level2": ["消火栓维修", "消防管网维修", "消防系统改造", "消防设备更换"],
    },
    {
        "level1": "电梯",
        "reason": "电梯和电梯部件场景优先限定在电梯分类内",
        "any_keywords": ["电梯", "扶梯", "轿厢", "层门", "主机", "钢丝绳", "抱闸"],
        "allowed_level2": ["电梯维修", "电梯更换", "电梯改造升级", "电梯部件更换"],
    },
    {
        "level1": "监控",
        "reason": "监控系统和监控设备场景优先限定在监控分类内",
        "any_keywords": ["监控", "摄像头", "球机", "录像", "存储"],
        "allowed_level2": ["监控设备更换", "监控系统升级", "摄像头安装维修", "视频存储系统改造"],
    },
]


def find_boundary_decision(text: str) -> Optional[BoundaryDecision]:
    for rule in BOUNDARY_RULES:
        any_keywords = rule.get("any_keywords", [])
        all_keywords = rule.get("all_keywords", [])
        none_keywords = rule.get("none_keywords", [])

        if any_keywords and not any(keyword in text for keyword in any_keywords):
            continue
        if all_keywords and not all(keyword in text for keyword in all_keywords):
            continue
        if none_keywords and any(keyword in text for keyword in none_keywords):
            continue

        decision: BoundaryDecision = {
            "level1": rule["level1"],
            "reason": rule["reason"],
        }
        if "allowed_level2" in rule:
            decision["allowed_level2"] = list(rule["allowed_level2"])
        return decision

    return None
