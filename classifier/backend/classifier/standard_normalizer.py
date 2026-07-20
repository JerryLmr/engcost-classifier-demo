import re
import unicodedata
from dataclasses import dataclass


ACTION_HINT_TERMS = (
    "维修",
    "修缮",
    "修补",
    "更换",
    "更新",
    "改造",
    "新增",
    "扩建",
    "升级",
    "粉刷",
    "防水",
    "翻新",
    "整修",
    "安装",
    "加装",
    "铺设",
)

REVIEW_HINT_TERMS = (
    "咨询",
    "咨询服务",
    "设计",
    "设计服务",
    "监理",
    "检测",
    "审计",
    "维保",
    "维保服务",
    "招标代理",
    "综合品质改造",
    "品质改造",
    "土建维修",
)

NOISE_PATTERNS = (
    r"\b20\d{2}年(?:第[一二三四1-4]季度)?\b",
    r"\b\d{4}年(?:第[一二三四1-4]季度)?\b",
    r"\b\d+(?:#|号楼|幢|栋)\b",
)

NOISE_TERMS = (
    "工程",
    "项目",
    "施工",
    "合同",
    "承包合同",
    "采购合同",
    "小区",
    "大厦",
    "花园",
    "公寓",
)


@dataclass(frozen=True)
class NormalizedProjectText:
    original_text: str
    normalized_text: str
    retrieval_text: str
    action_hints: tuple[str, ...]
    review_hints: tuple[str, ...]


def _compact_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"\s+", "", normalized)


def _hits(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for term in terms:
        if term and term in text and term not in seen:
            seen.append(term)
    return tuple(seen)


def normalize_project_text(project_name: str) -> NormalizedProjectText:
    original = str(project_name or "").strip()
    normalized = _compact_text(original)
    retrieval = normalized
    for pattern in NOISE_PATTERNS:
        retrieval = re.sub(pattern, "", retrieval)
    for term in NOISE_TERMS:
        retrieval = retrieval.replace(term, "")
    return NormalizedProjectText(
        original_text=original,
        normalized_text=normalized,
        retrieval_text=retrieval or normalized,
        action_hints=_hits(normalized, ACTION_HINT_TERMS),
        review_hints=_hits(normalized, REVIEW_HINT_TERMS),
    )
