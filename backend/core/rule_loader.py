import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TypedDict

from core.config import RULE_CONFIG_DIR, RULE_CONFIG_FALLBACK_DIR, RULE_SOURCE
from core.rule_validator import validate_json_config_dir, validate_ruleset
from data.boundaries import BOUNDARY_RULES
from data.categories import CATEGORY_TREE
from data.rules import DETAILED_LEVEL2_RULES, LEVEL1_RULES, LEVEL2_RULES
from data.structure_rules import DOMAIN_STRONG_KEYWORDS, SAME_DOMAIN_COMPONENTS


class RuleSet(TypedDict):
    category_tree: Dict[str, list[str]]
    category_lines: str
    level1_rules: Dict[str, list[list[Any]]]
    level2_rules: Dict[str, Dict[str, list[list[Any]]]]
    detailed_level2_rules: Dict[str, Dict[str, Dict[str, Any]]]
    boundary_rules: list[Dict[str, Any]]
    domain_strong_keywords: Dict[str, list[str]]
    same_domain_components: Dict[str, Dict[str, list[str]]]


def build_category_lines(category_tree: Mapping[str, list[str]]) -> str:
    return "\n".join(
        f"- {parent}：" + "、".join(children)
        for parent, children in category_tree.items()
    )


def _clone(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return [_clone(item) for item in value]
    return value


def get_python_ruleset() -> RuleSet:
    category_tree = _clone(CATEGORY_TREE)
    ruleset: RuleSet = {
        "category_tree": category_tree,
        "category_lines": build_category_lines(category_tree),
        "level1_rules": _clone(LEVEL1_RULES),
        "level2_rules": _clone(LEVEL2_RULES),
        "detailed_level2_rules": _clone(DETAILED_LEVEL2_RULES),
        "boundary_rules": _clone(BOUNDARY_RULES),
        "domain_strong_keywords": _clone(DOMAIN_STRONG_KEYWORDS),
        "same_domain_components": _clone(SAME_DOMAIN_COMPONENTS),
    }
    validate_ruleset(ruleset, source="python_ruleset")
    return ruleset


def load_json_ruleset(config_dir: Optional[Path] = None) -> RuleSet:
    requested_dir = (config_dir or RULE_CONFIG_DIR).resolve()
    config_candidates = [requested_dir]
    fallback_resolved = RULE_CONFIG_FALLBACK_DIR.resolve()
    if fallback_resolved not in config_candidates:
        config_candidates.append(fallback_resolved)

    config_dir = None
    for candidate in config_candidates:
        try:
            validate_json_config_dir(candidate)
            config_dir = candidate
            break
        except (FileNotFoundError, ValueError):
            continue
    if config_dir is None:
        validate_json_config_dir(requested_dir)
        config_dir = requested_dir

    with (config_dir / "taxonomy.json").open("r", encoding="utf-8") as fp:
        category_tree = json.load(fp)
    with (config_dir / "level1_rules.json").open("r", encoding="utf-8") as fp:
        level1_rules = json.load(fp)
    with (config_dir / "level2_rules.json").open("r", encoding="utf-8") as fp:
        level2_rules = json.load(fp)
    with (config_dir / "detailed_level2_rules.json").open("r", encoding="utf-8") as fp:
        detailed_level2_rules = json.load(fp)
    with (config_dir / "boundary_rules.json").open("r", encoding="utf-8") as fp:
        boundary_rules = json.load(fp)
    with (config_dir / "structure_rules.json").open("r", encoding="utf-8") as fp:
        structure_rules = json.load(fp)

    ruleset: RuleSet = {
        "category_tree": category_tree,
        "category_lines": build_category_lines(category_tree),
        "level1_rules": level1_rules,
        "level2_rules": level2_rules,
        "detailed_level2_rules": detailed_level2_rules,
        "boundary_rules": boundary_rules,
        "domain_strong_keywords": structure_rules["domain_strong_keywords"],
        "same_domain_components": structure_rules["same_domain_components"],
    }
    validate_ruleset(ruleset, source=f"json_ruleset({config_dir})")
    return ruleset


@lru_cache(maxsize=4)
def get_ruleset(source: Optional[str] = None) -> RuleSet:
    source = (source or RULE_SOURCE).lower()
    if source == "python":
        return get_python_ruleset()
    if source == "json":
        return load_json_ruleset()
    raise ValueError(f"未知 RULE_SOURCE: {source}")


def clear_ruleset_cache() -> None:
    get_ruleset.cache_clear()
