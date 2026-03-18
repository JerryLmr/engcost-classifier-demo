from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_RULESET_KEYS = {
    "category_tree",
    "level1_rules",
    "level2_rules",
    "detailed_level2_rules",
    "boundary_rules",
    "domain_strong_keywords",
    "same_domain_components",
}


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ensure_keyword_pairs(value: Any, path: str) -> None:
    _ensure(isinstance(value, list), f"{path} 必须是列表")
    for item in value:
        _ensure(isinstance(item, list) and len(item) == 2, f"{path} 中的规则必须是 [keyword, weight]")
        keyword, weight = item
        _ensure(isinstance(keyword, str) and keyword, f"{path} 中 keyword 必须是非空字符串")
        _ensure(isinstance(weight, int) and weight > 0, f"{path} 中 weight 必须是正整数")


def validate_ruleset(ruleset: Mapping[str, Any], source: str = "ruleset") -> None:
    missing = REQUIRED_RULESET_KEYS - set(ruleset)
    _ensure(not missing, f"{source} 缺少必要字段: {', '.join(sorted(missing))}")

    category_tree = ruleset["category_tree"]
    _ensure(isinstance(category_tree, dict) and category_tree, f"{source}.category_tree 必须是非空字典")
    for level1, level2_list in category_tree.items():
        _ensure(isinstance(level1, str) and level1, f"{source}.category_tree 一级分类必须是非空字符串")
        _ensure(isinstance(level2_list, list) and level2_list, f"{source}.category_tree[{level1}] 必须是非空列表")
        for level2 in level2_list:
            _ensure(isinstance(level2, str) and level2, f"{source}.category_tree[{level1}] 二级分类必须是非空字符串")

    level1_rules = ruleset["level1_rules"]
    _ensure(isinstance(level1_rules, dict), f"{source}.level1_rules 必须是字典")
    for level1, rules in level1_rules.items():
        _ensure(level1 in category_tree, f"{source}.level1_rules 包含未知一级分类: {level1}")
        _ensure_keyword_pairs(rules, f"{source}.level1_rules[{level1}]")

    level2_rules = ruleset["level2_rules"]
    _ensure(isinstance(level2_rules, dict), f"{source}.level2_rules 必须是字典")
    for level1, mapping in level2_rules.items():
        _ensure(level1 in category_tree, f"{source}.level2_rules 包含未知一级分类: {level1}")
        _ensure(isinstance(mapping, dict), f"{source}.level2_rules[{level1}] 必须是字典")
        for level2, rules in mapping.items():
            _ensure(level2 in category_tree[level1], f"{source}.level2_rules[{level1}] 包含未知二级分类: {level2}")
            _ensure_keyword_pairs(rules, f"{source}.level2_rules[{level1}][{level2}]")

    detailed_level2_rules = ruleset["detailed_level2_rules"]
    _ensure(isinstance(detailed_level2_rules, dict), f"{source}.detailed_level2_rules 必须是字典")
    for level1, mapping in detailed_level2_rules.items():
        _ensure(level1 in category_tree, f"{source}.detailed_level2_rules 包含未知一级分类: {level1}")
        _ensure(isinstance(mapping, dict), f"{source}.detailed_level2_rules[{level1}] 必须是字典")
        for level2, rule in mapping.items():
            _ensure(level2 in category_tree[level1], f"{source}.detailed_level2_rules[{level1}] 包含未知二级分类: {level2}")
            _ensure(isinstance(rule, dict), f"{source}.detailed_level2_rules[{level1}][{level2}] 必须是字典")
            for field in ("object_keywords", "action_keywords", "weak_keywords"):
                if field in rule:
                    _ensure_keyword_pairs(rule[field], f"{source}.detailed_level2_rules[{level1}][{level2}].{field}")
            if "default_on_object" in rule:
                _ensure(isinstance(rule["default_on_object"], bool), f"{source}.detailed_level2_rules[{level1}][{level2}].default_on_object 必须是布尔值")
            if "min_score" in rule:
                _ensure(isinstance(rule["min_score"], int) and rule["min_score"] > 0, f"{source}.detailed_level2_rules[{level1}][{level2}].min_score 必须是正整数")

    boundary_rules = ruleset["boundary_rules"]
    _ensure(isinstance(boundary_rules, list), f"{source}.boundary_rules 必须是列表")
    for index, rule in enumerate(boundary_rules):
        path = f"{source}.boundary_rules[{index}]"
        _ensure(isinstance(rule, dict), f"{path} 必须是字典")
        level1 = rule.get("level1")
        _ensure(level1 in category_tree, f"{path}.level1 必须是已知一级分类")
        for field in ("any_keywords", "all_keywords", "none_keywords", "allowed_level2"):
            if field in rule:
                _ensure(isinstance(rule[field], list), f"{path}.{field} 必须是列表")
        for level2 in rule.get("allowed_level2", []):
            _ensure(level2 in category_tree[level1], f"{path}.allowed_level2 包含未知二级分类: {level2}")

    domain_strong_keywords = ruleset["domain_strong_keywords"]
    _ensure(isinstance(domain_strong_keywords, dict), f"{source}.domain_strong_keywords 必须是字典")
    for level1, keywords in domain_strong_keywords.items():
        _ensure(level1 in category_tree, f"{source}.domain_strong_keywords 包含未知一级分类: {level1}")
        _ensure(isinstance(keywords, list), f"{source}.domain_strong_keywords[{level1}] 必须是列表")
        for keyword in keywords:
            _ensure(isinstance(keyword, str) and keyword, f"{source}.domain_strong_keywords[{level1}] 必须是非空字符串")

    same_domain_components = ruleset["same_domain_components"]
    _ensure(isinstance(same_domain_components, dict), f"{source}.same_domain_components 必须是字典")
    for level1, components in same_domain_components.items():
        _ensure(level1 in category_tree, f"{source}.same_domain_components 包含未知一级分类: {level1}")
        _ensure(isinstance(components, dict), f"{source}.same_domain_components[{level1}] 必须是字典")
        for component, keywords in components.items():
            _ensure(isinstance(component, str) and component, f"{source}.same_domain_components[{level1}] 组件名必须是非空字符串")
            _ensure(isinstance(keywords, list), f"{source}.same_domain_components[{level1}][{component}] 必须是列表")
            for keyword in keywords:
                _ensure(isinstance(keyword, str) and keyword, f"{source}.same_domain_components[{level1}][{component}] 必须是非空字符串")


def validate_json_config_dir(config_dir: Path) -> None:
    required_files = [
        "taxonomy.json",
        "level1_rules.json",
        "level2_rules.json",
        "detailed_level2_rules.json",
        "boundary_rules.json",
        "structure_rules.json",
    ]
    for filename in required_files:
        path = config_dir / filename
        _ensure(path.exists(), f"缺少配置文件: {path}")
