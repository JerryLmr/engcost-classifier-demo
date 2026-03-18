#!/usr/bin/env python3
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from data.boundaries import BOUNDARY_RULES  # noqa: E402
from data.categories import CATEGORY_TREE  # noqa: E402
from data.rules import DETAILED_LEVEL2_RULES, LEVEL1_RULES, LEVEL2_RULES  # noqa: E402
from data.structure_rules import DOMAIN_STRONG_KEYWORDS, SAME_DOMAIN_COMPONENTS  # noqa: E402


CONFIG_DIR = BACKEND_DIR / "config"


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def dump_json(name: str, payload) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / name
    with path.open("w", encoding="utf-8") as fp:
        json.dump(normalize(payload), fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    print(f"[OK ] {path}")


def main() -> int:
    dump_json("taxonomy.json", CATEGORY_TREE)
    dump_json("level1_rules.json", LEVEL1_RULES)
    dump_json("level2_rules.json", LEVEL2_RULES)
    dump_json("detailed_level2_rules.json", DETAILED_LEVEL2_RULES)
    dump_json("boundary_rules.json", BOUNDARY_RULES)
    dump_json(
        "structure_rules.json",
        {
            "domain_strong_keywords": DOMAIN_STRONG_KEYWORDS,
            "same_domain_components": SAME_DOMAIN_COMPONENTS,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
