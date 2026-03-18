import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rule_loader import get_python_ruleset, load_json_ruleset


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


class RuleLoaderTestCase(unittest.TestCase):
    def test_json_ruleset_matches_python_baseline(self):
        python_ruleset = get_python_ruleset()
        json_ruleset = load_json_ruleset()

        self.assertEqual(normalize(json_ruleset["category_tree"]), normalize(python_ruleset["category_tree"]))
        self.assertEqual(normalize(json_ruleset["level1_rules"]), normalize(python_ruleset["level1_rules"]))
        self.assertEqual(normalize(json_ruleset["level2_rules"]), normalize(python_ruleset["level2_rules"]))
        self.assertEqual(normalize(json_ruleset["detailed_level2_rules"]), normalize(python_ruleset["detailed_level2_rules"]))
        self.assertEqual(normalize(json_ruleset["boundary_rules"]), normalize(python_ruleset["boundary_rules"]))
        self.assertEqual(normalize(json_ruleset["domain_strong_keywords"]), normalize(python_ruleset["domain_strong_keywords"]))
        self.assertEqual(normalize(json_ruleset["same_domain_components"]), normalize(python_ruleset["same_domain_components"]))

    def test_invalid_allowed_level2_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp = Path(tmp_dir)
            baseline = get_python_ruleset()
            payloads = {
                "taxonomy.json": baseline["category_tree"],
                "level1_rules.json": baseline["level1_rules"],
                "level2_rules.json": baseline["level2_rules"],
                "detailed_level2_rules.json": baseline["detailed_level2_rules"],
                "boundary_rules.json": baseline["boundary_rules"],
                "structure_rules.json": {
                    "domain_strong_keywords": baseline["domain_strong_keywords"],
                    "same_domain_components": baseline["same_domain_components"],
                },
            }
            payloads["boundary_rules.json"][0]["allowed_level2"] = ["不存在的二级类"]
            for name, payload in payloads.items():
                with (temp / name).open("w", encoding="utf-8") as fp:
                    json.dump(normalize(payload), fp, ensure_ascii=False, indent=2)
            with self.assertRaises(ValueError):
                load_json_ruleset(temp)


if __name__ == "__main__":
    unittest.main()
