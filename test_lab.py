import json
import unittest

from lab import make_split, rules, score_predictions, validate


class BenchmarkTests(unittest.TestCase):
    def test_split_isolation_and_schema(self):
        splits = [make_split(split) for split in ("train", "dev", "test")]
        seen = set()
        for rows in splits:
            jobs = {row["spec"]["job"] for row in rows}
            self.assertTrue(seen.isdisjoint(jobs))
            seen.update(jobs)
            self.assertEqual(len(rows), len({row["requirement"] for row in rows}))
            for row in rows:
                self.assertEqual(validate(json.dumps(row["spec"]), row["spec"]), "exact_match")

    def test_invalid_types_duplicates_and_wrong_values(self):
        expected = {"job": "cache_refresh", "cadence": "daily", "retries": 1}
        self.assertEqual(validate('{"job":"cache_refresh","cadence":"daily","retries":true}', expected),
                         "invalid_schema")
        self.assertEqual(validate('{"retries":0,"retries":1}', expected), "invalid_json")
        self.assertEqual(validate('{"job":"cache_refresh","cadence":[],"retries":1}', expected),
                         "invalid_schema")
        self.assertEqual(validate(json.dumps(dict(expected, retries=2)), expected), "wrong_value")

    def test_rules_baseline_distinguishes_number_formats(self):
        report = score_predictions(make_split("test"), rules)
        self.assertEqual(report["examples"], 72)
        self.assertEqual(report["exact_match"], 36)
        self.assertEqual(report["outcomes"]["invalid_schema"], 36)
        expanded = score_predictions(make_split("test"), lambda text: rules(text, recognize_words=True))
        self.assertEqual(expanded["exact_match"], 72)


if __name__ == "__main__":
    unittest.main()
