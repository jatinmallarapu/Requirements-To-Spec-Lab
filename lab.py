"""Synthetic scheduled-job benchmark and dependency-free baselines."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import random
import re

CADENCES = {
    "hourly": ("hourly", "every hour", "once an hour"),
    "daily": ("daily", "every day", "once a day"),
    "weekly": ("weekly", "every week", "once a week"),
}
TEMPLATES = {
    "train": (
        "The {job} job runs {cadence} and may retry up to {n} times.",
        "Schedule {job} {cadence}, with a maximum of {n} retries.",
        "Allow {n} retries for {job}, which runs {cadence}.",
    ),
    "dev": ("Run {job} {cadence}. Cap retries at {n}.",),
    "test": (
        "Execute {job} {cadence}. Retry no more than {n} times.",
        "Give {job} a retry budget of {n}. Its schedule is {cadence}.",
    ),
}
JOB_PATTERN = r"\b[a-z]+_[a-z]+\b"
WORDS = ("zero", "one", "two", "three")
JOB_NAMES = {
    "train": ("inventory_sync", "catalog_refresh", "image_resize", "usage_rollup"),
    "dev": ("archive_cleanup", "report_export"),
    "test": ("cache_refresh", "snapshot_backup", "queue_cleanup"),
}


def make_split(split):
    rows = []
    for job, (cadence, aliases), n, template in itertools.product(
        JOB_NAMES[split], CADENCES.items(), range(len(WORDS)), TEMPLATES[split]
    ):
        # Hold out number wording in one test template.
        number = WORDS[n] if split == "test" and template == TEMPLATES["test"][1] else n
        rows.append({
            "requirement": template.format(job=job, cadence=aliases[n % len(aliases)], n=number),
            "spec": {"job": job, "cadence": cadence, "retries": n},
        })
    random.Random(42).shuffle(rows)
    return rows


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def validate(raw, expected):
    try:
        obj = json.loads(raw, object_pairs_hook=unique_object)
    except ValueError:
        return "invalid_json"
    if (
        not isinstance(obj, dict)
        or set(obj) != {"job", "cadence", "retries"}
        or not isinstance(obj["job"], str)
        or not re.fullmatch(JOB_PATTERN, obj["job"])
        or not isinstance(obj["cadence"], str)
        or obj["cadence"] not in CADENCES
        or type(obj["retries"]) is not int
        or not 0 <= obj["retries"] < len(WORDS)
    ):
        return "invalid_schema"
    return "exact_match" if obj == expected else "wrong_value"


def extract_job(text):
    match = re.search(JOB_PATTERN, text)
    return match.group() if match else ""


def rules(text, recognize_words=False):
    cadence = next(
        (key for key, aliases in CADENCES.items() if any(
            re.search(r"\b" + re.escape(alias) + r"\b", text) for alias in aliases
        )),
        None,
    )
    number = re.search(r"\b[0-3]\b", text)
    retries = int(number.group()) if number else None
    if retries is None and recognize_words:
        word = re.search(r"\b(" + "|".join(WORDS) + r")\b", text)
        retries = WORDS.index(word.group()) if word else None
    return json.dumps({
        "job": extract_job(text),
        "cadence": cadence,
        "retries": retries,
    })


def tokenize(text):
    return re.findall(r"[a-z]+|[0-9]+", re.sub(JOB_PATTERN, " JOB ", text).lower())


def fit_classifier(rows, field):
    counts = {}
    for row in rows:
        label = str(row["spec"][field])
        counts.setdefault(label, Counter()).update(tokenize(row["requirement"]))
    return {label: dict(tokens) for label, tokens in counts.items()}


def classify(text, counts):
    vocabulary = set().union(*(tokens.keys() for tokens in counts.values()))
    features = [token for token in tokenize(text) if token in vocabulary]

    def score(label):
        tokens = counts[label]
        denominator = sum(tokens.values()) + len(vocabulary)
        return sum(math.log((tokens.get(token, 0) + 1) / denominator) for token in features)

    return max(sorted(counts), key=score)


def learned(text, weights):
    return json.dumps({
        "job": extract_job(text),
        "cadence": classify(text, weights["cadence"]),
        "retries": int(classify(text, weights["retries"])),
    })


def score_predictions(rows, predict):
    outcomes = Counter()
    predictions = []
    for row in rows:
        response = predict(row["requirement"])
        status = validate(response, row["spec"])
        outcomes[status] += 1
        predictions.append({**row, "response": response, "status": status})
    n = len(rows)
    return {
        "examples": n,
        "json_valid": n - outcomes["invalid_json"],
        "schema_valid": outcomes["wrong_value"] + outcomes["exact_match"],
        "exact_match": outcomes["exact_match"],
        "outcomes": dict(sorted(outcomes.items())),
        "predictions": predictions,
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = make_split("train")
    weights = {field: fit_classifier(train, field) for field in ("cadence", "retries")}
    weight_path = args.output / "baseline_weights.json"
    write_json(weight_path, weights)
    weights = json.loads(weight_path.read_text(encoding="utf-8"))
    test = make_split("test")
    report = {
        "split_sizes": {split: len(make_split(split)) for split in TEMPLATES},
        "numeric_rules": score_predictions(test, rules),
        "rules_with_number_words": score_predictions(test, lambda text: rules(text, recognize_words=True)),
        "learned": score_predictions(test, lambda text: learned(text, weights)),
    }
    write_json(args.output / "baseline_report.json", report)


if __name__ == "__main__":
    main()
