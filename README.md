# Requirements to Spec Lab

A portfolio experiment in requirements interpretation. Given a short software requirement, produce a JSON specification for a scheduled job with a job identifier, cadence, and maximum retry count. All examples are synthetic.

> Example input: The inventory_sync job runs daily and may retry up to 2 times.
>
> Expected output: `{"job":"inventory_sync","cadence":"daily","retries":2}`

The benchmark holds out both job identifiers and requirement phrasings. It measures JSON validity, schema validity, and exact match, and records failures by cause. The rule-based baseline and learned baseline run offline. A separate supervised fine-tuning path adapts [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) with [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora).

## Reproduce the benchmark

Run these commands from the repository root with Python 3.12.

```bash
python -m unittest -v
python lab.py
```

The synthetic dataset generator creates 144 training, 24 development, and 72 test examples. Splits use disjoint job identifiers and templates. The second test template spells out retry counts. The run saves learned baseline weights and example-level predictions with failure categories under `results/`.

| Baseline | Valid JSON | Valid schema | Exact match |
| --- | ---: | ---: | ---: |
| Numeric rules | 72/72 | 36/72 | 36/72 |
| Multinomial Naive Bayes with identifier extraction | 72/72 | 72/72 | 63/72 |
| Rules with number-word support | 72/72 | 72/72 | 72/72 |

The expanded rules use knowledge of the benchmark's number formats and provide a domain-aware reference. The learned baseline made nine value errors. A perfect score on the expanded rules reflects the small synthetic grammar. The experiment does not measure performance on free-form requirements or real repositories.

## Fine-tune the open-weight model

```bash
python -m pip install -r requirements.txt
python sft.py
```

The experiment freezes the base model and trains LoRA weights on the query and value projections. Loss applies to the JSON answer and its end token. The evaluator compares the original model and a fresh reload of the saved adapter on the same test set with greedy decoding. The report records the resolved base revision, training configuration, counts, and predictions.

The [Benchmark workflow](../../actions) runs tests and a four-example LoRA smoke experiment on relevant pushes. To run the complete experiment, open the workflow, click **Run workflow**, and select `train_lora`. A successful LoRA job uploads the adapter and report as an artifact. The measured offline baseline results are checked in under [results/summary.json](results/summary.json).
