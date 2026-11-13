"""Adapt Qwen with LoRA and compare it with the same unadapted model."""
import argparse
import json
from pathlib import Path

from lab import make_split, score_predictions, write_json

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
SYSTEM = (
    "Convert the scheduled-job requirement to one JSON object only. "
    "Use exactly these keys: job (the job identifier), cadence "
    "(hourly, daily or weekly), retries (integer from 0 to 3)."
)


def prompt_ids(tokenizer, text):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def training_example(tokenizer, row):
    prompt = prompt_ids(tokenizer, row["requirement"])
    answer = tokenizer.encode(json.dumps(row["spec"], separators=(",", ":")), add_special_tokens=False)
    answer.append(tokenizer.eos_token_id)
    input_ids = prompt + answer
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt) + answer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/qwen"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Use a subset per split for a smoke run")
    args = parser.parse_args()
    if args.epochs < 1 or (args.limit is not None and args.limit < 1):
        parser.error("epochs and limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from huggingface_hub import model_info
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Trainer, TrainingArguments, set_seed,
    )

    set_seed(42)
    revision = model_info(MODEL, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base.to(device)

    def generate(model, text):
        ids = torch.tensor([prompt_ids(tokenizer, text)], device=device)
        with torch.inference_mode():
            result = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=96,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True).strip()

    train, dev, test = (make_split(name)[:args.limit] for name in ("train", "dev", "test"))
    base.eval()
    baseline = score_predictions(test, lambda text: generate(base, text))
    adapted = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    ))
    trainer = Trainer(
        model=adapted,
        args=TrainingArguments(
            output_dir=str(args.output / "trainer"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=3e-4,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            seed=42,
        ),
        train_dataset=[training_example(tokenizer, row) for row in train],
        eval_dataset=[training_example(tokenizer, row) for row in dev],
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
    )
    trainer.train()
    adapter_path = args.output / "adapter"
    adapted.save_pretrained(adapter_path, safe_serialization=True)
    # Reload the saved weights against a clean copy of the pinned base.
    del trainer, adapted, base
    if device == "cuda":
        torch.cuda.empty_cache()
    clean = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision)
    restored = PeftModel.from_pretrained(clean, adapter_path).to(device).eval()
    write_json(args.output / "report.json", {
        "model": MODEL, "revision": revision, "torch_version": str(torch.__version__),
        "device": device, "seed": 42, "epochs": args.epochs, "smoke_limit": args.limit,
        "split_sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "base": baseline,
        "lora": score_predictions(test, lambda text: generate(restored, text)),
    })


if __name__ == "__main__":
    main()
