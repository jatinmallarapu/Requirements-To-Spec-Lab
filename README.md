# Requirements to Spec Lab

A portfolio experiment in requirements interpretation. Given a short software requirement, produce a JSON specification for a scheduled job with a job identifier, cadence, and maximum retry count. All examples are synthetic.

> Example input: The inventory_sync job runs daily and may retry up to 2 times.
>
> Expected output: `{"job":"inventory_sync","cadence":"daily","retries":2}`

The benchmark holds out both job identifiers and requirement phrasings. It measures JSON validity, schema validity, and exact match, and records failures by cause. The rule-based baseline and learned baseline run offline. A separate supervised fine-tuning path adapts [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) with [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora).
