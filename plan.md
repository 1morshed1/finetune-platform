## Project J: Fine-Tuning Pipeline as a Service

### Why This Project

You list fine-tuning experience on your CV (HRNet pose estimation with pruning and quantization). But that's a single model for a single task. This project generalizes that experience into a platform: users upload a dataset, select a base model, configure training parameters, and the system handles the rest — data validation, training job orchestration, evaluation, model registry, and one-click deployment.

This completes the ML lifecycle. You've built inference (serving models), retrieval (RAG), orchestration (agents), evaluation (EvalKit), monitoring (MLOps pipeline), and feature serving. Fine-tuning is the missing piece: creating the models themselves.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Fine-Tuning Platform                                                        │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  API (FastAPI)                                                       │   │
│  │                                                                      │   │
│  │  POST /api/v1/datasets                                               │   │
│  │    Upload training data (JSONL: {prompt, completion} pairs)          │   │
│  │    Auto-validate: format check, dedup, toxicity scan, split          │   │
│  │                                                                      │   │
│  │  POST /api/v1/fine-tune                                              │   │
│  │    {dataset_id, base_model, hyperparameters}                         │   │
│  │    Creates a training job → Airflow DAG                              │   │
│  │                                                                      │   │
│  │  GET /api/v1/jobs/:id                                                │   │
│  │    Status, training curves (loss, eval metrics), ETA                 │   │
│  │                                                                      │   │
│  │  GET /api/v1/jobs/:id/logs                                           │   │
│  │    Stream training logs via SSE                                      │   │
│  │                                                                      │   │
│  │  POST /api/v1/models/:id/deploy                                     │   │
│  │    Deploy fine-tuned model to inference endpoint                     │   │
│  │                                                                      │   │
│  │  POST /api/v1/models/:id/evaluate                                   │   │
│  │    Run eval suite against fine-tuned model (uses Project A)          │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Data Validation Pipeline                                            │   │
│  │                                                                      │   │
│  │  On dataset upload:                                                  │   │
│  │  1. Format validation (JSONL, required fields, encoding)             │   │
│  │  2. Deduplication (exact + near-duplicate via MinHash)               │   │
│  │  3. Quality scoring (perplexity per example, flag outliers)          │   │
│  │  4. Toxicity/safety scan (flag potentially harmful examples)         │   │
│  │  5. Train/validation split (90/10, stratified if labels present)     │   │
│  │  6. Token count analysis (distribution, total cost estimate)         │   │
│  │  7. Generate data quality report                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Training Orchestrator (Airflow)                                     │   │
│  │                                                                      │   │
│  │  DAG: fine_tune_job                                                  │   │
│  │                                                                      │   │
│  │  1. Prepare: download dataset, format for training framework         │   │
│  │  2. Train:                                                           │   │
│  │     - LoRA/QLoRA via Hugging Face PEFT (for open models)             │   │
│  │     - OR OpenAI fine-tuning API (for GPT models)                     │   │
│  │     - Log metrics to MLflow at each step                             │   │
│  │  3. Evaluate:                                                        │   │
│  │     - Run validation set through fine-tuned model                    │   │
│  │     - Compare against base model on same examples                   │   │
│  │     - Compute perplexity, BLEU/ROUGE (if applicable), custom metrics│   │
│  │  4. Register:                                                        │   │
│  │     - If metrics meet threshold → register in MLflow model registry  │   │
│  │     - If not → mark job as "completed_below_threshold"               │   │
│  │  5. Notify: send results to user via webhook/email                   │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Model Registry (MLflow)                                             │   │
│  │                                                                      │   │
│  │  Each fine-tuned model stores:                                       │   │
│  │  - Base model name and version                                       │   │
│  │  - Training dataset reference                                        │   │
│  │  - Hyperparameters used                                              │   │
│  │  - Training metrics (loss curves, eval scores)                       │   │
│  │  - Model artifacts (LoRA adapters or full weights)                   │   │
│  │  - Comparison vs base model                                          │   │
│  │  - Lineage: which dataset + config produced this model               │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Deployment (Kubernetes)                                             │   │
│  │                                                                      │   │
│  │  On deploy:                                                          │   │
│  │  1. Load base model + LoRA adapter into inference server             │   │
│  │     (vLLM or Ollama with custom model)                               │   │
│  │  2. Create K8s Deployment + Service + Ingress                        │   │
│  │  3. Register in Project 12 (LLM Gateway) as a new model             │   │
│  │  4. Canary: route 10% traffic to fine-tuned model                    │   │
│  │  5. Monitor: if error rate spikes, auto-rollback                     │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Dashboard:                                                                  │
│  - Job status (queued, training, evaluating, complete, failed)              │
│  - Live training loss curves                                                │
│  - Side-by-side: base model vs fine-tuned model on example prompts          │
│  - Cost tracker: compute cost + API cost per job                            │
│  - Model leaderboard: all fine-tuned models ranked by eval score            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Implementation: Data Quality Report

This is the part that differentiates a platform from a script:

```python
class DataQualityReport:
    def generate(self, dataset: list[dict]) -> QualityReport:
        total_examples = len(dataset)
        
        # Token distribution
        prompt_tokens = [count_tokens(ex["prompt"]) for ex in dataset]
        completion_tokens = [count_tokens(ex["completion"]) for ex in dataset]
        
        # Deduplication
        unique_prompts = len(set(ex["prompt"] for ex in dataset))
        near_dupes = self.find_near_duplicates(dataset)
        
        # Quality scoring (use perplexity from a small model)
        quality_scores = self.score_quality(dataset)
        low_quality = [i for i, s in enumerate(quality_scores) if s < 0.3]
        
        # Cost estimation
        total_tokens = sum(prompt_tokens) + sum(completion_tokens)
        estimated_cost_openai = total_tokens / 1_000_000 * 8.0  # GPT-4o-mini fine-tuning rate
        estimated_cost_local = self.estimate_gpu_hours(total_tokens) * 0.50  # spot GPU price
        
        return QualityReport(
            total_examples=total_examples,
            unique_examples=unique_prompts,
            near_duplicates=len(near_dupes),
            prompt_token_stats=Stats(
                mean=np.mean(prompt_tokens),
                median=np.median(prompt_tokens),
                p95=np.percentile(prompt_tokens, 95),
                max=max(prompt_tokens),
            ),
            completion_token_stats=Stats(
                mean=np.mean(completion_tokens),
                median=np.median(completion_tokens),
                p95=np.percentile(completion_tokens, 95),
                max=max(completion_tokens),
            ),
            low_quality_examples=low_quality,
            estimated_cost_openai=estimated_cost_openai,
            estimated_cost_local=estimated_cost_local,
            estimated_training_time_minutes=total_tokens / 50_000,  # rough estimate
            recommendation=self.generate_recommendation(total_examples, unique_prompts, low_quality),
        )
    
    def generate_recommendation(self, total, unique, low_quality) -> str:
        issues = []
        if unique / total < 0.9:
            issues.append(f"High duplication rate ({100-unique/total*100:.0f}%). Remove duplicates to avoid overfitting.")
        if len(low_quality) / total > 0.1:
            issues.append(f"{len(low_quality)} examples flagged as low quality. Review and remove or improve them.")
        if total < 100:
            issues.append("Dataset is small (<100 examples). Consider adding more examples for better generalization.")
        if total > 10000:
            issues.append("Large dataset. Consider using a subset for initial experiments to reduce cost.")
        
        if not issues:
            return "Dataset looks good. Ready for fine-tuning."
        return "Issues found:\n" + "\n".join(f"- {i}" for i in issues)
```

### How This Connects to Everything

The fine-tuning platform is the keystone that ties your other projects together:

- **Project A (EvalKit):** evaluates the fine-tuned model against the base model.
- **Project C (MLOps Pipeline):** monitors the fine-tuned model in production for drift.
- **Project D (Inference Server):** serves the fine-tuned model with LoRA adapter hot-loading.
- **Project 12 (LLM Gateway):** routes traffic to the fine-tuned model alongside commercial models.
- **Project G (Prompt Management):** manages the prompts used with the fine-tuned model.

You've built the entire AI lifecycle.
