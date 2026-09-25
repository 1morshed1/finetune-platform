"""Command-line entrypoint: ``ftp <command>``.

Commands:
  serve    Run the API + dashboard (uvicorn).
  demo     End-to-end, in-process: make a sample dataset, validate it, run a
           fine-tune job, and print the report, metrics, and leaderboard. No
           server needed — great for seeing the whole pipeline at once.
  sample   Write a sample JSONL dataset to a path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _sample_rows(n: int = 60) -> list[dict]:
    """A small synthetic instruction dataset (with a couple of intentional
    duplicates so the quality report has something to flag)."""
    topics = ["gravity", "photosynthesis", "recursion", "inflation", "vaccines",
              "encryption", "evolution", "tides", "black holes", "entropy"]
    rows = []
    for i in range(n):
        t = topics[i % len(topics)]
        rows.append({
            "prompt": f"Q{i}: Explain {t} in one clear sentence for a beginner (context {i}).",
            "completion": f"{t.capitalize()} is a concept best understood as example number {i}.",
            "label": t,
        })
    # Inject 3 exact duplicates to demonstrate dedup detection.
    rows.extend(rows[:3])
    return rows


def cmd_sample(args: argparse.Namespace) -> int:
    rows = _sample_rows(args.n)
    path = Path(args.path)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} examples to {path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Starting Fine-Tuning Platform on http://{args.host}:{args.port}")
    uvicorn.run("ftp.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the entire pipeline in-process and print a readable summary."""
    import os

    os.environ.setdefault("FTP_SIM_STEP_DELAY", "0")  # instant training for the demo

    from ftp.config import get_settings
    from ftp.data.validator import validate_file
    from ftp.db import Dataset, DatasetStatus, Job, JobStatus, ModelRecord, new_id, session_scope
    from ftp.registry import leaderboard
    from ftp.training import run_job

    settings = get_settings()
    settings.ensure_dirs()

    # 1. Sample dataset -> disk.
    path = settings.datasets_dir / "demo.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _sample_rows(60)), encoding="utf-8")

    # 2. Validate.
    print("\n=== 1. VALIDATE DATASET ===")
    result = validate_file(path, settings)
    rep = result.report
    print(f"  status            : {result.status}")
    print(f"  total examples    : {rep.total_examples}")
    print(f"  exact duplicates  : {rep.exact_duplicates}")
    print(f"  near duplicates   : {rep.near_duplicates}")
    pt = rep.prompt_token_stats
    print(f"  prompt tokens     : mean={pt.mean} p95={pt.p95}")
    print(f"  est. cost (openai): ${rep.cost.openai_usd}   (local gpu): ${rep.cost.local_gpu_usd}")
    print(f"  train/val split   : {rep.train_size}/{rep.val_size}")
    print(f"  recommendation    : {rep.recommendation.splitlines()[0]}")

    with session_scope() as s:
        ds = Dataset(id=new_id("ds"), name="demo", filename="demo.jsonl", path=str(path),
                     num_examples=rep.total_examples, status=DatasetStatus.READY,
                     quality_report=rep.to_dict())
        s.add(ds)
        s.flush()
        ds_id = ds.id
        job = Job(dataset_id=ds_id, base_model="demo", trainer_backend="sim",
                  hyperparameters={"epochs": 3, "batch_size": 8, "learning_rate": 2e-4,
                                   "lora_r": 8})
        s.add(job)
        s.flush()
        job_id = job.id

    # 3. Fine-tune (runs the whole DAG inline).
    print("\n=== 2. FINE-TUNE (prepare -> train -> evaluate -> register) ===")
    run_job(job_id)

    with session_scope() as s:
        job = s.get(Job, job_id)
        print(f"  job status        : {job.status}")
        final = (job.metrics or {}).get("final", {})
        print(f"  final train loss  : {final.get('train_loss')}")
        print(f"  final eval loss   : {final.get('eval_loss')}")
        print(f"  perplexity        : {final.get('perplexity')}")
        model = s.get(ModelRecord, job.model_id) if job.model_id else None
        if model:
            cmp = model.comparison_vs_base or {}
            print(f"  registered model  : {model.id} "
                  f"({model.name} v{model.version}) [{model.status}]")
            print(f"  vs base           : {cmp.get('verdict')}")

    print("\n=== 3. LEADERBOARD ===")
    with session_scope() as s:
        for i, m in enumerate(leaderboard(s), 1):
            el = (m.metrics or {}).get("eval_loss")
            print(f"  {i}. {m.name} v{m.version}  eval_loss={el}  [{m.status}]")

    if job.status == JobStatus.COMPLETED:
        print("\nDone. Run `ftp serve` and open http://localhost:8000 for the dashboard.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ftp", description="Fine-Tuning Platform CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run the API + dashboard")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_demo = sub.add_parser("demo", help="Run the full pipeline in-process")
    p_demo.set_defaults(func=cmd_demo)

    p_sample = sub.add_parser("sample", help="Write a sample JSONL dataset")
    p_sample.add_argument("path", help="Output path, e.g. sample.jsonl")
    p_sample.add_argument("-n", type=int, default=60, help="Number of examples")
    p_sample.set_defaults(func=cmd_sample)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
