# Reproducing V2.7

This repository separates three levels of reproducibility:

1. **CPU-only code and benchmark verification** validates the deterministic suite,
   tests, reports, and paired statistics.
2. **Evidence replay** regenerates the headline reports from the committed raw
   Dev36 trajectories without rerunning a model.
3. **Model rerun or retraining** requires separately obtained third-party model
   files and local adapters. Model weights are intentionally excluded from the
   curated public Git repository.

## Requirements

- Python 3.11+
- A recent `pip` or `uv`
- NVIDIA GPU and the `train` extra only for local model inference or training

## CPU-only verification

```text
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
device-agent-arena-v2 validate-composition-suite \
  --suite-root data/v2/deviceops-ul-v2-composition-v0.1
python tools/public_release.py --project-root . audit
```

On PowerShell, replace the line continuation character `\` with a backtick or place
the command on one line.

## Regenerate the final reports from raw trajectories

Use new output paths; report commands intentionally refuse to overwrite evidence.

```text
device-agent-arena-v2 report-composition \
  --input artifacts/v2/layered/runs/qwen3-1.7b-v2.7.1-refined-direct-dev36-tail-v0.1.jsonl \
  --json-output artifacts/cache/reproduced-direct.json

device-agent-arena-v2 report-composition \
  --input artifacts/v2/layered/runs/functiongemma-v2.5-fastpath-qwen-v2.7.1-dev36-tail-v0.2.jsonl \
  --json-output artifacts/cache/reproduced-layered.json

device-agent-arena-v2 compare-composition \
  --baseline-run artifacts/v2/layered/runs/qwen3-1.7b-v2.7.1-refined-direct-dev36-tail-v0.1.jsonl \
  --candidate-run artifacts/v2/layered/runs/functiongemma-v2.5-fastpath-qwen-v2.7.1-dev36-tail-v0.2.jsonl \
  --output artifacts/cache/reproduced-paired.json \
  --bootstrap-samples 10000 --seed 20260904
```

The expected headline values are Safe Strict `22/36 -> 31/36`, nine paired wins,
27 ties, zero losses, and a +25.0 percentage-point difference with a descriptive
task-level 95% paired-bootstrap interval of +11.1 to +38.9 points.

## Model and training requirements

- Qwen3-1.7B is obtained from its original distributor under Apache License 2.0.
- FunctionGemma-270M is obtained under Google's Gemma terms.
- The Qwen V2.7 refinement configuration continues from earlier adapters. Exact
  training therefore requires the declared initial adapter chain; the public Git
  export contains configs, dataset manifests, selected curricula, and training
  records, but not model weights.
- Run metadata pins model revisions, seeds, chat-template hashes, adapter hashes,
  source-tree hashes, and decoding settings. Hardware-dependent latency should not
  be expected to reproduce bit-for-bit.

## Produce a clean public tree

The complete local workspace may contain caches and historical experiments. Export
only the audited public allowlist into a new, nonexistent directory:

```text
python tools/public_release.py --project-root . audit
python tools/public_release.py --project-root . export ../device-agent-arena-public
```

The exporter never deletes or overwrites the destination. It writes
`PUBLIC_RELEASE_MANIFEST.json` with a SHA-256 and byte count for every copied file.
