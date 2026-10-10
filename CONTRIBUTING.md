# Contributing

Contributions are welcome when they preserve the benchmark's reproducibility and
claim discipline.

## Before opening a pull request

1. Create or update a focused test at the public behavior boundary.
2. Keep Train, Dev, and Test provenance explicit. Test tasks must never feed training
   data generation.
3. Never hand-edit aggregate metrics. Generate reports from committed raw runs.
4. Do not commit checkpoints, caches, credentials, user data, or hidden test secrets.
5. Run:

   ```text
   python -m pytest -q
   python -m ruff check .
   python tools/public_release.py --project-root . audit
   ```

## Reporting results

Every result should name the task split, sample size, model revision, adapter,
decoding configuration, source revision, hardware, and known limitations. Label
post-hoc diagnostics as post-hoc. A development result must not be described as a
sealed test or production success rate.

By submitting a contribution, you agree that it is licensed under Apache License
2.0 and that you have the right to submit it.
