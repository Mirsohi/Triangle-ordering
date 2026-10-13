# DOVE baseline and reference kernels

- `src/dove/`: the gradient-based DOVE baseline (smoothed logistic relaxation of the work), plus the reference orders and helpers that `1_preregistered_eval` imports and hash-checks.
- `cpp/`: the reference exact adaptive-forward triangle lister and exact refinement kernels.
- `third_party/` and `scripts/setup_volt.sh`: how to obtain and patch VOLT (Neigh; GPL-3.0). VOLT itself is not redistributed.
- `tests/test_dove.py`: unit tests.
- `data/heldout_benchmark/PROTOCOL.md`: the held-out data protocol.

Install the dependencies with `pip install -r requirements.txt`. Then add `src` to `PYTHONPATH`.
