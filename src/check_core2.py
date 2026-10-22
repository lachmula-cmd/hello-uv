import json
import tempfile
from pathlib import Path

import core2

results = []


def report(label, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}     {label} {detail}")
    results.append(ok)


def run(label, func_name, *args):
    func = getattr(core2, func_name, None)
    if func is None:
        print(f"MISSING  {func_name}() is not defined in core2.py")
        results.append(False)
        return None, False
    try:
        return func(*args), True
    except Exception as e:
        print(f"ERROR    {label}: {type(e).__name__}: {e}")
        results.append(False)
        return None, False


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    (tmp / "nums.txt").write_text("2\n4\n\n6\n8\n")
    got, ran = run("read_numbers", "read_numbers", tmp / "nums.txt")
    if ran:
        report("read_numbers(nums.txt)", got == [2.0, 4.0, 6.0, 8.0], f"-> {got!r} (expected [2.0, 4.0, 6.0, 8.0]; blank line skipped)")

    got, ran = run("safe_mean([2, 4, 6, 8])", "safe_mean", [2, 4, 6, 8])
    if ran:
        report("safe_mean([2, 4, 6, 8])", got == 5.0, f"-> {got!r} (expected 5.0)")

    got, ran = run("safe_mean([])", "safe_mean", [])
    if ran:
        report("safe_mean([])", got is None, f"-> {got!r} (expected None, not a ZeroDivisionError)")

    got, ran = run("write_summary", "write_summary", tmp / "summary.txt", [2.0, 4.0, 6.0, 8.0])
    if ran:
        text = (tmp / "summary.txt").read_text() if (tmp / "summary.txt").exists() else "(no file written)"
        report("write_summary(...)", text.strip() == "n=4 mean=5.0", f"-> file says {text.strip()!r} (expected 'n=4 mean=5.0')")

    (tmp / "config.json").write_text(json.dumps({"lr": 0.01, "epochs": 5, "model": "mlp"}))
    got, ran = run("load_config", "load_config", tmp / "config.json")
    if ran:
        report("load_config(config.json)", got == {"lr": 0.01, "epochs": 5, "model": "mlp"}, f"-> {got!r}")

print(f"\n{sum(results)} of {len(results)} checks passed")