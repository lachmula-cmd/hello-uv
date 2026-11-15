import numpy as np
import gd

results = []


def report(label, ok, detail=""):
    print(("PASS     " if ok else "FAIL     ") + label + (f"  ({detail})" if detail and not ok else ""))
    results.append(ok)


def run(label, func_name, *args):
    func = getattr(gd, func_name, None)
    if func is None:
        print(f"MISSING  {func_name}() is not defined in gd.py")
        results.append(False)
        return None
    try:
        return func(*args)
    except Exception as e:
        report(label, False, f"{type(e).__name__}: {e}")
        return None


x = np.array([1.0, 2.0, 3.0])
y = np.array([3.0, 5.0, 7.0])          # the line y = 2x + 1 again

# 1 one step
got = run("1 step", "step", 0.0, 0.0, x, y, 0.05)
if got is not None:
    ok = isinstance(got, tuple) and len(got) == 2
    report("1 step returns the pair (new w, new b)", ok, f"got {got}")
    if ok:
        report("1 step from (0, 0) with lr = 0.05 lands on (1.1333, 0.5)", np.allclose(got, [17 / 15, 0.5]), f"got {got}")

# 2 descend: shape of the answer
got = run("2 descend", "descend", 0.0, 0.0, x, y, 0.05, 3)
if got is not None:
    ok = isinstance(got, tuple) and len(got) == 3
    report("2 descend returns (w, b, losses)", ok, f"got {type(got).__name__} of length {len(got) if hasattr(got, '__len__') else '?'}")
    if ok:
        losses = list(got[2])
        report("2 losses has steps + 1 entries (the start, then one per step)", len(losses) == 4, f"got {len(losses)}")
        report("2 the first three steps give 27.67, 5.49, 1.09, 0.22",
               len(losses) == 4 and np.allclose(losses, [83 / 3, 5.488519, 1.089712, 0.217207], atol=1e-4), f"got {[round(v, 4) for v in losses]}")

# 3 it finds the line
got = run("3 descend 1000 steps", "descend", 0.0, 0.0, x, y, 0.05, 1000)
if got is not None and isinstance(got, tuple) and len(got) == 3:
    w, b, losses = got
    report("3 after 1000 steps with lr = 0.05, w is 2 and b is 1", np.allclose([w, b], [2, 1], atol=1e-4), f"got w={w:.4g}, b={b:.4g}")
    report("3 with lr = 0.05 the loss never goes up", bool(np.all(np.diff(losses) <= 1e-12)))

# 4 it works on other data (no hard-coded answers)
x2, y2 = np.array([0.0, 1.0, 2.0, 3.0]), np.array([4.0, 3.0, 2.0, 1.0])   # y = -x + 4
got = run("4 other data", "descend", 0.0, 0.0, x2, y2, 0.05, 2000)
if got is not None and isinstance(got, tuple) and len(got) == 3:
    report("4 on y = -x + 4 it finds w = -1, b = 4", np.allclose(got[:2], [-1, 4], atol=1e-3), f"got w={got[0]:.4g}, b={got[1]:.4g}")

# 5 too big a learning rate
got = run("5 lr too big", "descend", 0.0, 0.0, x, y, 0.19, 50)
if got is not None and isinstance(got, tuple) and len(got) == 3:
    report("5 with lr = 0.19 the loss blows up (last > first)", got[2][-1] > got[2][0], f"first {got[2][0]:.4g}, last {got[2][-1]:.4g}")

print(f"\n{sum(results)} of {len(results)} checks passed")