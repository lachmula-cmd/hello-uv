import numpy as np
import torch
from torch import nn
import hello_uv.gd as gd

results = []


def report(label, ok, detail=""):
    print(("PASS     " if ok else "FAIL     ") + label + (f"  ({detail})" if detail and not ok else ""))
    results.append(ok)


def attempt(label, fn):
    try:
        return fn()
    except Exception as e:
        report(label, False, f"{type(e).__name__}: {e}")
        return None


try:
    import hello_uv.torchfit as tf
    LinReg, fit = tf.LinReg, tf.fit
except Exception as e:
    print(f"MISSING  LinReg or fit() in src/hello_uv/torchfit.py  ({type(e).__name__}: {e})")
    raise SystemExit("\n0 of 10 checks passed")

x = np.array([1.0, 2.0, 3.0])
y = np.array([3.0, 5.0, 7.0])          # the line y = 2x + 1, as in check_gd.py
x_before = x.copy()

# 1-2 the model
m = attempt("1 LinReg() can be created", LinReg)
if m is not None:
    n = sum(p.numel() for p in m.parameters())
    report("1 LinReg is an nn.Module with exactly 2 numbers to learn (w and b)", isinstance(m, nn.Module) and n == 2, f"found {n}")
    out = attempt("2 model(x) runs", lambda: m(torch.from_numpy(x).reshape(-1, 1)))
    if out is not None:
        report("2 a (3, 1) column of inputs gives a (3, 1) column of predictions", tuple(out.shape) == (3, 1), f"got shape {tuple(out.shape)}")
else:
    report("2 model(x) needs a model first", False)

# 3 shape of fit's answer
got = attempt("3 fit runs", lambda: fit(x, y, 0.05, 3))
if got is not None:
    ok = isinstance(got, tuple) and len(got) == 3 and type(got[0]) is float and type(got[1]) is float \
        and isinstance(got[2], list) and len(got[2]) == 3 and all(type(v) is float for v in got[2])
    report("3 fit returns (w, b, losses): two floats and a list of `steps` floats", ok, f"got {got!r}")
    # 4 identical to your M1 gradient descent
    w1, b1, l1 = gd.descend(0.0, 0.0, x, y, 0.05, 3)
    report("4 3 SGD steps match your gd.descend: same losses, same w and b",
           ok and np.allclose(got[2], l1[:3], atol=1e-6) and np.allclose(got[:2], [w1, b1], atol=1e-6),
           f"descend losses {np.round(l1[:3], 4)}, w={w1:.4f}, b={b1:.4f}")
else:
    report("4 skipped because fit did not run", False)

# 5 finds the line
got = attempt("5 fit 1000 steps", lambda: fit(x, y, 0.05, 1000))
if got is not None:
    report("5 SGD, lr = 0.05, 1000 steps: w is 2 and b is 1", np.allclose(got[:2], [2, 1], atol=1e-4), f"got w={got[0]:.4g}, b={got[1]:.4g}")

# 6 agrees with descend on random data
rng = np.random.default_rng(18)
ok, detail = True, ""
for _ in range(3):
    xs, ys = rng.normal(size=20), rng.normal(size=20)
    try:
        a, t = gd.descend(0.0, 0.0, xs, ys, 0.1, 50), fit(xs, ys, 0.1, 50)
        if not np.allclose(a[:2], t[:2], atol=1e-6):
            ok, detail = False, f"descend w, b = {a[0]:.4f}, {a[1]:.4f} vs fit {t[0]:.4f}, {t[1]:.4f}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
report("6 SGD agrees with your gd.descend on 3 random data sets", ok, detail)

# 7 x unchanged
report("7 your x array is unchanged afterwards", np.array_equal(x, x_before), f"x is now {x}")

# 8 fresh start every call
pair = attempt("8 two calls in a row", lambda: (fit(x, y, 0.05, 5), fit(x, y, 0.05, 5)))
if pair is not None:
    g1, g2 = pair
    report("8 two calls in a row give the same answer (new model, starting at 0, each call)", np.allclose(g1[:2], g2[:2]), f"{g1[:2]} then {g2[:2]}")

# 9 lr too big, same as M1
got = attempt("9 lr too big", lambda: fit(x, y, 0.19, 50))
if got is not None:
    report("9 SGD with lr = 0.19 blows up, as in M1 (last loss > first)", got[2][-1] > got[2][0], f"first {got[2][0]:.4g}, last {got[2][-1]:.4g}")

# 10 Adam
got = attempt("10 Adam", lambda: fit(x, y, 0.1, 2000, opt="adam"))
if got is not None:
    report("10 opt='adam', lr = 0.1, 2000 steps also finds w = 2, b = 1", np.allclose(got[:2], [2, 1], atol=1e-3), f"got w={got[0]:.4g}, b={got[1]:.4g}")
