import numpy as np
import torch
import hello_uv.linreg as linreg

results = []


def report(label, ok, detail=""):
    print(("PASS     " if ok else "FAIL     ") + label + (f"  ({detail})" if detail and not ok else ""))
    results.append(ok)


try:
    import hello_uv.torchgrad as tg
    f = tg.autograd_gradient
except Exception as e:
    print(f"MISSING  autograd_gradient() in src/hello_uv/torchgrad.py  ({type(e).__name__}: {e})")
    raise SystemExit(f"\n0 of 7 checks passed")

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 7.0])
x_before = x.copy()

# 1 runs and returns two plain Python floats
try:
    got = f(1.0, 0.0, x, y)
    ok = isinstance(got, tuple) and len(got) == 2 and all(type(v) is float for v in got)
    report("1 returns a pair of plain Python floats (use .item())", ok, f"got {got!r}")
except Exception as e:
    got = None
    report("1 autograd_gradient runs", False, f"{type(e).__name__}: {e}")

# 2 known value
if got is not None:
    report("2 at (w=1, b=0) the gradient is (-11.333, -4.667)", np.allclose(got, [-34 / 3, -14 / 3], atol=1e-4), f"got {got}")

# 3 your NumPy arrays are not changed
report("3 your x array is unchanged afterwards", np.array_equal(x, x_before), f"x is now {x}")

# 4 agrees with your formula on random data
rng = np.random.default_rng(16)
ok, detail = True, ""
for _ in range(3):
    xs, ys = rng.normal(size=20), rng.normal(size=20)
    w, b = rng.normal(), rng.normal()
    try:
        a, t = linreg.gradient(w, b, xs, ys), f(w, b, xs, ys)
        if not np.allclose(a, t, atol=1e-4):
            ok, detail = False, f"formula {a} vs autograd {t}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
report("4 autograd agrees with your gradient() on 3 random data sets", ok, detail)

# 5 agrees with your numeric check
try:
    n, t = linreg.numeric_gradient(0.3, -0.7, x, y), f(0.3, -0.7, x, y)
    report("5 autograd agrees with your numeric_gradient()", np.allclose(n, t, atol=1e-4), f"numeric {n} vs autograd {t}")
except Exception as e:
    report("5 autograd vs numeric", False, f"{type(e).__name__}: {e}")

# 6 calling twice gives the same answer (no gradients piling up)
try:
    g1, g2 = f(0.5, 0.5, x, y), f(0.5, 0.5, x, y)
    report("6 two calls in a row give the same answer (fresh tensors each call)", np.allclose(g1, g2), f"{g1} then {g2}")
except Exception as e:
    report("6 two calls in a row", False, f"{type(e).__name__}: {e}")

# 7 at the best line the gradient is zero
xl = np.array([1.0, 2.0, 3.0]); yl = 2 * xl + 1
try:
    report("7 on the exact line y = 2x + 1, the gradient at (2, 1) is (0, 0)", np.allclose(f(2.0, 1.0, xl, yl), [0, 0], atol=1e-6), f"got {f(2.0, 1.0, xl, yl)}")
except Exception as e:
    report("7 gradient at the minimum", False, f"{type(e).__name__}: {e}")

print(f"\n{sum(results)} of {len(results)} checks passed")