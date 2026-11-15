import numpy as np
import hello_uv.linreg as linreg

results = []


def report(label, ok, detail=""):
    print(("PASS     " if ok else "FAIL     ") + label + (f"  ({detail})" if detail and not ok else ""))
    results.append(ok)


def run(label, func_name, *args):
    func = getattr(linreg, func_name, None)
    if func is None:
        print(f"MISSING  {func_name}() is not defined in linreg.py")
        results.append(False)
        return None
    try:
        return func(*args)
    except Exception as e:
        report(label, False, f"{type(e).__name__}: {e}")
        return None


x = np.array([1.0, 2.0, 3.0])
y = np.array([3.0, 5.0, 7.0])          # these points lie exactly on y = 2x + 1

# 1 predict
got = run("1 predict", "predict", 2.0, 1.0, x)
if got is not None:
    report("1 predict(2, 1, x) gives [3, 5, 7]", np.allclose(got, [3, 5, 7]), f"got {got}")

# 2 mse
got = run("2 mse", "mse", 2.0, 1.0, x, y)
if got is not None:
    report("2 mse at the true line (w=2, b=1) is 0", np.isclose(got, 0.0), f"got {got}")
got = run("2 mse", "mse", 0.0, 0.0, x, y)
if got is not None:
    report("2 mse(0, 0) is 83/3 = 27.667 (a MEAN, not a sum)", np.isclose(got, 83 / 3), f"got {got}")

# 3 gradient
got = run("3 gradient", "gradient", 0.0, 0.0, x, y)
if got is not None:
    ok_shape = isinstance(got, tuple) and len(got) == 2
    report("3 gradient returns a tuple (dL/dw, dL/db)", ok_shape, f"got {type(got).__name__} {got}")
    if ok_shape:
        report("3 dL/dw at (0, 0) is -22.667", np.isclose(got[0], -68 / 3), f"got {got[0]}")
        report("3 dL/db at (0, 0) is -10", np.isclose(got[1], -10.0), f"got {got[1]}")
got = run("3 gradient at the minimum", "gradient", 2.0, 1.0, x, y)
if got is not None and isinstance(got, tuple) and len(got) == 2:
    report("3 gradient at the true line is (0, 0)", np.allclose(got, [0, 0]), f"got {got}")

# 4 numeric_gradient: the gradient check
got = run("4 numeric_gradient", "numeric_gradient", 1.0, 0.0, x, y)
if got is not None:
    report("4 numeric_gradient at (1, 0) is close to (-13.333, -6)", np.allclose(got, [-40 / 3, -6.0], atol=1e-4), f"got {got}")
if hasattr(linreg, "gradient") and hasattr(linreg, "numeric_gradient"):
    rng = np.random.default_rng(0)
    xs, ys = rng.normal(size=20), rng.normal(size=20)
    try:
        a, n = linreg.gradient(0.3, -0.7, xs, ys), linreg.numeric_gradient(0.3, -0.7, xs, ys)
        report("4 your formula agrees with the numerical check on random data", np.allclose(a, n, atol=1e-4), f"formula {a} vs numeric {n}")
    except Exception as e:
        report("4 formula vs numerical check", False, f"{type(e).__name__}: {e}")

# 5 one downhill step
if all(hasattr(linreg, f) for f in ("gradient", "mse")):
    try:
        dw, db = linreg.gradient(0.0, 0.0, x, y)
        w1, b1 = 0.0 - 0.05 * dw, 0.0 - 0.05 * db
        report("5 one step against the gradient lowers the loss (27.67 -> 5.49)", linreg.mse(w1, b1, x, y) < linreg.mse(0.0, 0.0, x, y), f"after the step: {linreg.mse(w1, b1, x, y)}")
    except Exception as e:
        report("5 one downhill step", False, f"{type(e).__name__}: {e}")

print(f"\n{sum(results)} of {len(results)} checks passed")