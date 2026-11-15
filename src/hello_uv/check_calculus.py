import numpy as np
import hello_uv.calculus as calculus

results = []


def report(label, ok, detail=""):
    print(("PASS     " if ok else "FAIL     ") + label + (f"  ({detail})" if detail and not ok else ""))
    results.append(ok)


def run(label, func_name, *args, **kwargs):
    func = getattr(calculus, func_name, None)
    if func is None:
        print(f"MISSING  {func_name}() is not defined in calculus.py")
        results.append(False)
        return None
    try:
        return func(*args, **kwargs)
    except Exception as e:
        report(label, False, f"{type(e).__name__}: {e}")
        return None


def cube(x):
    return x ** 3          # exact slope: 3x^2, so 12 at x = 2


def square(x):
    return x ** 2          # exact slope: 2x


got = run("1 forward_diff", "forward_diff", cube, 2.0, h=0.1)
if got is not None:
    report("1 forward_diff(cube, 2, h=0.1) is 12.61", np.isclose(got, 12.61), f"got {got}")
got = run("1 forward_diff small h", "forward_diff", square, 3.0)
if got is not None:
    report("1 forward_diff(square, 3) with the default h is close to 6", abs(got - 6) < 1e-3, f"got {got}")

got = run("2 central_diff", "central_diff", cube, 2.0, h=0.1)
if got is not None:
    report("2 central_diff(cube, 2, h=0.1) is 12.01", np.isclose(got, 12.01), f"got {got}")
    f = calculus.forward_diff(cube, 2.0, h=0.1) if hasattr(calculus, "forward_diff") else None
    if f is not None:
        report("2 central is closer to the true 12 than forward", abs(got - 12) < abs(f - 12))

xs = np.linspace(-2, 2, 9)
got = run("3 slope_curve", "slope_curve", square, xs)
if got is not None:
    report("3 slope_curve gives one slope per x", np.shape(got) == (9,), f"got shape {np.shape(got)}")
    report("3 slope_curve of x^2 matches 2x", np.shape(got) == (9,) and np.allclose(got, 2 * xs, atol=1e-4))

got = run("4 flattest_x", "flattest_x", xs, 2 * xs)
if got is not None:
    report("4 flattest_x finds x = 0 for the slopes of x^2", np.isclose(got, 0.0), f"got {got}")
got = run("4 flattest_x again", "flattest_x", np.array([0.0, 1.0, 2.0, 3.0]), np.array([-4.0, -1.0, 0.5, 3.0]))
if got is not None:
    report("4 flattest_x uses the size of the slope, not its sign", np.isclose(got, 2.0), f"got {got}")

print(f"\n{sum(results)} of {len(results)} checks passed")