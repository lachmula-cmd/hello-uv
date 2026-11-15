import numpy as np
import layers

results = []


def check(label, compute, want):
    try:
        got = compute()
    except Exception as e:
        print(f"ERROR    {label}: {type(e).__name__}: {e}")
        results.append(False)
        return
    if isinstance(want, np.ndarray):
        got_arr = np.asarray(got, dtype=float)
        ok = got_arr.shape == want.shape and np.allclose(got_arr, want)
        shown = got_arr.tolist()
        want_shown = want.tolist()
    else:
        ok = got == want
        shown = got
        want_shown = want
    print(f"{'PASS' if ok else 'FAIL'}     {label} -> {shown} (expected {want_shown})")
    results.append(ok)


# Made-up toy numbers: not the numbers in section 6.
W = [[1, 0, 2], [0, -1, 1]]
b = [1, -5]
x = np.array([1, 2, 3])

check("Module().training", lambda: layers.Module().training, True)
check("Module().num_params()", lambda: layers.Module().num_params(), 0)
check("Linear(W, b).W is float", lambda: layers.Linear(W, b).W.dtype == float, True)
check("Linear(W, b).training", lambda: layers.Linear(W, b).training, True)
check("Linear(W, b).forward(x)", lambda: layers.Linear(W, b).forward(x), np.array([8.0, -4.0]))
check("Linear(W, b)(x)  (calling the object)", lambda: layers.Linear(W, b)(x), np.array([8.0, -4.0]))
check("Linear(W, b).num_params()", lambda: layers.Linear(W, b).num_params(), 8)
check("ReLU()([-2, 0, 3])", lambda: layers.ReLU()(np.array([-2.0, 0.0, 3.0])), np.array([0.0, 0.0, 3.0]))
check("ReLU().num_params()", lambda: layers.ReLU().num_params(), 0)
check("isinstance(ReLU(), Module)", lambda: isinstance(layers.ReLU(), layers.Module), True)


def seq():
    return layers.Sequential([layers.Linear(W, b), layers.ReLU(), layers.Linear([[1, 1]], [-5])])


check("Sequential(...)(x)", lambda: seq()(x), np.array([3.0]))
check("Sequential(...).num_params()", lambda: seq().num_params(), 11)


def after_eval():
    s = seq()
    s.eval()
    return [s.training] + [layer.training for layer in s.layers]


check("after .eval(): model and every layer", after_eval, [False, False, False, False])

print(f"\n{sum(results)} of {len(results)} checks passed")