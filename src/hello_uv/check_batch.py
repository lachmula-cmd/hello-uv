import numpy as np
import hello_uv.batch as batch

results = []


def check(label, func_name, args, want):
    func = getattr(batch, func_name, None)
    if func is None:
        print(f"MISSING  {func_name}() is not defined in batch.py")
        results.append(False)
        return
    try:
        got = np.asarray(func(*args))
    except Exception as e:
        print(f"ERROR    {label}: {type(e).__name__}: {e}")
        results.append(False)
        return
    want = np.asarray(want)
    ok = got.shape == want.shape and np.allclose(got, want)
    print(f"{'PASS' if ok else 'FAIL'}     {label} -> shape {got.shape}, {np.round(got, 3).tolist()} (expected shape {want.shape}, {np.round(want, 3).tolist()})")
    results.append(ok)


imgs = np.arange(12).reshape(3, 2, 2)          # 3 tiny 2x2 "images"
X = np.array([[3.0, 4.0], [0.0, 2.0], [6.0, 8.0], [-3.0, -4.0]])

check("flatten_images(3 tiny images)", "flatten_images", (imgs,), np.arange(12, dtype=float).reshape(3, 4))
check("normalise_rows(X)", "normalise_rows", (X,), [[0.6, 0.8], [0.0, 1.0], [0.6, 0.8], [-0.6, -0.8]])
check("cosine_to_all(X, 0)", "cosine_to_all", (X, 0), [1.0, 0.8, 1.0, -1.0])
check("top_k_similar(X, 0, 2)", "top_k_similar", (X, 0, 2), [2, 1])
check("centre_columns([[1, 2], [3, 6]])", "centre_columns", (np.array([[1.0, 2.0], [3.0, 6.0]]),), [[-1.0, -2.0], [1.0, 2.0]])

print(f"\n{sum(results)} of {len(results)} checks passed")