import numpy as np
import arrays

results = []


def check(label, func_name, args, want):
    func = getattr(arrays, func_name, None)
    if func is None:
        print(f"MISSING  {func_name}() is not defined in arrays.py")
        results.append(False)
        return
    try:
        got = func(*args)
    except Exception as e:
        print(f"ERROR    {label}: {type(e).__name__}: {e}")
        results.append(False)
        return
    if isinstance(want, np.ndarray):
        got_arr = np.asarray(got)
        ok = got_arr.shape == want.shape and np.allclose(got_arr, want)
    elif isinstance(want, float):
        ok = isinstance(got, (int, float, np.floating)) and abs(float(got) - want) < 1e-9
    else:
        ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}     {label} -> {np.asarray(got).tolist() if hasattr(got, 'shape') else got} (expected {want.tolist() if isinstance(want, np.ndarray) else want})")
    results.append(ok)


toy = np.arange(16).reshape(4, 4)   # rows: [0 1 2 3], [4 5 6 7], [8 9 10 11], [12 13 14 15]

check("crop_center(toy, 2)", "crop_center", (toy, 2), np.array([[5, 6], [9, 10]]))
check("flip_horizontal(toy)[0]", "flip_horizontal", (toy[:1],), np.array([[3, 2, 1, 0]]))
check("count_bright(toy, 10)", "count_bright", (toy, 10), 5)
check("row_means(toy)", "row_means", (toy,), np.array([1.5, 5.5, 9.5, 13.5]))
check("normalise(uint8 [0, 51, 255])", "normalise", (np.array([0, 51, 255], dtype=np.uint8),), np.array([0.0, 0.2, 1.0]))
check("cosine([1, 0], [0, 1])", "cosine", (np.array([1, 0]), np.array([0, 1])), 0.0)
check("cosine([1, 2], [2, 4])", "cosine", (np.array([1, 2]), np.array([2, 4])), 1.0)
check("cosine([1, 0], [-3, 0])", "cosine", (np.array([1, 0]), np.array([-3, 0])), -1.0)

print(f"\n{sum(results)} of {len(results)} checks passed")