import numpy as np
import hello_uv.imagetools as imagetools

results = []


def report(label, ok, detail=""):
    print(("PASS     " if ok else "FAIL     ") + label + (f"  ({detail})" if detail and not ok else ""))
    results.append(ok)


def run(label, func_name, *args):
    func = getattr(imagetools, func_name, None)
    if func is None:
        print(f"MISSING  {func_name}() is not defined in imagetools.py")
        results.append(False)
        return None
    try:
        return func(*args)
    except Exception as e:
        report(label, False, f"{type(e).__name__}: {e}")
        return None


def same(got, want):
    return isinstance(got, np.ndarray) and got.shape == want.shape and np.array_equal(got, want)


u8 = lambda rows: np.array(rows, dtype=np.uint8)

# 1 invert
got = run("1 invert", "invert", u8([[0, 255], [10, 200]]))
if got is not None:
    report("1 invert turns 0 into 255 and 200 into 55", same(got, u8([[255, 0], [245, 55]])), f"got {got.tolist()}")

# 2 adjust_contrast
img = u8([[0, 100], [200, 100]])          # mean is 100
before = img.copy()
got = run("2 adjust_contrast", "adjust_contrast", img, 2.0)
if got is not None:
    report("2 contrast x2 stretches away from the mean and clips at 255", same(got, u8([[0, 100], [255, 100]])), f"got {got.tolist()}")
    report("2 the result is still uint8", got.dtype == np.uint8, f"got dtype {got.dtype}")
got = run("2 adjust_contrast 0.5", "adjust_contrast", img, 0.5)
if got is not None:
    report("2 contrast x0.5 squeezes towards the mean", same(got, u8([[50, 100], [150, 100]])), f"got {got.tolist()}")
if hasattr(imagetools, "adjust_contrast"):
    report("2 the original image is unchanged", np.array_equal(img, before))

# 3 shift
a = np.arange(9, dtype=np.uint8).reshape(3, 3)
before = a.copy()
got = run("3 shift right", "shift", a, 1, 0)
if got is not None:
    report("3 shift(a, 1, 0) moves right and fills the gap with 0", same(got, u8([[0, 0, 1], [0, 3, 4], [0, 6, 7]])), f"got {got.tolist()}")
got = run("3 shift left and down", "shift", a, -1, 1)
if got is not None:
    report("3 shift(a, -1, 1) moves left and down (nothing wraps round)", same(got, u8([[0, 0, 0], [1, 2, 0], [4, 5, 0]])), f"got {got.tolist()}")
got = run("3 shift by zero", "shift", a, 0, 0)
if got is not None:
    report("3 shift(a, 0, 0) changes nothing", same(got, a), f"got {got.tolist()}")
if hasattr(imagetools, "shift"):
    report("3 the original image is unchanged", np.array_equal(a, before))

# 4 rotate
rng = np.random.default_rng(0)
big = rng.integers(0, 256, size=(28, 28), dtype=np.uint8)
got = run("4 rotate 90", "rotate", big, 90)
if got is not None:
    report("4 rotate(img, 90) matches np.rot90 (anticlockwise)", same(got, np.rot90(big)))
got = run("4 rotate 15", "rotate", big, 15)
if got is not None:
    report("4 rotate(img, 15) keeps the 28 x 28 shape", np.shape(got) == (28, 28), f"got shape {np.shape(got)}")

print(f"\n{sum(results)} of {len(results)} checks passed")