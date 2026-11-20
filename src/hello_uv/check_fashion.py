from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

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
    import hello_uv.fashion as fa
    ArrayDataset, get_fashion, make_loader, class_counts = fa.ArrayDataset, fa.get_fashion, fa.make_loader, fa.class_counts
except Exception as e:
    print(f"MISSING  ArrayDataset, get_fashion, make_loader or class_counts in src/hello_uv/fashion.py  ({type(e).__name__}: {e})")
    raise SystemExit("\n0 of 10 checks passed")

# a tiny made-up data set: 10 rows of 2 numbers, labels 0..9
x = np.arange(20, dtype=np.float64).reshape(10, 2)
y = np.arange(10)

# 1 the Dataset
ds = attempt("1 ArrayDataset(x, y) can be created", lambda: ArrayDataset(x, y))
if ds is not None:
    n = attempt("1 len(ds)", lambda: len(ds))
    if n is not None:
        report("1 ArrayDataset is a Dataset and len() gives 10 rows", isinstance(ds, Dataset) and n == 10, f"len {n}")

# 2 one item
item = attempt("2 ds[3]", lambda: ds[3]) if ds is not None else None
if item is not None:
    ok = isinstance(item, tuple) and len(item) == 2 and torch.is_tensor(item[0]) and item[0].dtype == torch.float32 \
        and torch.equal(item[0], torch.tensor([6.0, 7.0])) and int(item[1]) == 3
    report("2 ds[3] is (tensor([6., 7.]) as float32, label 3)", ok, f"got {item!r}")
else:
    report("2 ds[3] needs a working ArrayDataset", False)

# 3 batches in order
loader = attempt("3 make_loader runs", lambda: make_loader(ds, 4, False)) if ds is not None else None
if loader is not None:
    batches = attempt("3 looping over the loader", lambda: list(loader))
    if batches is not None:
        shapes = [tuple(b[0].shape) for b in batches]
        report("3 batch_size 4, no shuffle: batches of shape (4, 2), (4, 2), (2, 2), first labels 0-3",
               shapes == [(4, 2), (4, 2), (2, 2)] and batches[0][1].tolist() == [0, 1, 2, 3], f"shapes {shapes}")
else:
    report("3 make_loader needs a working ArrayDataset", False)

# 4 shuffling
if ds is not None:
    def epochs():
        torch.manual_seed(19)
        out = []
        for _ in range(5):
            out.append(torch.cat([lab for _, lab in make_loader(ds, 4, True)]).tolist())
        return out
    eps = attempt("4 shuffled loader", epochs)
    if eps is not None:
        every_once = all(sorted(e) == list(range(10)) for e in eps)
        mixed = any(e != list(range(10)) for e in eps)
        report("4 shuffle=True: each pass sees every row exactly once, in a new order", every_once and mixed, f"passes {eps[:2]}")
else:
    report("4 shuffling needs a working ArrayDataset", False)

# 5-7 FashionMNIST
print("         (loading FashionMNIST; the first run downloads about 30 MB)")
train = attempt("5 get_fashion(train=True)", lambda: get_fashion(train=True))
test = attempt("5 get_fashion(train=False)", lambda: get_fashion(train=False))
if train is not None and test is not None:
    report("5 60000 training images and 10000 test images", (len(train), len(test)) == (60000, 10000), f"got {len(train)}, {len(test)}")
    img, lab = train[0]
    ok = torch.is_tensor(img) and tuple(img.shape) == (1, 28, 28) and img.dtype == torch.float32 and 0 <= img.min() and img.max() <= 1
    report("6 an image is a float32 tensor of shape (1, 28, 28) with values from 0 to 1", ok,
           f"got {type(img).__name__}" + (f" {tuple(img.shape)} {img.dtype}" if torch.is_tensor(img) else " (did you forget the transform?)"))
    fl = attempt("7 make_loader on FashionMNIST", lambda: make_loader(train, 64, True))
    first = attempt("7 taking the first batch", lambda: next(iter(fl))) if fl is not None else None
    if first is not None:
        X, Y = first
        report("7 batch_size 64: X is (64, 1, 28, 28), Y is (64,) of whole numbers, 938 batches per pass",
               tuple(X.shape) == (64, 1, 28, 28) and tuple(Y.shape) == (64,) and Y.dtype == torch.int64 and len(fl) == 938,
               f"X {tuple(X.shape)}, Y {tuple(Y.shape)} {Y.dtype}, {len(fl)} batches")
else:
    report("6 needs get_fashion to work", False)
    report("7 needs get_fashion to work", False)

# 8-9 class_counts
if ds is not None:
    yy = np.array([0, 0, 0, 1, 1, 9, 9, 9, 9, 2])
    c = attempt("8 class_counts on made-up labels", lambda: class_counts(make_loader(ArrayDataset(x, yy), 3, False)))
    if c is not None:
        report("8 class_counts on labels 0,0,0,1,1,9,9,9,9,2 gives [3, 2, 1, 0, 0, 0, 0, 0, 0, 4]",
               list(c) == [3, 2, 1, 0, 0, 0, 0, 0, 0, 4], f"got {c}")
else:
    report("8 class_counts needs a working ArrayDataset", False)
if test is not None:
    c = attempt("9 class_counts on FashionMNIST", lambda: class_counts(make_loader(test, 1000, False)))
    if c is not None:
        report("9 the 10000 test images hold exactly 1000 of each of the 10 classes", list(c) == [1000] * 10, f"got {c}")
else:
    report("9 needs get_fashion to work", False)

# 10 the data stays out of git
gi = Path(".gitignore")
report("10 .gitignore keeps the data/ folder out of git", gi.exists() and any(l.strip().strip("/") == "data" for l in gi.read_text().splitlines()),
       "add a line: data/")

print(f"\n{sum(results)} of {len(results)} checks passed")