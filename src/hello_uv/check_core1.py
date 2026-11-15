import math
import hello_uv.core1 as core1

results = []

def check(label, func_name, args, want):
    func = getattr(core1, func_name, None)
    if func is None:
        print(f"MISSING  {func_name}() is not defined in core1.py")
        results.append(False)
        return
    try:
        got = func(*args)
    except Exception as e:
        print(f"ERROR    {label}: {type(e).__name__}: {e}")
        results.append(False)
        return
    ok = math.isclose(got, want) if isinstance(want, float) else got == want
    print(f"{'PASS' if ok else 'FAIL'}     {label} -> {got!r} (expected {want!r})")
    results.append(ok)

check("f(3)", "f", (3,), 17)
check("f(0)", "f", (0,), -1)
check("total([1, 2, 3, 4])", "total", ([1, 2, 3, 4],), 10)
check("product([1, 2, 3])", "product", ([1, 2, 3],), 6)
check("mean([2, 4, 6, 8])", "mean", ([2, 4, 6, 8],), 5.0)
check("sum_log([2, 4, 8])", "sum_log", ([2, 4, 8],), math.log(64))
print(f"\n{sum(results)} of {len(results)} checks passed")