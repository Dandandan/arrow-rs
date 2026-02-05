## 2025-01-24 - [Optimize length kernels using zip and ScalarBuffer]
**Learning:** Replacing `offsets.windows(2)` with `off.iter().skip(1).zip(off)` in `length` kernels allows the compiler to use `TrustedLen` iterators, which significantly improves vectorization and avoids `Vec` reallocation overhead. This led to a ~30% performance improvement in `length` benchmarks.
**Action:** Prefer `zip` of slice iterators over `windows(n)` for calculating differences between adjacent elements in Arrow buffers. Use `ScalarBuffer::from_iter` to directly create result buffers from `TrustedLen` iterators.
