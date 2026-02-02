## 2025-05-15 - [BooleanBuffer::from_iter regression]
**Learning:** Contrary to initial assumptions, `BooleanBuffer::from_iter(iter)` can be significantly slower (~30% regression) than `BooleanBuffer::from(iter.collect::<Vec<bool>>())` for simple predicates like `StartsWith`. The `From<Vec<bool>>` implementation likely utilizes more optimized bit-packing routines (potentially SIMD) than the generic iterator implementation for boolean values.
**Action:** Always benchmark `BooleanBuffer::from_iter` against `collect::<Vec<bool>>().into()` when optimizing Arrow kernels. Don't assume avoiding the intermediate `Vec<bool>` allocation is always faster.

## 2025-05-15 - [Redundant chars().count() in substring_by_char]
**Learning:** Unconditionally calculating `val.chars().count()` in Unicode-aware kernels is a performance anti-pattern if the count is only needed for specific cases (like negative indices).
**Action:** Move `chars().count()` into the conditional block where it's actually required. This simple change yielded a ~12% improvement in the substring benchmark.
