## 2025-05-15 - Benchmark Noise and Thermal Throttling
**Learning:** Criterion benchmarks in this environment can show significant noise and a downward trend in performance (likely due to thermal throttling or shared resources), which can hide small optimizations or even appear as regressions. Establishing a fresh baseline for the unmodified code immediately before benchmarking the optimized code is crucial.
**Action:** Always establishment a fresh baseline if unexpected regressions appear in untouched code paths.

## 2025-05-15 - Unicode vs ASCII in Substring
**Learning:** `substring_by_char` was unconditionally counting characters even when it was redundant. Adding an `is_ascii` fast path yields a massive (~56%) speedup for ASCII-only strings because it avoids UTF-8 decoding entirely.
**Action:** Look for `is_ascii` fast-path opportunities in other Unicode-aware kernels.
