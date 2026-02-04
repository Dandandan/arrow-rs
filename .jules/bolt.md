## 2025-01-24 - Specializing Predicates for StringViewArray
**Learning:** Specializing `Predicate::evaluate_array` for `StringViewArray` by manually iterating over `views()` and leveraging bitwise masks for prefixes ≤ 4 bytes or length/prefix short-circuits for longer strings significantly improves performance for `Eq`, `StartsWith`, `Contains`, and `IEqAscii` (e.g., ~51% speedup for equality). This avoids the overhead of materializing `&str` objects required by `BooleanArray::from_unary`.
**Action:** Always check if a generic kernel can be specialized for `ByteViewArray` by leveraging its inlined length and prefix.

## 2025-01-24 - Safe bitmask calculation for short prefixes
**Learning:** When calculating a bitmask for short string prefixes (0-4 bytes) from a `u32` view in `StringViewArray`, avoid `0xFFFFFFFF >> (8 * (4 - len))` because it panics on overflow when `len` is 0 in Rust.
**Action:** Use `if len >= 4 { 0xFFFFFFFFu32 } else { (1u32 << (len * 8)) - 1 }` as a safe alternative for prefix masking.
