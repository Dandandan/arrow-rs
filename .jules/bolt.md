## 2025-05-23 - [Inlined prefix short-circuiting for StringViewArray StartsWith]
**Learning:** StringViewArray (GenericByteViewArray) stores the first 4 bytes of data directly in the u128 view. For predicates like StartsWith, we can short-circuit the comparison by checking these 4 bytes before accessing the data buffers. This significantly reduces cache misses for mismatches.
**Action:** Always check if string operations on StringViewArray can benefit from the inlined 4-byte prefix or the 12-byte inlined short string values.
