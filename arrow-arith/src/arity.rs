// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Kernels for operating on [`PrimitiveArray`]s

use arrow_array::builder::BufferBuilder;
use arrow_array::*;
use arrow_buffer::buffer::NullBuffer;
use arrow_buffer::ArrowNativeType;
use arrow_buffer::MutableBuffer;
use arrow_data::ArrayData;
use arrow_schema::ArrowError;

/// See [`PrimitiveArray::unary`]
pub fn unary<I, F, O>(array: &PrimitiveArray<I>, op: F) -> PrimitiveArray<O>
where
    I: ArrowPrimitiveType,
    O: ArrowPrimitiveType,
    F: Fn(I::Native) -> O::Native,
{
    array.unary(op)
}

/// See [`PrimitiveArray::unary_mut`]
pub fn unary_mut<I, F>(
    array: PrimitiveArray<I>,
    op: F,
) -> Result<PrimitiveArray<I>, PrimitiveArray<I>>
where
    I: ArrowPrimitiveType,
    F: Fn(I::Native) -> I::Native,
{
    array.unary_mut(op)
}

/// See [`PrimitiveArray::try_unary`]
pub fn try_unary<I, F, O>(array: &PrimitiveArray<I>, op: F) -> Result<PrimitiveArray<O>, ArrowError>
where
    I: ArrowPrimitiveType,
    O: ArrowPrimitiveType,
    F: Fn(I::Native) -> Result<O::Native, ArrowError>,
{
    array.try_unary(op)
}

/// See [`PrimitiveArray::try_unary_mut`]
pub fn try_unary_mut<I, F>(
    array: PrimitiveArray<I>,
    op: F,
) -> Result<Result<PrimitiveArray<I>, ArrowError>, PrimitiveArray<I>>
where
    I: ArrowPrimitiveType,
    F: Fn(I::Native) -> Result<I::Native, ArrowError>,
{
    array.try_unary_mut(op)
}

/// Allies a binary infallable function to two [`PrimitiveArray`]s,
/// producing a new [`PrimitiveArray`]
///
/// # Details
///
/// Given two arrays of length `len`, calls `op(a[i], b[i])` for `i` in `0..len`, collecting
/// the results in a [`PrimitiveArray`].
///
/// If any index is null in either `a` or `b`, the
/// corresponding index in the result will also be null
///
/// Like [`unary`], the `op` is evaluated for every element in the two arrays,
/// including those elements which are NULL. This is beneficial as the cost of
/// the operation is low compared to the cost of branching, and especially when
/// the operation can be vectorised, however, requires `op` to be infallible for
/// all possible values of its inputs
///
/// # Errors
///
/// * if the arrays have different lengths.
///
/// # Example
/// ```
/// # use arrow_arith::arity::binary;
/// # use arrow_array::{Float32Array, Int32Array};
/// # use arrow_array::types::Int32Type;
/// let a = Float32Array::from(vec![Some(5.1f32), None, Some(6.8), Some(7.2)]);
/// let b = Int32Array::from(vec![1, 2, 4, 9]);
/// // compute int(a) + b for each element
/// let c = binary(&a, &b, |a, b| a as i32 + b).unwrap();
/// assert_eq!(c, Int32Array::from(vec![Some(6), None, Some(10), Some(16)]));
/// ```
pub fn binary<A, B, F, O>(
    a: &PrimitiveArray<A>,
    b: &PrimitiveArray<B>,
    op: F,
) -> Result<PrimitiveArray<O>, ArrowError>
where
    A: ArrowPrimitiveType,
    B: ArrowPrimitiveType,
    O: ArrowPrimitiveType,
    F: Fn(A::Native, B::Native) -> O::Native,
{
    if a.len() != b.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform binary operation on arrays of different length".to_string(),
        ));
    }

    if a.is_empty() {
        return Ok(PrimitiveArray::from(ArrayData::new_empty(&O::DATA_TYPE)));
    }

    let nulls = NullBuffer::union(a.logical_nulls().as_ref(), b.logical_nulls().as_ref());

    let values = a
        .values()
        .into_iter()
        .zip(b.values())
        .map(|(l, r)| op(*l, *r));

    let buffer: Vec<_> = values.collect();
    Ok(PrimitiveArray::new(buffer.into(), nulls))
}

/// Applies a binary and infallible function to values in two arrays, replacing
/// the values in the first array in place.
///
/// # Details
///
/// Given two arrays of length `len`, calls `op(a[i], b[i])` for `i` in
/// `0..len`, modifying the [`PrimitiveArray`] `a` in place, if possible.
///
/// If any index is null in either `a` or `b`, the corresponding index in the
/// result will also be null.
///
/// # Buffer Reuse
///
/// If the underlying buffers in `a` are not shared with other arrays,  mutates
/// the underlying buffer in place, without allocating.
///
/// If the underlying buffer in `a` are shared, returns Err(self)
///
/// Like [`unary`] the provided function is evaluated for every index, ignoring validity. This
/// is beneficial when the cost of the operation is low compared to the cost of branching, and
/// especially when the operation can be vectorised, however, requires `op` to be infallible
/// for all possible values of its inputs
///
/// # Errors
///
/// * If the arrays have different lengths
/// * If the array is not mutable (see "Buffer Reuse")
///
/// # See Also
///
/// * Documentation on [`PrimitiveArray::unary_mut`] for operating on [`ArrayRef`].
///
/// # Example
/// ```
/// # use arrow_arith::arity::binary_mut;
/// # use arrow_array::{Float32Array, Int32Array};
/// # use arrow_array::types::Int32Type;
/// // compute a + b for each element
/// let a = Float32Array::from(vec![Some(5.1f32), None, Some(6.8)]);
/// let b = Int32Array::from(vec![Some(1), None, Some(2)]);
/// // compute a + b, updating the value in a in place if possible
/// let a = binary_mut(a, &b, |a, b| a + b as f32).unwrap().unwrap();
/// // a is updated in place
/// assert_eq!(a, Float32Array::from(vec![Some(6.1), None, Some(8.8)]));
/// ```
///
/// # Example with shared buffers
/// ```
/// # use arrow_arith::arity::binary_mut;
/// # use arrow_array::Float32Array;
/// # use arrow_array::types::Int32Type;
/// let a = Float32Array::from(vec![Some(5.1f32), None, Some(6.8)]);
/// let b = Float32Array::from(vec![Some(1.0f32), None, Some(2.0)]);
/// // a_clone shares the buffer with a
/// let a_cloned = a.clone();
/// // try to update a in place, but it is shared. Returns Err(a)
/// let a = binary_mut(a, &b, |a, b| a + b).unwrap_err();
/// assert_eq!(a_cloned, a);
/// // drop shared reference
/// drop(a_cloned);
/// // now a is not shared, so we can update it in place
/// let a = binary_mut(a, &b, |a, b| a + b).unwrap().unwrap();
/// assert_eq!(a, Float32Array::from(vec![Some(6.1), None, Some(8.8)]));
/// ```
pub fn binary_mut<T, U, F>(
    a: PrimitiveArray<T>,
    b: &PrimitiveArray<U>,
    op: F,
) -> Result<Result<PrimitiveArray<T>, ArrowError>, PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    U: ArrowPrimitiveType,
    F: Fn(T::Native, U::Native) -> T::Native,
{
    if a.len() != b.len() {
        return Ok(Err(ArrowError::ComputeError(
            "Cannot perform binary operation on arrays of different length".to_string(),
        )));
    }

    if a.is_empty() {
        return Ok(Ok(PrimitiveArray::from(ArrayData::new_empty(
            &T::DATA_TYPE,
        ))));
    }

    let mut builder = a.into_builder()?;

    builder
        .values_slice_mut()
        .iter_mut()
        .zip(b.values())
        .for_each(|(l, r)| *l = op(*l, *r));

    let array = builder.finish();

    // The builder has the null buffer from `a`, it is not changed.
    let nulls = NullBuffer::union(array.logical_nulls().as_ref(), b.logical_nulls().as_ref());

    let array_builder = array.into_data().into_builder().nulls(nulls);

    let array_data = unsafe { array_builder.build_unchecked() };
    Ok(Ok(PrimitiveArray::<T>::from(array_data)))
}

/// Applies the provided fallible binary operation across `a` and `b`.
///
/// This will return any error encountered, or collect the results into
/// a [`PrimitiveArray`]. If any index is null in either `a`
/// or `b`, the corresponding index in the result will also be null
///
/// Like [`try_unary`] the function is only evaluated for non-null indices
///
/// # Error
///
/// Return an error if the arrays have different lengths or
/// the operation is under erroneous
pub fn try_binary<A: ArrayAccessor, B: ArrayAccessor, F, O>(
    a: A,
    b: B,
    op: F,
) -> Result<PrimitiveArray<O>, ArrowError>
where
    O: ArrowPrimitiveType,
    F: Fn(A::Item, B::Item) -> Result<O::Native, ArrowError>,
{
    if a.len() != b.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform a binary operation on arrays of different length".to_string(),
        ));
    }
    if a.is_empty() {
        return Ok(PrimitiveArray::from(ArrayData::new_empty(&O::DATA_TYPE)));
    }
    let len = a.len();

    if a.null_count() == 0 && b.null_count() == 0 {
        try_binary_no_nulls(len, a, b, op)
    } else {
        let nulls =
            NullBuffer::union(a.logical_nulls().as_ref(), b.logical_nulls().as_ref()).unwrap();

        let mut buffer = BufferBuilder::<O::Native>::new(len);
        buffer.append_n_zeroed(len);
        let slice = buffer.as_slice_mut();

        nulls.try_for_each_valid_idx(|idx| {
            unsafe {
                *slice.get_unchecked_mut(idx) = op(a.value_unchecked(idx), b.value_unchecked(idx))?
            };
            Ok::<_, ArrowError>(())
        })?;

        let values = buffer.finish().into();
        Ok(PrimitiveArray::new(values, Some(nulls)))
    }
}

/// Applies the provided fallible binary operation across `a` and `b` by mutating the mutable
/// [`PrimitiveArray`] `a` with the results.
///
/// Returns any error encountered, or collects the results into a [`PrimitiveArray`] as return
/// value. If any index is null in either `a` or `b`, the corresponding index in the result will
/// also be null.
///
/// Like [`try_unary`] the function is only evaluated for non-null indices.
///
/// See [`binary_mut`] for errors and buffer reuse information.
pub fn try_binary_mut<T, F>(
    a: PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
    op: F,
) -> Result<Result<PrimitiveArray<T>, ArrowError>, PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    F: Fn(T::Native, T::Native) -> Result<T::Native, ArrowError>,
{
    if a.len() != b.len() {
        return Ok(Err(ArrowError::ComputeError(
            "Cannot perform binary operation on arrays of different length".to_string(),
        )));
    }
    let len = a.len();

    if a.is_empty() {
        return Ok(Ok(PrimitiveArray::from(ArrayData::new_empty(
            &T::DATA_TYPE,
        ))));
    }

    if a.null_count() == 0 && b.null_count() == 0 {
        try_binary_no_nulls_mut(len, a, b, op)
    } else {
        let nulls =
            create_union_null_buffer(a.logical_nulls().as_ref(), b.logical_nulls().as_ref())
                .unwrap();

        let mut builder = a.into_builder()?;

        let slice = builder.values_slice_mut();

        let r = nulls.try_for_each_valid_idx(|idx| {
            unsafe {
                *slice.get_unchecked_mut(idx) =
                    op(*slice.get_unchecked(idx), b.value_unchecked(idx))?
            };
            Ok::<_, ArrowError>(())
        });
        if let Err(err) = r {
            return Ok(Err(err));
        }
        let array_builder = builder.finish().into_data().into_builder();
        let array_data = unsafe { array_builder.nulls(Some(nulls)).build_unchecked() };
        Ok(Ok(PrimitiveArray::<T>::from(array_data)))
    }
}

/// Computes the union of the nulls in two optional [`NullBuffer`] which
/// is not shared with the input buffers.
///
/// The union of the nulls is the same as `NullBuffer::union(lhs, rhs)` but
/// it does not increase the reference count of the null buffer.
fn create_union_null_buffer(
    lhs: Option<&NullBuffer>,
    rhs: Option<&NullBuffer>,
) -> Option<NullBuffer> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(NullBuffer::new(lhs.inner() & rhs.inner())),
        (Some(n), None) | (None, Some(n)) => Some(NullBuffer::new(n.inner() & n.inner())),
        (None, None) => None,
    }
}

/// This intentional inline(never) attribute helps LLVM optimize the loop.
#[inline(never)]
fn try_binary_no_nulls<A: ArrayAccessor, B: ArrayAccessor, F, O>(
    len: usize,
    a: A,
    b: B,
    op: F,
) -> Result<PrimitiveArray<O>, ArrowError>
where
    O: ArrowPrimitiveType,
    F: Fn(A::Item, B::Item) -> Result<O::Native, ArrowError>,
{
    let mut buffer = MutableBuffer::new(len * O::Native::get_byte_width());
    for idx in 0..len {
        unsafe {
            buffer.push_unchecked(op(a.value_unchecked(idx), b.value_unchecked(idx))?);
        };
    }
    Ok(PrimitiveArray::new(buffer.into(), None))
}

/// This intentional inline(never) attribute helps LLVM optimize the loop.
#[inline(never)]
fn try_binary_no_nulls_mut<T, F>(
    len: usize,
    a: PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
    op: F,
) -> Result<Result<PrimitiveArray<T>, ArrowError>, PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    F: Fn(T::Native, T::Native) -> Result<T::Native, ArrowError>,
{
    let mut builder = a.into_builder()?;
    let slice = builder.values_slice_mut();

    for idx in 0..len {
        unsafe {
            match op(*slice.get_unchecked(idx), b.value_unchecked(idx)) {
                Ok(value) => *slice.get_unchecked_mut(idx) = value,
                Err(err) => return Ok(Err(err)),
            };
        };
    }
    Ok(Ok(builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::*;
    use std::sync::Arc;

    #[test]
    #[allow(deprecated)]
    fn test_unary_f64_slice() {
        let input = Float64Array::from(vec![Some(5.1f64), None, Some(6.8), None, Some(7.2)]);
        let input_slice = input.slice(1, 4);
        let result = unary(&input_slice, |n| n.round());
        assert_eq!(
            result,
            Float64Array::from(vec![None, Some(7.0), None, Some(7.0)])
        );
    }

    #[test]
    fn test_binary_mut() {
        let a = Int32Array::from(vec![15, 14, 9, 8, 1]);
        let b = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let c = binary_mut(a, &b, |l, r| l + r).unwrap().unwrap();

        let expected = Int32Array::from(vec![Some(16), None, Some(12), None, Some(6)]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_binary_mut_null_buffer() {
        let a = Int32Array::from(vec![Some(3), Some(4), Some(5), Some(6), None]);

        let b = Int32Array::from(vec![Some(10), Some(11), Some(12), Some(13), Some(14)]);

        let r1 = binary_mut(a, &b, |a, b| a + b).unwrap();

        let a = Int32Array::from(vec![Some(3), Some(4), Some(5), Some(6), None]);
        let b = Int32Array::new(
            vec![10, 11, 12, 13, 14].into(),
            Some(vec![true, true, true, true, true].into()),
        );

        // unwrap here means that no copying occured
        let r2 = binary_mut(a, &b, |a, b| a + b).unwrap();
        assert_eq!(r1.unwrap(), r2.unwrap());
    }

    #[test]
    fn test_try_binary_mut() {
        let a = Int32Array::from(vec![15, 14, 9, 8, 1]);
        let b = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let c = try_binary_mut(a, &b, |l, r| Ok(l + r)).unwrap().unwrap();

        let expected = Int32Array::from(vec![Some(16), None, Some(12), None, Some(6)]);
        assert_eq!(c, expected);

        let a = Int32Array::from(vec![15, 14, 9, 8, 1]);
        let b = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let c = try_binary_mut(a, &b, |l, r| Ok(l + r)).unwrap().unwrap();
        let expected = Int32Array::from(vec![16, 16, 12, 12, 6]);
        assert_eq!(c, expected);

        let a = Int32Array::from(vec![15, 14, 9, 8, 1]);
        let b = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let _ = try_binary_mut(a, &b, |l, r| {
            if l == 1 {
                Err(ArrowError::InvalidArgumentError(
                    "got error".parse().unwrap(),
                ))
            } else {
                Ok(l + r)
            }
        })
        .unwrap()
        .expect_err("should got error");
    }

    #[test]
    fn test_try_binary_mut_null_buffer() {
        let a = Int32Array::from(vec![Some(3), Some(4), Some(5), Some(6), None]);

        let b = Int32Array::from(vec![Some(10), Some(11), Some(12), Some(13), Some(14)]);

        let r1 = try_binary_mut(a, &b, |a, b| Ok(a + b)).unwrap();

        let a = Int32Array::from(vec![Some(3), Some(4), Some(5), Some(6), None]);
        let b = Int32Array::new(
            vec![10, 11, 12, 13, 14].into(),
            Some(vec![true, true, true, true, true].into()),
        );

        // unwrap here means that no copying occured
        let r2 = try_binary_mut(a, &b, |a, b| Ok(a + b)).unwrap();
        assert_eq!(r1.unwrap(), r2.unwrap());
    }

    #[test]
    fn test_unary_dict_mut() {
        let values = Int32Array::from(vec![Some(10), Some(20), None]);
        let keys = Int8Array::from_iter_values([0, 0, 1, 2]);
        let dictionary = DictionaryArray::new(keys, Arc::new(values));

        let updated = dictionary.unary_mut::<_, Int32Type>(|x| x + 1).unwrap();
        let typed = updated.downcast_dict::<Int32Array>().unwrap();
        assert_eq!(typed.value(0), 11);
        assert_eq!(typed.value(1), 11);
        assert_eq!(typed.value(2), 21);

        let values = updated.values();
        assert!(values.is_null(2));
    }
}
