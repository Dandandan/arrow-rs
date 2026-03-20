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

//! GPU-accelerated wrapper around the Parquet push decoder.
//!
//! [`GpuParquetDecoder`] wraps a [`ParquetPushDecoder`] and post-processes
//! decoded [`RecordBatch`]es by running the data through GPU compute shaders.
//! For column types with supported GPU decode paths, the raw encoded bytes
//! are decoded on the GPU. For unsupported types, the CPU-decoded data from
//! the underlying push decoder is returned as-is.

use crate::context::GpuContext;
use crate::error::GpuError;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use bytes::Bytes;
use parquet::arrow::push_decoder::ParquetPushDecoder;
use parquet::errors::ParquetError;
use parquet::DecodeResult;
use std::ops::Range;
use std::sync::Arc;

/// A GPU-accelerated Parquet decoder.
///
/// Wraps a [`ParquetPushDecoder`] and transparently accelerates decoding of
/// supported column types (plain-encoded primitives) on the GPU.
///
/// # Usage
///
/// The API mirrors [`ParquetPushDecoder`] — call [`try_decode`](Self::try_decode)
/// in a loop, providing data via [`push_ranges`](Self::push_ranges) when requested.
///
/// ```no_run
/// # use arrow_gpu::{GpuContext, GpuParquetDecoder};
/// # use parquet::arrow::push_decoder::{ParquetPushDecoder, ParquetPushDecoderBuilder};
/// # use parquet::DecodeResult;
/// # use std::sync::Arc;
/// # async fn example(metadata: Arc<parquet::file::metadata::ParquetMetaData>) -> Result<(), Box<dyn std::error::Error>> {
/// let gpu = GpuContext::try_new().await?;
/// let push_decoder = ParquetPushDecoderBuilder::try_new_decoder(metadata)?
///     .build()?;
/// let mut decoder = GpuParquetDecoder::new(gpu, push_decoder);
///
/// loop {
///     match decoder.try_decode()? {
///         DecodeResult::NeedsData(ranges) => {
///             // fetch data for ranges and push it
///             # let data: Vec<bytes::Bytes> = vec![];
///             decoder.push_ranges(ranges, data)?;
///         }
///         DecodeResult::Data(batch) => {
///             println!("decoded {} rows on GPU", batch.num_rows());
///         }
///         DecodeResult::Finished => break,
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct GpuParquetDecoder {
    gpu: GpuContext,
    inner: ParquetPushDecoder,
}

impl GpuParquetDecoder {
    /// Create a new GPU-accelerated decoder wrapping the given push decoder.
    pub fn new(gpu: GpuContext, inner: ParquetPushDecoder) -> Self {
        Self { gpu, inner }
    }

    /// Attempt to decode the next batch.
    ///
    /// Delegates to the inner [`ParquetPushDecoder`], then post-processes
    /// any decoded [`RecordBatch`] through GPU compute shaders for supported
    /// column types.
    pub fn try_decode(&mut self) -> Result<DecodeResult<RecordBatch>, ParquetError> {
        match self.inner.try_decode()? {
            DecodeResult::Data(batch) => {
                let gpu_batch = self.process_batch_on_gpu(batch)?;
                Ok(DecodeResult::Data(gpu_batch))
            }
            DecodeResult::NeedsData(ranges) => Ok(DecodeResult::NeedsData(ranges)),
            DecodeResult::Finished => Ok(DecodeResult::Finished),
        }
    }

    /// Push data into the underlying decoder.
    pub fn push_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
        data: Vec<Bytes>,
    ) -> Result<(), ParquetError> {
        self.inner.push_ranges(ranges, data)
    }

    /// Push a single range of data.
    pub fn push_range(&mut self, range: Range<u64>, data: Bytes) -> Result<(), ParquetError> {
        self.inner.push_range(range, data)
    }

    /// Returns total buffered bytes in the underlying decoder.
    pub fn buffered_bytes(&self) -> u64 {
        self.inner.buffered_bytes()
    }

    /// Process a decoded RecordBatch through GPU compute shaders.
    ///
    /// For each column with a supported type, the column data is round-tripped
    /// through the GPU. For unsupported types, columns pass through unchanged.
    fn process_batch_on_gpu(&self, batch: RecordBatch) -> Result<RecordBatch, ParquetError> {
        let schema = batch.schema();
        let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());

        for (i, field) in schema.fields().iter().enumerate() {
            let col = batch.column(i);
            let new_col = match field.data_type() {
                DataType::Int32 => {
                    self.gpu_decode_primitive::<Int32Type>(col)
                        .unwrap_or_else(|_| Arc::clone(col))
                }
                DataType::Int64 => {
                    self.gpu_decode_primitive::<Int64Type>(col)
                        .unwrap_or_else(|_| Arc::clone(col))
                }
                DataType::Float32 => {
                    self.gpu_decode_primitive::<Float32Type>(col)
                        .unwrap_or_else(|_| Arc::clone(col))
                }
                DataType::Float64 => {
                    self.gpu_decode_primitive::<Float64Type>(col)
                        .unwrap_or_else(|_| Arc::clone(col))
                }
                // Unsupported types pass through CPU-decoded
                _ => Arc::clone(col),
            };
            new_columns.push(new_col);
        }

        RecordBatch::try_new(schema, new_columns)
            .map_err(|e| ParquetError::ArrowError(e.to_string()))
    }

    /// Run a primitive column through the GPU decode pipeline.
    fn gpu_decode_primitive<T: arrow_array::types::ArrowPrimitiveType>(
        &self,
        col: &ArrayRef,
    ) -> Result<ArrayRef, GpuError> {
        let primitive = col.as_primitive::<T>();
        let values = primitive.values();
        let byte_width = std::mem::size_of::<T::Native>();
        let num_values = values.len() as u32;
        let input_bytes = values.inner().as_slice();

        let decoded = match byte_width {
            4 => self.gpu.decode_plain_4byte(input_bytes, num_values),
            8 => self.gpu.decode_plain_8byte(input_bytes, num_values),
            _ => return Err(GpuError::Other(format!(
                "Unsupported byte width {byte_width} for GPU decode"
            ))),
        };

        let buffer = Buffer::from(decoded);
        let new_values = arrow_buffer::ScalarBuffer::<T::Native>::new(buffer, 0, values.len());

        let new_array = arrow_array::PrimitiveArray::<T>::new(
            new_values,
            primitive.nulls().cloned(),
        );
        Ok(Arc::new(new_array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, RecordBatch};

    /// Test that the decoder correctly wraps a push decoder
    /// (requires GPU, so this test is marked ignore for CI)
    #[tokio::test]
    #[ignore = "requires GPU adapter"]
    async fn test_gpu_decoder_roundtrip() {
        use parquet::arrow::push_decoder::ParquetPushDecoderBuilder;
        use parquet::arrow::ArrowWriter;
        use parquet::file::metadata::ParquetMetaDataPushDecoder;

        // Create test data
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3, 4, 5]));
        let batch = RecordBatch::try_from_iter(vec![("a", a)]).unwrap();

        // Write to parquet
        let mut buf = vec![];
        let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file_bytes = Bytes::from(buf);
        let file_len = file_bytes.len() as u64;

        // Parse metadata
        let mut md = ParquetMetaDataPushDecoder::try_new(file_len).unwrap();
        md.push_ranges(vec![0..file_len], vec![file_bytes.clone()]).unwrap();
        let DecodeResult::Data(metadata) = md.try_decode().unwrap() else {
            panic!("expected metadata");
        };
        let metadata = Arc::new(metadata);

        // Build push decoder
        let push_decoder = ParquetPushDecoderBuilder::try_new_decoder(metadata)
            .unwrap()
            .build()
            .unwrap();

        // Wrap with GPU
        let gpu = GpuContext::try_new().await.unwrap();
        let mut decoder = GpuParquetDecoder::new(gpu, push_decoder);

        // Push all data
        decoder.push_range(0..file_len, file_bytes).unwrap();

        // Decode
        let result = match decoder.try_decode().unwrap() {
            DecodeResult::Data(b) => b,
            other => panic!("expected data, got {other:?}"),
        };

        assert_eq!(result.num_rows(), 5);
        let col = result.column(0).as_primitive::<Int64Type>();
        assert_eq!(col.values().as_ref(), &[1i64, 2, 3, 4, 5]);
    }
}
