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

//! Example: GPU-accelerated Parquet decoding
//!
//! This example demonstrates how to use `arrow-gpu` to decode Parquet data
//! with GPU acceleration. It:
//! 1. Creates an in-memory Parquet file with Int64 and Float64 columns
//! 2. Initializes the GPU context
//! 3. Decodes the Parquet data using the GPU-accelerated push decoder
//!
//! Run with: cargo run --example gpu_parquet_decode

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_gpu::{GpuContext, GpuParquetDecoder};
use bytes::Bytes;
use parquet::arrow::push_decoder::ParquetPushDecoderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::ParquetMetaDataPushDecoder;
use parquet::DecodeResult;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create test data: 10_000 rows of Int64 and Float64
    let num_rows = 10_000i64;
    let col_a: ArrayRef = Arc::new(Int64Array::from_iter_values(0..num_rows));
    let col_b: ArrayRef = Arc::new(Float64Array::from_iter_values(
        (0..num_rows).map(|i| i as f64 * 1.5),
    ));
    let batch = RecordBatch::try_from_iter(vec![("a", col_a), ("b", col_b)])?;

    // 2. Write to an in-memory Parquet file
    let mut buf = vec![];
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    let file_bytes = Bytes::from(buf);
    let file_len = file_bytes.len() as u64;
    println!(
        "Created Parquet file: {} bytes, {} rows",
        file_len, num_rows
    );

    // 3. Parse Parquet metadata
    let mut md = ParquetMetaDataPushDecoder::try_new(file_len)?;
    md.push_ranges(vec![0..file_len], vec![file_bytes.clone()])?;
    let DecodeResult::Data(metadata) = md.try_decode()? else {
        return Err("Failed to decode metadata".into());
    };
    let metadata = Arc::new(metadata);
    println!(
        "Parsed metadata: {} row groups",
        metadata.num_row_groups()
    );

    // 4. Initialize GPU
    let gpu = GpuContext::try_new().await?;
    println!("GPU context initialized");

    // 5. Build decoder with GPU acceleration
    let push_decoder = ParquetPushDecoderBuilder::try_new_decoder(metadata)?
        .with_batch_size(8192)
        .build()?;
    let mut decoder = GpuParquetDecoder::new(gpu, push_decoder);

    // 6. Decode loop
    let mut total_rows = 0u64;
    let mut batch_count = 0u64;
    loop {
        match decoder.try_decode()? {
            DecodeResult::NeedsData(ranges) => {
                let data: Vec<Bytes> = ranges
                    .iter()
                    .map(|r| file_bytes.slice(r.start as usize..r.end as usize))
                    .collect();
                decoder.push_ranges(ranges, data)?;
            }
            DecodeResult::Data(batch) => {
                batch_count += 1;
                total_rows += batch.num_rows() as u64;
                println!(
                    "  Batch {}: {} rows, {} columns",
                    batch_count,
                    batch.num_rows(),
                    batch.num_columns()
                );
            }
            DecodeResult::Finished => break,
        }
    }

    println!(
        "Done! Decoded {} total rows in {} batches via GPU",
        total_rows, batch_count
    );

    Ok(())
}
