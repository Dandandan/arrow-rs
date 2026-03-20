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

//! GPU-accelerated Parquet decoding for Apache Arrow
//!
//! This crate provides GPU-accelerated decoding of Parquet data into Arrow arrays
//! using WebGPU compute shaders via [`wgpu`]. It builds on top of the
//! [`ParquetPushDecoder`](parquet::arrow::push_decoder::ParquetPushDecoder) to handle
//! Parquet I/O and page-level framing, then offloads the actual data decoding
//! (plain encoding, RLE, definition levels) to the GPU.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
//! │  Parquet File     │────▶│  PushDecoder     │────▶│  GPU Decoder     │
//! │  (bytes on disk)  │     │  (page framing)  │     │  (compute shader)│
//! └──────────────────┘     └──────────────────┘     └──────────────────┘
//!                                                           │
//!                                                           ▼
//!                                                    ┌──────────────┐
//!                                                    │ Arrow Arrays │
//!                                                    │ (CPU memory) │
//!                                                    └──────────────┘
//! ```
//!
//! # Supported Encodings
//!
//! Currently supports GPU-accelerated decoding for:
//! - **Plain encoding** for primitive types (Int32, Int64, Float32, Float64)
//! - **Boolean plain encoding** (bit-unpacking on GPU)
//! - Automatic fallback to CPU for unsupported encodings
//!
//! # Example
//!
//! ```no_run
//! use arrow_gpu::{GpuContext, GpuParquetDecoder};
//! use parquet::arrow::push_decoder::ParquetPushDecoderBuilder;
//! use parquet::file::metadata::ParquetMetaData;
//! use std::sync::Arc;
//!
//! # async fn example(metadata: Arc<ParquetMetaData>) -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize GPU
//! let gpu = GpuContext::try_new().await?;
//!
//! // Build a push decoder from parquet metadata
//! let push_decoder = ParquetPushDecoderBuilder::try_new_decoder(metadata)?
//!     .with_batch_size(8192)
//!     .build()?;
//!
//! // Wrap it with GPU acceleration
//! let mut decoder = GpuParquetDecoder::new(gpu, push_decoder);
//!
//! // Use the same push-based API — GPU decoding happens transparently
//! // when decoding plain-encoded primitive columns
//! # Ok(())
//! # }
//! ```

mod context;
mod decoder;
mod error;
mod shaders;

pub use context::GpuContext;
pub use decoder::GpuParquetDecoder;
pub use error::GpuError;
