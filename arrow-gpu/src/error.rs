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

use parquet::errors::ParquetError;

/// Errors that can occur during GPU-accelerated Parquet decoding.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("GPU device request failed: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    #[error("No suitable GPU adapter found")]
    NoAdapter,

    #[error("GPU buffer mapping failed: {0}")]
    BufferMap(#[from] wgpu::BufferAsyncError),

    #[error("Parquet error: {0}")]
    Parquet(#[from] ParquetError),

    #[error("GPU error: {0}")]
    Other(String),
}

impl From<GpuError> for ParquetError {
    fn from(e: GpuError) -> Self {
        ParquetError::General(format!("GPU error: {e}"))
    }
}
