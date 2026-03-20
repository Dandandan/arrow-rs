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

//! GPU device context and pipeline management.

use crate::error::GpuError;
use crate::shaders;
use std::sync::Arc;
use wgpu::{self, ComputePipeline};

/// Holds the GPU device, queue, and pre-compiled compute pipelines for
/// Parquet page decoding.
///
/// Create via [`GpuContext::try_new`]. This is cheap to clone (uses `Arc` internally).
#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<GpuContextInner>,
}

struct GpuContextInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Pipeline for decoding 4-byte plain-encoded values (Int32, Float32)
    decode_plain_i32_pipeline: ComputePipeline,
    /// Pipeline for decoding 8-byte plain-encoded values (Int64, Float64)
    decode_plain_i64_pipeline: ComputePipeline,
    /// Pipeline for unpacking booleans from bit-packed bytes
    decode_plain_bool_pipeline: ComputePipeline,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", &"<wgpu::Device>")
            .finish()
    }
}

impl GpuContext {
    /// Initialize the GPU context, requesting a device and compiling shaders.
    ///
    /// Returns an error if no suitable GPU adapter is available.
    pub async fn try_new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        log::info!("Using GPU adapter: {:?}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("arrow-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            }, None)
            .await?;

        let decode_plain_i32_pipeline =
            shaders::create_decode_plain_i32_pipeline(&device);
        let decode_plain_i64_pipeline =
            shaders::create_decode_plain_i64_pipeline(&device);
        let decode_plain_bool_pipeline =
            shaders::create_decode_plain_bool_pipeline(&device);

        Ok(Self {
            inner: Arc::new(GpuContextInner {
                device,
                queue,
                decode_plain_i32_pipeline,
                decode_plain_i64_pipeline,
                decode_plain_bool_pipeline,
            }),
        })
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    pub(crate) fn decode_plain_i32_pipeline(&self) -> &ComputePipeline {
        &self.inner.decode_plain_i32_pipeline
    }

    pub(crate) fn decode_plain_i64_pipeline(&self) -> &ComputePipeline {
        &self.inner.decode_plain_i64_pipeline
    }

    pub(crate) fn decode_plain_bool_pipeline(&self) -> &ComputePipeline {
        &self.inner.decode_plain_bool_pipeline
    }

    /// Decode plain-encoded 4-byte values (i32 / f32) on the GPU.
    ///
    /// `input` is the raw page data (little-endian 4-byte values packed contiguously).
    /// Returns the decoded bytes (same content, but processed through GPU pipeline
    /// which also validates and copies the data).
    pub(crate) fn decode_plain_4byte(&self, input: &[u8], num_values: u32) -> Vec<u8> {
        self.run_plain_pipeline(
            self.decode_plain_i32_pipeline(),
            input,
            num_values,
            4,
        )
    }

    /// Decode plain-encoded 8-byte values (i64 / f64) on the GPU.
    pub(crate) fn decode_plain_8byte(&self, input: &[u8], num_values: u32) -> Vec<u8> {
        self.run_plain_pipeline(
            self.decode_plain_i64_pipeline(),
            input,
            num_values,
            8,
        )
    }

    /// Decode plain-encoded booleans on the GPU (bit-unpacking).
    ///
    /// `input` is bit-packed boolean data. Returns a buffer of `num_values` bytes
    /// where each byte is 0 or 1, suitable for constructing a BooleanBuffer.
    pub(crate) fn decode_plain_bool(&self, input: &[u8], num_values: u32) -> Vec<u8> {
        let device = self.device();
        let queue = self.queue();

        let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bool_input"),
            size: input.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&input_buf, 0, input);

        // Output: one byte per boolean value
        let output_size = num_values as u64;
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bool_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [num_values];
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bool_params"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck_cast_slice(&params));

        let pipeline = self.decode_plain_bool_pipeline();
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bool_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bool_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bool_pass"),
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((num_values + 63) / 64, 1, 1);
        }

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bool_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range().to_vec();
        staging_buf.unmap();
        data
    }

    /// Run a plain-encoding decode pipeline (for fixed-width types).
    fn run_plain_pipeline(
        &self,
        pipeline: &ComputePipeline,
        input: &[u8],
        num_values: u32,
        value_size: u32,
    ) -> Vec<u8> {
        let device = self.device();
        let queue = self.queue();

        let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plain_input"),
            size: input.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&input_buf, 0, input);

        let output_size = (num_values * value_size) as u64;
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plain_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [num_values];
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plain_params"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck_cast_slice(&params));

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("plain_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("plain_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("plain_pass"),
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((num_values + 63) / 64, 1, 1);
        }

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plain_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range().to_vec();
        staging_buf.unmap();
        data
    }
}

/// Safe cast of a `&[u32]` to `&[u8]` (little-endian).
fn bytemuck_cast_slice(data: &[u32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    }
}
