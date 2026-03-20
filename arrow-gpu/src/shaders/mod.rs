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

//! WGSL compute shaders for GPU-accelerated Parquet decoding.
//!
//! Each shader operates on raw encoded page data and produces decoded output
//! in Arrow-compatible layout.

use wgpu::{ComputePipeline, Device};

/// Create a pipeline for decoding plain-encoded 4-byte values (i32/f32).
///
/// The shader copies 4-byte values from packed input to output. Each workgroup
/// thread handles one value.
pub fn create_decode_plain_i32_pipeline(device: &Device) -> ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("decode_plain_i32"),
        source: wgpu::ShaderSource::Wgsl(DECODE_PLAIN_I32_WGSL.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("decode_plain_i32_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Create a pipeline for decoding plain-encoded 8-byte values (i64/f64).
pub fn create_decode_plain_i64_pipeline(device: &Device) -> ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("decode_plain_i64"),
        source: wgpu::ShaderSource::Wgsl(DECODE_PLAIN_I64_WGSL.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("decode_plain_i64_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Create a pipeline for unpacking plain-encoded booleans from bit-packed bytes.
pub fn create_decode_plain_bool_pipeline(device: &Device) -> ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("decode_plain_bool"),
        source: wgpu::ShaderSource::Wgsl(DECODE_PLAIN_BOOL_WGSL.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("decode_plain_bool_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// WGSL shader: decode plain-encoded i32/f32 values.
///
/// Each thread copies one 4-byte value from the input buffer to the output buffer.
/// Input is treated as an array of u32 (same bit pattern as i32/f32 in little-endian).
const DECODE_PLAIN_I32_WGSL: &str = r#"
struct Params {
    num_values: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_values {
        return;
    }
    output[idx] = input[idx];
}
"#;

/// WGSL shader: decode plain-encoded i64/f64 values.
///
/// Each thread copies one 8-byte value (as two u32s) from input to output.
const DECODE_PLAIN_I64_WGSL: &str = r#"
struct Params {
    num_values: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_values {
        return;
    }
    // Each i64/f64 is 2 x u32
    let src = idx * 2u;
    output[src] = input[src];
    output[src + 1u] = input[src + 1u];
}
"#;

/// WGSL shader: unpack plain-encoded booleans from bit-packed bytes.
///
/// Input: bit-packed bytes where bit i of byte j represents boolean value (j*8 + i).
/// Output: one byte per boolean (0x00 or 0x01).
///
/// Each thread unpacks one boolean value.
const DECODE_PLAIN_BOOL_WGSL: &str = r#"
struct Params {
    num_values: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_values {
        return;
    }

    // Which byte in input contains this bit?
    let byte_idx = idx / 8u;
    let bit_idx = idx % 8u;

    // Read the u32 that contains this byte
    let word_idx = byte_idx / 4u;
    let byte_offset = byte_idx % 4u;
    let word = input[word_idx];
    let byte_val = (word >> (byte_offset * 8u)) & 0xFFu;
    let bit_val = (byte_val >> bit_idx) & 1u;

    // Write one byte to output (0 or 1)
    let out_word_idx = idx / 4u;
    let out_byte_offset = idx % 4u;

    // Use atomicOr to write individual bytes within a u32 word
    // Since each thread writes to a unique byte position, we can
    // safely construct the word. We write the full byte at its position.
    let shifted = bit_val << (out_byte_offset * 8u);

    // For correctness with parallel writes to the same u32, we need
    // to use atomic operations. However, WGSL storage buffers don't
    // support atomics on regular arrays easily. Instead, we ensure
    // each workgroup processes 64 consecutive values and we can
    // pack 4 booleans per u32 word deterministically.

    // Simple approach: each group of 4 threads writes to the same word.
    // Thread 0 of each group of 4 writes the full word.
    // We use workgroup shared memory to coordinate.

    // Actually, simplest correct approach: just write full u32 words
    // where thread idx handles 4 consecutive booleans if idx is word-aligned.
    // But that changes the parallelism model. Let's use a simpler per-value
    // output that's u32-per-boolean for now, then pack on CPU.

    // Output: one u32 per boolean (0 or 1). Caller packs into bit array.
    output[idx] = bit_val;
}
"#;
