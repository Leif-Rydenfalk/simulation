// simulation/wgpu_buffer.rs
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferDescriptor,
    BufferSize, BufferUsages, Device, Queue, ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::simulation::wgpu_utility::WgpuUtility;

// Equivalent to CudaBuffer
pub struct WgpuBuffer {
    pub buffers: Vec<Buffer>,
    pub size: usize,
    pub storage_binding_layout: BindGroupLayout,
    pub storage_binding: BindGroup,
}

impl WgpuBuffer {
    pub fn new(device: &Device, sizes: Vec<usize>, label: &str) -> Self {
        let buffers = WgpuUtility::bulk_allocate(device, sizes.clone(), label);
        let size = buffers.len();

        // Create storage buffer binding layout and binding
        let storage_binding_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some(&format!("{}_storage_binding_layout", label)),
        });

        let entries: Vec<BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| BindGroupEntry {
                binding: i as u32,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer,
                    offset: 0,
                    size: None,
                }),
            })
            .collect();

        let storage_binding = device.create_bind_group(&BindGroupDescriptor {
            layout: &storage_binding_layout,
            entries: &entries,
            label: Some(&format!("{}_storage_binding", label)),
        });

        Self {
            buffers,
            size,
            storage_binding_layout,
            storage_binding,
        }
    }

    pub fn new_sized(device: &Device, element_size: usize, count: usize, label: &str) -> Self {
        let sizes = vec![element_size * count];
        Self::new(device, sizes, label)
    }
}

// Equivalent to CudaGLBuffer - handles interop between wgpu and rendering
pub struct WgpuGLBuffer {
    pub buffers: Vec<Buffer>,
    pub size: usize,
    pub vertex_binding_layout: BindGroupLayout,
    pub vertex_binding: BindGroup,
}

impl WgpuGLBuffer {
    pub fn new(device: &Device, sizes: Vec<usize>, label: &str) -> Self {
        let buffers = WgpuUtility::bulk_allocate(device, sizes.clone(), label);
        let size = buffers.len();

        // Create vertex buffer binding layout and binding
        let vertex_binding_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some(&format!("{}_vertex_binding_layout", label)),
        });

        let entries: Vec<BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| BindGroupEntry {
                binding: i as u32,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer,
                    offset: 0,
                    size: None,
                }),
            })
            .collect();

        let vertex_binding = device.create_bind_group(&BindGroupDescriptor {
            layout: &vertex_binding_layout,
            entries: &entries,
            label: Some(&format!("{}_vertex_binding", label)),
        });

        Self {
            buffers,
            size,
            vertex_binding_layout,
            vertex_binding,
        }
    }

    pub fn new_sized(device: &Device, element_size: usize, count: usize, label: &str) -> Self {
        let sizes = vec![element_size * count];
        Self::new(device, sizes, label)
    }

    // Map the buffer for access by the vertex shader
    pub fn map_buffers(&self, queue: &Queue) {
        // In wgpu we don't need to explicitly map resources like in CUDA
        // Instead, we write to the buffers using the queue
    }

    // Unmap the buffer after use
    pub fn unmap_buffers(&self) {
        // In wgpu we don't need to explicitly unmap resources
    }
}
