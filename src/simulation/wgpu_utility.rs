// simulation/wgpu_utility.rs
use std::mem::size_of;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, Device, Queue,
    util::{BufferInitDescriptor, DeviceExt},
};

pub struct WgpuUtility;

impl WgpuUtility {
    pub fn bulk_allocate(device: &Device, sizes: Vec<usize>, label: &str) -> Vec<Buffer> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("{}_{}", label, i)),
                    size: size as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect()
    }

    pub fn allocate_and_copy<T: bytemuck::Pod>(
        device: &Device,
        queue: &Queue,
        data: &[T],
        label: &str,
    ) -> Buffer {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });
        buffer
    }

    pub fn copy<T: bytemuck::Pod>(
        queue: &Queue,
        dst: &Buffer,
        src: &[T],
        start: usize,
        end: usize,
    ) {
        let offset = start * size_of::<T>();
        let data = &src[start..end];
        queue.write_buffer(dst, offset as u64, bytemuck::cast_slice(data));
    }

    pub fn copy_buffer(queue: &Queue, dst: &Buffer, src: &Buffer, start: usize, size: usize) {
        // In a full implementation, we would use an encoder and submit a copy command
        // For simplicity, this just represents the operation
        let encoder = queue
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Buffer Encoder"),
            });

        // encoder.copy_buffer_to_buffer(src, start as u64, dst, start as u64, size as u64);
        // queue.submit(std::iter::once(encoder.finish()));
    }
}
