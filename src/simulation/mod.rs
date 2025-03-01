// simulation/mod.rs
pub mod helper_grid;
pub mod particle_grid;
pub mod particle_manager;
pub mod wgpu_buffer;
pub mod wgpu_utility;

use std::sync::Arc;
use wgpu::{Device, Queue};

pub struct Simulation {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    // Add other shared resources here
}

impl Simulation {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self { device, queue }
    }
}
