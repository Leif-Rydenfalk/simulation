use crate::input::Input;
use crate::math::camera::Camera;
use winit::event::ElementState;
use winit::event::MouseButton;

/// Struct for C2DController
pub struct C2DController {
    pub camera: Option<Camera>,
}

impl C2DController {
    /// Create a new C2DController
    pub fn new() -> Self {
        Self { camera: None }
    }

    /// Connect the controller to a camera
    pub fn connect(&mut self, camera: Camera) {
        self.camera = Some(camera);
    }

    /// Update the camera based on input
    pub fn update(&mut self, input: &mut Input, sensitivity: f32, zoom_speed: f32) {
        if let Some(camera) = &mut self.camera {
            // Check if right mouse button is pressed (similar to GLFW_PRESS).
            if input.get_mouse_button(MouseButton::Right, false) == ElementState::Pressed {
                // Update the camera translation (pan the camera).
                camera.translation -= input.get_mouse_delta() * sensitivity / camera.zoom;
            }

            // Get the scroll input.
            let scroll = input.get_scroll(true);

            // Zoom in/out based on scroll value.
            if scroll > 0.0 {
                camera.zoom *= zoom_speed;
            } else if scroll < 0.0 {
                camera.zoom /= zoom_speed;
            }
        }
    }
}
