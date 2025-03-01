use nalgebra_glm::Vec2;
use std::collections::HashMap;
use winit::{
    event::{ElementState, MouseButton, MouseScrollDelta},
    keyboard::KeyCode,
};

pub mod g2dcontroller;

/// Struct to track input state.
pub struct Input {
    // Window dimensions.
    pub width: u32,
    pub height: u32,
    // Scroll value (y-axis only).
    scroll: f32,
    // Keyboard key states.
    keyboard: HashMap<KeyCode, ElementState>,
    // Mouse button states.
    mouse: HashMap<MouseButton, ElementState>,
    // Mouse position (with y flipped so origin is bottom-left).
    mouse_pos: Vec2,
    last_mouse_pos: Vec2,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            scroll: 0.0,
            keyboard: HashMap::new(),
            mouse: HashMap::new(),
            mouse_pos: Vec2::new(0.0, 0.0),
            last_mouse_pos: Vec2::new(0.0, 0.0),
        }
    }
}

impl Input {
    /// Create a new input state with the given window dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Process a keyboard event.
    pub fn key_event(&mut self, key: KeyCode, state: ElementState) {
        self.keyboard.insert(key, state);
    }

    /// Process a mouse button event.
    pub fn mouse_button_event(&mut self, button: MouseButton, state: ElementState) {
        self.mouse.insert(button, state);
    }

    /// Process a cursor (mouse movement) event.
    /// Flips the y-coordinate to match bottom-left origin.
    pub fn cursor_event(&mut self, x: f64, y: f64) {
        self.last_mouse_pos = self.mouse_pos;
        self.mouse_pos = Vec2::new(x as f32, self.height as f32 - y as f32);
    }

    /// Process a scroll event.
    pub fn scroll_event(&mut self, delta: MouseScrollDelta) {
        match delta {
            MouseScrollDelta::LineDelta(_, y) => self.scroll = y,
            MouseScrollDelta::PixelDelta(pos) => self.scroll = pos.y as f32,
        }
    }

    /// Get the state of a key, optionally resetting it to Released.
    pub fn get_key(&mut self, key: KeyCode, reset: bool) -> ElementState {
        let state = *self.keyboard.get(&key).unwrap_or(&ElementState::Released);
        if reset {
            self.keyboard.insert(key, ElementState::Released);
        }
        state
    }

    /// Get the state of a mouse button, optionally resetting it to Released.
    pub fn get_mouse_button(&mut self, button: MouseButton, reset: bool) -> ElementState {
        let state = *self.mouse.get(&button).unwrap_or(&ElementState::Released);
        if reset {
            self.mouse.insert(button, ElementState::Released);
        }
        state
    }

    /// Get the current scroll value, optionally resetting it to zero.
    pub fn get_scroll(&mut self, reset: bool) -> f32 {
        let s = self.scroll;
        if reset {
            self.scroll = 0.0;
        }
        s
    }

    /// Get the current mouse position.
    pub fn get_mouse(&self) -> Vec2 {
        self.mouse_pos
    }

    /// Get the current mouse delta.
    pub fn get_mouse_delta(&self) -> Vec2 {
        self.mouse_pos - self.last_mouse_pos
    }

    /// Update the window dimensions (e.g. on resize).
    pub fn update_window_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}
