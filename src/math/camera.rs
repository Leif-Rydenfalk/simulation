use nalgebra_glm as glm;

pub struct Camera {
    pub width: f32,
    pub height: f32,
    pub center: glm::Vec2,
    pub translation: glm::Vec2,
    pub zoom: f32,
    projection: glm::Mat4,
    view: glm::Mat4,
    inv_view: glm::Mat4,
}

impl Camera {
    /// Creates a new Camera with an orthographic projection.
    /// The projection covers (0, width) horizontally and (0, height) vertically,
    /// with a near plane at -1.0 and a far plane at 1.0.
    pub fn new(w: f32, h: f32) -> Self {
        // Create an orthographic projection matrix.
        let projection = glm::ortho(0.0, w, 0.0, h, -1.0, 1.0);
        Self {
            width: w,
            height: h,
            center: glm::vec2(w * 0.5, h * 0.5),
            translation: glm::vec2(0.0, 0.0),
            zoom: 1.0,
            projection,
            view: glm::Mat4::identity(),
            inv_view: glm::Mat4::identity(),
        }
    }

    /// Updates the view matrix based on the current translation and zoom.
    /// The view matrix is built by:
    /// 1. Translating to the center,
    /// 2. Scaling (applying zoom),
    /// 3. Translating by the negative of the translation and center.
    /// The inverse of the view matrix is also computed.
    pub fn update(&mut self) {
        // Start with an identity matrix.
        self.view = glm::Mat4::identity();
        // Translate by the center.
        self.view = glm::translate(&self.view, &glm::vec3(self.center.x, self.center.y, 0.0));
        // Apply zoom (scaling).
        self.view = glm::scale(&self.view, &glm::vec3(self.zoom, self.zoom, 1.0));
        // Translate back by the translation offset and center.
        self.view = glm::translate(
            &self.view,
            &glm::vec3(
                -self.translation.x - self.center.x,
                -self.translation.y - self.center.y,
                0.0,
            ),
        );
        // Update the inverse view matrix.
        self.inv_view = glm::inverse(&self.view);
    }

    /// Transforms a point from world space to viewport space.
    pub fn world_to_viewport(&self, pos: &mut glm::Vec2) {
        let temp = glm::vec4(pos.x, pos.y, 0.0, 1.0);
        let transformed = self.view * temp;
        pos.x = transformed.x;
        pos.y = transformed.y;
    }

    /// Transforms a point from viewport space back to world space.
    pub fn viewport_to_world(&self, pos: &mut glm::Vec2) {
        let temp = glm::vec4(pos.x, pos.y, 0.0, 1.0);
        let transformed = self.inv_view * temp;
        pos.x = transformed.x;
        pos.y = transformed.y;
    }

    /// Returns the current mouse position in world coordinates.
    /// The actual retrieval of the mouse position is left as a placeholder.
    pub fn get_mouse(&self) -> glm::Vec2 {
        // Replace the following line with your input-handling system.
        let mut mouse_pos = get_mouse_position();
        self.viewport_to_world(&mut mouse_pos);
        mouse_pos
    }

    /// Getter for the current projection matrix
    pub fn get_projection(&self) -> glm::Mat4 {
        self.projection
    }
}

/// Placeholder function for getting the mouse position.
/// In a real application, integrate with your input library.
fn get_mouse_position() -> glm::Vec2 {
    // For example purposes, we return (0, 0).
    glm::vec2(0.0, 0.0)
}
