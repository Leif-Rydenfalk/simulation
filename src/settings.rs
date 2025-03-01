pub mod settings {
    // World dimensions
    pub const WORLD_WIDTH: i32 = 40000;
    pub const WORLD_HEIGHT: i32 = WORLD_WIDTH / 4;
    pub const CENTER_X: i32 = WORLD_WIDTH / 2;
    pub const CENTER_Y: i32 = WORLD_HEIGHT / 2;

    // Physics settings
    pub const SUBSTEPS: i32 = 6;
    pub const MAX_CHECKS: i32 = 16;
    pub const DT: f32 = 1.0 / 120.0 / SUBSTEPS as f32;

    // Particle settings
    pub const RADIUS: i32 = 10;
    pub const CELL_SIZE: i32 = RADIUS * 2;
    pub const ESTIMATED_COLUMNS: i32 = WORLD_WIDTH / CELL_SIZE + 1;
    pub const ESTIMATED_ROWS: i32 = WORLD_HEIGHT / CELL_SIZE + 1;

    // Gravity
    pub const G: f32 = 600.0;
    pub const G_DT_DT: f32 = G * DT * DT;

    // Mouse interaction
    pub const MOUSE_FORCE: f32 = 0.05 * DT;
    pub const MOUSE_MAX_DISTANCE: f32 = RADIUS as f32 * 50.0;
    pub const MOUSE_SENSITIVITY: f32 = 2.5;
    pub const MOUSE_ZOOM_SPEED: f32 = 1.25;

    // Spring constraints
    pub const SPRING_INITIAL_STRETCH: f32 = 1.001;
    pub const SPRING_HARDNESS: f32 = 250.0 * DT;
    pub const MIN_SPRING_DISTANCE: f32 = 0.001;
    pub const MAX_SPRING_DISTANCE: f32 = RADIUS as f32 * 0.3;

    // Physics constraints
    pub const MAX_VELOCITY: f32 = RADIUS as f32 * 0.6;
    pub const AIR_RESIST_SCALE: f32 = 0.15;
    pub const FRICTION_SCALE: f32 = 0.15;
    pub const FLOOR_HARDNESS: f32 = 80.0 * DT;
    pub const PARTICLE_HARDNESS: f32 = 80.0 * DT;
    pub const MAX_PARTICLES: i32 = 1 << 22;

    // Soft body parameters
    pub const R_BODY_COUNT: i32 = 64;
    pub const BODY_COUNT: i32 = R_BODY_COUNT * R_BODY_COUNT;

    // Other
    pub const SLOWMO_SCALE: i32 = 4;
}
