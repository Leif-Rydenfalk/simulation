// simulation/helper_grid.rs
use crate::simulation::settings::{CELL_SIZE, CENTER_X, ESTIMATED_COLUMNS, MAX_CHECKS};

// Constants that would normally be in Settings.h
// const CENTER_X: f32 = 0.0;
// const CELL_SIZE: f32 = 1.0;
// const ESTIMATED_COLUMNS: i32 = 100;
// const MAX_CHECKS: usize = 32;

// WGSL shader code for helper grid functions
pub const HELPER_GRID_SHADER: &str = r#"
fn convert(value: f32) -> i32 {
    return i32((value + CENTER_X * 1.05) / CELL_SIZE);
}

fn hasher(r: i32, c: i32, n: i32) -> i32 {
    return (r * ESTIMATED_COLUMNS + c) % n;
}

fn query(r: i32, c: i32, hashes: array<i32>, starts: array<i32>, n: i32, outputs: array<i32>, output_start: i32) -> i32 {
    var output_idx = output_start;
    let h = hasher(r, c, n);
    var index = starts[h];

    if (index == 2147483647) { // INT_MAX
        return output_idx;
    }

    while (index < n && hashes[index] == h && output_idx < MAX_CHECKS) {
        outputs[output_idx] = index;
        output_idx += 1;
        index += 1;
    }

    return output_idx;
}

fn query_radius_box(x: f32, y: f32, r: f32, hashes: array<i32>, starts: array<i32>, n: i32, outputs: array<i32>, output_start: i32) -> i32 {
    let bottom_row = convert(y - r);
    let bottom_column = convert(x - r);
    let top_row = convert(y + r);
    let top_column = convert(x + r);
    
    var output_idx = output_start;
    
    for (var i = bottom_row; i <= top_row; i++) {
        for (var j = bottom_column; j <= top_column; j++) {
            output_idx = query(i, j, hashes, starts, n, outputs, output_idx);
        }
    }
    
    return output_idx;
}
"#;

// Rust equivalents of the device functions - useful for CPU-side verification or hybrid processing
pub fn convert(value: f32) -> i32 {
    ((value + CENTER_X * 1.05) / CELL_SIZE) as i32
}

pub fn hasher(r: i32, c: i32, n: i32) -> i32 {
    (r * ESTIMATED_COLUMNS + c) % n
}

pub fn query(
    r: i32,
    c: i32,
    hashes: &[i32],
    starts: &[i32],
    n: i32,
    outputs: &mut [i32],
    output_start: usize,
) -> usize {
    let mut output_idx = output_start;
    let h = hasher(r, c, n);
    let mut index = starts[h as usize];

    if index == i32::MAX {
        return output_idx;
    }

    while index < n && hashes[index as usize] == h && output_idx < MAX_CHECKS {
        outputs[output_idx] = index;
        output_idx += 1;
        index += 1;
    }

    output_idx
}

pub fn query_radius_box(
    x: f32,
    y: f32,
    r: f32,
    hashes: &[i32],
    starts: &[i32],
    n: i32,
    outputs: &mut [i32],
    output_start: usize,
) -> usize {
    let bottom_row = convert(y - r);
    let bottom_column = convert(x - r);
    let top_row = convert(y + r);
    let top_column = convert(x + r);

    let mut output_idx = output_start;

    for i in bottom_row..=top_row {
        for j in bottom_column..=top_column {
            output_idx = query(i, j, hashes, starts, n, outputs, output_idx);
        }
    }

    output_idx
}
