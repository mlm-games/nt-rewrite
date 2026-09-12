//! 2D space: positions, arena bounds, collision push-outs.
//!
//! The bevy build carried positions in `Transform.translation` (`Vec3`);
//! the port stores plain [`Pos`] (`Vec2`) and keeps every helper 2D.
//! Laws are byte-identical to `world.rs` (`clamp_to_arena`,
//! `resolve_prop_collision`).

use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::comps_a::{ARENA_H, ARENA_W, FloorMask};

/// World position in pixels, y-down (GameMaker convention).
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct Pos(pub Vec2);

/// Player body radius in pixels.
pub const PLAYER_RADIUS: f32 = 8.0;

/// Clamp a body inside the arena rectangle.
pub fn clamp_to_arena(pos: &mut Vec2, radius: f32) {
    pos.x = pos.x.clamp(-ARENA_W / 2.0 + radius, ARENA_W / 2.0 - radius);
    pos.y = pos.y.clamp(-ARENA_H / 2.0 + radius, ARENA_H / 2.0 - radius);
}

/// Push a circle out of AABB props (`(center, size)` pairs). Pure over
/// an iterator so tests feed fixtures without a world.
pub fn resolve_prop_collision(
    pos: &mut Vec2,
    radius: f32,
    props: impl Iterator<Item = (Vec2, Vec2)>,
) {
    for (center, size) in props {
        let half = size / 2.0;
        let closest = Vec2::new(
            pos.x.clamp(center.x - half.x, center.x + half.x),
            pos.y.clamp(center.y - half.y, center.y + half.y),
        );
        let d = *pos - closest;
        let dist = d.length();
        if dist >= radius || dist <= 0.0001 {
            continue;
        }
        pos.x += d.x / dist * (radius - dist);
        pos.y += d.y / dist * (radius - dist);
    }
}

/// Push a circle out of unwalkable floor-mask cells.
pub fn resolve_mask_circle(mask: &FloorMask, pos: &mut Vec2, radius: f32) {
    mask.resolve_circle(pos, radius);
}

