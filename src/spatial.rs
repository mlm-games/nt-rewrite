//! 2D space: positions, arena bounds, collision push-outs.
//!
//! The bevy build carried positions in `Transform.translation` (`Vec3`);
//! the port stores plain [`Pos`] (`Vec2`) and keeps every helper 2D.
//! Laws are byte-identical to `world.rs` (`clamp_to_arena`,
//! `resolve_prop_collision`).

use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::comps_a::{ARENA_H, ARENA_W, FloorMask, TILE};
use crate::projectile_math::{
    bounce_velocity, circle_aabb_normal, circle_aabb_penetration, swept_circle_aabb,
};

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidContact {
    pub normal: Vec2,
    pub penetration: f32,
    pub fraction: f32,
}

#[derive(Clone, Copy, Debug)]
struct SolidAabb {
    min: Vec2,
    max: Vec2,
}

impl SolidAabb {
    fn from_center_size(center: Vec2, size: Vec2) -> Self {
        let half = size.abs() * 0.5;
        Self {
            min: center - half,
            max: center + half,
        }
    }

    fn center_half(self) -> (Vec2, Vec2) {
        ((self.min + self.max) * 0.5, (self.max - self.min) * 0.5)
    }
}

fn solid_shapes(
    start: Vec2,
    displacement: Vec2,
    radius: f32,
    props: &[(Vec2, Vec2)],
    mask: Option<&FloorMask>,
) -> Vec<SolidAabb> {
    let mut shapes = Vec::with_capacity(props.len() + 5);
    for &(center, size) in props {
        if size.x > 0.0 || size.y > 0.0 {
            shapes.push(SolidAabb::from_center_size(center, size));
        }
    }

    let extent = 100_000.0;
    shapes.push(SolidAabb {
        min: Vec2::new(-ARENA_W * 0.5 - extent, -ARENA_H * 0.5 - extent),
        max: Vec2::new(-ARENA_W * 0.5, ARENA_H * 0.5 + extent),
    });
    shapes.push(SolidAabb {
        min: Vec2::new(ARENA_W * 0.5, -ARENA_H * 0.5 - extent),
        max: Vec2::new(ARENA_W * 0.5 + extent, ARENA_H * 0.5 + extent),
    });
    shapes.push(SolidAabb {
        min: Vec2::new(-ARENA_W * 0.5 - extent, -ARENA_H * 0.5 - extent),
        max: Vec2::new(ARENA_W * 0.5 + extent, -ARENA_H * 0.5),
    });
    shapes.push(SolidAabb {
        min: Vec2::new(-ARENA_W * 0.5 - extent, ARENA_H * 0.5),
        max: Vec2::new(ARENA_W * 0.5 + extent, ARENA_H * 0.5 + extent),
    });

    if let Some(mask) = mask
        && !mask.cells.is_empty()
    {
        let min = start.min(start + displacement) - Vec2::splat(radius + TILE);
        let max = start.max(start + displacement) + Vec2::splat(radius + TILE);
        let min_cell_x =
            ((min.x / TILE).floor() as i32).max((-ARENA_W * 0.5 / TILE).floor() as i32);
        let max_cell_x =
            ((max.x / TILE).floor() as i32).min((ARENA_W * 0.5 / TILE).ceil() as i32 - 1);
        let min_cell_y =
            ((min.y / TILE).floor() as i32).max((-ARENA_H * 0.5 / TILE).floor() as i32);
        let max_cell_y =
            ((max.y / TILE).floor() as i32).min((ARENA_H * 0.5 / TILE).ceil() as i32 - 1);
        for y in min_cell_y..=max_cell_y {
            for x in min_cell_x..=max_cell_x {
                if mask.cells.contains(&(x, y)) {
                    continue;
                }
                shapes.push(SolidAabb::from_center_size(
                    Vec2::new(x as f32 * TILE + TILE * 0.5, y as f32 * TILE + TILE * 0.5),
                    Vec2::splat(TILE),
                ));
            }
        }
    }
    shapes
}

fn first_swept_contact(
    start: Vec2,
    displacement: Vec2,
    radius: f32,
    shapes: &[SolidAabb],
) -> Option<SolidContact> {
    let mut first: Option<SolidContact> = None;
    for shape in shapes {
        let (center, half) = shape.center_half();
        let Some(contact) = swept_circle_aabb(start, displacement, radius, center, half) else {
            continue;
        };
        if contact.fraction <= 1e-6
            && contact.penetration <= 1e-5
            && displacement.dot(contact.normal) >= -1e-6
        {
            continue;
        }
        if first.is_none_or(|old| contact.fraction < old.fraction) {
            first = Some(SolidContact {
                normal: contact.normal,
                penetration: contact.penetration,
                fraction: contact.fraction,
            });
        }
    }
    first
}

fn first_current_contact(pos: Vec2, radius: f32, shapes: &[SolidAabb]) -> Option<SolidContact> {
    let mut first: Option<SolidContact> = None;
    for shape in shapes {
        let (center, half) = shape.center_half();
        let Some(normal) = circle_aabb_normal(pos, radius, center, half) else {
            continue;
        };
        let Some(penetration) = circle_aabb_penetration(pos, radius, center, half) else {
            continue;
        };
        let candidate = SolidContact {
            normal,
            penetration,
            fraction: 0.0,
        };
        if first.is_none_or(|old| candidate.penetration > old.penetration) {
            first = Some(candidate);
        }
    }
    first
}

pub fn solid_contact(
    pos: Vec2,
    radius: f32,
    props: &[(Vec2, Vec2)],
    mask: Option<&FloorMask>,
) -> Option<SolidContact> {
    let shapes = solid_shapes(pos, Vec2::ZERO, radius, props, mask);
    first_current_contact(pos, radius, &shapes)
}

pub fn move_contact_solid(
    pos: &mut Vec2,
    displacement: Vec2,
    radius: f32,
    props: &[(Vec2, Vec2)],
    mask: Option<&FloorMask>,
) -> Option<SolidContact> {
    clamp_to_arena(pos, radius);
    let shapes = solid_shapes(*pos, displacement, radius, props, mask);
    let mut remaining = displacement;
    let mut first = None;
    for _ in 0..4 {
        let Some(contact) = first_swept_contact(*pos, remaining, radius, &shapes) else {
            *pos += remaining;
            clamp_to_arena(pos, radius);
            break;
        };
        if first.is_none() {
            first = Some(contact);
        }
        *pos += remaining * contact.fraction;
        if contact.penetration > 0.0 {
            *pos += contact.normal * (contact.penetration + 0.001);
        }
        let consumed = remaining * contact.fraction;
        remaining -= consumed;
        remaining -= contact.normal * remaining.dot(contact.normal);
        if remaining.length_squared() <= 1e-8 {
            break;
        }
    }
    clamp_to_arena(pos, radius);
    first
}

pub fn move_bounce_solid(
    pos: &mut Vec2,
    velocity: &mut Vec2,
    radius: f32,
    dt: f32,
    props: &[(Vec2, Vec2)],
    mask: Option<&FloorMask>,
    bounce: bool,
) -> Option<SolidContact> {
    move_bounce_solid_displacement(
        pos,
        velocity,
        *velocity * dt.max(0.0),
        radius,
        props,
        mask,
        bounce,
    )
}

pub fn move_bounce_solid_displacement(
    pos: &mut Vec2,
    velocity: &mut Vec2,
    displacement: Vec2,
    radius: f32,
    props: &[(Vec2, Vec2)],
    mask: Option<&FloorMask>,
    bounce: bool,
) -> Option<SolidContact> {
    clamp_to_arena(pos, radius);
    let shapes = solid_shapes(*pos, displacement, radius, props, mask);
    let mut remaining = displacement;
    let mut first = None;
    for _ in 0..4 {
        let Some(contact) = first_swept_contact(*pos, remaining, radius, &shapes) else {
            *pos += remaining;
            clamp_to_arena(pos, radius);
            break;
        };
        if first.is_none() {
            first = Some(contact);
        }
        *pos += remaining * contact.fraction;
        if contact.penetration > 0.0 {
            *pos += contact.normal * (contact.penetration + 0.001);
        }
        let consumed = remaining * contact.fraction;
        remaining -= consumed;
        if bounce {
            if velocity.dot(contact.normal) < 0.0 {
                *velocity = bounce_velocity(*velocity, contact.normal);
            }
            if remaining.dot(contact.normal) < 0.0 {
                remaining = bounce_velocity(remaining, contact.normal);
            }
        } else {
            remaining -= contact.normal * remaining.dot(contact.normal);
        }
        if remaining.length_squared() <= 1e-8 {
            break;
        }
    }
    clamp_to_arena(pos, radius);
    first
}

pub fn potential_step_solid(
    pos: &mut Vec2,
    target: Vec2,
    distance: f32,
    radius: f32,
    props: &[(Vec2, Vec2)],
    mask: Option<&FloorMask>,
) -> Option<SolidContact> {
    let delta = target - *pos;
    if delta.length_squared() <= 1e-8 {
        return None;
    }
    let step = delta.normalize() * distance.max(0.0);
    move_contact_solid(pos, step, radius, props, mask)
}
