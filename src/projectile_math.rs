//! Projectile collision math. Verbatim port (glam `Vec2`,
//! `bevy_ecs` entity ids); includes nt's own oracle tests.

use bevy_ecs::prelude::Entity;
use glam::Vec2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweptCircleAabb {
    pub normal: Vec2,
    pub penetration: f32,
    pub fraction: f32,
}

pub fn split_directions(base_dir: Vec2, pellets: u8, spread: f32, samples: &[f32]) -> Vec<Vec2> {
    let base_angle = if base_dir.length_squared() > 1e-6 {
        base_dir.y.atan2(base_dir.x)
    } else {
        0.0
    };
    let mut out = Vec::with_capacity(pellets as usize);
    for i in 0..pellets as usize {
        let t = samples.get(i).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        let angle = base_angle + t * spread;
        out.push(Vec2::new(angle.cos(), angle.sin()));
    }
    out
}

pub fn circle_aabb_normal(pos: Vec2, radius: f32, center: Vec2, half: Vec2) -> Option<Vec2> {
    let half = half.abs();
    let closest = Vec2::new(
        pos.x.clamp(center.x - half.x, center.x + half.x),
        pos.y.clamp(center.y - half.y, center.y + half.y),
    );
    let delta = pos - closest;
    let d2 = delta.length_squared();
    if d2 > radius * radius {
        return None;
    }
    if d2 > 1e-8 {
        return Some(delta.normalize());
    }

    let dx = half.x - (pos.x - center.x).abs();
    let dy = half.y - (pos.y - center.y).abs();
    let sx = if pos.x < center.x { -1.0 } else { 1.0 };
    let sy = if pos.y < center.y { -1.0 } else { 1.0 };
    if dx < dy {
        Some(Vec2::new(sx, 0.0))
    } else {
        Some(Vec2::new(0.0, sy))
    }
}

pub fn circle_aabb_penetration(pos: Vec2, radius: f32, center: Vec2, half: Vec2) -> Option<f32> {
    let half = half.abs();
    let closest = Vec2::new(
        pos.x.clamp(center.x - half.x, center.x + half.x),
        pos.y.clamp(center.y - half.y, center.y + half.y),
    );
    let delta = pos - closest;
    let distance = delta.length();
    if distance > radius {
        return None;
    }
    if distance > 1e-8 {
        return Some(radius - distance);
    }
    let dx = half.x - (pos.x - center.x).abs();
    let dy = half.y - (pos.y - center.y).abs();
    Some(radius + dx.min(dy))
}

pub fn circle_aabb_overlap(
    pos: Vec2,
    radius: f32,
    center: Vec2,
    half: Vec2,
) -> Option<(Vec2, f32)> {
    let half = half.abs();
    let closest = Vec2::new(
        pos.x.clamp(center.x - half.x, center.x + half.x),
        pos.y.clamp(center.y - half.y, center.y + half.y),
    );
    let delta = pos - closest;
    let d2 = delta.length_squared();
    if d2 > radius * radius {
        return None;
    }
    let normal = if d2 > 1e-8 {
        delta.normalize()
    } else {
        let dx = half.x - (pos.x - center.x).abs();
        let dy = half.y - (pos.y - center.y).abs();
        let sx = if pos.x < center.x { -1.0 } else { 1.0 };
        let sy = if pos.y < center.y { -1.0 } else { 1.0 };
        if dx < dy {
            Vec2::new(sx, 0.0)
        } else {
            Vec2::new(0.0, sy)
        }
    };
    let distance = delta.length();
    if distance > radius {
        return None;
    }
    let penetration = if distance > 1e-8 {
        radius - distance
    } else {
        let dx = half.x - (pos.x - center.x).abs();
        let dy = half.y - (pos.y - center.y).abs();
        radius + dx.min(dy)
    };
    Some((normal, penetration))
}

pub fn swept_circle_aabb(
    start: Vec2,
    displacement: Vec2,
    radius: f32,
    center: Vec2,
    half: Vec2,
) -> Option<SweptCircleAabb> {
    let half = half.abs();
    let radius = radius.max(0.0);
    let distance = |t: f32| {
        let p = start + displacement * t;
        let closest = Vec2::new(
            p.x.clamp(center.x - half.x, center.x + half.x),
            p.y.clamp(center.y - half.y, center.y + half.y),
        );
        (p - closest).length()
    };
    let distance_squared = |t: f32| {
        let p = start + displacement * t;
        let closest = Vec2::new(
            p.x.clamp(center.x - half.x, center.x + half.x),
            p.y.clamp(center.y - half.y, center.y + half.y),
        );
        (p - closest).length_squared()
    };
    let d0 = distance(0.0);
    if d0 <= radius + 1e-5 {
        let normal = circle_aabb_normal(start, radius, center, half).unwrap_or_else(|| {
            let p = start - center;
            if p.x.abs() > p.y.abs() {
                Vec2::new(if p.x < 0.0 { -1.0 } else { 1.0 }, 0.0)
            } else {
                Vec2::new(0.0, if p.y < 0.0 { -1.0 } else { 1.0 })
            }
        });
        return Some(SweptCircleAabb {
            normal,
            penetration: circle_aabb_penetration(start, radius, center, half).unwrap_or(0.0),
            fraction: 0.0,
        });
    }

    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..48 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if distance_squared(m1) < distance_squared(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let min_t = (lo + hi) * 0.5;
    if distance(min_t) > radius + 1e-4 {
        return None;
    }

    let mut lo = 0.0;
    let mut hi = min_t;
    for _ in 0..40 {
        let mid = (lo + hi) * 0.5;
        if distance(mid) <= radius + 1e-5 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let fraction = hi.clamp(0.0, 1.0);
    let point = start + displacement * fraction;
    let normal = circle_aabb_normal(point, radius + 1e-3, center, half).unwrap_or_else(|| {
        let closest = Vec2::new(
            point.x.clamp(center.x - half.x, center.x + half.x),
            point.y.clamp(center.y - half.y, center.y + half.y),
        );
        (point - closest).normalize_or_zero()
    });
    Some(SweptCircleAabb {
        normal,
        penetration: circle_aabb_penetration(point, radius, center, half).unwrap_or(0.0),
        fraction,
    })
}

pub fn arena_wall_normal(pos: Vec2, radius: f32, arena_w: f32, arena_h: f32) -> Option<Vec2> {
    let hx = arena_w * 0.5 - radius;
    let hy = arena_h * 0.5 - radius;
    let mut n = Vec2::ZERO;
    if pos.x > hx {
        n.x = 1.0;
    } else if pos.x < -hx {
        n.x = -1.0;
    }
    if pos.y > hy {
        n.y = 1.0;
    } else if pos.y < -hy {
        n.y = -1.0;
    }
    if n == Vec2::ZERO {
        None
    } else {
        Some(n.normalize_or_zero())
    }
}

pub fn bounce_velocity(vel: Vec2, normal: Vec2) -> Vec2 {
    vel - 2.0 * vel.dot(normal) * normal
}

pub fn should_despawn_after_hit(
    damaged: bool,
    pierce_left_before: Option<u8>,
) -> (bool, Option<u8>) {
    match pierce_left_before {
        Some(n) if n > 0 && damaged => (false, Some(n - 1)),
        Some(n) if n > 0 && !damaged => (false, Some(n)),
        Some(0) | None if damaged => (true, pierce_left_before.map(|_| 0)),
        _ => (true, pierce_left_before),
    }
}

pub fn record_hit(set: &mut Vec<Entity>, target: Entity) -> bool {
    if set.contains(&target) {
        false
    } else {
        set.push(target);
        true
    }
}
