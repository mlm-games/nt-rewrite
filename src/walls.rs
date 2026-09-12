//! Wall breaking + throne-room props. Ported from the bevy reference
//! `game/walls.rs` (`apply_pending_wall_breaks`,
//! `queue_wall_breaks_in_radius`, `queue_wall_breaks_along_segment`,
//! `segment_hits_wall`, `segment_hits_wall_query`,
//! `segment_hits_wall_legacy`, `reset_hammerhead_budget`,
//! `CountedGenerator`, `handle_throne_room_props`,
//! `update_carpet_occupancy`) with positions as [`Pos`] (`Vec2`)
//! instead of `Transform.translation`.
//!
//! Render split: the floor-sprite entity spawned per broken wall and the
//! throne-room art stay out (renderer resolves floor from the
//! [`FloorMask`]); bursts route through [`crate::effects::spawn_burst`]
//! and trauma through `repame_fx::Trauma`, matching the bevy
//! `VfxSpawner`/`ScreenEffects` call sites one-for-one.
//!
//! Pipeline note: this is the *only* drain of the existing
//! [`PendingWallBreak`] queue (spawned by hammerhead chewing,
//! boss charges, portal clears, delayed boss spawns). Wall entities are
//! `(WallTile, WallCell, Pos)` plus an optional [`WallVisuals`] parts
//! list; the flush matches by cell *or* by proximity (`WALL_PX * 0.75`),
//! byte-identical to bevy.

use bevy_ecs::prelude::*;
use repame_fx::Trauma;

use crate::combat::PendingEnemySpawn;
use crate::comps_a::{
    ARENA_H, ARENA_W, FloorMask, FloorStarted, GameCleanup, HammerheadBudget, Health, LevelCleanup,
    PendingWallBreak, Player, Run, Toast, WallCell, WallTile, WallVisuals,
};
use crate::comps_b::{
    BigGenerator, BossBrain, Enemy, Prop, ThroneCarpet, ThroneRoomState, ThroneStatueProp,
};
use crate::data::EnemyKind;
use crate::effects::spawn_burst;
use crate::msg::Queue;
use crate::spatial::Pos;
use crate::worldgen::WALL_PX;

/// Floor cell owning a wall cell (bevy `world::floor_cell_for_wall`
/// parity: walls are half-resolution, `div_euclid(2)` maps back).
#[inline]
pub fn floor_cell_for_wall(wx: i32, wy: i32) -> (i32, i32) {
    (wx.div_euclid(2), wy.div_euclid(2))
}

/// Flush queued wall breaks: despawn the marker, then every wall that
/// matches by cell or sits within `WALL_PX * 0.75` of the break point
/// (one marker can break several walls, bevy parity). Broken walls open
/// their owner floor cell, pop a dust burst, and add 0.06 trauma each.
pub fn apply_pending_wall_breaks(
    mut commands: Commands,
    mut mask: ResMut<FloorMask>,
    mut trauma: ResMut<Trauma>,
    pending: Query<(Entity, &PendingWallBreak)>,
    walls: Query<(Entity, &WallCell, &Pos, Option<&WallVisuals>), With<WallTile>>,
) {
    for (marker_e, brk) in &pending {
        commands.entity(marker_e).despawn();

        for (wall_e, cell, wpos, visuals) in &walls {
            let wpos = wpos.0;
            if (cell.0, cell.1) != brk.cell && wpos.distance(brk.pos) > WALL_PX * 0.75 {
                continue;
            }

            if let Some(visuals) = visuals {
                for part in &visuals.parts {
                    commands.entity(*part).despawn();
                }
            }
            commands.entity(wall_e).despawn();

            if brk.spawn_floor {
                mask.cells.insert(floor_cell_for_wall(cell.0, cell.1));
                // Visual-only floor sprite entity omitted: the renderer
                // draws floor from `FloorMask`, which now covers the cell.
            }

            let mut rng = rand::rng();
            spawn_burst(
                &mut commands,
                &mut rng,
                wpos,
                8,
                [0.7, 0.65, 0.55, 1.0],
                (40.0, 140.0),
            );
            trauma.add(0.06);
        }
    }
}

/// Queue breaks for every wall in `radius` of `pos` (bevy parity).
/// Walls arrive as a `(center, cell)` snapshot so callers don't thread
/// queries through helpers (same convention as `boss_ai`).
pub fn queue_wall_breaks_in_radius(
    commands: &mut Commands,
    walls: &[(glam::Vec2, (i32, i32))],
    pos: glam::Vec2,
    radius: f32,
) {
    for (wpos, cell) in walls {
        if wpos.distance(pos) <= radius {
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                PendingWallBreak {
                    cell: *cell,
                    pos: *wpos,
                    spawn_floor: true,
                },
            ));
        }
    }
}

/// Queue breaks along a segment by stamping the radius helper every
/// `WALL_PX * 0.5` px (bevy parity).
pub fn queue_wall_breaks_along_segment(
    commands: &mut Commands,
    walls: &[(glam::Vec2, (i32, i32))],
    from: glam::Vec2,
    to: glam::Vec2,
    half_width: f32,
) {
    let delta = to - from;
    let len = delta.length().max(1.0);
    let dir = delta / len;
    let steps = (len / (WALL_PX * 0.5)).ceil() as i32;
    for i in 0..=steps {
        let p = from + dir * (i as f32 * WALL_PX * 0.5);
        queue_wall_breaks_in_radius(commands, walls, p, half_width);
    }
}

/// Segment-vs-wall test over the [`FloorMask`] (bevy parity: 12 px
/// samples, arena-exterior samples ignored).
pub fn segment_hits_wall(a: glam::Vec2, b: glam::Vec2, mask: &FloorMask) -> bool {
    let delta = b - a;
    let len = delta.length();
    if len < 1.0 {
        return false;
    }
    let dir = delta / len;
    let steps = (len / 12.0).ceil() as i32;
    for i in 1..steps.max(1) {
        let p = a + dir * (i as f32 * 12.0);
        if !mask.is_walkable(p) && p.x.abs() < ARENA_W / 2.0 && p.y.abs() < ARENA_H / 2.0 {
            return true;
        }
    }
    false
}

/// Segment-vs-wall test against wall entities (bevy
/// `segment_hits_wall_query` parity, which delegates to the legacy
/// entity walk). Thin wrapper over [`segment_hits_wall_legacy`].
pub fn segment_hits_wall_query(
    a: glam::Vec2,
    b: glam::Vec2,
    walls: &Query<(&WallCell, &Pos), With<WallTile>>,
) -> bool {
    let positions: Vec<glam::Vec2> = walls.iter().map(|(_, p)| p.0).collect();
    segment_hits_wall_legacy(a, b, &positions)
}

/// Legacy entity walk core, pure over wall centers so tests feed
/// fixtures without a world: 12 px samples, hit within `WALL_PX * 0.55`
/// of any wall center.
pub fn segment_hits_wall_legacy(
    a: glam::Vec2,
    b: glam::Vec2,
    wall_positions: &[glam::Vec2],
) -> bool {
    let delta = b - a;
    let len = delta.length();
    if len < 1.0 {
        return false;
    }
    let dir = delta / len;
    let steps = (len / 12.0).ceil() as i32;
    for i in 1..steps.max(1) {
        let p = a + dir * (i as f32 * 12.0);
        for wpos in wall_positions {
            if wpos.distance(p) < WALL_PX * 0.55 {
                return true;
            }
        }
    }
    false
}

/// Floor-start reset: any queued [`FloorStarted`] restores the
/// hammerhead chew budget (20) and clears throne-room progress.
pub fn reset_hammerhead_budget(
    mut started: ResMut<Queue<FloorStarted>>,
    mut budget: ResMut<HammerheadBudget>,
    mut throne: ResMut<ThroneRoomState>,
) {
    if !started.drain().is_empty() {
        budget.remaining = HammerheadBudget::default().remaining;
        throne.reset();
    }
}

/// Marker for an already-counted generator (prevents double counting
/// while the dead prop entity lingers).
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct CountedGenerator;

/// Throne-room prop deaths: destroyed big generators announce
/// `GENERATOR x/y` and, once all are down, halve the Throne (loop 0
/// only) with a `THE THRONE WEAKENS` toast; destroyed statues pop their
/// guardian ring as deferred spawns (bevy parity, drained by the shared
/// [`PendingEnemySpawn`] flush).
pub fn handle_throne_room_props(
    mut commands: Commands,
    mut throne_room: ResMut<ThroneRoomState>,
    mut toast: ResMut<Toast>,
    run: Res<Run>,
    mut bosses: Query<(&Enemy, &mut Health), With<BossBrain>>,
    props: Query<
        (
            Entity,
            &Prop,
            Option<&BigGenerator>,
            Option<&ThroneStatueProp>,
            &Pos,
            Option<&CountedGenerator>,
        ),
    >,
) {
    for (e, prop, big_gen, statue, pos, counted) in &props {
        if prop.hp > 0 {
            continue;
        }

        if big_gen.is_some() {
            if counted.is_some() {
                continue;
            }
            commands.entity(e).insert(CountedGenerator);
            let before = throne_room.generators_destroyed;
            throne_room.note_generator_destroyed();
            if throne_room.generators_destroyed != before {
                toast.show(&format!(
                    "GENERATOR {}/{}",
                    throne_room.generators_destroyed, throne_room.generators_total
                ));
            }
            if throne_room.all_generators_down && !throne_room.halved_throne {
                throne_room.halved_throne = true;
                toast.show("THE THRONE WEAKENS");
                if run.loop_count == 0 {
                    for (enemy, mut hp) in &mut bosses {
                        if enemy.kind == EnemyKind::Throne {
                            hp.hp = (hp.hp / 2).max(1);
                            hp.max = (hp.max / 2).max(1);
                        }
                    }
                }
            }
        }

        if let Some(statue) = statue {
            let center = pos.0;
            for i in 0..statue.guardian_count {
                let ang = i as f32 * std::f32::consts::TAU / statue.guardian_count as f32;
                let p = center + glam::Vec2::new(ang.cos(), ang.sin()) * 36.0;
                commands.spawn(PendingEnemySpawn {
                    kind: EnemyKind::PalaceGuardian,
                    pos: p,
                    difficulty: 1.0,
                    loops: run.loop_count,
                });
            }
        }
    }
}

/// Carpet occupancy flag for throne-room logic (AABB test, bevy parity).
pub fn update_carpet_occupancy(
    mut throne_room: ResMut<ThroneRoomState>,
    player_q: Query<&Pos, With<Player>>,
    carpets: Query<(&Pos, &ThroneCarpet)>,
) {
    throne_room.player_on_carpet = false;
    let Ok(player) = player_q.single() else {
        return;
    };
    let p = player.0;
    for (carpet_pos, carpet) in &carpets {
        let c = carpet_pos.0;
        let d = (p - c).abs();
        if d.x <= carpet.half_extents.x && d.y <= carpet.half_extents.y {
            throne_room.player_on_carpet = true;
            break;
        }
    }
}

