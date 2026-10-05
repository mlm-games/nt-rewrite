//! Wall breaking + throne-room props, ported from the bevy reference
//! `game/walls.rs` with positions as [`Pos`] (`Vec2`) instead of
//! `Transform.translation`.
//! Render split: the floor-sprite entity spawned per broken wall and the
//! throne-room art stay out (renderer resolves floor from the [`FloorMask`]);
//! bursts route through [`crate::effects::spawn_burst`] and trauma through
//! `repame_fx::Trauma`, matching the bevy `VfxSpawner`/`ScreenEffects` call
//! sites one-for-one.
//! Pipeline note: the *only* drain of the [`PendingWallBreak`] queue (hammerhead
//! chewing, boss charges, portal clears, delayed boss spawns). Wall entities are
//! `(WallTile, WallCell, Pos)`; the flush matches by cell *or* by proximity
//! (`WALL_PX * 0.75`), byte-identical to bevy.

use bevy_ecs::prelude::*;
use repame_fx::Trauma;

use crate::combat::queue_enemy_spawn;
use crate::comps_a::{
    ARENA_H, ARENA_W, FloorMask, FloorStarted, GameCleanup, HammerheadBudget, Health, LevelCleanup,
    PendingWallBreak, Player, Run, Toast, TopSmalls, WallCell, WallTile,
};
use crate::comps_b::{
    BigGenerator, BossBrain, Enemy, Prop, PropSprites, ThroneCarpet, ThroneRoomState,
    ThroneStatueProp, UnbreakableProp,
};
use crate::data::EnemyKind;
use crate::effects::spawn_burst;
use crate::environment::PropDeathEffect;
use crate::msg::Queue;
use crate::spatial::Pos;
use crate::worldgen::WALL_PX;

pub use crate::comps_a::floor_cell_for_wall;

/// Flush queued wall breaks: every wall matching by cell *or* within
/// `WALL_PX * 0.75` of the break point goes (one marker can break several
/// walls, bevy parity).
/// GML `scrWallDestroy` destroys the wall and creates a 16x16 `FloorExplo` at its
/// position - so the hole is one wall cell wide, and any sibling `Wall`s inside
/// the same 32x32 `Floor` neighbour stay solid. `FloorExplo/Create_0:19-27` then
/// re-closes the hole: for each of the 8 neighbours at +/-16 px it creates a new
/// `Wall` wherever there is neither a `Floor` nor a `Wall`, which is what stops
/// the wall ring from degrading when a break has no floor on the far side.
pub fn apply_pending_wall_breaks(
    mut commands: Commands,
    mut mask: ResMut<FloorMask>,
    mut tops: ResMut<TopSmalls>,
    mut trauma: ResMut<Trauma>,
    pending: Query<(Entity, &PendingWallBreak)>,
    walls: Query<(Entity, &WallCell, &Pos), With<WallTile>>,
) {
    // `queue_wall_breaks_along_segment` stamps a marker every `WALL_PX * 0.5`
    // along a charge, so the same wall is named many times per boss swing. Bevy
    // despawns are deferred, so without this the wall is despawned, burst and
    // shaken once per marker.
    let mut handled: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();
    let mut resealed: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();
    // GML `FloorExplo/Create_0:19-27` runs before its `Top` spawns, so the new
    // `TopSmall`s test the re-sealed ring: track the post-break wall set here
    // rather than the deferred entity view.
    let mut live_walls: std::collections::HashSet<(i32, i32)> =
        walls.iter().map(|(_, cell, _)| (cell.0, cell.1)).collect();

    for (marker_e, brk) in &pending {
        commands.entity(marker_e).despawn();

        let mut broken: Vec<((i32, i32), glam::Vec2)> = Vec::new();
        for (wall_e, cell, wpos) in &walls {
            let wpos = wpos.0;
            if (cell.0, cell.1) != brk.cell && wpos.distance(brk.pos) > WALL_PX * 0.75 {
                continue;
            }
            if !handled.insert((cell.0, cell.1)) {
                continue;
            }

            commands.entity(wall_e).despawn();
            live_walls.remove(&(cell.0, cell.1));

            if brk.spawn_floor {
                // The 16x16 cell, not the 32x32 `Floor` that covers it: GML's
                // `FloorExplo` mask is 16x16 (`mskFloorExplo.yy:74,89`).
                mask.opened.insert((cell.0, cell.1));
            }
            broken.push(((cell.0, cell.1), wpos));

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

        if !brk.spawn_floor {
            continue;
        }

        for broken_cell in &broken {
            let (cx, cy) = broken_cell.0;
            // `FloorExplo/Create_0:19-27`: re-close a hole that has no floor on
            // the far side, so the ring survives a break in the void.
            for (dx, dy) in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (1, -1),
                (1, 1),
                (-1, 1),
            ] {
                let n = (cx + dx, cy + dy);
                if mask.opened.contains(&n) || mask.cells.contains(&floor_cell_for_wall(n.0, n.1)) {
                    continue;
                }
                if live_walls.contains(&n) {
                    continue;
                }
                if broken.iter().any(|((bx, by), _)| (*bx, *by) == n) {
                    continue;
                }
                if !resealed.insert(n) {
                    continue;
                }
                live_walls.insert(n);
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    WallTile,
                    WallCell(n.0, n.1),
                    Prop {
                        size: glam::Vec2::splat(WALL_PX),
                        hp: 9999,
                        destructible: false,
                        explosive: false,
                    },
                    Pos(crate::worldgen::wall_center(n.0, n.1)),
                ));
            }
        }

        // `FloorExplo/Create_0:43-50`: the explosion's own `Top`s extend the
        // Trans ring one step out, and deliberately leave the hole itself
        // bare. Runs after the re-seal so `TopSmall/Create_0`'s `Wall`/`Floor`
        // tests see the new ring and the holes already opened, like GML's
        // event order.
        for broken_cell in &broken {
            tops.spawn_around_break(broken_cell.0, &mask.cells, &live_walls, &mask.opened);
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

/// GML `objects/Nothing/Create_0.gml:5-11` also runs here: the moment a
/// `Nothing` (Throne) exists, every `BigGeneratorInactive` becomes a real
/// `BigGenerator`. Destroyed generators toast `GENERATOR x/y`; all down halves
/// the Throne (loop 0 only) with `THE THRONE WEAKENS`. Destroyed statues pop
/// guardians as deferred spawns (drained by the shared [`PendingEnemySpawn`]
/// flush).
#[allow(clippy::type_complexity)]
pub fn handle_throne_room_props(
    mut commands: Commands,
    mut throne_room: ResMut<ThroneRoomState>,
    mut toast: ResMut<Toast>,
    run: Res<Run>,
    mut bosses: Query<(&Enemy, &mut Health), With<BossBrain>>,
    mut props: Query<(
        Entity,
        &mut Prop,
        Option<&BigGenerator>,
        Option<&ThroneStatueProp>,
        Option<&UnbreakableProp>,
        &Pos,
        Option<&CountedGenerator>,
        Option<Mut<PropSprites>>,
    )>,
) {
    for (e, mut prop, big_gen, statue, unbreakable, pos, counted, mut sprites) in &mut props {
        if unbreakable.is_some() {
            if big_gen.is_some() && !prop.destructible {
                // GML `Nothing/Create_0.gml:5-11`: swap the inactive
                // casing for a live generator (its art and the
                // `max_hp` from `BigGenerator/Create_0.gml:1-4`).
                prop.hp = if run.loop_count == 0 { 230 } else { 50 };
                prop.destructible = true;
                if let Some(sprites) = sprites.as_deref_mut() {
                    sprites.idle = "images/sprBigGenerator.png";
                    sprites.hurt = "images/sprBigGeneratorHurt.png";
                    sprites.dead = "images/sprBigGeneratorDead.png";
                }
                commands.entity(e).remove::<UnbreakableProp>();
                commands.entity(e).insert(PropDeathEffect::big_generator());
            }
            continue;
        }

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

        if statue.is_some() && counted.is_none() {
            commands.entity(e).insert(CountedGenerator);
            // GML `ThroneStatue/Destroy_0.gml:5-7` verbatim:
            // `repeat (1 + GameCont.loops) { instance_create(x, y, Guardian) }`
            // - uncapped, all at the statue's exact position.
            for _ in 0..1 + run.loop_count {
                queue_enemy_spawn(
                    &mut commands,
                    EnemyKind::Guardian,
                    pos.0,
                    1.0,
                    run.loop_count,
                );
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
