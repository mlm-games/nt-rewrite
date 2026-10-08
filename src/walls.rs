//! Wall breaking + throne-room props. Positions are [`Pos`] (`Vec2`), the
//! stand-in for GML's plain `x`/`y`.
//! Render split: the floor-sprite entity spawned per broken wall and the
//! throne-room art stay out (renderer resolves floor from the
//! [`FloorMask`]); bursts route through
//! [`crate::effects::spawn_burst`] and trauma through `repame_fx::Trauma`
//! (GML `scr_screenshake`).
//! Pipeline note: the *only* drain of the [`PendingWallBreak`] queue
//! (hammerhead chewing, boss charges, portal clears, delayed boss spawns).
//! Wall entities are `(WallTile, WallCell, Pos)`; the flush matches by cell
//! *or* by proximity (`WALL_PX * 0.75`, 12 px), the port's stand-in for
//! GML's `collision_rectangle` bbox chain in `scrWallDestroy`
//! (`scripts/scrWallDestroy/scrWallDestroy.gml:7-14`).

use bevy_ecs::prelude::*;
use repame_fx::Trauma;

use crate::audio::AudioCue;
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
/// walls, as GML's `scrWallDestroy` chain does).
/// GML `scrWallDestroy` destroys the wall and creates a 16x16 `FloorExplo` at its
/// position - so the hole is one wall cell wide, and any sibling `Wall`s inside
/// the same 32x32 `Floor` neighbour stay solid. `FloorExplo/Create_0:19-27` then
/// re-closes the hole: for each of the 8 neighbours at +/-16 px it creates a new
/// `Wall` wherever there is neither a `Floor` nor a `Wall`, which is what stops
/// the wall ring from degrading when a break has no floor on the far side.
#[allow(clippy::too_many_arguments)]
pub fn apply_pending_wall_breaks(
    mut commands: Commands,
    mut mask: ResMut<FloorMask>,
    mut tops: ResMut<TopSmalls>,
    mut trauma: ResMut<Trauma>,
    run: Res<Run>,
    mut cues: ResMut<Queue<AudioCue>>,
    pending: Query<(Entity, &PendingWallBreak)>,
    walls: Query<(Entity, &WallCell, &Pos), With<WallTile>>,
) {
    // `queue_wall_breaks_along_segment` stamps a marker every `WALL_PX * 0.5`
    // along a charge, so the same wall is named many times per boss swing.
    // `Commands` despawns are deferred, so without this every wall would be
    // burst and shaken once per marker.
    let mut despawned: std::collections::HashSet<Entity> = std::collections::HashSet::new();
    // `scrWallDestroy`'s `do/until` loops on `collision_rectangle` over the dead
    // wall's bbox, so every `Wall` sharing that cell dies with it - and each one
    // would spawn its own `FloorExplo`. `setup.rs` merges `wall_cells` and
    // `small_walls` into one set before spawning and the re-seal is keyed by
    // `live_walls`, so the port never stacks two: `broken_cells` keeps the
    // effects to one per cell and `despawned` still kills a duplicate if one
    // ever appears, which is the state GML reaches routinely and the port must
    // not answer with wall art over a hole.
    let mut broken_cells: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();
    let mut resealed: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();
    // GML `FloorExplo/Create_0:19-27` runs before its `Top` spawns, so the new
    // `TopSmall`s test the re-sealed ring: track the post-break wall set here
    // rather than the deferred entity view.
    let mut live_walls: std::collections::HashSet<(i32, i32)> =
        walls.iter().map(|(_, cell, _)| (cell.0, cell.1)).collect();
    let break_sound = wall_break_sound(crate::worldgen::gml_area_from_run(&run));

    for (marker_e, brk) in &pending {
        commands.entity(marker_e).despawn();

        let mut broken: Vec<((i32, i32), glam::Vec2)> = Vec::new();
        for (wall_e, cell, wpos) in &walls {
            let wpos = wpos.0;
            if (cell.0, cell.1) != brk.cell && wpos.distance(brk.pos) > WALL_PX * 0.75 {
                continue;
            }
            if !despawned.insert(wall_e) {
                continue;
            }

            commands.entity(wall_e).despawn();
            live_walls.remove(&(cell.0, cell.1));
            if !broken_cells.insert((cell.0, cell.1)) {
                continue;
            }

            if brk.spawn_floor {
                // The 16x16 cell, not the 32x32 `Floor` that covers it: GML's
                // `FloorExplo` mask is 16x16 (`mskFloorExplo.yy:74,89`).
                mask.opened.insert((cell.0, cell.1));
                // `FloorExplo/Create_0:3-7`: `scrWallBreakSound()` per break.
                cues.push(AudioCue {
                    name: break_sound,
                    volume: 0.4,
                    variance: 0.2,
                });
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

        // `FloorExplo/Create_0:46-51`: `with (Wall)` inside 32 px destroys any
        // wall standing on a `FloorExplo` (`position_meeting(x, y, FloorExplo)`)
        // - no new hole, no re-seal, no rubble. The same pass re-derives `visible`
        // and `l/r/w/h`; the port reads both from the mask at draw time. The
        // port's own invariants keep this from firing (the re-seal skips opened
        // cells and every wall at a broken cell is despawned above), so it only
        // matters if a duplicate ever survives a break - and then it is the
        // difference between rubble and a wall drawn on top of it.
        for ((bx, by), _) in &broken {
            let origin = glam::Vec2::new(*bx as f32 * WALL_PX, *by as f32 * WALL_PX);
            for (wall_e, cell, _) in &walls {
                // `point_distance` is origin-to-origin, so measure the
                // candidate's origin too, not its `Pos` centre.
                let at = glam::Vec2::new(cell.0 as f32 * WALL_PX, cell.1 as f32 * WALL_PX);
                if despawned.contains(&wall_e)
                    || at.distance(origin) > 32.0
                    || !mask.opened.contains(&(cell.0, cell.1))
                {
                    continue;
                }
                despawned.insert(wall_e);
                commands.entity(wall_e).despawn();
                live_walls.remove(&(cell.0, cell.1));
            }
        }
    }
}

/// GML `scrWallBreakSound` (`scripts/scrWallBreakSound`): `GameCont.area` picks
/// the bank, `snd_play_pitchvol(_snd, 0.2, 0.4)` plays it at 0.4 with +/-0.2
/// pitch.
fn wall_break_sound(gml_area: i32) -> &'static str {
    match gml_area {
        // area_sewers, area_palace, area_pizza_sewers, area_mansion, area_crib
        2 | 7 | 102 | 103 | 107 => "sndWallBreakBrick",
        3 => "sndWallBreakScrap",
        // area_caves, area_cursed_caves
        4 | 104 => "sndWallBreakCrystal",
        6 => "sndWallBreakLabs",
        105 => "sndWallBreakJungle",
        101 => "sndOasisExplosionSmall",
        _ => "sndWallBreak",
    }
}

/// Queue breaks for every wall whose box is within `radius` of `pos`. Port-only:
/// GML has no radius break, it destroys one wall per collision
/// (`BanditBoss/Collision_Wall.gml:4-6` and `Van/Collision_Wall.gml:4` both
/// call `scrWallDestroy(other.id)`), so the radius is the port's batch
/// stand-in.
/// Walls arrive as a `(center, cell)` snapshot so callers don't thread
/// queries through helpers (same convention as `boss_ai`).
///
/// `radius` is the mover's body radius and the test is circle-vs-wall-box, like
/// the GML event it stands in for: a `Wall` is a 16x16 solid, so contact starts
/// at `radius + WALL_PX * 0.5` from the wall centre (`+ radius + 8 * sqrt(2)`
/// at a corner). Measuring centre-to-centre instead left that whole band
/// unbroken, so a charging boss visibly pressed through a wall and left it
/// standing.
pub fn queue_wall_breaks_in_radius(
    commands: &mut Commands,
    walls: &[(glam::Vec2, (i32, i32))],
    pos: glam::Vec2,
    radius: f32,
) {
    let half = glam::Vec2::splat(WALL_PX * 0.5);
    for (wpos, cell) in walls {
        let closest = glam::Vec2::new(
            pos.x.clamp(wpos.x - half.x, wpos.x + half.x),
            pos.y.clamp(wpos.y - half.y, wpos.y + half.y),
        );
        if pos.distance(closest) > radius {
            continue;
        }
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

/// Queue breaks along a segment by stamping the radius helper every
/// `WALL_PX * 0.5` px (8 px). Port-only, same law as the radius helper:
/// GML's charge only breaks the wall it collides with
/// (`BanditBoss/Collision_Wall.gml:4-6`). The stamp radius is the body
/// radius, so the sweep covers the same walls a stepped circle would.
pub fn queue_wall_breaks_along_segment(
    commands: &mut Commands,
    walls: &[(glam::Vec2, (i32, i32))],
    from: glam::Vec2,
    to: glam::Vec2,
    radius: f32,
) {
    let delta = to - from;
    let len = delta.length().max(1.0);
    let dir = delta / len;
    let steps = (len / (WALL_PX * 0.5)).ceil() as i32;
    for i in 0..=steps {
        let p = from + dir * (i as f32 * WALL_PX * 0.5);
        queue_wall_breaks_in_radius(commands, walls, p, radius);
    }
}

/// Segment-vs-wall test over the [`FloorMask`]: 12 px samples,
/// arena-exterior samples ignored. Port-only discretisation - GML traces
/// the ray exactly, `collision_line(x, y, target.x, target.y, Wall, true,
/// true)` (`scripts/scrTargetIsVisible/scrTargetIsVisible.gml:11`,
/// `Bandit/Alarm_1.gml:5`).
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

/// Segment-vs-wall test against wall entities, sampling the same 12 px
/// ray GML traces exactly (see [`segment_hits_wall`]). Thin wrapper over
/// [`segment_hits_wall_legacy`].
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
/// (8.8 px) of any wall center. Port-only, same discretisation as
/// [`segment_hits_wall`].
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

/// Carpet occupancy flag for throne-room logic (AABB test). Port-only:
/// GML's `Carpet` has no event code and no player-on-carpet flag, its
/// only reader being the footstep material
/// (`scripts/scrFootSteps/scrFootSteps.gml:17`).
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
