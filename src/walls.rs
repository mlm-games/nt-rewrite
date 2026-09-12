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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comps_a::TILE;
    use crate::time::{GTimer, TimerMode};

    /// Schedule-test probe for [`super::segment_hits_wall_query`] (a
    /// `Query` cannot be built outside a system, so the wrapper is
    /// exercised via a system writing here).
    #[derive(Resource, Default)]
    struct HitProbe {
        hit: bool,
        miss: bool,
    }

    fn mask_with(cells: &[(i32, i32)]) -> FloorMask {
        FloorMask {
            cells: cells.iter().copied().collect(),
            cols: 80,
            rows: 52,
        }
    }

    fn spawn_wall(world: &mut World, cell: (i32, i32), pos: glam::Vec2) -> Entity {        world
            .spawn((
                GameCleanup,
                LevelCleanup,
                WallTile,
                WallCell(cell.0, cell.1),
                Pos(pos),
            ))
            .id()
    }

    #[test]
    fn flush_breaks_matched_walls_and_opens_floor() {
        let mut world = World::new();
        world.insert_resource(mask_with(&[(0, 0)]));
        world.insert_resource(Trauma::default());
        let near = spawn_wall(&mut world, (4, 6), glam::Vec2::new(100.0, 100.0));
        let far = spawn_wall(&mut world, (9, 9), glam::Vec2::new(900.0, 900.0));
        world.spawn(PendingWallBreak {
            cell: (4, 6),
            pos: glam::Vec2::new(100.0, 100.0),
            spawn_floor: true,
        });
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(apply_pending_wall_breaks);
        schedule.run(&mut world);

        assert!(world.get_entity(near).is_err(), "matched wall despawned");
        assert!(world.get_entity(far).is_ok(), "distant wall survives");
        assert!(
            world
                .query::<&PendingWallBreak>()
                .iter(&world)
                .next()
                .is_none(),
            "marker drained"
        );
        // Owner floor cell (4,6) -> (2,3) opens.
        assert!(world.resource::<FloorMask>().cells.contains(&(2, 3)));
        assert!((world.resource::<Trauma>().amount - 0.06).abs() < 1e-6);
    }

    #[test]
    fn flush_matches_by_proximity_without_cell_match() {
        let mut world = World::new();
        world.insert_resource(mask_with(&[(0, 0)]));
        world.insert_resource(Trauma::default());
        // Wrong cell, but 5 px from the break point (< 16 * 0.75 = 12).
        let close = spawn_wall(&mut world, (0, 0), glam::Vec2::new(105.0, 100.0));
        world.spawn(PendingWallBreak {
            cell: (7, 7),
            pos: glam::Vec2::new(100.0, 100.0),
            spawn_floor: false,
        });
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(apply_pending_wall_breaks);
        schedule.run(&mut world);
        assert!(world.get_entity(close).is_err());
        // No floor opened when spawn_floor is false.
        assert_eq!(world.resource::<FloorMask>().cells.len(), 1);
    }

    #[test]
    fn flush_despawns_visual_parts() {
        let mut world = World::new();
        world.insert_resource(mask_with(&[(0, 0)]));
        world.insert_resource(Trauma::default());
        let part = world.spawn_empty().id();
        world.spawn((
            GameCleanup,
            LevelCleanup,
            WallTile,
            WallCell(1, 1),
            Pos(glam::Vec2::new(50.0, 50.0)),
            WallVisuals { parts: vec![part] },
        ));
        world.spawn(PendingWallBreak {
            cell: (1, 1),
            pos: glam::Vec2::new(50.0, 50.0),
            spawn_floor: true,
        });
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(apply_pending_wall_breaks);
        schedule.run(&mut world);
        assert!(world.get_entity(part).is_err(), "visual part despawned");
    }

    #[test]
    fn queue_in_radius_and_along_segment() {
        let mut world = World::new();
        let walls = vec![
            (glam::Vec2::new(0.0, 0.0), (0, 0)),
            (glam::Vec2::new(100.0, 0.0), (6, 0)),
        ];
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(
            move |mut commands: Commands| {
                queue_wall_breaks_in_radius(
                    &mut commands,
                    &walls,
                    glam::Vec2::new(10.0, 0.0),
                    20.0,
                );
            },
        );
        schedule.run(&mut world);
        let mut q = world.query::<&PendingWallBreak>();
        assert_eq!(q.iter(&world).count(), 1, "only the near wall queued");

        // Segment stamping reaches the far wall.
        let mut world = World::new();
        let walls = vec![
            (glam::Vec2::new(0.0, 0.0), (0, 0)),
            (glam::Vec2::new(100.0, 0.0), (6, 0)),
        ];
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(
            move |mut commands: Commands| {
                queue_wall_breaks_along_segment(
                    &mut commands,
                    &walls,
                    glam::Vec2::ZERO,
                    glam::Vec2::new(100.0, 0.0),
                    TILE * 0.85,
                );
            },
        );
        schedule.run(&mut world);
        let mut q = world.query::<&PendingWallBreak>();
        assert!(
            q.iter(&world).count() >= 2,
            "segment stamps hit both walls"
        );
    }

    #[test]
    fn segment_mask_hit_and_miss() {
        // Walkable strip along x in [-64, 64]; everything else solid.
        let mask = mask_with(&[(-2, 0), (-1, 0), (0, 0), (1, 0)]);
        assert!(!segment_hits_wall(
            glam::Vec2::new(-64.0, 16.0),
            glam::Vec2::new(64.0, 16.0),
            &mask
        ));
        assert!(segment_hits_wall(
            glam::Vec2::new(-64.0, 200.0),
            glam::Vec2::new(64.0, 200.0),
            &mask
        ));
        assert!(!segment_hits_wall(
            glam::Vec2::ZERO,
            glam::Vec2::new(0.5, 0.0),
            &mask
        ), "degenerate segment never hits");
    }

    #[test]
    fn segment_legacy_hit_and_miss() {
        let walls = vec![glam::Vec2::new(60.0, 0.0)];
        assert!(segment_hits_wall_legacy(
            glam::Vec2::ZERO,
            glam::Vec2::new(120.0, 0.0),
            &walls
        ));
        assert!(!segment_hits_wall_legacy(
            glam::Vec2::ZERO,
            glam::Vec2::new(120.0, 200.0),
            &walls
        ));
        assert!(!segment_hits_wall_legacy(
            glam::Vec2::ZERO,
            glam::Vec2::new(0.5, 0.0),
            &walls
        ));
    }

    #[test]
    fn segment_query_delegates_to_legacy() {
        let mut world = World::new();
        spawn_wall(&mut world, (3, 0), glam::Vec2::new(60.0, 0.0));
        world.insert_resource(HitProbe::default());
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(
            |walls: Query<(&WallCell, &Pos), With<WallTile>>, mut probe: ResMut<HitProbe>| {
                probe.hit = segment_hits_wall_query(
                    glam::Vec2::ZERO,
                    glam::Vec2::new(120.0, 0.0),
                    &walls,
                );
                probe.miss = segment_hits_wall_query(
                    glam::Vec2::ZERO,
                    glam::Vec2::new(120.0, 200.0),
                    &walls,
                );
            },
        );
        schedule.run(&mut world);
        let probe = world.resource::<HitProbe>();
        assert!(probe.hit && !probe.miss);
    }

    #[test]
    fn hammerhead_budget_resets_on_floor_start() {
        let mut world = World::new();
        world.insert_resource(Queue::<FloorStarted>::default());
        world.insert_resource(HammerheadBudget { remaining: 3 });
        let mut throne = ThroneRoomState::default();
        throne.generators_destroyed = 2;
        world.insert_resource(throne);
        world
            .resource_mut::<Queue<FloorStarted>>()
            .push(FloorStarted {
                floor: 2,
                area: crate::data::AreaId::Sewers,
            });
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(reset_hammerhead_budget);
        schedule.run(&mut world);
        assert_eq!(world.resource::<HammerheadBudget>().remaining, 20);
        assert_eq!(
            world.resource::<ThroneRoomState>().generators_destroyed,
            0
        );

        // No event: budget untouched.
        world.resource_mut::<HammerheadBudget>().remaining = 5;
        schedule.run(&mut world);
        assert_eq!(world.resource::<HammerheadBudget>().remaining, 5);
    }

    #[test]
    fn throne_generators_count_halve_and_announce() {
        let mut world = World::new();
        world.insert_resource(ThroneRoomState::default());
        world.insert_resource(Toast::default());
        world.insert_resource(Run::default());
        let throne_e = world
            .spawn((
                Enemy {
                    kind: EnemyKind::Throne,
                    score: 0,
                    touch_damage: 0,
                    rad_drop: 0,
                    drop_chance: 0,
                    weapon_chance: 0,
                },
                BossBrain::new(EnemyKind::Throne, glam::Vec2::ZERO),
                Health {
                    hp: 100,
                    max: 100,
                    invuln: GTimer::from_seconds(0.01, TimerMode::Once),
                },
            ))
            .id();
        let mut gens = Vec::new();
        for i in 0..4 {
            gens.push(
                world
                    .spawn((
                        Prop {
                            size: glam::Vec2::splat(16.0),
                            hp: 0,
                            destructible: true,
                            explosive: false,
                        },
                        BigGenerator { index: i },
                        Pos(glam::Vec2::new(i as f32 * 40.0, 0.0)),
                    ))
                    .id(),
            );
        }
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(handle_throne_room_props);
        schedule.run(&mut world);
        let throne_room = world.resource::<ThroneRoomState>().clone();
        assert_eq!(throne_room.generators_destroyed, 4);
        assert!(throne_room.all_generators_down && throne_room.halved_throne);
        assert_eq!(world.resource::<Toast>().text, "THE THRONE WEAKENS");
        let hp = world.get::<Health>(throne_e).expect("throne alive");
        assert_eq!((hp.hp, hp.max), (50, 50));
        for g in gens {
            assert!(
                world.get::<CountedGenerator>(g).is_some(),
                "counted marker set"
            );
        }
        // Second run: no double count while markers linger.
        schedule.run(&mut world);
        assert_eq!(
            world
                .resource::<ThroneRoomState>()
                .generators_destroyed,
            4
        );
    }

    #[test]
    fn throne_statue_pops_guardian_ring() {
        let mut world = World::new();
        world.insert_resource(ThroneRoomState::default());
        world.insert_resource(Toast::default());
        world.insert_resource(Run::default());
        world.spawn((
            Prop {
                size: glam::Vec2::splat(24.0),
                hp: 0,
                destructible: true,
                explosive: false,
            },
            ThroneStatueProp { guardian_count: 4 },
            Pos(glam::Vec2::ZERO),
        ));
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(handle_throne_room_props);
        schedule.run(&mut world);
        let mut q = world.query::<&PendingEnemySpawn>();
        let spawns: Vec<_> = q.iter(&world).collect();
        assert_eq!(spawns.len(), 4);
        for s in &spawns {
            assert_eq!(s.kind, EnemyKind::PalaceGuardian);
            assert!((s.pos.length() - 36.0).abs() < 1e-4);
        }
    }

    #[test]
    fn carpet_occupancy_flags_player_inside() {
        let mut world = World::new();
        world.insert_resource(ThroneRoomState::default());
        world.spawn((Player::default(), Pos(glam::Vec2::new(5.0, 5.0))));
        world.spawn((
            Pos(glam::Vec2::ZERO),
            ThroneCarpet {
                half_extents: glam::Vec2::new(32.0, 32.0),
            },
        ));
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(update_carpet_occupancy);
        schedule.run(&mut world);
        assert!(world.resource::<ThroneRoomState>().player_on_carpet);

        let mut pq = world.query::<&mut Pos>();
        for mut p in pq.iter_mut(&mut world) {
            if p.0 == glam::Vec2::new(5.0, 5.0) {
                p.0 = glam::Vec2::new(500.0, 500.0);
            }
        }
        schedule.run(&mut world);
        assert!(!world.resource::<ThroneRoomState>().player_on_carpet);
    }
}
