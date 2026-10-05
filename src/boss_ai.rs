//! Boss AI. Ported from the bevy reference `game/boss_ai.rs` with positions
//! as [`Pos`] (`Vec2`) instead of `Transform.translation` (`Vec3`).
//!
//! Render split: `Sprite`/`Anchor`/`Transform` writes, fire-strip swaps
//! (`play_fire`) and `VfxSpawner` bursts stay out; kept: movement impulses,
//! fan/ring volleys with full combat traits, `Explosion` + `Beam` spawns,
//! pending-spawn queues, trauma, wall-break queues. Muzzle markers ride
//! [`show_enemy_fire`] as for normal enemies (no-op without a seeded
//! [`SpriteAnim`]).
//!
//! Timer adaptation: bevy `Timer` -> [`GTimer`] (`tick(dt)`, then
//! `just_finished()`/`finished()`). bevy `short_ready_timer` (finished from
//! birth) has no `GTimer` equivalent: [`ready_timer`] double-ticks a 10 ms
//! `Once` timer into the same observable state (finished, not
//! just-finished).
//!
//! LOS adaptation: bevy traced wall *entities* (`segment_hits_wall_query`);
//! headless uses the `(center, cell)` wall snapshot
//! (`segment_hits_wall_legacy`). Wall *breaking* during charges still
//! queues [`PendingWallBreak`]s against the wall entities.
//!
//! `Collision_Wall` adaptation: the GML event fires on mask overlap after
//! motion, so `boss_wall_law` is a response-only pass
//! (`move_bounce_solid_displacement`, zero displacement) run by the
//! dispatcher after the handler, plus the per-kind destroy/bounce split.
//!
//! the `boss_ai` dispatcher carries 12 params (bevy_ecs 16-param cap), so no
//! record-split; handlers take snapshots (`&[(Vec2, Vec2)]` props,
//! `&[(Vec2, (i32, i32))]` walls) so tests drive them without a world.

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::anim::SpriteAnim;
use crate::audio::{AudioCue, GameAudio};
use crate::combat::{queue_enemy_spawn, queue_enemy_spawn_no_kill};
use crate::comps_a::{
    BossIntro, BouncesLeft, DamageSource, FloorMask, GameCleanup, Health, Hitbox, LevelCleanup,
    NextHurt, PendingWallBreak, Player, Projectile, ProjectileAccel, ProjectileFade,
    ProjectileFriction, ProjectileTyp, RaceState, Run, ShellWallBounce, Team, Toast, TopSmalls,
    Velocity, WallCell, WallTile, gml_motion_add_clamp,
};
use crate::comps_b::{
    Beam, BigGenerator, BossBrain, BossPhase, CustomExplosion, Enemy, EnemyBrain, HitWarning,
    HurtAnim, HyperCrystalArm, HyperOrbitCrystal, HyperState, InvisiWall, MomShot, NecroReviveArea,
    Portal, PortalClear, Prop, PropNestMarkers, PropSprites, SecretEntrance, TechnoVisual,
    TechnomancerState, ThroneBall, ThroneStatueProp,
};
use crate::data::{AreaId, EnemyKind};
use crate::enemies::show_enemy_fire;
use crate::enemy_data::{EnemyDef, enemy_def};
use crate::msg::Queue;
use crate::spatial::{
    Pos, clamp_to_arena, move_bounce_solid, move_bounce_solid_displacement, resolve_prop_collision,
    solid_contact,
};
use crate::time::{GTimer, TimerMode};

/// Evenly spaced fan around `base_angle` (`spread` radians between shots).
pub fn fan_angles(base_angle: f32, count: usize, spread: f32) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![base_angle];
    }
    let center = (count as f32 - 1.0) * 0.5;
    (0..count)
        .map(|i| base_angle + (i as f32 - center) * spread)
        .collect()
}

/// Full-circle ring starting at `phase`.
pub fn ring_angles(count: usize, phase: f32) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    let step = std::f32::consts::TAU / count as f32;
    (0..count).map(|i| phase + step * i as f32).collect()
}

/// Unit vector for an angle.
pub fn dir_from_angle(angle: f32) -> glam::Vec2 {
    glam::Vec2::new(angle.cos(), angle.sin()).normalize_or_zero()
}

/// Orbit-crystal count scales with loop (GML `cnumber = 3 + loops*2`:
/// 3/5/7…).
pub fn hyper_orbit_count(loop_count: u32) -> usize {
    3 + loop_count as usize * 2
}

// Shared firing / spawn helpers (boss-specific thin wrappers; enemy fire
// lives in `crate::enemies` and is NOT duplicated here).

/// Loop-scaled spawn difficulty for boss adds (bevy parity).
pub fn difficulty_for_loop(enraged: bool) -> f32 {
    1.0 + if enraged { 0.25 } else { 0.0 }
}

/// Clamp speed (bevy `limit_velocity` parity).
fn limit_velocity(vel: &mut Velocity, max: f32) {
    if vel.0.length() > max {
        vel.0 = vel.0.normalize_or_zero() * max;
    }
}

/// Finished-from-birth, silent-until-re-armed timer (bevy
/// `short_ready_timer` parity - see module docs).
fn ready_timer() -> GTimer {
    let mut t = GTimer::from_seconds(0.01, TimerMode::Once);
    t.tick(0.01);
    t.tick(0.0);
    t
}

/// GML `alarm[n] = k` (k in 30 Hz steps).
fn gml_alarm(frames: f32) -> GTimer {
    GTimer::from_seconds(frames / 30.0, TimerMode::Once)
}

/// GML `alarm[n] = -1` (negative alarms never fire).
fn gml_alarm_off() -> GTimer {
    GTimer::from_seconds(1.0e6, TimerMode::Once)
}

/// GML `move_bounce_solid(true)` as a response-only pass: the handler has
/// already integrated the step, so the displacement is zero and this only
/// pushes out of solid overlap and reflects the velocity.
fn boss_bounce_solid(
    pos: &mut glam::Vec2,
    vel: &mut glam::Vec2,
    radius: f32,
    props: &[(glam::Vec2, glam::Vec2)],
    mask: &FloorMask,
) -> bool {
    move_bounce_solid_displacement(pos, vel, glam::Vec2::ZERO, radius, props, Some(mask), true)
        .is_some()
}

/// GML `enemy/Collision_Wall.gml:13-39` `busycollisions` slide: step the
/// blocked axis by `friction` px/step until the next step is free.
fn boss_friction_slide(
    pos: glam::Vec2,
    vel: &mut glam::Vec2,
    radius: f32,
    props: &[(glam::Vec2, glam::Vec2)],
    mask: &FloorMask,
    friction: f32,
) {
    if friction <= 0.0 {
        return;
    }
    for horizontal in [true, false] {
        let value = if horizontal { vel.x } else { vel.y };
        if value.abs() <= 1e-6 {
            continue;
        }
        let free = |v: f32| -> bool {
            let step = if horizontal {
                glam::Vec2::new(v, 0.0)
            } else {
                glam::Vec2::new(0.0, v)
            } / 30.0;
            solid_contact(pos + step, radius, props, Some(mask)).is_none()
        };
        if free(value) {
            continue;
        }
        let sign = if value > 0.0 { 1.0 } else { -1.0 };
        let mut current = value;
        for _ in 0..4096 {
            current -= sign * friction * 30.0;
            if current.abs() <= 1e-6 || free(current) {
                break;
            }
        }
        if horizontal {
            vel.x = current
        } else {
            vel.y = current
        }
    }
}

/// GML `Collision_Wall` for the bosses that do not resolve their own. The
/// handler already integrated the step, so the pass is response-only.
#[allow(clippy::too_many_arguments)]
fn boss_wall_law(
    commands: &mut Commands,
    walls: &[(glam::Vec2, (i32, i32))],
    pos: &mut glam::Vec2,
    vel: &mut glam::Vec2,
    kind: EnemyKind,
    radius: f32,
    props: &[(glam::Vec2, glam::Vec2)],
    mask: &FloorMask,
    loops: u32,
) {
    // `busycollisions` is `GameCont.loops <= 3` (`enemy/Create_0.gml:29`).
    let busy = loops <= 3;
    match kind {
        // Handled inside the handler (its own `Collision_Wall.gml`).
        EnemyKind::BigBandit | EnemyKind::BigDog | EnemyKind::Captain => (),
        // `Nothing/Collision_Wall.gml:4-5` and `TechnoMancer/Collision_Wall.gml:1`.
        EnemyKind::Throne | EnemyKind::Technomancer => {
            crate::walls::queue_wall_breaks_in_radius(commands, walls, *pos, radius + 8.0);
            if kind == EnemyKind::Throne {
                boss_bounce_solid(pos, vel, radius, props, mask);
            }
        }
        // Inherits `enemy/Collision_Wall.gml`: bounce plus the friction slide.
        EnemyKind::ThroneII | EnemyKind::LilHunter | EnemyKind::YvBoss => {
            if boss_bounce_solid(pos, vel, radius, props, mask) && busy {
                boss_friction_slide(*pos, vel, radius, props, mask, 0.4);
            }
        }
        // `ScrapBoss`, `HyperCrystal`, `FrogQueen`, `CrownGuardian`: bare
        // `move_bounce_solid(true)`, so no slide.
        _ => {
            boss_bounce_solid(pos, vel, radius, props, mask);
        }
    }
}

/// Queue wall breaks along a charge segment (bevy
/// `queue_wall_breaks_along_segment` parity over the `(center, cell)`
/// snapshot; `WALL_PX = 16` like the bevy build).
fn queue_wall_breaks_along_segment(
    commands: &mut Commands,
    walls: &[(glam::Vec2, (i32, i32))],
    from: glam::Vec2,
    to: glam::Vec2,
    half_width: f32,
) {
    const WALL_PX: f32 = 16.0;
    let delta = to - from;
    let len = delta.length().max(1.0);
    let dir = delta / len;
    let steps = (len / (WALL_PX * 0.5)).ceil() as i32;
    for i in 0..=steps {
        let p = from + dir * (i as f32 * WALL_PX * 0.5);
        for (wpos, cell) in walls {
            if wpos.distance(p) <= half_width {
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
}

/// Single boss projectile (bevy `fire_projectile` parity: team +
/// combat traits + source, no art; bevy boss shots carry no slash `typ`
/// or fade either, so none is added here).
#[allow(clippy::too_many_arguments)]
pub fn fire_projectile(
    commands: &mut Commands,
    owner: Entity,
    pos: glam::Vec2,
    dir: glam::Vec2,
    team: Team,
    speed: f32,
    damage: i32,
    lifetime: f32,
    radius: f32,
    knockback: f32,
    kind: EnemyKind,
) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        team,
        Projectile {
            damage,
            life: GTimer::from_seconds(lifetime, TimerMode::Once),
            radius,
            knockback,
            explosive: false,
            source: Some(DamageSource::enemy(owner, kind)),
        },
        Velocity(dir * speed),
        Pos(pos),
    ));
}

/// Fan volley around `dir` (bevy `fire_fan_with_kind` parity).
#[allow(clippy::too_many_arguments)]
pub fn fire_fan_with_kind(
    commands: &mut Commands,
    owner: Entity,
    pos: glam::Vec2,
    dir: glam::Vec2,
    team: Team,
    count: usize,
    spread: f32,
    speed: f32,
    damage: i32,
    lifetime: f32,
    radius: f32,
    kind: EnemyKind,
) {
    let base = dir.y.atan2(dir.x);
    for angle in fan_angles(base, count, spread) {
        let shot_dir = dir_from_angle(angle);
        fire_projectile(
            commands,
            owner,
            pos + shot_dir * 20.0,
            shot_dir,
            team,
            speed,
            damage,
            lifetime,
            radius,
            120.0,
            kind,
        );
    }
}

/// Fan volley without a boss kind (bevy `fire_fan` parity: `Bandit`
/// source fallback, exactly like bevy's `None` kind).
#[allow(clippy::too_many_arguments)]
pub fn fire_fan(
    commands: &mut Commands,
    owner: Entity,
    pos: glam::Vec2,
    dir: glam::Vec2,
    team: Team,
    count: usize,
    spread: f32,
    speed: f32,
    damage: i32,
    lifetime: f32,
    radius: f32,
) {
    fire_fan_with_kind(
        commands,
        owner,
        pos,
        dir,
        team,
        count,
        spread,
        speed,
        damage,
        lifetime,
        radius,
        EnemyKind::Bandit,
    );
}

/// Full-circle volley (bevy `fire_ring_with_kind` parity).
#[allow(clippy::too_many_arguments)]
pub fn fire_ring_with_kind(
    commands: &mut Commands,
    owner: Entity,
    pos: glam::Vec2,
    team: Team,
    count: usize,
    phase: f32,
    speed: f32,
    damage: i32,
    lifetime: f32,
    radius: f32,
    kind: EnemyKind,
) {
    for angle in ring_angles(count, phase) {
        let dir = dir_from_angle(angle);
        fire_projectile(
            commands,
            owner,
            pos + dir * 22.0,
            dir,
            team,
            speed,
            damage,
            lifetime,
            radius,
            100.0,
            kind,
        );
    }
}

/// Full-circle volley without a boss kind (bevy `fire_ring` parity).
#[allow(clippy::too_many_arguments)]
pub fn fire_ring(
    commands: &mut Commands,
    owner: Entity,
    pos: glam::Vec2,
    team: Team,
    count: usize,
    phase: f32,
    speed: f32,
    damage: i32,
    lifetime: f32,
    radius: f32,
) {
    fire_ring_with_kind(
        commands,
        owner,
        pos,
        team,
        count,
        phase,
        speed,
        damage,
        lifetime,
        radius,
        EnemyKind::Bandit,
    );
}

/// Timed enemy beam (bevy `spawn_enemy_beam` parity; art/rotation resolve
/// renderer-side from `Beam.dir/length/width`).
pub fn spawn_enemy_beam(
    commands: &mut Commands,
    center: glam::Vec2,
    dir: glam::Vec2,
    length: f32,
    width: f32,
    damage: i32,
    duration: f32,
) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        Beam {
            team: Team::Enemy,
            dir,
            length,
            width,
            damage,
            knockback: 40.0,
            // Bevy `spawn_enemy_beam` sprite tint (green, alpha 0.65).
            color: [0.55, 1.0, 0.6, 0.65],
            timer: GTimer::from_seconds(duration, TimerMode::Once),
            tick: GTimer::from_seconds(0.06, TimerMode::Repeating),
            source: None,
        },
        Pos(center),
    ));
}

// Dispatcher.

/// Boss brains: enrage check, timer ticks, per-kind handler, arena clamp
/// (bevy `boss_ai` top to bottom; non-bosses `continue` before the first
/// timer tick, exactly like bevy).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn boss_ai(
    time: Res<SimTime>,
    mut commands: Commands,
    mut run: ResMut<Run>,
    mut trauma: ResMut<Trauma>,
    mut toast: ResMut<Toast>,
    player_q: Query<(&Pos, &Velocity, &RaceState), (With<Player>, Without<Enemy>)>,
    mut bosses: Query<
        (
            Entity,
            &mut Enemy,
            &mut BossBrain,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            &mut Health,
            Option<&mut SpriteAnim>,
            Option<&HurtAnim>,
            Option<&mut TechnomancerState>,
            Option<&mut HyperState>,
        ),
        (With<Enemy>, With<BossBrain>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos, Option<&ThroneStatueProp>), With<Prop>>,
    walls: Query<(&WallCell, &Pos), (With<WallTile>, Without<Enemy>)>,
    children: Query<(Entity, &Enemy, &Pos), (With<Enemy>, Without<BossBrain>)>,
    portals: Query<Entity, With<Portal>>,
    corpses: Query<&Pos, (With<crate::comps_b::Corpse>, Without<Prop>, Without<Enemy>)>,
    wall_ids: Query<(Entity, &Pos), (With<WallTile>, Without<Enemy>)>,
    catalog: Res<repame_anim::AnimCatalog>,
    mask: Res<FloorMask>,
    mut tops: ResMut<TopSmalls>,
) {
    let Ok((player_pos, player_vel, race_state)) = player_q.single() else {
        return;
    };
    let player_pos = player_pos.0;
    let player_velocity = player_vel.0;
    let rogue_present = race_state.race == crate::data::RaceId::Rogue;
    let dt = time.delta_secs;

    // Snapshots so handlers stay plain functions (no query threading).
    let prop_shapes: Vec<(glam::Vec2, glam::Vec2)> =
        props.iter().map(|(_, p, pp, _)| (pp.0, p.size)).collect();
    let statue_list: Vec<(Entity, glam::Vec2)> = props
        .iter()
        .filter(|(_, _, _, s)| s.is_some())
        .map(|(e, _, pp, _)| (e, pp.0))
        .collect();
    let wall_shapes: Vec<(glam::Vec2, (i32, i32))> =
        walls.iter().map(|(c, p)| (p.0, (c.0, c.1))).collect();
    let missile_count = children
        .iter()
        .filter(|(_, e, _)| e.kind == EnemyKind::ScrapBossMissile)
        .count();
    let other_enemies = children.iter().count();
    let foe_list: Vec<Entity> = children.iter().map(|(e, _, _)| e).collect();
    // GML `Nothing/Other_10`: the Throne destroys Portals every step.
    let portal_list: Vec<Entity> = portals.iter().collect();
    // GML `Nothing2/Other_10`: every step turns walls invisible.
    let wall_list: Vec<Entity> = wall_ids.iter().map(|(e, _)| e).collect();
    // GML hatch gate: `Exploder + 2 * SuperFrog < 8` (port Ballguy is
    // GML's Exploder).
    let frog_count = children
        .iter()
        .filter(|(_, e, _)| e.kind == EnemyKind::Ballguy || e.kind == EnemyKind::SuperFrog)
        .map(|(_, e, _)| if e.kind == EnemyKind::SuperFrog { 2 } else { 1 })
        .sum();
    // GML `TechnoMancer/Alarm_1.gml:28` gates emplacing on the live Turret
    // count, and `Alarm_1:13-19` / `Alarm_2` need the nearest Corpse.
    let corpse_list: Vec<glam::Vec2> = corpses.iter().map(|c| c.0).collect();
    // GML holds the shell in `HyperCrystal.crystal[]`; here the crystals are
    // ordinary enemies without a `BossBrain`, so `children` already lists them.
    let crystal_list: Vec<(Entity, glam::Vec2)> = children
        .iter()
        .filter(|(_, e, _)| {
            matches!(
                e.kind,
                EnemyKind::LaserCrystal | EnemyKind::LightningCrystal | EnemyKind::InvLaserCrystal
            )
        })
        .map(|(e, _, pp)| (e, pp.0))
        .collect();
    let live_crystals = crystal_list.len();
    let turret_count = children
        .iter()
        .filter(|(_, e, _)| e.kind == EnemyKind::Turret)
        .count();

    for (
        entity,
        mut enemy,
        mut boss,
        mut brain,
        mut vel,
        mut pos,
        health,
        mut anim,
        hurt,
        mut tech,
        mut hyper,
    ) in &mut bosses
    {
        let kind = enemy.kind;
        let def = enemy_def(kind);
        if !def.boss {
            continue;
        }

        let epos = pos.0;
        let to_player = player_pos - epos;
        let dir = to_player.normalize_or_zero();
        let hurting = hurt.is_some();

        boss.enraged = health.hp <= (health.max / 2).max(1);
        boss.phase_timer.tick(dt);
        boss.attack_timer.tick(dt);
        boss.special_timer.tick(dt);
        brain.melee.tick(dt);

        let fired = match kind {
            EnemyKind::BigBandit => big_bandit_ai(
                &mut commands,
                &mut trauma,
                entity,
                kind,
                &mut boss,
                &mut brain,
                &mut enemy,
                &health,
                &mut vel,
                &mut pos,
                def,
                epos,
                player_pos,
                dir,
                dt,
                &prop_shapes,
                &wall_shapes,
                &mask,
                run.loop_count,
            ),
            EnemyKind::BigDog => big_dog_ai(
                &mut commands,
                &mut trauma,
                entity,
                kind,
                &mut boss,
                &mut brain,
                &mut vel,
                &mut pos,
                &health,
                def,
                epos,
                player_pos,
                dt,
                &prop_shapes,
                &mask,
                run.loop_count,
                missile_count,
            ),
            EnemyKind::LilHunter => lil_hunter_ai(
                &mut commands,
                &mut trauma,
                &mut toast,
                entity,
                kind,
                &mut boss,
                &mut brain,
                &mut vel,
                &mut pos,
                &health,
                def,
                epos,
                player_pos,
                player_velocity,
                dir,
                dt,
                &prop_shapes,
                &wall_shapes,
                run.loop_count,
                other_enemies,
                rogue_present,
                &mask,
                &mut run,
            ),
            EnemyKind::Throne => throne_ai(
                &mut commands,
                &mut trauma,
                hurting,
                entity,
                &mut boss,
                &mut brain,
                &mut vel,
                &mut pos,
                &health,
                epos,
                player_pos,
                run.loop_count,
                dt,
                &foe_list,
                &statue_list,
                &portal_list,
                &wall_shapes,
                def.radius,
            ),
            EnemyKind::ThroneII => {
                // GML `Nothing2/Other_10:12-13` (and
                // `Nothing2Appear/Other_7:3`):
                // `with (TopSmall) instance_destroy()` wipes the whole Trans
                // ring every step the second throne boss is live.
                tops.clear();
                throne_ii_ai(
                    &mut commands,
                    &mut trauma,
                    &mut toast,
                    entity,
                    &mut boss,
                    &mut brain,
                    &mut vel,
                    &mut pos,
                    &health,
                    epos,
                    player_pos,
                    dt,
                    run.loop_count,
                    &wall_list,
                )
            }
            EnemyKind::Hyper => {
                if let Some(h) = hyper.as_deref_mut() {
                    hyper_ai(
                        &mut commands,
                        &mut toast,
                        entity,
                        &mut vel,
                        &mut pos,
                        h,
                        epos,
                        player_pos,
                        crate::enemies::line_of_sight_public(epos, player_pos, &mask),
                        &crystal_list,
                        live_crystals,
                        run.loop_count,
                        &mut run,
                    );
                }
                false
            }
            EnemyKind::FrogQueen => frog_queen_ai(
                &mut commands,
                &mut trauma,
                &mut toast,
                entity,
                &mut boss,
                &mut brain,
                &mut vel,
                &mut pos,
                def,
                epos,
                player_pos,
                dir,
                dt,
                frog_count,
                &prop_shapes,
                &wall_shapes,
                run.loop_count,
            ),
            EnemyKind::Technomancer => {
                if let Some(st) = tech.as_deref_mut() {
                    technomancer_ai(
                        &mut commands,
                        &catalog,
                        &mut toast,
                        st,
                        epos,
                        player_pos,
                        &corpse_list,
                        turret_count,
                        run.loop_count,
                        &mask,
                    );
                }
                false
            }
            EnemyKind::Captain => captain_ai(
                &mut commands,
                &mut trauma,
                &mut toast,
                entity,
                kind,
                &mut boss,
                &mut brain,
                &mut enemy,
                &health,
                &mut vel,
                &mut pos,
                def,
                epos,
                player_pos,
                dir,
                dt,
                &prop_shapes,
                &wall_shapes,
                &mask,
            ),
            EnemyKind::YvBoss => yv_boss_ai(
                &mut commands,
                &mut trauma,
                &mut toast,
                entity,
                &mut boss,
                &mut brain,
                &mut vel,
                &mut pos,
                epos,
                player_pos,
                player_velocity,
                dt,
                &wall_shapes,
            ),
            _ => false,
        };

        if fired {
            show_enemy_fire(
                &mut commands,
                &catalog,
                entity,
                def.sprite,
                anim.as_deref_mut(),
                hurting,
            );
        }

        boss_wall_law(
            &mut commands,
            &wall_shapes,
            &mut pos.0,
            &mut vel.0,
            kind,
            def.radius,
            &prop_shapes,
            &mask,
            run.loop_count,
        );
        clamp_to_arena(&mut pos.0, def.radius);
    }
}

// Big Bandit.

/// Verbatim `objects/BanditBoss` law: decide (`Alarm_1`), shotgun burst
/// (`Alarm_2`), telegraph/dash/recovery (`Alarm_3`..`Alarm_5`), walk + dash
/// (`Other_10`), destroy-or-bounce wall law (`Collision_Wall`).
///
/// register map: `attack_timer`/`special_timer`/`phase_timer` =
/// `alarm[1]`/`alarm[2]`/`alarm[4]`; `phase` = `charge` (Telegraph 0,
/// Charging 1, Cooldown -1); `boss.aux` = `intro`; `pattern_index` =
/// `chargewait`; `brain.fire`/`ammo`/`walk`/`gunangle` = `shot`/`ammo`/
/// `walk`/`gunangle` (radians); `boss.target` = `direction`.
#[allow(clippy::too_many_arguments)]
fn big_bandit_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    owner: Entity,
    kind: EnemyKind,
    boss: &mut BossBrain,
    brain: &mut EnemyBrain,
    enemy: &mut Enemy,
    health: &Health,
    vel: &mut Velocity,
    pos: &mut Pos,
    def: EnemyDef,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    dir: glam::Vec2,
    dt: f32,
    props: &[(glam::Vec2, glam::Vec2)],
    walls: &[(glam::Vec2, (i32, i32))],
    mask: &FloorMask,
    loops: u32,
) -> bool {
    let mut rng = rand::rng();
    let frames = dt * crate::SIM_HZ as f32;
    let mut fired = false;

    // GML `Create_0`: `alarm[1] = 1`, `chargewait = 2`, `charge = 0`,
    // `ammo = 10`, `shot = 0`, `walk = 0`, `meleedamage = 0`, `intro = 0`.
    if brain.burst_left == 0 {
        brain.burst_left = 1;
        boss.phase = BossPhase::Idle;
        boss.attack_timer = gml_alarm(1.0);
        boss.special_timer = gml_alarm_off();
        boss.pattern_index = 2;
        brain.ammo = 10;
        brain.gunangle = rng.random_range(0.0..std::f32::consts::TAU);
        boss.target = -dir;
        enemy.touch_damage = 0;
    }

    // GML `Other_10:3-18`: `if charge` is true for both `charge = 1` and the
    // `charge = -1` recovery, so the dash impulses run through both; the walk
    // adds 2 along `direction` and 1 along `gunangle` capped at 3.
    if boss.phase == BossPhase::Charging || boss.phase == BossPhase::Cooldown {
        gml_motion_add_clamp(&mut vel.0, boss.target, 2.0, 5.0, dt);
        gml_motion_add_clamp(
            &mut vel.0,
            glam::Vec2::from_angle(brain.gunangle),
            2.0,
            5.0,
            dt,
        );
    } else {
        if brain.walk > 0.0 {
            gml_motion_add_clamp(&mut vel.0, boss.target, 2.0, 3.0, dt);
            gml_motion_add_clamp(
                &mut vel.0,
                glam::Vec2::from_angle(brain.gunangle),
                1.0,
                3.0,
                dt,
            );
            brain.walk = (brain.walk - frames).max(0.0);
        }
        limit_velocity(vel, 3.0 * 30.0);
    }

    // GML `Alarm_1` (decide).
    if boss.attack_timer.just_finished() {
        let mut period = 30.0 + rng.random_range(0.0..60.0);
        if loops > 0 {
            period = 20.0 + rng.random_range(0.0..50.0);
        }
        boss.attack_timer = gml_alarm(period);
        enemy.touch_damage = 0;

        let dist = epos.distance(player_pos);
        let los = !crate::walls::segment_hits_wall(epos, player_pos, mask);
        let intro = boss.aux >= 1.0;
        if dist < 240.0 || !intro {
            if los && dist > 48.0 && intro {
                if rng.random_range(0.0..3.0) < 2.0 {
                    brain.ammo = if loops > 0 { 15 } else { 10 };
                    // GML scopes the whole burst arm on `GameCont.loops`, so
                    // on loop 0 only `ammo` is raised and `alarm[2]` is never
                    // armed.
                    if loops > 0 {
                        boss.special_timer = gml_alarm(1.0);
                        brain.gunangle = dir.y.atan2(dir.x);
                        boss.attack_timer = gml_alarm(70.0 + rng.random_range(0.0..5.0));
                    }
                }
            } else if brain.fire > 0 || health.hp < health.max || !intro {
                boss.pattern_index += 1;
                if dist < 96.0 {
                    boss.pattern_index += 1;
                }
                if boss.pattern_index >= 2 || !intro {
                    boss.pattern_index = 0;
                    // `alarm[3] = 1` then `alarm[1] = -1`; `Alarm_3` runs in
                    // the same step and zeroes `walk` (`Alarm_3:3`).
                    boss.phase = BossPhase::Telegraph;
                    boss.phase_timer = gml_alarm(15.0);
                    boss.attack_timer = gml_alarm_off();
                    brain.walk = 0.0;
                    trauma.add(0.08);
                }
            }
        }

        // GML `Alarm_1:29-35`: three draws when far, two when close.
        let away_dir = -dir;
        let base = away_dir.y.atan2(away_dir.x);
        let mut heading = base + rng.random_range(-90.0f32..90.0).to_radians();
        let mut walk = 10.0 + rng.random_range(0.0..10.0);
        if dist > 64.0 {
            walk = 40.0;
            heading = base + rng.random_range(-45.0f32..45.0).to_radians();
        }
        brain.walk = walk;
        boss.target = glam::Vec2::from_angle(heading);
        // GML `Alarm_1:30` `speed = 0.4` overwrites the walk impulses.
        vel.0 = boss.target * (0.4 * 30.0);
    }

    // GML `Alarm_2` (shotgun burst, one `EnemyBullet1` every 4 steps).
    if boss.special_timer.just_finished() {
        if brain.ammo > 0 {
            brain.fire = 1;
            brain.ammo -= 1;
            if brain.ammo == 7 && loops > 0 {
                brain.gunangle = dir.y.atan2(dir.x);
            }
            boss.special_timer = gml_alarm(4.0);
            brain.walk = 0.0;
            gml_motion_add_clamp(
                &mut vel.0,
                glam::Vec2::from_angle(brain.gunangle + std::f32::consts::PI),
                1.0,
                3.0,
                dt,
            );
            let sdir = glam::Vec2::from_angle(
                brain.gunangle + rng.random_range(-15.0f32..=15.0).to_radians(),
            );
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                Team::Enemy,
                Projectile {
                    damage: 3,
                    life: GTimer::from_seconds(3.2, TimerMode::Once),
                    radius: 4.5,
                    knockback: 120.0,
                    explosive: false,
                    source: Some(DamageSource::enemy(owner, kind)),
                },
                ProjectileTyp(1),
                ProjectileFade("images/sprEnemyBulletHit.png"),
                Velocity(sdir * 240.0),
                Pos(epos + sdir * 20.0),
            ));
            fired = true;
        } else {
            // GML `Alarm_2:22`.
            boss.attack_timer = gml_alarm(60.0 + rng.random_range(0.0..10.0));
        }
    }

    // GML `Alarm_4` / `Alarm_5` (dash timing and `meleedamage`).
    match boss.phase {
        BossPhase::Telegraph if boss.phase_timer.just_finished() => {
            enemy.touch_damage = 10;
            boss.phase = BossPhase::Charging;
            boss.phase_timer = gml_alarm(if boss.aux >= 1.0 { 20.0 } else { 5.0 });
            brain.gunangle = dir.y.atan2(dir.x);
            // GML `Alarm_4:12` `motion_add(gunangle, 10)`; `Other_10` runs
            // before the alarms, so the 10 px/step survives this step.
            vel.0 += glam::Vec2::from_angle(brain.gunangle) * (10.0 * 30.0) * frames;
            trauma.add(0.18);
        }
        BossPhase::Charging if boss.phase_timer.just_finished() => {
            boss.phase = BossPhase::Cooldown;
            boss.phase_timer = gml_alarm(10.0);
        }
        BossPhase::Cooldown if boss.phase_timer.just_finished() => {
            boss.phase = BossPhase::Idle;
            boss.aux = 1.0;
            boss.attack_timer = gml_alarm(45.0 + rng.random_range(0.0..30.0));
        }
        _ => (),
    }
    if boss.phase != BossPhase::Charging {
        enemy.touch_damage = 0;
    }

    // GML `Collision_Wall.gml:4-8`: `charge > 0 || !intro` destroys the tile.
    // Only the tile-breaking branch needs the swept segment, so only it
    // pre-moves; `move_bounce_solid` does its own translation and doing
    // both integrated `vel * dt` twice.
    if boss.phase == BossPhase::Charging || boss.aux == 0.0 {
        let before = pos.0;
        pos.0 += vel.0 * dt;
        queue_wall_breaks_along_segment(commands, walls, before, pos.0, def.radius * 0.9);
    } else {
        move_bounce_solid(
            &mut pos.0,
            &mut vel.0,
            def.radius,
            dt,
            props,
            Some(mask),
            true,
        );
    }
    resolve_prop_collision(&mut pos.0, def.radius, props.iter().copied());
    fired
}

// Scrap Boss (Big Dog).

/// Verbatim `objects/ScrapBoss` law: decide tick (`Alarm_0`), spin-fire
/// tick (`Alarm_1`), walk/homing (`Other_10`).
///
/// register map: `attack_timer`/`special_timer` = `alarm[0]`/`alarm[1]`;
/// `brain.ammo`/`walk`/`gunangle` = `ammo`/`walk`/`gunangle` (radians);
/// `boss.aux` = `turn`; `boss.target` = move heading.
#[allow(clippy::too_many_arguments)]
fn big_dog_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    owner: Entity,
    kind: EnemyKind,
    boss: &mut BossBrain,
    brain: &mut EnemyBrain,
    vel: &mut Velocity,
    pos: &mut Pos,
    health: &Health,
    def: EnemyDef,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    dt: f32,
    props: &[(glam::Vec2, glam::Vec2)],
    mask: &FloorMask,
    loop_count: u32,
    missiles: usize,
) -> bool {
    let mut rng = rand::rng();
    let mut fired = false;

    // GML `Create_0`: `alarm[0] = 30`, `ammo = 15`, `turn = ±1`.
    if boss.aux == 0.0 && brain.ammo == 0 {
        boss.aux = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        brain.ammo = 15;
        brain.gunangle = (player_pos - epos).y.atan2((player_pos - epos).x);
        boss.attack_timer = GTimer::from_seconds(1.0 /* 30 ticks */, TimerMode::Once);
    }

    let to_player = player_pos - epos;

    // GML only ever loads `alarm[1]` from the spin branch of `Alarm_0`, so
    // the decide tick has to disarm it explicitly; `BossBrain` starts it as a
    // free-running repeat.
    if boss.attack_timer.just_finished() {
        boss.special_timer = GTimer::disarmed();
    }

    // GML `Alarm_1` (spin fire).
    if boss.special_timer.is_finished() && boss.special_timer.just_finished() {
        if brain.ammo > 0 {
            brain.ammo -= 1;
            // Drift toward the target fanned by the spin direction.
            let drift = to_player.y.atan2(to_player.x) + boss.aux * 80.0_f32.to_radians();
            gml_motion_add_clamp(&mut vel.0, glam::Vec2::from_angle(drift), 0.3, 3.0, dt);
            let count = 6 + loop_count as usize;
            let step = 360.0 / count as f32;
            for _ in 0..count {
                let sdir = glam::Vec2::from_angle(brain.gunangle);
                // GML spawn offset: `(24·cos g, 16·sin g)`.
                let off = glam::Vec2::new(24.0 * brain.gunangle.cos(), 16.0 * brain.gunangle.sin());
                fire_projectile(
                    commands,
                    owner,
                    epos + off,
                    sdir,
                    Team::Enemy,
                    60.0,
                    3,
                    3.0,
                    4.0,
                    120.0,
                    kind,
                );
                brain.gunangle += step.to_radians();
            }
            brain.gunangle += (4.0 * boss.aux).to_radians();
            fired = true;
            boss.special_timer = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
        } else {
            boss.attack_timer = GTimer::from_seconds(20.0 / 30.0, TimerMode::Once);
        }
    }

    // GML `Alarm_0` (decide).
    if boss.attack_timer.just_finished() {
        if rng.random::<f32>() < 1.0 / 3.0 {
            // Spin attack.
            boss.special_timer = GTimer::from_seconds(15.0 / 30.0, TimerMode::Once);
            // GML keeps `ammo` real: `10 + 10 * (1 - hp / max_hp)` spent one
            // per `Alarm_1`, so the volley is the ceiling of that value.
            let frac = (health.hp as f32 / health.max.max(1) as f32).clamp(0.0, 1.0);
            brain.ammo = (10.0 + 10.0 * (1.0 - frac)).ceil() as u8;
            boss.aux = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            brain.walk = 0.0;
            vel.0 = glam::Vec2::ZERO;
        } else {
            brain.ammo = 0;
            if rng.random::<f32>() < 1.0 / (3.0 + missiles as f32 / 2.0) {
                for _ in 0..3 {
                    queue_enemy_spawn_no_kill(
                        &mut *commands,
                        EnemyKind::ScrapBossMissile,
                        epos,
                        1.0,
                        loop_count,
                    );
                }
                boss.attack_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
            } else {
                brain.walk = rng.random_range(20.0..30.0);
                let head = glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU));
                gml_motion_add_clamp(&mut vel.0, head, 1.0, 3.0, dt);
                boss.target = head;
                boss.attack_timer =
                    GTimer::from_seconds((brain.walk + 10.0) / 30.0, TimerMode::Once);
            }
        }
    }

    // GML `Other_10` locomotion.
    if vel.0.length() > 90.0 {
        vel.0 = vel.0.normalize() * 90.0;
    }
    if brain.walk > 0.0 {
        if vel.0.length() > 60.0 {
            vel.0 = vel.0.normalize() * 60.0;
        }
        gml_motion_add_clamp(&mut vel.0, boss.target, 0.5, 2.0, dt);
        gml_motion_add_clamp(&mut vel.0, to_player.normalize_or_zero(), 0.5, 2.0, dt);
        brain.walk -= dt * 30.0;
        if brain.walk < 0.0 {
            brain.walk = 0.0;
        }
    }
    // GML `Other_10:32`: `speed = 1` overwrites whatever the walk law built.
    if brain.ammo > 0 {
        vel.0 = vel.0.normalize_or_zero() * 30.0;
    }
    move_bounce_solid(
        &mut pos.0,
        &mut vel.0,
        def.radius,
        dt,
        props,
        Some(mask),
        true,
    );
    let _ = trauma;
    fired
}

// Lil Hunter.

/// Verbatim `objects/LilHunter` + `objects/LilHunterFly` law: bouncer fan
/// and sniper hose (`Alarm_1`), anti-camp liftoff (`Alarm_2`), walk/dodge
/// (`Other_10`), teleport flight with landing fire ring, 80-flame death
/// ring (`Destroy_0`).
///
/// register map: `attack_timer`/`special_timer` = `alarm[1]`/`alarm[2]`;
/// `brain.ammo` = `spawns`; `brain.burst_left` = init/intro flag;
/// `brain.walk`/`dash`/`gunangle` = `walk`/`dodge`/`gunangle` (radians);
/// `boss.target` = move heading; `boss.aux` = fly height `z`; `boss.phase`
/// Teleport = airborne (`pattern_index` 1 ascend, 2 descend). (Rogue
/// force-liftoff, taunts, `LilHunterDie`/music cues out as meta/visual.)
#[allow(clippy::too_many_arguments)]
fn lil_hunter_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    toast: &mut Toast,
    owner: Entity,
    kind: EnemyKind,
    boss: &mut BossBrain,
    brain: &mut EnemyBrain,
    vel: &mut Velocity,
    pos: &mut Pos,
    health: &Health,
    def: EnemyDef,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    player_velocity: glam::Vec2,
    dir: glam::Vec2,
    dt: f32,
    props: &[(glam::Vec2, glam::Vec2)],
    walls: &[(glam::Vec2, (i32, i32))],
    loop_count: u32,
    others: usize,
    rogue_present: bool,
    mask: &FloorMask,
    run: &mut Run,
) -> bool {
    let mut rng = rand::rng();
    let mut fired = false;

    // GML `Create_0`: `spawns = 6`, `alarm[1] = 30..120`, `alarm[2] = 30`.
    if brain.burst_left == 0 {
        brain.burst_left = 1;
        brain.ammo = 6;
        boss.attack_timer =
            GTimer::from_seconds(rng.random_range(30.0..=120.0) / 30.0, TimerMode::Once);
        boss.special_timer = GTimer::from_seconds(1.0 /* 30 ticks */, TimerMode::Once);
    }

    let to_player = player_pos - epos;
    let dist = to_player.length();
    let aim = to_player.y.atan2(to_player.x);

    if boss.phase == BossPhase::Teleport {
        if boss.pattern_index == 1 {
            // Ascend 8 px/tick until offscreen (port: height past 160).
            boss.aux -= 240.0 * dt;
            if boss.aux <= -160.0 {
                if others == 0 {
                    // GML `LilHunterFly/Step_0:5-31`: the last boss of its
                    // kind leaves an `IDPDSpawn` on the way out.
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        crate::comps_b::LilHunterEscape { at: epos },
                        Pos(epos),
                    ));
                    commands.entity(owner).despawn();
                    return false;
                }
                pos.0 = player_pos;
                if rng.random::<f32>() < 1.0 / 3.0 {
                    let a = rng.random_range(0.0..std::f32::consts::TAU);
                    pos.0 += glam::Vec2::from_angle(a) * 120.0;
                }
                boss.pattern_index = 2;
            }
        } else {
            // Land 10 px/tick.
            boss.aux += 300.0 * dt;
            if boss.aux >= 0.0 {
                boss.aux = 0.0;
                trauma.add(0.2);
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    PortalClear {
                        timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                        scale: 1.0,
                    },
                    Pos(pos.0),
                ));
                lil_hunter_fire_ring(commands, pos.0, props, Some(mask));
                fired = true;
                boss.attack_timer =
                    GTimer::from_seconds(rng.random_range(20.0..=30.0) / 30.0, TimerMode::Once);
                if brain.burst_left == 1 {
                    brain.burst_left = 2;
                    toast.show("LIL HUNTER");
                    if loop_count == 0 {
                        commands.spawn((
                            GameCleanup,
                            BossIntro {
                                timer: GTimer::from_seconds(1.1, TimerMode::Once),
                            },
                        ));
                    }
                }
                boss.phase = BossPhase::Idle;
            }
        }
        return fired;
    }

    let wall_centers: Vec<glam::Vec2> = walls.iter().map(|(c, _)| *c).collect();
    let los =
        dist < 512.0 && !crate::walls::segment_hits_wall_legacy(epos, player_pos, &wall_centers);

    // GML `Alarm_2` (anti-camp).
    if boss.special_timer.just_finished() {
        if player_velocity.length_squared() < 1.0 {
            boss.special_timer = GTimer::from_seconds(2.0 / 30.0, TimerMode::Once);
        } else {
            boss.phase = BossPhase::Teleport;
            boss.pattern_index = 1;
        }
    }

    // GML `Alarm_1` (brain).
    if boss.attack_timer.just_finished() {
        boss.attack_timer =
            GTimer::from_seconds((20.0 + rng.random_range(0.0..=6.0)) / 30.0, TimerMode::Once);
        // GML: Rogue in the run with no other enemies forces liftoff.
        let liftoff = rogue_present && others == 0
            || rng.random::<f32>() < 1.0 / 30.0
            || (dist < 64.0 && rng.random::<f32>() < 2.0 / 3.0)
            || (dist > 160.0 && rng.random::<f32>() < 1.0 / 16.0);
        if liftoff {
            boss.phase = BossPhase::Teleport;
            boss.pattern_index = 1;
        } else if los {
            if rng.random::<f32>() < 3.0 / 4.0 {
                if dist < 140.0 {
                    // Bouncer fan: 11 + loops shots from
                    // `gunangle - 50 - loops * 10`, stepping 10 degrees.
                    brain.gunangle = aim + rng.random_range(-25.0..=25.0_f32).to_radians();
                    let mut addang = -50.0 - loop_count as f32 * 10.0;
                    for _ in 0..11 + loop_count as usize {
                        let ang = brain.gunangle + addang.to_radians();
                        let sdir = glam::Vec2::from_angle(ang);
                        fire_projectile(
                            commands,
                            owner,
                            epos + sdir * 20.0,
                            sdir,
                            Team::Enemy,
                            90.0,
                            3,
                            3.0,
                            4.0,
                            120.0,
                            kind,
                        );
                        addang += 10.0;
                    }
                    fired = true;
                    let d = boss.attack_timer.duration() / (1.0 + loop_count as f32 * 2.0);
                    boss.attack_timer = GTimer::from_seconds(d.max(1.0 / 30.0), TimerMode::Once);
                    // Retreat from the target.
                    let back = (-dir
                        + glam::Vec2::from_angle(rng.random_range(-10.0..=10.0_f32).to_radians()))
                    .normalize_or_zero();
                    vel.0 = back * 12.0;
                    boss.target = back;
                    brain.walk = 20.0;
                    brain.gunangle = aim;
                } else {
                    // Sniper hose: 10 + 2 * loops bolts at 7..13 px/tick.
                    boss.attack_timer = GTimer::from_seconds(
                        (5.0 + rng.random_range(0.0..=6.0)) / 30.0,
                        TimerMode::Once,
                    );
                    brain.gunangle = aim + rng.random_range(-15.0..=15.0_f32).to_radians();
                    for _ in 0..10 + loop_count as usize * 2 {
                        let sdir = glam::Vec2::from_angle(brain.gunangle);
                        fire_projectile(
                            commands,
                            owner,
                            epos + sdir * 20.0,
                            sdir,
                            Team::Enemy,
                            rng.random_range(210.0..=390.0),
                            3,
                            2.5,
                            4.0,
                            120.0,
                            kind,
                        );
                    }
                    fired = true;
                }
            } else {
                // Break off: short retreat.
                let back = (-dir
                    + glam::Vec2::from_angle(rng.random_range(-10.0..=10.0_f32).to_radians()))
                .normalize_or_zero();
                vel.0 = back * 12.0;
                boss.target = back;
                brain.walk = rng.random_range(8.0..=12.0);
                boss.attack_timer = GTimer::from_seconds(brain.walk / 30.0, TimerMode::Once);
                brain.gunangle = aim;
            }
        } else if rng.random::<f32>() < 0.5
            && brain.ammo > 0
            && (health.hp as f32) < (health.max as f32) * (brain.ammo as f32) / 6.0
        {
            // GML `Alarm_1:70-87`: 30 `IDPDPortalCharge` motes, then
            // `1 + max(0, loops - 1)` `IDPDSpawn`s. Every portal bumps
            // `GameCont.popolevel` in `Create_0` before any of their
            // `Alarm_1`s roll, so the whole batch shares the final level.
            brain.walk = 0.0;
            let waves = 1 + loop_count.saturating_sub(1) as usize;
            run.popolevel += waves as f32;
            for _ in 0..waves {
                for kind in crate::idpd::roll_idpd_table(loop_count, run.area, run.popolevel, true)
                {
                    queue_enemy_spawn(&mut *commands, kind, epos, 1.0, loop_count);
                }
            }
            brain.ammo -= 1;
            let d = boss.attack_timer.duration() + 45.0 / 30.0;
            boss.attack_timer = GTimer::from_seconds(d, TimerMode::Once);
        } else {
            // Reposition: long retreat, brain ticks 3x faster.
            let back = (-dir
                + glam::Vec2::from_angle(rng.random_range(-10.0..=10.0_f32).to_radians()))
            .normalize_or_zero();
            vel.0 = back * 12.0;
            boss.target = back;
            brain.walk = rng.random_range(40.0..=50.0);
            brain.gunangle = aim;
            let d = boss.attack_timer.duration() / 3.0;
            boss.attack_timer = GTimer::from_seconds(d, TimerMode::Once);
        }
        if brain.walk > 0.0 {
            gml_motion_add_clamp(&mut vel.0, -dir, 0.3, 4.0, dt);
        }
    }

    // GML `Other_10` locomotion: always moving, 1..4 px/tick.
    if brain.walk > 0.0 {
        gml_motion_add_clamp(&mut vel.0, boss.target, 0.8, 4.0, dt);
        brain.walk -= dt * 30.0;
        if brain.walk < 0.0 {
            brain.walk = 0.0;
        }
    }
    // Dodge dash: 4 ticks of 6 px/tick along the heading.
    if brain.dash > 0.0 {
        pos.0 += boss.target * 180.0 * dt;
        brain.dash -= dt * 30.0;
    } else if health.hp > 0 && rng.random::<f32>() < dt * 30.0 / 90.0 {
        if dist <= 64.0 {
            boss.target = dir;
        }
        brain.dash = 4.0;
    }
    let sp = vel.0.length();
    if sp > 120.0 {
        vel.0 = vel.0.normalize() * 120.0;
    } else if sp < 30.0 {
        vel.0 = boss.target.normalize_or_zero() * 30.0;
        if vel.0.length_squared() < 0.001 {
            vel.0 = -dir * 30.0;
        }
    }
    pos.0 += vel.0 * dt;
    resolve_prop_collision(&mut pos.0, def.radius, props.iter().copied());
    clamp_to_arena(&mut pos.0, def.radius);
    fired
}

/// GML `LilHunterFly/Step_0:63-77` and `LilHunter/Destroy_0:23-36`: 80
/// `sprFireLilHunter` `TrapFire`s at 2 + random(0.2) px/step, stepping 4.5
/// degrees, each nudged up to 12px clear of geometry along its own heading.
pub fn lil_hunter_fire_ring(
    commands: &mut Commands,
    at: glam::Vec2,
    props: &[(glam::Vec2, glam::Vec2)],
    mask: Option<&FloorMask>,
) {
    let mut rng = rand::rng();
    let mut ang = rng.random_range(0.0..std::f32::consts::TAU);
    for _ in 0..80 {
        ang += 4.5_f32.to_radians();
        let d = glam::Vec2::from_angle(ang);
        let speed = (2.0 + rng.random_range(0.0..0.2)) * 30.0;
        let mut origin = at;
        crate::spatial::move_contact_solid(&mut origin, d * 12.0, 0.0, props, mask);
        crate::environment::spawn_trap_fire_with_image(
            commands,
            origin,
            d,
            speed,
            Team::Enemy,
            None,
            "images/sprFireLilHunter.png",
            Some(d.y.atan2(d.x)),
        );
    }
}

/// Per-kind taunt line (`snd<Kind>Taunt`, GML-verbatim names - e.g.
/// FrogQueen taunts `sndBallMamaTaunt`, YV `sndGunGodTaunt`). Kinds
/// without a GML taunt return `None`.
pub fn taunt_cue_name(kind: EnemyKind) -> Option<&'static str> {
    match kind {
        EnemyKind::BigBandit => Some("sndBigBanditTaunt"),
        EnemyKind::BigDog => Some("sndBigDogTaunt"),
        EnemyKind::Hyper => Some("sndHyperCrystalTaunt"),
        EnemyKind::LilHunter => Some("sndLilHunterTaunt"),
        EnemyKind::Technomancer => Some("sndLastTaunt"),
        EnemyKind::YvBoss => Some("sndGunGodTaunt"),
        EnemyKind::Throne => Some("sndNothingTaunt"),
        EnemyKind::ThroneII => Some("sndNothing2Taunt"),
        _ => None,
    }
}

/// GML boss taunts (`BanditBoss/Other_10:25-30` et al.): with no live
/// Player, `tauntdelay` ticks and the kind's taunt fires once past the
/// gate (>50; only Throne-I/`Nothing` waits >150 per
/// `Nothing/Other_10:52`, `Nothing2` uses 50). YV resets the delay
/// while the player lives (`YVBoss/Step_0:27`).
pub fn tick_boss_taunts(
    mut cues: ResMut<Queue<AudioCue>>,
    players: Query<Entity, With<Player>>,
    mut bosses: Query<(&Enemy, &mut BossBrain)>,
) {
    if players.single().is_ok() {
        for (enemy, mut brain) in &mut bosses {
            if enemy.kind == EnemyKind::YvBoss {
                brain.tauntdelay = 0;
            }
        }
        return;
    }
    for (enemy, mut brain) in &mut bosses {
        if brain.taunt {
            continue;
        }
        let Some(cue_name) = taunt_cue_name(enemy.kind) else {
            continue;
        };
        brain.tauntdelay += 1;
        let gate = if matches!(enemy.kind, EnemyKind::Throne) {
            150
        } else {
            50
        };
        if brain.tauntdelay > gate {
            brain.taunt = true;
            cues.push(AudioCue {
                name: cue_name,
                volume: 1.0,
                variance: 0.0,
            });
        }
    }
}

// Throne.

/// Verbatim `objects/Nothing` law: walk-in brain with statue-break
/// (`Alarm_1`), mirrored Horror-fan triplets (`Alarm_2`), capped stomp
/// (`Other_10`).
///
/// register map: `attack_timer`/`special_timer` = `alarm[1]`/`alarm[2]`;
/// `brain.ammo`/`walk`/`gunangle` = `ammo`/`walk`/`addangle` (radians);
/// `boss.aux` = `mode`; `boss.target` = `walkdir` heading; `pattern_index` =
/// `introwalk`; `brain.burst_left` = `dmg` counter (reset while hurt). The
/// `NothingBeam` charge delay rides `BossPhase::Telegraph`. (Statue art,
/// flame sprites, hurt-voice tiers out as visual/audio.)
#[allow(clippy::too_many_arguments)]
/// GML `objects/Nothing/Collision_prop.gml:4-5` verbatim:
/// `if other.object_index != BigGenerator { other.hp = 0 }` - the Throne body
/// annihilates every prop it overlaps, which is what makes a barrel
/// chain-explode when it walks through one. Statues take the
/// `ThroneStatue/Step_1.gml:11` route instead: direct `instance_destroy` on
/// `place_meeting(x, y, Nothing)`, because they are `canbreak = 0` and so
/// damage-immune, and their `Destroy_0` spawns the guardians. Generators are
/// exempt under the `object_index` test, which also covers
/// `BigGeneratorInactive` (converts to `BigGenerator` in place via
/// `Nothing/Create_0.gml:5-11`).
pub fn throne_annihilate_props(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    save: Res<crate::savedata_part::SaveData>,
    run: Res<Run>,
    mut secrets: ResMut<crate::secrets::SecretTriggers>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    throne_q: Query<(&Pos, &Enemy), (With<Enemy>, Without<Prop>)>,
    statue_q: Query<(Entity, &Pos), (With<ThroneStatueProp>, Without<Player>)>,
    generators: Query<(), (With<BigGenerator>, Without<Player>)>,
    mut props: Query<
        (
            Entity,
            &mut Prop,
            &Pos,
            Option<&crate::environment::PropDeathEffect>,
            Option<&PropSprites>,
            Option<&crate::comps_b::SpecialPropDeath>,
            Option<&mut NextHurt>,
        ),
        With<Prop>,
    >,
    entrances: Query<&SecretEntrance>,
    nests: Query<&PropNestMarkers, With<Prop>>,
    player_q: Query<&Player, With<Player>>,
) {
    let Some((throne_pos, _)) = throne_q.iter().find(|(_, e)| e.kind == EnemyKind::Throne) else {
        return;
    };
    let tpos = throne_pos.0;
    let reach = enemy_def(EnemyKind::Throne).radius;

    for (e, spos) in statue_q.iter() {
        let Ok((_, p, _, _, _, _, _)) = props.get(e) else {
            continue;
        };
        let half = p.size / 2.0;
        if (spos.0 - tpos).abs().max_element() > reach + half.max_element() {
            continue;
        }
        commands.entity(e).despawn();
        for _ in 0..1 + run.loop_count {
            queue_enemy_spawn(
                &mut commands,
                EnemyKind::Guardian,
                spos.0,
                1.0,
                run.loop_count,
            );
        }
    }

    let mut crushed: Vec<(Entity, glam::Vec2, i32)> = Vec::new();
    let hasted = player_q
        .single()
        .is_ok_and(|p| crate::pickups::haste_crown(p) > 0);
    for (prop_e, prop, ppos, _, _, _, _) in props.iter() {
        if !prop.destructible || generators.contains(prop_e) {
            continue;
        }
        let half = prop.size / 2.0;
        let closest = glam::Vec2::new(
            tpos.x.clamp(ppos.0.x - half.x, ppos.0.x + half.x),
            tpos.y.clamp(ppos.0.y - half.y, ppos.0.y + half.y),
        );
        if tpos.distance(closest) > reach {
            continue;
        }
        crushed.push((prop_e, ppos.0, prop.hp.max(1)));
    }
    for (prop_e, center, hp) in crushed {
        crate::spawns::damage_destructible_prop(
            &mut commands,
            &catalog,
            &mut props,
            &entrances,
            &nests,
            &mut secrets,
            &audio,
            &mut cues,
            save.settings.particles,
            prop_e,
            center,
            hp,
            None,
            None,
            run.loop_count,
            hasted,
        );
    }
}

fn throne_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    hurting: bool,
    owner: Entity,
    boss: &mut BossBrain,
    brain: &mut EnemyBrain,
    vel: &mut Velocity,
    pos: &mut Pos,
    health: &Health,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    loop_count: u32,
    dt: f32,
    foes: &[Entity],
    statues: &[(Entity, glam::Vec2)],
    portals: &[Entity],
    walls: &[(glam::Vec2, (i32, i32))],
    radius: f32,
) -> bool {
    let mut rng = rand::rng();
    let mut fired = false;

    // GML `Create_0`: `alarm[1] = 30`. `friction = 1.7` is only read by
    // `enemy/Collision_Wall`, so no per-step drag is applied.
    if boss.aux == 0.0 && brain.burst_left == 0 && boss.phase == BossPhase::Idle {
        brain.burst_left = 1;
        boss.pattern_index = 0;
        boss.attack_timer = GTimer::from_seconds(1.0 /* 30 ticks */, TimerMode::Once);
    }

    // `with enemy { destroy }` (everything but fellow bosses) and
    // `with Portal { destroy }`.
    for foe in foes {
        commands.entity(*foe).despawn();
    }
    for portal in portals {
        commands.entity(*portal).despawn();
    }

    let to_player = player_pos - epos;
    let dist = to_player.length();
    let frac = (health.hp as f32 / health.max.max(1) as f32).clamp(0.0, 1.0);

    // Delayed `NothingBeam` after the 30-tick charge telegraph.
    if boss.phase == BossPhase::Telegraph && boss.phase_timer.just_finished() {
        spawn_enemy_beam(
            commands,
            epos + glam::Vec2::new(0.0, 48.0),
            glam::Vec2::new(0.0, 1.0),
            520.0,
            28.0,
            5,
            2.0,
        );
        trauma.add(0.2);
        fired = true;
        boss.phase = BossPhase::Idle;
    }

    // GML `Alarm_2`: mirrored Horror triplets, 6 rounds per volley tick.
    if boss.special_timer.just_finished() && brain.ammo > 0 {
        boss.special_timer = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
        for flip in [1.0_f32, -1.0] {
            for (dx, off) in [(40.0, 20.0_f32), (56.0, 0.0), (72.0, -20.0)] {
                let ang = 270.0_f32.to_radians() + (brain.gunangle + off.to_radians()) * flip;
                let sdir = glam::Vec2::from_angle(ang);
                fire_projectile(
                    commands,
                    owner,
                    epos + glam::Vec2::new(-dx * flip, 50.0),
                    sdir,
                    Team::Enemy,
                    180.0,
                    2,
                    3.0,
                    4.5,
                    120.0,
                    EnemyKind::Throne,
                );
            }
        }
        fired = true;
        brain.ammo -= 1;
    }

    // GML `Alarm_1` (brain). While the beam charges the brain holds
    // (GML re-arms long cooldowns; the guard also covers test-rate
    // timers that would otherwise reset the charge every tick).
    if boss.attack_timer.just_finished() && boss.phase != BossPhase::Telegraph {
        // GML wall-clear: overlapping walls are destroyed outright.
        for (wpos, cell) in walls {
            if wpos.distance(epos) < radius + 8.0 {
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
        brain.walk = 0.0;
        boss.attack_timer = GTimer::from_seconds(130.0 / 30.0, TimerMode::Once);
        if frac <= 0.4 {
            boss.attack_timer = GTimer::from_seconds(70.0 / 30.0, TimerMode::Once);
        }
        if dist > 290.0 || player_pos.y < epos.y + 48.0 || boss.pattern_index == 0 {
            // Walk the arena x toward the player height.
            boss.pattern_index = 1;
            let ang = (player_pos.y - epos.y).atan2(boss.home.x - epos.x);
            boss.target = glam::Vec2::from_angle(ang);
            brain.walk = 40.0;
            boss.attack_timer = GTimer::from_seconds(40.0 / 30.0, TimerMode::Once);
        } else {
            if player_pos.y < epos.y + 48.0
                || player_pos.x < epos.x - 140.0
                || player_pos.x > epos.x + 140.0
            {
                // Smash the nearest statue and strafe (`mode = 1`).
                if let Some((statue, spos)) = statues
                    .iter()
                    .min_by(|a, b| {
                        a.1.distance_squared(epos)
                            .partial_cmp(&b.1.distance_squared(epos))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied()
                {
                    commands.entity(statue).despawn();
                    // GML `ThroneStatue/Destroy_0.gml:5-7`:
                    // `repeat (1 + GameCont.loops) { instance_create(x, y, Guardian) }`
                    // - the plain `Guardian`, at the statue's exact position.
                    for _ in 0..1 + loop_count {
                        queue_enemy_spawn(
                            &mut *commands,
                            EnemyKind::Guardian,
                            spos,
                            1.0,
                            loop_count,
                        );
                    }
                }
                let d = boss.attack_timer.duration() / 4.0;
                boss.attack_timer = GTimer::from_seconds(d, TimerMode::Once);
                boss.aux = 1.0;
            }
            if boss.aux == 0.0 {
                if player_pos.x < epos.x + 32.0
                    && player_pos.x > epos.x - 32.0
                    && rng.random::<f32>() < 2.0 / 3.0
                {
                    // Centered: beam with a 30-tick charge.
                    boss.set_phase(BossPhase::Telegraph, 1.0);
                } else if frac <= 0.4 {
                    boss.attack_timer = GTimer::from_seconds(20.0 / 30.0, TimerMode::Once);
                    brain.gunangle = [-30.0_f32, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
                        [rng.random_range(0..7)]
                    .to_radians();
                    brain.ammo = (3 + loop_count) as u8;
                    boss.special_timer = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                } else {
                    boss.attack_timer = GTimer::from_seconds(60.0 / 30.0, TimerMode::Once);
                    brain.gunangle =
                        [-20.0_f32, -10.0, 0.0, 10.0, 20.0][rng.random_range(0..5)].to_radians();
                    brain.ammo = (8 + loop_count) as u8;
                    boss.special_timer = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                }
            }
        }
        if boss.aux == 1.0 {
            // Twin `BigGuardianBullet` from the flanks.
            for sx in [-70.0_f32, 70.0] {
                let jitter = rng.random_range(-25.0..=25.0_f32).to_radians();
                let ang = to_player.y.atan2(to_player.x) + jitter;
                let sdir = glam::Vec2::from_angle(ang);
                fire_projectile(
                    commands,
                    owner,
                    epos + glam::Vec2::new(sx, 10.0),
                    sdir,
                    Team::Enemy,
                    rng.random_range(210.0..=240.0),
                    12,
                    3.0,
                    7.0,
                    200.0,
                    EnemyKind::Throne,
                );
            }
            fired = true;
        }
        // `mode = choose(mode, mode, 0, 0, 1)`, pushed to 1 by damage or
        // by the target standing above the throne.
        let roll = rng.random_range(0..5);
        boss.aux = match roll {
            0 | 1 => boss.aux,
            2 | 3 => 0.0,
            _ => 1.0,
        };
        if rng.random_range(0.0..brain.burst_left as f32) > 150.0 && rng.random_bool(0.5) {
            boss.aux = 1.0;
        }
        if player_pos.y < epos.y {
            boss.aux = 1.0;
        }
    }

    // GML `Other_10` locomotion: stomp-step down + surge, capped at 4.
    if brain.walk > 0.0 {
        trauma.add(0.08);
        pos.0.y += 30.0 * dt;
        gml_motion_add_clamp(&mut vel.0, boss.target, 2.0, 4.0, dt);
        brain.walk -= dt * 30.0;
        if brain.walk < 0.0 {
            brain.walk = 0.0;
        }
    } else if vel.0.length() > 30.0 {
        vel.0 = vel.0.normalize() * 30.0;
    }
    if hurting {
        brain.burst_left = 0;
    } else {
        // Uncapped `dmg`: long fights push `mode = 1` like GML.
        brain.burst_left += 1;
    }
    pos.0 += vel.0 * dt;
    fired
}

// Throne II.

/// Verbatim `objects/Nothing2` law: strafe walk (`Alarm_0`), the three
/// rotating attacks with haste (`Alarm_1`), capped surge (`Other_10`).
///
/// register map: `attack_timer` = `alarm[1]`, `special_timer` = `alarm[0]`
/// (strafe), `phase_timer` = `alarm[2]` (intro); `boss.aux` = `attack`;
/// `brain.ammo`/`walk`/`gunangle` = `shots`/`walk`/`aimdir` (radians);
/// `brain.strafe_dir` = `side`; `brain.dash` = `flip`; `boss.target` =
/// `walkdir` heading. (Top kills and taunts out as visual/audio.)
#[allow(clippy::too_many_arguments)]
fn throne_ii_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    toast: &mut Toast,
    owner: Entity,
    boss: &mut BossBrain,
    brain: &mut EnemyBrain,
    vel: &mut Velocity,
    pos: &mut Pos,
    health: &Health,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    dt: f32,
    loop_count: u32,
    walls: &[Entity],
) -> bool {
    let mut rng = rand::rng();
    let mut fired = false;

    // GML `Nothing2/Other_10`: walls turn invisible (still solid via
    // the floor mask; tops have no port entities).
    for w in walls {
        commands.entity(*w).remove::<WallTile>();
        commands.entity(*w).insert(InvisiWall);
    }

    // GML `Create_0`: `alarm[0] = 1`, `alarm[1] = 90`, `alarm[2] = 2`,
    // `attack = 1`, `side = ±1`, `flip = ±1`.
    if boss.aux == 0.0 {
        boss.aux = 1.0;
        brain.ammo = 0;
        brain.strafe_dir = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        brain.dash = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        brain.gunangle = rng.random_range(0.0..std::f32::consts::TAU);
        boss.target = glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU));
        brain.walk = 0.0;
        boss.special_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
        boss.attack_timer = GTimer::from_seconds(90.0 / 30.0, TimerMode::Once);
        boss.phase_timer = GTimer::from_seconds(2.0 / 30.0, TimerMode::Once);
    }

    let to_player = player_pos - epos;
    let aim = to_player.y.atan2(to_player.x);
    let frac = (health.hp as f32 / health.max.max(1) as f32).clamp(0.0, 1.0);

    // GML `Alarm_2`: intro on loop 1.
    if boss.phase_timer.just_finished() && loop_count == 1 {
        toast.show("THRONE II");
        commands.spawn((
            GameCleanup,
            BossIntro {
                timer: GTimer::from_seconds(1.1, TimerMode::Once),
            },
        ));
    }

    // GML `Alarm_0` (strafe).
    if boss.special_timer.just_finished() {
        boss.special_timer = GTimer::from_seconds(1.0 /* 30 ticks */, TimerMode::Once);
        let head = glam::Vec2::from_angle(aim + (50.0 + rng.random_range(0.0..=20.0)) * brain.dash);
        boss.target = head;
        brain.walk = 60.0;
        if rng.random::<f32>() < 0.1 {
            brain.dash = -brain.dash;
        }
    }

    // GML `Alarm_1` (attack).
    if boss.attack_timer.just_finished() {
        boss.attack_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
        let mut exited = false;
        if boss.aux == 1.0 {
            brain.ammo += 1;
            brain.gunangle = aim + (30.0 + rng.random_range(0.0..=10.0)) * brain.strafe_dir;
            let sdir = glam::Vec2::from_angle(brain.gunangle);
            fire_projectile(
                commands,
                owner,
                epos + sdir * 20.0,
                sdir,
                Team::Enemy,
                rng.random_range(150.0..=180.0),
                12,
                3.0,
                8.0,
                200.0,
                EnemyKind::ThroneII,
            );
            fired = true;
            brain.strafe_dir = -brain.strafe_dir;
            if brain.ammo == (2 + loop_count) as u8 {
                brain.ammo = 0;
                let d = boss.attack_timer.duration() + 100.0 / 30.0;
                boss.attack_timer = GTimer::from_seconds(d, TimerMode::Once);
                boss.aux = [1.0, 2.0, 3.0][rng.random_range(0..3)];
                exited = true;
            }
        }
        if !exited && boss.aux == 2.0 {
            // Rotating hose: nudge in, then one bolt per step.
            pos.0 += to_player.normalize_or_zero() * 1.5;
            let count = 10 + loop_count as usize;
            let step = 360.0 / count as f32;
            for _ in 0..count {
                let sdir = glam::Vec2::from_angle(brain.gunangle);
                fire_projectile(
                    commands,
                    owner,
                    epos + sdir * 20.0,
                    sdir,
                    Team::Enemy,
                    240.0,
                    5,
                    2.5,
                    5.0,
                    150.0,
                    EnemyKind::ThroneII,
                );
                brain.gunangle += step.to_radians();
            }
            fired = true;
            brain.walk = 0.0;
            brain.ammo += 1;
            boss.attack_timer = GTimer::from_seconds(2.0 / 30.0, TimerMode::Once);
            if brain.ammo == (15 + loop_count * 5) as u8 {
                brain.ammo = 0;
                let d = boss.attack_timer.duration() + 40.0 / 30.0;
                boss.attack_timer = GTimer::from_seconds(d, TimerMode::Once);
                brain.gunangle = rng.random_range(0.0..std::f32::consts::TAU);
                boss.aux = [1.0, 3.0][rng.random_range(0..2)];
                exited = true;
            }
        }
        if !exited && boss.aux == 3.0 {
            // `Throne2Ball` ring with independent random speeds.
            let count = 4 + loop_count as usize;
            let step = 360.0 / count as f32;
            let mut ang = rng.random_range(0.0..std::f32::consts::TAU);
            for _ in 0..count {
                let sdir = glam::Vec2::from_angle(ang);
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Team::Enemy,
                    Projectile {
                        damage: 12,
                        life: GTimer::from_seconds(
                            (40.0 + loop_count as f32 * 10.0) / 30.0 + 2.0,
                            TimerMode::Once,
                        ),
                        radius: 10.0,
                        knockback: 100.0,
                        explosive: false,
                        source: Some(DamageSource::enemy(owner, EnemyKind::ThroneII)),
                    },
                    crate::comps_a::ProjectileTyp(2),
                    ThroneBall {
                        timeout: 0.0,
                        angle: ang,
                        sounded: true,
                    },
                    Velocity(sdir * rng.random_range(120.0..=300.0)),
                    Pos(epos + sdir * 20.0),
                ));
                ang += step.to_radians();
            }
            fired = true;
            brain.strafe_dir = -brain.strafe_dir;
            brain.ammo = 0;
            let d = boss.attack_timer.duration() + 90.0 / 30.0;
            boss.attack_timer = GTimer::from_seconds(d, TimerMode::Once);
            boss.aux = [1.0, 2.0, 3.0][rng.random_range(0..3)];
            exited = true;
        }
        if !exited {
            let d = boss.attack_timer.duration() / (2.0 - frac);
            boss.attack_timer = GTimer::from_seconds(d, TimerMode::Once);
        }
    }

    // GML `Other_10`: surge while strafing, crawl otherwise.
    if brain.walk > 0.0 {
        if vel.0.length_squared() > 0.001 {
            vel.0 = vel.0.normalize() * 45.0;
        } else {
            vel.0 = boss.target * 45.0;
        }
        let frames = dt * crate::SIM_HZ as f32;
        vel.0 += boss.target * (4.0 * 30.0) * frames;
        brain.walk -= dt * 30.0;
        if brain.walk < 0.0 {
            brain.walk = 0.0;
        }
    } else if vel.0.length_squared() > 0.001 {
        vel.0 = vel.0.normalize() * 15.0;
    }
    pos.0 += vel.0 * dt;
    let _ = trauma;
    fired
}

// Hyper Crystal.

/// Verbatim `objects/HyperCrystal` law.
///
/// `Alarm_1` (50 ticks) either re-seeds the whole crystal shell when
/// `crystals == 0`, releases it once at most half survive, or widens the
/// shell and freezes the core while the player is out of sight. `Alarm_2`
/// (armed only by that last branch) is the signature "shoot the crystal next
/// to you": it arms `alarm[4]` on whichever crystal is nearest the player when
/// that crystal is inside 140px. `Other_10` eases `dist` toward `wantdist`
/// at 5/step and holds still for `nospin` frames.
#[allow(clippy::too_many_arguments)]
fn hyper_ai(
    commands: &mut Commands,
    toast: &mut Toast,
    owner: Entity,
    vel: &mut Velocity,
    pos: &mut Pos,
    state: &mut HyperState,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    los: bool,
    crystals: &[(Entity, glam::Vec2)],
    live_crystals: usize,
    loop_count: u32,
    run: &mut Run,
) -> bool {
    let dt = 1.0 / crate::SIM_HZ as f32;
    state.alarm1.tick(dt);
    state.alarm2.tick(dt);

    // GML `Other_10:4-5`: the shell radius eases toward `wantdist` at 5/step.
    if state.dist < state.wantdist {
        state.dist += 5.0;
    } else if state.dist > state.wantdist {
        state.dist -= 5.0;
    }
    // GML `Other_10:6-17`: `nospin` freezes the core, otherwise it drifts
    // along `direction` capped at 1.5 px/step, with a 50-frame `fastspin`
    // burst right after a respawn.
    if state.nospin > 0.0 {
        state.nospin -= 1.0;
        vel.0 = glam::Vec2::ZERO;
    } else {
        let mut step = 1.0;
        if state.fastspin > 0.0 {
            state.fastspin -= 1.0;
            step += 20.0;
        }
        state.angle += step;
        vel.0 = glam::Vec2::from_angle(state.angle.to_radians()) * 0.5 * 30.0;
        limit_velocity(vel, 1.5 * 30.0);
    }
    pos.0 += vel.0 * dt;

    if state.alarm1.just_finished() {
        if !state.crystals {
            // GML `Alarm_1:11-26`: a released shell is re-seeded in full, so
            // the fight keeps regenerating adds.
            hyper_ensure_orbit(commands, owner, epos, loop_count, false, run);
            state.crystals = true;
            state.fastspin = 50.0;
            state.dist = 0.0;
            state.wantdist = 25.0;
            state.alarm1 = gml_alarm(50.0);
        } else {
            // GML `Alarm_1:34-36`: at most half alive releases the shell.
            let total = hyper_orbit_count(loop_count);
            if live_crystals <= total / 2 {
                state.crystals = false;
                state.alarm1 = gml_alarm(50.0);
            } else {
                state.wantdist = 80.0;
                // GML `Alarm_1:40-46`: player out of sight.
                if !los && state.intro {
                    state.wantdist = 120.0;
                    state.nospin = 50.0;
                    state.alarm2 = gml_alarm(50.0);
                    state.alarm1 = gml_alarm(80.0);
                }
            }
        }
    }

    // GML `HyperCrystal/Alarm_2.gml`: the crystal nearest the player is
    // flagged `explode = 1` and has its `alarm[4]` loaded with 40.
    if state.alarm2.just_finished() {
        let victim = crystals
            .iter()
            .min_by(|a, b| {
                a.1.distance_squared(player_pos)
                    .total_cmp(&b.1.distance_squared(player_pos))
            })
            .filter(|(_, c)| c.distance(player_pos) < 140.0);
        match victim {
            Some((at, _)) if state.intro => {
                commands.entity(*at).insert(HyperCrystalArm {
                    timer: gml_alarm(40.0),
                });
            }
            // GML `Alarm_2:11` - nothing in reach, so retry almost at once.
            _ => state.alarm1 = gml_alarm(5.0),
        }
    }

    // GML `Alarm_1:4-10`: the intro card, once, on a clear line.
    if !state.intro && los {
        state.intro = true;
        state.alarm3 = gml_alarm(2.0);
    }
    if state.alarm3.just_finished() {
        toast.show("HYPER CRYSTAL");
        commands.spawn((
            GameCleanup,
            BossIntro {
                timer: GTimer::from_seconds(1.1, TimerMode::Once),
            },
        ));
        state.alarm3 = gml_alarm_off();
    }

    false
}

/// GML `LaserCrystal/Alarm_4.gml:1-16`: `hp = 0`, four 48px wall holes, and a
/// `5 + loops * 2` beam radial.
fn hyper_detonate_crystal(commands: &mut Commands, at: glam::Vec2, loop_count: u32) {
    let lasers = 5 + loop_count as usize * 2;
    for angle in ring_angles(lasers, 0.0) {
        spawn_enemy_beam(commands, at, dir_from_angle(angle), 420.0, 12.0, 2, 0.28);
    }
    for off in [
        glam::Vec2::new(-48.0, 0.0),
        glam::Vec2::new(48.0, 0.0),
        glam::Vec2::new(0.0, -48.0),
        glam::Vec2::new(0.0, 48.0),
    ] {
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            PortalClear {
                timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                scale: 1.0,
            },
            Pos(at + off),
        ));
    }
}

/// Seed the orbit-crystal shell (GML: `cnumber = 3 + loops*2` real
/// `LaserCrystal` enemies - `InvLaserCrystal`/invariants in area 104 -
/// which the core then herds; each spawn decrements the kill count).
pub fn hyper_ensure_orbit(
    commands: &mut Commands,
    owner: Entity,
    pos: glam::Vec2,
    loop_count: u32,
    enraged: bool,
    run: &mut Run,
) {
    let n = (hyper_orbit_count(loop_count) + usize::from(enraged)).min(12);
    for i in 0..n {
        // GML type law: LaserCrystal, or area-104 triple-inv + lightning.
        let kind = if run.area == AreaId::CursedCaves {
            match rand::rng().random_range(0..4) {
                0 | 1 | 2 => EnemyKind::InvLaserCrystal,
                _ => EnemyKind::LightningCrystal,
            }
        } else {
            EnemyKind::LaserCrystal
        };
        let def = enemy_def(kind);
        let hp = crate::enemies::spawn_hp(kind, def.hp, loop_count);
        run.total_kills = run.total_kills.saturating_sub(1);
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            Team::Enemy,
            Enemy {
                kind,
                score: def.score,
                touch_damage: def.touch_damage,
                rad_drop: def.rad_drop,
                drop_chance: def.drop_chance,
                weapon_chance: def.weapon_chance,
                give_kill: true,
            },
            Health {
                hp,
                max: hp,
                invuln: ready_timer(),
            },
            NextHurt::default(),
            Hitbox { radius: def.radius },
            Velocity(glam::Vec2::ZERO),
            // GML `HyperCrystal/Alarm_1:22` creates every crystal on the core.
            Pos(pos),
            HyperOrbitCrystal { owner, slot: i },
        ));
    }
}

/// GML `HyperCrystal/Other_10.gml:18-33`: while the shell is up, every live
/// crystal is parked on the core and pushed out along `angle` by `dist` -
/// except any already under a third of its health, which the core lets go of
/// and which then drifts on its own `direction`. The armed crystal's
/// `alarm[4]` is resolved here too, since only the crystal knows where it
/// ended up.
pub fn tick_hyper_orbit_crystals(
    time: Res<SimTime>,
    mut commands: Commands,
    run: Res<Run>,
    mut q: Query<
        (
            &mut Pos,
            &mut Health,
            &HyperOrbitCrystal,
            Option<&mut HyperCrystalArm>,
        ),
        (With<HyperOrbitCrystal>, Without<Enemy>),
    >,
    cores: Query<(&Pos, &HyperState), (With<Enemy>, With<HyperState>)>,
) {
    let dt = time.delta_secs;
    for (mut pos, mut health, crystal, mut arm) in &mut q {
        // GML `Other_10:23` - a crystal under a third of its health is let go.
        if health.hp <= health.max / 3 {
            continue;
        }
        let Ok((core_pos, state)) = cores.get(crystal.owner) else {
            continue;
        };
        pos.0 = core_pos.0 + dir_from_angle(state.angle.to_radians()) * state.dist;

        let Some(arm) = arm.as_deref_mut() else {
            continue;
        };
        arm.timer.tick(dt);
        if arm.timer.just_finished() {
            hyper_detonate_crystal(&mut commands, pos.0, run.loop_count);
            // GML `Alarm_4:1` is `hp = 0`, so the crystal still goes through
            // the normal death path and pays out its `scrDrop`.
            health.hp = 0;
        }
    }
}

// Captain.

/// Verbatim `objects/Last` law: decide (`Alarm_1`), two 30-round spin
/// patterns (`Alarm_2`), 17-step warp-out (`Alarm_3`), two-stage dash
/// (`Alarm_4`), shared reset + `LastBall` (`Alarm_5`), intro chain
/// (`Alarm_6`/`Alarm_7`), capped dash/walk (`Step_0`), destroy-or-bounce
/// wall law (`Collision_Wall`).
///
/// register map: `attack_timer`/`special_timer`/`phase_timer` =
/// `alarm[1]`/`alarm[2]`/`alarm[5]`; `pattern_index` = `attacktype`;
/// `brain.ammo`/`walk`/`gunangle` = `ammo`/`walk`/`gunangle` (radians);
/// `boss.target` = `direction`; `boss.aux` = `intro`; `brain.fire` =
/// `introcharge`; `brain.burst_timer` = `alarm[6]`/`alarm[7]`; `phase` =
/// `charge` + `drawspr` (Telegraph = `alarm[3]` warp-out, Charging =
/// `charge == 1`, Cooldown = `charge == -1`, Radial = `sprLastSpin`,
/// Landing = `sprLastWarpIn`).
#[allow(clippy::too_many_arguments)]
/// Verbatim `objects/FrogQueen` law: aim-drift brain (`Alarm_1`), hatch
/// stream into a single gas mortar (`Alarm_2`), capped waddle (`Other_10`).
///
/// register map: `attack_timer`/`special_timer` = `alarm[1]`/`alarm[2]`;
/// `brain.ammo`/`walk`/`gunangle` = `ammo`/`walk`/`gunangle` (radians);
/// `boss.target` = move heading (`direction`); `pattern_index` = `intro`;
/// `brain.burst_left` = init flag. (Friction 0, taunts,
/// `FrogQueenDeath`/music cues, frog-pistol tribute out as feel/visual/meta.)
#[allow(clippy::too_many_arguments)]
fn frog_queen_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    toast: &mut Toast,
    owner: Entity,
    boss: &mut BossBrain,
    brain: &mut EnemyBrain,
    vel: &mut Velocity,
    pos: &mut Pos,
    def: EnemyDef,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    dir: glam::Vec2,
    dt: f32,
    frogs: usize,
    props: &[(glam::Vec2, glam::Vec2)],
    walls: &[(glam::Vec2, (i32, i32))],
    loops: u32,
) -> bool {
    let mut rng = rand::rng();
    let mut fired = false;

    // GML `Create_0`: `alarm[1] = 60`, aim at the target.
    if brain.burst_left == 0 {
        brain.burst_left = 1;
        brain.gunangle = dir.y.atan2(dir.x);
        boss.attack_timer = GTimer::from_seconds(60.0 / 30.0, TimerMode::Once);
    }

    let to_player = player_pos - epos;
    let dist = to_player.length();
    let aim = to_player.y.atan2(to_player.x);
    let wall_centers: Vec<glam::Vec2> = walls.iter().map(|(c, _)| *c).collect();
    let los =
        dist < 512.0 && !crate::walls::segment_hits_wall_legacy(epos, player_pos, &wall_centers);

    // GML `Alarm_1` (brain).
    if boss.attack_timer.just_finished() {
        boss.attack_timer = GTimer::from_seconds(
            (30.0 + rng.random_range(0.0..=20.0)) / 30.0,
            TimerMode::Once,
        );
        brain.walk = 0.0;
        boss.target = if los {
            glam::Vec2::from_angle(aim + rng.random_range(-10.0..=10.0_f32).to_radians())
        } else {
            glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU))
        };
        if rng.random::<f32>() < 0.25 {
            boss.special_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
        } else {
            brain.walk = 50.0;
            if (rng.random::<f32>() < 1.0 / 3.0 && los) || dist < 160.0 {
                if boss.pattern_index == 0 && loops == 1 {
                    boss.pattern_index = 1;
                    toast.show("MOM");
                    commands.spawn((
                        GameCleanup,
                        BossIntro {
                            timer: GTimer::from_seconds(1.1, TimerMode::Once),
                        },
                    ));
                }
                brain.walk += 30.0;
                boss.attack_timer = GTimer::from_seconds(brain.walk / 30.0, TimerMode::Once);
                brain.ammo = (2 + loops) as u8;
                boss.special_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
            }
        }
    }

    // GML `Alarm_2`: hatch stream while loaded, then one gas mortar.
    if boss.special_timer.just_finished() {
        if brain.ammo > 0 {
            if frogs < 8 {
                queue_enemy_spawn(&mut *commands, EnemyKind::FrogEgg, epos, 1.0, loops);
            }
            brain.ammo -= 1;
            if brain.ammo > 0 {
                boss.special_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
            }
        } else {
            let jitter = rng.random_range(-30.0..=30.0_f32).to_radians();
            let sdir = glam::Vec2::from_angle(aim + jitter);
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                Team::Enemy,
                Projectile {
                    damage: 5,
                    life: GTimer::from_seconds(4.0, TimerMode::Once),
                    radius: 7.0,
                    knockback: 150.0,
                    explosive: false,
                    source: Some(DamageSource::enemy(owner, EnemyKind::FrogQueen)),
                },
                crate::comps_a::ProjectileTyp(2),
                MomShot,
                Velocity(sdir * 120.0),
                Pos(epos + sdir * 20.0),
            ));
            fired = true;
            trauma.add(0.15);
        }
    }

    // GML `Other_10`: waddle capped at 1.5 + loops / 2 px/tick.
    let cap = 45.0 + loops as f32 * 15.0;
    if vel.0.length() > cap {
        vel.0 = vel.0.normalize() * cap;
    }
    if brain.walk > 0.0 {
        gml_motion_add_clamp(&mut vel.0, boss.target, 0.6, 1.5 + loops as f32 / 2.0, dt);
        brain.walk -= dt * 30.0;
        if brain.walk < 0.0 {
            brain.walk = 0.0;
        }
    }
    if vel.0.length() > cap {
        vel.0 = vel.0.normalize() * cap;
    }
    pos.0 += vel.0 * dt;
    resolve_prop_collision(&mut pos.0, def.radius, props.iter().copied());
    fired
}

/// Verbatim `objects/TechnoMancer` law: a six-alarm machine over
/// `main`/`intro`/`drawspr` that never moves (`Other_10`: `speed = 0`,
/// `x = xstart`). The live instance alternates between reviving corpses and
/// emplacing turrets; the others stay dormant.
///
/// `alarm[1]` is the 90-tick decide. While `main` and mid-appearance it
/// finishes appearing and stands down; while awake it revives three corpses
/// when one is visible, else emplaces `1 + loops` turrets while fewer than
/// `4 * loops` exist; otherwise it disowns the fight.
#[allow(clippy::too_many_arguments)]
fn technomancer_ai(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    toast: &mut Toast,
    state: &mut TechnomancerState,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    corpses: &[glam::Vec2],
    turrets: usize,
    loops: u32,
    mask: &FloorMask,
) {
    let dt = 1.0 / crate::SIM_HZ as f32;
    for a in [
        &mut state.alarm1,
        &mut state.alarm2,
        &mut state.alarm4,
        &mut state.alarm5,
        &mut state.alarm6,
    ] {
        a.tick(dt);
    }

    // GML `Alarm_4`: the fight passes to whichever instance is nearest.
    if state.alarm4.just_finished() {
        state.visual = TechnoVisual::Inactive;
        state.main = true;
        state.visual = TechnoVisual::Appear;
        state.alarm5 = gml_alarm(17.0);
    }

    // GML `Alarm_3` (`scrBossIntro(7)`), once the instance is awake.
    if state.alarm5.just_finished() && !state.intro && state.visual == TechnoVisual::Active {
        state.intro = true;
        toast.show("TECHNOMANCER");
        commands.spawn((
            GameCleanup,
            BossIntro {
                timer: GTimer::from_seconds(1.1, TimerMode::Once),
            },
        ));
    }

    // GML `Alarm_6`: the emplaced turrets, each snapped to its floor tile.
    if state.alarm6.just_finished() {
        spawn_techno_turrets(commands, catalog, epos, 1 + loops, mask);
    }

    // GML `Alarm_2`: three corpses at the player-facing bearing and +-80deg.
    if state.alarm2.just_finished() {
        revive_techno_corpses(commands, epos, player_pos, corpses, mask);
    }

    if !state.alarm1.just_finished() {
        return;
    }
    state.alarm1 = gml_alarm(90.0);

    if !state.main {
        return;
    }

    match state.visual {
        // GML `Alarm_1:4-9`: finish appearing, then stand down as `main`.
        TechnoVisual::Appear => {
            state.visual = TechnoVisual::Active;
            state.main = false;
            state.alarm5 = gml_alarm(17.0);
        }
        TechnoVisual::Inactive => {}
        _ => {
            // GML `Alarm_1:13-27`. The revive gate is `random(5) < 6`, always
            // true in GML, so any visible corpse outranks emplacing turrets.
            if nearest_walkable(corpses, epos, mask).is_some() {
                state.alarm5 = gml_alarm(70.0);
                state.alarm2 = gml_alarm(55.0);
            } else if (turrets as u32) < 4 * loops {
                state.alarm5 = gml_alarm(52.0);
                state.alarm6 = gml_alarm(35.0);
            } else {
                // GML `Alarm_1:36-46`: not the elected instance.
                state.visual = TechnoVisual::Disappear;
                state.alarm4 = gml_alarm(22.0);
                state.main = false;
            }
        }
    }
}

/// GML `TechnoMancer/Alarm_6.gml:1-9`: `1 + loops` turrets at 60..160px out,
/// each moved onto its nearest floor tile.
fn spawn_techno_turrets(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    at: glam::Vec2,
    count: u32,
    mask: &FloorMask,
) {
    let mut rng = rand::rng();
    for _ in 0..count {
        let ang = rng.random_range(0.0..std::f32::consts::TAU);
        let d = 60.0 + rng.random_range(0.0..100.0);
        queue_enemy_spawn(
            commands,
            EnemyKind::Turret,
            snap_to_floor(at + glam::Vec2::new(ang.cos(), ang.sin()) * d, mask),
            1.0,
            0,
        );
    }
    let _ = catalog;
}

/// GML `TechnoMancer/Alarm_2.gml:1-21`: revive the nearest corpse to each of
/// three bearings around the player-facing one, each 80px out plus a fresh
/// +-40 jitter, and only when the line is clear.
fn revive_techno_corpses(
    commands: &mut Commands,
    at: glam::Vec2,
    player_pos: glam::Vec2,
    corpses: &[glam::Vec2],
    mask: &FloorMask,
) {
    let mut rng = rand::rng();
    let base = (at - player_pos).y.atan2((at - player_pos).x);
    for bearing in [
        base,
        base + 80.0_f32.to_radians(),
        base - 80.0_f32.to_radians(),
    ] {
        let probe = at
            + glam::Vec2::new(bearing.cos(), bearing.sin()) * 80.0
            + glam::Vec2::new(rng.random_range(-40.0..40.0), rng.random_range(-40.0..40.0));
        let Some(corpse) = nearest_walkable(corpses, probe, mask) else {
            continue;
        };
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            NecroReviveArea {
                target: corpse,
                timer: GTimer::from_seconds(15.0 / 30.0, TimerMode::Once),
            },
            Pos(corpse),
        ));
    }
}

/// Nearest point to `from` with a clear line, if any.
fn nearest_walkable(
    points: &[glam::Vec2],
    from: glam::Vec2,
    mask: &FloorMask,
) -> Option<glam::Vec2> {
    points
        .iter()
        .filter(|p| crate::enemies::line_of_sight_public(from, **p, mask))
        .min_by(|a, b| {
            from.distance_squared(**a)
                .total_cmp(&from.distance_squared(**b))
        })
        .copied()
}

/// GML `instance_nearest(x, y, Floor)` then `x = dir.x + 16`.
fn snap_to_floor(at: glam::Vec2, mask: &FloorMask) -> glam::Vec2 {
    mask.cells
        .iter()
        .map(|c| mask.cell_center(*c))
        .min_by(|a, b| a.distance_squared(at).total_cmp(&b.distance_squared(at)))
        .unwrap_or(at)
}

fn captain_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    toast: &mut Toast,
    owner: Entity,
    kind: EnemyKind,
    boss: &mut BossBrain,
    brain: &mut EnemyBrain,
    enemy: &mut Enemy,
    health: &Health,
    vel: &mut Velocity,
    pos: &mut Pos,
    def: EnemyDef,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    dir: glam::Vec2,
    dt: f32,
    props: &[(glam::Vec2, glam::Vec2)],
    walls: &[(glam::Vec2, (i32, i32))],
    mask: &FloorMask,
) -> bool {
    let mut rng = rand::rng();
    let frames = dt * crate::SIM_HZ as f32;
    let mut fired = false;
    let aim = dir.y.atan2(dir.x);

    // GML `Create_0`: `alarm[1] = 1`, `meleedamage = 10`, `charge = 0`,
    // `introcharge = 0`, `attacktype = 0`, `ammo = 0`, `walk = 0`.
    if brain.burst_left == 0 {
        brain.burst_left = 1;
        boss.phase = BossPhase::Idle;
        boss.attack_timer = gml_alarm(1.0);
        boss.special_timer = gml_alarm_off();
        brain.gunangle = rng.random_range(0.0..std::f32::consts::TAU);
        enemy.touch_damage = 10;
        captain_popo_explosion(commands, epos, owner, kind);
    }

    // GML `Step_0:5-16`: the dash sets `speed = 14` along `gunangle`, the
    // walk adds 2 along `direction` and 1 along `gunangle` capped at 3.
    if boss.phase == BossPhase::Charging {
        vel.0 = glam::Vec2::from_angle(brain.gunangle) * (14.0 * 30.0);
    } else {
        if brain.walk > 0.0 {
            brain.walk = (brain.walk - frames).max(0.0);
            gml_motion_add_clamp(&mut vel.0, boss.target, 2.0, 3.0, dt);
            gml_motion_add_clamp(
                &mut vel.0,
                glam::Vec2::from_angle(brain.gunangle),
                1.0,
                3.0,
                dt,
            );
        }
        limit_velocity(vel, 3.0 * 30.0);
    }

    // GML `Alarm_1` (decide).
    if boss.attack_timer.just_finished() {
        let dist = epos.distance(player_pos);
        let blocked = crate::walls::segment_hits_wall(epos, player_pos, mask);
        // GML `Alarm_1:5` - `CrystalShield` has no GML object in this
        // rewrite, so that disjunct is dropped.
        let dash = brain.fire == 0
            || (rng.random_range(0.0..3.0) < 1.0
                && (blocked || (dist > 90.0 && rng.random_range(0.0..2.0) < 1.0)));
        if dash {
            brain.fire = 1;
            boss.phase = BossPhase::Telegraph;
            boss.phase_timer = gml_alarm(10.0);
            if boss.aux == 0.0 {
                // `alarm[6] = 16` then `alarm[7] = 2` -> the boss intro.
                brain.burst_timer = gml_alarm(18.0);
            }
            trauma.add(0.06);
        } else if (dist > 120.0 && rng.random_range(0.0..2.0) < 1.0)
            || rng.random_range(0.0..4.0) < 1.0
        {
            boss.phase = BossPhase::Jumping;
            boss.phase_timer = gml_alarm(17.0);
            trauma.add(0.1);
        } else {
            boss.pattern_index = usize::from(rng.random_bool(0.5));
            let side = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            brain.gunangle = aim + side * (10.0 + rng.random_range(0.0f32..15.0)).to_radians();
            brain.ammo = 30;
            boss.special_timer = gml_alarm(8.0);
            boss.phase = BossPhase::Radial;
            boss.phase_timer = gml_alarm(47.0);
            trauma.add(0.12);
        }
    }

    // GML `Alarm_2` (spin fire).
    if boss.phase == BossPhase::Radial && boss.special_timer.just_finished() {
        if boss.pattern_index == 0 {
            if brain.ammo > 0 {
                brain.ammo = brain.ammo.saturating_sub(4);
                boss.special_timer = gml_alarm(4.0);
                brain.gunangle += 9.0_f32.to_radians();
                for _ in 0..4 {
                    captain_idpd_bullet(commands, owner, kind, epos, brain.gunangle, 4.0);
                    captain_idpd_bullet(commands, owner, kind, epos, -brain.gunangle, 4.0);
                    brain.gunangle += 90.0_f32.to_radians();
                }
                fired = true;
            }
        } else if brain.ammo > 0 {
            brain.ammo -= 1;
            boss.special_timer = gml_alarm(2.0);
            // GML `alarm[5] += 1` extends the remaining spin by one step.
            let remain = (boss.phase_timer.remaining_secs() * 30.0).floor() + 1.0;
            boss.phase_timer = gml_alarm(remain);
            gml_motion_add_clamp(
                &mut vel.0,
                glam::Vec2::from_angle(brain.gunangle + std::f32::consts::PI),
                0.5,
                3.0,
                dt,
            );
            let spread = (brain.ammo as f32 * 5.0 + 16.0).to_radians();
            for half in [1.0_f32, 0.5] {
                for sign in [-1.0_f32, 1.0] {
                    captain_idpd_bullet(
                        commands,
                        owner,
                        kind,
                        epos,
                        brain.gunangle + sign * spread * half,
                        12.0,
                    );
                }
            }
            fired = true;
        }
    }

    // GML `Alarm_3` (warp next to the player).
    if boss.phase == BossPhase::Jumping && boss.phase_timer.just_finished() {
        let mut target_pos = pos.0;
        for _ in 0..64 {
            let cand = glam::Vec2::new(
                player_pos.x + rng.random_range(0.0..320.0) - 160.0,
                player_pos.y + rng.random_range(0.0..320.0) - 160.0,
            );
            target_pos = cand;
            if cand.distance(player_pos) > 80.0
                && cand.distance(pos.0) > 60.0
                && mask.is_walkable(cand)
            {
                break;
            }
        }
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            PortalClear {
                timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                scale: 1.0,
            },
            Pos(pos.0),
        ));
        pos.0 = mask.cell_center(mask.world_to_cell(target_pos));
        vel.0 = glam::Vec2::ZERO;
        boss.phase = BossPhase::Landing;
        boss.phase_timer = gml_alarm(20.0);
        trauma.add(0.2);
    }

    // GML `Alarm_4` (dash start, then dash end).
    match boss.phase {
        BossPhase::Telegraph if boss.phase_timer.just_finished() => {
            boss.phase = BossPhase::Charging;
            boss.phase_timer = gml_alarm(10.0);
            captain_popo_explosion(commands, pos.0, owner, kind);
            brain.gunangle = aim + rng.random_range(-15.0f32..=15.0).to_radians();
            vel.0 += glam::Vec2::from_angle(brain.gunangle) * (10.0 * 30.0) * frames;
            trauma.add(0.14);
        }
        BossPhase::Charging if boss.phase_timer.just_finished() => {
            boss.phase = BossPhase::Cooldown;
            boss.phase_timer = gml_alarm(12.0);
        }
        _ => (),
    }

    // GML `Alarm_5` (spin / dash / warp reset, plus the `LastBall`).
    if matches!(
        boss.phase,
        BossPhase::Radial | BossPhase::Cooldown | BossPhase::Landing
    ) && boss.phase_timer.just_finished()
    {
        if boss.phase == BossPhase::Landing {
            // GML `Alarm_5:7-11` `LastBall` at 6 px/tick straight at the
            // target; `tick_last_balls` owns its eight-ring death burst.
            let sdir = (player_pos - pos.0).normalize_or_zero();
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                crate::comps_b::LastBall,
                Velocity(sdir * (6.0 * 30.0)),
                Pos(pos.0),
            ));
            trauma.add(0.14);
        }
        boss.phase = BossPhase::Idle;
        let frac = (health.hp as f32 / health.max.max(1) as f32).clamp(0.0, 1.0);
        boss.attack_timer = gml_alarm(10.0 + frac * 15.0);
    }

    // GML `Alarm_6` -> `Alarm_7` -> `scrBossIntro(8)`.
    brain.burst_timer.tick(dt);
    if brain.burst_timer.just_finished() {
        boss.aux = 1.0;
        toast.show("CAPTAIN");
        commands.spawn((
            GameCleanup,
            BossIntro {
                timer: GTimer::from_seconds(1.1, TimerMode::Once),
            },
        ));
    }

    // GML `Collision_Wall.gml:4-9`: `charge > 0` destroys the tile. As
    // above, only the breaking branch pre-moves.
    if boss.phase == BossPhase::Charging {
        let before = pos.0;
        pos.0 += vel.0 * dt;
        queue_wall_breaks_along_segment(commands, walls, before, pos.0, def.radius * 0.9);
    } else {
        move_bounce_solid(
            &mut pos.0,
            &mut vel.0,
            def.radius,
            dt,
            props,
            Some(mask),
            true,
        );
    }
    resolve_prop_collision(&mut pos.0, def.radius, props.iter().copied());
    fired
}

/// GML `objects/PopoExplosion`: a 64x64 `damage = 8` blast on the `team_popo`
/// side. The port has no third team, so only the damage and the shake land.
fn captain_popo_explosion(commands: &mut Commands, at: glam::Vec2, owner: Entity, kind: EnemyKind) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        crate::combat::Explosion {
            timer: GTimer::from_seconds(0.05, TimerMode::Once),
            radius: 32.0,
            damage: 8,
            team: Team::Enemy,
            hits_player: true,
            source: Some(DamageSource::enemy(owner, kind)),
        },
        Pos(at),
    ));
}

/// GML `Last/Alarm_2` `IDPDBullet`: an `EnemyBullet1` (`damage = 3`,
/// `typ = 1`, `knockback_speed = 4`) with the IDPD hit sprite.
fn captain_idpd_bullet(
    commands: &mut Commands,
    owner: Entity,
    kind: EnemyKind,
    at: glam::Vec2,
    angle: f32,
    speed: f32,
) {
    let sdir = glam::Vec2::from_angle(angle);
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        Team::Enemy,
        Projectile {
            damage: 3,
            life: GTimer::from_seconds(3.0, TimerMode::Once),
            radius: 4.5,
            knockback: 120.0,
            explosive: false,
            source: Some(DamageSource::enemy(owner, kind)),
        },
        ProjectileTyp(1),
        ProjectileFade("images/sprIDPDBulletHit.png"),
        Velocity(sdir * (speed * 30.0)),
        Pos(at),
    ));
}

// Old Guardian.

/// Kiting fan + enrage-scaled ring (bevy `old_guardian_ai` parity).
#[allow(clippy::too_many_arguments)]
// YV (Gun God).

/// Verbatim `objects/YVBoss` law: weapon-switch brain (`Alarm_1`), per-weapon
/// fire tick (`Alarm_2`), cooldown gate (`Alarm_4`), intro on first empty
/// revolver (`Alarm_5`).
///
/// register map: `boss.pattern_index` = `wep` (0 golden revolver, 1 golden
/// shotgun, 2 golden bazooka, 3 minigun); `brain.ammo` = `ammo`;
/// `brain.gunangle` (radians) = `gunangle`; `boss.aux` = `minigun_side`;
/// `boss.phase` Idle = pre-intro. `attack_timer`/`special_timer`/
/// `phase_timer` are `alarm[1]`/`alarm[2]`/`alarm[4]` in seconds.
#[allow(clippy::too_many_arguments)]
fn yv_boss_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    toast: &mut Toast,
    owner: Entity,
    boss: &mut BossBrain,
    brain: &mut EnemyBrain,
    vel: &mut Velocity,
    pos: &mut Pos,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    player_velocity: glam::Vec2,
    dt: f32,
    walls: &[(glam::Vec2, (i32, i32))],
) -> bool {
    const REVOLVER: usize = 0;
    const SHOTGUN: usize = 1;
    const BAZOOKA: usize = 2;
    const MINIGUN: usize = 3;

    let mut rng = rand::rng();
    let mut fired = false;

    // GML `Create_0`: golden revolver, 5 rounds, brain in 20 ticks,
    // first shot in 8 ticks.
    if boss.aux == 0.0 && brain.ammo == 0 && boss.phase == BossPhase::Idle {
        boss.pattern_index = REVOLVER;
        brain.ammo = 5;
        boss.aux = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        brain.gunangle = (player_pos - epos).y.atan2((player_pos - epos).x);
        boss.attack_timer = GTimer::from_seconds(20.0 / 30.0, TimerMode::Once);
        boss.special_timer = GTimer::from_seconds(8.0 / 30.0, TimerMode::Once);
        boss.phase_timer = GTimer::from_seconds(0.01, TimerMode::Once);
    }

    let to_player = player_pos - epos;
    let dist = to_player.length();
    let aim = to_player.y.atan2(to_player.x);
    let wall_centers: Vec<glam::Vec2> = walls.iter().map(|(c, _)| *c).collect();
    let los =
        dist < 512.0 && !crate::walls::segment_hits_wall_legacy(epos, player_pos, &wall_centers);
    let can_shoot = boss.phase_timer.finished();

    // GML `Alarm_2` (fire tick).
    if boss.special_timer.just_finished() {
        // `(ammo--) <= 0`: empty click knocks back and walks off.
        if brain.ammo == 0 {
            let rand_dir = glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU));
            gml_motion_add_clamp(&mut vel.0, rand_dir, 1.0, 4.0, dt);
            gml_motion_add_clamp(&mut vel.0, to_player.normalize_or_zero(), 4.0, 4.0, dt);
            // GML `Alarm_2:7` `scrWalk(direction, 1, 10, 30)`.
            boss.target = vel.0.normalize_or_zero();
            gml_motion_add_clamp(&mut vel.0, boss.target, 1.0, 4.0, dt);
            brain.walk = rng.random_range(10.0..=30.0);
        } else {
            brain.ammo -= 1;
            match boss.pattern_index {
                REVOLVER => {
                    brain.gunangle = aim;
                    if brain.ammo == 0 {
                        // Last round: twin shot, 5 degrees apart.
                        brain.gunangle -= 2.0_f32.to_radians();
                        for _ in 0..2 {
                            let jitter = rng.random_range(-3.0..=3.0_f32).to_radians();
                            let sdir = glam::Vec2::from_angle(brain.gunangle + jitter);
                            fire_projectile(
                                commands,
                                owner,
                                epos + sdir * 20.0,
                                sdir,
                                Team::Enemy,
                                480.0,
                                3,
                                3.0,
                                4.5,
                                210.0,
                                EnemyKind::YvBoss,
                            );
                            brain.gunangle += 5.0_f32.to_radians();
                        }
                        fired = true;
                        // GML `Alarm_5` (intro) on first empty revolver.
                        if boss.phase == BossPhase::Idle {
                            boss.phase = BossPhase::Cooldown;
                            toast.show("Y.V.");
                            commands.spawn((
                                GameCleanup,
                                BossIntro {
                                    timer: GTimer::from_seconds(1.1, TimerMode::Once),
                                },
                            ));
                        }
                    } else {
                        let sdir = glam::Vec2::from_angle(brain.gunangle);
                        fire_projectile(
                            commands,
                            owner,
                            epos + sdir * 20.0,
                            sdir,
                            Team::Enemy,
                            480.0,
                            3,
                            3.0,
                            4.5,
                            210.0,
                            EnemyKind::YvBoss,
                        );
                        fired = true;
                    }
                    trauma.add(0.2);
                    boss.special_timer = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                }
                SHOTGUN => {
                    brain.gunangle = aim;
                    for _ in 0..18 {
                        let jitter = rng.random_range(-30.0..=30.0_f32).to_radians();
                        let sdir = glam::Vec2::from_angle(brain.gunangle + jitter);
                        // GML `EnemyBullet3`: `wallbounce = 0`, `friction = 0.6`,
                        // bounce `min(18, speed * 0.8)`, fade under 6 px/step.
                        commands.spawn((
                            GameCleanup,
                            LevelCleanup,
                            Team::Enemy,
                            Projectile {
                                damage: 1,
                                life: GTimer::from_seconds(4.0, TimerMode::Once),
                                radius: 4.0,
                                knockback: 120.0,
                                explosive: false,
                                source: Some(DamageSource::enemy(owner, EnemyKind::YvBoss)),
                            },
                            ProjectileTyp(1),
                            ProjectileFriction(0.6),
                            BouncesLeft(255),
                            ShellWallBounce {
                                add: 0.0,
                                cap: 18.0 * 30.0,
                                decay: 0.9,
                                rearm: None,
                            },
                            ProjectileFade("images/sprEBullet3Disappear.png"),
                            Velocity(sdir * rng.random_range(12.0..=18.0) * 30.0),
                            Pos(epos + sdir * 20.0),
                        ));
                    }
                    fired = true;
                    trauma.add(0.4);
                }
                BAZOOKA => {
                    for i in -2..=2 {
                        if i == 0 {
                            continue;
                        }
                        let jitter = rng.random_range(-3.0..=3.0_f32).to_radians();
                        let ang = brain.gunangle + i as f32 * 3.0_f32.to_radians() + jitter;
                        let sdir = glam::Vec2::from_angle(ang);
                        // GML `Rocket`: `active` only after `alarm[1] = 5`, then
                        // `motion_add_m(direction, accel = 2, maxspeed = 12)`,
                        // destroyed by wall or hitme -> `Explosion`.
                        commands.spawn((
                            GameCleanup,
                            LevelCleanup,
                            Team::Enemy,
                            Projectile {
                                damage: 20,
                                life: GTimer::from_seconds(4.0, TimerMode::Once),
                                radius: 6.0,
                                knockback: 300.0,
                                explosive: true,
                                source: Some(DamageSource::enemy(owner, EnemyKind::YvBoss)),
                            },
                            ProjectileTyp(2),
                            ProjectileAccel {
                                rate: 2.0,
                                max: 12.0,
                                arm: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                            },
                            CustomExplosion::default(),
                            Velocity(sdir * 90.0),
                            Pos(epos + sdir * 20.0),
                        ));
                    }
                    fired = true;
                    trauma.add(0.4);
                }
                _ => {
                    let jitter = rng.random_range(-5.0..=5.0_f32).to_radians();
                    let sdir = glam::Vec2::from_angle(brain.gunangle + jitter);
                    fire_projectile(
                        commands,
                        owner,
                        epos + sdir * 20.0,
                        sdir,
                        Team::Enemy,
                        480.0,
                        3,
                        3.0,
                        4.5,
                        210.0,
                        EnemyKind::YvBoss,
                    );
                    brain.gunangle += boss.aux * rng.random_range(0.8..=1.0_f32).to_radians();
                    fired = true;
                    trauma.add(0.12);
                    boss.special_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                }
            }
        }
    }

    // GML `Alarm_1` (brain).
    if boss.attack_timer.just_finished() {
        if !boss.special_timer.finished() || boss.phase == BossPhase::Idle {
            // Fire pending or pre-intro: retry next tick (`alarm[1] = 1`).
            boss.attack_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
        } else if !can_shoot {
            // GML `Alarm_1:11-20`: `scrWalk(random_angle, 1, 10, 30)`, then
            // `scrTargetIsVisible(target, 90)` gates the gun (90 px, not
            // "any range").
            let head = glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU));
            gml_motion_add_clamp(&mut vel.0, head, 1.0, 4.0, dt);
            boss.target = head;
            brain.walk = rng.random_range(10.0..=30.0);
            let heading = if vel.0.length_squared() > 1e-8 {
                vel.0.normalize_or_zero()
            } else {
                boss.target
            };
            brain.gunangle = if los && dist <= 90.0 {
                aim
            } else {
                heading.y.atan2(heading.x)
            };
            // `alarm[1] = walk + irandom(10) + 10`, halved on cooldown.
            let wait = (brain.walk + rng.random_range(0.0..=10.0) + 10.0) * 0.5;
            boss.attack_timer = gml_alarm(wait);
        } else {
            brain.gunangle = aim;
            if los {
                let player_slow = player_velocity.length() < 30.0;
                if dist <= 64.0
                    && boss.pattern_index != SHOTGUN
                    && (player_slow || rng.random::<f32>() < 0.5)
                {
                    boss.pattern_index = SHOTGUN;
                    brain.ammo = 1;
                    // GML `Alarm_1:33` `instance_create(x, y, HitWarning)`.
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        HitWarning {
                            timer: GTimer::from_seconds(0.5, TimerMode::Once),
                        },
                        Pos(epos),
                    ));
                    boss.phase_timer = gml_alarm(rng.random_range(30.0..=40.0));
                    boss.special_timer = gml_alarm(10.0);
                } else if (dist <= 110.0 || rng.random::<f32>() < 1.0 / 3.0)
                    && boss.pattern_index != REVOLVER
                {
                    boss.pattern_index = REVOLVER;
                    brain.ammo = 5;
                    boss.phase_timer = gml_alarm(25.0 + rng.random_range(0.0..=15.0));
                    boss.special_timer = gml_alarm(5.0);
                } else {
                    boss.pattern_index = MINIGUN;
                    boss.aux = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
                    brain.ammo = 90;
                    brain.gunangle -=
                        (45.0 * boss.aux + rng.random_range(-10.0..=10.0)).to_radians();
                    boss.phase_timer = gml_alarm(180.0 + rng.random_range(0.0..=40.0));
                    boss.special_timer = gml_alarm(4.0);
                }
            } else {
                // GML `Alarm_1:56-63`: the wall on the line to the target
                // within 64 px -> `mp_potential_step_object` 4 px/step toward
                // the player plus `scrWalk(direction, 4, 10, 20)`; GML
                // `Alarm_1:62` `alarm[1] = walk` is overwritten below.
                let blocked_close = wall_centers.iter().any(|w| {
                    let t = ((w.x - epos.x) * to_player.x + (w.y - epos.y) * to_player.y)
                        / to_player.length_squared().max(1.0);
                    t > 0.0 && t < 1.0 && (*w - epos).length() < 64.0
                });
                if blocked_close {
                    pos.0 += to_player.normalize_or_zero() * (4.0 * 30.0) * (dt * 30.0);
                    let heading = vel.0.normalize_or_zero();
                    gml_motion_add_clamp(&mut vel.0, heading, 4.0, 4.0, dt);
                    brain.walk = rng.random_range(10.0..=20.0);
                    brain.gunangle = heading.y.atan2(heading.x);
                } else {
                    boss.pattern_index = BAZOOKA;
                    brain.ammo = 1;
                }
                // GML `Alarm_1:70-71` (no line of sight only).
                boss.phase_timer = gml_alarm(60.0);
                boss.special_timer = gml_alarm(7.0);
            }
            // GML `Alarm_1:74-77`: `alarm[4]` is never -1 here, so the brain
            // always re-arms at 5-15 and `walk` is zeroed - YV stands still
            // through the wall-adjacent branch too.
            boss.attack_timer = gml_alarm(rng.random_range(5.0..=15.0));
            brain.walk = 0.0;
        }
    }

    // GML `Step_0` locomotion: stand while loaded, drift while walking.
    if brain.ammo > 0 {
        brain.walk = 0.0;
    } else if brain.walk > 0.0 {
        gml_motion_add_clamp(&mut vel.0, boss.target, 0.5, 4.0, dt);
        brain.walk -= dt * 30.0;
        if brain.walk < 0.0 {
            brain.walk = 0.0;
        }
    }
    if vel.0.length() > 120.0 {
        vel.0 = vel.0.normalize() * 120.0;
    }
    pos.0 += vel.0 * dt;
    fired
}

// Tests: headless parity for routing, volley counts, orbit, difficulty.
