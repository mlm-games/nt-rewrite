//! Boss AI. Ported from the bevy reference `game/boss_ai.rs` (`boss_ai`
//! dispatcher plus every boss handler) with positions as [`Pos`] (`Vec2`)
//! instead of `Transform.translation` (`Vec3`).
//!
//! Render split: `Sprite`/`Anchor`/`Transform.rotation`/`Transform.scale`
//! writes, fire-strip swaps (`play_fire`), and `VfxSpawner` bursts stay
//! out — the render phase resolves visuals from sim state. Gameplay
//! effects are kept: movement impulses, fan/ring volleys with full combat
//! traits, `Explosion` + `Beam` spawns, pending-spawn queues, trauma, and
//! wall-break queues. Muzzle markers ride [`show_enemy_fire`] exactly like
//! normal enemies (no-op without a seeded [`SpriteAnim`]).
//!
//! Timer adaptation: bevy `Timer` -> [`GTimer`]; `tick(dt)` returns `()`,
//! then `just_finished()`/`finished()` are queried. The bevy
//! `short_ready_timer` (finished from birth) has no direct `GTimer`
//! equivalent, so the local [`ready_timer`] double-ticks a 10 ms `Once`
//! timer into the same observable state (finished, not just-finished).
//!
//! LOS adaptation: bevy traced wall *entities*
//! (`segment_hits_wall_query`); the headless build does the same through
//! the `(center, cell)` wall snapshot (`segment_hits_wall_legacy`).
//! Wall *breaking* during charges still
//! queues [`PendingWallBreak`]s against the wall entities.
//!
//! The `boss_ai` dispatcher carries 12 params (under the bevy_ecs 16-param
//! cap), so no record-split is needed. Handlers are plain functions over
//! snapshots (`&[(Vec2, Vec2)]` props, `&[(Vec2, (i32, i32))]` walls) so
//! tests can drive them without a world.

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::anim::SpriteAnim;
use crate::audio::AudioCue;
use crate::combat::{Explosion, PendingEnemySpawn};
use crate::comps_a::{
    BossIntro, DamageSource, GameCleanup, Health, Hitbox, LevelCleanup, NextHurt,
    PendingWallBreak, Player, Projectile, RaceState, Run, Team, Toast, Velocity,
    WallCell, WallTile, apply_gml_friction, gml_motion_add_clamp,
};
use crate::comps_b::{
    Beam, BossBrain, BossPhase, Enemy, EnemyBrain, HazardCloud, HurtAnim, HyperOrbitCrystal,
    InvisiWall, MomShot, Portal, PortalClear, Prop, ThroneBall, ThroneStatueProp,
};
use crate::data::{AreaId, EnemyKind};
use crate::enemies::show_enemy_fire;
use crate::enemy_data::{EnemyDef, enemy_def};
use crate::msg::Queue;
use crate::spatial::{Pos, clamp_to_arena, resolve_prop_collision};
use crate::time::{GTimer, TimerMode};

// ---------------------------------------------------------------------------
// Pattern helpers (bevy `boss_patterns.rs` parity, `glam::Vec2` throughout).
// ---------------------------------------------------------------------------

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

/// Aim ahead of a moving target (clamped 1.2 s lead, NaN-safe).
pub fn lead_target(
    shooter: glam::Vec2,
    target: glam::Vec2,
    target_vel: glam::Vec2,
    projectile_speed: f32,
) -> glam::Vec2 {
    let to = target - shooter;
    let dist = to.length();
    if projectile_speed <= 1.0 || dist <= 1.0 {
        return to.normalize_or_zero();
    }
    let time = (dist / projectile_speed).clamp(0.0, 1.2);
    (target + target_vel * time - shooter).normalize_or_zero()
}

/// `lines` evenly spaced directions covering the full circle.
pub fn split_line_dirs(base_angle: f32, lines: usize) -> Vec<f32> {
    if lines == 0 {
        return Vec::new();
    }
    let step = std::f32::consts::TAU / lines as f32;
    (0..lines).map(|i| base_angle + step * i as f32).collect()
}

/// Star burst points (full circle, `phase` offset).
pub fn star_angles(points: usize, phase: f32) -> Vec<f32> {
    ring_angles(points.max(1), phase)
}

/// Orbit-crystal count scales with loop (GML `cnumber = 3 + loops*2`:
/// 3/5/7…).
pub fn hyper_orbit_count(loop_count: u32) -> usize {
    3 + loop_count as usize * 2
}

/// Point on a circle around `center`.
pub fn orbit_point(center: glam::Vec2, radius: f32, angle: f32) -> glam::Vec2 {
    center + dir_from_angle(angle) * radius
}

// ---------------------------------------------------------------------------
// Shared firing / spawn helpers (boss-specific thin wrappers; enemy fire
// lives in `crate::enemies` and is NOT duplicated here).
// ---------------------------------------------------------------------------

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
/// `short_ready_timer` parity — see module docs).
fn ready_timer() -> GTimer {
    let mut t = GTimer::from_seconds(0.01, TimerMode::Once);
    t.tick(0.01);
    t.tick(0.0);
    t
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

/// Timed enemy explosion at a point (bevy `Explosion` spawn parity).
fn spawn_explosion(
    commands: &mut Commands,
    owner: Entity,
    kind: EnemyKind,
    pos: glam::Vec2,
    radius: f32,
    damage: i32,
    fuse: f32,
) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        Explosion {
            timer: GTimer::from_seconds(fuse, TimerMode::Once),
            radius,
            damage,
            team: Team::Enemy,
            hits_player: true,
            source: Some(DamageSource::enemy(owner, kind)),
        },
        Pos(pos),
    ));
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
        commands, owner, pos, dir, team, count, spread, speed, damage, lifetime, radius,
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
        commands, owner, pos, team, count, phase, speed, damage, lifetime, radius,
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

// ---------------------------------------------------------------------------
// Dispatcher.
// ---------------------------------------------------------------------------

/// Boss brains: enrage check, timer ticks, per-kind handler, arena clamp
/// (bevy `boss_ai` top to bottom; non-bosses `continue` before the first
/// timer tick, exactly like bevy).
#[allow(clippy::too_many_arguments)]
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
            &Enemy,
            &mut BossBrain,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            &mut Health,
            Option<&mut SpriteAnim>,
            Option<&HurtAnim>,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos, Option<&ThroneStatueProp>), With<Prop>>,
    walls: Query<(&WallCell, &Pos), (With<WallTile>, Without<Enemy>)>,
    children: Query<(Entity, &Enemy, &Pos), (With<Enemy>, Without<BossBrain>)>,
    portals: Query<Entity, With<Portal>>,
    wall_ids: Query<(Entity, &Pos), (With<WallTile>, Without<Enemy>)>,
    catalog: Res<repame_anim::AnimCatalog>,
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
        .filter(|(_, e, _)| {
            e.kind == EnemyKind::Ballguy || e.kind == EnemyKind::SuperFrog
        })
        .map(|(_, e, _)| if e.kind == EnemyKind::SuperFrog { 2 } else { 1 })
        .sum();

    for (entity, enemy, mut boss, mut brain, mut vel, mut pos, health, mut anim, hurt) in
        &mut bosses
    {
        let def = enemy_def(enemy.kind);
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

        if !matches!(
            enemy.kind,
            EnemyKind::BigBandit | EnemyKind::BigBanditLoop | EnemyKind::FrogQueen
        ) {
            apply_gml_friction(&mut vel.0, 0.4, dt);
        }

        let fired = match enemy.kind {
            EnemyKind::BigBandit | EnemyKind::BigBanditLoop => big_bandit_ai(
                &mut commands,
                &mut trauma,
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
                &prop_shapes,
                &wall_shapes,
            ),
            EnemyKind::BigDog | EnemyKind::BigDogLoop => big_dog_ai(
                &mut commands,
                &mut trauma,
                entity,
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
                run.loop_count,
                missile_count,
            ),
            EnemyKind::LilHunter | EnemyKind::LilHunterLoop => lil_hunter_ai(
                &mut commands,
                &mut trauma,
                &mut toast,
                entity,
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
            EnemyKind::ThroneII => throne_ii_ai(
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
            ),
            EnemyKind::Hyper => hyper_ai(
                &mut commands,
                &mut trauma,
                entity,
                &mut boss,
                &mut vel,
                &mut pos,
                def,
                epos,
                player_pos,
                dt,
                run.loop_count,
                &mut run,
            ),
            EnemyKind::Mom => mom_ai(
                &mut commands,
                &mut trauma,
                entity,
                &mut boss,
                &mut vel,
                &mut pos,
                def,
                epos,
                player_pos,
                dir,
                dt,
                &prop_shapes,
                run.loop_count,
            ),
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
                technomancer_ai(
                    &mut commands,
                    &mut trauma,
                    &mut boss,
                    &mut vel,
                    epos,
                    run.loop_count,
                );
                false
            }
            EnemyKind::Captain => captain_ai(
                &mut commands,
                &mut trauma,
                entity,
                &mut boss,
                &mut vel,
                &mut pos,
                def,
                epos,
                player_pos,
                dir,
                dt,
                &prop_shapes,
                &wall_shapes,
            ),
            EnemyKind::OldGuardian => old_guardian_ai(
                &mut commands,
                &mut trauma,
                entity,
                &mut boss,
                &mut vel,
                &mut pos,
                def,
                epos,
                player_pos,
                dir,
                dt,
                &prop_shapes,
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

        clamp_to_arena(&mut pos.0, def.radius);
    }
}

// ---------------------------------------------------------------------------
// Big Bandit.
// ---------------------------------------------------------------------------

/// Charge telegraphs + shotgun bursts while kiting (bevy `big_bandit_ai`
/// parity; flip/scale/fire-strip writes omitted as visual-only).
/// Returns true when a shot was fired (muzzle marker).
#[allow(clippy::too_many_arguments)]
fn big_bandit_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
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
    props: &[(glam::Vec2, glam::Vec2)],
    walls: &[(glam::Vec2, (i32, i32))],
) -> bool {
    let looped = def.name.contains("Loop");
    let kind = if looped {
        EnemyKind::BigBanditLoop
    } else {
        EnemyKind::BigBandit
    };
    let mut fired = false;

    apply_gml_friction(&mut vel.0, 0.4, dt);

    match boss.phase {
        BossPhase::Idle | BossPhase::Cooldown => {
            if brain.walk > 0.0 {
                let face = (boss.target - epos).normalize_or_zero();
                let move_dir = if vel.0.length_squared() > 0.0 {
                    vel.0.normalize_or_zero()
                } else {
                    -dir
                };
                gml_motion_add_clamp(&mut vel.0, move_dir, 1.0, 3.0, dt);
                if face.length_squared() > 0.0001 {
                    gml_motion_add_clamp(&mut vel.0, face, 0.5, 3.0, dt);
                }
                brain.walk -= dt * 30.0;
                if brain.walk < 0.0 {
                    brain.walk = 0.0;
                }
            }
            pos.0 += vel.0 * dt;
            resolve_prop_collision(&mut pos.0, def.radius, props.iter().copied());

            if boss.attack_timer.just_finished() {
                let dist = epos.distance(player_pos);
                // Bevy gates the burst on wall-ENTITY line of sight
                // (`segment_hits_wall_query`), not the mask trace.
                let wall_centers: Vec<glam::Vec2> =
                    walls.iter().map(|(c, _)| *c).collect();
                let los = !crate::walls::segment_hits_wall_legacy(epos, player_pos, &wall_centers);
                let period = if looped {
                    (20.0 + rand::rng().random_range(0.0..50.0)) / 30.0
                } else {
                    (30.0 + rand::rng().random_range(0.0..60.0)) / 30.0
                };
                boss.attack_timer = GTimer::from_seconds(period, TimerMode::Once);

                let intro_done = boss.pattern_index > 0 || boss.phase == BossPhase::Cooldown;
                let should_burst = los
                    && dist > 48.0
                    && dist < 240.0
                    && intro_done
                    && rand::rng().random::<f32>() < 2.0 / 3.0;
                if should_burst {
                    brain.ammo = if looped { 15 } else { 10 };
                    brain.burst_left = brain.ammo as usize;
                    brain.burst_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                    brain.gunangle = dir.y.atan2(dir.x);
                    boss.set_phase(BossPhase::Radial, 2.5);
                    boss.attack_timer = GTimer::from_seconds(70.0 / 30.0, TimerMode::Once);
                } else {
                    let mut chargewait = boss.pattern_index.saturating_add(1);
                    if dist < 96.0 {
                        chargewait = chargewait.saturating_add(1);
                    }
                    boss.pattern_index = chargewait;

                    let intro_charge = epos.distance(boss.home) < 1.0;
                    if chargewait >= 2 || intro_charge {
                        boss.pattern_index = 0;
                        boss.target = player_pos;
                        brain.gunangle = dir.y.atan2(dir.x);
                        boss.set_phase(BossPhase::Telegraph, 15.0 / 30.0);
                        vel.0 *= 0.2;
                        trauma.add(0.08);
                    }
                }

                let away = -dir;
                let ang =
                    away.y.atan2(away.x) + rand::rng().random_range(-90f32..90.0).to_radians();
                vel.0 = glam::Vec2::new(ang.cos(), ang.sin()) * (0.4 * 30.0);
                brain.walk = if dist > 64.0 {
                    40.0
                } else {
                    10.0 + rand::rng().random_range(0.0..10.0)
                };
            }
        }

        BossPhase::Radial => {
            brain.burst_timer.tick(dt);
            brain.walk = 0.0;
            if brain.ammo > 0 && brain.burst_timer.just_finished() {
                let spread = rand::rng().random_range(-15f32..15.0).to_radians();
                let ang = brain.gunangle + spread;
                let sdir = glam::Vec2::new(ang.cos(), ang.sin());

                fire_projectile(
                    commands,
                    owner,
                    epos + sdir * 20.0,
                    sdir,
                    Team::Enemy,
                    240.0,
                    3,
                    3.2,
                    4.5,
                    120.0,
                    kind,
                );
                fired = true;

                gml_motion_add_clamp(&mut vel.0, -sdir, 1.0, 5.0, dt);

                brain.ammo -= 1;
                brain.burst_left = brain.burst_left.saturating_sub(1);
                if looped && brain.ammo == 7 {
                    brain.gunangle = dir.y.atan2(dir.x);
                }
                brain.burst_timer = GTimer::from_seconds(4.0 / 30.0, TimerMode::Once);
            }
            if brain.ammo == 0 {
                boss.set_phase(
                    BossPhase::Cooldown,
                    (60.0 + rand::rng().random_range(0.0..10.0)) / 30.0,
                );
                boss.attack_timer = GTimer::from_seconds(
                    (60.0 + rand::rng().random_range(0.0..10.0)) / 30.0,
                    TimerMode::Once,
                );
            }
            pos.0 += vel.0 * dt;
            resolve_prop_collision(&mut pos.0, def.radius, props.iter().copied());
        }

        BossPhase::Telegraph => {
            vel.0 *= 0.5_f32.powf(dt * 30.0);
            pos.0 += vel.0 * dt;
            if boss.phase_timer.just_finished() {
                let charge_dir = (boss.target - epos).normalize_or_zero();
                brain.gunangle = charge_dir.y.atan2(charge_dir.x);
                vel.0 = charge_dir * (2.0 * 30.0);
                boss.set_phase(BossPhase::Charging, 0.55);
                trauma.add(0.18);
            }
        }

        BossPhase::Charging => {
            let move_dir = vel.0.normalize_or_zero();
            let gun = glam::Vec2::new(brain.gunangle.cos(), brain.gunangle.sin());
            gml_motion_add_clamp(&mut vel.0, move_dir, 2.0, 5.0, dt);
            gml_motion_add_clamp(&mut vel.0, gun, 2.0, 5.0, dt);
            let before = pos.0;
            pos.0 += vel.0 * dt;
            queue_wall_breaks_along_segment(commands, walls, before, pos.0, def.radius * 0.9);
            resolve_prop_collision(&mut pos.0, def.radius, props.iter().copied());
            if boss.phase_timer.just_finished() {
                boss.set_phase(BossPhase::Cooldown, 0.55);
                vel.0 *= 0.15;
                boss.pattern_index = 0;
            }
        }

        _ => {
            boss.set_phase(BossPhase::Idle, 0.1);
        }
    }
    fired
}

// ---------------------------------------------------------------------------
// Scrap Boss (Big Dog).
// ---------------------------------------------------------------------------

/// Verbatim `objects/ScrapBoss` law: decide tick (`Alarm_0`), spin-fire
/// tick (`Alarm_1`), walk/homing locomotion (`Other_10`).
///
/// Register map: `attack_timer` = `alarm[0]`, `special_timer` = `alarm[1]`,
/// `brain.ammo` = `ammo`, `boss.aux` = `turn`, `brain.walk` = `walk`,
/// `brain.gunangle` = `gunangle` (radians), `boss.target` = move heading.
/// (Wall bounce from `Collision_Wall` is out: port enemies do not collide
/// walls, they clamp to the arena.)
#[allow(clippy::too_many_arguments)]
fn big_dog_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    owner: Entity,
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
    loop_count: u32,
    missiles: usize,
) -> bool {
    let kind = if def.name.contains("Loop") {
        EnemyKind::BigDogLoop
    } else {
        EnemyKind::BigDog
    };
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

    // GML `Alarm_1` (spin fire).
    if boss.special_timer.just_finished() {
        if brain.ammo > 0 {
            brain.ammo -= 1;
            // Drift toward the target fanned by the spin direction.
            let drift =
                to_player.y.atan2(to_player.x) + boss.aux * 80.0_f32.to_radians();
            gml_motion_add_clamp(
                &mut vel.0,
                glam::Vec2::from_angle(drift),
                0.3,
                3.0,
                dt,
            );
            let count = 6 + loop_count as usize;
            let step = 360.0 / count as f32;
            for _ in 0..count {
                let sdir = glam::Vec2::from_angle(brain.gunangle);
                // GML spawn offset: `(24·cos g, 16·sin g)`.
                let off = glam::Vec2::new(
                    24.0 * brain.gunangle.cos(),
                    16.0 * brain.gunangle.sin(),
                );
                fire_projectile(
                    commands, owner, epos + off, sdir, Team::Enemy,
                    60.0, 3, 3.0, 4.0, 120.0, kind,
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
            let frac = (health.hp as f32 / health.max.max(1) as f32).clamp(0.0, 1.0);
            brain.ammo = (10.0 + 10.0 * (1.0 - frac)).round() as u8;
            boss.aux = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            brain.walk = 0.0;
            vel.0 = glam::Vec2::ZERO;
        } else {
            brain.ammo = 0;
            if rng.random::<f32>() < 1.0 / (3.0 + missiles as f32 / 2.0) {
                for _ in 0..3 {
                    commands.spawn(PendingEnemySpawn {
                        kind: EnemyKind::ScrapBossMissile,
                        pos: epos,
                        difficulty: 1.0,
                        loops: loop_count,
                    });
                }
                boss.attack_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
            } else {
                brain.walk = rng.random_range(20.0..=30.0);
                let head = glam::Vec2::from_angle(
                    rng.random_range(0.0..std::f32::consts::TAU),
                );
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
    if brain.ammo > 0 && vel.0.length_squared() > 0.001 {
        vel.0 = vel.0.normalize() * 30.0;
    }
    pos.0 += vel.0 * dt;
    resolve_prop_collision(&mut pos.0, def.radius, props.iter().copied());
    let _ = trauma;
    fired
}

// ---------------------------------------------------------------------------
// Lil Hunter.
// ---------------------------------------------------------------------------

/// Verbatim `objects/LilHunter` + `objects/LilHunterFly` law: bouncer fan
/// and sniper hose (`Alarm_1`), anti-camp liftoff (`Alarm_2`), walk/dodge
/// locomotion (`Other_10`), teleport flight with landing fire ring, and
/// the 80-flame death ring (`Destroy_0`).
///
/// Register map: `attack_timer` = `alarm[1]`, `special_timer` = `alarm[2]`,
/// `brain.ammo` = `spawns`, `brain.burst_left` = init/intro flag,
/// `brain.walk` = `walk`, `brain.dash` = `dodge`, `brain.gunangle` =
/// `gunangle` (radians), `boss.target` = move heading, `boss.aux` = fly
/// height `z`, `boss.phase` Teleport = airborne (`pattern_index` 1 ascend,
/// 2 descend). (Rogue force-liftoff, taunts, and `LilHunterDie`/music cues
/// are out as meta/visual.)
#[allow(clippy::too_many_arguments)]
fn lil_hunter_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    toast: &mut Toast,
    owner: Entity,
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
    run: &mut Run,
) -> bool {
    let looped = def.name.contains("Loop");
    let kind = if looped {
        EnemyKind::LilHunterLoop
    } else {
        EnemyKind::LilHunter
    };
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

    // --- Flight (`LilHunterFly/Step_0`). ---
    if boss.phase == BossPhase::Teleport {
        if boss.pattern_index == 1 {
            // Ascend 8 px/tick until offscreen (port: height past 160).
            boss.aux -= 240.0 * dt;
            if boss.aux <= -160.0 {
                if others == 0 {
                    // Nobody left: drop an `IDPDSpawn` and leave.
                    commands.spawn(PendingEnemySpawn {
                        kind: EnemyKind::IdpdGrunt,
                        pos: epos,
                        difficulty: 1.0,
                        loops: loop_count,
                    });
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
                lil_hunter_fire_ring(commands, pos.0);
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
    let los = dist < 512.0
        && !crate::walls::segment_hits_wall_legacy(epos, player_pos, &wall_centers);

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
                            commands, owner, epos + sdir * 20.0, sdir, Team::Enemy,
                            90.0, 3, 3.0, 4.0, 120.0, kind,
                        );
                        addang += 10.0;
                    }
                    fired = true;
                    let d = boss.attack_timer.duration()
                        / (1.0 + loop_count as f32 * 2.0);
                    boss.attack_timer =
                        GTimer::from_seconds(d.max(1.0 / 30.0), TimerMode::Once);
                    // Retreat from the target.
                    let back = (-dir
                        + glam::Vec2::from_angle(
                            rng.random_range(-10.0..=10.0_f32).to_radians(),
                        ))
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
                    + glam::Vec2::from_angle(
                        rng.random_range(-10.0..=10.0_f32).to_radians(),
                    ))
                .normalize_or_zero();
                vel.0 = back * 12.0;
                boss.target = back;
                brain.walk = rng.random_range(8.0..=12.0);
                boss.attack_timer =
                    GTimer::from_seconds(brain.walk / 30.0, TimerMode::Once);
                brain.gunangle = aim;
            }
        } else if rng.random::<f32>() < 0.5
            && brain.ammo > 0
            && (health.hp as f32) < (health.max as f32) * (brain.ammo as f32) / 6.0
        {
            // IDPD summon through converging portal charges: one portal
            // per wave, each rolling the GML spawn table.
            brain.walk = 0.0;
            for _ in 0..1 + (loop_count.saturating_sub(1)) as usize {
                run.popolevel += 1;
                for kind in crate::idpd::roll_idpd_table(
                    loop_count,
                    run.area,
                    run.popolevel,
                    true,
                ) {
                    commands.spawn(PendingEnemySpawn {
                        kind,
                        pos: epos,
                        difficulty: 1.0,
                        loops: loop_count,
                    });
                }
            }
            brain.ammo -= 1;
            let d = boss.attack_timer.duration() + 45.0 / 30.0;
            boss.attack_timer = GTimer::from_seconds(d, TimerMode::Once);
        } else {
            // Reposition: long retreat, brain ticks 3x faster.
            let back = (-dir
                + glam::Vec2::from_angle(
                    rng.random_range(-10.0..=10.0_f32).to_radians(),
                ))
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

/// The 80-flame landing/death ring (`sprFireLilHunter` `TrapFire` at
/// 2 px/tick stepping 4.5 degrees): short-lived fire clouds fanning out
/// from the impact.
pub fn lil_hunter_fire_ring(commands: &mut Commands, at: glam::Vec2) {
    use crate::data::HazardKind;
    let mut ang = rand::rng().random_range(0.0..std::f32::consts::TAU);
    for _ in 0..80 {
        ang += 4.5_f32.to_radians();
        let d = glam::Vec2::from_angle(ang);
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            Team::Enemy,
            HazardCloud {
                kind: HazardKind::Fire,
                radius: 10.0,
                damage: 2,
                timer: GTimer::from_seconds(4.0, TimerMode::Once),
                tick: GTimer::from_seconds(0.5, TimerMode::Repeating),
            },
            Pos(at + d * 36.0),
        ));
    }
}

/// Per-kind taunt line (`snd<Kind>Taunt`, GML-verbatim names — e.g.
/// FrogQueen taunts `sndBallMamaTaunt`, YV `sndGunGodTaunt`). Kinds
/// without a GML taunt return `None`.
pub fn taunt_cue_name(kind: EnemyKind) -> Option<&'static str> {
    match kind {
        EnemyKind::BigBandit | EnemyKind::BigBanditLoop => Some("sndBigBanditTaunt"),
        EnemyKind::BigDog | EnemyKind::BigDogLoop => Some("sndBigDogTaunt"),
        EnemyKind::Mom => Some("sndBallMamaTaunt"),
        EnemyKind::Hyper => Some("sndHyperCrystalTaunt"),
        EnemyKind::LilHunter | EnemyKind::LilHunterLoop => Some("sndLilHunterTaunt"),
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
        let gate = if matches!(enemy.kind, EnemyKind::Throne) { 150 } else { 50 };
        if brain.tauntdelay > gate {
            brain.taunt = true;
            cues.push(AudioCue {
                name: cue_name,
                volume: 0.7,
                variance: 0.05,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Throne.
// ---------------------------------------------------------------------------

/// Verbatim `objects/Nothing` law: walk-in brain with statue-break
/// (`Alarm_1`), mirrored Horror-fan triplets (`Alarm_2`), and capped
/// stomp locomotion (`Other_10`).
///
/// Register map: `attack_timer` = `alarm[1]`, `special_timer` = `alarm[2]`,
/// `brain.ammo` = `ammo`, `brain.gunangle` = `addangle` (radians),
/// `boss.aux` = `mode`, `brain.walk` = `walk`, `boss.target` = `walkdir`
/// heading, `pattern_index` = `introwalk`, `brain.burst_left` = `dmg`
/// counter (reset while hurt). The `NothingBeam` charge delay rides
/// `BossPhase::Telegraph`. (Statue art, flame sprites, and hurt-voice
/// tiers are out as visual/audio.)
#[allow(clippy::too_many_arguments)]
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

    // GML `Create_0`: `alarm[1] = 30`, `friction = 1.7` (dispatcher
    // already applies 0.4, so top up the remaining 1.3 here).
    if boss.aux == 0.0 && brain.burst_left == 0 && boss.phase == BossPhase::Idle {
        brain.burst_left = 1;
        boss.pattern_index = 0;
        boss.attack_timer = GTimer::from_seconds(1.0 /* 30 ticks */, TimerMode::Once);
    }
    apply_gml_friction(&mut vel.0, 1.3, dt);

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
                let ang =
                    270.0_f32.to_radians() + (brain.gunangle + off.to_radians()) * flip;
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
                    for _ in 0..1 + loop_count {
                        commands.spawn(PendingEnemySpawn {
                            kind: EnemyKind::PalaceGuardian,
                            pos: spos,
                            difficulty: 1.0,
                            loops: loop_count,
                        });
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
                    brain.gunangle = [
                        -30.0_f32,
                        -20.0,
                        -10.0,
                        0.0,
                        10.0,
                        20.0,
                        30.0,
                    ][rng.random_range(0..7)]
                    .to_radians();
                    brain.ammo = (3 + loop_count) as u8;
                    boss.special_timer = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                } else {
                    boss.attack_timer = GTimer::from_seconds(60.0 / 30.0, TimerMode::Once);
                    brain.gunangle =
                        [-20.0_f32, -10.0, 0.0, 10.0, 20.0][rng.random_range(0..5)]
                            .to_radians();
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

// ---------------------------------------------------------------------------
// Throne II.
// ---------------------------------------------------------------------------

/// Verbatim `objects/Nothing2` law: strafe walk (`Alarm_0`), the three
/// rotating attacks with haste (`Alarm_1`), and capped surge locomotion
/// (`Other_10`).
///
/// Register map: `attack_timer` = `alarm[1]`, `special_timer` = `alarm[0]`
/// (strafe), `phase_timer` = `alarm[2]` (intro), `boss.aux` = `attack`,
/// `brain.ammo` = `shots`, `brain.gunangle` = `aimdir` (radians),
/// `brain.strafe_dir` = `side`, `brain.dash` = `flip`, `boss.target` =
/// `walkdir` heading, `brain.walk` = `walk`. (Top kills and taunts are
/// out as visual/audio.)
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
        boss.target =
            glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU));
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
        let head = glam::Vec2::from_angle(
            aim + (50.0 + rng.random_range(0.0..=20.0)) * brain.dash,
        );
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
            brain.gunangle =
                aim + (30.0 + rng.random_range(0.0..=10.0)) * brain.strafe_dir;
            let sdir = glam::Vec2::from_angle(brain.gunangle);
            fire_projectile(
                commands, owner, epos + sdir * 20.0, sdir, Team::Enemy,
                rng.random_range(150.0..=180.0), 12, 3.0, 8.0, 200.0,
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
                    commands, owner, epos + sdir * 20.0, sdir, Team::Enemy,
                    240.0, 5, 2.5, 5.0, 150.0, EnemyKind::ThroneII,
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

// ---------------------------------------------------------------------------
// Hyper Crystal.
// ---------------------------------------------------------------------------

/// Slow drifter seeding orbit crystals + seeker detonations (bevy
/// `hyper_ai` parity).
#[allow(clippy::too_many_arguments)]
fn hyper_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    owner: Entity,
    boss: &mut BossBrain,
    vel: &mut Velocity,
    pos: &mut Pos,
    def: EnemyDef,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    dt: f32,
    loop_count: u32,
    run: &mut Run,
) -> bool {
    let desired = (boss.home - epos) * 0.4 + (player_pos - epos) * 0.1;
    if desired.length_squared() > 1.0 {
        vel.0 += desired.normalize_or_zero() * def.accel * 0.12 * dt;
    }
    limit_velocity(vel, def.speed.max(35.0));
    pos.0 += vel.0 * dt;

    if boss.attack_timer.just_finished() && boss.pattern_index == 0 {
        boss.pattern_index += 1;
        hyper_ensure_orbit(commands, owner, epos, loop_count, boss.enraged, run);
    } else if boss.attack_timer.just_finished() {
        boss.pattern_index += 1;
    }

    if boss.special_timer.just_finished() && epos.distance(player_pos) > 220.0 {
        hyper_search_detonate(commands, trauma, owner, player_pos, loop_count, boss.enraged);
        boss.set_phase(BossPhase::Cooldown, 0.8);
    }
    false
}

/// Seeker detonation on the player: contact blast + beam ring (bevy
/// `hyper_search_detonate` parity).
pub fn hyper_search_detonate(
    commands: &mut Commands,
    trauma: &mut Trauma,
    owner: Entity,
    player_pos: glam::Vec2,
    loop_count: u32,
    enraged: bool,
) {
    trauma.add(0.3);
    let lasers = 7 + loop_count as usize * 2 + usize::from(enraged);
    spawn_explosion(commands, owner, EnemyKind::Hyper, player_pos, 90.0, 6, 0.03);
    for angle in ring_angles(lasers, 0.0) {
        let dir = dir_from_angle(angle);
        spawn_enemy_beam(commands, player_pos + dir * 210.0, dir, 420.0, 12.0, 2, 0.28);
    }
}

/// Seed the orbit-crystal shell (GML: `cnumber = 3 + loops*2` real
/// `LaserCrystal` enemies — `InvLaserCrystal`/invariants in area 104 —
/// which the core then herds; each spawn decrements the kill count).
pub fn hyper_ensure_orbit(
    commands: &mut Commands,
    owner: Entity,
    pos: glam::Vec2,
    loop_count: u32,
    enraged: bool,
    run: &mut Run,
) {
    let wanted = (hyper_orbit_count(loop_count) + usize::from(enraged)).min(12);
    let n = wanted;
    for i in 0..n {
        let angle = i as f32 / n as f32 * std::f32::consts::TAU;
        let radius = 70.0 + (i % 3) as f32 * 12.0;
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
            },
            Health {
                hp,
                max: hp,
                invuln: ready_timer(),
            },
            NextHurt::default(),
            Hitbox {
                radius: def.radius,
            },
            Velocity(glam::Vec2::ZERO),
            Pos(pos + dir_from_angle(angle) * radius),
            HyperOrbitCrystal {
                owner,
                angle,
                radius,
                angular_speed: 1.15 + (i as f32) * 0.04,
                fire_timer: GTimer::from_seconds(
                    1.4 + (i % 3) as f32 * 0.35,
                    TimerMode::Repeating,
                ),
            },
        ));
    }
}

/// Orbit crystals around their core, firing beams; orphaned crystals
/// drift and pop (bevy `tick_hyper_orbit_crystals` parity, `Pos`-based).
pub fn tick_hyper_orbit_crystals(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut Pos, &mut Velocity, &mut HyperOrbitCrystal)>,
    cores: Query<&Pos, (With<Enemy>, Without<HyperOrbitCrystal>)>,
) {
    let dt = time.delta_secs;
    for (entity, mut pos, mut vel, mut crystal) in q.iter_mut() {
        let Ok(core_pos) = cores.get(crystal.owner) else {
            vel.0 *= 0.92;
            if vel.0.length() < 5.0 {
                commands.entity(entity).despawn();
            }
            continue;
        };

        crystal.angle += crystal.angular_speed * dt;
        let center = core_pos.0;
        pos.0 = center + dir_from_angle(crystal.angle) * crystal.radius;
        vel.0 = glam::Vec2::ZERO;

        crystal.fire_timer.tick(dt);
        if !crystal.fire_timer.just_finished() {
            continue;
        }

        let origin = pos.0;
        let aim = dir_from_angle(crystal.angle + std::f32::consts::FRAC_PI_2);
        spawn_enemy_beam(&mut commands, origin + aim * 210.0, aim, 420.0, 12.0, 2, 0.28);
    }
}

// ---------------------------------------------------------------------------
// Mom.
// ---------------------------------------------------------------------------

/// Kiting spore ring + egg spawns (bevy `mom_ai` parity).
#[allow(clippy::too_many_arguments)]
fn mom_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    owner: Entity,
    boss: &mut BossBrain,
    vel: &mut Velocity,
    pos: &mut Pos,
    def: EnemyDef,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    dir: glam::Vec2,
    dt: f32,
    props: &[(glam::Vec2, glam::Vec2)],
    loops: u32,
) -> bool {
    let mut fired = false;

    let desired = if epos.distance(player_pos) < 120.0 {
        -dir
    } else {
        dir
    };
    vel.0 += desired * def.accel * 0.5 * dt;
    limit_velocity(vel, def.speed.max(50.0));
    pos.0 += vel.0 * dt;
    resolve_prop_collision(&mut pos.0, def.radius, props.iter().copied());

    if boss.attack_timer.just_finished() {
        fire_ring_with_kind(
            commands,
            owner,
            epos,
            Team::Enemy,
            10 + usize::from(boss.enraged) * 4,
            boss.pattern_index as f32 * 0.11,
            90.0,
            2,
            2.5,
            5.0,
            EnemyKind::Mom,
        );
        fired = true;
        boss.pattern_index += 1;
        trauma.add(0.12);
    }

    if boss.special_timer.just_finished() {
        boss.set_phase(BossPhase::Spawning, 0.4);
        for i in 0..3 {
            let a = i as f32 * std::f32::consts::TAU / 3.0 + boss.pattern_index as f32 * 0.4;
            commands.spawn(PendingEnemySpawn {
                kind: EnemyKind::FrogEgg,
                pos: epos + glam::Vec2::new(a.cos(), a.sin()) * 48.0,
                difficulty: difficulty_for_loop(boss.enraged),
                loops,
            });
        }
        trauma.add(0.18);
    }

    if matches!(boss.phase, BossPhase::Spawning) && boss.phase_timer.just_finished() {
        boss.set_phase(BossPhase::Idle, 0.1);
    }
    fired
}

// ---------------------------------------------------------------------------
// Frog Queen.
// ---------------------------------------------------------------------------

/// Verbatim `objects/FrogQueen` law: aim-drift brain (`Alarm_1`), the
/// hatch stream into a single gas mortar (`Alarm_2`), and capped waddle
/// locomotion (`Other_10`).
///
/// Register map: `attack_timer` = `alarm[1]`, `special_timer` = `alarm[2]`,
/// `brain.ammo` = `ammo`, `brain.walk` = `walk`, `boss.target` = move
/// heading (`direction`), `brain.gunangle` = `gunangle` (radians),
/// `pattern_index` = `intro`, `brain.burst_left` = init flag. (Friction 0,
/// taunts, `FrogQueenDeath`/music cues, and the frog-pistol tribute are
/// out as feel/visual/meta.)
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
    let los = dist < 512.0
        && !crate::walls::segment_hits_wall_legacy(epos, player_pos, &wall_centers);

    // GML `Alarm_1` (brain).
    if boss.attack_timer.just_finished() {
        boss.attack_timer =
            GTimer::from_seconds((30.0 + rng.random_range(0.0..=20.0)) / 30.0, TimerMode::Once);
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
                boss.attack_timer =
                    GTimer::from_seconds(brain.walk / 30.0, TimerMode::Once);
                brain.ammo = (2 + loops) as u8;
                boss.special_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
            }
        }
    }

    // GML `Alarm_2`: hatch stream while loaded, then one gas mortar.
    if boss.special_timer.just_finished() {
        if brain.ammo > 0 {
            if frogs < 8 {
                commands.spawn(PendingEnemySpawn {
                    kind: EnemyKind::FrogEgg,
                    pos: epos,
                    difficulty: 1.0,
                    loops,
                });
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

// ---------------------------------------------------------------------------
// Technomancer.
// ---------------------------------------------------------------------------

/// Stationary summoner alternating Necromancer/Freak + Freak packs
/// (bevy `technomancer_ai` parity).
fn technomancer_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    boss: &mut BossBrain,
    vel: &mut Velocity,
    epos: glam::Vec2,
    loops: u32,
) {
    if boss.attack_timer.just_finished() {
        boss.pattern_index += 1;
        let kind = if boss.pattern_index % 2 == 0 {
            EnemyKind::Necromancer
        } else {
            EnemyKind::Freak
        };
        let ang = boss.pattern_index as f32 * 1.7;
        commands.spawn(PendingEnemySpawn {
            kind,
            pos: epos + glam::Vec2::new(ang.cos(), ang.sin()) * 90.0,
            difficulty: difficulty_for_loop(boss.enraged),
            loops,
        });
        trauma.add(0.1);
    }

    if boss.special_timer.just_finished() {
        let n = if boss.enraged { 4 } else { 2 };
        for i in 0..n {
            let a = i as f32 * (std::f32::consts::TAU / n as f32);
            commands.spawn(PendingEnemySpawn {
                kind: EnemyKind::Freak,
                pos: epos + glam::Vec2::new(a.cos(), a.sin()) * 110.0,
                difficulty: 1.15,
                loops,
            });
        }
        trauma.add(0.22);
    }

    vel.0 = glam::Vec2::ZERO;
}

// ---------------------------------------------------------------------------
// Captain.
// ---------------------------------------------------------------------------

/// Kiting fan shooter with charge/teleport telegraphs (bevy `captain_ai`
/// parity; scale pulse omitted as visual-only).
#[allow(clippy::too_many_arguments)]
fn captain_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    owner: Entity,
    boss: &mut BossBrain,
    vel: &mut Velocity,
    pos: &mut Pos,
    def: EnemyDef,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    dir: glam::Vec2,
    dt: f32,
    props: &[(glam::Vec2, glam::Vec2)],
    walls: &[(glam::Vec2, (i32, i32))],
) -> bool {
    let mut fired = false;

    match boss.phase {
        BossPhase::Idle | BossPhase::Cooldown => {
            let desired = if epos.distance(player_pos) < 100.0 {
                -dir
            } else {
                dir
            };
            vel.0 += desired * def.accel * 0.65 * dt;
            limit_velocity(vel, def.speed);
            pos.0 += vel.0 * dt;
            resolve_prop_collision(&mut pos.0, def.radius, props.iter().copied());

            if boss.attack_timer.just_finished() {
                fire_fan_with_kind(
                    commands,
                    owner,
                    epos,
                    dir,
                    Team::Enemy,
                    def.bullets_per_shot.max(5),
                    def.fan_spread,
                    def.projectile_speed,
                    def.projectile_damage,
                    def.projectile_lifetime,
                    def.projectile_radius,
                    EnemyKind::Captain,
                );
                fired = true;
            }

            if boss.special_timer.just_finished() && epos.distance(player_pos) < 560.0 {
                boss.target = player_pos;
                boss.set_phase(BossPhase::Telegraph, 0.22);
                vel.0 *= 0.25;
            }
        }

        BossPhase::Telegraph => {
            vel.0 *= 0.8_f32.powf(dt * crate::SIM_HZ as f32);
            if boss.phase_timer.just_finished() {
                if boss.pattern_index % 2 == 0 {
                    boss.set_phase(BossPhase::Charging, 0.35);
                    vel.0 = (boss.target - epos).normalize_or_zero() * 720.0;
                    trauma.add(0.14);
                } else {
                    // Bevy sets Teleport then immediately Cooldown (the
                    // teleport lands, then the cooldown runs).
                    boss.set_phase(BossPhase::Teleport, 0.05);
                    pos.0 = player_pos + dir * 90.0;
                    trauma.add(0.2);
                    boss.set_phase(BossPhase::Cooldown, 0.4);
                }
                boss.pattern_index += 1;
            }
        }

        BossPhase::Charging => {
            let before = pos.0;
            pos.0 += vel.0 * dt;
            let after = pos.0;
            queue_wall_breaks_along_segment(commands, walls, before, after, def.radius * 0.9);

            if boss.phase_timer.just_finished() {
                vel.0 *= 0.15;
                boss.set_phase(BossPhase::Cooldown, 0.45);
                fire_ring_with_kind(
                    commands,
                    owner,
                    pos.0,
                    Team::Enemy,
                    12,
                    boss.pattern_index as f32 * 0.17,
                    140.0,
                    3,
                    2.0,
                    4.0,
                    EnemyKind::Captain,
                );
                fired = true;
            }
        }

        _ => {
            boss.set_phase(BossPhase::Idle, 0.1);
        }
    }
    fired
}

// ---------------------------------------------------------------------------
// Old Guardian.
// ---------------------------------------------------------------------------

/// Kiting fan + enrage-scaled ring (bevy `old_guardian_ai` parity).
#[allow(clippy::too_many_arguments)]
fn old_guardian_ai(
    commands: &mut Commands,
    trauma: &mut Trauma,
    owner: Entity,
    boss: &mut BossBrain,
    vel: &mut Velocity,
    pos: &mut Pos,
    def: EnemyDef,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    dir: glam::Vec2,
    dt: f32,
    props: &[(glam::Vec2, glam::Vec2)],
) -> bool {
    let mut fired = false;

    let desired = if epos.distance(player_pos) < 90.0 {
        -dir
    } else {
        dir
    };
    vel.0 += desired * def.accel * 0.55 * dt;
    limit_velocity(vel, def.speed);
    pos.0 += vel.0 * dt;
    resolve_prop_collision(&mut pos.0, def.radius, props.iter().copied());

    if boss.attack_timer.just_finished() {
        fire_fan_with_kind(
            commands,
            owner,
            epos,
            dir,
            Team::Enemy,
            def.bullets_per_shot.max(4),
            def.fan_spread,
            def.projectile_speed,
            def.projectile_damage,
            def.projectile_lifetime,
            def.projectile_radius,
            EnemyKind::OldGuardian,
        );
        fired = true;
    }

    if boss.special_timer.just_finished() {
        fire_ring_with_kind(
            commands,
            owner,
            epos,
            Team::Enemy,
            10 + usize::from(boss.enraged) * 4,
            boss.pattern_index as f32 * 0.19,
            120.0,
            3,
            2.2,
            4.0,
            EnemyKind::OldGuardian,
        );
        fired = true;
        boss.pattern_index += 1;
        trauma.add(0.16);
    }
    fired
}

// ---------------------------------------------------------------------------
// YV (Gun God).
// ---------------------------------------------------------------------------

/// Verbatim `objects/YVBoss` law: weapon-switch brain (`Alarm_1`),
/// per-weapon fire tick (`Alarm_2`), cooldown gate (`Alarm_4`),
/// intro on first empty revolver (`Alarm_5`).
///
/// Register map: `boss.pattern_index` = `wep` (0 golden revolver, 1
/// golden shotgun, 2 golden bazooka, 3 minigun), `brain.ammo` = `ammo`,
/// `brain.gunangle` (radians) = `gunangle`, `boss.aux` = `minigun_side`,
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
    let los = dist < 512.0
        && !crate::walls::segment_hits_wall_legacy(epos, player_pos, &wall_centers);
    let can_shoot = boss.phase_timer.finished();

    // GML `Alarm_2` (fire tick).
    if boss.special_timer.just_finished() {
        // `(ammo--) <= 0`: empty click knocks back and walks off.
        if brain.ammo == 0 {
            let rand_dir = glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU));
            gml_motion_add_clamp(&mut vel.0, rand_dir, 1.0, 4.0, dt);
            gml_motion_add_clamp(&mut vel.0, to_player.normalize_or_zero(), 4.0, 4.0, dt);
            // `scrWalk(direction, 1, 10, 30)`.
            boss.target = vel.0.normalize_or_zero();
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
                                commands, owner, epos + sdir * 20.0, sdir, Team::Enemy,
                                480.0, 3, 3.0, 4.5, 210.0, EnemyKind::YvBoss,
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
                            commands, owner, epos + sdir * 20.0, sdir, Team::Enemy,
                            480.0, 3, 3.0, 4.5, 210.0, EnemyKind::YvBoss,
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
                        fire_projectile(
                            commands, owner, epos + sdir * 20.0, sdir, Team::Enemy,
                            rng.random_range(360.0..=540.0), 1, 2.0, 4.0, 120.0,
                            EnemyKind::YvBoss,
                        );
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
                        fire_projectile(
                            commands, owner, epos + sdir * 20.0, sdir, Team::Enemy,
                            90.0, 20, 4.0, 6.0, 300.0, EnemyKind::YvBoss,
                        );
                    }
                    fired = true;
                    trauma.add(0.4);
                }
                _ => {
                    let jitter = rng.random_range(-5.0..=5.0_f32).to_radians();
                    let sdir = glam::Vec2::from_angle(brain.gunangle + jitter);
                    fire_projectile(
                        commands, owner, epos + sdir * 20.0, sdir, Team::Enemy,
                        480.0, 3, 3.0, 4.5, 210.0, EnemyKind::YvBoss,
                    );
                    brain.gunangle +=
                        boss.aux * rng.random_range(0.8..=1.0_f32).to_radians();
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
            // Cooldown walk: face the target when visible, else move dir.
            let face = if los {
                (player_pos - epos).normalize_or_zero()
            } else {
                vel.0.normalize_or_zero()
            };
            if face.length_squared() > 0.0001 {
                brain.gunangle = face.y.atan2(face.x);
            }
            boss.target = glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU));
            brain.walk = rng.random_range(10.0..=30.0);
            // `alarm[1] = walk + irandom(10) + 10`, halved on cooldown.
            let wait = (brain.walk + rng.random_range(0.0..=10.0) + 10.0) * 0.5;
            boss.attack_timer = GTimer::from_seconds(wait / 30.0, TimerMode::Once);
        } else {
            brain.gunangle = aim;
            // Set when the brain picks an attack below (`alarm[4] != -1`).
            let mut picked = false;
            if los {
                let player_slow = player_velocity.length() < 30.0;
                if dist <= 64.0
                    && boss.pattern_index != SHOTGUN
                    && (player_slow || rng.random::<f32>() < 0.5)
                {
                    boss.pattern_index = SHOTGUN;
                    brain.ammo = 1;
                    picked = true;
                    boss.phase_timer =
                        GTimer::from_seconds(rng.random_range(30.0..=40.0) / 30.0, TimerMode::Once);
                    boss.special_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
                } else if (dist <= 110.0 || rng.random::<f32>() < 1.0 / 3.0)
                    && boss.pattern_index != REVOLVER
                {
                    boss.pattern_index = REVOLVER;
                    brain.ammo = 5;
                    picked = true;
                    boss.phase_timer =
                        GTimer::from_seconds((25.0 + rng.random_range(0.0..=15.0)) / 30.0, TimerMode::Once);
                    boss.special_timer = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                } else {
                    boss.pattern_index = MINIGUN;
                    boss.aux = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
                    brain.ammo = 90;
                    picked = true;
                    brain.gunangle -= (45.0 * boss.aux + rng.random_range(-10.0..=10.0)).to_radians();
                    boss.phase_timer =
                        GTimer::from_seconds((180.0 + rng.random_range(0.0..=40.0)) / 30.0, TimerMode::Once);
                    boss.special_timer = GTimer::from_seconds(4.0 / 30.0, TimerMode::Once);
                }
            } else {
                // No line of sight: step in when wall-adjacent, else bazooka.
                let blocked_close = wall_centers.iter().any(|w| {
                    let t = ((w.x - epos.x) * to_player.x + (w.y - epos.y) * to_player.y)
                        / to_player.length_squared().max(1.0);
                    t > 0.0 && t < 1.0 && (*w - (epos + to_player * t)).length() < 64.0
                });
                if blocked_close {
                    let step = to_player.normalize_or_zero();
                    gml_motion_add_clamp(&mut vel.0, step, 4.0, 4.0, dt);
                    boss.target = step;
                    brain.walk = rng.random_range(10.0..=20.0);
                    brain.gunangle = step.y.atan2(step.x);
                    boss.attack_timer =
                        GTimer::from_seconds(brain.walk / 30.0, TimerMode::Once);
                } else {
                    boss.pattern_index = BAZOOKA;
                    brain.ammo = 1;
                    picked = true;
                }
                boss.phase_timer = GTimer::from_seconds(60.0 / 30.0, TimerMode::Once);
                boss.special_timer = GTimer::from_seconds(7.0 / 30.0, TimerMode::Once);
            }
            if picked {
                boss.attack_timer =
                    GTimer::from_seconds(rng.random_range(5.0..=15.0) / 30.0, TimerMode::Once);
                brain.walk = 0.0;
            }
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

// ---------------------------------------------------------------------------
// Tests: headless parity for routing, volley counts, orbit, difficulty.
// ---------------------------------------------------------------------------

