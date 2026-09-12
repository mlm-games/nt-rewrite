//! Player movement. Ported from nt's `game/player.rs::player_move`
//! with positions as [`Pos`] instead of `Transform.translation`
//! (all laws byte-identical, including the no-hard-clamp rule that
//! preserves knockback/lunge impulses above max speed).

use bevy_ecs::prelude::*;
use glam::Vec2;
use rand::RngExt;
use repame_sim::SimTime;

use crate::audio::AudioCue;
use crate::comps_a::{
    AbilityHazard, AimDir, DamageSource, FloorMask, GameCleanup, Health, HitId,
    Inventory, LevelCleanup, Player, Projectile, RaceState, Team, Velocity,
};
use crate::comps_b::{
    Ally, Dash, Enemy, EnemyBrain, FrogCharge, HazardCloud, HorrorCharge, Portal, PortalState,
    PortalSucking, Prop, Shield, Telekinesis, WeaponVisual, WeaponVisualOwner,
};
use crate::data::{AbilityKind, EnemyKind, HazardKind, RaceId, WeaponId};
use crate::weapon_runtime::weapon_runtime_def;
use crate::input::NtInput;
use crate::msg::Queue;
use crate::spatial::{Pos, clamp_to_arena, resolve_mask_circle, resolve_prop_collision};
use crate::time::{GTimer, TimerMode};

use super::spatial::PLAYER_RADIUS;

/// Walk accel + friction + dash integration, then prop / mask / arena
/// push-outs. Single player entity (returns silently without one).
pub fn player_move(
    time: Res<SimTime>,
    mut commands: Commands,
    input: Res<NtInput>,
    mask: Res<FloorMask>,
    mut q: Query<
        (
            Entity,
            &Player,
            &mut Velocity,
            &mut Pos,
            Option<&mut Dash>,
            Option<&PortalSucking>,
        ),
        (With<Player>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
) {
    let Ok((entity, player, mut vel, mut pos, dash, sucking)) = q.single_mut() else {
        return;
    };
    if sucking.is_some() {
        vel.0 = glam::Vec2::ZERO;
        return;
    }

    let dt = time.delta_secs;

    if let Some(mut dash) = dash {
        dash.timer.tick(time.delta_secs);
        vel.0 = dash.dir * 950.0;
        pos.0 += vel.0 * dt;

        if dash.timer.just_finished() {
            commands.entity(entity).remove::<Dash>();
        }
    } else {
        // GML Player/Step_0 only clamps while adding walk accel: external
        // impulses above maxspeed (melee lunge, knockback) are preserved
        // and decay via friction instead of being hard-clamped away.
        let max_speed = player.speed * player.speed_mult;
        if input.move_axis != glam::Vec2::ZERO && vel.0.length() < max_speed {
            let dir = input.move_axis.normalize_or_zero();
            vel.0 += dir * player.accel * dt;
            if vel.0.length() > max_speed {
                vel.0 = vel.0.normalize_or_zero() * max_speed;
            }
        }

        crate::comps_a::apply_gml_friction(&mut vel.0, player.friction, dt);
        pos.0 += vel.0 * dt;
    }

    resolve_prop_collision(
        &mut pos.0,
        PLAYER_RADIUS,
        props.iter().map(|(_, prop, p)| (p.0, prop.size)),
    );
    resolve_mask_circle(&mask, &mut pos.0, PLAYER_RADIUS);
    clamp_to_arena(&mut pos.0, PLAYER_RADIUS);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn test_world() -> bevy_ecs::prelude::World {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        world.insert_resource(NtInput::default());
        world.insert_resource(FloorMask {
            cells: HashSet::from([(0, 0)]),
            cols: 64,
            rows: 64,
        });
        world
    }

    #[test]
    fn eyes_monsterstyle_shoves_nearby_enemies_released() {
        use crate::comps_b::Enemy as EnemyComp;
        use crate::data::{RaceId, UltraMutationId};
        let mut world = test_world();
        world.init_resource::<Queue<AudioCue>>();
        let mut player = Player::default();
        player.ability = crate::data::AbilityKind::Telekinesis;
        player.ultra = Some(UltraMutationId::EyesMonsterStyle);
        world.spawn((
            player,
            Pos(glam::Vec2::ZERO),
            Velocity(glam::Vec2::ZERO),
            Health {
                hp: 8,
                max: 8,
                invuln: GTimer::disarmed(),
            },
            AimDir(glam::Vec2::X),
        ));
        let foe = world
            .spawn((
                EnemyComp {
                    kind: crate::data::EnemyKind::Bandit,
                    score: 0,
                    touch_damage: 0,
                    rad_drop: 0,
                    drop_chance: 0,
                    weapon_chance: 0,
                },
                Pos(glam::Vec2::new(100.0, 0.0)),
                Velocity(glam::Vec2::ZERO),
            ))
            .id();
        // Spec released (default input): MonsterStyle shove applies.
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(tick_hold_abilities);
        for _ in 0..10 {
            schedule.run(&mut world);
        }
        let p = world.get::<Pos>(foe).unwrap().0;
        assert!(p.x > 100.0, "shoved outward, got {p:?}");
        let _ = RaceId::Eyes;
    }

    #[test]
    fn walk_accelerates_and_caps_at_max_speed() {
        let mut world = test_world();
        let mut player = Player::default();
        player.speed = 100.0;
        player.accel = 1000.0;
        player.friction = 0.0;
        let e = world
            .spawn((player, Velocity(glam::Vec2::ZERO), Pos(glam::Vec2::ZERO)))
            .id();
        world.resource_mut::<NtInput>().move_axis = glam::Vec2::new(1.0, 0.0);
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(player_move);
        for _ in 0..60 {
            schedule.run(&mut world);
        }
        let v = world.get::<Velocity>(e).unwrap().0;
        let p = world.get::<Pos>(e).unwrap().0;
        assert!((v.length() - 100.0).abs() < 1.0, "capped, got {v:?}");
        assert!(p.x > 0.0, "moved right, got {p:?}");
    }

    #[test]
    fn knockback_above_max_is_preserved() {
        let mut world = test_world();
        let mut player = Player::default();
        player.speed = 100.0;
        player.accel = 1000.0;
        player.friction = 0.0;
        let e = world
            .spawn((
                player,
                Velocity(glam::Vec2::new(400.0, 0.0)),
                Pos(glam::Vec2::ZERO),
            ))
            .id();
        // No input: friction is zero, impulse must survive untouched.
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(player_move);
        schedule.run(&mut world);
        let v = world.get::<Velocity>(e).unwrap().0;
        assert!((v.x - 400.0).abs() < 1e-3, "preserved, got {v:?}");
    }

    #[test]
    fn sucking_zeroes_velocity() {
        let mut world = test_world();
        let portal = world.spawn_empty().id();
        let e = world
            .spawn((
                Player::default(),
                Velocity(glam::Vec2::new(50.0, 0.0)),
                Pos(glam::Vec2::ZERO),
                PortalSucking {
                    portal,
                    timer: crate::time::GTimer::from_seconds(1.0, crate::time::TimerMode::Once),
                    start_pos: glam::Vec2::ZERO,
                    target_pos: glam::Vec2::ZERO,
                },
            ))
            .id();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(player_move);
        schedule.run(&mut world);
        assert_eq!(world.get::<Velocity>(e).unwrap().0, glam::Vec2::ZERO);
    }

    #[test]
    fn loop_sfx_edges_start_and_stop() {
        use crate::comps_b::{Portal, PortalPhase};
        use crate::data::SkinLetter;
        let mut world = test_world();
        world.init_resource::<Queue<AudioCue>>();
        let mut player = Player::default();
        player.ability = crate::data::AbilityKind::Telekinesis;
        let e = world
            .spawn((
                player,
                Pos(glam::Vec2::ZERO),
                Velocity(glam::Vec2::ZERO),
                Health {
                    hp: 8,
                    max: 8,
                    invuln: GTimer::disarmed(),
                },
                RaceState {
                    race: RaceId::Eyes,
                    skin: SkinLetter::A,
                },
            ))
            .id();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(tick_loop_sfx);
        // Eyes hold starts the loop (GML `scrPowers` Eyes branch).
        world.resource_mut::<NtInput>().spec_held = true;
        schedule.run(&mut world);
        assert!(world.get::<Player>(e).unwrap().eyes_loop_on);
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert!(cues.iter().any(|c| c.name == "sndEyesLoop"));
        // Release stops it; holding again restarts (edge-triggered).
        world.resource_mut::<NtInput>().spec_held = false;
        schedule.run(&mut world);
        assert!(!world.get::<Player>(e).unwrap().eyes_loop_on);
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert!(cues.iter().any(|c| c.name == "stop_sndEyesLoop"));
        schedule.run(&mut world);
        assert!(world.resource::<Queue<AudioCue>>().is_empty());
        // Chicken headless debt starts its loop; payback stops + regen.
        world.get_mut::<Player>(e).unwrap().headloses = 1;
        {
            let mut rq = world.query::<&mut RaceState>();
            rq.iter_mut(&mut world).next().unwrap().race = RaceId::Chicken;
        }
        schedule.run(&mut world);
        assert!(world.get::<Player>(e).unwrap().chicken_headless_loop_on);
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert!(
            cues
                .iter()
                .any(|c| c.name == "sndChickenHeadlessLoop")
        );
        world.get_mut::<Player>(e).unwrap().headloses = 0;
        schedule.run(&mut world);
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert!(
            cues
                .iter()
                .any(|c| c.name == "stop_sndChickenHeadlessLoop")
        );
        assert!(cues.iter().any(|c| c.name == "sndChickenRegenHead"));
        // Portal loop runs while Idle, stops on Disappear.
        let portal = world
            .spawn((
                Portal,
                PortalState {
                    kind: 1,
                    phase: PortalPhase::Spawn,
                    endgame: 100.0,
                    close: false,
                    anim: GTimer::from_seconds(0.25, TimerMode::Once),
                    loop_on: false,
                },
                Pos(glam::Vec2::ZERO),
            ))
            .id();
        schedule.run(&mut world);
        assert!(world.resource::<Queue<AudioCue>>().is_empty());
        world.get_mut::<PortalState>(portal).unwrap().phase = PortalPhase::Idle;
        schedule.run(&mut world);
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert!(cues.iter().any(|c| c.name == "sndPortalLoop"));
        world.get_mut::<PortalState>(portal).unwrap().phase = PortalPhase::Disappear;
        schedule.run(&mut world);
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert!(cues.iter().any(|c| c.name == "stop_sndPortalLoop"));
    }
}

// ---------------------------------------------------------------------------
// SIM-side player systems: headless port of the non-render half of
// `nt-recreated-bevy/src/game/player.rs`.
// PORTED (pure sim): `player_aim` (stick path verbatim + shell-fed
// mouse path), `weapon_switch` (timers untouched, bevy parity),
// `tick_player_timers`, `ally_ai`, `tick_hold_abilities`,
// `held_weapon_angle`, `steroids_secondary_slot`,
// `ensure_weapon_visual` / `tick_weapon_visuals` (entity + wkick/wep
// state; art/pose resolve renderer-side).
// RENDERER-OWNED (resolved in `render.rs` from sim state, no systems):
// `face_aim` (flip from AimDir/Velocity), `blink_player` (alpha from
// invuln).
// Conventions: `Timer` -> `GTimer`, `Transform` -> [`Pos`], audio spawns ->
// [`AudioCue`]s in a [`Queue`], `Time<Fixed>` -> [`SimTime`],
// thread rng -> `rand::rng()`.
// ---------------------------------------------------------------------------

/// Stick aim -> [`AimDir`]. Bevy `player_aim` stick path verbatim: any
/// nonzero deflection steers (no dead zone — bevy normalizes directly).
/// The mouse path lives in the shell: every frame
/// [`App::feed_input`](crate::App::feed_input) steers `aim_axis` at the
/// latest viewport hover (world coords from the live fit, same role as
/// bevy's `viewport_to_world_2d` cursor ray), so this system steers
/// `AimDir` at the cursor without knowing about pointers. With no
/// deflection the last aim is kept (bevy only overwrote aim when a
/// cursor ray hit). `Sprite` flips skipped.
pub fn player_aim(input: Res<NtInput>, mut player_q: Query<&mut AimDir, With<Player>>) {
    let Ok(mut aim) = player_q.single_mut() else {
        return;
    };
    if input.aim_axis != Vec2::ZERO {
        aim.0 = input.aim_axis.normalize_or_zero();
    }
}

/// Equip/cycle weapons. Slot-select + cycle logic is bevy-verbatim
/// (skips `NONE` slots, wraps with `weapon_slots`); audio is a `pickup`
/// cue at 0.25/0.05 (bevy `play_sfx_varied` parity). Fire timers are
/// untouched (bevy keeps the old cooldown running across a switch).
pub fn weapon_switch(
    mut input: ResMut<NtInput>,
    mut q: Query<&mut Inventory, With<Player>>,
    mut cues: ResMut<Queue<AudioCue>>,
) {
    let Ok(mut inv) = q.single_mut() else {
        return;
    };
    let mut switched = false;

    if let Some(slot) = input.take_weapon_slot()
        && slot < inv.weapon_slots
        && inv.weapons[slot] != WeaponId::NONE
        && slot != inv.current
    {
        inv.current = slot;
        switched = true;
    }

    let cycle = input.take_cycle_weapon();
    if cycle != 0 && inv.weapon_slots > 1 {
        let direction = if cycle > 0 { 1 } else { inv.weapon_slots - 1 };

        for step in 1..=inv.weapon_slots {
            let slot = (inv.current + step * direction) % inv.weapon_slots;
            if inv.weapons[slot] != WeaponId::NONE {
                switched |= slot != inv.current;
                inv.current = slot;
                break;
            }
        }
    }

    if switched {
        // GML `Step_0:30`: swap pops `swapanim`.
        inv.swapanim = 1.0;
        cues.push(AudioCue {
            name: "sndAmmoPickup",
            volume: 0.25,
            variance: 0.05,
        });
    }
}

/// Held-gun entities, sim half (bevy `ensure_weapon_visual` /
/// `tick_weapon_visuals` minus `Sprite`/`Transform`: art, anchor, pose
/// and `flip_y` resolve renderer-side from `WeaponVisual`, so the sim
/// only owns `owner` / `wep_id` / `wep_angle` / `wkick` / `slot`).
pub fn ensure_weapon_visual(
    mut commands: Commands,
    player_q: Query<
        (Entity, &Inventory, &AimDir, &RaceState),
        (With<Player>, Without<WeaponVisualOwner>),
    >,
) {
    let Ok((player_e, inv, _aim, race_state)) = player_q.single() else {
        return;
    };
    let id = inv.weapons[inv.current];
    if id == WeaponId::NONE {
        return;
    }
    let dual = race_state.race == RaceId::Steroids
        && inv.weapon_slots > 1
        && inv.weapons[(inv.current + 1) % inv.weapon_slots] != WeaponId::NONE;
    commands.entity(player_e).insert(WeaponVisualOwner);
    commands.spawn((
        GameCleanup,
        WeaponVisual {
            owner: player_e,
            wkick: 0.0,
            wep_id: id,
            wep_angle: 0.0,
            slot: 0,
        },
    ));
    if dual {
        let second = inv.weapons[(inv.current + 1) % inv.weapon_slots];
        commands.spawn((
            GameCleanup,
            WeaponVisual {
                owner: player_e,
                wkick: 0.0,
                wep_id: second,
                wep_angle: 0.0,
                slot: 1,
            },
        ));
    }
}

pub fn tick_weapon_visuals(
    time: Res<SimTime>,
    mut commands: Commands,
    player_q: Query<
        (
            Entity,
            &AimDir,
            &Inventory,
            &RaceState,
            Option<&PortalSucking>,
        ),
        With<Player>,
    >,
    mut vis_q: Query<(Entity, &mut WeaponVisual), Without<Player>>,
) {
    let dt = time.delta_secs;
    let Ok((player_e, _aim, inv, race_state, sucking)) = player_q.single() else {
        for (e, _) in &vis_q {
            commands.entity(e).despawn();
        }
        return;
    };
    if sucking.is_some() {
        for (e, _) in &vis_q {
            commands.entity(e).despawn();
        }
        commands.entity(player_e).remove::<WeaponVisualOwner>();
        return;
    }
    let dual = race_state.race == RaceId::Steroids && inv.weapon_slots > 1;

    let mut seen = [false; 2];
    for (_e, wv) in vis_q.iter() {
        if wv.owner == player_e && (wv.slot as usize) < 2 {
            seen[wv.slot as usize] = true;
        }
    }
    if dual {
        let second = inv.weapons[(inv.current + 1) % inv.weapon_slots];
        if second == WeaponId::NONE && seen[1] {
            for (e, wv) in vis_q.iter() {
                if wv.owner == player_e && wv.slot == 1 {
                    commands.entity(e).despawn();
                }
            }
            seen[1] = false;
        } else if second != WeaponId::NONE && !seen[1] {
            commands.spawn((
                GameCleanup,
                WeaponVisual {
                    owner: player_e,
                    wkick: 0.0,
                    wep_id: second,
                    wep_angle: 0.0,
                    slot: 1,
                },
            ));
            seen[1] = true;
        }
    } else if seen[1] {
        for (e, wv) in vis_q.iter() {
            if wv.owner == player_e && wv.slot == 1 {
                commands.entity(e).despawn();
            }
        }
    }
    let _ = seen;
    let step = dt * crate::SIM_HZ as f32;
    for (_e, mut wv) in &mut vis_q {
        if wv.owner != player_e {
            continue;
        }
        let slot_idx = if dual {
            (inv.current + wv.slot as usize) % inv.weapon_slots
        } else {
            if wv.slot != 0 {
                continue;
            }
            inv.current
        };
        let id = inv.weapons[slot_idx];
        if wv.wkick > 0.0 {
            wv.wkick = (wv.wkick - step).max(0.0);
        } else if wv.wkick < 0.0 {
            wv.wkick = (wv.wkick + step).min(0.0);
        }
        if wv.wep_id != id {
            wv.wep_id = id;
            wv.wep_angle = if weapon_runtime_def(id).melee.is_some() {
                if rand::rng().random_bool(0.5) {
                    120.0
                } else {
                    -120.0
                }
            } else {
                0.0
            };
        }
    }
}

/// Tick player-scoped cooldowns. Bevy-verbatim: ability cooldown, hurt
/// invuln, shield and telekinesis timers. Fire-rate timers tick in
/// `player_fire` (already ported) and are not touched here.
pub fn tick_player_timers(
    time: Res<SimTime>,
    mut q: Query<(
        &mut Player,
        &mut Health,
        &mut Inventory,
        Option<&RaceState>,
        Option<&mut Shield>,
        Option<&mut Telekinesis>,
    )>,
) {
    let dt = time.delta_secs;
    for (mut player, mut health, mut inv, race, shield, telek) in &mut q {
        player.ability_cooldown.tick(dt);
        health.invuln.tick(dt);
        // GML `Step_0:19` + `:605-609`: `swapanim` and
        // `trigger_fingers_shine` decay.
        inv.swapanim = (inv.swapanim - dt * 30.0).max(0.0);
        inv.shine = (inv.shine - dt * 30.0 * 0.4).max(0.0);
        // GML Gun Warrant: 7 s of free ammo after exiting a portal.
        if player.warrant > 0.0 {
            player.warrant = (player.warrant - dt * 30.0).max(0.0);
            if player.warrant <= 0.0 {
                let base = race.is_some_and(|r| r.race == crate::data::RaceId::Robot)
                    || matches!(
                        player.ultra,
                        Some(
                            crate::data::UltraMutationId::RobotRefinedTaste
                                | crate::data::UltraMutationId::RobotRegurgitate
                        )
                    );
                player.free_ammo = base;
            }
        }
        if let Some(mut s) = shield {
            s.timer.tick(dt);
        }
        if let Some(mut t) = telek {
            t.timer.tick(dt);
        }
    }
}

/// Rebel ally AI, headless: seek the nearest enemy at 140 px/s, fire an
/// ally bolt every `shoot` tick, despawn when `life` ends. Ally spawns
/// live in `player_fire` (`SpawnAlly`) and `crown` (Love) and are reused
/// here unchanged. Art stripped (sim-only `Pos` projectile); the bolt
/// bolt. Reuses the ally representation from `player_fire` (`SpawnAlly`)
/// and `crown` (Love): same comps, same 140 px/s + bolt stats.
#[allow(clippy::type_complexity)]
pub fn ally_ai(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mut allies: Query<(Entity, &mut Ally, &mut Pos, &mut Velocity), (With<Ally>, Without<Enemy>)>,
    enemies: Query<&Pos, (With<Enemy>, Without<Ally>)>,
) {
    let dt = time.delta_secs;
    for (e, mut ally, mut apos, mut vel) in &mut allies {
        ally.life.tick(dt);
        ally.shoot.tick(dt);
        if ally.life.just_finished() {
            commands.entity(e).despawn();
            continue;
        }
        let pos = apos.0;
        let mut best = None::<(f32, Vec2)>;
        for epos in &enemies {
            let p = epos.0;
            let d = p.distance_squared(pos);
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, p));
            }
        }
        if let Some((_, target)) = best {
            let dir = (target - pos).normalize_or_zero();
            vel.0 = dir * 140.0;
            apos.0 += vel.0 * dt;
            if ally.shoot.just_finished() {
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Projectile {
                        damage: 2,
                        life: GTimer::from_seconds(0.7, TimerMode::Once),
                        radius: 4.0,
                        knockback: 20.0,
                        explosive: false,
                        source: Some(DamageSource {
                            owner: e,
                            team: Team::Player,
                            hit_id: HitId::Other(1),
                            enemy_kind: None,
                        }),
                    },
                    Team::Player,
                    Velocity(dir * 380.0),
                    Pos(pos),
                ));
                cues.push(AudioCue {
                    name: "bolt",
                    volume: 0.7,
                    variance: 0.05,
                });
            }
        }
    }
}

/// Hold abilities (GML hold_spec RMB): Eyes telekinesis push/pull, Horror
/// rad-drain beam, Frog charge/release. Headless: catalog sprites
/// stripped, sim spawns + cues kept, laws verbatim. `RaceState` was
/// spawns + cues kept, laws verbatim. `RaceState` was
/// unused in bevy (`let _ = race`) and is dropped from the query.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn tick_hold_abilities(
    time: Res<SimTime>,
    mut commands: Commands,
    input: Res<NtInput>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut player_q: Query<
        (
            Entity,
            &Pos,
            &mut Player,
            &mut Health,
            &mut Velocity,
            &AimDir,
        ),
        (With<Player>, Without<Enemy>, Without<Projectile>),
    >,
    mut enemies: Query<
        (&mut Pos, &mut Velocity),
        (With<Enemy>, Without<Player>, Without<Projectile>),
    >,
    mut projectiles: Query<
        (&Pos, &mut Velocity, &Team, Option<&Projectile>),
        (With<Projectile>, Without<Player>, Without<Enemy>),
    >,
    mut horror_q: Query<&mut HorrorCharge>,
    mut frog_q: Query<&mut FrogCharge>,
    mut telek_q: Query<&mut Telekinesis>,
) {
    let Ok((player_e, ppos, mut player, mut health, mut pvel, aim)) = player_q.single_mut() else {
        return;
    };
    let held = input.spec_held;
    let pos = ppos.0;
    let dt = time.delta_secs;

    if player.ability == AbilityKind::Telekinesis && held {
        let strength = if player.throne_butt { 60.0 } else { 30.0 };
        if let Ok(mut t) = telek_q.single_mut() {
            t.timer = GTimer::from_seconds(0.25, TimerMode::Once);
        } else {
            commands.entity(player_e).insert(Telekinesis {
                timer: GTimer::from_seconds(0.25, TimerMode::Once),
            });
        }
        for (epos, mut evel) in &mut enemies {
            let epos_v = epos.0;
            if (epos_v.x - pos.x).abs() > 160.0 || (epos_v.y - pos.y).abs() > 120.0 {
                continue;
            }
            let to_player = (pos - epos_v).normalize_or_zero();
            evel.0 += to_player * strength * dt;
        }
        for (ppos_proj, mut v, team, _) in &mut projectiles {
            if *team != Team::Enemy {
                continue;
            }
            let ppos = ppos_proj.0;
            if (ppos.x - pos.x).abs() > 160.0 || (ppos.y - pos.y).abs() > 120.0 {
                continue;
            }
            let out = (ppos - pos).normalize_or_zero();
            v.0 += out * strength * dt;
        }
        // GML `ProjectileStyle`: held telekinesis also reels the
        // player's own shots into an 8 px orbit (non-laser/lightning).
        if player.ultra == Some(crate::data::UltraMutationId::EyesProjectileStyle) {
            for (ppos_proj, mut v, team, proj) in &mut projectiles {
                if *team != Team::Player {
                    continue;
                }
                if proj.is_none_or(|p| p.source.is_none_or(|s| s.owner != player_e)) {
                    continue;
                }
                let target = pos + v.0.normalize_or_zero() * 8.0;
                let pull = (target - ppos_proj.0).normalize_or_zero();
                let cur = v.0;
                v.0 = cur + pull * 60.0 * dt;
                if cur.length() < 480.0 {
                    v.0 += cur.normalize_or_zero() * 30.0 * dt;
                }
            }
        }
    } else if player.ability == AbilityKind::Telekinesis
        && player.ultra == Some(crate::data::UltraMutationId::EyesMonsterStyle)
    {
        // GML `scrPowers` Eyes release: every non-held step shoves
        // enemies within 130 px one px outward.
        for (mut epos, _) in &mut enemies {
            let cur = epos.0;
            let d = cur.distance(pos);
            if d <= 130.0 && d > 0.001 {
                epos.0 = cur + (cur - pos).normalize_or_zero() * 30.0 * dt;
            }
        }
    }

    if player.ability == AbilityKind::HorrorBeam {
        if held {
            let mut time_val = if let Ok(c) = horror_q.get(player_e) {
                c.time
            } else {
                commands.entity(player_e).insert(HorrorCharge { time: 0.0 });
                0.0
            };
            let cost = (time_val + 1.0).floor() as u32;
            if player.rads >= cost && cost > 0 {
                player.rads -= cost;
                time_val += 0.03 * dt * 30.0;
                if let Ok(mut c) = horror_q.get_mut(player_e) {
                    c.time = time_val;
                }
                let n = (time_val + 1.0).round() as usize;
                let dir = aim.0.normalize_or_zero();
                for _ in 0..n.min(12) {
                    let jitter = Vec2::new(
                        rand::rng().random_range(-8.0..8.0),
                        rand::rng().random_range(-8.0..8.0),
                    );
                    let bdir = (dir * 24.0 + jitter).normalize_or_zero();
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Player,
                        Projectile {
                            damage: 3,
                            life: GTimer::from_seconds(1.2, TimerMode::Once),
                            radius: 5.0,
                            knockback: 60.0,
                            explosive: false,
                            source: Some(DamageSource::player_weapon(player_e, WeaponId::NONE)),
                        },
                        Velocity(bdir * 360.0),
                        Pos(pos + dir * 18.0),
                    ));
                }
                if player.throne_butt && rand::rng().random_range(0..30) == 0 {
                    health.hp = (health.hp + 1).min(health.max);
                }
            }
        } else {
            if let Ok(mut c) = horror_q.get_mut(player_e) {
                c.time = 0.0;
            }
            if horror_q.get(player_e).is_ok() {
                commands.entity(player_e).remove::<HorrorCharge>();
            }
        }
    }

    if player.ability == AbilityKind::ToxicPuke {
        if held {
            let mut gas = if let Ok(c) = frog_q.get(player_e) {
                c.gas
            } else {
                commands.entity(player_e).insert(FrogCharge { gas: 0.0 });
                0.0
            };
            if gas < 30.0 {
                gas += dt * 30.0;
                if let Ok(mut c) = frog_q.get_mut(player_e) {
                    c.gas = gas.min(30.0);
                }
            }
            pvel.0 = Vec2::ZERO;
        } else if frog_q.get(player_e).is_ok() {
            let gas = frog_q.get(player_e).map(|c| c.gas).unwrap_or(0.0);
            commands.entity(player_e).remove::<FrogCharge>();
            let n = gas.round() as usize;
            if n > 0 {
                for _ in 0..n.min(30) {
                    let off = Vec2::new(
                        rand::rng().random_range(-10.0..10.0),
                        rand::rng().random_range(-10.0..10.0),
                    );
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        AbilityHazard,
                        HazardCloud {
                            kind: HazardKind::Toxic,
                            radius: 26.0,
                            damage: 3,
                            timer: GTimer::from_seconds(4.0, TimerMode::Once),
                            tick: GTimer::from_seconds(0.3, TimerMode::Repeating),
                        },
                        Pos(pos + off),
                    ));
                }
                cues.push(AudioCue {
                    name: "sndExplosionL",
                    volume: 0.9,
                    variance: 0.04,
                });
            }
        }
    } else if frog_q.get(player_e).is_ok() {
        commands.entity(player_e).remove::<FrogCharge>();
    }
}

/// Looping one-shots in headless cue form (GML `snd_play_loop` /
/// `snd_stop`: `scrPowers` Eyes/Horror holds, `Player/Step_0` frog +
/// chicken-headless, `Portal/Other_7`, `Salamander/Alarm_2`). Backend
/// wiring is out of scope — start cues carry the loop sound name, stop
/// cues the same name under `stop_`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoopSfx {
    Eyes,
    Horror,
    Frog,
    ChickenHeadless,
    Portal,
    Salamander,
}

impl LoopSfx {
    pub fn start_cue(self, throne_butt: bool) -> &'static str {
        match self {
            // GML `scrPowers`: `_tb ? sndEyesLoopUpg : sndEyesLoop`.
            LoopSfx::Eyes => {
                if throne_butt {
                    "sndEyesLoopUpg"
                } else {
                    "sndEyesLoop"
                }
            }
            // GML `scrPowers:296`: `_tb ? sndHorrorLoopTB : sndHorrorLoop`.
            LoopSfx::Horror => {
                if throne_butt {
                    "sndHorrorLoopTB"
                } else {
                    "sndHorrorLoop"
                }
            }
            // GML `Player/Step_0:653-657`: frog loop, Butt variant.
            LoopSfx::Frog => {
                if throne_butt {
                    "sndFrogLoopButt"
                } else {
                    "sndFrogLoop"
                }
            }
            // GML `Player/Step_0:235`.
            LoopSfx::ChickenHeadless => "sndChickenHeadlessLoop",
            // GML `Portal/Other_7`.
            LoopSfx::Portal => "sndPortalLoop",
            // GML `Salamander/Alarm_2` (`myloop`).
            LoopSfx::Salamander => "sndSalamanderFireLoop",
        }
    }

    pub fn stop_cue(self, throne_butt: bool) -> &'static str {
        match self {
            LoopSfx::Eyes => {
                if throne_butt {
                    "stop_sndEyesLoopUpg"
                } else {
                    "stop_sndEyesLoop"
                }
            }
            LoopSfx::Horror => {
                if throne_butt {
                    "stop_sndHorrorLoopTB"
                } else {
                    "stop_sndHorrorLoop"
                }
            }
            LoopSfx::Frog => {
                if throne_butt {
                    "stop_sndFrogLoopButt"
                } else {
                    "stop_sndFrogLoop"
                }
            }
            LoopSfx::ChickenHeadless => "stop_sndChickenHeadlessLoop",
            LoopSfx::Portal => "stop_sndPortalLoop",
            LoopSfx::Salamander => "stop_sndSalamanderFireLoop",
        }
    }
}

fn push_loop_cue(cues: &mut Queue<AudioCue>, name: &'static str) {
    cues.push(AudioCue {
        name,
        volume: 0.6,
        variance: 0.0,
    });
}

/// Edge-triggered loop cues with state on the `Player`/`PortalState`
/// comps (plus per-entity vanish tracking for portal/salamander
/// stop-on-remove, mirroring GML `CleanUp_0`).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn tick_loop_sfx(
    input: Res<NtInput>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut player_q: Query<(&mut Player, &Health, &RaceState), With<Player>>,
    mut portals: Query<(Entity, &mut PortalState), With<Portal>>,
    salamanders: Query<(Entity, &Enemy, &EnemyBrain), (With<Enemy>, Without<Player>)>,
    mut looping_portals: Local<std::collections::HashSet<Entity>>,
    mut looping_salamanders: Local<std::collections::HashSet<Entity>>,
) {
    if let Ok((mut player, health, race_state)) = player_q.single_mut() {
        let held = input.spec_held;
        let tb = player.throne_butt;
        // GML Eyes hold (`scrPowers` Eyes branch).
        let eyes = race_state.race == RaceId::Eyes
            && player.ability == AbilityKind::Telekinesis
            && held;
        if eyes != player.eyes_loop_on {
            player.eyes_loop_on = eyes;
            push_loop_cue(
                &mut cues,
                if eyes {
                    LoopSfx::Eyes.start_cue(tb)
                } else {
                    LoopSfx::Eyes.stop_cue(tb)
                },
            );
        }
        // GML Horror hold (`scrPowers:296-298`).
        let horror = race_state.race == RaceId::Horror
            && player.ability == AbilityKind::HorrorBeam
            && held;
        if horror != player.horror_loop_on {
            player.horror_loop_on = horror;
            push_loop_cue(
                &mut cues,
                if horror {
                    LoopSfx::Horror.start_cue(tb)
                } else {
                    LoopSfx::Horror.stop_cue(tb)
                },
            );
        }
        // GML frog hold (`Player/Step_0:653-657`, Butt variant).
        let frog =
            race_state.race == RaceId::Frog && player.ability == AbilityKind::ToxicPuke && held;
        if frog != player.frog_loop_on {
            player.frog_loop_on = frog;
            push_loop_cue(
                &mut cues,
                if frog {
                    LoopSfx::Frog.start_cue(tb)
                } else {
                    LoopSfx::Frog.stop_cue(tb)
                },
            );
        }
        // GML chicken headless (`Player/Step_0:206-235`): loop while the
        // head debt is banked; stop + regen cue on heal (`Step_0:47-48`).
        let headless =
            race_state.race == RaceId::Chicken && health.hp > 0 && player.headloses > 0;
        if headless != player.chicken_headless_loop_on {
            player.chicken_headless_loop_on = headless;
            if headless {
                push_loop_cue(&mut cues, LoopSfx::ChickenHeadless.start_cue(false));
            } else {
                push_loop_cue(&mut cues, LoopSfx::ChickenHeadless.stop_cue(false));
                push_loop_cue(&mut cues, "sndChickenRegenHead");
            }
        }
    }

    // GML `Portal/Other_7` (start on anim) vs `Alarm_1`/`CleanUp_0`/
    // `Other_5` (stop on close/remove): loop while Idle.
    let mut live_portals = std::collections::HashSet::new();
    for (e, mut st) in &mut portals {
        live_portals.insert(e);
        let want = st.phase == crate::comps_b::PortalPhase::Idle;
        if want && !st.loop_on {
            st.loop_on = true;
            looping_portals.insert(e);
            push_loop_cue(&mut cues, LoopSfx::Portal.start_cue(false));
        } else if !want && st.loop_on {
            st.loop_on = false;
            looping_portals.remove(&e);
            push_loop_cue(&mut cues, LoopSfx::Portal.stop_cue(false));
        }
    }
    for gone in looping_portals.iter().filter(|e| !live_portals.contains(e)).copied().collect::<Vec<_>>() {
        looping_portals.remove(&gone);
        push_loop_cue(&mut cues, LoopSfx::Portal.stop_cue(false));
    }

    // GML `Salamander/Alarm_2` fire loop: on while a live Salamander is
    // mid-volley (`burst_left`), off when it ends or dies.
    let mut live_sal = std::collections::HashSet::new();
    for (e, enemy, brain) in &salamanders {
        if enemy.kind != EnemyKind::Salamander {
            continue;
        }
        live_sal.insert(e);
        if brain.burst_left > 0 && looping_salamanders.insert(e) {
            push_loop_cue(&mut cues, LoopSfx::Salamander.start_cue(false));
        } else if brain.burst_left == 0 && looping_salamanders.remove(&e) {
            push_loop_cue(&mut cues, LoopSfx::Salamander.stop_cue(false));
        }
    }
    for gone in looping_salamanders.iter().filter(|e| !live_sal.contains(e)).copied().collect::<Vec<_>>() {
        looping_salamanders.remove(&gone);
        push_loop_cue(&mut cues, LoopSfx::Salamander.stop_cue(false));
    }
}

/// GML Player/Draw_0 held-weapon angle: gunangle + wepangle * (1 - wkick/20).
pub fn held_weapon_angle(aim_angle: f32, wep_angle_deg: f32, wkick: f32) -> f32 {
    aim_angle + wep_angle_deg.to_radians() * (1.0 - wkick / 20.0)
}

/// Steroids second slot mirrors the other live slot. Verbatim bevy helper,
/// shared with the fire path (`player_fire` imports this instead of
/// keeping a second copy).
pub fn steroids_secondary_slot(current: usize, slots: usize) -> usize {
    if slots > 1 {
        (current + 1) % slots
    } else {
        current
    }
}

#[cfg(test)]
mod sim_parity_tests {
    use super::*;
    use crate::comps_a::FireCooldown;

    const DT: f32 = 1.0 / 30.0;

    fn sim_world() -> World {
        let mut world = World::new();
        world.insert_resource(SimTime {
            elapsed_secs: 0.0,
            delta_secs: DT,
        });
        world.insert_resource(NtInput::default());
        world.insert_resource(Queue::<AudioCue>::default());
        world
    }

    fn disarmed_cooldown() -> FireCooldown {
        FireCooldown {
            timer: GTimer::disarmed(),
            burst_left: 0,
            burst_timer: GTimer::disarmed(),
            timer_b: GTimer::disarmed(),
            burst_left_b: 0,
            burst_timer_b: GTimer::disarmed(),
        }
    }

    fn spawn_player(world: &mut World) -> Entity {
        world
            .spawn((
                Player::default(),
                AimDir(Vec2::X),
                Pos(Vec2::ZERO),
                Velocity(Vec2::ZERO),
                Health {
                    hp: 8,
                    max: 8,
                    invuln: GTimer::disarmed(),
                },
                Inventory {
                    weapons: [WeaponId::REVOLVER, WeaponId::SHOTGUN, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
                    weapon_slots: 2,
                    current: 0,
                    ammo: [0; 6],
                },
                disarmed_cooldown(),
            ))
            .id()
    }

    fn spawn_enemy(world: &mut World, at: Vec2) -> Entity {
        world
            .spawn((
                Enemy {
                    kind: crate::data::EnemyKind::Bandit,
                    score: 1,
                    touch_damage: 1,
                    rad_drop: 1,
                    drop_chance: 0,
                    weapon_chance: 0,
                },
                Team::Enemy,
                Pos(at),
                Velocity(Vec2::ZERO),
            ))
            .id()
    }

    fn run_aim(world: &mut World) {
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(player_aim);
        sched.run(world);
    }

    fn run_switch(world: &mut World) {
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(weapon_switch);
        sched.run(world);
    }

    #[test]
    fn aim_follows_stick_verbatim() {
        // Bevy has no dead zone: any nonzero deflection steers,
        // normalized; zero keeps the last aim.
        let mut world = sim_world();
        let e = spawn_player(&mut world);
        world.resource_mut::<NtInput>().aim_axis = Vec2::new(0.1, 0.1);
        run_aim(&mut world);
        let aim = world.get::<AimDir>(e).unwrap().0;
        let want = Vec2::new(0.1, 0.1).normalize();
        assert!((aim - want).length() < 1e-5, "got {aim:?}");
        // Zero deflection keeps the last aim.
        world.resource_mut::<NtInput>().aim_axis = Vec2::ZERO;
        run_aim(&mut world);
        let aim = world.get::<AimDir>(e).unwrap().0;
        assert!((aim - want).length() < 1e-5, "got {aim:?}");
        // Full deflection steers aim, normalized.
        world.resource_mut::<NtInput>().aim_axis = Vec2::new(0.0, 1.0);
        run_aim(&mut world);
        let aim = world.get::<AimDir>(e).unwrap().0;
        assert!((aim - Vec2::Y).length() < 1e-5, "got {aim:?}");
        // Diagonal follows direction.
        world.resource_mut::<NtInput>().aim_axis = Vec2::new(1.0, 1.0);
        run_aim(&mut world);
        let aim = world.get::<AimDir>(e).unwrap().0;
        let want = Vec2::new(1.0, 1.0).normalize();
        assert!((aim - want).length() < 1e-5, "got {aim:?}");
    }

    #[test]
    fn weapon_visual_spawns_and_tracks_swaps() {
        use crate::data::{RaceId, SkinLetter};

        let mut world = sim_world();
        let e = spawn_player(&mut world);
        world.entity_mut(e).insert(RaceState {
            race: RaceId::Fish,
            skin: SkinLetter::A,
        });
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems((ensure_weapon_visual, tick_weapon_visuals));
        sched.run(&mut world);
        // One gun visual for the current revolver.
        let guns: Vec<WeaponVisual> = world
            .query::<&WeaponVisual>()
            .iter(&world)
            .copied()
            .collect();
        assert_eq!(guns.len(), 1);
        assert_eq!(guns[0].wep_id, WeaponId::REVOLVER);
        assert_eq!(guns[0].owner, e);
        // Switch weapon: visual follows the new id next tick.
        world.get_mut::<Inventory>(e).unwrap().current = 1;
        sched.run(&mut world);
        let guns: Vec<WeaponVisual> = world
            .query::<&WeaponVisual>()
            .iter(&world)
            .copied()
            .collect();
        assert_eq!(guns.len(), 1);
        assert_eq!(guns[0].wep_id, WeaponId::SHOTGUN);
        // wkick decays toward zero.
        let gun_e = world
            .query_filtered::<Entity, With<WeaponVisual>>()
            .iter(&world)
            .next()
            .unwrap();
        world.get_mut::<WeaponVisual>(gun_e).unwrap().wkick = 6.0;
        sched.run(&mut world);
        let wk = world.get::<WeaponVisual>(gun_e).unwrap().wkick;
        assert!(wk < 6.0 && wk >= 0.0, "wkick decays, got {wk}");
    }

    #[test]
    fn weapon_switch_direct_swap_keeps_cooldown_and_cues() {
        // Bevy keeps the old cooldown running across a switch (no
        // disarm/reset of fire timers).
        let mut world = sim_world();
        let e = spawn_player(&mut world);
        {
            let mut cd = world.get_mut::<FireCooldown>(e).unwrap();
            cd.timer = GTimer::from_seconds(1.0, TimerMode::Once);
            cd.burst_left = 2;
        }
        world.resource_mut::<NtInput>().select_weapon(1);
        run_switch(&mut world);
        assert_eq!(world.get::<Inventory>(e).unwrap().current, 1);
        let cd = world.get::<FireCooldown>(e).unwrap();
        assert!(!cd.timer.is_finished(), "cooldown keeps running");
        assert_eq!(cd.burst_left, 2, "burst queue untouched");
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].name, "sndAmmoPickup");
        assert!((cues[0].volume - 0.25).abs() < 1e-6);
        // Selecting the live slot is a no-op: no cue, cooldown untouched.
        world.resource_mut::<NtInput>().select_weapon(1);
        run_switch(&mut world);
        assert_eq!(world.get::<Inventory>(e).unwrap().current, 1);
        assert!(world.resource::<Queue<AudioCue>>().is_empty());
    }

    #[test]
    fn weapon_switch_cycle_skips_empty_slots() {
        let mut world = sim_world();
        let e = spawn_player(&mut world);
        world.get_mut::<Inventory>(e).unwrap().weapons =
            [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::SHOTGUN];
        world.get_mut::<Inventory>(e).unwrap().weapon_slots = 3;
        world.resource_mut::<NtInput>().cycle_weapon(1);
        run_switch(&mut world);
        assert_eq!(world.get::<Inventory>(e).unwrap().current, 2);
        // Backward cycle wraps to slot 0.
        world.resource_mut::<NtInput>().cycle_weapon(-1);
        run_switch(&mut world);
        assert_eq!(world.get::<Inventory>(e).unwrap().current, 0);
        // Out-of-range select is ignored (pulse still consumed, no cue).
        world.resource_mut::<Queue<AudioCue>>().drain();
        world.resource_mut::<NtInput>().select_weapon(9);
        run_switch(&mut world);
        assert_eq!(world.get::<Inventory>(e).unwrap().current, 0);
        assert!(world.resource::<Queue<AudioCue>>().is_empty());
    }

    #[test]
    fn steroids_secondary_slot_index() {
        assert_eq!(steroids_secondary_slot(0, 2), 1);
        assert_eq!(steroids_secondary_slot(1, 2), 0);
        assert_eq!(steroids_secondary_slot(0, 1), 0);
        assert_eq!(steroids_secondary_slot(2, 3), 0);
        // Swap exchanges roles.
        let a_sec = steroids_secondary_slot(0, 2);
        let b_sec = steroids_secondary_slot(1, 2);
        assert_eq!(a_sec, 1);
        assert_eq!(b_sec, 0);
    }

    #[test]
    fn player_timers_tick_down() {
        let mut world = sim_world();
        let e = spawn_player(&mut world);
        world.get_mut::<Player>(e).unwrap().ability_cooldown =
            GTimer::from_seconds(1.0, TimerMode::Once);
        world.get_mut::<Health>(e).unwrap().invuln =
            GTimer::from_seconds(0.5, TimerMode::Once);
        world.entity_mut(e).insert(Shield {
            timer: GTimer::from_seconds(1.6, TimerMode::Once),
        });
        world.entity_mut(e).insert(Telekinesis {
            timer: GTimer::from_seconds(1.4, TimerMode::Once),
        });
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_player_timers);
        sched.run(&mut world);
        assert!(
            (world
                .get::<Player>(e)
                .unwrap()
                .ability_cooldown
                .elapsed_secs()
                - DT)
                .abs()
                < 1e-6
        );
        assert!(
            (world.get::<Health>(e).unwrap().invuln.remaining_secs() - (0.5 - DT)).abs() < 1e-5
        );
        assert!((world.get::<Shield>(e).unwrap().timer.remaining_secs() - (1.6 - DT)).abs() < 1e-5);
        assert!(
            (world
                .get::<Telekinesis>(e)
                .unwrap()
                .timer
                .remaining_secs()
                - (1.4 - DT))
                .abs()
                < 1e-5
        );
        // A long tick latches everything finished.
        world.resource_mut::<SimTime>().delta_secs = 2.0;
        sched.run(&mut world);
        assert!(world
            .get::<Player>(e)
            .unwrap()
            .ability_cooldown
            .is_finished());
        assert!(world.get::<Health>(e).unwrap().invuln.is_finished());
    }

    #[test]
    fn ally_seeks_nearest_enemy_and_shoots() {
        let mut world = sim_world();
        let ally = world
            .spawn((
                Ally {
                    life: GTimer::from_seconds(12.0, TimerMode::Once),
                    shoot: GTimer::from_seconds(8.0 / 30.0, TimerMode::Repeating),
                },
                Team::Player,
                Pos(Vec2::ZERO),
                Velocity(Vec2::ZERO),
            ))
            .id();
        spawn_enemy(&mut world, Vec2::new(100.0, 0.0));
        spawn_enemy(&mut world, Vec2::new(400.0, 0.0));
        world.resource_mut::<SimTime>().delta_secs = 0.3;
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(ally_ai);
        sched.run(&mut world);
        // Moved toward the NEAREST enemy (+x), not past it.
        let pos = world.get::<Pos>(ally).unwrap().0;
        assert!(pos.x > 0.0 && pos.x <= 100.0, "got {pos:?}");
        assert!(pos.y.abs() < 1e-4, "got {pos:?}");
        // Shoot interval elapsed -> one ally bolt + bolt cue.
        assert_eq!(world.query::<&Projectile>().iter(&world).count(), 1);
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert!(cues.iter().any(|c| c.name == "bolt"));
    }

    #[test]
    fn ally_expires_with_life_timer() {
        let mut world = sim_world();
        let ally = world
            .spawn((
                Ally {
                    life: GTimer::from_seconds(0.1, TimerMode::Once),
                    shoot: GTimer::from_seconds(8.0 / 30.0, TimerMode::Repeating),
                },
                Team::Player,
                Pos(Vec2::ZERO),
                Velocity(Vec2::ZERO),
            ))
            .id();
        world.resource_mut::<SimTime>().delta_secs = 0.2;
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(ally_ai);
        sched.run(&mut world);
        assert!(world.get_entity(ally).is_err());
    }

    #[test]
    fn horror_hold_charges_and_release_resets() {
        let mut world = sim_world();
        let e = spawn_player(&mut world);
        {
            let mut p = world.get_mut::<Player>(e).unwrap();
            p.ability = AbilityKind::HorrorBeam;
            p.rads = 10;
        }
        world.resource_mut::<NtInput>().spec_held = true;
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_hold_abilities);
        for _ in 0..3 {
            sched.run(&mut world);
        }
        // 0.03 charge/tick, 1 rad/tick at cost floor(time+1). Note the
        // one-tick command-flush latency (bevy-identical): the first tick
        // queues the `HorrorCharge` insert, so after 3 ticks time is 0.06.
        let charge = world.get::<HorrorCharge>(e).expect("charging");
        assert!((charge.time - 0.06).abs() < 1e-5, "got {}", charge.time);
        assert_eq!(world.get::<Player>(e).unwrap().rads, 7);
        assert_eq!(world.query::<&Projectile>().iter(&world).count(), 3);
        // Release zeroes and removes the charge.
        world.resource_mut::<NtInput>().spec_held = false;
        sched.run(&mut world);
        assert!(world.get::<HorrorCharge>(e).is_none());
    }

    #[test]
    fn frog_hold_roots_and_release_spawns_clouds() {
        let mut world = sim_world();
        let e = spawn_player(&mut world);
        world.get_mut::<Player>(e).unwrap().ability = AbilityKind::ToxicPuke;
        world.get_mut::<Velocity>(e).unwrap().0 = Vec2::new(50.0, 0.0);
        world.resource_mut::<NtInput>().spec_held = true;
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_hold_abilities);
        sched.run(&mut world);
        // Charging roots the player. Gas builds at 30/s, but the first
        // tick only queues the `FrogCharge` insert (command-flush latency,
        // bevy-identical), so run once more to observe the charge.
        assert_eq!(world.get::<Velocity>(e).unwrap().0, Vec2::ZERO);
        sched.run(&mut world);
        let gas = world.get::<FrogCharge>(e).expect("charging").gas;
        assert!((gas - DT * 30.0).abs() < 1e-5, "got {gas}");
        // Release with a fixed charge spawns that many clouds + boom cue.
        world.entity_mut(e).insert(FrogCharge { gas: 5.0 });
        world.resource_mut::<NtInput>().spec_held = false;
        sched.run(&mut world);
        assert!(world.get::<FrogCharge>(e).is_none());
        assert_eq!(world.query::<&HazardCloud>().iter(&world).count(), 5);
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert!(cues.iter().any(|c| c.name == "sndExplosionL"));
    }

    #[test]
    fn telekinesis_hold_pulls_enemies_and_pushes_shots() {
        let mut world = sim_world();
        let e = spawn_player(&mut world);
        world.get_mut::<Player>(e).unwrap().ability = AbilityKind::Telekinesis;
        world.resource_mut::<NtInput>().spec_held = true;
        let foe = spawn_enemy(&mut world, Vec2::new(100.0, 0.0));
        // Far outside the 160x120 box: untouched.
        let far = spawn_enemy(&mut world, Vec2::new(500.0, 0.0));
        // Enemy shot inside the box gets pushed away from the player.
        world.spawn((
            Projectile {
                damage: 1,
                life: GTimer::from_seconds(1.0, TimerMode::Once),
                radius: 3.0,
                knockback: 0.0,
                explosive: false,
                source: None,
            },
            Team::Enemy,
            Pos(Vec2::new(50.0, 0.0)),
            Velocity(Vec2::ZERO),
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_hold_abilities);
        sched.run(&mut world);
        // 30 px/s pull over one tick toward the player (-x).
        let v = world.get::<Velocity>(foe).unwrap().0;
        assert!((v.x + 30.0 * DT).abs() < 1e-5, "got {v:?}");
        assert_eq!(world.get::<Velocity>(far).unwrap().0, Vec2::ZERO);
        assert!(world.get::<Telekinesis>(e).is_some());
        // The enemy shot (a `Projectile`, not an `Enemy`) is pushed away.
        let mut shots = world.query::<(&Projectile, &Velocity)>();
        assert_eq!(shots.iter(&world).count(), 1);
        for (_, vel) in shots.iter(&world) {
            assert!(vel.0.x > 0.0, "shot pushed out, got {:?}", vel.0);
        }
    }

    #[test]
    fn projectile_style_reels_own_shots() {
        use crate::data::UltraMutationId;
        let mut world = sim_world();
        let e = spawn_player(&mut world);
        {
            let mut p = world.get_mut::<Player>(e).unwrap();
            p.ability = AbilityKind::Telekinesis;
            p.ultra = Some(UltraMutationId::EyesProjectileStyle);
        }
        world.resource_mut::<NtInput>().spec_held = true;
        // Own shot drifting past: reeled toward the 8 px orbit.
        let mine = world
            .spawn((
                Projectile {
                    damage: 1,
                    life: GTimer::from_seconds(5.0, TimerMode::Once),
                    radius: 3.0,
                    knockback: 0.0,
                    explosive: false,
                    source: Some(DamageSource::player_weapon(e, WeaponId::NONE)),
                },
                Team::Player,
                Pos(Vec2::new(100.0, 0.0)),
                Velocity(Vec2::new(0.0, 60.0)),
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_hold_abilities);
        sched.run(&mut world);
        let v = world.get::<Velocity>(mine).unwrap().0;
        // Pulled toward the orbit point ahead: -x, and the slow-shot
        // boost grows overall speed.
        assert!(v.x < 0.0, "reeled in, got {v:?}");
        assert!(v.length() > 60.0, "slow shots speed up, got {v:?}");
    }

    #[test]
    fn held_weapon_angle_laws() {
        // Guns sit on aim.
        assert!((held_weapon_angle(0.5, 0.0, 0.0) - 0.5).abs() < 1e-6);
        // Melee rests cocked ±120°.
        let rest = held_weapon_angle(0.0, 120.0, 0.0);
        assert!((rest - 120.0_f32.to_radians()).abs() < 1e-6);
        // Full kick settles back onto aim.
        assert!((held_weapon_angle(0.7, 120.0, 20.0) - 0.7).abs() < 1e-6);
        // Negative wkick (swing start) sweeps further out.
        let sweep = held_weapon_angle(0.0, 120.0, -4.0);
        assert!((sweep - 120.0_f32.to_radians() * 1.2).abs() < 1e-6);
        // Half kick halves the offset.
        let half = held_weapon_angle(1.0, -120.0, 10.0);
        assert!((half - (1.0 - 120.0_f32.to_radians() * 0.5)).abs() < 1e-6);
    }
}
