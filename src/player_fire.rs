//! Player combat: firing, melee, abilities, and ability-field ticks.
//!
//! Headless port of the COMBAT half of
//! `nt-recreated-bevy/src/game/player.rs`: `player_fire`,
//! `fire_burst_volley`, `fire_one_gun`, `apply_weapon_mutation_mods`,
//! `spawn_pellets`, `slash_life_secs`, `flip_melee_angle`, `melee_attack`,
//! `pay_fire_cost`, `spawn_beam_shot`, `spawn_player_projectile`,
//! `spawn_player_projectile_with_source`, `hammerhead_chew`,
//! `move_swing_fx` (lifetime tick only — see below), `tick_snare_zones`,
//! `tick_slowed`, `tick_portal_strikes`, `tick_hazard_clouds`,
//! `player_ability`.
//!
//! Deferred (already ported or render-only, per the task):
//! - `player_move`, `face_aim`, `player_aim`, `weapon_switch`,
//!   `tick_player_timers`, `blink_player` — live in `crate::player` or
//!   are render-only; not duplicated here.
//! - `ally_ai`, `tick_hold_abilities` (Eyes/Horror/Frog hold), weapon
//!   visuals (`ensure_weapon_visual`, `tick_weapon_visuals`,
//!   `held_weapon_angle`) — render or a separate slice; not ported.
//! - `move_swing_fx` has no hitbox motion in bevy (slash hitboxes ride
//!   `Velocity` integration in `move_projectiles`); the port keeps the
//!   `SwingFx` lifetime tick so the marker cannot leak.
//!
//! Headless adaptations (no bevy engine):
//! - `Transform` -> [`Pos`]; `Timer` -> [`GTimer`]; `Time<Fixed>` ->
//!   [`SimTime`]; thread rng -> `rand::rng()`.
//! - Audio spawns -> [`AudioCue`]s pushed into [`Queue`] (the platform
//!   layer plays them). Stems mirror the bevy names (`shoot`,
//!   `shotgun`, `empty`, …).
//! - `VfxSpawner` bursts -> [`spawn_burst`]; damage numbers ->
//!   `repame_fx::spawn_number`; trauma/hitstop/slow-mo/rumble keep
//!   their sim-side sinks.
//! - Shell casings, gun visuals, juice pop-ins, screen
//!   flashes are render juice and omitted (no `todo!()` stubs). Muzzle
//!   flashes keep a 3-tick [`FiredWeapon`] marker for the render phase.
//! - Projectile art (catalog strips, anchors, anims) is renderer-side
//!   here (see `spawns.rs`); spawns carry sim markers only
//!   (`ProjectileTyp`, `ProjectileFade` paths, `PlasmaSize`, …).
//! - Slash anim length is catalog data in bevy; headless slashes use
//!   the 3-frame default (`slash_life_secs(3)`).
//! - `ProjectileArchetype` (bevy `projectile_archetypes.rs`) is
//!   re-expressed minimally as [`FireArch`]: only the fields the fire
//!   path branches on.

use bevy_ecs::prelude::*;
use glam::Vec2;
use rand::RngExt;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::audio::AudioCue;
use crate::comps_a::{
    AbilityHazard, AimDir, BouncesLeft, ChainLightning, DamageSource, DiscFlight, FireCooldown,
    FlameShellSlowDeath, FlameTrail, GameCleanup, GrenadeFuse, HammerheadBudget, Health, HitId,
    Hitbox, HitsAllTeams, Homing, Inventory, LevelCleanup, PendingWallBreak, PiercesLeft, Player,
    Projectile, ProjectileFade, ProjectileFriction, ProjectileHitSet, ProjectileTyp,
    ProjectileVisual, RaceState,
    Run, SaveDirty, ShellBonus, ShellWallBounce, SlashProjectile, SpawnGrace, SpawnHazardOnDeath,
    SplitOnDeath, Sticky, Team, Toast, Velocity, WallCell, WallTile,
};
use crate::comps_b::{
    Ally, BloodAmmo, ChestKind, CryAnim, CustomExplosion, Dash, DeploysSentry, Enemy, HazardCloud,
    PickupKind, PlasmaBurst, PopPopCharges, PortalStrike, PortalSucking, Prop, PropSprites,
    SecretEntrance, Shield, Slowed, SnareZone, SpawnsWeaponPickup, SwingFx, Telekinesis,
    WeaponVisual,
};
use crate::environment::{PropDeathEffect, spawn_prop_corpse, spawn_prop_death_effect};
use crate::worldgen::WALL_PX;
use crate::data::{
    AbilityKind, AmmoKind, AreaId, CrownKind, HazardDef, HazardKind, MutationId, RaceId, SplitDef,
    UltraMutationId, WeaponId, WeaponKind, ammo_pickup_amount,
};
use crate::effects::{
    ChromaticAberration, FiredWeapon, HitStop, RumbleRequest, SlowMotion, chromatic_pulse,
    rumble, slow_motion, spawn_burst,
};
use crate::input::NtInput;
use crate::msg::Queue;
use crate::pickups::{spawn_chest, spawn_flung_weapon_pickup, spawn_pickup, spawn_rad_burst};
use crate::savedata_part::{SaveData, character_def, check_progress_unlocks};
use crate::secrets::SecretTriggers;
use crate::spatial::{Pos, PLAYER_RADIUS};
use crate::time::{GTimer, TimerMode};
use crate::weapon_runtime::{
    MeleeDef, WeaponDef, base_weapon_name, gml_fire_push_px_s, gml_melee_wkick,
    melee_projectile_spec, weapon_meta, weapon_runtime_def, weapon_sleep_secs,
};
use crate::weapons_data::AmmoType;

// ---------------------------------------------------------------------------
// Small local helpers
// ---------------------------------------------------------------------------

/// Push a fire-and-forget cue (bevy `GameAudio::play_*` parity for the
/// weapon stems that the headless `GameAudio` bank does not own yet).
fn cue(cues: &mut Queue<AudioCue>, name: &'static str, volume: f32, variance: f32) {
    cues.push(AudioCue {
        name,
        volume,
        variance,
    });
}

/// Ammo kind for a weapon id (same rule as the bevy build: `AmmoType`
/// maps 1:1 onto `AmmoKind`).
fn weapon_ammo(id: WeaponId) -> AmmoKind {
    match weapon_meta(id).wep_type {
        AmmoType::None => AmmoKind::None,
        AmmoType::Bullets => AmmoKind::Bullets,
        AmmoType::Shells => AmmoKind::Shells,
        AmmoType::Bolts => AmmoKind::Bolts,
        AmmoType::Explosives => AmmoKind::Explosives,
        AmmoType::Energy => AmmoKind::Energy,
    }
}

/// Steroids second slot: single shared definition in `crate::player`
/// (bevy-verbatim body); reused by the fire path.
use crate::player::steroids_secondary_slot;

// ---------------------------------------------------------------------------
// Projectile archetype (minimal headless `ProjectileArchetype`)
// ---------------------------------------------------------------------------

/// Headless beam spec (bevy `BeamSpec` with `Color` -> `[f32; 4]`).
#[derive(Clone, Copy, Debug)]
pub struct BeamShot {
    pub length: f32,
    pub width: f32,
    pub damage: i32,
    pub knockback: f32,
    pub duration: f32,
    pub tick: f32,
    pub color: [f32; 4],
}

/// Only the archetype fields the fire path branches on (bevy
/// `ProjectileArchetype` parity, minus art).
#[derive(Clone, Copy, Debug, Default)]
pub struct FireArch {
    pub homing: Option<Homing>,
    pub sticky: Option<Sticky>,
    pub chain: Option<ChainLightning>,
    pub sentry: Option<DeploysSentry>,
    pub custom: Option<CustomExplosion>,
    pub blood: Option<BloodAmmo>,
    pub pickup: Option<SpawnsWeaponPickup>,
    pub beam: Option<BeamShot>,
    pub plasma: Option<PlasmaBurst>,
    pub hits_all: bool,
}

/// Archetype lookup by (base) weapon name, mirroring bevy
/// `projectile_archetype` (GOLDEN/ULTRA/CURSED prefixes strip to base).
fn projectile_arch(id: WeaponId) -> FireArch {
    let base = base_weapon_name(weapon_meta(id).wep_name);
    match base {
        "SENTRY GUN" => FireArch {
            sentry: Some(DeploysSentry {
                life: 14.0,
                fire_interval: 0.18,
                range: 360.0,
                projectile_speed: 640.0,
                projectile_damage: 3,
            }),
            ..FireArch::default()
        },
        "SMART GUN" => FireArch {
            homing: Some(Homing {
                turn_rate: 8.0,
                acquire_range: 420.0,
            }),
            ..FireArch::default()
        },
        "SEEKER PISTOL" | "SEEKER SHOTGUN" => FireArch {
            homing: Some(Homing {
                turn_rate: 5.5,
                acquire_range: 420.0,
            }),
            ..FireArch::default()
        },
        "STICKY LAUNCHER" => FireArch {
            sticky: Some(Sticky::default()),
            custom: Some(CustomExplosion {
                radius: 32.0,
                count: 3,
                spread: 16.0,
            }),
            ..FireArch::default()
        },
        "GRENADE LAUNCHER" | "GOLDEN GRENADE LAUNCHER" => FireArch {
            custom: Some(CustomExplosion::default()),
            ..FireArch::default()
        },
        "NUKE LAUNCHER" => FireArch {
            custom: Some(CustomExplosion {
                radius: 32.0,
                count: 8,
                spread: 12.0,
            }),
            ..FireArch::default()
        },
        "ION CANNON" => FireArch {
            beam: Some(BeamShot {
                length: 760.0,
                width: 28.0,
                damage: 18,
                knockback: 120.0,
                duration: 0.2,
                tick: 1.0 / 30.0,
                color: [0.52, 0.85, 1.0, 1.0],
            }),
            ..FireArch::default()
        },
        "LASER CANNON" => FireArch {
            beam: Some(BeamShot {
                length: 820.0,
                width: 24.0,
                damage: 18,
                knockback: 100.0,
                duration: 0.18,
                tick: 1.0 / 30.0,
                color: [1.0, 0.18, 0.14, 1.0],
            }),
            ..FireArch::default()
        },
        "BLOOD LAUNCHER" => FireArch {
            blood: Some(BloodAmmo { hp_cost: 1 }),
            ..FireArch::default()
        },
        "BLOOD CANNON" => FireArch {
            blood: Some(BloodAmmo { hp_cost: 2 }),
            ..FireArch::default()
        },
        "GUN GUN" => FireArch {
            pickup: Some(SpawnsWeaponPickup {
                weapon: None,
                decide_extra: 10,
            }),
            ..FireArch::default()
        },
        "LIGHTNING PISTOL" | "LIGHTNING SMG" => FireArch {
            chain: Some(ChainLightning {
                jumps_left: 1,
                range: 170.0,
                falloff: 0.8,
            }),
            ..FireArch::default()
        },
        "LIGHTNING RIFLE" => FireArch {
            chain: Some(ChainLightning {
                jumps_left: 2,
                range: 190.0,
                falloff: 0.75,
            }),
            ..FireArch::default()
        },
        "LIGHTNING SHOTGUN" => FireArch {
            chain: Some(ChainLightning {
                jumps_left: 2,
                range: 155.0,
                falloff: 0.8,
            }),
            ..FireArch::default()
        },
        "LIGHTNING CANNON" => FireArch {
            chain: Some(ChainLightning {
                jumps_left: 4,
                range: 220.0,
                falloff: 0.7,
            }),
            ..FireArch::default()
        },
        "PLASMA GUN" => FireArch {
            plasma: Some(PlasmaBurst {
                pellets: 4,
                speed: 300.0,
                damage: 2,
                lifetime: 0.32,
                radius: 3.0,
                knockback: 24.0,
                color: [0.35, 1.0, 0.42, 1.0],
                size: Vec2::splat(7.0),
            }),
            ..FireArch::default()
        },
        "PLASMA RIFLE" => FireArch {
            plasma: Some(PlasmaBurst {
                pellets: 5,
                speed: 320.0,
                damage: 2,
                lifetime: 0.34,
                radius: 3.0,
                knockback: 24.0,
                color: [0.3, 1.0, 0.38, 1.0],
                size: Vec2::splat(7.0),
            }),
            ..FireArch::default()
        },
        "PLASMA MINIGUN" => FireArch {
            plasma: Some(PlasmaBurst {
                pellets: 3,
                speed: 330.0,
                damage: 1,
                lifetime: 0.26,
                radius: 2.5,
                knockback: 16.0,
                color: [0.32, 1.0, 0.4, 1.0],
                size: Vec2::splat(6.0),
            }),
            ..FireArch::default()
        },
        "PLASMA CANNON" | "SUPER PLASMA CANNON" => FireArch {
            plasma: Some(PlasmaBurst {
                pellets: 8,
                speed: 360.0,
                damage: 3,
                lifetime: 0.38,
                radius: 4.0,
                knockback: 40.0,
                color: [0.36, 1.0, 0.45, 1.0],
                size: Vec2::splat(8.0),
            }),
            ..FireArch::default()
        },
        "DEVASTATOR" => FireArch {
            plasma: Some(PlasmaBurst {
                pellets: 10,
                speed: 380.0,
                damage: 4,
                lifetime: 0.42,
                radius: 4.0,
                knockback: 48.0,
                color: [0.38, 1.0, 0.48, 1.0],
                size: Vec2::splat(9.0),
            }),
            ..FireArch::default()
        },
        "DISC GUN" | "SUPER DISC GUN" | "BOUNCER SMG" | "BOUNCER SHOTGUN" => FireArch {
            hits_all: true,
            ..FireArch::default()
        },
        "CROSSBOW" | "HEAVY CROSSBOW" | "AUTO CROSSBOW" | "SUPER CROSSBOW"
        | "HEAVY AUTO CROSSBOW" | "ULTRA CROSSBOW" | "SPLINTER GUN" | "SPLINTER PISTOL"
        | "SUPER SPLINTER GUN" | "TOXIC BOW" => FireArch {
            sticky: Some(Sticky::default()),
            ..FireArch::default()
        },
        _ => {
            if base.contains("DISC") || base.contains("BOUNCER") {
                FireArch {
                    hits_all: true,
                    ..FireArch::default()
                }
            } else {
                FireArch::default()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Chained records (deaths.rs pattern): the fire chain takes >16 params in
// bevy, so systems pack them into these two records instead.
// ---------------------------------------------------------------------------

/// Feel sinks + toast shared by the whole fire chain.
pub struct FireFx<'a> {
    pub trauma: &'a mut Trauma,
    pub hitstop: &'a mut HitStop,
    pub cues: &'a mut Queue<AudioCue>,
    pub rumble: &'a mut Queue<RumbleRequest>,
    pub toast: &'a mut Toast,
    pub shake_scale: f32,
    pub underwater: bool,
    /// Strip catalog (bevy `AssetCatalog`): slash life reads the fired
    /// strip's frame count, never a constant.
    pub catalog: &'a repame_anim::AnimCatalog,
}

/// One gun's shot: shooter, muzzle origin, aim, and resolved def.
#[derive(Clone, Copy, Debug)]
pub struct GunShot {
    pub player_ent: Entity,
    pub pos: Vec2,
    pub aim: Vec2,
    pub weapon_id: WeaponId,
    pub def: WeaponDef,
    pub visual_slot: u8,
}

// ---------------------------------------------------------------------------
// player_fire
// ---------------------------------------------------------------------------

/// Firing Cadence: burst continuations, then primary / Steroids-secondary
/// intents. `Transform` -> [`Pos`]; pulses drained every tick.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn player_fire(
    time: Res<SimTime>,
    mut input: ResMut<NtInput>,
    mut commands: Commands,
    mut trauma: ResMut<Trauma>,
    mut hitstop: ResMut<HitStop>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut rumble_q: ResMut<Queue<RumbleRequest>>,
    mut toast: ResMut<Toast>,
    mut run: ResMut<Run>,
    save: Res<SaveData>,
    mut player_q: Query<
        (
            Entity,
            &Pos,
            &AimDir,
            &mut Player,
            &mut Health,
            &RaceState,
            Option<&PortalSucking>,
        ),
        With<Player>,
    >,
    mut fire_q: Query<(&mut FireCooldown, &mut Inventory, &mut Velocity), With<Player>>,
    mut vis_q: Query<&mut WeaponVisual>,
    mut pop_q: Query<&mut PopPopCharges>,
    catalog: Res<repame_anim::AnimCatalog>,
    mut tut: Option<ResMut<crate::state::TutorialState>>,
) {
    let Ok((player_ent, pos, aim, mut player, mut health, race_state, sucking)) =
        player_q.single_mut()
    else {
        return;
    };
    if sucking.is_some() {
        return;
    }
    let Ok((mut cooldown, mut inv, mut vel)) = fire_q.single_mut() else {
        return;
    };

    let is_steroids = race_state.race == RaceId::Steroids;
    let shake_scale: f32 = save.settings.screenshake.clamp(0.0, 2.0);

    let dt = time.delta_secs;
    cooldown.timer.tick(dt);
    cooldown.burst_timer.tick(dt);
    // GML `Player/Step_0`: reload finish flips the melee mirror.
    if cooldown.timer.just_finished() {
        inv.wepflip *= -1.0;
    }
    if cooldown.timer_b.just_finished() {
        inv.bwepflip *= -1.0;
    }
    cooldown.timer_b.tick(dt);
    cooldown.burst_timer_b.tick(dt);

    let fire_held = input.fire_held;
    let fire_pressed = input.take_fire_pressed();
    let spec_held = input.spec_held;
    let spec_pressed = input.take_spec_pressed();

    let primary_id = inv.weapons[inv.current];
    let primary_def = weapon_runtime_def(primary_id);

    // Second slot mirrors the other live slot.
    let second_slot = steroids_secondary_slot(inv.current, inv.weapon_slots);
    let secondary_id = if is_steroids {
        inv.weapons[second_slot]
    } else {
        WeaponId::NONE
    };
    let secondary_def = weapon_runtime_def(secondary_id);

    let mut fx = FireFx {
        trauma: &mut trauma,
        hitstop: &mut hitstop,
        cues: &mut cues,
        rumble: &mut rumble_q,
        toast: &mut toast,
        shake_scale,
        underwater: matches!(run.area, AreaId::Oasis),
        catalog: &catalog,
    };
    let origin = pos.0;
    let aim_v = aim.0;

    if cooldown.burst_left > 0 && cooldown.burst_timer.is_finished() && primary_id != WeaponId::NONE
    {
        let shot = GunShot {
            player_ent,
            pos: origin,
            aim: aim_v,
            weapon_id: primary_id,
            def: primary_def,
            visual_slot: 0,
        };
        // GML `scrFire`: non-melee gunfire records `hasfiredshots`.
        if primary_def.melee.is_none() {
            run.shots_fired += 1;
        }
        fire_burst_volley(
            &mut commands,
            &mut fx,
            &shot,
            &player,
            &mut pop_q,
            &mut vis_q,
        );
        cooldown.burst_left -= 1;
        cooldown.burst_timer = GTimer::from_seconds(primary_def.burst_interval, TimerMode::Once);
    }
    if is_steroids
        && cooldown.burst_left_b > 0
        && cooldown.burst_timer_b.is_finished()
        && secondary_id != WeaponId::NONE
    {
        let shot = GunShot {
            player_ent,
            pos: origin,
            aim: aim_v,
            weapon_id: secondary_id,
            def: secondary_def,
            visual_slot: 1,
        };
        if secondary_def.melee.is_none() {
            run.shots_fired += 1;
        }
        fire_burst_volley(
            &mut commands,
            &mut fx,
            &shot,
            &player,
            &mut pop_q,
            &mut vis_q,
        );
        cooldown.burst_left_b -= 1;
        cooldown.burst_timer_b =
            GTimer::from_seconds(secondary_def.burst_interval, TimerMode::Once);
    }

    let primary_intent = if primary_def.automatic || is_steroids {
        fire_held
    } else {
        fire_pressed
    };

    let secondary_intent = is_steroids && spec_held && secondary_id != WeaponId::NONE;

    if primary_id != WeaponId::NONE && primary_intent && cooldown.timer.is_finished() {
        let shot = GunShot {
            player_ent,
            pos: origin,
            aim: aim_v,
            weapon_id: primary_id,
            def: primary_def,
            visual_slot: 0,
        };
        if primary_def.melee.is_none() {
            run.shots_fired += 1;
        }
        // GML `scrPlayerFiring:45` verbatim: a fired shot latches the
        // tutorial Shooting step.
        if let Some(tut) = tut.as_deref_mut() {
            tut.complete_step(crate::state::TutorialStep::Shooting);
        }
        fire_one_gun(
            &mut commands,
            &mut fx,
            &shot,
            &mut player,
            &mut inv,
            &mut health,
            &mut vel,
            &mut cooldown,
            &mut pop_q,
            &mut vis_q,
        );
    }

    if secondary_intent && cooldown.timer_b.is_finished() {
        let secondary_def = weapon_runtime_def(secondary_id);
        let shot = GunShot {
            player_ent,
            pos: origin,
            aim: aim_v,
            weapon_id: secondary_id,
            def: secondary_def,
            visual_slot: 1,
        };
        if secondary_def.melee.is_none() {
            run.shots_fired += 1;
        }
        fire_one_gun(
            &mut commands,
            &mut fx,
            &shot,
            &mut player,
            &mut inv,
            &mut health,
            &mut vel,
            &mut cooldown,
            &mut pop_q,
            &mut vis_q,
        );
    }

    let _ = spec_pressed;
}

fn fire_burst_volley(
    commands: &mut Commands,
    fx: &mut FireFx,
    shot: &GunShot,
    player: &Player,
    pop_q: &mut Query<&mut PopPopCharges>,
    vis_q: &mut Query<&mut WeaponVisual>,
) {
    for mut wv in vis_q.iter_mut() {
        if wv.owner == shot.player_ent && wv.slot == shot.visual_slot {
            wv.wkick = shot.def.recoil;
        }
    }
    spawn_pellets(commands, fx, shot, player);
    if let Ok(mut charges) = pop_q.get_mut(shot.player_ent)
        && charges.0 > 0
    {
        charges.0 -= 1;
        spawn_pellets(commands, fx, shot, player);
        if charges.0 == 0 {
            commands.entity(shot.player_ent).remove::<PopPopCharges>();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fire_one_gun(
    commands: &mut Commands,
    fx: &mut FireFx,
    shot: &GunShot,
    player: &mut Player,
    inv: &mut Inventory,
    health: &mut Health,
    vel: &mut Velocity,
    cooldown: &mut FireCooldown,
    pop_q: &mut Query<&mut PopPopCharges>,
    vis_q: &mut Query<&mut WeaponVisual>,
) {
    let def = shot.def;
    let archetype = projectile_arch(shot.weapon_id);

    if def.rad_cost > 0 && player.rads < def.rad_cost {
        fx.toast.show("NOT ENOUGH RADS");
        cue(fx.cues, "ultra_empty", 0.5, 0.05);
        for mut wv in vis_q.iter_mut() {
            if wv.owner == shot.player_ent && wv.slot == shot.visual_slot {
                wv.wkick = -2.0;
            }
        }
        return;
    }

    // GML `NadeBurst` ammo is `3 + Death-crown` at fire time; the
    // grenade-shotgun pellet count is `(3|4) + Death-crown`. Both
    // resolve here (burst volleys + immediate arms share this path).
    let mut def = shot.def;
    if shot.weapon_id == WeaponId(80) && def.burst_shots > 1 {
        def.burst_shots += usize::from(player.crown == CrownKind::Death);
    }
    // GML grenade shotguns roll `(3|4) + Death-crown` pellets per
    // trigger pull; the crown bonus rides the def so burst-less dups
    // and volleys agree.
    if shot.weapon_id == WeaponId(79) || shot.weapon_id == WeaponId(85) {
        def.pellets += usize::from(player.crown == CrownKind::Death);
    }
    let shot = GunShot { def, ..*shot };
    let shot = &shot;

    if def.melee.is_none() {
        // GML `scrPlayerFiring`: blood weapons consume ammo normally;
        // on empty (and only via the press path, not bursts/dups) the
        // click refills cost-worth of ammo for 1 HP (`scrBloodAmmoRefill`)
        // and fires this same click. The sim models that as: ammo
        // shortfall on a blood weapon with hp > 1 prepays 1 HP for
        // `ammo_cost` ammo, then the normal deduction below fires.
        if archetype.blood.is_some() && !player.free_ammo {
            let slot = inv.ammo_mut(def.ammo);
            if *slot < def.ammo_cost && health.hp > 1 {
                health.hp -= 1;
                *slot += def.ammo_cost;
                repame_fx::spawn_number(
                    commands,
                    shot.pos.x,
                    shot.pos.y,
                    "1".to_string(),
                    [1.0, 0.35, 0.35, 1.0],
                );
                cue(fx.cues, "sndBloodHurt", 0.8, 0.2);
                fx.hitstop.trigger(0.35, 0.08);
            }
        }
        match pay_fire_cost(inv, health, def.ammo, def.ammo_cost, None, player.free_ammo) {
            AmmoPayment::Paid => {}
            AmmoPayment::Failed => {
                if inv.ammo_of(def.ammo) > 0 {
                    fx.toast.show("NOT ENOUGH AMMO");
                } else {
                    fx.toast.show("EMPTY");
                }
                cue(fx.cues, "empty", 0.5, 0.05);
                for mut wv in vis_q.iter_mut() {
                    if wv.owner == shot.player_ent && wv.slot == shot.visual_slot {
                        wv.wkick = -2.0;
                    }
                }
                return;
            }
        }
        if def.rad_cost > 0 {
            player.rads = player.rads.saturating_sub(def.rad_cost);
        }
    }

    let stress_bonus = if player.stress {
        (1.0 - health.hp as f32 / health.max.max(1) as f32).max(0.0)
    } else {
        0.0
    };
    let cd = def.cooldown * player.fire_rate_mult / (1.0 + stress_bonus);

    let timer = if shot.visual_slot == 0 {
        &mut cooldown.timer
    } else {
        &mut cooldown.timer_b
    };
    *timer = GTimer::from_seconds(cd.max(0.03), TimerMode::Once);
    match def.ammo {
        AmmoKind::Shells => cue(fx.cues, "shot_reload", 0.4, 0.1),
        AmmoKind::Bolts => cue(fx.cues, "cross_reload", 0.4, 0.1),
        AmmoKind::Explosives => cue(fx.cues, "nade_reload", 0.4, 0.1),
        AmmoKind::Energy => {
            if def.name.contains("LIGHTNING") {
                cue(fx.cues, "lightning_reload", 0.4, 0.1);
            } else {
                cue(fx.cues, "plasma_reload", 0.4, 0.1);
            }
        }
        _ => {}
    }

    if let Some(melee) = def.melee {
        melee_attack(commands, fx, shot, player, health, vel, melee, vis_q);
        return;
    }

    spawn_pellets(commands, fx, shot, player);
    vel.0 -= shot.aim.normalize_or_zero() * gml_fire_push_px_s(shot.def.name);
    for mut wv in vis_q.iter_mut() {
        if wv.owner == shot.player_ent && wv.slot == shot.visual_slot {
            wv.wkick = def.recoil;
        }
    }
    if player.recycle_gland
        && def.ammo == AmmoKind::Bullets
        && def.melee.is_none()
        && rand::rng().random_range(0..5) == 0
    {
        let slot = inv.ammo_mut(AmmoKind::Bullets);
        *slot = (*slot + 1).min(player.ammo_cap(AmmoKind::Bullets));
    }

    if let Ok(mut charges) = pop_q.get_mut(shot.player_ent)
        && charges.0 > 0
    {
        charges.0 -= 1;
        let mut can_dup = true;
        if def.melee.is_none() && def.ammo != AmmoKind::None && def.ammo_cost > 0 {
            match pay_fire_cost(inv, health, def.ammo, def.ammo_cost, None, player.free_ammo) {
                AmmoPayment::Paid => {}
                AmmoPayment::Failed => {
                    can_dup = false;
                }
            }
        }
        if can_dup {
            spawn_pellets(commands, fx, shot, player);
            let mult = if player.throne_butt
                || matches!(player.ultra, Some(UltraMutationId::VenuzBack2Bizniz))
            {
                3.0
            } else {
                2.0
            };
            let timer = if shot.visual_slot == 0 {
                &mut cooldown.timer
            } else {
                &mut cooldown.timer_b
            };
            timer.set_duration((def.cooldown * player.fire_rate_mult * mult).max(0.03));
            vel.0 -= shot.aim.normalize_or_zero() * 8.0 * 30.0 * 0.15;
        }
        if charges.0 == 0 {
            commands.entity(shot.player_ent).remove::<PopPopCharges>();
        }
    }

    if def.burst_shots > 1 {
        if shot.visual_slot == 0 {
            cooldown.burst_left = def.burst_shots - 1;
            cooldown.burst_timer = GTimer::from_seconds(def.burst_interval, TimerMode::Once);
        } else {
            cooldown.burst_left_b = def.burst_shots - 1;
            cooldown.burst_timer_b = GTimer::from_seconds(def.burst_interval, TimerMode::Once);
        }
    }
}

fn apply_weapon_mutation_mods(def: &mut WeaponDef, arch: &mut FireArch, player: &Player) {
    def.damage = ((def.damage as f32) * player.ultra_damage_mult).round() as i32;

    if player.laser_brain && def.ammo == AmmoKind::Energy && def.melee.is_none() {
        def.damage = ((def.damage as f32) * 1.35).round() as i32;
        def.speed *= 1.15;
        def.size *= 1.15;
        def.projectile_radius *= 1.15;
    }

    if player.shotgun_shoulders && def.ammo == AmmoKind::Shells && def.melee.is_none() {
        def.bounces = def.bounces.max(5);
        def.lifetime *= 1.25;
    }

    if player.bolt_marrow && def.ammo == AmmoKind::Bolts && def.melee.is_none() {
        arch.homing = Some(arch.homing.unwrap_or(Homing {
            turn_rate: 7.0,
            acquire_range: 420.0,
        }));
    }
}

fn spawn_pellets(commands: &mut Commands, fx: &mut FireFx, shot: &GunShot, player: &Player) {
    let id = shot.weapon_id;
    let sleep = weapon_sleep_secs(id);
    if sleep > 0.0 {
        fx.hitstop.trigger((sleep * 8.0).clamp(0.15, 0.85), sleep);
    }
    fx.trauma.add(shot.def.shake * fx.shake_scale);
    rumble(fx.rumble, 0.08, shot.def.shake, 0.07);

    let kind: WeaponKind = id.into();
    if fx.underwater {
        cue(fx.cues, "fire_gml_water", 0.7, 0.05);
    } else {
        let legacy_fallback = matches!(
            kind,
            WeaponKind::Revolver
                | WeaponKind::Machinegun
                | WeaponKind::Smg
                | WeaponKind::AssaultRifle
                | WeaponKind::Shotgun
                | WeaponKind::Crossbow
                | WeaponKind::GrenadeLauncher
        );
        if legacy_fallback {
            let name = shot.def.name;
            let is_gold =
                name.contains("GOLDEN") || name.contains("GOLD ") || name.starts_with("GOLD");
            if is_gold {
                cue(fx.cues, "fire_gml", 0.7, 0.05);
            } else {
                match kind {
                    WeaponKind::Revolver => cue(fx.cues, "shoot", 0.7, 0.05),
                    WeaponKind::Machinegun | WeaponKind::Smg | WeaponKind::AssaultRifle => {
                        cue(fx.cues, "machine", 0.6, 0.05);
                    }
                    WeaponKind::Shotgun => cue(fx.cues, "shotgun", 0.8, 0.04),
                    WeaponKind::Crossbow => cue(fx.cues, "bolt", 0.7, 0.05),
                    WeaponKind::GrenadeLauncher => cue(fx.cues, "explode", 0.8, 0.04),
                    _ => cue(fx.cues, "fire_gml", 0.7, 0.05),
                }
            }
        } else {
            cue(fx.cues, "fire_gml", 0.7, 0.05);
        }
    }

    // GML `scrFire` style: Bullet1 spawns near the body (player x/y plus a
    // tiny forward nudge), not a fixed 24px out. Keep longer muzzles for
    // bolts/beams/launchers whose strips originate ahead of the grip.
    let muzzle_dist = match shot.def.ammo {
        AmmoKind::Bullets | AmmoKind::Shells => 8.0,
        _ if shot.def.melee.is_some() => 0.0,
        _ => 24.0,
    };
    let muzzle = shot.pos + shot.aim.normalize_or_zero() * muzzle_dist;
    // Render-phase muzzle tongue (3-tick marker; expiry in Always tail).
    // Firing logic untouched: bursts/pellets/beams below are unchanged.
    commands.spawn(FiredWeapon::new(muzzle, shot.aim));
    if shot.def.muzzle_burst > 0 {
        let mut rng = rand::rng();
        spawn_burst(
            commands,
            &mut rng,
            muzzle,
            shot.def.muzzle_burst,
            [1.0, 0.85, 0.25, 1.0],
            (40.0, 120.0),
        );
    }

    let mut archetype = projectile_arch(id);
    let mut def = shot.def;
    apply_weapon_mutation_mods(&mut def, &mut archetype, player);

    if let Some(beam) = archetype.beam {
        spawn_beam_shot(
            commands,
            muzzle,
            shot.aim.normalize_or_zero(),
            beam,
            Some(DamageSource::player_weapon(shot.player_ent, id)),
        );
        return;
    }

    if let Some(sentry) = archetype.sentry {
        let sentry_arch = FireArch {
            sentry: Some(sentry),
            ..FireArch::default()
        };
        spawn_player_projectile_with_source(
            commands,
            muzzle,
            shot.aim.normalize_or_zero(),
            260.0,
            0,
            0.9,
            6.0,
            0.0,
            false,
            def.color,
            Vec2::splat(10.0),
            0,
            0,
            None,
            None,
            sentry_arch,
            Some(DamageSource::player_weapon(shot.player_ent, id)),
            Some(id),
        );
        return;
    }

    let mut rng = rand::rng();
    let spread = def.spread * player.spread_mult * player.accuracy;

    let pierce = if archetype.chain.is_some() {
        0
    } else {
        def.pierce
    };
    // GML grenade shotguns roll speed 10-15 px/frame per pellet
    // (SmallGrenade); Death-crown pellets resolved at the `fire_one_gun`
    // gate above ride `def.pellets` here.
    for _ in 0..def.pellets {
        let base_angle = shot.aim.y.atan2(shot.aim.x);
        let angle = base_angle + rng.random_range(-spread..spread);
        let dir = Vec2::new(angle.cos(), angle.sin());
        let speed = if def.ammo == AmmoKind::Shells {
            rng.random_range(360.0..540.0)
        } else {
            def.speed
        };
        spawn_player_projectile_with_source(
            commands,
            muzzle,
            dir,
            speed,
            def.damage,
            def.lifetime,
            def.projectile_radius,
            def.knockback * player.knockback_mult,
            def.explosive,
            def.color,
            def.size,
            def.bounces,
            pierce,
            def.hazard,
            def.split,
            archetype,
            Some(DamageSource::player_weapon(shot.player_ent, id)),
            Some(id),
        );
    }
    // Shell casings are render juice (GroundPhysics sprites); omitted.
}

/// GML slash life = anim length at image_speed 0.4 (12 anim-fps).
/// With the victim nexthurt+5 gate (hits land on ticks 0, 5, 10, ...),
/// life under 10 ticks caps one swing at 2 hits on the same enemy.
pub fn slash_life_secs(frames: u32) -> f32 {
    (frames.max(1) as f32 / 12.0).clamp(0.09, 0.6)
}

/// GML scrFire melee: `wepangle *= -1` on every swing, so the held weapon
/// alternates sides each click (fresh roll only if it was somehow 0).
pub fn flip_melee_angle(wep_angle: f32) -> f32 {
    if wep_angle == 0.0 {
        if rand::rng().random_bool(0.5) {
            120.0
        } else {
            -120.0
        }
    } else {
        -wep_angle
    }
}

#[allow(clippy::too_many_arguments)]
fn melee_attack(
    commands: &mut Commands,
    fx: &mut FireFx,
    shot: &GunShot,
    player: &mut Player,
    health: &mut Health,
    vel: &mut Velocity,
    melee: MeleeDef,
    vis_q: &mut Query<&mut WeaponVisual>,
) {
    let _ = melee;
    fx.trauma.add(shot.def.shake.max(0.12) * fx.shake_scale);
    cue(fx.cues, "melee", 0.7, 0.05);
    for mut wv in vis_q.iter_mut() {
        if wv.owner == shot.player_ent && wv.slot == shot.visual_slot {
            wv.wkick = gml_melee_wkick(shot.def.name);
            wv.wep_angle = flip_melee_angle(wv.wep_angle);
        }
    }

    vel.0 -= shot.aim.normalize_or_zero() * gml_fire_push_px_s(shot.def.name);
    player.melee_flip = !player.melee_flip;

    let mega = shot.def.name == "BLACK SWORD" && (health.hp <= 0 || health.max <= 0);
    if mega {
        fx.hitstop.trigger(0.35, 0.08);
    }
    let spec = melee_projectile_spec(shot.def.name);
    let dmg = if mega {
        80
    } else {
        spec.damage_override.unwrap_or(shot.def.damage)
    };
    let longarms = player
        .mutations
        .iter()
        .filter(|m| **m == MutationId::LongArms)
        .count() as f32;
    let player_pos = shot.pos;
    let aim_angle = shot.aim.y.atan2(shot.aim.x);

    // Bevy reads the fired strip (`anim_opt.def.frames`, mega sprite
    // when applicable), defaulting to 3 — never a per-kind constant.
    let slash_path = if mega {
        spec.mega_sprite.unwrap_or(spec.sprite)
    } else {
        spec.sprite
    };
    let slash_frames = fx.catalog.def(slash_path).map(|d| d.frames).unwrap_or(3);
    let life_secs = slash_life_secs(slash_frames);
    for i in 0..spec.pellets.max(1) {
        let shift = if spec.pellets > 1 {
            (i as f32 - (spec.pellets as f32 - 1.0) * 0.5) * spec.shift_deg
        } else {
            0.0
        };
        let ang = aim_angle + shift.to_radians() * player.accuracy.max(0.2);
        let dir = Vec2::new(ang.cos(), ang.sin());
        let spawn_pos = player_pos + dir * (longarms * 20.0);
        let speed = (spec.speed_f + longarms * 3.0) * 30.0;
        // GML mask hitbox (mskSlash 64x48 / mskMegaSlash 96x72); Shank
        // uses its own sprite bbox. Values are bevy-verbatim.
        let (slash_reach, slash_back, slash_half) = if mega {
            (71.0, 24.0, 36.0)
        } else if spec.shank {
            (36.0, -6.0, 5.0)
        } else {
            (47.0, 16.0, 24.0)
        };
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            Team::Player,
            Projectile {
                damage: dmg,
                life: GTimer::from_seconds(life_secs, TimerMode::Once),
                radius: 12.0,
                knockback: shot.def.knockback,
                explosive: false,
                source: Some(DamageSource::player_weapon(shot.player_ent, WeaponId::NONE)),
            },
            Velocity(dir * speed),
            ProjectileFriction(0.1),
            PiercesLeft(255),
            ProjectileHitSet::default(),
            SlashProjectile {
                typ: 0,
                shank: spec.shank,
                walled: false,
                hit: false,
                guitar: spec.guitar,
                electric_guitar: spec.electric_guitar,
                blood: spec.blood,
                lightning: spec.lightning,
                hammer_wallbreak: spec.hammer_wallbreak,
                reach: slash_reach,
                back: slash_back,
                half_width: slash_half,
                dir,
            },
            // Art (slash strip + anim) resolves renderer-side from
            // `SlashProjectile`; the hitbox above is the sim truth.
            Pos(spawn_pos),
        ));
    }
    rumble(fx.rumble, 0.3, 0.5, 0.15);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AmmoPayment {
    Paid,
    Failed,
}

pub fn pay_fire_cost(
    inv: &mut Inventory,
    _health: &mut Health,
    ammo: AmmoKind,
    amount: i32,
    _blood: Option<BloodAmmo>,
    free: bool,
) -> AmmoPayment {
    if amount <= 0 || free {
        return AmmoPayment::Paid;
    }

    let slot = inv.ammo_mut(ammo);
    if *slot >= amount {
        *slot -= amount;
        return AmmoPayment::Paid;
    }

    AmmoPayment::Failed
}

pub fn spawn_beam_shot(
    commands: &mut Commands,
    pos: Vec2,
    dir: Vec2,
    spec: BeamShot,
    source: Option<DamageSource>,
) {
    let dir = dir.normalize_or_zero();
    let center = pos + dir * (spec.length * 0.5);
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        Team::Player,
        crate::comps_b::Beam {
            team: Team::Player,
            dir,
            length: spec.length,
            width: spec.width,
            damage: spec.damage,
            knockback: spec.knockback,
            color: spec.color,
            timer: GTimer::from_seconds(spec.duration, TimerMode::Once),
            tick: GTimer::from_seconds(spec.tick, TimerMode::Repeating),
            source,
        },
        // Orientation rides `Beam.dir`; the renderer sizes from
        // length/width (no Transform rotation write headless).
        Pos(center),
    ));
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_player_projectile(
    commands: &mut Commands,
    pos: Vec2,
    dir: Vec2,
    speed: f32,
    damage: i32,
    lifetime: f32,
    radius: f32,
    knockback: f32,
    explosive: bool,
    color: [f32; 3],
    size: Vec2,
) {
    spawn_player_projectile_with_source(
        commands,
        pos,
        dir,
        speed,
        damage,
        lifetime,
        radius,
        knockback,
        explosive,
        color,
        size,
        0,
        0,
        None,
        None,
        FireArch::default(),
        None,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_player_projectile_with_source(
    commands: &mut Commands,
    pos: Vec2,
    dir: Vec2,
    speed: f32,
    damage: i32,
    lifetime: f32,
    radius: f32,
    knockback: f32,
    explosive: bool,
    color: [f32; 3],
    size: Vec2,
    bounces: u8,
    pierce: u8,
    hazard: Option<HazardDef>,
    split: Option<SplitDef>,
    archetype: FireArch,
    source: Option<DamageSource>,
    weapon: Option<WeaponId>,
) {
    let _ = (color, size);
    // GML shell stats follow the PROJECTILE object, not the ammo type:
    // pop gun fires Bullet2, sluggers fire Slug variants, ultra shotgun
    // fires UltraShell, flak fires FlakBullet (no bounce, bonus 2).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum ShellKind {
        Bullet2,
        Slug,
        HeavySlug,
        HyperSlug,
        UltraShell,
        FlameShell,
        Flak,
    }
    let ammo_of_weapon = weapon.map(weapon_ammo).unwrap_or(AmmoKind::None);
    let shell_kind: Option<ShellKind> = weapon.and_then(|w| {
        let full = weapon_meta(w).wep_name;
        let base = base_weapon_name(full);
        if split.is_some() && ammo_of_weapon == AmmoKind::Shells {
            return Some(ShellKind::Flak);
        }
        if base.contains("HYPER SLUGGER") {
            return Some(ShellKind::HyperSlug);
        }
        if base.contains("HEAVY SLUGGER") {
            return Some(ShellKind::HeavySlug);
        }
        if base.contains("SLUGGER") {
            return Some(ShellKind::Slug);
        }
        if full == "ULTRA SHOTGUN" {
            return Some(ShellKind::UltraShell);
        }
        if base.contains("FLAME SHOTGUN") {
            return Some(ShellKind::FlameShell);
        }
        if base.contains("POP GUN") {
            return Some(ShellKind::Bullet2);
        }
        if ammo_of_weapon == AmmoKind::Shells {
            return Some(ShellKind::Bullet2);
        }
        None
    });
    // GML shotgun pellets (Bullet2/Slug/UltraShell) have no lifetime: they
    // fly until friction slows them under speed 6, then fade out.
    let lifetime = if shell_kind.is_some_and(|k| k != ShellKind::Flak) {
        lifetime.max(4.0)
    } else {
        lifetime
    };
    let mut ec = commands.spawn((
        GameCleanup,
        LevelCleanup,
        Team::Player,
        Projectile {
            damage,
            life: GTimer::from_seconds(lifetime, TimerMode::Once),
            radius,
            knockback,
            explosive,
            source,
        },
        Velocity(dir * speed),
        // Art resolves renderer-side from `ProjectileTyp`/fade keys;
        // no catalog strip is attached headless.
        Pos(pos),
    ));

    #[derive(Clone, Copy)]
    struct ShellStats {
        friction: f32,
        fade: Option<&'static str>,
        base: f32,
        shoulders: f32,
        cap: f32,
        decay: f32,
        bonus: i32,
        rearm: Option<(f32, i32)>,
        flak: bool,
    }
    let shell_stats: Option<ShellStats> = match shell_kind {
        None => None,
        // GML FlakBullet: friction 0.4, pointblank bonus 2, never bounces.
        // Super Flak carries its own fade sprite.
        Some(ShellKind::Flak) => Some(ShellStats {
            friction: 0.4,
            fade: weapon
                .map(|w| weapon_meta(w).wep_name.contains("SUPER FLAK"))
                .unwrap_or(false)
                .then_some("images/sprSuperFlakHit.png"),
            base: 0.0,
            shoulders: 0.0,
            cap: 480.0,
            decay: 0.95,
            bonus: 2,
            rearm: None,
            flak: true,
        }),
        Some(ShellKind::Bullet2) => Some(ShellStats {
            friction: 0.6,
            fade: Some("images/sprBullet2Disappear.png"),
            base: 0.0,
            shoulders: 5.0,
            cap: 480.0,
            decay: 0.95,
            bonus: 1,
            rearm: Some((0.0, 1)),
            flak: false,
        }),
        Some(ShellKind::Slug) => Some(ShellStats {
            friction: 0.8,
            fade: Some("images/sprSlugDisappear.png"),
            base: 0.0,
            shoulders: 4.0,
            cap: 540.0,
            decay: 0.9,
            bonus: 2,
            rearm: None,
            flak: false,
        }),
        Some(ShellKind::HeavySlug) => Some(ShellStats {
            friction: 1.0,
            fade: Some("images/sprHeavySlugDisappear.png"),
            base: 2.0,
            shoulders: 6.0,
            cap: 480.0,
            decay: 0.95,
            // GML HeavySlug starts with bonus 0; wallbounce > 2 re-arms 10.
            bonus: 0,
            rearm: Some((2.0, 10)),
            flak: false,
        }),
        Some(ShellKind::HyperSlug) => Some(ShellStats {
            friction: 0.8,
            fade: Some("images/sprSlugDisappear.png"),
            base: 0.0,
            shoulders: 5.0,
            cap: 480.0,
            decay: 0.95,
            bonus: 2,
            rearm: Some((0.0, 2)),
            flak: false,
        }),
        Some(ShellKind::UltraShell) => Some(ShellStats {
            friction: 0.3,
            fade: Some("images/sprUltraShellDisappear.png"),
            base: 0.0,
            shoulders: 0.0,
            cap: 480.0,
            decay: 0.95,
            bonus: 2,
            // Inherits Bullet2's wall event (re-arm while wallbounce > 0).
            rearm: Some((0.0, 2)),
            flak: false,
        }),
        Some(ShellKind::FlameShell) => Some(ShellStats {
            friction: 0.6,
            // GML FlameShell/Step_0 destroys outright under speed 5 with no
            // fade anim; see FlameShellSlowDeath.
            fade: None,
            base: 0.0,
            shoulders: 1.0,
            cap: 480.0,
            decay: 0.95,
            bonus: 1,
            rearm: Some((0.0, 1)),
            flak: false,
        }),
    };

    if shell_kind == Some(ShellKind::FlameShell) {
        ec.insert(FlameShellSlowDeath);
    }
    if bounces > 0 {
        ec.insert(BouncesLeft(bounces));
        if let Some(st) = shell_stats
            && !st.flak
        {
            ec.insert(ShellWallBounce {
                add: st.shoulders,
                cap: st.cap,
                decay: st.decay,
                rearm: st.rearm,
            });
        }
    } else if let Some(w) = weapon {
        if shell_kind == Some(ShellKind::Flak) {
            // GML FlakBullet never bounces (dies on wall + splits).
        } else if let Some(st) = shell_stats {
            ec.insert(BouncesLeft(255));
            ec.insert(ShellWallBounce {
                add: st.base,
                cap: st.cap,
                decay: st.decay,
                rearm: st.rearm,
            });
        }
        let base = base_weapon_name(weapon_meta(w).wep_name);
        if base.contains("BOUNCER") {
            ec.insert(BouncesLeft(1));
        }
        if base.contains("DISC") {
            ec.insert(BouncesLeft(255));
            ec.insert(DiscFlight { dist: 0.0, home: pos });
        }
    }
    if let Some(w) = weapon {
        // Bevy gates on ids 7/44 (plain + golden launcher only):
        // grenade shotguns/rifles/ultras keep their shell behavior.
        let full = weapon_meta(w).wep_name;
        if full == "GRENADE LAUNCHER" || full == "GOLDEN GRENADE LAUNCHER" {
            ec.insert(ProjectileFriction(0.1));
            ec.insert(GrenadeFuse {
                smoke_armed: false,
                friction_switched: false,
                alarm1: GTimer::from_seconds(6.0 / 30.0, TimerMode::Once),
            });
        } else if let Some(st) = shell_stats {
            ec.insert(ProjectileFriction(st.friction));
            ec.insert(ShellBonus {
                timer: GTimer::from_seconds(2.0 / 30.0, TimerMode::Once),
                bonus: st.bonus,
            });
        }
    }
    if pierce > 0 || archetype.chain.is_some() {
        ec.insert(PiercesLeft(pierce));
        ec.insert(ProjectileHitSet::default());
    }
    if let Some(spec) = hazard {
        ec.insert(SpawnHazardOnDeath(spec));
        if spec.kind == HazardKind::Fire {
            ec.insert(FlameTrail {
                timer: GTimer::from_seconds(0.12, TimerMode::Repeating),
                spec,
            });
        }
    }
    if let Some(spec) = split {
        ec.insert(SplitOnDeath(spec));
    }
    if let Some(homing) = archetype.homing {
        ec.insert(homing);
    }
    if let Some(sticky) = archetype.sticky {
        ec.insert(sticky);
        if let Some(w) = weapon
            && weapon_ammo(w) == AmmoKind::Bolts
            && pierce == 0
            && archetype.chain.is_none()
        {
            ec.insert(PiercesLeft(3));
            ec.insert(ProjectileHitSet::default());
        }
    }
    if let Some(chain) = archetype.chain {
        ec.remove::<PiercesLeft>();
        ec.insert(chain);
    }
    if let Some(sentry) = archetype.sentry {
        ec.insert(sentry);
    }
    if let Some(custom) = archetype.custom {
        ec.insert(custom);
    }
    if let Some(blood) = archetype.blood {
        ec.insert(blood);
    }
    if let Some(pickup) = archetype.pickup {
        ec.insert(pickup);
    }
    if let Some(plasma) = archetype.plasma {
        ec.insert(plasma);
        ec.insert(crate::comps_a::PlasmaSize(1.0));
    }
    if archetype.hits_all {
        ec.insert(HitsAllTeams);

        ec.insert(SpawnGrace(GTimer::from_seconds(2.0 / 30.0, TimerMode::Once)));
    }

    if let Some(w) = weapon {
        let ammo = weapon_ammo(w);
        let typ = match ammo {
            AmmoKind::Bolts | AmmoKind::Energy => 2,
            _ => 1,
        };
        ec.insert(ProjectileTyp(typ));
    }

    // Exact GML projectile object art. `ProjectileTyp` is not enough:
    // Bullet1.typ == 1 and Bullet2.typ == 1 in the GameMaker source.
    let projectile_visual: Option<ProjectileVisual> = (|| {
        let w = weapon?;
        let ammo = weapon_ammo(w);
        let base = base_weapon_name(weapon_meta(w).wep_name);

        if shell_kind == Some(ShellKind::Bullet2) {
            return Some(ProjectileVisual {
                sprite: "images/sprBullet2.png",
                mask: Some("images/mskBullet2.png"),
                fade: Some("images/sprBullet2Disappear.png"),
            });
        }

        if ammo == AmmoKind::Bullets
            && !base.contains("DISC")
            && !base.contains("BOUNCER")
        {
            return Some(ProjectileVisual {
                sprite: "images/sprBullet1.png",
                mask: Some("images/mskBullet1.png"),
                fade: Some("images/sprBulletHit.png"),
            });
        }

        None
    })();

    if let Some(v) = projectile_visual {
        ec.insert(v);
    }

    let fade: Option<ProjectileFade> = (|| {
        if let Some(v) = projectile_visual {
            return v.fade.map(ProjectileFade);
        }
        if let Some(st) = shell_stats {
            return st.fade.map(ProjectileFade);
        }
        let w = weapon?;
        let ammo = weapon_ammo(w);
        if ammo != AmmoKind::Bullets {
            return None;
        }
        let base = base_weapon_name(weapon_meta(w).wep_name);
        if base.contains("DISC") {
            return None;
        }
        Some(ProjectileFade("images/sprBulletHit.png"))
    })();
    if let Some(f) = fade {
        ec.insert(f);
    }
    // Juice::shake on explosives is render juice; omitted.
}

// ---------------------------------------------------------------------------
// hammerhead_chew
// ---------------------------------------------------------------------------

/// Hammerhead wall/prop chewing while sprinting. Prop corpse art and
/// death-effect particles are render-side; the sim keeps hp damage,
/// explosive blasts, secret-entrance reveals, and the wall-break budget.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn hammerhead_chew(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cooldown: Local<f32>,
    mut budget: ResMut<HammerheadBudget>,
    catalog: Res<repame_anim::AnimCatalog>,
    player_q: Query<(Entity, &Pos, &Player, &Velocity), With<Player>>,
    mut props: Query<
        (
            Entity,
            &mut Prop,
            &Pos,
            Option<&PropDeathEffect>,
            Option<&PropSprites>,
        ),
        (With<Prop>, Without<WallTile>),
    >,
    walls: Query<(Entity, &WallCell, &Pos), With<WallTile>>,
    entrances: Query<&SecretEntrance>,
    mut secrets: ResMut<SecretTriggers>,
) {
    *cooldown -= time.delta_secs;
    if *cooldown > 0.0 {
        return;
    }

    let Ok((player_entity, player_pos, player, vel)) = player_q.single() else {
        return;
    };
    if !player.hammerhead {
        return;
    }
    if vel.0.length_squared() < 40.0 * 40.0 {
        return;
    }

    let pos = player_pos.0;
    let push = vel.0.normalize_or_zero();
    let probe = pos + push * (PLAYER_RADIUS + 6.0);

    if budget.remaining > 0 {
        for (_, cell, wpos) in &walls {
            let wpos = wpos.0;
            if wpos.distance(probe) > WALL_PX * 0.85 {
                continue;
            }

            if (wpos - pos).dot(push) < 0.0 {
                continue;
            }
            budget.remaining -= 1;
            *cooldown = 0.08;
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                PendingWallBreak {
                    cell: (cell.0, cell.1),
                    pos: wpos,
                    spawn_floor: true,
                },
            ));
            return;
        }
    }

    for (prop_e, mut prop, prop_pos, death_effect, sprites) in &mut props {
        if !prop.destructible {
            continue;
        }
        let center = prop_pos.0;
        let half = prop.size / 2.0;
        let closest = Vec2::new(
            pos.x.clamp(center.x - half.x, center.x + half.x),
            pos.y.clamp(center.y - half.y, center.y + half.y),
        );
        if pos.distance(closest) > PLAYER_RADIUS + 6.0 {
            continue;
        }

        *cooldown = 0.25;
        prop.hp -= 1;
        if prop.hp <= 0 {
            if let Some(ps) = sprites.copied() {
                spawn_prop_corpse(&mut commands, &catalog, center, &ps);
            }
            spawn_prop_death_effect(
                &mut commands,
                center,
                death_effect.copied(),
                prop.explosive,
                Some(DamageSource {
                    owner: player_entity,
                    team: Team::Player,
                    hit_id: HitId::Other(301),
                    enemy_kind: None,
                }),
            );
            if let Ok(entrance) = entrances.get(prop_e) {
                secrets.queue(entrance.target);
            }
            commands.entity(prop_e).try_despawn();
        }
        return;
    }
}

// ---------------------------------------------------------------------------
// move_swing_fx (lifetime tick only)
// ---------------------------------------------------------------------------

/// Lifetime tick for swing markers. Slash hitbox motion itself is
/// `Velocity` integration in `move_projectiles`; this only retires the
/// marker so it cannot leak.
pub fn move_swing_fx(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut SwingFx)>,
) {
    for (e, mut fx) in &mut q {
        fx.timer.tick(time.delta_secs);
        if fx.timer.just_finished() {
            commands.entity(e).despawn();
        }
    }
}

/// GML Cuz `spr_cry` swap lifetime: retire the marker headlessly.
pub fn tick_cry_anim(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut CryAnim)>,
) {
    for (e, mut cry) in &mut q {
        cry.timer.tick(time.delta_secs);
        if cry.timer.just_finished() {
            commands.entity(e).remove::<CryAnim>();
        }
    }
}

// ---------------------------------------------------------------------------
// Ability-field ticks
// ---------------------------------------------------------------------------

#[allow(clippy::type_complexity)]
pub fn tick_snare_zones(
    time: Res<SimTime>,
    mut commands: Commands,
    player_q: Query<&Player, With<Player>>,
    mut zones: Query<(Entity, &Pos, &mut SnareZone)>,
    mut enemies: Query<(Entity, &Pos, &mut Health), (With<Enemy>, Without<Slowed>)>,
) {
    let throne_butt = player_q.single().map(|p| p.throne_butt).unwrap_or(false);
    for (e, zpos, mut zone) in &mut zones {
        zone.timer.tick(time.delta_secs);
        if zone.timer.just_finished() {
            commands.entity(e).despawn();
            continue;
        }
        let z = zpos.0;
        for (ee, epos, mut health) in &mut enemies {
            if epos.0.distance(z) <= zone.radius {
                if throne_butt && health.hp <= (health.max / 3).max(1) && health.hp > 0 {
                    health.hp = 0;
                }
                commands.entity(ee).insert(Slowed {
                    timer: GTimer::from_seconds(0.4, TimerMode::Once),
                    factor: if throne_butt { 0.02 } else { zone.slow },
                });
            }
        }
    }
}

pub fn tick_slowed(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut Slowed, &mut Velocity), With<Enemy>>,
) {
    for (e, mut s, mut vel) in &mut q {
        s.timer.tick(time.delta_secs);
        vel.0 *= s.factor;
        if s.timer.just_finished() {
            commands.entity(e).remove::<Slowed>();
        }
    }
}

pub fn tick_portal_strikes(
    time: Res<SimTime>,
    mut commands: Commands,
    mut trauma: ResMut<Trauma>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut q: Query<(Entity, &Pos, &mut PortalStrike)>,
    mut enemies: Query<(&Pos, &mut Health), With<Enemy>>,
) {
    for (e, spos, mut strike) in &mut q {
        strike.timer.tick(time.delta_secs);
        if !strike.timer.just_finished() {
            continue;
        }
        let pos = spos.0;
        for (epos, mut h) in &mut enemies {
            if epos.0.distance(pos) <= strike.radius {
                h.hp -= strike.damage;
            }
        }
        trauma.add(0.4);
        cue(&mut cues, "sndExplosionL", 0.9, 0.04);
        let mut rng = rand::rng();
        spawn_burst(
            &mut commands,
            &mut rng,
            pos,
            28,
            [0.3, 0.9, 1.0, 1.0],
            (120.0, 360.0),
        );
        commands.entity(e).despawn();
    }
}

#[allow(clippy::type_complexity)]
pub fn tick_hazard_clouds(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &Pos, &mut HazardCloud), (With<AbilityHazard>, Without<Team>)>,
    mut enemies: Query<(&Pos, &mut Health), With<Enemy>>,
) {
    for (e, cpos, mut cloud) in &mut q {
        cloud.timer.tick(time.delta_secs);
        cloud.tick.tick(time.delta_secs);
        if cloud.timer.just_finished() {
            commands.entity(e).despawn();
            continue;
        }
        if !cloud.tick.just_finished() {
            continue;
        }
        let pos = cpos.0;
        for (epos, mut h) in &mut enemies {
            if epos.0.distance(pos) <= cloud.radius {
                h.hp -= cloud.damage;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cuz emotional helpers + Robot eat shared drops
// ---------------------------------------------------------------------------

/// GML `scr_ultra_get(Race.Cuz, UltraSkill.Emotional)` level (0/1).
pub fn cuz_emotional_level(ultra: Option<UltraMutationId>) -> u32 {
    u32::from(ultra == Some(UltraMutationId::CuzEmotional))
}

/// GML `scrPlayerUpdateCuzAmmo`:
/// `cuz_ammo_max = 3*(back_muscle+1)*(Emotional+1)`.
pub fn cuz_ammo_max_for(back_muscle: u32, emotional: u32) -> u8 {
    (3 * (back_muscle + 1) * (emotional + 1)).min(u8::MAX as u32) as u8
}

/// Refresh `cuz_ammo_max` after back-muscle/ultra changes (GML
/// `scrPlayerUpdateCuzAmmo`, called from `scrUltras` + skills).
pub fn refresh_cuz_ammo_max(player: &mut Player) {
    player.cuz_ammo_max = cuz_ammo_max_for(player.back_muscle, cuz_emotional_level(player.ultra));
}

/// GML `scrRobotEat` core (`scrPowers.gml:621-662`), shared by the
/// active and the portal auto-collect path: golden weapon →
/// `repeat(4+throne_butt)` HP-or-ammo (`random(max_hp)>hp` and not
/// life-crown → HP else ammo); Regurgitate 43% →
/// love-crown ? AmmoChest : hurt && `random(3)<2` ? HealthChest :
/// `choose(WeaponChest,AmmoChest,AmmoChest)`; then `repeat(1+throne_butt)`
/// HP/ammo by the same hurt rule. With `auto_collect`, HP/ammo apply
/// directly (GML `event_perform(ev_collision, Player)`); chests stay on
/// the ground. (Rads/curse handling lives in the active arm — GML keeps
/// it in `scrPowers`, not `scrRobotEat`.)
pub fn robot_eat_drops(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: Vec2,
    player: &Player,
    health: &mut Health,
    inv: &mut Inventory,
    weapon: WeaponId,
    auto_collect: bool,
) {
    let tb = u32::from(player.throne_butt);
    let life_crown = player.crown == CrownKind::Life;
    let ammo_cap = |player: &Player, kind: AmmoKind| player.ammo_cap(kind);
    let medkit_mult = player.medkit_mult;
    let mut rng = rand::rng();

    // One HP-or-ammo drop; `hp_amount` is the medkit size when HP wins.
    let drop_hp_or_ammo = |commands: &mut Commands,
                               pos: Vec2,
                               player: &Player,
                               health: &mut Health,
                               inv: &mut Inventory,
                               rng: &mut rand::rngs::ThreadRng,
                               hp_amount: i32| {
        let wants_hp = !life_crown && rng.random_range(0..health.max.max(1)) as i32 > health.hp;
        if wants_hp {
            if auto_collect {
                let heal = (hp_amount as f32 * medkit_mult).round() as i32;
                health.hp = (health.hp + heal).min(health.max);
            } else {
                let off = Vec2::new(
                    rng.random_range(-12.0..12.0),
                    rng.random_range(-12.0..12.0),
                );
                spawn_pickup(
                    commands,
                    catalog,
                    PickupKind::Medkit(hp_amount),
                    pos + off,
                    0,
                    false,
                );
            }
        } else {
            let kind = match rng.random_range(0..5) {
                0 => AmmoKind::Bullets,
                1 => AmmoKind::Shells,
                2 => AmmoKind::Bolts,
                3 => AmmoKind::Explosives,
                _ => AmmoKind::Energy,
            };
            if auto_collect {
                let cap = ammo_cap(player, kind);
                let slot = inv.ammo_mut(kind);
                *slot = (*slot + ammo_pickup_amount(kind)).min(cap);
            } else {
                let off = Vec2::new(
                    rng.random_range(-12.0..12.0),
                    rng.random_range(-12.0..12.0),
                );
                spawn_pickup(
                    commands,
                    catalog,
                    PickupKind::Ammo(kind, ammo_pickup_amount(kind)),
                    pos + off,
                    0,
                    false,
                );
            }
        }
    };

    if weapon_meta(weapon).wep_gold {
        for _ in 0..(4 + tb) {
            drop_hp_or_ammo(commands, pos, player, health, inv, &mut rng, 2);
        }
    }
    if matches!(player.ultra, Some(UltraMutationId::RobotRegurgitate))
        && rng.random::<f32>() <= 0.43
    {
        // GML Regurgitate roll (no life-crown gate on this branch).
        if player.crown == CrownKind::Love {
            spawn_chest(
                commands,
                catalog,
                ChestKind::Ammo,
                pos + Vec2::new(16.0, 0.0),
            );
        } else if rng.random_range(0..health.max.max(1)) as i32 > health.hp
            && rng.random_range(0..3) < 2
        {
            spawn_chest(
                commands,
                catalog,
                ChestKind::Health,
                pos + Vec2::new(16.0, 0.0),
            );
        } else {
            let kind = match rng.random_range(0..3) {
                0 => ChestKind::Weapon,
                _ => ChestKind::Ammo,
            };
            spawn_chest(commands, catalog, kind, pos + Vec2::new(16.0, 0.0));
        }
    }
    for _ in 0..(1 + tb) {
        drop_hp_or_ammo(commands, pos, player, health, inv, &mut rng, 2);
    }
}

// ---------------------------------------------------------------------------
// player_ability
// ---------------------------------------------------------------------------

/// One-shot racial abilities. Visual bursts are omitted; every sim effect
/// (timers, damage, spawns, ammo/hp costs, unlocks) is kept. Steroids has
/// no tap ability (bevy early-return parity).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn player_ability(
    mut input: ResMut<NtInput>,
    mut commands: Commands,
    mut trauma: ResMut<Trauma>,
    mut chroma: ResMut<ChromaticAberration>,
    mut slow_mo: ResMut<SlowMotion>,
    mut hitstop: ResMut<HitStop>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut rumble_q: ResMut<Queue<RumbleRequest>>,
    mut toast: ResMut<Toast>,
    mut save: ResMut<SaveData>,
    mut dirty: ResMut<SaveDirty>,
    catalog: Res<repame_anim::AnimCatalog>,
    mut tut: Option<ResMut<crate::state::TutorialState>>,
    mut player_q: Query<
        (
            Entity,
            &Pos,
            &mut Player,
            &mut Health,
            &mut Velocity,
            &AimDir,
            &mut Inventory,
            &RaceState,
            Option<&mut Shield>,
            Option<&mut Telekinesis>,
        ),
        (With<Player>, Without<Enemy>),
    >,
    mut enemies: Query<(Entity, &Pos, &mut Health), (With<Enemy>, Without<Player>)>,
) {
    let Ok((
        player_e,
        ppos,
        mut player,
        mut health,
        mut vel,
        aim,
        mut inv,
        race_state,
        shield,
        telek,
    )) = player_q.single_mut()
    else {
        return;
    };

    let fire = input.take_ability_pressed();
    if race_state.race == RaceId::Steroids {
        return;
    }
    if !fire {
        return;
    }

    // GML `scrPowers:460` verbatim: any spec press latches the tutorial
    // Power step, regardless of the race effect below.
    if let Some(tut) = tut.as_deref_mut() {
        tut.complete_step(crate::state::TutorialStep::Power);
    }

    let pos = ppos.0;
    let aim_v = aim.0;
    let ability = player.ability;

    let ability_mult = if player.throne_butt {
        player.ultra_ability_mult * 1.35
    } else {
        player.ultra_ability_mult
    };

    match ability {
        AbilityKind::Flip => {
            let dir = if input.move_axis != Vec2::ZERO {
                input.move_axis.normalize()
            } else {
                aim_v
            };
            commands.entity(player_e).insert(Dash {
                timer: GTimer::from_seconds(0.18 * ability_mult.clamp(1.0, 1.6), TimerMode::Once),
                dir,
            });
            health.invuln = GTimer::from_seconds(15.0 / 30.0, TimerMode::Once);
            vel.0 = dir * 900.0;
            trauma.add(0.12);
            slow_motion(&mut slow_mo, 0.55, 0.2);
            rumble(&mut rumble_q, 0.2, 0.2, 0.1);
            cue(&mut cues, "bolt", 0.7, 0.05);
        }
        AbilityKind::Shield => {
            let timer =
                GTimer::from_seconds(1.6 * ability_mult.clamp(1.0, 2.0), TimerMode::Once);
            if let Some(mut s) = shield {
                s.timer = timer;
            } else {
                commands.entity(player_e).insert(Shield { timer });
            }
            trauma.add(0.08);
            cue(&mut cues, "sndAmmoPickup", 0.5, 0.15);
        }
        AbilityKind::Telekinesis => {
            let timer = GTimer::from_seconds(1.4, TimerMode::Once);
            if let Some(mut t) = telek {
                t.timer = timer;
            } else {
                commands.entity(player_e).insert(Telekinesis { timer });
            }
            cue(&mut cues, "sndPortalOpen", 0.7, 0.05);
        }
        AbilityKind::Detonate => {
            if health.hp <= 1 {
                return;
            }
            health.hp -= 1;
            let radius = 150.0 * ability_mult.clamp(1.0, 2.0);
            let damage = (3.0 * ability_mult).round() as i32;
            for (_, epos, mut ehealth) in &mut enemies {
                if epos.0.distance(pos) < radius {
                    ehealth.hp -= damage;
                }
            }
            trauma.add(0.5);
            chromatic_pulse(&mut chroma, 0.4);
            rumble(&mut rumble_q, 0.6, 0.8, 0.25);
            hitstop.trigger(0.25, 0.12);
            cue(&mut cues, "sndExplosionL", 0.9, 0.04);
        }
        AbilityKind::Snare => {
            commands.spawn((
                LevelCleanup,
                SnareZone {
                    timer: GTimer::from_seconds(2.5 * ability_mult.clamp(1.0, 2.0), TimerMode::Once),
                    radius: 110.0 * ability_mult.clamp(1.0, 1.8),
                    slow: (0.35 / ability_mult).clamp(0.12, 0.35),
                },
                Pos(pos + aim_v * 70.0),
            ));
            cue(&mut cues, "sndAmmoPickup", 0.5, 0.15);
        }
        AbilityKind::PopPop => {
            let charges = if player.throne_butt
                || matches!(player.ultra, Some(UltraMutationId::VenuzBack2Bizniz))
            {
                2
            } else {
                1
            };
            commands.entity(player_e).insert(PopPopCharges(charges));
            cue(&mut cues, "bolt", 0.7, 0.05);
        }
        AbilityKind::GetLoaded => {
            for slot in 0..inv.weapon_slots {
                let w = inv.weapons[slot];
                if w == WeaponId::NONE {
                    continue;
                }
                let kind = weapon_ammo(w);
                if kind == AmmoKind::None {
                    continue;
                }
                let add = match kind {
                    AmmoKind::Bullets => 32,
                    AmmoKind::Shells => 8,
                    AmmoKind::Bolts => 6,
                    AmmoKind::Explosives => 4,
                    AmmoKind::Energy => 10,
                    AmmoKind::None => 0,
                };
                let add = ((add as f32) * ability_mult).round() as i32;
                let slot = inv.ammo_mut(kind);
                *slot = (*slot + add).min(player.ammo_cap(kind));
            }
            cue(&mut cues, "sndAmmoPickup", 0.5, 0.15);
        }
        AbilityKind::EatWeapon => {
            // GML Robot (`scrPowers.gml` Robot case + `scrRobotEat`).
            let slot = inv.current;
            let w = inv.weapons[slot];
            if w == WeaponId::NONE {
                return;
            }
            let meta = weapon_meta(w);
            let wep_rads = meta.wep_rads;
            let was_cursed = inv.cursed[slot];
            inv.weapons[slot] = WeaponId::NONE;
            inv.cursed[slot] = false;
            if let Some(next) = (0..inv.weapon_slots).find(|&i| inv.weapons[i] != WeaponId::NONE) {
                inv.current = next;
            }
            robot_eat_drops(
                &mut commands,
                &catalog,
                pos,
                &mut player,
                &mut health,
                &mut inv,
                w,
                false,
            );
            if was_cursed {
                // GML curse eat: self-hit 7 + 10 `Curse` motes.
                health.hp -= 7;
                let mut rng = rand::rng();
                for _ in 0..10 {
                    let off = Vec2::new(rng.random_range(-8.0..8.0), rng.random_range(-8.0..8.0));
                    spawn_pickup(&mut commands, &catalog, PickupKind::Curse, pos + off, 0, false);
                }
            }
            if wep_rads > 0 {
                // GML `scrRadDrop(x, y, 15, false, false)`.
                spawn_rad_burst(&mut commands, &catalog, pos, 15);
            }

            let unlocked = check_progress_unlocks(&mut save, 0, 0, false, true, false);
            for race in unlocked {
                dirty.0 = true;
                toast.show(&format!(
                    "UNLOCKED {}",
                    character_def(race).name.to_ascii_uppercase()
                ));
            }
            let tb = player.throne_butt;
            cue(
                &mut cues,
                if tb { "sndRobotEatUpg" } else { "sndRobotEat" },
                0.7,
                0.05,
            );
            let mut rng = rand::rng();
            spawn_burst(
                &mut commands,
                &mut rng,
                pos,
                8,
                [0.6, 0.6, 0.6, 1.0],
                (30.0, 90.0),
            );
        }
        AbilityKind::Throw => {
            // GML Chicken (`scrPowers.gml:218-242`): fling the held gun
            // (cursed guns refuse with `sndCursedReminder`).
            let slot = inv.current;
            let held = inv.weapons[slot];
            if held == WeaponId::NONE {
                return;
            }
            if inv.cursed[slot] {
                cue(&mut cues, "sndCursedReminder", 0.6, 0.05);
                return;
            }
            let aim_angle = aim_v.y.atan2(aim_v.x);
            let mut rng = rand::rng();
            let fling =
                aim_angle + rng.random_range(-2.0f32.to_radians()..2.0f32.to_radians());
            let determination =
                matches!(player.ultra, Some(UltraMutationId::ChickenDetermination));
            spawn_flung_weapon_pickup(
                &mut commands,
                &catalog,
                held,
                pos + aim_v.normalize_or_zero() * 18.0,
                fling,
                Team::Player,
                player_e,
                if determination { 60 } else { 0 },
            );
            inv.weapons[slot] = WeaponId::NONE;
            inv.cursed[slot] = false;
            if let Some(next) = (0..inv.weapon_slots).find(|&i| inv.weapons[i] != WeaponId::NONE) {
                inv.current = next;
            }
            cue(&mut cues, "sndChickenThrow", 0.7, 0.05);
        }
        AbilityKind::SpawnAlly => {
            let has_ally = false; // TODO: query live allies for cost 2 gate.
            let cost = if has_ally || matches!(player.ultra, Some(UltraMutationId::RebelRiot)) {
                2
            } else {
                1
            };
            if health.hp <= cost {
                return;
            }
            health.hp -= cost;
            let ally_count = if matches!(player.ultra, Some(UltraMutationId::RebelRiot)) {
                2
            } else {
                1
            };
            for i in 0..ally_count {
                let side = Vec2::new(-aim_v.y, aim_v.x)
                    * ((i as f32) - (ally_count as f32 - 1.0) * 0.5)
                    * 22.0;
                let spawn_at = pos + aim_v * 28.0 + side;
                let ally_hp = 12;
                commands.spawn((
                    LevelCleanup,
                    Ally {
                        life: GTimer::from_seconds(12.0, TimerMode::Once),
                        shoot: GTimer::from_seconds(
                            if player.throne_butt {
                                5.0 / 30.0
                            } else {
                                8.0 / 30.0
                            },
                            TimerMode::Repeating,
                        ),
                    },
                    Team::Player,
                    Health {
                        hp: ally_hp,
                        max: ally_hp,
                        invuln: GTimer::from_seconds(0.5, TimerMode::Once),
                    },
                    Hitbox { radius: 10.0 },
                    Velocity(Vec2::ZERO),
                    Pos(spawn_at),
                ));
            }
            cue(&mut cues, "sndPortalOpen", 0.7, 0.05);
        }
        AbilityKind::HorrorBeam => {
            let dir = aim_v.normalize_or_zero();
            let beam_len = 320.0 * ability_mult.clamp(1.0, 1.8);
            let beam_damage = (4.0 * ability_mult).round() as i32;
            let beam_width = 22.0 * ability_mult.sqrt();
            for (_, epos, mut ehealth) in &mut enemies {
                let to = epos.0 - pos;
                let proj = to.dot(dir);
                if proj < 0.0 || proj > beam_len {
                    continue;
                }
                let lateral = (to - dir * proj).length();
                if lateral < beam_width {
                    ehealth.hp -= beam_damage;
                }
            }
            commands.spawn((
                LevelCleanup,
                AbilityHazard,
                HazardCloud {
                    kind: HazardKind::Toxic,
                    radius: 28.0,
                    damage: 1,
                    timer: GTimer::from_seconds(0.8, TimerMode::Once),
                    tick: GTimer::from_seconds(0.15, TimerMode::Repeating),
                },
                Pos(pos + dir * 160.0),
            ));
            trauma.add(0.18);
            cue(&mut cues, "bolt", 0.7, 0.05);
        }
        AbilityKind::PortalStrike => {
            if player.rogue_ammo == 0 {
                return;
            }
            player.rogue_ammo = player.rogue_ammo.saturating_sub(1);
            let target = pos + aim_v.normalize_or_zero() * 180.0;
            commands.spawn((
                LevelCleanup,
                PortalStrike {
                    timer: GTimer::from_seconds(0.55, TimerMode::Once),
                    radius: 90.0,
                    damage: 8,
                },
                Pos(target),
            ));
            cue(&mut cues, "sndPortalOpen", 0.7, 0.05);
        }
        AbilityKind::RocketBarrage => {
            let slot = inv.ammo_mut(AmmoKind::Explosives);
            if *slot < 3 {
                return;
            }
            *slot -= 3;
            let base = aim_v.normalize_or_zero();
            let rockets = if matches!(player.ultra, Some(UltraMutationId::BigDogHeavyArtillery)) {
                -3..=3
            } else {
                -2..=2
            };
            for i in rockets {
                let ang = (i as f32) * 0.12;
                let dir = Vec2::new(
                    base.x * ang.cos() - base.y * ang.sin(),
                    base.x * ang.sin() + base.y * ang.cos(),
                )
                .normalize_or_zero();
                commands.spawn((
                    LevelCleanup,
                    Projectile {
                        damage: 3,
                        life: GTimer::from_seconds(0.9, TimerMode::Once),
                        radius: 6.0,
                        knockback: 40.0,
                        explosive: true,
                        source: Some(DamageSource::player_weapon(
                            player_e,
                            WeaponId::GRENADE_LAUNCHER,
                        )),
                    },
                    Team::Player,
                    Velocity(dir * 420.0),
                    Pos(pos),
                ));
            }
            trauma.add(0.25);
            cue(&mut cues, "sndExplosionL", 0.9, 0.04);
        }
        AbilityKind::BloodGamble => {
            let cur = inv.weapons[inv.current.min(inv.weapon_slots.saturating_sub(1))];
            if cur == WeaponId::NONE {
                return;
            }
            let meta = weapon_meta(cur);
            let ammo_kind = weapon_ammo(cur);
            let amount = ammo_pickup_amount(ammo_kind).max(1);
            let cost = meta.wep_cost as i32;
            if cost <= 0 {
                return;
            }
            player.skeleton_gamble += 1;
            let mut rng = rand::rng();
            let proc = rng.random_range(0..amount) < cost;
            let tb_gate = !player.throne_butt || rng.random_range(0..3) < 2;
            if proc && tb_gate {
                health.hp -= 1;
                player.skeleton_gamble = 0;
            }
            cue(&mut cues, "sndAmmoPickup", 0.5, 0.15);
        }
        AbilityKind::ToxicPuke => {
            let spot = pos + aim_v.normalize_or_zero() * 48.0;
            commands.spawn((
                LevelCleanup,
                AbilityHazard,
                HazardCloud {
                    kind: HazardKind::Toxic,
                    radius: 70.0 * ability_mult.clamp(1.0, 2.0),
                    damage: ((1.0 * ability_mult).ceil() as i32).max(1),
                    timer: GTimer::from_seconds(3.0 * ability_mult.clamp(1.0, 1.8), TimerMode::Once),
                    tick: GTimer::from_seconds(0.25, TimerMode::Repeating),
                },
                Pos(spot),
            ));
            cue(&mut cues, "sndExplosionL", 0.9, 0.04);
        }
        AbilityKind::CuzSwap => {
            // GML Cuz (`scrPowers.gml:430-454`): ring of
            // `20*(1+Emotional)` tears + `spr_cry` swap.
            if player.cuz_ammo == 0 {
                cue(&mut cues, "sndCuzCryAttackNoAmmo", 0.6, 0.05);
                return;
            }
            player.cuz_ammo = player.cuz_ammo.saturating_sub(1);
            let emotional = cuz_emotional_level(player.ultra);
            cue(
                &mut cues,
                if emotional > 0 {
                    "sndCuzCryAttackUltraB"
                } else {
                    "sndCuzCryAttack"
                },
                0.7,
                0.05,
            );
            let tears = 20 * (1 + emotional);
            let step = std::f32::consts::TAU / tears as f32;
            for i in 0..tears {
                let ang = i as f32 * step;
                let dir = Vec2::new(ang.cos(), ang.sin());
                spawn_player_projectile_with_source(
                    &mut commands,
                    pos + dir * 16.0,
                    dir,
                    180.0,
                    3,
                    1.5,
                    5.0,
                    120.0,
                    false,
                    [0.6, 0.9, 1.0],
                    Vec2::new(8.0, 8.0),
                    1,
                    0,
                    None,
                    None,
                    FireArch::default(),
                    Some(DamageSource::player_weapon(player_e, WeaponId::NONE)),
                    None,
                );
            }
            commands.entity(player_e).insert(CryAnim {
                timer: GTimer::from_seconds(0.4, TimerMode::Once),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: headless parity for the combat block
// ---------------------------------------------------------------------------

