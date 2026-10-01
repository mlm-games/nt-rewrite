//! Player combat: firing, melee, abilities, and ability-field ticks.
//!
//! Headless port of the COMBAT half of
//! the GML player scripts: `player_fire`,
//! `fire_burst_volley`, `fire_one_gun`, `apply_weapon_mutation_mods`,
//! `spawn_pellets`, `slash_life_secs`, `flip_melee_angle`, `melee_attack`,
//! `pay_fire_cost`, `spawn_beam_shot`, `spawn_player_projectile`,
//! `spawn_player_projectile_with_source`, `hammerhead_chew`,
//! `move_swing_fx` (lifetime tick only - see below), `tick_snare_zones`,
//! `tick_slowed`, `tick_portal_strikes`, `tick_hazard_clouds`,
//! `player_ability`.
//!
//! Deferred (already ported or render-only, per the task):
//! - `player_move`, `face_aim`, `player_aim`, `weapon_switch`,
//!   `tick_player_timers`, `blink_player` - live in `crate::player` or
//!   are render-only; not duplicated here.
//! - `ally_ai`, `tick_hold_abilities` (Eyes/Horror/Frog hold), weapon
//!   visuals (`ensure_weapon_visual`, `tick_weapon_visuals`,
//!   `held_weapon_angle`) - render or a separate slice; not ported.
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

use crate::audio::{AudioCue, MainVol};
use crate::combat::Explosion;
use crate::comps_a::{
    AbilityHazard, AimDir, BouncesLeft, ChainLightning, CurrentFrame, DamageSource, DiscFlight,
    FireCooldown, FlameShellSlowDeath, FlameTrail, FloorMask, GameCleanup, GrenadeFuse,
    HammerheadBudget, Health, HitId, Hitbox, HitsAllTeams, Homing, Inventory, LevelCleanup,
    NextHurt, PendingWallBreak, PiercesLeft, Player, Projectile, ProjectileFade,
    ProjectileFriction, ProjectileHitSet, ProjectileTyp, ProjectileVisual, RaceState,
    RecycleGlandYield, Run, SaveDirty, ShellBonus, ShellWallBounce, SlashProjectile, SpawnGrace,
    SpawnHazardOnDeath, SplitOnDeath, Sticky, Team, Toast, Velocity, WallCell, WallTile,
    gml_motion_add_clamp,
};
use crate::comps_b::{
    Ally, BIG_DOG_MISSILE_DAMAGE, BIG_DOG_MISSILE_HP, BIG_DOG_MISSILE_RADIUS, BigDogMissileState,
    BloodAmmo, ChestKind, CryAnim, CustomExplosion, Dash, DeploysSentry, Enemy, GmlImage,
    HazardCloud, HorrorCharge, NativeAngle, NativeDepth, NativeExplosionKind, NativeFlip,
    PickupKind, PlasmaBurst, PopPopCharges, PortalStrike, PortalSucking, Prop, PropSprites,
    SecretEntrance, Shield, Slowed, SpawnsWeaponPickup, SwingFx, Telekinesis, WeaponVisual,
};
use crate::data::{
    AbilityKind, AmmoKind, AreaId, CrownKind, HazardDef, HazardKind, MutationId, RaceId, SplitDef,
    UltraMutationId, WeaponId, ammo_pickup_amount,
};
use crate::effects::{
    ChromaticAberration, FiredWeapon, HitStop, RumbleRequest, SlowMotion, chromatic_pulse, rumble,
    slow_motion, spawn_burst,
};
use crate::environment::{
    PropDeathEffect, spawn_native_explosion_visual, spawn_native_smoke_mote, spawn_prop_corpse,
    spawn_prop_death_effect,
};
use crate::input::NtInput;
use crate::msg::Queue;
use crate::pickups::{spawn_chest_with, spawn_flung_weapon_pickup, spawn_pickup, spawn_rad_burst};
use crate::savedata_part::{SaveData, character_def, check_progress_unlocks};
use crate::secrets::SecretTriggers;
use crate::spatial::{PLAYER_RADIUS, Pos, move_bounce_solid, move_contact_solid};
use crate::time::{GTimer, TimerMode};
use crate::weapon_runtime::{
    MeleeDef, WeaponDef, base_weapon_name, gml_fire_push_px_s, gml_melee_wkick,
    melee_projectile_spec, weapon_meta, weapon_runtime_def, weapon_sleep_secs,
};
use crate::weapons_data::AmmoType;
use crate::worldgen::WALL_PX;

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

pub fn gun_reload_fx(
    cues: &mut Queue<AudioCue>,
    weapon_id: WeaponId,
    primary_id: WeaponId,
    inv: &Inventory,
    laser_brain: bool,
) {
    if weapon_id == WeaponId::NONE {
        return;
    }
    let meta = weapon_meta(weapon_id);
    let ammo_type = meta.wep_type;
    let cost = i32::from(meta.wep_cost);

    if ammo_type != AmmoType::None
        && inv.ammo_of(weapon_ammo(weapon_id)) < cost
        && weapon_ammo(primary_id) != AmmoKind::None
    {
        cue(cues, "sndEmpty", 1.0, 0.0);
    }

    if ammo_type == AmmoType::None {
        cue(cues, "sndMeleeFlip", 1.0, 0.0);
    } else if ammo_type == AmmoType::Bolts {
        cue(cues, "sndCrossReload", 1.0, 0.0);
    }

    let name = meta.wep_name;
    if name.starts_with("PLASMA") {
        cue(
            cues,
            if laser_brain {
                "sndPlasmaReloadUpg"
            } else {
                "sndPlasmaReload"
            },
            1.0,
            0.0,
        );
    }
    if name.starts_with("LIGHTNING") {
        cue(cues, "sndLightningReload", 1.0, 0.0);
    }
    if name.starts_with("GRENADE") || matches!(meta.id, 47 | 54 | 72 | 78 | 122) {
        cue(cues, "sndNadeReload", 1.0, 0.0);
    }
    if ammo_type == AmmoType::Shells {
        cue(cues, "sndShotReload", 1.0, 0.0);
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
    pub plasma_scale: f32,
    pub hits_all: bool,
    pub spin: Option<SpinSpawn>,
}

/// GML `objects/DogSpinAttack/*`: a body that rides its creator, throws a
/// 6-round `AllyBullet` volley every 5 steps, spins `4 * turn` degrees
/// per volley, and self-destructs when `ammo` (15) runs out.
#[derive(Component, Clone, Copy, Debug)]
pub struct SpinAttack {
    pub creator: Entity,
    /// GML `turn = choose(1, -1)`.
    pub turn: f32,
    pub direction: f32,
    pub ammo: i32,
    /// GML `alarm[0]`: 7 steps to the first volley, 5 thereafter.
    pub alarm: GTimer,
    /// GML `scr_ultra_get(Race.BigDog, UltraSkill.UltraSpin)` mirror volley.
    pub ultra_spin: bool,
}

/// Per-shot spin parameters (`scrFire.gml:789-796`).
#[derive(Clone, Copy, Debug)]
pub struct SpinSpawn {
    pub ammo: i32,
    pub ultra_spin: bool,
}

/// Archetype lookup by (base) weapon name, mirroring bevy
/// `projectile_archetype` (GOLDEN/ULTRA/CURSED prefixes strip to base).
fn projectile_arch(id: WeaponId) -> FireArch {
    let full = weapon_meta(id).wep_name;
    if full == "ULTRA GRENADE LAUNCHER" {
        return FireArch {
            custom: Some(CustomExplosion {
                visual: Some(NativeExplosionKind::Green),
                ..CustomExplosion::default()
            }),
            ..FireArch::default()
        };
    }
    let base = base_weapon_name(full);
    match base {
        "DOG SPIN ATTACK" => FireArch {
            spin: Some(SpinSpawn {
                ammo: 15,
                ultra_spin: false,
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
                visual: None,
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
                visual: None,
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
            custom: Some(CustomExplosion {
                radius: 32.0,
                count: 1,
                spread: 0.0,
                visual: Some(NativeExplosionKind::Meat),
            }),
            ..FireArch::default()
        },
        "BLOOD CANNON" => FireArch {
            blood: Some(BloodAmmo { hp_cost: 2 }),
            custom: Some(CustomExplosion {
                radius: 32.0,
                count: 1,
                spread: 0.0,
                visual: Some(NativeExplosionKind::Meat),
            }),
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
        "CROSSBOW"
        | "HEAVY CROSSBOW"
        | "AUTO CROSSBOW"
        | "SUPER CROSSBOW"
        | "HEAVY AUTO CROSSBOW"
        | "ULTRA CROSSBOW"
        | "SPLINTER GUN"
        | "SPLINTER PISTOL"
        | "SUPER SPLINTER GUN"
        | "TOXIC BOW" => FireArch {
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
    pub mainvol: &'a mut MainVol,
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
    pub burst: bool,
}

// ---------------------------------------------------------------------------
// player_fire
// ---------------------------------------------------------------------------

/// Firing Cadence: burst continuations, then primary / Steroids-secondary
/// intents. `Transform` -> [`Pos`]; pulses drained every tick.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn player_fire(
    mut time_and_save: ParamSet<(Res<SimTime>, Res<SaveData>)>,
    mut input: ResMut<NtInput>,
    mut commands: Commands,
    mut trauma: ResMut<Trauma>,
    mut hitstop: ResMut<HitStop>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut rumble_q: ResMut<Queue<RumbleRequest>>,
    mut toast: ResMut<Toast>,
    mut mainvol: ResMut<MainVol>,
    mut run: ResMut<Run>,
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
    let shake_scale: f32 = time_and_save.p1().settings.screenshake.clamp(0.0, 2.0);

    let dt = time_and_save.p0().delta_secs;

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

    cooldown.timer.tick(dt);
    cooldown.burst_timer.tick(dt);
    // GML `Player/Step_0`: reload finish flips the melee mirror.
    if cooldown.timer.just_finished() {
        gun_reload_fx(&mut cues, primary_id, primary_id, &inv, player.laser_brain);
        inv.wepflip *= -1.0;
    }
    if cooldown.timer_b.just_finished() {
        gun_reload_fx(
            &mut cues,
            secondary_id,
            primary_id,
            &inv,
            player.laser_brain,
        );
        inv.bwepflip *= -1.0;
    }
    cooldown.timer_b.tick(dt);
    cooldown.burst_timer_b.tick(dt);

    let fire_held = input.fire_held;
    // GML `JoystickAttack` swaps edges by design: the attack finger's
    // lift lands in `fire_released`, which counts as the shot edge
    // (`press_fire` on release) alongside the normal press edge.
    let fire_pressed = input.take_fire_pressed() || input.take_fire_released();
    let spec_held = input.spec_held;
    let spec_pressed = input.take_spec_pressed();

    let mut fx = FireFx {
        trauma: &mut trauma,
        hitstop: &mut hitstop,
        cues: &mut cues,
        rumble: &mut rumble_q,
        toast: &mut toast,
        shake_scale,
        underwater: matches!(run.area, AreaId::Oasis),
        catalog: &catalog,
        mainvol: &mut mainvol,
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
            burst: true,
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
            burst: true,
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
            burst: false,
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
            burst: false,
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
        cue(fx.cues, "sndUltraEmpty", 1.0, 0.2);
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
                cue(fx.cues, "sndBloodHurt", 1.0, 0.2);
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
                cue(fx.cues, "sndEmpty", 1.0, 0.0);
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

    if player.laser_brain && def.name == "DEVASTATOR" {
        def.speed *= 0.6;
    }
    if player.laser_brain && arch.plasma.is_some() && def.name != "DEVASTATOR" {
        arch.plasma_scale = 1.2;
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

fn fire_cue_push(out: &mut Vec<AudioCue>, name: &'static str, volume: f32, variance: f32) {
    if !out.iter().any(|c| c.name == name) {
        out.push(AudioCue {
            name,
            volume,
            variance,
        });
    }
}

fn weapon_fire_cues(fx: &mut FireFx, id: WeaponId, burst: bool, player: &Player, mega: bool) {
    let underwater = fx.underwater;
    let meta = weapon_meta(id);
    let gold = meta.wep_gold;
    let laser_brain = player.laser_brain;
    let electric = melee_projectile_spec(meta.wep_name).electric_guitar;

    let mut out: Vec<AudioCue> = Vec::new();
    let duck = std::cell::Cell::new(0.0f32);
    let gun = |out: &mut Vec<AudioCue>, stem: &'static str, variance: f32| {
        let stem = if underwater && stem != "sndOasisMelee" {
            "sndOasisShoot"
        } else {
            stem
        };
        duck.set(0.3);
        fire_cue_push(out, stem, 1.0, variance);
    };
    let big = |out: &mut Vec<AudioCue>, stem: &'static str, variance: f32| {
        let stem = if underwater { "sndOasisShoot" } else { stem };
        duck.set(0.33);
        fire_cue_push(out, stem, 1.0, variance);
    };
    let play = |out: &mut Vec<AudioCue>, stem: &'static str| fire_cue_push(out, stem, 1.0, 0.0);

    if !burst && underwater && weapon_ammo(id) == AmmoKind::None {
        gun(&mut out, "sndOasisMelee", 0.2);
    }

    if burst {
        match id.0 {
            17 | 103 => gun(
                &mut out,
                if gold {
                    "sndGoldMachinegun"
                } else {
                    "sndMachinegun"
                },
                0.2,
            ),
            23 => play(&mut out, "sndSlugger"),
            26 => gun(&mut out, "sndHyperRifle", 0.1),
            71 => gun(&mut out, "sndPopgun", 0.2),
            80 => gun(&mut out, "sndGrenadeRifle", 0.2),
            81 => play(&mut out, "sndRogueRifle"),
            106 => gun(&mut out, "sndHeavyMachinegun", 0.2),
            _ => {}
        }
    } else {
        match id.0 {
            1 | 39 | 56 => gun(
                &mut out,
                if gold { "sndGoldPistol" } else { "sndPistol" },
                0.2,
            ),
            2 => gun(&mut out, "sndTripleMachinegun", 0.2),
            3 | 40 => gun(
                &mut out,
                if gold { "sndGoldWrench" } else { "sndWrench" },
                0.2,
            ),
            4 | 41 => gun(
                &mut out,
                if gold {
                    "sndGoldMachinegun"
                } else {
                    "sndMachinegun"
                },
                0.2,
            ),
            5 => play(&mut out, if gold { "sndGoldShotgun" } else { "sndShotgun" }),
            6 | 11 | 43 => gun(
                &mut out,
                if gold {
                    "sndGoldCrossbow"
                } else {
                    "sndCrossbow"
                },
                0.2,
            ),
            7 | 44 => gun(
                &mut out,
                if gold { "sndGoldGrenade" } else { "sndGrenade" },
                0.2,
            ),
            8 => play(&mut out, "sndDoubleShotgun"),
            9 => gun(&mut out, "sndMinigun", 0.2),
            10 => play(&mut out, "sndShotgun"),
            12 => gun(&mut out, "sndSuperCrossbow", 0.2),
            13 => gun(&mut out, "sndShovel", 0.2),
            14 | 84 | 102 => gun(
                &mut out,
                if gold { "sndGoldRocket" } else { "sndRocket" },
                0.2,
            ),
            15 => play(&mut out, "sndGrenade"),
            16 => gun(&mut out, "sndPistol", 0.2),
            17 | 103 => gun(
                &mut out,
                if gold {
                    "sndGoldMachinegun"
                } else {
                    "sndMachinegun"
                },
                0.2,
            ),
            18 | 123 => gun(&mut out, "sndDiscgun", 0.2),
            19 | 20 | 28 | 45 => gun(
                &mut out,
                if gold {
                    if laser_brain {
                        "sndGoldLaserUpg"
                    } else {
                        "sndGoldLaser"
                    }
                } else if laser_brain {
                    "sndLaserUpg"
                } else {
                    "sndLaser"
                },
                0.2,
            ),
            21 | 22 | 99 => gun(
                &mut out,
                if gold { "sndGoldSlugger" } else { "sndSlugger" },
                0.2,
            ),
            23 => play(&mut out, "sndSlugger"),
            24 => gun(
                &mut out,
                if laser_brain {
                    "sndEnergySwordUpg"
                } else {
                    "sndEnergySword"
                },
                0.2,
            ),
            25 => big(&mut out, "sndSuperSlugger", 0.2),
            26 => gun(&mut out, "sndHyperRifle", 0.2),
            27 | 101 => gun(
                &mut out,
                if gold {
                    "sndGoldScrewdriver"
                } else {
                    "sndScrewdriver"
                },
                0.2,
            ),
            29 => gun(&mut out, "sndBloodLauncher", 0.2),
            30 => gun(&mut out, "sndSplinterGun", 0.2),
            31 => gun(&mut out, "sndCrossbow", 0.2),
            32 => gun(&mut out, "sndGrenade", 0.2),
            33 => {
                gun(&mut out, "sndWaveGun", 0.2);
                play(&mut out, "sndShotgun");
            }
            34 | 98 => gun(
                &mut out,
                if gold {
                    if laser_brain {
                        "sndGoldPlasmaUpg"
                    } else {
                        "sndGoldPlasma"
                    }
                } else if laser_brain {
                    "sndPlasmaUpg"
                } else {
                    "sndPlasma"
                },
                0.2,
            ),
            35 => gun(
                &mut out,
                if laser_brain {
                    "sndPlasmaBigUpg"
                } else {
                    "sndPlasmaBig"
                },
                0.2,
            ),
            36 => gun(
                &mut out,
                if laser_brain {
                    "sndEnergyHammer"
                } else {
                    "sndEnergyHammerUpg"
                },
                0.2,
            ),
            37 => play(&mut out, "sndJackHammer"),
            38 => gun(&mut out, "sndFlakCannon", 0.2),
            42 => play(&mut out, "sndGoldShotgun"),
            46 => gun(&mut out, "sndChickenSword", 0.2),
            47 | 122 => gun(&mut out, "sndNukeFire", 0.2),
            48 => gun(
                &mut out,
                if laser_brain {
                    "sndLaserUpg"
                } else {
                    "sndLaser"
                },
                0.2,
            ),
            49 => gun(&mut out, "sndQuadMachinegun", 0.2),
            52 => gun(&mut out, "sndFlare", 0.2),
            53 => gun(
                &mut out,
                if laser_brain {
                    "sndEnergyScrewdriverUpg"
                } else {
                    "sndEnergyScrewdriver"
                },
                0.2,
            ),
            54 => gun(&mut out, "sndHyperLauncher", 0.2),
            55 => gun(&mut out, "sndLaserCannonCharge", 0.2),
            57 | 64 => gun(
                &mut out,
                if laser_brain {
                    "sndLightningPistolUpg"
                } else {
                    "sndLightningPistol"
                },
                0.2,
            ),
            58 => gun(
                &mut out,
                if laser_brain {
                    "sndLightningRifleUpg"
                } else {
                    "sndLightningRifle"
                },
                0.2,
            ),
            59 => gun(
                &mut out,
                if laser_brain {
                    "sndLightningShotgunUpg"
                } else {
                    "sndLightningShotgun"
                },
                0.2,
            ),
            60 => gun(&mut out, "sndSuperFlakCannon", 0.2),
            61 => play(&mut out, "sndSawedOffShotgun"),
            62 => gun(&mut out, "sndSplinterPistol", 0.2),
            63 => {
                gun(&mut out, "sndSuperSplinterGun", 0.2);
                gun(&mut out, "sndSplinterGun", 0.1);
            }
            65 => gun(&mut out, "sndSmartgun", 0.2),
            66 | 105 => gun(&mut out, "sndHeavyCrossbow", 0.2),
            67 => gun(&mut out, "sndBloodHammer", 0.2),
            68 => gun(
                &mut out,
                if laser_brain {
                    "sndLightningCannonUpg"
                } else {
                    "sndLightningCannon"
                },
                0.2,
            ),
            69 => gun(&mut out, "sndPopgun", 0.2),
            70 => gun(
                &mut out,
                if laser_brain {
                    "sndPlasmaRifleUpg"
                } else {
                    "sndPlasmaRifle"
                },
                0.2,
            ),
            71 => gun(&mut out, "sndPopgun", 0.2),
            72 => gun(&mut out, "sndToxicLauncher", 0.2),
            73 => big(&mut out, "sndFlameCannon", 0.2),
            74 => gun(&mut out, "sndLightningHammer", 0.2),
            75 | 77 => play(&mut out, "sndFireShotgun"),
            76 => gun(&mut out, "sndDoubleFireShotgun", 0.2),
            78 => gun(&mut out, "sndClusterLauncher", 0.2),
            79 | 85 => gun(&mut out, "sndGrenadeShotgun", 0.2),
            80 => gun(&mut out, "sndGrenadeRifle", 0.2),
            81 => play(&mut out, "sndRogueRifle"),
            82 => big(&mut out, "sndConfettiGun", 0.2),
            83 => big(&mut out, "sndDoubleMinigun", 0.2),
            86 => big(&mut out, "sndUltraPistol", 0.2),
            87 => big(
                &mut out,
                if laser_brain {
                    "sndUltraLaserUpg"
                } else {
                    "sndUltraLaser"
                },
                0.2,
            ),
            88 => gun(&mut out, "sndHammer", 0.2),
            89 | 90 => gun(&mut out, "sndHeavyRevolver", 0.2),
            91 => gun(&mut out, "sndHeavySlugger", 0.2),
            92 => big(&mut out, "sndUltraShovel", 0.2),
            93 => big(&mut out, "sndUltraShotgun", 0.2),
            94 => big(&mut out, "sndUltraCrossbow", 0.2),
            95 => big(&mut out, "sndUltraGrenade", 0.2),
            96 => gun(
                &mut out,
                if laser_brain {
                    "sndPlasmaMinigunUpg"
                } else {
                    "sndPlasmaMinigun"
                },
                0.2,
            ),
            97 => gun(
                &mut out,
                if laser_brain {
                    "sndDevastatorUpg"
                } else {
                    "sndDevastator"
                },
                0.2,
            ),
            100 => gun(&mut out, "sndGoldSplinterGun", 0.2),
            104 => gun(&mut out, "sndSuperDiscGun", 0.2),
            106 => gun(&mut out, "sndHeavyMachinegun", 0.2),
            107 => gun(&mut out, "sndBloodCannon", 0.2),
            108 => gun(&mut out, "sndBigDogSpin", 0.2),
            110 => gun(&mut out, "sndIncinerator", 0.2),
            111 => gun(
                &mut out,
                if laser_brain {
                    "sndPlasmaHugeUpg"
                } else {
                    "sndPlasmaHuge"
                },
                0.2,
            ),
            112 => gun(&mut out, "sndSeekerPistol", 0.2),
            113 => gun(&mut out, "sndSeekerShotgun", 0.2),
            114 => gun(&mut out, "sndEraser", 0.2),
            115 | 128 => gun(
                &mut out,
                if electric {
                    "sndElectricGuitar"
                } else {
                    "sndGuitar"
                },
                0.2,
            ),
            116 => gun(&mut out, "sndBouncerSmg", 0.2),
            117 => gun(&mut out, "sndBouncerShotgun", 0.2),
            118 => gun(&mut out, "sndHyperSlugger", 0.2),
            119 => gun(&mut out, "sndSuperBazooka", 0.2),
            120 | 127 => gun(
                &mut out,
                if gold {
                    "sndGoldFrogPistol"
                } else {
                    "sndFrogPistol"
                },
                0.2,
            ),
            121 => gun(
                &mut out,
                if mega {
                    "sndBlackSwordMega"
                } else {
                    "sndBlackSword"
                },
                0.2,
            ),
            124 => gun(&mut out, "sndHeavyNader", 0.2),
            125 => gun(&mut out, "sndGunGun", 0.2),
            _ => {}
        }
    }

    for c in out {
        cue(fx.cues, c.name, c.volume, c.variance);
    }
    if duck.get() > 0.0 {
        fx.mainvol.duck(duck.get());
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

    weapon_fire_cues(fx, id, shot.burst, player, false);

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
    // GML `ClusterNade/Destroy_0.gml:1`: `8 + scrCrownCheck(crwn_death)`
    // children. The count is baked here because the split is resolved at
    // the projectile's death, long after the fire.
    if base_weapon_name(def.name) == "CLUSTER LAUNCHER"
        && let Some(split) = &mut def.split
    {
        split.pellets += u8::from(player.crown == CrownKind::Death);
    }

    if let Some(mut beam) = archetype.beam {
        beam.damage = def.damage;
        spawn_beam_shot(
            commands,
            muzzle,
            shot.aim.normalize_or_zero(),
            beam,
            Some(DamageSource::player_weapon(shot.player_ent, id)),
        );
        return;
    }

    // GML `scrFire.gml:361-370`: the sentry gun is not a projectile at
    // all - `instance_create(x, y, SentryGun) { motion_add(_gunangle, 6) }`
    // deploys the turret body on the spot.
    if base_weapon_name(def.name) == "SENTRY GUN" {
        crate::spawns::spawn_sentry_gun(
            commands,
            shot.pos,
            shot.aim.normalize_or_zero(),
            Team::Player,
        );
        return;
    }

    if let Some(spin) = archetype.spin {
        spawn_dog_spin_attack(
            commands,
            shot.pos,
            shot.player_ent,
            SpinSpawn {
                ultra_spin: matches!(player.ultra, Some(UltraMutationId::BigDogHeavyArtillery)),
                ..spin
            },
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

/// GML `scrFire.gml:789-796` / `DogSpinAttack/Create_0.gml:1-7`.
fn spawn_dog_spin_attack(
    commands: &mut Commands,
    pos: Vec2,
    creator: Entity,
    spec: SpinSpawn,
) {
    let mut rng = rand::rng();
    commands.spawn((
        LevelCleanup,
        SpinAttack {
            creator,
            turn: if rng.random_bool(0.5) { 1.0 } else { -1.0 },
            direction: rng.random_range(0.0..std::f32::consts::TAU),
            ammo: spec.ammo,
            alarm: GTimer::from_seconds(7.0 / 30.0, TimerMode::Once),
            ultra_spin: spec.ultra_spin,
        },
        Pos(pos),
    ));
}

/// GML `DogSpinAttack/Alarm_0.gml:18-58`. Driven from [`move_swing_fx`]
/// because `schedule.rs` owns the system list and has no dedicated slot
/// for the spin body.
fn tick_dog_spin_attacks(
    time: Res<SimTime>,
    commands: &mut Commands,
    spins: &mut Query<(Entity, &mut SpinAttack, &mut Pos), (With<SpinAttack>, Without<Player>)>,
    creators: &Query<&Pos, With<Player>>,
) {
    let dt = time.delta_secs;
    for (e, mut spin, mut pos) in spins.iter_mut() {
        let Ok(owner) = creators.get(spin.creator) else {
            commands.entity(e).despawn();
            continue;
        };
        // GML `Alarm_0.gml:6-16`: the body rides the creator's position.
        pos.0 = owner.0;
        spin.alarm.tick(dt);
        if !spin.alarm.just_finished() {
            continue;
        }

        for volley in 0..usize::from(spin.ultra_spin) + 1 {
            let sign = if volley == 0 { 1.0 } else { -1.0 };
            for _ in 0..6 {
                let ang = spin.direction * sign;
                spawn_ally_bullet(
                    commands,
                    pos.0 + Vec2::new(24.0 * ang.cos(), 16.0 * ang.sin()),
                    Vec2::new(ang.cos(), ang.sin()) * 60.0,
                    spin.creator,
                );
                spin.direction += 60.0_f32.to_radians();
            }
        }

        spin.direction += 4.0_f32.to_radians() * spin.turn;
        spin.ammo -= 1;
        if spin.ammo <= 0 {
            commands.entity(e).despawn();
            continue;
        }
        spin.alarm = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
    }
}

/// GML `AllyBullet/Create_0.gml:3-5`: `typ = 1` (deflectable), damage 3,
/// `spr_fade = sprAllyBulletHit`, destroyed on a wall.
fn spawn_ally_bullet(
    commands: &mut Commands,
    pos: Vec2,
    vel: Vec2,
    creator: Entity,
) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        Team::Player,
        Projectile {
            damage: 3,
            life: GTimer::from_seconds(4.0, TimerMode::Once),
            radius: 4.0,
            knockback: 120.0,
            explosive: false,
            source: Some(DamageSource::player_weapon(creator, WeaponId::NONE)),
        },
        Velocity(vel),
        ProjectileTyp(1),
        ProjectileFade("images/sprAllyBulletHit.png"),
        Pos(pos),
    ));
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
    let mega = shot.def.name == "BLACK SWORD" && (health.hp <= 0 || health.max <= 0);
    weapon_fire_cues(fx, shot.weapon_id, shot.burst, player, mega);
    for mut wv in vis_q.iter_mut() {
        if wv.owner == shot.player_ent && wv.slot == shot.visual_slot {
            wv.wkick = gml_melee_wkick(shot.def.name);
            wv.wep_angle = flip_melee_angle(wv.wep_angle);
        }
    }

    vel.0 -= shot.aim.normalize_or_zero() * gml_fire_push_px_s(shot.def.name);
    player.melee_flip = !player.melee_flip;

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
    // GML `scrFire` melee arms all do `instance_create(x, y, Dust)` at
    // the player on every swing; the puff re-anchors the eye at the
    // swing origin (bevy never ported it).
    {
        let mut rng = rand::rng();
        crate::effects::spawn_burst(
            commands,
            &mut rng,
            player_pos,
            1,
            [0.75, 0.72, 0.68, 1.0],
            (20.0, 60.0),
        );
    }

    // Bevy reads the fired strip (`anim_opt.def.frames`, mega sprite
    // when applicable), defaulting to 3 - never a per-kind constant.
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
            ec.insert(DiscFlight {
                dist: 0.0,
                home: pos,
            });
        }
    }
    if let Some(w) = weapon {
        // Bevy gates on ids 7/44 (plain + golden launcher only):
        // grenade shotguns/rifles/ultras keep their shell behavior.
        let full = weapon_meta(w).wep_name;
        // GML `scrFire` spawns the base `Grenade` object for the grenade
        // launcher (`Grenade/Create_0.gml:10-11` -> friction 0.1 with
        // `alarm[1] = 6`), the sticky launcher (`:219-223` reuses
        // `Grenade` with `sticky = true`) and the cluster launcher
        // (`ClusterNade` inherits `Grenade`), so all three run
        // `Grenade/Alarm_1` (friction -> 0.4 + 4 `Smoke`).
        if full == "GRENADE LAUNCHER"
            || full == "GOLDEN GRENADE LAUNCHER"
            || full == "STICKY LAUNCHER"
            || full == "CLUSTER LAUNCHER"
        {
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
        ec.insert(crate::comps_a::PlasmaSize(
            if archetype.plasma_scale > 0.0 {
                archetype.plasma_scale
            } else {
                1.0
            },
        ));
    }
    if archetype.hits_all {
        ec.insert(HitsAllTeams);

        ec.insert(SpawnGrace(GTimer::from_seconds(
            2.0 / 30.0,
            TimerMode::Once,
        )));
    }

    if let Some(w) = weapon {
        let full = weapon_meta(w).wep_name;
        if weapon_ammo(w) == AmmoKind::Bullets {
            let recycle_yield = if shell_kind == Some(ShellKind::Bullet2) {
                0
            } else if full.contains("BOUNCER") {
                1
            } else if full.contains("HEAVY REVOLVER")
                || full.contains("HEAVY MACHINEGUN")
                || full.contains("ULTRA REVOLVER")
            {
                2
            } else {
                1
            };
            if recycle_yield > 0 {
                ec.insert(RecycleGlandYield(recycle_yield));
            }
        }
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

        if ammo == AmmoKind::Bullets && !base.contains("DISC") && !base.contains("BOUNCER") {
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
    save: Res<crate::savedata_part::SaveData>,
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
                &catalog,
                save.settings.particles,
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
/// marker so it cannot leak. Also drives the `DogSpinAttack` body
/// (`objects/DogSpinAttack/Alarm_0.gml`), which has no slot of its own
/// in `schedule.rs`.
pub fn move_swing_fx(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut SwingFx)>,
    mut spins: Query<(Entity, &mut SpinAttack, &mut Pos), (With<SpinAttack>, Without<Player>)>,
    creators: Query<&Pos, With<Player>>,
) {
    for (e, mut fx) in &mut q {
        fx.timer.tick(time.delta_secs);
        if fx.timer.just_finished() {
            commands.entity(e).despawn();
        }
    }
    tick_dog_spin_attacks(time, &mut commands, &mut spins, &creators);
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

/// GML `objects/TangleSeed/*` (`scrPowers.gml:126`): a 12 px/step seed
/// that becomes a [`Tangle`] on the first `hitme` or `Wall` it touches.
#[derive(Component, Clone, Copy, Debug)]
pub struct SnareSeed {
    pub creator: Entity,
    /// GML `TangleSeed/Destroy_0.gml` under
    /// `scr_ultra_get(Race.Plant, UltraSkill.Trapper)`: the destroy path
    /// (wall contact and enemy contact both run it) also seeds a ring of
    /// 5 tangles at `move_contact_solid(_ang, 26 + irandom(8))`.
    pub trapper: bool,
}

/// GML `objects/Tangle/*`: the snare body. No lifetime and no timer - it
/// persists until the Plant's next ability press runs
/// `instance_destroy(Tangle)` (`scrPowers.gml:128`).
#[derive(Component, Clone, Copy, Debug)]
pub struct Tangle {
    pub creator: Entity,
    pub team: Team,
}

/// `sprTangle` collision box is 47x33; the overlap test is a circle at
/// half the shorter side. GML gives `Tangle` a real mask (its sprite has
/// no `spriteMaskId`), so `TrapFire/Collision_Tangle` can see it - the
/// port carries the same circle as a [`Hitbox`].
pub const TANGLE_RADIUS: f32 = 16.0;

/// GML `scripts/scrPowers/scrPowers.gml:116-131` (Race.Plant) plus
/// `scripts/scrControlAutoSnare/scrControlAutoSnare.gml`. The seed flies
/// at `gunangle` and only becomes a snare where it lands, so the arm
/// spawns a travelling [`SnareSeed`] and clears every live [`Tangle`].
fn spawn_snare_seed(
    commands: &mut Commands,
    cues: &mut Queue<AudioCue>,
    pos: Vec2,
    dir: Vec2,
    creator: Entity,
    throne_butt: bool,
    trapper: bool,
) {
    commands.spawn((
        LevelCleanup,
        SnareSeed { creator, trapper },
        Velocity(dir * 12.0 * 30.0),
        GmlImage::new("images/sprTangleSeed.png", 1, 0.0),
        NativeDepth(1.0),
        Pos(pos),
    ));
    cue(
        cues,
        if throne_butt {
            "sndPlantFireTB"
        } else {
            "sndPlantFire"
        },
        1.0,
        0.0,
    );
}

/// GML `TangleSeed/Collision_hitme.gml:1-8` +
/// `Collision_Wall.gml:1-9` + `Tangle/Create_0.gml:1-14`.
fn plant_tangle(
    commands: &mut Commands,
    cues: &mut Queue<AudioCue>,
    pos: Vec2,
    team: Team,
    creator: Entity,
    throne_butt: bool,
    trapper: bool,
    rng: &mut rand::rngs::ThreadRng,
) {
    let ultra = if trapper { "sndPlantSnareTrapperTB" } else { "sndPlantSnareTB" };
    let plain = if trapper {
        "sndPlantSnareTrapper"
    } else {
        "sndPlantSnare"
    };
    cue(cues, if throne_butt { ultra } else { plain }, 1.0, 0.0);
    // GML `Tangle/Create_0.gml:4-5`: `image_xscale = choose(1, -1)`,
    // `image_speed = 0.4` over the 6-frame strip.
    let body = |commands: &mut Commands, at: Vec2, rng: &mut rand::rngs::ThreadRng| {
        commands.spawn((
            LevelCleanup,
            Tangle { creator, team },
            // GML `Tangle` inherits `Player`, so it carries a body: the
            // mask `TrapFire/Collision_Tangle.gml` collides against.
            Hitbox {
                radius: TANGLE_RADIUS,
            },
            GmlImage::new("images/sprTangle.png", 6, 0.4),
            NativeFlip(rng.random_bool(0.5)),
            NativeAngle(0.0),
            NativeDepth(1.0),
            Pos(at),
        ));
    };
    body(commands, pos, rng);
    if trapper {
        let mut ang = rng.random_range(0.0..std::f32::consts::TAU);
        for _ in 0..5 {
            let at = pos
                + Vec2::new(ang.cos(), ang.sin()) * rng.random_range(26.0..34.0);
            body(commands, at, rng);
            ang += 72.0_f32.to_radians();
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn tick_snare_zones(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Option<Res<FloorMask>>,
    player_q: Query<(&Player, &Team), With<Player>>,
    mut seeds: Query<(Entity, &mut Pos, &SnareSeed, &Velocity), (With<SnareSeed>, Without<Player>)>,
    // `Without<SnareSeed>` keeps this disjoint from the travelling-seed
    // query above, which holds `&mut Pos`; a seed is despawned as it
    // plants, so no entity is ever both.
    tangles: Query<&Pos, (With<Tangle>, Without<Player>, Without<SnareSeed>)>,
    mut enemies: Query<
        (
            Entity,
            &mut Pos,
            &Hitbox,
            &mut Health,
            Option<&mut Velocity>,
        ),
        (With<Enemy>, Without<Player>, Without<SnareSeed>, Without<Tangle>),
    >,
) {
    let dt = time.delta_secs;
    let (throne_butt, team) = player_q
        .single()
        .map(|(p, t)| (p.throne_butt, *t))
        .unwrap_or((false, Team::Player));
    let walkable = |p: Vec2| mask.as_deref().is_none_or(|m| m.is_walkable(p));

    // Seed travel: GML `TangleSeed` keeps `speed = 12` forever and is
    // consumed by the first body or wall it reaches.
    let mut rng = rand::rng();
    for (e, mut spos, seed, vel) in &mut seeds {
        spos.0 += vel.0 * dt;
        let hit_wall = !walkable(spos.0);
        let hit_body = enemies
            .iter()
            .any(|(_, epos, hitbox, _, _)| epos.0.distance(spos.0) <= hitbox.radius);
        if hit_wall || hit_body {
            plant_tangle(
                &mut commands,
                &mut cues,
                spos.0,
                team,
                seed.creator,
                throne_butt,
                seed.trapper,
                &mut rng,
            );
            commands.entity(e).despawn();
        }
    }

    // GML `Tangle/Collision_enemy.gml:4-22`, once per tick of overlap.
    for tangle_pos in &tangles {
        let tpos = tangle_pos.0;
        for (_, mut epos, hitbox, mut health, mut vel) in &mut enemies {
            if epos.0.distance(tpos) > hitbox.radius + TANGLE_RADIUS {
                continue;
            }
            if let Some(vel) = vel.as_mut() {
                let step = vel.0 * dt;
                let rewound = epos.0 - step * 0.9;
                if walkable(rewound) {
                    epos.0 = rewound;
                } else {
                    epos.0 -= step;
                }
            }
            // GML :13-21 - float compare against `max_hp * 0.33`, a
            // 5 px/step shove away from the tangle, then `hp = 0`.
            if throne_butt && (health.hp as f32) <= health.max as f32 * 0.33 {
                if let Some(vel) = vel.as_mut() {
                    vel.0 += (epos.0 - tpos).normalize_or_zero() * 5.0 * 30.0;
                }
                health.hp = 0;
            }
        }
    }
}

/// GML has no per-tick `Slowed` component: the Plant snare is the
/// positional rewind in [`tick_snare_zones`]. Kept as a live system
/// because `schedule.rs` links it.
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
    save: Res<SaveData>,
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
        cue(&mut cues, "sndIDPDNadeExplo", 1.0, 0.0);
        spawn_native_explosion_visual(
            &mut commands,
            save.settings.particles,
            pos,
            NativeExplosionKind::Popo,
            false,
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
/// the ground. (Rads/curse handling lives in the active arm - GML keeps
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
    run_area: crate::data::AreaId,
    gen_seed: u64,
) {
    let tb = u32::from(player.throne_butt);
    let life_crown = player.crown == CrownKind::Life;
    let ammo_cap = |player: &Player, kind: AmmoKind| player.ammo_cap(kind);
    // GML `HPPickup/Collision_Player.gml:11-14` heals exactly `num`
    // (2, 4 with Second Stomach, +1 with Haste) - no crown or ultra
    // multiplier exists on this path.
    let medkit_num = crate::pickups::hppickup_num(player);
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
                health.hp = (health.hp + hp_amount).min(health.max);
            } else {
                let off = Vec2::new(rng.random_range(-12.0..12.0), rng.random_range(-12.0..12.0));
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
                let off = Vec2::new(rng.random_range(-12.0..12.0), rng.random_range(-12.0..12.0));
                // GML `scrPowers.gml:637,655` `__spawn_pickup(AmmoPickup,
                // _auto_collect)` creates a real `AmmoPickup`, so
                // `AmmoPickup/Create_0.gml:13-18`'s `CursedPickup`
                // conversion applies - its gate is `instance_exists(Player)`,
                // not `GenCont`. A conversion discards the type resolved
                // here, exactly as GML re-rolls it in
                // `AmmoPickup/Collision_Player`.
                crate::pickups::maybe_cursed_ammo(
                    commands,
                    catalog,
                    crate::pickups::count_cursed(inv),
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
            drop_hp_or_ammo(commands, pos, player, health, inv, &mut rng, medkit_num);
        }
    }
    if matches!(player.ultra, Some(UltraMutationId::RobotRegurgitate))
        && rng.random::<f32>() <= 0.43
    {
        // GML Regurgitate roll (no life-crown gate on this branch).
        // `__spawn_pickup` creates at the caller's own `x, y` with no
        // offset, and it spawns CHESTS (`scrPowers.gml:639-648`), so no
        // `CursedPickup` conversion is involved - but the area/ultra art
        // variants still apply.
        let ctx = crate::pickups::ChestCtx {
            worldgen: false,
            area: run_area,
            crown: player.crown,
            ambidextrous: matches!(player.ultra, Some(UltraMutationId::SteroidsAmbidextrous)),
            get_loaded: matches!(player.ultra, Some(UltraMutationId::SteroidsGetArmed)),
            gen_seed,
            order: 0,
        };
        if player.crown == CrownKind::Love {
            spawn_chest_with(commands, catalog, ChestKind::Ammo, pos, &ctx);
        } else if rng.random_range(0..health.max.max(1)) as i32 > health.hp
            && rng.random_range(0..3) < 2
        {
            spawn_chest_with(commands, catalog, ChestKind::Health, pos, &ctx);
        } else {
            let kind = match rng.random_range(0..3) {
                0 => ChestKind::Weapon,
                _ => ChestKind::Ammo,
            };
            spawn_chest_with(commands, catalog, kind, pos, &ctx);
        }
    }
    for _ in 0..(1 + tb) {
        drop_hp_or_ammo(commands, pos, player, health, inv, &mut rng, medkit_num);
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
    mut persist: ParamSet<(
        ResMut<SaveData>,
        ResMut<SaveDirty>,
        Res<Run>,
    )>,
    catalog: Res<repame_anim::AnimCatalog>,
    mut tut: Option<ResMut<crate::state::TutorialState>>,
    mut player_q: Query<
        (
            Entity,
            &Pos,
            &mut Player,
            &mut Health,
            &mut Velocity,
            &mut AimDir,
            &mut Inventory,
            &RaceState,
            Option<&mut Shield>,
            Option<&mut Telekinesis>,
            Option<&Dash>,
            Option<&HorrorCharge>,
        ),
        (With<Player>, Without<Enemy>),
    >,
    mut walls_and_allies: ParamSet<(
        Query<(Entity, &WallCell, &Pos), With<WallTile>>,
        Query<Entity, With<Ally>>,
        Query<&SnareSeed>,
        Query<Entity, (With<Tangle>, Without<Player>)>,
    )>,
    mut enemies: Query<(Entity, &Pos, &mut Health), (With<Enemy>, Without<Player>)>,
) {
    let Ok((
        player_e,
        ppos,
        mut player,
        mut health,
        mut vel,
        mut aim,
        mut inv,
        race_state,
        shield,
        telek,
        dash,
        horror_charge,
    )) = player_q.single_mut()
    else {
        return;
    };

    let fire = input.take_ability_pressed();
    if race_state.race == RaceId::Steroids {
        return;
    }
    // GML `scrControlAutoSnare` verbatim: Plant auto-fires the snare
    // off EITHER fire edge (press or release) when the aim ray hits an
    // unsnared enemy within half a view (213px) with clear walls. The
    // shot aims itself at the victim (`scrPowers` runs with the snapped
    // `gunangle`, restored after): the port writes the snapped aim for
    // this tick and the next `player_aim` pass re-steers from live
    // input. Touch routes through the same take-once `fire_pressed`
    // (press edge) / `fire_released` (attack-finger lift) channels as
    // desktop, so this covers touch AND desktop identically - no
    // touch-only latch needed here. (No `scr_player_pref(my_player,
    // "plant")` gate in the port: no per-race pref store exists; Plant
    // always snares.)
    {
        let press_edge = input.peek_fire_pressed();
        let release_edge = input.take_touch_released_fire();
        // GML `scrControlAutoSnare.gml:21-23`: a `TangleSeed` already in
        // flight blocks the auto-snare.
        let seed_in_flight = walls_and_allies
            .p2()
            .iter()
            .any(|s: &SnareSeed| s.creator == player_e);
        if race_state.race == RaceId::Plant
            && player.ability == AbilityKind::Snare
            && (press_edge || release_edge)
            && !seed_in_flight
        {
            let from = ppos.0;
            let dir = aim.0;
            let mut victim: Option<(glam::Vec2, f32)> = None;
            for (_, epos, _) in &enemies {
                let to = epos.0 - from;
                let along = to.dot(dir);
                if along < 0.0 || along > 213.0 {
                    continue;
                }
                let side = (to - dir * along).length();
                if side > 12.0 {
                    continue;
                }
                if victim.is_none_or(|(_, bd)| along < bd) {
                    victim = Some((epos.0, along));
                }
            }
            if let Some((vpos, _)) = victim
                && !walls_block_snare(walls_and_allies.p0(), from, vpos)
            {
                let snapped = (vpos - from).normalize_or_zero();
                aim.0 = snapped;
                let live_tangles: Vec<Entity> = walls_and_allies.p3().iter().collect();
                plant_snare(
                    &mut commands,
                    &mut cues,
                    from,
                    snapped,
                    player_e,
                    &player,
                    &live_tangles,
                );
                return;
            }
        }
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

    /// GML `scrPowers.gml:116-131` (Race.Plant) shared by the tap ability
    /// and `scrControlAutoSnare`: fire the seed down `aim` and clear every
    /// live `Tangle`.
    #[allow(clippy::too_many_arguments)]
    fn plant_snare(
        commands: &mut Commands,
        cues: &mut Queue<AudioCue>,
        pos: glam::Vec2,
        aim_v: glam::Vec2,
        player_e: Entity,
        player: &Player,
        live_tangles: &[Entity],
    ) {
        let trapper = matches!(player.ultra, Some(UltraMutationId::PlantTrapper));
        spawn_snare_seed(
            commands,
            cues,
            pos,
            aim_v,
            player_e,
            player.throne_butt,
            trapper,
        );
        for tangle in live_tangles {
            commands.entity(*tangle).despawn();
        }
    }

    /// GML wall segment test for the auto-snare ray (`collision_line`
    /// against `Wall`): true when any wall cell center comes within 8px of
    /// the from→to segment.
    fn walls_block_snare(
        walls: Query<(Entity, &WallCell, &Pos), With<WallTile>>,
        from: glam::Vec2,
        to: glam::Vec2,
    ) -> bool {
        let d = to - from;
        let len2 = d.length_squared().max(1e-6);
        for (_, _, wpos) in walls {
            let t = ((wpos.0 - from).dot(d) / len2).clamp(0.0, 1.0);
            if (wpos.0 - (from + d * t)).length() < 8.0 {
                return true;
            }
        }
        false
    }

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
            if player.throne_butt {
                cue(&mut cues, "sndFishRollUpg", 1.0, 0.0);
            } else if dash.is_none() {
                cue(&mut cues, "sndRoll", 1.0, 0.0);
            }
        }
        AbilityKind::Shield => {
            let timer = GTimer::from_seconds(1.6 * ability_mult.clamp(1.0, 2.0), TimerMode::Once);
            if let Some(mut s) = shield {
                s.timer = timer;
            } else {
                commands.entity(player_e).insert(Shield { timer });
            }
            trauma.add(0.08);
            cue(
                &mut cues,
                if matches!(player.ultra, Some(UltraMutationId::CrystalJuggernaut)) {
                    "sndCrystalJuggernaut"
                } else {
                    "sndCrystalShield"
                },
                1.0,
                0.0,
            );
        }
        AbilityKind::Telekinesis => {
            let timer = GTimer::from_seconds(1.4, TimerMode::Once);
            if let Some(mut t) = telek {
                t.timer = timer;
            } else {
                commands.entity(player_e).insert(Telekinesis { timer });
            }
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
            // GML `scrPowers.gml:120-129`: a `TangleSeed` already in
            // flight blocks the press.
            let seed_in_flight = walls_and_allies
                .p2()
                .iter()
                .any(|s: &SnareSeed| s.creator == player_e);
            if seed_in_flight {
                return;
            }
            let live_tangles: Vec<Entity> = walls_and_allies.p3().iter().collect();
            plant_snare(
                &mut commands,
                &mut cues,
                pos,
                aim_v,
                player_e,
                &player,
                &live_tangles,
            );
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
            cue(
                &mut cues,
                if player.throne_butt {
                    "sndPopPopUpg"
                } else {
                    "sndPopPop"
                },
                1.0,
                0.0,
            );
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
                persist.p2().area,
                persist.p2().gen_seed,
            );
            if was_cursed {
                // GML curse eat: self-hit 7 + 10 `Curse` motes.
                health.hp -= 7;
                let mut rng = rand::rng();
                for _ in 0..10 {
                    let off = Vec2::new(rng.random_range(-8.0..8.0), rng.random_range(-8.0..8.0));
                    spawn_pickup(
                        &mut commands,
                        &catalog,
                        PickupKind::Curse,
                        pos + off,
                        0,
                        false,
                    );
                }
            }
            if wep_rads > 0 {
                // GML `scrRadDrop(x, y, 15, false, false)`.
                spawn_rad_burst(&mut commands, &catalog, pos, 15);
            }

            let unlocked = check_progress_unlocks(&mut persist.p0(), 0, 0, false, true, false);
            for race in unlocked {
                persist.p1().0 = true;
                toast.show(&format!(
                    "UNLOCKED {}",
                    character_def(race).name.to_ascii_uppercase()
                ));
            }
            let tb = player.throne_butt;
            cue(
                &mut cues,
                if tb { "sndRobotEatUpg" } else { "sndRobotEat" },
                1.0,
                0.0,
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
                cue(&mut cues, "sndCursedReminder", 1.0, 0.0);
                return;
            }
            let aim_angle = aim_v.y.atan2(aim_v.x);
            let mut rng = rand::rng();
            let fling = aim_angle + rng.random_range(-2.0f32.to_radians()..2.0f32.to_radians());
            let determination = matches!(player.ultra, Some(UltraMutationId::ChickenDetermination));
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
            cue(&mut cues, "sndChickenThrow", 1.0, 0.0);
        }
        AbilityKind::SpawnAlly => {
            let has_ally = !walls_and_allies.p1().is_empty();
            let cost = if has_ally || matches!(player.ultra, Some(UltraMutationId::RebelRiot)) {
                2
            } else {
                1
            };
            if health.hp <= cost {
                return;
            }
            if player.throne_butt {
                cue(&mut cues, "sndSpawnSuperAlly", 1.0, 0.0);
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
                cue(&mut cues, "sndAllySpawn", 1.0, 0.2);
            }
        }
        AbilityKind::HorrorBeam => {
            let cost = horror_charge.map_or(1.0, |c| c.time + 1.0).floor() as u32;
            if player.rads < cost {
                cue(&mut cues, "sndHorrorEmpty", 1.0, 0.0);
                return;
            }
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
            cue(&mut cues, "sndHorrorBeam", 1.0, 0.0);
        }
        AbilityKind::PortalStrike => {
            if player.rogue_ammo == 0 {
                cue(&mut cues, "sndPortalStrikeEmpty", 1.0, 0.0);
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
            cue(&mut cues, "sndRogueAim", 1.0, 0.0);
        }
        AbilityKind::RocketBarrage => {
            let slot = inv.ammo_mut(AmmoKind::Explosives);
            if *slot < 3 {
                return;
            }
            *slot -= 3;
            let wall_shapes: Vec<(Vec2, Vec2)> = walls_and_allies
                .p0()
                .iter()
                .map(|(_, _, wall_pos)| (wall_pos.0, Vec2::splat(WALL_PX)))
                .collect();
            let speed = (2.0 + f32::from(player.throne_butt)) * 30.0;
            let mut rng = rand::rng();
            for _ in 0..3 {
                let angle = rng.random_range(0.0..std::f32::consts::TAU);
                let dir = Vec2::from_angle(angle);
                let mut spawn_pos = pos;
                move_contact_solid(
                    &mut spawn_pos,
                    dir * 14.0,
                    BIG_DOG_MISSILE_RADIUS,
                    &wall_shapes,
                    None,
                );
                let mut missile = BigDogMissileState::new(player_e);
                missile.throne_butt = player.throne_butt;
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Team::Player,
                    Health {
                        hp: BIG_DOG_MISSILE_HP,
                        max: BIG_DOG_MISSILE_HP,
                        invuln: GTimer::disarmed(),
                    },
                    Hitbox {
                        radius: BIG_DOG_MISSILE_RADIUS,
                    },
                    NextHurt::default(),
                    missile,
                    GmlImage::new("images/sprScrapBossMissileIdle.png", 4, 0.4),
                    NativeAngle(dir.y.atan2(dir.x)),
                    NativeDepth(-2.0),
                    Velocity(dir * speed),
                    Pos(spawn_pos),
                ));
            }
            trauma.add(0.25);
            cue(&mut cues, "sndBigDogMissile", 1.0, 0.0);
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
        }
        AbilityKind::ToxicPuke => {}
        AbilityKind::CuzSwap => {
            // GML Cuz (`scrPowers.gml:430-454`): ring of
            // `20*(1+Emotional)` tears + `spr_cry` swap.
            if player.cuz_ammo == 0 {
                cue(&mut cues, "sndCuzCryAttackNoAmmo", 1.0, 0.0);
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
                1.0,
                0.0,
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

pub fn tick_big_dog_missiles(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Res<FloorMask>,
    frame: Res<CurrentFrame>,
    save: Option<Res<SaveData>>,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
    creators: Query<(&AimDir, &Player), With<Player>>,
    mut sets: ParamSet<(
        Query<
            (
                Entity,
                &mut BigDogMissileState,
                &mut Health,
                &mut Velocity,
                &mut Pos,
                Option<&mut GmlImage>,
                Option<&mut NativeAngle>,
            ),
            (With<BigDogMissileState>, Without<Prop>),
        >,
        Query<
            (
                Entity,
                &Team,
                &Pos,
                &Hitbox,
                &Enemy,
                &mut Health,
                Option<&mut NextHurt>,
                Option<&mut Velocity>,
            ),
            (
                With<Enemy>,
                Without<Prop>,
                Without<Projectile>,
                Without<BigDogMissileState>,
            ),
        >,
    )>,
) {
    let dt = time.delta_secs;
    let solids: Vec<(Vec2, Vec2)> = props
        .iter()
        .map(|(_, prop, pos)| (pos.0, prop.size))
        .collect();
    let spawn_explosion = |commands: &mut Commands, at: Vec2| {
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            Explosion {
                timer: GTimer::from_seconds(0.05, TimerMode::Once),
                radius: 32.0,
                damage: 0,
                team: Team::Player,
                hits_player: false,
                source: None,
            },
            Pos(at),
        ));
    };
    let mut rng = rand::rng();
    let missile_entities: Vec<Entity> = {
        let missiles = sets.p0();
        missiles
            .iter()
            .map(|(entity, _, _, _, _, _, _)| entity)
            .collect()
    };
    for entity in missile_entities {
        let missile_pos = {
            let mut missiles = sets.p0();
            let Ok((_, mut state, health, mut vel, mut pos, mut image, mut angle)) =
                missiles.get_mut(entity)
            else {
                continue;
            };
            state.fuse.tick(dt);
            state.hurt_timer.tick(dt);
            if state.fuse.just_finished() || health.hp <= 0 {
                spawn_explosion(&mut commands, pos.0);
                cue(&mut cues, "sndExplosion", 1.0, 0.0);
                commands.entity(entity).despawn();
                continue;
            }

            let hurt_finished = state.hurt_timer.just_finished();
            if let Some(image) = image.as_deref_mut()
                && (image.finished || hurt_finished)
            {
                *image = GmlImage::new("images/sprScrapBossMissileIdle.png", 4, 0.4);
            }

            let previous_pos = pos.0;
            let (aim, throne_butt) = creators
                .get(state.creator)
                .map(|(aim, player)| (aim.0, state.throne_butt || player.throne_butt))
                .unwrap_or((vel.0.normalize_or_zero(), state.throne_butt));
            gml_motion_add_clamp(&mut vel.0, aim, 0.1, 1000.0, dt);
            let speed = (2.0 + f32::from(throne_butt)) * 30.0;
            if vel.0.length_squared() > 1e-6 {
                vel.0 = vel.0.normalize() * speed;
            } else {
                vel.0 = aim.normalize_or_zero() * speed;
            }
            if move_bounce_solid(
                &mut pos.0,
                &mut vel.0,
                BIG_DOG_MISSILE_RADIUS,
                dt,
                &solids,
                Some(&mask),
                true,
            )
            .is_some()
            {
                spawn_explosion(&mut commands, pos.0);
                commands.entity(entity).despawn();
                continue;
            }
            if let Some(angle) = angle.as_deref_mut() {
                angle.0 = vel.0.y.atan2(vel.0.x);
            }
            state.trail_timer = state.trail_timer.wrapping_add(1) % 4;
            if rng.random::<f32>() < 0.25 {
                spawn_native_smoke_mote(
                    &mut commands,
                    save.as_ref().is_some_and(|save| save.settings.particles),
                    previous_pos,
                    Vec2::ZERO,
                    rng.random_range(0.5..1.5),
                );
            }
            pos.0
        };

        let hit = {
            let mut targets = sets.p1();
            let mut hit = false;
            for (
                _,
                target_team,
                target_pos,
                target_hitbox,
                _,
                mut target_health,
                mut next_hurt,
                mut target_velocity,
            ) in &mut targets
            {
                if *target_team != Team::Enemy
                    || target_health.hp <= 0
                    || missile_pos.distance(target_pos.0)
                        > BIG_DOG_MISSILE_RADIUS + target_hitbox.radius
                {
                    continue;
                }
                if next_hurt.as_ref().is_some_and(|next| next.0 > frame.0) {
                    continue;
                }
                target_health.hp -= BIG_DOG_MISSILE_DAMAGE;
                if let Some(next) = next_hurt.as_deref_mut() {
                    next.0 = frame.0 + 5;
                }
                if let Some(mut velocity) = target_velocity.take() {
                    velocity.0 += (target_pos.0 - missile_pos).normalize_or_zero() * 120.0;
                }
                hit = true;
                break;
            }
            hit
        };
        if !hit {
            continue;
        }

        let mut missiles = sets.p0();
        let Ok((_, mut state, mut health, _, pos, _, _)) = missiles.get_mut(entity) else {
            continue;
        };
        health.hp -= 3;
        state.hit_enemy();
        if health.hp <= 0 {
            spawn_explosion(&mut commands, pos.0);
            commands.entity(entity).despawn();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: headless parity for the combat block
// ---------------------------------------------------------------------------
