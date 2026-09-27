//! Enemy AI. Ported from the bevy reference `game/enemies.rs` (`enemy_ai`
/// and its spawn/fire/tick helpers) with positions as [`Pos`] (`Vec2`)
/// instead of `Transform.translation` (`Vec3`).
///
/// Render split: `Sprite`/`Anchor`/`Transform.rotation` writes, hurt/fire
/// strip swaps, `Juice::pop_in`, and `VfxSpawner` bursts stay out — the
/// render phase resolves visuals from sim state. Gameplay effects are
/// kept: movement impulses, projectile spawns with full combat traits
/// (`typ`/fade/friction/bounce/split/homing/fuse), pending-spawn queues,
/// trauma, hitstop, toasts, and audio stems routed as [`AudioCue`]s
/// through the [`Queue`] (the bevy build played them via direct asset
/// loads inside the 16-param system; here they queue like every other
/// sim system).
///
/// Timer adaptation: bevy `Timer` -> [`GTimer`]; `tick()` returns `()`,
/// then `just_finished()`/`finished()` are queried. The bevy
/// `ready_timer()` (finished from birth, silent until re-armed) has no
/// direct `GTimer` equivalent — `GTimer::disarmed()` reports
/// `just_finished()` on *every* tick — so the local [`ready_timer`]
/// double-ticks a 10 ms `Once` timer into the same observable state
/// (finished, not just-finished).
///
/// `enemy_ai` carries 14 params (under the bevy_ecs 16-param cap), so no
/// record-split is needed; the verbatim per-kind ticks are separate
/// systems, and the Inspector tail lives in [`tick_bigmaggot_inspector`].
use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bevy_ecs::system::EntityCommands;
use rand::RngExt;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::anim::{SpriteAnim, derive_hurt_path, derive_walk_path};
use crate::audio::AudioCue;
use crate::combat::{
    PendingEnemySpawn, apply_birth_overrides, queue_enemy_spawn, queue_enemy_spawn_birth,
};
use crate::comps_a::{
    ARENA_H, ARENA_W, BossIntro, BouncesLeft, CurrentFrame, DamageSource, Euphoria, FloorMask,
    GameCleanup, GrenadeFuse, Health, HeavyHeart, Hitbox, Homing, LevelCleanup, NextHurt, Player,
    Projectile, ProjectileFade, ProjectileFriction, ProjectileTyp, Run, ScarierFace,
    ShellWallBounce, SplitOnDeath, Team, Toast, Velocity, WallCell, WallTile, apply_gml_friction,
    gml_motion_add_clamp,
};
use crate::comps_b::{
    BossBrain, Corpse, CorpseCollision, EliteBlocker, Enemy, EnemyBrain, FxAngle, GmlImage,
    HitWarning, HurtAnim, IdpdShieldUnit, IdpdVanBrain, LilHunterDie, MaggotSpawnCharge,
    MaggotSpawnInternalDrain, MomShot, NativeAngle, NativeDepth, PendingDelayedBoss, Pickup,
    PickupLifetime, PopoNadeM, PortalClear, Prop, ProtoGuardian, SCRAP_BOSS_MISSILE_RADIUS,
    ScrapBossMissileState, ShieldFollower, StaticFx, ThroneBall, ToxicGasState,
};
use crate::data::{AreaId, EnemyKind, SplitDef};
use crate::effects::{HitStop, spawn_burst};
use crate::enemy_data::{EnemyDef, enemy_def};
use crate::msg::Queue;
use crate::spatial::{
    Pos, clamp_to_arena, move_bounce_solid, potential_step_solid, resolve_prop_collision,
    solid_contact,
};
use crate::time::{GTimer, TimerMode};

fn scarier_spawn_hp(kind: EnemyKind, base_hp: i32, loops: u32) -> i32 {
    let l = loops as f32;
    let hp = match kind {
        EnemyKind::BigBandit | EnemyKind::BigBanditLoop => 100.0 * (1.0 + l / 3.0),
        EnemyKind::BigDog | EnemyKind::BigDogLoop => 300.0 * (1.0 + l / 1.2),
        EnemyKind::Throne => 1500.0 * (1.0 + l / 3.0),
        EnemyKind::ThroneII => 600.0 * (1.0 + l / 3.0),
        EnemyKind::Hyper => 550.0 * (1.0 + l / 3.0),
        EnemyKind::Technomancer => 350.0 * (1.0 + l / 3.0),
        EnemyKind::LilHunter | EnemyKind::LilHunterLoop => 140.0 * (1.0 + l / 3.0),
        EnemyKind::FrogQueen => 490.0 * (1.0 + l / 3.0),
        EnemyKind::Captain => 1100.0 * (1.0 + l / 3.0),
        EnemyKind::ProtoStatue => 120.0,
        _ => base_hp as f32 * (1.0 + l / 20.0),
    };
    (hp * 0.8).floor() as i32
}

/// Floor-scaled HP multiplier (bevy `world::difficulty_multiplier`
/// parity: +5% per loop, +1.5% per route floor). Kept for the speed law
/// only; HP follows the GML spawn law below (no within-loop scaling).
pub fn difficulty_multiplier(floor: u32) -> f32 {
    let loop_n = ((floor.max(1) - 1) / 15) as f32;
    let rf = ((floor.max(1) - 1) % 15) as f32;
    1.0 + loop_n * 0.05 + rf * 0.015
}

/// GML spawn-HP law verbatim (`enemy/Create_0`: every enemy scales
/// `max_hp *= 1 + loops/20`; bosses override with their own formulas,
/// single-player values). `base_hp` is the table (loop-0) value.
///
/// Boss laws (all `1 + loops/3` except ScrapBoss `/1.2` and
/// ProtoStatue flat 120): BigBandit, Throne, ThroneII, Hyper+Technomancer
/// (coop `((pc/2)+0.5)` factor is 1.0 solo — GML `/` is float division,
/// so `(1/2)+0.5 = 1.0`), LilHunter, FrogQueen, Last (`Last`: base 1100).
/// Single-player-only bosses with no loop term stay flat: YV (700 +
/// coop-only scaling). `Mom`/`Captain`/`OldGuardian`/`PalaceGuardian`
/// have no GML object (spawn-table-only kinds); they ride the default
/// `/20` law like every other non-boss.
pub fn spawn_hp(kind: EnemyKind, base_hp: i32, loops: u32) -> i32 {
    let l = loops as f32;
    // GML writes each object's own `Create_0` expression first, then
    // `enemy/Create_0:7` multiplies *every* enemy by `1 + loops / 20`.
    // `ceil` only where the object's own line uses it; GML keeps `hp` a real
    // otherwise.
    let (hp, ceil) = match kind {
        EnemyKind::BigBandit | EnemyKind::BigBanditLoop => (100.0 * (1.0 + l / 3.0), true),
        EnemyKind::BigDog | EnemyKind::BigDogLoop => (300.0 * (1.0 + l / 1.2), true),
        EnemyKind::Throne => (1500.0 * (1.0 + l / 3.0), false),
        EnemyKind::ThroneII => (600.0 * (1.0 + l / 3.0), false),
        // `HyperCrystal`: `550 * ((player_count / 2) + 0.5)` -> 550 solo.
        EnemyKind::Hyper => (550.0 * (1.0 + l / 3.0), false),
        // `TechnoMancer`: `350 * ((player_count / 2) + 0.5)` -> 350 solo.
        EnemyKind::Technomancer => (350.0 * (1.0 + l / 3.0), false),
        EnemyKind::LilHunter | EnemyKind::LilHunterLoop => (140.0 * (1.0 + l / 3.0), false),
        EnemyKind::FrogQueen => (490.0 * (1.0 + l / 3.0), true),
        // `Last`: `1100 * (1 + loops / 3)`.
        EnemyKind::Captain => (1100.0 * (1.0 + l / 3.0), false),
        // `ProtoStatue`: `120 * (1 + loops / 10)`.
        EnemyKind::ProtoStatue => (120.0 * (1.0 + l / 10.0), false),
        // `MeleeFake`'s parent is `prop`, not `enemy`, so it never picks up
        // the universal `1 + loops / 20`.
        EnemyKind::MeleeFake => return base_hp.max(1),
        _ => (base_hp as f32, false),
    };
    let scaled = hp * (1.0 + l / 20.0);
    let out = if ceil { scaled.ceil() } else { scaled };
    out.round().max(1.0) as i32
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EnemySpawnContext {
    pub subarea: u32,
    pub blood_crown: bool,
    pub scarier_face: bool,
    pub heavy_heart: bool,
}

/// Full enemy spawn: base bundle from [`crate::setup::spawn_enemy`]
/// (cleanup markers, `Team`, `Pos`, `Velocity`, `Hitbox`, table `Enemy`)
/// plus difficulty/face/heart scaling, the randomized [`EnemyBrain`]
/// attack/strafe/gunangle state, boss/IDPD brains, and a [`SpriteAnim`]
/// seed when the catalog carries the idle strip (so `tick_fire_anims`
/// can resolve [`crate::comps_b::FireAnim`] markers from
/// [`show_enemy_fire`]; visual strips themselves stay render-side).
pub fn spawn_enemy(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: EnemyKind,
    pos: glam::Vec2,
    difficulty: f32,
    scarier_face: bool,
    heavy_heart: bool,
    loops: u32,
) -> Entity {
    spawn_enemy_with_context(
        commands,
        catalog,
        kind,
        pos,
        difficulty,
        scarier_face,
        heavy_heart,
        loops,
        EnemySpawnContext::default(),
    )
}

pub fn spawn_enemy_with_context(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: EnemyKind,
    pos: glam::Vec2,
    difficulty: f32,
    scarier_face: bool,
    heavy_heart: bool,
    loops: u32,
    context: EnemySpawnContext,
) -> Entity {
    let mut rng = rand::rng();
    spawn_enemy_impl(
        commands,
        catalog,
        kind,
        pos,
        difficulty,
        scarier_face,
        heavy_heart,
        loops,
        context,
        true,
        None,
        &mut rng,
    )
}

fn spawn_enemy_impl(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: EnemyKind,
    pos: glam::Vec2,
    difficulty: f32,
    scarier_face: bool,
    heavy_heart: bool,
    loops: u32,
    context: EnemySpawnContext,
    give_kill: bool,
    birth: Option<&PendingEnemySpawn>,
    rng: &mut impl RngExt,
) -> Entity {
    let def = enemy_def(kind);
    let e = crate::setup::spawn_enemy(commands, kind, pos, true);

    // GML HP law (loop scaling); scarier-face 0.8 floor kept.
    let scarier_face = scarier_face || context.scarier_face;
    let hp = if scarier_face {
        scarier_spawn_hp(kind, def.hp, loops)
    } else {
        spawn_hp(kind, def.hp, loops)
    };
    let speed = if matches!(
        kind,
        EnemyKind::Bandit
            | EnemyKind::SnowBandit
            | EnemyKind::Maggot
            | EnemyKind::MaggotSpawn
            | EnemyKind::BigMaggot
            | EnemyKind::JungleFly
            | EnemyKind::FiredMaggot
            | EnemyKind::Scorpion
            | EnemyKind::GoldScorpion
            | EnemyKind::Sniper
            | EnemyKind::JungleBandit
            | EnemyKind::MeleeBandit
            | EnemyKind::Ballguy
    ) {
        if kind == EnemyKind::Ballguy { 1.0 } else { 0.0 }
    } else {
        def.speed * (0.9 + 0.02 * difficulty)
    };
    let weapon_chance = if heavy_heart || context.heavy_heart {
        def.weapon_chance + 9
    } else {
        def.weapon_chance
    };

    let mut ec = commands.entity(e);
    ec.insert(Enemy {
        kind,
        score: def.score,
        touch_damage: def.touch_damage,
        rad_drop: def.rad_drop,
        drop_chance: def.drop_chance,
        weapon_chance,
        give_kill: give_kill && !matches!(kind, EnemyKind::FastRat | EnemyKind::ScrapBossMissile),
    });
    ec.insert(NextHurt::default());
    ec.insert(Hitbox { radius: def.radius });
    let initial_gunangle = match kind {
        EnemyKind::Bandit
        | EnemyKind::SnowBandit
        | EnemyKind::JungleFly
        | EnemyKind::Scorpion
        | EnemyKind::GoldScorpion
        | EnemyKind::Sniper
        | EnemyKind::JungleBandit
        | EnemyKind::MeleeBandit => rng.random_range(0.0..std::f32::consts::TAU),
        EnemyKind::Maggot
        | EnemyKind::MaggotSpawn
        | EnemyKind::BigMaggot
        | EnemyKind::FiredMaggot
        | EnemyKind::Ballguy => 0.0,
        _ => rng.random_range(0.0..std::f32::consts::TAU),
    };
    let initial_heading = if kind == EnemyKind::Ballguy {
        (pos - glam::Vec2::new(10016.0, 10016.0))
            .y
            .atan2((pos - glam::Vec2::new(10016.0, 10016.0)).x)
    } else {
        0.0
    };
    let initial_velocity = match kind {
        EnemyKind::Ballguy => glam::Vec2::from_angle(initial_heading) * 30.0,
        EnemyKind::ScrapBossMissile => {
            glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU)) * 60.0
        }
        _ => glam::Vec2::ZERO,
    };
    let mut health = Health {
        hp,
        max: hp,
        invuln: ready_timer(),
    };
    let mut velocity = Velocity(initial_velocity);
    if let Some(birth) = birth {
        apply_birth_overrides(&mut velocity, &mut health, birth);
    }
    ec.insert(health);
    ec.insert(velocity);
    if kind == EnemyKind::ScrapBossMissile {
        ec.insert(ScrapBossMissileState::new(loops));
    }
    // GML `Create_0` `alarm[1]`, in frames. `random(n)` yields 0..n-1.
    let attack_frames = match kind {
        EnemyKind::MaggotSpawn | EnemyKind::FiredMaggot => 0.0,
        EnemyKind::Rat | EnemyKind::Bandit | EnemyKind::SnowBandit
        | EnemyKind::JungleBandit => 30.0 + rng.random_range(0.0..90.0),
        EnemyKind::FastRat | EnemyKind::Ratking => 1.0 + rng.random_range(0.0..90.0),
        EnemyKind::RobotGuard => 80.0,
        EnemyKind::Maggot | EnemyKind::RadMaggot | EnemyKind::FireBaller
        | EnemyKind::SuperFireBaller => 10.0 + rng.random_range(0.0..10.0),
        EnemyKind::BigMaggot => 45.0 + rng.random_range(0.0..10.0),
        EnemyKind::JungleFly => 50.0 + rng.random_range(0.0..10.0),
        EnemyKind::Scorpion | EnemyKind::GoldScorpion => 30.0 + rng.random_range(0.0..90.0),
        EnemyKind::Sniper => 60.0 + rng.random_range(0.0..90.0),
        EnemyKind::MeleeBandit | EnemyKind::Assassin => 90.0 + rng.random_range(0.0..90.0),
        EnemyKind::Ballguy => 40.0 + rng.random_range(0.0..40.0),
        EnemyKind::Turret => 60.0 + rng.random_range(0.0..60.0),
        EnemyKind::SnowTank => 30.0 + rng.random_range(0.0..10.0),
        EnemyKind::GoldSnowtank => 120.0 + rng.random_range(0.0..10.0),
        EnemyKind::LaserCrystal | EnemyKind::LightningCrystal => 50.0 + rng.random_range(0.0..90.0),
        EnemyKind::InvLaserCrystal => 30.0 + rng.random_range(0.0..90.0),
        EnemyKind::Guardian | EnemyKind::CrownGuardian => 40.0 + rng.random_range(0.0..10.0),
        EnemyKind::DogGuardian => 120.0 + rng.random_range(0.0..10.0),
        // `20 + random(10)` for the freaks and the ExploGuardian.
        EnemyKind::Freak
        | EnemyKind::ExploFreak
        | EnemyKind::RhinoFreak
        | EnemyKind::PopoFreak
        | EnemyKind::ExploGuardian => 20.0 + rng.random_range(0.0..10.0),
        // IDPD `30 + random(15)`; `EliteGrunt` is a fixed 25.
        EnemyKind::IdpdElite => 25.0,
        EnemyKind::IdpdGrunt
        | EnemyKind::IdpdShield
        | EnemyKind::IdpdInspector
        | EnemyKind::EliteInspector
        | EnemyKind::EliteShielder => 30.0 + rng.random_range(0.0..15.0),
        // `30 + random(90)` covers Gator, BuffGator, Raven, Spider,
        // InvSpider, BoneFish, Turtle, Molefish, Molesarge, Jock and
        // Necromancer.
        EnemyKind::Gator
        | EnemyKind::BuffGator
        | EnemyKind::Raven
        | EnemyKind::Spider
        | EnemyKind::InvSpider
        | EnemyKind::BoneFish
        | EnemyKind::Turtle
        | EnemyKind::Molefish
        | EnemyKind::Molesarge
        | EnemyKind::Jock
        | EnemyKind::Necromancer => 30.0 + rng.random_range(0.0..90.0),
        EnemyKind::Crab => 50.0 + rng.random_range(0.0..90.0),
        EnemyKind::Salamander => 60.0 + rng.random_range(0.0..90.0),
        EnemyKind::Mimic | EnemyKind::SuperMimic | EnemyKind::WepMimic => {
            90.0 + rng.random_range(0.0..150.0)
        }
        EnemyKind::FrogEgg => 120.0,
        EnemyKind::SuperFrog => 40.0 + rng.random_range(0.0..40.0),
        _ => def.attack_cooldown * 30.0 * rng.random_range(0.5..1.5),
    };
    let attack = if matches!(kind, EnemyKind::MaggotSpawn | EnemyKind::FiredMaggot) {
        GTimer::disarmed()
    } else {
        GTimer::from_seconds(attack_frames / 30.0, TimerMode::Once)
    };
    ec.insert(EnemyBrain {
        speed,
        accel: def.accel,
        preferred_range: def.preferred_range,
        shoot_range: def.shoot_range,
        attack,
        fire_alarm: GTimer::from_seconds(def.attack_cooldown, TimerMode::Once),
        burst_left: 0,
        burst_timer: ready_timer(),
        dash: 0.0,
        strafe_dir: if rng.random_bool(0.5) { 1.0 } else { -1.0 },
        strafe_timer: GTimer::from_seconds(rng.random_range(0.8..1.6), TimerMode::Once),
        melee: ready_timer(),
        wkick: 0.0,
        walk: 0.0,
        slash_delay: 0.0,
        ammo: match kind {
            EnemyKind::Scorpion | EnemyKind::GoldScorpion => 10,
            EnemyKind::JungleFly => 3,
            EnemyKind::IdpdGrunt => 2,
            EnemyKind::IdpdInspector => 4,
            EnemyKind::IdpdElite => 3,
            EnemyKind::Jock => 5,
            _ => 0,
        },
        gunangle: initial_gunangle,
        heading: initial_heading,
        rage: 0.0,
        fire: if kind == EnemyKind::JungleFly { 10 } else { 0 },
        friction: if kind == EnemyKind::FiredMaggot {
            0.0
        } else {
            0.4
        },
        close: false,
        wepangle: if kind == EnemyKind::MeleeBandit {
            if rng.random_bool(0.5) { -140.0 } else { 140.0 }
        } else {
            0.0
        },
        wepflip: 1.0,
        weapon_alarm: 0.0,
        burrow_state: 0,
        burrow_alarm0: 0.0,
        burrow_alarm1: 0.0,
        burrow_angle: 0.0,
        sniper_aiming: false,
        maggot_spawn_charging: false,
        maggot_spawn_charge_ticks: 0.0,
        maggot_spawn_facing: 1.0,
    });
    if def.boss {
        ec.insert(BossBrain::new(kind, pos));
    }
    match kind {
        EnemyKind::IdpdVan => {
            ec.insert((IdpdVanBrain::default(), IdpdShieldUnit));
        }
        EnemyKind::IdpdShield => {
            ec.insert(IdpdShieldUnit);
        }
        EnemyKind::ProtoStatue => {
            ec.insert(ProtoGuardian::default());
        }
        _ => {}
    }
    // Bevy parity: every enemy carries its strip table (`EnemySprites`)
    // plus the seeded idle `SpriteAnim`; the switch/hurt/fire systems
    // resolve walk/hurt/fire strips from the table each tick.
    ec.insert(crate::comps_b::EnemySprites {
        idle: def.sprite,
        walk: derive_walk_path(def.sprite),
        hurt: derive_hurt_path(def.sprite),
        charge: crate::anim::derive_charge_path(def.sprite),
    });
    // Seed the idle strip only (walk/hurt/fire strips resolve
    // renderer-side); without a catalog entry no anim rides along and
    // `show_enemy_fire` becomes a silent no-op for this enemy.
    if let Some(anim_def) = catalog.def(def.sprite) {
        ec.insert(SpriteAnim::new(def.sprite, anim_def));
    }
    drop(ec);
    if kind == EnemyKind::SuperFrog {
        for _ in 0..10 {
            let angle = rng.random_range(0.0..std::f32::consts::TAU);
            let speed = rng.random_range(0.2..1.7) * 30.0;
            let mut gas = ToxicGasState::new();
            gas.grow_speed = 0.003 + rng.random_range(0.0..0.002);
            gas.rot =
                (1.0 + rng.random_range(0.0..=3.0)) * if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            spawn_toxic_gas(commands, pos, glam::Vec2::from_angle(angle) * speed, gas);
        }
    }
    if kind == EnemyKind::Scorpion {
        let morph_limit = if context.blood_crown {
            30.0 * 0.7
        } else {
            30.0
        };
        let morph_roll = rng.random_range(0.0..morph_limit);
        if morph_roll < 1.0 + loops as f32 * 5.0 && context.subarea > 1 {
            let gold = spawn_enemy_impl(
                commands,
                catalog,
                EnemyKind::GoldScorpion,
                pos,
                difficulty,
                scarier_face,
                heavy_heart,
                loops,
                context,
                true,
                None,
                rng,
            );
            commands.entity(e).despawn();
            return gold;
        }
    }
    e
}

/// Thin wrapper matching the bevy call-site order.
pub fn spawn_enemy_at(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: EnemyKind,
    pos: glam::Vec2,
    difficulty: f32,
    scarier_face: bool,
    heavy_heart: bool,
    loops: u32,
    context: EnemySpawnContext,
) -> Entity {
    spawn_enemy_with_context(
        commands,
        catalog,
        kind,
        pos,
        difficulty,
        scarier_face,
        heavy_heart,
        loops,
        context,
    )
}

pub fn flush_pending_enemy_spawns(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    run: Res<Run>,
    scarier: Res<ScarierFace>,
    heavy_heart: Res<HeavyHeart>,
    pending: Query<(Entity, &PendingEnemySpawn)>,
) {
    let context = EnemySpawnContext {
        subarea: run.floor_in_area,
        blood_crown: run.blood_crown,
        scarier_face: scarier.0,
        heavy_heart: heavy_heart.0,
    };
    let mut rng = rand::rng();
    for (entity, spawn) in pending.iter() {
        spawn_enemy_impl(
            &mut commands,
            &catalog,
            spawn.kind,
            spawn.pos,
            spawn.difficulty,
            false,
            heavy_heart.0,
            spawn.loops,
            context,
            spawn.give_kill,
            Some(spawn),
            &mut rng,
        );
        commands.entity(entity).despawn();
    }
}

/// Random arena point at least `min_from_center` from the origin
/// (bevy parity, including the corner fallback).
pub fn random_spawn_pos(rng: &mut impl RngExt, min_from_center: f32) -> glam::Vec2 {
    for _ in 0..64 {
        let x = rng.random_range(-ARENA_W / 2.0 + 80.0..ARENA_W / 2.0 - 80.0);
        let y = rng.random_range(-ARENA_H / 2.0 + 80.0..ARENA_H / 2.0 - 80.0);
        let p = glam::Vec2::new(x, y);
        if p.length() >= min_from_center {
            return p;
        }
    }
    glam::Vec2::new(ARENA_W / 2.0 - 80.0, 0.0)
}

/// Finished-from-birth, silent-until-re-armed timer (bevy `ready_timer`
/// parity — see module docs for why `GTimer::disarmed()` is wrong here).
fn ready_timer() -> GTimer {
    let mut t = GTimer::from_seconds(0.01, TimerMode::Once);
    t.tick(0.01);
    t.tick(0.0);
    t
}

/// Enemy telegraph cue through the sim audio queue (bevy
/// `play_enemy_cue` parity: 0.6 vol, 0.05 variance; the bevy build
/// loaded `audio/{stem}.wav` directly to dodge the 16-param cap — the
/// queue carries the stem name instead and the audio layer resolves it).
fn enemy_cue(cues: &mut Queue<AudioCue>, stem: &'static str) {
    cues.push(AudioCue {
        name: stem,
        volume: 0.6,
        variance: 0.05,
    });
}

/// Wall-aware sight check (bevy parity: 8 px samples, 16 px tile-center
/// recheck, arena-exterior samples ignored).
fn has_line_of_sight(from: glam::Vec2, to: glam::Vec2, mask: &FloorMask) -> bool {
    let dir = to - from;
    let dist = dir.length();
    if dist < 1.0 {
        return true;
    }
    let steps = (dist / 8.0).ceil().max(4.0) as usize;
    for i in 1..steps {
        let t = i as f32 / steps as f32;
        let p = from + dir * t;
        let tile_check = glam::Vec2::new(
            (p.x / 16.0).floor() * 16.0 + 8.0,
            (p.y / 16.0).floor() * 16.0 + 8.0,
        );
        if !mask.is_walkable(p) && !mask.is_walkable(tile_check) {
            if p.x.abs() < ARENA_W / 2.0 && p.y.abs() < ARENA_H / 2.0 {
                return false;
            }
        }
    }
    true
}

#[inline]
fn sync_heading(brain: &mut EnemyBrain, vel: &Velocity) {
    if vel.0.length_squared() > 0.000001 {
        brain.heading = vel.0.y.atan2(vel.0.x);
    }
    brain.speed = vel.0.length() / 30.0;
}

#[inline]
fn set_gml_direction(brain: &mut EnemyBrain, vel: &mut Velocity, angle: f32) {
    brain.heading = angle;
    let speed = vel.0.length();
    if speed > 0.000001 {
        vel.0 = glam::Vec2::from_angle(angle) * speed;
    }
    sync_heading(brain, vel);
}

#[inline]
fn set_gml_speed(brain: &mut EnemyBrain, vel: &mut Velocity, speed_f: f32) {
    let speed = speed_f * 30.0;
    if speed > 0.000001 {
        vel.0 = glam::Vec2::from_angle(brain.heading) * speed;
    } else {
        vel.0 = glam::Vec2::ZERO;
    }
    sync_heading(brain, vel);
}

#[inline]
fn add_gml_motion(brain: &mut EnemyBrain, vel: &mut Velocity, angle: f32, impulse_f: f32, dt: f32) {
    gml_motion_add_clamp(
        &mut vel.0,
        glam::Vec2::from_angle(angle),
        impulse_f,
        1000.0,
        dt,
    );
    sync_heading(brain, vel);
}

#[inline]
fn cap_gml_speed(brain: &mut EnemyBrain, vel: &mut Velocity, cap_f: f32) {
    let cap = cap_f * 30.0;
    if vel.0.length() > cap {
        vel.0 = vel.0.normalize() * cap;
    }
    sync_heading(brain, vel);
}

#[inline]
fn walk_step(brain: &mut EnemyBrain, vel: &mut Velocity, impulse_f: f32, cap_f: f32, dt: f32) {
    if brain.walk <= 0.0 {
        return;
    }
    let angle = brain.heading;
    add_gml_motion(brain, vel, angle, impulse_f, dt);
    cap_gml_speed(brain, vel, cap_f);
    brain.walk = (brain.walk - dt * 30.0).max(0.0);
}

#[inline]
fn tick_verbatim_cooldown(brain: &mut EnemyBrain, dt: f32) {
    brain.melee.tick(dt);
    brain.wkick = (brain.wkick - dt * 30.0).max(0.0);
}

fn has_verbatim_tick(kind: EnemyKind) -> bool {
    matches!(
        kind,
        EnemyKind::Bandit
            | EnemyKind::SnowBandit
            | EnemyKind::Maggot
            | EnemyKind::MaggotSpawn
            | EnemyKind::BigMaggot
            | EnemyKind::JungleFly
            | EnemyKind::FiredMaggot
            | EnemyKind::Scorpion
            | EnemyKind::GoldScorpion
            | EnemyKind::Sniper
            | EnemyKind::JungleBandit
            | EnemyKind::MeleeBandit
            | EnemyKind::Ballguy
    )
}

#[derive(Clone, Copy)]
enum EnemyWallLaw {
    Parent,
    PlainBounce,
    Slide,
    SlidePush,
}

fn wall_law(kind: EnemyKind) -> EnemyWallLaw {
    match kind {
        EnemyKind::Ballguy
        | EnemyKind::BigMaggot
        | EnemyKind::Scorpion
        | EnemyKind::GoldScorpion
        | EnemyKind::Sniper => EnemyWallLaw::PlainBounce,
        EnemyKind::JungleFly => EnemyWallLaw::Slide,
        EnemyKind::Maggot => EnemyWallLaw::SlidePush,
        _ => EnemyWallLaw::Parent,
    }
}

fn prop_shapes(props: &Query<(Entity, &Prop, &Pos), With<Prop>>) -> Vec<(glam::Vec2, glam::Vec2)> {
    props
        .iter()
        .map(|(_, prop, pos)| (pos.0, prop.size))
        .collect()
}

/// GML per-object speed ceiling in px/frame, from each object's `Other_10` /
/// `Step_0` `if (speed > N) speed = N`. `Spider`/`InvSpider` use the
/// `maxspeed` variable their `Alarm_1` sets (3 idle, 5 while chasing).
/// A `f32::INFINITY` cap means the object has no ceiling of its own.
fn gml_speed_cap(kind: EnemyKind) -> f32 {
    let frames = match kind {
        EnemyKind::RhinoFreak => 1.0,
        EnemyKind::Ratking => 2.0,
        EnemyKind::FireBaller => 2.0,
        EnemyKind::Mimic | EnemyKind::SuperMimic | EnemyKind::WepMimic => 2.0,
        EnemyKind::Guardian | EnemyKind::CrownGuardian => 0.6,
        EnemyKind::SuperFireBaller => 1.5,
        EnemyKind::LaserCrystal | EnemyKind::InvLaserCrystal => 1.5,
        EnemyKind::LightningCrystal => 1.8,
        EnemyKind::DogGuardian => 2.0,
        EnemyKind::Salamander => 2.5,
        EnemyKind::ExploGuardian | EnemyKind::SuperFrog => 2.5,
        EnemyKind::ExploFreak | EnemyKind::Gator | EnemyKind::BuffGator | EnemyKind::Bandit => 3.0,
        EnemyKind::Jock | EnemyKind::MeleeBandit | EnemyKind::Assassin => 3.0,
        EnemyKind::Necromancer => 3.0,
        EnemyKind::Raven | EnemyKind::JungleBandit => 3.5,
        EnemyKind::Molefish | EnemyKind::Molesarge => 3.5,
        EnemyKind::Freak | EnemyKind::PopoFreak | EnemyKind::BoneFish | EnemyKind::Rat => 4.0,
        EnemyKind::FastRat => 4.5,
        EnemyKind::Crab => 4.5,
        EnemyKind::Wolf => 5.0,
        EnemyKind::Turtle => 5.0,
        // `Spider`/`InvSpider` cap at `maxspeed`; the chase cap is applied by
        // the spider's own alarm, so 5 is the outer bound here.
        EnemyKind::Spider | EnemyKind::InvSpider => 5.0,
        // `SnowBot` runs at cap 3 and 8 while sliding (`spr_fire`).
        EnemyKind::RobotGuard => 8.0,
        _ => f32::INFINITY,
    };
    frames * crate::SIM_HZ as f32
}

/// GML `Other_10` walk law: `(motion_add(direction, impulse), speed cap)` in
/// px/frame, read off each object's own step handler.
fn gml_walk_law(kind: EnemyKind) -> (f32, f32) {
    match kind {
        EnemyKind::Gator
        | EnemyKind::BuffGator
        | EnemyKind::Bandit
        | EnemyKind::SnowBandit
        | EnemyKind::Jock
        | EnemyKind::Raven
        | EnemyKind::Rat
        | EnemyKind::BigRat
        | EnemyKind::Necromancer => (0.8, gml_speed_cap_frames(kind)),
        EnemyKind::FastRat => (0.8, 4.5),
        EnemyKind::Ratking => (0.5, 2.0),
        EnemyKind::BoneFish | EnemyKind::Molefish | EnemyKind::Molesarge => {
            (0.8, gml_speed_cap_frames(kind))
        }
        EnemyKind::Salamander => (2.0, 2.5),
        EnemyKind::Crab => (1.5, 4.5),
        EnemyKind::Spider | EnemyKind::InvSpider => (2.0, 5.0),
        EnemyKind::MeleeBandit | EnemyKind::Assassin => (2.0, 3.0),
        EnemyKind::Freak | EnemyKind::ExploFreak | EnemyKind::PopoFreak => {
            (gml_walk_impulse(kind), gml_speed_cap_frames(kind))
        }
        EnemyKind::RhinoFreak => (0.8, 1.0),
        EnemyKind::Turtle => (1.0, 5.0),
        EnemyKind::Wolf => (1.0, 5.0),
        EnemyKind::FireBaller | EnemyKind::SuperFireBaller | EnemyKind::SuperFrog => {
            (0.6, gml_speed_cap_frames(kind))
        }
        EnemyKind::SnowTank | EnemyKind::GoldSnowtank => (0.6, 1.5),
        EnemyKind::Guardian | EnemyKind::CrownGuardian => (0.6, 0.6),
        EnemyKind::DogGuardian => (0.4, 2.0),
        EnemyKind::ExploGuardian => (0.5, 2.5),
        EnemyKind::LaserCrystal | EnemyKind::LightningCrystal | EnemyKind::InvLaserCrystal => {
            (0.5, gml_speed_cap_frames(kind))
        }
        _ => (0.4, gml_speed_cap_frames(kind)),
    }
}

/// GML `motion_add(direction, N)` from an object's step handler.
fn gml_walk_impulse(kind: EnemyKind) -> f32 {
    match kind {
        EnemyKind::Freak | EnemyKind::PopoFreak => 0.55,
        EnemyKind::ExploFreak => 0.6,
        _ => 0.8,
    }
}

/// [`gml_speed_cap`] without the px/s conversion.
fn gml_speed_cap_frames(kind: EnemyKind) -> f32 {
    gml_speed_cap(kind) / crate::SIM_HZ as f32
}

fn wall_probe_axis(
    pos: glam::Vec2,
    radius: f32,
    value: f32,
    horizontal: bool,
    friction: f32,
    solids: &[(glam::Vec2, glam::Vec2)],
    mask: &FloorMask,
) -> f32 {
    if friction <= 0.0 || value.abs() <= 1e-6 {
        return value;
    }
    let sign = if value > 0.0 { 1.0 } else { -1.0 };
    let mut current = value;
    for _ in 0..4096 {
        let candidate = if horizontal {
            pos + glam::Vec2::new(current / 30.0, 0.0)
        } else {
            pos + glam::Vec2::new(0.0, current / 30.0)
        };
        if solid_contact(candidate, radius, solids, Some(mask)).is_none() {
            break;
        }
        current -= sign * friction * 30.0;
        if current.abs() <= 1e-6 {
            break;
        }
    }
    current
}

fn integrate_verbatim(
    brain: &mut EnemyBrain,
    vel: &mut Velocity,
    pos: &mut Pos,
    solids: &[(glam::Vec2, glam::Vec2)],
    mask: &FloorMask,
    positions: &[(glam::Vec2, i32)],
    kind: EnemyKind,
    epos: glam::Vec2,
    radius: f32,
    dt: f32,
    separate_enemies: bool,
    loops: u32,
    frame: u64,
    law: EnemyWallLaw,
) {
    // GML `friction` is only read by `enemy/Collision_Wall`'s slide loop;
    // `enemy` is a non-physics object so no built-in drag runs per step.
    let saved_direction = vel.0.normalize_or_zero();
    let saved_speed = vel.0.length();
    let parent = matches!(law, EnemyWallLaw::Parent) && loops <= 3;
    let contact = move_bounce_solid(
        &mut pos.0,
        &mut vel.0,
        radius,
        dt,
        solids,
        Some(mask),
        !matches!(law, EnemyWallLaw::Slide | EnemyWallLaw::SlidePush),
    );

    if parent && contact.is_some() {
        let post_speed = vel.0.length();
        if saved_direction.length_squared() > 1e-8 {
            vel.0 += saved_direction * post_speed;
        }
        if vel.0.length_squared() > 1e-8 {
            vel.0 = vel.0.normalize() * saved_speed;
        } else {
            vel.0 = glam::Vec2::ZERO;
        }
        if brain.friction > 0.0 {
            vel.0.x = wall_probe_axis(pos.0, radius, vel.0.x, true, brain.friction, solids, mask);
            vel.0.y = wall_probe_axis(pos.0, radius, vel.0.y, false, brain.friction, solids, mask);
        }
    } else if matches!(law, EnemyWallLaw::SlidePush)
        && let Some(contact) = contact
    {
        vel.0 += contact.normal * 30.0;
    }

    if separate_enemies {
        separate(positions, epos, &mut vel.0, kind, loops, frame);
    }
    sync_heading(brain, vel);
}

fn zero_damage_contact_push(
    brain: &mut EnemyBrain,
    vel: &mut Velocity,
    player_pos: Option<glam::Vec2>,
    pos: glam::Vec2,
    radius: f32,
    dt: f32,
) {
    if let Some(player_pos) = player_pos
        && pos.distance(player_pos) <= 8.0 + radius
    {
        add_gml_motion(
            brain,
            vel,
            (pos - player_pos).y.atan2((pos - player_pos).x),
            1.0,
            dt,
        );
    }
}

fn nearest_floor_point(mask: &FloorMask, point: glam::Vec2) -> glam::Vec2 {
    if mask.is_walkable(point) {
        return point;
    }
    let mut best = point;
    let mut best_d = f32::MAX;
    for cell in &mask.cells {
        let center = mask.cell_center(*cell);
        let d = center.distance_squared(point);
        if d < best_d {
            best_d = d;
            best = center;
        }
    }
    best
}

fn blocked_by_geometry(
    props: &Query<(Entity, &Prop, &Pos), With<Prop>>,
    mask: &FloorMask,
    point: glam::Vec2,
    radius: f32,
) -> bool {
    if !mask.cells.is_empty() && !mask.is_walkable(point) {
        return true;
    }
    let mut resolved = point;
    resolve_prop_collision(
        &mut resolved,
        radius,
        props.iter().map(|(_, p, pp)| (pp.0, p.size)),
    );
    resolved.distance_squared(point) > 0.0001
}

fn blocked_by_wall(mask: &FloorMask, point: glam::Vec2, radius: f32) -> bool {
    (!mask.cells.is_empty() && !mask.is_walkable(point))
        || point.x.abs() + radius > ARENA_W / 2.0
        || point.y.abs() + radius > ARENA_H / 2.0
}

pub fn spawn_toxic_gas(
    commands: &mut Commands,
    pos: glam::Vec2,
    velocity: glam::Vec2,
    state: ToxicGasState,
) -> Entity {
    let mut rng = rand::rng();
    let mut image = GmlImage::new("images/sprToxicGas.png", 5, 0.0);
    image.phase = rng.random_range(0.0..5.0);
    commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            Pos(pos),
            Velocity(velocity),
            state,
            image,
            NativeAngle(rng.random_range(0.0..std::f32::consts::TAU)),
            NativeDepth(-2.0),
        ))
        .id()
}

fn queue_motion_spawn(
    commands: &mut Commands,
    kind: EnemyKind,
    pos: glam::Vec2,
    velocity: glam::Vec2,
    difficulty: f32,
    loops: u32,
    give_kill: bool,
) -> Entity {
    queue_enemy_spawn_birth(
        commands,
        kind,
        pos,
        difficulty,
        loops,
        give_kill,
        Some(velocity),
        false,
    )
}

fn queue_fired_maggot(commands: &mut Commands, pos: glam::Vec2, angle: f32, loops: u32) -> Entity {
    queue_motion_spawn(
        commands,
        EnemyKind::FiredMaggot,
        pos,
        glam::Vec2::from_angle(angle) * 5.0 * 30.0,
        1.0,
        loops,
        true,
    )
}

fn queue_conversion_maggot(
    commands: &mut Commands,
    pos: glam::Vec2,
    angle: f32,
    loops: u32,
) -> Entity {
    queue_enemy_spawn_birth(
        commands,
        EnemyKind::Maggot,
        pos,
        1.0,
        loops,
        false,
        Some(glam::Vec2::from_angle(angle) * 4.0 * 30.0),
        false,
    )
}

/// table-driven fire, per bevy `enemy_ai` top to bottom (bosses `continue`
/// before their first timer tick — boss brains live elsewhere).
#[allow(clippy::too_many_arguments)]
pub fn enemy_ai(
    time: Res<SimTime>,
    mut commands: Commands,
    mut trauma: ResMut<Trauma>,
    euphoria: Res<Euphoria>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut ratking_cd: Local<HashMap<Entity, GTimer>>,
    mut wolf_roll: Local<HashMap<Entity, GTimer>>,
    mut charge_state: Local<HashMap<Entity, GTimer>>,
    player_q: Query<(&Pos, &Player), (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            Option<&BossBrain>,
            Option<&mut SpriteAnim>,
            Option<&HurtAnim>,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
    corpses: Query<(Entity, &Corpse, &Pos), (With<Corpse>, Without<Enemy>)>,
    catalog: Res<repame_anim::AnimCatalog>,
) {
    let Ok((player_pos, player)) = player_q.single() else {
        return;
    };
    let player_pos = player_pos.0;
    let dt = time.delta_secs;
    let mut rng = rand::rng();

    let euphoria = euphoria.0 || player.euphoria;

    // Pre-move snapshot for separation (bevy parity: pushes use the
    // snapshot, applied to the live position).
    let positions: Vec<(glam::Vec2, i32)> = enemies
        .iter()
        .map(|(_, enemy, _, _, pos, _, _, _)| (pos.0, crate::enemy_data::gml_size(enemy.kind)))
        .collect();
    let current_frame = (time.elapsed_secs / dt as f64) as u64;

    for (entity, enemy, mut brain, mut vel, mut pos, boss, mut anim, hurt) in &mut enemies {
        let epos = pos.0;
        let to_player = player_pos - epos;
        let dist = to_player.length();
        let dir = to_player.normalize_or_zero();

        let def = enemy_def(enemy.kind);

        if !has_verbatim_tick(enemy.kind) {
            brain.wkick = (brain.wkick - dt * 30.0).max(0.0);
        }

        if boss.is_some() {
            continue;
        }

        if enemy.kind == EnemyKind::IdpdVan {
            vel.0 = glam::Vec2::ZERO;
            continue;
        }

        // Kinds with dedicated verbatim ticks below own their motion and
        // fire law; the generic chase must not double-drive them.
        if matches!(
            enemy.kind,
            EnemyKind::Bandit
                | EnemyKind::SnowBandit
                | EnemyKind::Maggot
                | EnemyKind::MaggotSpawn
                | EnemyKind::BigMaggot
                | EnemyKind::JungleFly
                | EnemyKind::FiredMaggot
                | EnemyKind::Scorpion
                | EnemyKind::GoldScorpion
                | EnemyKind::Sniper
                | EnemyKind::JungleBandit
                | EnemyKind::MeleeBandit
                | EnemyKind::Assassin
                | EnemyKind::Ballguy
                | EnemyKind::EliteInspector
                | EnemyKind::EliteShielder
                | EnemyKind::ScrapBossMissile
                | EnemyKind::ProtoStatue
        ) {
            continue;
        }

        let emplacement = matches!(
            enemy.kind,
            EnemyKind::Turret
                | EnemyKind::Crystal
                | EnemyKind::LaserCrystal
                | EnemyKind::LightningCrystal
                | EnemyKind::InvLaserCrystal
        );

        brain.melee.tick(dt);
        // GML steps every alarm on every instance each step, so the decide
        // alarm and the fire alarm each tick exactly once per frame here and
        // every block below only reads `just_finished()`.
        brain.attack.tick(dt);
        brain.fire_alarm.tick(dt);

        if brain.walk > 0.0 {
            let (impulse_f, cap_f) = gml_walk_law(enemy.kind);
            let walk_dir = if vel.0.length_squared() > 1.0 {
                vel.0.normalize_or_zero()
            } else {
                dir
            };
            gml_motion_add_clamp(&mut vel.0, walk_dir, impulse_f, cap_f, dt);
            brain.walk -= dt * 30.0;
            if brain.walk < 0.0 {
                brain.walk = 0.0;
            }
        }

        // Turtle fire-strip swap: visual-only, omitted (renderer resolves
        // strips from state; no gameplay effect).

        let was_dashing = brain.dash > 0.0;

        if matches!(enemy.kind, EnemyKind::DogGuardian)
            && !was_dashing
            && dist < 160.0
            && dist > 40.0
            && brain.melee.is_finished()
            && rng.random::<f32>() < 0.67
        {
            brain.dash = 0.42;
            brain.melee = GTimer::from_seconds(10.0, TimerMode::Once);
            vel.0 = dir * 700.0;
        }

        if enemy.kind == EnemyKind::Raven && !was_dashing && brain.melee.is_finished() {
            brain.dash = 0.2;
            brain.melee = GTimer::from_seconds(rng.random_range(0.9..1.8), TimerMode::Once);
            let side = glam::Vec2::new(-dir.y, dir.x) * brain.strafe_dir;
            vel.0 = (dir * -0.35 + side).normalize() * 420.0;
        }

        if enemy.kind == EnemyKind::Guardian
            && brain.melee.is_finished()
            && dist < 480.0
            && rng.random_bool(0.012)
        {
            let want = def.preferred_range.max(140.0);
            let jump_dir = if dist > want { dir } else { -dir };
            let cand = epos + jump_dir * rng.random_range(70.0..130.0);
            if mask.is_walkable(cand) {
                pos.0 = cand;
                vel.0 = glam::Vec2::ZERO;
                spawn_burst(
                    &mut commands,
                    &mut rng,
                    epos,
                    8,
                    [0.3, 1.0, 0.45, 1.0],
                    (40.0, 120.0),
                );
            }
            brain.melee = GTimer::from_seconds(2.2, TimerMode::Once);
        }

        if enemy.kind == EnemyKind::PalaceGuardian
            && !was_dashing
            && dist < 80.0
            && dist > 24.0
            && brain.melee.is_finished()
        {
            brain.dash = 0.18;
            brain.melee = GTimer::from_seconds(0.9, TimerMode::Once);
            vel.0 = dir * 540.0;
        }
        if brain.dash > 0.0 {
            brain.dash = (brain.dash - dt).max(0.0);
        }

        let dashing = brain.dash > 0.0;

        if emplacement {
            vel.0 = glam::Vec2::ZERO;
        } else if dashing {
            pos.0 += vel.0 * dt;
        } else if brain.speed > 0.0 {
            // Kinds with their own `Alarm_1` further down must not also run
            // the generic decide, or its re-arm starves their own decide.
            let owns_decide = matches!(
                enemy.kind,
                EnemyKind::Gator
                    | EnemyKind::BuffGator
                    | EnemyKind::Jock
                    | EnemyKind::Molefish
                    | EnemyKind::Molesarge
                    | EnemyKind::FireBaller
                    | EnemyKind::SuperFireBaller
                    | EnemyKind::Necromancer
                    | EnemyKind::Mimic
                    | EnemyKind::SuperMimic
                    | EnemyKind::WepMimic
                    | EnemyKind::LaserCrystal
                    | EnemyKind::LightningCrystal
                    | EnemyKind::InvLaserCrystal
                    | EnemyKind::SnowTank
                    | EnemyKind::GoldSnowtank
                    | EnemyKind::Guardian
                    | EnemyKind::ExploGuardian
            );
            if !owns_decide && brain.attack.just_finished() {
                let los = has_line_of_sight(epos, player_pos, &mask);
                let base_ang = dir.y.atan2(dir.x);

                let (impulse, _cap, far_walk, close_walk, wander_walk) = match enemy.kind {
                    EnemyKind::Gator | EnemyKind::BuffGator => {
                        (0.8, 3.0, 10.0..14.0, 40.0..50.0, 20.0..30.0)
                    }
                    EnemyKind::Freak | EnemyKind::ExploFreak => {
                        (0.55, 4.0, 18.0..22.0, 12.0..18.0, 10.0..16.0)
                    }
                    EnemyKind::RhinoFreak => (0.8, 1.0, 18.0..22.0, 12.0..18.0, 10.0..16.0),
                    EnemyKind::Spider | EnemyKind::InvSpider => {
                        (2.0, 5.0, 15.0..20.0, 10.0..14.0, 10.0..20.0)
                    }
                    EnemyKind::Crab => (1.5, 4.5, 8.0..14.0, 50.0..60.0, 20.0..30.0),
                    EnemyKind::Turtle => (1.0, 5.0, 40.0..60.0, 40.0..60.0, 40.0..60.0),
                    EnemyKind::Salamander => (2.0, 2.5, 40.0..50.0, 20.0..30.0, 10.0..20.0),
                    EnemyKind::FireBaller | EnemyKind::SuperFireBaller => {
                        (0.6, 2.0, 8.0..12.0, 10.0..14.0, 10.0..16.0)
                    }
                    EnemyKind::Jock => (0.8, 3.0, 10.0..14.0, 40.0..50.0, 20.0..30.0),
                    EnemyKind::Molefish | EnemyKind::Molesarge => {
                        (0.8, 3.5, 10.0..14.0, 20.0..30.0, 20.0..30.0)
                    }
                    EnemyKind::Raven => (0.8, 3.5, 20.0..30.0, 40.0..50.0, 20.0..30.0),
                    EnemyKind::Rat
                    | EnemyKind::Ratking
                    | EnemyKind::FastRat
                    | EnemyKind::BigRat => (0.8, 4.0, 10.0..16.0, 40.0..50.0, 10.0..25.0),
                    EnemyKind::Wolf => (0.8, 4.0, 10.0..16.0, 20.0..30.0, 12.0..20.0),
                    EnemyKind::Assassin => (0.8, 4.0, 10.0..14.0, 20.0..28.0, 16.0..24.0),
                    EnemyKind::LightningCrystal => (0.5, 1.5, 10.0..14.0, 10.0..14.0, 10.0..20.0),
                    _ => (0.4, 4.0, 6.0..14.0, 18.0..28.0, 10.0..18.0),
                };

                if los {
                    if dist > 80.0 {
                        if rng.random::<f32>() < 0.35 {
                            brain.walk = 0.0;
                            vel.0 *= 0.5;
                        } else {
                            let ang = base_ang + rng.random_range(-45_f32..45.0).to_radians();
                            let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                            vel.0 = wdir * (impulse * 30.0);
                            brain.walk = rng.random_range(far_walk);
                            brain.gunangle = base_ang;
                        }
                    } else if dist < 44.0 {
                        let away = -dir;
                        let ang =
                            away.y.atan2(away.x) + rng.random_range(-15_f32..15.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (impulse * 30.0);
                        brain.walk = rng.random_range(close_walk);
                        brain.gunangle = base_ang;
                    } else {
                        let ang = base_ang + rng.random_range(-90_f32..90.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (impulse * 30.0);
                        brain.walk = rng.random_range(6.0..10.0);
                    }

                    let attack_secs = match enemy.kind {
                        EnemyKind::Rat
                        | EnemyKind::FastRat
                        | EnemyKind::BigRat
                        | EnemyKind::Ratking => rng.random_range(10.0..40.0) / 30.0,
                        EnemyKind::Freak | EnemyKind::ExploFreak => {
                            rng.random_range(6.0..11.0) / 30.0
                        }
                        EnemyKind::RhinoFreak | EnemyKind::DogGuardian | EnemyKind::Turtle => {
                            rng.random_range(6.0..11.0) / 30.0
                        }
                        EnemyKind::Spider | EnemyKind::InvSpider => {
                            rng.random_range(20.0..30.0) / 30.0
                        }
                        EnemyKind::Crab => rng.random_range(10.0..20.0) / 30.0,
                        EnemyKind::Salamander => rng.random_range(10.0..60.0) / 30.0,
                        EnemyKind::Assassin | EnemyKind::Wolf => rng.random_range(6.0..11.0) / 30.0,
                        _ => rng.random_range(0.35..0.75),
                    };
                    brain.attack = GTimer::from_seconds(attack_secs, TimerMode::Once);
                } else if rng.random::<f32>() < 0.4 {
                    let ang = rng.random_range(0.0..std::f32::consts::TAU);
                    let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                    vel.0 = wdir * (impulse * 30.0);
                    brain.walk = rng.random_range(wander_walk);
                    brain.attack = GTimer::from_seconds(
                        (brain.walk + 10.0 + rng.random_range(0.0..18.0)) / 30.0,
                        TimerMode::Once,
                    );
                } else {
                    brain.attack =
                        GTimer::from_seconds(rng.random_range(0.3..0.6), TimerMode::Once);
                }
            }

            let cap = gml_speed_cap(enemy.kind);
            if vel.0.length() > cap {
                vel.0 = vel.0.normalize() * cap;
            }
        } else {
            vel.0 = glam::Vec2::ZERO;
        }

        // GML `enemy/Collision_Wall`: `move_bounce_solid(true)`, plus the
        // `busycollisions` (loops <= 3) branch that restores the pre-bounce
        // heading/speed and then slides each blocked axis off by `friction`.
        let solids = prop_shapes(&props);
        let saved_direction = vel.0.normalize_or_zero();
        let saved_speed = vel.0.length();
        let contact = move_bounce_solid(
            &mut pos.0,
            &mut vel.0,
            def.radius,
            dt,
            &solids,
            Some(&mask),
            true,
        );
        if contact.is_some() && run.loop_count <= 3 {
            let post_speed = vel.0.length();
            if saved_direction.length_squared() > 1e-8 {
                vel.0 += saved_direction * post_speed;
            }
            if vel.0.length_squared() > 1e-8 {
                vel.0 = vel.0.normalize() * saved_speed;
            } else {
                vel.0 = glam::Vec2::ZERO;
            }
            if brain.friction > 0.0 {
                vel.0.x = wall_probe_axis(pos.0, def.radius, vel.0.x, true, brain.friction, &solids, &mask);
                vel.0.y = wall_probe_axis(pos.0, def.radius, vel.0.y, false, brain.friction, &solids, &mask);
            }
        }

        // InvSpider/InvLaserCrystal fade: visual-only, omitted.

        separate(&positions, epos, &mut vel.0, enemy.kind, run.loop_count, current_frame);

        if enemy.kind == EnemyKind::Necromancer {
            if brain.attack.just_finished() {
                brain.attack = GTimer::from_seconds(def.attack_cooldown, TimerMode::Once);
                let mut best: Option<(Entity, glam::Vec2)> = None;
                let mut best_d = 160.0;
                for (ce, _, cpos) in &corpses {
                    let d = cpos.0.distance(epos);
                    if d < best_d {
                        best_d = d;
                        best = Some((ce, cpos.0));
                    }
                }
                if let Some((ce, cpos)) = best {
                    commands.entity(ce).despawn();
                    let revived = if run.loop_count >= 1
                        && matches!(run.area, AreaId::Labs | AreaId::Palace)
                    {
                        EnemyKind::PopoFreak
                    } else {
                        EnemyKind::Freak
                    };
                    queue_enemy_spawn(&mut commands, revived, cpos, 1.0, run.loop_count);
                } else if (positions.len() as u32) < 40 {
                    let ang = rng.random_range(0.0..std::f32::consts::TAU);
                    let p = epos + glam::Vec2::new(ang.cos(), ang.sin()) * 40.0;
                    queue_enemy_spawn(&mut commands, EnemyKind::Freak, p, 1.0, run.loop_count);
                }
            }
        }

        if enemy.kind == EnemyKind::Ratking {
            let cd = ratking_cd
                .entry(entity)
                .or_insert_with(|| GTimer::from_seconds(1.0, TimerMode::Once));
            if brain.burst_left > 0 {
                brain.burst_timer.tick(dt);
                if brain.burst_timer.just_finished() {
                    let spread = rng.random_range(-20_f32..20.0).to_radians();
                    let base = dir.y.atan2(dir.x);
                    let ang = base + spread;
                    let off = glam::Vec2::new(ang.cos(), ang.sin()) * 12.0;
                    queue_enemy_spawn(
                        &mut commands,
                        EnemyKind::FastRat,
                        epos + off,
                        1.0,
                        run.loop_count,
                    );
                    brain.burst_left = brain.burst_left.saturating_sub(1);
                    if brain.burst_left > 0 {
                        brain.burst_timer = GTimer::from_seconds(6.0 / 30.0, TimerMode::Once);
                    }
                }
            } else {
                cd.tick(dt);
                if cd.just_finished() {
                    let los = has_line_of_sight(epos, player_pos, &mask);
                    if los && rng.random::<f32>() < 0.34 {
                        brain.burst_left = rng.random_range(3..=5);
                        brain.burst_timer = GTimer::from_seconds(6.0 / 30.0, TimerMode::Once);
                        *cd = GTimer::from_seconds(
                            (30.0 + rng.random_range(0.0..5.0)) / 30.0,
                            TimerMode::Once,
                        );
                    } else {
                        *cd = GTimer::from_seconds(
                            (30.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                }
            }
        }

        if matches!(
            enemy.kind,
            EnemyKind::LaserCrystal
                | EnemyKind::LightningCrystal
                | EnemyKind::InvLaserCrystal
                | EnemyKind::SnowTank
                | EnemyKind::GoldSnowtank
                | EnemyKind::Guardian
                | EnemyKind::ExploGuardian
        ) {
            let (charge_frames, min_range, max_range, aim_chance): (f32, f32, f32, f32) =
                match enemy.kind {
                    EnemyKind::LaserCrystal | EnemyKind::InvLaserCrystal => {
                        (30.0, 64.0, 160.0, 1.0)
                    }
                    EnemyKind::LightningCrystal => (20.0, 0.0, 96.0, 1.0),
                    EnemyKind::SnowTank => (40.0, 64.0, 240.0, 1.0 / 6.0),
                    EnemyKind::GoldSnowtank => (10.0, 64.0, 160.0, 0.5),
                    EnemyKind::Guardian => (12.0, 0.0, 999.0, 1.0),
                    EnemyKind::ExploGuardian => (60.0, 0.0, 90.0, 1.0),
                    _ => (30.0, 0.0, 999.0, 1.0),
                };
            if let Some(chg) = charge_state.get_mut(&entity) {
                chg.tick(dt);
                if chg.just_finished() {
                    charge_state.remove(&entity);
                    let gdir = glam::Vec2::new(brain.gunangle.cos(), brain.gunangle.sin());
                    if matches!(enemy.kind, EnemyKind::SnowTank | EnemyKind::GoldSnowtank) {
                        brain.burst_left = 16;
                        brain.burst_timer = GTimer::from_seconds(2.0 / 30.0, TimerMode::Once);
                        brain.strafe_dir = 0.0;
                    } else if matches!(
                        enemy.kind,
                        EnemyKind::Guardian | EnemyKind::ExploGuardian
                    ) {
                        // GML creates the whole volley in the alarm itself:
                        // Guardian 3 bullets (centre + 40 deg pair), ExploGuardian
                        // 14 bullets 24 deg apart.
                        fire_guardian_volley(
                            &mut commands,
                            &mut rng,
                            entity,
                            enemy.kind,
                            epos,
                            brain.gunangle,
                            euphoria,
                        );
                    } else {
                        brain.burst_left = def.bullets_per_shot;
                        brain.burst_timer =
                            GTimer::from_seconds(def.burst_interval, TimerMode::Once);
                        fire_enemy_bullet(
                            &mut commands,
                            &mut rng,
                            entity,
                            enemy,
                            def,
                            epos,
                            gdir,
                            euphoria,
                        );
                        brain.burst_left = brain.burst_left.saturating_sub(1);
                    }
                    show_enemy_fire(
                        &mut commands,
                        &catalog,
                        entity,
                        def.sprite,
                        anim.as_deref_mut(),
                        hurt.is_some(),
                    );
                    if enemy.kind == EnemyKind::GoldSnowtank {
                        let gdir = glam::Vec2::new(brain.gunangle.cos(), brain.gunangle.sin());
                        spawn_enemy_projectile(
                            &mut commands,
                            entity,
                            enemy.kind,
                            epos + gdir * 20.0,
                            gdir * 60.0,
                            5,
                            3.0,
                            5.0,
                            150.0,
                            true,
                        );
                    }
                }
            } else if brain.burst_left == 0 {
                if brain.attack.just_finished() {
                    let los = has_line_of_sight(epos, player_pos, &mask);
                    let in_range = dist >= min_range && dist <= max_range;
                    let guardian_ok = if enemy.kind == EnemyKind::Guardian {
                        los && ((dist > 96.0 && rng.random::<f32>() < 0.67)
                            || rng.random::<f32>() < 0.33)
                    } else {
                        los && in_range && rng.random::<f32>() < aim_chance
                    };
                    let explo_ok = if enemy.kind == EnemyKind::ExploGuardian {
                        los && dist <= 90.0
                    } else {
                        guardian_ok
                    };
                    if explo_ok {
                        brain.gunangle = dir.y.atan2(dir.x);
                        show_enemy_fire(
                            &mut commands,
                            &catalog,
                            entity,
                            def.sprite,
                            anim.as_deref_mut(),
                            hurt.is_some(),
                        );
                        match enemy.kind {
                            EnemyKind::LaserCrystal | EnemyKind::InvLaserCrystal => {
                                enemy_cue(&mut cues, "sndLaserCrystalCharge");
                            }
                            EnemyKind::LightningCrystal => {
                                enemy_cue(&mut cues, "sndLightningCrystalCharge");
                            }
                            EnemyKind::SnowTank => {
                                enemy_cue(&mut cues, "sndSnowTankAim");
                            }
                            EnemyKind::GoldSnowtank => {
                                enemy_cue(&mut cues, "sndGoldTankAim");
                            }
                            EnemyKind::ExploGuardian => {
                                enemy_cue(&mut cues, "sndExploGuardianCharge");
                            }
                            _ => {}
                        }
                        charge_state.insert(
                            entity,
                            GTimer::from_seconds(charge_frames / 30.0, TimerMode::Once),
                        );
                        brain.attack = GTimer::from_seconds(
                            def.attack_cooldown + charge_frames / 30.0,
                            TimerMode::Once,
                        );
                    } else {
                        let cd_secs = match enemy.kind {
                            EnemyKind::SnowTank => (40.0 + rng.random_range(0.0..30.0)) / 30.0,
                            EnemyKind::GoldSnowtank => (15.0 + rng.random_range(0.0..5.0)) / 30.0,
                            EnemyKind::Guardian => (10.0 + rng.random_range(0.0..40.0)) / 30.0,
                            EnemyKind::ExploGuardian => (6.0 + rng.random_range(0.0..5.0)) / 30.0,
                            EnemyKind::LaserCrystal | EnemyKind::InvLaserCrystal => {
                                (30.0 + rng.random_range(0.0..10.0)) / 30.0
                            }
                            _ => def.attack_cooldown,
                        };
                        brain.attack = GTimer::from_seconds(cd_secs, TimerMode::Once);
                        if matches!(enemy.kind, EnemyKind::SnowTank | EnemyKind::GoldSnowtank) {
                            let base =
                                dir.y.atan2(dir.x) + std::f32::consts::FRAC_PI_2 * brain.strafe_dir;
                            let wdir = glam::Vec2::new(base.cos(), base.sin());
                            vel.0 = wdir * (0.6 * 30.0);
                            brain.walk = 20.0 + rng.random_range(0.0..10.0);
                        }
                    }
                }
            } else if brain.burst_left > 0 {
                brain.burst_timer.tick(dt);
                if brain.burst_timer.just_finished() {
                    if matches!(enemy.kind, EnemyKind::SnowTank | EnemyKind::GoldSnowtank) {
                        let wave = brain.strafe_dir;
                        let amp = if enemy.kind == EnemyKind::SnowTank {
                            20.0_f32.to_radians()
                        } else {
                            15.0_f32.to_radians()
                        };
                        for i in [-1.0_f32, 1.0] {
                            let ang = brain.gunangle + (wave.sin() * amp) * i;
                            let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                            spawn_enemy_projectile(
                                &mut commands,
                                entity,
                                enemy.kind,
                                epos + sdir * 20.0,
                                sdir * def.projectile_speed,
                                def.projectile_damage,
                                def.projectile_lifetime,
                                def.projectile_radius,
                                150.0,
                                false,
                            );
                        }
                        brain.strafe_dir = wave + 0.1;
                        brain.burst_left = brain.burst_left.saturating_sub(1);
                        brain.burst_timer = GTimer::from_seconds(2.0 / 30.0, TimerMode::Once);
                        if brain.burst_left == 0 {
                            brain.attack = GTimer::from_seconds(
                                match enemy.kind {
                                    EnemyKind::SnowTank => {
                                        (40.0 + rng.random_range(0.0..30.0)) / 30.0
                                    }
                                    _ => (15.0 + rng.random_range(0.0..5.0)) / 30.0,
                                },
                                TimerMode::Once,
                            );
                        }
                    } else {
                        let gdir = glam::Vec2::new(brain.gunangle.cos(), brain.gunangle.sin());
                        fire_enemy_bullet(
                            &mut commands,
                            &mut rng,
                            entity,
                            enemy,
                            def,
                            epos,
                            gdir,
                            euphoria,
                        );
                        brain.burst_left = brain.burst_left.saturating_sub(1);
                        if brain.burst_left == 0 {
                            brain.attack =
                                GTimer::from_seconds(def.attack_cooldown, TimerMode::Once);
                        } else {
                            brain.burst_timer =
                                GTimer::from_seconds(def.burst_interval, TimerMode::Once);
                        }
                    }
                    show_enemy_fire(
                        &mut commands,
                        &catalog,
                        entity,
                        def.sprite,
                        anim.as_deref_mut(),
                        hurt.is_some(),
                    );
                }
            }
        }

        if enemy.kind == EnemyKind::Wolf {
            let cd = wolf_roll
                .entry(entity)
                .or_insert_with(|| GTimer::from_seconds(1.5, TimerMode::Once));
            cd.tick(dt);
            if cd.just_finished() {
                let los = has_line_of_sight(epos, player_pos, &mask);
                if los && rng.random::<f32>() < 0.5 {
                    let base = dir.y.atan2(dir.x);
                    for off in [0.0_f32, 20.0, -20.0] {
                        let ang = base + off.to_radians();
                        let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                        spawn_enemy_projectile(
                            &mut commands,
                            entity,
                            enemy.kind,
                            epos + sdir * 20.0,
                            sdir * 120.0,
                            2,
                            3.0,
                            4.0,
                            150.0,
                            false,
                        );
                    }
                    show_enemy_fire(
                        &mut commands,
                        &catalog,
                        entity,
                        def.sprite,
                        anim.as_deref_mut(),
                        hurt.is_some(),
                    );
                }
                *cd = GTimer::from_seconds(
                    (30.0 + rng.random_range(0.0..20.0)) / 30.0,
                    TimerMode::Once,
                );
            }
        }

        if matches!(enemy.kind, EnemyKind::Gator | EnemyKind::BuffGator) {
            if brain.attack.just_finished() {
                let los = has_line_of_sight(epos, player_pos, &mask);
                if los {
                    if dist > 48.0 && dist < 128.0 && rng.random::<f32>() < 0.34 {
                        brain.gunangle = dir.y.atan2(dir.x);
                        spawn_hit_warning(&mut commands, epos);
                        brain.walk = if enemy.kind == EnemyKind::BuffGator {
                            -15.0
                        } else {
                            -10.0
                        };
                        brain.attack = GTimer::from_seconds(
                            (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                    } else if dist <= 48.0 || dist >= 128.0 {
                        let ang =
                            (-dir).y.atan2((-dir).x) + rng.random_range(-10_f32..10.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (0.4 * 30.0);
                        brain.walk = 40.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = dir.y.atan2(dir.x);
                        brain.attack = GTimer::from_seconds(
                            (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                    } else {
                        let ang = dir.y.atan2(dir.x) + rng.random_range(-90_f32..90.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (0.4 * 30.0);
                        brain.walk = 10.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = dir.y.atan2(dir.x);
                        brain.attack = GTimer::from_seconds(
                            (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                } else {
                    brain.attack = GTimer::from_seconds(
                        (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                        TimerMode::Once,
                    );
                }
            }
            if brain.walk < 0.0 {
                brain.walk += dt * 30.0;
                if brain.walk >= 0.0 {
                    brain.walk = 0.0;
                    brain.gunangle = dir.y.atan2(dir.x);
                    if enemy.kind == EnemyKind::BuffGator {
                        enemy_cue(&mut cues, "sndFlakCannon");
                        let ang = brain.gunangle + rng.random_range(-25_f32..25.0).to_radians();
                        let spd = rng.random_range(240.0..300.0);
                        fire_enemy_flak(&mut commands, entity, enemy.kind, epos, ang, spd);
                    } else {
                        enemy_cue(&mut cues, "sndShotgun");
                        for _ in 0..6 {
                            let ang = brain.gunangle + rng.random_range(-25_f32..25.0).to_radians();
                            let spd = rng.random_range(300.0..420.0);
                            fire_enemy_shell(&mut commands, entity, enemy.kind, epos, ang, spd);
                        }
                    }
                    trauma.add(0.1);
                }
            }
        }

        if matches!(
            enemy.kind,
            EnemyKind::IdpdGrunt | EnemyKind::IdpdInspector | EnemyKind::IdpdElite
        ) && brain.ammo > 0
        {
            let los = has_line_of_sight(epos, player_pos, &mask);
            let should_throw = (!los && dist > 64.0) || (los && rng.random::<f32>() < 0.02);
            if should_throw {
                brain.ammo = brain.ammo.saturating_sub(1);
                let base = dir.y.atan2(dir.x);
                let ang = base + rng.random_range(-10_f32..10.0).to_radians();
                let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                commands
                    .spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Projectile {
                            damage: 5,
                            life: GTimer::from_seconds(2.0, TimerMode::Once),
                            radius: 6.0,
                            knockback: 150.0,
                            explosive: true,
                            source: Some(DamageSource::enemy(entity, enemy.kind)),
                        },
                        Velocity(sdir * 300.0),
                        Pos(epos + sdir * 16.0),
                        BouncesLeft(255),
                        ProjectileFriction(0.1),
                        GrenadeFuse {
                            smoke_armed: false,
                            friction_switched: false,
                            alarm1: GTimer::from_seconds(6.0 / 30.0, TimerMode::Once),
                        },
                    ))
                    .insert(ProjectileTyp(1));
            }
        }

        if matches!(
            enemy.kind,
            EnemyKind::Mimic | EnemyKind::SuperMimic | EnemyKind::WepMimic
        ) {
            if brain.attack.just_finished() {
                // Tell-strip swap: visual-only, omitted; the taunt cue stays.
                if enemy.kind == EnemyKind::SuperMimic || enemy.kind == EnemyKind::WepMimic {
                    enemy_cue(&mut cues, "sndHPMimicTaunt");
                } else {
                    enemy_cue(&mut cues, "sndMimicSlurp");
                }
                let cd_secs = if enemy.kind == EnemyKind::SuperMimic {
                    (150.0 + rng.random_range(0.0..180.0)) / 30.0
                } else {
                    (90.0 + rng.random_range(0.0..150.0)) / 30.0
                };
                brain.attack = GTimer::from_seconds(cd_secs, TimerMode::Once);
            }
            if dist < 200.0 {
                let chase = dir * 60.0;
                vel.0 = vel.0.lerp(chase, 0.1);
            }
        }

        if matches!(enemy.kind, EnemyKind::IdpdShield) {
            brain.melee.tick(dt);

            if brain.slash_delay > 0.0 {
                brain.slash_delay -= dt * 30.0;
                if brain.slash_delay <= 0.0 {
                    brain.slash_delay = 0.0;
                    // Melee slash visual: GML never damages with it, so the
                    // headless build keeps the timing state only.
                }
            }
            if brain.melee.just_finished() {
                // PopoShield follower: sim side keeps the owner-tracking
                // marker; the shield art resolves renderer-side.
                let base = brain.gunangle;
                let off = glam::Vec2::new(base.cos(), base.sin()) * 16.0;
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    ShieldFollower { owner: entity },
                    Pos(epos + off),
                ));
                brain.melee = GTimer::from_seconds(85.0 / 30.0, TimerMode::Once);
            }
        }

        if enemy.kind == EnemyKind::FrogQueen {
            if brain.burst_left > 0 {
                brain.burst_timer.tick(dt);
                if brain.burst_timer.just_finished() {
                    if (positions.len() as u32) < 30 {
                        queue_enemy_spawn(
                            &mut commands,
                            EnemyKind::FrogEgg,
                            epos,
                            1.0,
                            run.loop_count,
                        );
                    }
                    brain.burst_left = brain.burst_left.saturating_sub(1);
                    if brain.burst_left > 0 {
                        brain.burst_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
                    } else {
                        let base =
                            dir.y.atan2(dir.x) + rng.random_range(-15_f32..15.0).to_radians();
                        let sdir = glam::Vec2::new(base.cos(), base.sin());
                        spawn_enemy_projectile(
                            &mut commands,
                            entity,
                            enemy.kind,
                            epos + sdir * 20.0,
                            sdir * 120.0,
                            5,
                            4.0,
                            6.0,
                            150.0,
                            false,
                        );
                        show_enemy_fire(
                            &mut commands,
                            &catalog,
                            entity,
                            def.sprite,
                            anim.as_deref_mut(),
                            hurt.is_some(),
                        );
                        brain.attack = GTimer::from_seconds(
                            (30.0 + rng.random_range(0.0..20.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                }
            } else {
                if brain.attack.just_finished() {
                    let los = has_line_of_sight(epos, player_pos, &mask);
                    if los && rng.random::<f32>() < 0.25 {
                        brain.burst_left = (2 + run.loop_count).min(6) as usize;
                        brain.burst_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
                    } else {
                        brain.walk = 50.0;
                        brain.attack = GTimer::from_seconds(
                            (30.0 + rng.random_range(0.0..20.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                }
            }
        }

        if enemy.kind == EnemyKind::Jock {
            if brain.attack.just_finished() {
                let los = has_line_of_sight(epos, player_pos, &mask);
                if los {
                    if dist > 96.0 {
                        brain.gunangle = dir.y.atan2(dir.x);
                        let chance = 8 - brain.ammo as i32;
                        if brain.ammo > 0 && rng.random_range(0..chance.max(1)) == 0 {
                            brain.ammo = brain.ammo.saturating_sub(1);
                            let base = brain.gunangle;
                            let ang = base + rng.random_range(-10_f32..10.0).to_radians();
                            let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                            commands
                                .spawn((
                                    GameCleanup,
                                    LevelCleanup,
                                    Team::Enemy,
                                    Projectile {
                                        damage: def.projectile_damage,
                                        life: GTimer::from_seconds(
                                            def.projectile_lifetime,
                                            TimerMode::Once,
                                        ),
                                        radius: def.projectile_radius,
                                        knockback: 150.0,
                                        explosive: true,
                                        source: Some(DamageSource::enemy(entity, enemy.kind)),
                                    },
                                    Velocity(sdir * 60.0),
                                    Pos(epos + sdir * 20.0),
                                    Homing {
                                        turn_rate: 3.0,
                                        acquire_range: 600.0,
                                    },
                                ))
                                .insert(ProjectileTyp(2));
                            show_enemy_fire(
                                &mut commands,
                                &catalog,
                                entity,
                                def.sprite,
                                anim.as_deref_mut(),
                                hurt.is_some(),
                            );
                            brain.attack = GTimer::from_seconds(8.0 / 30.0, TimerMode::Once);
                        } else if rng.random::<f32>() < 0.67 {
                            let ang =
                                dir.y.atan2(dir.x) + rng.random_range(-40_f32..40.0).to_radians();
                            let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                            vel.0 = wdir * (0.4 * 30.0);
                            brain.walk = 10.0 + rng.random_range(0.0..10.0);
                            brain.gunangle = dir.y.atan2(dir.x);
                            brain.attack = GTimer::from_seconds(
                                (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                                TimerMode::Once,
                            );
                        } else {
                            brain.attack = GTimer::from_seconds(
                                (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                                TimerMode::Once,
                            );
                        }
                    } else {
                        let ang = dir.y.atan2(dir.x) + rng.random_range(-5_f32..5.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (0.4 * 30.0);
                        brain.walk = 40.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = dir.y.atan2(dir.x);
                        brain.attack = GTimer::from_seconds(
                            (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                } else if rng.random::<f32>() < 0.25 {
                    let ang = rng.random_range(0.0..std::f32::consts::TAU);
                    vel.0 = glam::Vec2::new(ang.cos(), ang.sin()) * (0.4 * 30.0);
                    brain.walk = 20.0 + rng.random_range(0.0..10.0);
                    brain.attack = GTimer::from_seconds(
                        (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                        TimerMode::Once,
                    );
                } else {
                    brain.attack = GTimer::from_seconds(
                        (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                        TimerMode::Once,
                    );
                }
            }
        }

        let uses_charge = matches!(
            enemy.kind,
            EnemyKind::LaserCrystal
                | EnemyKind::LightningCrystal
                | EnemyKind::InvLaserCrystal
                | EnemyKind::SnowTank
                | EnemyKind::GoldSnowtank
                | EnemyKind::Guardian
                | EnemyKind::ExploGuardian
                | EnemyKind::Jock
        );
        if def.bullets_per_shot > 0
            && dist < brain.shoot_range
            && !dashing
            && !uses_charge
            && !matches!(enemy.kind, EnemyKind::Gator | EnemyKind::BuffGator)
        {
            if def.burst {
                if brain.burst_left > 0 {
                    brain.burst_timer.tick(dt);
                    if brain.burst_timer.just_finished() {
                        fire_enemy_bullet(
                            &mut commands,
                            &mut rng,
                            entity,
                            enemy,
                            def,
                            epos,
                            dir,
                            euphoria,
                        );
                        show_enemy_fire(
                            &mut commands,
                            &catalog,
                            entity,
                            def.sprite,
                            anim.as_deref_mut(),
                            hurt.is_some(),
                        );
                        brain.burst_left -= 1;
                        if brain.burst_left == 0 {
                            brain.fire_alarm =
                                GTimer::from_seconds(def.attack_cooldown, TimerMode::Once);
                        }
                    }
                } else {
                    if brain.fire_alarm.just_finished() {
                        brain.burst_left = def.bullets_per_shot;
                        brain.burst_timer =
                            GTimer::from_seconds(def.burst_interval, TimerMode::Once);
                        fire_enemy_bullet(
                            &mut commands,
                            &mut rng,
                            entity,
                            enemy,
                            def,
                            epos,
                            dir,
                            euphoria,
                        );
                        show_enemy_fire(
                            &mut commands,
                            &catalog,
                            entity,
                            def.sprite,
                            anim.as_deref_mut(),
                            hurt.is_some(),
                        );
                        brain.burst_left -= 1;
                    }
                }
            } else {
                if brain.fire_alarm.just_finished() {
                    fire_enemy_shot(&mut commands, &mut rng, entity, enemy, def, epos, dir);
                    show_enemy_fire(
                        &mut commands,
                        &catalog,
                        entity,
                        def.sprite,
                        anim.as_deref_mut(),
                        hurt.is_some(),
                    );
                    brain.fire_alarm =
                        GTimer::from_seconds(def.attack_cooldown, TimerMode::Once);
                }
            }
        }
    }
}

pub fn tick_bandit(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    run: Res<Run>,
    mut cues: ResMut<Queue<AudioCue>>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (Entity, &Enemy, &mut EnemyBrain, &mut Velocity, &mut Pos),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
) {
    let dt = time.delta_secs;
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let positions: Vec<(glam::Vec2, i32)> = enemies.iter().map(|(_, enemy, _, _, p)| (p.0, crate::enemy_data::gml_size(enemy.kind))).collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos) in &mut enemies {
        if !matches!(enemy.kind, EnemyKind::Bandit | EnemyKind::SnowBandit) {
            continue;
        }
        let epos = pos.0;
        tick_verbatim_cooldown(&mut brain, dt);
        sync_heading(&mut brain, &vel);
        brain.attack.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds((20.0 + rng.random_range(0.0..10.0)) / 30.0, TimerMode::Once);
            if let Some(target) = player_pos {
                let delta = target - epos;
                let target_dir = delta.y.atan2(delta.x);
                let target_dist = delta.length();
                if has_line_of_sight(epos, target, &mask) {
                    if target_dist > 48.0 {
                        if rng.random::<f32>() < 0.25 {
                            brain.wkick = 4.0;
                            let angle =
                                brain.gunangle + rng.random_range(-10.0_f32..10.0).to_radians();
                            let d = glam::Vec2::from_angle(angle);
                            let bullet = spawn_enemy_projectile(
                                &mut commands,
                                entity,
                                enemy.kind,
                                epos,
                                d * 4.0 * 30.0,
                                3,
                                3.5,
                                4.0,
                                150.0,
                                false,
                            );
                            commands.entity(bullet).insert((
                                ProjectileTyp(1),
                                ProjectileFade("images/sprEnemyBulletHit.png"),
                            ));
                            enemy_cue(&mut cues, "sndEnemyFire");
                            brain.gunangle = target_dir;
                            brain.attack = GTimer::from_seconds(
                                (20.0 + rng.random_range(0.0..5.0)) / 30.0,
                                TimerMode::Once,
                            );
                        } else {
                            set_gml_direction(
                                &mut brain,
                                &mut vel,
                                target_dir + rng.random_range(-90.0_f32..90.0).to_radians(),
                            );
                            set_gml_speed(&mut brain, &mut vel, 0.4);
                            brain.walk = 10.0 + rng.random_range(0.0..10.0);
                            brain.gunangle = target_dir;
                        }
                    } else {
                        // GML `Bandit/Alarm_1`: inside 48 px it uses
                        // `point_direction(target.x, target.y, x, y)` (target ->
                        // self), so the retreat runs *away* from the player.
                        // `target_dir` is the opposite bearing.
                        set_gml_direction(
                            &mut brain,
                            &mut vel,
                            target_dir
                                + std::f32::consts::PI
                                + rng.random_range(-10.0_f32..10.0).to_radians(),
                        );
                        set_gml_speed(&mut brain, &mut vel, 0.4);
                        brain.walk = 40.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = target_dir;
                    }
                } else if rng.random::<f32>() < 0.25 {
                    let angle = rng.random_range(0.0..std::f32::consts::TAU);
                    add_gml_motion(&mut brain, &mut vel, angle, 0.4, dt);
                    brain.walk = 20.0 + rng.random_range(0.0..10.0);
                    brain.attack = GTimer::from_seconds(
                        (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                        TimerMode::Once,
                    );
                    brain.gunangle = brain.heading;
                }
            } else if rng.random::<f32>() < 0.1 {
                let angle = rng.random_range(0.0..std::f32::consts::TAU);
                add_gml_motion(&mut brain, &mut vel, angle, 0.4, dt);
                brain.walk = 20.0 + rng.random_range(0.0..10.0);
                brain.attack = GTimer::from_seconds(
                    (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                    TimerMode::Once,
                );
                brain.gunangle = brain.heading;
            }
        }
        walk_step(&mut brain, &mut vel, 0.8, 3.0, dt);
        integrate_verbatim(
            &mut brain,
            &mut vel,
            &mut pos,
            &solids,
            &mask,
            &positions,
            enemy.kind,
            epos,
            enemy_def(enemy.kind).radius,
            dt,
            true,
            run.loop_count,
            current_frame,
            wall_law(enemy.kind),
        );
        zero_damage_contact_push(
            &mut brain,
            &mut vel,
            player_pos,
            pos.0,
            enemy_def(enemy.kind).radius,
            dt,
        );
    }
}

pub fn tick_maggot(
    time: Res<SimTime>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            Option<&HurtAnim>,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
) {
    let dt = time.delta_secs;
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let positions: Vec<(glam::Vec2, i32)> = enemies.iter().map(|(_, enemy, _, _, p, _)| (p.0, crate::enemy_data::gml_size(enemy.kind))).collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (_entity, enemy, mut brain, mut vel, mut pos, hurt) in &mut enemies {
        if enemy.kind != EnemyKind::Maggot {
            continue;
        }
        let epos = pos.0;
        tick_verbatim_cooldown(&mut brain, dt);
        sync_heading(&mut brain, &vel);
        brain.attack.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds((30.0 + rng.random_range(0.0..20.0)) / 30.0, TimerMode::Once);
            if let Some(target) = player_pos
                && has_line_of_sight(epos, target, &mask)
            {
                set_gml_direction(
                    &mut brain,
                    &mut vel,
                    (target - epos).y.atan2((target - epos).x)
                        + rng.random_range(-10.0_f32..10.0).to_radians(),
                );
            } else {
                add_gml_motion(
                    &mut brain,
                    &mut vel,
                    rng.random_range(0.0..std::f32::consts::TAU),
                    0.5,
                    dt,
                );
            }
            if blocked_by_wall(&mask, pos.0, enemy_def(enemy.kind).radius) {
                pos.0 = nearest_floor_point(&mask, pos.0);
            }
        }
        if !hurt.is_some_and(|h| !h.timer.is_finished()) {
            let heading = brain.heading;
            add_gml_motion(&mut brain, &mut vel, heading, 0.6, dt);
        }
        cap_gml_speed(&mut brain, &mut vel, 2.0);
        integrate_verbatim(
            &mut brain,
            &mut vel,
            &mut pos,
            &solids,
            &mask,
            &positions,
            enemy.kind,
            epos,
            enemy_def(enemy.kind).radius,
            dt,
            true,
            run.loop_count,
            current_frame,
            wall_law(enemy.kind),
        );
    }
}

pub fn tick_maggot_spawn(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            &mut Health,
            Option<&HurtAnim>,
            Option<&mut MaggotSpawnCharge>,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
) {
    let dt = time.delta_secs;
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let positions: Vec<(glam::Vec2, i32)> = enemies.iter().map(|(_, enemy, _, _, p, _, _, _) | (p.0, crate::enemy_data::gml_size(enemy.kind))).collect();
    let solids = prop_shapes(&props);
    let spawns: Vec<(Entity, glam::Vec2)> = enemies
        .iter()
        .filter_map(|(entity, enemy, _, _, pos, _, _, _)| {
            (enemy.kind == EnemyKind::MaggotSpawn).then_some((entity, pos.0))
        })
        .collect();
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos, mut health, hurt, mut charge) in &mut enemies {
        if enemy.kind != EnemyKind::MaggotSpawn {
            continue;
        }
        let epos = pos.0;
        tick_verbatim_cooldown(&mut brain, dt);
        vel.0 = glam::Vec2::ZERO;
        let mut started_charge = false;
        if let Some(target) = player_pos {
            if epos.distance(target) < 64.0
                && !brain.maggot_spawn_charging
                && charge.is_none()
                && !hurt.is_some_and(|h| !h.timer.is_finished())
            {
                health.hp -= 1;
                brain.maggot_spawn_charging = true;
                brain.maggot_spawn_charge_ticks = 19.0;
                brain.maggot_spawn_facing = if target.x >= epos.x { 1.0 } else { -1.0 };
                let mut charge_state = MaggotSpawnCharge::new();
                charge_state.facing = brain.maggot_spawn_facing;
                commands.entity(entity).insert(charge_state);
                commands.entity(entity).insert(MaggotSpawnInternalDrain);
                started_charge = true;
            }
        }
        if brain.maggot_spawn_charging && !started_charge {
            brain.maggot_spawn_charge_ticks =
                (brain.maggot_spawn_charge_ticks - dt * crate::SIM_HZ as f32).max(0.0);
            if let Some(charge) = charge.as_deref_mut() {
                charge.image.advance(dt * crate::SIM_HZ as f32);
                charge.ticks_left = brain.maggot_spawn_charge_ticks;
            }
            if brain.maggot_spawn_charge_ticks == 0.0 {
                brain.maggot_spawn_charging = false;
                commands.entity(entity).remove::<MaggotSpawnCharge>();
            }
        }
        if charge.is_some() && !brain.maggot_spawn_charging {
            commands.entity(entity).remove::<MaggotSpawnCharge>();
        }
        integrate_verbatim(
            &mut brain,
            &mut vel,
            &mut pos,
            &solids,
            &mask,
            &positions,
            enemy.kind,
            epos,
            enemy_def(enemy.kind).radius,
            dt,
            false,
            run.loop_count,
            current_frame,
            wall_law(enemy.kind),
        );
        let radius = enemy_def(enemy.kind).radius;
        for (other_entity, other_pos) in &spawns {
            if *other_entity == entity || entity.index() >= other_entity.index() {
                continue;
            }
            if pos.0.distance(*other_pos) > radius * 2.0 {
                continue;
            }
            let jitter = glam::Vec2::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
            let to_other = *other_pos + jitter - pos.0;
            let angle = to_other.y.atan2(to_other.x);
            let mx = angle.cos() * 8.0;
            let my = angle.sin() * 8.0;
            let x = pos.0 + glam::Vec2::new(mx, 0.0);
            if !blocked_by_geometry(&props, &mask, x, 0.0) && !blocked_by_wall(&mask, x, 0.0) {
                pos.0.x += mx;
            }
            let y = pos.0 + glam::Vec2::new(0.0, my);
            if !blocked_by_geometry(&props, &mask, y, 0.0) && !blocked_by_wall(&mask, y, 0.0) {
                pos.0.y += my;
            }
        }
        zero_damage_contact_push(
            &mut brain,
            &mut vel,
            player_pos,
            pos.0,
            enemy_def(enemy.kind).radius,
            dt,
        );
    }
}

pub fn tick_bigmaggot(
    time: Res<SimTime>,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            &Health,
            Option<&HurtAnim>,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
) {
    let dt = time.delta_secs;
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let positions: Vec<(glam::Vec2, i32)> = enemies.iter().map(|(_, enemy, _, _, p, _, _) | (p.0, crate::enemy_data::gml_size(enemy.kind))).collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (_entity, enemy, mut brain, mut vel, mut pos, health, hurt) in &mut enemies {
        if enemy.kind != EnemyKind::BigMaggot || brain.burrow_state != 0 {
            continue;
        }
        let epos = pos.0;
        tick_verbatim_cooldown(&mut brain, dt);
        sync_heading(&mut brain, &vel);
        brain.attack.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds((20.0 + rng.random_range(0.0..20.0)) / 30.0, TimerMode::Once);
            let mut burrow = false;
            if let Some(target) = player_pos {
                let target_dir = (target - epos).y.atan2((target - epos).x);
                if has_line_of_sight(epos, target, &mask) {
                    if brain.rage == 0.0 {
                        brain.rage = 1.0;
                        set_gml_direction(
                            &mut brain,
                            &mut vel,
                            target_dir + rng.random_range(-30.0_f32..30.0).to_radians(),
                        );
                    }
                } else if health.hp < health.max && rng.random::<f32>() < 0.5 {
                    burrow = true;
                } else {
                    brain.rage = 0.0;
                    add_gml_motion(
                        &mut brain,
                        &mut vel,
                        rng.random_range(0.0..std::f32::consts::TAU),
                        1.0,
                        dt,
                    );
                }
            } else {
                brain.rage = 0.0;
                add_gml_motion(
                    &mut brain,
                    &mut vel,
                    rng.random_range(0.0..std::f32::consts::TAU),
                    1.0,
                    dt,
                );
            }
            if burrow {
                enemy_cue(&mut cues, "sndBigMaggotBurrow");
                brain.rage = 0.0;
                brain.burrow_state = 1;
                brain.burrow_alarm0 = 30.0;
                brain.burrow_alarm1 = 0.0;
                brain.burrow_angle = rng.random_range(0.0..std::f32::consts::TAU);
                vel.0 = glam::Vec2::ZERO;
                sync_heading(&mut brain, &vel);
                continue;
            }
        }
        if !hurt.is_some_and(|h| !h.timer.is_finished()) {
            let heading = brain.heading;
            add_gml_motion(&mut brain, &mut vel, heading, 0.5, dt);
        }
        if vel.0.length() < 0.5 * 30.0 {
            set_gml_speed(&mut brain, &mut vel, 0.5);
        }
        let cap = 1.0 + brain.rage * 2.0;
        cap_gml_speed(&mut brain, &mut vel, cap);
        integrate_verbatim(
            &mut brain,
            &mut vel,
            &mut pos,
            &solids,
            &mask,
            &positions,
            enemy.kind,
            epos,
            enemy_def(enemy.kind).radius,
            dt,
            true,
            run.loop_count,
            current_frame,
            wall_law(enemy.kind),
        );
    }
}

pub fn tick_bigmaggot_burrow(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Res<FloorMask>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (Entity, &Enemy, &mut EnemyBrain, &mut Velocity, &mut Pos),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
) {
    let dt = time.delta_secs;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let mut rng = rand::rng();
    for (_entity, enemy, mut brain, mut vel, mut pos) in &mut enemies {
        if enemy.kind != EnemyKind::BigMaggot || brain.burrow_state == 0 {
            continue;
        }
        tick_verbatim_cooldown(&mut brain, dt);
        if brain.burrow_state == 2 {
            brain.attack.tick(dt);
            if brain.attack.just_finished() {
                enemy_cue(&mut cues, "sndBigMaggotUnburrow");
            }
        }
        if brain.burrow_state == 1 && brain.burrow_alarm0 > 0.0 {
            brain.burrow_alarm0 -= dt * 30.0;
            if brain.burrow_alarm0 <= 0.0 {
                brain.burrow_angle = rng.random_range(0.0..std::f32::consts::TAU);
                if let Some(target) = player_pos {
                    let dest = target + glam::Vec2::from_angle(brain.burrow_angle) * 64.0;
                    if !blocked_by_geometry(&props, &mask, dest, 0.0)
                        && !blocked_by_wall(&mask, dest, 0.0)
                    {
                        pos.0 = dest;
                        vel.0 = glam::Vec2::ZERO;
                        sync_heading(&mut brain, &vel);
                        enemy_cue(&mut cues, "sndBigMaggotUnburrowSand");
                        brain.burrow_state = 2;
                        brain.burrow_alarm1 = 5.0 / 0.4;
                        brain.attack = GTimer::from_seconds((5.0 / 0.4) / 30.0, TimerMode::Once);
                        spawn_burst(
                            &mut commands,
                            &mut rng,
                            dest,
                            10,
                            [0.6, 0.45, 0.3, 1.0],
                            (40.0, 140.0),
                        );
                    } else {
                        brain.burrow_alarm0 = 1.0;
                    }
                }
            }
        } else if brain.burrow_state == 2 {
            brain.burrow_alarm1 -= dt * 30.0;
            if brain.burrow_alarm1 <= 0.0 {
                brain.burrow_state = 0;
                brain.attack = GTimer::from_seconds(
                    (10.0 + rng.random_range(0.0..10.0)) / 30.0,
                    TimerMode::Once,
                );
                brain.walk = 0.0;
                brain.burrow_alarm0 = 0.0;
                brain.burrow_alarm1 = 0.0;
            }
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub fn tick_jungle_fly(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            Option<&HurtAnim>,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
) {
    let dt = time.delta_secs;
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let positions: Vec<(glam::Vec2, i32)> = enemies.iter().map(|(_, enemy, _, _, p, _) | (p.0, crate::enemy_data::gml_size(enemy.kind))).collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (_entity, enemy, mut brain, mut vel, mut pos, hurt) in &mut enemies {
        if enemy.kind != EnemyKind::JungleFly {
            continue;
        }
        let epos = pos.0;
        tick_verbatim_cooldown(&mut brain, dt);
        sync_heading(&mut brain, &vel);
        brain.attack.tick(dt);
        brain.burst_timer.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds((15.0 + rng.random_range(0.0..5.0)) / 30.0, TimerMode::Once);
            if let Some(target) = player_pos {
                let target_dir = (target - epos).y.atan2((target - epos).x);
                if has_line_of_sight(epos, target, &mask) && rng.random::<f32>() > 0.2 {
                    if rng.random::<f32>() < 1.0 / 6.0
                        && brain.ammo > 0
                        && epos.distance(target) > 96.0
                    {
                        brain.ammo -= 1;
                        brain.fire = 6;
                        brain.burst_timer = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                        brain.attack = GTimer::from_seconds(1.0, TimerMode::Once);
                        brain.rage = 0.0;
                        brain.gunangle = target_dir + rng.random_range(-3.0_f32..3.0).to_radians();
                    } else {
                        if brain.rage == 0.0 {
                            brain.rage = 1.0;
                        }
                        set_gml_direction(
                            &mut brain,
                            &mut vel,
                            target_dir + rng.random_range(-30.0_f32..30.0).to_radians(),
                        );
                    }
                } else {
                    brain.rage = 0.0;
                    add_gml_motion(
                        &mut brain,
                        &mut vel,
                        rng.random_range(0.0..std::f32::consts::TAU),
                        1.0,
                        dt,
                    );
                }
            } else {
                brain.rage = 0.0;
                add_gml_motion(
                    &mut brain,
                    &mut vel,
                    rng.random_range(0.0..std::f32::consts::TAU),
                    1.0,
                    dt,
                );
            }
        }
        if brain.burst_timer.just_finished() && brain.fire > 0 {
            let angle = brain.gunangle + rng.random_range(-1.0_f32..1.0).to_radians();
            queue_fired_maggot(&mut commands, epos, angle, run.loop_count);
            cues.push(AudioCue {
                name: "sndFlyFire",
                volume: 0.3,
                variance: 0.05,
            });
            brain.burst_timer = GTimer::from_seconds(2.0 / 30.0, TimerMode::Once);
            brain.fire -= 1;
        }
        if !hurt.is_some_and(|h| !h.timer.is_finished()) {
            let heading = brain.heading;
            add_gml_motion(&mut brain, &mut vel, heading, 1.0, dt);
        }
        let cap = 2.0 + brain.rage * 2.0;
        if vel.0.length() > cap * 30.0 {
            cap_gml_speed(&mut brain, &mut vel, cap);
        }
        if vel.0.length() < 30.0 {
            set_gml_speed(&mut brain, &mut vel, 1.0);
        }
        integrate_verbatim(
            &mut brain,
            &mut vel,
            &mut pos,
            &solids,
            &mask,
            &positions,
            enemy.kind,
            epos,
            enemy_def(enemy.kind).radius,
            dt,
            true,
            run.loop_count,
            current_frame,
            wall_law(enemy.kind),
        );
    }
}

pub fn tick_fired_maggot(
    time: Res<SimTime>,
    mut commands: Commands,
    run: Res<Run>,
    mask: Res<FloorMask>,
    player_q: Query<(&Pos, &Health), (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (Entity, &Enemy, &mut EnemyBrain, &mut Velocity, &mut Pos),
        (With<Enemy>, Without<Prop>),
    >,
) {
    let dt = time.delta_secs;
    let player = player_q.single().ok().map(|(p, h)| (p.0, h.hp > 0));
    for (entity, enemy, mut brain, mut vel, mut pos) in &mut enemies {
        if enemy.kind != EnemyKind::FiredMaggot {
            continue;
        }
        tick_verbatim_cooldown(&mut brain, dt);
        sync_heading(&mut brain, &vel);
        set_gml_speed(&mut brain, &mut vel, 7.0);
        let epos = pos.0;
        let next = epos + vel.0 * dt;
        if let Some((player_pos, true)) = player
            && next.distance(player_pos) <= 8.0 + enemy_def(enemy.kind).radius
        {
            let angle = (player_pos - epos).y.atan2((player_pos - epos).x);
            queue_conversion_maggot(&mut commands, next, angle, run.loop_count);
            commands.entity(entity).despawn();
            continue;
        }
        if blocked_by_wall(&mask, next, enemy_def(enemy.kind).radius) {
            let angle = brain.heading + std::f32::consts::PI;
            queue_conversion_maggot(&mut commands, next, angle, run.loop_count);
            commands.entity(entity).despawn();
            continue;
        }
        pos.0 += vel.0 * dt;
        sync_heading(&mut brain, &vel);
    }
}
#[allow(clippy::too_many_arguments)]
pub fn tick_scorpion(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            Option<&mut SpriteAnim>,
            Option<&HurtAnim>,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
    catalog: Res<repame_anim::AnimCatalog>,
) {
    let dt = time.delta_secs;
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let positions: Vec<(glam::Vec2, i32)> = enemies.iter().map(|(_, enemy, _, _, p, _, _)| (p.0, crate::enemy_data::gml_size(enemy.kind))).collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos, mut anim, hurt) in &mut enemies {
        if enemy.kind != EnemyKind::Scorpion {
            continue;
        }
        let epos = pos.0;
        tick_verbatim_cooldown(&mut brain, dt);
        sync_heading(&mut brain, &vel);
        brain.attack.tick(dt);
        brain.burst_timer.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds((30.0 + rng.random_range(0.0..10.0)) / 30.0, TimerMode::Once);
            if let Some(target) = player_pos {
                let target_dir = (target - epos).y.atan2((target - epos).x);
                set_gml_direction(
                    &mut brain,
                    &mut vel,
                    target_dir
                        + rng.random_range(-60.0_f32..60.0).to_radians()
                        + std::f32::consts::PI,
                );
                brain.walk = rng.random_range(10.0..20.0);
                set_gml_speed(&mut brain, &mut vel, 0.4);
                if has_line_of_sight(epos, target, &mask)
                    && epos.distance(target) <= 210.0
                    && rng.random::<f32>() < 0.5
                {
                    brain.attack = GTimer::from_seconds(
                        (30.0 + rng.random_range(0.0..5.0)) / 30.0,
                        TimerMode::Once,
                    );
                    brain.burst_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                    brain.gunangle = target_dir;
                    brain.ammo = 10;
                    enemy_cue(&mut cues, "sndScorpionFireStart");
                }
                if epos.distance(target) < 64.0 {
                    let mut angle = target_dir + rng.random_range(-10.0_f32..10.0).to_radians();
                    if epos.distance(target) > 32.0 {
                        angle += std::f32::consts::PI;
                    }
                    set_gml_direction(&mut brain, &mut vel, angle);
                    brain.walk = 40.0;
                }
            } else {
                add_gml_motion(
                    &mut brain,
                    &mut vel,
                    rng.random_range(0.0..std::f32::consts::TAU),
                    0.4,
                    dt,
                );
                brain.walk = rng.random_range(10.0..20.0);
                brain.attack = GTimer::from_seconds(
                    (brain.walk + rng.random_range(10..=30) as f32) / 30.0,
                    TimerMode::Once,
                );
            }
        }
        if brain.burst_timer.just_finished() {
            if brain.ammo > 0 {
                brain.ammo -= 1;
                brain.burst_timer = GTimer::from_seconds(2.0 / 30.0, TimerMode::Once);
                let angle = brain.gunangle + rng.random_range(-20.0_f32..20.0).to_radians();
                let bullet = spawn_enemy_projectile(
                    &mut commands,
                    entity,
                    enemy.kind,
                    epos,
                    glam::Vec2::from_angle(angle) * rng.random_range(3.0_f32..4.0) * 30.0,
                    2,
                    3.0,
                    4.0,
                    150.0,
                    false,
                );
                commands.entity(bullet).insert((
                    ProjectileTyp(2),
                    ProjectileFade("images/sprScorpionBulletHit.png"),
                ));
                enemy_cue(&mut cues, "sndScorpionFire");
                show_enemy_fire(
                    &mut commands,
                    &catalog,
                    entity,
                    enemy_def(enemy.kind).sprite,
                    anim.as_deref_mut(),
                    hurt.is_some(),
                );
            } else {
                brain.attack = GTimer::from_seconds(
                    (40.0 + rng.random_range(0.0..10.0)) / 30.0,
                    TimerMode::Once,
                );
            }
        }
        walk_step(&mut brain, &mut vel, 2.0, 4.0, dt);
        integrate_verbatim(
            &mut brain,
            &mut vel,
            &mut pos,
            &solids,
            &mask,
            &positions,
            enemy.kind,
            epos,
            enemy_def(enemy.kind).radius,
            dt,
            true,
            run.loop_count,
            current_frame,
            wall_law(enemy.kind),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn tick_gold_scorpion(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            Option<&mut SpriteAnim>,
            Option<&HurtAnim>,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
    catalog: Res<repame_anim::AnimCatalog>,
) {
    let dt = time.delta_secs;
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let positions: Vec<(glam::Vec2, i32)> = enemies.iter().map(|(_, enemy, _, _, p, _, _)| (p.0, crate::enemy_data::gml_size(enemy.kind))).collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos, mut anim, hurt) in &mut enemies {
        if enemy.kind != EnemyKind::GoldScorpion {
            continue;
        }
        let epos = pos.0;
        tick_verbatim_cooldown(&mut brain, dt);
        sync_heading(&mut brain, &vel);
        brain.attack.tick(dt);
        brain.burst_timer.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds((30.0 + rng.random_range(0.0..10.0)) / 30.0, TimerMode::Once);
            if let Some(target) = player_pos {
                let target_dir = (target - epos).y.atan2((target - epos).x);
                if has_line_of_sight(epos, target, &mask) && rng.random::<f32>() < 0.5 {
                    brain.ammo = 20;
                    brain.walk = 0.0;
                    brain.burst_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                    enemy_cue(&mut cues, "sndGoldScorpionFire");
                    brain.gunangle = target_dir;
                    brain.attack = GTimer::from_seconds(
                        (20.0 + rng.random_range(0.0..5.0)) / 30.0,
                        TimerMode::Once,
                    );
                }
                set_gml_direction(
                    &mut brain,
                    &mut vel,
                    target_dir + rng.random_range(-60.0_f32..60.0).to_radians(),
                );
                set_gml_speed(&mut brain, &mut vel, 0.4);
                if brain.ammo == 0 {
                    brain.walk = 10.0 + rng.random_range(0.0..10.0);
                }
                if epos.distance(target) < 64.0 {
                    brain.walk = 40.0;
                    set_gml_direction(
                        &mut brain,
                        &mut vel,
                        target_dir + rng.random_range(-10.0_f32..10.0).to_radians(),
                    );
                }
                if brain.ammo == 0 {
                    add_gml_motion(&mut brain, &mut vel, target_dir, 0.3, dt);
                }
            } else if rng.random::<f32>() < 0.1 {
                add_gml_motion(
                    &mut brain,
                    &mut vel,
                    rng.random_range(0.0..std::f32::consts::TAU),
                    0.4,
                    dt,
                );
                brain.walk = 10.0 + rng.random_range(0.0..10.0);
                brain.attack = GTimer::from_seconds(
                    (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                    TimerMode::Once,
                );
                brain.gunangle = brain.heading;
            }
        }
        if brain.burst_timer.just_finished() {
            if brain.ammo > 0 {
                brain.ammo -= 1;
                brain.burst_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                let angle1 = brain.gunangle + rng.random_range(-5.0_f32..5.0).to_radians();
                let angle2 = brain.gunangle + rng.random_range(-40.0_f32..40.0).to_radians();
                let b1 = spawn_enemy_projectile(
                    &mut commands,
                    entity,
                    enemy.kind,
                    epos,
                    glam::Vec2::from_angle(angle1) * (5.0 + rng.random_range(0.0..1.0)) * 30.0,
                    2,
                    3.0,
                    4.0,
                    150.0,
                    false,
                );
                let b2 = spawn_enemy_projectile(
                    &mut commands,
                    entity,
                    enemy.kind,
                    epos,
                    glam::Vec2::from_angle(angle2) * (1.5 + rng.random_range(0.0..0.5)) * 30.0,
                    2,
                    3.0,
                    4.0,
                    150.0,
                    false,
                );
                commands.entity(b1).insert(ProjectileTyp(2));
                commands.entity(b2).insert(ProjectileTyp(2));
                enemy_cue(&mut cues, "sndScorpionFire");
                show_enemy_fire(
                    &mut commands,
                    &catalog,
                    entity,
                    enemy_def(enemy.kind).sprite,
                    anim.as_deref_mut(),
                    hurt.is_some(),
                );
            } else {
                brain.attack = GTimer::from_seconds(
                    (40.0 + rng.random_range(0.0..10.0)) / 30.0,
                    TimerMode::Once,
                );
            }
        }
        walk_step(&mut brain, &mut vel, 2.0, 3.0, dt);
        if vel.0.length() < 30.0 && brain.ammo < 1 {
            set_gml_speed(&mut brain, &mut vel, 1.0);
        }
        integrate_verbatim(
            &mut brain,
            &mut vel,
            &mut pos,
            &solids,
            &mask,
            &positions,
            enemy.kind,
            epos,
            enemy_def(enemy.kind).radius,
            dt,
            true,
            run.loop_count,
            current_frame,
            wall_law(enemy.kind),
        );
    }
}

pub fn tick_sniper(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (Entity, &Enemy, &mut EnemyBrain, &mut Velocity, &mut Pos),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
) {
    let dt = time.delta_secs;
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let positions: Vec<(glam::Vec2, i32)> = enemies.iter().map(|(_, enemy, _, _, p) | (p.0, crate::enemy_data::gml_size(enemy.kind))).collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos) in &mut enemies {
        if enemy.kind != EnemyKind::Sniper {
            continue;
        }
        let epos = pos.0;
        tick_verbatim_cooldown(&mut brain, dt);
        sync_heading(&mut brain, &vel);
        brain.attack.tick(dt);
        brain.burst_timer.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds((20.0 + rng.random_range(0.0..10.0)) / 30.0, TimerMode::Once);
            if !brain.sniper_aiming {
                if let Some(target) = player_pos {
                    let target_dir = (target - epos).y.atan2((target - epos).x);
                    if has_line_of_sight(epos, target, &mask) {
                        if epos.distance(target) > 96.0 {
                            if rng.random::<f32>() < 2.0 / 3.0 {
                                enemy_cue(&mut cues, "sndSniperTarget");
                                brain.walk = 0.0;
                                brain.attack = GTimer::from_seconds(40.0 / 30.0, TimerMode::Once);
                                brain.burst_timer = GTimer::from_seconds(1.0, TimerMode::Once);
                                brain.sniper_aiming = true;
                            } else {
                                set_gml_direction(
                                    &mut brain,
                                    &mut vel,
                                    target_dir + rng.random_range(-80.0_f32..80.0).to_radians(),
                                );
                                set_gml_speed(&mut brain, &mut vel, 0.4);
                                brain.walk = 10.0 + rng.random_range(0.0..10.0);
                                brain.gunangle = target_dir;
                            }
                        } else {
                            set_gml_direction(
                                &mut brain,
                                &mut vel,
                                target_dir + rng.random_range(-10.0_f32..10.0).to_radians(),
                            );
                            set_gml_speed(&mut brain, &mut vel, 0.4);
                            brain.walk = 40.0 + rng.random_range(0.0..10.0);
                            brain.gunangle = target_dir;
                        }
                    } else if rng.random::<f32>() < 0.25 {
                        add_gml_motion(
                            &mut brain,
                            &mut vel,
                            rng.random_range(0.0..std::f32::consts::TAU),
                            0.4,
                            dt,
                        );
                        brain.walk = 20.0 + rng.random_range(0.0..10.0);
                        brain.attack = GTimer::from_seconds(
                            (brain.walk + 2.0 + rng.random_range(0.0..5.0)) / 30.0,
                            TimerMode::Once,
                        );
                        brain.gunangle = brain.heading;
                    }
                } else if rng.random::<f32>() < 0.1 {
                    add_gml_motion(
                        &mut brain,
                        &mut vel,
                        rng.random_range(0.0..std::f32::consts::TAU),
                        0.4,
                        dt,
                    );
                    brain.walk = 20.0 + rng.random_range(0.0..10.0);
                    brain.attack = GTimer::from_seconds(
                        (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                        TimerMode::Once,
                    );
                    brain.gunangle = brain.heading;
                }
            }
        }
        if brain.sniper_aiming && brain.burst_timer.just_finished() {
            brain.wkick = 7.0;
            enemy_cue(&mut cues, "sndSniperFire");
            for offset in [4.0_f32, -4.0, 0.0] {
                let angle = brain.gunangle + offset.to_radians();
                let bullet = spawn_enemy_projectile(
                    &mut commands,
                    entity,
                    enemy.kind,
                    epos,
                    glam::Vec2::from_angle(angle) * 16.0 * 30.0,
                    3,
                    2.0,
                    3.5,
                    150.0,
                    false,
                );
                commands.entity(bullet).insert((
                    ProjectileTyp(1),
                    ProjectileFade("images/sprEnemyBulletHit.png"),
                ));
            }
            if let Some(target) = player_pos {
                brain.gunangle = (target - epos).y.atan2((target - epos).x);
            }
            brain.attack =
                GTimer::from_seconds((40.0 + rng.random_range(0.0..5.0)) / 30.0, TimerMode::Once);
            brain.sniper_aiming = false;
        }
        walk_step(&mut brain, &mut vel, 0.8, 1.5, dt);
        if brain.sniper_aiming
            && brain.burst_timer.remaining_secs() > 5.0 / 30.0
            && let Some(target) = player_pos
        {
            brain.gunangle = (target - epos).y.atan2((target - epos).x);
        }
        cap_gml_speed(&mut brain, &mut vel, 1.5);
        integrate_verbatim(
            &mut brain,
            &mut vel,
            &mut pos,
            &solids,
            &mask,
            &positions,
            enemy.kind,
            epos,
            enemy_def(enemy.kind).radius,
            dt,
            true,
            run.loop_count,
            current_frame,
            wall_law(enemy.kind),
        );
        zero_damage_contact_push(
            &mut brain,
            &mut vel,
            player_pos,
            pos.0,
            enemy_def(enemy.kind).radius,
            dt,
        );
    }
}

pub fn tick_jungle_bandit(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (Entity, &Enemy, &mut EnemyBrain, &mut Velocity, &mut Pos),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
) {
    let dt = time.delta_secs;
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let positions: Vec<(glam::Vec2, i32)> = enemies.iter().map(|(_, enemy, _, _, p) | (p.0, crate::enemy_data::gml_size(enemy.kind))).collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos) in &mut enemies {
        if enemy.kind != EnemyKind::JungleBandit {
            continue;
        }
        let epos = pos.0;
        tick_verbatim_cooldown(&mut brain, dt);
        sync_heading(&mut brain, &vel);
        brain.attack.tick(dt);
        brain.burst_timer.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds((10.0 + rng.random_range(0.0..5.0)) / 30.0, TimerMode::Once);
            if let Some(target) = player_pos {
                let target_dir = (target - epos).y.atan2((target - epos).x);
                if has_line_of_sight(epos, target, &mask) {
                    if epos.distance(target) > 48.0 {
                        if rng.random::<f32>() < 0.5 && epos.distance(target) <= 96.0 {
                            enemy_cue(&mut cues, "sndEnemyFire");
                            brain.gunangle = target_dir;
                            brain.ammo = 6;
                            brain.burst_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                            brain.attack = GTimer::from_seconds(
                                (20.0 + rng.random_range(0.0..5.0)) / 30.0,
                                TimerMode::Once,
                            );
                        } else {
                            set_gml_direction(
                                &mut brain,
                                &mut vel,
                                target_dir + rng.random_range(-90.0_f32..90.0).to_radians(),
                            );
                            set_gml_speed(&mut brain, &mut vel, 0.4);
                            brain.walk = 10.0 + rng.random_range(0.0..10.0);
                            brain.gunangle = target_dir;
                        }
                    } else {
                        set_gml_direction(
                            &mut brain,
                            &mut vel,
                            target_dir
                            + std::f32::consts::PI
                            + rng.random_range(-10.0_f32..10.0).to_radians(),
                        );
                        set_gml_speed(&mut brain, &mut vel, 0.4);
                        brain.walk = 40.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = target_dir;
                    }
                } else if rng.random::<f32>() < 0.25 {
                    add_gml_motion(
                        &mut brain,
                        &mut vel,
                        rng.random_range(0.0..std::f32::consts::TAU),
                        0.4,
                        dt,
                    );
                    brain.walk = 20.0 + rng.random_range(0.0..10.0);
                    brain.attack = GTimer::from_seconds(
                        (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                        TimerMode::Once,
                    );
                    brain.gunangle = brain.heading;
                }
            } else if rng.random::<f32>() < 0.1 {
                add_gml_motion(
                    &mut brain,
                    &mut vel,
                    rng.random_range(0.0..std::f32::consts::TAU),
                    0.4,
                    dt,
                );
                brain.walk = 20.0 + rng.random_range(0.0..10.0);
                brain.attack = GTimer::from_seconds(
                    (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                    TimerMode::Once,
                );
                brain.gunangle = brain.heading;
            }
        }
        if brain.burst_timer.just_finished() && brain.ammo > 0 {
            brain.wkick = 4.0;
            let angle = brain.gunangle + rng.random_range(-8.0_f32..8.0).to_radians();
            let bullet = spawn_enemy_projectile(
                &mut commands,
                entity,
                enemy.kind,
                epos,
                glam::Vec2::from_angle(angle) * (13.0 - rng.random_range(0.0_f32..2.0)) * 30.0,
                1,
                3.0,
                3.5,
                150.0,
                false,
            );
            commands.entity(bullet).insert((
                ProjectileFriction(0.6),
                BouncesLeft(255),
                ShellWallBounce {
                    add: 0.0,
                    cap: 18.0 * 30.0,
                    decay: 0.9,
                    rearm: None,
                },
                ProjectileTyp(1),
                ProjectileFade("images/sprEBullet3Disappear.png"),
            ));
            enemy_cue(&mut cues, "sndPopgun");
            brain.ammo -= 1;
            brain.burst_timer = GTimer::from_seconds(4.0 / 30.0, TimerMode::Once);
        }
        walk_step(&mut brain, &mut vel, 0.8, 3.5, dt);
        integrate_verbatim(
            &mut brain,
            &mut vel,
            &mut pos,
            &solids,
            &mask,
            &positions,
            enemy.kind,
            epos,
            enemy_def(enemy.kind).radius,
            dt,
            true,
            run.loop_count,
            current_frame,
            wall_law(enemy.kind),
        );
        zero_damage_contact_push(
            &mut brain,
            &mut vel,
            player_pos,
            pos.0,
            enemy_def(enemy.kind).radius,
            dt,
        );
    }
}

pub fn tick_melee_bandit(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (Entity, &Enemy, &mut EnemyBrain, &mut Velocity, &mut Pos),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
) {
    let dt = time.delta_secs;
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let positions: Vec<(glam::Vec2, i32)> = enemies.iter().map(|(_, enemy, _, _, p) | (p.0, crate::enemy_data::gml_size(enemy.kind))).collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos) in &mut enemies {
        // GML `JungleAssassin` declares `MeleeBandit` as its parent object, so
        // it inherits every one of MeleeBandit's events.
        if !matches!(enemy.kind, EnemyKind::MeleeBandit | EnemyKind::Assassin) {
            continue;
        }
        let epos = pos.0;
        tick_verbatim_cooldown(&mut brain, dt);
        sync_heading(&mut brain, &vel);
        let mut slash_armed_this_tick = false;
        if brain.weapon_alarm > 0.0 {
            brain.weapon_alarm -= dt * 30.0;
            if brain.weapon_alarm <= 0.0 {
                brain.weapon_alarm = 0.0;
                brain.wepangle = -brain.wepangle;
            }
        }
        brain.attack.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds((10.0 + rng.random_range(0.0..5.0)) / 30.0, TimerMode::Once);
            if let Some(target) = player_pos {
                let aim_dir = (target - epos).y.atan2((target - epos).x);
                let far_dir = (epos - target).y.atan2((epos - target).x);
                if has_line_of_sight(epos, target, &mask) {
                    if epos.distance(target) < 64.0 {
                        brain.weapon_alarm = 20.0;
                        enemy_cue(&mut cues, "sndAssassinAttack");
                        brain.wepangle = -brain.wepangle;
                        let gunangle = brain.gunangle;
                        add_gml_motion(&mut brain, &mut vel, gunangle, 6.0, dt);
                        brain.gunangle = aim_dir;
                        brain.slash_delay = 10.0;
                        slash_armed_this_tick = true;
                        spawn_hit_warning(&mut commands, epos);
                        brain.attack = GTimer::from_seconds(
                            (50.0 + rng.random_range(0.0..6.0)) / 30.0,
                            TimerMode::Once,
                        );
                    } else {
                        set_gml_direction(
                            &mut brain,
                            &mut vel,
                            far_dir + rng.random_range(-10.0_f32..10.0).to_radians(),
                        );
                        set_gml_speed(&mut brain, &mut vel, 0.4);
                        brain.walk = 40.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = aim_dir;
                    }
                } else if rng.random::<f32>() < 0.25 {
                    add_gml_motion(
                        &mut brain,
                        &mut vel,
                        rng.random_range(0.0..std::f32::consts::TAU),
                        0.4,
                        dt,
                    );
                    brain.walk = 20.0 + rng.random_range(0.0..10.0);
                    brain.attack = GTimer::from_seconds(
                        (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                        TimerMode::Once,
                    );
                    brain.gunangle = brain.heading;
                }
            } else if rng.random::<f32>() < 0.1 {
                add_gml_motion(
                    &mut brain,
                    &mut vel,
                    rng.random_range(0.0..std::f32::consts::TAU),
                    0.4,
                    dt,
                );
                brain.walk = 20.0 + rng.random_range(0.0..10.0);
                brain.attack = GTimer::from_seconds(
                    (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                    TimerMode::Once,
                );
                brain.gunangle = brain.heading;
            }
        }
        if brain.slash_delay > 0.0 && !slash_armed_this_tick {
            brain.slash_delay -= dt * 30.0;
            if brain.slash_delay <= 0.0 {
                brain.slash_delay = 0.0;
                let d = glam::Vec2::from_angle(
                    brain.gunangle + rng.random_range(-5.0_f32..5.0).to_radians(),
                );
                let slash = spawn_enemy_projectile(
                    &mut commands,
                    entity,
                    enemy.kind,
                    epos,
                    d * 2.0 * 30.0,
                    5,
                    0.4,
                    4.0,
                    150.0,
                    false,
                );
                commands.entity(slash).insert(ProjectileTyp(0));
                brain.wepangle *= -1.0;
                brain.attack = GTimer::from_seconds(
                    brain.attack.remaining_secs() + 7.0 / 30.0,
                    TimerMode::Once,
                );
            }
        }
        if brain.walk > 0.0 {
            brain.walk = (brain.walk - dt * 30.0).max(0.0);
            if brain.slash_delay == 0.0 {
                let heading = brain.heading;
                add_gml_motion(&mut brain, &mut vel, heading, 2.0, dt);
                if let Some(target) = player_pos {
                    potential_step_solid(
                        &mut pos.0,
                        target,
                        2.0,
                        enemy_def(enemy.kind).radius,
                        &solids,
                        Some(&mask),
                    );
                }
            }
        }
        cap_gml_speed(&mut brain, &mut vel, 3.0);
        integrate_verbatim(
            &mut brain,
            &mut vel,
            &mut pos,
            &solids,
            &mask,
            &positions,
            enemy.kind,
            epos,
            enemy_def(enemy.kind).radius,
            dt,
            true,
            run.loop_count,
            current_frame,
            wall_law(enemy.kind),
        );
        zero_damage_contact_push(
            &mut brain,
            &mut vel,
            player_pos,
            pos.0,
            enemy_def(enemy.kind).radius,
            dt,
        );
    }
}

pub fn tick_ballguy(
    time: Res<SimTime>,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    player_q: Query<(&Pos, &Health), (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            &mut Health,
            Option<&HurtAnim>,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
) {
    let dt = time.delta_secs;
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    let player = player_q.single().ok().map(|(p, h)| (p.0, h.hp > 0));
    let positions: Vec<(glam::Vec2, i32)> = enemies.iter().map(|(_, enemy, _, _, p, _, _) | (p.0, crate::enemy_data::gml_size(enemy.kind))).collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (_entity, enemy, mut brain, mut vel, mut pos, mut health, hurt) in &mut enemies {
        if enemy.kind != EnemyKind::Ballguy {
            continue;
        }
        let epos = pos.0;
        tick_verbatim_cooldown(&mut brain, dt);
        sync_heading(&mut brain, &vel);
        if let Some((target, alive)) = player
            && alive
            && !brain.close
            && epos.distance(target) < 64.0
        {
            enemy_cue(&mut cues, "sndFrogClose");
            brain.close = true;
        }
        brain.attack.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds((30.0 + rng.random_range(0.0..20.0)) / 30.0, TimerMode::Once);
            if let Some(target) = player.map(|(p, _)| p) {
                if has_line_of_sight(epos, target, &mask) {
                    set_gml_direction(
                        &mut brain,
                        &mut vel,
                        (target - epos).y.atan2((target - epos).x)
                            + rng.random_range(-10.0_f32..10.0).to_radians(),
                    );
                } else {
                    add_gml_motion(
                        &mut brain,
                        &mut vel,
                        rng.random_range(0.0..std::f32::consts::TAU),
                        0.5,
                        dt,
                    );
                }
            } else {
                add_gml_motion(
                    &mut brain,
                    &mut vel,
                    rng.random_range(0.0..std::f32::consts::TAU),
                    0.5,
                    dt,
                );
            }
        }
        if !hurt.is_some_and(|h| !h.timer.is_finished()) {
            let heading = brain.heading;
            add_gml_motion(&mut brain, &mut vel, heading, 0.6, dt);
        }
        set_gml_speed(&mut brain, &mut vel, 3.0);
        integrate_verbatim(
            &mut brain,
            &mut vel,
            &mut pos,
            &solids,
            &mask,
            &positions,
            enemy.kind,
            epos,
            enemy_def(enemy.kind).radius,
            dt,
            true,
            run.loop_count,
            current_frame,
            wall_law(enemy.kind),
        );
        if let Some((target, true)) = player
            && pos.0.distance(target) <= 8.0 + enemy_def(enemy.kind).radius
        {
            health.hp = 0;
            add_gml_motion(
                &mut brain,
                &mut vel,
                (pos.0 - target).y.atan2((pos.0 - target).x),
                1.0,
                dt,
            );
        }
    }
}

pub fn tick_bigmaggot_inspector(
    time: Res<SimTime>,
    mask: Res<FloorMask>,
    mut ctrl: Local<HashMap<Entity, f32>>,
    player_q: Query<(&Pos, &Player), (With<Player>, Without<Enemy>)>,
    mut player_vel: Query<&mut Velocity, (With<Player>, Without<Enemy>)>,
    enemies: Query<(Entity, &Enemy, &Pos), With<Enemy>>,
) {
    let Ok((player_pos, _)) = player_q.single() else {
        return;
    };
    let player_pos = player_pos.0;
    let dt = time.delta_secs;

    for (entity, enemy, pos) in &enemies {
        let epos = pos.0;
        let to_player = player_pos - epos;
        let dist = to_player.length();

        if enemy.kind == EnemyKind::IdpdInspector && dist < 240.0 && dist > 1.0 {
            let los = has_line_of_sight(epos, player_pos, &mask);
            if los {
                let t = ctrl.entry(entity).or_insert(0.0);
                *t += dt;
                if *t > 0.5 {
                    if let Ok(mut pv) = player_vel.single_mut() {
                        let pull = (epos - player_pos).normalize_or_zero() * 30.0;
                        pv.0 += pull * dt;
                        if pv.0.length() > 200.0 {
                            pv.0 = pv.0.normalize() * 200.0;
                        }
                    }
                }
            } else {
                ctrl.remove(&entity);
            }
        } else if enemy.kind == EnemyKind::IdpdInspector {
            ctrl.remove(&entity);
        }
    }
}

/// Fire-strip swap for the render phase (bevy `play_fire` parity:
/// repath the live anim to the `derive_fire_path` strip as a 0.25 s
/// oneshot; skipped for hurting enemies or missing art).
pub fn show_enemy_fire(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    entity: Entity,
    idle: &'static str,
    anim: Option<&mut SpriteAnim>,
    hurting: bool,
) {
    if hurting {
        return;
    }
    let Some(anim) = anim else {
        return;
    };
    let Some(fire_path) = derive_fire_path(idle) else {
        return;
    };
    let Some(def) = catalog.def(fire_path) else {
        return;
    };
    anim.set_path(fire_path, def, true);
    commands
        .entity(entity)
        .try_insert(crate::comps_b::FireAnim {
            idle,
            walk: None,
            timer: GTimer::from_seconds(0.25, TimerMode::Once),
        });
}

/// GML `spr_fire` table (per-kind `Create_0.gml`): idle strip -> fire
/// strip. Mimics are intentionally absent: their `*Fire` strip is the
/// `spr_walk` strip (`Mimic/Create_0.gml:7`), already covered by
/// `derive_walk_path`, and routing it through the `FireAnim` oneshot
/// would drop the walk strip on restore (`walk: None`).
pub fn derive_fire_path(idle: &'static str) -> Option<&'static str> {
    match idle {
        "images/sprBanditBossIdle.png" => Some("images/sprBanditBossFire.png"),
        "images/sprCrabIdle.png" => Some("images/sprCrabFire.png"),
        "images/sprCrownGuardianIdle.png" => Some("images/sprCrownGuardianFire.png"),
        "images/sprExploGuardianIdle.png" => Some("images/sprExploGuardianFire.png"),
        "images/sprFireBallerIdle.png" => Some("images/sprFireBallerFire.png"),
        "images/sprSuperFireBallerIdle.png" => Some("images/sprSuperFireBallerFire.png"),
        "images/sprFrogQueenIdle.png" => Some("images/sprFrogQueenFire.png"),
        "images/sprGoldScorpionIdle.png" => Some("images/sprGoldScorpionFire.png"),
        "images/sprGuardianIdle.png" => Some("images/sprGuardianFire.png"),
        "images/sprInvLaserCrystalIdle.png" => Some("images/sprInvLaserCrystalFire.png"),
        "images/sprJockIdle.png" => Some("images/sprJockFire.png"),
        "images/sprLaserCrystalIdle.png" => Some("images/sprLaserCrystalFire.png"),
        "images/sprLightningCrystalIdle.png" => Some("images/sprLightningCrystalFire.png"),
        "images/sprRatkingIdle.png" => Some("images/sprRatkingFire.png"),
        "images/sprSalamanderIdle.png" => Some("images/sprSalamanderFire.png"),
        "images/sprScorpionIdle.png" => Some("images/sprScorpionFire.png"),
        "images/sprScrapBossIdle.png" => Some("images/sprScrapBossFire.png"),
        "images/sprSnowBotIdle.png" => Some("images/sprSnowBotFire.png"),
        "images/sprTurretIdle.png" => Some("images/sprTurretFire.png"),
        "images/sprTurtleIdle.png" => Some("images/sprTurtleFire.png"),
        "images/sprWolfIdle.png" => Some("images/sprWolfFire.png"),
        _ => None,
    }
}

/// Shared single-projectile spawn for enemy fire (art handles omitted;
/// the renderer derives the strip from `Projectile` + owner kind).
fn spawn_enemy_projectile(
    commands: &mut Commands,
    owner: Entity,
    kind: EnemyKind,
    pos: glam::Vec2,
    vel: glam::Vec2,
    damage: i32,
    lifetime: f32,
    radius: f32,
    knockback: f32,
    explosive: bool,
) -> Entity {
    commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            Team::Enemy,
            Projectile {
                damage,
                life: GTimer::from_seconds(lifetime, TimerMode::Once),
                radius,
                knockback,
                explosive,
                source: Some(DamageSource::enemy(owner, kind)),
            },
            Velocity(vel),
            Pos(pos),
        ))
        .id()
}

/// Gator Alarm_2 pellet (friction 0.6, full bounce, deflectable shell;
/// JungleBandit fires the same object).
fn fire_enemy_shell(
    commands: &mut Commands,
    owner: Entity,
    kind: EnemyKind,
    pos: glam::Vec2,
    angle: f32,
    speed: f32,
) {
    let sdir = glam::Vec2::new(angle.cos(), angle.sin());
    let e = spawn_enemy_projectile(
        commands,
        owner,
        kind,
        pos + sdir * 16.0,
        sdir * speed,
        1,
        3.0,
        3.5,
        150.0,
        false,
    );
    commands.entity(e).insert((
        ProjectileFriction(0.6),
        BouncesLeft(255),
        // GML EnemyBullet3 wall: cap 18 px/step, decay 0.9, no re-arm.
        ShellWallBounce {
            add: 0.0,
            cap: 540.0,
            decay: 0.9,
            rearm: None,
        },
        ProjectileTyp(1),
        ProjectileFade("images/sprEBullet3Disappear.png"),
    ));
}

/// BuffGator Alarm_2 flak (friction 0.4, no direct damage; splits into
/// 16 shells on death).
fn fire_enemy_flak(
    commands: &mut Commands,
    owner: Entity,
    kind: EnemyKind,
    pos: glam::Vec2,
    angle: f32,
    speed: f32,
) {
    let sdir = glam::Vec2::new(angle.cos(), angle.sin());
    let e = spawn_enemy_projectile(
        commands,
        owner,
        kind,
        pos + sdir * 16.0,
        sdir * speed,
        0,
        3.0,
        6.0,
        150.0,
        false,
    );
    commands.entity(e).insert((
        ProjectileFriction(0.4),
        ProjectileTyp(1),
        ProjectileFade("images/sprEnemyBulletHit.png"),
        SplitOnDeath(SplitDef {
            pellets: 16,
            spread: std::f32::consts::PI,
            speed: 300.0,
            damage: 1,
            lifetime: 2.5,
            radius: 3.0,
            knockback: 150.0,
            color: [1.0, 0.7, 0.3, 1.0],
            size: glam::Vec2::new(8.0, 3.0),
        }),
    ));
}

/// Single aimed enemy bullet with table spread (euphoria slows enemy
/// shots to 80%, bevy parity).
#[allow(clippy::too_many_arguments)]
pub fn fire_enemy_bullet(
    commands: &mut Commands,
    rng: &mut impl RngExt,
    owner: Entity,
    enemy: &Enemy,
    def: EnemyDef,
    pos: glam::Vec2,
    dir: glam::Vec2,
    euphoria: bool,
) {
    let base = dir.y.atan2(dir.x);
    let angle = base + rng.random_range(-def.projectile_spread..def.projectile_spread);
    let shot_dir = glam::Vec2::new(angle.cos(), angle.sin());
    let speed = def.projectile_speed * if euphoria { 0.8 } else { 1.0 };
    let e = spawn_enemy_projectile(
        commands,
        owner,
        enemy.kind,
        pos + shot_dir * 20.0,
        shot_dir * speed,
        def.projectile_damage,
        def.projectile_lifetime,
        def.projectile_radius,
        150.0,
        explosive_kind(enemy.kind),
    );
    finish_enemy_bullet(&mut commands.entity(e), enemy.kind);
}

/// Only Jock rockets explode on contact (bevy parity).
pub fn explosive_kind(kind: EnemyKind) -> bool {
    matches!(kind, EnemyKind::Jock)
}

/// `Guardian/Alarm_1` fires 3 `GuardianBullet`s in one instant: a centre
/// shot at 1 px/step and a pair at +/-40 deg, 2 px/step. `ExploGuardian/
/// Alarm_2` fires 14 `ExploguardianBullet`s at 10 px/step, 24 deg apart.
fn fire_guardian_volley(
    commands: &mut Commands,
    rng: &mut impl RngExt,
    owner: Entity,
    kind: EnemyKind,
    pos: glam::Vec2,
    gunangle: f32,
    euphoria: bool,
) {
    let slow = if euphoria { 0.8 } else { 1.0 };
    let shoot = |commands: &mut Commands, angle: f32, speed: f32| {
        let d = glam::Vec2::new(angle.cos(), angle.sin());
        let e = spawn_enemy_projectile(
            commands,
            owner,
            kind,
            pos,
            d * (speed * 30.0 * slow),
            5,
            3.0,
            4.0,
            150.0,
            false,
        );
        finish_enemy_bullet(&mut commands.entity(e), kind);
    };
    if kind == EnemyKind::Guardian {
        for (off, spd) in [(-40.0_f32, 2.0_f32), (0.0, 1.0), (40.0, 2.0)] {
            shoot(commands, gunangle + off.to_radians(), spd);
        }
    } else {
        let start = rng.random_range(0.0..std::f32::consts::TAU);
        for i in 0..14 {
            shoot(commands, start + (i as f32 * 24.0).to_radians(), 10.0);
        }
    }
}

/// Per-kind projectile traits: slash `typ` (0 = ignores slashes,
/// 1 = deflectable, 2 = slash-destructible), contact fade, and the
/// gator-family friction+bounce shell profile.
pub fn finish_enemy_bullet(ec: &mut EntityCommands, kind: EnemyKind) {
    let typ = match kind {
        EnemyKind::Scorpion | EnemyKind::GoldScorpion => 2,
        EnemyKind::Guardian | EnemyKind::Turtle => 0,
        EnemyKind::ExploGuardian | EnemyKind::Jock => 2,
        _ => 1,
    };
    ec.insert(ProjectileTyp(typ));
    if !explosive_kind(kind) {
        let fade_path = if matches!(kind, EnemyKind::Scorpion | EnemyKind::GoldScorpion) {
            "images/sprScorpionBulletHit.png"
        } else if matches!(
            kind,
            EnemyKind::IdpdGrunt | EnemyKind::IdpdInspector | EnemyKind::IdpdElite
        ) {
            "images/sprIDPDBulletHit.png"
        } else if matches!(
            kind,
            EnemyKind::Gator
                | EnemyKind::BuffGator
                | EnemyKind::JungleBandit
                | EnemyKind::Molesarge
        ) {
            "images/sprEBullet3Disappear.png"
        } else {
            "images/sprEnemyBulletHit.png"
        };
        ec.insert(ProjectileFade(fade_path));
    }
    if matches!(
        kind,
        EnemyKind::Gator | EnemyKind::BuffGator | EnemyKind::JungleBandit | EnemyKind::Molesarge
    ) {
        ec.insert(ProjectileFriction(0.6));
        ec.insert(BouncesLeft(255));
        ec.insert(ShellWallBounce {
            add: 0.0,
            cap: 540.0,
            decay: 0.9,
            rearm: None,
        });
    }
}

/// Per-pellet aim offsets and random spread for `fire_enemy_shot`, in radians.
///
/// GML spells most of these out per object rather than deriving them:
/// `Molesarge/Alarm_1` fires at `gunangle + {0, -15, +15, -30, +30}` with no
/// extra jitter, `SuperFireBaller/Alarm_1` uses `orandom(6)`, and
/// `Molefish/Alarm_1` uses `random(4) - 2`. Everything else keeps the evenly
/// spaced table fan and the table's spread.
fn gml_fan(kind: EnemyKind, total: usize, table_spread: f32) -> ([f32; 8], f32) {
    let mut offsets = [0.0f32; 8];
    match kind {
        EnemyKind::Molesarge => {
            const D: [f32; 5] = [0.0, -15.0, 15.0, -30.0, 30.0];
            for (i, o) in offsets.iter_mut().enumerate().take(5) {
                *o = D[i].to_radians();
            }
            (offsets, 0.0)
        }
        EnemyKind::SuperFireBaller => {
            for o in offsets.iter_mut().take(total) {
                *o = 0.0;
            }
            (offsets, 6.0f32.to_radians())
        }
        EnemyKind::Molefish => (offsets, 2.0f32.to_radians()),
        _ => {
            for i in 0..offsets.len().min(total.max(1)) {
                offsets[i] = (i as f32 - (total as f32 - 1.0) * 0.5) * table_spread;
            }
            (offsets, 0.06)
        }
    }
}

/// Fan volley: `bullets_per_shot` pellets around the aim with per-pellet
/// jitter (SuperFireBaller's per-pellet speeds preserved).
#[allow(clippy::too_many_arguments)]
pub fn fire_enemy_shot(
    commands: &mut Commands,
    rng: &mut impl RngExt,
    owner: Entity,
    enemy: &Enemy,
    def: EnemyDef,
    pos: glam::Vec2,
    dir: glam::Vec2,
) {
    let base = dir.y.atan2(dir.x);
    let total = def.bullets_per_shot;
    let (offsets, spread) = gml_fan(enemy.kind, total, def.fan_spread);
    for i in 0..total {
        let offset = offsets.get(i).copied().unwrap_or(0.0);
        let angle = base + offset + rng.random_range(-spread..spread);
        let shot_dir = glam::Vec2::new(angle.cos(), angle.sin());
        // `Molesarge/Alarm_1` re-rolls `10 + random(2)` per pellet.
        let speed = if enemy.kind == EnemyKind::Molesarge {
            rng.random_range(10.0..12.0) * crate::SIM_HZ as f32
        } else if enemy.kind == EnemyKind::SuperFireBaller {
            [90.0, 120.0, 150.0][(i as usize).min(2)]
        } else {
            def.projectile_speed
        };
        let e = spawn_enemy_projectile(
            commands,
            owner,
            enemy.kind,
            pos + shot_dir * 20.0,
            shot_dir * speed,
            def.projectile_damage,
            def.projectile_lifetime,
            def.projectile_radius,
            150.0,
            explosive_kind(enemy.kind),
        );
        finish_enemy_bullet(&mut commands.entity(e), enemy.kind);
    }
}

/// Flush a kill-gated boss spawn: once enough trash died, pop the boss
/// out of a wall (or open floor), shake, toast, and hitstop (bevy
/// parity, including the hardcoded "BIG BANDIT" toast).
#[allow(clippy::too_many_arguments)]
pub fn tick_delayed_boss_spawns(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    run: Res<Run>,
    scarier: Res<ScarierFace>,
    heavy_heart: Res<HeavyHeart>,
    mask: Res<FloorMask>,
    mut trauma: ResMut<Trauma>,
    mut hitstop: ResMut<HitStop>,
    mut toast: ResMut<Toast>,
    pending: Query<(Entity, &PendingDelayedBoss)>,
    enemies: Query<&Enemy, With<Enemy>>,
    player_q: Query<&Pos, With<Player>>,
    walls: Query<
        (
            Entity,
            &crate::comps_a::WallCell,
            &Pos,
            Option<&crate::comps_b::ScreenEnd>,
        ),
        With<crate::comps_a::WallTile>,
    >,
) {
    let Ok((marker_e, pending_boss)) = pending.single() else {
        return;
    };

    let living_trash = enemies.iter().filter(|e| !enemy_def(e.kind).boss).count() as u32;
    let killed = pending_boss.initial_trash.saturating_sub(living_trash);
    if killed < pending_boss.kills_needed() {
        return;
    }

    let Ok(player_pos) = player_q.single() else {
        return;
    };
    let player_pos = player_pos.0;

    let mut best_wall: Option<(glam::Vec2, (i32, i32))> = None;
    let mut best_score = f32::MAX;
    if pending_boss.from_wall {
        for (_, cell, pos, screen_end) in &walls {
            let p = pos.0;
            let d = p.distance(player_pos);
            if d < 120.0 || d > 260.0 {
                continue;
            }
            let mut score = (d - 180.0).abs() + (p.y - player_pos.y).abs() * 0.25;
            if screen_end.is_some() {
                score -= 20.0;
            }
            if score < best_score {
                best_score = score;
                best_wall = Some((p, (cell.0, cell.1)));
            }
        }
    }

    let spawn_pos = if let Some((p, _)) = best_wall {
        p
    } else {
        let mut rng = rand::rng();
        let mut best = mask.random_floor_pos(&mut rng, 120.0);
        for _ in 0..32 {
            let ang = rng.random_range(0.0..std::f32::consts::TAU);
            let cand =
                player_pos + glam::Vec2::new(ang.cos(), ang.sin()) * rng.random_range(140.0..240.0);
            if mask.is_walkable(cand) {
                best = cand;
                break;
            }
        }
        best
    };

    commands.entity(marker_e).despawn();
    trauma.add(0.3);

    if let Some((p, cell)) = best_wall {
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)] {
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                crate::comps_a::PendingWallBreak {
                    cell: (cell.0 + dx, cell.1 + dy),
                    pos: p,
                    spawn_floor: true,
                },
            ));
        }
    }

    spawn_enemy_at(
        &mut commands,
        &catalog,
        pending_boss.kind,
        spawn_pos,
        difficulty_multiplier(run.floor),
        false,
        false,
        run.loop_count,
        EnemySpawnContext {
            subarea: run.floor_in_area,
            blood_crown: run.blood_crown,
            scarier_face: scarier.0,
            heavy_heart: heavy_heart.0,
        },
    );

    commands.spawn((
        GameCleanup,
        BossIntro {
            timer: GTimer::from_seconds(1.1, TimerMode::Once),
        },
    ));
    toast.show("BIG BANDIT");
    hitstop.trigger(0.2, 0.15);
}

pub fn tick_frog_eggs(
    time: Res<SimTime>,
    mut commands: Commands,
    run: Res<Run>,
    mut q: Query<(Entity, &Enemy, &mut EnemyBrain, &Pos), With<Enemy>>,
) {
    for (e, enemy, mut brain, pos) in &mut q {
        if enemy.kind != EnemyKind::FrogEgg {
            continue;
        }
        brain.attack.tick(time.delta_secs);
        if !brain.attack.just_finished() {
            continue;
        }
        let hatch = pos.0;
        commands.entity(e).despawn();

        queue_enemy_spawn(
            &mut commands,
            EnemyKind::Ballguy,
            hatch,
            1.0,
            run.loop_count,
        );

        for i in 0..8 {
            let ang = (i as f32) * std::f32::consts::TAU / 8.0;
            let d = glam::Vec2::new(ang.cos(), ang.sin());
            spawn_enemy_projectile(
                &mut commands,
                e,
                enemy.kind,
                hatch,
                d * 240.0,
                3,
                1.1,
                4.0,
                100.0,
                false,
            );
        }
    }
}

/// GML `LilHunter/Destroy_0:13-21` head spawn (`LilHunterDie/Create_0`):
/// speed 2 px/step toward the nearest player (else random),
/// `sndLilHunterBreak` + `PortalClear`.
/// (`scrOnPopoKill` has no port equivalent — skipped with this note;
/// its music-cue side effects ride the audio layer.)
pub fn spawn_lil_hunter_die(
    commands: &mut Commands,
    cues: &mut Queue<AudioCue>,
    pos: glam::Vec2,
    team: Team,
    target: Option<Entity>,
    target_pos: Option<glam::Vec2>,
) -> Entity {
    let mut rng = rand::rng();
    let dir = target_pos
        .map(|t| (t - pos).normalize_or_zero())
        .filter(|d| d.length_squared() > 0.001)
        .unwrap_or_else(|| glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU)));
    let e = commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            team,
            LilHunterDie {
                trn: (rng.random_range(0.0..5.0) + 5.0)
                    * if rng.random_bool(0.5) { 1.0 } else { -1.0 },
                bounces: 0,
                target,
                ticks: 0,
            },
            Velocity(dir * 2.0 * 30.0),
            Pos(pos),
            FxAngle(dir.y.atan2(dir.x).to_degrees() - 90.0),
        ))
        .id();
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        PortalClear {
            timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
            scale: 1.0,
        },
        Pos(pos),
    ));
    cues.push(AudioCue {
        name: "sndLilHunterBreak",
        volume: 0.8,
        variance: 0.2,
    });
    e
}

/// GML `LilHunter/Destroy_0` 80-`TrapFire` ring (`sprFireLilHunter`,
/// speed 2+random(0.2) px/step stepping 4.5°, `move_contact_solid`
/// baked as a clamped ≤12 px slide, team inherited).
pub fn spawn_lil_hunter_trapfire(commands: &mut Commands, at: glam::Vec2, team: Team) {
    let mut rng = rand::rng();
    let mut ang = rng.random_range(0.0..std::f32::consts::TAU);
    for _ in 0..80 {
        ang += 4.5f32.to_radians();
        let d = glam::Vec2::from_angle(ang);
        let speed = (2.0 + rng.random_range(0.0..0.2)) * 30.0;
        let mut p = at + d * 12.0;
        p.x = p.x.clamp(-ARENA_W * 0.5 + 8.0, ARENA_W * 0.5 - 8.0);
        p.y = p.y.clamp(-ARENA_H * 0.5 + 8.0, ARENA_H * 0.5 - 8.0);
        crate::environment::spawn_trap_fire_with_image(
            commands,
            p,
            d,
            speed,
            team,
            None,
            "images/sprFireLilHunter.png",
            Some(d.y.atan2(d.x)),
        );
    }
}

/// GML `LilHunterDie/Step_0` + `Collision_Wall`: per-step Smoke,
/// `image_angle = direction-90`, accel while `bounces <= 3`
/// (`speed < 6 → +2`, `+0.05`; `direction += trn`;
/// `trn += random(1)-0.5`; past 3 `direction += 59`), wall bounces
/// counted (past 3 the head stalls; otherwise a `PortalClear` pops).
pub fn tick_lil_hunter_die(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(
        Entity,
        &mut Pos,
        &mut Velocity,
        &mut LilHunterDie,
        Option<&mut FxAngle>,
    )>,
) {
    let dt = time.delta_secs;
    let mut rng = rand::rng();
    for (e, mut pos, mut vel, mut die, angle) in &mut q {
        die.ticks += 1;
        if die.ticks > 15 * 30 {
            commands.entity(e).despawn();
            continue;
        }
        spawn_burst(
            &mut commands,
            &mut rng,
            pos.0,
            1,
            [0.7, 0.7, 0.7, 1.0],
            (20.0, 60.0),
        );
        // GML speeds are px/step; the sim runs px/s (×30).
        let mut speed = vel.0.length() / 30.0;
        let mut dir_ang = if speed > 0.01 {
            vel.0.y.atan2(vel.0.x)
        } else {
            0.0
        };
        if die.bounces <= 3 {
            if speed < 6.0 {
                speed += 2.0;
            }
            speed += 0.05;
            dir_ang += die.trn.to_radians();
            die.trn += rng.random_range(0.0..1.0) - 0.5;
        } else {
            dir_ang += 59.0f32.to_radians();
        }
        vel.0 = glam::Vec2::new(dir_ang.cos(), dir_ang.sin()) * speed * 30.0;
        pos.0 += vel.0 * dt;
        let r = 8.0;
        let mut bounced = false;
        if pos.0.x < -ARENA_W * 0.5 + r || pos.0.x > ARENA_W * 0.5 - r {
            vel.0.x = -vel.0.x;
            bounced = true;
        }
        if pos.0.y < -ARENA_H * 0.5 + r || pos.0.y > ARENA_H * 0.5 - r {
            vel.0.y = -vel.0.y;
            bounced = true;
        }
        pos.0.x = pos.0.x.clamp(-ARENA_W * 0.5 + r, ARENA_W * 0.5 - r);
        pos.0.y = pos.0.y.clamp(-ARENA_H * 0.5 + r, ARENA_H * 0.5 - r);
        if bounced {
            die.bounces += 1;
            if die.bounces > 3 {
                // GML `Collision_Wall`: `alarm[2] = 15; speed = 0`.
                vel.0 = glam::Vec2::ZERO;
            } else {
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    PortalClear {
                        timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                        scale: 1.0,
                    },
                    Pos(pos.0),
                ));
            }
        }
        if let Some(mut a) = angle {
            a.0 = vel.0.y.atan2(vel.0.x).to_degrees() - 90.0;
        }
        let _ = e;
    }
}

/// Verbatim `objects/ScrapBossMissile` law (`Other_10` + `Alarm_0`):
/// homing drift 0.1 px/tick toward the target at forced speed 2 px/tick,
/// and on loops > 0 a trail bullet (`EnemyBullet1` stats: damage 3 at own
/// speed + 2 px/tick) every `max(1, 12 - loops)` ticks.
pub fn tick_scrap_missiles(
    time: Res<SimTime>,
    mut commands: Commands,
    run: Res<Run>,
    mask: Res<FloorMask>,
    catalog: Res<repame_anim::AnimCatalog>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
    mut q: Query<
        (
            Entity,
            &mut ScrapBossMissileState,
            &mut Health,
            &mut Velocity,
            &mut Pos,
            Option<&mut SpriteAnim>,
        ),
        (
            With<ScrapBossMissileState>,
            With<Enemy>,
            Without<Player>,
            Without<Prop>,
        ),
    >,
) {
    let dt = time.delta_secs;
    let player_pos = player_q.single().ok().map(|p| p.0);
    let solids = prop_shapes(&props);
    for (entity, mut state, mut health, mut vel, mut pos, mut anim) in &mut q {
        state.fuse.tick(dt);
        if state.hurt {
            state.hurt_timer.tick(dt);
            if state.hurt_timer.just_finished() {
                state.hurt = false;
                commands.entity(entity).remove::<HurtAnim>();
            }
        }
        if state.fuse.just_finished() {
            health.hp = 0;
            continue;
        }
        if health.hp <= 0 {
            continue;
        }

        if !state.hurt
            && let Some(target) = player_pos
        {
            let to_player = target - pos.0;
            gml_motion_add_clamp(&mut vel.0, to_player.normalize_or_zero(), 0.1, 2.0, dt);
        }
        if vel.0.length_squared() > 0.001 {
            vel.0 = vel.0.normalize() * 60.0;
        } else if let Some(target) = player_pos {
            vel.0 = (target - pos.0).normalize_or_zero() * 60.0;
        }

        let wall = move_bounce_solid(
            &mut pos.0,
            &mut vel.0,
            SCRAP_BOSS_MISSILE_RADIUS,
            dt,
            &solids,
            Some(&mask),
            true,
        );
        if wall.is_some() {
            health.hp -= 3;
            state.hurt = true;
            state.hurt_timer = GTimer::from_seconds(50.0 / 30.0, TimerMode::Once);
            state.fuse = GTimer::from_seconds(50.0 / 30.0, TimerMode::Once);
            if let Some(anim) = anim.as_deref_mut()
                && let Some(def) = catalog.def("images/sprScrapBossMissileHurt.png")
            {
                anim.set_path("images/sprScrapBossMissileHurt.png", def, true);
            }
            commands.entity(entity).insert(HurtAnim {
                idle: "images/sprScrapBossMissileIdle.png",
                walk: Some("images/sprScrapBossMissileIdle.png"),
                hurt: "images/sprScrapBossMissileHurt.png",
                timer: GTimer::from_seconds(50.0 / 30.0, TimerMode::Once),
                was_moving: false,
            });
        }

        if run.loop_count > 0 {
            state.trail_timer.tick(dt);
            if state.trail_timer.just_finished() {
                state.trail_timer = GTimer::from_seconds(
                    (12u32.saturating_sub(run.loop_count).max(1)) as f32 / 30.0,
                    TimerMode::Once,
                );
                let heading = vel.0.normalize_or_zero();
                let source = state
                    .creator
                    .map(|creator| DamageSource::enemy(creator, EnemyKind::BigDog))
                    .or(Some(DamageSource::enemy(
                        entity,
                        EnemyKind::ScrapBossMissile,
                    )));
                let trail = commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Team::Enemy,
                    Projectile {
                        damage: 3,
                        life: GTimer::from_seconds(2.5, TimerMode::Once),
                        radius: 4.0,
                        knockback: 120.0,
                        explosive: false,
                        source,
                    },
                    ProjectileTyp(1),
                    Velocity(heading * 120.0),
                    Pos(pos.0),
                ));
                let _ = trail;
            }
        }
    }
}

pub fn tick_toxic_gas(
    time: Res<SimTime>,
    mut commands: Commands,
    run: Res<Run>,
    mask: Res<FloorMask>,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
    frame: Res<CurrentFrame>,
    mut sets: ParamSet<(
        Query<
            (
                Entity,
                &mut ToxicGasState,
                &mut Pos,
                &mut Velocity,
                Option<&mut NativeAngle>,
            ),
            (With<ToxicGasState>, Without<Prop>),
        >,
        Query<
            (
                Entity,
                &Pos,
                &Team,
                &Hitbox,
                &mut Health,
                Option<&mut NextHurt>,
                Option<&Enemy>,
                Option<&crate::comps_a::RaceState>,
            ),
            (
                With<Hitbox>,
                Or<(With<Enemy>, With<crate::comps_a::Player>)>,
                Without<Projectile>,
                Without<ToxicGasState>,
                Without<Prop>,
            ),
        >,
    )>,
) {
    let dt = time.delta_secs;
    let solids = prop_shapes(&props);
    let mut converted = Vec::<Entity>::new();
    let gas_entities: Vec<Entity> = {
        let gases = sets.p0();
        gases.iter().map(|(entity, _, _, _, _)| entity).collect()
    };
    for gas_entity in gas_entities {
        let gas_data = {
            let mut gases = sets.p0();
            let Ok((_, mut gas, mut pos, mut vel, angle)) = gases.get_mut(gas_entity) else {
                continue;
            };
            gas.age += dt * 30.0;
            if gas.age >= 10.0 {
                gas.typ = 2;
            }
            apply_gml_friction(&mut vel.0, gas.friction, dt);
            move_bounce_solid(
                &mut pos.0,
                &mut vel.0,
                gas.radius,
                dt,
                &solids,
                Some(&mask),
                true,
            );
            if gas.scale + gas.grow_speed < 1.0 {
                gas.scale += gas.grow_speed;
            }
            if gas.grow_speed > -0.005 {
                gas.grow_speed -= 0.0002;
            }
            if gas.scale < 0.4 {
                gas.grow_speed -= 0.01;
                if let Some(mut angle) = angle {
                    angle.0 += gas.rot.to_radians() * dt * 30.0;
                }
            }
            if gas.scale < 0.0 {
                commands.entity(gas_entity).despawn();
                None
            } else {
                Some((pos.0, gas.radius))
            }
        };
        let Some((gas_pos, gas_radius)) = gas_data else {
            continue;
        };

        let target = {
            let targets = sets.p1();
            targets.iter().find_map(
                |(
                    target_entity,
                    target_pos,
                    target_team,
                    target_hitbox,
                    target_health,
                    next_hurt,
                    enemy,
                    player_race,
                )| {
                    if converted.contains(&target_entity)
                        || target_health.hp <= 0
                        || gas_pos.distance(target_pos.0) > gas_radius + target_hitbox.radius
                    {
                        return None;
                    }
                    let kind = enemy.map(|e| e.kind);
                    let conversion =
                        *target_team == Team::Enemy && kind == Some(EnemyKind::Ballguy);
                    if *target_team == Team::Enemy
                        && matches!(
                            kind,
                            Some(EnemyKind::SuperFrog) | Some(EnemyKind::FrogQueen)
                        )
                    {
                        return None;
                    }
                    if *target_team == Team::Player {
                        if player_race.is_some_and(|race| race.race == crate::data::RaceId::Frog)
                            || !target_health.invuln.is_finished()
                        {
                            return None;
                        }
                    } else if next_hurt.as_ref().is_some_and(|next| next.0 > frame.0) {
                        return None;
                    }
                    Some((target_entity, target_pos.0, *target_team, conversion))
                },
            )
        };
        let Some((target_entity, target_pos, target_team, conversion)) = target else {
            continue;
        };
        if conversion {
            queue_enemy_spawn(
                &mut commands,
                EnemyKind::SuperFrog,
                target_pos,
                1.0,
                run.loop_count,
            );
            commands.entity(target_entity).despawn();
            converted.push(target_entity);
            continue;
        }

        let mut targets = sets.p1();
        let Ok((_, _, _, _, mut target_health, mut next_hurt, _, _)) =
            targets.get_mut(target_entity)
        else {
            continue;
        };
        if target_team == Team::Player {
            target_health.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
        }
        target_health.hp -= 1;
        if target_team == Team::Enemy
            && let Some(next) = next_hurt.as_deref_mut()
        {
            next.0 = frame.0 + 5;
        }
        commands.entity(gas_entity).despawn();
    }
}

/// Verbatim `objects/Throne2Ball` law (`Step_0`): friction 0.25 bleeds
/// speed; once stalled, `timeout` accrues and past 15 ticks the ball
/// sprays an `EnemyBullet2` (Horror stats: damage 2 at 10 px/tick) along
/// the latched `angle` every tick, dying past `40 + loops * 10` ticks.
/// While stalled but young it aims at the nearest player (±30 degrees).
/// (Position integration rides `move_projectiles`; the aim-converge
/// particles are visual-only.)
pub fn tick_throne_balls(
    time: Res<SimTime>,
    mut commands: Commands,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut q: Query<(Entity, &mut Velocity, &Pos, &mut ThroneBall), With<Projectile>>,
) {
    let Ok(player_pos) = player_q.single() else {
        return;
    };
    let dt = time.delta_secs;
    let cap = 40.0 + run.loop_count as f32 * 10.0;
    let mut rng = rand::rng();
    for (entity, mut vel, pos, mut ball) in &mut q {
        crate::comps_a::apply_gml_friction(&mut vel.0, 0.25, dt);
        if vel.0.length() >= 1.0 {
            continue;
        }
        vel.0 = glam::Vec2::ZERO;
        ball.timeout += dt * 30.0;
        if ball.timeout > cap {
            commands.entity(entity).despawn();
            continue;
        }
        if ball.timeout > 15.0 {
            ball.sounded = false;
            let sdir = glam::Vec2::from_angle(ball.angle);
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                Team::Enemy,
                Projectile {
                    damage: 2,
                    life: GTimer::from_seconds(2.0, TimerMode::Once),
                    radius: 4.5,
                    knockback: 120.0,
                    explosive: false,
                    source: Some(DamageSource::enemy(entity, EnemyKind::ThroneII)),
                },
                Velocity(sdir * 300.0),
                Pos(pos.0 + sdir * 12.0),
            ));
        } else {
            let aim = (player_pos.0 - pos.0).y.atan2((player_pos.0 - pos.0).x);
            ball.angle = aim + rng.random_range(-30.0..=30.0_f32).to_radians();
        }
    }
}

/// Verbatim `objects/MomProjectile/Step_0`: every tick the gas mortar
/// leaves a `ToxicGas` cloud at its tail.
pub fn tick_mom_shots(
    mut commands: Commands,
    q: Query<(&Team, &Pos), (With<Projectile>, With<MomShot>)>,
) {
    let mut rng = rand::rng();
    for (team, pos) in &q {
        let off = glam::Vec2::new(rng.random_range(-2.0..=2.0), 0.0);
        let a = rng.random_range(0.0..std::f32::consts::TAU);
        let speed = rng.random_range(0.2..1.7) * 30.0;
        let mut gas = ToxicGasState::new();
        gas.grow_speed = 0.003 + rng.random_range(0.0..0.002);
        gas.rot =
            (1.0 + rng.random_range(0.0..=3.0)) * if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        spawn_toxic_gas(
            &mut commands,
            pos.0 + off,
            glam::Vec2::from_angle(a) * speed,
            gas,
        );
        let _ = team;
    }
}

/// Verbatim `objects/SuperFrog/Alarm_2`: every 3 ticks a stray
/// `EnemyBullet2` (damage 2 at 2 px/tick) leaves at a random angle.
pub fn tick_super_frogs(
    mut commands: Commands,
    mut ticks: Local<HashMap<Entity, u8>>,
    q: Query<(Entity, &Enemy, &Pos), With<Enemy>>,
) {
    let mut rng = rand::rng();
    for (entity, enemy, pos) in &q {
        if enemy.kind != EnemyKind::SuperFrog {
            continue;
        }
        let t = ticks.entry(entity).or_insert(0);
        *t = t.wrapping_add(1);
        if *t % 3 != 0 {
            continue;
        }
        let a = rng.random_range(0.0..std::f32::consts::TAU);
        let d = glam::Vec2::from_angle(a);
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            Team::Enemy,
            Projectile {
                damage: 2,
                life: GTimer::from_seconds(2.5, TimerMode::Once),
                radius: 4.0,
                knockback: 120.0,
                explosive: false,
                source: Some(DamageSource::enemy(entity, enemy.kind)),
            },
            Velocity(d * 60.0),
            Pos(pos.0 + d * 10.0),
        ));
    }
}

/// Verbatim `objects/EliteInspector` law (`Alarm_1`/`Alarm_2`/`Other_10`):
/// freeze-gated control field that drags projectiles and pulls the
/// player, close-range baton dash-slash (`EnemySlash` damage 8), and
/// `PopoNade` lobs at the last-seen position (5-grenade budget).
///
/// Register map: `brain.attack` = `alarm[1]`, `brain.slash_delay` =
/// `alarm[2]` countdown (ticks), `brain.ammo` = `grenades`,
/// `brain.burst_left` = `freeze`, `brain.walk` = `walk`,
/// `brain.gunangle` = `gunangle` (radians), `brain.strafe_dir` = `control`
/// (0/1), `boss.target`-equivalent heading in `EnemyBrain` is unused so
/// the move heading rides `Velocity`, last-seen rides a local map.
/// (Baton art, `wepangle` flips, and enter/taunt sounds are out.)
#[allow(clippy::too_many_arguments)]
pub fn tick_elite_inspectors(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut player_vel: Query<&mut Velocity, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            &Health,
        ),
        With<Enemy>,
    >,
    mut shots: Query<
        (&mut Pos, &Team, Option<&ProjectileTyp>, Option<&PopoNadeM>),
        (With<Projectile>, Without<Enemy>, Without<Player>),
    >,
    mut last_seen: Local<HashMap<Entity, glam::Vec2>>,
    mut headings: Local<HashMap<Entity, glam::Vec2>>,
    mut inited: Local<std::collections::HashSet<Entity>>,
) {
    let Ok(player_pos) = player_q.single() else {
        return;
    };
    let player_speed_sq = player_vel
        .single()
        .map(|v| v.0.length_squared())
        .unwrap_or(0.0);
    let dt = time.delta_secs;
    let mut rng = rand::rng();
    inited.retain(|e| enemies.contains(*e));
    last_seen.retain(|e, _| enemies.contains(*e));
    headings.retain(|e, _| enemies.contains(*e));
    let nades_live = shots.iter().filter(|(_, _, _, m)| m.is_some()).count();

    for (entity, enemy, mut brain, mut vel, mut pos, health) in &mut enemies {
        if enemy.kind != EnemyKind::EliteInspector {
            continue;
        }
        let epos = pos.0;
        if !inited.contains(&entity) {
            inited.insert(entity);
            // GML `Create_0`: `walk = 30`, `grenades = 5`,
            // `alarm[1] = 30 + random(15)`.
            brain.walk = 30.0;
            brain.ammo = 5;
            brain.burst_left = 0;
            brain.strafe_dir = 0.0;
            brain.slash_delay = 0.0;
            brain.gunangle = rng.random_range(0.0..std::f32::consts::TAU);
            brain.attack =
                GTimer::from_seconds(rng.random_range(30.0..=45.0) / 30.0, TimerMode::Once);
            last_seen.insert(entity, epos);
        }

        let to_player = player_pos.0 - epos;
        let dist = to_player.length();
        let aim = to_player.y.atan2(to_player.x);
        let los = has_line_of_sight(epos, player_pos.0, &mask);
        let freeze = brain.burst_left;

        // GML `Other_10`: freeze accrues while the target moves (or the
        // inspector is damaged); +3 while the target can shoot (port:
        // any living player counts as armed).
        if player_speed_sq > 0.001 || health.hp < health.max {
            brain.burst_left += 1;
        }
        brain.burst_left += 3;

        // Walk locomotion along the latched heading, capped at
        // 3.5 px/tick.
        let head = headings.get(&entity).copied().unwrap_or(glam::Vec2::X);
        if brain.walk > 0.0 {
            gml_motion_add_clamp(&mut vel.0, head, 0.8, 3.5, dt);
            brain.walk -= dt * 30.0;
            if brain.walk < 0.0 {
                brain.walk = 0.0;
            }
        }
        if vel.0.length() > 105.0 {
            vel.0 = vel.0.normalize() * 105.0;
        }

        // Control field: repel foreign projectiles, pull the player.
        if brain.strafe_dir == 1.0 {
            for (mut spos, team, typ, _) in &mut shots {
                if *team == Team::Enemy {
                    continue;
                }
                if typ.is_some_and(|t| t.0 == 0) {
                    continue;
                }
                let push = (spos.0 - epos).normalize_or_zero() * 60.0 * dt;
                spos.0 += push;
            }
            if dist < 160.0 && dist > 0.001 {
                if let Ok(mut pv) = player_vel.single_mut() {
                    pv.0 += to_player.normalize_or_zero() * 60.0 * dt;
                }
            }
        }

        // Baton slash delay (`alarm[2]`).
        if brain.slash_delay > 0.0 {
            brain.slash_delay -= dt * 30.0;
            if brain.slash_delay <= 0.0 {
                brain.slash_delay = 0.0;
                let sdir = glam::Vec2::from_angle(brain.gunangle);
                vel.0 += sdir * 180.0;
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Team::Enemy,
                    Projectile {
                        damage: 8,
                        life: GTimer::from_seconds(0.4, TimerMode::Once),
                        radius: 10.0,
                        knockback: 150.0,
                        explosive: false,
                        source: Some(DamageSource::enemy(entity, enemy.kind)),
                    },
                    Velocity(sdir * 60.0),
                    Pos(epos + sdir * 12.0),
                ));
                brain.attack =
                    GTimer::from_seconds(rng.random_range(15.0..=20.0) / 30.0, TimerMode::Once);
            }
        }

        // GML `Alarm_1` (brain).
        brain.attack.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds(rng.random_range(20.0..=30.0) / 30.0, TimerMode::Once);
            if brain.strafe_dir == 1.0 && rng.random::<f32>() < 0.5 {
                brain.strafe_dir = 0.0;
            }
            if rng.random::<f32>() < 0.5 && freeze > 40 {
                brain.strafe_dir = 1.0;
                let d = brain.attack.duration() + 10.0 / 30.0;
                brain.attack = GTimer::from_seconds(d, TimerMode::Once);
            }
            if los {
                brain.gunangle = aim;
                last_seen.insert(entity, player_pos.0);
                if rng.random::<f32>() < 2.0 / 3.0 && freeze > 40 && dist < 64.0 {
                    // Telegraph the baton dash.
                    brain.slash_delay = 5.0;
                    let d = brain.attack.duration() + 5.0 / 30.0;
                    brain.attack = GTimer::from_seconds(d, TimerMode::Once);
                    spawn_hit_warning(&mut commands, epos);
                    brain.walk = 0.0;
                } else {
                    let head = glam::Vec2::from_angle(
                        aim + rng.random_range(-20.0..=20.0_f32).to_radians(),
                    );
                    headings.insert(entity, head);
                    vel.0 = head * 12.0;
                    brain.walk = rng.random_range(30.0..=40.0);
                    let d = brain.attack.duration() / 2.0;
                    brain.attack = GTimer::from_seconds(d, TimerMode::Once);
                }
            } else if rng.random::<f32>() < 2.0 / 3.0 {
                let head = glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU));
                headings.insert(entity, head);
                brain.gunangle = head.y.atan2(head.x);
                brain.walk = rng.random_range(20.0..=30.0);
                vel.0 = head * 12.0;
            } else {
                // `PopoNade` at the last-seen position (budget-gated).
                let seen = last_seen.get(&entity).copied().unwrap_or(player_pos.0);
                let seen_d = seen.distance(player_pos.0);
                let self_seen_d = epos.distance(seen);
                let gated = rng.random::<f32>() < 1.0 / (3.0 + nades_live as f32 * 3.0)
                    && brain.ammo > 0
                    && freeze > 40
                    && dist < 160.0
                    && seen_d < 160.0
                    && self_seen_d > 64.0;
                if gated || rng.random::<f32>() < 1.0 / 8.0 {
                    if brain.ammo > 0 {
                        brain.ammo -= 1;
                    }
                    let nang = (seen - epos).y.atan2((seen - epos).x)
                        + rng.random_range(-10.0..=10.0_f32).to_radians();
                    let ndir = glam::Vec2::from_angle(nang);
                    brain.gunangle = nang;
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Projectile {
                            damage: 0,
                            life: GTimer::from_seconds(3.0, TimerMode::Once),
                            radius: 6.0,
                            knockback: 0.0,
                            explosive: true,
                            source: Some(DamageSource::enemy(entity, enemy.kind)),
                        },
                        ProjectileTyp(1),
                        PopoNadeM,
                        crate::comps_b::CustomExplosion {
                            radius: 70.0,
                            count: 1,
                            spread: 0.0,
                            visual: Some(crate::comps_b::NativeExplosionKind::Popo),
                        },
                        Velocity(ndir * 300.0),
                        Pos(epos + ndir * 12.0),
                    ));
                }
            }
        }

        pos.0 += vel.0 * dt;
        clamp_to_arena(&mut pos.0, 10.0);
    }
}

/// Verbatim `objects/EliteShielder` law (`Alarm_1`/`Alarm_2`/`Other_10`):
/// freeze-gated 6-round `PopoPlasma` bursts (damage 8 at 1.5 px/tick),
/// and the `EliteShield` carry that teleports the shielder to a floor
/// 120..300 px away. Register map mirrors the Inspector
/// (`brain.attack` = `alarm[1]`, `brain.slash_delay` = `alarm[2]`
/// countdown, `brain.ammo` = burst rounds, `brain.burst_left` = `freeze`).
/// (The shield's projectile-block field has no port equivalent and is
/// out; the teleport + disappear poof are kept.)
#[allow(clippy::too_many_arguments)]
pub fn tick_elite_shielders(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    player_q: Query<(&Pos, &Velocity), (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            &Health,
        ),
        With<Enemy>,
    >,
    mut headings: Local<HashMap<Entity, glam::Vec2>>,
    mut inited: Local<std::collections::HashSet<Entity>>,
) {
    let Ok((player_pos, player_v)) = player_q.single() else {
        return;
    };
    let dt = time.delta_secs;
    let mut rng = rand::rng();
    inited.retain(|e| enemies.contains(*e));
    headings.retain(|e, _| enemies.contains(*e));

    for (entity, enemy, mut brain, mut vel, mut pos, health) in &mut enemies {
        if enemy.kind != EnemyKind::EliteShielder {
            continue;
        }
        let epos = pos.0;
        if !inited.contains(&entity) {
            inited.insert(entity);
            // GML `Create_0`: `walk = 30`, `freeze = 20`,
            // `alarm[1] = 30 + random(15)`.
            brain.walk = 30.0;
            brain.ammo = 0;
            brain.burst_left = 20;
            brain.slash_delay = 0.0;
            brain.gunangle = rng.random_range(0.0..std::f32::consts::TAU);
            brain.attack =
                GTimer::from_seconds(rng.random_range(30.0..=45.0) / 30.0, TimerMode::Once);
            headings.insert(entity, glam::Vec2::X);
        }

        let to_player = player_pos.0 - epos;
        let dist = to_player.length();
        let aim = to_player.y.atan2(to_player.x);
        let los = has_line_of_sight(epos, player_pos.0, &mask);
        let freeze = brain.burst_left;

        // GML `Other_10` freeze (target `can_shoot` reads true on players).
        if player_v.0.length_squared() > 0.001 && health.hp < health.max {
            brain.burst_left += 1;
        }
        brain.burst_left += 3;

        let head = headings.get(&entity).copied().unwrap_or(glam::Vec2::X);
        if brain.walk > 0.0 {
            gml_motion_add_clamp(&mut vel.0, head, 0.8, 3.5, dt);
            brain.walk -= dt * 30.0;
            if brain.walk < 0.0 {
                brain.walk = 0.0;
            }
        }
        if vel.0.length() > 105.0 {
            vel.0 = vel.0.normalize() * 105.0;
        }

        // Plasma burst tick (`alarm[2]`).
        if brain.slash_delay > 0.0 {
            brain.slash_delay -= dt * 30.0;
            if brain.slash_delay <= 0.0 {
                if brain.ammo > 0 {
                    let jitter = rng.random_range(-10.0..=10.0_f32).to_radians();
                    let sdir = glam::Vec2::from_angle(brain.gunangle + jitter);
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Projectile {
                            damage: 8,
                            life: GTimer::from_seconds(3.0, TimerMode::Once),
                            radius: 5.0,
                            knockback: 60.0,
                            explosive: false,
                            source: Some(DamageSource::enemy(entity, enemy.kind)),
                        },
                        ProjectileTyp(2),
                        Velocity(sdir * 45.0),
                        Pos(epos + sdir * 12.0),
                    ));
                    // Recoil 0.5 px/tick opposite the muzzle.
                    vel.0 += glam::Vec2::from_angle(brain.gunangle + std::f32::consts::PI) * 15.0;
                    brain.slash_delay = 4.0;
                    brain.ammo -= 1;
                } else {
                    brain.slash_delay = 0.0;
                }
            }
        }

        // GML `Alarm_1` (brain).
        brain.attack.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds(rng.random_range(15.0..=20.0) / 30.0, TimerMode::Once);
            if los {
                brain.gunangle = aim;
                if rng.random::<f32>() < 3.0 / 4.0 && freeze > 40 && dist < 150.0 {
                    brain.ammo = 6;
                    brain.slash_delay = 5.0;
                    brain.attack = GTimer::from_seconds(20.0 / 30.0, TimerMode::Once);
                } else if rng.random::<f32>() < 1.0 / 3.0 {
                    // `EliteShield` carry: teleport to a floor 120..300
                    // away, poof at the destination.
                    let mut dest = epos;
                    for _ in 0..100 {
                        let a = rng.random_range(0.0..std::f32::consts::TAU);
                        let d = rng.random_range(120.0..=300.0);
                        let cand = epos + glam::Vec2::from_angle(a) * d;
                        if mask.is_walkable(cand) {
                            dest = cand;
                            break;
                        }
                    }
                    pos.0 = dest;
                    // `EliteShield` anchor: pins the creator 60 ticks,
                    // blocks incoming fire, then poofs (`Alarm_0`).
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Pos(dest),
                        EliteBlocker {
                            owner: entity,
                            timer: GTimer::from_seconds(60.0 / 30.0, TimerMode::Once),
                        },
                    ));
                    brain.attack = GTimer::from_seconds(85.0 / 30.0, TimerMode::Once);
                    vel.0 = glam::Vec2::ZERO;
                    brain.walk = 0.0;
                } else {
                    // GML compares against the constant `target.y > 64`
                    // (a bool): reproduce the literal law.
                    let probe = glam::Vec2::new(player_pos.0.x, 1.0);
                    let head = if epos.distance(probe) > 64.0 {
                        glam::Vec2::from_angle(
                            aim + rng.random_range(-25.0..=25.0_f32).to_radians(),
                        )
                    } else {
                        glam::Vec2::from_angle(
                            aim + std::f32::consts::PI
                                + rng.random_range(-45.0..=45.0_f32).to_radians(),
                        )
                    };
                    headings.insert(entity, head);
                    vel.0 = head * 12.0;
                    brain.walk = rng.random_range(10.0..=20.0);
                    if freeze < 40 {
                        let d = brain.attack.duration() + rng.random_range(0.0..=30.0) / 30.0;
                        brain.attack = GTimer::from_seconds(d, TimerMode::Once);
                    }
                }
            } else if rng.random::<f32>() < 1.0 / 3.0 {
                let head = glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU));
                headings.insert(entity, head);
                brain.gunangle = head.y.atan2(head.x);
                brain.walk = rng.random_range(20.0..=30.0);
                vel.0 = head * 12.0;
            } else if freeze > 40 && rng.random::<f32>() < 0.25 {
                let mut dest = epos;
                for _ in 0..100 {
                    let a = rng.random_range(0.0..std::f32::consts::TAU);
                    let d = rng.random_range(120.0..=300.0);
                    let cand = epos + glam::Vec2::from_angle(a) * d;
                    if mask.is_walkable(cand) {
                        dest = cand;
                        break;
                    }
                }
                pos.0 = dest;
                // `EliteShield` anchor: pins the creator 60 ticks,
                // blocks incoming fire, then poofs (`Alarm_0`).
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Team::Enemy,
                    Pos(dest),
                    EliteBlocker {
                        owner: entity,
                        timer: GTimer::from_seconds(60.0 / 30.0, TimerMode::Once),
                    },
                ));
                brain.attack = GTimer::from_seconds(75.0 / 30.0, TimerMode::Once);
                vel.0 = glam::Vec2::ZERO;
                brain.walk = 0.0;
            }
        }

        pos.0 += vel.0 * dt;
        clamp_to_arena(&mut pos.0, 11.0);
    }
}

/// Verbatim `objects/EliteShield` law (`Step_0` + `Alarm_0` +
/// `Collision_projectile`): while armed the anchor pins its creator to
/// itself, deflects `typ` 1 shots back to the IDPD team, destroys `typ`
/// 2 shots, then poofs away.
pub fn tick_elite_blockers(
    time: Res<SimTime>,
    mut commands: Commands,
    mut blockers: Query<(Entity, &Pos, &mut EliteBlocker), Without<Enemy>>,
    mut owners: Query<&mut Pos, With<Enemy>>,
    mut shots: Query<
        (
            Entity,
            &Pos,
            &mut Team,
            &mut Velocity,
            Option<&ProjectileTyp>,
        ),
        (With<Projectile>, Without<Enemy>),
    >,
) {
    let dt = time.delta_secs;
    for (b, bpos, mut blocker) in &mut blockers {
        blocker.timer.tick(dt);
        if blocker.timer.just_finished() {
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                Pos(bpos.0),
                StaticFx {
                    path: "images/sprEliteShielderShieldDisappear.png",
                },
                PickupLifetime {
                    timer: GTimer::from_seconds(1.0, TimerMode::Once),
                },
            ));
            commands.entity(b).despawn();
            continue;
        }
        if let Ok(mut opos) = owners.get_mut(blocker.owner) {
            opos.0 = bpos.0;
        }
        for (s, spos, mut team, mut vel, typ) in &mut shots {
            if *team != Team::Player {
                continue;
            }
            if spos.0.distance(bpos.0) > 20.0 {
                continue;
            }
            match typ.map(|t| t.0).unwrap_or(0) {
                1 => {
                    // Deflect: re-team + fling away from the shield.
                    *team = Team::Enemy;
                    let away = (spos.0 - bpos.0).normalize_or_zero();
                    let speed = vel.0.length().max(60.0);
                    vel.0 = away * speed;
                }
                2 => {
                    commands.entity(s).despawn();
                }
                _ => {}
            }
        }
    }
}

/// Verbatim `objects/ProtoStatue` law (`Step_0`): rad snapshots the
/// player's rads on placement; past 24 rads the statue charges (halves
/// its own HP, +2 IDPD portals); past 30% damage it phases (+2 IDPD
/// portals, once). Portals roll the GML `IDPDSpawn` table.
pub fn tick_proto_statues(
    mut commands: Commands,
    mut run: ResMut<Run>,
    player_q: Query<(&Player, &Pos), (With<Player>, Without<Enemy>)>,
    mut q: Query<(Entity, &Enemy, &mut Health, &Pos, &mut ProtoGuardian), With<Enemy>>,
) {
    let Ok((player, player_pos)) = player_q.single() else {
        return;
    };
    for (_, enemy, mut health, pos, mut statue) in &mut q {
        if enemy.kind != EnemyKind::ProtoStatue {
            continue;
        }
        if !statue.init {
            statue.init = true;
            statue.rad = player.rads;
        }
        let mut waves = 0;
        if statue.rad > 24 && !statue.charged {
            statue.charged = true;
            health.hp = (health.hp / 2).max(1);
            waves += 2;
        }
        if !statue.phased && (health.hp as f32) < (health.max as f32) * 0.7 && health.hp > 0 {
            statue.phased = true;
            waves += 2;
        }
        for _ in 0..waves {
            run.popolevel += 1;
            for kind in crate::idpd::roll_idpd_table(run.loop_count, run.area, run.popolevel, false)
            {
                queue_enemy_spawn(&mut commands, kind, pos.0, 1.0, run.loop_count);
            }
        }
        let _ = player_pos;
    }
}

/// Boss intro banner timing (bevy parity).
pub fn tick_boss_intro(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut BossIntro)>,
) {
    for (e, mut intro) in &mut q {
        intro.timer.tick(time.delta_secs);
        if intro.timer.just_finished() {
            commands.entity(e).despawn();
        }
    }
}

/// Melee telegraph marker (GML `sprAssassinNotice` at y-16): sim side
/// keeps the timed marker; the sprite resolves renderer-side.
fn spawn_hit_warning(commands: &mut Commands, pos: glam::Vec2) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        HitWarning {
            timer: GTimer::from_seconds(0.5, TimerMode::Once),
        },
        Pos(pos + glam::Vec2::new(0.0, -16.0)),
    ));
}

/// Expire telegraph markers (bonus port — bevy `tick_hit_warnings`
/// minus the anim-end branch, which has no headless equivalent).
pub fn tick_hit_warnings(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut HitWarning)>,
) {
    for (e, mut w) in &mut q {
        w.timer.tick(time.delta_secs);
        if w.timer.just_finished() {
            commands.entity(e).despawn();
        }
    }
}

/// PopoShield follower tracking (bonus port — bevy
/// `tick_shield_followers` minus the rotation write, which the renderer
/// derives from the owner's `gunangle`).
pub fn tick_shield_followers(
    mut commands: Commands,
    owners: Query<(Entity, &Pos, &EnemyBrain), With<Enemy>>,
    mut shields: Query<(Entity, &ShieldFollower, &mut Pos), Without<Enemy>>,
) {
    for (e, sh, mut pos) in shields.iter_mut() {
        if let Ok((_, owner_pos, brain)) = owners.get(sh.owner) {
            let base = brain.gunangle;
            let off = glam::Vec2::new(base.cos(), base.sin()) * 16.0;
            pos.0 = owner_pos.0 + off;
        } else {
            commands.entity(e).despawn();
        }
    }
}

/// GML `enemy/Collision_enemy`: a 1 px/frame velocity impulse away from each
/// overlapping enemy, gated on `size <= other.size` and — once
/// `busycollisions` is false (`GameCont.loops > 3`) — only on frames where
/// `current_frame % 30` is 0. Each contributing pair also draws the two
/// `orandom(1)` jitters and caps speed at 16 px/frame.
fn separate(
    positions: &[(glam::Vec2, i32)],
    epos: glam::Vec2,
    vel: &mut glam::Vec2,
    kind: EnemyKind,
    loops: u32,
    current_frame: u64,
) {
    let size = crate::enemy_data::gml_size(kind);
    let gated = loops > 3 && current_frame % 30 != 0;
    if gated {
        return;
    }
    let mut rng = rand::rng();
    for (other, other_size) in positions {
        if size > *other_size {
            continue;
        }
        if epos.distance(*other) >= 28.0 {
            continue;
        }
        let jitter = glam::Vec2::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );
        let away = (epos - (*other + jitter)).normalize_or_zero();
        *vel += away * crate::SIM_HZ as f32;
    }
    *vel = vel.clamp_length_max(16.0 * crate::SIM_HZ as f32);
}

/// Corpse slide + expiry (bevy `enemies.rs:2552` parity: `Corpse` life
/// ticks, corpses drift with GML 0.4 friction, expiry despawns).
/// Port adaptation: `Transform.translation` is [`Pos`] here; corpses
/// without [`Velocity`] (player-kill drops) only tick life.
///
/// Also slides `GroundPhysics` gibs/debris (player-death blood gibs):
/// GML gives them flat friction like every ground slide, and nothing
/// else ticks them — without this they coast at full speed for their
/// whole 0.9 s life and land ~144 px away.
pub fn tick_corpses(
    time: Res<SimTime>,
    mask: Res<FloorMask>,
    mut commands: Commands,
    walls: Query<
        &Pos,
        (
            With<WallCell>,
            With<WallTile>,
            Without<Corpse>,
            Without<crate::comps_b::GroundPhysics>,
        ),
    >,
    mut q: Query<
        (
            Entity,
            &mut Corpse,
            Option<&mut Velocity>,
            Option<&mut Pos>,
            Option<&mut CorpseCollision>,
        ),
        With<Corpse>,
    >,
    mut gibs: Query<
        (&mut Pos, &mut crate::comps_b::GroundPhysics),
        (Without<Corpse>, Without<Pickup>),
    >,
) {
    let dt = time.delta_secs;
    let solids: Vec<_> = walls
        .iter()
        .map(|pos| (pos.0, glam::Vec2::splat(8.0)))
        .collect();
    for (e, mut c, vel, pos, collision) in &mut q {
        c.life.tick(dt);
        if c.life.just_finished() {
            commands.entity(e).despawn();
            continue;
        }
        if let (Some(mut v), Some(mut p)) = (vel, pos) {
            apply_gml_friction(&mut v.0, 0.4, dt);
            if let Some(mut collision) = collision {
                v.0 = v.0.clamp_length_max(16.0 * 30.0);
                move_bounce_solid(
                    &mut p.0,
                    &mut v.0,
                    collision.radius,
                    dt,
                    &solids,
                    Some(&mask),
                    false,
                );
                collision.settled = v.0 == glam::Vec2::ZERO;
            } else {
                p.0 += v.0 * dt;
            }
        }
    }
    for (mut p, mut g) in &mut gibs {
        apply_gml_friction(&mut g.vel, 0.4, dt);
        p.0 += g.vel * dt;
    }
}

#[cfg(test)]
mod spawn_hp_tests {
    use super::*;
    use crate::enemy_data::enemy_def;

    #[test]
    fn loop0_matches_table() {
        for kind in [
            EnemyKind::Bandit,
            EnemyKind::FrogQueen,
            EnemyKind::Captain,
            EnemyKind::YvBoss,
            EnemyKind::ProtoStatue,
            EnemyKind::Scorpion,
        ] {
            assert_eq!(spawn_hp(kind, enemy_def(kind).hp, 0), enemy_def(kind).hp);
        }
    }

    /// Every boss expression is followed by `enemy/Create_0:7`'s universal
    /// `*= 1 + loops / 20`, which the old port dropped for the kinds that
    /// carry their own multiplier.
    #[test]
    fn boss_laws_include_the_universal_loop_term() {
        // ceil(100 * 2) * 1.15
        assert_eq!(spawn_hp(EnemyKind::BigBandit, 100, 3), 230);
        // ceil(490 * 2) * 1.15
        assert_eq!(spawn_hp(EnemyKind::FrogQueen, 490, 3), 1127);
        // 1100 * 2 * 1.15
        assert_eq!(spawn_hp(EnemyKind::Captain, 1100, 3), 2530);
        // 1500 * 2 * 1.15
        assert_eq!(spawn_hp(EnemyKind::Throne, 1500, 3), 3450);
        // 140 * 2 * 1.15
        assert_eq!(spawn_hp(EnemyKind::LilHunter, 140, 3), 322);
        // 550 * 2 * 1.15
        assert_eq!(spawn_hp(EnemyKind::Hyper, 550, 3), 1265);
        // 350 * 2 * 1.15
        assert_eq!(spawn_hp(EnemyKind::Technomancer, 350, 3), 805);
    }

    #[test]
    fn flat_and_default_laws() {
        // ProtoStatue: 120 * (1 + 5/10) * (1 + 5/20)
        assert_eq!(spawn_hp(EnemyKind::ProtoStatue, 120, 5), 225);
        // 700 * (1 + 4/20)
        assert_eq!(spawn_hp(EnemyKind::YvBoss, 700, 4), 840);
        assert_eq!(spawn_hp(EnemyKind::Scorpion, 16, 20), 32);
        // ScrapBoss: ceil(300 * 6) * 1.3
        assert_eq!(spawn_hp(EnemyKind::BigDog, 300, 6), 2340);
        // MeleeFake's parent is `prop`, so no loop scaling at all.
        assert_eq!(spawn_hp(EnemyKind::MeleeFake, 8, 9), 8);
        assert_eq!(scarier_spawn_hp(EnemyKind::Assassin, 7, 1), 5);
    }
}
