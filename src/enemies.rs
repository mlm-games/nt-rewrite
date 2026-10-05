//! Enemy AI. Ported from the bevy reference `game/enemies.rs` (`enemy_ai`
/// and its spawn/fire/tick helpers); positions as [`Pos`] (`Vec2`), not
/// `Transform.translation` (`Vec3`). Sprite/rotation writes, hurt/fire strip
/// swaps, `Juice::pop_in` and `VfxSpawner` bursts stay render-side; bevy played
/// audio by direct asset load inside the 16-param system, here stems queue as
/// [`AudioCue`]s through the [`Queue`].
/// Timer adaptation: bevy `Timer` -> [`GTimer`]; `tick()` returns `()` then
/// `just_finished()`/`finished()` are queried. bevy `ready_timer()` (finished
/// from birth, silent until re-armed) has no direct `GTimer` equivalent -
/// `GTimer::disarmed()` reports `just_finished()` on *every* tick - so
/// [`ready_timer`] double-ticks a 10 ms `Once` timer into the same observable
/// state (finished, not just-finished).
/// `enemy_ai` carries 14 params, under the bevy_ecs 16-param cap; the Inspector
/// tail lives in [`tick_bigmaggot_inspector`].
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
    ARENA_H, ARENA_W, BossIntro, BouncesLeft, CurrentFrame, DamageSource, DogGuardianLeap,
    DogGuardianPose, Euphoria, FireCooldown, FloorMask, GameCleanup, Health, HeavyHeart, Hitbox,
    Homing, LevelCleanup, NextHurt, Player, Projectile, ProjectileFade, ProjectileFriction,
    ProjectileTyp, ProjectileVisual, Run, ScarierFace, ShellWallBounce, SplitOnDeath, Team, Toast,
    Velocity, WallCell, WallTile, apply_gml_friction, gml_motion_add_clamp,
};
use crate::comps_b::{
    BossBrain, Corpse, CorpseCollision, CrownPedestal, EliteBlocker, Enemy, EnemyBrain, FxAngle,
    GmlImage, HitWarning, HurtAnim, IdpdShieldUnit, IdpdVanBrain, LilHunterDie, MaggotSpawnCharge,
    MaggotSpawnInternalDrain, MomShot, NativeAngle, NativeDepth, NecroReviveArea,
    PendingDelayedBoss, Pickup, PickupLifetime, PopoNadeM, PopoShieldM, PortalClear, Prop,
    PropSprites, ProtoGuardian, SCRAP_BOSS_MISSILE_RADIUS, ScrapBossMissileState, ShieldFollower,
    SpecialPropDeath, StaticFx, ThroneBall, ToxicGasState, YvCouch,
};
use crate::data::{AreaId, EnemyKind, SplitDef};
use crate::effects::{HitStop, spawn_burst};
use crate::enemy_data::{EnemyDef, enemy_def};
use crate::msg::Queue;
use crate::spatial::{
    Pos, build_solid_shapes, clamp_to_arena, contact_at, move_bounce_solid, potential_step_solid,
    resolve_prop_collision,
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

/// GML spawn-HP law verbatim: the object's own `Create_0` expression,
/// then `enemy/Create_0:7` scales *every* enemy by `max_hp *= 1 +
/// loops/20`. `base_hp` is the table (loop-0) value.
/// Boss: `1 + loops/3` except ScrapBoss `/1.2` and ProtoStatue flat 120 -
/// BigBandit, Throne, ThroneII, Hyper, Technomancer, LilHunter,
/// FrogQueen, Last (base 1100). Their coop `((pc/2)+0.5)` factor is 1.0
/// solo: GML `/` is float division, so `(1/2)+0.5 = 1.0`.
/// No loop term: YV (700 + coop-only scaling). `Mom`/`Captain`/
/// `OldGuardian`/`PalaceGuardian` have no GML object (spawn-table-only
/// kinds) and ride the default `/20` law.
pub fn spawn_hp(kind: EnemyKind, base_hp: i32, loops: u32) -> i32 {
    let l = loops as f32;
    // `ceil` only where the object's own `Create_0` line uses it; GML keeps
    // `hp` a real otherwise.
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
        // `MeleeFake` and `IceFlower` are `prop`-parented, so they never pick
        // up the universal `1 + loops / 20`.
        EnemyKind::MeleeFake | EnemyKind::IceFlower => return base_hp.max(1),
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

/// Full enemy spawn: base bundle from [`crate::setup::spawn_enemy`], plus
/// difficulty/face/heart scaling, the randomized [`EnemyBrain`] state,
/// boss/IDPD brains, and a [`SpriteAnim`] seed when the catalog carries
/// the idle strip (so `tick_fire_anims` can resolve
/// [`crate::comps_b::FireAnim`] from [`show_enemy_fire`]).
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
    if kind == EnemyKind::DogGuardian {
        ec.insert(DogGuardianLeap {
            pose: DogGuardianPose::Ground,
            z: 0.0,
            zspeed: 0.0,
        });
    }
    // GML `Create_0` `alarm[1]`, in frames. `random(n)` yields 0..n-1.
    let attack_frames = match kind {
        EnemyKind::MaggotSpawn | EnemyKind::FiredMaggot => 0.0,
        EnemyKind::Rat | EnemyKind::Bandit | EnemyKind::SnowBandit | EnemyKind::JungleBandit => {
            30.0 + rng.random_range(0.0..90.0)
        }
        EnemyKind::FastRat | EnemyKind::Ratking => 1.0 + rng.random_range(0.0..90.0),
        EnemyKind::RobotGuard => 80.0,
        EnemyKind::Maggot
        | EnemyKind::RadMaggot
        | EnemyKind::FireBaller
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
        // `Wolf/Create_0:18` `60 + random(40)`, `HostileHorror/Create_0:16`
        // `60 + random(10)`, `LilHunter/Create_0:22` `irandom_range(30, 120)`.
        EnemyKind::Wolf => 60.0 + rng.random_range(0.0..40.0),
        EnemyKind::HostileHorror => 60.0 + rng.random_range(0.0..10.0),
        EnemyKind::LilHunter => rng.random_range(30.0..121.0),
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
        attack,
        fire_alarm: GTimer::from_seconds(attack_frames / 30.0, TimerMode::Once),
        burst_left: 0,
        burst_timer: ready_timer(),
        dash: 0.0,
        strafe_dir: if rng.random_bool(0.5) { 1.0 } else { -1.0 },
        strafe_timer: GTimer::from_seconds(rng.random_range(0.8..1.6), TimerMode::Once),
        melee: ready_timer(),
        wkick: 0.0,
        walk: if kind == EnemyKind::IdpdElite {
            30.0
        } else {
            0.0
        },
        slash_delay: 0.0,
        ammo: match kind {
            EnemyKind::Scorpion | EnemyKind::GoldScorpion => 10,
            EnemyKind::JungleFly => 3,
            // GML `EliteGrunt/Create_0:39` `ammo = 3` is the `Alarm_2`
            // burst count, distinct from `grenades = 4`.
            EnemyKind::IdpdElite => 3,
            EnemyKind::Jock => 5,
            _ => 0,
        },
        grenades: match kind {
            // GML `Create_0` `grenades`: Grunt 2, Inspector 4, EliteGrunt 4.
            EnemyKind::IdpdGrunt => 2,
            EnemyKind::IdpdInspector | EnemyKind::IdpdElite => 4,
            _ => 0,
        },
        // GML `Create_0` `roll = 1` and `angle = 0` for all four.
        roll: true,
        roll_angle: 0.0,
        // GML `EliteGrunt/Create_0:40` `fuel = 100`; the other three do
        // not carry a fuel register.
        fuel: 100.0,
        // GML `Create_0` `freeze`: Grunt/EliteGrunt/Inspector 0,
        // Shielder 20.
        freeze: if kind == EnemyKind::IdpdShield {
            20.0
        } else {
            0.0
        },
        last_seen: pos,
        right: if rng.random_bool(0.5) { 1.0 } else { -1.0 },
        control: false,
        ratking_spawns: 0.0,
        ratking_rage: false,
        // GML `Spider/Create_0:19` `maxspeed = 3`; `Alarm_1` raises it to 5
        // only on the close chase.
        maxspeed: if matches!(kind, EnemyKind::Spider | EnemyKind::InvSpider) {
            3.0
        } else {
            f32::INFINITY
        },
        gunoffset: 0.0,
        walkdir: initial_heading,
        // GML `Salamander/Create_0:5` `wave = random(6.2)`;
        // `HostileHorror/Create_0:19` `charge = 0`.
        wave: rng.random_range(0.0..6.2),
        charge: 0.0,
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
        EnemyKind::Hyper => {
            ec.insert(crate::comps_b::HyperState::default());
        }
        EnemyKind::Technomancer => {
            ec.insert(crate::comps_b::TechnomancerState::default());
        }
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

/// Enemies that spawned inside a solid cell (the secret-entrance guard ring in
/// `setup::spawn_secret_entrances`, the vault-statue bandits, the boss
/// coordinates) start in contact, so `move_contact_solid` walks them out while
/// `separate`'s jitter flings them.
pub fn unstuck_enemies(mask: Res<FloorMask>, mut q: Query<(&mut Pos, &mut Velocity), With<Enemy>>) {
    for (mut pos, mut vel) in &mut q {
        if mask.snap_inside(&mut pos.0) {
            vel.0 = glam::Vec2::ZERO;
        }
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
/// parity - see module docs for why `GTimer::disarmed()` is wrong here).
fn ready_timer() -> GTimer {
    let mut t = GTimer::from_seconds(0.01, TimerMode::Once);
    t.tick(0.01);
    t.tick(0.0);
    t
}

/// Enemy telegraph cue through the sim audio queue (GML `snd_play`/
/// `snd_play_hit`/`snd_play_hit_big` all hand the instance
/// `UberCont.opt_sndvol`, so the cue sits at 1.0 vol, 0.05 variance; the bevy
/// build loaded `audio/{stem}.wav` directly to dodge the 16-param cap - the
/// queue carries the stem name instead and the audio layer resolves it).
fn enemy_cue(cues: &mut Queue<AudioCue>, stem: &'static str) {
    cues.push(AudioCue {
        name: stem,
        volume: 1.0,
        variance: 0.05,
    });
}

/// True when a kind owns a dedicated verbatim `tick_*` in this module. The
/// generic block in [`enemy_ai`] must skip these, or they fire twice per
/// cycle: once through their own GML `Alarm_1` port and once through the
/// table-driven fallback.
fn has_dedicated_tick(kind: EnemyKind) -> bool {
    matches!(
        kind,
        EnemyKind::Bandit
            | EnemyKind::SnowBandit
            | EnemyKind::Scorpion
            | EnemyKind::GoldScorpion
            | EnemyKind::JungleBandit
            | EnemyKind::JungleFly
            | EnemyKind::Sniper
            | EnemyKind::IdpdGrunt
            | EnemyKind::IdpdElite
            | EnemyKind::IdpdShield
            | EnemyKind::IdpdInspector
            | EnemyKind::Turret
            | EnemyKind::Ratking
    )
}

/// GML `collision_line(x, y, target.x, target.y, Wall, 0, 0) < 0` plus each
/// object's own `Alarm_1` fire condition, in GML short-circuit order so the
/// roll sequence matches.
fn gml_wants_fire(kind: EnemyKind, los: bool, dist: f32, rng: &mut rand::rngs::ThreadRng) -> bool {
    if !los {
        return false;
    }
    let mut roll = |n: f32| rng.random_range(0.0..n);
    match kind {
        // GML `PopoFreak/Alarm_1:7`.
        EnemyKind::PopoFreak => (64.0..160.0).contains(&dist) && roll(3.0) < 2.0,
        // GML `Raven/Alarm_1:5-6`.
        EnemyKind::Raven => dist > 64.0 && roll(6.0) < 1.0,
        // GML `Salamander/Alarm_1:5`, `Crab/Alarm_1:4`.
        EnemyKind::Salamander | EnemyKind::Crab => roll(2.0) < 1.0,
        // GML `Molefish/Alarm_1:6-7`.
        EnemyKind::Molefish => dist > 96.0 && roll(4.0) < 1.0,
        // GML `Molesarge/Alarm_1:6-7`.
        EnemyKind::Molesarge => dist < 120.0 && roll(3.0) < 1.0,
        // GML `FireBaller/Alarm_1:10`, `SuperFireBaller/Alarm_1:10`.
        EnemyKind::FireBaller | EnemyKind::SuperFireBaller => roll(3.0) < 1.0,
        // GML `Guardian/Alarm_1:13`, under `justfired == 0`.
        EnemyKind::CrownGuardian => (dist > 96.0 && roll(3.0) < 2.0) || roll(3.0) < 1.0,
        // GML `HostileHorror/Alarm_1:5-6`.
        EnemyKind::HostileHorror => dist > 48.0 && roll(2.0 + dist / 100.0) < 1.0,
        // GML `SuperFrog/Alarm_1:4` states no fire condition of its own.
        _ => true,
    }
}

fn gml_fire_rearm_secs(kind: EnemyKind, rng: &mut impl RngExt) -> f32 {
    match kind {
        EnemyKind::FireBaller | EnemyKind::SuperFireBaller => 8.0 / 30.0,
        EnemyKind::Molefish => (20.0 + rnd(rng, 5.0)) / 30.0,
        EnemyKind::Molesarge => (30.0 + rnd(rng, 5.0)) / 30.0,
        EnemyKind::CrownGuardian => 12.0 / 30.0,
        _ => {
            let def = enemy_def(kind);
            def.attack_cooldown + rng.random_range(0.0..def.attack_jitter / 30.0)
        }
    }
}

/// Wall-aware sight check (bevy parity: 8 px samples, 16 px tile-center
/// recheck, arena-exterior samples ignored).
pub fn line_of_sight_public(from: glam::Vec2, to: glam::Vec2, mask: &FloorMask) -> bool {
    has_line_of_sight(from, to, mask)
}

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

/// GML `Other_10` shape shared by `Bandit`, `Scorpion`, `GoldScorpion`,
/// `Sniper` and `JungleBandit`: the impulse is gated on `walk > 0` but the
/// cap is **unconditional** (`if speed > N speed = N` sits outside the
/// block) - gating it too let `separate` ratchet a stopped enemy up to the
/// 16 px/frame separation clamp.
#[inline]
fn walk_step(brain: &mut EnemyBrain, vel: &mut Velocity, impulse_f: f32, cap_f: f32, dt: f32) {
    if brain.walk > 0.0 {
        let heading = brain.heading;
        add_gml_motion(brain, vel, heading, impulse_f, dt);
        brain.walk = (brain.walk - dt * 30.0).max(0.0);
    }
    cap_gml_speed(brain, vel, cap_f);
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

/// GML per-object `Collision_Wall`. Only objects that do **not** override it
/// inherit `enemy/Collision_Wall`'s `busycollisions` branch: save direction
/// and speed, bounce, `motion_add(_dir, speed)`, restore `speed`, slide each
/// blocked axis by `friction`. Everything listed below overrides the event,
/// so the restore and friction slide are dead code for them.
fn wall_law(kind: EnemyKind) -> EnemyWallLaw {
    match kind {
        // `Maggot/Collision_Wall.gml:4-5`: slide, then push 1 px/frame off the
        // wall it just touched.
        EnemyKind::Maggot => EnemyWallLaw::SlidePush,
        // `move_bounce_solid(false)` - slide, no bounce.
        EnemyKind::Crab | EnemyKind::JungleFly => EnemyWallLaw::Slide,
        // `move_bounce_solid(true)` and nothing else.
        EnemyKind::Ballguy
        | EnemyKind::BigMaggot
        | EnemyKind::Scorpion
        | EnemyKind::GoldScorpion
        | EnemyKind::Sniper
        | EnemyKind::CrownGuardian
        | EnemyKind::DogGuardian
        | EnemyKind::FireBaller
        | EnemyKind::SuperFireBaller
        | EnemyKind::FrogQueen
        | EnemyKind::Guardian
        | EnemyKind::InvLaserCrystal
        | EnemyKind::LaserCrystal
        | EnemyKind::LightningCrystal
        | EnemyKind::InvSpider
        | EnemyKind::Spider
        | EnemyKind::Mimic
        | EnemyKind::SuperMimic
        | EnemyKind::WepMimic
        | EnemyKind::RadMaggot
        | EnemyKind::Ratking
        | EnemyKind::Salamander
        | EnemyKind::ScrapBossMissile
        | EnemyKind::SuperFrog
        | EnemyKind::Turtle
        | EnemyKind::Wolf => EnemyWallLaw::PlainBounce,
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
        EnemyKind::Freak | EnemyKind::BoneFish | EnemyKind::Rat => 4.0,
        // `BigRat` is the port's scaled `Rat`; `LilHunter/Other_10:7` caps at
        // 4 and `RadMaggot/Other_10:5` at 2.5. Without these three the
        // `separate` push could drive them to its 16 px/frame clamp.
        EnemyKind::BigRat => 4.0,
        EnemyKind::LilHunter | EnemyKind::LilHunterLoop => 4.0,
        EnemyKind::RadMaggot => 2.5,
        // `PopoFreak/Other_10:9` caps at 4.5, not the 4 its siblings use.
        EnemyKind::PopoFreak | EnemyKind::FastRat => 4.5,
        EnemyKind::Crab => 4.5,
        // `HostileHorror/Other_10:9` `if (speed > 4.5) speed = 4.5`.
        EnemyKind::HostileHorror => 4.5,
        // IDPD `Other_10`: `Grunt`/`EliteGrunt` 3, `Shielder` 3.5,
        // `Inspector` 3. The elites are separate kinds of their own and were
        // falling through to the `_` arm below.
        EnemyKind::IdpdGrunt | EnemyKind::IdpdElite | EnemyKind::IdpdInspector => 3.0,
        EnemyKind::IdpdShield => 3.5,
        // `EliteInspector/Other_10:7`, `EliteShielder/Other_10:8`.
        EnemyKind::EliteInspector | EnemyKind::EliteShielder => 3.5,
        // GML caps these conditionally on `spr_fire` / `maxspeed`. `Wolf` is at
        // its resting 3.5 here; the roll raises it to 5 via `Other_10:12-16`.
        EnemyKind::Wolf => 3.5,
        EnemyKind::Turtle => 5.0,
        // `Spider`/`InvSpider` cap at `maxspeed`; the chase cap is applied by
        // the spider's own alarm, so 5 is the outer bound here.
        EnemyKind::Spider | EnemyKind::InvSpider => 5.0,
        // `SnowBot/Other_10:14-16` caps at 8 while sliding; the resting 3 is
        // applied by its own motion-law arm above.
        EnemyKind::RobotGuard => 3.0,
        _ => f32::INFINITY,
    };
    frames * crate::SIM_HZ as f32
}

fn gml_speed_floor(kind: EnemyKind) -> f32 {
    match kind {
        // `LilHunter/Other_10:9`, `JungleFly/Other_10:8`
        EnemyKind::LilHunter | EnemyKind::LilHunterLoop | EnemyKind::JungleFly => 1.0,
        _ => 0.0,
    }
}

fn gml_clamp_speed(brain: &mut EnemyBrain, vel: &mut Velocity, kind: EnemyKind, cap_f: f32) {
    cap_gml_speed(brain, vel, cap_f);
    let floor_f = gml_speed_floor(kind);
    if floor_f > 0.0 {
        let floor = floor_f * crate::SIM_HZ as f32;
        if vel.0.length() < floor {
            set_gml_speed(brain, vel, floor_f);
        }
    }
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
        | EnemyKind::Necromancer
        // `LilHunter/Other_10:4` and `HostileHorror/Other_10:4` push at 0.8
        // as well; both used to fall to the 0.4 default arm below.
        | EnemyKind::LilHunter
        | EnemyKind::LilHunterLoop
        | EnemyKind::HostileHorror => (0.8, gml_speed_cap_frames(kind)),
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
        EnemyKind::Wolf => (1.0, 3.5),
        EnemyKind::ExploGuardian => (0.5, 2.5),
        _ => (0.4, gml_speed_cap_frames(kind)),
    }
}

/// GML objects whose `Other_10`/`Step_0` applies `motion_add` **every step**,
/// with no `walk` gate at all, as `(impulse, cap)` in px/frame. These drift
/// continuously and only change heading by bouncing, so they must not be
/// routed through [`gml_walk_law`], which is gated on `walk > 0`.
fn gml_constant_drift(kind: EnemyKind) -> Option<(f32, f32)> {
    Some(match kind {
        // `Guardian/Step_0:9-10`, `CrownGuardian/Step_0`.
        EnemyKind::Guardian | EnemyKind::CrownGuardian => (0.6, 0.6),
        // `LaserCrystal/Other_10:2-3`, `LightningCrystal`, `InvLaserCrystal`.
        EnemyKind::LaserCrystal => (0.5, 1.5),
        EnemyKind::LightningCrystal => (0.5, 1.8),
        EnemyKind::InvLaserCrystal => (0.5, 1.5),
        // `RadMaggot/Other_10:3` has NO `walk` gate, and its `Alarm_1` never
        // arms one (it only turns toward the target), so the walk law left it
        // standing still.
        EnemyKind::RadMaggot => (0.6, 2.5),
        EnemyKind::FireBaller => (0.6, 2.0),
        EnemyKind::SuperFireBaller => (0.6, 1.5),
        // `SuperFrog/Other_10:3,12`: 0.6 impulse every step then `speed = 2.5`.
        EnemyKind::SuperFrog => (0.6, 2.5),
        EnemyKind::SnowTank | EnemyKind::GoldSnowtank => (0.6, 1.5),
        EnemyKind::DogGuardian => (0.4, 2.0),
        _ => return None,
    })
}

/// The laser crystals gate their drift on `sprite_index != spr_fire` rather
/// than on the hurt sprite, so they stop drifting while charging.
fn gml_drift_stops_while_firing(kind: EnemyKind) -> bool {
    matches!(
        kind,
        EnemyKind::LaserCrystal | EnemyKind::LightningCrystal | EnemyKind::InvLaserCrystal
    )
}

/// GML objects that arm `walk` and never spend it, so the impulse runs every
/// step until the alarm re-arms: `Freak/Other_10:5-7`,
/// `ExploFreak/Other_10:6-8`, `RhinoFreak/Other_10:6-8` have no `walk -= 1`;
/// `ExploGuardian/Other_10` leans on `Alarm_1:3` re-arming `walk`.
/// `PopoFreak` is deliberately NOT here: `PopoFreak/Other_10:5` does
/// `walk -= 1` - on this list it drifted at its 4.5 px/frame cap forever.
fn gml_walk_never_decrements(kind: EnemyKind) -> bool {
    matches!(
        kind,
        EnemyKind::Freak | EnemyKind::ExploFreak | EnemyKind::RhinoFreak | EnemyKind::ExploGuardian
    )
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
    let axis = if horizontal {
        glam::Vec2::new(1.0, 0.0)
    } else {
        glam::Vec2::new(0.0, 1.0)
    };
    let start = pos + axis * (value / 30.0);
    let displacement = axis * (-sign * friction * 4095.0);
    let shapes = build_solid_shapes(start, displacement, radius, solids, Some(mask));
    let mut current = value;
    for _ in 0..4096 {
        let candidate = pos + axis * (current / 30.0);
        if contact_at(candidate, radius, &shapes).is_none() {
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
    entity: Entity,
    solids: &[(glam::Vec2, glam::Vec2)],
    mask: &FloorMask,
    positions: &[(Entity, glam::Vec2, i32, f32)],
    kind: EnemyKind,
    epos: glam::Vec2,
    radius: f32,
    dt: f32,
    separate_enemies: bool,
    loops: u32,
    frame: u64,
    law: EnemyWallLaw,
) {
    // `enemy/Create_0.gml` `friction = 0.4`
    apply_gml_friction(&mut vel.0, brain.friction, dt);
    let saved_direction = vel.0.normalize_or_zero();
    let saved_speed = vel.0.length();
    let parent = matches!(law, EnemyWallLaw::Parent) && loops <= 3;
    // GML's `move_bounce_solid(bounce)` argument is the *bounce* flag, not a
    // "silent" suppressor: `false` slides along the surface and kills the
    // normal component, which is what `Maggot/Collision_Wall.gml:4` and
    // `JungleFly/Collision_Wall.gml:4` ask for. Every other law passes `true`.
    let bounce = matches!(law, EnemyWallLaw::Parent | EnemyWallLaw::PlainBounce);
    let contact = move_bounce_solid(
        &mut pos.0,
        &mut vel.0,
        radius,
        dt,
        solids,
        Some(mask),
        bounce,
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
        separate(positions, entity, epos, &mut vel.0, kind, loops, frame);
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

/// GML `random(n) < 1`: a 1-in-n roll.
#[inline]
fn gml_chance(rng: &mut (impl RngExt + ?Sized), n: f32) -> bool {
    rng.random::<f32>() < 1.0 / n
}

/// GML `random(n) < k`: the literal form the source writes, e.g.
/// `random(4) < 3` (three in four) or `random(3) < 2`.
#[inline]
fn gml_roll(rng: &mut (impl RngExt + ?Sized), n: f32, k: f32) -> bool {
    rng.random_range(0.0..n) < k
}

/// GML `random_angle` (`random(360)`), radians.
#[inline]
fn gml_random_angle(rng: &mut (impl RngExt + ?Sized)) -> f32 {
    rng.random_range(0.0..std::f32::consts::TAU)
}

/// GML `choose(a, b, c)`.
#[inline]
fn gml_choose(rng: &mut (impl RngExt + ?Sized), options: &[f32]) -> f32 {
    options[rng.random_range(0..options.len())]
}

/// GML `point_direction(x1, y1, x2, y2)`, radians.
#[inline]
fn gml_point_direction(from: glam::Vec2, to: glam::Vec2) -> f32 {
    (to.y - from.y).atan2(to.x - from.x)
}

/// Per-object GML `Alarm_1` state. Each field is one GML instance variable;
/// the decide reads and writes them exactly as the `.gml` source does.
struct GmlDecide<'a> {
    kind: EnemyKind,
    /// `x, y`.
    pos: glam::Vec2,
    /// `target.x, target.y`; GML's `scrTarget()` picks the nearest Player.
    target: glam::Vec2,
    /// `collision_line(x, y, target.x, target.y, Wall, 0, 0) < 0`.
    sees: bool,
    /// `point_distance(x, y, target.x, target.y)`.
    dist: f32,
    /// `hp`, `max_hp` (the Raven and HostileHorror hurt checks).
    hp: f32,
    max_hp: f32,
    dt: f32,
    enemy: &'a mut Enemy,
    brain: &'a mut EnemyBrain,
    vel: &'a mut Velocity,
}

impl GmlDecide<'_> {
    /// GML `mcr_target_direction`: the bearing from self to target.
    #[inline]
    fn toward(&self) -> f32 {
        (self.target.y - self.pos.y).atan2(self.target.x - self.pos.x)
    }

    /// GML `point_direction(target.x, target.y, x, y)`: the bearing from
    /// TARGET to self. The reversed argument order is GML's own, and the
    /// `Rat`/`Raven`/`HostileHorror` walks rely on it.
    #[inline]
    fn away(&self) -> f32 {
        gml_point_direction(self.target, self.pos)
    }

    /// GML `alarm[1] = n` (frames; `random(k)` contributes 0..k-1).
    #[inline]
    fn arm(&mut self, frames: f32) {
        self.brain.attack = GTimer::from_seconds(frames / 30.0, TimerMode::Once);
    }

    /// The live `alarm[1]`, in frames.
    #[inline]
    fn armed(&self) -> f32 {
        self.brain.attack.duration() * 30.0
    }

    /// GML `alarm[1] += n`.
    #[inline]
    fn arm_add(&mut self, frames: f32) {
        self.arm(self.armed() + frames);
    }

    /// GML `alarm[1] /= n`.
    #[inline]
    fn arm_div(&mut self, n: f32) {
        self.arm(self.armed() / n);
    }

    /// GML `alarm[2] = n`.
    #[inline]
    fn arm2(&mut self, frames: f32) {
        self.brain.burst_timer = GTimer::from_seconds(frames / 30.0, TimerMode::Once);
    }

    #[inline]
    fn direction(&mut self, angle: f32) {
        set_gml_direction(self.brain, self.vel, angle);
    }

    /// GML `speed = n` (px/frame).
    #[inline]
    fn speed(&mut self, px_per_frame: f32) {
        set_gml_speed(self.brain, self.vel, px_per_frame);
    }

    /// GML `motion_add(angle, n)` (px/frame).
    #[inline]
    fn impulse(&mut self, angle: f32, px_per_frame: f32) {
        add_gml_motion(self.brain, self.vel, angle, px_per_frame, self.dt);
    }

    /// GML `gunangle = <deg>`.
    #[inline]
    fn aim(&mut self, degrees: f32) {
        self.brain.gunangle = degrees.to_radians();
    }

    /// GML `right = -1 / 1` off the target's side.
    #[inline]
    fn face_target_x(&mut self) {
        if self.target.x < self.pos.x {
            self.brain.right = -1.0;
        } else if self.target.x > self.pos.x {
            self.brain.right = 1.0;
        }
    }

    /// GML `if (hspeed > 0) right = 1 else if (hspeed < 0) right = -1`.
    #[inline]
    fn face_hspeed(&mut self) {
        self.brain.right = gml_right_from_hspeed(self.vel.0);
    }
}

/// GML `random(n)`, in degrees.
#[inline]
fn rnd<R: RngExt + ?Sized>(rng: &mut R, n: f32) -> f32 {
    rng.random_range(0.0..n)
}

/// GML `random(n) + half`, then negated: the `(random(80) - 40)` shape
/// without a second draw, so a +/- spread costs one RNG value like GML.
#[inline]
fn spread<R: RngExt + ?Sized>(rng: &mut R, n: f32) -> f32 {
    rnd(rng, n) - n * 0.5
}

/// GML `orandom(n)`: `±n/2` degrees.
#[inline]
fn ornd<R: RngExt + ?Sized>(rng: &mut R, n: f32) -> f32 {
    rng.random_range(-n * 0.5..n * 0.5)
}

/// GML `objects/<kind>/Alarm_2` for the objects whose decide only staged a
/// volley: spend `ammo`, re-arm `alarm[2]` on its own period, and - where
/// the object has an `else` arm - re-arm the DECIDE at the post-volley
/// value (crab, turret and salamander idle a different stretch after
/// firing than before it).
fn gml_alarm_2_volley(
    kind: EnemyKind,
    entity: Entity,
    commands: &mut Commands,
    cues: &mut Queue<AudioCue>,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    brain: &mut EnemyBrain,
    particles_on: bool,
    rng: &mut impl RngExt,
) {
    let gunangle = brain.gunangle;
    match kind {
        // GML `Crab/Alarm_2`: a 20-degree fork of `EnemyBullet2` at
        // 5..7 px/frame, one frame apart, then a 40..50 frame rest.
        EnemyKind::Crab => {
            if brain.ammo > 0 {
                brain.walk = 0.0;
                for side in [-20.0_f32, 20.0] {
                    let ang = gunangle + (side + ornd(rng, 3.0)).to_radians();
                    let e = spawn_enemy_projectile(
                        commands,
                        entity,
                        kind,
                        epos,
                        glam::Vec2::from_angle(ang)
                            * ((5.0 + rnd(rng, 2.0)) * crate::SIM_HZ as f32),
                        2,
                        3.0,
                        4.0,
                        120.0,
                        false,
                    );
                    commands.entity(e).insert((
                        ProjectileTyp(2),
                        ProjectileFade("images/sprScorpionBulletHit.png"),
                        ProjectileVisual {
                            sprite: "images/sprScorpionBullet.png",
                            mask: None,
                            fade: None,
                        },
                    ));
                }
                brain.ammo -= 1;
                brain.burst_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                enemy_cue(cues, "sndOasisCrabAttack");
            } else {
                brain.attack =
                    GTimer::from_seconds((40.0 + rnd(rng, 10.0)) / 30.0, TimerMode::Once);
            }
        }
        // GML `Raven/Alarm_2`: a single `EnemyBullet1` every 5 frames for
        // three rounds. No `else` arm, so the decide keeps its own value.
        EnemyKind::Raven => {
            if brain.ammo > 0 {
                brain.wkick = 5.0;
                let ang = gunangle + (rnd(rng, 16.0) - 8.0).to_radians();
                let e = spawn_enemy_projectile(
                    commands,
                    entity,
                    kind,
                    epos,
                    glam::Vec2::from_angle(ang) * (4.0 * crate::SIM_HZ as f32),
                    3,
                    3.0,
                    4.0,
                    120.0,
                    false,
                );
                commands.entity(e).insert((
                    ProjectileTyp(1),
                    ProjectileFade("images/sprEnemyBulletHit.png"),
                    ProjectileVisual {
                        sprite: "images/sprEnemyBullet1.png",
                        mask: None,
                        fade: None,
                    },
                ));
                brain.ammo -= 1;
                brain.burst_timer = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                enemy_cue(cues, "sndEnemyFire");
            }
        }
        // GML `Salamander/Alarm_2`: 45 `TrapFire` jets in a sweeping arc
        // (`sin(wave) * 70`), one per frame, then a 10..20 frame rest.
        EnemyKind::Salamander => {
            if brain.ammo > 0 {
                if brain.ammo == 45 {
                    enemy_cue(cues, "sndSalamanderFire");
                }
                brain.walk = 0.0;
                brain.ammo -= 1;
                brain.wave += 0.03;
                brain.burst_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                let ang = gunangle + (brain.wave.sin() * 70.0).to_radians();
                crate::environment::spawn_trap_fire_with_image(
                    commands,
                    epos + glam::Vec2::from_angle(ang) * 12.0,
                    glam::Vec2::from_angle(ang),
                    6.0 * crate::SIM_HZ as f32,
                    Team::Enemy,
                    Some(DamageSource::enemy(entity, kind)),
                    "images/sprTrapFire.png",
                    None,
                );
                if particles_on {
                    crate::environment::spawn_native_smoke_mote(
                        commands,
                        true,
                        epos,
                        glam::Vec2::ZERO,
                        0.0,
                    );
                }
            } else {
                enemy_cue(cues, "sndSalamanderEndFire");
                brain.attack =
                    GTimer::from_seconds((10.0 + rnd(rng, 10.0)) / 30.0, TimerMode::Once);
            }
        }
        // GML `PopoFreak/Alarm_2`: a wide and a tight `IDPDBullet` every
        // frame for eight rounds, re-aiming only on the first. No `else`.
        EnemyKind::PopoFreak => {
            if brain.ammo > 0 {
                if brain.ammo == 8 {
                    brain.gunangle = (player_pos.y - epos.y)
                        .atan2(player_pos.x - epos.x)
                        .to_degrees()
                        + rnd(rng, 90.0)
                        - 45.0;
                }
                brain.ammo -= 1;
                brain.wkick = 5.0;
                for spread_n in [100.0_f32, 40.0] {
                    let ang = brain.gunangle + spread(rng, spread_n).to_radians();
                    let e = spawn_enemy_projectile(
                        commands,
                        entity,
                        kind,
                        epos,
                        glam::Vec2::from_angle(ang)
                            * ((4.0 + rnd(rng, 3.0)) * crate::SIM_HZ as f32),
                        3,
                        3.0,
                        4.0,
                        120.0,
                        false,
                    );
                    commands.entity(e).insert((
                        ProjectileTyp(1),
                        ProjectileFade("images/sprIDPDBulletHit.png"),
                        ProjectileVisual {
                            sprite: "images/sprIDPDBullet.png",
                            mask: None,
                            fade: None,
                        },
                    ));
                }
                brain.burst_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                enemy_cue(cues, "sndGruntFire");
            }
        }
        // GML `HostileHorror/Alarm_2` is empty; its spray runs every step in
        // `Other_10:20-33` instead (see `hostile_horror_spray`).
        _ => {}
    }
}

/// GML `HostileHorror/Other_10:20-33`: the radial spray. Every step it
/// re-aims at the player and buys `round(charge + 1)` bullets out of its own
/// `raddrop`, growing `charge` by 0.1 - the pattern widens quadratically and
/// eventually runs the boss's 90 rads dry. `charge` resets to 0 when `ammo`
/// is 0.
/// `HorrorBullet` lives 300 frames (`Create_0:4` `alarm[1] = 300`), is
/// slash-destructible, knocks back 2, drops no rads. Its `damage` is the one
/// value the export lacks (the `folders/Objects/Projectiles` parent is not
/// shipped); 1 here, matching the one-pixel spray it draws.
fn hostile_horror_spray(
    commands: &mut Commands,
    entity: Entity,
    epos: glam::Vec2,
    player_pos: glam::Vec2,
    enemy: &mut Enemy,
    brain: &mut EnemyBrain,
    rng: &mut impl RngExt,
) {
    if brain.ammo == 0 {
        brain.charge = 0.0;
        return;
    }
    brain.gunangle = (player_pos.y - epos.y)
        .atan2(player_pos.x - epos.x)
        .to_degrees()
        + brain.gunoffset;
    let cost = (brain.charge + 1.0).round() as usize;
    if enemy.rad_drop >= cost {
        enemy.rad_drop -= cost;
        for _ in 0..cost {
            let reach = 2.0 + brain.charge;
            let at = epos
                + glam::Vec2::new(
                    rnd(rng, reach) * gml_choose(rng, &[1.0, -1.0]),
                    rnd(rng, reach) * gml_choose(rng, &[1.0, -1.0]),
                );
            let e = spawn_enemy_projectile(
                commands,
                entity,
                EnemyKind::HostileHorror,
                at,
                glam::Vec2::from_angle(brain.gunangle) * (12.0 * crate::SIM_HZ as f32),
                1,
                10.0,
                4.0,
                2.0,
                false,
            );
            commands.entity(e).insert((
                ProjectileTyp(2),
                ProjectileFade("images/sprHorrorHit.png"),
                ProjectileVisual {
                    sprite: "images/sprHorrorBullet.png",
                    mask: None,
                    fade: None,
                },
            ));
        }
        brain.charge += 0.1;
    }
    brain.ammo -= 1;
}

/// GML `objects/<kind>/Alarm_1` for the objects that reach the generic decide,
/// transcribed arm for arm from the `.gml` source; each arms `alarm[1]`
/// itself, so the idle cadence matches.
/// Volley-staging arms (`ammo` + `alarm[2]`) still draw their rolls and set
/// the registers, so the decide's cadence is unchanged; only the objects
/// whose `Alarm_2` the port models (the Wolf roll, below) draw the volley.
#[allow(clippy::too_many_arguments)]
fn gml_alarm_1_decide<R: RngExt + ?Sized>(
    kind: EnemyKind,
    pos: glam::Vec2,
    target: glam::Vec2,
    mask: &FloorMask,
    health: &Health,
    enemy: &mut Enemy,
    brain: &mut EnemyBrain,
    vel: &mut Velocity,
    dt: f32,
    rng: &mut R,
    entity: Entity,
    leaps: &mut HashMap<Entity, DogLeap>,
) {
    let to_target = target - pos;
    let mut d = GmlDecide {
        kind,
        pos,
        target,
        sees: has_line_of_sight(pos, target, mask),
        dist: to_target.length(),
        hp: health.hp as f32,
        max_hp: health.max as f32,
        dt,
        enemy,
        brain,
        vel,
    };

    match kind {
        // GML `Freak/Alarm_1`, `ExploFreak/Alarm_1` and
        // `RhinoFreak/Alarm_1`: a fixed short cadence, `walk = 20` on sight,
        // and no distance bands at all.
        EnemyKind::Freak => freak_alarm_1(&mut d, 80.0, rng),
        EnemyKind::ExploFreak | EnemyKind::RhinoFreak => freak_alarm_1(&mut d, 180.0, rng),
        EnemyKind::PopoFreak => popo_freak_alarm_1(&mut d, rng),
        EnemyKind::Rat | EnemyKind::BigRat | EnemyKind::FastRat => rat_alarm_1(&mut d, rng),
        EnemyKind::Wolf => wolf_alarm_1(&mut d, rng),
        EnemyKind::Raven => raven_alarm_1(&mut d, rng),
        EnemyKind::Spider | EnemyKind::InvSpider => spider_alarm_1(&mut d, rng),
        EnemyKind::Crab => crab_alarm_1(&mut d, rng),
        EnemyKind::Turtle => turtle_alarm_1(&mut d, rng),
        EnemyKind::Salamander => salamander_alarm_1(&mut d, rng),
        EnemyKind::BoneFish => bone_fish_alarm_1(&mut d, rng),
        EnemyKind::RobotGuard => snow_bot_alarm_1(&mut d, rng),
        EnemyKind::HostileHorror => hostile_horror_alarm_1(&mut d, rng),
        // GML `RadMaggot/Alarm_1` and `SuperFrog/Alarm_1` are the Maggot's
        // decide: turn toward the target, or scatter.
        EnemyKind::RadMaggot | EnemyKind::SuperFrog => maggot_alarm_1(&mut d, rng, 30.0, 20.0),
        EnemyKind::Molefish => molefish_alarm_1(&mut d, rng),
        EnemyKind::Molesarge => molesarge_alarm_1(&mut d, rng),
        EnemyKind::FireBaller => fireballer_alarm_1(&mut d, rng),
        EnemyKind::SuperFireBaller => super_fireballer_alarm_1(&mut d, rng),
        EnemyKind::CrownGuardian => guardian_alarm_1(&mut d, rng),
        EnemyKind::DogGuardian => {
            let leap = leaps.entry(entity).or_default();
            dog_guardian_alarm_1(&mut d, leap, rng);
        }
        // GML `objects/Crystal` has no `Alarm_1` at all, so nothing here is
        // reachable for it; an unmodelled object parks its alarm instead of
        // spinning on the default.
        _ => gml_alarm_1_fallback(&mut d, rng),
    }
}

/// Every kind above re-arms inside its own transcribe; this arm only exists
/// so a kind added to the generic path without a transcription does not spin
/// on `brain.attack`'s initial duration forever.
fn gml_alarm_1_fallback(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(30.0 + rnd(rng, 20.0));
}

/// GML `Freak/Alarm_1:1-9` (impulse `random(80) - 40`) with
/// `ExploFreak/Alarm_1:1-9` / `RhinoFreak/Alarm_1:1-9`
/// (`random(180) - 90`) - identical but for that spread.
fn freak_alarm_1(d: &mut GmlDecide<'_>, spread_n: f32, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(6.0 + rnd(rng, 5.0));
    if d.sees {
        d.brain.walk = 20.0;
        d.impulse(d.toward() + spread(rng, spread_n).to_radians(), 1.5);
    } else {
        d.impulse(gml_random_angle(rng), 0.5);
    }
}

/// GML `PopoFreak/Alarm_1`: `walk = 20` on sight, and the burst arm trades it
/// for `walk = 6` plus 30 extra frames on the decide.
fn popo_freak_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(15.0 + rnd(rng, 5.0));
    if d.sees {
        d.brain.walk = 20.0;
        d.impulse(d.toward() + (rnd(rng, 90.0) - 45.0).to_radians(), 5.0);
        if d.dist < 160.0 && d.dist > 64.0 && gml_chance(rng, 3.0) {
            d.aim(d.toward().to_degrees() + rnd(rng, 90.0) - 45.0);
            d.arm_add(30.0);
            d.arm2(15.0);
            d.brain.walk = 6.0;
            d.brain.ammo = 8;
        }
    } else {
        if gml_chance(rng, 4.0) {
            d.brain.walk = 20.0;
        }
        d.impulse(gml_random_angle(rng), 3.0);
    }
    d.brain.walkdir = d.brain.heading;
}

/// GML `Rat/Alarm_1` (and `FastRat/Alarm_1`, which differs only in
/// `random(20) - 10` for `orandom(10)`): `walk = 40 + random(10)` on ANY
/// line of sight and `alarm[1] = walk`, so the rat's cadence IS its walk.
fn rat_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(10.0 + rnd(rng, 30.0));
    if d.sees {
        let jitter = if d.kind == EnemyKind::FastRat {
            rnd(rng, 20.0) - 10.0
        } else {
            ornd(rng, 10.0)
        };
        d.direction(d.toward() + jitter.to_radians());
        d.speed(0.4);
        d.brain.walk = 40.0 + rnd(rng, 10.0);
        d.arm(d.brain.walk);
    } else if gml_chance(rng, 5.0) {
        d.impulse(gml_random_angle(rng), 0.4);
        d.brain.walk = 10.0 + rnd(rng, 15.0);
        d.arm(d.brain.walk + 10.0 + rnd(rng, 30.0));
    }
}

/// GML `Wolf/Alarm_1:2-3`: `alarm[1] = 30 + random(20); walk = alarm[1]` on
/// every path, so the wolf's walk length IS its decide period.
fn wolf_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(30.0 + rnd(rng, 20.0));
    d.brain.walk = d.armed();
    if d.sees {
        if gml_chance(rng, 2.0) {
            d.direction(d.toward());
            d.arm2(10.0);
            d.arm(30.0);
        } else {
            d.direction(d.toward() + ornd(rng, 15.0).to_radians());
            if gml_chance(rng, 4.0) {
                d.speed(0.0);
                d.brain.walk = 0.0;
            }
        }
    } else {
        if gml_chance(rng, 3.0) {
            d.speed(0.0);
            d.brain.walk = 0.0;
        }
        d.impulse(gml_random_angle(rng), 2.0);
        d.impulse(d.toward(), 1.5);
    }
}

/// GML `Raven/Alarm_1`: the close branch turns the bird AWAY from the target
/// (`point_direction(target.x, target.y, x, y)`), the far branch walks.
fn raven_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(20.0 + rnd(rng, 10.0));
    if d.sees {
        if d.dist > 64.0 {
            if gml_chance(rng, 6.0) {
                d.arm2(1.0);
                d.brain.ammo = 3;
                d.aim(d.toward().to_degrees());
                d.arm(20.0 + rnd(rng, 5.0));
            } else {
                if gml_chance(rng, 4.0) {
                    d.direction(d.toward() + (rnd(rng, 90.0) - 45.0).to_radians());
                }
                d.speed(0.4);
                d.brain.walk = 20.0 + rnd(rng, 10.0);
                d.aim(d.toward().to_degrees());
            }
        } else {
            d.direction(d.away() + (rnd(rng, 20.0) - 10.0).to_radians());
            d.speed(0.4);
            d.brain.walk = 40.0 + rnd(rng, 10.0);
            d.aim(d.toward().to_degrees());
        }
        d.face_target_x();
    } else if gml_chance(rng, 3.0) {
        d.impulse(gml_random_angle(rng), 0.4);
        d.brain.walk = 20.0 + rnd(rng, 10.0);
        d.arm(d.brain.walk + 10.0 + rnd(rng, 30.0));
        d.aim(d.brain.heading.to_degrees());
        d.face_hspeed();
    } else if (d.hp < d.max_hp || gml_chance(rng, 50.0)) && gml_chance(rng, 2.0) {
        // GML `scrRavenLift()` on self and the furthest raven; the port has
        // no lift for the raven, so only the roll is spent.
    }
}

/// GML `Spider/Alarm_1` and `InvSpider/Alarm_1` (identical): the chase needs
/// sight AND `point_distance < 96`, and it is the only path that raises
/// `maxspeed` from 3 to 5.
fn spider_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(20.0 + rnd(rng, 10.0));
    d.brain.maxspeed = 3.0;
    if d.sees {
        if d.dist < 96.0 {
            d.brain.maxspeed = 5.0;
            d.direction(d.toward() + (rnd(rng, 80.0) - 40.0).to_radians());
            d.speed(0.4);
            d.brain.walk = 15.0 + rnd(rng, 5.0);
            d.arm(d.brain.walk + 5.0);
        } else if gml_chance(rng, 2.0) {
            d.impulse(gml_random_angle(rng), 0.4);
            d.brain.walk = 10.0 + rnd(rng, 10.0);
            d.arm(d.brain.walk + 10.0 + rnd(rng, 10.0));
        }
    } else if gml_chance(rng, 4.0) {
        d.impulse(gml_random_angle(rng), 0.4);
        d.brain.walk = 10.0 + rnd(rng, 10.0);
        d.arm(d.brain.walk + 10.0 + rnd(rng, 10.0));
    }
}

/// GML `Crab/Alarm_1`: the charge arm re-arms on a SHORTER period than the
/// default, and the `walk = 50` arm on a longer one.
fn crab_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(10.0 + rnd(rng, 10.0));
    if d.sees {
        if gml_chance(rng, 2.0) {
            d.brain.ammo = 8;
            d.arm2(2.0);
            d.aim(d.toward().to_degrees() + rnd(rng, 40.0) - 20.0);
            d.arm(10.0 + rnd(rng, 5.0));
        } else if d.dist < 120.0 && gml_roll(rng, 4.0, 3.0) {
            d.brain.walk = 50.0;
            d.aim(d.toward().to_degrees() + rnd(rng, 20.0) - 10.0);
            d.arm(50.0 + rnd(rng, 10.0));
        } else {
            d.direction(d.toward() + (rnd(rng, 160.0) - 80.0).to_radians());
            d.aim(d.brain.heading.to_degrees());
            d.speed(0.4);
            d.brain.walk = 10.0 + rnd(rng, 10.0);
        }
        d.face_target_x();
    } else if gml_chance(rng, 10.0) {
        d.impulse(gml_random_angle(rng), 0.4);
        d.aim(d.brain.heading.to_degrees());
        d.brain.walk = 20.0 + rnd(rng, 10.0);
        d.arm(d.brain.walk + 10.0 + rnd(rng, 30.0));
        d.face_hspeed();
    }
}

/// GML `Turtle/Alarm_1`: `walk = 0` first, then either `walk = 50` or
/// `alarm[1] /= 2` - the turtle's long walk is bought with a doubled
/// cadence, not a longer one. The else arm covers BOTH "no sight" and
/// "too far", so a blocked turtle still turns and still rolls.
fn turtle_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(50.0 + rnd(rng, 10.0));
    d.brain.walk = 0.0;
    if d.sees && d.dist < 320.0 {
        d.direction(d.toward() + (rnd(rng, 60.0) - 30.0).to_radians());
        if gml_roll(rng, 3.0, 2.0) {
            d.brain.walk = 50.0;
        } else {
            d.arm_div(2.0);
        }
    } else {
        d.direction(gml_random_angle(rng));
        if gml_chance(rng, 4.0) {
            d.brain.walk = 50.0;
        } else {
            d.arm_div(2.0);
        }
    }
}

/// GML `Salamander/Alarm_1`: `walk = 0` up front, and the walk arm ADDS 40
/// frames to the decide rather than replacing it. The first arm is
/// `!collision_line && random(2) < 1`, so a blocked salamander falls
/// through to the same two arms a sighted one does.
fn salamander_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(10.0 + rnd(rng, 10.0));
    d.brain.walk = 0.0;
    if d.sees && gml_chance(rng, 2.0) {
        d.brain.ammo = 45;
        d.arm2(5.0);
        d.aim(d.toward().to_degrees() + rnd(rng, 40.0) - 20.0);
    } else if gml_chance(rng, 20.0) {
        d.brain.ammo = 45;
        d.arm2(5.0);
        d.aim(gml_random_angle(rng).to_degrees());
    } else {
        d.direction(d.toward() + (rnd(rng, 100.0) - 50.0).to_radians());
        d.speed(0.4);
        d.brain.walk = 40.0 + rnd(rng, 10.0);
        d.arm_add(40.0);
    }
    d.face_target_x();
}

/// GML `BoneFish/Alarm_1`: `alarm[1] = 5` (a very fast decide) but every
/// sight re-arms it to the full `walk`.
fn bone_fish_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(5.0);
    if d.sees {
        d.direction(d.toward() + (rnd(rng, 20.0) - 10.0).to_radians());
        d.speed(0.4);
        d.brain.walk = 40.0 + rnd(rng, 10.0);
        d.arm(d.brain.walk);
    } else if gml_chance(rng, 5.0) {
        d.impulse(gml_random_angle(rng), 0.4);
        d.brain.walk = 10.0 + rnd(rng, 15.0);
        d.arm(d.brain.walk + 10.0 + rnd(rng, 30.0));
    }
}

/// GML `SnowBot/Alarm_1`: a FIXED 40-frame decide (no `random` on the
/// re-arm), and the "do nothing" arm still sets `walk = 30`. `meleedamage` is
/// both what `Other_10:8` tests for the charge strip and what raises the
/// sled's speed cap from 3 to 8.
fn snow_bot_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(40.0);
    d.enemy.touch_damage = 0;
    if d.sees {
        if d.dist < 120.0 && gml_chance(rng, 4.0) {
            d.brain.walk = 40.0;
            d.arm(40.0);
            d.aim(d.toward().to_degrees() + rnd(rng, 30.0) - 15.0);
            d.enemy.touch_damage = 4;
        } else if gml_chance(rng, 5.0) {
            d.aim(gml_random_angle(rng).to_degrees());
            d.brain.walk = 30.0;
            d.face_hspeed();
        }
        d.face_target_x();
    } else if gml_chance(rng, 5.0) {
        d.aim(gml_random_angle(rng).to_degrees());
        d.brain.walk = 30.0;
        d.face_hspeed();
    }
}

/// GML `HostileHorror/Alarm_1`: the guard is `!collision_line ||
/// random(6) < 1`, so a blocked horror acts anyway a sixth of the time.
/// The spray arm's chance scales with distance (`random(2 +
/// point_distance / 100)`, the argument floored), and the walk arm picks a
/// side with `choose(1, -1)`.
fn hostile_horror_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(10.0 + rnd(rng, 10.0));
    if !d.sees && !gml_chance(rng, 6.0) {
        // GML's `else if random(4) < 1` arm.
        if gml_chance(rng, 4.0) {
            d.impulse(gml_random_angle(rng), 0.4);
            d.brain.walk = 20.0 + rnd(rng, 10.0);
            d.face_hspeed();
        }
        return;
    }
    if d.dist > 48.0 || (gml_chance(rng, 8.0) && enemy_def(d.kind).rad_drop > 0) {
        if gml_chance(rng, 2.0 + (d.dist / 100.0).floor()) {
            d.brain.ammo = 30;
            d.aim(d.toward().to_degrees());
            d.brain.gunoffset = rnd(rng, 20.0) - 10.0;
        } else if gml_roll(rng, 4.0, 3.0) {
            let side = gml_choose(rng, &[1.0, -1.0]);
            d.direction(d.toward() + ((40.0 + rnd(rng, 60.0)) * side).to_radians());
            d.speed(0.4);
            d.brain.walk = 20.0 + rnd(rng, 10.0);
            d.aim(d.toward().to_degrees());
        }
    } else {
        d.direction(d.toward());
        d.speed(0.4);
        d.brain.walk = 20.0 + rnd(rng, 10.0);
        d.aim(d.toward().to_degrees());
    }
    d.face_target_x();
}

/// GML `Maggot/Alarm_1` / `RadMaggot/Alarm_1` / `SuperFrog/Alarm_1`: turn on
/// sight, otherwise scatter. The maggot's own ticker uses the same shape.
fn maggot_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized), base: f32, spread: f32) {
    d.arm(base + rnd(rng, spread));
    if d.sees {
        d.direction(d.toward() + (rnd(rng, 20.0) - 10.0).to_radians());
    } else {
        d.impulse(gml_random_angle(rng), 0.5);
    }
}

fn molefish_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(15.0 + rnd(rng, 10.0));
    if d.sees {
        if d.dist > 96.0 {
            if gml_chance(rng, 4.0) {
                d.aim(d.toward().to_degrees());
                d.brain.wkick = 4.0;
                d.arm(20.0 + rnd(rng, 5.0));
            } else {
                d.direction(
                    d.toward()
                        + ((45.0 + rnd(rng, 90.0)) * gml_choose(rng, &[1.0, -1.0])).to_radians(),
                );
                d.speed(0.4);
                d.brain.walk = 10.0 + rnd(rng, 10.0);
                d.aim(d.toward().to_degrees());
            }
        } else {
            d.direction(d.away() + (rnd(rng, 20.0) - 10.0).to_radians());
            d.speed(0.4);
            d.brain.walk = 20.0 + rnd(rng, 10.0);
            d.aim(d.toward().to_degrees());
        }
        d.face_target_x();
    } else if gml_chance(rng, 4.0) {
        d.impulse(gml_random_angle(rng), 0.4);
        d.brain.walk = 20.0 + rnd(rng, 10.0);
        d.arm(d.brain.walk + 5.0);
        d.aim(d.brain.heading.to_degrees());
        d.face_hspeed();
    }
}

fn molesarge_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(15.0 + rnd(rng, 10.0));
    if d.sees {
        if d.dist < 120.0 {
            if gml_chance(rng, 3.0) {
                d.aim(d.toward().to_degrees() + (rnd(rng, 20.0) - 10.0));
                d.brain.wkick = 8.0;
                d.impulse(d.brain.gunangle + std::f32::consts::PI, 3.0);
                d.arm(30.0 + rnd(rng, 5.0));
            } else {
                d.direction(
                    d.toward()
                        + ((45.0 + rnd(rng, 90.0)) * gml_choose(rng, &[1.0, -1.0])).to_radians(),
                );
                d.speed(0.4);
                d.brain.walk = 10.0 + rnd(rng, 10.0);
                d.aim(d.toward().to_degrees());
            }
        } else {
            d.direction(d.toward() + (rnd(rng, 20.0) - 10.0).to_radians());
            d.speed(0.4);
            d.brain.walk = 30.0 + rnd(rng, 10.0);
            d.aim(d.toward().to_degrees());
        }
        d.face_target_x();
    } else if gml_chance(rng, 4.0) {
        d.impulse(gml_random_angle(rng), 0.4);
        d.brain.walk = 20.0 + rnd(rng, 10.0);
        d.arm(d.brain.walk + 5.0);
        d.aim(d.brain.heading.to_degrees());
        d.face_hspeed();
    }
}

fn fireballer_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(20.0 + rnd(rng, 10.0));
    if d.sees {
        if gml_chance(rng, 3.0) {
            d.direction(d.toward() + std::f32::consts::PI);
            d.arm(8.0);
        } else {
            d.direction(d.toward() + (rnd(rng, 20.0) - 10.0).to_radians());
        }
    } else {
        d.impulse(gml_random_angle(rng), 1.0);
        d.brain.walk = 0.0;
    }
}

fn super_fireballer_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    d.arm(20.0 + rnd(rng, 10.0));
    if d.sees {
        if rnd(rng, 5.0) >= 4.0 {
            d.direction(d.toward() + ornd(rng, 10.0).to_radians());
            d.brain.walk = 0.0;
            return;
        }
        d.direction(d.toward() + std::f32::consts::PI);
        d.brain.walk = 0.0;
        d.arm(8.0);
    } else {
        d.impulse(gml_random_angle(rng), 1.0);
        d.brain.walk = 0.0;
    }
}

fn guardian_alarm_1(d: &mut GmlDecide<'_>, rng: &mut (impl RngExt + ?Sized)) {
    let justfired = d.armed() == 12.0;
    d.arm(10.0 + rnd(rng, 40.0));
    if justfired {
        d.arm_add(10.0);
    }
    if d.sees {
        d.direction(d.toward() + (rnd(rng, 180.0) - 90.0).to_radians());
        if ((d.dist > 96.0 && gml_roll(rng, 3.0, 2.0)) || gml_roll(rng, 3.0, 1.0)) && !justfired {
            d.aim(d.toward().to_degrees());
            d.arm(12.0);
        } else if gml_chance(rng, 2.0) {
            d.arm_add(60.0);
        }
    }
}

/// GML `DogGuardian`'s leap registers (`leap`, `z`, `zspeed`, `jumpdir`,
/// `jumpdist`, `bounced`), held per instance because `EnemyBrain` has no
/// slot for them.
#[derive(Clone, Copy)]
pub struct DogLeap {
    /// `Alarm_1:15` `jumpdist = point_distance(...) * 1.1`, spent by `Alarm_2`.
    dist: f32,
    /// `Alarm_1:16` `jumpdir = mcr_target_direction + orandom(20)`.
    dir: f32,
    /// `Alarm_2:2` `leap = jumpdist / 8`: frames of flight left, 0 grounded.
    leap: f32,
    /// `Alarm_2:3` `zspeed = leap / 4`, stepped by `Other_10:25-26`.
    z: f32,
    zspeed: f32,
    /// `Alarm_3` (`alarm[3] = 16` from `Alarm_2:4`): frames until landing.
    land: f32,
    /// `Collision_Wall:5` `bounced = 1`: a wall hit ends the `jumpdir` steer.
    bounced: bool,
    /// `Other_10:7` drifts only while `spr_idle == sprDogGuardianWalk`
    /// (`Alarm_1:4` sets it; the staging arm and `Alarm_3:2` clear it).
    walking: bool,
    pose: DogGuardianPose,
}

impl Default for DogLeap {
    fn default() -> Self {
        Self {
            dist: 0.0,
            dir: 0.0,
            leap: 0.0,
            z: 0.0,
            zspeed: 0.0,
            land: 0.0,
            bounced: false,
            walking: true,
            pose: DogGuardianPose::Ground,
        }
    }
}

fn dog_leap_airborne(brain: &mut EnemyBrain, vel: &mut Velocity, leap: &mut DogLeap, dt: f32) {
    if !leap.bounced {
        set_gml_direction(brain, vel, leap.dir);
    }
    set_gml_speed(brain, vel, 8.0);
    let frames = dt * 30.0;
    leap.leap -= frames;
    if leap.leap < 0.0 {
        leap.leap = 0.0;
    }
    leap.z += leap.zspeed * frames;
    leap.zspeed -= 0.5 * frames;
}

fn dog_guardian_alarm_1(
    d: &mut GmlDecide<'_>,
    leap: &mut DogLeap,
    rng: &mut (impl RngExt + ?Sized),
) {
    d.arm(30.0 + rnd(rng, 20.0));
    leap.walking = true;
    leap.pose = DogGuardianPose::Ground;
    d.impulse(gml_random_angle(rng), 0.5);
    if d.sees {
        if d.dist < 100.0 + rnd(rng, 60.0) && gml_roll(rng, 3.0, 2.0) {
            leap.bounced = false;
            leap.walking = false;
            leap.pose = DogGuardianPose::Charge;
            d.speed(0.0);
            leap.dist = d.dist * 1.1;
            leap.dir = d.toward() + ornd(rng, 20.0).to_radians();
            d.arm2(10.0);
            d.arm(300.0);
        } else {
            d.impulse(d.toward(), 1.4);
        }
    }
}

/// GML `SnowBot`'s `sprite_index == spr_fire`, the key `Other_10:11-13`
/// picks between the 8 and the 3 px/frame cap. `Alarm_1:13-14` raises it
/// on the charge arm and `Alarm_1:3-4` on every other arm; a hit overwrites
/// it with `spr_hurt`, and `enemy/Step_0:27-29` then drops to `spr_idle`/
/// `spr_walk`, never back to `spr_fire`.
#[derive(Default, Clone, Copy)]
pub struct SnowBotFire {
    armed: bool,
    hurting: bool,
}

/// table-driven fire, per bevy `enemy_ai` top to bottom (bosses `continue`
/// before their first timer tick - boss brains live elsewhere).
#[allow(clippy::too_many_arguments)]
pub fn enemy_ai(
    time: Res<SimTime>,
    mut commands: Commands,
    mut trauma: ResMut<Trauma>,
    euphoria: Res<Euphoria>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    save: Res<crate::savedata_part::SaveData>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut charge_state: Local<HashMap<Entity, GTimer>>,
    mut snowbot_fire: Local<HashMap<Entity, SnowBotFire>>,
    mut dog_leaps: Local<HashMap<Entity, DogLeap>>,
    player_q: Query<(&Pos, &Player), (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &mut Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            &Health,
            Option<&BossBrain>,
            Option<&mut SpriteAnim>,
            Option<&HurtAnim>,
            Option<&mut DogGuardianLeap>,
            Option<&mut NextHurt>,
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
    // GML `Smoke/Create_0.gml:15` self-gates on `UberCont.opt_prtcls`.
    let particles_on = save.settings.particles;

    // Pre-move snapshot for separation (bevy parity: pushes use the
    // snapshot, applied to the live position).
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, pos, _, _, _, _, _, _)| {
            (
                e,
                pos.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
    let current_frame = (time.elapsed_secs / dt as f64) as u64;
    snowbot_fire.retain(|e, _| positions.iter().any(|(pe, ..)| pe == e));
    dog_leaps.retain(|e, _| positions.iter().any(|(pe, ..)| pe == e));

    for (
        entity,
        mut enemy,
        mut brain,
        mut vel,
        mut pos,
        health,
        boss,
        mut anim,
        hurt,
        mut dog_pose,
        mut next_hurt,
    ) in &mut enemies
    {
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
                | EnemyKind::MeleeFake
                | EnemyKind::Ratking
                | EnemyKind::Turret
                | EnemyKind::IceFlower
        ) {
            continue;
        }

        // GML `objects/Crystal` has no `Alarm_1` and no `Other_10`, so it is
        // a stationary hazard: it must not run a decide at all.
        if enemy.kind == EnemyKind::Crystal {
            vel.0 = glam::Vec2::ZERO;
            continue;
        }

        // Only `Turret/Other_10` (`speed = 0; x = xprevious`) is a true
        // emplacement. The laser crystals drift, so pinning them froze them.
        let emplacement = enemy.kind == EnemyKind::Turret;

        // Kinds with a dedicated verbatim ticker own their `Other_10` and alarm
        // cadence; running the generic timers/walk/decide too summed both impulses
        // (IDPD 0.8 + 0.4, Ratking 0.5 + 0.5), counted `walk` down twice a step and
        // fired every alarm twice. Their tickers apply the cap themselves.
        let owns_motion = has_dedicated_tick(enemy.kind);
        // `DogGuardian/Other_10:18-24` puts `speed = 8` outside both caps
        // while the leap is up.
        let mut dog_airborne = false;
        let solids = prop_shapes(&props);

        if !owns_motion {
            brain.melee.tick(dt);
            // GML steps every alarm on every instance each step, so each ticks
            // exactly once per frame here; every block below only reads
            // `just_finished()`.
            brain.attack.tick(dt);
            brain.fire_alarm.tick(dt);
            brain.burst_timer.tick(dt);
            if enemy.kind == EnemyKind::DogGuardian && brain.burst_timer.just_finished() {
                let leap = dog_leaps.entry(entity).or_default();
                leap.leap = leap.dist / 8.0;
                leap.zspeed = leap.leap / 4.0;
                leap.land = 16.0;
                leap.pose = DogGuardianPose::Airborne;
                enemy_cue(&mut cues, "sndDogGuardianJump");
            }

            // GML `HostileHorror/Other_10:20-33`: the spray runs every
            // step, not off an alarm, so it is not in `gml_alarm_2_volley`.
            if enemy.kind == EnemyKind::HostileHorror {
                hostile_horror_spray(
                    &mut commands,
                    entity,
                    epos,
                    player_pos,
                    &mut enemy,
                    &mut brain,
                    &mut rng,
                );
            }

            // Objects whose `Other_10`/`Step_0` applies `motion_add` every step with
            // no `walk` gate: they drift continuously and only change heading by
            // bouncing. `Guardian/Step_0:9-10` caps at 0.6, so the `walk` law
            // pinned them whenever `walk` was 0.
            let heading = brain.heading;
            if let Some((impulse_f, cap_f)) = gml_constant_drift(enemy.kind) {
                // `sprite_index != spr_hurt`, except the laser crystals which gate on
                // `sprite_index != spr_fire`. `DogGuardian` takes
                // `Other_10:18-35` while `leap` is up and gates the drift on
                // `spr_idle == sprDogGuardianWalk` (`Other_10:7`).
                if enemy.kind == EnemyKind::DogGuardian {
                    let leap = dog_leaps.entry(entity).or_default();
                    if leap.land > 0.0 {
                        leap.land -= dt * 30.0;
                        if leap.land <= 0.0 {
                            leap.land = 0.0;
                            leap.leap = 0.0;
                            leap.z = 0.0;
                            leap.zspeed = 0.0;
                            leap.walking = false;
                            leap.pose = DogGuardianPose::Land;
                            set_gml_speed(&mut brain, &mut vel, 0.0);
                            brain.attack = GTimer::from_seconds(8.0 / 30.0, TimerMode::Once);
                            trauma.add(0.1);
                            enemy_cue(&mut cues, "sndDogGuardianLand");
                            commands.spawn((
                                GameCleanup,
                                LevelCleanup,
                                PortalClear {
                                    timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                                    scale: 1.0,
                                },
                                Pos(epos),
                            ));
                            crate::environment::spawn_dust_ring_12(
                                &mut commands,
                                particles_on,
                                epos + glam::Vec2::new(0.0, 10.0),
                            );
                        }
                    }
                    if leap.leap > 0.0 {
                        dog_airborne = true;
                        dog_leap_airborne(&mut brain, &mut vel, leap, dt);
                        if let Some(nh) = next_hurt.as_deref_mut() {
                            nh.0 = 0;
                        }
                    } else {
                        // `Other_10:4-5` `z = 0; zspeed = 0` on the ground.
                        leap.z = 0.0;
                        leap.zspeed = 0.0;
                        if leap.land > 0.0 {
                            leap.pose = DogGuardianPose::Ground;
                        }
                    }
                }
                let charging = charge_state.contains_key(&entity);
                let walking = enemy.kind != EnemyKind::DogGuardian
                    || dog_leaps.get(&entity).map_or(true, |s| s.walking);
                if walking
                    && !dog_airborne
                    && hurt.is_none()
                    && brain.dash <= 0.0
                    && !(charging && gml_drift_stops_while_firing(enemy.kind))
                {
                    add_gml_motion(&mut brain, &mut vel, heading, impulse_f, dt);
                }
                if !dog_airborne {
                    gml_clamp_speed(&mut brain, &mut vel, enemy.kind, cap_f);
                }
            } else if enemy.kind == EnemyKind::RobotGuard {
                // GML `SnowBot/Other_10:3-6`: the impulse runs along `gunangle`, not
                // `direction`, so the sled steers off its own heading.
                // `Other_10:11-13` caps at 8 while `sprite_index == spr_fire`
                // and 3 otherwise: `Alarm_1:13-14` raises the sprite WITH
                // `meleedamage` on the charge arm, but a hit swaps in `spr_hurt`
                // and `enemy/Step_0:27-29` drops to `spr_idle`/`spr_walk`, never
                // back to `spr_fire` - `touch_damage` alone outlives the sprite
                // and the decide re-arms it.
                if brain.walk > 0.0 {
                    let gun = brain.gunangle;
                    add_gml_motion(&mut brain, &mut vel, gun, 1.0, dt);
                    brain.walk -= dt * 30.0;
                    if brain.walk < 0.0 {
                        brain.walk = 0.0;
                    }
                }
                let fire = snowbot_fire.entry(entity).or_default();
                if hurt.is_some() && !fire.hurting {
                    fire.armed = false;
                }
                fire.hurting = hurt.is_some();
                let cap = if fire.armed { 8.0 } else { 3.0 };
                gml_clamp_speed(&mut brain, &mut vel, enemy.kind, cap);
            } else if brain.walk > 0.0 {
                let (impulse_f, cap_f) = gml_walk_law(enemy.kind);
                // GML `PopoFreak/Other_10:6-7` swaps the push to `walkdir` at 1
                // while the hurt sprite is up; `Freak/Other_10:8`,
                // `ExploFreak/Other_10:8` and `RhinoFreak/Other_10:8` gate the
                // whole `motion_add` on `sprite_index != spr_hurt`.
                let hurt_gated = matches!(
                    enemy.kind,
                    EnemyKind::Freak | EnemyKind::ExploFreak | EnemyKind::RhinoFreak
                );
                if enemy.kind == EnemyKind::PopoFreak && hurt.is_some() {
                    let impulse_angle = brain.walkdir;
                    add_gml_motion(&mut brain, &mut vel, impulse_angle, 1.0, dt);
                } else if !hurt_gated || hurt.is_none() {
                    add_gml_motion(&mut brain, &mut vel, heading, impulse_f, dt);
                }
                // The walk law's cap is `Other_10`'s own `if (speed > N) speed =
                // N`, OUTSIDE the `walk` gate - dropping it let `separate` drive
                // these to its 16 px/frame clamp.
                gml_clamp_speed(&mut brain, &mut vel, enemy.kind, cap_f);
                // GML `Spider/Other_10:12` caps at `maxspeed`, which
                // `Spider/Alarm_1` holds at 3 and raises to 5 only on the close
                // chase, so it overrides the walk law's flat 5.
                if brain.maxspeed.is_finite() {
                    let cap_f = brain.maxspeed / crate::SIM_HZ as f32;
                    gml_clamp_speed(&mut brain, &mut vel, enemy.kind, cap_f);
                }
                // `Freak/Other_10:5`, `ExploFreak/Other_10:6` and
                // `RhinoFreak/Other_10:6` have no `walk -= 1`, so the impulse
                // runs every step while the alarm keeps `walk` armed.
                if !gml_walk_never_decrements(enemy.kind) {
                    brain.walk -= dt * 30.0;
                    if brain.walk < 0.0 {
                        brain.walk = 0.0;
                    }
                }
            }

            if matches!(enemy.kind, EnemyKind::ExploFreak | EnemyKind::RhinoFreak) {
                potential_step_solid(
                    &mut pos.0,
                    player_pos,
                    1.0,
                    def.radius,
                    &solids,
                    Some(&mask),
                );
            }
        }

        // Turtle fire-strip swap: visual-only, omitted (renderer resolves
        // strips from state; no gameplay effect).

        let was_dashing = brain.dash > 0.0;

        if enemy.kind == EnemyKind::Raven && !was_dashing && brain.melee.is_finished() {
            brain.dash = 0.2;
            brain.melee = GTimer::from_seconds(rng.random_range(0.9..1.8), TimerMode::Once);
            let side = glam::Vec2::new(-dir.y, dir.x) * brain.strafe_dir;
            vel.0 = (dir * -0.35 + side).normalize() * 420.0;
        }

        // GML `Alarm_2` for the objects whose decide only staged `ammo`.
        if enemy.kind != EnemyKind::DogGuardian && brain.burst_timer.just_finished() {
            gml_alarm_2_volley(
                enemy.kind,
                entity,
                &mut commands,
                &mut cues,
                epos,
                player_pos,
                &mut brain,
                particles_on,
                &mut rng,
            );
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

        // GML `Guardian` has no lunge: `Step_0:9-10` applies
        // `motion_add(direction, 0.6)` and caps at 0.6 px/frame every step. The
        // port invented an 18 px/frame dash, 30x that ceiling, so it is gone.
        if brain.dash > 0.0 {
            brain.dash = (brain.dash - dt).max(0.0);
        }

        let dashing = brain.dash > 0.0;

        if emplacement {
            vel.0 = glam::Vec2::ZERO;
        } else if dashing {
            // GML's dash only sets `speed`/`direction` (`Raven/Alarm_1.gml`); the
            // translation happens once, inside `enemy/Collision_Wall`'s
            // `move_bounce_solid` below. Moving here too integrated `vel * dt`
            // twice, and the second (swept) test started from the unchecked
            // first position - a dash into a wall resolved from inside
            // geometry and fought the bounce.
        } else {
            // Not gated on live velocity: GML's decide alarm fires on its own
            // schedule whatever `speed` is, and "no motion law" is static,
            // which `emplacement` already encodes (`Turret/Other_10` and
            // `Technomancer/Other_10` both set `speed = 0`). Gating on a live
            // `speed` swallowed the decide on any frame the enemy was at rest
            // and its re-arm never came back.
            // Kinds with their own `Alarm_1` further down must not also run the
            // generic decide, or its re-arm starves their own.
            let owns_decide = matches!(
                enemy.kind,
                EnemyKind::Gator
                    | EnemyKind::BuffGator
                    | EnemyKind::Jock
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
            if !owns_motion && !owns_decide && brain.attack.just_finished() {
                gml_alarm_1_decide(
                    enemy.kind,
                    epos,
                    player_pos,
                    &mask,
                    health,
                    &mut enemy,
                    &mut brain,
                    &mut vel,
                    dt,
                    &mut rng,
                    entity,
                    &mut dog_leaps,
                );
                if enemy.kind == EnemyKind::RobotGuard {
                    snowbot_fire.entry(entity).or_default().armed = enemy.touch_damage == 4;
                }
            }

            if enemy.kind == EnemyKind::DogGuardian
                && let Some(pose_state) = dog_pose.as_deref_mut()
                && let Some(leap) = dog_leaps.get(&entity)
            {
                pose_state.pose = leap.pose;
                pose_state.z = leap.z;
                pose_state.zspeed = leap.zspeed;
            }

            let cap = gml_speed_cap(enemy.kind);
            if !dog_airborne && vel.0.length() > cap {
                vel.0 = vel.0.normalize() * cap;
            }
        }

        // `enemy/Create_0.gml` `friction = 0.4`
        apply_gml_friction(&mut vel.0, brain.friction, dt);

        // GML `enemy/Collision_Wall`, per object. Objects that override the
        // event get only their own `move_bounce_solid` argument: no
        // `busycollisions` restore, no `friction` axis slide.
        let law = wall_law(enemy.kind);
        let saved_direction = vel.0.normalize_or_zero();
        let saved_speed = vel.0.length();
        let contact = move_bounce_solid(
            &mut pos.0,
            &mut vel.0,
            def.radius,
            dt,
            &solids,
            Some(&mask),
            matches!(law, EnemyWallLaw::Parent | EnemyWallLaw::PlainBounce),
        );
        if matches!(law, EnemyWallLaw::Parent) && contact.is_some() && run.loop_count <= 3 {
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
                vel.0.x = wall_probe_axis(
                    pos.0,
                    def.radius,
                    vel.0.x,
                    true,
                    brain.friction,
                    &solids,
                    &mask,
                );
                vel.0.y = wall_probe_axis(
                    pos.0,
                    def.radius,
                    vel.0.y,
                    false,
                    brain.friction,
                    &solids,
                    &mask,
                );
            }
        } else if matches!(law, EnemyWallLaw::SlidePush)
            && let Some(contact) = contact
        {
            // `Maggot/Collision_Wall.gml:5`.
            vel.0 += contact.normal * 30.0;
        }

        if enemy.kind == EnemyKind::DogGuardian && contact.is_some() {
            if let Some(leap) = dog_leaps.get_mut(&entity) {
                let airborne = leap.leap > 0.0;
                leap.bounced = true;
                if airborne {
                    enemy_cue(&mut cues, "sndDogGuardianBounce");
                }
            }
        }

        // InvSpider/InvLaserCrystal fade: visual-only, omitted.

        // GML objects that raise and clear their own `meleedamage` in
        // `Other_10`, and objects that override `Collision_Player` so contact
        // never reaches `scr_hit`.
        match enemy.kind {
            // `Turtle/Other_10`: 4 only while `walk > 0`, else 0.
            EnemyKind::Turtle => enemy.touch_damage = if brain.walk > 0.0 { 4 } else { 0 },
            // `FiredMaggot` replaces `Collision_Player` outright, so its
            // `meleedamage = 1` is dead code.
            EnemyKind::FiredMaggot => enemy.touch_damage = 0,
            // `ExploFreak/Collision_Wall`: it blows itself up on any wall
            // contact while the player is within 96 px.
            EnemyKind::ExploFreak if contact.is_some() && dist < 96.0 => {
                enemy_cue(&mut cues, "snd_mele");
                commands.entity(entity).despawn();
                continue;
            }
            _ => {}
        }

        separate(
            &positions,
            entity,
            epos,
            &mut vel.0,
            enemy.kind,
            run.loop_count,
            current_frame,
        );

        // GML keeps `direction`/`speed` authoritative: `motion_add` and
        // `move_bounce_solid` leave `direction` pointing along the current
        // velocity, which the next frame's `Other_10` reads. Without this,
        // `brain.heading` stayed frozen at the spawn value and every walk
        // impulse went one way for the enemy's whole life.
        sync_heading(&mut brain, &vel);

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
                    } else if matches!(enemy.kind, EnemyKind::Guardian | EnemyKind::ExploGuardian) {
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
                            &*enemy,
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
                    if enemy.kind == EnemyKind::Guardian {
                        brain.attack = GTimer::from_seconds(
                            (20.0 + rng.random_range(0.0..40.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
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
                            120.0,
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
                        brain.attack = if enemy.kind == EnemyKind::Guardian {
                            GTimer::from_seconds(charge_frames / 30.0, TimerMode::Once)
                        } else {
                            GTimer::from_seconds(
                                def.attack_cooldown + charge_frames / 30.0,
                                TimerMode::Once,
                            )
                        };
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
                                120.0,
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
                            &*enemy,
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

        // GML `Wolf/Alarm_2`: the roll is three `EnemyBullet1` along
        // `direction` (+/-20) and it zeroes `walk`. `Alarm_1` is what arms
        // it (`alarm[2] = 10; alarm[1] = 30`), so this reads that register
        // instead of rolling a second, unsynchronised chance.
        if enemy.kind == EnemyKind::Wolf {
            if brain.burst_timer.just_finished() {
                for off in [0.0_f32, 20.0, -20.0] {
                    let ang = brain.heading + off.to_radians();
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
                        120.0,
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
                brain.walk = 0.0;
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

        // GML `Mimic`/`SuperMimic`/`WepMimic/Other_10` has **no** `motion_add`:
        // the mimic never self-moves, it only spins in place pretending to be a
        // weapon. The port lerped it toward the player at up to 2 px/frame, so
        // every mimic in the level crept across the room.
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
                            120.0,
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
            && !dashing
            && !uses_charge
            && !has_dedicated_tick(enemy.kind)
            && !matches!(
                enemy.kind,
                EnemyKind::Gator
                    | EnemyKind::BuffGator
                    | EnemyKind::Crab
                    | EnemyKind::Raven
                    | EnemyKind::Salamander
                    | EnemyKind::PopoFreak
                    | EnemyKind::HostileHorror
            )
        {
            let fire_due = brain.fire_alarm.just_finished();
            if fire_due {
                brain.fire_alarm = GTimer::from_seconds(
                    gml_fire_rearm_secs(enemy.kind, &mut rng),
                    TimerMode::Once,
                );
            }
            // GML gates every shooter on sight plus a per-object distance and
            // roll, in `Alarm_1` short-circuit order. `shoot_range` was
            // standing in for that and let anything shoot through walls.
            let los = has_line_of_sight(epos, player_pos, &mask);
            if gml_wants_fire(enemy.kind, los, dist, &mut rng) {
                if def.burst {
                    if brain.burst_left > 0 {
                        if brain.burst_timer.just_finished() {
                            fire_enemy_bullet(
                                &mut commands,
                                &mut rng,
                                entity,
                                &*enemy,
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
                                brain.fire_alarm = GTimer::from_seconds(
                                    gml_fire_rearm_secs(enemy.kind, &mut rng),
                                    TimerMode::Once,
                                );
                            }
                        }
                    } else if fire_due {
                        brain.burst_left = def.bullets_per_shot;
                        brain.burst_timer =
                            GTimer::from_seconds(def.burst_interval, TimerMode::Once);
                        fire_enemy_bullet(
                            &mut commands,
                            &mut rng,
                            entity,
                            &*enemy,
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
                } else if fire_due {
                    fire_enemy_shot(&mut commands, &mut rng, entity, &*enemy, def, epos, dir);
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
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, p)| {
            (
                e,
                p.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
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
                                120.0,
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
            entity,
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
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, p, _)| {
            (
                e,
                p.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos, hurt) in &mut enemies {
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
            entity,
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
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, p, _, _, _)| {
            (
                e,
                p.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
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
            entity,
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
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, p, _, _)| {
            (
                e,
                p.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos, health, hurt) in &mut enemies {
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
            entity,
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
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, p, _)| {
            (
                e,
                p.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos, hurt) in &mut enemies {
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
                volume: 1.0,
                variance: 0.3,
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
            entity,
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
        // GML `FiredMaggot/Collision_Player` has no `instance_destroy`: the
        // fired maggot keeps flying and injects a Maggot on *every* step it
        // overlaps the player. `point_direction(other.x, other.y, x, y)` is
        // evaluated inside the new Maggot's `with` scope, i.e. player -> maggot,
        // so the spawn is thrown away from the player.
        if let Some((player_pos, true)) = player
            && next.distance(player_pos) <= 8.0 + enemy_def(enemy.kind).radius
        {
            let away = (next - player_pos).y.atan2((next - player_pos).x);
            queue_conversion_maggot(&mut commands, next, away, run.loop_count);
        }
        // GML `FiredMaggot/Collision_Wall`: bounce, spawn the Maggot, then
        // destroy the fired maggot. `other` is the Wall instance, so the
        // spawn is thrown away from the wall's origin.
        if blocked_by_wall(&mask, next, enemy_def(enemy.kind).radius) {
            let wall = nearest_floor_point(&mask, next);
            let away = (next - wall).y.atan2((next - wall).x);
            queue_conversion_maggot(&mut commands, next, away, run.loop_count);
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
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, p, _, _)| {
            (
                e,
                p.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
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
                    120.0,
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
            entity,
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
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, p, _, _)| {
            (
                e,
                p.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
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
                    120.0,
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
                    120.0,
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
            entity,
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
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, p)| {
            (
                e,
                p.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
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
                    120.0,
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
            entity,
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
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, p)| {
            (
                e,
                p.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
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
                120.0,
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
            entity,
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
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, p)| {
            (
                e,
                p.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
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
                    120.0,
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
            entity,
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
    let positions: Vec<(Entity, glam::Vec2, i32, f32)> = enemies
        .iter()
        .map(|(e, enemy, _, _, p, _, _)| {
            (
                e,
                p.0,
                crate::enemy_data::gml_size(enemy.kind),
                crate::enemy_data::enemy_def(enemy.kind).radius,
            )
        })
        .collect();
    let solids = prop_shapes(&props);
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos, mut health, hurt) in &mut enemies {
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
            entity,
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
        120.0,
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
        120.0,
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
        pos,
        shot_dir * speed,
        def.projectile_damage,
        def.projectile_lifetime,
        def.projectile_radius,
        120.0,
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
            120.0,
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

/// Per-pellet aim offsets and random spread for `fire_enemy_shot`, radians.
/// GML spells most of these out per object: `Molesarge/Alarm_1` fires at
/// `gunangle + {0, -15, +15, -30, +30}` with no extra jitter,
/// `SuperFireBaller/Alarm_1` uses `orandom(6)`, `Molefish/Alarm_1` uses
/// `random(4) - 2`. Everything else keeps the evenly spaced table fan and
/// the table's spread.
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
            pos,
            shot_dir * speed,
            def.projectile_damage,
            def.projectile_lifetime,
            def.projectile_radius,
            120.0,
            explosive_kind(enemy.kind),
        );
        finish_enemy_bullet(&mut commands.entity(e), enemy.kind);
    }
}

/// GML `WantBoss/Step_0` + `WantBoss/Alarm_0`: the Big Bandit's arming gate and
/// wall breach. Once it arms, the bandit climbs out of a wall near the player,
/// the wall breaks, and the screen shakes.
#[allow(clippy::too_many_arguments)]
pub fn tick_delayed_boss_spawns(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    run: Res<Run>,
    scarier: Res<ScarierFace>,
    heavy_heart: Res<HeavyHeart>,
    mask: Res<FloorMask>,
    mut trauma: ResMut<Trauma>,
    mut hitstop: ResMut<HitStop>,
    mut toast: ResMut<Toast>,
    triggers: Res<crate::secrets::SecretTriggers>,
    mut pending: Query<(Entity, &mut PendingDelayedBoss)>,
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
    // GML `WantBoss/Step_0:4-7,39-41`: the marker gives up on a floor with no
    // trash left, and on a non-final subarea once the floor is cleared.
    let living = enemies.iter().filter(|e| !enemy_def(e.kind).boss).count() as u32;
    let rad_maggots = enemies
        .iter()
        .filter(|e| e.kind == EnemyKind::RadMaggot)
        .count() as u32;

    for (marker_e, mut pending_boss) in &mut pending {
        if living == 0 {
            commands.entity(marker_e).despawn();
            continue;
        }

        // GML `WantBoss/Step_0:15`: `instance_number(enemy) -
        // instance_number(RadMaggot) > enemies * treshhold`. Unhatched rad
        // maggots are excluded from the count.
        let surviving = living.saturating_sub(rad_maggots);
        let killed = pending_boss.initial_trash.saturating_sub(surviving);
        if killed < pending_boss.kills_needed() {
            continue;
        }

        // GML `WantBoss/Step_0:20-23`. `detect_oasis_eligibility` already
        // evaluates exactly this: every chest open, none of the special chests
        // left, and at most 2% of the floor's trash dead.
        if pending_boss.require_open_chests && !triggers.oasis_chests_ready {
            continue;
        }

        // GML `WantBoss/Step_0:16-17`: a 4 s beat before the breach.
        if pending_boss.arm_delay > 0.0 {
            pending_boss.arm_delay -= time.delta_secs;
            continue;
        }

        let Ok(player_pos) = player_q.single() else {
            continue;
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

        let kind = pending_boss.kind;

        let spawn_pos = if let Some((p, _)) = best_wall {
            p
        } else {
            let mut rng = rand::rng();
            let mut best = mask.random_floor_pos(&mut rng, 120.0);
            for _ in 0..32 {
                let ang = rng.random_range(0.0..std::f32::consts::TAU);
                let cand = player_pos
                    + glam::Vec2::new(ang.cos(), ang.sin()) * rng.random_range(140.0..240.0);
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
            kind,
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
/// (`scrOnPopoKill` has no port equivalent - skipped with this note;
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
        volume: 1.0,
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
        apply_gml_friction(&mut vel.0, 0.4, dt);

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
                rate: anim.as_deref().map(|a| a.fps).unwrap_or(1.0).max(1.0),
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
        // GML `ToxicGas/Create_0.gml` inherits `damagesource/Create_0.gml:4`.
        target_health.hp -= 3;
        if target_team == Team::Enemy
            && let Some(next) = next_hurt.as_deref_mut()
        {
            next.0 = frame.0 + 5;
        }
        commands.entity(gas_entity).despawn();
    }
}

/// Verbatim `objects/Throne2Ball` law (`Step_0`): friction 0.25 bleeds speed;
/// once stalled, `timeout` accrues and past 15 ticks the ball sprays an
/// `EnemyBullet2` (Horror stats: damage 2 at 10 px/tick) along the latched
/// `angle` every tick, dying past `40 + loops * 10` ticks; while stalled but
/// young it aims at the nearest player (±30 degrees). (Position integration
/// rides `move_projectiles`; aim-converge particles are visual-only.)
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

/// GML `scrRight(0)`: `hspeed > 0 ? 1 : -1`.
fn gml_right_from_hspeed(vel: glam::Vec2) -> f32 {
    if vel.x > 0.0 { 1.0 } else { -1.0 }
}

/// GML `scrRight(1)`: east/west facing from `gunangle`, which GML holds in
/// degrees. Returns `1` for `> 270` or the `0..90` wedge, else `-1`.
fn gml_right_from_gunangle(gunangle_rad: f32) -> f32 {
    let deg = gunangle_rad.to_degrees().rem_euclid(360.0);
    if deg > 270.0 || (deg > 0.0 && deg < 90.0) {
        1.0
    } else {
        -1.0
    }
}

/// GML `objects/Grunt` and `objects/EliteGrunt` - the two IDPD units that
/// roll. `Alarm_1` picks a roll, a shot, a walk or a `PopoNade`/`IDPDRocket`
/// lob; every aggressive arm is gated on `freeze > 40`. `freeze` accrues 1
/// frame a step while the target moves or this object is damaged
/// (`Grunt/Other_10:14-18`), +3 while the player cannot shoot
/// (`Grunt/Other_10:16-18`, `scrFire.gml:8`).
/// Register map: `brain.attack` = `alarm[1]`, `burst_timer` = `alarm[2]`,
/// `roll` = `roll`, `roll_angle` = `angle`, `last_seen` = `lastx/lasty`,
/// `gunangle` = `gunangle` (radians); `freeze`, `fuel`, `grenades`, `right`,
/// `walk`, `wkick` and `ammo` keep their GML names. GML `direction` rides a
/// local heading map, as in [`tick_elite_inspectors`].
#[allow(clippy::too_many_arguments)]
pub fn tick_popo_rolls(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    player_q: Query<(&Pos, &Velocity, &FireCooldown), (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &Pos,
            &Health,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    nades: Query<(), (With<Projectile>, With<PopoNadeM>)>,
    mut cues: ResMut<Queue<AudioCue>>,
    catalog: Res<repame_anim::AnimCatalog>,
    save: Res<crate::savedata_part::SaveData>,
    mut headings: Local<HashMap<Entity, glam::Vec2>>,
    mut males: Local<HashMap<Entity, bool>>,
) {
    let Ok((player_pos, player_vel, player_cd)) = player_q.single() else {
        return;
    };
    let player_moving = player_vel.0.length_squared() > 0.001;
    let can_shoot = player_cd.timer.is_finished();
    let dt = time.delta_secs;
    let frames = dt * crate::SIM_HZ as f32;
    let mut rng = rand::rng();
    headings.retain(|e, _| enemies.contains(*e));
    males.retain(|e, _| enemies.contains(*e));
    let nades_live = nades.iter().count() as f32;
    let particles_on = save.settings.particles;

    for (entity, enemy, mut brain, mut vel, pos, health) in &mut enemies {
        let elite = enemy.kind == EnemyKind::IdpdElite;
        if !elite && enemy.kind != EnemyKind::IdpdGrunt {
            continue;
        }
        let male = *males.entry(entity).or_insert_with(|| rng.random_bool(0.5));
        let epos = pos.0;
        let to_player = player_pos.0 - epos;
        let dist = to_player.length();
        let aim = to_player.y.atan2(to_player.x);
        let los = has_line_of_sight(epos, player_pos.0, &mask);
        let heading = headings
            .entry(entity)
            .or_insert_with(|| glam::Vec2::from_angle(aim))
            .clone();

        // GML `Grunt/Other_10` / `EliteGrunt/Step_0`
        if !brain.roll {
            if elite {
                brain.fuel = 100.0;
                brain.roll_angle = 0.0;
            }
            if brain.walk > 0.0 {
                *headings.get_mut(&entity).unwrap() = heading;
                gml_motion_add_clamp(&mut vel.0, heading, 0.8, 3.0, dt);
                brain.walk -= frames;
                if brain.walk < 0.0 {
                    brain.walk = 0.0;
                }
            }
            let cap = 3.0 * 30.0;
            if vel.0.length() > cap {
                vel.0 = vel.0.normalize() * cap;
            }
            if player_moving || health.hp < health.max {
                brain.freeze += 1.0;
            }
            if !can_shoot {
                brain.freeze += 3.0;
            }
        } else if elite {
            // `EliteGrunt/Step_0:33-45`
            brain.fuel -= 1.0;
            if brain.fuel <= 0.0 {
                brain.roll = false;
                brain.fuel = 0.0;
            }
            gml_motion_add_clamp(&mut vel.0, glam::Vec2::from_angle(aim), 0.4, 7.0, dt);
            vel.0 = glam::Vec2::from_angle(aim) * 7.0 * 30.0;
            *headings.get_mut(&entity).unwrap() = glam::Vec2::from_angle(aim);
            brain.roll_angle = aim.to_degrees() - 90.0;
            crate::environment::spawn_motes(
                &mut commands,
                &catalog,
                particles_on,
                epos + glam::Vec2::new(rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0)),
                crate::comps_b::MoteStrip::Dust,
                1,
            );
        } else {
            // `Grunt/Other_10:24-33`
            vel.0 = heading * 5.0 * 30.0;
            brain.roll_angle += 40.0 * brain.right;
            if brain.roll_angle.abs() > 720.0 {
                brain.roll_angle = 0.0;
                brain.roll = false;
            }
            crate::environment::spawn_motes(
                &mut commands,
                &catalog,
                particles_on,
                epos + glam::Vec2::new(rng.random_range(-3.0..3.0), rng.random_range(0.0..6.0)),
                crate::comps_b::MoteStrip::Dust,
                1,
            );
        }

        // GML `EliteGrunt/Alarm_2` (3-round burst)
        if elite {
            brain.burst_timer.tick(dt);
            if brain.burst_timer.just_finished() {
                if brain.ammo > 0 {
                    enemy_cue(&mut cues, "sndGruntFire");
                    brain.wkick = 5.0;
                    fire_popo_bullet(
                        &mut commands,
                        entity,
                        enemy.kind,
                        epos,
                        brain.gunangle,
                        10.0,
                        rng.random_range(-2.0..2.0),
                        2.0,
                    );
                    brain.ammo -= 1;
                    brain.burst_timer = GTimer::from_seconds(3.0 / 30.0, TimerMode::Once);
                } else {
                    brain.burst_timer = GTimer::disarmed();
                }
            }
        }

        // GML `Alarm_1`
        brain.attack.tick(dt);
        if !brain.attack.just_finished() {
            continue;
        }
        // GML `Grunt/Alarm_1:1` and `EliteGrunt/Alarm_1:1-2`, which also
        // carries the elite's `if random(3) < 1 roll = 0` roll-out.
        let mut next = if elite {
            let base = rng.random_range(13.0..18.0);
            if rng.random_range(0.0..3.0) < 1.0 {
                brain.roll = false;
            }
            base
        } else {
            rng.random_range(15.0..35.0)
        };

        let roll_chance = if elite {
            // `EliteGrunt/Alarm_1:17` is a plain `random(2) < 1`; the roll
            // for `Grunt` is `Alarm_1:5`, sized by its own hp.
            0.5
        } else {
            // GML `Grunt/Alarm_1:5`: `random(hp / 2 + 2 + can_shoot * 3) < 1`.
            1.0 / (health.hp as f32 / 2.0 + 2.0 + if can_shoot { 3.0 } else { 0.0 })
        };

        if !elite && brain.roll {
            brain.attack = GTimer::from_seconds(next / 30.0, TimerMode::Once);
            continue;
        }

        if !brain.roll {
            if elite {
                // `EliteGrunt/Alarm_1:8-24`
                if dist < 64.0 && rng.random_range(0.0..4.0) < 3.0 {
                    brain.gunangle = aim;
                    brain.roll = false;
                    gml_motion_add_clamp(&mut vel.0, glam::Vec2::from_angle(aim), 10.0, 3.0, dt);
                    brain.walk = 20.0 + rng.random_range(0.0..10.0);
                    brain.right = gml_right_from_hspeed(vel.0);
                    next /= 3.0;
                } else if rng.random_range(0.0..2.0) < 1.0 && brain.freeze > 40.0 {
                    let turn: f32 = if dist > 150.0 {
                        rng.random_range(-30.0..=30.0)
                    } else {
                        rng.random_range(70.0..=130.0)
                            * if rng.random_bool(0.5) { 1.0 } else { -1.0 }
                    };
                    *headings.get_mut(&entity).unwrap() =
                        glam::Vec2::from_angle(aim + turn.to_radians());
                    vel.0 = glam::Vec2::from_angle(aim + turn.to_radians()) * 4.0 * 30.0;
                    brain.roll = true;
                    brain.roll_angle = 0.0;
                    enemy_cue(&mut cues, "sndEliteGruntRoll");
                    crate::environment::spawn_motes(
                        &mut commands,
                        &catalog,
                        particles_on,
                        epos,
                        crate::comps_b::MoteStrip::Dust,
                        1,
                    );
                }
            } else {
                // `Grunt/Alarm_1:5-9`
                if rng.random_range(0.0..1.0) < roll_chance && brain.freeze > 40.0 {
                    let turn: f32 = if dist > 150.0 {
                        rng.random_range(-30.0..=30.0)
                    } else {
                        (70.0 + rng.random_range(0.0..60.0))
                            * if rng.random_bool(0.5) { 1.0 } else { -1.0 }
                    };
                    *headings.get_mut(&entity).unwrap() =
                        glam::Vec2::from_angle(aim + turn.to_radians());
                    vel.0 = glam::Vec2::from_angle(aim + turn.to_radians()) * 4.0 * 30.0;
                    brain.roll = true;
                    brain.roll_angle = 0.0;
                    enemy_cue(&mut cues, "sndRoll");
                    crate::environment::spawn_motes(
                        &mut commands,
                        &catalog,
                        particles_on,
                        epos,
                        crate::comps_b::MoteStrip::Dust,
                        1,
                    );
                }
            }
        }

        if los {
            // GML `Grunt/Alarm_1:12-31`, `EliteGrunt/Alarm_1:27-42`.
            if elite {
                brain.gunangle = aim + rng.random_range(-15.0_f32..=15.0).to_radians();
                brain.right = gml_right_from_gunangle(brain.gunangle);
            } else {
                brain.gunangle = aim;
                if player_pos.0.x < epos.x {
                    brain.right = -1.0;
                } else if player_pos.0.x > epos.x {
                    brain.right = 1.0;
                    brain.last_seen = player_pos.0;
                }
            }
            let want_shot = if elite {
                rng.random_range(0.0..3.0) < 2.0 && brain.freeze > 40.0
            } else {
                rng.random_range(0.0..2.0) < 1.0 && brain.freeze > 40.0
            };
            if want_shot {
                if elite {
                    brain.ammo = 3;
                    brain.burst_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                    next = 14.0 + rng.random_range(0.0..2.0);
                } else {
                    enemy_cue(&mut cues, "sndGruntFire");
                    brain.wkick = 4.0;
                    fire_popo_bullet(
                        &mut commands,
                        entity,
                        enemy.kind,
                        epos,
                        brain.gunangle,
                        8.0,
                        rng.random_range(-3.0..3.0),
                        3.0,
                    );
                    next = 3.0 + rng.random_range(0.0..2.0);
                }
            } else {
                let turn: f32 = if dist > 48.0 {
                    rng.random_range(-25.0..=25.0)
                } else {
                    180.0 + rng.random_range(-25.0..=25.0)
                };
                *headings.get_mut(&entity).unwrap() =
                    glam::Vec2::from_angle(aim + turn.to_radians());
                vel.0 = glam::Vec2::from_angle(aim + turn.to_radians()) * 0.4 * 30.0;
                brain.walk = 10.0 + rng.random_range(0.0..10.0);
                if brain.freeze < 40.0 {
                    next += rng.random_range(0.0..30.0);
                }
            }
        } else if rng.random_range(0.0..4.0) < 1.0 {
            // GML `Grunt/Alarm_1:36-42`, `EliteGrunt/Alarm_1:44-50`: wander.
            let a = rng.random_range(0.0..std::f32::consts::TAU);
            let d = glam::Vec2::from_angle(a);
            *headings.get_mut(&entity).unwrap() = d;
            gml_motion_add_clamp(&mut vel.0, d, 0.4, 3.0, dt);
            brain.walk = 20.0 + rng.random_range(0.0..10.0);
            if elite {
                brain.gunangle = a;
                brain.right = gml_right_from_hspeed(vel.0);
            } else {
                brain.gunangle = a;
                brain.right = gml_right_from_hspeed(vel.0);
            }
        } else {
            // GML `Grunt/Alarm_1:43-58` (`PopoNade`),
            // `EliteGrunt/Alarm_1:51-64` (`IDPDRocket`).
            let nade_gate = rng.random_range(0.0..(5.0 + nades_live * 3.0)) < 1.0;
            let (has_grenades, want_freeze, want_range) = if elite {
                (brain.grenades > 0, brain.freeze > 40.0, dist < 180.0)
            } else {
                // GML also allows the lob when the player has drifted far
                // from the last sighting: `... or random(12) < 1`.
                let near_last = brain.last_seen.distance(player_pos.0) < 96.0
                    && epos.distance(brain.last_seen) > 64.0;
                (
                    brain.grenades > 0,
                    brain.freeze > 40.0,
                    near_last || rng.random_range(0.0..12.0) < 1.0,
                )
            };
            if elite {
                if has_grenades && want_freeze && rng.random_range(0.0..5.0) < 1.0 && want_range {
                    next += 30.0;
                    brain.roll = false;
                    brain.walk = 0.0;
                    brain.grenades -= 1;
                    brain.gunangle = aim + rng.random_range(-10.0_f32..=10.0).to_radians();
                    brain.wkick = 8.0;
                    enemy_cue(&mut cues, "sndEliteGruntRocketFire");
                    fire_popo_rocket(
                        &mut commands,
                        entity,
                        enemy.kind,
                        epos,
                        brain.gunangle,
                        rng.random_range(-10.0_f32..=10.0).to_radians(),
                        10.0,
                    );
                }
            } else if nade_gate && has_grenades && want_freeze && want_range {
                brain.grenades -= 1;
                let to_last = brain.last_seen - epos;
                brain.gunangle = to_last.y.atan2(to_last.x);
                brain.wkick = 8.0;
                enemy_cue(
                    &mut cues,
                    if male {
                        "sndGruntThrowNadeM"
                    } else {
                        "sndGruntThrowNadeF"
                    },
                );
                fire_popo_nade(
                    &mut commands,
                    entity,
                    enemy.kind,
                    epos,
                    brain.gunangle,
                    rng.random_range(-10.0_f32..=10.0).to_radians(),
                    10.0,
                );
            }
        }

        brain.attack = GTimer::from_seconds(next / 30.0, TimerMode::Once);
    }
}

/// GML `IDPDBullet` (inherits `EnemyBullet1`: damage 3, `typ = 1`
/// deflectable, `knockback_speed = 4`).
#[allow(clippy::too_many_arguments)]
fn fire_popo_bullet(
    commands: &mut Commands,
    owner: Entity,
    kind: EnemyKind,
    at: glam::Vec2,
    gunangle: f32,
    speed: f32,
    spread_deg: f32,
    _lifetime_frames: f32,
) {
    let a = gunangle + spread_deg.to_radians();
    let d = glam::Vec2::from_angle(a);
    let e = spawn_enemy_projectile(
        commands,
        owner,
        kind,
        at,
        d * speed * 30.0,
        3,
        3.0,
        4.5,
        120.0,
        false,
    );
    // GML `EnemyBullet1/Create_0.gml:4` `spr_fade = sprEnemyBulletHit`, which
    // the shared spawner leaves off. The Turret's `Alarm_2` is the one caller
    // that fires that object.
    if kind == EnemyKind::Turret {
        commands
            .entity(e)
            .insert(ProjectileFade("images/sprEnemyBulletHit.png"));
    }
}

/// GML `PopoNade` (inherits `Grenade`): `friction = 0`, `typ = 1`,
/// `damage = 0`, `alarm[0] = 90`, `alarm[1] = 10`, `flash_at = 20`.
#[allow(clippy::too_many_arguments)]
fn fire_popo_nade(
    commands: &mut Commands,
    owner: Entity,
    kind: EnemyKind,
    at: glam::Vec2,
    gunangle: f32,
    spread: f32,
    speed: f32,
) {
    let d = glam::Vec2::from_angle(gunangle + spread);
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        Team::Enemy,
        PopoNadeM,
        Projectile {
            damage: 0,
            life: GTimer::from_seconds(90.0 / 30.0, TimerMode::Once),
            radius: 5.0,
            knockback: 300.0,
            explosive: false,
            source: Some(DamageSource::enemy(owner, kind)),
        },
        ProjectileFriction(0.0),
        Velocity(d * speed * 30.0),
        Pos(at),
    ));
}

/// GML `IDPDRocket`: `alarm[1] = 5` before `active`, `damage = 4`,
/// `typ = 2` destructable.
#[allow(clippy::too_many_arguments)]
fn fire_popo_rocket(
    commands: &mut Commands,
    owner: Entity,
    kind: EnemyKind,
    at: glam::Vec2,
    gunangle: f32,
    spread: f32,
    speed: f32,
) {
    let d = glam::Vec2::from_angle(gunangle + spread);
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        Team::Enemy,
        Projectile {
            damage: 4,
            life: GTimer::from_seconds(3.0, TimerMode::Once),
            radius: 5.0,
            knockback: 120.0,
            explosive: false,
            source: Some(DamageSource::enemy(owner, kind)),
        },
        ProjectileTyp(2),
        Velocity(d * speed * 30.0),
        Pos(at),
    ));
}

/// GML `objects/VenuzTV/Destroy_0` and `objects/VaultStatue/Destroy_0`: both
/// props raise other objects on death, which the generic prop-damage path has
/// no queries to do, so this owns their death outright (the generic path sees
/// [`SpecialPropDeath`] and leaves them at `hp <= 0`).
/// `VaultStatue/Destroy_0.gml:1-14`: raise a `CrownGuardian` on the spot, zero
/// every *other* statue (one hit cascades), destroy the `CrownPickup`, leave a
/// `sprVaultStatueDead` corpse. GML defers `instance_destroy`, so N statues
/// yield N guardians.
/// `VenuzTV/Destroy_0.gml:3-14`: eight money feathers, destroy the
/// `YungVenuzCouch`, raise `YVBoss`, destroy the `VenuzCouch`.
/// GML `objects/NecroReviveArea/Alarm_0.gml`: after 15 frames the nearest
/// corpse is re-created as a `Necromancer`, but only if the marker still
/// overlaps it and the spot is free. `GameCont.kills -= 1` gives the kill back.
pub fn tick_necro_revive_areas(
    time: Res<SimTime>,
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mask: Res<FloorMask>,
    mut areas: Query<(Entity, &Pos, &mut NecroReviveArea), With<NecroReviveArea>>,
    corpses: Query<(Entity, &Pos), (With<Corpse>, Without<NecroReviveArea>)>,
) {
    let dt = time.delta_secs;
    for (e, apos, mut area) in &mut areas {
        area.timer.tick(dt);
        if !area.timer.just_finished() {
            continue;
        }
        if let Some((ce, cpos)) = corpses
            .iter()
            .min_by(|a, b| {
                apos.0.distance_squared(a.1 .0)
                    .total_cmp(&apos.0.distance_squared(b.1 .0))
            })
            // GML `place_meeting(x, y, other)` against `sprNecroReviveArea`'s
            // 33x36 bbox.
            && (apos.0.x - cpos.0.x).abs() <= 17.0
            && (apos.0.y - cpos.0.y).abs() <= 18.0
            && mask.is_walkable(cpos.0)
        {
            commands.entity(ce).despawn();
            queue_enemy_spawn(&mut commands, EnemyKind::Necromancer, cpos.0, 1.0, 0);
        }
        commands.entity(e).despawn();
        cues.push(AudioCue {
            name: "sndNecromancerRevive",
            volume: 0.2,
            variance: 0.0,
        });
    }
}

pub fn tick_special_props(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    run: Res<Run>,
    mut save: ResMut<crate::savedata_part::SaveData>,
    mut dirty: ResMut<crate::comps_a::SaveDirty>,
    mut props: Query<
        (
            Entity,
            &mut Prop,
            &Pos,
            Option<&SpecialPropDeath>,
            Option<&crate::crown::VaultStatue>,
            Option<&PropSprites>,
        ),
        (With<Prop>, Without<Player>, Without<Enemy>),
    >,
    mut pedestals: Query<Entity, (With<CrownPedestal>, Without<Prop>)>,
    mut couches: Query<(Entity, &Pos), (With<YvCouch>, Without<Prop>)>,
) {
    let mut rng = rand::rng();
    let dying: Vec<(Entity, glam::Vec2, SpecialPropDeath)> = props
        .iter()
        .filter_map(|(e, prop, pos, kind, _, _)| {
            (prop.hp <= 0)
                .then(|| kind.map(|k| (e, pos.0, *k)))
                .flatten()
        })
        .collect();
    for (entity, at, kind) in dying {
        if let Ok((_, _, _, _, _, sprites)) = props.get(entity) {
            if let Some(ps) = sprites {
                crate::environment::spawn_prop_corpse(&mut commands, &catalog, at, &ps);
            }
        }
        match kind {
            SpecialPropDeath::VaultStatue => {
                queue_enemy_spawn(
                    &mut commands,
                    EnemyKind::CrownGuardian,
                    at,
                    1.0,
                    run.loop_count,
                );
                for (other, mut oprop, _, _, is_statue, _) in &mut props {
                    if other != entity && is_statue.is_some() {
                        oprop.hp = 0;
                    }
                }
                for pedestal in &mut pedestals {
                    commands.entity(pedestal).despawn();
                }
            }
            SpecialPropDeath::VenuzTv => {
                for _ in 0..8 {
                    let a = rng.random_range(0.0..std::f32::consts::TAU);
                    crate::environment::spawn_native_streak(
                        &mut commands,
                        false,
                        at,
                        a,
                        3.0 * 30.0,
                    );
                }
                for (couch, cpos) in &mut couches {
                    commands.entity(couch).despawn();
                    queue_enemy_spawn(
                        &mut commands,
                        EnemyKind::YvBoss,
                        cpos.0,
                        1.0,
                        run.loop_count,
                    );
                }
                // GML `GameCont/Other_5.gml:42` / `YungCuz/Alarm_2.gml:2`.
                if crate::savedata_part::try_unlock_race(&mut save, crate::data::RaceId::Cuz) {
                    dirty.0 = true;
                }
            }
        }
        commands.entity(entity).despawn();
    }
}

/// GML `objects/IceFlower` - the only route into `area_jungle`; the port
/// previously had no flower, so LAST WISH was spent on nothing.
/// `Create_0.gml:7-10` snaps the flower onto the nearest `Floor` and nudges it
/// clear of geometry. `Player/Collision_IceFlower.gml:4-22` feeds it on an
/// interact press (1 damage, blood, `feed++`) then runs the flower's own step
/// on the spot, so the fourth feed opens the route in the same press. Lines
/// 24-29 drag the player in at 1 px a step.
pub fn tick_ice_flowers(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mask: Res<FloorMask>,
    input: Res<crate::input::NtInput>,
    mut triggers: ResMut<crate::secrets::SecretTriggers>,
    mut player_q: Query<(&mut Player, &mut Pos, &mut Health), (With<Player>, Without<Enemy>)>,
    mut flowers: Query<
        (Entity, &mut Pos, &mut crate::progression::IceFlowerFeed),
        (With<Enemy>, Without<Player>),
    >,
    mut enemy_shots: Query<(Entity, &Team), With<Projectile>>,
    mut enemies: Query<(Entity, &mut Health), With<Enemy>>,
    mut inited: Local<std::collections::HashSet<Entity>>,
) {
    if flowers.is_empty() {
        return;
    }
    let mut rng = rand::rng();
    inited.retain(|e| flowers.contains(*e));
    let pressed = input.peek_interact_pressed();
    let Ok((mut player, mut ppos, mut php)) = player_q.single_mut() else {
        return;
    };

    for (entity, mut pos, mut feed) in &mut flowers {
        if inited.insert(entity) {
            // GML `Create_0.gml:7-10`: sit on the nearest floor tile, then
            // push clear of anything solid.
            let cell = mask.world_to_cell(pos.0);
            let seated = mask.cell_center(cell);
            let mut at = glam::Vec2::new(seated.x - 16.0, seated.y - 16.0);
            mask.resolve_circle(&mut at, 18.0);
            let away = rng.random_range(0.0..std::f32::consts::TAU);
            pos.0 = at + glam::Vec2::from_angle(away) * 16.0;
            mask.resolve_circle(&mut pos.0, 18.0);
        }

        let to_flower = pos.0 - ppos.0;
        let dist = to_flower.length();

        if pressed && dist < 40.0 {
            // GML `Player/Collision_IceFlower.gml:6-19`, in draw order: one
            // damage tick, the blood fan, then `feed++`.
            php.hp -= 1;
            let mut dir = rng.random_range(0.0..std::f32::consts::TAU);
            for _ in 0..2 + rng.random_range(0..3) {
                crate::environment::spawn_native_streak(
                    &mut commands,
                    false,
                    pos.0,
                    dir,
                    5.0 * 30.0,
                );
                dir += (60.0 + rng.random_range(0.0..30.0_f32)).to_radians();
            }
            feed.0 += 1;
        }

        // GML `:24-29`: haul the player in one pixel a step, per axis, only
        // where the destination is clear.
        if dist > 0.0 {
            let step = glam::Vec2::from_angle(to_flower.y.atan2(to_flower.x));
            let nx = ppos.0 + glam::Vec2::new(step.x, 0.0);
            if mask.is_walkable(nx) {
                ppos.0 = nx;
            }
            let ny = ppos.0 + glam::Vec2::new(0.0, step.y);
            if mask.is_walkable(ny) {
                ppos.0 = ny;
            }
        }

        if feed.0 >= 4 {
            crate::progression::ice_flower_jungle(
                &mut commands,
                &catalog,
                &mut triggers,
                &mut player,
                &mut enemy_shots,
                &mut enemies,
                entity,
                pos.0,
            );
        }
    }
}

/// GML `objects/Shielder` and `objects/Inspector` - the two non-rolling IDPD
/// gunners. `Shielder` alternates an 8-round `IDPDBullet` burst, a
/// `PopoShield`, and a walk; `Inspector` mind-controls the player, slugs, and
/// lobs `PopoNade` at the last-seen position. Both re-arm inside the branch
/// they took, so the cadence is branch-specific - no table cooldown here.
/// Register map: `brain.attack` = `alarm[1]`, `burst_timer` = `alarm[2]`,
/// `last_seen` = `lastx/lasty`, `gunangle` = `gunangle` (radians);
/// `ammo`, `freeze`, `grenades`, `right`, `control`, `walk` and `wkick` keep
/// their GML names. GML `direction` rides the local heading map.
#[allow(clippy::too_many_arguments)]
pub fn tick_popo_gunners(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    mut player_q: Query<(&Velocity, &mut Pos, &FireCooldown), (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &Pos,
            &Health,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    shields: Query<&PopoShieldM, (With<PopoShieldM>, Without<Enemy>)>,
    nades: Query<(), (With<Projectile>, With<PopoNadeM>)>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut trauma: ResMut<Trauma>,
    mut headings: Local<HashMap<Entity, glam::Vec2>>,
    mut males: Local<HashMap<Entity, bool>>,
) {
    let Ok((pvel, mut ppos, player_cd)) = player_q.single_mut() else {
        return;
    };
    let player_moving = pvel.0.length_squared() > 0.001;
    let can_shoot = player_cd.timer.is_finished();
    let player_at = ppos.0;
    let dt = time.delta_secs;
    let frames = dt * crate::SIM_HZ as f32;
    let mut rng = rand::rng();
    headings.retain(|e, _| enemies.contains(*e));
    males.retain(|e, _| enemies.contains(*e));
    let nades_live = nades.iter().count() as f32;
    // `Shielder/Alarm_2` opens fire only when it has no shield up.
    let shielded: std::collections::HashSet<Entity> = shields.iter().map(|s| s.creator).collect();

    for (entity, enemy, mut brain, mut vel, pos, health) in &mut enemies {
        let shielder = enemy.kind == EnemyKind::IdpdShield;
        if !shielder && enemy.kind != EnemyKind::IdpdInspector {
            continue;
        }
        let male = *males.entry(entity).or_insert_with(|| rng.random_bool(0.5));
        let epos = pos.0;
        let to_player = player_at - epos;
        let dist = to_player.length();
        let aim = to_player.y.atan2(to_player.x);
        let los = has_line_of_sight(epos, player_at, &mask);
        let heading = headings
            .entry(entity)
            .or_insert_with(|| glam::Vec2::from_angle(aim))
            .clone();

        // GML `Other_10`
        if brain.walk > 0.0 {
            gml_motion_add_clamp(
                &mut vel.0,
                heading,
                0.8,
                if shielder { 3.5 } else { 3.0 },
                dt,
            );
            brain.walk -= frames;
            if brain.walk < 0.0 {
                brain.walk = 0.0;
            }
        }
        let cap = if shielder { 3.5 } else { 3.0 } * 30.0;
        if vel.0.length() > cap {
            vel.0 = vel.0.normalize() * cap;
        }
        if player_moving || health.hp < health.max {
            brain.freeze += 1.0;
        }
        if !can_shoot {
            brain.freeze += 3.0;
        }
        // `Inspector/Other_10:20-28`: the control field drags the player one
        // pixel per step toward the inspector, per axis, only where free.
        if !shielder && brain.control {
            let d = epos - player_at;
            if d.length() < 240.0 {
                let step = glam::Vec2::from_angle(d.y.atan2(d.x));
                let cand = player_at + step;
                if mask.is_walkable(cand) {
                    ppos.0 = cand;
                }
            }
        }

        // GML `Shielder/Alarm_2`
        if shielder {
            brain.burst_timer.tick(dt);
            if brain.burst_timer.just_finished() && brain.ammo > 0 && !shielded.contains(&entity) {
                brain.wkick = 5.0;
                gml_motion_add_clamp(
                    &mut vel.0,
                    glam::Vec2::from_angle(brain.gunangle + std::f32::consts::PI),
                    0.5,
                    3.5,
                    dt,
                );
                enemy_cue(&mut cues, "sndGruntFire");
                fire_popo_bullet(
                    &mut commands,
                    entity,
                    enemy.kind,
                    epos,
                    brain.gunangle,
                    8.0,
                    rng.random_range(-10.0..=10.0_f32),
                    0.0,
                );
                brain.burst_timer = GTimer::from_seconds(3.0 / 30.0, TimerMode::Once);
                brain.ammo -= 1;
            }
        }

        // GML `Alarm_1`
        brain.attack.tick(dt);
        if !brain.attack.just_finished() {
            continue;
        }
        if shielder {
            let mut next = rng.random_range(15.0..20.0);
            if los {
                brain.gunangle = aim;
                brain.right = if player_at.x < epos.x {
                    -1.0
                } else if player_at.x > epos.x {
                    1.0
                } else {
                    brain.right
                };
                if rng.random_range(0.0..2.0) < 1.0 && brain.freeze > 40.0 && dist <= 250.0 {
                    brain.burst_timer = GTimer::from_seconds(2.0 / 30.0, TimerMode::Once);
                    brain.ammo = 8;
                    next = 50.0;
                } else if rng.random_range(0.0..3.0) < 1.0 {
                    enemy_cue(
                        &mut cues,
                        if male {
                            "sndShielderShieldM"
                        } else {
                            "sndShielderShieldF"
                        },
                    );
                    raise_popo_shield(&mut commands, entity, epos);
                    next = 85.0;
                    vel.0 = glam::Vec2::ZERO;
                    brain.walk = 0.0;
                } else {
                    let turn: f32 = if dist > 64.0 {
                        rng.random_range(-25.0..=25.0)
                    } else {
                        180.0 + rng.random_range(-45.0..=45.0)
                    };
                    set_heading(&mut headings, entity, aim + turn.to_radians());
                    vel.0 = glam::Vec2::from_angle(aim + turn.to_radians()) * 0.4 * 30.0;
                    brain.walk = 10.0 + rng.random_range(0.0..10.0);
                    if brain.freeze < 40.0 {
                        next += rng.random_range(0.0..30.0);
                    }
                }
            } else if rng.random_range(0.0..3.0) < 1.0 {
                let a = rng.random_range(0.0..std::f32::consts::TAU);
                set_heading(&mut headings, entity, a);
                gml_motion_add_clamp(&mut vel.0, glam::Vec2::from_angle(a), 0.4, 3.5, dt);
                brain.walk = 20.0 + rng.random_range(0.0..10.0);
                brain.gunangle = a;
                brain.right = gml_right_from_hspeed(vel.0);
            } else if brain.freeze > 40.0 && rng.random_range(0.0..4.0) < 1.0 && dist < 96.0 {
                enemy_cue(
                    &mut cues,
                    if male {
                        "sndShielderShieldM"
                    } else {
                        "sndShielderShieldF"
                    },
                );
                raise_popo_shield(&mut commands, entity, epos);
                next = 75.0;
                vel.0 = glam::Vec2::ZERO;
                brain.walk = 0.0;
            }
            // GML's `else if random(10) < 1 and roll = 0` no-target arm is
            // dead: `Shielder/Create_0` sets `roll = 1` and nothing clears
            // it, so it never runs.
            brain.attack = GTimer::from_seconds(next / 30.0, TimerMode::Once);
            continue;
        }

        // GML `Inspector/Alarm_1`
        let mut next = rng.random_range(20.0..40.0);
        let was_control = brain.control;
        brain.control = false;
        if rng.random_range(0.0..3.0) < 1.0 && brain.freeze > 40.0 {
            brain.control = true;
        } else if los {
            brain.gunangle = aim;
            if player_at.x < epos.x {
                brain.right = -1.0;
            } else if player_at.x > epos.x {
                brain.right = 1.0;
                brain.last_seen = player_at;
            }
            if rng.random_range(0.0..2.0) < 1.0 && brain.freeze > 40.0 {
                // GML `PopoSlug`: friction 0.8, knockback 8, damage 5,
                // `typ = 1`.
                enemy_cue(&mut cues, "sndGruntFire");
                let a = brain.gunangle + rng.random_range(-6.0..=6.0_f32).to_radians();
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Team::Enemy,
                    Projectile {
                        damage: 5,
                        life: GTimer::from_seconds(2.5, TimerMode::Once),
                        radius: 5.0,
                        knockback: 240.0,
                        explosive: false,
                        source: Some(DamageSource::enemy(entity, enemy.kind)),
                    },
                    ProjectileFriction(0.8),
                    ProjectileTyp(1),
                    Velocity(glam::Vec2::from_angle(a) * 16.0 * 30.0),
                    Pos(epos),
                ));
                brain.wkick = 8.0;
                trauma.add(3.0 / 20.0);
                next = 20.0 + rng.random_range(0.0..10.0);
            } else {
                let turn: f32 = if dist > 48.0 {
                    rng.random_range(-25.0..=25.0)
                } else {
                    180.0 + rng.random_range(-25.0..=25.0)
                };
                set_heading(&mut headings, entity, aim + turn.to_radians());
                vel.0 = glam::Vec2::from_angle(aim + turn.to_radians()) * 0.4 * 30.0;
                brain.walk = 10.0 + rng.random_range(0.0..10.0);
                if brain.freeze < 40.0 {
                    next += rng.random_range(0.0..30.0);
                }
            }
        } else if rng.random_range(0.0..4.0) < 1.0 {
            let a = rng.random_range(0.0..std::f32::consts::TAU);
            set_heading(&mut headings, entity, a);
            gml_motion_add_clamp(&mut vel.0, glam::Vec2::from_angle(a), 0.4, 3.0, dt);
            brain.walk = 20.0 + rng.random_range(0.0..10.0);
            brain.gunangle = a;
            brain.right = gml_right_from_hspeed(vel.0);
        } else {
            let gate = rng.random_range(0.0..(5.0 + nades_live * 3.0)) < 1.0;
            let near_last =
                brain.last_seen.distance(player_at) < 96.0 && epos.distance(brain.last_seen) > 64.0;
            if gate
                && brain.grenades > 0
                && brain.freeze > 40.0
                && (near_last || rng.random_range(0.0..12.0) < 1.0)
            {
                brain.grenades -= 1;
                let to_last = brain.last_seen - epos;
                brain.gunangle = to_last.y.atan2(to_last.x);
                brain.wkick = 8.0;
                enemy_cue(
                    &mut cues,
                    if male {
                        "sndGruntThrowNadeM"
                    } else {
                        "sndGruntThrowNadeF"
                    },
                );
                fire_popo_nade(
                    &mut commands,
                    entity,
                    enemy.kind,
                    epos,
                    brain.gunangle,
                    rng.random_range(-10.0..=10.0_f32).to_radians(),
                    10.0,
                );
            }
        }
        if !brain.control && was_control {
            enemy_cue(
                &mut cues,
                if male {
                    "sndInspectorEndM"
                } else {
                    "sndInspectorEndF"
                },
            );
        } else if brain.control && !was_control {
            enemy_cue(
                &mut cues,
                if male {
                    "sndInspectorStartM"
                } else {
                    "sndInspectorStartF"
                },
            );
        }
        brain.attack = GTimer::from_seconds(next / 30.0, TimerMode::Once);
    }
}

fn set_heading(headings: &mut HashMap<Entity, glam::Vec2>, entity: Entity, angle: f32) {
    headings.insert(entity, glam::Vec2::from_angle(angle));
}

/// GML `Shielder/Alarm_1`: `instance_create(x, y, PopoShield)` with
/// `creator = other.id`.
fn raise_popo_shield(commands: &mut Commands, creator: Entity, at: glam::Vec2) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        PopoShieldM {
            creator,
            frames: 60.0,
        },
        Pos(at),
    ));
}

/// GML `PopoShield/Step_2` + `Other_7` + `Collision_projectile`.
pub fn tick_popo_shields(
    mut commands: Commands,
    mut shields: Query<(Entity, &Pos, &mut PopoShieldM), (With<PopoShieldM>, Without<Enemy>)>,
    mut owners: Query<&mut Pos, (With<Enemy>, Without<PopoShieldM>)>,
    mut timers: Query<&mut EnemyBrain, (With<Enemy>, Without<PopoShieldM>)>,
    mut shots: Query<
        (
            Entity,
            &Pos,
            &mut Team,
            &mut Velocity,
            Option<&ProjectileTyp>,
        ),
        (With<Projectile>, Without<Enemy>, Without<Player>),
    >,
) {
    for (entity, spos, mut shield) in &mut shields {
        if let Ok(mut opos) = owners.get_mut(shield.creator) {
            opos.0 = spos.0;
        }
        shield.frames -= 1.0;
        if shield.frames > 0.0 {
            for (s, ppos, mut team, mut vel, typ) in &mut shots {
                if *team != Team::Player {
                    continue;
                }
                if ppos.0.distance(spos.0) > 20.0 {
                    continue;
                }
                match typ.map(|t| t.0).unwrap_or(0) {
                    1 => {
                        *team = Team::Enemy;
                        vel.0 = (ppos.0 - spos.0).normalize_or_zero() * vel.0.length().max(60.0);
                    }
                    2 => {
                        commands.entity(s).despawn();
                    }
                    _ => {}
                }
            }
            continue;
        }
        // GML `Other_7`: the pop charges its owner's next decision.
        if let Ok(mut brain) = timers.get_mut(shield.creator) {
            brain.attack = GTimer::from_seconds(
                (brain.attack.duration() + 20.0 / 30.0).max(0.0),
                TimerMode::Once,
            );
        }
        commands.entity(entity).despawn();
    }
}

/// GML `objects/Turret`. Bolted down (`Other_10` pins `x/y` to
/// `xprevious/yprevious`), dies off-floor. `Alarm_1` arms a 10-round burst only
/// on sight inside 160 px; `Alarm_3`, set to 1 at create, is the one-shot that
/// carves every wall it is embedded in.
/// Register map: `brain.attack` = `alarm[1]`, `burst_timer` = `alarm[2]`;
/// `ammo` and `gunangle` keep their GML names.
pub fn tick_turrets(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    mut cues: ResMut<Queue<AudioCue>>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (Entity, &Enemy, &mut EnemyBrain, &mut Velocity, &Pos),
        (With<Enemy>, Without<Prop>),
    >,
    walls: Query<(&Pos, &WallCell), (With<WallTile>, Without<Enemy>, Without<Player>)>,
    mut inited: Local<std::collections::HashSet<Entity>>,
) {
    let Ok(player_pos) = player_q.single() else {
        return;
    };
    let dt = time.delta_secs;
    let mut rng = rand::rng();
    inited.retain(|e| enemies.contains(*e));
    let wall_snapshot: Vec<(glam::Vec2, (i32, i32))> =
        walls.iter().map(|(p, c)| (p.0, (c.0, c.1))).collect();

    for (entity, enemy, mut brain, mut vel, pos) in &mut enemies {
        if enemy.kind != EnemyKind::Turret {
            continue;
        }
        let epos = pos.0;
        if !inited.contains(&entity) {
            inited.insert(entity);
            // GML `Create_0`: `ammo = 0`, `gunangle = random_angle`,
            // `offset = 0`, `alarm[3] = 1`. `Alarm_3` destroys every wall
            // the turret is embedded in, and it lands on this first step.
            brain.ammo = 0;
            brain.gunangle = rng.random_range(0.0..std::f32::consts::TAU);
            crate::walls::queue_wall_breaks_in_radius(&mut commands, &wall_snapshot, epos, 32.0);
        }

        // GML `Other_10`: pinned in place, and dies off-floor.
        vel.0 = glam::Vec2::ZERO;
        if !mask.is_walkable(epos) {
            commands.entity(entity).despawn();
            continue;
        }

        // GML `Alarm_2`: the burst itself.
        brain.burst_timer.tick(dt);
        if brain.burst_timer.just_finished() {
            if brain.ammo > 0 {
                brain.ammo -= 1;
                brain.burst_timer = GTimer::from_seconds(3.0 / 30.0, TimerMode::Once);
                let a = brain.gunangle + rng.random_range(-4.0..=4.0_f32).to_radians();
                fire_popo_bullet(&mut commands, entity, enemy.kind, epos, a, 8.0, 0.0, 0.0);
                enemy_cue(&mut cues, "sndTurretFire");
            } else {
                // GML `Alarm_2:12-13`: back to idle and poll again soon.
                brain.attack =
                    GTimer::from_seconds(rng.random_range(10.0..=15.0) / 30.0, TimerMode::Once);
            }
        }

        // GML `Alarm_1`.
        brain.attack.tick(dt);
        if !brain.attack.just_finished() {
            continue;
        }
        let next = rng.random_range(50.0..=60.0);
        let to_player = player_pos.0 - epos;
        let dist = to_player.length();
        if has_line_of_sight(epos, player_pos.0, &mask)
            && rng.random_range(0.0..4.0) < 3.0
            && dist < 160.0
        {
            brain.ammo = 10;
            brain.burst_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
            let offset = rng.random_range(-5.0..=5.0_f32);
            brain.gunangle = to_player.y.atan2(to_player.x) + offset.to_radians();
        }
        brain.attack = GTimer::from_seconds(next / 30.0, TimerMode::Once);
    }
}

/// GML `objects/MeleeFake`: a dormant `MeleeBandit` that wakes into its real
/// self. Until then it does not move and does not attack -- the port had it
/// chasing and hitting for 1.
/// GML `Step_0` wakes on any of: damaged (`hp < max_hp`), alone
/// (`!instance_number(enemy)`), or the player within 64 px on a clear line with
/// no `Portal` on the floor. `Destroy_0` runs the *real* object's create/destroy
/// pair, so a fake killed while dormant still drops the real assassin's rads.
pub fn tick_melee_fakes(
    mut commands: Commands,
    run: Res<Run>,
    mask: Res<FloorMask>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    enemies: Query<&Pos, (With<Enemy>, Without<Player>)>,
    fakes: Query<(Entity, &Enemy, &Pos, &Health), (With<Enemy>, Without<Player>, Without<Prop>)>,
    portals: Query<(), (With<crate::comps_b::Portal>, Without<Enemy>)>,
) {
    if fakes.is_empty() {
        return;
    }
    let has_portal = !portals.is_empty();
    let Ok(player_pos) = player_q.single() else {
        return;
    };
    // GML `!instance_number(enemy)`: any other live enemy keeps it asleep.
    let alive: Vec<glam::Vec2> = enemies.iter().map(|p| p.0).collect();

    for (entity, enemy, pos, health) in &fakes {
        if enemy.kind != EnemyKind::MeleeFake {
            continue;
        }
        let wakes = health.hp < health.max
            || alive.is_empty()
            || (!has_portal
                && pos.0.distance(player_pos.0) <= 64.0
                && has_line_of_sight(pos.0, player_pos.0, &mask));
        if !wakes {
            continue;
        }
        queue_enemy_spawn(
            &mut commands,
            EnemyKind::MeleeBandit,
            pos.0,
            1.0,
            run.loop_count,
        );
        commands.entity(entity).despawn();
    }
}

/// GML `objects/Ratking`, and the `instance_change(RatkingRage, false)` it
/// rolls into after vomiting more than 24 rats (`Alarm_2:14-19`). The rage form
/// charges at touch 4, destroys the walls it hits (`Collision_Wall:4`), and
/// bursts five `FastRat` plus five `AcidStreak` on death.
/// Register map: `brain.attack` = `alarm[1]`, `burst_timer` = `alarm[2]`,
/// `ammo` = `ammo`, `ratking_spawns` = `spawns`, `ratking_rage` = the
/// `instance_change`; `brain.mydir`-equivalent rides the local heading map.
pub fn tick_ratking(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &mut Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &Pos,
            &Health,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    walls: Query<(&Pos, &WallCell), (With<WallTile>, Without<Enemy>, Without<Player>)>,
    mut headings: Local<HashMap<Entity, glam::Vec2>>,
) {
    let Ok(player_pos) = player_q.single() else {
        return;
    };
    let dt = time.delta_secs;
    let frames = dt * crate::SIM_HZ as f32;
    let mut rng = rand::rng();
    headings.retain(|e, _| enemies.contains(*e));
    let wall_snapshot: Vec<(glam::Vec2, (i32, i32))> =
        walls.iter().map(|(p, c)| (p.0, (c.0, c.1))).collect();

    for (entity, mut enemy, mut brain, mut vel, pos, _health) in &mut enemies {
        if enemy.kind != EnemyKind::Ratking {
            continue;
        }
        let epos = pos.0;
        let to_player = player_pos.0 - epos;
        let dist = to_player.length();
        let aim = to_player.y.atan2(to_player.x);
        let heading = headings
            .entry(entity)
            .or_insert_with(|| glam::Vec2::from_angle(aim))
            .clone();

        // GML `Other_10`
        if brain.walk > 0.0 {
            let cap = if brain.ratking_rage { 6.0 } else { 2.0 };
            gml_motion_add_clamp(
                &mut vel.0,
                heading,
                if brain.ratking_rage { 1.5 } else { 0.5 },
                cap,
                dt,
            );
            brain.walk -= frames;
            if brain.walk < 0.0 {
                brain.walk = 0.0;
            }
        }
        let cap = if brain.ratking_rage { 6.0 } else { 2.0 } * 30.0;
        if vel.0.length() > cap {
            vel.0 = vel.0.normalize() * cap;
        }

        // GML `Collision_Wall`: only the rage form breaks walls.
        if brain.ratking_rage {
            crate::walls::queue_wall_breaks_in_radius(
                &mut commands,
                &wall_snapshot,
                epos,
                crate::enemy_data::enemy_def(enemy.kind).radius,
            );
        }

        // GML `Alarm_2`: the rat stream
        brain.burst_timer.tick(dt);
        if brain.burst_timer.just_finished() {
            if brain.ammo > 0 {
                brain.ratking_spawns += 1.0;
                brain.ammo -= 1;
                brain.burst_timer = GTimer::from_seconds(6.0 / 30.0, TimerMode::Once);
                let a = brain.gunangle + rng.random_range(-20.0..=20.0_f32).to_radians();
                queue_enemy_spawn_birth(
                    &mut commands,
                    EnemyKind::FastRat,
                    epos,
                    1.0,
                    run.loop_count,
                    false,
                    Some(glam::Vec2::from_angle(a) * rng.random_range(3.0..=4.0) * 30.0),
                    false,
                );
            } else {
                brain.attack =
                    GTimer::from_seconds(rng.random_range(40.0..=50.0) / 30.0, TimerMode::Once);
                if brain.ratking_spawns > 24.0 && rng.random_range(0.0..4.0) < 3.0 {
                    brain.ratking_rage = true;
                    brain.walk = 0.0;
                    brain.attack = GTimer::from_seconds(1.0, TimerMode::Once);
                }
            }
        }

        // GML `Alarm_1`
        brain.attack.tick(dt);
        if !brain.attack.just_finished() {
            continue;
        }
        let mut next = rng.random_range(30.0..=40.0);
        if brain.ratking_rage {
            // GML `RatkingRage/Alarm_1:4-15`
            if dist < 100.0 {
                // GML `RatkingRage/Alarm_1:11` `meleedamage = 4`.
                enemy.touch_damage = 4;
                brain.walk = 40.0 + rng.random_range(0.0..10.0);
                let d =
                    glam::Vec2::from_angle(aim + rng.random_range(-10.0..=10.0_f32).to_radians());
                *headings.get_mut(&entity).unwrap() = d;
                vel.0 = d * 0.4 * 30.0;
                if player_pos.0.x < epos.x {
                    brain.right = -1.0;
                } else if player_pos.0.x > epos.x {
                    brain.right = 1.0;
                }
            }
        } else {
            // GML `Ratking/Alarm_1:4-30`
            next = rng.random_range(30.0..=40.0);
            brain.walk = 10.0 + rng.random_range(0.0..10.0);
            if has_line_of_sight(epos, player_pos.0, &mask) && rng.random_range(0.0..3.0) < 1.0 {
                brain.ammo = [3.0, 4.0, 5.0][rng.random_range(0..3)] as u8;
                brain.burst_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                brain.gunangle = aim;
                next = rng.random_range(30.0..=35.0);
                brain.walk = 40.0 + rng.random_range(0.0..10.0);
            }
            let away = aim + std::f32::consts::PI;
            let mut d = away + rng.random_range(-40.0..=40.0_f32).to_radians();
            vel.0 = glam::Vec2::ZERO;
            if dist < 64.0 {
                brain.walk = 40.0 + rng.random_range(0.0..10.0);
                let flip: f32 = if rng.random_range(0.0..3.0) < 2.0 {
                    180.0
                } else {
                    0.0
                };
                d = away + rng.random_range(-20.0..=20.0_f32).to_radians() + flip.to_radians();
            }
            set_heading(&mut headings, entity, d);
            vel.0 = glam::Vec2::from_angle(d) * 0.4 * 30.0;
            if player_pos.0.x < epos.x {
                brain.right = -1.0;
            } else if player_pos.0.x > epos.x {
                brain.right = 1.0;
            }
        }
        brain.attack = GTimer::from_seconds(next / 30.0, TimerMode::Once);
    }
}

/// Verbatim `objects/EliteInspector` law (`Alarm_1`/`Alarm_2`/`Other_10`):
/// freeze-gated control field that drags projectiles and pulls the player,
/// close-range baton dash-slash (`EnemySlash` damage 8), `PopoNade` lobs at the
/// last-seen position (5-grenade budget).
/// Register map: `brain.attack` = `alarm[1]`, `slash_delay` = `alarm[2]`
/// countdown (ticks), `ammo` = `grenades`, `burst_left` = `freeze`,
/// `gunangle` = `gunangle` (radians), `strafe_dir` = `control` (0/1);
/// `walk` keeps its GML name. `boss.target`-equivalent heading in `EnemyBrain`
/// is unused, so the move heading rides `Velocity` and last-seen a local map.
/// (Baton art, `wepangle` flips, enter/taunt sounds are out.)
#[allow(clippy::too_many_arguments)]
pub fn tick_elite_inspectors(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    player_q: Query<(&Pos, &FireCooldown), (With<Player>, Without<Enemy>)>,
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
    let Ok((player_pos, player_cd)) = player_q.single() else {
        return;
    };
    let can_shoot = player_cd.timer.is_finished();
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
            brain.freeze = 0.0;
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
        let freeze = brain.freeze;

        // GML `EliteInspector/Other_10:10-16`: `target.speed > 0 or
        // hp < max_hp`, then `if target.can_shoot freeze += 3` - the
        // elite gate is the positive form, live during the player's
        // reload window.
        if player_speed_sq > 0.001 || health.hp < health.max {
            brain.freeze += 1.0;
        }
        if can_shoot {
            brain.freeze += 3.0;
        }

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
            if rng.random::<f32>() < 0.5 && freeze > 40.0 {
                brain.strafe_dir = 1.0;
                let d = brain.attack.duration() + 10.0 / 30.0;
                brain.attack = GTimer::from_seconds(d, TimerMode::Once);
            }
            if los {
                brain.gunangle = aim;
                last_seen.insert(entity, player_pos.0);
                if rng.random::<f32>() < 2.0 / 3.0 && freeze > 40.0 && dist < 64.0 {
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
                    && freeze > 40.0
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

        apply_gml_friction(&mut vel.0, 0.4, dt);
        pos.0 += vel.0 * dt;
        clamp_to_arena(&mut pos.0, 10.0);
    }
}

/// Verbatim `objects/EliteShielder` law (`Alarm_1`/`Alarm_2`/`Other_10`):
/// freeze-gated 6-round `PopoPlasma` bursts (damage 8 at 1.5 px/tick), and the
/// `EliteShield` carry that teleports the shielder to a floor 120..300 px away.
/// Register map mirrors the Inspector (`brain.attack` = `alarm[1]`,
/// `slash_delay` = `alarm[2]` countdown, `ammo` = burst rounds,
/// `burst_left` = `freeze`).
/// (The shield's projectile-block field has no port equivalent; the teleport +
/// disappear poof are kept.)
#[allow(clippy::too_many_arguments)]
pub fn tick_elite_shielders(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    player_q: Query<(&Pos, &Velocity, &FireCooldown), (With<Player>, Without<Enemy>)>,
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
    let Ok((player_pos, player_v, player_cd)) = player_q.single() else {
        return;
    };
    let can_shoot = player_cd.timer.is_finished();
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
            brain.freeze = 20.0;
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
        let freeze = brain.freeze;

        // GML `EliteShielder/Other_10:12-16`: `target.speed > 0 && hp <
        // max_hp`, then `if target[$ "can_shoot"] freeze += 3` - the
        // `&&` gate and the positive `can_shoot` form.
        if player_v.0.length_squared() > 0.001 && health.hp < health.max {
            brain.freeze += 1.0;
        }
        if can_shoot {
            brain.freeze += 3.0;
        }

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
                if rng.random::<f32>() < 3.0 / 4.0 && freeze > 40.0 && dist < 150.0 {
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
                    if freeze < 40.0 {
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
            } else if freeze > 40.0 && rng.random::<f32>() < 0.25 {
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

        apply_gml_friction(&mut vel.0, 0.4, dt);
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

/// Expire telegraph markers (bonus port - bevy `tick_hit_warnings`
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

/// PopoShield follower tracking (bonus port - bevy
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
/// overlapping enemy, gated on `size <= other.size` and - once
/// `busycollisions` is false (`GameCont.loops > 3`) - only on frames where
/// `current_frame % 30` is 0. Each contributing pair also draws the two
/// `orandom(1)` jitters and caps speed at 16 px/frame.
fn separate(
    positions: &[(Entity, glam::Vec2, i32, f32)],
    entity: Entity,
    epos: glam::Vec2,
    vel: &mut glam::Vec2,
    kind: EnemyKind,
    loops: u32,
    current_frame: u64,
) {
    let size = crate::enemy_data::gml_size(kind);
    let radius = crate::enemy_data::enemy_def(kind).radius;
    let gated = loops > 3 && current_frame % 30 != 0;
    if gated {
        return;
    }
    let mut rng = rand::rng();
    for (other_entity, other, other_size, other_radius) in positions {
        // GML collision events only fire against *other* instances. The
        // snapshot includes self at distance 0, so it always passed the overlap
        // test and contributed a `normalize(-orandom)` impulse every step - with
        // nothing removing it between decides, every enemy random-walked up to
        // the 16 px/frame cap (480 px/s) and stayed.
        if *other_entity == entity {
            continue;
        }
        if size > *other_size {
            continue;
        }
        // GML fires on mask overlap. The port approximates each mask as a
        // circle, so the test is circle-circle overlap - a flat constant
        // separated a 16px Maggot pair ~2x too eagerly and pushed
        // wall-hugging enemies back into the wall.
        if epos.distance(*other) >= radius + *other_radius {
            continue;
        }
        let jitter = glam::Vec2::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
        let away = (epos - (*other + jitter)).normalize_or_zero();
        *vel += away * crate::SIM_HZ as f32;
    }
    *vel = vel.clamp_length_max(16.0 * crate::SIM_HZ as f32);
}

/// Corpse slide + expiry (bevy `enemies.rs:2552` parity: `Corpse` life ticks,
/// corpses drift with GML 0.4 friction, expiry despawns). `Transform.translation`
/// is [`Pos`] here; corpses without [`Velocity`] (player-kill drops) only tick
/// life.
/// Also slides `GroundPhysics` gibs/debris: GML gives them flat friction and
/// nothing else ticks them - without this they coast at full speed for their
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

#[cfg(test)]
mod gml_volley_cadence_tests {
    use super::*;

    /// GML `SnowBot/Other_10:14-16` caps the sled at 8 px/frame only while
    /// `meleedamage != 0`; at rest it is 3. The shared per-kind cap has to
    /// carry the resting value, because the 8 is applied by the sled's own
    /// motion-law arm off `touch_damage`.
    #[test]
    fn snowbot_resting_cap_is_three() {
        assert_eq!(gml_speed_cap_frames(EnemyKind::RobotGuard), 3.0);
    }

    /// GML `Salamander/Create_0:5` `wave = random(6.2)` and
    /// `HostileHorror/Create_0:19` `charge = 0` are both read by the volley
    /// code: `wave` picks the jet arc, `charge` the fan size. A brain left at
    /// zero would fire a straight 45-jet column and a 1-bullet spray.
    #[test]
    fn volley_registers_start_at_their_gml_values() {
        let mut world = bevy_ecs::world::World::new();
        let catalog = crate::setup::empty_anim_catalog();
        let mut commands = world.commands();
        for kind in [EnemyKind::Salamander, EnemyKind::HostileHorror] {
            spawn_enemy(
                &mut commands,
                &catalog,
                kind,
                glam::Vec2::ZERO,
                1.0,
                false,
                false,
                0,
            );
        }
        world.flush();

        let mut q = world.query::<(&Enemy, &EnemyBrain)>();
        let mut saw_salamander = false;
        for (enemy, brain) in q.iter(&world) {
            assert_eq!(brain.charge, 0.0, "{:?} charge", enemy.kind);
            if enemy.kind == EnemyKind::Salamander {
                saw_salamander = true;
                assert!(
                    (0.0..6.2).contains(&brain.wave),
                    "wave is random(6.2), got {}",
                    brain.wave
                );
            }
        }
        assert!(saw_salamander, "the salamander did not spawn");
    }

    /// GML `Crab/Alarm_2:13-14` re-arms the decide at `40 + random(10)` once
    /// the eight rounds are spent, against `Alarm_1`'s opening
    /// `10 + random(10)`. The dispatcher is the only place that knows the
    /// post-volley window, so pin both.
    #[test]
    fn crab_rest_after_a_volley_is_longer_than_before() {
        let mut rng = rand::rng();
        for _ in 0..64 {
            let before = 10.0 + rnd(&mut rng, 10.0);
            let after = 40.0 + rnd(&mut rng, 10.0);
            assert!((10.0..20.0).contains(&before), "{before}");
            assert!((40.0..50.0).contains(&after), "{after}");
        }
    }

    /// GML `HostileHorror/Other_10:22-31`: the fan is `round(charge + 1)`
    /// bullets paid for out of `raddrop`, and `charge` grows 0.1 a step - so
    /// the bill is quadratic. The whole 30-round burst must still fit inside
    /// the 90-rar pool (`Create_0:14`), or the tail of the burst would go out
    /// silent and the boss would under-perform.
    #[test]
    fn hostile_horror_fits_its_whole_burst_in_the_rad_pool() {
        let pool = enemy_def(EnemyKind::HostileHorror).rad_drop as f32;
        let mut charge = 0.0_f32;
        let mut spent = 0.0;
        let mut fired = 0;
        for _ in 0..30 {
            let cost = (charge + 1.0).round();
            if spent + cost > pool {
                break;
            }
            spent += cost;
            fired += cost as usize;
            charge += 0.1;
        }
        assert!(spent <= pool, "{spent} > {pool}");
        assert!(fired > 30, "the fan must widen past one bullet a step");
    }
}

#[cfg(test)]
mod gml_motion_cap_tests {
    use super::*;

    /// The walk law's `(impulse, cap)` pair must agree with GML for every
    /// object that reaches it. `cap_f` was computed and thrown away for the
    /// life of the port, so the `separate` push was free to drive these to its
    /// 16 px/frame clamp instead of GML's own ceiling.
    #[test]
    fn walk_law_caps_match_gml_other_10() {
        // (kind, GML `Other_10` `motion_add`, GML `if speed > N`)
        let table = [
            (EnemyKind::Rat, 0.8, 4.0),           // `Rat/Other_10:3-5,8`
            (EnemyKind::BigRat, 0.8, 4.0),        // port-scaled `Rat`
            (EnemyKind::FastRat, 0.8, 4.5),       // `FastRat/Other_10:3-5,8`
            (EnemyKind::BoneFish, 0.8, 4.0),      // `BoneFish/Other_10:3-5,8`
            (EnemyKind::Raven, 0.8, 3.5),         // `Raven/Other_10:3-5,7`
            (EnemyKind::Freak, 0.55, 4.0),        // `Freak/Other_10:5-7,9`
            (EnemyKind::ExploFreak, 0.6, 3.0),    // `ExploFreak/Other_10:6-8,10`
            (EnemyKind::RhinoFreak, 0.8, 1.0),    // `RhinoFreak/Other_10:6-8,10`
            (EnemyKind::PopoFreak, 0.55, 4.5),    // `PopoFreak/Other_10:5-9`
            (EnemyKind::Crab, 1.5, 4.5),          // `Crab/Other_10:3-5,7`
            (EnemyKind::Turtle, 1.0, 5.0),        // `Turtle/Other_10:3-8`
            (EnemyKind::Salamander, 2.0, 2.5),    // `Salamander/Other_10:3-5,7`
            (EnemyKind::Wolf, 1.0, 3.5),          // `Wolf/Other_10:3-5,16`
            (EnemyKind::HostileHorror, 0.8, 4.5), // `HostileHorror/Other_10:3-5,8`
            (EnemyKind::Spider, 2.0, 5.0),        // `Spider/Other_10:3-5,12` (maxspeed)
            (EnemyKind::InvSpider, 2.0, 5.0),     // `InvSpider/Other_10:3-5,12`
        ];
        for (kind, impulse, cap) in table {
            let (got_impulse, got_cap) = gml_walk_law(kind);
            assert!(
                (got_impulse - impulse).abs() < 1e-6,
                "{kind:?} impulse {got_impulse} != {impulse}"
            );
            assert!(
                (got_cap - cap).abs() < 1e-6,
                "{kind:?} cap {got_cap} != {cap}"
            );
            // The shared per-kind cap must agree too, or the unconditional
            // clamp at the end of `enemy_ai` fights the walk law.
            assert!(
                (gml_speed_cap_frames(kind) - cap).abs() < 1e-6,
                "{kind:?} gml_speed_cap disagrees with its walk law"
            );
        }
    }

    /// GML `PopoFreak/Other_10:5` does `walk -= 1`. Treating it as a
    /// never-decrementing object made every PopoFreak that had seen the player
    /// drift at its 4.5 px/frame cap permanently.
    #[test]
    fn popo_freak_spends_its_walk() {
        assert!(!gml_walk_never_decrements(EnemyKind::PopoFreak));
        // `Freak`/`ExploFreak`/`RhinoFreak` genuinely have no `walk -= 1`.
        for kind in [
            EnemyKind::Freak,
            EnemyKind::ExploFreak,
            EnemyKind::RhinoFreak,
        ] {
            assert!(gml_walk_never_decrements(kind), "{kind:?}");
        }
    }

    /// GML `RadMaggot/Other_10:3` applies `motion_add` with no `walk` gate, and
    /// its `Alarm_1` never arms one - so it needs the constant-drift law, not
    /// the walk law that gates on `walk > 0`.
    #[test]
    fn rad_maggot_drifts_without_a_walk_gate() {
        assert_eq!(gml_constant_drift(EnemyKind::RadMaggot), Some((0.6, 2.5)));
        assert_eq!(gml_constant_drift(EnemyKind::SuperFrog), Some((0.6, 2.5)));
    }

    /// Every kind the generic decide routes through must have a finite ceiling,
    /// or enemy-vs-enemy separation can fling it across the room.
    #[test]
    fn every_routed_kind_has_a_speed_ceiling() {
        for kind in [
            EnemyKind::Freak,
            EnemyKind::ExploFreak,
            EnemyKind::RhinoFreak,
            EnemyKind::PopoFreak,
            EnemyKind::Rat,
            EnemyKind::BigRat,
            EnemyKind::FastRat,
            EnemyKind::Wolf,
            EnemyKind::Raven,
            EnemyKind::Spider,
            EnemyKind::InvSpider,
            EnemyKind::Crab,
            EnemyKind::Turtle,
            EnemyKind::Salamander,
            EnemyKind::BoneFish,
            EnemyKind::RobotGuard,
            EnemyKind::HostileHorror,
            EnemyKind::RadMaggot,
            EnemyKind::SuperFrog,
            EnemyKind::LilHunter,
        ] {
            assert!(
                gml_speed_cap(kind).is_finite(),
                "{kind:?} has no speed ceiling"
            );
        }
    }
}

#[cfg(test)]
mod gml_rorand_tests {
    use super::*;

    #[test]
    fn orandom_samples_plus_minus_half() {
        let mut rng = rand::rng();
        for n in [1.0_f32, 3.0, 10.0, 15.0, 16.0] {
            for _ in 0..256 {
                let v = ornd(&mut rng, n);
                assert!(
                    (-n * 0.5..=n * 0.5).contains(&v),
                    "orandom({n}) sampled {v}, outside -n/2..+n/2"
                );
            }
        }
    }
}
