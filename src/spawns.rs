//! Combat spawn helpers. Ported from nt's `game/combat.rs` spawners plus
//! prop-death damage glue.
//! Render split: `SpriteAnim` when the catalog has the strip, else bare with
//! renderer fallback (split/plasma art keys off `ProjectileTyp`, sentries and
//! hazards off their markers). Bevy pop-ins and tint flashes dropped as render
//! juice.

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_anim::AnimCatalog;

use crate::anim::SpriteAnim;
use crate::audio::{AudioCue, GameAudio};
use crate::comps_a::{
    BouncesLeft, DamageSource, GameCleanup, LevelCleanup, NextHurt, PlasmaSize, Projectile,
    ProjectileFade, ProjectileFriction, ProjectileTyp, ShellBonus, ShellWallBounce,
    SpawnHazardOnDeath, SplitOnDeath, Team, Velocity,
};
use crate::comps_b::SpecialPropDeath;
use crate::comps_b::{
    CustomExplosion, DeploysSentry, ExplosionVisual, NativeExplosionKind, PlasmaBurst, PortalClear,
    Prop, PropNestMarkers, PropSprites, SecretEntrance, SentryTurret,
    SpawnsWeaponPickup,
};
use crate::data::{EnemyKind, HazardDef, SplitDef};
use crate::environment::PropDeathEffect;
use crate::msg::Queue;
use crate::pickups::{random_weapon, spawn_pickup, spawn_rad, spawn_rad_burst};
use crate::projectile_math::split_directions;
use crate::secrets::SecretTriggers;
use crate::spatial::Pos;
use crate::time::{GTimer, TimerMode};

use super::combat::{Explosion, LingeringBlast, queue_enemy_spawn};

/// Timed explosion entity (sticky blasts, barrels, booms).
pub fn spawn_explosion_with_source_radius(
    commands: &mut Commands,
    pos: glam::Vec2,
    damage: i32,
    source: Option<DamageSource>,
    radius: f32,
    team: Team,
    hits_player: bool,
) {
    spawn_explosion_with_source_radius_kind(
        commands,
        pos,
        damage,
        source,
        radius,
        team,
        hits_player,
        None,
    );
}

pub fn spawn_explosion_with_source_radius_kind(
    commands: &mut Commands,
    pos: glam::Vec2,
    damage: i32,
    source: Option<DamageSource>,
    radius: f32,
    team: Team,
    hits_player: bool,
    visual: Option<NativeExplosionKind>,
) -> Entity {
    let mut entity = commands.spawn((
        GameCleanup,
        LevelCleanup,
        Explosion {
            timer: GTimer::from_seconds(0.05, TimerMode::Once),
            radius,
            damage,
            team,
            hits_player,
            source,
        },
        LingeringBlast {
            duration: GTimer::from_seconds(0.75, TimerMode::Once),
            tick: GTimer::from_seconds(1.0 / 30.0, TimerMode::Repeating),
            hit: Vec::new(),
        },
        Pos(pos),
    ));
    if let Some(kind) = visual {
        entity.insert(ExplosionVisual(kind));
    }
    entity.id()
}

/// Hazard cloud from area definitions.
pub fn spawn_hazard_cloud(commands: &mut Commands, pos: glam::Vec2, team: Team, spec: HazardDef) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        team,
        crate::comps_b::HazardCloud {
            kind: spec.kind,
            radius: spec.radius,
            damage: spec.damage,
            timer: GTimer::from_seconds(spec.duration, TimerMode::Once),
            tick: GTimer::from_seconds(spec.tick, TimerMode::Repeating),
        },
        Pos(pos),
    ));
}

/// Child object a death-split table spawns. GML picks the child per
/// parent's `Destroy_0`; the port keys the family off the
/// `(team, damage, pellets)` signature each table is written with in
/// `weapon_runtime::set_split` / `enemies::fire_enemy_flak`, because
/// `SplitDef` (in `data.rs`) has no child-object field.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SplitChild {
    /// GML `ClusterNade/Destroy_0.gml:1-13` - `SmallGrenade` on
    /// `random_angle` at `random_range(3, 5)` px/step, each inheriting
    /// `motion_add(other.direction, 2)`; `friction = 0.4`;
    /// `alarm[0] = irandom_range(10, 20)`; `Collision_Wall` destroys it.
    SmallGrenade,
    /// GML `FlakBullet/Destroy_0.gml:1-7` - `Bullet2` on `random_angle`
    /// at `8 + random(8)` px/step; bounces with cap 16 / decay 0.95.
    Bullet2,
    /// GML `SuperFlakBullet/Destroy_0.gml:3-11` - a 5-bullet ring
    /// (`_ang += 72`) of `FlakBullet` at `random_range(12, 16)` px/step;
    /// `Collision_Wall` destroys it.
    FlakBullet,
    /// GML `EnemyFlak/Destroy_0.gml:1-3` - `EnemyBullet3` on
    /// `random_angle` at `random_range(8, 12)` px/step; bounces with
    /// `min(18, speed * 0.8 + wallbounce)` and `wallbounce *= 0.9`.
    EnemyBullet3,
}

fn split_child(team: Team, split: SplitDef) -> SplitChild {
    // Death-crown bumps the cluster's count 8 -> 9, hence the range.
    match (team, split.damage) {
        (Team::Enemy, 1) => SplitChild::EnemyBullet3,
        (Team::Player, 5) if split.pellets >= 8 => SplitChild::SmallGrenade,
        (Team::Player, 8) => SplitChild::FlakBullet,
        _ => SplitChild::Bullet2,
    }
}

/// Split burst (shotgun fans). Art keys off `ProjectileTyp(1)`
/// renderer-side.
pub fn spawn_split_projectiles(
    commands: &mut Commands,
    pos: glam::Vec2,
    team: Team,
    split: SplitDef,
    source: Option<DamageSource>,
    base_dir: glam::Vec2,
) {
    let child = split_child(team, split);
    let mut rng = rand::rng();
    let samples: Vec<f32> = (0..split.pellets)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    // GML `SuperFlakBullet/Destroy_0.gml:3-11` walks a fixed 72-degree
    // ring from a random start angle instead of scattering.
    let ring_base = rng.random_range(0.0..std::f32::consts::TAU);
    let mut dirs = split_directions(base_dir, split.pellets, split.spread, &samples);
    if child == SplitChild::FlakBullet {
        let step = std::f32::consts::TAU / split.pellets.max(1) as f32;
        for (i, dir) in dirs.iter_mut().enumerate() {
            let ang = ring_base + i as f32 * step;
            *dir = glam::Vec2::new(ang.cos(), ang.sin());
        }
    }

    // GML `ClusterNade/Destroy_0.gml:10` adds `motion_add(other.direction, 2)`
    // - 2 px/step along the parent rocket's heading - to every child.
    let inherited = match child {
        SplitChild::SmallGrenade => base_dir * 2.0 * 30.0,
        _ => glam::Vec2::ZERO,
    };

    for dir in dirs {
        match child {
            SplitChild::SmallGrenade => {
                let speed = rng.random_range(3.0..5.0) * 30.0;
                let life = rng.random_range(10..=20) as f32 / 30.0;
                let scatter =
                    glam::Vec2::new(rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0));
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    team,
                    Projectile {
                        damage: split.damage,
                        life: GTimer::from_seconds(life, TimerMode::Once),
                        radius: split.radius,
                        knockback: split.knockback,
                        explosive: false,
                        source,
                    },
                    Velocity(dir * speed + inherited),
                    ProjectileFriction(0.4),
                    ProjectileTyp(1),
                    Pos(pos + scatter),
                ));
            }
            SplitChild::FlakBullet => {
                let speed = rng.random_range(12.0..16.0) * 30.0;
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    team,
                    Projectile {
                        damage: split.damage,
                        life: GTimer::from_seconds(split.lifetime, TimerMode::Once),
                        radius: split.radius,
                        knockback: split.knockback,
                        explosive: false,
                        source,
                    },
                    Velocity(dir * speed),
                    ProjectileFriction(0.4),
                    ProjectileTyp(1),
                    Pos(pos),
                ));
            }
            SplitChild::EnemyBullet3 => {
                let speed = rng.random_range(8.0..12.0) * 30.0;
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    team,
                    Projectile {
                        damage: split.damage,
                        life: GTimer::from_seconds(split.lifetime, TimerMode::Once),
                        radius: split.radius,
                        knockback: split.knockback,
                        explosive: false,
                        source,
                    },
                    Velocity(dir * speed),
                    ProjectileFriction(0.6),
                    BouncesLeft(255),
                    ShellWallBounce {
                        add: 0.0,
                        cap: 540.0,
                        decay: 0.9,
                        rearm: None,
                    },
                    ShellBonus {
                        timer: GTimer::from_seconds(2.0 / 30.0, TimerMode::Once),
                        bonus: 1,
                    },
                    ProjectileTyp(1),
                    ProjectileFade("images/sprEBullet3Disappear.png"),
                    Pos(pos),
                ));
            }
            SplitChild::Bullet2 => {
                let speed = rng.random_range(8.0..16.0) * 30.0;
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    team,
                    Projectile {
                        damage: split.damage,
                        life: GTimer::from_seconds(split.lifetime, TimerMode::Once),
                        radius: split.radius,
                        knockback: split.knockback,
                        explosive: false,
                        source,
                    },
                    Velocity(dir * speed),
                    ProjectileFriction(0.6),
                    BouncesLeft(255),
                    ShellWallBounce {
                        add: 0.0,
                        cap: 480.0,
                        decay: 0.95,
                        rearm: Some((0.0, 1)),
                    },
                    ShellBonus {
                        timer: GTimer::from_seconds(2.0 / 30.0, TimerMode::Once),
                        bonus: 1,
                    },
                    ProjectileTyp(1),
                    ProjectileFade(match team {
                        Team::Enemy => "images/sprEBullet3Disappear.png",
                        _ => "images/sprBullet2Disappear.png",
                    }),
                    Pos(pos),
                ));
            }
        }
    }
}

/// Plasma fan children. Art keys off `ProjectileTyp(2)` + size
/// threshold renderer-side (big/small split at 14px, bevy parity).
pub fn spawn_plasma_children(
    commands: &mut Commands,
    pos: glam::Vec2,
    team: Team,
    plasma: PlasmaBurst,
    source: Option<DamageSource>,
    base_dir: glam::Vec2,
) {
    let base_angle = if base_dir.length_squared() > 0.0 {
        base_dir.y.atan2(base_dir.x)
    } else {
        0.0
    };

    for i in 0..plasma.pellets.max(1) {
        let t = i as f32 / plasma.pellets.max(1) as f32;
        let angle = base_angle + t * std::f32::consts::TAU;
        let dir = glam::Vec2::new(angle.cos(), angle.sin());

        commands.spawn((
            GameCleanup,
            LevelCleanup,
            team,
            Projectile {
                damage: plasma.damage,
                life: GTimer::from_seconds(plasma.lifetime, TimerMode::Once),
                radius: plasma.radius,
                knockback: plasma.knockback,
                explosive: false,
                source,
            },
            Velocity(dir * plasma.speed),
            PlasmaSize((plasma.size.x + plasma.size.y) * 0.5),
            ProjectileTyp(2),
            Pos(pos),
        ));
    }
}

/// Sentry turret marker (art resolves renderer-side).
pub fn spawn_sentry_turret(commands: &mut Commands, pos: glam::Vec2, spec: DeploysSentry) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        Team::Player,
        SentryTurret {
            life: Some(GTimer::from_seconds(spec.life, TimerMode::Once)),
            fire: GTimer::from_seconds(spec.fire_interval, TimerMode::Repeating),
            first_shot: 0.0,
            ammo: i32::MAX,
            range: spec.range,
            projectile_speed: spec.projectile_speed,
            projectile_damage: spec.projectile_damage,
        },
        Pos(pos),
    ));
}

/// GML `SentryGun/Create_0.gml` + `Alarm_0.gml` verbatim: `max_hp = 10`,
/// `ammo = 24` spent one bullet per 5-frame alarm, `alarm[0] = 30` on the
/// first pass, `friction = 0.2`, and `scrFire.gml:367` gives it
/// `motion_add(_gunangle, 6)`. Bullet: `Bullet1` at
/// `motion_add(gunangle + random(12) - 6, 16)` - 16 px/step = 480 px/s,
/// damage 3.
pub fn spawn_sentry_gun(commands: &mut Commands, pos: glam::Vec2, dir: glam::Vec2, team: Team) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        team,
        SentryTurret {
            // GML has no lifetime here: the body ends on `ammo <= 0`
            // (`Alarm_0.gml:56`) or `hp <= 0` (`Step_0.gml:1`).
            life: None,
            fire: GTimer::from_seconds(5.0 / 30.0, TimerMode::Repeating),
            // `alarm[0] = 30` before the 5-step cadence takes over.
            first_shot: 30.0,
            ammo: 24,
            // GML has no range gate: the alarm fires at any enemy with a
            // clear `collision_line` (`Alarm_0.gml:19`).
            range: f32::INFINITY,
            projectile_speed: 16.0 * 30.0,
            projectile_damage: 3,
        },
        crate::comps_a::Health {
            hp: 10,
            max: 10,
            invuln: GTimer::disarmed(),
        },
        crate::comps_a::Hitbox { radius: 12.0 },
        crate::comps_b::NativeMotion {
            velocity: dir.normalize_or_zero() * 6.0 * 30.0,
            friction: 0.2,
            radius: 12.0,
            wall: crate::comps_b::NativeWallMotion::Bounce,
            tick: 0,
        },
        // GML `SentryGun/Create_0.gml:13,19`: `gunangle = random_angle`,
        // `right = choose(1, -1)`. The renderer orients by `NativeAngle`.
        crate::comps_b::GmlImage::new("images/sprSentryGun.png", 1, 0.0),
        crate::comps_b::NativeAngle(0.0),
        crate::comps_b::NativeFlip(rand::rng().random_bool(0.5)),
        crate::comps_b::NativeDepth(-1.0),
        Pos(pos),
    ));
}

pub fn spawn_weapon_pickup_from_projectile(
    commands: &mut Commands,
    catalog: &AnimCatalog,
    pos: glam::Vec2,
    spec: SpawnsWeaponPickup,
    decide: Option<&crate::decide_wep::DecideCtx>,
) {
    if pos.x.abs() > crate::comps_a::ARENA_W / 2.0 + 32.0
        || pos.y.abs() > crate::comps_a::ARENA_H / 2.0 + 32.0
    {
        return;
    }

    let weapon = spec.weapon.unwrap_or_else(|| match decide {
        // GML `wep_gun_gun`: `scrDecideWep(10)` at the muzzle.
        Some(ctx) => crate::decide_wep::decide_wep(&mut rand::rng(), ctx, spec.decide_extra, false),
        None => random_weapon(&mut rand::rng()),
    });
    spawn_pickup(
        commands,
        catalog,
        crate::comps_b::PickupKind::Weapon(weapon),
        pos,
        0,
        false,
    );
}

/// GML scrBulletHitFX: oneshot fade at pos (bevy shape: oneshot 0.3 s
/// when stripped, static 0.2 s otherwise), oriented by `angle`
/// renderer-side via [`FxAngle`].
pub fn spawn_bullet_fade_fx(
    commands: &mut Commands,
    catalog: &AnimCatalog,
    pos: glam::Vec2,
    angle: f32,
    fade_path: &'static str,
) {
    if let Some(def) = catalog.def(fade_path) {
        let mut ec = commands.spawn((
            GameCleanup,
            LevelCleanup,
            Pos(pos),
            crate::comps_b::FxAngle(angle),
        ));
        ec.insert(SpriteAnim::oneshot(fade_path, def));
        ec.insert(crate::comps_b::PickupLifetime {
            timer: GTimer::from_seconds(0.3, TimerMode::Once),
        });
    } else {
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            Pos(pos),
            crate::comps_b::StaticFx { path: fade_path },
            crate::comps_b::FxAngle(angle),
            crate::comps_b::PickupLifetime {
                timer: GTimer::from_seconds(0.2, TimerMode::Once),
            },
        ));
    }
}

/// Removal cascade: fade fx, sentry, explosions (incl. Jock bonus),
/// hazards, splits, pickup spec, plasma children. Bevy parity.
#[allow(clippy::too_many_arguments)]
pub fn on_projectile_removed(
    commands: &mut Commands,
    catalog: &AnimCatalog,
    pos: glam::Vec2,
    team: Team,
    source: Option<DamageSource>,
    hazard: Option<SpawnHazardOnDeath>,
    split: Option<SplitOnDeath>,
    base_dir: glam::Vec2,
    explosive: bool,
    damage: i32,
    custom_explosion: Option<CustomExplosion>,
    deploys_sentry: Option<DeploysSentry>,
    spawn_pickup_spec: Option<SpawnsWeaponPickup>,
    decide: Option<&crate::decide_wep::DecideCtx>,
    plasma_burst: Option<PlasmaBurst>,
    fade: Option<ProjectileFade>,
    already_faded: bool,
) {
    if let Some(f) = fade
        && !already_faded
    {
        let angle = if base_dir.length_squared() > 0.0 {
            base_dir.y.atan2(base_dir.x)
        } else {
            0.0
        };
        spawn_bullet_fade_fx(commands, catalog, pos, angle, f.0);
    }
    if let Some(spec) = deploys_sentry {
        spawn_sentry_turret(commands, pos, spec);
    }

    if explosive {
        let visual = custom_explosion.and_then(|c| c.visual);
        // GML `Explosion/Create_0.gml:3` sets `damage = 5` and
        // `SmallExplosion` inherits it, so a rocket's blast is 5 even
        // though `Rocket/Create_0.gml:4` gives the projectile 20 - the
        // projectile's `damage` only lands on a direct hit. `GreenExplosion`
        // is 12 (`GreenExplosion/Create_0.gml:3`).
        let blast_damage = match visual {
            Some(NativeExplosionKind::Green) => 12,
            Some(NativeExplosionKind::Meat) | Some(NativeExplosionKind::Popo) => damage,
            _ if team == Team::Player => 5,
            _ => damage,
        };
        let (radius, count, spread) = custom_explosion
            .map(|c| (c.radius, c.count.max(1), c.spread))
            .unwrap_or((32.0, 1, 0.0));
        if count <= 1 {
            spawn_explosion_with_source_radius_kind(
                commands,
                pos,
                blast_damage,
                source,
                radius,
                team,
                true,
                visual,
            );
        } else {
            let ang0 = rand::rng().random_range(0.0..std::f32::consts::TAU);
            for k in 0..count {
                let ang = ang0 + k as f32 * std::f32::consts::TAU / count as f32;
                let off = glam::Vec2::new(ang.cos(), ang.sin()) * spread;
                spawn_explosion_with_source_radius_kind(
                    commands,
                    pos + off,
                    blast_damage,
                    source,
                    radius,
                    team,
                    true,
                    visual,
                );
            }
        }
        if source.is_some_and(|s| s.enemy_kind == Some(EnemyKind::Jock)) {
            let off = if base_dir.length_squared() > 0.0 {
                base_dir.normalize_or_zero() * 24.0
            } else {
                glam::Vec2::new(24.0, 0.0)
            };
            spawn_explosion_with_source_radius(
                commands,
                pos - off,
                blast_damage,
                source,
                radius,
                team,
                true,
            );
        }
    }

    if let Some(SpawnHazardOnDeath(spec)) = hazard {
        spawn_hazard_cloud(commands, pos, team, spec);
    }

    if let Some(SplitOnDeath(spec)) = split {
        spawn_split_projectiles(commands, pos, team, spec, source, base_dir);
    }

    if let Some(spec) = spawn_pickup_spec {
        spawn_weapon_pickup_from_projectile(commands, catalog, pos, spec, decide);
    }

    if let Some(plasma) = plasma_burst {
        spawn_plasma_children(commands, pos, team, plasma, source, base_dir);
    }

    // GML `MomProjectile/Destroy_0`: 25 scattered `ToxicGas` clouds plus
    // a `PortalClear`. FrogQueen-kind projectiles are exclusively the
    // gas mortar, so the source kind is a reliable tag.
    if source.is_some_and(|s| s.enemy_kind == Some(EnemyKind::FrogQueen)) {
        let mut rng = rand::rng();
        for _ in 0..25 {
            let a = rng.random_range(0.0..std::f32::consts::TAU);
            let d = rng.random_range(0.0..=48.0);
            let off = glam::Vec2::from_angle(a) * d;
            let speed = rng.random_range(0.2..1.7) * 30.0;
            let mut gas = crate::comps_b::ToxicGasState::new();
            gas.grow_speed = 0.003 + rng.random_range(0.0..0.002);
            gas.rot =
                (1.0 + rng.random_range(0.0..=3.0)) * if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            crate::enemies::spawn_toxic_gas(
                commands,
                pos + off,
                glam::Vec2::from_angle(a) * speed,
                gas,
            );
        }
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            PortalClear {
                timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                scale: 1.0,
            },
            Pos(pos),
        ));
    }
}

/// GML `scripts/scrDrop/scrDrop.gml:7-9`: the roll is relative to
/// `instance_nearest(x, y, Player)` and bails when there is none, so the
/// player context is optional in the port. `decide` is the shared
/// `scrDecideWep` cache (only read on the weapon branch).
pub struct DropCtx<'a> {
    pub player: &'a crate::comps_a::Player,
    pub inv: &'a crate::comps_a::Inventory,
    pub health: &'a crate::comps_a::Health,
    pub decide: Option<&'a crate::decide_wep::DecideCtx>,
}

/// Damage a destructible prop. Signature adapted: audio cues queue instead of
/// audio commands.
/// No `scrDrop` context: `Cocoon/Destroy_0.gml:2` then has no player to weigh
/// against and drops nothing (GML's own `if (_player == noone) exit`).
#[allow(clippy::too_many_arguments)]
pub fn damage_destructible_prop(
    commands: &mut Commands,
    catalog: &AnimCatalog,
    props: &mut Query<
        (
            Entity,
            &mut Prop,
            &Pos,
            Option<&PropDeathEffect>,
            Option<&PropSprites>,
            Option<&SpecialPropDeath>,
            Option<&mut NextHurt>,
        ),
        With<Prop>,
    >,
    entrances: &Query<&SecretEntrance>,
    nests: &Query<&PropNestMarkers, With<Prop>>,
    secrets: &mut SecretTriggers,
    audio: &GameAudio,
    cues: &mut Queue<AudioCue>,
    particles_on: bool,
    prop_e: Entity,
    center: glam::Vec2,
    damage: i32,
    source: Option<DamageSource>,
    nexthurt_window: Option<u64>,
    loops: u32,
    hasted: bool,
) {
    damage_destructible_prop_ctx(
        commands,
        catalog,
        props,
        entrances,
        nests,
        secrets,
        audio,
        cues,
        particles_on,
        prop_e,
        center,
        damage,
        source,
        nexthurt_window,
        loops,
        hasted,
        None,
    );
}

/// [`damage_destructible_prop`] with the GML `scrDrop` player context
/// (`Cocoon/Destroy_0.gml:2`), so those props roll against real ammo need.
#[allow(clippy::too_many_arguments)]
pub fn damage_destructible_prop_ctx(
    commands: &mut Commands,
    catalog: &AnimCatalog,
    props: &mut Query<
        (
            Entity,
            &mut Prop,
            &Pos,
            Option<&PropDeathEffect>,
            Option<&PropSprites>,
            Option<&SpecialPropDeath>,
            Option<&mut NextHurt>,
        ),
        With<Prop>,
    >,
    entrances: &Query<&SecretEntrance>,
    nests: &Query<&PropNestMarkers, With<Prop>>,
    secrets: &mut SecretTriggers,
    audio: &GameAudio,
    cues: &mut Queue<AudioCue>,
    particles_on: bool,
    prop_e: Entity,
    center: glam::Vec2,
    damage: i32,
    source: Option<DamageSource>,
    nexthurt_window: Option<u64>,
    loops: u32,
    hasted: bool,
    drop: Option<DropCtx<'_>>,
) {
    let mut dead = false;
    let mut legacy_explosive = false;
    let mut death_copy: Option<PropDeathEffect> = None;
    let mut sprites_copy: Option<PropSprites> = None;
    if let Ok((_, mut prop, _, de, sprites, special, nexthurt)) = props.get_mut(prop_e) {
        let hp_before = prop.hp;
        prop.hp -= damage.max(1);
        if let Some(window) = nexthurt_window
            && let Some(mut nh) = nexthurt
        {
            nh.0 = window;
        }
        // `try_despawn` is deferred, so a second killer in the same frame
        // still matches this entity; GML destroys an instance once.
        if prop.hp <= 0 && hp_before > 0 {
            // `VaultStatue` and `VenuzTV` own their death: both raise other
            // objects, which the generic path has no queries to do.
            if special.is_none() {
                dead = true;
            }
        }
        audio.play_hit(cues);
        legacy_explosive = prop.explosive;
        death_copy = de.copied();
        sprites_copy = sprites.copied();
    }
    if !dead {
        return;
    }
    if let Some(ps) = sprites_copy {
        crate::environment::spawn_prop_corpse(commands, catalog, center, &ps);
    }
    crate::environment::spawn_prop_death_effect(
        commands,
        catalog,
        particles_on,
        center,
        death_copy,
        legacy_explosive,
        source,
        loops,
        hasted,
        audio,
        cues,
    );
    commands.entity(prop_e).try_despawn();
    if let Ok(entrance) = entrances.get(prop_e) {
        secrets.queue(entrance.target);
    }
    let nest_flags = nests.get(prop_e).copied().unwrap_or_default();
    if nest_flags.snowman {
        let mut rng = rand::rng();
        for _ in 0..3 {
            queue_enemy_spawn(
                &mut *commands,
                EnemyKind::Bandit,
                center + glam::Vec2::new(rng.random_range(-4.0..4.0), rng.random_range(-4.0..4.0)),
                1.0,
                loops,
            );
        }
        for _ in 0..6 {
            spawn_rad(commands, catalog, center, 1);
        }
    }
    if nest_flags.cocoon {
        let mut rng = rand::rng();
        // GML `Cocoon/Destroy_0.gml:1-2` verbatim: `random(3) < 1` spawns
        // a `Gator`, otherwise `scrDrop(30, 0)` - a 30% drop that weighs
        // the player's ammo need and can pay a `HealthChest`.
        if rng.random_range(0..3i32) < 1 {
            queue_enemy_spawn(&mut *commands, EnemyKind::Gator, center, 1.0, loops);
        } else if let Some(drop) = drop {
            crate::pickups::maybe_spawn_drop(
                commands,
                catalog,
                center,
                30,
                0,
                drop.player,
                drop.inv,
                drop.health,
                loops,
                drop.decide,
            );
        }
    }
    if nest_flags.mutant_tube {
        let mut rng = rand::rng();
        for _ in 0..8 {
            queue_enemy_spawn(
                &mut *commands,
                EnemyKind::Freak,
                center + glam::Vec2::new(rng.random_range(-4.0..4.0), rng.random_range(-4.0..4.0)),
                1.0,
                loops,
            );
        }
    }
    if nest_flags.soda_machine {
        let mut rng = rand::rng();
        for _ in 0..20 {
            let a = rng.random_range(0.0..std::f32::consts::TAU);
            let d = glam::Vec2::new(a.cos(), a.sin());
            let s = rng.random_range(3.0..7.0) * 30.0;
            spawn_pickup(
                commands,
                catalog,
                crate::comps_b::PickupKind::Medkit(1),
                center
                    + glam::Vec2::new(rng.random_range(-4.0..4.0), rng.random_range(0.0..16.0))
                    + d * s * 0.05,
                loops,
                false,
            );
        }
    }
    if nest_flags.pizza_box && rand::rng().random_range(0.0..1.0) < 0.2 {
        spawn_pickup(
            commands,
            catalog,
            crate::comps_b::PickupKind::Medkit(2),
            center,
            loops,
            false,
        );
    }
    if nest_flags.small_gen {
        // GML `SmallGenerator/Create_0.gml:13` `raddrop = 5`, paid by
        // `prop/Destroy_0.gml:12`.
        spawn_rad_burst(commands, catalog, center, 5);
    }
}
