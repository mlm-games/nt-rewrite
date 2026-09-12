//! Combat spawn helpers. Ported from nt's `game/combat.rs` spawners
//! plus prop-death damage glue.
//!
//! Render split: entities spawn with `SpriteAnim` when the catalog has
//! the strip; otherwise bare with renderer fallback (split/plasma art
//! keys off `ProjectileTyp`, sentries/hazards off their markers).
//! Juice pop-ins and bevy tint flashes are render juice (skipped).

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_anim::AnimCatalog;

use crate::anim::SpriteAnim;
use crate::audio::{AudioCue, GameAudio};
use crate::comps_a::{
    DamageSource, GameCleanup, LevelCleanup, NextHurt, PlasmaSize, Projectile, ProjectileFade,
    ProjectileFriction, ProjectileTyp, ShellBonus, ShellWallBounce, SpawnHazardOnDeath,
    SplitOnDeath, Team, Velocity,
};
use crate::comps_b::{
    CustomExplosion, DeploysSentry, GoldBarrelDrop, HazardCloud, PlasmaBurst, PortalClear, Prop,
    PropSprites, RadChestContainer, SecretEntrance, SentryTurret, SnowmanAmbush,
    SpawnsWeaponPickup,
};
use crate::data::{EnemyKind, HazardDef, HazardKind, SplitDef};
use crate::environment::PropDeathEffect;
use crate::msg::Queue;
use crate::pickups::{random_gold_weapon_fallback, random_weapon, spawn_pickup, spawn_rad};
use crate::projectile_math::split_directions;
use crate::secrets::SecretTriggers;
use crate::spatial::Pos;
use crate::time::{GTimer, TimerMode};

use super::combat::{Explosion, LingeringBlast, PendingEnemySpawn};

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
    commands.spawn((
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
    let mut rng = rand::rng();
    let samples: Vec<f32> = (0..split.pellets)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();

    for dir in split_directions(base_dir, split.pellets, split.spread, &samples) {
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
            Velocity(dir * split.speed),
            ProjectileFriction(0.6),
            crate::comps_a::BouncesLeft(255),
            ShellWallBounce {
                add: 0.0,
                cap: 480.0,
                decay: 0.95,
                rearm: None,
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
            life: GTimer::from_seconds(spec.life, TimerMode::Once),
            fire: GTimer::from_seconds(spec.fire_interval, TimerMode::Repeating),
            range: spec.range,
            projectile_speed: spec.projectile_speed,
            projectile_damage: spec.projectile_damage,
        },
        Pos(pos),
    ));
}

pub fn spawn_weapon_pickup_from_projectile(
    commands: &mut Commands,
    catalog: &AnimCatalog,
    pos: glam::Vec2,
    spec: SpawnsWeaponPickup,
) {
    if pos.x.abs() > crate::comps_a::ARENA_W / 2.0 + 32.0
        || pos.y.abs() > crate::comps_a::ARENA_H / 2.0 + 32.0
    {
        return;
    }

    let weapon = spec
        .weapon
        .unwrap_or_else(|| random_weapon(&mut rand::rng()));
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
        let (radius, count, spread) = custom_explosion
            .map(|c| (c.radius, c.count.max(1), c.spread))
            .unwrap_or((32.0, 1, 0.0));
        if count <= 1 {
            spawn_explosion_with_source_radius(commands, pos, damage, source, radius, team, true);
        } else {
            let ang0 = rand::rng().random_range(0.0..std::f32::consts::TAU);
            for k in 0..count {
                let ang = ang0 + k as f32 * std::f32::consts::TAU / count as f32;
                let off = glam::Vec2::new(ang.cos(), ang.sin()) * spread;
                spawn_explosion_with_source_radius(
                    commands,
                    pos + off,
                    damage,
                    source,
                    radius,
                    team,
                    true,
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
                damage,
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
        spawn_weapon_pickup_from_projectile(commands, catalog, pos, spec);
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
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                team,
                HazardCloud {
                    kind: HazardKind::Toxic,
                    radius: 16.0,
                    damage: 1,
                    timer: GTimer::from_seconds(6.0, TimerMode::Once),
                    tick: GTimer::from_seconds(0.5, TimerMode::Repeating),
                },
                Pos(pos + off),
            ));
        }
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            PortalClear {
                timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
            },
            Pos(pos),
        ));
    }
}

/// Damage a destructible prop; on death run its corpse/effect chain,
/// trigger secrets, spawn ambushes/drops, and despawn. Signature
/// adapted: audio cues queue instead of audio commands.
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
            Option<&mut NextHurt>,
        ),
        With<Prop>,
    >,
    entrances: &Query<&SecretEntrance>,
    snowmen: &Query<&SnowmanAmbush>,
    gold_barrels: &Query<&GoldBarrelDrop>,
    rad_chests: &Query<&RadChestContainer>,
    secrets: &mut SecretTriggers,
    audio: &GameAudio,
    cues: &mut Queue<AudioCue>,
    prop_e: Entity,
    center: glam::Vec2,
    damage: i32,
    source: Option<DamageSource>,
    nexthurt_window: Option<u64>,
    loops: u32,
) {
    let mut dead = false;
    let mut legacy_explosive = false;
    let mut death_copy: Option<PropDeathEffect> = None;
    let mut sprites_copy: Option<PropSprites> = None;
    if let Ok((_, mut prop, _, de, sprites, nexthurt)) = props.get_mut(prop_e) {
        prop.hp -= damage.max(1);
        if let Some(window) = nexthurt_window
            && let Some(mut nh) = nexthurt
        {
            nh.0 = window;
        }
        if prop.hp <= 0 {
            dead = true;
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
        center,
        death_copy,
        legacy_explosive,
        source,
    );
    commands.entity(prop_e).try_despawn();
    if let Ok(entrance) = entrances.get(prop_e) {
        secrets.queue(entrance.target);
    }
    if snowmen.get(prop_e).is_ok() {
        let mut rng = rand::rng();
        for _ in 0..3 {
            commands.spawn(PendingEnemySpawn {
                kind: EnemyKind::Bandit,
                pos: center
                    + glam::Vec2::new(rng.random_range(-4.0..4.0), rng.random_range(-4.0..4.0)),
                difficulty: 1.0,
                loops,
            });
        }
        for _ in 0..6 {
            spawn_rad(commands, catalog, center, 1);
        }
    }
    if gold_barrels.get(prop_e).is_ok() {
        let weapon = random_gold_weapon_fallback(&mut rand::rng());
        spawn_pickup(
            commands,
            catalog,
            crate::comps_b::PickupKind::Weapon(weapon),
            center + glam::Vec2::new(0.0, -14.0),
            0,
            false,
        );
    }
    if rad_chests.get(prop_e).is_ok() {
        let mut rng = rand::rng();
        for _ in 0..25 {
            let ang = rng.random_range(0.0..std::f32::consts::TAU);
            let d = rng.random_range(6.0..26.0);
            spawn_pickup(
                commands,
                catalog,
                crate::comps_b::PickupKind::Rad(1),
                center + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                0,
                false,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comps_b::HazardCloud;

    #[test]
    fn explosion_carries_source_and_team() {
        let mut world = bevy_ecs::prelude::World::new();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(move |mut commands: Commands| {
            spawn_explosion_with_source_radius(
                &mut commands,
                glam::Vec2::new(5.0, 6.0),
                9,
                None,
                46.0,
                Team::Enemy,
                true,
            );
        });
        schedule.run(&mut world);
        let mut q = world.query::<(&Explosion, &Pos)>();
        let (boom, pos) = q.iter(&world).next().expect("explosion");
        assert_eq!((boom.radius, boom.damage), (46.0, 9));
        assert_eq!(boom.team, Team::Enemy);
        assert!(boom.hits_player);
        assert_eq!(pos.0, glam::Vec2::new(5.0, 6.0));
    }

    #[test]
    fn hazard_cloud_copies_spec() {
        use crate::data::{HazardDef, HazardKind};
        let mut world = bevy_ecs::prelude::World::new();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(move |mut commands: Commands| {
            spawn_hazard_cloud(
                &mut commands,
                glam::Vec2::ZERO,
                Team::Player,
                HazardDef {
                    kind: HazardKind::Fire,
                    radius: 10.0,
                    damage: 2,
                    duration: 3.0,
                    tick: 0.5,
                    color: [1.0, 0.0, 0.0, 1.0],
                },
            );
        });
        schedule.run(&mut world);
        let mut q = world.query::<(&HazardCloud, &Pos)>();
        let (hz, _) = q.iter(&world).next().expect("cloud");
        assert_eq!((hz.radius, hz.damage), (10.0, 2));
        assert_eq!(hz.timer.duration(), 3.0);
        assert_eq!(hz.tick.duration(), 0.5);
    }

    #[test]
    fn sentry_copies_spec_timers() {
        let mut world = bevy_ecs::prelude::World::new();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(move |mut commands: Commands| {
            spawn_sentry_turret(
                &mut commands,
                glam::Vec2::ZERO,
                DeploysSentry {
                    life: 5.0,
                    fire_interval: 0.5,
                    range: 100.0,
                    projectile_speed: 200.0,
                    projectile_damage: 3,
                },
            );
        });
        schedule.run(&mut world);
        let mut q = world.query::<(&SentryTurret, &Team)>();
        let (turret, team) = q.iter(&world).next().expect("turret");
        assert_eq!(*team, Team::Player);
        assert_eq!(turret.range, 100.0);
        assert_eq!(turret.projectile_damage, 3);
    }
}
