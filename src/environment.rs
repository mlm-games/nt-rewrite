//! Area hazards, surface zones, and prop-death effects. Ported from
//! nt's `game/environment.rs` (data + spawners + sim-half ticks;
//! `animate_environment`'s alpha law lives in [`SurfacePulse::alpha_at`]
//! and applies renderer-side; art paths resolve renderer-side from the
//! recorded [`PulseSprite`]). Hazard tint art likewise resolves
//! renderer-side from the spec; colors below are the exact bevy values
//! as arrays.

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_sim::SimTime;

use crate::anim::SpriteAnim;
use crate::combat::{Explosion, HitFlash};
use crate::comps_a::{
    ARENA_H, ARENA_W, DamageSource, FloorMask, GameCleanup, Health, LevelCleanup, Player,
    Projectile, ProjectileTyp, Team, Velocity, boiling_veins_damage,
};
use crate::comps_b::{
    Dash, Enemy, FxAngle, GmlImage, GroundPhysics, Mote, MoteScale, MoteStrip, NativeAngle,
    NativeDepth, NativeExplosionKind, NativeFlip, NativeLifetime, NativeMotion, NativeWallMotion,
    PickupLifetime, Prop, PropSprites, TopSmall, TrapFire,
};
use crate::enemy_data::enemy_def;
use crate::secrets::SecretTriggers;
use crate::spatial::{Pos, move_bounce_solid};
use crate::time::{GTimer, TimerMode};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SurfaceKind {
    Cobweb,
    Ice,
}

/// Friction/conveyor patch (bevy `SurfaceZone` parity: kind +
/// half-extents; the center rides on [`Pos`], matching bevy's
/// `Transform + SurfaceZone` pair).
#[derive(Component, Clone, Copy, Debug)]
pub struct SurfaceZone {
    pub kind: SurfaceKind,
    pub half_size: glam::Vec2,
}

pub fn point_in_zone(point: glam::Vec2, center: glam::Vec2, half_size: glam::Vec2) -> bool {
    let delta = (point - center).abs();
    delta.x <= half_size.x && delta.y <= half_size.y
}

/// Cobweb wins over ice on overlap (bevy parity).
pub fn surface_at_point(
    point: glam::Vec2,
    zones: impl IntoIterator<Item = (glam::Vec2, SurfaceZone)>,
) -> Option<SurfaceKind> {
    let mut result = None;

    for (center, zone) in zones {
        if !point_in_zone(point, center, zone.half_size) {
            continue;
        }

        match zone.kind {
            SurfaceKind::Cobweb => return Some(SurfaceKind::Cobweb),
            SurfaceKind::Ice => result = Some(SurfaceKind::Ice),
        }
    }

    result
}

/// Per-surface velocity rewrite (bevy `surface_velocity` parity:
/// cobweb damps and caps low, ice counteracts base friction so bodies
/// glide and caps high). `base_friction`/`max_speed` come from the
/// actor (player stats, enemy def, or the 0.84/240 fallback).
pub fn surface_velocity(
    kind: SurfaceKind,
    velocity: glam::Vec2,
    dt: f32,
    base_friction: f32,
    max_speed: f32,
) -> glam::Vec2 {
    match kind {
        SurfaceKind::Cobweb => {
            let retention = 0.72_f32.powf(dt * crate::SIM_HZ as f32);
            let mut next = velocity * retention;
            let cap = max_speed.max(1.0) * 0.52;

            if next.length() > cap {
                next = next.normalize_or_zero() * cap;
            }

            next
        }

        SurfaceKind::Ice => {
            let friction = base_friction.clamp(0.05, 0.999);
            let compensation = (1.0 / friction).powf(dt * crate::SIM_HZ as f32);
            let retention = 0.992_f32.powf(dt * crate::SIM_HZ as f32);

            let mut next = velocity * compensation * retention;
            let cap = max_speed.max(1.0) * 1.28;

            if next.length() > cap {
                next = next.normalize_or_zero() * cap;
            }

            next
        }
    }
}

/// Rewrite actor velocities standing on a surface zone (bevy
/// `apply_surface_effects` parity). Dashing actors skip; players use
/// their own friction/speed, enemies their def speed (floored at 40).
#[allow(clippy::type_complexity)]
pub fn apply_surface_effects(
    time: Res<SimTime>,
    zones: Query<(&Pos, &SurfaceZone)>,
    mut actors: Query<
        (
            &Pos,
            &mut Velocity,
            Option<&Player>,
            Option<&Enemy>,
            Option<&Dash>,
        ),
        (Without<SurfaceZone>, Without<Projectile>),
    >,
) {
    let dt = time.delta_secs;

    for (pos, mut velocity, player, enemy, dash) in &mut actors {
        if dash.is_some() {
            continue;
        }

        let Some(surface) = surface_at_point(
            pos.0,
            zones.iter().map(|(zone_pos, zone)| (zone_pos.0, *zone)),
        ) else {
            continue;
        };

        let (friction, max_speed) = if let Some(player) = player {
            (player.friction, player.speed * player.speed_mult)
        } else if let Some(enemy) = enemy {
            let def = enemy_def(enemy.kind);
            (0.84, def.speed.max(40.0))
        } else {
            (0.84, 240.0)
        };

        velocity.0 = surface_velocity(surface, velocity.0, dt, friction, max_speed);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EnvironmentHazardKind {
    Fire,
    Toxic,
}

impl EnvironmentHazardKind {
    pub fn hit_id(self) -> crate::comps_a::HitId {
        match self {
            EnvironmentHazardKind::Fire => crate::comps_a::HitId::Fire,
            EnvironmentHazardKind::Toxic => crate::comps_a::HitId::Toxic,
        }
    }

    pub fn color(self) -> [f32; 4] {
        match self {
            EnvironmentHazardKind::Fire => [1.0, 0.43, 0.10, 0.36],
            EnvironmentHazardKind::Toxic => [0.30, 0.88, 0.30, 0.36],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EnvironmentHazardSpec {
    pub kind: EnvironmentHazardKind,
    pub radius: f32,
    pub damage: i32,
    pub duration: f32,
    pub tick: f32,
    pub hurts_player: bool,
    pub hurts_enemies: bool,
}

impl EnvironmentHazardSpec {
    pub fn toxic_barrel() -> Self {
        Self {
            kind: EnvironmentHazardKind::Toxic,
            radius: 62.0,
            damage: 1,
            duration: 3.0,
            tick: 0.24,
            hurts_player: true,
            hurts_enemies: true,
        }
    }

    pub fn fire_trap() -> Self {
        Self {
            kind: EnvironmentHazardKind::Fire,
            radius: 42.0,
            damage: 2,
            duration: 9_999.0,
            tick: 0.38,
            hurts_player: true,
            hurts_enemies: true,
        }
    }

    pub fn mine_fire() -> Self {
        Self {
            kind: EnvironmentHazardKind::Fire,
            radius: 48.0,
            damage: 1,
            duration: 0.9,
            tick: 0.18,
            hurts_player: true,
            hurts_enemies: true,
        }
    }

    pub fn ground_flame() -> Self {
        Self {
            kind: EnvironmentHazardKind::Fire,
            radius: 14.0,
            damage: 1,
            duration: 2.5,
            tick: 0.5,
            hurts_player: true,
            hurts_enemies: true,
        }
    }
}

#[derive(Component, Clone, Debug)]
pub struct EnvironmentHazard {
    pub spec: EnvironmentHazardSpec,
    pub life: GTimer,
    pub damage_tick: GTimer,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct GroundFlame {
    pub alarm: f32,
    pub big: bool,
    pub disappearing: bool,
}

impl EnvironmentHazard {
    pub fn new(spec: EnvironmentHazardSpec) -> Self {
        Self {
            life: GTimer::from_seconds(spec.duration.max(0.01), TimerMode::Once),
            damage_tick: GTimer::from_seconds(spec.tick.max(0.01), TimerMode::Repeating),
            spec,
        }
    }
}

pub fn spawn_environment_hazard(
    commands: &mut Commands,
    pos: glam::Vec2,
    spec: EnvironmentHazardSpec,
) -> Entity {
    commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            EnvironmentHazard::new(spec),
            // Bevy attaches the hazard pulse + a solid-color sprite here
            // (no image); the renderer draws the tinted rect with the
            // `animate_environment` alpha law.
            SurfacePulse::hazard(pos.x * 0.013 + pos.y * 0.009),
            Pos(pos),
        ))
        .id()
}

/// Bevy `SurfacePulse` verbatim: per-entity alpha throb applied every
/// frame by `animate_environment` (renderer-side here — same wave law,
/// driven by the sim clock).
#[derive(Component, Clone, Copy, Debug)]
pub struct SurfacePulse {
    pub speed: f32,
    pub min_alpha: f32,
    pub max_alpha: f32,
    pub phase: f32,
}

impl SurfacePulse {
    pub fn subtle(phase: f32) -> Self {
        Self {
            speed: 1.8,
            min_alpha: 0.55,
            max_alpha: 0.86,
            phase,
        }
    }

    pub fn hazard(phase: f32) -> Self {
        Self {
            speed: 4.2,
            min_alpha: 0.52,
            max_alpha: 0.95,
            phase,
        }
    }

    /// Bevy `animate_environment` alpha law at time `now` (seconds).
    pub fn alpha_at(&self, now: f32) -> f32 {
        let wave = 0.5 + 0.5 * (now * self.speed + self.phase).sin();
        self.min_alpha + (self.max_alpha - self.min_alpha) * wave
    }
}

/// Renderer-side record for bevy `sprite_from_candidates` decal
/// entities (cobweb / ice / fire-trap visuals): the resolved art path
/// (`None` = solid fallback rect, bevy `Sprite` without image), base
/// tint, drawn size, and x-flip. Alpha always comes from the paired
/// [`SurfacePulse`] via `alpha_at` (bevy `animate_environment`
/// overwrites the sprite alpha every frame).
#[derive(Component, Clone, Copy, Debug)]
pub struct PulseSprite {
    pub path: Option<&'static str>,
    pub tint: [f32; 4],
    pub size: f32,
    pub flip_x: bool,
}

/// Bevy `sprite_from_candidates` pick law verbatim: first
/// catalog-present candidate wins (no hash-pick).
pub fn pick_first_present(
    catalog: &repame_anim::AnimCatalog,
    candidates: &[&'static str],
) -> Option<&'static str> {
    candidates
        .iter()
        .copied()
        .find(|p| catalog.def(p).is_some())
}

/// Damage actors standing in live hazards (bevy
/// `tick_environment_hazards` gameplay half: life/damage-tick timers,
/// team gates, radius + player-invuln gates, Boiling Veins fire
/// immunity at/below threshold, player invuln refresh +
/// `mark_damage_taken`). Hit-flash rides the existing [`HitFlash`]
/// marker and damage numbers the `repame_fx` floaters; the pulse alpha
/// rides [`SurfacePulse`] renderer-side (`animate_environment` law).
#[allow(clippy::type_complexity)]
pub fn tick_environment_hazards(
    time: Res<SimTime>,
    mut commands: Commands,
    mut secrets: ResMut<SecretTriggers>,
    mut hazards: Query<(Entity, &Pos, &mut EnvironmentHazard)>,
    mut targets: Query<
        (Entity, &Pos, &Team, &mut Health, Option<&Player>),
        (
            Without<EnvironmentHazard>,
            Without<Projectile>,
            Without<Explosion>,
        ),
    >,
) {
    let dt = time.delta_secs;
    for (hazard_entity, hazard_pos, mut hazard) in hazards.iter_mut() {
        hazard.life.tick(dt);
        hazard.damage_tick.tick(dt);

        if hazard.life.just_finished() {
            commands.entity(hazard_entity).despawn();
            continue;
        }

        if !hazard.damage_tick.just_finished() {
            continue;
        }

        let center = hazard_pos.0;

        for (target_entity, target_pos, team, mut health, player) in targets.iter_mut() {
            let is_player = *team == Team::Player;

            if is_player && !hazard.spec.hurts_player {
                continue;
            }
            if !is_player && !hazard.spec.hurts_enemies {
                continue;
            }
            if target_pos.0.distance(center) > hazard.spec.radius {
                continue;
            }
            if is_player && !health.invuln.is_finished() {
                continue;
            }

            let mut damage = hazard.spec.damage;
            if is_player
                && hazard.spec.kind == EnvironmentHazardKind::Fire
                && let Some(player) = player
                && player.boiling_veins
            {
                damage = boiling_veins_damage(health.hp, damage, player.veins_threshold);
            }

            health.hp -= damage;

            if is_player {
                health.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                secrets.mark_damage_taken();
            }

            HitFlash::apply(&mut commands, target_entity, hazard.spec.kind.color(), 0.08);
            repame_fx::spawn_number(
                &mut commands,
                target_pos.0.x,
                target_pos.0.y,
                format!("{}", damage),
                hazard.spec.kind.color(),
            );
        }
    }
}

/// Arena-bounds check for hazard/prop placement (bevy
/// `valid_environment_position` parity).
pub fn valid_environment_position(pos: glam::Vec2, radius: f32) -> bool {
    pos.x.abs() <= ARENA_W * 0.5 - radius && pos.y.abs() <= ARENA_H * 0.5 - radius
}

/// Explosion payload shared by prop deaths and mines.
#[derive(Clone, Copy, Debug)]
pub struct ExplosionPayload {
    pub radius: f32,
    pub damage: i32,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PropDeathEffect {
    pub explosion: Option<ExplosionPayload>,
    pub hazard: Option<EnvironmentHazardSpec>,
    pub ground_flames: u8,
    pub dust_ring: u8,
    pub feather_burst: Option<FeatherBurst>,
}

/// Visual-only petal/leaf/money scatter (GML `Feather` with a tinted
/// sprite: Bush leaves, MoneyPile bills). Real sprite motes now —
/// see [`Mote`].
#[derive(Clone, Copy, Debug)]
pub struct FeatherBurst {
    pub count: u8,
    pub strip: MoteStrip,
}

impl PropDeathEffect {
    pub fn toxic_barrel() -> Self {
        Self {
            explosion: Some(ExplosionPayload {
                radius: 95.0,
                damage: 5,
            }),
            hazard: Some(EnvironmentHazardSpec::toxic_barrel()),
            ground_flames: 4,
            ..Default::default()
        }
    }

    pub fn car() -> Self {
        Self {
            explosion: Some(ExplosionPayload {
                radius: 155.0,
                damage: 8,
            }),
            hazard: None,
            ground_flames: 6,
            ..Default::default()
        }
    }

    pub fn mine() -> Self {
        Self {
            explosion: Some(ExplosionPayload {
                radius: 118.0,
                damage: 7,
            }),
            hazard: Some(EnvironmentHazardSpec::mine_fire()),
            ground_flames: 4,
            ..Default::default()
        }
    }

    pub fn legacy_barrel() -> Self {
        Self {
            explosion: Some(ExplosionPayload {
                radius: 110.0,
                damage: 6,
            }),
            hazard: None,
            ground_flames: 4,
            ..Default::default()
        }
    }

    pub fn small_generator() -> Self {
        Self {
            explosion: Some(ExplosionPayload {
                radius: 110.0,
                damage: 12,
            }),
            hazard: None,
            ground_flames: 6,
            ..Default::default()
        }
    }

    pub fn big_generator() -> Self {
        Self {
            explosion: Some(ExplosionPayload {
                radius: 130.0,
                damage: 8,
            }),
            hazard: None,
            ground_flames: 6,
            ..Default::default()
        }
    }

    pub fn dust_ring() -> Self {
        Self {
            dust_ring: 10,
            ..Default::default()
        }
    }

    pub fn leaves() -> Self {
        Self {
            feather_burst: Some(FeatherBurst {
                count: 5,
                strip: MoteStrip::Leaf,
            }),
            ..Default::default()
        }
    }

    pub fn money() -> Self {
        Self {
            feather_burst: Some(FeatherBurst {
                count: 10,
                strip: MoteStrip::Money,
            }),
            ..Default::default()
        }
    }
}

pub fn spawn_prop_corpse(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    sprites: &PropSprites,
) {
    let mut ec = commands.spawn((GameCleanup, LevelCleanup, *sprites, Pos(pos)));
    if sprites.dead == "images/sprScorchmark.png" {
        ec.insert((NativeDepth(5.0), GmlImage::new(sprites.dead, 1, 0.0)));
    } else {
        if let Some(def) = catalog.def(sprites.dead) {
            ec.insert(SpriteAnim::oneshot(sprites.dead, def));
        }
        ec.insert(PickupLifetime {
            timer: GTimer::from_seconds(12.0, TimerMode::Once),
        });
    }
}

pub fn spawn_prop_death_effect(
    mut commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    particles_on: bool,
    pos: glam::Vec2,
    explicit: Option<PropDeathEffect>,
    legacy_explosive: bool,
    source: Option<DamageSource>,
) {
    let effect = explicit.or_else(|| legacy_explosive.then_some(PropDeathEffect::legacy_barrel()));

    let Some(effect) = effect else {
        let mut rng = rand::rng();
        crate::effects::spawn_burst(
            &mut commands,
            &mut rng,
            pos,
            8,
            [0.78, 0.65, 0.42, 1.0],
            (50.0, 150.0),
        );
        return;
    };

    if let Some(explosion) = effect.explosion {
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            crate::combat::Explosion {
                timer: GTimer::from_seconds(0.04, TimerMode::Once),
                radius: explosion.radius,
                damage: explosion.damage,
                team: Team::Player,
                hits_player: true,
                source,
            },
            Pos(pos),
        ));
    }

    if let Some(hazard) = effect.hazard {
        spawn_environment_hazard(commands, pos, hazard);
    }

    if effect.ground_flames > 0 {
        let mut rng = rand::rng();
        for _ in 0..effect.ground_flames {
            let off = glam::Vec2::new(rng.random_range(-24.0..24.0), rng.random_range(-24.0..24.0));
            spawn_ground_flame(commands, pos + off, false);
        }
    }

    if effect.dust_ring > 0 {
        spawn_motes(
            commands,
            catalog,
            particles_on,
            pos,
            MoteStrip::Dust,
            effect.dust_ring as usize,
        );
    }

    if let Some(feather) = effect.feather_burst {
        spawn_motes(
            commands,
            catalog,
            particles_on,
            pos,
            feather.strip,
            feather.count as usize,
        );
    }
}

pub fn spawn_native_smoke_mote(
    commands: &mut Commands,
    particles_on: bool,
    pos: glam::Vec2,
    direction: glam::Vec2,
    speed: f32,
) {
    if !particles_on {
        return;
    }
    let mut rng = rand::rng();
    let angle = rng.random_range(0.0..std::f32::consts::TAU);
    let motion_angle = rng.random_range(0.0..std::f32::consts::TAU);
    let spin = (1.0 + rng.random_range(0.0..=3.0)) * if rng.random_bool(0.5) { 1.0 } else { -1.0 };
    let mut image = GmlImage::new("images/sprSmoke.png", 5, 0.0);
    image.phase = rng.random_range(0.0..5.0);
    let motion = if direction.length_squared() > 0.0 {
        direction.normalize_or_zero()
    } else {
        glam::Vec2::from_angle(motion_angle)
    };
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        GroundPhysics {
            vel: motion * speed * 30.0,
            rotspeed: spin,
        },
        FxAngle(angle),
        Mote {
            friction: 0.1,
            spin,
            grow: rng.random_range(0.0..0.005),
            grow_decay: 0.001,
            sway: false,
            lifetime: None,
            fall: None,
            bounce_period: 3,
            kill_on_spiral: true,
            tick: 0,
        },
        MoteScale(0.8),
        MoteStrip::Smoke,
        image,
        NativeDepth(-1.0),
        Pos(pos),
    ));
}

fn spawn_native_dust_mote(
    commands: &mut Commands,
    particles_on: bool,
    pos: glam::Vec2,
    direction: glam::Vec2,
    speed: f32,
) {
    if !particles_on {
        return;
    }
    let mut rng = rand::rng();
    let spin = (1.0 + rng.random_range(0.0..=3.0)) * if rng.random_bool(0.5) { 1.0 } else { -1.0 };
    let mut image = GmlImage::new("images/sprDust.png", 5, 0.0);
    image.phase = rng.random_range(0.0..5.0);
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        GroundPhysics {
            vel: direction.normalize_or_zero() * speed * 30.0,
            rotspeed: spin,
        },
        FxAngle(rng.random_range(0.0..std::f32::consts::TAU)),
        Mote {
            friction: 0.3,
            spin,
            grow: rng.random_range(0.05..0.10),
            grow_decay: 0.02,
            sway: false,
            lifetime: None,
            fall: None,
            bounce_period: 0,
            kill_on_spiral: false,
            tick: 0,
        },
        MoteScale(0.7),
        MoteStrip::Dust,
        image,
        NativeDepth(-1.0),
        Pos(pos),
    ));
}

pub fn spawn_native_streak(
    commands: &mut Commands,
    acid: bool,
    pos: glam::Vec2,
    angle: f32,
    speed: f32,
) -> Entity {
    let path = if acid {
        "images/sprAcidStreak.png"
    } else {
        "images/sprBloodStreak.png"
    };
    commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            Pos(pos),
            NativeMotion {
                velocity: glam::Vec2::from_angle(angle) * speed * 30.0,
                friction: 0.4,
                radius: 8.0,
                wall: NativeWallMotion::Stop,
                tick: 0,
            },
            NativeAngle(angle),
            NativeDepth(-4.0),
            GmlImage::animated(path, 7, 0.4, true),
        ))
        .id()
}

pub fn spawn_exploder_explo(
    commands: &mut Commands,
    particles_on: bool,
    pos: glam::Vec2,
    direction: glam::Vec2,
    speed: f32,
) -> Entity {
    let mut rng = rand::rng();
    let start = rng.random_range(0.0..std::f32::consts::TAU);
    for i in 0..6 {
        let angle = start + i as f32 * std::f32::consts::TAU / 6.0;
        spawn_native_smoke_mote(
            commands,
            particles_on,
            pos,
            glam::Vec2::from_angle(angle),
            4.0 + rng.random_range(0.0..1.0),
        );
    }
    let motion = direction.normalize_or_zero();
    commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            Pos(pos),
            NativeMotion {
                velocity: motion * speed * 30.0,
                friction: 0.0,
                radius: 12.0,
                wall: NativeWallMotion::Bounce,
                tick: 0,
            },
            NativeAngle(motion.y.atan2(motion.x)),
            NativeLifetime { ticks: 15.0 },
            NativeDepth(-5.0),
            GmlImage::animated("images/sprExploderExplo.png", 6, 0.4, true),
        ))
        .id()
}

pub fn spawn_ground_flame(commands: &mut Commands, pos: glam::Vec2, big: bool) -> Entity {
    let mut rng = rand::rng();
    let (base, _, frames) = if big {
        (
            "images/sprGroundFlameBig.png",
            "images/sprGroundFlameBigDisappear.png",
            8,
        )
    } else {
        (
            "images/sprGroundFlame.png",
            "images/sprGroundFlameDisappear.png",
            8,
        )
    };
    let mut image = GmlImage::new(base, frames, 0.4);
    image.phase = 0.0;
    commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            Pos(pos),
            GroundFlame {
                alarm: 300.0 + rng.random_range(0.0..120.0),
                big,
                disappearing: false,
            },
            image,
            NativeFlip(rng.random_bool(0.5)),
            NativeDepth(-1.0),
        ))
        .id()
}

pub fn spawn_trap_fire(
    commands: &mut Commands,
    pos: glam::Vec2,
    direction: glam::Vec2,
    speed: f32,
    team: Team,
    source: Option<DamageSource>,
) -> Entity {
    spawn_trap_fire_with_image(
        commands,
        pos,
        direction,
        speed,
        team,
        source,
        "images/sprTrapFire.png",
        None,
    )
}

pub fn spawn_trap_fire_with_image(
    commands: &mut Commands,
    pos: glam::Vec2,
    direction: glam::Vec2,
    speed: f32,
    team: Team,
    source: Option<DamageSource>,
    path: &'static str,
    image_angle: Option<f32>,
) -> Entity {
    let mut rng = rand::rng();
    let image_speed = 0.2 + rng.random_range(0.0..0.1);
    let image = GmlImage::new(path, 7, image_speed);
    let native_angle = image_angle.unwrap_or_else(|| rng.random_range(0.0..std::f32::consts::TAU));
    commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            TrapFire,
            team,
            Projectile {
                damage: 1,
                life: GTimer::from_seconds(4.0, TimerMode::Once),
                radius: 8.0,
                knockback: 0.0,
                explosive: false,
                source,
            },
            ProjectileTyp(2),
            Velocity(direction.normalize_or_zero() * speed),
            NativeAngle(native_angle),
            NativeDepth(-2.0),
            image,
            Pos(pos),
        ))
        .id()
}

pub fn spawn_native_scorch(commands: &mut Commands, pos: glam::Vec2, green: bool) {
    let path = if green {
        "images/sprScorchGreen.png"
    } else {
        "images/sprScorch.png"
    };
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        Pos(pos),
        NativeDepth(7.0),
        GmlImage::new(path, 1, 0.0),
    ));
}

pub fn spawn_native_explosion_visual(
    commands: &mut Commands,
    particles_on: bool,
    pos: glam::Vec2,
    kind: NativeExplosionKind,
    on_floor: bool,
) -> Entity {
    let (path, frames) = match kind {
        NativeExplosionKind::Standard => ("images/sprExplosion.png", 9),
        NativeExplosionKind::Small => ("images/sprSmallExplosion.png", 7),
        NativeExplosionKind::Green => ("images/sprGreenExplosion.png", 9),
        NativeExplosionKind::Meat => ("images/sprMeatExplosion.png", 6),
        NativeExplosionKind::Popo => ("images/sprPopoExplo.png", 8),
    };
    let mut rng = rand::rng();
    match kind {
        NativeExplosionKind::Standard | NativeExplosionKind::Green => {
            let count = 20;
            for _ in 0..count / 2 {
                let a = rng.random_range(0.0..std::f32::consts::TAU);
                spawn_native_smoke_mote(
                    commands,
                    particles_on,
                    pos,
                    glam::Vec2::from_angle(a),
                    2.0 + rng.random_range(0.0..3.0),
                );
            }
            let start = rng.random_range(0.0..std::f32::consts::TAU);
            for i in 0..count {
                let a = start + i as f32 * std::f32::consts::TAU / count as f32;
                spawn_native_dust_mote(commands, particles_on, pos, glam::Vec2::from_angle(a), 6.0);
            }
        }
        NativeExplosionKind::Small => {
            let count = 8;
            for _ in 0..count / 2 {
                let a = rng.random_range(0.0..std::f32::consts::TAU);
                spawn_native_smoke_mote(
                    commands,
                    particles_on,
                    pos,
                    glam::Vec2::from_angle(a),
                    2.0 + rng.random_range(0.0..3.0),
                );
            }
            let start = rng.random_range(0.0..std::f32::consts::TAU);
            for i in 0..count {
                let a = start + i as f32 * std::f32::consts::TAU / count as f32;
                spawn_native_dust_mote(commands, particles_on, pos, glam::Vec2::from_angle(a), 6.0);
            }
        }
        NativeExplosionKind::Meat => {
            let start = rng.random_range(0.0..std::f32::consts::TAU);
            for i in 0..6 {
                spawn_native_smoke_mote(
                    commands,
                    particles_on,
                    pos,
                    glam::Vec2::from_angle(start + i as f32 * std::f32::consts::TAU / 6.0),
                    4.0 + rng.random_range(0.0..1.0),
                );
            }
        }
        NativeExplosionKind::Popo => {
            for _ in 0..10 {
                let a = rng.random_range(0.0..std::f32::consts::TAU);
                spawn_native_smoke_mote(
                    commands,
                    particles_on,
                    pos,
                    glam::Vec2::from_angle(a),
                    2.0 + rng.random_range(0.0..3.0),
                );
            }
            let start = rng.random_range(0.0..std::f32::consts::TAU);
            for i in 0..20 {
                let a = start + i as f32 * std::f32::consts::TAU / 20.0;
                spawn_native_dust_mote(commands, particles_on, pos, glam::Vec2::from_angle(a), 6.0);
            }
            for _ in 0..10 {
                let a = rng.random_range(0.0..std::f32::consts::TAU);
                spawn_native_smoke_mote(
                    commands,
                    particles_on,
                    pos,
                    glam::Vec2::from_angle(a),
                    1.0 + rng.random_range(0.0..2.0),
                );
            }
            let start = rng.random_range(0.0..std::f32::consts::TAU);
            for i in 0..20 {
                let a = start + i as f32 * std::f32::consts::TAU / 20.0;
                spawn_native_dust_mote(commands, particles_on, pos, glam::Vec2::from_angle(a), 6.0);
            }
        }
    }
    let visual = commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            Pos(pos),
            NativeDepth(-5.0),
            GmlImage::animated(path, frames, 0.4, true),
        ))
        .id();
    if on_floor
        && matches!(
            kind,
            NativeExplosionKind::Standard | NativeExplosionKind::Green
        )
    {
        spawn_native_scorch(commands, pos, false);
    }
    visual
}

pub fn tick_native_lifetimes(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut NativeLifetime)>,
) {
    let steps = time.delta_secs * crate::SIM_HZ as f32;
    for (entity, mut life) in &mut q {
        life.ticks -= steps;
        if life.ticks <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}

pub fn tick_native_motion(
    time: Res<SimTime>,
    mask: Option<Res<FloorMask>>,
    mut sets: ParamSet<(
        Query<(&Prop, &Pos), With<Prop>>,
        Query<(&mut Pos, &mut NativeMotion)>,
    )>,
) {
    let dt = time.delta_secs;
    let solids: Vec<(glam::Vec2, glam::Vec2)> = {
        let props = sets.p0();
        props.iter().map(|(prop, pos)| (pos.0, prop.size)).collect()
    };
    for (mut pos, mut motion) in &mut sets.p1() {
        motion.tick = motion.tick.wrapping_add(1);
        let friction = motion.friction;
        crate::comps_a::apply_gml_friction(&mut motion.velocity, friction, dt);
        let bounce = match motion.wall {
            NativeWallMotion::Bounce => true,
            NativeWallMotion::BounceEveryThird => motion.tick % 3 == 0,
            NativeWallMotion::Stop => false,
        };
        let radius = motion.radius;
        let contact = move_bounce_solid(
            &mut pos.0,
            &mut motion.velocity,
            radius,
            dt,
            &solids,
            mask.as_deref(),
            bounce,
        );
        if contact.is_some() && matches!(motion.wall, NativeWallMotion::Stop) {
            motion.velocity = glam::Vec2::ZERO;
        }
        if motion.velocity.length_squared() <= 1e-8 {
            motion.velocity = glam::Vec2::ZERO;
        }
    }
}

pub fn tick_ground_flames(time: Res<SimTime>, mut q: Query<(&mut GroundFlame, &mut GmlImage)>) {
    let steps = time.delta_secs * crate::SIM_HZ as f32;
    for (mut flame, mut image) in &mut q {
        if flame.disappearing {
            continue;
        }
        flame.alarm -= steps;
        if flame.alarm <= 0.0 {
            flame.disappearing = true;
            let path = if flame.big {
                "images/sprGroundFlameBigDisappear.png"
            } else {
                "images/sprGroundFlameDisappear.png"
            };
            image.set_path(path, 4);
            image.looping = false;
            image.destroy_on_end = true;
        }
    }
}

/// `GroundPhysics` + `FxAngle` + `Mote` + `MoteScale` + `SpriteAnim`
/// entity per mote, `PickupLifetime` expiry. Speed/friction/spin/grow
/// laws are verbatim from the `Create_0` handlers (speeds converted
/// px/step → px/s at 30 Hz); the step integration lives in
/// [`tick_motes`].
#[allow(clippy::too_many_arguments)]
pub fn spawn_motes(
    commands: &mut Commands,
    _catalog: &repame_anim::AnimCatalog,
    particles_on: bool,
    pos: glam::Vec2,
    strip: MoteStrip,
    count: usize,
) {
    if !particles_on {
        return;
    }
    let mut rng = rand::rng();
    for _ in 0..count {
        if matches!(strip, MoteStrip::Smoke) {
            spawn_native_smoke_mote(
                commands,
                true,
                pos,
                glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU)),
                0.5,
            );
            continue;
        }
        let (
            path,
            frames,
            speed_lo,
            speed_hi,
            friction,
            grow,
            grow_decay,
            scale,
            spin_max,
            sway,
            lifetime,
            image_speed,
            bounce_period,
        ) = match strip {
            MoteStrip::Dust => (
                "images/sprDust.png",
                5,
                0.0,
                2.0,
                0.3,
                0.0,
                0.02,
                0.7,
                4.0,
                false,
                None,
                0.0,
                0,
            ),
            MoteStrip::Leaf | MoteStrip::Money | MoteStrip::Raven => (
                match strip {
                    MoteStrip::Leaf => "images/sprLeaf.png",
                    MoteStrip::Money => "images/sprMoney.png",
                    _ => "images/sprRavenFeather.png",
                },
                match strip {
                    MoteStrip::Leaf => 5,
                    MoteStrip::Money => 1,
                    _ => 1,
                },
                1.8,
                3.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                true,
                Some((150.0 + rng.random_range(0.0..30.0)) / 30.0),
                if matches!(strip, MoteStrip::Money | MoteStrip::Raven) {
                    0.0
                } else {
                    0.4
                },
                1,
            ),
            MoteStrip::Curse => (
                "images/sprCurse.png",
                6,
                0.2,
                0.7,
                0.005,
                0.0,
                0.0,
                1.0,
                0.0,
                false,
                None,
                0.3 + rng.random_range(0.0..0.1),
                0,
            ),
            MoteStrip::Smoke => unreachable!(),
        };
        let a = rng.random_range(0.0..std::f32::consts::TAU);
        let dir = glam::Vec2::from_angle(a);
        let mut vel = dir * rng.random_range(speed_lo..speed_hi) * 30.0;
        if matches!(strip, MoteStrip::Curse) {
            vel = glam::Vec2::new(
                rng.random_range(-0.2..0.2) * 30.0,
                -(0.5 + rng.random_range(0.0..0.5)) * 30.0,
            );
        }
        if sway {
            vel.y -= rng.random_range(0.9..1.2) * 30.0;
        }
        let spin = if sway {
            rng.random_range(-3.0..3.0)
        } else if spin_max > 0.0 {
            rng.random_range(1.0..=spin_max) * if rng.random_bool(0.5) { 1.0 } else { -1.0 }
        } else {
            0.0
        };
        let fall = if sway {
            Some(40.0 + rng.random_range(0.0..30.0))
        } else {
            None
        };
        let grow = match strip {
            MoteStrip::Dust => rng.random_range(0.05..0.10),
            _ => grow,
        };
        let mut image = if matches!(strip, MoteStrip::Curse) {
            GmlImage::animated(path, frames, image_speed, true)
        } else {
            GmlImage::new(path, frames, image_speed)
        };
        if image_speed == 0.0 {
            image.phase = rng.random_range(0.0..frames as f32);
        }
        let depth = match strip {
            MoteStrip::Leaf | MoteStrip::Money | MoteStrip::Raven => 1.0,
            MoteStrip::Dust | MoteStrip::Smoke | MoteStrip::Curse => -1.0,
        };
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            GroundPhysics {
                vel,
                rotspeed: spin,
            },
            FxAngle(rng.random_range(0.0..std::f32::consts::TAU)),
            Mote {
                friction,
                spin,
                grow,
                grow_decay,
                sway,
                lifetime,
                fall,
                bounce_period,
                kill_on_spiral: false,
                tick: 0,
            },
            MoteScale(scale),
            strip,
            image,
            NativeDepth(depth),
            Pos(pos),
        ));
    }
}

/// Floor mine trigger (bevy `environment.rs:467-509` sim half:
/// trigger radius + mine payload; the screen-effect pulse and corpse
/// art resolve renderer-side, trauma stays sim-side).
/// Port adaptation: positions are [`Pos`], teams gate the trigger.
#[derive(Component, Clone, Copy, Debug)]
pub struct ProximityMine {
    pub trigger_radius: f32,
    pub payload: PropDeathEffect,
}

impl Default for ProximityMine {
    fn default() -> Self {
        Self {
            trigger_radius: 54.0,
            payload: PropDeathEffect::mine(),
        }
    }
}

/// GML `Dust/Step_0` + `Smoke/Step_0` + `Feather/Step_0` + `Curse`
/// integration (sim half): flat-friction slide + spin + grow/decay on
/// `MoteScale`, feather fall-sway (`x += 0.35*sin(fall/7)`,
/// `y += 0.3`, `speed *= 0.9` over 0.2, `image_speed = 0` at rest),
/// wall bounce (Smoke every 3rd tick, Feather at half speed —
/// `move_bounce_solid`). `PickupLifetime` expiry is handled by
/// `tick_hit_effects`; scale <= 0 despawns like GML's
/// `image_xscale < 0` kill.
pub fn tick_motes(
    time: Res<SimTime>,
    mut commands: Commands,
    mut sets: ParamSet<(
        Query<
            (
                Entity,
                &mut Pos,
                &mut crate::comps_b::GroundPhysics,
                &mut FxAngle,
                &mut Mote,
                &mut MoteScale,
                Option<&mut GmlImage>,
            ),
            Without<crate::comps_a::WallTile>,
        >,
        Query<&Pos, With<TopSmall>>,
    )>,
    floor: Option<Res<crate::comps_a::FloorMask>>,
    spiral: Option<Res<crate::vortex::SpiralCtl>>,
) {
    let dt = time.delta_secs;
    let steps = dt * crate::SIM_HZ as f32;
    let floor = floor.as_deref();
    let spiral_alive = spiral.is_some_and(|s| s.alive);
    let top_positions: Vec<glam::Vec2> = {
        let top_small = sets.p1();
        top_small.iter().map(|p| p.0).collect()
    };
    let bounce = |pos: glam::Vec2, vel: &mut glam::Vec2, keep: f32| {
        let Some(floor) = floor else { return };
        if floor.is_walkable(pos) {
            return;
        }
        let center = floor.cell_center(floor.world_to_cell(pos));
        if (pos.x - center.x).abs() > (pos.y - center.y).abs() {
            vel.x = -vel.x * keep;
        } else {
            vel.y = -vel.y * keep;
        }
    };
    for (e, mut pos, mut ground, mut angle, mut mote, mut scale, image) in &mut sets.p0() {
        mote.tick = mote.tick.wrapping_add(1);
        let falling = if let Some(fall) = mote.fall {
            let next = fall - steps;
            mote.fall = Some(next);
            if next <= 0.0 {
                if let Some(mut image) = image {
                    image.image_speed = 0.0;
                }
                false
            } else {
                true
            }
        } else {
            mote.sway
        };
        if let Some(mut life) = mote.lifetime {
            life -= dt;
            mote.lifetime = Some(life);
            if life <= 0.0 {
                commands.entity(e).despawn();
                continue;
            }
        }
        if mote.kill_on_spiral && spiral_alive {
            commands.entity(e).despawn();
            continue;
        }
        if mote.kill_on_spiral && top_positions.iter().any(|top| top.distance(pos.0) <= 12.0) {
            commands.entity(e).despawn();
            continue;
        }
        if mote.sway && falling {
            pos.0.x += 0.35 * (steps * 4.0).sin() * 30.0 * dt;
            pos.0.y += 0.3 * 30.0 * dt;
            let sp = ground.vel.length();
            if sp > 0.2 * 30.0 {
                ground.vel *= 0.9_f32.powf(steps);
            } else {
                ground.vel = glam::Vec2::ZERO;
            }
            if mote.bounce_period == 0 || mote.tick % mote.bounce_period.max(1) as u32 == 0 {
                bounce(pos.0, &mut ground.vel, 0.5);
            }
        } else if mote.friction > 0.0 {
            crate::comps_a::apply_gml_friction(&mut ground.vel, mote.friction, dt);
        }
        pos.0 += ground.vel * dt;
        angle.0 += mote.spin * std::f32::consts::PI / 180.0 * steps;
        if mote.grow_decay > 0.0 {
            scale.0 += mote.grow * steps;
            mote.grow -= mote.grow_decay * steps;
            if scale.0 < 0.0 {
                commands.entity(e).despawn();
            }
        }
        if mote.bounce_period > 0 && mote.tick % mote.bounce_period.max(1) as u32 == 0 {
            bounce(pos.0, &mut ground.vel, 1.0);
        }
    }
}

/// GML `Explosion/Create_0` mote ring (sim half): `count/2` `Smoke`
/// at `2+random(3)` plus `count` `Dust` fanned at speed 6 around a
/// random start angle (`_angle += 360/count`). Full size is 20 + 10
/// smoke, small (`SmallExplosion`) is 8 + 4. Gated on `particles_on`
/// (`UberCont.opt_prtcls`).
pub fn spawn_explosion_motes(
    commands: &mut Commands,
    _catalog: &repame_anim::AnimCatalog,
    particles_on: bool,
    pos: glam::Vec2,
    small: bool,
) {
    let kind = if small {
        NativeExplosionKind::Small
    } else {
        NativeExplosionKind::Standard
    };
    spawn_native_explosion_visual(commands, particles_on, pos, kind, false);
}

/// Area-fog scroll (GML `objects/TopCont`: `Create_0` sets
/// `fogscroll = 0`; `Draw_0` advances it). Only the sewers family
/// draws fog; the scroll itself is area-independent.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Default)]
pub struct FogState {
    /// GML `fogscroll` px (wraps at the 480 px tile width).
    pub scroll: f32,
}

/// Fog tile width the scroll wraps at (GML `if fogscroll >= 480`).
pub const FOG_WRAP: f32 = 480.0;

/// One fog-scroll step (GML `TopCont/Draw_0` verbatim:
/// `fogscroll += timescale * 0.5`, wrap at 480). `steps` is
/// timescale steps (`delta * 30`); skipped while paused by the
/// system below (GML `if !UberCont.paused`).
pub fn fog_scroll_step(scroll: f32, steps: f32) -> f32 {
    let mut next = scroll + steps * 0.5;
    if next >= FOG_WRAP {
        next -= FOG_WRAP;
    }
    next
}

/// Advance the fog scroll at the fixed cadence (GML `Draw_0` ran per
/// render frame; headless runs it per 30 Hz tick with
/// `timescale ~= delta * 30`). Paused ticks freeze it, exactly like
/// GML's `UberCont.paused` gate.
pub fn tick_fog(
    time: Res<SimTime>,
    paused: Option<Res<crate::state::Paused>>,
    fog: Option<ResMut<FogState>>,
) {
    let Some(mut fog) = fog else {
        return;
    };
    if paused.is_some_and(|p| p.0) {
        return;
    }
    fog.scroll = fog_scroll_step(fog.scroll, time.delta_secs * 30.0);
}

/// Detonate mines touched by player/enemy actors (bevy parity:
/// corpse + death effect + 0.20 trauma, then despawn).
pub fn tick_proximity_mines(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    save: Res<crate::savedata_part::SaveData>,
    mut trauma: ResMut<repame_fx::Trauma>,
    mines: Query<(Entity, &Pos, &ProximityMine, Option<&PropSprites>), With<Prop>>,
    targets: Query<(&Pos, &Team), Without<ProximityMine>>,
) {
    for (mine_e, mine_pos, mine, sprites) in &mines {
        let center = mine_pos.0;
        let triggered = targets.iter().any(|(target_pos, team)| {
            matches!(*team, Team::Player | Team::Enemy)
                && target_pos.0.distance(center) <= mine.trigger_radius
        });
        if !triggered {
            continue;
        }
        if let Some(ps) = sprites.copied() {
            spawn_prop_corpse(&mut commands, &catalog, center, &ps);
        }
        spawn_prop_death_effect(
            &mut commands,
            &catalog,
            save.settings.particles,
            center,
            Some(mine.payload),
            false,
            None,
        );
        trauma.add(0.20);
        commands.entity(mine_e).despawn();
    }
}
