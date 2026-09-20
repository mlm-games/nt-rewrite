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
    ARENA_H, ARENA_W, DamageSource, GameCleanup, Health, LevelCleanup, Player, Projectile, Team,
    Velocity,
};
use crate::comps_b::{
    Dash, Enemy, FxAngle, Mote, MoteScale, MoteStrip, PickupLifetime, Prop, PropSprites,
};
use crate::enemy_data::enemy_def;
use crate::secrets::SecretTriggers;
use crate::spatial::Pos;
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
            duration: 12.0,
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

            if is_player
                && hazard.spec.kind == EnvironmentHazardKind::Fire
                && let Some(player) = player
                && player.boiling_veins
                && health.hp <= player.veins_threshold
            {
                continue;
            }

            health.hp -= hazard.spec.damage;

            if is_player {
                health.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                secrets.mark_damage_taken();
            }

            HitFlash::apply(&mut commands, target_entity, hazard.spec.kind.color(), 0.08);
            repame_fx::spawn_number(
                &mut commands,
                target_pos.0.x,
                target_pos.0.y,
                format!("{}", hazard.spec.damage),
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
    if let Some(def) = catalog.def(sprites.dead) {
        ec.insert(SpriteAnim::oneshot(sprites.dead, def));
    }
    // Bevy expiry: corpses rot after 12 s (`PickupLifetime` doubles as
    // the corpse query shape).
    ec.insert(PickupLifetime {
        timer: GTimer::from_seconds(12.0, TimerMode::Once),
    });
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

        let mut rng = rand::rng();
        crate::effects::spawn_burst(
            &mut commands,
            &mut rng,
            pos,
            24,
            [1.0, 0.52, 0.16, 1.0],
            (100.0, 360.0),
        );
    }

    if let Some(hazard) = effect.hazard {
        spawn_environment_hazard(commands, pos, hazard);
    }

    if effect.ground_flames > 0 {
        let mut rng = rand::rng();
        for _ in 0..effect.ground_flames {
            let off = glam::Vec2::new(
                rng.random_range(-24.0..24.0),
                rng.random_range(-24.0..24.0),
            );
            spawn_environment_hazard(commands, pos + off, EnvironmentHazardSpec::ground_flame());
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

/// GML `Dust`/`Smoke`/`Feather`/`Curse` spawner (sim half): one
/// `GroundPhysics` + `FxAngle` + `Mote` + `MoteScale` + `SpriteAnim`
/// entity per mote, `PickupLifetime` expiry. Speed/friction/spin/grow
/// laws are verbatim from the `Create_0` handlers (speeds converted
/// px/step → px/s at 30 Hz); the step integration lives in
/// [`tick_motes`].
#[allow(clippy::too_many_arguments)]
pub fn spawn_motes(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
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
        let (speed_lo, speed_hi, friction, life, scale0, spin_max, sway) = match strip {
            MoteStrip::Dust => (0.0, 2.0, 0.3, 0.6, 0.7, 4.0, false),
            MoteStrip::Smoke => (0.5, 1.5, 0.1, 2.0, 0.8, 4.0, false),
            MoteStrip::Leaf | MoteStrip::Money | MoteStrip::Raven => {
                (1.8, 3.0, 0.0, 5.5, 1.0, 3.0, true)
            }
            MoteStrip::Curse => (0.2, 0.7, 0.005, 3.0, 1.0, 0.0, false),
        };
        let a = rng.random_range(0.0..std::f32::consts::TAU);
        let dir = glam::Vec2::new(a.cos(), a.sin());
        let mut vel = dir * rng.random_range(speed_lo..speed_hi) * 30.0;
        if sway {
            vel.y -= rng.random_range(0.9..1.2) * 30.0;
        }
        let spin = if spin_max > 0.0 {
            rng.random_range(1.0..spin_max) * if rng.random_bool(0.5) { 1.0 } else { -1.0 }
        } else {
            0.0
        };
        let grow = match strip {
            MoteStrip::Dust => rng.random_range(0.05..0.10),
            MoteStrip::Smoke => rng.random_range(0.0..0.005),
            _ => 0.0,
        };
        let grow_decay = match strip {
            MoteStrip::Dust => 0.02,
            MoteStrip::Smoke => 0.001,
            _ => 0.0,
        };
        let path: &'static str = match strip {
            MoteStrip::Dust => "images/sprDust.png",
            MoteStrip::Smoke => "images/sprSmoke.png",
            MoteStrip::Leaf => "images/sprLeaf.png",
            MoteStrip::Money => "images/sprMoney.png",
            MoteStrip::Raven => "images/sprRavenFeather.png",
            MoteStrip::Curse => "images/sprCurse.png",
        };
        let mut ec = commands.spawn((
            GameCleanup,
            LevelCleanup,
            crate::comps_b::GroundPhysics {
                vel,
                rotspeed: rng.random_range(0.7..1.0)
                    * if rng.random_bool(0.5) { 1.0 } else { -1.0 },
            },
            FxAngle(rng.random_range(0.0..std::f32::consts::TAU)),
            Mote {
                friction,
                spin,
                grow,
                grow_decay,
                sway,
            },
            MoteScale(scale0),
            strip,
            Pos(pos),
            PickupLifetime {
                timer: GTimer::from_seconds(life, TimerMode::Once),
            },
        ));
        if let Some(def) = catalog.def(path) {
            let mut anim = SpriteAnim::new(path, def);
            anim.frame = rng.random_range(0..def.frames.max(1));
            ec.insert(anim);
        }
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
    mut q: Query<(
        Entity,
        &mut Pos,
        &mut crate::comps_b::GroundPhysics,
        &mut FxAngle,
        &mut Mote,
        &mut MoteScale,
    ), Without<crate::comps_a::WallTile>>,
    walls: Query<(&crate::comps_a::WallCell, &Pos), With<crate::comps_a::WallTile>>,
    _frame: Res<crate::state::CurrentFrame>,
) {
    let dt = time.delta_secs;
    let wall_cells: Vec<((i32, i32), glam::Vec2)> =
        walls.iter().map(|(c, p)| ((c.0, c.1), p.0)).collect();
    let bounce = |pos: glam::Vec2, vel: &mut glam::Vec2, keep: f32| {
        for ((cx, cy), wp) in &wall_cells {
            let half = 8.0;
            let closest = glam::Vec2::new(
                pos.x.clamp(wp.x - half, wp.x + half),
                pos.y.clamp(wp.y - half, wp.y + half),
            );
            if pos.distance(closest) > 4.0 {
                continue;
            }
            let _ = (cx, cy);
            if (pos.x - closest.x).abs() > (pos.y - closest.y).abs() {
                vel.x = -vel.x * keep;
            } else {
                vel.y = -vel.y * keep;
            }
            break;
        }
    };
    for (e, mut pos, mut ground, mut angle, mut mote, mut scale) in &mut q {
        if mote.sway {
            pos.0.x += 0.35 * (30.0 * dt).sin() * 30.0 * dt;
            pos.0.y += 0.3 * 30.0 * dt;
            let sp = ground.vel.length();
            if sp > 0.2 * 30.0 {
                ground.vel *= 0.9_f32.powf(dt * 30.0);
            } else {
                ground.vel = glam::Vec2::ZERO;
            }
            bounce(pos.0, &mut ground.vel, 0.5);
        } else if mote.friction > 0.0 {
            crate::comps_a::apply_gml_friction(&mut ground.vel, mote.friction, dt);
        }
        pos.0 += ground.vel * dt;
        angle.0 += mote.spin * dt;
        if mote.grow_decay > 0.0 {
            scale.0 += mote.grow * dt * 30.0;
            mote.grow -= mote.grow_decay * dt * 30.0;
            if scale.0 < 0.0 {
                commands.entity(e).despawn();
            }
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
    catalog: &repame_anim::AnimCatalog,
    particles_on: bool,
    pos: glam::Vec2,
    small: bool,
) {
    if !particles_on {
        return;
    }
    let mut rng = rand::rng();
    let dust_n = if small { 8 } else { 20 };
    let smoke_n = dust_n / 2;
    for _ in 0..smoke_n {
        spawn_motes(commands, catalog, true, pos, MoteStrip::Smoke, 1);
    }
    let start = rng.random_range(0.0..std::f32::consts::TAU);
    for i in 0..dust_n {
        let a = start + i as f32 * std::f32::consts::TAU / dust_n as f32;
        let dir = glam::Vec2::new(a.cos(), a.sin());
        let mut ec = commands.spawn((
            GameCleanup,
            LevelCleanup,
            crate::comps_b::GroundPhysics {
                vel: dir * 6.0 * 30.0,
                rotspeed: rng.random_range(0.7..1.0)
                    * if rng.random_bool(0.5) { 1.0 } else { -1.0 },
            },
            FxAngle(rng.random_range(0.0..std::f32::consts::TAU)),
            Mote {
                friction: 0.3,
                spin: rng.random_range(1.0..4.0)
                    * if rng.random_bool(0.5) { 1.0 } else { -1.0 },
                grow: rng.random_range(0.05..0.10),
                grow_decay: 0.02,
                sway: false,
            },
            MoteScale(0.7),
            MoteStrip::Dust,
            Pos(pos),
            PickupLifetime {
                timer: GTimer::from_seconds(0.6, TimerMode::Once),
            },
        ));
        if let Some(def) = catalog.def("images/sprDust.png") {
            let mut anim = SpriteAnim::new("images/sprDust.png", def);
            anim.frame = rng.random_range(0..def.frames.max(1));
            ec.insert(anim);
        }
    }
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

