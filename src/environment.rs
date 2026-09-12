//! Area hazards, surface zones, and prop-death effects. Ported from
//! nt's `game/environment.rs` (data + spawners + sim-half ticks;
//! `animate_environment`'s alpha law lives in [`SurfacePulse::alpha_at`]
//! and applies renderer-side; art paths resolve renderer-side from the
//! recorded [`PulseSprite`]). Hazard tint art likewise resolves
//! renderer-side from the spec; colors below are the exact bevy values
//! as arrays.

use bevy_ecs::prelude::*;
use repame_sim::SimTime;

use crate::anim::SpriteAnim;
use crate::combat::{Explosion, HitFlash};
use crate::comps_a::{
    ARENA_H, ARENA_W, DamageSource, GameCleanup, Health, LevelCleanup, Player, Projectile, Team,
    Velocity,
};
use crate::comps_b::{Dash, Enemy, PickupLifetime, Prop, PropSprites};
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
}

impl PropDeathEffect {
    pub fn toxic_barrel() -> Self {
        Self {
            explosion: Some(ExplosionPayload {
                radius: 95.0,
                damage: 5,
            }),
            hazard: Some(EnvironmentHazardSpec::toxic_barrel()),
        }
    }

    pub fn car() -> Self {
        Self {
            explosion: Some(ExplosionPayload {
                radius: 155.0,
                damage: 8,
            }),
            hazard: None,
        }
    }

    pub fn mine() -> Self {
        Self {
            explosion: Some(ExplosionPayload {
                radius: 118.0,
                damage: 7,
            }),
            hazard: Some(EnvironmentHazardSpec::mine_fire()),
        }
    }

    pub fn legacy_barrel() -> Self {
        Self {
            explosion: Some(ExplosionPayload {
                radius: 110.0,
                damage: 6,
            }),
            hazard: None,
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
        spawn_prop_death_effect(&mut commands, center, Some(mine.payload), false, None);
        trauma.add(0.20);
        commands.entity(mine_e).despawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repame_anim::{AnimCatalog, AtlasDesc};
    use repame_fx::Trauma;

    fn empty_catalog() -> AnimCatalog {
        AnimCatalog::from_json("{}", AtlasDesc {
            size: 128,
            max_pages: 1,
        padding: 0,
        })
        .unwrap()
    }

    #[test]
    fn proximity_mine_detonates_on_approach() {
        let mut world = bevy_ecs::prelude::World::new();
        world.insert_resource(empty_catalog());
        world.insert_resource(Trauma::default());
        let mine = world
            .spawn((
                Prop {
                    size: glam::Vec2::splat(18.0),
                    hp: 2,
                    destructible: true,
                    explosive: false,
                },
                PropSprites {
                    idle: "images/sprMine.png",
                    hurt: "images/sprMine.png",
                    dead: "images/sprMine.png",
                    flip_x: false,
                },
                ProximityMine::default(),
                Pos(glam::Vec2::ZERO),
            ))
            .id();
        world.spawn((Pos(glam::Vec2::new(10.0, 0.0)), Team::Player));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_proximity_mines);
        sched.run(&mut world);
        world.flush();
        assert!(world.get_entity(mine).is_err(), "mine consumed");
        assert!(
            world.resource::<Trauma>().amount > 0.0,
            "detonation adds trauma"
        );
        assert_eq!(
            world.query::<&crate::combat::Explosion>().iter(&world).count(),
            1
        );
    }

    #[test]
    fn presets_match_bevy_values() {
        let toxic = PropDeathEffect::toxic_barrel();
        assert_eq!(
            (
                toxic.explosion.unwrap().radius,
                toxic.explosion.unwrap().damage
            ),
            (95.0, 5)
        );
        assert!(toxic.hazard.is_some());
        assert!(PropDeathEffect::car().hazard.is_none());
        let hz = EnvironmentHazardSpec::mine_fire();
        assert_eq!((hz.radius, hz.tick), (48.0, 0.18));
        assert_eq!(EnvironmentHazardKind::Fire.color(), [1.0, 0.43, 0.10, 0.36]);
    }

    #[test]
    fn hazard_timers_derive_from_spec() {
        let hz = EnvironmentHazard::new(EnvironmentHazardSpec::toxic_barrel());
        assert_eq!(hz.life.duration(), 3.0);
        assert_eq!(hz.damage_tick.duration(), 0.24);
    }

    #[test]
    fn surface_pulse_alpha_law_matches_bevy() {
        // Bevy `animate_environment`: min + (max - min) * (0.5 + 0.5 *
        // sin(now * speed + phase)).
        let subtle = SurfacePulse::subtle(0.0);
        assert_eq!((subtle.speed, subtle.min_alpha, subtle.max_alpha), (1.8, 0.55, 0.86));
        let hazard = SurfacePulse::hazard(0.0);
        assert_eq!((hazard.speed, hazard.min_alpha, hazard.max_alpha), (4.2, 0.52, 0.95));
        // Wave trough/peak hit the bounds exactly.
        let trough_t = (3.0 * std::f32::consts::FRAC_PI_2 - hazard.phase) / hazard.speed;
        assert!((hazard.alpha_at(trough_t) - 0.52).abs() < 1e-5);
        let peak_t = (std::f32::consts::FRAC_PI_2 - hazard.phase) / hazard.speed;
        assert!((hazard.alpha_at(peak_t) - 0.95).abs() < 1e-5);
        // Mid-wave is the midpoint.
        assert!((hazard.alpha_at(-hazard.phase / hazard.speed) - 0.735).abs() < 1e-5);
    }

    #[test]
    fn hazard_spawn_carries_spill_phase_pulse() {
        let mut world = bevy_ecs::prelude::World::new();
        let pos = glam::Vec2::new(100.0, 50.0);
        let mut cmds = world.commands();
        spawn_environment_hazard(&mut cmds, pos, EnvironmentHazardSpec::mine_fire());
        world.flush();
        let pulse = world.query::<&SurfacePulse>().single(&world).unwrap();
        assert!(
            (pulse.phase - (100.0 * 0.013 + 50.0 * 0.009)).abs() < 1e-4,
            "spill phase law, got {}",
            pulse.phase
        );
        assert_eq!((pulse.speed, pulse.min_alpha), (4.2, 0.52));
    }

    #[test]
    fn cobweb_wins_and_point_test_is_aabb() {
        let ice = SurfaceZone {
            kind: SurfaceKind::Ice,
            half_size: glam::Vec2::splat(100.0),
        };
        let web = SurfaceZone {
            kind: SurfaceKind::Cobweb,
            half_size: glam::Vec2::splat(10.0),
        };
        assert!(point_in_zone(
            glam::Vec2::new(5.0, 5.0),
            glam::Vec2::ZERO,
            web.half_size
        ));
        assert!(!point_in_zone(
            glam::Vec2::new(50.0, 0.0),
            glam::Vec2::ZERO,
            web.half_size
        ));
        assert_eq!(
            surface_at_point(
                glam::Vec2::ZERO,
                [(glam::Vec2::ZERO, ice), (glam::Vec2::ZERO, web)]
            ),
            Some(SurfaceKind::Cobweb)
        );
        assert_eq!(
            surface_at_point(glam::Vec2::new(50.0, 0.0), [(glam::Vec2::ZERO, web)]),
            None
        );
    }

    #[test]
    fn surface_velocity_conveyor_and_snare() {
        let dt = 1.0 / 30.0;
        // Ice counteracts base friction: slow bodies speed up toward the
        // 1.28x cap (conveyor feel).
        let fast = surface_velocity(SurfaceKind::Ice, glam::Vec2::new(100.0, 0.0), dt, 0.45, 120.0);
        assert!(
            (fast.length() - 120.0 * 1.28).abs() < 1e-3,
            "ice pins fast bodies to the high cap, got {fast:?}"
        );
        let slow = surface_velocity(SurfaceKind::Ice, glam::Vec2::new(10.0, 0.0), dt, 0.45, 120.0);
        assert!(
            slow.length() > 10.0,
            "ice accelerates slow bodies, got {slow:?}"
        );
        // Cobweb damps and caps low.
        let snared =
            surface_velocity(SurfaceKind::Cobweb, glam::Vec2::new(100.0, 0.0), dt, 0.45, 120.0);
        assert!(
            (snared.length() - 120.0 * 0.52).abs() < 1e-3,
            "cobweb pins to the low cap, got {snared:?}"
        );
    }

    #[test]
    fn apply_surface_effects_conveys_and_skips_dash() {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = repame_sim::SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        world.spawn((
            Pos(glam::Vec2::ZERO),
            SurfaceZone {
                kind: SurfaceKind::Ice,
                half_size: glam::Vec2::splat(100.0),
            },
        ));
        let mut player = Player::default();
        player.speed = 120.0;
        player.speed_mult = 1.0;
        player.friction = 0.45;
        let rider = world
            .spawn((Pos(glam::Vec2::ZERO), Velocity(glam::Vec2::new(10.0, 0.0)), player))
            .id();
        let dasher = world
            .spawn((
                Pos(glam::Vec2::ZERO),
                Velocity(glam::Vec2::new(10.0, 0.0)),
                crate::comps_b::Dash {
                    timer: GTimer::from_seconds(0.2, TimerMode::Once),
                    dir: glam::Vec2::X,
                },
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(apply_surface_effects);
        sched.run(&mut world);
        assert!(
            world.get::<Velocity>(rider).unwrap().0.length() > 10.0,
            "ice conveys the rider"
        );
        assert_eq!(
            world.get::<Velocity>(dasher).unwrap().0,
            glam::Vec2::new(10.0, 0.0),
            "dashing actors skip surface effects"
        );
    }

    fn live_health(hp: i32) -> Health {
        Health {
            hp,
            max: hp,
            invuln: GTimer::from_seconds(0.0, TimerMode::Once),
        }
    }

    #[test]
    fn hazard_tick_damages_marking_player_and_enemy() {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = repame_sim::SimTime::default();
        time.delta_secs = 0.24;
        world.insert_resource(time);
        world.insert_resource(crate::secrets::SecretTriggers::default());
        world.spawn((
            EnvironmentHazard::new(EnvironmentHazardSpec::toxic_barrel()),
            Pos(glam::Vec2::ZERO),
        ));
        let player_e = world
            .spawn((
                Pos(glam::Vec2::new(10.0, 0.0)),
                Team::Player,
                live_health(8),
                Player::default(),
            ))
            .id();
        let enemy_e = world
            .spawn((
                Pos(glam::Vec2::new(20.0, 0.0)),
                Team::Enemy,
                live_health(5),
            ))
            .id();
        let far_e = world
            .spawn((Pos(glam::Vec2::new(500.0, 0.0)), Team::Enemy, live_health(5)))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_environment_hazards);
        sched.run(&mut world);

        assert_eq!(world.get::<Health>(player_e).unwrap().hp, 7);
        assert!(
            !world
                .get::<Health>(player_e)
                .unwrap()
                .invuln
                .is_finished(),
            "player invuln refreshes"
        );
        assert!(world.get::<crate::combat::HitFlash>(player_e).is_some());
        assert_eq!(world.get::<Health>(enemy_e).unwrap().hp, 4);
        assert_eq!(world.get::<Health>(far_e).unwrap().hp, 5);
        let secrets = world.resource::<crate::secrets::SecretTriggers>();
        assert!(secrets.damage_taken_this_floor);
        assert!(!secrets.oasis_eligible);
        assert_eq!(
            world
                .query::<&repame_fx::DamageNumber>()
                .iter(&world)
                .count(),
            2,
            "one floater per victim"
        );
    }

    #[test]
    fn hazard_tick_expires_and_respects_gates() {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = repame_sim::SimTime::default();
        time.delta_secs = 5.0;
        world.insert_resource(time);
        world.insert_resource(crate::secrets::SecretTriggers::default());
        // Short-lived hazard: expires instead of ticking.
        let hz = world
            .spawn((
                EnvironmentHazard::new(EnvironmentHazardSpec::toxic_barrel()),
                Pos(glam::Vec2::ZERO),
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_environment_hazards);
        sched.run(&mut world);
        assert!(world.get_entity(hz).is_err(), "spent hazard despawns");

        // Boiling Veins at/below threshold ignores fire.
        let mut time2 = repame_sim::SimTime::default();
        time2.delta_secs = 0.38;
        world.insert_resource(time2);
        world.spawn((
            EnvironmentHazard::new(EnvironmentHazardSpec::fire_trap()),
            Pos(glam::Vec2::ZERO),
        ));
        let mut veins = Player::default();
        veins.boiling_veins = true;
        veins.veins_threshold = 4;
        let immune = world
            .spawn((
                Pos(glam::Vec2::ZERO),
                Team::Player,
                live_health(4),
                veins,
            ))
            .id();
        sched.run(&mut world);
        assert_eq!(world.get::<Health>(immune).unwrap().hp, 4);
    }

    #[test]
    fn valid_positions_stay_inside_arena() {
        assert!(valid_environment_position(glam::Vec2::ZERO, 8.0));
        assert!(!valid_environment_position(
            glam::Vec2::new(crate::comps_a::ARENA_W, 0.0),
            8.0
        ));
    }

    #[test]
    fn fog_scroll_advances_half_per_timescale_step_and_wraps() {
        // GML TopCont/Draw_0: fogscroll += timescale * 0.5, wrap at 480.
        assert_eq!(fog_scroll_step(0.0, 1.0), 0.5);
        assert_eq!(fog_scroll_step(10.0, 2.0), 11.0);
        assert!((fog_scroll_step(479.8, 1.0) - 0.3).abs() < 1e-4);
        assert!((fog_scroll_step(0.0, 960.0) - 0.0).abs() < 1e-3);
    }

    #[test]
    fn fog_tick_freezes_while_paused() {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        world.insert_resource(FogState::default());
        world.init_resource::<crate::state::Paused>();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_fog);
        sched.run(&mut world);
        assert!((world.resource::<FogState>().scroll - 0.5).abs() < 1e-6);
        world.resource_mut::<crate::state::Paused>().0 = true;
        sched.run(&mut world);
        assert!((world.resource::<FogState>().scroll - 0.5).abs() < 1e-6);
        // Missing resource: no-op, not a panic (schedule-safe).
        world.remove_resource::<FogState>();
        sched.run(&mut world);
    }
}
