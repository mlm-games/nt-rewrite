//! Screen/game feel sinks: flash, trauma, rumble, particle bursts.
//!
//! Split by ownership: `Trauma` reuses `repame_fx` verbatim (its `add`
//! matches nt's law exactly); `FlashWhite` ports verbatim instead —
//! `repame_fx::Flash` counts in ticks while nt's flash is seconds-based,
//! and the renderer reads `amount` directly. Bursts spawn
//! `repame_fx::Particle`s with nt's exact size/lifetime dice.

use bevy_ecs::prelude::*;
use glam::Vec2;
use repame_fx::{DamageNumber, EaseKind, Flash, Gradient, Particle, Trauma, step_numbers, step_particles};
use repame_sim::SimTime;

use crate::msg::Queue;
use crate::time::{GTimer, TimerMode};

/// Fullscreen white flash (game-utils `FlashWhite` parity).
#[derive(Resource, Clone, Debug, Default)]
pub struct FlashWhite {
    pub amount: f32,
    pub timer: GTimer,
}

/// Fire a white flash for `duration` seconds.
pub fn flash_white(flash: &mut FlashWhite, duration: f32) {
    flash.amount = 1.0;
    flash.timer = GTimer::from_seconds(duration, TimerMode::Once);
}

/// Rumble request for the platform layer (all pads; nt is single-player
/// so per-pad targeting from the bevy build is intentionally dropped).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RumbleRequest {
    pub weak: f32,
    pub strong: f32,
    pub duration_secs: f32,
}

/// Queue a rumble (game-utils `GameFeel::rumble_controller` parity,
/// minus per-gamepad routing).
pub fn rumble(queue: &mut Queue<RumbleRequest>, weak: f32, strong: f32, duration_secs: f32) {
    queue.push(RumbleRequest {
        weak: weak.clamp(0.0, 1.0),
        strong: strong.clamp(0.0, 1.0),
        duration_secs: duration_secs.max(0.0),
    });
}

/// One-shot colored burst (game-utils `VfxSpawner::spawn_burst`
/// parity): `count` dots, uniform directions, `speed_range` px/s,
/// 3–7 px size, 0.4–0.9 s life. `rng` is caller-supplied so tests
/// seed it (the bevy build used thread rng).
pub fn spawn_burst(
    commands: &mut Commands,
    rng: &mut impl rand::RngExt,
    pos: glam::Vec2,
    count: usize,
    color: [f32; 4],
    speed_range: (f32, f32),
) {
    use std::f32::consts::TAU;
    for _ in 0..count {
        let angle = rng.random_range(0.0..TAU);
        let speed = rng.random_range(speed_range.0..speed_range.1);
        commands.spawn(Particle {
            pos: [pos.x, pos.y],
            vel: [angle.cos() * speed, angle.sin() * speed],
            age_ticks: 0,
            life_ticks: (rng.random_range(0.4..0.9) * 30.0) as i32,
            size_px: rng.random_range(3.0..7.0),
            gravity_pps2: 0.0,
            drag_per_sec: 0.0,
            gradient: Gradient::solid(color),
            ease: EaseKind::Linear,
            spawner: None,
            page: 0,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
        });
    }
}

/// Muzzle-flash marker for the render phase. No `MuzzleFlash` comp or
/// `FiredWeapon` marker existed: firing only spawned yellow
/// [`Particle`] bursts via `muzzle_burst`, which have no quad mapping.
/// `spawn_pellets` spawns one of these per volley (pos = muzzle origin,
/// dir = shot aim); [`tick_fired_weapons`] expires it after
/// [`FIRED_WEAPON_TICKS`] sim ticks so the renderer can draw a short
/// muzzle tongue without touching firing logic.
pub const FIRED_WEAPON_TICKS: u8 = 3;

#[derive(Component, Clone, Copy, Debug)]
pub struct FiredWeapon {
    pub pos: Vec2,
    pub dir: Vec2,
    pub ticks_left: u8,
}

impl FiredWeapon {
    pub fn new(pos: Vec2, dir: Vec2) -> Self {
        Self {
            pos,
            dir: dir.normalize_or_zero(),
            ticks_left: FIRED_WEAPON_TICKS,
        }
    }
}

/// Expire lapsed muzzle markers (Always tail; ungated like
/// `tick_hit_flash` so stranded markers drain even off-gameplay).
pub fn tick_fired_weapons(mut commands: Commands, mut q: Query<(Entity, &mut FiredWeapon)>) {
    for (e, mut m) in &mut q {
        m.ticks_left = m.ticks_left.saturating_sub(1);
        if m.ticks_left == 0 {
            commands.entity(e).despawn();
        }
    }
}

/// Step tick-based FX state once per sim tick (Always tail):
/// [`Particle`] integrate + despawn, [`DamageNumber`] rise + despawn,
/// [`Trauma`] decay, and [`FlashWhite`] fade-out.
///
/// `repame-fx` steps on 100 Hz ticks while the sim runs 30 Hz, so
/// `ticks = round(delta * 100)` (3 per steady tick; 0 pauses, matching
/// the `ticks <= 0` no-op in the fx step fns). `Trauma` previously
/// never decayed sim-side and `FlashWhite.timer` was never ticked, so
/// both stuck at peak until now; this is the bevy `2d_screen_shake`
/// 1.5/s decay and the game-utils flash fade.
pub fn step_fx(
    time: Res<SimTime>,
    mut commands: Commands,
    mut trauma: Option<ResMut<Trauma>>,
    mut flash: Option<ResMut<FlashWhite>>,
    mut particles: Query<(Entity, &mut Particle)>,
    mut numbers: Query<(Entity, &mut DamageNumber)>,
) {
    let dt = time.delta_secs;
    let ticks = (dt * 100.0).round() as i32;
    if let Some(ref mut tr) = trauma {
        tr.decay(ticks);
    }
    if let Some(ref mut fl) = flash {
        fl.timer.tick(dt);
        if fl.timer.is_finished() {
            fl.amount = 0.0;
        }
    }
    step_particles(&mut commands, &mut particles, ticks);
    step_numbers(&mut commands, &mut numbers, ticks);
}

/// Camera shake offset for the view layer: [`Trauma::offset`] at the
/// sim clock. Thin getter over the `repame-fx` resources registered by
/// [`repame_fx::init_resources`] (`Trauma`) + [`SimTime`]; zeros when
/// either is absent (e.g. unit-test worlds).
pub fn trauma_offset(world: &World) -> (f32, f32, f32) {
    let t = world
        .get_resource::<SimTime>()
        .map(|s| s.elapsed_secs as f32)
        .unwrap_or(0.0);
    world
        .get_resource::<Trauma>()
        .map(|tr| tr.offset(t))
        .unwrap_or((0.0, 0.0, 0.0))
}

/// Fullscreen overlay color for the view layer. Prefers the sim-side
/// [`FlashWhite`] (seconds-based, the only flash the sim triggers);
/// falls back to the `repame-fx` [`Flash`] resource (tick-based, from
/// [`repame_fx::init_resources`]) so engine-driven flashes still
/// resolve. `None` when idle so the view can skip the overlay write.
pub fn flash_rgba(world: &World) -> Option<[f32; 4]> {
    if let Some(fw) = world.get_resource::<FlashWhite>()
        && fw.amount > 0.0
    {
        return Some([1.0, 1.0, 1.0, fw.amount.clamp(0.0, 1.0)]);
    }
    world.get_resource::<Flash>().and_then(|f| f.rgba())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_fires_and_rumble_queues() {
        let mut flash = FlashWhite::default();
        flash_white(&mut flash, 0.2);
        assert_eq!(flash.amount, 1.0);
        assert_eq!(flash.timer.duration(), 0.2);
        let mut q: Queue<RumbleRequest> = Queue::default();
        rumble(&mut q, 0.2, 0.8, 0.16);
        let got = q.drain();
        assert_eq!(
            got,
            vec![RumbleRequest {
                weak: 0.2,
                strong: 0.8,
                duration_secs: 0.16,
            }]
        );
    }

    #[test]
    fn burst_spawns_ranged_dots() {
        let mut world = bevy_ecs::prelude::World::new();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(|mut commands: Commands| {
            let mut rng = rand::rng();
            spawn_burst(
                &mut commands,
                &mut rng,
                glam::Vec2::new(10.0, 20.0),
                12,
                [1.0, 0.1, 0.08, 1.0],
                (80.0, 220.0),
            );
        });
        schedule.run(&mut world);
        let mut q = world.query::<&Particle>();
        assert_eq!(q.iter(&world).count(), 12);
        for p in q.iter(&world) {
            assert!(
                p.life_ticks >= 12 && p.life_ticks <= 27,
                "got {}",
                p.life_ticks
            );
            let speed = glam::Vec2::from(p.vel).length();
            assert!((80.0..=220.0).contains(&speed), "got {speed}");
        }
    }

    fn fx_world() -> bevy_ecs::prelude::World {
        let mut world = bevy_ecs::prelude::World::new();
        world.insert_resource(SimTime {
            delta_secs: 1.0 / 30.0,
            ..Default::default()
        });
        world.init_resource::<Trauma>();
        world.init_resource::<FlashWhite>();
        world
    }

    #[test]
    fn muzzle_expires_after_three_ticks() {
        let mut world = fx_world();
        world.spawn(FiredWeapon::new(
            glam::Vec2::new(10.0, 20.0),
            glam::Vec2::X,
        ));
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(tick_fired_weapons);
        schedule.run(&mut world);
        schedule.run(&mut world);
        assert_eq!(
            world.query::<&FiredWeapon>().iter(&world).count(),
            1,
            "still live after 2 ticks"
        );
        schedule.run(&mut world);
        assert_eq!(
            world.query::<&FiredWeapon>().iter(&world).count(),
            0,
            "expired after {FIRED_WEAPON_TICKS} ticks"
        );
    }

    #[test]
    fn step_fx_ages_particles_numbers_and_fades_feel() {
        use repame_fx::{Gradient, spawn_number};

        let mut world = fx_world();
        world.resource_mut::<Trauma>().add(1.0);
        {
            let mut flash = world.resource_mut::<FlashWhite>();
            flash_white(&mut flash, 0.1);
        }
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems((
            |mut commands: Commands| {
                commands.spawn(Particle {
                    pos: [0.0, 0.0],
                    vel: [0.0, 0.0],
                    age_ticks: 0,
                    life_ticks: 60,
                    size_px: 4.0,
                    gravity_pps2: 0.0,
                    drag_per_sec: 0.0,
                    gradient: Gradient::solid([1.0, 0.0, 0.0, 1.0]),
                    ease: EaseKind::Linear,
                    spawner: None,
                    page: 0,
                    uv_min: [0.0, 0.0],
                    uv_max: [1.0, 1.0],
                });
                spawn_number(&mut commands, 5.0, 5.0, "3", [1.0, 1.0, 1.0, 1.0]);
            },
            step_fx,
        ));
        // First pass only flushes the deferred spawns (`step_fx` runs
        // before they apply); the solo pass below does the stepping.
        schedule.run(&mut world);
        let mut solo = bevy_ecs::schedule::Schedule::default();
        solo.add_systems(step_fx);
        solo.run(&mut world);
        // 1/30 s -> 3 fx ticks: particle aged, number rose, trauma decayed.
        let p = world.query::<&Particle>().iter(&world).next().unwrap();
        assert_eq!(p.age_ticks, 3);
        let n = world
            .query::<&DamageNumber>()
            .iter(&world)
            .next()
            .unwrap();
        assert!(n.y < 5.0, "numbers rise, y = {}", n.y);
        assert!(
            world.resource::<Trauma>().amount < 1.0,
            "trauma decays sim-side"
        );
        // Flash still live at 1/30 s < 0.05 s; run past it and it clears.
        assert!(flash_rgba(&world).is_some());
        for _ in 0..4 {
            solo.run(&mut world);
        }
        assert_eq!(world.resource::<FlashWhite>().amount, 0.0);
        assert_eq!(flash_rgba(&world), None);
    }

    #[test]
    fn trauma_offset_nonzero_after_add_and_zero_without() {
        let empty = bevy_ecs::prelude::World::new();
        assert_eq!(trauma_offset(&empty), (0.0, 0.0, 0.0));
        let mut world = fx_world();
        assert_eq!(trauma_offset(&world), (0.0, 0.0, 0.0));
        world.resource_mut::<Trauma>().add(1.0);
        // Elapsed 0 hits a Perlin zero is possible; advance the clock to
        // a nonzero sample point like the view layer would see.
        world.resource_mut::<SimTime>().elapsed_secs = 1.23;
        let (dx, dy, _) = trauma_offset(&world);
        assert!(
            dx.abs() + dy.abs() > 0.0,
            "full trauma shakes the camera"
        );
    }
}

/// Hit-stop dip state (game-utils `HitStop` parity).
#[derive(Resource, Clone, Debug)]
pub struct HitStop {
    /// Whether a dip is currently active.
    pub active: bool,
    /// Virtual-time speed factor while recovering.
    pub scale: f32,
    /// Recovery timer (real time, unaffected by the dip itself).
    pub recover: crate::time::GTimer,
    /// Initial dip scale applied immediately on trigger.
    pub start_scale: f32,
}

impl Default for HitStop {
    fn default() -> Self {
        Self {
            active: false,
            scale: 1.0,
            recover: crate::time::GTimer::disarmed(),
            start_scale: 1.0,
        }
    }
}

impl HitStop {
    pub fn trigger(&mut self, scale: f32, recover_secs: f32) {
        use crate::time::TimerMode;
        // Bevy `HitStop::trigger` weakest-wins guard: a live stop keeps
        // the strongest dip (lowest scale); weaker triggers defer.
        let scale = scale.clamp(0.01, 1.0);
        if self.active && scale >= self.scale {
            return;
        }
        self.start_scale = scale;
        self.scale = self.start_scale;
        self.recover = crate::time::GTimer::from_seconds(recover_secs.max(0.0), TimerMode::Once);
        self.active = true;
    }

    pub fn cancel(&mut self) {
        self.active = false;
        self.scale = 1.0;
    }
}

/// Slow-motion state (game-utils `SlowMotion` parity).
#[derive(Resource, Clone, Debug)]
pub struct SlowMotion {
    pub active: bool,
    pub scale: f32,
    pub timer: crate::time::GTimer,
}

impl Default for SlowMotion {
    fn default() -> Self {
        Self {
            active: false,
            scale: 1.0,
            timer: crate::time::GTimer::disarmed(),
        }
    }
}

/// Engage slow motion (game-utils `GameFeel::slow_motion` parity).
pub fn slow_motion(slow_mo: &mut SlowMotion, scale: f32, duration_secs: f32) {
    use crate::time::TimerMode;
    slow_mo.scale = scale.clamp(0.01, 1.0);
    slow_mo.timer = crate::time::GTimer::from_seconds(duration_secs.max(0.0), TimerMode::Once);
    slow_mo.active = true;
}

/// Chromatic aberration strength (game-utils parity).
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct ChromaticAberration(pub f32);

/// Raise the aberration floor (never lowers — decay is renderer-side).
pub fn chromatic_pulse(chrom: &mut ChromaticAberration, strength: f32) {
    chrom.0 = chrom.0.max(strength);
}
