//! Vortex spiral sim state. Headless port of the sim half of the bevy
//! reference `game/vortex.rs`: the [`SpiralKind`] enum, the [`SpiralCtl`]
//! angle-advance law ([`spiral_angle_inc`] + the per-tick emission), and
//! the snapshot the background pass consumes.
//!
//! Everything bevy-render stays out: no `Handle<Image>`, no materials, no
//! `sync_spiral_cpu_layer` GPU writes, no plugin. Star/vard dots and the
//! retired-entity list were renderer-side (`ChildOf(camera)` entities), so
//! stars/vards survive here as data-only (positions still integrate, which
//! keeps the shared RNG stream and debris spawn cadence identical) while
//! the entity handles are dropped.
//!
//! Timing is [`repame_sim::SimTime`] driven: [`tick_spiral`] advances the
//! control at the GML 30 Hz cadence (`dt * 30`), exactly like the bevy
//! `vortex_tick` system. Area selection goes through [`gml_area_for_area`]
//! (bevy `gml_area_for_bevy_area` by variant name) into
//! [`SpiralKind::for_gml_area`]; `AreaId::Loop` maps to GML area 1, i.e.
//! `Normal` — there is no loop-count branch in the reference.

use bevy_ecs::prelude::*;
use repame_sim::SimTime;
use crate::vortex_pass::{VORTEX_DEBRIS, VORTEX_WISPS, VortexSnapshot};

use crate::data::AreaId;

/// Wisp ring slots (matches `VORTEX_WISPS`; the shader indexes slots by
/// `(birth-1) % N`).
pub const MAX_WISPS: usize = VORTEX_WISPS;
/// Deterministic-stream salts (head-start, angle, per-tick rate).
const STREAM_HEAD_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
const STREAM_ANGLE_SALT: u64 = 0xBF58_476D_1CE4_E5B9;
const STREAM_RATE_SALT: u64 = 0x94D0_49BB_1331_11EB;
/// Debris ring slots (matches `VORTEX_DEBRIS`).
pub const MAX_DEBRIS: usize = VORTEX_DEBRIS;
/// Warmup ticks so a freshly spawned spiral is already full.
const WARMUP_TICKS: u32 = 150;
/// Ticks after death before the spiral is done (bevy
/// `despawn_vortex_when_done` drain constant).
pub const DRAIN_TICKS: f32 = 26.0;

/// GUI-space size the spiral laws are written in (bevy `ui_art` values).
pub const GUI_W: f32 = 320.0;
pub const GUI_H: f32 = 240.0;

/// Look center + visible extent in wisp coord space. Bevy parity is the 6x
/// quad centered on the camera.
pub const VORTEX_VIEW: [f32; 4] = [160.0, 120.0, 1920.0, 1440.0];

/// Spiral visual variant, selected by GML area.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SpiralKind {
    #[default]
    Normal,
    Proto,
    Idpd,
    Venuz,
}

impl SpiralKind {
    /// GML area -> variant table (bevy `SpiralKind::for_gml_area`,
    /// verbatim, including Jungle 105 falling through to `Normal` with
    /// the crystal flag carried separately in `kindpacked`).
    pub fn for_gml_area(area: u8) -> Self {
        match area {
            100 => Self::Proto,
            106 => Self::Idpd,
            103 | 107 => Self::Venuz,
            _ => Self::Normal,
        }
    }
}

/// Target `AreaId` -> GML area int (bevy `gml_area_for_bevy_area`, matched
/// by variant name; target discriminants differ so this is NOT `as u8`).
pub fn gml_area_for_area(area: AreaId) -> u8 {
    match area {
        AreaId::Campfire => 0,
        AreaId::Desert => 1,
        AreaId::Sewers => 2,
        AreaId::Scrapyards => 3,
        AreaId::CrystalCaves => 4,
        AreaId::FrozenCity => 5,
        AreaId::Labs => 6,
        AreaId::Palace => 7,
        AreaId::Vault => 100,
        AreaId::Oasis => 101,
        AreaId::PizzaSewers => 102,
        AreaId::City => 103,
        AreaId::CursedCaves => 104,
        AreaId::Jungle => 105,
        AreaId::HQ => 106,
        AreaId::CrownVault => 100,
        AreaId::Loop => 1,
    }
}

/// Area -> spiral variant (area/loop selection: `Loop` rides GML area 1,
/// hence `Normal`, exactly like the reference).
pub fn kind_for_area(area: AreaId) -> SpiralKind {
    SpiralKind::for_gml_area(gml_area_for_area(area))
}

/// Area-flavoured debris sprite pick (bevy `variant_debris_for_gml_area`,
/// verbatim; the path is data here, upload stays renderer-side).
fn variant_debris_for_gml_area(area: u8) -> Option<(&'static str, usize)> {
    match area {
        1 => Some(("images/sprBanditHurt.png", 1)),
        2 => Some(("images/sprRatHurt.png", 1)),
        3 => Some(("images/sprCarIdle.png", 1)),
        4 => Some(("images/sprSpiderHurt.png", 1)),
        5 => Some(("images/sprFrozenCar.png", 1)),
        6 => Some(("images/sprFreak1Hurt.png", 1)),
        102 => Some(("images/sprSlice.png", 1)),
        _ => None,
    }
}

/// Venuz starfield mote (data-only; the bevy entity handle was render).
#[derive(Clone, Copy, Debug)]
pub struct Star {
    pub alive: bool,
    pub xstart: f32,
    pub ystart: f32,
    pub dist: f32,
    pub angle: f32,
    pub grow: f32,
    pub xscale: f32,
    pub frame: f32,
}

/// Area-flavoured debris mote (data-only; the bevy entity handle was render).
#[derive(Clone, Copy, Debug)]
pub struct Vard {
    pub alive: bool,
    pub xstart: f32,
    pub ystart: f32,
    pub dist: f32,
    pub angle: f32,
    pub turnspeed: f32,
    pub rotspeed: f32,
    pub grow: f32,
    pub xscale: f32,
    pub image_angle: f32,
    pub path: &'static str,
    pub frame: usize,
}

/// Per-wisp lightning-stream state (GML `Spiral/Create_0` + `Step_0` +
/// `scrDrawSpiral`): `lanim` starts at `-random(300)` and grows
/// `0.2 + random(0.3)` per tick; `langle` is the `random_angle`
/// bolt offset. `lanim in (0, 6)` shows the `sprPortalLightning`
/// stream at rotation `image_angle + langle`. The GML dice are a
/// deterministic splitmix stream over `(seed, birth, tick)` so
/// snapshots stay contract-stable across runs with the same seed.
/// `langle` is radians; dead slots hold `lanim = -1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WispStream {
    pub lanim: f32,
    pub langle: f32,
}

impl WispStream {
    pub fn dead() -> Self {
        Self {
            lanim: -1.0,
            langle: 0.0,
        }
    }

    /// True while the bolt shows (GML `lanim > 0 && lanim < 6`).
    pub fn bolt_visible(&self) -> bool {
        self.lanim > 0.0 && self.lanim < 6.0
    }
}

/// Deterministic [0, 1) draw (splitmix64 over seed/birth/tick/salt:
/// GML `random()` replacement — stable across platforms and `rand`
/// versions, so seeded runs snapshot identically).
fn stream_hash01(seed: u64, birth: u32, tick: u32, salt: u64) -> f32 {
    let mut z = seed
        .wrapping_add((birth as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((tick as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(salt.wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f32) / 9_007_199_254_740_992.0
}

/// Ribbon debris mote (feeds `debris_ring`, hence the snapshot).
#[derive(Clone, Copy, Debug)]
pub struct Debris {
    pub alive: bool,
    pub xstart: f32,
    pub ystart: f32,
    pub dist: f32,
    pub angle: f32,
    pub turnspeed: f32,
    pub rotspeed: f32,
    pub xscale: f32,
    pub grow: f32,
    pub image_angle: f32,
    pub frame: f32,
}

/// Headless spiral control (bevy `SpiralCtl` minus the render-only
/// `retired` entity list; `head`/`dhead` are `pub` here so the write-only
/// ring cursors never trip dead-code lints outside test builds).
#[derive(Resource, Debug)]
pub struct SpiralCtl {
    pub angle: f32,
    pub ticks: f32,
    acc: f32,
    pub ring: Vec<[f32; 4]>,
    pub head: usize,
    debris: Vec<Debris>,
    pub debris_ring: Vec<[f32; 4]>,
    pub dhead: usize,
    pub stars: Vec<Star>,
    pub vards: Vec<Vard>,
    pub alive: bool,
    pub death_tick: Option<f32>,
    pub drain_bias: f32,
    pub kind: SpiralKind,
    pub gml_area: u8,
    /// Deterministic-stream seed (run seed; snapshots with equal seeds
    /// are bit-identical).
    pub seed: u64,
    /// Per-ring-slot lightning streams, indexed exactly like `ring`.
    pub streams: Vec<WispStream>,
}

impl SpiralCtl {
    pub fn warmed_up() -> Self {
        Self::warmed_up_for_gml_area(0)
    }

    pub fn warmed_up_for_area(area: AreaId) -> Self {
        Self::warmed_up_for_gml_area(gml_area_for_area(area))
    }

    /// Seeded warmup: the lightning streams roll from `seed` (run
    /// seed), so equal seeds snapshot identically. Spiral angle and
    /// debris stay `rand`-driven as before — only streams are seeded.
    pub fn warmed_up_for_area_seeded(area: AreaId, seed: u64) -> Self {
        Self::warmed_up_for_gml_area_seeded(gml_area_for_area(area), seed)
    }

    pub fn warmed_up_for_gml_area(gml_area: u8) -> Self {
        Self::warmed_up_for_gml_area_seeded(gml_area, 0)
    }

    pub fn warmed_up_for_gml_area_seeded(gml_area: u8, seed: u64) -> Self {
        let mut ctl = Self {
            angle: rand::random::<f32>() * 360.0,
            ticks: 0.0,
            acc: 0.0,
            ring: vec![[-1.0; 4]; MAX_WISPS],
            head: 0,
            debris: (0..MAX_DEBRIS)
                .map(|_| Debris {
                    alive: false,
                    xstart: 0.0,
                    ystart: 0.0,
                    dist: 0.0,
                    angle: 0.0,
                    turnspeed: 0.0,
                    rotspeed: 0.0,
                    xscale: 0.0,
                    grow: 0.0,
                    image_angle: 0.0,
                    frame: 0.0,
                })
                .collect(),
            debris_ring: vec![[-1000.0; 4]; MAX_DEBRIS],
            dhead: 0,
            stars: Vec::new(),
            vards: Vec::new(),
            alive: true,
            death_tick: None,
            drain_bias: 0.0,
            kind: SpiralKind::for_gml_area(gml_area),
            gml_area,
            seed,
            streams: vec![WispStream::dead(); MAX_WISPS],
        };
        for _ in 0..WARMUP_TICKS {
            ctl.tick_once();
        }
        ctl
    }

    /// Re-warm under a new stream seed (run seed). Same seed + same
    /// ticks = identical streams and snapshot.
    pub fn with_seed(self, seed: u64) -> Self {
        Self::warmed_up_for_gml_area_seeded(self.gml_area, seed)
    }

    /// Mark the spiral dead (bevy `mark_vortex_dead` / `teardown_vortex`):
    /// births freeze, `drain_bias` fast-forwards the scale age instead.
    pub fn kill(&mut self) {
        if self.alive {
            self.alive = false;
            self.death_tick = Some(self.ticks);
        }
    }

    /// Drain finished (bevy `despawn_vortex_when_done` gate, minus the
    /// state-gated despawn which stays shell-side).
    pub fn is_done(&self) -> bool {
        if self.alive {
            return false;
        }
        match self.death_tick {
            Some(death) => self.ticks - death >= DRAIN_TICKS,
            None => false,
        }
    }

    fn tick_once(&mut self) {
        self.ticks += 1.0;
        // GML `Spiral/Step_0` bolt clock first: live wisps advance
        // before this tick's birth runs, so a newborn's `Create_0`
        // head-start takes its first `Step_0` increment next tick.
        for slot in 0..MAX_WISPS {
            if self.ring[slot][2] < 0.0 {
                continue;
            }
            let birth = self.ring[slot][2] as u32;
            self.streams[slot].lanim += 0.2
                + stream_hash01(self.seed, birth, self.ticks as u32, STREAM_RATE_SALT) * 0.3;
        }
        if self.alive {
            let kind = self.kind;

            self.angle += spiral_angle_inc(self.angle, kind);
            let (x, y) = if matches!(kind, SpiralKind::Idpd | SpiralKind::Venuz) {
                (GUI_W / 2.0, GUI_H / 2.0)
            } else {
                orbit(self.angle)
            };
            if kind == SpiralKind::Venuz {
                self.push_star(x, y);
            } else {
                let mut rot = (self.angle + 45.0).to_radians();
                if kind == SpiralKind::Idpd && (self.ticks as i64 % 11) <= 1 {
                    // GML only swaps sprite_index to sprSpiralIDPD2 here;
                    // the angle is untouched. Variant rides in the sign bit.
                    rot = -rot.abs();
                }
                // Ring slot must match shader lookup `(birth-1) % N`:
                // birth == ticks here, so slot is (ticks-1) % N.
                let slot = (self.ticks as usize - 1) % MAX_WISPS;
                self.ring[slot] = [x, y, self.ticks, rot];
                self.head = (slot + 1) % MAX_WISPS;
                // GML `Spiral/Create_0`: `lanim = -random(300)`,
                // `langle = random_angle` (deterministic stream).
                let birth = self.ticks as u32;
                self.streams[slot] = WispStream {
                    lanim: -stream_hash01(self.seed, birth, 0, STREAM_HEAD_SALT) * 300.0,
                    langle: stream_hash01(self.seed, birth, 0, STREAM_ANGLE_SALT)
                        * std::f32::consts::TAU,
                };

                let proto = kind == SpiralKind::Proto;
                if rand::random::<f32>() * 16.0 < 1.0
                    && (proto || rand::random::<f32>() * 3.0 < 1.0)
                {
                    if rand::random::<f32>() * 50.0 < 1.0
                        && let Some((path, frame)) = variant_debris_for_gml_area(self.gml_area)
                    {
                        self.push_vard(x, y, path, frame);
                    } else {
                        let d = &mut self.debris[self.dhead];
                        *d = Debris {
                            alive: true,
                            xstart: x,
                            ystart: y,
                            dist: rand::random::<f32>() * 135.0 + 10.0,
                            angle: rand::random::<f32>() * 360.0,
                            turnspeed: rand::random::<f32>() * 8.0 - 4.0,
                            rotspeed: rand::random::<f32>() * 16.0 - 8.0,
                            xscale: 0.0,
                            grow: 0.0,
                            image_angle: 0.0,
                            frame: (rand::random::<f32>() * 4.0).floor().min(3.0),
                        };
                        self.dhead = (self.dhead + 1) % MAX_DEBRIS;
                    }
                }
            }
        } else {
            // Drain (SpiralCont dead): GML speeds growth (grow*=1.5,
            // destroy at 3.0 not 2.5) but `lanim` keeps realtime cadence.
            // Do NOT rewind births here — the shader indexes slots by
            // `(birth-1) % N`, so rewinding would orphan live wisps.
            // Scale fast-forward is carried by drain_bias instead.
            self.drain_bias += 5.5;
        }

        let drain = !self.alive;
        for (i, d) in self.debris.iter_mut().enumerate() {
            if !d.alive {
                continue;
            }
            let (rad, dir) = (d.dist * d.xscale, d.angle.to_radians());
            let dx = rad * dir.cos();
            let dy = -rad * dir.sin();
            d.angle += d.turnspeed;
            d.dist += d.grow;
            d.grow += 0.0005;
            d.xscale += d.grow / 1.5;
            d.grow = (d.grow + 1.0) * (1.0 + 0.001 * d.xscale) - 1.0;
            if drain {
                d.grow *= 1.5;
            }
            d.grow *= d.xscale * 0.05 + 1.0;
            d.image_angle += d.rotspeed;
            if dx + d.xstart < -16.0
                || dx + d.xstart > GUI_W + 16.0
                || dy + d.ystart < -16.0
                || dy + d.ystart > GUI_H + 16.0
            {
                d.alive = false;
                self.debris_ring[i] = [-1000.0; 4];
                continue;
            }

            self.debris_ring[i] = [
                d.xstart + dx,
                d.ystart + dy,
                d.image_angle.to_radians(),
                d.frame + d.xscale / 32.0,
            ];
        }

        for s in self.stars.iter_mut() {
            if !s.alive {
                continue;
            }
            s.dist += s.grow;
            s.grow += 0.0005;
            s.xscale += s.grow / 1.5;
            s.grow = (s.grow + 1.0) * (1.0 + 0.001 * s.xscale) - 1.0;
            if drain {
                s.grow *= 1.5;
            }
            s.grow *= s.xscale / 20.0 + 1.0;
            if s.xscale > 30.0 {
                s.alive = false;
            }
        }

        for v in self.vards.iter_mut() {
            if !v.alive {
                continue;
            }
            let (rad, dir) = (v.dist * v.xscale, v.angle.to_radians());
            let dx = rad * dir.cos();
            let dy = -rad * dir.sin();
            v.angle += v.turnspeed;
            v.dist += v.grow;
            v.grow += 0.0005;
            v.xscale += v.grow / 1.5;
            v.grow = (v.grow + 1.0) * (1.0 + 0.001 * v.xscale) - 1.0;
            if drain {
                v.grow *= 1.5;
            }
            v.grow *= v.xscale * 0.05 + 1.0;
            v.image_angle += v.rotspeed;
            if dx + v.xstart < -16.0
                || dx + v.xstart > GUI_W + 16.0
                || dy + v.ystart < -16.0
                || dy + v.ystart > GUI_H + 16.0
            {
                v.alive = false;
            }
        }
    }

    fn push_star(&mut self, x: f32, y: f32) {
        let star = Star {
            alive: true,
            xstart: x,
            ystart: y,
            dist: rand::random::<f32>() * 135.0 + 10.0,
            angle: rand::random::<f32>() * 360.0,
            grow: 0.0,
            xscale: 0.0,
            frame: if rand::random::<f32>() * 4.0 < 1.0 {
                1.0
            } else {
                0.0
            },
        };
        if let Some(slot) = self.stars.iter_mut().find(|s| !s.alive) {
            *slot = star;
        } else {
            self.stars.push(star);
        }
    }

    fn push_vard(&mut self, x: f32, y: f32, path: &'static str, frame: usize) {
        let vard = Vard {
            alive: true,
            xstart: x,
            ystart: y,
            dist: rand::random::<f32>() * 135.0 + 10.0,
            angle: rand::random::<f32>() * 360.0,
            turnspeed: rand::random::<f32>() * 8.0 - 4.0,
            rotspeed: rand::random_range(20.0..30.0)
                * if rand::random_bool(0.5) { 1.0 } else { -1.0 },
            grow: 0.0,
            xscale: 0.0,
            image_angle: 0.0,
            path,
            frame,
        };
        if let Some(slot) = self.vards.iter_mut().find(|v| !v.alive) {
            *slot = vard;
        } else {
            self.vards.push(vard);
        }
    }

    /// Advance the accumulator by `dt_ticks` 30 Hz ticks (bevy `step`,
    /// verbatim).
    pub fn step(&mut self, dt_ticks: f32) {
        self.acc += dt_ticks;
        while self.acc >= 1.0 {
            self.acc -= 1.0;
            self.tick_once();
        }
    }

    /// Kill-plane threshold: 2.5 alive, 3.0 while draining (bevy
    /// `vortex_thresh`, verbatim).
    pub fn thresh(&self) -> f32 {
        vortex_thresh(self.alive)
    }

    /// Shader variant pack: kind discriminant plus the Jungle-105 crystal
    /// flag (bevy `vortex_tick`/`ensure_vortex_quad`, verbatim).
    pub fn kindpacked(&self) -> f32 {
        self.kind as u8 as f32 + if self.gml_area == 105 { 4.0 } else { 0.0 }
    }

    /// Snapshot for the background pass. Produces exactly the type
    /// [`VortexPass`](crate::vortex_pass::VortexPass) consumes (converter, not
    /// an engine change): `glob_a = (ticks, drain_bias, bg_r, bg_g)`,
    /// `glob_b = (bg_b, bg_alpha, thresh, kindpacked)` with the bevy
    /// paddings (`-1` wisps, `-1000` debris), plus each wisp's own
    /// `lanim`/`langle` stream (GML `Spiral` bolt clock) indexed like
    /// the ring. Background is always black
    /// (bevy `background_color`); `bg_alpha` is shell-selected (bevy
    /// `vortex_needs_black`: 1 while a floor transition/level-up cover
    /// runs, 0 over menus).
    pub fn snapshot(&self, bg_alpha: f32) -> VortexSnapshot {
        let mut wisps = [[-1.0; 4]; VORTEX_WISPS];
        for (dst, src) in wisps.iter_mut().zip(self.ring.iter()) {
            *dst = *src;
        }
        let mut debris = [[-1000.0, 0.0, 0.0, 0.0]; VORTEX_DEBRIS];
        for (dst, src) in debris.iter_mut().zip(self.debris_ring.iter()) {
            *dst = *src;
        }
        let mut streams = [[-1.0, 0.0]; VORTEX_WISPS];
        for (dst, src) in streams.iter_mut().zip(self.streams.iter()) {
            *dst = [src.lanim, src.langle];
        }
        VortexSnapshot {
            wisps,
            debris,
            streams,
            ticks: self.ticks,
            drain_bias: self.drain_bias,
            bg_rgb: [0.0, 0.0, 0.0],
            bg_alpha,
            thresh: self.thresh(),
            kindpacked: self.kindpacked(),
            view: VORTEX_VIEW,
        }
    }
}

/// Per-tick angle advance (bevy `spiral_angle_inc`, verbatim): Proto runs
/// ~10 deg/tick with a slow wobble plus uniform ±1 jitter, everything
/// else runs a deterministic ~8 deg/tick wobble.
pub fn spiral_angle_inc(angle: f32, kind: SpiralKind) -> f32 {
    if kind == SpiralKind::Proto {
        10.0 + deg_sin(angle / 300.0) * 2.0 + (rand::random::<f32>() * 2.0 - 1.0)
    } else {
        8.0 + deg_sin(angle / 300.0)
    }
}

/// Wisp emitter position for the angle (bevy `orbit`, verbatim).
pub fn orbit(angle: f32) -> (f32, f32) {
    (
        GUI_W / 2.0 + deg_sin(angle / 921.0) * deg_sin(angle / 500.0) * 80.0,
        GUI_H / 2.0 + deg_cos(angle / 583.0) * deg_sin(angle / 500.0) * 50.0,
    )
}

// GML uses degrees, the sim uses radians.
fn deg_sin(deg: f32) -> f32 {
    deg.to_radians().sin()
}

fn deg_cos(deg: f32) -> f32 {
    deg.to_radians().cos()
}

/// GML Spiral/Step_0 destroy plane: 2.5 alive, 3.0 while draining.
pub fn vortex_thresh(alive: bool) -> f32 {
    if alive { 2.5 } else { 3.0 }
}

/// Fixed-step driver at the GML 30 Hz cadence (bevy `vortex_tick` rate
/// half: `step(dt * 30)`; uniform upload stays renderer-side).
pub fn tick_spiral(time: Res<SimTime>, ctl: Option<ResMut<SpiralCtl>>) {
    let Some(mut ctl) = ctl else {
        return;
    };
    ctl.step(time.delta_secs * 30.0);
}

