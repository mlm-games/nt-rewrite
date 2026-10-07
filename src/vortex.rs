//! Vortex spiral sim state: the headless half of the `SpiralCont` / `Spiral` /
//! `SpiralDebris` / `SpiralStar` object set - the [`SpiralKind`] enum, the
//! [`SpiralCtl`] angle-advance law, the deterministic per-seed random stream,
//! and the snapshot the background pass consumes.
//!
//! Port-only: no `Handle<Image>`, no materials, no GPU writes, no plugin.
//! Star/vard dots are data-only - positions still integrate, keeping the
//! shared RNG stream and debris spawn cadence identical - while the draw call
//! lives in the background pass.
//!
//! Timing is fixed-step driven: the shell calls [`SpiralCtl::step`] once per
//! GML 30 Hz tick (`options/main/options_main.yy:18`,
//! `"option_game_speed":30`). Area selection goes through
//! [`gml_area_for_area`] into [`SpiralKind::for_gml_area`]; `AreaId::Loop`
//! maps to GML area 1, i.e. `Normal` - GML has no loop-count branch.

use crate::vortex_pass::{
    VARD_CELL_SIZES, VARD_FRAME_COUNTS, VORTEX_DEBRIS, VORTEX_VARDS, VORTEX_WISPS, VortexSnapshot,
    vard_slot,
};
use bevy_ecs::prelude::*;

use crate::data::AreaId;

/// Wisp ring slots (matches `VORTEX_WISPS`; the shader indexes slots by
/// `(birth-1) % N`).
pub const MAX_WISPS: usize = VORTEX_WISPS;
/// Deterministic-stream salts (head-start, angle, per-tick rate).
const STREAM_HEAD_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
const STREAM_ANGLE_SALT: u64 = 0xBF58_476D_1CE4_E5B9;
const STREAM_RATE_SALT: u64 = 0x94D0_49BB_1331_11EB;
const STREAM_PROTO_SALT: u64 = 0xD6E8_FEB8_6659_FD93;
const STREAM_RANDOM_SALT: u64 = 0xA076_1D64_78BD_642F;
/// Debris ring slots (matches `VORTEX_DEBRIS`).
pub const MAX_DEBRIS: usize = VORTEX_DEBRIS;
/// Warmup ticks so a freshly spawned spiral is already full.
const WARMUP_TICKS: u32 = 150;

/// GUI-space size the spiral laws are written in (GML
/// `game_screen_width/height` base: HEIGHT is always 240, WIDTH is the live
/// view width - `view_width = 240 * aspect` with `opt_resolution` on
/// (default), 320 portrait-floored. GML `SpiralCont/Step_0` centers on
/// `view_width div 2`, so the vortex tracks wide windows; the sim carries the
/// live width in [`SpiralCtl::view_w`] (refreshed per frame by the shell;
/// warmups default to the 320 base).
pub const GUI_W: f32 = 320.0;
pub const GUI_H: f32 = 240.0;

/// Fallback look center + visible extent in wisp coord space: the 320x240 base
/// view 1:1. The live snapshot overrides with the live GUI view
/// (`view_w/2, 120, view_w, 240`) so a fullscreen quad maps screen px to GUI px
/// like GML (`display_set_gui_size` = view size). The extent must stay 1:1:
/// the shader reads it as the wisp coord -> GUI px mapping, so any scale here
/// shrinks every wisp toward the center and leaves the corners bare.
pub const VORTEX_VIEW: [f32; 4] = [160.0, 120.0, 320.0, 240.0];

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
    /// GML area -> variant table (GML `SpiralCont/Create_0.gml:24-32`
    /// verbatim, including Jungle 105 falling through to `Normal`; the
    /// crystal flag rides separately in `kindpacked`).
    pub fn for_gml_area(area: u8) -> Self {
        match area {
            100 => Self::Proto,
            106 => Self::Idpd,
            103 | 107 => Self::Venuz,
            _ => Self::Normal,
        }
    }
}

/// Target `AreaId` -> GML area int (GML `macros_general.gml:538-553`,
/// matched by variant name; target discriminants differ so this is NOT
/// `as u8`).
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
        // GML `area_crib` (macros_general.gml:553).
        AreaId::Crib => 107,
        AreaId::CrownVault => 100,
        AreaId::Loop => 1,
    }
}

/// Re-warm the view spiral for a fresh campfire-logo room (GML
/// `Vlambeer/Create_0` quit branch / `BackButton/Other_10` Menu branch:
/// `instance_create(0, 0, SpiralCont)` builds a LIVE cont with the
/// `repeat 150` warmup, never the previous run's leftover drain). Called from
/// the quit-to-menu action arms - the state-entry lifecycle in `lib.rs` only
/// fires on `AppState` edges, which the actions already consumed. Prefers the
/// `Run`'s seed so the menu vortex stays in the run's deterministic stream;
/// seed 0 when no run exists.
pub fn rewarm_view_spiral(world: &mut World, view_w: f32) {
    let seed = world
        .get_resource::<crate::comps_a::Run>()
        .map(|r| r.gen_seed)
        .unwrap_or(0);
    // The logo room is campfire (`setup_logo_room` resets the run there);
    // GML reads `GameCont.area` at cont creation, which is campfire here.
    let ctl = SpiralCtl::warmed_up_for_gml_area_seeded_in_view(0, seed, view_w);
    world.insert_resource(ctl);
}

/// Area-flavoured debris sprite pick (GML `SpiralDebris/Create_0.gml:14-25`
/// verbatim; the strip upload stays renderer-side).
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

/// Venuz starfield mote (data-only; GML `SpiralStar` draws itself, here the
/// background pass draws it from this state).
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
    pub draw_x: f32,
    pub draw_y: f32,
}

/// Area-flavoured debris mote (data-only; GML `SpiralDebris` draws itself,
/// here the background pass draws it from this state).
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
    pub draw_x: f32,
    pub draw_y: f32,
    pub draw_angle: f32,
    pub sound_played: bool,
    pub path: &'static str,
    pub frame: usize,
}

impl Vard {
    pub fn flyby_due(&mut self) -> bool {
        if !self.sound_played && self.xscale > 1.3 {
            self.sound_played = true;
            return true;
        }
        false
    }
}

/// Per-wisp lightning-stream state (GML `Spiral/Create_0` + `Step_0` +
/// `scrDrawSpiral`): `lanim` starts at `-random(300)`, grows
/// `0.2 + random(0.3)` per tick; `langle` is the `random_angle` bolt offset,
/// in radians. `lanim in (0, 6)` shows the `sprPortalLightning` stream at
/// rotation `image_angle + langle`. The GML dice are a deterministic splitmix
/// stream over `(seed, birth, tick)`, so snapshots stay contract-stable across
/// runs with the same seed. Dead slots hold `lanim = -1`. `sound_played` is GML
/// `Spiral.lsound`: the `sndPortalLightning{1..8}` one-shot fires once per
/// wisp, the first tick its bolt becomes visible outside menus (drained by the
/// shell audio layer; see `WispStream::bolt_sound_due`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WispStream {
    pub lanim: f32,
    pub langle: f32,
    pub sound_played: bool,
}

impl WispStream {
    pub fn dead() -> Self {
        Self {
            lanim: -1.0,
            langle: 0.0,
            sound_played: false,
        }
    }

    /// True while the bolt shows (GML `lanim > 0 && lanim < 6`).
    pub fn bolt_visible(&self) -> bool {
        self.lanim > 0.0 && self.lanim < 6.0
    }

    /// GML `scrDrawSpiral` bolt-sound gate: fires once per wisp the first tick
    /// the bolt is visible (`lanim in (0, 6)`); the caller gates further on
    /// non-menu (GML plays only when `!_is_menu`), emits `sndPortalLightning{1..8}`
    /// at pitch `0.9 + rand * 0.2`.
    pub fn bolt_sound_due(&mut self) -> bool {
        if !self.sound_played && self.bolt_visible() {
            self.sound_played = true;
            return true;
        }
        false
    }
}

/// Deterministic [0, 1) draw (splitmix64 over seed/birth/tick/salt:
/// GML `random()` replacement - stable across platforms and `rand`
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

/// Deterministic small-int pick in `0..n` off the same splitmix stream (GML
/// `irandom(n - 1)` replacement for the draw-script one-shots: bolt
/// `sndPortalLightning{1..8}` variant, flyby `sndPortalFlyby{1..4}` variant).
/// `slot` decorrelates wisps/motes; `tag` decorrelates the two rolls from each
/// other and from the `stream_hash01` lanes.
pub fn stream_pick(seed: u64, slot: u32, tag: u64) -> u32 {
    let mut z = seed
        .wrapping_add((slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(tag.wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as u32
}

/// Ribbon debris mote (feeds `debris_ring`, hence the snapshot).
/// `sound_played` is GML `SpiralDebris.sound` verbatim: the `sndPortalFlyby`
/// one-shot fires once when `image_xscale > 1.3` (drained by the shell
/// audio layer; see `Debris::flyby_due`).
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
    /// GML strip frame (`random(image_number)` float; the shader floors).
    pub frame: f32,
    pub sound_played: bool,
}

impl Debris {
    /// GML `SpiralDebris/Step_0` sound gate verbatim: fires once when
    /// the mote grows past 1.3 (marks played; the caller emits the
    /// `sndPortalFlyby{1..4}` cue with pitch 0.1 + ambience volume).
    pub fn flyby_due(&mut self) -> bool {
        if !self.sound_played && self.xscale > 1.3 {
            self.sound_played = true;
            return true;
        }
        false
    }
}

/// Headless spiral control (the `SpiralCont` owner state, minus the
/// renderer-only retired-entity list; `head`/`dhead` are `pub` so the
/// write-only ring cursors never trip dead-code lints outside test builds).
///
/// GML notes (`objects/SpiralCont/Step_0.gml`, `objects/Spiral/Step_0.gml`,
/// `objects/SpiralDebris/Step_0.gml`, `objects/SpiralStar/Step_0.gml`):
/// - Wisp births inherit the emitter pos (`instance_create(x, y, Spiral)` at
///   the cont's just-stepped `x/y`), so each `xstart/ystart` is that birth pos,
///   NOT the cont's live pos; the ring stores the birth pos per slot.
/// - `image_angle` is stored in RADIANS on each `Spiral` (`other.image_angle`
///   is degrees and GML trig takes degrees, but the shader's `cos/sin(rot)`
///   needs radians).
/// - `SpiralStar` has NO alpha gate (spirals never despawn either; only debris
///   culls offscreen). The star pass is additive white with `1 - xscale` black;
///   the wisp pass uses real art, never a white quad.
#[derive(Resource, Debug)]
pub struct SpiralCtl {
    pub angle: f32,
    pub ticks: f32,
    acc: f32,
    pub ring: Vec<[f32; 4]>,
    pub head: usize,
    /// GML `SpiralDebris` motes (`sound_played` drained per step for
    /// the `sndPortalFlyby` one-shot; see `drain_spiral_sounds`).
    pub debris: Vec<Debris>,
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
    rng_counter: u32,
    /// Per-ring-slot lightning streams, indexed exactly like `ring`.
    pub streams: Vec<WispStream>,
    wisp_initial: Vec<f32>,
    wisp_xscale: Vec<f32>,
    wisp_grow: Vec<f32>,
    /// GML `SpiralCont.bossfight` (`instance_exists(Nothing2) ||
    /// instance_exists(Nothing2Appear) || instance_exists(NothingSpiral)`): no
    /// `SpiralDebris` births while set. GML `Nothing2/Create_0` spawns
    /// `NothingSpiral` + a fresh `SpiralCont` on throne-II rise, so the live
    /// port condition is a live Throne-II enemy in any phase (`SpawnThroneII`
    /// pending, `ThroneII` fighting, `Nothing2Death` pageant), refreshed per
    /// tick by the shell from the live `Enemy` query; sim-only warmups default
    /// it off.
    pub bossfight_suppressed: bool,
    /// Live GUI view width in px (GML `view_width`: 240 * aspect with
    /// `opt_resolution` on, 320 portrait-floored). The emitter orbit centers on
    /// `view_w/2` (`SpiralCont/Step_0`: `view_width div 2`); the snapshot maps
    /// `view = (view_w/2, 120, view_w, 240)`. The shell refreshes this per
    /// frame; warmups default to 320.
    pub view_w: f32,
}

impl SpiralCtl {
    /// Seeded warmup: the spiral and debris streams roll from `seed` (run
    /// seed), so equal seeds snapshot identically.
    pub fn warmed_up_for_area_seeded(area: AreaId, seed: u64) -> Self {
        Self::warmed_up_for_gml_area_seeded(gml_area_for_area(area), seed)
    }

    pub fn warmed_up_for_gml_area(gml_area: u8) -> Self {
        Self::warmed_up_for_gml_area_seeded(gml_area, 0)
    }

    pub fn warmed_up_for_gml_area_seeded(gml_area: u8, seed: u64) -> Self {
        Self::warmed_up_for_gml_area_seeded_in_view(gml_area, seed, GUI_W)
    }

    pub fn warmed_up_for_gml_area_seeded_in_view(gml_area: u8, seed: u64, view_w: f32) -> Self {
        let mut ctl = Self {
            angle: 0.0,
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
                    sound_played: false,
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
            rng_counter: 0,
            bossfight_suppressed: false,
            view_w,
            streams: vec![WispStream::dead(); MAX_WISPS],
            wisp_initial: vec![0.0; MAX_WISPS],
            wisp_xscale: vec![0.0; MAX_WISPS],
            wisp_grow: vec![0.0; MAX_WISPS],
        };
        ctl.angle = ctl.random01() * 360.0;
        for _ in 0..WARMUP_TICKS {
            ctl.tick_once();
        }
        ctl
    }

    /// Mark the spiral dead: births freeze and the per-wisp death time drives
    /// the drain law. GML drops the `SpiralCont` instance outright
    /// (`GenCont/Destroy_0.gml:187`), which the remaining motes detect as
    /// `!instance_exists(SpiralCont)` (`Spiral/Step_0.gml:16-22`).
    pub fn kill(&mut self) {
        if self.alive {
            self.alive = false;
            self.death_tick = Some(self.ticks);
        }
    }

    /// Drain finished: every wisp past the kill plane AND every debris mote
    /// culled AND every star/vard dead. GML has no timer here - `SpiralCont`
    /// destroys itself only via the Step_0 gate, and `Menu/Draw_0` keeps calling
    /// `scrDrawSpiral` (drawing the leftover motes) as long as the campfire room
    /// lives. A tick-count gate unmounted the layer after ~0.9 s with motes
    /// still swirling (the "no vortex on the title screen" bug); `kill()` only
    /// freezes births, so the layer must stay mounted until the sky is empty.
    pub fn is_done(&self) -> bool {
        if self.alive {
            return false;
        }
        let wisps_live = self.ring.iter().enumerate().any(|(slot, w)| {
            if w[2] < 0.0 {
                return false;
            }
            if self.ticks - w[2] < 0.0 {
                return true;
            }
            self.wisp_xscale[slot] <= self.thresh()
        });
        if wisps_live {
            return false;
        }
        if self.debris.iter().any(|d| d.alive) {
            return false;
        }
        if self.stars.iter().any(|s| s.alive) {
            return false;
        }
        if self.vards.iter().any(|v| v.alive) {
            return false;
        }
        true
    }

    /// Reference recurrence used by the headless tests. Runtime keeps the
    /// equivalent state incrementally in `wisp_xscale` and `wisp_grow`.
    #[cfg(test)]
    fn wisp_scale_at(&self, birth: f32) -> f32 {
        if birth < 0.0 {
            return 0.0;
        }
        let age = self.ticks - birth;
        if age <= 0.0 {
            return 0.0;
        }
        let proto = self.kind == SpiralKind::Proto;
        let live_ticks = match self.death_tick {
            Some(d) => (d - birth).clamp(0.0, age),
            None => age,
        };
        let mut grow = 0.0f32;
        let slot = (birth.floor() as usize).wrapping_sub(1) % MAX_WISPS;
        let mut xs = self.wisp_initial[slot];
        let mut t = 0.0f32;
        while t < age {
            let step = (age - t).min(1.0);
            grow += (0.0002 + if proto { 0.0003 } else { 0.0 }) * step;
            xs += grow * step;
            grow = (grow + 1.0) * (1.0 + 0.0005 * xs) - 1.0;
            if t >= live_ticks {
                grow *= 1.5f32.powf(step);
            }
            t += step;
        }
        xs
    }

    /// GML `Spiral/Step_0` growth law: `grow += 0.0002` (+0.0003 on the
    /// proto strip),
    /// `xscale += grow`, `grow = (grow+1)*(1+0.0005*xscale)-1`, drain
    /// `grow *= 1.5`. Headless mirror of one wisp tick for tests.
    pub fn step_wisp_grow(grow: f32, xscale: f32, proto: bool, drain: bool) -> (f32, f32) {
        let mut grow = grow + 0.0002 + if proto { 0.0003 } else { 0.0 };
        let xscale = xscale + grow;
        grow = (grow + 1.0) * (1.0 + 0.0005 * xscale) - 1.0;
        if drain {
            grow *= 1.5;
        }
        (grow, xscale)
    }

    fn random01(&mut self) -> f32 {
        self.rng_counter = self.rng_counter.wrapping_add(1);
        stream_hash01(self.seed, self.rng_counter, 0, STREAM_RANDOM_SALT)
    }

    fn random_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.random01()
    }

    fn random_int(&mut self, min: i32, max: i32) -> f32 {
        min as f32 + (self.random01() * (max - min + 1) as f32).floor()
    }

    fn tick_once(&mut self) {
        self.ticks += 1.0;
        let drain = !self.alive;
        // GML `Spiral/Step_0` bolt clock first: live wisps advance
        // before this tick's birth runs, so a newborn's `Create_0`
        // head-start takes its first `Step_0` increment next tick.
        for slot in 0..MAX_WISPS {
            if self.ring[slot][2] < 0.0 {
                continue;
            }
            let (grow, xscale) = Self::step_wisp_grow(
                self.wisp_grow[slot],
                self.wisp_xscale[slot],
                self.kind == SpiralKind::Proto,
                drain,
            );
            self.wisp_grow[slot] = grow;
            self.wisp_xscale[slot] = xscale;
            let birth = self.ring[slot][2] as u32;
            self.streams[slot].lanim +=
                0.2 + stream_hash01(self.seed, birth, self.ticks as u32, STREAM_RATE_SALT) * 0.3;
        }
        if self.alive {
            let kind = self.kind;

            let angle_inc = spiral_angle_inc(self.angle, kind);
            let jitter = if kind == SpiralKind::Proto {
                self.random_int(-1, 1)
            } else {
                0.0
            };
            self.angle += angle_inc + jitter;
            // GML `SpiralCont/Step_0` verbatim: IDPD/Venuz lock to the
            // VIEW center (`view_width div 2`, `view_height div 2`);
            // Normal/Proto drift around it on the sine orbit (also
            // view-centered - `x = _cx + ...`, never camera-centered).
            let (x, y) = if matches!(kind, SpiralKind::Idpd | SpiralKind::Venuz) {
                (self.view_w / 2.0, GUI_H / 2.0)
            } else {
                orbit(self.angle, self.view_w)
            };
            if kind == SpiralKind::Venuz {
                self.push_star(x, y);
            } else {
                // GML `SpiralCont/Step_0`: `image_angle = other.image_angle`
                // copies the cont angle in DEGREES, then `scrDrawSpiral` draws at
                // `image_angle + 45` with GML-degree trig. The shader samples with
                // radians trig, so the ring stores `(angle_deg + 45).to_radians()`
                // (rotation only; `+45` is wisp-art-only - the bolt pass strips it).
                let mut rot = (self.angle + 45.0).to_radians();
                if kind == SpiralKind::Idpd && (self.ticks as i64 % 11) <= 1 {
                    // GML only swaps sprite_index to sprSpiralIDPD2 here;
                    // the angle is untouched. Variant rides in the sign bit.
                    rot = -rot.abs();
                }
                // Ring slot must match shader lookup `(birth-1) % N`:
                // birth == ticks here, so slot is (ticks-1) % N.
                let slot = (self.ticks as usize - 1) % MAX_WISPS;
                // GML `instance_create(x, y, Spiral)`: the wisp's
                // `xstart/ystart` freeze at the emitter pos (the cont's
                // just-stepped `x/y`), NOT the cont's live pos.
                self.ring[slot] = [x, y, self.ticks, rot];
                self.head = (slot + 1) % MAX_WISPS;
                // GML `Spiral/Create_0`: `lanim = -random(300)`,
                // `langle = random_angle` (deterministic stream).
                let birth = self.ticks as u32;
                let proto = kind == SpiralKind::Proto;
                self.wisp_initial[slot] = if proto {
                    stream_hash01(self.seed, birth, 0, STREAM_PROTO_SALT) * 0.01
                } else {
                    0.0
                };
                self.wisp_xscale[slot] = self.wisp_initial[slot];
                self.wisp_grow[slot] = 0.0;
                self.streams[slot] = WispStream {
                    lanim: -stream_hash01(self.seed, birth, 0, STREAM_HEAD_SALT) * 300.0,
                    langle: stream_hash01(self.seed, birth, 0, STREAM_ANGLE_SALT)
                        * std::f32::consts::TAU,
                    sound_played: false,
                };

                // GML `!bossfight` gate verbatim: no debris births while
                // the Throne-II fight runs (`Nothing2`/`Nothing2Appear`/
                // `NothingSpiral` alive); the port reads it off
                // `bossfight_suppressed`.
                let debris_ok = !self.bossfight_suppressed
                    && self.random01() * 16.0 < 1.0
                    && (proto || self.random01() * 3.0 < 1.0);
                if debris_ok {
                    if self.random01() * 50.0 < 1.0
                        && let Some((path, frame)) = variant_debris_for_gml_area(self.gml_area)
                    {
                        self.push_vard(x, y, path, frame);
                    } else {
                        // GML `SpiralDebris/Create_0` runs at birth, then the SAME
                        // tick's `Step_0` integrates once before the first draw
                        // (`repeat 150` warmup steps self then motes, live ticks
                        // create-then-step in order). A mote stored at xscale 0 and
                        // drawn next frame would pop full-size through the
                        // `frame + xscale/32` packing (ring write below), so
                        // integrate once here: the birth frame already carries it.
                        let slot = self.dhead;
                        let frame = (self.random01() * 4.0).floor().min(3.0);
                        let image_angle = self.random_range(0.0, 360.0);
                        self.debris[slot] = Debris {
                            alive: true,
                            xstart: x,
                            ystart: y,
                            dist: self.random_int(10, 145),
                            angle: self.random_range(0.0, 360.0),
                            turnspeed: self.random_int(-4, 4),
                            rotspeed: self.random_int(-8, 8),
                            xscale: 0.0,
                            grow: 0.0,
                            image_angle,
                            // GML `sprDebrisN` default arm: `image_index =
                            // random(image_number)` is float, but the ring packs
                            // `frame + xscale/32` and the shader splits with
                            // `floor`/`fract` - so the frame MUST be integral (the
                            // port floors + clamps the roll to 0..3), else the frame
                            // fraction leaks into `fract` and newborns decode at
                            // xscale up to 32 (the "debris spawns massive" bug).
                            frame,
                            sound_played: false,
                        };
                        self.step_debris_slot(slot, false);
                        self.dhead = (slot + 1) % MAX_DEBRIS;
                    }
                }
            }
        } else {
            // Drain (SpiralCont dead): GML speeds growth (grow*=1.5,
            // destroy at 3.0 not 2.5) but `lanim` keeps realtime cadence.
            // Do NOT rewind births here - the shader indexes slots by
            // `(birth-1) % N`, so rewinding would orphan live wisps.
            // The CPU snapshot applies the same drain law per birth.
            self.drain_bias += 5.5;
        }

        for i in 0..MAX_DEBRIS {
            if !self.debris[i].alive {
                continue;
            }
            // Newborns already integrated at birth (see the birth site
            // above); skip the pre-growth mote so it advances once per
            // tick like GML, not twice.
            if self.debris[i].xscale == 0.0 && self.debris[i].grow == 0.0 {
                continue;
            }
            self.step_debris_slot(i, drain);
        }

        for s in self.stars.iter_mut() {
            if !s.alive {
                continue;
            }
            let (rad, dir) = (s.dist * s.xscale, s.angle.to_radians());
            s.draw_x = s.xstart + rad * dir.cos();
            s.draw_y = s.ystart - rad * dir.sin();
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

        for i in 0..self.vards.len() {
            if self.vards[i].alive {
                self.step_vard_slot(i, drain);
            }
        }
    }

    fn push_star(&mut self, x: f32, y: f32) {
        let star = Star {
            alive: true,
            xstart: x,
            ystart: y,
            dist: self.random_int(10, 145),
            angle: self.random_range(0.0, 360.0),
            grow: 0.0,
            xscale: 0.0,
            frame: if self.random01() * 4.0 < 1.0 {
                1.0
            } else {
                0.0
            },
            draw_x: x,
            draw_y: y,
        };
        if let Some(slot) = self.stars.iter_mut().find(|s| !s.alive) {
            *slot = star;
        } else {
            self.stars.push(star);
        }
    }

    fn push_vard(&mut self, x: f32, y: f32, path: &'static str, frame: usize) {
        let mut vard = Vard {
            alive: true,
            xstart: x,
            ystart: y,
            dist: self.random_int(10, 145),
            angle: self.random_range(0.0, 360.0),
            turnspeed: self.random_int(-4, 4),
            rotspeed: self.random_range(20.0, 30.0)
                * if self.random01() < 0.5 { 1.0 } else { -1.0 },
            grow: 0.0,
            xscale: 0.0,
            image_angle: self.random_range(0.0, 360.0),
            draw_x: x,
            draw_y: y,
            draw_angle: 0.0,
            sound_played: false,
            path,
            frame,
        };
        vard.draw_angle = vard.image_angle;
        let index = if let Some(index) = self.vards.iter().position(|v| !v.alive) {
            self.vards[index] = vard;
            index
        } else {
            self.vards.push(vard);
            self.vards.len() - 1
        };
        self.step_vard_slot(index, false);
    }

    /// One GML `SpiralDebris/Step_0` integration over mote `i` (order:
    /// pos from current angle/radius, then angle/dist/grow/xscale advance, cull
    /// at view ± 16, ring write with the `frame + xscale/32` pack). Shared by
    /// the birth site (newborns step once in their birth tick, like GML's
    /// Create-then-Step order) and the per-tick loop.
    fn step_debris_slot(&mut self, i: usize, drain: bool) {
        // Split borrow: the mote mutably, the ring slot mutably.
        let (dx, dy, rot_rad, packed, culled) = {
            let d = &mut self.debris[i];
            // GML `lengthdir_x(r, a) = r*cos(a)`, `lengthdir_y(r, a) =
            // -r*sin(a)` (y-down, 90 = south).
            let (rad, dir) = (d.dist * d.xscale, d.angle.to_radians());
            let dx = rad * dir.cos();
            let dy = -rad * dir.sin();
            d.angle += d.turnspeed;
            d.dist += d.grow;
            d.grow += 0.0005;
            d.xscale += d.grow / 1.5;
            d.grow = (d.grow + 1.0) * (1.0 + 0.001 * d.xscale) - 1.0;
            if self.alive && self.kind == SpiralKind::Proto {
                d.grow *= d.xscale * 0.1 + 1.0;
            }
            if drain {
                d.grow *= 1.5;
            }
            d.grow *= d.xscale * 0.05 + 1.0;
            d.image_angle += d.rotspeed;
            // GML `Step_0` cull: view rect ± 16, but against the LIVE view width
            // (`view_width`, 426 at 16:9), not the 320 base. The old `GUI_W` bound
            // killed side-drifting motes up to 106px before they left the screen.
            let culled = dx + d.xstart < -16.0
                || dx + d.xstart > self.view_w + 16.0
                || dy + d.ystart < -16.0
                || dy + d.ystart > GUI_H + 16.0;
            if culled {
                d.alive = false;
            }
            (
                d.xstart + dx,
                d.ystart + dy,
                d.image_angle.to_radians(),
                d.frame + d.xscale / 32.0,
                culled,
            )
        };
        self.debris_ring[i] = if culled {
            // Parked sentinel: x < -100 is the ONLY component the shader tests.
            // Must be `[-1000, 0, 0, 0]` - `[-1000; 4]` smuggles `frame=0,
            // xscale=32` into slot 3, which the shader unpacks as a FULL-SIZE
            // rock (the "debris spawns massive" bug).
            [-1000.0, 0.0, 0.0, 0.0]
        } else {
            [dx, dy, rot_rad, packed]
        };
    }

    fn step_vard_slot(&mut self, i: usize, drain: bool) {
        let v = &mut self.vards[i];
        let (rad, dir) = (v.dist * v.xscale, v.angle.to_radians());
        let dx = rad * dir.cos();
        let dy = -rad * dir.sin();
        v.draw_x = v.xstart + dx;
        v.draw_y = v.ystart + dy;
        v.draw_angle = v.image_angle;
        v.angle += v.turnspeed;
        v.dist += v.grow;
        v.grow += 0.0005;
        v.xscale += v.grow / 1.5;
        v.grow = (v.grow + 1.0) * (1.0 + 0.001 * v.xscale) - 1.0;
        if self.alive && self.kind == SpiralKind::Proto {
            v.grow *= v.xscale * 0.1 + 1.0;
        }
        if drain {
            v.grow *= 1.5;
        }
        v.grow *= v.xscale * 0.05 + 1.0;
        v.image_angle += v.rotspeed;
        if dx + v.xstart < -16.0
            || dx + v.xstart > self.view_w + 16.0
            || dy + v.ystart < -16.0
            || dy + v.ystart > GUI_H + 16.0
        {
            v.alive = false;
        }
    }

    /// Advance the accumulator by `dt_ticks` 30 Hz ticks (the headless stand-in
    /// for GML's per-step object events at `option_game_speed: 30`).
    pub fn step(&mut self, dt_ticks: f32) {
        self.acc += dt_ticks;
        while self.acc >= 1.0 {
            self.acc -= 1.0;
            self.tick_once();
        }
    }

    /// Kill-plane threshold: 2.5 alive, 3.0 while draining (GML
    /// `Spiral/Step_0.gml:16-22`: `_m = 2.5`, or `3` with no `SpiralCont`).
    pub fn thresh(&self) -> f32 {
        vortex_thresh(self.alive)
    }

    /// Shader variant pack: kind discriminant plus the Jungle-105 crystal
    /// flag (port-only packing; GML picks the variant with `sprite_index`
    /// in `SpiralCont/Step_0.gml:36-45`).
    pub fn kindpacked(&self) -> f32 {
        self.kind as u8 as f32 + if self.gml_area == 105 { 4.0 } else { 0.0 }
    }

    pub fn emitter_pos(&self) -> (f32, f32) {
        if matches!(self.kind, SpiralKind::Idpd | SpiralKind::Venuz) {
            (self.view_w / 2.0, GUI_H / 2.0)
        } else {
            orbit(self.angle, self.view_w)
        }
    }

    /// Snapshot for the background pass: exactly what
    /// [`VortexPass`](crate::vortex_pass::VortexPass) consumes (converter, not
    /// engine change). `glob_a = (ticks, drain_bias, bg_r, bg_g)`,
    /// `glob_b = (bg_b, bg_alpha, thresh, kindpacked)` with the port's dead-slot
    /// paddings (`-1` wisps, `-1000` debris), plus each wisp's `lanim`/`langle`
    /// stream (GML `Spiral` bolt clock) indexed like the ring. Background is
    /// always black (GML `scrDrawSpiral.gml:4,8-11` `draw_clear(c_black)` off
    /// `Menu`); `bg_alpha` follows the same script (opaque everywhere except the
    /// campfire title, whose `Menu/Draw_0.gml:4` call takes the `_is_menu` arm
    /// that skips the clear; see the `bg_alpha` match at the `VortexPass` mount
    /// in `lib.rs`). Sound flags (`WispStream::sound_played`,
    /// `Debris::sound_played`) stay sim-side - the snapshot carries no audio, the
    /// shell drains them directly (GML plays them inline in the draw script).
    pub fn snapshot(&self, bg_alpha: f32) -> VortexSnapshot {
        self.snapshot_with_lightning(bg_alpha, true)
    }

    pub fn snapshot_with_lightning(&self, bg_alpha: f32, draw_bolts: bool) -> VortexSnapshot {
        self.snapshot_with_render_mode(bg_alpha, draw_bolts, true)
    }

    pub fn snapshot_with_render_mode(
        &self,
        bg_alpha: f32,
        draw_bolts: bool,
        draw_details: bool,
    ) -> VortexSnapshot {
        let mut wisps = [[-1.0; 4]; VORTEX_WISPS];
        for (dst, src) in wisps.iter_mut().zip(self.ring.iter()) {
            *dst = *src;
        }
        let mut debris = [[-1000.0, 0.0, 0.0, 0.0]; VORTEX_DEBRIS];
        for (dst, src) in debris.iter_mut().zip(self.debris_ring.iter()) {
            *dst = *src;
        }
        let mut streams = [[-1.0, 0.0, 0.0]; VORTEX_WISPS];
        for (slot, (dst, src)) in streams.iter_mut().zip(self.streams.iter()).enumerate() {
            let scale = self.ring[slot][2]
                .ge(&0.0)
                .then_some(self.wisp_xscale[slot])
                .unwrap_or(0.0);
            *dst = [src.lanim, src.langle, scale];
        }
        let mut stars = [[-1000.0, 0.0, 0.0, 0.0]; VORTEX_WISPS];
        for (dst, src) in stars
            .iter_mut()
            .zip(self.stars.iter().filter(|src| src.alive))
        {
            *dst = [src.draw_x, src.draw_y, src.xscale, src.frame];
        }
        let mut vards = [[-1000.0, 0.0, 0.0, 0.0]; VORTEX_VARDS];
        let mut vard_meta = [[0.0; 4]; VORTEX_VARDS];
        let mut vard_index = 0usize;
        for src in self.vards.iter().filter(|v| v.alive) {
            let Some(slot) = vard_slot(src.path) else {
                continue;
            };
            if vard_index >= VORTEX_VARDS {
                break;
            }
            let (cell_w, cell_h) = VARD_CELL_SIZES[slot];
            vards[vard_index] = [
                src.draw_x,
                src.draw_y,
                src.draw_angle.to_radians(),
                src.xscale,
            ];
            let frame = src.frame.min(VARD_FRAME_COUNTS[slot].saturating_sub(1));
            vard_meta[vard_index] = [frame as f32, cell_w, cell_h, slot as f32];
            vard_index += 1;
        }
        VortexSnapshot {
            wisps,
            debris,
            streams,
            stars,
            vards,
            vard_meta,
            ticks: self.ticks,
            drain_bias: self.drain_bias,
            bg_rgb: [0.0, 0.0, 0.0],
            bg_alpha,
            thresh: self.thresh(),
            kindpacked: self.kindpacked(),
            draw_bolts: if draw_bolts { 1.0 } else { 0.0 },
            draw_details: if draw_details { 1.0 } else { 0.0 },
            view: [self.view_w / 2.0, GUI_H / 2.0, self.view_w, GUI_H],
        }
    }
}

/// Per-tick angle advance without the Proto RNG lane. The caller adds
/// the per-seed jitter used by Proto; all other kinds use the exact
/// deterministic wobble.
pub fn spiral_angle_inc(angle: f32, kind: SpiralKind) -> f32 {
    if kind == SpiralKind::Proto {
        10.0 + deg_sin(angle / 300.0) * 2.0
    } else {
        8.0 + deg_sin(angle / 300.0)
    }
}

/// Wisp emitter position for the angle (GML `SpiralCont/Step_0`
/// orbit verbatim): view-centered (`_cx = view_width div 2`) with the
/// ±80x/±50y sine drift. `view_w` is the live GUI view width.
pub fn orbit(angle: f32, view_w: f32) -> (f32, f32) {
    (
        view_w / 2.0 + deg_sin(angle / 921.0) * deg_sin(angle / 500.0) * 80.0,
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

#[cfg(test)]
mod vortex_ui_parity {
    use super::*;

    /// GML `Spiral/Step_0` growth recurrence verbatim: iterating
    /// `step_wisp_grow` 119 times from 0 reaches the reference tail
    /// scale (2.539587).
    #[test]
    fn wisp_growth_matches_reference_tail() {
        let (mut grow, mut xs) = (0.0f32, 0.0f32);
        for _ in 0..119 {
            (grow, xs) = SpiralCtl::step_wisp_grow(grow, xs, false, false);
        }
        assert!(
            (xs - 2.539_587).abs() < 1e-3,
            "wisp xscale at age 119 diverged from the reference tail: {xs}"
        );
    }

    /// The view-width cull law: motes past the LIVE width + 16 die, motes
    /// inside the wide margin (beyond the 320 base) survive. GML
    /// `view_width` (426 at 16:9), not the 320 base.
    #[test]
    fn debris_cull_uses_live_view_width() {
        let mut ctl = SpiralCtl::warmed_up_for_gml_area(1);
        ctl.view_w = 426.0;
        let idx = 0;
        // Mote drawn at x = 400 (inside 426+16, outside 320+16).
        ctl.debris[idx] = Debris {
            alive: true,
            xstart: 400.0,
            ystart: 120.0,
            dist: 0.0,
            angle: 0.0,
            turnspeed: 0.0,
            rotspeed: 0.0,
            xscale: 1.0,
            grow: 0.0,
            image_angle: 0.0,
            frame: 0.0,
            sound_played: true,
        };
        ctl.step_debris_slot(idx, false);
        assert!(
            ctl.debris[idx].alive,
            "mote at x=400 culled under a 426-wide view (320-base law)"
        );
    }

    /// Quit-to-menu builds a LIVE campfire cont (GML `Vlambeer/Create_0`
    /// quit branch / `BackButton` Menu branch): births resume, never a
    /// stale drain from the dead run.
    #[test]
    fn rewarm_view_spiral_is_live_campfire() {
        let mut world = World::new();
        world.insert_resource(crate::comps_a::Run::default());
        let mut dead = SpiralCtl::warmed_up_for_gml_area(1);
        dead.kill();
        // Step the drain until the sky is actually empty (GML has no
        // timer here - motes die on the kill plane / view cull).
        for _ in 0..400 {
            dead.step(1.0);
            if dead.is_done() {
                break;
            }
        }
        assert!(dead.is_done());
        world.insert_resource(dead);
        rewarm_view_spiral(&mut world, GUI_W);
        let ctl = world.resource::<SpiralCtl>();
        assert!(ctl.alive, "menu spiral must be live (fresh SpiralCont)");
        assert!(!ctl.is_done());
        assert_eq!(ctl.kind, SpiralKind::Normal);
    }

    /// A freshly killed full spiral is NOT done: the leftover motes are
    /// still swirling (GML `Menu/Draw_0` keeps drawing them). The old
    /// tick-count gate called this done after ~0.9 s - the "no vortex on
    /// the title screen" bug.
    #[test]
    fn killed_spiral_stays_mounted_while_motes_live() {
        let mut ctl = SpiralCtl::warmed_up_for_gml_area_seeded(0, 1234);
        ctl.view_w = 426.0;
        ctl.kill();
        assert!(
            !ctl.is_done(),
            "freshly killed spiral must stay mounted (motes still live)"
        );
        let snap = ctl.snapshot(0.0);
        let live_wisps = snap.wisps.iter().filter(|w| w[2] >= 0.0).count();
        assert!(live_wisps > 64, "warmup must leave a full ring behind");
    }

    /// GML `Vlambeer/Alarm_0` parity: the boot spiral is LIVE from construction
    /// (warmed cont, births on) so the logo screen can mount the vortex layer
    /// under the `Logo`; a dead-on-arrival spiral left the reel black (the "no
    /// vortex on the logo screen" bug).
    #[test]
    fn boot_spiral_is_live_for_logo_mount() {
        let ctl = SpiralCtl::warmed_up_for_gml_area_seeded(0, 1234);
        assert!(ctl.alive, "boot spiral must be live (fresh SpiralCont)");
        assert!(!ctl.is_done());
    }

    /// GML `Spiral/Step_0` two-phase drain verbatim: a wisp born 1 tick
    /// before the kill crosses the 3.0 kill plane after ~19 drain ticks
    /// (not ~110 - the old law applied the 1.5x drain factor from
    /// birth), and a mid-ring wisp stays visibly live mid-drain.
    #[test]
    fn drain_kill_plane_matches_gml_two_phase_law() {
        let mut ctl = SpiralCtl::warmed_up_for_gml_area_seeded(0, 1234);
        ctl.view_w = 426.0;
        ctl.kill();
        let death = ctl.ticks;
        // Youngest wisp: born 1 tick before the kill.
        let young = death - 1.0;
        // 10 drain ticks in: GML grows it to ~0.3, still swirling.
        ctl.ticks = death + 10.0;
        let s = ctl.wisp_scale_at(young);
        assert!(
            s < 3.0,
            "young wisp must still swirl 10 ticks after kill, got {s}"
        );
        // 25 drain ticks in: GML blows it past the plane (~42).
        ctl.ticks = death + 25.0;
        let s = ctl.wisp_scale_at(young);
        assert!(
            s > 3.0,
            "young wisp must be culled 25 ticks after kill, got {s}"
        );
    }
}
