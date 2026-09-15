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

use crate::vortex_pass::{VORTEX_DEBRIS, VORTEX_WISPS, VortexSnapshot};
use bevy_ecs::prelude::*;
use repame_sim::SimTime;

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

/// GUI-space size the spiral laws are written in (GML
/// `game_screen_width/height` base; the HEIGHT is always 240, the WIDTH
/// is the live view width — `view_width = 240 * aspect` with
/// `opt_resolution` on (default), 320 portrait-floored. GML
/// `SpiralCont/Step_0` centers on `view_width div 2`, so the vortex
/// tracks wide windows; the sim carries the live width in
/// [`SpiralCtl::view_w`] ( refreshed per frame by the shell; warmups
/// default to the 320 base).
pub const GUI_W: f32 = 320.0;
pub const GUI_H: f32 = 240.0;

/// Fallback look center + visible extent in wisp coord space: the
/// 320x240 base view 1:1. The live snapshot overrides this with the
/// live GUI view (`view_w/2, 120, view_w, 240`) so the fullscreen quad
/// maps screen px to GUI px exactly like GML (`display_set_gui_size`
/// = view size). The old 6x value (`[160, 120, 1920, 1440]`) was a
/// mistranslation of bevy's 6x WORLD-space mesh size: on a fullscreen
/// quad it shrank every wisp 6x toward the center, so the vortex
/// never filled the corners.
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

/// Re-warm the view spiral for a fresh campfire-logo room (GML
/// `Vlambeer/Create_0` quit branch / `BackButton/Other_10` Menu branch:
/// `instance_create(0, 0, SpiralCont)` builds a LIVE cont with the
/// `repeat 150` warmup, never the previous run's leftover drain).
/// Called from the quit-to-menu action arms (the state-entry lifecycle in
/// `lib.rs` only fires on `AppState` edges, which the actions already
/// consumed). Prefers the `Run`'s seed when present so the menu vortex
/// stays in the run's deterministic stream; seed 0 when no run exists.
pub fn rewarm_view_spiral(world: &mut World) {
    let seed = world
        .get_resource::<crate::comps_a::Run>()
        .map(|r| r.gen_seed)
        .unwrap_or(0);
    // The logo room is campfire (`setup_logo_room` resets the run there);
    // GML reads `GameCont.area` at cont creation, which is campfire here.
    let mut ctl = SpiralCtl::warmed_up_for_area_seeded(AreaId::Campfire, seed);
    ctl.view_w = GUI_W;
    world.insert_resource(ctl);
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
/// `sound_played` is GML `Spiral.lsound` verbatim: the
/// `sndPortalLightning{1..8}` one-shot fires once per wisp, the first
/// tick its bolt becomes visible outside menus (drained by the shell
/// audio layer; see `WispStream::bolt_sound_due`).
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

    /// GML `scrDrawSpiral` bolt-sound gate verbatim: fires once per
    /// wisp, the first tick the bolt is visible (`lanim in (0, 6)`;
    /// the caller additionally gates on non-menu, since GML only
    /// plays the sound when `!_is_menu`). Marks played; the caller
    /// emits `sndPortalLightning{1..8}` at pitch `0.9 + rand * 0.2`.
    pub fn bolt_sound_due(&mut self) -> bool {
        if !self.sound_played && self.bolt_visible() {
            self.sound_played = true;
            return true;
        }
        false
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

/// Deterministic small-int pick in `0..n` off the same splitmix stream
/// (GML `irandom(n - 1)` replacement for the draw-script one-shots:
/// bolt `sndPortalLightning{1..8}` variant, flyby `sndPortalFlyby{1..4}`
/// variant). `slot` decorrelates wisps/motes; `tag` decorrelates the
/// two rolls from each other and from the `stream_hash01` lanes.
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

/// Headless spiral control (bevy `SpiralCtl` minus the render-only
/// `retired` entity list; `head`/`dhead` are `pub` here so the write-only
/// ring cursors never trip dead-code lints outside test builds).
///
/// GML notes (`objects/SpiralCont/Step_0.gml`, `objects/Spiral/Step_0.gml`,
/// `objects/SpiralDebris/Step_0.gml`, `objects/SpiralStar/Step_0.gml`):
/// - Wisp births inherit the emitter pos (`instance_create(x, y, Spiral)`
///   at the cont's just-stepped `x/y`); each wisp's `xstart/ystart` is
///   that birth pos, NOT the cont's live pos. The ring therefore stores
///   the birth pos per slot.
/// - `image_angle` is stored in RADIANS on each `Spiral` (`other.
///   image_angle` is degrees; GML trig takes degrees so the drawn value
///   is deg-based, but the stored `image_angle` field itself is the
///   radian conversion — the shader's `cos/sin(rot)` needs radians).
/// - `SpiralStar` has NO alpha gate (spirals never despawn either;
///   only debris culls offscreen). The star shader pass is additive
///   white with `1 - xscale` black; the wisp pass uses real art, never
///   a white quad.
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
    /// Per-ring-slot lightning streams, indexed exactly like `ring`.
    pub streams: Vec<WispStream>,
    /// GML `SpiralCont.bossfight` verbatim (`instance_exists(Nothing2) ||
    /// instance_exists(Nothing2Appear) || instance_exists(NothingSpiral)`):
    /// while set, no `SpiralDebris` births. GML `Nothing2/Create_0`
    /// spawns `NothingSpiral` + a fresh `SpiralCont` on throne-II rise,
    /// so the live port condition is a live Throne-II enemy (any phase:
    /// `SpawnThroneII` pending, `ThroneII` fighting, `Nothing2Death`
    /// pageant) — refreshed per tick by the shell from the live `Enemy`
    /// query. Sim-only warmups default it off.
    pub bossfight_suppressed: bool,
    /// Live GUI view width in px (GML `view_width`: 240 * aspect with
    /// `opt_resolution` on, 320 portrait-floored). The emitter orbit
    /// centers on `view_w/2` (`SpiralCont/Step_0`: `view_width div 2`);
    /// the snapshot maps `view = (view_w/2, 120, view_w, 240)`. The
    /// shell refreshes this per frame; warmups default to 320.
    pub view_w: f32,
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
            bossfight_suppressed: false,
            view_w: GUI_W,
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

    /// GML `Spiral/Step_0` growth law verbatim (shared by the shader's
    /// `SCALE_TABLE`): `grow += 0.0002` (+0.0003 on the proto strip),
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
            self.streams[slot].lanim +=
                0.2 + stream_hash01(self.seed, birth, self.ticks as u32, STREAM_RATE_SALT) * 0.3;
        }
        if self.alive {
            let kind = self.kind;

            self.angle += spiral_angle_inc(self.angle, kind);
            // GML `SpiralCont/Step_0` verbatim: IDPD/Venuz lock to the
            // VIEW center (`view_width div 2`, `view_height div 2`);
            // Normal/Proto drift around it on the sine orbit (also
            // view-centered — `x = _cx + ...`, never camera-centered).
            let (x, y) = if matches!(kind, SpiralKind::Idpd | SpiralKind::Venuz) {
                (self.view_w / 2.0, GUI_H / 2.0)
            } else {
                orbit(self.angle, self.view_w)
            };
            if kind == SpiralKind::Venuz {
                self.push_star(x, y);
            } else {
                // GML `SpiralCont/Step_0`: `image_angle = other.image_angle`
                // copies the cont angle in DEGREES, then `scrDrawSpiral`
                // draws at `image_angle + 45` with GML-degree trig. The
                // shader samples with radians trig, so the ring stores
                // `(angle_deg + 45).to_radians()` (rotation only; the `+45`
                // is wisp-art-only — the bolt pass strips it back out).
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
                self.streams[slot] = WispStream {
                    lanim: -stream_hash01(self.seed, birth, 0, STREAM_HEAD_SALT) * 300.0,
                    langle: stream_hash01(self.seed, birth, 0, STREAM_ANGLE_SALT)
                        * std::f32::consts::TAU,
                    sound_played: false,
                };

                let proto = kind == SpiralKind::Proto;
                // GML `!bossfight` gate verbatim: no debris births while
                // the Throne-II fight runs (`Nothing2`/`Nothing2Appear`/
                // `NothingSpiral` alive); the port reads it off
                // `bossfight_suppressed`.
                let debris_ok = !self.bossfight_suppressed
                    && rand::random::<f32>() * 16.0 < 1.0
                    && (proto || rand::random::<f32>() * 3.0 < 1.0);
                if debris_ok {
                    if rand::random::<f32>() * 50.0 < 1.0
                        && let Some((path, frame)) = variant_debris_for_gml_area(self.gml_area)
                    {
                        self.push_vard(x, y, path, frame);
                    } else {
                        // GML `SpiralDebris/Create_0` runs at birth, then
                        // the SAME tick's `Step_0` integrates once before
                        // the first draw (`repeat 150` warmup steps self
                        // then motes, and live ticks create-then-step in
                        // order). A mote stored with xscale 0 and drawn
                        // next frame would pop full-size through the
                        // `frame + xscale/32` packing (see the ring write
                        // below); integrate once here so the birth frame
                        // already carries the first growth step.
                        let slot = self.dhead;
                        self.debris[slot] = Debris {
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
                            // GML `sprDebrisN` default arm: `image_index =
                            // random(image_number)` is float, but the
                            // ring packs `frame + xscale/32` and the
                            // shader splits with `floor`/`fract` — so
                            // the frame MUST be integral (bevy floors
                            // + clamps to 0..3 verbatim), else the
                            // frame fraction leaks into `fract` and
                            // newborns decode at xscale up to 32
                            // (the "debris spawns massive" bug).
                            frame: (rand::random::<f32>() * 4.0).floor().min(3.0),
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
            // Do NOT rewind births here — the shader indexes slots by
            // `(birth-1) % N`, so rewinding would orphan live wisps.
            // Scale fast-forward is carried by drain_bias instead.
            self.drain_bias += 5.5;
        }

        let drain = !self.alive;
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
            // Live-width cull like debris above (GML `view_width`).
            if dx + v.xstart < -16.0
                || dx + v.xstart > self.view_w + 16.0
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

    /// One GML `SpiralDebris/Step_0` integration over mote `i`
    /// (verbatim order: pos from current angle/radius, then
    /// angle/dist/grow/xscale advance, cull at view ± 16, ring write
    /// with the `frame + xscale/32` pack). Shared by the birth site
    /// (newborns step once in their birth tick, exactly like GML's
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
            if drain {
                d.grow *= 1.5;
            }
            d.grow *= d.xscale * 0.05 + 1.0;
            d.image_angle += d.rotspeed;
            // GML `Step_0` cull verbatim: view rect ± 16 — but against
            // the LIVE view width (`view_width`, 426 at 16:9), not the
            // 320 base. The old `GUI_W` bound killed side-drifting motes
            // up to 106px before they left the screen.
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
            // Parked sentinel: x < -100 (the ONLY component the shader
            // tests). Must be `[-1000, 0, 0, 0]` — `[-1000; 4]`
            // smuggles `frame=0, xscale=32` into slot 3, which the
            // shader unpacks as a FULL-SIZE rock (the "debris spawns
            // massive" bug).
            [-1000.0, 0.0, 0.0, 0.0]
        } else {
            [dx, dy, rot_rad, packed]
        };
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
    /// (bevy `background_color`); `bg_alpha` follows GML `scrDrawSpiral`
    /// verbatim (opaque everywhere except the campfire title; see the
    /// `bg_alpha` match at the `VortexPass` mount in `lib.rs`).
    /// Sound flags (`WispStream::sound_played`, `Debris::sound_played`)
    /// stay sim-side — the snapshot carries no audio, the shell drains
    /// the flags directly (GML plays them inline in the draw script).
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
        // GML `SpiralStar/Step_0` integrates draw pos on the mote
        // (`x = xstart + lengthdir_x(dist * xscale, angle)` with the
        // xscale-only radius on BOTH axes); the snapshot bakes the same
        // pos here so the shader stays a pure sampler.
        let mut stars = [[-1000.0, 0.0, 0.0, 0.0]; VORTEX_WISPS];
        for (dst, src) in stars.iter_mut().zip(self.stars.iter()) {
            if !src.alive {
                continue;
            }
            let rad = src.dist * src.xscale;
            let dir = src.angle.to_radians();
            *dst = [
                src.xstart + rad * dir.cos(),
                src.ystart - rad * dir.sin(),
                src.xscale,
                src.frame,
            ];
        }
        VortexSnapshot {
            wisps,
            debris,
            streams,
            stars,
            ticks: self.ticks,
            drain_bias: self.drain_bias,
            bg_rgb: [0.0, 0.0, 0.0],
            bg_alpha,
            thresh: self.thresh(),
            kindpacked: self.kindpacked(),
            // Live GUI view rect: `display_set_gui_size(view)` makes GUI
            // px == view px 1:1, so the fullscreen quad maps uv 1:1 onto
            // `(view_w, 240)` centered at `(view_w/2, 120)` — GML draws
            // wisps at `view + local`, i.e. screen px == GUI px.
            view: [self.view_w / 2.0, GUI_H / 2.0, self.view_w, GUI_H],
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

/// Wisp emitter position for the angle (GML `SpiralCont/Step_0`
/// orbit verbatim): view-centered (`_cx = view_width div 2`) with the
/// ±80x/±50y sine drift. `view_w` is the live GUI view width.
pub fn orbit(angle: f32, view_w: f32) -> (f32, f32) {
    (
        view_w / 2.0 + deg_sin(angle / 921.0) * deg_sin(angle / 500.0) * 80.0,
        GUI_H / 2.0 + deg_cos(angle / 583.0) * deg_sin(angle / 500.0) * 50.0,
    )
}

/// Legacy 320-base orbit (warmup/tests without a live view width).
pub fn orbit_base(angle: f32) -> (f32, f32) {
    orbit(angle, GUI_W)
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

#[cfg(test)]
mod vortex_ui_parity {
    use super::*;

    /// GML `Spiral/Step_0` growth recurrence verbatim: iterating
    /// `step_wisp_grow` 119 times from 0 must reproduce the shader's
    /// `SCALE_TABLE[119]` (2.539587) — the table the fullscreen pass
    /// sizes every wisp from.
    #[test]
    fn wisp_growth_matches_shader_table_tail() {
        let (mut grow, mut xs) = (0.0f32, 0.0f32);
        for _ in 0..119 {
            (grow, xs) = SpiralCtl::step_wisp_grow(grow, xs, false, false);
        }
        assert!(
            (xs - 2.539_587).abs() < 1e-3,
            "wisp xscale at age 119 diverged from SCALE_TABLE[119]: {xs}"
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
        dead.ticks += DRAIN_TICKS + 1.0;
        assert!(dead.is_done());
        world.insert_resource(dead);
        rewarm_view_spiral(&mut world);
        let ctl = world.resource::<SpiralCtl>();
        assert!(ctl.alive, "menu spiral must be live (fresh SpiralCont)");
        assert!(!ctl.is_done());
        assert_eq!(ctl.kind, SpiralKind::Normal);
    }
}
