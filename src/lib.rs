//! nt on repame: 30 Hz fixed-step sim + repose views, no bevy.
//!
//! Strangler layout: `sim/` (components, resources, systems),
//! `render/` (snapshot producers), `ui/`, `audio.rs`, `save.rs`,
//! `input.rs` land here slice by slice from `../nt-recreated-bevy`.
//!
//! Playable view wiring (this module):
//! - [`App`] owns the fixed-step [`Sim`] world, the full sim [`Schedule`]
//!   ([`build_sim_schedule`](crate::schedule::build_sim_schedule)), the
//!   [`SpiralCtl`] vortex state, the smoothed view [`Camera2d`], and an
//!   optional [`RenderAssets`] catalog.
//! - Per frame [`App::view`] (called by [`root_view`]) drains staged shell
//!   input into [`NtInput`], advances the sim at [`SIM_HZ`], then builds a
//!   repose view: sprite viewport ([`Viewport2dGpu`] with assets, canvas
//!   [`Viewport2d`] placeholder without), HUD [`Text`] overlay from
//!   [`hud_gui_texts_dp`](crate::render::hud_gui_texts_dp) (bevy
//!   `nt_hud_overlay` verbatim) and menu overlays from
//!   [`menu_gui_texts_dp`](crate::render::menu_gui_texts_dp).
//!
//! Assets resolution ([`resolve_assets_dir`], in order):
//! 1. `$NT_ASSETS` (must contain `images/anims.json`),
//! 2. `<exe-dir>/assets`,
//! 3. `<cwd>/assets`,
//! 4. `<cwd>/../nt-recreated-bevy/assets` (dev checkout convenience).
//! Use `cargo run` from the crate dir with the bevy checkout beside it, or
//! set `NT_ASSETS=/path/to/assets` explicitly:
//! `NT_ASSETS=../nt-recreated-bevy/assets cargo run`.
//! Loading is explicit ([`App::load_assets`], called by `main.rs`), so
//! [`App::new`] never touches disk and headless tests stay fast. Without
//! assets the game still runs: placeholder quads stand in for sprites and
//! no system depends on art.
//!
//! Input-event mapping (this repose version has no per-frame key polling:
//! [`Scheduler`]/[`RenderContext`] carry no input state; keys arrive via
//! `Modifier::on_key_event` on the focused root, pointer clicks via the
//! viewport `on_event` [`PickEvent`], both through the pilot-style raw-pointer
//! staging pattern):
//! - WASD/arrows -> move, mouse cursor position -> aim (every frame),
//!   Space / left-click -> fire, Shift / right-click -> ability+spec
//!   (GML `spec` on `mb_right`; Shift synthesizes the same codes from
//!   `KeyEvent.modifiers`), E/F/Q/G/Tab/Enter -> interact/confirm,
//!   1-5 -> weapon slots / mutation picks / title cursor / main-menu+
//!   pause rows, Left/Right arrows -> title cursor + mutation highlight,
//!   Up/Down (+Left/Right) -> main-menu cursor + settings cursor/values,
//!   click in game -> aim-at-click + fire, click in menus -> positional
//!   buttons (main-menu rows, title pods/GO/loadout, pause/settings/
//!   credits/stats buttons via screen-position routing; strays dropped),
//!   right-click in settings/credits -> Back (GML `BackButton`
//!   `mb_right`), Space on title -> loadout panel, Esc -> pause toggle
//!   (+ overlay unwind + main-menu overlay close), R -> game-over retry.
//! - NOT wired (needs shell services this repose version lacks): held-mouse
//!   continuous fire (clicks are single-frame edges; hover aims every
//!   frame), gamepad sticks/triggers (buttons/axes drain through
//!   `stage_gamepad`), text entry for profile/color inputs (buttons only;
//!   color cycles presets), key rebinding (REMAP screen is display-only).
//!
//! Fidelity notes: view-layer camera smoothing never feeds back into the
//! sim; [`SpiralCtl`] steps at the fixed cadence inside [`App::advance`]
//! (sim-pure); the sim schedule in `schedule.rs` is unchanged.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::vortex_pass::{VortexPass, VortexTexture};
use bevy_ecs::prelude::*;
use glam::Vec2;
use rand::RngExt;
use repame_anim::{AnimCatalog, AtlasDesc};
use repame_sim::{Sim, SimTime};
use repame_sprite::{
    BatchDesc, Camera2d, FrameInput, GeomHandle, PickEvent, SpriteBatch, SpriteInstance,
    Viewport2d, Viewport2dGpu,
};
use repose_canvas::Embedded;
use repose_core::PaddingValues;
use repose_core::input::{
    Key, KeyEvent, KeyEventType, PointerButton, PointerEvent, PointerEventKind,
};
use repose_core::prelude::{AlignItems, Modifier};
use repose_core::{
    Color, Dp, FocusRequester, RenderContext, Scheduler, Sp, View, remember, request_frame,
};
use repose_render_wgpu::Callback;
use repose_ui::{AnnotatedText, Box as UiBox, Column, Text, TextStyle, ViewExt, ZStack};

use crate::audio::UiAction;
use crate::comps_a::CurrentFrame as CombatFrame;
use crate::comps_a::{NT_CAM_SCALE, Player, Projectile, WallCell, WallTile};
use crate::comps_b::{Enemy, Pickup, Prop};
use crate::data::AreaId;
use crate::input::{
    GamepadState, KeyCode, MouseState, NtInput, TouchContact, sample_gamepads, sample_keyboard,
    sample_touch,
};
use crate::render::{
    ATLAS_PAGES, ATLAS_SIZE, CamPoi, CamStepInput, GmlCamera, RenderAssets, Z_BLOOM, Z_CROSSHAIR,
    Z_FAINTED, Z_FOG, Z_FX, Z_HUD, Z_MENU, Z_PORTAL_INDICATOR, Z_SHADOW, Z_SIDEART,
    Z_SPIRAL_FIGURES, Z_SPLASH, background_color, bloom_sprites, cam_viewdist_for,
    crosshair_sprites, decode_png, fainted_bar_sprites, fog_sprites, fx_instances, fx_texts,
    gml_camera_step, gml_view_scale, gml_view_size, hud_gui_texts_dp, hud_sprites, menu_gui_texts,
    menu_gui_texts_dp, menu_gui_texts_vw, menu_sprites, portal_indicator_sprites, shadow_sprites,
    sideart_sprites, spiral_figures, splash_sprites, stamp_z, view_rect_world, world_camera,
    world_instances,
};
use crate::schedule::build_sim_schedule;
use crate::setup::setup_run_with_seed;
use crate::spatial::Pos;
use crate::state::menus::{MenuEdge, MenuState, apply_menu_action};
use crate::state::{AppState, OverlayMenu};
use crate::vortex::{SpiralCtl, gml_area_for_area};

pub mod anim;
pub mod audio;
pub mod boss_ai;
pub mod combat;
pub mod comps_a;
pub mod comps_b;
pub mod crown;
pub mod data;
mod dead_path_part;
pub mod deaths;
pub mod effects;
pub mod enemies;
pub mod enemy_data;
pub mod environment;
pub mod hud;
pub mod idpd;
mod ids_part;
pub mod input;
pub mod loop_transition;
pub mod msg;
pub mod pickups;
pub mod player;
pub mod player_fire;
pub mod progression;
pub mod projectile_math;
pub mod render;
pub mod savedata_part;
pub mod schedule;
pub mod secrets;
pub mod setup;
pub mod spatial;
pub mod spawns;
pub mod state;
pub mod time;
pub mod vortex;
pub mod vortex_pass;
pub mod walls;
pub mod weapon_runtime;
pub mod weapons_data;
pub mod worldgen;

pub use ids_part::{EnemyKind, MutationId, UltraMutationId};

/// Fixed sim rate, matching the bevy build (`NT_SIM_HZ = 30`).
pub const SIM_HZ: f64 = 30.0;

/// Pixel UI font bytes (bevy `app.rs::NT_UI_FONT` verbatim: Silkscreen,
/// the fntM1 stand-in; OFL text beside it in `assets/fonts/`).
pub const NT_UI_FONT: &[u8] = include_bytes!("../assets/fonts/Silkscreen-Regular.ttf");
/// Family name the overlay [`Text`](repose_ui::Text) views request.
pub const NT_UI_FONT_FAMILY: &str = "Silkscreen";

/// Register the pixel UI font once (idempotent across boots/tests).
pub fn ensure_nt_font() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        repose_text::register_font_data(NT_UI_FONT);
    });
}

/// Steps run since boot (headless-observable sim heartbeat).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Resource)]
pub struct TickCount(pub u64);

/// Staged pointer click: world position (aim-at-click + fire) plus
/// canvas-dp position (menu button hit-testing, camera-independent:
/// overlay text is screen-anchored through the live GML GUI map, so
/// world coords would unproject through the wrong frame).
#[derive(Clone, Copy, Debug)]
struct StagedClick {
    world: Vec2,
    dp: [f32; 2],
}

/// The game: fixed-step sim plus presentational (view-layer) state.
///
/// `sim`/`schedule`/`spiral` are sim-pure; `cam`, `held`/`edges`/`clicks`/
/// `hover` and the asset catalog are view-layer staging. [`App::advance`]
/// runs the sim; [`App::view`] stages input, advances, then composes
/// the frame.
pub struct App {
    pub sim: Sim,
    schedule: Schedule,
    accum: Duration,
    /// Smoothed follow camera (VIEW layer only — sim stays fixed-step pure).
    pub cam: Camera2d,
    /// GML `BackCont` camera state verbatim (view-layer state).
    gml_cam: GmlCamera,
    /// Previous frame's floor-transition flag (generation end snaps).
    was_transitioning: bool,
    /// Previous frame's app state (entering InGame snaps the camera on
    /// the fresh player instead of swooping from the menu look point).
    was_state: AppState,
    /// Previous fixed step's app state (drives the spiral lifecycle in
    /// [`App::advance`]: Loading entry re-warms, InGame entry kills —
    /// separate from the view-side `was_state`, which updates a frame
    /// later in [`App::view`]).
    adv_state: AppState,
    /// Previous fixed step's generation-cover flag (floor transition or
    /// mutation/ultra offer). Rising edge re-warms the spiral
    /// (`GenCont`/`LevCont` build a fresh `SpiralCont`); falling edge
    /// kills it (`GenCont/Destroy` destroys it at generation end).
    adv_cover: bool,
    /// Run seed the spiral was last warmed/killed for. `setup_run`
    /// writes `AppState::InGame` directly mid-schedule (not via an
    /// edge the lifecycle can see), so the InGame-entry kill keys off
    /// a seed change here instead of `adv_state`.
    adv_seed: u64,
    /// GML area the spiral was last warmed/killed for. The drift area
    /// arm compares against this stamp (not the live spiral's area):
    /// the Loading warmup carries the menu-room area, and without the
    /// stamp the first InGame tick reads area drift and rewarms right
    /// after the entry kill (the load-end double start).
    adv_area: u8,
    spiral: SpiralCtl,
    assets: Option<RenderAssets>,
    /// Art dir the catalog loaded from (vortex background textures
    /// decode from here on area switches).
    assets_dir: Option<PathBuf>,
    /// Decoded vortex background textures for `vortex_tex_area`
    /// (slots: spiral, bolt, debris, proto, idpd, idpd2).
    vortex_tex: Vec<VortexTexture>,
    vortex_tex_area: Option<u8>,
    // -- staged shell input (drained into `NtInput`/`MenuEdge` per frame) --
    held: HashSet<KeyCode>,
    edges: Vec<KeyCode>,
    clicks: Vec<StagedClick>,
    /// Latest cursor world position (viewport `Hover`, single slot: only
    /// the newest position matters). Steers `aim_axis` every frame in
    /// game, bevy `player_aim` mouse-path parity.
    hover: Option<Vec2>,
    /// Latest cursor position in window-physical px (root
    /// `on_pointer_move`, y-down). Unlike [`App::hover`] (a world point
    /// baked through the camera at event time), this stays valid as the
    /// camera moves: [`App::feed_input`] unprojects it through the
    /// *current* camera each frame — bevy `player_aim`
    /// (`window.cursor_position()` + `viewport_to_world_2d`) parity. The
    /// player therefore keeps aiming at the on-screen cursor while
    /// walking, instead of at a stale world point behind them.
    cursor_px: Option<Vec2>,
    pause_edge: bool,
    restart_edge: bool,
    interact_edge: bool,
    /// Physical keyboard Shift held (tracked from `KeyEvent.modifiers`;
    /// `Key` has no Shift variant, so this owns the release edge the
    /// synthesized [`KeyCode::ShiftLeft`] cannot).
    shift_held: bool,
    /// Right mouse held (GML `spec`/ability on `mb_right` parity).
    rmb_held: bool,
    /// Left mouse held (GML `fire` on `mb_left` parity): set on primary
    /// pointer-down, cleared on primary pointer-up, so automatic weapons
    /// keep firing while held (clicks alone are single-frame edges).
    lmb_held: bool,
    /// Right-button down/up edges staged this window. The viewport emits
    /// buttonless [`PickEvent`]s for right clicks too, so these flags let
    /// [`App::feed_input`] drop the spurious picks (and route Back over
    /// settings/credits like GML `BackButton`).
    rmb_down_edge: bool,
    rmb_up_edge: bool,
    /// Staged gamepad snapshots, one per pad (bevy `sample_input`
    /// `gamepads` query order verbatim: keyboard, then pads in order,
    /// then touch).
    pads: Vec<GamepadState>,
    /// Live touch/pen contacts by pointer id: (touchdown, current),
    /// screen-px y-down (bevy `Touches` parity — `feed_input`
    /// synthesizes the per-frame contact list from this map, so held
    /// contacts keep steering like bevy's `touches.iter()`).
    touch_active: HashMap<u64, (Vec2, Vec2)>,
    /// Contacts that began this tick (bevy `iter_just_pressed` parity).
    touch_new: HashSet<u64>,
    /// Viewport width in screen px for the touch button zones (bevy
    /// reads `window.width()`; refreshed from the frame geometry).
    view_width: f32,
    /// Last frame's canvas-dp viewport + dp scale (menu button
    /// hit-testing runs in this space; refreshed every frame).
    view_viewport_dp: [f32; 2],
    view_density: f32,
    /// Live GML view size in world px ([`gml_view_size`]: 426x240 at
    /// 16:9, refreshed every frame; the fixed-step camera reads the
    /// last one so headless ticks have a stable window).
    view_world_size: [f32; 2],
    /// Staged menu button actions (pause/settings/credits `on_click`
    /// handlers stage here; drained through [`apply_menu_action`] at
    /// the top of [`App::feed_input`], before the sim advances).
    menu_actions: Vec<UiAction>,
    // -- last-frame diagnostics (headless-observable) --
    last_sprite_count: usize,
    last_bg_alpha: f32,
}

impl App {
    /// Game boot (what `main` uses): Splash state, no run yet. The menu
    /// state machine drives Splash -> MainMenu -> Title -> Loading, and
    /// Loading runs `setup_run` into InGame (bevy `app.rs` flow).
    pub fn new() -> Self {
        Self::boot(rand::random(), false)
    }

    /// Deterministic boot (tests pin the floor seed). Never touches disk:
    /// assets load explicitly via [`App::load_assets`].
    /// Starts straight inside a live run (bevy parity: as if Loading just
    /// finished `setup_run`).
    pub fn new_with_seed(seed: u64) -> Self {
        Self::boot(seed, true)
    }

    fn boot(seed: u64, setup: bool) -> Self {
        ensure_nt_font();
        let mut sim = Sim::new(Duration::from_secs_f64(1.0 / SIM_HZ));
        sim.world.init_resource::<TickCount>();
        init_schedule_resources(&mut sim.world);
        // `Sim` owns the heartbeat tick only; the full sim (including the
        // `state::CurrentFrame` tick) runs in `schedule` via `advance`.
        sim.add_system(tick_counter);
        if setup {
            setup_run_with_seed(&mut sim.world, seed);
        }
        // Mirror the frame counter for combat readers (same as the
        // schedule test harness; combat systems read `comps_a` copy).
        let frame = sim.world.resource::<state::CurrentFrame>().0;
        sim.world.resource_mut::<CombatFrame>().0 = frame;

        let area = sim
            .world
            .get_resource::<crate::comps_a::Run>()
            .map(|r| r.area)
            .unwrap_or(AreaId::Desert);
        let spiral = SpiralCtl::warmed_up_for_area_seeded(area, seed);
        // GML `BackCont` boot: the 320x240-base view opens snapped
        // on the player (`force_snap_camera_position` on generation
        // end; GML has no zoom).
        let cam = world_camera(
            player_pos(&mut sim.world).unwrap_or(Vec2::ZERO),
            NT_CAM_SCALE,
        );

        Self {
            sim,
            schedule: build_sim_schedule(),
            accum: Duration::ZERO,
            cam,
            gml_cam: GmlCamera {
                snap: true,
                ..Default::default()
            },
            was_transitioning: false,
            was_state: AppState::default(),
            adv_state: AppState::default(),
            adv_cover: false,
            adv_seed: 0,
            adv_area: 0,
            spiral,
            assets: None,
            assets_dir: None,
            vortex_tex: Vec::new(),
            vortex_tex_area: None,
            held: HashSet::new(),
            edges: Vec::new(),
            clicks: Vec::new(),
            hover: None,
            cursor_px: None,
            pause_edge: false,
            restart_edge: false,
            interact_edge: false,
            shift_held: false,
            rmb_held: false,
            lmb_held: false,
            rmb_down_edge: false,
            rmb_up_edge: false,
            pads: Vec::new(),
            touch_active: HashMap::new(),
            touch_new: HashSet::new(),
            view_width: 1280.0,
            view_viewport_dp: [1280.0, 720.0],
            view_density: 1.0,
            view_world_size: [426.0, 240.0],
            menu_actions: Vec::new(),
            last_sprite_count: 0,
            last_bg_alpha: 1.0,
        }
    }

    /// Load the full art catalog from [`resolve_assets_dir`]. Call once
    /// from `main`; returns the dir used. Failure leaves the graceful
    /// placeholder path in place (game logic never depends on art).
    pub fn load_assets(&mut self) -> anyhow::Result<PathBuf> {
        let dir =
            resolve_assets_dir().ok_or_else(|| anyhow::anyhow!("no assets dir (set NT_ASSETS)"))?;
        self.load_assets_from(&dir)?;
        Ok(dir)
    }

    /// Load the full art catalog from an explicit dir (tests + shells).
    pub fn load_assets_from(&mut self, dir: &Path) -> anyhow::Result<()> {
        let assets = RenderAssets::load(dir)?;
        let json = std::fs::read_to_string(dir.join("images").join("anims.json"))?;
        let catalog = AnimCatalog::from_json(
            &json,
            AtlasDesc {
                size: ATLAS_SIZE,
                max_pages: ATLAS_PAGES,
                padding: 0,
            },
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Future spawns (next floors, portal waves) resolve art variants
        // against the full catalog; existing entities keep recorded paths.
        self.sim.world.insert_resource(catalog);
        self.assets = Some(assets);
        self.assets_dir = Some(dir.to_path_buf());
        self.vortex_tex.clear();
        self.vortex_tex_area = None;
        Ok(())
    }

    /// Vortex background textures for a GML area (headless-observable;
    /// same decode the live frame uses on area switches).
    pub fn debug_vortex_textures(dir: &Path, gml_area: u8) -> Vec<VortexTexture> {
        Self::load_vortex_textures(dir, gml_area)
    }

    /// Vortex background textures for a GML area (decoded once per
    /// area; slots match the shader bindings: spiral, bolt, debris,
    /// proto, idpd, idpd2, star). Missing files fall back to area 0 debris
    /// so the pass always has seven bound slots.
    fn load_vortex_textures(dir: &Path, gml_area: u8) -> Vec<VortexTexture> {
        fn decode(dir: &Path, name: &str) -> Option<(u32, u32, Vec<u8>)> {
            decode_png(&dir.join("images").join(format!("{name}.png"))).ok()
        }
        // GML `SpiralDebris/Create_0` verbatim: the mote strip is
        // `sprDebris + GameCont.area` — the FULL GML area number, not the
        // art variant. Campfire (area 0) reuses `sprDebris0`; secret areas
        // use their own (`sprDebris101` …). Never `sprDebris1` here.
        let debris = decode(dir, &format!("sprDebris{gml_area}"))
            .or_else(|| decode(dir, "sprDebris0"))
            .unwrap_or((1u32, 1u32, vec![255, 255, 255, 255]));
        let mut out = Vec::new();
        for (slot, stem) in [
            "sprSpiral",
            "sprPortalLightning",
            "",
            "sprSpiralProto",
            "sprSpiralIDPD",
            "sprSpiralIDPD2",
            "sprSpiralStar",
        ]
        .into_iter()
        .enumerate()
        {
            let (w, h, rgba) = if stem.is_empty() {
                debris.clone()
            } else {
                decode(dir, stem).unwrap_or((1u32, 1u32, vec![255, 255, 255, 255]))
            };
            out.push(VortexTexture {
                slot: slot as u32,
                w,
                h,
                rgba,
            });
        }
        out
    }

    pub fn has_assets(&self) -> bool {
        self.assets.is_some()
    }

    pub fn last_sprite_count(&self) -> usize {
        self.last_sprite_count
    }

    pub fn last_bg_alpha(&self) -> f32 {
        self.last_bg_alpha
    }

    /// Spiral snapshot for one rendered frame (headless-observable).
    pub fn spiral_snapshot(&self, bg_alpha: f32) -> crate::vortex_pass::VortexSnapshot {
        self.spiral.snapshot(bg_alpha)
    }

    /// Spiral liveness for the vortex mount decision (headless
    /// diagnostics, same pattern as `last_sprite_count`).
    pub fn spiral_alive(&self) -> bool {
        self.spiral.alive
    }

    /// Spiral drain completion (mount decision: a done spiral stays
    /// gone until the next cover re-warms it).
    pub fn spiral_done(&self) -> bool {
        self.spiral.is_done()
    }

    /// Spiral clock (headless freeze assertion).
    pub fn spiral_ticks(&self) -> f32 {
        self.spiral.ticks
    }

    /// Live GUI view width the spiral simulates in (headless diagnostics).
    pub fn spiral_view_w(&self) -> f32 {
        self.spiral.view_w
    }

    /// Spiral emitter angle in degrees (headless motion diagnostics).
    pub fn spiral_angle(&self) -> f32 {
        self.spiral.angle
    }

    /// Bolt-visible wisp count in the current snapshot (headless
    /// lightning diagnostics: GML `lanim in (0, 6)` per wisp).
    pub fn spiral_bolts(&self) -> usize {
        self.spiral
            .streams
            .iter()
            .filter(|s| s.bolt_visible())
            .count()
    }

    /// Live debris motes in the current ring (headless cull diagnostics).
    pub fn spiral_debris_live(&self) -> usize {
        self.spiral.debris.iter().filter(|d| d.alive).count()
    }

    /// Drain fast-forward bias (headless drain diagnostics).
    pub fn spiral_drain_bias(&self) -> f32 {
        self.spiral.drain_bias
    }

    /// Live star motes (headless drain diagnostics).
    pub fn spiral_stars_live(&self) -> usize {
        self.spiral.stars.iter().filter(|s| s.alive).count()
    }

    /// Live variant-debris motes (headless drain diagnostics).
    pub fn spiral_vards_live(&self) -> usize {
        self.spiral.vards.iter().filter(|v| v.alive).count()
    }

    /// Advance wall-clock time into fixed steps. Returns steps run.
    ///
    /// Each step: `Sim::tick` (clock + heartbeat), frame-counter mirror,
    /// full sim schedule, one vortex tick. Area switches re-warm the
    /// spiral (sim-pure; view smoothing never feeds back here).
    pub fn advance(&mut self, dt: Duration) -> u32 {
        self.accum += dt;
        let mut ran = 0;
        while self.accum >= self.sim.step {
            self.accum -= self.sim.step;
            self.sim.tick();
            let frame = self.sim.world.resource::<state::CurrentFrame>().0;
            self.sim.world.resource_mut::<CombatFrame>().0 = frame;
            self.schedule.run(&mut self.sim.world);
            // GML `BackCont/Step_0`: the follow camera steps once per
            // game step (30 Hz), never per render frame, and freezes
            // over pause/menus/game over.
            self.step_camera_fixed();
            // GML `Nothing2/Create_0` sets `bossfight` on its fresh
            // `SpiralCont`: no debris births while Throne II runs (any
            // phase — pending spawn, live fight, death pageant).
            self.spiral.bossfight_suppressed = self
                .sim
                .world
                .query::<&crate::comps_b::Enemy>()
                .iter(&self.sim.world)
                .any(|e| e.kind == crate::data::EnemyKind::ThroneII)
                || self
                    .sim
                    .world
                    .query::<&crate::comps_b::CampfireState>()
                    .iter(&self.sim.world)
                    .any(|c| matches!(c.phase, crate::comps_b::CampfirePhase::SpawnThroneII));
            // Pause freezes the spiral: GML `UberCont/Step_1`
            // `instance_deactivate_all` leaves `SpiralCont` (not in the
            // activate list) frozen while the pause menu is up. Sounds
            // drain below regardless (already gated on audibility).
            let paused = self
                .sim
                .world
                .get_resource::<crate::state::Paused>()
                .is_some_and(|p| p.0);
            if !paused {
                self.spiral.step(1.0);
            }
            // GML `scrDrawSpiral` bolt/debris one-shots fire inline in
            // the draw script: lightning `sndPortalLightning{1..8}` once
            // per wisp (non-menu only), flyby `sndPortalFlyby{1..4}` once
            // per debris mote at `xscale > 1.3` (any caller). Drain here,
            // right after the step, so each fires exactly once.
            self.drain_spiral_sounds();
            // Lifecycle FIRST, drift second: the drift rewarm skips the
            // InGame-entry tick itself (see maybe_rewarm_spiral), so the
            // entry kill lands on the post-transition spiral while
            // settled-state drift still re-warms on real portal/new-run
            // area/seed changes.
            self.step_spiral_lifecycle();
            self.maybe_rewarm_spiral();
            ran += 1;
        }
        ran
    }

    /// Spiral lifecycle per fixed step (GML `SpiralCont` create/destroy
    /// parity, bevy `mark_vortex_dead` / `ensure_spiral_for_levelup`
    /// parity — the view-layer half; the sim `SpiralCtl` resource that
    /// the ambience duck keys off is untouched):
    /// - Entering Loading warms a FRESH spiral for the run (`Vlambeer`
    ///   builds a fresh `SpiralCont` + `GenCont` on every `room_restart`,
    ///   including RETRY — a continued mid-flight spiral would jump).
    /// - Entering InGame kills it (`GenCont/Destroy` destroys the cont
    ///   at generation end; the 26-tick drain plays out like bevy's).
    /// - A rising generation cover (floor transition or mutation/ultra
    ///   offer) re-warms (`GenCont`/`LevCont` rooms build theirs); the
    ///   falling edge kills (`GenCont/Destroy` at the end of the
    ///   transition). Live gameplay, Title (no `SpiralCont` in the
    ///   `MenuGen` room) and GameOver therefore show no spiral — just
    ///   the flat area colour, exactly like GML.
    fn step_spiral_lifecycle(&mut self) {
        let state = self
            .sim
            .world
            .get_resource::<AppState>()
            .copied()
            .unwrap_or_default();
        let run = self.sim.world.get_resource::<crate::comps_a::Run>();
        let (area, seed) = run
            .map(|r| (r.area, r.gen_seed))
            .unwrap_or((AreaId::Desert, 0));
        // Bevy `mark_vortex_dead` is a one-way latch per tick: a kill
        // and a rewarm must NEVER both fire in one call. On the
        // `setup_run` tick the fresh seed arms the entry kill below AND
        // a pending pick raises the cover edge further down — without
        // the latch the cover rewarm rebuilds a live 150-tick spiral in
        // the same tick the kill just drained (the load-end double
        // start, ticks 185→150 over live play).
        let mut killed_this_tick = false;
        if state == AppState::Loading && self.adv_state != AppState::Loading {
            self.spiral = SpiralCtl::warmed_up_for_area_seeded(area, seed);
            self.adv_seed = seed;
            self.adv_area = gml_area_for_area(area);
        }
        // `setup_run` writes InGame directly mid-schedule (no observable
        // edge: `tick_loading` → `setup_run` runs inside the same fixed
        // step, so `adv_state` is already InGame here). Key the entry
        // kill off the run-seed change instead: a fresh seed in InGame
        // means a new run just landed (`GenCont/Destroy` destroys the
        // cont at generation end). The drift rewarm below runs after
        // and only tracks settled states, so it cannot resurrect this
        // kill on later ticks (seed matches by then).
        if state == AppState::InGame && seed != self.adv_seed {
            self.spiral.kill();
            self.adv_seed = seed;
            self.adv_area = gml_area_for_area(area);
            killed_this_tick = true;
        }
        // GML `Vlambeer/Create_0` `want_quit_to_menu` branch verbatim:
        // quitting to the logo menu builds a FRESH live `SpiralCont`
        // (`instance_create(x, y, SpiralCont)` with its `repeat 150`
        // warmup) — never the previous run's leftover drain. That fresh
        // cont is already built by the quit ACTION arms
        // (`ConfirmPause(0)` / `QuitToTitle` call `rewarm_view_spiral`
        // before `goto_state`); the lifecycle only covers direct
        // `goto_state(MainMenu)` shells like tests. It must NOT fire on
        // the boot Splash→MainMenu edge: the boot spiral has stepped
        // since launch and GML's logo-room cont (created once in
        // `Vlambeer/Alarm_0`) is the same object still swirling — a
        // rewarm here restarts the vortex under the PLAY rows (the
        // "vortex starts twice" bug).
        if state == AppState::MainMenu
            && self.adv_state != AppState::MainMenu
            && self.adv_state != AppState::Splash
            && !self.spiral.alive
        {
            self.spiral = SpiralCtl::warmed_up_for_area_seeded(AreaId::Campfire, seed);
        }
        // GML `PlayButton/Other_10:108` verbatim: entering the campfire
        // char-select DESTROYS the `SpiralCont` (`instance_destroy`) while
        // the leftover `Spiral/SpiralDebris/SpiralStar` motes keep stepping
        // in a cont-less room (drain growth 1.5x, wisp kill-plane 3.0, no
        // new births — but `lanim` keeps realtime cadence, so bolts keep
        // flashing). The `Menu/Draw_0` `scrDrawSpiral` call draws that
        // still-live remnant transparently (no `draw_clear`) over the
        // campfire camp. Only once every mote is culled does the flat
        // camp show. Killing (not re-warming) here reproduces it; the
        // layer stays mounted until `is_done` (~a second of visible
        // remnant, exactly like GML).
        if state == AppState::Title && self.adv_state != AppState::Title {
            self.spiral.kill();
            killed_this_tick = true;
        }
        let cover = state == AppState::InGame
            && (self
                .sim
                .world
                .get_resource::<crate::comps_b::FloorTransition>()
                .is_some_and(|f| f.active)
                || self
                    .sim
                    .world
                    .get_resource::<crate::comps_a::PendingMutation>()
                    .is_some()
                || self
                    .sim
                    .world
                    .get_resource::<crate::comps_a::PendingUltra>()
                    .is_some());
        // Latch: a tick that killed never rewarms. The cover edge still
        // records so the NEXT tick rewarms if the cover is genuinely
        // held (one tick of delay, invisible); the falling edge still
        // kills (already dead — no-op).
        if killed_this_tick {
            // no rewarm this tick
        } else if cover && !self.adv_cover {
            self.spiral = SpiralCtl::warmed_up_for_area_seeded(area, seed);
        } else if !cover && self.adv_cover {
            self.spiral.kill();
        }
        self.adv_state = state;
        self.adv_cover = cover;
    }

    /// One fixed-step GML camera step (`objects/BackCont/Step_0.gml`).
    /// Runs inside [`App::advance`], once per sim tick: POI pull + aim
    /// lean + shake sample feed [`gml_camera_step`] at exactly 30 Hz
    /// (per-frame stepping + per-frame `round()` juddered at 60+ fps).
    /// Frozen unless the run is live (InGame, unpaused, no overlay, no
    /// game over) so pause/menus keep the last look point.
    fn step_camera_fixed(&mut self) {
        let live = self
            .sim
            .world
            .get_resource::<AppState>()
            .is_some_and(|s| *s == AppState::InGame)
            && !self
                .sim
                .world
                .get_resource::<crate::state::Paused>()
                .is_some_and(|p| p.0)
            && self
                .sim
                .world
                .get_resource::<OverlayMenu>()
                .is_none_or(|o| *o == OverlayMenu::None)
            && !self
                .sim
                .world
                .get_resource::<crate::comps_a::Run>()
                .is_some_and(|r| r.game_over);
        if !live {
            return;
        }
        let rest = self.cam.center;
        // (`cursor_to_world` borrows `self.cam` only, so resolve the
        // live cursor before the `world` borrow below.)
        let live_hover = self.cursor_to_world().or(self.hover);
        let world = &mut self.sim.world;
        let player = player_pos(world).unwrap_or(rest);
        // Current weapon drives the aim-lean divisor (melee 8, bolts
        // 3, else 4).
        let wep = world
            .query::<(&Pos, &crate::comps_a::Player, &crate::comps_a::Inventory)>()
            .iter(world)
            .next()
            .map(|(_, _, inv)| inv.weapons[0])
            .unwrap_or(crate::data::WeaponId::NONE);
        // GML `KeyCont.dis_fire`: cursor distance in world px. The
        // lean caps at 48 px (bevy `player_aim` `MAX_LOOK` parity: the
        // playable builds clamp the lookahead there; unbounded
        // `dis/viewdist` drifts whole screens when the cursor sits at
        // a window edge and never feels like the original). Uses the
        // live cursor unprojection (see `cursor_to_world`), so the lean
        // follows the on-screen cursor as the camera moves.
        // (`cursor_to_world` borrows `self.cam` only, so it was
        // resolved into `live_hover` before the `world` borrow above.)
        let (aim_dir, aim_dis) = match live_hover {
            Some(h) => {
                let d = h - player;
                let len = d.length();
                if len > 1e-6 {
                    (d / len, len)
                } else {
                    (Vec2::X, 0.0)
                }
            }
            None => (Vec2::X, 0.0),
        };
        let aim_dis = aim_dis.min(crate::render::CAM_MAX_LOOK);
        // GML POI chain: Portal, then victory/sit markers (the
        // `BecomeNothing`/`NothingDeath` kinds have no port counterpart
        // yet) — nearest instance wins, cap only for portals.
        let poi = nearest_poi(world, player);
        let shake_scale = world
            .get_resource::<crate::savedata_part::SaveData>()
            .map(|s| s.settings.screenshake.clamp(0.0, 2.0))
            .unwrap_or(1.0);
        // Trauma feeds GML `BackCont.shake` in px (max 20 at full
        // trauma). Trauma owns the decay (1.5/s in `step_fx`), so the
        // GML-law decay inside the step is parked (`timescale = 0`).
        let shake_px = world
            .get_resource::<repame_fx::Trauma>()
            .map(|t| t.amount * t.max_translation_px)
            .unwrap_or(0.0);
        self.gml_cam.shake = shake_px;
        let mut rng = rand::rng();
        let step_in = CamStepInput {
            player,
            aim_dir,
            aim_dis,
            viewdist: cam_viewdist_for(wep),
            poi,
            shake_scale,
            timescale: 0.0,
            jx: rng.random_range(-1.0..1.0),
            jy: rng.random_range(-1.0..1.0),
        };
        const STEP_DT: f32 = 1.0 / SIM_HZ as f32;
        let vw_vh = self.view_world_size;
        gml_camera_step(&mut self.gml_cam, vw_vh[0], vw_vh[1], &step_in, STEP_DT);
    }

    fn maybe_rewarm_spiral(&mut self) {
        // GML `room_restart` parity: a LIVE gameplay area/seed change
        // (portal, new run) re-warms the spiral for the new room —
        // `Vlambeer/Create_0` builds a fresh `SpiralCont` on every
        // gameplay restart. Menu rooms never restart the vortex: the
        // logo-room cont is created once (`Vlambeer/Alarm_0`) and the
        // campfire room inherits the drain (`PlayButton/Other_10`
        // destroys the cont; `Menu/Draw_0` draws the leftovers). So a
        // menu-room resource reset (`setup_logo_room` on the
        // Splash→MainMenu edge rewriting Desert/seed into
        // Campfire/seed-0) must NOT read as area/seed drift — that
        // restarted the vortex under the PLAY rows (the "vortex starts
        // twice" bug). The lifecycle step owns the menu transitions;
        // this only tracks live gameplay drift.
        let state = self
            .sim
            .world
            .get_resource::<AppState>()
            .copied()
            .unwrap_or_default();
        if !matches!(state, AppState::Loading | AppState::InGame) {
            return;
        }
        // The lifecycle owns the InGame-entry kill above (it stamps
        // `adv_seed` on the kill tick), so drift here only fires on a
        // LATER seed change — a real portal/new-run area/seed change in
        // settled play. Comparing against the lifecycle's stamp (not
        // the spiral's own seed) is what stops the load-end double
        // start: on the entry tick both read the fresh seed and the
        // rewarm stays quiet.
        let run = self.sim.world.get_resource::<crate::comps_a::Run>();
        let (area, seed) = run
            .map(|r| (r.area, r.gen_seed))
            .unwrap_or((AreaId::Desert, 0));
        if seed != self.adv_seed {
            // Fresh seed in InGame = a new run just landed: GML
            // `GenCont/Destroy` destroys the cont at generation end, so
            // kill (drain), never rewarm. The mid-run floor swap does
            // NOT change the seed (`tick_floor_transition` keeps it;
            // only secret/loop routing re-derives it and those ride a
            // cover rewarm below), so this arm cannot fire in settled
            // play.
            self.spiral.kill();
            self.adv_seed = seed;
        } else if gml_area_for_area(area) != self.adv_area {
            // Same run, new area art (portal kept the seed): re-warm so
            // the debris strip follows the GML area. Compares against
            // the lifecycle stamp, NOT the live spiral: the Loading
            // warmup carries the menu-room area, so the first InGame
            // tick would otherwise read area drift and rewarm right
            // after the entry kill (the load-end double start).
            self.spiral = SpiralCtl::warmed_up_for_area_seeded(area, seed);
            self.adv_area = gml_area_for_area(area);
        }
    }

    /// Drain the spiral one-shots after each fixed step (GML
    /// `scrDrawSpiral` plays them inline while drawing): bolt
    /// `sndPortalLightning{1..8}` once per wisp (the draw script gates
    /// on `!_is_menu`, i.e. every state but the campfire title — the
    /// same `bg_alpha == 1` set the snapshot uses), flyby
    /// `sndPortalFlyby{1..4}` once per debris mote at `xscale > 1.3`.
    /// Variant/stem rolls use the spiral's deterministic stream off the
    /// run seed so equal seeds sound identical.
    fn drain_spiral_sounds(&mut self) {
        use crate::audio::AudioCue;
        use crate::msg::Queue;
        // GML `draw_clear` caller: everyone but `Menu`. The port's
        // `bg_alpha` is 0 only on the title, so reuse the last snapshot
        // decision (updated every `view`; defaults to sounding before
        // the first frame, exactly like a fresh GML room entering draw).
        let audible = self.last_bg_alpha > 0.5;
        let seed = self.spiral.seed;
        let mut cues: Vec<AudioCue> = Vec::new();
        for (i, s) in self.spiral.streams.iter_mut().enumerate() {
            if audible && s.bolt_sound_due() {
                let n = 1 + (crate::vortex::stream_pick(seed, i as u32, 7) % 8);
                cues.push(AudioCue {
                    name: match n {
                        1 => "sndPortalLightning1",
                        2 => "sndPortalLightning2",
                        3 => "sndPortalLightning3",
                        4 => "sndPortalLightning4",
                        5 => "sndPortalLightning5",
                        6 => "sndPortalLightning6",
                        7 => "sndPortalLightning7",
                        _ => "sndPortalLightning8",
                    },
                    // GML `snd_play(_sound, 0.9 + random(0.2), 1)`.
                    volume: 1.0,
                    variance: 0.0,
                });
            }
        }
        for (i, d) in self.spiral.debris.iter_mut().enumerate() {
            if d.flyby_due() {
                let n = 1 + (crate::vortex::stream_pick(seed, 10_000 + i as u32, 11) % 4);
                let vol = self
                    .sim
                    .world
                    .get_resource::<crate::savedata_part::SaveData>()
                    .map(|s| s.settings.ambience_volume)
                    .unwrap_or(1.0);
                cues.push(AudioCue {
                    name: match n {
                        1 => "sndPortalFlyby1",
                        2 => "sndPortalFlyby2",
                        3 => "sndPortalFlyby3",
                        _ => "sndPortalFlyby4",
                    },
                    // GML `snd_play_pitchvol(_snd, 0.1, opt_ambvol)`.
                    volume: vol,
                    variance: 0.1,
                });
            }
        }
        if !cues.is_empty() {
            self.sim.world.init_resource::<Queue<AudioCue>>();
            let mut q = self.sim.world.resource_mut::<Queue<AudioCue>>();
            for c in cues {
                q.push(c);
            }
        }
    }

    /// Stage one shell key event (called from the root `on_key_event`
    /// handler; see the module docs for the mapping table).
    fn handle_key(&mut self, ke: &KeyEvent) {
        let down = matches!(ke.event_type, KeyEventType::Down);
        // Shift has no `Key` variant, but every `KeyEvent` carries
        // `modifiers.shift`: synthesize the `ShiftLeft` codes bevy
        // `sample_input` reads (ability/spec) from it. Release clears
        // unless the right mouse button holds the shared code.
        if ke.modifiers.shift {
            self.shift_held = true;
            if down && !ke.is_repeat && !self.held.contains(&KeyCode::ShiftLeft) {
                self.held.insert(KeyCode::ShiftLeft);
                self.edges.push(KeyCode::ShiftLeft);
            }
        } else {
            self.shift_held = false;
            if !self.rmb_held {
                self.held.remove(&KeyCode::ShiftLeft);
            }
        }
        match &ke.key {
            Key::Escape => {
                if down && !ke.is_repeat {
                    self.pause_edge = true;
                }
            }
            Key::Enter => {
                if down && !ke.is_repeat {
                    self.interact_edge = true;
                }
            }
            Key::Space => self.stage_code(KeyCode::Space, down, ke.is_repeat),
            Key::Tab => self.stage_code(KeyCode::Tab, down, ke.is_repeat),
            Key::ArrowUp => self.stage_code(KeyCode::ArrowUp, down, ke.is_repeat),
            Key::ArrowDown => self.stage_code(KeyCode::ArrowDown, down, ke.is_repeat),
            Key::ArrowLeft => self.stage_code(KeyCode::ArrowLeft, down, ke.is_repeat),
            Key::ArrowRight => self.stage_code(KeyCode::ArrowRight, down, ke.is_repeat),
            Key::Character(c) => {
                let c = c.to_ascii_lowercase();
                if c == ' ' {
                    self.stage_code(KeyCode::Space, down, ke.is_repeat);
                } else if c == 'r' {
                    if down && !ke.is_repeat {
                        self.restart_edge = true;
                    }
                } else if let Some(code) = keycode_for_char(c) {
                    self.stage_code(code, down, ke.is_repeat);
                }
            }
            _ => {}
        }
    }

    fn stage_code(&mut self, code: KeyCode, down: bool, is_repeat: bool) {
        if down {
            self.held.insert(code);
            if !is_repeat {
                self.edges.push(code);
            }
        } else {
            self.held.remove(&code);
        }
    }

    /// Right mouse button down (root pointer handler, `Secondary` only):
    /// GML `spec` on `mb_right` shares the synthesized [`KeyCode::ShiftLeft`]
    /// channel so `sample_keyboard` raises spec/ability through the bevy
    /// path. The viewport also stages a buttonless pick for the press;
    /// [`App::feed_input`] drops it via [`App::rmb_down_edge`].
    fn rmb_down(&mut self) {
        if !self.held.contains(&KeyCode::ShiftLeft) {
            self.held.insert(KeyCode::ShiftLeft);
            self.edges.push(KeyCode::ShiftLeft);
        }
        self.rmb_down_edge = true;
        self.rmb_held = true;
    }

    /// Right mouse button up: release the shared code unless the
    /// physical Shift key is still down.
    fn rmb_up(&mut self) {
        self.rmb_held = false;
        self.rmb_up_edge = true;
        if !self.shift_held {
            self.held.remove(&KeyCode::ShiftLeft);
        }
    }

    /// Left mouse button down (root pointer handler, `Primary` only):
    /// latches [`App::lmb_held`] so automatic weapons keep firing while
    /// held. The viewport `Press` still stages the aim/fire click edge.
    fn lmb_down(&mut self) {
        self.lmb_held = true;
    }

    /// Left mouse button up: release the held latch.
    fn lmb_up(&mut self) {
        self.lmb_held = false;
    }

    /// Stage the cursor's window-physical px position (root
    /// `on_pointer_move`, y-down). Stored raw — [`App::cursor_to_world`]
    /// unprojects it through the live camera each frame.
    fn cursor_move(&mut self, phys_px: Vec2) {
        self.cursor_px = Some(phys_px);
    }

    /// Live cursor in world coords: the staged window-physical px point
    /// unprojected through this frame's camera (`Camera2d::dp_to_world_pt`
    /// over the dp viewport extent — bevy `player_aim`
    /// `viewport_to_world_2d` parity). `None` until the first pointer
    /// move; callers fall back to the last viewport `Hover` world point
    /// (touch/pen never stage cursor moves).
    fn cursor_to_world(&self) -> Option<Vec2> {
        let px = self.cursor_px?;
        let d = self.view_density.max(1e-6);
        let dp = [px.x / d, px.y / d];
        let extent = camera_fit_extent(self.view_viewport_dp, self.view_density);
        // `world_size` here is the dp viewport extent; `dp_to_world_pt`
        // divides the dp point by the same fit the viewport paints with.
        Some(self.cam.dp_to_world_pt(self.view_viewport_dp, extent, dp))
    }

    /// Stage one gamepad snapshot for this tick (shells map
    /// `repame-shell` `GamepadEvent`s / platform pad state onto
    /// [`GamepadState`]; drained by [`App::feed_input`] in stage order).
    pub fn stage_gamepad(&mut self, pad: GamepadState) {
        self.pads.push(pad);
    }

    /// Stage one menu button action (pause/settings/credits `on_click`
    /// handlers call this; drained through [`apply_menu_action`] at the
    /// top of [`App::feed_input`], before the sim advances).
    pub fn stage_menu_action(&mut self, action: UiAction) {
        self.menu_actions.push(action);
    }

    /// Stage one viewport click (world aim position + screen px for
    /// menu hit-testing; drained by [`App::feed_input`]).
    fn stage_click(&mut self, world: Vec2, screen: [f32; 2]) {
        let d = repose_core::locals::effective_density_scale().max(1e-6);
        self.clicks.push(StagedClick {
            world,
            dp: [screen[0] / d, screen[1] / d],
        });
    }

    /// Touch/pen contact began (shell maps pointer-down screen-px
    /// y-down coordinates; mouse never lands here — clicks/hover
    /// cover it).
    pub fn touch_down(&mut self, id: u64, screen: Vec2) {
        self.touch_active.insert(id, (screen, screen));
        self.touch_new.insert(id);
    }

    /// Touch/pen contact moved (ignored unless the contact began with
    /// [`App::touch_down`], so pen hovers never become sticks).
    pub fn touch_move(&mut self, id: u64, screen: Vec2) {
        if let Some(contact) = self.touch_active.get_mut(&id) {
            contact.1 = screen;
        }
    }

    /// Touch/pen contact ended (up/cancel/leave).
    pub fn touch_up(&mut self, id: u64) {
        self.touch_active.remove(&id);
        self.touch_new.remove(&id);
    }

    /// Drain staged shell input into sim resources (runs before
    /// [`App::advance`] each frame; pulses are take-once downstream).
    fn feed_input(&mut self) {
        // Staged menu button actions (pause/settings/credits `on_click`
        // handlers): apply headlessly before the sim advances, so a
        // click lands the same tick as a keyboard edge would.
        for action in self.menu_actions.drain(..) {
            apply_menu_action(&mut self.sim.world, action);
        }

        let just: HashSet<KeyCode> = self.edges.drain(..).collect();
        let state = self
            .sim
            .world
            .get_resource::<AppState>()
            .copied()
            .unwrap_or_default();
        let paused = self
            .sim
            .world
            .get_resource::<crate::state::Paused>()
            .is_some_and(|p| p.0);
        let overlay = self
            .sim
            .world
            .get_resource::<OverlayMenu>()
            .copied()
            .unwrap_or_default();
        // Bevy gates every InGame menu path on `run.game_over` only
        // (`handle_pause_input`, `handle_mutation_keys`,
        // `handle_death_restart`); the `MenuState.game_over` snapshot is
        // display data for the render layer, never an input gate.
        let game_over = self
            .sim
            .world
            .get_resource::<crate::comps_a::Run>()
            .is_some_and(|r| r.game_over);
        // Read the pending offer directly, not the `MenuState` mirror
        // (the mirror updates inside the schedule, a frame after the
        // offer opens/closes — the stale frame fired the gun on level-up
        // and let one aim frame through).
        let offer_open = state == AppState::InGame
            && (self
                .sim
                .world
                .get_resource::<crate::comps_a::PendingMutation>()
                .is_some()
                || self
                    .sim
                    .world
                    .get_resource::<crate::comps_a::PendingUltra>()
                    .is_some());
        // Live gameplay: no pause, no overlay, no game over. Only here
        // do hover/clicks steer aim and fire; over menus the buttons
        // stage `UiAction`s instead (a bare viewport click does nothing
        // so it can never resume through a MENU/RETRY press).
        let live_play = state == AppState::InGame
            && !offer_open
            && !paused
            && !game_over
            && overlay == OverlayMenu::None;
        let menu_open =
            state == AppState::InGame && (paused || overlay != OverlayMenu::None) && !game_over;

        // Keyboard cursor nav where the sim protocol speaks slots/cycle:
        // title pods (Left/Right) and mutation highlight (Left/Right).
        // `cycle_weapon` pulses are taken by `tick_menus` the same step.
        if state == AppState::Title || offer_open {
            let mut cycle: i8 = 0;
            if just.contains(&KeyCode::ArrowLeft) {
                cycle -= 1;
            }
            if just.contains(&KeyCode::ArrowRight) {
                cycle += 1;
            }
            if cycle != 0 {
                self.sim.world.resource_mut::<NtInput>().cycle_weapon(cycle);
            }
        }
        // Menu cursor nav (take-once steps for `tick_menus`): main-menu
        // rows (Up/Down) and settings rows/values (Up/Down/Left/Right).
        // Gameplay is gated off in both places, so the arrows are free.
        {
            let settings_open = overlay == OverlayMenu::Settings;
            let mut dv: i8 = 0;
            let mut dh: i8 = 0;
            if state == AppState::MainMenu || settings_open {
                if just.contains(&KeyCode::ArrowUp) {
                    dv -= 1;
                }
                if just.contains(&KeyCode::ArrowDown) {
                    dv += 1;
                }
            }
            if settings_open {
                if just.contains(&KeyCode::ArrowLeft) {
                    dh -= 1;
                }
                if just.contains(&KeyCode::ArrowRight) {
                    dh += 1;
                }
            }
            if dv != 0 || dh != 0 {
                self.sim
                    .world
                    .resource_mut::<NtInput>()
                    .push_menu_nav(dv, dh);
            }
        }
        // Right-button press: no viewport pick is staged for it (RMB is
        // handled via `rmb_down_edge` only), so there is nothing to drop
        // here. Each arm below decides what RMB means (Back over
        // settings/credits, silent elsewhere) without eating a coincident
        // left click.
        let rmb_down = std::mem::replace(&mut self.rmb_down_edge, false);
        // Release stages no pick; drain the flag so it never leaks into
        // a later window.
        let _ = std::mem::replace(&mut self.rmb_up_edge, false);

        // No fire pulses from clicks over the main menu / title: both
        // route positionally now, and Title takes `fire` (Space) as the
        // loadout toggle — a pod click must not toggle it as a side
        // effect. Splash/Loading advance via the click→interact arm
        // below, so they stage no fire pulse either (one pulse per
        // click, not two).
        // `clicks` are single-frame edges (staged on Press, drained at the
        // end of this fn); `lmb_held` latches primary down/up at the root
        // so automatic weapons keep firing while the button is held.
        let mouse_down_edge = !self.clicks.is_empty();
        let mouse_down = (mouse_down_edge || self.lmb_held)
            && !menu_open
            && !game_over
            && !offer_open
            && state != AppState::MainMenu
            && state != AppState::Title
            && state != AppState::Splash
            && state != AppState::Loading;
        let mouse = MouseState {
            left_held: mouse_down,
            left_pressed: mouse_down_edge
                && !menu_open
                && !game_over
                && !offer_open
                && state != AppState::MainMenu
                && state != AppState::Title
                && state != AppState::Splash
                && state != AppState::Loading,
            ..MouseState::default()
        };
        {
            let mut input = self.sim.world.resource_mut::<NtInput>();
            sample_keyboard(&self.held, &just, &mouse, &mut input);
            // Bevy `sample_input` layering verbatim: gamepads in query
            // order, then touch. Sticks overwrite nonzero axes; pulses
            // OR-accumulate; slots replace; cycle saturating-adds.
            let pads = self.pads.drain(..).collect::<Vec<_>>();
            sample_gamepads(&pads, &mut input);
            // Touch contacts synthesize fresh every tick from the live
            // map (bevy `touches.iter()` yields all pressed contacts;
            // `touch_new` marks this tick's `iter_just_pressed`).
            if !self.touch_active.is_empty() {
                // `PickEvent` screens and `sched.size` are physical px;
                // `sample_touch` zones are authored in dp (96dp button
                // strip, 56dp stick scale). Convert both to dp so HiDPI
                // windows keep dp-sized zones instead of shrinking them
                // by the density.
                let d = repose_core::locals::effective_density_scale().max(1e-6);
                let contacts: Vec<TouchContact> = self
                    .touch_active
                    .iter()
                    .map(|(id, (start, pos))| TouchContact {
                        start: *start / d,
                        pos: *pos / d,
                        just_pressed: self.touch_new.contains(id),
                    })
                    .collect();
                self.touch_new.clear();
                sample_touch(&contacts, self.view_width / d, &mut input);
            }
        }
        // Right-click shares the `ShiftLeft` spec channel (GML `spec`
        // on `mb_right`); on Title that channel toggles loadout/hardmode,
        // so a right-click would toggle panels as a side effect. Drop
        // the pulse — Back only travels via Esc here.
        if rmb_down && state == AppState::Title {
            self.sim.world.resource_mut::<NtInput>().take_spec_pressed();
        }
        // Mutation digits travel the bevy two-step (`weapon_slot` pulse →
        // `route_mutation_digit` → Select/Pick). No direct `MutationChoice`
        // write here: that committed on the first press, bypassing the
        // highlight law (`handle_mutation_keys`).
        // Shell edges with no `KeyCode` (Esc / R / Enter).
        let had_pause = self.pause_edge;
        let had_restart = self.restart_edge;
        let had_interact = self.interact_edge;
        if self.pause_edge || self.restart_edge {
            let mut edge = self.sim.world.resource_mut::<MenuEdge>();
            edge.pause_pressed |= self.pause_edge;
            edge.restart_pressed |= self.restart_edge;
        }
        self.pause_edge = false;
        self.restart_edge = false;
        if self.interact_edge {
            self.sim.world.resource_mut::<NtInput>().press_interact();
            self.interact_edge = false;
        }
        // Splash advances on any key/mouse edge (bevy `boot_intro` law):
        // arrows/WASD/digits never stage fire/interact/spec pulses, so
        // without this a keyboard-only shell stalls on the logo until the
        // timeout. Any `just` key, pad edge, or staged click counts.
        if state == AppState::Splash
            && (!just.is_empty()
                || had_pause
                || had_restart
                || had_interact
                || !self.clicks.is_empty())
        {
            self.sim.world.resource_mut::<NtInput>().press_interact();
        }

        // Per-frame cursor aim (bevy `player_aim` mouse path): the
        // cursor's *screen* position unprojected through the current
        // camera steers `aim_axis` every tick, not just on hover events,
        // so `AimDir` tracks the on-screen cursor continuously while the
        // player walks (bevy reads `window.cursor_position()` live each
        // frame; a latched world point would go stale as the camera
        // moves). The follow camera reads the same live point's distance
        // directly as GML `dis_fire`).
        // Stick input wins when nonzero (bevy precedence:
        // `sample_keyboard` leaves `aim_axis` zero, a gamepad shell may
        // layer on top). Frozen outside live play (pause/menus/game
        // over keep the last aim; GML `KeyCont` is dead there too).
        // Clicks: aim-at-click + fire in live game, confirm in menus. A
        // click is a single-frame edge (continuous hold needs shell
        // pressed-state). Clicks over an open pause/settings/credits
        // menu hit-test the screen-anchored buttons by dp position
        // (repose-hit-test independent); a stray click stages nothing,
        // so it can never fire the gun or resume through a MENU press.
        if live_play {
            let player_pos = self
                .sim
                .world
                .query::<(&Pos, &Player)>()
                .iter(&self.sim.world)
                .next()
                .map(|(p, _)| p.0);
            if let Some(pp) = player_pos {
                // Live cursor, unprojected through this frame's camera
                // (see `cursor_px`): valid even when the pointer hasn't
                // moved since the camera did.
                let aim_hover = self.cursor_to_world().or(self.hover);
                if let Some(hover) = aim_hover {
                    let mut input = self.sim.world.resource_mut::<NtInput>();
                    if input.aim_axis == Vec2::ZERO {
                        let dir = hover - pp;
                        if dir.length_squared() > 1e-6 {
                            input.aim_axis = dir.normalize_or_zero();
                        }
                    }
                }
                if let Some(click) = self.clicks.last().copied() {
                    let mut input = self.sim.world.resource_mut::<NtInput>();
                    let dir = click.world - pp;
                    if dir.length_squared() > 1e-6 {
                        input.aim_axis = dir.normalize_or_zero();
                    }
                    input.fire_held = true;
                    input.press_fire();
                }
            }
            self.clicks.clear();
        } else if menu_open {
            // Open menu over a live run: right-click steps back (GML
            // `BackButton` `mb_right` parity — Settings pops one level
            // via `SettingsBack`, Credits closes), left clicks route at
            // the menu buttons, then drop either way. A coincident left
            // click still routes (RMB never eats it).
            if rmb_down && overlay == OverlayMenu::Settings {
                apply_menu_action(&mut self.sim.world, UiAction::SettingsBack);
            } else if rmb_down && overlay == OverlayMenu::Credits {
                apply_menu_action(&mut self.sim.world, UiAction::CloseOverlay);
            } else if let Some(click) = self.clicks.last().copied() {
                let viewport_dp = self.view_viewport_dp;
                let kind = menu_overlay_kind(
                    state,
                    overlay,
                    &self.sim.world.resource::<MenuState>(),
                    game_over,
                );
                if let Some(kind) = kind {
                    if let Some(action) =
                        route_menu_click(&mut self.sim.world, kind, click.dp, viewport_dp)
                    {
                        apply_menu_action(&mut self.sim.world, action);
                    }
                }
            }
            self.clicks.clear();
        } else if game_over {
            if let Some(click) = self.clicks.last().copied() {
                let viewport_dp = self.view_viewport_dp;
                if let Some(action) = route_menu_click(
                    &mut self.sim.world,
                    MenuOverlay::GameOver,
                    click.dp,
                    viewport_dp,
                ) {
                    apply_menu_action(&mut self.sim.world, action);
                }
            }
            self.clicks.clear();
        } else if state == AppState::MainMenu {
            // Settings/Credits/Stats open over the buttons (GML MenuOptions
            // / DrawStats parity): route through the live overlay kind so
            // mouse works on those pages, not just the keyboard. RMB steps
            // back (Settings pops a level, others close).
            if rmb_down && overlay == OverlayMenu::Settings {
                apply_menu_action(&mut self.sim.world, UiAction::SettingsBack);
            } else if rmb_down && matches!(overlay, OverlayMenu::Credits | OverlayMenu::Stats) {
                apply_menu_action(&mut self.sim.world, UiAction::CloseOverlay);
            } else if let Some(click) = self.clicks.last().copied() {
                let viewport_dp = self.view_viewport_dp;
                let kind = menu_overlay_kind(
                    state,
                    overlay,
                    &self.sim.world.resource::<MenuState>(),
                    game_over,
                )
                .unwrap_or(MenuOverlay::MainMenu);
                if let Some(action) =
                    route_menu_click(&mut self.sim.world, kind, click.dp, viewport_dp)
                {
                    apply_menu_action(&mut self.sim.world, action);
                }
            }
            self.clicks.clear();
        } else if state == AppState::Title {
            // Settings/Credits open over the campfire: route those through
            // the menu router (mouse + RMB-back); otherwise pods / GO /
            // loadout zones (GML campfire parity, strays do nothing).
            if rmb_down && overlay == OverlayMenu::Settings {
                apply_menu_action(&mut self.sim.world, UiAction::SettingsBack);
            } else if rmb_down && overlay == OverlayMenu::Credits {
                apply_menu_action(&mut self.sim.world, UiAction::CloseOverlay);
            } else if let Some(click) = self.clicks.last().copied() {
                let viewport_dp = self.view_viewport_dp;
                if overlay == OverlayMenu::Settings || overlay == OverlayMenu::Credits {
                    let kind = menu_overlay_kind(
                        state,
                        overlay,
                        &self.sim.world.resource::<MenuState>(),
                        game_over,
                    );
                    if let Some(kind) = kind
                        && let Some(action) =
                            route_menu_click(&mut self.sim.world, kind, click.dp, viewport_dp)
                    {
                        apply_menu_action(&mut self.sim.world, action);
                    }
                } else if let Some(action) = self.route_title_click(click.dp, viewport_dp) {
                    apply_menu_action(&mut self.sim.world, action);
                }
            }
            self.clicks.clear();
        } else if offer_open {
            // Mutation/ultra offer: right-button is silent (never confirms
            // or eats the left click).
            if let Some(click) = self.clicks.last().copied() {
                let viewport_dp = self.view_viewport_dp;
                let vw = crate::render::gml_view_size(viewport_dp)[0];
                let k = (viewport_dp[1].max(1.0) / 240.0).max(1e-6);
                if k.is_finite()
                    && let Some(action) = crate::render::mutation_icon_hit_action(
                        &mut self.sim.world,
                        click.dp[0] / k,
                        click.dp[1] / k,
                        vw,
                    )
                {
                    apply_menu_action(&mut self.sim.world, action);
                }
            }
            self.clicks.clear();
        } else if let Some(_click) = self.clicks.last().copied() {
            // Splash/Loading advance on any mouse button (bevy `boot_intro`
            // any-key/mouse law).
            self.clicks.clear();
            self.sim.world.resource_mut::<NtInput>().press_interact();
        } else {
            self.clicks.clear();
        }
    }

    /// Title click router with asset-aware geometry (same native sizes
    /// the sprite layer draws with, else the 20px fallback).
    fn route_title_click(&mut self, dp: [f32; 2], viewport_dp: [f32; 2]) -> Option<UiAction> {
        let vw = gml_view_size(viewport_dp)[0];
        let k = (viewport_dp[1].max(1.0) / 240.0).max(1e-6);
        if !k.is_finite() {
            return None;
        }
        let (slot_h, crownsize, skinsize) = match &self.assets {
            Some(a) => (
                a.native_size("images/sprCharSelect.png")
                    .map(|s| s.y)
                    .unwrap_or(20.0),
                a.native_size("images/sprLoadoutCrown.png")
                    .map(|s| s.y - 4.0)
                    .unwrap_or(20.0),
                a.native_size("images/sprLoadoutSkin.png")
                    .map(|s| s.x - 4.0)
                    .unwrap_or(20.0),
            ),
            None => (20.0, 20.0, 20.0),
        };
        crate::render::title_click_action(
            &mut self.sim.world,
            dp[0] / k,
            dp[1] / k,
            vw,
            slot_h,
            crownsize,
            skinsize,
        )
    }

    /// Build this frame's view: stage input, advance the sim, snapshot
    /// sim+render+vortex+HUD+menus into repose views.
    pub fn view(&mut self, sched: &mut Scheduler, _ctx: &RenderContext, dt: Duration) -> View {
        request_frame();
        // Touch button zones read the viewport width (bevy
        // `window.width()`); shells must stage contacts in the same px
        // space as `sched.size`.
        if sched.size.0 > 0 {
            self.view_width = sched.size.0 as f32;
        }
        self.feed_input();
        self.advance(dt);
        // GML `game_end` parity for the QUIT row (bevy `AppExit` has no
        // headless window service; the desktop shell exits here).
        if self
            .sim
            .world
            .get_resource::<crate::state::QuitRequested>()
            .is_some_and(|q| q.0)
        {
            std::process::exit(0);
        }

        let viewport_px = {
            let (w, h) = sched.size;
            if w == 0 || h == 0 {
                [1280.0, 720.0]
            } else {
                [w as f32, h as f32]
            }
        };
        // `Scheduler.size` is physical px; the framing contract (viewport
        // paint, picks, HUD dp law) works in dp.
        let density = repose_core::locals::effective_density_scale();
        let viewport_dp = [
            viewport_px[0] / density.max(1e-6),
            viewport_px[1] / density.max(1e-6),
        ];
        let world_size = camera_fit_extent(viewport_px, density);
        // GML `scrSetViewSize` verbatim: the framed view is always 240
        // world px tall (`gml_view_size`), so the camera scale is derived
        // per window, not fixed (`gml_view_scale`: 1280x720 → 1/3).
        let gml_scale = gml_view_scale(viewport_dp);
        let gml_view = gml_view_size(viewport_dp);
        self.view_world_size = gml_view;
        self.view_viewport_dp = viewport_dp;
        self.view_density = density.max(1e-6);
        // GML `SpiralCont/Step_0` centers the emitter on `view_width
        // div 2`: refresh the live GUI width so the vortex (and its
        // snapshot view rect) tracks wide windows 1:1.
        self.spiral.view_w = gml_view[0];

        let area = self
            .sim
            .world
            .get_resource::<crate::comps_a::Run>()
            .map(|r| r.area)
            .unwrap_or(AreaId::Desert);
        let state = self
            .sim
            .world
            .get_resource::<AppState>()
            .copied()
            .unwrap_or_default();
        let overlay = self
            .sim
            .world
            .get_resource::<OverlayMenu>()
            .copied()
            .unwrap_or_default();
        let run_game_over = self
            .sim
            .world
            .get_resource::<crate::comps_a::Run>()
            .is_some_and(|r| r.game_over);
        let menu_kind = {
            let menu = self.sim.world.resource_mut::<MenuState>();
            menu_overlay_kind(state, overlay, &menu, run_game_over)
        };
        // GML `Vlambeer/Alarm_0`: the `SpiralCont` exists only once the
        // reel reaches the `Logo` phase (`mode >= 3` fires the alarm
        // that creates cont + `Logo` and destroys the `Vlambeer` card;
        // the port's mode 4 is that logo phase). Modes 0-3 are black +
        // text with no spiral caller.
        let logo_live = menu_kind != Some(MenuOverlay::Splash)
            || self
                .sim
                .world
                .get_resource::<crate::state::SplashState>()
                .is_some_and(|s| s.mode >= 4);
        let paused = self
            .sim
            .world
            .get_resource::<crate::state::Paused>()
            .is_some_and(|p| p.0);
        let game_over = self
            .sim
            .world
            .get_resource::<crate::comps_a::Run>()
            .is_some_and(|r| r.game_over);

        // Follow camera: the GML `BackCont` look point steps at
        // 30 Hz in [`App::step_camera_fixed`] (frozen over
        // pause/menus/game over, so menus keep the last camera).
        // GML has no zoom: the view is always 240 world px tall
        // (`gml_view_size`, `gml_view_scale` units per px), snapped on
        // room start by `gml_cam.snap`. `gml_cam` tracks the top-left
        // corner; the GPU camera centers the look point.
        let center = Vec2::new(
            self.gml_cam.x + gml_view[0] * 0.5,
            self.gml_cam.y + gml_view[1] * 0.5,
        );
        self.cam = world_camera(center, gml_scale);
        self.cam.offset = Vec2::ZERO;

        // Sprite layer: real instances with assets, placeholder quads
        // without. The batch is the headless-verifiable artifact (push +
        // camera wired here); `Viewport2dGpu` rebuilds its own internally.
        let (mut sprites, texts) = if self.assets.is_some() {
            let assets = self.assets.as_ref().expect("checked");
            // Cursor world position for the GML crosshair distance
            // (`KeyCont.dis_fire` parity; `None` until the first hover).
            // Uses the live cursor unprojection when available so the
            // crosshair tracks the on-screen cursor as the camera moves
            // (same source as aim; falls back to the last Hover).
            self.sim.world.insert_resource(crate::render::HoverWorld(
                self.cursor_to_world().or(self.hover),
            ));
            // Blob shadows first (GML `shad` surface: under the actors).
            // Every layer stamps its z-ladder rung (render.rs `Z_*`,
            // GML `__global_object_depths` order): without rungs every
            // sprite shares z=0 and the engine's `(blend, z, page)` sort
            // lets atlas page lottery HUD bars under floor tiles. Push
            // order matches rung order, so the canvas path (push-ordered)
            // and the GPU path (z-sorted) agree.
            // GML `scrGameIsGenerationScreen` verbatim: while the
            // generation conts own the draw (`GenCont` behind Loading,
            // `LevCont` behind the mutation/ultra offer) the room draws
            // NOTHING but the spiral + cover text — the old floor sits
            // under an opaque vortex backdrop and would paint straight
            // through the fullscreen pass (the "tiles over the vortex"
            // bug). The campfire title is NOT a generation screen for
            // draw purposes: `MenuGen` builds the camp, then `Menu`
            // draws spiral remnant + camp + pods + portraits (`Menu`
            // draw scripts own that chrome, not `TopCont` — so the
            // TopCont-sourced HUD bars stay off but the Menu chrome
            // stays on).
            let generation_screen = matches!(
                menu_kind,
                Some(MenuOverlay::Loading) | Some(MenuOverlay::Mutation)
            ) || self
                .sim
                .world
                .get_resource::<crate::comps_b::FloorTransition>()
                .is_some_and(|f| f.active);
            let playing = !generation_screen;
            let mut s = if playing {
                shadow_sprites(&mut self.sim.world, assets)
            } else {
                Vec::new()
            };
            stamp_z(&mut s, Z_SHADOW);
            let mut w = if playing {
                world_instances(&mut self.sim.world, assets)
            } else {
                Vec::new()
            };
            stamp_z(&mut w, crate::render::Z_WORLD);
            s.extend(w);
            let mut f = if playing {
                fx_instances(&mut self.sim.world, assets)
            } else {
                Vec::new()
            };
            stamp_z(&mut f, Z_FX);
            s.extend(f);
            // Additive bloom over the world (GML `scrDrawBloom`, gated
            // on `opt_bloom` inside).
            let mut b = if playing {
                bloom_sprites(&mut self.sim.world, assets)
            } else {
                Vec::new()
            };
            stamp_z(&mut b, Z_BLOOM);
            s.extend(b);
            // Area fog over the room (GML TopCont/Draw_0, sewers only).
            let mut fog = if playing {
                fog_sprites(
                    &mut self.sim.world,
                    assets,
                    viewport_dp,
                    world_size,
                    &self.cam,
                )
            } else {
                Vec::new()
            };
            stamp_z(&mut fog, Z_FOG);
            s.extend(fog);
            let view = view_rect_world(viewport_dp, world_size, &self.cam);
            // Ghost decay runs on the render dt (GML steps it per sim
            // tick at 30 Hz).
            let hud_dt = dt.as_secs_f32().clamp(0.0, 0.1);
            // World crosshair (GML `TopCont/Draw_0`, over the room,
            // under the HUD text) plus coop fainted bars at the
            // view-clamped positions. All three world-HUD passes
            // (`with Player` crosshair, `Revive` bars, `Portal`
            // arrow) require live run actors; the Loading room is
            // empty by construction (`goto_state` teardown + GML
            // `room_restart`), so they self-suppress there exactly
            // like GML (and stay off on Title/menus via their own
            // live-run gates).
            let mut cross = if playing {
                crosshair_sprites(&mut self.sim.world, assets, hud_dt)
            } else {
                Vec::new()
            };
            stamp_z(&mut cross, Z_CROSSHAIR);
            s.extend(cross);
            let mut faint = if playing {
                fainted_bar_sprites(&mut self.sim.world, assets, view)
            } else {
                Vec::new()
            };
            stamp_z(&mut faint, Z_FAINTED);
            s.extend(faint);
            // Offscreen portal arrow (GML `TopCont/Draw_0` tail).
            let mut portal = if playing {
                portal_indicator_sprites(
                    &mut self.sim.world,
                    assets,
                    viewport_dp,
                    world_size,
                    &self.cam,
                )
            } else {
                Vec::new()
            };
            stamp_z(&mut portal, Z_PORTAL_INDICATOR);
            s.extend(portal);
            // Spiral CPU layer (GML `scrDrawSpiral` center figures):
            // crown orbit + player hurt figures ride every spiral
            // caller — `GenCont`, `LevCont`, `NothingSpiral` and the
            // logo spirals alike. NOT the campfire `Menu`
            // (`PlayButton` destroys the `SpiralCont` on entry, so the
            // `with SpiralCont` figure block has no instance) and NOT
            // GameOver (the dead run's cont died at generation end).
            // Gated on the mounted vortex layer so figures never float
            // over the flat campfire camp or the game-over dim.
            // NOTE: `center` is the vortex look point (GUI view
            // center), not the world camera — GML draws figures at
            // `view + cont.x/y` (view-local coords), independent of
            // the room camera.
            let vortex_mounted_later = self.assets.is_some()
                && !self.vortex_tex.is_empty()
                && !matches!(menu_kind, Some(MenuOverlay::Title))
                && logo_live
                && (self.spiral.alive || !self.spiral.is_done());
            if vortex_mounted_later {
                let gui_view = gml_view_size(viewport_dp);
                let figs_center =
                    Vec2::new(view[0] + gui_view[0] * 0.5, view[1] + gui_view[1] * 0.5);
                let mut figs =
                    spiral_figures(&mut self.sim.world, assets, figs_center, self.spiral.angle);
                stamp_z(&mut figs, Z_SPIRAL_FIGURES);
                s.extend(figs);
            }
            // View-anchored HUD bars (Draw-GUI-64: above all world-space
            // layers). GML `GenCont/Draw_0` draws ONLY spiral +
            // GENERATING + roadmap — no PlayerHUD/MiscHUD — and the
            // Loading room's run is pre-`setup_run` (no live run actors
            // yet), so the HUD stays off while Loading like GML (TopCont
            // draws the HUD only once the run's actors exist).
            let hud_view = view_rect_world(viewport_dp, world_size, &self.cam);
            let loading_cover = state == AppState::Loading
                || self
                    .sim
                    .world
                    .get_resource::<crate::comps_b::FloorTransition>()
                    .is_some_and(|f| f.active)
                || self
                    .sim
                    .world
                    .get_resource::<crate::comps_a::PendingMutation>()
                    .is_some()
                || self
                    .sim
                    .world
                    .get_resource::<crate::comps_a::PendingUltra>()
                    .is_some();
            let mut h = if loading_cover || generation_screen {
                Vec::new()
            } else {
                hud_sprites(&mut self.sim.world, assets, hud_view, hud_dt)
            };
            stamp_z(&mut h, Z_HUD);
            s.extend(h);
            // Boot reel (`Vlambeer/Draw_0` + `Logo/Draw_0`).
            if menu_kind == Some(MenuOverlay::Splash) {
                let mut splash = splash_sprites(&mut self.sim.world, assets, view);
                stamp_z(&mut splash, Z_SPLASH);
                s.extend(splash);
            }
            // Menu art sprites (char pods, portrait, loadout, splats).
            // `Menu` draws these in `Draw_0`/`Draw_74` on every screen
            // it owns (campfire title AND generation covers — the offer
            // row is `LevCont` chrome but the pods/portraits persist
            // underneath in GML, just buried under the opaque spiral).
            // They ride the sprite viewport UNDER the opaque vortex
            // pass, so covers stay clean while the title keeps its
            // camp chrome.
            if let Some(kind) = menu_kind {
                let mut menu = menu_sprites(
                    kind,
                    &mut self.sim.world,
                    assets,
                    viewport_dp,
                    world_size,
                    &self.cam,
                );
                stamp_z(&mut menu, Z_MENU);
                s.extend(menu);
            }
            // Sideart chrome around the view (GML `UberCont/Draw_74`:
            // over everything, game and menus alike — but never over a
            // generation screen, whose draw scripts own the full frame).
            if playing {
                let mut side = sideart_sprites(
                    &mut self.sim.world,
                    assets,
                    viewport_dp,
                    world_size,
                    &self.cam,
                );
                stamp_z(&mut side, Z_SIDEART);
                s.extend(side);
            }
            let texts = if playing {
                fx_texts(&mut self.sim.world)
            } else {
                Vec::new()
            };
            (s, texts)
        } else {
            (placeholder_instances(&mut self.sim.world), Vec::new())
        };
        // Cover flag for the chrome below: same generation-screen law
        // as the sprite composer above (its `playing` is block-local).
        // Canvas text rides above the opaque vortex pass, so the HUD
        // rows need the same gate or HP/level/ammo/FLOOR paints over
        // the spiral mid-transition.
        let cover_chrome_off = matches!(
            menu_kind,
            Some(MenuOverlay::Loading) | Some(MenuOverlay::Mutation)
        ) || self
            .sim
            .world
            .get_resource::<crate::comps_b::FloorTransition>()
            .is_some_and(|f| f.active);
        let _ = &mut sprites;
        let mut batch = SpriteBatch::new(
            self.assets
                .as_ref()
                .map(|a| a.batch_desc())
                .unwrap_or_default(),
        );
        batch.set_camera(self.cam.fit_matrix(viewport_dp, world_size));
        for s in &sprites {
            batch.push_sprite(s);
        }
        self.last_sprite_count = batch.len();

        // Vortex snapshot -> mounted background pass. `bg_alpha` is
        // GML `scrDrawSpiral` verbatim: `draw_clear(c_black)` runs in
        // every caller EXCEPT `Menu` — i.e. opaque black behind the
        // spiral on Logo/Splash/MainMenu/Loading/covers, transparent
        // over the campfire camp on Title. InGame mounts the pass only
        // while a floor transition or mutation/ultra cover runs (the
        // only spiral callers in a run).
        let ft_active = self
            .sim
            .world
            .get_resource::<crate::comps_b::FloorTransition>()
            .is_some_and(|f| f.active);
        let pending_pick = self
            .sim
            .world
            .get_resource::<crate::comps_a::PendingMutation>()
            .is_some()
            || self
                .sim
                .world
                .get_resource::<crate::comps_a::PendingUltra>()
                .is_some();
        let bg_alpha = match state {
            AppState::Title => 0.0,
            AppState::Splash | AppState::MainMenu | AppState::Loading => 1.0,
            AppState::InGame => {
                if ft_active || pending_pick {
                    1.0
                } else {
                    0.0
                }
            }
        };
        let snap = self.spiral.snapshot(bg_alpha);
        self.last_bg_alpha = snap.bg_alpha;
        // Vortex art follows the GML area (debris strip is per-area);
        // decode once per area, not per frame.
        let gml_area = gml_area_for_area(area);
        if self.assets_dir.is_some() && self.vortex_tex_area != Some(gml_area) {
            if let Some(dir) = self.assets_dir.clone() {
                self.vortex_tex = Self::load_vortex_textures(&dir, gml_area);
                self.vortex_tex_area = Some(gml_area);
            }
        }
        // Logo/Splash mounts only once the `Logo` exists (gated on
        // `logo_live` above). Title mounts while its entry drain plays
        // out (GML `Menu/Draw_0` draws the leftover motes transparently
        // over the camp; once done the flat camp shows). Every other
        // caller in the list mounts unconditionally (caller list
        // documented at the figure gate above).
        let splash_gated = !logo_live;
        let vortex_layer = if self.assets.is_some()
            && !self.vortex_tex.is_empty()
            && !splash_gated
            && (self.spiral.alive || !self.spiral.is_done())
        {
            let mut pass = VortexPass::new(snap);
            pass.extend_textures(self.vortex_tex.clone());
            Some(Embedded(
                Modifier::new().fill_max_size().hit_passthrough(),
                Callback::new(pass),
            ))
        } else {
            None
        };
        // The area fill only shows where GML paints it: `GenCont/Create_0`
        // `background_set_colour(scrAreaGetBackroundColor(GameCont.area))`
        // runs once per generated floor, and the campfire title inherits
        // the same call via `MenuGen`. But the fill must NEVER sit under
        // a mounted vortex pass as a fullscreen quad: GML's
        // `Menu/Draw_0` calls `scrDrawSpiral()` FIRST (transparent, no
        // `draw_clear`) and draws the camp floors OVER it — the spiral
        // shows through the gaps between floor tiles. A fullscreen fill
        // quad in the sprite viewport would bury the vortex layer
        // underneath (the "no vortex on the title screen" bug). So:
        // mounted vortex pass -> no fill (the pass's own bg_alpha owns
        // the backdrop: opaque black on Loading/covers, transparent on
        // Title); unmounted pass -> the flat room colour stands in.
        // Splash/MainMenu have no area yet (`Vlambeer/Create_0` never
        // sets a colour; `Vlambeer/Draw_0` clears black). Live gameplay
        // past the spiral drain and GameOver (whose spiral died at
        // generation end) also fall back to the flat room colour.
        let background = if vortex_layer.is_some() {
            None
        } else if matches!(
            menu_kind,
            Some(MenuOverlay::Splash) | Some(MenuOverlay::MainMenu)
        ) {
            Some([0.0, 0.0, 0.0, 1.0])
        } else {
            Some(background_color(area))
        };
        // Fullscreen overlays: hit flashes (white). Menu dimming is
        // the scrim `UiBox` above (bevy parity: one 230-black layer
        // over everything, background included), never the viewport
        // tint (that would double-dim the sprites).
        let dim_menu = matches!(
            menu_kind,
            Some(MenuOverlay::Pause)
                | Some(MenuOverlay::Settings)
                | Some(MenuOverlay::Credits)
                | Some(MenuOverlay::GameOver)
                | Some(MenuOverlay::Stats)
        );
        let overlay_color = crate::effects::flash_rgba(&self.sim.world);

        // GML `scrGameIsGenerationScreen` verbatim: no PlayerHUD text
        // while a generation screen owns the frame (`GenCont` behind
        // Loading, `LevCont` behind the offer, mid-run `FloorTransition`
        // covers). The canvas text layer rides ABOVE the opaque vortex
        // pass, so an ungated `hud_rows` paints HP/level/ammo/FLOOR
        // straight over the spiral (the cover-text leak).
        let hud_rows = if state == AppState::InGame && menu_kind.is_none() && !cover_chrome_off {
            hud_overlay_lines(&mut self.sim.world, viewport_dp)
        } else {
            Vec::new()
        };
        let menu_rows = menu_kind.map(|k| menu_gui_texts_dp(k, &mut self.sim.world, viewport_dp));

        // Viewport input snapshot (owned from here on; handlers below only
        // touch staged input through the raw pointer).
        let frame = FrameInput {
            cam: self.cam,
            world_size,
            // Cold-start dp fallback for the GPU prepare path (once the
            // first paint lands, `FrameGeom` carries the real viewport).
            viewport_dp,
            sprites,
            texts,
            background,
            overlay_color,
            chroma: 0.0,
        };

        let app_ptr = self as *mut App;
        let viewport = if self.assets.is_some() {
            let geom = GeomHandle::new();
            let uploads = self
                .assets
                .as_mut()
                .map(|a| a.take_uploads())
                .unwrap_or_default();
            let desc: BatchDesc = self
                .assets
                .as_ref()
                .map(|a| a.batch_desc())
                .unwrap_or_default();
            Viewport2dGpu(frame, geom, uploads, desc, move |ev| {
                // SAFETY: synchronous compose-time dispatch only (same
                // shape as the rozvp pilot runner).
                let app = unsafe { &mut *app_ptr };
                match ev {
                    PickEvent::Press { world, screen } | PickEvent::Click { world, screen } => {
                        app.lmb_down();
                        app.stage_click(world, screen)
                    }
                    PickEvent::Hover { world } => app.hover = Some(world),
                    // Touch contacts (screen px, y-down) feed bevy's
                    // touch zones; taps still land as clicks above.
                    PickEvent::TouchDown { id, screen } => {
                        app.touch_down(id, Vec2::new(screen[0], screen[1]))
                    }
                    PickEvent::TouchMove { id, screen } => {
                        app.touch_move(id, Vec2::new(screen[0], screen[1]))
                    }
                    PickEvent::TouchUp { id } => app.touch_up(id),
                }
            })
        } else {
            let geom = GeomHandle::new();
            Viewport2d(frame, geom, move |ev| {
                // SAFETY: synchronous compose-time dispatch only.
                let app = unsafe { &mut *app_ptr };
                match ev {
                    PickEvent::Press { world, screen } | PickEvent::Click { world, screen } => {
                        app.lmb_down();
                        app.stage_click(world, screen)
                    }
                    PickEvent::Hover { world } => app.hover = Some(world),
                    PickEvent::TouchDown { id, screen } => {
                        app.touch_down(id, Vec2::new(screen[0], screen[1]))
                    }
                    PickEvent::TouchMove { id, screen } => {
                        app.touch_move(id, Vec2::new(screen[0], screen[1]))
                    }
                    PickEvent::TouchUp { id } => app.touch_up(id),
                }
            })
        };

        // Focusable root so hardware keys reach the staging feed (same
        // shape as the rozvp pilot root).
        let app_ptr = self as *mut App;
        let focus = remember(FocusRequester::new);
        let fr_positioned = (*focus).clone();
        let root_mod = Modifier::new()
            .fill_max_size()
            .focusable(true)
            .focus_requester((*focus).clone())
            .on_globally_positioned(move |_| {
                fr_positioned.request_focus();
            })
            .on_key_event(move |ke: KeyEvent| {
                // SAFETY: synchronous compose-time dispatch only.
                let app = unsafe { &mut *app_ptr };
                app.handle_key(&ke);
                false
            });
        // Right mouse button (GML `mb_right` ability / menu Back): the
        // viewport reports buttonless picks, so stage spec here from the
        // raw event (the viewport's spurious pick is dropped in
        // `feed_input` via the rmb edges).
        let rmb_ptr_down = self as *mut App;
        let rmb_ptr_up = self as *mut App;
        let lmb_ptr_down = self as *mut App;
        let lmb_ptr_up = self as *mut App;
        let cursor_ptr = self as *mut App;
        let root_mod = root_mod
            // Live cursor in window-physical px (bevy `player_aim`
            // `window.cursor_position()` parity): fires on every pointer
            // move even when the camera — and therefore the viewport
            // `Hover` world point — hasn't been recomputed, so aim tracks
            // the on-screen cursor while the player walks. Touch/pen
            // moves are skipped (sticks own those; `TouchMove` covers
            // them).
            .on_pointer_move(move |ev: PointerEvent| {
                if matches!(ev.kind, repose_core::input::PointerKind::Mouse) {
                    // SAFETY: synchronous compose-time dispatch only.
                    let app = unsafe { &mut *cursor_ptr };
                    let p = ev.position_in_window();
                    app.cursor_move(Vec2::new(p.x, p.y));
                }
            })
            .on_pointer_down(move |ev: PointerEvent| {
                if matches!(ev.event, PointerEventKind::Down(PointerButton::Secondary)) {
                    // SAFETY: synchronous compose-time dispatch only.
                    let app = unsafe { &mut *rmb_ptr_down };
                    app.rmb_down();
                }
                if matches!(ev.event, PointerEventKind::Down(PointerButton::Primary)) {
                    // SAFETY: synchronous compose-time dispatch only.
                    let app = unsafe { &mut *lmb_ptr_down };
                    app.lmb_down();
                }
            })
            .on_pointer_up(move |ev: PointerEvent| {
                if matches!(ev.event, PointerEventKind::Up(PointerButton::Secondary)) {
                    // SAFETY: synchronous compose-time dispatch only.
                    let app = unsafe { &mut *rmb_ptr_up };
                    app.rmb_up();
                }
                if matches!(ev.event, PointerEventKind::Up(PointerButton::Primary)) {
                    // SAFETY: synchronous compose-time dispatch only.
                    let app = unsafe { &mut *lmb_ptr_up };
                    app.lmb_up();
                }
            });

        // HUD overlay (GML `scrDrawPlayerHUD` + `scrDrawMiscHUD`
        // verbatim): Silkscreen rows at 320x240 GUI positions over the
        // viewport.
        // Bottom-up: vortex portal first, then the sprite viewport,
        // then HUD/menu chrome.
        let mut layers = Vec::new();
        if let Some(vortex) = vortex_layer {
            layers.push(vortex);
        }
        layers.push(viewport);
        if !hud_rows.is_empty() {
            layers.push(
                ZStack(Modifier::new().fill_max_size().hit_passthrough())
                    .child(hud_rows.iter().map(gui_text_layer).collect::<Vec<_>>()),
            );
        }
        // GML `scrLetterbox` (36 px bars): cinematic chrome over menus,
        // transitions, game over, and boss intros — but NOT the
        // campfire title (`Menu/Create_0`: `scrLetterbox(false, 0)`).
        let live_now = state == AppState::InGame && menu_kind.is_none() && !paused && !game_over;
        let boss_intro = self
            .sim
            .world
            .query::<&crate::comps_a::BossIntro>()
            .iter(&self.sim.world)
            .next()
            .is_some();
        let transitioning = self
            .sim
            .world
            .get_resource::<crate::comps_b::FloorTransition>()
            .is_some_and(|f| f.active);
        // GML `GenCont/Destroy`: generation end snaps the camera.
        if self.was_transitioning && !transitioning {
            self.gml_cam.snap = true;
        }
        self.was_transitioning = transitioning;
        if self.was_state != AppState::InGame && state == AppState::InGame {
            self.gml_cam.snap = true;
        }
        // GML `room_restart` parity (`Vlambeer/Create_0` logo branch):
        // the quit-to-menu room recenters the camera (0,0) over black
        // with a fresh live spiral — the previous run's look point and
        // drain never carry over.
        if matches!(state, AppState::MainMenu)
            && !matches!(self.was_state, AppState::MainMenu)
        {
            self.gml_cam.snap = true;
        }
        // GML `Menu/Create_0` verbatim: `with Campfire
        // scr_camera_set_position(x, y)` — the campfire actor sits at
        // world (64,64), so the view top-left snaps to (64,64) and the
        // camp + spiral center stay framed. (The fixed-step camera only
        // runs InGame, and only InGame consumes `snap`, so menus must
        // place the view camera directly here.)
        if matches!(state, AppState::Title) {
            let vw_vh = self.view_world_size;
            self.cam = world_camera(
                Vec2::new(64.0 + vw_vh[0] * 0.5, 64.0 + vw_vh[1] * 0.5),
                gml_view_scale(self.view_viewport_dp),
            );
            self.cam.offset = Vec2::ZERO;
        }
        self.was_state = state;

        // `scrLetterbox(false,0)`). Loading IS letterboxed: GML
        // `GenCont/Create_0` ends with `scrLetterbox(true)`, so the
        // GENERATING screen sits between the bars.
        let bare_room = matches!(
            menu_kind,
            Some(MenuOverlay::Splash | MenuOverlay::MainMenu | MenuOverlay::Title)
        );
        let letterboxed = (!live_now && !bare_room) || boss_intro || transitioning;
        if letterboxed {
            // GML `LETTERBOX_SIZE 36` view px tall (`scrLetterbox`):
            // 36 GUI px → dp at the live GUI scale (720p → 108 dp).
            let bar_dp = 36.0 * (viewport_dp[1].max(1.0) / 240.0);
            let bar = || {
                UiBox(
                    Modifier::new()
                        .fill_max_width()
                        .height(Dp(bar_dp))
                        .background(Color::from_rgba(0, 0, 0, 255))
                        .hit_passthrough(),
                )
            };
            let spacer = UiBox(
                Modifier::new()
                    .fill_max_width()
                    .fill_max_height()
                    .hit_passthrough(),
            );
            layers.push(
                Column(Modifier::new().fill_max_size().hit_passthrough()).child(vec![
                    bar(),
                    spacer,
                    bar(),
                ]),
            );
        }
        if let Some(rows) = menu_rows {
            // GML `GameOver/Draw_0:7-10` dims with `draw_set_alpha(0.7)`
            // (178/255); pause/settings/credits/stats sit on the
            // near-opaque bevy `scrim` (230/255).
            let scrim_alpha = if menu_kind == Some(MenuOverlay::GameOver) {
                178
            } else {
                230
            };
            if dim_menu {
                layers.push(UiBox(
                    Modifier::new()
                        .fill_max_size()
                        .background(Color::from_rgba(0, 0, 0, scrim_alpha))
                        .hit_passthrough(),
                ));
            }
            if !rows.is_empty() {
                // Plain text rows (GML draws the buttons as text;
                layers.push(
                    ZStack(Modifier::new().fill_max_size().hit_passthrough())
                        .child(rows.iter().map(gui_text_layer).collect::<Vec<_>>()),
                );
            }
        }
        ZStack(root_mod).child(layers)
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

fn tick_counter(mut ticks: ResMut<TickCount>) {
    ticks.0 += 1;
}

/// Resources the full sim schedule reads that `setup_run` does not own
/// (mirrors the `insert_schedule_resources` list in `schedule.rs` tests:
/// fixed clock, frame counters, feel sinks, audio queues, death channel,
/// area features, progression extras).
///
/// Startup defaults for run-scoped resources (Run, Score, FloorMask,
/// SaveData, Toast, menus...): bevy inserts these at app build and
/// `setup_run` resets them in place, so booting at Splash (pre-run)
/// never hits a missing-resource validation error.
fn init_schedule_resources(world: &mut World) {
    use crate::audio::{AudioCue, GameAudio};
    use crate::combat::DeathEvents;
    use crate::comps_a::{
        CurrentFrame as CombatFrame, Euphoria, FloorMask, FloorStarted, HammerheadBudget,
        HeavyHeart, LastDamageTaken, MutationChoice, OpenMind, Run, SaveDirty, ScarierFace, Score,
        SelectedCharacter, Toast,
    };
    use crate::comps_b::{
        FloorTransition, IdpdRaidState, LoopTransition, PortalCarriedWeapons, ThroneRoomState,
    };
    use crate::effects::{ChromaticAberration, FlashWhite, HitStop, RumbleRequest, SlowMotion};
    use crate::msg::Queue;
    use crate::progression::DeferredFloorGen;
    use crate::savedata_part::{CrystalDamageTaken, SaveData};
    use crate::state::CurrentFrame as StateFrame;

    const DT: f32 = 1.0 / 30.0;
    let mut time = SimTime::default();
    time.delta_secs = DT;
    world.insert_resource(time);
    world.insert_resource(StateFrame::default());
    world.insert_resource(CombatFrame::default());
    world.insert_resource(repame_fx::Trauma::default());
    world.insert_resource(HitStop::default());
    world.insert_resource(SlowMotion::default());
    world.insert_resource(ChromaticAberration::default());
    world.insert_resource(FlashWhite::default());
    world.insert_resource(GameAudio);
    world.insert_resource(Queue::<AudioCue>::default());
    world.insert_resource(Queue::<RumbleRequest>::default());
    world.insert_resource(DeathEvents::default());
    world.insert_resource(crate::secrets::SecretTriggers::default());
    world.insert_resource(IdpdRaidState::default());
    world.insert_resource(ThroneRoomState::default());
    world.insert_resource(HammerheadBudget::default());
    world.insert_resource(LastDamageTaken::default());
    world.insert_resource(CrystalDamageTaken::default());
    world.insert_resource(PortalCarriedWeapons::default());
    world.init_resource::<crate::pickups::WeaponLabel>();
    world.insert_resource(FloorTransition::default());
    world.init_resource::<NtInput>();
    world.init_resource::<crate::state::Paused>();
    world.init_resource::<AppState>();
    // GML `MakeGame` boot law (disclaimer/save-continue/recontinue cap):
    // headless defaults boot straight to the menu; disk shells set the
    // fields before the first tick.
    world.init_resource::<crate::state::BootFlags>();
    // Area-fog scroll (GML TopCont `fogscroll`, persistent like the
    // controller itself: kept across floors, reset only on reboot).
    world.init_resource::<crate::environment::FogState>();
    // Run-scoped startup defaults (bevy app-build parity): `setup_run`
    // resets every one of these in place, so menus/pre-run frames run
    // the full schedule without missing-resource errors.
    world.insert_resource(Score::default());
    world.insert_resource(Run::default());
    world.insert_resource(FloorMask::default());
    world.insert_resource(SaveDirty::default());
    world.insert_resource(Toast::default());
    world.insert_resource(SelectedCharacter::default());
    world.insert_resource(SaveData::default());
    world.insert_resource(Queue::<FloorStarted>::default());
    world.insert_resource(DeferredFloorGen::default());
    world.insert_resource(LoopTransition::default());
    world.insert_resource(MutationChoice::default());
    world.insert_resource(ScarierFace::default());
    world.insert_resource(Euphoria::default());
    world.insert_resource(OpenMind::default());
    world.insert_resource(HeavyHeart::default());
    world.init_resource::<crate::state::OverlayMenu>();
    world.init_resource::<crate::state::PendingUnpause>();
    world.init_resource::<crate::state::QuitRequested>();
    world.init_resource::<crate::state::menus::MenuState>();
    world.init_resource::<crate::state::menus::MenuEdge>();
    world.init_resource::<crate::audio::AudioChannels>();
    world.init_resource::<Queue<crate::audio::UiBridgeAction>>();
    world.init_resource::<Queue<crate::audio::ReactiveAudioRequest>>();
    world.init_resource::<repame_anim::AnimCatalog>();
}

/// Resolve the art dir: `$NT_ASSETS` -> exe-dir `assets` -> cwd `assets` ->
/// dev-checkout `../nt-recreated-bevy/assets`. Returns `None` when no dir
/// holds `images/anims.json` (placeholder path stays active).
pub fn resolve_assets_dir() -> Option<PathBuf> {
    let has_catalog = |p: &Path| p.join("images").join("anims.json").is_file();
    if let Ok(p) = std::env::var("NT_ASSETS") {
        let p = PathBuf::from(p);
        if has_catalog(&p) {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for cand in [dir.join("assets"), dir.join("../assets")] {
                if has_catalog(&cand) {
                    return Some(cand);
                }
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for cand in [cwd.join("assets"), cwd.join("../nt-recreated-bevy/assets")] {
            if has_catalog(&cand) {
                return Some(cand);
            }
        }
    }
    None
}

/// Fit extent for the follow camera: the viewport size in dp. GML has
/// no zoom: the per-frame [`gml_view_scale`](crate::render::gml_view_scale)
/// carries `units_per_pixel`, so the engine fit shows the live GML view
/// (1280x720 -> 426x240). Pre-scaling here would apply the scale twice
/// (once in the extent, once in the fit).
///
/// `viewport_px` is physical pixels (`Scheduler.size`); `density` is
/// the dp->px scale, so a 1.25x HiDPI window still frames the same view.
pub fn camera_fit_extent(viewport_px: [f32; 2], density: f32) -> [f32; 2] {
    let d = if density.is_finite() && density > 1e-6 {
        density
    } else {
        1.0
    };
    [viewport_px[0] / d, viewport_px[1] / d]
}

/// Player position from sim truth. `None` when no player exists
/// (menus keep the last camera).
fn player_pos(world: &mut World) -> Option<Vec2> {
    world
        .query::<(&Pos, &Player)>()
        .iter(world)
        .next()
        .map(|(p, _)| p.0)
}

/// GML `BackCont` POI chain for the camera (`objects/BackCont/Step_0`):
/// Portal first, then the victory/sit markers — nearest instance of the
/// first non-empty kind wins. Only portals carry the 72 px cap here;
/// the `BecomeNothing` / `NothingDeath` kinds and the `TutCont` +
/// `WeaponChest` override have no port counterpart (no tutorial).
fn nearest_poi(world: &mut World, player: Vec2) -> Option<CamPoi> {
    let mut best: Option<(f32, CamPoi)> = None;
    // Portals outrank everything.
    {
        let mut q = world.query::<(&Pos, &crate::comps_b::Portal)>();
        for (p, _) in q.iter(world) {
            let d = player.distance(p.0);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((
                    d,
                    CamPoi {
                        pos: p.0,
                        capped: true,
                    },
                ));
            }
        }
    }
    if best.is_some() {
        return best.map(|(_, poi)| poi);
    }
    {
        let mut q = world.query::<(&Pos, &crate::comps_b::ThroneVictory)>();
        for (p, _) in q.iter(world) {
            let d = player.distance(p.0);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((
                    d,
                    CamPoi {
                        pos: p.0,
                        capped: false,
                    },
                ));
            }
        }
    }
    {
        let mut q = world.query::<(&Pos, &crate::comps_b::SitZone)>();
        for (p, _) in q.iter(world) {
            let d = player.distance(p.0);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((
                    d,
                    CamPoi {
                        pos: p.0,
                        capped: false,
                    },
                ));
            }
        }
    }
    if best.is_some() {
        return best.map(|(_, poi)| poi);
    }
    best.map(|(_, poi)| poi)
}

/// Backend-neutral char -> [`KeyCode`] (letters lowered by the caller).
fn keycode_for_char(c: char) -> Option<KeyCode> {
    match c {
        'w' => Some(KeyCode::KeyW),
        'a' => Some(KeyCode::KeyA),
        's' => Some(KeyCode::KeyS),
        'd' => Some(KeyCode::KeyD),
        'e' => Some(KeyCode::KeyE),
        'f' => Some(KeyCode::KeyF),
        'q' => Some(KeyCode::KeyQ),
        'g' => Some(KeyCode::KeyG),
        '1' => Some(KeyCode::Digit1),
        '2' => Some(KeyCode::Digit2),
        '3' => Some(KeyCode::Digit3),
        '4' => Some(KeyCode::Digit4),
        '5' => Some(KeyCode::Digit5),
        _ => None,
    }
}

/// Placeholder sprite layer (no-assets path): colored quads from sim truth
/// (canvas draws solid tint rects, so uv 0..1 / page 0 suffices). Covers
/// walls, props, pickups, enemies, the player, and projectiles.
pub fn placeholder_instances(world: &mut World) -> Vec<SpriteInstance> {
    fn quad(center: Vec2, size: f32, tint: [f32; 4]) -> SpriteInstance {
        SpriteInstance {
            center,
            size: Vec2::splat(size),
            color: tint,
            ..Default::default()
        }
    }

    let mut out = Vec::new();
    {
        let mut q = world.query_filtered::<&Pos, With<WallTile>>();
        for pos in q.iter(world) {
            out.push(quad(pos.0, 32.0, [0.35, 0.33, 0.38, 1.0]));
        }
    }
    {
        let mut q = world.query_filtered::<(&Pos, &Prop), Without<WallTile>>();
        for (pos, _) in q.iter(world) {
            out.push(quad(pos.0, 20.0, [0.85, 0.55, 0.20, 1.0]));
        }
    }
    {
        let mut q = world.query::<(&Pos, &Pickup)>();
        for (pos, _) in q.iter(world) {
            out.push(quad(pos.0, 12.0, [0.20, 0.85, 0.85, 1.0]));
        }
    }
    {
        let mut q = world.query::<(&Pos, &Enemy)>();
        for (pos, _) in q.iter(world) {
            out.push(quad(pos.0, 16.0, [0.85, 0.25, 0.25, 1.0]));
        }
    }
    {
        let mut q = world.query::<(&Pos, &Player)>();
        for (pos, _) in q.iter(world) {
            out.push(quad(pos.0, 16.0, [0.30, 0.85, 0.35, 1.0]));
        }
    }
    {
        let mut q = world.query::<(&Pos, &Projectile)>();
        for (pos, _) in q.iter(world) {
            out.push(quad(pos.0, 8.0, [0.95, 0.90, 0.30, 1.0]));
        }
    }
    // Keep the borrow checker honest about the WallCell import parity
    // with `world_instances` (wall bodies carry it sim-side).
    {
        let mut q = world.query::<(&WallCell, &Pos)>();
        let _ = q.iter(world).count();
    }
    out
}

/// Positioned HUD rows ([`GuiRow`](crate::render::GuiRow)): GML
/// `scrDrawPlayerHUD` texts (HP `hp/max` at GUI (67,7), level at
/// (11,16), per-slot ammo at (42+slot*44,21), red `LOW HP` at (110,7))
/// plus the `scrDrawMiscHUD` right-aligned clock/map rows at
/// `view_width - 2`, mapped through the live GML GUI law
/// ([`gui_texts_dp`](crate::render::gui_texts_dp): GUI height 240,
/// full live width). `LOW HP` blinks on the shell beat (GML
/// `sin(wave)` gate over the `drawlowhp` hurt window).
pub fn hud_overlay_lines(world: &mut World, canvas_dp: [f32; 2]) -> Vec<crate::render::GuiRow> {
    // GML `sin(wave) > 0` blink gate over the hurt/low-ammo windows
    // (`wave` ticks per step; the shell approximates it at 12 Hz).
    let blink = world
        .get_resource::<crate::SimTime>()
        .map(|t| (t.elapsed_secs * 12.0).sin() > 0.0)
        .unwrap_or(true);
    hud_gui_texts_dp(world, canvas_dp)
        .into_iter()
        .filter(|(segs, _, _, _, _, _)| {
            if segs.len() != 1 {
                return true;
            }
            let t = segs[0].0.as_str();
            // `LOW HP` + the whole low-ammo block share the GML
            // `sin(wave) > 0` blink gate. `SkillText` toasts blink on
            // `disappear % 2` — same shell beat. (Toast text arrives
            // `@d`-tagged; strip the tag for the match.)
            let plain = t.strip_prefix("@d").unwrap_or(t);
            if plain == "LOW HP"
                || plain == "EMPTY"
                || plain == "NOT ENOUGH RADS"
                || plain.starts_with("LOW ")
                || plain.starts_with("NOT ENOUGH ")
                || t.starts_with("@d")
            {
                return blink;
            }
            true
        })
        .collect()
}

/// Positioned overlay text layer: a full-size padded column holding the
/// fixed-width box the Silkscreen line sits in. Right-aligned rows
/// (misc-HUD clock/area) right-align inside their box. GML
/// `draw_text_nt` `@`-tags arrive pre-split as color runs
/// ([`GuiRow`](crate::render::GuiRow)); multi-run rows render as
/// `AnnotatedText` so `@w/@s/@r/@g/@y/@b/@p` colors show verbatim
/// (single-run rows keep the plain `Text` path).
fn gui_text_layer(row: &crate::render::GuiRow) -> View {
    let align = if row.3 {
        AlignItems::CENTER
    } else if row.5 {
        AlignItems::FLEX_END
    } else {
        AlignItems::FLEX_START
    };
    let color_of = |c: [f32; 4]| {
        Color::from_rgba(
            (c[0] * 255.0) as u8,
            (c[1] * 255.0) as u8,
            (c[2] * 255.0) as u8,
            (c[3] * 255.0) as u8,
        )
    };
    let mut text = if row.0.len() == 1 {
        Text(row.0[0].0.clone())
            .size(Sp(row.2))
            .font_family(NT_UI_FONT_FAMILY)
            .color(color_of(row.0[0].1))
            .single_line()
    } else {
        let mut plain = String::new();
        let mut spans = Vec::new();
        for (seg, c) in &row.0 {
            let start = plain.len();
            plain.push_str(seg);
            let end = plain.len();
            spans.push(repose_core::text::TextSpan {
                start,
                end,
                style: repose_core::text::SpanStyle {
                    color: Some(color_of(*c)),
                    ..Default::default()
                },
                url: None,
            });
        }
        AnnotatedText(repose_core::text::AnnotatedString::new(plain, spans))
            .size(Sp(row.2))
            .font_family(NT_UI_FONT_FAMILY)
            .single_line()
    };
    if row.3 {
        text = text.text_align(repose_core::text::TextAlign::Center);
    } else if row.5 {
        text = text.text_align(repose_core::text::TextAlign::Right);
    }
    Column(
        Modifier::new()
            .fill_max_size()
            .padding_values(PaddingValues {
                left: Dp(row.1[0]),
                right: Dp(0.0),
                top: Dp(row.1[1]),
                bottom: Dp(0.0),
            })
            .align_items(AlignItems::FLEX_START)
            .hit_passthrough(),
    )
    .child(
        Column(
            Modifier::new()
                .width(Dp(row.4))
                .align_items(align)
                .hit_passthrough(),
        )
        .child(text),
    )
}

/// Menu button click routing (GML `PauseButton` objects +
/// `scrMakePauseButtons` layout parity; positions live in
/// `menu_gui_texts` in `render.rs`, actions in `apply_menu_action`):
/// pause MENU/RETRY arm the quit/restart confirm, the confirm swaps in
/// BACK + QUIT/RETRY, CONTINUE resumes, SETTINGS opens settings.
/// Settings/credits BACK rows unwind one level. Every other row/text
/// is inert (`None`).
fn menu_button_action(
    kind: MenuOverlay,
    text: &str,
    pause_confirm: Option<u8>,
) -> Option<UiAction> {
    match kind {
        MenuOverlay::MainMenu => match text {
            // CO-OP has no action (GML early-exits on `!available`);
            // the router stings `sndNoSelect` via `menu_row_denied`.
            "PLAY" => Some(UiAction::MainMenuPlay),
            "SETTINGS" => Some(UiAction::OpenSettings),
            "STATS" => Some(UiAction::ShowStats),
            "QUIT" => Some(UiAction::QuitApp),
            "NORMAL" => Some(UiAction::PlaySubmenu(0)),
            "DAILY" => Some(UiAction::PlaySubmenu(1)),
            "WEEKLY" => Some(UiAction::PlaySubmenu(2)),
            "HARD" => Some(UiAction::PlaySubmenu(3)),
            "CUSTOM" => Some(UiAction::PlaySubmenu(4)),
            "BACK" => Some(UiAction::ClosePlaySubmenu),
            _ => None,
        },
        MenuOverlay::Stats => match text {
            "BACK" => Some(UiAction::CloseOverlay),
            _ => None,
        },
        MenuOverlay::Pause => match pause_confirm {
            None => match text {
                "CONTINUE" => Some(UiAction::Resume),
                "MENU" => Some(UiAction::ShowPauseConfirm(0)),
                "RETRY" => Some(UiAction::ShowPauseConfirm(1)),
                "SETTINGS" => Some(UiAction::OpenSettings),
                _ => None,
            },
            Some(0) => match text {
                "BACK" => Some(UiAction::CancelPauseConfirm),
                "QUIT" => Some(UiAction::ConfirmPause(0)),
                _ => None,
            },
            Some(_) => match text {
                "BACK" => Some(UiAction::CancelPauseConfirm),
                "RETRY" => Some(UiAction::ConfirmPause(1)),
                _ => None,
            },
        },
        MenuOverlay::Settings => match text {
            "BACK" => Some(UiAction::SettingsBack),
            _ => None,
        },
        MenuOverlay::Credits => match text {
            "BACK" => Some(UiAction::CloseOverlay),
            _ => None,
        },
        MenuOverlay::GameOver => match text {
            "MENU" => Some(UiAction::ConfirmPause(0)),
            "RETRY" => Some(UiAction::ConfirmPause(1)),
            _ => None,
        },
        _ => None,
    }
}

/// Menu click router: canvas-dp click position -> [`UiAction`].
/// Hit-tests the screen-anchored overlay buttons in live GML GUI space
/// (GUI height 240, width [`gml_view_size`]; dp per GUI px `k =
/// h/240`), independent of repose hit-testing: the dp layout boxes are
/// full-width alignment boxes that overlap (e.g. SETTINGS spans the
/// MENU column), so only tight per-glyph boxes route correctly.
/// Positions come from [`menu_gui_texts_vw`](crate::render::menu_gui_texts_vw)
/// (GML layout parity); actions from [`menu_button_action`].
fn route_menu_click(
    world: &mut World,
    kind: MenuOverlay,
    dp: [f32; 2],
    viewport_dp: [f32; 2],
) -> Option<UiAction> {
    let vw = gml_view_size(viewport_dp)[0];
    let k = (viewport_dp[1].max(1.0) / 240.0).max(1e-6);
    if !k.is_finite() {
        return None;
    }
    let gx = dp[0] / k;
    let gy = dp[1] / k;
    // Settings rows route through the hot table (per-row toggle /
    // stepper semantics live there, next to the layout).
    if kind == MenuOverlay::Settings {
        let page = world
            .get_resource::<MenuState>()
            .map(|m| m.settings_page)
            .unwrap_or(0);
        return crate::render::settings_click_action(world, page, gx, gy, vw);
    }
    let confirm = world
        .get_resource::<MenuState>()
        .and_then(|m| m.pause_confirm);
    let hit = menu_gui_texts_vw(kind, world, vw)
        .into_iter()
        .find_map(|t| {
            let action = menu_button_action(kind, &t.text, confirm)?;
            // Bevy `bigname_button_at` parity: every menu button owns a
            // fixed 120x22 GUI box centered on its (gx, gy) (the dp text
            const HW: f32 = 60.0;
            const HH: f32 = 11.0;
            let (hx, hy) = if kind == MenuOverlay::MainMenu && t.text == "BACK" {
                (16.0, 20.0)
            } else {
                (t.gx, t.gy)
            };
            if (gx - hx).abs() <= HW && (gy - hy).abs() <= HH {
                Some(action)
            } else {
                None
            }
        });
    if hit.is_some() {
        return hit;
    }
    if kind == MenuOverlay::Credits {
        return Some(UiAction::AdvanceCredits);
    }
    if menu_row_denied(kind, world, gx, gy, vw) {
        crate::state::menus::emit_denied(world);
    }
    None
}

/// Disabled-but-visible rows that sting `sndNoSelect` on click (GML
/// unavailable-button parity). Today only MainMenu CO-OP.
fn menu_row_denied(kind: MenuOverlay, world: &mut World, gx: f32, gy: f32, vw: f32) -> bool {
    if kind != MenuOverlay::MainMenu {
        return false;
    }
    menu_gui_texts_vw(kind, world, vw)
        .into_iter()
        .any(|t| t.text == "CO-OP" && (gx - t.gx).abs() <= 60.0 && (gy - t.gy).abs() <= 11.0)
}

/// Menu overlay selection from state (pure view-model; priority:
/// game-over > pause/settings/credits > mutation offer > none in game).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuOverlay {
    Splash,
    MainMenu,
    Loading,
    Title,
    Pause,
    Settings,
    Credits,
    Mutation,
    GameOver,
    Stats,
}

pub fn menu_overlay_kind(
    state: AppState,
    overlay: OverlayMenu,
    menu: &MenuState,
    game_over: bool,
) -> Option<MenuOverlay> {
    match state {
        AppState::Splash => Some(MenuOverlay::Splash),
        // Overlays surface over the main menu (GML spawns MenuOptions /
        // DrawStats over the buttons); gameplay has no other path here.
        AppState::MainMenu => match overlay {
            OverlayMenu::Settings => Some(MenuOverlay::Settings),
            OverlayMenu::Credits => Some(MenuOverlay::Credits),
            OverlayMenu::Stats => Some(MenuOverlay::Stats),
            _ => Some(MenuOverlay::MainMenu),
        },
        AppState::Loading => Some(MenuOverlay::Loading),
        // Settings/Credits opened over the campfire surface above it
        // (same as MainMenu); otherwise the title pods own the clicks.
        AppState::Title => match overlay {
            OverlayMenu::Settings => Some(MenuOverlay::Settings),
            OverlayMenu::Credits => Some(MenuOverlay::Credits),
            _ => Some(MenuOverlay::Title),
        },
        AppState::InGame => {
            if game_over {
                return Some(MenuOverlay::GameOver);
            }
            match overlay {
                OverlayMenu::Pause => Some(MenuOverlay::Pause),
                OverlayMenu::Settings => Some(MenuOverlay::Settings),
                OverlayMenu::Credits => Some(MenuOverlay::Credits),
                // Stats only opens over the main menu; it can never be
                // live here, so there is nothing to show.
                OverlayMenu::Stats => None,
                OverlayMenu::None => {
                    if menu.mutation_count > 0 {
                        Some(MenuOverlay::Mutation)
                    } else {
                        None
                    }
                }
            }
        }
    }
}

/// Menu overlay lines for one selected overlay: the text column of
/// [`menu_gui_texts`](crate::render::menu_gui_texts) (bevy panel
/// strings verbatim; positions/colors/sizes live in the GUI rows).
pub fn menu_overlay_lines(kind: MenuOverlay, world: &mut World) -> Vec<String> {
    menu_gui_texts(kind, world)
        .into_iter()
        .map(|t| t.text)
        .collect()
}

/// Root view: playable game view (sim + render + vortex + HUD + menus).
/// Thin wrapper over [`App::view`] so `main.rs` stays trivial.
pub fn root_view(sched: &mut Scheduler, ctx: &RenderContext, app: &mut App, dt: Duration) -> View {
    app.view(sched, ctx, dt)
}
