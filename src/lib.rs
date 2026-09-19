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
//! Input-event mapping (keys arrive via `Modifier::on_key_event` on the
//! focused root, pointer clicks via the viewport `on_event`
//! [`PickEvent`] — whose `Press`/`Click` now carry their
//! [`PointerButton`], so no parallel root button handlers are needed —
//! staged into the sim through `feed_input`):
//! - WASD/arrows -> move, mouse cursor position -> aim (every frame),
//!   Space / left-click -> fire, Shift / right-click -> ability+spec
//!   (GML `spec` on `mb_right`; Shift arrives as its own `Key`
//!   variants now), E/F/Q/G/Tab/Enter -> interact/confirm,
//!   1-5 -> weapon slots / mutation picks / title cursor / main-menu+
//!   pause rows, Left/Right arrows -> title cursor + mutation highlight,
//!   Up/Down (+Left/Right) -> main-menu cursor + settings cursor/values,
//!   click in game -> aim-at-click + fire, click in menus -> positional
//!   buttons (main-menu rows, title pods/GO/loadout, pause/settings/
//!   credits/stats buttons via screen-position routing; strays dropped),
//!   right-click in settings/credits -> Back (GML `BackButton`
//!   `mb_right`), Space on title -> loadout panel, Esc -> pause toggle
//!   (+ overlay unwind + main-menu overlay close), R -> game-over retry.
//! - NOT wired: text entry for profile/color inputs (buttons only;
//!   color cycles presets). Key rebinding works through the REMAP
//!   capture; gamepad sticks/triggers drain through `stage_gamepad`.
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
    Key, KeyEvent, KeyEventType, PhysicalKey, PointerButton, PointerEvent,
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
    GamepadState, KeyCode, MouseState, NtInput, TouchContact, sample_gamepads, sample_touch,
};
use crate::render::{
    ATLAS_PAGES, ATLAS_SIZE, CamPoi, CamStepInput, GmlCamera, RenderAssets, Z_BLOOM, Z_CROSSHAIR,
    Z_FAINTED, Z_FOG, Z_FX, Z_HUD, Z_MENU, Z_PORTAL_INDICATOR, Z_SHADOW, Z_SIDEART,
    Z_SPIRAL_FIGURES, Z_SPLASH, background_color, bloom_sprites, cam_viewdist_for,
    crosshair_sprites, decode_png, fainted_bar_sprites, fog_sprites, fx_instances, fx_texts,
    gml_camera_step, gml_view_scale, gml_view_size, hud_gui_texts_dp, hud_sprites, menu_gui_texts,
    menu_gui_texts_dp, menu_gui_texts_vw, menu_sprites, portal_indicator_sprites, shadow_sprites,
    sideart_sprites, spiral_figures, splash_sprites, stamp_z, title_cam_focus, title_camera_step,
    view_rect_world, world_camera,
    world_instances,
};
use crate::schedule::build_sim_schedule;
use crate::setup::setup_run_with_seed;
use crate::spatial::Pos;
use crate::state::menus::{MenuEdge, MenuState, apply_menu_action};
use crate::state::{AppState, OverlayMenu};
use crate::keymap::{InputMapState, KeyBindings};
use crate::vortex::{SpiralCtl, gml_area_for_area};

pub mod anim;
pub mod audio;
pub mod boss_ai;
pub mod combat;
pub mod comps_a;
pub mod comps_b;
pub mod crown;
pub mod data;
pub mod decide_wep;
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
pub mod keymap;

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
    /// Disk path the save was loaded from (`load_save` records it;
    /// `save_now` flushes here on every dirty write, GML `scrSave`
    /// parity — without it a restart shows the old mappings).
    save_path: Option<PathBuf>,
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
    /// Decoded hardware cursor (`CursorIcon::Custom` payload) for the
    /// live `opt_crosshair` frame + `opt_cursorcol` tint. Cached so the
    /// runner's `create_custom_cursor` handle rebuilds only when the
    /// art changes (frame/tint/pixels/scale); cleared when assets unload.
    cursor_img: Option<std::sync::Arc<repose_core::CustomCursorImage>>,
    cursor_img_key: Option<(i32, [u32; 3], u64, u32)>,
    /// Decoded vortex background textures for `vortex_tex_area`
    /// (slots: spiral, bolt, debris, proto, idpd, idpd2).
    vortex_tex: Vec<VortexTexture>,
    vortex_tex_area: Option<u8>,
    // -- staged shell input (drained into `NtInput`/`MenuEdge` per frame) --
    // `held` is the event-staged key set: `on_key_event` presses land
    // here through `stage_code`/`stage_physical`, so held keys survive
    // focus moves without depending on a focused widget (GML
    // `keyboard_check` parity). Layout independent: AZERTY reports the
    // same `KeyW` position GML's `ord("W")` binds. Cleared on window
    // focus loss.
    held: HashSet<KeyCode>,
    edges: Vec<KeyCode>,
    /// Whether the window currently has focus. `false` (set by the
    /// shell's focus handler) drops `held` outright — winit delivers
    /// no key-ups across an alt-tab.
    window_focused: bool,
    clicks: Vec<repame_input::StagedClick>,
    /// Screen-anchored pointer staging (raw window-physical px in, live
    /// world point out via [`App::live_cursor_world`]). The freshness
    /// discipline lives in [`repame_input::AimTracker`]; unprojection
    /// needs this frame's camera so it stays here.
    aim: repame_input::AimTracker,
    pause_edge: bool,
    restart_edge: bool,
    interact_edge: bool,
    /// Right mouse held (GML `spec`/ability on `mb_right` parity).
    rmb_held: bool,
    /// Left mouse held (GML `fire` on `mb_left` parity): set on primary
    /// pointer-down, cleared on primary pointer-up, so automatic weapons
    /// keep firing while held (clicks alone are single-frame edges).
    lmb_held: bool,
    /// Right-button down/up edges staged this window. The viewport
    /// `PickEvent` carries its button, so secondary presses stage here
    /// directly (and route Back over settings/credits like GML
    /// `BackButton`).
    rmb_down_edge: bool,
    rmb_up_edge: bool,
    /// Staged gamepad snapshots, one per pad (bevy `sample_input`
    /// `gamepads` query order verbatim: keyboard, then pads in order,
    /// then touch).
    pads: Vec<GamepadState>,
    /// Latched gamepad presence: set while any staged snapshot shows a
    /// connected pad, cleared when a frame stages none. `pads` drains
    /// every frame so it can't answer "is a pad in use" outside
    /// `feed_input`; this carries that fact to the cursor gate.
    pad_live: bool,
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
        let spiral =
            SpiralCtl::warmed_up_for_gml_area_seeded_in_view(gml_area_for_area(area), seed, 426.0);
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
            save_path: None,
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
            cursor_img: None,
            cursor_img_key: None,
            vortex_tex: Vec::new(),
            vortex_tex_area: None,
            held: HashSet::new(),
            edges: Vec::new(),
            window_focused: true,
            clicks: Vec::new(),
            aim: repame_input::AimTracker::default(),
            pause_edge: false,
            restart_edge: false,
            interact_edge: false,
            rmb_held: false,
            lmb_held: false,
            rmb_down_edge: false,
            rmb_up_edge: false,
            pads: Vec::new(),
            pad_live: false,
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
        self.cursor_img = None;
        self.cursor_img_key = None;
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

    /// Load the save file into the sim (`SaveData` resource) and sync
    /// the editable keymap from its rows (GML `scrOptionsLoadKeymaps`
    /// on boot). Disk shells call this once after `App::new`; missing
    /// files keep defaults. Returns the save used.
    pub fn load_save(&mut self, path: &Path) -> crate::savedata_part::SaveData {
        let save = crate::savedata_part::load_or_default(path);
        let map = save.key_bindings.to_keymap();
        self.sim.world.insert_resource(save);
        self.sim.world.init_resource::<InputMapState>();
        self.sim.world.resource_mut::<InputMapState>().map = map;
        self.save_path = Some(path.to_path_buf());
        self.sim
            .world
            .resource::<crate::savedata_part::SaveData>()
            .clone()
    }

    /// Flush the live save to disk when dirty (GML `scrSave` on every
    /// rebind/option change: `scrOptionsSaveKeymaps` + `scrSave`). The
    /// dirty flag alone never reaches disk — without this a restart
    /// shows the old mappings. Called from `persist_keymap`.
    fn save_now(&mut self) {
        let dirty = self
            .sim
            .world
            .get_resource::<crate::comps_a::SaveDirty>()
            .is_some_and(|d| d.0);
        if !dirty {
            return;
        }
        if let (Some(path), Some(save)) = (
            self.save_path.clone(),
            self.sim
                .world
                .get_resource::<crate::savedata_part::SaveData>()
                .cloned(),
        ) {
            if let Err(e) = crate::savedata_part::store_save_to_file(&save, &path) {
                eprintln!("nt: save failed ({}): {e}", path.display());
            } else {
                self.sim.world.resource_mut::<crate::comps_a::SaveDirty>().0 = false;
            }
        }
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
            let view_w = self.spiral.view_w;
            self.spiral =
                SpiralCtl::warmed_up_for_gml_area_seeded_in_view(gml_area_for_area(area), seed, view_w);
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
            let view_w = self.spiral.view_w;
            self.spiral = SpiralCtl::warmed_up_for_gml_area_seeded_in_view(0, seed, view_w);
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
            let view_w = self.spiral.view_w;
            self.spiral =
                SpiralCtl::warmed_up_for_gml_area_seeded_in_view(gml_area_for_area(area), seed, view_w);
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
        let live_hover = self.live_cursor_world();
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
            self.spiral = SpiralCtl::warmed_up_for_gml_area_seeded_in_view(
                gml_area_for_area(area),
                seed,
                self.spiral.view_w,
            );
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
}

pub(crate) mod nt_shortcuts {
    use std::cell::Cell;
    use std::rc::Rc;

    use repose_core::input::{Key, Modifiers};
    use repose_core::shortcuts;

    use super::App;

    pub const PAUSE: &str = "nt.pause";
    pub const RESTART: &str = "nt.restart";
    pub const CONFIRM: &str = "nt.confirm";

    pub fn map() -> shortcuts::ShortcutMap {
        shortcuts::ShortcutMap::new()
            .bind(Key::Escape, Modifiers::default(), action_for(PAUSE))
            .bind(Key::Character('r'), Modifiers::default(), action_for(RESTART))
            .bind(Key::Enter, Modifiers::default(), action_for(CONFIRM))
    }

    fn action_for(name: &'static str) -> shortcuts::Action {
        shortcuts::Action::Custom(name.into())
    }

    thread_local! {
        static MAP_INSTALLED: Cell<bool> = const { Cell::new(false) };
    }

    pub fn install(app: *mut App) {
        // Process-lifetime install, called every frame from `view`:
        // the map is pushed once (a push per frame would grow the scope
        // stack unboundedly), while the handler is replaced every call
        // so the raw `App` pointer never dangles across moves. Deliberately
        // not `scoped_effect`: there is no owning scope to clean up after.
        if !MAP_INSTALLED.with(|c| c.replace(true)) {
            let _ = shortcuts::InstallShortcutMap(map());
        }
        let _ = shortcuts::InstallShortcutHandler(Rc::new(move |action| {
            let app = unsafe { &mut *app };
            match action {
                shortcuts::Action::Custom(key) if key.as_ref() == PAUSE => {
                    app.pause_edge = true;
                    true
                }
                shortcuts::Action::Custom(key) if key.as_ref() == RESTART => {
                    app.restart_edge = true;
                    true
                }
                shortcuts::Action::Custom(key) if key.as_ref() == CONFIRM => {
                    app.interact_edge = true;
                    true
                }
                _ => false,
            }
        }));
    }
}

impl App {
    /// Stage one shell key event (called from the root `on_key_event`
    /// handler; see the module docs for the mapping table). Character
    /// keys are matched by physical position (`KeyW`, not `'w'`), so
    /// non-US layouts move the same way GML's `ord("W")` does on a US
    /// board.
    fn handle_key(&mut self, ke: &KeyEvent) {
        let down = matches!(ke.event_type, KeyEventType::Down);
        match &ke.key {
            // Esc / Enter / R stage ONLY via the scoped global shortcut
            // map (`nt_shortcuts::map` + runtime `dispatch_action`). No
            // direct staging here: this bubble return must stay a no-op
            // for them so the shortcut path has single ownership.
            Key::Escape | Key::Enter => {}
            Key::ShiftLeft => self.stage_code(KeyCode::ShiftLeft, down, ke.is_repeat),
            Key::ShiftRight => self.stage_code(KeyCode::ShiftRight, down, ke.is_repeat),
            Key::Space => self.stage_code(KeyCode::Space, down, ke.is_repeat),
            Key::Tab => self.stage_code(KeyCode::Tab, down, ke.is_repeat),
            Key::ArrowUp => self.stage_code(KeyCode::ArrowUp, down, ke.is_repeat),
            Key::ArrowDown => self.stage_code(KeyCode::ArrowDown, down, ke.is_repeat),
            Key::ArrowLeft => self.stage_code(KeyCode::ArrowLeft, down, ke.is_repeat),
            Key::ArrowRight => self.stage_code(KeyCode::ArrowRight, down, ke.is_repeat),
            Key::Character(c) => {
                // Physical position first: the typed [`PhysicalKey`]
                // owns `held` levels + letter/digit edges, so non-US
                // layouts move the same way GML's `ord("W")` does on a
                // US board. The glyph below is only the fallback for
                // synthetic events (tests).
                if let Some(key) = ke.physical {
                    // Capture first, before `stage_physical` can swallow
                    // the press: `stage_physical` returns early while a
                    // capture is armed (the key must not fire gameplay),
                    // and Space's `Key::Space` match arm below never runs
                    // on the physical path — without this Space could
                    // never rebind.
                    if down && !ke.is_repeat && self.capture_armed() {
                        self.capture_physical_press(key);
                        return;
                    }
                    self.stage_physical(key, down, ke.is_repeat);
                    return;
                }
                let c = c.to_ascii_lowercase();
                if c == ' ' {
                    self.stage_code(KeyCode::Space, down, ke.is_repeat);
                } else if c == 'r' {
                    self.stage_physical(PhysicalKey::KeyR, down, ke.is_repeat);
                } else {
                    let _ = c;
                    if down && !ke.is_repeat {
                        self.capture_key_press(&ke.key);
                    }
                }
            }
            Key::Backspace => {
                // GML REMAP cancel key (Backspace clears the pending
                // rebind). Position-staged so layouts agree.
                if down && !ke.is_repeat {
                    self.cancel_remap_capture();
                }
            }
            _ => {}
        }
    }

    /// Stage one physical key transition. Down transitions push press
    /// edges (deduped against `held`); releases only clear `held`.
    pub fn stage_physical_key(&mut self, key: PhysicalKey, down: bool) {
        self.stage_physical(key, down, false);
    }

    fn stage_physical(&mut self, key: PhysicalKey, down: bool, is_repeat: bool) {
        // Capture first: the remap gesture accepts ANY physical key,
        // including keys with no gameplay `KeyCode` (KeyX etc. never
        // reach the table below). GML captures any pressed input.
        if down && !is_repeat && self.capture_armed() {
            self.capture_physical_press(key);
            return;
        }
        let Some(code) = crate::input::keycode_for_physical(key) else {
            return;
        };
        // All edges stage here: focus-routed `handle_key` lands fresh
        // presses in `edges`; `held`-insert dedups double delivery, so
        // one landing pushes one edge.
        if down {
            let fresh = self.held.insert(code);
            if fresh && !is_repeat {
                self.edges.push(code);
            }
        } else {
            self.held.remove(&code);
        }
    }

    /// Window focus changed (shell forwards winit `Focused`). Losing
    /// focus drops every held key + edge: no key-ups arrive across an
    /// alt-tab, and GML's `keyboard_check` reads all-up there too.
    pub fn set_window_focused(&mut self, focused: bool) {
        self.window_focused = focused;
        if !focused {
            self.held.clear();
            self.edges.clear();
            self.lmb_held = false;
            self.rmb_held = false;
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

    /// REMAP capture: the next pressed input resolves the pending
    /// rebind (`Key[$ key][type] = k`, `Other_10:374-404`). Mouse
    /// buttons capture from the viewport press path (`pick_down`);
    /// keys capture here and in `stage_physical`.
    /// Returns `true` while a capture is armed (callers skip normal
    /// staging so the capture key does not fire gameplay).
    pub(crate) fn capture_armed(&self) -> bool {
        self.sim
            .world
            .get_resource::<InputMapState>()
            .is_some_and(|s| s.capture.is_some())
    }

    /// Resolve the pending capture with a mouse button (viewport press
    /// path). GML only captures `mb_left`/`mb_right` for the keyboard
    /// side; gamepad-side captures resolve from pad edges in
    /// `feed_input`.
    fn capture_mouse_press(&mut self, left: bool) {
        use repame_input::KeymapEntry;
        self.sim.world.init_resource::<InputMapState>();
        let entry = if left {
            KeymapEntry::Mouse(repose_core::input::PointerButton::Primary)
        } else {
            KeymapEntry::Mouse(repose_core::input::PointerButton::Secondary)
        };
        let mut state = self.sim.world.resource_mut::<InputMapState>();
        let Some(capture) = state.capture.clone() else {
            return;
        };
        if capture.device == repame_input::KeymapDevice::KeyboardMouse {
            state.map.resolve_capture(&capture, Some(entry));
            state.capture = None;
            self.persist_keymap();
        }
    }

    pub(crate) fn capture_key_press(&mut self, key: &repose_core::input::Key) {
        use repame_input::KeymapEntry;
        self.sim.world.init_resource::<InputMapState>();
        // Glyph path: Space/Tab/Enter/Escape map explicitly; any other
        // glyph resolves as its lowercase char (same swallow bug as the
        // physical path had — GML captures any pressed input).
        let chord = match key {
            repose_core::input::Key::Space => repose_core::shortcuts::KeyChord::new(
                repose_core::input::Key::Space,
                repose_core::input::Modifiers::default(),
            ),
            repose_core::input::Key::Tab => repose_core::shortcuts::KeyChord::new(
                repose_core::input::Key::Tab,
                repose_core::input::Modifiers::default(),
            ),
            repose_core::input::Key::Enter => repose_core::shortcuts::KeyChord::new(
                repose_core::input::Key::Enter,
                repose_core::input::Modifiers::default(),
            ),
            repose_core::input::Key::Escape => repose_core::shortcuts::KeyChord::new(
                repose_core::input::Key::Escape,
                repose_core::input::Modifiers::default(),
            ),
            repose_core::input::Key::ShiftLeft => repose_core::shortcuts::KeyChord::new(
                repose_core::input::Key::ShiftLeft,
                repose_core::input::Modifiers::default(),
            ),
            repose_core::input::Key::ShiftRight => repose_core::shortcuts::KeyChord::new(
                repose_core::input::Key::ShiftRight,
                repose_core::input::Modifiers::default(),
            ),
            repose_core::input::Key::Character(c) => repose_core::shortcuts::KeyChord::new(
                repose_core::input::Key::Character(c.to_ascii_lowercase()),
                repose_core::input::Modifiers::default(),
            ),
            _ => return,
        };
        let mut state = self.sim.world.resource_mut::<InputMapState>();
        let Some(capture) = state.capture.clone() else {
            return;
        };
        if capture.device == repame_input::KeymapDevice::KeyboardMouse {
            state
                .map
                .resolve_capture(&capture, Some(KeymapEntry::Key(chord)));
            state.capture = None;
            self.persist_keymap();
        }
    }

    pub(crate) fn capture_physical_press(&mut self, key: PhysicalKey) {
        use repame_input::KeymapEntry;
        self.sim.world.init_resource::<InputMapState>();
        if !self.capture_armed() {
            return;
        }
        let Some(chord) = repame_input::chord_for_physical(key) else {
            return;
        };
        let mut state = self.sim.world.resource_mut::<InputMapState>();
        let Some(capture) = state.capture.clone() else {
            return;
        };
        if capture.device == repame_input::KeymapDevice::KeyboardMouse {
            state
                .map
                .resolve_capture(&capture, Some(KeymapEntry::Key(chord)));
            state.capture = None;
            self.persist_keymap();
        }
    }

    fn cancel_remap_capture(&mut self) {
        self.sim.world.init_resource::<InputMapState>();
        let mut state = self.sim.world.resource_mut::<InputMapState>();
        // GML Backspace clears the entry (`Key[$ key][type]` stays but
        // reads unbound); Escape cancels the gesture.
        if state.capture.is_some() {
            let capture = state.capture.clone().expect("checked");
            state.map.resolve_capture(&capture, None);
            state.capture = None;
            self.persist_keymap();
        }
    }

    /// Write the live keymap back into the save resource + dirty flag
    /// (GML `scrOptionsSaveKeymaps` + `scrSave` on every rebind).
    fn persist_keymap(&mut self) {
        let rows = self
            .sim
            .world
            .get_resource::<InputMapState>()
            .map(|s| KeyBindings::from_keymap(&s.map))
            .unwrap_or_default();
        self.sim.world.init_resource::<crate::savedata_part::SaveData>();
        self.sim
            .world
            .resource_mut::<crate::savedata_part::SaveData>()
            .key_bindings = rows;
        self.sim.world.init_resource::<crate::comps_a::SaveDirty>();
        self.sim.world.resource_mut::<crate::comps_a::SaveDirty>().0 = true;
        self.save_now();
    }

    /// Viewport mouse-button down: secondary stages the RMB edge
    /// (GML `spec` / menu Back; the sampler reads it through the
    /// `MouseState::right_*` channel), primary latches `lmb_held` so
    /// automatic weapons keep firing while held. A pending REMAP
    /// capture eats the press instead (GML captures `mb_left` /
    /// `mb_right` as the new binding).
    fn pick_down(&mut self, button: PointerButton) {
        let left = button == PointerButton::Primary;
        if self.capture_armed() {
            self.capture_mouse_press(left);
            return;
        }
        if left {
            self.lmb_held = true;
            return;
        }
        self.rmb_down_edge = !self.rmb_held;
        self.rmb_held = true;
    }

    /// Viewport mouse-button release: the `Click` up-edge carries the
    /// same button, so the held latch clears here (the polled snapshot
    /// in `feed_polled` repairs a release outside the window).
    fn pick_up(&mut self, button: PointerButton) {
        if button == PointerButton::Primary {
            self.lmb_held = false;
        } else {
            self.rmb_held = false;
            self.rmb_up_edge = true;
        }
    }

    /// Stage the cursor's window-physical px position (root
    /// `on_pointer_move`, y-down: `position_in_window()`, the same
    /// space [`App::stage_hover`] stages from the viewport). Stored raw
    /// — [`App::px_to_world`] unprojects it through the live camera
    /// each frame.
    fn cursor_move(&mut self, phys_px: Vec2) {
        self.aim.cursor_move(phys_px);
    }

    /// Stage one viewport hover: the baked world point AND the raw
    /// window-physical `screen` px. The engine contract on
    /// `PickEvent::Hover.screen` is explicit: "Games stage this raw and
    /// unproject through the live camera each frame (screen-anchored
    /// aim)". The baked point is the stale fallback (touch/pen never
    /// stage cursor moves, so theirs is the only cursor); the px is
    /// the live source `live_cursor_world` prefers. `cursor_move`
    /// (root `on_pointer_move`) stages window px in the same space and
    /// refreshes the same way — the viewport hover px arrives on every
    /// free move, which is exactly when the root handler stays silent
    /// (free moves dispatch only to the topmost region).
    fn stage_hover(&mut self, world: Vec2, screen: [f32; 2]) {
        self.aim.stage_hover(world, screen);
    }

    /// Live cursor in world coords: staged window-physical px
    /// unprojected through this frame's camera (`Camera2d::dp_to_world_pt`
    /// over the dp viewport extent — bevy `player_aim`
    /// `viewport_to_world_2d` parity). `None` until the first pointer
    /// move; callers fall back to the last viewport `Hover` world point
    /// (touch/pen never stage cursor moves).
    /// The px sources win outright while fresh (they re-unproject every
    /// frame, so aim stays glued to the on-screen pointer instead of
    /// sliding on the ground as the camera moves); once both go stale
    /// the baked hover point takes over so the cursor never freezes on
    /// an old camera. `hover_px` is the primary source — the viewport
    /// `Hover` fires on every free move, while root `cursor_move` runs
    /// almost exclusively while a button is held (capture-path
    /// dispatch reaches ancestors) and its px goes stale ~30 frames
    /// after the last drag.
    fn live_cursor_world(&self) -> Option<Vec2> {
        // Staging discipline (freshness order, stale fallback) lives in
        // `AimTracker`; unprojection needs this frame's camera, so it
        // stays here.
        let aim = &self.aim;
        if let Some(px) = aim.live_px() {
            return self.px_to_world(Some(px)).or_else(|| aim.baked());
        }
        aim.baked().or_else(|| self.px_to_world(None))
    }

    /// Unproject one window-physical px point through this frame's
    /// camera. Pure viewport math lives in
    /// [`repame_sprite::unproject_px`]; the NT wrappings (`Option`
    /// staging, NT's dp-viewport fields) stay here.
    fn px_to_world(&self, px: Option<Vec2>) -> Option<Vec2> {
        let px = px?;
        let d = self.view_density.max(1e-6);
        // `camera_fit_extent` takes PHYSICAL px (it divides by density
        // itself): stored dp must be scaled back up first, or the
        // extent double-divides and the aim scale collapses (~2.4x too
        // fast at density 1.25).
        let extent = camera_fit_extent(
            [self.view_viewport_dp[0] * d, self.view_viewport_dp[1] * d],
            self.view_density,
        );
        Some(repame_sprite::unproject_px(
            px,
            self.view_viewport_dp,
            extent,
            self.view_density,
            &self.cam,
        ))
    }

    /// Decode the live `sprCrosshair` frame + `opt_cursorcol` tint into
    /// the hardware cursor payload (`cursor_img`). The mechanical pixel
    /// work lives in [`repame_sprite::cursor_frame`]; this owns only
    /// settings reads, file IO, and the `cursor_img` cache. GML draws
    /// the raw strip cell at GUI mouse (`Draw_75`, no lerp, alpha 1);
    /// the hotspot is the catalog origin (crosshair strips center on
    /// (8,8)). Missing assets clear the payload — callers fall back to
    /// `Hidden` (the old software path is gone).
    ///
    /// Size law: GML draws the cell at scale 1 in GUI space
    /// (`device_mouse_x_to_gui`, GUI stretched over the window), so the
    /// 16px cell covers `16 * window_w / view_w` screen px (48 at
    /// 1280x720). The OS cursor buffer is physical px while
    /// `units_per_pixel` is dp, so magnification is
    /// `density / units_per_pixel` — not `1 / units_per_pixel`, which
    /// undersizes the cursor on fractional display scales. The scale
    /// rides the cache key so resizes and monitor moves re-decode.
    fn refresh_cursor_img(&mut self, frame: i32, tint: [f32; 4]) {
        let Some(assets) = self.assets.as_ref() else {
            self.cursor_img = None;
            self.cursor_img_key = None;
            return;
        };
        let Some(dir) = self.assets_dir.clone() else {
            self.cursor_img = None;
            self.cursor_img_key = None;
            return;
        };
        let frames = crate::render::strip_frames_pub(assets, "images/sprCrosshair.png").max(1) as i32;
        let path = dir.join("images").join("sprCrosshair.png");
        let Ok((sw, sh, rgba)) = crate::render::decode_png(&path) else {
            self.cursor_img = None;
            self.cursor_img_key = None;
            return;
        };
        let Some((_, def)) = assets.uv("images/sprCrosshair.png", frame) else {
            self.cursor_img = None;
            self.cursor_img_key = None;
            return;
        };
        let mag = (self.view_density.max(1e-6)
            / crate::render::gml_view_scale(self.view_viewport_dp).max(1e-6))
        .round()
        .clamp(1.0, 8.0) as u32;
        let Some(built) = repame_sprite::cursor_frame(
            &rgba,
            sw,
            sh,
            frames as u32,
            frame,
            (def.w, def.h),
            (def.xorigin, def.yorigin),
            tint,
            mag,
            {
                let mut h: u64 = 0xcbf29ce484222325;
                for b in rgba.iter().copied() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                h
            },
        ) else {
            self.cursor_img = None;
            self.cursor_img_key = None;
            return;
        };
        if self.cursor_img_key == Some(built.key) {
            return;
        }
        self.cursor_img = Some(std::sync::Arc::new(repose_core::CustomCursorImage {
            rgba: built.rgba.into(),
            size: built.size,
            hotspot: built.hotspot,
        }));
        self.cursor_img_key = Some(built.key);
    }

    /// Stage one gamepad snapshot for this tick (shells map
    /// `repame-shell` `GamepadEvent`s / platform pad state onto
    /// [`GamepadState`]; drained by [`App::feed_input`] in stage order).
    /// Also latches [`App::pad_live`]: shells stage only live pads, so
    /// any snapshot means a pad is in use this frame.
    pub fn stage_gamepad(&mut self, pad: GamepadState) {
        self.pads.push(pad);
        self.pad_live = true;
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
        self.clicks.push(repame_input::StagedClick {
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

    /// Sync event-staged levels against the platform snapshot
    /// (`Scheduler::held_keys` + mouse levels). Event handlers own
    /// press edges; the polled set only repairs what they missed:
    /// release a key/button the handlers never saw go up (release
    /// outside the window, swallowed key-up), but never force a down
    /// the handlers missed. Keyboard and mouse share the rule; Esc/R
    /// shortcut edges are untouched (shortcuts own those).
    pub fn feed_polled(&mut self, sched: &Scheduler) {
        if !sched.window_focused {
            self.set_window_focused(false);
            return;
        }
        self.window_focused = true;
        repame_input::reconcile_held(
            &mut self.held,
            &sched.held_keys,
            crate::input::keycode_for_physical,
            crate::input::physical_keys_for_code,
        );
        if !sched.mouse_primary {
            self.lmb_held = false;
        }
        if !sched.mouse_secondary {
            self.rmb_held = false;
        }
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
            && overlay == OverlayMenu::None
            && self
                .sim
                .world
                .get_resource::<MenuState>()
                .is_none_or(|m| m.unlock_queue.is_empty());
        let menu_open = state == AppState::InGame
            && (paused
                || overlay != OverlayMenu::None
                || self
                    .sim
                    .world
                    .get_resource::<MenuState>()
                    .is_some_and(|m| !m.unlock_queue.is_empty()))
            && !game_over;

        // Context switch (Godot `_gui_input`-before-`_unhandled_input`
        // parity): one screen owns Space/arrows per frame. Menus consume
        // them as nav/confirm; live gameplay derives action pulses from
        // them; never both. `just` still fans out to every rebound
        // action inside the active context, so shared bindings all fire.
        let in_menu = !live_play;
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
        // Right-button press: staged straight from the viewport's
        // buttoned `PickEvent` (see the viewport `on_event` closures),
        // so there is nothing to drop here. Each arm below decides
        // what RMB means (Back over settings/credits, silent
        // elsewhere) without eating a coincident left click.
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
        // Right-button gameplay pulses: the viewport `PickEvent` button
        // stages `rmb_down_edge`/`rmb_held` directly, and the keymap
        // `Spec` row is mouse-Secondary, read through
        // `MouseState::right_*` below.
        let mouse = MouseState {
            right_held: self.rmb_held,
            right_pressed: rmb_down,
            ..mouse
        };
        {
            self.sim.world.init_resource::<InputMapState>();
            let keymap = self.sim.world.resource::<InputMapState>().clone();
            let mut input = self.sim.world.resource_mut::<NtInput>();
            // Menu screens own Space/arrows: strip their edges before the
            // gameplay sampler so one press can't both confirm a menu
            // row and pulse a gameplay action (fire/swap).
            let (held, just) = if in_menu {
                let held: HashSet<KeyCode> = self
                    .held
                    .iter()
                    .copied()
                    .filter(|c| !menu_owned_key(*c))
                    .collect();
                let sampled: HashSet<KeyCode> =
                    just.iter().copied().filter(|c| !menu_owned_key(*c)).collect();
                (held, sampled)
            } else {
                (self.held.clone(), just.clone())
            };
            crate::input::sample_keyboard_mapped(&held, &just, &mouse, Some(&keymap), &mut input);
            // Bevy `sample_input` layering verbatim: gamepads in query
            // order, then touch. Sticks overwrite nonzero axes; pulses
            // OR-accumulate; slots replace; cycle saturating-adds.
            let pads = self.pads.drain(..).collect::<Vec<_>>();
            // No snapshots staged this frame = no live pad: unlatch so
            // the cursor gate falls back to keyboard mode.
            self.pad_live = !pads.is_empty();
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
        // Right-click and Shift share the `spec` action (GML `spec`
        // on `mb_right` plus the Shift keyboard fallback); on Title
        // that action toggles loadout/hardmode, so a right-click
        // would toggle panels as a side effect. Drop the pulse —
        // Back only travels via Esc here.
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
                let aim_hover = self.live_cursor_world();
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
        self.feed_polled(sched);
        self.feed_input();
        self.advance(dt);
        // Any system may have dirtied the save mid-tick (settings,
        // remap reset, unlocks...); flush here so a restart always
        // shows the latest mappings, GML `scrSave` parity.
        self.save_now();
        // GML `UberCont/Step_0:175-183` cursor law, computed from the
        // post-tick sim state: keyboard mode hides the OS cursor (the
        // game draws its own crosshair at the live cursor position),
        // menus/mouse mode shows it. Touch-driven (no keyboard, no
        // pad) always shows it — fingers need no cursor and the GML
        // `opt_keyboard` branch draws nothing on touch either.
        let keyboard_mode = {
            let gamepad = self
                .sim
                .world
                .get_resource::<crate::savedata_part::SaveData>()
                .is_some_and(|s| s.settings.gamepad_enabled)
                && self.pad_live;
            !gamepad && self.touch_active.is_empty()
        };
        let overlay_now = self
            .sim
            .world
            .get_resource::<OverlayMenu>()
            .copied()
            .unwrap_or_default();
        let paused_now = self
            .sim
            .world
            .get_resource::<crate::state::Paused>()
            .is_some_and(|p| p.0);
        // GML `UberCont/Step_0:175-183` has no state carve-outs at
        // all: from the first splash frame, desktop keyboard mode
        // hides the OS cursor (`opt_keyboard` defaults true on
        // desktop and nothing in boot/menus/death clears it), and the
        // game crosshair (`UberCont/Draw_75`, gated only on
        // `window_get_cursor() == cr_none` + `show_crosshair`) draws
        // over everything — splash reel, menus, campfire, death
        // screen alike. Pause/overlay are the only hides (GML
        // `PauseImage` / `MenuOptions` take over the pointer there).
        let hide_os_cursor =
            keyboard_mode && !paused_now && overlay_now == OverlayMenu::None;
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
        // Hardware cursor request: computed inside the sprite borrow
        // (save reads only), refreshed after it ends (`refresh_cursor_img`
        // needs `&mut self`). `None` when assets are absent or the gate
        // is off — the refresh then clears the payload.
        let mut cursor_req: Option<(i32, [f32; 4])> = None;
        let (mut sprites, texts) = if self.assets.is_some() {
            let assets = self.assets.as_ref().expect("checked");
            // Cursor world position for the GML crosshair distance
            // (`KeyCont.dis_fire` parity; `None` until the first hover).
            // Uses the live cursor unprojection when available so the
            // crosshair tracks the on-screen cursor as the camera moves
            // (same source as aim; falls back to the last Hover).
            self.sim.world.insert_resource(crate::render::HoverWorld(
                self.live_cursor_world(),
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
            // GML `TopCont/Draw_0:43` device gate lives inside
            // `crosshair_sprites` (skip only for a keyboard-driven
            // local: keyboard mode on, gamepad mode off). No live-pad
            // half here: GML's `is_gamepad(index)` is the sticky
            // `opt_gamepad` setting (`scrHandleInputsGeneral`), not
            // per-frame pad activity — an idle-but-enabled pad still
            // draws the lerped crosshair.
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
            // Hardware cursor (GML `UberCont/Draw_75` verbatim): in
            // keyboard mode the OS cursor carries `sprCrosshair[opt_crosshair]`
            // (`opt_cursorcol`, alpha 1) — composited by the OS, so it sits
            // above every sprite and UI layer with zero frame lag, on EVERY
            // screen (gameplay included; the lerped `TopCont` crosshair is the
            // gamepad-mode counterpart and replaces this there — never
            // both, else a double cursor). GML gates only on
            // `window_get_cursor() == cr_none` + `scrCanDrawCursor()`
            // (`show_crosshair`, desktop/keyboard, no spawner); the
            // port's `keyboard_mode` (gamepad off, no touch) carries
            // the cursor-hidden half. No per-screen kind list:
            // the old 5-kind gate left keyboard gameplay cursorless.
            // Skipped while paused/an overlay owns the pointer (those
            // show the OS cursor) and on touch input (no cursor at all).
            // Keyboard-driven local (`opt_keyboard && !opt_gamepad`,
            // the same `keyboard_local` law as `crosshair_sprites`):
            // gamepad mode draws the lerped crosshair instead. The pixels
            // decode once per (frame, tint) into `cursor_img`; the runner
            // caches the OS handle by content hash.
            let keyboard_local = self
                .sim
                .world
                .get_resource::<crate::savedata_part::SaveData>()
                .map(|s| !s.settings.gamepad_enabled)
                .unwrap_or(true);
            let menu_crosshair =
                keyboard_mode && keyboard_local && !paused && overlay == OverlayMenu::None;
            cursor_req = if menu_crosshair {
                let frame = self
                    .sim
                    .world
                    .get_resource::<crate::savedata_part::SaveData>()
                    .map(|sv| sv.settings.crosshair as i32)
                    .unwrap_or(0);
                let tint = self
                    .sim
                    .world
                    .get_resource::<crate::savedata_part::SaveData>()
                    .map(|sv| sv.settings.cursorcol_rgba())
                    .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                Some((frame, tint))
            } else {
                None
            };
            // Sideart chrome around the view (GML `UberCont/Draw_74`:
            // GUI Begin — before the GUI-64 HUD/menus and the Draw_75
            // cursor, so it rungs above the room chrome but below every
            // HUD/menu/cursor sprite; never over a generation screen,
            // whose draw scripts own the full frame).
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
        // Hardware cursor refresh runs after the sprite borrow ends
        // (it needs `&mut self` for the `cursor_img` cache).
        match cursor_req {
            Some((frame, tint)) => self.refresh_cursor_img(frame, tint),
            None => {
                self.cursor_img = None;
                self.cursor_img_key = None;
            }
        }
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
        let mut vortex_layer = if self.assets.is_some()
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
                | Some(MenuOverlay::Unlock)
        );
        let overlay_color = crate::effects::flash_rgba(&self.sim.world);

        // GML `scrGameIsGenerationScreen` verbatim: no PlayerHUD text
        // while a generation screen owns the frame (`GenCont` behind
        // Loading, `LevCont` behind the offer, mid-run `FloorTransition`
        // covers). The canvas text layer rides ABOVE the opaque vortex
        // pass, so an ungated `hud_rows` paints HP/level/ammo/FLOOR
        // straight over the spiral (the cover-text leak). The tutorial
        // letterbox bar rides `menu_rows` instead (see below).
        let hud_rows = if state == AppState::InGame && menu_kind.is_none() && !cover_chrome_off {
            hud_overlay_lines(&mut self.sim.world, viewport_dp)
        } else {
            Vec::new()
        };
        let mut menu_rows =
            menu_kind.map(|k| menu_gui_texts_dp(k, &mut self.sim.world, viewport_dp));
        // GML `TutCont/Draw_64` verbatim: the step instruction bar draws
        // at the letterbox bottom until the exit portal exists — over
        // live HUD, never instead of it.
        if state == AppState::InGame
            && self
                .sim
                .world
                .get_resource::<crate::comps_a::Run>()
                .is_some_and(|r| r.tutorial)
            && !self
                .sim
                .world
                .get_resource::<crate::state::TutorialState>()
                .is_some_and(|t| t.portal_open)
        {
            let tut_rows = crate::render::tutorial_texts(&mut self.sim.world, viewport_dp);
            match &mut menu_rows {
                Some(rows) => rows.extend(tut_rows),
                None => menu_rows = Some(tut_rows),
            }
        }

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
                    PickEvent::Press {
                        world,
                        screen,
                        button,
                    } => {
                        app.pick_down(button);
                        if button == PointerButton::Primary {
                            app.stage_click(world, screen)
                        }
                    }
                    PickEvent::Click {
                        button, ..
                    } => {
                        app.pick_up(button);
                    }
                    PickEvent::Hover { world, screen } => app.stage_hover(world, screen),
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
                    PickEvent::Press {
                        world,
                        screen,
                        button,
                    } => {
                        app.pick_down(button);
                        if button == PointerButton::Primary {
                            app.stage_click(world, screen)
                        }
                    }
                    PickEvent::Click {
                        button, ..
                    } => {
                        app.pick_up(button);
                    }
                    PickEvent::Hover { world, screen } => app.stage_hover(world, screen),
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
        // shape as the rozvp pilot root). `on_focus_changed(false)` is
        // the focus-loss hook: winit delivers no key-ups across an
        // alt-tab, so held keys + edges drop here (GML's
        // `keyboard_check` reads all-up while unfocused).
        // Cursor: the scheduler override wins over hover (hover would
        // always report Default over the viewport and un-hide the
        // pointer). GML `UberCont/Step_0:175-183`: keyboard mode hides
        // the OS cursor, menus/mouse mode shows it. When the hardware
        // cursor art is live (`cursor_img`, same gate as the decode
        // above), the OS composites the crosshair itself — topmost,
        // zero lag, no sprite needed.
        sched.cursor_override = Some(if hide_os_cursor {
            match &self.cursor_img {
                Some(img) => repose_core::CursorIcon::Custom(img.clone()),
                None => repose_core::CursorIcon::Hidden,
            }
        } else {
            repose_core::CursorIcon::Default
        });
        let app_ptr = self as *mut App;
        nt_shortcuts::install(app_ptr);
        let key_ptr = self as *mut App;
        let focus = remember(FocusRequester::new);
        let fr_positioned = (*focus).clone();
        let focus_ptr = self as *mut App;
        let root_mod = Modifier::new()
            .fill_max_size()
            .focusable(true)
            .focus_requester((*focus).clone())
            .on_globally_positioned(move |_| {
                fr_positioned.request_focus();
            })
            .on_focus_changed(move |focused| {
                // SAFETY: synchronous compose-time dispatch only.
                let app = unsafe { &mut *focus_ptr };
                app.set_window_focused(focused);
            })
            .on_preview_key_event(move |ke: KeyEvent| {
                // Capture REMAP keys before the runtime's Space/Enter
                // keyboard-activation consumes them: when a rebind is
                // armed, the next pressed key resolves the capture and
                // never reaches gameplay or button activation. Preview
                // runs root-first, ahead of focus dispatch + shortcuts.
                // SAFETY: synchronous compose-time dispatch only.
                if matches!(ke.event_type, KeyEventType::Down) && !ke.is_repeat {
                    let app = unsafe { &mut *key_ptr };
                    if app.capture_armed() {
                        if let Some(key) = ke.physical {
                            app.capture_physical_press(key);
                        } else {
                            app.capture_key_press(&ke.key);
                        }
                        return true;
                    }
                }
                false
            })
            .on_key_event(move |ke: KeyEvent| {
                // SAFETY: synchronous compose-time dispatch only.
                // Esc/R/Enter also run through the scoped shortcut handler
                // (`nt_shortcuts::install`); returning false lets the
                // runtime resolve the shortcut after the bubble finishes.
                let app = unsafe { &mut *app_ptr };
                app.handle_key(&ke);
                false
            });
        // Right mouse button (GML `mb_right` ability / menu Back): the
        // viewport `PickEvent` button stages `rmb_down`/`lmb_down`
        // directly, so only the held latches (`rmb_held` for spec
        // hold, `lmb_held` for autofire) are repaired here from the
        // raw events. The cursor move stays (free moves dispatch only
        // to the topmost region, so the root handler alone misses
        // them — see `stage_hover`).
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
            });

        // HUD overlay (GML `scrDrawPlayerHUD` + `scrDrawMiscHUD`
        // verbatim): Silkscreen rows at 320x240 GUI positions over the
        // viewport.
        // Bottom-up: transparent drain passes sit ABOVE the sprite
        // viewport, opaque covers sit BELOW it. GML draws the spiral
        // inside the caller's draw event, i.e. over the room actors
        // behind it: `TopCont/Draw_0` redraws leftover `Spiral` wisps
        // at depth -15 (in front of `Floor` 8) during the live-game
        // drain, and `Menu/Draw_0` calls `scrDrawSpiral()` before its
        // own chrome. `bg_alpha` tells the two apart: 0 means a
        // transparent drain (Title remnant, live InGame drain) whose
        // wisps must paint over tiles/player/HUD bars; 1 means an
        // opaque cover (Splash logo, MainMenu, Loading, offers, floor
        // transition) whose chrome (logo, buttons, roadmap, cards)
        // lives in the viewport and must stay over the black.
        let mut layers = Vec::new();
        let vortex_above = vortex_layer.is_some() && self.last_bg_alpha < 0.5;
        if !vortex_above {
            if let Some(vortex) = vortex_layer.take() {
                layers.push(vortex);
            }
        }
        layers.push(viewport);
        if vortex_above {
            if let Some(vortex) = vortex_layer.take() {
                layers.push(vortex);
            }
        }
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
        // GML `Menu/Create_0:104-110` + `Menu/Step_1` verbatim: the
        // title view centers on the selected race's camper
        // (`Menu.char[race]`; Random centers on `char[0]`, the
        // Campfire at (64,64)) via `t_lerp` at 0.1 per step. The old
        // code parked the view top-left at (64,64) — half a screen
        // right/down of GML — leaving the camp left with background
        // filling the right. (The fixed-step camera only runs InGame,
        // and only InGame consumes `snap`, so menus step + place the
        // view camera directly here.)
        if matches!(state, AppState::Title) {
            let vw_vh = self.view_world_size;
            let focus =
                title_cam_focus(&mut self.sim.world).unwrap_or(Vec2::new(64.0, 64.0));
            let snap = !matches!(self.was_state, AppState::Title);
            title_camera_step(
                &mut self.gml_cam,
                vw_vh[0],
                vw_vh[1],
                focus,
                dt.as_secs_f32().clamp(0.0, 0.1),
                snap,
            );
            self.cam = world_camera(
                Vec2::new(
                    self.gml_cam.x + vw_vh[0] * 0.5,
                    self.gml_cam.y + vw_vh[1] * 0.5,
                ),
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
            // near-opaque bevy `scrim` (230/255). The Draw_75 cursor
            // draws after, so it stays full-bright over the dim.
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
    // Editable controls: default map, then overlay the save file's
    // rows when present (GML `scrOptionsLoadKeymaps` on boot). The
    // disk shell calls `App::load_save` after boot; headless/tests
    // keep GML defaults.
    world.init_resource::<crate::keymap::InputMapState>();
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
/// Takes PHYSICAL px (`Scheduler.size`).
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
            out.push(quad(pos.0, 16.0, [0.35, 0.33, 0.38, 1.0]));
        }
    }
    // Assetless floors: the GPU path draws lit strips over mask cells
    // (`world_instances`); without this the placeholder viewport is
    // walls floating on the flat room colour.
    if let Some(mask) = world.get_resource::<crate::comps_a::FloorMask>() {
        for cell in &mask.cells {
            out.push(quad(
                Vec2::new(
                    cell.0 as f32 * crate::comps_a::TILE + crate::comps_a::TILE * 0.5,
                    cell.1 as f32 * crate::comps_a::TILE + crate::comps_a::TILE * 0.5,
                ),
                32.0,
                [0.55, 0.45, 0.32, 1.0],
            ));
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
        .filter(|(segs, _, _, _, _, _, _)| {
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
    // Bigname rows (GML `draw_text_bigname`): fill+stroke faux-bold
    // approximates the heavy fntBig glyphs with Silkscreen.
    if row.6 {
        text = text.fill_and_stroke(0.04);
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
        // GML `UnlockScreen/Mouse_56` verbatim: the panel dismisses on
        // click once `can_continue` fires (`Alarm_1`, 20 steps after
        // show). Headless has no show timer, so any CONTINUE click
        // dismisses the head of the queue.
        MenuOverlay::Unlock => match text {
            "CONTINUE" => Some(UiAction::DismissUnlock),
            _ => None,
        },
        _ => None,
    }
}

/// Menu-owned keys (Godot `_gui_input`-before-`_unhandled_input`
/// parity): while a menu owns the screen, Space/arrows/Tab drive nav
/// and confirm only — `feed_input` strips them before the gameplay
/// sampler so one press can't both confirm a row and pulse a gameplay
/// action. Gameplay keeps them via `held`/`just` on `live_play`.
fn menu_owned_key(code: KeyCode) -> bool {
    matches!(
        code,
        KeyCode::Space
            | KeyCode::Tab
            | KeyCode::ArrowUp
            | KeyCode::ArrowDown
            | KeyCode::ArrowLeft
            | KeyCode::ArrowRight
    )
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
    /// GML `UnlockScreen` panel over live gameplay: the head of
    /// `MenuState::unlock_queue` (race/skin unlocks queue here; the
    /// shell dismisses via `dismiss_unlock` once confirmed).
    Unlock,
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
            // GML `UnlockScreen` panels surface over gameplay (TopCont
            // draws the queued head while a run is live); they take
            // precedence over pause/settings/credits/mutation so the
            // unlock is seen before any other overlay.
            if !menu.unlock_queue.is_empty() {
                return Some(MenuOverlay::Unlock);
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


#[cfg(test)]
mod cursor_staging_tests {
    use super::*;

    /// Reported bug verbatim: after any click-drag, free mouse moves
    /// stopped moving the crosshair — it only followed while dragging.
    /// Root cause: repose dispatches free moves ONLY to the topmost
    /// region, so the root's `cursor_move` (the `cursor_px` source)
    /// runs almost exclusively on capture-path (button-held) moves,
    /// while viewport `Hover` fires on free moves. The viewport hover
    /// px now feeds the same live unprojection, so a stale drag-era
    /// root px can never shadow it. The px sources win only while fresh
    /// (re-unprojected every frame, so aim stays glued to the pointer
    /// instead of sliding on the ground); once stale the baked hover
    /// point takes over.
    #[test]
    fn fresh_px_wins_stale_px_yields_to_hover() {
        let mut app = App::new_with_seed(4242);
        // Hover stages the baked point AND its raw px; the drag-era
        // root px is fresh too. Unprojection through the live camera
        // must not return the stale baked point.
        app.stage_hover(Vec2::new(10.0, 10.0), [0.0, 0.0]);
        app.cursor_move(Vec2::new(100.0, 100.0));
        let live = app.live_cursor_world().expect("cursor staged");
        assert!(
            (live - Vec2::new(10.0, 10.0)).length() > 1.0,
            "fresh px must unproject through the live camera, got {live:?}"
        );
        // Root px goes stale (no root moves for a while, e.g. free
        // mouse play): the viewport hover px stays fresh and keeps
        // owning aim — the cursor never freezes on an old camera.
        for _ in 0..40 {
            app.stage_hover(Vec2::new(10.0, 10.0), [50.0, 50.0]);
        }
        let live = app.live_cursor_world().expect("hover staged");
        assert!(
            (live - Vec2::new(10.0, 10.0)).length() > 1.0,
            "fresh hover px must keep unprojecting, got {live:?}"
        );
    }

    /// Reported bug verbatim: the crosshair drifted across the screen
    /// as the camera moved even with the pointer still — the world
    /// point was baked through an old camera and the hover screen px
    /// was dropped (`_screen`). GML draws the raw cursor at GUI mouse
    /// coords (screen-anchored, never camera-following). Regression:
    /// same screen px under a moved camera must unproject to the new
    /// camera-relative point (i.e. track the camera 1:1), not stick to
    /// the old world point.
    #[test]
    fn hover_px_tracks_moved_camera() {
        let mut app = App::new_with_seed(4242);
        app.stage_hover(Vec2::new(10.0, 10.0), [640.0, 360.0]);
        let before = app.live_cursor_world().expect("cursor staged");
        // Pan the camera 100 world units right; the pointer hasn't
        // moved (same screen px, fresh hover).
        app.cam.center += Vec2::new(100.0, 0.0);
        app.stage_hover(Vec2::new(10.0, 10.0), [640.0, 360.0]);
        let after = app.live_cursor_world().expect("cursor staged");
        let shift = (after - before).length();
        assert!(
            (shift - 100.0).abs() < 1.0,
            "screen-anchored cursor must track the camera 1:1, shifted {shift} for a 100-unit pan (before {before:?}, after {after:?})"
        );
    }

    #[test]
    fn no_staging_yet_is_none() {
        let app = App::new_with_seed(4242);
        assert_eq!(app.live_cursor_world(), None);
    }

    #[test]
    fn global_shortcut_map_binds_pause_restart_confirm() {
        use repose_core::input::{Key, Modifiers};
        use repose_core::shortcuts::{Action, KeyChord, resolve_action};
        let map = nt_shortcuts::map();
        for (key, want) in [
            (Key::Escape, nt_shortcuts::PAUSE),
            (Key::Character('r'), nt_shortcuts::RESTART),
            (Key::Enter, nt_shortcuts::CONFIRM),
        ] {
            let chord = KeyChord::new(key.clone(), Modifiers::default());
            assert_eq!(
                map.action_for(&chord),
                Some(Action::Custom(want.into())),
                "map must bind {want}"
            );
            // Installing wires the map into the global scope stack, so the
            // runtime's `resolve_action` (the `dispatch_action` path) sees
            // the chord even with no compose scope mounted.
            let mut app = App::new_with_seed(4242);
            let ptr = &mut app as *mut App;
            nt_shortcuts::install(ptr);
            assert_eq!(
                resolve_action(KeyChord::new(key, Modifiers::default())),
                Some(Action::Custom(want.into())),
                "installed map must resolve {want}"
            );
        }
    }

    #[test]
    fn installed_shortcut_handler_stages_edges() {
        use repose_core::shortcuts::{Action, handle};
        let mut app = App::new_with_seed(4242);
        let ptr = &mut app as *mut App;
        nt_shortcuts::install(ptr);
        assert!(handle(Action::Custom(nt_shortcuts::PAUSE.into())));
        assert!(app.pause_edge);
        assert!(handle(Action::Custom(nt_shortcuts::RESTART.into())));
        assert!(app.restart_edge);
        assert!(handle(Action::Custom(nt_shortcuts::CONFIRM.into())));
        assert!(app.interact_edge);
        assert!(!handle(Action::Custom("nt.unknown".into())));
    }

    /// E-key interact regression: the KeyE edge staged through
    /// `handle_key` must surface as `interact_pressed` after the
    /// gameplay sampler runs. Catches the sampler reading a
    /// rebound/missing Pick row.
    #[test]
    fn key_e_stages_interact_pulse() {
        use crate::input::{KeyCode, MouseState, NtInput};
        let mut app = App::new_with_seed(4242);
        app.stage_physical_key(PhysicalKey::KeyE, true);
        let just: std::collections::HashSet<KeyCode> = app.edges.drain(..).collect();
        assert!(just.contains(&KeyCode::KeyE), "KeyE must stage an edge");
        let keymap = app.sim.world.resource::<InputMapState>().clone();
        let mut out = NtInput::default();
        crate::input::sample_keyboard_mapped(
            &app.held,
            &just,
            &MouseState::default(),
            Some(&keymap),
            &mut out,
        );
        assert!(
            out.peek_interact_pressed(),
            "KeyE edge must raise interact_pressed"
        );
    }

    /// Reported bug verbatim: on the REMAP page, pressing most keys
    /// did nothing — the physical capture table only mapped ~20 names
    /// (WASD/arrows/digits/space/tab) and silently swallowed the rest.
    /// GML captures any pressed input, so the table now derives every
    /// `KeyX`/`DigitN` name generically.
    #[test]
    fn remap_capture_accepts_any_key() {
        use repose_core::input::{Key, KeyEvent, KeyEventType, Modifiers, PhysicalKey};
        for key in [
            PhysicalKey::KeyX,
            PhysicalKey::KeyC,
            PhysicalKey::KeyV,
            PhysicalKey::KeyZ,
            PhysicalKey::KeyH,
            PhysicalKey::KeyM,
            PhysicalKey::Digit6,
            PhysicalKey::ArrowUp,
        ] {
            let mut app = App::new_with_seed(4242);
            crate::state::menus::apply_menu_action(
                &mut app.sim.world,
                crate::audio::UiAction::RemapControl("north".to_string()),
            );
            assert!(app.capture_armed(), "capture must arm for {key:?}");
            // Live shell path: focus-routed KeyEvent with physical key.
            app.handle_key(&KeyEvent {
                key: Key::Character('x'),
                modifiers: Modifiers::default(),
                is_repeat: false,
                event_type: KeyEventType::Down,
                utf16_code_point: 0,
                physical: Some(key),
            });
            assert!(
                !app.capture_armed(),
                "pressing {key:?} must resolve the capture"
            );
        }
    }

    /// REMAP capture resolves from the focus-routed `handle_key` path;
    /// Esc/Enter/R stage only via the shortcut handler.
    #[test]
    fn focus_key_resolves_capture_and_pause() {
        use repose_core::input::{Key, KeyEvent, KeyEventType, Modifiers, PhysicalKey};
        use repose_core::shortcuts::{Action, handle};
        let mut app = App::new_with_seed(4242);
        let ptr = &mut app as *mut App;
        nt_shortcuts::install(ptr);
        app.handle_key(&KeyEvent {
            key: Key::Escape,
            modifiers: Modifiers::default(),
            is_repeat: false,
            event_type: KeyEventType::Down,
            utf16_code_point: 0,
            physical: None,
        });
        assert!(
            !app.pause_edge,
            "Escape KeyEvent alone must not stage pause_edge"
        );
        assert!(handle(Action::Custom(nt_shortcuts::PAUSE.into())));
        assert!(app.pause_edge, "shortcut handler must stage pause_edge");
        crate::state::menus::apply_menu_action(
            &mut app.sim.world,
            crate::audio::UiAction::RemapControl("north".to_string()),
        );
        assert!(app.capture_armed());
        app.handle_key(&KeyEvent {
            key: Key::Character('x'),
            modifiers: Modifiers::default(),
            is_repeat: false,
            event_type: KeyEventType::Down,
            utf16_code_point: 0,
            physical: Some(PhysicalKey::KeyX),
        });
        assert!(!app.capture_armed(), "KeyX must resolve the capture");
        let entry = app
            .sim
            .world
            .resource::<InputMapState>()
            .map
            .keyboard(&crate::keymap::NtAction::North);
        assert!(
            matches!(entry, repame_input::KeymapEntry::Key(_)),
            "capture must rebind north, got {entry:?}"
        );
    }

    /// Full rebound loop: rebind north to Z through a capture, persist +
    /// reload through save rows (GML `scrOptionsSaveKeymaps`/
    /// `scrOptionsLoadKeymaps`), then press KeyZ and confirm the rebound
    /// key steers movement. Catches encode/decode drift and sampler
    /// mismatch in one pass.
    #[test]
    fn rebound_key_drives_gameplay_after_save_round_trip() {
        use repose_core::input::{Key, KeyEvent, KeyEventType, Modifiers, PhysicalKey};
        let press = |key: Key, physical: PhysicalKey| KeyEvent {
            key,
            modifiers: Modifiers::default(),
            is_repeat: false,
            event_type: KeyEventType::Down,
            utf16_code_point: 0,
            physical: Some(physical),
        };
        let mut app = App::new_with_seed(4242);
        crate::state::menus::apply_menu_action(
            &mut app.sim.world,
            crate::audio::UiAction::RemapControl("north".to_string()),
        );
        app.handle_key(&press(Key::Character('z'), PhysicalKey::KeyZ));
        assert!(!app.capture_armed());
        // Persist + reload like a reboot.
        let rows = app
            .sim
            .world
            .resource::<InputMapState>()
            .map
            .clone();
        let saved = crate::keymap::KeyBindings::from_keymap(&rows);
        let back = saved.to_keymap();
        app.sim.world.resource_mut::<InputMapState>().map = back;
        // Release Z, then press it fresh and sample movement.
        app.stage_physical_key(PhysicalKey::KeyZ, false);
        app.stage_physical_key(PhysicalKey::KeyZ, true);
        let mut out = crate::input::NtInput::default();
        let just: std::collections::HashSet<crate::input::KeyCode> =
            app.edges.drain(..).collect();
        let keymap = app.sim.world.resource::<InputMapState>().clone();
        let mouse = crate::input::MouseState::default();
        crate::input::sample_keyboard_mapped(&app.held, &just, &mouse, Some(&keymap), &mut out);
        assert!(
            out.move_axis.y < -0.5,
            "rebound Z must steer north, got {:?}",
            out.move_axis
        );
    }

    /// Sharing law: one landing must push one edge per action bound to
    /// it, so every rebound action sharing Space fires (fire AND walk
    /// AND swap from one Space press). Regression: `stage_physical`
    /// suppressed Space edges as "owned elsewhere", which starved all
    /// but one sharing action of its edge.
    #[test]
    fn shared_space_fires_every_bound_action() {
        use crate::input::{KeyCode, MouseState, NtInput};
        use crate::keymap::NtAction;
        use repame_input::{Keymap, KeymapDevice, KeymapEntry};
        use repose_core::input::{Key, Modifiers};
        use repose_core::shortcuts::KeyChord;
        use std::collections::HashSet;
        let space = || KeymapEntry::Key(KeyChord::new(Key::Space, Modifiers::default()));
        let mut map = Keymap::new();
        map.set_keyboard(NtAction::Fire, space());
        map.set_keyboard(NtAction::North, space());
        map.set_keyboard(NtAction::Swap, space());
        let state = InputMapState {
            map,
            capture: None,
        };
        let held: HashSet<KeyCode> = [KeyCode::Space].into_iter().collect();
        let just: HashSet<KeyCode> = [KeyCode::Space].into_iter().collect();
        let mut out = NtInput::default();
        crate::input::sample_keyboard_mapped(
            &held,
            &just,
            &MouseState::default(),
            Some(&state),
            &mut out,
        );
        assert!(out.fire_held, "shared Space must hold fire");
        assert!(
            out.take_fire_pressed(),
            "shared Space must pulse fire_pressed"
        );
        assert!(
            out.move_axis.y < -0.5,
            "shared Space must steer north, got {:?}",
            out.move_axis
        );
        assert!(
            out.take_cycle_weapon() == 1,
            "shared Space must pulse swap cycle"
        );
        let _ = KeymapDevice::KeyboardMouse;
    }

    /// Context switch: Space on a menu screen must not pulse gameplay
    /// actions. One Space press over Title toggles the loadout path
    /// exactly once with zero `fire_pressed`.
    #[test]
    fn menu_space_never_pulses_gameplay_fire() {
        use crate::input::{KeyCode, MouseState, NtInput};
        use crate::keymap::{InputMapState, NtAction};
        use repame_input::{Keymap, KeymapEntry};
        use repose_core::input::{Key, Modifiers};
        use repose_core::shortcuts::KeyChord;
        use std::collections::HashSet;
        let space = || KeymapEntry::Key(KeyChord::new(Key::Space, Modifiers::default()));
        let mut map = Keymap::new();
        map.set_keyboard(NtAction::Fire, space());
        map.set_keyboard(NtAction::North, space());
        let state = InputMapState {
            map,
            capture: None,
        };
        let held: HashSet<KeyCode> = [KeyCode::Space].into_iter().collect();
        let just: HashSet<KeyCode> = [KeyCode::Space].into_iter().collect();
        let strip = |c: KeyCode| !menu_owned_key(c);
        let held: HashSet<KeyCode> = held.into_iter().filter(|c| strip(*c)).collect();
        let just: HashSet<KeyCode> = just.into_iter().filter(|c| strip(*c)).collect();
        let mut out = NtInput::default();
        crate::input::sample_keyboard_mapped(
            &held,
            &just,
            &MouseState::default(),
            Some(&state),
            &mut out,
        );
        assert!(
            !out.take_fire_pressed(),
            "menu-owned Space must not pulse fire"
        );
        assert!(
            !out.fire_held,
            "menu-owned Space must not hold fire"
        );
    }

    /// Context switch: arrows via `handle_key` (focus-routed) must still
    /// drive menu nav on MainMenu and on an open Settings overlay.
    #[test]
    fn focus_arrows_drive_menu_nav() {
        use repose_core::input::{Key, KeyEvent, KeyEventType, Modifiers, PhysicalKey};
        for (state, overlay) in [
            (AppState::MainMenu, OverlayMenu::None),
            (AppState::InGame, OverlayMenu::Settings),
        ] {
            let mut app = App::new_with_seed(4242);
            app.sim.world.insert_resource(state);
            app.sim.world.insert_resource(overlay);
            app.sim.world.init_resource::<NtInput>();
            app.handle_key(&KeyEvent {
                key: Key::ArrowDown,
                modifiers: Modifiers::default(),
                is_repeat: false,
                event_type: KeyEventType::Down,
                utf16_code_point: 0,
                physical: Some(PhysicalKey::ArrowDown),
            });
            app.feed_input();
            let (dv, dh) = app.sim.world.resource_mut::<NtInput>().take_menu_nav();
            assert_eq!(
                (dv, dh),
                (1, 0),
                "ArrowDown must step menu nav on {state:?}/{overlay:?}"
            );
        }
    }
    /// Live click-to-arm chain: a click on the REMAP page's first row
    /// (through `route_menu_click` → `settings_click_action` → hot
    /// row → `RemapControl` → `begin_capture`) must arm the capture —
    /// the link between the drawn rows and the key handler above. The
    /// click dp comes from the row's own GUI box (as the viewport
    /// stages it), so this also proves the drawn rows and the hot
    /// rows agree.
    #[test]
    fn remap_row_click_arms_capture() {
        let mut app = App::new_with_seed(4242);
        app.sim.world.insert_resource(AppState::MainMenu);
        app.sim.world.insert_resource(OverlayMenu::Settings);
        app.sim
            .world
            .resource_mut::<MenuState>()
            .settings_page = 13;
        let rows = crate::render::settings_hot_rows(13, 320.0);
        let row = rows[0];
        // `route_menu_click` takes canvas dp + viewport dp: GUI px
        // scale with k = h/240 (720p → 3.0).
        let k = 3.0f32;
        let viewport_dp = [1280.0f32, 720.0];
        let dp = [row.cx * k, row.gy * k];
        let action = route_menu_click(
            &mut app.sim.world,
            MenuOverlay::Settings,
            dp,
            viewport_dp,
        );
        assert!(
            matches!(action, Some(crate::audio::UiAction::RemapControl(_))),
            "click on row 0 must arm a remap, got {action:?}"
        );
        crate::state::menus::apply_menu_action(&mut app.sim.world, action.unwrap());
        assert!(app.capture_armed(), "capture must arm after the row click");
    }
}
