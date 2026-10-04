//! nt on repame: 30 Hz fixed-step sim + repose views, no bevy.
//!
//! [`App::view`] (from [`root_view`]) drains staged shell input into
//! [`NtInput`], advances the sim at [`SIM_HZ`] through the full sim
//! [`Schedule`], then composes: sprite viewport ([`Viewport2dGpu`] with
//! assets, canvas [`Viewport2d`] placeholder without), HUD [`Text`] from
//! [`hud_gui_texts_dp`](crate::render::hud_gui_texts_dp) (bevy
//! `nt_hud_overlay` verbatim), menus from
//! [`menu_gui_texts_dp`](crate::render::menu_gui_texts_dp).
//!
//! Assets ([`resolve_assets_dir`], in order): 1. `$NT_ASSETS` (must
//! contain `images/anims.ron`), 2. `<exe-dir>/assets`, 3. `<cwd>/assets`
//! (so `cargo run` from the crate dir works). Loading is explicit
//! ([`App::load_assets`], called by `main.rs`), so [`App::new`] never
//! touches disk; assetless still runs (placeholder quads, no system
//! depends on art).
//!
//! Input: keys via `Modifier::on_key_event` on the focused root, clicks
//! via the viewport `on_event` [`PickEvent`] - whose `Press`/`Click`
//! carry their [`PointerButton`], so no parallel root button handlers -
//! staged through `feed_input`.
//! - WASD/arrows -> move, mouse cursor -> aim (every frame), Space /
//!   left-click -> fire, Shift / right-click -> ability+spec (GML `spec`
//!   on `mb_right`; Shift arrives as its own `Key` variants now),
//!   E/F/Q/G/Tab/Enter -> interact/confirm, 1-5 -> weapon slots /
//!   mutation picks / title cursor / main-menu+pause rows, Left/Right ->
//!   title cursor + mutation highlight, Up/Down (+Left/Right) ->
//!   main-menu cursor + settings cursor/values, click in game ->
//!   aim-at-click + fire, click in menus -> positional buttons
//!   (main-menu rows, title pods/GO/loadout, pause/settings/credits/
//!   stats via screen-position routing; strays dropped), right-click in
//!   settings/credits -> Back (GML `BackButton` `mb_right`), Space on
//!   title -> loadout panel, Esc -> pause toggle (+ overlay unwind +
//!   main-menu overlay close), R -> game-over retry.
//! - NOT wired: text entry for profile/color inputs (buttons only; color
//!   cycles presets). Rebinding works through the REMAP capture; gamepad
//!   sticks/triggers drain through `stage_gamepad`.
//!
//! Fidelity: view-layer camera smoothing never feeds back into the sim;
//! [`SpiralCtl`] steps at the fixed cadence inside [`App::advance`]
//! (sim-pure); the sim schedule in `schedule.rs` is unchanged.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use web_time::Duration;

use crate::vortex_pass::{VARD_VARIANTS, VortexPass, VortexTexture};
use bevy_ecs::prelude::*;
use glam::Vec2;
use rand::RngExt;
use repame_anim::{AnimCatalog, AtlasDesc};
use repame_sim::{Sim, SimTime};
use repame_sprite::{
    BatchDesc, Camera2d, FrameInput, GeomHandle, PickEvent, SpriteBatch, SpriteInstance,
    Viewport2dGpuWithHudShared, Viewport2dGpuWithIdShared, Viewport2dShared,
};
use repose_canvas::Embedded;
use repose_core::PaddingValues;
use repose_core::input::{Key, KeyEvent, KeyEventType, PhysicalKey, PointerButton};
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
use crate::input::{GamepadState, KeyCode, MouseState, NtInput, sample_gamepads_mapped};
use crate::keymap::{InputMapState, KeyBindings};
use crate::render::{
    ATLAS_PAGES, ATLAS_SIZE, CamPoi, CamStepInput, GmlCamera, RenderAssets, SettingSliderTarget,
    StaticWorldCache, Z_BLOOM, Z_CROSSHAIR, Z_FAINTED, Z_FOG, Z_FX, Z_HUD, Z_MENU,
    Z_PORTAL_INDICATOR, Z_SHADOW, Z_SIDEART, Z_SPIRAL_FIGURES, Z_SPLASH, Z_TOUCH, background_color,
    bloom_sprites, cam_viewdist_for, crosshair_sprites, decode_png, fainted_bar_sprites,
    fog_sprites, fx_instances, fx_texts, gml_camera_step, gml_view_size, hud_gui_texts_dp,
    hud_sprites, letterbox_sprites, menu_gui_texts, menu_gui_texts_dp, menu_gui_texts_vw,
    menu_sprites, pause_button_sprites, portal_indicator_sprites, settings_slider_hit,
    settings_slider_value, shadow_sprites, sideart_sprites, spiral_figures, splash_sprites,
    srgb_to_linear, stamp_z, title_cam_focus, title_camera_step, touch_sprites, view_rect_world,
    world_camera, world_instances_cached,
};
use crate::schedule::build_sim_schedule;
use crate::setup::setup_run_with_seed;
use crate::spatial::Pos;
use crate::state::menus::{MenuEdge, MenuState, apply_menu_action, emit_sfx};
use crate::state::{AppState, OverlayMenu};
use crate::vortex::{SpiralCtl, gml_area_for_area};

pub mod anim;
pub mod assetfs;
pub mod audio;
pub mod audio_host;
pub mod boss_ai;
pub mod combat;
pub mod comps_a;
pub mod comps_b;
pub mod crown;
pub mod data;
mod dead_path_part;
pub mod deaths;
pub mod decide_wep;
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
#[cfg(target_arch = "wasm32")]
pub mod web;
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

/// GML `GameCont` crib-trip room-start flags: `gocrib` raised by a Venuz/Cuz
/// ultra pick (`UltraIcon/Other_10.gml:12-13`), consumed by the next Room
/// Start (`GameCont/Other_5.gml:30-31`); `can_advance` = the
/// `can_advance_stage` latch leaving the floor counter alone on that same
/// Room Start (`GameCont/Other_5.gml:49-52`); `fromcrib` = room being left
/// is the crib (`GameCont/Other_5.gml:24,58-62`). `GenCont/Create_0.gml:48`
/// reads `gocrib` to suppress the Patience mutation offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Resource)]
pub struct CribTrip {
    pub gocrib: bool,
    pub fromcrib: bool,
    pub can_advance: bool,
    /// GML `GameCont.lastarea` / `lastsubarea`, written when leaving for
    /// the crib (`GameCont/Other_5.gml:33-34`) and read back on the way
    /// out (`:58-61`).
    pub last_area: Option<crate::data::AreaId>,
    pub last_subarea: u32,
}

impl Default for CribTrip {
    /// GML `GameCont/Create_0.gml:57,89-90`.
    fn default() -> Self {
        Self {
            gocrib: false,
            fromcrib: false,
            can_advance: true,
            last_area: None,
            last_subarea: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SettingsSliderPointer {
    Mouse,
    Touch(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SettingsSliderDrag {
    page: u8,
    target: SettingSliderTarget,
    pointer: SettingsSliderPointer,
}

/// The game: fixed-step sim plus presentational (view-layer) state.
///
/// sim-pure: `sim`/`schedule`/`spiral`; view-layer: `cam`,
/// `held`/`edges`/`clicks`/`hover`, asset catalog. [`App::advance`] runs
/// the sim; [`App::view`] stages input, advances, composes.
pub struct App {
    pub sim: Sim,
    schedule: Schedule,
    accum: Duration,
    /// Disk path the save was loaded from (`load_save` records it;
    /// `save_now` flushes here on every dirty write, GML `scrSave`
    /// parity - without it a restart shows the old mappings).
    save_path: Option<PathBuf>,
    /// Smoothed follow camera (VIEW layer only - sim stays fixed-step pure).
    pub cam: Camera2d,
    /// GML `BackCont` camera state verbatim (view-layer state).
    gml_cam: GmlCamera,
    /// Previous frame's floor-transition flag (generation end snaps).
    was_transitioning: bool,
    /// Previous frame's app state (entering InGame snaps the camera on
    /// the fresh player instead of swooping from the menu look point).
    was_state: AppState,
    last_splash_mode: u8,
    letterbox_frame: f32,
    /// Previous fixed step's app state (drives the spiral lifecycle in
    /// [`App::advance`]: Loading entry re-warms, InGame entry kills -
    /// separate from the view-side `was_state`, which updates a frame
    /// later in [`App::view`]).
    adv_state: AppState,
    /// Previous fixed step's generation-cover flag (floor transition or
    /// mutation/ultra offer). Rising edge re-warms the spiral
    /// (`GenCont`/`LevCont` build a fresh `SpiralCont`); falling edge
    /// kills it (`GenCont/Destroy` destroys it at generation end).
    adv_cover: bool,
    cover_rewarm_pending: bool,
    adv_throne_ii: bool,
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
    static_world_cache: StaticWorldCache,
    /// Art dir the catalog loaded from (vortex background textures
    /// decode from here on area switches).
    assets_dir: Option<PathBuf>,
    /// Decoded hardware cursor (`CursorIcon::Custom` payload) for the
    /// live `opt_crosshair` frame + `opt_cursorcol` tint. Cached so the
    /// runner's `create_custom_cursor` handle rebuilds only when the
    /// art changes (frame/tint/pixels/scale); cleared when assets unload.
    cursor_img: Option<std::sync::Arc<repose_core::CustomCursorImage>>,
    cursor_img_key: Option<(i32, [u32; 3], u64, u32)>,
    last_live_frame: Option<Arc<FrameInput>>,
    /// Decoded vortex background textures for `vortex_tex_area`
    /// (slots: spiral, bolt, debris, proto, idpd, idpd2, star, variants).
    vortex_tex: Vec<VortexTexture>,
    vortex_tex_area: Option<u8>,
    vortex_tex_gen: u64,
    // staged shell input (drained into `NtInput`/`MenuEdge` per frame):
    // staging owns every level - `staging.held` (layout-independent
    // physical positions, GML `keyboard_check` parity),
    // `staging.lmb_held/rmb_held`, `staging.edges`; `feed_polled` repairs
    // in, `feed_input` samples out (one store both ways).
    /// Whether the window currently has focus. `false` (set by the shell's
    /// focus handler) drops `held` outright - winit delivers no key-ups
    /// across an alt-tab.
    window_focused: bool,
    staging: std::rc::Rc<std::cell::RefCell<repame_shell::Staging>>,
    shortcut_edges: repame_shell::SharedEdges,
    /// Last mouse position in window-physical px, copied from
    /// `Scheduler::pointer_pos_px` at the top of every `view` (the runtime
    /// maintains it on move/press/release, bypassing focus dispatch).
    /// Drives [`App::live_cursor_world`]; `None` until the first mouse
    /// move. Stored: `feed_input` and the fixed-step camera run without
    /// the `Scheduler` at hand.
    polled_pointer_px: Option<Vec2>,
    pause_edge: bool,
    restart_edge: bool,
    interact_edge: bool,
    /// Latched gamepad presence: set while any staged snapshot shows a
    /// connected pad, cleared when a frame stages none. `pads` drains
    /// every frame so it can't answer "is a pad in use" outside
    /// `feed_input`; this carries that fact to the cursor gate.
    pad_live: bool,
    /// Live touch finger ids last frame: viewport lifts bypass
    /// `App::touch_up`, so ids missing this frame latch the release
    /// edge in `feed_input`.
    last_touch_ids: Vec<i64>,
    touch_menu_positions: Vec<(i64, Vec2)>,
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
    settings_slider_drag: Option<SettingsSliderDrag>,
    last_sprite_count: usize,
    last_touch_count: usize,
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

        let area = if setup {
            sim.world
                .get_resource::<crate::comps_a::Run>()
                .map(|r| r.area)
                .unwrap_or(AreaId::Desert)
        } else {
            AreaId::Campfire
        };
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
            last_splash_mode: 0,
            letterbox_frame: 0.0,
            adv_state: AppState::default(),
            adv_cover: false,
            cover_rewarm_pending: false,
            adv_throne_ii: false,
            adv_seed: 0,
            adv_area: 0,
            spiral,
            assets: None,
            static_world_cache: StaticWorldCache::default(),
            assets_dir: None,
            cursor_img: None,
            cursor_img_key: None,
            last_live_frame: None,
            vortex_tex: Vec::new(),
            vortex_tex_area: None,
            vortex_tex_gen: 0,
            window_focused: true,
            staging: repame_shell::Staging::shared(),
            shortcut_edges: repame_shell::shared_edges(),
            polled_pointer_px: None,
            pause_edge: false,
            restart_edge: false,
            interact_edge: false,
            pad_live: false,
            last_touch_ids: Vec::new(),
            touch_menu_positions: Vec::new(),
            view_width: 1280.0,
            view_viewport_dp: [1280.0, 720.0],
            view_density: 1.0,
            view_world_size: [426.0, 240.0],
            menu_actions: Vec::new(),
            settings_slider_drag: None,
            last_sprite_count: 0,
            last_touch_count: 0,
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
    /// `dir` is a prefix only: on Android every file resolves through
    /// the APK `assets/` table by its `images/…` tail
    /// (`read_asset_bytes`), so `<files>/assets` works without copying
    /// a single PNG to internal storage.
    pub fn load_assets_from(&mut self, dir: &Path) -> anyhow::Result<()> {
        let assets = RenderAssets::load(dir)?;
        let text = crate::render::read_asset_catalog(dir)?;
        let catalog = AnimCatalog::from_ron(
            &text,
            AtlasDesc {
                size: ATLAS_SIZE,
                max_pages: ATLAS_PAGES,
                padding: 0,
            },
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Pre-asset spawns (headless `setup_run` runs against the empty
        // catalog) carry no `SpriteAnim` and, for enemies, no
        // `EnemySprites` table - backfill both so `player_anim_switch` /
        // `enemy_anim_switch` / `hurt_on_damage` match post-asset spawns
        // (bevy parity: every actor spawns with its strip table + seeded
        // idle anim).
        self.sim.world.insert_resource(catalog);
        crate::anim::backfill_spawn_anims(&mut self.sim.world);
        self.assets = Some(assets);
        self.static_world_cache.clear();
        self.assets_dir = Some(dir.to_path_buf());
        self.cursor_img = None;
        self.cursor_img_key = None;
        self.last_live_frame = None;
        self.vortex_tex.clear();
        self.vortex_tex_area = None;
        self.vortex_tex_gen = self.vortex_tex_gen.wrapping_add(1).max(1);
        Ok(())
    }

    /// Vortex background textures for a GML area (headless-observable;
    /// same decode the live frame uses on area switches).
    pub fn debug_vortex_textures(dir: &Path, gml_area: u8) -> Vec<VortexTexture> {
        Self::load_vortex_textures(dir, gml_area, 1)
    }

    /// Vortex background textures for a GML area (decoded once per
    /// area; slots match the shader bindings: spiral, bolt, debris,
    /// proto, idpd, idpd2, star, and seven variant debris strips).
    /// Missing files fall back to area 0 debris so the pass always has
    /// fourteen bound slots.
    fn load_vortex_textures(dir: &Path, gml_area: u8, generation: u64) -> Vec<VortexTexture> {
        fn decode(dir: &Path, name: &str) -> Option<(u32, u32, Vec<u8>)> {
            decode_png(&dir.join("images").join(format!("{name}.png"))).ok()
        }
        // GML `SpiralDebris/Create_0` verbatim: the mote strip is
        // `sprDebris + GameCont.area` - the FULL GML area number, not the
        // art variant. Campfire (area 0) reuses `sprDebris0`; secret areas
        // use their own (`sprDebris101` …). Never `sprDebris1` here.
        let debris = decode(dir, &format!("sprDebris{gml_area}"))
            .or_else(|| decode(dir, "sprDebris0"))
            .unwrap_or((1u32, 1u32, vec![255, 255, 255, 255]));
        let mut out = Vec::new();
        let stems = [
            "sprSpiral",
            "sprPortalLightning",
            "",
            "sprSpiralProto",
            "sprSpiralIDPD",
            "sprSpiralIDPD2",
            "sprSpiralStar",
        ];
        for (slot, stem) in stems.into_iter().enumerate() {
            let (w, h, rgba) = if stem.is_empty() {
                debris.clone()
            } else {
                decode(dir, stem).unwrap_or((1u32, 1u32, vec![255, 255, 255, 255]))
            };
            out.push(VortexTexture {
                slot: slot as u32,
                w,
                h,
                rgba: rgba.into(),
                generation,
            });
        }
        for (index, stem) in VARD_VARIANTS.into_iter().enumerate() {
            let (w, h, rgba) = decode(dir, stem.strip_prefix("images/").unwrap_or(stem))
                .unwrap_or((1u32, 1u32, vec![0, 0, 0, 0]));
            out.push(VortexTexture {
                slot: (7 + index) as u32,
                w,
                h,
                rgba: rgba.into(),
                generation,
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
        self.sim.world.resource_mut::<InputMapState>().session.map = map;
        self.save_path = Some(path.to_path_buf());
        self.sim
            .world
            .resource::<crate::savedata_part::SaveData>()
            .clone()
    }

    /// Flush the live save to disk when dirty (GML `scrSave` on every
    /// rebind/option change: `scrOptionsSaveKeymaps` + `scrSave`). The
    /// dirty flag alone never reaches disk - without this a restart
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

    /// Touch chrome sprite count for device logs: proves the sticks /
    /// buttons drew (GML `scrDrawMobileControls` parity) without a
    /// screenshot. Set every `view` alongside `last_sprite_count`
    /// (0 when the touch-device gate is off).
    pub fn last_touch_count(&self) -> usize {
        self.last_touch_count
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
            // phase - pending spawn, live fight, death pageant).
            let throne_ii_active = self
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
            self.spiral.bossfight_suppressed = throne_ii_active;
            // Ordinary pause freezes the spiral: GML
            // `UberCont/Step_1` deactivates the room. Generation covers
            // are the exception: `GenCont`/`LevCont` keep drawing their
            // live spiral even though the offer sets the global pause
            // flag. Sounds follow the same gate below.
            let paused = self
                .sim
                .world
                .get_resource::<crate::state::Paused>()
                .is_some_and(|p| p.0);
            let state = self
                .sim
                .world
                .get_resource::<AppState>()
                .copied()
                .unwrap_or_default();
            let splash_mode = self
                .sim
                .world
                .get_resource::<crate::state::SplashState>()
                .map(|splash| splash.mode)
                .unwrap_or(0);
            let spiral_cover = self
                .sim
                .world
                .get_resource::<crate::comps_b::FloorTransition>()
                .is_some_and(|transition| transition.active)
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
            let splash_without_cont =
                state == AppState::Splash && (splash_mode < 4 || self.last_splash_mode < 4);
            if (!paused || spiral_cover) && !splash_without_cont {
                self.spiral.step(1.0);
            }
            // GML `scrDrawSpiral` bolt/debris one-shots fire inline in
            // the draw script: lightning `sndPortalLightning{1..8}` once
            // per wisp (non-menu only), flyby `sndPortalFlyby{1..4}` once
            // per debris mote at `xscale > 1.3` (any caller). Drain here,
            // right after the step, so each fires exactly once.
            let full_spiral = state != AppState::InGame || self.spiral.alive || spiral_cover;
            if !splash_without_cont && (!paused || spiral_cover) && full_spiral {
                self.drain_spiral_sounds();
            }
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

    /// Spiral lifecycle per fixed step (view-layer half of bevy
    /// `mark_vortex_dead` / `ensure_spiral_for_levelup`; the sim
    /// `SpiralCtl` the ambience duck keys off is untouched):
    /// - Loading entry warms a FRESH spiral (`Vlambeer` builds a fresh
    ///   `SpiralCont` + `GenCont` on every `room_restart`, including RETRY
    ///   - a continued mid-flight spiral would jump).
    /// - InGame entry kills it (`GenCont/Destroy`; the 26-tick drain plays
    ///   out like bevy's).
    /// - Rising generation cover (floor transition or mutation/ultra offer)
    ///   re-warms (`GenCont`/`LevCont` rooms build theirs); falling edge
    ///   kills (`GenCont/Destroy`). Live gameplay, Title (no `SpiralCont`
    ///   in the `MenuGen` room) and GameOver show only the flat area colour.
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
        // Bevy `mark_vortex_dead` is a one-way latch per tick: a kill and a
        // rewarm must NEVER both fire in one call. On the `setup_run` tick
        // the fresh seed arms the entry kill below AND a pending pick
        // raises the cover edge further down - without the latch the cover
        // rewarm rebuilds a live 150-tick spiral in the same tick the kill
        // just drained (the load-end double start, ticks 185->150).
        let mut killed_this_tick = false;
        let throne_ii_active = self
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
        let splash_mode = self
            .sim
            .world
            .get_resource::<crate::state::SplashState>()
            .map(|splash| splash.mode)
            .unwrap_or(0);
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
        if state == AppState::Splash && splash_mode >= 4 && self.last_splash_mode < 4 {
            let view_w = self.spiral.view_w;
            self.spiral = SpiralCtl::warmed_up_for_gml_area_seeded_in_view(0, seed, view_w);
        }
        self.last_splash_mode = if state == AppState::Splash {
            splash_mode
        } else {
            0
        };
        if state == AppState::Loading && self.adv_state != AppState::Loading {
            let view_w = self.spiral.view_w;
            self.spiral = SpiralCtl::warmed_up_for_gml_area_seeded_in_view(
                gml_area_for_area(area),
                seed,
                view_w,
            );
            self.adv_seed = seed;
            self.adv_area = gml_area_for_area(area);
        }
        // A generation seed change normally marks a new run and drains
        // the old cont. During an active generation cover it instead
        // identifies the fresh room and is rewarmed below.
        if state == AppState::InGame && seed != self.adv_seed {
            self.adv_seed = seed;
            self.adv_area = gml_area_for_area(area);
            if cover {
                self.cover_rewarm_pending = true;
            } else {
                self.spiral.kill();
                killed_this_tick = true;
            }
        }
        if state == AppState::InGame && throne_ii_active && !self.adv_throne_ii {
            let view_w = self.spiral.view_w;
            self.spiral = SpiralCtl::warmed_up_for_gml_area_seeded_in_view(
                gml_area_for_area(area),
                seed,
                view_w,
            );
            self.spiral.bossfight_suppressed = true;
            self.adv_seed = seed;
            self.adv_area = gml_area_for_area(area);
            killed_this_tick = false;
        }
        // GML `Vlambeer/Create_0` `want_quit_to_menu` branch verbatim: quitting to
        // the logo menu builds a FRESH live `SpiralCont`
        // (`instance_create(x, y, SpiralCont)` with its `repeat 150`
        // warmup), never the previous run's leftover drain. Quit ACTION
        // arms (`ConfirmPause(0)` / `QuitToTitle` call `rewarm_view_spiral`
        // before `goto_state`) already build it; this covers direct
        // `goto_state(MainMenu)` shells like tests. Must NOT fire on the
        // boot Splash->MainMenu edge: GML's logo-room cont (created once in
        // `Vlambeer/Alarm_0`) is the same object still swirling, and a
        // rewarm here restarts the vortex under the PLAY rows (the "vortex
        // starts twice" bug).
        if state == AppState::MainMenu
            && self.adv_state != AppState::MainMenu
            && self.adv_state != AppState::Splash
            && !self.spiral.alive
        {
            let view_w = self.spiral.view_w;
            self.spiral = SpiralCtl::warmed_up_for_gml_area_seeded_in_view(0, seed, view_w);
        }
        if state == AppState::MainMenu
            && self
                .sim
                .world
                .get_resource::<OverlayMenu>()
                .is_some_and(|overlay| *overlay == OverlayMenu::Credits)
            && self.spiral.is_done()
        {
            let view_w = self.spiral.view_w;
            self.spiral = SpiralCtl::warmed_up_for_gml_area_seeded_in_view(0, seed, view_w);
        }
        // GML `PlayButton/Other_10:108` verbatim: campfire char-select
        // DESTROYS the `SpiralCont` while leftover `Spiral/SpiralDebris/
        // SpiralStar` motes keep stepping in a cont-less room (drain growth
        // 1.5x, wisp kill-plane 3.0, no new births, but `lanim` keeps
        // realtime cadence so bolts keep flashing). `Menu/Draw_0`'s
        // `scrDrawSpiral` draws that remnant transparently (no `draw_clear`)
        // over the camp; only once every mote is culled does the flat camp
        // show. Hence kill, not re-warm, with the layer mounted until
        // `is_done` (~a second of remnant, exactly like GML).
        if state == AppState::Title && self.adv_state != AppState::Title {
            self.spiral.kill();
            killed_this_tick = true;
        }
        if cover {
            if killed_this_tick {
                self.cover_rewarm_pending = true;
            } else if !self.adv_cover || self.cover_rewarm_pending {
                let view_w = self.spiral.view_w;
                self.spiral = SpiralCtl::warmed_up_for_gml_area_seeded_in_view(
                    gml_area_for_area(area),
                    seed,
                    view_w,
                );
                self.cover_rewarm_pending = false;
            }
        } else {
            if self.adv_cover {
                self.spiral.kill();
            }
            self.cover_rewarm_pending = false;
        }
        self.adv_state = state;
        self.adv_cover = cover;
        self.adv_throne_ii = state == AppState::InGame && throne_ii_active;
    }

    /// One fixed-step GML camera step (`objects/BackCont/Step_0.gml`): POI
    /// pull + aim lean + shake sample feed [`gml_camera_step`] at exactly
    /// 30 Hz (per-frame stepping + per-frame `round()` juddered at 60+
    /// fps). Frozen unless live (InGame, unpaused, no overlay, no game
    /// over).
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
        // Cursor-lean source by device (GML `scrHandleInputs` law): touch aim
        // comes from the attack stick (`dir_fire`/`vdis` off
        // `JoystickAttack`), never a cursor point - the stick latches its
        // heading on release (GML `dir_fire` is never zeroed; `vdis`
        // decays at 2 px/tick) so the lean glides home. Reading the cursor
        // on touch re-aimed the gun AND leaned the camera at the tap point
        // while the shot went stick-side.
        #[cfg(target_os = "android")]
        let (aim_dir, aim_dis) = {
            let input = self.sim.world.resource::<NtInput>();
            let dis = input.touch_dis;
            let dir_deg = input.attack_stick.map(|s| s.dir).unwrap_or(0.0);
            let rad = dir_deg.to_radians();
            (Vec2::new(rad.cos(), rad.sin()), dis)
        };
        #[cfg(not(target_os = "android"))]
        let step_in = {
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
                .map(|(_, _, inv)| inv.weapons[inv.current])
                .unwrap_or(crate::data::WeaponId::NONE);
            // GML `KeyCont.dis_fire`: cursor distance in world px, capped at
            // 48 px (bevy `player_aim` `MAX_LOOK`; unbounded `dis/viewdist`
            // drifts whole screens at a window edge). Uses the live cursor
            // unprojection, so the lean follows the cursor as the camera
            // moves. (`cursor_to_world` borrows `self.cam` only, so it was
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
            // GML POI chain: Portal, then victory/sit markers (`BecomeNothing`/
            // `NothingDeath` have no port counterpart yet) - nearest wins,
            // cap only for portals.
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
            CamStepInput {
                player,
                aim_dir,
                aim_dis,
                viewdist: cam_viewdist_for(wep),
                poi,
                shake_scale,
                timescale: 0.0,
                jx: rng.random_range(-1.0..1.0),
                jy: rng.random_range(-1.0..1.0),
            }
        };
        #[cfg(target_os = "android")]
        let step_in = {
            let world = &mut self.sim.world;
            let player = player_pos(world).unwrap_or(rest);
            let wep = world
                .query::<(&Pos, &crate::comps_a::Player, &crate::comps_a::Inventory)>()
                .iter(world)
                .next()
                .map(|(_, _, inv)| inv.weapons[inv.current])
                .unwrap_or(crate::data::WeaponId::NONE);
            let shake_scale = world
                .get_resource::<crate::savedata_part::SaveData>()
                .map(|s| s.settings.screenshake.clamp(0.0, 2.0))
                .unwrap_or(1.0);
            let shake_px = world
                .get_resource::<repame_fx::Trauma>()
                .map(|t| t.amount * t.max_translation_px)
                .unwrap_or(0.0);
            self.gml_cam.shake = shake_px;
            let mut rng = rand::rng();
            CamStepInput {
                player,
                aim_dir,
                aim_dis,
                viewdist: cam_viewdist_for(wep),
                poi: nearest_poi(world, player),
                shake_scale,
                timescale: 0.0,
                jx: rng.random_range(-1.0..1.0),
                jy: rng.random_range(-1.0..1.0),
            }
        };
        const STEP_DT: f32 = 1.0 / SIM_HZ as f32;
        let vw_vh = self.view_world_size;
        gml_camera_step(&mut self.gml_cam, vw_vh[0], vw_vh[1], &step_in, STEP_DT);
    }

    fn maybe_rewarm_spiral(&mut self) {
        // GML `room_restart` parity: a LIVE gameplay area/seed change (portal,
        // new run) re-warms the spiral - `Vlambeer/Create_0` builds a fresh
        // `SpiralCont` on every gameplay restart. Menu rooms never restart
        // the vortex (logo-room cont created once in `Vlambeer/Alarm_0`;
        // campfire inherits the drain, `PlayButton/Other_10` destroys the
        // cont and `Menu/Draw_0` draws the leftovers), so a menu-room
        // resource reset (`setup_logo_room` on the Splash->MainMenu edge
        // rewriting Desert/seed into Campfire/seed-0) must NOT read as
        // area/seed drift - that restarted the vortex under the PLAY rows
        // (the "vortex starts twice" bug).
        let state = self
            .sim
            .world
            .get_resource::<AppState>()
            .copied()
            .unwrap_or_default();
        if !matches!(state, AppState::Loading | AppState::InGame) {
            return;
        }
        // The lifecycle owns generation-cover seed changes, so a seed change here
        // is outside a cover = a new run boundary.
        let run = self.sim.world.get_resource::<crate::comps_a::Run>();
        let (area, seed) = run
            .map(|r| (r.area, r.gen_seed))
            .unwrap_or((AreaId::Desert, 0));
        if seed != self.adv_seed {
            // A seed change outside a generation cover is a new run
            // boundary. GML `GenCont/Destroy` drains that old cont.
            self.spiral.kill();
            self.adv_seed = seed;
        } else if gml_area_for_area(area) != self.adv_area {
            // Same run, new area art (portal kept the seed): re-warm so the debris
            // strip follows the GML area. Compares against the lifecycle
            // stamp, NOT the live spiral: the Loading warmup carries the
            // menu-room area, so the first InGame tick would otherwise
            // rewarm right after the entry kill (load-end double start).
            self.spiral = SpiralCtl::warmed_up_for_gml_area_seeded_in_view(
                gml_area_for_area(area),
                seed,
                self.spiral.view_w,
            );
            self.adv_area = gml_area_for_area(area);
        }
    }

    /// Drain the spiral one-shots after each fixed step (GML `scrDrawSpiral`
    /// plays them inline while drawing): bolt `sndPortalLightning{1..8}`
    /// once per wisp (the draw script gates on `!_is_menu`, i.e. every
    /// state but the campfire title - the same `bg_alpha == 1` set the
    /// snapshot uses), flyby `sndPortalFlyby{1..4}` once per debris mote
    /// at `xscale > 1.3`. Variant/stem rolls come off the spiral's
    /// deterministic run-seed stream so equal seeds sound identical.
    fn drain_spiral_sounds(&mut self) {
        use crate::audio::AudioCue;
        use crate::msg::Queue;
        // GML `draw_clear` caller: everyone but `Menu`, and `bg_alpha` is 0 only on
        // the title, so reuse the last snapshot decision (defaults to
        // sounding before the first frame, like a fresh GML room).
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
                    variance: 0.2,
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
        for (i, vard) in self.spiral.vards.iter_mut().enumerate() {
            if vard.flyby_due() {
                let n = 1 + (crate::vortex::stream_pick(seed, 20_000 + i as u32, 11) % 4);
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

pub mod nt_shortcuts {
    use repose_core::shortcuts::{Action, ShortcutMap, ShortcutState};

    use super::App;

    pub const PAUSE: &str = "nt.pause";
    pub const RESTART: &str = "nt.restart";
    pub const CONFIRM: &str = "nt.confirm";

    pub fn map() -> ShortcutMap {
        repame_shell::game_shortcut_map(PAUSE, RESTART, CONFIRM)
    }

    pub fn state(app: &App) -> ShortcutState {
        let mut state = ShortcutState::new();
        state.default_map = map();
        let inner = app.shortcut_edges.clone();
        state.handler = Some(std::rc::Rc::new(move |action| {
            let mut e = inner.borrow_mut();
            match action {
                Action::Custom(key) if key.as_ref() == PAUSE => {
                    e.pause = true;
                    true
                }
                Action::Custom(key) if key.as_ref() == RESTART => {
                    e.restart = true;
                    true
                }
                Action::Custom(key) if key.as_ref() == CONFIRM => {
                    e.confirm = true;
                    true
                }
                _ => false,
            }
        }));
        state
    }

    pub fn drain(app: &mut App) {
        let (pause, restart, confirm) = repame_shell::take_shortcut_edges(&app.shortcut_edges);
        app.pause_edge |= pause;
        app.restart_edge |= restart;
        app.interact_edge |= confirm;
    }

    /// Prime the process-global maps the runner falls back to
    /// (idempotent, same bindings every call). Live `view` calls this
    /// once per frame; tests call it once per tape.
    pub fn install_map_once() {
        repame_shell::install_map(map());
    }

    /// Test-owned shortcut state wired to `app`'s edges: `resolve` +
    /// `handle` on this replace the global install + global dispatch
    /// in tests, so parallel tests never share shortcut state.
    pub fn test_state(app: &App) -> ShortcutState {
        state(app)
    }
}

impl App {
    /// Shared key-event entry: the live root `on_key_event` closure and
    /// the golden demo route here. Character keys match by physical
    /// position (`KeyW`, not `'w'`), so non-US layouts move the same
    /// way GML's `ord("W")` does on a US board.
    pub fn handle_key(&mut self, ke: &KeyEvent) {
        let down = matches!(ke.event_type, KeyEventType::Down);
        if down && !ke.is_repeat && self.capture_armed() {
            if let Some(key) = ke.physical {
                self.capture_physical_press(key);
            } else {
                self.capture_key_press(&ke.key);
            }
            return;
        }
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
                if down && !ke.is_repeat {
                    let overlay = self
                        .sim
                        .world
                        .get_resource::<OverlayMenu>()
                        .copied()
                        .unwrap_or_default();
                    if overlay == OverlayMenu::Settings {
                        self.menu_actions.push(UiAction::SettingsBack);
                    } else {
                        self.cancel_remap_capture();
                    }
                }
            }
            _ => {}
        }
    }

    /// Stage one physical key transition. Down transitions push press
    /// edges (deduped against `held`); releases only clear `held`.
    /// Golden-demo hook: the tape drives keys without a live `KeyEvent`
    /// route, so it stages here directly into shared staging.
    pub fn stage_physical_key(&mut self, key: PhysicalKey, down: bool) {
        self.stage_physical(key, down, false);
    }

    fn stage_physical(&mut self, key: PhysicalKey, down: bool, is_repeat: bool) {
        self.staging
            .borrow_mut()
            .stage_physical(key, down, is_repeat);
    }

    /// Window focus changed (shell forwards winit `Focused`). Losing
    /// focus drops every held key + edge: no key-ups arrive across an
    /// alt-tab, and GML's `keyboard_check` reads all-up there too.
    /// Staging owns the full cancel (keys, edges, clicks, mouse, touch,
    /// pads); the `window_focused` mirror just tracks it.
    pub fn set_window_focused(&mut self, focused: bool) {
        self.staging.borrow_mut().set_window_focused(focused);
        if !focused {
            self.settings_slider_drag = None;
        }
        self.window_focused = self.staging.borrow().window_focused;
    }

    pub fn stage_key(&mut self, ke: &KeyEvent) {
        self.handle_key(ke);
    }

    /// Release half of `stage_physical_key`: clears the staging level so
    /// the release path matches hardware key-up exactly.
    pub fn staging_key_up(&mut self, key: PhysicalKey) {
        self.staging.borrow_mut().stage_physical(key, false, false);
    }

    fn stage_code(&mut self, code: KeyCode, down: bool, _is_repeat: bool) {
        let Some(key) = crate::input::physical_key_for_code(code) else {
            return;
        };
        self.staging
            .borrow_mut()
            .stage_physical(key, down, _is_repeat);
    }

    /// REMAP capture: the next pressed input resolves the pending rebind
    /// (`Key[$ key][type] = k`, `Other_10:374-404`). Mouse buttons capture
    /// from the viewport press path (`pick_down`); keys here and in
    /// `stage_physical`. `true` while armed - callers skip normal staging
    /// so the capture key does not fire gameplay.
    pub(crate) fn capture_armed(&self) -> bool {
        self.sim
            .world
            .get_resource::<InputMapState>()
            .is_some_and(|s| s.armed())
    }

    /// Focus-preview capture entry: `on_preview_key_event` routes here BEFORE
    /// the runtime shortcut dispatch, so Esc/Enter/R rebind while a capture
    /// is armed (the runtime would otherwise consume them as
    /// pause/restart/confirm). `false` on non-armed events so normal
    /// dispatch proceeds.
    pub fn preview_capture_key(&mut self, ke: &KeyEvent) -> bool {
        if !matches!(ke.event_type, KeyEventType::Down) || ke.is_repeat || !self.capture_armed() {
            return false;
        }
        if let Some(key) = ke.physical {
            self.capture_physical_press(key);
        } else {
            self.capture_key_press(&ke.key);
        }
        true
    }

    /// Resolve the pending capture with a mouse button (viewport press
    /// path). GML only captures `mb_left`/`mb_right` for the keyboard
    /// side; gamepad-side captures resolve from pad edges in
    /// `feed_input`.
    fn capture_mouse_press(&mut self, left: bool) {
        self.sim.world.init_resource::<InputMapState>();
        let done = self
            .sim
            .world
            .resource_mut::<InputMapState>()
            .resolve_mouse(left);
        if done {
            self.persist_keymap();
        }
    }

    pub(crate) fn capture_key_press(&mut self, key: &repose_core::input::Key) {
        self.sim.world.init_resource::<InputMapState>();
        let done = self
            .sim
            .world
            .resource_mut::<InputMapState>()
            .resolve_key(key);
        if done {
            self.persist_keymap();
        }
    }

    pub(crate) fn capture_physical_press(&mut self, key: PhysicalKey) {
        self.sim.world.init_resource::<InputMapState>();
        if !self.capture_armed() {
            return;
        }
        let done = self
            .sim
            .world
            .resource_mut::<InputMapState>()
            .resolve_physical(key);
        if done {
            self.persist_keymap();
        }
    }

    fn cancel_remap_capture(&mut self) {
        self.sim.world.init_resource::<InputMapState>();
        let mut state = self.sim.world.resource_mut::<InputMapState>();
        if state.armed() {
            state.resolve(None);
            drop(state);
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
            .map(|s| KeyBindings::from_keymap(&s.session.map))
            .unwrap_or_default();
        self.sim
            .world
            .init_resource::<crate::savedata_part::SaveData>();
        self.sim
            .world
            .resource_mut::<crate::savedata_part::SaveData>()
            .key_bindings = rows;
        self.sim.world.init_resource::<crate::comps_a::SaveDirty>();
        self.sim.world.resource_mut::<crate::comps_a::SaveDirty>().0 = true;
        self.save_now();
    }

    /// Viewport mouse-button down: secondary stages the RMB edge (GML `spec` /
    /// menu Back; sampled through `MouseState::right_*`), primary latches
    /// `lmb_held` so automatic weapons keep firing while held. A pending
    /// REMAP capture eats the press instead (GML captures `mb_left` /
    /// `mb_right`). Public for the golden demo walkthrough
    /// (`tests/golden_demo.rs`), which drives the live viewport path.
    pub fn pick_down(&mut self, button: PointerButton) {
        if self.capture_armed() {
            let left = button == PointerButton::Primary;
            self.capture_mouse_press(left);
            return;
        }
        self.staging.borrow_mut().pick_down(button);
    }

    /// Public for the golden demo walkthrough (`tests/golden_demo.rs`).
    pub fn pick_up(&mut self, button: PointerButton) {
        self.staging.borrow_mut().pick_up(button);
    }

    /// Runtime-owned cursor position: [`App::view`] copies
    /// `Scheduler::pointer_pos_px` into `polled_pointer_px` every frame
    /// (GML `mouse_x`/`mouse_y` parity: no focus dispatch, no hit regions,
    /// no staleness clock). Unprojected through this frame's camera so aim
    /// stays glued to the on-screen pointer (bevy `player_aim`
    /// `viewport_to_world_2d` parity); `None` until the first mouse move.
    pub fn stage_pointer_px(&mut self, phys_px: Option<[f32; 2]>) {
        self.polled_pointer_px = phys_px.map(|[x, y]| Vec2::new(x, y));
    }

    /// Live cursor in world coords: polled window-physical px
    /// unprojected through this frame's camera. Pure viewport math
    /// lives in [`repame_sprite::unproject_px`]; the NT wrappings
    /// (`Option` polling, NT's dp-viewport fields) stay here.
    fn live_cursor_world(&self) -> Option<Vec2> {
        self.px_to_world(self.polled_pointer_px)
    }

    /// Unproject one window-physical px point through this frame's
    /// camera. Pure viewport math lives in
    /// [`repame_sprite::unproject_px`]; the NT wrappings (`Option`
    /// polling, NT's dp-viewport fields) stay here.
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

    //// Decode the live `sprCrosshair` frame + `opt_cursorcol` tint into the
    /// hardware cursor payload (`cursor_img`); mechanical pixel work lives in
    /// [`repame_sprite::cursor_frame`], this owns settings reads, file IO and
    /// the `cursor_img` cache. GML draws the raw strip cell at GUI mouse
    /// (`Draw_75`, no lerp, alpha 1) with the catalog-origin hotspot (crosshair
    /// strips center on (8,8)). Missing assets clear the payload and callers
    /// fall back to `Hidden`.
    ///
    /// Size law: GML draws the cell at scale 1 in GUI space
    /// (`device_mouse_x_to_gui`, GUI stretched over the window), so the 16px
    /// cell covers `16 * window_w / view_w` screen px (48 at 1280x720). The OS
    /// cursor buffer is physical px while `units_per_pixel` is dp, so
    /// magnification is `density / units_per_pixel` - not
    /// `1 / units_per_pixel`, which undersizes the cursor on fractional display
    /// scales. The scale rides the cache key so resizes and monitor moves
    /// re-decode.
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
        let frames =
            crate::render::strip_frames_pub(assets, "images/sprCrosshair.png").max(1) as i32;
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
        // Physical px per world unit: density (px per dp) times the
        // frame's dp-per-world contain scale.
        let mag = (self.view_density.max(1e-6) * self.gml_frame().dp_per_world)
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
        self.staging.borrow_mut().stage_gamepad(pad);
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
    /// Public for the golden demo walkthrough, same path as live.
    pub fn stage_click(&mut self, world: Vec2, screen: [f32; 2]) {
        let d = repose_core::locals::effective_density_scale().max(1e-6);
        self.staging.borrow_mut().stage_click(world, screen, d);
    }

    pub fn touch_down(&mut self, id: u64, screen: Vec2) {
        let d = repose_core::locals::effective_density_scale().max(1e-6);
        self.staging.borrow_mut().touch_down(id, screen / d);
    }

    pub fn touch_move(&mut self, id: u64, screen: Vec2) {
        let d = repose_core::locals::effective_density_scale().max(1e-6);
        self.staging.borrow_mut().touch_move(id, screen / d);
    }

    pub fn touch_up(&mut self, id: u64) {
        // GML reads press/release edges off the live touch slot, but the shell
        // drops contacts on lift - latch the lifted id here and consume it
        // next `sample_touch` (swapped attack edges, release-driven
        // consumers like the Plant auto-snare). Lift routing reads
        // `NtInput`'s claims: the attack finger id means fire, anything
        // else no edge; stale ids (stick already released) latch nothing.
        self.staging.borrow_mut().touch_up(id);
        self.sim
            .world
            .resource_mut::<NtInput>()
            .note_touch_released(id as i64);
    }

    /// Sync event-staged levels against the platform snapshot
    /// (`Scheduler::held_keys` + mouse levels). Event handlers own press
    /// edges; the polled set only repairs what they missed: release a
    /// key/button never seen going up (release outside the window,
    /// swallowed key-up), never force a missed down. Keyboard and mouse
    /// share the rule; Esc/R shortcut edges stay with the shortcuts.
    pub fn feed_polled(&mut self, sched: &Scheduler) {
        // Single store: staging owns held keys + mouse levels, so sampling and
        // repair read the same fields.
        self.staging.borrow_mut().feed_polled(sched);
        self.window_focused = self.staging.borrow().window_focused;
    }

    /// Drain staged shell input into sim resources (runs before
    /// [`App::advance`] each frame; pulses are take-once downstream).
    /// Public for the golden demo walkthrough, which drives the same
    /// per-frame order as `view` without a live `Scheduler`.
    pub fn feed_input(&mut self) {
        for action in self.menu_actions.drain(..) {
            apply_menu_action(&mut self.sim.world, action);
        }
        // GML per-frame mouse sync: `mouse_ui_hovered` runs inside every button's
        // Step (not on click), so hover/cursor state must refresh with no
        // click staged. GUI point = the polled pointer copied at the top of
        // `view`; touch/gamepad frames leave the keyboard cursors alone
        // (GML `!is_gamepad()` guard).
        self.tick_menu_hover();
        let was_armed = self.capture_armed();
        {
            let (pending_physical, pending_key, pending_mouse) = {
                let mut staging = self.staging.borrow_mut();
                if was_armed {
                    (
                        staging.take_capture_physical(),
                        staging.take_capture_key(),
                        staging.take_capture_mouse(),
                    )
                } else {
                    staging.capture_pending_physical = None;
                    staging.capture_pending_key = None;
                    staging.capture_pending_mouse = None;
                    (None, None, None)
                }
            };
            if was_armed {
                if let Some(key) = pending_physical {
                    self.capture_physical_press(key);
                    self.staging.borrow_mut().edges.clear();
                } else if let Some(key) = pending_key {
                    if key == Key::Backspace {
                        self.cancel_remap_capture();
                    } else {
                        self.capture_key_press(&key);
                    }
                    self.staging.borrow_mut().edges.clear();
                } else if let Some(left) = pending_mouse {
                    self.capture_mouse_press(left);
                }
            }
            self.staging.borrow_mut().capture_armed = self.capture_armed();
        }
        nt_shortcuts::drain(self);
        // Staging owns held levels; translate physical edges into the sampler's
        // `KeyCode` space here.
        let mut staging = self.staging.borrow_mut();
        let mut staged_held: HashSet<KeyCode> = HashSet::new();
        for key in staging.held.iter() {
            if let Some(code) = crate::input::keycode_for_physical(*key) {
                staged_held.insert(code);
            }
        }
        let mut just: HashSet<KeyCode> = HashSet::new();
        for code in staging.take_edges() {
            if let Some(mapped) = crate::input::keycode_for_physical(code) {
                staged_held.insert(mapped);
                just.insert(mapped);
            }
        }
        let rmb_down = staging.take_rmb_down();
        let staging_clicks = staging.take_clicks();
        let staging_lmb = staging.lmb_held;
        let staging_rmb = staging.rmb_held;
        // Touch contacts MUST be snapshotted in the same borrow as the clicks:
        // `touch_contacts` clears `touch_new` (the sampler's `just_pressed`
        // edge), so a second call at the sampler would see cleared edges -
        // every press arrives a frame late as a hold with no edge (sticks
        // never claim, buttons never pulse).
        let touch_contacts_snap = staging.touch_contacts();
        drop(staging);
        let mut touch_menu_positions = std::mem::take(&mut self.touch_menu_positions);
        for contact in &touch_contacts_snap {
            let id = contact.id as i64;
            if contact.just_pressed {
                touch_menu_positions.push((id, contact.pos));
            } else if let Some((_, position)) =
                touch_menu_positions.iter_mut().find(|entry| entry.0 == id)
            {
                *position = contact.pos;
            }
        }
        let mut released_touch_clicks: Vec<(i64, Vec2)> = Vec::new();
        let mut slider_touch_ids: HashSet<i64> = HashSet::new();
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
        let run_over = self
            .sim
            .world
            .get_resource::<crate::comps_a::Run>()
            .is_some_and(|run| run.game_over);
        let game_over = crate::state::menus::game_over_visible(&self.sim.world);
        let mut slider_click_consumed = false;
        if overlay != OverlayMenu::Settings {
            self.finish_settings_slider_drag();
        } else {
            let page = self
                .sim
                .world
                .get_resource::<MenuState>()
                .map(|menu| menu.settings_page)
                .unwrap_or(0);
            let vw = gml_view_size(self.view_viewport_dp)[0];
            if self
                .settings_slider_drag
                .is_some_and(|drag| drag.page != page)
            {
                self.finish_settings_slider_drag();
            }
            if let Some(drag) = self.settings_slider_drag {
                match drag.pointer {
                    SettingsSliderPointer::Mouse => {
                        if !staging_lmb {
                            self.finish_settings_slider_drag();
                        } else if let Some(px) = self.polled_pointer_px {
                            let d = self.view_density.max(1e-6);
                            let k = (self.view_viewport_dp[1].max(1.0) / 240.0).max(1e-6);
                            self.update_settings_slider_drag(px.x / d / k, vw);
                        }
                    }
                    SettingsSliderPointer::Touch(id) => {
                        if let Some(contact) =
                            touch_contacts_snap.iter().find(|contact| contact.id == id)
                        {
                            if let Some([gx, _]) =
                                self.settings_gui_point_dp([contact.pos.x, contact.pos.y])
                            {
                                self.update_settings_slider_drag(gx, vw);
                            }
                        } else {
                            slider_touch_ids.insert(id as i64);
                            self.finish_settings_slider_drag();
                        }
                    }
                }
            }
            if self.settings_slider_drag.is_none()
                && let Some(click) = staging_clicks.last().copied()
                && let Some([gx, gy]) = self.settings_gui_point_dp(click.dp)
                && let Some((_, target)) = settings_slider_hit(page, gx, gy, vw)
            {
                self.begin_settings_slider_drag(page, target, SettingsSliderPointer::Mouse, gx, vw);
                slider_click_consumed = true;
            }
            if self.settings_slider_drag.is_none() {
                for contact in touch_contacts_snap
                    .iter()
                    .filter(|contact| contact.just_pressed)
                {
                    let Some([gx, gy]) = self.settings_gui_point_dp([contact.pos.x, contact.pos.y])
                    else {
                        continue;
                    };
                    let Some((_, target)) = settings_slider_hit(page, gx, gy, vw) else {
                        continue;
                    };
                    self.begin_settings_slider_drag(
                        page,
                        target,
                        SettingsSliderPointer::Touch(contact.id),
                        gx,
                        vw,
                    );
                    slider_touch_ids.insert(contact.id as i64);
                }
            }
        }
        // Read the pending offer directly, not the `MenuState` mirror
        // (the mirror updates inside the schedule, a frame after the
        // offer opens/closes - the stale frame fired the gun on level-up
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
            && !run_over
            && overlay == OverlayMenu::None
            && self
                .sim
                .world
                .get_resource::<MenuState>()
                .is_none_or(|m| !m.unlock.visible);
        let menu_open = state == AppState::InGame
            && (paused
                || overlay != OverlayMenu::None
                || self
                    .sim
                    .world
                    .get_resource::<MenuState>()
                    .is_some_and(|m| m.unlock.visible))
            && !run_over;

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
            let pause_open = overlay == OverlayMenu::Pause;
            let mut dv: i8 = 0;
            let mut dh: i8 = 0;
            if state == AppState::MainMenu || settings_open || pause_open {
                if just.contains(&KeyCode::ArrowUp) {
                    dv -= 1;
                }
                if just.contains(&KeyCode::ArrowDown) {
                    dv += 1;
                }
            }
            if settings_open || pause_open {
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
        let state = self
            .sim
            .world
            .get_resource::<AppState>()
            .copied()
            .unwrap_or_default();
        let (keyboard_mode, gamepad_mode) = crate::input::gml_input_device(
            self.sim
                .world
                .get_resource::<crate::savedata_part::SaveData>(),
        );
        let touch_only_device = cfg!(target_os = "android") || !(keyboard_mode || gamepad_mode);
        let mouse_down_edge = !staging_clicks.is_empty() && !touch_only_device;
        let mouse_down = (mouse_down_edge || (staging_lmb && !touch_only_device))
            && !menu_open
            && !run_over
            && !offer_open
            && state != AppState::MainMenu
            && state != AppState::Title
            && state != AppState::Splash
            && state != AppState::Loading;
        let mouse = MouseState {
            left_held: mouse_down,
            left_pressed: mouse_down_edge
                && !menu_open
                && !run_over
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
            right_held: staging_rmb,
            right_pressed: rmb_down,
            ..mouse
        };
        {
            self.sim.world.init_resource::<InputMapState>();
            let keymap = self.sim.world.resource::<InputMapState>().clone();
            let (touch_scale, touch_split_fire, touch_stick_regions) = self
                .sim
                .world
                .get_resource::<crate::savedata_part::SaveData>()
                .map(|s| {
                    (
                        s.settings.controls_scale,
                        s.settings.split_fire,
                        s.settings.stick_regions,
                    )
                })
                .unwrap_or((0.5, false, false));
            let pad_pause_confirm = self
                .sim
                .world
                .get_resource::<MenuState>()
                .and_then(|menu| menu.pause_confirm)
                .is_some();
            let pad_back_submenu = self
                .sim
                .world
                .get_resource::<MenuState>()
                .is_some_and(|menu| menu.play_submenu);
            // Resolved before the `NtInput` borrow below (it reads `cam`
            // and the cached viewport, not the world).
            let frame = self.gml_frame();
            let mut input = self.sim.world.resource_mut::<NtInput>();
            // Menu screens own Space/arrows: strip their edges before the
            // gameplay sampler so one press can't both confirm a menu
            // row and pulse a gameplay action (fire/swap).
            let (held, just) = if in_menu {
                let held: HashSet<KeyCode> = staged_held
                    .iter()
                    .copied()
                    .filter(|c| !menu_owned_key(*c))
                    .collect();
                let sampled: HashSet<KeyCode> = just
                    .iter()
                    .copied()
                    .filter(|c| !menu_owned_key(*c))
                    .collect();
                (held, sampled)
            } else {
                (staged_held.clone(), just.clone())
            };
            crate::input::sample_keyboard_mapped(&held, &just, &mouse, Some(&keymap), &mut input);
            let pads = self.staging.borrow_mut().take_pads();
            self.pad_live = self.staging.borrow().pad_live;
            if pads.iter().any(|pad| pad.east_pressed) {
                // GML `BackButton/Step_0:22`: `gp_face2` is one of the four
                // triggers feeding the single back event (with Esc,
                // Backspace and the keyboard RMB), so it takes the same law
                // as RMB: `SettingsBack` on options, `CloseOverlay` on
                // `Credits`/`DrawStats` (`BackButton/Other_10`), resume off
                // the pause screen, `PlayButton` list closes (`Other_10`'s
                // `instance_exists(PlayButton)` branch); the main menu owns
                // no BackButton.
                match overlay {
                    OverlayMenu::Settings => self.menu_actions.push(UiAction::SettingsBack),
                    OverlayMenu::Pause => {
                        let action = if pad_pause_confirm {
                            UiAction::CancelPauseConfirm
                        } else {
                            UiAction::Resume
                        };
                        self.menu_actions.push(action);
                    }
                    OverlayMenu::Credits | OverlayMenu::Stats => {
                        self.menu_actions.push(UiAction::CloseOverlay);
                    }
                    OverlayMenu::None if pad_back_submenu => {
                        self.menu_actions.push(UiAction::ClosePlaySubmenu);
                    }
                    _ => {}
                }
            }
            let menu_gamepad =
                offer_open || matches!(overlay, OverlayMenu::Pause | OverlayMenu::Settings);
            let menu_cycle = if menu_gamepad {
                input.take_cycle_weapon()
            } else {
                0
            };
            let menu_slot = if menu_gamepad {
                input.take_weapon_slot()
            } else {
                None
            };
            if menu_gamepad {
                let (dv, dh) = pads.iter().fold((0_i8, 0_i8), |(value, horizontal), pad| {
                    (
                        value.saturating_add(
                            i8::from(pad.dpad_down_pressed)
                                .saturating_sub(i8::from(pad.dpad_up_pressed)),
                        ),
                        horizontal.saturating_add(
                            i8::from(pad.dpad_right_pressed)
                                .saturating_sub(i8::from(pad.dpad_left_pressed)),
                        ),
                    )
                });
                if dv != 0 || dh != 0 {
                    input.push_menu_nav(dv, dh);
                }
            }
            sample_gamepads_mapped(&pads, Some(&keymap), &mut input);
            if menu_gamepad {
                let _ = input.take_cycle_weapon();
                let _ = input.take_weapon_slot();
                if menu_cycle != 0 {
                    input.cycle_weapon(menu_cycle);
                }
                if let Some(slot) = menu_slot {
                    input.select_weapon(slot);
                }
            }
            {
                // Contacts were snapshotted alongside the clicks above
                // (`touch_contacts_snap`); re-calling `touch_contacts` here
                // would clear `touch_new` and hand the sampler dead edges.
                // Viewport lifts bypass `App::touch_up` (staging-only
                // closure), so any contact-count drop is a lift: latch the
                // missing ids against the stick claims
                // (`note_touch_released` drops stale ids) and run the
                // sampler on every tick, empty or not, so claims release and
                // lift edges fire on the lift frame.
                let contacts = touch_contacts_snap.clone();
                let live: std::collections::HashSet<i64> =
                    contacts.iter().map(|c| c.id as i64).collect();
                // GML spawns touch objects on mobile only; on desktop the sampler must
                // not run - it zeroes `move_axis`/`aim_axis` and would eat
                // the keyboard vote every frame. Run only while fingers are
                // down, a claim is live, or a lift edge is pending; the lift
                // frame still runs once (prior ids / latched lifts) so claims
                // release and edges fire.
                let had = !self.last_touch_ids.is_empty();
                for id in self.last_touch_ids.drain(..) {
                    if !live.contains(&id) {
                        input.note_touch_released(id);
                    }
                }
                self.last_touch_ids = live.into_iter().collect();
                let mut still_down = Vec::with_capacity(touch_menu_positions.len());
                for (id, position) in touch_menu_positions.drain(..) {
                    if self.last_touch_ids.contains(&id) {
                        still_down.push((id, position));
                    } else {
                        released_touch_clicks.push((id, position));
                    }
                }
                touch_menu_positions = still_down;
                let need_touch = !contacts.is_empty()
                    || had
                    || !input.touch_lifted.is_empty()
                    || input.move_stick.is_some_and(|s| s.touch >= 0)
                    || input.attack_stick.is_some_and(|s| s.touch >= 0);
                if need_touch {
                    // GML `device_mouse_*_to_gui` reads GUI px (view px, e.g. 320x240 on
                    // this phone), not css-dp: convert the dp-space contacts
                    // through the live frame (which also subtracts the
                    // pillarbox offset) before the sampler compares them to
                    // the stick/button homes.
                    let gui_contacts: Vec<crate::input::TouchContact> = contacts
                        .iter()
                        .map(|c| {
                            let start = frame.dp_to_gui([c.start.x, c.start.y]);
                            let pos = frame.dp_to_gui([c.pos.x, c.pos.y]);
                            crate::input::TouchContact {
                                id: c.id,
                                start: Vec2::new(start[0], start[1]),
                                pos: Vec2::new(pos[0], pos[1]),
                                just_pressed: c.just_pressed,
                            }
                        })
                        .collect();
                    let gui_width = frame.gui_width();
                    crate::input::sample_touch_full(
                        &gui_contacts,
                        gui_width,
                        touch_scale,
                        touch_split_fire,
                        touch_stick_regions,
                        &mut input,
                    );
                }
            }
        }
        self.touch_menu_positions = touch_menu_positions;
        // Right-click and Shift share the `spec` action (GML `spec`
        // on `mb_right` plus the Shift keyboard fallback); on Title
        // that action toggles loadout/hardmode, so a right-click
        // would toggle panels as a side effect. Drop the pulse -
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
        // arrows/WASD/digits never stage fire/interact/spec pulses, so without
        // this a keyboard-only shell stalls on the logo until the timeout. Any
        // `just` key, pad edge, or tap counts - edge split like GML
        // `mouse_ui_clicked`: Android on the lift, desktop on the press.
        if state == AppState::Splash
            && (!just.is_empty()
                || had_pause
                || had_restart
                || had_interact
                || (!cfg!(target_os = "android") && !staging_clicks.is_empty())
                || (cfg!(target_os = "android") && !released_touch_clicks.is_empty()))
        {
            self.sim.world.resource_mut::<NtInput>().press_interact();
        }

        // Per-frame cursor aim (bevy `player_aim` mouse path): the cursor's
        // *screen* position unprojected through the current camera steers
        // `aim_axis` every tick, not just on hover events - a latched world
        // point would go stale as the camera moves. The follow camera reads the
        // same live point's distance as GML `dis_fire`. Stick input wins when
        // nonzero (bevy precedence: `sample_keyboard` leaves `aim_axis` zero, a
        // gamepad shell may layer on top). Frozen outside live play.
        // Clicks: aim-at-click + fire in live game, confirm in menus; a click is
        // a single-frame edge (continuous hold needs shell pressed-state). Over
        // an open pause/settings/credits menu a click hit-tests the
        // screen-anchored buttons by dp position (independent of the repose
        // hit-test); a stray click stages nothing, so it can never fire the gun
        // or resume through a MENU press.
        if live_play {
            let player_pos = self
                .sim
                .world
                .query::<(&Pos, &Player)>()
                .iter(&self.sim.world)
                .next()
                .map(|(p, _)| p.0);
            if let Some(pp) = player_pos {
                // Live cursor through this frame's camera: valid even when the pointer
                // hasn't moved since the camera did. Desktop-only - on touch
                // there is no cursor (aim comes from the attack stick; staged
                // hover is a finger drag, not a pointer).
                if !touch_only_device {
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
                }
                if let Some(click) = staging_clicks.last().copied() {
                    // Desktop click-to-fire only. On a touch device a bare tap is
                    // NOTHING: GML gameplay never reads a tap point
                    // (`JoystickAttack` aims/fires solely off its claimed
                    // stick; taps advance only splash/menus via
                    // `mouse_ui_clicked`). The tap already fed the sampler
                    // as a rebuilt contact above, so stick zones still claim
                    // taps that land on them.
                    if !touch_only_device {
                        let mut input = self.sim.world.resource_mut::<NtInput>();
                        let dir = click.world - pp;
                        if dir.length_squared() > 1e-6 {
                            input.aim_axis = dir.normalize_or_zero();
                        }
                        input.fire_held = true;
                        input.press_fire();
                    }
                }
            }
            // In-run pause button (GML `UberCont/Draw_64:51-60`): the only pause
            // route a touch device has (no Esc key), and the only one a mouse
            // gets either. GML loops `device_mouse_check_button_released`
            // over touches 0..4 (released contacts on every platform), so a
            // press in the circle that lifts pauses. The port splits the
            // edge like GML `mouse_ui_clicked`: Android reads the lifts
            // (`released_touch_clicks`), desktop stays on `staging_clicks`
            // (pointer-DOWN presses, not releases - known GML divergence).
            {
                let frame = self.gml_frame();
                let vw = frame.gui_width();
                let clicked = !cfg!(target_os = "android")
                    && staging_clicks.last().is_some_and(|c| {
                        let g = frame.dp_to_gui(c.dp);
                        crate::input::pause_button_hit(g[0], g[1], vw)
                    });
                let tapped = cfg!(target_os = "android")
                    && released_touch_clicks.iter().any(|(_, p)| {
                        let g = frame.dp_to_gui([p.x, p.y]);
                        crate::input::pause_button_hit(g[0], g[1], vw)
                    });
                if clicked || tapped {
                    self.sim.world.resource_mut::<MenuEdge>().pause_pressed = true;
                    released_touch_clicks.clear();
                }
            }
        } else if menu_open && !offer_open {
            // Open menu over a live run: right-click steps back (GML
            // `BackButton` `mb_right` parity - Settings pops one level
            // via `SettingsBack`, Credits closes), left clicks route at
            // the menu buttons, then drop either way. A coincident left
            // click still routes (RMB never eats it).
            if rmb_down && overlay == OverlayMenu::Settings {
                apply_menu_action(&mut self.sim.world, UiAction::SettingsBack);
            } else if rmb_down && overlay == OverlayMenu::Credits {
                apply_menu_action(&mut self.sim.world, UiAction::CloseOverlay);
            } else if !cfg!(target_os = "android")
                && !slider_click_consumed
                && let Some(click) = staging_clicks.last().copied()
            {
                let frame = self.gml_frame();
                let kind = menu_overlay_kind(
                    state,
                    overlay,
                    &self.sim.world.resource::<MenuState>(),
                    game_over,
                );
                if let Some(kind) = kind {
                    if let Some(action) = route_menu_click(
                        &mut self.sim.world,
                        kind,
                        frame.dp_to_gui(click.dp),
                        frame.gui_width(),
                    ) {
                        apply_menu_action(&mut self.sim.world, action);
                    }
                }
            }
            if cfg!(target_os = "android") && !released_touch_clicks.is_empty() {
                let frame = self.gml_frame();
                let kind = menu_overlay_kind(
                    state,
                    overlay,
                    &self.sim.world.resource::<MenuState>(),
                    game_over,
                );
                for (id, point) in released_touch_clicks.drain(..) {
                    if slider_touch_ids.contains(&id) {
                        continue;
                    }
                    let gui = frame.dp_to_gui([point.x, point.y]);
                    let action = if kind == Some(MenuOverlay::Mutation) {
                        crate::render::mutation_icon_hit_action(
                            &mut self.sim.world,
                            gui[0],
                            gui[1],
                            frame.gui_width(),
                        )
                    } else {
                        kind.and_then(|kind| {
                            route_menu_click(&mut self.sim.world, kind, gui, frame.gui_width())
                        })
                    };
                    if let Some(action) = action {
                        apply_menu_action(&mut self.sim.world, action);
                        break;
                    }
                }
            }
        } else if game_over {
            if cfg!(target_os = "android") {
                let frame = self.gml_frame();
                for (id, point) in released_touch_clicks.drain(..) {
                    if slider_touch_ids.contains(&id) {
                        continue;
                    }
                    let gui = frame.dp_to_gui([point.x, point.y]);
                    if let Some(action) = route_menu_click(
                        &mut self.sim.world,
                        MenuOverlay::GameOver,
                        gui,
                        frame.gui_width(),
                    ) {
                        apply_menu_action(&mut self.sim.world, action);
                        break;
                    }
                }
            } else if let Some(click) = staging_clicks.last().copied() {
                let frame = self.gml_frame();
                if let Some(action) = route_menu_click(
                    &mut self.sim.world,
                    MenuOverlay::GameOver,
                    frame.dp_to_gui(click.dp),
                    frame.gui_width(),
                ) {
                    apply_menu_action(&mut self.sim.world, action);
                }
            }
        } else if state == AppState::MainMenu {
            // Settings/Credits/Stats open over the buttons (GML MenuOptions
            // / DrawStats parity): route through the live overlay kind so
            // mouse works on those pages, not just the keyboard. RMB steps
            // back (Settings pops a level, others close).
            if rmb_down && overlay == OverlayMenu::Settings {
                apply_menu_action(&mut self.sim.world, UiAction::SettingsBack);
            } else if rmb_down && matches!(overlay, OverlayMenu::Credits | OverlayMenu::Stats) {
                apply_menu_action(&mut self.sim.world, UiAction::CloseOverlay);
            } else if !cfg!(target_os = "android")
                && !slider_click_consumed
                && let Some(click) = staging_clicks.last().copied()
            {
                let frame = self.gml_frame();
                let kind = menu_overlay_kind(
                    state,
                    overlay,
                    &self.sim.world.resource::<MenuState>(),
                    game_over,
                )
                .unwrap_or(MenuOverlay::MainMenu);
                if let Some(action) = route_menu_click(
                    &mut self.sim.world,
                    kind,
                    frame.dp_to_gui(click.dp),
                    frame.gui_width(),
                ) {
                    apply_menu_action(&mut self.sim.world, action);
                }
            } else if cfg!(target_os = "android") && !released_touch_clicks.is_empty() {
                let frame = self.gml_frame();
                let kind = menu_overlay_kind(
                    state,
                    overlay,
                    &self.sim.world.resource::<MenuState>(),
                    game_over,
                )
                .unwrap_or(MenuOverlay::MainMenu);
                for (id, point) in released_touch_clicks.drain(..) {
                    if slider_touch_ids.contains(&id) {
                        continue;
                    }
                    let gui = frame.dp_to_gui([point.x, point.y]);
                    if let Some(action) =
                        route_menu_click(&mut self.sim.world, kind, gui, frame.gui_width())
                    {
                        apply_menu_action(&mut self.sim.world, action);
                        break;
                    }
                }
            }
        } else if state == AppState::Title {
            // Settings/Credits open over the campfire: route those through
            // the menu router (mouse + RMB-back); otherwise pods / GO /
            // loadout zones (GML campfire parity, strays do nothing).
            if rmb_down && overlay == OverlayMenu::Settings {
                apply_menu_action(&mut self.sim.world, UiAction::SettingsBack);
            } else if rmb_down && overlay == OverlayMenu::Credits {
                apply_menu_action(&mut self.sim.world, UiAction::CloseOverlay);
            } else if !cfg!(target_os = "android")
                && !slider_click_consumed
                && let Some(click) = staging_clicks.last().copied()
            {
                let frame = self.gml_frame();
                if overlay == OverlayMenu::Settings || overlay == OverlayMenu::Credits {
                    let kind = menu_overlay_kind(
                        state,
                        overlay,
                        &self.sim.world.resource::<MenuState>(),
                        game_over,
                    );
                    if let Some(kind) = kind
                        && let Some(action) = route_menu_click(
                            &mut self.sim.world,
                            kind,
                            frame.dp_to_gui(click.dp),
                            frame.gui_width(),
                        )
                    {
                        apply_menu_action(&mut self.sim.world, action);
                    }
                } else if let Some(action) =
                    self.route_title_click(frame.dp_to_gui(click.dp), frame.gui_width())
                {
                    apply_menu_action(&mut self.sim.world, action);
                }
            } else if cfg!(target_os = "android") && !released_touch_clicks.is_empty() {
                let frame = self.gml_frame();
                let kind = if overlay == OverlayMenu::Settings || overlay == OverlayMenu::Credits {
                    menu_overlay_kind(
                        state,
                        overlay,
                        &self.sim.world.resource::<MenuState>(),
                        game_over,
                    )
                } else {
                    None
                };
                for (id, point) in released_touch_clicks.drain(..) {
                    if slider_touch_ids.contains(&id) {
                        continue;
                    }
                    let gui = frame.dp_to_gui([point.x, point.y]);
                    let action =
                        if overlay == OverlayMenu::Settings || overlay == OverlayMenu::Credits {
                            kind.and_then(|kind| {
                                route_menu_click(&mut self.sim.world, kind, gui, frame.gui_width())
                            })
                        } else {
                            self.route_title_click(gui, frame.gui_width())
                        };
                    if let Some(action) = action {
                        apply_menu_action(&mut self.sim.world, action);
                        break;
                    }
                }
            }
        } else if offer_open
            && menu_overlay_kind(
                state,
                overlay,
                &self.sim.world.resource::<MenuState>(),
                game_over,
            ) == Some(MenuOverlay::Mutation)
        {
            // Mutation/ultra offer: right-button is silent (never confirms
            // or eats the left click).
            //
            // GML `SkillIcon/Mouse_4` is ONE left-press event per icon:
            // select when unselected, commit (`event_user(0)`) when
            // already selected. The port models that as
            // `SelectMutation`/`PickMutation` and the two-stage law reads
            // live `mutation_selected`, so routing ONE gesture through the
            // hit-test twice selects then instantly claims in the same frame
            // (the "mutation collects on the first tap" bug). A touch finger
            // arrives as both a released contact and a pointer click, so
            // Android routes only the lifts and the press click is the
            // desktop path (GML `mouse_ui_clicked`: mobile reads release,
            // desktop reads press).
            let frame = self.gml_frame();
            let mut routed = false;
            if cfg!(target_os = "android") {
                for (id, point) in released_touch_clicks.drain(..) {
                    if slider_touch_ids.contains(&id) {
                        continue;
                    }
                    let gui = frame.dp_to_gui([point.x, point.y]);
                    if let Some(action) = crate::render::mutation_icon_hit_action(
                        &mut self.sim.world,
                        gui[0],
                        gui[1],
                        frame.gui_width(),
                    ) {
                        apply_menu_action(&mut self.sim.world, action);
                        routed = true;
                        break;
                    }
                }
            }
            if !routed
                && !cfg!(target_os = "android")
                && let Some(click) = staging_clicks.last().copied()
            {
                let gui = frame.dp_to_gui(click.dp);
                if let Some(action) = crate::render::mutation_icon_hit_action(
                    &mut self.sim.world,
                    gui[0],
                    gui[1],
                    frame.gui_width(),
                ) {
                    apply_menu_action(&mut self.sim.world, action);
                }
            }
        } else if let Some(_click) = staging_clicks.last().copied() {
            let frame = self.gml_frame();
            for (id, point) in released_touch_clicks {
                if slider_touch_ids.contains(&id) {
                    continue;
                }
                let gui = frame.dp_to_gui([point.x, point.y]);
                if let Some(action) = crate::render::mutation_icon_hit_action(
                    &mut self.sim.world,
                    gui[0],
                    gui[1],
                    frame.gui_width(),
                ) {
                    apply_menu_action(&mut self.sim.world, action);
                    break;
                }
            }
        } else if let Some(_click) = staging_clicks.last().copied() {
            // Splash/Loading advance on any mouse button (bevy `boot_intro`
            // any-key/mouse law).

            self.sim.world.resource_mut::<NtInput>().press_interact();
        } else {
        }
    }

    /// Title click router with asset-aware geometry (same native sizes
    /// the sprite layer draws with, else the 20px fallback). `gui` and
    /// `vw` are GML view px.
    fn route_title_click(&mut self, gui: [f32; 2], vw: f32) -> Option<UiAction> {
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
            gui[0],
            gui[1],
            vw,
            slot_h,
            crownsize,
            skinsize,
        )
    }

    /// The live GML frame (see [`GmlFrame`](crate::render::GmlFrame)):
    /// the one place the world rect, the pillarbox box, and the dp→GUI
    /// scale come from.
    pub fn gml_frame(&self) -> crate::render::GmlFrame {
        crate::render::gml_frame(self.view_viewport_dp, self.view_world_size, &self.cam)
    }

    /// GUI-space cursor for the menu hover sync: polled window-physical px
    /// mapped into the GML view. `None` until the first mouse move, or while
    /// a touch holds the device (GML `mouse_ui_hovered` is a bare collision
    /// test - the pointer keeps driving `pointed_item` under the GAMEPAD
    /// switch, only `MainMenuButton`'s hover flag is `!is_gamepad()`-gated -
    /// and touch has no cursor).
    fn menu_gui_point(&self) -> Option<[f32; 2]> {
        if !self.staging.borrow().touch_active.is_empty() {
            return None;
        }
        let px = self.polled_pointer_px?;
        let d = self.view_density.max(1e-6);
        if !d.is_finite() || d <= 0.0 {
            return None;
        }
        Some(self.gml_frame().dp_to_gui([px.x / d, px.y / d]))
    }

    fn settings_gui_point_dp(&self, dp: [f32; 2]) -> Option<[f32; 2]> {
        Some(self.gml_frame().dp_to_gui(dp))
    }

    fn settings_slider_current(&self, target: SettingSliderTarget) -> Option<f32> {
        let save = self
            .sim
            .world
            .get_resource::<crate::savedata_part::SaveData>()?;
        Some(match target {
            SettingSliderTarget::Volume(crate::render::VolumeChannel::Master) => {
                save.settings.master_volume
            }
            SettingSliderTarget::Volume(crate::render::VolumeChannel::Music) => {
                save.settings.music_volume
            }
            SettingSliderTarget::Volume(crate::render::VolumeChannel::Ambience) => {
                save.settings.ambience_volume
            }
            SettingSliderTarget::Volume(crate::render::VolumeChannel::Sfx) => {
                save.settings.sfx_volume
            }
            SettingSliderTarget::Slider("screenshake") => save.settings.screenshake,
            SettingSliderTarget::Slider("freezeframes") => save.settings.freezeframes,
            SettingSliderTarget::Slider("controls_scale") => save.settings.controls_scale,
            SettingSliderTarget::Slider(_) => 0.0,
        })
    }

    fn apply_settings_slider_live(&mut self, target: SettingSliderTarget, value: f32) {
        let value = match target {
            SettingSliderTarget::Volume(_) => value.clamp(0.0, 1.0),
            SettingSliderTarget::Slider("screenshake") => value.clamp(0.0, 2.0),
            SettingSliderTarget::Slider(_) => value.clamp(0.0, 1.0),
        };
        self.sim
            .world
            .init_resource::<crate::savedata_part::SaveData>();
        {
            let mut save = self
                .sim
                .world
                .resource_mut::<crate::savedata_part::SaveData>();
            match target {
                SettingSliderTarget::Volume(crate::render::VolumeChannel::Master) => {
                    save.settings.master_volume = value;
                }
                SettingSliderTarget::Volume(crate::render::VolumeChannel::Music) => {
                    save.settings.music_volume = value;
                }
                SettingSliderTarget::Volume(crate::render::VolumeChannel::Ambience) => {
                    save.settings.ambience_volume = value;
                }
                SettingSliderTarget::Volume(crate::render::VolumeChannel::Sfx) => {
                    save.settings.sfx_volume = value;
                }
                SettingSliderTarget::Slider("screenshake") => save.settings.screenshake = value,
                SettingSliderTarget::Slider("freezeframes") => save.settings.freezeframes = value,
                SettingSliderTarget::Slider("controls_scale") => {
                    save.settings.controls_scale = value;
                }
                SettingSliderTarget::Slider(_) => {}
            }
        }
        if let SettingSliderTarget::Volume(channel) = target {
            self.sim
                .world
                .init_resource::<crate::audio::AudioChannels>();
            let mut channels = self.sim.world.resource_mut::<crate::audio::AudioChannels>();
            match channel {
                crate::render::VolumeChannel::Master => channels.master = value,
                crate::render::VolumeChannel::Music => channels.music = value,
                crate::render::VolumeChannel::Sfx => channels.sfx = value,
                crate::render::VolumeChannel::Ambience => {}
            }
        }
        self.sim.world.init_resource::<crate::comps_a::SaveDirty>();
        self.sim.world.resource_mut::<crate::comps_a::SaveDirty>().0 = true;
    }

    fn begin_settings_slider_drag(
        &mut self,
        page: u8,
        target: SettingSliderTarget,
        pointer: SettingsSliderPointer,
        gx: f32,
        vw: f32,
    ) {
        let value = settings_slider_value(target, gx, vw);
        self.apply_settings_slider_live(target, value);
        self.settings_slider_drag = Some(SettingsSliderDrag {
            page,
            target,
            pointer,
        });
        emit_sfx(
            &mut self.sim.world,
            crate::audio::AudioCue {
                name: "sndSlider",
                volume: 1.0,
                variance: 0.0,
            },
        );
    }

    fn update_settings_slider_drag(&mut self, gx: f32, vw: f32) {
        let Some(drag) = self.settings_slider_drag else {
            return;
        };
        let value = settings_slider_value(drag.target, gx, vw);
        if self
            .settings_slider_current(drag.target)
            .is_none_or(|current| (current - value).abs() > 0.0001)
        {
            self.apply_settings_slider_live(drag.target, value);
        }
    }

    fn finish_settings_slider_drag(&mut self) {
        if self.settings_slider_drag.take().is_some() {
            emit_sfx(
                &mut self.sim.world,
                crate::audio::AudioCue {
                    name: "sndSliderLetGo",
                    volume: 1.0,
                    variance: 0.0,
                },
            );
        }
    }

    /// GML `mouse_ui_hovered` Step parity, once per `feed_input` before any
    /// click routes: each visible menu owns per-instance `hover` state the
    /// mouse sets by collision and clears on leave, with `sndHover` on entry
    /// (`MainMenuButton/Step_0`, `PlayButton/Step_0`, `PauseButton/Step_0`,
    /// `CharSelect/Draw_0`, `SkillIcon/Mouse_4`, `GoButton/Draw_0`,
    /// `MenuOptions/Other_10` `pointed_item`). Keyboard/gamepad paths never
    /// write here - gamepad owns `gamepad_sel` via `scrGamepadUIControl`,
    /// keyboard owns the `*_cursor` fields.
    fn tick_menu_hover(&mut self) {
        let Some([gx, gy]) = self.menu_gui_point() else {
            return;
        };
        let vw = crate::render::gml_view_size(self.view_viewport_dp)[0];
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
        let game_over = crate::state::menus::game_over_visible(&self.sim.world);
        let kind = menu_overlay_kind(
            state,
            overlay,
            &self.sim.world.resource::<MenuState>(),
            game_over,
        );
        let Some(kind) = kind else { return };
        match kind {
            // GML main-menu rows are separate `MainMenuButton` instances
            // (one per label); the PLAY submenu swaps them for
            // `PlayButton` instances. The mouse points rows by collision
            // (`mouse_ui_hovered`), the keyboard mirrors `gamepad_sel`
            // into the cursor - both highlight through the same color.
            MenuOverlay::MainMenu => {
                let in_submenu = self
                    .sim
                    .world
                    .get_resource::<MenuState>()
                    .is_some_and(|m| m.play_submenu);
                let rows = crate::render::menu_gui_texts_vw(kind, &mut self.sim.world, vw);
                for (i, t) in rows.iter().enumerate() {
                    if menu_button_action(kind, &t.text, None).is_none() {
                        continue;
                    }
                    let (hx, hy) = if t.text == "BACK" {
                        (16.0, 20.0)
                    } else {
                        (t.gx, t.gy)
                    };
                    if (gx - hx).abs() <= 60.0 && (gy - hy).abs() <= 11.0 {
                        let update = {
                            let menu = self.sim.world.resource::<MenuState>();
                            if in_submenu {
                                menu.play_cursor != i
                            } else {
                                menu.main_menu_cursor != i
                            }
                        };
                        if update {
                            if let Some(mut menu) = self.sim.world.get_resource_mut::<MenuState>() {
                                if in_submenu {
                                    menu.play_cursor = i;
                                } else {
                                    menu.main_menu_cursor = i;
                                }
                            }
                            crate::state::menus::emit_hover(&mut self.sim.world);
                        }
                        break;
                    }
                }
            }
            // GML `MenuOptions/Other_10`: while `mouse_active` an
            // available row owns `pointed_item` by `point_in_rectangle`
            // (`Other_10.gml:597` gates on `_opt.available`); keyboard
            // arrows clear `mouse_active` and own it instead.
            MenuOverlay::Settings => {
                let page = self
                    .sim
                    .world
                    .get_resource::<MenuState>()
                    .map(|m| m.settings_page)
                    .unwrap_or(0);
                let back_x = if cfg!(target_os = "android") {
                    24.0
                } else {
                    16.0
                };
                let back_hover =
                    gx >= back_x - 20.0 && gx <= back_x + 20.0 && gy >= 0.0 && gy <= 40.0;
                if let Some(mut menu) = self.sim.world.get_resource_mut::<MenuState>() {
                    menu.settings_back_hover = back_hover;
                }
                let rows = crate::render::settings_hot_rows(page, vw);
                let hit = rows.iter().enumerate().find(|(_, r)| {
                    (gy - r.gy).abs() <= 7.0
                        && (gx - r.cx).abs() <= r.hw
                        && crate::render::settings_row_available(&self.sim.world, page, r)
                });
                if let Some((idx, _)) = hit {
                    let cur = self
                        .sim
                        .world
                        .get_resource::<MenuState>()
                        .map(|m| m.settings_cursor)
                        .unwrap_or(0);
                    if cur != idx {
                        if let Some(mut menu) = self.sim.world.get_resource_mut::<MenuState>() {
                            menu.settings_cursor = idx;
                        }
                        crate::state::menus::emit_hover(&mut self.sim.world);
                    }
                }
            }
            // GML `CharSelect/Draw_0`: `point_in_rectangle` on the pod
            // bbox syncs `selected` to the pointed pod and raises the
            // name tooltip; leaving every bbox clears it.
            MenuOverlay::Title => {
                if overlay == OverlayMenu::Settings || overlay == OverlayMenu::Credits {
                    return;
                }
                let roster = crate::state::menus::visible_roster(
                    self.sim
                        .world
                        .get_resource::<crate::savedata_part::SaveData>(),
                );
                let slot_h = self
                    .assets
                    .as_ref()
                    .and_then(|a| a.native_size("images/sprCharSelect.png").map(|s| s.y))
                    .unwrap_or(20.0);
                let pods = crate::render::char_pod_layout([vw, 240.0], roster.len(), slot_h);
                let mut pointed: Option<usize> = None;
                for (i, pos) in pods.iter().enumerate() {
                    if gx >= pos[0]
                        && gx <= pos[0] + crate::render::TITLE_POD_W
                        && gy >= pos[1]
                        && gy <= pos[1] + crate::render::TITLE_POD_H
                    {
                        pointed = Some(i);
                        break;
                    }
                }
                match pointed {
                    Some(i) => {
                        let cur = self
                            .sim
                            .world
                            .get_resource::<MenuState>()
                            .map(|m| m.title_cursor)
                            .unwrap_or(0);
                        if cur != i {
                            if let Some(mut menu) = self.sim.world.get_resource_mut::<MenuState>() {
                                menu.title_cursor = i;
                                menu.title_pod_pointed = true;
                            }
                            crate::state::menus::emit_hover(&mut self.sim.world);
                        } else if let Some(mut menu) =
                            self.sim.world.get_resource_mut::<MenuState>()
                        {
                            menu.title_pod_pointed = true;
                        }
                    }
                    None => {
                        if let Some(mut menu) = self.sim.world.get_resource_mut::<MenuState>() {
                            menu.title_pod_pointed = false;
                        }
                    }
                }
            }
            // GML `SkillIcon/Mouse_4`: first click on an unselected card
            // only highlights (with `sndHover`); the second commits.
            // Pointing here mirrors the highlight half so the render
            // layer shows it before any click.
            MenuOverlay::Mutation => {
                let n = self
                    .sim
                    .world
                    .get_resource::<crate::comps_a::PendingUltra>()
                    .map(|u| u.choices.len())
                    .or_else(|| {
                        self.sim
                            .world
                            .get_resource::<crate::comps_a::PendingMutation>()
                            .map(|p| p.choices.len())
                    })
                    .unwrap_or(0);
                if n == 0 {
                    return;
                }
                let is_ultra = self
                    .sim
                    .world
                    .get_resource::<crate::comps_a::PendingUltra>()
                    .is_some();
                let step = (vw / (n as f32 + 1.0)).floor().min(32.0);
                let scale = if is_ultra {
                    1.0
                } else {
                    (step / 32.0).max(0.65)
                };
                let half = (step as i32 / 2) as f32;
                let xview_shift = if n >= 10 { -12.0 } else { 0.0 };
                let start_x = vw * 0.5 + xview_shift - (n as f32 - 1.0) * half;
                let icon_y = 240.0 - 21.0;
                let hw = 24.0 * scale * 0.5;
                let menu = self.sim.world.resource::<MenuState>().clone();
                let mut pointed = None;
                for i in 0..n {
                    let cx = start_x + i as f32 * step;
                    let card_y = icon_y + menu.mutation_appear_y.get(i).copied().unwrap_or(0.0)
                        - if menu.mutation_selected == Some(i) {
                            1.0
                        } else {
                            0.0
                        };
                    if !is_ultra
                        && menu
                            .mutation_appear_y
                            .get(i)
                            .is_some_and(|value| *value > 0.01)
                    {
                        continue;
                    }
                    let top = card_y - 16.0 * scale;
                    let hh = 32.0 * scale;
                    if (gx - cx).abs() <= hw && gy >= top && gy <= top + hh {
                        pointed = Some(i);
                        break;
                    }
                }
                match pointed {
                    Some(i) if menu.mutation_selected != Some(i) => {
                        crate::state::menus::set_mutation_selection(&mut self.sim.world, i);
                        crate::state::menus::emit_hover(&mut self.sim.world);
                    }
                    None => crate::state::menus::clear_mutation_selection(&mut self.sim.world),
                    _ => {}
                }
            }
            // GML `PauseButton/Step_0`: same collision hover law as the
            // main menu (entry stings `sndHover`); the port's rows are
            // stateless so the sting gates on the pointed label.
            MenuOverlay::Pause | MenuOverlay::GameOver => {
                let rows = crate::render::menu_gui_texts_vw(kind, &mut self.sim.world, vw);
                let confirm = self
                    .sim
                    .world
                    .get_resource::<MenuState>()
                    .and_then(|m| m.pause_confirm);
                let mut pointed = false;
                for t in &rows {
                    if menu_button_action(kind, &t.text, confirm).is_none() {
                        continue;
                    }
                    if (gx - t.gx).abs() <= 60.0 && (gy - t.gy).abs() <= 11.0 {
                        pointed = true;
                        if kind == MenuOverlay::Pause {
                            let index = match t.text.as_str() {
                                "MENU" | "BACK" => 0,
                                "RETRY" => 1,
                                "SETTINGS" => 2,
                                "CONTINUE" => 3,
                                "QUIT" => 1,
                                _ => 0,
                            };
                            if let Some(mut menu) = self.sim.world.get_resource_mut::<MenuState>() {
                                menu.pause_cursor = index;
                            }
                        }
                        crate::state::menus::emit_hover_if_changed(&mut self.sim.world, &t.text);
                        break;
                    }
                }
                if !pointed && let Some(mut menu) = self.sim.world.get_resource_mut::<MenuState>() {
                    menu.hover_label.clear();
                }
            }
            MenuOverlay::Unlock => {
                let band = crate::state::menus::UNLOCK_CONTINUE_BAND;
                let (can_continue, was_pointed) = self
                    .sim
                    .world
                    .get_resource::<MenuState>()
                    .map(|m| (m.unlock.can_continue, m.unlock.pointed))
                    .unwrap_or((false, false));
                let pointed = can_continue && gy >= band;
                if let Some(mut menu) = self.sim.world.get_resource_mut::<MenuState>() {
                    menu.unlock.pointed = pointed;
                }
                if pointed != was_pointed {
                    crate::state::menus::emit_hover(&mut self.sim.world);
                }
            }
            _ => {}
        }
    }

    pub fn drain_audio_cues(&mut self) -> Vec<repame_audio::Cue> {
        if let Some(mut q) = self
            .sim
            .world
            .get_resource_mut::<crate::msg::Queue<crate::audio::AudioCue>>()
        {
            q.drain()
                .iter()
                .map(|c| repame_audio::Cue {
                    name: c.name,
                    volume: c.volume,
                    variance: c.variance,
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn area_audio_snapshot(
        &mut self,
    ) -> (
        Option<crate::audio::MusicCue>,
        Option<crate::audio::AmbienceCue>,
        f32,
        f32,
    ) {
        let Some(state) = self
            .sim
            .world
            .get_resource::<crate::audio::AreaAudioState>()
        else {
            return (None, None, 0.0, 0.0);
        };
        (
            state.current_music,
            state.current_ambience,
            state.music_volume,
            state.ambience_volume,
        )
    }

    /// Build this frame's view: stage input, advance the sim, snapshot
    /// sim+render+vortex+HUD+menus into repose views.
    pub fn view(&mut self, sched: &mut Scheduler, _ctx: &RenderContext, dt: Duration) -> View {
        request_frame();
        // Runtime-owned cursor position (GML `mouse_x`/`mouse_y` parity):
        // `ReposeRuntime` maintains it on mouse move/press/release outside
        // focus dispatch, so one poll replaces the root `cursor_move` +
        // viewport `Hover` dual staging (and its 30-frame stale clock).
        self.polled_pointer_px = sched.pointer_pos_px.map(|(x, y)| Vec2::new(x, y));
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
        // GML `UberCont/Step_0:175-183` cursor law, computed from the post-tick
        // sim state: keyboard mode hides the OS cursor (the game draws its own
        // crosshair at the live cursor position), mouse mode shows it. The
        // GML branch reads raw `opt_keyboard` - the `opt_gamepad` switch never
        // restores the pointer. Touch frames always show it (fingers need no
        // cursor, and the GML `opt_keyboard` branch draws nothing on touch).
        let keyboard_mode = {
            let keyboard = self
                .sim
                .world
                .get_resource::<crate::savedata_part::SaveData>()
                .is_some_and(|s| s.settings.keyboard_enabled);
            keyboard && self.staging.borrow().touch_active.is_empty()
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
        // GML `UberCont/Step_0:175-183` has no state carve-outs: from the first
        // splash frame, desktop keyboard mode hides the OS cursor
        // (`opt_keyboard` defaults true on desktop, nothing in boot/menus/
        // death clears it) and the game crosshair (`UberCont/Draw_75`, gated
        // only on `window_get_cursor() == cr_none` + `show_crosshair`) draws
        // over everything. Pause/overlay are the only hides.
        let hide_os_cursor = keyboard_mode && !paused_now && overlay_now == OverlayMenu::None;
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
        // GML `scrSetViewSize` + `display_set_gui_size`: the GUI IS the view, so
        // there is ONE rect. The viewport gets the GML view in world units and
        // a 1:1 camera, and the engine's contain-fit scales it into the canvas
        // and centres the remainder = the letterbox/pillarbox. (The old shape
        // - whole canvas + scaled camera - always filled the window, which is
        // why portrait showed 320x668 of world under a 320x240 GUI.)
        let gml_view = gml_view_size(viewport_dp);
        let world_size = gml_view;
        let gml_scale = 1.0;
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
            .get_resource::<AppState>()
            .is_some_and(|state| *state == AppState::Splash)
            .then_some(AreaId::Campfire)
            .or_else(|| {
                self.sim
                    .world
                    .get_resource::<crate::comps_a::Run>()
                    .map(|r| r.area)
            })
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
        let run_game_over = crate::state::menus::game_over_visible(&self.sim.world);
        let menu_kind = {
            let menu = self.sim.world.resource_mut::<MenuState>();
            menu_overlay_kind(state, overlay, &menu, run_game_over)
        };
        // GML `Vlambeer/Alarm_0`: modes 0-2 are flat boot cards, while
        // mode 4 creates `SpiralCont` + `Logo`; the spiral then persists
        // through the main-menu handoff.
        let splash_mode = self
            .sim
            .world
            .get_resource::<crate::state::SplashState>()
            .map(|splash| splash.mode)
            .unwrap_or(0);
        let splash_logo = menu_kind == Some(MenuOverlay::Splash) && splash_mode >= 4;
        let paused = self
            .sim
            .world
            .get_resource::<crate::state::Paused>()
            .is_some_and(|p| p.0);
        let game_over = crate::state::menus::game_over_visible(&self.sim.world);
        let ft_active = self
            .sim
            .world
            .get_resource::<crate::comps_b::FloorTransition>()
            .is_some_and(|transition| transition.active);
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
        let spiral_cover = state == AppState::Loading || ft_active || pending_pick;

        if matches!(state, AppState::Title) {
            let vw_vh = self.view_world_size;
            let focus = title_cam_focus(&mut self.sim.world).unwrap_or(Vec2::new(64.0, 64.0));
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
                gml_scale,
            );
            self.cam.offset = Vec2::ZERO;
        }

        // Follow camera: the GML `BackCont` look point steps at 30 Hz in
        // [`App::step_camera_fixed`] (frozen over pause/menus/game over, so
        // menus keep the last camera). GML has no zoom - the view is always
        // 240 world px tall (`gml_view_size`, `gml_view_scale` units per px),
        // snapped on room start by `gml_cam.snap`. `gml_cam` tracks the
        // top-left corner; the GPU camera centers the look point.
        let center = Vec2::new(
            self.gml_cam.x + gml_view[0] * 0.5,
            self.gml_cam.y + gml_view[1] * 0.5,
        );
        self.cam = world_camera(center, gml_scale);
        self.cam.offset = Vec2::ZERO;

        // Sprite layer: real instances with assets, placeholder quads without; the
        // batch is the headless-verifiable artifact (`Viewport2dGpu` rebuilds
        // its own internally). Hardware cursor request: computed inside the
        // sprite borrow (save reads only), refreshed after it ends
        // (`refresh_cursor_img` needs `&mut self`); `None` when assets are
        // absent or the gate is off.
        let mut cursor_req: Option<(i32, [f32; 4])> = None;
        let mut chrome_sprites: Vec<SpriteInstance> = Vec::new();
        let mut front_chrome_sprites: Vec<SpriteInstance> = Vec::new();
        let mut spiral_figure_sprites: Vec<SpriteInstance> = Vec::new();
        let mut cover_hud_sprites: Vec<SpriteInstance> = Vec::new();
        let (mut sprites, texts) = if self.assets.is_some() {
            let assets = self.assets.as_ref().expect("checked");
            // Cursor world position for the GML crosshair distance
            // (`KeyCont.dis_fire` parity; `None` until the first hover).
            // Uses the live cursor unprojection when available so the
            // crosshair tracks the on-screen cursor as the camera moves
            // (same source as aim; falls back to the last Hover).
            self.sim
                .world
                .insert_resource(crate::render::HoverWorld(self.live_cursor_world()));
            // Blob shadows first (GML `shad` surface: under the actors).
            // Every layer stamps its z-ladder rung (render.rs `Z_*`, GML
            // `__global_object_depths` order): without rungs every sprite shares
            // z=0 and the engine's `(blend, z, page)` sort lets atlas page
            // lottery HUD bars under floor tiles. Push order matches rung
            // order, so the canvas path (push-ordered) and the GPU path
            // (z-sorted) agree.
            // GML `scrGameIsGenerationScreen` verbatim: while the generation
            // conts own the draw (`GenCont` behind Loading, `LevCont` behind
            // the mutation/ultra offer) the room draws NOTHING but the spiral
            // + cover text - the old floor sits under an opaque vortex backdrop
            // and would paint through the fullscreen pass (the "tiles over the
            // vortex" bug). The campfire title is NOT a generation screen for
            // draw purposes: `MenuGen` builds the camp, then `Menu` (whose draw
            // scripts own the chrome, not `TopCont`) draws spiral remnant +
            // camp + pods + portraits, so TopCont-sourced HUD bars stay off
            // but the Menu chrome stays on.
            let generation_screen = spiral_cover
                || matches!(
                    menu_kind,
                    Some(MenuOverlay::Loading) | Some(MenuOverlay::Mutation)
                );
            let playing = !generation_screen;
            let mut s = if playing {
                shadow_sprites(&mut self.sim.world, assets)
            } else {
                Vec::new()
            };
            stamp_z(&mut s, Z_SHADOW);
            let mut w = if playing {
                world_instances_cached(&mut self.sim.world, assets, &mut self.static_world_cache)
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
            // World crosshair (GML `TopCont/Draw_0`, over the room, under the
            // HUD text) plus coop fainted bars at the view-clamped positions.
            // All three world-HUD passes (`with Player` crosshair, `Revive`
            // bars, `Portal` arrow) require live run actors; the Loading
            // room is empty by construction (`goto_state` teardown + GML
            // `room_restart`), so they self-suppress there exactly like GML.
            // GML `TopCont/Draw_0:43` device gate lives inside
            // `crosshair_sprites` (skip only for a keyboard-driven local:
            // keyboard mode on, gamepad mode off). No live-pad half here:
            // GML's `is_gamepad(index)` is the sticky `opt_gamepad` setting
            // (`scrHandleInputsGeneral`), not per-frame pad activity - an
            // idle-but-enabled pad still draws the lerped crosshair.
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
            // Spiral CPU layer (GML `scrDrawSpiral` center figures): crown orbit +
            // player hurt figures ride every spiral caller - `GenCont`,
            // `LevCont`, `NothingSpiral`, the logo spirals alike. NOT the
            // campfire `Menu` (`PlayButton` destroys the `SpiralCont` on
            // entry, so the `with SpiralCont` figure block has no instance)
            // and NOT GameOver (the dead run's cont died at generation end),
            // so the figures never float over the flat camp or the dim. They
            // follow the live SpiralCont emitter position, which is view-local
            // and independent of the room camera.
            let vortex_mounted_later = self.assets.is_some()
                && !self.vortex_tex.is_empty()
                && !matches!(menu_kind, Some(MenuOverlay::Title))
                && (!matches!(menu_kind, Some(MenuOverlay::Splash)) || splash_logo)
                && (!paused || spiral_cover)
                && !game_over
                && (self.spiral.alive || !self.spiral.is_done());
            if vortex_mounted_later && (self.spiral.alive || spiral_cover) {
                let (emitter_x, emitter_y) = self.spiral.emitter_pos();
                let figs_center = Vec2::new(view[0] + emitter_x, view[1] + emitter_y);
                let mut figs =
                    spiral_figures(&mut self.sim.world, assets, figs_center, self.spiral.angle);
                stamp_z(&mut figs, Z_SPIRAL_FIGURES);
                spiral_figure_sprites = figs;
            }
            // View-anchored HUD bars (Draw-GUI-64: above all world-space layers).
            // GML `GenCont/Draw_0` draws ONLY spiral + GENERATING + roadmap -
            // no PlayerHUD/MiscHUD - and the Loading room's run is
            // pre-`setup_run` (no live run actors), so the HUD stays off while
            // Loading like GML. LevCont's mutation offer keeps the live
            // PlayerHUD enabled.
            let hud_view = view_rect_world(viewport_dp, world_size, &self.cam);
            let loading_cover = state == AppState::Loading
                || self
                    .sim
                    .world
                    .get_resource::<crate::comps_b::FloorTransition>()
                    .is_some_and(|f| f.active);
            let mut h = if loading_cover {
                Vec::new()
            } else {
                hud_sprites(&mut self.sim.world, assets, hud_view, hud_dt)
            };
            stamp_z(&mut h, Z_HUD);
            if pending_pick {
                cover_hud_sprites = h;
            } else {
                s.extend(h);
            }
            // Touch controls (`scrDrawMobileControls`, `TopCont/Draw_64` tail):
            // sticks + buttons over the HUD, under splash/menus. GML gate
            // verbatim (`TopCont/Draw_64:23`): `drawcontrols` (off during
            // cinematic/throne-sit/unlock) + live player + touch device
            // (`!opt_keyboard && !opt_gamepad`) + not the layout editor. The
            // port also requires live play (menus/game-over have no sticks);
            // mutation offers stay touch-interactive while paused.
            let mutation_screen = menu_kind == Some(MenuOverlay::Mutation);
            let touch_paused = self
                .sim
                .world
                .get_resource::<crate::state::Paused>()
                .is_some_and(|p| p.0);
            let touch_overlay = self
                .sim
                .world
                .get_resource::<OverlayMenu>()
                .copied()
                .unwrap_or_default();
            let touch_live = (playing || mutation_screen)
                && state == AppState::InGame
                && !game_over
                && self
                    .sim
                    .world
                    .query::<&crate::comps_a::Player>()
                    .iter(&self.sim.world)
                    .next()
                    .is_some()
                && (!touch_paused || mutation_screen)
                && touch_overlay == OverlayMenu::None
                && self
                    .sim
                    .world
                    .get_resource::<crate::state::menus::MenuState>()
                    .is_none_or(|m| !m.unlock.visible)
                && self
                    .sim
                    .world
                    .query::<(&Pos, &crate::comps_b::ThroneSit)>()
                    .iter(&self.sim.world)
                    .next()
                    .is_none()
                && {
                    let (keyboard, gamepad) = crate::input::gml_input_device(
                        self.sim
                            .world
                            .get_resource::<crate::savedata_part::SaveData>(),
                    );
                    !keyboard && !gamepad
                };
            let gamepad_live = playing
                && self
                    .sim
                    .world
                    .get_resource::<crate::savedata_part::SaveData>()
                    .is_some_and(|s| s.settings.gamepad_enabled)
                && self.pad_live;
            if touch_live || gamepad_live {
                let mut t = touch_sprites(
                    &mut self.sim.world,
                    assets,
                    viewport_dp,
                    world_size,
                    &self.cam,
                );
                stamp_z(&mut t, Z_TOUCH);
                self.last_touch_count = t.len();
                s.extend(t);
            } else {
                self.last_touch_count = 0;
            }
            // In-run pause button (GML `UberCont/Draw_64:46-61`). Live
            // play only, and above every other chrome rung: `UberCont`
            // (-1000) is lower than `TopCont` (-15) and `Menu` (-1001).
            // GML's `!want_pause` is the port's "not paused, no overlay".
            if state == AppState::InGame && !paused && !game_over && menu_kind.is_none() {
                s.extend(crate::render::pause_button_sprite(
                    &mut self.sim.world,
                    assets,
                    viewport_dp,
                    world_size,
                    &self.cam,
                ));
            }
            // Boot reel (`Vlambeer/Draw_0` + `Logo/Draw_0`).
            if menu_kind == Some(MenuOverlay::Splash) {
                let mut splash = splash_sprites(&mut self.sim.world, assets, view);
                stamp_z(&mut splash, Z_SPLASH);
                s.extend(splash);
            }
            // Menu art sprites (char pods, portrait, loadout, splats), drawn by `Menu`
            // in `Draw_0`/`Draw_74` on every screen it owns. Title, Loading,
            // Mutation, Pause, Settings and GameOver chrome use a separate
            // viewport so the letterbox and vortex passes sit between the
            // room and the menu layer at their GML depth boundaries.
            if let Some(kind) = if ft_active {
                Some(MenuOverlay::Loading)
            } else {
                menu_kind
            } {
                let mut menu = menu_sprites(
                    kind,
                    &mut self.sim.world,
                    assets,
                    viewport_dp,
                    world_size,
                    &self.cam,
                );
                if kind == MenuOverlay::Pause {
                    front_chrome_sprites = pause_button_sprites(
                        &mut self.sim.world,
                        assets,
                        viewport_dp,
                        world_size,
                        &self.cam,
                    );
                }
                let menu_z = if menu_kind == Some(MenuOverlay::Mutation) {
                    Z_HUD - 1.0
                } else {
                    Z_MENU
                };
                stamp_z(&mut menu, menu_z);
                if matches!(
                    kind,
                    MenuOverlay::Title
                        | MenuOverlay::Loading
                        | MenuOverlay::Mutation
                        | MenuOverlay::Pause
                        | MenuOverlay::Settings
                        | MenuOverlay::GameOver
                ) {
                    chrome_sprites = menu;
                } else {
                    s.extend(menu);
                }
            }
            // Hardware cursor (GML `UberCont/Draw_75` verbatim): while the OS pointer
            // is hidden the game hands it `sprCrosshair[opt_crosshair]`
            // (`opt_cursorcol`, alpha 1) - composited by the OS, so it sits above
            // every sprite and UI layer with zero frame lag, on EVERY screen
            // (gameplay included). GML gates only on
            // `window_get_cursor() == cr_none` + `scrCanDrawCursor()`
            // (`show_crosshair`, desktop/keyboard, no spawner): the
            // `opt_gamepad` switch never withdraws it, so with both switches on
            // the raw crosshair draws alongside the lerped `TopCont` one. The
            // port's `keyboard_mode` (raw `keyboard_enabled`, no touch) carries
            // the cursor-hidden half; deliberately no per-screen kind list (the
            // old 5-kind gate left keyboard gameplay cursorless). Skipped while
            // paused/an overlay owns the pointer and on touch (no cursor at
            // all). Pixels decode once per (frame, tint) into `cursor_img`; the
            // runner caches the OS handle by content hash.
            let menu_crosshair = keyboard_mode && !paused && overlay == OverlayMenu::None;
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
            // GUI Begin - before the GUI-64 HUD/menus and the Draw_75
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
        let cover_chrome_off = state == AppState::Loading
            || self
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

        // Vortex snapshot -> mounted background pass. `bg_alpha` is GML
        // `scrDrawSpiral` verbatim: `draw_clear(c_black)` runs in every caller
        // EXCEPT `Menu` - opaque black behind the logo spiral, MainMenu,
        // Loading and run covers; transparent over the campfire camp on
        // Title. InGame mounts the pass only while a floor transition or
        // mutation/ultra cover runs (the only spiral callers in a run).
        let full_spiral = match state {
            AppState::Title | AppState::Splash | AppState::MainMenu | AppState::Loading => true,
            AppState::InGame => ft_active || pending_pick || self.spiral.alive,
        };
        let (bg_alpha, draw_bolts, draw_details) = match state {
            AppState::Title => (0.0, false, true),
            AppState::Splash | AppState::MainMenu | AppState::Loading => (1.0, true, true),
            AppState::InGame if full_spiral => (1.0, true, true),
            AppState::InGame => (0.0, false, false),
        };
        self.last_bg_alpha = bg_alpha;
        // Vortex art follows the GML area (debris strip is per-area);
        // decode once per area, not per frame.
        let gml_area = gml_area_for_area(area);
        if self.assets_dir.is_some() && self.vortex_tex_area != Some(gml_area) {
            if let Some(dir) = self.assets_dir.clone() {
                self.vortex_tex_gen = self.vortex_tex_gen.wrapping_add(1).max(1);
                let generation = self.vortex_tex_gen;
                self.vortex_tex = Self::load_vortex_textures(&dir, gml_area, generation);
                self.vortex_tex_area = Some(gml_area);
            }
        }
        // The vortex layer mounts only where a GML spiral caller exists:
        // `Vlambeer/Alarm_0` creates `SpiralCont` at the mode-3 -> 4
        // transition, so the pass stays absent for modes 0-3, then remains
        // through the main-menu handoff. Paused/GameOver draw their captured
        // room or dead-run panel instead of a live spiral.
        let splash = menu_kind == Some(MenuOverlay::Splash);
        let vortex_requested = self.assets.is_some()
            && !self.vortex_tex.is_empty()
            && (!splash || splash_logo)
            && (!paused || spiral_cover)
            && !game_over
            && (self.spiral.alive || !self.spiral.is_done() || spiral_cover);
        let snap = vortex_requested.then(|| {
            self.spiral
                .snapshot_with_render_mode(bg_alpha, draw_bolts, draw_details)
        });
        let mut vortex_layer = snap.map(|snap| {
            let mut pass = VortexPass::new(snap);
            pass.extend_textures(self.vortex_tex.clone());
            // The engine hands this node's rect to the pass as the viewport, so
            // anchoring to the GML box both clips the quad to the pillarbox
            // and keeps its `view` uv->GUI mapping 1:1 with the batch.
            // `fill_max_size` spread the spiral over the bars: the batch
            // frames in-shader, a fullscreen pass only sees the raw canvas.
            let box_dp = self.gml_frame().box_dp;
            Embedded(
                Modifier::new()
                    .absolute()
                    .size(Dp(box_dp[2]), Dp(box_dp[3]))
                    .offset(Some(Dp(box_dp[0])), Some(Dp(box_dp[1])), None, None)
                    .hit_passthrough(),
                Callback::new(pass),
            )
        });
        // GML depth law: LOWER depth draws IN FRONT (manual: "-1000 is
        // drawn on top of -100, which is drawn on top of 0"), and
        // `scrDrawSpiral` opens with `draw_clear(c_black)` at `SpiralCont`
        // depth -101. `Logo` sits at -10000 - frontmost rung, shared with
        // `MainMenuButton` - so the boot-reel logo paints OVER the cleared
        // spiral. Only the logo needs the pass underneath (it is the sole
        // sprite in the viewport then); `Z_SPIRAL_FIGURES` < `Z_SPLASH`
        // keeps the spiral center figures behind the logo as in GML.
        let vortex_above = vortex_layer.is_some() && !splash_logo;
        if !vortex_above {
            sprites.extend(std::mem::take(&mut spiral_figure_sprites));
        }
        // The area fill only shows where GML paints it: `GenCont/Create_0`'s
        // `background_set_colour(scrAreaGetBackroundColor(GameCont.area))`
        // runs once per generated floor, and the campfire title inherits it via
        // `MenuGen`. The fill must NEVER sit under a mounted vortex pass as a
        // fullscreen quad: `Menu/Draw_0` calls `scrDrawSpiral()` FIRST
        // (transparent, no `draw_clear`) and draws the camp floors OVER it, so
        // the spiral shows through the gaps between tiles - a fill quad would
        // bury the vortex layer (the "no vortex on the title screen" bug). The
        // pass's own bg_alpha owns the backdrop (opaque black on Loading/covers,
        // transparent on Title). Splash/MainMenu have no area yet
        // (`Vlambeer/Create_0` never sets a colour; `Vlambeer/Draw_0` clears
        // black); live gameplay past the spiral drain and GameOver (spiral died
        // at generation end) also fall back to the flat room colour.
        //
        // That fill is a GPU write into the sRGB target, which re-encodes on
        // write, so it must be linear like every sprite tint (`tint_to_linear`)
        // - `background_color` is authored in sRGB and raw double-encoding sent
        // Desert's #af8f6a to screen as #d8c5ad, a pale void stopping at the
        // wall art.
        let background = if vortex_layer.is_some() && bg_alpha > 0.0 {
            None
        } else if matches!(
            menu_kind,
            Some(MenuOverlay::Splash) | Some(MenuOverlay::MainMenu)
        ) {
            Some([0.0, 0.0, 0.0, 1.0])
        } else {
            Some(background_color(area).map(srgb_to_linear))
        };
        // Fullscreen overlays: hit flashes (white). Menu dimming is
        // the scrim `UiBox` above (bevy parity: one 230-black layer
        // over everything, background included), never the viewport
        // tint (that would double-dim the sprites).
        let dim_menu = state == AppState::InGame
            && matches!(
                menu_kind,
                Some(MenuOverlay::Pause)
                    | Some(MenuOverlay::Settings)
                    | Some(MenuOverlay::Credits)
                    | Some(MenuOverlay::GameOver)
                    | Some(MenuOverlay::Stats)
                    | Some(MenuOverlay::Unlock)
            );
        let overlay_color = crate::effects::flash_rgba(&self.sim.world);

        // GML `scrGameIsGenerationScreen` keeps PlayerHUD text off for
        // Loading and floor-transition covers; `LevCont/Draw_64` explicitly
        // keeps it on for mutation offers. The canvas text layer rides ABOVE
        // the opaque vortex pass, so the remaining cover gates prevent
        // HP/level/ammo/FLOOR from painting over the spiral.
        let hud_rows = if state == AppState::InGame
            && !cover_chrome_off
            && matches!(
                menu_kind,
                None | Some(MenuOverlay::Mutation) | Some(MenuOverlay::Pause)
            ) {
            hud_overlay_lines(&mut self.sim.world, viewport_dp)
        } else {
            Vec::new()
        };
        let gen_cover = state == AppState::Loading || ft_active;
        let mut menu_rows = if gen_cover {
            Some(menu_gui_texts_dp(
                MenuOverlay::Loading,
                &mut self.sim.world,
                viewport_dp,
            ))
        } else {
            menu_kind.map(|k| menu_gui_texts_dp(k, &mut self.sim.world, viewport_dp))
        };
        // GML `TutCont/Draw_64` verbatim: the step instruction bar draws
        // at the letterbox bottom until the exit portal exists - over
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

        if state != AppState::InGame {
            self.last_live_frame = None;
        }

        let frame = Arc::new(FrameInput {
            cam: self.cam,
            world_size,
            viewport_dp,
            sprites,
            texts,
            background,
            overlay_color,
            chroma: 0.0,
        });
        let live_offer = menu_kind == Some(MenuOverlay::Mutation)
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
        let freeze_surface = paused && !live_offer;
        let display_frame = if freeze_surface {
            self.last_live_frame
                .clone()
                .unwrap_or_else(|| frame.clone())
        } else {
            if !paused {
                self.last_live_frame = Some(frame.clone());
            }
            frame.clone()
        };

        let staging = self.staging.clone();
        let viewport = if self.assets.is_some() {
            let geom = GeomHandle::new();
            let uploads = self
                .assets
                .as_ref()
                .map(|a| a.take_uploads())
                .unwrap_or_default();
            let desc: BatchDesc = self
                .assets
                .as_ref()
                .map(|a| a.batch_desc())
                .unwrap_or_default();
            Viewport2dGpuWithHudShared(display_frame, geom, uploads, desc, move |ev| {
                let mut staging = staging.borrow_mut();
                match ev {
                    PickEvent::Press {
                        world,
                        screen,
                        button,
                    } => {
                        staging.pick_down(button);
                        if button == PointerButton::Primary {
                            let d = repose_core::locals::effective_density_scale().max(1e-6);
                            staging.stage_click(world, screen, d);
                        }
                    }
                    PickEvent::Click { button, .. } => {
                        staging.pick_up(button);
                    }
                    PickEvent::Hover { .. } => {}
                    PickEvent::TouchDown { id, screen } => {
                        let d = repose_core::locals::effective_density_scale().max(1e-6);
                        staging.touch_down(id, Vec2::new(screen[0] / d, screen[1] / d))
                    }
                    PickEvent::TouchMove { id, screen } => {
                        let d = repose_core::locals::effective_density_scale().max(1e-6);
                        staging.touch_move(id, Vec2::new(screen[0] / d, screen[1] / d))
                    }
                    PickEvent::TouchUp { id } => {
                        staging.touch_up(id);
                    }
                }
            })
        } else {
            let geom = GeomHandle::new();
            let staging = self.staging.clone();
            Viewport2dShared(display_frame, geom, move |ev| {
                let mut staging = staging.borrow_mut();
                match ev {
                    PickEvent::Press {
                        world,
                        screen,
                        button,
                    } => {
                        staging.pick_down(button);
                        if button == PointerButton::Primary {
                            let d = repose_core::locals::effective_density_scale().max(1e-6);
                            staging.stage_click(world, screen, d);
                        }
                    }
                    PickEvent::Click { button, .. } => {
                        staging.pick_up(button);
                    }
                    PickEvent::Hover { .. } => {}
                    PickEvent::TouchDown { id, screen } => {
                        let d = repose_core::locals::effective_density_scale().max(1e-6);
                        staging.touch_down(id, Vec2::new(screen[0] / d, screen[1] / d))
                    }
                    PickEvent::TouchMove { id, screen } => {
                        let d = repose_core::locals::effective_density_scale().max(1e-6);
                        staging.touch_move(id, Vec2::new(screen[0] / d, screen[1] / d))
                    }
                    PickEvent::TouchUp { id } => {
                        staging.touch_up(id);
                    }
                }
            })
        };
        let make_chrome_view = |sprites: Vec<SpriteInstance>, id: &'static str| {
            if sprites.is_empty() {
                return None;
            }
            let chrome_frame = FrameInput {
                cam: self.cam,
                world_size,
                viewport_dp,
                sprites,
                texts: Vec::new(),
                background: None,
                overlay_color: None,
                chroma: 0.0,
            };
            if self.assets.is_some() {
                let desc = self
                    .assets
                    .as_ref()
                    .map(|a| a.batch_desc())
                    .unwrap_or_default();
                let mut view = Viewport2dGpuWithIdShared(
                    Arc::new(chrome_frame),
                    GeomHandle::new(),
                    self.assets
                        .as_ref()
                        .map(|a| a.take_uploads())
                        .unwrap_or_default(),
                    desc,
                    id,
                    |_| {},
                );
                view.modifier = view.modifier.hit_passthrough();
                Some(view)
            } else {
                let mut view = Viewport2dShared(Arc::new(chrome_frame), GeomHandle::new(), |_| {});
                view.modifier = view.modifier.hit_passthrough();
                Some(view)
            }
        };
        let mut chrome_view = make_chrome_view(chrome_sprites, "viewport2d.menu");
        let cover_hud_view = if pending_pick {
            make_chrome_view(cover_hud_sprites, "viewport2d.cover.hud")
        } else {
            None
        };
        let front_chrome_view = make_chrome_view(front_chrome_sprites, "viewport2d.menu.front");
        let spiral_figure_view = if vortex_above {
            make_chrome_view(spiral_figure_sprites, "viewport2d.spiral.figures")
        } else {
            None
        };

        // Focusable root so hardware keys reach the staging feed. Focus loss
        // drops held keys + edges here: winit delivers no key-ups across an
        // alt-tab (GML's `keyboard_check` reads all-up while unfocused).
        // Cursor: the scheduler override wins over hover (hover would always
        // report Default over the viewport and un-hide the pointer). GML
        // `UberCont/Step_0:175-183`: keyboard mode hides the OS cursor,
        // menus/mouse mode shows it. With live cursor art (`cursor_img`,
        // same gate as the decode above) the OS composites the crosshair
        // itself - topmost, zero lag, no sprite needed.
        sched.cursor_override = Some(if hide_os_cursor {
            match &self.cursor_img {
                Some(img) => repose_core::CursorIcon::Custom(img.clone()),
                None => repose_core::CursorIcon::Hidden,
            }
        } else {
            repose_core::CursorIcon::Default
        });
        // Game shortcuts (Esc/R/Enter -> pause/restart/confirm), composed
        // once per frame (mount-once under the hood).
        let shortcut_edges = self.shortcut_edges.clone();
        repame_shell::install_game_shortcuts(
            &shortcut_edges,
            nt_shortcuts::PAUSE,
            nt_shortcuts::RESTART,
            nt_shortcuts::CONFIRM,
        );
        // REMAP capture preview: while a rebind is armed, the next key press
        // resolves the capture here (tunnel phase) before the runtime shortcut
        // dispatch can consume Esc/Enter/R as pause/restart/confirm. `App` is
        // borrowed through a raw pointer: the closure only touches staged
        // input through it, like the viewport `PickEvent` closures below.
        let capture_app: *mut App = self as *mut App;
        let capture_preview =
            move |ke: KeyEvent| unsafe { (*capture_app).preview_capture_key(&ke) };
        let focus = remember(FocusRequester::new);
        let fr_positioned = (*focus).clone();
        let focus_staging = self.staging.clone();
        let key_staging = self.staging.clone();
        let root_mod = Modifier::new()
            .fill_max_size()
            .focusable(true)
            .focus_requester((*focus).clone())
            .on_preview_key_event(capture_preview)
            .on_globally_positioned(move |_| {
                fr_positioned.request_focus();
            })
            .on_focus_changed(move |focused| {
                focus_staging.borrow_mut().set_window_focused(focused);
            })
            .on_key_event(move |ke: KeyEvent| {
                key_staging.borrow_mut().handle_key(&ke);
                false
            });
        // Right mouse button (GML `mb_right` ability / menu Back): the
        // viewport `PickEvent` button stages `rmb_down`/`lmb_down`
        // directly, so nothing else is needed here - the runtime-owned
        // `Scheduler::pointer_pos_px` already tracks the cursor every
        // frame outside focus dispatch.

        // HUD overlay (GML `scrDrawPlayerHUD` + `scrDrawMiscHUD` verbatim):
        // Silkscreen rows at 320x240 GUI positions over the viewport.
        // Bottom-up: live-game and Title remnant drains sit above the sprite
        // viewport; opaque covers sit below their chrome. The GML depth table
        // puts `Menu`/`TopCont` in front of the floor, so the Title
        // transition intentionally draws its remaining spiral over the
        // campfire map while the room chrome stays above it.
        let mut layers = Vec::new();
        // Pillarbox: black outside the GML box, so a non-16:9 window
        // shows bars instead of stretching the view (see `gml_frame`).
        let gml = self.gml_frame();
        let bars = gml.box_dp;
        let canvas_dp = viewport_dp;
        let inset = bars[0] > 0.5
            || bars[1] > 0.5
            || bars[0] + bars[2] < canvas_dp[0] - 0.5
            || bars[1] + bars[3] < canvas_dp[1] - 0.5;
        if inset {
            let black = UiBox(
                Modifier::new()
                    .fill_max_size()
                    .background(Color::from_rgba(0, 0, 0, 255))
                    .hit_passthrough(),
            );
            layers.push(black);
        }
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
        if let Some(figures) = spiral_figure_view {
            layers.push(figures);
        }
        if !pending_pick && !hud_rows.is_empty() {
            layers.push(
                ZStack(Modifier::new().fill_max_size().hit_passthrough())
                    .child(hud_rows.iter().map(gui_text_layer).collect::<Vec<_>>()),
            );
        }
        // GML `scrLetterbox` (36 px bars): cinematic chrome over menus,
        // transitions, game over, and boss intros - but NOT the
        // campfire title (`Menu/Create_0`: `scrLetterbox(false, 0)`).
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
        // with a fresh live spiral - the previous run's look point and
        // drain never carry over.
        if matches!(state, AppState::MainMenu) && !matches!(self.was_state, AppState::MainMenu) {
            self.gml_cam.snap = true;
        }
        let entered_title = state == AppState::Title && self.was_state != AppState::Title;
        self.was_state = state;
        if entered_title {
            self.letterbox_frame = 0.0;
        }

        // GML keeps an explicit `UberCont.letterbox` flag rather than deriving
        // it from the current room: enabled in `Vlambeer/Create_0` for
        // logo/main-menu, disabled by `Menu/Create_0` for the campfire title
        // (whose Menu draw calls `scrDrawLetterbox` directly), re-enabled by
        // `GenCont`/`LevCont`, pause and GameOver; Credits disables it.
        let letterboxed = match state {
            AppState::Splash | AppState::Loading => true,
            AppState::MainMenu => menu_kind != Some(MenuOverlay::Credits),
            AppState::Title => false,
            AppState::InGame => {
                matches!(
                    menu_kind,
                    Some(
                        MenuOverlay::Pause
                            | MenuOverlay::Settings
                            | MenuOverlay::Mutation
                            | MenuOverlay::GameOver
                    )
                ) || boss_intro
                    || transitioning
                    || spiral_cover
            }
        };
        let title_menu_letterbox = state == AppState::Title;
        let letterbox_target = if letterboxed { 3.0 } else { 0.0 };
        let letterbox_step = (dt.as_secs_f32() * 30.0).clamp(0.0, 1.0);
        if self.letterbox_frame < letterbox_target {
            self.letterbox_frame = (self.letterbox_frame + letterbox_step).min(letterbox_target);
        } else if self.letterbox_frame > letterbox_target {
            self.letterbox_frame = (self.letterbox_frame - letterbox_step).max(letterbox_target);
        }
        let letterbox_visible = title_menu_letterbox || self.letterbox_frame > 0.0;
        let letterbox_frame = if title_menu_letterbox {
            3
        } else {
            self.letterbox_frame.floor() as i32
        };
        let mut letterbox_view: Option<View> = None;
        if letterbox_visible {
            // GML `LETTERBOX_SIZE 36` view px tall (`scrLetterbox`): 36 GUI px
            // -> dp through the frame's dp-per-world scale, anchored to the
            // pillarbox box (not the canvas edges) so the bars hug the view.
            let gml = self.gml_frame();
            let bar_dp = (36.0 * gml.dp_per_world).ceil();
            let mut has_art = false;
            if let Some(assets) = self.assets.as_ref() {
                let art =
                    letterbox_sprites(assets, viewport_dp, world_size, &self.cam, letterbox_frame);
                if !art.is_empty() {
                    let frame = FrameInput {
                        cam: self.cam,
                        world_size,
                        viewport_dp,
                        sprites: art,
                        texts: Vec::new(),
                        background: None,
                        overlay_color: None,
                        chroma: 0.0,
                    };
                    let mut view = Viewport2dGpuWithIdShared(
                        Arc::new(frame),
                        GeomHandle::new(),
                        assets.take_uploads(),
                        assets.batch_desc(),
                        "viewport2d.letterbox",
                        |_| {},
                    );
                    view.modifier = view.modifier.hit_passthrough();
                    let mut children = Vec::new();
                    let bar = Dp(bar_dp);
                    let width = Dp(gml.box_dp[2]);
                    let left = Dp(gml.box_dp[0]);
                    let top = Dp(gml.box_dp[1]);
                    children.push(UiBox(
                        Modifier::new()
                            .absolute()
                            .size(width, bar)
                            .offset(Some(left), Some(top), None, None)
                            .background(Color::from_rgba(0, 0, 0, 255))
                            .hit_passthrough(),
                    ));
                    children.push(UiBox(
                        Modifier::new()
                            .absolute()
                            .size(width, bar)
                            .offset(
                                Some(left),
                                Some(Dp(gml.box_dp[1] + gml.box_dp[3] - bar_dp)),
                                None,
                                None,
                            )
                            .background(Color::from_rgba(0, 0, 0, 255))
                            .hit_passthrough(),
                    ));
                    children.push(view);
                    letterbox_view = Some(
                        ZStack(Modifier::new().fill_max_size().hit_passthrough()).child(children),
                    );
                    has_art = true;
                }
            }
            if !has_art {
                // No art: the same bars, but inset by the pillarbox
                // offsets so they land on the view edges.
                let pad_top = Dp(gml.box_dp[1]);
                let pad_bottom = Dp(viewport_dp[1] - gml.box_dp[1] - gml.box_dp[3]);
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
                letterbox_view = Some(
                    Column(Modifier::new().fill_max_size().hit_passthrough()).child(vec![
                        UiBox(
                            Modifier::new()
                                .fill_max_width()
                                .height(pad_top)
                                .background(Color::from_rgba(0, 0, 0, 255))
                                .hit_passthrough(),
                        ),
                        bar(),
                        spacer,
                        bar(),
                        UiBox(
                            Modifier::new()
                                .fill_max_width()
                                .height(pad_bottom)
                                .background(Color::from_rgba(0, 0, 0, 255))
                                .hit_passthrough(),
                        ),
                    ]),
                );
            }
        }
        let letterbox_before_content = menu_kind
            .is_some_and(|kind| !matches!(kind, MenuOverlay::Credits | MenuOverlay::Unlock));
        if let Some(rows) = menu_rows {
            // GML `GameOver/Draw_0:7-10` dims with `draw_set_alpha(0.7)`
            // (178/255); pause/settings/credits/stats sit on the
            // near-opaque bevy `scrim` (230/255). The Draw_75 cursor
            // draws after, so it stays full-bright over the dim.
            let scrim_alpha = match menu_kind {
                Some(MenuOverlay::Pause | MenuOverlay::GameOver) => 178,
                Some(MenuOverlay::Unlock) => 153,
                _ => 230,
            };
            if dim_menu {
                layers.push(UiBox(
                    Modifier::new()
                        .fill_max_size()
                        .background(Color::from_rgba(0, 0, 0, scrim_alpha))
                        .hit_passthrough(),
                ));
            }
            if letterbox_before_content {
                if let Some(letterbox) = letterbox_view.take() {
                    layers.push(letterbox);
                }
            }
            if !gen_cover && let Some(chrome) = chrome_view.take() {
                layers.push(chrome);
            }
            if !rows.is_empty() {
                // Plain text rows (GML draws the buttons as text;
                layers.push(
                    ZStack(Modifier::new().fill_max_size().hit_passthrough())
                        .child(rows.iter().map(gui_text_layer).collect::<Vec<_>>()),
                );
            }
            if gen_cover && let Some(chrome) = chrome_view.take() {
                layers.push(chrome);
            }
            if !letterbox_before_content {
                if let Some(letterbox) = letterbox_view.take() {
                    layers.push(letterbox);
                }
            }
        }
        if pending_pick {
            if !hud_rows.is_empty() {
                layers.push(
                    ZStack(Modifier::new().fill_max_size().hit_passthrough())
                        .child(hud_rows.iter().map(gui_text_layer).collect::<Vec<_>>()),
                );
            }
            if let Some(hud) = cover_hud_view {
                layers.push(hud);
            }
        }
        if let Some(letterbox) = letterbox_view.take() {
            layers.push(letterbox);
        }
        if let Some(front_chrome) = front_chrome_view {
            layers.push(front_chrome);
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

/// Resources the sim schedule reads that `setup_run` does not own (mirrors
/// the `insert_schedule_resources` list in `schedule.rs` tests), plus
/// startup defaults for run-scoped resources (Run, Score, FloorMask,
/// SaveData, Toast, menus...): bevy inserts these at app build and
/// `setup_run` resets them in place, so booting at Splash (pre-run) never
/// hits a missing-resource validation error.
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
    world.init_resource::<crate::pickups::GunDecideCache>();
    world.insert_resource(FloorTransition::default());
    world.init_resource::<NtInput>();
    world.init_resource::<crate::state::Paused>();
    world.init_resource::<crate::state::ActButton>();
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
    world.insert_resource(crate::comps_a::TopSmalls::default());
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
    world.init_resource::<CribTrip>();
    world.init_resource::<crate::state::OverlayMenu>();
    world.init_resource::<crate::state::PendingUnpause>();
    world.init_resource::<crate::state::QuitRequested>();
    world.init_resource::<crate::state::menus::MenuState>();
    world.init_resource::<crate::state::menus::MenuEdge>();
    world.init_resource::<crate::audio::AudioChannels>();
    world.init_resource::<Queue<crate::audio::UiBridgeAction>>();
    world.init_resource::<repame_anim::AnimCatalog>();
    // Editable controls: default map, then overlay the save file's
    // rows when present (GML `scrOptionsLoadKeymaps` on boot). The
    // disk shell calls `App::load_save` after boot; headless/tests
    // keep GML defaults.
    world.init_resource::<crate::keymap::InputMapState>();
}

/// Resolve the art dir: installed assets zip -> `$NT_ASSETS` -> exe-dir
/// `assets` -> cwd `assets`. Returns `None` when no source holds an
/// animation catalog (placeholder path stays active).
pub fn resolve_assets_dir() -> Option<PathBuf> {
    if crate::assetfs::installed() {
        return Some(PathBuf::from("/nt-assets"));
    }
    let has_catalog = |p: &Path| {
        p.join("images").join("anims.ron").is_file()
            || p.join("images").join("anims.json").is_file()
    };
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
        let cand = cwd.join("assets");
        if has_catalog(&cand) {
            return Some(cand);
        }
    }
    None
}

/// Fit extent helper: viewport size in dp, used only where a dp-space size
/// is needed before the GML frame resolves (the live frame passes
/// [`render::gml_view_size`] as `world_size` so the engine contain-fit owns
/// the scale and pillarbox). Takes PHYSICAL px (`Scheduler.size`).
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
/// Portal first, then the victory/sit markers - nearest instance of the
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
/// `scrDrawPlayerHUD` texts (HP `hp/max` at GUI (67,7), level at (11,16),
/// per-slot ammo at (42+slot*44,21), red `LOW HP` at (110,7)) plus the
/// `scrDrawMiscHUD` right-aligned clock/map rows at `view_width - 2`,
/// mapped through the live GML GUI law ([`gui_texts_dp`]:
/// GUI height 240, full live width). `LOW HP` blinks on the shell beat (GML
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
            // `disappear % 2` - same shell beat. (Toast text arrives
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
/// fixed-width box the Silkscreen line sits in (right-aligned misc-HUD
/// clock/area rows right-align inside their box). GML `draw_text_nt`
/// `@`-tags arrive pre-split as color runs ([`GuiRow`]); multi-run rows
/// render as `AnnotatedText` so `@w/@s/@r/@g/@y/@b/@p` colors show
/// verbatim (single-run rows keep the plain `Text` path).
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
/// `scrMakePauseButtons` layout parity; positions in `render.rs`'s
/// `menu_gui_texts`, actions in `apply_menu_action`): pause MENU/RETRY arm
/// the quit/restart confirm, the confirm swaps in BACK + QUIT/RETRY,
/// CONTINUE resumes, SETTINGS opens settings. Settings/credits BACK rows
/// unwind one level; every other row/text is inert (`None`).
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
/// and confirm only - `feed_input` strips them before the gameplay
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

/// Menu click router: canvas-dp click position -> [`UiAction`]. Hit-tests
/// the screen-anchored overlay buttons in live GML GUI space (GUI height
/// 240, width [`gml_view_size`]; dp per GUI px `k = h/240`) independently of
/// repose hit-testing: the dp layout boxes are full-width alignment boxes
/// that overlap (e.g. SETTINGS spans the MENU column), so only tight
/// per-glyph boxes route correctly. Positions from
/// [`menu_gui_texts_vw`](crate::render::menu_gui_texts_vw) (GML layout
/// parity); actions from [`menu_button_action`].
fn route_menu_click(
    world: &mut World,
    kind: MenuOverlay,
    gui: [f32; 2],
    vw: f32,
) -> Option<UiAction> {
    let (gx, gy) = (gui[0], gui[1]);
    // Settings rows route through the hot table (per-row toggle /
    // stepper semantics live there, next to the layout).
    if kind == MenuOverlay::Settings {
        let back_x = if cfg!(target_os = "android") {
            24.0
        } else {
            16.0
        };
        if gx >= back_x - 20.0 && gx <= back_x + 20.0 && gy >= 0.0 && gy <= 40.0 {
            return Some(UiAction::SettingsBack);
        }
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
            if kind == MenuOverlay::Pause {
                let index = match t.text.as_str() {
                    "MENU" | "BACK" => 0,
                    "RETRY" | "QUIT" => 1,
                    "SETTINGS" => 2,
                    "CONTINUE" => 3,
                    _ => usize::MAX,
                };
                if index != usize::MAX
                    && world
                        .get_resource::<MenuState>()
                        .and_then(|m| m.pause_appear.get(index).copied())
                        .is_some_and(|appear| appear >= 2.0)
                {
                    return None;
                }
            } else if kind == MenuOverlay::GameOver
                && world
                    .get_resource::<MenuState>()
                    .is_some_and(|menu| menu.go_appear > 0.0)
            {
                return None;
            }
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
            // GML `TopCont/Step_2.gml:18-26`: an `UnlockScreen` instance
            // short-circuits the block, so `GameOver` is never created
            // while an unlock is pending.
            if !menu.unlock_queue.is_empty() && menu.unlock.visible {
                return Some(MenuOverlay::Unlock);
            }
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

#[cfg(test)]
mod cursor_staging_tests {
    use super::*;

    #[test]
    fn polled_px_unprojects_through_live_camera() {
        let mut app = App::new_with_seed(4242);
        app.stage_pointer_px(Some([0.0, 0.0]));
        let live = app.live_cursor_world().expect("cursor staged");
        assert!(
            (live - Vec2::new(10.0, 10.0)).length() > 1.0,
            "polled px must unproject through the live camera, got {live:?}"
        );
    }

    /// Reported bug verbatim: the crosshair drifted across the screen as the
    /// camera moved even with the pointer still - the world point was baked
    /// through an old camera and the hover screen px was dropped
    /// (`_screen`). GML draws the raw cursor at GUI mouse coords
    /// (screen-anchored, never camera-following). Regression: same screen px
    /// under a moved camera must unproject to the new camera-relative point
    /// (track the camera 1:1), not stick to the old world point.
    #[test]
    fn hover_px_tracks_moved_camera() {
        let mut app = App::new_with_seed(4242);
        app.stage_pointer_px(Some([640.0, 360.0]));
        let before = app.live_cursor_world().expect("cursor staged");
        // Pan the camera 100 world units right; the pointer hasn't
        // moved (same screen px, still polled).
        app.cam.center += Vec2::new(100.0, 0.0);
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
        use repose_core::shortcuts::Action;
        let state = nt_shortcuts::state(&App::new_with_seed(4242));
        for (key, want) in [
            (Key::Escape, nt_shortcuts::PAUSE),
            (Key::Character('r'), nt_shortcuts::RESTART),
            (Key::Enter, nt_shortcuts::CONFIRM),
        ] {
            let chord = repose_core::shortcuts::KeyChord::new(key, Modifiers::default());
            assert_eq!(
                state.default_map.action_for(&chord),
                Some(Action::Custom(want.into())),
                "map must bind {want}"
            );
            assert_eq!(
                state.resolve_action(&chord),
                Some(Action::Custom(want.into())),
                "state must resolve {want} without a global install"
            );
        }
    }

    #[test]
    fn installed_shortcut_handler_stages_edges() {
        use repose_core::shortcuts::Action;
        let mut app = App::new_with_seed(4242);
        let state = nt_shortcuts::state(&app);
        assert!(state.handle(Action::Custom(nt_shortcuts::PAUSE.into())));
        nt_shortcuts::drain(&mut app);
        assert!(app.pause_edge);
        assert!(state.handle(Action::Custom(nt_shortcuts::RESTART.into())));
        nt_shortcuts::drain(&mut app);
        assert!(app.restart_edge);
        assert!(state.handle(Action::Custom(nt_shortcuts::CONFIRM.into())));
        nt_shortcuts::drain(&mut app);
        assert!(app.interact_edge);
        assert!(!state.handle(Action::Custom("nt.unknown".into())));
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
        let mut staging = app.staging.borrow_mut();
        let edges = staging.take_edges();
        let mut held = std::collections::HashSet::new();
        for key in staging.held.iter() {
            if let Some(code) = crate::input::keycode_for_physical(*key) {
                held.insert(code);
            }
        }
        drop(staging);
        let just: std::collections::HashSet<KeyCode> = edges
            .into_iter()
            .filter_map(crate::input::keycode_for_physical)
            .collect();
        assert!(just.contains(&KeyCode::KeyE), "KeyE must stage an edge");
        let keymap = app.sim.world.resource::<InputMapState>().clone();
        let mut out = NtInput::default();
        crate::input::sample_keyboard_mapped(
            &held,
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
    /// did nothing - the physical capture table only mapped ~20 names
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
            // Live shell path: shared staging (mirrors the armed flag in
            // `feed_input`) stashes the press; the next `feed_input`
            // resolves it.
            app.staging.borrow_mut().capture_armed = true;
            app.staging.borrow_mut().handle_key(&KeyEvent {
                key: Key::Character('x'),
                modifiers: Modifiers::default(),
                is_repeat: false,
                event_type: KeyEventType::Down,
                utf16_code_point: 0,
                physical: Some(key),
            });
            app.feed_input();
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
        let mut app = App::new_with_seed(4242);
        let state = nt_shortcuts::state(&app);
        app.staging.borrow_mut().handle_key(&KeyEvent {
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
        assert!(state.handle(repose_core::shortcuts::Action::Custom(
            nt_shortcuts::PAUSE.into()
        )));
        nt_shortcuts::drain(&mut app);
        assert!(app.pause_edge, "shortcut handler must stage pause_edge");
        crate::state::menus::apply_menu_action(
            &mut app.sim.world,
            crate::audio::UiAction::RemapControl("north".to_string()),
        );
        assert!(app.capture_armed());
        app.staging.borrow_mut().capture_armed = true;
        app.staging.borrow_mut().handle_key(&KeyEvent {
            key: Key::Character('x'),
            modifiers: Modifiers::default(),
            is_repeat: false,
            event_type: KeyEventType::Down,
            utf16_code_point: 0,
            physical: Some(PhysicalKey::KeyX),
        });
        app.feed_input();
        assert!(!app.capture_armed(), "KeyX must resolve the capture");
        let entry = app
            .sim
            .world
            .resource::<InputMapState>()
            .session
            .map
            .keyboard(&crate::keymap::NtAction::North);
        assert!(
            matches!(
                entry,
                repame_input::KeymapEntry::Key(_) | repame_input::KeymapEntry::Physical(_)
            ),
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
        app.staging.borrow_mut().capture_armed = true;
        app.staging
            .borrow_mut()
            .handle_key(&press(Key::Character('z'), PhysicalKey::KeyZ));
        app.feed_input();
        assert!(!app.capture_armed());
        // Persist + reload like a reboot.
        let rows = app
            .sim
            .world
            .resource::<InputMapState>()
            .session
            .map
            .clone();
        let saved = crate::keymap::KeyBindings::from_keymap(&rows);
        let back = saved.to_keymap();
        app.sim.world.resource_mut::<InputMapState>().session.map = back;
        // Release Z, then press it fresh and sample movement.
        app.stage_physical_key(PhysicalKey::KeyZ, false);
        app.stage_physical_key(PhysicalKey::KeyZ, true);
        let mut out = crate::input::NtInput::default();
        let mut staging = app.staging.borrow_mut();
        let edges = staging.take_edges();
        let mut held = std::collections::HashSet::new();
        for key in staging.held.iter() {
            if let Some(code) = crate::input::keycode_for_physical(*key) {
                held.insert(code);
            }
        }
        drop(staging);
        let just: std::collections::HashSet<crate::input::KeyCode> = edges
            .into_iter()
            .filter_map(crate::input::keycode_for_physical)
            .collect();
        let keymap = app.sim.world.resource::<InputMapState>().clone();
        let mouse = crate::input::MouseState::default();
        crate::input::sample_keyboard_mapped(&held, &just, &mouse, Some(&keymap), &mut out);
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
        let mut state = InputMapState::default();
        state.session.map = map;
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
        let mut state = InputMapState::default();
        state.session.map = map;
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
        assert!(!out.fire_held, "menu-owned Space must not hold fire");
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
            app.staging.borrow_mut().handle_key(&KeyEvent {
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
    /// Live click-to-arm chain: a click on the REMAP page's first row (through
    /// `route_menu_click` → `settings_click_action` → hot row →
    /// `RemapControl` → `begin_capture`) must arm the capture. The click dp
    /// comes from the row's own GUI box (as the viewport stages it), so this
    /// also proves the drawn rows and the hot rows agree.
    #[test]
    fn remap_row_click_arms_capture() {
        let mut app = App::new_with_seed(4242);
        app.sim.world.insert_resource(AppState::MainMenu);
        app.sim.world.insert_resource(OverlayMenu::Settings);
        app.sim.world.resource_mut::<MenuState>().settings_page = 13;
        let rows = crate::render::settings_hot_rows(13, 320.0);
        let row = rows[0];
        // `route_menu_click` takes GML view px + the GUI width, so the
        // row's own GUI position is the click position.
        let action = route_menu_click(
            &mut app.sim.world,
            MenuOverlay::Settings,
            [row.cx, row.gy],
            320.0,
        );
        assert!(
            matches!(action, Some(crate::audio::UiAction::RemapControl(_))),
            "click on row 0 must arm a remap, got {action:?}"
        );
        crate::state::menus::apply_menu_action(&mut app.sim.world, action.unwrap());
        assert!(app.capture_armed(), "capture must arm after the row click");
    }
}

/// Android entry (`cargo rapk`), repadio-shaped: the lib target owns
/// `android_main` (a bin target's symbol never lands in the `.so` the
/// NativeActivity loader opens). Boot threads the internal data path
/// through save + asset lookups since cwd is `/` on device.
#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub extern "C" fn android_main(android_app: winit::platform::android::activity::AndroidApp) {
    rlobkit_app_events::android_log::init(env!("CARGO_PKG_NAME"), "warn");
    rlobkit_app_events::system_bars::set_immersive_sticky(true);
    repose_core::locals::set_theme_default(repose_core::locals::Theme::default());
    crate::render::init_apk_assets(android_app.asset_manager());
    let files_dir = android_app.internal_data_path();
    let mut app = App::new();
    if let Some(dir) = files_dir.as_ref() {
        match app.load_assets_from(&dir.join("assets")) {
            Ok(()) => {
                log::info!("nt: assets loaded from {}", dir.join("assets").display());
            }
            Err(e) => {
                log::warn!("nt: files/assets load failed: {e:?}; trying default search");
                match app.load_assets() {
                    Ok(found) => {
                        log::info!("nt: assets loaded from {}", found.display());
                    }
                    Err(e2) => {
                        log::error!("nt: running without assets: {e2:?}; placeholder renderer");
                    }
                }
            }
        }
    } else {
        log::warn!("nt: no internal data path; trying default asset search");
        match app.load_assets() {
            Ok(found) => {
                log::info!("nt: assets loaded from {}", found.display());
            }
            Err(e) => {
                log::error!("nt: running without assets: {e:?}; placeholder renderer");
            }
        }
    }
    let save_path = match files_dir {
        Some(dir) => dir.join(crate::savedata_part::save_file_name()),
        None => crate::savedata_part::save_file_path(),
    };
    let save = app.load_save(&save_path);
    log::info!(
        "nt: save loaded from {} (version {})",
        save_path.display(),
        save.version
    );
    let mut audio = crate::audio_host::AudioHost::new();
    let mut last = web_time::Instant::now();
    if let Err(e) = repame_shell::run_android(android_app, move |sched, ctx| {
        // Android controller buttons land in the runner's pad map via the platform
        // `AndroidBackend`, which forwards South/East/Start/DPad as synthetic
        // keys (Space, Esc, Enter, arrows) into the normal key path -
        // repadio-shaped, no game-side pad bridge needed.
        let now = web_time::Instant::now();
        let dt = now
            .duration_since(last)
            .min(web_time::Duration::from_secs_f32(0.25));
        last = now;
        let view = root_view(sched, ctx, &mut app, dt);
        audio.pump(dt.as_secs_f32(), &mut app);
        // Touch-chrome verdict (logcat, ~1/s): sprites pushed this
        // frame + touch batch size + state. Proves the sticks/buttons
        // drew without a screenshot.
        {
            use web_time::{Duration, Instant};
            static LAST: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
            let tick = LAST.lock().map(|mut g| {
                let now = Instant::now();
                let due = g.is_none_or(|t| now.duration_since(t) >= Duration::from_secs(1));
                if due {
                    *g = Some(now);
                }
                due
            });
            if tick.unwrap_or(false) {
                let input_dbg = app
                    .sim
                    .world
                    .get_resource::<crate::input::NtInput>()
                    .map(|i| {
                        format!(
                            "mvA=({:.2},{:.2}) aimA=({:.2},{:.2}) fire={} fpress={} spec={} pick={} dis={:.1}",
                            i.move_axis.x,
                            i.move_axis.y,
                            i.aim_axis.x,
                            i.aim_axis.y,
                            i.fire_held,
                            i.peek_fire_pressed(),
                            i.spec_held,
                            i.peek_interact_pressed(),
                            i.touch_dis,
                        )
                    })
                    .unwrap_or_else(|| "no-input".to_string());
                let state = app
                    .sim
                    .world
                    .get_resource::<crate::state::AppState>()
                    .copied()
                    .unwrap_or_default();
                let mv = app
                    .sim
                    .world
                    .get_resource::<crate::input::NtInput>()
                    .and_then(|i| i.move_stick)
                    .map(|s| (s.anchor.x, s.anchor.y, s.touch, s.dis))
                    .unwrap_or((-1.0, -1.0, -99, -1.0));
                let at = app
                    .sim
                    .world
                    .get_resource::<crate::input::NtInput>()
                    .and_then(|i| i.attack_stick)
                    .map(|s| (s.anchor.x, s.anchor.y, s.touch, s.dis))
                    .unwrap_or((-1.0, -1.0, -99, -1.0));
                let vw = app.view_viewport_dp;
                let ws = app.view_world_size;
                log::info!(
                    "nt: touch sprites={} touch_batch={} state={state:?} move=({:.0},{:.0},t{},d{:.0}) atk=({:.0},{:.0},t{},d{:.0}) vw=({:.0},{:.0}) ws=({:.0},{:.0}) {input_dbg}",
                    app.last_sprite_count(),
                    app.last_touch_count(),
                    mv.0,
                    mv.1,
                    mv.2,
                    mv.3,
                    at.0,
                    at.1,
                    at.2,
                    at.3,
                    vw[0],
                    vw[1],
                    ws[0],
                    ws[1],
                );
            }
        }
        view
    }) {
        log::error!("nt: android run failed: {e:?}");
    }
}
