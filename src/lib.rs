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
//!   Space / left-click -> fire, E/F/Q/G/Tab/Enter ->
//!   interact/confirm, 1-4 -> weapon slots / mutation picks / title cursor,
//!   Left/Right arrows -> title cursor + mutation highlight, click in game ->
//!   aim-at-click + fire, click in menus -> confirm (pause/settings/
//!   credits buttons are clickable and stage `UiAction`s instead; bare
//!   viewport clicks over those menus are dropped), Esc -> pause toggle,
//!   R -> game-over retry.
//! - NOT wired (needs shell services this repose version lacks): held-mouse
//!   continuous fire (clicks are single-frame edges; hover aims every
//!   frame), right-mouse / Shift ability+spec pulses (`Key` has no Shift
//!   variant and picks carry no button), gamepad sticks/triggers, settings sliders/buttons (pause
//!   menu buttons work via screen-position routing; settings rows are display-only).
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
use repose_core::input::{Key, KeyEvent, KeyEventType};
use repose_core::prelude::{AlignItems, Modifier};
use repose_core::{
    Color, Dp, FocusRequester, RenderContext, Scheduler, Sp, View, remember, request_frame,
};
use repose_render_wgpu::Callback;
use repose_ui::{Box as UiBox, Column, Text, TextStyle, ViewExt, ZStack};

use crate::audio::UiAction;
use crate::comps_a::CurrentFrame as CombatFrame;
use crate::comps_a::{NT_CAM_SCALE, Player, Projectile, WallCell, WallTile};
use crate::comps_b::{Enemy, Pickup, Prop};
use crate::data::AreaId;
use crate::input::{
    GamepadState, KeyCode, MouseState, NtInput, TouchContact, sample_gamepads, sample_keyboard,
    sample_mutation_digits, sample_touch,
};
use crate::render::{
    ATLAS_PAGES, ATLAS_SIZE, CamPoi, CamStepInput, GmlCamera, RenderAssets, background_color,
    bloom_sprites, cam_viewdist_for, crosshair_sprites, decode_png, fainted_bar_sprites,
    fog_sprites, fx_instances, fx_texts, gml_camera_step, gml_view_scale, gml_view_size,
    hud_gui_texts_dp, hud_sprites, menu_gui_texts, menu_gui_texts_dp, menu_gui_texts_vw,
    menu_sprites, portal_indicator_sprites, shadow_sprites, sideart_sprites, spiral_figures,
    splash_sprites, view_rect_world, world_camera, world_instances,
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
    pause_edge: bool,
    restart_edge: bool,
    interact_edge: bool,
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
            spiral,
            assets: None,
            assets_dir: None,
            vortex_tex: Vec::new(),
            vortex_tex_area: None,
            held: HashSet::new(),
            edges: Vec::new(),
            clicks: Vec::new(),
            hover: None,
            pause_edge: false,
            restart_edge: false,
            interact_edge: false,
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

    /// Vortex background textures for a GML area (decoded once per
    /// area; slots match the shader bindings: spiral, bolt, debris,
    /// proto, idpd, idpd2). Missing files fall back to area 0 debris
    /// so the pass always has six bound slots.
    fn load_vortex_textures(dir: &Path, gml_area: u8) -> Vec<VortexTexture> {
        fn decode(dir: &Path, name: &str) -> Option<(u32, u32, Vec<u8>)> {
            decode_png(&dir.join("images").join(format!("{name}.png"))).ok()
        }
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
            self.spiral.step(1.0);
            self.maybe_rewarm_spiral();
            ran += 1;
        }
        ran
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
        // a window edge and never feels like the original).
        let (aim_dir, aim_dis) = match self.hover {
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
        let run = self.sim.world.get_resource::<crate::comps_a::Run>();
        let (area, seed) = run
            .map(|r| (r.area, r.gen_seed))
            .unwrap_or((AreaId::Desert, 0));
        if gml_area_for_area(area) != self.spiral.gml_area {
            self.spiral = SpiralCtl::warmed_up_for_area_seeded(area, seed);
        }
    }

    /// Stage one shell key event (called from the root `on_key_event`
    /// handler; see the module docs for the mapping table).
    fn handle_key(&mut self, ke: &KeyEvent) {
        let down = matches!(ke.event_type, KeyEventType::Down);
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
        use crate::comps_a::MutationChoice;

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
        let game_over = self
            .sim
            .world
            .get_resource::<crate::comps_a::Run>()
            .is_some_and(|r| r.game_over)
            || self
                .sim
                .world
                .get_resource::<MenuState>()
                .is_some_and(|m| m.game_over.is_some());
        let offer_open = state == AppState::InGame
            && self
                .sim
                .world
                .get_resource::<MenuState>()
                .is_some_and(|m| m.mutation_count > 0);
        // Live gameplay: no pause, no overlay, no game over. Only here
        // do hover/clicks steer aim and fire; over menus the buttons
        // stage `UiAction`s instead (a bare viewport click does nothing
        // so it can never resume through a MENU/RETRY press).
        let live_play = state == AppState::InGame && !offer_open && !paused && !game_over;
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

        let mouse_down = !self.clicks.is_empty() && !menu_open && !game_over;
        let mouse = MouseState {
            left_held: mouse_down,
            left_pressed: mouse_down,
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
                let contacts: Vec<TouchContact> = self
                    .touch_active
                    .iter()
                    .map(|(id, (start, pos))| TouchContact {
                        start: *start,
                        pos: *pos,
                        just_pressed: self.touch_new.contains(id),
                    })
                    .collect();
                self.touch_new.clear();
                sample_touch(&contacts, self.view_width, &mut input);
            }
        }
        // Raw Digit1-4 mutation protocol (direct `MutationChoice` write).
        {
            let mut choice = self.sim.world.resource_mut::<MutationChoice>();
            sample_mutation_digits(&just, &mut choice);
        }
        // Shell edges with no `KeyCode` (Esc / R / Enter).
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

        // Per-frame cursor aim (bevy `player_aim` mouse path): the latest
        // hover steers `aim_axis` every tick, not just on clicks, so
        // `AimDir` tracks the cursor continuously (the follow camera
        // reads the hover distance directly as GML `dis_fire`).
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
                if let Some(hover) = self.hover {
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
            // Open menu over a live run: route the click at the menu
            // buttons, then drop it either way.
            if let Some(click) = self.clicks.last().copied() {
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
            // GML `GameOver`: any click retries through the same
            // `MenuEdge` restart as KeyR (Loading).
            if self.clicks.last().copied().is_some() {
                self.sim.world.resource_mut::<MenuEdge>().restart_pressed = true;
            }
            self.clicks.clear();
        } else if let Some(_click) = self.clicks.last().copied() {
            self.clicks.clear();
            self.sim.world.resource_mut::<NtInput>().press_interact();
        } else {
            self.clicks.clear();
        }
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
            let game_over = run_game_over || menu.game_over.is_some();
            menu_overlay_kind(state, overlay, &menu, game_over)
        };
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
            self.sim
                .world
                .insert_resource(crate::render::HoverWorld(self.hover));
            // Blob shadows first (GML `shad` surface: under the actors).
            let mut s = shadow_sprites(&mut self.sim.world, assets);
            s.extend(world_instances(&mut self.sim.world, assets));
            s.extend(fx_instances(&mut self.sim.world, assets));
            // Additive bloom over the world (GML `scrDrawBloom`, gated
            // on `opt_bloom` inside).
            s.extend(bloom_sprites(&mut self.sim.world, assets));
            // View-anchored HUD (bevy camera-child GUI parity): the rect
            // is computed first so bars/icons land on the live view.
            let hud_view = view_rect_world(viewport_dp, world_size, &self.cam);
            // Ghost decay runs on the render dt (GML steps it per sim
            // tick at 30 Hz).
            let hud_dt = dt.as_secs_f32().clamp(0.0, 0.1);
            s.extend(hud_sprites(&mut self.sim.world, assets, hud_view, hud_dt));
            // Area fog over the room (GML TopCont/Draw_0, sewers only)
            // plus coop fainted bars at the view-clamped positions.
            s.extend(fog_sprites(
                &mut self.sim.world,
                assets,
                viewport_dp,
                world_size,
                &self.cam,
            ));
            let view = view_rect_world(viewport_dp, world_size, &self.cam);
            s.extend(fainted_bar_sprites(&mut self.sim.world, assets, view));
            // World crosshair + offscreen portal arrow (GML
            // `TopCont/Draw_0`, over the room, under the HUD text).
            s.extend(crosshair_sprites(&mut self.sim.world, assets, hud_dt));
            s.extend(portal_indicator_sprites(
                &mut self.sim.world,
                assets,
                viewport_dp,
                world_size,
                &self.cam,
            ));
            // Boot reel (`Vlambeer/Draw_0` + `Logo/Draw_0`).
            if menu_kind == Some(MenuOverlay::Splash) {
                s.extend(splash_sprites(&mut self.sim.world, assets, view));
            }
            // Menu art sprites (char pods, portrait, loadout, splats).
            if let Some(kind) = menu_kind {
                s.extend(menu_sprites(
                    kind,
                    &mut self.sim.world,
                    assets,
                    viewport_dp,
                    world_size,
                    &self.cam,
                ));
            }
            // Spiral CPU layer (GML `scrDrawSpiral` center figures):
            // crown orbit + player hurt figures over the vortex
            // background, i.e. whenever the live-gameplay background is
            // off.
            let live = state == AppState::InGame && menu_kind.is_none() && !paused && !game_over;
            if !live {
                s.extend(spiral_figures(
                    &mut self.sim.world,
                    assets,
                    self.cam.center,
                    self.spiral.angle,
                ));
            }
            // Sideart chrome around the view (GML `UberCont/Draw_74`:
            // over everything, game and menus alike).
            s.extend(sideart_sprites(
                &mut self.sim.world,
                assets,
                viewport_dp,
                world_size,
                &self.cam,
            ));
            let texts = fx_texts(&mut self.sim.world);
            (s, texts)
        } else {
            (placeholder_instances(&mut self.sim.world), Vec::new())
        };
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
        // shell-selected (1 over live gameplay, 0 over menus); the
        // snapshot drives `VortexPass` (GML `scrDrawSpiral` fullscreen
        // quad) mounted as the bottom layer, so the swirling portal
        // shows behind sprites in game and menus alike (pause/game
        // over dim it through the viewport overlay).
        let bg_alpha = if state == AppState::InGame && menu_kind.is_none() && !paused && !game_over
        {
            1.0
        } else {
            0.0
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
        let vortex_layer = if self.assets.is_some()
            && !self.vortex_tex.is_empty()
            && !matches!(menu_kind, Some(MenuOverlay::Splash))
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
        // Pre-run rooms (splash boot reel, logo menu) sit on black
        // (GML `Vlambeer/Draw_0` clears black); the area fill only
        // shows once a run exists. No flat fill under the mounted
        // vortex pass (it would cover the spiral).
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
        );
        let overlay_color = crate::effects::flash_rgba(&self.sim.world);

        let hud_rows = if state == AppState::InGame {
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
                ZStack(Modifier::new().fill_max_size())
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
        // No bars over the pre-run rooms (boot reel, logo menu and
        // campfire run on a bare view; GML only letterboxes gameplay
        // menus, transitions and intros).
        let bare_room = matches!(
            menu_kind,
            Some(
                MenuOverlay::Splash
                    | MenuOverlay::MainMenu
                    | MenuOverlay::Title
                    | MenuOverlay::Loading
            )
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
            layers.push(Column(Modifier::new().fill_max_size()).child(vec![bar(), spacer, bar()]));
        }
        if let Some(rows) = menu_rows {
            // Bevy `scrim` / game-over panel parity: Pause/Settings/
            // Credits/GameOver sit on near-opaque black (230/255), so
            // the frozen room barely reads through. Title/Mutation/
            // MainMenu/Splash/Loading read through to the spiral/pods
            // underneath.
            if dim_menu {
                layers.push(UiBox(
                    Modifier::new()
                        .fill_max_size()
                        .background(Color::from_rgba(0, 0, 0, 230))
                        .hit_passthrough(),
                ));
            }
            if !rows.is_empty() {
                // Plain text rows (GML draws the buttons as text;
                // clicks route by dp position in `feed_input` via
                // `route_menu_click`, repose-hit-test independent).
                layers.push(
                    ZStack(Modifier::new().fill_max_size())
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
    let blink = world
        .get_resource::<crate::SimTime>()
        .map(|t| (t.elapsed_secs * 12.0).sin() > 0.0)
        .unwrap_or(true);
    hud_gui_texts_dp(world, canvas_dp)
        .into_iter()
        .filter(|(t, _, _, _, _, _, _)| t != "LOW HP" || blink)
        .collect()
}

/// Positioned overlay text layer (bevy `nt_text_at` placement
/// verbatim): a full-size padded column holding the fixed-width box
/// the Silkscreen line sits in. Right-aligned rows (misc-HUD clock/
/// area) right-align inside their box.
fn gui_text_layer(row: &crate::render::GuiRow) -> View {
    let align = if row.4 {
        AlignItems::CENTER
    } else if row.6 {
        AlignItems::FLEX_END
    } else {
        AlignItems::FLEX_START
    };
    let c = Color::from_rgba(
        (row.2[0] * 255.0) as u8,
        (row.2[1] * 255.0) as u8,
        (row.2[2] * 255.0) as u8,
        (row.2[3] * 255.0) as u8,
    );
    let mut text = Text(row.0.clone())
        .size(Sp(row.3))
        .font_family(NT_UI_FONT_FAMILY)
        .color(c)
        .single_line();
    if row.6 {
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
            .align_items(AlignItems::FLEX_START),
    )
    .child(Column(Modifier::new().width(Dp(row.5)).align_items(align)).child(text))
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
    let confirm = world
        .get_resource::<MenuState>()
        .and_then(|m| m.pause_confirm);
    menu_gui_texts_vw(kind, world, vw)
        .into_iter()
        .find_map(|t| {
            let action = menu_button_action(kind, &t.text, confirm)?;
            // Bevy `bigname_button_at` parity: every menu button owns a
            // fixed 120x22 GUI box centered on its (gx, gy) (the dp text
            // boxes are full-width alignment boxes and overlap, so they
            // can never route). The pause columns and rows stay disjoint
            // at this size.
            const HW: f32 = 60.0;
            const HH: f32 = 11.0;
            let gx = dp[0] / k;
            let gy = dp[1] / k;
            if (gx - t.gx).abs() <= HW && (gy - t.gy).abs() <= HH {
                Some(action)
            } else {
                None
            }
        })
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
}

pub fn menu_overlay_kind(
    state: AppState,
    overlay: OverlayMenu,
    menu: &MenuState,
    game_over: bool,
) -> Option<MenuOverlay> {
    match state {
        AppState::Splash => Some(MenuOverlay::Splash),
        AppState::MainMenu => Some(MenuOverlay::MainMenu),
        AppState::Loading => Some(MenuOverlay::Loading),
        AppState::Title => Some(MenuOverlay::Title),
        AppState::InGame => {
            if game_over || menu.game_over.is_some() {
                return Some(MenuOverlay::GameOver);
            }
            match overlay {
                OverlayMenu::Pause => Some(MenuOverlay::Pause),
                OverlayMenu::Settings => Some(MenuOverlay::Settings),
                OverlayMenu::Credits => Some(MenuOverlay::Credits),
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

