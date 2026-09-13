//! App states and run flags. Mirrors the bevy `AppState` machine; the
//! shell driver (not shown) transitions these, systems gate on them.
//!
//! State-layer port of `nt-recreated-bevy/src/app.rs` (transition laws
//! only, no rendering) plus `src/screens/mod.rs` (loading law):
//! - `AppState` boot order Splash -> MainMenu -> Loading -> Title ->
//!   InGame (same as the bevy build).
//! - `OverlayMenu` / `PendingUnpause` / `Paused` pause laws
//!   (`handle_pause_input`, `tick_pending_unpause`, `reset_pause_on_exit`,
//!   `force_death_overlay_state`).
//! - `SplashState` boot-intro law (`game/ui_art.rs::boot_intro`:
//!   modes 0-3 advance on press or `MODE_SECS` timeout, mode 4 plays the
//!   logo gunfire and leaves on press).
//! - `LoadingState` loading law (`screens/mod.rs::tick_loading`: 1.2 s
//!   minimum, then InGame via the existing `setup_run` entry).
//!
//! Splash auto-advance is opt-in ([`SplashAutoAdvance`]): bevy parity is
//! press-only everywhere, and unattended boots (tests, kiosk) insert
//! `SplashAutoAdvance(true)` to leave the logo ~1 s after the gun reel.
//!
//! Fidelity compromises (need shell/window services):
//! - Animated `Transition<AppState>` (fade/circle wipe, `block_input`)
//!   is deferred to the repose shell: `goto_state` transitions
//!   instantly. `tick_escape_pause` still takes a `block_input` flag so
//!   the shell can gate it once transitions exist.
//! - Asset-gated loading progress (`AssetsLoading` + `AssetServer`) is
//!   headless-complete (progress = 1.0); only the 1.2 s floor remains.
//! - Splash logo gunfire SFX/shake/sprites are render; the headless
//!   `SplashState` keeps mode + timer + gun count. The press path is
//!   bevy-verbatim (mode 4 leaves only on press); the timed auto-advance
//!   additionally requires [`SplashAutoAdvance`], so unattended boots
//!   reach the menu without changing attended UX.
//! - `QuitApp` has no window service headless: it sets `QuitRequested`,
//!   which the shell polls.
//! - Locale/i18n (`LocaleResources`) is shell-side; language gating uses
//!   the `AVAILABLE_LANGUAGES` list in `menus` (same codes as bevy
//!   `LOCALES`).

use bevy_ecs::prelude::*;
use repame_sim::SimTime;

use crate::time::{GTimer, TimerMode};

/// Menu state machines (character/loadout/mutation/pause/settings/
/// unlock/game-over). Lives here as `state::menus` so `lib.rs` stays
/// untouched.
#[path = "menus.rs"]
pub mod menus;

/// Top-level app state. Boot order: Splash -> MainMenu -> Loading ->
/// Title -> InGame (same as the bevy build).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Resource)]
pub enum AppState {
    #[default]
    Splash,
    MainMenu,
    Loading,
    Title,
    InGame,
}

/// Pause flag. Sim systems early-out while set (except `Always` sets,
/// which mirror the bevy build's unticked-by-pause selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub struct Paused(pub bool);

/// Fixed-step frame counter (bevy `CurrentFrame` equivalent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub struct CurrentFrame(pub u64);

pub fn tick_current_frame(mut frame: ResMut<CurrentFrame>) {
    frame.0 = frame.0.wrapping_add(1);
}

/// Pause overlay selector (bevy `OverlayMenu` verbatim, minus rendering).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub enum OverlayMenu {
    #[default]
    None,
    Settings,
    Credits,
    Pause,
    /// Run-stats panel over the main menu (GML `DrawStats` parity;
    /// bevy left STATS inert, so this variant is port-only).
    Stats,
}

/// Delayed unpause (bevy `PendingUnpause` verbatim law: 0.2 s `Once`
/// timer armed by Resume/CloseOverlay/Escape, on expiry clears itself
/// and unpauses; `GTimer` replaces bevy `Timer`).
#[derive(Debug, Clone, Default, Resource)]
pub struct PendingUnpause(pub Option<GTimer>);

/// Delay bevy arms before lifting pause (Resume, CloseOverlay on the
/// pause overlay, Escape out of pause).
pub const UNPAUSE_DELAY_SECS: f32 = 0.2;

/// Boot-intro state (bevy `BootState` mode/timer half in
/// `game/ui_art.rs`; entities/sprites/audio deferred to render).
#[derive(Debug, Clone, Resource)]
pub struct SplashState {
    pub mode: u8,
    pub t: f32,
    pub guns: u8,
}

impl Default for SplashState {
    fn default() -> Self {
        Self {
            mode: 0,
            t: 0.0,
            guns: 0,
        }
    }
}

/// Per-mode auto-advance timeouts (GML `Vlambeer/Create_0` + `Alarm_0`:
/// mode 0 runs 120 steps, then 60 per mode with +60 on mode 2, at
/// 30 steps/s => [4, 2, 4, 2] s; bevy `MODE_SECS` verbatim).
pub const SPLASH_MODE_SECS: [f32; 4] = [4.0, 2.0, 4.0, 2.0];

/// Logo gunfire step times (bevy `STEP_T` verbatim, mode 4).
pub const SPLASH_GUN_STEPS: [f32; 7] = [
    1.0,
    1.0 + 2.0 / 30.0,
    1.0 + 4.0 / 30.0,
    1.0 + 6.0 / 30.0,
    1.0 + 8.0 / 30.0,
    1.0 + 10.0 / 30.0,
    1.0 + 10.0 / 30.0 + 20.0 / 30.0,
];

/// Headless hold after the gun sequence before auto-advancing (bevy
/// has none — it waits for a press; applies only with
/// [`SplashAutoAdvance`] set, see [`tick_splash`]).
pub const SPLASH_LOGO_HOLD_SECS: f32 = 1.0;

/// Opt-in unattended splash advance (tests, kiosk shells). Absent or
/// `false` (the default): mode 4 is press-only, bevy parity. `true`:
/// mode 4 also leaves ~[`SPLASH_LOGO_HOLD_SECS`] after the gun reel, so
/// boots with no input source still reach the menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub struct SplashAutoAdvance(pub bool);

/// Loading-screen state (bevy `LoadingTimer` half of
/// `screens/mod.rs`; asset handles deferred, progress headless-1.0).
#[derive(Debug, Clone, Resource)]
pub struct LoadingState {
    pub t: f32,
    pub progress: f32,
    /// GML `GenCont` tip verbatim (`draw_text_nt(_cx, _cy + 24, "@s" +
    /// tip)`): picked once per load so it stays stable across draws
    /// (lazy-filled by the loading text layer; empty until first draw).
    pub tip: String,
}

impl Default for LoadingState {
    fn default() -> Self {
        Self {
            t: 0.0,
            progress: 1.0,
            tip: String::new(),
        }
    }
}

/// Minimum loading-screen time (bevy `LoadingTimer(1.2 s)` verbatim).
pub const LOADING_MIN_SECS: f32 = 1.2;

/// Headless quit signal (bevy `AppExit::Success`; no window service
/// headless, so the shell polls this).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub struct QuitRequested(pub bool);

/// Scene-transition input block (bevy `Transition<AppState>.block_input`
/// parity). The port flips states instantly (`goto_state`), so no system
/// ever raises this — it exists so `gameplay_active` keeps the bevy
/// gate shape instead of silently dropping a conjunct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub struct TransitionBlock(pub bool);

/// Instant state transition with bevy `reset_pause_on_exit` side
/// effects (paused/overlay/pending cleared; `Run::game_over` cleared
/// when leaving InGame or entering a menu state; menu transients
/// cleared). Replaces bevy `NextState` + animated `Transition`
/// (deferred to the shell — see module docs).
pub fn goto_state(world: &mut World, next: AppState) {
    let prev = world
        .get_resource::<AppState>()
        .copied()
        .unwrap_or_default();
    if prev == next {
        return;
    }
    world.insert_resource(next);
    reset_pause_state(world);
    if prev == AppState::InGame
        || matches!(
            next,
            AppState::Splash | AppState::MainMenu | AppState::Title
        )
    {
        if let Some(mut run) = world.get_resource_mut::<crate::comps_a::Run>() {
            run.game_over = false;
        }
    }
    match next {
        AppState::MainMenu => {
            world.init_resource::<menus::MenuState>();
            if let Some(mut menu) = world.get_resource_mut::<menus::MenuState>() {
                menu.main_menu_cursor = 0;
                menu.settings_cursor = 0;
                menu.play_submenu = false;
                menu.play_cursor = 0;
            }
        }
        AppState::Loading => {
            world.insert_resource(LoadingState::default());
        }
        AppState::Title => {
            let gml = world
                .get_resource::<crate::comps_a::SelectedCharacter>()
                .map(|s| s.0 as usize)
                .unwrap_or(crate::data::RaceId::Fish as usize);
            let cursor =
                menus::roster_position(world.get_resource::<crate::savedata_part::SaveData>(), gml)
                    .unwrap_or(0);
            world.init_resource::<menus::MenuState>();
            if let Some(mut menu) = world.get_resource_mut::<menus::MenuState>() {
                menu.title_go_visible = false;
                menu.loadout_open = false;
                menu.title_cursor = cursor;
                menu.mutation_selected = None;
            }
            // GML `MenuGen`: the title screen is a real campfire floor.
            crate::setup::setup_title_campfire(world);
        }
        AppState::Splash => {
            world.insert_resource(SplashState::default());
        }
        _ => {}
    }
}

/// Bevy `reset_pause_on_exit` verbatim (pause/overlay/pending/menu
/// transients cleared; caller owns the `Run::game_over` edge, see
/// `goto_state`).
pub fn reset_pause_state(world: &mut World) {
    world.init_resource::<Paused>();
    world.init_resource::<OverlayMenu>();
    world.init_resource::<PendingUnpause>();
    world.init_resource::<menus::MenuState>();
    if let Some(mut paused) = world.get_resource_mut::<Paused>() {
        paused.0 = false;
    }
    if let Some(mut overlay) = world.get_resource_mut::<OverlayMenu>() {
        *overlay = OverlayMenu::None;
    }
    if let Some(mut pending) = world.get_resource_mut::<PendingUnpause>() {
        pending.0 = None;
    }
    if let Some(mut menu) = world.get_resource_mut::<menus::MenuState>() {
        menu.pause_confirm = None;
        menu.settings_page = 0;
        menu.settings_page_stack.clear();
        menu.mutation_selected = None;
        menu.game_over = None;
        menu.settings_cursor = 0;
        menu.play_submenu = false;
        menu.play_cursor = 0;
    }
}

/// Splash tick (bevy `boot_intro` state half verbatim, plus the
/// opt-in logo-hold auto-advance for unattended boots). `pressed` = any
/// key/mouse edge this tick (bevy: any just-pressed key or mouse
/// button). Only runs in `Splash`; finishing enters `MainMenu`.
pub fn tick_splash(world: &mut World, dt: f32, pressed: bool) {
    if world
        .get_resource::<AppState>()
        .copied()
        .unwrap_or_default()
        != AppState::Splash
    {
        return;
    }
    world.init_resource::<SplashState>();
    let auto = world
        .get_resource::<SplashAutoAdvance>()
        .is_some_and(|a| a.0);
    let done = {
        let mut splash = world.resource_mut::<SplashState>();
        if splash.mode < 4 {
            splash.t += dt;
            let advance = pressed || splash.t >= SPLASH_MODE_SECS[splash.mode as usize];
            if advance {
                splash.mode += 1;
                splash.t = 0.0;
                splash.guns = 0;
            }
            false
        } else {
            splash.t += dt;
            while (splash.guns as usize) < SPLASH_GUN_STEPS.len()
                && splash.t >= SPLASH_GUN_STEPS[splash.guns as usize]
            {
                splash.guns += 1;
            }
            if pressed {
                if splash.guns == 0 {
                    // Bevy fast-forward: jump near the sequence start.
                    splash.t = splash.t.max(1.0 - 10.0 / 30.0);
                    false
                } else {
                    true
                }
            } else if splash.guns as usize >= SPLASH_GUN_STEPS.len()
                && splash.t >= SPLASH_GUN_STEPS[SPLASH_GUN_STEPS.len() - 1] + SPLASH_LOGO_HOLD_SECS
                && auto
            {
                true
            } else {
                false
            }
        }
    };
    if done {
        goto_state(world, AppState::MainMenu);
    }
}

/// Loading tick (bevy `tick_loading` law: assets at headless-1.0, wait
/// out the 1.2 s floor, then InGame through the existing `setup_run`
/// entry — never duplicated here). Only runs in `Loading`.
pub fn tick_loading(world: &mut World, dt: f32) {
    if world
        .get_resource::<AppState>()
        .copied()
        .unwrap_or_default()
        != AppState::Loading
    {
        return;
    }
    world.init_resource::<LoadingState>();
    let ready = {
        let mut loading = world.resource_mut::<LoadingState>();
        loading.t += dt;
        loading.progress = 1.0;
        loading.t >= LOADING_MIN_SECS
    };
    if ready {
        crate::setup::setup_run(world);
        reset_pause_state(world);
    }
}

/// Escape-pause tick (bevy `handle_pause_input` verbatim, minus engine
/// key reads: the shell passes `escape_pressed`). Gated to InGame,
/// `block_input` (transition animation, shell-owned), and live runs
/// (game-over swallows Escape, bevy parity).
pub fn tick_escape_pause(
    paused: &mut Paused,
    overlay: &mut OverlayMenu,
    pending: &mut PendingUnpause,
    menu: &mut menus::MenuState,
    game_over: bool,
    block_input: bool,
    escape_pressed: bool,
) {
    if !escape_pressed || block_input || game_over {
        return;
    }
    match *overlay {
        OverlayMenu::None if !paused.0 => {
            paused.0 = true;
            *overlay = OverlayMenu::Pause;
            pending.0 = None;
            menu.pause_confirm = None;
            menu.settings_page = 0;
            menu.settings_page_stack.clear();
        }
        OverlayMenu::Pause => {
            if menu.pause_confirm.is_some() {
                // Bevy: Escape dismisses the quit/restart confirm first.
                menu.pause_confirm = None;
                return;
            }
            *overlay = OverlayMenu::None;
            pending.0 = Some(GTimer::from_seconds(UNPAUSE_DELAY_SECS, TimerMode::Once));
        }
        OverlayMenu::Settings | OverlayMenu::Credits => {
            if !menu.settings_page_stack.is_empty() || menu.settings_page != 0 {
                if let Some(prev) = menu.settings_page_stack.pop() {
                    menu.settings_page = prev;
                } else {
                    menu.settings_page = 0;
                }
                menu.settings_cursor = 0;
            } else if paused.0 {
                *overlay = OverlayMenu::Pause;
                menu.settings_page = 0;
                menu.settings_page_stack.clear();
                menu.settings_cursor = 0;
            } else {
                *overlay = OverlayMenu::None;
                menu.settings_cursor = 0;
            }
        }
        // `None` while already paused (Resume path owns that edge).
        OverlayMenu::None => {}
        // Stats only ever opens over the main menu (never InGame, so
        // this arm is unreachable in practice); close it like Credits.
        OverlayMenu::Stats => {
            *overlay = OverlayMenu::None;
        }
    }
}

/// Pending-unpause tick (bevy `tick_pending_unpause` verbatim over
/// `SimTime`).
pub fn tick_pending_unpause(
    time: Res<SimTime>,
    mut pending: ResMut<PendingUnpause>,
    mut paused: ResMut<Paused>,
) {
    let Some(timer) = pending.0.as_mut() else {
        return;
    };
    timer.tick(time.delta_secs);
    if timer.just_finished() {
        pending.0 = None;
        paused.0 = false;
    }
}

/// Death overlay guard (bevy `force_death_overlay_state` verbatim:
/// game-over forces unpaused + no overlay + no pending while InGame).
pub fn force_death_overlay_state(
    state: Res<AppState>,
    run: Option<Res<crate::comps_a::Run>>,
    mut paused: ResMut<Paused>,
    mut overlay: ResMut<OverlayMenu>,
    mut pending: ResMut<PendingUnpause>,
) {
    if *state != AppState::InGame {
        return;
    }
    let Some(run) = run else {
        return;
    };
    if run.game_over {
        paused.0 = false;
        *overlay = OverlayMenu::None;
        pending.0 = None;
    }
}

/// Channel sync (bevy `sync_shared_ui` audio half verbatim:
/// master/sfx/music follow settings every tick).
pub fn sync_audio_channels(
    save: Res<crate::savedata_part::SaveData>,
    mut channels: ResMut<crate::audio::AudioChannels>,
) {
    channels.master = save.settings.master_volume;
    channels.sfx = save.settings.sfx_volume;
    channels.music = save.settings.music_volume;
}

/// Save sanitize (bevy `sanitize_save` headless law: version mismatch
/// sanitizes loadouts and stamps; covers both the migrate and the
/// added-save arms without engine change detection).
pub fn tick_sanitize_save(mut save: ResMut<crate::savedata_part::SaveData>) {
    if save.version != crate::savedata_part::SAVE_VERSION {
        save.sanitize_loadouts();
        save.version = crate::savedata_part::SAVE_VERSION;
    }
}
