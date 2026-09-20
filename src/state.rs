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

/// Tutorial steps (GML `TutCont/Create_0` `TutorialStep` verbatim:
/// Walking=1, PickingUp, Shooting, Swapping, Power, Fin, NUM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TutorialStep {
    #[default]
    Walking = 1,
    PickingUp = 2,
    Shooting = 3,
    Swapping = 4,
    Power = 5,
    Fin = 6,
}

impl TutorialStep {
    pub fn next(self) -> Self {
        match self {
            TutorialStep::Walking => TutorialStep::PickingUp,
            TutorialStep::PickingUp => TutorialStep::Shooting,
            TutorialStep::Shooting => TutorialStep::Swapping,
            TutorialStep::Swapping => TutorialStep::Power,
            TutorialStep::Power => TutorialStep::Fin,
            TutorialStep::Fin => TutorialStep::Fin,
        }
    }
}

/// Tutorial controller state (GML `objects/TutCont` verbatim, minus
/// visuals): the scripted first-floor walkthrough. `step` is the
/// current step, `complete` latches the 30-step advance (`Alarm_0`),
/// `timer` counts it down, `portal_open` latches the Fin exit (GML
/// spawns the `Portal` past `Fin`; that portal runs `game_restart()`,
/// not a floor advance).
#[derive(Debug, Clone, Resource)]
pub struct TutorialState {
    pub step: TutorialStep,
    pub complete: bool,
    pub timer: GTimer,
    pub portal_open: bool,
}

impl Default for TutorialState {
    fn default() -> Self {
        Self {
            step: TutorialStep::Walking,
            complete: false,
            timer: GTimer::from_seconds(0.0, TimerMode::Once),
            portal_open: false,
        }
    }
}

impl TutorialState {
    /// GML `complete_step` verbatim: only the current step latches,
    /// once (`alarm[0] = 30`).
    pub fn complete_step(&mut self, step: TutorialStep) {
        if self.step == step && !self.complete {
            self.complete = true;
            self.timer = GTimer::from_seconds(30.0 / 30.0, TimerMode::Once);
        }
    }
}

/// Marker requesting a tutorial-exit run restart (GML
/// `Portal/Alarm_1` `game_restart()` arm). `tick_menus` consumes it
/// into the `Loading` path (same as death RETRY); the flag lives on
/// the player so no new resource threads the 16-param system cap.
#[derive(Debug, Clone, Copy, Default, bevy_ecs::prelude::Component)]
pub struct TutorialRestart;

/// Tutorial advance driver: [`tick_tutorial`] needs `&mut World`, so it
/// cannot sit in the schedule directly. Runs it from the headless
/// world each fixed step (GML `TutCont/Alarm_0` cadence).
pub fn drive_tutorial(world: &mut World) {
    let dt = world
        .get_resource::<SimTime>()
        .map(|t| t.delta_secs)
        .unwrap_or(1.0 / 30.0);
    tick_tutorial(world, dt);
}

/// Tutorial advance tick (GML `TutCont/Alarm_0` verbatim, minus the
/// scripted `WeaponChest` spawn which rides worldgen): on timer expiry
/// clear the latch, step forward, re-arm 45 steps on `Fin`, and past
/// `Fin` latch the exit portal open (once). Clamps at `Fin` (GML
/// `NUM - 1`).
pub fn tick_tutorial(world: &mut World, dt: f32) {
    let run_tutorial = world
        .get_resource::<crate::comps_a::Run>()
        .is_some_and(|r| r.tutorial);
    if !run_tutorial {
        return;
    }
    world.init_resource::<TutorialState>();
    world.init_resource::<crate::comps_a::Toast>();
    let mut tut = world.resource_mut::<TutorialState>();
    tut.timer.tick(dt);
    if tut.complete && tut.timer.just_finished() {
        tut.complete = false;
        tut.step = tut.step.next();
        if tut.step == TutorialStep::Fin {
            tut.timer = GTimer::from_seconds(45.0 / 30.0, TimerMode::Once);
            tut.portal_open = true;
            world.resource_mut::<crate::comps_a::Toast>().show("COOL, WE'RE DONE HERE!");
        }
    }
}

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

/// GML `MakeGame` boot flags verbatim (`Create_0` + `Alarm_0` + `Vlambeer`
/// recontinue arm): the fan-recreation disclaimer gate, the save-continue
/// roadmap, and the recontinue cap. Headless defaults preserve the
/// current boot (disclaimer accepted, no save file on disk) so existing
/// shells/tests boot straight to the menu; shells with disk set the
/// fields before the first tick.
#[derive(Debug, Clone, Resource)]
pub struct BootFlags {
    /// GML `save etc/disclaimer`: once true the disclaimer screen never
    /// shows again. Default true headless (accepted).
    pub disclaimer_accepted: bool,
    /// GML `disclaimer` frame counter: `CLICK TO CONTINUE` appears at
    /// `>= 90` steps.
    pub disclaimer_t: u32,
    /// GML `file_exists(savegame_file)`: a continued run waits on the
    /// roadmap prompt instead of booting to the menu.
    pub has_save_file: bool,
    /// GML `global.recontinued_times`: more than 2 recontinuations per
    /// level deletes the save.
    pub recontinued_times: u32,
    /// GML `UberCont.continued_run`: set while loading a save.
    pub continued_run: bool,
    /// GML `UberCont.want_quit_to_menu`: quit-to-menu takes the Logo vs
    /// MenuGen fast path instead of the normal boot.
    pub want_quit_to_menu: bool,
}

impl Default for BootFlags {
    fn default() -> Self {
        Self {
            disclaimer_accepted: true,
            disclaimer_t: 0,
            has_save_file: false,
            recontinued_times: 0,
            continued_run: false,
            want_quit_to_menu: false,
        }
    }
}

/// GML `MakeGame/Draw_0` disclaimer tick verbatim over 30 Hz steps:
/// returns true once the run may continue (accepted + past 90 frames +
/// pressed). The shell owns the prompt text/layout.
pub fn tick_disclaimer(flags: &mut BootFlags, dt_steps: f32, pressed: bool) -> bool {
    if flags.disclaimer_accepted && flags.disclaimer_t >= 90 {
        return true;
    }
    flags.disclaimer_t = flags.disclaimer_t.saturating_add(dt_steps.max(0.0) as u32);
    if flags.disclaimer_t >= 90 && pressed {
        flags.disclaimer_accepted = true;
        return true;
    }
    false
}

/// GML `Vlambeer/Create_0` level-entry choice verbatim: after a room
/// start with a live `GameCont`, skill/crown/ultra points open `LevCont`
/// (the mutation/crown draft); otherwise `GenCont` builds the floor.
/// `patiencepick` suppresses the skill arm on level continuations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelEntry {
    LevCont,
    GenCont,
}

pub fn choose_level_entry(
    skillpoints: u32,
    crownpoints: u32,
    ultrapoints: u32,
    patiencepick_continuation: bool,
) -> LevelEntry {
    let can_skill = !patiencepick_continuation;
    if (skillpoints > 0 && can_skill) || crownpoints > 0 || ultrapoints > 0 {
        LevelEntry::LevCont
    } else {
        LevelEntry::GenCont
    }
}

/// GML recontinue cap verbatim (`Vlambeer/Create_0:21-26`): past 2
/// recontinuations the save is deleted instead of loaded.
pub fn recontinue_deletes_save(recontinued_times: u32) -> bool {
    recontinued_times > 2
}

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
///
/// GML room-restart parity: entering `MainMenu` from anywhere rebuilds
/// the logo room (the blanket teardown + campfire `Run` reset live in
/// `setup_title_campfire`; calling it here as well as in the action arms
/// keeps direct `goto_state(MainMenu)` callers — tests, splash timeout —
/// on the same clean-room law, and it is idempotent). Entering `Title`
/// rebuilds the campfire room the same way. Both are skipped when the
/// world already reads as a fresh campfire room so repeated enters stay
/// free.
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
            ensure_menu_room(world);
        }
        AppState::Loading => {
            // GML `room_restart` parity (bevy `teardown_game` on InGame
            // exit): the generating room starts empty — `GenCont` draws
            // only the spiral + GENERATING + roadmap. Without this the
            // stale Title camp (fresh runs) or dead run (RETRY) renders
            // through the whole 1.2 s load. `setup_run` re-teardowns at
            // the end of the load; both are idempotent. Run/MenuState
            // resources survive (setup_run resets them) — only session
            // entities + the floor mask go.
            crate::setup::teardown_session_entities(world);
            world.init_resource::<crate::comps_a::FloorMask>();
            *world.resource_mut::<crate::comps_a::FloorMask>() =
                crate::comps_a::FloorMask::default();
            world.insert_resource(LoadingState::default());
        }
        AppState::Title => {
            let gml = world
                .get_resource::<crate::comps_a::SelectedCharacter>()
                .map(|s| s.0 as usize)
                .unwrap_or(crate::data::RaceId::Random as usize);
            let cursor =
                menus::roster_position(world.get_resource::<crate::savedata_part::SaveData>(), gml)
                    .unwrap_or(0);
            world.init_resource::<menus::MenuState>();
            if let Some(mut menu) = world.get_resource_mut::<menus::MenuState>() {
                menu.title_go_visible = false;
                menu.loadout_open = false;
                menu.title_cursor = cursor;
                menu.mutation_selected = None;
                // GML `Menu/Create_0:63-65` verbatim fresh-menu anim state.
                menu.portrait_offsets = [0.0; 4];
                menu.textappear = [2.0; 4];
                menu.splatindex = 0.0;
                menu.loadout_frame = 0.0;
                menu.unlock_hint.clear();
                menu.unlock_hint_pop = 0.0;
                menu.unlock_hint_t = 0.0;
                menu.weekly_run_menu = false;
            }
            crate::setup::setup_title_campfire(world);
        }
        AppState::Splash => {
            world.insert_resource(SplashState::default());
        }
        _ => {}
    }
}

/// GML logo-room law shared by the `MainMenu` entry above: rebuild the
/// empty logo room unless the world already reads as one (no session
/// instances + campfire `Run` + empty mask). Keeps repeated `MainMenu`
/// enters free while guaranteeing no dead-run world ever sits under the
/// PLAY rows.
fn ensure_menu_room(world: &mut World) {
    let stale_instances = world
        .query::<Entity>()
        .iter(world)
        .filter(|e| {
            world
                .get_entity(*e)
                .is_ok_and(|r| r.get::<bevy_ecs::resource::IsResource>().is_none())
        })
        .next()
        .is_some();
    let campfire_run = world
        .get_resource::<crate::comps_a::Run>()
        .is_some_and(|r| r.area == crate::data::AreaId::Campfire && !r.game_over);
    let empty_mask = world
        .get_resource::<crate::comps_a::FloorMask>()
        .is_none_or(|m| m.cells.is_empty());
    if stale_instances || !campfire_run || !empty_mask {
        crate::setup::setup_logo_room(world);
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
/// `block_input` (transition animation, shell-owned), live runs
/// (game-over swallows Escape, bevy parity), and non-generating rooms:
/// GML `UberCont/Step_1` only honors `want_pause` when no `GenCont`
/// exists, so Escape during a floor transition or mutation/ultra offer
/// is swallowed.
pub fn tick_escape_pause(
    paused: &mut Paused,
    overlay: &mut OverlayMenu,
    pending: &mut PendingUnpause,
    menu: &mut menus::MenuState,
    game_over: bool,
    block_input: bool,
    escape_pressed: bool,
    generating: bool,
) {
    if !escape_pressed || block_input || game_over || generating {
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
            // Esc on the pause menu resumes through the same delayed
            // path as the Resume button: overlay clears now, `paused`
            // follows when the 0.2 s timer drains.
            // paused.0 = false;
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
