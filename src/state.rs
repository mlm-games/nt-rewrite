//! App states and run flags. GML has no state enum
//! - the boot is a room hop plus instance spawns (`MakeGame/Alarm_0.gml:7`
//!   `room_goto(romGame)` then `Vlambeer/Create_0` picking MainMenu /
//!   GenCont / LevCont), so [`AppState`] is this port's stand-in; the shell
//!   driver, not shown here, transitions them and systems gate on them.
//!
//! transition laws only, no rendering:
//! - boot order Splash -> MainMenu -> Title -> Loading -> InGame; a run
//!   save on disk goes Splash -> Loading instead
//! - pause laws: [`tick_escape_pause`], [`tick_pending_unpause`],
//!   [`reset_pause_state`], [`force_death_overlay_state`]
//! - splash ([`tick_splash`]): modes 0-3 advance on press or
//!   [`SPLASH_MODE_SECS`] timeout, mode 4 plays the logo gun reel
//! - loading ([`tick_loading`]): [`LOADING_MIN_SECS`] floor, then InGame
//!   via `setup_run`
//!
//! The logo leaves on a press only, as in GML (`Logo/Draw_0.gml:7-8` ->
//! `Logo/Mouse_53.gml:22-28`); [`SplashAutoAdvance`] opts unattended boots
//! (tests, kiosk) into leaving it ~1 s after the gun reel.
//!
//! port-only compromises (need shell/window services):
//! - no animated transition: [`goto_state`] swaps instantly and
//!   [`TransitionBlock`] is the hook a shell fade/wipe would ride. GML
//!   blocks input with a frame counter, not a flag
//!   (`UberCont/Step_0.gml:8-15`)
//! - no async asset load: `progress` is pinned to 1.0, so only the
//!   [`LOADING_MIN_SECS`] floor remains. GML's loading screen is the
//!   generation screen, which counts floors over the area goal
//!   (`GenCont/Draw_0.gml:9-11`)
//! - splash shake/sprites are render; [`tick_splash`] emits the cues (GML
//!   `Vlambeer/Create_0:139` `sndVlambeer`, `Vlambeer/Alarm_0:13`
//!   `sndRestart`, `Logo/Alarm_0:18` `sndMachinegun`)
//! - [`QuitRequested`] stands in for GML's `game_end()`
//!   (`MainMenuButton/Other_10.gml:100-104`); the shell polls it
//! - no locale resources: GML builds per-language stores at boot
//!   (`MakeGame/Create_0:6-7`, `scripts/Language/Language.gml`), the port
//!   gates on `AVAILABLE_LANGUAGES` in `menus`

use bevy_ecs::prelude::*;
use repame_sim::SimTime;

use crate::time::{GTimer, TimerMode};

/// Menu state machines (character/loadout/mutation/pause/settings/
/// unlock/game-over). Lives here as `state::menus` so `lib.rs` stays
/// untouched.
#[path = "menus.rs"]
pub mod menus;

/// Top-level app state. Boot order: Splash -> MainMenu -> Title ->
/// Loading -> InGame (a run save on disk goes Splash -> Loading
/// instead).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Resource)]
pub enum AppState {
    #[default]
    Splash,
    MainMenu,
    Loading,
    Title,
    InGame,
}

/// Pause flag. Sim systems early-out while set; the `Always` set keeps
/// ticking, standing in for the instance list GML keeps live under
/// `instance_deactivate_all` (GML `UberCont/Step_1.gml:12-20`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub struct Paused(pub bool);

/// Fixed-step frame counter (GML `UberCont/Step_0.gml:106`
/// `current_frame ++` once per step at 30 steps/s).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub struct CurrentFrame(pub u64);

pub fn tick_current_frame(mut frame: ResMut<CurrentFrame>) {
    frame.0 = frame.0.wrapping_add(1);
}

/// Pause overlay selector: GML's pause rows are `PauseButton` images
/// plus whatever instance sits over them (GML
/// `scrMakePauseButtons.gml:10-33`,
/// `PauseButton/Other_10.gml:62-68` for `MenuOptions`), minus rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub enum OverlayMenu {
    #[default]
    None,
    Settings,
    Credits,
    Pause,
    /// Run-stats panel over the main menu. GML's STATS row is live:
    /// the main-menu button 3 spawns `DrawStats` (GML
    /// `MainMenuButton/Other_10.gml:90-97`, drawn by
    /// `DrawStats/Draw_0.gml:5`); the panel is created over the menu,
    /// not the pause overlay.
    Stats,
}

/// Delayed unpause: a 0.2 s `Once` timer armed by Resume/CloseOverlay/
/// Escape, which clears itself and unpauses on expiry. The delay itself is
/// port-only
/// - GML unpauses inside the same event (GML
///   `PauseButton/Other_10.gml:71-79` Continue, `scrGamePause.gml:36-53`)
///   and instead swallows the repeat with `block_input_frames` (GML
///   `UberCont/Step_0.gml:8-15`).
#[derive(Debug, Clone, Default, Resource)]
pub struct PendingUnpause(pub Option<GTimer>);

/// Delay armed before lifting pause (Resume, CloseOverlay on the pause
/// overlay, Escape out of pause) - port-only, see [`PendingUnpause`].
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
/// visuals). `complete` latches the 30-step advance (`Alarm_0`) counted
/// down by `timer`; `portal_open` latches the Fin exit (GML spawns the
/// `Portal` past `Fin`; it runs `game_restart()`, not a floor advance).
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
            self.timer = GTimer::from_seconds(1.0, TimerMode::Once);
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

/// GML `TutCont/Alarm_0.gml:9-12`: once `step_current` has run past `Fin` and
/// no `Portal` lives, the tutorial raises its own exit portal on the room
/// sentinel (`Corpse/Alarm_0:1-2` blocks the ordinary clear portal while
/// `TutCont` exists). `Portal/Alarm_1` then `game_restart()`s rather than
/// flipping the floor. Needs `&mut World` for the catalog, so it rides the
/// schedule the way [`drive_tutorial`] does.
pub fn drive_tutorial_exit(world: &mut World) {
    if !world
        .get_resource::<crate::comps_a::Run>()
        .is_some_and(|r| r.tutorial)
    {
        return;
    }
    if !world
        .get_resource::<TutorialState>()
        .is_some_and(|t| t.portal_open)
    {
        return;
    }
    let already = {
        let mut portals = world.query_filtered::<Entity, With<crate::comps_b::Portal>>();
        portals.iter(world).next().is_some()
    };
    if already {
        return;
    }

    // GML `Portal/Step_0.gml:17-20`: an idle portal drags the player in, and
    // `portal_attract` gates on `run.portal_open`.
    world.resource_mut::<crate::comps_a::Run>().portal_open = true;
    // GML `Portal/Create_0.gml:5` picks `sndOasisPortal` off
    // `GameCont.underwater`, which the desert tutorial can never be.
    world
        .resource_mut::<crate::msg::Queue<crate::audio::AudioCue>>()
        .push(crate::audio::AudioCue {
            name: "sndPortalOpen",
            volume: 1.0,
            variance: 0.0,
        });

    // GML `Portal/Create_0.gml:7-11`: every non-player projectile dies with the
    // portal that replaces the floor.
    let enemy_shots: Vec<Entity> = {
        let mut shots = world.query::<(Entity, &crate::comps_a::Team)>();
        shots
            .iter(world)
            .filter(|(_, team)| **team != crate::comps_a::Team::Player)
            .map(|(e, _)| e)
            .collect()
    };

    // GML `instance_create(10016, 10016, Portal)`.
    let at = glam::Vec2::splat(crate::worldgen::TILE as f32 * 0.5);
    world.resource_scope(|world, catalog: Mut<repame_anim::AnimCatalog>| {
        let mut commands = world.commands();
        for shot in enemy_shots {
            commands.entity(shot).despawn();
        }
        crate::progression::spawn_portal_at(&mut commands, &catalog, at, 1);
    });
    world.flush();
}

/// Tutorial advance tick (GML `TutCont/Alarm_0` verbatim): on timer expiry
/// clear the latch, step forward, re-arm 45 steps on `Fin`, raise the gun
/// when `PickingUp` comes up, and past `Fin` latch the exit portal open.
/// Clamps at `Fin` (GML `NUM - 1`).
pub fn tick_tutorial(world: &mut World, dt: f32) {
    if !world
        .get_resource::<crate::comps_a::Run>()
        .is_some_and(|r| r.tutorial)
    {
        return;
    }
    world.init_resource::<TutorialState>();
    {
        let mut tut = world.resource_mut::<TutorialState>();
        tut.timer.tick(dt);
        if !tut.complete || !tut.timer.just_finished() {
            return;
        }
        tut.complete = false;
        let prev = tut.step;
        tut.step = tut.step.next();
        if tut.step == TutorialStep::Fin && !tut.portal_open {
            // GML `Alarm_0:8` `if (step_current == Fin) alarm[0] = 45`.
            // Nothing ever calls `complete_step(Fin)`, so this timer alone
            // carries the tutorial to its exit. The `!portal_open` guard is
            // GML's `exit` at `Alarm_0:14-17`: `step_current` is clamped to
            // `NUM - 1` there and nothing re-arms `alarm[0]`, so once the
            // exit portal is up the hold must stop - `Fin.next()` is `Fin`,
            // so without this it would re-arm every 45 steps for good.
            tut.complete = true;
            tut.timer = GTimer::from_seconds(45.0 / 30.0, TimerMode::Once);
        }
        if prev == TutorialStep::Fin {
            // GML `TutCont/Alarm_0.gml:12-15`: `step_current++` takes it past
            // `Fin`, so `step_current > Fin` writes `game.tutorial = false`
            // and spawns the exit `Portal`. Until this point the "COOL, WE'RE
            // DONE HERE!" bar is still on screen, because `Draw_64` only
            // hides it once the Portal exists.
            tut.portal_open = true;
            world
                .resource_mut::<crate::savedata_part::SaveData>()
                .tutorial_done = true;
            world.resource_mut::<crate::comps_a::SaveDirty>().0 = true;
        }
    }

    // GML `TutCont/Alarm_0.gml:19-53`: the gun appears only now. Up to 256
    // random sites (the last attempt the player) are probed for one at least
    // 32 px from any `Wall` or `hitme`; the chest is then shoved
    // `move_contact_solid(random_angle, 32 + random(72))` behind a 20-dust
    // fan. The room sentinel is the fallback site.
    if world.resource::<TutorialState>().step != TutorialStep::PickingUp {
        return;
    }
    use rand::RngExt;
    let mut rng = rand::rng();
    let player: Vec<glam::Vec2> = world
        .query_filtered::<&crate::spatial::Pos, With<crate::comps_a::Player>>()
        .iter(world)
        .map(|p| p.0)
        .collect();
    let corpses: Vec<glam::Vec2> = world
        .query_filtered::<&crate::spatial::Pos, With<crate::comps_b::Corpse>>()
        .iter(world)
        .map(|p| p.0)
        .collect();
    let mut site = None;
    if let Some(mask) = world.get_resource::<crate::comps_a::FloorMask>() {
        for i in 0..256 {
            // GML `Alarm_0:22-25`: every attempt rolls a random `Floor`,
            // and only the last one (`++i >= 256`) falls back to a random
            // `Player`.
            let at = if i == 255 {
                player.first().copied()
            } else {
                Some(mask.random_floor_pos(&mut rng, 0.0))
            };
            let Some(at) = at else { continue };
            // `distance_to_object(Wall) < 32` rejects the site, so a 32 px
            // step in every direction has to stay clear.
            let clear = [
                glam::Vec2::new(32.0, 0.0),
                glam::Vec2::new(-32.0, 0.0),
                glam::Vec2::new(0.0, 32.0),
                glam::Vec2::new(0.0, -32.0),
            ]
            .into_iter()
            .all(|d| mask.is_walkable(at + d));
            if !clear || corpses.iter().any(|c| c.distance(at) < 32.0) {
                continue;
            }
            site = Some(at);
            break;
        }
    }
    let at = site.unwrap_or_else(|| glam::Vec2::splat(crate::worldgen::TILE as f32 * 0.5));
    let dir = rng.random_range(0.0..std::f32::consts::TAU);
    let shove = rng.random_range(32.0..104.0);
    let mut rest = at + glam::Vec2::from_angle(dir) * shove;
    if let Some(mask) = world.get_resource::<crate::comps_a::FloorMask>() {
        crate::spatial::move_contact_solid(
            &mut rest,
            glam::Vec2::from_angle(dir) * shove,
            12.0,
            &[],
            Some(mask),
        );
    }
    // GML `:37-42`: a 20-mote fan stepping 18 degrees off one heading.
    let motes: Vec<f32> = (0..20)
        .map(|_| (6.0 - rng.random_range(0.0..1.0)) * 30.0)
        .collect();
    let fan = dir;
    world.commands().queue(move |world: &mut World| {
        let frames = world
            .resource::<repame_anim::AnimCatalog>()
            .def("images/sprWeaponChest.png")
            .map(|def| def.frames)
            .unwrap_or(1) as u32;
        let mut commands = world.commands();
        crate::pickups::spawn_chest_frames(
            &mut commands,
            crate::comps_b::ChestKind::Weapon,
            rest,
            &crate::pickups::ChestCtx::default(),
            frames,
        );
        let mut dust = fan;
        for speed in motes {
            crate::environment::spawn_native_streak(&mut commands, false, at, dust, speed);
            dust += 18.0_f32.to_radians();
        }
    });
}

/// Boot-intro state: `Vlambeer`'s `mode` plus the per-mode alarm it
/// re-arms (GML `Vlambeer/Create_0.gml:130-131`, `Alarm_0.gml:7-11`),
/// with mode 4 standing for the `Logo` object GML spawns in place of
/// `Vlambeer` (GML `Vlambeer/Alarm_0.gml:1-4`), whose `image_index`
/// `guns` counts. Entities/sprites deferred to render, splash cues
/// emitted by [`tick_splash`].
#[derive(Debug, Clone, Resource)]
pub struct SplashState {
    pub mode: u8,
    pub t: f32,
    pub guns: u8,
    /// GML `MakeGame/Create_0:37` `loading`: a run save on disk replaces the
    /// logo room with the "CONTINUE THIS SAVED RUN?" prompt, so the splash
    /// boots straight into [`SPLASH_MODE_LOAD`].
    pub load: LoadPromptState,
}

impl Default for SplashState {
    fn default() -> Self {
        Self {
            mode: 0,
            t: 0.0,
            guns: 0,
            load: LoadPromptState::default(),
        }
    }
}

/// GML `MakeGame/Draw_0:71-176`: the mid-run save prompt. `loading` counts the
/// 15-frame black fade-in, `pos` reveals one roadmap node per frame, `posy`
/// slides the crown in, `pointed_item` is 1 for YES / 2 for NO / -1 for
/// neither and drives the hover sting.
#[derive(Debug, Clone, Copy, Default, Resource)]
pub struct LoadPromptState {
    pub loading: u32,
    pub pos: u32,
    pub posy: i32,
    pub pointed_item: i32,
}

/// Splash mode for the load prompt (`Vlambeer/Create_0`'s `else` branch is the
/// logo room; the `file_exists(savegame_file)` branch never reaches it).
pub const SPLASH_MODE_LOAD: u8 = 5;

/// GML `MakeGame/Draw_0:143` `if (loading < 15)`.
pub const LOAD_FADE_FRAMES: u32 = 15;

/// GML `MakeGame/Draw_0:86-87`: YES/NO sit `gui_w/2 - 64` and `gui_w/2 + 48`,
/// `gui_h/2 + 40 + posy` down, each inside a 16px `point_in_circle`.
pub fn load_prompt_rows(vw: f32, posy: i32) -> [(f32, f32); 2] {
    let cx = vw * 0.5;
    let options_y = 120.0 + 40.0 + posy as f32;
    [(cx - 64.0, options_y), (cx + 48.0, options_y)]
}

pub const LOAD_PROMPT_RADIUS: f32 = 16.0;

/// GML `MakeGame/Draw_0:156-168`: the prompt's commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadChoice {
    Pending,
    Continue,
    Discard,
}

/// GML `MakeGame/Draw_0:71-169` per-draw law. `mouse` is the GUI-space cursor
/// (`None` on a pad), `pad_yes`/`pad_no` are the raw `gp_face1`/`gp_face2`
/// edges. Returns the choice plus the hover sting.
pub fn tick_load_prompt(
    state: &mut LoadPromptState,
    mouse: Option<[f32; 2]>,
    pad: bool,
    waypoints: usize,
    clicked: bool,
    pad_yes: bool,
    pad_no: bool,
) -> (LoadChoice, bool) {
    let rows = load_prompt_rows(320.0, state.posy);
    let (mut point_left, mut point_right) = (false, false);
    if let Some([mx, my]) = mouse {
        for (i, (x, y)) in rows.iter().enumerate() {
            let hit = (mx - x).powi(2) + (my - y).powi(2) <= LOAD_PROMPT_RADIUS.powi(2);
            if i == 0 {
                point_left = hit;
            } else {
                point_right = hit;
            }
        }
    }
    if pad {
        point_left = true;
        point_right = true;
    }

    let previous = state.pointed_item;
    if point_left && state.pointed_item != 1 {
        state.pointed_item = 1;
    } else if point_right && state.pointed_item != 2 {
        state.pointed_item = 2;
    } else if !point_left && !point_right {
        state.pointed_item = -1;
    }
    let sting = previous != state.pointed_item && state.pointed_item != -1;

    if (state.pos as usize) >= waypoints {
        state.posy = (state.posy - 8).max(0);
    } else {
        state.pos += 1;
    }
    if state.loading < LOAD_FADE_FRAMES {
        state.loading += 1;
    }

    if !clicked && !pad_yes && !pad_no {
        return (LoadChoice::Pending, sting);
    }
    if point_left || pad_yes {
        (LoadChoice::Continue, sting)
    } else if point_right || pad_no {
        (LoadChoice::Discard, sting)
    } else {
        (LoadChoice::Pending, sting)
    }
}

/// Per-mode auto-advance timeouts, verbatim: GML
/// `Vlambeer/Create_0.gml:130` arms `alarm[0] = 120` for mode 0 and
/// `Alarm_0.gml:8-11` re-arms 60 steps per mode with `+60` on mode 2,
/// which at 30 steps/s (`options/main/options_main.yy:18`) is
/// [4, 2, 4, 2] s.
pub const SPLASH_MODE_SECS: [f32; 4] = [4.0, 2.0, 4.0, 2.0];

/// Logo gunfire step times (mode 4), verbatim: GML `Logo/Create_0.gml:2`
/// arms `alarm[0] = 30` for the first frame, then `Alarm_0.gml:17` re-arms
/// 2 steps between frames except 20 after frame 6, and `:7-12` fires the
/// finale on frame 7.
pub const SPLASH_GUN_STEPS: [f32; 7] = [
    1.0,
    1.0 + 2.0 / 30.0,
    1.0 + 4.0 / 30.0,
    1.0 + 6.0 / 30.0,
    1.0 + 8.0 / 30.0,
    1.0 + 10.0 / 30.0,
    1.0 + 10.0 / 30.0 + 20.0 / 30.0,
];

/// Headless hold after the gun sequence before auto-advancing (port-only
/// - GML leaves the logo on a press only, `Logo/Draw_0.gml:7-8` ->
///   `Logo/Mouse_53.gml:22-28`; applies only with [`SplashAutoAdvance`]
///   set, see [`tick_splash`]).
pub const SPLASH_LOGO_HOLD_SECS: f32 = 1.0;

/// Opt-in unattended splash advance (tests, kiosk shells). Absent or
/// `false` (the default): mode 4 is press-only, as in GML. `true`:
/// mode 4 also leaves ~[`SPLASH_LOGO_HOLD_SECS`] after the gun reel, so
/// boots with no input source still reach the menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub struct SplashAutoAdvance(pub bool);

/// GML `ButtonAct` fade, verbatim (`Create_0` `alpha = 3`, `active = 0`;
/// `Other_10` per step: `active` → `alpha = 1.1, active = 0`, else
/// `alpha -= 0.1`; `scrDrawPlayerHUD` raises `active` while the player
/// stands on a pickup). `alpha > 0` is the WHOLE visibility law for the
/// button and its pickup art (`scrDrawMobileControls`: `alpha = 1` with
/// no `_player`, `else if (alpha <= 0) continue`), so it lights only
/// near a pickup, plus a ~3 s grace at run start.
#[derive(Debug, Clone, Copy, Resource)]
pub struct ActButton {
    pub alpha: f32,
    pub active: bool,
}

impl Default for ActButton {
    fn default() -> Self {
        Self {
            alpha: 3.0,
            active: false,
        }
    }
}

impl ActButton {
    /// `Other_10` step, then the draw's raise. GML runs the step before
    /// `scrDrawPlayerHUD` sets `active`, so the fade lags the scan by one
    /// frame; the order here reproduces that.
    pub fn step(&mut self) {
        if self.active {
            self.alpha = 1.1;
            self.active = false;
        } else if self.alpha > 0.0 {
            self.alpha = (self.alpha - 0.1).max(0.0);
        }
    }
}

/// Loading-screen state: the clock plus the tip GML picks once per floor
/// (`GenCont/Create_0.gml:16` `scrTips()`). `progress` is port-only - the
/// shell has no async asset load, and GML's generation screen reports
/// floors built over the area goal instead (`GenCont/Draw_0.gml:9-11`).
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

/// Minimum loading-screen time, port-only: GML has no such floor. Its
/// generation screen ends when `FloorMaker` is gone
/// (`GenCont/Step_0.gml:3`, `:14-15` arming `alarm[0] = 3` /
/// `alarm[2] = 2`) or at the `alarm[5] = 600` safety break
/// (`GenCont/Create_0.gml:41`, `GenCont/Alarm_5.gml:2`).
pub const LOADING_MIN_SECS: f32 = 1.2;

/// GML `MakeGame` boot flags verbatim (`Create_0` + `Alarm_0` + `Vlambeer`
/// recontinue arm): disclaimer gate, save-continue roadmap, recontinue cap.
/// Headless defaults keep the current boot (disclaimer accepted, no save
/// on disk) so shells/tests reach the menu; shells with disk set the
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

/// GML `Vlambeer/Create_0:10-40`, the `file_exists(savegame_file)` boot branch:
/// load, drop the save once the recontinue cap trips, bump the counter and
/// write it straight back (`scrSavegameSave` on line 39) so the next crash
/// resumes from the bumped count. Fed by the intro prompt's parsed save, which
/// the prompt parks in [`crate::run_save::PendingRunSave`].
pub fn begin_continued_run(world: &mut World) -> Result<(), String> {
    let save = world
        .remove_resource::<crate::run_save::PendingRunSave>()
        .map(|pending| pending.0)
        .ok_or_else(|| "no pending run save".to_string())?;
    crate::setup::setup_continued_run(world, &save);
    world.init_resource::<BootFlags>();
    {
        let mut flags = world.resource_mut::<BootFlags>();
        if recontinue_deletes_save(flags.recontinued_times) {
            crate::run_save::delete_run_save();
        }
        flags.recontinued_times += 1;
        flags.continued_run = true;
    }
    let _ = crate::run_save::save_run(world);
    Ok(())
}

/// GML `MakeGame/Draw_0:165-168`: the prompt's NO path - the save is dropped
/// and the boot restarts, so the run never comes back.
pub fn discard_saved_run(world: &mut World) {
    crate::run_save::delete_run_save();
    world.init_resource::<BootFlags>();
    let mut flags = world.resource_mut::<BootFlags>();
    flags.has_save_file = false;
    flags.continued_run = false;
    flags.recontinued_times = 0;
}

/// Headless quit signal. GML's main-menu QUIT restarts the room and calls
/// `game_end()` (`MainMenuButton/Other_10.gml:100-104`); with no engine
/// window service headless the port raises a flag and the shell polls it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub struct QuitRequested(pub bool);

/// Scene-transition input block, port-only. The port flips states
/// instantly (`goto_state`), so no system ever raises this - it exists so
/// `gameplay_active` keeps the gate shape instead of silently dropping a
/// conjunct, and so a shell-driven fade can hold input. GML blocks input
/// with a frame countdown instead (GML `UberCont/Step_0.gml:8-15`,
/// armed at `PauseButton/Other_10.gml:76`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub struct TransitionBlock(pub bool);

/// Instant state transition with pause teardown side effects (paused/
/// overlay/pending cleared; `Run::game_over` cleared when leaving InGame or
/// entering a menu state; menu transients cleared). GML has no animated
/// state transition to defer
/// - a state swap is a `room_goto` / `room_restart`
///   (`MakeGame/Alarm_0.gml:7`, `scrRunStart.gml:75`); see the module docs
///   for the shell hook.
///
/// GML room-restart parity: entering `MainMenu` from anywhere rebuilds the
/// logo room and `Title` the campfire room (blanket teardown + campfire
/// `Run` reset live in `setup_title_campfire`; calling it here as well as
/// in the action arms keeps direct `goto_state` callers
/// - tests, splash timeout
/// - on the same clean-room law, and it is idempotent). Both skip when the
///   world already reads as a fresh campfire room.
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
                menu.settings_cursor = usize::MAX;
                menu.play_submenu = false;
                menu.play_cursor = 0;
            }
            ensure_menu_room(world);
        }
        AppState::Loading => {
            // GML `room_restart` parity on an InGame exit
            // (`scrGameRestart.gml:24` `scrCleanupSessionInstances()`): the
            // generating room starts empty
            // - `GenCont` draws only spiral + GENERATING + roadmap;
            //   otherwise the stale Title camp (fresh runs) or dead run
            //   (RETRY) renders through the whole 1.2 s load. `setup_run`
            //   re-teardowns at load end; both idempotent. Run/MenuState
            //   survive (setup_run resets them)
            // - only session entities + the floor mask go.
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
            world.init_resource::<crate::msg::Queue<crate::audio::AudioCue>>();
            world
                .resource_mut::<crate::msg::Queue<crate::audio::AudioCue>>()
                .push(crate::audio::AudioCue {
                    name: "sndVlambeer",
                    volume: 1.0,
                    variance: 0.0,
                });
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

/// Pause teardown on a state swap (pause/overlay/pending/menu transients
/// cleared; caller owns the `Run::game_over` edge, see `goto_state`). GML
/// clears the same flags from the save event and the unpause path
/// (`UberCont/Other_5.gml:5-6` `want_pause`/`want_restart = 0`,
/// `scrGamePause.gml:46-48` `paused = false`, `want_pause = 0`).
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
        menu.pause_cursor = 0;
        menu.pause_splat = 0.0;
        menu.pause_appear = [1.0, 2.0, 3.0, 3.0];
        menu.pause_portrait_anim = 0.0;
        menu.hover_label.clear();
        menu.settings_page = 0;
        menu.settings_page_stack.clear();
        menu.mutation_selected = None;
        menu.game_over = None;
        menu.settings_cursor = usize::MAX;
        menu.settings_splats.clear();
        menu.settings_back_hover = false;
        menu.unlock = crate::state::menus::UnlockPopupState::default();
        menu.play_submenu = false;
        menu.play_cursor = 0;
    }
}

/// Splash tick (the mode timers plus the opt-in logo-hold auto-advance
/// for unattended boots). `pressed` = any key/mouse edge this tick. Only
/// runs in `Splash`; finishing enters `MainMenu`.
///
/// GML splash cues per event: `sndVlambeer` once at reel creation
/// (`Vlambeer/Create_0:139`, the fresh-boot `else` branch; quit-to-menu
/// and continue-run branches `exit` before it), `sndRestart` on each of
/// the three `Vlambeer/Alarm_0:13` `mode++` advances (timer- or
/// press-triggered alike - `Draw_0.gml:5-6` performs the same alarm; the
/// mode 3 -> 4 arm creates the logo and plays none), `sndMachinegun` per
/// gun-step increment 1..=6 (`Logo/Alarm_0:18`), `Logo/Alarm_0:8-12`
/// finale `sndShovel` + `sndMeatExplo` + `sndExplosion` on increment 7
/// (its `sndLogoLoop` half is a looping ambience track,
/// `audio::AmbienceCue::LogoLoop`).
pub fn tick_splash(world: &mut World, dt: f32, pressed: bool) {
    if world
        .get_resource::<AppState>()
        .copied()
        .unwrap_or_default()
        != AppState::Splash
    {
        return;
    }
    let fresh = world.get_resource::<SplashState>().is_none();
    world.init_resource::<SplashState>();
    // GML `MakeGame/Draw_0:71` `if (loading)`: a run save replaces the logo
    // room entirely, so none of the mode timers below run.
    if world.resource::<SplashState>().mode == SPLASH_MODE_LOAD {
        return;
    }
    let auto = world
        .get_resource::<SplashAutoAdvance>()
        .is_some_and(|a| a.0);
    let mut cues: Vec<crate::audio::AudioCue> = Vec::new();
    let mut cue = |name: &'static str| {
        cues.push(crate::audio::AudioCue {
            name,
            volume: 1.0,
            variance: 0.0,
        })
    };
    if fresh {
        cue("sndVlambeer");
    }
    let done = {
        let mut splash = world.resource_mut::<SplashState>();
        if splash.mode < 4 {
            splash.t += dt;
            let advance = pressed || splash.t >= SPLASH_MODE_SECS[splash.mode as usize];
            if advance {
                if splash.mode < 3 {
                    cue("sndRestart");
                }
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
                if (splash.guns as usize) < SPLASH_GUN_STEPS.len() {
                    cue("sndMachinegun");
                } else {
                    cue("sndShovel");
                    cue("sndMeatExplo");
                    cue("sndExplosion");
                }
            }
            if pressed {
                if splash.guns == 0 {
                    // GML fast-forward: still on frame 0, a press clamps
                    // the pending alarm to 10 steps (GML
                    // `Logo/Mouse_53.gml:30-32`), so pull `t` to 10 steps
                    // short of the first `SPLASH_GUN_STEPS` entry.
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
    if !cues.is_empty() {
        world.init_resource::<crate::msg::Queue<crate::audio::AudioCue>>();
        let mut q = world.resource_mut::<crate::msg::Queue<crate::audio::AudioCue>>();
        for c in cues {
            q.push(c);
        }
    }
    if done {
        goto_state(world, AppState::MainMenu);
    }
}

/// Loading tick (no asset wait - `progress` stays at 1.0, port's
/// [`LOADING_MIN_SECS`] floor, then InGame through the existing
/// `setup_run` entry - never duplicated here). Only runs in `Loading`.
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
        // GML `MakeGame/Draw_0:161` `room_goto(romGame)` re-enters
        // `Vlambeer/Create_0`, which sees the save on disk and takes the
        // continue branch instead of building a fresh run.
        let resumed = world
            .get_resource::<crate::run_save::PendingRunSave>()
            .is_some();
        if resumed {
            let _ = begin_continued_run(world);
        } else {
            crate::setup::setup_run(world);
        }
        // GML `Vlambeer/Create_0:111-117`: entering the game room with a live
        // `GameCont` saves the run, except right after a continue-load (that
        // branch wrote the bumped counter already) and inside the credits or
        // a cinematic (the throne sit), which delete the save instead.
        let cinematic = world
            .get_resource::<crate::comps_a::Run>()
            .is_some_and(|r| r.won);
        if resumed {
            world.init_resource::<crate::state::BootFlags>();
            world
                .resource_mut::<crate::state::BootFlags>()
                .continued_run = false;
        } else if !cinematic {
            let _ = crate::run_save::save_run(world);
        }
        reset_pause_state(world);
        // A continued run can come back holding an owed pick (GML
        // `Vlambeer/Create_0:98-105` opens LevCont on a just-loaded room start
        // too), so the reset above must not leave that offer running unpaused.
        let holds_offer = world
            .get_resource::<crate::comps_a::PendingMutation>()
            .is_some()
            || world
                .get_resource::<crate::comps_a::PendingUltra>()
                .is_some();
        if holds_offer {
            if let Some(mut paused) = world.get_resource_mut::<Paused>() {
                paused.0 = true;
            }
        }
    }
}

/// Escape-pause tick (minus engine key reads: the shell passes
/// `escape_pressed` where GML polls `vk_escape`/`vk_backspace`/ `gp_start`
/// into a per-index `KeyCont.press_paus` latch, GML
/// `UberCont/Step_0.gml:35-65`). Gated to InGame, `block_input` (transition
/// animation, shell-owned), live runs and non-generating rooms, matching
/// GML's `scrGameCanPause` gate (`GenCont`, `Credits`, `Cinematic`, no
/// `Player`, `GameOver`, `want_pause`, `romInit`, GML
/// `scrGamePause.gml:84-87`) plus `UberCont/Step_1.gml:3-24`, which only
/// honours `want_pause` with a live `Player` and no `GenCont` or `GameOver`
/// - so Escape during a floor transition or a mutation/ultra offer, and on
///   the game-over screen, is swallowed.
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
            menu.pause_cursor = 0;
            menu.pause_splat = 0.0;
            menu.pause_appear = [1.0, 2.0, 3.0, 3.0];
            menu.pause_portrait_anim = 180.0;
            menu.hover_label.clear();
            menu.settings_page = 0;
            menu.settings_page_stack.clear();
            menu.settings_splats.clear();
        }
        OverlayMenu::Pause => {
            if menu.pause_confirm.is_some() {
                // Port-only: GML's Escape looks for the Continue button
                // (`image_index == 3`) and finds nothing while the
                // confirm rows (QUIT 5 / RETRY 6 / BACK 4,
                // GML `PauseButton/Other_10.gml:19-41`,
                // `UberCont/Step_0.gml:52-60`), so the press is swallowed
                // and only the BACK row clears the confirm.
                menu.pause_confirm = None;
                menu.pause_cursor = 0;
                menu.pause_appear = [1.0, 2.0, 3.0, 3.0];
                menu.hover_label.clear();
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
                menu.settings_cursor = usize::MAX;
            } else if paused.0 {
                *overlay = OverlayMenu::Pause;
                menu.settings_page = 0;
                menu.settings_page_stack.clear();
                menu.settings_cursor = usize::MAX;
                menu.pause_confirm = None;
                menu.pause_cursor = 0;
                menu.pause_appear = [1.0, 2.0, 3.0, 3.0];
                menu.hover_label.clear();
            } else {
                *overlay = OverlayMenu::None;
                menu.settings_cursor = usize::MAX;
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

/// Pending-unpause tick over `SimTime`: on expiry the arm clears itself
/// and the pause lifts (port-only delay, see [`PendingUnpause`]).
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

/// Death overlay guard: while InGame a dead run holds no pause flag, no
/// overlay and no pending unpause. GML keeps the same shape by building
/// the game-over rows directly into the pause overlay (GML
/// `GameOver/Create_0.gml:21-28` two `PauseButton`s, MENU and RETRY, no
/// Continue) and only spawning them unpaused (GML `TopCont/Step_2.gml:27`
/// `!UberCont.paused && !instance_exists(GameOver)`).
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

/// Channel sync: master/sfx/music follow the save settings every tick. GML
/// applies them per category from `UberCont.opt_*` in `scrVolume.gml:1-25`
/// (master gain, music stems, ambient loops), which `scrOptionsUpdate`
/// re-reads from the save file (`scrOptionsUpdate.gml:13-16`, `:89`); the
/// port tracks one triple continuously instead of re-reading per event.
pub fn sync_audio_channels(
    save: Res<crate::savedata_part::SaveData>,
    mut channels: ResMut<crate::audio::AudioChannels>,
) {
    channels.master = save.settings.master_volume;
    channels.sfx = save.settings.sfx_volume;
    channels.music = save.settings.music_volume;
}

/// Save sanitize (headless law: version mismatch sanitizes loadouts and
/// stamps; covers both the migrate and the added-save arms without engine
/// change detection). The version stamp itself is port-only
/// - GML persists a flat key/value blob (`scrSave.gml:4-5` `savepath = ...
///   .sav`, `:40-42` per-key writes) plus a versionless buffer for the run
///   save (`scripts/scrSavegame/scrSavegame.gml:9-22`).
pub fn tick_sanitize_save(mut save: ResMut<crate::savedata_part::SaveData>) {
    if save.version != crate::savedata_part::SAVE_VERSION {
        let from = save.version;
        crate::savedata_part::migrate_save_on_load(&mut save, from);
        save.version = crate::savedata_part::SAVE_VERSION;
    }
}
