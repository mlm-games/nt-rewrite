//! Headless menu/state-machine layer. State-only port of
//! `nt-recreated-bevy/src/app.rs` (`process_ui_actions`,
//! `handle_pause_input`, `handle_mutation_keys`, `handle_death_restart`)
//! and `src/menus/` (character select, loadout select, mutation choice,
//! pause, settings, unlock popups, game-over data). No rendering: repose
//! views come later and read these resources.
//!
//! [`apply_menu_action`] ports every `UiAction` arm's state law verbatim
//! (reusing the canonical [`crate::audio::UiAction`] + [`crate::audio::UiBridgeAction`]
//! queue and the [`crate::audio::ui_action_to_cue`] mapping for audio).
//! [`tick_menus`] is the headless `Update` driver: it drains the action
//! queue and routes [`NtInput`] pulses + [`MenuEdge`] shell edges per
//! app state. Register it as an exclusive system before
//! `handle_mutation_choice` so picks written here resolve the same tick.
//!
//! Input map (headless choices, documented because bevy was mouse/key
//! driven and several bevy keys have no headless counterpart):
//! - Title: `cycle_weapon` moves the character cursor (wraps 17 pods),
//!   `weapon_slot` jumps to a pod, `interact` confirms (re-clicking the
//!   selected race starts loading, bevy parity), `spec` toggles the
//!   loadout panel shut when open.
//! - Mutation: `weapon_slot` routes through the bevy two-step
//!   (`SelectMutation` highlight then `PickMutation` commit, same as
//!   bevy `handle_mutation_keys`), `cycle_weapon` moves the highlight,
//!   `interact` commits the highlight. Raw Digit1-4 shell edges should
//!   keep using `sample_mutation_digits` (direct `MutationChoice`
//!   protocol, `input.rs`); both feed `handle_mutation_choice`
//!   identically.
//! - Pause: `interact` resumes, `spec` closes the top overlay,
//!   `MenuEdge::pause_pressed` (Escape) toggles with bevy's confirm/
//!   settings-stack laws.
//! - Game over: `MenuEdge::restart_pressed` (KeyR) restarts via Loading,
//!   `interact` quits to the menu.
//! - Splash: any of fire/interact/spec advances (bevy: any key/mouse).
//! - MainMenu: `interact` plays (bevy PLAY item).
//!
//! Fidelity compromises (need shell/window services):
//! - `KeyCode` has no Escape/KeyR, so those arrive as [`MenuEdge`]
//!   (shell sets, `tick_menus` clears). Everything else uses `NtInput`
//!   pulses with take-once semantics.
//! - Mouse hover (`title_hover_race`, `main_menu_hover`), portrait/text
//!   anim timers, and all drawing are render-phase (no state kept).
//! - `SaveManager` disk writes become `SaveDirty(true)`; the existing
//!   `flush_dirty_save*` ownership is unchanged.
//! - Locale switching applies `SaveData.settings.language` directly
//!   (no `LocaleResources` headless); `SetLanguage` writes through even
//!   for unknown codes, exactly like bevy (only the effective locale
//!   was gated there).
//! - Unlock popups: bevy `unlock_popup.rs` is a placeholder and unlocks
//!   surface as toasts in `apply_floor_reach_unlocks`; the headless
//!   queue ([`MenuState::unlock_queue`]) exists with push/dismiss laws,
//!   but no producer wires into it yet (deferred with the toast bridge).
//! - Denied picks (locked race/crown/skin) have no variant in the static
//!   `ui_action_to_cue` map, so they emit `UiBack` directly (bevy played
//!   `sndNoSelect`).

use bevy_ecs::prelude::*;

use crate::audio::{
    ReactiveAudioRequest, UiAction, UiBridgeAction, crown_select_sfx, denied_sfx, hover_sfx,
    race_select_sfx, skin_select_sfx, ui_action_sfx, ui_action_to_cue,
};
use crate::comps_a::{
    MutationChoice, PendingMutation, PendingUltra, Player, Run, SaveDirty, Score, SelectedCharacter,
};
use crate::data::{CrownKind, RaceId};
use crate::input::NtInput;
use crate::msg::Queue;
use crate::savedata_part::{SAVE_VERSION, SaveData, crown_gml_to_port, crown_port_to_gml};
use crate::state::{AppState, OverlayMenu, PendingUnpause, QuitRequested, goto_state};
use crate::time::{GTimer, TimerMode};

/// Language codes bevy offered (`LOCALES` in `app.rs` verbatim).
pub const AVAILABLE_LANGUAGES: [&str; 7] = ["en", "es", "fr", "de", "ja", "zh", "pt"];

/// Character pods in bevy `CHAR_SELECT_RACES` order (gml id =
/// discriminant: Random 0 .. Cuz 16).
pub const CHAR_SELECT_ORDER: [RaceId; 17] = [
    RaceId::Random,
    RaceId::Fish,
    RaceId::Crystal,
    RaceId::Eyes,
    RaceId::Melting,
    RaceId::Plant,
    RaceId::Venuz,
    RaceId::Steroids,
    RaceId::Robot,
    RaceId::Chicken,
    RaceId::Rebel,
    RaceId::Horror,
    RaceId::Rogue,
    RaceId::BigDog,
    RaceId::Skeleton,
    RaceId::Frog,
    RaceId::Cuz,
];

/// Bevy `race_from_gml_id` verbatim over the headless roster.
pub fn race_from_gml_id(id: usize) -> Option<RaceId> {
    CHAR_SELECT_ORDER
        .iter()
        .copied()
        .find(|r| *r as usize == id)
}

/// Crown port id (`CrownKind` discriminant).
pub fn crown_to_u8(crown: CrownKind) -> u8 {
    crown as u8
}

/// Bevy `CrownKind::cycle` verbatim: one step along `ALL` (any nonzero
/// `dir` moves a single step; positive forward, negative back).
pub fn crown_cycle(id: u8, dir: i8) -> u8 {
    const ALL: [u8; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let current = ALL.iter().position(|&c| c == id).unwrap_or(0);
    let next = if dir >= 0 {
        (current + 1) % ALL.len()
    } else {
        (current + ALL.len() - 1) % ALL.len()
    };
    ALL[next]
}

/// Bevy `CrownKind::short_name` verbatim over port ids.
pub fn crown_short_name(id: u8) -> &'static str {
    match CrownKind::from_u8(id) {
        CrownKind::None => "NONE",
        CrownKind::Death => "DEATH",
        CrownKind::Life => "LIFE",
        CrownKind::Haste => "HASTE",
        CrownKind::Guns => "GUNS",
        CrownKind::Hatred => "HATRED",
        CrownKind::Blood => "BLOOD",
        CrownKind::Destiny => "DESTINY",
        CrownKind::Love => "LOVE",
        CrownKind::Risk => "RISK",
        CrownKind::Curses => "CURSES",
        CrownKind::Luck => "LUCK",
        CrownKind::Protection => "PROTECTION",
    }
}

/// Unlock notification (toast-bridge producer lands later; queue +
/// dismiss laws live here).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnlockPopup {
    Race(RaceId),
    Skin(RaceId, u8),
}

/// Game-over screen data (bevy `game_over_panel` reads verbatim:
/// area, loop, kills, score, best, mutation count; toast stays in the
/// `Toast` resource for the shell).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct GameOverScreen {
    pub world: u32,
    pub floor_in_world: u32,
    pub loop_count: u32,
    pub total_kills: u32,
    pub score: u32,
    pub high_score: u32,
    pub best_floor: u32,
    pub mutation_count: usize,
}

/// Snapshot the game-over screen (bevy `hud.rs` death-mutation law:
/// count comes from the dead player's held mutations).
pub fn capture_game_over(world: &mut World) -> Option<GameOverScreen> {
    let run = world.get_resource::<Run>()?;
    if !run.game_over {
        return None;
    }
    let screen = GameOverScreen {
        world: run.world,
        floor_in_world: crate::worldgen::floor_in_world(run.floor),
        loop_count: run.loop_count,
        total_kills: run.total_kills,
        score: world.get_resource::<Score>().map(|s| s.0).unwrap_or(0),
        high_score: world
            .get_resource::<SaveData>()
            .map(|s| s.high_score)
            .unwrap_or(0),
        best_floor: world
            .get_resource::<SaveData>()
            .map(|s| s.best_floor)
            .unwrap_or(0),
        mutation_count: world
            .query::<&Player>()
            .iter(world)
            .next()
            .map(|p| p.mutations.len())
            .unwrap_or(0),
    };
    Some(screen)
}

/// Canonical menu UI state (headless half of bevy `SharedUi`: every
/// field the views will need, none of the pixels).
#[derive(Debug, Clone, Resource)]
pub struct MenuState {
    /// Cursor into [`CHAR_SELECT_ORDER`] (keyboard/headless stand-in
    /// for bevy mouse hover + click pods).
    pub title_cursor: usize,
    /// GO button armed (bevy `title_go_visible`).
    pub title_go_visible: bool,
    /// Loadout panel open (bevy `loadout_open`).
    pub loadout_open: bool,
    /// Highlighted mutation card (bevy `mutation_selected`).
    pub mutation_selected: Option<usize>,
    /// Live mutation offer mirror (bevy `mutation_choices` half:
    /// ultra-first precedence, length resets the highlight).
    pub mutation_count: usize,
    pub mutation_is_ultra: bool,
    /// Pause quit/restart confirm (bevy `pause_confirm`: 0 quit, 1 restart).
    pub pause_confirm: Option<u8>,
    /// Hardmode armed for the next run (GML PlayButton image 3; needs
    /// the loop-2 unlock).
    pub hardmode_selected: bool,
    /// Settings page + drill stack (bevy `settings_page[_stack]`).
    pub settings_page: u8,
    pub settings_page_stack: Vec<u8>,
    /// Main-menu keyboard cursor over the 5 labels (GML gamepad_sel /
    /// bevy `main_menu_hover` parity; 0 PLAY .. 4 QUIT).
    pub main_menu_cursor: usize,
    /// Settings keyboard cursor over the page's actionable rows
    /// (`settings_hot_rows` order in `render.rs`; GML `pointed_item`).
    pub settings_cursor: usize,
    /// Pending unlock popups (producer deferred; see module docs).
    pub unlock_queue: Vec<UnlockPopup>,
    /// Live game-over snapshot (`None` until the run ends).
    pub game_over: Option<GameOverScreen>,
}

impl Default for MenuState {
    fn default() -> Self {
        Self {
            title_cursor: RaceId::Fish as usize,
            title_go_visible: false,
            loadout_open: false,
            mutation_selected: None,
            mutation_count: 0,
            mutation_is_ultra: false,
            pause_confirm: None,
            hardmode_selected: false,
            settings_page: 0,
            settings_page_stack: Vec::new(),
            main_menu_cursor: 0,
            settings_cursor: 0,
            unlock_queue: Vec::new(),
            game_over: None,
        }
    }
}

/// Shell-provided key edges with no headless `KeyCode` (Escape/KeyR).
/// Set by the shell each tick; `tick_menus` consumes them.
#[derive(Debug, Clone, Copy, Default, Resource)]
pub struct MenuEdge {
    pub pause_pressed: bool,
    pub restart_pressed: bool,
}

/// Push an unlock notification (producer wiring deferred).
pub fn push_unlock(menu: &mut MenuState, popup: UnlockPopup) {
    menu.unlock_queue.push(popup);
}

/// Dismiss the oldest unlock notification; `true` when one was shown.
pub fn dismiss_unlock(menu: &mut MenuState) -> bool {
    if menu.unlock_queue.is_empty() {
        return false;
    }
    menu.unlock_queue.remove(0);
    true
}

/// Bevy `handle_mutation_keys` routing verbatim: out-of-range digits
/// are dropped; an already-highlighted card commits (`PickMutation`),
/// otherwise it highlights (`SelectMutation`).
pub fn route_mutation_digit(menu: &MenuState, idx: usize) -> Option<UiAction> {
    if idx >= menu.mutation_count {
        return None;
    }
    if menu.mutation_selected == Some(idx) {
        Some(UiAction::PickMutation(idx))
    } else {
        Some(UiAction::SelectMutation(idx))
    }
}

/// Mutation offer mirror (bevy `hud.rs` sync half: ultra offers win,
/// a length change clears the highlight, a stale highlight clamps to
/// `None`). Headless keeps count + ultra flag (names render later).
pub fn tick_mutation_mirror(
    menu: &mut MenuState,
    pending: Option<&PendingMutation>,
    ultra: Option<&PendingUltra>,
) {
    let (count, is_ultra) = if let Some(ultra) = ultra {
        (ultra.choices.len(), true)
    } else if let Some(pending) = pending {
        (pending.choices.len(), false)
    } else {
        (0, false)
    };
    if count != menu.mutation_count {
        menu.mutation_selected = None;
    }
    menu.mutation_count = count;
    menu.mutation_is_ultra = is_ultra;
    if menu.mutation_selected.is_some_and(|sel| sel >= count) {
        menu.mutation_selected = None;
    }
}

/// Emit the mapped UI cue for an applied action (no-op when the static
/// map has none), plus the bevy-exact one-shot stems (`ui_action_sfx`).
fn emit_cue(world: &mut World, action: &UiAction) {
    if let Some(cue) = ui_action_to_cue(action) {
        world.init_resource::<Queue<ReactiveAudioRequest>>();
        world
            .resource_mut::<Queue<ReactiveAudioRequest>>()
            .push(ReactiveAudioRequest::new(cue));
    }
    let sfx = ui_action_sfx(action);
    if !sfx.is_empty() {
        world.init_resource::<Queue<crate::audio::AudioCue>>();
        let mut q = world.resource_mut::<Queue<crate::audio::AudioCue>>();
        for cue in sfx {
            q.push(cue);
        }
    }
}

/// Emit a denial sting (locked pick; bevy `sndNoSelect` 0.5).
/// `pub(crate)` so the shell click router (`lib.rs`) can sting
/// disabled rows (CO-OP) that carry no [`UiAction`].
pub(crate) fn emit_denied(world: &mut World) {
    world.init_resource::<Queue<crate::audio::AudioCue>>();
    world
        .resource_mut::<Queue<crate::audio::AudioCue>>()
        .push(denied_sfx());
}

/// Push a raw one-shot stem for site-context picks (character, skin,
/// crown, mutation highlight).
pub fn emit_sfx(world: &mut World, cue: crate::audio::AudioCue) {
    world.init_resource::<Queue<crate::audio::AudioCue>>();
    world
        .resource_mut::<Queue<crate::audio::AudioCue>>()
        .push(cue);
}

/// Mark the save dirty (headless stand-in for bevy `SaveManager::save`;
/// the existing `flush_dirty_save*` ownership is unchanged).
fn mark_dirty(world: &mut World) {
    world.init_resource::<SaveDirty>();
    world.resource_mut::<SaveDirty>().0 = true;
}

/// Apply one menu action: every bevy `process_ui_actions` arm's state
/// law, verbatim, minus rendering/audio-asset spawning (cues go to the
/// [`ReactiveAudioRequest`] queue) and minus disk writes (dirty flag).
pub fn apply_menu_action(world: &mut World, action: UiAction) {
    match action {
        UiAction::StartGame => {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.title_go_visible = false;
            }
            emit_cue(world, &UiAction::StartGame);
            goto_state(world, AppState::Loading);
        }
        UiAction::MainMenuPlay => {
            emit_cue(world, &UiAction::MainMenuPlay);
            goto_state(world, AppState::Title);
        }
        UiAction::OpenSettings => {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.settings_page = 0;
                menu.settings_page_stack.clear();
                menu.pause_confirm = None;
                menu.settings_cursor = 0;
            }
            world.init_resource::<OverlayMenu>();
            *world.resource_mut::<OverlayMenu>() = OverlayMenu::Settings;
            emit_cue(world, &UiAction::OpenSettings);
        }
        UiAction::ShowStats => {
            // GML `DrawStats` parity over the main menu (stats read live
            // from `SaveData` in the render layer; no page state).
            world.init_resource::<OverlayMenu>();
            *world.resource_mut::<OverlayMenu>() = OverlayMenu::Stats;
            emit_cue(world, &UiAction::ShowStats);
        }
        UiAction::OpenCredits => {
            world.init_resource::<OverlayMenu>();
            *world.resource_mut::<OverlayMenu>() = OverlayMenu::Credits;
            emit_cue(world, &UiAction::OpenCredits);
        }
        UiAction::CloseOverlay => {
            // Bevy restores the saved locale here; headless applies the
            // language immediately, so there is nothing to restore.
            world.init_resource::<crate::state::Paused>();
            let paused = world.resource::<crate::state::Paused>().0;
            world.init_resource::<OverlayMenu>();
            let mut overlay = world.resource_mut::<OverlayMenu>();
            match *overlay {
                OverlayMenu::Settings | OverlayMenu::Credits if paused => {
                    *overlay = OverlayMenu::Pause;
                    drop(overlay);
                    if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                        menu.settings_page = 0;
                        menu.settings_page_stack.clear();
                        menu.settings_cursor = 0;
                    }
                }
                OverlayMenu::Pause if paused => {
                    *overlay = OverlayMenu::None;
                    drop(overlay);
                    world.init_resource::<PendingUnpause>();
                    world.resource_mut::<PendingUnpause>().0 = Some(GTimer::from_seconds(
                        crate::state::UNPAUSE_DELAY_SECS,
                        TimerMode::Once,
                    ));
                }
                _ => {
                    *overlay = OverlayMenu::None;
                }
            }
            emit_cue(world, &UiAction::CloseOverlay);
        }
        UiAction::Resume => {
            world.init_resource::<OverlayMenu>();
            *world.resource_mut::<OverlayMenu>() = OverlayMenu::None;
            world.init_resource::<PendingUnpause>();
            world.resource_mut::<PendingUnpause>().0 = Some(GTimer::from_seconds(
                crate::state::UNPAUSE_DELAY_SECS,
                TimerMode::Once,
            ));
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.pause_confirm = None;
            }
            emit_cue(world, &UiAction::Resume);
        }
        UiAction::QuitToTitle => {
            world.init_resource::<crate::state::Paused>();
            world.resource_mut::<crate::state::Paused>().0 = false;
            emit_cue(world, &UiAction::QuitToTitle);
            goto_state(world, AppState::MainMenu);
        }
        UiAction::QuitApp => {
            world.init_resource::<QuitRequested>();
            world.resource_mut::<QuitRequested>().0 = true;
            emit_cue(world, &UiAction::QuitApp);
        }
        UiAction::SetMasterVol(v) => {
            let v = v.clamp(0.0, 1.0);
            world.init_resource::<SaveData>();
            world.resource_mut::<SaveData>().settings.master_volume = v;
            world.init_resource::<crate::audio::AudioChannels>();
            world.resource_mut::<crate::audio::AudioChannels>().master = v;
            mark_dirty(world);
        }
        UiAction::SetSfxVol(v) => {
            let v = v.clamp(0.0, 1.0);
            world.init_resource::<SaveData>();
            world.resource_mut::<SaveData>().settings.sfx_volume = v;
            world.init_resource::<crate::audio::AudioChannels>();
            world.resource_mut::<crate::audio::AudioChannels>().sfx = v;
            mark_dirty(world);
        }
        UiAction::SetMusicVol(v) => {
            let v = v.clamp(0.0, 1.0);
            world.init_resource::<SaveData>();
            world.resource_mut::<SaveData>().settings.music_volume = v;
            world.init_resource::<crate::audio::AudioChannels>();
            world.resource_mut::<crate::audio::AudioChannels>().music = v;
            mark_dirty(world);
        }
        UiAction::SetAmbienceVol(v) => {
            let v = v.clamp(0.0, 1.0);
            world.init_resource::<SaveData>();
            world.resource_mut::<SaveData>().settings.ambience_volume = v;
            mark_dirty(world);
        }
        UiAction::SaveSettings => {
            // Bevy copies the staged SharedUi into the save; headless
            // writes through immediately, so this only persists + closes.
            mark_dirty(world);
            world.init_resource::<crate::state::Paused>();
            let paused = world.resource::<crate::state::Paused>().0;
            world.init_resource::<OverlayMenu>();
            *world.resource_mut::<OverlayMenu>() = if paused {
                OverlayMenu::Pause
            } else {
                OverlayMenu::None
            };
            emit_cue(world, &UiAction::SaveSettings);
        }
        UiAction::NextLanguage => {
            world.init_resource::<SaveData>();
            let mut save = world.resource_mut::<SaveData>();
            let current = save.settings.language.clone();
            let idx = AVAILABLE_LANGUAGES
                .iter()
                .position(|l| *l == current)
                .unwrap_or(0);
            let next = (idx + 1) % AVAILABLE_LANGUAGES.len();
            save.settings.language = AVAILABLE_LANGUAGES[next].to_string();
            drop(save);
            mark_dirty(world);
            emit_cue(world, &UiAction::NextLanguage);
        }
        UiAction::SetLanguage(lang) => {
            // Bevy gates only the live locale; the save write is
            // unconditional — mirrored here.
            world.init_resource::<SaveData>();
            world.resource_mut::<SaveData>().settings.language = lang.clone();
            mark_dirty(world);
            emit_cue(world, &UiAction::SetLanguage(lang));
        }
        UiAction::SettingsCategory(cat) => {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                let cur = menu.settings_page;
                menu.settings_page_stack.push(cur);
                menu.settings_page = cat;
                menu.settings_cursor = 0;
            }
            emit_cue(world, &UiAction::SettingsCategory(cat));
        }
        UiAction::SettingsBack => {
            let should_close = {
                match world.get_resource_mut::<MenuState>() {
                    Some(mut menu) => {
                        menu.settings_cursor = 0;
                        if let Some(prev) = menu.settings_page_stack.pop() {
                            menu.settings_page = prev;
                            false
                        } else if menu.settings_page != 0 {
                            menu.settings_page = 0;
                            false
                        } else {
                            true
                        }
                    }
                    None => true,
                }
            };
            if should_close {
                world.init_resource::<crate::state::Paused>();
                let paused = world.resource::<crate::state::Paused>().0;
                world.init_resource::<OverlayMenu>();
                let paused_overlay = matches!(
                    *world.resource::<OverlayMenu>(),
                    OverlayMenu::Settings | OverlayMenu::Credits
                ) && paused;
                if paused_overlay {
                    *world.resource_mut::<OverlayMenu>() = OverlayMenu::Pause;
                } else {
                    *world.resource_mut::<OverlayMenu>() = OverlayMenu::None;
                }
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.settings_page = 0;
                    menu.settings_page_stack.clear();
                }
                emit_cue(world, &UiAction::SettingsBack);
            } else {
                emit_cue(world, &UiAction::SettingsBack);
            }
        }
        UiAction::ShowPauseConfirm(kind) => {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.pause_confirm = Some(kind);
            }
            emit_cue(world, &UiAction::ShowPauseConfirm(kind));
        }
        UiAction::CancelPauseConfirm => {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.pause_confirm = None;
            }
            emit_cue(world, &UiAction::CancelPauseConfirm);
        }
        UiAction::ConfirmPause(kind) => {
            if kind == 0 {
                // Quit to menu (bevy clears transition + NextState MainMenu).
                world.init_resource::<crate::state::Paused>();
                world.resource_mut::<crate::state::Paused>().0 = false;
                emit_cue(world, &UiAction::ConfirmPause(kind));
                goto_state(world, AppState::MainMenu);
            } else {
                // Restart via loading.
                world.init_resource::<crate::state::Paused>();
                world.resource_mut::<crate::state::Paused>().0 = false;
                emit_cue(world, &UiAction::ConfirmPause(kind));
                goto_state(world, AppState::Loading);
            }
        }
        UiAction::SelectCharacter(i) => {
            let Some(race) = race_from_gml_id(i) else {
                return;
            };
            world.init_resource::<SaveData>();
            if !world.resource::<SaveData>().race_unlocked(race) {
                emit_denied(world);
                return;
            }
            let already = world
                .get_resource::<SelectedCharacter>()
                .is_some_and(|s| s.0 as usize == i);
            if already {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.title_go_visible = false;
                }
                emit_cue(world, &UiAction::SelectCharacter(i));
                goto_state(world, AppState::Loading);
                return;
            }
            world.init_resource::<SelectedCharacter>();
            world.resource_mut::<SelectedCharacter>().0 = race;
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.title_cursor = i;
                menu.title_go_visible = true;
                // Bevy closes the loadout only for the Dog/Skeleton/Frog
                // trio (Random keeps whatever the toggle set; the render
                // layer hides the panel for `selected == 0`).
                if matches!(
                    race,
                    RaceId::BigDog | RaceId::Skeleton | RaceId::Frog
                ) {
                    menu.loadout_open = false;
                }
            }
            emit_sfx(world, race_select_sfx(race));
            emit_cue(world, &UiAction::SelectCharacter(i));
        }
        UiAction::SelectSkin(s) => {
            world.init_resource::<SelectedCharacter>();
            let race = world.resource::<SelectedCharacter>().0;
            if race == RaceId::Random {
                return;
            }
            world.init_resource::<SaveData>();
            let already = world
                .resource::<SaveData>()
                .race_loadout(race)
                .preferred_skin
                == s;
            if already {
                return;
            }
            if world.resource::<SaveData>().skin_unlocked(race, s) {
                world
                    .resource_mut::<SaveData>()
                    .race_loadout_mut(race)
                    .preferred_skin = s;
                mark_dirty(world);
                emit_sfx(world, skin_select_sfx(s));
                emit_cue(world, &UiAction::SelectSkin(s));
            } else {
                emit_denied(world);
            }
        }
        UiAction::ToggleLoadout => {
            // Bevy flips unconditionally; the render layer hides the
            // panel for Random (`selected == 0`), and trio gating lives
            // in the select-close above.
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.loadout_open = !menu.loadout_open;
            }
            emit_cue(world, &UiAction::ToggleLoadout);
        }
        UiAction::ToggleHardmode => {
            // GML PlayButton image 3: hardmode needs the loop-2 unlock.
            let unlocked = world
                .get_resource::<SaveData>()
                .is_some_and(|s| s.hardmode_unlocked);
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                if unlocked {
                    menu.hardmode_selected = !menu.hardmode_selected;
                }
            }
            if unlocked {
                emit_cue(world, &UiAction::ToggleHardmode);
            } else {
                emit_denied(world);
            }
        }
        UiAction::CycleStartWeapon(_) => {
            world.init_resource::<SelectedCharacter>();
            let race = world.resource::<SelectedCharacter>().0;
            world.init_resource::<SaveData>();
            // Bevy reads the raw row (not the sanitized `race_loadout`
            // copy, which drops orphan stored guns on read).
            let touched = {
                let mut save = world.resource_mut::<SaveData>();
                let lo = save.race_loadout_mut(race);
                if lo.stored_weapon.0 != 0 {
                    lo.start_weapon = if lo.start_weapon.0 == 0 {
                        lo.stored_weapon
                    } else {
                        crate::data::WEAPON_NONE
                    };
                    true
                } else {
                    false
                }
            };
            if touched {
                mark_dirty(world);
            }
            emit_cue(world, &action);
        }
        UiAction::CycleStoredWeapon(_) => {
            // Bevy arm is an empty body; kept as an accepted no-op.
        }
        UiAction::CycleCrown(dir) => {
            world.init_resource::<SelectedCharacter>();
            let race = world.resource::<SelectedCharacter>().0;
            world.init_resource::<SaveData>();
            let mut next = world.resource::<SaveData>().race_loadout(race).start_crown;
            for _ in 0..CrownKind::ALL.len() {
                next = crown_cycle(next, dir);
                let gml = crown_port_to_gml(next);
                if world.resource::<SaveData>().crown_unlocked(race, gml) || next == 0 {
                    break;
                }
            }
            world
                .resource_mut::<SaveData>()
                .race_loadout_mut(race)
                .start_crown = next;
            mark_dirty(world);
            emit_cue(world, &UiAction::CycleCrown(dir));
        }
        UiAction::SelectCrown(crown_id) => {
            world.init_resource::<SelectedCharacter>();
            let race = world.resource::<SelectedCharacter>().0;
            if race == RaceId::Random {
                return;
            }
            world.init_resource::<SaveData>();
            let already = world.resource::<SaveData>().race_loadout(race).start_crown
                == crown_gml_to_port(crown_id);
            if already {
                return;
            }
            if world.resource::<SaveData>().crown_unlocked(race, crown_id) {
                world
                    .resource_mut::<SaveData>()
                    .race_loadout_mut(race)
                    .start_crown = crown_gml_to_port(crown_id);
                mark_dirty(world);
                emit_sfx(world, crown_select_sfx());
                emit_cue(world, &UiAction::SelectCrown(crown_id));
            } else {
                emit_denied(world);
            }
        }
        UiAction::SelectMutation(idx) => {
            // Bevy: already-highlighted card commits (`MutationChoice`)
            // and clears; otherwise it highlights with `sndHover`.
            let commit = world
                .get_resource::<MenuState>()
                .and_then(|menu| menu.mutation_selected)
                == Some(idx);
            if commit {
                world.init_resource::<MutationChoice>();
                world.resource_mut::<MutationChoice>().0 = Some(idx);
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.mutation_selected = None;
                }
            } else {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.mutation_selected = Some(idx);
                }
                emit_sfx(world, hover_sfx());
            }
            emit_cue(world, &UiAction::SelectMutation(idx));
        }
        UiAction::PickMutation(idx) => {
            // Bevy `PickMutation`: commits only when the card is already
            // highlighted, then clears the highlight; otherwise it just
            // highlights. The commit feeds `MutationChoice`, which
            // `handle_mutation_choice` takes exactly once.
            let commit = world
                .get_resource::<MenuState>()
                .and_then(|menu| menu.mutation_selected)
                == Some(idx);
            if commit {
                world.init_resource::<MutationChoice>();
                world.resource_mut::<MutationChoice>().0 = Some(idx);
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.mutation_selected = None;
                }
            } else if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.mutation_selected = Some(idx);
                emit_sfx(world, hover_sfx());
            }
            emit_cue(world, &UiAction::PickMutation(idx));
        }
        UiAction::SettingToggle(ref key) => {
            world.init_resource::<SaveData>();
            let known = {
                let mut save = world.resource_mut::<SaveData>();
                apply_setting_toggle(&mut save, key)
            };
            if known {
                mark_dirty(world);
                emit_cue(world, &action);
            }
        }
        UiAction::SettingSlider { ref key, value } => {
            world.init_resource::<SaveData>();
            let known = {
                let mut save = world.resource_mut::<SaveData>();
                apply_setting_slider(&mut save, key, value)
            };
            if known {
                mark_dirty(world);
            }
        }
        UiAction::SettingCycle { ref key, dir } => {
            world.init_resource::<SaveData>();
            let known = {
                let mut save = world.resource_mut::<SaveData>();
                apply_setting_cycle(&mut save, key, dir)
            };
            if known {
                mark_dirty(world);
                emit_cue(world, &action);
            }
        }
        UiAction::SettingInput { ref key, ref value } => {
            world.init_resource::<SaveData>();
            let known = {
                let mut save = world.resource_mut::<SaveData>();
                apply_setting_input(&mut save, key, value)
            };
            if known {
                mark_dirty(world);
                emit_cue(world, &action);
            }
        }
        UiAction::SettingResetOptions => {
            // Bevy resets the whole save (progress included) — mirrored.
            world.init_resource::<SaveData>();
            *world.resource_mut::<SaveData>() = SaveData::default();
            debug_assert_eq!(world.resource::<SaveData>().version, SAVE_VERSION);
            mark_dirty(world);
            emit_cue(world, &UiAction::SettingResetOptions);
        }
        UiAction::SettingEraseProgress => {
            world.init_resource::<SaveData>();
            {
                let mut save = world.resource_mut::<SaveData>();
                save.high_score = 0;
                save.best_floor = 0;
                save.total_runs = 0;
                save.total_kills = 0;
                save.unlocked_characters = vec!["Fish".to_string()];
                save.races.clear();
                save.crown_got.clear();
            }
            mark_dirty(world);
            emit_cue(world, &UiAction::SettingEraseProgress);
        }
        UiAction::SettingViewCredits => {
            world.init_resource::<OverlayMenu>();
            *world.resource_mut::<OverlayMenu>() = OverlayMenu::Credits;
            emit_cue(world, &UiAction::SettingViewCredits);
        }
        UiAction::SettingOpenSubcategory(cat) => {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                let cur = menu.settings_page;
                menu.settings_page_stack.push(cur);
                menu.settings_page = cat;
                menu.settings_cursor = 0;
            }
            emit_cue(world, &UiAction::SettingOpenSubcategory(cat));
        }
    }
}

/// Named-boolean toggle law (bevy `SettingToggle` arm verbatim;
/// `false` = unknown key, ignored like bevy's warn-and-continue).
fn apply_setting_toggle(save: &mut SaveData, key: &str) -> bool {
    let s = &mut save.settings;
    match key {
        "volume_3dsound" => s.volume_3dsound = !s.volume_3dsound,
        "bloom" => s.bloom = !s.bloom,
        "particles" => s.particles = !s.particles,
        "show_hud" => s.show_hud = !s.show_hud,
        "show_timer" => s.show_timer = !s.show_timer,
        "show_area" => s.show_area = !s.show_area,
        "boss_intros" => s.boss_intros = !s.boss_intros,
        "auto_pause" => s.auto_pause = !s.auto_pause,
        "pause_button" => s.pause_button = !s.pause_button,
        "achievements_popup" => s.achievements_popup = !s.achievements_popup,
        "vsync" => s.vsync = !s.vsync,
        "fullscreen" => s.fullscreen = !s.fullscreen,
        "widescreen" => s.widescreen = !s.widescreen,
        "gamepad_enabled" => s.gamepad_enabled = !s.gamepad_enabled,
        "aim_assist" => s.aim_assist = !s.aim_assist,
        "auto_aim" => s.auto_aim = !s.auto_aim,
        "volume_controls" => s.volume_controls = !s.volume_controls,
        "split_fire" => s.split_fire = !s.split_fire,
        "fixed_sight" => s.fixed_sight = !s.fixed_sight,
        "show_tutorial" => s.show_tutorial = !s.show_tutorial,
        key if key.starts_with("cprefs_") => {
            let idx: usize = key
                .strip_prefix("cprefs_")
                .and_then(|x| x.parse().ok())
                .unwrap_or(99);
            return apply_cprefs_toggle(save, idx);
        }
        _ => return false,
    }
    true
}

/// `cprefs_N` toggle law (bevy index-match verbatim).
fn apply_cprefs_toggle(save: &mut SaveData, idx: usize) -> bool {
    let s = &mut save.settings;
    match idx {
        0 => s.cprefs_eyes = !s.cprefs_eyes,
        1 => s.cprefs_melting = !s.cprefs_melting,
        2 => s.cprefs_plant = !s.cprefs_plant,
        3 => s.cprefs_yv = !s.cprefs_yv,
        4 => s.cprefs_steroids = !s.cprefs_steroids,
        5 => s.cprefs_horror = !s.cprefs_horror,
        6 => s.cprefs_rogue = !s.cprefs_rogue,
        7 => s.cprefs_skeleton = !s.cprefs_skeleton,
        _ => return false,
    }
    true
}

/// Slider law (bevy `SettingSlider` arm verbatim: 0..2 clamp, with the
/// `controls_scale` 0..1 sub-clamp).
fn apply_setting_slider(save: &mut SaveData, key: &str, value: f32) -> bool {
    let v = value.clamp(0.0, 2.0);
    match key {
        "screenshake" => save.settings.screenshake = v,
        "freezeframes" => save.settings.freezeframes = v,
        "controls_scale" => save.settings.controls_scale = v.clamp(0.0, 1.0),
        _ => return false,
    }
    true
}

/// Cycle law (bevy `SettingCycle` arm verbatim: mod-4 wraps, 1-based
/// `pixel_mode`).
fn apply_setting_cycle(save: &mut SaveData, key: &str, dir: i8) -> bool {
    let s = &mut save.settings;
    match key {
        "crosshair" => s.crosshair = (s.crosshair as i16 + dir as i16).rem_euclid(4) as u8,
        "sideart" => s.sideart = (s.sideart as i16 + dir as i16).rem_euclid(4) as u8,
        "pixel_mode" => {
            s.pixel_mode = ((s.pixel_mode as i16 - 1 + dir as i16).rem_euclid(4) + 1) as u8;
        }
        "gamepad_type" => {
            s.gamepad_type = (s.gamepad_type as i16 + dir as i16).rem_euclid(4) as u8;
        }
        _ => return false,
    }
    true
}

/// Text-input law (bevy `SettingInput` arm verbatim).
fn apply_setting_input(save: &mut SaveData, key: &str, value: &str) -> bool {
    match key {
        "player_color_hex" => save.settings.player_color_hex = value.to_string(),
        "profile_name" => save.settings.profile_name = value.to_string(),
        _ => return false,
    }
    true
}

/// Headless `Update` driver: drains the [`UiBridgeAction`] queue, then
/// routes pulses/edges per app state (see module docs for the map).
/// Designed as an exclusive system; run before `handle_mutation_choice`.
pub fn tick_menus(world: &mut World) {
    world.init_resource::<MenuState>();
    world.init_resource::<MenuEdge>();
    world.init_resource::<OverlayMenu>();
    world.init_resource::<crate::state::Paused>();
    world.init_resource::<PendingUnpause>();
    world.init_resource::<AppState>();
    world.init_resource::<NtInput>();
    world.init_resource::<Queue<UiBridgeAction>>();
    world.init_resource::<Queue<ReactiveAudioRequest>>();

    let dt = world
        .get_resource::<repame_sim::SimTime>()
        .map(|t| t.delta_secs)
        .unwrap_or(1.0 / 30.0);
    let edge = std::mem::take(&mut *world.resource_mut::<MenuEdge>());

    for bridged in world.resource_mut::<Queue<UiBridgeAction>>().drain() {
        apply_menu_action(world, bridged.0);
    }

    match world.resource::<AppState>() {
        AppState::Splash => {
            let pressed = {
                world.init_resource::<NtInput>();
                let mut input = world.resource_mut::<NtInput>();
                input.take_fire_pressed()
                    || input.take_interact_pressed()
                    || input.take_spec_pressed()
                    || input.take_ability_pressed()
            };
            crate::state::tick_splash(world, dt, pressed);
        }
        AppState::Loading => {
            crate::state::tick_loading(world, dt);
        }
        AppState::MainMenu => {
            tick_main_menu_input(world, edge);
        }
        AppState::Title => {
            tick_title_input(world);
        }
        AppState::InGame => {
            tick_ingame_menu(world, edge);
        }
    }
}

/// Main-menu input routing (GML `MainMenuButton` parity): Up/Down move
/// the cursor over the 5 labels, Enter/interact activates, Digit1-5
/// jump straight to a row, Escape closes an open overlay
/// (Settings/Stats/Credits). Disabled rows (CO-OP) sting `sndNoSelect`
/// like GML's early-`exit` on `!available`.
fn tick_main_menu_input(world: &mut World, edge: MenuEdge) {
    world.init_resource::<MenuState>();
    // Escape over an open overlay closes it (the InGame escape tick
    // never runs here).
    if edge.pause_pressed
        && world
            .get_resource::<OverlayMenu>()
            .is_some_and(|o| *o != OverlayMenu::None)
    {
        apply_menu_action(world, UiAction::CloseOverlay);
        return;
    }
    let (nav_v, nav_h, slot, confirm) = {
        let mut input = world.resource_mut::<NtInput>();
        let (dv, dh) = input.take_menu_nav();
        (
            dv,
            dh,
            input.take_weapon_slot(),
            input.take_interact_pressed(),
        )
    };
    // Settings opened from the main menu owns the keyboard: Up/Down move
    // the settings cursor, Left/Right step values, Enter activates (same
    // as the InGame settings arm below).
    if world
        .get_resource::<OverlayMenu>()
        .is_some_and(|o| *o == OverlayMenu::Settings)
    {
        tick_settings_nav(world, nav_v, nav_h, confirm);
        return;
    }
    if nav_v != 0 {
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            menu.main_menu_cursor =
                (menu.main_menu_cursor as i16 + nav_v as i16).rem_euclid(5) as usize;
        }
        emit_sfx(world, hover_sfx());
    }
    // Digits jump: slot 0..4 maps straight onto rows 0..4 (Digit5 feeds
    // slot 4 for QUIT).
    if let Some(slot) = slot {
        if slot < 5 {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.main_menu_cursor = slot;
            }
            activate_main_menu_row(world, slot);
            return;
        }
    }
    if confirm {
        let cursor = world
            .get_resource::<MenuState>()
            .map(|menu| menu.main_menu_cursor)
            .unwrap_or(0);
        activate_main_menu_row(world, cursor);
    }
}

/// Activate one main-menu row (cursor or digit path shared).
fn activate_main_menu_row(world: &mut World, row: usize) {
    match row {
        0 => apply_menu_action(world, UiAction::MainMenuPlay),
        1 => emit_denied(world),
        2 => apply_menu_action(world, UiAction::OpenSettings),
        3 => apply_menu_action(world, UiAction::ShowStats),
        4 => apply_menu_action(world, UiAction::QuitApp),
        _ => {}
    }
}

/// Title input routing (cursor nav + confirm + loadout back).
fn tick_title_input(world: &mut World) {
    let (cycle, slot, confirm, back, fire) = {
        let mut input = world.resource_mut::<NtInput>();
        (
            input.take_cycle_weapon(),
            input.take_weapon_slot(),
            input.take_interact_pressed(),
            input.take_spec_pressed(),
            input.take_fire_pressed(),
        )
    };
    // Space (fire) toggles the loadout panel: `spec` (Shift/right-click)
    // has no shell key mapping, so without this the loadout/hardmode
    // switch is unreachable from the keyboard.
    if fire {
        apply_menu_action(world, UiAction::ToggleLoadout);
    }
    if cycle != 0 {
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            let len = CHAR_SELECT_ORDER.len() as i16;
            menu.title_cursor = (menu.title_cursor as i16 + cycle as i16).rem_euclid(len) as usize;
        }
    }
    if let Some(slot) = slot {
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            menu.title_cursor = slot % CHAR_SELECT_ORDER.len();
        }
    }
    if back {
        let open = world
            .get_resource::<MenuState>()
            .is_some_and(|menu| menu.loadout_open);
        if open {
            apply_menu_action(world, UiAction::ToggleLoadout);
        } else {
            apply_menu_action(world, UiAction::ToggleHardmode);
        }
    }
    if confirm {
        let cursor = world
            .get_resource::<MenuState>()
            .map(|menu| menu.title_cursor)
            .unwrap_or(RaceId::Fish as usize);
        apply_menu_action(world, UiAction::SelectCharacter(cursor));
    }
}

/// Settings keyboard driver: cursor over the page's hot rows (Up/Down
/// with `sndHover` feedback + white highlight in the render layer),
/// values stepped with Left/Right, rows committed with Enter (steppers
/// treat Enter as +1, like the `>` button).
fn tick_settings_nav(world: &mut World, nav_v: i8, nav_h: i8, confirm: bool) {
    let page = world
        .get_resource::<MenuState>()
        .map(|m| m.settings_page)
        .unwrap_or(0);
    let n = crate::render::settings_hot_rows(page, 320.0).len();
    if n == 0 {
        return;
    }
    if nav_v != 0 {
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            let cur = menu.settings_cursor.min(n - 1);
            menu.settings_cursor = (cur as i16 + nav_v as i16).rem_euclid(n as i16) as usize;
        }
        emit_sfx(world, hover_sfx());
    }
    // Re-clamp after page jumps (cursor resets to 0 on drill, but a
    // language set keeps the page with new length).
    let cursor = world
        .get_resource::<MenuState>()
        .map(|m| m.settings_cursor.min(n - 1))
        .unwrap_or(0);
    if nav_h != 0 {
        if let Some(action) = crate::render::settings_hot_action(world, page, cursor, nav_h) {
            apply_menu_action(world, action);
        }
    } else if confirm {
        // Enter resolves with dir 0, which steppers treat as +1 (same
        // as the `>` button).
        if let Some(action) = crate::render::settings_hot_action(world, page, cursor, 0) {
            apply_menu_action(world, action);
        }
    }
}

/// In-game menu routing (game-over > pause/overlay > mutation offer).
fn tick_ingame_menu(world: &mut World, edge: MenuEdge) {
    // Offer mirror (bevy hud sync ran before input handling; same law
    // as `tick_mutation_mirror`, inlined for the borrow checker).
    {
        let (count, is_ultra) = if let Some(ultra) = world.get_resource::<PendingUltra>() {
            (ultra.choices.len(), true)
        } else if let Some(pending) = world.get_resource::<PendingMutation>() {
            (pending.choices.len(), false)
        } else {
            (0, false)
        };
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            if count != menu.mutation_count {
                menu.mutation_selected = None;
            }
            menu.mutation_count = count;
            menu.mutation_is_ultra = is_ultra;
            if menu.mutation_selected.is_some_and(|sel| sel >= count) {
                menu.mutation_selected = None;
            }
        }
    }

    let game_over = world.get_resource::<Run>().is_some_and(|run| run.game_over);

    // Escape toggles pause (bevy `handle_pause_input`; transitions never
    // block headless — no `Transition` resource exists here).
    if edge.pause_pressed && !game_over {
        world.init_resource::<crate::state::Paused>();
        world.init_resource::<OverlayMenu>();
        world.init_resource::<PendingUnpause>();
        world.init_resource::<MenuState>();
        let mut paused = world
            .remove_resource::<crate::state::Paused>()
            .unwrap_or_default();
        let mut overlay = world.remove_resource::<OverlayMenu>().unwrap_or_default();
        let mut pending = world
            .remove_resource::<PendingUnpause>()
            .unwrap_or_default();
        let mut menu = world.remove_resource::<MenuState>().unwrap_or_default();
        crate::state::tick_escape_pause(
            &mut paused,
            &mut overlay,
            &mut pending,
            &mut menu,
            false,
            false,
            true,
        );
        world.insert_resource(paused);
        world.insert_resource(overlay);
        world.insert_resource(pending);
        world.insert_resource(menu);
    }

    // Snapshot the game-over screen once per death.
    let needs_capture = game_over
        && world
            .get_resource::<MenuState>()
            .is_some_and(|menu| menu.game_over.is_none());
    if needs_capture {
        let screen = capture_game_over(world);
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            menu.game_over = screen;
        }
    }

    let (cycle, slot, confirm, nav_v, nav_h) = {
        // Take-once pulses are shared with gameplay systems that run
        // later in the schedule (`weapon_switch` takes cycle/slot,
        // `collect_pickups` takes interact, weapons/abilities take spec;
        // bevy has no menu consumer for these). Only take when a menu is
        // actually open — otherwise E / 1-4 / right-click would be
        // swallowed every tick and weapons could never be picked up.
        // `spec` is taken and dropped: ability lives in gameplay (gated),
        // and overlay Back travels via Esc / right-click instead, so a
        // Shift press never closes a menu by accident.
        let menu_open = game_over
            || world
                .get_resource::<MenuState>()
                .is_some_and(|menu| menu.mutation_count > 0)
            || *world.resource::<OverlayMenu>() != OverlayMenu::None
            || world
                .get_resource::<MenuState>()
                .is_some_and(|menu| !menu.unlock_queue.is_empty());
        if menu_open {
            let mut input = world.resource_mut::<NtInput>();
            let _ = input.take_spec_pressed();
            let (dv, dh) = input.take_menu_nav();
            (
                input.take_cycle_weapon(),
                input.take_weapon_slot(),
                input.take_interact_pressed(),
                dv,
                dh,
            )
        } else {
            (0, None, false, 0, 0)
        }
    };

    if game_over {
        // Bevy `handle_death_restart` (KeyR) + game-over click (menu).
        if edge.restart_pressed {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.title_go_visible = false;
            }
            goto_state(world, AppState::Loading);
        } else if confirm {
            apply_menu_action(world, UiAction::QuitToTitle);
        }
        return;
    }

    let overlay = world.resource::<OverlayMenu>();
    let overlay_kind = *overlay;
    // Keyboard shortcuts for the pause buttons (1-4): MENU / RETRY /
    // SETTINGS / CONTINUE. Mouse already works via `route_menu_click`;
    // without this the buttons are unreachable from the keyboard.
    // While the quit/restart confirm is open, Enter commits it and
    // digits 1-2 pick MENU/RETRY directly (Escape dismisses via the
    // escape tick); other rows are inert until the confirm resolves.
    if overlay_kind == OverlayMenu::Pause
        && let Some(slot) = slot
    {
        let confirming = world
            .get_resource::<MenuState>()
            .and_then(|m| m.pause_confirm);
        if let Some(_kind) = confirming {
            match slot {
                0 | 1 => {
                    apply_menu_action(world, UiAction::ConfirmPause(slot as u8));
                    return;
                }
                _ => {
                    apply_menu_action(world, UiAction::CancelPauseConfirm);
                    return;
                }
            }
        } else {
            match slot {
                0 => {
                    apply_menu_action(world, UiAction::ShowPauseConfirm(0));
                    return;
                }
                1 => {
                    apply_menu_action(world, UiAction::ShowPauseConfirm(1));
                    return;
                }
                2 => {
                    apply_menu_action(world, UiAction::OpenSettings);
                    return;
                }
                3 => {
                    apply_menu_action(world, UiAction::Resume);
                    return;
                }
                _ => {}
            }
        }
    }
    match overlay_kind {
        OverlayMenu::Pause if confirm => {
            if let Some(kind) = world
                .get_resource::<MenuState>()
                .and_then(|m| m.pause_confirm)
            {
                apply_menu_action(world, UiAction::ConfirmPause(kind));
            } else {
                apply_menu_action(world, UiAction::Resume);
            }
            return;
        }
        OverlayMenu::None => {
            let has_unlocks = world
                .get_resource::<MenuState>()
                .is_some_and(|menu| !menu.unlock_queue.is_empty());
            if confirm && has_unlocks {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    dismiss_unlock(&mut menu);
                }
                return;
            }
        }
        // Settings keyboard nav (GML `pointed_item`/`kh` parity):
        // Up/Down move the cursor, Left/Right step values, Enter
        // activates. Overlay Back travels via Esc / right-click, never
        // Shift (which now carries ability pulses).
        OverlayMenu::Settings => {
            tick_settings_nav(world, nav_v, nav_h, confirm);
            return;
        }
        // Credits/Stats have no rows: BACK/Esc/right-click only.
        _ => {
            return;
        }
    }

    let offer_open = world
        .get_resource::<MenuState>()
        .is_some_and(|menu| menu.mutation_count > 0)
        || world.get_resource::<PendingMutation>().is_some()
        || world.get_resource::<PendingUltra>().is_some();
    if offer_open {
        if let Some(slot) = slot {
            let routed = world
                .get_resource::<MenuState>()
                .and_then(|menu| route_mutation_digit(menu, slot));
            if let Some(action) = routed {
                apply_menu_action(world, action);
            }
        }
        if cycle != 0 {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                let count = menu.mutation_count.max(1) as i16;
                let cur = menu.mutation_selected.unwrap_or(0) as i16;
                menu.mutation_selected = Some((cur + cycle as i16).rem_euclid(count) as usize);
            }
        }
        if confirm {
            let selected = world
                .get_resource::<MenuState>()
                .and_then(|menu| menu.mutation_selected);
            if let Some(selected) = selected {
                apply_menu_action(world, UiAction::PickMutation(selected));
            }
        }
        return;
    }
    let _ = (cycle, slot);
}
