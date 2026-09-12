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
    MutationChoice, PendingMutation, PendingUltra, Player, Run, SaveDirty, Score,
    SelectedCharacter,
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
    CHAR_SELECT_ORDER.iter().copied().find(|r| *r as usize == id)
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
        floor_in_world: run.floor_in_area,
        loop_count: run.loop_count,
        total_kills: run.total_kills,
        score: world.get_resource::<Score>().map(|s| s.0).unwrap_or(0),
        high_score: world.get_resource::<SaveData>().map(|s| s.high_score).unwrap_or(0),
        best_floor: world.get_resource::<SaveData>().map(|s| s.best_floor).unwrap_or(0),
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
fn emit_denied(world: &mut World) {
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
            }
            world.init_resource::<OverlayMenu>();
            *world.resource_mut::<OverlayMenu>() = OverlayMenu::Settings;
            emit_cue(world, &UiAction::OpenSettings);
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
                    }
                }
                OverlayMenu::Pause if paused => {
                    *overlay = OverlayMenu::None;
                    drop(overlay);
                    world.init_resource::<PendingUnpause>();
                    world.resource_mut::<PendingUnpause>().0 =
                        Some(GTimer::from_seconds(crate::state::UNPAUSE_DELAY_SECS, TimerMode::Once));
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
            world.resource_mut::<PendingUnpause>().0 =
                Some(GTimer::from_seconds(crate::state::UNPAUSE_DELAY_SECS, TimerMode::Once));
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
            *world.resource_mut::<OverlayMenu>() =
                if paused { OverlayMenu::Pause } else { OverlayMenu::None };
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
            }
            emit_cue(world, &UiAction::SettingsCategory(cat));
        }
        UiAction::SettingsBack => {
            let should_close = {
                match world.get_resource_mut::<MenuState>() {
                    Some(mut menu) => {
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
                let paused_overlay =
                    matches!(*world.resource::<OverlayMenu>(), OverlayMenu::Settings | OverlayMenu::Credits)
                        && paused;
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
                if matches!(race, RaceId::BigDog | RaceId::Skeleton | RaceId::Frog) {
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
            let already = world.resource::<SaveData>().race_loadout(race).preferred_skin == s;
            if already {
                return;
            }
            if world.resource::<SaveData>().skin_unlocked(race, s) {
                world.resource_mut::<SaveData>().race_loadout_mut(race).preferred_skin = s;
                mark_dirty(world);
                emit_sfx(world, skin_select_sfx(s));
                emit_cue(world, &UiAction::SelectSkin(s));
            } else {
                emit_denied(world);
            }
        }
        UiAction::ToggleLoadout => {
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
                emit_cue(world, &UiAction::CancelPauseConfirm);
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
            world.resource_mut::<SaveData>().race_loadout_mut(race).start_crown = next;
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
                world.resource_mut::<SaveData>().race_loadout_mut(race).start_crown =
                    crown_gml_to_port(crown_id);
                mark_dirty(world);
                emit_sfx(world, crown_select_sfx());
                emit_cue(world, &UiAction::SelectCrown(crown_id));
            } else {
                emit_denied(world);
            }
        }
        UiAction::SelectMutation(idx) => {
            // Bevy: first click highlights (no bounds check there either).
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.mutation_selected = Some(idx);
            }
            emit_sfx(world, hover_sfx());
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
            };
            crate::state::tick_splash(world, dt, pressed);
        }
        AppState::Loading => {
            crate::state::tick_loading(world, dt);
        }
        AppState::MainMenu => {
            let confirm = world.resource_mut::<NtInput>().take_interact_pressed();
            if confirm {
                apply_menu_action(world, UiAction::MainMenuPlay);
            }
        }
        AppState::Title => {
            tick_title_input(world);
        }
        AppState::InGame => {
            tick_ingame_menu(world, edge);
        }
    }
}

/// Title input routing (cursor nav + confirm + loadout back).
fn tick_title_input(world: &mut World) {
    let (cycle, slot, confirm, back) = {
        let mut input = world.resource_mut::<NtInput>();
        (
            input.take_cycle_weapon(),
            input.take_weapon_slot(),
            input.take_interact_pressed(),
            input.take_spec_pressed(),
        )
    };
    if cycle != 0 {
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            let len = CHAR_SELECT_ORDER.len() as i16;
            menu.title_cursor =
                (menu.title_cursor as i16 + cycle as i16).rem_euclid(len) as usize;
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

/// In-game menu routing (game-over > mutation offer > pause > unlocks).
fn tick_ingame_menu(world: &mut World, edge: MenuEdge) {
    // Offer mirror (bevy hud sync ran before input handling).
    {
        let has_pending = world.get_resource::<PendingMutation>().is_some();
        let has_ultra = world.get_resource::<PendingUltra>().is_some();
        let (count, is_ultra) = if has_ultra {
            (
                world.get_resource::<PendingUltra>().map(|p| p.choices.len()).unwrap_or(0),
                true,
            )
        } else if has_pending {
            (
                world.get_resource::<PendingMutation>().map(|p| p.choices.len()).unwrap_or(0),
                false,
            )
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
        let mut paused = world.remove_resource::<crate::state::Paused>().unwrap_or_default();
        let mut overlay = world.remove_resource::<OverlayMenu>().unwrap_or_default();
        let mut pending = world.remove_resource::<PendingUnpause>().unwrap_or_default();
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
        && world.get_resource::<MenuState>().is_some_and(|menu| menu.game_over.is_none());
    if needs_capture {
        let screen = capture_game_over(world);
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            menu.game_over = screen;
        }
    }

    let (cycle, slot, confirm, back) = {
        // Take-once pulses are shared with gameplay systems that run
        // later in the schedule (`weapon_switch` takes cycle/slot,
        // `collect_pickups` takes interact, weapons/abilities take spec;
        // bevy has no menu consumer for these). Only take when a menu is
        // actually open — otherwise E / 1-4 / right-click would be
        // swallowed every tick and weapons could never be picked up.
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
            (
                input.take_cycle_weapon(),
                input.take_weapon_slot(),
                input.take_interact_pressed(),
                input.take_spec_pressed(),
            )
        } else {
            (0, None, false, false)
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

    let offer_open = world
        .get_resource::<MenuState>()
        .is_some_and(|menu| menu.mutation_count > 0);
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
                menu.mutation_selected =
                    Some((cur + cycle as i16).rem_euclid(count) as usize);
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

    let overlay = world.resource::<OverlayMenu>();
    let overlay_kind = *overlay;
    match overlay_kind {
        OverlayMenu::Pause if confirm => {
            apply_menu_action(world, UiAction::Resume);
        }
        OverlayMenu::None => {
            // Unlock popups dismiss on confirm when nothing else owns it.
            let has_unlocks = world
                .get_resource::<MenuState>()
                .is_some_and(|menu| !menu.unlock_queue.is_empty());
            if confirm && has_unlocks {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    dismiss_unlock(&mut menu);
                }
            }
        }
        _ => {
            if back {
                apply_menu_action(world, UiAction::CloseOverlay);
            }
        }
    }
    let _ = (cycle, slot);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::ReactiveCue;
    use crate::comps_a::{Health, Inventory, RaceState};
    use crate::data::{MutationId, SkinLetter, WeaponId};
    use crate::savedata_part::try_unlock_race;
    use repame_sim::SimTime;

    fn sim_time(world: &mut World) {
        let mut time = SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
    }

    fn menu_world(state: AppState) -> World {
        let mut world = World::new();
        world.insert_resource(state);
        world.init_resource::<MenuState>();
        world.init_resource::<MenuEdge>();
        world.init_resource::<OverlayMenu>();
        world.init_resource::<crate::state::Paused>();
        world.init_resource::<PendingUnpause>();
        world.init_resource::<NtInput>();
        world.init_resource::<SaveData>();
        world.init_resource::<SelectedCharacter>();
        world.init_resource::<MutationChoice>();
        world.init_resource::<SaveDirty>();
        world.init_resource::<Queue<UiBridgeAction>>();
        world.init_resource::<Queue<ReactiveAudioRequest>>();
        sim_time(&mut world);
        world
    }

    fn drained_cues(world: &mut World) -> Vec<ReactiveAudioRequest> {
        world.resource_mut::<Queue<ReactiveAudioRequest>>().drain()
    }

    fn drained_sfx(world: &mut World) -> Vec<crate::audio::AudioCue> {
        world.resource_mut::<Queue<crate::audio::AudioCue>>().drain()
    }

    #[test]
    fn ingame_pulses_survive_without_open_menu() {
        // Bevy parity: no menu takes interact/cycle/slot/spec — only
        // `collect_pickups` / `weapon_switch` / abilities do. With no
        // overlay, offer, game over, or unlocks, `tick_menus` must leave
        // every pulse for the gameplay systems (else E never picks up
        // weapons and 1-4 never switches).
        use crate::input::{MouseState, sample_keyboard};
        use std::collections::HashSet;

        let mut world = menu_world(AppState::InGame);
        {
            let mut input = world.resource_mut::<NtInput>();
            input.press_interact();
            input.cycle_weapon(1);
            input.select_weapon(1);
            sample_keyboard(
                &HashSet::new(),
                &HashSet::new(),
                &MouseState {
                    right_pressed: true,
                    ..MouseState::default()
                },
                &mut input,
            );
        }
        tick_menus(&mut world);
        let mut input = world.resource_mut::<NtInput>();
        assert!(
            input.take_interact_pressed(),
            "E survives for weapon pickup"
        );
        assert_eq!(input.take_cycle_weapon(), 1, "cycle survives");
        assert_eq!(input.take_weapon_slot(), Some(1), "slot survives");
        assert!(input.take_spec_pressed(), "spec survives for abilities");
    }

    #[test]
    fn pause_overlay_still_consumes_confirm() {
        // Gating must not break real menus: an open pause overlay takes
        // the confirm pulse and resumes.
        let mut world = menu_world(AppState::InGame);
        *world.resource_mut::<OverlayMenu>() = OverlayMenu::Pause;
        world.resource_mut::<NtInput>().press_interact();
        tick_menus(&mut world);
        assert!(
            !world.resource::<NtInput>().peek_interact_pressed(),
            "pause consumed the confirm"
        );
        assert_eq!(
            *world.resource::<OverlayMenu>(),
            OverlayMenu::None,
            "confirm resumes"
        );
    }
    #[test]
    fn splash_reaches_menu_through_driver() {
        let mut world = menu_world(AppState::Splash);
        for _ in 0..600 {
            tick_menus(&mut world);
            if *world.resource::<AppState>() == AppState::MainMenu {
                break;
            }
        }
        assert_eq!(*world.resource::<AppState>(), AppState::MainMenu);
    }

    #[test]
    fn title_confirm_starts_loading_then_ingame_via_setup() {
        let mut world = menu_world(AppState::Title);
        // Cursor on Fish (gml 1), already selected -> confirm starts loading.
        world.resource_mut::<MenuState>().title_cursor = RaceId::Fish as usize;
        world.resource_mut::<NtInput>().press_interact();
        tick_menus(&mut world);
        assert_eq!(*world.resource::<AppState>(), AppState::Loading);
        for _ in 0..40 {
            tick_menus(&mut world);
            if *world.resource::<AppState>() == AppState::InGame {
                break;
            }
        }
        assert_eq!(*world.resource::<AppState>(), AppState::InGame);
        assert!(world.query::<&Player>().iter(&world).count() > 0);
        assert_eq!(world.resource::<Run>().floor, 1);
    }

    #[test]
    fn pause_toggles_with_pending_delay_through_driver() {
        let mut world = World::new();
        crate::setup::setup_run_with_seed(&mut world, 0x9A05);
        sim_time(&mut world);
        assert!(!world.resource::<crate::state::Paused>().0);

        world.resource_mut::<MenuEdge>().pause_pressed = true;
        tick_menus(&mut world);
        assert!(world.resource::<crate::state::Paused>().0);
        assert_eq!(*world.resource::<OverlayMenu>(), OverlayMenu::Pause);

        // Resume action arms the delay; expiry lifts pause.
        apply_menu_action(&mut world, UiAction::Resume);
        assert_eq!(*world.resource::<OverlayMenu>(), OverlayMenu::None);
        assert!(world.resource::<PendingUnpause>().0.is_some());
        let mut sched = Schedule::default();
        sched.add_systems(crate::state::tick_pending_unpause);
        for _ in 0..6 {
            sched.run(&mut world);
        }
        assert!(!world.resource::<crate::state::Paused>().0);
        assert!(world.resource::<PendingUnpause>().0.is_none());
    }

    #[test]
    fn escape_out_of_pause_arms_delay() {
        let mut world = menu_world(AppState::InGame);
        world.resource_mut::<crate::state::Paused>().0 = true;
        *world.resource_mut::<OverlayMenu>() = OverlayMenu::Pause;
        world.resource_mut::<MenuEdge>().pause_pressed = true;
        tick_menus(&mut world);
        assert_eq!(*world.resource::<OverlayMenu>(), OverlayMenu::None);
        assert!(world.resource::<PendingUnpause>().0.is_some());
        // Paused until the timer drains.
        assert!(world.resource::<crate::state::Paused>().0);
    }

    #[test]
    fn mutation_two_step_commits_and_resumes() {
        let mut world = World::new();
        crate::setup::setup_run_with_seed(&mut world, 0x9077);
        sim_time(&mut world);
        world.init_resource::<MenuState>();
        world.init_resource::<MenuEdge>();
        world.init_resource::<OverlayMenu>();
        world.init_resource::<Queue<UiBridgeAction>>();
        world.init_resource::<Queue<ReactiveAudioRequest>>();
        world.insert_resource(PendingMutation {
            choices: vec![MutationId::RhinoSkin, MutationId::EagleEyes],
        });
        world.init_resource::<crate::audio::GameAudio>();
        world.init_resource::<Queue<crate::audio::AudioCue>>();
        world.init_resource::<repame_fx::Trauma>();
        world.init_resource::<crate::effects::ChromaticAberration>();
        world.init_resource::<crate::effects::SlowMotion>();

        // Slot pulse highlights; interact commits the highlight.
        world.resource_mut::<NtInput>().select_weapon(0);
        tick_menus(&mut world);
        assert_eq!(world.resource::<MenuState>().mutation_selected, Some(0));
        assert_eq!(world.resource::<MutationChoice>().0, None);
        world.resource_mut::<NtInput>().press_interact();
        tick_menus(&mut world);
        assert_eq!(world.resource::<MutationChoice>().0, Some(0));
        assert_eq!(world.resource::<MenuState>().mutation_selected, None);

        // Out-of-range slot pulses are dropped (bevy bounds law).
        world.resource_mut::<NtInput>().select_weapon(7);
        tick_menus(&mut world);
        assert_eq!(world.resource::<MenuState>().mutation_selected, None);

        // The existing choice protocol consumes the pick and resumes.
        let mut sched = Schedule::default();
        sched.add_systems(crate::progression::handle_mutation_choice);
        sched.run(&mut world);
        assert_eq!(world.resource::<MutationChoice>().0, None);
        assert!(world.get_resource::<PendingMutation>().is_none());
        assert!(!world.resource::<crate::state::Paused>().0);
        let hp = world
            .query_filtered::<&Health, With<Player>>()
            .single(&world)
            .unwrap()
            .max;
        assert!(hp > 8, "Rhino Skin applied, max hp {hp}");
    }

    #[test]
    fn character_select_gates_locked_races() {
        let mut world = menu_world(AppState::Title);
        // Crystal (gml 2) starts locked: denied, selection kept, denial sting.
        apply_menu_action(&mut world, UiAction::SelectCharacter(RaceId::Crystal as usize));
        assert_eq!(world.resource::<SelectedCharacter>().0, RaceId::Fish);
        let sfx = drained_sfx(&mut world);
        assert_eq!(sfx.len(), 1);
        assert_eq!(sfx[0].name, "sndNoSelect");
        assert!((sfx[0].volume - 0.5).abs() < 1e-6);

        // Unknown gml ids are ignored silently.
        apply_menu_action(&mut world, UiAction::SelectCharacter(99));
        assert_eq!(world.resource::<SelectedCharacter>().0, RaceId::Fish);

        // Unlock -> select writes character + arms GO + confirm cue.
        assert!(try_unlock_race(world.resource_mut::<SaveData>().into_inner(), RaceId::Crystal));
        apply_menu_action(&mut world, UiAction::SelectCharacter(RaceId::Crystal as usize));
        assert_eq!(world.resource::<SelectedCharacter>().0, RaceId::Crystal);
        assert!(world.resource::<MenuState>().title_go_visible);
        let cues = drained_cues(&mut world);
        assert!(cues.iter().any(|c| c.cue == ReactiveCue::UiConfirm));
        // GML race select sting: Crystal (gml 2) → sndMutant2Slct.
        let sfx = drained_sfx(&mut world);
        assert_eq!(sfx.len(), 1);
        assert_eq!(sfx[0].name, "sndMutant2Slct");
        assert!((sfx[0].volume - 1.0).abs() < 1e-6);

        // Re-clicking the selected race starts loading (bevy law).
        apply_menu_action(&mut world, UiAction::SelectCharacter(RaceId::Crystal as usize));
        assert_eq!(*world.resource::<AppState>(), AppState::Loading);
    }

    #[test]
    fn skin_select_gates_locked_skins() {
        let mut world = menu_world(AppState::Title);
        assert!(try_unlock_race(world.resource_mut::<SaveData>().into_inner(), RaceId::Crystal));
        world.resource_mut::<SelectedCharacter>().0 = RaceId::Crystal;
        // Locked B skin: denied, preferred kept.
        apply_menu_action(&mut world, UiAction::SelectSkin(1));
        assert_eq!(world.resource::<SaveData>().race_loadout(RaceId::Crystal).preferred_skin, 0);
        assert!(!world.resource::<SaveDirty>().0);
        // Random race ignores skin picks entirely.
        world.resource_mut::<SelectedCharacter>().0 = RaceId::Random;
        apply_menu_action(&mut world, UiAction::SelectSkin(1));
        // Unlock path writes + dirties.
        world.resource_mut::<SelectedCharacter>().0 = RaceId::Crystal;
        world.resource_mut::<SaveData>().race_loadout_mut(RaceId::Crystal).unlocked_skins[1] = true;
        apply_menu_action(&mut world, UiAction::SelectSkin(1));
        assert_eq!(world.resource::<SaveData>().race_loadout(RaceId::Crystal).preferred_skin, 1);
        assert!(world.resource::<SaveDirty>().0);
    }

    #[test]
    fn crown_select_gates_locked_crowns_and_cycles() {
        let mut world = menu_world(AppState::Title);
        // Locked crown (gml 5): denied, stamp kept.
        apply_menu_action(&mut world, UiAction::SelectCrown(5));
        assert_eq!(world.resource::<SaveData>().race_loadout(RaceId::Fish).start_crown, 0);
        // Unlocking auto-equips (bevy `unlock_crown` stamps the port id),
        // so pick another open crown: gml 0-1 start open (port 0).
        world.resource_mut::<SaveData>().unlock_crown(RaceId::Fish, 3);
        assert_eq!(world.resource::<SaveData>().race_loadout(RaceId::Fish).start_crown, 2);
        world.resource_mut::<SaveDirty>().0 = false;
        apply_menu_action(&mut world, UiAction::SelectCrown(1));
        assert_eq!(world.resource::<SaveData>().race_loadout(RaceId::Fish).start_crown, 0);
        assert!(world.resource::<SaveDirty>().0);
        // Re-selecting the equipped crown is a silent no-op.
        let _ = drained_cues(&mut world);
        apply_menu_action(&mut world, UiAction::SelectCrown(1));
        assert!(drained_cues(&mut world).is_empty());
        // Cycling forward from None skips locked ports, lands port 2.
        apply_menu_action(&mut world, UiAction::CycleCrown(1));
        assert_eq!(world.resource::<SaveData>().race_loadout(RaceId::Fish).start_crown, 2);
        // Random race ignores crown picks (stamp untouched).
        world.resource_mut::<SelectedCharacter>().0 = RaceId::Random;
        apply_menu_action(&mut world, UiAction::SelectCrown(3));
        assert_eq!(world.resource::<SaveData>().race_loadout(RaceId::Fish).start_crown, 2);
    }

    #[test]
    fn start_weapon_toggle_needs_stored_gun() {
        let mut world = menu_world(AppState::Title);
        // No stored gun: no-op (bevy guard), still an accepted action.
        apply_menu_action(&mut world, UiAction::CycleStartWeapon(1));
        assert_eq!(
            world.resource::<SaveData>().race_loadout(RaceId::Fish).start_weapon,
            WeaponId::NONE
        );
        assert!(!world.resource::<SaveDirty>().0);
        // Stored gun present: toggles start between stored and none.
        world.resource_mut::<SaveData>().race_loadout_mut(RaceId::Fish).stored_weapon =
            WeaponId::REVOLVER;
        apply_menu_action(&mut world, UiAction::CycleStartWeapon(1));
        assert_eq!(
            world.resource::<SaveData>().race_loadout(RaceId::Fish).start_weapon,
            WeaponId::REVOLVER
        );
        apply_menu_action(&mut world, UiAction::CycleStartWeapon(1));
        assert_eq!(
            world.resource::<SaveData>().race_loadout(RaceId::Fish).start_weapon,
            WeaponId::NONE
        );
        // Stored-weapon cycle is a bevy empty-body no-op; the raw row
        // keeps the stored gun (the sanitized `race_loadout` copy drops
        // orphan stored guns on read — bevy `race_loadout` law).
        apply_menu_action(&mut world, UiAction::CycleStoredWeapon(1));
        assert_eq!(
            world.resource_mut::<SaveData>().race_loadout_mut(RaceId::Fish).stored_weapon,
            WeaponId::REVOLVER
        );
    }

    #[test]
    fn settings_sliders_clamp_and_persist_dirty() {
        let mut world = menu_world(AppState::InGame);
        apply_menu_action(&mut world, UiAction::SetMasterVol(2.0));
        assert_eq!(world.resource::<SaveData>().settings.master_volume, 1.0);
        assert_eq!(world.resource::<crate::audio::AudioChannels>().master, 1.0);
        apply_menu_action(&mut world, UiAction::SetSfxVol(-3.0));
        assert_eq!(world.resource::<SaveData>().settings.sfx_volume, 0.0);
        apply_menu_action(
            &mut world,
            UiAction::SettingSlider { key: "screenshake".to_string(), value: 5.0 },
        );
        assert_eq!(world.resource::<SaveData>().settings.screenshake, 2.0);
        apply_menu_action(
            &mut world,
            UiAction::SettingSlider { key: "freezeframes".to_string(), value: -1.0 },
        );
        assert_eq!(world.resource::<SaveData>().settings.freezeframes, 0.0);
        apply_menu_action(
            &mut world,
            UiAction::SettingSlider { key: "controls_scale".to_string(), value: 9.0 },
        );
        assert_eq!(world.resource::<SaveData>().settings.controls_scale, 1.0);
        // Unknown sliders are ignored without dirtying.
        world.resource_mut::<SaveDirty>().0 = false;
        apply_menu_action(
            &mut world,
            UiAction::SettingSlider { key: "nope".to_string(), value: 1.0 },
        );
        assert!(!world.resource::<SaveDirty>().0);

        // Cycles wrap (mod 4; 1-based pixel_mode).
        apply_menu_action(&mut world, UiAction::SettingCycle { key: "crosshair".to_string(), dir: 1 });
        assert_eq!(world.resource::<SaveData>().settings.crosshair, 1);
        world.resource_mut::<SaveData>().settings.crosshair = 3;
        apply_menu_action(&mut world, UiAction::SettingCycle { key: "crosshair".to_string(), dir: 1 });
        assert_eq!(world.resource::<SaveData>().settings.crosshair, 0);
        world.resource_mut::<SaveData>().settings.pixel_mode = 4;
        apply_menu_action(&mut world, UiAction::SettingCycle { key: "pixel_mode".to_string(), dir: 1 });
        assert_eq!(world.resource::<SaveData>().settings.pixel_mode, 1);

        // Toggles flip (incl. cprefs), unknown keys ignored.
        assert!(world.resource::<SaveData>().settings.show_hud);
        apply_menu_action(&mut world, UiAction::SettingToggle("show_hud".to_string()));
        assert!(!world.resource::<SaveData>().settings.show_hud);
        assert!(world.resource::<SaveDirty>().0);
        world.resource_mut::<SaveDirty>().0 = false;
        apply_menu_action(&mut world, UiAction::SettingToggle("cprefs_2".to_string()));
        assert!(world.resource::<SaveData>().settings.cprefs_plant);
        assert!(world.resource::<SaveDirty>().0);
        world.resource_mut::<SaveDirty>().0 = false;
        apply_menu_action(&mut world, UiAction::SettingToggle("bogus".to_string()));
        assert!(!world.resource::<SaveDirty>().0);

        // Text inputs write through verbatim.
        apply_menu_action(
            &mut world,
            UiAction::SettingInput { key: "profile_name".to_string(), value: "Vlambeer".to_string() },
        );
        assert_eq!(world.resource::<SaveData>().settings.profile_name, "Vlambeer");

        // Languages cycle the bevy list; explicit sets write through.
        apply_menu_action(&mut world, UiAction::NextLanguage);
        assert_eq!(world.resource::<SaveData>().settings.language, "es");
        apply_menu_action(&mut world, UiAction::SetLanguage("ja".to_string()));
        assert_eq!(world.resource::<SaveData>().settings.language, "ja");

        // Settings page stack pushes/pops, then closes to the right overlay.
        world.resource_mut::<crate::state::Paused>().0 = true;
        *world.resource_mut::<OverlayMenu>() = OverlayMenu::Settings;
        apply_menu_action(&mut world, UiAction::SettingsCategory(3));
        assert_eq!(world.resource::<MenuState>().settings_page, 3);
        apply_menu_action(&mut world, UiAction::SettingsBack);
        assert_eq!(world.resource::<MenuState>().settings_page, 0);
        apply_menu_action(&mut world, UiAction::SettingsBack);
        assert_eq!(*world.resource::<OverlayMenu>(), OverlayMenu::Pause);
    }

    #[test]
    fn settings_reset_and_erase_match_bevy() {
        let mut world = menu_world(AppState::InGame);
        world.resource_mut::<SaveData>().high_score = 999;
        world.resource_mut::<SaveData>().settings.master_volume = 0.1;
        apply_menu_action(&mut world, UiAction::SettingEraseProgress);
        assert_eq!(world.resource::<SaveData>().high_score, 0);
        assert_eq!(world.resource::<SaveData>().unlocked_characters, vec!["Fish".to_string()]);
        assert!(world.resource::<SaveData>().races.is_empty());
        // Erase keeps options (bevy law).
        assert_eq!(world.resource::<SaveData>().settings.master_volume, 0.1);
        apply_menu_action(&mut world, UiAction::SettingResetOptions);
        assert_eq!(world.resource::<SaveData>().settings.master_volume, 1.0);
        assert!(world.resource::<SaveDirty>().0);
    }

    #[test]
    fn pause_confirm_quit_vs_restart() {
        let mut world = menu_world(AppState::InGame);
        world.resource_mut::<crate::state::Paused>().0 = true;
        *world.resource_mut::<OverlayMenu>() = OverlayMenu::Pause;
        apply_menu_action(&mut world, UiAction::ShowPauseConfirm(1));
        assert_eq!(world.resource::<MenuState>().pause_confirm, Some(1));
        apply_menu_action(&mut world, UiAction::CancelPauseConfirm);
        assert_eq!(world.resource::<MenuState>().pause_confirm, None);

        apply_menu_action(&mut world, UiAction::ConfirmPause(0));
        assert_eq!(*world.resource::<AppState>(), AppState::MainMenu);
        assert!(!world.resource::<crate::state::Paused>().0);

        let mut world = menu_world(AppState::InGame);
        apply_menu_action(&mut world, UiAction::ConfirmPause(1));
        assert_eq!(*world.resource::<AppState>(), AppState::Loading);
    }

    #[test]
    fn game_over_captures_then_restarts_or_quits() {
        let mut world = World::new();
        crate::setup::setup_run_with_seed(&mut world, 0xD1ED);
        sim_time(&mut world);
        world.init_resource::<MenuState>();
        world.init_resource::<MenuEdge>();
        world.init_resource::<Queue<UiBridgeAction>>();
        world.init_resource::<Queue<ReactiveAudioRequest>>();
        world.resource_mut::<Run>().game_over = true;
        world.resource_mut::<Run>().total_kills = 41;
        world.resource_mut::<Score>().0 = 1337;

        tick_menus(&mut world);
        let screen = world.resource::<MenuState>().game_over.expect("snapshot");
        assert_eq!((screen.total_kills, screen.score), (41, 1337));
        // Death guard runs alongside: pause/overlay stay clear.
        let mut sched = Schedule::default();
        sched.add_systems(crate::state::force_death_overlay_state);
        world.resource_mut::<crate::state::Paused>().0 = true;
        sched.run(&mut world);
        assert!(!world.resource::<crate::state::Paused>().0);

        // R restarts through loading into a fresh run.
        world.resource_mut::<MenuEdge>().restart_pressed = true;
        tick_menus(&mut world);
        assert_eq!(*world.resource::<AppState>(), AppState::Loading);
        for _ in 0..40 {
            tick_menus(&mut world);
        }
        assert_eq!(*world.resource::<AppState>(), AppState::InGame);
        assert!(!world.resource::<Run>().game_over);

        // Click (interact) quits to the menu instead.
        world.resource_mut::<Run>().game_over = true;
        tick_menus(&mut world);
        assert!(world.resource::<MenuState>().game_over.is_some());
        world.resource_mut::<NtInput>().press_interact();
        tick_menus(&mut world);
        assert_eq!(*world.resource::<AppState>(), AppState::MainMenu);
    }

    #[test]
    fn quit_app_sets_flag_and_loadout_toggles() {
        let mut world = menu_world(AppState::MainMenu);
        apply_menu_action(&mut world, UiAction::QuitApp);
        assert!(world.resource::<QuitRequested>().0);

        let mut world = menu_world(AppState::Title);
        assert!(!world.resource::<MenuState>().loadout_open);
        apply_menu_action(&mut world, UiAction::ToggleLoadout);
        assert!(world.resource::<MenuState>().loadout_open);

        // BigDog-likes force the loadout shut on select (bevy law).
        assert!(try_unlock_race(world.resource_mut::<SaveData>().into_inner(), RaceId::Skeleton));
        apply_menu_action(&mut world, UiAction::SelectCharacter(RaceId::Skeleton as usize));
        assert!(!world.resource::<MenuState>().loadout_open);
    }

    #[test]
    fn unlock_queue_push_dismiss_and_mirror_laws() {
        let mut world = menu_world(AppState::InGame);
        push_unlock(world.resource_mut::<MenuState>().into_inner(), UnlockPopup::Race(RaceId::Crystal));
        push_unlock(
            world.resource_mut::<MenuState>().into_inner(),
            UnlockPopup::Skin(RaceId::Fish, 1),
        );
        // Confirm dismisses one popup when nothing else owns input.
        world.resource_mut::<NtInput>().press_interact();
        tick_menus(&mut world);
        assert_eq!(world.resource::<MenuState>().unlock_queue.len(), 1);

        // Ultra offers take precedence in the mirror; len change clears.
        world.insert_resource(PendingMutation { choices: vec![MutationId::RabbitPaw] });
        world.insert_resource(PendingUltra {
            choices: vec![
                crate::data::UltraMutationId::FishGunWarrant,
                crate::data::UltraMutationId::FishConfiscate,
            ],
        });
        tick_menus(&mut world);
        let menu = world.resource::<MenuState>();
        assert!((menu.mutation_count, menu.mutation_is_ultra) == (2, true));
        world.remove_resource::<PendingUltra>();
        world.resource_mut::<MenuState>().mutation_selected = Some(0);
        tick_menus(&mut world);
        let menu = world.resource::<MenuState>();
        assert!((menu.mutation_count, menu.mutation_is_ultra) == (1, false));
        assert_eq!(menu.mutation_selected, None);
    }

    #[test]
    fn title_cursor_wraps_and_slot_jumps() {
        let mut world = menu_world(AppState::Title);
        world.resource_mut::<MenuState>().title_cursor = 0;
        world.resource_mut::<NtInput>().cycle_weapon(-1);
        tick_menus(&mut world);
        assert_eq!(world.resource::<MenuState>().title_cursor, CHAR_SELECT_ORDER.len() - 1);
        world.resource_mut::<NtInput>().select_weapon(3);
        tick_menus(&mut world);
        // Slot 3 == Melting pod; Melting is locked so cursor moves, select denied.
        assert_eq!(world.resource::<MenuState>().title_cursor, 3);
        assert_eq!(world.resource::<SelectedCharacter>().0, RaceId::Fish);
    }

    #[test]
    fn hud_half_mirrors_race_skin_and_crown_names() {
        // Documents the render-later read path: everything the title
        // views need is derivable from save + selection, no mirror kept.
        let world = menu_world(AppState::Title);
        let save = world.resource::<SaveData>();
        let sel = world.resource::<SelectedCharacter>().0;
        let lo = save.race_loadout(sel);
        assert_eq!(crate::savedata_part::character_def(sel).name, "Fish");
        assert_eq!(crown_short_name(lo.start_crown), "NONE");
        assert_eq!(RaceState { race: sel, skin: SkinLetter::A }.race, RaceId::Fish);
        let _ = Inventory {
            weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
            cursed: [false, false, false],
            swapanim: 0.0,
            shine: 0.0,
            wepflip: 1.0,
            bwepflip: 1.0,
            weapon_slots: 2,
            current: 0,
            ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
        };
    }
}
