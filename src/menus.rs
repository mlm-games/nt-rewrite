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
//! - Title: `cycle_weapon` moves the character cursor over the visible
//!   pod roster (wraps; GML `_char_list` order), `weapon_slot` jumps to
//!   a gml-id pod (hidden-and-locked races sting `sndNoSelect`),
//!   `interact` confirms (re-clicking the selected race starts loading,
//!   bevy parity), `spec` toggles the loadout panel shut when open.
//! - Mutation: `weapon_slot` (Digit1-4) routes through the bevy two-step
//!   (`SelectMutation` highlight then `PickMutation` commit, same as
//!   bevy `handle_mutation_keys`), `cycle_weapon` moves the highlight,
//!   `interact` commits the highlight.
//! - Pause: `interact` resumes, `spec` closes the top overlay,
//!   `MenuEdge::pause_pressed` (Escape) toggles with bevy's confirm/
//!   settings-stack laws.
//! - Game over: `MenuEdge::restart_pressed` (KeyR) restarts via Loading;
//!   MENU/RETRY buttons route to `ConfirmPause(0/1)` (GML direct actions,
//!   no confirm); stray clicks do nothing.
//! - Splash: any key/mouse edge advances (bevy `boot_intro` law).
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

/// GML `scrRaceIsHidden` verbatim (default `count_in_unlockable=true`,
/// the `Menu/Create_0` call): BigDog always hidden; Frog/Skeleton
/// hidden while countable.
pub fn race_is_hidden(race: RaceId) -> bool {
    matches!(race, RaceId::BigDog | RaceId::Frog | RaceId::Skeleton)
}

/// GML `Menu/Create_0` pod roster verbatim: every race that is not
/// hidden, plus hidden races once unlocked (`cgot`, i.e.
/// [`SaveData::race_unlocked`]). Fresh saves show 14 pods (step 20);
/// fully unlocked saves show 17 (step 16). [`MenuState::title_cursor`]
/// indexes this roster, NOT the gml id.
pub fn visible_roster(save: Option<&SaveData>) -> Vec<RaceId> {
    CHAR_SELECT_ORDER
        .iter()
        .copied()
        .filter(|r| !race_is_hidden(*r) || save.is_some_and(|s| s.race_unlocked(*r)))
        .collect()
}

/// Roster position of a gml id (`None` for hidden-and-locked races,
/// which have no pod).
pub fn roster_position(save: Option<&SaveData>, gml: usize) -> Option<usize> {
    visible_roster(save).iter().position(|r| *r as usize == gml)
}

/// GML `scrRaceGetUnlockDescription` verbatim
/// (`scripts/scrRaces/scrRaces.gml`): locked-pod hint text for
/// `Menu.unlock_hint` (touch path in `CharSelect/Mouse_4`). Unlocalized
/// English defaults; loc lives shell-side.
pub fn unlock_hint_for_race(race: RaceId) -> String {
    match race {
        RaceId::Fish | RaceId::Crystal => "UNLOCKED FROM THE START".to_string(),
        RaceId::Eyes => "REACH THE SEWERS".to_string(),
        RaceId::Melting => "DIE".to_string(),
        RaceId::Plant => "REACH THE SCRAPYARD".to_string(),
        RaceId::Venuz => "REACH 3-?".to_string(),
        RaceId::Steroids => "REACH THE LABS".to_string(),
        RaceId::Robot => "REACH THE FROZEN CITY".to_string(),
        RaceId::Chicken => "REACH 5-?".to_string(),
        RaceId::Rebel => "??? THE GAME".to_string(),
        RaceId::Horror => "DEFEAT WILD HORROR".to_string(),
        RaceId::Rogue => "DEFEAT THE NUCLEAR THRONE".to_string(),
        RaceId::BigDog => "BEAT THE BIG DOG".to_string(),
        RaceId::Skeleton | RaceId::Frog => "SECRET CHARACTER".to_string(),
        RaceId::Cuz | RaceId::Random => "???".to_string(),
    }
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

/// GML `scr_death_cause_is_valid` + `scrDeathCauseGetSprite` verbatim
/// over the port [`HitId`](crate::comps_a::HitId): enemy hits (either
/// an explicit kind or an `Enemy(id)` hit) resolve to the killer's
/// idle strip (`enemy_def(kind).sprite`, the same `spr*Idle` table GML
/// `scrDeathCauseDefine`s); `Explosion` → `sprExplosion`, `Toxic` →
/// `sprToxicGas`, `Fire`/`Trap` → `sprTrapGameover`. Everything else
/// (contact/bullets/crowns/unknown) is not a valid GML cause and draws
/// nothing — exactly like the `sprite_exists` gate in
/// `GameOver/Draw_0`.
pub fn deathcause_sprite_for_hit(
    hit: Option<crate::comps_a::HitId>,
    enemy_kind: Option<crate::data::EnemyKind>,
) -> Option<&'static str> {
    use crate::comps_a::HitId;
    if let Some(kind) = enemy_kind {
        return Some(crate::enemy_data::enemy_def(kind).sprite);
    }
    match hit {
        Some(HitId::Enemy(id)) => {
            crate::data::EnemyKind::from_u16(id).map(|k| crate::enemy_data::enemy_def(k).sprite)
        }
        Some(HitId::Explosion(_)) => Some("images/sprExplosion.png"),
        Some(HitId::Toxic) => Some("images/sprToxicGas.png"),
        Some(HitId::Fire) | Some(HitId::Trap) => Some("images/sprTrapGameover.png"),
        _ => None,
    }
}

/// GML `scr_race_get_skin_subimage` verbatim (`scrRaces.gml:117`):
/// Robot-D sits at 56 (NTT bug), else `(skin + (race-1)*2)` for A/B
/// and `(skin*16 + (race-1))` for C/D; race 0 (Random) draws nothing
/// (-1). `race_gml` is the gml id, `skin` the letter index (A=0..).
pub fn race_skin_subimage(race_gml: usize, skin: u8) -> i32 {
    if race_gml == 0 {
        return -1;
    }
    if race_gml == crate::data::RaceId::Robot as usize && skin == 3 {
        return 56;
    }
    if skin < 2 {
        (skin as i32) + (race_gml as i32 - 1) * 2
    } else {
        (skin as i32) * 16 + (race_gml as i32 - 1)
    }
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
    /// GML `GameCont.deathcause` sprite (`GameOver/Draw_0`:
    /// `scrDeathCauseGetSprite`, `-1` animated frame). `None` when the
    /// cause is not a valid GML cause (contact/bullets/crowns).
    pub deathcause_sprite: Option<&'static str>,
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
        deathcause_sprite: world
            .get_resource::<crate::comps_a::LastDamageTaken>()
            .and_then(|last| deathcause_sprite_for_hit(last.hit_id, last.enemy_kind)),
    };
    Some(screen)
}

/// Canonical menu UI state (headless half of bevy `SharedUi`: every
/// field the views will need, none of the pixels).
#[derive(Debug, Clone, Resource)]
pub struct MenuState {
    /// Cursor into the visible pod roster ([`visible_roster`], GML
    /// `_char_list` order — a roster index, not a gml id).
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
    /// PLAY-submenu open (GML `PlayButton` rows replace the main-menu
    /// buttons until one fires or BackButton/Escape closes).
    pub play_submenu: bool,
    /// PLAY-submenu keyboard cursor over [`play_rows`].
    pub play_cursor: usize,
    /// Settings keyboard cursor over the page's actionable rows
    /// (`settings_hot_rows` order in `render.rs`; GML `pointed_item`).
    pub settings_cursor: usize,
    /// Pending unlock popups (producer deferred; see module docs).
    pub unlock_queue: Vec<UnlockPopup>,
    /// Credits section index (GML `Credits.show` over `credittext`).
    pub credits_section: usize,
    /// Seconds on the current credits section (GML `timer`).
    pub credits_t: f32,
    /// Credits pan offset in GUI px for tall sections (GML `scroll`).
    pub credits_scroll: f32,
    /// Live game-over snapshot (`None` until the run ends).
    pub game_over: Option<GameOverScreen>,
    /// GML `GameOver/Create_0` anim state verbatim: `death_pos` (waypoint
    /// reveal prefix, +1/draw capped at `waypoints`), `offsety` (128 ->
    /// 0 at 32/draw), `splatimg` (0 -> 2 at 0.7/draw once the
    /// letterbox is open; the port has no letterbox gate so it always
    /// animates), and the two `PauseButton` `appear = 3 + image`
    /// stagger (3 MENU, 4 RETRY, -1/tick; buttons clickable at 0).
    /// Reset on capture, ticked in `tick_ingame_menu`.
    pub go_death_pos: f32,
    pub go_offsety: f32,
    pub go_splat: f32,
    pub go_appear: f32,
    /// GML `Menu` campfire anim state verbatim (`Create_0:63-65`,
    /// `Other_11:17-37`): per-player portrait slide offsets (180 on
    /// select, then 180→90→-2→0), text typewriter stages (2 hidden on
    /// select, approaching 0), `splatindex` 0→3 at 0.4/step, and the
    /// loadout open frame. Ticked in `tick_title_input`.
    pub portrait_offsets: [f32; 4],
    pub textappear: [f32; 4],
    pub splatindex: f32,
    pub loadout_frame: f32,
    /// GML `Menu.unlock_hint/unlock_hint_pop/alarm[11]` verbatim: touch
    /// locked picks show the unlock description for 90 steps.
    pub unlock_hint: String,
    pub unlock_hint_pop: f32,
    pub unlock_hint_t: f32,
    /// GML `Menu.weekly` verbatim: weekly runs bypass race locks and hide
    /// the pod roster treatment (`can = unlocked || weekly_run`).
    pub weekly_run_menu: bool,
}

impl Default for MenuState {
    fn default() -> Self {
        Self {
            // GML `Menu/Create_0` starts on Random (`race = Race.Random`,
            // head of `_char_list`): roster index 0, not a gml id.
            title_cursor: 0,
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
            play_submenu: false,
            play_cursor: 0,
            credits_section: 0,
            credits_t: 0.0,
            credits_scroll: 0.0,
            settings_cursor: 0,
            unlock_queue: Vec::new(),
            game_over: None,
            go_death_pos: 0.0,
            go_offsety: 128.0,
            go_splat: 0.0,
            go_appear: 4.0,
            portrait_offsets: [0.0; 4],
            textappear: [2.0; 4],
            splatindex: 0.0,
            loadout_frame: 0.0,
            unlock_hint: String::new(),
            unlock_hint_pop: 0.0,
            unlock_hint_t: 0.0,
            weekly_run_menu: false,
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

/// GML `MainMenuButton/Other_10` PLAY-submenu rows verbatim: NORMAL
/// always; DAILY/WEEKLY when the tutorial is done; HARD when loop 2
/// cleared (`hardgot`); CUSTOM last. A single row auto-fires (GML
/// `event_user(0)`), so fresh/tutorial profiles skip the submenu.
/// DAILY/WEEKLY draw `c_uidark`-dimmed when `!can_daily/can_weekly`
/// (offline: the port has no daily/weekly backend, so both read
/// unavailable) but STAY clickable — GML opens the Leaderboards
/// instead of starting the run. The port has no Leaderboards entity,
/// so the click stings `sndNoSelect` (same feedback class).
pub fn play_rows(save: &SaveData) -> Vec<u8> {
    let mut rows = vec![0];
    if !save.settings.show_tutorial {
        rows.push(1);
        rows.push(2);
        if save.hardmode_unlocked {
            rows.push(3);
        }
        rows.push(4);
    }
    rows
}

/// GML `UberCont.can_daily/can_weekly` verbatim: both gate on the
/// online daily/weekly fetch (`Other_62`: `daily_seed > 0` /
/// `weekly_data[? "seed"]`). The port has no online backend, so both
/// are always false — DAILY/WEEKLY always draw dimmed (GML
/// `image_blend = c_uidark` arm in `MainMenuButton/Other_10`).
pub fn play_row_available(_save: &SaveData, row: u8) -> bool {
    !matches!(row, 1 | 2)
}

/// GML `PlayButton` label verbatim (`scrMenuButtonName`).
pub fn play_row_name(row: u8) -> &'static str {
    match row {
        0 => "NORMAL",
        1 => "DAILY",
        2 => "WEEKLY",
        3 => "HARD",
        _ => "CUSTOM",
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
    apply_mutation_mirror(menu, count, is_ultra);
}

/// Pure mirror law shared by [`tick_mutation_mirror`] and the InGame tick
/// below (single source; the InGame call site can only take owned lens
/// because of the `World` borrow checker).
fn apply_mutation_mirror(menu: &mut MenuState, count: usize, is_ultra: bool) {
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

/// Emit the bevy `SettingsBack` pop one-shot (`sndClickBack` 0.6). The
/// reactive `UiClick` cue comes from the static `ui_action_to_cue` map via
/// `emit_cue` (both pop and close paths emit it, bevy parity).
fn emit_click_back(world: &mut World) {
    world.init_resource::<Queue<crate::audio::AudioCue>>();
    world
        .resource_mut::<Queue<crate::audio::AudioCue>>()
        .push(crate::audio::AudioCue {
            name: "sndClickBack",
            volume: 0.6,
            variance: 0.0,
        });
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
                // Fresh run: drop any stale mutation highlight (bevy
                // `reset_hud_flags` clears `mutation_selected`; the mirror
                // only resets on count change, so a same-length offer in
                // the next run would otherwise inherit it and commit on
                // one click).
                menu.mutation_selected = None;
            }
            emit_cue(world, &UiAction::StartGame);
            goto_state(world, AppState::Loading);
        }
        UiAction::MainMenuPlay => {
            let rows = world
                .get_resource::<SaveData>()
                .map(|s| play_rows(&s))
                .unwrap_or(vec![0]);
            if rows.len() <= 1 {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.play_submenu = false;
                    menu.hardmode_selected = false;
                }
                emit_cue(world, &UiAction::MainMenuPlay);
                goto_state(world, AppState::Title);
            } else {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.play_submenu = true;
                    menu.play_cursor = 0;
                }
                emit_cue(world, &UiAction::MainMenuPlay);
            }
        }
        UiAction::PlaySubmenu(row) => match row {
            0 => {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.play_submenu = false;
                    menu.hardmode_selected = false;
                }
                emit_cue(world, &UiAction::PlaySubmenu(row));
                goto_state(world, AppState::Title);
            }
            3 => {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.play_submenu = false;
                    menu.hardmode_selected = true;
                }
                emit_cue(world, &UiAction::PlaySubmenu(row));
                goto_state(world, AppState::Title);
            }
            // GML `PlayButton/Other_10` verbatim: DAILY/WEEKLY with
            // `!can_daily/can_weekly` open the Leaderboards (daily /
            // weekly board) + `sndMenuScores` INSTEAD of starting a run;
            // CUSTOM loads the custom presets into
            // `MenuOptions(CustomMode)`. The port has neither backend,
            // so all three sting `sndNoSelect` (same feedback class as
            // the unavailable-button early-`exit`).
            _ => {
                emit_cue(world, &UiAction::PlaySubmenu(row));
                emit_denied(world);
            }
        },
        UiAction::ClosePlaySubmenu => {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.play_submenu = false;
                menu.play_cursor = 0;
            }
            emit_cue(world, &UiAction::ClosePlaySubmenu);
        }
        UiAction::AdvanceCredits => {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                let n = crate::render::credit_section_count();
                menu.credits_section = (menu.credits_section + 1) % n;
                menu.credits_t = 0.0;
                menu.credits_scroll = 0.0;
            }
            emit_cue(world, &UiAction::AdvanceCredits);
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
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.credits_section = 0;
                menu.credits_t = 0.0;
                menu.credits_scroll = 0.0;
            }
            emit_cue(world, &UiAction::OpenCredits);
        }
        UiAction::CloseOverlay => {
            // Bevy restores the saved locale here; headless applies the
            // language immediately, so there is nothing to restore.
            // Bevy never resets the settings page/stack here (that lives
            // only in the `SettingsBack` close path).
            world.init_resource::<crate::state::Paused>();
            let paused = world.resource::<crate::state::Paused>().0;
            world.init_resource::<OverlayMenu>();
            let mut overlay = world.resource_mut::<OverlayMenu>();
            match *overlay {
                OverlayMenu::Settings | OverlayMenu::Credits if paused => {
                    *overlay = OverlayMenu::Pause;
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
            crate::setup::setup_logo_room(world);
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
                // GML `scrOptionsMenu.gml:160-172` verbatim: push only when
                // actually changing category.
                if cur != cat {
                    menu.settings_page_stack.push(cur);
                }
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
                emit_click_back(world);
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
                crate::setup::setup_logo_room(world);
                world.init_resource::<crate::state::Paused>();
                world.resource_mut::<crate::state::Paused>().0 = false;
                emit_cue(world, &UiAction::ConfirmPause(kind));
                goto_state(world, AppState::MainMenu);
            } else {
                // Restart via loading (fresh run: drop stale highlight).
                world.init_resource::<crate::state::Paused>();
                world.resource_mut::<crate::state::Paused>().0 = false;
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.mutation_selected = None;
                }
                emit_cue(world, &UiAction::ConfirmPause(kind));
                goto_state(world, AppState::Loading);
            }
        }
        UiAction::SelectCharacter(i) => {
            let Some(race) = race_from_gml_id(i) else {
                return;
            };
            world.init_resource::<SaveData>();
            // GML `CharSelect/Draw_0:8` verbatim: `can = unlocked ||
            // weekly_run`. Weekly runs bypass the lock (the pod draws
            // unlocked); the click then proceeds to select/start.
            let weekly = world
                .get_resource::<MenuState>()
                .is_some_and(|m| m.weekly_run_menu);
            if !world.resource::<SaveData>().race_unlocked(race) && !weekly {
                // GML `Mouse_4:6-15` verbatim: locked picks sting
                // `sndNoSelect`; touch also raises the race unlock hint
                // on `Menu` for 90 steps (`alarm[11]`).
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.unlock_hint = unlock_hint_for_race(race);
                    menu.unlock_hint_pop = 2.0;
                    menu.unlock_hint_t = 90.0 / 30.0;
                }
                emit_denied(world);
                return;
            }
            let already = world
                .get_resource::<SelectedCharacter>()
                .is_some_and(|s| s.0 as usize == i);
            if already {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.title_go_visible = false;
                    menu.mutation_selected = None;
                }
                emit_cue(world, &UiAction::SelectCharacter(i));
                goto_state(world, AppState::Loading);
                return;
            }
            world.init_resource::<SelectedCharacter>();
            world.resource_mut::<SelectedCharacter>().0 = race;
            let pos = roster_position(world.get_resource::<SaveData>(), i);
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                if let Some(pos) = pos {
                    menu.title_cursor = pos;
                }
                // GML `CharSelect/Mouse_4`: `with GoButton if (!visible)` —
                // the GO reveal fires only on the first pick.
                if !menu.title_go_visible {
                    menu.title_go_visible = true;
                }

                // GML `scrCampfireMenuSelectionChange` verbatim anim half:
                // the picking player's portrait slides (180) and the name
                // text hides (2) before typing back in (ticked in
                // `tick_title_input`). Crown/skin/weapon sync lives in the
                // save stamps the run setup already reads, so only the
                // anim + panel gate apply here.
                menu.portrait_offsets[0] = 180.0;
                menu.textappear[0] = 2.0;
                // GML `if !scr_loadout_is_available_for_race loadout_open=false`.
                if !crate::render::loadout_available_for_race(race) {
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
                // GML skin pick verbatim: the portrait slides again.
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.portrait_offsets[0] = 180.0;
                }
                emit_sfx(world, skin_select_sfx(s));
                emit_cue(world, &UiAction::SelectSkin(s));
            } else {
                emit_denied(world);
            }
        }
        UiAction::ToggleLoadout => {
            // GML `scrMenuDrawLoadout:662-695` verbatim gate: no panel for
            // Random and races without a loadout (the trio); otherwise
            // Space/click flips `loadout_open`.
            world.init_resource::<SelectedCharacter>();
            world.init_resource::<SaveData>();
            let race = world.resource::<SelectedCharacter>().0;
            if !crate::render::loadout_available_for_race(race) {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.loadout_open = false;
                }
                emit_denied(world);
                return;
            }
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
            // GML loadout has no crown row for Random (mirrors SelectCrown).
            if race == RaceId::Random {
                return;
            }
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
            {
                let mut save = world.resource_mut::<SaveData>();
                // Bevy saves + clicks unconditionally (unknown keys only
                // log a warning there).
                let _ = apply_setting_toggle(&mut save, key);
            }
            mark_dirty(world);
            emit_cue(world, &action);
        }
        UiAction::SettingSlider { ref key, value } => {
            world.init_resource::<SaveData>();
            {
                let mut save = world.resource_mut::<SaveData>();
                // Bevy saves + plays `sndSliderLetGo` unconditionally
                // (unknown sliders only warn).
                let _ = apply_setting_slider(&mut save, key, value);
            }
            mark_dirty(world);
            emit_cue(world, &action);
        }
        UiAction::SettingCycle { ref key, dir } => {
            world.init_resource::<SaveData>();
            {
                let mut save = world.resource_mut::<SaveData>();
                // Bevy saves + clicks unconditionally (unknown keys warn).
                let _ = apply_setting_cycle(&mut save, key, dir);
            }
            mark_dirty(world);
            emit_cue(world, &action);
        }
        UiAction::SettingInput { ref key, ref value } => {
            world.init_resource::<SaveData>();
            {
                let mut save = world.resource_mut::<SaveData>();
                // Bevy saves + clicks unconditionally (unknown keys warn).
                let _ = apply_setting_input(&mut save, key, value);
            }
            mark_dirty(world);
            emit_cue(world, &action);
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
                save.total_wins = 0;
                save.total_deaths = 0;
                save.total_loops = 0;
                save.total_time_steps = 0;
                save.hard_runs = 0;
                save.win_streak_cur = 0;
                save.win_streak_best = 0;
                save.best_streak_race = 0;
                save.best_time_steps = 0;
                save.best_time_race = 0;
                save.best_run_kills = 0;
                save.best_run_race = 0;
                save.best_run_area = 0;
                save.best_run_sub = 0;
                save.best_run_loop = 0;
                save.hard_best_kills = 0;
                save.hard_best_race = 0;
                save.hard_best_area = 0;
                save.hard_best_sub = 0;
                save.hard_best_loop = 0;
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
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.credits_section = 0;
                menu.credits_t = 0.0;
                menu.credits_scroll = 0.0;
            }
            emit_cue(world, &UiAction::SettingViewCredits);
        }
        UiAction::SettingOpenSubcategory(cat) => {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                let cur = menu.settings_page;
                // Same GML `category != _category` gate as SettingsCategory.
                if cur != cat {
                    menu.settings_page_stack.push(cur);
                }
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
            tick_title_input(world, edge);
        }
        AppState::InGame => {
            tick_ingame_menu(world, edge);
        }
    }

    // GML `Credits/Create_0/Step_0/Other_11` cycler verbatim over sim
    // seconds: `timer` opens at 60 steps (2 s), then 180 steps (6 s) per
    // section; tall sections (`height > gui_h - 36`) set `largetext`,
    // grow `height += gui_h` and pan `scroll = height` down (`scroll`
    // ticks in `MenuState::credits_scroll`). Clicks (non-scroll touch)
    // force-advance. `AdvanceCredits` = the click arm.
    if world
        .get_resource::<OverlayMenu>()
        .is_some_and(|o| *o == OverlayMenu::Credits)
        && let Some(mut menu) = world.get_resource_mut::<MenuState>()
    {
        menu.credits_t += dt;
        let n = crate::render::credit_section_count();
        let rows = crate::render::CREDIT_SECTIONS[menu.credits_section % n].len() as f32;
        // Section text height in GUI px (one 12px line per row); the
        // first section opens after the 60-step intro, the rest after
        // 180 steps each.
        let tall = rows * 12.0 > 240.0 - 36.0;
        let limit = if menu.credits_section == 0 && menu.credits_t < 60.0 {
            60.0
        } else {
            180.0
        };
        if tall {
            let height = rows * 12.0 + 240.0;
            menu.credits_scroll = height - (menu.credits_t * 30.0).clamp(0.0, height);
        } else {
            menu.credits_scroll = 0.0;
        }
        if menu.credits_t >= limit / 30.0 {
            menu.credits_section = (menu.credits_section + 1) % n;
            menu.credits_t = 0.0;
            menu.credits_scroll = 0.0;
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
    // Escape over an open overlay steps back (Settings pops one level via
    // `SettingsBack`, Credits/Stats close; the InGame escape tick never
    // runs here).
    if edge.pause_pressed
        && world
            .get_resource::<OverlayMenu>()
            .is_some_and(|o| *o != OverlayMenu::None)
    {
        let is_settings = world
            .get_resource::<OverlayMenu>()
            .is_some_and(|o| *o == OverlayMenu::Settings);
        if is_settings {
            apply_menu_action(world, UiAction::SettingsBack);
        } else {
            apply_menu_action(world, UiAction::CloseOverlay);
        }
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
    if world
        .get_resource::<MenuState>()
        .is_some_and(|menu| menu.play_submenu)
    {
        if edge.pause_pressed {
            apply_menu_action(world, UiAction::ClosePlaySubmenu);
            return;
        }
        let n = world
            .get_resource::<SaveData>()
            .map(|s| play_rows(&s).len())
            .unwrap_or(1)
            .max(1);
        if nav_v != 0 {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.play_cursor =
                    (menu.play_cursor as i16 + nav_v as i16).rem_euclid(n as i16) as usize;
            }
            emit_sfx(world, hover_sfx());
        }
        if let Some(slot) = slot {
            let row = world
                .get_resource::<SaveData>()
                .map(|s| play_rows(&s))
                .unwrap_or(vec![0])
                .get(slot)
                .copied();
            if let Some(row) = row {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.play_cursor = slot;
                }
                apply_menu_action(world, UiAction::PlaySubmenu(row));
                return;
            }
        }
        if confirm {
            let row = world
                .get_resource::<MenuState>()
                .and_then(|menu| {
                    world
                        .get_resource::<SaveData>()
                        .map(|s| play_rows(&s))
                        .unwrap_or(vec![0])
                        .get(menu.play_cursor)
                        .copied()
                })
                .unwrap_or(0);
            apply_menu_action(world, UiAction::PlaySubmenu(row));
        }
        return;
    }
    if nav_v != 0 {
        let landed_available = world
            .get_resource::<MenuState>()
            .map(|menu| (menu.main_menu_cursor as i16 + nav_v as i16).rem_euclid(5) as usize)
            .is_some_and(|row| matches!(row, 0 | 2 | 3 | 4));
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            menu.main_menu_cursor =
                (menu.main_menu_cursor as i16 + nav_v as i16).rem_euclid(5) as usize;
        }
        if landed_available {
            emit_sfx(world, hover_sfx());
        }
    }

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

/// Title input routing (cursor nav + confirm + loadout back). Settings /
/// Credits opened over the campfire own the keyboard first (settings nav
/// + Escape-back, credits Escape-back); otherwise cursor nav + confirm +
/// loadout back.
fn tick_title_input(world: &mut World, edge: MenuEdge) {
    // Overlay-first: Escape steps back, settings arrows/enter drive the
    // hot rows. Credits has no rows (Back/Esc only).
    let overlay = world
        .get_resource::<OverlayMenu>()
        .copied()
        .unwrap_or(OverlayMenu::None);
    if overlay == OverlayMenu::Settings || overlay == OverlayMenu::Credits {
        if edge.pause_pressed {
            if overlay == OverlayMenu::Settings {
                apply_menu_action(world, UiAction::SettingsBack);
            } else {
                apply_menu_action(world, UiAction::CloseOverlay);
            }
            return;
        }
        if overlay == OverlayMenu::Credits {
            // Swallow title keys under Credits so Space/E can't toggle
            // panels or confirm pods behind it.
            let mut input = world.resource_mut::<NtInput>();
            let _ = input.take_cycle_weapon();
            let _ = input.take_weapon_slot();
            let _ = input.take_interact_pressed();
            let _ = input.take_spec_pressed();
            let _ = input.take_fire_pressed();
            let _ = input.take_menu_nav();
            return;
        }
        let (nav_v, nav_h, confirm) = {
            let mut input = world.resource_mut::<NtInput>();
            let (dv, dh) = input.take_menu_nav();
            // Swallow title-only pulses under settings so they can't leak
            // to pods/loadout behind the panel.
            let _ = input.take_cycle_weapon();
            let _ = input.take_weapon_slot();
            let _ = input.take_spec_pressed();
            let _ = input.take_fire_pressed();
            (dv, dh, input.take_interact_pressed())
        };
        tick_settings_nav(world, nav_v, nav_h, confirm);
        return;
    }
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
    // GML `scrMenuDrawLoadout` toggle law: Space (or the splat click)
    // flips `loadout_open` — but ONLY for races with a panel and outside
    // event runs (`scr_loadout_is_available_for_race`, `scrGameIsEventRun`
    // gate; the port has no event runs so that half is vacuous). Space
    // does NOT start the run; `SelectCharacter` (pod re-click) does.
    if fire {
        apply_menu_action(world, UiAction::ToggleLoadout);
    }
    if cycle != 0 {
        let len = visible_roster(world.get_resource::<SaveData>())
            .len()
            .max(1) as i16;
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            menu.title_cursor = (menu.title_cursor as i16 + cycle as i16).rem_euclid(len) as usize;
        }
    }
    if let Some(slot) = slot {
        match roster_position(world.get_resource::<SaveData>(), slot) {
            Some(pos) => {
                if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                    menu.title_cursor = pos;
                }
            }
            None => {
                emit_denied(world);
            }
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
        // GML `CharSelect/Mouse_4` verbatim: confirming the ALREADY
        // selected race starts the run immediately (`scrRunStart` via
        // `StartGame`); confirming another pod only re-selects (reveal
        // GO via `SelectCharacter`). The headless confirm (E/Enter) and
        // the pod click share this law: `SelectCharacter` starts when
        // `_pinst.race == _race`, so issuing it unconditionally is GML.
        let roster = visible_roster(world.get_resource::<SaveData>());
        let cursor = world
            .get_resource::<MenuState>()
            .map(|menu| menu.title_cursor)
            .unwrap_or(0);
        let gml = roster
            .get(cursor)
            .map(|r| *r as usize)
            .unwrap_or(RaceId::Fish as usize);
        apply_menu_action(world, UiAction::SelectCharacter(gml));
    }
    // GML `Menu/Other_11:14-37` verbatim anim tick (runs every step while
    // the campfire menu is up): loadout open frame, per-player text slide,
    // portrait slide state machine, splat 0→3, unlock-hint 90-step timer.
    tick_title_anim(world);
}

/// GML `Menu/Other_11:14-37` verbatim over sim steps (`timescale` = 1 per
/// 30 Hz tick): `loadout_frame` approaches open/closed, `textappear`
/// approaches 0, `portrait_offsets` walks 180→90→-2→0, `splatindex`
/// climbs 0→3 at 0.4, and the touch `unlock_hint` counts down from 90
/// steps (`alarm[11]`).
pub fn tick_title_anim(world: &mut World) {
    let dt_steps = world
        .get_resource::<repame_sim::SimTime>()
        .map(|t| (t.delta_secs * 30.0).max(0.0))
        .unwrap_or(1.0);
    let Some(mut menu) = world.get_resource_mut::<MenuState>() else {
        return;
    };
    // Loadout open frame: GML approaches `sprite_get_number-1` when open
    // else 0. The strip length is renderer-owned; the headless law uses 3
    // frames (closed 0 → open 3) so `loadout_frame >= 2` still means full
    // view like the bevy layer expects.
    let target = if menu.loadout_open { 3.0 } else { 0.0 };
    menu.loadout_frame = approach_f(menu.loadout_frame, target, dt_steps);
    for i in 0..4 {
        if menu.textappear[i] != 0.0 {
            menu.textappear[i] = approach_f(menu.textappear[i], 0.0, dt_steps);
        }
        if menu.portrait_offsets[i] != 0.0 {
            let amount = menu.portrait_offsets[i].min(180.0);
            if amount == -2.0 {
                menu.portrait_offsets[i] = 0.0;
            } else if amount == 90.0 {
                menu.portrait_offsets[i] = -2.0;
            } else {
                menu.portrait_offsets[i] = 90.0;
            }
        }
    }
    if menu.splatindex < 3.0 {
        menu.splatindex = (menu.splatindex + 0.4 * dt_steps).min(3.0);
    }
    if menu.unlock_hint_t > 0.0 {
        menu.unlock_hint_t = (menu.unlock_hint_t - dt_steps / 30.0).max(0.0);
        if menu.unlock_hint_t <= 0.0 {
            menu.unlock_hint.clear();
            menu.unlock_hint_pop = 0.0;
        } else if menu.unlock_hint_pop > 0.0 {
            menu.unlock_hint_pop = (menu.unlock_hint_pop - dt_steps).max(0.0);
        }
    }
}

fn approach_f(v: f32, target: f32, step: f32) -> f32 {
    if v < target {
        (v + step).min(target)
    } else if v > target {
        (v - step).max(target)
    } else {
        v
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
/// GML `approach` verbatim for the game-over anim: move `v` toward
/// `target` by `delta` without overshooting.
fn approach(v: f32, target: f32, delta: f32) -> f32 {
    if v < target {
        (v + delta).min(target)
    } else if v > target {
        (v - delta).max(target)
    } else {
        v
    }
}

fn tick_ingame_menu(world: &mut World, edge: MenuEdge) {
    // Offer mirror (bevy `sync_hud` order: mirror before input handling;
    // law shared with `tick_mutation_mirror` via `apply_mutation_mirror`
    // — lens are copied out first for the `World` borrow checker).
    {
        let (count, is_ultra) = if let Some(ultra) = world.get_resource::<PendingUltra>() {
            (ultra.choices.len(), true)
        } else if let Some(pending) = world.get_resource::<PendingMutation>() {
            (pending.choices.len(), false)
        } else {
            (0, false)
        };
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            apply_mutation_mirror(&mut menu, count, is_ultra);
        }
    }

    let game_over = world.get_resource::<Run>().is_some_and(|run| run.game_over);

    // Escape toggles pause (bevy `handle_pause_input`; transitions never
    // block headless — no `Transition` resource exists here). GML
    // `UberCont/Step_1` swallows the pause request while a generation
    // cover runs (`GenCont`/`LevCont` rooms: floor transition or
    // mutation/ultra offer).
    if edge.pause_pressed && !game_over {
        let generating = world
            .get_resource::<crate::comps_b::FloorTransition>()
            .is_some_and(|f| f.active)
            || world.get_resource::<PendingMutation>().is_some()
            || world.get_resource::<PendingUltra>().is_some();
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
            generating,
        );
        world.insert_resource(paused);
        world.insert_resource(overlay);
        world.insert_resource(pending);
        world.insert_resource(menu);
    }

    // Snapshot the game-over screen once per death (GML `GameOver/Create_0`
    // verbatim: `death_pos = 0`, `offsety = 128`, `splatimg = 0`, the two
    // `PauseButton`s at `appear = 3 + image`).
    let needs_capture = game_over
        && world
            .get_resource::<MenuState>()
            .is_some_and(|menu| menu.game_over.is_none());
    if needs_capture {
        let screen = capture_game_over(world);
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            menu.game_over = screen;
            menu.go_death_pos = 0.0;
            menu.go_offsety = 128.0;
            menu.go_splat = 0.0;
            menu.go_appear = 4.0;
        }
    }

    // Fresh run clears a stale snapshot (bevy rebuilds the panel per
    // death; headless keeps it until the next death).
    if !game_over
        && world
            .get_resource::<MenuState>()
            .is_some_and(|menu| menu.game_over.is_some())
    {
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            menu.game_over = None;
            menu.go_death_pos = 0.0;
            menu.go_offsety = 128.0;
            menu.go_splat = 0.0;
            menu.go_appear = 4.0;
        }
    }

    // GML `GameOver/Draw_0` anim verbatim (per-draw advances, here per
    // 30 Hz tick scaled by `dt * 30`): `death_pos` reveals one waypoint
    // per tick capped at the log length; `offsety` slides 128 -> 0 at
    // 32/tick; `splatimg` eases 0 -> 2 at 0.7/tick (the GML
    // `letterbox_frame >= 2` gate has no port counterpart, so it always
    // animates once dead); `appear` ticks -1/step on the buttons.
    if game_over {
        let dt = world
            .get_resource::<repame_sim::SimTime>()
            .map(|t| t.delta_secs)
            .unwrap_or(1.0 / 30.0);
        let steps = (dt * 30.0).max(0.0);
        let total = world
            .get_resource::<crate::comps_a::Run>()
            .map(|r| r.waypoints.len() as f32)
            .unwrap_or(0.0);
        if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
            if menu.go_death_pos < total {
                menu.go_death_pos = (menu.go_death_pos + steps).min(total);
            }
            if menu.go_offsety > 0.0 {
                menu.go_offsety = approach(menu.go_offsety, 0.0, 32.0 * steps);
            }
            menu.go_splat = approach(menu.go_splat, 2.0, 0.7 * steps);
            if menu.go_appear > 0.0 {
                menu.go_appear = (menu.go_appear - steps).max(0.0);
            }
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
        let menu_open = game_over
            || world
                .get_resource::<MenuState>()
                .is_some_and(|menu| menu.mutation_count > 0)
            || world.get_resource::<PendingMutation>().is_some()
            || world.get_resource::<PendingUltra>().is_some()
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
        // Bevy `handle_death_restart` (KeyR) + game-over click in `lib.rs`
        // (full-panel `QuitToTitle`). Keyboard interact/Enter never quits
        // here — bevy has no such path.
        if edge.restart_pressed {
            if let Some(mut menu) = world.get_resource_mut::<MenuState>() {
                menu.title_go_visible = false;
                menu.mutation_selected = None;
            }
            goto_state(world, AppState::Loading);
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
