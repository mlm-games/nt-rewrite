//! Mechanical port of pure-data items from nt-recreated-bevy (keep semantics byte-identical).
//! Sources:
//! - `src/save.rs`: `SAVE_VERSION`, `SaveData`, `SettingsData` (+ `default_*` helpers, `Default` impls)
//! - `src/game/components.rs`: `RaceLoadout` (imported by `src/save.rs` as
//!   `crate::game::components::RaceLoadout`; canonical definition, not an invention)
//! - `src/game/generated/unlocks.rs`: `try_unlock_race`, `check_kill_unlocks`,
//!   `try_unlock_skin`, `try_unlock_skeleton` (complete bodies, verbatim logic)
//! - `src/game/content.rs`: `PassiveKind` (verbatim companion required by `CharacterDef`,
//!   same source file, not an invention), `CharacterDef`, `character_def`
//! Transforms: `RaceId` -> `crate::data::RaceId`, `WeaponId`/`PLAYABLE_RACES` -> `crate::data::*`,
//! `EnemyKind` -> `crate::data::EnemyKind`, `AbilityKind` -> `crate::data::AbilityKind`,
//! `Color::srgb(r,g,b)` -> `[r, g, b, 1.0f32]`. Field names/order, match arms, comments preserved.
//! `SaveData` keeps `#[derive(Resource)]`: it becomes a sim resource.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

use crate::comps_a::{Health, Inventory, Player, RaceState, Run};
use crate::data::{AreaId, CrownKind, MutationId, RaceId, SkinLetter, WeaponId};
use crate::keymap::KeyBindings;

pub const SAVE_VERSION: u32 = 5;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RaceLoadout {
    pub unlocked: bool,
    pub unlocked_skins: [bool; 4],

    pub preferred_skin: u8,
    pub stored_weapon: crate::data::WeaponId,
    pub start_weapon: crate::data::WeaponId,
    pub start_crown: u8,
}

#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct SaveData {
    // Old saves fill defaults via serde.
    #[serde(default)]
    pub version: u32,
    pub high_score: u32,
    #[serde(default)]
    pub best_floor: u32,
    #[serde(default)]
    pub total_runs: u32,
    /// GML `save game.tutorial` completion bit (`TutCont/Alarm_0`
    /// writes `false` before spawning the exit portal). Separate from
    /// `settings.show_tutorial` (the instruction-bar display toggle).
    #[serde(default)]
    pub tutorial_done: bool,
    #[serde(default)]
    pub total_kills: u32,
    /// GML `ctot_wins` sum (stats DAILY/HARD/STREAK denominators).
    #[serde(default)]
    pub total_wins: u32,
    /// GML `ctot_dead` sum.
    #[serde(default)]
    pub total_deaths: u32,
    /// GML `ctot_loop` sum.
    #[serde(default)]
    pub total_loops: u32,
    /// GML `tot_time` in 30Hz steps (stats TOTAL time).
    #[serde(default)]
    pub total_time_steps: u64,
    /// GML `ctot_hard` sum (hardmode runs).
    #[serde(default)]
    pub hard_runs: u32,
    /// GML win-streak state (`cbst_strk` needs the live streak).
    #[serde(default)]
    pub win_streak_cur: u32,
    #[serde(default)]
    pub win_streak_best: u32,
    /// GML `cbst_strk` race, as a gml id.
    #[serde(default)]
    pub best_streak_race: u8,
    /// GML `cbst_fast` (0 = no wins yet): fastest win in 30Hz steps.
    #[serde(default)]
    pub best_time_steps: u32,
    /// GML fastest-win race, as a gml id.
    #[serde(default)]
    pub best_time_race: u8,
    /// GML `cbst_kill` best run (kills + race + map).
    #[serde(default)]
    pub best_run_kills: u32,
    #[serde(default)]
    pub best_run_race: u8,
    #[serde(default)]
    pub best_run_area: i32,
    #[serde(default)]
    pub best_run_sub: u32,
    #[serde(default)]
    pub best_run_loop: u32,
    /// GML `hbst_*` best hardmode run (kills + race + map), written by
    /// `scrPlayerUpdateBestRunStats` only when `scrGameIsHardmode()`.
    /// Global aggregate like `best_run_*` (GML keeps per-race arrays;
    /// the stats screen only shows the global max). New in v5: old
    /// saves fill zeros via serde defaults, then `tick_sanitize_save`
    /// stamps the version.
    #[serde(default)]
    pub hard_best_kills: u32,
    #[serde(default)]
    pub hard_best_race: u8,
    #[serde(default)]
    pub hard_best_area: i32,
    #[serde(default)]
    pub hard_best_sub: u32,
    #[serde(default)]
    pub hard_best_loop: u32,
    /// GML `etc.hard`: hardmode unlocked (loop 2 reached).
    #[serde(default)]
    pub hardmode_unlocked: bool,
    #[serde(default)]
    pub unlocked_characters: Vec<String>,
    #[serde(default)]
    pub races: BTreeMap<crate::data::RaceId, RaceLoadout>,

    // Crowns 0-1 start open, rest unlock in-run.
    #[serde(default)]
    pub crown_got: BTreeMap<crate::data::RaceId, [bool; 14]>,
    /// Races that completed a loop (GML `UberCont.ctot_loop` parity:
    /// Fish/B needs a loop as every non-hidden character).
    #[serde(default)]
    pub race_looped: BTreeMap<crate::data::RaceId, bool>,
    #[serde(default)]
    pub achievements: BTreeMap<String, bool>,
    #[serde(default)]
    pub unlocked_cheats: bool,
    #[serde(default)]
    pub settings: SettingsData,
    #[serde(default)]
    pub key_bindings: KeyBindings,
}

/// GML `scripts/scrAchievements` registry: 59 entries (0-58) as
/// `(UPPER_NAME, UPPER_TEXT, hidden, type)`. Types: 0 char/skin, 1
/// boss/gaming, 2 meta.
pub fn achievement_def(id: u8) -> Option<(&'static str, &'static str, bool, u8)> {
    Some(match id {
        0 => ("MELTING UNLOCKED", "DIE", false, 0),
        1 => ("EYES UNLOCKED", "REACH 2-1", false, 0),
        2 => ("PLANT UNLOCKED", "REACH 3-1", false, 0),
        3 => ("Y.V. UNLOCKED", "REACH 3-?", false, 0),
        4 => ("STEROIDS UNLOCKED", "REACH 6-1", false, 0),
        5 => ("ROBOT UNLOCKED", "REACH 5-1", false, 0),
        6 => ("CHICKEN UNLOCKED", "REACH 5-?", false, 0),
        7 => ("REBEL UNLOCKED", "LOOP PAST\nTHE NUCLEAR THRONE", true, 0),
        8 => ("HORROR UNLOCKED", "DEFEAT HORROR", true, 0),
        9 => ("ROGUE UNLOCKED", "REACH THE NUCLEAR THRONE", false, 0),
        10 => ("FISH CAN ROLL", "LOOP AS EVERY CHARACTER", true, 0),
        11 => ("CRYSTAL CAN SHIELD", "REACH 4-? AS CRYSTAL", false, 0),
        12 => (
            "EVERYTHING HURTS",
            "AS MELTING, REACH THE NUCLEAR THRONE\nWITHOUT RHINO SKIN AND STRONG SPIRIT",
            false,
            0,
        ),
        13 => ("MMMMMMHMMM!", "REACH 2-? AS EYES", false, 0),
        14 => (
            "BLOOD BLOOD BLOOD",
            "REACH THE NUCLEAR THRONE\nIN UNDER 10 MINUTES AS PLANT",
            false,
            0,
        ),
        15 => (
            "VERIFIED",
            "UNLOCK A GOLDEN WEAPON\nFOR EVERY CHARACTER",
            false,
            0,
        ),
        16 => ("SCIENCE", "DEFEAT THE TECHNOMANCER AS STEROIDS", true, 0),
        17 => ("6E 69 63 65", "EAT A HYPER WEAPON AS ROBOT", true, 0),
        18 => (
            "WAY OF THE CHICKEN",
            "REACH 2-1 ON HARD MODE AS CHICKEN",
            true,
            0,
        ),
        19 => ("FORGET THE OLD DAYS", "DEFEAT MOM AS REBEL", true, 0),
        20 => ("THRILLER", "DEFEAT HYPER CRYSTAL AS HORROR", true, 0),
        21 => ("NEVER LOOK BACK", "DEFEAT CAPTAIN AS ROGUE", true, 0),
        22 => ("CROWN LIFE", "UNLOCK A CROWN AS ANY CHARACTER", false, 0),
        23 => ("ULTRA TIME", "REACH LEVEL ULTRA AS ANY CHARACTER", false, 0),
        24 => (
            "GOOD FIND",
            "UNLOCK A GOLDEN WEAPON\nAS ANY CHARACTER",
            false,
            0,
        ),
        25 => (
            "GOOD RIDDANCE",
            "UNLOCK A GOLDEN DISC GUN\nOR GOLDEN NUKE LAUNCHER",
            true,
            0,
        ),
        26 => ("NOT BAD", "REACH 7-3 IN DAILY RUN", false, 2),
        27 => ("UNSTOPPABLE", "REACH LEVEL ULTRA AS SKELETON", true, 0),
        28 => ("FROG ZONE", "PLAY AS FROG", true, 0),
        29 => (
            "IMPOSSIBLE",
            "SIT ON THE NUCLEAR THRONE\nAS HEADLESS CHICKEN",
            true,
            0,
        ),
        30 => (
            "SINCERE APOLOGIES",
            "KILL YOURSELF WITH A DISC GUN",
            true,
            0,
        ),
        31 => ("BANDIT STOPPER", "DEFEAT BIG BANDIT", false, 1),
        32 => ("DOG OWNER", "DEFEAT BIG DOG", false, 1),
        33 => ("HUNTER KILLER", "DEFEAT LIL HUNTER", false, 1),
        34 => ("THRONE SITTER", "DEFEAT THE NUCLEAR THRONE", true, 1),
        35 => ("ADVANCED SITTER", "DEFEAT THRONE II", true, 1),
        36 => ("FROG SLAYER", "DEFEAT MOM", true, 1),
        37 => ("CRYSTAL SMASHER", "DEFEAT HYPER CRYSTAL", true, 1),
        38 => ("TECHNO KILLER", "DEFEAT TECHNOMANCER", true, 1),
        39 => (
            "VAULT RAIDER",
            "UNLOCK ALL CROWNS\nAS ANY CHARACTER",
            false,
            1,
        ),
        40 => ("GO HARD", "UNLOCK HARD MODE", true, 1),
        41 => ("THE STRUGGLE CONTINUES", "LOOP THE GAME", false, 2),
        42 => ("THE STRUGGLE IS OVER", "DEFEAT CAPTAIN", true, 2),
        43 => ("ULTRA MUTANT", "GET 100% OF THE UNLOCKS", false, 2),
        44 => ("GUNZ GOD", "REACH Y.V.'S MANSION.", false, 0),
        45 => (
            "ROUND AND HANDSOME",
            "CARRY 3 GOLDEN WEAPONS AS CUZ.",
            true,
            0,
        ),
        46 => ("RETIREMENT", "UNLOCK ALL B-SKINS.", true, 0),
        47 => (
            "CRYSTAL CAN HANDLE THIS",
            "SURVIVE OVER 100 DAMAGE AS CRYSTAL.",
            true,
            0,
        ),
        48 => (
            "HHMMMM!",
            "REACH THE NUCLEAR THRONE\nWITHOUT FIRING A SHOT AS EYES.",
            true,
            0,
        ),
        49 => ("MOLTEN", "HAVE 12 MUTATIONS AS MELTING.", true, 0),
        50 => ("KILL KILL KILL", "BLOOD BLOOD BLOOD.", true, 0),
        51 => ("THANKS GUN GOD", "DEFEAT A GUN GOD.", true, 0),
        52 => (
            "APPRECIATE REVOLVERS",
            "REACH THE NUCLEAR THRONE\nWITHOUT PICKING UP ANY WEAPONS\nAS STEROIDS.",
            true,
            0,
        ),
        53 => (
            "63 72 75 6E 63 68 79",
            "EAT THE RUSTY REVOLVER AS ROBOT.",
            true,
            0,
        ),
        54 => (
            "AMATEUR HOUR IS OVER",
            "DEFEAT EVERY BOSS\nWITH THE BLACK SWORD.",
            true,
            0,
        ),
        55 => ("BIGGEST BANDIT", "DEFEAT 1000 BANDITS IN TOTAL.", true, 0),
        56 => (
            "DRAMA",
            "REACH THE I.D.P.D. HEADQUARTERS\nWITH 3 OR LESS MUTATIONS AS HORROR.",
            true,
            0,
        ),
        57 => ("FORGIVENESS", "DON'T DEFEAT LIL HUNTER AS ROGUE.", true, 0),
        58 => ("STRAPPED", "CARRY 6 CURSED WEAPONS AS CUZ.", true, 0),
        _ => return None,
    })
}

/// GML `scrAchievementUnlock`: no-op when already held (custom-mode
/// gate lives with callers: the port has no custom runs). Returns true
/// on a fresh unlock; completing the set auto-awards ULTRA MUTANT (43).
pub fn unlock_achievement(save: &mut SaveData, id: u8) -> bool {
    if achievement_def(id).is_none() {
        return false;
    }
    let key = id.to_string();
    if save.achievements.get(&key).copied().unwrap_or(false) {
        return false;
    }
    save.achievements.insert(key, true);
    if id != 43
        && (0..=58).all(|i| {
            save.achievements
                .get(&i.to_string())
                .copied()
                .unwrap_or(false)
        })
    {
        save.achievements.insert("43".to_string(), true);
    }
    true
}

/// Character unlock (0-9) + Cuz (44) achievement ids by race.
pub fn achievement_for_race(race: crate::data::RaceId) -> Option<u8> {
    use crate::data::RaceId;
    Some(match race {
        RaceId::Melting => 0,
        RaceId::Eyes => 1,
        RaceId::Plant => 2,
        // Y.V. is boss-only in the port (no playable race).
        RaceId::Steroids => 4,
        RaceId::Robot => 5,
        RaceId::Chicken => 6,
        RaceId::Rebel => 7,
        RaceId::Horror => 8,
        RaceId::Rogue => 9,
        RaceId::Cuz => 44,
        _ => return None,
    })
}

/// B-skin (10-21, Cuz 45) / C-skin (46-57, Cuz 58) achievement ids.
pub fn achievement_for_skin(race: crate::data::RaceId, skin: usize) -> Option<u8> {
    use crate::data::RaceId;
    let order = match race {
        RaceId::Fish => 0,
        RaceId::Crystal => 1,
        RaceId::Melting => 2,
        RaceId::Eyes => 3,
        RaceId::Plant => 4,
        // Y.V. is boss-only in the port (no playable race/skin).
        RaceId::Steroids => 6,
        RaceId::Robot => 7,
        RaceId::Chicken => 8,
        RaceId::Rebel => 9,
        RaceId::Horror => 10,
        RaceId::Rogue => 11,
        _ => return None,
    };
    match skin {
        1 => Some(10 + order),
        2 => Some(46 + order),
        _ => None,
    }
}

/// Boss-kill (31-38, 42, 51) achievement ids by kind.
pub fn achievement_for_boss(kind: crate::data::EnemyKind) -> Option<u8> {
    use crate::data::EnemyKind;
    Some(match kind {
        EnemyKind::BigBandit | EnemyKind::BigBanditLoop => 31,
        EnemyKind::BigDog | EnemyKind::BigDogLoop => 32,
        EnemyKind::LilHunter | EnemyKind::LilHunterLoop => 33,
        EnemyKind::Throne => 34,
        EnemyKind::ThroneII => 35,
        EnemyKind::Mom => 36,
        EnemyKind::Hyper => 37,
        EnemyKind::Technomancer => 38,
        EnemyKind::Captain => 42,
        EnemyKind::YvBoss => 51,
        _ => return None,
    })
}

// Save-phase extensions (ported from `src/save.rs` gameplay methods):
// `race_unlocked` (with the `unlocked_characters` name fallback),
// `race_loadout`, `sanitize_loadouts`, `crown_row`, `crown_unlocked`,
// `any_crown_unlocked`, `unlock_crown`, `crown_port_to_gml`,
// `crown_gml_to_port`, plus std::fs JSON IO (`save_file_path`,
// `serialize_save`, `parse_save`, `store_save_to_file`,
// `load_save_from_file`, `load_or_default`) and the full
// `src/game/skin_unlocks.rs` surface (`check_area_skins`,
// `tick_area_skins`, `check_robot_weapon_skins`, `tick_robot_skins`,
// `CrystalDamageTaken`, `tick_crystal_damage`, `tick_global_skins`).
// Transforms: `RaceId` -> `crate::data::RaceId`, `WeaponId` ->
// `crate::data::WeaponId`, `AreaId` -> `crate::data::AreaId`,
// `SkinLetter` -> `crate::data::SkinLetter`, `WeaponId` ammo names via
// `crate::weapon_runtime::weapon_meta`; `ResMut<SaveData>` systems kept.

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct SettingsData {
    pub master_volume: f32,
    pub sfx_volume: f32,
    pub music_volume: f32,
    pub ambience_volume: f32,
    pub language: String,
    #[serde(default = "default_volume_3dsound")]
    pub volume_3dsound: bool,
    #[serde(default = "default_f")]
    pub screenshake: f32,
    #[serde(default = "default_f")]
    pub freezeframes: f32,
    #[serde(default)]
    pub bloom: bool,
    #[serde(default = "default_true")]
    pub particles: bool,
    #[serde(default = "default_true")]
    pub show_hud: bool,
    #[serde(default = "default_true")]
    pub show_timer: bool,
    #[serde(default = "default_true")]
    pub show_area: bool,
    #[serde(default = "default_true")]
    pub boss_intros: bool,
    #[serde(default = "default_true")]
    pub auto_pause: bool,
    #[serde(default = "default_true")]
    pub pause_button: bool,
    #[serde(default)]
    pub achievements_popup: bool,
    #[serde(default = "default_true")]
    pub vsync: bool,
    #[serde(default = "default_true")]
    pub fullscreen: bool,
    #[serde(default)]
    pub widescreen: bool,
    #[serde(default)]
    pub crosshair: u8,
    #[serde(default)]
    pub sideart: u8,
    #[serde(default = "default_pixel_mode")]
    pub pixel_mode: u8,
    #[serde(default)]
    pub gamepad_enabled: bool,
    #[serde(default)]
    pub gamepad_type: u8,
    #[serde(default)]
    pub aim_assist: bool,
    #[serde(default)]
    pub auto_aim: bool,
    #[serde(default)]
    pub volume_controls: bool,
    #[serde(default)]
    pub split_fire: bool,
    #[serde(default)]
    pub fixed_sight: bool,
    #[serde(default = "default_controls_scale")]
    pub controls_scale: f32,
    #[serde(default = "default_true")]
    pub show_tutorial: bool,
    #[serde(default)]
    pub player_color_hex: String,
    #[serde(default)]
    pub profile_name: String,
    #[serde(default)]
    pub cprefs_eyes: bool,
    #[serde(default)]
    pub cprefs_melting: bool,
    #[serde(default)]
    pub cprefs_plant: bool,
    #[serde(default)]
    pub cprefs_yv: bool,
    #[serde(default)]
    pub cprefs_steroids: bool,
    #[serde(default)]
    pub cprefs_horror: bool,
    #[serde(default)]
    pub cprefs_rogue: bool,
    #[serde(default)]
    pub cprefs_skeleton: bool,
}

fn default_volume_3dsound() -> bool {
    true
}
fn default_f() -> f32 {
    1.0
}
fn default_true() -> bool {
    true
}
fn default_pixel_mode() -> u8 {
    1
}
fn default_controls_scale() -> f32 {
    0.5
}

impl Default for SettingsData {
    fn default() -> Self {
        Self {
            master_volume: 1.0,
            sfx_volume: 1.0,
            music_volume: 0.8,
            ambience_volume: 1.0,
            language: "en".to_string(),
            volume_3dsound: true,
            screenshake: 1.0,
            freezeframes: 1.0,
            bloom: true,
            particles: true,
            show_hud: true,
            show_timer: false,
            show_area: true,
            boss_intros: true,
            auto_pause: true,
            pause_button: true,
            achievements_popup: true,
            vsync: false,
            fullscreen: true,
            widescreen: false,
            crosshair: 0,
            sideart: 0,
            pixel_mode: 1,
            gamepad_enabled: false,
            gamepad_type: 0,
            aim_assist: false,
            auto_aim: false,
            volume_controls: false,
            split_fire: false,
            fixed_sight: false,
            controls_scale: 0.5,
            show_tutorial: true,
            player_color_hex: String::new(),
            profile_name: String::new(),
            cprefs_eyes: true,
            cprefs_melting: true,
            cprefs_plant: false,
            cprefs_yv: true,
            cprefs_steroids: true,
            cprefs_horror: true,
            cprefs_rogue: true,
            cprefs_skeleton: false,
        }
    }
}

impl SettingsData {
    /// GML `UberCont.opt_healthcol` (`scrOptionsUpdate:96-100`): the
    /// player color, or default red `make_color_rgb(252, 56, 0)` when
    /// unset. Tints the `sprHealthFill` frames (whose strip columns
    /// already carry the per-frame shades).
    pub fn healthcol_rgba(&self) -> [f32; 4] {
        let hex = self.player_color_hex.trim_start_matches('#');
        if hex.len() == 6 {
            if let (Ok(r), Ok(g), Ok(b)) = (
                u8::from_str_radix(&hex[0..2], 16),
                u8::from_str_radix(&hex[2..4], 16),
                u8::from_str_radix(&hex[4..6], 16),
            ) {
                return [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0];
            }
        }
        [252.0 / 255.0, 56.0 / 255.0, 0.0, 1.0]
    }

    /// GML `UberCont.opt_cursorcol` (`scrOptionsUpdate:92-104`): the
    /// player color, or `c_white` when unset (GML leaves the 0 black
    /// only on `opt_healthcol`). Tints the `UberCont/Draw_75` raw
    /// cursor the same way `healthcol_rgba` tints the health bar.
    pub fn cursorcol_rgba(&self) -> [f32; 4] {
        let hex = self.player_color_hex.trim_start_matches('#');
        if hex.len() == 6 {
            if let (Ok(r), Ok(g), Ok(b)) = (
                u8::from_str_radix(&hex[0..2], 16),
                u8::from_str_radix(&hex[2..4], 16),
                u8::from_str_radix(&hex[4..6], 16),
            ) {
                if r != 0 || g != 0 || b != 0 {
                    return [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0];
                }
            }
        }
        [1.0, 1.0, 1.0, 1.0]
    }
}

impl Default for SaveData {
    fn default() -> Self {
        let mut races = BTreeMap::new();
        for &r in crate::data::PLAYABLE_RACES.iter() {
            races.insert(
                r,
                RaceLoadout {
                    // GML `scrInit.gml:162-163` verbatim: fresh saves hold
                    // Random, Fish and Crystal (`cgot` for all three).
                    unlocked: matches!(
                        r,
                        crate::data::RaceId::Fish | crate::data::RaceId::Crystal
                    ),
                    unlocked_skins: [true, false, false, false],
                    preferred_skin: 0,
                    stored_weapon: crate::data::WeaponId(0),
                    start_weapon: crate::data::WeaponId(0),
                    start_crown: 0,
                },
            );
        }
        Self {
            version: SAVE_VERSION,
            high_score: 0,
            best_floor: 0,
            total_runs: 0,
            total_kills: 0,
            total_wins: 0,
            total_deaths: 0,
            total_loops: 0,
            total_time_steps: 0,
            hard_runs: 0,
            win_streak_cur: 0,
            win_streak_best: 0,
            best_streak_race: 0,
            best_time_steps: 0,
            best_time_race: 0,
            best_run_kills: 0,
            best_run_race: 0,
            best_run_area: 0,
            best_run_sub: 0,
            best_run_loop: 0,
            hard_best_kills: 0,
            hard_best_race: 0,
            hard_best_area: 0,
            hard_best_sub: 0,
            hard_best_loop: 0,
            unlocked_characters: vec!["Fish".to_string(), "Crystal".to_string()],
            races,
            crown_got: BTreeMap::new(),
            race_looped: BTreeMap::new(),
            achievements: BTreeMap::new(),
            unlocked_cheats: false,
            hardmode_unlocked: false,
            tutorial_done: false,
            settings: SettingsData::default(),
            key_bindings: KeyBindings::default(),
        }
    }
}

// NOTE: `is_race_unlocked` / `check_progress_unlocks` above are the
// `src/game/generated/unlocks.rs` gameplay slice that already lived here;
// the `skin_unlocks.rs` half now lives in the section below.
// Still deferred: `Versioned` (no game_utils headless).

// TODO(port): try_unlock_race calls is_race_unlocked (src/game/generated/unlocks.rs) and
// SaveData::race_loadout_mut (src/save.rs); neither is part of this slice. Body kept verbatim.
pub fn try_unlock_race(save: &mut SaveData, race: crate::data::RaceId) -> bool {
    try_unlock_race_with_menu(save, race, None)
}

/// Race unlock with the GML `UnlockScreen` queue half: on a fresh
/// unlock, queues the race popup (GML `scrRaceUnlock` calls
/// `scrUnlockScreenCreate`). Pass the live `MenuState` when the caller
/// owns it; `None` keeps the save-only law (headless save paths).
pub fn try_unlock_race_with_menu(
    save: &mut SaveData,
    race: crate::data::RaceId,
    menu: Option<&mut crate::state::menus::MenuState>,
) -> bool {
    if is_race_unlocked(save, race) {
        return false;
    }
    let name = character_def(race).name.to_string();
    save.race_loadout_mut(race).unlocked = true;
    if !save
        .unlocked_characters
        .iter()
        .any(|s| s.eq_ignore_ascii_case(&name))
    {
        save.unlocked_characters.push(name);
    }
    if let Some(id) = achievement_for_race(race) {
        unlock_achievement(save, id);
    }
    if let Some(menu) = menu {
        crate::state::menus::push_race_unlock(menu, race);
    }
    true
}

pub fn check_kill_unlocks(
    save: &mut SaveData,
    kind: crate::data::EnemyKind,
    race: crate::data::RaceId,
    menu: Option<&mut crate::state::menus::MenuState>,
) -> CheckKillUnlocks {
    use crate::data::EnemyKind;
    let mut races = Vec::new();
    let mut skins: Vec<(crate::data::RaceId, u8)> = Vec::new();
    let mut award = |save: &mut SaveData,
                     menu: &mut Option<&mut crate::state::menus::MenuState>,
                     r: crate::data::RaceId| {
        if try_unlock_race_with_menu(save, r, menu.as_deref_mut()) {
            races.push(r);
        }
    };
    let mut award_skin = |save: &mut SaveData,
                          menu: &mut Option<&mut crate::state::menus::MenuState>,
                          r: crate::data::RaceId,
                          s: usize| {
        if try_unlock_skin_with_menu(save, r, s, menu.as_deref_mut())
            && let Ok(s) = u8::try_from(s)
        {
            skins.push((r, s));
        }
    };
    let mut menu = menu;

    match kind {
        EnemyKind::BigDog | EnemyKind::BigDogLoop => {
            award(save, &mut menu, crate::data::RaceId::BigDog);
        }
        EnemyKind::Mom => {
            award(save, &mut menu, crate::data::RaceId::Frog);
        }
        // GML hatches `HostileHorror` from a starved rad chest and
        // unlocks Horror on the encounter (kill-gated here: every
        // spawned horror that dies counts).
        EnemyKind::HostileHorror => {
            award(save, &mut menu, crate::data::RaceId::Horror);
        }
        _ => {}
    }

    match kind {
        EnemyKind::FrogQueen if race == crate::data::RaceId::Rebel => {
            award_skin(save, &mut menu, crate::data::RaceId::Rebel, 1);
        }
        EnemyKind::Hyper if race == crate::data::RaceId::Horror => {
            award_skin(save, &mut menu, crate::data::RaceId::Horror, 1);
        }
        EnemyKind::Technomancer if race == crate::data::RaceId::Steroids => {
            award_skin(save, &mut menu, crate::data::RaceId::Steroids, 1);
        }
        EnemyKind::Captain => {
            if race == crate::data::RaceId::Rogue {
                award_skin(save, &mut menu, crate::data::RaceId::Rogue, 1);
            }
            if race == crate::data::RaceId::Cuz {
                award_skin(save, &mut menu, crate::data::RaceId::Venuz, 2);
            }
        }
        EnemyKind::Bandit if race == crate::data::RaceId::Rebel => {
            award_skin(save, &mut menu, crate::data::RaceId::Rebel, 2);
        }
        EnemyKind::LilHunter | EnemyKind::LilHunterLoop if race == crate::data::RaceId::Rogue => {
            award_skin(save, &mut menu, crate::data::RaceId::Rogue, 2);
        }
        EnemyKind::Throne | EnemyKind::ThroneII => {
            if race == crate::data::RaceId::Melting {
                award_skin(save, &mut menu, crate::data::RaceId::Melting, 1);
                award_skin(save, &mut menu, crate::data::RaceId::Melting, 2);
            }
            if race == crate::data::RaceId::Plant {
                award_skin(save, &mut menu, crate::data::RaceId::Plant, 1);
                award_skin(save, &mut menu, crate::data::RaceId::Plant, 2);
            }
            if race == crate::data::RaceId::Eyes {
                award_skin(save, &mut menu, crate::data::RaceId::Eyes, 2);
            }
            if race == crate::data::RaceId::Steroids {
                award_skin(save, &mut menu, crate::data::RaceId::Steroids, 2);
            }
        }
        _ => {}
    };

    CheckKillUnlocks { races, skins }
}

/// Fresh unlocks from [`check_kill_unlocks`]: newly-unlocked races
/// (toast + popup) plus newly-unlocked `(race, skin-index)` pairs
/// (popup only — GML skins surface as `UnlockScreen` panels, and the
/// port previously dropped them on the floor).
pub struct CheckKillUnlocks {
    pub races: Vec<crate::data::RaceId>,
    pub skins: Vec<(crate::data::RaceId, u8)>,
}

fn try_unlock_skin(save: &mut SaveData, race: crate::data::RaceId, skin: usize) -> bool {
    try_unlock_skin_with_menu(save, race, skin, None)
}

fn try_unlock_skin_with_menu(
    save: &mut SaveData,
    race: crate::data::RaceId,
    skin: usize,
    menu: Option<&mut crate::state::menus::MenuState>,
) -> bool {
    let Some(lo) = save.races.get_mut(&race) else {
        return false;
    };
    if !lo.unlocked || lo.unlocked_skins.get(skin).copied() != Some(false) {
        return false;
    }
    lo.unlocked_skins[skin] = true;
    if let Some(menu) = menu
        && let Ok(skin_u8) = u8::try_from(skin)
    {
        crate::state::menus::push_skin_unlock(menu, race, skin_u8);
    }
    true
}

/// GML `scrUnlocksThroneDefeat` (via SitDown): Melting-B with neither
/// Rhino Skin nor Strong Spirit, Plant-B on a sub-10-minute run
/// (`tottimer < 18000` steps), Eyes-C pacifist (`!hasfiredshots`),
/// Steroids-C without picking up a gun. Returns `(race, skin)` pairs
/// for toasts.
pub fn throne_defeat_skins(
    save: &mut SaveData,
    race: crate::data::RaceId,
    mutations: &[crate::data::MutationId],
    tottimer: u32,
    shots_fired: u32,
    weapons_picked: u32,
) -> Vec<(crate::data::RaceId, crate::data::SkinLetter)> {
    use crate::data::{MutationId, RaceId, SkinLetter};
    let mut got = Vec::new();
    let mut award = |save: &mut SaveData, r: RaceId, s: SkinLetter| {
        if try_unlock_skin(save, r, s as usize) {
            got.push((r, s));
        }
    };
    match race {
        RaceId::Melting
            if !mutations.contains(&MutationId::RhinoSkin)
                && !mutations.contains(&MutationId::StrongSpirit) =>
        {
            award(save, RaceId::Melting, SkinLetter::B);
        }
        RaceId::Plant if tottimer < 18000 => {
            award(save, RaceId::Plant, SkinLetter::B);
        }
        RaceId::Eyes if shots_fired == 0 => {
            award(save, RaceId::Eyes, SkinLetter::C);
        }
        RaceId::Steroids if weapons_picked == 0 => {
            award(save, RaceId::Steroids, SkinLetter::C);
        }
        _ => {}
    }
    got
}

pub fn try_unlock_skeleton(save: &mut SaveData) -> bool {
    try_unlock_race(save, crate::data::RaceId::Skeleton)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PassiveKind {
    None,
    ShieldOnHit,
    ChainExplosions,
    FastReload,
    Headless,
    FreeAmmo,
}

pub struct CharacterDef {
    pub name: &'static str,
    pub color: [f32; 4],
    pub max_hp: i32,
    pub speed_mult: f32,
    pub pickup_range: f32,
    pub ability: crate::data::AbilityKind,
    pub passive: PassiveKind,
    pub sprite: &'static str,

    pub walk_sprite: &'static str,
}

/// GML `scrRaceGetPassiveSkillDescription` verbatim
/// (`scripts/scrRaces/scrRaces.gml:249`): unlocalized campfire passive
/// line (first half of the two-line skills text). `@w/@s/@r/@g/@y/@b`
/// color tags ride through to the `draw_text_nt` backend.
pub fn race_passive_text(race: crate::data::RaceId) -> &'static str {
    match race {
        crate::data::RaceId::Random => "???",
        crate::data::RaceId::Fish => "GETS MORE @yAMMO@w",
        crate::data::RaceId::Crystal => "MORE MAX @rHP@w",
        crate::data::RaceId::Eyes => "SEES IN THE DARK",
        crate::data::RaceId::Melting => "LESS MAX @rHP@w#MORE @gRADS@w",
        crate::data::RaceId::Plant => "IS FASTER",
        crate::data::RaceId::Venuz => "HIGHER @wRATE OF FIRE@s",
        crate::data::RaceId::Steroids => "INACCURATE#AUTOMATIC WEAPONS",
        crate::data::RaceId::Robot => "FINDS BETTER TECH",
        crate::data::RaceId::Chicken => "HARD TO KILL",
        crate::data::RaceId::Rebel => "PORTALS @rHEAL@w",
        crate::data::RaceId::Horror => "EXTRA @gMUTATION@w CHOICE",
        crate::data::RaceId::Rogue => "BLAST ARMOR, @bHEAT@w",
        crate::data::RaceId::BigDog => "MORE @rHP@w#SPIN ATTACK",
        crate::data::RaceId::Skeleton => "LESS HP, SPEED, AND ACCURACY",
        crate::data::RaceId::Frog => "CAN'T STAND STILL#TOXIC IMMUNITY",
        crate::data::RaceId::Cuz => "3 WEAPONS",
    }
}

/// GML `scrRaceGetActiveSkillDescription` verbatim
/// (`scripts/scrRaces/scrRaces.gml:275`): unlocalized campfire active
/// line (second half of the two-line skills text).
pub fn race_active_text(race: crate::data::RaceId) -> &'static str {
    match race {
        crate::data::RaceId::Random => "???",
        crate::data::RaceId::Fish => "CAN @wROLL@s",
        crate::data::RaceId::Crystal => "CAN @wSHIELD@s",
        crate::data::RaceId::Eyes => "TELEKINESIS",
        crate::data::RaceId::Melting => "EXPLODE @wCORPSES@s",
        crate::data::RaceId::Plant => "@wSNARE@s ENEMIES",
        crate::data::RaceId::Venuz => "@wPOP POP",
        crate::data::RaceId::Steroids => "DUAL WIELDING",
        crate::data::RaceId::Robot => "CAN EAT @wWEAPONS@s",
        crate::data::RaceId::Chicken => "CAN THROW @wWEAPONS@s",
        crate::data::RaceId::Rebel => "CAN SPAWN @wALLIES@s",
        crate::data::RaceId::Horror => "@gRADIATION@w BEAM",
        crate::data::RaceId::Rogue => "@bPORTAL STRIKE@s",
        crate::data::RaceId::BigDog => "MISSILES",
        crate::data::RaceId::Skeleton => "BLOOD GAMBLE",
        crate::data::RaceId::Frog => "@gTOXIC@w CLOUD",
        crate::data::RaceId::Cuz => "@bCRY",
    }
}

pub fn character_def(id: crate::data::RaceId) -> CharacterDef {
    match id {
        crate::data::RaceId::Fish => CharacterDef {
            name: "Fish",
            color: [0.25, 0.95, 0.35, 1.0f32],
            max_hp: 8,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::Flip,
            passive: PassiveKind::None,
            sprite: "images/sprMutant1Idle.png",
            walk_sprite: "images/sprMutant1Walk.png",
        },
        crate::data::RaceId::Crystal => CharacterDef {
            name: "Crystal",
            color: [0.35, 0.65, 1.0, 1.0f32],
            max_hp: 10,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::Shield,
            passive: PassiveKind::ShieldOnHit,
            sprite: "images/sprMutant2Idle.png",
            walk_sprite: "images/sprMutant2Walk.png",
        },
        crate::data::RaceId::Eyes => CharacterDef {
            name: "Eyes",
            color: [0.85, 0.4, 1.0, 1.0f32],
            max_hp: 8,
            speed_mult: 1.0,
            pickup_range: 175.0,
            ability: crate::data::AbilityKind::Telekinesis,
            passive: PassiveKind::None,
            sprite: "images/sprMutant3Idle.png",
            walk_sprite: "images/sprMutant3Walk.png",
        },
        crate::data::RaceId::Melting => CharacterDef {
            name: "Melting",
            color: [0.95, 0.85, 0.45, 1.0f32],
            max_hp: 2,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::Detonate,
            passive: PassiveKind::ChainExplosions,
            sprite: "images/sprMutant4Idle.png",
            walk_sprite: "images/sprMutant4Walk.png",
        },
        crate::data::RaceId::Plant => CharacterDef {
            name: "Plant",
            color: [0.3, 0.85, 0.35, 1.0f32],
            max_hp: 8,
            speed_mult: 1.125,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::Snare,
            passive: PassiveKind::None,
            sprite: "images/sprMutant5Idle.png",
            walk_sprite: "images/sprMutant5Walk.png",
        },
        crate::data::RaceId::Venuz => CharacterDef {
            name: "Venuz",
            color: [0.85, 0.7, 0.2, 1.0f32],
            max_hp: 8,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::PopPop,
            passive: PassiveKind::None,
            sprite: "images/sprMutant6Idle.png",
            walk_sprite: "images/sprMutant6Walk.png",
        },
        crate::data::RaceId::Steroids => CharacterDef {
            name: "Steroids",
            color: [0.9, 0.25, 0.25, 1.0f32],
            max_hp: 8,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::GetLoaded,
            passive: PassiveKind::FastReload,
            sprite: "images/sprMutant7Idle.png",
            walk_sprite: "images/sprMutant7Walk.png",
        },

        crate::data::RaceId::Robot => CharacterDef {
            name: "Robot",
            color: [0.6, 0.6, 0.65, 1.0f32],
            max_hp: 8,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::EatWeapon,
            passive: PassiveKind::FreeAmmo,
            sprite: "images/sprMutant8Idle.png",
            walk_sprite: "images/sprMutant8Walk.png",
        },
        crate::data::RaceId::Chicken => CharacterDef {
            name: "Chicken",
            color: [0.95, 0.9, 0.6, 1.0f32],
            max_hp: 8,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::Throw,
            passive: PassiveKind::Headless,
            sprite: "images/sprMutant9Idle.png",
            walk_sprite: "images/sprMutant9Walk.png",
        },
        crate::data::RaceId::Rebel => CharacterDef {
            name: "Rebel",
            color: [0.75, 0.25, 0.55, 1.0f32],
            max_hp: 8,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::SpawnAlly,
            passive: PassiveKind::None,
            sprite: "images/sprMutant10Idle.png",
            walk_sprite: "images/sprMutant10Walk.png",
        },
        crate::data::RaceId::Horror => CharacterDef {
            name: "Horror",
            color: [0.5, 0.35, 0.85, 1.0f32],
            max_hp: 8,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::HorrorBeam,
            passive: PassiveKind::None,
            sprite: "images/sprMutant11Idle.png",
            walk_sprite: "images/sprMutant11Walk.png",
        },
        crate::data::RaceId::Rogue => CharacterDef {
            name: "Rogue",
            color: [0.35, 0.35, 0.45, 1.0f32],
            max_hp: 8,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::PortalStrike,
            passive: PassiveKind::None,
            sprite: "images/sprMutant12Idle.png",
            walk_sprite: "images/sprMutant12Walk.png",
        },
        crate::data::RaceId::BigDog => CharacterDef {
            name: "Big Dog",
            color: [0.55, 0.38, 0.28, 1.0f32],
            max_hp: 300,
            speed_mult: 0.5,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::RocketBarrage,
            passive: PassiveKind::None,
            sprite: "images/sprMutant13Idle.png",
            walk_sprite: "images/sprMutant13Walk.png",
        },
        crate::data::RaceId::Skeleton => CharacterDef {
            name: "Skeleton",
            color: [0.9, 0.9, 0.92, 1.0f32],
            max_hp: 4,
            speed_mult: 0.75,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::BloodGamble,
            passive: PassiveKind::None,
            sprite: "images/sprMutant14Idle.png",
            walk_sprite: "images/sprMutant14Walk.png",
        },
        crate::data::RaceId::Frog => CharacterDef {
            name: "Frog",
            color: [0.45, 0.8, 0.55, 1.0f32],
            max_hp: 8,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::ToxicPuke,
            passive: PassiveKind::None,
            sprite: "images/sprMutant15Idle.png",
            walk_sprite: "images/sprMutant15Walk.png",
        },
        crate::data::RaceId::Cuz => CharacterDef {
            name: "Cuz",
            color: [0.8, 0.65, 0.35, 1.0f32],
            max_hp: 8,
            speed_mult: 1.0,
            pickup_range: 95.0,
            ability: crate::data::AbilityKind::CuzSwap,
            passive: PassiveKind::None,
            sprite: "images/sprMutant16Idle.png",
            walk_sprite: "images/sprMutant16Walk.png",
        },
        crate::data::RaceId::Random => character_def(crate::data::RaceId::Fish),
    }
}

impl SaveData {
    pub fn race_unlocked(&self, race: RaceId) -> bool {
        if race == RaceId::Random {
            return true;
        }

        if let Some(lo) = self.races.get(&race)
            && lo.unlocked
        {
            return true;
        }

        let name = character_def(race).name;
        race == RaceId::Fish
            || self
                .unlocked_characters
                .iter()
                .any(|s| s.eq_ignore_ascii_case(name))
    }

    pub fn race_loadout_mut(&mut self, race: RaceId) -> &mut RaceLoadout {
        self.races.entry(race).or_insert_with(|| RaceLoadout {
            unlocked: race == RaceId::Fish,
            unlocked_skins: [true, false, false, false],
            preferred_skin: 0,
            stored_weapon: WeaponId(0),
            start_weapon: WeaponId(0),
            start_crown: 0,
        })
    }

    pub fn race_loadout(&self, race: RaceId) -> RaceLoadout {
        let mut lo = self.races.get(&race).cloned().unwrap_or(RaceLoadout {
            unlocked: self.race_unlocked(race),
            unlocked_skins: [true, false, false, false],
            preferred_skin: 0,
            stored_weapon: WeaponId(0),
            start_weapon: WeaponId(0),
            start_crown: 0,
        });

        // Drop stored gun with no start gun.
        if lo.start_weapon == WeaponId(0) {
            lo.stored_weapon = WeaponId(0);
        }

        // Convert port crown ids to GML ids.
        if lo.start_crown != 0 {
            let gml = crown_port_to_gml(lo.start_crown);
            if !self.crown_unlocked(race, gml) {
                lo.start_crown = 0;
            }
        }

        lo
    }

    pub fn sanitize_loadouts(&mut self) {
        for lo in self.races.values_mut() {
            if lo.start_weapon.0 != 0 && lo.start_weapon != lo.stored_weapon {
                lo.start_weapon = WeaponId(0);
            }

            if lo.stored_weapon.0 == 0 {
                lo.start_weapon = WeaponId(0);
            }
        }

        let mut to_reset = Vec::new();
        for (race, lo) in self.races.iter() {
            if lo.start_crown != 0 {
                let gml = crown_port_to_gml(lo.start_crown);
                if !self.crown_unlocked(*race, gml) {
                    to_reset.push(*race);
                }
            }
        }
        for race in to_reset {
            if let Some(lo) = self.races.get_mut(&race) {
                lo.start_crown = 0;
            }
        }
    }

    fn crown_row(&self, race: RaceId) -> [bool; 14] {
        const BASE: [bool; 14] = [
            true, true, false, false, false, false, false, false, false, false, false, false,
            false, false,
        ];
        self.crown_got.get(&race).copied().unwrap_or(BASE)
    }

    pub fn crown_unlocked(&self, race: RaceId, crown: u8) -> bool {
        if crown as usize >= 14 || race == RaceId::Random {
            return false;
        }
        self.race_unlocked(race) && self.crown_row(race)[crown as usize]
    }

    pub fn any_crown_unlocked(&self, race: RaceId) -> bool {
        if race == RaceId::Random {
            return false;
        }
        let row = self.crown_row(race);
        (2..14).any(|i| row[i])
    }

    pub fn skin_unlocked(&self, race: RaceId, skin: u8) -> bool {
        if skin == 0 {
            return true;
        }
        // GML `scr_loadout_race_set_skin` verbatim: `cgot && cskingot`.
        // Skin letters past the race max (`scrRaceGetMaxSkinCount`, secret
        // gate unmodeled so BigDog/Frog hold 1, everything else 3) never
        // unlock even if a flag was set.
        if !self.race_unlocked(race) {
            return false;
        }
        if (skin as usize) >= race_max_skin_count(race) {
            return false;
        }
        race != RaceId::Random
            && self.races.get(&race).is_some_and(|lo| {
                lo.unlocked_skins
                    .get(skin as usize)
                    .copied()
                    .unwrap_or(false)
            })
    }

    pub fn unlock_crown(&mut self, race: RaceId, crown: u8) {
        if (crown as usize) >= 14 || race == RaceId::Random || !self.race_unlocked(race) {
            return;
        }
        let row = self.crown_got.entry(race).or_insert({
            let mut r = [false; 14];
            r[0] = true;
            r[1] = true;
            r
        });
        if row[crown as usize] {
            return;
        }
        row[crown as usize] = true;

        self.race_loadout_mut(race).start_crown = crown_gml_to_port(crown);
    }
}

/// Port crown id (save identity) to GML crown index: GML crowns are
/// 1-based with an extra offset (bevy `content.rs` verbatim).
pub fn crown_port_to_gml(id: u8) -> u8 {
    if id == 0 { 1 } else { id + 1 }
}

/// GML crown index back to port id (bevy `content.rs` verbatim).
pub fn crown_gml_to_port(id: u8) -> u8 {
    if id <= 1 { 0 } else { id - 1 }
}

pub fn is_race_unlocked(save: &SaveData, race: RaceId) -> bool {
    match race {
        RaceId::Fish | RaceId::Random => true,
        _ => save.race_unlocked(race),
    }
}

/// GML `scrInitStats` progress count verbatim (`progress/maxprogress`
/// for the stats unlocks row): races 1..16 skip the kinda-secret trio
/// entirely; loadout races count crowns 1..=13 (`crownmax`) plus the
/// race unlock (no max bump, verbatim) plus skins 1..max-1; hardmode
/// adds one each side. The `crownmax + 1` stored-weapon slot only feeds
/// the per-race tally (`race_prog_max`), never the global one —
/// mirrored by ignoring it here, as do stored weapons. Skin maxima
/// come from [`race_max_skin_count`] (`scrRaceGetMaxSkinCount(race,
/// false)` verbatim: BigDog/Frog hold 1, everything else 3 without
/// the hidden-NTT gate the port does not model).
pub fn race_max_skin_count(race: crate::data::RaceId) -> usize {
    use crate::data::RaceId;
    match race {
        RaceId::BigDog | RaceId::Frog => 1,
        _ => 3,
    }
}

/// GML `scrRaceGetMaxSkinCount` verbatim (kept as the single source;
/// [`race_max_skin_count`] is the `_show_secret=false` projection the
/// progress count uses).
pub fn unlock_progress(save: &SaveData) -> (u32, u32) {
    use crate::data::RaceId;
    let mut progress = 0u32;
    let mut maxprogress = 0u32;
    for gml in 1..16u8 {
        let Some(race) = crate::state::menus::race_from_gml_id(gml as usize) else {
            continue;
        };
        if crate::state::menus::race_is_hidden(race) {
            continue;
        }
        if !matches!(
            race,
            RaceId::Random | RaceId::BigDog | RaceId::Skeleton | RaceId::Frog
        ) {
            // GML `scrInitStats:36-44`: the global max grows once per
            // crown (`maxprogress++` for `i = 1..=crownmax`); the
            // `crownmax + 1` stored-weapon slot only feeds the per-race
            // tally, never the global one.
            let row = save.crown_row(race);
            for id in 1..=13usize {
                maxprogress += 1;
                if row[id] {
                    progress += 1;
                }
            }
        }
        if save.race_unlocked(race) {
            progress += 1;
        }
        // GML `scrInitStats:57-65`: `for skin = 1; skin <
        // scrRaceGetMaxSkinCount(race, false); skin++`.
        let max_skins = race_max_skin_count(race);
        let skins = save
            .races
            .get(&race)
            .map(|l| l.unlocked_skins)
            .unwrap_or([true, false, false, false]);
        for skin in 1..max_skins {
            maxprogress += 1;
            if skins.get(skin).copied().unwrap_or(false) {
                progress += 1;
            }
        }
    }
    maxprogress += 1;
    if save.hardmode_unlocked {
        progress += 1;
    }
    (progress, maxprogress)
}

/// GML `scrPlayerUpdateBestRunStats` best-kill branch verbatim
/// (custom-mode gate lives with callers): normal runs feed `cbst_*`
/// (`best_run_*` here), hardmode runs feed `hbst_*` (`hard_best_*`).
/// Strict `>` only, like GML.
pub fn update_best_run_stats(
    save: &mut SaveData,
    race_gml: u8,
    area: i32,
    sub: u32,
    lp: u32,
    kills: u32,
    hardmode: bool,
) {
    if hardmode {
        if kills > save.hard_best_kills {
            save.hard_best_kills = kills;
            save.hard_best_race = race_gml;
            save.hard_best_area = area;
            save.hard_best_sub = sub;
            save.hard_best_loop = lp;
        }
    } else if kills > save.best_run_kills {
        save.best_run_kills = kills;
        save.best_run_race = race_gml;
        save.best_run_area = area;
        save.best_run_sub = sub;
        save.best_run_loop = lp;
    }
}

pub fn check_progress_unlocks(
    save: &mut SaveData,
    floor: u32,
    loop_count: u32,
    died: bool,
    ate_weapon: bool,
    cleared_throne: bool,
) -> Vec<RaceId> {
    let mut got = Vec::new();
    let mut award = |save: &mut SaveData, r: RaceId| {
        if try_unlock_race(save, r) {
            got.push(r);
        }
    };

    if floor >= 4 {
        award(save, RaceId::Crystal);
    }

    if floor >= 5 {
        award(save, RaceId::Eyes);
    }

    if died {
        award(save, RaceId::Melting);
    }

    if floor >= 7 {
        award(save, RaceId::Plant);
    }

    if floor >= 9 {
        award(save, RaceId::Venuz);
    }

    if floor >= 11 {
        award(save, RaceId::Chicken);
    }

    if floor >= 12 {
        award(save, RaceId::Steroids);
    }

    if ate_weapon {
        award(save, RaceId::Robot);
    }

    if floor >= 15 || cleared_throne {
        award(save, RaceId::Horror);
    }

    if loop_count >= 1 {
        award(save, RaceId::Rebel);
        award(save, RaceId::Rogue);
    }

    if loop_count >= 2 || cleared_throne {
        award(save, RaceId::Cuz);
    }

    got
}

// ---------------------------------------------------------------------------
// Save file IO. Headless replacement for the bevy build's game_utils
// `SavePlugin::<SaveData>` (`SaveManager::new("com", "nt-recreated",
// "nt-recreated-bevy", "save.ron", SAVE_VERSION)`, see
// nt-recreated-bevy `src/app.rs`).
//
// Fidelity compromise (save path location): without the `directories`
// crate and without RON in this crate, the sim keeps it simple and
// portable — project-local JSON instead of the OS data dir + RON:
// `./nt-save.json` under the process working directory, overridable
// via the `NT_SAVE_PATH` env var (tests point it at a temp file).
// `serialize_save` / `parse_save` are pure (no disk) so round-trips
// stay unit-testable; only `store_save_to_file` /
// `load_save_from_file` touch `std::fs`.
// ---------------------------------------------------------------------------

/// Save file name for the project-local path.
pub fn save_file_name() -> &'static str {
    "nt-save.json"
}

/// Portable save path: `$NT_SAVE_PATH` when set, else
/// `<current_dir>/nt-save.json`.
pub fn save_file_path() -> PathBuf {
    if let Ok(p) = std::env::var("NT_SAVE_PATH")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(save_file_name())
}

/// Serialize a save to JSON (pretty; version stamp included).
pub fn serialize_save(save: &SaveData) -> Result<String, String> {
    serde_json::to_string_pretty(save).map_err(|e| e.to_string())
}

/// Parse save JSON back, then `sanitize_loadouts` (drops stale
/// start-crown/weapon picks the same way a fresh load does).
pub fn parse_save(text: &str) -> Result<SaveData, String> {
    let mut save: SaveData = serde_json::from_str(text).map_err(|e| e.to_string())?;
    save.sanitize_loadouts();
    Ok(save)
}

/// Write a save to disk (creates parent dirs).
pub fn store_save_to_file(save: &SaveData, path: &Path) -> Result<(), String> {
    let text = serialize_save(save)?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, text).map_err(|e| e.to_string())
}

/// Read a save from disk (`Err` when missing/corrupt; callers fall
/// back to `SaveData::default()`).
pub fn load_save_from_file(path: &Path) -> Result<SaveData, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    parse_save(&text)
}

/// Load a save, or default when the file is missing/corrupt.
pub fn load_or_default(path: &Path) -> SaveData {
    load_save_from_file(path).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Skin unlocks. Port of `nt-recreated-bevy/src/game/skin_unlocks.rs`
// (all 8 items): area, robot-weapon, crystal-damage and global checks
// plus their tick systems. Transforms: `AreaId` -> `crate::data::AreaId`,
// components -> `crate::comps_a`, `RaceId`/`SkinLetter` ->
// `crate::data`, `weapon_id_name` ->
// `crate::weapon_runtime::weapon_id_name`, `PLAYABLE_RACES` ->
// `crate::data::PLAYABLE_RACES`. No sprite/UI code in source; nothing
// omitted.
// ---------------------------------------------------------------------------

/// Area-gated race/skin unlocks (called when the run enters an area).
/// GML `scrUnlocks` area switch verbatim: race unlocks are unconditional
/// (single-player: the current run's race is the only present player),
/// Chicken/B needs hardmode (not modeled — stays locked), HQ Horror/C
/// needs ≤3 mutations held. Fresh unlocks also queue the GML
/// `UnlockScreen` popup (race or skin); pass the live menu when the
/// caller owns it.
pub fn check_area_skins(
    save: &mut SaveData,
    area: AreaId,
    loop_count: u32,
    skills_len: usize,
    race: RaceId,
    hardmode: bool,
    mut menu: Option<&mut crate::state::menus::MenuState>,
) {
    match area {
        AreaId::Sewers => {
            try_unlock_race_with_menu(save, RaceId::Eyes, menu.as_deref_mut());
            // GML `GameCont/Other_5`: Chicken-B needs hardmode.
            if hardmode && race == RaceId::Chicken {
                try_unlock_letter_with_menu(
                    save,
                    RaceId::Chicken,
                    SkinLetter::B,
                    menu.as_deref_mut(),
                );
            }
        }
        AreaId::PizzaSewers => {
            try_unlock_letter_with_menu(save, RaceId::Eyes, SkinLetter::B, menu.as_deref_mut());
        }
        AreaId::Scrapyards => {
            try_unlock_race_with_menu(save, RaceId::Plant, menu.as_deref_mut());
        }
        // Port `City` is YV's Mansion (GML 103): Venuz.
        AreaId::City => {
            try_unlock_race_with_menu(save, RaceId::Venuz, menu.as_deref_mut());
        }
        AreaId::CursedCaves => {
            try_unlock_letter_with_menu(
                save,
                RaceId::Crystal,
                SkinLetter::B,
                menu.as_deref_mut(),
            );
        }
        // Port `FrozenCity` is GML city (route floors 9-11): Robot.
        AreaId::FrozenCity => {
            try_unlock_race_with_menu(save, RaceId::Robot, menu.as_deref_mut());
        }
        AreaId::Jungle => {
            try_unlock_race_with_menu(save, RaceId::Chicken, menu.as_deref_mut());
        }
        AreaId::Labs => {
            try_unlock_race_with_menu(save, RaceId::Steroids, menu.as_deref_mut());
        }
        AreaId::Desert if loop_count >= 1 => {
            try_unlock_race_with_menu(save, RaceId::Rebel, menu.as_deref_mut());
        }
        AreaId::HQ if skills_len <= 3 => {
            try_unlock_letter_with_menu(
                save,
                RaceId::Horror,
                SkinLetter::C,
                menu.as_deref_mut(),
            );
        }
        _ => {}
    }
}

fn try_unlock_letter(save: &mut SaveData, race: RaceId, skin: SkinLetter) -> bool {
    try_unlock_letter_with_menu(save, race, skin, None)
}

fn try_unlock_letter_with_menu(
    save: &mut SaveData,
    race: RaceId,
    skin: SkinLetter,
    menu: Option<&mut crate::state::menus::MenuState>,
) -> bool {
    let ok = try_unlock_skin_with_menu(save, race, skin as usize, menu);
    if ok {
        if let Some(id) = achievement_for_skin(race, skin as usize) {
            unlock_achievement(save, id);
        }
        if race == RaceId::Cuz {
            unlock_achievement(save, if skin == SkinLetter::B { 45 } else { 58 });
        }
    }
    ok
}

/// Area-skin tick: fires when the `Run` resource changes (area/loop
/// advance), using the player race (or the selected character when no
/// player exists yet).
pub fn tick_area_skins(
    mut save: ResMut<SaveData>,
    run: Res<Run>,
    player_q: Query<(&Player, &RaceState), With<Player>>,
    menu: Option<ResMut<crate::state::menus::MenuState>>,
) {
    if !run.is_changed() {
        return;
    }
    let (skills_len, race) = player_q
        .iter()
        .next()
        .map(|(p, r)| (p.mutations.len(), r.race))
        .unwrap_or((0, RaceId::Fish));
    let mut menu_opt = menu.map(|m| m.into_inner());
    check_area_skins(
        &mut save,
        run.area,
        run.loop_count,
        skills_len,
        race,
        run.hardmode,
        menu_opt.as_deref_mut(),
    );
}

/// Robot weapon-name skin checks ("hyper" -> B, rusty revolver -> C).
pub fn check_robot_weapon_skins(save: &mut SaveData, race: RaceId, weapon_name: &str) {
    if race != RaceId::Robot {
        return;
    }
    let lower = weapon_name.to_ascii_lowercase();
    if lower.contains("hyper") {
        try_unlock_letter(save, RaceId::Robot, SkinLetter::B);
    }
    if lower.contains("rusty") && lower.contains("revolver") {
        try_unlock_letter(save, RaceId::Robot, SkinLetter::C);
    }
}

/// Robot-skin tick: scans the player's held weapons by table name.
pub fn tick_robot_skins(
    mut save: ResMut<SaveData>,
    player_q: Query<(&RaceState, &Inventory), With<Player>>,
) {
    for (rs, inv) in &player_q {
        if rs.race != RaceId::Robot {
            continue;
        }
        for w in inv.weapons.iter() {
            let name = crate::weapon_runtime::weapon_id_name(*w).to_ascii_lowercase();
            if name.contains("hyper") {
                try_unlock_letter(&mut save, RaceId::Robot, SkinLetter::B);
            }
            if name.contains("rusty") && name.contains("revolver") {
                try_unlock_letter(&mut save, RaceId::Robot, SkinLetter::C);
            }
        }
    }
}

/// Accumulated Crystal damage tracker (100+ damage unlocks skin C).
#[derive(Resource, Default)]
pub struct CrystalDamageTaken {
    pub total: f32,
    pub last_hp: i32,
}

/// Crystal-skin tick: tracks HP drops on the Crystal player.
pub fn tick_crystal_damage(
    mut save: ResMut<SaveData>,
    mut dmg: ResMut<CrystalDamageTaken>,
    player_q: Query<(&RaceState, &Health), With<Player>>,
) {
    for (rs, health) in &player_q {
        if rs.race != RaceId::Crystal {
            continue;
        }
        if dmg.last_hp == 0 {
            dmg.last_hp = health.max;
        }
        if health.hp < dmg.last_hp {
            let diff = (dmg.last_hp - health.hp) as f32;
            dmg.total += diff;
            if dmg.total >= 100.0 {
                try_unlock_letter(&mut save, RaceId::Crystal, SkinLetter::C);
            }
        }
        dmg.last_hp = health.hp;
    }
}

/// Global skin checks (GML `scrUnlocksCharacterStats` +
/// `scrUnlocksPlayerEquipment` verbatim, minus achievements/dailies):
/// all-golden stored weapons (Venuz B), loop-as-every-character (Fish
/// B), all B skins (Fish C), held golden/cursed counts (Cuz B/C),
/// 12+ mutations held (Melting C), 3+ blood sources held (Plant C).
pub fn tick_global_skins(
    mut save: ResMut<SaveData>,
    _run: Res<Run>,
    player_q: Query<(&Inventory, &Player, &RaceState), With<Player>>,
) {
    let all_golden = crate::data::PLAYABLE_RACES.iter().all(|&r| {
        if matches!(r, RaceId::BigDog | RaceId::Skeleton | RaceId::Frog) {
            return true;
        }
        save.races
            .get(&r)
            .map(|lo| {
                let name =
                    crate::weapon_runtime::weapon_id_name(lo.stored_weapon).to_ascii_lowercase();
                name.contains("golden")
            })
            .unwrap_or(false)
    });
    if all_golden {
        try_unlock_letter(&mut save, RaceId::Venuz, SkinLetter::B);
    }

    // Fish/B: looped as every non-hidden character.
    let looped_all = crate::data::PLAYABLE_RACES.iter().all(|&r| {
        if matches!(r, RaceId::BigDog | RaceId::Skeleton | RaceId::Frog) {
            return true;
        }
        save.race_looped.get(&r).copied().unwrap_or(false)
    });
    if looped_all && save.race_unlocked(RaceId::Fish) {
        try_unlock_letter(&mut save, RaceId::Fish, SkinLetter::B);
    }

    let all_b = crate::data::PLAYABLE_RACES.iter().all(|&r| {
        if matches!(r, RaceId::BigDog | RaceId::Skeleton | RaceId::Frog) {
            return true;
        }
        save.races
            .get(&r)
            .map(|lo| lo.unlocked_skins[1])
            .unwrap_or(false)
    });
    if all_b {
        try_unlock_letter(&mut save, RaceId::Fish, SkinLetter::C);
    }

    for (inv, player, race_state) in &player_q {
        // GML `FROG_ZONE`: playing as Frog.
        if race_state.race == RaceId::Frog {
            unlock_achievement(&mut save, 28);
        }
        let golden_count = inv
            .weapons
            .iter()
            .filter(|w| {
                let n = crate::weapon_runtime::weapon_id_name(**w).to_ascii_lowercase();
                n.contains("golden")
            })
            .count();
        // GML `GOOD_FIND`: a golden weapon held as any character.
        if golden_count >= 1 {
            unlock_achievement(&mut save, 24);
        }
        let cursed_count = inv.cursed.iter().filter(|c| **c).count();

        if golden_count >= 3 {
            try_unlock_letter(&mut save, RaceId::Cuz, SkinLetter::B);
        }
        if cursed_count >= 6 {
            try_unlock_letter(&mut save, RaceId::Cuz, SkinLetter::C);
        }

        // Melting/C: 12+ mutations held.
        if player.mutations.len() >= 12 {
            try_unlock_letter(&mut save, RaceId::Melting, SkinLetter::C);
        }

        // Plant/C: 3+ blood sources (blood crown, Bloodlust, blood
        // launcher/cannon/hammer held).
        let mut blood = 0;
        if player.crown == CrownKind::Blood {
            blood += 1;
        }
        if player.mutations.contains(&MutationId::Bloodlust) {
            blood += 1;
        }
        for w in inv.weapons.iter() {
            let n = crate::weapon_runtime::weapon_id_name(*w).to_ascii_uppercase();
            if n == "BLOOD LAUNCHER" || n == "BLOOD CANNON" || n == "BLOOD HAMMER" {
                blood += 1;
            }
        }
        if blood >= 3 {
            try_unlock_letter(&mut save, RaceId::Plant, SkinLetter::C);
        }
    }
}
