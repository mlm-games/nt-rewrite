//! Audio selection: cues, area music/ambience.
//! Systems here output DATA and the platform layer (repame-audio) plays it:
//! * one-shots ([`AudioCue`]) queue with stem + volume + pitch variance; the
//!   pitch is sampled at play time, as GML does inside the play script
//!   (GML `scripts/snd_play_hit.gml:11`), and the backend resolves stems
//!   against its own asset store.
//! * area music/ambience ([`sync_area_audio`]) resolves the GML `MusCont`
//!   selection law to bare GML stems in [`AreaAudioState`], polled by the
//!   backend every tick.
//! Only the cues the ported systems need exist yet.

use bevy_ecs::prelude::*;
use repame_sim::SimTime;

use crate::comps_a::{Player, RaceState, Run};
use crate::comps_b::{
    BossBrain, CampfireProp, CrownPedestal, Enemy, FloorTransition, LoopTransition,
};
use crate::data::{AreaId, EnemyKind, RaceId};
use crate::msg::Queue;
use crate::state::{AppState, Paused};

/// One fire-and-forget sound with playback variation.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioCue {
    pub name: &'static str,
    pub volume: f32,
    pub variance: f32,
}

/// Runtime mix buses. GML option defaults (`scrOptionsUpdate.gml:13-16`:
/// `save_get_option(..., true)` -> 1.0 for master/sfx/music/ambient;
/// `ui` is port-only at 1); the settings->channel sync from
/// the save slice writes here later.
#[derive(Resource, Debug, Clone, Copy)]
pub struct AudioChannels {
    pub master: f32,
    pub sfx: f32,
    pub music: f32,
    pub ui: f32,
}

impl Default for AudioChannels {
    fn default() -> Self {
        Self {
            master: 1.0,
            sfx: 1.0,
            music: 1.0,
            ui: 1.0,
        }
    }
}

impl AudioChannels {
    pub fn sfx_bus(&self) -> f32 {
        (self.master * self.sfx).clamp(0.0, 1.0)
    }

    pub fn music_bus(&self) -> f32 {
        (self.master * self.music).clamp(0.0, 1.0)
    }

    pub fn ui_bus(&self) -> f32 {
        (self.master * self.ui).clamp(0.0, 1.0)
    }
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct MainVol {
    value: f32,
}

impl Default for MainVol {
    fn default() -> Self {
        Self { value: 1.0 }
    }
}

impl MainVol {
    pub fn duck(&mut self, gain: f32) {
        self.value = gain;
    }

    pub fn step(&mut self, dt_secs: f32) -> f32 {
        self.value = 1.0 - (1.0 - self.value) * 0.6f32.powf(dt_secs * 30.0);
        if 1.0 - self.value < 1e-4 {
            self.value = 1.0;
        }
        self.value
    }
}

pub fn mainvol_gain(world: &World) -> f32 {
    world.get_resource::<MainVol>().map_or(1.0, |mv| mv.value)
}

/// Sound bank handle (asset paths resolve platform-side).
#[derive(Resource, Debug, Default)]
pub struct GameAudio;

impl GameAudio {
    fn cue(cues: &mut Queue<AudioCue>, name: &'static str, volume: f32, variance: f32) {
        cues.push(AudioCue {
            name,
            volume,
            variance,
        });
    }

    /// Hit thock (GML `Bullet1/Collision_Wall.gml:7` and 15 sibling
    /// `*/Collision_Wall.gml` sites - `snd_play_hit(sndHitWall, 0.2)`,
    /// with `WepPickup/Collision_Wall.gml:6` on the script's `0.2`
    /// default).
    pub fn play_hit(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndHitWall", 1.0, 0.2);
    }

    /// Explosion boom (GML `Grenade/Destroy_0.gml:3`
    /// `snd_play(sndExplosionL)`).
    pub fn play_boom(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndExplosionL", 1.0, 0.0);
    }

    /// Pickup blip (GML `AmmoPickup/Collision_Player.gml:26`
    /// `snd_play(sndAmmoPickup)`).
    pub fn play_pickup(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndAmmoPickup", 1.0, 0.0);
    }

    /// Portal whoosh (GML `GenCont/Destroy_0.gml:93` /
    /// `Portal/Create_0.gml:5` `snd_play(sndPortalOpen)`).
    pub fn play_portal(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndPortalOpen", 1.0, 0.0);
    }

    // follow the GML call site cited on each helper.

    /// Explosion crack (GML `Sniper/Destroy_0.gml:6`
    /// `snd_play(sndExplosion)`).
    pub fn play_explode(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndExplosion", 1.0, 0.0);
    }

    /// Chest zap (GML `AmmoChest/Collision_PortalShock.gml:7`
    /// `snd_play(sndChest)`).
    pub fn play_chest(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndChest", 1.0, 0.0);
    }

    /// Proto-chest open (GML `ProtoChest/Collision_Player.gml:24`
    /// `snd_play(sndWeaponChest)`).
    pub fn play_weapon_chest(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndWeaponChest", 1.0, 0.0);
    }

    /// Health chest (GML `HealthChest/Collision_Player.gml:18`
    /// `snd_play(... sndHealthChestBig : sndHealthChest)`).
    pub fn play_health_chest(&self, cues: &mut Queue<AudioCue>, big: bool) {
        Self::cue(
            cues,
            if big {
                "sndHealthChestBig"
            } else {
                "sndHealthChest"
            },
            1.0,
            0.0,
        );
    }

    /// Pickup fade-out (GML `AmmoPickup/Alarm_0.gml:3`
    /// `snd_play(sndPickupDisappear)`).
    pub fn play_pickup_disappear(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndPickupDisappear", 1.0, 0.0);
    }

    /// GML `Rad/Step_0.gml:33` `snd_play(sndRadPickup)`.
    pub fn play_rad_pickup(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndRadPickup", 1.0, 0.0);
    }

    /// GML `HPPickup/Collision_Player.gml:18`
    /// `scr_skill_get(mut_second_stomach) ? sndHPPickupBig : sndHPPickup`.
    pub fn play_hp_pickup(&self, cues: &mut Queue<AudioCue>, big: bool) {
        Self::cue(
            cues,
            if big { "sndHPPickupBig" } else { "sndHPPickup" },
            1.0,
            0.0,
        );
    }

    /// GML `AmmoPickup/Collision_Player.gml:26` `sndAmmoPickup`.
    pub fn play_ammo_pickup(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndAmmoPickup", 1.0, 0.0);
    }

    /// GML `WeaponChest/Collision_Player.gml:26-34`: Oasis
    /// `sndOasisChest` / Curses `sndCursedChest` / else `sndWeaponChest`.
    pub fn play_weapon_chest_open(
        &self,
        cues: &mut Queue<AudioCue>,
        underwater: bool,
        cursed: bool,
    ) {
        Self::cue(
            cues,
            if underwater {
                "sndOasisChest"
            } else if cursed {
                "sndCursedChest"
            } else {
                "sndWeaponChest"
            },
            1.0,
            0.0,
        );
    }

    /// GML `AmmoChest/Collision_Player.gml:23` / `AmmoChestMystery:32`
    /// / `IDPDChest/Collision_Player.gml:15`:
    /// `snd_play(GameCont.underwater ? sndOasisChest : sndAmmoChest)`.
    pub fn play_ammo_chest_open(&self, cues: &mut Queue<AudioCue>, underwater: bool) {
        Self::cue(
            cues,
            if underwater {
                "sndOasisChest"
            } else {
                "sndAmmoChest"
            },
            1.0,
            0.0,
        );
    }

    /// GML `GoldChest/Collision_Player.gml:10` `sndGoldChest`.
    pub fn play_gold_chest(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndGoldChest", 1.0, 0.0);
    }

    /// GML `RogueChest/Collision_Player.gml:17` (and
    /// `RogueAmmo/Collision_Player.gml:20`) `sndRogueCanister`.
    pub fn play_rogue_canister(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndRogueCanister", 1.0, 0.0);
    }

    /// GML `RadChest/Destroy_0.gml:11-12` / `RadMaggotChest/Destroy_0.gml:11`
    /// `snd_play(sndEXPChest)` - the rad-chest family, never the
    /// generic pickup blip.
    pub fn play_exp_chest(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndEXPChest", 1.0, 0.0);
    }

    /// GML `CursedPickup/Create_0.gml:8` `snd_play_hit(sndCursedPickup, 0.2)`.
    /// `snd_play_hit`'s 2nd arg is random PITCH and `_gain` is
    /// `UberCont.opt_sndvol` (1.0) - see `scripts/snd_play_hit`; the port's
    /// 4th field is the same pitch jitter law, so it carries `0.2` as-is.
    pub fn play_cursed_pickup(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndCursedPickup", 1.0, 0.2);
    }

    /// GML `CursedPickup/Alarm_0.gml:1-4`:
    /// `snd_play(sndExplosionS)` + `snd_play(sndCursedPickupDisappear)`.
    pub fn play_cursed_pickup_disappear(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndExplosionS", 1.0, 0.0);
        Self::cue(cues, "sndCursedPickupDisappear", 1.0, 0.0);
    }

    /// GML `BigGenerator/Destroy_0.gml:34` `snd_play(sndNothingGenerators)`.
    pub fn play_nothing_generators(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndNothingGenerators", 1.0, 0.0);
    }

    /// GML `Player/Collision_WepPickup.gml:14-27`: guitar /
    /// electric-guitar / gold / plain weapon collect stems.
    pub fn play_weapon_pickup(&self, cues: &mut Queue<AudioCue>, weapon: crate::data::WeaponId) {
        let meta = crate::weapon_runtime::weapon_meta(weapon);
        let name = if meta.wep_gold {
            "sndGoldPickup"
        } else if meta.wep_sprt == "sprGuitar" {
            "sndGuitarPickup"
        } else if meta.wep_sprt == "sprElectricGuitar" {
            "sndSwapElectricGuitar"
        } else {
            "sndWeaponPickup"
        };
        Self::cue(cues, name, 1.0, 0.0);
    }

    /// GML `BigWeaponChest/Collision_Player.gml:26-28` /
    /// `CursedBigChest:29-31`: `snd_play_hit(sndBig*Chest)` plus the
    /// per-race `snd_play_hit_big(snd_chst)` thump. `chst` stem is
    /// `scr_race_get_sound(race, "Chst", sndMutant1Chst)`.
    pub fn play_big_chest_open(
        &self,
        cues: &mut Queue<AudioCue>,
        cursed: bool,
        chst: &'static str,
    ) {
        Self::cue(
            cues,
            if cursed {
                "sndBigCursedChest"
            } else {
                "sndBigWeaponChest"
            },
            1.0,
            0.2,
        );
        Self::cue(cues, chst, 1.0, 0.2);
    }

    /// GML `Player/Collision_WepPickup.gml:67` `snd_play(wep_swap[wep])`
    /// - the per-weapon swap stem.
    pub fn play_weapon_swap(&self, cues: &mut Queue<AudioCue>, weapon: crate::data::WeaponId) {
        let swap = crate::weapon_runtime::weapon_meta(weapon).wep_swap;
        if !swap.is_empty() {
            Self::cue(cues, swap, 1.0, 0.0);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum MusicCue {
    Desert,
    Sewers,
    Scrapyards,
    CrystalCaves,
    FrozenCity,
    Labs,
    Palace,
    Oasis,
    PizzaSewers,
    CursedCaves,
    Jungle,
    Vault,
    CrownVault,
    Hq,
    HqRogue,
    City,
    Campfire,
    /// GML `MusCont/Alarm_11.gml:30` `"mus" + string(area)` with
    /// `area_crib = 107` -> `mus107`.
    Crib,

    BossDesert,
    BossSewers,
    BossScrapyards,
    BossCrystalCaves,
    BossFrozenCity,
    BossLabs,
    BossPalace,
    BossCursedCaves,
    BossHq,
    BossCampfire,
    /// GML `MusCont/Alarm_2.gml:16` `case area_crib: song = musBoss9`.
    BossCrib,
    /// GML `MusCont/Alarm_4.gml:3` `song = mus100b`.
    BossCrownGuardian,
    /// GML `MusCont/Alarm_5.gml:4` `song = musBoss4B`.
    BossThroneII,
    /// GML `MusCont/Alarm_11.gml:10` BigDog-race `song = musBoss2`.
    RaceBigDog,
    /// GML `MusCont/Alarm_3.gml:5` `song = musBossDead`.
    BossDead,
    TitleThemeA,
    TitleThemeB,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum AmbienceCue {
    None,
    /// GML `Menu/Create_0.gml:96` char-select bed `amb0`.
    Menu,
    /// GML `Logo/Alarm_0.gml:8` boot-logo bed `sndLogoLoop`
    /// (`snd_play_ambience`, looped until `Logo/Destroy_0.gml:1`
    /// `snd_stop(sndLogoLoop)` - the Splash -> MainMenu handoff).
    LogoLoop,
    /// GML `MusCont/Alarm_11.gml:49,55` `amb0b` - the `audio_exists`
    /// fallback for any area whose `amb<area>` asset is missing (the
    /// crib) and the campfire special case.
    Rest,
    /// GML `MusCont/Alarm_5.gml:5` `amb = amb0c`.
    ThroneII,
    Desert,
    Sewers,
    Scrapyards,
    CrystalCaves,
    FrozenCity,
    Labs,
    Palace,
    Vault,
    Oasis,
    PizzaSewers,
    City,
    CursedCaves,
    Jungle,
    Hq,
}

/// Current area loop selection. The backend polls this resource and
/// plays [`music_path`]/[`ambience_path`] on a cue change (instant
/// switch, GML `snd_play_music`/`snd_play_ambience` parity);
/// `music_volume`/`ambience_volume` are the absolute per-layer gains
/// ([`tick_area_audio_fades`], master excluded).
#[derive(Resource, Debug)]
pub struct AreaAudioState {
    pub current_music: Option<MusicCue>,
    pub current_ambience: Option<AmbienceCue>,
    pub music_volume: f32,
    pub ambience_volume: f32,
    boss_alive: bool,
    jingle_in: f32,
    boss_dead_pending: bool,
    boss_dead: bool,
    boss_dead_area: AreaId,
    silent_until_room: bool,
    title_in: f32,
    room: (u32, AreaId),
}

impl Default for AreaAudioState {
    fn default() -> Self {
        Self {
            current_music: None,
            current_ambience: None,
            music_volume: 0.0,
            ambience_volume: 0.0,
            boss_alive: false,
            jingle_in: 0.0,
            boss_dead_pending: false,
            boss_dead: false,
            boss_dead_area: AreaId::default(),
            silent_until_room: false,
            title_in: TITLE_A_SECS,
            room: (0, AreaId::Campfire),
        }
    }
}

#[derive(Resource, Debug)]
pub struct AmbFilter(pub f32);

impl Default for AmbFilter {
    fn default() -> Self {
        Self(1.0)
    }
}

pub fn music_for_area(area: AreaId) -> MusicCue {
    match area {
        AreaId::Desert => MusicCue::Desert,
        AreaId::Sewers => MusicCue::Sewers,
        AreaId::Scrapyards => MusicCue::Scrapyards,
        AreaId::CrystalCaves => MusicCue::CrystalCaves,
        AreaId::FrozenCity => MusicCue::FrozenCity,
        AreaId::Labs => MusicCue::Labs,
        AreaId::Palace => MusicCue::Palace,

        AreaId::Oasis => MusicCue::Oasis,
        AreaId::PizzaSewers => MusicCue::PizzaSewers,
        AreaId::CursedCaves => MusicCue::CursedCaves,
        AreaId::Jungle => MusicCue::Jungle,
        AreaId::Vault => MusicCue::Vault,
        AreaId::CrownVault => MusicCue::CrownVault,
        AreaId::HQ => MusicCue::Hq,
        AreaId::City => MusicCue::City,
        AreaId::Campfire => MusicCue::Campfire,
        AreaId::Crib => MusicCue::Crib,

        AreaId::Loop => MusicCue::Desert,
    }
}

pub fn ambience_for_area(area: AreaId) -> AmbienceCue {
    match area {
        AreaId::Desert => AmbienceCue::Desert,
        AreaId::Sewers => AmbienceCue::Sewers,
        AreaId::Scrapyards => AmbienceCue::Scrapyards,
        AreaId::CrystalCaves => AmbienceCue::CrystalCaves,
        AreaId::FrozenCity => AmbienceCue::FrozenCity,
        AreaId::Labs => AmbienceCue::Labs,
        AreaId::Palace => AmbienceCue::Palace,

        AreaId::Oasis => AmbienceCue::Oasis,
        AreaId::PizzaSewers => AmbienceCue::PizzaSewers,
        AreaId::CursedCaves => AmbienceCue::CursedCaves,
        AreaId::Jungle => AmbienceCue::Jungle,
        AreaId::Vault => AmbienceCue::Vault,
        AreaId::CrownVault => AmbienceCue::Vault,
        AreaId::HQ => AmbienceCue::Hq,
        AreaId::City => AmbienceCue::City,
        AreaId::Campfire => AmbienceCue::Rest,
        // GML `MusCont/Alarm_11.gml:31` looks up `amb107`, which the pack
        // does not ship, so `audio_exists` fails and the fallback at :49
        // (`amb = amb0b`) is the crib bed.
        AreaId::Crib => AmbienceCue::Rest,

        AreaId::Loop => AmbienceCue::Desert,
    }
}

/// GML `MusCont/Alarm_2.gml` stage-dependent boss switch:
/// `Alarm_5.gml:4`/`Alarm_4.gml:3` specials first, then the per-area
/// cases; `None` = no case (`Alarm_2.gml:17`) and the caller keeps
/// the current bed.
pub fn boss_music_cue(area: AreaId, throne_ii: bool, guardian: bool) -> Option<MusicCue> {
    if throne_ii {
        return Some(MusicCue::BossThroneII);
    }
    if guardian {
        return Some(MusicCue::BossCrownGuardian);
    }
    match area {
        AreaId::Desert | AreaId::Loop => Some(MusicCue::BossDesert),
        AreaId::Sewers => Some(MusicCue::BossSewers),
        AreaId::Scrapyards => Some(MusicCue::BossScrapyards),
        AreaId::CrystalCaves => Some(MusicCue::BossCrystalCaves),
        AreaId::FrozenCity => Some(MusicCue::BossFrozenCity),
        AreaId::Labs => Some(MusicCue::BossLabs),
        AreaId::Palace => Some(MusicCue::BossPalace),
        AreaId::Campfire => Some(MusicCue::BossCampfire),
        AreaId::CursedCaves => Some(MusicCue::BossCursedCaves),
        AreaId::HQ => Some(MusicCue::BossHq),
        AreaId::Crib => Some(MusicCue::BossCrib),
        _ => None,
    }
}

/// GML `MusCont` stem law: `"mus" + string(area)` at
/// `Alarm_11.gml:30` (campfire/HQ overrides at :53-68) plus the
/// `Alarm_2`/`Alarm_3`/`Alarm_4`/`Alarm_5`/`Alarm_0` specials, as bare
/// GML asset stems the backend resolves.
pub fn music_path(cue: MusicCue) -> Option<&'static str> {
    Some(match cue {
        MusicCue::Desert => "mus1",
        MusicCue::Sewers => "mus2",
        MusicCue::Scrapyards => "mus3",
        MusicCue::CrystalCaves => "mus4",
        MusicCue::FrozenCity => "mus5",
        MusicCue::Labs => "mus6",
        MusicCue::Palace => "mus7",
        MusicCue::Vault => "mus100",
        MusicCue::CrownVault => "mus100",
        MusicCue::Oasis => "mus101",
        MusicCue::PizzaSewers => "mus102",
        MusicCue::City => "mus103",
        MusicCue::CursedCaves => "mus104",
        MusicCue::Jungle => "mus105",
        MusicCue::Hq => "mus106",
        MusicCue::HqRogue => "mus106b",
        MusicCue::Crib => "mus107",
        MusicCue::Campfire => "musBoss4Silence",
        MusicCue::BossDesert => "musBoss1",
        MusicCue::BossSewers => "musBoss5",
        MusicCue::BossScrapyards => "musBoss2",
        MusicCue::BossCrystalCaves => "musBoss6",
        MusicCue::BossFrozenCity => "musBoss3",
        MusicCue::BossLabs => "musBoss7",
        MusicCue::BossPalace => "musBoss4A",
        MusicCue::BossCursedCaves => "musBoss6B",
        MusicCue::BossHq => "musBoss8",
        MusicCue::BossCampfire => "musBoss4B",
        MusicCue::BossCrib => "musBoss9",
        MusicCue::BossCrownGuardian => "mus100b",
        MusicCue::BossThroneII => "musBoss4B",
        MusicCue::RaceBigDog => "musBoss2",
        MusicCue::BossDead => "musBossDead",
        MusicCue::TitleThemeA => "musThemeA",
        MusicCue::TitleThemeB => "musThemeB",
    })
}

/// GML `MusCont` stem law: `"amb" + string(area)` at
/// `Alarm_11.gml:31` (`amb0b` fallback at :49, campfire at :55, HQ at
/// :67), plus `Menu/Create_0.gml:96`, `Alarm_5.gml:5` and
/// `Logo/Alarm_0.gml:8`; `None` is the BigDog-race `amb = -1` at
/// `Alarm_11.gml:11`.
pub fn ambience_path(cue: AmbienceCue) -> Option<&'static str> {
    match cue {
        AmbienceCue::None => None,
        AmbienceCue::Menu => Some("amb0"),
        AmbienceCue::LogoLoop => Some("sndLogoLoop"),
        AmbienceCue::Rest => Some("amb0b"),
        AmbienceCue::ThroneII => Some("amb0c"),
        AmbienceCue::Desert => Some("amb1"),
        AmbienceCue::Sewers => Some("amb2"),
        AmbienceCue::Scrapyards => Some("amb3"),
        AmbienceCue::CrystalCaves => Some("amb4"),
        AmbienceCue::FrozenCity => Some("amb5"),
        AmbienceCue::Labs => Some("amb6"),
        AmbienceCue::Palace => Some("amb7"),
        AmbienceCue::Vault => Some("amb100"),
        AmbienceCue::Oasis => Some("amb101"),
        AmbienceCue::PizzaSewers => Some("amb102"),
        AmbienceCue::City => Some("amb103"),
        AmbienceCue::CursedCaves => Some("amb104"),
        AmbienceCue::Jungle => Some("amb105"),
        AmbienceCue::Hq => Some("amb106"),
    }
}

/// GML `MusCont/Alarm_11.gml:53-56` campfire special case wins, else
/// the `"mus" + string(area)` area bed.
pub fn desired_music_cue(area: AreaId, campfire_present: bool) -> MusicCue {
    if campfire_present {
        MusicCue::Campfire
    } else {
        music_for_area(area)
    }
}

/// GML `MusCont/Alarm_11.gml:55` campfire `amb0b` wins, else the
/// `"amb" + string(area)` area bed.
pub fn desired_ambience_cue(area: AreaId, campfire_present: bool) -> AmbienceCue {
    if campfire_present {
        AmbienceCue::Rest
    } else {
        ambience_for_area(area)
    }
}

/// GML `MusCont/Alarm_1.gml:16` `alarm[3] = 180` frames at 30 fps.
const BOSS_JINGLE_SECS: f32 = 180.0 / 30.0;
/// GML `MusCont/Create_0.gml:10-12`
/// `audio_sound_length(musThemeA) * 30 - 175` on the 48.0 s asset.
const TITLE_A_SECS: f32 = (48.0 * 30.0 - 175.0) / 30.0;
/// Ambience filter target (GML `MusCont/Step_0.gml:4-8`: the filter ducks
/// toward 0.2 while the spiral background or a pause holds, else 1.0).
pub fn amb_filter_target(paused: bool, vortex_suppressed: bool) -> f32 {
    if paused || vortex_suppressed {
        0.2
    } else {
        1.0
    }
}

pub fn step_amb_filter(filter: &mut AmbFilter, target: f32, dt: f32) {
    let step = 3.0 * dt;
    if filter.0 < target {
        filter.0 = (filter.0 + step).min(target);
    } else if filter.0 > target {
        filter.0 = (filter.0 - step).max(target);
    }
}

pub fn update_amb_filter(
    time: Res<SimTime>,
    paused: Res<Paused>,
    spiral: Option<Res<crate::vortex::SpiralCtl>>,
    mut filter: ResMut<AmbFilter>,
) {
    step_amb_filter(
        &mut filter,
        amb_filter_target(paused.0, spiral.is_some()),
        time.delta_secs,
    );
}

/// GML `MusCont` selection law per app state: menus hold the title theme
/// (A→B timer at `Create_0.gml:10-12`, char-select bed `Menu/Create_0.gml:96`),
/// Loading previews the area, InGame runs the `Alarm_11` area bed under the
/// boss-death jingle / boss-dead / big-dog / campfire priority, and Splash runs
/// `Logo/Alarm_0.gml:8` `sndLogoLoop` on the ambience slot - leaving Splash
/// re-evaluates it to `None`, which is the backend stop
/// (`Logo/Destroy_0.gml:1` `snd_stop`).
pub fn sync_area_audio(
    time: Res<SimTime>,
    app_state: Res<AppState>,
    splash: Option<Res<crate::state::SplashState>>,
    run: Res<Run>,
    transition: Res<LoopTransition>,
    floor: Res<FloorTransition>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut state: ResMut<AreaAudioState>,
    campfires: Query<(), With<CampfireProp>>,
    bosses: Query<&Enemy, With<BossBrain>>,
    enemies: Query<&Enemy>,
    pedestals: Query<(), With<CrownPedestal>>,
    players: Query<&RaceState, With<Player>>,
) {
    let dt = time.delta_secs;
    let in_game = *app_state == AppState::InGame;
    let campfire_present = transition.campfire_active || !campfires.is_empty();
    let guardian_present = enemies.iter().any(|e| e.kind == EnemyKind::CrownGuardian);
    let boss_alive_now = !bosses.is_empty() || guardian_present;
    let throne_ii_present =
        transition.throne_ii_alive || bosses.iter().any(|e| e.kind == EnemyKind::ThroneII);
    let player_present = !players.is_empty();
    let bigdog = players.iter().any(|r| r.race == RaceId::BigDog);
    let rogue = players.iter().any(|r| r.race == RaceId::Rogue);

    let room = (run.floor, run.area);
    if state.room != room {
        state.room = room;
        state.boss_dead_pending = false;
        state.silent_until_room = false;
        if bigdog {
            state.boss_dead = false;
        }
    }

    let boss_alive_prev = state.boss_alive;
    state.boss_alive = in_game && boss_alive_now;
    if in_game && boss_alive_prev && !boss_alive_now && !floor.active {
        state.jingle_in = BOSS_JINGLE_SECS;
        state.boss_dead_pending = true;
        let name = if pedestals.is_empty() {
            "sndBossWin"
        } else {
            "sndVaultBossWin"
        };
        cues.push(AudioCue {
            name,
            volume: 1.0,
            variance: 0.0,
        });
    }

    if state.boss_dead && (run.area != state.boss_dead_area || boss_alive_now) {
        state.boss_dead = false;
    }
    if state.jingle_in > 0.0 {
        state.jingle_in -= dt;
        if state.jingle_in <= 0.0 {
            state.jingle_in = 0.0;
            if state.boss_dead_pending {
                state.boss_dead_pending = false;
                if player_present && !matches!(run.area, AreaId::Palace | AreaId::HQ) {
                    state.boss_dead = true;
                    state.boss_dead_area = run.area;
                } else {
                    state.silent_until_room = true;
                }
            }
        }
    }
    if in_game && run.game_over {
        state.boss_dead = false;
        state.silent_until_room = true;
    }
    if !in_game && state.title_in > 0.0 {
        state.title_in = (state.title_in - dt).max(0.0);
    }

    let (wanted_music, wanted_ambience): (Option<MusicCue>, Option<AmbienceCue>) = if !in_game {
        if *app_state == AppState::Loading {
            (
                Some(desired_music_cue(run.area, campfire_present)),
                Some(desired_ambience_cue(run.area, campfire_present)),
            )
        } else {
            let title = if state.title_in > 0.0 {
                MusicCue::TitleThemeA
            } else {
                MusicCue::TitleThemeB
            };
            // GML `Logo/Alarm_0.gml:8`: the loop starts on the seventh
            // gun step; leaving `Splash` re-evaluates this to `None`,
            // which the backend stops (`Logo/Destroy_0.gml:1`
            // `snd_stop(sndLogoLoop)`).
            let logo_loop = *app_state == AppState::Splash
                && splash
                    .as_ref()
                    .is_some_and(|s| s.guns as usize >= crate::state::SPLASH_GUN_STEPS.len());
            let ambience = if logo_loop {
                Some(AmbienceCue::LogoLoop)
            } else if *app_state == AppState::Title {
                Some(AmbienceCue::Menu)
            } else {
                None
            };
            (Some(title), ambience)
        }
    } else {
        let wanted_ambience = if bigdog {
            Some(AmbienceCue::None)
        } else if throne_ii_present {
            Some(AmbienceCue::ThroneII)
        } else {
            Some(desired_ambience_cue(run.area, campfire_present))
        };
        let wanted_music = if run.game_over {
            None
        } else {
            let fallback = if state.silent_until_room {
                None
            } else if state.boss_dead {
                Some(MusicCue::BossDead)
            } else if bigdog {
                Some(MusicCue::RaceBigDog)
            } else if campfire_present {
                Some(MusicCue::Campfire)
            } else if run.area == AreaId::HQ && rogue {
                Some(MusicCue::HqRogue)
            } else {
                Some(music_for_area(run.area))
            };
            if state.jingle_in > 0.0 {
                None
            } else if boss_alive_now {
                boss_music_cue(run.area, throne_ii_present, guardian_present).or(fallback)
            } else {
                fallback
            }
        };
        (wanted_music, wanted_ambience)
    };

    if wanted_music == Some(MusicCue::TitleThemeA)
        && state.current_music != Some(MusicCue::TitleThemeA)
    {
        cues.push(AudioCue {
            name: "sndRestart",
            volume: 1.0,
            variance: 0.0,
        });
    }
    state.current_music = wanted_music;
    state.current_ambience = wanted_ambience;
}

/// Remix layer volumes every tick (GML `MusCont/Step_0.gml:13-14`:
/// song `opt_musvol`, ambience `opt_ambvol * ambfilter`; the backend
/// applies `channels.master` itself).
pub fn tick_area_audio_fades(
    channels: Res<AudioChannels>,
    amb_filter: Res<AmbFilter>,
    save: Option<Res<crate::savedata_part::SaveData>>,
    mut state: ResMut<AreaAudioState>,
) {
    let music_on = if state.current_music.is_some() || state.jingle_in > 0.0 {
        1.0
    } else {
        0.0
    };
    let ambience_on = if state
        .current_ambience
        .is_some_and(|c| c != AmbienceCue::None)
    {
        1.0
    } else {
        0.0
    };
    let ambience_base = save.map_or(channels.music, |s| s.settings.ambience_volume);

    state.music_volume = (channels.music * music_on).clamp(0.0, 1.0);
    state.ambience_volume = (ambience_base * amb_filter.0 * ambience_on).clamp(0.0, 1.0);
}

/// Silence both buses, dropping the current cues and timers (GML
/// `MusCont/Other_5.gml:7-8` stops `song` and `amb` on room end).
pub fn reset_area_audio(mut state: ResMut<AreaAudioState>) {
    *state = AreaAudioState::default();
}

/// Head init for the area-audio resources the schedule's first tick
/// reads as bare `Res`/`ResMut` (`sync_area_audio`,
/// `update_amb_filter`, `tick_area_audio_fades`), plus the per-step
/// mainvol recovery GML runs at `UberCont/Step_0.gml:99-103` before
/// the step's ducks (`lerp(mainvol, 1, 0.4)` at 30 Hz).
pub fn init_area_audio_resources(world: &mut World) {
    world.init_resource::<AreaAudioState>();
    world.init_resource::<AmbFilter>();
    world.init_resource::<Queue<AudioCue>>();
    world.init_resource::<MainVol>();
    let dt = world
        .get_resource::<SimTime>()
        .map_or(0.0, |t| t.delta_secs);
    world.resource_mut::<MainVol>().step(dt);
}

/// Headless menu-action mirror (variant shapes kept so the mapping below
/// ports the GML sites verbatim; the real menu slice owns the canonical enum
/// later).
#[derive(Clone, Debug)]
pub enum UiAction {
    StartGame,
    MainMenuPlay,
    OpenSettings,
    OpenCredits,
    CloseOverlay,
    Resume,
    QuitToTitle,
    QuitApp,
    SetMasterVol(f32),
    SetSfxVol(f32),
    SetMusicVol(f32),
    SetAmbienceVol(f32),
    SaveSettings,
    NextLanguage,
    SetLanguage(String),
    SettingsCategory(u8),
    SettingsBack,
    ShowPauseConfirm(u8),
    CancelPauseConfirm,
    ConfirmPause(u8),
    SelectCharacter(usize),
    SelectSkin(u8),
    ToggleLoadout,
    ToggleHardmode,
    CycleStartWeapon(i8),
    CycleStoredWeapon(i8),
    CycleCrown(i8),
    SelectCrown(u8),
    SelectMutation(usize),
    PickMutation(usize),
    SettingToggle(String),
    SettingSlider {
        key: String,
        value: f32,
    },
    SettingCycle {
        key: String,
        dir: i8,
    },
    SettingInput {
        key: String,
        value: String,
    },
    SettingResetOptions,
    SettingEraseProgress,
    SettingViewCredits,
    SettingOpenSubcategory(u8),
    /// Open the run-stats panel (GML `MainMenuButton/Other_10.gml:93,95`
    /// creates `DrawStats` and plays `sndMenuStats`).
    ShowStats,
    /// Fire one PLAY-submenu row (GML `PlayButton` num 0 NORMAL, 1 DAILY,
    /// 2 WEEKLY, 3 HARD, 4 CUSTOM). NORMAL/HARD proceed to character
    /// select; DAILY/WEEKLY/CUSTOM need unported online/custom systems.
    PlaySubmenu(u8),
    /// Close the PLAY submenu (GML `BackButton` over the PlayButtons).
    ClosePlaySubmenu,
    /// Advance the credits section (GML `Credits` click `_force`).
    AdvanceCredits,
    /// Dismiss the head of the unlock queue (GML `UnlockScreen`
    /// `Mouse_56` click once `can_continue` fires).
    DismissUnlock,
    /// Arm a REMAP capture for the named control (`fire`, `spec`,
    /// `swap`, `pick`, `north`, `south`, `west`, `east`). The next
    /// pressed key/mouse button becomes the keyboard-side binding.
    RemapControl(String),
    /// Restore GML `scrKeymapsSetup` defaults on both sides.
    RemapReset,
}

/// Bridged menu action (headless bridge message -> [`Queue`]).
#[derive(Clone, Debug)]
pub struct UiBridgeAction(pub UiAction);

/// UI one-shot map; every GML site behind these arms is a plain
/// `snd_play(stem)` - gain 1.0, no pitch jitter.
/// Context-free actions only; character/skin/crown/mutation picks need
/// site context (see the `*_sfx` helpers below) and sliders commit per
/// change.
pub fn ui_action_sfx(action: &UiAction) -> Vec<AudioCue> {
    let mut out = Vec::new();
    let mut push = |stem: &'static str| {
        out.push(AudioCue {
            name: stem,
            volume: 1.0,
            variance: 0.0,
        });
    };
    match action {
        UiAction::MainMenuPlay => {
            push("sndClick");
            push("sndMenuCharSelect");
        }
        UiAction::PlaySubmenu(0) | UiAction::PlaySubmenu(3) => {
            push("sndMenuCharSelect");
            push("sndClick");
        }
        UiAction::PlaySubmenu(_) => {
            push("sndClick");
        }
        UiAction::ClosePlaySubmenu => {
            push("sndClickBack");
        }
        UiAction::AdvanceCredits => {
            // GML `Credits/Step_0` advances on click with no named
            // sting (the section change itself is the feedback).
        }
        UiAction::DismissUnlock => {
            // GML `UnlockScreen/Mouse_56` dismisses on click with no
            // named sting (the FIFO chain advancing is the feedback).
        }
        UiAction::RemapControl(_) => {
            // GML `MenuOptions/Other_10.gml:703` generic option click
            // on arming; the capture resolve sting (`sndSliderLetGo`,
            // `MenuOptions/Other_10.gml:401`) fires on capture.
            push("sndClick");
        }
        UiAction::RemapReset => {
            push("sndRestart");
        }
        UiAction::QuitToTitle => {
            push("sndClickBack");
        }
        UiAction::QuitApp => {
            push("sndClick");
        }
        // GML `PauseButton/Other_10` plays NO transition sound on
        // MENU/RETRY/SETTINGS/CONTINUE/BACK/QUIT (hover `sndHover`
        // only) - the confirm swap is silent.
        UiAction::ShowPauseConfirm(_)
        | UiAction::CancelPauseConfirm
        | UiAction::ConfirmPause(_) => {}
        UiAction::SettingToggle(_)
        | UiAction::SettingCycle { .. }
        | UiAction::SettingInput { .. } => {
            push("sndClick");
        }
        UiAction::SetMasterVol(_)
        | UiAction::SetSfxVol(_)
        | UiAction::SetMusicVol(_)
        | UiAction::SetAmbienceVol(_) => {
            push("sndSliderLetGo");
        }
        // GML `MainMenuButton/Other_10.gml:84-85` settings case.
        UiAction::OpenSettings => {
            push("sndClick");
            push("sndMenuOptions");
        }
        // GML `MenuOptions/Other_20.gml:250` ViewCredits click.
        UiAction::OpenCredits => {
            push("sndMenuCredits");
        }
        // GML `BackButton/Other_10.gml:201` unconditional tail
        // `snd_play(sndClickBack)` - the close path either way.
        UiAction::SettingsBack | UiAction::CloseOverlay | UiAction::SaveSettings => {
            push("sndClickBack");
        }
        // GML `MenuOptions/Other_20.gml:630` language row click.
        UiAction::NextLanguage => {
            push("sndClick");
        }
        UiAction::SetLanguage(_) => {
            push("sndClick");
        }
        // GML `scrOptionsMenu.gml:215`: Main category opens the options
        // sting, every other category clicks.
        UiAction::SettingsCategory(0) => {
            push("sndMenuOptions");
        }
        UiAction::SettingsCategory(_) => {
            push("sndClick");
        }
        UiAction::SettingSlider { .. } => {
            push("sndSliderLetGo");
        }
        UiAction::SettingResetOptions | UiAction::SettingEraseProgress => {
            push("sndClick");
        }
        UiAction::SettingViewCredits => {
            push("sndMenuCredits");
        }
        UiAction::ShowStats => {
            push("sndMenuStats");
        }
        UiAction::SettingOpenSubcategory(_) => {
            push("sndClick");
        }
        _ => {}
    }
    out
}

/// GML `scr_race_get_sound(race, "Slct")` verbatim: `sndMutant{N}Slct`
/// by GML race id (= port `RaceId` discriminant, Random = 0);
/// Skeleton (14, no Slct asset) falls back to `sndBloodGamble`.
pub fn race_select_sfx(race: crate::data::RaceId) -> AudioCue {
    let name = match race as u8 {
        14 => "sndBloodGamble",
        0 => "sndMutant0Slct",
        1 => "sndMutant1Slct",
        2 => "sndMutant2Slct",
        3 => "sndMutant3Slct",
        4 => "sndMutant4Slct",
        5 => "sndMutant5Slct",
        6 => "sndMutant6Slct",
        7 => "sndMutant7Slct",
        8 => "sndMutant8Slct",
        9 => "sndMutant9Slct",
        10 => "sndMutant10Slct",
        11 => "sndMutant11Slct",
        12 => "sndMutant12Slct",
        13 => "sndMutant13Slct",
        15 => "sndMutant15Slct",
        16 => "sndMutant16Slct",
        _ => "sndMutant0Slct",
    };
    AudioCue {
        name,
        volume: 1.0,
        variance: 0.0,
    }
}

/// GML `scrRunStart.gml:39`
/// `snd_play(scr_race_get_sound(race, "Cnfm", sndMutant0Cnfm))` verbatim:
/// `sndMutant{N}Cnfm` by GML race id (= port `RaceId` discriminant,
/// Random = 0); Skeleton (14) has no Cnfm asset and maps to
/// `sndMutant14Turn`.
pub fn race_confirm_sfx(race: crate::data::RaceId) -> AudioCue {
    let name = match race as u8 {
        14 => "sndMutant14Turn",
        0 => "sndMutant0Cnfm",
        1 => "sndMutant1Cnfm",
        2 => "sndMutant2Cnfm",
        3 => "sndMutant3Cnfm",
        4 => "sndMutant4Cnfm",
        5 => "sndMutant5Cnfm",
        6 => "sndMutant6Cnfm",
        7 => "sndMutant7Cnfm",
        8 => "sndMutant8Cnfm",
        9 => "sndMutant9Cnfm",
        10 => "sndMutant10Cnfm",
        11 => "sndMutant11Cnfm",
        12 => "sndMutant12Cnfm",
        13 => "sndMutant13Cnfm",
        15 => "sndMutant15Cnfm",
        16 => "sndMutant16Cnfm",
        _ => "sndMutant0Cnfm",
    };
    AudioCue {
        name,
        volume: 1.0,
        variance: 0.0,
    }
}

/// GML `scrCampfireMenuCreate.gml:887-892` skin pick: `sndMenuCSkin`
/// at `random_range(0.95, 1.05)`, `sndMenuBSkin` at exactly 1
/// (`_skin_id == SkinLetter.B`), `sndMenuASkin` at `0.95 + random(0.1)`
/// - pitch jitter of ±0.05 either way, i.e. variance 0.1.
pub fn skin_select_sfx(skin: u8) -> AudioCue {
    let (name, variance) = match skin {
        2 => ("sndMenuCSkin", 0.1),
        1 => ("sndMenuBSkin", 0.0),
        _ => ("sndMenuASkin", 0.1),
    };
    AudioCue {
        name,
        volume: 1.0,
        variance,
    }
}

/// Denial sting (GML `CharSelect/Mouse_4.gml:13` `snd_play(sndNoSelect)`).
pub fn denied_sfx() -> AudioCue {
    AudioCue {
        name: "sndNoSelect",
        volume: 1.0,
        variance: 0.0,
    }
}

/// Hover sting (GML `MainMenuButton/Step_0.gml:21`,
/// `SkillIcon/Mouse_4.gml:12`, `SkillIcon/Mouse_10.gml:9`
/// `snd_play(sndHover)`).
pub fn hover_sfx() -> AudioCue {
    AudioCue {
        name: "sndHover",
        volume: 1.0,
        variance: 0.0,
    }
}

/// Crown pick sting (GML `scrCampfireMenuCreate.gml:822`
/// `snd_play(sndMenuCrown, 0.95 + random(0.1))` - ±0.05 pitch jitter).
pub fn crown_select_sfx() -> AudioCue {
    AudioCue {
        name: "sndMenuCrown",
        volume: 1.0,
        variance: 0.1,
    }
}
