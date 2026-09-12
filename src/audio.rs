//! Audio selection: cues, area music/ambience, reactive stingers.
//!
//! The bevy build spawned `AudioPlayer` entities straight from systems
//! and resolved files through `AssetCatalog` + `AssetServer`; here
//! systems output DATA and the platform layer (repame-audio) plays it:
//!
//! * one-shots ([`AudioCue`]) queue with stem + volume + pitch variance;
//!   the backend samples the pitch at play time (bevy
//!   `AudioM::play_sfx_varied` parity) and resolves
//!   [`resolve_sfx_path`] against its asset store.
//! * reactive stingers ([`ReactiveCue`]) arrive as
//!   [`ReactiveAudioRequest`]s, pass bus/throttle/selection in
//!   [`play_reactive_audio_requests`], and land as
//!   [`ResolvedReactiveCue`]s in a backend-neutral queue the audio
//!   backend drains (no `Handle<AudioSource>`, no `AssetServer`).
//! * area music/ambience (`music_for_area`, `ambience_for_area`,
//!   `boss_music_for_kind`) resolve to first-candidate stems with
//!   fade gains in [`AreaAudioState`], which the backend polls.
//! * the combat-intensity layer ([`CombatIntensityLayer`]) carries the
//!   smoothed `current` per area; the backend mixes it with
//!   [`intensity_bus`].
//!
//! Only the cues needed by ported systems exist yet for gameplay SFX;
//! the full GML stem table for `GameAudio` lands below with the
//! weapon-fire dispatch.

use std::collections::{HashMap, HashSet};

use bevy_ecs::prelude::*;
use repame_sim::SimTime;

use crate::comps_a::{GameCleanup, Health, Player, Run};
use crate::comps_b::{BossBrain, CampfireProp, Enemy, LoopTransition};
use crate::data::{AreaId, EnemyKind};
use crate::msg::Queue;
use crate::state::{AppState, Paused};
use crate::time::{GTimer, TimerMode};

/// One fire-and-forget sound with playback variation.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioCue {
    pub name: &'static str,
    pub volume: f32,
    pub variance: f32,
}

/// Runtime mix buses. Mirrors bevy `AudioChannels` defaults
/// (master 1, sfx 1, music 0.8, ui 1); the settings->channel sync from
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
            music: 0.8,
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

/// Stem -> path resolution (bevy `resolve_sfx` selection half).
/// The catalog existence scan lives backend-side (the headless sim has
/// no filesystem); the law kept here is the deterministic fallback:
/// `audio/{stem}.wav` when no `audio| sounds/{stem}.{ogg,wav,mp3,flac}`
/// hit exists.
pub fn resolve_sfx_path(stem: &str) -> String {
    format!("audio/{stem}.wav")
}

/// Reactive music/stinger requests (nt `ReactiveCue` parity, full
/// variant list — the audio layer resolves them).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ReactiveCue {
    LevelUp,
    MutationChosen,
    UltraChosen,
    BossAppear,
    BossDefeated,
    PlayerCritical,
    PlayerDeath,
    PortalOpen,
    PortalEnter,
    SecretFound,
    WeaponPickup,
    ChestOpen,
    LoopComplete,
    ThroneRises,
    IdpdIncoming,

    Kill,
    KillStreak,

    UiClick,
    UiBack,
    UiConfirm,
    UiCycle,
}

/// Queued reactive cue marker (drained by the audio layer).
#[derive(bevy_ecs::prelude::Component, Clone, Copy, Debug)]
pub struct QueuedReactiveCue(pub ReactiveCue);

/// In-sim reactive request (bevy `Message` -> [`Queue`] channel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReactiveAudioRequest {
    pub cue: ReactiveCue,
}

impl ReactiveAudioRequest {
    pub fn new(cue: ReactiveCue) -> Self {
        Self { cue }
    }
}

/// Backend-neutral resolved stinger: first-candidate stem plus the
/// mixed volume. The backend verifies file existence and plays it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedReactiveCue {
    pub cue: ReactiveCue,
    pub path: &'static str,
    pub volume: f32,
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

    /// Player-hurt sting (bevy `play_hurt`: sndPlayerHit, 0.7, 0.05).
    pub fn play_hurt(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndPlayerHit", 0.7, 0.05);
    }

    /// Hit thock (bevy `play_hit`: sndHitWall, 0.45, 0.15).
    pub fn play_hit(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndHitWall", 0.45, 0.15);
    }

    /// Explosion boom (bevy `play_boom`: sndExplosionL, 0.9, 0.04).
    pub fn play_boom(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndExplosionL", 0.9, 0.04);
    }

    /// Level-up jingle (bevy `play_levelup`: sndLevelUp, 0.8, 0.03).
    pub fn play_levelup(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndLevelUp", 0.8, 0.03);
    }

    /// Pickup blip (bevy `play_pickup`: sndAmmoPickup, 0.5, 0.15).
    pub fn play_pickup(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndAmmoPickup", 0.5, 0.15);
    }

    /// Portal whoosh (bevy `play_portal`: sndPortalOpen, 0.7, 0.05).
    pub fn play_portal(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndPortalOpen", 0.7, 0.05);
    }

    /// Player-death sting (bevy `play_death`: sndPlayerDeath, 0.9, 0.02).
    pub fn play_death(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndPlayerDeath", 0.9, 0.02);
    }

    // --- Full GML stem table (bevy `GameAudio::load` selection half).
    // Stems are the `snd*` asset names; volume/var mirror the bevy
    // `play_*` methods exactly. The pre-existing short-name cues above
    // predate this slice and are left untouched.

    /// Revolver/pop fire (bevy `play_shoot`: sndPistol, 0.5, 0.12).
    pub fn play_shoot(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndPistol", 0.5, 0.12);
    }

    /// Machinegun chatter (bevy `play_machine`: sndMachinegun, 0.4, 0.15).
    pub fn play_machine(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndMachinegun", 0.4, 0.15);
    }

    /// Shotgun blast (bevy `play_shotgun`: sndShotgun, 0.6, 0.1).
    pub fn play_shotgun(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndShotgun", 0.6, 0.1);
    }

    /// Crossbow bolt (bevy `play_bolt`: sndCrossbow, 0.5, 0.08).
    pub fn play_bolt(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndCrossbow", 0.5, 0.08);
    }

    /// Melee swing (bevy `play_melee`: sndHammer, 0.5, 0.1).
    pub fn play_melee(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndHammer", 0.5, 0.1);
    }

    /// Explosion crack (bevy `play_explode`: sndExplosion, 0.7, 0.06).
    pub fn play_explode(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndExplosion", 0.7, 0.06);
    }

    /// Chest open (bevy `play_chest`: sndChest, 0.6, 0.05).
    pub fn play_chest(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndChest", 0.6, 0.05);
    }

    /// Weapon chest (bevy `play_weapon_chest`: sndWeaponChest, 0.6, 0.05).
    pub fn play_weapon_chest(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndWeaponChest", 0.6, 0.05);
    }

    /// Ammo chest (bevy `play_ammo_chest`: sndAmmoChest, 0.6, 0.05).
    pub fn play_ammo_chest(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndAmmoChest", 0.6, 0.05);
    }

    /// Health chest (GML `sndHealthChest` / `sndHealthChestBig` with
    /// Second Stomach, 0.6, 0.05).
    pub fn play_health_chest(&self, cues: &mut Queue<AudioCue>, big: bool) {
        Self::cue(
            cues,
            if big {
                "sndHealthChestBig"
            } else {
                "sndHealthChest"
            },
            0.6,
            0.05,
        );
    }

    /// Cursed big chest (GML `sndBigCursedChest` via `snd_play_hit`).
    pub fn play_cursed_chest(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndBigCursedChest", 0.6, 0.05);
    }

    /// Pickup fade-out (bevy `play_pickup_disappear`: sndPickupDisappear,
    /// 0.4, 0.1).
    pub fn play_pickup_disappear(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndPickupDisappear", 0.4, 0.1);
    }

    /// Dry fire (bevy `play_empty`: sndEmpty, 0.6, 0.05).
    pub fn play_empty(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndEmpty", 0.6, 0.05);
    }

    /// Ultra dry fire (bevy `play_ultra_empty`: sndUltraEmpty, 0.6, 0.05).
    pub fn play_ultra_empty(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndUltraEmpty", 0.6, 0.05);
    }

    /// Melee flip (bevy `play_melee_flip`: sndMeleeFlip, 0.5, 0.08).
    pub fn play_melee_flip(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndMeleeFlip", 0.5, 0.08);
    }

    /// Crossbow reload (bevy `play_cross_reload`: sndCrossReload, 0.5, 0.08).
    pub fn play_cross_reload(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndCrossReload", 0.5, 0.08);
    }

    /// Shotgun reload (bevy `play_shot_reload`: sndShotReload, 0.5, 0.08).
    pub fn play_shot_reload(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndShotReload", 0.5, 0.08);
    }

    /// Grenade reload (bevy `play_nade_reload`: sndNadeReload, 0.5, 0.08).
    pub fn play_nade_reload(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndNadeReload", 0.5, 0.08);
    }

    /// Plasma reload (bevy `play_plasma_reload`: sndPlasmaReload, 0.5, 0.08).
    pub fn play_plasma_reload(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndPlasmaReload", 0.5, 0.08);
    }

    /// Lightning reload (bevy `play_lightning_reload`:
    /// sndLightningReload, 0.5, 0.08).
    pub fn play_lightning_reload(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndLightningReload", 0.5, 0.08);
    }

    /// Sniper acquire (bevy `play_sniper_target`: sndSniperTarget, 0.6, 0.05).
    pub fn play_sniper_target(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndSniperTarget", 0.6, 0.05);
    }

    /// Sniper shot (bevy `play_sniper_fire`: sndSniperFire, 0.5, 0.08).
    pub fn play_sniper_fire(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndSniperFire", 0.5, 0.08);
    }

    /// Assassin lunge (bevy `play_assassin_attack`: sndAssassinAttack,
    /// 0.6, 0.05).
    pub fn play_assassin_attack(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndAssassinAttack", 0.6, 0.05);
    }

    /// Laser-crystal charge (bevy `play_laser_charge`:
    /// sndLaserCrystalCharge, 0.5, 0.05).
    pub fn play_laser_charge(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndLaserCrystalCharge", 0.5, 0.05);
    }

    /// Lightning-crystal charge (bevy `play_lightning_charge`:
    /// sndLightningCrystalCharge, 0.5, 0.05).
    pub fn play_lightning_charge(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndLightningCrystalCharge", 0.5, 0.05);
    }

    /// Snowtank aim (bevy `play_snowtank_aim`: sndSnowTankAim, 0.6, 0.05).
    pub fn play_snowtank_aim(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndSnowTankAim", 0.6, 0.05);
    }

    /// Gold-tank aim (bevy `play_goldtank_aim`: sndGoldTankAim, 0.6, 0.05).
    pub fn play_goldtank_aim(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndGoldTankAim", 0.6, 0.05);
    }

    /// Explo-guardian charge (bevy `play_explo_charge`:
    /// sndExploGuardianCharge, 0.6, 0.05).
    pub fn play_explo_charge(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndExploGuardianCharge", 0.6, 0.05);
    }

    /// Mimic slurp (bevy `play_mimic_slurp`: sndMimicSlurp, 0.6, 0.05).
    pub fn play_mimic_slurp(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndMimicSlurp", 0.6, 0.05);
    }

    /// IDPD van warning (bevy `play_van_warning`: sndVanWarning, 0.7, 0.05).
    pub fn play_van_warning(&self, cues: &mut Queue<AudioCue>) {
        Self::cue(cues, "sndVanWarning", 0.7, 0.05);
    }

    /// GML weapon-name -> (stem, volume, var) dispatch (bevy
    /// `play_weapon_fire` selection half, verbatim branch order).
    pub fn weapon_fire_cue(weapon_name: &str, underwater: bool) -> (&'static str, f32, f32) {
        if underwater {
            return ("sndOasisShoot", 0.5, 0.1);
        }
        let n = weapon_name;
        let is_gold = n.contains("GOLDEN") || n.contains("GOLD ") || n.starts_with("GOLD");
        if is_gold {
            if n.contains("PISTOL") || n.contains("REVOLVER") {
                return ("sndGoldPistol", 0.5, 0.1);
            } else if n.contains("MACHINEGUN") || n.contains("SMG") || n.contains("MINIGUN") {
                return ("sndGoldMachinegun", 0.5, 0.1);
            } else if n.contains("SHOTGUN") || n.contains("ERASER") {
                return ("sndGoldShotgun", 0.5, 0.1);
            } else if n.contains("CROSSBOW") || n.contains("XBOW") {
                return ("sndGoldCrossbow", 0.5, 0.1);
            } else if n.contains("GRENADE")
                || n.contains("ROCKET")
                || n.contains("NUKE")
                || n.contains("BAZOOKA")
            {
                return ("sndGoldGrenade", 0.5, 0.1);
            } else if n.contains("PLASMA") {
                return ("sndGoldPlasma", 0.5, 0.1);
            } else if n.contains("LASER") || n.contains("ION") {
                return ("sndGoldLaser", 0.5, 0.1);
            }
        }
        if n.contains("PLASMA") || n.contains("DEVASTATOR") || n == "GUN GUN" {
            ("sndPlasma", 0.55, 0.08)
        } else if n.contains("LASER") || n.contains("ION") {
            ("sndLaser", 0.55, 0.08)
        } else if n.contains("LIGHTNING") {
            ("sndLightningPistol", 0.55, 0.08)
        } else if n.contains("FLAME")
            || n.contains("DRAGON")
            || n.contains("FLARE")
            || n.contains("INCINERATOR")
        {
            ("sndFlameCannon", 0.55, 0.08)
        } else if n.contains("DISC") || n.contains("BOUNCER") {
            ("sndDiscgun", 0.55, 0.08)
        } else if n.contains("SLUGGER") {
            ("sndSlugger", 0.6, 0.08)
        } else if n.contains("SPLINTER") || n.contains("SEEKER") || n.contains("TOXIC") {
            ("sndSplinterGun", 0.5, 0.08)
        } else if n.contains("GRENADE")
            || n.contains("BAZOOKA")
            || n.contains("NUKE")
            || n.contains("ROCKET")
            || n.contains("CLUSTER")
            || n.contains("BLOOD")
            || n.contains("FLAK")
            || n.contains("NADER")
        {
            ("sndGrenade", 0.6, 0.08)
        } else if n.contains("CROSSBOW") || n.contains("HEAVY XBOW") {
            ("sndCrossbow", 0.5, 0.08)
        } else if n.contains("SHOTGUN")
            || n.contains("ERASER")
            || n.contains("WAVE")
            || n.contains("SLUGGER")
        {
            ("sndShotgun", 0.6, 0.1)
        } else if n.contains("MACHINEGUN")
            || n.contains("SMG")
            || n.contains("MINIGUN")
            || n.contains("ASSAULT")
            || n.contains("QUAD")
            || n.contains("POP RIFLE")
            || n.contains("ROGUE")
            || n.contains("HEAVY")
        {
            ("sndMachinegun", 0.4, 0.15)
        } else if n.contains("REVOLVER")
            || n.contains("PISTOL")
            || n.contains("SMART")
            || n.contains("POP GUN")
            || n.contains("FROG")
        {
            ("sndPistol", 0.5, 0.12)
        } else if n.contains("SENTRY") {
            ("sndMachinegun", 0.4, 0.15)
        } else {
            ("sndPistol", 0.5, 0.12)
        }
    }

    /// Weapon fire one-shot (bevy `play_weapon_fire`).
    pub fn play_weapon_fire(&self, cues: &mut Queue<AudioCue>, weapon_name: &str) {
        let (stem, vol, var) = Self::weapon_fire_cue(weapon_name, false);
        Self::cue(cues, stem, vol, var);
    }

    /// GML `snd_play_gun` tiers with underwater override (bevy
    /// `play_weapon_fire_gml`).
    pub fn play_weapon_fire_gml(
        &self,
        cues: &mut Queue<AudioCue>,
        weapon_name: &str,
        underwater: bool,
    ) {
        let (stem, vol, var) = Self::weapon_fire_cue(weapon_name, underwater);
        Self::cue(cues, stem, vol, var);
    }
}

// --- Area music / ambience selection (bevy `ambience.rs` port) ---

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
    City,
    Campfire,

    BossBigBandit,
    BossBigDog,
    BossLilHunter,
    BossThrone,
    BossThroneII,
    TitleTheme,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum AmbienceCue {
    None,
    DesertWind,
    SewerDrip,
    ScrapHum,
    CrystalHum,
    FrozenWind,
    LabBuzz,
    PalaceFire,
    OasisBreeze,
    JungleBugs,
    VaultHum,
    HqSirens,
    CampfireCrackle,
    CityNoise,
}

/// Current area loop selection. The backend polls this resource: on a
/// cue change it crossfades from its previous stem to the new
/// [`music_path`]/[`ambience_path`] over the bevy fade durations;
/// `music_volume`/`ambience_volume` already include pause dim,
/// channel mix, ambience scale/filter, and the switch fades.
#[derive(Resource, Debug)]
pub struct AreaAudioState {
    pub current_music: Option<MusicCue>,
    pub current_ambience: Option<AmbienceCue>,
    pub music_gain: f32,
    pub ambience_gain: f32,
    pub music_volume: f32,
    pub ambience_volume: f32,
    pub music_fade: AreaAudioFader,
    pub ambience_fade: AreaAudioFader,
}

impl Default for AreaAudioState {
    fn default() -> Self {
        Self {
            current_music: None,
            current_ambience: None,
            music_gain: 0.0,
            ambience_gain: 0.0,
            music_volume: 0.0,
            ambience_volume: 0.0,
            music_fade: AreaAudioFader::default(),
            ambience_fade: AreaAudioFader::default(),
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

/// Headless fade voice (bevy `AreaAudioFader` minus the entity
/// despawn flag — a single gain per bus replaces the overlapping
/// fade-out/fade-in entity pair).
#[derive(Clone, Copy, Debug)]
pub struct AreaAudioFader {
    pub start: f32,
    pub end: f32,
    pub timer: GTimer,
}

impl Default for AreaAudioFader {
    fn default() -> Self {
        Self {
            start: 0.0,
            end: 0.0,
            timer: GTimer::from_seconds(0.0, TimerMode::Once),
        }
    }
}

impl AreaAudioFader {
    pub fn value(&self) -> f32 {
        self.start + (self.end - self.start) * self.timer.fraction()
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

        AreaId::Loop => MusicCue::Desert,
    }
}

pub fn ambience_for_area(area: AreaId) -> AmbienceCue {
    match area {
        AreaId::Desert => AmbienceCue::DesertWind,
        AreaId::Sewers => AmbienceCue::SewerDrip,
        AreaId::Scrapyards => AmbienceCue::ScrapHum,
        AreaId::CrystalCaves => AmbienceCue::CrystalHum,
        AreaId::FrozenCity => AmbienceCue::FrozenWind,
        AreaId::Labs => AmbienceCue::LabBuzz,
        AreaId::Palace => AmbienceCue::PalaceFire,

        AreaId::Oasis => AmbienceCue::OasisBreeze,
        AreaId::PizzaSewers => AmbienceCue::SewerDrip,
        AreaId::CursedCaves => AmbienceCue::CrystalHum,
        AreaId::Jungle => AmbienceCue::JungleBugs,
        AreaId::Vault => AmbienceCue::VaultHum,
        AreaId::CrownVault => AmbienceCue::VaultHum,
        AreaId::HQ => AmbienceCue::HqSirens,
        AreaId::City => AmbienceCue::CityNoise,
        AreaId::Campfire => AmbienceCue::CampfireCrackle,

        AreaId::Loop => AmbienceCue::DesertWind,
    }
}

pub fn boss_music_for_kind(kind: EnemyKind) -> Option<MusicCue> {
    match kind {
        EnemyKind::BigBandit | EnemyKind::BigBanditLoop => Some(MusicCue::BossBigBandit),
        EnemyKind::BigDog | EnemyKind::BigDogLoop => Some(MusicCue::BossBigDog),
        EnemyKind::LilHunter | EnemyKind::LilHunterLoop => Some(MusicCue::BossLilHunter),
        EnemyKind::Throne => Some(MusicCue::BossThrone),
        EnemyKind::ThroneII => Some(MusicCue::BossThroneII),
        _ => None,
    }
}

pub fn music_candidates(cue: MusicCue) -> &'static [&'static str] {
    match cue {
        MusicCue::Desert => &[
            "audio/mus1.ogg",
            "audio/mus1b.ogg",
            "audio/music/desert.ogg",
            "audio/music/musDesert.ogg",
            "sounds/music/desert.ogg",
        ],
        MusicCue::Sewers => &[
            "audio/mus2.ogg",
            "audio/music/sewers.ogg",
            "audio/music/musSewers.ogg",
            "sounds/music/sewers.ogg",
        ],
        MusicCue::Scrapyards => &[
            "audio/mus3.ogg",
            "audio/mus3b.ogg",
            "audio/music/scrapyards.ogg",
            "audio/music/musScrapyards.ogg",
            "sounds/music/scrapyards.ogg",
        ],
        MusicCue::CrystalCaves => &[
            "audio/mus4.ogg",
            "audio/music/crystal_caves.ogg",
            "audio/music/crystalcaves.ogg",
            "audio/music/musCrystal.ogg",
            "sounds/music/crystal_caves.ogg",
        ],
        MusicCue::FrozenCity => &[
            "audio/mus5.ogg",
            "audio/mus5b.ogg",
            "audio/music/frozen_city.ogg",
            "audio/music/frozencity.ogg",
            "audio/music/musFrozen.ogg",
            "sounds/music/frozen_city.ogg",
        ],
        MusicCue::Labs => &[
            "audio/mus6.ogg",
            "audio/music/labs.ogg",
            "audio/music/musLabs.ogg",
            "sounds/music/labs.ogg",
        ],
        MusicCue::Palace => &[
            "audio/mus7.ogg",
            "audio/mus7b.ogg",
            "audio/music/palace.ogg",
            "audio/music/musPalace.ogg",
            "sounds/music/palace.ogg",
        ],
        MusicCue::Oasis => &[
            "audio/mus100.ogg",
            "audio/mus100b.ogg",
            "audio/music/oasis.ogg",
            "audio/music/musOasis.ogg",
            "sounds/music/oasis.ogg",
        ],
        MusicCue::PizzaSewers => &[
            "audio/mus101.ogg",
            "audio/music/pizza_sewers.ogg",
            "audio/music/pizzasewers.ogg",
            "audio/music/musPizza.ogg",
            "sounds/music/pizza_sewers.ogg",
        ],
        MusicCue::CursedCaves => &[
            "audio/mus102.ogg",
            "audio/music/cursed_caves.ogg",
            "audio/music/cursedcaves.ogg",
            "audio/music/musCursed.ogg",
            "sounds/music/cursed_caves.ogg",
        ],
        MusicCue::Jungle => &[
            "audio/mus103.ogg",
            "audio/music/jungle.ogg",
            "audio/music/musJungle.ogg",
            "sounds/music/jungle.ogg",
        ],
        MusicCue::Vault => &[
            "audio/mus104.ogg",
            "audio/music/vault.ogg",
            "audio/music/musVault.ogg",
            "sounds/music/vault.ogg",
        ],
        MusicCue::CrownVault => &[
            "audio/mus105.ogg",
            "audio/music/crown_vault.ogg",
            "audio/music/crownvault.ogg",
            "audio/music/musCrownVault.ogg",
            "sounds/music/crown_vault.ogg",
        ],
        MusicCue::Hq => &[
            "audio/mus106.ogg",
            "audio/mus106b.ogg",
            "audio/music/hq.ogg",
            "audio/music/idpd_hq.ogg",
            "audio/music/musHq.ogg",
            "sounds/music/hq.ogg",
        ],
        MusicCue::City => &[
            "audio/mus107.ogg",
            "audio/music/city.ogg",
            "audio/music/yv_mansion.ogg",
            "audio/music/musCity.ogg",
            "sounds/music/city.ogg",
        ],
        MusicCue::Campfire => &[
            "audio/musBoss4Silence.ogg",
            "audio/musboss4silence.ogg",
            "audio/music/campfire.ogg",
            "audio/music/rest.ogg",
            "audio/music/musCampfire.ogg",
            "sounds/music/campfire.ogg",
        ],
        MusicCue::TitleTheme => &[
            "audio/musthemea.ogg",
            "audio/musThemeA.ogg",
            "audio/musthemeb.ogg",
            "audio/musThemeB.ogg",
            "audio/musthemep.ogg",
            "audio/music/title.ogg",
            "audio/music/musThemeA.ogg",
            "sounds/music/title.ogg",
        ],
        MusicCue::BossBigBandit => &[
            "audio/musBoss1.ogg",
            "audio/musboss1.ogg",
            "audio/music/boss_big_bandit.ogg",
            "audio/music/big_bandit.ogg",
            "audio/music/musBossBandit.ogg",
            "sounds/music/boss_big_bandit.ogg",
        ],
        MusicCue::BossBigDog => &[
            "audio/musBoss2.ogg",
            "audio/musboss2.ogg",
            "audio/music/boss_big_dog.ogg",
            "audio/music/big_dog.ogg",
            "audio/music/musBossDog.ogg",
            "sounds/music/boss_big_dog.ogg",
        ],
        MusicCue::BossLilHunter => &[
            "audio/musBoss3.ogg",
            "audio/musboss3.ogg",
            "audio/music/boss_lil_hunter.ogg",
            "audio/music/lil_hunter.ogg",
            "audio/music/musBossHunter.ogg",
            "sounds/music/boss_lil_hunter.ogg",
        ],
        MusicCue::BossThrone => &[
            "audio/musBoss4A.ogg",
            "audio/musBoss4B.ogg",
            "audio/musboss4a.ogg",
            "audio/music/boss_throne.ogg",
            "audio/music/throne.ogg",
            "audio/music/musThrone.ogg",
            "sounds/music/boss_throne.ogg",
        ],
        MusicCue::BossThroneII => &[
            "audio/musBoss5.ogg",
            "audio/musboss5.ogg",
            "audio/musBoss6.ogg",
            "audio/music/boss_throne_ii.ogg",
            "audio/music/throne_ii.ogg",
            "audio/music/musThrone2.ogg",
            "sounds/music/boss_throne_ii.ogg",
        ],
    }
}

pub fn ambience_candidates(cue: AmbienceCue) -> &'static [&'static str] {
    match cue {
        AmbienceCue::None => &[],
        AmbienceCue::DesertWind => &[
            "audio/amb0.ogg",
            "audio/ambience/desert_wind.ogg",
            "audio/ambient/desert.ogg",
            "sounds/ambience/desert_wind.ogg",
        ],
        AmbienceCue::SewerDrip => &[
            "audio/amb1.ogg",
            "audio/ambience/sewer_drip.ogg",
            "audio/ambient/sewers.ogg",
            "sounds/ambience/sewer_drip.ogg",
        ],
        AmbienceCue::ScrapHum => &[
            "audio/amb2.ogg",
            "audio/ambience/scrap_hum.ogg",
            "audio/ambient/scrapyards.ogg",
            "sounds/ambience/scrap_hum.ogg",
        ],
        AmbienceCue::CrystalHum => &[
            "audio/amb3.ogg",
            "audio/ambience/crystal_hum.ogg",
            "audio/ambient/crystal.ogg",
            "sounds/ambience/crystal_hum.ogg",
        ],
        AmbienceCue::FrozenWind => &[
            "audio/amb4.ogg",
            "audio/ambience/frozen_wind.ogg",
            "audio/ambient/frozen.ogg",
            "sounds/ambience/frozen_wind.ogg",
        ],
        AmbienceCue::LabBuzz => &[
            "audio/amb5.ogg",
            "audio/ambience/lab_buzz.ogg",
            "audio/ambient/labs.ogg",
            "sounds/ambience/lab_buzz.ogg",
        ],
        AmbienceCue::PalaceFire => &[
            "audio/amb6.ogg",
            "audio/ambience/palace_fire.ogg",
            "audio/ambient/palace.ogg",
            "sounds/ambience/palace_fire.ogg",
        ],
        AmbienceCue::OasisBreeze => &[
            "audio/amb0b.ogg",
            "audio/ambience/oasis_breeze.ogg",
            "audio/ambient/oasis.ogg",
            "sounds/ambience/oasis_breeze.ogg",
        ],
        AmbienceCue::JungleBugs => &[
            "audio/amb0c.ogg",
            "audio/ambience/jungle_bugs.ogg",
            "audio/ambient/jungle.ogg",
            "sounds/ambience/jungle_bugs.ogg",
        ],
        AmbienceCue::VaultHum => &[
            "audio/amb101.ogg",
            "audio/ambience/vault_hum.ogg",
            "audio/ambient/vault.ogg",
            "sounds/ambience/vault_hum.ogg",
        ],
        AmbienceCue::HqSirens => &[
            "audio/amb107.ogg",
            "audio/ambience/hq_sirens.ogg",
            "audio/ambient/hq.ogg",
            "sounds/ambience/hq_sirens.ogg",
        ],
        AmbienceCue::CampfireCrackle => &[
            "audio/amb105.ogg",
            "audio/ambience/campfire_crackle.ogg",
            "audio/ambient/campfire.ogg",
            "sounds/ambience/campfire_crackle.ogg",
        ],
        AmbienceCue::CityNoise => &[
            "audio/amb106.ogg",
            "audio/ambience/city_noise.ogg",
            "audio/ambient/city.ogg",
            "sounds/ambience/city_noise.ogg",
        ],
    }
}

/// First-candidate resolution (bevy `pick_audio_handle` minus the
/// catalog/`AssetServer` half, which lives backend-side).
pub fn music_path(cue: MusicCue) -> Option<&'static str> {
    music_candidates(cue).first().copied()
}

/// First-candidate resolution for ambience loops.
pub fn ambience_path(cue: AmbienceCue) -> Option<&'static str> {
    ambience_candidates(cue).first().copied()
}

/// Campfire rest wins, then the first boss with its own theme, then
/// the area track (bevy `desired_music_cue`).
pub fn desired_music_cue(
    area: AreaId,
    campfire_present: bool,
    boss_kinds: impl IntoIterator<Item = EnemyKind>,
) -> MusicCue {
    if campfire_present {
        return MusicCue::Campfire;
    }

    for kind in boss_kinds {
        if let Some(cue) = boss_music_for_kind(kind) {
            return cue;
        }
    }

    music_for_area(area)
}

/// Campfire crackle wins, else the area bed (bevy
/// `desired_ambience_cue`).
pub fn desired_ambience_cue(area: AreaId, campfire_present: bool) -> AmbienceCue {
    if campfire_present {
        return AmbienceCue::CampfireCrackle;
    }

    ambience_for_area(area)
}

pub const MUSIC_FADE_SECS: f32 = 1.2;
pub const AMBIENCE_FADE_SECS: f32 = 0.8;
pub const AMBIENCE_BASE_SCALE: f32 = 0.55;
pub const MUSIC_PAUSE_DIM: f32 = 0.45;
const GAME_OVER_FADE_SECS: f32 = 0.18;

/// Ambience filter target (bevy `update_amb_filter` law: duck to 0.2
/// while paused or while the spiral background state exists, else 1.0).
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

/// Area loop selection with bevy `AppState` gating: Splash silences,
/// Loading previews the area, menus hold the title theme, game-over
/// silences, InGame follows campfire/boss/area priority.
pub fn sync_area_audio(
    app_state: Res<AppState>,
    run: Res<Run>,
    transition: Res<LoopTransition>,
    mut state: ResMut<AreaAudioState>,
    campfires: Query<(), With<CampfireProp>>,
    bosses: Query<&Enemy, With<BossBrain>>,
) {
    let campfire_present = transition.campfire_active || !campfires.is_empty();

    let (wanted_music, wanted_ambience): (Option<MusicCue>, Option<AmbienceCue>) =
        if *app_state != AppState::InGame {
            if *app_state == AppState::Loading {
                (
                    Some(desired_music_cue(
                        run.area,
                        campfire_present,
                        bosses.iter().map(|e| e.kind),
                    )),
                    Some(desired_ambience_cue(run.area, campfire_present)),
                )
            } else if *app_state == AppState::Splash {
                (None, None)
            } else {
                (Some(MusicCue::TitleTheme), None)
            }
        } else if run.game_over {
            (None, None)
        } else {
            (
                Some(desired_music_cue(
                    run.area,
                    campfire_present,
                    bosses.iter().map(|e| e.kind),
                )),
                Some(desired_ambience_cue(run.area, campfire_present)),
            )
        };

    if state.current_music != wanted_music {
        // Compromise: one gain per bus instead of overlapping fade-out /
        // fade-in voices. Fades start from the current gain so a switch
        // holds level (bevy net crossfade) and game-over dips to silence.
        let fade_secs = if wanted_music.is_none() && run.game_over {
            GAME_OVER_FADE_SECS
        } else {
            MUSIC_FADE_SECS
        };
        let end = if wanted_music.is_some() { 1.0 } else { 0.0 };
        state.music_fade = AreaAudioFader {
            start: state.music_gain,
            end,
            timer: GTimer::from_seconds(fade_secs, TimerMode::Once),
        };
        state.current_music = wanted_music;
    }

    if state.current_ambience != wanted_ambience {
        let end = if wanted_ambience.is_some_and(|c| c != AmbienceCue::None) {
            1.0
        } else {
            0.0
        };
        state.ambience_fade = AreaAudioFader {
            start: state.ambience_gain,
            end,
            timer: GTimer::from_seconds(AMBIENCE_FADE_SECS, TimerMode::Once),
        };
        state.current_ambience = wanted_ambience;
    }
}

/// Advance switch fades and remix volumes (covers both bevy
/// `tick_area_audio_fades` and `sync_area_audio_volumes`: volumes are
/// recomputed every tick from the current gains, not just while a
/// fade is active).
pub fn tick_area_audio_fades(
    time: Res<SimTime>,
    channels: Res<AudioChannels>,
    paused: Res<Paused>,
    amb_filter: Res<AmbFilter>,
    mut state: ResMut<AreaAudioState>,
) {
    state.music_fade.timer.tick(time.delta_secs);
    state.ambience_fade.timer.tick(time.delta_secs);
    state.music_gain = state.music_fade.value().clamp(0.0, 1.0);
    state.ambience_gain = state.ambience_fade.value().clamp(0.0, 1.0);

    let music_dim = if paused.0 { MUSIC_PAUSE_DIM } else { 1.0 };
    let music_base = channels.master * channels.music * music_dim;
    let ambience_base = channels.master * channels.music * AMBIENCE_BASE_SCALE * amb_filter.0;

    state.music_volume = (music_base * state.music_gain).clamp(0.0, 1.0);
    state.ambience_volume = (ambience_base * state.ambience_gain).clamp(0.0, 1.0);
}

/// Silence both buses (bevy `despawn_area_audio`).
pub fn reset_area_audio(mut state: ResMut<AreaAudioState>) {
    *state = AreaAudioState::default();
}

// --- Reactive audio selection (bevy `reactive_audio.rs` port) ---

pub fn cue_candidates(cue: ReactiveCue) -> &'static [&'static str] {
    match cue {
        ReactiveCue::LevelUp => &[
            "audio/sfx/snd_levelup.ogg",
            "audio/sfx/snd_mutation.ogg",
            "sounds/snd_levelup.ogg",
        ],
        ReactiveCue::MutationChosen => &[
            "audio/sfx/snd_mutation_chosen.ogg",
            "audio/sfx/snd_mutation.ogg",
            "sounds/snd_mutation.ogg",
        ],
        ReactiveCue::UltraChosen => &[
            "audio/sfx/snd_ultra_chosen.ogg",
            "audio/sfx/snd_mutation_chosen.ogg",
            "audio/sfx/snd_levelup.ogg",
        ],
        ReactiveCue::BossAppear => &[
            "audio/sfx/snd_boss_appear.ogg",
            "audio/sfx/snd_boss_intro.ogg",
            "sounds/snd_boss_intro.ogg",
        ],
        ReactiveCue::BossDefeated => &[
            "audio/sfx/snd_boss_dead.ogg",
            "audio/sfx/snd_boss_defeated.ogg",
            "sounds/snd_boss_dead.ogg",
        ],
        ReactiveCue::PlayerCritical => &[
            "audio/sfx/snd_hurt_critical.ogg",
            "audio/sfx/snd_low_health.ogg",
            "audio/sfx/snd_hurt.ogg",
        ],
        ReactiveCue::PlayerDeath => &[
            "audio/sfx/snd_player_dead.ogg",
            "audio/sfx/snd_death.ogg",
            "sounds/snd_death.ogg",
        ],
        ReactiveCue::PortalOpen => &[
            "audio/sfx/snd_portal_open.ogg",
            "audio/sfx/snd_portal.ogg",
            "sounds/snd_portal.ogg",
        ],
        ReactiveCue::PortalEnter => &[
            "audio/sfx/snd_portal_enter.ogg",
            "audio/sfx/snd_portal.ogg",
            "sounds/snd_portal.ogg",
        ],
        ReactiveCue::SecretFound => &["audio/sfx/snd_secret.ogg", "audio/sfx/snd_secret_found.ogg"],
        ReactiveCue::WeaponPickup => &[
            "audio/sfx/snd_weapon_pickup.ogg",
            "audio/sfx/snd_pickup.ogg",
        ],
        ReactiveCue::ChestOpen => &["audio/sfx/snd_chest_open.ogg", "audio/sfx/snd_pickup.ogg"],
        ReactiveCue::LoopComplete => &[
            "audio/sfx/snd_loop_complete.ogg",
            "audio/sfx/snd_levelup.ogg",
        ],
        ReactiveCue::ThroneRises => &[
            "audio/sfx/snd_throne_rises.ogg",
            "audio/sfx/snd_boss_intro.ogg",
        ],
        ReactiveCue::IdpdIncoming => {
            &["audio/sfx/snd_idpd_incoming.ogg", "audio/sfx/snd_alarm.ogg"]
        }
        ReactiveCue::Kill => &["audio/sfx/snd_kill.ogg", "audio/sfx/snd_hit.ogg"],
        ReactiveCue::KillStreak => &["audio/sfx/snd_streak.ogg", "audio/sfx/snd_levelup.ogg"],
        ReactiveCue::UiClick => &["audio/sfx/ui_click.ogg", "audio/sfx/snd_ui_click.ogg"],
        ReactiveCue::UiBack => &["audio/sfx/ui_back.ogg", "audio/sfx/snd_ui_back.ogg"],
        ReactiveCue::UiConfirm => &["audio/sfx/ui_confirm.ogg", "audio/sfx/snd_ui_confirm.ogg"],
        ReactiveCue::UiCycle => &["audio/sfx/ui_cycle.ogg", "audio/sfx/snd_ui_cycle.ogg"],
    }
}

pub fn cue_base_volume(cue: ReactiveCue) -> f32 {
    match cue {
        ReactiveCue::PlayerDeath => 1.0,
        ReactiveCue::BossAppear | ReactiveCue::BossDefeated | ReactiveCue::ThroneRises => 0.95,
        ReactiveCue::LoopComplete => 0.95,
        ReactiveCue::LevelUp
        | ReactiveCue::MutationChosen
        | ReactiveCue::UltraChosen
        | ReactiveCue::SecretFound
        | ReactiveCue::IdpdIncoming => 0.85,
        ReactiveCue::PortalOpen | ReactiveCue::PortalEnter | ReactiveCue::PlayerCritical => 0.75,
        ReactiveCue::KillStreak => 0.70,
        ReactiveCue::WeaponPickup | ReactiveCue::ChestOpen | ReactiveCue::UiBack => 0.58,
        ReactiveCue::UiConfirm => 0.58,
        ReactiveCue::UiClick => 0.48,
        ReactiveCue::UiCycle => 0.40,
        ReactiveCue::Kill => 0.32,
    }
}

pub fn cue_throttle_seconds(cue: ReactiveCue) -> f32 {
    match cue {
        ReactiveCue::Kill => 0.12,
        ReactiveCue::UiCycle => 0.06,
        ReactiveCue::WeaponPickup | ReactiveCue::ChestOpen => 0.25,
        ReactiveCue::PlayerCritical => 1.5,
        ReactiveCue::KillStreak => 2.5,
        _ => 0.38,
    }
}

pub fn throttle_allows(cue: ReactiveCue, last_fired: Option<f32>, now: f32) -> bool {
    let Some(last) = last_fired else {
        return true;
    };

    now - last >= cue_throttle_seconds(cue).max(1.0 / 30.0)
}

/// IDPD kind check (bevy `game/idpd.rs::is_idpd_kind` parity: the
/// inspector is NOT an IDPD raid kind).
pub fn is_idpd_kind(kind: EnemyKind) -> bool {
    matches!(
        kind,
        EnemyKind::IdpdGrunt | EnemyKind::IdpdShield | EnemyKind::IdpdElite | EnemyKind::IdpdVan
    )
}

#[derive(Resource, Default)]
pub struct ReactiveAudioState {
    last_fired: HashMap<ReactiveCue, f32>,
    known_bosses: HashSet<Entity>,
    last_player_level: Option<u32>,
    last_player_hp: Option<i32>,
    low_hp_armed: bool,
    kill_streak: u32,
    kill_streak_last: f32,
}

impl ReactiveAudioState {
    pub fn reset(&mut self) {
        self.last_fired.clear();
        self.known_bosses.clear();
        self.last_player_level = None;
        self.last_player_hp = None;
        self.low_hp_armed = true;
        self.kill_streak = 0;
        self.kill_streak_last = 0.0;
    }

    fn mark_fired(&mut self, cue: ReactiveCue, now: f32) {
        self.last_fired.insert(cue, now);
    }

    fn last_fired(&self, cue: ReactiveCue) -> Option<f32> {
        self.last_fired.get(&cue).copied()
    }

    fn note_kill(&mut self, now: f32) -> bool {
        if now - self.kill_streak_last > 3.5 {
            self.kill_streak = 0;
        }

        self.kill_streak_last = now;
        self.kill_streak += 1;

        self.kill_streak % 10 == 0
    }
}

pub fn reset_reactive_audio_state(mut state: ResMut<ReactiveAudioState>) {
    state.reset();
}

pub fn flush_queued_cues(
    mut commands: Commands,
    queued: Query<(Entity, &QueuedReactiveCue)>,
    mut requests: ResMut<Queue<ReactiveAudioRequest>>,
) {
    for (entity, cue) in queued.iter() {
        requests.push(ReactiveAudioRequest::new(cue.0));
        commands.entity(entity).despawn();
    }
}

/// Selection half of bevy `play_reactive_audio_requests`: bus gate,
/// per-cue throttle, first-candidate resolution, base*bus volume —
/// emitted as backend-neutral [`ResolvedReactiveCue`]s. The actual
/// spawn/playback is backend-owned.
pub fn play_reactive_audio_requests(
    time: Res<SimTime>,
    channels: Res<AudioChannels>,
    mut state: ResMut<ReactiveAudioState>,
    mut requests: ResMut<Queue<ReactiveAudioRequest>>,
    mut out: ResMut<Queue<ResolvedReactiveCue>>,
) {
    let now = time.elapsed_secs as f32;
    let bus = channels.master.clamp(0.0, 1.0) * channels.sfx.clamp(0.0, 1.0);

    if bus <= 0.0 {
        requests.drain();
        return;
    }

    for request in requests.drain() {
        if !throttle_allows(request.cue, state.last_fired(request.cue), now) {
            continue;
        }

        // No catalog in the sim: always emit the first candidate; the
        // backend drops missing files. Mark fired either way (bevy did
        // the same when no file existed) so a missing asset can't spin.
        let path = cue_candidates(request.cue)
            .first()
            .copied()
            .unwrap_or("audio/sfx/snd_hit.ogg");
        out.push(ResolvedReactiveCue {
            cue: request.cue,
            path,
            volume: (cue_base_volume(request.cue) * bus).clamp(0.0, 1.0),
        });

        state.mark_fired(request.cue, now);
    }
}

pub fn observe_player_audio_state(
    mut state: ResMut<ReactiveAudioState>,
    player_q: Query<(&Player, &Health), With<Player>>,
    mut requests: ResMut<Queue<ReactiveAudioRequest>>,
) {
    let Ok((player, health)) = player_q.single() else {
        state.last_player_level = None;
        state.last_player_hp = None;
        state.low_hp_armed = true;
        return;
    };

    if let Some(prev_level) = state.last_player_level
        && player.level > prev_level
    {
        requests.push(ReactiveAudioRequest::new(ReactiveCue::LevelUp));
    }
    state.last_player_level = Some(player.level);

    let crit = (health.max as f32 * 0.10).ceil() as i32;
    if let Some(prev_hp) = state.last_player_hp
        && health.hp < prev_hp
        && health.hp > 0
        && health.hp <= crit
    {
        requests.push(ReactiveAudioRequest::new(ReactiveCue::PlayerCritical));
    }

    let low = (health.max as f32 * 0.25).ceil() as i32;
    if health.hp > low {
        state.low_hp_armed = true;
    } else if health.hp > 0 && state.low_hp_armed {
        requests.push(ReactiveAudioRequest::new(ReactiveCue::PlayerCritical));
        state.low_hp_armed = false;
    }

    state.last_player_hp = Some(health.hp);
}

pub fn observe_boss_audio_state(
    mut state: ResMut<ReactiveAudioState>,
    bosses: Query<Entity, With<BossBrain>>,
    mut requests: ResMut<Queue<ReactiveAudioRequest>>,
) {
    let current: HashSet<Entity> = bosses.iter().collect();

    if current.difference(&state.known_bosses).next().is_some() {
        requests.push(ReactiveAudioRequest::new(ReactiveCue::BossAppear));
    }

    if state.known_bosses.difference(&current).next().is_some() {
        requests.push(ReactiveAudioRequest::new(ReactiveCue::BossDefeated));
    }

    state.known_bosses = current;
}

pub fn observe_kill_audio_state(
    time: Res<SimTime>,
    mut state: ResMut<ReactiveAudioState>,
    mut removed: RemovedComponents<Enemy>,
    mut requests: ResMut<Queue<ReactiveAudioRequest>>,
) {
    let now = time.elapsed_secs as f32;

    for _ in removed.read() {
        requests.push(ReactiveAudioRequest::new(ReactiveCue::Kill));

        if state.note_kill(now) {
            requests.push(ReactiveAudioRequest::new(ReactiveCue::KillStreak));
        }
    }
}

// --- UI action -> cue mapping (bevy `menus::UiAction` mirror) ---

/// Headless mirror of the bevy menu `UiAction` (variant shapes kept so
/// the mapping below ports verbatim; the real menu slice owns the
/// canonical enum later).
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
    /// Open the run-stats panel (GML `DrawStats` parity over the main
    /// menu; bevy left STATS inert).
    ShowStats,
}

/// Bridged menu action (bevy `UiBridgeAction` message -> [`Queue`]).
#[derive(Clone, Debug)]
pub struct UiBridgeAction(pub UiAction);

pub fn ui_action_to_cue(action: &UiAction) -> Option<ReactiveCue> {
    match action {
        UiAction::StartGame | UiAction::MainMenuPlay | UiAction::Resume => {
            Some(ReactiveCue::UiConfirm)
        }

        UiAction::QuitToTitle | UiAction::QuitApp => Some(ReactiveCue::UiBack),

        UiAction::ToggleLoadout | UiAction::ToggleHardmode => Some(ReactiveCue::UiBack),

        UiAction::OpenSettings
        | UiAction::OpenCredits
        | UiAction::CloseOverlay
        | UiAction::SaveSettings => Some(ReactiveCue::UiClick),

        UiAction::SelectCharacter(_)
        | UiAction::PickMutation(_)
        | UiAction::SelectMutation(_)
        | UiAction::SelectCrown(_) => Some(ReactiveCue::UiConfirm),

        UiAction::SelectSkin(_)
        | UiAction::NextLanguage
        | UiAction::CycleStartWeapon(_)
        | UiAction::CycleStoredWeapon(_)
        | UiAction::CycleCrown(_) => Some(ReactiveCue::UiCycle),

        UiAction::SetMasterVol(_)
        | UiAction::SetSfxVol(_)
        | UiAction::SetMusicVol(_)
        | UiAction::SetAmbienceVol(_)
        | UiAction::SetLanguage(_) => None,
        UiAction::SettingsCategory(_)
        | UiAction::SettingsBack
        | UiAction::ShowPauseConfirm(_)
        | UiAction::CancelPauseConfirm
        | UiAction::ConfirmPause(_) => Some(ReactiveCue::UiClick),
        UiAction::SettingToggle(_)
        | UiAction::SettingCycle { .. }
        | UiAction::SettingInput { .. }
        | UiAction::SettingResetOptions
        | UiAction::SettingEraseProgress
        | UiAction::SettingViewCredits
        | UiAction::SettingOpenSubcategory(_)
        | UiAction::ShowStats => Some(ReactiveCue::UiClick),
        UiAction::SettingSlider { .. } => None,
    }
}

pub fn play_ui_action_audio(
    mut inbox: ResMut<Queue<UiBridgeAction>>,
    mut requests: ResMut<Queue<ReactiveAudioRequest>>,
) {
    for bridged in inbox.drain() {
        if let Some(cue) = ui_action_to_cue(&bridged.0) {
            requests.push(ReactiveAudioRequest::new(cue));
        }
    }
}

/// Bevy `app.rs` UI one-shot map verbatim (stems + volumes; variance 0
/// — `play_ui_sfx` plays dry). Context-free actions only; character/
/// skin/crown/mutation picks need site context (see the `*_sfx`
/// helpers below) and sliders commit per change.
pub fn ui_action_sfx(action: &UiAction) -> Vec<AudioCue> {
    let mut out = Vec::new();
    let mut push = |stem: &'static str, volume: f32| {
        out.push(AudioCue {
            name: stem,
            volume,
            variance: 0.0,
        });
    };
    match action {
        UiAction::MainMenuPlay => {
            push("sndClick", 0.7);
            push("sndMenuCharSelect", 0.7);
        }
        UiAction::QuitToTitle | UiAction::QuitApp => {
            push("sndClickBack", 0.6);
        }
        UiAction::ShowPauseConfirm(_) => {
            push("sndClick", 0.7);
        }
        UiAction::CancelPauseConfirm => {
            push("sndClickBack", 0.6);
        }
        UiAction::ConfirmPause(_) => {
            push("sndClick", 0.7);
        }
        UiAction::SettingToggle(_)
        | UiAction::SettingCycle { .. }
        | UiAction::SettingInput { .. } => {
            push("sndClick", 0.6);
        }
        UiAction::SetMasterVol(_)
        | UiAction::SetSfxVol(_)
        | UiAction::SetMusicVol(_)
        | UiAction::SetAmbienceVol(_) => {
            push("sndSliderLetGo", 0.5);
        }
        UiAction::SetLanguage(_) | UiAction::SettingsCategory(_) => {
            push("sndClick", 0.7);
        }
        UiAction::SettingSlider { .. } => {
            push("sndSliderLetGo", 0.5);
        }
        UiAction::SettingResetOptions | UiAction::SettingEraseProgress => {
            push("sndClick", 0.7);
        }
        UiAction::SettingViewCredits => {
            push("sndMenuCredits", 0.7);
        }
        UiAction::ShowStats => {
            push("sndMenuStats", 0.7);
        }
        UiAction::SettingOpenSubcategory(_) => {
            push("sndClick", 0.7);
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

/// Bevy `SelectSkin` cue verbatim: A/B/C skin sting by slot, 1.0.
pub fn skin_select_sfx(skin: u8) -> AudioCue {
    let name = match skin {
        2 => "sndMenuCSkin",
        1 => "sndMenuBSkin",
        _ => "sndMenuASkin",
    };
    AudioCue {
        name,
        volume: 1.0,
        variance: 0.0,
    }
}

/// Denial sting (bevy `sndNoSelect`, 0.5).
pub fn denied_sfx() -> AudioCue {
    AudioCue {
        name: "sndNoSelect",
        volume: 0.5,
        variance: 0.0,
    }
}

/// Mutation highlight sting (bevy `sndHover`, 0.45).
pub fn hover_sfx() -> AudioCue {
    AudioCue {
        name: "sndHover",
        volume: 0.45,
        variance: 0.0,
    }
}

/// Crown pick sting (bevy `sndMenuCrown`, 1.0).
pub fn crown_select_sfx() -> AudioCue {
    AudioCue {
        name: "sndMenuCrown",
        volume: 1.0,
        variance: 0.0,
    }
}

// --- Combat intensity layer (bevy `reactive_audio.rs` tail) ---

#[derive(Component)]
pub struct CombatIntensityLayer {
    #[allow(dead_code)]
    pub area: AreaId,
    pub current: f32,
}

#[derive(Resource, Default)]
pub struct CombatIntensityState {
    last_area: Option<AreaId>,
}

pub fn intensity_candidates(area: AreaId) -> &'static [&'static str] {
    match area {
        AreaId::Desert => &[
            "audio/music/mus_desert_intensity.ogg",
            "audio/music/desert_intensity.ogg",
        ],
        AreaId::Sewers | AreaId::PizzaSewers => &[
            "audio/music/mus_sewers_intensity.ogg",
            "audio/music/sewers_intensity.ogg",
        ],
        AreaId::Scrapyards => &[
            "audio/music/mus_scrapyard_intensity.ogg",
            "audio/music/scrapyard_intensity.ogg",
        ],
        AreaId::CrystalCaves | AreaId::CursedCaves => &[
            "audio/music/mus_caves_intensity.ogg",
            "audio/music/caves_intensity.ogg",
        ],
        AreaId::FrozenCity => &[
            "audio/music/mus_frozen_intensity.ogg",
            "audio/music/frozen_intensity.ogg",
        ],
        AreaId::Labs => &[
            "audio/music/mus_labs_intensity.ogg",
            "audio/music/labs_intensity.ogg",
        ],
        AreaId::Palace => &[
            "audio/music/mus_palace_intensity.ogg",
            "audio/music/palace_intensity.ogg",
        ],
        AreaId::HQ => &[
            "audio/music/mus_hq_intensity.ogg",
            "audio/music/hq_intensity.ogg",
        ],
        _ => &[],
    }
}

/// First-candidate stem for the intensity layer, if the area has one.
pub fn intensity_path(area: AreaId) -> Option<&'static str> {
    intensity_candidates(area).first().copied()
}

pub fn combat_intensity_score(enemies_total: usize, idpd_count: usize, boss_count: usize) -> u32 {
    let ordinary = enemies_total.saturating_sub(idpd_count + boss_count);

    ordinary as u32 + idpd_count as u32 * 2 + boss_count as u32 * 4
}

pub fn combat_intensity_target(score: u32) -> f32 {
    match score {
        0..=2 => 0.0,
        3..=9 => (score - 2) as f32 / 7.0,
        _ => 1.0,
    }
}

pub fn smooth_value(current: f32, target: f32, dt: f32, half_life: f32) -> f32 {
    if dt <= 0.0 || half_life <= 0.0 {
        return target;
    }

    let keep = 2.0_f32.powf(-dt / half_life);
    target + (current - target) * keep
}

/// Intensity mix bus (bevy law: master * music * 0.42). The backend
/// multiplies this by the layer `current` it polls.
pub fn intensity_bus(channels: &AudioChannels) -> f32 {
    channels.master.clamp(0.0, 1.0) * channels.music.clamp(0.0, 1.0) * 0.42
}

pub fn intensity_volume(current: f32, channels: &AudioChannels) -> f32 {
    (current * intensity_bus(channels)).clamp(0.0, 1.0)
}

pub fn reset_combat_intensity(
    mut commands: Commands,
    mut state: ResMut<CombatIntensityState>,
    layers: Query<Entity, With<CombatIntensityLayer>>,
) {
    state.last_area = None;

    for entity in layers.iter() {
        commands.entity(entity).despawn();
    }
}

/// Track the per-area intensity layer and smooth its `current` toward
/// the combat target. The looped playback + sink volumes are
/// backend-owned (it polls the layer `current` and mixes with
/// [`intensity_bus`]); without a catalog the layer spawns whenever
/// the area has candidates and the backend skips missing files.
pub fn update_combat_intensity_audio(
    mut commands: Commands,
    time: Res<SimTime>,
    run: Option<Res<Run>>,
    transition: Option<Res<LoopTransition>>,
    mut state: ResMut<CombatIntensityState>,
    enemies: Query<&Enemy>,
    bosses: Query<(), With<BossBrain>>,
    mut layers: Query<(Entity, &mut CombatIntensityLayer)>,
) {
    let Some(run) = run else {
        return;
    };

    let suppressed = transition
        .as_ref()
        .is_some_and(|t| t.campfire_active || t.throne_ii_alive);

    if state.last_area != Some(run.area) {
        for (entity, _) in layers.iter() {
            commands.entity(entity).despawn();
        }

        state.last_area = Some(run.area);

        if intensity_path(run.area).is_some() {
            commands.spawn((
                GameCleanup,
                CombatIntensityLayer {
                    area: run.area,
                    current: 0.0,
                },
            ));
        }

        return;
    }

    let mut enemy_total = 0usize;
    let mut idpd_total = 0usize;

    for enemy in enemies.iter() {
        enemy_total += 1;
        if is_idpd_kind(enemy.kind) {
            idpd_total += 1;
        }
    }

    let boss_total = bosses.iter().count();
    let score = combat_intensity_score(enemy_total, idpd_total, boss_total);
    let mut target = combat_intensity_target(score);

    if suppressed {
        target = 0.0;
    }

    for (_, mut layer) in &mut layers {
        layer.current = smooth_value(layer.current, target, time.delta_secs, 0.55);
    }
}
