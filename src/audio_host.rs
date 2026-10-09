use std::collections::HashSet;
use std::path::{Path, PathBuf};

use web_time::{Duration, Instant};

use rand::RngExt;
use repame_audio::{Audio, AudioChannel, CueDef, CueRequest, Variation};

use crate::App;

const EXTS: [&str; 4] = ["ogg", "wav", "mp3", "flac"];
const STOP_PREFIX: &str = "stop_";

struct PendingCue {
    stem: &'static str,
    volume: f32,
    variance: f32,
}

pub struct AudioHost {
    sfx: Audio,
    amb: Option<Audio>,
    epoch: Instant,
    sounds: Option<PathBuf>,
    stems: HashSet<String>,
    missing: HashSet<String>,
    pending: Vec<PendingCue>,
    tracks: HashSet<String>,
    amb_tracks: HashSet<String>,
    last_miss_log: Option<Instant>,
    music_want: Option<&'static str>,
    amb_want: Option<&'static str>,
    channels: Option<[f32; 5]>,
}

impl AudioHost {
    pub fn new() -> Self {
        let sfx = Audio::try_init().unwrap_or_else(|e| {
            log::warn!("nt: sfx audio device failed ({e}); running silent");
            Audio::noop()
        });
        let amb = match Audio::try_init() {
            Ok(amb) => Some(amb),
            Err(e) => {
                log::warn!("nt: ambience audio device failed ({e}); ambience off");
                None
            }
        };
        let sounds = crate::resolve_assets_dir().map(|dir| dir.join("sounds"));
        let stems = match &sounds {
            Some(dir) => scan_stems(dir),
            None => {
                log::warn!("nt: no assets dir; sound cues disabled");
                HashSet::new()
            }
        };
        log::info!(
            "nt: audio live={} ambience={} stems={}",
            sfx.is_live(),
            amb.as_ref().is_some_and(|a| a.is_live()),
            stems.len()
        );
        Self {
            sfx,
            amb,
            epoch: Instant::now(),
            sounds,
            stems,
            tracks: HashSet::new(),
            amb_tracks: HashSet::new(),
            missing: HashSet::new(),
            pending: Vec::new(),
            last_miss_log: None,
            music_want: None,
            amb_want: None,
            channels: None,
        }
    }

    /// Re-arm every device the browser autoplay gate left suspended.
    ///
    /// Both engines open their own `AudioContext`, so both need the nudge.
    pub fn unlock_all(&self) {
        self.sfx.unlock();
        if let Some(amb) = &self.amb {
            amb.unlock();
        }
    }

    pub fn pump(&mut self, dt_secs: f32, app: &mut App) {
        self.sfx.update(dt_secs);
        self.sfx.take_finished();
        if let Some(amb) = &mut self.amb {
            amb.update(dt_secs);
            amb.take_finished();
        }
        let (music, ambience, music_volume, ambience_volume) = app.area_audio_snapshot();
        let channels = app
            .sim
            .world
            .get_resource::<crate::audio::AudioChannels>()
            .copied()
            .unwrap_or_default();
        let mainvol = crate::audio::mainvol_gain(&app.sim.world);
        let slider = channels.sfx;
        self.sync_channels(
            channels.master,
            channels.sfx * mainvol,
            channels.ui,
            music_volume,
            ambience_volume,
        );
        for cue in app.drain_audio_cues() {
            self.play_cue(&cue, slider);
        }
        self.flush_pending();
        self.sync_music(music.and_then(crate::audio::music_path));
        self.sync_ambience(ambience.and_then(crate::audio::ambience_path));
    }

    fn play_cue(&mut self, cue: &repame_audio::Cue, slider: f32) {
        // A `stop_`-prefixed cue is GML `audio_stop_sound`: the loop stems
        // have no such file on disk, so the prefix is the only marker that
        // the bank owns a live voice to silence.
        if let Some(looping) = cue.name.strip_prefix(STOP_PREFIX) {
            self.pending.retain(|p| p.stem != looping);
            if self.sfx.is_live() {
                self.sfx.stop_cue(looping);
            }
            return;
        }
        if !self.sfx.is_live() {
            return;
        }
        let mut volume = cue.volume;
        match stem_bus(cue.name) {
            AudioChannel::Sfx => {
                let power = sfx_slider_power(cue.name);
                if power != 0 && slider > 0.0 {
                    volume *= slider.powi(power);
                }
            }
            AudioChannel::Ui if hit_family(cue.name) => volume *= slider,
            _ => {}
        }
        if self.request_cue(cue.name) && self.play(cue.name, volume, cue.variance) {
            return;
        }
        // The decode is still in flight and this cue is already gone from
        // the queue, so an edge-triggered loop would never start at all.
        if !self.missing.contains(cue.name) && !self.queued(cue.name) {
            self.pending.push(PendingCue {
                stem: cue.name,
                volume,
                variance: cue.variance,
            });
        }
    }

    fn queued(&self, stem: &str) -> bool {
        self.pending.iter().any(|p| p.stem == stem)
    }

    /// Retry the cues whose first decode had not landed yet.
    fn flush_pending(&mut self) {
        if !self.sfx.is_live() {
            self.pending.clear();
            return;
        }
        let mut still_decoding = Vec::new();
        for pending in std::mem::take(&mut self.pending) {
            if self.missing.contains(pending.stem) {
                continue;
            }
            if !self.play(pending.stem, pending.volume, pending.variance) {
                still_decoding.push(pending);
            }
        }
        self.pending = still_decoding;
    }

    /// False while the bank has no decoded stem under that name yet.
    fn play(&mut self, stem: &str, volume: f32, variance: f32) -> bool {
        let jitter = rand::rng().random::<f32>();
        let rate = 1.0 + (2.0 * jitter - 1.0) * variance * 0.5;
        let now_ms = self.epoch.elapsed().as_millis() as u64;
        self.sfx
            .bank()
            .play_at_ms(stem, volume, rate, 0.0, now_ms)
            .is_some()
    }

    /// Ask the bank for a stem, decoding off-thread on first sight.
    ///
    /// `Queued` means the decode is still in flight; the caller retries
    /// from [`Self::pending`]. `Failed` is terminal, so the stem goes in
    /// `missing` and is never requested again.
    fn request_cue(&mut self, stem: &str) -> bool {
        if self.missing.contains(stem) {
            return false;
        }
        let Some(bytes) = self.read_stem(stem) else {
            self.mark_missing(stem, "no file");
            return false;
        };
        let def = CueDef {
            bus: stem_bus(stem),
            gain: 1.0,
            cooldown_ms: 0,
            max_voices: 1,
            pitch_wobble: 0.0,
            variation: Variation::RoundRobin,
        };
        match self.sfx.request_cue(stem, def, &[bytes.as_slice()]) {
            CueRequest::Ready => true,
            CueRequest::Queued => false,
            CueRequest::Failed(error) => {
                self.mark_missing(stem, &format!("decode failed: {error}"));
                false
            }
        }
    }

    fn ensure_track(&mut self, stem: &str, ambience: bool) -> bool {
        let loaded = if ambience {
            &self.amb_tracks
        } else {
            &self.tracks
        };
        if loaded.contains(stem) {
            return true;
        }
        if self.missing.contains(stem) {
            return false;
        }
        let live = if ambience {
            self.amb.as_ref().is_some_and(|a| a.is_live())
        } else {
            self.sfx.is_live()
        };
        if !live {
            return false;
        }
        let Some(bytes) = self.read_stem(stem) else {
            self.mark_missing(stem, "no file");
            return false;
        };
        let audio = if ambience {
            let Some(amb) = self.amb.as_mut() else {
                return false;
            };
            amb
        } else {
            &mut self.sfx
        };
        match audio.load_track(stem, 1.0, &bytes) {
            Ok(()) => {
                if ambience {
                    self.amb_tracks.insert(stem.to_owned());
                } else {
                    self.tracks.insert(stem.to_owned());
                }
                true
            }
            Err(e) => {
                self.mark_missing(stem, &format!("decode failed: {e}"));
                false
            }
        }
    }

    fn sync_music(&mut self, want: Option<&'static str>) {
        if want == self.music_want {
            return;
        }
        self.music_want = want;
        if !self.sfx.is_live() {
            return;
        }
        let ok = want.is_some_and(|stem| self.ensure_track(stem, false));
        match want {
            Some(stem) if ok => {
                if self.sfx.now_playing() != Some(stem) {
                    self.sfx.play_music(stem, 0.0);
                }
            }
            Some(_) | None => self.sfx.stop_music(0.0),
        }
    }

    fn sync_ambience(&mut self, want: Option<&'static str>) {
        if self.amb.is_none() || want == self.amb_want {
            return;
        }
        self.amb_want = want;
        let ok = want.is_some_and(|stem| self.ensure_track(stem, true));
        let Some(amb) = self.amb.as_mut() else {
            return;
        };
        match want {
            Some(stem) if ok => {
                if amb.now_playing() != Some(stem) {
                    amb.play_music(stem, 0.0);
                }
            }
            Some(_) | None => {
                if amb.now_playing().is_some() {
                    amb.stop_music(0.0);
                }
            }
        }
    }

    fn sync_channels(&mut self, master: f32, sfx: f32, ui: f32, music: f32, ambience: f32) {
        let want = [master, sfx, ui, music, ambience];
        if self.channels == Some(want) {
            return;
        }
        self.channels = Some(want);
        self.sfx.set_channel(AudioChannel::Master, master);
        self.sfx.set_channel(AudioChannel::Sfx, sfx);
        self.sfx.set_channel(AudioChannel::Ui, ui);
        self.sfx.set_channel(AudioChannel::Music, music);
        if let Some(amb) = &mut self.amb {
            amb.set_channel(AudioChannel::Master, master);
            amb.set_channel(AudioChannel::Music, ambience);
        }
    }

    fn read_stem(&self, stem: &str) -> Option<Vec<u8>> {
        read_stem_bytes(&self.stem_path(stem)?)
    }

    fn stem_path(&self, stem: &str) -> Option<PathBuf> {
        if !self.stems.contains(stem) {
            return None;
        }
        let root = self.sounds.as_ref()?;
        for ext in EXTS {
            let path = root.join(stem).join(format!("{stem}.{ext}"));
            if file_exists(&path) {
                return Some(path);
            }
        }
        for ext in EXTS {
            let path = root.join(format!("{stem}.{ext}"));
            if file_exists(&path) {
                return Some(path);
            }
        }
        None
    }

    fn mark_missing(&mut self, stem: &str, why: &str) {
        if !self.missing.insert(stem.to_owned()) {
            return;
        }
        let now = Instant::now();
        if self
            .last_miss_log
            .is_some_and(|t| now.duration_since(t) < Duration::from_secs(1))
        {
            return;
        }
        self.last_miss_log = Some(now);
        log::warn!("nt: audio stem `{stem}` dropped ({why})");
    }
}

fn stem_bus(stem: &str) -> AudioChannel {
    if matches!(stem, "sndBossWin" | "sndVaultBossWin") {
        AudioChannel::Music
    } else if hit_family(stem) || LOOP_FAMILY.binary_search(&stem).is_ok() {
        AudioChannel::Ui
    } else {
        AudioChannel::Sfx
    }
}

const HIT_FAMILY: [&str; 29] = [
    "sndAllySpawn",
    "sndBigCursedChest",
    "sndBloodHurt",
    "sndCuzBye",
    "sndCuzGreet",
    "sndCuzOutaway",
    "sndDogGuardianBounce",
    "sndDogGuardianJump",
    "sndDogGuardianLand",
    "sndEliteGruntRocketFire",
    "sndEliteGruntRoll",
    "sndExploGuardianCharge",
    "sndExplosionXL",
    "sndFlyFire",
    "sndGoldTankAim",
    "sndHPMimicTaunt",
    "sndHitWall",
    "sndImpWristHit",
    "sndImpWristKill",
    "sndLastEnemy",
    "sndLightningCrystalCharge",
    "sndLilHunterBreak",
    "sndMimicSlurp",
    "sndNothing2DeadStart",
    "sndRecGlandProc",
    "sndSalamanderEndFire",
    "sndSalamanderFire",
    "sndTurretFire",
    "sndUltraEmpty",
];

const LOOP_FAMILY: [&str; 8] = [
    "sndChickenHeadlessLoop",
    "sndEyesLoop",
    "sndEyesLoopUpg",
    "sndFrogLoop",
    "sndFrogLoopButt",
    "sndHorrorLoop",
    "sndHorrorLoopTB",
    "sndSalamanderFireLoop",
];

fn hit_family(stem: &str) -> bool {
    HIT_FAMILY.binary_search(&stem).is_ok()
}

const SFX_SLIDER_POWER: [(&str, i32); 353] = [
    ("sndAmmoChest", 1),
    ("sndAmmoPickup", 1),
    ("sndAssassinAttack", 1),
    ("sndBallMamaDead2", 2),
    ("sndBallMamaFire", 2),
    ("sndBallMamaHalfHP", 2),
    ("sndBallMamaLowHP", 2),
    ("sndBallMamaTaunt", 1),
    ("sndBigBanditTaunt", 1),
    ("sndBigDogMissile", 1),
    ("sndBigDogTaunt", 1),
    ("sndBigMaggotBurrow", 1),
    ("sndBigMaggotUnburrow", 1),
    ("sndBigMaggotUnburrowSand", 1),
    ("sndBlackSword", 2),
    ("sndBlackSwordMega", 2),
    ("sndBloodCannon", 2),
    ("sndBloodGamble", 1),
    ("sndBloodHammer", 2),
    ("sndBloodLauncher", 2),
    ("sndBouncerShotgun", 2),
    ("sndBouncerSmg", 2),
    ("sndCarLoop", 1),
    ("sndChest", 1),
    ("sndChickenRegenHead", 1),
    ("sndChickenReturn", 1),
    ("sndChickenSword", 2),
    ("sndChickenThrow", 1),
    ("sndChickenUltraA", 1),
    ("sndChickenUltraB", 1),
    ("sndClick", 1),
    ("sndClickBack", 1),
    ("sndClusterLauncher", 2),
    ("sndConfettiGun", 2),
    ("sndCrossReload", 1),
    ("sndCrossbow", 2),
    ("sndCrystalJuggernaut", 1),
    ("sndCrystalShield", 1),
    ("sndCrystalUltraA", 1),
    ("sndCrystalUltraB", 1),
    ("sndCursedChest", 1),
    ("sndCursedPickupDisappear", 1),
    ("sndCursedReminder", 1),
    ("sndCuzCryAttack", 1),
    ("sndCuzCryAttackNoAmmo", 1),
    ("sndCuzCryAttackUltraB", 1),
    ("sndCuzCryBonus1", 1),
    ("sndCuzCryBonus10", 1),
    ("sndCuzCryBonus2", 1),
    ("sndCuzCryBonus3", 1),
    ("sndCuzCryBonus4", 1),
    ("sndCuzCryBonus5", 1),
    ("sndCuzCryBonus6", 1),
    ("sndCuzCryBonus7", 1),
    ("sndCuzCryBonus8", 1),
    ("sndCuzCryBonus9", 1),
    ("sndCuzCryNew", 1),
    ("sndCuzUltraA", 1),
    ("sndCuzUltraB", 1),
    ("sndDevastator", 2),
    ("sndDevastatorUpg", 2),
    ("sndDiscgun", 2),
    ("sndDoubleFireShotgun", 2),
    ("sndDoubleMinigun", 2),
    ("sndDoubleShotgun", 1),
    ("sndEXPChest", 1),
    ("sndElectricGuitar", 2),
    ("sndEliteIDPDPortalSpawn", 1),
    ("sndEmpty", 1),
    ("sndEnemyFire", 1),
    ("sndEnergyHammer", 2),
    ("sndEnergyHammerUpg", 2),
    ("sndEnergyScrewdriver", 2),
    ("sndEnergyScrewdriverUpg", 2),
    ("sndEnergySword", 2),
    ("sndEnergySwordUpg", 2),
    ("sndEraser", 2),
    ("sndExplosionCar", 2),
    ("sndEyesUltraA", 1),
    ("sndEyesUltraB", 1),
    ("sndFireShotgun", 1),
    ("sndFishRollUpg", 1),
    ("sndFishUltraA", 1),
    ("sndFishUltraB", 1),
    ("sndFlameCannon", 2),
    ("sndFlare", 2),
    ("sndFrogClose", 1),
    ("sndFrogEggOpen1", 2),
    ("sndFrogEggOpen2", 2),
    ("sndFrogEggSpawn1", 2),
    ("sndFrogEggSpawn2", 2),
    ("sndFrogEggSpawn3", 2),
    ("sndFrogEnd", 1),
    ("sndFrogEndButt", 1),
    ("sndFrogExplode", 2),
    ("sndFrogGasRelease", 1),
    ("sndFrogPistol", 2),
    ("sndFrogStart", 1),
    ("sndFrogStartButt", 1),
    ("sndFrogUltraA", 1),
    ("sndFrogUltraB", 1),
    ("sndGoldChest", 1),
    ("sndGoldCrossbow", 2),
    ("sndGoldFrogPistol", 2),
    ("sndGoldGrenade", 2),
    ("sndGoldLaser", 2),
    ("sndGoldLaserUpg", 2),
    ("sndGoldMachinegun", 2),
    ("sndGoldPickup", 1),
    ("sndGoldPistol", 2),
    ("sndGoldPlasma", 2),
    ("sndGoldPlasmaUpg", 2),
    ("sndGoldRocket", 2),
    ("sndGoldScorpionFire", 1),
    ("sndGoldScrewdriver", 2),
    ("sndGoldShotgun", 2),
    ("sndGoldSlugger", 2),
    ("sndGoldSplinterGun", 2),
    ("sndGoldWrench", 2),
    ("sndGrenadeRifle", 2),
    ("sndGrenadeShotgun", 2),
    ("sndGruntThrowNadeF", 1),
    ("sndGruntThrowNadeM", 1),
    ("sndGuitar", 2),
    ("sndGuitarPickup", 1),
    ("sndGunGodIntro", 2),
    ("sndGunGodLowHP", 2),
    ("sndGunGodTaunt", 1),
    ("sndGunGun", 2),
    ("sndHPPickup", 1),
    ("sndHPPickupBig", 1),
    ("sndHammer", 2),
    ("sndHealthChest", 1),
    ("sndHealthChestBig", 1),
    ("sndHeavyCrossbow", 2),
    ("sndHeavyMachinegun", 2),
    ("sndHeavyNader", 2),
    ("sndHeavyRevolver", 2),
    ("sndHeavySlugger", 2),
    ("sndHorrorBeam", 1),
    ("sndHorrorEmpty", 1),
    ("sndHorrorUltraA", 1),
    ("sndHorrorUltraB", 1),
    ("sndHorrorUltraC", 1),
    ("sndHover", 1),
    ("sndHyperCrystalTaunt", 1),
    ("sndHyperLauncher", 2),
    ("sndHyperRifle", 2),
    ("sndHyperSlugger", 2),
    ("sndIDPDPortalSpawn", 1),
    ("sndIncinerator", 2),
    ("sndInspectorEndF", 1),
    ("sndInspectorEndM", 1),
    ("sndInspectorStartF", 1),
    ("sndInspectorStartM", 1),
    ("sndJackHammer", 1),
    ("sndLaserCannonCharge", 2),
    ("sndLastTaunt", 1),
    ("sndLevelUltra", 1),
    ("sndLevelUp", 1),
    ("sndLightningCannon", 2),
    ("sndLightningCannonUpg", 2),
    ("sndLightningHammer", 2),
    ("sndLightningPistolUpg", 2),
    ("sndLightningReload", 1),
    ("sndLightningRifle", 2),
    ("sndLightningRifleUpg", 2),
    ("sndLightningShotgun", 2),
    ("sndLightningShotgunUpg", 2),
    ("sndLilHunterTaunt", 1),
    ("sndMeatExplo", 1),
    ("sndMeleeFlip", 1),
    ("sndMeleeWall", 1),
    ("sndMeltingUltraA", 1),
    ("sndMeltingUltraB", 1),
    ("sndMenuASkin", 1),
    ("sndMenuBSkin", 1),
    ("sndMenuCSkin", 1),
    ("sndMenuCharSelect", 1),
    ("sndMenuCredits", 1),
    ("sndMenuCrown", 1),
    ("sndMenuOptions", 1),
    ("sndMenuStats", 1),
    ("sndMinigun", 2),
    ("sndMutBackMuscle", 1),
    ("sndMutBloodLust", 1),
    ("sndMutBoilingVeins", 1),
    ("sndMutBoltMarrow", 1),
    ("sndMutEagleEyes", 1),
    ("sndMutEuphoria", 1),
    ("sndMutExtraFeet", 1),
    ("sndMutGammaGuts", 1),
    ("sndMutHammerhead", 1),
    ("sndMutHeavyHeart", 1),
    ("sndMutImpactWrists", 1),
    ("sndMutLaserBrain", 1),
    ("sndMutLastWish", 1),
    ("sndMutLongArms", 1),
    ("sndMutLuckyShot", 1),
    ("sndMutOpenMind", 1),
    ("sndMutPatience", 1),
    ("sndMutPlutoniumHunger", 1),
    ("sndMutRabbitPaw", 1),
    ("sndMutRecycleGland", 1),
    ("sndMutRhinoSkin", 1),
    ("sndMutScarierFace", 1),
    ("sndMutSecondStomach", 1),
    ("sndMutSharpTeeth", 1),
    ("sndMutShotgunFingers", 1),
    ("sndMutStress", 1),
    ("sndMutStrongSpirit", 1),
    ("sndMutThroneButt", 1),
    ("sndMutTriggerFingers", 1),
    ("sndMutant0Cnfm", 1),
    ("sndMutant0Slct", 1),
    ("sndMutant10Cnfm", 1),
    ("sndMutant10Slct", 1),
    ("sndMutant11Cnfm", 1),
    ("sndMutant11Slct", 1),
    ("sndMutant12Cnfm", 1),
    ("sndMutant12Slct", 1),
    ("sndMutant13Cnfm", 1),
    ("sndMutant13Slct", 1),
    ("sndMutant15Cnfm", 1),
    ("sndMutant15Slct", 1),
    ("sndMutant16Cnfm", 1),
    ("sndMutant16Slct", 1),
    ("sndMutant1Cnfm", 1),
    ("sndMutant1Slct", 1),
    ("sndMutant2Cnfm", 1),
    ("sndMutant2Slct", 1),
    ("sndMutant3Cnfm", 1),
    ("sndMutant3Slct", 1),
    ("sndMutant4Cnfm", 1),
    ("sndMutant4Slct", 1),
    ("sndMutant5Cnfm", 1),
    ("sndMutant5Slct", 1),
    ("sndMutant6Cnfm", 1),
    ("sndMutant6Slct", 1),
    ("sndMutant7Cnfm", 1),
    ("sndMutant7Slct", 1),
    ("sndMutant8Cnfm", 1),
    ("sndMutant8Slct", 1),
    ("sndMutant9Cnfm", 1),
    ("sndMutant9Slct", 1),
    ("sndNadeReload", 1),
    ("sndNoSelect", 1),
    ("sndNothing2Taunt", 1),
    ("sndNothingDeath1", 1),
    ("sndNothingGenerators", 1),
    ("sndNothingTaunt", 1),
    ("sndNukeFire", 2),
    ("sndOasisChest", 1),
    ("sndOasisCrabAttack", 1),
    ("sndOasisPopo", 1),
    ("sndOasisPortal", 1),
    ("sndPickupDisappear", 1),
    ("sndPlantFire", 1),
    ("sndPlantFireTB", 1),
    ("sndPlantSnare", 1),
    ("sndPlantSnareTB", 1),
    ("sndPlantSnareTrapper", 1),
    ("sndPlantSnareTrapperTB", 1),
    ("sndPlantUltraA", 1),
    ("sndPlantUltraB", 1),
    ("sndPlasma", 2),
    ("sndPlasmaBig", 2),
    ("sndPlasmaBigUpg", 2),
    ("sndPlasmaHuge", 2),
    ("sndPlasmaHugeUpg", 2),
    ("sndPlasmaMinigun", 2),
    ("sndPlasmaMinigunUpg", 2),
    ("sndPlasmaReload", 1),
    ("sndPlasmaReloadUpg", 1),
    ("sndPlasmaRifle", 2),
    ("sndPlasmaRifleUpg", 2),
    ("sndPlasmaUpg", 2),
    ("sndPopPop", 2),
    ("sndPopPopUpg", 2),
    ("sndPortalClose", 1),
    ("sndPortalLightning1", 1),
    ("sndPortalLightning2", 1),
    ("sndPortalLightning3", 1),
    ("sndPortalLightning4", 1),
    ("sndPortalLightning5", 1),
    ("sndPortalLightning6", 1),
    ("sndPortalLightning7", 1),
    ("sndPortalLightning8", 1),
    ("sndPortalOpen", 1),
    ("sndPortalStrikeEmpty", 1),
    ("sndQuadMachinegun", 2),
    ("sndRadPickup", 1),
    ("sndRebelUltraA", 1),
    ("sndRebelUltraB", 1),
    ("sndRestart", 1),
    ("sndRobotEat", 1),
    ("sndRobotEatUpg", 1),
    ("sndRobotUltraA", 1),
    ("sndRobotUltraB", 1),
    ("sndRocket", 2),
    ("sndRogueAim", 1),
    ("sndRogueCanister", 1),
    ("sndRogueRifle", 1),
    ("sndRogueUltraA", 1),
    ("sndRogueUltraB", 1),
    ("sndRoll", 1),
    ("sndSawedOffShotgun", 1),
    ("sndScorpionFire", 1),
    ("sndScorpionFireStart", 1),
    ("sndScrewdriver", 2),
    ("sndSeekerPistol", 2),
    ("sndSeekerShotgun", 2),
    ("sndShotReload", 1),
    ("sndShotgun", 1),
    ("sndSkeletonUltraA", 1),
    ("sndSkeletonUltraB", 1),
    ("sndSlider", 1),
    ("sndSliderLetGo", 1),
    ("sndSmartgun", 2),
    ("sndSniperFire", 1),
    ("sndSniperTarget", 1),
    ("sndSnowTankAim", 1),
    ("sndSpawnSuperAlly", 1),
    ("sndSplinterGun", 2),
    ("sndSplinterPistol", 2),
    ("sndSteroidsUltraA", 1),
    ("sndSteroidsUltraB", 1),
    ("sndSuperBazooka", 2),
    ("sndSuperCrossbow", 2),
    ("sndSuperDiscGun", 2),
    ("sndSuperFlakCannon", 2),
    ("sndSuperSlugger", 2),
    ("sndSuperSplinterGun", 2),
    ("sndSwapElectricGuitar", 1),
    ("sndSwapGold", 1),
    ("sndToxicLauncher", 2),
    ("sndTripleMachinegun", 2),
    ("sndUltraCrossbow", 2),
    ("sndUltraGrenade", 2),
    ("sndUltraLaser", 2),
    ("sndUltraLaserUpg", 2),
    ("sndUltraPistol", 2),
    ("sndUltraShotgun", 2),
    ("sndUltraShovel", 2),
    ("sndUseCar", 1),
    ("sndUseVan", 1),
    ("sndVanWarning", 1),
    ("sndVlambeer", 1),
    ("sndWaveGun", 2),
    ("sndWeaponPickup", 1),
    ("sndWrench", 2),
    ("sndYVUltraA", 1),
    ("sndYVUltraB", 1),
];

fn sfx_slider_power(stem: &str) -> i32 {
    match SFX_SLIDER_POWER.binary_search_by(|(name, _)| (*name).cmp(stem)) {
        Ok(index) => SFX_SLIDER_POWER[index].1,
        Err(_) => 0,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn scan_stems(root: &Path) -> HashSet<String> {
    let mut stems = HashSet::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return stems;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                stems.insert(name.to_owned());
            }
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default();
        if !EXTS.contains(&ext) {
            continue;
        }
        if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
            stems.insert(name.to_owned());
        }
    }
    stems
}

/// Web: the sounds dir lives in the installed assets zip, so scan the
/// store's `sounds/` keys with the same top-level shape as `read_dir`.
#[cfg(target_arch = "wasm32")]
fn scan_stems(root: &Path) -> HashSet<String> {
    let Some(prefix) = root.file_name().and_then(|n| n.to_str()) else {
        return HashSet::new();
    };
    let prefix = format!("{prefix}/");
    let mut stems = HashSet::new();
    for key in crate::assetfs::keys() {
        let Some(rel) = key.strip_prefix(prefix.as_str()) else {
            continue;
        };
        let mut parts = rel.split('/');
        let Some(first) = parts.next() else {
            continue;
        };
        if parts.next().is_some() {
            stems.insert(first.to_owned());
            continue;
        }
        let ext = Path::new(first)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default();
        if !EXTS.contains(&ext) {
            continue;
        }
        if let Some(name) = Path::new(first).file_stem().and_then(|s| s.to_str()) {
            stems.insert(name.to_owned());
        }
    }
    stems
}

#[cfg(not(target_arch = "wasm32"))]
fn file_exists(path: &Path) -> bool {
    path.is_file()
}

#[cfg(target_arch = "wasm32")]
fn file_exists(path: &Path) -> bool {
    crate::assetfs::has(path)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_stem_bytes(path: &Path) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

#[cfg(target_arch = "wasm32")]
fn read_stem_bytes(path: &Path) -> Option<Vec<u8>> {
    crate::assetfs::get(path)
}
