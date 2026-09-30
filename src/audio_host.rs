use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rand::RngExt;
use repame_audio::{Audio, AudioChannel, CueDef, Variation};

use crate::App;

const EXTS: [&str; 4] = ["ogg", "wav", "mp3", "flac"];

pub struct AudioHost {
    sfx: Audio,
    amb: Option<Audio>,
    epoch: Instant,
    sounds: Option<PathBuf>,
    stems: HashSet<String>,
    cues: HashSet<String>,
    tracks: HashSet<String>,
    amb_tracks: HashSet<String>,
    missing: HashSet<String>,
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
            cues: HashSet::new(),
            tracks: HashSet::new(),
            amb_tracks: HashSet::new(),
            missing: HashSet::new(),
            last_miss_log: None,
            music_want: None,
            amb_want: None,
            channels: None,
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
        self.sync_channels(
            channels.master,
            channels.sfx,
            channels.ui,
            music_volume,
            ambience_volume,
        );
        for cue in app.drain_audio_cues() {
            self.play_cue(&cue);
        }
        self.sync_music(music.and_then(crate::audio::music_path));
        self.sync_ambience(ambience.and_then(crate::audio::ambience_path));
    }

    fn play_cue(&mut self, cue: &repame_audio::Cue) {
        if !self.sfx.is_live() || !self.ensure_cue(cue.name) {
            return;
        }
        let jitter = rand::rng().random::<f32>();
        let rate = 1.0 + (2.0 * jitter - 1.0) * cue.variance * 0.5;
        let now_ms = self.epoch.elapsed().as_millis() as u64;
        let _ = self
            .sfx
            .bank()
            .play_at_ms(cue.name, cue.volume, rate, 0.0, now_ms);
    }

    fn ensure_cue(&mut self, stem: &str) -> bool {
        if self.cues.contains(stem) {
            return true;
        }
        if self.missing.contains(stem) {
            return false;
        }
        let Some(bytes) = self.read_stem(stem) else {
            self.mark_missing(stem, "no file");
            return false;
        };
        let def = CueDef {
            bus: AudioChannel::Sfx,
            gain: 1.0,
            cooldown_ms: 0,
            max_voices: 1,
            pitch_wobble: 0.0,
            variation: Variation::RoundRobin,
        };
        match self.sfx.load_cue(stem, def, &[bytes.as_slice()]) {
            Ok(()) => {
                self.cues.insert(stem.to_owned());
                true
            }
            Err(e) => {
                self.mark_missing(stem, &format!("decode failed: {e}"));
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
        std::fs::read(self.stem_path(stem)?).ok()
    }

    fn stem_path(&self, stem: &str) -> Option<PathBuf> {
        if !self.stems.contains(stem) {
            return None;
        }
        let root = self.sounds.as_ref()?;
        for ext in EXTS {
            let path = root.join(stem).join(format!("{stem}.{ext}"));
            if path.is_file() {
                return Some(path);
            }
        }
        for ext in EXTS {
            let path = root.join(format!("{stem}.{ext}"));
            if path.is_file() {
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
