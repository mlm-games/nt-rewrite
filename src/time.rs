//! Game timers. GML counts `alarm[n]` down in steps and runs the matching
//! `Alarm_n` event once when it lands, re-arming inside the handler for repeats
//! (`Bandit/Create_0.gml:24`, `Bandit/Alarm_1.gml:1,46`); `-1` disarms
//! (`BanditBoss/Alarm_1.gml:24`). [`GTimer`] is the headless stand-in over plain
//! seconds, ticked at the fixed step (`dt = 1/30`, GML's 30 steps/s):
//! `tick` / `finished` / `just_finished` / `duration` / `fraction`, once vs
//! repeating. `just_finished` is true only on the completing tick, because
//! dozens of systems branch on that single-tick edge.

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

/// Once (clamp at duration) vs repeating (wrap elapsed) expiry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimerMode {
    Once,
    Repeating,
}

/// Countdown/up timer over plain seconds (the GML alarm stand-in).
#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GTimer {
    elapsed: f32,
    dur: f32,
    mode: TimerMode,
    just_finished: bool,
}

impl Default for GTimer {
    /// Disarmed (finished from birth), the GML `alarm[n] = -1` state.
    fn default() -> Self {
        Self::disarmed()
    }
}

impl GTimer {
    pub fn from_seconds(dur: f32, mode: TimerMode) -> Self {
        Self {
            elapsed: 0.0,
            dur: dur.max(0.0),
            mode,
            just_finished: false,
        }
    }

    /// Disarmed timer (finished from birth).
    pub fn disarmed() -> Self {
        Self {
            elapsed: 0.0,
            dur: 0.0,
            mode: TimerMode::Once,
            just_finished: false,
        }
    }

    /// Advance by `dt` (no return value; query `just_finished()` /
    /// `finished()` afterwards, GML's read-back from the alarm handler). A
    /// finished `Once` (including zero-duration) never re-fires: GML alarms
    /// are one-shot too, so a repeat comes from re-arming.
    pub fn tick(&mut self, dt: f32) {
        self.just_finished = false;
        if self.dur <= 0.0 {
            return;
        }
        match self.mode {
            TimerMode::Once => {
                let was = self.finished();
                self.elapsed = (self.elapsed + dt).min(self.dur);
                self.just_finished = !was && self.finished();
            }
            TimerMode::Repeating => {
                self.elapsed += dt;
                if self.elapsed >= self.dur {
                    self.elapsed %= self.dur;
                    self.just_finished = true;
                }
            }
        }
    }

    /// True only on the tick the timer completed (cleared next tick).
    pub fn just_finished(&self) -> bool {
        self.just_finished
    }

    /// Latching finished state (stays true for `Once`).
    pub fn finished(&self) -> bool {
        self.elapsed >= self.dur
    }

    /// Alias for call sites that read `is_finished`.
    pub fn is_finished(&self) -> bool {
        self.finished()
    }

    pub fn duration(&self) -> f32 {
        self.dur
    }

    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed
    }

    pub fn remaining_secs(&self) -> f32 {
        (self.dur - self.elapsed).max(0.0)
    }

    /// 0..1 progress (1.0 for zero-duration timers).
    pub fn fraction(&self) -> f32 {
        if self.dur <= 0.0 {
            return 1.0;
        }
        (self.elapsed / self.dur).clamp(0.0, 1.0)
    }

    /// Remaining seconds for countdown users.
    pub fn remaining(&self) -> f32 {
        self.remaining_secs()
    }

    /// Re-arm to full duration.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.just_finished = false;
    }

    /// Re-arm with a new duration (elapsed preserved).
    pub fn set_duration(&mut self, dur: f32) {
        self.dur = dur.max(0.0);
    }
}
