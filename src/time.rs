//! Game timers. The bevy build stored `bevy::time::Timer` inside
//! components (`Health.invuln`, `FireCooldown`, `GrenadeFuse`, …);
//! `bevy_time` is engine, so the port mirrors its tick surface
//! (`tick` / `finished` / `just_finished` / `duration` / `fraction`,
//! once vs repeating) over plain seconds. Fixed-step driven
//! (`dt = 1/30`); `just_finished` is true only on the completing tick,
//! exactly like bevy, because dozens of systems branch on it.

use bevy_ecs::prelude::*;

/// Once (clamp at duration) vs repeating (wrap elapsed) expiry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimerMode {
    Once,
    Repeating,
}

/// Countdown/up timer with bevy `Timer::tick` semantics.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct GTimer {
    elapsed: f32,
    dur: f32,
    mode: TimerMode,
    just_finished: bool,
}

impl Default for GTimer {
    /// Disarmed (finished from birth), matching bevy `Timer` users that
    /// construct-then-`set_duration`.
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

    /// Advance by `dt` (bevy `Timer::tick` parity: no return value;
    /// query `just_finished()` / `finished()` afterwards). A finished
    /// `Once` (including zero-duration) never re-fires, exactly like
    /// bevy's early-out on `is_finished`.
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

    /// Alias matching bevy call sites (`is_finished`).
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

    /// Remaining seconds (bevy `remaining` parity for countdown users).
    pub fn remaining(&self) -> f32 {
        self.remaining_secs()
    }

    /// Re-arm to full duration.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.just_finished = false;
    }

    /// Re-arm with a new duration (elapsed preserved, like bevy).
    pub fn set_duration(&mut self, dur: f32) {
        self.dur = dur.max(0.0);
    }
}

