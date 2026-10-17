//! Date-gated events, GML `UberCont/Other_4.gml:33-36`: `april_fools`,
//! `xmas`, `halloween` and `birthday` read the local clock once per boot
//! and gate flavor across the game (bandit costumes, snow, fog, tips, the
//! party-gun drop).

#[cfg(not(target_arch = "wasm32"))]
use jiff::Zoned;

use bevy_ecs::prelude::*;

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct DateEvents {
    pub april_fools: bool,
    pub xmas: bool,
    pub halloween: bool,
    pub birthday: bool,
}

impl DateEvents {
    pub fn today() -> Self {
        let (day, month) = local_day_month();
        Self {
            april_fools: day == 1 && month == 4,
            xmas: (day >= 25 && month == 12) || (day <= 7 && month == 1),
            halloween: day >= 30 && month == 10,
            birthday: day == 7 && month == 8,
        }
    }
}

/// `UberCont` session flag `global.party_gun_special_drop` rides along:
/// the birthday party gun spawns once per boot.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct DateRuntime {
    pub events: DateEvents,
    pub party_gun_dropped: bool,
}

#[cfg(not(target_arch = "wasm32"))]
fn local_day_month() -> (i8, i8) {
    let now = Zoned::now();
    (now.day(), now.month())
}

/// Browser build: `Date.now()` milliseconds resolve in UTC (jiff's system
/// tz has no wasm database), via the civil-from-days algorithm.
#[cfg(target_arch = "wasm32")]
fn local_day_month() -> (i8, i8) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as i8;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as i8;
    (day, month)
}
