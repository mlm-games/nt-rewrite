//! GML's own random stream, for the draws the original makes from `global.rng_state[]`.
//!
//! GameMaker's built-in `random()`/`irandom()` are a different generator (the
//! runner's own), so this type is **not** the port's general RNG: it is the law
//! of the scripts the game defines for itself. The decompiled source states it:
//!
//! ```text
//! scripts/scrRngStatesInit:   rng_m 2147483647, rng_a 1103515245, rng_c 12345
//! scripts/rng_next_int:      state = (rng_a * state + rng_c) % rng_m
//! scripts/rng_next_float:    next_int / (rng_m - 1)
//! scripts/rng_float:         next_float * n
//! scripts/rng_choose:        argument[1 + rng_next_int % (argument_count - 1)]
//! ```
//!
//! `scrRngStatesInit` seeds ten states, named by its `RNGStates` enum:
//! `Generation 0`, `Enemies 1`, `Props 2`, `WeaponDrops 3`, `Chest 4`,
//! `Skill 5`, `Popo 8`, `Pickups 9` (6 and 7 are unnamed, `NUM_RNG_STATES` is
//! 10). `scrRngStatesReset` re-seeds them every floor:
//!
//! ```text
//! _number = (global.seed + area*80687 + subarea*56041 + loops*90023) % 0x7fffffff
//! rng_state[i] = (_number + 79379 * i) % 0x7fffffff
//! ```
//!
//! with `area_vault` taking `global.seed`, `area_hq` forcing
//! `rng_state[Generation] = subarea + 24`, and `levseed` reading the
//! `Generation` state.
//!
//! The port today uses this stream where the original does: the big chests'
//! `WeaponDrops` roll ([`crate::decide_wep::decide_wep_seeded`]) and the
//! `Popo` portal placement ([`crate::idpd`]). It does **not** yet implement
//! `scrRngStatesReset`'s per-floor seeding: those two sites seed directly from
//! the run instead, which is a known fidelity gap recorded here rather than
//! hidden.
//!
//! Everything else the port draws comes from [`crate::rng_streams::SimStreams`]
//! (the engine's `SimRng`), because those draws correspond to the original's
//! built-in `random()` calls and the runner's generator is its own, separate
//! law.

use serde::{Deserialize, Serialize};

/// GML `scripts/scrRngStatesInit` `rng_m`: the modulus of the LCG.
pub const RNG_M: i64 = 2_147_483_647;

/// GML `scripts/scrRngStatesInit` `rng_a`: the multiplier of the LCG.
pub const RNG_A: i64 = 1_103_515_245;

/// GML `scripts/scrRngStatesInit` `rng_c`: the increment of the LCG.
pub const RNG_C: i64 = 12_345;

/// GML `scrRngStatesReset`: the per-state step of the seed derivation,
/// `rng_state[i] = (_number + 79379 * i) % 0x7fffffff`.
pub const RNG_STATE_STEP: i64 = 79_379;

/// The GML states the port draws from, by the names `scrRngStatesInit`'s
/// `RNGStates` enum gives them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum RngState {
    Generation = 0,
    Enemies = 1,
    Props = 2,
    WeaponDrops = 3,
    Chest = 4,
    Skill = 5,
    Popo = 8,
    Pickups = 9,
}

/// One GameMaker stream: the LCG `scripts/rng_next_int` drives.
///
/// One integer of state, exactly as GML keeps `global.rng_state[i]`, so a save
/// can carry the position in one number. The state is `serde`-serialisable and
/// `Copy` for that reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GmlRng {
    state: i32,
}

impl GmlRng {
    /// A stream at `seed`: GML `random_set_seed`.
    pub const fn seeded(seed: i32) -> Self {
        Self { state: seed }
    }

    /// The stream for one `RNGStates` entry at `seed`, as `scrRngStatesReset`
    /// derives it: `(seed + 79379 * state) % 0x7fffffff`.
    pub const fn at_state(seed: i32, state: RngState) -> Self {
        Self {
            state: ((seed as i64 + RNG_STATE_STEP * state as i64) % RNG_M) as i32,
        }
    }

    /// GML `rng_next_int`: `state = (rng_a * state + rng_c) % rng_m`, and
    /// that new state is the value.
    pub fn next_int(&mut self) -> i32 {
        self.state = ((RNG_A * self.state as i64 + RNG_C) % RNG_M) as i32;
        self.state
    }

    /// GML `rng_next_float`: `next_int / (rng_m - 1)`, so a draw can reach
    /// exactly `1.0`.
    pub fn next_float(&mut self) -> f32 {
        self.next_int() as f32 / (RNG_M - 1) as f32
    }

    /// GML `rng_float`: `next_float * n`.
    pub fn float(&mut self, max: f32) -> f32 {
        self.next_float() * max
    }

    /// GML `rng_choose`: the zero-based index into `count` arguments.
    ///
    /// A count of zero returns `0` and consumes no draw.
    pub fn choose(&mut self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        let index = self.next_int() as i64 % count as i64;
        if index < 0 { 0 } else { index as usize }
    }

    /// GML `irandom_range(lo, hi)`, inclusive of both ends.
    ///
    /// The span is computed in 64 bits so a range wide enough to overflow an
    /// `i32` still draws. An empty range (`hi <= lo`) returns `lo` and
    /// consumes no draw.
    pub fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        if hi <= lo {
            return lo;
        }
        let span = hi as i64 - lo as i64 + 1;
        lo + (self.next_int() as i64 % span).max(0) as i32
    }

    /// The stream position: the LCG state, GML `global.rng_state[i]`.
    pub const fn state(&self) -> i32 {
        self.state
    }

    /// GML `random_set_seed`: the stream continues from `state`.
    pub const fn set_state(&mut self, state: i32) {
        self.state = state;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seed the conformance values below were taken at.
    const SEED: i32 = 1_234_567;

    #[test]
    fn the_lcg_law_is_pinned_value_for_value() {
        let mut rng = GmlRng::seeded(SEED);
        let ints = [
            rng.next_int(),
            rng.next_int(),
            rng.next_int(),
            rng.next_int(),
            rng.next_int(),
            rng.next_int(),
            rng.next_int(),
            rng.next_int(),
        ];
        assert_eq!(
            ints,
            [
                2027313107, 5238083, 21869072, 56711793, 1076153581, 1398781804, 503576447,
                935315329
            ]
        );

        let mut rng = GmlRng::seeded(SEED);
        let floats = [
            rng.next_float(),
            rng.next_float(),
            rng.next_float(),
            rng.next_float(),
            rng.next_float(),
            rng.next_float(),
        ];
        assert_eq!(
            floats,
            [
                0.94404125,
                0.0024391725,
                0.01018358,
                0.026408486,
                0.5011231,
                0.65135854
            ]
        );

        let mut rng = GmlRng::seeded(SEED);
        let picks = [
            rng.choose(5),
            rng.choose(5),
            rng.choose(5),
            rng.choose(5),
            rng.choose(5),
            rng.choose(5),
        ];
        assert_eq!(picks, [2, 3, 2, 3, 1, 4]);

        let mut rng = GmlRng::seeded(SEED);
        let ranges = [
            rng.range_i32(1, 127),
            rng.range_i32(1, 127),
            rng.range_i32(1, 127),
        ];
        assert_eq!(ranges, [43, 96, 54]);
    }

    #[test]
    fn the_state_derivation_is_pinned() {
        assert_eq!(
            GmlRng::at_state(1_234_567, RngState::Generation).state(),
            1_234_567
        );
        assert_eq!(
            GmlRng::at_state(1_234_567, RngState::Enemies).state(),
            1_313_946
        );
        assert_eq!(
            GmlRng::at_state(1_234_567, RngState::WeaponDrops).state(),
            1_472_704
        );
        assert_eq!(GmlRng::at_state(42, RngState::Popo).state(), 635_074);
    }

    #[test]
    fn a_save_resumes_the_identical_stream() {
        let mut rng = GmlRng::seeded(SEED);
        rng.next_int();
        rng.next_int();
        let state = rng.state();

        let mut restored = GmlRng::seeded(0);
        restored.set_state(state);
        assert_eq!(restored.next_int(), 21_869_072);
    }
}
