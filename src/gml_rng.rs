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
//! The port uses this stream where the original does: the `Popo` portal
//! placement ([`crate::idpd`]) and every draw `scrRngStatesReset` derives.
//! Chest weapon rolls read it through [`RngStates`] only for their creation
//! dropseed; the roll itself is the built-in generator
//! ([`crate::runner_rng`]).
//!
//! [`RngStates`] is `global.rng_state[]` itself: ten slots, re-seeded on every
//! floor by [`RngStates::reset`] (GML `scrRngStatesReset`), which also derives
//! the floor's built-in seed `_number`.
//!
//! Everything else the port draws corresponds to the original's built-in
//! `random()` family, which is [`crate::runner_rng::RunnerRng`]: the runtime's
//! own WELL512, not this LCG.

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

/// GML `RNGStates.NUM_RNG_STATES`: ten slots, six of them named.
pub const NUM_RNG_STATES: usize = 10;

/// GML `area_vault`: the reset takes the run seed whole here.
pub const AREA_VAULT: i32 = 100;

/// GML `area_hq`: the reset pins `Generation` to `subarea + 24` here.
pub const AREA_HQ: i32 = 106;

/// `global.rng_state[]`, plus the built-in generator's floor seed.
///
/// `scrRngStatesReset` runs once per floor (GML `objects/Vlambeer/Create_0:42`,
/// the room's controller), so every field the port derives from it is a
/// per-floor value:
///
/// ```text
/// _number = (global.seed + area*80687 + subarea*56041 + loops*90023) % 0x7fffffff
/// if (area == area_vault) _number = global.seed
/// random_set_seed(_number)
/// rng_state[i] = (_number + 79379 * i) % 0x7fffffff
/// if (area == area_hq) rng_state[Generation] = subarea + 24
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RngStates {
    states: [GmlRng; NUM_RNG_STATES],
    builtin_seed: u32,
}

impl RngStates {
    /// GML `scrRngStatesReset()`, with `run_seed` as `global.seed`, `area` as
    /// the GML area number, `subarea` as `GameCont.subarea` and `loops` as
    /// `GameCont.loops`.
    pub fn reset(run_seed: u64, area: i32, subarea: u32, loops: u32) -> Self {
        let span = RNG_M as i128;
        let mut number = (run_seed as i128
            + i128::from(area) * 80_687
            + i128::from(subarea) * 56_041
            + i128::from(loops) * 90_023)
            % span;
        if area == AREA_VAULT {
            number = run_seed as i128;
        }

        let mut states = [GmlRng::seeded(0); NUM_RNG_STATES];
        for (i, state) in states.iter_mut().enumerate() {
            *state = GmlRng::seeded(
                ((number + i128::from(RNG_STATE_STEP) * i as i128) % span) as i32,
            );
        }
        if area == AREA_HQ {
            states[0] = GmlRng::seeded(subarea as i32 + 24);
        }

        // `random_set_seed(floor(_seed))` masks the low 32 bits, so the vault's
        // unreduced seed reaches the runner the same way it does in GML.
        let builtin_seed = if area == AREA_VAULT {
            (number as u64 & 0xFFFF_FFFF) as u32
        } else {
            number as u32
        };

        Self {
            states,
            builtin_seed,
        }
    }

    /// The named state's stream, GML `global.rng_state[RNGStates.X]`.
    pub fn get(&mut self, state: RngState) -> &mut GmlRng {
        &mut self.states[state as usize]
    }

    /// A state by index, for the two slots `RNGStates` leaves unnamed.
    pub fn idx(&mut self, index: usize) -> &mut GmlRng {
        &mut self.states[index]
    }

    /// GML `scrRngStatesReset`'s `random_set_seed(_number)`: the seed the
    /// floor's built-in generator starts from.
    pub const fn builtin_seed(&self) -> u32 {
        self.builtin_seed
    }

    /// GML `GameCont.levseed`, which is `rng_state[Generation]`.
    pub const fn levseed(&self) -> i32 {
        self.states[RngState::Generation as usize].state()
    }

    /// Every state, in `RNGStates` order.
    pub const fn states(&self) -> &[GmlRng; NUM_RNG_STATES] {
        &self.states
    }
}

impl Default for RngStates {
    /// `scrRngStatesInit` before a run exists: seed 1, campfire, first subarea.
    fn default() -> Self {
        Self::reset(1, 0, 1, 0)
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

    /// `scrRngStatesReset`, including the two areas that step out of the
    /// formula: the vault takes the run seed whole, HQ pins `Generation`.
    #[test]
    fn the_floor_reset_follows_the_gml_formula() {
        let (seed, area, subarea, loops) = (1_234u64, 1i32, 3u32, 1u32);
        let states = RngStates::reset(seed, area, subarea, loops);
        let number = (seed as i64
            + i64::from(area) * 80_687
            + i64::from(subarea) * 56_041
            + i64::from(loops) * 90_023)
            % RNG_M;
        assert_eq!(states.builtin_seed(), number as u32);
        for i in 0..NUM_RNG_STATES {
            let want = ((number + RNG_STATE_STEP * i as i64) % RNG_M) as i32;
            assert_eq!(states.states()[i].state(), want);
        }
        assert_eq!(states.levseed(), states.states()[0].state());

        let vault = RngStates::reset(seed, AREA_VAULT, subarea, loops);
        assert_eq!(vault.builtin_seed(), seed as u32);
        assert_eq!(vault.states()[0].state(), (seed as i64 % RNG_M) as i32);

        let hq = RngStates::reset(seed, AREA_HQ, 5, 0);
        assert_eq!(hq.levseed(), 5 + 24);

        // A different floor is a different stream, everywhere.
        assert_ne!(
            RngStates::reset(seed, area, subarea, loops).states()[1].state(),
            RngStates::reset(seed, area, subarea + 1, loops).states()[1].state()
        );
    }
}
