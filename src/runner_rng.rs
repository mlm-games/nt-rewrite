//! GameMaker's built-in generator: the law behind `random()`, `irandom()`,
//! `irandom_range()`, `choose()` and `random_set_seed()`.
//!
//! The original draws from two generators, and this is the big one: everything
//! that is *not* `global.rng_state[]` (that one is [`crate::gml_rng`]).
//!
//! The law is read off the runner the game ships
//! (`nuclear_throne_new/game/nuclearthrone`, the Linux export; the installed
//! Studio runtime at `runtime-2026.100.0.1098` agrees word for word):
//!
//! - `random_set_seed` (`0x2277870`): `s = s * 214013 + 2531011` over `u32`,
//!   sixteen times, storing `s >> 16` into each state word; then `index = 0`.
//! - the draw (`0x2277960`) is a WELL512 step over the 16-word state, with the
//!   polynomial `0xDA442D20` and a `(s[i_13] ^ a) << 18` mixing term (the
//!   published WELL512 uses `0xDA442D24` and `b << 18`; this is the variant the
//!   runtime actually carries, and the one every built-in function agrees with).
//! - `random(x)` (`0x2277a10`): one draw, `x * (draw / 2^32)`; `random(1)` is
//!   the draw divided by `2^32`.
//! - `irandom(n)` is `irandom_range(0, n)`; `irandom_range(a, b)` takes the
//!   63-bit value `((draw2 & 0x7fffffff) << 32) | draw1` and reduces it modulo
//!   `|a - b| + 1` (plain modulo, not rejection), added to the low end.
//! - `choose` takes `args[draw % n]`, one draw.
//!
//! `random_use_old_version` (which would switch to the poly `0xDA442D24`) is
//! never called by the source, and this runner contains only the poly above, so
//! that is the law on every build the game ships.
//!
//! The port uses this generator for the draws the original makes from it,
//! wherever the original re-seeds before drawing (`random_set_seed(x)` followed
//! by rolls) the sequence therefore matches the original exactly; the ambient
//! per-subsystem streams in [`crate::rng_streams`] use the same law but draw in
//! the port's own order, which is not the original's instance order.

use repame_sim::rng::Rng;

const LCG_MUL: u32 = 214_013;
const LCG_ADD: u32 = 2_531_011;
const WELL_POLY: u32 = 0xDA44_2D20;
const TWO_POW_32: f64 = 4_294_967_296.0;

/// The runner's generator: 16 state words, an index, and the seed
/// `random_get_seed` would return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunnerRng {
    state: [u32; 16],
    index: u32,
    seed: u32,
}

impl RunnerRng {
    /// GML `random_set_seed(seed)`.
    pub fn seeded(seed: u32) -> Self {
        let mut state = [0u32; 16];
        let mut s = seed;
        for word in &mut state {
            s = s.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
            *word = s >> 16;
        }
        Self {
            state,
            index: 0,
            seed,
        }
    }

    /// GML `random_get_seed()`.
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// One raw draw: the WELL512 step at `0x2277960`.
    pub fn raw_u32(&mut self) -> u32 {
        let i = self.index as usize;
        let a = self.state[i];
        let c0 = self.state[(i + 13) & 15];
        let x = c0 ^ a;
        let b = x ^ (a << 16) ^ (c0 << 15);
        let c_raw = self.state[(i + 9) & 15];
        let c = c_raw ^ (c_raw >> 11);
        let a_new = b ^ c;
        self.state[i] = a_new;
        let d = (a_new << 5) & WELL_POLY;
        let j = (i + 15) & 15;
        let a2 = self.state[j];
        let r = a2 ^ (a2 << 2) ^ d ^ c ^ (c << 28) ^ (x << 18);
        self.state[j] = r;
        self.index = j as u32;
        r
    }

    /// The 63-bit value `irandom`/`irandom_range` reduce: two draws, the second
    /// masked to its low 31 bits.
    pub fn draw63(&mut self) -> u64 {
        let low = self.raw_u32() as u64;
        let high = (self.raw_u32() as u64) & 0x7fff_ffff;
        (high << 32) | low
    }

    /// GML `random(1)`: one draw over `2^32`.
    pub fn unit(&mut self) -> f64 {
        f64::from(self.raw_u32()) / TWO_POW_32
    }

    /// GML `random(max)`.
    pub fn gml_random(&mut self, max: f64) -> f64 {
        self.unit() * max
    }

    /// GML `random_range(a, b)`: the two ends swap rather than panic.
    pub fn gml_random_range(&mut self, a: f64, b: f64) -> f64 {
        let (lo, span) = if b <= a { (b, a - b) } else { (a, b - a) };
        lo + self.unit() * span
    }

    /// GML `iScript_Random`: the 63-bit draw modulo the span, signed.
    fn signed_span(&mut self, span: i64) -> i64 {
        if span == 0 {
            return 0;
        }
        let sign = if span > -1 { 1 } else { -1 };
        let width = span * sign;
        (self.draw63() % width as u64) as i64 * sign
    }

    /// GML `irandom(n)`, inclusive, and negative `n` counts downwards.
    pub fn irandom(&mut self, n: i32) -> i32 {
        let sign = if n > -1 { 1 } else { -1 };
        self.signed_span(i64::from(n) + i64::from(sign)) as i32
    }

    /// GML `irandom_range(a, b)`, inclusive of both ends.
    pub fn irandom_range(&mut self, a: i32, b: i32) -> i32 {
        let (lo, hi) = if b <= a { (b, a) } else { (a, b) };
        lo.wrapping_add(self.signed_span(i64::from(hi) - i64::from(lo) + 1) as i32)
    }

    /// GML `choose`: one draw modulo the argument count.
    pub fn choose(&mut self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        (self.raw_u32() as usize) % count
    }

    /// The state words, for tests and for pinning a replay.
    pub fn state_words(&self) -> [u32; 16] {
        self.state
    }
}

impl Rng for RunnerRng {
    fn next_u64(&mut self) -> u64 {
        self.draw63()
    }

    fn next_u32(&mut self) -> u32 {
        self.raw_u32()
    }

    fn unit_f32(&mut self) -> f32 {
        self.unit() as f32
    }

    fn unit_f64(&mut self) -> f64 {
        self.unit()
    }

    fn bool(&mut self) -> bool {
        self.unit() < 0.5
    }

    fn below(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            return 0;
        }
        (self.draw63() % u64::from(bound)) as u32
    }

    fn below_u64(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        self.draw63() % bound
    }
}

/// The GML integer-roll shapes, so one function can roll on the runtime's
/// generator ([`RunnerRng`]) or on the game's own `rng_state[]` LCG
/// ([`crate::gml_rng::GmlRng`]).
pub trait GmlDraw {
    /// GML `irandom(n)`, inclusive, negative `n` counting downwards.
    fn irandom(&mut self, n: i32) -> i32;

    /// GML `irandom_range(a, b)`, inclusive of both ends.
    fn irandom_range(&mut self, a: i32, b: i32) -> i32;
}

impl GmlDraw for RunnerRng {
    fn irandom(&mut self, n: i32) -> i32 {
        RunnerRng::irandom(self, n)
    }

    fn irandom_range(&mut self, a: i32, b: i32) -> i32 {
        RunnerRng::irandom_range(self, a, b)
    }
}

impl GmlDraw for crate::gml_rng::GmlRng {
    fn irandom(&mut self, n: i32) -> i32 {
        self.range_i32(0, n)
    }

    fn irandom_range(&mut self, a: i32, b: i32) -> i32 {
        self.range_i32(a, b)
    }
}

impl<T: GmlDraw + ?Sized> GmlDraw for &mut T {
    fn irandom(&mut self, n: i32) -> i32 {
        (**self).irandom(n)
    }

    fn irandom_range(&mut self, a: i32, b: i32) -> i32 {
        (**self).irandom_range(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The draw sequence, straight off the runner's instructions: the seeding
    /// loop at `0x2277870` and the WELL512 step at `0x2277960` transliterated
    /// one register at a time. If either the poly, a shift or the seed step is
    /// wrong, these move.
    #[test]
    fn draws_match_the_runner_word_for_word() {
        let mut rng = RunnerRng::seeded(1);
        let want = [
            2_890_553_580,
            3_794_959_378,
            1_569_430_999,
            1_417_655_494,
            3_612_992_126,
            3_169_889_685,
            4_272_636_520,
            989_713_543,
        ];
        for value in want {
            assert_eq!(rng.raw_u32(), value);
        }

        let mut rng = RunnerRng::seeded(12_345);
        let want = [
            2_404_397_659,
            1_312_964_952,
            3_362_076_531,
            4_101_477_359,
            38_709_702,
            1_230_521_605,
            3_905_153_179,
            1_702_720_303,
        ];
        for value in want {
            assert_eq!(rng.raw_u32(), value);
        }
    }

    #[test]
    fn seeding_is_the_lcg_taken_every_seventeenth_bit() {
        let rng = RunnerRng::seeded(1);
        let mut s = 1u32;
        for word in rng.state_words() {
            s = s.wrapping_mul(214_013).wrapping_add(2_531_011);
            assert_eq!(word, s >> 16);
            assert_eq!(word >> 16, 0);
        }
        assert_eq!(RunnerRng::seeded(7).seed(), 7);
    }

    /// `random(1)` is one draw over `2^32`; `irandom_range` is the 63-bit value
    /// modulo the span; `choose` is one draw modulo the count.
    #[test]
    fn value_shapes_are_the_runners() {
        let mut rng = RunnerRng::seeded(1);
        assert_eq!(rng.unit(), 2_890_553_580.0 / 4_294_967_296.0);
        let mut rng = RunnerRng::seeded(1);
        assert_eq!(
            rng.gml_random(10.0),
            2_890_553_580.0 / 4_294_967_296.0 * 10.0
        );

        let mut rng = RunnerRng::seeded(1);
        let want: Vec<i32> = (0..4).map(|_| rng.irandom_range(0, 9)).collect();
        assert_eq!(want, [0, 3, 8, 8]);

        let mut rng = RunnerRng::seeded(1);
        let want: Vec<i32> = (0..4).map(|_| rng.irandom_range(10, 19)).collect();
        assert_eq!(want, [10, 13, 18, 18]);

        let mut rng = RunnerRng::seeded(1);
        let first = rng.raw_u32() as usize;
        assert_eq!(RunnerRng::seeded(1).choose(5), first % 5);
        assert_eq!(RunnerRng::seeded(1).irandom(9), RunnerRng::seeded(1).irandom_range(0, 9));
        assert_eq!(RunnerRng::seeded(1).irandom(0), 0);

        // A roll `random(2) < 1` is the first draw over `2^32` under `1/2`.
        assert!(RunnerRng::seeded(1).gml_random(2.0) > 1.0);
    }

    /// `below` is the runner's modulo, not the engine's rejection, so it consumes
    /// exactly two draws; a reversed `irandom_range` swaps like the runner does.
    #[test]
    fn ranges_use_the_runner_modulo() {
        let mut rng = RunnerRng::seeded(4242);
        let mut twin = RunnerRng::seeded(4242);
        assert_eq!(rng.below(1000), (twin.draw63() % 1000) as u32);
        assert_eq!(
            RunnerRng::seeded(4242).irandom_range(9, 0),
            RunnerRng::seeded(4242).irandom_range(0, 9)
        );
        assert_eq!(
            RunnerRng::seeded(4242).irandom(-4),
            -RunnerRng::seeded(4242).irandom(4)
        );
    }
}
