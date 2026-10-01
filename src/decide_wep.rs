//! Weapon choice. GML `scripts/scrDecideWep/scrDecideWep.gml` verbatim
//! (plus `scrDecideWepGold`): tier-gated uniform roll over ids 1..127
//! with owned-weapon, tutorial, and special-gate rejection.
//!
//! `GameCont.hard` is the live `Run.hard` counter (see [`game_hard`]
//! below): `Create_0` seeds it at 0 (13 in hardmode) and `Other_5:136`
//! adds `hardmode ? 2 : 1` per room, so it equals
//! `scrAreaGetDifficulty(area, subarea, loops)`.

use bevy_ecs::prelude::*;
use rand::RngExt;

use crate::comps_a::{GameCleanup, Health, Hitbox, LevelCleanup, Run, Team, Velocity};
use crate::comps_b::Ally;
use crate::data::{AmmoKind, RaceId, WeaponId};
use crate::spatial::Pos;
use crate::time::{GTimer, TimerMode};
use crate::weapon_runtime::weapon_meta;
use crate::weapons_data::AmmoType;

/// Gold pools (`scrDecideWepGold` verbatim): pre-loop ids 40-45,
/// looped ids 98-103.
pub const GOLD_TIER1: [WeaponId; 6] = [
    WeaponId(40),
    WeaponId(41),
    WeaponId(42),
    WeaponId(43),
    WeaponId(44),
    WeaponId(45),
];
pub const GOLD_TIER2: [WeaponId; 6] = [
    WeaponId(98),
    WeaponId(99),
    WeaponId(100),
    WeaponId(101),
    WeaponId(102),
    WeaponId(103),
];

/// Special-gate ids (`scrDecideWep:44-49` verbatim).
pub const SUPER_DISC_GUN: WeaponId = WeaponId(104);
pub const GOLDEN_NUKE_LAUNCHER: WeaponId = WeaponId(122);
pub const GOLDEN_DISC_GUN: WeaponId = WeaponId(123);
pub const GUN_GUN: WeaponId = WeaponId(125);
/// GML `macros_general:480` `wep_frog_pistol`; `WEAPON_GOLDEN_FROG_PISTOL`
/// already lives in `data.rs`. Used by the `ProtoChest` room-end upgrade.
pub const FROG_PISTOL: WeaponId = WeaponId(120);
pub const GOLDEN_FROG_PISTOL: WeaponId = crate::data::WEAPON_GOLDEN_FROG_PISTOL;

/// GML `GameCont.hard` for the current room - the live counter
/// `scrDecideWep` reads (`_tier_max = GameCont.hard + _extra`).
/// `GameCont/Create_0:9` seeds it at 0 and `:85-87` at 13 (also
/// `loops++`) under hardmode; `GameCont/Other_5:136` then adds
/// `hardmode ? 2 : 1` on every room end. It is deliberately NOT
/// `scrAreaGetDifficulty` (that formula is only used by custom runs,
/// `scripts/scrRunStart/scrRunStart.gml:64`): on the 15-floor route the
/// live counter trails `scrAreaGetDifficulty` by `loops + 1`, and under
/// hardmode it starts 4 below it. Deriving it from `Run.floor` instead
/// double-counts the loop, because `Run.floor` is the *global* floor
/// (`31 * loops + route_floor`).
pub fn game_hard(run: &Run) -> i32 {
    run.hard as i32
}

/// GML `scripts/scrRngStatesInit` LCG constants (`rng_m`, `rng_a`, `rng_c`).
pub const RNG_M: i64 = 2_147_483_647;
pub const RNG_A: i64 = 1_103_515_245;
pub const RNG_C: i64 = 12_345;

/// GML `RNGStates.WeaponDrops` (index 3) as a standalone LCG stream:
/// `rng_next_int` is `state = (rng_a * state + rng_c) % rng_m` and
/// `rng_next_float` divides that by `rng_m - 1`. `random_set_seed` is what
/// `BigWeaponChest/Collision_Player.gml:14` and
/// `CursedBigChest/Collision_Player.gml:14` do with the chest's
/// `dropseed` right before rolling their three weapons, so a chest's
/// contents are fixed at spawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeaponDropsRng {
    state: i32,
}

impl WeaponDropsRng {
    /// GML `global.rng_state[RNGStates.X] = (_number + 79379 * X) % 0x7fffffff`.
    pub fn new(seed: i32) -> Self {
        Self { state: seed }
    }

    /// GML `random_set_seed`.
    pub fn set_seed(&mut self, seed: i32) {
        self.state = seed;
    }

    /// GML `rng_next_int`.
    pub fn next_int(&mut self) -> i32 {
        self.state = ((RNG_A * self.state as i64 + RNG_C) % RNG_M) as i32;
        self.state
    }

    /// GML `rng_next_float`.
    pub fn next_float(&mut self) -> f32 {
        self.next_int() as f32 / (RNG_M - 1) as f32
    }

    /// GML `rng_float(_state, _number)`.
    pub fn float(&mut self, n: f32) -> f32 {
        self.next_float() * n
    }

    /// GML `rng_choose(_state, ...)`: `argument[1 + rng_next_int % count]`,
    /// returned as the zero-based index.
    pub fn choose(&mut self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        (self.next_int() % count as i32).max(0) as usize
    }

    /// GML `irandom_range(lo, hi)` on this stream, inclusive.
    pub fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        if hi <= lo {
            return lo;
        }
        lo + (self.next_int() % (hi - lo + 1)).max(0)
    }
}

/// Everything `decide_wep` needs from the world (single-player: the one
/// player; GML reads `instance_nearest(x, y, Player)`).
#[derive(Clone)]
pub struct DecideCtx {
    /// `GameCont.hard` (`scrAreaGetDifficulty`: `route_floor +
    /// loops*16`; hardmode's 13 start + 2/floor is already inside the live
    /// `Run.hard`). Build it with [`game_hard`].
    pub hard: i32,
    pub hardmode: bool,
    /// Robot players in the run (single-player: 0/1).
    pub robots: u32,
    /// Any Robot holding Refined Taste.
    pub refined_taste: bool,
    /// Current crown is Guns.
    pub crown_guns: bool,
    /// Tutorial level active.
    pub tutorial: bool,
    /// Target race (Steroids skips the owned check).
    pub target_race: RaceId,
    /// Target's held weapon ids.
    pub owned: Vec<WeaponId>,
}

/// GML `scrDecideWep(_extra, _curse)`, drawing ids from the global stream.
/// Weapon chests use this; `BigWeaponChest` / `CursedBigChest` use
/// [`decide_wep_seeded`] so their contents are fixed at spawn by
/// `random_set_seed(dropseed)`.
pub fn decide_wep(rng: &mut impl RngExt, ctx: &DecideCtx, extra: i32, curse: bool) -> WeaponId {
    decide_wep_drawing(&mut || WeaponId(rng.random_range(1..=127)), ctx, extra, curse)
}

/// Same law, drawing ids from the `RNGStates.WeaponDrops` LCG
/// (`BigWeaponChest/Collision_Player.gml:14` and
/// `CursedBigChest/Collision_Player.gml:14` `random_set_seed(dropseed)`).
pub fn decide_wep_seeded(
    rng: &mut WeaponDropsRng,
    ctx: &DecideCtx,
    extra: i32,
    curse: bool,
) -> WeaponId {
    decide_wep_drawing(&mut || WeaponId(rng.range_i32(1, 127) as u8), ctx, extra, curse)
}

/// GML `scrDecideWep` with the `irandom_range(1, maxwep - 1)` id draw
/// supplied by the caller, so the global stream and the WeaponDrops LCG
/// share one copy of the tier/rejection law.
fn decide_wep_drawing(
    draw: &mut dyn FnMut() -> WeaponId,
    ctx: &DecideCtx,
    extra: i32,
    curse: bool,
) -> WeaponId {
    let mut tier_max = ctx.hard + extra;
    if ctx.hardmode {
        tier_max = (tier_max - 13) / 3;
    }
    tier_max += ctx.robots as i32;
    let mut tier_min = -1;
    if curse {
        tier_min = (tier_max + extra).clamp(1, 3);
    }
    if ctx.refined_taste {
        tier_min = (tier_max + extra).clamp(1, 6);
    }
    if tier_min > tier_max {
        tier_max = tier_min + 1;
    }
    let mut wep = WeaponId::REVOLVER;
    for _ in 0..999 {
        let id = draw();
        let meta = weapon_meta(id);
        if meta.wep_area < 0 || meta.wep_area as i32 > tier_max || (meta.wep_area as i32) < tier_min
        {
            continue;
        }
        if ctx.target_race != RaceId::Steroids && ctx.owned.contains(&id) {
            continue;
        }
        if ctx.tutorial {
            let ty = match meta.wep_type {
                AmmoType::None => AmmoKind::None,
                AmmoType::Bullets => AmmoKind::Bullets,
                AmmoType::Shells => AmmoKind::Shells,
                AmmoType::Bolts => AmmoKind::Bolts,
                AmmoType::Explosives => AmmoKind::Explosives,
                AmmoType::Energy => AmmoKind::Energy,
            };
            if ty == AmmoKind::None || ty == AmmoKind::Explosives {
                continue;
            }
        }
        if (id == SUPER_DISC_GUN && !curse)
            || ((id == GOLDEN_DISC_GUN || id == GOLDEN_NUKE_LAUNCHER) && !ctx.hardmode)
            || (id == GUN_GUN && !ctx.crown_guns)
        {
            continue;
        }
        wep = id;
        break;
    }
    wep
}

/// GML `scrDecideWep.gml:20-23` verbatim:
/// ```gml
/// if (scr_ultra_get(Race.Robot, UltraSkill.RefinedTaste)) {
///     instance_create(x, y, RobotA)
///     _tier_min = median(6, 1, _tier_max + _extra)
/// }
/// ```
/// The tier half lives in [`decide_wep`]; every caller rolls through this
/// (or [`decide_wep_seeded_at`]) so the ally is never dropped. GML
/// `objects/RobotA` is an eventless FX object (`eventList` empty, no
/// parent), so it lands as the port's `Ally` (same shape as Rebel's
/// Riot summon) and answers with that object's create sting.
pub fn decide_wep_at(
    commands: &mut Commands,
    rng: &mut impl RngExt,
    ctx: &DecideCtx,
    pos: glam::Vec2,
    extra: i32,
    curse: bool,
) -> WeaponId {
    let wep = decide_wep(rng, ctx, extra, curse);
    spawn_refined_taste_ally(commands, ctx, pos);
    wep
}

/// [`decide_wep_at`] on the `RNGStates.WeaponDrops` LCG: the big-chest
/// spawn-fixed roll.
pub fn decide_wep_seeded_at(
    commands: &mut Commands,
    rng: &mut WeaponDropsRng,
    ctx: &DecideCtx,
    pos: glam::Vec2,
    extra: i32,
    curse: bool,
) -> WeaponId {
    let wep = decide_wep_seeded(rng, ctx, extra, curse);
    spawn_refined_taste_ally(commands, ctx, pos);
    wep
}

fn spawn_refined_taste_ally(commands: &mut Commands, ctx: &DecideCtx, pos: glam::Vec2) {
    if !ctx.refined_taste {
        return;
    }
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        Ally {
            life: GTimer::from_seconds(12.0, TimerMode::Once),
            shoot: GTimer::from_seconds(8.0 / 30.0, TimerMode::Repeating),
        },
        Team::Player,
        Health {
            hp: 12,
            max: 12,
            invuln: GTimer::from_seconds(0.5, TimerMode::Once),
        },
        Hitbox { radius: 10.0 },
        Velocity(glam::Vec2::ZERO),
        Pos(pos),
    ));
    commands.queue(|world: &mut World| {
        world.init_resource::<crate::msg::Queue<crate::audio::AudioCue>>();
        if let Some(mut cues) =
            world.get_resource_mut::<crate::msg::Queue<crate::audio::AudioCue>>()
        {
            cues.push(crate::audio::AudioCue {
                name: "sndAllySpawn",
                volume: 1.0,
                variance: 0.2,
            });
        }
    });
}

/// GML `scrDecideWepGold`: tier pool by loop count, owned-reject loop
/// (Steroids exempt). Unbounded in GML; capped defensively here.
pub fn decide_wep_gold(
    rng: &mut impl RngExt,
    loops: u32,
    owned: &[WeaponId],
    is_steroids: bool,
) -> WeaponId {
    let pool = if loops > 0 { GOLD_TIER2 } else { GOLD_TIER1 };
    // GML `do { _wep = rng_choose(...) } until (steroids || !has_weapon)`.
    let mut wep = pool[rng.random_range(0..pool.len())];
    for _ in 0..1000 {
        wep = pool[rng.random_range(0..pool.len())];
        if is_steroids || !owned.contains(&wep) {
            break;
        }
    }
    wep
}

#[cfg(test)]
mod decide_wep_tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn ctx(hard: i32) -> DecideCtx {
        DecideCtx {
            hard,
            hardmode: false,
            robots: 0,
            refined_taste: false,
            crown_guns: false,
            tutorial: false,
            target_race: RaceId::Fish,
            owned: Vec::new(),
        }
    }

    #[test]
    fn floor1_tier_caps_at_2() {
        let c = ctx(1);
        let mut rng: StdRng = StdRng::seed_from_u64(7);
        for _ in 0..200 {
            let w = decide_wep(&mut rng, &c, 0, false);
            let area = weapon_meta(w).wep_area;
            assert!(area >= 0 && area <= 1, "id {} area {area}", w.0);
        }
    }

    #[test]
    fn chest_extra_widens_tier() {
        let c = ctx(1);
        let mut rng: StdRng = StdRng::seed_from_u64(42);
        let mut saw_high = false;
        for _ in 0..500 {
            let w = decide_wep(&mut rng, &c, 1, false);
            if weapon_meta(w).wep_area == 2 {
                saw_high = true;
                break;
            }
        }
        assert!(saw_high);
    }

    #[test]
    fn owned_rejected_for_non_steroids() {
        let mut c = ctx(30);
        c.owned = vec![WeaponId(4), WeaponId(5), WeaponId(6)];
        let mut rng: StdRng = StdRng::seed_from_u64(9);
        for _ in 0..200 {
            let w = decide_wep(&mut rng, &c, 0, false);
            assert!(!c.owned.contains(&w) || weapon_meta(w).wep_area < 0);
        }
    }

    #[test]
    fn gold_pools_split_by_loops() {
        let mut rng: StdRng = StdRng::seed_from_u64(3);
        for _ in 0..50 {
            let w = decide_wep_gold(&mut rng, 0, &[], false);
            assert!(GOLD_TIER1.contains(&w));
            let w = decide_wep_gold(&mut rng, 2, &[], false);
            assert!(GOLD_TIER2.contains(&w));
        }
    }

    #[test]
    fn special_gates_hold() {
        let c = ctx(30);
        let mut rng: StdRng = StdRng::seed_from_u64(11);
        for _ in 0..500 {
            let w = decide_wep(&mut rng, &c, 0, false);
            assert_ne!(w, SUPER_DISC_GUN);
            assert_ne!(w, GOLDEN_DISC_GUN);
            assert_ne!(w, GOLDEN_NUKE_LAUNCHER);
            assert_ne!(w, GUN_GUN);
        }
    }
}
