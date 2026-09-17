//! Weapon choice. GML `scripts/scrDecideWep/scrDecideWep.gml` verbatim
//! (plus `scrDecideWepGold`): tier-gated uniform roll over ids 1..127
//! with owned-weapon, tutorial, and special-gate rejection.
//!
//! `GameCont.hard` is `run.floor + loops*16 (+13 hardmode)` (see
//! `worldgen::game_hard`; `GameCont/Create_0` + `Other_5`).

use rand::RngExt;

use crate::data::{AmmoKind, RaceId, WeaponId};
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

/// Everything `decide_wep` needs from the world (single-player: the one
/// player; GML reads `instance_nearest(x, y, Player)`).
pub struct DecideCtx {
    /// `GameCont.hard` (int: `floor + loops*16 + hardmode?13:0`).
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

/// GML `scrDecideWep(_extra, _curse)`.
pub fn decide_wep(rng: &mut impl RngExt, ctx: &DecideCtx, extra: i32, curse: bool) -> WeaponId {
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
    let mut wep = WeaponId(1);
    for _ in 0..999 {
        let id = WeaponId(rng.random_range(1..=127));
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

/// GML `scrDecideWepGold`: tier pool by loop count, owned-reject loop
/// (Steroids exempt). Unbounded in GML; capped defensively here.
pub fn decide_wep_gold(
    rng: &mut impl RngExt,
    loops: u32,
    owned: &[WeaponId],
    is_steroids: bool,
) -> WeaponId {
    let pool = if loops > 0 { GOLD_TIER2 } else { GOLD_TIER1 };
    let mut wep = pool[rng.random_range(0..pool.len())];
    for _ in 0..1000 {
        let id = pool[rng.random_range(0..pool.len())];
        if is_steroids || !owned.contains(&id) {
            wep = id;
            break;
        }
        wep = id;
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
