//! Level-generation core, mechanically ported from
//! `nt-recreated-bevy/src/game/world.rs` (plus `is_secret_area` from
//! `nt-recreated-bevy/src/game/secret_areas.rs`).
//!
//! Scope: `LevelPlan`, `PropKind`, `ChestSpawn`, `Maker` + `step_delta`,
//! `rng_choose`, `turn_table`, `gml_area`, `gml_area_from_run`,
//! `generation_goal`, `generation_goal_for_run`, `is_screen_end_wall`,
//! `floor_cell_for_wall`, `wall_cell_at`, `generate_level`,
//! `generate_palace_last`, `generate_campfire`, `generate_hq_last`,
//! `world_of`, `floor_in_world`, `boss_for_floor`,
//! `boss_for_floor_and_loop`, `is_secret_area`, plus the `run_for` test
//! helper and the oracle tests that only touch ported items.
//!
//! Enemy tables (`apply_loop_elite_substitutions`,
//! `loop_elite_candidates`, `default_area_enemies`), wall helpers
//! (`walls_cover_tile`, `populate_throne_room`, `trim_chests`,
//! `big_bandit_count`) and the full `populate` live here too.
//!
//! Transform notes:
//! - `use bevy::...` lines dropped. `crate::game::areas::AreaId` becomes
//!   `crate::data::AreaId`; `crate::game::secret_areas::is_secret_area`
//!   becomes the local `is_secret_area` ported below. Inner
//!   `use crate::game::areas::AreaId;` lines are covered by the top import.
//! - Logic, RNG call order, tables and comments are byte-identical to source.
//!
//! TODO(port) index: all helpers landed (cell_center_*, wall_*,
//! side_solid, walls_cover_tile*, is_boss_subarea*, game_hard,
//! populate + tables). Remaining world.rs surface (spawning entities
//! from plans, wall visuals) belongs to the setup phase, not this file.

use glam::Vec2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::HashSet;

use crate::comps_a::Run;
use crate::comps_b::ChestKind;
use crate::data::{AreaId, EnemyKind};

// Supporting consts from world.rs:16 (`WALL_PX`) and components.rs:47
// (`TILE`, via `crate::game::components::*`), copied verbatim (required by
// the byte-identical `wall_cell_at` / `cell_center_*` below).
pub const WALL_PX: f32 = 16.0;
pub const TILE: f32 = 32.0;

pub struct LevelPlan {
    pub floor_cells: Vec<(i32, i32)>,
    pub wall_cells: HashSet<(i32, i32)>,

    pub small_walls: Vec<(i16, i16)>,
    pub bones: Vec<(Vec2, bool)>,
    pub bone_sprite: &'static str,
    pub details: Vec<Vec2>,
    pub props: Vec<(PropKind, Vec2)>,
    pub chests: Vec<ChestSpawn>,
    pub enemies: Vec<(EnemyKind, Vec2)>,
    pub boss: Option<EnemyKind>,

    pub boss_count: u32,

    pub styleb: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PropKind {
    Cactus,
    BigSkull,
    GroundDecal,
    Barrel,
    Pipe,
    Tires,

    ToxicBarrel,
    Car,
    Cocoon,
    Snowman,
    Torch,

    GoldBarrel,

    BonePile,
    NightBonePile,
    NightCactus,
    Crystal,
    Hydrant,
    StreetLight,
    SodaMachine,
    Tube,
    MutantTube,
    Pillar,
    SmallGenerator,
    Anchor,
    WaterPlant,
    OasisBarrel,
    WaterMine,
    MoneyPile,
    YVStatue,
    Bush,
    BigFlower,
    PizzaBox,
    PlantPot,

    Cobweb,
    IcePatch,
    FireTrap,
    Mine,

    BigGenerator,
    ThroneStatue,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ChestSpawn {
    Weapon(Vec2),
    Ammo(Vec2),
    Rad(Vec2),
    /// Post-`scrPopChests` special kinds (permutations only; the
    /// generator never emits these).
    Custom(ChestKind, Vec2),
}

impl ChestSpawn {
    fn pos(self) -> Vec2 {
        match self {
            ChestSpawn::Weapon(p) | ChestSpawn::Ammo(p) | ChestSpawn::Rad(p) => p,
            ChestSpawn::Custom(_, p) => p,
        }
    }
}

fn gml_area(floor: u32) -> i32 {
    let rf = ((floor.max(1) - 1) % 15) + 1;
    match rf {
        1..=3 => 1,
        4 => 2,
        5..=7 => 3,
        8 => 4,
        9..=11 => 5,
        12 => 6,
        13..=15 => 7,
        _ => 7,
    }
}

pub fn gml_area_from_run(run: &Run) -> i32 {
    match run.area {
        AreaId::Desert => 1,
        AreaId::Oasis => 101,
        AreaId::Sewers => 2,
        AreaId::PizzaSewers => 102,
        AreaId::Scrapyards => 3,

        AreaId::City => 103,
        AreaId::CrystalCaves => 4,
        AreaId::CursedCaves => 104,
        AreaId::Vault | AreaId::CrownVault => 100,
        AreaId::FrozenCity => 5,
        AreaId::Jungle => 105,
        AreaId::Labs => 6,
        AreaId::HQ => 106,
        AreaId::Palace | AreaId::Campfire => 7,
        _ => gml_area(run.floor),
    }
}

pub fn generation_goal(floor: u32) -> usize {
    let _ = floor;
    110
}

fn generation_goal_for_run(run: &Run) -> usize {
    // GML `GenCont/Create_0`: a fresh tutorial run builds the 5-floor
    // `TutCont` arena instead of the area goal.
    if run.tutorial {
        return 5;
    }
    if is_secret_area(run.area) {
        return match run.area {
            AreaId::CrownVault | AreaId::Vault => 40,
            AreaId::PizzaSewers => 70,
            AreaId::City => 130,
            AreaId::Oasis => 130,
            AreaId::HQ => {
                if run.floor_in_area >= 3 {
                    48
                } else {
                    110
                }
            }
            AreaId::CursedCaves => 100,
            AreaId::Jungle => 110,
            _ => 90,
        };
    }

    if run.area == AreaId::Palace {
        let rf = ((run.floor.max(1) - 1) % 15) + 1;
        if rf == 15 {
            return 420;
        }
        return 130;
    }
    if run.area == AreaId::Campfire {
        return 60;
    }
    generation_goal(run.floor)
}

pub fn is_secret_area(area: AreaId) -> bool {
    matches!(
        area,
        AreaId::Oasis
            | AreaId::PizzaSewers
            | AreaId::CursedCaves
            | AreaId::Jungle
            | AreaId::Vault
            | AreaId::CrownVault
            | AreaId::HQ
            | AreaId::City
    )
}

pub fn is_screen_end_wall(wx: i32, wy: i32, floor_set: &HashSet<(i32, i32)>) -> bool {
    let owner = floor_cell_for_wall(wx, wy);
    let neighbors = [
        (owner.0 - 1, owner.1),
        (owner.0 + 1, owner.1),
        (owner.0, owner.1 - 1),
        (owner.0, owner.1 + 1),
    ];
    let floor_n = neighbors.iter().filter(|c| floor_set.contains(c)).count();
    floor_n <= 1
}

#[derive(Clone, Copy)]
struct Maker {
    x: i32,
    y: i32,

    dir: i32,
}

impl Maker {
    /// GML direction law (`scrMakeFloor`: `x += lengthdir_x(32, dir)`,
    /// `y += lengthdir_y(32, dir)`, y-down): 0 = east, 90 = south (+y),
    /// 180 = west, 270 = north (−y). (A y-up `(0,−1)` mapping for 90
    /// would mirror every level north–south.)
    fn step_delta(&self) -> (i32, i32) {
        match self.dir {
            0 => (1, 0),
            90 => (0, 1),
            180 => (-1, 0),
            _ => (0, -1),
        }
    }
}

fn rng_choose<'a, T>(rng: &mut StdRng, items: &'a [T]) -> T
where
    T: Copy,
{
    items[rng.random_range(0..items.len())]
}

fn turn_table(rng: &mut StdRng, area: i32) -> i32 {
    const Z: i32 = 0;
    match area {
        0 => rng_choose(rng, &[Z, Z, 90, -90, 90, -90, 180]),
        // No `case 1` in GML: area 1 falls through to `default`.
        2 | 102 => rng_choose(rng, &[Z, Z, Z, Z, Z, Z, Z, Z, Z, 90, -90, 90, -90, 180]),
        3 => rng_choose(rng, &[Z, Z, Z, Z, Z, 90, -90]),

        4 => rng_choose(rng, &[Z, Z, Z, Z, Z, 90, -90, 180]),

        5 => {
            let tail = rng_choose(rng, &[Z, 90, -90]);
            rng_choose(
                rng,
                &[Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, 180, 180, tail],
            )
        }
        6 => rng_choose(rng, &[Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, 90, -90, 180]),
        7 => rng_choose(rng, &[Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, 90, -90, 180]),
        100 => rng_choose(rng, &[Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, 90, -90, 180, 180]),
        101 => rng_choose(rng, &[Z, Z, Z, Z, 90, -90, 90, -90, 180]),
        103 => rng_choose(rng, &[Z, Z, Z, Z, 90, -90, 180]),
        105 => rng_choose(rng, &[Z, Z, Z, Z, Z, Z, 90, -90, 90, -90, 180]),
        106 => rng_choose(rng, &[Z, Z, 90, -90, 90, -90, 180]),
        _ => rng_choose(rng, &[Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, 90, -90, 90, -90, 180]),
    }
}

// TODO(port): calls `cell_center_px`, `build_walls` and `populate`, whose
// bodies live in world.rs:818/837/874 and are outside this port's scope.
// Call sites below are preserved byte-identical (same RNG call order).
/// GML `GenCont/Step_0` safespawn shift verbatim: once the makers finish,
/// if the spawn ring is already full (`_numfloors >= _maxfloors`) the
/// whole level drifts one `safedir` step and grows a fresh centre floor.
/// GML runs this once per step (repeated over frames); the port settles
/// the same loop inline before walls go up (bounded 16; GML is unbounded
/// per-frame but converges the same way).
/// Skipped exactly where `scrAreaHasSafespawn` is false (campfire, crib,
/// vault, palace/HQ finales; the port has no Crib area so that arm is
/// vacuous).
fn apply_safespawn_shift(plan: &mut LevelPlan, rng: &mut StdRng, run: &Run) {
    let no_safe = matches!(
        run.area,
        AreaId::Campfire | AreaId::Vault | AreaId::CrownVault
    ) || (run.area == AreaId::Palace && ((run.floor.max(1) - 1) % 15) + 1 == 15)
        || (run.area == AreaId::HQ && run.floor_in_area >= 3);
    if no_safe {
        return;
    }
    let safedis: f32 = if run.loop_count > 0 { 96.0 } else { 64.0 };
    let maxfloors = ((safedis / 32.0) * 2.5).ceil() as usize;
    let (dx, dy) = match rng.random_range(0..4) {
        0 => (1, 0),
        1 => (0, 1),
        2 => (-1, 0),
        _ => (0, -1),
    };
    let delta_px = Vec2::new(dx as f32 * TILE, dy as f32 * TILE);
    // GML counts live `Floor` instances (duplicates stack); the port's
    // deduped cells track the surplus in `stacked`. Verbatim exit law:
    // `if (_numfloors < _maxfloors) exit` — thin rings do nothing, full
    // rings shift.
    let mut stacked = 0usize;
    for _ in 0..16 {
        let near = plan
            .floor_cells
            .iter()
            .filter(|(cx, cy)| {
                let (px, py) = cell_center_i(*cx, *cy);
                px * px + py * py <= safedis * safedis
            })
            .count()
            + stacked;
        if near < maxfloors {
            break;
        }
        for (cx, cy) in plan.floor_cells.iter_mut() {
            *cx += dx;
            *cy += dy;
        }
        for chest in plan.chests.iter_mut() {
            let p = match chest {
                ChestSpawn::Weapon(p) | ChestSpawn::Ammo(p) | ChestSpawn::Rad(p) => p,
                ChestSpawn::Custom(_, p) => p,
            };
            *p += delta_px;
        }
        if plan.floor_cells.contains(&(0, 0)) {
            stacked += 1;
        } else {
            plan.floor_cells.push((0, 0));
        }
    }
}
pub fn generate_level(run: &Run) -> LevelPlan {
    let area = gml_area_from_run(run);

    if area == 7 && ((run.floor.max(1) - 1) % 15) + 1 == 15 {
        return generate_palace_last(run);
    }
    if run.area == AreaId::Campfire {
        return generate_campfire(run);
    }
    if area == 106 && run.floor_in_area >= 3 {
        return generate_hq_last(run);
    }
    let goal = generation_goal_for_run(run);
    let mut rng = StdRng::seed_from_u64(run.gen_seed);

    let styleb = rng.random::<f32>() * 6.0 < 1.0;

    let mut plan = LevelPlan {
        floor_cells: Vec::new(),
        wall_cells: HashSet::new(),
        small_walls: Vec::new(),
        bones: Vec::new(),
        bone_sprite: "images/sprBones.png",
        details: Vec::new(),
        props: Vec::new(),
        chests: Vec::new(),
        enemies: Vec::new(),
        boss: None,
        boss_count: 1,
        styleb,
    };

    let mut seen = HashSet::new();
    let stamp_cell = |p: (i32, i32), seen: &mut HashSet<(i32, i32)>, out: &mut Vec<(i32, i32)>| {
        if seen.insert(p) {
            out.push(p);
        }
    };

    let mut makers = vec![Maker {
        x: 0,
        y: 0,
        dir: rng_choose(&mut rng, &[0, 0, 90, 180, 270]),
    }];
    stamp_cell((0, 0), &mut seen, &mut plan.floor_cells);

    let mut guard = 0;
    while !makers.is_empty() && plan.floor_cells.len() <= goal {
        guard += 1;
        if guard > 200_000 {
            break;
        }

        let n_makers = makers.len();
        let mut next_makers = Vec::with_capacity(n_makers);
        let mut new_branches = Vec::new();

        for mi in 0..n_makers {
            let mut m = makers[mi];

            let (dx, dy) = m.step_delta();
            m.x += dx;
            m.y += dy;
            let (mx, my) = (m.x, m.y);

            match area {
                1 => {
                    if rng.random::<f32>() * 2.0 < 1.0 {
                        for p in [(mx, my), (mx + 1, my), (mx + 1, my + 1), (mx, my + 1)] {
                            stamp_cell(p, &mut seen, &mut plan.floor_cells);
                        }
                    } else {
                        stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
                    }
                }
                3 => {
                    let is_max = run.floor_in_area >= 3;
                    if rng.random::<f32>() * 8.0 < 1.0 || is_max {
                        let (xoff, yoff) = if is_max {
                            let xo = rng_choose(&mut rng, &[0, 1, 0, 0, -1]);
                            let yo = rng_choose(&mut rng, &[0, 1, 0, 0, -1]);
                            (xo, yo)
                        } else {
                            (0, 0)
                        };

                        for dy2 in -1..=1 {
                            for dx2 in -1..=1 {
                                stamp_cell(
                                    (mx + xoff + dx2, my + yoff + dy2),
                                    &mut seen,
                                    &mut plan.floor_cells,
                                );
                            }
                        }
                    } else {
                        stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
                    }
                }
                5 => {
                    if rng.random::<f32>() * 11.0 < 1.0 {
                        if rng.random::<f32>() * 2.0 < 1.0 {
                            for p in [
                                (mx + 1, my),
                                (mx + 1, my + 1),
                                (mx, my + 1),
                                (mx, my - 1),
                                (mx - 1, my),
                                (mx + 1, my - 1),
                                (mx - 1, my - 1),
                                (mx - 1, my + 1),
                            ] {
                                stamp_cell(p, &mut seen, &mut plan.floor_cells);
                            }
                        } else {
                            for p in [
                                (mx + 2, my - 2),
                                (mx + 2, my - 1),
                                (mx + 2, my),
                                (mx + 2, my + 1),
                                (mx + 2, my + 2),
                                (mx - 2, my - 2),
                                (mx - 2, my - 1),
                                (mx - 2, my),
                                (mx - 2, my + 1),
                                (mx - 2, my + 2),
                                (mx, my - 2),
                                (mx - 1, my - 2),
                                (mx + 1, my - 2),
                                (mx, my + 2),
                                (mx - 1, my + 2),
                                (mx + 1, my + 2),
                            ] {
                                stamp_cell(p, &mut seen, &mut plan.floor_cells);
                            }
                        }
                        stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
                    } else {
                        stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
                    }
                }
                7 => {
                    if rng.random::<f32>() * 16.0 < 1.0 {
                        for dy2 in -1..=2 {
                            for dx2 in -1..=2 {
                                stamp_cell((mx + dx2, my + dy2), &mut seen, &mut plan.floor_cells);
                            }
                        }
                    } else {
                        for p in [(mx, my), (mx + 1, my), (mx + 1, my + 1), (mx, my + 1)] {
                            stamp_cell(p, &mut seen, &mut plan.floor_cells);
                        }
                    }
                }
                100 => {
                    if rng.random::<f32>() * 8.0 < 1.0 {
                        if rng.random_range(0..3) == 1 {
                            for o in [-2, -1, 0, 1, 2] {
                                stamp_cell((mx + o, my), &mut seen, &mut plan.floor_cells);
                            }
                        } else {
                            for o in [-2, -1, 0, 1, 2] {
                                stamp_cell((mx, my + o), &mut seen, &mut plan.floor_cells);
                            }
                        }
                    } else {
                        stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
                    }
                }
                103 | 107 => {
                    if !plan.floor_cells.is_empty() && plan.floor_cells.len() % 12 == 0 {
                        let (dx2, dy2) = m.step_delta();
                        m.x += dx2;
                        m.y += dy2;
                        let (nx, ny) = (m.x, m.y);
                        // GML ring only: 8 cells, no center.
                        for p in [
                            (nx + 1, ny),
                            (nx + 1, ny + 1),
                            (nx, ny + 1),
                            (nx, ny - 1),
                            (nx - 1, ny),
                            (nx + 1, ny - 1),
                            (nx - 1, ny - 1),
                            (nx - 1, ny + 1),
                        ] {
                            stamp_cell(p, &mut seen, &mut plan.floor_cells);
                        }
                    } else {
                        stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
                    }
                }
                106 => {
                    if !plan.floor_cells.is_empty() && plan.floor_cells.len() % 8 == 0 {
                        let (dx2, dy2) = m.step_delta();
                        m.x += dx2 * 2;
                        m.y += dy2 * 2;
                        let (nx, ny) = (m.x, m.y);
                        // GML 16-cell cross verbatim (5+5 columns, 3+3 rows).
                        for p in [
                            (nx - 2, ny - 2),
                            (nx - 2, ny - 1),
                            (nx - 2, ny),
                            (nx - 2, ny + 1),
                            (nx - 2, ny + 2),
                            (nx + 2, ny - 2),
                            (nx + 2, ny - 1),
                            (nx + 2, ny),
                            (nx + 2, ny + 1),
                            (nx + 2, ny + 2),
                            (nx - 1, ny + 2),
                            (nx, ny + 2),
                            (nx + 1, ny + 2),
                            (nx - 1, ny - 2),
                            (nx, ny - 2),
                            (nx + 1, ny - 2),
                        ] {
                            stamp_cell(p, &mut seen, &mut plan.floor_cells);
                        }
                        m.x += dx2 * 2;
                        m.y += dy2 * 2;
                    } else if rng.random::<f32>() * 3.0 < 1.0 {
                        for p in [
                            (mx, my),
                            (mx + 1, my),
                            (mx + 1, my + 1),
                            (mx, my + 1),
                            (mx, my - 1),
                            (mx - 1, my),
                            (mx + 1, my - 1),
                            (mx - 1, my - 1),
                            (mx - 1, my + 1),
                        ] {
                            stamp_cell(p, &mut seen, &mut plan.floor_cells);
                        }
                    } else {
                        for _ in 0..4 {
                            stamp_cell((m.x, m.y), &mut seen, &mut plan.floor_cells);
                            let (dx2, dy2) = m.step_delta();
                            m.x += dx2;
                            m.y += dy2;
                            stamp_cell((m.x, m.y), &mut seen, &mut plan.floor_cells);
                            plan.chests.push(ChestSpawn::Ammo(cell_center_px(m.x, m.y)));
                        }
                        if rng.random::<f32>() * 3.0 < 1.0 {
                            for p in [
                                (mx + 1, my),
                                (mx + 1, my + 1),
                                (mx, my + 1),
                                (mx, my - 1),
                                (mx - 1, my),
                                (mx + 1, my - 1),
                                (mx - 1, my - 1),
                                (mx - 1, my + 1),
                            ] {
                                stamp_cell(p, &mut seen, &mut plan.floor_cells);
                            }
                        }
                    }
                }
                101 => {
                    stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
                    if rng.random::<f32>() * 3.0 < 1.0 {
                        for p in [(mx - 1, my), (mx + 1, my), (mx, my - 1), (mx, my + 1)] {
                            stamp_cell(p, &mut seen, &mut plan.floor_cells);
                        }
                    }
                }
                104 => {
                    if plan.floor_cells.len() < 4 {
                        for p in [
                            (mx - 1, my),
                            (mx - 1, my - 1),
                            (mx - 1, my + 1),
                            (mx + 1, my),
                            (mx + 1, my - 1),
                            (mx + 1, my + 1),
                            (mx, my + 1),
                            (mx, my - 1),
                        ] {
                            stamp_cell(p, &mut seen, &mut plan.floor_cells);
                        }
                    }
                    m.x += rng_choose(&mut rng, &[0, 2, -2]);
                    m.y += rng_choose(&mut rng, &[0, 2, -2]);
                    let (nx, ny) = (m.x, m.y);
                    for p in [
                        (nx - 1, ny),
                        (nx - 1, ny - 1),
                        (nx - 1, ny + 1),
                        (nx + 1, ny),
                        (nx + 1, ny - 1),
                        (nx + 1, ny + 1),
                        (nx, ny + 1),
                        (nx, ny - 1),
                    ] {
                        stamp_cell(p, &mut seen, &mut plan.floor_cells);
                    }
                    stamp_cell((nx, ny), &mut seen, &mut plan.floor_cells);
                }
                105 => {
                    if rng.random::<f32>() * 4.0 < 1.0 {
                        for p in [(mx, my), (mx + 1, my), (mx + 1, my + 1), (mx, my + 1)] {
                            stamp_cell(p, &mut seen, &mut plan.floor_cells);
                        }
                    } else {
                        stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
                    }
                }
                _ => {
                    stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
                }
            }

            let trn = turn_table(&mut rng, area);
            m.dir = (m.dir + trn).rem_euclid(360);

            if area == 6 && trn.abs() == 90 && rng.random::<f32>() * 2.0 < 1.0 {
                for p in [
                    (mx + 1, my),
                    (mx + 1, my + 1),
                    (mx, my + 1),
                    (mx, my - 1),
                    (mx - 1, my),
                    (mx + 1, my - 1),
                    (mx - 1, my - 1),
                    (mx - 1, my + 1),
                ] {
                    stamp_cell(p, &mut seen, &mut plan.floor_cells);
                }
            }

            let dist_from_spawn = ((mx * 32).pow(2) + (my * 32).pow(2)) as f32;
            if dist_from_spawn > 48.0 * 48.0
                && (trn == 180 || (trn.abs() == 90 && (area == 3 || area == 104)))
            {
                // GML stamps the turn Floor unconditionally, then the
                // weapon chest everywhere except areas 107/0.
                stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
                if area != 107 && area != 0 {
                    plan.chests.push(ChestSpawn::Weapon(cell_center_px(mx, my)));
                }
            }

            let n = (next_makers.len() + new_branches.len() + (n_makers - mi)) as f32;
            let mut dies = match area {
                0 => rng.random::<f32>() * (19.0 + n) > 22.0,
                // `101 | 105` below overlaps an earlier arm in the bevy
                // source too (world.rs:645) — kept byte-identical.
                #[allow(unreachable_patterns)]
                1 | 101 | 105 => rng.random::<f32>() * (19.0 + n) > 20.0,
                2 => rng.random::<f32>() * (14.0 + n) > 15.0,
                3 => rng.random::<f32>() * (39.0 + n) > 40.0,
                4 | 104 => {
                    if area == 104 && rng.random::<f32>() * 4.0 >= 1.0 {
                        false
                    } else {
                        rng.random::<f32>() * (9.0 + n) > 10.0
                    }
                }
                5 => rng.random::<f32>() * (14.0 + n) > 15.0,
                6 => rng.random::<f32>() * (21.0 + n) > 22.0,
                7 => rng.random::<f32>() * (8.0 + n) > 9.0,
                102 => rng.random::<f32>() * (9.0 + n) > 10.0,
                103 | 107 => rng.random::<f32>() * (31.0 + n) > 32.0,
                106 => false,
                _ => rng.random::<f32>() * (19.0 + n) > 20.0,
            };

            if area == 7 && !dies {
                dies = rng.random::<f32>() * (9.0 + n) > 10.0;
            }

            if dies && dist_from_spawn > 48.0 * 48.0 {
                // GML area-0: the AmmoChest line is commented out —
                // Floor only.
                if area != 0 {
                    plan.chests.push(ChestSpawn::Ammo(cell_center_px(mx, my)));
                }
                stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
            }

            if area == 106 && dist_from_spawn > 48.0 * 48.0 && rng.random::<f32>() * 10.0 < 1.0 {
                plan.chests.push(ChestSpawn::Ammo(cell_center_px(mx, my)));
                stamp_cell((mx, my), &mut seen, &mut plan.floor_cells);
            }

            if dies {
                continue;
            }

            if area == 106 {
                if plan.floor_cells.len() > makers.len() * 28 {
                    new_branches.push(Maker {
                        x: mx,
                        y: my,
                        dir: m.dir,
                    });
                }
                next_makers.push(m);
                continue;
            }

            let branches = match area {
                0 => rng.random::<f32>() * 4.0 < 1.0,
                1 | 101 => rng.random::<f32>() * 8.0 < 1.0,
                2 => rng.random::<f32>() * 15.0 < 1.0,
                3 => rng.random::<f32>() * 25.0 < 1.0,
                4 | 104 => rng.random::<f32>() * 4.0 < 1.0,
                5 => rng.random::<f32>() * 15.0 < 1.0,
                6 => rng.random::<f32>() * 20.0 < 1.0,
                7 => rng.random::<f32>() * 16.0 < 1.0,
                102 => rng.random::<f32>() * 5.0 < 1.0,
                103 | 107 => rng.random::<f32>() * 20.0 < 1.0,
                // Unreachable in bevy too (`1 | 101` above covers 101) —
                // kept byte-identical.
                #[allow(unreachable_patterns)]
                101 | 105 => rng.random::<f32>() * 14.0 < 1.0,
                _ => false,
            };

            let branches = branches || (area == 7 && rng.random::<f32>() * 5.0 < 1.0);
            if branches {
                new_branches.push(Maker {
                    x: mx,
                    y: my,
                    dir: m.dir,
                });
            }

            next_makers.push(m);
        }

        makers = next_makers;
        makers.extend(new_branches);
    }

    apply_safespawn_shift(&mut plan, &mut rng, run);

    if let Some(&(fx, fy)) = plan
        .floor_cells
        .iter()
        .max_by_key(|c| c.0.abs() + c.1.abs())
    {
        plan.chests.push(ChestSpawn::Rad(cell_center_px(fx, fy)));
    }

    let floors = plan.floor_cells.clone();
    build_walls(run, &floors, &mut plan);
    let walls = plan.wall_cells.clone();
    populate(run, &floors, &walls, &mut plan, &mut rng);
    plan
}

// TODO(port): calls `build_walls` and `populate` (world.rs:837/874, outside
// this port's scope). Body below is otherwise byte-identical to world.rs:680.
fn generate_palace_last(run: &Run) -> LevelPlan {
    let mut rng = StdRng::seed_from_u64(run.gen_seed);
    let _ = &mut rng;
    let mut plan = LevelPlan {
        floor_cells: Vec::new(),
        wall_cells: HashSet::new(),
        small_walls: Vec::new(),
        bones: Vec::new(),
        bone_sprite: "images/sprBones.png",
        details: Vec::new(),
        props: Vec::new(),
        chests: Vec::new(),
        enemies: Vec::new(),
        boss: None,
        boss_count: 1,
        styleb: false,
    };
    let mut seen = HashSet::new();
    for fy in 0..48 {
        let diy = fy as i32 * 32;
        for fx in 0..8 {
            if diy < -43 && (fx == 0 || fx == 7) {
                continue;
            }
            let c = (fx - 4, fy - 24);
            if seen.insert(c) {
                plan.floor_cells.push(c);
            }
        }
    }
    let floors = plan.floor_cells.clone();
    build_walls(run, &floors, &mut plan);
    let walls = plan.wall_cells.clone();
    populate(run, &floors, &walls, &mut plan, &mut rng);
    plan
}

// TODO(port): calls `build_walls` and `populate` (world.rs:837/874, outside
// this port's scope). Body below is otherwise byte-identical to world.rs:717.
fn generate_campfire(run: &Run) -> LevelPlan {
    let mut rng = StdRng::seed_from_u64(run.gen_seed);
    let mut plan = LevelPlan {
        floor_cells: Vec::new(),
        wall_cells: HashSet::new(),
        small_walls: Vec::new(),
        bones: Vec::new(),
        bone_sprite: "images/sprBones.png",
        details: Vec::new(),
        props: Vec::new(),
        chests: Vec::new(),
        enemies: Vec::new(),
        boss: None,
        boss_count: 1,
        styleb: false,
    };
    let mut seen = HashSet::new();
    for xx in -2..=2 {
        for yy in -1..=1 {
            let c = (xx, yy);
            if seen.insert(c) {
                plan.floor_cells.push(c);
            }
        }
    }

    for _ in 0..40 {
        let idx = rng.random_range(0..plan.floor_cells.len());
        let (cx, cy) = plan.floor_cells[idx];
        let dir = rng.random_range(0..4);
        let (nx, ny) = match dir {
            0 => (cx + 1, cy),
            1 => (cx - 1, cy),
            2 => (cx, cy + 1),
            _ => (cx, cy - 1),
        };
        if seen.insert((nx, ny)) {
            plan.floor_cells.push((nx, ny));
        }
        if plan.floor_cells.len() >= 60 {
            break;
        }
    }
    let floors = plan.floor_cells.clone();
    build_walls(run, &floors, &mut plan);
    let walls = plan.wall_cells.clone();
    populate(run, &floors, &walls, &mut plan, &mut rng);
    plan
}

// TODO(port): calls `build_walls` and `populate` (world.rs:837/874, outside
// this port's scope). Body below is otherwise byte-identical to world.rs:767.
fn generate_hq_last(run: &Run) -> LevelPlan {
    let mut rng = StdRng::seed_from_u64(run.gen_seed);
    let _ = &mut rng;
    let mut plan = LevelPlan {
        floor_cells: Vec::new(),
        wall_cells: HashSet::new(),
        small_walls: Vec::new(),
        bones: Vec::new(),
        bone_sprite: "images/sprBones.png",
        details: Vec::new(),
        props: Vec::new(),
        chests: Vec::new(),
        enemies: Vec::new(),
        boss: None,
        boss_count: 1,
        styleb: true,
    };
    let mut seen = HashSet::new();
    for fx in 0..10 {
        for fy in 0..10 {
            let c = (fx - 5, fy - 5);
            if seen.insert(c) {
                plan.floor_cells.push(c);
            }
        }
    }

    for fx in 0..8 {
        for (sx, sy) in [(fx - 4, 6), (fx - 4, -7)] {
            if seen.insert((sx, sy)) {
                plan.floor_cells.push((sx, sy));
            }
            if seen.insert((sx, sy + 1)) {
                plan.floor_cells.push((sx, sy + 1));
            }
        }
    }
    for fy in 0..4 {
        for (sx, sy) in [(6, fy - 2), (-7, fy - 2)] {
            if seen.insert((sx, sy)) {
                plan.floor_cells.push((sx, sy));
            }
        }
    }
    let floors = plan.floor_cells.clone();
    build_walls(run, &floors, &mut plan);
    let walls = plan.wall_cells.clone();
    populate(run, &floors, &walls, &mut plan, &mut rng);
    plan
}

// world.rs:1904-1927, verbatim (Open Mind mutation: two bonus chests,
// skipped for chest-less areas). Seeded: GML draws from the Generation
// stream, so callers pass the run seed.
pub fn apply_open_mind_bonus(
    plan: &mut LevelPlan,
    area: AreaId,
    floor_in_area: u32,
    seed: u64,
) {
    let no_chests = matches!(
        area,
        AreaId::Campfire | AreaId::Vault | AreaId::CrownVault
    ) || (area == AreaId::HQ && floor_in_area >= 3);
    if no_chests || plan.floor_cells.is_empty() {
        return;
    }
    use rand::{RngExt, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x0BAD_C0DE);
    for _ in 0..2 {
        let idx = rng.random_range(0..plan.floor_cells.len());
        let (cx, cy) = plan.floor_cells[idx];
        let pos = cell_center_px(cx, cy);

        match rng.random_range(0..3) {
            0 => plan.chests.push(ChestSpawn::Weapon(pos)),
            1 => plan.chests.push(ChestSpawn::Ammo(pos)),
            _ => plan.chests.push(ChestSpawn::Rad(pos)),
        }
    }
}

/// GML `scrPopChests` input: everything the permutation pass reads.
/// `seed` threads the Generation RNG stream (GML `random`/`irandom`
/// inside `scrPopChests` draw from the level-generation stream, so equal
/// `gen_seed`s permute identically). Callers pass the run's `gen_seed`
/// mixed with a fixed salt (floor number lives in the plan already via
/// `area`/`subarea`; the salt keeps first-floor and portal permutations
/// on disjoint substreams).
pub struct ChestPermuteCtx {
    pub area: AreaId,
    pub loops: u32,
    pub subarea: u32,
    pub rogue_in_run: bool,
    pub player_half_health: bool,
    pub crown_life: bool,
    pub crown_love: bool,
    pub open_mind: bool,
    pub noradch: u32,
    pub nochest: u32,
    pub same_weapons_for: u32,
    pub horror_done: bool,
    pub hardmode: bool,
    pub player_pos: Vec2,
    pub seed: u64,
}

/// GML `scrPopChests` output: whether a `HostileHorror` hatched (the
/// caller records `Run.horror`). Mimics/horrors ride `plan.enemies`;
/// special chests ride `plan.chests` as `ChestSpawn::Custom`.
pub struct ChestPermuteOut {
    pub horror: bool,
}

/// Verbatim `scripts/scrPopChests/scrPopChests.gml`: vault proto-chest +
/// chestless areas, Open-Mind bonus counts, trim to 1 + bonus per base
/// kind (GML destroys nearest-to-`10016+orandom(250)`; the port shuffles
/// with the seeded stream then truncates — same count law, stable order),
/// rad permutations (Rogue / noradch horror-or-big / half-health /
/// desert styleb maggot), crown Life/Love conversions, mimic rolls, and
/// the hardmode desert 1-1 `BigWeaponChest` arm.
pub fn apply_chest_permutations(
    plan: &mut LevelPlan,
    ctx: ChestPermuteCtx,
) -> ChestPermuteOut {
    use rand::{RngExt, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(ctx.seed);
    let mut out = ChestPermuteOut { horror: false };

    // Snapshot base-kind positions (customs from earlier passes ride
    // along untouched).
    let mut weapons: Vec<Vec2> = Vec::new();
    let mut ammos: Vec<Vec2> = Vec::new();
    let mut rads: Vec<Vec2> = Vec::new();
    let mut customs: Vec<(ChestKind, Vec2)> = Vec::new();
    for c in plan.chests.iter().copied() {
        match c {
            ChestSpawn::Weapon(p) => weapons.push(p),
            ChestSpawn::Ammo(p) => ammos.push(p),
            ChestSpawn::Rad(p) => rads.push(p),
            ChestSpawn::Custom(k, p) => customs.push((k, p)),
        }
    }

    // Vault: furthest weapon chest becomes the ProtoChest; the three
    // base kinds are otherwise destroyed (`_tot = 0`). The ProtoStatue
    // guardian (+4 Bandits, GML `Create_0`) watches the proto chest.
    if ctx.area == AreaId::Vault || ctx.area == AreaId::CrownVault {
        plan.chests.clear();
        for (k, p) in customs {
            plan.chests.push(ChestSpawn::Custom(k, p));
        }
        if let Some(p) = weapons
            .iter()
            .max_by(|a, b| {
                a.length_squared()
                    .partial_cmp(&b.length_squared())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
        {
            plan.chests.push(ChestSpawn::Custom(ChestKind::Proto, p));
            plan.enemies.push((EnemyKind::ProtoStatue, p + Vec2::new(0.0, 64.0)));
            for i in 0..4 {
                let a = i as f32 * std::f32::consts::TAU / 4.0;
                plan.enemies.push((
                    EnemyKind::Bandit,
                    p + Vec2::new(a.cos(), a.sin()) * 48.0,
                ));
            }
        }
        return out;
    }

    // Campfire and the HQ finale hold no chests.
    if ctx.area == AreaId::Campfire || (ctx.area == AreaId::HQ && ctx.subarea >= 3) {
        plan.chests.clear();
        for (k, p) in customs {
            plan.chests.push(ChestSpawn::Custom(k, p));
        }
        return out;
    }

    // Open-Mind bonus counts, then trim each base kind to 1 + bonus
    // (GML destroys nearest-to-random-point; uniform shuffle here).
    let (mut wb, mut ab, mut rb) = (0usize, 0usize, 0usize);
    if ctx.open_mind {
        for _ in 0..2 {
            match rng.random_range(0..3) {
                0 => wb += 1,
                1 => ab += 1,
                _ => rb += 1,
            }
        }
    }
    let mut trim = |v: &mut Vec<Vec2>, keep: usize| {
        // Fisher-Yates shuffle, then truncate.
        for i in (1..v.len()).rev() {
            let j = rng.random_range(0..=i);
            v.swap(i, j);
        }
        v.truncate(keep);
    };
    trim(&mut weapons, 1 + wb);
    trim(&mut ammos, 1 + ab);
    trim(&mut rads, 1 + rb);

    // GML `scrReplacePropWithChest`: a missing base kind converts the
    // furthest eligible prop past 160 px (statues/decals excluded).
    let top_up = |v: &mut Vec<Vec2>, keep: usize| {
        if keep == 0 || !v.is_empty() {
            return;
        }
        let mut best: Option<(Vec2, f32)> = None;
        for (kind, at) in &plan.props {
            match kind {
                PropKind::ThroneStatue | PropKind::YVStatue | PropKind::GroundDecal => continue,
                _ => {}
            }
            let d = at.length_squared();
            if d > 160.0 * 160.0 && best.is_none_or(|(_, bd)| d > bd) {
                best = Some((*at, d));
            }
        }
        if let Some((at, _)) = best {
            v.push(at);
        }
    };
    top_up(&mut weapons, 1 + wb);
    top_up(&mut ammos, 1 + ab);
    top_up(&mut rads, 1 + rb);

    // Rad permutations, in GML order.
    let mut rad_out: Vec<ChestSpawn> = Vec::new();
    for p in rads {
        if ctx.rogue_in_run {
            rad_out.push(ChestSpawn::Custom(ChestKind::Rogue, p));
            continue;
        }
        if ctx.noradch > 0 {
            if ctx.noradch >= 2 && !ctx.horror_done && !out.horror {
                plan.enemies.push((EnemyKind::HostileHorror, p));
                out.horror = true;
            } else {
                rad_out.push(ChestSpawn::Custom(ChestKind::RadBig, p));
            }
            continue;
        }
        if ctx.player_half_health && rng.random_range(0.0..2.0) < 1.0 {
            rad_out.push(ChestSpawn::Custom(ChestKind::Health, p));
            continue;
        }
        if plan.styleb && ctx.area == AreaId::Desert && rng.random_range(0.0..3.0) < 1.0 {
            rad_out.push(ChestSpawn::Custom(ChestKind::RadMaggot, p));
        } else {
            rad_out.push(ChestSpawn::Rad(p));
        }
    }

    // Base survivors.
    let mut final_chests: Vec<ChestSpawn> =
        weapons.into_iter().map(ChestSpawn::Weapon).collect();
    final_chests.extend(ammos.into_iter().map(ChestSpawn::Ammo));
    final_chests.extend(rad_out);
    for (k, p) in customs {
        final_chests.push(ChestSpawn::Custom(k, p));
    }

    // Crown Life: every Rad becomes Health. Crown Love: every non-Proto,
    // non-Rogue chest plus every Rad becomes Ammo.
    if ctx.crown_life {
        for c in final_chests.iter_mut() {
            if matches!(c, ChestSpawn::Rad(_)) {
                let p = c.pos();
                *c = ChestSpawn::Custom(ChestKind::Health, p);
            }
        }
    }
    if ctx.crown_love {
        for c in final_chests.iter_mut() {
            let p = c.pos();
            match *c {
                ChestSpawn::Custom(ChestKind::Proto, _)
                | ChestSpawn::Custom(ChestKind::Rogue, _) => {}
                _ => *c = ChestSpawn::Custom(ChestKind::Ammo, p),
            }
        }
    }

    // Mimic rolls (GML order; BigWeapon needs nochest and no existing
    // BigWeapon; the crib gate is all-true since random(3) < 3).
    let sewers_gate = (ctx.area != AreaId::Desert
        && ctx.area != AreaId::Campfire)
        || ctx.loops > 0;
    let mut no_big_yet = !final_chests.iter().any(|c| {
        matches!(c, ChestSpawn::Custom(ChestKind::BigWeapon, _))
    });
    let mut swapped: Vec<ChestSpawn> = Vec::with_capacity(final_chests.len());
    for c in final_chests {
        match c {
            ChestSpawn::Ammo(p)
                if sewers_gate && rng.random_range(0.0..11.0) < 1.0 =>
            {
                plan.enemies.push((EnemyKind::Mimic, p));
            }
            ChestSpawn::Weapon(p)
                if rng.random_range(0.0..4.0) < ctx.nochest as f32 && no_big_yet =>
            {
                no_big_yet = false;
                swapped.push(ChestSpawn::Custom(ChestKind::BigWeapon, p));
            }
            ChestSpawn::Weapon(p)
                if ctx.same_weapons_for > 4
                    && rng.random_range(0.0..100.0) < (ctx.same_weapons_for - 4) as f32 =>
            {
                plan.enemies.push((EnemyKind::WepMimic, p));
            }
            ChestSpawn::Custom(ChestKind::Health, p)
                if sewers_gate && rng.random_range(0.0..51.0) < 1.0 =>
            {
                plan.enemies.push((EnemyKind::SuperMimic, p));
            }
            other => swapped.push(other),
        }
    }
    plan.chests = swapped;
    // GML hardmode desert 1-1: every player gets a BigWeaponChest
    // (`hardmode && loops - hardmode <= 0`).
    if ctx.hardmode
        && ctx.loops as i32 - i32::from(ctx.hardmode) <= 0
        && ctx.area == AreaId::Desert
        && ctx.subarea == 1
    {
        plan.chests.push(ChestSpawn::Custom(
            ChestKind::BigWeapon,
            ctx.player_pos,
        ));
    }
    out
}

// world.rs:818-835, verbatim (`Vec2` is glam here, already imported;
// `TILE` above is components.rs:47 verbatim).
fn cell_center_px(cx: i32, cy: i32) -> Vec2 {
    Vec2::new(cx as f32 * TILE + TILE * 0.5, cy as f32 * TILE + TILE * 0.5)
}

fn cell_center_i(cx: i32, cy: i32) -> (f32, f32) {
    (cx as f32 * TILE + TILE * 0.5, cy as f32 * TILE + TILE * 0.5)
}

pub fn wall_center(wx: i32, wy: i32) -> Vec2 {
    Vec2::new(
        wx as f32 * WALL_PX + WALL_PX * 0.5,
        wy as f32 * WALL_PX + WALL_PX * 0.5,
    )
}

pub fn wall_top_left(wx: i32, wy: i32) -> Vec2 {
    Vec2::new(wx as f32 * WALL_PX, (wy as f32 + 1.0) * WALL_PX)
}

pub(crate) fn build_walls(_run: &Run, floors: &[(i32, i32)], plan: &mut LevelPlan) {
    let floor_set: std::collections::HashSet<(i32, i32)> = floors.iter().copied().collect();

    for &(cx, cy) in floors {
        let probes = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (2, -1),
            (2, 0),
            (2, 1),
            (-1, 0),
            (-1, 1),
            (-1, 2),
            (0, 2),
            (1, 2),
            (2, 2),
        ];
        for (ox, oy) in probes {
            let wx = cx * 2 + ox;
            let wy = cy * 2 + oy;

            let owner = (wx.div_euclid(2), wy.div_euclid(2));
            if floor_set.contains(&owner) {
                continue;
            }
            plan.wall_cells.insert((wx, wy));
        }
    }
}

// world.rs:869-871, verbatim.
fn side_solid(walls: &std::collections::HashSet<(i32, i32)>, cx: i32, cy: i32, dx: i32) -> bool {
    let wx = cx * 2 + if dx < 0 { -1 } else { 2 };
    walls.contains(&(wx, cy * 2)) && walls.contains(&(wx, cy * 2 + 1))
}

// world.rs:1864-1868, verbatim.
fn walls_cover_tile_with_smalls(plan: &LevelPlan, cx: i32, cy: i32) -> bool {
    plan.small_walls
        .iter()
        .any(|&(wx, wy)| (wx as i32).div_euclid(2) == cx && (wy as i32).div_euclid(2) == cy)
}

// world.rs:127-138, verbatim except `crate::game::secret_areas::is_secret_area`
// becomes the local `is_secret_area` (same port convention as the file header).
// (`is_boss_subarea` itself is ported alongside: `is_boss_subarea_run`'s
// verbatim body calls it, and it is pure over `u32`.)
fn is_boss_subarea(floor: u32) -> bool {
    let rf = ((floor.max(1) - 1) % 15) + 1;

    matches!(rf, 3 | 7 | 11 | 15)
}

fn is_boss_subarea_run(run: &Run) -> bool {
    if is_secret_area(run.area) {
        return false;
    }
    is_boss_subarea(run.floor)
}

// GML `scrAreaGetDifficulty` verbatim: subarea + loops*16 +
// max-subareas of previous areas. Since route floor-1 = previous
// max-subareas + (subarea-1), that collapses to floor + loops*16
// (the old `floor-1` form undercounted by exactly 1 past floor 1).
// Hardmode starts at `hard = 13` (GML `GameCont/Create_0`).
pub fn game_hard(run: &Run) -> f32 {
    let loops = run.loop_count as f32;
    (run.floor as f32 + loops * 16.0 + if run.hardmode { 13.0 } else { 0.0 }).max(1.0)
}

// world.rs:874-1695, verbatim except `crate::game::areas::AreaId::X` becomes
// `AreaId::X` (top import), `crate::game::secret_areas::is_secret_area`
// becomes the local `is_secret_area`, and the inner
// `use crate::game::areas::AreaId;` is dropped (covered by the top import).
// Every branch, table and `rng` call is preserved in order, including the
// duplicated `sy`/`wx` probe lines in the small-walls loop (the second
// `rng.random_range` draw is load-bearing for seeded determinism).
fn populate(
    run: &Run,
    floors: &[(i32, i32)],
    walls: &std::collections::HashSet<(i32, i32)>,
    plan: &mut LevelPlan,
    mut rng: &mut StdRng,
) {
    let area = gml_area_from_run(run);
    let boss_sub = is_boss_subarea_run(run);

    let hard = game_hard(run);
    let enemy_min = (3.0 + hard / 1.5).floor().max(3.0) as usize;

    let mut prop_tiles: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();

    let small_walls_allowed = !boss_sub
        && !matches!(
            run.area,
            AreaId::HQ | AreaId::Vault | AreaId::CrownVault | AreaId::Labs | AreaId::Campfire
        )
        && !(((run.floor.max(1) - 1) % 15) + 1 == 15 && run.area == AreaId::Palace);
    for &(cx, cy) in floors {
        let (px, py) = cell_center_i(cx, cy);
        let dist_sq = px * px + py * py;

        if small_walls_allowed && rng.random::<f32>() * 5.0 < 1.0 && dist_sq > 100.0 * 100.0 {
            let sx = px + rng.random_range(-8.0..8.0);
            // Dead draws, kept verbatim: the bevy source draws (and
            // discards) here, and the RNG sequence must not shift.
            let _sy = py + rng.random_range(-8.0..8.0);
            let _wx = (sx / WALL_PX).floor() as i32;
            let sy = py + rng.random_range(-8.0..8.0);
            let wx = (sx / WALL_PX).floor() as i32;
            let wy = (sy / WALL_PX).floor() as i32;
            plan.small_walls.push((wx as i16, wy as i16));
            prop_tiles.insert((cx, cy));
        }
    }

    let (bone_sprite, bone_chance, bone_lower_only) = match run.area {
        AreaId::Desert => ("images/sprBones.png", 1.0, false),
        AreaId::Campfire => ("images/sprNightBones.png", 1.0, false),
        AreaId::Scrapyards => ("images/sprScrapDecal.png", 1.0 / 7.0, false),
        AreaId::City | AreaId::FrozenCity => ("images/sprIceDecal.png", 1.0 / 7.0, false),
        AreaId::CrystalCaves => ("images/sprCaveDecal.png", 1.0 / 9.0, false),
        AreaId::CursedCaves => ("images/sprInvCaveDecal.png", 1.0 / 9.0, false),
        AreaId::Oasis => ("images/sprCoral.png", 1.0 / 9.0, false),
        AreaId::Sewers => ("images/sprSewerDecal.png", 1.0 / 10.0, true),
        AreaId::PizzaSewers => ("images/sprPizzaSewerDecal.png", 1.0 / 10.0, true),
        AreaId::Jungle => ("images/sprJungleDecal.png", 1.0 / 10.0, true),
        _ => ("images/sprBones.png", 0.0, false),
    };
    plan.bone_sprite = bone_sprite;

    for &(cx, cy) in floors {
        let (px, py) = cell_center_i(cx, cy);

        if rng.random::<f32>() * 6.0 < 1.0 {
            plan.details.push(Vec2::new(
                px + rng.random_range(-14.0..14.0),
                py + rng.random_range(-14.0..14.0),
            ));
        }

        if bone_chance > 0.0
            && side_solid(walls, cx, cy, -1)
            && side_solid(walls, cx, cy, 1)
            && !walls_cover_tile_with_smalls(plan, cx, cy)
        {
            if bone_lower_only {
                if rng.random::<f32>() < bone_chance {
                    plan.bones.push((Vec2::new(px - 16.0, py), false));
                    plan.bones.push((Vec2::new(px + 16.0, py), true));
                }
            } else {
                let spots = [
                    (Vec2::new(px - 16.0, py - 16.0), false),
                    (Vec2::new(px - 16.0, py), false),
                    (Vec2::new(px + 16.0, py - 16.0), true),
                    (Vec2::new(px + 16.0, py), true),
                ];
                for (pos, flip) in spots {
                    if bone_chance >= 1.0 || rng.random::<f32>() < bone_chance {
                        plan.bones.push((pos, flip));
                    }
                }
            }
        }
    }

    for &(cx, cy) in floors {
        if prop_tiles.contains(&(cx, cy)) {
            continue;
        }
        let (px, py) = cell_center_i(cx, cy);
        let dist_sq = px * px + py * py;

        let unlikeliness = if run.area == AreaId::Jungle {
            2.0
        } else if run.area == AreaId::Campfire {
            7.0
        } else {
            10.0
        };

        let is_secret = is_secret_area(run.area);
        let kind = if is_secret {
            match run.area {
                AreaId::Oasis => {
                    let r: f32 = rng.random();
                    if r < 0.025 {
                        PropKind::Anchor
                    } else if r < 0.35 {
                        PropKind::WaterPlant
                    } else if r < 0.50 {
                        PropKind::OasisBarrel
                    } else if r < 0.62 {
                        PropKind::WaterMine
                    } else {
                        PropKind::GroundDecal
                    }
                }
                AreaId::PizzaSewers => {
                    if rng.random::<f32>() < 0.7 {
                        PropKind::PizzaBox
                    } else {
                        PropKind::GroundDecal
                    }
                }
                AreaId::Jungle => {
                    if rng.random::<f32>() * 30.0 < 1.0 {
                        PropKind::BigFlower
                    } else if rng.random::<f32>() < 0.55 {
                        PropKind::Bush
                    } else {
                        PropKind::GroundDecal
                    }
                }
                AreaId::CursedCaves => PropKind::GroundDecal,
                AreaId::City => {
                    let r = rng.random_range(0..10);
                    match r {
                        0..=3 => PropKind::MoneyPile,
                        4 => PropKind::YVStatue,
                        5 => PropKind::GoldBarrel,
                        _ => PropKind::GroundDecal,
                    }
                }
                AreaId::Vault | AreaId::CrownVault => PropKind::Torch,
                AreaId::HQ => {
                    if rng.random::<f32>() < 0.5 {
                        PropKind::PlantPot
                    } else {
                        PropKind::GroundDecal
                    }
                }
                _ => PropKind::GroundDecal,
            }
        } else {
            match area {
                1 => {
                    if rng.random::<f32>() * 60.0 < 1.0 {
                        PropKind::BigSkull
                    } else if plan.styleb && rng.random::<f32>() * 5.0 < 1.0 {
                        PropKind::BonePile
                    } else if rng.random::<f32>() * 4.0 < 3.0 {
                        if plan.styleb {
                            PropKind::NightCactus
                        } else {
                            PropKind::Cactus
                        }
                    } else {
                        PropKind::GroundDecal
                    }
                }

                2 => {
                    if dist_sq < 96.0 * 96.0 {
                        PropKind::GroundDecal
                    } else {
                        let roll = rng.random_range(0..7);
                        match roll {
                            0..=3 => PropKind::Pipe,
                            4..=5 => PropKind::ToxicBarrel,
                            _ => PropKind::GroundDecal,
                        }
                    }
                }

                3 => {
                    let roll = rng.random_range(0..7);
                    match roll {
                        0..=2 => PropKind::Tires,
                        3..=4 => PropKind::Car,
                        _ => PropKind::GroundDecal,
                    }
                }

                4 => {
                    let r: f32 = rng.random();
                    if r < 0.25 {
                        PropKind::Crystal
                    } else if r < 0.45 {
                        PropKind::Cocoon
                    } else if r < 0.55 {
                        PropKind::BonePile
                    } else if r < 0.75 {
                        PropKind::Cobweb
                    } else {
                        PropKind::GroundDecal
                    }
                }

                5 => {
                    if dist_sq < 32.0 * 32.0 {
                        PropKind::GroundDecal
                    } else {
                        let r: f32 = rng.random();
                        if r < 0.18 {
                            PropKind::IcePatch
                        } else if r < 0.28 {
                            PropKind::Snowman
                        } else if r < 0.36 {
                            PropKind::SodaMachine
                        } else if r < 0.44 {
                            PropKind::StreetLight
                        } else if r < 0.54 {
                            if dist_sq < 128.0 * 128.0 {
                                PropKind::GroundDecal
                            } else {
                                PropKind::Hydrant
                            }
                        } else if r < 0.60 {
                            if dist_sq < 128.0 * 128.0 {
                                PropKind::GroundDecal
                            } else {
                                PropKind::Car
                            }
                        } else {
                            PropKind::GroundDecal
                        }
                    }
                }

                6 => {
                    let r: f32 = rng.random();
                    if r < 0.30 {
                        PropKind::Tube
                    } else if r < 0.38 {
                        PropKind::MutantTube
                    } else if r < 0.50 {
                        PropKind::ToxicBarrel
                    } else if r < 0.60 {
                        PropKind::FireTrap
                    } else if r < 0.65 {
                        PropKind::Mine
                    } else {
                        PropKind::GroundDecal
                    }
                }

                7 => {
                    let r: f32 = rng.random();
                    if r < 0.20 {
                        PropKind::Pillar
                    } else if r < 0.35 {
                        PropKind::SmallGenerator
                    } else if r < 0.42 {
                        PropKind::Torch
                    } else if r < 0.50 {
                        PropKind::FireTrap
                    } else if r < 0.55 {
                        PropKind::Mine
                    } else {
                        PropKind::GroundDecal
                    }
                }

                _ => PropKind::GroundDecal,
            }
        };

        let threshold = match kind {
            PropKind::Cobweb | PropKind::IcePatch => 2.6,
            PropKind::FireTrap => 1.35,
            PropKind::Mine => 0.85,
            _ => 1.0,
        };

        if rng.random::<f32>() * unlikeliness > threshold {
            continue;
        }

        let too_close = match kind {
            PropKind::Anchor => false,
            PropKind::WaterPlant | PropKind::OasisBarrel | PropKind::WaterMine => {
                dist_sq < 96.0 * 96.0
            }
            PropKind::MoneyPile | PropKind::YVStatue | PropKind::GoldBarrel => {
                dist_sq < 64.0 * 64.0
            }
            _ => false,
        };
        if too_close {
            continue;
        }

        let claims_tile = !matches!(
            kind,
            PropKind::GroundDecal | PropKind::Cobweb | PropKind::IcePatch | PropKind::FireTrap
        );

        if claims_tile && dist_sq < 64.0 * 64.0 {
            continue;
        }

        if claims_tile {
            prop_tiles.insert((cx, cy));
        }
        plan.props.push((kind, Vec2::new(px, py)));
    }

    let rf_route = ((run.floor.max(1) - 1) % 15) + 1;
    let skip_enemies = boss_sub && rf_route == 15;

    // GML `scrPopulate` asks for 120 px (150 on the city boss floor) but
    // `scrPopEnemies` itself refuses anything under 160 px, so the
    // effective clear ring is 160 everywhere.
    let spawn_dist = 160.0;
    let mut enemy_tiles: Vec<(EnemyKind, Vec2)> = Vec::new();
    for &(cx, cy) in floors {
        if skip_enemies {
            break;
        }
        let (px, py) = cell_center_i(cx, cy);
        let dist_sq = px * px + py * py;
        if dist_sq < spawn_dist * spawn_dist || prop_tiles.contains(&(cx, cy)) {
            continue;
        }
        if walls_cover_tile(walls, cx, cy)
            || plan
                .small_walls
                .iter()
                .any(|&(wx, wy)| (wx as i32).div_euclid(2) == cx && (wy as i32).div_euclid(2) == cy)
        {
            continue;
        }

        let chance = hard / (10.0 + hard);
        // GML `scrPopulate` fires `scrPopEnemies` twice per floor: the
        // open roll above plus the Blood-crown bonus roll below, which
        // ignores the enemy cap (`random(8 + hard) < hard`).
        let normal = rng.random::<f32>() < chance || enemy_tiles.len() < enemy_min;
        let extra = run.blood_crown && rng.random::<f32>() < hard / (8.0 + hard);

        let center = Vec2::new(px, py);
        let pick_kind = |rng: &mut StdRng, w: &[EnemyKind]| w[rng.random_range(0..w.len())];
        let loop_extras = loop_elite_candidates(area, run.loop_count);

        for pass in 0..2 {
            // Pass 0 rolls the open table; pass 1 re-rolls it for the
            // Blood crown. (`continue` below skips the pass, matching
            // GML's per-`scrPopEnemies`-call return.)
            if (pass == 0 && !normal) || (pass == 1 && !extra) {
                continue;
            }

            {
                let mut secret_kinds: Vec<EnemyKind> = match run.area {
                AreaId::Oasis => {
                    if rng.random::<f32>() * 4.0 < 1.0 {
                        vec![EnemyKind::Crab]
                    } else if rng.random::<f32>() * 3.0 < 1.0 {
                        vec![
                            EnemyKind::BoneFish,
                            EnemyKind::BoneFish,
                            EnemyKind::BoneFish,
                        ]
                    } else {
                        Vec::new()
                    }
                }
                AreaId::PizzaSewers => vec![EnemyKind::Turtle],
                AreaId::Jungle => {
                    if rng.random::<f32>() * 8.0 < 1.0 {
                        vec![EnemyKind::JungleFly]
                    } else if rng.random::<f32>() * 30.0 < 1.0 {
                        plan.props.push((PropKind::Barrel, center));
                        vec![
                            EnemyKind::JungleBandit,
                            EnemyKind::JungleBandit,
                            EnemyKind::JungleBandit,
                        ]
                    } else {
                        let k = pick_kind(
                            &mut rng,
                            &[
                                EnemyKind::JungleBandit,
                                EnemyKind::JungleBandit,
                                EnemyKind::JungleBandit,
                                EnemyKind::JungleBandit,
                                EnemyKind::JungleBandit,
                                EnemyKind::Maggot,
                                EnemyKind::Assassin,
                                EnemyKind::Assassin,
                            ],
                        );
                        vec![k]
                    }
                }
                AreaId::CursedCaves => {
                    if rng.random::<f32>() * 5.0 < 4.0 {
                        let k = pick_kind(
                            &mut rng,
                            &[
                                EnemyKind::InvSpider,
                                EnemyKind::InvSpider,
                                EnemyKind::InvSpider,
                                EnemyKind::InvSpider,
                                EnemyKind::InvLaserCrystal,
                                EnemyKind::InvLaserCrystal,
                            ],
                        );
                        vec![k]
                    } else {
                        Vec::new()
                    }
                }
                AreaId::City => {
                    if rng.random::<f32>() * 5.0 < 1.0 {
                        let k = pick_kind(
                            &mut rng,
                            &[
                                EnemyKind::FireBaller,
                                EnemyKind::Jock,
                                EnemyKind::FireBaller,
                                EnemyKind::Jock,
                                EnemyKind::FireBaller,
                                EnemyKind::SuperFireBaller,
                            ],
                        );
                        vec![k]
                    } else if rng.random::<f32>() * 4.0 < 1.0 {
                        if rng.random::<f32>() * 5.0 < 1.0 {
                            plan.props.push((PropKind::GoldBarrel, center));
                        }
                        let k = pick_kind(
                            &mut rng,
                            &[
                                EnemyKind::Molefish,
                                EnemyKind::Molefish,
                                EnemyKind::Molefish,
                                EnemyKind::Molefish,
                                EnemyKind::Molesarge,
                            ],
                        );
                        vec![k]
                    } else {
                        Vec::new()
                    }
                }
                AreaId::Vault | AreaId::CrownVault => {
                    let k = pick_kind(
                        &mut rng,
                        &[
                            EnemyKind::RobotGuard,
                            EnemyKind::Turret,
                            EnemyKind::IdpdElite,
                            EnemyKind::CrownGuardian,
                            EnemyKind::CrownGuardian,
                        ],
                    );
                    vec![k]
                }
                AreaId::HQ => {
                    if rng.random::<f32>() * 7.0 < 1.0 {
                        let k = pick_kind(
                            &mut rng,
                            &[
                                EnemyKind::IdpdElite,
                                EnemyKind::IdpdShield,
                                EnemyKind::IdpdInspector,
                            ],
                        );
                        vec![k]
                    } else if rng.random::<f32>() * 4.0 < 1.0 {
                        std::iter::repeat_n(EnemyKind::IdpdGrunt, 5).collect()
                    } else if rng.random::<f32>() * 3.0 < 1.0 {
                        let k = pick_kind(
                            &mut rng,
                            &[
                                EnemyKind::IdpdGrunt,
                                EnemyKind::IdpdShield,
                                EnemyKind::IdpdInspector,
                            ],
                        );
                        vec![k]
                    } else {
                        Vec::new()
                    }
                }
                _ => Vec::new(),
            };
            if !secret_kinds.is_empty() {
                for (i, k) in secret_kinds.drain(..).enumerate() {
                    let jitter =
                        Vec2::new(((i % 3) as f32 - 1.0) * 18.0, ((i / 3) as f32 - 0.5) * 18.0);
                    enemy_tiles.push((k, center + jitter));
                }
                continue;
            }
        }

        match area {
            1 => {
                // GML `scrPopEnemies` desert arm verbatim: loop invaders,
                // then the styleb big-maggot nest, then maggot/scorpion,
                // then the barrel ambush, then the bandit default.
                // (`_loop_rand = random(loops)`; 0 loops never fires.)
                let loop_rand = if run.loop_count == 0 {
                    0.0
                } else {
                    rng.random_range(0.0..run.loop_count as f32)
                };
                if rng.random_range(0.0..2.0) < loop_rand {
                    let k = pick_kind(
                        &mut rng,
                        &[
                            EnemyKind::Scorpion,
                            EnemyKind::Scorpion,
                            EnemyKind::Bandit,
                            EnemyKind::Bandit,
                            EnemyKind::Maggot,
                            EnemyKind::JungleFly,
                            EnemyKind::JungleFly,
                            EnemyKind::MeleeBandit,
                            EnemyKind::Sniper,
                        ],
                    );
                    enemy_tiles.push((k, center));
                } else if plan.styleb {
                    let k = pick_kind(
                        &mut rng,
                        &[
                            EnemyKind::MaggotSpawn,
                            EnemyKind::BigMaggot,
                            EnemyKind::BigMaggot,
                            EnemyKind::Maggot,
                        ],
                    );
                    enemy_tiles.push((k, center));
                } else if rng.random::<f32>() * 7.0 < 1.0 {
                    let k = pick_kind(&mut rng, &[EnemyKind::MaggotSpawn, EnemyKind::Scorpion]);
                    enemy_tiles.push((k, center));
                } else if rng.random::<f32>() * 30.0 < 1.0 {
                    plan.props.push((PropKind::Barrel, center));
                    for _ in 0..3 {
                        enemy_tiles.push((
                            EnemyKind::Bandit,
                            center
                                + Vec2::new(
                                    rng.random_range(-2.0..2.0),
                                    rng.random_range(-2.0..2.0),
                                ),
                        ));
                    }
                } else {
                    let mut cands = vec![
                        EnemyKind::Bandit,
                        EnemyKind::Bandit,
                        EnemyKind::Bandit,
                        EnemyKind::Bandit,
                        EnemyKind::Bandit,
                        EnemyKind::Bandit,
                        EnemyKind::Maggot,
                        EnemyKind::Scorpion,
                    ];
                    cands.extend(loop_extras.iter().copied());
                    let k = pick_kind(&mut rng, &cands);
                    enemy_tiles.push((k, center));
                }
            }
            2 => {
                if run.loop_count > 0 && rng.random::<f32>() * 3.0 >= 1.0 {
                    let mut cands = vec![
                        EnemyKind::Ratking,
                        EnemyKind::Ratking,
                        EnemyKind::BuffGator,
                        EnemyKind::LaserCrystal,
                        EnemyKind::Rat,
                        EnemyKind::Ballguy,
                        EnemyKind::Ballguy,
                        EnemyKind::FrogEgg,
                    ];
                    cands.extend(loop_extras.iter().copied());
                    let k = pick_kind(&mut rng, &cands);
                    enemy_tiles.push((k, center));
                } else if rng.random::<f32>() * 9.0 < 1.0 {
                    let k = pick_kind(
                        &mut rng,
                        &[
                            EnemyKind::Ballguy,
                            EnemyKind::Ratking,
                            EnemyKind::MeleeBandit,
                        ],
                    );
                    enemy_tiles.push((k, center));
                } else {
                    let mut cands = vec![
                        EnemyKind::Rat,
                        EnemyKind::Rat,
                        EnemyKind::Rat,
                        EnemyKind::Maggot,
                        EnemyKind::Gator,
                        EnemyKind::Bandit,
                    ];
                    cands.extend(loop_extras.iter().copied());
                    let k = pick_kind(&mut rng, &cands);
                    enemy_tiles.push((k, center));
                }
            }
            3 => {
                let roll: f32 = rng.random();
                let mut cands = if roll * 4.0 < 1.0 {
                    vec![
                        EnemyKind::MeleeBandit,
                        EnemyKind::Sniper,
                        EnemyKind::MeleeBandit,
                        EnemyKind::Sniper,
                        EnemyKind::Ballguy,
                    ]
                } else if roll * 10.0 < 1.0 {
                    vec![
                        EnemyKind::Raven,
                        EnemyKind::Raven,
                        EnemyKind::Raven,
                        EnemyKind::Raven,
                    ]
                } else if roll * 20.0 < 1.0 {
                    vec![EnemyKind::Salamander]
                } else {
                    vec![
                        EnemyKind::Raven,
                        EnemyKind::Raven,
                        EnemyKind::Raven,
                        EnemyKind::Bandit,
                    ]
                };
                cands.extend(loop_extras.iter().copied());
                let k = pick_kind(&mut rng, &cands);
                enemy_tiles.push((k, center));
            }
            4 => {
                let mut cands = if run.loop_count > 0 && rng.random_bool(0.5) {
                    vec![
                        EnemyKind::LaserCrystal,
                        EnemyKind::LaserCrystal,
                        EnemyKind::RhinoFreak,
                        EnemyKind::LightningCrystal,
                        EnemyKind::BuffGator,
                        EnemyKind::ExploFreak,
                        EnemyKind::Spider,
                        EnemyKind::Spider,
                    ]
                } else {
                    vec![
                        EnemyKind::Spider,
                        EnemyKind::Spider,
                        EnemyKind::Spider,
                        EnemyKind::Spider,
                        EnemyKind::LaserCrystal,
                        EnemyKind::LaserCrystal,
                    ]
                };
                cands.extend(loop_extras.iter().copied());
                let k = pick_kind(&mut rng, &cands);
                enemy_tiles.push((k, center));
            }
            5 => {
                let mut frozen = if run.loop_count > 0 && rng.random_bool(0.5) {
                    vec![
                        EnemyKind::RobotGuard,
                        EnemyKind::RobotGuard,
                        EnemyKind::SnowTank,
                        EnemyKind::DogGuardian,
                        EnemyKind::ExploGuardian,
                        EnemyKind::Wolf,
                        EnemyKind::Necromancer,
                    ]
                } else {
                    vec![
                        EnemyKind::RobotGuard,
                        EnemyKind::RobotGuard,
                        EnemyKind::RobotGuard,
                        EnemyKind::SnowTank,
                        EnemyKind::Wolf,
                        EnemyKind::Wolf,
                    ]
                };
                frozen.extend(loop_extras.iter().copied());
                let k = pick_kind(&mut rng, &frozen);
                enemy_tiles.push((k, center));
            }
            6 => {
                let mut late = if run.loop_count > 0 && rng.random_bool(0.5) {
                    vec![
                        EnemyKind::Ratking,
                        EnemyKind::RhinoFreak,
                        EnemyKind::ExploFreak,
                        EnemyKind::Necromancer,
                        EnemyKind::LaserCrystal,
                        EnemyKind::Turret,
                    ]
                } else {
                    vec![
                        EnemyKind::Freak,
                        EnemyKind::Freak,
                        EnemyKind::Freak,
                        EnemyKind::Necromancer,
                        EnemyKind::ExploFreak,
                        EnemyKind::RhinoFreak,
                    ]
                };
                late.extend(std::iter::repeat_n(
                    EnemyKind::IdpdGrunt,
                    (run.loop_count.min(3) * 2) as usize,
                ));
                late.extend(loop_extras.iter().copied());
                let k = pick_kind(&mut rng, &late);
                enemy_tiles.push((k, center));
            }
            7 => {
                let mut palace = if run.loop_count > 0 && rng.random_bool(0.5) {
                    vec![
                        EnemyKind::ExploGuardian,
                        EnemyKind::DogGuardian,
                        EnemyKind::DogGuardian,
                        EnemyKind::Sniper,
                        EnemyKind::ExploFreak,
                        EnemyKind::JungleBandit,
                        EnemyKind::JungleBandit,
                    ]
                } else {
                    vec![
                        EnemyKind::Guardian,
                        EnemyKind::Guardian,
                        EnemyKind::Guardian,
                        EnemyKind::ExploGuardian,
                        EnemyKind::ExploGuardian,
                        EnemyKind::DogGuardian,
                    ]
                };
                palace.extend(std::iter::repeat_n(
                    EnemyKind::IdpdGrunt,
                    (run.loop_count.min(3)) as usize,
                ));
                palace.extend(loop_extras.iter().copied());
                let k = pick_kind(&mut rng, &palace);
                enemy_tiles.push((k, center));
            }
                _ => {}
            }
        }
    }

    if !skip_enemies {
        let mut extras = floors
            .iter()
            .copied()
            .filter(|(cx, cy)| {
                let (px, py) = cell_center_i(*cx, *cy);
                let d = px * px + py * py;
                d >= 160.0 * 160.0 && !prop_tiles.contains(&(*cx, *cy))
            })
            .collect::<Vec<_>>();
        extras.sort_by_key(|&(cx, cy)| {
            let (px, py) = cell_center_i(cx, cy);
            -((px * px + py * py) as i32)
        });
        let mut ei = 0;
        while enemy_tiles.len() < enemy_min && ei < extras.len() {
            let (cx, cy) = extras[ei];
            ei += 1;
            let center = cell_center_px(cx, cy);
            if enemy_tiles.iter().any(|(_, p)| p.distance(center) < 8.0) {
                continue;
            }
            let cands = default_area_enemies(area, run.loop_count);
            if cands.is_empty() {
                break;
            }
            let k = cands[rng.random_range(0..cands.len())];
            enemy_tiles.push((k, center));
        }
    }
    plan.enemies = enemy_tiles;

    if boss_sub {
        let kind = boss_for_floor_and_loop(run.floor, run.loop_count);
        plan.boss = Some(kind);
        if matches!(kind, EnemyKind::BigBandit | EnemyKind::BigBanditLoop) {
            plan.boss_count = big_bandit_count(run.loop_count);
        }
    } else {
        match run.area {
            AreaId::PizzaSewers => plan.boss = Some(EnemyKind::FrogQueen),
            AreaId::Sewers if run.loop_count >= 1 => plan.boss = Some(EnemyKind::Mom),
            AreaId::Labs if run.loop_count >= 1 => plan.boss = Some(EnemyKind::Technomancer),
            AreaId::CrystalCaves if run.loop_count >= 1 => plan.boss = Some(EnemyKind::Hyper),
            AreaId::CrownVault | AreaId::Vault => plan.boss = Some(EnemyKind::OldGuardian),
            AreaId::HQ => plan.boss = Some(EnemyKind::Captain),
            _ => {}
        }
    }

    let rf = ((run.floor.max(1) - 1) % 15) + 1;
    if rf == 15 && !is_secret_area(run.area) {
        populate_throne_room(run, plan);
    }

    let is_no_chest_area = matches!(
        run.area,
        AreaId::Campfire | AreaId::Vault | AreaId::CrownVault
    ) || (run.area == AreaId::HQ && run.floor_in_area >= 3);
    if !is_no_chest_area {
        let has_weapon = plan
            .chests
            .iter()
            .any(|c| matches!(c, ChestSpawn::Weapon(_)));
        let has_ammo = plan.chests.iter().any(|c| matches!(c, ChestSpawn::Ammo(_)));
        if !has_weapon || !has_ammo {
            let mut best: Option<(f32, usize)> = None;
            for (i, (_, p)) in plan.props.iter().enumerate() {
                let d2 = p.length_squared();
                if d2 < 160.0 * 160.0 {
                    continue;
                }
                if best.map(|(bd, _)| d2 > bd).unwrap_or(true) {
                    best = Some((d2, i));
                }
            }
            if let Some((_, idx)) = best {
                let pos = plan.props[idx].1;
                plan.props.remove(idx);
                if !has_weapon {
                    plan.chests.push(ChestSpawn::Weapon(pos));
                } else {
                    plan.chests.push(ChestSpawn::Ammo(pos));
                }
            }
        }
    }

    trim_chests(&mut plan.chests);

    // GML `GenCont/Alarm_0` tutorial arm verbatim: the `TutCont` level
    // holds no roaming enemies, no chests and no boss wants (its weapon
    // chest and portal are scripted later by `TutCont` itself, which the
    // port does not model yet).
    if run.tutorial {
        plan.enemies.clear();
        plan.chests.clear();
        plan.boss = None;
    }
}

pub fn boss_for_floor_and_loop(floor: u32, loop_count: u32) -> EnemyKind {
    let rf = ((floor.max(1) - 1) % 15) + 1;
    match rf {
        3 if loop_count > 0 => EnemyKind::BigBanditLoop,
        3 => EnemyKind::BigBandit,
        7 if loop_count > 0 => EnemyKind::BigDogLoop,
        7 => EnemyKind::BigDog,
        11 if loop_count > 0 => EnemyKind::LilHunterLoop,
        11 => EnemyKind::LilHunter,
        15 => EnemyKind::Throne,
        _ => EnemyKind::BigBandit,
    }
}

#[allow(dead_code)]
pub fn boss_for_floor(floor: u32) -> EnemyKind {
    boss_for_floor_and_loop(floor, (floor.max(1) - 1) / 15)
}

pub fn floor_in_world(floor: u32) -> u32 {
    let rf = ((floor.max(1) - 1) % 15) + 1;
    match rf {
        1..=3 => rf,
        4 => 1,
        5..=7 => rf - 4,
        8 => 1,
        9..=11 => rf - 8,
        12 => 1,
        13..=15 => rf - 12,
        _ => 1,
    }
}

pub fn world_of(floor: u32) -> u32 {
    let rf = ((floor.max(1) - 1) % 15) + 1;
    match rf {
        1..=3 => 1,
        4 => 2,
        5..=7 => 3,
        8 => 4,
        9..=11 => 5,
        12 => 6,
        13..=15 => 7,
        _ => 1,
    }
}

pub fn floor_cell_for_wall(wx: i32, wy: i32) -> (i32, i32) {
    (wx.div_euclid(2), wy.div_euclid(2))
}

pub fn wall_cell_at(pos: Vec2) -> (i32, i32) {
    (
        (pos.x / WALL_PX).floor() as i32,
        (pos.y / WALL_PX).floor() as i32,
    )
}

fn apply_loop_elite_substitutions(
    table: &mut Vec<(EnemyKind, usize)>,
    area: AreaId,
    loop_count: u32,
) {
    if loop_count == 0 {
        return;
    }

    let l = loop_count.min(4) as usize;

    match area {
        AreaId::Desert => {
            table.push((EnemyKind::Scorpion, 4 + l * 2));
            table.push((EnemyKind::JungleFly, 3 + l));
            table.push((EnemyKind::MeleeBandit, 3 + l));
            table.push((EnemyKind::Sniper, 3 + l));
            if loop_count >= 2 {
                table.push((EnemyKind::GoldScorpion, 2 + l));
            }
        }
        AreaId::Sewers => {
            table.push((EnemyKind::Ratking, 4 + l * 2));
            table.push((EnemyKind::BuffGator, 3 + l));
            table.push((EnemyKind::IdpdShield, 2 + l));
        }
        AreaId::Scrapyards => {
            table.push((EnemyKind::Sniper, 4 + l * 2));
            table.push((EnemyKind::MeleeBandit, 3 + l));
            table.push((EnemyKind::Salamander, 2 + l));
            if loop_count >= 2 {
                table.push((EnemyKind::BuffGator, 3 + l));
                table.push((EnemyKind::RobotGuard, 2 + l));
            }
        }
        AreaId::CrystalCaves => {
            table.push((EnemyKind::RhinoFreak, 4 + l * 2));
            table.push((EnemyKind::ExploFreak, 3 + l));
            table.push((EnemyKind::LightningCrystal, 3 + l));
            if loop_count >= 2 {
                table.push((EnemyKind::IdpdElite, 3 + l));
            }
        }
        AreaId::FrozenCity => {
            table.push((EnemyKind::SnowTank, 4 + l));
            table.push((EnemyKind::DogGuardian, 3 + l));
            table.push((EnemyKind::ExploGuardian, 3 + l));
            table.push((EnemyKind::Necromancer, 2 + l));
            if loop_count >= 2 {
                table.push((EnemyKind::GoldSnowtank, 2 + l));
            }
        }
        AreaId::Labs => {
            table.push((EnemyKind::Ratking, 4 + l));
            table.push((EnemyKind::RhinoFreak, 4 + l * 2));
            table.push((EnemyKind::ExploFreak, 4 + l));
            if loop_count >= 2 {
                table.push((EnemyKind::IdpdElite, 5 + l));
            }
        }
        AreaId::Palace => {
            table.push((EnemyKind::Sniper, 3 + l));
            table.push((EnemyKind::ExploFreak, 3 + l));
            table.push((EnemyKind::JungleBandit, 3 + l * 2));
            if loop_count >= 1 {
                table.push((EnemyKind::PopoFreak, 2 + l));
                table.push((EnemyKind::HostileHorror, 1 + l));
            }
            if loop_count >= 2 {
                table.push((EnemyKind::IdpdElite, 5 + l));
            }
        }
        _ => {}
    }
}

fn loop_elite_candidates(area_num: i32, loop_count: u32) -> Vec<EnemyKind> {
    let area = match area_num {
        1 => AreaId::Desert,
        2 => AreaId::Sewers,
        3 => AreaId::Scrapyards,
        4 => AreaId::CrystalCaves,
        5 => AreaId::FrozenCity,
        6 => AreaId::Labs,
        7 => AreaId::Palace,
        _ => return Vec::new(),
    };

    let mut table = Vec::new();
    apply_loop_elite_substitutions(&mut table, area, loop_count);

    let mut out = Vec::new();
    for (kind, weight) in table {
        for _ in 0..weight.min(8) {
            out.push(kind);
        }
    }
    out
}

fn default_area_enemies(area: i32, loop_count: u32) -> Vec<EnemyKind> {
    let mut c = match area {
        1 => vec![
            EnemyKind::Bandit,
            EnemyKind::Bandit,
            EnemyKind::Bandit,
            EnemyKind::Maggot,
            EnemyKind::Scorpion,
        ],
        2 => vec![
            EnemyKind::Rat,
            EnemyKind::Rat,
            EnemyKind::Maggot,
            EnemyKind::Gator,
            EnemyKind::MeleeFake,
        ],
        3 => vec![
            EnemyKind::Raven,
            EnemyKind::Raven,
            EnemyKind::Raven,
            EnemyKind::Sniper,
            EnemyKind::Ballguy,
        ],
        4 => vec![
            EnemyKind::Spider,
            EnemyKind::Spider,
            EnemyKind::LaserCrystal,
            EnemyKind::LaserCrystal,
            EnemyKind::RadMaggot,
        ],
        5 => vec![
            EnemyKind::RobotGuard,
            EnemyKind::RobotGuard,
            EnemyKind::SnowTank,
            EnemyKind::Wolf,
            EnemyKind::Wolf,
        ],
        6 => vec![
            EnemyKind::Freak,
            EnemyKind::Freak,
            EnemyKind::Necromancer,
            EnemyKind::ExploFreak,
            EnemyKind::RhinoFreak,
        ],
        7 => vec![
            EnemyKind::Guardian,
            EnemyKind::ExploGuardian,
            EnemyKind::DogGuardian,
        ],
        _ => vec![
            EnemyKind::RobotGuard,
            EnemyKind::Assassin,
            EnemyKind::Freak,
            EnemyKind::Turret,
            EnemyKind::SuperFrog,
        ],
    };
    c.extend(loop_elite_candidates(area, loop_count));
    c
}

fn walls_cover_tile(walls: &std::collections::HashSet<(i32, i32)>, cx: i32, cy: i32) -> bool {
    for ox in 0..2 {
        for oy in 0..2 {
            if walls.contains(&(cx * 2 + ox, cy * 2 + oy)) {
                return true;
            }
        }
    }
    false
}

fn populate_throne_room(_run: &Run, plan: &mut LevelPlan) {
    plan.props
        .retain(|(k, _)| !matches!(k, PropKind::Mine | PropKind::FireTrap));
    plan.enemies.clear();

    let gens = [
        Vec2::new(-220.0, 120.0),
        Vec2::new(220.0, 120.0),
        Vec2::new(-220.0, -120.0),
        Vec2::new(220.0, -120.0),
    ];
    for p in gens {
        plan.props.push((PropKind::BigGenerator, p));
    }

    for i in 0..9 {
        let y = -320.0 + i as f32 * 80.0;
        plan.props.push((PropKind::ThroneStatue, Vec2::new(0.0, y)));
    }
    plan.boss = Some(EnemyKind::Throne);
    plan.boss_count = 1;
}

fn trim_chests(chests: &mut Vec<ChestSpawn>) {
    use std::collections::HashMap;
    let mut furthest: HashMap<u8, ChestSpawn> = HashMap::new();
    for c in chests.iter().copied() {
        let key = match c {
            ChestSpawn::Weapon(_) => 0u8,
            ChestSpawn::Ammo(_) => 1,
            ChestSpawn::Rad(_) => 2,
            // Permutation customs postdate the trim; keep them as-is.
            ChestSpawn::Custom(_, _) => continue,
        };
        let d = c.pos().length_squared();
        let keep = match furthest.get(&key) {
            Some(existing) => d > existing.pos().length_squared(),
            None => true,
        };
        if keep {
            furthest.insert(key, c);
        }
    }
    chests.clear();
    for (_, c) in furthest {
        chests.push(c);
    }
}

pub fn big_bandit_count(loop_count: u32) -> u32 {
    if loop_count == 0 {
        1
    } else {
        loop_count.saturating_mul(2).max(2)
    }
}

