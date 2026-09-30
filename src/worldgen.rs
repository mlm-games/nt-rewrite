//! Level-generation core, mechanically ported from
//! `world.rs` floor generation (plus `is_secret_area` from
//! the GML secret-area scripts).
//!
//! Scope: `LevelPlan`, `PropKind`, `ChestSpawn`, `Gen`/`Maker` +
//! `step_delta`, `rng_choose`, `turn_table`, `gml_area_from_run`,
//! `generation_goal`, `generation_goal_for_run`, `is_screen_end_wall`,
//! `floor_cell_for_wall`, `wall_cell_at`, `generate_level`,
//! `generate_palace_last`, `generate_campfire`, `generate_crib`,
//! `generate_hq_last`,
//! `world_of`, `floor_in_world`, `is_secret_area`.
//!
//! Wall helpers (`wall_point`, `nearest_wall`, `populate_throne_room`,
//! `big_bandit_count`) and the full `populate` live here too.
//!
//! Transform notes:
//! - `use bevy::...` lines dropped. `crate::game::areas::AreaId` becomes
//!   `crate::data::AreaId`; `crate::game::secret_areas::is_secret_area`
//!   becomes the local `is_secret_area` ported below. Inner
//!   `use crate::game::areas::AreaId;` lines are covered by the top import.
//! - Logic, RNG call order, tables and comments are byte-identical to source.

use glam::Vec2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::{HashMap, HashSet};

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
    pub population_events: Vec<PopulationEvent>,
    /// GML `scrPopEnemies.gml:120-121`: the palace `random(16) < 1` roll
    /// raises an `IDPDSpawn` portal (positions only -- `setup.rs` spawns
    /// them through `crate::idpd::spawn_idpd_spawn`).
    pub idpd_portals: Vec<Vec2>,
    pub boss: Option<EnemyKind>,

    pub boss_count: u32,

    pub styleb: bool,

    /// GML `GenCont`'s `lowx`/`lowy` after the max-y search at
    /// `Alarm_2.gml:29-37` -- the crib's lower-slab anchor, in pixels.
    /// Crib only; `scrPopChests` plants the Giant chest pairs from it.
    pub crib_anchor: Option<Vec2>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PopulationEvent {
    Enemy { kind: EnemyKind, pos: Vec2 },
    Prop { kind: PropKind, pos: Vec2 },
    Chest(ChestSpawn),
    PortalClear { pos: Vec2, scale: f32 },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PropKind {
    /// GML spawns nothing on the roll (e.g. the last palace subarea, or
    /// a city tile inside 128 px with no earlier branch taken).
    None,
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

    /// GML `objects/Trap`: the solid flamethrower on a scrapyards small
    /// wall (`scrPopProps.gml:24-28`).
    Trap,

    BigGenerator,
    BigGeneratorInactive,
    ThroneStatue,

    /// GML `objects/VenuzTV`: `max_hp = 1000`, `size = 5`
    /// (`VenuzTV/Create_0.gml:1-2`), the crib's destructible television.
    VenuzTV,
    /// GML `objects/VenuzCouch`: `max_hp = 1e10`, `size = 3`
    /// (`VenuzCouch/Create_0.gml:1-2`). `Step_1` re-pins `hp = max_hp`
    /// every step, so damage can never break it.
    VenuzCouch,
    /// GML `objects/VenuzCarpet`: no parent, no events, decoration only.
    VenuzCarpet,
    /// GML `objects/CarVenusFixed`: `max_hp = 25`, `size = 1`
    /// (`CarVenusFixed/Create_0.gml:1,5`).
    CarVenusFixed,
    /// GML `objects/GiantWeaponChest` / `GiantAmmoChest`: no parent, no hp,
    /// opened by `Collision_Player` (Crown Love picks the ammo variant).
    GiantWeaponChest,
    GiantAmmoChest,
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
        AreaId::Crib => 107,
        AreaId::Palace | AreaId::Campfire => 7,
        // GML's between-loop portal room borrows the Desert's tiles, the same
        // mapping `vortex::SpiralKind::for_gml_area` uses.
        AreaId::Loop => 1,
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
            AreaId::CursedCaves => 110,
            AreaId::Jungle => 110,
            // GML `scrAreaGetGenerationGoal:129` `if (area == area_crib)
            // return 20`.
            AreaId::Crib => 20,
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

/// GML `GameCont/Other_5.gml:54` `_is_secret = (area >= 100)`, so the
/// crib (107) is a secret area like every other 1xx room.
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
            | AreaId::Crib
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

/// GML `scrAreaGetMaxSubarea`.
pub fn gml_max_subarea(area: i32) -> u32 {
    match area {
        1 | 3 | 5 | 7 | 106 => 3,
        _ => 1,
    }
}

/// GML keeps five RNG states and re-seeds GameMaker's default `random()`
/// stream between `scrPopulate` phases from `rng_next_int(<state>)`
/// (`scrPopulate.gml:35,184,202,223,241,272`); `scrPopEnemies.gml:10`
/// re-seeds that same default stream from the advanced `Enemies` LCG on
/// *every* call. The port draws one `StdRng` per phase out of `run.gen_seed`
/// so a phase can no longer shift its neighbours, but neither the re-seed
/// chain nor the advanced LCG is ported, so GameMaker's exact sequences
/// still differ.
const RNG_DEFAULT: u64 = 0x2545_F491_4F6C_DD1D;
const RNG_BONES: u64 = 0xBF58_476D_1CE4_E5B9;
const RNG_ENEMIES: u64 = 0x94D0_49BB_1331_11EB;
const RNG_ENEMY_CALL: u64 = 0xD6E8_FEB8_6659_FD93;
const RNG_PROPS: u64 = 0xA076_1D64_78BD_642F;
const RNG_CHEST: u64 = 0xE703_7ED1_A0B4_28DB;
const RNG_PIZZA: u64 = 0x8EBC_6AF0_9C88_C6E3;
/// GML draws the crib's `CarVenusFixed` roll from the default
/// `random(5)` stream (`GenCont/Alarm_2.gml:95`) inside `Alarm_2`, one
/// step before `Alarm_0` reseeds that stream for `scrPopulate`; the port
/// gives it its own phase stream like every other phase.
const RNG_CRIB: u64 = 0x8A5C_D1F0_3B77_E2A4;

fn phase_rng(seed: u64, state: u64) -> StdRng {
    StdRng::seed_from_u64(seed ^ state)
}

/// GML `point_distance(floor.x, floor.y, 10016, 10016)`. A `Floor` instance
/// sits at `10000 + 32k`, so the spawn point is a half tile off the tile
/// centre -- `scrMakeFloor`, `FloorMaker/Step_0`, `scrPopulate` and
/// `scrPopEnemies` all measure from the tile *origin*, 16px from the
/// port's (16, 16) room centre.
fn cell_dist2_origin(cx: i32, cy: i32) -> f32 {
    let (x, y) = cell_center_i(cx, cy);
    let (dx, dy) = (x - TILE, y - TILE);
    dx * dx + dy * dy
}

/// GML `point_distance(bbox_center_x, bbox_center_y, 10016, 10016)` -- the
/// metric `scrPopProps` uses for `_spawn_distance` (the tile centre, one
/// half tile from the port's (16, 16) room centre).
fn cell_dist2_center(cx: i32, cy: i32) -> f32 {
    let (x, y) = cell_center_i(cx, cy);
    let (dx, dy) = (x - TILE * 0.5, y - TILE * 0.5);
    dx * dx + dy * dy
}

#[derive(Clone, Copy)]
struct Maker {
    x: i32,
    y: i32,

    dir: i32,
    /// GML `FloorMaker/Create_0:3` `styleb`, 1 maker in 6. See
    /// [`Gen::create_maker`] for why the rewrite's float `rng_float` is read
    /// as the original integer `random(6)`.
    styleb: bool,
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

/// GML `Floor/Create_0:1-4` destroys a `Floor` that overlaps an existing
/// one, so `instance_number(Floor)` is exactly the number of distinct cells
/// the maker loop has stamped -- which is what `plan.floor_cells` counts.
fn live_makers(n: usize, i: usize, next: &[Maker], branches: &[Maker]) -> i64 {
    // GML `instance_destroy` is deferred to the end of the step, so a maker
    // that just killed itself still counts for the rest of the frame, as
    // does every maker spawned earlier in the frame.
    (n - i + next.len() + branches.len()) as i64
}

/// GML `scrMakeFloor.gml:53-59`, the 8-cell ring (no centre), in draw order.
const RING8: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (0, -1),
    (-1, 0),
    (1, -1),
    (-1, -1),
    (-1, 1),
];

/// GML `scrMakeFloor.gml:210-217` -- cursed caves stamps the same eight cells
/// in a different order.
const RING8_CURSED: [(i32, i32); 8] = [
    (-1, 0),
    (-1, -1),
    (-1, 1),
    (1, 0),
    (1, -1),
    (1, 1),
    (0, 1),
    (0, -1),
];

struct Gen {
    area: i32,
    is_last: bool,
    rng: StdRng,
    seen: HashSet<(i32, i32)>,
    /// GML `Floor/Create_0:16` copies the styleb of the maker nearest the
    /// new floor; the port records the *creating* maker's styleb (first
    /// stamp wins, matching the duplicate-destroy).
    styleb_cells: HashMap<(i32, i32), bool>,
    root_styleb: Option<bool>,
    plan: LevelPlan,
}

impl Gen {
    fn new(run: &Run, area: i32, styleb: bool) -> Gen {
        Gen {
            area,
            is_last: run.floor_in_area >= gml_max_subarea(area),
            rng: StdRng::seed_from_u64(run.gen_seed),
            seen: HashSet::new(),
            styleb_cells: HashMap::new(),
            root_styleb: None,
            plan: LevelPlan {
                floor_cells: Vec::new(),
                wall_cells: HashSet::new(),
                small_walls: Vec::new(),
                bones: Vec::new(),
                bone_sprite: "images/sprBones.png",
                details: Vec::new(),
                props: Vec::new(),
                chests: Vec::new(),
                enemies: Vec::new(),
                population_events: Vec::new(),
                idpd_portals: Vec::new(),
                boss: None,
                boss_count: 1,
                styleb,
                crib_anchor: None,
            },
        }
    }

    fn f(&mut self, n: f32) -> f32 {
        self.rng.random::<f32>() * n
    }

    fn floor(&mut self, cx: i32, cy: i32, styleb: bool) {
        if self.seen.insert((cx, cy)) {
            self.styleb_cells.insert((cx, cy), styleb);
            self.plan.floor_cells.push((cx, cy));
        }
    }

    /// GML `point_distance(x, y, 10016, 10016) > 48` gates every chest and
    /// the terminating floor stamp.
    fn far(&self, cx: i32, cy: i32) -> bool {
        cell_dist2_origin(cx, cy) > 48.0 * 48.0
    }

    fn chest(&mut self, kind: fn(Vec2) -> ChestSpawn, cx: i32, cy: i32) {
        self.plan.chests.push(kind(cell_center_px(cx, cy)));
    }

    /// GML `objects/FloorMaker/Create_0.gml`: a direction draw, a styleb
    /// draw and a `Floor` stamp -- for the initial maker *and* for every
    /// `instance_create(x, y, FloorMaker)` branch.
    fn create_maker(&mut self, x: i32, y: i32) -> Maker {
        let dir = rng_choose(&mut self.rng, &[0, 0, 90, 180, 270]);
        // GML `FloorMaker/Create_0:3` reads `styleb = !rng_float(Generation, 6
        // - (area == 104 * 4))`. `104 * 4` is 416, never an area, so the
        // divisor is 6 everywhere. The original NT expression is
        // `!random(6)` -- an *integer* 0..5, true only at 0 -- so styleb is
        // 1 in 6. The rewrite's `rng_float` returns a real, and a literal
        // `!real` is `real == 0` (LCG hit 0, ~1 in 2.1e9), which would kill
        // the whole alternate-material path in `Floor/Create_0:49-109`.
        let mut styleb = self.f(6.0) < 1.0;
        if self.area == 100 {
            styleb = false;
        }
        if self.root_styleb.is_none() {
            self.root_styleb = Some(styleb);
        }
        self.floor(x, y, styleb);
        Maker { x, y, dir, styleb }
    }

    /// GML `scripts/scrMakeFloor/scrMakeFloor.gml:11-240`. Several cases
    /// mutate the maker (`x += ldrx(64, direction)`, `x += rng_choose(...)`),
    /// so every later stage reads the live position, not the step position.
    fn make_floor(&mut self, m: &mut Maker) {
        let s = m.styleb;
        match self.area {
            1 => {
                if self.f(2.0) < 1.0 {
                    for (dx, dy) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                        self.floor(m.x + dx, m.y + dy, s);
                    }
                } else {
                    self.floor(m.x, m.y, s);
                }
            }
            3 => {
                if self.f(8.0) < 1.0 || self.is_last {
                    let (xo, yo) = if self.is_last {
                        (
                            rng_choose(&mut self.rng, &[0, 1, 0, 0, -1]),
                            rng_choose(&mut self.rng, &[0, 1, 0, 0, -1]),
                        )
                    } else {
                        (0, 0)
                    };
                    for (dx, dy) in [
                        (0, 0),
                        (1, 0),
                        (1, 1),
                        (0, 1),
                        (0, -1),
                        (-1, 0),
                        (1, -1),
                        (-1, -1),
                        (-1, 1),
                    ] {
                        self.floor(m.x + xo + dx, m.y + yo + dy, s);
                    }
                } else {
                    self.floor(m.x, m.y, s);
                }
            }
            5 => {
                if self.f(11.0) < 1.0 {
                    if self.f(2.0) < 1.0 {
                        for (dx, dy) in RING8 {
                            self.floor(m.x + dx, m.y + dy, s);
                        }
                    } else {
                        for (dx, dy) in [
                            (2, -2),
                            (2, -1),
                            (2, 0),
                            (2, 1),
                            (2, 2),
                            (-2, -2),
                            (-2, -1),
                            (-2, 0),
                            (-2, 1),
                            (-2, 2),
                            (0, -2),
                            (-1, -2),
                            (1, -2),
                            (0, 2),
                            (-1, 2),
                            (1, 2),
                        ] {
                            self.floor(m.x + dx, m.y + dy, s);
                        }
                    }
                } else {
                    self.floor(m.x, m.y, s);
                }
            }
            7 => {
                if self.f(16.0) < 1.0 {
                    for dy in -1..=2 {
                        for dx in -1..=2 {
                            self.floor(m.x + dx, m.y + dy, s);
                        }
                    }
                } else {
                    for (dx, dy) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                        self.floor(m.x + dx, m.y + dy, s);
                    }
                }
            }
            100 => {
                if self.f(8.0) < 1.0 {
                    if rng_choose(&mut self.rng, &[0, 1, 2]) == 1 {
                        for dx in [1, 2, 0, -1, -2] {
                            self.floor(m.x + dx, m.y, s);
                        }
                    } else {
                        for dy in [1, 2, 0, -1, -2] {
                            self.floor(m.x, m.y + dy, s);
                        }
                    }
                } else {
                    self.floor(m.x, m.y, s);
                }
            }
            103 | 107 => {
                let n = self.plan.floor_cells.len();
                if n != 0 && n % 12 == 0 {
                    let (dx, dy) = m.step_delta();
                    m.x += dx;
                    m.y += dy;
                    for (ox, oy) in RING8 {
                        self.floor(m.x + ox, m.y + oy, s);
                    }
                } else {
                    self.floor(m.x, m.y, s);
                }
            }
            106 => {
                let n = self.plan.floor_cells.len();
                if n != 0 && n % 8 == 0 {
                    let (dx, dy) = m.step_delta();
                    m.x += dx * 2;
                    m.y += dy * 2;
                    for (ox, oy) in [
                        (-2, -2),
                        (-2, -1),
                        (-2, 0),
                        (-2, 1),
                        (-2, 2),
                        (2, -2),
                        (2, -1),
                        (2, 0),
                        (2, 1),
                        (2, 2),
                        (-1, 2),
                        (0, 2),
                        (1, 2),
                        (-1, -2),
                        (0, -2),
                        (1, -2),
                    ] {
                        self.floor(m.x + ox, m.y + oy, s);
                    }
                    m.x += dx * 2;
                    m.y += dy * 2;
                } else if self.f(3.0) < 1.0 {
                    self.floor(m.x, m.y, s);
                    for (ox, oy) in RING8 {
                        self.floor(m.x + ox, m.y + oy, s);
                    }
                } else {
                    for _ in 0..4 {
                        self.floor(m.x, m.y, s);
                        let (dx, dy) = m.step_delta();
                        m.x += dx;
                        m.y += dy;
                        self.floor(m.x, m.y, s);
                        self.chest(ChestSpawn::Ammo, m.x, m.y);
                    }
                    if self.f(3.0) < 1.0 {
                        for (ox, oy) in RING8 {
                            self.floor(m.x + ox, m.y + oy, s);
                        }
                    }
                }
            }
            101 => {
                self.floor(m.x, m.y, s);
                if self.f(3.0) < 1.0 {
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        self.floor(m.x + dx, m.y + dy, s);
                    }
                }
            }
            104 => {
                if self.plan.floor_cells.len() < 4 {
                    for (ox, oy) in RING8_CURSED {
                        self.floor(m.x + ox, m.y + oy, s);
                    }
                }
                m.x += rng_choose(&mut self.rng, &[0, 2, -2]);
                m.y += rng_choose(&mut self.rng, &[0, 2, -2]);
                for (ox, oy) in RING8_CURSED {
                    self.floor(m.x + ox, m.y + oy, s);
                }
            }
            105 => {
                if self.f(4.0) < 1.0 {
                    for (dx, dy) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                        self.floor(m.x + dx, m.y + dy, s);
                    }
                } else {
                    self.floor(m.x, m.y, s);
                }
            }
            _ => {
                self.floor(m.x, m.y, s);
            }
        }
    }

    /// GML `scripts/scrMakeFloor/scrMakeFloor.gml:242-458` plus
    /// `objects/FloorMaker/Step_0.gml:69-78`. `instance_destroy()` is
    /// deferred, so a maker that just died still evaluates its branch roll
    /// (and still spawns the child).
    fn run_makers(&mut self, goal: usize, mut makers: Vec<Maker>) {
        let mut guard = 0u32;
        while !makers.is_empty() {
            guard += 1;
            if guard > 200_000 {
                break;
            }
            let n = makers.len();
            let mut next: Vec<Maker> = Vec::with_capacity(n);
            let mut branches: Vec<Maker> = Vec::new();

            for i in 0..n {
                let mut m = makers[i];

                if self.plan.floor_cells.len() > goal {
                    if self.far(m.x, m.y) {
                        self.floor(m.x, m.y, m.styleb);
                        self.chest(ChestSpawn::Rad, m.x, m.y);
                    }
                    continue;
                }

                // GML `FloorMaker/Step_0`: the maker steps one 32px tile along
                // `direction` BEFORE the area event, so every stamp below
                // lands on the cell it just moved into.
                let (dx, dy) = m.step_delta();
                m.x += dx;
                m.y += dy;

                self.make_floor(&mut m);
                let trn = turn_table(&mut self.rng, self.area);
                m.dir = (m.dir + trn).rem_euclid(360);

                if self.area == 6 && trn.abs() == 90 && self.f(2.0) < 1.0 {
                    for (ox, oy) in RING8 {
                        self.floor(m.x + ox, m.y + oy, m.styleb);
                    }
                    if self.f(3.0) < 1.0 {
                        // GML spawns four `Server` objects here
                        // (`scrMakeFloor.gml:297-302`); the port has no
                        // `Server` prop, so only the four rolls are kept.
                        for _ in 0..4 {
                            let _ = self.f(4.0) < 3.0;
                        }
                    }
                }

                if (trn == 180 || (trn.abs() == 90 && (self.area == 3 || self.area == 104)))
                    && self.far(m.x, m.y)
                {
                    self.floor(m.x, m.y, m.styleb);
                    if self.area != 107 && self.area != 0 {
                        self.chest(ChestSpawn::Weapon, m.x, m.y);
                    }
                }

                let mut destroyed = false;
                match self.area {
                    0 => {
                        let live = live_makers(n, i, &next, &branches) as f32;
                        if self.f(19.0 + live) > 22.0 {
                            destroyed = true;
                            if self.far(m.x, m.y) {
                                self.floor(m.x, m.y, m.styleb);
                            }
                        }
                        if self.f(4.0) < 1.0 {
                            branches.push(self.create_maker(m.x, m.y));
                        }
                    }
                    106 => {
                        if self.f(10.0) < 1.0 && self.far(m.x, m.y) {
                            self.chest(ChestSpawn::Ammo, m.x, m.y);
                            self.floor(m.x, m.y, m.styleb);
                        }
                        let live = live_makers(n, i, &next, &branches);
                        if self.plan.floor_cells.len() as i64 > live * 28 {
                            branches.push(self.create_maker(m.x, m.y));
                        }
                    }
                    1 | 101 => {
                        let live = live_makers(n, i, &next, &branches) as f32;
                        if self.f(19.0 + live) > 20.0 {
                            destroyed = true;
                            if self.far(m.x, m.y) {
                                self.chest(ChestSpawn::Ammo, m.x, m.y);
                                self.floor(m.x, m.y, m.styleb);
                            }
                        }
                        if self.f(8.0) < 1.0 {
                            branches.push(self.create_maker(m.x, m.y));
                        }
                    }
                    2 => {
                        let live = live_makers(n, i, &next, &branches) as f32;
                        if self.f(14.0 + live) > 15.0 {
                            if self.far(m.x, m.y) {
                                self.chest(ChestSpawn::Ammo, m.x, m.y);
                                self.floor(m.x, m.y, m.styleb);
                            }
                            destroyed = true;
                        }
                        if self.f(15.0) < 1.0 {
                            branches.push(self.create_maker(m.x, m.y));
                        }
                    }
                    3 => {
                        let live = live_makers(n, i, &next, &branches) as f32;
                        if self.f(39.0 + live) > 40.0 {
                            if self.far(m.x, m.y) {
                                self.chest(ChestSpawn::Ammo, m.x, m.y);
                                self.floor(m.x, m.y, m.styleb);
                            }
                            destroyed = true;
                        }
                        if self.f(25.0) < 1.0 {
                            branches.push(self.create_maker(m.x, m.y));
                        }
                    }
                    4 | 104 => {
                        if !(self.area == 104 && self.f(4.0) >= 1.0) {
                            let live = live_makers(n, i, &next, &branches) as f32;
                            if self.f(9.0 + live) > 10.0 {
                                destroyed = true;
                                if self.far(m.x, m.y) {
                                    self.chest(ChestSpawn::Ammo, m.x, m.y);
                                    self.floor(m.x, m.y, m.styleb);
                                }
                            }
                            if self.f(4.0) < 1.0 {
                                branches.push(self.create_maker(m.x, m.y));
                            }
                        }
                    }
                    5 => {
                        let live = live_makers(n, i, &next, &branches) as f32;
                        if self.f(14.0 + live) > 15.0 {
                            destroyed = true;
                            if self.far(m.x, m.y) {
                                self.floor(m.x, m.y, m.styleb);
                                self.chest(ChestSpawn::Ammo, m.x, m.y);
                            }
                        }
                        if self.f(15.0) < 1.0 {
                            branches.push(self.create_maker(m.x, m.y));
                        }
                    }
                    6 => {
                        let live = live_makers(n, i, &next, &branches) as f32;
                        if self.f(21.0 + live) > 22.0 {
                            destroyed = true;
                            if self.far(m.x, m.y) {
                                self.floor(m.x, m.y, m.styleb);
                                self.chest(ChestSpawn::Ammo, m.x, m.y);
                            }
                        }
                        if self.f(20.0) < 1.0 {
                            branches.push(self.create_maker(m.x, m.y));
                        }
                    }
                    7 | 102 => {
                        if self.area == 7 {
                            let live = live_makers(n, i, &next, &branches) as f32;
                            if self.f(8.0 + live) > 9.0 {
                                destroyed = true;
                                if self.far(m.x, m.y) {
                                    self.chest(ChestSpawn::Ammo, m.x, m.y);
                                    self.floor(m.x, m.y, m.styleb);
                                }
                            }
                            if self.f(16.0) < 1.0 {
                                branches.push(self.create_maker(m.x, m.y));
                            }
                        }
                        let live = live_makers(n, i, &next, &branches) as f32;
                        if self.f(9.0 + live) > 10.0 {
                            if self.far(m.x, m.y) {
                                self.chest(ChestSpawn::Ammo, m.x, m.y);
                                self.floor(m.x, m.y, m.styleb);
                            }
                            destroyed = true;
                        }
                        if self.f(5.0) < 1.0 {
                            branches.push(self.create_maker(m.x, m.y));
                        }
                    }
                    103 | 107 => {
                        let live = live_makers(n, i, &next, &branches) as f32;
                        if self.f(31.0 + live) > 32.0 {
                            destroyed = true;
                            if self.far(m.x, m.y) {
                                self.floor(m.x, m.y, m.styleb);
                                self.chest(ChestSpawn::Ammo, m.x, m.y);
                            }
                        }
                        if self.f(20.0) < 1.0 {
                            branches.push(self.create_maker(m.x, m.y));
                        }
                    }
                    _ => {}
                }

                if self.area == 101 || self.area == 105 {
                    let live = live_makers(n, i, &next, &branches) as f32;
                    if self.f(19.0 + live) > 20.0 {
                        destroyed = true;
                        if self.far(m.x, m.y) {
                            self.chest(ChestSpawn::Ammo, m.x, m.y);
                            self.floor(m.x, m.y, m.styleb);
                        }
                    }
                    if self.f(14.0) < 1.0 {
                        branches.push(self.create_maker(m.x, m.y));
                    }
                }

                if !destroyed {
                    next.push(m);
                }
            }

            makers = next;
            makers.extend(branches);
        }
    }

    fn finish(mut self) -> (LevelPlan, HashMap<(i32, i32), bool>) {
        if let Some(root) = self.root_styleb {
            self.plan.styleb = root;
        }
        (self.plan, self.styleb_cells)
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

/// GML `GenCont/Step_0` safespawn shift verbatim: once the makers finish,
/// if the spawn ring is already full (`_numfloors >= _maxfloors`) the
/// whole level drifts one `safedir` step and grows a fresh centre floor.
/// GML runs this once per step (repeated over frames); the port settles
/// the same loop inline before walls go up (bounded 16; GML is unbounded
/// per-frame but converges the same way).
/// Skipped exactly where `scrAreaHasSafespawn` is false (campfire, crib,
/// vault, palace/HQ finales).
fn apply_safespawn_shift(plan: &mut LevelPlan, run: &Run, gen_rng: &mut StdRng) {
    let no_safe = matches!(
        run.area,
        AreaId::Campfire | AreaId::Vault | AreaId::CrownVault | AreaId::Crib
    ) || (run.area == AreaId::Palace && ((run.floor.max(1) - 1) % 15) + 1 == 15)
        || (run.area == AreaId::HQ && run.floor_in_area >= 3);
    if no_safe {
        return;
    }
    let safedis: f32 = if run.loop_count > 0 { 96.0 } else { 64.0 };
    let maxfloors = ((safedis / 32.0) * 2.5).ceil() as usize;
    // GML `GenCont/Create_0:33`: `safedir = choose(0, 90, 180, 270)` off the
    // Generation stream. GML draws it in `Create_0`, before any floor stamp,
    // which would reorder every existing draw here, so the port takes it from
    // the tail of that same stream instead.
    let (dx, dy) = match rng_choose(gen_rng, &[0i32, 90, 180, 270]) {
        0 => (1, 0),
        90 => (0, 1),
        180 => (-1, 0),
        _ => (0, -1),
    };
    let delta_px = Vec2::new(dx as f32 * TILE, dy as f32 * TILE);
    // GML counts live `Floor` instances (duplicates stack); the port's
    // deduped cells track the surplus in `stacked`. Verbatim exit law:
    // `if (_numfloors < _maxfloors) exit` — thin rings do nothing, full
    // rings shift.
    let mut stacked = 0usize;
    for _ in 0..16 {
        // GML `GenCont/Step_0:8`: `distance_to_point(10016, 10016)` measured
        // from the Floor ORIGIN, not the tile centre.
        let near = plan
            .floor_cells
            .iter()
            .filter(|(cx, cy)| cell_dist2_origin(*cx, *cy) <= safedis * safedis)
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

    if run.area == AreaId::Campfire {
        return generate_campfire(run);
    }
    if area == 7 && ((run.floor.max(1) - 1) % 15) + 1 == 15 {
        return generate_palace_last(run);
    }
    if area == 106 && run.floor_in_area >= 3 {
        return generate_hq_last(run);
    }
    if run.area == AreaId::Crib {
        return generate_crib(run);
    }
    let goal = generation_goal_for_run(run);

    let mut genr = Gen::new(run, area, false);
    let initial = genr.create_maker(0, 0);
    genr.run_makers(goal, vec![initial]);
    apply_safespawn_shift(&mut genr.plan, run, &mut genr.rng);
    let (mut plan, styleb_cells) = genr.finish();

    let floors = plan.floor_cells.clone();
    build_walls(run, &floors, &mut plan);
    let walls = plan.wall_cells.clone();
    populate(run, &floors, &walls, &mut plan, &styleb_cells);
    plan
}

/// GML `objects/FloorMaker/Step_0.gml:4-67`: the palace finale is laid out by
/// the maker's own Step, not by `scrMakeFloor`, and `GenCont/Alarm_0` skips
/// `scrPopulate` there, so no bones, props, enemies or chests.
fn generate_palace_last(run: &Run) -> LevelPlan {
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
        population_events: Vec::new(),
        idpd_portals: Vec::new(),
        boss: None,
        boss_count: 1,
        styleb: false,
        crib_anchor: None,
    };
    let mut seen = HashSet::new();
    // GML `FloorMaker/Step_0.gml:12-31`: 48 rows upward from the spawn tile,
    // 8 columns from `dix = -4`, the outermost two columns cut for the last
    // four rows (`diy < -43`). GML writes the grid at `x + dix*32 + 16`, a
    // half-tile offset the 32px cell grid cannot hold, so the +16 is dropped.
    for row in 0..48i32 {
        let diy = -row;
        for col in 0..8i32 {
            let dix = col - 4;
            if (dix == -4 || dix == 3) && diy < -43 {
                continue;
            }
            let c = (dix, diy);
            if seen.insert(c) {
                plan.floor_cells.push(c);
            }
        }
        // GML :27-30 -- every fifth row between 10 and 33 tiles up.
        if (-diy) % 5 == 0 && -diy > 9 && -diy < 34 {
            let py = (diy * 32 - 32) as f32;
            for sx in [-80.0f32, 112.0f32] {
                plan.props.push((PropKind::ThroneStatue, Vec2::new(sx, py)));
            }
        }
    }
    // GML :33-49: four inactive generators; the two at `diy*32 + 384 + 64`
    // sit 20 rows further up, and `NothingInactive` marks the far end.
    for sx in [176.0f32, -144.0f32] {
        for py in [-928.0f32, -1088.0f32] {
            plan.props.push((PropKind::BigGeneratorInactive, Vec2::new(sx, py)));
        }
    }
    // GML :51-63: each inactive generator lays six `Floor`s. `bbox_top` of
    // `sprBigGeneratorInactive` is 1, so `(bbox_top div 32) + yy` is 0..2 --
    // the arena's own bottom three rows, not rows under the generator.
    for (gx, _) in [(176.0f32, 0.0f32), (-144.0f32, 0.0f32)] {
        let gcol = (gx / 32.0).floor() as i32;
        for yy in 0..3i32 {
            for xx in 0..2i32 {
                let c = (gcol + xx - 1, yy);
                if seen.insert(c) {
                    plan.floor_cells.push(c);
                }
            }
        }
    }
    let floors = plan.floor_cells.clone();
    build_walls(run, &floors, &mut plan);
    populate_throne_room(&mut plan);
    plan
}

fn generate_campfire(run: &Run) -> LevelPlan {
    // GML `objects/GenCont/Create_0.gml:20-30`: a 5x3 floor block, then SEVEN
    // `FloorMaker` instances stacked on the spawn point, plus the ordinary one
    // at :43 -- eight makers running the area-0 turn table.
    let mut genr = Gen::new(run, 0, false);
    for xx in -2..=2 {
        for yy in -1..=1 {
            genr.floor(xx, yy, false);
        }
    }
    let mut makers = Vec::with_capacity(8);
    for _ in 0..7 {
        makers.push(genr.create_maker(0, 0));
    }
    makers.push(genr.create_maker(0, 0));
    let goal = generation_goal_for_run(run);
    genr.run_makers(goal, makers);
    let (mut plan, styleb_cells) = genr.finish();

    let floors = plan.floor_cells.clone();
    build_walls(run, &floors, &mut plan);
    let walls = plan.wall_cells.clone();
    populate(run, &floors, &walls, &mut plan, &styleb_cells);
    plan
}

/// GML `objects/GenCont/Alarm_2.gml:28-105` verbatim. The crib is an
/// ordinary area-107 maker run (`goal = 20`,
/// `scrAreaGetGenerationGoal.gml:129`) plus the two hand-laid slabs and
/// their props. GML's `FloorMaker/Step_0.gml:82-86` sets
/// `GenCont.alarm[0] = 3` and `alarm[2] = 2` in the same step, so
/// `Alarm_2` fires BEFORE `Alarm_0` -- the slab floors are in the cell
/// list `mcr_floor_make_walls` and `scrPopulate` walk, so they are added
/// before those two passes here.
fn generate_crib(run: &Run) -> LevelPlan {
    let mut genr = Gen::new(run, 107, false);
    let initial = genr.create_maker(0, 0);
    genr.run_makers(generation_goal_for_run(run), vec![initial]);
    let (mut plan, styleb_cells) = genr.finish();

    build_crib_rooms(&mut plan, run);

    let floors = plan.floor_cells.clone();
    build_walls(run, &floors, &mut plan);
    let walls = plan.wall_cells.clone();
    populate(run, &floors, &walls, &mut plan, &styleb_cells);

    // GML `GenCont/Destroy_0.gml:126-139`: the crib keeps no enemy (and no
    // chestprop but the two Giant kinds, which `apply_chest_permutations`
    // already reduced to none). The `with Wall` half of `:127-129` is a
    // no-op here: `build_walls` never stamps a wall whose owning 32px
    // cell is floor, which is exactly `place_meeting(wall.x, wall.y,
    // Floor)`.
    plan.enemies.clear();
    plan.population_events
        .retain(|event| !matches!(event, PopulationEvent::Enemy { .. }));
    plan
}

/// GML `GenCont/Alarm_2.gml:29-104`: the lower 11x7 slab with the TV,
/// couch, two `MoneyPile`s and the carpet, then the upper 10x7 slab with
/// its `CarVenusFixed` scatter. The Giant chest pairs (`:57-74`) need the
/// live Open-Mind level and Crown Love, so they ride
/// [`apply_chest_permutations`] off [`LevelPlan::crib_anchor`] -- GML
/// creates them in this same alarm, before `scrPopulate` runs, and
/// `scrPopChests` leaves them alone (they have no `chestprop` parent).
fn build_crib_rooms(plan: &mut LevelPlan, run: &Run) {
    let mut cells = plan.floor_cells.clone();
    let mut seen: HashSet<(i32, i32)> = cells.iter().copied().collect();
    let push_cell = |cells: &mut Vec<(i32, i32)>, seen: &mut HashSet<(i32, i32)>, c: (i32, i32)| {
        if seen.insert(c) {
            cells.push(c);
        }
    };

    // GML :29-37 -- `lowx`/`lowy` settle on the LOWEST existing `Floor`.
    // The test `y > other.lowy` is strict, so the FIRST of several equal
    // rows keeps the anchor: scanning in reverse and taking `max_by_key`
    // (which returns the LAST maximum) yields that first one.
    let (bx, by) = cells
        .iter()
        .rev()
        .max_by_key(|(_, cy)| *cy)
        .copied()
        .unwrap_or((0, 0));
    let lowx = bx * TILE as i32;
    let lowy = by * TILE as i32;
    let low = Vec2::new(lowx as f32, lowy as f32);
    plan.crib_anchor = Some(low);

    // GML :39-50 -- 11 columns x 7 rows, starting one tile under `lowy`
    // and running DOWN (+y). `Floor/Create_0.gml:1-4` destroys an
    // overlapping stamp, so the dedupe is the same law.
    let mut dix = -160i32;
    for _ in 0..11 {
        let mut diy = 0i32;
        for _ in 0..7 {
            push_cell(
                &mut cells,
                &mut seen,
                ((lowx + dix) / TILE as i32, (lowy + 32 + diy) / TILE as i32),
            );
            diy += 32;
        }
        dix += 32;
    }
    plan.floor_cells = cells.clone();

    // GML :52-55, :76 -- instance positions are pixel-exact, so they ride
    // `plan.props` verbatim.
    for (kind, at) in [
        (PropKind::VenuzTV, (16.0f32, 248.0f32)),
        (PropKind::VenuzCouch, (16.0, 104.0)),
        (PropKind::MoneyPile, (-48.0, 104.0)),
        (PropKind::MoneyPile, (80.0, 104.0)),
        (PropKind::VenuzCarpet, (16.0, 104.0)),
    ] {
        plan.props
            .push((kind, Vec2::new(low.x + at.0, low.y + at.1)));
    }

    // GML :78-86 -- now the HIGHEST `Floor`, searched over every instance
    // INCLUDING the lower slab just laid. That slab sits strictly below
    // every pre-existing floor, so the minimum is unchanged; the strict
    // `y < other.lowy` makes the first of equal rows win, which is
    // `min_by_key`'s tie rule.
    let (ux, uy) = plan
        .floor_cells
        .iter()
        .min_by_key(|(_, cy)| *cy)
        .copied()
        .unwrap_or((0, 0));
    let upx = ux * TILE as i32;
    let upy = uy * TILE as i32;

    // GML :88-104 -- 10 columns x 7 rows running UP (-y) from `upy + 32`,
    // each stamped floor rolling `random(5) < 1 || instance_number
    // (CarVenusFixed) == 0` for a car past 96px from the spawn point. The
    // `&&` short-circuits on the distance test, so the draw is only paid
    // past 96px.
    let mut rng = phase_rng(run.gen_seed, RNG_CRIB);
    let mut cars = 0usize;
    let mut dix = -160i32;
    for _ in 0..10 {
        let mut diy = 0i32;
        for _ in 0..7 {
            let cx = (upx + dix) / TILE as i32;
            let cy = (upy + 32 + diy) / TILE as i32;
            push_cell(&mut cells, &mut seen, (cx, cy));
            let at = Vec2::new(cx as f32 * TILE + TILE * 0.5, cy as f32 * TILE + TILE * 0.5);
            if at.distance(Vec2::splat(TILE * 0.5)) > 96.0
                && (rng.random::<f32>() * 5.0 < 1.0 || cars == 0)
            {
                cars += 1;
                plan.props.push((PropKind::CarVenusFixed, at));
            }
            diy -= 32;
        }
        dix += 32;
    }
    plan.floor_cells = cells;
}

fn generate_hq_last(run: &Run) -> LevelPlan {
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
        population_events: Vec::new(),
        idpd_portals: Vec::new(),
        boss: None,
        boss_count: 1,
        styleb: true,
        crib_anchor: None,
    };
    let mut seen = HashSet::new();
    let mut cells: Vec<(i32, i32)> = Vec::new();
    macro_rules! stamp {
        ($c:expr) => {
            if seen.insert($c) {
                cells.push($c);
            }
        };
    }
    // GML `objects/FloorMaker/Create_0.gml:37-48`.
    for row in 0..10i32 {
        for col in 0..10i32 {
            stamp!((col - 5, -row));
        }
    }
    // GML :50-81 -- the shaft pairs and the two `LastIntro` wings. Every GML
    // x lands on a half tile, so the cells collapse onto the 32px grid.
    for c in [
        (-1, -11),
        (-1, -10),
        (0, -11),
        (0, -10),
        (-1, -11),
        (-1, -10),
        (1, -11),
        (1, -10),
        (-1, 1),
        (-1, 2),
        (0, 1),
        (0, 2),
        (-1, 1),
        (-1, 2),
        (1, 1),
        (1, 2),
        (-7, -6),
        (-6, -6),
        (-6, -5),
        (-6, -4),
        (-6, -3),
        (5, -6),
        (5, -5),
        (5, -4),
        (5, -3),
        (6, -6),
        (6, -5),
        (6, -4),
        (6, -3),
    ] {
        stamp!(c);
    }
    plan.floor_cells = cells;
    // GML :85-88: four hand-placed `Wall`s, which `mcr_floor_make_walls`
    // never generates.
    for (wx, wy) in [(-4i32, -13i32), (-4, -2), (5, -13), (5, -2)] {
        plan.wall_cells.insert((wx, wy));
    }
    // GML :89-92: four `PlantPot`s.
    for (px, py) in [
        (-56.0f32, -240.0f32),
        (-56.0, -32.0),
        (88.0, -240.0),
        (88.0, -32.0),
    ] {
        plan.props.push((PropKind::PlantPot, Vec2::new(px, py)));
    }
    let floors = plan.floor_cells.clone();
    build_walls(run, &floors, &mut plan);
    plan
}


fn rebuild_population_events(plan: &mut LevelPlan, base_events: &[PopulationEvent]) {
    let mut events: Vec<PopulationEvent> = base_events
        .iter()
        .copied()
        .filter(|event| !matches!(event, PopulationEvent::Chest(_)))
        .collect();
    events.extend(plan.chests.iter().copied().map(PopulationEvent::Chest));
    let base_enemy_count = events
        .iter()
        .filter(|event| matches!(event, PopulationEvent::Enemy { .. }))
        .count();
    if plan.enemies.len() >= base_enemy_count {
        events.extend(
            plan.enemies
                .iter()
                .skip(base_enemy_count)
                .copied()
                .map(|(kind, pos)| PopulationEvent::Enemy { kind, pos }),
        );
    }
    plan.population_events = events;
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
    /// GML raises the tutorial's weapon chest from `TutCont/Alarm_0:53`
    /// (`if (!_any) instance_create(10016, 10016, WeaponChest)`), and
    /// `scrPopChests` — the only caller of the trim, `scrPopulate:224` —
    /// never runs for the 5-floor `TutCont` arena. Its `do…until` removes the
    /// last chest of a kind, so trimming the tutorial would delete the only
    /// gun pickup in the level.
    pub tutorial: bool,
    pub player_pos: Vec2,
    pub seed: u64,
}

/// GML `scrPopChests` output: whether a `HostileHorror` hatched (the
/// caller records `Run.horror`). Mimics/horrors ride `plan.enemies`;
/// special chests ride `plan.chests` as `ChestSpawn::Custom`.
pub struct ChestPermuteOut {
    pub horror: bool,
}

/// GML `scrPopChests.gml:37-41,45-49,53-57`.
fn trim_chest_kind(list: &mut Vec<Vec2>, keep: usize, rng: &mut StdRng) {
    loop {
        let target = Vec2::new(
            rng.random_range(-250.0..250.0),
            rng.random_range(-250.0..250.0),
        );
        if !list.is_empty() {
            let mut best = 0usize;
            let mut best_d = f32::MAX;
            for (i, p) in list.iter().enumerate() {
                let d = p.distance_squared(target);
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            list.remove(best);
        }
        if list.len() <= keep {
            break;
        }
    }
}

/// GML `scrPopChests.gml:192-222`.
fn replace_prop_with_chest(
    plan: &mut LevelPlan,
    list: &mut Vec<Vec2>,
    area: &AreaId,
    base_events: &mut Vec<PopulationEvent>,
) {
    if !list.is_empty() || *area == AreaId::Campfire || is_secret_area(*area) {
        return;
    }
    // GML skips a fixed list of prop objects; none of the `PropKind`s the
    // port can emit is on it, so every prop is a candidate.
    let spawn = Vec2::new(TILE * 0.5, TILE * 0.5);
    let mut best: Option<(usize, Vec2, f32)> = None;
    for (index, &(_, at)) in plan.props.iter().enumerate() {
        let d2 = at.distance_squared(spawn);
        if d2 > 160.0 * 160.0 && best.is_none_or(|(_, _, bd)| d2 > bd) {
            best = Some((index, at, d2));
        }
    }
    let Some((index, at, _)) = best else {
        return;
    };
    let (prop_kind, _) = plan.props.remove(index);
    if let Some(event_index) = base_events.iter().position(|event| {
        matches!(
            event,
            PopulationEvent::Prop {
                kind: event_kind,
                pos: event_pos,
            } if *event_kind == prop_kind && *event_pos == at
        )
    }) {
        base_events.remove(event_index);
    }
    list.push(at);
}

/// Verbatim `scripts/scrPopChests/scrPopChests.gml`: vault proto-chest +
/// chestless areas, Open-Mind bonus counts, trim to 1 + bonus per base
/// kind (GML destroys nearest-to-`10016+orandom(250)`; the port shuffles
/// with the seeded stream then truncates — same count law, stable order),
/// rad permutations (Rogue / noradch horror-or-big / half-health /
/// desert styleb maggot), crown Life/Love conversions, mimic rolls, and
/// the hardmode desert 1-1 `BigWeaponChest` arm.
pub fn apply_chest_permutations(plan: &mut LevelPlan, ctx: ChestPermuteCtx) -> ChestPermuteOut {
    let mut rng = phase_rng(ctx.seed, RNG_CHEST);
    let mut out = ChestPermuteOut { horror: false };
    let mut base_events: Vec<PopulationEvent> = plan
        .population_events
        .iter()
        .copied()
        .filter(|event| !matches!(event, PopulationEvent::Chest(_)))
        .collect();
    if base_events.is_empty() {
        base_events.extend(
            plan.enemies
                .iter()
                .copied()
                .map(|(kind, pos)| PopulationEvent::Enemy { kind, pos }),
        );
        base_events.extend(
            plan.props
                .iter()
                .copied()
                .map(|(kind, pos)| PopulationEvent::Prop { kind, pos }),
        );
    }

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
        let mut events = base_events.clone();
        for (k, p) in customs {
            let chest = ChestSpawn::Custom(k, p);
            plan.chests.push(chest);
            events.push(PopulationEvent::Chest(chest));
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
            let chest = ChestSpawn::Custom(ChestKind::Proto, p);
            plan.chests.push(chest);
            events.push(PopulationEvent::Chest(chest));
            let statue = (EnemyKind::ProtoStatue, p + Vec2::new(0.0, 64.0));
            plan.enemies.push(statue);
            events.push(PopulationEvent::Enemy {
                kind: statue.0,
                pos: statue.1,
            });
            for i in 0..4 {
                let a = i as f32 * std::f32::consts::TAU / 4.0;
                let enemy = (EnemyKind::Bandit, p + Vec2::new(a.cos(), a.sin()) * 48.0);
                plan.enemies.push(enemy);
                events.push(PopulationEvent::Enemy {
                    kind: enemy.0,
                    pos: enemy.1,
                });
            }
        }
        plan.population_events = events;
        return out;
    }

    // GML `scrPopChests.gml:19-23,65-69`: the crib sets `_tot_chests = 0`,
    // so every Ammo/Weapon/Rad chest the area-107 maker stamped is
    // destroyed. What survives is the `(1 + open_mind)` pair of Giant
    // chests `GenCont/Alarm_2.gml:57-74` planted a step earlier -- Crown
    // Love picks the ammo pair. `if (instance_exists(Player))` is always
    // true: the crib is only ever entered through a portal.
    if ctx.area == AreaId::Crib {
        let mut events = base_events.clone();
        plan.chests.clear();
        if let Some(low) = plan.crib_anchor {
            let kind = if ctx.crown_love {
                PropKind::GiantAmmoChest
            } else {
                PropKind::GiantWeaponChest
            };
            let open_mind = u32::from(ctx.open_mind);
            let mut dx = 90.0f32 + open_mind as f32 * 32.0;
            let mut dy = 64.0f32;
            for _ in 0..=open_mind {
                for at in [
                    Vec2::new(low.x + 16.0 - dx, low.y + dy),
                    Vec2::new(low.x + 16.0 + dx, low.y + dy),
                ] {
                    plan.props.push((kind, at));
                    events.push(PopulationEvent::Prop { kind, pos: at });
                }
                dy -= 28.0;
                dx -= 48.0;
            }
        }
        for (k, p) in customs {
            plan.chests.push(ChestSpawn::Custom(k, p));
        }
        plan.population_events = events;
        return out;
    }

    // GML `GenCont/Alarm_0.gml:45-48`: the palace and HQ finales never call
    // `scrPopulate`, and every chest they own is destroyed on entry.
    let finale = (ctx.area == AreaId::Palace && ctx.subarea >= 3)
        || (ctx.area == AreaId::HQ && ctx.subarea >= 3);
    if ctx.area == AreaId::Campfire || finale {
        plan.chests.clear();
        for (k, p) in customs {
            plan.chests.push(ChestSpawn::Custom(k, p));
        }
        rebuild_population_events(plan, &base_events);
        return out;
    }

    // GML `scrPopChests.gml:25-31`: Open Mind adds `2 * level` extra chests
    // worth of *allowance* (one `choose(1,2,3)` per roll) -- it never creates
    // one itself.
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

    // GML `scrPopChests.gml:35-58`: `do { destroy the chest nearest
    // 10016 + orandom(250) } until instance_number(kind) <= _tot + _bonus`.
    // The body always runs once, so a kind holding exactly `_tot` chests
    // comes out empty. The tutorial never reaches this pass.
    if !ctx.tutorial {
        trim_chest_kind(&mut weapons, 1 + wb, &mut rng);
        trim_chest_kind(&mut rads, 1 + rb, &mut rng);
        trim_chest_kind(&mut ammos, 1 + ab, &mut rng);
    }

    // GML `scrPopChests.gml:61-63` then `scrPopChests.gml:192-222`: one call
    // per kind, each refusing when that kind still exists, on the campfire,
    // on every `area >= 100` secret, and on the HQ finale -- so route areas
    // only. It converts the furthest prop past 160 px.
    for list in [&mut rads, &mut weapons, &mut ammos] {
        replace_prop_with_chest(plan, list, &ctx.area, &mut base_events);
    }

    // Rad permutations, in GML order.
    let mut rad_out: Vec<(ChestSpawn, Option<(EnemyKind, Vec2)>)> = Vec::new();
    for p in rads {
        if ctx.rogue_in_run {
            rad_out.push((ChestSpawn::Custom(ChestKind::Rogue, p), None));
            continue;
        }
        if ctx.noradch > 0 {
            if ctx.noradch >= 2 && !ctx.horror_done && !out.horror {
                rad_out.push((ChestSpawn::Rad(p), Some((EnemyKind::HostileHorror, p))));
                out.horror = true;
            } else {
                rad_out.push((ChestSpawn::Custom(ChestKind::RadBig, p), None));
            }
            continue;
        }
        if ctx.player_half_health && rng.random_range(0.0..2.0) < 1.0 {
            rad_out.push((ChestSpawn::Custom(ChestKind::Health, p), None));
            continue;
        }
        if plan.styleb && ctx.area == AreaId::Desert && rng.random_range(0.0..3.0) < 1.0 {
            rad_out.push((ChestSpawn::Custom(ChestKind::RadMaggot, p), None));
        } else {
            rad_out.push((ChestSpawn::Rad(p), None));
        }
    }

    // Base survivors.
    let mut final_chests: Vec<(ChestSpawn, Option<(EnemyKind, Vec2)>)> = weapons
        .into_iter()
        .map(|p| (ChestSpawn::Weapon(p), None))
        .collect();
    final_chests.extend(ammos.into_iter().map(|p| (ChestSpawn::Ammo(p), None)));
    final_chests.extend(rad_out);
    for (k, p) in customs {
        final_chests.push((ChestSpawn::Custom(k, p), None));
    }

    // Crown Life: every Rad becomes Health. Crown Love: every non-Proto,
    // non-Rogue chest plus every Rad becomes Ammo.
    if ctx.crown_life {
        for (c, _) in final_chests.iter_mut() {
            // GML `scrPopChests.gml:124-129` is `with RadChest`, which is
            // hierarchy-inclusive: plain `RadChest`, `RadChestBig` AND
            // `RadMaggotChest` (the latter two both declare
            // `parentObjectId: RadChest`).
            if matches!(&*c, ChestSpawn::Rad(_))
                || matches!(
                    &*c,
                    ChestSpawn::Custom(ChestKind::RadBig | ChestKind::RadMaggot, _)
                )
            {
                let p = (*c).pos();
                *c = ChestSpawn::Custom(ChestKind::Health, p);
            }
        }
    }
    if ctx.crown_love {
        for (c, _) in final_chests.iter_mut() {
            let p = c.pos();
            match *c {
                ChestSpawn::Custom(ChestKind::Proto, _)
                | ChestSpawn::Custom(ChestKind::Rogue, _) => {}
                _ => *c = ChestSpawn::Custom(ChestKind::Ammo, p),
            }
        }
    }

    // Mimic rolls (GML order; BigWeapon needs nochest and no existing
    // BigWeapon; the crib returns above, its `random(3) < 3` gate at
    // `scrPopChests.gml:162` being all-true).
    let sewers_gate = (ctx.area != AreaId::Desert && ctx.area != AreaId::Campfire) || ctx.loops > 0;
    let mut no_big_yet = !final_chests
        .iter()
        .any(|entry| matches!(entry.0, ChestSpawn::Custom(ChestKind::BigWeapon, _)));
    let mut swapped: Vec<ChestSpawn> = Vec::with_capacity(final_chests.len());
    let mut phase_events: Vec<PopulationEvent> = Vec::with_capacity(final_chests.len());
    let mut phase_enemies: Vec<(EnemyKind, Vec2)> = Vec::new();
    for (c, rad_enemy) in final_chests {
        if let Some((kind, pos)) = rad_enemy {
            phase_enemies.push((kind, pos));
            phase_events.push(PopulationEvent::Enemy { kind, pos });
            continue;
        }
        match c {
            ChestSpawn::Ammo(p) if sewers_gate && rng.random_range(0.0..11.0) < 1.0 => {
                phase_enemies.push((EnemyKind::Mimic, p));
                phase_events.push(PopulationEvent::Enemy {
                    kind: EnemyKind::Mimic,
                    pos: p,
                });
            }
            ChestSpawn::Weapon(p)
                if rng.random_range(0.0..4.0) < ctx.nochest as f32 && no_big_yet =>
            {
                no_big_yet = false;
                let chest = ChestSpawn::Custom(ChestKind::BigWeapon, p);
                swapped.push(chest);
                phase_events.push(PopulationEvent::Chest(chest));
            }
            ChestSpawn::Weapon(p)
                if ctx.same_weapons_for > 4
                    && rng.random_range(0.0..100.0) < (ctx.same_weapons_for - 4) as f32 =>
            {
                phase_enemies.push((EnemyKind::WepMimic, p));
                phase_events.push(PopulationEvent::Enemy {
                    kind: EnemyKind::WepMimic,
                    pos: p,
                });
            }
            ChestSpawn::Custom(ChestKind::Health, p)
                if sewers_gate && rng.random_range(0.0..51.0) < 1.0 =>
            {
                phase_enemies.push((EnemyKind::SuperMimic, p));
                phase_events.push(PopulationEvent::Enemy {
                    kind: EnemyKind::SuperMimic,
                    pos: p,
                });
            }
            other => {
                swapped.push(other);
                phase_events.push(PopulationEvent::Chest(other));
            }
        }
    }
    plan.chests = swapped;
    plan.enemies.extend(phase_enemies);
    let mut events = base_events;
    events.extend(phase_events);
    // GML hardmode desert 1-1: every player gets a BigWeaponChest
    // (`hardmode && loops - hardmode <= 0`).
    if ctx.hardmode
        && ctx.loops as i32 - i32::from(ctx.hardmode) <= 0
        && ctx.area == AreaId::Desert
        && ctx.subarea == 1
    {
        let chest = ChestSpawn::Custom(ChestKind::BigWeapon, ctx.player_pos);
        plan.chests.push(chest);
        events.push(PopulationEvent::Chest(chest));
    }
    plan.population_events = events;
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
    // GML `mcr_floor_make_walls` verbatim: the probes are the 12 cells
    // of the 16px ring around the 32px floor tile, i.e. half-open
    // [x-16, x+48) x [y-16, y+48) in floor-px minus the 2x2 floor block
    // itself: corners (-1,-1), (2,-1), (-1,2), (2,2) are single cells,
    // the other 8 come in wall/floor pairs (two 16px halves each).
    // `position_meeting` tests the point, not the cell, so the diagonal
    // corners probe, the pairs probe both halves, and every probe only
    // fires where the owning 32px tile is not floor.
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

/// GML `place_meeting(x, y, Wall)` reduced to a point lookup: a `Wall` is
/// `sprWall*Bot`, bbox `0,0..15,15`, so a point belongs to the 16px cell it
/// falls in. Generated walls never occupy a floor's own 2x2 block, but a
/// `scrPopProps` small wall can -- and GML tests exactly that.
fn wall_point_in(walls: &HashSet<(i32, i32)>, small: &[(i16, i16)], px: f32, py: f32) -> bool {
    let wx = (px / WALL_PX).floor() as i32;
    let wy = (py / WALL_PX).floor() as i32;
    walls.contains(&(wx, wy)) || small.contains(&(wx as i16, wy as i16))
}

fn wall_point(plan: &LevelPlan, walls: &HashSet<(i32, i32)>, px: f32, py: f32) -> bool {
    wall_point_in(walls, &plan.small_walls, px, py)
}

/// GML `instance_nearest(x, y, Wall)` (measured to the instance origin, which
/// is the top-left of its 16px mask).
fn nearest_wall(plan: &LevelPlan, walls: &HashSet<(i32, i32)>, px: f32, py: f32) -> Option<Vec2> {
    let mut best: Option<(f32, Vec2)> = None;
    for &(wx, wy) in walls {
        let at = Vec2::new(wx as f32 * WALL_PX, wy as f32 * WALL_PX);
        let d = at.distance_squared(Vec2::new(px, py));
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, at));
        }
    }
    for &(wx, wy) in &plan.small_walls {
        let at = Vec2::new(wx as f32 * WALL_PX, wy as f32 * WALL_PX);
        let d = at.distance_squared(Vec2::new(px, py));
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, at));
        }
    }
    best.map(|(_, at)| at)
}

/// GML `GameCont.hard` is a live accumulator, not `scrAreaGetDifficulty`:
/// `GameCont/Create_0.gml:9` seeds it at 0 (`:85-88` sets 13 and bumps
/// `loops` in hardmode) and `GameCont/Other_5.gml:136` adds
/// `scrGameIsHardmode() ? 2 : 1` on every room advance -- including the
/// secret areas that `Other_5` routes back to. `scrPopulate.gml:4` reads the
/// accumulator, so floor 1 populates with `hard = 0`.
pub fn game_hard(run: &Run) -> f32 {
    run.hard as f32
}

fn enemy_body_on_rect(pos: Vec2, radius: f32, x: f32, y: f32) -> bool {
    let near = Vec2::new(pos.x.clamp(x, x + TILE), pos.y.clamp(y, y + TILE));
    pos.distance(near) <= radius
}

/// GML `scripts/scrPopulate/scrPopulate.gml` end to end, with each phase on
/// its own stream (see `phase_rng`). The chest restriction lives in
/// `apply_chest_permutations` because the trim needs the Open-Mind count.
fn populate(
    run: &Run,
    floors: &[(i32, i32)],
    walls: &HashSet<(i32, i32)>,
    plan: &mut LevelPlan,
    styleb_cells: &HashMap<(i32, i32), bool>,
) {
    let area = gml_area_from_run(run);
    let is_last = run.floor_in_area >= gml_max_subarea(area);
    let hard = game_hard(run);
    let enemy_cap = 3.0 + hard / 1.5;
    let plan_styleb = plan.styleb;
    let styleb_of = |cx: i32, cy: i32| *styleb_cells.get(&(cx, cy)).unwrap_or(&plan_styleb);

    let mut prop_tiles: HashSet<(i32, i32)> = HashSet::new();
    for chest in &plan.chests {
        let pos = chest.pos();
        prop_tiles.insert((
            ((pos.x - TILE * 0.5) / TILE).floor() as i32,
            ((pos.y - TILE * 0.5) / TILE).floor() as i32,
        ));
    }
    for &(_, pos) in &plan.props {
        prop_tiles.insert((
            ((pos.x - TILE * 0.5) / TILE).floor() as i32,
            ((pos.y - TILE * 0.5) / TILE).floor() as i32,
        ));
    }
    let mut events: Vec<PopulationEvent> = plan
        .enemies
        .iter()
        .copied()
        .map(|(kind, pos)| PopulationEvent::Enemy { kind, pos })
        .chain(
            plan.props
                .iter()
                .copied()
                .map(|(kind, pos)| PopulationEvent::Prop { kind, pos }),
        )
        .collect();
    let mut enemy_calls: u64 = 0;

    // GML :16-31 -- default stream, `_spawndist = 120` for this pass (the
    // city-boss 150 override only lands on :33, after it).
    let mut rng = phase_rng(run.gen_seed, RNG_DEFAULT);
    for &(cx, cy) in floors {
        if rng.random::<f32>() * (10.0 + hard) < hard
            && cell_dist2_origin(cx, cy) > 120.0 * 120.0
            && !prop_tiles.contains(&(cx, cy))
        {
            let mut er = phase_rng(
                run.gen_seed.wrapping_add(enemy_calls.wrapping_mul(0x9E37_79B9)),
                RNG_ENEMY_CALL,
            );
            enemy_calls += 1;
            let props_before = plan.props.len();
            let enemies_before = plan.enemies.len();
            scr_pop_enemies(
                run,
                area,
                styleb_of(cx, cy),
                is_last,
                &mut plan.enemies,
                &mut plan.props,
                &mut prop_tiles,
                &mut plan.idpd_portals,
                &mut er,
                (cx, cy),
                &plan.small_walls,
                walls,
            );
            record_population_delta(
                &mut events,
                props_before,
                &plan.props,
                enemies_before,
                &plan.enemies,
            );
        }
        // GML :26-30 -- `random_range(bbox_left, bbox_right)`, i.e. an offset
        // in [0,31) from the tile ORIGIN.
        if rng.random::<f32>() * 6.0 < 1.0 {
            let ux = rng.random_range(0.0..31.0);
            let uy = rng.random_range(0.0..31.0);
            plan.details.push(Vec2::new(cx as f32 * TILE + ux, cy as f32 * TILE + uy));
        }
    }

    // GML :37-177 -- bone decals, on the Generation stream. `None` means the
    // area has no bone branch at all (mansion, labs, palace, vault, HQ, crib
    // and pizza sewers: the second `area_sewers` arm at :158-166 is a dead
    // duplicate of the first, so 102 never matches it).
    let bone_pass: Option<(i32, bool)> = match area {
        1 => Some((0, false)),
        0 => Some((0, false)),
        3 => Some((7, false)),
        5 => Some((7, false)),
        4 => Some((9, false)),
        104 => Some((9, false)),
        101 => Some((9, false)),
        2 => Some((10, true)),
        105 => Some((10, true)),
        _ => None,
    };
    plan.bone_sprite = match area {
        0 => "images/sprNightBones.png",
        3 => "images/sprScrapDecal.png",
        5 => "images/sprIceDecal.png",
        4 => "images/sprCaveDecal.png",
        104 => "images/sprInvCaveDecal.png",
        101 => "images/sprCoral.png",
        2 => "images/sprSewerDecal.png",
        105 => "images/sprJungleDecal.png",
        _ => "images/sprBones.png",
    };
    if let Some((roll, paired)) = bone_pass {
        let mut rng = phase_rng(run.gen_seed, RNG_BONES);
        for &(cx, cy) in floors {
            let (ox, oy) = (cx as f32 * TILE, cy as f32 * TILE);
            // GML :41 etc -- three single-point probes, not an edge test:
            // `!place_free(x - 32, y) && !place_free(x + 32, y) &&
            // place_free(x, y)`.
            if wall_point(plan, walls, ox - 32.0, oy)
                || !wall_point(plan, walls, ox + 32.0, oy)
                || wall_point(plan, walls, ox, oy)
            {
                continue;
            }
            if paired {
                if roll == 0 || rng.random_range(0i32..roll) < 1 {
                    plan.bones.push((Vec2::new(ox, oy + 16.0), false));
                    plan.bones.push((Vec2::new(ox + 32.0, oy + 16.0), true));
                }
            } else {
                for (dx, dy, flip) in [
                    (0.0, 0.0, false),
                    (0.0, 16.0, false),
                    (32.0, 0.0, true),
                    (32.0, 16.0, true),
                ] {
                    if roll == 0 || rng.random_range(0i32..roll) < 1 {
                        plan.bones.push((Vec2::new(ox + dx, oy + dy), flip));
                    }
                }
            }
        }
    }

    // GML :183-198 -- the capped pass, then the Blood-crown pass.
    let mut rng = phase_rng(run.gen_seed, RNG_ENEMIES);
    let spawndist = if area == 5 && is_last { 150.0 } else { 120.0 };
    for &(cx, cy) in floors {
        if (plan.enemies.len() as f32) < enemy_cap
            && cell_dist2_origin(cx, cy) > spawndist * spawndist
            && !prop_tiles.contains(&(cx, cy))
        {
            let mut er = phase_rng(
                run.gen_seed.wrapping_add(enemy_calls.wrapping_mul(0x9E37_79B9)),
                RNG_ENEMY_CALL,
            );
            enemy_calls += 1;
            let props_before = plan.props.len();
            let enemies_before = plan.enemies.len();
            scr_pop_enemies(
                run,
                area,
                styleb_of(cx, cy),
                is_last,
                &mut plan.enemies,
                &mut plan.props,
                &mut prop_tiles,
                &mut plan.idpd_portals,
                &mut er,
                (cx, cy),
                &plan.small_walls,
                walls,
            );
            record_population_delta(
                &mut events,
                props_before,
                &plan.props,
                enemies_before,
                &plan.enemies,
            );
        }
        if run.blood_crown
            && rng.random::<f32>() * (8.0 + hard) < hard
            && cell_dist2_origin(cx, cy) > spawndist * spawndist
            && !prop_tiles.contains(&(cx, cy))
        {
            let mut er = phase_rng(
                run.gen_seed.wrapping_add(enemy_calls.wrapping_mul(0x9E37_79B9)),
                RNG_ENEMY_CALL,
            );
            enemy_calls += 1;
            let props_before = plan.props.len();
            let enemies_before = plan.enemies.len();
            scr_pop_enemies(
                run,
                area,
                styleb_of(cx, cy),
                is_last,
                &mut plan.enemies,
                &mut plan.props,
                &mut prop_tiles,
                &mut plan.idpd_portals,
                &mut er,
                (cx, cy),
                &plan.small_walls,
                walls,
            );
            record_population_delta(
                &mut events,
                props_before,
                &plan.props,
                enemies_before,
                &plan.enemies,
            );
        }
    }

    // GML :201-203 -- `with (Floor) scrPopProps()`. The small-wall pass shares
    // the function, so it interleaves with the per-area prop chain instead of
    // running as a separate sweep.
    let chest_tiles: HashSet<(i32, i32)> = plan
        .chests
        .iter()
        .map(|c| {
            let pos = c.pos();
            (
                ((pos.x - TILE * 0.5) / TILE).floor() as i32,
                ((pos.y - TILE * 0.5) / TILE).floor() as i32,
            )
        })
        .collect();
    let mut enemy_tiles: HashSet<(i32, i32)> = HashSet::new();
    for &(kind, pos) in &plan.enemies {
        let r = crate::enemy_data::enemy_def(kind).radius;
        let min = (
            ((pos.x - r) / TILE).floor() as i32,
            ((pos.y - r) / TILE).floor() as i32,
        );
        let max = (
            ((pos.x + r) / TILE).floor() as i32,
            ((pos.y + r) / TILE).floor() as i32,
        );
        for ex in min.0..=max.0 {
            for ey in min.1..=max.1 {
                if enemy_body_on_rect(pos, r, ex as f32 * TILE, ey as f32 * TILE) {
                    enemy_tiles.insert((ex, ey));
                }
            }
        }
    }
    let mut rng = phase_rng(run.gen_seed, RNG_PROPS);
    for &(cx, cy) in floors {
        if prop_tiles.contains(&(cx, cy)) || enemy_tiles.contains(&(cx, cy)) {
            continue;
        }
        let (ox, oy) = (cx as f32 * TILE, cy as f32 * TILE);
        let (px, py) = cell_center_i(cx, cy);
        let d2 = cell_dist2_center(cx, cy);
        let styleb = styleb_of(cx, cy);

        // GML :12-32. `random(5) < 1` is the FIRST operand, so every floor of
        // every area pays the draw before any area test.
        let walls_ok = rng.random::<f32>() * 5.0 < 1.0
            && d2 > 100.0 * 100.0
            && !(area == 106 || area == 100 || (area == 0 && run.loop_count == 0) || area == 107 || area == 6)
            && (area != 102 || rng.random::<f32>() * 3.0 < 1.0)
            && !(area == 3 && is_last)
            && !(area == 7 && is_last)
            && (area != 5 || rng.random::<f32>() * 3.0 < 1.0)
            && area != 102
            && !(area == 2 && styleb);
        if walls_ok {
            // GML :17-18 -- two draws spanning the 32px bbox, each snapped
            // down to a 16px multiple.
            let ux = rng.random_range(0.0..31.0);
            let uy = rng.random_range(0.0..31.0);
            let sx = (ux as i32).div_euclid(16) * 16;
            let sy = (uy as i32).div_euclid(16) * 16;
            if !wall_point(plan, walls, ox + sx as f32, oy + sy as f32)
                && !plan.enemies.iter().any(|&(kind, pos)| {
                    enemy_body_on_rect(
                        pos,
                        crate::enemy_data::enemy_def(kind).radius,
                        ox + sx as f32,
                        oy + sy as f32,
                    )
                })
            {
                plan.small_walls
                    .push(((cx * 2 + sx / 16) as i16, (cy * 2 + sy / 16) as i16));
                prop_tiles.insert((cx, cy));
                // GML :24-28 -- the scrapyards trap, on the tile's own corner.
                if area == 3
                    && rng.random::<f32>() * 4.0 < 1.0
                    && d2 > 64.0 * 64.0
                    && sx == 0
                    && sy == 0
                    && !chest_tiles.contains(&(cx, cy))
                {
                    let pos = Vec2::new(ox, oy);
                    plan.props.push((PropKind::Trap, pos));
                    events.push(PopulationEvent::Prop {
                        kind: PropKind::Trap,
                        pos,
                    });
                }
            }
            continue;
        }

        // GML :34-38.
        let unlikeliness: i32 = if area == 105 {
            2
        } else if area == 0 {
            7
        } else {
            10
        };
        if rng.random_range(0i32..unlikeliness) > 1 {
            continue;
        }

        // GML :40-135, one `else if` chain on `spawnarea`.
        let mut made: Vec<PropKind> = Vec::new();
        if area == 1 {
            if rng.random::<f32>() * 60.0 < 1.0 {
                made.push(PropKind::BigSkull);
            } else if styleb && rng.random::<f32>() * 5.0 < 1.0 {
                made.push(PropKind::BonePile);
            } else {
                made.push(rng_choose(
                    &mut rng,
                    &[
                        PropKind::Cactus,
                        PropKind::Cactus,
                        PropKind::GroundDecal,
                        PropKind::Cactus,
                    ],
                ));
            }
        } else if area == 106 {
            // GML :51-61. The `TopPot` triple has no `PropKind`, so its roll
            // is kept and nothing is emitted.
            if !styleb && !is_last {
                made.push(PropKind::PlantPot);
            }
            if is_last {
                let _ = rng.random::<f32>() * 6.0 < 1.0;
            }
        } else if area == 2 && d2 > 96.0 * 96.0 {
            made.push(rng_choose(
                &mut rng,
                &[
                    PropKind::Pipe,
                    PropKind::Pipe,
                    PropKind::ToxicBarrel,
                    PropKind::Pipe,
                    PropKind::Pipe,
                    PropKind::ToxicBarrel,
                    PropKind::GroundDecal,
                ],
            ));
        } else if area == 0 {
            made.push(rng_choose(
                &mut rng,
                &[
                    PropKind::NightCactus,
                    PropKind::NightCactus,
                    PropKind::NightBonePile,
                    PropKind::GroundDecal,
                ],
            ));
        } else if area == 4 {
            if styleb && rng.random::<f32>() * 5.0 < 1.0 {
                made.push(PropKind::BonePile);
            } else {
                made.push(rng_choose(
                    &mut rng,
                    &[
                        PropKind::Crystal,
                        PropKind::Crystal,
                        PropKind::GroundDecal,
                        PropKind::Cocoon,
                    ],
                ));
            }
        } else if area == 104 {
            if styleb && rng.random::<f32>() * 5.0 < 1.0 {
                made.push(PropKind::BonePile);
            } else {
                made.push(PropKind::GroundDecal);
            }
        } else if area == 3 {
            made.push(rng_choose(
                &mut rng,
                &[
                    PropKind::Tires,
                    PropKind::Car,
                    PropKind::Tires,
                    PropKind::Car,
                    PropKind::Car,
                    PropKind::Tires,
                    PropKind::GroundDecal,
                ],
            ));
        } else if area == 5 && d2 > 32.0 * 32.0 {
            if rng.random::<f32>() * 35.0 < 1.0 {
                made.push(rng_choose(
                    &mut rng,
                    &[PropKind::Snowman, PropKind::SodaMachine],
                ));
            } else if rng.random::<f32>() * 3.0 < 1.0 {
                if rng.random::<f32>() * 2.0 < 1.0 {
                    if let Some(at) = nearest_wall(plan, walls, ox, oy) {
                        prop_tiles.insert((cx, cy));
                        let pos = at + Vec2::new(8.0, 8.0);
                        plan.props.push((PropKind::GroundDecal, pos));
                        events.push(PopulationEvent::Prop {
                            kind: PropKind::GroundDecal,
                            pos,
                        });
                        continue;
                    }
                } else {
                    made.push(PropKind::StreetLight);
                }
            } else if d2 > 128.0 * 128.0 {
                made.push(rng_choose(
                    &mut rng,
                    &[PropKind::Hydrant, PropKind::Car],
                ));
            }
        } else if area == 6 && rng.random::<f32>() * 4.0 < 1.0 {
            made.push(rng_choose(
                &mut rng,
                &[
                    PropKind::Tube,
                    PropKind::Tube,
                    PropKind::Tube,
                    PropKind::Tube,
                    PropKind::MutantTube,
                ],
            ));
        } else if area == 7 && !is_last {
            made.push(rng_choose(
                &mut rng,
                &[
                    PropKind::Pillar,
                    PropKind::SmallGenerator,
                    PropKind::GroundDecal,
                ],
            ));
        } else if area == 100 {
            made.push(PropKind::Torch);
        } else if area == 101 {
            if rng.random::<f32>() * 40.0 < 1.0 {
                made.push(PropKind::Anchor);
            } else if d2 > 96.0 * 96.0 {
                made.push(rng_choose(
                    &mut rng,
                    &[
                        PropKind::WaterPlant,
                        PropKind::WaterPlant,
                        PropKind::GroundDecal,
                        PropKind::GroundDecal,
                        PropKind::OasisBarrel,
                        PropKind::WaterMine,
                        PropKind::WaterMine,
                    ],
                ));
            }
        } else if area == 103 && d2 > 64.0 * 64.0 {
            made.push(rng_choose(
                &mut rng,
                &[
                    PropKind::MoneyPile,
                    PropKind::MoneyPile,
                    PropKind::MoneyPile,
                    PropKind::YVStatue,
                    PropKind::GoldBarrel,
                    PropKind::MoneyPile,
                ],
            ));
        } else if area == 102 {
            made.push(rng_choose(
                &mut rng,
                &[
                    PropKind::PizzaBox,
                    PropKind::PizzaBox,
                    PropKind::GroundDecal,
                ],
            ));
        } else if area == 105 {
            if rng.random::<f32>() * 30.0 < 1.0 {
                made.push(rng_choose(
                    &mut rng,
                    &[
                        PropKind::BigFlower,
                        PropKind::BigFlower,
                        PropKind::GroundDecal,
                    ],
                ));
            } else {
                made.push(PropKind::Bush);
            }
        }

        if made.is_empty() {
            continue;
        }
        // GML :98 -- the street light keeps its `orandom(4)` offsets; every
        // other prop lands on the tile centre.
        let pos = if made.contains(&PropKind::StreetLight) {
            Vec2::new(
                px + rng.random_range(-4.0..4.0),
                py + rng.random_range(-4.0..4.0),
            )
        } else {
            Vec2::new(px, py)
        };
        prop_tiles.insert((cx, cy));
        for kind in made {
            plan.props.push((kind, pos));
            events.push(PopulationEvent::Prop { kind, pos });
        }
    }

    // GML :241,259-269 -- pizza sewers hold no boss: every enemy is destroyed
    // and the furthest floor gets four turtles and a rat.
    if area == 102 {
        plan.enemies.clear();
        let mut rng = phase_rng(run.gen_seed, RNG_PIZZA);
        if let Some(&(fx, fy)) = plan
            .floor_cells
            .iter()
            .max_by(|a, b| {
                cell_dist2_origin(a.0, a.1)
                    .partial_cmp(&cell_dist2_origin(b.0, b.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            let (px, py) = cell_center_i(fx, fy);
            for _ in 0..4 {
                let dx = rng.random_range(-2.0..2.0);
                let dy = rng.random_range(-2.0..2.0);
                let pos = Vec2::new(px + dx, py + dy);
                plan.enemies.push((EnemyKind::Turtle, pos));
                events.push(PopulationEvent::Enemy {
                    kind: EnemyKind::Turtle,
                    pos,
                });
            }
            let pos = Vec2::new(px, py);
            plan.enemies.push((EnemyKind::Rat, pos));
            events.push(PopulationEvent::Enemy {
                kind: EnemyKind::Rat,
                pos,
            });
        }
    }

    // GML :319-376.
    plan.boss = boss_for_run(run, area, is_last);
    if let Some(kind) = plan.boss {
        if matches!(kind, EnemyKind::BigBandit | EnemyKind::BigBanditLoop) {
            plan.boss_count = big_bandit_count(run.loop_count);
        }
    }

    // GML `GenCont/Alarm_0:24-37` -- the `TutCont` level keeps no roaming
    // enemy, no chest and no boss; the scripted `WeaponChest` (GML
    // `TutCont/Alarm_0` on entering PickingUp) ships in the plan so the
    // walkthrough has a gun to pick up.
    if run.tutorial {
        plan.enemies.clear();
        plan.chests.clear();
        plan.boss = None;
        if let Some(&(fx, fy)) = plan.floor_cells.iter().max_by_key(|c| c.0.abs() + c.1.abs()) {
            plan.chests.push(ChestSpawn::Weapon(cell_center_px(fx, fy)));
        }
        events.retain(|event| !matches!(event, PopulationEvent::Enemy { .. }));
    }
    events.extend(plan.chests.iter().copied().map(PopulationEvent::Chest));
    plan.population_events = events;
}

/// GML `scrPopulate.gml:319-377` plus the secret-area bosses at :347-377.
fn boss_for_run(run: &Run, area: i32, is_last: bool) -> Option<EnemyKind> {
    // The between-loop portal room maps to GML area 1 for its tiles but has no
    // `scrPopulate` boss branch, and `WantBoss` is not placed there.
    if run.area == AreaId::Loop {
        return None;
    }
    // GML :319-321 -- `WantBoss` sits outside the `_has_boss` gate, so the
    // Big Bandit guards every desert subarea.
    if area == 1 && !run.tutorial {
        return Some(big_bandit_kind(run.loop_count));
    }
    if !is_last {
        return None;
    }
    if is_secret_area(run.area) {
        return match area {
            2 if run.loop_count > 0 => Some(EnemyKind::FrogQueen),
            4 | 104 if run.loop_count > 0 => Some(EnemyKind::Hyper),
            6 if run.loop_count > 0 => Some(EnemyKind::Technomancer),
            _ => None,
        };
    }
    match area {
        3 => Some(big_dog_kind(run.loop_count)),
        5 => Some(lil_hunter_kind(run.loop_count)),
        7 => Some(EnemyKind::Throne),
        _ => None,
    }
}

fn big_bandit_kind(loop_count: u32) -> EnemyKind {
    if loop_count > 0 {
        EnemyKind::BigBanditLoop
    } else {
        EnemyKind::BigBandit
    }
}

fn big_dog_kind(loop_count: u32) -> EnemyKind {
    if loop_count > 0 {
        EnemyKind::BigDogLoop
    } else {
        EnemyKind::BigDog
    }
}

fn lil_hunter_kind(loop_count: u32) -> EnemyKind {
    if loop_count > 0 {
        EnemyKind::LilHunterLoop
    } else {
        EnemyKind::LilHunter
    }
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

/// Re-export of [`crate::comps_a::floor_cell_for_wall`], which now backs both
/// the 32x32 `FloorMask` and the 16x16 destroyed-wall layer.
pub use crate::comps_a::floor_cell_for_wall;

pub fn wall_cell_at(pos: Vec2) -> (i32, i32) {
    (
        (pos.x / WALL_PX).floor() as i32,
        (pos.y / WALL_PX).floor() as i32,
    )
}

fn pop_spawn<T: Copy>(rng: &mut StdRng, center: Vec2, table: &[T]) -> (T, Vec2) {
    let x = rng.random_range(-2.0..2.0);
    let y = rng.random_range(-2.0..2.0);
    let value = table[rng.random_range(0..table.len())];
    (value, center + Vec2::new(x, y))
}

fn spawn_pop_enemy(
    enemies: &mut Vec<(EnemyKind, Vec2)>,
    rng: &mut StdRng,
    center: Vec2,
    table: &[EnemyKind],
) {
    let (kind, pos) = pop_spawn(rng, center, table);
    enemies.push((kind, pos));
}

fn spawn_pop_prop(
    props: &mut Vec<(PropKind, Vec2)>,
    prop_tiles: &mut HashSet<(i32, i32)>,
    cell: (i32, i32),
    rng: &mut StdRng,
    center: Vec2,
    table: &[PropKind],
) {
    let (kind, pos) = pop_spawn(rng, center, table);
    props.push((kind, pos));
    prop_tiles.insert(cell);
}

fn record_population_delta(
    events: &mut Vec<PopulationEvent>,
    props_before: usize,
    props: &[(PropKind, Vec2)],
    enemies_before: usize,
    enemies: &[(EnemyKind, Vec2)],
) {
    for &(kind, pos) in props.iter().skip(props_before) {
        events.push(PopulationEvent::Prop { kind, pos });
    }
    for &(kind, pos) in enemies.iter().skip(enemies_before) {
        events.push(PopulationEvent::Enemy { kind, pos });
    }
}

fn scr_pop_enemies(
    run: &Run,
    area: i32,
    styleb: bool,
    is_last: bool,
    enemies: &mut Vec<(EnemyKind, Vec2)>,
    props: &mut Vec<(PropKind, Vec2)>,
    prop_tiles: &mut HashSet<(i32, i32)>,
    idpd_portals: &mut Vec<Vec2>,
    rng: &mut StdRng,
    cell: (i32, i32),
    small_walls: &[(i16, i16)],
    walls: &HashSet<(i32, i32)>,
) {
    let (cx, cy) = cell;
    let center = Vec2::from(cell_center_i(cx, cy));
    // GML `scrPopEnemies.gml:8` -- one origin-distance test and a single-point
    // `place_meeting(x, y, Wall)` at the Floor's own origin.
    if cell_dist2_origin(cx, cy) < 160.0 * 160.0
        || wall_point_in(walls, small_walls, cx as f32 * TILE, cy as f32 * TILE)
    {
        return;
    }
    // GML `scrPopEnemies:12`: `var _loop_rand = random(_loops)` -- a real in
    // [0, loops), drawn per enemy call; `random(0)` is 0, so `random(2) <
    // _loop_rand` never fires on the first playthrough.
    let loop_rand = if run.loop_count == 0 {
        0.0
    } else {
        rng.random::<f32>() * run.loop_count as f32
    };
    if run.area == AreaId::Campfire {
        return;
    }

    match area {
        1 => {
            if run.tutorial {
                return;
            }
            if rng.random::<f32>() * 2.0 < loop_rand {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
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
            } else if styleb {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::MaggotSpawn,
                        EnemyKind::BigMaggot,
                        EnemyKind::BigMaggot,
                        EnemyKind::Maggot,
                    ],
                );
            } else if rng.random::<f32>() * 7.0 < 1.0 {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[EnemyKind::MaggotSpawn, EnemyKind::Scorpion],
                );
            } else if rng.random::<f32>() * 30.0 < 1.0 {
                spawn_pop_prop(props, prop_tiles, cell, rng, center, &[PropKind::Barrel]);
                for _ in 0..3 {
                    spawn_pop_enemy(enemies, rng, center, &[EnemyKind::Bandit]);
                }
            } else {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::Bandit,
                        EnemyKind::Bandit,
                        EnemyKind::Bandit,
                        EnemyKind::Bandit,
                        EnemyKind::Bandit,
                        EnemyKind::Bandit,
                        EnemyKind::Maggot,
                        EnemyKind::Scorpion,
                    ],
                );
            }
        }
        2 => {
            if rng.random::<f32>() * 2.0 < loop_rand {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::Ratking,
                        EnemyKind::Ratking,
                        EnemyKind::BuffGator,
                        EnemyKind::LaserCrystal,
                        EnemyKind::Rat,
                        EnemyKind::Ballguy,
                        EnemyKind::Ballguy,
                        EnemyKind::SuperFireBaller,
                    ],
                );
            } else if styleb {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::Rat,
                        EnemyKind::Rat,
                        EnemyKind::Gator,
                        EnemyKind::Gator,
                        EnemyKind::Ballguy,
                    ],
                );
            } else if rng.random::<f32>() * 9.0 < 1.0 {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::Ballguy,
                        EnemyKind::Ratking,
                        EnemyKind::Ballguy,
                        EnemyKind::Ratking,
                        EnemyKind::Ballguy,
                        EnemyKind::Ratking,
                        EnemyKind::MeleeFake,
                    ],
                );
            } else {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::Rat,
                        EnemyKind::Rat,
                        EnemyKind::Rat,
                        EnemyKind::Rat,
                        EnemyKind::Rat,
                        EnemyKind::Rat,
                        EnemyKind::Rat,
                        EnemyKind::Bandit,
                    ],
                );
            }
        }
        3 => {
            if rng.random::<f32>() * 5.0 < 4.0 && (!is_last || rng.random::<f32>() * 2.0 < 1.0) {
                if rng.random::<f32>() * 2.0 < loop_rand {
                    spawn_pop_enemy(
                        enemies,
                        rng,
                        center,
                        &[
                            EnemyKind::Sniper,
                            EnemyKind::Sniper,
                            EnemyKind::MeleeFake,
                            EnemyKind::MeleeFake,
                            EnemyKind::Salamander,
                            EnemyKind::RobotGuard,
                            EnemyKind::Raven,
                            EnemyKind::BuffGator,
                            EnemyKind::Raven,
                        ],
                    );
                } else if styleb && rng.random::<f32>() * 3.0 < 1.0 {
                    spawn_pop_enemy(enemies, rng, center, &[EnemyKind::Salamander]);
                } else if rng.random::<f32>() * 4.0 < 1.0 {
                    spawn_pop_enemy(
                        enemies,
                        rng,
                        center,
                        &[
                            EnemyKind::MeleeBandit,
                            EnemyKind::Sniper,
                            EnemyKind::MeleeFake,
                            EnemyKind::Sniper,
                            EnemyKind::MeleeFake,
                            EnemyKind::Sniper,
                            EnemyKind::Sniper,
                            EnemyKind::Ballguy,
                        ],
                    );
                } else if rng.random::<f32>() * 10.0 < 1.0 {
                    if rng.random::<f32>() * 8.0 < 1.0 {
                        spawn_pop_prop(props, prop_tiles, cell, rng, center, &[PropKind::Car]);
                    }
                    spawn_pop_enemy(enemies, rng, center, &[EnemyKind::Raven, EnemyKind::Raven]);
                    spawn_pop_enemy(enemies, rng, center, &[EnemyKind::Raven, EnemyKind::Raven]);
                } else if rng.random::<f32>() * 20.0 < 1.0 {
                    spawn_pop_enemy(enemies, rng, center, &[EnemyKind::Salamander]);
                } else if rng.random::<f32>() * 4.0 < 3.0 {
                    spawn_pop_enemy(
                        enemies,
                        rng,
                        center,
                        &[EnemyKind::Raven, EnemyKind::Raven, EnemyKind::Bandit],
                    );
                }
            }
        }
        4 => {
            if rng.random::<f32>() * 2.0 < loop_rand {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::LaserCrystal,
                        EnemyKind::LaserCrystal,
                        EnemyKind::LaserCrystal,
                        EnemyKind::RhinoFreak,
                        EnemyKind::LightningCrystal,
                        EnemyKind::BuffGator,
                        EnemyKind::ExploFreak,
                        EnemyKind::Spider,
                        EnemyKind::Spider,
                    ],
                );
            } else {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::Spider,
                        EnemyKind::Spider,
                        EnemyKind::Spider,
                        EnemyKind::Spider,
                        EnemyKind::LaserCrystal,
                    ],
                );
            }
        }
        5 => {
            if rng.random::<f32>() * 2.0 < loop_rand {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::SnowTank,
                        EnemyKind::SnowTank,
                        EnemyKind::DogGuardian,
                        EnemyKind::ExploGuardian,
                        EnemyKind::RobotGuard,
                        EnemyKind::RobotGuard,
                        EnemyKind::RobotGuard,
                        EnemyKind::Wolf,
                        EnemyKind::Necromancer,
                    ],
                );
            } else if rng.random::<f32>() * 3.0 < 2.0 {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::RobotGuard,
                        EnemyKind::RobotGuard,
                        EnemyKind::RobotGuard,
                        EnemyKind::SnowTank,
                        EnemyKind::Wolf,
                        EnemyKind::Wolf,
                    ],
                );
            }
        }
        6 => {
            if rng.random::<f32>() * 2.0 < loop_rand {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::Ratking,
                        EnemyKind::RhinoFreak,
                        EnemyKind::ExploFreak,
                        EnemyKind::Necromancer,
                        EnemyKind::RhinoFreak,
                        EnemyKind::LaserCrystal,
                        EnemyKind::Turret,
                    ],
                );
            } else if rng.random::<f32>() * 14.0 < 1.0 {
                for _ in 0..10 {
                    // GML `scrPopEnemies.gml:103-104`: sixteen entries, 13 of
                    // them `Freak`.
                    spawn_pop_enemy(
                        enemies,
                        rng,
                        center,
                        &[
                            EnemyKind::Freak,
                            EnemyKind::Freak,
                            EnemyKind::Freak,
                            EnemyKind::Freak,
                            EnemyKind::Freak,
                            EnemyKind::Freak,
                            EnemyKind::Freak,
                            EnemyKind::Freak,
                            EnemyKind::Freak,
                            EnemyKind::Freak,
                            EnemyKind::ExploFreak,
                            EnemyKind::ExploFreak,
                            EnemyKind::RhinoFreak,
                            EnemyKind::Freak,
                            EnemyKind::Freak,
                            EnemyKind::Freak,
                        ],
                    );
                }
            } else if rng.random::<f32>() * 8.0 < 1.0 {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::Necromancer,
                        EnemyKind::Necromancer,
                        EnemyKind::Necromancer,
                        EnemyKind::ExploFreak,
                        EnemyKind::RhinoFreak,
                        EnemyKind::Necromancer,
                        EnemyKind::Necromancer,
                        EnemyKind::Turret,
                        EnemyKind::Turret,
                        EnemyKind::Turret,
                        EnemyKind::Necromancer,
                    ],
                );
            }
        }
        7 => {
            // GML `scrPopEnemies:113`: `if (_is_last || random(2) > 1) break`.
            // `random(2)` is a real in [0, 2), so half the eligible floors
            // skip the palace roll and only the last subarea always skips;
            // the `||` short-circuits, so the last subarea pays no draw.
            if is_last || rng.random::<f32>() * 2.0 > 1.0 {
                return;
            }
            if rng.random::<f32>() * 2.0 < loop_rand {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::ExploGuardian,
                        EnemyKind::DogGuardian,
                        EnemyKind::Sniper,
                        EnemyKind::DogGuardian,
                        EnemyKind::ExploGuardian,
                        EnemyKind::ExploFreak,
                        EnemyKind::JungleBandit,
                        EnemyKind::JungleBandit,
                    ],
                );
            } else if rng.random::<f32>() * 4.0 < 1.0 {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::ExploGuardian,
                        EnemyKind::DogGuardian,
                        EnemyKind::Guardian,
                        EnemyKind::Guardian,
                        EnemyKind::Guardian,
                        EnemyKind::Guardian,
                    ],
                );
            } else if rng.random::<f32>() * 16.0 < 1.0 {
                // GML `scrPopEnemies.gml:120-121`: `__spawn(IDPDSpawn)` -- the
                // IDPD portal spawner, not a live grunt. Same `bbox_center +
                // orandom(2)` placement and single-entry `irandom(argument_count
                // - 1)` pick as every spawn; `setup.rs` raises it through
                // `crate::idpd::spawn_idpd_spawn`.
                idpd_portals.push(pop_spawn(rng, center, &[PropKind::None]).1);
            }
        }
        101 => {
            if rng.random::<f32>() * 4.0 < 1.0 {
                spawn_pop_enemy(enemies, rng, center, &[EnemyKind::Crab]);
            } else if rng.random::<f32>() * 3.0 < 1.0 {
                for _ in 0..3 {
                    spawn_pop_enemy(enemies, rng, center, &[EnemyKind::BoneFish]);
                }
            }
        }
        102 => {
            spawn_pop_enemy(enemies, rng, center, &[EnemyKind::Turtle]);
        }
        103 => {
            if rng.random::<f32>() * 5.0 < 1.0 {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::FireBaller,
                        EnemyKind::Jock,
                        EnemyKind::FireBaller,
                        EnemyKind::Jock,
                        EnemyKind::FireBaller,
                        EnemyKind::SuperFireBaller,
                    ],
                );
            } else if rng.random::<f32>() * 4.0 < 1.0 {
                if rng.random::<f32>() * 5.0 < 1.0 {
                    spawn_pop_prop(
                        props,
                        prop_tiles,
                        cell,
                        rng,
                        center,
                        &[PropKind::GoldBarrel],
                    );
                }
                for _ in 0..3 {
                    spawn_pop_enemy(
                        enemies,
                        rng,
                        center,
                        &[
                            EnemyKind::Molefish,
                            EnemyKind::Molefish,
                            EnemyKind::Molefish,
                            EnemyKind::Molefish,
                            EnemyKind::Molesarge,
                        ],
                    );
                }
            }
        }
        104 => {
            if rng.random::<f32>() * 5.0 < 4.0 {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::InvSpider,
                        EnemyKind::InvSpider,
                        EnemyKind::InvSpider,
                        EnemyKind::InvSpider,
                        EnemyKind::InvLaserCrystal,
                        EnemyKind::InvLaserCrystal,
                    ],
                );
            }
        }
        105 => {
            if rng.random::<f32>() * 8.0 < 1.0 {
                spawn_pop_enemy(enemies, rng, center, &[EnemyKind::JungleFly]);
            } else if rng.random::<f32>() * 30.0 < 1.0 {
                spawn_pop_prop(props, prop_tiles, cell, rng, center, &[PropKind::Barrel]);
                spawn_pop_enemy(enemies, rng, center, &[EnemyKind::JungleBandit]);
                spawn_pop_enemy(enemies, rng, center, &[EnemyKind::JungleBandit]);
                spawn_pop_enemy(enemies, rng, center, &[EnemyKind::JungleBandit]);
            } else {
                spawn_pop_enemy(
                    enemies,
                    rng,
                    center,
                    &[
                        EnemyKind::JungleBandit,
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
            }
        }
        106 => {
            if is_last {
                return;
            }
            if rng.random::<f32>() * 12.0 < 1.0 || enemies.is_empty() {
                if rng.random::<f32>() * 7.0 < 1.0 {
                    spawn_pop_enemy(
                        enemies,
                        rng,
                        center,
                        &[
                            EnemyKind::IdpdElite,
                            EnemyKind::EliteShielder,
                            EnemyKind::EliteInspector,
                        ],
                    );
                } else if rng.random::<f32>() * 4.0 < 1.0 {
                    for _ in 0..5 {
                        spawn_pop_enemy(enemies, rng, center, &[EnemyKind::IdpdGrunt]);
                    }
                } else if rng.random::<f32>() * 3.0 < 1.0 {
                    spawn_pop_enemy(
                        enemies,
                        rng,
                        center,
                        &[
                            EnemyKind::IdpdGrunt,
                            EnemyKind::IdpdShield,
                            EnemyKind::IdpdInspector,
                        ],
                    );
                }
            }
        }
        _ => {}
    }
}

#[derive(Default)]
pub struct LoopClusterOut {
    pub portal_clears: Vec<Vec2>,
}

fn cluster_kind(kind: EnemyKind) -> EnemyKind {
    match kind {
        EnemyKind::GoldScorpion => EnemyKind::Scorpion,
        EnemyKind::GoldSnowtank => EnemyKind::SnowTank,
        EnemyKind::LightningCrystal => EnemyKind::LaserCrystal,
        EnemyKind::BuffGator => EnemyKind::Gator,
        _ => kind,
    }
}

fn cluster_source_skips(kind: EnemyKind, loops: u32, rng: &mut StdRng) -> bool {
    // GML `scrPopulate:288-294`: inside `if (_loops > 0 && ...)`,
    // `if (random(60) > _loops || ...) continue`. `random(60)` is a real in
    // [0, 60), so `loops / 60` of the sources cluster. The GML text names all
    // four mimics even though they are `chestprop` children rather than
    // `enemy` instances, so the port keeps the list verbatim.
    rng.random::<f32>() * 60.0 > loops as f32
        || matches!(
            kind,
            EnemyKind::Mimic
                | EnemyKind::SuperMimic
                | EnemyKind::WepMimic
                | EnemyKind::MaggotSpawn
        )
}

pub fn apply_loop_population_clusters(
    events: &mut Vec<PopulationEvent>,
    loops: u32,
    area: AreaId,
    rng: &mut StdRng,
) -> LoopClusterOut {
    let mut out = LoopClusterOut::default();
    if loops == 0 || area == AreaId::PizzaSewers {
        return out;
    }

    let sources = events.clone();
    let mut next = Vec::with_capacity(events.len());
    for event in sources {
        let PopulationEvent::Enemy {
            kind: source_kind,
            pos: source_pos,
        } = event
        else {
            next.push(event);
            continue;
        };
        next.push(event);
        if cluster_source_skips(source_kind, loops, rng) {
            continue;
        }
        let kind = cluster_kind(source_kind);
        for _ in 0..loops.saturating_add(3) {
            let x = rng.random_range(-4.0..4.0);
            let y = rng.random_range(-4.0..4.0);
            next.push(PopulationEvent::Enemy {
                kind,
                pos: source_pos + Vec2::new(x, y),
            });
        }
        // GML `scrPopulate.gml:307`: `distance_to_point(10016, 10016) < 128` on
        // the enemy instance, so the half-tile offset applies here too.
        if source_pos.distance_squared(Vec2::splat(TILE * 0.5)) < 128.0 * 128.0 {
            out.portal_clears.push(source_pos);
            next.push(PopulationEvent::PortalClear {
                pos: source_pos,
                scale: 0.6,
            });
        }
    }
    *events = next;
    out
}

pub fn apply_loop_enemy_clusters(
    enemies: &mut Vec<(EnemyKind, Vec2)>,
    loops: u32,
    area: AreaId,
    rng: &mut StdRng,
) -> LoopClusterOut {
    let mut out = LoopClusterOut::default();
    if loops == 0 || area == AreaId::PizzaSewers {
        return out;
    }

    let sources = enemies.clone();
    for (source_kind, source_pos) in sources {
        if cluster_source_skips(source_kind, loops, rng) {
            continue;
        }
        let kind = cluster_kind(source_kind);
        for _ in 0..loops.saturating_add(3) {
            let x = rng.random_range(-4.0..4.0);
            let y = rng.random_range(-4.0..4.0);
            enemies.push((kind, source_pos + Vec2::new(x, y)));
        }
        // GML `scrPopulate.gml:307`: `distance_to_point(10016, 10016) < 128` on
        // the enemy instance, so the half-tile offset applies here too.
        if source_pos.distance_squared(Vec2::splat(TILE * 0.5)) < 128.0 * 128.0 {
            out.portal_clears.push(source_pos);
        }
    }
    out
}

/// GML `objects/FloorMaker/Step_0.gml:27-49` -- `plan` already holds the
/// arena floors and the ten statues; only the four inactive generators and
/// the boss flag are added here. `NothingInactive` has no `PropKind`.
fn populate_throne_room(plan: &mut LevelPlan) {
    plan.enemies.clear();
    plan.boss = Some(EnemyKind::Throne);
    plan.boss_count = 1;
}

/// GML `objects/WantBoss/Create_0.gml:3`: `number = max(GameCont.loops * 2, 1)`.
pub fn big_bandit_count(loop_count: u32) -> u32 {
    if loop_count == 0 {
        1
    } else {
        loop_count.saturating_mul(2).max(1)
    }
}
