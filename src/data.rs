//! Game data ids and tables. Byte-exact port of the pure-data enums
//! in nt's `content.rs` / `areas.rs` (`EnemyKind`, `MutationId`,
//! `UltraMutationId` live in `ids_part.rs`, verified identical).
//! Bevy-colored fields become portable types (`Color` -> `[f32; 4]`,
//! `Vec2` -> glam); values are untouched.

pub use super::ids_part::{EnemyKind, MutationId, UltraMutationId};

use glam::Vec2;
use serde::{Deserialize, Serialize};

/// Area ids on the 15-floor route. Discriminants are route identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AreaId {
    Desert = 0,
    Sewers = 1,
    PizzaSewers = 2,
    Scrapyards = 3,
    CursedCaves = 4,
    CrystalCaves = 5,
    FrozenCity = 6,
    City = 7,
    Jungle = 8,
    Labs = 9,
    Palace = 10,
    HQ = 11,
    Oasis = 12,
    Vault = 13,
    CrownVault = 14,
    Campfire = 15,
    Loop = 16,
}

impl Default for AreaId {
    fn default() -> Self {
        Self::Desert
    }
}

impl AreaId {
    pub fn from_route_floor(floor: u32) -> AreaId {
        let loop_count = (floor.max(1) - 1) / 15;
        area_for_floor(floor, loop_count)
    }
}

pub fn area_for_floor(floor: u32, _loop_count: u32) -> AreaId {
    let route_floor = (floor.max(1) - 1) % 15 + 1;
    match route_floor {
        1..=3 => AreaId::Desert,
        4 => AreaId::Sewers,
        5..=7 => AreaId::Scrapyards,
        8 => AreaId::CrystalCaves,
        9..=11 => AreaId::FrozenCity,
        12 => AreaId::Labs,
        13..=15 => AreaId::Palace,
        _ => unreachable!(),
    }
}

pub fn route_coordinates(floor: u32) -> (u32, u32) {
    let route_floor = (floor.max(1) - 1) % 15 + 1;
    match route_floor {
        1..=3 => (1, route_floor),
        4 => (2, 1),
        5..=7 => (3, route_floor - 4),
        8 => (4, 1),
        9..=11 => (5, route_floor - 8),
        12 => (6, 1),
        13..=15 => (7, route_floor - 12),
        _ => unreachable!(),
    }
}

/// Ammo kinds. Discriminants are save/ammo identity.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum AmmoKind {
    None = 0,
    Bullets = 1,
    Shells = 2,
    Bolts = 3,
    Explosives = 4,
    Energy = 5,
}

/// Weapon-data ammo typing (mirrors `weapons_data::AmmoType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AmmoType {
    None = 0,
    Bullets = 1,
    Shells = 2,
    Bolts = 3,
    Explosives = 4,
    Energy = 5,
}

impl AmmoType {
    pub const fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::None,
            1 => Self::Bullets,
            2 => Self::Shells,
            3 => Self::Bolts,
            4 => Self::Explosives,
            5 => Self::Energy,
            _ => Self::None,
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Bullets => "Bullets",
            Self::Shells => "Shells",
            Self::Bolts => "Bolts",
            Self::Explosives => "Explosives",
            Self::Energy => "Energy",
        }
    }
}

/// Playable races. Discriminants are save identity.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum RaceId {
    Random = 0,
    Fish = 1,
    Crystal = 2,
    Eyes = 3,
    Melting = 4,
    Plant = 5,
    Venuz = 6,
    Steroids = 7,
    Robot = 8,
    Chicken = 9,
    Rebel = 10,
    Horror = 11,
    Rogue = 12,
    BigDog = 13,
    Skeleton = 14,
    Frog = 15,
    Cuz = 16,
}

pub const PLAYABLE_RACES: [RaceId; 16] = [
    RaceId::Fish,
    RaceId::Crystal,
    RaceId::Eyes,
    RaceId::Melting,
    RaceId::Plant,
    RaceId::Venuz,
    RaceId::Steroids,
    RaceId::Robot,
    RaceId::Chicken,
    RaceId::Rebel,
    RaceId::Horror,
    RaceId::Rogue,
    RaceId::BigDog,
    RaceId::Skeleton,
    RaceId::Frog,
    RaceId::Cuz,
];

pub type CharacterId = RaceId;

pub const CHARACTERS: [CharacterId; PLAYABLE_RACES.len()] = PLAYABLE_RACES;

/// Melee-capable weapon kinds (data subset; full defs land with weapons).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeaponKind {
    None,
    Revolver,
    Machinegun,
    Smg,
    AssaultRifle,
    Shotgun,
    Crossbow,
    GrenadeLauncher,
    Wrench,
    Sledgehammer,
}

/// Weapon id (content index, not dense).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct WeaponId(pub u8);

impl WeaponId {
    pub const NONE: Self = Self(0);
    pub const REVOLVER: Self = Self(1);
    pub const WRENCH: Self = Self(3);
    pub const MACHINEGUN: Self = Self(4);
    pub const SHOTGUN: Self = Self(5);
    pub const CROSSBOW: Self = Self(6);
    pub const GRENADE_LAUNCHER: Self = Self(7);
    pub const SMG: Self = Self(16);
    pub const ASSAULT_RIFLE: Self = Self(17);
    pub const SLEDGEHAMMER: Self = Self(88);
}

impl Default for WeaponId {
    fn default() -> Self {
        Self(0)
    }
}

pub const WEAPON_NONE: WeaponId = WeaponId(0);
pub const WEAPON_REVOLVER: WeaponId = WeaponId(1);

pub fn resolve_start_weapon(raw: WeaponId) -> WeaponId {
    if raw == WEAPON_NONE {
        WEAPON_REVOLVER
    } else {
        raw
    }
}

impl From<WeaponKind> for WeaponId {
    fn from(k: WeaponKind) -> Self {
        match k {
            WeaponKind::None => Self(0),
            WeaponKind::Revolver => Self(1),
            WeaponKind::Wrench => Self(3),
            WeaponKind::Machinegun => Self(4),
            WeaponKind::Shotgun => Self(5),
            WeaponKind::Crossbow => Self(6),
            WeaponKind::GrenadeLauncher => Self(7),
            WeaponKind::Smg => Self(16),
            WeaponKind::AssaultRifle => Self(17),
            WeaponKind::Sledgehammer => Self(88),
        }
    }
}

impl From<WeaponId> for WeaponKind {
    fn from(id: WeaponId) -> Self {
        match id.0 {
            1 => WeaponKind::Revolver,
            3 => WeaponKind::Wrench,
            4 => WeaponKind::Machinegun,
            5 => WeaponKind::Shotgun,
            6 => WeaponKind::Crossbow,
            7 => WeaponKind::GrenadeLauncher,
            16 => WeaponKind::Smg,
            17 => WeaponKind::AssaultRifle,
            88 => WeaponKind::Sledgehammer,
            _ => WeaponKind::None,
        }
    }
}

/// Hazard element kinds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HazardKind {
    Fire,
    Toxic,
}

/// Area-hazard definition. `color` is linear `[r, g, b, a]`
/// (bevy `Color` in the original).
#[derive(Clone, Copy, Debug)]
pub struct HazardDef {
    pub kind: HazardKind,
    pub radius: f32,
    pub damage: i32,
    pub duration: f32,
    pub tick: f32,
    pub color: [f32; 4],
}

/// Death-split definition (see `HazardDef.color` for the swap).
#[derive(Clone, Copy, Debug)]
pub struct SplitDef {
    pub pellets: u8,
    pub spread: f32,
    pub speed: f32,
    pub damage: i32,
    pub lifetime: f32,
    pub radius: f32,
    pub knockback: f32,
    pub color: [f32; 4],
    pub size: Vec2,
}

pub fn ammo_max(kind: AmmoKind) -> i32 {
    match kind {
        AmmoKind::None => 0,
        AmmoKind::Bullets => 255,
        AmmoKind::Shells | AmmoKind::Bolts | AmmoKind::Explosives | AmmoKind::Energy => 55,
    }
}

pub fn ammo_pickup_amount(kind: AmmoKind) -> i32 {
    match kind {
        AmmoKind::None => 0,
        AmmoKind::Bullets => 32,
        AmmoKind::Shells => 8,
        AmmoKind::Bolts => 7,
        AmmoKind::Explosives => 6,
        AmmoKind::Energy => 10,
    }
}

/// Crown kinds. Discriminants are save identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CrownKind {
    None = 0,
    Death = 1,
    Life = 2,
    Haste = 3,
    Guns = 4,
    Hatred = 5,
    Blood = 6,
    Destiny = 7,
    Love = 8,
    Risk = 9,
    Curses = 10,
    Luck = 11,
    Protection = 12,
}

impl CrownKind {
    pub const ALL: [CrownKind; 13] = [
        CrownKind::None,
        CrownKind::Death,
        CrownKind::Life,
        CrownKind::Haste,
        CrownKind::Guns,
        CrownKind::Hatred,
        CrownKind::Blood,
        CrownKind::Destiny,
        CrownKind::Love,
        CrownKind::Risk,
        CrownKind::Curses,
        CrownKind::Luck,
        CrownKind::Protection,
    ];

    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => CrownKind::Death,
            2 => CrownKind::Life,
            3 => CrownKind::Haste,
            4 => CrownKind::Guns,
            5 => CrownKind::Hatred,
            6 => CrownKind::Blood,
            7 => CrownKind::Destiny,
            8 => CrownKind::Love,
            9 => CrownKind::Risk,
            10 => CrownKind::Curses,
            11 => CrownKind::Luck,
            12 => CrownKind::Protection,
            _ => CrownKind::None,
        }
    }
}

/// Skin letter picks (A-D).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SkinLetter {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
}

impl SkinLetter {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::A),
            1 => Some(Self::B),
            2 => Some(Self::C),
            3 => Some(Self::D),
            _ => None,
        }
    }
}

/// Character ability kinds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AbilityKind {
    Flip,
    Shield,
    Telekinesis,
    Detonate,
    Snare,
    PopPop,
    GetLoaded,
    EatWeapon,
    Throw,
    SpawnAlly,
    HorrorBeam,
    PortalStrike,
    RocketBarrage,
    BloodGamble,
    ToxicPuke,
    CuzSwap,
}

/// Secret-area targets (area + HUD coordinates).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SecretTarget {
    Oasis,
    PizzaSewers,
    YvMansion,
    CursedCaves,
    Jungle,
    Vault,
    CrownVault,
    Hq,
}

impl SecretTarget {
    pub fn area(self) -> AreaId {
        match self {
            SecretTarget::Oasis => AreaId::Oasis,
            SecretTarget::PizzaSewers => AreaId::PizzaSewers,
            SecretTarget::YvMansion => AreaId::City,
            SecretTarget::CursedCaves => AreaId::CursedCaves,
            SecretTarget::Jungle => AreaId::Jungle,
            SecretTarget::Vault => AreaId::Vault,
            SecretTarget::CrownVault => AreaId::CrownVault,
            SecretTarget::Hq => AreaId::HQ,
        }
    }

    pub fn display(self) -> (u32, u32) {
        match self {
            SecretTarget::Oasis => (1, 5),
            SecretTarget::PizzaSewers => (2, 5),
            SecretTarget::YvMansion => (3, 5),
            SecretTarget::CursedCaves => (4, 5),
            SecretTarget::Jungle => (5, 5),
            SecretTarget::Vault => (0, 1),
            SecretTarget::CrownVault => (0, 2),
            SecretTarget::Hq => (0, 3),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SecretTarget::Oasis => "OASIS",
            SecretTarget::PizzaSewers => "PIZZA SEWERS",
            SecretTarget::YvMansion => "Y.V. MANSION",
            SecretTarget::CursedCaves => "CURSED CAVES",
            SecretTarget::Jungle => "JUNGLE",
            SecretTarget::Vault => "VAULT",
            SecretTarget::CrownVault => "CROWN VAULT",
            SecretTarget::Hq => "I.D.P.D. HQ",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Oracle tests ported from nt's `areas.rs` (route identity).
    #[test]
    fn normal_route_matches_world_order() {
        assert_eq!(area_for_floor(1, 0), AreaId::Desert);
        assert_eq!(area_for_floor(3, 0), AreaId::Desert);
        assert_eq!(area_for_floor(4, 0), AreaId::Sewers);
        assert_eq!(area_for_floor(5, 0), AreaId::Scrapyards);
        assert_eq!(area_for_floor(8, 0), AreaId::CrystalCaves);
        assert_eq!(area_for_floor(9, 0), AreaId::FrozenCity);
        assert_eq!(area_for_floor(12, 0), AreaId::Labs);
        assert_eq!(area_for_floor(13, 0), AreaId::Palace);
        assert_eq!(area_for_floor(15, 0), AreaId::Palace);
        assert_eq!(area_for_floor(16, 1), AreaId::Desert);
    }

    #[test]
    fn route_repeats_after_throne() {
        assert_eq!(route_coordinates(4), (2, 1));
        assert_eq!(route_coordinates(8), (4, 1));
        assert_eq!(route_coordinates(12), (6, 1));
        assert_eq!(route_coordinates(13), (7, 1));
        assert_eq!(route_coordinates(16), (1, 1));
        assert_eq!(route_coordinates(31), (1, 1));
    }

    #[test]
    fn weapon_roundtrip_and_defaults() {
        assert_eq!(WeaponId::default(), WeaponId::NONE);
        assert_eq!(WeaponKind::from(WeaponId::SHOTGUN), WeaponKind::Shotgun);
        assert_eq!(WeaponId::from(WeaponKind::Shotgun), WeaponId::SHOTGUN);
        assert_eq!(WeaponKind::from(WeaponId(99)), WeaponKind::None);
        assert_eq!(resolve_start_weapon(WEAPON_NONE), WEAPON_REVOLVER);
        assert_eq!(ammo_max(AmmoKind::Bullets), 255);
        assert_eq!(ammo_pickup_amount(AmmoKind::Energy), 10);
        assert_eq!(AmmoType::from_u8(9), AmmoType::None);
    }

    #[test]
    fn playable_roster_shape() {
        assert_eq!(PLAYABLE_RACES.len(), 16);
        assert_eq!(CHARACTERS[0], RaceId::Fish);
        assert_eq!(RaceId::Fish as u8, 1);
        assert_eq!(AreaId::default(), AreaId::Desert);
    }
}
