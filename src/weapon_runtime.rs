//! Weapon runtime definitions: pure-data port of
//! `nt-recreated-bevy/src/game/weapon_runtime.rs`, plus the `WeaponDef` /
//! `MeleeDef` structs and the legacy `weapon_def` table from bevy
//! `content.rs` (~lines 832-1223).
//!
//! Sim-side conventions:
//! - bevy `Color::srgb(r, g, b)` -> `[f32; 3]` RGB array (the renderer
//!   resolves presentation later); values are untouched.
//! - `HazardDef` / `SplitDef` are reused from `crate::data` (their color
//!   is `[f32; 4]`); bevy `Color::srgba(r, g, b, a)` maps component-wise.
//! - bevy `Vec2 size` -> `glam::Vec2`.
//! - `WeaponId` is `crate::data::WeaponId(pub u8)`; `.0` indexes `WEAPONS`.
//! - Pure data/logic only: zero ECS imports, zero bevy imports.
//!
//! `weapon_runtime_def` is the key API the firing phase will call:
//! `WeaponId -> WeaponDef` with all profile layers applied in the same
//! order as bevy (legacy-or-family base -> exact -> variant -> normalize).

use glam::Vec2;

use crate::data::{AmmoKind, HazardDef, HazardKind, SplitDef, WeaponId, WeaponKind};
use crate::weapons_data::{AmmoType, WEAPONS, WeaponData};

/// Melee arc definition (bevy `content.rs::MeleeDef` verbatim).
#[derive(Clone, Copy, Debug)]
pub struct MeleeDef {
    pub range: f32,
    pub arc: f32,
}

/// Full per-weapon firing definition (bevy `content.rs::WeaponDef`
/// verbatim, except `color: Color -> [f32; 3]` and `Vec2` -> `glam::Vec2`).
#[derive(Clone, Copy, Debug)]
pub struct WeaponDef {
    pub name: &'static str,
    pub ammo: AmmoKind,
    pub ammo_cost: i32,
    pub rad_cost: u32,
    pub cooldown: f32,
    pub damage: i32,
    pub pellets: usize,
    pub speed: f32,
    pub lifetime: f32,
    pub spread: f32,
    pub recoil: f32,
    pub shake: f32,
    pub projectile_radius: f32,
    pub knockback: f32,
    pub automatic: bool,
    pub explosive: bool,
    pub burst_shots: usize,
    pub burst_interval: f32,
    pub melee: Option<MeleeDef>,
    pub color: [f32; 3],
    pub size: Vec2,
    pub muzzle_burst: usize,
    pub bounces: u8,
    pub pierce: u8,
    pub hazard: Option<HazardDef>,
    pub split: Option<SplitDef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ProjectileKind {
    Bullet,
    Shell,
    Bolt,
    Explosive,
    Energy,
    Melee,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct ExplosionSpec {
    pub radius: f32,
    pub damage: i32,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct MeleeSpec {
    pub range: f32,
    pub arc: f32,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct WeaponRuntime {
    pub projectile_kind: ProjectileKind,
    pub pellets: u8,
    pub spread_deg: f32,
    pub speed: f32,
    pub lifetime_frames: u16,
    pub damage: i32,
    pub recoil: f32,
    pub explosion: Option<ExplosionSpec>,
    pub melee: Option<MeleeSpec>,
    pub cooldown_frames: u16,
    pub automatic: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeaponFamily {
    Empty,
    MeleeLight,
    MeleeHeavy,
    Pistol,
    Automatic,
    BurstRifle,
    Shotgun,
    Slugger,
    Crossbow,
    Splinter,
    Disc,
    Explosive,
    Flame,
    Laser,
    Plasma,
    Lightning,
    Toxic,
    Deployable,
    Novelty,
}

#[allow(dead_code)]
pub fn weapon_family(id: WeaponId) -> WeaponFamily {
    let id = sanitize_weapon_id(id);
    if id == WeaponId::NONE {
        return WeaponFamily::Empty;
    }

    family_for(weapon_meta(id))
}

fn family_for(meta: &WeaponData) -> WeaponFamily {
    let name = meta.wep_name;

    if name.is_empty() {
        return WeaponFamily::Empty;
    }

    if meta.wep_mele {
        if name.contains("HAMMER")
            || name.contains("SHOVEL")
            || name.contains("SLEDGE")
            || name.contains("GUITAR")
            || name.contains("BAT")
        {
            return WeaponFamily::MeleeHeavy;
        }

        return WeaponFamily::MeleeLight;
    }

    if name.contains("SENTRY") {
        return WeaponFamily::Deployable;
    }

    if name.contains("TOXIC") {
        return WeaponFamily::Toxic;
    }

    if name.contains("FLAME")
        || name.contains("FLARE")
        || name == "DRAGON"
        || name.contains("INCINERATOR")
    {
        return WeaponFamily::Flame;
    }

    if name.contains("LIGHTNING") {
        return WeaponFamily::Lightning;
    }

    if name.contains("PLASMA") || name.contains("DEVASTATOR") {
        return WeaponFamily::Plasma;
    }

    if name.contains("LASER") || name.contains("ION CANNON") {
        return WeaponFamily::Laser;
    }

    if name.contains("DISC GUN") {
        return WeaponFamily::Disc;
    }

    if name.contains("FLAK") {
        return WeaponFamily::Explosive;
    }

    if name.contains("SPLINTER") {
        return WeaponFamily::Splinter;
    }

    if name.contains("SLUGGER") {
        return WeaponFamily::Slugger;
    }

    if name.contains("CROSSBOW") || name.ends_with(" BOW") {
        return WeaponFamily::Crossbow;
    }

    if name.contains("SHOTGUN") || name == "WAVE GUN" {
        return WeaponFamily::Shotgun;
    }

    if name.contains("GRENADE")
        || name.contains("BAZOOKA")
        || name.contains("LAUNCHER")
        || name.contains("NUKE")
        || name.contains("JACKHAMMER")
        || name.contains("CLUSTER")
    {
        return WeaponFamily::Explosive;
    }

    if name.contains("RIFLE") {
        return WeaponFamily::BurstRifle;
    }

    if name.contains("MACHINEGUN") || name.contains("MINIGUN") || name.contains("SMG") {
        return WeaponFamily::Automatic;
    }

    if name.contains("PISTOL")
        || name.contains("REVOLVER")
        || name == "SMART GUN"
        || name == "POP GUN"
    {
        return WeaponFamily::Pistol;
    }

    match meta.wep_type {
        AmmoType::None => WeaponFamily::Novelty,
        AmmoType::Bullets => {
            if meta.wep_auto {
                WeaponFamily::Automatic
            } else {
                WeaponFamily::Pistol
            }
        }
        AmmoType::Shells => WeaponFamily::Shotgun,
        AmmoType::Bolts => WeaponFamily::Crossbow,
        AmmoType::Explosives => WeaponFamily::Explosive,
        AmmoType::Energy => {
            if meta.wep_auto {
                WeaponFamily::Laser
            } else {
                WeaponFamily::Plasma
            }
        }
    }
}

pub fn weapon_sleep_secs(id: WeaponId) -> f32 {
    let fam = weapon_family(id);
    let meta = weapon_meta(sanitize_weapon_id(id));
    let name = meta.wep_name;

    let frames: f32 = match base_weapon_name(name) {
        "REVOLVER" | "PISTOL" => 2.0,
        "MACHINEGUN" | "SMG" => 1.0,
        "ASSAULT RIFLE" => 3.0,
        "SHOTGUN" | "DOUBLE SHOTGUN" | "SAWED-OFF SHOTGUN" => 5.0,
        "SLUGGER" | "HEAVY SLUGGER" => 6.0,
        "CROSSBOW" | "HEAVY CROSSBOW" => 4.0,
        "GRENADE LAUNCHER" | "BAZOOKA" | "NUKE LAUNCHER" => 8.0,
        "SUPER PLASMA CANNON" | "PLASMA CANNON" => 10.0,
        "LIGHTNING HAMMER" | "HAMMER" | "SLEDGEHAMMER" => 8.0,
        "SCREWDRIVER" | "WRENCH" | "GUITAR" => 4.0,
        "LASER RIFLE" | "LASER PISTOL" => 2.0,
        "ENERGY SWORD" | "ENERGY SCREWDRIVER" => 5.0,
        _ => match fam {
            WeaponFamily::Empty => 0.0,
            WeaponFamily::Automatic | WeaponFamily::Pistol => 1.5,
            WeaponFamily::BurstRifle => 3.0,
            WeaponFamily::Shotgun => 5.0,
            WeaponFamily::Slugger | WeaponFamily::Crossbow => 5.0,
            WeaponFamily::Explosive => 7.0,
            WeaponFamily::Plasma | WeaponFamily::Laser => 4.0,
            WeaponFamily::Lightning | WeaponFamily::Flame => 3.0,
            WeaponFamily::MeleeLight => 3.0,
            WeaponFamily::MeleeHeavy => 7.0,
            WeaponFamily::Disc | WeaponFamily::Splinter => 4.0,
            WeaponFamily::Toxic | WeaponFamily::Deployable | WeaponFamily::Novelty => 2.0,
        },
    };

    let mult = if name.starts_with("ULTRA ") {
        1.25
    } else if name.starts_with("CURSED ") {
        1.1
    } else {
        1.0
    };

    (frames * mult / 30.0).clamp(0.0, 0.45)
}

pub fn base_weapon_name(name: &str) -> &str {
    let stripped = name
        .strip_prefix("ULTRA ")
        .or_else(|| name.strip_prefix("CURSED "))
        .or_else(|| name.strip_prefix("GOLDEN "))
        .unwrap_or(name);
    stripped
}

/// GML `scr_weapon_post` 5th arg (`_knockback`, px/frame) per weapon,
/// converted to px/s (×30). Positive = shoved backwards
/// (`motion_add(_gunangle + 180, kb)`), negative = lunged forwards (melee).
/// Omitted 5th arg defaults to 0 — most guns (incl. grenade launcher,
/// bazooka, sticky) have NO self-push.
pub fn gml_fire_push_px_s(full_name: &str) -> f32 {
    let key = if full_name.starts_with("ULTRA ") {
        full_name
    } else {
        base_weapon_name(full_name)
    };
    let kb_frames: f32 = match key {
        "DOUBLE SHOTGUN" | "SAWED-OFF SHOTGUN" => 2.0,
        "MINIGUN" => 0.6,
        "DOUBLE MINIGUN" => 0.7,
        "SUPER CROSSBOW" => 1.0,
        "SUPER BAZOOKA" => 1.0,
        "SUPER SLUGGER" => 3.0,
        "ASSAULT SLUGGER" => -7.0,
        "HYPER RIFLE" => -4.0,
        "WAVE GUN" | "PLASMA GUN" | "PLASMA RIFLE" => 3.0,
        "PLASMA CANNON" => 6.0,
        "SUPER PLASMA CANNON" => 16.0,
        "PLASMA MINIGUN" => 2.0,
        "LIGHTNING CANNON" => 6.0,
        "LASER PISTOL" | "LASER RIFLE" | "LASER MINIGUN" => 0.6,
        "DEVASTATOR" => 5.0,
        "ERASER" => 2.0,
        "WRENCH" | "SHOVEL" | "SLEDGEHAMMER" | "GUITAR" | "ELECTRIC GUITAR" | "BLOOD HAMMER" => {
            -6.0
        }
        "ENERGY SWORD" | "LIGHTNING HAMMER" | "ENERGY HAMMER" => -7.0,
        "SCREWDRIVER" | "CHICKEN SWORD" => -4.0,
        "ENERGY SCREWDRIVER" => -5.0,
        "ULTRA SHOVEL" => -8.0,
        "BLACK SWORD" => -8.0,
        _ => 0.0,
    };
    kb_frames * 30.0
}

/// GML `scr_weapon_post` 4th arg (`wkick`, visual gun kick) for melee.
pub fn gml_melee_wkick(full_name: &str) -> f32 {
    let key = if full_name.starts_with("ULTRA ") {
        full_name
    } else {
        base_weapon_name(full_name)
    };
    match key {
        _ if key.contains("SCREWDRIVER") => -8.0,
        "BLACK SWORD" => -7.0,
        "ULTRA SHOVEL" | "CHICKEN SWORD" => -6.0,
        "ENERGY HAMMER" => -3.0,
        _ => -4.0,
    }
}

/// GML melee projectile spec per weapon (scrFire arms). Speed is px/frame,
/// converted to px/s (x30) at spawn. `pellets` + `shifts` mirror the
/// `for(i) scr_projectile_create(Slash) + scr_projectile_shift(i*deg)` arms.
#[derive(Clone, Copy, Debug)]
pub struct MeleeProjectileSpec {
    pub sprite: &'static str,
    pub speed_f: f32,
    pub damage_override: Option<i32>,
    pub typ: u8,
    pub shank: bool,
    pub pellets: usize,
    pub shift_deg: f32,
    pub guitar: bool,
    pub electric_guitar: bool,
    pub blood: bool,
    pub lightning: bool,
    pub hammer_wallbreak: bool,
    pub mega_sprite: Option<&'static str>,
}

pub fn melee_projectile_spec(weapon_name: &str) -> MeleeProjectileSpec {
    // NOTE: base_weapon_name strips ULTRA, so match the full name first.
    let key = if weapon_name.starts_with("ULTRA ") {
        weapon_name
    } else {
        base_weapon_name(weapon_name)
    };
    match key {
        "WRENCH" => MeleeProjectileSpec {
            sprite: "images/sprSlash.png", speed_f: 2.0, damage_override: Some(8),
            typ: 0, shank: false, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "SHOVEL" => MeleeProjectileSpec {
            sprite: "images/sprHeavySlash.png", speed_f: 3.0, damage_override: Some(16),
            typ: 0, shank: false, pellets: 3, shift_deg: 60.0,
            guitar: false, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "CHICKEN SWORD" => MeleeProjectileSpec {
            sprite: "images/sprSlash.png", speed_f: 0.0, damage_override: Some(6),
            typ: 0, shank: false, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "SCREWDRIVER" => MeleeProjectileSpec {
            sprite: "images/sprShank.png", speed_f: 3.0, damage_override: Some(6),
            typ: 0, shank: true, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "ENERGY SCREWDRIVER" => MeleeProjectileSpec {
            sprite: "images/sprEnergyShank.png", speed_f: 3.0, damage_override: Some(22),
            typ: 0, shank: true, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "ENERGY SWORD" => MeleeProjectileSpec {
            sprite: "images/sprEnergySlash.png", speed_f: 0.0, damage_override: Some(22),
            typ: 0, shank: false, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "ENERGY HAMMER" => MeleeProjectileSpec {
            sprite: "images/sprEnergyHammerSlash.png", speed_f: 2.0, damage_override: Some(44),
            typ: 0, shank: false, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: true, mega_sprite: None,
        },
        "BLOOD HAMMER" => MeleeProjectileSpec {
            sprite: "images/sprBloodSlash.png", speed_f: 2.0, damage_override: Some(14),
            typ: 0, shank: false, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: false, blood: true, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "LIGHTNING HAMMER" => MeleeProjectileSpec {
            sprite: "images/sprLightningSlash.png", speed_f: 2.0, damage_override: Some(12),
            typ: 0, shank: false, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: false, blood: false, lightning: true,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "SLEDGEHAMMER" => MeleeProjectileSpec {
            sprite: "images/sprHeavySlash.png", speed_f: 2.0, damage_override: Some(24),
            typ: 0, shank: false, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "GUITAR" => MeleeProjectileSpec {
            sprite: "images/sprHeavySlash.png", speed_f: 2.0, damage_override: Some(26),
            typ: 0, shank: false, pellets: 1, shift_deg: 0.0,
            guitar: true, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "ELECTRIC GUITAR" => MeleeProjectileSpec {
            sprite: "images/sprHeavySlash.png", speed_f: 2.0, damage_override: Some(26),
            typ: 0, shank: false, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: true, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "ULTRA SHOVEL" => MeleeProjectileSpec {
            sprite: "images/sprUltraSlash.png", speed_f: 3.0, damage_override: Some(30),
            typ: 0, shank: false, pellets: 3, shift_deg: 60.0,
            guitar: false, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
        "BLACK SWORD" => MeleeProjectileSpec {
            sprite: "images/sprSlash.png", speed_f: 2.0, damage_override: Some(12),
            typ: 0, shank: false, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: Some("images/sprMegaSlash.png"),
        },
        _ => MeleeProjectileSpec {
            sprite: "images/sprSlash.png", speed_f: 2.0, damage_override: None,
            typ: 0, shank: false, pellets: 1, shift_deg: 0.0,
            guitar: false, electric_guitar: false, blood: false, lightning: false,
            hammer_wallbreak: false, mega_sprite: None,
        },
    }
}

/// GML melee swing sprite per weapon (scrFire spawns Slash/Shank/EnergySlash
/// variants; Shovel/Sledge/Guitar use sprHeavySlash, Screwdriver uses Shank,
/// EnergySword uses EnergySlash, Black Sword mega uses MegaSlash, Ultra
/// Shovel uses UltraSlash, Blood Hammer uses BloodSlash, Lightning Hammer
/// uses LightningSlash, Energy Hammer uses EnergyHammerSlash).
pub fn melee_swing_sprite(weapon_name: &str) -> &'static str {
    match base_weapon_name(weapon_name) {
        "SHOVEL" | "SLEDGEHAMMER" | "GUITAR" | "ELECTRIC GUITAR" => "images/sprHeavySlash.png",
        "SCREWDRIVER" => "images/sprShank.png",
        "ENERGY SCREWDRIVER" => "images/sprEnergyShank.png",
        "ENERGY SWORD" => "images/sprEnergySlash.png",
        "ENERGY HAMMER" => "images/sprEnergyHammerSlash.png",
        "BLOOD HAMMER" => "images/sprBloodSlash.png",
        "LIGHTNING HAMMER" => "images/sprLightningSlash.png",
        "ULTRA SHOVEL" => "images/sprUltraSlash.png",
        "BLACK SWORD" => "images/sprSlash.png",
        "WRENCH" | "CHICKEN SWORD" => "images/sprSlash.png",
        _ => "images/sprSlash.png",
    }
}

#[allow(dead_code)]
pub fn weapon_runtime(id: WeaponId) -> WeaponRuntime {
    let id = sanitize_weapon_id(id);

    if id == WeaponId::NONE {
        return WeaponRuntime {
            projectile_kind: ProjectileKind::Melee,
            pellets: 0,
            spread_deg: 0.0,
            speed: 0.0,
            lifetime_frames: 0,
            damage: 0,
            recoil: 0.0,
            explosion: None,
            melee: None,
            cooldown_frames: 1,
            automatic: false,
        };
    }

    let meta = weapon_meta(id);
    let def = weapon_runtime_def(id);

    WeaponRuntime {
        projectile_kind: projectile_kind_for(&def),
        pellets: def.pellets.min(u8::MAX as usize) as u8,
        spread_deg: def.spread,
        speed: def.speed,
        lifetime_frames: seconds_to_frames(def.lifetime),
        damage: def.damage,
        recoil: def.recoil,
        explosion: def.explosive.then_some(ExplosionSpec {
            radius: 32.0,
            damage: def.damage,
        }),
        melee: def.melee.map(|melee| MeleeSpec {
            range: melee.range,
            arc: melee.arc,
        }),
        cooldown_frames: meta.wep_load.max(1),
        automatic: meta.wep_auto,
    }
}

#[allow(dead_code)]
fn seconds_to_frames(seconds: f32) -> u16 {
    (seconds.max(0.0) * 30.0)
        .round()
        .clamp(0.0, u16::MAX as f32) as u16
}

#[allow(dead_code)]
fn projectile_kind_for(def: &WeaponDef) -> ProjectileKind {
    if def.melee.is_some() {
        return ProjectileKind::Melee;
    }

    if def.explosive {
        return ProjectileKind::Explosive;
    }

    match def.ammo {
        AmmoKind::None => ProjectileKind::Melee,
        AmmoKind::Bullets => ProjectileKind::Bullet,
        AmmoKind::Shells => ProjectileKind::Shell,
        AmmoKind::Bolts => ProjectileKind::Bolt,
        AmmoKind::Explosives => ProjectileKind::Explosive,
        AmmoKind::Energy => ProjectileKind::Energy,
    }
}

fn ammo_kind(meta: &WeaponData) -> AmmoKind {
    match meta.wep_type {
        AmmoType::None => AmmoKind::None,
        AmmoType::Bullets => AmmoKind::Bullets,
        AmmoType::Shells => AmmoKind::Shells,
        AmmoType::Bolts => AmmoKind::Bolts,
        AmmoType::Explosives => AmmoKind::Explosives,
        AmmoType::Energy => AmmoKind::Energy,
    }
}

pub fn weapon_runtime_def(id: WeaponId) -> WeaponDef {
    let id = sanitize_weapon_id(id);

    if id == WeaponId::NONE {
        return weapon_def(WeaponKind::None);
    }

    let meta = weapon_meta(id);
    let legacy: WeaponKind = id.into();

    let mut def = if legacy != WeaponKind::None {
        let mut legacy_def = weapon_def(legacy);

        legacy_def.name = meta.wep_name;
        legacy_def.ammo = ammo_kind(meta);
        legacy_def.ammo_cost = i32::from(meta.wep_cost);
        legacy_def.rad_cost = u32::from(meta.wep_rads);
        legacy_def.cooldown = f32::from(meta.wep_load.max(1)) / 30.0;
        legacy_def.automatic = meta.wep_auto;
        legacy_def
    } else {
        metadata_base_def(meta)
    };

    if legacy == WeaponKind::None {
        apply_family_profile(&mut def, family_for(meta), meta);
    }
    apply_exact_profile(&mut def, meta);
    apply_variant_tuning(&mut def, meta);
    normalize_def(&mut def, meta);

    def
}

fn metadata_base_def(meta: &WeaponData) -> WeaponDef {
    WeaponDef {
        name: meta.wep_name,
        ammo: ammo_kind(meta),
        ammo_cost: i32::from(meta.wep_cost),
        rad_cost: u32::from(meta.wep_rads),
        cooldown: f32::from(meta.wep_load.max(1)) / 30.0,
        damage: 3,
        pellets: 1,
        speed: 480.0,
        lifetime: 1.0,
        spread: 0.07,
        recoil: 3.0,
        shake: 0.08,
        projectile_radius: 4.0,
        knockback: 90.0,
        automatic: meta.wep_auto,
        explosive: false,
        burst_shots: 1,
        burst_interval: 0.0,
        melee: None,
        color: [0.9, 0.9, 0.9],
        size: Vec2::splat(12.0),
        muzzle_burst: 2,
        bounces: 0,
        pierce: 0,
        hazard: None,
        split: None,
    }
}

fn apply_family_profile(def: &mut WeaponDef, family: WeaponFamily, meta: &WeaponData) {
    let cost = i32::from(meta.wep_cost.max(1));
    let area = i32::from(meta.wep_area.max(0));

    match family {
        WeaponFamily::Empty => {}

        WeaponFamily::MeleeLight => {
            set_melee(def, 7 + cost * 2 + area / 5, 66.0, 2.05, 2.5, [0.82, 0.84, 0.88]);
        }

        WeaponFamily::MeleeHeavy => {
            set_melee(def, 14 + cost * 3 + area / 4, 80.0, 2.45, 5.5, [0.88, 0.78, 0.48]);
        }

        WeaponFamily::Pistol => {
            set_ranged(def, 3 + cost / 2, 1, 560.0, 0.85, 0.06, 3.0, 4.0, 75.0,
                [0.95, 0.9, 0.65], Vec2::new(10.0, 3.0));
        }

        WeaponFamily::Automatic => {
            set_ranged(def, 3, 1, 610.0, 0.72, 0.11, 2.2, 3.5, 48.0,
                [1.0, 0.88, 0.5], Vec2::new(10.0, 3.0));
        }

        WeaponFamily::BurstRifle => {
            set_ranged(def, 4 + cost / 3, 1, 650.0, 0.82, 0.055, 4.0, 4.0, 72.0,
                [1.0, 0.9, 0.52], Vec2::new(13.0, 3.0));

            if cost >= 3 {
                def.burst_shots = 3;
                def.burst_interval = 2.0 / 30.0;
            }
        }

        WeaponFamily::Shotgun => {
            set_ranged(def, 2, (6 + cost * 2) as usize, 400.0, 0.34, 0.31, 7.0, 3.5, 30.0,
                [1.0, 0.82, 0.36], Vec2::new(8.0, 3.0));
        }

        WeaponFamily::Slugger => {
            set_ranged(def, 14 + cost * 2, 1, 510.0, 0.7, 0.07, 9.0, 6.0, 170.0,
                [0.95, 0.83, 0.42], Vec2::new(14.0, 5.0));
        }

        WeaponFamily::Crossbow => {
            set_ranged(def, 10 + cost * 2, 1, 660.0, 1.05, 0.025, 6.0, 4.0, 135.0,
                [0.96, 0.84, 0.45], Vec2::new(17.0, 4.0));

            def.pierce = cost.saturating_sub(1).min(5) as u8;
        }

        WeaponFamily::Splinter => {
            set_ranged(def, 3, (4 + cost).clamp(4, 10) as usize, 600.0, 0.58, 0.18, 4.0, 3.0, 42.0,
                [0.68, 0.48, 0.32], Vec2::new(10.0, 3.0));
        }

        WeaponFamily::Disc => {
            set_ranged(def, 7 + cost, 1, 430.0, 2.4, 0.025, 3.5, 8.0, 125.0,
                [0.68, 0.94, 1.0], Vec2::splat(14.0));

            def.bounces = (5 + cost).clamp(6, 14) as u8;
            def.muzzle_burst = 0;
        }

        WeaponFamily::Explosive => {
            set_explosive(def, 6 + cost * 3, 1, 330.0, 0.9, 0.07, 8.0, 7.0, 165.0,
                [1.0, 0.57, 0.2], Vec2::splat(11.0));
        }

        WeaponFamily::Flame => {
            set_ranged(def, 2 + cost, (2 + cost).clamp(2, 8) as usize, 285.0, 0.35, 0.28, 3.0, 5.0, 28.0,
                [1.0, 0.48, 0.14], Vec2::new(10.0, 6.0));

            set_fire_hazard(def, 32.0 + cost as f32 * 3.0, 1 + cost / 3,
                0.75 + cost as f32 * 0.08, 0.13);
        }

        WeaponFamily::Laser => {
            set_ranged(def, 3 + cost, 1, 880.0, 0.42, 0.025, 2.5, 3.5, 35.0,
                [1.0, 0.24, 0.2], Vec2::new(18.0, 3.0));

            def.pierce = cost.saturating_sub(1).min(5) as u8;
        }

        WeaponFamily::Plasma => {
            set_explosive(def, 7 + cost * 2, 1, 300.0, 1.0, 0.055, 6.0, 8.0, 135.0,
                [0.3, 1.0, 0.35], Vec2::splat(13.0));
        }

        WeaponFamily::Lightning => {
            set_ranged(def, 3 + cost, 1, 940.0, 0.38, 0.04, 3.0, 4.0, 30.0,
                [0.72, 0.92, 1.0], Vec2::new(18.0, 4.0));

            def.pierce = cost.clamp(1, 5) as u8;
        }

        WeaponFamily::Toxic => {
            set_ranged(def, 5 + cost, 1, 410.0, 0.82, 0.045, 5.0, 6.0, 95.0,
                [0.42, 0.88, 0.38], Vec2::new(13.0, 5.0));

            set_toxic_hazard(def, 45.0 + cost as f32 * 4.0, 1 + cost / 3,
                1.8 + cost as f32 * 0.15, 0.24);
        }

        WeaponFamily::Deployable => {
            set_ranged(def, 3, 8, 620.0, 0.72, 0.2, 2.0, 3.5, 40.0,
                [0.65, 0.7, 0.76], Vec2::new(9.0, 3.0));

            def.burst_shots = 3;
            def.burst_interval = 2.0 / 30.0;
        }

        WeaponFamily::Novelty => match meta.wep_type {
            AmmoType::None => {
                set_melee(def, 10 + area / 3, 70.0, 2.1, 3.0, [0.9, 0.65, 0.85]);
            }
            AmmoType::Bullets => {
                set_ranged(def, 3, 1, 540.0, 0.75, 0.12, 3.0, 4.0, 55.0,
                    [0.95, 0.62, 0.82], Vec2::new(10.0, 4.0));
            }
            AmmoType::Shells => {
                set_ranged(def, 2, 6, 390.0, 0.35, 0.32, 6.0, 4.0, 30.0,
                    [0.95, 0.62, 0.82], Vec2::new(8.0, 4.0));
            }
            AmmoType::Bolts => {
                set_ranged(def, 9, 1, 590.0, 0.9, 0.08, 5.0, 4.0, 100.0,
                    [0.95, 0.62, 0.82], Vec2::new(14.0, 4.0));
            }
            AmmoType::Explosives => {
                set_explosive(def, 8, 1, 320.0, 0.8, 0.1, 7.0, 6.0, 130.0,
                    [0.95, 0.62, 0.82], Vec2::splat(10.0));
            }
            AmmoType::Energy => {
                set_ranged(def, 5, 1, 760.0, 0.48, 0.06, 3.0, 4.0, 45.0,
                    [0.95, 0.62, 0.82], Vec2::new(16.0, 4.0));
            }
        },
    }
}

// NOTE: the trailing `"HEAVY CROSSBOW" | "HEAVY AUTO CROSSBOW"` arm is
// shadowed by the two earlier single-name arms above it; it is dead in the
// bevy source too and is kept verbatim for table fidelity (the allow keeps
// the build warning-free).
#[allow(unreachable_patterns)]
fn apply_exact_profile(def: &mut WeaponDef, meta: &WeaponData) {
    let name = base_weapon_name(meta.wep_name);

    match name {
        "REVOLVER" => {
            set_ranged(def, 3, 1, 480.0, 0.82, 0.07, 3.0, 4.0, 75.0,
                [0.95, 0.9, 0.65], Vec2::new(10.0, 3.0));
        }

        "TRIPLE MACHINEGUN" => {
            set_ranged(def, 3, 3, 610.0, 0.7, 0.2, 4.0, 3.5, 45.0,
                [1.0, 0.86, 0.45], Vec2::new(10.0, 3.0));
        }

        "WRENCH" => {
            set_melee(def, 8, 68.0, 2.1, 3.5, [0.76, 0.78, 0.82]);
        }

        "MACHINEGUN" => {
            set_ranged(def, 3, 1, 610.0, 0.72, 0.105, 2.6, 3.5, 48.0,
                [1.0, 0.86, 0.45], Vec2::new(10.0, 3.0));
        }

        "SHOTGUN" => {
            set_ranged(def, 2, 7, 410.0, 0.34, 0.35, 7.0, 3.5, 28.0,
                [1.0, 0.8, 0.3], Vec2::new(8.0, 3.0));
        }

        "CROSSBOW" => {
            set_ranged(def, 20, 1, 720.0, 1.0, 0.025, 6.0, 4.0, 130.0,
                [0.95, 0.84, 0.45], Vec2::new(17.0, 4.0));
        }

        "GRENADE LAUNCHER" => {
            set_explosive(def, 15, 1, 300.0, 2.0, 0.052, 5.0, 4.0, 200.0,
                [1.0, 0.58, 0.2], Vec2::splat(6.0));
            def.bounces = 4;
        }

        "DOUBLE SHOTGUN" => {
            set_ranged(def, 2, 14, 410.0, 0.34, 0.52, 12.0, 3.5, 32.0,
                [1.0, 0.78, 0.27], Vec2::new(8.0, 3.0));
        }

        "MINIGUN" => {
            set_ranged(def, 3, 1, 640.0, 0.68, 0.23, 1.8, 3.0, 38.0,
                [1.0, 0.88, 0.48], Vec2::new(10.0, 3.0));
        }

        "AUTO SHOTGUN" => {
            set_ranged(def, 2, 6, 430.0, 0.32, 0.26, 5.0, 3.5, 24.0,
                [1.0, 0.8, 0.3], Vec2::new(8.0, 3.0));
        }

        "AUTO CROSSBOW" => {
            set_ranged(def, 20, 1, 720.0, 1.0, 0.085, 4.0, 4.0, 100.0,
                [0.95, 0.84, 0.45], Vec2::new(15.0, 4.0));
        }

        "SUPER CROSSBOW" => {
            set_ranged(def, 20, 5, 720.0, 1.05, 0.17, 14.0, 4.5, 130.0,
                [1.0, 0.9, 0.55], Vec2::new(18.0, 5.0));

            def.pierce = 1;
        }

        "SHOVEL" => {
            set_melee(def, 16, 84.0, 2.55, 7.0, [0.84, 0.77, 0.48]);

            def.pellets = 3;
        }

        "BAZOOKA" => {
            set_explosive(def, 20, 1, 250.0, 1.4, 0.05, 13.0, 7.0, 220.0,
                [1.0, 0.45, 0.13], Vec2::new(14.0, 8.0));
        }

        "STICKY LAUNCHER" => {
            set_explosive(def, 15, 1, 350.0, 1.7, 0.055, 7.0, 7.0, 145.0,
                [1.0, 0.6, 0.22], Vec2::splat(10.0));
        }

        "SMG" => {
            set_ranged(def, 3, 1, 480.0, 0.65, 0.28, 1.7, 3.5, 35.0,
                [1.0, 0.85, 0.47], Vec2::new(9.0, 3.0));
        }

        "ASSAULT RIFLE" => {
            set_ranged(def, 3, 1, 480.0, 0.78, 0.035, 4.0, 4.0, 65.0,
                [1.0, 0.88, 0.47], Vec2::new(12.0, 3.0));

            def.burst_shots = 3;
            def.burst_interval = 2.0 / 30.0;
        }

        "DISC GUN" => {
            set_ranged(def, 6, 1, 150.0, 2.2, 0.087, 3.0, 8.0, 120.0,
                [0.7, 0.95, 1.0], Vec2::splat(14.0));

            def.bounces = 6;
            def.muzzle_burst = 0;
        }

        "SUPER DISC GUN" => {
            set_ranged(def, 6, 5, 150.0, 2.8, 0.035, 4.0, 10.0, 180.0,
                [0.75, 1.0, 1.0], Vec2::splat(18.0));

            def.bounces = 12;
            def.muzzle_burst = 0;
        }

        "LASER PISTOL" => {
            set_ranged(def, 2, 1, 850.0, 0.42, 0.018, 2.5, 3.5, 30.0,
                [1.0, 0.22, 0.18], Vec2::new(18.0, 3.0));

            def.pierce = 1;
        }

        "LASER RIFLE" => {
            set_ranged(def, 2, 1, 900.0, 0.48, 0.05, 3.5, 3.5, 35.0,
                [1.0, 0.2, 0.16], Vec2::new(21.0, 3.0));

            def.pierce = 2;
        }

        "LASER MINIGUN" => {
            set_ranged(def, 2, 1, 820.0, 0.38, 0.21, 1.6, 3.0, 24.0,
                [1.0, 0.22, 0.17], Vec2::new(15.0, 3.0));

            def.pierce = 1;
        }

        "SLUGGER" => {
            set_ranged(def, 22, 1, 500.0, 0.66, 0.085, 9.0, 6.0, 170.0,
                [0.95, 0.82, 0.42], Vec2::new(14.0, 5.0));
        }

        "GATLING SLUGGER" => {
            set_ranged(def, 22, 1, 560.0, 0.62, 0.105, 5.0, 6.0, 135.0,
                [0.95, 0.8, 0.38], Vec2::new(14.0, 5.0));
        }

        "ASSAULT SLUGGER" => {
            set_ranged(def, 22, 1, 540.0, 0.62, 0.07, 8.0, 6.0, 150.0,
                [0.96, 0.81, 0.4], Vec2::new(14.0, 5.0));

            def.burst_shots = 3;
            def.burst_interval = 3.0 / 30.0;
        }

        "ENERGY SWORD" => {
            set_melee(def, 22, 82.0, 2.5, 4.0, [0.25, 0.86, 1.0]);
        }

        "SUPER SLUGGER" => {
            set_ranged(def, 22, 5, 560.0, 0.68, 0.18, 15.0, 6.0, 160.0,
                [1.0, 0.86, 0.44], Vec2::new(14.0, 5.0));
        }

        "HYPER RIFLE" => {
            set_ranged(def, 3, 1, 600.0, 0.55, 0.033, 2.0, 3.0, 40.0,
                [1.0, 0.95, 0.6], Vec2::new(16.0, 3.0));

            def.burst_shots = 6;
            def.burst_interval = 1.0 / 30.0;
        }

        "SCREWDRIVER" => {
            set_melee(def, 6, 58.0, 1.35, 1.7, [0.82, 0.84, 0.87]);
        }

        "ENERGY SCREWDRIVER" => {
            set_melee(def, 22, 64.0, 1.5, 2.0, [0.25, 0.86, 1.0]);
        }

        "BLOOD LAUNCHER" => {
            set_explosive(def, 10, 1, 330.0, 0.85, 0.105, 7.0, 7.0, 150.0,
                [0.9, 0.16, 0.18], Vec2::splat(11.0));
        }

        "BLOOD CANNON" => {
            set_explosive(def, 45, 1, 320.0, 0.62, 0.08, 10.0, 8.0, 220.0,
                [0.9, 0.18, 0.18], Vec2::splat(12.0));

            set_split(def, 6, 0.7, 380.0, 2, 0.34, 3.0, 30.0,
                [1.0, 0.3, 0.3], Vec2::new(7.0, 3.0));
        }

        "SPLINTER GUN" => {
            set_ranged(def, 4, 5, 620.0, 0.58, 0.18, 4.0, 3.0, 38.0,
                [0.68, 0.47, 0.3], Vec2::new(10.0, 3.0));
        }

        "SPLINTER PISTOL" => {
            set_ranged(def, 4, 4, 580.0, 0.52, 0.09, 3.0, 3.0, 34.0,
                [0.68, 0.47, 0.3], Vec2::new(9.0, 3.0));
        }

        "SUPER SPLINTER GUN" => {
            set_ranged(def, 4, 10, 650.0, 0.62, 0.26, 7.0, 3.5, 45.0,
                [0.72, 0.5, 0.32], Vec2::new(11.0, 3.0));
        }

        "TOXIC BOW" => {
            set_ranged(def, 16, 1, 620.0, 1.0, 0.02, 6.0, 4.5, 120.0,
                [0.42, 0.9, 0.37], Vec2::new(17.0, 4.0));

            set_toxic_hazard(def, 46.0, 1, 2.0, 0.24);
        }

        "SENTRY GUN" => {
            set_ranged(def, 3, 8, 620.0, 0.72, 0.2, 2.0, 3.5, 40.0,
                [0.62, 0.68, 0.76], Vec2::new(9.0, 3.0));

            def.burst_shots = 3;
            def.burst_interval = 2.0 / 30.0;
        }

        "WAVE GUN" => {
            set_ranged(def, 3, 2, 430.0, 0.5, 0.5, 7.0, 4.0, 42.0,
                [0.55, 0.86, 1.0], Vec2::new(10.0, 4.0));
        }

        "PLASMA GUN" => {
            set_explosive(def, 4, 1, 300.0, 1.1, 0.07, 5.0, 8.0, 120.0,
                [0.3, 1.0, 0.36], Vec2::splat(13.0));
        }

        "PLASMA RIFLE" => {
            set_explosive(def, 4, 1, 330.0, 1.0, 0.05, 4.0, 7.0, 105.0,
                [0.28, 1.0, 0.34], Vec2::splat(12.0));
        }

        "PLASMA MINIGUN" => {
            set_explosive(def, 4, 1, 350.0, 0.85, 0.13, 2.5, 6.0, 75.0,
                [0.28, 1.0, 0.34], Vec2::splat(10.0));
        }

        "PLASMA CANNON" | "DEVASTATOR" => {
            set_explosive(def, 15, 1, 260.0, 1.45, 0.035, 14.0, 11.0, 260.0,
                [0.35, 1.0, 0.4], Vec2::splat(18.0));
        }

        "ENERGY HAMMER" => {
            set_melee(def, 44, 92.0, 2.7, 8.0, [0.25, 0.85, 1.0]);
        }

        "JACKHAMMER" => {
            set_melee(def, 12, 62.0, 1.6, 4.0, [1.0, 0.58, 0.2]);
        }

        "FLAK CANNON" => {
            set_explosive(def, 8, 1, 340.0,
                // GML FlakBullet has no lifetime: friction slows it to a
                // stop, then it splits. Long cap here; stop-detonate governs.
                1.2, 0.07, 10.0, 7.0, 200.0,
                [1.0, 0.72, 0.3], Vec2::splat(11.0));

            set_split(def, 16, std::f32::consts::PI, 420.0, 3, 0.32, 3.0, 50.0,
                [1.0, 0.88, 0.55], Vec2::new(8.0, 3.0));
        }

        "SUPER FLAK CANNON" => {
            set_explosive(def, 45, 1, 360.0, 1.2, 0.05, 12.0, 8.0, 240.0,
                [1.0, 0.66, 0.22], Vec2::splat(12.0));

            set_split(def, 20, std::f32::consts::PI, 460.0, 3, 0.36, 3.0, 55.0,
                [1.0, 0.9, 0.6], Vec2::new(8.0, 3.0));
        }

        "CHICKEN SWORD" => {
            set_melee(def, 6, 72.0, 2.1, 2.5, [0.92, 0.92, 0.96]);
        }

        "NUKE LAUNCHER" => {
            set_explosive(def, 50, 1, 220.0, 1.7, 0.035, 18.0, 12.0, 320.0,
                [1.0, 0.38, 0.1], Vec2::splat(20.0));
        }

        "ION CANNON" => {
            set_ranged(def, 18, 1, 820.0, 0.7, 0.02, 9.0, 7.0, 150.0,
                [0.52, 0.85, 1.0], Vec2::new(25.0, 6.0));

            def.pierce = 5;
        }

        "QUADRUPLE MACHINEGUN" => {
            set_ranged(def, 3, 4, 650.0, 0.72, 0.16, 6.0, 3.5, 45.0,
                [1.0, 0.88, 0.48], Vec2::new(10.0, 3.0));
        }

        "FLAMETHROWER" => {
            set_ranged(def, 2, 5, 250.0, 0.22, 0.35, 1.2, 5.0, 18.0,
                [1.0, 0.55, 0.15], Vec2::new(10.0, 6.0));

            set_fire_hazard(def, 34.0, 1, 0.8, 0.12);
        }

        "DRAGON" => {
            set_ranged(def, 3, 7, 280.0, 0.28, 0.42, 2.0, 6.0, 25.0,
                [1.0, 0.4, 0.08], Vec2::new(12.0, 7.0));

            set_fire_hazard(def, 40.0, 1, 1.0, 0.12);
        }

        "FLARE GUN" => {
            set_ranged(def, 10, 1, 300.0, 1.1, 0.12, 6.0, 6.0, 90.0,
                [1.0, 0.32, 0.1], Vec2::splat(10.0));

            set_fire_hazard(def, 42.0, 2, 1.5, 0.18);
        }

        "HYPER LAUNCHER" => {
            set_explosive(def, 7, 2, 480.0, 0.65, 0.05, 6.0, 6.0, 100.0,
                [1.0, 0.5, 0.16], Vec2::splat(9.0));
        }

        "LASER CANNON" => {
            set_ranged(def, 18, 1, 980.0, 0.7, 0.012, 10.0, 6.0, 130.0,
                [1.0, 0.16, 0.12], Vec2::new(28.0, 5.0));

            def.pierce = 6;
        }

        "RUSTY REVOLVER" => {
            set_ranged(def, 2, 1, 510.0, 0.8, 0.12, 3.0, 4.0, 55.0,
                [0.67, 0.48, 0.31], Vec2::new(9.0, 3.0));
        }

        "LIGHTNING PISTOL" => {
            set_ranged(def, 4, 1, 920.0, 0.35, 0.06, 2.5, 4.0, 25.0,
                [0.75, 0.9, 1.0], Vec2::new(18.0, 4.0));

            def.pierce = 1;
        }

        "LIGHTNING RIFLE" => {
            set_ranged(def, 6, 1, 980.0, 0.42, 0.05, 4.0, 4.0, 35.0,
                [0.7, 0.95, 1.0], Vec2::new(22.0, 4.0));

            def.pierce = 2;
        }

        "LIGHTNING SHOTGUN" => {
            set_ranged(def, 3, 8, 860.0, 0.26, 0.3, 7.0, 3.0, 20.0,
                [0.75, 0.95, 1.0], Vec2::new(14.0, 3.0));

            def.pierce = 1;
        }

        "LIGHTNING SMG" => {
            set_ranged(def, 3, 1, 900.0, 0.32, 0.2, 2.0, 3.0, 18.0,
                [0.72, 0.93, 1.0], Vec2::new(14.0, 3.0));

            def.pierce = 1;
        }

        "LIGHTNING CANNON" => {
            set_ranged(def, 15, 1, 520.0, 1.1, 0.08, 10.0, 9.0, 140.0,
                [0.65, 0.9, 1.0], Vec2::splat(17.0));

            def.pierce = 4;
        }

        "LIGHTNING HAMMER" => {
            set_melee(def, 14, 90.0, 2.7, 7.0, [0.65, 0.9, 1.0]);

            def.pierce = 0;
            def.hazard = None;
        }

        "SAWED-OFF SHOTGUN" => {
            set_ranged(def, 2, 20, 390.0, 0.28, 0.78, 14.0, 3.0, 35.0,
                [1.0, 0.78, 0.28], Vec2::new(8.0, 3.0));
        }

        "SMART GUN" => {
            set_ranged(def, 3, 1, 480.0, 0.78, 0.087, 2.0, 4.0, 60.0,
                [0.45, 0.85, 1.0], Vec2::new(12.0, 3.0));
        }

        "HEAVY CROSSBOW" => {
            set_ranged(def, 50, 1, 480.0, 1.3, 0.015, 10.0, 6.0, 250.0,
                [0.84, 0.72, 0.3], Vec2::new(24.0, 6.0));

            def.pierce = 5;
        }

        "HEAVY AUTO CROSSBOW" => {
            set_ranged(def, 50, 1, 480.0, 1.0, 0.12, 6.5, 5.0, 140.0,
                [1.0, 0.88, 0.5], Vec2::new(16.0, 5.0));

            def.pierce = 2;
        }

        "BLOOD HAMMER" => {
            set_melee(def, 14, 80.0, 2.35, 5.0, [0.9, 0.12, 0.15]);
        }

        "POP GUN" => {
            set_ranged(def, 2, 1, 560.0, 0.52, 0.07, 2.0, 3.0, 20.0,
                [1.0, 0.72, 0.85], Vec2::new(8.0, 3.0));
        }

        "POP RIFLE" => {
            set_ranged(def, 2, 1, 610.0, 0.6, 0.08, 3.0, 3.0, 30.0,
                [1.0, 0.72, 0.85], Vec2::new(10.0, 3.0));

            def.burst_shots = 3;
            def.burst_interval = 2.0 / 30.0;
        }

        "TOXIC LAUNCHER" => {
            set_explosive(def, 7, 1, 340.0, 0.7, 0.05, 9.0, 7.0, 170.0,
                [0.45, 0.9, 0.4], Vec2::splat(11.0));

            set_toxic_hazard(def, 56.0, 1, 2.4, 0.25);
        }

        "FLAME CANNON" => {
            set_ranged(def, 9, 1, 300.0, 0.45, 0.14, 8.0, 7.0, 110.0,
                [1.0, 0.5, 0.18], Vec2::splat(12.0));

            set_fire_hazard(def, 48.0, 1, 1.1, 0.15);
        }

        "FLAME SHOTGUN" => {
            set_ranged(def, 2, 6, 300.0, 0.24, 0.42, 4.0, 4.0, 22.0,
                [1.0, 0.55, 0.18], Vec2::new(9.0, 5.0));

            set_fire_hazard(def, 32.0, 1, 0.8, 0.12);
        }

        "DOUBLE FLAME SHOTGUN" => {
            set_ranged(def, 2, 14, 300.0, 0.24, 0.5, 6.5, 4.0, 25.0,
                [1.0, 0.55, 0.2], Vec2::new(9.0, 5.0));

            set_fire_hazard(def, 34.0, 1, 0.9, 0.12);
        }

        "AUTO FLAME SHOTGUN" => {
            set_ranged(def, 2, 6, 300.0, 0.22, 0.35, 3.5, 4.0, 20.0,
                [1.0, 0.58, 0.2], Vec2::new(9.0, 5.0));

            set_fire_hazard(def, 30.0, 1, 0.75, 0.12);
        }

        "CLUSTER LAUNCHER" => {
            set_explosive(def, 8, 1, 310.0, 0.72, 0.14, 9.0, 7.0, 170.0,
                [1.0, 0.62, 0.22], Vec2::splat(11.0));

            set_split(def, 6, 0.75, 340.0, 3, 0.4, 3.0, 45.0,
                [1.0, 0.78, 0.38], Vec2::splat(7.0));
        }

        "GRENADE SHOTGUN" => {
            set_explosive(def, 4, 4, 360.0, 0.5, 0.3, 9.0, 5.0, 70.0,
                [1.0, 0.62, 0.22], Vec2::splat(8.0));
        }

        "AUTO GRENADE SHOTGUN" => {
            set_explosive(def, 4, 3, 360.0, 0.46, 0.26, 6.0, 5.0, 65.0,
                [1.0, 0.62, 0.22], Vec2::splat(8.0));
        }

        "GRENADE RIFLE" => {
            set_explosive(def, 5, 1, 405.0, 0.65, 0.07, 5.0, 5.0, 85.0,
                [1.0, 0.61, 0.2], Vec2::splat(9.0));

            // GML `NadeBurst`: 3 volleys (+Death-crown, resolved in the
            // fire path) at 2-tick intervals.
            def.burst_shots = 3;
            def.burst_interval = 2.0 / 30.0;
        }

        "ROGUE RIFLE" => {
            set_ranged(def, 3, 1, 480.0, 0.75, 0.033, 4.0, 4.0, 70.0,
                [0.45, 0.8, 1.0], Vec2::new(14.0, 3.0));

            def.burst_shots = 2;
            def.burst_interval = 2.0 / 30.0;
        }

        "PARTY GUN" => {
            set_ranged(def, 4, 1, 240.0, 0.7, 0.14, 3.0, 4.0, 35.0,
                [1.0, 0.38, 0.86], Vec2::splat(8.0));
        }

        "DOUBLE MINIGUN" => {
            set_ranged(def, 3, 2, 650.0, 0.68, 0.25, 3.0, 3.0, 38.0,
                [1.0, 0.88, 0.48], Vec2::new(10.0, 3.0));
        }

        "GATLING BAZOOKA" => {
            set_explosive(def, 10, 1, 280.0, 1.2, 0.2, 8.0, 7.0, 180.0,
                [1.0, 0.46, 0.13], Vec2::new(13.0, 7.0));
        }

        "SEEKER PISTOL" => {
            set_ranged(def, 9, 2, 240.0, 1.9, 0.52, 4.0, 4.0, 60.0,
                [1.0, 0.55, 0.85], Vec2::new(11.0, 4.0));

            def.pierce = 1;
        }

        "SEEKER SHOTGUN" => {
            set_ranged(def, 9, 6, 240.0, 1.9, 1.22, 7.0, 4.0, 90.0,
                [1.0, 0.55, 0.85], Vec2::new(11.0, 4.0));

            def.pierce = 1;
        }

        "FROG PISTOL" | "GOLDEN FROG PISTOL" => {
            set_ranged(def, 2, 3, 360.0, 1.0, 0.105, 4.0, 4.0, 60.0,
                [0.5, 0.9, 0.4], Vec2::new(8.0, 4.0));
        }

        "HYPER SLUGGER" => {
            set_ranged(def, 26, 1, 360.0, 0.7, 0.035, 16.0, 11.0, 260.0,
                [1.0, 0.85, 0.4], Vec2::new(16.0, 6.0));
        }

        "HEAVY ASSAULT RIFLE" => {
            set_ranged(def, 7, 1, 480.0, 0.85, 0.018, 7.0, 4.5, 140.0,
                [1.0, 0.88, 0.35], Vec2::new(12.0, 3.5));

            def.burst_shots = 3;
            def.burst_interval = 2.0 / 30.0;
        }

        "ERASER" => {
            set_ranged(def, 2, 17, 420.0, 0.35, 0.017, 7.0, 3.0, 40.0,
                [0.9, 0.95, 1.0], Vec2::new(14.0, 3.0));
        }

        "HEAVY REVOLVER" => {
            set_ranged(def, 7, 1, 720.0, 0.95, 0.02, 9.0, 4.5, 140.0,
                [1.0, 0.9, 0.4], Vec2::new(12.0, 4.0));
        }

        "HEAVY MACHINEGUN" => {
            set_ranged(def, 7, 1, 700.0, 0.85, 0.07, 4.5, 4.0, 70.0,
                [1.0, 0.88, 0.35], Vec2::new(12.0, 3.5));
        }

        "SLEDGEHAMMER" => {
            set_melee(def, 24, 80.0, 2.45, 6.0, [0.88, 0.78, 0.48]);
        }

        "GUITAR" | "ELECTRIC GUITAR" => {
            set_melee(def, 26, 80.0, 2.45, 6.0, [0.9, 0.7, 0.3]);
        }

        "BLACK SWORD" => {
            set_melee(def, 12, 66.0, 2.1, 4.0, [0.2, 0.2, 0.25]);
        }

        "HEAVY SLUGGER" => {
            set_ranged(def, 60, 1, 390.0, 0.7, 0.07, 34.0, 14.0, 320.0,
                [1.0, 0.85, 0.4], Vec2::new(16.0, 6.0));
        }

        "HEAVY CROSSBOW" | "HEAVY AUTO CROSSBOW" => {
            set_ranged(def, 50, 1, 480.0, 1.1, 0.025, 50.0, 6.0, 220.0,
                [1.0, 0.9, 0.5], Vec2::new(18.0, 5.0));
            def.pierce = 5;
        }

        "ULTRA REVOLVER" => {
            set_ranged(def, 18, 1, 720.0, 0.8, 0.05, 12.0, 6.0, 160.0,
                [0.95, 0.4, 1.0], Vec2::new(12.0, 4.0));
        }

        "ULTRA SHOTGUN" => {
            set_ranged(def, 6, 9, 450.0, 0.34, 0.38, 44.0, 5.0, 90.0,
                [0.95, 0.5, 1.0], Vec2::new(8.0, 3.0));
        }

        "SUPER PLASMA CANNON" => {
            set_explosive(def, 25, 1, 120.0, 2.5, 0.02, 40.0, 15.0, 400.0,
                [0.4, 1.0, 0.5], Vec2::splat(24.0));
        }

        _ => {}
    }
}

fn apply_variant_tuning(def: &mut WeaponDef, meta: &WeaponData) {
    let name = meta.wep_name;

    if meta.wep_gold || name.starts_with("GOLDEN ") {
        def.color = [1.0, 0.82, 0.2];
        def.muzzle_burst = def.muzzle_burst.saturating_add(1);

        match base_weapon_name(name) {
            "SHOTGUN" => {
                def.pellets = def.pellets.saturating_add(1);
            }
            "DISC GUN" => {
                def.bounces = def.bounces.saturating_add(2);
                def.speed *= 1.08;
            }
            "SPLINTER GUN" => {
                def.pellets = def.pellets.saturating_add(1);
            }
            _ => {}
        }
    }

    if name.starts_with("ULTRA ") {
        def.color = [0.95, 0.4, 1.0];

        let explicit_ultra = matches!(
            name,
            "ULTRA REVOLVER" | "ULTRA SHOTGUN" | "ULTRA CROSSBOW" | "ULTRA GRENADE LAUNCHER"
        );
        if !explicit_ultra {
            def.damage = ((def.damage as f32) * 1.35).round() as i32;
        }
        def.recoil *= 1.2;
        def.shake *= 1.25;

        if def.melee.is_none() {
            def.projectile_radius *= 1.15;
            def.knockback *= 1.2;
        }
    }

    if name.starts_with("CURSED ") {
        def.color = [0.7, 0.28, 0.9];
        def.damage = ((def.damage as f32) * 1.2).round() as i32;
        def.shake *= 1.15;
    }

    if name.contains("BLOOD") {
        def.color = [0.9, 0.14, 0.18];
    }
}

fn normalize_def(def: &mut WeaponDef, meta: &WeaponData) {
    def.name = meta.wep_name;
    def.ammo = ammo_kind(meta);
    def.ammo_cost = i32::from(meta.wep_cost);
    def.rad_cost = u32::from(meta.wep_rads);
    def.cooldown = f32::from(meta.wep_load.max(1)) / 30.0;
    def.automatic = meta.wep_auto;

    def.damage = def.damage.max(0);
    def.lifetime = def.lifetime.max(1.0 / 30.0);
    def.spread = def.spread.max(0.0);
    def.recoil = def.recoil.max(0.0);
    def.shake = def.shake.max(0.0);
    def.projectile_radius = def.projectile_radius.max(0.0);
    def.knockback = def.knockback.max(0.0);
    def.burst_shots = def.burst_shots.max(1);
    def.burst_interval = def.burst_interval.max(0.0);

    if def.melee.is_some() {
        def.speed = 0.0;
        def.pellets = 0;
        def.projectile_radius = 0.0;
        def.bounces = 0;
        def.pierce = 0;
        def.hazard = None;
        def.split = None;
    } else {
        def.speed = def.speed.max(1.0);
        def.pellets = def.pellets.max(1);
        def.projectile_radius = def.projectile_radius.max(1.0);
    }
}

/// Legacy per-kind base table (bevy `content.rs::weapon_def` verbatim;
/// `Color -> [f32; 3]`, `frames(f) = f / 30.0` inlined).
pub fn weapon_def(kind: WeaponKind) -> WeaponDef {
    match kind {
        WeaponKind::None => WeaponDef {
            name: "None",
            ammo: AmmoKind::Bullets,
            ammo_cost: 0,
            rad_cost: 0,
            cooldown: 1.0,
            damage: 0,
            pellets: 0,
            speed: 0.0,
            lifetime: 0.1,
            spread: 0.0,
            recoil: 0.0,
            shake: 0.0,
            projectile_radius: 0.0,
            knockback: 0.0,
            automatic: false,
            explosive: false,
            burst_shots: 0,
            burst_interval: 0.0,
            melee: None,
            color: [0.4, 0.4, 0.4],
            size: Vec2::new(1.0, 1.0),
            muzzle_burst: 0,
            bounces: 0,
            pierce: 0,
            hazard: None,
            split: None,
        },
        WeaponKind::Revolver => WeaponDef {
            name: "Revolver",
            ammo: AmmoKind::Bullets,
            ammo_cost: 1,
            rad_cost: 0,
            cooldown: frames(6.0),
            damage: 3,
            pellets: 1,
            speed: 480.0,
            lifetime: 0.95,
            spread: 0.07,
            recoil: 5.0,
            shake: 0.1,
            projectile_radius: 4.0,
            knockback: 150.0,
            automatic: false,
            explosive: false,
            burst_shots: 1,
            burst_interval: 0.0,
            melee: None,
            color: [1.0, 0.9, 0.25],
            size: Vec2::new(16.0, 5.0),
            muzzle_burst: 4,
            bounces: 0,
            pierce: 0,
            hazard: None,
            split: None,
        },
        WeaponKind::Machinegun => WeaponDef {
            name: "Machinegun",
            ammo: AmmoKind::Bullets,
            ammo_cost: 1,
            rad_cost: 0,
            cooldown: frames(5.0),
            damage: 3,
            pellets: 1,
            speed: 480.0,
            lifetime: 0.85,
            spread: 0.105,
            recoil: 3.5,
            shake: 0.06,
            projectile_radius: 3.0,
            knockback: 70.0,
            automatic: true,
            explosive: false,
            burst_shots: 1,
            burst_interval: 0.0,
            melee: None,
            color: [1.0, 1.0, 0.35],
            size: Vec2::new(12.0, 4.0),
            muzzle_burst: 2,
            bounces: 0,
            pierce: 0,
            hazard: None,
            split: None,
        },
        WeaponKind::Smg => WeaponDef {
            name: "SMG",
            ammo: AmmoKind::Bullets,
            ammo_cost: 1,
            rad_cost: 0,
            cooldown: frames(3.0),
            damage: 3,
            pellets: 1,
            speed: 480.0,
            lifetime: 0.7,
            spread: 0.28,
            recoil: 2.5,
            shake: 0.04,
            projectile_radius: 3.0,
            knockback: 50.0,
            automatic: true,
            explosive: false,
            burst_shots: 1,
            burst_interval: 0.0,
            melee: None,
            color: [1.0, 0.85, 0.3],
            size: Vec2::new(11.0, 4.0),
            muzzle_burst: 1,
            bounces: 0,
            pierce: 0,
            hazard: None,
            split: None,
        },
        WeaponKind::AssaultRifle => WeaponDef {
            name: "Assault Rifle",
            ammo: AmmoKind::Bullets,
            ammo_cost: 3,
            rad_cost: 0,
            cooldown: frames(11.0),
            damage: 3,
            pellets: 1,
            speed: 480.0,
            lifetime: 0.9,
            spread: 0.035,
            recoil: 4.0,
            shake: 0.07,
            projectile_radius: 3.5,
            knockback: 60.0,
            automatic: true,
            explosive: false,
            burst_shots: 3,
            burst_interval: frames(1.0),
            melee: None,
            color: [0.95, 0.95, 0.5],
            size: Vec2::new(13.0, 4.0),
            muzzle_burst: 2,
            bounces: 0,
            pierce: 0,
            hazard: None,
            split: None,
        },
        WeaponKind::Shotgun => WeaponDef {
            name: "Shotgun",
            ammo: AmmoKind::Bullets,
            ammo_cost: 1,
            rad_cost: 0,
            cooldown: frames(17.0),
            damage: 2,
            pellets: 7,
            speed: 450.0,
            lifetime: 0.45,
            spread: 0.35,
            recoil: 16.0,
            shake: 0.24,
            projectile_radius: 4.0,
            knockback: 90.0,
            automatic: false,
            explosive: false,
            burst_shots: 1,
            burst_interval: 0.0,
            melee: None,
            color: [1.0, 0.72, 0.26],
            size: Vec2::new(10.0, 4.0),
            muzzle_burst: 6,
            bounces: 0,
            pierce: 0,
            hazard: None,
            split: None,
        },
        WeaponKind::Crossbow => WeaponDef {
            name: "Crossbow",
            ammo: AmmoKind::Bolts,
            ammo_cost: 1,
            rad_cost: 0,
            cooldown: frames(26.0),
            damage: 20,
            pellets: 1,
            speed: 720.0,
            lifetime: 1.2,
            spread: 0.015,
            recoil: 10.0,
            shake: 0.18,
            projectile_radius: 5.0,
            knockback: 300.0,
            automatic: false,
            explosive: false,
            burst_shots: 1,
            burst_interval: 0.0,
            melee: None,
            color: [0.65, 0.35, 0.12],
            size: Vec2::new(24.0, 5.0),
            muzzle_burst: 3,
            bounces: 0,
            pierce: 0,
            hazard: None,
            split: None,
        },
        WeaponKind::GrenadeLauncher => WeaponDef {
            name: "Grenade Launcher",
            ammo: AmmoKind::Explosives,
            ammo_cost: 1,
            rad_cost: 0,
            cooldown: frames(20.0),
            damage: 15,
            pellets: 1,
            speed: 300.0,
            lifetime: 1.4,
            spread: 0.04,
            recoil: 18.0,
            shake: 0.3,
            projectile_radius: 7.0,
            knockback: 350.0,
            automatic: false,
            explosive: true,
            burst_shots: 1,
            burst_interval: 0.0,
            melee: None,
            color: [0.25, 0.95, 0.25],
            size: Vec2::splat(12.0),
            muzzle_burst: 5,
            bounces: 0,
            pierce: 0,
            hazard: None,
            split: None,
        },
        WeaponKind::Wrench => WeaponDef {
            name: "Wrench",
            ammo: AmmoKind::None,
            ammo_cost: 0,
            rad_cost: 0,
            cooldown: frames(22.0),
            damage: 8,
            pellets: 0,
            speed: 0.0,
            lifetime: 0.0,
            spread: 0.0,
            recoil: 0.0,
            shake: 0.0,
            projectile_radius: 0.0,
            knockback: 300.0,
            automatic: false,
            explosive: false,
            burst_shots: 1,
            burst_interval: 0.0,
            melee: Some(MeleeDef { range: 70.0, arc: 2.2 }),
            color: [0.7, 0.7, 0.75],
            size: Vec2::splat(20.0),
            muzzle_burst: 0,
            bounces: 0,
            pierce: 0,
            hazard: None,
            split: None,
        },
        WeaponKind::Sledgehammer => WeaponDef {
            name: "Sledgehammer",
            ammo: AmmoKind::None,
            ammo_cost: 0,
            rad_cost: 0,
            cooldown: frames(35.0),
            damage: 24,
            pellets: 0,
            speed: 0.0,
            lifetime: 0.0,
            spread: 0.0,
            recoil: 0.0,
            shake: 0.0,
            projectile_radius: 0.0,
            knockback: 600.0,
            automatic: false,
            explosive: false,
            burst_shots: 1,
            burst_interval: 0.0,
            melee: Some(MeleeDef { range: 96.0, arc: 2.6 }),
            color: [0.55, 0.5, 0.6],
            size: Vec2::splat(26.0),
            muzzle_burst: 0,
            bounces: 0,
            pierce: 0,
            hazard: None,
            split: None,
        },
    }
}

#[inline]
const fn frames(f: f32) -> f32 {
    f / 30.0
}

pub fn weapon_meta(id: WeaponId) -> &'static WeaponData {
    WEAPONS.get(id.0 as usize).unwrap_or(&WEAPONS[0])
}

pub fn sanitize_weapon_id(id: WeaponId) -> WeaponId {
    if (id.0 as usize) < WEAPONS.len() {
        id
    } else {
        WeaponId::NONE
    }
}

/// Ammo kind for a weapon id (bevy `content.rs::weapon_ammo` verbatim:
/// full `WEAPONS`-table lookup, `None` for melee/empty).
pub fn weapon_ammo(id: WeaponId) -> AmmoKind {
    if id == WeaponId::NONE {
        return AmmoKind::None;
    }

    match weapon_meta(sanitize_weapon_id(id)).wep_type {
        AmmoType::None => AmmoKind::None,
        AmmoType::Bullets => AmmoKind::Bullets,
        AmmoType::Shells => AmmoKind::Shells,
        AmmoType::Bolts => AmmoKind::Bolts,
        AmmoType::Explosives => AmmoKind::Explosives,
        AmmoType::Energy => AmmoKind::Energy,
    }
}

/// Display name for a weapon id (bevy `content.rs::weapon_id_name`
/// verbatim: `"NONE"` for empty, otherwise the table name; out-of-range
/// ids clamp to the empty row via `weapon_meta`, same as bevy).
pub fn weapon_id_name(id: WeaponId) -> &'static str {
    if id == WeaponId::NONE {
        return "NONE";
    }

    weapon_meta(id).wep_name
}

#[allow(clippy::too_many_arguments)]
fn set_ranged(
    def: &mut WeaponDef,
    damage: i32,
    pellets: usize,
    speed: f32,
    lifetime: f32,
    spread: f32,
    recoil: f32,
    radius: f32,
    knockback: f32,
    color: [f32; 3],
    size: Vec2,
) {
    def.damage = damage;
    def.pellets = pellets;
    def.speed = speed;
    def.lifetime = lifetime;
    def.spread = spread;
    def.recoil = recoil;
    def.shake = (recoil / 50.0).clamp(0.03, 0.5);
    def.projectile_radius = radius;
    def.knockback = knockback;
    def.explosive = false;
    def.melee = None;
    def.color = color;
    def.size = size;
    def.muzzle_burst = ((recoil / 2.0).round() as usize).clamp(1, 8);
    def.bounces = 0;
    def.pierce = 0;
    def.hazard = None;
    def.split = None;
}

#[allow(clippy::too_many_arguments)]
fn set_explosive(
    def: &mut WeaponDef,
    damage: i32,
    pellets: usize,
    speed: f32,
    lifetime: f32,
    spread: f32,
    recoil: f32,
    radius: f32,
    knockback: f32,
    color: [f32; 3],
    size: Vec2,
) {
    set_ranged(
        def, damage, pellets, speed, lifetime, spread, recoil, radius, knockback, color, size,
    );

    def.explosive = true;
    def.shake = (recoil / 35.0).clamp(0.1, 0.65);
}

fn set_melee(def: &mut WeaponDef, damage: i32, range: f32, arc: f32, recoil: f32, color: [f32; 3]) {
    def.damage = damage;
    def.pellets = 0;
    def.speed = 0.0;
    def.lifetime = 0.12;
    def.spread = 0.0;
    def.recoil = recoil;
    def.shake = (recoil / 45.0).clamp(0.04, 0.45);
    def.projectile_radius = 0.0;
    def.knockback = recoil * 18.0;
    def.explosive = false;
    def.burst_shots = 1;
    def.burst_interval = 0.0;
    def.melee = Some(MeleeDef { range, arc });
    def.color = color;
    def.size = Vec2::new(range, 5.0);
    def.muzzle_burst = 0;
    def.bounces = 0;
    def.pierce = 0;
    def.hazard = None;
    def.split = None;
}

fn set_fire_hazard(def: &mut WeaponDef, radius: f32, damage: i32, duration: f32, tick: f32) {
    def.hazard = Some(HazardDef {
        kind: HazardKind::Fire,
        radius,
        damage,
        duration,
        tick,
        color: [1.0, 0.48, 0.12, 0.3],
    });
}

fn set_toxic_hazard(def: &mut WeaponDef, radius: f32, damage: i32, duration: f32, tick: f32) {
    def.hazard = Some(HazardDef {
        kind: HazardKind::Toxic,
        radius,
        damage,
        duration,
        tick,
        color: [0.34, 0.9, 0.34, 0.34],
    });
}

#[allow(clippy::too_many_arguments)]
fn set_split(
    def: &mut WeaponDef,
    pellets: u8,
    spread: f32,
    speed: f32,
    damage: i32,
    lifetime: f32,
    radius: f32,
    knockback: f32,
    color: [f32; 3],
    size: Vec2,
) {
    def.split = Some(SplitDef {
        pellets,
        spread,
        speed,
        damage,
        lifetime,
        radius,
        knockback,
        color: [color[0], color[1], color[2], 1.0],
        size,
    });
}

