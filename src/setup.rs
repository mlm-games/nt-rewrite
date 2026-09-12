//! Floor setup: mask building + entity spawning from plans.
//!
//! Save-phase extension: loadout-driven run setup. Bevy `setup_run`
//! (`nt-recreated-bevy/src/game/progression.rs:48-260`) resolves the
//! save loadout (character stats, `start_crown` stamp, skins, starting
//! weapons via `sanitize_weapon_id` + `starting_ammo_for`) and spawns
//! the player; that logic lands here as pure helpers
//! (`resolve_run_loadout`, `starting_ammo_for`, `build_player_bundle`)
//! plus the headless entry points (`setup_run`, `setup_run_with_seed`)
//! so tests never touch disk or RNG.
//!
//! Headless adaptations (render/UI only, gameplay untouched):
//! - No `AssetServer`/`AssetCatalog` sprites: `PlayerAnim` keeps path
//!   strings (`idle`/`walk` from `character_def`, `hurt` from the local
//!   `derive_hurt_path`); strips, anchors and `Juice::pop_in` are render.
//! - No camera: `CameraFollow` retargeting is shell-side.
//! - No `UiBridge`/`OverlayMenu`/`PendingUnpause`: the headless sim has
//!   no menu bridge; `AppState::InGame` + `Paused(false)` carry the state.
//! - No `world::spawn_level` visuals: floor/wall/decal `Sprite`s stay
//!   renderer-owned, but the sim half (`spawn_level` below: wall bodies,
//!   props, chests, enemies, throne-room extras, mines) runs here so a
//!   headless `setup_run` yields a playable floor.
//! - Timers use disarmed `GTimer`s (bevy `ready_timer` parity: finished
//!   from birth, silent until re-armed).

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_anim::AnimCatalog;
use repame_fx::{DamageNumber, Particle};

use crate::anim::PlayerAnim;
use crate::comps_a::{
    ARENA_H, ARENA_W, AimDir, CrownState, Euphoria, FireCooldown, FloorMask, FloorStarted,
    GameCleanup, Health, HeavyHeart, Hitbox, Inventory, LevelCleanup, MAX_AMMO_TYPES,
    MAX_WEAPON_SLOTS, MutationChoice, NextHurt, OpenMind, PLAYER_BASE_SPEED, PLAYER_RADIUS, Player,
    RaceState, Run, SaveDirty, ScarierFace, Score, SelectedCharacter, Team, Toast, Velocity,
    WallCell, WallTile,
};
use crate::comps_b::{
    BigGenerator, BloodFlower, ChestKind, CrownPedestal, Enemy, GoldBarrelDrop, GoldCar,
    LoopTransition, ManholeCover, PendingDelayedBoss, PortalClear, Prop, PropHpTracker,
    PropSprites, ProtoStatue, RadChestContainer, SecretEntrance, SnowmanAmbush, ThroneCarpet,
    ThroneStatueProp,
};
use crate::crown::{apply_crown_to_spawn, crown_name_for_toast};
use crate::data::{
    AmmoKind, AreaId, CrownKind, EnemyKind, RaceId, SecretTarget, SkinLetter, WEAPON_REVOLVER,
    WeaponId, area_for_floor, resolve_start_weapon,
};
use crate::enemies::{difficulty_multiplier, spawn_enemy_at};
use crate::enemy_data::enemy_def;
use crate::environment::{
    EnvironmentHazardSpec, PropDeathEffect, ProximityMine, PulseSprite, SurfacePulse,
    pick_first_present, spawn_environment_hazard,
};
use crate::msg::Queue;
use crate::pickups::spawn_chest;
use crate::progression::DeferredFloorGen;
use crate::savedata_part::{PassiveKind, SaveData, character_def};
use crate::spatial::Pos;
use crate::time::GTimer;
use crate::weapon_runtime::{sanitize_weapon_id, weapon_ammo};
use crate::worldgen::{self, ChestSpawn, LevelPlan, PropKind, is_screen_end_wall};

/// Floor mask from a generated plan (bevy parity: cells verbatim,
/// dims from the arena constants).
pub fn build_floor_mask(plan: &LevelPlan) -> FloorMask {
    FloorMask {
        cells: plan.floor_cells.iter().copied().collect(),
        cols: (ARENA_W / crate::comps_a::TILE) as i32,
        rows: (ARENA_H / crate::comps_a::TILE) as i32,
    }
}

/// Player spawn with default Fish stats (kept byte-identical: the
/// enemy-phase test dummies use this as a targeting stand-in).
/// Loadout-driven spawns go through `spawn_player_loaded`.
/// Tag is `GameCleanup` only (bevy parity): the player survives portal
/// floor swaps (`tick_portal_suck` despawns `LevelCleanup`) and dies
/// with the run (`teardown_game` despawns `GameCleanup`).
pub fn spawn_player(commands: &mut Commands, pos: glam::Vec2) -> Entity {
    commands
        .spawn((
            GameCleanup,
            Player::default(),
            Team::Player,
            Pos(pos),
            Velocity(glam::Vec2::ZERO),
            Hitbox { radius: 8.0 },
            Health {
                hp: 8,
                max: 8,
                invuln: GTimer::disarmed(),
            },
        ))
        .id()
}

/// Enemy spawn with table stats (hp/radius from `enemy_def`).
pub fn spawn_enemy(commands: &mut Commands, kind: EnemyKind, pos: glam::Vec2) -> Entity {
    let def = enemy_def(kind);
    commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            Enemy {
                kind,
                score: 5,
                touch_damage: def.touch_damage,
                rad_drop: def.rad_drop,
                drop_chance: def.drop_chance,
                weapon_chance: def.weapon_chance,
            },
            Team::Enemy,
            Pos(pos),
            Velocity(glam::Vec2::ZERO),
            Hitbox { radius: def.radius },
            Health {
                hp: def.hp,
                max: def.hp,
                invuln: GTimer::disarmed(),
            },
        ))
        .id()
}

/// Hurt-sprite path for a player idle sprite. Bevy `anim.rs` maps
/// every `*Idle.png` to `*Hurt.png` (including B/C skin variants);
/// run setup only ever passes the 16 base `sprMutant{N}Idle` sprites,
/// so the exact table is those 16 arms plus an identity fallback.
pub fn derive_hurt_path(idle: &'static str) -> &'static str {
    match idle {
        "images/sprMutant1Idle.png" => "images/sprMutant1Hurt.png",
        "images/sprMutant2Idle.png" => "images/sprMutant2Hurt.png",
        "images/sprMutant3Idle.png" => "images/sprMutant3Hurt.png",
        "images/sprMutant4Idle.png" => "images/sprMutant4Hurt.png",
        "images/sprMutant5Idle.png" => "images/sprMutant5Hurt.png",
        "images/sprMutant6Idle.png" => "images/sprMutant6Hurt.png",
        "images/sprMutant7Idle.png" => "images/sprMutant7Hurt.png",
        "images/sprMutant8Idle.png" => "images/sprMutant8Hurt.png",
        "images/sprMutant9Idle.png" => "images/sprMutant9Hurt.png",
        "images/sprMutant10Idle.png" => "images/sprMutant10Hurt.png",
        "images/sprMutant11Idle.png" => "images/sprMutant11Hurt.png",
        "images/sprMutant12Idle.png" => "images/sprMutant12Hurt.png",
        "images/sprMutant13Idle.png" => "images/sprMutant13Hurt.png",
        "images/sprMutant14Idle.png" => "images/sprMutant14Hurt.png",
        "images/sprMutant15Idle.png" => "images/sprMutant15Hurt.png",
        "images/sprMutant16Idle.png" => "images/sprMutant16Hurt.png",
        _ => idle,
    }
}

/// Start ammo is pickup amount x3 (bevy `progression.rs`
/// `starting_ammo_for` verbatim, over the full `WEAPONS` table via
/// `weapon_ammo`: Fish bonus and Haste +1 apply per kind, then x3; the
/// max across held weapons wins per slot; melee/empty grant nothing).
pub fn starting_ammo_for(
    weapons: &[WeaponId; MAX_WEAPON_SLOTS],
    race: RaceId,
    crown: CrownKind,
) -> [i32; MAX_AMMO_TYPES] {
    let mut ammo = [0; MAX_AMMO_TYPES];

    let fish = if race == RaceId::Fish { 1 } else { 0 };
    let haste = if crown == CrownKind::Haste { 1 } else { 0 };

    let typ_ammo = |kind: AmmoKind| -> i32 {
        let base = match kind {
            AmmoKind::Bullets => 32,
            AmmoKind::Shells => 8,
            AmmoKind::Bolts => 7,
            AmmoKind::Explosives => 6,
            AmmoKind::Energy => 10,
            AmmoKind::None => 0,
        };
        let fish_bonus = match kind {
            AmmoKind::Bullets => 8 * fish,
            AmmoKind::Shells => 2 * fish,
            AmmoKind::Bolts => 2 * fish,
            AmmoKind::Explosives => 2 * fish,
            AmmoKind::Energy => 3 * fish,
            AmmoKind::None => 0,
        };
        base + fish_bonus + haste
    };

    for &weapon in weapons {
        if weapon == WeaponId::NONE {
            continue;
        }

        let kind = weapon_ammo(weapon);
        let index = match kind {
            AmmoKind::None => continue,
            AmmoKind::Bullets => 1,
            AmmoKind::Shells => 2,
            AmmoKind::Bolts => 3,
            AmmoKind::Explosives => 4,
            AmmoKind::Energy => 5,
        };

        let amount = typ_ammo(kind) * 3;

        ammo[index] = ammo[index].max(amount);
    }

    ammo
}

/// Resolved run loadout: crown (from the `start_crown` stamp), skin
/// (preferred when unlocked, else A), equipped weapons (sanitized;
/// empty start falls back to revolver), and weapon slots.
pub struct RunLoadout {
    pub crown: CrownKind,
    pub skin: u8,
    pub equipped: [WeaponId; MAX_WEAPON_SLOTS],
    pub weapon_slots: usize,
}

/// Resolve a run loadout from the save (bevy `setup_run` lines
/// 99-128 verbatim, minus engine types).
pub fn resolve_run_loadout(save: &SaveData, race: RaceId) -> RunLoadout {
    let loadout = save.race_loadout(race);
    let crown = CrownKind::from_u8(loadout.start_crown);

    let skin = if save.skin_unlocked(race, loadout.preferred_skin) {
        loadout.preferred_skin
    } else {
        0
    };

    let primary = resolve_start_weapon(sanitize_weapon_id(loadout.start_weapon));

    let explicit_start = sanitize_weapon_id(loadout.start_weapon) != WeaponId::NONE;

    let mut secondary = {
        let saved = sanitize_weapon_id(loadout.stored_weapon);
        if !explicit_start || saved == primary {
            WeaponId::NONE
        } else {
            saved
        }
    };

    // Steroids alone dual-wields.
    if race == RaceId::Steroids && secondary == WeaponId::NONE {
        secondary = WEAPON_REVOLVER;
    }

    RunLoadout {
        crown,
        skin,
        equipped: [primary, secondary, WeaponId::NONE],
        weapon_slots: if race == RaceId::Cuz { 3 } else { 2 },
    }
}

/// Player components for a race + loadout (bevy `setup_run` lines
/// 130-182 verbatim, minus sprites/camera: character stats, BigDog
/// ammo override, crown spawn application).
pub struct PlayerBundle {
    pub player: Player,
    pub race_state: RaceState,
    pub inv: Inventory,
    pub health: Health,
    pub crown_state: CrownState,
    pub cooldown: FireCooldown,
    pub anim: PlayerAnim,
}

pub fn build_player_bundle(race: RaceId, loadout: &RunLoadout) -> PlayerBundle {
    let def = character_def(race);

    let fire_rate_mult = if def.passive == PassiveKind::FastReload {
        0.8
    } else {
        1.0
    };

    let mut player = Player {
        speed: PLAYER_BASE_SPEED,
        speed_mult: def.speed_mult,
        pickup_range: def.pickup_range,
        fire_rate_mult,

        spread_mult: if race == RaceId::Steroids { 1.8 } else { 1.0 },
        chain_explosions: def.passive == PassiveKind::ChainExplosions,
        shield_on_hit: def.passive == PassiveKind::ShieldOnHit,
        ability: def.ability,
        headless_ready: def.passive == PassiveKind::Headless,
        free_ammo: def.passive == PassiveKind::FreeAmmo,
        crown: loadout.crown,
        ..Default::default()
    };

    let mut starting_ammo = starting_ammo_for(&loadout.equipped, race, loadout.crown);

    if race == RaceId::BigDog {
        starting_ammo[1] = 255;
        starting_ammo[4] = 44;
    }

    let mut inv = Inventory {
        weapons: loadout.equipped,
        cursed: [false, false, false],
        swapanim: 0.0,
        shine: 0.0,
        wepflip: 1.0,
        bwepflip: 1.0,
        weapon_slots: loadout.weapon_slots,
        current: 0,
        ammo: starting_ammo,
    };

    let mut health = Health {
        hp: def.max_hp,
        max: def.max_hp,
        invuln: GTimer::disarmed(),
    };

    apply_crown_to_spawn(loadout.crown, &mut player, &mut health, &mut inv);

    PlayerBundle {
        player,
        race_state: RaceState {
            race,
            skin: SkinLetter::from_u8(loadout.skin).unwrap_or(SkinLetter::A),
        },
        inv,
        health,
        crown_state: CrownState::new(loadout.crown),
        cooldown: FireCooldown {
            timer: GTimer::disarmed(),
            burst_left: 0,
            burst_timer: GTimer::disarmed(),
            timer_b: GTimer::disarmed(),
            burst_left_b: 0,
            burst_timer_b: GTimer::disarmed(),
        },
        anim: PlayerAnim {
            idle: def.sprite,
            walk: def.walk_sprite,
            hurt: derive_hurt_path(def.sprite),
            moving: false,
        },
    }
}

/// Spawn a loadout-built player (tags + aim + position; shared by
/// `setup_run` so spawn code is not duplicated). `GameCleanup` only —
/// same portal-survival reason as [`spawn_player`].
pub fn spawn_player_loaded(
    commands: &mut Commands,
    pos: glam::Vec2,
    bundle: PlayerBundle,
) -> Entity {
    commands
        .spawn((
            GameCleanup,
            bundle.player,
            bundle.race_state,
            bundle.inv,
            bundle.cooldown,
            bundle.health,
            bundle.crown_state,
            Team::Player,
            Hitbox {
                radius: PLAYER_RADIUS,
            },
            AimDir(glam::Vec2::Y),
            Velocity(glam::Vec2::ZERO),
            bundle.anim,
            Pos(pos),
        ))
        .id()
}

/// Headless run setup (bevy `setup_run` resource flow verbatim, minus
/// engine/UI: score/dirty/run reset, mutation-flag resources,
/// loadout player spawn, floor mask from the generated plan,
/// `FloorStarted` queue, crown toast; `AppState::InGame` marks the
/// transition the bevy caller scheduled around `setup_run`).
pub fn setup_run(world: &mut World) {
    let seed: u64 = rand::rng().random_range(0..u64::MAX);
    setup_run_with_seed(world, seed);
}

/// Deterministic `setup_run` (tests pin the floor seed).
pub fn setup_run_with_seed(world: &mut World, seed: u64) {
    {
        let stale: Vec<Entity> = world
            .query_filtered::<Entity, Or<(With<GameCleanup>, With<LevelCleanup>)>>()
            .iter(world)
            .collect();
        for e in stale {
            world.despawn(e);
        }
        let floaters: Vec<Entity> = world
            .query_filtered::<Entity, Or<(With<DamageNumber>, With<Particle>)>>()
            .iter(world)
            .collect();
        for e in floaters {
            world.despawn(e);
        }
    }

    world.init_resource::<Score>();
    world.init_resource::<Run>();
    world.init_resource::<FloorMask>();
    world.init_resource::<SaveDirty>();
    world.init_resource::<crate::input::NtInput>();
    world.init_resource::<crate::state::Paused>();
    world.init_resource::<Toast>();
    world.init_resource::<SelectedCharacter>();
    world.init_resource::<SaveData>();
    world.init_resource::<Queue<FloorStarted>>();
    world.init_resource::<crate::state::AppState>();
    world.init_resource::<DeferredFloorGen>();
    world.init_resource::<LoopTransition>();
    world.init_resource::<MutationChoice>();
    world.init_resource::<ScarierFace>();
    world.init_resource::<Euphoria>();
    world.init_resource::<OpenMind>();
    world.init_resource::<HeavyHeart>();
    // Menu/state-machine layer (menus tick before `handle_mutation_choice`
    // and must never panic on missing resources).
    world.init_resource::<crate::state::OverlayMenu>();
    world.init_resource::<crate::state::PendingUnpause>();
    world.init_resource::<crate::state::QuitRequested>();
    world.init_resource::<crate::state::menus::MenuState>();
    world.init_resource::<crate::state::menus::MenuEdge>();
    world.init_resource::<crate::audio::AudioChannels>();
    world.init_resource::<Queue<crate::audio::UiBridgeAction>>();
    world.init_resource::<Queue<crate::audio::ReactiveAudioRequest>>();
    // Area-fog scroll persists like GML's persistent TopCont
    // (init-only: never reset mid-run).
    world.init_resource::<crate::environment::FogState>();

    world.remove_resource::<crate::comps_a::PendingMutation>();
    world.remove_resource::<crate::comps_a::PendingUltra>();
    world.insert_resource(crate::comps_b::FloorTransition::default());
    world.resource_mut::<Queue<FloorStarted>>().drain();

    {
        world.resource_mut::<Score>().0 = 0;
        world.resource_mut::<SaveDirty>().0 = false;
        // GML run start with hardmode (`hard = 13`, `loops++`).
        let hardmode = world
            .get_resource::<crate::state::menus::MenuState>()
            .is_some_and(|m| m.hardmode_selected)
            && world
                .get_resource::<SaveData>()
                .is_some_and(|s| s.hardmode_unlocked);
        if let Some(mut menu) = world.get_resource_mut::<crate::state::menus::MenuState>() {
            menu.hardmode_selected = false;
        }
        let mut run = world.resource_mut::<Run>();
        run.floor = 1;
        run.world = 1;
        run.area = area_for_floor(1, 0);
        run.loop_count = u32::from(hardmode);
        run.hardmode = hardmode;
        run.floor_in_area = 1;
        run.gen_seed = seed;
        run.portal_open = false;
        run.game_over = false;
        run.total_kills = 0;
        run.blackswords = 0;
        world.resource_mut::<crate::state::Paused>().0 = false;
        *world.resource_mut::<Toast>() = Toast::default();
        *world.resource_mut::<crate::state::AppState>() = crate::state::AppState::InGame;
        world.resource_mut::<DeferredFloorGen>().0 = false;
        // Entering a run clears menu transients (pause/overlay/pending
        // mirror the bevy `setup_run` + `reset_pause_on_exit` flow) but
        // preserves the title cursor.
        *world.resource_mut::<crate::state::OverlayMenu>() = crate::state::OverlayMenu::None;
        world.resource_mut::<crate::state::PendingUnpause>().0 = None;
        {
            let mut menu = world.resource_mut::<crate::state::menus::MenuState>();
            menu.pause_confirm = None;
            menu.settings_page = 0;
            menu.settings_page_stack.clear();
            menu.mutation_selected = None;
            menu.mutation_count = 0;
            menu.mutation_is_ultra = false;
            menu.game_over = None;
        }
        *world.resource_mut::<LoopTransition>() = LoopTransition::default();
        // Fresh-run HUD/view transients: ghost-fill + crosshair lerp must
        // not carry the death position/HP into the new run (one-frame
        // flash / crosshair swoop on spawn).
        world.remove_resource::<crate::hud::HudBars>();
        world.remove_resource::<crate::render::CrosshairState>();
        world.resource_mut::<MutationChoice>().0 = None;
        world.resource_mut::<ScarierFace>().0 = false;
        world.resource_mut::<Euphoria>().0 = false;
        world.resource_mut::<OpenMind>().0 = false;
        world.resource_mut::<HeavyHeart>().0 = false;
        // Spiral background state (bevy inserts `SpiralCtl` at run
        // setup; the ambience duck keys off its presence). Streams roll
        // from the run seed so equal seeds snapshot identically.
        let area = world.resource::<Run>().area;
        world.insert_resource(crate::vortex::SpiralCtl::warmed_up_for_area_seeded(
            area, seed,
        ));
    }

    let race = world.resource::<SelectedCharacter>().0;
    let save = world.resource::<SaveData>().clone();
    let loadout = resolve_run_loadout(&save, race);
    let bundle = build_player_bundle(race, &loadout);

    spawn_player_loaded(
        &mut world.commands(),
        glam::Vec2::new(crate::comps_a::TILE * 0.5, crate::comps_a::TILE * 0.5),
        bundle,
    );
    // `World::commands` only queues: flush so the player exists before
    // the mask/queue writes below (system callers flush on schedule run).
    world.flush();

    let plan: LevelPlan = {
        let run = world.resource::<Run>();
        worldgen::generate_level(&run)
    };
    // Level entities spawn here (same plan, same order as the mask
    // build): `Run` leaves the world as an owned value because the
    // spawn call reads it while `Commands` holds `&mut World`.
    if world.get_resource::<AnimCatalog>().is_none() {
        world.insert_resource(empty_anim_catalog());
    }
    let run = world.remove_resource::<Run>().unwrap_or_default();
    world.resource_scope(|world, mut mask: Mut<FloorMask>| {
        world.resource_scope(|world, catalog: Mut<AnimCatalog>| {
            let mut commands = world.commands();
            spawn_level(&mut commands, &catalog, &run, &plan, &mut mask);
        })
    });
    world.flush();
    world.insert_resource(run);

    world
        .resource_mut::<Queue<FloorStarted>>()
        .push(FloorStarted {
            floor: 1,
            area: area_for_floor(1, 0),
        });

    if loadout.crown != CrownKind::None {
        world
            .resource_mut::<Toast>()
            .show(&format!("{} equipped", crown_name_for_toast(loadout.crown)));
    }
}

/// Seeded position hash (bevy `world::wall_hash` verbatim: splitmix64
/// over seed ^ wall coords ^ salt).
fn wall_hash(seed: u64, wx: i32, wy: i32, salt: u64) -> u64 {
    let mut x = seed
        ^ ((wx as i64 as u64) << 32)
        ^ (wy as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ salt;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn prop_hash_pick(seed: u64, pos: glam::Vec2, salt: u64, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let wx = (pos.x / 32.0).floor() as i32;
    let wy = (pos.y / 32.0).floor() as i32;
    (wall_hash(seed, wx, wy, salt) % n as u64) as usize
}

fn prop_hash_flip(seed: u64, pos: glam::Vec2, salt: u64) -> bool {
    let wx = (pos.x / 32.0).floor() as i32;
    let wy = (pos.y / 32.0).floor() as i32;
    wall_hash(seed, wx, wy, salt) % 2 == 0
}

/// Empty animation catalog for headless setup: entity spawns record
/// art paths but attach no strips (spawn fns skip missing defs, bevy
/// `catalog.has` parity via `catalog.def(...).is_some()`).
pub fn empty_anim_catalog() -> repame_anim::AnimCatalog {
    repame_anim::AnimCatalog::from_json(
        "{}",
        repame_anim::AtlasDesc {
            size: 128,
            max_pages: 1,
            padding: 0,
        },
    )
    .expect("empty catalog parses")
}

/// Ordinary-prop art candidates (bevy `spawn_prop` first-listed order).
/// Only the picked path is recorded: strips resolve renderer-side.
fn prop_candidates(kind: PropKind) -> &'static [&'static str] {
    match kind {
        PropKind::Cactus => &[
            "images/sprCactus.png",
            "images/sprCactus2.png",
            "images/sprCactus3.png",
        ],
        PropKind::BigSkull => &["images/sprBigSkullOpen.png"],
        PropKind::GroundDecal => &["images/sprDetail0.png"],
        PropKind::Barrel => &["images/sprBarrel.png"],
        PropKind::Pipe => &["images/sprSewerPipe.png"],
        PropKind::Tires => &["images/sprTires.png"],
        PropKind::ToxicBarrel => &["images/sprToxicBarrel.png"],
        PropKind::Car => &["images/sprCarIdle.png"],
        PropKind::Cocoon => &["images/sprCocoon.png"],
        PropKind::Snowman => &["images/sprSnowMan.png"],
        PropKind::Torch => &["images/sprTorch.png"],
        PropKind::GoldBarrel => &["images/sprGoldBarrel.png"],
        PropKind::BonePile => &["images/sprBonePileIdle.png"],
        PropKind::NightBonePile => &["images/sprNightBonePileIdle.png"],
        PropKind::NightCactus => &[
            "images/sprNightCactus.png",
            "images/sprNightCactus2.png",
            "images/sprNightCactus3.png",
        ],
        PropKind::Crystal => &["images/sprCrystalProp.png"],
        PropKind::Hydrant => &["images/sprHydrant.png", "images/sprIcicle.png"],
        PropKind::StreetLight => &["images/sprStreetLight.png"],
        PropKind::SodaMachine => &["images/sprSodaMachine.png", "images/sprNewsStand.png"],
        PropKind::Tube => &["images/sprTube.png"],
        PropKind::MutantTube => &["images/sprMutantTube.png"],
        PropKind::Pillar => &["images/sprNuclearPillar.png"],
        PropKind::SmallGenerator => &["images/sprSmallGenerator.png"],
        PropKind::Anchor => &["images/sprAnchor.png"],
        PropKind::WaterPlant => &["images/sprWaterPlant.png", "images/sprWaterPlant2.png"],
        PropKind::OasisBarrel => &["images/sprOasisBarrel.png"],
        PropKind::WaterMine => &["images/sprWaterMine.png"],
        PropKind::MoneyPile => &["images/sprMoneyPile.png"],
        PropKind::YVStatue => &["images/sprYVStatue.png"],
        PropKind::Bush => &["images/sprBushIdle.png"],
        PropKind::BigFlower => &["images/sprBigFlowerIdle.png"],
        PropKind::PizzaBox => &["images/sprPizzaBox.png"],
        PropKind::PlantPot => &["images/sprPlantPotIdle.png"],
        PropKind::BigGenerator => &["images/sprBigGenerator.png"],
        PropKind::ThroneStatue => &["images/sprThroneStatue.png"],
        PropKind::Cobweb | PropKind::IcePatch | PropKind::FireTrap | PropKind::Mine => {
            &["images/sprDetail0.png"]
        }
    }
}

/// Ordinary-prop combat stats: (collision size, hp, legacy explosive,
/// death effect). Bevy `spawn_prop` table verbatim, art columns dropped.
fn prop_stats(kind: PropKind, loop_count: u32) -> (f32, i32, bool, Option<PropDeathEffect>) {
    match kind {
        PropKind::Cactus | PropKind::NightCactus => (24.0, 2, false, None),
        PropKind::BigSkull => (32.0, 50, false, None),
        PropKind::Barrel => (24.0, 1, true, None),
        PropKind::Pipe => (24.0, 1, false, None),
        PropKind::Tires => (28.0, 6, false, None),
        PropKind::ToxicBarrel => (24.0, 1, false, Some(PropDeathEffect::toxic_barrel())),
        PropKind::Car => (38.0, 20, false, Some(PropDeathEffect::car())),
        PropKind::Cocoon => (24.0, 8, false, None),
        PropKind::Snowman => (24.0, 10, false, None),
        PropKind::Torch => (12.0, 20, false, None),
        PropKind::GoldBarrel => (24.0, 1, false, Some(PropDeathEffect::legacy_barrel())),
        PropKind::BonePile | PropKind::NightBonePile => (22.0, 2, false, None),
        PropKind::Crystal => (22.0, 2, false, None),
        PropKind::Hydrant => (24.0, 5, false, None),
        PropKind::StreetLight => (20.0, 5, false, None),
        PropKind::SodaMachine => (26.0, 24, false, None),
        PropKind::Tube => (20.0, 2, false, None),
        PropKind::MutantTube => (24.0, 24, false, None),
        PropKind::Pillar => (24.0, 70, false, None),
        PropKind::SmallGenerator => (24.0, 40, false, None),
        PropKind::Anchor => (28.0, 50, false, None),
        PropKind::WaterPlant => (20.0, 2, false, None),
        PropKind::OasisBarrel => (22.0, 2, false, None),
        PropKind::WaterMine => (20.0, 20, false, Some(PropDeathEffect::mine())),
        PropKind::MoneyPile => (22.0, 1, false, None),
        PropKind::YVStatue => (22.0, 15, false, None),
        PropKind::Bush => (22.0, 1, false, None),
        PropKind::BigFlower => (24.0, 8, false, None),
        PropKind::PizzaBox => (22.0, 4, false, None),
        PropKind::PlantPot => (20.0, 3, false, None),
        PropKind::BigGenerator => (40.0, if loop_count == 0 { 230 } else { 50 }, false, None),
        PropKind::ThroneStatue => (32.0, 1000, false, None),
        PropKind::GroundDecal
        | PropKind::Cobweb
        | PropKind::IcePatch
        | PropKind::FireTrap
        | PropKind::Mine => (0.0, 9_999, false, None),
    }
}

/// Per-route-floor ground-decal art (bevy `area_sprites` 6th column
/// verbatim: the `GroundDecal` prop draws the area top-decal strip).
pub fn ground_decal_for_floor(floor: u32) -> &'static str {
    let rf = ((floor.max(1) - 1) % 15) + 1;
    match rf {
        3 => "images/sprNightDesertTopDecal.png",
        4 => "images/sprTopDecalSewers.png",
        5..=7 => "images/sprTopDecalScrapyard.png",
        8 => "images/sprTopDecalCave.png",
        9..=11 | 12 => "images/sprTopDecalCity.png",
        13..=15 => "images/sprPalaceTopDecal.png",
        _ => "images/sprDesertTopDecal.png",
    }
}

/// Pick the recorded art path: hash-pick among catalog-present
/// candidates (bevy parity), else the first candidate. hurt/dead fall
/// back to idle — strips resolve renderer-side.
fn pick_prop_idle(
    catalog: &repame_anim::AnimCatalog,
    seed: u64,
    kind: PropKind,
    pos: glam::Vec2,
) -> (&'static str, bool) {
    let candidates = prop_candidates(kind);
    let present: Vec<&'static str> = candidates
        .iter()
        .copied()
        .filter(|p| catalog.def(p).is_some())
        .collect();
    let idle = if present.len() <= 1 {
        candidates
            .iter()
            .copied()
            .find(|p| catalog.def(p).is_some())
            .unwrap_or(candidates[0])
    } else {
        present[prop_hash_pick(seed, pos, 0x52, present.len())]
    };
    let flip = if kind == PropKind::SodaMachine {
        false
    } else {
        prop_hash_flip(seed, pos, 0x53)
    };
    (idle, flip)
}

/// Prop sim half (bevy `world::spawn_prop` minus sprites/anchors:
/// `Prop` + tracker + `NextHurt` + recorded art paths + death effect +
/// kind markers). Functional kinds: `FireTrap` emits its hazard (plus
/// the trap visual), `Mine` becomes a proximity mine (plus its throb),
/// `Cobweb`/`IcePatch` become `SurfaceZone` patches (plus subtle pulse
/// decals), `Torch` throbs. `GroundDecal` records art only (non-solid,
/// bevy parity).
pub fn spawn_prop_sim(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    run: &Run,
    kind: PropKind,
    pos: glam::Vec2,
) -> Option<Entity> {
    match kind {
        PropKind::Cobweb => {
            // Bevy `spawn_prop` Cobweb arm verbatim: zone + subtle pulse
            // + first-present decal sprite (srgba tint, 36px).
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                crate::environment::SurfaceZone {
                    kind: crate::environment::SurfaceKind::Cobweb,
                    half_size: glam::Vec2::splat(18.0),
                },
                SurfacePulse::subtle(pos.x * 0.017),
                PulseSprite {
                    path: pick_first_present(
                        catalog,
                        &[
                            "images/sprCobweb.png",
                            "images/sprSpiderWeb.png",
                            "images/sprWeb.png",
                            "images/sprCocoon.png",
                            "images/sprBones.png",
                        ],
                    ),
                    tint: [0.78, 0.78, 0.72, 0.62],
                    size: 36.0,
                    flip_x: false,
                },
                crate::spatial::Pos(pos),
            ));
            return None;
        }
        PropKind::IcePatch => {
            // Bevy IcePatch arm verbatim: zone + subtle pulse + decal.
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                crate::environment::SurfaceZone {
                    kind: crate::environment::SurfaceKind::Ice,
                    half_size: glam::Vec2::splat(20.0),
                },
                SurfacePulse::subtle(pos.y * 0.014),
                PulseSprite {
                    path: pick_first_present(
                        catalog,
                        &["images/sprIceDecal.png", "images/sprIcePatch.png"],
                    ),
                    tint: [0.62, 0.86, 1.0, 0.58],
                    size: 40.0,
                    flip_x: false,
                },
                crate::spatial::Pos(pos),
            ));
            return None;
        }
        PropKind::FireTrap => {
            // Bevy FireTrap arm verbatim: hazard entity (which carries
            // the hazard pulse via `spawn_environment_hazard`) plus the
            // 32px fire visual with the same fire tint.
            let e = spawn_environment_hazard(commands, pos, EnvironmentHazardSpec::fire_trap());
            commands.entity(e).insert(PulseSprite {
                path: pick_first_present(
                    catalog,
                    &[
                        "images/sprTrapFire.png",
                        "images/sprFireTrap.png",
                        "images/sprFireTrapIdle.png",
                        "images/sprTorchFire.png",
                        "images/sprTorch.png",
                        "images/sprFlameBall.png",
                    ],
                ),
                tint: EnvironmentHazardSpec::fire_trap().kind.color(),
                size: 32.0,
                flip_x: false,
            });
            // Bevy's trap arm phases the shared hazard pulse
            // `hazard(pos.x * 0.011)` (distinct from spill hazards).
            commands
                .entity(e)
                .insert(SurfacePulse::hazard(pos.x * 0.011));
            return None;
        }
        PropKind::Mine => {
            let idle_path: &'static str = "images/sprMine.png";
            let flip = prop_hash_flip(run.gen_seed, pos, 0x51);
            return Some(
                commands
                    .spawn((
                        GameCleanup,
                        LevelCleanup,
                        Prop {
                            size: glam::Vec2::splat(18.0),
                            hp: 2,
                            destructible: true,
                            explosive: false,
                        },
                        PropHpTracker { last_hp: 2 },
                        NextHurt::default(),
                        PropSprites {
                            idle: idle_path,
                            hurt: idle_path,
                            dead: idle_path,
                            flip_x: flip,
                        },
                        ProximityMine::default(),
                        PropDeathEffect::mine(),
                        // Bevy mine arm: the prop sprite itself throbs
                        // (`SurfacePulse::hazard(pos.y * 0.019)`).
                        SurfacePulse::hazard(pos.y * 0.019),
                        Pos(pos),
                    ))
                    .id(),
            );
        }
        PropKind::GroundDecal => {
            // Bevy draws the route floor's top-decal strip here (gray
            // tint renderer-side), falling back to detail art when the
            // catalog lacks it.
            let decal = ground_decal_for_floor(run.floor);
            let idle = if catalog.def(decal).is_some() {
                decal
            } else {
                pick_prop_idle(catalog, run.gen_seed, kind, pos).0
            };
            return Some(
                commands
                    .spawn((
                        GameCleanup,
                        LevelCleanup,
                        PropSprites {
                            idle,
                            hurt: idle,
                            dead: idle,
                            flip_x: false,
                        },
                        crate::comps_b::GroundDecalTint,
                        Pos(pos),
                    ))
                    .id(),
            );
        }
        _ => {}
    }

    let (size, hp, explosive, effect) = prop_stats(kind, run.loop_count);
    let (idle, flip) = pick_prop_idle(catalog, run.gen_seed, kind, pos);
    let mut ec = commands.spawn((
        GameCleanup,
        LevelCleanup,
        Prop {
            size: glam::Vec2::splat(size),
            hp,
            destructible: true,
            explosive,
        },
        PropHpTracker { last_hp: hp },
        NextHurt::default(),
        PropSprites {
            idle,
            hurt: idle,
            dead: idle,
            flip_x: flip,
        },
        Pos(pos),
    ));
    if let Some(fx) = effect {
        ec.insert(fx);
    }
    match kind {
        PropKind::Snowman => {
            ec.insert(SnowmanAmbush);
        }
        // Bevy `spawn_prop` Torch arm: the torch flame throbs.
        PropKind::Torch => {
            ec.insert(SurfacePulse::hazard(pos.x * 0.01 + pos.y * 0.02));
        }
        PropKind::GoldBarrel => {
            ec.insert(GoldBarrelDrop);
        }
        PropKind::BigGenerator => {
            ec.insert(BigGenerator { index: 0 });
        }
        PropKind::ThroneStatue => {
            ec.insert(ThroneStatueProp {
                guardian_count: (1 + run.loop_count).min(6) as u8,
            });
        }
        _ => {}
    }
    Some(ec.id())
}

/// Rad chest container (bevy `spawn_rad_container` sim half: destructible
/// prop + container marker; opening logic lives in
/// `tick_rad_container_contact`'s port phase).
pub fn spawn_rad_container(commands: &mut Commands, seed: u64, pos: glam::Vec2) -> Entity {
    let idle_path: &'static str = "images/sprRadChest.png";
    commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            Prop {
                size: glam::Vec2::splat(26.0),
                hp: 4,
                destructible: true,
                explosive: false,
            },
            PropHpTracker { last_hp: 4 },
            NextHurt::default(),
            PropSprites {
                idle: idle_path,
                hurt: idle_path,
                dead: idle_path,
                flip_x: prop_hash_flip(seed, pos, 0x54),
            },
            RadChestContainer,
            Pos(pos),
        ))
        .id()
}

/// Secret entrances for the run's area slot (bevy
/// `spawn_secret_entrances` sim half: `SecretEntrance` + `Prop` +
/// vault-guard spawns; art stays renderer-owned).
pub fn spawn_secret_entrances(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    run: &Run,
) {
    let slot: Option<(SecretTarget, glam::Vec2, f32)> = match (run.area, run.floor_in_area) {
        (AreaId::Sewers, _) => Some((
            SecretTarget::PizzaSewers,
            glam::Vec2::new(220.0, -120.0),
            28.0,
        )),
        (AreaId::Desert, 2) | (AreaId::Scrapyards, 2) | (AreaId::FrozenCity, 2) => Some((
            SecretTarget::CrownVault,
            glam::Vec2::new(-240.0, 160.0),
            34.0,
        )),
        (AreaId::Scrapyards, 1) => {
            Some((SecretTarget::YvMansion, glam::Vec2::new(260.0, 140.0), 36.0))
        }
        (AreaId::FrozenCity, 1) => {
            Some((SecretTarget::Jungle, glam::Vec2::new(-260.0, -140.0), 30.0))
        }
        _ => None,
    };
    let Some((target, pos, size)) = slot else {
        return;
    };
    let hp = if matches!(target, SecretTarget::CrownVault | SecretTarget::Vault) {
        120 + run.loop_count as i32 * 12
    } else {
        6
    };
    let idle: &'static str = match target {
        SecretTarget::PizzaSewers => "images/sprPipe.png",
        SecretTarget::CrownVault | SecretTarget::Vault => "images/sprOldGuardianStatue.png",
        SecretTarget::YvMansion => "images/sprCarIdle.png",
        SecretTarget::Jungle => "images/sprBushIdle.png",
        _ => "images/sprDetail0.png",
    };
    let mut ec = commands.spawn((
        GameCleanup,
        LevelCleanup,
        SecretEntrance { target },
        Prop {
            size: glam::Vec2::splat(size),
            hp,
            destructible: true,
            explosive: false,
        },
        PropHpTracker { last_hp: hp },
        NextHurt::default(),
        PropSprites {
            idle,
            hurt: idle,
            dead: idle,
            flip_x: false,
        },
        Pos(pos),
    ));
    match target {
        SecretTarget::PizzaSewers => {
            ec.insert(ManholeCover);
        }
        SecretTarget::CrownVault | SecretTarget::Vault => {
            ec.insert(ProtoStatue);
            // GML has no SnowBandit kind (Bandit xmas sprite-swap only);
            // vault guards are plain Bandits.
            let guard = EnemyKind::Bandit;
            for i in 0..4 {
                let ang = i as f32 * std::f32::consts::FRAC_PI_2;
                let p = pos + glam::Vec2::from_angle(ang) * 36.0;
                spawn_enemy_at(
                    commands,
                    catalog,
                    guard,
                    p,
                    difficulty_multiplier(run.floor),
                    false,
                    false,
                    run.loop_count,
                );
            }
        }
        SecretTarget::YvMansion => {
            ec.insert(GoldCar);
        }
        SecretTarget::Jungle => {
            ec.insert(BloodFlower);
        }
        _ => {}
    }
}

/// Floor entity spawn from a generated plan (bevy `world::spawn_level`
/// sim half: mask rebuild (via [`build_floor_mask`]), wall bodies,
/// props, secret entrances, chests, enemies, boss extras, throne carpet,
/// crown-vault pedestal). Floor/wall/decal/bone/detail `Sprite`s,
/// transition quads and anchors are renderer-owned and skipped.
/// Face/heart modifiers pass `false` (bevy flush parity).
pub fn spawn_level(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    run: &Run,
    plan: &LevelPlan,
    mask: &mut FloorMask,
) {
    *mask = build_floor_mask(plan);

    let floor_set: std::collections::HashSet<(i32, i32)> =
        plan.floor_cells.iter().copied().collect();
    let mut wall_set: std::collections::HashSet<(i32, i32)> = plan.wall_cells.clone();
    for &(wx, wy) in &plan.small_walls {
        wall_set.insert((wx as i32, wy as i32));
    }
    let mut all_walls: Vec<(i32, i32)> = wall_set.into_iter().collect();
    all_walls.sort_unstable();
    for (wx, wy) in all_walls {
        let c = crate::worldgen::wall_center(wx, wy);
        let wall_e = commands
            .spawn((
                GameCleanup,
                LevelCleanup,
                WallTile,
                WallCell(wx, wy),
                Prop {
                    size: glam::Vec2::splat(crate::worldgen::WALL_PX),
                    hp: 9999,
                    destructible: false,
                    explosive: false,
                },
                Pos(c),
            ))
            .id();
        if is_screen_end_wall(wx, wy, &floor_set) {
            commands.entity(wall_e).insert(crate::comps_b::ScreenEnd);
        }
    }

    for (kind, pos) in &plan.props {
        spawn_prop_sim(commands, catalog, run, *kind, *pos);
    }

    spawn_secret_entrances(commands, catalog, run);

    for chest in &plan.chests {
        match *chest {
            // GML `scrPopChests` cursed-caves arm: every WeaponChest
            // becomes a CursedBigChest plus a `PortalClear`.
            ChestSpawn::Weapon(p) if run.area == AreaId::CursedCaves => {
                spawn_chest(commands, catalog, ChestKind::CursedBig, p);
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    PortalClear {
                        timer: crate::time::GTimer::from_seconds(
                            5.0 / 30.0,
                            crate::time::TimerMode::Once,
                        ),
                    },
                    crate::spatial::Pos(p),
                ));
            }
            ChestSpawn::Weapon(p) => spawn_chest(commands, catalog, ChestKind::Weapon, p),
            ChestSpawn::Ammo(p) => spawn_chest(commands, catalog, ChestKind::Ammo, p),
            ChestSpawn::Custom(kind, p) => spawn_chest(commands, catalog, kind, p),
            ChestSpawn::Rad(p) => {
                spawn_rad_container(commands, run.gen_seed, p);
            }
        }
    }

    let difficulty = difficulty_multiplier(run.floor);
    for (kind, pos) in &plan.enemies {
        spawn_enemy_at(
            commands,
            catalog,
            *kind,
            *pos,
            difficulty,
            false,
            false,
            run.loop_count,
        );
    }
    if let Some(kind) = plan.boss {
        match kind {
            EnemyKind::BigBandit | EnemyKind::BigBanditLoop => {
                let n = plan.boss_count.max(1);
                for i in 0..n {
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        PendingDelayedBoss {
                            kind,
                            initial_trash: (plan.enemies.len() as u32).max(1),
                            kill_fraction: 0.10 + (i as f32) * 0.02,
                            from_wall: true,
                        },
                    ));
                }
            }
            EnemyKind::BigDog | EnemyKind::BigDogLoop => {
                spawn_enemy_at(
                    commands,
                    catalog,
                    kind,
                    glam::Vec2::new(-280.0, 180.0),
                    difficulty,
                    false,
                    false,
                    run.loop_count,
                );
            }
            other => {
                let pos = match other {
                    EnemyKind::Throne => glam::Vec2::new(0.0, 200.0),
                    EnemyKind::Mom => glam::Vec2::new(0.0, -40.0),
                    EnemyKind::Technomancer => glam::Vec2::new(0.0, 0.0),
                    EnemyKind::Captain => glam::Vec2::new(0.0, 80.0),
                    EnemyKind::OldGuardian => glam::Vec2::new(0.0, 60.0),
                    EnemyKind::Hyper => glam::Vec2::new(0.0, 0.0),
                    _ => glam::Vec2::new(320.0, -160.0),
                };
                spawn_enemy_at(
                    commands,
                    catalog,
                    other,
                    pos,
                    difficulty,
                    false,
                    false,
                    run.loop_count,
                );
            }
        }
    }

    if matches!(plan.boss, Some(EnemyKind::Throne)) {
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            ThroneCarpet {
                half_extents: glam::Vec2::new(36.0, 240.0),
            },
            Pos(glam::Vec2::ZERO),
        ));
    }

    if run.area == AreaId::CrownVault {
        let pool = [
            CrownKind::Life,
            CrownKind::Haste,
            CrownKind::Guns,
            CrownKind::Blood,
            CrownKind::Luck,
            CrownKind::Protection,
            CrownKind::Love,
            CrownKind::Risk,
            CrownKind::Destiny,
            CrownKind::Curses,
            CrownKind::Hatred,
        ];
        let kind = pool[rand::rng().random_range(0..pool.len())];
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            CrownPedestal { kind },
            Pos(glam::Vec2::new(0.0, 40.0)),
        ));
    }
}

