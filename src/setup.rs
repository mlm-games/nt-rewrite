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
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use repame_anim::AnimCatalog;

use crate::anim::{PlayerAnim, SpriteAnim};
use crate::comps_a::{
    ARENA_H, ARENA_W, AimDir, CrownState, Euphoria, FireCooldown, FloorMask, FloorStarted,
    GameCleanup, Health, HeavyHeart, Hitbox, Inventory, LevelCleanup, MAX_AMMO_TYPES,
    MAX_WEAPON_SLOTS, MutationChoice, NextHurt, OpenMind, PLAYER_BASE_SPEED, PLAYER_RADIUS,
    PendingMutation, PendingUltra, Player, RaceState, Run, SaveDirty, ScarierFace, Score,
    SelectedCharacter, Team, Toast, Velocity, WallCell, WallTile,
};
use crate::comps_b::{
    BigGenerator, BloodFlower, ChestKind, CrownPedestal, Enemy, FloorTransition, GoldBarrelDrop,
    GoldCar, LoopTransition, ManholeCover, PendingDelayedBoss, PortalClear, Prop, PropHpTracker,
    PropSprites, ProtoStatue, RadChestContainer, SecretEntrance, SnowmanAmbush, ThroneCarpet,
    ThroneStatueProp,
};
use crate::crown::{apply_crown_to_spawn, crown_name_for_toast};
use crate::data::{
    AmmoKind, AreaId, CrownKind, EnemyKind, RaceId, SecretTarget, SkinLetter, WEAPON_REVOLVER,
    WeaponId, ammo_max, area_for_floor, race_starter_weapon, resolve_start_weapon,
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

/// Start ammo is pickup amount x3 per held weapon (GML `scrCreatePlayers`
/// calls `scrPlayerGiveAmmo(type, pickup * 3)` separately for `wep` and
/// `bwep`, each capped at the type max): dual-wielded Steroids stack to
/// 6x, capped. Fish bonus and Haste +1 apply per kind, then x3. Melee
/// (Chicken sword) and empty grant nothing.
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

        // GML `scrPlayerGiveAmmo` caps every grant at the type max.
        ammo[index] = (ammo[index] + typ_ammo(kind) * 3).min(ammo_max(kind));
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
/// 99-128, plus the GML `scrCreatePlayers` race rules bevy skips:
/// a golden-frog-pistol start forces Frog, a locked Skeleton falls back
/// to Melting, and an empty start falls back to the race starter from
/// `scrRaceGetStarterWeapon` — not bare revolver).
pub fn resolve_run_loadout(save: &SaveData, race: RaceId) -> RunLoadout {
    // GML `scrCreatePlayers`: `cwep == wep_golden_frog_pistol` forces Frog
    // (skin/crown then come from the Frog loadout below).
    let mut race = race;
    if race != RaceId::Frog
        && sanitize_weapon_id(save.race_loadout(race).start_weapon)
            == crate::data::WEAPON_GOLDEN_FROG_PISTOL
    {
        race = RaceId::Frog;
    }
    // GML `scrCreatePlayers` (single-player, non-event): a still-locked
    // Skeleton plays Melting instead.
    if race == RaceId::Skeleton && !save.race_unlocked(RaceId::Skeleton) {
        race = RaceId::Melting;
    }

    let loadout = save.race_loadout(race);
    let crown = CrownKind::from_u8(loadout.start_crown);

    let skin = if save.skin_unlocked(race, loadout.preferred_skin) {
        loadout.preferred_skin
    } else {
        0
    };

    let raw_start = sanitize_weapon_id(loadout.start_weapon);
    let explicit_start = raw_start != WeaponId::NONE;

    // GML `scr_loadout_race_get_start_weapon`: the default start is the
    // race starter (Venuz/Cuz gold revolver, Chicken sword, Rogue rifle,
    // BigDog spin, Skeleton rusty, Frog gold frog).
    let primary = if explicit_start {
        resolve_start_weapon(raw_start)
    } else {
        race_starter_weapon(race)
    };

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

/// Roll a Random pick into an unlocked race (GML `scrCreatePlayers`
/// verbatim: `irandom_range(Fish, Cuz)` rejecting locked races and
/// BigDog; Fish is always unlocked so the loop terminates). The RNG is
/// the caller's split stream so the level-generation stream never shifts.
pub fn roll_random_race(save: &SaveData, race: RaceId, rng: &mut StdRng) -> RaceId {
    if race != RaceId::Random {
        return race;
    }
    for _ in 0..500 {
        let gml = rng.random_range(1..=16);
        if gml == RaceId::BigDog as usize {
            continue;
        }
        let Some(candidate) = crate::state::menus::race_from_gml_id(gml) else {
            continue;
        };
        if save.race_unlocked(candidate) {
            return candidate;
        }
    }
    RaceId::Fish
}

/// GML `scrRunStart:12-33` random-crown roll verbatim (`macros_general`:
/// `crwn_random` 0, `crwn_none` 1, real crowns 2..`crownmax` 13):
/// `do _crown = irandom_range(2, crownmax)` until every player has the
/// crown unlocked (50 tries), else `crwn_none`. Single-player here, so
/// one race is checked. Returns the port `CrownKind` (GML id mapped via
/// `crown_gml_to_port`: GML 2..13 → port 1..12).
/// The title loadout grid has no random slot yet, so run setup keeps the
/// stamped crown; this helper carries the law for that slot.
pub fn roll_random_crown(save: &SaveData, race: RaceId, rng: &mut StdRng) -> CrownKind {
    for _ in 0..50 {
        let gml: u8 = rng.random_range(2..=13);
        let port = CrownKind::from_u8(gml.saturating_sub(1));
        if port == CrownKind::None {
            continue;
        }
        if save.crown_unlocked(race, gml) {
            return port;
        }
    }
    CrownKind::None
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

        // GML `scrPlayerRaceChange`: Steroids `accuracy = 1.8`, Skeleton
        // `accuracy = 1.5` (spread_mult is the port's accuracy axis).
        spread_mult: if race == RaceId::Steroids {
            1.8
        } else if race == RaceId::Skeleton {
            1.5
        } else {
            1.0
        },
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
/// GML `scrCleanupSessionInstances` verbatim
/// (`scrCleanupSessionInstances.gml:1-11`): `with all { if (object_index
/// == UberCont || == CoopController || == Console) continue;
/// instance_destroy(id, false) }`. The port's persistent controllers carry
/// no entities (resources own that state), so this despawns every live
/// entity — the `scrGameRestart` quit/restart path destroys the whole
/// session before rebuilding. `setup_run_with_seed` and
/// `setup_title_campfire` both funnel through here so menu transitions
/// can never inherit a dead run's world.
pub fn teardown_session_entities(world: &mut World) {
    //
    let stale: Vec<Entity> = world
        .query::<Entity>()
        .iter(world)
        .filter(|e| {
            world
                .get_entity(*e)
                .is_ok_and(|r| r.get::<bevy_ecs::resource::IsResource>().is_none())
        })
        .collect();
    for e in stale {
        let _ = world.despawn(e);
    }
}

pub fn setup_run(world: &mut World) {
    let seed: u64 = rand::rng().random_range(0..u64::MAX);
    setup_run_with_seed(world, seed);
}

/// Deterministic `setup_run` (tests pin the floor seed).
pub fn setup_run_with_seed(world: &mut World, seed: u64) {
    teardown_session_entities(world);

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
        // `UberCont.hardmode` persists once set (PlayButton image 3);
        // `scrGameRestart` never clears it, so retries keep it. The
        // port likewise consumes without clearing — the menu re-arms
        // the flag on every PLAY path into Title.
        let hardmode = world
            .get_resource::<crate::state::menus::MenuState>()
            .is_some_and(|m| m.hardmode_selected)
            && world
                .get_resource::<SaveData>()
                .is_some_and(|s| s.hardmode_unlocked);
        // GML `GenCont/Create_0`: the `game.tutorial` save flag forces the
        // 5-floor `TutCont` level. The port has no tutorial steps and no
        // completion event, so the layout applies to first-ever runs only
        // (`total_runs == 0`); afterwards the flag reads as completed.
        let tutorial = world
            .get_resource::<SaveData>()
            .is_some_and(|s| s.settings.show_tutorial && s.total_runs == 0);
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
        run.tottimer = 0;
        run.popolevel = 0;
        run.nochest = 0;
        run.noradch = 0;
        run.same_weapons_for = 0;
        run.horror = false;
        run.shots_fired = 0;
        run.weapons_picked = 0;
        run.won = false;
        run.tutorial = tutorial;
        run.blood_crown = false;
        run.waypoints.clear();
        run.push_waypoint();
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

    let picked = world.resource::<SelectedCharacter>().0;
    let save = world.resource::<SaveData>().clone();
    // GML `scrCreatePlayers` Random roll on its own split stream (level
    // generation keeps its own `gen_seed` stream either way).
    let mut roll_rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let race = roll_random_race(&save, picked, &mut roll_rng);
    let loadout = resolve_run_loadout(&save, race);
    // GML `scrPopulate` Blood-crown extra pass reads the run-start crown.
    world.resource_mut::<Run>().blood_crown = loadout.crown == CrownKind::Blood;
    let bundle = build_player_bundle(race, &loadout);

    spawn_player_loaded(
        &mut world.commands(),
        glam::Vec2::new(crate::comps_a::TILE * 0.5, crate::comps_a::TILE * 0.5),
        bundle,
    );
    // `World::commands` only queues: flush so the player exists before
    // the mask/queue writes below (system callers flush on schedule run).
    world.flush();

    let mut plan: LevelPlan = {
        let run = world.resource::<Run>();
        worldgen::generate_level(&run)
    };
    // GML `GenCont/Alarm_0 -> scrPopulate -> scrPopChests` verbatim: the
    // first floor trims to 1 weapon / 1 ammo / 1 rad chest (plus Open-Mind
    // widening, rad permutations, mimic rolls, hardmode BigWeaponChest).
    // Seeded off the Generation stream so equal seeds permute identically.
    {
        let gen_seed = world.resource::<Run>().gen_seed;
        let open_mind = world.get_resource::<OpenMind>().is_some_and(|o| o.0);
        let player_spawn = glam::Vec2::new(crate::comps_a::TILE * 0.5, crate::comps_a::TILE * 0.5);
        // Player hp is in the just-spawned bundle: read it back for the
        // half-health HealthChest arm (GML reads the live player).
        let (half, rogue, life, love, hardmode, area, sub, loops) = {
            let mut half = false;
            let mut rogue = race == RaceId::Rogue;
            let mut q = world.query::<(&crate::comps_a::Health, &crate::comps_a::RaceState)>();
            for (hp, rs) in q.iter(world) {
                half = hp.hp * 2 < hp.max;
                rogue = rs.race == RaceId::Rogue;
                break;
            }
            let r = world.resource::<crate::comps_a::Run>();
            (
                half,
                rogue,
                loadout.crown == CrownKind::Life,
                loadout.crown == CrownKind::Love,
                r.hardmode,
                r.area,
                r.floor_in_area,
                r.loop_count,
            )
        };
        let out = worldgen::apply_chest_permutations(
            &mut plan,
            worldgen::ChestPermuteCtx {
                area,
                loops,
                subarea: sub,
                rogue_in_run: rogue,
                player_half_health: half,
                crown_life: life,
                crown_love: love,
                open_mind,
                noradch: 0,
                nochest: 0,
                same_weapons_for: 0,
                horror_done: false,
                hardmode,
                player_pos: player_spawn,
                seed: gen_seed ^ 0xC0E5_75EED,
            },
        );
        if out.horror {
            world.resource_mut::<crate::comps_a::Run>().horror = true;
        }
    }
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
    // GML `GenCont/Destroy:186-187` verbatim:
    // `instance_destroy(SpiralCont)` once generation lands. The run
    // setup warms the sim `SpiralCtl` for the generating room (see
    // above); the built level means generation end, so the cont dies
    // here instead of leaking a live spiral into settled play.
    world.remove_resource::<crate::vortex::SpiralCtl>();

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

/// Prop hurt/dead art for a picked idle strip (GML prop objects carry
/// `spr_idle / spr_hurt / spr_dead` triples; the pack mirrors them as
/// `sprXHurt.png` / `sprXDead.png`, with `*Idle` idles stripping the
/// suffix — except `sprCarIdle`, whose hurt strip is `sprCarHurt`).
/// Returns static candidates; callers keep `idle` when the catalog lacks
/// the strip (GML-equivalent: no hit anim / plain debris).
fn prop_hurt_dead_paths(idle: &'static str) -> (&'static str, &'static str) {
    match idle {
        "images/sprBushIdle.png" => ("images/sprBushHurt.png", "images/sprBushDead.png"),
        "images/sprBigFlowerIdle.png" => {
            ("images/sprBigFlowerHurt.png", "images/sprBigFlowerDead.png")
        }
        "images/sprBonePileIdle.png" => {
            ("images/sprBonePileHurt.png", "images/sprBonePileDead.png")
        }
        "images/sprNightBonePileIdle.png" => (
            "images/sprNightBonePileHurt.png",
            "images/sprNightBonePileDead.png",
        ),
        "images/sprPlantPotIdle.png" => {
            ("images/sprPlantPotHurt.png", "images/sprPlantPotDead.png")
        }
        "images/sprCarIdle.png" => ("images/sprCarHurt.png", "images/sprCarIdle.png"),
        "images/sprMine.png" | "images/sprMineIdle.png" => {
            ("images/sprMine.png", "images/sprMine.png")
        }
        "images/sprCactus.png" => ("images/sprCactusHurt.png", "images/sprCactusDead.png"),
        "images/sprCactus2.png" => ("images/sprCactus2Hurt.png", "images/sprCactus2Dead.png"),
        "images/sprCactus3.png" => ("images/sprCactus3Hurt.png", "images/sprCactus3Dead.png"),
        "images/sprBigSkullOpen.png" => (
            "images/sprBigSkullOpenHurt.png",
            "images/sprBigSkullOpen.png",
        ),
        "images/sprBarrel.png" => ("images/sprBarrelHurt.png", "images/sprBarrelDead.png"),
        "images/sprSewerPipe.png" => ("images/sprSewerPipeHurt.png", "images/sprSewerPipeDead.png"),
        "images/sprTires.png" => ("images/sprTiresHurt.png", "images/sprTiresDead.png"),
        "images/sprToxicBarrel.png" => (
            "images/sprToxicBarrelHurt.png",
            "images/sprToxicBarrelDead.png",
        ),
        "images/sprCocoon.png" => ("images/sprCocoonHurt.png", "images/sprCocoonDead.png"),
        "images/sprSnowMan.png" => ("images/sprSnowManHurt.png", "images/sprSnowManDead.png"),
        "images/sprTorch.png" => ("images/sprTorchHurt.png", "images/sprTorchDead.png"),
        "images/sprGoldBarrel.png" => (
            "images/sprGoldBarrelHurt.png",
            "images/sprGoldBarrelDead.png",
        ),
        "images/sprNightCactus.png" => (
            "images/sprNightCactusHurt.png",
            "images/sprNightCactusDead.png",
        ),
        "images/sprNightCactus2.png" => (
            "images/sprNightCactus2Hurt.png",
            "images/sprNightCactus2Dead.png",
        ),
        "images/sprNightCactus3.png" => (
            "images/sprNightCactus3Hurt.png",
            "images/sprNightCactus3Dead.png",
        ),
        "images/sprCrystalProp.png" => (
            "images/sprCrystalPropHurt.png",
            "images/sprCrystalPropDead.png",
        ),
        "images/sprHydrant.png" => ("images/sprHydrantHurt.png", "images/sprHydrantDead.png"),
        "images/sprIcicle.png" => ("images/sprIcicleHurt.png", "images/sprIcicleDead.png"),
        "images/sprStreetLight.png" => (
            "images/sprStreetLightHurt.png",
            "images/sprStreetLightDead.png",
        ),
        "images/sprSodaMachine.png" => (
            "images/sprSodaMachineHurt.png",
            "images/sprSodaMachineDead.png",
        ),
        "images/sprNewsStand.png" => ("images/sprNewsStandHurt.png", "images/sprNewsStandDead.png"),
        "images/sprTube.png" => ("images/sprTubeHurt.png", "images/sprTubeDead.png"),
        "images/sprMutantTube.png" => (
            "images/sprMutantTubeHurt.png",
            "images/sprMutantTubeDead.png",
        ),
        "images/sprNuclearPillar.png" => (
            "images/sprNuclearPillarHurt.png",
            "images/sprNuclearPillarDead.png",
        ),
        "images/sprSmallGenerator.png" => (
            "images/sprSmallGeneratorHurt.png",
            "images/sprSmallGeneratorDead.png",
        ),
        "images/sprAnchor.png" => ("images/sprAnchorHurt.png", "images/sprAnchorDead.png"),
        "images/sprWaterPlant.png" => (
            "images/sprWaterPlantHurt.png",
            "images/sprWaterPlantDead.png",
        ),
        "images/sprWaterPlant2.png" => (
            "images/sprWaterPlant2Hurt.png",
            "images/sprWaterPlant2Dead.png",
        ),
        "images/sprOasisBarrel.png" => (
            "images/sprOasisBarrelHurt.png",
            "images/sprOasisBarrelDead.png",
        ),
        "images/sprWaterMine.png" => ("images/sprWaterMineHurt.png", "images/sprWaterMineDead.png"),
        "images/sprMoneyPile.png" => ("images/sprMoneyPileHurt.png", "images/sprMoneyPileDead.png"),
        "images/sprYVStatue.png" => ("images/sprYVStatueHurt.png", "images/sprYVStatueDead.png"),
        "images/sprPizzaBox.png" => ("images/sprPizzaBoxHurt.png", "images/sprPizzaBoxDead.png"),
        "images/sprBigGenerator.png" => (
            "images/sprBigGeneratorHurt.png",
            "images/sprBigGeneratorDead.png",
        ),
        "images/sprThroneStatue.png" => (
            "images/sprThroneStatue.png",
            "images/sprThroneStatueDead.png",
        ),
        _ => (idle, idle),
    }
}

/// Resolve idle → hurt/dead against the catalog (missing strips fall
/// back to idle, which keeps the generic hurt flash / plain debris).
fn resolve_prop_art(
    catalog: &repame_anim::AnimCatalog,
    idle: &'static str,
) -> (&'static str, &'static str) {
    let (hurt_c, dead_c) = prop_hurt_dead_paths(idle);
    let hurt = if catalog.def(hurt_c).is_some() {
        hurt_c
    } else {
        idle
    };
    let dead = if catalog.def(dead_c).is_some() {
        dead_c
    } else {
        idle
    };
    (hurt, dead)
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
            let (hurt, dead) = resolve_prop_art(catalog, idle_path);
            let mut ec = commands.spawn((
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
                    hurt,
                    dead,
                    flip_x: flip,
                },
                ProximityMine::default(),
                PropDeathEffect::mine(),
                // Bevy mine arm: the prop sprite itself throbs
                // (`SurfacePulse::hazard(pos.y * 0.019)`).
                SurfacePulse::hazard(pos.y * 0.019),
                Pos(pos),
            ));
            // Hurt-flash tracker: `prop_hurt_on_damage` bails without it.
            if let Some(def) = catalog.def(idle_path) {
                ec.insert(SpriteAnim::new(idle_path, def));
            }
            return Some(ec.id());
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
    // GML prop objects swap to hurt/dead strips on damage/death; the
    // hurt system (`prop_hurt_on_damage`) and corpse spawner
    // (`spawn_prop_corpse`) both read these, and the hurt system bails
    // without a `SpriteAnim` tracker — so record real paths and insert
    // the tracker (missing strips fall back to idle).
    let (hurt, dead) = resolve_prop_art(catalog, idle);
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
            hurt,
            dead,
            flip_x: flip,
        },
        Pos(pos),
    ));
    if let Some(def) = catalog.def(idle) {
        ec.insert(SpriteAnim::new(idle, def));
    }
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
pub fn spawn_rad_container(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    seed: u64,
    pos: glam::Vec2,
) -> Entity {
    let idle_path: &'static str = "images/sprRadChest.png";
    // `play_hurt` falls back to idle when the strip is absent, so the
    // literal is safe without a catalog gate here.
    let hurt_path: &'static str = "images/sprRadChestHurt.png";
    let mut ec = commands.spawn((
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
            hurt: hurt_path,
            dead: idle_path,
            flip_x: prop_hash_flip(seed, pos, 0x54),
        },
        RadChestContainer,
        Pos(pos),
    ));
    if let Some(def) = catalog.def(idle_path) {
        ec.insert(SpriteAnim::new(idle_path, def));
    }
    ec.id()
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

/// Wall bodies for a wall-cell set (shared by [`spawn_level`] and the
/// title campfire below): indestructible `WallTile` bodies plus the
/// screen-end mark. `floor_set` drives `is_screen_end_wall`.
fn spawn_wall_tiles(
    commands: &mut Commands,
    wall_cells: Vec<(i32, i32)>,
    floor_set: &std::collections::HashSet<(i32, i32)>,
) {
    let mut all_walls = wall_cells;
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
        if is_screen_end_wall(wx, wy, floor_set) {
            commands.entity(wall_e).insert(crate::comps_b::ScreenEnd);
        }
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
    spawn_wall_tiles(&mut *commands, wall_set.into_iter().collect(), &floor_set);

    for (kind, pos) in &plan.props {
        spawn_prop_sim(commands, catalog, run, *kind, *pos);
    }

    spawn_secret_entrances(commands, catalog, run);

    // Bandit-camp spots (GML `scrPopulate`: every free weapon/ammo/rad
    // chest camps a Bandit; `BigWeaponChest` and `RadMaggotChest` are
    // excluded, everything else qualifies).
    let mut bandit_spots: Vec<glam::Vec2> = Vec::new();
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
                        scale: 1.0,
                    },
                    crate::spatial::Pos(p),
                ));
                bandit_spots.push(p);
            }
            ChestSpawn::Weapon(p) => {
                spawn_chest(commands, catalog, ChestKind::Weapon, p);
                bandit_spots.push(p);
            }
            ChestSpawn::Ammo(p) => {
                spawn_chest(commands, catalog, ChestKind::Ammo, p);
                bandit_spots.push(p);
            }
            ChestSpawn::Custom(kind, p) => {
                spawn_chest(commands, catalog, kind, p);
                // GML excludes only the mimic-rolled `BigWeaponChest` (a
                // `WeaponChest` child with a foreign object id) and the
                // `RadMaggotChest` (`with RadChest` skips it).
                if !matches!(kind, ChestKind::BigWeapon | ChestKind::RadMaggot) {
                    bandit_spots.push(p);
                }
            }
            ChestSpawn::Rad(p) => {
                spawn_rad_container(commands, catalog, run.gen_seed, p);
                bandit_spots.push(p);
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

    // GML `scrPopulate` bandit camps: a Bandit on every free chest spot,
    // except Crown-Vault-style finale floors (which hold no chests anyway)
    // and the Mansion/HQ secret slots. GML area ints: city 5, vault 100,
    // mansion 103, hq 106.
    {
        let g = crate::worldgen::gml_area_from_run(run);
        if (g < 5 || g > 100) && g != 103 && g != 106 {
            for spot in bandit_spots {
                if !mask.is_walkable(spot) {
                    continue;
                }
                spawn_enemy_at(
                    commands,
                    catalog,
                    EnemyKind::Bandit,
                    spot,
                    difficulty,
                    false,
                    false,
                    run.loop_count,
                );
            }
        }
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

/// Title campfire backdrop (GML `MenuGen/Create_0` + `Alarm_1` +
/// GML `Vlambeer` logo-room branch verbatim (`Vlambeer/Create_0` with
/// `want_quit_to_menu` and no `CoopController`): the room restarts with
/// no gameplay instances — just `Logo` + `SpiralCont` over the black
/// clear color. The port has no room primitive, so this is the blanket
/// teardown plus the resource half of the room restart (empty campfire
/// `Run` for the spiral variant, silenced area audio, cleared
/// transition/offer covers). The campfire floor itself is NOT built here
/// (GML builds it only in `MenuGen`, i.e. on PLAY into the title) — the
/// logo menu sits over black, and `world_instances` emits nothing with
/// an empty mask.
pub fn setup_logo_room(world: &mut World) {
    teardown_session_entities(world);
    world.init_resource::<FloorMask>();
    (*world.resource_mut::<FloorMask>()) = FloorMask::default();
    reset_menu_room_resources(world);
}

/// Shared resource half of the menu-room restarts (`setup_logo_room`
/// above + `setup_title_campfire` below): empty campfire `Run` (GML
/// `GameCont` reads campfire with no run alive → spiral variant +
/// debris strip), silenced area audio (`audio_stop_all`), cleared
/// transition/offer covers.
fn reset_menu_room_resources(world: &mut World) {
    // GML `GameCont/Create_0` verbatim: a fresh cont resets every run
    // counter (area/loops/chests/kills/timers/flags/waypoints). Full
    // default + campfire identity, so no field can leak a dead run
    // into the menu room.
    world.insert_resource(Run::default());
    {
        let mut run = world.resource_mut::<Run>();
        run.floor = 0;
        run.world = 0;
        run.area = crate::data::AreaId::Campfire;
        run.floor_in_area = 0;
    }
    world.init_resource::<crate::audio::AreaAudioState>();
    *world.resource_mut::<crate::audio::AreaAudioState>() = crate::audio::AreaAudioState::default();
    world.init_resource::<crate::audio::GameAudio>();
    world.remove_resource::<PendingMutation>();
    world.remove_resource::<PendingUltra>();
    world.insert_resource(FloorTransition::default());
    // GML `scrGameRestart` quit arm verbatim (`continued_run=false` +
    // `scrCleanupSessionInstances`): no offer/flag/toast state survives
    // into the menu room or the next run. Score re-seeds in
    // `setup_run`; everything else resets here.
    world.init_resource::<Score>();
    world.resource_mut::<Score>().0 = 0;
    world.init_resource::<crate::comps_a::Toast>();
    *world.resource_mut::<crate::comps_a::Toast>() = crate::comps_a::Toast::default();
    world.init_resource::<crate::comps_a::MutationChoice>();
    world.resource_mut::<crate::comps_a::MutationChoice>().0 = None;
    world.init_resource::<crate::comps_a::ScarierFace>();
    world.resource_mut::<crate::comps_a::ScarierFace>().0 = false;
    world.init_resource::<crate::comps_a::Euphoria>();
    world.resource_mut::<crate::comps_a::Euphoria>().0 = false;
    world.init_resource::<crate::comps_a::OpenMind>();
    world.resource_mut::<crate::comps_a::OpenMind>().0 = false;
    world.init_resource::<crate::comps_a::HeavyHeart>();
    world.resource_mut::<crate::comps_a::HeavyHeart>().0 = false;
    // `LoopTransition` stays present but defaulted: bevy systems take
    // it as a bare `Res`/`ResMut` param (ungated `Always` set), so
    // removing it panics the schedule every tick on menu rooms. Stale
    // loop-portal flags must still not leak, hence the default write.
    world.init_resource::<crate::comps_b::LoopTransition>();
    *world.resource_mut::<crate::comps_b::LoopTransition>() =
        crate::comps_b::LoopTransition::default();
    world.remove_resource::<crate::hud::HudBars>();
}

/// `scrCampfireMenuCreate` sim half: 3x4 jittered 3x3 floor patches,
/// cardinal-neighbour fill, ring walls, 1-in-6 NightCactus/TopDecal
/// dressing, `PortalClear` per camper, and the Campfire/LogMenu/CampChar
/// actors; no enemies, chests, makers or run state). Entities carry
/// `GameCleanup`/`LevelCleanup` so run setup clears them; the `FloorMask`
/// lets title-time systems collide against the same walls the menu draws.
pub fn setup_title_campfire(world: &mut World) {
    teardown_session_entities(world);
    world.init_resource::<FloorMask>();
    if world.get_resource::<AnimCatalog>().is_none() {
        world.insert_resource(empty_anim_catalog());
    }

    // GML `MenuGen/Create_0` verbatim: 3 rows x 4 cols of 3x3 patches.
    // `dix` starts 32 for row 0 but resets to 0 after each row (so
    // rows 1-2 use dix=0,32,64,96); `diy=32+row*32` px; one
    // `mody=choose(32,0,-32)` jitters BOTH axes of the patch. In cells
    // (px/32): base `(dix_c+col+mody_c, 1+row+mody_c)` with
    // `dix_c = 1` on row 0 else `0`, `mody_c in {1,0,-1}`. GML draws
    // from the live global RNG stream, so every title visit differs;
    // the port likewise rolls live entropy (no two camps alike).
    let mut rng = rand::rng();
    let mut seen = std::collections::HashSet::new();
    let mut floors: Vec<(i32, i32)> = Vec::new();
    for row in 0..3 {
        let dix_c = if row == 0 { 1 } else { 0 };
        for col in 0..4 {
            let mody_c: i32 = [1, 0, -1][rng.random_range(0..3)];
            let bx = dix_c + col + mody_c;
            let by = 1 + row + mody_c;
            for ox in -1..=1 {
                for oy in -1..=1 {
                    let c = (bx + ox, by + oy);
                    if seen.insert(c) {
                        floors.push(c);
                    }
                }
            }
        }
    }
    // `MenuGen/Alarm_1` cardinal fill verbatim (neighbour floors pop in).
    let snapshot = floors.clone();
    for (cx, cy) in snapshot {
        for c in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)] {
            if seen.insert(c) {
                floors.push(c);
            }
        }
    }
    // `MenuGen/Create_0:38-40` FloorMakers verbatim: 4 makers at
    // `choose(0,32,64,96,128)` px each axis, `goal = 50` under MenuGen.
    // The 12 patches + fill already exceed 50 floors, so every maker
    // lays exactly its spawn cell on the first step (`Floor > goal`
    // arm) — up to 4 satellite cells, duplicates popping themselves
    // (`Floor/Create_0` overlap arm, matched by `seen` here).
    for _ in 0..4 {
        let c = (
            rng.random_range(0..=4),
            rng.random_range(0..=4),
        );
        if seen.insert(c) {
            floors.push(c);
        }
    }

    // Save read up front: camper placement (below) needs unlocks, and
    // dressing gates on the placed campers.
    let save = world
        .get_resource::<SaveData>()
        .cloned()
        .unwrap_or_default();
    let unlocked = |gml: usize| -> bool {
        crate::state::menus::race_from_gml_id(gml).is_some_and(|r| save.race_unlocked(r))
    };
    // Campfire at GML (64,64); fixed starters Fish (64,32), Crystal
    // (64,96), Eyes (104,64), Melting (24,64).
    let camp_px = glam::Vec2::new(64.0, 64.0);
    let mut campers: Vec<glam::Vec2> = Vec::new();
    let mut fixed_campers: Vec<(usize, glam::Vec2)> = Vec::new();
    // GML race ids: Fish 1, Crystal 2, Eyes 3, Melting 4.
    for (gml, at) in [
        (1usize, glam::Vec2::new(64.0, 32.0)),
        (2, glam::Vec2::new(64.0, 96.0)),
        (3, glam::Vec2::new(104.0, 64.0)),
        (4, glam::Vec2::new(24.0, 64.0)),
    ] {
        if !unlocked(gml) {
            continue;
        }
        fixed_campers.push((gml, at));
        campers.push(at);
    }
    // Plant (5) .. Cuz (16), skipping locked; GML scatters with
    // `move_contact_solid(random_angle, 32+iter*2+random(32)+...)`
    // until 32px clear of every camper. The port walks the same
    // distance law off the fixed stream and clamps onto floors
    // (no physics here; contact-slide is renderer/collision-side).
    // BigDog (13) keeps its four `PortalClear` dressings.
    let floor_px: Vec<glam::Vec2> = floors
        .iter()
        .map(|(cx, cy)| {
            glam::Vec2::new(
                *cx as f32 * crate::comps_a::TILE + crate::comps_a::TILE * 0.5,
                *cy as f32 * crate::comps_a::TILE + crate::comps_a::TILE * 0.5,
            )
        })
        .collect();
    let nearest_floor = |at: glam::Vec2| -> glam::Vec2 {
        floor_px
            .iter()
            .min_by(|a, b| {
                a.distance_squared(at)
                    .partial_cmp(&b.distance_squared(at))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(at)
    };
    // Chicken TV anchor for the TV arm below.
    let mut chicken_at: Option<glam::Vec2> = None;
    // BigDog sleeper anchor for the four-clear arm below.
    let mut bigdog_at: Option<glam::Vec2> = None;
    let mut scattered: Vec<(usize, glam::Vec2)> = Vec::new();
    let mut scatter_clears: Vec<glam::Vec2> = Vec::new();
    for gml in 5..=16usize {
        if !unlocked(gml) {
            continue;
        }
        // GML starts scattered campers at the campfire.
        let mut at = camp_px;
        // GML `scrCampfireMenuCreate:44-62` scatter verbatim:
        // up to 50 placement tries, then up to 50 more that
        // each pop a `PortalClear` at the failed spot (blasting
        // room in crowded seeds) before giving up at 100.
        for iter in 0..100usize {
            let dist = 32.0
                + iter as f32 * 2.0
                + rng.random_range(0.0..32.0)
                + rng.random_range(0.0..64.0) * rng.random_range(0.0..1.0);
            let ang = rng.random_range(0.0..std::f32::consts::TAU);
            let cand = camp_px + glam::Vec2::from_angle(ang) * dist;
            let cand = nearest_floor(cand);
            let clear = campers.iter().all(|c| c.distance(cand) >= 32.0);
            // GML chicken arm also keeps 16px above her head clear.
            let chicken_clear = gml != 9
                || campers
                    .iter()
                    .all(|c| (*c - (cand + glam::Vec2::new(0.0, -32.0))).length() >= 16.0);
            if clear && chicken_clear {
                at = cand;
                break;
            }
            at = cand;
            if iter >= 50 {
                scatter_clears.push(cand);
            }
        }
        at = nearest_floor(at);
        scattered.push((gml, at));
        campers.push(at);
        if gml == 9 {
            chicken_at = Some(at);
        }
        if gml == 13 {
            bigdog_at = Some(at);
        }
    }

    // `MenuGen/Alarm_1` dressing verbatim: per floor `random(6)<1`, then
    // `irandom(21)` — nonzero rolls a NightCactus, zero rolls a
    // TopDecalNightDesert. GML gates the cactus on
    // `distance_to_object(CampChar)>24 && distance_to_object(NightCactus)>16`
    // against the live actors — Alarm_1 runs after ALL campers
    // (fixed + scattered) are placed, so the port gates on the full
    // `campers` list too. Floors are dressed in plan order so the fixed
    // stream matches.
    let mut cacti: Vec<glam::Vec2> = Vec::new();
    let mut decals: Vec<glam::Vec2> = Vec::new();
    for (cx, cy) in &floors {
        if rng.random_range(0.0..6.0) >= 1.0 {
            continue;
        }
        let at = glam::Vec2::new(
            *cx as f32 * crate::comps_a::TILE + crate::comps_a::TILE * 0.5,
            *cy as f32 * crate::comps_a::TILE + crate::comps_a::TILE * 0.5,
        );
        if rng.random_range(0..22) != 0 {
            let near_camper = campers.iter().any(|s| s.distance(at) <= 24.0);
            let near_cactus = cacti.iter().any(|c| c.distance(at) <= 16.0);
            if !near_camper && !near_cactus {
                cacti.push(at);
            }
        } else {
            decals.push(at);
        }
    }

    let camp_run = Run {
        area: AreaId::Campfire,
        ..Default::default()
    };
    let mut plan = LevelPlan {
        floor_cells: floors.clone(),
        wall_cells: std::collections::HashSet::new(),
        small_walls: Vec::new(),
        bones: Vec::new(),
        bone_sprite: "images/sprNightBones.png",
        details: Vec::new(),
        props: Vec::new(),
        chests: Vec::new(),
        enemies: Vec::new(),
        boss: None,
        boss_count: 1,
        styleb: false,
    };
    worldgen::build_walls(&camp_run, &floors, &mut plan);
    let floor_set: std::collections::HashSet<(i32, i32)> =
        plan.floor_cells.iter().copied().collect();
    world.resource_scope(|world, mut mask: Mut<FloorMask>| {
        world.resource_scope(|world, catalog: Mut<AnimCatalog>| {
            let mut commands = world.commands();
            *mask = build_floor_mask(&plan);
            spawn_wall_tiles(
                &mut commands,
                plan.wall_cells.into_iter().collect(),
                &floor_set,
            );
            let mut commands = world.commands();
            for at in cacti {
                spawn_prop_sim(
                    &mut commands,
                    &catalog,
                    &camp_run,
                    crate::worldgen::PropKind::NightCactus,
                    at,
                );
            }
            // `Alarm_1` topdecal half: night-desert top decals ride the
            // `GroundDecal` prop (art resolves to the night strip
            // renderer-side via the Campfire area).
            for at in decals {
                spawn_prop_sim(
                    &mut commands,
                    &catalog,
                    &camp_run,
                    crate::worldgen::PropKind::GroundDecal,
                    at,
                );
            }
            // `scrCampfireMenuCreate` actors verbatim (positions in world
            // px, same as GML): Campfire (64,64) + LogMenu (64,32), four
            // fixed starters, scattered Plant..Cuz, chicken TV, BigDog
            // sleepers. Positions were computed above (dressing gates on
            // them); only unlocked races got campers (locked return
            // `noone` in GML). Every camper pops a `PortalClear`
            // (`MenuGen/Alarm_1`).
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                crate::comps_b::TitleCampfire,
                Pos(camp_px),
            ));
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                crate::comps_b::TitleLogMenu,
                Pos(glam::Vec2::new(64.0, 32.0)),
            ));
            // GML race ids: Fish 1, Crystal 2, Eyes 3, Melting 4.
            for (gml, at) in fixed_campers {
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    crate::comps_b::TitleCampChar {
                        race_gml: gml,
                        fixed: true,
                    },
                    Pos(at),
                ));
            }
            // Scattered Plant..Cuz from the precomputed positions
            // above (the distance law ran there, against the full
            // camper list, like GML).
            for (gml, at) in scattered {
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    crate::comps_b::TitleCampChar {
                        race_gml: gml,
                        fixed: false,
                    },
                    Pos(at),
                ));
            }
            // Scatter-overflow clears (one per failed try past 50).
            for clear_at in scatter_clears {
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    PortalClear {
                        timer: crate::time::GTimer::from_seconds(
                            5.0 / 30.0,
                            crate::time::TimerMode::Once,
                        ),
                    scale: 1.0,
                    },
                    Pos(clear_at),
                ));
            }
            // Chicken TV + 2 half-scale clears (`scrCampfireMenuCreate`
            // chicken arm verbatim: TV at `x + orandom(2), y + orandom(4)
            // - 32`, clears at `(x, y + 16)` and `(x, y)` rescaled 0.5).
            if let Some(at) = chicken_at {
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    crate::comps_b::TitleTv,
                    Pos(glam::Vec2::new(
                        at.x + rng.random_range(-2.0..2.0),
                        at.y - 32.0 + rng.random_range(-4.0..4.0),
                    )),
                ));
                for off in [glam::Vec2::new(0.0, 16.0), glam::Vec2::ZERO] {
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        PortalClear {
                            timer: crate::time::GTimer::from_seconds(
                                5.0 / 30.0,
                                crate::time::TimerMode::Once,
                            ),
                            scale: 0.5,
                        },
                        Pos(at + off),
                    ));
                }
            }
            if let Some(at) = bigdog_at {
                for off in [
                    glam::Vec2::new(-32.0, 0.0),
                    glam::Vec2::new(32.0, 0.0),
                    glam::Vec2::new(0.0, -32.0),
                    glam::Vec2::new(0.0, 32.0),
                ] {
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        PortalClear {
                            timer: crate::time::GTimer::from_seconds(
                                5.0 / 30.0,
                                crate::time::TimerMode::Once,
                            ),
                        scale: 1.0,
                        },
                        Pos(at + off),
                    ));
                }
            }
            for at in campers {
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    PortalClear {
                        timer: crate::time::GTimer::from_seconds(
                            5.0 / 30.0,
                            crate::time::TimerMode::Once,
                        ),
                        scale: 1.0,
                    },
                    Pos(at),
                ));
            }
        })
    });
    world.flush();
    reset_menu_room_resources(world);
}

#[cfg(test)]
mod verbatim_title_to_first_level {
    use super::*;
    use crate::comps_b::Enemy;
    use crate::data::RaceId;

    #[test]
    fn fresh_save_holds_fish_and_crystal() {
        let save = SaveData::default();
        assert!(save.race_unlocked(RaceId::Fish));
        assert!(save.race_unlocked(RaceId::Crystal));
        assert!(save.race_unlocked(RaceId::Random));
        assert!(!save.race_unlocked(RaceId::Plant));
    }

    #[test]
    fn title_cursor_defaults_to_random_head() {
        assert_eq!(crate::state::menus::MenuState::default().title_cursor, 0);
        let save = SaveData::default();
        let roster = crate::state::menus::visible_roster(Some(&save));
        assert_eq!(roster.len(), 14);
        assert!(roster.contains(&RaceId::Crystal));
    }

    #[test]
    fn race_starters_match_gml_table() {
        use crate::data::{
            WEAPON_CHICKEN_SWORD, WEAPON_DOG_SPIN_ATTACK, WEAPON_GOLDEN_FROG_PISTOL,
            WEAPON_GOLDEN_REVOLVER, WEAPON_REVOLVER, WEAPON_ROGUE_RIFLE, WEAPON_RUSTY_REVOLVER,
            race_starter_weapon,
        };
        assert_eq!(race_starter_weapon(RaceId::Fish), WEAPON_REVOLVER);
        assert_eq!(race_starter_weapon(RaceId::Venuz), WEAPON_GOLDEN_REVOLVER);
        assert_eq!(race_starter_weapon(RaceId::Cuz), WEAPON_GOLDEN_REVOLVER);
        assert_eq!(race_starter_weapon(RaceId::Chicken), WEAPON_CHICKEN_SWORD);
        assert_eq!(race_starter_weapon(RaceId::Rogue), WEAPON_ROGUE_RIFLE);
        assert_eq!(race_starter_weapon(RaceId::BigDog), WEAPON_DOG_SPIN_ATTACK);
        assert_eq!(race_starter_weapon(RaceId::Skeleton), WEAPON_RUSTY_REVOLVER);
        assert_eq!(race_starter_weapon(RaceId::Frog), WEAPON_GOLDEN_FROG_PISTOL);
    }

    #[test]
    fn fresh_loadouts_deal_race_starters() {
        let save = SaveData::default();
        assert_eq!(
            resolve_run_loadout(&save, RaceId::Chicken).equipped[0],
            crate::data::WEAPON_CHICKEN_SWORD
        );
        assert_eq!(
            resolve_run_loadout(&save, RaceId::Rogue).equipped[0],
            crate::data::WEAPON_ROGUE_RIFLE
        );
        assert_eq!(
            resolve_run_loadout(&save, RaceId::Frog).equipped[0],
            crate::data::WEAPON_GOLDEN_FROG_PISTOL
        );
        // Locked Skeleton falls back to Melting (starter revolver either way).
        assert_eq!(
            resolve_run_loadout(&save, RaceId::Skeleton).equipped[0],
            crate::data::WEAPON_REVOLVER
        );
    }

    #[test]
    fn start_ammo_sums_dual_wield_and_caps() {
        // Steroids dual revolvers stack 32*3 per gun (GML gives each gun).
        let ammo = starting_ammo_for(
            &[WeaponId::REVOLVER, WeaponId::REVOLVER, WeaponId::NONE],
            RaceId::Steroids,
            CrownKind::None,
        );
        assert_eq!(ammo[AmmoKind::Bullets as usize], 192);
        // Melee start grants nothing.
        let ammo = starting_ammo_for(
            &[
                crate::data::WEAPON_CHICKEN_SWORD,
                WeaponId::NONE,
                WeaponId::NONE,
            ],
            RaceId::Chicken,
            CrownKind::None,
        );
        assert_eq!(ammo, [0; MAX_AMMO_TYPES]);
    }

    #[test]
    fn random_roll_skips_bigdog_and_locked() {
        let save = SaveData::default();
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let race = roll_random_race(&save, RaceId::Random, &mut rng);
            assert_ne!(race, RaceId::Random);
            assert_ne!(race, RaceId::BigDog);
            assert!(save.race_unlocked(race));
        }
        // Non-random passes through untouched.
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(
            roll_random_race(&save, RaceId::Plant, &mut rng),
            RaceId::Plant
        );
    }

    #[test]
    fn skeleton_spread_matches_gml_accuracy() {
        let save = SaveData::default();
        let mut save = save;
        save.race_loadout_mut(RaceId::Skeleton).unlocked = true;
        let loadout = resolve_run_loadout(&save, RaceId::Skeleton);
        let bundle = build_player_bundle(RaceId::Skeleton, &loadout);
        assert_eq!(bundle.player.spread_mult, 1.5);
        let loadout = resolve_run_loadout(&save, RaceId::Steroids);
        let bundle = build_player_bundle(RaceId::Steroids, &loadout);
        assert_eq!(bundle.player.spread_mult, 1.8);
    }

    #[test]
    fn tutorial_first_level_is_small_and_empty() {
        let mut world = World::new();
        world.insert_resource(SaveData::default());
        world.insert_resource(SelectedCharacter(RaceId::Fish));
        setup_run_with_seed(&mut world, 1234);
        let run = world.resource::<Run>();
        assert!(run.tutorial);
        assert!(world.query::<&Enemy>().iter(&world).next().is_none());
        let mask = world.resource::<FloorMask>();
        assert!(!mask.cells.is_empty() && mask.cells.len() < 40);
    }

    #[test]
    fn chicken_first_level_starts_with_sword_and_bandits() {
        let mut world = World::new();
        let save = SaveData {
            total_runs: 1,
            ..SaveData::default()
        };
        world.insert_resource(save);
        world.insert_resource(SelectedCharacter(RaceId::Chicken));
        setup_run_with_seed(&mut world, 4242);
        let run = world.resource::<Run>();
        assert!(!run.tutorial);
        let mut invs: Vec<_> = world
            .query::<&Inventory>()
            .iter(&world)
            .map(|i| i.weapons[0])
            .collect();
        assert_eq!(invs.pop(), Some(crate::data::WEAPON_CHICKEN_SWORD));
        assert!(world.query::<&Enemy>().iter(&world).count() > 0);
    }

    #[test]
    fn title_builds_campfire_backdrop() {
        let mut world = World::new();
        setup_title_campfire(&mut world);
        assert!(world.resource::<FloorMask>().cells.len() > 40);
        assert!(
            world
                .query_filtered::<Entity, With<WallTile>>()
                .iter(&world)
                .count()
                > 0
        );
    }

    #[test]
    fn title_spawns_campfire_actors_verbatim() {
        use crate::comps_b::{TitleCampChar, TitleCampfire, TitleLogMenu};
        let mut world = World::new();
        world.insert_resource(SaveData::default());
        setup_title_campfire(&mut world);
        // GML `scrCampfireMenuCreate`: exactly one Campfire + LogMenu.
        assert_eq!(
            world
                .query_filtered::<Entity, With<TitleCampfire>>()
                .iter(&world)
                .count(),
            1
        );
        assert_eq!(
            world
                .query_filtered::<Entity, With<TitleLogMenu>>()
                .iter(&world)
                .count(),
            1
        );
        // Fresh save unlocks Fish + Crystal: both fixed starters exist.
        let mut campers: Vec<usize> = world
            .query::<&TitleCampChar>()
            .iter(&world)
            .map(|c| c.race_gml)
            .collect();
        campers.sort_unstable();
        assert!(campers.contains(&1), "Fish camper missing: {:?}", campers);
        assert!(
            campers.contains(&2),
            "Crystal camper missing: {:?}",
            campers
        );
        // Campfire sits at GML (64,64).
        let mut fires: Vec<glam::Vec2> = world
            .query::<(&TitleCampfire, &crate::spatial::Pos)>()
            .iter(&world)
            .map(|(_, p)| p.0)
            .collect();
        assert_eq!(fires.pop(), Some(glam::Vec2::new(64.0, 64.0)));
    }

    #[test]
    fn first_floor_trims_chests_to_one_each() {
        use crate::comps_b::{ChestKind, Pickup, PickupKind, RadChestContainer};
        let mut world = World::new();
        world.insert_resource(SaveData {
            total_runs: 1,
            ..SaveData::default()
        });
        world.insert_resource(SelectedCharacter(RaceId::Fish));
        setup_run_with_seed(&mut world, 4242);
        let chest_kind = |p: &Pickup| -> Option<ChestKind> {
            match p.kind {
                PickupKind::Chest(k) => Some(k),
                _ => None,
            }
        };
        let weapons = world
            .query::<&Pickup>()
            .iter(&world)
            .filter_map(chest_kind)
            .filter(|k| *k == ChestKind::Weapon)
            .count();
        let ammos = world
            .query::<&Pickup>()
            .iter(&world)
            .filter_map(chest_kind)
            .filter(|k| *k == ChestKind::Ammo)
            .count();
        let rads = world
            .query_filtered::<Entity, With<RadChestContainer>>()
            .iter(&world)
            .count()
            + world
                .query::<&Pickup>()
                .iter(&world)
                .filter_map(chest_kind)
                .filter(|k| {
                    matches!(
                        *k,
                        ChestKind::Rad
                            | ChestKind::RadBig
                            | ChestKind::RadMaggot
                            | ChestKind::Health
                            | ChestKind::Rogue
                    )
                })
                .count();
        // GML `scrPopChests` verbatim: 1 base survivor per kind (mimic
        // rolls can only remove, never add, on floor 1).
        assert!(weapons <= 1, "weapons: {}", weapons);
        assert!(ammos <= 1, "ammos: {}", ammos);
        assert!(rads <= 1, "rads: {}", rads);
    }

    #[test]
    fn random_crown_roll_respects_unlocks() {
        use rand::{SeedableRng, rngs::StdRng};
        // Fresh Fish holds no real crowns: 50 locked rolls → none.
        let save = SaveData::default();
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(
            roll_random_crown(&save, RaceId::Fish, &mut rng),
            CrownKind::None
        );
        // All crowns open: every roll hits, never none.
        let mut save = SaveData::default();
        for gml in 2..=13u8 {
            save.unlock_crown(RaceId::Fish, gml);
        }
        let mut rng = StdRng::seed_from_u64(7);
        assert_ne!(
            roll_random_crown(&save, RaceId::Fish, &mut rng),
            CrownKind::None
        );
    }

    #[test]
    fn background_colors_match_gml_area_table() {
        use crate::data::AreaId;
        let px = |hex: u32| -> [f32; 4] {
            [
                ((hex >> 16) & 0xFF) as f32 / 255.0,
                ((hex >> 8) & 0xFF) as f32 / 255.0,
                (hex & 0xFF) as f32 / 255.0,
                1.0,
            ]
        };
        assert_eq!(
            crate::render::background_color(AreaId::Desert),
            px(0xaf8f6a)
        );
        assert_eq!(
            crate::render::background_color(AreaId::Campfire),
            px(0x6a7aaf)
        );
        assert_eq!(
            crate::render::background_color(AreaId::Palace),
            px(0x611d24)
        );
        assert_eq!(crate::render::background_color(AreaId::Labs), px(0x091c20));
    }

    #[test]
    fn boot_level_entry_and_recontinue_laws() {
        use crate::state::{LevelEntry, choose_level_entry, recontinue_deletes_save};
        // No points → GenCont (first level); any draft opens LevCont.
        assert_eq!(choose_level_entry(0, 0, 0, false), LevelEntry::GenCont);
        assert_eq!(choose_level_entry(1, 0, 0, false), LevelEntry::LevCont);
        assert_eq!(choose_level_entry(0, 1, 0, false), LevelEntry::LevCont);
        assert_eq!(choose_level_entry(0, 0, 1, false), LevelEntry::LevCont);
        // Patience continuation suppresses the skill arm only.
        assert_eq!(choose_level_entry(1, 0, 0, true), LevelEntry::GenCont);
        assert_eq!(choose_level_entry(0, 0, 1, true), LevelEntry::LevCont);
        // Recontinue cap: past 2 the save is deleted.
        assert!(!recontinue_deletes_save(2));
        assert!(recontinue_deletes_save(3));
    }

    #[test]
    fn title_anim_tick_matches_menu_other11() {
        use crate::state::menus::{MenuState, tick_title_anim};
        let mut world = World::new();
        world.insert_resource(MenuState {
            portrait_offsets: [180.0, 0.0, 0.0, 0.0],
            textappear: [2.0, 0.0, 0.0, 0.0],
            splatindex: 0.0,
            loadout_open: true,
            loadout_frame: 0.0,
            ..MenuState::default()
        });
        world.insert_resource(repame_sim::SimTime {
            delta_secs: 1.0 / 30.0,
            ..Default::default()
        });
        tick_title_anim(&mut world);
        let menu = world.resource::<MenuState>();
        assert_eq!(menu.portrait_offsets[0], 90.0);
        assert_eq!(menu.textappear[0], 1.0);
        assert_eq!(menu.splatindex, 0.4);
        assert_eq!(menu.loadout_frame, 1.0);
    }

    /// Reported bug verbatim: dying then clicking MENU must not leave
    /// the previous game's background over the title screen. GML
    /// `scrGameRestart(true)` destroys the session and restarts the
    /// room, so the logo menu rebuilds over an empty campfire — never
    /// over the dead run's floor.
    #[test]
    fn menu_after_death_clears_run_world() {
        use crate::audio::UiAction;
        use crate::state::AppState;
        use crate::state::menus::{MenuState, apply_menu_action};
        let mut world = World::new();
        world.insert_resource(SaveData {
            total_runs: 1,
            ..SaveData::default()
        });
        world.insert_resource(SelectedCharacter(RaceId::Fish));
        setup_run_with_seed(&mut world, 4242);
        let run_cells = world.resource::<FloorMask>().cells.len();
        assert!(run_cells > 0);
        world.resource_mut::<Run>().game_over = true;
        world.insert_resource(AppState::InGame);
        world.init_resource::<MenuState>();
        apply_menu_action(&mut world, UiAction::ConfirmPause(0));
        assert_eq!(
            *world.resource::<AppState>(),
            AppState::MainMenu,
            "MENU must land on the logo menu"
        );
        assert!(
            !world
                .query::<&crate::comps_b::Enemy>()
                .iter(&world)
                .next()
                .is_some(),
            "dead-run enemies draw behind the menu"
        );
        assert!(
            world.resource::<FloorMask>().cells.is_empty(),
            "dead-run floor draws behind the logo menu"
        );
        let run = world.resource::<Run>();
        assert_eq!(run.area, crate::data::AreaId::Campfire);
        assert!(!run.game_over);
    }

    /// Pause-MENU mid-run takes the same clean-room path (GML
    /// `scrGameRestart(true)` regardless of death): the logo menu sits
    /// over the empty room, and PLAY rebuilds the campfire from there.
    #[test]
    fn pause_menu_mid_run_clears_world() {
        use crate::audio::UiAction;
        use crate::state::AppState;
        use crate::state::menus::{MenuState, apply_menu_action};
        let mut world = World::new();
        world.insert_resource(SaveData {
            total_runs: 1,
            ..SaveData::default()
        });
        world.insert_resource(SelectedCharacter(RaceId::Fish));
        setup_run_with_seed(&mut world, 4242);
        world.insert_resource(AppState::InGame);
        world.init_resource::<MenuState>();
        apply_menu_action(&mut world, UiAction::ConfirmPause(0));
        assert_eq!(*world.resource::<AppState>(), AppState::MainMenu);
        assert!(
            world
                .query::<&crate::comps_b::Enemy>()
                .iter(&world)
                .next()
                .is_none(),
            "run enemies survive pause-MENU"
        );
        assert_eq!(world.resource::<Run>().area, crate::data::AreaId::Campfire);
    }

    /// PLAY from the logo menu rebuilds the campfire title room (GML
    /// `PlayButton` → `MenuGen`): entering Title never inherits the
    /// previous room's instances.
    #[test]
    fn title_entry_rebuilds_campfire_room() {
        use crate::audio::UiAction;
        use crate::comps_b::TitleCampfire;
        use crate::state::AppState;
        use crate::state::menus::apply_menu_action;
        let mut world = World::new();
        world.insert_resource(SaveData::default());
        world.insert_resource(SelectedCharacter(RaceId::Fish));
        world.insert_resource(AppState::MainMenu);
        apply_menu_action(&mut world, UiAction::MainMenuPlay);
        assert_eq!(*world.resource::<AppState>(), AppState::Title);
        assert_eq!(
            world
                .query_filtered::<Entity, With<TitleCampfire>>()
                .iter(&world)
                .count(),
            1
        );
        assert_eq!(world.resource::<Run>().area, crate::data::AreaId::Campfire);
    }

    /// Transition guard table (checked against GML per screen).
    /// GML law per cover screen:
    /// - Loading (`GenCont/Draw_0`): `scrDrawSpiral()` (opaque clear) +
    ///   GENERATING + tip + roadmap. No world, no HUD, no menu chrome.
    /// - Mutation/ultra offer (`LevCont/Draw_0`): `scrDrawSpiral()`
    ///   (opaque clear) + offer title/subtitle + icons. No world, no
    ///   HUD; the offer chrome is `LevCont`'s own (the port's Mutation
    ///   overlay), not the campfire `Menu` chrome.
    /// - Mid-run floor transition (same `GenCont` room, `room_restart`
    ///   already destroyed the old room): spiral + text only.
    /// - Title (`Menu/Draw_0`): spiral remnant transparently (no clear)
    ///   UNDER camp + pods + portraits. World ON, menu chrome ON, HUD
    ///   bars off (TopCont draws no HUD in the `MenuGen` room).
    /// The composer expresses this as three gates in `App::view`
    /// (`generation_screen`, `loading_cover`/`cover_chrome_off`,
    /// `bg_alpha`) — this test pins the gate inputs per screen so a
    /// future gate edit must keep all four screens exact.
    #[test]
    fn transition_cover_law_matches_gml_per_screen() {
        use crate::state::AppState;
        use crate::{MenuOverlay, menu_overlay_kind};
        use crate::comps_a::{PendingMutation, PendingUltra};
        use crate::comps_b::FloorTransition;
        use crate::state::OverlayMenu;
        use crate::state::menus::MenuState;
        let menu = MenuState::default();
        let overlay = OverlayMenu::None;
        let cover_of = |state: AppState,
                        overlay: OverlayMenu,
                        menu: &MenuState,
                        world: &World|
         -> (bool, bool, Option<MenuOverlay>) {
            let kind = menu_overlay_kind(state, overlay, menu, false);
            let ft = world
                .get_resource::<FloorTransition>()
                .is_some_and(|f| f.active);
            let pending = world.get_resource::<PendingMutation>().is_some()
                || world.get_resource::<PendingUltra>().is_some();
            let generation_screen = matches!(
                kind,
                Some(MenuOverlay::Loading) | Some(MenuOverlay::Mutation)
            ) || ft;
            let cover_chrome_off = matches!(
                kind,
                Some(MenuOverlay::Loading) | Some(MenuOverlay::Mutation)
            ) || ft;
            let bg_opaque = match state {
                AppState::Title => false,
                AppState::Splash | AppState::MainMenu | AppState::Loading => true,
                AppState::InGame => ft || pending,
            };
            assert_eq!(
                generation_screen, cover_chrome_off,
                "sprite gate and text gate must agree"
            );
            (generation_screen, bg_opaque, kind)
        };
        let fresh = World::new();
        let (is_cover, opaque, kind) =
            cover_of(AppState::Loading, overlay, &menu, &fresh);
        assert!(is_cover && opaque && kind == Some(MenuOverlay::Loading));
        // Offer path: `menu_overlay_kind` reads the `mutation_count`
        // mirror, which `tick_ingame_menu` syncs from `Pending*` at the
        // head of the same schedule tick — so a freshly inserted offer
        // reads live play until the mirror runs. Mirror it here the way
        // the schedule does, then the cover must hold.
        let mut offer = World::new();
        offer.insert_resource(PendingMutation {
            choices: vec![crate::ids_part::MutationId::RhinoSkin],
        });
        offer.insert_resource(crate::state::menus::MenuState::default());
        {
            let (count, is_ultra) =
                if let Some(ultra) = offer.get_resource::<PendingUltra>() {
                    (ultra.choices.len(), true)
                } else if let Some(pending) = offer.get_resource::<PendingMutation>() {
                    (pending.choices.len(), false)
                } else {
                    (0, false)
                };
            if let Some(mut m) = offer.get_resource_mut::<crate::state::menus::MenuState>() {
                crate::state::menus::apply_mutation_mirror(&mut m, count, is_ultra);
            }
        }
        let offer_menu = offer.resource::<crate::state::menus::MenuState>().clone();
        let (is_cover, opaque, kind) =
            cover_of(AppState::InGame, OverlayMenu::None, &offer_menu, &offer);
        assert!(is_cover, "mirrored offer must read as a generation cover");
        assert!(opaque, "pending offer is an opaque cover");
        assert_eq!(kind, Some(MenuOverlay::Mutation));
        let mut ft = World::new();
        ft.insert_resource(FloorTransition {
            active: true,
            ..Default::default()
        });
        let (is_cover, opaque, _) =
            cover_of(AppState::InGame, OverlayMenu::None, &menu, &ft);
        assert!(is_cover && opaque);
        let (is_cover, opaque, kind) =
            cover_of(AppState::Title, OverlayMenu::None, &menu, &fresh);
        assert!(!is_cover && !opaque && kind == Some(MenuOverlay::Title));
    }

    /// Reported bug verbatim: the loading screen must show the vortex,
    /// not the previous room. GML `room_restart` hands `GenCont` a fresh
    /// room, so GENERATING draws over spiral + black only. Entering
    /// Loading therefore tears down session entities + the floor mask
    /// up front (bevy `teardown_game` on InGame exit); `setup_run`
    /// rebuilds at the end of the load. Covers both the fresh-run path
    /// (Title camp) and RETRY after death (dead run).
    #[test]
    fn loading_enter_clears_stale_world() {
        use crate::state::{AppState, goto_state};
        for seed in [4242u64, 777] {
            let mut world = World::new();
            world.insert_resource(SaveData {
                total_runs: 1,
                ..SaveData::default()
            });
            world.insert_resource(SelectedCharacter(RaceId::Fish));
            setup_run_with_seed(&mut world, seed);
            assert!(!world.resource::<FloorMask>().cells.is_empty());
            world.resource_mut::<Run>().game_over = true;
            world.insert_resource(AppState::InGame);
            goto_state(&mut world, AppState::Loading);
            assert_eq!(*world.resource::<AppState>(), AppState::Loading);
            assert!(
                world
                    .query::<&crate::comps_b::Enemy>()
                    .iter(&world)
                    .next()
                    .is_none(),
                "dead-run enemies render through the loading screen"
            );
            assert!(
                world.resource::<FloorMask>().cells.is_empty(),
                "dead-run floor renders through the loading screen"
            );
            // Run selection resources survive for setup_run at load end.
            assert!(world.get_resource::<SelectedCharacter>().is_some());
            assert!(world.get_resource::<SaveData>().is_some());
        }
    }

    /// Fresh Title → Loading drops the campfire camp the same way (GML
    /// GO → `room_restart` destroys the camp before `GenCont` builds).
    #[test]
    fn loading_enter_clears_title_camp() {
        use crate::state::{AppState, goto_state};
        let mut world = World::new();
        world.insert_resource(SaveData::default());
        world.insert_resource(AppState::Title);
        setup_title_campfire(&mut world);
        assert!(!world.resource::<FloorMask>().cells.is_empty());
        goto_state(&mut world, AppState::Loading);
        assert!(world.resource::<FloorMask>().cells.is_empty());
        assert!(
            world
                .query_filtered::<Entity, With<crate::comps_b::TitleCampfire>>()
                .iter(&world)
                .count()
                == 0
        );
    }
}
