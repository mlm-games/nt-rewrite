//! Level/mutation/portal progression. Port of
//! `progression.rs` GML port minus the three deferred
//! items (`setup_run`, `begin_between_floor_skill_picks` as a public
//! system, `starting_ammo_for` — reasons in the module docs below).
//!
//! Render split: bursts stay (sim-side `Particle` spawns, deaths.rs
//! parity); sprite strips, anchors, flips, portal deco sprites, hurt
//! anims, and loading-tip UI text are omitted (render/UI phase), never
//! stubbed. Key input (`Digit1..4`) does not exist headless: mutation
//! picks arrive via the `MutationChoice` resource (tests/UI write it).
//!
//! Deferred, with reason:
//! - `setup_run`: needs save-loadout (`race_loadout`, skins) + weapons
//!   table + world spawn; lands with the save phase.
//! - `begin_between_floor_skill_picks` (public system): needs
//!   mutation-pick UI state; lands with the UI phase. Its resource flow
//!   (pause + `PendingMutation`/`PendingUltra`) is ported as a private
//!   helper because `tick_portal_suck` / `handle_mutation_choice` call
//!   it.
//! - `starting_ammo_for`: needs the `WEAPONS` ammo table; lands with
//!   the weapons phase (ammo mapping now rides `weapon_runtime`).
//! - `tick_floor_transition` stage 2 runs the full bevy law: fresh
//!   plan + Open Mind bonus + `setup::spawn_level` entity spawn.
//! - Save writes: the sim has no `SaveManager`; `flush_dirty_save*`
//!   only manage the `SaveDirty` flag (the write itself lands with the
//!   save phase).

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::audio::{AudioCue, GameAudio, QueuedReactiveCue, ReactiveCue};
use crate::combat::queue_enemy_spawn;
use crate::comps_a::{
    Euphoria, FloorMask, FloorStarted, GameCleanup, Health, HeavyHeart, Inventory, LevelCleanup,
    MutationChoice, OpenMind, PendingMutation, PendingUltra, Player, Projectile, RaceState, Run,
    SaveDirty, ScarierFace, Team, Toast, Velocity,
};
use crate::comps_b::{
    ChestKind, CrownPedestal, Enemy, FloorTransition, GroundPhysics, LoopTransition, OpenedChest,
    Pickup, PickupCurse, PickupKind, Portal, PortalCarriedWeapons, PortalClear, PortalClosing,
    PortalPhase, PortalShock, PortalState, PortalSucking, Prop, PropSprites, RadChestContainer,
    SecretEntrance, SitZone, ThroneSit,
};
use crate::data::{
    AmmoKind, AreaId, CrownKind, MutationId, RaceId, SecretTarget, UltraMutationId, ammo_max,
    ammo_pickup_amount, area_for_floor, route_coordinates,
};
use crate::effects::{
    ChromaticAberration, FlashWhite, SlowMotion, chromatic_pulse, flash_white, slow_motion,
    spawn_burst,
};
use crate::environment::{PropDeathEffect, spawn_prop_corpse, spawn_prop_death_effect};
use crate::msg::Queue;
use crate::pickups::ProtoChestState;
use crate::savedata_part::SaveData;
use crate::spatial::Pos;
use crate::state::{AppState, Paused};
use crate::time::{GTimer, TimerMode};

// ---------------------------------------------------------------------------
// Mutation data (byte-exact tables from bevy `content.rs`).
// ---------------------------------------------------------------------------

pub const ALL_MUTATIONS: [MutationId; 28] = [
    MutationId::RhinoSkin,
    MutationId::PlutoniumHunger,
    MutationId::TriggerFingers,
    MutationId::RabbitPaw,
    MutationId::SecondStomach,
    MutationId::ScarierFace,
    MutationId::BoilingVeins,
    MutationId::ImpactWrists,
    MutationId::ExtraFeet,
    MutationId::Bloodlust,
    MutationId::LuckyShot,
    MutationId::GammaGuts,
    MutationId::BackMuscle,
    MutationId::Euphoria,
    MutationId::LongArms,
    MutationId::Stress,
    MutationId::EagleEyes,
    MutationId::OpenMind,
    MutationId::StrongSpirit,
    MutationId::SharpTeeth,
    MutationId::LastWish,
    MutationId::BoltMarrow,
    MutationId::Hammerhead,
    MutationId::LaserBrain,
    MutationId::RecycleGland,
    MutationId::ShotgunShoulders,
    MutationId::ThroneButt,
    MutationId::Patience,
];

/// GML mutation display name + description verbatim
/// (`scripts/scrSkills/scrSkills.gml`: `skill_name[]`/`skill_text[]`,
/// ids 1-29): unlocalized defaults, `@`-color tags included — the
/// `draw_text_nt` backend resolves them. `mutation_skill_index`
/// (`hud.rs`) maps these ids to the same 1-29 frames GML uses.
pub fn mutation_name(id: MutationId) -> (&'static str, &'static str) {
    match id {
        MutationId::RhinoSkin => ("RHINO SKIN", "+4 @rMAX HP"),
        MutationId::ExtraFeet => ("EXTRA FEET", "MORE @wSPEED#@sWALK NORMALY ON ALL TERRAIN"),
        MutationId::PlutoniumHunger => (
            "PLUTONIUM HUNGER",
            "ATTRACT @wDROPS@s AND @gRADS@s FROM FURTHER",
        ),
        MutationId::RabbitPaw => ("RABBIT PAW", "MORE @rHP@s AND @yAMMO@s DROPS"),
        MutationId::ThroneButt => ("THRONE BUTT", "UPGRADES YOUR @wSPECIAL ABILITY"),
        MutationId::LuckyShot => ("LUCKY SHOT", "SOME KILLS REGENERATE @yAMMO"),
        MutationId::Bloodlust => ("BLOODLUST", "SOME KILLS REGENERATE @rHP"),
        MutationId::GammaGuts => ("GAMMA GUTS", "@wENEMIES@s TOUCHING YOU TAKE DAMAGE"),
        MutationId::SecondStomach => ("SECOND STOMACH", "MORE @rHP@s FROM MEDKITS"),
        MutationId::BackMuscle => ("BACK MUSCLE", "HIGHER @yAMMO@s MAX"),
        MutationId::ScarierFace => ("SCARIER FACE", "LESS @wENEMY @rHP"),
        MutationId::Euphoria => ("EUPHORIA", "SLOWER @wENEMY@s BULLETS"),
        MutationId::LongArms => ("LONG ARMS", "MORE @wMELEE@s RANGE"),
        MutationId::BoilingVeins => (
            "BOILING VEINS",
            "@wNO DAMAGE@s FROM EXPLOSIONS AND FIRE#WHEN UNDER 4 @rHP",
        ),
        MutationId::ShotgunShoulders => ("SHOTGUN SHOULDERS", "@wSHELLS@s BOUNCE FURTHER"),
        MutationId::RecycleGland => ("RECYCLE GLANDS", "MOST HIT @wBULLETS@s BECOME @yAMMO"),
        MutationId::LaserBrain => ("LASER BRAIN", "@wENERGY@s WEAPONS DEAL MORE @wDAMAGE@s"),
        MutationId::LastWish => ("LAST WISH", "GET FULL @rHEALTH @sAND SOME @yAMMO"),
        MutationId::EagleEyes => ("EAGLE EYES", "BETTER ACCURACY"),
        MutationId::ImpactWrists => ("IMPACT WRISTS", "CORPSES FLY & HIT HARDER"),
        MutationId::BoltMarrow => ("BOLT MARROW", "HOMING @wBOLTS"),
        MutationId::Stress => ("STRESS", "HIGHER RATE OF FIRE#AS @rHP@s GETS LOWER"),
        MutationId::TriggerFingers => ("TRIGGER FINGERS", "KILLS LOWER YOUR RELOAD TIME"),
        MutationId::SharpTeeth => (
            "SHARP TEETH",
            "DAMAGE TAKEN IS DEALT TO#ALL ENEMIES ON SCREEN",
        ),
        MutationId::Patience => ("PATIENCE", "@gMUTATE@s LATER"),
        MutationId::Hammerhead => ("HAMMER HEAD", "BREAK TROUGH LIMITED AMOUNT OF WALLS"),
        MutationId::StrongSpirit => (
            "STRONG SPIRIT",
            "PREVENT DEATH ONCE#RECHARGE AT FULL @rHP@s IN NEXT LEVEL",
        ),
        MutationId::OpenMind => ("OPEN MIND", "EXTRA @wCHESTS@s SPAWN"),
        MutationId::HeavyHeart => ("HEAVY HEART", "MORE WEAPON DROPS"),
    }
}

pub fn ultra_choices_for(race: RaceId) -> Vec<UltraMutationId> {
    match race {
        RaceId::Fish | RaceId::Random => vec![
            UltraMutationId::FishConfiscate,
            UltraMutationId::FishGunWarrant,
        ],
        RaceId::Crystal => vec![
            UltraMutationId::CrystalFortress,
            UltraMutationId::CrystalJuggernaut,
        ],
        RaceId::Eyes => vec![
            UltraMutationId::EyesProjectileStyle,
            UltraMutationId::EyesMonsterStyle,
        ],
        RaceId::Melting => vec![
            UltraMutationId::MeltingBrainCapacity,
            UltraMutationId::MeltingDetachment,
        ],
        RaceId::Plant => vec![UltraMutationId::PlantTrapper, UltraMutationId::PlantKiller],
        RaceId::Venuz => vec![
            UltraMutationId::VenuzGunGod,
            UltraMutationId::VenuzBack2Bizniz,
        ],
        RaceId::Steroids => vec![
            UltraMutationId::SteroidsAmbidextrous,
            UltraMutationId::SteroidsGetArmed,
        ],
        RaceId::Robot => vec![
            UltraMutationId::RobotRefinedTaste,
            UltraMutationId::RobotRegurgitate,
        ],
        RaceId::Chicken => vec![
            UltraMutationId::ChickenHarderToKill,
            UltraMutationId::ChickenDetermination,
        ],
        RaceId::Rebel => vec![
            UltraMutationId::RebelPersonalGuard,
            UltraMutationId::RebelRiot,
        ],
        RaceId::Horror => vec![
            UltraMutationId::HorrorStalker,
            UltraMutationId::HorrorAnomaly,
            UltraMutationId::HorrorMeltdown,
        ],
        RaceId::Rogue => vec![
            UltraMutationId::RoguePortalStrike,
            UltraMutationId::RogueSuperBlastArmor,
        ],
        RaceId::BigDog => vec![
            UltraMutationId::BigDogGuardian,
            UltraMutationId::BigDogHeavyArtillery,
        ],
        RaceId::Skeleton => vec![
            UltraMutationId::SkeletonBloodArmor,
            UltraMutationId::SkeletonNecromancy,
        ],
        RaceId::Frog => vec![
            UltraMutationId::FrogSwampBody,
            UltraMutationId::FrogToxicLord,
        ],
        RaceId::Cuz => vec![UltraMutationId::CuzHoarder, UltraMutationId::CuzEmotional],
    }
}

/// GML `LevCont/Create_0:143-149`: the Destiny crown trims the ultra
/// offer to `1 + scrPlayerCountRace(Race.Horror)` (2 normally, 3 for
/// Horror).
pub fn ultra_choices_for_crown(race: RaceId, destiny: bool) -> Vec<UltraMutationId> {
    let mut choices = ultra_choices_for(race);
    if destiny {
        choices.truncate(if race == RaceId::Horror { 2 } else { 1 });
    }
    choices
}

/// GML ultra display name + description verbatim
/// (`scripts/scrUltras/scrUltras.gml`: `ultr_name[race,tier]` /
/// `ultr_text[race,tier]`, unlocalized defaults, `@` tags included).
/// GML indexes by (race gml id, tier 1-3); each offer follows the
/// source tier order, including Horror's third choice.
pub fn ultra_mutation_name(id: UltraMutationId) -> (&'static str, &'static str) {
    match id {
        UltraMutationId::FishGunWarrant => (
            "GUN WARRANT",
            "INFINITE AMMO THE FIRST 7 SECONDS#AFTER EXITING A @pPORTAL",
        ),
        UltraMutationId::FishConfiscate => ("CONFISCATE", "ENEMIES SOMETIMES DROP CHESTS"),
        UltraMutationId::CrystalFortress => ("FORTRESS", "+6 MAX HP"),
        UltraMutationId::CrystalJuggernaut => ("JUGGERNAUT", "MOVE WHEN SHIELDING"),
        UltraMutationId::EyesMonsterStyle => (
            "MONSTER STYLE",
            "PUSH NEARBY ENEMIES AWAY#WHEN NOT USING TELEKINESIS",
        ),
        UltraMutationId::EyesProjectileStyle => {
            ("PROJECTILE STYLE", "TELEKINESIS HOLDS YOUR PROJECTILES")
        }
        UltraMutationId::MeltingBrainCapacity => ("BRAIN CAPACITY", "BLOW UP LOW HP ENEMIES"),
        UltraMutationId::MeltingDetachment => {
            ("DETACHMENT", "3 MORE MUTATIONS#LOSE HALF OF YOUR HP")
        }
        UltraMutationId::PlantTrapper => ("TRAPPER", "BIG SNARE"),
        UltraMutationId::PlantKiller => ("KILLER", "KILLING SNARED ENEMY SPAWN SAPLINGS"),
        UltraMutationId::VenuzBack2Bizniz => ("BACK 2 BIZNIZ", "FREE POP POP UPGRADE"),
        UltraMutationId::VenuzGunGod => ("IMA GUN GOD", "HIGHER RATE OF FIRE"),
        UltraMutationId::SteroidsAmbidextrous => ("AMBIDEXTROUS", "DOUBLE WEAPONS FROM CHESTS"),
        UltraMutationId::SteroidsGetArmed => ("GET LOADED", "AMMO CHESTS CONTAIN ALL AMMO TYPES"),
        UltraMutationId::RobotRefinedTaste => (
            "REFINED TASTE",
            "HIGH TIER WEAPONS ONLY#AUTO EAT WEAPONS LEFT BEHIND",
        ),
        UltraMutationId::RobotRegurgitate => (
            "REGURGITATE",
            "EATING WEAPONS CAN DROP CHESTS#AUTO EAT WEAPONS LEFT BEHIND",
        ),
        UltraMutationId::ChickenHarderToKill => ("HARDER TO KILL", "KILLS EXTEND BLEED TIME"),
        UltraMutationId::ChickenDetermination => (
            "DETERMINATION",
            "THROWN WEAPONS CAN TELEPORT BACK#TO YOUR SECONDARY SLOT",
        ),
        UltraMutationId::RebelPersonalGuard => (
            "PERSONAL GUARD",
            "START A LEVEL WITH 2 ALLIES#ALL ALLIES HAVE MORE HP",
        ),
        UltraMutationId::RebelRiot => ("RIOT", "DOUBLE ALLY SPAWNS"),
        UltraMutationId::HorrorStalker => ("STALKER", "ENEMIES EXPLODE IN RADIATION ON DEATH"),
        UltraMutationId::HorrorAnomaly => ("ANOMALY", "@pPORTAL@s APPEAR EARLIER"),
        UltraMutationId::HorrorMeltdown => ("MELTDOWN", "DOUBLE @gRAD@s CAPACITY"),
        UltraMutationId::RogueSuperBlastArmor => ("SUPER BLAST ARMOR", "SUPER BLAST ARMOR"),
        UltraMutationId::RoguePortalStrike => (
            "SUPER PORTAL STRIKE",
            "DOUBLE PORTAL STRIKE PICKUPS#AND CAPACITY",
        ),
        UltraMutationId::BigDogHeavyArtillery => ("ULTRA MISSILES", "@wROCKETS SHOOT BULLETS"),
        UltraMutationId::BigDogGuardian => ("ULTRA SPIN", "@wIMPROVED SPIN ATTACK"),
        UltraMutationId::SkeletonBloodArmor => ("REDEMPTION", "BACK IN THE FLESH"),
        UltraMutationId::SkeletonNecromancy => {
            ("DAMNATION", "FAST RECHARGE AFTER#USING BLOOD GAMBLE")
        }
        UltraMutationId::FrogToxicLord => ("INTIMACY", "CONTINUOUSLY SPREAD TOXIC GAS"),
        UltraMutationId::FrogSwampBody => ("DISTANCE", "@gRADIATION@s CREATES TOXIC GAS"),
        UltraMutationId::CuzHoarder => ("ARSENAL", "TWICE AS MANY GUNS"),
        UltraMutationId::CuzQuickSwap => ("QUICK SWAP", "SWAP ABILITY IS NEARLY INSTANT"),
        UltraMutationId::CuzEmotional => ("EMOTIONAL", "TWICE AS MANY @bTEARS@w"),
    }
}

// ---------------------------------------------------------------------------
// Small local helpers (pure ports of bevy helpers owned by other phases).
// ---------------------------------------------------------------------------

/// Deferred floor generation flag (bevy `DeferredFloorGen` parity).
#[derive(Resource, Default)]
pub struct DeferredFloorGen(pub bool);

/// GML `GenCont/Destroy_0:112-124`: entering `area_city` (`FrozenCity`)
/// subarea 1 with `mut_last_wish` seeds that room's generation with an
/// `IceFlower` — the furthest `prop` is `instance_change`d into one, or
/// (no props) a random `enemy` is replaced by one. `IceFlower/Create_0`:
/// `max_hp = 450`, `size = 3`, `name = "FEED"`, `feed = 0`.
/// `setup::spawn_level` reads the flag; the floor transition clears it.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct IceFlowerSeed(pub bool);

/// GML `IceFlower/Create_0.gml:23` `feed = 0`. The player's
/// `press_pick` bumps it (`Player/Collision_IceFlower.gml:17`); at
/// `feed >= 4` the flower opens the jungle (`IceFlower/Step_0.gml:4`).
/// `IceFlower` is a `prop`, so this rides the enemy entity the port spawns
/// for `EnemyKind::IceFlower`.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct IceFlowerFeed(pub u8);

/// Floor-seed hash (bevy `derive_floor_seed` verbatim).
fn derive_floor_seed(prev: u64, floor: u32, area: u8, loop_count: u32) -> u64 {
    let mut x = prev
        .wrapping_add(floor as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(area as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(loop_count as u64 + 1);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Secret-area display name (bevy `SecretTarget::name` parity).
fn secret_name(target: SecretTarget) -> &'static str {
    target.name()
}

/// Floor to return to after leaving a secret area. GML
/// `GameCont/Other_5:64-82` verbatim: the room-end secret exits, and
/// the stale `_is_secret` taken before the reassignment skips the normal
/// advance, so the player lands exactly on that area's last subarea.
/// - pizza sewers -> `area_scrapyards; subarea = 0` (+1 -> 1-1)
/// - oasis / mansion -> `area_scrapyards; subarea = 3`
/// - cursed caves -> `area_city; subarea = 0` (+1 -> 1-1)
/// - jungle -> `area_city; subarea = 3`
fn secret_return_floor(target: SecretTarget, current_floor: u32) -> u32 {
    match target {
        SecretTarget::PizzaSewers => 5,
        SecretTarget::Oasis | SecretTarget::YvMansion => 7,
        SecretTarget::CursedCaves => 9,
        SecretTarget::Jungle => 11,
        SecretTarget::Vault | SecretTarget::CrownVault | SecretTarget::Hq => {
            current_floor.saturating_add(1)
        }
    }
}

fn target_for_secret_area(area: AreaId) -> Option<SecretTarget> {
    match area {
        AreaId::Oasis => Some(SecretTarget::Oasis),
        AreaId::PizzaSewers => Some(SecretTarget::PizzaSewers),
        AreaId::City => Some(SecretTarget::YvMansion),
        AreaId::CursedCaves => Some(SecretTarget::CursedCaves),
        AreaId::Jungle => Some(SecretTarget::Jungle),
        AreaId::Vault => Some(SecretTarget::Vault),
        AreaId::CrownVault => Some(SecretTarget::CrownVault),
        AreaId::HQ => Some(SecretTarget::Hq),
        _ => None,
    }
}

/// Secret/normal floor advance on portal exit (bevy
/// `apply_secret_transition` verbatim, minus engine types).
fn apply_secret_transition(
    run: &mut Run,
    triggers: &mut crate::secrets::SecretTriggers,
    crib: &mut crate::CribTrip,
    save: &mut SaveData,
    dirty: &mut SaveDirty,
) -> Option<SecretTarget> {
    // GML `GameCont/Other_5.gml:24-44`: the crib hop runs before anything
    // else, including the queued secret, and short-circuits the advance.
    crib.fromcrib = false;
    if crib.gocrib {
        crib.gocrib = false;
        if run.area != AreaId::Crib {
            crib.last_area = Some(run.area);
            crib.last_subarea = run.floor_in_area;
        }
        run.area = AreaId::Crib;
        run.floor_in_area = 1;
        run.world = 0;
        run.portal_open = false;
        let prev = run.gen_seed;
        run.gen_seed = derive_floor_seed(prev, run.floor, 107, run.loop_count);
        if crate::savedata_part::try_unlock_race(save, RaceId::Cuz) {
            dirty.0 = true;
        }
        triggers.reset_floor_flags();
        return None;
    }
    // GML `:49-52`: a stage that cannot advance yet holds the room.
    if !crib.can_advance {
        crib.can_advance = true;
        return None;
    }

    // GML `GameCont/Other_5:136` (Room End): `hard += hardmode ? 2 : 1`.
    run.hard += if run.hardmode { 2 } else { 1 };

    if let Some(target) = triggers.take_queued() {
        if matches!(target, SecretTarget::Vault | SecretTarget::CrownVault) {
            triggers.vaults_entered = triggers.vaults_entered.saturating_add(1);
        }
        let (world, floor_in_area) = target.display();
        let prev = run.gen_seed;
        run.area = target.area();
        run.world = world;
        run.floor_in_area = floor_in_area;
        run.portal_open = false;
        run.gen_seed = derive_floor_seed(prev, run.floor, run.area as u8, run.loop_count);
        triggers.last_secret = Some(target);
        triggers.reset_floor_flags();
        return Some(target);
    }

    let previous_secret = target_for_secret_area(run.area);

    // GML `GameCont/Other_5.gml:58-62`: leaving the crib restores the area
    // it was entered from, one subarea in.
    if run.area == AreaId::Crib {
        if let Some(area) = crib.last_area {
            run.area = area;
            run.floor_in_area = crib.last_subarea;
        }
        crib.fromcrib = true;
        // The crib is a detour, so the global floor never moved and the
        // roadmap coordinate is the room we came from.
        run.world = route_coordinates(run.floor).0;
        run.portal_open = false;
        triggers.reset_floor_flags();
        return None;
    }

    if let Some(previous_secret) = previous_secret {
        let floor = secret_return_floor(previous_secret, run.floor);
        run.floor = floor.max(1);
        run.loop_count = (run.floor - 1) / 15;
        let (world, floor_in_area) = route_coordinates(run.floor);
        let prev = run.gen_seed;
        run.world = world;
        run.floor_in_area = floor_in_area;
        run.area = area_for_floor(run.floor, run.loop_count);
        run.portal_open = false;
        run.gen_seed = derive_floor_seed(prev, run.floor, run.area as u8, run.loop_count);
        triggers.reset_floor_flags();
        return None;
    }

    run.floor += 1;
    run.loop_count = (run.floor - 1) / 15;
    let (world, floor_in_area) = route_coordinates(run.floor);
    let prev = run.gen_seed;
    run.world = world;
    run.floor_in_area = floor_in_area;
    run.area = area_for_floor(run.floor, run.loop_count);
    run.portal_open = false;
    run.gen_seed = derive_floor_seed(prev, run.floor, run.area as u8, run.loop_count);
    triggers.reset_floor_flags();
    None
}

/// Loop-portal transition (bevy `try_apply_loop_portal_transition`
/// verbatim, minus engine types). Loop 1 starts at floor 16.
fn try_apply_loop_portal_transition(
    run: &mut Run,
    transition: &mut LoopTransition,
    trauma: &mut Trauma,
) -> bool {
    if !transition.consume_loop_ready() {
        return false;
    }

    let next_loop = run.loop_count + 1;
    run.loop_count = next_loop;
    run.floor = next_loop * 15 + 1;
    run.hard += if run.hardmode { 2 } else { 1 };

    let (world, floor_in_area) = route_coordinates(run.floor);
    let prev = run.gen_seed;
    run.world = world;
    run.floor_in_area = floor_in_area;
    run.area = area_for_floor(run.floor, run.loop_count);
    run.portal_open = false;
    run.gen_seed = derive_floor_seed(prev, run.floor, run.area as u8, run.loop_count);

    transition.last_completed_loop = next_loop;

    trauma.add(0.35);

    true
}

// ---------------------------------------------------------------------------
// Level-ups and mutations.
// ---------------------------------------------------------------------------

/// Spend banked rads into levels (cap 10; level 10 owes an ultra pick,
/// lower levels owe a mutation pick each), toasting + feedback on any
/// gain. `health`/`inv`/`race` are unused in the bevy body, so they are
/// not parameters here.
pub fn check_level_up(
    commands: &mut Commands,
    trauma: &mut Trauma,
    flash: &mut FlashWhite,
    player: &mut Player,
    toast: &mut Toast,
    audio: &GameAudio,
    cues: &mut Queue<AudioCue>,
    pos: glam::Vec2,
) {
    let mut leveled = false;

    while player.rads >= player.next_level_rads && player.level < 10 {
        player.rads -= player.next_level_rads;
        player.level += 1;
        player.next_level_rads = player.level.max(1) * 60;
        leveled = true;

        if player.level >= 10 && player.ultra.is_none() {
            player.ultra_pick_owed = true;
        } else if player.level < 10 {
            player.mutation_picks_owed = player.mutation_picks_owed.saturating_add(1);
        }
    }
    if player.level >= 10 {
        player.rads = player.rads.min(player.next_level_rads.max(1));
    }

    if leveled {
        toast.show(if player.ultra_pick_owed && player.level >= 10 {
            "LEVEL ULTRA!"
        } else {
            "LEVEL UP!"
        });
        level_up_feedback(
            commands,
            trauma,
            flash,
            audio,
            cues,
            pos,
            if player.level >= 10 {
                [1.0, 0.35, 1.0, 1.0]
            } else {
                [0.25, 1.0, 0.25, 1.0]
            },
        );
    }
}

/// Re-arm Strong Spirit at full HP after clearing the area it saved.
pub fn try_recharge_strong_spirit(player: &mut Player, health: &Health) {
    player.try_recharge_strong_spirit(health);
}

/// Queue the loading-screen floor transition (bevy
/// `try_start_pending_floor_gen` parity, including the per-area vortex
/// `SpiralCtl` re-warm the ambience duck keys off).
pub fn try_start_pending_floor_gen(commands: &mut Commands, run: &Run) {
    let tip = pick_loading_tip(run);
    commands.insert_resource(FloorTransition {
        active: true,
        stage: 1,
        timer: GTimer::from_seconds(0.05, TimerMode::Repeating),
        progress: 0.0,
        tip,
    });
    commands.insert_resource(crate::vortex::SpiralCtl::warmed_up_for_area_seeded(
        run.area,
        run.gen_seed,
    ));
}

/// Level-up juice: trauma + white flash + particle burst + jingle.
pub fn level_up_feedback(
    commands: &mut Commands,
    trauma: &mut Trauma,
    flash: &mut FlashWhite,
    audio: &GameAudio,
    cues: &mut Queue<AudioCue>,
    pos: glam::Vec2,
    color: [f32; 4],
) {
    trauma.add(0.35);
    flash_white(flash, 0.15);
    let mut rng = rand::rng();
    spawn_burst(commands, &mut rng, pos, 32, color, (120.0, 360.0));
    audio.play_levelup(cues);
}

/// Roll up to 4 unowned mutations (Destiny crowns roll 1; Patience
/// grants 4 next roll, then is consumed without repeating).
pub fn roll_mutations(player: &mut Player) -> Vec<MutationId> {
    roll_mutations_for(player, RaceId::Fish)
}

pub fn roll_mutations_for(player: &mut Player, race: RaceId) -> Vec<MutationId> {
    let mut rng = rand::rng();
    roll_mutations_with_for(player, race, &mut rng)
}

/// Seeded roll (deterministic under a seeded RNG; tests use this).
pub fn roll_mutations_with(player: &mut Player, rng: &mut impl rand::RngExt) -> Vec<MutationId> {
    roll_mutations_with_for(player, RaceId::Fish, rng)
}

pub fn roll_mutations_with_for(
    player: &mut Player,
    race: RaceId,
    rng: &mut impl rand::RngExt,
) -> Vec<MutationId> {
    let mut pool: Vec<MutationId> = ALL_MUTATIONS
        .iter()
        .copied()
        .filter(|m| {
            if *m == MutationId::Patience && player.patience_used {
                return false;
            }

            // GML `scrSkills.gml:196-198` (`scr_skill_can_appear`): the
            // Destiny crown hands out only one mutation, so Last Wish
            // (its jungle route) never appears for a non-Horror run.
            if *m == MutationId::LastWish
                && player.crown == CrownKind::Destiny
                && race != RaceId::Horror
            {
                return false;
            }

            !player.mutations.contains(m)
        })
        .collect();

    let mut out = Vec::new();

    let destiny = player.crown == CrownKind::Destiny;
    let mut want_base = if destiny { 1 } else { 4 };
    if race == RaceId::Horror {
        want_base += 1;
    }

    if player.patience_bonus && !destiny {
        want_base = 4;
    }
    let want = pool.len().min(want_base);

    player.patience_bonus = false;

    for _ in 0..want {
        let idx = rng.random_range(0..pool.len());
        out.push(pool.remove(idx));
    }

    let weapon_mutations = [
        MutationId::LongArms,
        MutationId::RecycleGland,
        MutationId::ShotgunShoulders,
        MutationId::BoilingVeins,
        MutationId::BoltMarrow,
        MutationId::LaserBrain,
    ];
    if !out.is_empty()
        && !player.heavy_heart_wanted
        && !player.mutations.contains(&MutationId::HeavyHeart)
        && weapon_mutations
            .iter()
            .filter(|mutation| player.mutations.contains(mutation))
            .count()
            >= 3
    {
        out[0] = MutationId::HeavyHeart;
        player.heavy_heart_wanted = true;
    }

    out
}

#[cfg(test)]
mod mutation_roll_tests {
    use super::{MutationId, roll_mutations_with};
    use crate::comps_a::Player;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn heavy_heart_replaces_first_choice_after_three_weapon_mutations() {
        let mut player = Player::default();
        player.mutations.extend([
            MutationId::LongArms,
            MutationId::RecycleGland,
            MutationId::BoltMarrow,
        ]);
        let mut rng = StdRng::seed_from_u64(7);
        let choices = roll_mutations_with(&mut player, &mut rng);
        assert_eq!(choices.first(), Some(&MutationId::HeavyHeart));
        assert!(player.heavy_heart_wanted);
    }
}

/// Mutation-pick UI state: flag resources written by `apply_mutation`.
#[derive(bevy_ecs::system::SystemParam)]
pub struct MutationFlagSet<'w> {
    pub scarier: ResMut<'w, ScarierFace>,
    pub euphoria: ResMut<'w, Euphoria>,
    pub open_mind: ResMut<'w, OpenMind>,
    pub heavy_heart: ResMut<'w, HeavyHeart>,
    /// `Option` so the mutation systems do not force the crib resource to
    /// exist in every test world that builds a flag set.
    pub crib: Option<ResMut<'w, crate::CribTrip>>,
}

/// Screen-feel sinks shared by level-up paths.
/// Route bookkeeping consumed by the room-end hop. Bundled because
/// `tick_portal_suck` is already at Bevy's 16-parameter cap.
#[derive(bevy_ecs::system::SystemParam)]
pub struct RouteBookkeeping<'w> {
    pub triggers: ResMut<'w, crate::secrets::SecretTriggers>,
    pub crib: ResMut<'w, crate::CribTrip>,
}

#[derive(bevy_ecs::system::SystemParam)]
pub struct LevelFx<'w> {
    pub trauma: ResMut<'w, Trauma>,
    pub chroma: ResMut<'w, ChromaticAberration>,
    pub slow_mo: ResMut<'w, SlowMotion>,
}

/// Mutation-choice flow state (pause / defer / toast / audio).
#[derive(bevy_ecs::system::SystemParam)]
pub struct ChoiceFlow<'w> {
    pub choice: ResMut<'w, MutationChoice>,
    pub paused: ResMut<'w, crate::state::Paused>,
    pub deferred: ResMut<'w, DeferredFloorGen>,
    pub toast: ResMut<'w, Toast>,
    pub run: Res<'w, Run>,
    pub audio: Res<'w, GameAudio>,
    pub cues: ResMut<'w, Queue<AudioCue>>,
}

/// Consume a pending ultra/mutation pick. Picks arrive via
/// `MutationChoice` (headless key/UI wiring lands with the UI phase);
/// exhausted pools heal to full instead. Grouped params keep the
/// system under the 16-param cap (deaths.rs precedent).
pub fn handle_mutation_choice(
    mut commands: Commands,
    ultra: Option<ResMut<PendingUltra>>,
    pending: Option<ResMut<PendingMutation>>,
    mut flow: ChoiceFlow,
    mut flags: MutationFlagSet,
    mut player_q: Query<(&mut Player, &mut Health, &mut Inventory, &RaceState), With<Player>>,
    mut fx: LevelFx,
    mut save: ResMut<SaveData>,
    mut dirty: ResMut<SaveDirty>,
) {
    if ultra.is_none() && pending.is_none() {
        if flow.choice.0.is_some() {
            flow.choice.0 = None;
        }
        return;
    }

    if !flow.paused.0 {
        flow.paused.0 = true;
    }

    let picked: Option<usize> = flow.choice.0.take();

    let Some(idx) = picked else {
        return;
    };

    if let Some(mut ultra) = ultra {
        let Some(id) = ultra.choices.get(idx).copied() else {
            return;
        };

        apply_ultra_mutation(
            &mut player_q,
            &mut flags,
            &mut fx,
            &mut flow.toast,
            &flow.audio,
            &mut flow.cues,
            id,
        );

        // GML `ULTRA_TIME`: reaching Level Ultra as any character.
        if crate::savedata_part::unlock_achievement(&mut save, 23) {
            dirty.0 = true;
            flow.toast.show("ULTRA TIME");
        }

        commands.spawn((GameCleanup, QueuedReactiveCue(ReactiveCue::UltraChosen)));

        ultra.choices.clear();
        commands.remove_resource::<PendingUltra>();

        if let Ok((mut player, mut health, _, race_state)) = player_q.single_mut() {
            player.ultra_pick_owed = false;

            if player.mutation_picks_owed > 0 {
                let choices = roll_mutations_for(&mut player, race_state.race);
                if choices.is_empty() {
                    health.hp = health.max;
                    try_recharge_strong_spirit(&mut player, &health);
                    player.mutation_picks_owed = 0;
                    flow.paused.0 = false;
                    if flow.deferred.0 {
                        try_start_pending_floor_gen(&mut commands, &flow.run);
                        flow.deferred.0 = false;
                    }
                } else {
                    commands.insert_resource(PendingMutation { choices });
                    flow.paused.0 = true;
                }
            } else {
                flow.paused.0 = false;
                if flow.deferred.0 {
                    try_start_pending_floor_gen(&mut commands, &flow.run);
                    flow.deferred.0 = false;
                }
            }
            let _ = race_state;
        } else {
            flow.paused.0 = false;
            if flow.deferred.0 {
                try_start_pending_floor_gen(&mut commands, &flow.run);
                flow.deferred.0 = false;
            }
        }
        return;
    }

    let Some(mut pending) = pending else {
        return;
    };

    let Some(id) = pending.choices.get(idx).copied() else {
        return;
    };

    apply_mutation(
        &mut player_q,
        &mut flags,
        &mut fx,
        &mut flow.toast,
        &flow.audio,
        &mut flow.cues,
        id,
    );

    commands.spawn((GameCleanup, QueuedReactiveCue(ReactiveCue::MutationChosen)));

    pending.choices.clear();
    commands.remove_resource::<PendingMutation>();

    if let Ok((mut player, mut health, _, race_state)) = player_q.single_mut() {
        player.mutation_picks_owed = player.mutation_picks_owed.saturating_sub(1);

        if player.ultra_pick_owed && player.ultra.is_none() && player.level >= 10 {
            let choices =
                ultra_choices_for_crown(race_state.race, player.crown == CrownKind::Destiny);
            commands.insert_resource(PendingUltra { choices });
            flow.paused.0 = true;
            return;
        }

        if player.mutation_picks_owed > 0 {
            let choices = roll_mutations_for(&mut player, race_state.race);
            if choices.is_empty() {
                health.hp = health.max;
                try_recharge_strong_spirit(&mut player, &health);
                player.mutation_picks_owed = 0;
                flow.paused.0 = false;
                if flow.deferred.0 {
                    try_start_pending_floor_gen(&mut commands, &flow.run);
                    flow.deferred.0 = false;
                }
            } else {
                commands.insert_resource(PendingMutation { choices });
                flow.paused.0 = true;
            }
            return;
        }
    }

    flow.paused.0 = false;
    if flow.deferred.0 {
        try_start_pending_floor_gen(&mut commands, &flow.run);
        flow.deferred.0 = false;
    }
}

/// Apply a mutation's stat effects (bevy `apply_mutation` verbatim,
/// minus sprite code — there was none — and minus the `Commands` param,
/// which the bevy body used only for the level-up sound; the headless
/// audio queue needs no commands).
pub fn apply_mutation(
    player_q: &mut Query<(&mut Player, &mut Health, &mut Inventory, &RaceState), With<Player>>,
    flags: &mut MutationFlagSet,
    fx: &mut LevelFx,
    toast: &mut Toast,
    audio: &GameAudio,
    cues: &mut Queue<AudioCue>,
    id: MutationId,
) {
    let Ok((mut player, mut health, mut inv, race_state)) = player_q.single_mut() else {
        return;
    };

    player.mutations.push(id);
    // GML `scrLevelUpScreenSubmit` verbatim: the pick spawns a
    // `SkillText` with `txt = loc("Skills", skill, "Name", ...)` (the
    // NAME only, not `Name: Desc`) drawn `@d`-dark at
    // `(view_center, view_height - height - 76)` blinking on
    // `disappear % 2`. The `Toast` resource is the headless `SkillText`.
    let (name, _) = mutation_name(id);
    toast.show(name);

    match id {
        MutationId::RhinoSkin => {
            health.max += 4;
            health.hp += 4;
        }
        MutationId::PlutoniumHunger => {
            player.pickup_range += 60.0;
        }
        MutationId::TriggerFingers => {}
        MutationId::RabbitPaw => {
            player.drop_mult += 0.4;
        }
        MutationId::SecondStomach => {
            player.medkit_mult = 2.0;
        }
        MutationId::ScarierFace => {
            flags.scarier.0 = true;
        }
        MutationId::BoilingVeins => {
            player.boiling_veins = true;
        }
        MutationId::ImpactWrists => {}
        MutationId::ExtraFeet => {
            player.speed_mult *= 1.5;
        }
        MutationId::Bloodlust => {
            player.bloodlust = true;
        }
        MutationId::LuckyShot => {
            player.lucky_shot = true;
        }
        MutationId::GammaGuts => {
            player.gamma_guts = true;
        }
        MutationId::BackMuscle => {
            player.back_muscle += 1;
            // GML `scrPlayerUpdateCuzAmmo`: back muscle also grows Cuz ammo.
            crate::player_fire::refresh_cuz_ammo_max(&mut *player);
            for kind in [
                AmmoKind::Bullets,
                AmmoKind::Shells,
                AmmoKind::Bolts,
                AmmoKind::Explosives,
                AmmoKind::Energy,
            ] {
                let cap = player.ammo_cap(kind);
                let a = inv.ammo_mut(kind);
                *a = (*a).min(cap);
            }
        }
        MutationId::Euphoria => {
            flags.euphoria.0 = true;
        }
        MutationId::LongArms => {
            player.melee_range_mult *= 1.5;
        }
        MutationId::Stress => {
            player.stress = true;
        }
        MutationId::EagleEyes => {
            player.spread_mult *= 0.4;
        }
        MutationId::OpenMind => {
            flags.open_mind.0 = true;
        }
        MutationId::HeavyHeart => {
            flags.heavy_heart.0 = true;
        }
        MutationId::StrongSpirit => {
            player.strong_spirit_ready = true;
            player.strong_spirit_spent = false;
            player.strong_spirit_area_cleared = false;
        }
        MutationId::SharpTeeth => {
            player.sharp_teeth = true;
        }
        MutationId::LastWish => {
            health.hp = health.max;
            try_recharge_strong_spirit(&mut player, &health);
            let add = |inv: &mut Inventory, player: &Player, kind: AmmoKind, amount: i32| {
                let cap = player.ammo_cap(kind);
                let slot = inv.ammo_mut(kind);
                *slot = (*slot + amount).min(cap);
            };
            add(&mut inv, &player, AmmoKind::Bullets, 200);
            add(&mut inv, &player, AmmoKind::Shells, 20);
            add(&mut inv, &player, AmmoKind::Bolts, 20);
            add(&mut inv, &player, AmmoKind::Explosives, 20);
            add(&mut inv, &player, AmmoKind::Energy, 20);
            player.last_wish_used = true;
        }

        MutationId::BoltMarrow => {
            player.bolt_marrow = true;
        }
        MutationId::Hammerhead => {
            player.hammerhead = true;
        }
        MutationId::LaserBrain => {
            player.laser_brain = true;
        }
        MutationId::RecycleGland => {
            player.recycle_gland = true;
        }
        MutationId::ShotgunShoulders => {
            player.shotgun_shoulders = true;
        }
        MutationId::ThroneButt => {
            player.throne_butt = true;
            apply_throne_butt_immediate_bonus(&mut player, &mut health, &mut inv, race_state.race);
        }
        MutationId::Patience => {
            player.patience_used = true;
            player.patience_bonus = true;
        }
    }

    fx.trauma.add(0.3);
    chromatic_pulse(&mut fx.chroma, 0.25);
    slow_motion(&mut fx.slow_mo, 0.5, 0.35);
    audio.play_levelup(cues);
}

/// Throne Butt's immediate per-race bonus (bevy verbatim).
pub fn apply_throne_butt_immediate_bonus(
    player: &mut Player,
    health: &mut Health,
    inv: &mut Inventory,
    race: RaceId,
) {
    player.ultra_ability_mult *= 1.15;

    match race {
        RaceId::Fish => {
            player.speed_mult *= 1.05;
        }
        RaceId::Crystal => {
            health.max += 2;
            health.hp += 2;
        }
        RaceId::Eyes => {
            player.pickup_range += 45.0;
        }
        RaceId::Melting => {
            player.chain_explosions = true;
        }
        RaceId::Plant => {
            player.speed_mult *= 1.08;
        }
        RaceId::Venuz => {
            player.fire_rate_mult *= 0.92;
        }
        RaceId::Steroids => {
            for kind in [
                AmmoKind::Bullets,
                AmmoKind::Shells,
                AmmoKind::Bolts,
                AmmoKind::Explosives,
                AmmoKind::Energy,
            ] {
                *inv.ammo_mut(kind) += ammo_pickup_amount(kind);
            }
        }
        RaceId::Robot => {
            player.free_ammo = true;
        }
        RaceId::Chicken => {
            player.headless_ready = true;
        }
        RaceId::Rebel => {
            health.hp = (health.hp + 1).min(health.max);
        }
        RaceId::Horror => {
            player.lucky_shot = true;
        }
        RaceId::Rogue => {
            player.boiling_veins = true;
        }
        RaceId::BigDog => {
            player.ultra_damage_mult *= 1.1;
        }
        RaceId::Skeleton => {
            player.bloodlust = true;
        }
        RaceId::Frog => {
            player.gamma_guts = true;
        }
        RaceId::Cuz | RaceId::Random => {
            player.fire_rate_mult *= 0.95;
        }
    }
}

/// Apply an ultra mutation's stat effects (bevy verbatim; same
/// `Commands` adaptation as `apply_mutation`).
pub fn apply_ultra_mutation(
    player_q: &mut Query<(&mut Player, &mut Health, &mut Inventory, &RaceState), With<Player>>,
    flags: &mut MutationFlagSet,
    fx: &mut LevelFx,
    toast: &mut Toast,
    audio: &GameAudio,
    cues: &mut Queue<AudioCue>,
    id: UltraMutationId,
) {
    let Ok((mut player, mut health, mut inv, race_state)) = player_q.single_mut() else {
        return;
    };
    let _ = flags;

    player.ultra = Some(id);

    // GML `UltraIcon/Other_10.gml:10-16`: a Venuz or Cuz ultra sends you to
    // the crib instead of advancing the stage.
    if let Some(crib) = flags.crib.as_mut() {
        if matches!(race_state.race, RaceId::Venuz | RaceId::Cuz) {
            crib.gocrib = true;
            crib.can_advance = false;
        }
    }

    match id {
        UltraMutationId::FishGunWarrant => {
            player.fire_rate_mult *= 0.75;
            player.spread_mult *= 0.85;
        }
        UltraMutationId::FishConfiscate => {
            player.drop_mult += 0.35;
            for kind in [
                AmmoKind::Bullets,
                AmmoKind::Shells,
                AmmoKind::Bolts,
                AmmoKind::Explosives,
                AmmoKind::Energy,
            ] {
                let slot = inv.ammo_mut(kind);
                *slot = (*slot + ammo_pickup_amount(kind) * 2).min(ammo_max(kind));
            }
        }

        UltraMutationId::CrystalFortress => {
            health.max += 6;
            health.hp += 6;
            player.ultra_ability_mult *= 1.4;
        }
        UltraMutationId::CrystalJuggernaut => {
            health.max += 3;
            health.hp += 3;
            player.speed_mult *= 1.18;
        }

        UltraMutationId::EyesMonsterStyle => {
            player.pickup_range += 160.0;
            player.ultra_ability_mult *= 1.45;
        }
        UltraMutationId::EyesProjectileStyle => {
            player.ultra_ability_mult *= 1.2;
            player.euphoria = true;
        }

        UltraMutationId::MeltingBrainCapacity => {
            player.chain_explosions = true;
            player.ultra_ability_mult *= 1.6;
        }
        UltraMutationId::MeltingDetachment => {
            health.max += 2;
            health.hp += 2;
            player.strong_spirit_ready = true;
        }

        UltraMutationId::PlantTrapper => {
            player.ultra_ability_mult *= 1.6;
            player.speed_mult *= 1.06;
        }
        UltraMutationId::PlantKiller => {
            player.speed_mult *= 1.18;
            player.fire_rate_mult *= 0.85;
        }

        UltraMutationId::VenuzBack2Bizniz => {
            player.ultra_ability_mult *= 1.5;
            player.fire_rate_mult *= 0.9;
        }
        UltraMutationId::VenuzGunGod => {
            player.fire_rate_mult *= 0.72;
            player.spread_mult *= 0.7;
        }

        UltraMutationId::SteroidsAmbidextrous => {
            player.fire_rate_mult *= 0.7;
            player.knockback_mult *= 0.85;
        }
        UltraMutationId::SteroidsGetArmed => {
            player.ultra_ability_mult *= 1.6;
            for kind in [
                AmmoKind::Bullets,
                AmmoKind::Shells,
                AmmoKind::Bolts,
                AmmoKind::Explosives,
                AmmoKind::Energy,
            ] {
                *inv.ammo_mut(kind) = player.ammo_cap(kind);
            }
        }

        UltraMutationId::RobotRefinedTaste => {
            player.free_ammo = true;
            player.medkit_mult *= 1.5;
        }
        UltraMutationId::RobotRegurgitate => {
            player.free_ammo = true;
            player.drop_mult += 0.5;
            player.ultra_ability_mult *= 1.35;
        }

        UltraMutationId::ChickenHarderToKill => {
            player.headless_ready = true;
            health.max += 2;
            health.hp += 2;
        }
        UltraMutationId::ChickenDetermination => {
            player.ultra_damage_mult *= 1.25;
            player.speed_mult *= 1.1;
        }

        UltraMutationId::RebelPersonalGuard => {
            player.ultra_ability_mult *= 1.45;
            health.max += 2;
            health.hp += 2;
        }
        UltraMutationId::RebelRiot => {
            player.ultra_ability_mult *= 1.8;
            player.fire_rate_mult *= 0.9;
        }

        UltraMutationId::HorrorStalker => {
            player.ultra_ability_mult *= 1.6;
            player.laser_brain = true;
        }
        UltraMutationId::HorrorAnomaly => {
            player.pickup_range += 80.0;
            player.lucky_shot = true;
            player.laser_brain = true;
        }
        UltraMutationId::HorrorMeltdown => {
            player.next_level_rads = player.next_level_rads.saturating_mul(2);
        }

        UltraMutationId::RogueSuperBlastArmor => {
            player.boiling_veins = true;
            player.veins_threshold = 6;
            health.max += 2;
            health.hp += 2;
        }
        UltraMutationId::RoguePortalStrike => {
            player.ultra_ability_mult *= 1.7;
            player.fire_rate_mult *= 0.9;
        }

        UltraMutationId::BigDogHeavyArtillery => {
            player.ultra_damage_mult *= 1.35;
            player.knockback_mult *= 1.25;
        }
        UltraMutationId::BigDogGuardian => {
            health.max += 5;
            health.hp += 5;
            player.strong_spirit_ready = true;
        }

        UltraMutationId::SkeletonBloodArmor => {
            health.max += 4;
            health.hp += 4;
            player.bloodlust = true;
        }
        UltraMutationId::SkeletonNecromancy => {
            player.bloodlust = true;
            player.recycle_gland = true;
            player.lucky_shot = true;
        }

        UltraMutationId::FrogToxicLord => {
            player.ultra_ability_mult *= 1.7;
            player.gamma_guts = true;
        }
        UltraMutationId::FrogSwampBody => {
            health.max += 3;
            health.hp += 3;
            player.boiling_veins = true;
        }

        UltraMutationId::CuzHoarder => {
            inv.weapon_slots = crate::comps_a::MAX_WEAPON_SLOTS;
            player.drop_mult += 0.25;
        }
        UltraMutationId::CuzQuickSwap => {
            inv.weapon_slots = crate::comps_a::MAX_WEAPON_SLOTS;
            player.fire_rate_mult *= 0.82;
            player.ultra_ability_mult *= 1.4;
        }
        UltraMutationId::CuzEmotional => {
            // GML `scrUltras`: Emotional refreshes `scrPlayerUpdateCuzAmmo`.
            crate::player_fire::refresh_cuz_ammo_max(&mut *player);
        }
    }

    fx.trauma.add(0.55);
    chromatic_pulse(&mut fx.chroma, 0.4);
    slow_motion(&mut fx.slow_mo, 0.35, 0.5);
    audio.play_levelup(cues);
    // GML `UltraIcon/Other_10` verbatim: the pick spawns a `SkillText`
    // with the ultra NAME (not `Name: Desc`) at the same SkillText
    // position. The `Toast` resource is the headless `SkillText`.
    toast.show(ultra_mutation_name(id).0);

    debug_assert!(
        ultra_choices_for(race_state.race).contains(&id) || race_state.race == RaceId::Random,
        "picked ultra {id:?} outside race {:?}",
        race_state.race,
    );
}

// ---------------------------------------------------------------------------
// Portals and floor transitions.
// ---------------------------------------------------------------------------

/// GML `Portal/Create_0.gml:1-22` (+ the `PortalL` ring): a `Portal` of
/// `type` (1 normal, 2 popo/HQ, 3 proto/vault), the enemy-shot clear, the
/// `PortalClear` + `PortalShock` children, and 4 `PortalL` bursts.
pub fn spawn_portal(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    enemy_shots: &mut Query<(Entity, &Team), With<Projectile>>,
    pos: glam::Vec2,
    kind: u8,
) {
    for (e, team) in &*enemy_shots {
        if *team != Team::Player {
            commands.entity(e).despawn();
        }
    }

    // Bevy rides the `sprPortalSpawn` oneshot strip for the Spawn
    // gate (and swaps it to the idle strip on finish); the renderer
    // draws the live strip (portal visuals batch). No strip in the
    // catalog → no anim, gate opens immediately (bevy `unwrap_or`).
    let mut pe = commands.spawn((
        GameCleanup,
        LevelCleanup,
        Portal,
        PortalState {
            kind,
            phase: PortalPhase::Spawn,
            endgame: 100.0,
            close: false,
            anim: GTimer::from_seconds(0.25, TimerMode::Once),
            loop_on: false,
        },
        Pos(pos),
    ));
    if let Some(def) = catalog.def("images/sprPortalSpawn.png") {
        pe.insert(crate::anim::SpriteAnim::oneshot(
            "images/sprPortalSpawn.png",
            def,
        ));
    }

    commands.spawn((
        GameCleanup,
        LevelCleanup,
        PortalShock {
            timer: GTimer::from_seconds(2.0 / 30.0, TimerMode::Once),
            radius: 72.0,
        },
        Pos(pos),
    ));
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        PortalClear {
            timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
            scale: 1.0,
        },
        Pos(pos),
    ));

    let mut rng = rand::rng();
    spawn_burst(
        commands,
        &mut rng,
        pos,
        4,
        [0.5, 0.8, 1.0, 1.0],
        (60.0, 160.0),
    );
}

/// GML `IceFlower/Step_0.gml:4-22` verbatim: the `feed >= 4` payoff, the
/// only way into `area_jungle`. `GameCont.area = area_jungle; subarea = 0`
/// pre-loads the jungle and the room-end bump (`GameCont/Other_5:134`)
/// then takes subarea to 1, the jungle's only floor; the port flips the
/// area on the portal transit instead, exactly as it already does for
/// `area_vault`, so the leg is queued here. `with (enemy) hp = 0` wipes
/// the floor (deaths run the normal cascade), a plain `type = 1`
/// `Portal` opens at the flower, `mut_last_wish` is refunded, then the
/// flower dies.
pub fn ice_flower_jungle(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    triggers: &mut crate::secrets::SecretTriggers,
    player: &mut Player,
    enemy_shots: &mut Query<(Entity, &Team), With<Projectile>>,
    enemies: &mut Query<(Entity, &mut Health), With<Enemy>>,
    flower: Entity,
    flower_pos: glam::Vec2,
) {
    triggers.queue(SecretTarget::Jungle);

    // `IceFlower` is a `prop`, so GML's `with (enemy)` never swept it.
    for (e, mut health) in enemies.iter_mut() {
        if e == flower {
            continue;
        }
        health.hp = 0;
    }

    spawn_portal(commands, catalog, enemy_shots, flower_pos, 1);

    // GML `IceFlower/Step_0:14-19`: the jungle secret eats the Last Wish
    // and hands the skill point back, so the next pick is free.
    if player.mutations.contains(&MutationId::LastWish) {
        player.mutations.retain(|m| *m != MutationId::LastWish);
        player.last_wish_used = false;
        player.mutation_picks_owed = player.mutation_picks_owed.saturating_add(1);
    }

    commands.entity(flower).despawn();
}

/// Open the exit portal once the area is clear: clear stray enemy fire,
/// spawn portal + shock + clear markers, juice, sting.
pub fn portal_check(
    mut commands: Commands,
    mut run: ResMut<Run>,
    loop_transition: Res<LoopTransition>,
    mut trauma: ResMut<Trauma>,
    mut chroma: ResMut<ChromaticAberration>,
    mask: Res<FloorMask>,
    enemies: Query<Entity, With<Enemy>>,
    mut enemy_shots: Query<(Entity, &Team), With<Projectile>>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    catalog: Res<repame_anim::AnimCatalog>,
    portals: Query<Entity, With<Portal>>,
    // GML `Corpse/Alarm_0:3`: the clear portal never spawns while a
    // `CrownPickup` / `VaultStatue` / `CrownGuardian` lives. The pickup is
    // the port's `CrownPedestal`, the guardian is an `Enemy` (gated above);
    // the statue is a `Prop`, so it needs its own query.
    vault: Query<Entity, With<CrownPedestal>>,
    vault_statues: Query<Entity, With<crate::crown::VaultStatue>>,
) {
    if run.game_over || run.portal_open {
        return;
    }

    if loop_transition.blocks_portal() {
        return;
    }
    if !enemies.is_empty() {
        return;
    }
    if !vault.is_empty() || !vault_statues.is_empty() {
        return;
    }
    if !portals.is_empty() {
        // GML `Corpse/Alarm_0:2`: the corpse-clear portal only spawns
        // when no `Portal` lives — a statue or pedestal portal already
        // opens the floor.
        run.portal_open = true;
        return;
    }

    run.portal_open = true;
    commands.spawn((GameCleanup, QueuedReactiveCue(ReactiveCue::PortalOpen)));

    let mut rng = rand::rng();
    let pos = mask.random_floor_pos(&mut rng, 80.0);

    // GML `Corpse/Alarm_0:23-28`: `type = 1`, `type = 2` for `area_hq`.
    // Type 3 exists only on the two statue/pedestal portals
    // (`ProtoStatue/Destroy_0:13-15`, `CrownPickup/Collision_Player:19`).
    let kind: u8 = if run.area == AreaId::HQ { 2 } else { 1 };

    spawn_portal(&mut commands, &catalog, &mut enemy_shots, pos, kind);

    trauma.add(0.25);
    chromatic_pulse(&mut chroma, 0.25);
    audio.play_portal(&mut cues);
}

/// Portal vortex drag: pulls the player and loose weapon pickups toward
/// an idle portal (axis-separated, walkability-gated). Never touches
/// `Velocity` (bevy parity). Wall line-of-sight from the bevy build is
/// omitted (walls phase).
pub fn portal_attract(
    time: Res<SimTime>,
    mut player_q: Query<
        (Entity, &mut Pos),
        (With<Player>, Without<Portal>, Without<PortalSucking>),
    >,
    mut weapon_q: Query<
        (Entity, &mut Pos, &Pickup, Option<&mut GroundPhysics>),
        (Without<Player>, Without<Portal>),
    >,
    portal_q: Query<(&Pos, Option<&PortalState>), With<Portal>>,
    mask: Res<FloorMask>,
    run: Res<Run>,
) {
    if run.game_over || !run.portal_open {
        return;
    }
    let Ok((portal_pos, st_opt)) = portal_q.single() else {
        return;
    };
    if let Some(st) = st_opt
        && st.phase != PortalPhase::Idle
    {
        return;
    }
    let tpos = portal_pos.0;
    let dt = time.delta_secs;

    let attract_step = |ppos: glam::Vec2| -> Option<(glam::Vec2, f32)> {
        let dist = ppos.distance(tpos);
        if dist > 96.0 || dist < 0.5 {
            return None;
        }
        if crate::walls::segment_hits_wall(ppos, tpos, &mask) {
            return None;
        }
        let spd = if dist > 48.0 { 2.0 } else { 5.0 };
        let dir = (tpos - ppos).normalize_or_zero();
        Some((dir, spd))
    };

    if let Ok((_player_e, mut ppos)) = player_q.single_mut() {
        if let Some((dir, spd)) = attract_step(ppos.0) {
            let delta = dir * spd * 30.0 * dt;
            let nx = glam::Vec2::new(ppos.0.x + delta.x, ppos.0.y);
            let ny = glam::Vec2::new(ppos.0.x, ppos.0.y + delta.y);
            if mask.is_walkable(nx) {
                ppos.0.x = nx.x;
            }
            if mask.is_walkable(ny) {
                ppos.0.y = ny.y;
            }
        }
    }

    for (_e, mut wpos, pickup, gp) in &mut weapon_q {
        let PickupKind::Weapon(_) = pickup.kind else {
            continue;
        };

        let Some((dir, spd)) = attract_step(wpos.0) else {
            continue;
        };

        let delta = dir * spd * 30.0 * dt;
        let nx = glam::Vec2::new(wpos.0.x + delta.x, wpos.0.y);
        let ny = glam::Vec2::new(wpos.0.x, wpos.0.y + delta.y);
        if mask.is_walkable(nx) {
            wpos.0.x = nx.x;
        }
        if mask.is_walkable(ny) {
            wpos.0.y = ny.y;
        }

        if let Some(mut gp) = gp {
            gp.vel *= 0.8;
        }
    }
}

/// Read-only world snapshot the per-kind chest payout needs. GML reads all
/// of it off `self` / `GameCont` / the nearest `Player` at open time.
pub(crate) struct ChestLootCtx {
    pub race: RaceId,
    /// `pickups::decide_ctx_for` — `GameCont.hard` plus the target player.
    pub decide: crate::decide_wep::DecideCtx,
    /// `chestprop/curse`, frozen in the chest's `Create_0`.
    pub curse: bool,
    /// `chestprop/dropseed`, frozen in the chest's `Create_0`.
    pub drop_seed: u32,
    pub underwater: bool,
    /// Steroids Ambidextrous (`UltraSkill.Ambidextrous`).
    pub ambidextrous: bool,
    /// `scrPlayerCountCursed` — gates `AmmoPickup/Create_0.gml:16`'s
    /// `CursedPickup` conversion.
    pub cursed_count: u32,
}

/// GML `prop/Destroy_0.gml:12` `if (raddrop > 0) scrRadDrop(x, y,
/// raddrop)`, reached from a portal shock through
/// `PortalShock/Collision_prop.gml`'s `other.hp = 0` (every `RadChest`
/// descendant).
fn shock_rad_drop(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    amount: u32,
    run: &Run,
    hasted: bool,
) {
    crate::pickups::scr_rad_drop(
        commands,
        catalog,
        pos,
        amount,
        run.loop_count,
        hasted,
        true,
        true,
    );
}

/// GML `event_perform(ev_collision, Player)` on `WeaponChest`,
/// `BigWeaponChest`, `CursedBigChest`, `GoldChest` and `IDPDChest` — every
/// one of those `Collision_PortalShock.gml:4` files is that one line — so a
/// portal shock must hand out EXACTLY the touch-open payout. The kinds that
/// hand-roll their own shock loot (`AmmoChest`, `AmmoChestMystery`,
/// `HealthChest`, `RogueChest`) and the `prop`-based rad chests
/// (`PortalShock/Collision_prop.gml` is `other.hp = 0`) stay at their own
/// call sites. `ProtoChest` has no `Collision_PortalShock` event at all.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chest_loot(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    audio: &GameAudio,
    cues: &mut Queue<AudioCue>,
    toast: &mut Toast,
    run: &mut Run,
    ctx: &ChestLootCtx,
    kind: ChestKind,
    pos: glam::Vec2,
    player_pos: glam::Vec2,
) {
    let mut rng = rand::rng();
    match kind {
        // GML `WeaponChest/Collision_Player.gml:11-24`: ONE
        // `scrDecideWep(1 + curse * 2, curse)` fills every drop, and
        // `_count` is 2 under Ambidextrous.
        ChestKind::Weapon => {
            let count = if ctx.ambidextrous { 2 } else { 1 };
            let extra = 1 + if ctx.curse { 2 } else { 0 };
            let mut chain = crate::pickups::DropSeedChain::new(ctx.drop_seed);
            let weapon = chain.roll(|r| {
                crate::decide_wep::decide_wep_at(commands, r, &ctx.decide, pos, extra, ctx.curse)
            });
            for _ in 0..count {
                let e = crate::pickups::spawn_pickup(
                    commands,
                    catalog,
                    PickupKind::Weapon(weapon),
                    pos + glam::Vec2::new(rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0)),
                    0,
                    false,
                );
                // GML `scrWeaponPickupCreate(..., true)`.
                commands
                    .entity(e)
                    .insert(crate::comps_b::WepPickupAmmo(true));
                if ctx.curse {
                    commands.entity(e).insert(PickupCurse);
                }
            }
            toast.show(crate::weapon_runtime::weapon_id_name(weapon));
            audio.play_weapon_chest_open(cues, ctx.underwater, ctx.curse);
        }
        // GML `GoldChest/Collision_Player.gml`: one `scrDecideWepGold`
        // pickup, no `PortalClear`, no `nochest` reset.
        ChestKind::Gold => {
            let weapon = crate::decide_wep::decide_wep_gold(
                &mut rng,
                run.loop_count,
                &ctx.decide.owned,
                ctx.race == RaceId::Steroids,
            );
            let e = crate::pickups::spawn_pickup(
                commands,
                catalog,
                PickupKind::Weapon(weapon),
                pos,
                0,
                false,
            );
            commands
                .entity(e)
                .insert(crate::comps_b::WepPickupAmmo(true));
            audio.play_gold_chest(cues);
        }
        // GML `BigWeaponChest/Collision_Player.gml:14-30`:
        // `random_set_seed(dropseed)` ONCE, then `scrDecideWep(1, false)`
        // per drop, a `PortalClear`, and `GameCont.nochest = 0`.
        ChestKind::BigWeapon => {
            let count = if ctx.ambidextrous { 4 } else { 3 };
            let mut chain = crate::pickups::DropSeedChain::new(ctx.drop_seed);
            for _ in 0..count {
                let at =
                    pos + glam::Vec2::new(rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0));
                let weapon = chain.roll(|r| {
                    crate::decide_wep::decide_wep_at(commands, r, &ctx.decide, at, 1, false)
                });
                let e = crate::pickups::spawn_pickup(
                    commands,
                    catalog,
                    PickupKind::Weapon(weapon),
                    at,
                    0,
                    false,
                );
                commands
                    .entity(e)
                    .insert(crate::comps_b::WepPickupAmmo(true));
                toast.show(crate::weapon_runtime::weapon_id_name(weapon));
            }
            spawn_shock_portal_clear(commands, pos);
            run.nochest = 0;
            audio.play_big_chest_open(cues, false, crate::pickups::chst_stem(ctx.race));
        }
        // GML `CursedBigChest/Collision_Player.gml:20`: `scrDecideWep(1 +
        // curse * 2, false)` with `curse = true` from `Create_0:3`, so
        // `extra = 3` and the curse rides the pickups. `:33` resets
        // `nochest`; `:17` and `Destroy_0:6` each spawn a `PortalClear`.
        ChestKind::CursedBig => {
            let count = if ctx.ambidextrous { 4 } else { 3 };
            let mut chain = crate::pickups::DropSeedChain::new(ctx.drop_seed);
            for _ in 0..count {
                let at =
                    pos + glam::Vec2::new(rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0));
                let weapon = chain.roll(|r| {
                    crate::decide_wep::decide_wep_at(commands, r, &ctx.decide, at, 3, false)
                });
                let e = crate::pickups::spawn_pickup(
                    commands,
                    catalog,
                    PickupKind::Weapon(weapon),
                    at,
                    0,
                    false,
                );
                commands
                    .entity(e)
                    .insert(crate::comps_b::WepPickupAmmo(true));
                commands.entity(e).insert(PickupCurse);
                toast.show(crate::weapon_runtime::weapon_id_name(weapon));
            }
            spawn_shock_portal_clear(commands, pos);
            spawn_shock_portal_clear(commands, pos);
            run.nochest = 0;
            audio.play_big_chest_open(cues, true, crate::pickups::chst_stem(ctx.race));
        }
        // GML `IDPDChest/Collision_Player.gml:11-15`: eight `AmmoPickup`s
        // ON the player, then `instance_destroy()` -> `Destroy_0`'s six
        // `IDPDSpawn` portals.
        ChestKind::Idpd => {
            for _ in 0..8 {
                crate::pickups::spawn_ammo_pickup(
                    commands,
                    catalog,
                    ctx.cursed_count,
                    player_pos,
                    run.loop_count,
                    false,
                );
            }
            crate::idpd::raise_idpd_portals(commands, run, cues, 0, pos);
            audio.play_ammo_chest_open(cues, ctx.underwater);
        }
        // No GML `Collision_PortalShock` reaches these; they keep the
        // hand-rolled payouts at their call sites.
        ChestKind::Ammo
        | ChestKind::Mystery
        | ChestKind::Health
        | ChestKind::Rogue
        | ChestKind::Rad
        | ChestKind::RadBig
        | ChestKind::RadMaggot
        | ChestKind::Proto => {}
    }
}

/// GML `BigWeaponChest/Collision_Player.gml:16` and
/// `CursedBigChest:17` + `Destroy_0:6` (which spawns a second one).
fn spawn_shock_portal_clear(commands: &mut Commands, pos: glam::Vec2) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        PortalClear {
            timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
            scale: 1.0,
        },
        Pos(pos),
    ));
}

/// Portal spawn shock: destroys destructible props in radius (corpses +
/// death effects + secret queues + the `raddrop` from
/// `prop/Destroy_0.gml:12`), pops chests into their contents, and clears
/// enemy projectiles. Expires on its timer.
#[allow(clippy::too_many_arguments)]
pub fn tick_portal_shock(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut toast: ResMut<Toast>,
    save: Res<crate::savedata_part::SaveData>,
    mut shocks: Query<(Entity, &Pos, &mut PortalShock)>,
    mut props: Query<
        (
            Entity,
            &mut Prop,
            &Pos,
            Option<&PropDeathEffect>,
            Option<&PropSprites>,
        ),
        With<Prop>,
    >,
    rad_containers: Query<(), With<RadChestContainer>>,
    mut chests: Query<
        (
            Entity,
            &Pos,
            &Pickup,
            Option<&crate::pickups::ChestCurse>,
            Option<&crate::comps_b::DropSeed>,
        ),
        (Without<OpenedChest>, Without<Player>),
    >,
    mut enemy_shots: Query<(Entity, &Pos, &Team), With<Projectile>>,
    entrances: Query<&SecretEntrance>,
    mut secrets: ResMut<crate::secrets::SecretTriggers>,
    mut run: ResMut<Run>,
    mut player_q: Query<(Entity, &Pos, &mut Player, &Inventory, &RaceState), Without<Pickup>>,
) {
    let Ok((_player_e, player_pos, mut player, inv, race_state)) = player_q.single_mut() else {
        return;
    };
    let player_pos = player_pos.0;
    let race = race_state.race;
    let hasted = player.crown == CrownKind::Haste;
    let underwater = run.area == AreaId::Oasis;
    let ambidextrous = matches!(
        player.ultra,
        Some(crate::data::UltraMutationId::SteroidsAmbidextrous)
    );
    let hp_num = crate::pickups::hppickup_num(&player);
    let second_stomach = player.mutations.contains(&MutationId::SecondStomach);
    let decide =
        crate::pickups::decide_ctx_for(&run, &player, race, inv, u32::from(race == RaceId::Robot));
    let dt = time.delta_secs;
    for (shock_e, shock_pos, mut shock) in &mut shocks {
        shock.timer.tick(dt);
        let center = shock_pos.0;

        let mut killed: Vec<(
            Entity,
            glam::Vec2,
            bool,
            Option<PropDeathEffect>,
            Option<PropSprites>,
            bool,
        )> = Vec::new();
        for (prop_e, mut prop, prop_pos, death, ps) in &mut props {
            if !prop.destructible || prop.hp <= 0 {
                continue;
            }
            let ppos = prop_pos.0;
            let half = prop.size * 0.5;
            let closest = glam::Vec2::new(
                center.x.clamp(ppos.x - half.x, ppos.x + half.x),
                center.y.clamp(ppos.y - half.y, ppos.y + half.y),
            );
            if center.distance(closest) > shock.radius {
                continue;
            }
            prop.hp = 0;
            killed.push((
                prop_e,
                ppos,
                prop.explosive,
                death.copied(),
                ps.copied(),
                // GML `prop/Destroy_0.gml:12` pays `raddrop` on the way
                // out; a `RadChest` prop is the 25-rad case.
                rad_containers.get(prop_e).is_ok(),
            ));
        }
        for (prop_e, ppos, explosive, death, ps, rad_drop) in killed {
            if let Some(sprites) = ps {
                spawn_prop_corpse(&mut commands, &catalog, ppos, &sprites);
            }
            spawn_prop_death_effect(
                &mut commands,
                &catalog,
                save.settings.particles,
                ppos,
                death,
                explosive,
                None,
            );
            if rad_drop {
                shock_rad_drop(&mut commands, &catalog, ppos, 25, &run, hasted);
            }
            if let Ok(entrance) = entrances.get(prop_e) {
                secrets.queue(entrance.target);
            }
            commands.entity(prop_e).despawn();
        }

        for (chest_e, chest_pos, pickup, chest_curse, drop_seed) in &mut chests {
            let PickupKind::Chest(kind) = pickup.kind else {
                continue;
            };
            let cpos = chest_pos.0;
            if center.distance(cpos) > shock.radius {
                continue;
            }

            commands.entity(chest_e).insert(OpenedChest(kind));
            // GML `Destroy_0` spawns `FXChestOpen` for every chest kind
            // except `RogueChest` (the only chest with no FX).
            if kind != ChestKind::Rogue {
                crate::pickups::spawn_fx_chest_open(&mut commands, &catalog, cpos, underwater);
            }
            let ctx = ChestLootCtx {
                race,
                decide: decide.clone(),
                curse: chest_curse.is_some_and(|c| c.0),
                drop_seed: drop_seed.map_or(0, |s| s.0),
                underwater,
                ambidextrous,
                cursed_count: crate::pickups::count_cursed(inv),
            };
            chest_loot(
                &mut commands,
                &catalog,
                &audio,
                &mut cues,
                &mut toast,
                &mut run,
                &ctx,
                kind,
                cpos,
                player_pos,
            );
            // GML hand-rolls the shock payout for the kinds with no
            // `event_perform(ev_collision, Player)`; `chest_loot` leaves
            // those to here.
            match kind {
                // `AmmoChest/Collision_PortalShock.gml:4-7`: `repeat 2
                // instance_create(x, y, AmmoPickup)` + `sndChest`.
                ChestKind::Ammo => {
                    for _ in 0..2 {
                        crate::pickups::spawn_ammo_pickup(
                            &mut commands,
                            &catalog,
                            ctx.cursed_count,
                            cpos,
                            run.loop_count,
                            hasted,
                        );
                    }
                    audio.play_chest(&mut cues);
                }
                // `AmmoChestMystery/Collision_PortalShock.gml:4-8`: the
                // same two pickups, but `sndAmmoChest`.
                ChestKind::Mystery => {
                    for _ in 0..2 {
                        crate::pickups::spawn_ammo_pickup(
                            &mut commands,
                            &catalog,
                            ctx.cursed_count,
                            cpos,
                            run.loop_count,
                            hasted,
                        );
                    }
                    audio.play_ammo_chest_open(&mut cues, underwater);
                }
                // `HealthChest/Collision_PortalShock.gml:4-6`: `repeat (2)
                // instance_create(x, y, HPPickup)` + `sndHealthChest`.
                ChestKind::Health => {
                    for _ in 0..2 {
                        crate::pickups::spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Medkit(hp_num),
                            cpos,
                            0,
                            false,
                        );
                    }
                    audio.play_health_chest(&mut cues, second_stomach);
                }
                // `RogueChest/Collision_PortalShock.gml:1-6`: a Rogue gets a
                // `RogueAmmo` (which then collides with the player, so the
                // canister refills), everyone else 25 rads.
                ChestKind::Rogue => {
                    if race == RaceId::Rogue {
                        player.rogue_ammo = player.rogue_ammo_max;
                    } else {
                        crate::pickups::scr_rad_drop(
                            &mut commands,
                            &catalog,
                            cpos,
                            25,
                            run.loop_count,
                            hasted,
                            false,
                            false,
                        );
                    }
                }
                // The rad chests are `prop` descendants in GML, so the
                // shock only sets `hp = 0` and `Destroy_0` pays the
                // `raddrop` (`RadChest` 25, `RadChestBig` 45, the maggot
                // chest inherits RadChest's 25).
                ChestKind::Rad => shock_rad_drop(&mut commands, &catalog, cpos, 25, &run, hasted),
                ChestKind::RadBig => {
                    shock_rad_drop(&mut commands, &catalog, cpos, 45, &run, hasted)
                }
                ChestKind::RadMaggot => {
                    // `RadMaggotChest/Destroy_0.gml:1-3`: a
                    // `RadMaggotExplosion`, whose Alarm_0 raises 20
                    // `RadMaggot` 8 ticks later.
                    shock_rad_drop(&mut commands, &catalog, cpos, 25, &run, hasted);
                    for _ in 0..20 {
                        let mut rng = rand::rng();
                        let a = rng.random_range(0.0..std::f32::consts::TAU);
                        let d = glam::Vec2::new(a.cos(), a.sin());
                        let s = rng.random_range(0.0..5.0) * 30.0;
                        let jitter = glam::Vec2::new(
                            rng.random_range(-4.0..4.0),
                            rng.random_range(-4.0..4.0),
                        );
                        queue_enemy_spawn(
                            &mut commands,
                            crate::data::EnemyKind::RadMaggot,
                            cpos + jitter + d * s * 0.05,
                            1.0,
                            run.loop_count,
                        );
                    }
                }
                ChestKind::Weapon
                | ChestKind::Gold
                | ChestKind::CursedBig
                | ChestKind::BigWeapon
                | ChestKind::Idpd
                | ChestKind::Proto => {}
            }
        }

        for (proj_e, proj_pos, team) in &mut enemy_shots {
            if *team == Team::Player {
                continue;
            }
            if proj_pos.0.distance(center) <= shock.radius {
                commands.entity(proj_e).despawn();
            }
        }
        if shock.timer.just_finished() {
            commands.entity(shock_e).despawn();
        }
    }
}

/// Portal spawn clear: breaks walls near the portal so it never opens
/// inside solid rock. Expires on its timer.
pub fn tick_portal_clear(
    time: Res<SimTime>,
    mut commands: Commands,
    mut clears: Query<(Entity, &Pos, &mut PortalClear)>,
    walls: Query<(&crate::comps_a::WallCell, &Pos), With<crate::comps_a::WallTile>>,
) {
    let dt = time.delta_secs;
    for (clear_e, clear_pos, mut clear) in &mut clears {
        clear.timer.tick(dt);
        let done = clear.timer.just_finished();
        if done {
            let center = clear_pos.0;
            for (cell, wpos) in &walls {
                if wpos.0.distance(center) < 48.0 {
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        crate::comps_a::PendingWallBreak {
                            cell: (cell.0, cell.1),
                            pos: wpos.0,
                            spawn_floor: true,
                        },
                    ));
                }
            }
            commands.entity(clear_e).despawn();
        }
    }
}

/// Run clock (GML `GameCont.tottimer`: advances one step per live
/// gameplay tick; HUD timer string + Plant/B throne-skin gate).
pub fn tick_run_clock(state: Res<AppState>, paused: Res<Paused>, mut run: ResMut<Run>) {
    if *state == AppState::InGame && !paused.0 && !run.game_over {
        run.tottimer = run.tottimer.saturating_add(1);
    }
}

/// Portal latch: touching the portal starts the 3 s close sequence
/// (the actual floor flip happens in `tick_portal_suck`). Robot eats
/// nearby loose weapons for ammo on entry.
pub fn portal_enter(
    mut commands: Commands,
    run: Res<Run>,
    catalog: Res<repame_anim::AnimCatalog>,
    mut portal_q: Query<
        (
            Entity,
            &Pos,
            Option<&PortalClosing>,
            Option<&mut PortalState>,
        ),
        With<Portal>,
    >,
    mut player_q: Query<
        (
            Entity,
            &Pos,
            &mut Velocity,
            &mut Health,
            &RaceState,
            &mut Inventory,
            &Player,
        ),
        (With<Player>, Without<Portal>, Without<PortalSucking>),
    >,
    mut weapon_q: Query<
        (Entity, &Pos, &Pickup, Option<&PickupCurse>),
        (Without<Player>, Without<Portal>),
    >,
) {
    if run.game_over {
        return;
    }

    let Ok((portal_e, portal_pos, closing, st_opt)) = portal_q.single_mut() else {
        return;
    };

    if closing.is_some() {
        return;
    }

    let Ok((_player_e, player_pos, mut vel, mut health, race_state, mut inv, player)) =
        player_q.single_mut()
    else {
        return;
    };

    let ppos = player_pos.0;
    let tpos = portal_pos.0;
    if ppos.distance(tpos) > 28.0 {
        return;
    }
    let _ = &mut vel;

    commands.entity(portal_e).insert(PortalClosing {
        timer: GTimer::from_seconds(90.0 / 30.0, TimerMode::Once),
    });

    if let Some(mut st) = st_opt {
        st.close = true;
        st.endgame = 30.0_f32.min(st.endgame);
    }

    health.invuln = GTimer::from_seconds(1.0 /* 30 ticks */, TimerMode::Once);

    if race_state.race == RaceId::Robot {
        for (wep_e, wep_pos, pickup, curse) in &mut weapon_q {
            let PickupKind::Weapon(w) = pickup.kind else {
                continue;
            };
            if wep_pos.0.distance(tpos) > 96.0 {
                continue;
            }
            // GML `Portal/Collision_Player`: visible non-cursed guns are
            // eaten with auto-collect (`scrRobotEat(other.wep, true)`).
            if curse.is_some() {
                continue;
            }

            crate::player_fire::robot_eat_drops(
                &mut commands,
                &catalog,
                wep_pos.0,
                player,
                &mut health,
                &mut inv,
                w,
                true,
                run.area,
                run.gen_seed,
            );
            let mut rng = rand::rng();
            spawn_burst(
                &mut commands,
                &mut rng,
                wep_pos.0,
                6,
                [0.6, 0.6, 0.6, 1.0],
                (30.0, 90.0),
            );
            // GML `RobotEat` (+TB variant): oneshot strip at the eaten
            // gun, destroyed on anim end (image_speed 0.4 → frames/12 s).
            let eat_path = if player.throne_butt {
                "images/sprRobotEatTB.png"
            } else {
                "images/sprRobotEat.png"
            };
            if let Some(def) = catalog.def(eat_path) {
                // GML `image_speed = 0.4` → 12 fps, destroyed on anim
                // end: timer and lifetime both run frames/12 s.
                let mut anim = crate::anim::SpriteAnim::oneshot(eat_path, def);
                anim.timer = GTimer::from_seconds(1.0 / 12.0, TimerMode::Repeating);
                let mut ee = commands.spawn((GameCleanup, LevelCleanup, Pos(wep_pos.0), anim));
                ee.insert(crate::comps_b::PickupLifetime {
                    timer: GTimer::from_seconds(
                        (def.frames.max(1) as f32 / 12.0).max(0.09),
                        TimerMode::Once,
                    ),
                });
            }
            commands.entity(wep_e).despawn();
        }
    }

    commands.spawn((GameCleanup, QueuedReactiveCue(ReactiveCue::PortalEnter)));
}

/// Portal suck: drags the player into the portal core, then flips the
/// level (loop / secret / normal advance), drains pending mutation
/// picks (deferring floor-gen while the pick UI owns the pause), and
/// otherwise kicks the loading transition. Rotation/shrink visuals
/// from the bevy build are omitted (render phase). GML
/// `Portal/Alarm_1` verbatim: a tutorial exit portal restarts the run
/// (`game_restart()`) instead of advancing the floor.
pub fn tick_portal_suck(
    time: Res<SimTime>,
    mut commands: Commands,
    mut run: ResMut<Run>,
    weapon_q: Query<&Pickup>,
    rad_props: Query<(), With<RadChestContainer>>,
    mut proto_chests: Query<(Entity, &mut ProtoChestState, Option<&OpenedChest>), Without<Player>>,
    mut player_q: Query<
        (
            Entity,
            &mut Pos,
            &mut Health,
            &mut Player,
            &RaceState,
            &mut Inventory,
            &mut PortalSucking,
        ),
        With<Player>,
    >,
    // GML `GenCont/Destroy_0.gml:186`: every level instance is destroyed
    // on room change; the carried proto chests survive.
    level_q: Query<Entity, With<LevelCleanup>>,
    mut loop_transition: ResMut<LoopTransition>,
    mut trauma: ResMut<Trauma>,
    mut toast: ResMut<Toast>,
    mut route: RouteBookkeeping,
    mut paused: ResMut<crate::state::Paused>,
    mut deferred: ResMut<DeferredFloorGen>,
    mut save: ResMut<SaveData>,
    mut dirty: ResMut<SaveDirty>,
) {
    let Ok((player_e, mut player_pos, mut health, mut player, race_state, mut inv, mut suck)) =
        player_q.single_mut()
    else {
        return;
    };

    suck.timer.tick(time.delta_secs);
    let t = suck.timer.fraction().clamp(0.0, 1.0);

    let ease = t * t;
    player_pos.0 = suck.start_pos.lerp(suck.target_pos, ease);

    trauma.add(0.02);
    if !suck.timer.just_finished() {
        return;
    }

    let portal_e = suck.portal;
    let race = race_state.race;
    commands.entity(player_e).remove::<PortalSucking>();

    // GML `Portal/Alarm_1` verbatim: the tutorial exit portal restarts
    // the run (`game_restart()`) instead of advancing the floor. The
    // port reboots run state in place (same `Loading` path as death
    // RETRY). Completion persists to the save first (GML
    // `save game.tutorial=false` in `TutCont/Alarm_0`, written before
    // the exit portal spawns) so the fresh run lands on the real
    // first floor instead of replaying the tutorial.
    if run.tutorial {
        run.tutorial = false;
        save.tutorial_done = true;
        dirty.0 = true;
        commands
            .entity(player_e)
            .insert(crate::state::TutorialRestart);
        return;
    }

    // GML `Portal/Alarm_1`: ground `WepPickup` only — visible and
    // non-persistent (portal-vacuumed `carried` guns are already
    // persistent/invisible in GML and never counted), desert subarea 1.
    if run.floor == 1 && run.floor_in_area == 1 && run.area == AreaId::Desert {
        let mut swords = 0u32;
        for pickup in &weapon_q {
            // GML `wep_black_sword` is 121 (`macros_general.gml:481`);
            // 46 is `wep_chicken_sword`.
            if matches!(
                pickup.kind,
                PickupKind::Weapon(w) if w == crate::data::WEAPON_BLACK_SWORD
            ) {
                swords += 1;
            }
        }
        run.blackswords += swords;
    }

    // GML `GameCont/Other_5:143-152` chest counters: unopened weapon/rad
    // caches feed `nochest`/`noradch` (desert 1-1 exempts `nochest`),
    // and every level ages `same_weapons_for`. `instance_exists` is
    // hierarchy-inclusive, so `CursedBigChest` and `GoldChest` (both
    // `WeaponChest` descendants) count toward `nochest` too, and a plain
    // `RadChest` is a `Prop` in the port rather than a `Pickup`.
    {
        let mut weapon_left = false;
        let mut rad_left = !rad_props.is_empty();
        for pickup in &weapon_q {
            match pickup.kind {
                PickupKind::Chest(
                    ChestKind::Weapon
                    | ChestKind::BigWeapon
                    | ChestKind::CursedBig
                    | ChestKind::Gold,
                ) => weapon_left = true,
                PickupKind::Chest(ChestKind::RadBig | ChestKind::RadMaggot) => rad_left = true,
                _ => {}
            }
        }
        run.same_weapons_for += 1;
        if weapon_left && !(run.area == AreaId::Desert && run.floor_in_area == 1) {
            run.nochest += 1;
        }
        if rad_left {
            run.noradch += 1;
        } else {
            run.noradch = 0;
        }
    }

    // GML `ProtoChest/Other_5.gml` (Room End = save) verbatim:
    //   if (wep == wep_frog_pistol) { wep = wep_golden_frog_pistol }
    //   if (sprite_index == sprProtoChestOpen) { protowep = wep_rusty_revolver;
    //                                          protocurse = false }
    //   else { protowep = wep; protocurse = curse }
    // The carried state rides the entity between floors; the persisted
    // half is `etc.protowep` (`scrSave.gml:44`), loaded at run start by
    // `PlayButton/Other_10.gml:11`.
    let mut proto_carriers = Vec::new();
    let mut stored = (run.protowep, run.protocurse);
    for (entity, mut state, opened) in &mut proto_chests {
        let mut weapon = if opened.is_some() {
            crate::data::WEAPON_RUSTY_REVOLVER
        } else {
            match &*state {
                ProtoChestState::Pending => run.protowep,
                ProtoChestState::Armed { weapon, .. } | ProtoChestState::Carried { weapon, .. } => {
                    *weapon
                }
            }
        };
        let cursed = if opened.is_some() {
            false
        } else {
            match &*state {
                ProtoChestState::Pending => run.protocurse,
                ProtoChestState::Armed { cursed, .. } | ProtoChestState::Carried { cursed, .. } => {
                    *cursed
                }
            }
        };
        if weapon == crate::decide_wep::FROG_PISTOL {
            weapon = crate::decide_wep::GOLDEN_FROG_PISTOL;
        }
        *state = ProtoChestState::Carried { weapon, cursed };
        stored = (weapon, cursed);
        proto_carriers.push(entity);
        commands
            .entity(entity)
            .remove::<LevelCleanup>()
            .remove::<Pickup>()
            .remove::<OpenedChest>()
            .remove::<crate::anim::SpriteAnim>();
    }
    if stored.0 != run.protowep || stored.1 != run.protocurse {
        run.protowep = stored.0;
        run.protocurse = stored.1;
        save.protowep = stored.0;
        dirty.0 = true;
    }

    for e in &level_q {
        if !proto_carriers.contains(&e) {
            commands.entity(e).despawn();
        }
    }
    commands.entity(portal_e).despawn();

    let prev_loop = run.loop_count;
    let looped = try_apply_loop_portal_transition(&mut run, &mut loop_transition, &mut trauma);

    if looped {
        // GML `GameCont/Other_5`: loop 2 unlocks hardmode.
        if run.loop_count >= 2 && !save.hardmode_unlocked {
            save.hardmode_unlocked = true;
            dirty.0 = true;
            toast.show("HARDMODE UNLOCKED");
            if crate::savedata_part::unlock_achievement(&mut save, 40) {
                toast.show("GO HARD");
            }
        }
        // GML `scrUnlocksWinOrLoop`/`GAME_LOOPED`: looping uncurses the
        // player and earns THE STRUGGLE CONTINUES.
        inv.cursed = [false, false, false];
        if crate::savedata_part::unlock_achievement(&mut save, 41) {
            dirty.0 = true;
            toast.show("THE STRUGGLE CONTINUES");
        }
        // GML `ctot_loop[race]`: looping as a race counts toward Fish/B.
        save.race_looped.insert(race, true);
        unlock_held_crown(race, player.crown, &mut save, &mut dirty, &mut toast);
        commands.spawn((GameCleanup, QueuedReactiveCue(ReactiveCue::LoopComplete)));
    }

    let entered_secret = if looped {
        None
    } else {
        apply_secret_transition(
            &mut run,
            &mut route.triggers,
            &mut route.crib,
            &mut save,
            &mut dirty,
        )
    };

    // GML `IceFlower/Step_0:14-19`: the jungle secret eats the Last Wish
    // and hands the skill point back, so the next pick is free.
    if entered_secret == Some(SecretTarget::Jungle)
        && player.mutations.contains(&MutationId::LastWish)
    {
        player.mutations.retain(|m| *m != MutationId::LastWish);
        player.last_wish_used = false;
        player.mutation_picks_owed = player.mutation_picks_owed.saturating_add(1);
    }

    // GML `GenCont/Destroy_0:112`: the room being *entered* is
    // `area_city` subarea 1 and the player still has `mut_last_wish` ->
    // that room's generation seeds an `IceFlower` (450 HP, `feed = 0`).
    if run.area == AreaId::FrozenCity
        && run.floor_in_area == 1
        && player.mutations.contains(&MutationId::LastWish)
    {
        commands.insert_resource(IceFlowerSeed(true));
    }

    if let Some(secret) = entered_secret {
        commands.spawn((GameCleanup, QueuedReactiveCue(ReactiveCue::SecretFound)));
        toast.show(&format!("ENTERING {}", secret_name(secret)));
    } else if !looped {
        commands.spawn((GameCleanup, QueuedReactiveCue(ReactiveCue::PortalEnter)));
        toast.show(&format!(
            "FLOOR {}-{}",
            run.world,
            crate::worldgen::floor_in_world(run.floor)
        ));
    } else {
        toast.show(&format!("LOOP {}", run.loop_count));
    }
    run.push_waypoint();
    if run.loop_count > prev_loop {
        save.total_loops += 1;
        dirty.0 = true;
    }

    if player.strong_spirit_spent {
        player.strong_spirit_area_cleared = true;
    }
    let _ = &mut health;

    if player.ultra_pick_owed || player.mutation_picks_owed > 0 {
        deferred.0 = true;
        begin_between_floor_skill_picks(&mut commands, &mut player, race, &mut paused);
        if player.mutation_picks_owed == 0 && !player.ultra_pick_owed {
            if !paused.0 {
                deferred.0 = false;
            } else {
                player_pos.0 = glam::Vec2::new(10000.0, 10000.0);
                return;
            }
        } else {
            player_pos.0 = glam::Vec2::new(10000.0, 10000.0);
            return;
        }
    }

    deferred.0 = false;
    try_start_pending_floor_gen(&mut commands, &run);

    player_pos.0 = glam::Vec2::new(10000.0, 10000.0);
}

/// GML `scrUnlocksWinOrLoop` (`scripts/scrUnlocks.gml:243-245`): holding
/// a crown through a loop or a win unlocks it for the race, and a fresh
/// unlock earns CROWN LIFE. Crowns are *not* unlocked when the pedestal
/// is taken (`CrownPickup/Collision_Player` has no unlock); VAULT_RAIDER
/// is menu-only (`scrLoadoutMenuInit.gml:27`).
fn unlock_held_crown(
    race: RaceId,
    crown: CrownKind,
    save: &mut SaveData,
    dirty: &mut SaveDirty,
    toast: &mut Toast,
) {
    if crown == CrownKind::None {
        return;
    }
    let gml = crate::crown::crown_port_to_gml(crown as u8);
    // GML `scrCrownUnlock:143-146` bails on an unavailable race or an
    // already-unlocked crown, and only then does the caller earn
    // CROWN LIFE.
    if !save.race_unlocked(race) || save.crown_unlocked(race, gml) {
        return;
    }
    save.unlock_crown(race, gml);
    if crate::savedata_part::unlock_achievement(save, 22) {
        toast.show("CROWN LIFE");
    }
    dirty.0 = true;
}

/// Between-floor skill picks. Private core of the deferred
/// `begin_between_floor_skill_picks` (pause + pending-pick resources);
/// the public system with mutation-pick UI state lands with the UI
/// phase.
fn begin_between_floor_skill_picks(
    commands: &mut Commands,
    player: &mut Player,
    race: RaceId,
    paused: &mut crate::state::Paused,
) {
    if player.ultra_pick_owed && player.ultra.is_none() && player.level >= 10 {
        paused.0 = true;
        let choices = ultra_choices_for_crown(race, player.crown == CrownKind::Destiny);
        commands.insert_resource(PendingUltra { choices });

        return;
    }

    if player.mutation_picks_owed > 0 {
        let choices = roll_mutations_for(player, race);
        if choices.is_empty() {
            while player.mutation_picks_owed > 0 {
                let c = roll_mutations_for(player, race);
                if c.is_empty() {
                    player.mutation_picks_owed = player.mutation_picks_owed.saturating_sub(1);
                } else {
                    paused.0 = true;
                    commands.insert_resource(PendingMutation { choices: c });
                    return;
                }
            }
            paused.0 = false;
            return;
        }
        paused.0 = true;
        commands.insert_resource(PendingMutation { choices });
    }
}

/// GML `SitDown`: sitting on the fallen throne (E at the sit zone)
/// locks the run into the 345-tick win beat, then ends the run won
/// with the victory toast and totals (sit/go-sit art omitted).
pub fn tick_throne_sit(
    time: Res<SimTime>,
    mut commands: Commands,
    input: ResMut<crate::input::NtInput>,
    mut run: ResMut<Run>,
    mut save: ResMut<SaveData>,
    mut dirty: ResMut<SaveDirty>,
    mut toast: ResMut<Toast>,
    mut player_q: Query<(Entity, &Pos, &mut Velocity, &RaceState, &Player), With<Player>>,
    mut sit_q: Query<(Entity, &mut ThroneSit), Without<SitZone>>,
    zones: Query<&Pos, With<SitZone>>,
) {
    let Ok((player_e, ppos, mut pvel, race_state, player)) = player_q.single_mut() else {
        return;
    };
    if let Ok((_, mut sit)) = sit_q.single_mut() {
        // Seated: hold still through the beat.
        pvel.0 = glam::Vec2::ZERO;
        sit.timer.tick(time.delta_secs);
        if sit.timer.just_finished() {
            commands.entity(player_e).remove::<ThroneSit>();
            run.game_over = true;
            run.won = true;
            if save.best_floor < run.floor {
                save.best_floor = run.floor;
            }
            save.total_runs += 1;
            save.total_kills = save.total_kills.saturating_add(run.total_kills);
            save.total_wins += 1;
            if run.hardmode {
                save.hard_runs += 1;
            }
            save.total_time_steps = save.total_time_steps.saturating_add(run.tottimer as u64);
            let race_gml = race_state.race as u8;
            save.win_streak_cur += 1;
            if save.win_streak_cur > save.win_streak_best {
                save.win_streak_best = save.win_streak_cur;
                save.best_streak_race = race_gml;
            }
            if save.best_time_steps == 0 || run.tottimer < save.best_time_steps {
                save.best_time_steps = run.tottimer;
                save.best_time_race = race_gml;
            }
            crate::savedata_part::update_best_run_stats(
                &mut save,
                race_gml,
                crate::worldgen::gml_area_from_run(&run),
                run.floor_in_area,
                run.loop_count,
                run.total_kills,
                run.hardmode,
            );
            dirty.0 = true;
            toast.show("THE STRUGGLE IS OVER");
        }
        return;
    }
    if !input.peek_interact_pressed() {
        return;
    }
    // Tutorial runs have no throne room yet; the unguarded take above
    // would eat the E press the weapon chest needs (pickup starvation).
    if run.tutorial {
        return;
    }
    let near = zones.iter().any(|z| z.0.distance(ppos.0) < 28.0);
    if near {
        commands.entity(player_e).insert(ThroneSit {
            timer: GTimer::from_seconds(345.0 / 30.0, TimerMode::Once),
        });
        run.won = true;
        toast.show("YOU SIT ON THE THRONE");
        // GML `SitDown/Other_7:22`: the win unlocks run as it starts.
        unlock_held_crown(
            race_state.race,
            player.crown,
            &mut save,
            &mut dirty,
            &mut toast,
        );
    }
}

/// Loading-screen floor transition. Stage 1 fills the progress bar;
/// stage 2 (after a 4-tick beat) flips run-side state: fresh plan +
/// Open Mind bonus + full `spawn_level` (mask, walls, props, chests,
/// enemies, bosses), `FloorStarted` event, +1 HP, strong-spirit
/// recharge, headless reset, player placement, carried-weapon drops,
/// juice. Bevy `progression.rs` stage-2 law verbatim.
/// Everything `spawn_level` needs about the run it is entering. Bundled
/// because `tick_floor_transition` sits at Bevy's 16-parameter cap.
#[derive(bevy_ecs::system::SystemParam)]
pub struct RoomEntry<'w, 's> {
    pub open_mind: Res<'w, OpenMind>,
    pub scarier: Res<'w, ScarierFace>,
    pub heavy_heart: Res<'w, HeavyHeart>,
    /// GML `GameCont.crownvisits`, which sets how many vault statues guard
    /// the pedestal (`CrownPickup/Create_0.gml:37-50`).
    pub triggers: Res<'w, crate::secrets::SecretTriggers>,
    /// GML `GenCont/Destroy_0.gml:112`: FrozenCity subarea 1 with LAST WISH
    /// turns a prop into the Jungle's only entrance.
    pub ice_flower: Option<Res<'w, IceFlowerSeed>>,
    /// GML `instance_exists(CrownObject)`, the second vault-statue gate.
    pub crowns: Query<'w, 's, Entity, With<crate::comps_b::CrownObject>>,
    pub portals: Query<'w, 's, Entity, With<crate::idpd::IdpdSpawnPortal>>,
}

pub fn tick_floor_transition(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut run: ResMut<Run>,
    mut mask: ResMut<FloorMask>,
    mut tops: ResMut<crate::comps_a::TopSmalls>,
    mut ft: ResMut<FloorTransition>,
    mut trauma: ResMut<Trauma>,
    mut chroma: ResMut<ChromaticAberration>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut floor_started: ResMut<Queue<FloorStarted>>,
    mut player_q: Query<(&mut Pos, &mut Health, &mut Player, &RaceState), With<Player>>,
    mut carried: ResMut<PortalCarriedWeapons>,
    room: RoomEntry,
) {
    let open_mind = room.open_mind;
    let scarier = room.scarier;
    let heavy_heart = room.heavy_heart;
    if !ft.active {
        return;
    }
    match ft.stage {
        1 => {
            ft.progress = (ft.progress + time.delta_secs * 0.85).min(1.0);
            if ft.progress >= 1.0 {
                ft.stage = 2;
                ft.timer = GTimer::from_seconds(4.0 / 30.0, TimerMode::Once);
            }
        }
        2 => {
            ft.timer.tick(time.delta_secs);
            if !ft.timer.just_finished() {
                return;
            }
            let Ok((mut pos, mut health, mut player, race)) = player_q.single_mut() else {
                return;
            };
            let crown_kind = player.crown;
            let player_ultra = player.ultra;
            // The tutorial arena never recurs past the first floor, and the
            // Blood-crown enemy pass follows the live crown (GML
            // `scrCrownCheck` at populate time).
            run.tutorial = false;
            run.blood_crown = player.crown == crate::data::CrownKind::Blood;
            let mut plan = crate::worldgen::generate_level(&run);
            // Player landing first (plan-space; `spawn_level` builds the
            // same mask from these cells, so placement below agrees).
            let landing = plan
                .floor_cells
                .iter()
                .min_by_key(|c| {
                    (((c.0 * 32 + 16) as f32).hypot((c.1 * 32 + 16) as f32) * 1000.0) as i32
                })
                .map(|(cx, cy)| glam::Vec2::new(*cx as f32 * 32.0 + 16.0, *cy as f32 * 32.0 + 16.0))
                .unwrap_or(glam::Vec2::ZERO);
            // GML `scrPopChests` (replaces the old Open-Mind top-up: the
            // bonus only widens the trim, exactly like GML). Seeded off
            // the run's Generation stream so portal floors permute
            // deterministically.
            let perm = crate::worldgen::apply_chest_permutations(
                &mut plan,
                crate::worldgen::ChestPermuteCtx {
                    area: run.area,
                    loops: run.loop_count,
                    subarea: run.floor_in_area,
                    rogue_in_run: race.race == crate::data::RaceId::Rogue,
                    player_half_health: health.hp * 2 < health.max,
                    crown_life: player.crown == crate::data::CrownKind::Life,
                    crown_love: player.crown == crate::data::CrownKind::Love,
                    open_mind: open_mind.0,
                    noradch: run.noradch,
                    nochest: run.nochest,
                    same_weapons_for: run.same_weapons_for,
                    horror_done: run.horror,
                    hardmode: run.hardmode,
                    tutorial: run.tutorial,
                    player_pos: landing,
                    seed: run.gen_seed ^ 0xC0E5_75EED,
                },
            );
            if perm.horror {
                run.horror = true;
            }
            let live_portals = room.portals.iter().count() as u32;
            crate::setup::spawn_level(
                &mut commands,
                &catalog,
                &mut run,
                scarier.0,
                heavy_heart.0,
                crown_kind,
                player_ultra,
                &plan,
                live_portals,
                &mut mask,
                &mut tops,
                room.triggers.vaults_entered,
                !room.crowns.is_empty(),
                room.ice_flower.is_some_and(|s| s.0),
            );
            commands.remove_resource::<IceFlowerSeed>();

            floor_started.push(FloorStarted {
                floor: run.floor,
                area: run.area,
            });
            health.hp = (health.hp + 1).min(health.max);
            try_recharge_strong_spirit(&mut player, &health);
            // GML Gun Warrant: exiting a portal starts 7 s of free ammo.
            if player.ultra == Some(crate::data::UltraMutationId::FishGunWarrant) {
                player.warrant = 210.0;
                player.free_ammo = true;
            }
            if crate::savedata_part::character_def(race.race).passive
                == crate::savedata_part::PassiveKind::Headless
            {
                player.headless_ready = true;
            }
            if let Some(c) = mask
                .cells
                .iter()
                .min_by_key(|c| (mask.cell_center(**c).length() * 1000.0) as i32)
            {
                pos.0 = mask.cell_center(*c);
            } else {
                pos.0 = glam::Vec2::ZERO;
            }
            run.portal_open = false;
            ft.active = false;

            // GML `GenCont/Destroy_0.gml:169-176`: on the campfire at
            // loop 1 the Fish gets a guitar dropped on the spawn tile
            // (electric variant on skin C). `scrWeaponPickupCreate`'s
            // `_has_ammo` defaults false, so it carries no ammo.
            if run.area == crate::data::AreaId::Campfire
                && run.loop_count == 1
                && race.race == crate::data::RaceId::Fish
            {
                let guitar = if race.skin == crate::data::SkinLetter::C {
                    crate::data::WEAPON_ELECTRIC_GUITAR
                } else {
                    crate::data::WEAPON_GUITAR
                };
                crate::pickups::spawn_pickup(
                    &mut commands,
                    &catalog,
                    PickupKind::Weapon(guitar),
                    pos.0,
                    0,
                    false,
                );
            }

            // GML `GenCont/Destroy_0.gml:178-182`: a desert entry with
            // banked `blackswords` re-drops that many Black Swords on the
            // spawn tile and zeroes the counter.
            if run.area == crate::data::AreaId::Desert
                && run.loop_count > 0
                && run.blackswords > 0
            {
                for _ in 0..run.blackswords {
                    crate::pickups::spawn_pickup(
                        &mut commands,
                        &catalog,
                        PickupKind::Weapon(crate::data::WEAPON_BLACK_SWORD),
                        pos.0,
                        0,
                        false,
                    );
                }
                run.blackswords = 0;
            }

            // GML `GenCont/Destroy:186-187` verbatim:
            // `instance_destroy(SpiralCont)` at generation end. The
            // view spiral dies in the lifecycle step; the sim
            // `SpiralCtl` (ambience-duck presence) warms per
            // generation (`try_start_pending_floor_gen`) and dies
            // here, never leaking a live cont into settled play.
            commands.remove_resource::<crate::vortex::SpiralCtl>();

            if !carried.0.is_empty() {
                let base = pos.0;
                for (i, w) in carried.0.drain(..).enumerate() {
                    let ang = (i as f32) * std::f32::consts::TAU / 4.0;
                    crate::pickups::spawn_pickup(
                        &mut commands,
                        &catalog,
                        PickupKind::Weapon(w),
                        base + glam::Vec2::new(ang.cos(), ang.sin()) * 24.0,
                        0,
                        false,
                    );
                }
            }

            trauma.add(0.55);
            chromatic_pulse(&mut chroma, 0.65);
            audio.play_portal(&mut cues);
        }
        _ => {}
    }
}

/// Random loading-screen tip (bevy `pick_loading_tip` verbatim).
pub fn pick_loading_tip(_run: &Run) -> String {
    const TIPS: &[&str] = &[
        "KILL ENEMIES TO LEVEL UP",
        "MUTATIONS STACK AFTER EACH LEVEL",
        "PORTALS OPEN WHEN THE AREA IS CLEAR",
        "WATCH YOUR AMMO - REVOLVER IS 3 DMG",
        "HOLD SHIFT TO AIM SLOWLY",
        "BOILING VEINS SAVES YOU AT LOW HP",
        "RHINO SKIN GIVES +4 MAX HP",
        "YOU CAN CARRY TWO WEAPONS",
        "RAD CANISTERS DROP FROM STRONG ENEMIES",
        "LOOP TO FIND NEW MUTATIONS",
    ];
    let mut rng = rand::rng();
    TIPS[rng.random_range(0..TIPS.len())].to_string()
}

/// Portal Disappear-end: flip the level via a near-instant
/// `PortalSucking` (the player is already at the portal core, so the
/// lerp is a no-op and `tick_portal_suck` performs the floor advance).
pub fn kick_portal_transition(
    commands: &mut Commands,
    player_q: &mut Query<(Entity, &Pos, Option<&PortalSucking>), (With<Player>, Without<Portal>)>,
    portal_e: Entity,
    portal_pos: glam::Vec2,
    run: &Run,
) {
    if run.game_over {
        return;
    }
    let Ok((player_e, ppos, suck_opt)) = player_q.single_mut() else {
        return;
    };
    if suck_opt.is_some() {
        return;
    }
    let start = ppos.0;
    commands.entity(player_e).insert(PortalSucking {
        portal: portal_e,
        timer: GTimer::from_seconds(0.1, TimerMode::Once),
        start_pos: start,
        target_pos: portal_pos,
    });
}

/// Portal state machine, gameplay half: Spawn (strip-gated) -> Idle +
/// second shock, Idle endgame countdown -> Disappear (strip-gated),
/// Disappear/close-expiry -> floor flip via `kick_portal_transition`.
/// Facing flips and ambient particles are render-phase work and omitted.
pub fn animate_portal(
    time: Res<SimTime>,
    mut commands: Commands,
    run: Res<Run>,
    catalog: Res<repame_anim::AnimCatalog>,
    mut player_q: Query<(Entity, &Pos, Option<&PortalSucking>), (With<Player>, Without<Portal>)>,
    mut q: Query<
        (
            Entity,
            &Pos,
            &mut PortalState,
            Option<&mut PortalClosing>,
            Option<&mut crate::anim::SpriteAnim>,
        ),
        With<Portal>,
    >,
) {
    let dt = time.delta_secs;
    let frames = dt * crate::SIM_HZ as f32;

    for (portal_e, portal_pos, mut st, mut closing_opt, mut anim_opt) in &mut q {
        let pos = portal_pos.0;

        if let Some(ref mut closing) = closing_opt {
            closing.timer.tick(dt);
            if closing.timer.just_finished() {
                kick_portal_transition(&mut commands, &mut player_q, portal_e, pos, &run);
            }
        }

        // Bevy gates Spawn on the spawn-strip oneshot finishing
        // (`finished || frame + 1 >= frames`; no strip → immediately),
        // then swaps the anim to the looping idle strip.
        if st.phase == PortalPhase::Spawn {
            let finished = anim_opt
                .as_ref()
                .map(|a| a.oneshot && (a.finished || a.frame + 1 >= a.frames.max(1)))
                .unwrap_or(true);
            if finished {
                if let (Some(anim), Some(def)) = (
                    anim_opt.as_mut(),
                    catalog.def(PortalState::idle_sprite(st.kind)),
                ) {
                    anim.set_path(PortalState::idle_sprite(st.kind), def, false);
                }
                st.phase = PortalPhase::Idle;
                st.phase = PortalPhase::Idle;
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    PortalShock {
                        timer: GTimer::from_seconds(2.0 / 30.0, TimerMode::Once),
                        radius: 72.0,
                    },
                    Pos(pos),
                ));
            }
            continue;
        }

        // Bevy gates Disappear on the disappear-strip oneshot the
        // same way (no strip → immediately).
        if st.phase == PortalPhase::Disappear {
            let done = anim_opt
                .as_ref()
                .map(|a| a.oneshot && (a.finished || a.frame + 1 >= a.frames.max(1)))
                .unwrap_or(true);
            if done {
                kick_portal_transition(&mut commands, &mut player_q, portal_e, pos, &run);
            }
            continue;
        }

        if st.phase != PortalPhase::Idle {
            continue;
        }

        if st.endgame < 100.0 {
            st.endgame -= frames;
            if st.endgame < 0.0 {
                st.phase = PortalPhase::Disappear;
                st.anim = GTimer::from_seconds(0.75, TimerMode::Once);
                if let (Some(anim), Some(def)) = (
                    anim_opt.as_mut(),
                    catalog.def(PortalState::disappear_sprite(st.kind)),
                ) {
                    anim.set_path(PortalState::disappear_sprite(st.kind), def, true);
                }
                continue;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unlocks, saves, run lifecycle.
// ---------------------------------------------------------------------------

/// Floor-reach race unlocks (Crystal at 4, Eyes at 5, …), once per
/// floor. Marks the save dirty and toasts each unlock.
pub fn apply_floor_reach_unlocks(
    mut applied: Local<u32>,
    run: Res<Run>,
    mut dirty: ResMut<SaveDirty>,
    mut toast: ResMut<Toast>,
    mut save: ResMut<SaveData>,
) {
    if run.floor <= *applied {
        return;
    }
    *applied = run.floor;
    let unlocked = crate::savedata_part::check_progress_unlocks(
        &mut save,
        run.floor,
        run.loop_count,
        false,
        false,
        false,
    );
    if !unlocked.is_empty() {
        dirty.0 = true;
        for race in unlocked {
            toast.show(&format!(
                "UNLOCKED {}",
                crate::savedata_part::character_def(race)
                    .name
                    .to_ascii_uppercase()
            ));
        }
    }
}

/// Periodic dirty-save flush: accumulate while dirty, clear the flag
/// every 5 s. The actual file write lands with the save phase (the sim
/// has no `SaveManager`), so clearing the flag IS the flush here.
pub fn flush_dirty_save(
    mut accumulator: Local<f32>,
    time: Res<SimTime>,
    mut dirty: ResMut<SaveDirty>,
) {
    if !dirty.0 {
        return;
    }
    *accumulator += time.delta_secs;
    if *accumulator >= 5.0 {
        *accumulator = 0.0;
        dirty.0 = false;
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------
