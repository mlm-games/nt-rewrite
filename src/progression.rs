//! Level/mutation/portal progression. Port of
//! `nt-recreated-bevy/src/game/progression.rs` minus the three deferred
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
use crate::comps_a::{
    Euphoria, FloorMask, FloorStarted, GameCleanup, Health, HeavyHeart, Inventory, LevelCleanup,
    MutationChoice, OpenMind, PendingMutation, PendingUltra, Player, Projectile, RaceState, Run,
    SaveDirty, ScarierFace, Team, Toast, Velocity,
};
use crate::comps_b::{
    ChestKind, Enemy, FloorTransition, GroundPhysics, LoopTransition, OpenedChest, Pickup,
    PickupCurse, PickupKind, Portal, PortalCarriedWeapons, PortalClear, PortalClosing, PortalPhase,
    PortalShock, PortalState, PortalSucking, Prop, PropSprites, SecretEntrance, SitZone, ThroneSit,
};
use crate::data::{
    AmmoKind, AreaId, CrownKind, MutationId, RaceId, SecretTarget, UltraMutationId, ammo_max,
    ammo_pickup_amount, area_for_floor, route_coordinates,
};
use crate::effects::{
    ChromaticAberration, FlashWhite, SlowMotion, chromatic_pulse, flash_white, slow_motion,
    spawn_burst,
};
use crate::enemy_data::enemy_def;
use crate::environment::{PropDeathEffect, spawn_prop_corpse, spawn_prop_death_effect};
use crate::msg::Queue;
use crate::savedata_part::SaveData;
use crate::spatial::Pos;
use crate::state::{AppState, Paused};
use crate::time::{GTimer, TimerMode};

// ---------------------------------------------------------------------------
// Mutation data (byte-exact tables from bevy `content.rs`).
// ---------------------------------------------------------------------------

/// All 29 mutations in bevy `ALL_MUTATIONS` order.
pub const ALL_MUTATIONS: [MutationId; 29] = [
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
    MutationId::HeavyHeart,
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

/// Mutation display name + description (bevy `mutation_def` parity).
pub fn mutation_name(id: MutationId) -> (&'static str, &'static str) {
    match id {
        MutationId::RhinoSkin => ("Rhino Skin", "+4 max HP"),
        MutationId::PlutoniumHunger => ("Plutonium Hunger", "Much larger pickup range"),
        MutationId::TriggerFingers => ("Trigger Fingers", "Kills lower reload time"),
        MutationId::RabbitPaw => ("Rabbit Paw", "Better chance for drops"),
        MutationId::SecondStomach => ("Second Stomach", "Medkits heal double"),
        MutationId::ScarierFace => ("Scarier Face", "Enemies have less HP"),
        MutationId::BoilingVeins => ("Boiling Veins", "Explosions can't drop you below 4 HP"),
        MutationId::ImpactWrists => ("Impact Wrists", "Weapons knock back harder"),
        MutationId::ExtraFeet => ("Extra Feet", "Move faster"),
        MutationId::Bloodlust => ("Bloodlust", "Kills sometimes heal you"),
        MutationId::LuckyShot => ("Lucky Shot", "Kills sometimes drop ammo"),
        MutationId::GammaGuts => ("Gamma Guts", "Enemies that touch you take damage"),
        MutationId::BackMuscle => ("Back Muscle", "Higher ammo capacity"),
        MutationId::Euphoria => ("Euphoria", "Enemy bullets are slower"),
        MutationId::LongArms => ("Long Arms", "Melee attacks reach further"),
        MutationId::Stress => ("Stress", "Fire faster at low health"),
        MutationId::EagleEyes => ("Eagle Eyes", "Better accuracy"),
        MutationId::OpenMind => ("Open Mind", "More chests spawn"),
        MutationId::HeavyHeart => ("Heavy Heart", "More weapon drops"),
        MutationId::StrongSpirit => ("Strong Spirit", "Prevents death, once"),
        MutationId::SharpTeeth => ("Sharp Teeth", "Damage taken also hurts nearby enemies"),
        MutationId::LastWish => ("Last Wish", "Heal and refill ammo when low"),
        MutationId::BoltMarrow => ("Bolt Marrow", "Bolts seek targets"),
        MutationId::Hammerhead => ("Hammerhead", "Chew through destructible props"),
        MutationId::LaserBrain => ("Laser Brain", "Energy weapons hit harder"),
        MutationId::RecycleGland => ("Recycle Gland", "Bullet weapons sometimes refund ammo"),
        MutationId::ShotgunShoulders => ("Shotgun Shoulders", "Shells bounce off walls"),
        MutationId::ThroneButt => ("Throne Butt", "Active ability is upgraded"),
        MutationId::Patience => ("Patience", "Skip now; get more choices next time"),
    }
}

/// Ultra choice pair per race (bevy `ultra_choices_for` parity).
pub fn ultra_choices_for(race: RaceId) -> [UltraMutationId; 2] {
    match race {
        RaceId::Fish | RaceId::Random => [
            UltraMutationId::FishGunWarrant,
            UltraMutationId::FishConfiscate,
        ],
        RaceId::Crystal => [
            UltraMutationId::CrystalFortress,
            UltraMutationId::CrystalJuggernaut,
        ],
        RaceId::Eyes => [
            UltraMutationId::EyesMonsterStyle,
            UltraMutationId::EyesProjectileStyle,
        ],
        RaceId::Melting => [
            UltraMutationId::MeltingBrainCapacity,
            UltraMutationId::MeltingDetachment,
        ],
        RaceId::Plant => [UltraMutationId::PlantTrapper, UltraMutationId::PlantKiller],
        RaceId::Venuz => [
            UltraMutationId::VenuzBack2Bizniz,
            UltraMutationId::VenuzGunGod,
        ],
        RaceId::Steroids => [
            UltraMutationId::SteroidsAmbidextrous,
            UltraMutationId::SteroidsGetArmed,
        ],
        RaceId::Robot => [
            UltraMutationId::RobotRefinedTaste,
            UltraMutationId::RobotRegurgitate,
        ],
        RaceId::Chicken => [
            UltraMutationId::ChickenHarderToKill,
            UltraMutationId::ChickenDetermination,
        ],
        RaceId::Rebel => [
            UltraMutationId::RebelPersonalGuard,
            UltraMutationId::RebelRiot,
        ],
        RaceId::Horror => [
            UltraMutationId::HorrorStalker,
            UltraMutationId::HorrorAnomaly,
        ],
        RaceId::Rogue => [
            UltraMutationId::RogueSuperBlastArmor,
            UltraMutationId::RoguePortalStrike,
        ],
        RaceId::BigDog => [
            UltraMutationId::BigDogHeavyArtillery,
            UltraMutationId::BigDogGuardian,
        ],
        RaceId::Skeleton => [
            UltraMutationId::SkeletonBloodArmor,
            UltraMutationId::SkeletonNecromancy,
        ],
        RaceId::Frog => [
            UltraMutationId::FrogToxicLord,
            UltraMutationId::FrogSwampBody,
        ],
        RaceId::Cuz => [UltraMutationId::CuzHoarder, UltraMutationId::CuzQuickSwap],
    }
}

/// Ultra display name + description (bevy `ultra_mutation_def` parity).
pub fn ultra_mutation_name(id: UltraMutationId) -> (&'static str, &'static str) {
    match id {
        UltraMutationId::FishGunWarrant => {
            ("Gun Warrant", "Faster gun handling and stronger rolls")
        }
        UltraMutationId::FishConfiscate => ("Confiscate", "Weapon pickups grant extra ammo"),
        UltraMutationId::CrystalFortress => ("Fortress", "Much more HP and longer shield"),
        UltraMutationId::CrystalJuggernaut => ("Juggernaut", "Move faster while protected"),
        UltraMutationId::EyesMonsterStyle => {
            ("Monster Style", "Telekinesis and pickup pull are stronger")
        }
        UltraMutationId::EyesProjectileStyle => {
            ("Projectile Style", "Enemy projectiles are slowed further")
        }
        UltraMutationId::MeltingBrainCapacity => {
            ("Brain Capacity", "Detonate reaches farther and hurts more")
        }
        UltraMutationId::MeltingDetachment => ("Detachment", "Gain emergency survivability"),
        UltraMutationId::PlantTrapper => ("Trapper", "Snare lasts longer and slows harder"),
        UltraMutationId::PlantKiller => ("Killer", "Move and fire faster"),
        UltraMutationId::VenuzBack2Bizniz => ("Back 2 Bizniz", "Pop Pop grants an extra charge"),
        UltraMutationId::VenuzGunGod => ("Ima Gun God", "Major fire-rate and accuracy boost"),
        UltraMutationId::SteroidsAmbidextrous => {
            ("Ambidextrous", "Faster fire and lower recoil feel")
        }
        UltraMutationId::SteroidsGetArmed => ("Get Armed", "Get Loaded refills more ammunition"),
        UltraMutationId::RobotRefinedTaste => {
            ("Refined Taste", "Ammo and weapon pickups heal more")
        }
        UltraMutationId::RobotRegurgitate => ("Regurgitate", "Eating weapons gives better rewards"),
        UltraMutationId::ChickenHarderToKill => {
            ("Harder To Kill", "Headless survival returns with more HP")
        }
        UltraMutationId::ChickenDetermination => ("Determination", "Thrown weapons hit harder"),
        UltraMutationId::RebelPersonalGuard => {
            ("Personal Guard", "Allies live longer and shoot faster")
        }
        UltraMutationId::RebelRiot => ("Riot", "Spawn more allies"),
        UltraMutationId::HorrorStalker => ("Stalker", "Beam and radiation effects are stronger"),
        UltraMutationId::HorrorAnomaly => ("Anomaly", "Energy weapons and pickups improve"),
        UltraMutationId::RogueSuperBlastArmor => {
            ("Super Blast Armor", "Explosion damage is greatly reduced")
        }
        UltraMutationId::RoguePortalStrike => {
            ("Ultra Portal Strike", "Portal strike is larger and faster")
        }
        UltraMutationId::BigDogHeavyArtillery => {
            ("Heavy Artillery", "Rocket barrage gains side rockets")
        }
        UltraMutationId::BigDogGuardian => ("Guardian", "Gain bulk and protection"),
        UltraMutationId::SkeletonBloodArmor => ("Blood Armor", "More HP and blood-fueled kills"),
        UltraMutationId::SkeletonNecromancy => {
            ("Necromancy", "Kills sometimes heal and refund ammo")
        }
        UltraMutationId::FrogToxicLord => ("Toxic Lord", "Toxic clouds are larger and longer"),
        UltraMutationId::FrogSwampBody => ("Swamp Body", "Gain bulk and blast resilience"),
        UltraMutationId::CuzHoarder => ("Hoarder", "Carry a full third weapon slot"),
        UltraMutationId::CuzQuickSwap => ("Quick Swap", "Swap ability is nearly instant"),
        UltraMutationId::CuzEmotional => ("Emotional", "Cry more tears, carry more ammo"),
    }
}

// ---------------------------------------------------------------------------
// Small local helpers (pure ports of bevy helpers owned by other phases).
// ---------------------------------------------------------------------------

/// Deferred floor generation flag (bevy `DeferredFloorGen` parity).
#[derive(Resource, Default)]
pub struct DeferredFloorGen(pub bool);

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

/// Floor to return to after leaving a secret area (bevy
/// `SecretTarget::return_floor` parity).
fn secret_return_floor(target: SecretTarget, current_floor: u32) -> u32 {
    match target {
        SecretTarget::Oasis | SecretTarget::PizzaSewers => 5,
        SecretTarget::YvMansion => 7,
        SecretTarget::CursedCaves => 9,
        SecretTarget::Jungle => 12,
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
) -> Option<SecretTarget> {
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
    let mut rng = rand::rng();
    roll_mutations_with(player, &mut rng)
}

/// Seeded roll (deterministic under a seeded RNG; tests use this).
pub fn roll_mutations_with(player: &mut Player, rng: &mut impl rand::RngExt) -> Vec<MutationId> {
    let mut pool: Vec<MutationId> = ALL_MUTATIONS
        .iter()
        .copied()
        .filter(|m| {
            if *m == MutationId::Patience && player.patience_used {
                return false;
            }

            !player.mutations.contains(m)
        })
        .collect();

    let mut out = Vec::new();

    let destiny = player.crown == CrownKind::Destiny;
    let want_base = if destiny { 1 } else { 4 };

    let want_base = if player.patience_bonus && !destiny {
        4
    } else {
        want_base
    };
    let want = pool.len().min(want_base);

    player.patience_bonus = false;

    for _ in 0..want {
        let idx = rng.random_range(0..pool.len());
        out.push(pool.remove(idx));
    }

    out
}

/// Mutation-pick UI state: flag resources written by `apply_mutation`.
#[derive(bevy_ecs::system::SystemParam)]
pub struct MutationFlagSet<'w> {
    pub scarier: ResMut<'w, ScarierFace>,
    pub euphoria: ResMut<'w, Euphoria>,
    pub open_mind: ResMut<'w, OpenMind>,
    pub heavy_heart: ResMut<'w, HeavyHeart>,
}

/// Screen-feel sinks shared by level-up paths.
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
                let choices = roll_mutations(&mut player);
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
            let choices = ultra_choices_for(race_state.race).to_vec();
            commands.insert_resource(PendingUltra { choices });
            flow.paused.0 = true;
            return;
        }

        if player.mutation_picks_owed > 0 {
            let choices = roll_mutations(&mut player);
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
    let (name, desc) = mutation_name(id);

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
            player.veins_threshold = 4;
        }
        MutationId::ImpactWrists => {
            player.knockback_mult *= 1.6;
        }
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
    toast.show(&format!("{name}: {desc}"));
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

    let (name, desc) = ultra_mutation_name(id);

    fx.trauma.add(0.55);
    chromatic_pulse(&mut fx.chroma, 0.4);
    slow_motion(&mut fx.slow_mo, 0.35, 0.5);
    audio.play_levelup(cues);
    toast.show(&format!("ULTRA - {name}: {desc}"));

    debug_assert!(
        ultra_choices_for(race_state.race).contains(&id) || race_state.race == RaceId::Random,
        "picked ultra {id:?} outside race {:?}",
        race_state.race,
    );
}

// ---------------------------------------------------------------------------
// Portals and floor transitions.
// ---------------------------------------------------------------------------

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
    enemy_shots: Query<(Entity, &Team), With<Projectile>>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    catalog: Res<repame_anim::AnimCatalog>,
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

    for (e, team) in &enemy_shots {
        if *team != Team::Player {
            commands.entity(e).despawn();
        }
    }

    run.portal_open = true;
    commands.spawn((GameCleanup, QueuedReactiveCue(ReactiveCue::PortalOpen)));

    let mut rng = rand::rng();
    let pos = mask.random_floor_pos(&mut rng, 80.0);

    let kind: u8 = match run.area {
        AreaId::HQ => 2,
        AreaId::Vault | AreaId::CrownVault => 3,
        _ => 1,
    };

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
        },
        Pos(pos),
    ));

    spawn_burst(
        &mut commands,
        &mut rng,
        pos,
        4,
        [0.5, 0.8, 1.0, 1.0],
        (60.0, 160.0),
    );

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

/// Portal spawn shock: destroys destructible props in radius (corpses +
/// death effects + secret queues), pops chests into their contents, and
/// clears enemy projectiles. Expires on its timer.
pub fn tick_portal_shock(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
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
    mut chests: Query<(Entity, &Pos, &Pickup), (Without<OpenedChest>, Without<Player>)>,
    mut enemy_shots: Query<(Entity, &Pos, &Team), With<Projectile>>,
    entrances: Query<&SecretEntrance>,
    mut secrets: ResMut<crate::secrets::SecretTriggers>,
    run: Res<Run>,
    player_q: Query<&Player>,
) {
    let hasted = player_q.single().is_ok_and(|p| p.crown == CrownKind::Haste);
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
            killed.push((prop_e, ppos, prop.explosive, death.copied(), ps.copied()));
        }
        for (prop_e, ppos, explosive, death, ps) in killed {
            if let Some(sprites) = ps {
                spawn_prop_corpse(&mut commands, &catalog, ppos, &sprites);
            }
            spawn_prop_death_effect(&mut commands, ppos, death, explosive, None);
            if let Ok(entrance) = entrances.get(prop_e) {
                secrets.queue(entrance.target);
            }
            commands.entity(prop_e).despawn();
        }

        for (chest_e, chest_pos, pickup) in &mut chests {
            let PickupKind::Chest(kind) = pickup.kind else {
                continue;
            };
            let cpos = chest_pos.0;
            if center.distance(cpos) > shock.radius {
                continue;
            }

            commands.entity(chest_e).insert(OpenedChest(kind));
            match kind {
                ChestKind::Weapon | ChestKind::Proto => {
                    let weapon = crate::pickups::random_weapon(&mut rand::rng());
                    crate::pickups::spawn_pickup(
                        &mut commands,
                        &catalog,
                        PickupKind::Weapon(weapon),
                        cpos,
                        0,
                        false,
                    );
                }
                ChestKind::Ammo => {
                    for _ in 0..2 {
                        crate::pickups::spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Ammo(AmmoKind::None, 0),
                            cpos,
                            run.loop_count,
                            hasted,
                        );
                    }
                }
                ChestKind::Health => {
                    crate::pickups::spawn_pickup(
                        &mut commands,
                        &catalog,
                        PickupKind::Medkit(4),
                        cpos,
                        0,
                        false,
                    );
                }
                ChestKind::CursedBig => {
                    let mut rng = rand::rng();
                    for _ in 0..3 {
                        let weapon = crate::pickups::random_weapon(&mut rng);
                        let e = crate::pickups::spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Weapon(weapon),
                            cpos + glam::Vec2::new(
                                rng.random_range(-2.0..2.0),
                                rng.random_range(-2.0..2.0),
                            ),
                            0,
                            false,
                        );
                        commands.entity(e).insert(crate::comps_b::PickupCurse);
                    }
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        crate::comps_b::PortalClear {
                            timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                        },
                        Pos(cpos),
                    ));
                }
                ChestKind::BigWeapon => {
                    let mut rng = rand::rng();
                    for _ in 0..3 {
                        let weapon = crate::pickups::random_weapon(&mut rng);
                        crate::pickups::spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Weapon(weapon),
                            cpos + glam::Vec2::new(
                                rng.random_range(-2.0..2.0),
                                rng.random_range(-2.0..2.0),
                            ),
                            0,
                            false,
                        );
                    }
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        crate::comps_b::PortalClear {
                            timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                        },
                        Pos(cpos),
                    ));
                }
                ChestKind::Rogue => {
                    for _ in 0..25 {
                        let ang = rand::rng().random_range(0.0..std::f32::consts::TAU);
                        let d = rand::rng().random_range(6.0..26.0);
                        crate::pickups::spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Rad(1),
                            cpos + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                            0,
                            false,
                        );
                    }
                }
                ChestKind::RadBig => {
                    for _ in 0..45 {
                        let ang = rand::rng().random_range(0.0..std::f32::consts::TAU);
                        let d = rand::rng().random_range(6.0..26.0);
                        crate::pickups::spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Rad(1),
                            cpos + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                            0,
                            false,
                        );
                    }
                }
                ChestKind::RadMaggot => {
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        crate::combat::Explosion {
                            timer: GTimer::from_seconds(0.05, TimerMode::Once),
                            radius: 70.0,
                            damage: 6,
                            team: crate::comps_a::Team::Enemy,
                            hits_player: true,
                            source: None,
                        },
                        Pos(cpos),
                    ));
                }
                ChestKind::Idpd => {
                    for _ in 0..8 {
                        crate::pickups::spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Ammo(AmmoKind::None, 0),
                            cpos,
                            run.loop_count,
                            false,
                        );
                    }
                }
                ChestKind::Rad => {
                    for _ in 0..25 {
                        // Bevy inline law (not `random_offset`): ring
                        // 6..26 px.
                        let ang = rand::rng().random_range(0.0..std::f32::consts::TAU);
                        let d = rand::rng().random_range(6.0..26.0);
                        crate::pickups::spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Rad(1),
                            cpos + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                            0,
                            false,
                        );
                    }
                }
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
        if clear.timer.just_finished() {
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
/// from the bevy build are omitted (render phase).
pub fn tick_portal_suck(
    time: Res<SimTime>,
    mut commands: Commands,
    mut run: ResMut<Run>,
    level_q: Query<Entity, With<LevelCleanup>>,
    weapon_q: Query<&Pickup>,
    _carried: Res<PortalCarriedWeapons>,
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
    mut loop_transition: ResMut<LoopTransition>,
    mut trauma: ResMut<Trauma>,
    mut toast: ResMut<Toast>,
    mut triggers: ResMut<crate::secrets::SecretTriggers>,
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

    // GML `Portal/Alarm_1`: ground `WepPickup` only — visible and
    // non-persistent (portal-vacuumed `carried` guns are already
    // persistent/invisible in GML and never counted), desert subarea 1.
    if run.floor == 1 && run.floor_in_area == 1 && run.area == AreaId::Desert {
        let mut swords = 0u32;
        for pickup in &weapon_q {
            if matches!(pickup.kind, PickupKind::Weapon(w) if w.0 == 46) {
                swords += 1;
            }
        }
        run.blackswords += swords;
    }

    // GML `GameCont/Other_5` chest counters: unopened weapon/rad
    // caches feed `nochest`/`noradch` (desert 1-1 exempts `nochest`),
    // and every level ages `same_weapons_for`.
    {
        let mut weapon_left = false;
        let mut rad_left = false;
        for pickup in &weapon_q {
            match pickup.kind {
                PickupKind::Chest(ChestKind::Weapon | ChestKind::BigWeapon) => weapon_left = true,
                PickupKind::Chest(ChestKind::Rad | ChestKind::RadBig | ChestKind::RadMaggot) => {
                    rad_left = true
                }
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

    for e in &level_q {
        commands.entity(e).despawn();
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
        commands.spawn((GameCleanup, QueuedReactiveCue(ReactiveCue::LoopComplete)));
    }

    let entered_secret = if looped {
        None
    } else {
        apply_secret_transition(&mut run, &mut triggers)
    };

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
        let choices = ultra_choices_for(race).to_vec();
        commands.insert_resource(PendingUltra { choices });

        return;
    }

    if player.mutation_picks_owed > 0 {
        let choices = roll_mutations(player);
        if choices.is_empty() {
            while player.mutation_picks_owed > 0 {
                let c = roll_mutations(player);
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
    mut input: ResMut<crate::input::NtInput>,
    mut run: ResMut<Run>,
    mut save: ResMut<SaveData>,
    mut dirty: ResMut<SaveDirty>,
    mut toast: ResMut<Toast>,
    mut player_q: Query<(Entity, &Pos, &mut Velocity, &RaceState), With<Player>>,
    mut sit_q: Query<(Entity, &mut ThroneSit), Without<SitZone>>,
    zones: Query<&Pos, With<SitZone>>,
) {
    let Ok((player_e, ppos, mut pvel, race_state)) = player_q.single_mut() else {
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
    if !input.take_interact_pressed() {
        return;
    }
    let near = zones.iter().any(|z| z.0.distance(ppos.0) < 28.0);
    if near {
        commands.entity(player_e).insert(ThroneSit {
            timer: GTimer::from_seconds(345.0 / 30.0, TimerMode::Once),
        });
        run.won = true;
        toast.show("YOU SIT ON THE THRONE");
    }
}

/// Loading-screen floor transition. Stage 1 fills the progress bar;
/// stage 2 (after a 4-tick beat) flips run-side state: fresh plan +
/// Open Mind bonus + full `spawn_level` (mask, walls, props, chests,
/// enemies, bosses), `FloorStarted` event, +1 HP, strong-spirit
/// recharge, headless reset, player placement, carried-weapon drops,
/// juice. Bevy `progression.rs` stage-2 law verbatim.
pub fn tick_floor_transition(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut run: ResMut<Run>,
    mut mask: ResMut<FloorMask>,
    mut ft: ResMut<FloorTransition>,
    mut trauma: ResMut<Trauma>,
    mut chroma: ResMut<ChromaticAberration>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut floor_started: ResMut<Queue<FloorStarted>>,
    mut player_q: Query<(&mut Pos, &mut Health, &mut Player, &RaceState), With<Player>>,
    mut carried: ResMut<PortalCarriedWeapons>,
    open_mind: Res<OpenMind>,
) {
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
            // bonus only widens the trim, exactly like GML).
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
                    player_pos: landing,
                },
            );
            if perm.horror {
                run.horror = true;
            }
            crate::setup::spawn_level(&mut commands, &catalog, &run, &plan, &mut mask);

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

/// Immediate single-fire save flush: clears a set dirty flag once;
/// later ticks are no-ops until something dirties again.
pub fn flush_dirty_save_once(mut dirty: ResMut<SaveDirty>) {
    if dirty.0 {
        dirty.0 = false;
    }
}

/// Boss HP bar readout: first boss enemy's (hp, max).
pub fn boss_info(q: &Query<(&Enemy, &Health), With<Enemy>>) -> Option<(u32, u32)> {
    for (enemy, health) in q {
        if enemy_def(enemy.kind).boss {
            return Some((health.hp.max(0) as u32, health.max as u32));
        }
    }
    None
}

/// Run teardown: despawn everything tagged `GameCleanup` and reset the
/// floor mask. Camera strips / damage numbers / juice particles from
/// the bevy build have no sim counterparts and are omitted.
pub fn cleanup_run(
    mut commands: Commands,
    q: Query<Entity, With<GameCleanup>>,
    mut mask: Option<ResMut<FloorMask>>,
) {
    for e in &q {
        commands.entity(e).despawn();
    }

    if let Some(m) = mask.as_mut() {
        **m = FloorMask::default();
    }
}

/// Full run teardown (bevy `mod.rs::teardown_game` parity).
/// `cleanup_run` covers `GameCleanup`, but sim floaters (`DamageNumber`)
/// and burst dots (`Particle`) spawn untagged — like the bevy
/// `DamageNumber`/`Particle`/`TrailGhost` queries — so they are drained
/// here too. (No `TrailGhost` equivalent exists in the sim.)
pub fn teardown_game(
    mut commands: Commands,
    q: Query<Entity, With<GameCleanup>>,
    numbers: Query<Entity, With<repame_fx::DamageNumber>>,
    particles: Query<Entity, With<repame_fx::Particle>>,
    mut mask: Option<ResMut<FloorMask>>,
) {
    for e in &q {
        commands.entity(e).despawn();
    }
    for e in &numbers {
        commands.entity(e).despawn();
    }
    for e in &particles {
        commands.entity(e).despawn();
    }

    if let Some(m) = mask.as_mut() {
        **m = FloorMask::default();
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------
