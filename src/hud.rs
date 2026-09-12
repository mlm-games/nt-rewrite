//! Headless HUD state. Pure-state port of the bevy reference
//! `game/hud.rs` (`sync_hud`, `reset_hud_flags`): the same sim values are
//! derived (hp/ammo/rads/weapon names/crown/mutations/kills/level
//! progress/boss bar/IDPD warning/toasts) but instead of writing through a
//! `UiBridge` resource into drawn widgets, they are collected into a plain
//! [`HudState`] that the repose-canvas UI polls after each tick. No drawing,
//! no bevy engine imports.

use bevy_ecs::prelude::*;

use crate::comps_a::{
    Health, Inventory, PendingMutation, PendingUltra, Player, Run, Score, SelectedCharacter, Toast,
};
use crate::comps_b::{BossBrain, Enemy, IdpdRaidState, Revive};
use crate::data::{AbilityKind, CrownKind};
use crate::enemy_data::enemy_def;
use crate::savedata_part::{SaveData, character_def};
use crate::spatial::Pos;
use crate::weapon_runtime::{weapon_id_name, weapon_meta};
use crate::worldgen::floor_in_world;

/// Pollable HUD snapshot. Field-for-field coverage of what bevy `sync_hud`
/// wrote into `SharedUi` for the in-game HUD (menu/settings/loadout fields
/// and the transient `mutation_selected` cursor stay UI-phase owned).
#[derive(Clone, Debug, PartialEq, Resource)]
pub struct HudBars {
    /// GML `lsthealth` ghost-fill persistence for the health bar
    /// (float like GML; decays per `render::hud_sprites`).
    pub last_hp: f32,
}

/// One coop downed marker for the fainted-bar draw (GML `TopCont/Draw_0`
/// `Revive` block): world pos plus the two alarm values that drive the
/// pulse (`alarm[4]`) vs bleed (`alarm[5]`) bars.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaintedBar {
    pub x: f32,
    pub y: f32,
    /// GML `alarm[4]` (300-step grace, red-black pulse bar).
    pub alarm4: f32,
    /// GML `alarm[5]` (30-step hurt pulse, red bar).
    pub alarm5: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HudState {
    pub game_over: bool,
    pub hp: i32,
    pub max_hp: i32,
    pub level: u32,
    pub rads: u32,
    pub max_rads: u32,
    pub weapons: Vec<String>,
    pub current_weapon: usize,
    /// Weapon ids parallel to `weapons` (bar sprite lookup).
    pub weapon_ids: Vec<crate::data::WeaponId>,
    /// Per-slot curse parallel to `weapons` (GML `c_curse` fog).
    pub weapon_cursed: Vec<bool>,
    pub ammo: [i32; 6],
    /// Per-visible-slot ammo for the weapon icons (-1 = melee / no slot,
    /// bevy `weapon_ammo` law).
    pub weapon_ammo: [i32; 2],
    pub ability: String,
    pub ability_ready: bool,
    pub crown: String,
    pub character: String,
    pub selected_character: usize,
    pub floor: u32,
    pub world: u32,
    pub floor_in_world: u32,
    pub loop_count: u32,
    pub score: u32,
    pub high_score: u32,
    pub best_floor: u32,
    /// Lifetime kills (save accumulation, bevy `SharedUi.total_kills`).
    pub total_kills: u32,
    /// This run's kills (`Run.total_kills`; bevy never surfaced it in
    /// `SharedUi`, but the task's kill-count parity needs it headless).
    pub kills: u32,
    pub boss_hp: u32,
    pub boss_max: u32,
    pub boss_name: String,
    /// An IDPD raid wave is queued (warning toast pending/flying). The bevy
    /// build surfaced this only as the "IDPD INCOMING" toast; headless gets
    /// the explicit bit so the canvas can draw the warning badge.
    pub idpd_warning: bool,
    pub toast: String,
    pub toast_timer: f32,
    /// GML run clock `M:SS.CC` (from `Run.tottimer` steps).
    pub timer_string: String,
    /// GML map name `A-S` + ` L#` on loops.
    pub area_string: String,
    pub mutation_choices: Vec<String>,
    pub mutation_choice_ids: Vec<u8>,
    pub death_mutation_ids: Vec<u8>,
    /// Live coop downed markers (GML `TopCont/Draw_0` fainted bars;
    /// empty in single-player — the port never spawns [`Revive`]).
    pub fainted_bars: Vec<FaintedBar>,
}

impl Default for HudState {
    /// Mirrors bevy `SharedUi::default` for the covered subset.
    fn default() -> Self {
        Self {
            game_over: false,
            hp: 10,
            max_hp: 10,
            level: 1,
            rads: 0,
            max_rads: 60,
            weapons: vec!["Revolver".to_string(), "Shotgun".to_string()],
            current_weapon: 0,
            weapon_ids: vec![crate::data::WeaponId::REVOLVER, crate::data::WeaponId::SHOTGUN],
            weapon_cursed: vec![false, false],
            ammo: [0; 6],
            weapon_ammo: [0, 0],
            ability: "Flip".to_string(),
            ability_ready: true,
            crown: "NONE".to_string(),
            character: "Fish".to_string(),
            selected_character: 1,
            floor: 1,
            world: 1,
            floor_in_world: 1,
            loop_count: 0,
            score: 0,
            high_score: 0,
            best_floor: 0,
            total_kills: 0,
            kills: 0,
            boss_hp: 0,
            boss_max: 0,
            boss_name: String::new(),
            idpd_warning: false,
            toast: String::new(),
            toast_timer: 0.0,
            timer_string: String::new(),
            area_string: String::new(),
            mutation_choices: Vec::new(),
            mutation_choice_ids: Vec::new(),
            death_mutation_ids: Vec::new(),
            fainted_bars: Vec::new(),
        }
    }
}

/// Cleared flags (bevy `reset_hud_flags` parity: game over off,
/// mutation picks/toast/boss/loop cleared IN PLACE — hp, weapons,
/// ammo, level and the rest are preserved).
pub fn reset_hud_state(hud: &mut HudState) {
    hud.game_over = false;
    hud.mutation_choices.clear();
    hud.mutation_choice_ids.clear();
    hud.death_mutation_ids.clear();
    hud.toast.clear();
    hud.toast_timer = 0.0;
    hud.boss_hp = 0;
    hud.boss_max = 0;
    hud.boss_name.clear();
    hud.fainted_bars.clear();
    hud.loop_count = 0;
}

/// GML `timer_string` verbatim: `M:SS.CC` from `tottimer` steps at
/// 30 steps/s (`round(timer / 30 * 100)` centiseconds).
pub fn run_timer_string(tottimer: u32) -> String {
    let minutes = tottimer / 1800;
    let seconds = (tottimer / 30) % 60;
    let centis = ((tottimer % 30) * 100 + 15) / 30;
    format!("{minutes}:{seconds:02}.{centis:02}")
}

/// GML `scrAreaGetMapName` verbatim (unlocalized strings): HQ shows
/// `HQ{sub}`, the crib shows `$$$`, secret areas `N-?`, the vault
/// `???`, otherwise `A-S`; plus ` L#` on loops.
pub fn run_area_string(run: &Run) -> String {
    let area = crate::worldgen::gml_area_from_run(run);
    let base = if area == 106 {
        format!("HQ{}", run.floor_in_area)
    } else if area == 107 {
        "$$$".to_string()
    } else if area > 100 {
        format!("{}-?", area - 100)
    } else if area == 100 {
        "???".to_string()
    } else {
        format!("{area}-{}", run.floor_in_area)
    };
    if run.loop_count != 0 {
        format!("{} L{}", base, run.loop_count)
    } else {
        base
    }
}

/// Derive the HUD snapshot from sim state (bevy `sync_hud` state
/// computation, `UiBridge` writes removed). Reads `&World` so the UI can
/// poll it without a mutable borrow.
pub fn sync_hud_state(world: &World) -> HudState {
    let mut hud = HudState::default();

    // Coop downed markers for the fainted-bar draw (GML `TopCont/Draw_0`
    // `with Revive` block: raw pos + alarms; the renderer clamps to the
    // view, exactly like GML's `clamp(x, view_xview + 30, ...)`).
    // Filled before the run gate: bars draw whenever a marker exists.
    for entity_ref in world.iter_entities() {
        let (Some(pos), Some(revive)) = (entity_ref.get::<Pos>(), entity_ref.get::<Revive>())
        else {
            continue;
        };
        hud.fainted_bars.push(FaintedBar {
            x: pos.0.x,
            y: pos.0.y,
            alarm4: revive.alarm4,
            alarm5: revive.alarm5,
        });
    }

    let Some(run) = world.get_resource::<Run>() else {
        // Bevy early-return: clear run-scoped fields, keep the rest default.
        hud.game_over = false;
        hud.mutation_choices.clear();
        hud.mutation_choice_ids.clear();
        hud.death_mutation_ids.clear();
        hud.boss_hp = 0;
        hud.boss_max = 0;
        hud.boss_name.clear();
        hud.loop_count = 0;
        hud.toast.clear();
        hud.toast_timer = 0.0;
        return hud;
    };

    hud.game_over = run.game_over;

    if let Some(toast) = world.get_resource::<Toast>() {
        hud.toast = toast.text.clone();
        hud.toast_timer = if toast.timer.duration() <= 0.0 {
            0.0
        } else {
            1.0 - toast.timer.fraction()
        };
    }

    // Boss bar: highest-max-HP boss (bevy `max_by_key(health.max)`).
    hud.boss_hp = 0;
    hud.boss_max = 0;
    hud.boss_name.clear();
    let mut best: Option<(i32, u32, String)> = None;
    for entity_ref in world.iter_entities() {
        let (Some(enemy), Some(health)) =
            (entity_ref.get::<Enemy>(), entity_ref.get::<Health>())
        else {
            continue;
        };
        if !entity_ref.contains::<BossBrain>() {
            continue;
        }
        if best.as_ref().map(|(m, _, _)| health.max >= *m).unwrap_or(true) {
            best = Some((
                health.max,
                health.hp.max(0) as u32,
                enemy_def(enemy.kind).name.to_string(),
            ));
        }
    }
    if let Some((max, hp, name)) = best {
        hud.boss_hp = hp;
        hud.boss_max = max.max(1) as u32;
        hud.boss_name = name;
    }

    if let Some(character) = world.get_resource::<SelectedCharacter>() {
        hud.character = character_def(character.0).name.to_string();
        hud.selected_character = character.0 as usize;
    }

    for entity_ref in world.iter_entities() {
        let (Some(player), Some(health), Some(inv)) = (
            entity_ref.get::<Player>(),
            entity_ref.get::<Health>(),
            entity_ref.get::<Inventory>(),
        ) else {
            continue;
        };
        // Bevy used `single()`: first player entity wins; headless sims
        // spawn exactly one.
        hud.hp = health.hp.max(0);
        hud.max_hp = health.max;
        hud.level = player.level;
        hud.rads = player.rads;
        hud.max_rads = player.next_level_rads;
        hud.weapons = (0..inv.weapon_slots)
            .map(|i| weapon_id_name(inv.weapons[i]).to_string())
            .collect();
        hud.current_weapon = inv.current;
        hud.weapon_ids = (0..inv.weapon_slots).map(|i| inv.weapons[i]).collect();
        hud.weapon_cursed = (0..inv.weapon_slots).map(|i| inv.cursed[i]).collect();
        hud.ammo = inv.ammo;

        let t1 = weapon_meta(inv.weapons[0]).wep_type as usize;
        let t2 = if inv.weapon_slots > 1 {
            weapon_meta(inv.weapons[1]).wep_type as usize
        } else {
            0
        };
        hud.weapon_ammo = [inv.ammo[t1.min(5)], inv.ammo[t2.min(5)]];
        if t1 == 0 {
            hud.weapon_ammo[0] = -1;
        }
        if t2 == 0 || inv.weapon_slots <= 1 {
            hud.weapon_ammo[1] = -1;
        }
        hud.ability = ability_name(player.ability).to_string();
        hud.ability_ready = player.ability_cooldown.is_finished();
        hud.crown = crown_short_name(player.crown).to_string();
        break;
    }

    hud.floor = run.floor;
    hud.world = run.world;
    hud.floor_in_world = floor_in_world(run.floor);
    hud.loop_count = run.loop_count;
    hud.kills = run.total_kills;

    if let Some(score) = world.get_resource::<Score>() {
        hud.score = score.0;
    }
    if let Some(save) = world.get_resource::<SaveData>() {
        hud.high_score = save.high_score;
        hud.best_floor = save.best_floor;
        hud.total_kills = save.total_kills;
    }

    // GML clock + map name (`scrDrawMiscHUD` bottom-right rows).
    hud.timer_string = run_timer_string(run.tottimer);
    hud.area_string = run_area_string(run);

    if let Some(raid) = world.get_resource::<IdpdRaidState>() {
        hud.idpd_warning = raid.pending_wave.is_some();
    }

    let (choices, ids) = if let Some(ultra) = world.get_resource::<PendingUltra>() {
        (
            ultra
                .choices
                .iter()
                .map(|u| {
                    let (name, desc) = crate::progression::ultra_mutation_name(*u);
                    format!("ULTRA: {name} - {desc}")
                })
                .collect::<Vec<_>>(),
            ultra
                .choices
                .iter()
                .map(|u| ultra_skill_index(*u))
                .collect(),
        )
    } else if let Some(p) = world.get_resource::<PendingMutation>() {
        (
            p.choices
                .iter()
                .map(|m| {
                    let (name, desc) = crate::progression::mutation_name(*m);
                    format!("{name} - {desc}")
                })
                .collect::<Vec<_>>(),
            p.choices.iter().map(|m| mutation_skill_index(*m)).collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    hud.mutation_choices = choices;
    hud.mutation_choice_ids = ids;
    // Note: bevy also maintained the `mutation_selected` cursor here
    // (reset when the choice list changed length / emptied). Cursor state is
    // UI-phase owned; the canvas resets its own cursor from
    // `mutation_choices.len()`.
    if run.game_over {
        for entity_ref in world.iter_entities() {
            if let Some(player) = entity_ref.get::<Player>() {
                hud.death_mutation_ids =
                    player.mutations.iter().map(|m| mutation_skill_index(*m)).collect();
                break;
            }
        }
    } else {
        hud.death_mutation_ids.clear();
    }

    hud
}

/// Ability display name (bevy `content.rs::ability_name` verbatim).
pub fn ability_name(kind: AbilityKind) -> &'static str {
    match kind {
        AbilityKind::Flip => "Flip",
        AbilityKind::Shield => "Shield",
        AbilityKind::Telekinesis => "Telekinesis",
        AbilityKind::Detonate => "Detonate",
        AbilityKind::Snare => "Snare",
        AbilityKind::PopPop => "Pop Pop",
        AbilityKind::GetLoaded => "Get Loaded",
        AbilityKind::EatWeapon => "Eat Weapon",
        AbilityKind::Throw => "Throw",
        AbilityKind::SpawnAlly => "Rebel Yell",
        AbilityKind::HorrorBeam => "Irradiate",
        AbilityKind::PortalStrike => "Portal Strike",
        AbilityKind::RocketBarrage => "Barrage",
        AbilityKind::BloodGamble => "Blood Gamble",
        AbilityKind::ToxicPuke => "Puke",
        AbilityKind::CuzSwap => "Extra Slot",
    }
}

/// Short crown label for the HUD icon row (bevy `CrownKind::short_name`).
pub fn crown_short_name(kind: CrownKind) -> &'static str {
    match kind {
        CrownKind::None => "NONE",
        CrownKind::Death => "DEATH",
        CrownKind::Life => "LIFE",
        CrownKind::Haste => "HASTE",
        CrownKind::Guns => "GUNS",
        CrownKind::Hatred => "HATRED",
        CrownKind::Blood => "BLOOD",
        CrownKind::Destiny => "DESTINY",
        CrownKind::Love => "LOVE",
        CrownKind::Risk => "RISK",
        CrownKind::Curses => "CURSES",
        CrownKind::Luck => "LUCK",
        CrownKind::Protection => "PROTECTION",
    }
}

/// Skill-icon id for a mutation (bevy `mutation_skill_index` verbatim).
pub fn mutation_skill_index(id: crate::data::MutationId) -> u8 {
    use crate::data::MutationId::*;
    match id {
        RhinoSkin => 1,
        ExtraFeet => 2,
        PlutoniumHunger => 3,
        RabbitPaw => 4,
        ThroneButt => 5,
        LuckyShot => 6,
        Bloodlust => 7,
        GammaGuts => 8,
        SecondStomach => 9,
        BackMuscle => 10,
        ScarierFace => 11,
        Euphoria => 12,
        LongArms => 13,
        BoilingVeins => 14,
        ShotgunShoulders => 15,
        RecycleGland => 16,
        LaserBrain => 17,
        LastWish => 18,
        EagleEyes => 19,
        ImpactWrists => 20,
        BoltMarrow => 21,
        Stress => 22,
        TriggerFingers => 23,
        SharpTeeth => 24,
        Patience => 25,
        Hammerhead => 26,
        StrongSpirit => 27,
        OpenMind => 28,
        HeavyHeart => 29,
    }
}

/// Skill-icon id for an ultra mutation (bevy `ultra_skill_index` verbatim).
pub fn ultra_skill_index(id: crate::data::UltraMutationId) -> u8 {
    use crate::data::UltraMutationId::*;
    match id {
        FishGunWarrant => 1,
        FishConfiscate => 2,
        CrystalFortress => 3,
        CrystalJuggernaut => 4,
        EyesMonsterStyle => 5,
        EyesProjectileStyle => 6,
        MeltingBrainCapacity => 7,
        MeltingDetachment => 8,
        PlantTrapper => 9,
        PlantKiller => 10,
        VenuzBack2Bizniz => 11,
        VenuzGunGod => 12,
        SteroidsAmbidextrous => 13,
        SteroidsGetArmed => 14,
        RobotRefinedTaste => 15,
        RobotRegurgitate => 16,
        ChickenHarderToKill => 17,
        ChickenDetermination => 18,
        RebelPersonalGuard => 19,
        RebelRiot => 20,
        HorrorStalker => 21,
        HorrorAnomaly => 22,
        RogueSuperBlastArmor => 23,
        RoguePortalStrike => 24,
        BigDogHeavyArtillery => 25,
        BigDogGuardian => 26,
        SkeletonBloodArmor => 27,
        SkeletonNecromancy => 28,
        FrogToxicLord => 29,
        FrogSwampBody => 30,
        CuzHoarder => 1,
        CuzQuickSwap => 2,
        CuzEmotional => 3,
    }
}

