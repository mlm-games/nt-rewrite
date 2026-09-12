//! Secret-area trigger state and detectors. `SecretTriggers` is a
//! verbatim port (no engine types); the observer/detect systems below
//! mirror `nt-recreated-bevy/src/game/secret_areas.rs` with positions
//! as [`Pos`] and the boss check via `enemy_def(kind).boss`.
//!
//! Already-ported elsewhere (not duplicated here):
//! `is_secret_area` lives in [`crate::worldgen`], and
//! `target_for_secret_area` / `apply_secret_transition` live in
//! `progression.rs` (private floor-advance helpers).

use bevy_ecs::prelude::*;
use repame_sim::SimTime;

use crate::comps_a::{Inventory, Player, RaceState, Run, Toast};
use crate::comps_b::{BossBrain, Enemy, Pickup, PickupKind};
use crate::data::{AreaId, RaceId, SecretTarget, WeaponId};
use crate::enemy_data::enemy_def;

/// Tracks secret eligibility across a floor run.
#[derive(Resource, Clone, Debug)]
pub struct SecretTriggers {
    queued: Option<SecretTarget>,
    pub last_secret: Option<SecretTarget>,
    pub oasis_eligible: bool,
    pub damage_taken_this_floor: bool,

    pub oasis_chests_ready: bool,

    pub oasis_bandit_timer: f32,
    pub oasis_bandit_alive: bool,

    pub oasis_floor_chests_initial: u32,
    pub oasis_floor_enemies_initial: u32,
    pub oasis_snapshot_done: bool,

    pub vaults_entered: u8,
}

impl Default for SecretTriggers {
    fn default() -> Self {
        Self {
            queued: None,
            last_secret: None,
            oasis_eligible: true,
            damage_taken_this_floor: false,
            oasis_chests_ready: false,
            oasis_bandit_timer: 0.0,
            oasis_bandit_alive: false,
            oasis_floor_chests_initial: 0,
            oasis_floor_enemies_initial: 1,
            oasis_snapshot_done: false,
            vaults_entered: 0,
        }
    }
}

impl SecretTriggers {
    pub fn queue(&mut self, target: SecretTarget) {
        if matches!(target, SecretTarget::Vault | SecretTarget::CrownVault)
            && self.vaults_entered >= 3
        {
            return;
        }

        let replace = match (self.queued, target) {
            (None, _) => true,
            (Some(SecretTarget::Oasis), _) => true,
            (Some(SecretTarget::PizzaSewers), SecretTarget::Vault | SecretTarget::CrownVault) => {
                true
            }
            (Some(SecretTarget::YvMansion), SecretTarget::Vault | SecretTarget::CrownVault) => true,
            (Some(SecretTarget::CursedCaves), SecretTarget::Vault | SecretTarget::CrownVault) => {
                true
            }
            (Some(SecretTarget::Jungle), SecretTarget::Vault | SecretTarget::CrownVault) => true,
            _ => false,
        };

        if replace {
            self.queued = Some(target);
        }
    }

    pub fn take_queued(&mut self) -> Option<SecretTarget> {
        self.queued.take()
    }

    pub fn queued(&self) -> Option<SecretTarget> {
        self.queued
    }

    pub fn reset_floor_flags(&mut self) {
        self.oasis_eligible = true;
        self.damage_taken_this_floor = false;
        self.oasis_chests_ready = false;
        self.oasis_bandit_timer = 0.0;
        self.oasis_bandit_alive = false;
        self.oasis_snapshot_done = false;
        self.oasis_floor_chests_initial = 0;
        self.oasis_floor_enemies_initial = 1;
    }

    pub fn mark_damage_taken(&mut self) {
        self.damage_taken_this_floor = true;
        self.oasis_eligible = false;
    }
}

/// Snapshot the Desert floor's chest/enemy counts once per floor (bevy
/// `observe_oasis_floor_start` parity). The per-floor reset rides on
/// `reset_floor_flags` via the floor-advance path, same as bevy.
pub fn observe_oasis_floor_start(
    run: Res<Run>,
    mut triggers: ResMut<SecretTriggers>,
    pickups_q: Query<&Pickup>,
    enemies_q: Query<&Enemy, Without<BossBrain>>,
) {
    if triggers.oasis_snapshot_done || !triggers.oasis_eligible {
        return;
    }
    if run.area != AreaId::Desert || run.floor_in_area > 3 {
        return;
    }
    triggers.oasis_floor_chests_initial = pickups_q
        .iter()
        .filter(|p| matches!(p.kind, PickupKind::Chest(_)))
        .count() as u32;
    triggers.oasis_floor_enemies_initial = (enemies_q
        .iter()
        .filter(|e| !enemy_def(e.kind).boss)
        .count() as u32)
        .max(1);
    triggers.oasis_snapshot_done = true;
}

/// Flag the floor as oasis-ready once every chest is opened while
/// (nearly) nothing was killed (bevy `detect_oasis_eligibility`
/// parity: <= 2% kills, <= 10% on 1-3).
pub fn detect_oasis_eligibility(
    run: Res<Run>,
    mut triggers: ResMut<SecretTriggers>,
    pickups_q: Query<&Pickup>,
    enemies_q: Query<&Enemy, Without<BossBrain>>,
) {
    if triggers.oasis_chests_ready
        || run.area != AreaId::Desert
        || run.floor_in_area > 3
        || !triggers.oasis_eligible
        || triggers.damage_taken_this_floor
    {
        return;
    }

    let chests_left = pickups_q
        .iter()
        .filter(|p| matches!(p.kind, PickupKind::Chest(_)))
        .count();
    if chests_left > 0 || !triggers.oasis_snapshot_done {
        return;
    }

    let living_trash = enemies_q
        .iter()
        .filter(|e| !enemy_def(e.kind).boss)
        .count() as u32;
    let killed = triggers
        .oasis_floor_enemies_initial
        .saturating_sub(living_trash);
    let kill_frac = killed as f32 / triggers.oasis_floor_enemies_initial.max(1) as f32;
    let max_kill = if run.floor_in_area == 3 { 0.10 } else { 0.02 };
    if kill_frac <= max_kill {
        triggers.oasis_chests_ready = true;
    }
}

/// Arm the 10 s bandit window once Big Bandit spawns on a ready floor;
/// killing him in time queues the Oasis (bevy
/// `tick_oasis_bandit_window` parity, `SimTime` for `Time<Fixed>`).
pub fn tick_oasis_bandit_window(
    time: Res<SimTime>,
    mut triggers: ResMut<SecretTriggers>,
    enemies_q: Query<&Enemy>,
) {
    if !triggers.oasis_chests_ready {
        return;
    }
    if triggers.damage_taken_this_floor || !triggers.oasis_eligible {
        triggers.oasis_chests_ready = false;
        return;
    }

    let bandit_alive = enemies_q.iter().any(|e| {
        matches!(
            e.kind,
            crate::data::EnemyKind::BigBandit | crate::data::EnemyKind::BigBanditLoop
        )
    });

    if bandit_alive && !triggers.oasis_bandit_alive {
        triggers.oasis_bandit_alive = true;
        triggers.oasis_bandit_timer = 10.0;
    }

    if !triggers.oasis_bandit_alive {
        return;
    }

    triggers.oasis_bandit_timer -= time.delta_secs;
    if !bandit_alive && triggers.oasis_bandit_timer > 0.0 {
        triggers.queue(SecretTarget::Oasis);
        triggers.oasis_chests_ready = false;
        triggers.oasis_bandit_alive = false;
    } else if triggers.oasis_bandit_timer <= 0.0 {
        triggers.oasis_chests_ready = false;
        triggers.oasis_bandit_alive = false;
    }
}

/// Carrying a cursed weapon through Crystal Caves queues the Cursed
/// Caves (bevy `detect_cursed_caves` parity).
pub fn detect_cursed_caves(
    run: Res<Run>,
    mut triggers: ResMut<SecretTriggers>,
    player_q: Query<&Inventory, With<Player>>,
) {
    if run.area != AreaId::CrystalCaves {
        return;
    }

    let Ok(inv) = player_q.single() else {
        return;
    };

    if inv.weapons.iter().any(|&w| is_cursed_weapon(w)) {
        triggers.queue(SecretTarget::CursedCaves);
    }
}

fn is_cursed_weapon(w: WeaponId) -> bool {
    if w.0 == 0 {
        return false;
    }
    if let Some(data) = crate::weapons_data::WEAPONS.get(w.0 as usize) {
        return data.wep_gold || data.wep_rads >= 12 || (90..=127).contains(&w.0);
    }
    (90..=127).contains(&w.0)
}

/// Queue the IDPD HQ for Rogues past loop 1 (Labs/Palace), plus the
/// small deterministic Labs loop-2+ roll (bevy `detect_hq` parity).
pub fn detect_hq(
    run: Res<Run>,
    mut triggers: ResMut<SecretTriggers>,
    player_q: Query<&RaceState, With<Player>>,
) {
    let is_rogue = player_q
        .single()
        .map(|r| r.race == RaceId::Rogue)
        .unwrap_or(false);

    if is_rogue && run.loop_count >= 1 && matches!(run.area, AreaId::Labs | AreaId::Palace) {
        triggers.queue(SecretTarget::Hq);
        return;
    }

    if run.area == AreaId::Labs && run.loop_count >= 2 {
        let roll =
            ((run.gen_seed ^ run.floor as u64).wrapping_mul(6364136223846793005) >> 56) as u8;
        if roll < 12 {
            triggers.queue(SecretTarget::Hq);
        }
    }
}

/// Announce a queued secret route while the toast is idle (bevy
/// `secret_debug_toast` parity; text via the existing `Toast`
/// resource).
pub fn secret_debug_toast(triggers: Res<SecretTriggers>, mut toast: ResMut<Toast>) {
    if let Some(target) = triggers.queued()
        && toast.timer.is_finished()
    {
        toast.show(&format!("SECRET ROUTE: {}", target.name()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comps_b::ChestKind;
    use crate::data::EnemyKind;

    fn desert_run() -> Run {
        Run {
            floor: 2,
            world: 1,
            area: AreaId::Desert,
            loop_count: 0,
            floor_in_area: 2,
            gen_seed: 0x1234,
            portal_open: false,
            game_over: false,
            total_kills: 0,
            blackswords: 0,
            tottimer: 0,
            popolevel: 0,
            nochest: 0,
            noradch: 0,
            same_weapons_for: 0,
            horror: false,
            shots_fired: 0,
            weapons_picked: 0,
            won: false,
            hardmode: false,
        }
    }

    fn trash(kind: EnemyKind) -> Enemy {
        Enemy {
            kind,
            score: 1,
            touch_damage: 1,
            rad_drop: 1,
            drop_chance: 0,
            weapon_chance: 0,
        }
    }

    fn chest() -> Pickup {
        Pickup {
            kind: PickupKind::Chest(ChestKind::Weapon),
        }
    }

    #[test]
    fn vault_queue_rules_and_damage_flags() {
        let mut s = SecretTriggers::default();
        assert!(s.oasis_eligible);
        s.queue(SecretTarget::Oasis);
        assert_eq!(s.take_queued(), Some(SecretTarget::Oasis));
        assert_eq!(s.take_queued(), None);
        s.mark_damage_taken();
        assert!(s.damage_taken_this_floor);
        assert!(!s.oasis_eligible);
        s.reset_floor_flags();
        assert!(s.oasis_eligible);
        assert!(!s.damage_taken_this_floor);
        // Vault cap: 3 entries max.
        for _ in 0..5 {
            s.queue(SecretTarget::Vault);
            s.vaults_entered += 1;
        }
        assert_eq!(s.take_queued(), Some(SecretTarget::Vault));
    }

    #[test]
    fn oasis_snapshot_then_eligibility_when_chests_cleared() {
        let mut world = World::new();
        world.insert_resource(desert_run());
        world.insert_resource(SecretTriggers::default());
        world.spawn(chest());
        world.spawn(chest());
        for _ in 0..10 {
            world.spawn(trash(EnemyKind::Bandit));
        }

        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems((observe_oasis_floor_start, detect_oasis_eligibility).chain());
        sched.run(&mut world);

        let triggers = world.resource::<SecretTriggers>();
        assert!(triggers.oasis_snapshot_done);
        assert_eq!(triggers.oasis_floor_chests_initial, 2);
        assert_eq!(triggers.oasis_floor_enemies_initial, 10);
        assert!(
            !triggers.oasis_chests_ready,
            "chests still on the floor: not ready"
        );

        // Open every chest without killing: eligibility flips.
        let chests: Vec<Entity> = world.query::<Entity>().iter(&world).collect();
        for e in chests {
            if world.get::<Pickup>(e).is_some() {
                world.despawn(e);
            }
        }
        sched.run(&mut world);
        assert!(world.resource::<SecretTriggers>().oasis_chests_ready);
    }

    #[test]
    fn oasis_kill_budget_is_floor_dependent() {
        // 20 trash, one kill = 5%: too many for 1-2 (2%), fine for 1-3 (10%).
        for floor_in_area in [2u32, 3u32] {
            let mut world = World::new();
            let mut run = desert_run();
            run.floor_in_area = floor_in_area;
            world.insert_resource(run);
            let mut triggers = SecretTriggers::default();
            triggers.oasis_snapshot_done = true;
            triggers.oasis_floor_enemies_initial = 20;
            world.insert_resource(triggers);
            for _ in 0..19 {
                world.spawn(trash(EnemyKind::Bandit));
            }
            let mut sched = bevy_ecs::schedule::Schedule::default();
            sched.add_systems(detect_oasis_eligibility);
            sched.run(&mut world);
            assert_eq!(
                world.resource::<SecretTriggers>().oasis_chests_ready,
                floor_in_area == 3,
                "floor_in_area {floor_in_area}"
            );
        }
    }

    #[test]
    fn oasis_damage_forfeits_eligibility() {
        let mut world = World::new();
        world.insert_resource(desert_run());
        let mut triggers = SecretTriggers::default();
        triggers.oasis_snapshot_done = true;
        triggers.oasis_chests_ready = true;
        triggers.mark_damage_taken();
        world.insert_resource(triggers);
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_oasis_bandit_window);
        let mut time = repame_sim::SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        sched.run(&mut world);
        assert!(!world.resource::<SecretTriggers>().oasis_chests_ready);
        assert_eq!(world.resource::<SecretTriggers>().queued(), None);
    }

    #[test]
    fn oasis_bandit_kill_queues_oasis_within_window() {
        let mut world = World::new();
        world.insert_resource(desert_run());
        let mut triggers = SecretTriggers::default();
        triggers.oasis_chests_ready = true;
        world.insert_resource(triggers);
        let mut time = repame_sim::SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        let bandit = world.spawn(trash(EnemyKind::BigBandit)).id();

        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_oasis_bandit_window);
        sched.run(&mut world);
        {
            let triggers = world.resource::<SecretTriggers>();
            assert!(triggers.oasis_bandit_alive);
            assert!((triggers.oasis_bandit_timer - (10.0 - 1.0 / 30.0)).abs() < 1e-5);
        }

        // Kill the bandit inside the window: Oasis queues.
        world.despawn(bandit);
        sched.run(&mut world);
        let triggers = world.resource::<SecretTriggers>();
        assert_eq!(triggers.queued(), Some(SecretTarget::Oasis));
        assert!(!triggers.oasis_chests_ready);
        assert!(!triggers.oasis_bandit_alive);
    }

    #[test]
    fn oasis_bandit_window_expiry_clears_without_queue() {
        let mut world = World::new();
        world.insert_resource(desert_run());
        let mut triggers = SecretTriggers::default();
        triggers.oasis_chests_ready = true;
        triggers.oasis_bandit_alive = true;
        triggers.oasis_bandit_timer = 0.01;
        world.insert_resource(triggers);
        world.spawn(trash(EnemyKind::BigBandit));
        let mut time = repame_sim::SimTime::default();
        time.delta_secs = 1.0;
        world.insert_resource(time);
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_oasis_bandit_window);
        sched.run(&mut world);
        let triggers = world.resource::<SecretTriggers>();
        assert_eq!(triggers.queued(), None);
        assert!(!triggers.oasis_chests_ready);
    }

    fn inventory_with(weapons: [WeaponId; 3]) -> Inventory {
        Inventory {
            weapons,
            cursed: [false, false, false],
            swapanim: 0.0,
            shine: 0.0,
            wepflip: 1.0,
            bwepflip: 1.0,
            weapon_slots: 3,
            current: 0,
            ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
        }
    }

    #[test]
    fn cursed_weapon_queues_cursed_caves() {
        let mut world = World::new();
        let mut run = desert_run();
        run.area = AreaId::CrystalCaves;
        world.insert_resource(run);
        world.insert_resource(SecretTriggers::default());
        world.spawn((
            Player::default(),
            inventory_with([WeaponId(90), WeaponId::REVOLVER, WeaponId::NONE]),
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(detect_cursed_caves);
        sched.run(&mut world);
        assert_eq!(
            world.resource::<SecretTriggers>().queued(),
            Some(SecretTarget::CursedCaves)
        );
    }

    #[test]
    fn clean_weapons_queue_nothing() {
        let mut world = World::new();
        let mut run = desert_run();
        run.area = AreaId::CrystalCaves;
        world.insert_resource(run);
        world.insert_resource(SecretTriggers::default());
        world.spawn((
            Player::default(),
            inventory_with([WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE]),
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(detect_cursed_caves);
        sched.run(&mut world);
        assert_eq!(world.resource::<SecretTriggers>().queued(), None);
    }

    #[test]
    fn cursed_weapon_predicate_shape() {
        assert!(!is_cursed_weapon(WeaponId::NONE));
        assert!(!is_cursed_weapon(WeaponId::REVOLVER));
        assert!(is_cursed_weapon(WeaponId(90)));
        assert!(is_cursed_weapon(WeaponId(127)));
        assert!(!is_cursed_weapon(WeaponId(128)));
    }

    #[test]
    fn rogue_past_loop_one_queues_hq() {
        let mut world = World::new();
        let mut run = desert_run();
        run.area = AreaId::Labs;
        run.loop_count = 1;
        world.insert_resource(run);
        world.insert_resource(SecretTriggers::default());
        world.spawn((
            Player::default(),
            RaceState {
                race: RaceId::Rogue,
                skin: crate::data::SkinLetter::A,
            },
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(detect_hq);
        sched.run(&mut world);
        assert_eq!(
            world.resource::<SecretTriggers>().queued(),
            Some(SecretTarget::Hq)
        );
    }

    #[test]
    fn non_rogue_loop_zero_queues_no_hq() {
        let mut world = World::new();
        let mut run = desert_run();
        run.area = AreaId::Labs;
        run.loop_count = 0;
        world.insert_resource(run);
        world.insert_resource(SecretTriggers::default());
        world.spawn((
            Player::default(),
            RaceState {
                race: RaceId::Fish,
                skin: crate::data::SkinLetter::A,
            },
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(detect_hq);
        sched.run(&mut world);
        assert_eq!(world.resource::<SecretTriggers>().queued(), None);
    }

    #[test]
    fn secret_debug_toast_announces_queued_route_when_idle() {
        let mut world = World::new();
        let mut triggers = SecretTriggers::default();
        triggers.queue(SecretTarget::Oasis);
        world.insert_resource(triggers);
        world.insert_resource(Toast::default());
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(secret_debug_toast);
        sched.run(&mut world);
        assert_eq!(world.resource::<Toast>().text, "SECRET ROUTE: OASIS");

        // Busy toast: no overwrite.
        world.resource_mut::<Toast>().show("REST");
        sched.run(&mut world);
        assert_eq!(world.resource::<Toast>().text, "REST");
    }
}
