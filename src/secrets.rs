//! Secret-area trigger state and detectors. `SecretTriggers` is a verbatim port
//! (no engine types); the observer/detect systems below mirror the GML
//! secret-area scripts with positions as [`Pos`] and the boss check via
//! `enemy_def(kind).boss`.
//! Already-ported elsewhere (not duplicated here): `is_secret_area` lives in
//! [`crate::worldgen`], and `target_for_secret_area` / `apply_secret_transition`
//! live in `progression.rs` (private floor-advance helpers).

use bevy_ecs::prelude::*;
use repame_sim::SimTime;

use crate::comps_a::{Inventory, Player, RaceState, Run, Toast};
use crate::comps_b::{BossBrain, Enemy, Pickup, PickupKind};
use crate::data::{AreaId, RaceId, SecretTarget};
use crate::enemy_data::enemy_def;

/// Tracks secret eligibility across a floor run.
#[derive(Resource, Clone, Debug)]
pub struct SecretTriggers {
    queued: Option<SecretTarget>,
    pub last_secret: Option<SecretTarget>,
    /// GML `CanOasis` (`WantBoss/Step_0:14-24`) needs a chest opened and
    /// no kills; taking damage is not part of it, so `oasis_eligible`
    /// only gates the per-floor snapshot.
    pub oasis_eligible: bool,
    /// Recorded for the HUD/debug pass only - GML has no damage
    /// condition on any secret route.
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
    triggers.oasis_floor_enemies_initial =
        (enemies_q.iter().filter(|e| !enemy_def(e.kind).boss).count() as u32).max(1);
    triggers.oasis_snapshot_done = true;
}

/// Flag the floor as oasis-ready once every chest is opened while
/// (nearly) nothing was killed. GML `WantBoss/Step_0:14-24` (the
/// `CanOasis` marker: at least one `ChestOpen`, no unopened `chestprop`
/// or rad/rogue chest, and fewer than 2% of the enemies dead - 10% on
/// the area's last subarea).
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
    if triggers.oasis_floor_chests_initial == 0 {
        return;
    }

    let living_trash = enemies_q.iter().filter(|e| !enemy_def(e.kind).boss).count() as u32;
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
/// killing him in time queues the Oasis. GML: the window is
/// `CanOasis/Alarm_0` (300 ticks) and the trigger is
/// `BanditBoss/Destroy_0:13-21` - no damage condition.
pub fn tick_oasis_bandit_window(
    time: Res<SimTime>,
    mut triggers: ResMut<SecretTriggers>,
    enemies_q: Query<&Enemy>,
) {
    if !triggers.oasis_chests_ready {
        return;
    }
    if !triggers.oasis_eligible {
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
/// Caves (bevy `detect_cursed_caves` parity). GML
/// `scrPlayerCountCursed` (`scripts/scrPlayerUncurse/scrPlayerUncurse.gml:3-15`)
/// counts cursed weapon *slots*, and the trigger itself is
/// `GameCont/Other_5:110-112` (`area == area_caves` plus a curse).
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

    if inv.cursed.iter().any(|c| *c) {
        triggers.queue(SecretTarget::CursedCaves);
    }
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
