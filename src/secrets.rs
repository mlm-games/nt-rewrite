//! Secret-area trigger state and detectors. `SecretTriggers` is a verbatim port
//! (no engine types); the observer/detect systems below mirror the GML
//! secret-area scripts with positions as [`Pos`] and the boss check via
//! `enemy_def(kind).boss`.
//! Already-ported elsewhere (not duplicated here): `is_secret_area` lives in
//! [`crate::worldgen`], and `target_for_secret_area` / `apply_secret_transition`
//! live in `progression.rs` (private floor-advance helpers).

use bevy_ecs::prelude::*;
use repame_sim::SimTime;

use crate::time::{GTimer, TimerMode};

use crate::comps_a::{Inventory, Player, Run, Toast};
use crate::comps_b::{BossBrain, Enemy, Pickup, PickupKind};
use crate::data::{AreaId, SecretTarget};
use crate::enemy_data::enemy_def;
use crate::spatial::Pos;

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

    pub oasis_floor_chests_initial: u32,
    pub oasis_floor_enemies_initial: u32,
    pub oasis_snapshot_done: bool,

    pub vaults_entered: u8,

    /// GML `Player/Collision_CarVenusFixed.gml:10-13`: driving the crib's
    /// parked car out drops the run one subarea further back than the crib's
    /// own exit portal does.
    pub car_out_of_crib: bool,
}

impl Default for SecretTriggers {
    fn default() -> Self {
        Self {
            queued: None,
            last_secret: None,
            oasis_eligible: true,
            damage_taken_this_floor: false,
            oasis_chests_ready: false,
            oasis_floor_chests_initial: 0,
            oasis_floor_enemies_initial: 1,
            oasis_snapshot_done: false,
            vaults_entered: 0,
            car_out_of_crib: false,
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

/// GML `PizzaEntrance/Collision_Explosion.gml`: the first `Explosion` that
/// touches the sewers manhole empties the floor and opens a portal to
/// `area_pizza_sewers` - but only while no `FrogQueen` is standing, and only
/// once (`image_index = 1` gates every later blast).
pub fn tick_pizza_entrances(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut run: ResMut<Run>,
    mut gates: Query<
        (Entity, &Pos, &mut crate::comps_b::PizzaEntrance),
        (
            With<crate::comps_b::PizzaEntrance>,
            Without<crate::comps_a::Player>,
        ),
    >,
    blasts: Query<
        (&Pos, &crate::combat::Explosion),
        (
            With<crate::combat::Explosion>,
            Without<crate::comps_a::Player>,
        ),
    >,
    mut enemies: Query<
        (Entity, &crate::comps_b::Enemy, &mut crate::comps_a::Health),
        (With<crate::comps_b::Enemy>, Without<crate::comps_a::Player>),
    >,
    mut shots: Query<(Entity, &crate::comps_a::Team), With<crate::Projectile>>,
) {
    if gates.is_empty() {
        return;
    }
    if enemies
        .iter()
        .any(|(_, enemy, _)| enemy.kind == crate::data::EnemyKind::FrogQueen)
    {
        return;
    }
    let fired: Vec<(Entity, glam::Vec2)> = gates
        .iter_mut()
        .filter(|(_, pos, gate)| {
            !gate.opened
                && blasts
                    .iter()
                    .any(|(bp, b)| bp.0.distance(pos.0) <= b.radius)
        })
        .map(|(entity, pos, _)| (entity, pos.0))
        .collect();
    for (entity, pos) in fired {
        commands.entity(entity).despawn();
        commands.spawn((
            crate::comps_a::GameCleanup,
            crate::comps_a::LevelCleanup,
            crate::comps_b::GroundDetail {
                path: "images/sprPizzaEntrance.png",
                frame: 1,
                flip_x: false,
            },
            Pos(pos),
        ));
        for (_, _, mut hp) in enemies.iter_mut() {
            hp.hp = 0;
        }
        let mut enemy_shots = &mut shots;
        crate::progression::spawn_portal(&mut commands, &catalog, &mut enemy_shots, pos, 1);
        run.queued_secret = Some(crate::data::SecretTarget::PizzaSewers);
    }
}

/// GML `Player/Collision_CarVenusFixed.gml`: an interact press on the
/// crib's parked car opens a plain `type = 1` portal, empties the floor and
/// routes the run either into `area_mansion` or - in the crib, or anywhere
/// the Cuz has followed you - back to `lastarea` / `lastsubarea - 1`, one
/// subarea further back than the crib's own exit portal.
pub fn tick_car_venus_fixed(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    input: Res<crate::input::NtInput>,
    mut run: ResMut<Run>,
    mut triggers: ResMut<SecretTriggers>,
    mut cues: ResMut<crate::msg::Queue<crate::audio::AudioCue>>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    cuz_q: Query<(), With<crate::comps_b::YungCuz>>,
    cars: Query<(Entity, &Pos, &crate::comps_b::PropSprites), Without<Player>>,
    mut shots: Query<(Entity, &crate::comps_a::Team), With<crate::Projectile>>,
    mut enemies: Query<(Entity, &mut crate::comps_a::Health), With<Enemy>>,
) {
    if cars.is_empty() || !input.peek_interact_pressed() {
        return;
    }
    let Ok(ppos) = player_q.single().map(|p| p.0) else {
        return;
    };
    let Some(car) = crate::pickups::nearest_car_venus(
        ppos,
        cars.iter()
            .filter(|(_, _, sprites)| {
                matches!(
                    sprites.idle,
                    "images/sprVenusCarFixed.png" | "images/sprVenuzCar2.png"
                )
            })
            .map(|(entity, pos, sprites)| (entity, pos.0, sprites.flip_x)),
    ) else {
        return;
    };
    // GML `Player/Collision_CarVenusFixed.gml:23` `instance_destroy(other.id,
    // false)`, which skips `Destroy_0` - no explosion, no replacement car.
    commands.entity(car).despawn();

    if run.area == AreaId::Crib || !cuz_q.is_empty() {
        triggers.car_out_of_crib = true;
    } else {
        triggers.queue(crate::data::SecretTarget::YvMansion);
    }
    let at = ppos;
    crate::progression::spawn_portal(&mut commands, &catalog, &mut shots, at, 1);
    for (_, mut health) in enemies.iter_mut() {
        health.hp = 0;
    }
    cues.push(crate::audio::AudioCue {
        name: "sndUseCar",
        volume: 1.0,
        variance: 0.0,
    });
    run.portal_open = false;
}

/// GML `CanOasis/Create_0.gml:2-4`: the moment the desert floor's chest
/// condition holds, a `CanOasis` opens the 300-step (10 s) window in which a
/// Big Bandit death reroutes the run to the Oasis. It closes itself again
/// when the window lapses.
pub fn tick_can_oasis(
    time: Res<SimTime>,
    mut commands: Commands,
    mut triggers: ResMut<SecretTriggers>,
    mut run: ResMut<Run>,
    mut open: Query<(Entity, &mut crate::comps_b::CanOasis)>,
) {
    if triggers.oasis_chests_ready && triggers.oasis_eligible {
        triggers.oasis_chests_ready = false;
        run.can_oasis = true;
        commands.spawn((
            crate::comps_a::GameCleanup,
            crate::comps_a::LevelCleanup,
            crate::comps_b::CanOasis {
                timer: GTimer::from_seconds(300.0 / 30.0, TimerMode::Once),
            },
        ));
    }
    let dt = time.delta_secs;
    let mut live = 0;
    for (entity, mut window) in &mut open {
        window.timer.tick(dt);
        if window.timer.just_finished() {
            commands.entity(entity).despawn();
        } else {
            live += 1;
        }
    }
    run.can_oasis = live > 0;
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

/// GML `Player/Collision_Van.gml`: driving a parked IDPD van takes the run
/// to the HQ, once. The van offers the ride only while `drawspr` is
/// `sprVanDeactivate` - from `Alarm_1` until `Alarm_3` clears `can_hq` 35
/// steps later - and a freak van never offers it at all. A second press on
/// any later van just kills it.
pub fn tick_van_hq(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    input: Res<crate::input::NtInput>,
    mut run: ResMut<Run>,
    mut triggers: ResMut<SecretTriggers>,
    mut cues: ResMut<crate::msg::Queue<crate::audio::AudioCue>>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Pos,
            &mut crate::comps_a::Health,
            Option<&crate::idpd::IdpdVanDeploy>,
        ),
        (With<Enemy>, Without<Player>),
    >,
    mut shots: Query<(Entity, &crate::comps_a::Team), With<crate::Projectile>>,
) {
    if enemies.is_empty() || !input.peek_interact_pressed() {
        return;
    }
    let Ok(ppos) = player_q.single().map(|p| p.0) else {
        return;
    };
    let Some(entity) = crate::pickups::nearest_prompt_span(
        ppos,
        enemies
            .iter()
            .filter(|(_, _, _, deploy)| deploy.is_some_and(|d| !d.freak && d.inert > 0.0))
            .map(|(entity, pos, _, _)| (entity, pos.0, crate::pickups::VAN_PROMPT)),
    ) else {
        return;
    };

    if run.tried_hq {
        // GML `Player/Collision_Van.gml:5-8`: `with (other) hp = 0` and out.
        if let Ok((_, _, mut hp, _)) = enemies.get_mut(entity) {
            hp.hp = 0;
        }
        return;
    }
    // GML `:12-22`: the self-assigning `hqarea = hqarea` means re-entering
    // from inside the HQ is a no-op, so only the first ride reroutes.
    if run.area != AreaId::HQ {
        triggers.queue(SecretTarget::Hq);
    }
    run.tried_hq = true;
    cues.push(crate::audio::AudioCue {
        name: "sndUseVan",
        volume: 1.0,
        variance: 0.0,
    });
    for (_, _, mut hp, _) in enemies.iter_mut() {
        hp.hp = 0;
    }
    commands.entity(entity).despawn();
    crate::progression::spawn_portal(&mut commands, &catalog, &mut shots, ppos, 2);
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
