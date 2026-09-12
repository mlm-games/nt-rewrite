//! Crown effects. Port of `nt-recreated-bevy/src/game/crown.rs` (all 10
//! functions): spawn-time stat application, per-tick Life / Protection /
//! Love / Curses / Luck / Love-convert behaviors, floor-start bonuses,
//! toast names, and pedestal pickup.
//!
//! Render split: ally/projectile/pickup spawns keep gameplay components
//! only (`Ally`, `Projectile`, `Pos`, …); sprite/handle/text code from
//! the bevy build is omitted (render/UI phase), never stubbed.

use std::collections::HashSet;

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_sim::SimTime;

use crate::comps_a::{
    CrownState, FloorStarted, GameCleanup, Health, Hitbox, Inventory, LevelCleanup, Player,
    Projectile, SaveDirty, SelectedCharacter, Team, Toast, Velocity,
};
use crate::comps_b::{Ally, ChestKind, CrownObject, CrownPedestal, Enemy, Pickup, PickupKind, Shield};
use crate::data::{AmmoKind, CrownKind, WeaponId, ammo_pickup_amount};
use crate::enemy_data::enemy_def;
use crate::msg::Queue;
use crate::savedata_part::SaveData;
use crate::spatial::Pos;
use crate::time::{GTimer, TimerMode};

/// Apply a crown's spawn-time stats. Dependency-free (pure `Player` /
/// `Health` / `Inventory` mutation) so `setup.rs` can call it later.
pub fn apply_crown_to_spawn(
    crown: CrownKind,
    player: &mut Player,
    health: &mut Health,
    inv: &mut Inventory,
) {
    player.crown = crown;

    match crown {
        CrownKind::None => {}

        CrownKind::Death => {
            health.max = (health.max - 1).max(1);
            health.hp = health.hp.min(health.max).max(1);
            player.drop_mult += 1.0;
        }

        CrownKind::Life => {
            player.medkit_mult *= 1.25;
        }

        CrownKind::Haste => {
            player.fire_rate_mult *= 1.0;
        }

        CrownKind::Guns => {
            player.drop_mult += 1.5;
        }

        CrownKind::Hatred => {
            player.fire_rate_mult *= 0.9;
            player.spread_mult *= 1.15;
        }

        CrownKind::Blood => {
            player.drop_mult += 0.5;
            player.pickup_range += 32.0;
        }

        CrownKind::Destiny => {
            if inv.weapons[1] == WeaponId::NONE {
                inv.weapons[1] = WeaponId::ASSAULT_RIFLE;
            }
            player.mutation_picks_owed += 1;
        }

        CrownKind::Love => {
            player.medkit_mult *= 1.1;
        }

        CrownKind::Risk => {
            player.drop_mult += 2.5;
            player.medkit_mult *= 0.65;
        }

        CrownKind::Curses => {
            player.drop_mult += 1.0;
            player.spread_mult *= 1.08;
        }

        CrownKind::Luck => {
            player.lucky_shot = true;
            player.drop_mult += 0.4;
        }

        CrownKind::Protection => {
            player.shield_on_hit = true;
        }
    }
}

/// Crown of Life: heal 1 HP every 2 s while below max.
pub fn tick_crown_life(
    time: Res<SimTime>,
    mut q: Query<(&mut CrownState, &mut Health), With<Player>>,
) {
    let dt = time.delta_secs;
    for (mut state, mut health) in &mut q {
        if state.crown != CrownKind::Life {
            continue;
        }

        state.life_timer.tick(dt);

        if state.life_timer.just_finished() && health.hp < health.max {
            health.hp = (health.hp + 1).min(health.max);
        }
    }
}

/// Crown of Protection: dropping to half HP or below grants a brief
/// shield (re-arms once healed above half).
pub fn tick_crown_protection(
    mut commands: Commands,
    mut q: Query<(Entity, &mut CrownState, &mut Health, Option<&Shield>), With<Player>>,
) {
    for (entity, mut state, mut health, shield) in &mut q {
        if state.crown != CrownKind::Protection {
            continue;
        }

        if health.hp > health.max / 2 {
            state.protection_ready = true;
            continue;
        }

        if !state.protection_ready {
            continue;
        }

        if shield.is_some() {
            continue;
        }

        state.protection_ready = false;
        health.invuln = GTimer::from_seconds(0.75, TimerMode::Once);

        commands.entity(entity).insert(Shield {
            timer: GTimer::from_seconds(1.25, TimerMode::Once),
        });
    }
}

/// Crown of Love: spawn a temporary ally every 35 s.
pub fn tick_crown_love(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(&mut CrownState, &Pos), With<Player>>,
) {
    let dt = time.delta_secs;
    for (mut state, pos) in &mut q {
        if state.crown != CrownKind::Love {
            continue;
        }

        state.love_timer.tick(dt);

        if !state.love_timer.just_finished() {
            continue;
        }

        let at = pos.0 + glam::Vec2::new(40.0, 0.0);

        commands.spawn((
            GameCleanup,
            LevelCleanup,
            Ally {
                life: GTimer::from_seconds(18.0, TimerMode::Once),
                shoot: GTimer::from_seconds(0.45, TimerMode::Repeating),
            },
            Team::Player,
            Health {
                hp: 10,
                max: 10,
                invuln: GTimer::from_seconds(0.35, TimerMode::Once),
            },
            Hitbox { radius: 10.0 },
            Velocity(glam::Vec2::ZERO),
            Pos(at),
        ));
    }
}

/// Crown of Curses: radial burst of 4 hostile bolts every 14 s.
pub fn tick_crown_curses(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(&mut CrownState, &Pos), With<Player>>,
) {
    let dt = time.delta_secs;
    for (mut state, pos) in &mut q {
        if state.crown != CrownKind::Curses {
            continue;
        }

        state.curses_timer.tick(dt);

        if !state.curses_timer.just_finished() {
            continue;
        }

        let center = pos.0;
        let mut rng = rand::rng();

        for _ in 0..4 {
            let dir = glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU));
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                Projectile {
                    damage: 2,
                    life: GTimer::from_seconds(0.75, TimerMode::Once),
                    radius: 4.0,
                    knockback: 10.0,
                    explosive: false,
                    source: None,
                },
                Team::Enemy,
                Velocity(dir * 230.0),
                Pos(center + dir * 80.0),
            ));
        }
    }
}

const DESTINY_POOL: [WeaponId; 7] = [
    WeaponId::ASSAULT_RIFLE,
    WeaponId::CROSSBOW,
    WeaponId::GRENADE_LAUNCHER,
    WeaponId(38),
    WeaponId(58),
    WeaponId(72),
    WeaponId(104),
];

/// Floor-start bonuses for Destiny (deterministic weapon swap), Risk
/// (ammo refill), Luck (clamp to 1 HP) and Guns (bonus weapon drop,
/// skipped in secret areas).
pub fn crown_floor_start_bonus(
    mut started: ResMut<Queue<FloorStarted>>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut q: Query<
        (
            &Player,
            &mut CrownState,
            &mut Inventory,
            &Pos,
            &mut Health,
        ),
        With<Player>,
    >,
) {
    let mut start: Option<FloorStarted> = None;
    for event in started.drain() {
        start = Some(event);
    }
    let Some(floor) = start else {
        return;
    };

    for (player, mut state, mut inv, pos, mut health) in &mut q {
        match player.crown {
            CrownKind::Destiny => {
                if !state.destiny_ready {
                    continue;
                }
                state.destiny_ready = false;

                let idx = ((pos.0.x.abs() as usize)
                    + player.level as usize * 3
                    + floor.floor as usize)
                    % DESTINY_POOL.len();
                let slot = inv.current.min(inv.weapon_slots.saturating_sub(1));
                inv.weapons[slot] = DESTINY_POOL[idx];
            }

            CrownKind::Risk => {
                for ammo in [
                    AmmoKind::Bullets,
                    AmmoKind::Shells,
                    AmmoKind::Bolts,
                    AmmoKind::Explosives,
                    AmmoKind::Energy,
                ] {
                    let slot = inv.ammo_mut(ammo);
                    *slot = (*slot + ammo_pickup_amount(ammo)).min(player.ammo_cap(ammo));
                }
            }

            CrownKind::Luck => {
                if health.hp > 1 {
                    health.hp = 1;
                }
            }

            CrownKind::Guns => {
                if crate::worldgen::is_secret_area(floor.area) {
                    continue;
                }
                crate::pickups::spawn_pickup(
                    &mut commands,
                    &catalog,
                    PickupKind::Weapon(WeaponId::ASSAULT_RIFLE),
                    pos.0 + glam::Vec2::new(36.0, -28.0),
                    0,
                    false,
                );
            }

            _ => {}
        }
    }
}

/// Toast label for a crown (bevy `crown_name` parity).
pub fn crown_name_for_toast(crown: CrownKind) -> &'static str {
    match crown {
        CrownKind::None => "No Crown",
        CrownKind::Death => "Crown of Death",
        CrownKind::Life => "Crown of Life",
        CrownKind::Haste => "Crown of Haste",
        CrownKind::Guns => "Crown of Guns",
        CrownKind::Hatred => "Crown of Hatred",
        CrownKind::Blood => "Crown of Blood",
        CrownKind::Destiny => "Crown of Destiny",
        CrownKind::Love => "Crown of Love",
        CrownKind::Risk => "Crown of Risk",
        CrownKind::Curses => "Crown of Curses",
        CrownKind::Luck => "Crown of Luck",
        CrownKind::Protection => "Crown of Protection",
    }
}

/// Crown of Love: weapon/rad chests become ammo chests. Runs on floor
/// start and continuously for boss-drop chests.
pub fn tick_crown_love_convert(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    player_q: Query<&Player, With<Player>>,
    mut chests: Query<(Entity, &Pickup, &Pos)>,
) {
    let Ok(player) = player_q.single() else {
        return;
    };
    if player.crown != CrownKind::Love {
        return;
    }
    for (e, pickup, pos) in &mut chests {
        // GML Crown Love: every chestprop (Weapon/Rad/Ammo/Health/
        // CursedBig — no Proto/Rogue kinds exist yet) becomes Ammo.
        let is_convertible = matches!(
            pickup.kind,
            PickupKind::Chest(ChestKind::Weapon)
                | PickupKind::Chest(ChestKind::Rad)
                | PickupKind::Chest(ChestKind::Ammo)
                | PickupKind::Chest(ChestKind::Health)
                | PickupKind::Chest(ChestKind::CursedBig)
        );
        if !is_convertible {
            continue;
        }
        let at = pos.0;
        commands.entity(e).despawn();
        crate::pickups::spawn_chest(&mut commands, &catalog, ChestKind::Ammo, at);
    }
}

/// Crown of Life: rad chests become health chests (GML `scrPopChests`
/// crowns region). Runs on floor start and continuously, mirroring the
/// Love converter.
pub fn tick_crown_life_convert(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    player_q: Query<&Player, With<Player>>,
    mut chests: Query<(Entity, &Pickup, &Pos)>,
) {
    let Ok(player) = player_q.single() else {
        return;
    };
    if player.crown != CrownKind::Life {
        return;
    }
    for (e, pickup, pos) in &mut chests {
        if !matches!(pickup.kind, PickupKind::Chest(ChestKind::Rad)) {
            continue;
        }
        let at = pos.0;
        commands.entity(e).despawn();
        crate::pickups::spawn_chest(&mut commands, &catalog, ChestKind::Health, at);
    }
}

/// Crown of Luck: 10% of non-boss enemies spawn at 1 HP. Each enemy
/// rolls once (tracked in `seen`).
pub fn tick_crown_luck(
    player_q: Query<&Player, With<Player>>,
    mut enemies: Query<(Entity, &Enemy, &mut Health), With<Enemy>>,
    mut seen: Local<HashSet<Entity>>,
) {
    let Ok(player) = player_q.single() else {
        return;
    };
    if player.crown != CrownKind::Luck {
        seen.clear();
        return;
    }
    let mut rng = rand::rng();
    for (e, enemy, mut health) in &mut enemies {
        if !seen.insert(e) {
            continue;
        }
        if enemy_def(enemy.kind).boss {
            continue;
        }
        if rng.random::<f32>() <= 0.1 {
            health.hp = 1;
        }
    }
}

/// GML `scrCrownSetCurrent` (`scrCrownCheck.gml:26-42`): while a crown
/// is held and a player exists, exactly one `CrownObject` exists (at the
/// player, Alarm-2 heartbeat via `refresh`); otherwise all are cleared.
pub fn tick_crown_object(
    mut commands: Commands,
    player_q: Query<(&Pos, &Player), With<Player>>,
    mut crowns: Query<(Entity, &mut CrownObject)>,
) {
    let Ok((ppos, player)) = player_q.single() else {
        for (e, _) in &crowns {
            commands.entity(e).despawn();
        }
        return;
    };
    if player.crown == CrownKind::None {
        for (e, _) in &crowns {
            commands.entity(e).despawn();
        }
        return;
    }
    let mut first = true;
    for (e, mut crown) in &mut crowns {
        if first {
            first = false;
            // GML `with CrownObject event_perform(ev_alarm, 2)`.
            crown.refresh = crown.refresh.wrapping_add(1);
        } else {
            commands.entity(e).despawn();
        }
    }
    if first {
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            CrownObject { refresh: 0 },
            Pos(ppos.0),
        ));
    }
}

/// Port id (save identity) for a crown: GML crowns are 1-based with an
/// extra offset (`crown_port_to_gml` parity).
fn crown_port_to_gml(id: u8) -> u8 {
    if id == 0 { 1 } else { id + 1 }
}

/// Crown pedestal pickup: touch range applies the crown, resets crown
/// state, records the unlock (row + next-run `start_crown` stamp via
/// `unlock_crown`, bevy parity), toasts, and consumes the pedestal.
pub fn tick_crown_pedestal(
    mut commands: Commands,
    mut toast: ResMut<Toast>,
    mut save: ResMut<SaveData>,
    mut dirty: ResMut<SaveDirty>,
    selected: Res<SelectedCharacter>,
    mut q_player: Query<
        (
            &Pos,
            &mut Player,
            &mut Health,
            &mut Inventory,
            &mut CrownState,
        ),
        With<Player>,
    >,
    pedestals: Query<(Entity, &Pos, &CrownPedestal)>,
    crowns: Query<Entity, With<CrownObject>>,
) {
    let Ok((ppos, mut player, mut health, mut inv, mut state)) = q_player.single_mut() else {
        return;
    };
    let p = ppos.0;
    for (e, pos, ped) in &pedestals {
        if pos.0.distance(p) > 28.0 {
            continue;
        }
        apply_crown_to_spawn(ped.kind, &mut player, &mut health, &mut inv);
        *state = CrownState::new(ped.kind);
        // GML `CrownPickup/Collision_Player`: taking a crown uncurses.
        inv.cursed = [false, false, false];

        let gml = crown_port_to_gml(ped.kind as u8);
        let had = (gml as usize) < 14
            && save
                .crown_got
                .get(&selected.0)
                .is_some_and(|r| r[gml as usize]);
        save.unlock_crown(selected.0, gml);
        if !had {
            dirty.0 = true;
        }
        // GML `CROWN_LIFE`: unlocking a crown as any character.
        if crate::savedata_part::unlock_achievement(&mut save, 22) {
            dirty.0 = true;
            toast.show("CROWN LIFE");
        }
        // GML `VAULT_RAIDER`: every crown held as any character.
        let raider = save.crown_got.values().any(|r| r.iter().all(|b| *b));
        if raider && crate::savedata_part::unlock_achievement(&mut save, 39) {
            dirty.0 = true;
            toast.show("VAULT RAIDER");
        }
        toast.show(&format!(
            "{} TAKEN",
            crown_name_for_toast(ped.kind).to_ascii_uppercase()
        ));
        commands.entity(e).despawn();
    }
    // GML `scrCrownSetCurrent` tail: taking a crown seats the
    // `CrownObject` (kept exact by `tick_crown_object` afterwards).
    if player.crown != CrownKind::None && crowns.single().is_err() {
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            CrownObject { refresh: 0 },
            Pos(p),
        ));
    }
}

