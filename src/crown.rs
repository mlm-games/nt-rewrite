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

#[cfg(test)]
mod tests {
    use super::*;
    use repame_anim::{AnimCatalog, AtlasDesc};

    fn sim_time() -> SimTime {
        SimTime {
            elapsed_secs: 0.0,
            delta_secs: 1.0 / 30.0,
        }
    }

    fn test_catalog() -> AnimCatalog {
        AnimCatalog::from_json(
            "{}",
            AtlasDesc {
                size: 128,
                max_pages: 1,
            padding: 0,
            },
        )
        .expect("empty catalog")
    }

    fn spawn_player(world: &mut World, crown: CrownKind, hp: i32, max: i32) -> Entity {
        let mut player = Player::default();
        player.crown = crown;
        world
            .spawn((
                player,
                Health {
                    hp,
                    max,
                    invuln: GTimer::from_seconds(0.0, TimerMode::Once),
                },
                Inventory {
                    weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
                    weapon_slots: 2,
                    current: 0,
                    ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
                },
                CrownState::new(crown),
                Pos(glam::Vec2::ZERO),
                Velocity(glam::Vec2::ZERO),
            ))
            .id()
    }

    #[test]
    fn crown_death_reduces_max_by_one() {
        let mut p = Player::default();
        let mut h = Health {
            hp: 8,
            max: 8,
            invuln: GTimer::from_seconds(0.0, TimerMode::Once),
        };
        let mut inv = Inventory {
            weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
            weapon_slots: 2,
            current: 0,
            ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
        };
        apply_crown_to_spawn(CrownKind::Death, &mut p, &mut h, &mut inv);
        assert_eq!(p.crown, CrownKind::Death);
        assert_eq!(h.max, 7);
        assert_eq!(h.hp, 7);
    }

    #[test]
    fn crown_luck_enables_lucky_shot() {
        let mut p = Player::default();
        let mut h = Health {
            hp: 8,
            max: 8,
            invuln: GTimer::from_seconds(0.0, TimerMode::Once),
        };
        let mut inv = Inventory {
            weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
            weapon_slots: 2,
            current: 0,
            ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
        };
        apply_crown_to_spawn(CrownKind::Luck, &mut p, &mut h, &mut inv);
        assert!(p.lucky_shot);
    }

    #[test]
    fn crown_destiny_fills_empty_stored_slot() {
        let mut p = Player::default();
        let mut h = Health {
            hp: 8,
            max: 8,
            invuln: GTimer::from_seconds(0.0, TimerMode::Once),
        };
        let mut inv = Inventory {
            weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
            weapon_slots: 2,
            current: 0,
            ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
        };
        apply_crown_to_spawn(CrownKind::Destiny, &mut p, &mut h, &mut inv);
        assert_ne!(inv.weapons[1], WeaponId::NONE);
        assert_eq!(p.mutation_picks_owed, 1);
    }

    #[test]
    fn crown_life_heals_after_two_seconds() {
        let mut world = World::new();
        world.insert_resource(sim_time());
        spawn_player(&mut world, CrownKind::Life, 5, 8);
        let mut sched = Schedule::default();
        sched.add_systems(tick_crown_life);
        for _ in 0..30 {
            sched.run(&mut world);
        }
        let mut q = world.query_filtered::<&Health, With<Player>>();
        assert_eq!(q.iter(&world).next().unwrap().hp, 5);
        // 2 s period (+ float margin): exactly one heal lands.
        for _ in 0..40 {
            sched.run(&mut world);
        }
        let mut q = world.query_filtered::<&Health, With<Player>>();
        assert_eq!(q.iter(&world).next().unwrap().hp, 6);
    }

    #[test]
    fn crown_protection_shields_below_half_and_rearms() {
        let mut world = World::new();
        let e = spawn_player(&mut world, CrownKind::Protection, 3, 8);
        let mut sched = Schedule::default();
        sched.add_systems(tick_crown_protection);
        sched.run(&mut world);
        assert!(world.get::<Shield>(e).is_some());
        assert!(!world.get::<CrownState>(e).unwrap().protection_ready);
        // Second tick with a shield present: no duplicate.
        sched.run(&mut world);
        // Heal above half: re-arms.
        world.get_mut::<Health>(e).unwrap().hp = 8;
        world.entity_mut(e).remove::<Shield>();
        sched.run(&mut world);
        assert!(world.get::<CrownState>(e).unwrap().protection_ready);
    }

    #[test]
    fn crown_love_spawns_ally_after_35s() {
        let mut world = World::new();
        world.insert_resource(sim_time());
        spawn_player(&mut world, CrownKind::Love, 8, 8);
        let mut sched = Schedule::default();
        sched.add_systems(tick_crown_love);
        for _ in 0..35 * 30 {
            sched.run(&mut world);
        }
        // 35 s period (+ float margin): at most the first ally so far.
        for _ in 0..5 {
            sched.run(&mut world);
        }
        let mut q = world.query::<(&Ally, &Pos)>();
        let (ally, pos) = q.iter(&world).next().expect("ally spawned");
        assert_eq!(pos.0, glam::Vec2::new(40.0, 0.0));
        assert_eq!(ally.life.duration(), 18.0);
    }

    #[test]
    fn crown_curses_bursts_four_enemy_bolts() {
        let mut world = World::new();
        world.insert_resource(sim_time());
        spawn_player(&mut world, CrownKind::Curses, 8, 8);
        let mut sched = Schedule::default();
        sched.add_systems(tick_crown_curses);
        for _ in 0..14 * 30 - 30 {
            sched.run(&mut world);
        }
        assert_eq!(world.query::<&Projectile>().iter(&world).count(), 0);
        // 14 s period (+ float margin): exactly one burst lands.
        for _ in 0..35 {
            sched.run(&mut world);
        }
        let mut q = world.query::<(&Projectile, &Team, &Velocity)>();
        let bolts: Vec<_> = q.iter(&world).collect();
        assert_eq!(bolts.len(), 4);
        for (proj, team, _) in &bolts {
            assert_eq!(proj.damage, 2);
            assert_eq!(**team, Team::Enemy);
        }
    }

    #[test]
    fn crown_floor_start_destiny_is_deterministic() {
        let run_once = || {
            let mut world = World::new();
            world.insert_resource(Queue::<FloorStarted>::default());
            world.insert_resource(test_catalog());
            let mut p = Player::default();
            p.crown = CrownKind::Destiny;
            p.level = 3;
            world.spawn((
                p,
                CrownState::new(CrownKind::Destiny),
                Inventory {
                    weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
                    weapon_slots: 2,
                    current: 0,
                    ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
                },
                Pos(glam::Vec2::new(100.0, 0.0)),
                Health {
                    hp: 8,
                    max: 8,
                    invuln: GTimer::from_seconds(0.0, TimerMode::Once),
                },
            ));
            world
                .resource_mut::<Queue<FloorStarted>>()
                .push(FloorStarted {
                    floor: 2,
                    area: crate::data::AreaId::Desert,
                });
            let mut sched = Schedule::default();
            sched.add_systems(crown_floor_start_bonus);
            sched.run(&mut world);
            let mut q = world.query::<(&Inventory, &CrownState)>();
            let (inv, st) = q.iter(&world).next().unwrap();
            (inv.weapons[0], st.destiny_ready)
        };
        let (w1, ready1) = run_once();
        let (w2, ready2) = run_once();
        assert_eq!(w1, w2);
        assert!(!ready1 && !ready2, "destiny consumed once");
    }

    #[test]
    fn crown_floor_start_risk_luck_guns() {
        let mut world = World::new();
        world.insert_resource(Queue::<FloorStarted>::default());
        world.insert_resource(test_catalog());
        // Risk: ammo refill.
        let mut risk = Player::default();
        risk.crown = CrownKind::Risk;
        world.spawn((
            risk,
            CrownState::new(CrownKind::Risk),
            Inventory {
                weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
                weapon_slots: 2,
                current: 0,
                ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
            },
            Pos(glam::Vec2::ZERO),
            Health {
                hp: 8,
                max: 8,
                invuln: GTimer::from_seconds(0.0, TimerMode::Once),
            },
        ));
        world
            .resource_mut::<Queue<FloorStarted>>()
            .push(FloorStarted {
                floor: 1,
                area: crate::data::AreaId::Desert,
            });
        let mut sched = Schedule::default();
        sched.add_systems(crown_floor_start_bonus);
        sched.run(&mut world);
        let mut q = world.query::<(&Player, &Inventory)>();
        let (_, inv) = q.iter(&world).next().unwrap();
        assert!(inv.ammo_of(AmmoKind::Bullets) > 0);

        // Luck: clamp to 1 HP. Guns: weapon drop in normal areas.
        let mut world = World::new();
        world.insert_resource(Queue::<FloorStarted>::default());
        world.insert_resource(test_catalog());
        let mut luck = Player::default();
        luck.crown = CrownKind::Luck;
        world.spawn((
            luck,
            CrownState::new(CrownKind::Luck),
            Inventory {
                weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
                weapon_slots: 2,
                current: 0,
                ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
            },
            Pos(glam::Vec2::ZERO),
            Health {
                hp: 6,
                max: 8,
                invuln: GTimer::from_seconds(0.0, TimerMode::Once),
            },
        ));
        let mut guns = Player::default();
        guns.crown = CrownKind::Guns;
        world.spawn((
            guns,
            CrownState::new(CrownKind::Guns),
            Inventory {
                weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
                weapon_slots: 2,
                current: 0,
                ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
            },
            Pos(glam::Vec2::ZERO),
            Health {
                hp: 8,
                max: 8,
                invuln: GTimer::from_seconds(0.0, TimerMode::Once),
            },
        ));
        world
            .resource_mut::<Queue<FloorStarted>>()
            .push(FloorStarted {
                floor: 1,
                area: crate::data::AreaId::Desert,
            });
        let mut sched = Schedule::default();
        sched.add_systems(crown_floor_start_bonus);
        sched.run(&mut world);
        let mut hq = world.query::<(&Player, &Health)>();
        for (p, h) in hq.iter(&world) {
            if p.crown == CrownKind::Luck {
                assert_eq!(h.hp, 1);
            }
        }
        assert_eq!(
            world
                .query::<&Pickup>()
                .iter(&world)
                .filter(|p| matches!(p.kind, PickupKind::Weapon(_)))
                .count(),
            1
        );

        // Guns skipped in secret areas.
        let mut world = World::new();
        world.insert_resource(Queue::<FloorStarted>::default());
        world.insert_resource(test_catalog());
        let mut guns = Player::default();
        guns.crown = CrownKind::Guns;
        world.spawn((
            guns,
            CrownState::new(CrownKind::Guns),
            Inventory {
                weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
                weapon_slots: 2,
                current: 0,
                ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
            },
            Pos(glam::Vec2::ZERO),
            Health {
                hp: 8,
                max: 8,
                invuln: GTimer::from_seconds(0.0, TimerMode::Once),
            },
        ));
        world
            .resource_mut::<Queue<FloorStarted>>()
            .push(FloorStarted {
                floor: 1,
                area: crate::data::AreaId::Vault,
            });
        let mut sched = Schedule::default();
        sched.add_systems(crown_floor_start_bonus);
        sched.run(&mut world);
        assert_eq!(world.query::<&Pickup>().iter(&world).count(), 0);
    }

    #[test]
    fn crown_love_converts_chests_to_ammo() {
        let mut world = World::new();
        world.insert_resource(test_catalog());
        let mut p = Player::default();
        p.crown = CrownKind::Love;
        world.spawn((p, Pos(glam::Vec2::ZERO)));
        world.spawn((
            Pickup {
                kind: PickupKind::Chest(ChestKind::Weapon),
            },
            Pos(glam::Vec2::new(10.0, 0.0)),
        ));
        world.spawn((
            Pickup {
                kind: PickupKind::Chest(ChestKind::Rad),
            },
            Pos(glam::Vec2::new(-10.0, 0.0)),
        ));
        let mut sched = Schedule::default();
        sched.add_systems(tick_crown_love_convert);
        sched.run(&mut world);
        let kinds: Vec<_> = world
            .query::<&Pickup>()
            .iter(&world)
            .map(|p| p.kind)
            .collect();
        assert_eq!(kinds.len(), 2);
        assert!(
            kinds
                .iter()
                .all(|k| matches!(k, PickupKind::Chest(ChestKind::Ammo)))
        );
    }

    #[test]
    fn crown_life_converts_rad_to_health() {
        let mut world = World::new();
        world.insert_resource(test_catalog());
        let mut p = Player::default();
        p.crown = CrownKind::Life;
        world.spawn((p, Pos(glam::Vec2::ZERO)));
        world.spawn((
            Pickup {
                kind: PickupKind::Chest(ChestKind::Rad),
            },
            Pos(glam::Vec2::new(10.0, 0.0)),
        ));
        world.spawn((
            Pickup {
                kind: PickupKind::Chest(ChestKind::Weapon),
            },
            Pos(glam::Vec2::new(-10.0, 0.0)),
        ));
        let mut sched = Schedule::default();
        sched.add_systems(tick_crown_life_convert);
        sched.run(&mut world);
        let kinds: Vec<_> = world
            .query::<&Pickup>()
            .iter(&world)
            .map(|p| p.kind)
            .collect();
        assert_eq!(kinds.len(), 2);
        assert!(
            kinds
                .iter()
                .any(|k| matches!(k, PickupKind::Chest(ChestKind::Health))),
            "rad becomes health"
        );
        assert!(
            kinds
                .iter()
                .any(|k| matches!(k, PickupKind::Chest(ChestKind::Weapon))),
            "weapon untouched"
        );
    }

    #[test]
    fn crown_luck_marks_seen_and_clears_when_removed() {
        use crate::comps_b::Enemy;
        let mut world = World::new();
        let mut p = Player::default();
        p.crown = CrownKind::Luck;
        world.spawn((p, Pos(glam::Vec2::ZERO)));
        for _ in 0..60 {
            world.spawn((
                Enemy {
                    kind: crate::data::EnemyKind::Bandit,
                    score: 0,
                    touch_damage: 1,
                    rad_drop: 0,
                    drop_chance: 0,
                    weapon_chance: 0,
                },
                Health {
                    hp: 8,
                    max: 8,
                    invuln: GTimer::from_seconds(0.0, TimerMode::Once),
                },
            ));
        }
        // Bosses are never touched.
        world.spawn((
            Enemy {
                kind: crate::data::EnemyKind::BigBandit,
                score: 0,
                touch_damage: 1,
                rad_drop: 0,
                drop_chance: 0,
                weapon_chance: 0,
            },
            Health {
                hp: 40,
                max: 40,
                invuln: GTimer::from_seconds(0.0, TimerMode::Once),
            },
        ));
        let mut sched = Schedule::default();
        sched.add_systems(tick_crown_luck);
        sched.run(&mut world);
        let mut eq = world.query::<(&Enemy, &Health)>();
        let mut ones = 0;
        for (e, h) in eq.iter(&world) {
            if enemy_def(e.kind).boss {
                assert_eq!(h.hp, 40);
            } else if h.hp == 1 {
                ones += 1;
            }
        }
        assert!(ones > 0, "expected some 1-HP rolls, got {ones}");
        // Second run: no re-rolls (seen set), crowns off clears it.
        {
            let mut pq = world.query::<&mut Player>();
            pq.iter_mut(&mut world).next().unwrap().crown = CrownKind::None;
        }
        sched.run(&mut world);
    }

    #[test]
    fn crown_pedestal_applies_unlocks_and_toasts() {
        let mut world = World::new();
        world.init_resource::<Toast>();
        world.init_resource::<SaveData>();
        world.init_resource::<SaveDirty>();
        world.init_resource::<SelectedCharacter>();
        let mut p = Player::default();
        p.crown = CrownKind::None;
        world.spawn((
            p,
            Pos(glam::Vec2::ZERO),
            Health {
                hp: 8,
                max: 8,
                invuln: GTimer::from_seconds(0.0, TimerMode::Once),
            },
            Inventory {
                weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
                weapon_slots: 2,
                current: 0,
                ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
            },
            CrownState::new(CrownKind::None),
        ));
        let ped = world
            .spawn((
                CrownPedestal {
                    kind: CrownKind::Death,
                },
                Pos(glam::Vec2::new(10.0, 0.0)),
            ))
            .id();
        let mut sched = Schedule::default();
        sched.add_systems(tick_crown_pedestal);
        sched.run(&mut world);
        assert!(world.get_entity(ped).is_err());
        let mut pq = world.query::<(&Player, &Health)>();
        let (pl, h) = pq.iter(&world).next().unwrap();
        assert_eq!(pl.crown, CrownKind::Death);
        assert_eq!(h.max, 7);
        assert_eq!(world.resource::<Toast>().text, "CROWN OF DEATH TAKEN");
        assert!(world.resource::<SaveDirty>().0);
    }

    #[test]
    fn crown_toast_names_cover_all_crowns() {
        for crown in CrownKind::ALL {
            assert!(!crown_name_for_toast(crown).is_empty());
        }
        assert_eq!(crown_name_for_toast(CrownKind::Luck), "Crown of Luck");
    }

    #[test]
    fn crown_object_gate_spawns_and_despawns() {
        // GML `scrCrownSetCurrent`: crown held + player → exactly one
        // `CrownObject` at the player; crownless → none.
        let mut world = World::new();
        spawn_player(&mut world, CrownKind::Death, 8, 8);
        let mut sched = Schedule::default();
        sched.add_systems(tick_crown_object);
        sched.run(&mut world);
        sched.run(&mut world);
        let mut q = world.query::<(&CrownObject, &Pos)>();
        let objs: Vec<_> = q.iter(&world).collect();
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].1.0, glam::Vec2::ZERO);
        assert_eq!(objs[0].0.refresh, 1);
        // Duplicates collapse back to one.
        world.spawn((CrownObject { refresh: 9 }, Pos(glam::Vec2::new(5.0, 5.0))));
        sched.run(&mut world);
        assert_eq!(world.query::<&CrownObject>().iter(&world).count(), 1);
        // Dropping the crown clears all.
        {
            let mut pq = world.query::<&mut Player>();
            pq.iter_mut(&mut world).next().unwrap().crown = CrownKind::None;
        }
        sched.run(&mut world);
        assert_eq!(world.query::<&CrownObject>().iter(&world).count(), 0);
    }

    #[test]
    fn crown_pedestal_seats_crown_object() {
        let mut world = World::new();
        world.init_resource::<Toast>();
        world.init_resource::<SaveData>();
        world.init_resource::<SaveDirty>();
        world.init_resource::<SelectedCharacter>();
        let mut p = Player::default();
        p.crown = CrownKind::None;
        world.spawn((
            p,
            Pos(glam::Vec2::ZERO),
            Health {
                hp: 8,
                max: 8,
                invuln: GTimer::from_seconds(0.0, TimerMode::Once),
            },
            Inventory {
                weapons: [WeaponId::REVOLVER, WeaponId::NONE, WeaponId::NONE],
                cursed: [false, false, false],
                swapanim: 0.0,
                shine: 0.0,
                wepflip: 1.0,
                bwepflip: 1.0,
                weapon_slots: 2,
                current: 0,
                ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
            },
            CrownState::new(CrownKind::None),
        ));
        world.spawn((
            CrownPedestal {
                kind: CrownKind::Guns,
            },
            Pos(glam::Vec2::new(10.0, 0.0)),
        ));
        let mut sched = Schedule::default();
        sched.add_systems(tick_crown_pedestal);
        sched.run(&mut world);
        assert_eq!(world.query::<&CrownObject>().iter(&world).count(), 1);
    }
}
