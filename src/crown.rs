//! Crown effects. Port of the GML crown scripts (all 10
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

use crate::audio::AudioCue;
use crate::comps_a::{
    CrownState, FloorStarted, GameCleanup, Health, Hitbox, Inventory, LevelCleanup, NextHurt, Player,
    Projectile, Run, Team, Toast, Velocity,
};
use crate::comps_b::{
    Ally, ChestKind, CrownObject, CrownPedestal, Enemy, Pickup, PickupKind, Prop, PropHpTracker,
    PropSprites, PropTier, RadChestContainer, Shield,
};
use crate::data::{AmmoKind, CrownKind, WeaponId, ammo_pickup_amount};
use crate::enemy_data::enemy_def;
use crate::msg::Queue;
use crate::spatial::Pos;
use crate::time::{GTimer, TimerMode};

/// GML `VaultStatue` (`objects/VaultStatue/Create_0.gml:1-3`: `max_hp
/// = 50`, `size = 2`, `rad = 0`) — the crown-vault guard. The body is a
/// `Prop`; this marker is the `instance_exists(VaultStatue)` test that
/// `CrownPickup/Collision_Player:18,24` gates the type-3 portal and the
/// statue kill on.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct VaultStatue;

/// GML `CrownPickup/Create_0.gml:39,48-49` `lengthdir_x/y(128, ang)`.
pub const VAULT_STATUE_DIST: f32 = 128.0;

/// GML `rng_choose(0, 0, 90, 180, 270)` (`CrownPickup/Create_0.gml:42,45`).
/// `rng_choose` samples its argument list uniformly, so the repeated `0` is
/// double-weighted: p(0) = 2/5, the rest 1/5 each.
fn vault_statue_angle(rng: &mut impl RngExt) -> f32 {
    match rng.random_range(0..5u32) {
        0 | 1 => 0.0,
        2 => 90.0,
        3 => 180.0,
        _ => 270.0,
    }
}

/// GML `CrownPickup/Create_0.gml:37-50` verbatim: the guard-statue angles
/// around the crown pedestal, in spawn order, in degrees.
/// `crownvisits >= 3` -> 4 at 0/90/180/270 and no RNG draw at all; else
/// `crownvisits > 1 || instance_exists(CrownObject)` -> 2 on distinct
/// `rng_choose` angles (the `do..until` rejection loop draws at least once);
/// else none. `crownvisits` is the port's `SecretTriggers::vaults_entered`,
/// already incremented on entry like GML's room-start
/// `GameCont/Other_4.gml:8`.
pub fn vault_statue_angles(
    crownvisits: u8,
    crown_object: bool,
    rng: &mut impl RngExt,
) -> Vec<f32> {
    if crownvisits >= 3 {
        return vec![0.0, 90.0, 180.0, 270.0];
    }
    if crownvisits <= 1 && !crown_object {
        return Vec::new();
    }
    let ang1 = vault_statue_angle(rng);
    let mut ang2 = vault_statue_angle(rng);
    while ang2 == ang1 {
        ang2 = vault_statue_angle(rng);
    }
    vec![ang1, ang2]
}

/// GML `VaultStatue/Create_0.gml:1-16`: a 50 hp, `size = 2` `Prop`.
/// `rad = 0` (`:3`) is not `raddrop` — `prop/Create_0.gml:7` already pins
/// `raddrop = 0`, so the statue drops nothing and the `ProtoStatue`
/// rad-charging branch never applies to it. Lines 18-51 stamp the statue's
/// own floor tiles, which the port builds once in `setup::spawn_level`.
pub fn spawn_vault_statue(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    rng: &mut impl RngExt,
) -> Entity {
    // GML `prop/Create_0.gml:13` `image_xscale = choose(1, -1)`.
    let flip = rng.random_bool(0.5);
    let idle = "images/sprVaultStatue.png";
    let mut ec = commands.spawn((
        GameCleanup,
        LevelCleanup,
        Prop {
            // `sprVaultStatue` is 24x48 with a (4,8)-(19,39) bbox.
            size: glam::Vec2::new(16.0, 32.0),
            hp: 50,
            destructible: true,
            explosive: false,
        },
        PropTier(2),
        PropHpTracker { last_hp: 50 },
        NextHurt::default(),
        PropSprites {
            idle,
            hurt: "images/sprVaultStatueHurt.png",
            dead: "images/sprVaultStatueDead.png",
            flip_x: flip,
        },
        VaultStatue,
        crate::comps_b::SpecialPropDeath::VaultStatue,
        Pos(pos),
    ));
    // GML `VaultStatue/Create_0.gml:13` `image_speed = 0.4`.
    if let Some(def) = catalog.def(idle) {
        ec.insert(crate::anim::SpriteAnim::with_image_speed(
            idle,
            def,
            crate::setup::PROP_IMAGE_SPEED,
        ));
    }
    ec.id()
}

/// GML `CrownPickup/Create_0.gml:37-50` at the pedestal: one statue per
/// chosen angle, 128 px out, `ang1` before `ang2`.
pub fn spawn_crown_vault_statues(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pedestal: glam::Vec2,
    crownvisits: u8,
    crown_object: bool,
    rng: &mut impl RngExt,
) {
    for deg in vault_statue_angles(crownvisits, crown_object, rng) {
        spawn_vault_statue(
            commands,
            catalog,
            pedestal + glam::Vec2::from_angle(deg.to_radians()) * VAULT_STATUE_DIST,
            rng,
        );
    }
}

/// Apply a crown's spawn-time stats. Dependency-free (pure `Player` /
/// `Health` / `Inventory` mutation) so `setup.rs` can call it later.
pub fn apply_crown_to_spawn(
    crown: CrownKind,
    player: &mut Player,
    health: &mut Health,
    _inv: &mut Inventory,
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
            // GML `scrCrownApplyEquipEffect:118-139`: Destiny's whole
            // effect is one free mutation pick (`codpick`), once per run.
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
    mut cues: ResMut<Queue<AudioCue>>,
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
        cues.push(AudioCue {
            name: "sndAllySpawn",
            volume: 1.0,
            variance: 0.2,
        });
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

/// Floor-start bonuses for Risk (ammo refill), Luck (clamp to 1 HP) and
/// Guns (bonus weapon drop, skipped in secret areas). GML has no
/// Destiny floor-start effect: "NARROW FUTURE" is the narrowed
/// level-up offer (`LevCont/Create_0:75,143-149`), not a weapon pool.
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

    for (player, _state, mut inv, pos, mut health) in &mut q {
        match player.crown {
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

/// GML crown name + description verbatim
/// (`scripts/scrCrowns/scrCrowns.gml`: `crown_name[]`/`crown_text[]`
/// by GML crown index, unlocalized defaults, `@` tags included).
/// Takes the GML crown index (see `crown_port_to_gml`).
pub fn crown_name_text_gml(gml: u8) -> (&'static str, &'static str) {
    match gml {
        0 => ("RANDOM", "???"),
        1 => ("NO CROWN", "A BARE HEAD#IS A FAIR HEAD"),
        2 => ("CROWN OF DEATH", "BIGGER @wEXPLOSIONS#@s-1 @rMAX HP@s"),
        3 => ("CROWN OF LIFE", "NO @rHP DROPS@s#@rBIG HP CHESTS@s MORE COMMON"),
        4 => ("CROWN OF HASTE", "@wPICKUPS@s FADE FAST#ARE WORTH MORE"),
        5 => ("CROWN OF GUNS", "NO @yAMMO DROPS@s#MORE @wWEAPON DROPS"),
        6 => ("CROWN OF HATRED", "TAKE @wDAMAGE@s AND GAIN @gRADS@s#WHEN OPENING @wCHESTS@s"),
        7 => ("CROWN OF BLOOD", "MORE @wENEMIES@s#FEWER @gRADS@s"),
        8 => ("CROWN OF DESTINY", "FREE @gMUTATION@s#NARROW FUTURE"),
        9 => ("CROWN OF LOVE", "@yAMMO@s CHESTS ONLY"),
        10 => ("CROWN OF LUCK", "START @wAREAS@s AT 1 @rHP@s#CHANCE @wENEMIES@s HAVE 1 @rHP@s"),
        11 => ("CROWN OF CURSES", "A LOT MORE @pCURSED CHESTS@s"),
        12 => ("CROWN OF RISK", "MORE @wDROPS@s WHEN AT FULL @rHP@s#LESS @wDROPS@s WHEN NOT"),
        _ => ("CROWN OF PROTECTINON", "@wWEAPONS@s CONTAIN @rHP@s#INSTEAD OF @yAMMO@s"),
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

/// GML runs the crown chest conversions ONCE inside `scrPopChests` (the
/// crowns region, lines 124-143), i.e. once per generated room — never per
/// frame. This keys the pass off `Run.gen_seed`, which changes exactly when
/// a new level is generated, so a mid-floor chest is NOT converted and a
/// generated one always is.
fn claim_crown_convert(run: &Run, done: &mut Local<Option<u64>>) -> bool {
    if done.as_ref() == Some(&run.gen_seed) {
        return false;
    }
    **done = Some(run.gen_seed);
    true
}

/// Crown of Love: `scripts/scrPopChests.gml:131-143` verbatim — `with
/// chestprop { if object_index != ProtoChest && object_index !=
/// RogueChest { -> AmmoChest } }` plus a second `with RadChest` arm.
/// `chestprop` is hierarchy-inclusive, so the set is every chest kind
/// except Proto and Rogue; the rad chests are `prop` descendants and get
/// their own arm (the port keeps plain `RadChest` as a `Prop`).
pub fn tick_crown_love_convert(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    run: Res<Run>,
    player_q: Query<&Player, With<Player>>,
    mut chests: Query<(Entity, &Pickup, &Pos)>,
    mut rad_props: Query<(Entity, &Pos), With<RadChestContainer>>,
    mut done: Local<Option<u64>>,
) {
    let Ok(player) = player_q.single() else {
        return;
    };
    if player.crown != CrownKind::Love || !claim_crown_convert(&run, &mut done) {
        return;
    }
    for (e, pickup, pos) in &mut chests {
        if !matches!(
            pickup.kind,
            PickupKind::Chest(
                ChestKind::Weapon
                    | ChestKind::BigWeapon
                    | ChestKind::CursedBig
                    | ChestKind::Gold
                    | ChestKind::Idpd
                    | ChestKind::Ammo
                    | ChestKind::Mystery
                    | ChestKind::Health
                    | ChestKind::RadBig
                    | ChestKind::RadMaggot
            )
        ) {
            continue;
        }
        let at = pos.0;
        commands.entity(e).despawn();
        crate::pickups::spawn_chest(&mut commands, &catalog, ChestKind::Ammo, at);
    }
    for (e, pos) in &mut rad_props {
        let at = pos.0;
        commands.entity(e).despawn();
        crate::pickups::spawn_chest(&mut commands, &catalog, ChestKind::Ammo, at);
    }
}

/// Crown of Life: `scripts/scrPopChests.gml:124-129` verbatim — `with
/// RadChest { -> HealthChest }`, hierarchy-inclusive over `RadChest`,
/// `RadChestBig` and `RadMaggotChest`.
pub fn tick_crown_life_convert(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    run: Res<Run>,
    player_q: Query<&Player, With<Player>>,
    mut chests: Query<(Entity, &Pickup, &Pos)>,
    mut rad_props: Query<(Entity, &Pos), With<RadChestContainer>>,
    mut done: Local<Option<u64>>,
) {
    let Ok(player) = player_q.single() else {
        return;
    };
    if player.crown != CrownKind::Life || !claim_crown_convert(&run, &mut done) {
        return;
    }
    for (e, pickup, pos) in &mut chests {
        if !matches!(
            pickup.kind,
            PickupKind::Chest(ChestKind::RadBig | ChestKind::RadMaggot)
        ) {
            continue;
        }
        let at = pos.0;
        commands.entity(e).despawn();
        crate::pickups::spawn_chest(&mut commands, &catalog, ChestKind::Health, at);
    }
    for (e, pos) in &mut rad_props {
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
pub fn crown_port_to_gml(id: u8) -> u8 {
    if id == 0 { 1 } else { id + 1 }
}

/// Crown pedestal pickup: touch range applies the crown, resets crown
/// state, uncurses, opens the type-3 vault portal when the pedestal is
/// undefended, zeroes every guard statue's hp, records the toast, and
/// consumes the pedestal.
/// Crown *unlocking* is not here: GML `scrCrownUnlock` only runs from
/// `scrUnlocksWinOrLoop` (`scripts/scrUnlocks.gml:243-245`).
pub fn tick_crown_pedestal(
    mut commands: Commands,
    mut toast: ResMut<Toast>,
    catalog: Res<repame_anim::AnimCatalog>,
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
    mut statues: Query<(Entity, &mut Prop), With<VaultStatue>>,
    crowns: Query<Entity, With<CrownObject>>,
    mut shots: Query<(Entity, &Team), With<Projectile>>,
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
        // GML `CrownPickup/Collision_Player:12-16`: taking a crown uncurses.
        inv.cursed = [false, false, false];

        // GML `CrownPickup/Collision_Player:18-20`: an undefended pedestal
        // (no `VaultStatue`) opens the type-3 portal on the spot, so the
        // vault can be left without clearing it.
        if statues.is_empty() {
            crate::progression::spawn_portal(&mut commands, &catalog, &mut shots, pos.0, 3);
        }

        // GML `CrownPickup/Collision_Player:24` `with (VaultStatue) hp = 0`:
        // taking the crown wakes the guard, and every statue's `Destroy_0`
        // zeroes the next one, so N statues field N `CrownGuardian`s.
        for (_, mut statue) in &mut statues {
            statue.hp = 0;
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

