//! Pickup / drop spawning. Ported from nt's `game/pickups.rs` spawn
//! helpers plus the combat drop fns (`spawn_rad_burst`,
//! `maybe_spawn_drop`, `spawn_chest`, …).
//!
//! Render split: only rads carry `SpriteAnim` (bevy parity); static
//! kinds resolve art renderer-side from the same kind -> path table.
//! Juice pop-ins are render juice and skipped (no sim effect).

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::anim::SpriteAnim;
use crate::audio::{AudioCue, GameAudio, QueuedReactiveCue, ReactiveCue};
use crate::combat::Explosion;
use crate::comps_a::{
    FloorMask, GameCleanup, Health, Inventory, LevelCleanup, Player, RaceState, Run, Team, Toast,
    MAX_WEAPON_SLOTS,
};
use crate::comps_b::{
    ChestKind, FlungWeapon, GroundPhysics, OpenedChest, Pickup, PickupCurse, PickupKind,
    PickupLifetime, Portal, PortalCarriedWeapons, PortalClear, Prop, RadChestContainer,
    Telekinesis, WepPickupAmmo,
};
use crate::data::{
    AbilityKind, AmmoKind, CrownKind, MutationId, UltraMutationId, WeaponId, ammo_max,
    ammo_pickup_amount,
};
use crate::effects::{
    ChromaticAberration, FlashWhite, RumbleRequest, chromatic_pulse, rumble, spawn_burst,
};
use crate::input::NtInput;
use crate::msg::Queue;
use crate::progression::{check_level_up, try_recharge_strong_spirit};
use crate::spatial::Pos;
use crate::time::{GTimer, TimerMode};
use crate::weapon_runtime::{weapon_ammo, weapon_id_name, weapon_runtime_def};

/// Art path for a weapon pickup. Weapons-phase stub: nt looks the
/// strip up per weapon id; until the weapons table lands, every weapon
/// uses nt's own ultimate fallback.
pub fn weapon_art_path(_id: WeaponId) -> &'static str {
    // TODO(weapons): resolve per-id strips from the weapons table.
    "images/sprRevolver.png"
}

fn pickup_sprite(kind: PickupKind) -> (&'static str, f32) {
    match kind {
        PickupKind::Rad(_) => ("images/sprRad.png", 12.0),
        PickupKind::Medkit(_) => ("images/sprHP.png", 16.0),
        PickupKind::Ammo(..) => ("images/sprAmmo.png", 12.0),
        PickupKind::Curse => ("images/sprCurse.png", 10.0),
        PickupKind::Weapon(k) => (weapon_art_path(k), 20.0),
        PickupKind::Chest(kind) => match kind {
            ChestKind::Weapon => ("images/sprWeaponChest.png", 32.0),
            ChestKind::Ammo => ("images/sprAmmoChest.png", 32.0),
            ChestKind::Rad => ("images/sprRadChest.png", 32.0),
            ChestKind::Health => ("images/sprHealthChest.png", 32.0),
            ChestKind::CursedBig => ("images/sprCursedChestBig.png", 32.0),
            ChestKind::Rogue => ("images/sprRogueAmmoChest.png", 32.0),
            ChestKind::Proto => ("images/sprProtoChest.png", 32.0),
            ChestKind::BigWeapon => ("images/sprWeaponChestBig.png", 32.0),
            ChestKind::RadBig => ("images/sprRadChestBig.png", 32.0),
            ChestKind::RadMaggot => ("images/sprRadChestMaggot.png", 32.0),
            ChestKind::Idpd => ("images/sprIDPDChest.png", 32.0),
        },
    }
}

pub fn spawn_pickup(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: PickupKind,
    pos: glam::Vec2,
    loops: u32,
    hasted: bool,
) -> Entity {
    let (path, _size) = pickup_sprite(kind);

    let mut rng = rand::rng();
    let mut ec = commands.spawn((GameCleanup, LevelCleanup, Pickup { kind }, Pos(pos)));
    // Only rads animate (bevy parity). Static kinds carry no render
    // handle here; the render phase maps kind -> art path itself
    // (same table as `pickup_sprite`).
    if matches!(kind, PickupKind::Rad(_)) {
        if let Some(def) = catalog.def(path) {
            let mut anim = SpriteAnim::new(path, def);
            anim.timer = GTimer::from_seconds(1.0 / 12.0, TimerMode::Repeating);
            anim.frame = rng.random_range(0..def.frames.max(1));
            ec.insert(anim);
        }
    }
    match kind {
        PickupKind::Rad(_) => {
            ec.insert(PickupLifetime {
                timer: GTimer::from_seconds(10.0 + rng.random_range(0.0..1.0), TimerMode::Once),
            });
        }
        PickupKind::Medkit(_) | PickupKind::Ammo(..) => {
            let init =
                ((200.0 + rng.random_range(0.0..30.0)) / ((5.0 + loops as f32) / 5.0)).ceil();
            let total_steps = if hasted { init / 3.0 } else { init } + 62.0;
            ec.insert(PickupLifetime {
                timer: GTimer::from_seconds(total_steps / 30.0, TimerMode::Once),
            });
        }
        PickupKind::Weapon(_) => {
            ec.insert(WepPickupAmmo(true));

            let ang = rng.random_range(0.0..std::f32::consts::TAU);
            ec.insert(GroundPhysics {
                vel: glam::Vec2::new(ang.cos(), ang.sin()) * rng.random_range(15.0..45.0),
                rotspeed: rng.random_range(0.7..1.0)
                    * if rng.random_bool(0.5) { 1.0 } else { -1.0 },
            });
        }
        PickupKind::Chest(_) => {}
        // GML `Curse` motes drift without a lifetime/physics setup.
        PickupKind::Curse => {}
    }
    ec.id()
}

pub fn spawn_chest(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: ChestKind,
    pos: glam::Vec2,
) {
    let path = match kind {
        ChestKind::Weapon => "images/sprWeaponChest.png",
        ChestKind::Ammo => "images/sprAmmoChest.png",
        ChestKind::Rad => "images/sprRadChest.png",
        ChestKind::Health => "images/sprHealthChest.png",
        ChestKind::CursedBig => "images/sprCursedChestBig.png",
        ChestKind::Rogue => "images/sprRogueAmmoChest.png",
        ChestKind::Proto => "images/sprProtoChest.png",
        ChestKind::BigWeapon => "images/sprWeaponChestBig.png",
        ChestKind::RadBig => "images/sprRadChestBig.png",
        ChestKind::RadMaggot => "images/sprRadChestMaggot.png",
        ChestKind::Idpd => "images/sprIDPDChest.png",
    };
    let mut ec = commands.spawn((
        GameCleanup,
        LevelCleanup,
        Pickup {
            kind: PickupKind::Chest(kind),
        },
        Pos(pos),
    ));
    if let Some(def) = catalog.def(path) {
        ec.insert(SpriteAnim::new(path, def));
    }
}

pub fn spawn_rad(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    amount: u32,
) {
    spawn_pickup(commands, catalog, PickupKind::Rad(amount), pos, 0, false);
}

pub fn spawn_rad_burst(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    mut amount: u32,
) {
    while amount > 15 {
        amount -= 10;
        spawn_rad(commands, catalog, pos + random_offset(), 10);
    }
    for _ in 0..amount {
        spawn_rad(commands, catalog, pos + random_offset(), 1);
    }
}

/// Random unit-ish offset for scatter drops (bevy parity).
pub fn random_offset() -> glam::Vec2 {
    let mut rng = rand::rng();
    let a = rng.random_range(0.0..std::f32::consts::TAU);
    let d = rng.random_range(0.0..22.0);
    glam::Vec2::new(a.cos(), a.sin()) * d
}

/// GML Chicken throw (`scripts/scrPowers/scrPowers.gml:218-242`): drop
/// the held weapon as a flung pickup — speed 16 px/step toward
/// `angle_rad` (gunangle±2), team/creator set, Determination ultra arming
/// the 60-tick `alarm[1]` return.
pub fn spawn_flung_weapon_pickup(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    weapon: WeaponId,
    pos: glam::Vec2,
    angle_rad: f32,
    team: Team,
    creator: Entity,
    return_ticks: u8,
) -> Entity {
    let e = spawn_pickup(commands, catalog, PickupKind::Weapon(weapon), pos, 0, false);
    let dir = glam::Vec2::new(angle_rad.cos(), angle_rad.sin());
    commands.entity(e).insert(GroundPhysics {
        vel: dir * 16.0 * 30.0,
        rotspeed: 0.0,
    });
    commands.entity(e).insert(FlungWeapon {
        team,
        creator,
        return_ticks,
    });
    e
}

/// GML `WepPickup/Alarm_1`: when the Determination return alarm fires,
/// fling the pickup back at its creator (`sndChickenReturn`).
pub fn tick_flung_weapons(
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mut q: Query<(Entity, &Pos, &mut FlungWeapon, Option<&mut GroundPhysics>)>,
    creators: Query<&Pos, Without<FlungWeapon>>,
) {
    for (e, pos, mut flung, ground) in &mut q {
        if flung.return_ticks == 0 {
            continue;
        }
        flung.return_ticks -= 1;
        if flung.return_ticks > 0 {
            continue;
        }
        let Ok(cpos) = creators.get(flung.creator) else {
            continue;
        };
        let dir = (cpos.0 - pos.0).normalize_or_zero();
        let vel = dir * 16.0 * 30.0;
        if let Some(mut g) = ground {
            g.vel = vel;
        } else {
            commands.entity(e).insert(GroundPhysics { vel, rotspeed: 0.0 });
        }
        cues.push(AudioCue {
            name: "sndChickenReturn",
            volume: 0.6,
            variance: 0.05,
        });
    }
}

/// GML `IDPDChest/Destroy_0`: outside worldgen (`instance_exists(GenCont)
/// exit` — the port's `generate_level` is pure, so the flag is
/// caller-provided), leave the `ChestOpen` visual (`sprIDPDChestOpen`),
/// an `FXChestOpen` burst, and 6 `IDPDSpawn`s. Returns false when skipped
/// during generation.
pub fn idpd_chest_destroy(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    during_worldgen: bool,
    loops: u32,
) -> bool {
    if during_worldgen {
        return false;
    }
    let mut ec = commands.spawn((
        GameCleanup,
        LevelCleanup,
        OpenedChest(ChestKind::Idpd),
        Pos(pos),
    ));
    if let Some(def) = catalog.def("images/sprIDPDChestOpen.png") {
        ec.insert(crate::anim::SpriteAnim::new(
            "images/sprIDPDChestOpen.png",
            def,
        ));
    }
    let mut rng = rand::rng();
    // GML `FXChestOpen` (visual pop).
    spawn_burst(
        commands,
        &mut rng,
        pos,
        10,
        [0.7, 0.9, 1.0, 1.0],
        (60.0, 200.0),
    );
    // GML `repeat 6 instance_create(x, y, IDPDSpawn)` (port: the spawn
    // table's default grunt; the alarm-1 table roll rides `idpd.rs`).
    for _ in 0..6 {
        crate::enemies::spawn_enemy(
            commands,
            catalog,
            crate::data::EnemyKind::IdpdGrunt,
            pos + random_offset() * 0.5,
            1.0,
            false,
            false,
            loops,
        );
    }
    true
}

pub fn maybe_spawn_drop(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    chance: usize,
    weapon_chance: usize,
    player: &Player,
    inv: &Inventory,
    health: &Health,
    loops: u32,
) {
    let mut rng = rand::rng();

    let need = scrub_need(inv, player);
    let paw = player.drop_mult;

    let mut weapon_chance = weapon_chance;
    if player.crown == CrownKind::Guns {
        weapon_chance += 9;
    }
    let mut chance = chance as f32;
    if player.crown == CrownKind::Risk {
        chance *= if health.hp >= health.max { 1.5 } else { 0.5 };
    }
    let roll = rng.random_range(0.0..100.0);

    let hasted = player.crown == CrownKind::Haste;

    if roll < (chance * (need + paw)) {
        let hardmode = loops > 0;
        let medkit_win = if hardmode {
            rng.random_range(0..30) < 15
        } else {
            rng.random_range(0..3) < 2
        };
        let life_blocks = player.crown == CrownKind::Life;
        let guns_blocks_ammo = player.crown == CrownKind::Guns;
        if rng.random_range(0..health.max.max(1)) as i32 > health.hp && medkit_win && !life_blocks {
            spawn_pickup(commands, catalog, PickupKind::Medkit(2), pos, loops, hasted);
        } else {
            if !guns_blocks_ammo {
                spawn_pickup(
                    commands,
                    catalog,
                    PickupKind::Ammo(AmmoKind::None, 0),
                    pos,
                    loops,
                    hasted,
                );
            }
        }
    } else if weapon_chance > 0 && rng.random_range(0.0..100.0) < weapon_chance as f32 {
        let weapon = random_weapon(&mut rng);
        spawn_pickup(commands, catalog, PickupKind::Weapon(weapon), pos, 0, false);
    }
}

fn random_ammo_kind(rng: &mut impl rand::RngExt) -> AmmoKind {
    match rng.random_range(0..5) {
        0 => AmmoKind::Bullets,
        1 => AmmoKind::Shells,
        2 => AmmoKind::Bolts,
        3 => AmmoKind::Explosives,
        _ => AmmoKind::Energy,
    }
}

pub fn random_weapon(rng: &mut impl rand::RngExt) -> WeaponId {
    match rng.random_range(0..8) {
        0 => WeaponId::MACHINEGUN,
        1 => WeaponId(5),
        2 => WeaponId::CROSSBOW,
        3 => WeaponId::GRENADE_LAUNCHER,
        4 => WeaponId::SMG,
        5 => WeaponId::ASSAULT_RIFLE,
        6 => WeaponId::WRENCH,
        _ => WeaponId::SLEDGEHAMMER,
    }
}

/// Gold-weapon roll (bevy `random_gold_weapon` parity: uniform over
/// the weapons table's gold flag, plain roll when empty).
pub fn random_gold_weapon_fallback(rng: &mut impl rand::RngExt) -> WeaponId {
    let gold: Vec<WeaponId> = crate::weapons_data::WEAPONS
        .iter()
        .filter(|w| w.wep_gold)
        .map(|w| WeaponId(w.id))
        .collect();
    if gold.is_empty() {
        return random_weapon(rng);
    }
    gold[rng.random_range(0..gold.len())]
}

fn scrub_need(inv: &Inventory, player: &Player) -> f32 {
    let mut need = 0.0;
    for w in inv.weapons.iter().take(inv.weapon_slots) {
        if *w == WeaponId::NONE {
            continue;
        }
        let def = weapon_runtime_def(*w);
        if def.melee.is_some() {
            need += 0.5;
            continue;
        }
        let cap = player.ammo_cap(def.ammo) as f32;
        let am = inv_ammo(inv, def.ammo) as f32;
        if am < cap * 0.2 {
            need += 0.75;
        } else if am > cap * 0.6 {
            need += 0.1;
        } else {
            need += 0.5;
        }
    }
    need
}

/// Ammo count for a kind (used by need/give calculations).
pub fn inv_ammo(inv: &Inventory, kind: AmmoKind) -> i32 {
    inv.ammo_of(kind)
}

pub fn give_ammo(inv: &mut Inventory, player: &Player) {
    let id = inv.weapons[inv.current];
    if id == WeaponId::NONE {
        let mut rng = rand::rng();
        let kind = random_ammo_kind(&mut rng);
        let slot = inv.ammo_mut(kind);
        let add = ammo_pickup_amount(kind);
        *slot = (*slot + add).min(player.ammo_cap(kind));
        return;
    }
    let def = weapon_runtime_def(id);
    if def.melee.is_some() {
        return;
    }
    let slot = inv.ammo_mut(def.ammo);
    let add = ammo_pickup_amount(def.ammo);
    *slot = (*slot + add).min(player.ammo_cap(def.ammo));
}

/// Toast expiry (bevy `pickups.rs:999` parity: duration-zero timers are
/// inert, otherwise the text clears when the 2.2 s timer lapses).
/// Text-only effect; kept so run-setup crown toasts fade headless.
pub fn tick_toast(time: Res<repame_sim::SimTime>, mut toast: ResMut<Toast>) {
    if toast.timer.duration() <= 0.0 {
        return;
    }
    toast.timer.tick(time.delta_secs);
    if toast.timer.is_finished() {
        toast.text.clear();
    }
}

// ---------------------------------------------------------------------------
// Pickup tick battery. Ported from nt's `game/pickups.rs` Progression-set
// systems (`tick_pickup_drag`, `collect_pickups`, `sync_weapon_label`,
// `tick_rad_container_contact`) plus the chest/ammo/weapon grant helpers.
// Render split: `Visibility` blink-out and sprite alpha fades are
// renderer-owned (skipped); the sim keeps lifetimes, motion, grants.
// ---------------------------------------------------------------------------

/// Loose-pickup drift: weapons near the portal get carried through,
/// rads slide toward the player once a portal exists, ammo/medkits
/// drift at the GML rate inside pickup range (mask-gated per axis).
pub fn tick_pickup_drag(
    time: Res<SimTime>,
    mut commands: Commands,
    portal_q: Query<&Pos, (With<Portal>, Without<Pickup>)>,
    mut carried: ResMut<PortalCarriedWeapons>,
    player_q: Query<(&Pos, &Player), With<Player>>,
    mask: Res<FloorMask>,
    mut pickups: Query<(Entity, &mut Pos, &Pickup), (Without<Player>, Without<Portal>)>,
) {
    let Ok((player_pos, player)) = player_q.single() else {
        return;
    };
    let player_pos = player_pos.0;
    let portal_pos = portal_q.single().ok().map(|p| p.0);
    let dt = time.delta_secs;
    let hunger = player.mutations.contains(&MutationId::PlutoniumHunger);

    let loose_range = 32.0 + if hunger { 64.0 } else { 0.0 };

    for (e, mut pos, pickup) in &mut pickups {
        let ppos = pos.0;
        match pickup.kind {
            PickupKind::Weapon(w) => {
                if portal_pos.is_some_and(|pp| ppos.distance(pp) < 20.0) {
                    carried.0.push(w);
                    commands.entity(e).try_despawn();
                }
            }
            PickupKind::Rad(_) => {
                if portal_pos.is_none() {
                    continue;
                }

                let dir = (player_pos - ppos).normalize_or_zero();
                pos.0 += dir * 360.0 * dt;

                if ppos.distance(portal_pos.unwrap_or(player_pos)) < 20.0 {
                    pos.0 = player_pos;
                }
            }
            PickupKind::Ammo(..) | PickupKind::Medkit(_) => {
                let in_range = ppos.distance(player_pos) < loose_range;
                if !in_range && portal_pos.is_none() {
                    continue;
                }

                let dir = (player_pos - ppos).normalize_or_zero();
                let delta = dir * 6.0 * 30.0 * dt;
                let nx = glam::Vec2::new(ppos.x + delta.x, ppos.y);
                if mask.is_walkable(nx) {
                    pos.0.x = nx.x;
                }
                let ny = glam::Vec2::new(pos.0.x, ppos.y + delta.y);
                if mask.is_walkable(ny) {
                    pos.0.y = ny.y;
                }

                if portal_pos.is_some_and(|pp| ppos.distance(pp) < 14.0) {
                    pos.0 = player_pos;
                }
            }
            _ => {}
        }
    }
}

/// Pickup collection: ground-physics slide, lifetime expiry, telekinesis
/// magnet, per-kind ranges, chest loot tables, rad/ammo/medkit/weapon
/// grants. Bevy `collect_pickups` parity minus render blinks.
#[allow(clippy::too_many_arguments)]
pub fn collect_pickups(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut trauma: ResMut<Trauma>,
    mut flash: ResMut<FlashWhite>,
    mut chroma: ResMut<ChromaticAberration>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut rumble_queue: ResMut<Queue<RumbleRequest>>,
    mut input: ResMut<NtInput>,
    mut run: ResMut<Run>,
    mut player_q: Query<
        (
            Entity,
            &Pos,
            &mut Player,
            &mut Health,
            &mut Inventory,
            Option<&Telekinesis>,
            Option<&RaceState>,
        ),
        (With<Player>, Without<Pickup>),
    >,
    mut pickups: Query<
        (
            Entity,
            &mut Pos,
            &Pickup,
            Option<&mut GroundPhysics>,
            Option<&mut PickupLifetime>,
            Option<&WepPickupAmmo>,
            Option<&PickupCurse>,
        ),
        Without<Player>,
    >,
    mut toast: ResMut<Toast>,
) {
    let Ok((_player_e, player_pos, mut player, mut health, mut inv, telek, race_opt)) =
        player_q.single_mut()
    else {
        return;
    };

    let player_pos = player_pos.0;
    let dt = time.delta_secs;
    let interact_pressed = input.take_interact_pressed();

    let telek_active = telek.is_some_and(|t| !t.timer.is_finished());
    let telek_mult = if telek_active {
        player.ultra_ability_mult
    } else {
        1.0
    };
    let magnet = if telek_active {
        player.pickup_range + 500.0 * telek_mult
    } else {
        player.pickup_range
    };

    let mut nearest_weapon: Option<(Entity, f32)> = None;
    for (e, pos, pickup, _, _, _, _) in pickups.iter() {
        if matches!(pickup.kind, PickupKind::Weapon(_)) {
            let d = player_pos.distance(pos.0);
            if d < 28.0 && nearest_weapon.is_none_or(|(_, bd)| d < bd) {
                nearest_weapon = Some((e, d));
            }
        }
    }

    for (pickup_e, mut pickup_pos, pickup, ground, lifetime, wep_ammo, pickup_curse) in &mut pickups {
        let pickup_pos_value = pickup_pos.0;
        let dist = player_pos.distance(pickup_pos_value);

        if let Some(mut gp) = ground {
            let speed = gp.vel.length();
            if speed > 0.5 {
                pickup_pos.0 += gp.vel * dt;
                gp.vel *= 0.4_f32.powf(dt * crate::SIM_HZ as f32);
            } else {
                gp.vel = glam::Vec2::ZERO;
            }
        }
        // Render split: bevy spun the sprite (`rotate_z`) while sliding;
        // headless pickups carry no angle channel, velocity decay above
        // is the sim effect.

        if let Some(mut lt) = lifetime {
            lt.timer.tick(time.delta_secs);
            if lt.timer.just_finished() {
                audio.play_pickup_disappear(&mut cues);
                commands.entity(pickup_e).try_despawn();
                continue;
            }
            // Render split: ammo/hp blink-out and sub-second alpha fade
            // are renderer-owned (skipped); expiry above is the sim law.
        }

        let is_chest = matches!(pickup.kind, PickupKind::Chest(_));
        let is_weapon = matches!(pickup.kind, PickupKind::Weapon(_));
        let is_rad = matches!(pickup.kind, PickupKind::Rad(_));
        let is_ammo = matches!(pickup.kind, PickupKind::Ammo(..));
        let is_medkit = matches!(pickup.kind, PickupKind::Medkit(_));
        if is_weapon {
            if telek_active && dist < magnet {
                let dir = (player_pos - pickup_pos_value).normalize_or_zero();
                pickup_pos.0 += dir * 900.0 * telek_mult * dt;
            }
        } else if is_ammo || is_medkit {
        } else if is_rad {
            let has_hunger = player.mutations.contains(&MutationId::PlutoniumHunger);
            let rad_range = 80.0 + if has_hunger { 60.0 } else { 0.0 };
            let magnet_to_player = dist < rad_range || (telek_active && dist < magnet);
            if magnet_to_player {
                let dir = (player_pos - pickup_pos_value).normalize_or_zero();

                let pull = if telek_active {
                    900.0 * telek_mult
                } else {
                    360.0
                };
                pickup_pos.0 += dir * pull * dt;
            }
        } else if !is_chest && dist < magnet {
            let dir = (player_pos - pickup_pos_value).normalize_or_zero();
            let pull = if telek_active {
                900.0 * telek_mult
            } else {
                460.0
            };
            pickup_pos.0 += dir * pull * dt;
        }

        if is_weapon {
            if dist > 28.0 {
                continue;
            }
            if nearest_weapon.is_none_or(|(e, _)| e != pickup_e) {
                continue;
            }
            if !interact_pressed {
                continue;
            }
        } else if is_ammo || is_medkit {
            if dist > 14.0 {
                continue;
            }
        } else if dist > 20.0 {
            continue;
        }

        if let PickupKind::Chest(chest) = pickup.kind {
            open_chest(&mut commands, pickup_e, chest);
            match chest {
                ChestKind::Weapon => {
                    // GML `WeaponChest/Create_0`: with a crown carried,
                    // `random(7) <= (Curses ? 4 : 1)` curses the cache
                    // (`_extra = 1 + curse * 2` tier, all drops cursed).
                    let curse_roll = rand::rng().random_range(0.0..7.0)
                        <= if player.crown == CrownKind::Curses {
                            4.0
                        } else {
                            1.0
                        };
                    let cursed =
                        player.crown != CrownKind::None && curse_roll;
                    let mut rng = rand::rng();
                    let count =
                        if matches!(player.ultra, Some(UltraMutationId::SteroidsAmbidextrous)) {
                            2
                        } else {
                            1
                        };
                    for _ in 0..count {
                        let weapon = random_weapon(&mut rng);
                        let e = spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Weapon(weapon),
                            pickup_pos_value
                                + glam::Vec2::new(
                                    rng.random_range(-2.0..2.0),
                                    rng.random_range(-2.0..2.0),
                                ),
                            0,
                            false,
                        );
                        if cursed {
                            commands.entity(e).insert(PickupCurse);
                        }
                        toast.show(weapon_id_name(weapon));
                    }
                    if cursed {
                        audio.play_cursed_chest(&mut cues);
                    } else {
                        audio.play_weapon_chest(&mut cues);
                    }
                }
                ChestKind::Ammo => {
                    let ammo = decide_ammo_type(&inv);
                    let amount = ammo_pickup_amount(ammo) * 2;
                    let cap = player.ammo_cap(ammo);
                    let slot = inv.ammo_mut(ammo);
                    let gained = (*slot + amount).min(cap) - *slot;
                    *slot += gained;
                    repame_fx::spawn_number(
                        &mut commands,
                        player_pos.x,
                        player_pos.y,
                        gained.to_string(),
                        [0.35, 0.7, 1.0, 1.0],
                    );
                    audio.play_ammo_chest(&mut cues);
                    toast.show("Ammo refilled");
                }
                ChestKind::Rad => {
                    // GML `RadChest/Collision_Player`: opening resets
                    // `noradch`.
                    run.noradch = 0;
                    let mut rng = rand::rng();
                    for _ in 0..25 {
                        let ang = rng.random_range(0.0..std::f32::consts::TAU);
                        let d = rng.random_range(6.0..26.0);
                        spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Rad(1),
                            pickup_pos_value
                                + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                            0,
                            false,
                        );
                    }
                    audio.play_pickup(&mut cues);
                }
                ChestKind::Health => {
                    // GML `HealthChest`: banked head-loss pays back one
                    // max-HP first, then heals `num` (4, 8 with Second
                    // Stomach) with popup.
                    if player.headloses > 0 {
                        player.headloses -= 1;
                        health.max += 1;
                    }
                    let big = player
                        .mutations
                        .contains(&MutationId::SecondStomach);
                    let num = if big { 8 } else { 4 };
                    health.hp = (health.hp + num).min(health.max);
                    repame_fx::spawn_number(
                        &mut commands,
                        player_pos.x,
                        player_pos.y,
                        num.to_string(),
                        [0.3, 1.0, 0.3, 1.0],
                    );
                    audio.play_health_chest(&mut cues, big);
                    toast.show("Healed");
                }
                ChestKind::CursedBig => {
                    // GML `CursedBigChest`: 3 weapons (4 with Steroids
                    // Ambidextrous), cursed, scattered ±2 px, plus a
                    // `PortalClear`.
                    let mut rng = rand::rng();
                    let count =
                        if matches!(player.ultra, Some(UltraMutationId::SteroidsAmbidextrous)) {
                            4
                        } else {
                            3
                        };
                    for _ in 0..count {
                        let weapon = random_weapon(&mut rng);
                        let e = spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Weapon(weapon),
                            pickup_pos_value
                                + glam::Vec2::new(
                                    rng.random_range(-2.0..2.0),
                                    rng.random_range(-2.0..2.0),
                                ),
                            0,
                            false,
                        );
                        commands.entity(e).insert(PickupCurse);
                        toast.show(weapon_id_name(weapon));
                    }
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        PortalClear {
                            timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                        },
                        Pos(pickup_pos_value),
                    ));
                    audio.play_cursed_chest(&mut cues);
                }
                ChestKind::Rogue => {
                    // GML `RogueChest`: 25 rads for non-Rogue, a
                    // `RogueAmmo` refill for Rogue.
                    let is_rogue = race_opt.is_some_and(|r| {
                        r.race == crate::data::RaceId::Rogue
                    });
                    if is_rogue {
                        player.rogue_ammo = player.rogue_ammo_max;
                        toast.show("Rogue ammo");
                    } else {
                        let mut rng = rand::rng();
                        for _ in 0..25 {
                            let ang = rng.random_range(0.0..std::f32::consts::TAU);
                            let d = rng.random_range(6.0..26.0);
                            spawn_pickup(
                                &mut commands,
                                &catalog,
                                PickupKind::Rad(1),
                                pickup_pos_value
                                    + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                                0,
                                false,
                            );
                        }
                    }
                    audio.play_pickup(&mut cues);
                }
                ChestKind::Proto => {
                    // GML `ProtoChest`: the vault prototype weapon (port
                    // has no vault-proto run artifact, so a random
                    // weapon); Hatred burns 1 HP and bursts 16 rads.
                    let weapon = random_weapon(&mut rand::rng());
                    spawn_pickup(
                        &mut commands,
                        &catalog,
                        PickupKind::Weapon(weapon),
                        pickup_pos_value,
                        0,
                        false,
                    );
                    if player.crown == CrownKind::Hatred && health.hp > 1 {
                        health.hp -= 1;
                        let mut rng = rand::rng();
                        for _ in 0..16 {
                            let ang = rng.random_range(0.0..std::f32::consts::TAU);
                            let d = rng.random_range(6.0..26.0);
                            spawn_pickup(
                                &mut commands,
                                &catalog,
                                PickupKind::Rad(1),
                                pickup_pos_value
                                    + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                                0,
                                false,
                            );
                        }
                    }
                    audio.play_weapon_chest(&mut cues);
                    toast.show(weapon_id_name(weapon));
                }
                ChestKind::BigWeapon => {
                    // GML `BigWeaponChest`: 3 weapons (4 Ambidextrous)
                    // scattered ±2 px plus a `PortalClear`; resets
                    // `nochest`.
                    let mut rng = rand::rng();
                    let count =
                        if matches!(player.ultra, Some(UltraMutationId::SteroidsAmbidextrous)) {
                            4
                        } else {
                            3
                        };
                    for _ in 0..count {
                        let weapon = random_weapon(&mut rng);
                        spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Weapon(weapon),
                            pickup_pos_value
                                + glam::Vec2::new(
                                    rng.random_range(-2.0..2.0),
                                    rng.random_range(-2.0..2.0),
                                ),
                            0,
                            false,
                        );
                        toast.show(weapon_id_name(weapon));
                    }
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        PortalClear {
                            timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                        },
                        Pos(pickup_pos_value),
                    ));
                    run.nochest = 0;
                    audio.play_weapon_chest(&mut cues);
                }
                ChestKind::RadBig => {
                    // GML `RadChestBig` (20 HP cache): 45 rads.
                    let mut rng = rand::rng();
                    for _ in 0..45 {
                        let ang = rng.random_range(0.0..std::f32::consts::TAU);
                        let d = rng.random_range(6.0..26.0);
                        spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Rad(1),
                            pickup_pos_value
                                + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                            0,
                            false,
                        );
                    }
                    audio.play_pickup(&mut cues);
                }
                ChestKind::RadMaggot => {
                    // GML `RadMaggotChest`: the trapped cache detonates
                    // when opened.
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Explosion {
                            timer: GTimer::from_seconds(0.05, TimerMode::Once),
                            radius: 70.0,
                            damage: 6,
                            team: Team::Enemy,
                            hits_player: true,
                            source: None,
                        },
                        Pos(pickup_pos_value),
                    ));
                    trauma.add(0.3);
                    audio.play_pickup(&mut cues);
                }
                ChestKind::Idpd => {
                    // GML `IDPDChest`: 8 ammo pickups on the opener.
                    let ammo = decide_ammo_type(&inv);
                    let amount = ammo_pickup_amount(ammo);
                    let mut rng = rand::rng();
                    for _ in 0..8 {
                        let ang = rng.random_range(0.0..std::f32::consts::TAU);
                        let d = rng.random_range(4.0..18.0);
                        spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Ammo(ammo, amount),
                            pickup_pos_value
                                + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                            0,
                            false,
                        );
                    }
                    audio.play_ammo_chest(&mut cues);
                    toast.show("Ammo refilled");
                }
            }
            trauma.add(0.15);
            rumble(&mut rumble_queue, 0.3, 0.4, 0.15);
            if player.crown == CrownKind::Hatred && health.hp > 1 {
                health.hp -= 1;
                let n = if matches!(chest, ChestKind::Rad) {
                    24
                } else {
                    16
                };
                let mut rng = rand::rng();
                for _ in 0..n {
                    let ang = rng.random_range(0.0..std::f32::consts::TAU);
                    let d = rng.random_range(6.0..26.0);
                    spawn_pickup(
                        &mut commands,
                        &catalog,
                        PickupKind::Rad(1),
                        pickup_pos_value + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                        0,
                        false,
                    );
                }
            }
            continue;
        }

        commands.entity(pickup_e).try_despawn();

        match pickup.kind {
            PickupKind::Rad(amount) => {
                player.rads += amount;
                chromatic_pulse(&mut chroma, 0.04);
                audio.play_pickup(&mut cues);
                check_level_up(
                    &mut commands,
                    &mut trauma,
                    &mut flash,
                    &mut player,
                    &mut toast,
                    &audio,
                    &mut cues,
                    player_pos,
                );
            }
            PickupKind::Medkit(amount) => {
                let heal = (amount as f32 * player.medkit_mult).round() as i32;
                health.hp = (health.hp + heal).min(health.max);
                try_recharge_strong_spirit(&mut player, &health);
                repame_fx::spawn_number(
                    &mut commands,
                    player_pos.x,
                    player_pos.y,
                    heal.to_string(),
                    [0.3, 1.0, 0.3, 1.0],
                );
                audio.play_pickup(&mut cues);
            }
            PickupKind::Ammo(..) => {
                let ammo = decide_ammo_type(&inv);
                let mut amount = ammo_pickup_amount(ammo);
                if player.crown == CrownKind::Haste {
                    amount += 1;
                }
                // GML `AmmoPickup`: with 2+ cursed guns held, half the
                // pickups convert to `CursedPickup` at 1.5x ammo.
                let held_cursed =
                    inv.cursed.iter().filter(|c| **c).count();
                if held_cursed >= 2 && rand::rng().random::<f32>() < 0.5 {
                    amount += amount / 2;
                }
                let fish_bonus = if player.ability == AbilityKind::Flip {
                    match ammo {
                        AmmoKind::None => 0,
                        AmmoKind::Bullets => 8,
                        AmmoKind::Shells
                        | AmmoKind::Bolts
                        | AmmoKind::Explosives
                        | AmmoKind::Energy => 2,
                    }
                } else {
                    0
                };
                let cap = player.ammo_cap(ammo);
                let slot = inv.ammo_mut(ammo);
                let gained = (amount + fish_bonus).min(cap - *slot).max(0);
                *slot += gained;

                if player.free_ammo && gained > 0 {
                    let heal = match player.ultra {
                        Some(
                            UltraMutationId::RobotRefinedTaste | UltraMutationId::RobotRegurgitate,
                        ) => 2,
                        _ => 1,
                    };
                    health.hp = (health.hp + heal).min(health.max);
                    try_recharge_strong_spirit(&mut player, &health);
                    repame_fx::spawn_number(
                        &mut commands,
                        player_pos.x,
                        player_pos.y,
                        heal.to_string(),
                        [0.55, 0.85, 0.95, 1.0],
                    );
                }

                repame_fx::spawn_number(
                    &mut commands,
                    player_pos.x,
                    player_pos.y,
                    gained.to_string(),
                    [0.35, 0.7, 1.0, 1.0],
                );

                let type_name = ammo_type_name(ammo);
                if *slot >= cap {
                    toast.show(&format!("MAX {type_name}"));
                } else {
                    toast.show(&format!("+{gained} {type_name}"));
                }
                audio.play_pickup(&mut cues);
            }
            PickupKind::Weapon(weapon) => {
                commands.spawn((
                    GameCleanup,
                    QueuedReactiveCue(ReactiveCue::WeaponPickup),
                ));

                // GML `Player/Collision_WepPickup`: a cursed held gun
                // cannot be swapped for an uncursed one unless a free
                // slot (or invalid bwep) takes it.
                let held_cursed = inv.cursed[inv.current.min(MAX_WEAPON_SLOTS - 1)];
                let pickup_cursed = pickup_curse.is_some();
                let free_slot = first_empty_weapon_slot(&inv).is_some();
                if held_cursed && !pickup_cursed && !free_slot {
                    toast.show("Cursed");
                    // Re-drop: the blanket despawn above already consumed
                    // the entity, but GML leaves the gun on the ground.
                    let e2 = spawn_pickup(
                        &mut commands,
                        &catalog,
                        PickupKind::Weapon(weapon),
                        pickup_pos_value,
                        0,
                        false,
                    );
                    commands
                        .entity(e2)
                        .insert(WepPickupAmmo(wep_ammo.is_some_and(|f| f.0)));
                    continue;
                }

                let has_ammo = wep_ammo.is_some_and(|f| f.0);
                equip_weapon(
                    &mut commands,
                    &catalog,
                    &mut inv,
                    weapon,
                    player_pos,
                    &player,
                    &mut health,
                    has_ammo,
                    pickup_cursed,
                );

                if matches!(player.ultra, Some(UltraMutationId::FishConfiscate)) {
                    let kind = weapon_ammo(weapon);
                    if kind != AmmoKind::None {
                        let add = ammo_pickup_amount(kind) * 2;
                        let slot = inv.ammo_mut(kind);
                        *slot = (*slot + add).min(player.ammo_cap(kind));
                        repame_fx::spawn_number(
                            &mut commands,
                            player_pos.x,
                            player_pos.y,
                            add.to_string(),
                            [0.9, 0.82, 0.25, 1.0],
                        );
                    }
                }

                if matches!(player.ultra, Some(UltraMutationId::RobotRefinedTaste)) {
                    health.hp = (health.hp + 1).min(health.max);
                }

                // GML `scrPlayerUpdateSameWeaponsFor`: a new weapon resets
                // the mimic clock.
                run.same_weapons_for = 0;
                // GML `GameCont.haspickedweps` (Steroids-C gate).
                run.weapons_picked += 1;

                audio.play_chest(&mut cues);
                toast.show(&format!("Picked up {}", weapon_id_name(weapon)));
            }
            PickupKind::Chest(_) => {}
            // GML `Curse` motes are ambient (no `Collision_Player`):
            // touching one just clears it.
            PickupKind::Curse => {}
        }
    }
}

/// Chest-open state flip (sim half): the pickup becomes an opened chest
/// carrying its kind, so the renderer can swap to bevy's kind-specific
/// open art (`open_chest` sprite swap, renderer-owned).
pub fn open_chest(commands: &mut Commands, e: Entity, kind: ChestKind) {
    commands.entity(e).remove::<Pickup>();
    commands.entity(e).insert(OpenedChest(kind));
}

/// Chest-open entry used by gameplay callers (mines, scripted opens).
/// Gameplay half: currently the same state flip as [`open_chest`].
pub fn open_chest_shock(commands: &mut Commands, e: Entity, kind: ChestKind) {
    open_chest(commands, e, kind);
}

/// Headless weapon-label state: the nearest in-range weapon name for the
/// HUD bridge to poll. Bevy `sync_weapon_label` spawned Text2d name /
/// gauge / prompt parts; sprites and gauge frames are renderer-owned
/// (deferred), the name string is the sim state.
#[derive(Resource, Default, Debug)]
pub struct WeaponLabel {
    pub text: String,
    pub target: Option<Entity>,
}

pub fn sync_weapon_label(
    player_q: Query<&Pos, With<Player>>,
    weapon_q: Query<(Entity, &Pos, &Pickup), (Without<Player>, Without<Portal>)>,
    mut label: ResMut<WeaponLabel>,
) {
    let Some(player_pos) = player_q.single().ok().map(|p| p.0) else {
        return;
    };

    let mut best: Option<(Entity, WeaponId)> = None;
    let mut best_dist = 18.0;
    for (e, pos, pickup) in &weapon_q {
        let PickupKind::Weapon(w) = pickup.kind else {
            continue;
        };
        let d = player_pos.distance(pos.0);
        if d < best_dist {
            best_dist = d;
            best = Some((e, w));
        }
    }

    if best.map(|(e, _)| e) != label.target {
        label.target = best.map(|(e, _)| e);
        label.text = best
            .map(|(_, w)| weapon_id_name(w).to_string())
            .unwrap_or_default();
    }
}

pub fn ammo_type_name(kind: AmmoKind) -> &'static str {
    match kind {
        AmmoKind::None => "NONE",
        AmmoKind::Bullets => "BULLETS",
        AmmoKind::Shells => "SHELLS",
        AmmoKind::Bolts => "BOLTS",
        AmmoKind::Explosives => "EXPLOSIVES",
        AmmoKind::Energy => "ENERGY",
    }
}

fn decide_ammo_type(inv: &Inventory) -> AmmoKind {
    let types = [
        weapon_ammo(inv.weapons[inv.current]),
        weapon_ammo(inv.weapons[1.min(inv.weapon_slots.saturating_sub(1))]),
    ];
    for ty in types {
        if ty != AmmoKind::None && inv.ammo_of(ty) < ammo_max(ty) {
            return ty;
        }
    }
    match rand::rng().random_range(1..=5) {
        1 => AmmoKind::Bullets,
        2 => AmmoKind::Shells,
        3 => AmmoKind::Bolts,
        4 => AmmoKind::Explosives,
        _ => AmmoKind::Energy,
    }
}

fn first_empty_weapon_slot(inv: &Inventory) -> Option<usize> {
    (0..inv.weapon_slots).find(|&i| inv.weapons[i] == WeaponId::NONE)
}

fn spawn_dropped_weapon(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    weapon: WeaponId,
    pos: glam::Vec2,
    curse: bool,
) {
    let e = spawn_pickup(
        commands,
        catalog,
        PickupKind::Weapon(weapon),
        pos + glam::Vec2::new(0.0, 24.0),
        0,
        false,
    );
    commands.entity(e).insert(WepPickupAmmo(false));
    if curse {
        commands.entity(e).insert(PickupCurse);
    }
}

#[allow(clippy::too_many_arguments)]
fn equip_weapon(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    inv: &mut Inventory,
    weapon: WeaponId,
    player_pos: glam::Vec2,
    player: &Player,
    health: &mut Health,
    has_ammo: bool,
    curse: bool,
) {
    if let Some(empty) = first_empty_weapon_slot(inv) {
        inv.weapons[empty] = weapon;
        inv.cursed[empty] = curse;
        inv.current = empty;
        grant_pickup_ammo(commands, inv, weapon, player_pos, player, health, has_ammo);
        return;
    }

    let slot = inv.current;
    let dropped = inv.weapons[slot];
    let dropped_curse = inv.cursed[slot];
    if dropped != WeaponId::NONE {
        spawn_dropped_weapon(commands, catalog, dropped, player_pos, dropped_curse);
    }
    inv.weapons[slot] = weapon;
    inv.cursed[slot] = curse;

    grant_pickup_ammo(commands, inv, weapon, player_pos, player, health, has_ammo);
}

#[allow(clippy::too_many_arguments)]
fn grant_pickup_ammo(
    commands: &mut Commands,
    inv: &mut Inventory,
    weapon: WeaponId,
    player_pos: glam::Vec2,
    player: &Player,
    health: &mut Health,
    has_ammo: bool,
) {
    let def = weapon_runtime_def(weapon);
    let second_stomach = player.mutations.contains(&MutationId::SecondStomach);
    match weapon_pickup_grant(has_ammo, def.melee.is_some(), player.crown, second_stomach) {
        WeaponPickupGrant::Nothing => {}
        WeaponPickupGrant::Heal(heal) => {
            health.hp = (health.hp + heal).min(health.max);
            repame_fx::spawn_number(
                commands,
                player_pos.x,
                player_pos.y,
                heal.to_string(),
                [0.3, 1.0, 0.3, 1.0],
            );
        }
        WeaponPickupGrant::Ammo => {
            let slot = inv.ammo_mut(def.ammo);
            let add = ammo_pickup_amount(def.ammo) * 2;
            let gained = add.min(player.ammo_cap(def.ammo) - *slot).max(0);
            *slot += gained;
            repame_fx::spawn_number(
                commands,
                player_pos.x,
                player_pos.y,
                gained.to_string(),
                [0.35, 0.7, 1.0, 1.0],
            );
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum WeaponPickupGrant {
    Nothing,
    Heal(i32),
    Ammo,
}

fn weapon_pickup_grant(
    has_ammo: bool,
    melee: bool,
    crown: CrownKind,
    second_stomach: bool,
) -> WeaponPickupGrant {
    if !has_ammo || melee {
        return WeaponPickupGrant::Nothing;
    }
    if crown == CrownKind::Protection {
        return WeaponPickupGrant::Heal(1 + i32::from(second_stomach));
    }
    WeaponPickupGrant::Ammo
}

/// Rad-container contact: touching the rad chest prop bursts 25 rads.
/// Bevy `tick_rad_container_contact` parity minus the debris burst art
/// (kept as a sim burst: the renderer draws `Particle`s already).
pub fn tick_rad_container_contact(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    player_q: Query<&Pos, With<Player>>,
    mut rad_q: Query<(Entity, &Pos, &Prop), With<RadChestContainer>>,
) {
    let Ok(player_pos) = player_q.single() else {
        return;
    };
    let player_pos = player_pos.0;
    for (e, pos, prop) in &mut rad_q {
        let center = pos.0;
        let half = prop.size * 0.5;

        let closest = glam::Vec2::new(
            player_pos.x.clamp(center.x - half.x, center.x + half.x),
            player_pos.y.clamp(center.y - half.y, center.y + half.y),
        );
        if player_pos.distance(closest) > crate::comps_a::PLAYER_RADIUS + 2.0 {
            if player_pos.distance(center) > half.x + crate::comps_a::PLAYER_RADIUS + 4.0 {
                continue;
            }
        }

        commands.entity(e).try_despawn();
        let mut rng = rand::rng();
        for _ in 0..25 {
            let ang = rng.random_range(0.0..std::f32::consts::TAU);
            let d = rng.random_range(6.0..26.0);
            spawn_pickup(
                &mut commands,
                &catalog,
                PickupKind::Rad(1),
                center + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                0,
                false,
            );
        }

        audio.play_boom(&mut cues);

        spawn_burst(
            &mut commands,
            &mut rng,
            center,
            8,
            [0.55, 0.55, 0.60, 1.0],
            (40.0, 120.0),
        );
    }
}

#[cfg(test)]
mod weapon_pickup_grant_tests {
    use super::*;

    #[test]
    fn dry_swap_drops_grant_nothing() {
        assert_eq!(
            weapon_pickup_grant(false, false, CrownKind::None, false),
            WeaponPickupGrant::Nothing
        );
    }

    #[test]
    fn melee_never_grants_ammo() {
        assert_eq!(
            weapon_pickup_grant(true, true, CrownKind::None, false),
            WeaponPickupGrant::Nothing
        );
    }

    #[test]
    fn fresh_ranged_drop_grants_ammo() {
        assert_eq!(
            weapon_pickup_grant(true, false, CrownKind::None, false),
            WeaponPickupGrant::Ammo
        );
    }

    #[test]
    fn protection_crown_heals_instead() {
        assert_eq!(
            weapon_pickup_grant(true, false, CrownKind::Protection, false),
            WeaponPickupGrant::Heal(1)
        );
        assert_eq!(
            weapon_pickup_grant(true, false, CrownKind::Protection, true),
            WeaponPickupGrant::Heal(2)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repame_anim::{AnimCatalog, AtlasDesc};

    const PICKUP_JSON: &str = r#"{
        "images/sprRad.png": {"frames": 4, "w": 12, "h": 12, "fps": 12.0, "xorigin": 6.0, "yorigin": 6.0},
        "images/sprHP.png": {"frames": 1, "w": 16, "h": 16, "fps": 1.0, "xorigin": 8.0, "yorigin": 8.0}
    }"#;

    fn pickup_catalog() -> AnimCatalog {
        AnimCatalog::from_json(
            PICKUP_JSON,
            AtlasDesc {
                size: 128,
                max_pages: 1,
            padding: 0,
            },
        )
        .unwrap()
    }

    #[test]
    fn rad_burst_splits_denomination() {
        let mut world = bevy_ecs::prelude::World::new();
        let catalog = pickup_catalog();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(move |mut commands: Commands| {
            spawn_rad_burst(&mut commands, &catalog, glam::Vec2::ZERO, 25);
        });
        schedule.run(&mut world);
        // 25 -> one 10 (25 > 15: 25-10=15, not > 15 stop) ... recount:
        // amount=25: 25>15 -> spawn 10, amount=15. 15>15 false. then 15 ones.
        // Total rads = 10 + 15 = 25 across 16 pickups.
        let mut q = world.query::<&Pickup>();
        assert_eq!(q.iter(&world).count(), 16);
        let total: u32 = q
            .iter(&world)
            .map(|p| match p.kind {
                PickupKind::Rad(n) => n,
                _ => 0,
            })
            .sum();
        assert_eq!(total, 25);
    }

    #[test]
    fn rad_gets_anim_medkit_gets_lifetime() {
        let mut world = bevy_ecs::prelude::World::new();
        let catalog = pickup_catalog();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(move |mut commands: Commands| {
            spawn_pickup(
                &mut commands,
                &catalog,
                PickupKind::Rad(5),
                glam::Vec2::ZERO,
                0,
                false,
            );
            spawn_pickup(
                &mut commands,
                &catalog,
                PickupKind::Medkit(2),
                glam::Vec2::ZERO,
                0,
                false,
            );
        });
        schedule.run(&mut world);
        let mut anims = world.query::<&crate::anim::SpriteAnim>();
        assert_eq!(anims.iter(&world).count(), 1, "only rad animates");
        let mut lives = world.query::<&crate::comps_b::PickupLifetime>();
        assert_eq!(lives.iter(&world).count(), 2);
    }

    #[test]
    fn flung_weapon_returns_on_alarm() {
        // GML `WepPickup/Alarm_1`: the Determination return alarm flings
        // the pickup back at its creator (`sndChickenReturn`).
        let mut world = bevy_ecs::prelude::World::new();
        world.insert_resource(Queue::<AudioCue>::default());
        let creator = world.spawn(Pos(glam::Vec2::new(100.0, 0.0))).id();
        let catalog = pickup_catalog();
        let mut spawn_sched = bevy_ecs::schedule::Schedule::default();
        spawn_sched.add_systems(move |mut commands: Commands| {
            spawn_flung_weapon_pickup(
                &mut commands,
                &catalog,
                WeaponId::REVOLVER,
                glam::Vec2::ZERO,
                0.0,
                Team::Player,
                creator,
                1,
            );
        });
        spawn_sched.run(&mut world);
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_flung_weapons);
        sched.run(&mut world);
        let mut q = world.query::<(Entity, &FlungWeapon, &GroundPhysics)>();
        let (e, flung, ground) = q
            .iter(&world)
            .next()
            .map(|(e, f, g)| (e, f.return_ticks, g.vel))
            .expect("flung pickup");
        assert_eq!(flung, 0);
        assert!((ground.length() - 480.0).abs() < 1e-3, "return speed 16, got {ground:?}");
        assert!(ground.x > 0.0, "back at the creator, got {ground:?}");
        let _ = e;
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert!(cues.iter().any(|c| c.name == "sndChickenReturn"));
    }

    #[test]
    fn idpd_chest_destroy_spawns_open_and_six_spawns() {
        // GML `IDPDChest/Destroy_0`: `ChestOpen` visual + `FXChestOpen` +
        // 6 `IDPDSpawn`s; skipped during worldgen (`GenCont`).
        let mut world = bevy_ecs::prelude::World::new();
        let catalog = pickup_catalog();
        let mut gen_sched = bevy_ecs::schedule::Schedule::default();
        gen_sched.add_systems(move |mut commands: Commands| {
            // Return value is asserted via the world: nothing spawns.
            assert!(!idpd_chest_destroy(
                &mut commands,
                &catalog,
                glam::Vec2::ZERO,
                true,
                0
            ));
        });
        gen_sched.run(&mut world);
        assert_eq!(world.query::<&OpenedChest>().iter(&world).count(), 0);
        let catalog = pickup_catalog();
        let mut live_sched = bevy_ecs::schedule::Schedule::default();
        live_sched.add_systems(move |mut commands: Commands| {
            assert!(idpd_chest_destroy(
                &mut commands,
                &catalog,
                glam::Vec2::new(50.0, 0.0),
                false,
                0
            ));
        });
        live_sched.run(&mut world);
        assert_eq!(world.query::<&OpenedChest>().iter(&world).count(), 1);
        let kinds: Vec<_> = world
            .query::<&OpenedChest>()
            .iter(&world)
            .map(|c| c.0)
            .collect();
        assert!(kinds.iter().all(|k| *k == ChestKind::Idpd));
        assert_eq!(
            world.query::<&crate::comps_b::Enemy>().iter(&world).count(),
            6
        );
    }

    #[test]
    fn inv_ammo_reads_slots() {
        let mut inv = Inventory {
            weapons: [WeaponId::NONE, WeaponId::NONE, WeaponId::NONE],
            cursed: [false, false, false],
            swapanim: 0.0,
            shine: 0.0,
            wepflip: 1.0,
            bwepflip: 1.0,
            weapon_slots: 1,
            current: 0,
            ammo: [0, 11, 0, 0, 0, 0],
        };
        assert_eq!(inv_ammo(&inv, AmmoKind::Bullets), 11);
        *inv.ammo_mut(AmmoKind::Bullets) += ammo_pickup_amount(AmmoKind::Bullets);
        assert_eq!(inv_ammo(&inv, AmmoKind::Bullets), 43);
    }

    #[test]
    fn scrub_need_follows_ammo_state() {
        // Bevy law: empty slots skipped; melee 0.5; dry (<20% cap)
        // 0.75; full (>60%) 0.1; else 0.5.
        let inv = Inventory {
            weapons: [WeaponId::NONE, WeaponId::NONE, WeaponId::NONE],
            cursed: [false, false, false],
            swapanim: 0.0,
            shine: 0.0,
            wepflip: 1.0,
            bwepflip: 1.0,
            weapon_slots: 1,
            current: 0,
            ammo: [0, 0, 0, 0, 0, 0],
        };
        let player = Player::default();
        assert_eq!(scrub_need(&inv, &player), 0.0);
        let mut melee = inv.clone();
        melee.weapons[0] = WeaponId::WRENCH;
        assert_eq!(scrub_need(&melee, &player), 0.5);
        let mut dry = inv.clone();
        dry.weapons[0] = WeaponId::REVOLVER;
        assert_eq!(scrub_need(&dry, &player), 0.75);
    }

    #[test]
    fn toast_clears_on_lapse_and_ignores_zero_timer() {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = repame_sim::SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        world.init_resource::<Toast>();
        world.resource_mut::<Toast>().show("hi");
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_toast);
        sched.run(&mut world);
        assert_eq!(world.resource::<Toast>().text, "hi");
        for _ in 0..70 {
            sched.run(&mut world);
        }
        assert!(world.resource::<Toast>().text.is_empty());

        // Zero-duration timer is inert (never clears foreign text).
        *world.resource_mut::<Toast>() = Toast::default();
        world.resource_mut::<Toast>().text = "sticky".to_string();
        sched.run(&mut world);
        assert_eq!(world.resource::<Toast>().text, "sticky");
    }
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    use crate::comps_a::{FloorMask, RaceState};
    use crate::data::{RaceId, SkinLetter};

    const DT: f32 = 1.0 / 30.0;

    fn sim_world() -> bevy_ecs::prelude::World {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = SimTime::default();
        time.delta_secs = DT;
        world.insert_resource(time);
        world.init_resource::<Toast>();
        world.init_resource::<GameAudio>();
        world.init_resource::<Queue<AudioCue>>();
        world.init_resource::<Queue<RumbleRequest>>();
        world.init_resource::<Trauma>();
        world.init_resource::<FlashWhite>();
        world.init_resource::<ChromaticAberration>();
        world.init_resource::<NtInput>();
        world.init_resource::<PortalCarriedWeapons>();
        world.init_resource::<WeaponLabel>();
        world.init_resource::<Run>();
        // Walkable 5x5 tile block around the origin so drag motion is
        // mask-legal; the void beyond stays unwalkable.
        let mut mask = FloorMask::default();
        for cx in -3..=3 {
            for cy in -2..=2 {
                mask.cells.insert((cx, cy));
            }
        }
        world.insert_resource(mask);
        world.insert_resource(
            repame_anim::AnimCatalog::from_json(
                "{}",
                repame_anim::AtlasDesc {
                    size: 64,
                    max_pages: 1,
                padding: 0,
                },
            )
            .unwrap(),
        );
        world
    }

    fn spawn_player(world: &mut bevy_ecs::prelude::World) -> bevy_ecs::prelude::Entity {
        world
            .spawn((
                Player::default(),
                Pos(glam::Vec2::ZERO),
                Health {
                    hp: 10,
                    max: 20,
                    invuln: GTimer::disarmed(),
                },
                Inventory {
                    weapons: [WeaponId::NONE, WeaponId::NONE, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
                    weapon_slots: 2,
                    current: 0,
                    ammo: [0, 0, 0, 0, 0, 0],
                },
                RaceState {
                    race: RaceId::Fish,
                    skin: SkinLetter::A,
                },
            ))
            .id()
    }

    fn run(world: &mut bevy_ecs::prelude::World, ticks: usize) {
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(
            (
                tick_pickup_drag,
                collect_pickups,
                sync_weapon_label,
                tick_rad_container_contact,
            )
                .chain(),
        );
        for _ in 0..ticks {
            sched.run(world);
        }
    }

    #[test]
    fn ammo_drifts_at_gml_rate_not_vacuum() {
        let mut world = sim_world();
        spawn_player(&mut world);
        let near = world
            .spawn((
                Pickup {
                    kind: PickupKind::Ammo(AmmoKind::None, 0),
                },
                Pos(glam::Vec2::new(25.0, 0.0)),
            ))
            .id();
        let far = world
            .spawn((
                Pickup {
                    kind: PickupKind::Ammo(AmmoKind::None, 0),
                },
                Pos(glam::Vec2::new(200.0, 0.0)),
            ))
            .id();
        // Far pickup stays put (out of drag range, no portal). Drag only:
        // the chained collect would eat the near ammo once it closes in.
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_pickup_drag);
        sched.run(&mut world);
        let dist = |world: &bevy_ecs::prelude::World, e: Entity| {
            world
                .get::<Pos>(e)
                .map(|p| p.0.distance(glam::Vec2::ZERO))
                .unwrap_or(f32::MAX)
        };
        let d = dist(&world, near);
        assert!(d < 25.0 && d > 5.0, "ammo drift wrong: {d}");
        assert!(
            (dist(&world, far) - 200.0).abs() < 0.01,
            "out-of-range ammo must not move"
        );
    }

    #[test]
    fn medkit_heals_on_touch() {
        let mut world = sim_world();
        let player = spawn_player(&mut world);
        world.spawn((
            Pickup {
                kind: PickupKind::Medkit(2),
            },
            Pos(glam::Vec2::ZERO),
        ));
        run(&mut world, 1);
        assert_eq!(world.get::<Health>(player).unwrap().hp, 12);
        assert_eq!(world.query::<&Pickup>().iter(&world).count(), 0);
    }

    #[test]
    fn ammo_grants_held_type_with_toast() {
        let mut world = sim_world();
        let player = spawn_player(&mut world);
        world.get_mut::<Inventory>(player).unwrap().weapons[0] = WeaponId::REVOLVER;
        world.spawn((
            Pickup {
                kind: PickupKind::Ammo(AmmoKind::None, 0),
            },
            Pos(glam::Vec2::ZERO),
        ));
        run(&mut world, 1);
        let inv = world.get::<Inventory>(player).unwrap();
        // 32 pickup + 8 Fish Flip bonus (default Flip ability, bevy parity).
        assert_eq!(inv.ammo_of(AmmoKind::Bullets), 40);
        assert_eq!(world.resource::<Toast>().text, "+40 BULLETS");
    }

    #[test]
    fn cursed_held_blocks_uncursed_swap() {
        let mut world = sim_world();
        let player = spawn_player(&mut world);
        // Full slots with a cursed current gun.
        {
            let mut inv = world.get_mut::<Inventory>(player).unwrap();
            inv.weapons = [WeaponId::REVOLVER, WeaponId::SHOTGUN, WeaponId::NONE];
            inv.cursed = [true, false, false];
            inv.weapon_slots = 2;
            inv.current = 0;
        }
        world.spawn((
            Pickup {
                kind: PickupKind::Weapon(WeaponId::CROSSBOW),
            },
            GroundPhysics {
                vel: glam::Vec2::ZERO,
                rotspeed: 0.0,
            },
            WepPickupAmmo(true),
            Pos(glam::Vec2::new(10.0, 0.0)),
        ));
        world.resource_mut::<NtInput>().press_interact();
        run(&mut world, 1);
        let inv = world.get::<Inventory>(player).unwrap();
        assert_eq!(inv.weapons[0], WeaponId::REVOLVER, "cursed gun stays");
        assert_eq!(
            world.query::<&Pickup>().iter(&world).count(),
            1,
            "uncursed gun left on the ground"
        );
        assert!(
            world.resource::<Toast>().text.contains("Cursed"),
            "reminder toasts"
        );
    }

    #[test]
    fn cursed_pickup_swaps_onto_cursed_held() {
        let mut world = sim_world();
        let player = spawn_player(&mut world);
        {
            let mut inv = world.get_mut::<Inventory>(player).unwrap();
            inv.weapons = [WeaponId::REVOLVER, WeaponId::SHOTGUN, WeaponId::NONE];
            inv.cursed = [true, false, false];
            inv.weapon_slots = 2;
            inv.current = 0;
        }
        world.spawn((
            Pickup {
                kind: PickupKind::Weapon(WeaponId::CROSSBOW),
            },
            GroundPhysics {
                vel: glam::Vec2::ZERO,
                rotspeed: 0.0,
            },
            WepPickupAmmo(true),
            PickupCurse,
            Pos(glam::Vec2::new(10.0, 0.0)),
        ));
        world.resource_mut::<NtInput>().press_interact();
        run(&mut world, 1);
        let inv = world.get::<Inventory>(player).unwrap();
        assert_eq!(inv.weapons[0], WeaponId::CROSSBOW, "cursed swaps for cursed");
        assert!(inv.cursed[0], "curse rides the slot");
    }

    #[test]
    fn weapon_equips_only_on_interact() {
        let mut world = sim_world();
        let player = spawn_player(&mut world);
        world.spawn((
            Pickup {
                kind: PickupKind::Weapon(WeaponId::REVOLVER),
            },
            GroundPhysics {
                vel: glam::Vec2::ZERO,
                rotspeed: 0.0,
            },
            WepPickupAmmo(true),
            Pos(glam::Vec2::new(10.0, 0.0)),
        ));
        // No interact pulse: gun stays on the ground.
        run(&mut world, 1);
        assert_eq!(
            world
                .get::<Inventory>(player)
                .unwrap()
                .weapons[0],
            WeaponId::NONE
        );
        world.resource_mut::<NtInput>().press_interact();
        run(&mut world, 1);
        let inv = world.get::<Inventory>(player).unwrap();
        assert_eq!(inv.weapons[0], WeaponId::REVOLVER);
        assert_eq!(inv.current, 0);
        assert!(
            world
                .resource::<Toast>()
                .text
                .contains(weapon_id_name(WeaponId::REVOLVER))
        );
    }

    #[test]
    fn weapon_chest_opens_loot_table() {
        let mut world = sim_world();
        spawn_player(&mut world);
        world.spawn((
            Pickup {
                kind: PickupKind::Chest(ChestKind::Weapon),
            },
            Pos(glam::Vec2::ZERO),
        ));
        run(&mut world, 1);
        assert_eq!(
            world.query::<&OpenedChest>().iter(&world).count(),
            1,
            "chest flips to opened"
        );
        assert!(
            world
                .query::<&Pickup>()
                .iter(&world)
                .any(|p| matches!(p.kind, PickupKind::Weapon(_))),
            "weapon chest drops a gun"
        );
        assert!(
            !world.resource::<Toast>().text.is_empty(),
            "chest names the gun"
        );
    }

    #[test]
    fn ammo_chest_refills_decided_type() {
        let mut world = sim_world();
        let player = spawn_player(&mut world);
        world.get_mut::<Inventory>(player).unwrap().weapons[0] = WeaponId::REVOLVER;
        world.spawn((
            Pickup {
                kind: PickupKind::Chest(ChestKind::Ammo),
            },
            Pos(glam::Vec2::ZERO),
        ));
        run(&mut world, 1);
        let inv = world.get::<Inventory>(player).unwrap();
        assert_eq!(inv.ammo_of(AmmoKind::Bullets), 64);
        assert_eq!(world.resource::<Toast>().text, "Ammo refilled");
    }

    #[test]
    fn rad_collect_levels_and_toasts() {
        let mut world = sim_world();
        let player = spawn_player(&mut world);
        world.spawn((
            Pickup {
                kind: PickupKind::Rad(60),
            },
            Pos(glam::Vec2::ZERO),
        ));
        run(&mut world, 1);
        let p = world.get::<Player>(player).unwrap();
        assert_eq!((p.level, p.rads), (2, 0));
        assert_eq!(world.resource::<Toast>().text, "LEVEL UP!");
    }

    #[test]
    fn rad_container_contact_bursts_rads() {
        let mut world = sim_world();
        spawn_player(&mut world);
        world.spawn((
            Prop {
                size: glam::Vec2::new(16.0, 16.0),
                hp: 4,
                destructible: true,
                explosive: false,
            },
            RadChestContainer,
            Pos(glam::Vec2::new(10.0, 0.0)),
        ));
        run(&mut world, 1);
        let rads = world
            .query::<&Pickup>()
            .iter(&world)
            .filter(|p| matches!(p.kind, PickupKind::Rad(_)))
            .count();
        assert_eq!(rads, 25);
    }

    #[test]
    fn weapon_label_names_near_gun_and_clears() {
        let mut world = sim_world();
        let player = spawn_player(&mut world);
        world.spawn((
            Pickup {
                kind: PickupKind::Weapon(WeaponId::REVOLVER),
            },
            Pos(glam::Vec2::new(10.0, 0.0)),
        ));
        run(&mut world, 1);
        assert_eq!(
            world.resource::<WeaponLabel>().text,
            weapon_id_name(WeaponId::REVOLVER)
        );
        world.get_mut::<Pos>(player).unwrap().0 = glam::Vec2::new(500.0, 0.0);
        run(&mut world, 1);
        assert!(world.resource::<WeaponLabel>().text.is_empty());
        assert!(world.resource::<WeaponLabel>().target.is_none());
    }

    #[test]
    fn expired_pickup_disappears_with_cue() {
        let mut world = sim_world();
        spawn_player(&mut world);
        // Far-away rad with a one-tick lifetime: drag can't reach it (no
        // portal), the lifetime drain despawns it with the disappear cue.
        world.spawn((
            Pickup {
                kind: PickupKind::Rad(1),
            },
            Pos(glam::Vec2::new(400.0, 0.0)),
            PickupLifetime {
                timer: GTimer::from_seconds(0.03, TimerMode::Once),
            },
        ));
        run(&mut world, 1);
        assert_eq!(world.query::<&Pickup>().iter(&world).count(), 0);
        assert!(
            world
                .resource_mut::<Queue<AudioCue>>()
                .drain()
                .iter()
                .any(|c| c.name == "sndPickupDisappear")
        );
    }

    #[test]
    fn open_chest_keeps_kind_for_open_art() {
        // Bevy `open_chest` swaps to kind-specific open art; the sim
        // records the kind on the marker for the renderer.
        let mut world = sim_world();
        for kind in [
            ChestKind::Weapon,
            ChestKind::Ammo,
            ChestKind::Rad,
            ChestKind::Health,
            ChestKind::CursedBig,
        ] {
            let e = world
                .spawn((
                    Pickup {
                        kind: PickupKind::Chest(kind),
                    },
                    Pos(glam::Vec2::ZERO),
                ))
                .id();
            let mut cmds = world.commands();
            open_chest(&mut cmds, e, kind);
            world.flush();
            assert!(world.get::<Pickup>(e).is_none());
            assert!(matches!(world.get::<OpenedChest>(e), Some(OpenedChest(k)) if *k == kind));
        }
    }

    #[test]
    fn health_chest_heals_num_with_stomach_bonus() {
        // GML `HealthChest`: heals 4 (8 with Second Stomach).
        let mut world = sim_world();
        spawn_player(&mut world);
        world.spawn((
            Pickup {
                kind: PickupKind::Chest(ChestKind::Health),
            },
            Pos(glam::Vec2::ZERO),
        ));
        run(&mut world, 1);
        let hp = world.query::<&Health>().iter(&world).next().unwrap().hp;
        assert_eq!(hp, 14, "10 + 4 heal");
        assert_eq!(
            world.query::<&OpenedChest>().iter(&world).count(),
            1
        );
    }

    #[test]
    fn cursed_big_chest_drops_cursed_weapons_and_clear() {
        // GML `CursedBigChest`: 3 cursed weapons + `PortalClear`.
        let mut world = sim_world();
        spawn_player(&mut world);
        world.spawn((
            Pickup {
                kind: PickupKind::Chest(ChestKind::CursedBig),
            },
            Pos(glam::Vec2::ZERO),
        ));
        run(&mut world, 1);
        assert_eq!(
            world.query::<&OpenedChest>().iter(&world).count(),
            1
        );
        let mut guns = 0;
        let mut cursed = 0;
        for (e, p) in world.query::<(Entity, &Pickup)>().iter(&world) {
            if matches!(p.kind, PickupKind::Weapon(_)) {
                guns += 1;
                if world.get::<PickupCurse>(e).is_some() {
                    cursed += 1;
                }
            }
        }
        assert_eq!(guns, 3, "three weapons");
        assert_eq!(cursed, 3, "all cursed");
        assert_eq!(
            world.query::<&crate::comps_b::PortalClear>().iter(&world).count(),
            1,
            "PortalClear spawns"
        );
    }
}
