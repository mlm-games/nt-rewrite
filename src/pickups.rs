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
                        scale: 1.0,
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
                        scale: 1.0,
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


