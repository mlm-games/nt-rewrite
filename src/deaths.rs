//! Player-death resolution. Third leg of the death pipeline, run after
//! `resolve_death_drops`: revives first (headless, strong spirit,
//! melting skeleton), else game over with corpse, drops, and save
//! writes. Split into two systems only because bevy_ecs caps systems
//! at 16 params; order matches the bevy function top to bottom.

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_fx::Trauma;

use crate::anim::{PlayerAnim, SpriteAnim};
use crate::audio::{AudioCue, GameAudio, QueuedReactiveCue, ReactiveCue};
use crate::comps_a::{
    AimDir, GameCleanup, Health, Inventory, LastDamageTaken, LevelCleanup, Player, Projectile,
    RaceState, Team, Velocity,
};
use crate::comps_b::{Corpse, Enemy, PickupKind, PickupLifetime, PlayerDying};
use crate::comps_b::{FloorTransition, Revive};
use crate::data::RaceId;
use crate::effects::{FlashWhite, RumbleRequest, flash_white};
use crate::msg::Queue;
use crate::pickups::spawn_pickup;
use crate::savedata_part::SaveData;
use crate::spatial::Pos;
use crate::time::{GTimer, TimerMode};

/// Headless / strong-spirit / melting-skeleton revives. Each returns
/// early like the bevy branch chain.
pub fn resolve_player_revives(
    mut commands: Commands,
    mut save: ResMut<SaveData>,
    mut dirty: ResMut<crate::comps_a::SaveDirty>,
    mut toast: ResMut<crate::comps_a::Toast>,
    mut trauma: ResMut<Trauma>,
    mut flash: ResMut<FlashWhite>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut player_q: Query<
        (
            Entity,
            &Pos,
            &mut Player,
            &mut Health,
            &Inventory,
            &mut RaceState,
        ),
        (With<Player>, Without<Enemy>),
    >,
    enemies: Query<(&Pos, &Team, &Health, Option<&Enemy>), With<Enemy>>,
) {
    let Ok((player_e, player_pos, mut player, mut phealth, _pinv, mut race_state)) =
        player_q.single_mut()
    else {
        return;
    };
    let _ = player_e;
    if phealth.hp > 0 {
        return;
    }

    if phealth.hp <= 0 && player.headless_ready {
        player.headless_ready = false;
        // GML head loss: bank up to 2 max-HP into `headloses`.
        let n = phealth.max.min(2).max(0);
        player.headloses += n as u32;
        phealth.max = (phealth.max - n).max(1);
        phealth.hp = 1;
        phealth.invuln = GTimer::from_seconds(1.5, TimerMode::Once);
        crate::combat::HitFlash::apply(&mut commands, player_e, [1.0, 0.95, 0.6, 1.0], 0.25);
        audio.play_pickup(&mut cues);
        let mut rng = rand::rng();
        crate::effects::spawn_burst(
            &mut commands,
            &mut rng,
            player_pos.0,
            12,
            [1.0, 0.95, 0.6, 1.0],
            (60.0, 160.0),
        );
        return;
    }

    if phealth.hp <= 0 {
        if player.strong_spirit_ready {
            player.strong_spirit_ready = false;
            player.strong_spirit_spent = true;
            let n = phealth.max.min(2).max(0);
            player.headloses += n as u32;
            phealth.max = (phealth.max - n).max(1);
            phealth.hp = 1;
            phealth.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
            crate::combat::HitFlash::apply(&mut commands, player_e, [0.3, 1.0, 0.5, 1.0], 0.3);
            audio.play_pickup(&mut cues);
            return;
        }

        if race_state.race == RaceId::Melting {
            let ppos = player_pos.0;
            let near_necro = enemies.iter().any(|(ntf, team, health, enemy)| {
                *team == Team::Enemy
                    && health.hp > 0
                    && enemy
                        .map(|e| e.kind == crate::data::EnemyKind::Necromancer)
                        .unwrap_or(false)
                    && ntf.0.distance(ppos) <= 96.0
            });
            if near_necro {
                let unlocked = crate::savedata_part::try_unlock_skeleton(&mut save);
                if unlocked {
                    dirty.0 = true;
                }

                let sk = crate::savedata_part::character_def(RaceId::Skeleton);
                race_state.race = RaceId::Skeleton;
                player.ability = sk.ability;
                player.chain_explosions = false;
                player.shield_on_hit = false;
                player.headless_ready = false;

                phealth.max = sk.max_hp.max(2);
                phealth.hp = phealth.max;
                phealth.invuln = GTimer::from_seconds(1.25, TimerMode::Once);
                player.ability_cooldown = GTimer::from_seconds(0.0, TimerMode::Once);

                toast.show(if unlocked {
                    "SKELETON UNLOCKED"
                } else {
                    "BACK FROM THE DEAD"
                });
                crate::combat::HitFlash::apply(
                    &mut commands,
                    player_e,
                    [0.95, 0.95, 1.0, 1.0],
                    0.35,
                );
                flash_white(&mut flash, 0.08);
                trauma.add(0.35);
                audio.play_portal(&mut cues);
                let mut rng = rand::rng();
                crate::effects::spawn_burst(
                    &mut commands,
                    &mut rng,
                    ppos,
                    28,
                    [0.9, 0.9, 1.0, 1.0],
                    (80.0, 220.0),
                );
                return;
            }
        }
    }
}

/// Coop downed timers (GML `Revive/Step_0` + `Alarm_4/5`): generation
/// (`FloorTransition.active`, the `GenCont`/`LevCont` stand-in) holds
/// every grace at 300, otherwise each [`Revive`] steps its alarms.
/// Hurt pulses only re-arm the timer here — applying the 1-damage
/// `Alarm_5` hit and spawning/despawning downed markers need coop
/// downing, which the port does not simulate yet (single-player).
pub fn tick_revive(
    time: Res<repame_sim::SimTime>,
    transition: Option<Res<FloorTransition>>,
    mut q: Query<&mut Revive>,
) {
    let generating = transition.is_some_and(|t| t.active);
    let steps = time.delta_secs * 30.0;
    for mut revive in q.iter_mut() {
        crate::comps_b::revive_step(&mut revive, steps, generating);
    }
}

/// Game-over branch: flags, death anim + marker, sting, corpse, weapon
/// and rad drops, crown bursts, blood gibs, save writes, toast.
#[allow(clippy::too_many_arguments)]
pub fn resolve_player_gameover(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut save: ResMut<SaveData>,
    mut dirty: ResMut<crate::comps_a::SaveDirty>,
    mut paused: ResMut<crate::state::Paused>,
    mut run: ResMut<crate::comps_a::Run>,
    mut toast: ResMut<crate::comps_a::Toast>,
    mut trauma: ResMut<Trauma>,
    mut chroma: ResMut<crate::effects::ChromaticAberration>,
    mut flash: ResMut<FlashWhite>,
    mut hitstop: ResMut<crate::effects::HitStop>,
    mut slow_mo: ResMut<crate::effects::SlowMotion>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut rumble_queue: ResMut<Queue<RumbleRequest>>,
    last_damage: Res<LastDamageTaken>,
    mut player_q: Query<(
        Entity,
        &Pos,
        &Player,
        &Health,
        &Inventory,
        &RaceState,
        Option<&PlayerAnim>,
        Option<&mut SpriteAnim>,
        Option<&AimDir>,
        Option<&Velocity>,
    )>,
) {
    let Ok((
        player_e,
        player_pos,
        player,
        phealth,
        pinv,
        race_state,
        pa_opt,
        mut anim_opt,
        aim_opt,
        vel_opt,
    )) = player_q.single_mut()
    else {
        return;
    };
    if phealth.hp > 0 || run.game_over {
        return;
    }
    // Revive paths above own these cases; game over handles the rest.
    if player.headless_ready || player.strong_spirit_ready {
        return;
    }

    run.game_over = true;

    if let (Some(pa), Some(anim)) = (pa_opt, anim_opt.as_deref_mut()) {
        let dead = crate::dead_path_part::derive_dead_path(pa.idle);
        if let Some(def) = catalog.def(dead) {
            anim.set_path(dead, def, true);
        }
    }
    commands.entity(player_e).insert(PlayerDying {
        timer: GTimer::from_seconds(0.85, TimerMode::Once),
    });
    commands.spawn(QueuedReactiveCue(ReactiveCue::PlayerDeath));
    GameAudio.play_death(&mut cues);

    let pos = player_pos.0;
    // Bevy `face_aim` law for the dying flip: live aim.x, else velocity.
    let player_flip_at_death = aim_opt
        .map(|a| a.0)
        .filter(|a| a.length_squared() > 0.001)
        .map(|a| a.x)
        .unwrap_or_else(|| vel_opt.map(|v| v.0.x).unwrap_or(0.0))
        < 0.0;
    let mut rng = rand::rng();
    let angle = rng.random_range(0.0..std::f32::consts::TAU);
    let dir = glam::Vec2::new(angle.cos(), angle.sin());
    let extra_frames = ((-phealth.hp as f32) / 5.0).max(0.0);
    let corpse_frames = (rng.random_range(1.0..3.0) + extra_frames).clamp(1.0, 16.0);
    let dead_path = pa_opt
        .map(|pa| crate::dead_path_part::derive_dead_path(pa.idle))
        .unwrap_or("images/sprMutant1Dead.png");
    let mut corpse_e = commands.spawn((
        GameCleanup,
        LevelCleanup,
        crate::comps_b::GroundPhysics {
            vel: dir * corpse_frames * 30.0,
            rotspeed: rng.random_range(-3.0..3.0),
        },
        PickupLifetime {
            timer: GTimer::from_seconds(12.0, TimerMode::Once),
        },
        Corpse {
            kind: crate::data::EnemyKind::Bandit,
            life: GTimer::from_seconds(12.0, TimerMode::Once),
            pos,
            // Bevy preserves the dying sprite's flip on the husk.
            flip_x: player_flip_at_death,
        },
        Pos(pos),
    ));
    if let Some(def) = catalog.def(dead_path) {
        let mut anim = SpriteAnim::new(dead_path, def);
        anim.oneshot = true;
        corpse_e.insert(anim);
    }
    for slot in [0, 1] {
        let wid = pinv.weapons[slot];
        if wid != crate::data::WeaponId::NONE {
            // GML death re-drops preserve the slot curse.
            let e = spawn_pickup(
                &mut commands,
                &catalog,
                PickupKind::Weapon(wid),
                pos + glam::Vec2::new(rng.random_range(-10.0..10.0), rng.random_range(-10.0..10.0)),
                0,
                false,
            );
            if pinv.cursed[slot] {
                commands.entity(e).insert(crate::comps_b::PickupCurse);
            }
        }
    }
    if race_state.race == RaceId::Horror && player.rads > 0 {
        spawn_pickup(
            &mut commands,
            &catalog,
            PickupKind::Rad(player.rads),
            pos,
            0,
            false,
        );
    }
    if player.crown == crate::data::CrownKind::Death {
        crate::effects::spawn_burst(
            &mut commands,
            &mut rng,
            pos,
            60,
            [1.0, 0.5, 0.15, 1.0],
            (100.0, 360.0),
        );
        crate::effects::spawn_burst(
            &mut commands,
            &mut rng,
            pos,
            30,
            [1.0, 0.9, 0.5, 1.0],
            (60.0, 220.0),
        );
        trauma.add(0.5);
    }
    if race_state.race == RaceId::BigDog {
        crate::effects::spawn_burst(
            &mut commands,
            &mut rng,
            pos,
            40,
            [1.0, 0.4, 0.1, 1.0],
            (130.0, 400.0),
        );
    }
    if race_state.race == RaceId::Cuz {
        // GML `Player/Destroy_0:51-61`: death cries the same
        // `20*(1+Emotional)` tear ring as the active.
        let tears = 20 * (1 + crate::player_fire::cuz_emotional_level(player.ultra));
        let step = std::f32::consts::TAU / tears as f32;
        for i in 0..tears {
            let ang = i as f32 * step;
            let dir = glam::Vec2::new(ang.cos(), ang.sin());
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                Team::Player,
                Projectile {
                    damage: 3,
                    life: GTimer::from_seconds(1.5, TimerMode::Once),
                    radius: 5.0,
                    knockback: 120.0,
                    explosive: false,
                    source: None,
                },
                Velocity(dir * 180.0),
                Pos(pos + dir * 16.0),
            ));
        }
    }
    for _ in 0..12 {
        let a = rng.random_range(0.0..std::f32::consts::TAU);
        let d = glam::Vec2::new(a.cos(), a.sin());
        let s = rng.random_range(50.0..160.0);
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            crate::comps_b::GroundPhysics {
                vel: d * s
                    + glam::Vec2::new(rng.random_range(-10.0..10.0), rng.random_range(-10.0..10.0)),
                rotspeed: rng.random_range(-6.0..6.0),
            },
            PickupLifetime {
                timer: GTimer::from_seconds(0.9, TimerMode::Once),
            },
            Gib {
                color: [0.82, 0.08, 0.12, 1.0],
            },
            Pos(pos),
        ));
    }

    crate::effects::spawn_burst(
        &mut commands,
        &mut rng,
        pos,
        10,
        [0.82, 0.08, 0.12, 1.0],
        (40.0, 120.0),
    );

    if save.best_floor < run.floor {
        save.best_floor = run.floor;
    }
    save.total_runs += 1;
    save.total_kills = save.total_kills.saturating_add(run.total_kills);
    crate::savedata_part::check_progress_unlocks(
        &mut save,
        run.floor,
        run.loop_count,
        true,
        false,
        false,
    );
    dirty.0 = true;
    paused.0 = false;
    toast.show(&format!("KILLED BY {}", last_damage.source_name));

    trauma.add(0.8);
    crate::effects::chromatic_pulse(&mut chroma, 0.7);
    flash_white(&mut flash, 0.06);
    hitstop.trigger(0.1, 0.25);
    crate::effects::slow_motion(&mut slow_mo, 0.3, 1.4);
    crate::effects::rumble(&mut rumble_queue, 0.9, 1.0, 0.6);
}

/// Blood-gib marker: positionally spawned gore the renderer draws as
/// red dots (bevy carried a tinted `Sprite`; the tint lives here so no
/// render handle is needed).
#[derive(bevy_ecs::prelude::Component, Clone, Copy, Debug)]
pub struct Gib {
    pub color: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;
    use repame_anim::{AnimCatalog, AtlasDesc};

    const DEATH_JSON: &str = r#"{
        "images/sprMutant1Dead.png": {"frames": 2, "w": 32, "h": 32, "fps": 8.0, "xorigin": 16.0, "yorigin": 16.0}
    }"#;

    fn death_catalog() -> AnimCatalog {
        AnimCatalog::from_json(
            DEATH_JSON,
            AtlasDesc {
                size: 128,
                max_pages: 1,
            padding: 0,
            },
        )
        .unwrap()
    }

    fn death_world() -> bevy_ecs::prelude::World {
        let mut world = bevy_ecs::prelude::World::new();
        world.init_resource::<crate::comps_a::SaveDirty>();
        world.init_resource::<crate::state::Paused>();
        world.init_resource::<crate::comps_a::Run>();
        world.init_resource::<crate::comps_a::Toast>();
        world.init_resource::<Trauma>();
        world.init_resource::<crate::effects::ChromaticAberration>();
        world.init_resource::<FlashWhite>();
        world.init_resource::<crate::effects::HitStop>();
        world.init_resource::<crate::effects::SlowMotion>();
        world.init_resource::<GameAudio>();
        world.init_resource::<Queue<AudioCue>>();
        world.init_resource::<Queue<RumbleRequest>>();
        world.init_resource::<LastDamageTaken>();
        world.init_resource::<SaveData>();
        world.insert_resource(death_catalog());
        world
    }

    fn spawn_dying_player(world: &mut bevy_ecs::prelude::World) -> bevy_ecs::prelude::Entity {
        let mut player = Player::default();
        player.headless_ready = false;
        player.strong_spirit_ready = false;
        world
            .spawn((
                player,
                Pos(glam::Vec2::ZERO),
                Health {
                    hp: 0,
                    max: 8,
                    invuln: GTimer::disarmed(),
                },
                Inventory {
                    weapons: [
                        crate::data::WeaponId::REVOLVER,
                        crate::data::WeaponId::NONE,
                        crate::data::WeaponId::NONE,
                    ],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
                    weapon_slots: 3,
                    current: 0,
                    ammo: [0, 0, 0, 0, 0, 0],
                },
                RaceState {
                    race: RaceId::Fish,
                    skin: crate::data::SkinLetter::A,
                },
                PlayerAnim {
                    idle: "images/sprMutant1Idle.png",
                    walk: "images/sprMutant1Walk.png",
                    hurt: "images/sprMutant1Hurt.png",
                    moving: false,
                },
                SpriteAnim::new(
                    "images/sprMutant1Idle.png",
                    death_catalog().def("images/sprMutant1Dead.png").unwrap(),
                ),
            ))
            .id()
    }

    #[test]
    fn headless_revive_cheats_death() {
        let mut world = death_world();
        let e = spawn_dying_player(&mut world);
        world.get_mut::<Player>(e).unwrap().headless_ready = true;
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(resolve_player_revives);
        schedule.run(&mut world);
        let health = world.get::<Health>(e).unwrap();
        assert_eq!(health.hp, 1);
        assert!(!world.get::<Player>(e).unwrap().headless_ready);
        // No flash on this path in bevy either (sting + burst only).
        assert!(!world.resource::<crate::comps_a::Run>().game_over);
    }

    #[test]
    fn strong_spirit_revive_spends_once() {
        let mut world = death_world();
        let e = spawn_dying_player(&mut world);
        let mut player = world.get_mut::<Player>(e).unwrap();
        player.strong_spirit_ready = true;
        drop(player);
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(resolve_player_revives);
        schedule.run(&mut world);
        let player = world.get::<Player>(e).unwrap();
        assert!(player.strong_spirit_spent);
        assert!(!player.strong_spirit_ready);
        assert_eq!(world.get::<Health>(e).unwrap().hp, 1);
    }

    #[test]
    fn melting_revive_becomes_skeleton_near_necro() {
        use crate::comps_b::Enemy;
        let mut world = death_world();
        let e = spawn_dying_player(&mut world);
        world.get_mut::<RaceState>(e).unwrap().race = RaceId::Melting;
        world.spawn((
            Enemy {
                kind: crate::data::EnemyKind::Necromancer,
                score: 0,
                touch_damage: 0,
                rad_drop: 0,
                drop_chance: 0,
                weapon_chance: 0,
            },
            Pos(glam::Vec2::new(50.0, 0.0)),
            Team::Enemy,
            Health {
                hp: 10,
                max: 10,
                invuln: GTimer::disarmed(),
            },
        ));
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(resolve_player_revives);
        schedule.run(&mut world);
        assert_eq!(world.get::<RaceState>(e).unwrap().race, RaceId::Skeleton);
        assert_eq!(
            world.get::<Health>(e).unwrap().hp,
            world.get::<Health>(e).unwrap().max
        );
        assert!(world.resource::<crate::comps_a::SaveDirty>().0);
    }

    #[test]
    fn game_over_corpse_drops_and_save() {
        let mut world = death_world();
        world.resource_mut::<crate::comps_a::Run>().floor = 3;
        world.resource_mut::<LastDamageTaken>().source_name = "BANDIT".to_string();
        let e = spawn_dying_player(&mut world);
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(resolve_player_gameover);
        schedule.run(&mut world);

        assert!(world.resource::<crate::comps_a::Run>().game_over);
        assert!(world.get::<PlayerDying>(e).is_some());
        assert_eq!(
            world.resource::<crate::comps_a::Toast>().text,
            "KILLED BY BANDIT"
        );
        // Corpse + gibs + weapon drop + death sting.
        assert_eq!(world.query::<&Corpse>().iter(&world).count(), 1);
        assert_eq!(world.query::<&Gib>().iter(&world).count(), 12);
        assert_eq!(
            world
                .query::<&crate::comps_b::Pickup>()
                .iter(&world)
                .filter(|p| matches!(p.kind, crate::comps_b::PickupKind::Weapon(_)))
                .count(),
            1
        );
        let mut cues = world.query::<&QueuedReactiveCue>();
        assert_eq!(cues.iter(&world).count(), 1);
        // Save writes.
        assert_eq!(world.resource::<SaveData>().best_floor, 3);
        assert_eq!(world.resource::<SaveData>().total_runs, 1);
        assert!(world.resource::<crate::comps_a::SaveDirty>().0);
        assert!(!world.resource::<crate::state::Paused>().0);
        // Feel sinks.
        assert!((world.resource::<Trauma>().amount - 0.8).abs() < 1e-6);
        assert_eq!(world.resource::<FlashWhite>().amount, 1.0);
        assert_eq!(world.resource::<Queue<AudioCue>>().len(), 1);
    }

    #[test]
    fn cuz_game_over_cries_tear_ring() {
        // GML `Player/Destroy_0:51-61`: same `20*(1+Emotional)` count as
        // the active (20 base, 40 with Emotional).
        for (ultra, want) in [
            (None, 20),
            (
                Some(crate::data::UltraMutationId::CuzEmotional),
                40,
            ),
        ] {
            let mut world = death_world();
            let e = spawn_dying_player(&mut world);
            world.get_mut::<RaceState>(e).unwrap().race = RaceId::Cuz;
            world.get_mut::<Player>(e).unwrap().ultra = ultra;
            let mut schedule = bevy_ecs::schedule::Schedule::default();
            schedule.add_systems(resolve_player_gameover);
            schedule.run(&mut world);
            assert_eq!(
                world.query::<&Projectile>().iter(&world).count(),
                want as usize,
                "ultra {ultra:?}"
            );
        }
    }
}
