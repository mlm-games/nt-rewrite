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
    save.total_deaths += 1;
    if run.hardmode {
        save.hard_runs += 1;
    }
    save.win_streak_cur = 0;
    save.total_time_steps = save.total_time_steps.saturating_add(run.tottimer as u64);
    crate::savedata_part::update_best_run_stats(
        &mut save,
        race_state.race as u8,
        crate::worldgen::gml_area_from_run(&run),
        run.floor_in_area,
        run.loop_count,
        run.total_kills,
        run.hardmode,
    );
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
