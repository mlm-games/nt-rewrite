//! Combat hit pipeline. Ported from nt's `game/combat.rs` in `NtSimSet`
/// order (Always → Input → Combat → Progression → Cleanup).
///
/// Render writes (`Sprite.image/rect`, tint restore) stay out: systems
/// mutate sim state and marker components, and the render phase resolves
/// visuals. Effect sinks (`Trauma`, `FlashWhite`, audio cues, rumble,
/// bursts, secrets) are ported alongside and asserted in tests.
use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::anim::SpriteAnim;
use crate::audio::{AudioCue, GameAudio};
use crate::comps_a::{
    ARENA_H, ARENA_W, BouncesLeft, ChainLightning, CurrentFrame, DamageSource, DiscFlight,
    FireCooldown, FlameShellSlowDeath, FlameTrail, FloorMask, GameCleanup, GrenadeFuse, Health,
    HitId, Hitbox, HitsAllTeams, Homing, Inventory, LastDamageTaken, LevelCleanup, LightningArc,
    NextHurt, PendingWallBreak, PiercesLeft, PlasmaSize, Player, Projectile, ProjectileFade,
    ProjectileFriction, ProjectileHitSet, ProjectileTyp, RaceState, Run, SaveDirty, Score,
    ShellBonus, ShellWallBounce, SlashProjectile, SpawnGrace, SpawnHazardOnDeath, SplitOnDeath,
    Sticky, Team, Toast, Velocity, WallCell, WallTile,
};
use crate::comps_b::{
    Beam, ChestKind, Corpse, CustomExplosion, DeploysSentry, Dying, Enemy, EnemyBrain,
    GoldBarrelDrop, HazardCloud, LoopTransition, Pickup, PickupLifetime, PlasmaBurst,
    Portal, PortalPhase, PortalShock, PortalState, Prop, PropSprites, RadChestContainer,
    SecretEntrance, SentryTurret, Shield, SnowmanAmbush,
    SpawnsWeaponPickup, ThroneRoomState,
};
use crate::data::{CrownKind, EnemyKind, HazardKind, MutationId, RaceId, WeaponId};
use crate::effects::{
    ChromaticAberration, FlashWhite, HitStop, RumbleRequest, SlowMotion, chromatic_pulse,
    flash_white, rumble, slow_motion, spawn_burst,
};
use crate::enemy_data::enemy_def;
use crate::environment::{PropDeathEffect, spawn_prop_corpse, spawn_prop_death_effect};
use crate::msg::Queue;
use crate::pickups::{
    give_ammo, maybe_spawn_drop, random_gold_weapon_fallback, random_offset, spawn_chest,
    spawn_pickup, spawn_rad, spawn_rad_burst,
};
use crate::projectile_math::{    arena_wall_normal, bounce_velocity, circle_aabb_normal, record_hit, should_despawn_after_hit,
};
use crate::secrets::SecretTriggers;
use crate::savedata_part::{SaveData, check_kill_unlocks};
use crate::spatial::{PLAYER_RADIUS, Pos};
use crate::spawns::{
    damage_destructible_prop, on_projectile_removed, spawn_explosion_with_source_radius,
    spawn_hazard_cloud,
};
use crate::time::{GTimer, TimerMode};

/// Hit-flash marker (game-utils `HitFlash` parity, renderer-resolved).
/// Sim side only tracks liveness: the tint applies while the marker is
/// present and restores when `tick_hit_flash` removes it. Colors are
/// sRGB-authored display colors (bevy `Color::srgb` parity); the renderer
/// linearizes them for the GPU.
#[derive(Component, Clone, Debug)]
pub struct HitFlash {
    pub color: [f32; 4],
    pub value: f32,
    pub timer: GTimer,
    pub original: Option<[f32; 4]>,
}

impl HitFlash {
    pub fn new(color: [f32; 4], duration_secs: f32) -> Self {
        Self::with_value(color, duration_secs, 0.85)
    }

    pub fn with_value(color: [f32; 4], duration_secs: f32, value: f32) -> Self {
        Self {
            color,
            value: value.clamp(0.0, 4.0),
            timer: GTimer::from_seconds(duration_secs, TimerMode::Once),
            original: None,
        }
    }

    pub fn for_damage(color: [f32; 4], damage: f32) -> Self {
        let d = (damage / 300.0).clamp(0.08, 0.16);
        Self::new(color, d)
    }

    pub fn apply(commands: &mut Commands, entity: Entity, color: [f32; 4], duration_secs: f32) {
        commands
            .entity(entity)
            .insert(Self::new(color, duration_secs));
    }
}

/// Pure knockback law (game-utils `GameFeel::apply_knockback` parity):
/// velocity is *set* to the directed force, not added.
pub fn apply_knockback(velocity: &mut glam::Vec2, dir: glam::Vec2, force: f32) {
    *velocity = dir.normalize_or_zero() * force;
}

/// Gamma Guts aura: enemies within 60px of a guts carrier whose
/// invulnerability lapsed take 6 and start fresh invulnerability.
/// Green hit-flash marker rides along for the renderer.
pub fn gamma_guts_aura(
    mut commands: Commands,
    player_q: Query<(&Pos, &Player), (With<Player>, Without<Enemy>)>,
    mut enemies: Query<(Entity, &Pos, &mut Health), (With<Enemy>, Without<Player>)>,
) {
    let Ok((ppos, player)) = player_q.single() else {
        return;
    };
    if !player.gamma_guts {
        return;
    }
    for (e, epos, mut health) in &mut enemies {
        if epos.0.distance(ppos.0) >= 60.0 {
            continue;
        }
        if !health.invuln.is_finished() {
            continue;
        }
        health.hp -= 6;
        health.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
        HitFlash::apply(&mut commands, e, [0.4, 1.0, 0.4, 1.0], 0.1);
    }
}

/// Expire lapsed hit-flash markers (tint restore happens renderer-side
/// from the live marker).
pub fn tick_hit_flash(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut HitFlash)>,
) {
    for (e, mut flash) in &mut q {
        flash.timer.tick(time.delta_secs);
        if flash.timer.just_finished() {
            commands.entity(e).remove::<HitFlash>();
        }
    }
}

/// Enemy contact damage: overlapping enemies with a cooled melee timer
/// hurt the player (unless invulnerable), knock back, flash, rumble,
/// sting audio, spawn a burst, and optionally raise a shield.
/// Sharp Teeth retaliates in 900px. Bevy parity throughout, including
/// the unused `flash` parameter.
pub fn contact_damage(
    mut commands: Commands,
    mut trauma: ResMut<Trauma>,
    mut flash: ResMut<FlashWhite>,
    audio: Res<GameAudio>,
    mut rumble_queue: ResMut<Queue<RumbleRequest>>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut secrets: ResMut<SecretTriggers>,
    mut last_damage: ResMut<LastDamageTaken>,
    mut player_q: Query<
        (Entity, &Pos, &mut Health, &mut Velocity, &Player),
        (With<Player>, Without<Enemy>),
    >,
    mut enemies: Query<
        (
            &Pos,
            &Enemy,
            &mut EnemyBrain,
            &mut Health,
            &Hitbox,
            Option<&mut Velocity>,
        ),
        (With<Enemy>, Without<Player>),
    >,
) {
    let _ = &mut flash;
    let Ok((player_e, player_pos, mut health, mut player_vel, player)) = player_q.single_mut()
    else {
        return;
    };

    if !health.invuln.is_finished() {
        return;
    }

    let player_pos = player_pos.0;
    let mut took_damage = 0;

    for (enemy_pos, enemy, mut brain, ehealth, enemy_hitbox, enemy_vel) in &mut enemies {
        if !brain.melee.is_finished() {
            continue;
        }
        if player_pos.distance(enemy_pos.0) >= PLAYER_RADIUS + enemy_hitbox.radius {
            continue;
        }

        if player.gamma_guts && ehealth.hp <= 6 {
            continue;
        }

        let damage = if brain.dash > 0.0 {
            10
        } else {
            enemy.touch_damage
        };
        if damage <= 0 {
            continue;
        }

        health.hp -= damage;
        took_damage = damage;
        health.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);

        brain.melee = GTimer::from_seconds(1.0, TimerMode::Once);
        secrets.mark_damage_taken();
        last_damage.note(Some(HitId::Contact), Some(enemy.kind));

        let away = (player_pos - enemy_pos.0).normalize_or_zero();
        apply_knockback(&mut player_vel.0, away, 120.0);

        if enemy_def(enemy.kind).size <= 2.0
            && let Some(mut evel) = enemy_vel
        {
            let push_enemy = (enemy_pos.0 - player_pos).normalize_or_zero();
            apply_knockback(&mut evel.0, push_enemy, 30.0);
        }

        HitFlash::apply(&mut commands, player_e, [1.0, 0.15, 0.1, 1.0], 0.18);
        trauma.add(0.35);
        // Single-player build: rumble routing is platform-side, so the
        // bevy per-gamepad fan-out collapses to one queued request.
        rumble(&mut rumble_queue, 0.2, 0.8, 0.16);
        audio.play_hurt(&mut cues);

        let mut rng = rand::rng();
        spawn_burst(
            &mut commands,
            &mut rng,
            player_pos,
            12,
            [1.0, 0.1, 0.08, 1.0],
            (80.0, 220.0),
        );

        if player.shield_on_hit {
            commands.entity(player_e).insert(Shield {
                timer: GTimer::from_seconds(0.7, TimerMode::Once),
            });
        }
        break;
    }

    if took_damage > 0 && player.sharp_teeth {
        for (enemy_pos, _, _, mut ehealth, _, _) in &mut enemies {
            if enemy_pos.0.distance(player_pos) <= 900.0 {
                ehealth.hp -= took_damage * 2;
            }
        }
    }
}

/// Timed area explosion (bevy `Explosion` parity).
#[derive(Component, Clone, Debug)]
pub struct Explosion {
    pub timer: GTimer,
    pub radius: f32,
    pub damage: i32,
    pub team: Team,
    pub hits_player: bool,
    pub source: Option<DamageSource>,
}

/// GML lingering blast: the explosion re-scans every 1/30 s for 0.75 s,
/// hitting each victim once (bevy `LingeringBlast` parity — walk-ins
/// caught like GML).
#[derive(Component, Clone, Debug)]
pub struct LingeringBlast {
    pub duration: GTimer,
    pub tick: GTimer,
    pub hit: Vec<Entity>,
}

/// One processed enemy death, handed from `resolve_enemy_deaths` to
/// `resolve_death_drops`. The split exists because bevy_ecs caps
/// systems at 16 params; iterating the record preserves the bevy
/// per-death multiplicity exactly (including repeated TriggerFingers
/// scaling with several deaths in one tick).
#[derive(Clone, Copy, Debug)]
pub struct DeathEvent {
    pub pos: glam::Vec2,
    pub kind: EnemyKind,
    pub rad_drop: usize,
    pub drop_chance: usize,
    pub weapon_chance: usize,
}

/// Death records of the current tick, drained by `resolve_death_drops`.
#[derive(Resource, Debug, Default)]
pub struct DeathEvents(pub Vec<DeathEvent>);

/// Deferred enemy spawn request, drained by the spawner system.
/// (Port of nt's `PendingEnemySpawn`.) `loops` drives the GML HP law.
#[derive(Component, Clone, Copy, Debug)]
pub struct PendingEnemySpawn {
    pub kind: EnemyKind,
    pub pos: glam::Vec2,
    pub difficulty: f32,
    pub loops: u32,
}

/// Enemy death resolution, slice A: despawn + corpse slide, kill
/// counting, Throne/ThroneII transitions, feel triggers, death burst,
/// Throne boom, hit sting.
///
/// Slice B (not here): per-kind spawn arms (Ballguy, ExploFreak,
/// BigMaggot, …), unlock/save writes, the player-death branch. Each is
/// marked TODO at its bevy location.
pub fn resolve_enemy_deaths(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut score: ResMut<Score>,
    mut run: ResMut<Run>,
    mut trauma: ResMut<Trauma>,
    mut chroma: ResMut<ChromaticAberration>,
    mut hitstop: ResMut<HitStop>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut deaths: ResMut<DeathEvents>,
    mut save: ResMut<SaveData>,
    mut dirty: ResMut<SaveDirty>,
    mut toast: ResMut<Toast>,
    player_q: Query<(Entity, &Pos, &Player, &RaceState), (With<Player>, Without<Enemy>)>,
    mut enemy_shots: Query<(Entity, &Team), With<Projectile>>,
    mut q: Query<
        (
            Entity,
            &Pos,
            &Team,
            &Health,
            Option<&Enemy>,
            Option<&Velocity>,
            Option<&crate::comps_b::ProtoGuardian>,
        ),
        (Without<Prop>, Without<Player>, Without<Dying>),
    >,
) {
    if run.game_over {
        return;
    }

    let Ok((player_e, player_pos, player, race_state)) = player_q.single() else {
        return;
    };
    let _ = player_e;

    let enemy_total = q
        .iter()
        .filter(|(_, _, team, _, _, _, _)| **team == Team::Enemy)
        .count();
    if enemy_total == 2 {
        audio.play_levelup(&mut cues);
    }

    for (e, pos, team, health, enemy, enemy_vel, statue) in &mut q {
        if *team != Team::Enemy || health.hp > 0 {
            continue;
        }

        let enemy = enemy.copied().unwrap_or(Enemy {
            kind: EnemyKind::Maggot,
            score: 1,
            touch_damage: 1,
            rad_drop: 1,
            drop_chance: 0,
            weapon_chance: 0,
        });
        let def = enemy_def(enemy.kind);
        let pos = pos.0;

        commands.entity(e).insert(Dying);
        commands.entity(e).despawn();
        if !def.boss && !matches!(enemy.kind, EnemyKind::IdpdVan | EnemyKind::FrogEgg) {
            let idle = def.sprite;
            let dead = crate::dead_path_part::derive_dead_path(idle);
            if let Some(corpse_def) = catalog.def(dead) {
                let mut corpse_anim = SpriteAnim::new(dead, corpse_def);
                corpse_anim.oneshot = true;
                let mut slide = enemy_vel.map(|v| v.0).unwrap_or(glam::Vec2::ZERO);
                if player.mutations.contains(&MutationId::ImpactWrists) {
                    slide += slide.normalize_or_zero() * 8.0 * 30.0;
                }
                let cap = 16.0 * 30.0 / def.size.max(1.0);
                if slide.length() > cap {
                    slide = slide.normalize_or_zero() * cap;
                }
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Corpse {
                        kind: enemy.kind,
                        life: GTimer::from_seconds(12.0, TimerMode::Once),
                        pos,
                        flip_x: false,
                    },
                    corpse_anim,
                    Pos(pos),
                    Velocity(slide),
                ));
            }
        }

        let give_kill = !matches!(enemy.kind, EnemyKind::FastRat);
        if give_kill {
            run.total_kills += 1;
            score.0 += enemy.score;
        }

        if score.0 > save.high_score {
            save.high_score = score.0;
            dirty.0 = true;
        }

        let newly = check_kill_unlocks(&mut save, enemy.kind, race_state.race);
        if !newly.is_empty() {
            dirty.0 = true;
            for r in newly {
                toast.show(&format!(
                    "{} UNLOCKED",
                    crate::savedata_part::character_def(r)
                        .name
                        .to_ascii_uppercase()
                ));
            }
        }

        // GML boss achievements (31-38, 42, 51).
        if let Some(id) = crate::savedata_part::achievement_for_boss(enemy.kind)
            && crate::savedata_part::unlock_achievement(&mut save, id)
        {
            dirty.0 = true;
            if let Some((name, _, _, _)) = crate::savedata_part::achievement_def(id) {
                toast.show(name);
            }
        }

        trauma.add(0.25);
        chromatic_pulse(&mut chroma, 0.08);

        let size_f = enemy_def(enemy.kind).size;
        let stop_dur = 0.02 + size_f * 0.015;
        hitstop.trigger(stop_dur.min(0.28), 0.055);

        let burst_count = if def.boss {
            40 + run.loop_count as usize * 6
        } else {
            14
        };
        let boom_radius = if enemy.kind == EnemyKind::Throne {
            130.0 + run.loop_count as f32 * 18.0
        } else {
            0.0
        };
        let mut rng = rand::rng();
        spawn_burst(
            &mut commands,
            &mut rng,
            pos,
            burst_count,
            [0.9, 0.18, 0.1, 1.0],
            (80.0, 260.0),
        );
        if boom_radius > 0.0 {
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                Explosion {
                    timer: GTimer::from_seconds(0.05, TimerMode::Once),
                    radius: boom_radius,
                    damage: 6 + run.loop_count as i32 * 2,
                    team: Team::Enemy,
                    hits_player: true,
                    source: Some(DamageSource::enemy(e, enemy.kind)),
                },
                Pos(pos),
            ));
        }

        audio.play_hit(&mut cues);

        match enemy.kind {
            EnemyKind::YvBoss => {
                // GML `YVBoss/Destroy_0`: 3 golden-weapon pickups.
                let mut rng = rand::rng();
                for _ in 0..3 {
                    let weapon = crate::pickups::random_gold_weapon_fallback(&mut rng);
                    crate::pickups::spawn_pickup(
                        &mut commands,
                        &catalog,
                        crate::comps_b::PickupKind::Weapon(weapon),
                        pos + glam::Vec2::new(
                            rng.random_range(-16.0..16.0),
                            rng.random_range(-16.0..16.0),
                        ),
                        0,
                        false,
                    );
                }
            }
            EnemyKind::BigDog | EnemyKind::BigDogLoop => {
                // GML `ScrapBoss/Destroy_0`: `sleep(50)` plus
                // `BigDogExplo` (scattered explosions + ground flames).
                hitstop.trigger(0.05, 0.05);
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Explosion {
                        timer: GTimer::from_seconds(0.05, TimerMode::Once),
                        radius: 120.0,
                        damage: 8,
                        team: Team::Enemy,
                        hits_player: true,
                        source: Some(DamageSource::enemy(e, enemy.kind)),
                    },
                    Pos(pos),
                ));
                trauma.add(0.3);
            }
            EnemyKind::LilHunter | EnemyKind::LilHunterLoop => {
                // GML `LilHunter/Destroy_0`: `PortalClear`, an explosion,
                // and the 80-flame ring.
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    crate::comps_b::PortalClear {
                        timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                    },
                    Pos(pos),
                ));
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Explosion {
                        timer: GTimer::from_seconds(0.05, TimerMode::Once),
                        radius: 90.0,
                        damage: 6,
                        team: Team::Enemy,
                        hits_player: true,
                        source: Some(DamageSource::enemy(e, enemy.kind)),
                    },
                    Pos(pos),
                ));
                crate::boss_ai::lil_hunter_fire_ring(&mut commands, pos);
                // GML `LilHunter/Destroy_0:13-21`: the head skitters on
                // (`LilHunterDie` inherits team) plus the 80-`TrapFire`
                // ring. `scrOnPopoKill` has no port equivalent — skipped
                // (its music-cue side effects ride the audio layer).
                crate::enemies::spawn_lil_hunter_die(
                    &mut commands,
                    &mut cues,
                    pos,
                    *team,
                    Some(player_e),
                    Some(player_pos.0),
                );
                crate::enemies::spawn_lil_hunter_trapfire(&mut commands, pos, *team);
            }
            EnemyKind::ThroneII => {
                // GML `Nothing2/Destroy_0`: clear enemy projectiles.
                for (proj_e, team) in &mut enemy_shots {
                    if *team != Team::Player {
                        commands.entity(proj_e).despawn();
                    }
                }
                // GML `Nothing2Death`: 80-tick explosion pageant, then
                // 10 flung `BigRad`s (corpse parts + `InvisiWall` clear
                // + `BigPortal-1` ride existing flows).
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    crate::comps_b::ThroneVictory {
                        timer: GTimer::from_seconds(80.0 / 30.0, TimerMode::Once),
                        pos,
                        bursts: 0,
                    },
                ));
                // GML `SitDown`: the throne awaits its sitter.
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    crate::comps_b::SitZone,
                    Pos(pos),
                ));
                // GML `scrUnlocksThroneDefeat` + `scrUnlocksWinOrLoop`
                // crown share (golden-store half has no port equivalent).
                for (r, s) in crate::savedata_part::throne_defeat_skins(
                    &mut save,
                    race_state.race,
                    &player.mutations,
                    run.tottimer,
                    run.shots_fired,
                    run.weapons_picked,
                ) {
                    dirty.0 = true;
                    toast.show(&format!(
                        "{} {:?} UNLOCKED",
                        crate::savedata_part::character_def(r)
                            .name
                            .to_ascii_uppercase(),
                        s
                    ));
                }
                if player.crown != crate::data::CrownKind::None {
                    let gml = crate::savedata_part::crown_port_to_gml(player.crown as u8);
                    save.unlock_crown(race_state.race, gml);
                    dirty.0 = true;
                    // GML `scrUnlocksWinOrLoop`: looping with a crown
                    // earns CROWN LIFE.
                    if crate::savedata_part::unlock_achievement(&mut save, 22) {
                        toast.show("CROWN LIFE");
                    }
                }
            }
            EnemyKind::Ballguy => {
                // GML `Exploder/Destroy_0`: 8 `EnemyBullet2` at 4 px/tick
                // plus 8 `AcidStreak` at 8 px/tick on the same headings.
                // (Fixed art strip; the renderer falls back to a tinted
                // dot when the catalog lacks the strip, like bevy.)
                let anim = catalog
                    .def("sprBouncerBullet")
                    .map(|def| SpriteAnim::new("sprBouncerBullet", def));
                for i in 0..8 {
                    let ang = (i as f32) * std::f32::consts::TAU / 8.0;
                    let d = glam::Vec2::new(ang.cos(), ang.sin());
                    let mut bullet = commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Projectile {
                            damage: 2,
                            life: GTimer::from_seconds(2.0, TimerMode::Once),
                            radius: 4.0,
                            knockback: 120.0,
                            explosive: false,
                            source: Some(DamageSource::enemy(e, enemy.kind)),
                        },
                        Velocity(d * 120.0),
                        Pos(pos),
                    ));
                    if let Some(anim) = anim.clone() {
                        bullet.insert(anim);
                    }
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Projectile {
                            damage: 2,
                            life: GTimer::from_seconds(1.1, TimerMode::Once),
                            radius: 4.0,
                            knockback: 100.0,
                            explosive: false,
                            source: Some(DamageSource::enemy(e, enemy.kind)),
                        },
                        Velocity(d * 240.0),
                        Pos(pos),
                    ));
                }
                trauma.add(0.12);
            }
            EnemyKind::ProtoStatue => {
                // GML `ProtoStatue/Destroy_0`: rad past 24 opens the
                // type-3 vault portal, else the rad pays out in
                // `BigRad`s (10s) plus single rads, all flung.
                let rad = statue.map(|s| s.rad).unwrap_or(0);
                let mut rng = rand::rng();
                if rad > 24 {
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Portal,
                        PortalState {
                            kind: 3,
                            phase: PortalPhase::Spawn,
                            endgame: 100.0,
                            close: false,
                            anim: GTimer::from_seconds(0.25, TimerMode::Once),
                            loop_on: false,
                        },
                        Pos(pos),
                    ));
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        PortalShock {
                            timer: GTimer::from_seconds(2.0 / 30.0, TimerMode::Once),
                            radius: 72.0,
                        },
                        Pos(pos),
                    ));
                } else {
                    let mut left = rad;
                    while left > 15 {
                        left -= 10;
                        let dir = glam::Vec2::from_angle(
                            rng.random_range(0.0..std::f32::consts::TAU),
                        );
                        let re = crate::pickups::spawn_pickup(
                            &mut commands,
                            &catalog,
                            crate::comps_b::PickupKind::Rad(10),
                            pos,
                            0,
                            false,
                        );
                        commands.entity(re).insert(crate::comps_b::GroundPhysics {
                            vel: dir * rng.random_range(60.0..=300.0),
                            rotspeed: 0.0,
                        });
                    }
                    for _ in 0..left {
                        let dir = glam::Vec2::from_angle(
                            rng.random_range(0.0..std::f32::consts::TAU),
                        );
                        let re = crate::pickups::spawn_pickup(
                            &mut commands,
                            &catalog,
                            crate::comps_b::PickupKind::Rad(1),
                            pos,
                            0,
                            false,
                        );
                        commands.entity(re).insert(crate::comps_b::GroundPhysics {
                            vel: dir * rng.random_range(60.0..=300.0),
                            rotspeed: 0.0,
                        });
                    }
                }
            }
            EnemyKind::SuperFrog => {
                // GML `SuperFrog/Destroy_0`: 20 `EnemyBullet2` at 4 px/tick
                // plus `AcidStreak` at 8 px/tick stepping 18 degrees, 40
                // `ToxicGas` clouds, and a `PortalClear`.
                let mut rng = rand::rng();
                let mut ang = rng.random_range(0.0..std::f32::consts::TAU);
                for _ in 0..20 {
                    ang += std::f32::consts::TAU / 20.0;
                    let d = glam::Vec2::new(ang.cos(), ang.sin());
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Projectile {
                            damage: 2,
                            life: GTimer::from_seconds(2.0, TimerMode::Once),
                            radius: 4.0,
                            knockback: 120.0,
                            explosive: false,
                            source: Some(DamageSource::enemy(e, enemy.kind)),
                        },
                        Velocity(d * 120.0),
                        Pos(pos),
                    ));
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Projectile {
                            damage: 2,
                            life: GTimer::from_seconds(1.1, TimerMode::Once),
                            radius: 4.0,
                            knockback: 100.0,
                            explosive: false,
                            source: Some(DamageSource::enemy(e, enemy.kind)),
                        },
                        Velocity(d * 240.0),
                        Pos(pos),
                    ));
                }
                for _ in 0..40 {
                    let a = rng.random_range(0.0..std::f32::consts::TAU);
                    let off = glam::Vec2::from_angle(a) * rng.random_range(0.0..=48.0);
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        HazardCloud {
                            kind: HazardKind::Toxic,
                            radius: 16.0,
                            damage: 1,
                            timer: GTimer::from_seconds(6.0, TimerMode::Once),
                            tick: GTimer::from_seconds(0.5, TimerMode::Repeating),
                        },
                        Pos(pos + off),
                    ));
                }
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    crate::comps_b::PortalClear {
                        timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                    },
                    Pos(pos),
                ));
                trauma.add(0.3);
            }
            EnemyKind::ExploFreak => {
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Explosion {
                        timer: GTimer::from_seconds(0.05, TimerMode::Once),
                        radius: 46.0,
                        damage: 5,
                        team: Team::Enemy,
                        hits_player: true,
                        source: Some(DamageSource {
                            owner: e,
                            team: Team::Enemy,
                            hit_id: HitId::Explosion(WeaponId::NONE),
                            enemy_kind: Some(enemy.kind),
                        }),
                    },
                    Pos(pos),
                ));
                let mut rng = rand::rng();
                spawn_burst(
                    &mut commands,
                    &mut rng,
                    pos,
                    18,
                    [1.0, 0.5, 0.15, 1.0],
                    (60.0, 220.0),
                );
                trauma.add(0.12);
            }
            EnemyKind::BigMaggot => {
                let mut rng = rand::rng();
                for _ in 0..6 {
                    let ang = rng.random_range(0.0..std::f32::consts::TAU);
                    let off = glam::Vec2::new(ang.cos(), ang.sin()) * 12.0;
                    commands.spawn(PendingEnemySpawn {
                        kind: EnemyKind::Maggot,
                        pos: pos + off,
                        difficulty: 1.0,
                        loops: run.loop_count,
                    });
                }
                let mut rng = rand::rng();
                spawn_burst(
                    &mut commands,
                    &mut rng,
                    pos,
                    16,
                    [0.95, 0.5, 0.2, 1.0],
                    (60.0, 220.0),
                );
            }
            _ => {}
        }

        deaths.0.push(DeathEvent {
            pos,
            kind: enemy.kind,
            rad_drop: enemy.rad_drop,
            // GML `LilHunter/Destroy_0`: `scrDrop(200, 0)`.
            drop_chance: if matches!(
                enemy.kind,
                EnemyKind::LilHunter | EnemyKind::LilHunterLoop
            ) {
                200
            } else {
                enemy.drop_chance
            },
            weapon_chance: enemy.weapon_chance,
        });
    }
}

/// Death drops, run chained directly after [`resolve_enemy_deaths`].
/// Split only because bevy_ecs caps systems at 16 params: iterating the
/// [`DeathEvents`] record preserves bevy's per-death multiplicity exactly
/// (including repeated TriggerFingers scaling with several deaths).
pub fn resolve_death_drops(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    audio: Res<GameAudio>,
    mut rumble_queue: ResMut<Queue<RumbleRequest>>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut run: ResMut<Run>,
    mut slow_mo: ResMut<SlowMotion>,
    mut flash: ResMut<FlashWhite>,
    mut hitstop: ResMut<HitStop>,
    mut trauma: ResMut<Trauma>,
    mut loop_transition: ResMut<LoopTransition>,
    throne_room: Res<ThroneRoomState>,
    mut toast: ResMut<Toast>,
    mut deaths: ResMut<DeathEvents>,
    mut player_q: Query<(
        Entity,
        &Pos,
        &Player,
        &mut Health,
        &mut Inventory,
        &mut RaceState,
    )>,
    mut fire_q: Query<&mut FireCooldown, (With<Player>, Without<Enemy>)>,
) {
    // NOTE: `mut pinv` trips a bogus `unused_mut` (removing it trips
    // E0596 on the reborrow in the lucky-shot branch instead) — allowed,
    // not worked around.
    #[allow(unused_mut)]
    let Ok((_, player_pos, player, mut phealth, mut pinv, mut race_state)) =
        player_q.single_mut()
    else {
        deaths.0.clear();
        return;
    };
    // Bevy passes the live player position (not the death spot) to the
    // Throne campfire.
    let player_pos_now = player_pos.0;
    for event in deaths.0.drain(..) {
        match event.kind {
            EnemyKind::Throne => {
                if throne_room.loop_eligible {
                    crate::loop_transition::begin_throne_campfire(
                        &mut commands,
                        &mut loop_transition,
                        &mut toast,
                        &mut trauma,
                        player_pos_now,
                    );
                } else {
                    run.game_over = true;
                    toast.show("THE NUCLEAR THRONE");
                    flash_white(&mut flash, 0.2);
                    trauma.add(0.5);
                    slow_motion(&mut slow_mo, 0.25, 2.0);
                }
            }
            EnemyKind::ThroneII => {
                loop_transition.throne_ii_defeated();
                crate::loop_transition::mark_throne_ii_defeated(&mut toast, &mut trauma);
                // GML `Nothing2/Destroy_0`: 2 weapon caches.
                let mut rng = rand::rng();
                for _ in 0..2 {
                    let weapon = crate::pickups::random_weapon(&mut rng);
                    crate::pickups::spawn_pickup(
                        &mut commands,
                        &catalog,
                        crate::comps_b::PickupKind::Weapon(weapon),
                        event.pos
                            + glam::Vec2::new(
                                rng.random_range(-16.0..16.0),
                                rng.random_range(-16.0..16.0),
                            ),
                        0,
                        false,
                    );
                }
            }
            EnemyKind::FrogQueen => {
                // GML `FrogQueen/Destroy_0`: golden-weapon holders earn
                // the frog pistol.
                let golden = pinv.weapons.iter().any(|w| {
                    crate::weapon_runtime::weapon_id_name(*w)
                        .to_ascii_uppercase()
                        .contains("GOLDEN")
                });
                if golden {
                    crate::pickups::spawn_pickup(
                        &mut commands,
                        &catalog,
                        crate::comps_b::PickupKind::Weapon(crate::data::WeaponId(120)),
                        event.pos,
                        0,
                        false,
                    );
                }
            }
            _ => {}
        }

        let def = enemy_def(event.kind);
        let pos = event.pos;
        let enemy = Enemy {
            kind: event.kind,
            score: 0,
            touch_damage: 0,
            rad_drop: event.rad_drop,
            drop_chance: event.drop_chance,
            weapon_chance: event.weapon_chance,
        };
        let mut rng = rand::rng();
        if player.bloodlust && rng.random_range(0..15) == 0 {
            phealth.hp = (phealth.hp + 2).min(phealth.max);
        }
        if player.lucky_shot && rng.random_range(0..10) == 0 {
            give_ammo(&mut pinv, &player);
        }

        if player.mutations.contains(&MutationId::TriggerFingers)
            && let Ok(mut fc) = fire_q.single_mut()
        {
            // GML `enemy/Destroy_0:23`: the killer's gun shines.
            pinv.shine = 6.0;
            if !fc.timer.is_finished() {
                fc.timer = GTimer::from_seconds(fc.timer.remaining_secs() * 0.6, TimerMode::Once);
            }
            if fc.burst_left > 0 {
                fc.burst_timer =
                    GTimer::from_seconds(fc.burst_timer.remaining_secs() * 0.6, TimerMode::Once);
            }
        }

        if def.boss {
            hitstop.trigger(0.28, 0.18);
            slow_motion(&mut slow_mo, 0.35, 0.55);
            flash_white(&mut flash, 0.12);
            toast.show(&format!(
                "{} DEFEATED",
                enemy_def(enemy.kind).name.to_ascii_uppercase()
            ));
        }

        if def.boss {
            audio.play_boom(&mut cues);
            rumble(&mut rumble_queue, 0.8, 1.0, 0.4);
            slow_motion(&mut slow_mo, 0.35, 0.6);
            let melting_bonus = if race_state.race == RaceId::Melting {
                1
            } else {
                0
            };
            let blood_tax = if player.crown == CrownKind::Blood {
                1
            } else {
                0
            };
            spawn_rad_burst(
                &mut commands,
                &catalog,
                pos,
                ((enemy.rad_drop as u32).min(24) + melting_bonus).saturating_sub(blood_tax),
            );

            spawn_chest(
                &mut commands,
                &catalog,
                ChestKind::Weapon,
                pos + random_offset() * 3.0,
            );
            for _ in 0..2 {
                maybe_spawn_drop(
                    &mut commands,
                    &catalog,
                    pos,
                    enemy.drop_chance,
                    enemy.weapon_chance,
                    &player,
                    &pinv,
                    &phealth,
                    run.loop_count,
                );
            }
        } else {
            if player.chain_explosions {
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Explosion {
                        timer: GTimer::from_seconds(0.05, TimerMode::Once),
                        radius: 100.0,
                        damage: 3,
                        team: Team::Player,
                        hits_player: false,
                        source: None,
                    },
                    Pos(pos),
                ));
            }

            let melting_bonus = if race_state.race == RaceId::Melting {
                1
            } else {
                0
            };
            let blood_tax = if player.crown == CrownKind::Blood {
                1
            } else {
                0
            };
            spawn_rad_burst(
                &mut commands,
                &catalog,
                pos,
                (enemy.rad_drop as u32 + melting_bonus).saturating_sub(blood_tax),
            );
            maybe_spawn_drop(
                &mut commands,
                &catalog,
                pos,
                enemy.drop_chance,
                enemy.weapon_chance,
                &player,
                &pinv,
                &phealth,
                run.loop_count,
            );
            if matches!(enemy.kind, EnemyKind::DogGuardian | EnemyKind::WepMimic) {
                maybe_spawn_drop(
                    &mut commands,
                    &catalog,
                    pos,
                    enemy.drop_chance,
                    enemy.weapon_chance,
                    &player,
                    &pinv,
                    &phealth,
                    run.loop_count,
                );
            }
        }
    }
}

/// Kill unlocks + save writes, and the player-death branch (slice B3).

/// Projectile integration + collision. Ported from nt's
/// `move_projectiles` with positions as [`Pos`].
///
/// Render split: rotation writes (`tf.rotation` orienting sprites along
/// velocity) are dropped — the renderer orients from `Velocity`
/// directly. Everything else (bounce math, fuses, cascades) is
/// byte-identical, including call order into the removal cascade.
#[allow(clippy::too_many_arguments)]
pub fn move_projectiles(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut q: Query<
        (
            Entity,
            &Team,
            &mut Projectile,
            &mut Velocity,
            &mut Pos,
            Option<&mut BouncesLeft>,
            Option<&mut ShellWallBounce>,
            Option<&mut Sticky>,
            Option<&SpawnHazardOnDeath>,
            Option<&SplitOnDeath>,
            Option<&CustomExplosion>,
            Option<&DeploysSentry>,
            Option<&SpawnsWeaponPickup>,
            Option<&PlasmaBurst>,
            Option<&GrenadeFuse>,
        ),
        (Without<Prop>, Without<SlashProjectile>),
    >,
    mut aux: ParamSet<(
        Query<&mut DiscFlight>,
        Query<&mut PlasmaSize>,
        Query<&ProjectileFade>,
        Query<&SpriteAnim>,
        Query<&mut ShellBonus>,
    )>,
    mut props: Query<
        (
            Entity,
            &mut Prop,
            &Pos,
            Option<&PropDeathEffect>,
            Option<&PropSprites>,
            Option<&mut NextHurt>,
        ),
        With<Prop>,
    >,
    entrances: Query<&SecretEntrance>,
    snowmen: Query<&SnowmanAmbush>,
    gold_barrels: Query<&GoldBarrelDrop>,
    rad_chests: Query<&RadChestContainer>,
    frame: Res<CurrentFrame>,
    run: Res<Run>,
    mut secrets: ResMut<SecretTriggers>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
) {
    let dt = time.delta_secs;

    for (
        e,
        team,
        mut p,
        mut vel,
        mut tpos,
        bounces,
        shell_bounce,
        sticky,
        hazard,
        split,
        custom_explosion,
        deploys_sentry,
        spawn_pickup_spec,
        plasma_burst,
        grenade_fuse,
    ) in &mut q
    {
        p.life.tick(time.delta_secs);
        let fade = aux.p2().get(e).ok().copied();
        let already_faded = match (fade, aux.p3().get(e).ok()) {
            (Some(f), Some(a)) => a.path == f.0,
            _ => false,
        };
        if sticky.as_ref().is_some_and(|s| s.armed) {
            if p.life.just_finished() {
                if p.explosive && sticky.as_ref().is_some_and(|s| s.stuck_to.is_some()) {
                    let center = tpos.0;
                    let ang0 = rand::rng().random_range(0.0..std::f32::consts::TAU);
                    for k in 0..3 {
                        let ang = ang0 + k as f32 * 120.0_f32.to_radians();
                        let off = glam::Vec2::new(ang.cos(), ang.sin()) * 16.0;
                        spawn_explosion_with_source_radius(
                            &mut commands,
                            center + off,
                            p.damage,
                            p.source,
                            custom_explosion.map(|c| c.radius).unwrap_or(32.0),
                            *team,
                            true,
                        );
                    }
                    audio.play_boom(&mut cues);
                } else {
                    on_projectile_removed(
                        &mut commands,
                        &catalog,
                        tpos.0,
                        *team,
                        p.source,
                        hazard.copied(),
                        split.copied(),
                        vel.0,
                        p.explosive,
                        p.damage,
                        custom_explosion.copied(),
                        deploys_sentry.copied(),
                        spawn_pickup_spec.copied(),
                        plasma_burst.copied(),
                        fade,
                        already_faded,
                    );
                }
                commands.entity(e).despawn();
            }
            continue;
        }

        tpos.0 += vel.0 * dt;
        let pos = tpos.0;
        if split.is_some() && vel.0 == glam::Vec2::ZERO {
            on_projectile_removed(
                &mut commands,
                &catalog,
                pos,
                *team,
                p.source,
                hazard.copied(),
                split.copied(),
                vel.0,
                p.explosive,
                p.damage,
                custom_explosion.copied(),
                deploys_sentry.copied(),
                spawn_pickup_spec.copied(),
                plasma_burst.copied(),
                fade,
                already_faded,
            );
            commands.entity(e).despawn();
            continue;
        }
        if let Ok(mut d) = aux.p0().get_mut(e) {
            d.dist += dt * 30.0;
        }
        let out = pos.x.abs() > ARENA_W / 2.0 + 80.0 || pos.y.abs() > ARENA_H / 2.0 + 80.0;

        if p.life.just_finished() || out {
            on_projectile_removed(
                &mut commands,
                &catalog,
                pos,
                *team,
                p.source,
                hazard.copied(),
                split.copied(),
                vel.0,
                p.explosive,
                p.damage,
                custom_explosion.copied(),
                deploys_sentry.copied(),
                spawn_pickup_spec.copied(),
                plasma_burst.copied(),
                fade,
                already_faded,
            );
            commands.entity(e).despawn();
            continue;
        }

        let mut hit_normal: Option<glam::Vec2> = None;
        let mut hit_prop: Option<(Entity, glam::Vec2, bool, Option<PropDeathEffect>)> = None;

        if let Some(n) = arena_wall_normal(pos, p.radius, ARENA_W, ARENA_H) {
            hit_normal = Some(n);
        }

        for (prop_e, prop, prop_pos, death, _, _) in props.iter() {
            let center = prop_pos.0;
            let half = prop.size * 0.5;
            if let Some(n) = circle_aabb_normal(pos, p.radius, center, half) {
                hit_normal = Some(n);
                hit_prop = Some((prop_e, center, prop.destructible, death.copied()));
                break;
            }
        }

        if let Some(normal) = hit_normal {
            // Disc travel first (shared borrows end immediately) so the
            // plasma-size borrow below never overlaps it.
            let disc_dist = aux.p0().get(e).ok().map(|d| d.dist);
            // Plasma shrink (bevy order: prop damage with NO re-hit
            // gate, then unconditional shrink + burst + rollback, then
            // despawn at <= 0.5 — never bounces).
            if let Ok(mut ps) = aux.p1().get_mut(e) {
                if let Some((prop_e, center, true, _)) = hit_prop {
                    let dmg = ((p.damage as f32 * ps.0).floor() as i32).max(1);
                    damage_destructible_prop(
                        &mut commands,
                        &catalog,
                        &mut props,
                        &entrances,
                        &snowmen,
                        &gold_barrels,
                        &rad_chests,
                        &mut secrets,
                        &audio,
                        &mut cues,
                        prop_e,
                        center,
                        dmg,
                        p.source,
                        None,
                        run.loop_count,
                    );
                }
                ps.0 -= 0.1;
                let mut rng = rand::rng();
                crate::effects::spawn_burst(
                    &mut commands,
                    &mut rng,
                    pos,
                    3,
                    [0.7, 0.7, 0.7, 0.7],
                    (20.0, 60.0),
                );
                audio.play_hit(&mut cues);
                tpos.0 -= vel.0 * dt;
                if ps.0 <= 0.5 {
                    on_projectile_removed(
                        &mut commands,
                        &catalog,
                        pos,
                        *team,
                        p.source,
                        hazard.copied(),
                        split.copied(),
                        vel.0,
                        p.explosive,
                        p.damage,
                        custom_explosion.copied(),
                        deploys_sentry.copied(),
                        spawn_pickup_spec.copied(),
                        plasma_burst.copied(),
                        fade,
                        already_faded,
                    );
                    commands.entity(e).despawn();
                }
                continue;
            }
            // Disc: gated prop damage, then range despawn past 50 —
            // otherwise falls through to sticky/bounce.
            if aux.p0().get(e).is_ok() {
                if let Some((prop_e, center, true, _)) = hit_prop {
                    let gated = props
                        .get(prop_e)
                        .ok()
                        .and_then(|(_, _, _, _, _, nh)| nh)
                        .is_some_and(|nh| nh.0 > frame.0);
                    if !gated {
                        damage_destructible_prop(
                            &mut commands,
                            &catalog,
                            &mut props,
                            &entrances,
                            &snowmen,
                            &gold_barrels,
                            &rad_chests,
                            &mut secrets,
                            &audio,
                            &mut cues,
                            prop_e,
                            center,
                            p.damage,
                            p.source,
                            Some(frame.0 + 5),
                            run.loop_count,
                        );
                    }
                    audio.play_hit(&mut cues);
                    continue;
                }
                if disc_dist.is_some_and(|d| d > 50.0) {
                    let mut rng = rand::rng();
                    crate::effects::spawn_burst(
                        &mut commands,
                        &mut rng,
                        pos,
                        6,
                        [0.8, 0.8, 0.85, 1.0],
                        (40.0, 120.0),
                    );
                    audio.play_hit(&mut cues);
                    on_projectile_removed(
                        &mut commands,
                        &catalog,
                        pos,
                        *team,
                        p.source,
                        hazard.copied(),
                        split.copied(),
                        vel.0,
                        p.explosive,
                        p.damage,
                        custom_explosion.copied(),
                        deploys_sentry.copied(),
                        spawn_pickup_spec.copied(),
                        plasma_burst.copied(),
                        fade,
                        already_faded,
                    );
                    commands.entity(e).despawn();
                    continue;
                }
            }
            if let Some(mut sticky_inner) = sticky {
                if !p.explosive
                    && let Some((prop_e, center, true, _)) = hit_prop
                {
                    let hp_before = props
                        .get(prop_e)
                        .map(|(_, prop, _, _, _, _)| prop.hp)
                        .unwrap_or(0);
                    damage_destructible_prop(
                        &mut commands,
                        &catalog,
                        &mut props,
                        &entrances,
                        &snowmen,
                        &gold_barrels,
                        &rad_chests,
                        &mut secrets,
                        &audio,
                        &mut cues,
                        prop_e,
                        center,
                        p.damage,
                        p.source,
                        None,
                        run.loop_count,
                    );
                    if hp_before < ((p.damage as f32 * 0.5).ceil() as i32) {
                        continue;
                    }
                }
                sticky_inner.armed = true;
                if let Some((prop_e, center, _, _)) = hit_prop {
                    sticky_inner.stuck_to = Some(prop_e);
                    sticky_inner.offset = pos - center;
                } else {
                    sticky_inner.stuck_to = None;
                    sticky_inner.offset = glam::Vec2::ZERO;
                }
                let mut rng = rand::rng();
                crate::effects::spawn_burst(
                    &mut commands,
                    &mut rng,
                    pos,
                    2,
                    [0.7, 0.68, 0.6, 1.0],
                    (15.0, 45.0),
                );
                audio.play_hit(&mut cues);
                vel.0 = glam::Vec2::ZERO;
                continue;
            }

            let hit_destructible = matches!(hit_prop, Some((_, _, true, _)));
            if let Some(mut bounce) = bounces
                && bounce.0 > 0
                && !hit_destructible
            {
                bounce.0 -= 1;
                let factor = if grenade_fuse.is_some() { 0.6 } else { 0.8 };
                let mut bounced = bounce_velocity(vel.0, normal) * factor;
                if let Some(mut wb) = shell_bounce {
                    // GML shell wall: speed*0.8 (in `factor`) + wallbounce,
                    // capped, then wallbounce decays. HeavySlug re-arms its
                    // pointblank bonus while wallbounce > 2.
                    let sp = bounced.length();
                    let add = wb.add * 30.0;
                    let new_sp = if sp + add > wb.cap { wb.cap } else { sp + add };
                    if sp > 0.001 {
                        bounced = bounced.normalize() * new_sp;
                    }
                    if let Some((threshold, amount)) = wb.rearm
                        && wb.add > threshold
                        && let Ok(mut b) = aux.p4().get_mut(e)
                    {
                        b.bonus = amount;
                        b.timer = GTimer::from_seconds(2.0 / 30.0, TimerMode::Once);
                    }
                    wb.add *= wb.decay;
                }
                vel.0 = bounced;
                // Rotation orients the sprite along velocity; the
                // renderer derives it from `Velocity`, so no write here.
                tpos.0 += normal * (p.radius * 0.6 + 0.5);

                let mut rng = rand::rng();
                crate::effects::spawn_burst(
                    &mut commands,
                    &mut rng,
                    pos,
                    1,
                    [0.62, 0.60, 0.55, 1.0],
                    (12.0, 35.0),
                );
                if aux.p0().get(e).is_ok() {
                    crate::effects::spawn_burst(
                        &mut commands,
                        &mut rng,
                        pos,
                        3,
                        [0.85, 0.87, 0.95, 1.0],
                        (30.0, 90.0),
                    );
                    audio.play_hit(&mut cues);
                }

                if bounced.length() > 180.0 {
                    audio.play_hit(&mut cues);
                }
                continue;
            }

            if let Some((prop_e, center, true, _)) = hit_prop {
                let mut dmg = p.damage.max(1);
                if let Ok(b) = aux.p4().get(e)
                    && !b.timer.is_finished()
                {
                    dmg += b.bonus;
                }
                damage_destructible_prop(
                    &mut commands,
                    &catalog,
                    &mut props,
                    &entrances,
                    &snowmen,
                    &gold_barrels,
                    &rad_chests,
                    &mut secrets,
                    &audio,
                    &mut cues,
                    prop_e,
                    center,
                    dmg,
                    p.source,
                    None,
                    run.loop_count,
                );
            }

            if !p.explosive {
                let mut rng = rand::rng();
                crate::effects::spawn_burst(
                    &mut commands,
                    &mut rng,
                    pos,
                    1,
                    [0.62, 0.60, 0.55, 1.0],
                    (20.0, 55.0),
                );

                // Wall-impact dust (bevy shape: `has`-gate, oneshot
                // `1/(fps*0.7)` 0.45 s when stripped, static 0.4 s
                // otherwise. The port has no bare-PNG probe sim-side,
                // so the static marker always spawns and the renderer
                // draws it only when resolvable.)
                if let Some(def) = catalog.def("images/sprDust.png") {
                    let mut anim = SpriteAnim::oneshot("images/sprDust.png", def);
                    anim.timer = GTimer::from_seconds(
                        1.0 / (def.fps * 0.7).max(1.0),
                        TimerMode::Repeating,
                    );
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Pos(pos),
                        anim,
                        crate::comps_b::PickupLifetime {
                            timer: GTimer::from_seconds(0.45, TimerMode::Once),
                        },
                    ));
                } else {
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Pos(pos),
                        crate::comps_b::StaticFx {
                            path: "images/sprDust.png",
                        },
                        crate::comps_b::PickupLifetime {
                            timer: GTimer::from_seconds(0.4, TimerMode::Once),
                        },
                    ));
                }

                audio.play_hit(&mut cues);
            }
            on_projectile_removed(
                &mut commands,
                &catalog,
                pos,
                *team,
                p.source,
                hazard.copied(),
                split.copied(),
                vel.0,
                p.explosive,
                p.damage,
                custom_explosion.copied(),
                deploys_sentry.copied(),
                spawn_pickup_spec.copied(),
                plasma_burst.copied(),
                fade,
                already_faded,
            );
            commands.entity(e).despawn();
        }
    }
}

/// Chain-lightning arcs to nearby enemies (bevy parity, `Pos`-based).
/// Arc length/angle ride the marker for the renderer.
#[allow(clippy::too_many_arguments)]
fn chain_to_nearby_targets(
    mut commands: &mut Commands,
    targets: &mut Query<
        (
            Entity,
            &Pos,
            &Team,
            &Hitbox,
            &mut Health,
            Option<&mut Velocity>,
            Option<&Shield>,
            Option<&mut NextHurt>,
        ),
        Without<Projectile>,
    >,
    proj: &Projectile,
    first_target: Option<Entity>,
    first_pos: glam::Vec2,
    range: f32,
    jumps: u8,
    falloff: f32,
) {
    let Some(first_e) = first_target else {
        return;
    };

    let mut visited: Vec<Entity> = vec![first_e];
    let mut current_pos = first_pos;
    let mut damage = proj.damage.max(1);

    for _ in 0..jumps {
        let mut best: Option<(Entity, glam::Vec2, f32)> = None;
        for (target_e, target_pos, target_team, ..) in targets.iter() {
            if *target_team != Team::Enemy || visited.contains(&target_e) {
                continue;
            }
            let pos = target_pos.0;
            let d2 = current_pos.distance_squared(pos);
            if d2 > range * range {
                continue;
            }
            if best.map(|(_, _, bd)| d2 < bd).unwrap_or(true) {
                best = Some((target_e, pos, d2));
            }
        }

        let Some((next_e, next_pos, _)) = best else {
            break;
        };

        damage = ((damage as f32) * falloff).round().max(1.0) as i32;

        for (target_e, _, _, _, mut health, vel_opt, _, _) in targets.iter_mut() {
            if target_e != next_e {
                continue;
            }
            health.hp -= damage;
            if let Some(mut vel) = vel_opt {
                apply_knockback(
                    &mut vel.0,
                    (next_pos - current_pos).normalize_or_zero(),
                    proj.knockback * 0.5,
                );
            }
            HitFlash::apply(commands, target_e, [0.7, 0.95, 1.0, 1.0], 0.08);
            repame_fx::spawn_number(
                commands,
                next_pos.x,
                next_pos.y,
                damage.to_string(),
                [0.7, 0.95, 1.0, 1.0],
            );
            let mut rng = rand::rng();
            crate::effects::spawn_burst(
                &mut commands,
                &mut rng,
                (current_pos + next_pos) * 0.5,
                6,
                [0.75, 0.95, 1.0, 1.0],
                (30.0, 90.0),
            );

            let mid = (current_pos + next_pos) * 0.5;
            let dist = current_pos.distance(next_pos);
            let ang = (next_pos - current_pos).y.atan2((next_pos - current_pos).x);
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                LightningArc {
                    timer: GTimer::from_seconds(0.09, TimerMode::Once),
                    len: dist,
                    angle: ang,
                },
                Pos(mid),
            ));
            break;
        }

        visited.push(next_e);
        current_pos = next_pos;
    }
}

fn retaliate_sharp_teeth(
    commands: &mut Commands,
    damage: i32,
    center: glam::Vec2,
    frame: &CurrentFrame,
    targets: &mut Query<
        (
            Entity,
            &Pos,
            &Team,
            &Hitbox,
            &mut Health,
            Option<&mut Velocity>,
            Option<&Shield>,
            Option<&mut NextHurt>,
        ),
        Without<Projectile>,
    >,
) {
    for (ee, epos, team, _, mut health, _, _, nexthurt) in targets.iter_mut() {
        if *team != Team::Enemy {
            continue;
        }
        if epos.0.distance(center) > 900.0 {
            continue;
        }
        health.hp -= damage * 2;
        if let Some(mut nh) = nexthurt {
            nh.0 = frame.0 + 5;
        }
        HitFlash::apply(commands, ee, [1.0, 0.4, 0.4, 1.0], 0.12);
    }
}

/// Screen-feel sinks for explosion presentation (bundled so
/// `apply_explosions` stays within the system-param limit).
#[derive(bevy_ecs::system::SystemParam)]
pub struct BoomFeel<'w> {
    pub trauma: ResMut<'w, Trauma>,
    pub hitstop: ResMut<'w, HitStop>,
    pub chroma: ResMut<'w, ChromaticAberration>,
}

/// Projectile-vs-target hits. Ported whole from nt's `projectile_hits`
/// (`Pos` for `Transform`, catalog paths for handles, cues for audio).
#[allow(clippy::too_many_arguments)]
pub fn projectile_hits(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut trauma: ResMut<Trauma>,
    mut hitstop: ResMut<HitStop>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut secrets: ResMut<SecretTriggers>,
    mut last_damage: ResMut<LastDamageTaken>,
    player_state: Query<&Player, With<Player>>,
    mut projectiles: Query<
        (
            Entity,
            &mut Pos,
            &Team,
            &Projectile,
            &mut Velocity,
            Option<&mut PiercesLeft>,
            Option<&mut ProjectileHitSet>,
            Option<&mut Sticky>,
            Option<&ChainLightning>,
            Option<&SpawnHazardOnDeath>,
            Option<&SplitOnDeath>,
            Option<&CustomExplosion>,
            Option<&DeploysSentry>,
            Option<&SpawnsWeaponPickup>,
            Option<&PlasmaBurst>,
        ),
        (Without<Hitbox>, Without<SlashProjectile>),
    >,
    mut aux: ParamSet<(
        Query<&ShellBonus>,
        Query<&mut PlasmaSize>,
        Query<&DiscFlight>,
        Query<&ProjectileFade>,
        Query<&SpriteAnim>,
        Query<Entity, With<HitsAllTeams>>,
        Query<Entity, With<SpawnGrace>>,
    )>,
    frame: Res<CurrentFrame>,
    time: Res<SimTime>,
    mut targets: Query<
        (
            Entity,
            &Pos,
            &Team,
            &Hitbox,
            &mut Health,
            Option<&mut Velocity>,
            Option<&Shield>,
            Option<&mut NextHurt>,
        ),
        Without<Projectile>,
    >,
) {
    let player = player_state.single().ok();

    let hits_all_set: std::collections::HashSet<Entity> = aux.p5().iter().collect();
    let grace_set: std::collections::HashSet<Entity> = aux.p6().iter().collect();

    for (
        proj_e,
        mut proj_pos,
        proj_team,
        proj,
        mut proj_vel,
        pierce,
        mut hit_set,
        mut sticky,
        chain,
        hazard,
        split,
        custom_explosion,
        deploys_sentry,
        spawn_pickup_spec,
        plasma_burst,
    ) in projectiles.iter_mut()
    {
        let proj_fade = aux.p3().get(proj_e).ok().copied();
        let proj_already_faded = match (proj_fade, aux.p4().get(proj_e).ok()) {
            (Some(f), Some(a)) => a.path == f.0,
            _ => false,
        };
        if sticky.as_ref().is_some_and(|s| s.armed) {
            continue;
        }

        // Bevy reads/mutates the projectile `Transform` in place here
        // (notably the plasma victim rollback writes back); keep the
        // `&mut Pos` binding live — no Vec2 shadow copy.
        let mut hit = false;
        let mut damaged = false;
        let mut hit_player = false;
        let mut hit_pos = proj_pos.0;
        let mut hit_target = None::<Entity>;
        let mut stuck_bolt = false;
        let is_disc = aux.p2().get(proj_e).is_ok();
        let is_plasma = aux.p1().get(proj_e).is_ok();
        let mut passthrough = false;
        let mut plasma_died = false;

        for (target_e, target_pos, target_team, hitbox, mut health, vel_opt, shield, nexthurt) in
            targets.iter_mut()
        {
            let hits_all = hits_all_set.contains(&proj_e);
            if !hits_all && *target_team == *proj_team {
                continue;
            }

            if grace_set.contains(&proj_e)
                && let Some(src) = proj.source
                && target_e == src.owner
            {
                continue;
            }

            if !is_disc
                && !is_plasma
                && let Some(set) = hit_set.as_ref()
                && set.0.contains(&target_e)
            {
                continue;
            }

            let target_pos = target_pos.0;
            if proj_pos.0.distance(target_pos) > proj.radius + hitbox.radius {
                continue;
            }

            if is_disc
                && let Some(nh) = nexthurt.as_ref()
                && nh.0 > frame.0
            {
                continue;
            }

            if let Some(ref mut sticky) = sticky
                && !sticky.armed
                && proj.explosive
            {
                sticky.armed = true;
                sticky.stuck_to = Some(target_e);
                    sticky.offset = proj_pos.0 - target_pos;
                proj_vel.0 = glam::Vec2::ZERO;
                break;
            }

            hit = true;
            hit_pos = target_pos;
            hit_target = Some(target_e);

            if *target_team == Team::Player
                && let Some(shield) = shield
                && !shield.timer.is_finished()
            {
                let mut rng = rand::rng();
                crate::effects::spawn_burst(
                    &mut commands,
                    &mut rng,
                    target_pos,
                    8,
                    [0.3, 0.65, 1.0, 1.0],
                    (60.0, 160.0),
                );
                audio.play_hit(&mut cues);
                break;
            }

            if *target_team == Team::Player && !health.invuln.is_finished() {
                break;
            }

            let mut dmg = proj.damage;
            if let Ok(bonus) = aux.p0().get(proj_e) {
                if !bonus.timer.is_finished() {
                    dmg += bonus.bonus;
                }
            }
            if let Ok(ps) = aux.p1().get(proj_e) {
                dmg = ((dmg as f32 * ps.0).floor() as i32).max(1);
            }
            let hp_before = health.hp;
            health.hp -= dmg;
            damaged = true;

            if *target_team == Team::Enemy
                && let Some(mut nh) = nexthurt
            {
                nh.0 = frame.0 + 5;
            }

            if *target_team == Team::Player {
                health.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                hit_player = true;
                secrets.mark_damage_taken();
                last_damage.note_from_source(proj.source.as_ref());
                audio.play_hurt(&mut cues);
            } else {
                audio.play_hit(&mut cues);
            }

            if let Some(mut vel) = vel_opt {
                apply_knockback(&mut vel.0, proj_vel.0.normalize_or_zero(), proj.knockback);
            }

            HitFlash::apply(&mut commands, target_e, [1.0, 1.0, 1.0, 1.0], 0.1);
            trauma.add(0.08);
            repame_fx::spawn_number(
                &mut commands,
                target_pos.x,
                target_pos.y,
                proj.damage.to_string(),
                [1.0, 0.92, 0.35, 1.0],
            );

            if is_disc || is_plasma {
                passthrough = true;
            }
            if is_plasma {
                if let Ok(mut ps) = aux.p1().get_mut(proj_e) {
                    ps.0 -= 0.1;
                    if ps.0 <= 0.5 {
                        plasma_died = true;
                    }
                }
                let dt = time.delta_secs;
                proj_pos.0 -= proj_vel.0 * dt;
                let mut rng = rand::rng();
                crate::effects::spawn_burst(
                    &mut commands,
                    &mut rng,
                    target_pos,
                    2,
                    [0.6, 0.6, 0.62, 0.7],
                    (15.0, 55.0),
                );
            }

            {
                let hit_sprite = match *proj_team {
                    Team::Player => "images/sprBulletHit.png",
                    Team::Enemy => "images/sprEnemyBulletHit.png",
                };
                // Bevy shape: `has`-gate, oneshot `1/(fps*1.5)` 0.2 s
                // when stripped, static 0.15 s otherwise (same
                // renderer-only caveat as the dust fallback above).
                if let Some(def) = catalog.def(hit_sprite) {
                    let mut anim = SpriteAnim::oneshot(hit_sprite, def);
                    anim.timer = GTimer::from_seconds(
                        1.0 / (def.fps * 1.5).max(1.0),
                        TimerMode::Repeating,
                    );
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Pos(target_pos),
                        anim,
                        crate::comps_b::PickupLifetime {
                            timer: GTimer::from_seconds(0.2, TimerMode::Once),
                        },
                    ));
                } else {
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Pos(target_pos),
                        crate::comps_b::StaticFx { path: hit_sprite },
                        crate::comps_b::PickupLifetime {
                            timer: GTimer::from_seconds(0.15, TimerMode::Once),
                        },
                    ));
                }
            }

            if proj.explosive {
                hitstop.trigger(0.12, 0.09);
            }

            if let Some(ref mut sticky) = sticky
                && !sticky.armed
                && !proj.explosive
            {
                if hp_before >= ((dmg as f32 * 0.5).ceil() as i32) {
                    sticky.armed = true;
                    sticky.stuck_to = Some(target_e);
                sticky.offset = proj_pos.0 - target_pos;
                    proj_vel.0 = glam::Vec2::ZERO;
                    stuck_bolt = true;
                }
            }
            if passthrough {
                continue;
            }
            break;
        }

        if stuck_bolt {
            continue;
        }
        if !hit {
            continue;
        }

        if hit_player
            && let Some(p) = &player
            && p.sharp_teeth
        {
            retaliate_sharp_teeth(&mut commands, proj.damage, hit_pos, &frame, &mut targets);
        }

        if damaged && !passthrough && !is_disc && !is_plasma {
            if let Some(target_e) = hit_target {
                if let Some(ref mut set) = hit_set {
                    record_hit(&mut set.0, target_e);
                } else {
                    commands
                        .entity(proj_e)
                        .try_insert(ProjectileHitSet(vec![target_e]));
                }
            }
        }

        let terminal = |commands: &mut Commands| {
            on_projectile_removed(
                commands,
                &catalog,
                hit_pos,
                *proj_team,
                proj.source,
                hazard.copied(),
                split.copied(),
                proj_vel.0,
                proj.explosive,
                proj.damage,
                custom_explosion.copied(),
                deploys_sentry.copied(),
                spawn_pickup_spec.copied(),
                plasma_burst.copied(),
                proj_fade,
                proj_already_faded,
            );
            commands.entity(proj_e).despawn();
        };

        if plasma_died {
            terminal(&mut commands);
            continue;
        }
        if passthrough {
            continue;
        }

        if damaged && let Some(ref chain) = chain {
            chain_to_nearby_targets(
                &mut commands,
                &mut targets,
                proj,
                hit_target,
                hit_pos,
                chain.range,
                chain.jumps_left,
                chain.falloff,
            );
            terminal(&mut commands);
            continue;
        }

        let pierce_left_before = pierce.as_ref().map(|p| p.0);
        let (despawn, pierce_left) = should_despawn_after_hit(damaged, pierce_left_before);
        if let (Some(mut p), Some(left)) = (pierce, pierce_left) {
            p.0 = left;
        }

        if despawn {
            terminal(&mut commands);
        }
    }
}

// ---------------------------------------------------------------------------
// Projectile-tick battery. Ported from nt's `game/combat.rs` Combat-set
// systems (`tick_homing_projectiles` … `tick_shell_bonus`,
// `tick_hazard_clouds`, `apply_explosions`), `Pos` for `Transform`,
// catalog paths for handles, cue queues for audio.
// ---------------------------------------------------------------------------

/// Steer homing projectiles toward the nearest in-range target
/// (player shots pick enemies, enemy shots track the player).
/// Bevy `tick_homing_projectiles` parity.
pub fn tick_homing_projectiles(
    time: Res<SimTime>,
    mut q: Query<(&Team, &Pos, &mut Velocity, &Homing), With<Projectile>>,
    enemies: Query<&Pos, With<Enemy>>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
) {
    let dt = time.delta_secs;

    for (team, pos, mut vel, homing) in &mut q {
        let pos = pos.0;

        let target = match *team {
            Team::Player => {
                let mut best = None::<(f32, glam::Vec2)>;
                for epos in &enemies {
                    let epos = epos.0;
                    let d2 = pos.distance_squared(epos);
                    if d2 > homing.acquire_range * homing.acquire_range {
                        continue;
                    }
                    if best.map(|(bd, _)| d2 < bd).unwrap_or(true) {
                        best = Some((d2, epos));
                    }
                }
                best.map(|(_, p)| p)
            }
            Team::Enemy => player_q.single().ok().map(|p| p.0),
        };

        let Some(target_pos) = target else {
            continue;
        };

        let speed = vel.0.length();
        if speed <= 1e-4 {
            continue;
        }

        let current_dir = vel.0.normalize_or_zero();
        let desired_dir = (target_pos - pos).normalize_or_zero();
        let step = (homing.turn_rate * dt).clamp(0.0, 1.0);
        let new_dir = current_dir.lerp(desired_dir, step).normalize_or_zero();

        vel.0 = new_dir * speed;
    }
}

/// Armed sticky projectiles stop dead and ride their stuck target.
/// Bevy `tick_sticky_projectiles` parity.
pub fn tick_sticky_projectiles(
    mut q: Query<(&mut Pos, &mut Velocity, &Sticky), With<Projectile>>,
    targets: Query<&Pos, Without<Projectile>>,
) {
    for (mut pos, mut vel, sticky) in &mut q {
        if !sticky.armed {
            continue;
        }

        vel.0 = glam::Vec2::ZERO;

        if let Some(target) = sticky.stuck_to
            && let Ok(target_pos) = targets.get(target)
        {
            pos.0 = target_pos.0 + sticky.offset;
        }
    }
}

fn distance_to_segment(p: glam::Vec2, a: glam::Vec2, b: glam::Vec2) -> f32 {
    let ab = b - a;
    let denom = ab.length_squared();
    if denom <= 1e-6 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / denom).clamp(0.0, 1.0);
    let closest = a + ab * t;
    p.distance(closest)
}

/// Beam tick damage along the oriented segment (5-frame hit invuln on
/// the player, knockback, flash). Render split: no tint writes, the
/// `HitFlash` marker rides for the renderer.
pub fn tick_beams(
    time: Res<SimTime>,
    mut commands: Commands,
    mut secrets: ResMut<SecretTriggers>,
    mut last_damage: ResMut<LastDamageTaken>,
    mut beams: Query<(Entity, &Pos, &mut Beam)>,
    mut targets: Query<
        (
            Entity,
            &Pos,
            &Team,
            &mut Health,
            Option<&mut Velocity>,
        ),
        (Without<Beam>, Without<Projectile>),
    >,
) {
    for (beam_e, beam_pos, mut beam) in beams.iter_mut() {
        beam.timer.tick(time.delta_secs);
        let expired = beam.timer.just_finished();
        beam.tick.tick(time.delta_secs);

        if !expired && !beam.tick.just_finished() {
            continue;
        }

        let center = beam_pos.0;
        let half = beam.dir.normalize_or_zero() * (beam.length * 0.5);
        let a = center - half;
        let b = center + half;

        for (target_e, target_pos, target_team, mut health, mut vel) in &mut targets {
            if *target_team == beam.team {
                continue;
            }

            let p = target_pos.0;
            if distance_to_segment(p, a, b) > beam.width * 0.5 {
                continue;
            }

            // Hit invuln lasts 5 frames.
            if *target_team == Team::Player && !health.invuln.is_finished() {
                continue;
            }

            health.hp -= beam.damage;

            if let Some(ref mut vel) = vel {
                apply_knockback(&mut vel.0, beam.dir, beam.knockback);
            }

            if *target_team == Team::Player {
                health.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                secrets.mark_damage_taken();
                last_damage.note_from_source(beam.source.as_ref());
            }

            HitFlash::apply(&mut commands, target_e, [1.0, 1.0, 1.0, 1.0], 0.08);
        }

        if expired {
            commands.entity(beam_e).despawn();
        }
    }
}

/// Sentry turrets tick life, then fire player bullets at the nearest
/// in-range enemy on interval. Render split: no sprite handle here;
/// the renderer keys art off the `Projectile` marker.
pub fn tick_sentry_turrets(
    time: Res<SimTime>,
    mut commands: Commands,
    enemies: Query<&Pos, With<Enemy>>,
    mut sentries: Query<(Entity, &Pos, &mut SentryTurret)>,
) {
    for (entity, pos, mut sentry) in &mut sentries {
        sentry.life.tick(time.delta_secs);
        if sentry.life.just_finished() {
            commands.entity(entity).despawn();
            continue;
        }

        sentry.fire.tick(time.delta_secs);
        if !sentry.fire.just_finished() {
            continue;
        }

        let pos = pos.0;
        let mut best = None::<(f32, glam::Vec2)>;
        for epos in &enemies {
            let target = epos.0;
            let d2 = pos.distance_squared(target);
            if d2 > sentry.range * sentry.range {
                continue;
            }
            if best.map(|(bd, _)| d2 < bd).unwrap_or(true) {
                best = Some((d2, target));
            }
        }

        let Some((_, target)) = best else {
            continue;
        };

        let dir = (target - pos).normalize_or_zero();

        commands.spawn((
            GameCleanup,
            LevelCleanup,
            Team::Player,
            Projectile {
                damage: sentry.projectile_damage,
                life: GTimer::from_seconds(0.9, TimerMode::Once),
                radius: 4.0,
                knockback: 24.0,
                explosive: false,
                source: None,
            },
            Velocity(dir * sentry.projectile_speed),
            Pos(pos),
        ));
    }
}

/// Expire spawn-grace markers (bevy `tick_spawn_grace` parity).
pub fn tick_spawn_grace(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut SpawnGrace)>,
) {
    for (e, mut g) in &mut q {
        g.0.tick(time.delta_secs);
        if g.0.is_finished() {
            commands.entity(e).remove::<SpawnGrace>();
        }
    }
}

/// Flame trails drip hazard clouds on interval.
pub fn tick_flame_trails(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(&Pos, &Team, &mut FlameTrail), With<Projectile>>,
) {
    for (pos, team, mut trail) in &mut q {
        trail.timer.tick(time.delta_secs);
        if !trail.timer.just_finished() {
            continue;
        }

        spawn_hazard_cloud(&mut commands, pos.0, *team, trail.spec);
    }
}

/// Lightning arcs expire on their timer. Render split: the bevy alpha
/// fade on `Sprite.color` is renderer-owned; the sim only drains liveness.
pub fn tick_lightning_arcs(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut LightningArc)>,
) {
    for (e, mut arc) in &mut q {
        arc.timer.tick(time.delta_secs);
        if arc.timer.just_finished() {
            commands.entity(e).despawn();
        }
    }
}

/// GML friction on projectiles carrying `ProjectileFriction`.
pub fn tick_projectile_friction(
    time: Res<SimTime>,
    mut q: Query<(&mut Velocity, &ProjectileFriction)>,
) {
    for (mut vel, friction) in &mut q {
        crate::comps_a::apply_gml_friction(&mut vel.0, friction.0, time.delta_secs);
    }
}

/// GML shell slowdown deaths: Bullet2/Slug/HeavySlug/HyperSlug/UltraShell
/// swap to their `spr_fade` at speed < 6 px/step and Other_7 destroys the
/// instance when the fade anim ends, so shorten life to the fade length.
/// FlameShell instead dies outright under 5 px/step with no fade (its
/// Destroy spawns the Flame, handled by the normal life-end path).
/// Render split: sprite image/rect writes are renderer-owned; the sim
/// swaps the `SpriteAnim` path marker and shortens life.
pub fn tick_bullet2_fade(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut q: Query<
        (
            Entity,
            &Velocity,
            Option<&SpriteAnim>,
            &mut Projectile,
            Option<&ProjectileFade>,
            Option<&FlameShellSlowDeath>,
        ),
        (
            With<Projectile>,
            With<ShellWallBounce>,
            Without<SlashProjectile>,
        ),
    >,
) {
    for (e, vel, anim_opt, mut proj, fade, flame) in &mut q {
        if flame.is_some() {
            if vel.0.length() < 150.0 {
                proj.life = GTimer::from_seconds(0.0, TimerMode::Once);
            }
            continue;
        }
        if vel.0.length() >= 180.0 {
            continue;
        }
        let Some(fade_path) = fade.map(|f| f.0) else {
            continue;
        };
        if anim_opt.as_ref().is_some_and(|a| a.path == fade_path)
            || catalog.def(fade_path).is_none()
        {
            continue;
        }
        if let Some(def) = catalog.def(fade_path) {
            let fade_len = def.frames as f32 / def.fps.max(1.0);
            if let Some(mut current) = anim_opt.cloned() {
                current.set_path(fade_path, def, false);
                commands.entity(e).insert(current);
            } else {
                commands
                    .entity(e)
                    .insert(SpriteAnim::new(fade_path, def));
            }
            proj.life = GTimer::from_seconds(fade_len.max(0.1), TimerMode::Once);
        }
    }
}

// Grenade fuse flips friction at 6 ticks.
pub fn tick_grenade_fuse(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(
        &mut GrenadeFuse,
        &mut ProjectileFriction,
        &Pos,
    )>,
) {
    for (mut fuse, mut friction, pos) in &mut q {
        if fuse.friction_switched {
            continue;
        }
        fuse.alarm1.tick(time.delta_secs);
        if !fuse.alarm1.just_finished() {
            continue;
        }
        friction.0 = 0.4;
        fuse.friction_switched = true;
        if !fuse.smoke_armed {
            fuse.smoke_armed = true;

            let mut rng = rand::rng();
            spawn_burst(
                &mut commands,
                &mut rng,
                pos.0,
                4,
                [0.55, 0.55, 0.55, 0.6],
                (12.0, 65.0),
            );
        }
    }
}

/// Shell point-blank bonus window drains on its timer.
pub fn tick_shell_bonus(time: Res<SimTime>, mut q: Query<&mut ShellBonus>) {
    for mut bonus in &mut q {
        bonus.timer.tick(time.delta_secs);
        if bonus.timer.is_finished() {
            bonus.bonus = 0;
        }
    }
}

/// Hit-effect oneshots (non-pickup `PickupLifetime` carriers: bullet-hit
/// FX, dust, fades) expire on their timer. Render split: the bevy alpha
/// fade in the last 0.12 s is renderer-owned.
pub fn tick_hit_effects(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut PickupLifetime), Without<Pickup>>,
) {
    for (e, mut lt) in &mut q {
        lt.timer.tick(time.delta_secs);
        if lt.timer.just_finished() {
            commands.entity(e).despawn();
        }
    }
}

/// GML slash hitbox: oriented sprite bbox, not a circle. True when circle
/// (c, r) touches the forward segment, padded by the sprite half-height.
fn slash_touches(
    pos: glam::Vec2,
    dir: glam::Vec2,
    reach: f32,
    back: f32,
    half_width: f32,
    c: glam::Vec2,
    r: f32,
) -> bool {
    let rel = c - pos;
    let along = rel.dot(dir);
    if along < -back - r || along > reach + r {
        return false;
    }
    let side = (rel - dir * along).length();
    side <= half_width + r
}

/// Segment-vs-AABB variant for props/walls: closest point on the slash
/// segment to the box center must land inside the padded box.
#[allow(clippy::too_many_arguments)]
fn slash_hits_aabb(
    pos: glam::Vec2,
    dir: glam::Vec2,
    reach: f32,
    back: f32,
    half_width: f32,
    center: glam::Vec2,
    half: glam::Vec2,
) -> bool {
    let a = pos - dir * back;
    let b = pos + dir * reach;
    let ab = b - a;
    let t = ((center - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
    let closest = a + ab * t;
    (closest.x - center.x).abs() <= half.x + half_width
        && (closest.y - center.y).abs() <= half.y + half_width
}

/// GML Slash/Shank projectiles (melee). Friction 0.1 slide, pierce via
/// nexthurt (no despawn on hit), shank passes walls, slash stops with
/// MeleeHitWall + shake damage/3 once, deflects enemy bullets (typ 1),
/// destroys typ 2 / redirects grenades, Blood/Lightning/Hammer extras.
/// Life end = anim end (Other_7 destroy); BloodSlash misses self-hit 1.
///
/// Render split: bevy oriented `Transform.rotation` along velocity and
/// spawned the MeleeHitWall sprite; here the slash direction derives
/// from `Velocity` directly (fallback +X once stopped) and the wall-hit
/// FX is skipped — trauma/audio/latch carry the sim effect.
#[allow(clippy::too_many_arguments)]
pub fn tick_slash_projectiles(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut trauma: ResMut<Trauma>,
    mut hitstop: ResMut<HitStop>,
    frame: Res<CurrentFrame>,
    mut slash_q: Query<
        (
            Entity,
            &mut Projectile,
            &mut Velocity,
            &mut Pos,
            &mut SlashProjectile,
        ),
        (
            With<SlashProjectile>,
            With<Team>,
            Without<Enemy>,
            Without<Prop>,
            Without<WallTile>,
        ),
    >,
    mut enemies: Query<
        (
            Entity,
            &Pos,
            &Team,
            &Hitbox,
            &mut Health,
            Option<&mut Velocity>,
            Option<&mut NextHurt>,
        ),
        (
            With<Enemy>,
            Without<Prop>,
            Without<SlashProjectile>,
            Without<Projectile>,
        ),
    >,
    mut props: Query<
        (
            Entity,
            &mut Prop,
            &Pos,
            Option<&PropDeathEffect>,
            Option<&PropSprites>,
            Option<&mut NextHurt>,
            Option<&SecretEntrance>,
        ),
        (With<Prop>, Without<Enemy>, Without<SlashProjectile>),
    >,
    mut eproj: Query<
        (
            Entity,
            &Pos,
            &mut Velocity,
            &mut Team,
            &Projectile,
            Option<&ProjectileTyp>,
            Option<&GrenadeFuse>,
            Option<&ProjectileFade>,
        ),
        (With<Projectile>, Without<SlashProjectile>),
    >,
    walls: Query<
        (Entity, &WallCell, &Pos),
        (
            With<WallTile>,
            Without<SlashProjectile>,
            Without<Projectile>,
        ),
    >,
    mut secrets: ResMut<SecretTriggers>,
    mut all_health: Query<&mut Health, Without<Enemy>>,
) {
    let dt = time.delta_secs;
    for (e, mut proj, mut vel, mut tpos, mut slash) in &mut slash_q {
        proj.life.tick(time.delta_secs);
        let slash_team = Team::Player;
        // Bevy oriented the sprite along velocity (`tf.rotation`) and
        // read the slash direction back from it, freezing when stopped;
        // `slash.dir` latches that law without a rotation channel.
        if vel.0.length_squared() > 1e-6 {
            slash.dir = vel.0.normalize_or_zero();
        }
        let slash_dir = slash.dir;
        let slash_ang = slash_dir.y.atan2(slash_dir.x);
        tpos.0 += vel.0 * dt;
        let pos = tpos.0;

        for (pe, ppos, mut pvel, mut pteam, pproj, ptyp, pfuse, pfade) in &mut eproj {
            let same_team = *pteam == slash_team;
            let is_grenade = pfuse.is_some();
            if same_team && !is_grenade {
                continue;
            }
            let typ = ptyp.map(|t| t.0).unwrap_or(1);
            if typ == 0 {
                continue;
            }
            let ppos = ppos.0;
            if !slash_touches(
                pos,
                slash_dir,
                slash.reach,
                slash.back,
                slash.half_width,
                ppos,
                pproj.radius,
            ) {
                continue;
            }
            if slash.shank || typ == 2 {
                if let Some(f) = pfade {
                    let ang = if pvel.0.length_squared() > 0.0 {
                        pvel.0.y.atan2(pvel.0.x)
                    } else {
                        slash_ang
                    };
                    crate::spawns::spawn_bullet_fade_fx(&mut commands, &catalog, ppos, ang, f.0);
                }
                if pproj.explosive && !is_grenade {
                    let off = slash_dir * 12.0;
                    spawn_explosion_with_source_radius(
                        &mut commands,
                        ppos + off,
                        pproj.damage,
                        pproj.source,
                        32.0,
                        *pteam,
                        true,
                    );
                    spawn_explosion_with_source_radius(
                        &mut commands,
                        ppos - off,
                        pproj.damage,
                        pproj.source,
                        32.0,
                        *pteam,
                        true,
                    );
                    audio.play_boom(&mut cues);
                }
                commands.entity(pe).despawn();
                slash.hit = true;
            } else if is_grenade {
                pvel.0 = slash_dir * 360.0;
                commands.entity(pe).try_insert(ProjectileFriction(0.1));
                let mut rng = rand::rng();
                spawn_burst(
                    &mut commands,
                    &mut rng,
                    ppos,
                    4,
                    [0.9, 0.9, 0.9, 1.0],
                    (40.0, 120.0),
                );
                audio.play_hit(&mut cues);
                slash.hit = true;
            } else {
                *pteam = slash_team;
                pvel.0 = slash_dir * pvel.0.length();
                let mut rng = rand::rng();
                spawn_burst(
                    &mut commands,
                    &mut rng,
                    ppos,
                    4,
                    [1.0, 1.0, 1.0, 1.0],
                    (40.0, 120.0),
                );
                audio.play_hit(&mut cues);
                slash.hit = true;
            }
        }

        for (ee, epos, eteam, ebox, mut ehealth, evel, nexthurt) in &mut enemies {
            if *eteam == slash_team {
                continue;
            }
            if let Some(nh) = nexthurt.as_ref()
                && nh.0 > frame.0
            {
                continue;
            }
            let epos = epos.0;
            if !slash_touches(
                pos,
                slash_dir,
                slash.reach,
                slash.back,
                slash.half_width,
                epos,
                ebox.radius,
            ) {
                continue;
            }
            ehealth.hp -= proj.damage;
            if let Some(mut nh) = nexthurt {
                nh.0 = frame.0 + 5;
            }
            if let Some(mut ev) = evel {
                apply_knockback(
                    &mut ev.0,
                    (epos - pos).normalize_or_zero(),
                    proj.knockback,
                );
            }
            HitFlash::apply(&mut commands, ee, [1.0, 1.0, 1.0, 1.0], 0.12);
            repame_fx::spawn_number(
                &mut commands,
                epos.x,
                epos.y,
                proj.damage.to_string(),
                [1.0, 0.95, 0.6, 1.0],
            );
            audio.play_hit(&mut cues);
            slash.hit = true;
            if slash.guitar || slash.electric_guitar {
                audio.play_hit(&mut cues);
            }
            if slash.lightning {
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    LightningArc {
                        timer: GTimer::from_seconds(0.12, TimerMode::Once),
                        len: slash.reach,
                        angle: slash_ang,
                    },
                    Pos(epos),
                ));
            }
            if slash.blood {
                let mut rng = rand::rng();
                spawn_burst(
                    &mut commands,
                    &mut rng,
                    epos,
                    10,
                    [0.8, 0.1, 0.1, 1.0],
                    (60.0, 200.0),
                );
            }
        }

        let mut dead_props: Vec<(
            Entity,
            glam::Vec2,
            bool,
            Option<PropDeathEffect>,
            Option<PropSprites>,
            Option<crate::data::SecretTarget>,
        )> = Vec::new();
        for (pe, mut prop, ppos, death, sprites, nexthurt, entrance) in &mut props {
            if !prop.destructible {
                continue;
            }
            if let Some(nh) = nexthurt.as_ref()
                && nh.0 > frame.0
            {
                continue;
            }
            let center = ppos.0;
            let half = prop.size * 0.5;
            if !slash_hits_aabb(
                pos,
                slash_dir,
                slash.reach,
                slash.back,
                slash.half_width,
                center,
                half,
            ) {
                continue;
            }
            prop.hp -= proj.damage.max(1);
            if let Some(mut nh) = nexthurt {
                nh.0 = frame.0 + 5;
            }
            audio.play_hit(&mut cues);
            slash.hit = true;
            if prop.hp <= 0 {
                dead_props.push((
                    pe,
                    center,
                    prop.explosive,
                    death.copied(),
                    sprites.copied(),
                    entrance.map(|s| s.target),
                ));
            }
        }
        for (pe, center, explosive, death, sprites, entrance) in dead_props {
            if let Some(ps) = sprites {
                spawn_prop_corpse(&mut commands, &catalog, center, &ps);
            }
            spawn_prop_death_effect(&mut commands, center, death, explosive, proj.source);
            if let Some(target) = entrance {
                secrets.queue(target);
            }
            commands.entity(pe).try_despawn();
        }

        if !slash.shank && !slash.walled {
            let mut wall_hit: Option<(Option<Entity>, glam::Vec2, f32)> = None;
            for (we, _cell, wpos) in &walls {
                let wpos = wpos.0;
                let half = glam::Vec2::splat(8.0);
                if !slash_hits_aabb(
                    pos,
                    slash_dir,
                    slash.reach,
                    slash.back,
                    slash.half_width,
                    wpos,
                    half,
                ) {
                    continue;
                }
                wall_hit = Some((Some(we), wpos, slash_ang));
                break;
            }
            if wall_hit.is_none() && (pos.x.abs() > ARENA_W / 2.0 || pos.y.abs() > ARENA_H / 2.0) {
                wall_hit = Some((None, pos, slash_ang));
            }
            if let Some((we, wpos, wang)) = wall_hit {
                tpos.0 -= vel.0 * dt;
                if slash.hammer_wallbreak
                    && let Some(we) = we
                    && let Ok((_, cell, wpos2)) = walls.get(we)
                {
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        PendingWallBreak {
                            cell: (cell.0, cell.1),
                            pos: wpos2.0,
                            spawn_floor: true,
                        },
                    ));
                    hitstop.trigger(0.15, 0.05);
                }
                // MeleeHitWall sprite (bevy `sprMeleeHitWall` SwingFx
                // 0.3 s at the wall pos, `wang` rotation); shake + sting
                // + latch carry the sim effect.
                let hit_path = "images/sprMeleeHitWall.png";
                let mut he = commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    crate::comps_b::SwingFx {
                        timer: GTimer::from_seconds(0.3, TimerMode::Once),
                        angle: wang,
                    },
                    Pos(wpos),
                ));
                if let Some(def) = catalog.def(hit_path) {
                    he.insert(SpriteAnim::new(hit_path, def));
                }
                trauma.add((proj.damage as f32 / 3.0 / 20.0).clamp(0.1, 0.5));
                audio.play_hit(&mut cues);
                slash.walled = true;
                vel.0 = glam::Vec2::ZERO;
            }
        }

        if proj.life.just_finished() {
            if slash.blood && !slash.hit {
                if let Some(src) = proj.source
                    && let Ok(mut h) = all_health.get_mut(src.owner)
                {
                    h.hp -= 1;
                }
                audio.play_hit(&mut cues);
            }
            commands.entity(e).try_despawn();
        }
    }
}

/// Timed hazard clouds (fire/toxic): tick damage on opposing teams in
/// radius, player hits gated by invuln and booked to secrets/last-damage.
/// This is combat's own cloud tick (enemy clouds); the AbilityHazard
/// variant lives in `player_fire`.
pub fn tick_hazard_clouds(
    time: Res<SimTime>,
    mut commands: Commands,
    mut secrets: ResMut<SecretTriggers>,
    mut last_damage: ResMut<LastDamageTaken>,
    mut clouds: Query<(Entity, &Team, &Pos, &mut HazardCloud), Without<crate::comps_a::AbilityHazard>>,
    mut targets: Query<
        (Entity, &Pos, &Team, &mut Health),
        (Without<HazardCloud>, Without<Projectile>),
    >,
) {
    for (cloud_e, cloud_team, cloud_pos, mut cloud) in &mut clouds {
        cloud.timer.tick(time.delta_secs);
        cloud.tick.tick(time.delta_secs);

        if cloud.timer.just_finished() {
            commands.entity(cloud_e).despawn();
            continue;
        }
        if !cloud.tick.just_finished() {
            continue;
        }

        let pos = cloud_pos.0;
        for (_, target_pos, target_team, mut health) in &mut targets {
            if *target_team == *cloud_team {
                continue;
            }
            if target_pos.0.distance(pos) > cloud.radius {
                continue;
            }
            if *target_team == Team::Player && !health.invuln.is_finished() {
                continue;
            }

            health.hp -= cloud.damage;
            if *target_team == Team::Player {
                health.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);

                secrets.mark_damage_taken();
                let hid = match cloud.kind {
                    HazardKind::Toxic => HitId::Toxic,
                    HazardKind::Fire => HitId::Fire,
                };
                last_damage.note(Some(hid), None);
            }
        }
    }
}

/// GML `Nothing2Death` pageant: ~3 scattered `GreenExplosion`s every 8
/// ticks across 80 ticks, then 10 flung `BigRad`s (10 rads each).
pub fn tick_throne_victory(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut trauma: ResMut<Trauma>,
    mut mask: ResMut<FloorMask>,
    mut q: Query<(Entity, &mut crate::comps_b::ThroneVictory)>,
    invisi: Query<(Entity, &Pos), With<crate::comps_b::InvisiWall>>,
) {
    let dt = time.delta_secs;
    let mut rng = rand::rng();
    for (e, mut v) in &mut q {
        // GML clears the invisible walls with the death pageant (mask
        // cells go too, so the room opens up).
        if v.bursts == 0 {
            let dead: Vec<(Entity, (i32, i32))> = invisi
                .iter()
                .map(|(w, wpos)| {
                    let cell = (
                        (wpos.0.x / crate::comps_a::TILE).floor() as i32,
                        (wpos.0.y / crate::comps_a::TILE).floor() as i32,
                    );
                    (w, cell)
                })
                .collect();
            for (w, cell) in dead {
                mask.cells.remove(&cell);
                commands.entity(w).despawn();
            }
        }
        v.timer.tick(dt);
        let elapsed = 80.0 - v.timer.remaining_secs() * 30.0;
        let step = (elapsed / 8.0).floor() as u8;
        if step > v.bursts && v.bursts < 10 {
            v.bursts = step;
            for _ in 0..3 {
                let at = v.pos
                    + glam::Vec2::new(
                        rng.random_range(-64.0..64.0),
                        rng.random_range(-50.0..50.0),
                    );
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Explosion {
                        timer: GTimer::from_seconds(0.05, TimerMode::Once),
                        radius: 80.0,
                        damage: 12,
                        team: Team::Enemy,
                        hits_player: true,
                        source: None,
                    },
                    Pos(at),
                ));
            }
                trauma.add(0.12);
            }

        if v.timer.just_finished() {
            for _ in 0..10 {
                let ang = rng.random_range(200.0..=340.0_f32).to_radians();
                let dir = glam::Vec2::from_angle(ang);
                let re = crate::pickups::spawn_pickup(
                    &mut commands,
                    &catalog,
                    crate::comps_b::PickupKind::Rad(10),
                    v.pos,
                    0,
                    false,
                );
                commands.entity(re).insert(crate::comps_b::GroundPhysics {
                    vel: dir * rng.random_range(150.0..=300.0),
                    rotspeed: 0.0,
                });
            }
            commands.entity(e).despawn();
        }
    }
}

/// Timed area explosion: trauma/chroma/hitstop/boom, player-team damage
/// to enemies + destructible props (corpse/effect chain, secret
/// entrances, snowman ambushes, gold/rad drops) + wall breaks,
/// hits-player damage with Boiling Veins law, Death-crown chains.
/// Bevy `apply_explosions` parity (`Pos` for `Transform`, prop marker
/// options folded into one query to fit the 16-param system cap).
#[allow(clippy::too_many_arguments)]
pub fn apply_explosions(
    time: Res<SimTime>,
    mut commands: Commands,
    mut feel: BoomFeel,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    catalog: Res<repame_anim::AnimCatalog>,
    run: Res<Run>,
    mut secrets: ResMut<SecretTriggers>,
    mut q: Query<
        (Entity, &mut Explosion, &Pos),
        (Without<Enemy>, Without<Player>, Without<Prop>),
    >,
    mut enemies: Query<
        (Entity, &Pos, &mut Health, &Hitbox, Option<&mut Velocity>),
        (With<Enemy>, Without<Player>),
    >,
    mut player_q: Query<
        (
            Entity,
            &Pos,
            &mut Health,
            &Player,
            Option<&mut Velocity>,
        ),
        (With<Player>, Without<Enemy>),
    >,
    mut props: Query<
        (
            Entity,
            &mut Prop,
            &Pos,
            Option<&PropDeathEffect>,
            Option<&PropSprites>,
            Option<&SecretEntrance>,
            Option<&SnowmanAmbush>,
            Option<&GoldBarrelDrop>,
            Option<&RadChestContainer>,
        ),
        (With<Prop>, Without<Player>),
    >,
    walls: Query<(Entity, &WallCell, &Pos), With<WallTile>>,
    mut lingering_q: Query<&mut LingeringBlast>,
    mut last_damage: ResMut<LastDamageTaken>,
) {
    let death_crown = player_q
        .single()
        .map(|(_, _, _, p, _)| p.crown == CrownKind::Death)
        .unwrap_or(false);
    for (e, mut boom, pos) in &mut q {
        boom.timer.tick(time.delta_secs);
        let fused = boom.timer.just_finished();

        // Lingering state machine (bevy verbatim): without a lingering
        // companion the fuse tick is the only scan; with one, re-scan
        // every 1/30 s until the 0.75 s duration lapses.
        let mut ling_guard = lingering_q.get_mut(e).ok();
        let mut hit_opt: Option<&mut Vec<Entity>> = None;
        match ling_guard.as_mut() {
            Some(ling) => {
                if !fused {
                    ling.duration.tick(time.delta_secs);
                    ling.tick.tick(time.delta_secs);
                    if ling.duration.just_finished() {
                        commands.entity(e).despawn();
                        continue;
                    }
                    if !ling.tick.just_finished() {
                        continue;
                    }
                }
                hit_opt = Some(&mut ling.hit);
            }
            None if !fused => continue,
            None => {}
        }

        let pos = pos.0;
        if fused {
            feel.trauma.add(0.45);
            chromatic_pulse(&mut feel.chroma, 0.3);
            feel.hitstop.trigger(0.14, 0.1);
            let mut rng = rand::rng();
            spawn_burst(
                &mut commands,
                &mut rng,
                pos,
                32,
                [1.0, 0.4, 0.1, 1.0],
                (130.0, 400.0),
            );
            spawn_burst(
                &mut commands,
                &mut rng,
                pos,
                16,
                [1.0, 0.9, 0.5, 1.0],
                (60.0, 220.0),
            );
            audio.play_boom(&mut cues);
        }

        if boom.team == Team::Player {
            for (ee, epos, mut health, hitbox, vel_opt) in &mut enemies {
                let target_pos = epos.0;
                if target_pos.distance(pos) >= boom.radius + hitbox.radius {
                    continue;
                }
                if hit_opt.as_ref().is_some_and(|hit| hit.contains(&ee)) {
                    continue;
                }
                health.hp -= boom.damage;
                if let Some(mut vel) = vel_opt {
                    if vel.0.length() < 480.0 {
                        vel.0 += (target_pos - pos).normalize_or_zero() * 180.0;
                        if vel.0.length() > 480.0 {
                            vel.0 = vel.0.normalize_or_zero() * 480.0;
                        }
                    }
                }
                HitFlash::apply(&mut commands, ee, [1.0, 1.0, 1.0, 1.0], 0.12);
                repame_fx::spawn_number(
                    &mut commands,
                    epos.0.x,
                    epos.0.y,
                    boom.damage.to_string(),
                    [1.0, 0.6, 0.2, 1.0],
                );
                if let Some(hit) = hit_opt.as_mut() {
                    hit.push(ee);
                }
            }
            let mut destroyed_props = Vec::new();
            for (prop_e, mut prop, ppos, death_effect, sprites, entrance, snowman, gold, rad) in
                &mut props
            {
                if !prop.destructible {
                    continue;
                }

                let center = ppos.0;
                let half = prop.size / 2.0;
                let closest = glam::Vec2::new(
                    pos.x.clamp(center.x - half.x, center.x + half.x),
                    pos.y.clamp(center.y - half.y, center.y + half.y),
                );
                if fused && pos.distance(closest) < boom.radius {
                    prop.hp -= boom.damage.max(1);
                    if prop.hp <= 0 {
                        destroyed_props.push((
                            prop_e,
                            center,
                            prop.explosive,
                            death_effect.copied(),
                            sprites.copied(),
                            entrance.map(|s| s.target),
                            snowman.is_some(),
                            gold.is_some(),
                            rad.is_some(),
                        ));
                    }
                }
            }

            for (
                prop_e,
                center,
                legacy_explosive,
                death_effect,
                sprites,
                entrance,
                is_snowman,
                is_gold,
                is_rad,
            ) in destroyed_props
            {
                if let Some(ps) = sprites {
                    spawn_prop_corpse(&mut commands, &catalog, center, &ps);
                }
                spawn_prop_death_effect(
                    &mut commands,
                    center,
                    death_effect,
                    legacy_explosive,
                    boom.source,
                );

                if let Some(target) = entrance {
                    secrets.queue(target);
                }

                if is_snowman {
                    let mut rng = rand::rng();
                    for _ in 0..3 {
                        commands.spawn(PendingEnemySpawn {
                            kind: EnemyKind::Bandit,
                            pos: center
                                + glam::Vec2::new(
                                    rng.random_range(-4.0..4.0),
                                    rng.random_range(-4.0..4.0),
                                ),
                            difficulty: 1.0,
                            loops: run.loop_count,
                        });
                    }
                    for _ in 0..6 {
                        spawn_rad(&mut commands, &catalog, center, 1);
                    }
                }

                if is_gold {
                    let weapon = random_gold_weapon_fallback(&mut rand::rng());
                    spawn_pickup(
                        &mut commands,
                        &catalog,
                        crate::comps_b::PickupKind::Weapon(weapon),
                        center + glam::Vec2::new(0.0, -14.0),
                        0,
                        false,
                    );
                }

                if is_rad {
                    let mut rng = rand::rng();
                    for _ in 0..25 {
                        let ang = rng.random_range(0.0..std::f32::consts::TAU);
                        let d = rng.random_range(6.0..26.0);
                        spawn_pickup(
                            &mut commands,
                            &catalog,
                            crate::comps_b::PickupKind::Rad(1),
                            center + glam::Vec2::new(ang.cos() * d, ang.sin() * d),
                            0,
                            false,
                        );
                    }
                }

                commands.entity(prop_e).try_despawn();
            }

            for (_, cell, wpos) in &walls {
                if fused && wpos.0.distance(pos) < boom.radius * 0.85 {
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        PendingWallBreak {
                            cell: (cell.0, cell.1),
                            pos: wpos.0,
                            spawn_floor: true,
                        },
                    ));
                }
            }
        }

        if boom.hits_player
            && let Ok((player_e, ppos, mut health, player, vel_opt)) = player_q.single_mut()
            && ppos.0.distance(pos) < boom.radius + PLAYER_RADIUS
            && health.invuln.is_finished()
            && !hit_opt.as_ref().is_some_and(|hit| hit.contains(&player_e))
        {
            let mut dmg = boom.damage;
            if player.boiling_veins {
                let floor = player.veins_threshold;
                dmg = if health.hp - dmg < floor {
                    (health.hp - floor).max(0)
                } else {
                    dmg
                };
            }
            health.hp -= dmg;
            if let Some(mut vel) = vel_opt {
                let away = (ppos.0 - pos).normalize_or_zero();
                if vel.0.length() < 480.0 {
                    vel.0 += away * 180.0;
                    if vel.0.length() > 480.0 {
                        vel.0 = vel.0.normalize_or_zero() * 480.0;
                    }
                }
            }
            health.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
            secrets.mark_damage_taken();
            last_damage.note_from_source(boom.source.as_ref());
            HitFlash::apply(&mut commands, player_e, [1.0, 0.3, 0.2, 1.0], 0.15);
            audio.play_hurt(&mut cues);
            if let Some(hit) = hit_opt.as_mut() {
                hit.push(player_e);
            }
        }

        if death_crown {
            let mut rng = rand::rng();
            for _ in 0..3 {
                let ang = rng.random_range(0.0..std::f32::consts::TAU);
                let at = pos + glam::Vec2::new(ang.cos(), ang.sin()) * 12.0;
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Explosion {
                        timer: GTimer::from_seconds(0.04, TimerMode::Once),
                        radius: 46.0,
                        damage: 3,
                        team: boom.team,
                        hits_player: boom.hits_player,
                        source: boom.source,
                    },
                    Pos(at),
                ));
            }
        }

        if hit_opt.is_none() {
            commands.entity(e).despawn();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repame_anim::AnimCatalog;

    fn aura_world() -> bevy_ecs::prelude::World {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        world
    }

    fn spawn_player(world: &mut bevy_ecs::prelude::World, guts: bool) -> bevy_ecs::prelude::Entity {
        let mut player = Player::default();
        player.gamma_guts = guts;
        world
            .spawn((player, Pos(glam::Vec2::ZERO), Velocity(glam::Vec2::ZERO)))
            .id()
    }

    fn spawn_enemy(
        world: &mut bevy_ecs::prelude::World,
        at: glam::Vec2,
        hp: i32,
    ) -> bevy_ecs::prelude::Entity {
        world
            .spawn((
                crate::comps_b::Enemy {
                    kind: crate::data::EnemyKind::Bandit,
                    score: 5,
                    touch_damage: 1,
                    rad_drop: 0,
                    drop_chance: 0,
                    weapon_chance: 0,
                },
                Pos(at),
                Health {
                    hp,
                    max: hp,
                    invuln: GTimer::disarmed(),
                },
                Velocity(glam::Vec2::ZERO),
            ))
            .id()
    }

    #[test]
    fn aura_hits_in_range_with_lapsed_invuln() {
        let mut world = aura_world();
        spawn_player(&mut world, true);
        let close = spawn_enemy(&mut world, glam::Vec2::new(30.0, 0.0), 20);
        let far = spawn_enemy(&mut world, glam::Vec2::new(500.0, 0.0), 20);
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(gamma_guts_aura);
        schedule.run(&mut world);
        assert_eq!(world.get::<Health>(close).unwrap().hp, 14);
        assert_eq!(world.get::<Health>(far).unwrap().hp, 20);
        assert!(!world.get::<Health>(close).unwrap().invuln.is_finished());
        assert!(world.get::<HitFlash>(close).is_some());
        assert!(world.get::<HitFlash>(far).is_none());
    }

    #[test]
    fn aura_skips_without_guts_or_on_invuln() {
        let mut world = aura_world();
        spawn_player(&mut world, false);
        let e = spawn_enemy(&mut world, glam::Vec2::new(10.0, 0.0), 20);
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(gamma_guts_aura);
        schedule.run(&mut world);
        assert_eq!(world.get::<Health>(e).unwrap().hp, 20);
    }

    #[test]
    fn flash_markers_expire() {
        let mut world = aura_world();
        let e = world.spawn_empty().id();
        world
            .entity_mut(e)
            .insert(HitFlash::new([1.0, 0.0, 0.0, 1.0], 0.1));
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(tick_hit_flash);
        schedule.run(&mut world);
        assert!(world.get::<HitFlash>(e).is_some(), "0.1 s not lapsed");
        for _ in 0..3 {
            schedule.run(&mut world);
        }
        assert!(world.get::<HitFlash>(e).is_none());
    }

    #[test]
    fn knockback_sets_directed_force() {
        let mut v = glam::Vec2::new(400.0, 0.0);
        apply_knockback(&mut v, glam::Vec2::new(0.0, 2.0), 120.0);
        assert_eq!(v, glam::Vec2::new(0.0, 120.0));
    }

    fn contact_world() -> (
        bevy_ecs::prelude::World,
        bevy_ecs::prelude::Entity,
        bevy_ecs::prelude::Entity,
    ) {
        let mut world = bevy_ecs::prelude::World::new();
        world.init_resource::<Trauma>();
        world.init_resource::<FlashWhite>();
        world.init_resource::<GameAudio>();
        world.init_resource::<Queue<RumbleRequest>>();
        world.init_resource::<Queue<AudioCue>>();
        world.init_resource::<SecretTriggers>();
        world.init_resource::<LastDamageTaken>();
        let player = world
            .spawn((
                Player::default(),
                Pos(glam::Vec2::ZERO),
                Velocity(glam::Vec2::ZERO),
                Health {
                    hp: 20,
                    max: 20,
                    invuln: GTimer::disarmed(),
                },
            ))
            .id();
        let enemy = world
            .spawn((
                Enemy {
                    kind: crate::data::EnemyKind::Bandit,
                    score: 5,
                    touch_damage: 3,
                    rad_drop: 0,
                    drop_chance: 0,
                    weapon_chance: 0,
                },
                Pos(glam::Vec2::new(10.0, 0.0)),
                Velocity(glam::Vec2::ZERO),
                EnemyBrain {
                    speed: 0.0,
                    accel: 0.0,
                    preferred_range: 0.0,
                    shoot_range: 0.0,
                    attack: GTimer::disarmed(),
                    burst_left: 0,
                    burst_timer: GTimer::disarmed(),
                    dash: 0.0,
                    strafe_dir: 0.0,
                    strafe_timer: GTimer::disarmed(),
                    melee: GTimer::disarmed(),
                    walk: 0.0,
                    ammo: 0,
                    slash_delay: 0.0,
                    gunangle: 0.0,
                },
                Health {
                    hp: 10,
                    max: 10,
                    invuln: GTimer::disarmed(),
                },
                crate::comps_a::Hitbox { radius: 8.0 },
            ))
            .id();
        (world, player, enemy)
    }

    #[test]
    fn contact_hurts_and_fires_every_sink() {
        let (mut world, player, enemy) = contact_world();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(contact_damage);
        schedule.run(&mut world);

        // Damage + fresh invulnerability.
        assert_eq!(world.get::<Health>(player).unwrap().hp, 17);
        assert!(!world.get::<Health>(player).unwrap().invuln.is_finished());
        // Melee goes on cooldown.
        assert!(!world.get::<EnemyBrain>(enemy).unwrap().melee.is_finished());
        // Knockback straight away from the enemy at 120.
        assert_eq!(
            world.get::<Velocity>(player).unwrap().0,
            glam::Vec2::new(-120.0, 0.0)
        );
        // Secrets + last-damage bookkeeping.
        assert!(world.resource::<SecretTriggers>().damage_taken_this_floor);
        assert!(!world.resource::<SecretTriggers>().oasis_eligible);
        let noted = world.resource::<LastDamageTaken>();
        assert_eq!(noted.hit_id, Some(crate::comps_a::HitId::Contact));
        assert_eq!(noted.enemy_kind, Some(crate::data::EnemyKind::Bandit));
        // Feel sinks.
        assert!((world.resource::<Trauma>().amount - 0.35).abs() < 1e-6);
        assert_eq!(world.resource::<Queue<RumbleRequest>>().len(), 1);
        assert_eq!(world.resource::<Queue<AudioCue>>().len(), 1);
        assert!(world.get::<HitFlash>(player).is_some());
        // 12-burst particles.
        let mut q = world.query::<&repame_fx::Particle>();
        assert_eq!(q.iter(&world).count(), 12);
        // No shield without the mutation.
        assert!(world.get::<Shield>(player).is_none());
    }

    #[test]
    fn contact_skips_on_invuln_or_cooldown() {
        let (mut world, player, enemy) = contact_world();
        world.get_mut::<Health>(player).unwrap().invuln =
            GTimer::from_seconds(1.0, TimerMode::Once);
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(contact_damage);
        schedule.run(&mut world);
        assert_eq!(world.get::<Health>(player).unwrap().hp, 20);
        assert!(world.resource::<Queue<AudioCue>>().is_empty());

        // Cooled invuln but enemy melee on cooldown: still nothing.
        world.get_mut::<Health>(player).unwrap().invuln = GTimer::disarmed();
        world.get_mut::<EnemyBrain>(enemy).unwrap().melee =
            GTimer::from_seconds(1.0, TimerMode::Once);
        schedule.run(&mut world);
        assert_eq!(world.get::<Health>(player).unwrap().hp, 20);
    }

    const DEATH_CATALOG_JSON: &str = r#"{
        "images/sprBanditDead.png": {"frames": 2, "w": 32, "h": 32, "fps": 8.0, "xorigin": 16.0, "yorigin": 16.0},
        "sprBouncerBullet": {"frames": 1, "w": 7, "h": 7, "fps": 8.0, "xorigin": 3.0, "yorigin": 3.0}
    }"#;

    fn death_world() -> bevy_ecs::prelude::World {
        let mut world = bevy_ecs::prelude::World::new();
        world.init_resource::<Score>();
        world.init_resource::<Run>();
        world.init_resource::<FloorMask>();
        world.init_resource::<SaveDirty>();
        world.init_resource::<crate::savedata_part::SaveData>();
        world.init_resource::<Trauma>();
        world.init_resource::<ChromaticAberration>();
        world.init_resource::<FlashWhite>();
        world.init_resource::<HitStop>();
        world.init_resource::<SlowMotion>();
        world.init_resource::<LoopTransition>();
        world.init_resource::<ThroneRoomState>();
        world.init_resource::<Toast>();
        world.init_resource::<GameAudio>();
        world.init_resource::<Queue<AudioCue>>();
        world.init_resource::<Queue<RumbleRequest>>();
        world.init_resource::<DeathEvents>();
        world.insert_resource(
            AnimCatalog::from_json(
                DEATH_CATALOG_JSON,
                repame_anim::AtlasDesc {
                    size: 128,
                    max_pages: 1,
                padding: 0,
                },
            )
            .unwrap(),
        );
        world.spawn((
            Player::default(),
            Pos(glam::Vec2::ZERO),
            Health {
                hp: 20,
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
                weapon_slots: 1,
                current: 0,
                ammo: [0, 0, 0, 0, 0, 0],
            },
            RaceState {
                race: crate::data::RaceId::Fish,
                skin: crate::data::SkinLetter::A,
            },
        ));
        world
    }

    fn spawn_dead_bandit(world: &mut bevy_ecs::prelude::World) -> bevy_ecs::prelude::Entity {
        world
            .spawn((
                GameCleanup,
                Team::Enemy,
                Enemy {
                    kind: crate::data::EnemyKind::Bandit,
                    score: 5,
                    touch_damage: 1,
                    rad_drop: 1,
                    drop_chance: 0,
                    weapon_chance: 0,
                },
                Pos(glam::Vec2::new(50.0, 0.0)),
                Velocity(glam::Vec2::new(10.0, 0.0)),
                Health {
                    hp: 0,
                    max: 4,
                    invuln: GTimer::disarmed(),
                },
            ))
            .id()
    }

    #[test]
    fn death_despawns_spawns_corpse_and_scores() {
        let mut world = death_world();
        let dead = spawn_dead_bandit(&mut world);
        // Second live enemy so enemy_total == 2 triggers the levelup sting.
        world.spawn((
            GameCleanup,
            Team::Enemy,
            Enemy {
                kind: crate::data::EnemyKind::Maggot,
                score: 5,
                touch_damage: 1,
                rad_drop: 1,
                drop_chance: 0,
                weapon_chance: 0,
            },
            Pos(glam::Vec2::new(400.0, 0.0)),
            Velocity(glam::Vec2::ZERO),
            Health {
                hp: 2,
                max: 2,
                invuln: GTimer::disarmed(),
            },
        ));
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        // Two schedules, run in order: ambiguous same-tick order
        // would let drops run on an empty record. Production chains
        // them via `Sim::add_chained_systems`.
        schedule.add_systems(resolve_enemy_deaths);
        let mut drops = bevy_ecs::schedule::Schedule::default();
        drops.add_systems(resolve_death_drops);
        schedule.run(&mut world);
        drops.run(&mut world);

        assert!(world.get_entity(dead).is_err(), "despawned");
        assert_eq!(world.resource::<Run>().total_kills, 1);
        assert_eq!(world.resource::<Score>().0, 5);
        assert_eq!(
            world.resource::<crate::savedata_part::SaveData>().high_score,
            5,
            "kill writes high score"
        );
        assert!(
            world.resource::<SaveDirty>().0,
            "kill dirties the save"
        );
        assert!((world.resource::<Trauma>().amount - 0.25).abs() < 1e-6);
        assert_eq!(world.resource::<ChromaticAberration>().0, 0.08);
        assert!(world.resource::<HitStop>().active);
        // Corpse with slide velocity preserved.
        let mut corpses = world.query::<(&Corpse, &Velocity, &SpriteAnim)>();
        let (corpse, vel, anim) = corpses.iter(&world).next().expect("corpse");
        assert_eq!(corpse.kind, crate::data::EnemyKind::Bandit);
        assert_eq!(vel.0, glam::Vec2::new(10.0, 0.0));
        assert!(anim.oneshot);
        // 14-burst + levelup sting + hit sting.
        let mut parts = world.query::<&repame_fx::Particle>();
        assert_eq!(parts.iter(&world).count(), 14);
        assert_eq!(world.resource::<Queue<AudioCue>>().len(), 2);
    }

    #[test]
    fn throne_death_without_loop_ends_run() {
        let mut world = death_world();
        let _throne = world
            .spawn((
                GameCleanup,
                Team::Enemy,
                Enemy {
                    kind: crate::data::EnemyKind::Throne,
                    score: 0,
                    touch_damage: 0,
                    rad_drop: 0,
                    drop_chance: 0,
                    weapon_chance: 0,
                },
                Pos(glam::Vec2::new(50.0, 0.0)),
                Velocity(glam::Vec2::ZERO),
                Health {
                    hp: 0,
                    max: 100,
                    invuln: GTimer::disarmed(),
                },
            ))
            .id();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        // Two schedules, run in order: ambiguous same-tick order
        // would let drops run on an empty record. Production chains
        // them via `Sim::add_chained_systems`.
        schedule.add_systems(resolve_enemy_deaths);
        let mut drops = bevy_ecs::schedule::Schedule::default();
        drops.add_systems(resolve_death_drops);
        schedule.run(&mut world);
        drops.run(&mut world);
        assert!(world.resource::<Run>().game_over);
        // Boss branch fires after the game-over branch and overwrites
        // the toast (bevy parity: last show wins).
        assert_eq!(world.resource::<Toast>().text, "THE THRONE DEFEATED");
        assert_eq!(world.resource::<FlashWhite>().amount, 1.0);
    }

    fn spawn_dead(
        world: &mut bevy_ecs::prelude::World,
        kind: crate::data::EnemyKind,
    ) -> bevy_ecs::prelude::Entity {
        world
            .spawn((
                GameCleanup,
                Team::Enemy,
                Enemy {
                    kind,
                    score: 5,
                    touch_damage: 1,
                    rad_drop: 1,
                    drop_chance: 0,
                    weapon_chance: 0,
                },
                Pos(glam::Vec2::new(50.0, 0.0)),
                Velocity(glam::Vec2::ZERO),
                Health {
                    hp: 0,
                    max: 10,
                    invuln: GTimer::disarmed(),
                },
            ))
            .id()
    }

    #[test]
    fn proto_statue_rich_death_opens_vault_portal() {
        use crate::data::EnemyKind;
        let mut world = death_world();
        world.spawn((
            GameCleanup,
            Team::Enemy,
            Enemy {
                kind: EnemyKind::ProtoStatue,
                score: 0,
                touch_damage: 0,
                rad_drop: 0,
                drop_chance: 0,
                weapon_chance: 0,
            },
            Pos(glam::Vec2::ZERO),
            Velocity(glam::Vec2::ZERO),
            Health {
                hp: 0,
                max: 120,
                invuln: GTimer::disarmed(),
            },
            crate::comps_b::ProtoGuardian {
                rad: 30,
                charged: true,
                phased: true,
                init: true,
            },
        ));
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(resolve_enemy_deaths);
        schedule.run(&mut world);
        let portals: Vec<_> = world
            .query::<(&Portal, &PortalState)>()
            .iter(&world)
            .collect();
        assert_eq!(portals.len(), 1, "rich statue opens a portal");
        assert_eq!(portals[0].1.kind, 3, "type-3 vault portal");
    }

    #[test]
    fn throne_victory_clears_invisible_walls() {
        let mut world = death_world();
        let mut time = SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        // One invisible wall on a mask cell.
        world.insert_resource({
            let mut mask = FloorMask::default();
            mask.cells.insert((0, 0));
            mask
        });
        world.spawn((
            GameCleanup,
            crate::comps_b::InvisiWall,
            Pos(glam::Vec2::new(16.0, 16.0)),
        ));
        world.spawn((
            GameCleanup,
            LevelCleanup,
            crate::comps_b::ThroneVictory {
                timer: GTimer::from_seconds(80.0 / 30.0, TimerMode::Once),
                pos: glam::Vec2::ZERO,
                bursts: 0,
            },
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_throne_victory);
        sched.run(&mut world);
        assert_eq!(
            world.query::<&crate::comps_b::InvisiWall>().iter(&world).count(),
            0,
            "pageant clears invisible walls"
        );
        assert!(
            !world
                .resource::<FloorMask>()
                .cells
                .contains(&(0, 0)),
            "mask cell opens too"
        );
    }

    #[test]
    fn throne_victory_pageant_bursts_then_rads() {
        let mut world = death_world();
        let mut time = SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        world.spawn((
            GameCleanup,
            LevelCleanup,
            crate::comps_b::ThroneVictory {
                timer: GTimer::from_seconds(80.0 / 30.0, TimerMode::Once),
                pos: glam::Vec2::ZERO,
                bursts: 0,
            },
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_throne_victory);
        for _ in 0..40 {
            sched.run(&mut world);
        }
        assert!(
            world.query::<&Explosion>().iter(&world).count() > 0,
            "pageant explodes mid-sequence"
        );
        for _ in 0..60 {
            sched.run(&mut world);
        }
        assert_eq!(
            world.query::<&crate::comps_b::ThroneVictory>().iter(&world).count(),
            0,
            "sequencer expires"
        );
        let rads = world
            .query::<&crate::comps_b::Pickup>()
            .iter(&world)
            .filter(|p| matches!(p.kind, crate::comps_b::PickupKind::Rad(10)))
            .count();
        assert_eq!(rads, 10, "ten BigRads flung");
    }

    #[test]
    fn per_kind_arms_spawn_their_payloads() {
        use crate::data::EnemyKind;
        let mut world = death_world();
        spawn_dead(&mut world, EnemyKind::Ballguy);
        spawn_dead(&mut world, EnemyKind::ExploFreak);
        spawn_dead(&mut world, EnemyKind::BigMaggot);
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        // Two schedules, run in order: ambiguous same-tick order
        // would let drops run on an empty record. Production chains
        // them via `Sim::add_chained_systems`.
        schedule.add_systems(resolve_enemy_deaths);
        let mut drops = bevy_ecs::schedule::Schedule::default();
        drops.add_systems(resolve_death_drops);
        schedule.run(&mut world);
        drops.run(&mut world);

        // Ballguy (GML Exploder): 8-bullet ring at 4 px/tick plus 8
        // acid streaks at 8 px/tick; the bullets are animated.
        let mut bullets = world.query::<(&Projectile, &SpriteAnim)>();
        assert_eq!(bullets.iter(&world).count(), 8);
        for (b, anim) in bullets.iter(&world) {
            assert_eq!((b.damage, b.radius), (2, 4.0));
            assert_eq!(anim.path, "sprBouncerBullet");
        }
        let acids = world.query::<&Projectile>().iter(&world).count();
        assert_eq!(acids, 16, "bullets plus acid streaks");
        // ExploFreak: timed 46px explosion.
        let mut booms = world.query::<&Explosion>();
        let boom = booms.iter(&world).next().expect("explosion");
        assert_eq!((boom.radius, boom.damage), (46.0, 5));
        assert!(boom.hits_player);
        // BigMaggot: 6 pending maggot spawns.
        let mut pending = world.query::<&PendingEnemySpawn>();
        assert_eq!(pending.iter(&world).count(), 6);
        for p in pending.iter(&world) {
            assert_eq!(p.kind, EnemyKind::Maggot);
        }
    }

    fn drops_world() -> bevy_ecs::prelude::World {
        let mut world = death_world();
        // Chain explosives + trigger fingers on the player.
        let mut player_q = world.query::<(Entity, &Player)>();
        let (e, _) = player_q.iter(&world).next().unwrap();
        let mut player = world.get_mut::<Player>(e).unwrap();
        player.chain_explosions = true;
        player
            .mutations
            .push(crate::data::MutationId::TriggerFingers);
        world.entity_mut(e).insert(FireCooldown {
            timer: GTimer::from_seconds(0.5, TimerMode::Once),
            burst_left: 1,
            burst_timer: GTimer::from_seconds(0.5, TimerMode::Once),
            timer_b: GTimer::disarmed(),
            burst_left_b: 0,
            burst_timer_b: GTimer::disarmed(),
        });
        world
    }

    #[test]
    fn drops_chain_explosions_and_trigger_scaling() {
        use crate::comps_a::FireCooldown;
        let mut world = drops_world();
        spawn_dead(&mut world, crate::data::EnemyKind::Bandit);
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(resolve_enemy_deaths);
        let mut drops = bevy_ecs::schedule::Schedule::default();
        drops.add_systems(resolve_death_drops);
        schedule.run(&mut world);
        drops.run(&mut world);
        // Chain explosives: player-team 100px explosion.
        let mut booms = world.query::<&Explosion>();
        assert_eq!(booms.iter(&world).count(), 1);
        let boom = booms.iter(&world).next().unwrap();
        assert_eq!((boom.radius, boom.damage), (100.0, 3));
        // TriggerFingers: live timers scaled to 60%.
        let mut fcq = world.query::<&FireCooldown>();
        let fc = fcq.iter(&world).next().unwrap();
        assert!((fc.timer.remaining_secs() - 0.3).abs() < 1e-4);
        assert!((fc.burst_timer.remaining_secs() - 0.3).abs() < 1e-4);
    }

    fn move_world() -> bevy_ecs::prelude::World {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        world.insert_resource(CurrentFrame::default());
        world.init_resource::<Run>();
        world.init_resource::<SecretTriggers>();
        world.init_resource::<GameAudio>();
        world.init_resource::<Queue<AudioCue>>();
        world.insert_resource(
            AnimCatalog::from_json(
                r#"{"images/sprDust.png": {"frames": 1, "w": 8, "h": 8, "fps": 1.0, "xorigin": 4.0, "yorigin": 4.0}}"#,
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

    fn spawn_bullet(
        world: &mut bevy_ecs::prelude::World,
        pos: glam::Vec2,
        vel: glam::Vec2,
        life_secs: f32,
    ) -> bevy_ecs::prelude::Entity {
        world
            .spawn((
                Team::Enemy,
                Projectile {
                    damage: 2,
                    life: GTimer::from_seconds(life_secs, TimerMode::Once),
                    radius: 4.0,
                    knockback: 60.0,
                    explosive: false,
                    source: None,
                },
                Velocity(vel),
                Pos(pos),
            ))
            .id()
    }

    fn run_move(world: &mut bevy_ecs::prelude::World, ticks: usize) {
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(move_projectiles);
        for _ in 0..ticks {
            schedule.run(world);
        }
    }

    #[test]
    fn life_expiry_despawns() {
        let mut world = move_world();
        let e = spawn_bullet(
            &mut world,
            glam::Vec2::ZERO,
            glam::Vec2::new(10.0, 0.0),
            0.05,
        );
        run_move(&mut world, 1);
        assert!(world.get_entity(e).is_ok(), "alive at 0.033s");
        run_move(&mut world, 1);
        assert!(world.get_entity(e).is_err(), "expired at 0.066s");
    }

    #[test]
    fn wall_bounce_reflects_and_counts_down() {
        use crate::comps_a::BouncesLeft;
        let mut world = move_world();
        let e = world
            .spawn((
                Team::Enemy,
                Projectile {
                    damage: 2,
                    life: GTimer::from_seconds(5.0, TimerMode::Once),
                    radius: 4.0,
                    knockback: 60.0,
                    explosive: false,
                    source: None,
                },
                Velocity(glam::Vec2::new(100.0, 0.0)),
                Pos(glam::Vec2::new(1277.0, 0.0)),
                BouncesLeft(2),
            ))
            .id();
        run_move(&mut world, 1);
        assert_eq!(world.get::<BouncesLeft>(e).unwrap().0, 1);
        let v = world.get::<Velocity>(e).unwrap().0;
        assert!(v.x < 0.0, "reflected, got {v:?}");
        assert!((v.length() - 80.0).abs() < 1.0, "0.8 factor, got {v:?}");
    }

    #[test]
    fn sticky_arms_and_stops_on_wall() {
        use crate::comps_a::Sticky;
        let mut world = move_world();
        let e = world
            .spawn((
                Team::Enemy,
                Projectile {
                    damage: 2,
                    life: GTimer::from_seconds(5.0, TimerMode::Once),
                    radius: 4.0,
                    knockback: 60.0,
                    explosive: false,
                    source: None,
                },
                Velocity(glam::Vec2::new(100.0, 0.0)),
                Pos(glam::Vec2::new(1277.0, 0.0)),
                Sticky::default(),
            ))
            .id();
        run_move(&mut world, 1);
        let s = world.get::<Sticky>(e).unwrap();
        assert!(s.armed);
        assert_eq!(world.get::<Velocity>(e).unwrap().0, glam::Vec2::ZERO);
    }

    #[test]
    fn explosive_expiry_spawns_blast() {
        let mut world = move_world();
        world.spawn((
            Team::Enemy,
            Projectile {
                damage: 7,
                life: GTimer::from_seconds(0.03, TimerMode::Once),
                radius: 4.0,
                knockback: 60.0,
                explosive: true,
                source: None,
            },
            Velocity(glam::Vec2::ZERO),
            Pos(glam::Vec2::ZERO),
        ));
        run_move(&mut world, 1);
        let mut booms = world.query::<&Explosion>();
        let boom = booms.iter(&world).next().expect("blast");
        assert_eq!((boom.radius, boom.damage), (32.0, 7));
    }

    #[test]
    fn out_of_arena_despawns() {
        let mut world = move_world();
        let e = spawn_bullet(
            &mut world,
            glam::Vec2::new(2000.0, 0.0),
            glam::Vec2::ZERO,
            5.0,
        );
        run_move(&mut world, 1);
        assert!(world.get_entity(e).is_err());
    }

    fn hits_world() -> bevy_ecs::prelude::World {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        world.init_resource::<Trauma>();
        world.init_resource::<HitStop>();
        world.init_resource::<GameAudio>();
        world.init_resource::<Queue<AudioCue>>();
        world.init_resource::<SecretTriggers>();
        world.init_resource::<LastDamageTaken>();
        world.init_resource::<CurrentFrame>();
        world.insert_resource(
            AnimCatalog::from_json(
                r#"{}"#,
                repame_anim::AtlasDesc {
                    size: 64,
                    max_pages: 1,
                padding: 0,
                },
            )
            .unwrap(),
        );
        world.spawn((
            Player::default(),
            Team::Player,
            Pos(glam::Vec2::ZERO),
            Velocity(glam::Vec2::ZERO),
            crate::comps_a::Hitbox { radius: 8.0 },
            Health {
                hp: 20,
                max: 20,
                invuln: GTimer::disarmed(),
            },
        ));
        world
    }

    fn spawn_foe(
        world: &mut bevy_ecs::prelude::World,
        at: glam::Vec2,
    ) -> bevy_ecs::prelude::Entity {
        world
            .spawn((
                Enemy {
                    kind: crate::data::EnemyKind::Bandit,
                    score: 5,
                    touch_damage: 1,
                    rad_drop: 1,
                    drop_chance: 0,
                    weapon_chance: 0,
                },
                Pos(at),
                Velocity(glam::Vec2::ZERO),
                Team::Enemy,
                Hitbox { radius: 8.0 },
                Health {
                    hp: 10,
                    max: 10,
                    invuln: GTimer::disarmed(),
                },
            ))
            .id()
    }

    fn spawn_shot(
        world: &mut bevy_ecs::prelude::World,
        at: glam::Vec2,
        team: Team,
    ) -> bevy_ecs::prelude::Entity {
        world
            .spawn((
                team,
                Projectile {
                    damage: 3,
                    life: GTimer::from_seconds(5.0, TimerMode::Once),
                    radius: 4.0,
                    knockback: 60.0,
                    explosive: false,
                    source: None,
                },
                Velocity(glam::Vec2::new(50.0, 0.0)),
                Pos(at),
            ))
            .id()
    }

    fn run_hits(world: &mut bevy_ecs::prelude::World) {
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(projectile_hits);
        schedule.run(world);
    }

    #[test]
    fn enemy_shot_hurts_player() {
        let mut world = hits_world();
        spawn_shot(&mut world, glam::Vec2::ZERO, Team::Enemy);
        run_hits(&mut world);
        let mut pq = world.query::<&Health>();
        let hp = pq.iter(&world).next().unwrap().hp;
        assert_eq!(hp, 17);
        assert!((world.resource::<Trauma>().amount - 0.08).abs() < 1e-6);
        // Damage number floater spawned.
        let mut nq = world.query::<&repame_fx::DamageNumber>();
        assert_eq!(nq.iter(&world).count(), 1);
    }

    #[test]
    fn shield_blocks_without_damage() {
        use crate::comps_b::Shield;
        let mut world = hits_world();
        let mut pq = world.query::<(Entity, &Player)>();
        let (e, _) = pq.iter(&world).next().unwrap();
        world.entity_mut(e).insert(Shield {
            timer: GTimer::from_seconds(1.0, TimerMode::Once),
        });
        spawn_shot(&mut world, glam::Vec2::ZERO, Team::Enemy);
        run_hits(&mut world);
        let mut hq = world.query::<&Health>();
        assert_eq!(hq.iter(&world).next().unwrap().hp, 20);
    }

    #[test]
    fn pierce_hits_two_foes_then_despawns() {
        use crate::comps_a::PiercesLeft;
        let mut world = hits_world();
        spawn_foe(&mut world, glam::Vec2::new(20.0, 0.0));
        spawn_foe(&mut world, glam::Vec2::new(40.0, 0.0));
        let shot = world
            .spawn((
                Team::Player,
                Projectile {
                    damage: 4,
                    life: GTimer::from_seconds(5.0, TimerMode::Once),
                    radius: 40.0,
                    knockback: 60.0,
                    explosive: false,
                    source: None,
                },
                Velocity(glam::Vec2::ZERO),
                Pos(glam::Vec2::ZERO),
                PiercesLeft(2),
            ))
            .id();
        run_hits(&mut world);
        run_hits(&mut world);
        let mut hq = world.query::<&Health>();
        let hps: Vec<i32> = hq.iter(&world).map(|h| h.hp).collect();
        assert!(hps.contains(&6), "both foes hit across ticks, got {hps:?}");
        assert_eq!(world.get::<PiercesLeft>(shot).unwrap().0, 0);
        assert!(world.get_entity(shot).is_ok(), "pierce left, still flying");
    }

    #[test]
    fn hitset_skips_repeat_hits() {
        let mut world = hits_world();
        spawn_foe(&mut world, glam::Vec2::new(10.0, 0.0));
        spawn_shot(&mut world, glam::Vec2::ZERO, Team::Player);
        run_hits(&mut world);
        run_hits(&mut world);
        let mut hq = world.query::<&Health>();
        // Player (20) untouched; foe hit exactly once (10 -> 7).
        let mut hps: Vec<i32> = hq.iter(&world).map(|h| h.hp).collect();
        hps.sort();
        assert_eq!(hps, vec![7, 20]);
    }
}

#[cfg(test)]
mod battery_tests {
    use super::*;

    const DT: f32 = 1.0 / 30.0;

    fn clocked_world() -> bevy_ecs::prelude::World {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = SimTime::default();
        time.delta_secs = DT;
        world.insert_resource(time);
        world.init_resource::<CurrentFrame>();
        world.init_resource::<Run>();
        world.init_resource::<SecretTriggers>();
        world.init_resource::<GameAudio>();
        world.init_resource::<Queue<AudioCue>>();
        world.init_resource::<Queue<RumbleRequest>>();
        world.init_resource::<Trauma>();
        world.init_resource::<HitStop>();
        world.init_resource::<ChromaticAberration>();
        world.init_resource::<FlashWhite>();
        world.init_resource::<LastDamageTaken>();
        world.insert_resource(
            repame_anim::AnimCatalog::from_json(
                r#"{"images/sprBullet2Disappear.png": {"frames": 4, "w": 8, "h": 8, "fps": 12.0, "xorigin": 4.0, "yorigin": 4.0}}"#,
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

    fn spawn_enemy_at(world: &mut bevy_ecs::prelude::World, at: glam::Vec2, hp: i32) {
        world.spawn((
            Enemy {
                kind: EnemyKind::Bandit,
                score: 5,
                touch_damage: 1,
                rad_drop: 0,
                drop_chance: 0,
                weapon_chance: 0,
            },
            Pos(at),
            Velocity(glam::Vec2::ZERO),
            Team::Enemy,
            Hitbox { radius: 8.0 },
            Health {
                hp,
                max: hp,
                invuln: GTimer::disarmed(),
            },
        ));
    }

    fn spawn_player_at(world: &mut bevy_ecs::prelude::World, at: glam::Vec2) {
        world.spawn((
            Player::default(),
            Team::Player,
            Pos(at),
            Velocity(glam::Vec2::ZERO),
            Hitbox { radius: 8.0 },
            Health {
                hp: 20,
                max: 20,
                invuln: GTimer::disarmed(),
            },
        ));
    }

    #[test]
    fn homing_curves_toward_target() {
        let mut world = clocked_world();
        spawn_enemy_at(&mut world, glam::Vec2::new(100.0, 60.0), 10);
        let shot = world
            .spawn((
                Team::Player,
                Projectile {
                    damage: 2,
                    life: GTimer::from_seconds(5.0, TimerMode::Once),
                    radius: 4.0,
                    knockback: 60.0,
                    explosive: false,
                    source: None,
                },
                Velocity(glam::Vec2::new(200.0, 0.0)),
                Pos(glam::Vec2::ZERO),
                Homing {
                    turn_rate: 3.0,
                    acquire_range: 500.0,
                },
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_homing_projectiles);
        for _ in 0..10 {
            sched.run(&mut world);
        }
        let v = world.get::<Velocity>(shot).unwrap().0;
        assert!((v.length() - 200.0).abs() < 1.0, "speed kept, got {v:?}");
        assert!(v.y > 20.0, "curved toward the enemy, got {v:?}");
    }

    #[test]
    fn homing_holds_course_without_target() {
        let mut world = clocked_world();
        let shot = world
            .spawn((
                Team::Player,
                Projectile {
                    damage: 2,
                    life: GTimer::from_seconds(5.0, TimerMode::Once),
                    radius: 4.0,
                    knockback: 60.0,
                    explosive: false,
                    source: None,
                },
                Velocity(glam::Vec2::new(200.0, 0.0)),
                Pos(glam::Vec2::ZERO),
                Homing {
                    turn_rate: 3.0,
                    acquire_range: 10.0,
                },
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_homing_projectiles);
        sched.run(&mut world);
        assert_eq!(
            world.get::<Velocity>(shot).unwrap().0,
            glam::Vec2::new(200.0, 0.0)
        );
    }

    #[test]
    fn sticky_rides_its_target() {
        let mut world = clocked_world();
        let anchor = world.spawn(Pos(glam::Vec2::new(50.0, 5.0))).id();
        let bolt = world
            .spawn((
                Team::Player,
                Projectile {
                    damage: 2,
                    life: GTimer::from_seconds(5.0, TimerMode::Once),
                    radius: 4.0,
                    knockback: 60.0,
                    explosive: false,
                    source: None,
                },
                Velocity(glam::Vec2::new(90.0, 0.0)),
                Pos(glam::Vec2::ZERO),
                Sticky {
                    armed: true,
                    stuck_to: Some(anchor),
                    offset: glam::Vec2::new(1.0, 2.0),
                },
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_sticky_projectiles);
        sched.run(&mut world);
        assert_eq!(world.get::<Velocity>(bolt).unwrap().0, glam::Vec2::ZERO);
        assert_eq!(
            world.get::<Pos>(bolt).unwrap().0,
            glam::Vec2::new(51.0, 7.0)
        );
    }

    #[test]
    fn armed_sticky_detonates_triple_blast() {
        let mut world = clocked_world();
        let anchor = world.spawn(Pos(glam::Vec2::new(40.0, 0.0))).id();
        let sticky = world
            .spawn((
                Team::Player,
                Projectile {
                    damage: 4,
                    life: GTimer::from_seconds(0.03, TimerMode::Once),
                    radius: 4.0,
                    knockback: 60.0,
                    explosive: true,
                    source: None,
                },
                Velocity(glam::Vec2::ZERO),
                Pos(glam::Vec2::new(40.0, 0.0)),
                Sticky {
                    armed: true,
                    stuck_to: Some(anchor),
                    offset: glam::Vec2::ZERO,
                },
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(move_projectiles);
        sched.run(&mut world);
        assert!(world.get_entity(sticky).is_err());
        assert_eq!(world.query::<&Explosion>().iter(&world).count(), 3);
    }

    #[test]
    fn sentry_fires_on_interval() {
        let mut world = clocked_world();
        spawn_enemy_at(&mut world, glam::Vec2::new(60.0, 0.0), 10);
        world.spawn((
            Team::Player,
            SentryTurret {
                life: GTimer::from_seconds(5.0, TimerMode::Once),
                fire: GTimer::from_seconds(0.1, TimerMode::Repeating),
                range: 200.0,
                projectile_speed: 300.0,
                projectile_damage: 3,
            },
            Pos(glam::Vec2::ZERO),
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_sentry_turrets);
        for _ in 0..4 {
            sched.run(&mut world);
        }
        let mut shots = world.query::<(&Projectile, &Team, &Velocity)>();
        let (shot, team, vel) = shots.iter(&world).next().expect("sentry shot");
        assert_eq!(*team, Team::Player);
        assert_eq!(shot.damage, 3);
        assert!((vel.0.length() - 300.0).abs() < 1.0);
    }

    #[test]
    fn sentry_dies_on_life_end() {
        let mut world = clocked_world();
        let turret = world
            .spawn((
                Team::Player,
                SentryTurret {
                    life: GTimer::from_seconds(0.03, TimerMode::Once),
                    fire: GTimer::from_seconds(5.0, TimerMode::Repeating),
                    range: 200.0,
                    projectile_speed: 300.0,
                    projectile_damage: 3,
                },
                Pos(glam::Vec2::ZERO),
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_sentry_turrets);
        sched.run(&mut world);
        assert!(world.get_entity(turret).is_err());
    }

    #[test]
    fn grenade_fuse_flips_friction_with_smoke() {
        let mut world = clocked_world();
        let nade = world
            .spawn((
                GrenadeFuse {
                    smoke_armed: false,
                    friction_switched: false,
                    alarm1: GTimer::from_seconds(0.2, TimerMode::Once),
                },
                ProjectileFriction(2.0),
                Pos(glam::Vec2::ZERO),
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems((tick_grenade_fuse, tick_projectile_friction).chain());
        for _ in 0..7 {
            sched.run(&mut world);
        }
        let fuse = world.get::<GrenadeFuse>(nade).unwrap();
        assert!(fuse.friction_switched && fuse.smoke_armed);
        assert_eq!(world.get::<ProjectileFriction>(nade).unwrap().0, 0.4);
        assert!(
            world.query::<&repame_fx::Particle>().iter(&world).count() >= 4,
            "smoke burst spawned"
        );
    }

    #[test]
    fn shell_bonus_window_drains() {
        let mut world = clocked_world();
        let e = world
            .spawn(ShellBonus {
                timer: GTimer::from_seconds(2.0 / 30.0, TimerMode::Once),
                bonus: 2,
            })
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_shell_bonus);
        sched.run(&mut world);
        assert_eq!(world.get::<ShellBonus>(e).unwrap().bonus, 2);
        sched.run(&mut world);
        sched.run(&mut world);
        assert_eq!(world.get::<ShellBonus>(e).unwrap().bonus, 0);
    }

    #[test]
    fn spawn_grace_expires() {
        let mut world = clocked_world();
        let e = world
            .spawn(SpawnGrace(GTimer::from_seconds(0.05, TimerMode::Once)))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_spawn_grace);
        sched.run(&mut world);
        assert!(world.get::<SpawnGrace>(e).is_some());
        sched.run(&mut world);
        assert!(world.get::<SpawnGrace>(e).is_none());
    }

    #[test]
    fn flame_trail_drips_clouds() {
        use crate::data::{HazardDef, HazardKind};
        let mut world = clocked_world();
        world.spawn((
            Team::Player,
            Projectile {
                damage: 1,
                life: GTimer::from_seconds(5.0, TimerMode::Once),
                radius: 4.0,
                knockback: 0.0,
                explosive: false,
                source: None,
            },
            Pos(glam::Vec2::ZERO),
            FlameTrail {
                timer: GTimer::from_seconds(0.03, TimerMode::Once),
                spec: HazardDef {
                    kind: HazardKind::Fire,
                    radius: 12.0,
                    damage: 1,
                    duration: 2.0,
                    tick: 0.5,
                    color: [1.0, 0.5, 0.0, 1.0],
                },
            },
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_flame_trails);
        sched.run(&mut world);
        assert_eq!(world.query::<&HazardCloud>().iter(&world).count(), 1);
    }

    #[test]
    fn lightning_arcs_expire() {
        let mut world = clocked_world();
        let arc = world
            .spawn((
                LightningArc {
                    timer: GTimer::from_seconds(0.03, TimerMode::Once),
                    len: 10.0,
                    angle: 0.0,
                },
                Pos(glam::Vec2::ZERO),
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_lightning_arcs);
        sched.run(&mut world);
        assert!(world.get_entity(arc).is_err());
    }

    #[test]
    fn hit_effects_expire() {
        let mut world = clocked_world();
        let fx = world
            .spawn(PickupLifetime {
                timer: GTimer::from_seconds(0.03, TimerMode::Once),
            })
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_hit_effects);
        sched.run(&mut world);
        assert!(world.get_entity(fx).is_err());
    }

    #[test]
    fn bullet2_fade_shortens_life() {
        let mut world = clocked_world();
        let shell = world
            .spawn((
                Team::Player,
                Projectile {
                    damage: 3,
                    life: GTimer::from_seconds(5.0, TimerMode::Once),
                    radius: 4.0,
                    knockback: 60.0,
                    explosive: false,
                    source: None,
                },
                Velocity(glam::Vec2::new(100.0, 0.0)),
                Pos(glam::Vec2::ZERO),
                ShellWallBounce {
                    add: 0.0,
                    cap: 480.0,
                    decay: 0.95,
                    rearm: None,
                },
                ProjectileFade("images/sprBullet2Disappear.png"),
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_bullet2_fade);
        sched.run(&mut world);
        let proj = world.get::<Projectile>(shell).unwrap();
        assert!(
            (proj.life.duration() - 4.0 / 12.0).abs() < 1e-4,
            "fade length, got {}",
            proj.life.duration()
        );
        assert!(world.get::<SpriteAnim>(shell).is_some());
    }

    #[test]
    fn flame_shell_dies_slow_with_no_fade() {
        let mut world = clocked_world();
        let shell = world
            .spawn((
                Team::Player,
                Projectile {
                    damage: 3,
                    life: GTimer::from_seconds(5.0, TimerMode::Once),
                    radius: 4.0,
                    knockback: 60.0,
                    explosive: false,
                    source: None,
                },
                Velocity(glam::Vec2::new(100.0, 0.0)),
                Pos(glam::Vec2::ZERO),
                ShellWallBounce {
                    add: 0.0,
                    cap: 480.0,
                    decay: 0.95,
                    rearm: None,
                },
                FlameShellSlowDeath,
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_bullet2_fade);
        sched.run(&mut world);
        assert_eq!(world.get::<Projectile>(shell).unwrap().life.duration(), 0.0);
    }

    #[test]
    fn beam_deals_tick_damage() {
        let mut world = clocked_world();
        spawn_enemy_at(&mut world, glam::Vec2::new(30.0, 0.0), 20);
        world.spawn((
            Beam {
                team: Team::Player,
                dir: glam::Vec2::X,
                length: 100.0,
                width: 20.0,
                damage: 2,
                knockback: 50.0,
                color: [1.0, 1.0, 1.0, 1.0],
                timer: GTimer::from_seconds(5.0, TimerMode::Once),
                tick: GTimer::from_seconds(0.03, TimerMode::Repeating),
                source: None,
            },
            Pos(glam::Vec2::ZERO),
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_beams);
        for _ in 0..3 {
            sched.run(&mut world);
        }
        let mut foes = world.query::<(&Enemy, &Health)>();
        let (_, hp) = foes.iter(&world).next().unwrap();
        assert!(hp.hp < 20, "beam ticked damage, hp {}", hp.hp);
    }

    #[test]
    fn beam_spares_off_axis_targets() {
        let mut world = clocked_world();
        spawn_enemy_at(&mut world, glam::Vec2::new(30.0, 60.0), 20);
        world.spawn((
            Beam {
                team: Team::Player,
                dir: glam::Vec2::X,
                length: 100.0,
                width: 20.0,
                damage: 2,
                knockback: 50.0,
                color: [1.0, 1.0, 1.0, 1.0],
                timer: GTimer::from_seconds(5.0, TimerMode::Once),
                tick: GTimer::from_seconds(0.03, TimerMode::Repeating),
                source: None,
            },
            Pos(glam::Vec2::ZERO),
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_beams);
        for _ in 0..3 {
            sched.run(&mut world);
        }
        let mut foes = world.query::<(&Enemy, &Health)>();
        let (_, hp) = foes.iter(&world).next().unwrap();
        assert_eq!(hp.hp, 20);
    }

    #[test]
    fn slash_lifetime_despawns() {
        let mut world = clocked_world();
        let slash = world
            .spawn((
                Team::Player,
                Projectile {
                    damage: 5,
                    life: GTimer::from_seconds(0.03, TimerMode::Once),
                    radius: 4.0,
                    knockback: 120.0,
                    explosive: false,
                    source: None,
                },
                Velocity(glam::Vec2::new(200.0, 0.0)),
                Pos(glam::Vec2::ZERO),
                SlashProjectile {
                    typ: 1,
                    shank: false,
                    walled: false,
                    hit: false,
                    guitar: false,
                    electric_guitar: false,
                    blood: false,
                    lightning: false,
                    hammer_wallbreak: false,
                    reach: 48.0,
                    back: 8.0,
                    half_width: 10.0,
                    dir: glam::Vec2::X,
                },
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_slash_projectiles);
        sched.run(&mut world);
        assert!(world.get_entity(slash).is_err(), "anim-end destroy");
    }

    #[test]
    fn slash_hits_enemy_in_arc() {
        let mut world = clocked_world();
        spawn_enemy_at(&mut world, glam::Vec2::new(30.0, 0.0), 20);
        world.spawn((
            Team::Player,
            Projectile {
                damage: 5,
                life: GTimer::from_seconds(5.0, TimerMode::Once),
                radius: 4.0,
                knockback: 120.0,
                explosive: false,
                source: None,
            },
            Velocity(glam::Vec2::new(200.0, 0.0)),
            Pos(glam::Vec2::ZERO),
            SlashProjectile {
                typ: 1,
                shank: false,
                walled: false,
                hit: false,
                guitar: false,
                electric_guitar: false,
                blood: false,
                lightning: false,
                hammer_wallbreak: false,
                reach: 48.0,
                back: 8.0,
                half_width: 10.0,
                dir: glam::Vec2::X,
            },
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_slash_projectiles);
        sched.run(&mut world);
        let mut foes = world.query::<(&Enemy, &Health)>();
        let (_, hp) = foes.iter(&world).next().unwrap();
        assert_eq!(hp.hp, 15);
    }

    #[test]
    fn slash_stops_at_wall_once() {
        let mut world = clocked_world();
        world.spawn((
            WallTile,
            WallCell(0, 0),
            Pos(glam::Vec2::new(30.0, 0.0)),
        ));
        let slash = world
            .spawn((
                Team::Player,
                Projectile {
                    damage: 5,
                    life: GTimer::from_seconds(5.0, TimerMode::Once),
                    radius: 4.0,
                    knockback: 120.0,
                    explosive: false,
                    source: None,
                },
                Velocity(glam::Vec2::new(200.0, 0.0)),
                Pos(glam::Vec2::ZERO),
                SlashProjectile {
                    typ: 1,
                    shank: false,
                    walled: false,
                    hit: false,
                    guitar: false,
                    electric_guitar: false,
                    blood: false,
                    lightning: false,
                    hammer_wallbreak: false,
                    reach: 48.0,
                    back: 8.0,
                    half_width: 10.0,
                    dir: glam::Vec2::X,
                },
            ))
            .id();
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_slash_projectiles);
        sched.run(&mut world);
        assert!(world.get::<SlashProjectile>(slash).unwrap().walled);
        assert_eq!(world.get::<Velocity>(slash).unwrap().0, glam::Vec2::ZERO);
    }

    #[test]
    fn stopped_slash_keeps_frozen_direction() {
        // Bevy freezes `tf.rotation` when the slash stops: post-wall
        // ticks keep testing the frozen direction, not +X.
        let mut world = clocked_world();
        spawn_enemy_at(&mut world, glam::Vec2::new(-30.0, 0.0), 20);
        // Live enemies carry a hurt gate (re-hits every 5 frames); the
        // shared helper omits it, so install one for the multi-tick run.
        let foe = world
            .query_filtered::<Entity, With<Enemy>>()
            .iter(&world)
            .next()
            .unwrap();
        world.entity_mut(foe).insert(NextHurt(0));
        world.spawn((
            WallTile,
            WallCell(0, 0),
            Pos(glam::Vec2::new(-60.0, 0.0)),
        ));
        world.spawn((
            Team::Player,
            Projectile {
                damage: 5,
                life: GTimer::from_seconds(5.0, TimerMode::Once),
                radius: 4.0,
                knockback: 120.0,
                explosive: false,
                source: None,
            },
            // Moving -X: latched dir is -X once the wall stops it.
            Velocity(glam::Vec2::new(-200.0, 0.0)),
            Pos(glam::Vec2::ZERO),
            SlashProjectile {
                typ: 1,
                shank: false,
                walled: false,
                hit: false,
                guitar: false,
                electric_guitar: false,
                blood: false,
                lightning: false,
                hammer_wallbreak: false,
                reach: 48.0,
                back: 8.0,
                half_width: 10.0,
                dir: glam::Vec2::NEG_X,
            },
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_slash_projectiles);
        sched.run(&mut world);
        sched.run(&mut world);
        let mut foes = world.query::<(&Enemy, &Health)>();
        let (_, hp) = foes.iter(&world).next().unwrap();
        assert_eq!(hp.hp, 15, "frozen -X slash still hits behind");
    }

    #[test]
    fn lingering_blast_catches_walk_ins_once() {
        // Bevy `LingeringBlast`: re-scans every 1/30 s for 0.75 s,
        // hitting each victim once (walk-ins caught, no double-hits).
        // Radius includes the target hitbox; hits knock back (cap 480).
        let mut world = clocked_world();
        spawn_enemy_at(&mut world, glam::Vec2::new(200.0, 0.0), 20);
        world.spawn((
            GameCleanup,
            LevelCleanup,
            Explosion {
                timer: GTimer::from_seconds(0.05, TimerMode::Once),
                radius: 60.0,
                damage: 6,
                team: Team::Player,
                hits_player: false,
                source: None,
            },
            LingeringBlast {
                duration: GTimer::from_seconds(0.75, TimerMode::Once),
                tick: GTimer::from_seconds(1.0 / 30.0, TimerMode::Repeating),
                hit: Vec::new(),
            },
            Pos(glam::Vec2::ZERO),
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(apply_explosions);
        fn foe_hp(world: &mut bevy_ecs::prelude::World) -> i32 {
            world
                .query::<(&Enemy, &Health)>()
                .iter(world)
                .next()
                .unwrap()
                .1
                .hp
        }
        sched.run(&mut world); // tick 1: pre-fuse scan, foe far
        assert_eq!(foe_hp(&mut world), 20);
        // Walk into radius (30 < 60 + hitbox 8).
        world
            .query_filtered::<&mut Pos, With<Enemy>>()
            .single_mut(&mut world)
            .unwrap()
            .0 = glam::Vec2::new(30.0, 0.0);
        sched.run(&mut world); // tick 2: fuse + scan
        assert_eq!(foe_hp(&mut world), 14, "walk-in caught");
        let vel = world
            .query_filtered::<&Velocity, With<Enemy>>()
            .single(&world)
            .unwrap()
            .0;
        assert!(vel.x > 100.0, "knockback applied, got {vel:?}");
        sched.run(&mut world); // tick 3: same victim, hit-set blocks
        assert_eq!(foe_hp(&mut world), 14, "no double-hit");
    }

    #[test]
    fn explosion_radius_includes_hitbox() {
        let mut world = clocked_world();
        // 67 < 60 + 8 (inside with padding), 73 > 68 (outside).
        spawn_enemy_at(&mut world, glam::Vec2::new(67.0, 0.0), 20);
        spawn_enemy_at(&mut world, glam::Vec2::new(73.0, 0.0), 20);
        world.spawn((
            GameCleanup,
            LevelCleanup,
            Explosion {
                timer: GTimer::from_seconds(0.01, TimerMode::Once),
                radius: 60.0,
                damage: 6,
                team: Team::Player,
                hits_player: false,
                source: None,
            },
            Pos(glam::Vec2::ZERO),
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(apply_explosions);
        sched.run(&mut world);
        let mut hps: Vec<i32> = world
            .query::<(&Enemy, &Health)>()
            .iter(&world)
            .map(|(_, h)| h.hp)
            .collect();
        hps.sort();
        assert_eq!(hps, vec![14, 20], "hitbox padding decides the edge");
    }

    #[test]
    fn explosion_damages_enemies_and_player() {        let mut world = clocked_world();
        spawn_enemy_at(&mut world, glam::Vec2::new(20.0, 0.0), 20);
        spawn_player_at(&mut world, glam::Vec2::new(-20.0, 0.0));
        world.spawn((
            GameCleanup,
            LevelCleanup,
            Explosion {
                timer: GTimer::from_seconds(0.03, TimerMode::Once),
                radius: 60.0,
                damage: 6,
                team: Team::Player,
                hits_player: true,
                source: None,
            },
            Pos(glam::Vec2::ZERO),
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(apply_explosions);
        sched.run(&mut world);
        let mut foes = world.query::<(&Enemy, &Health)>();
        let (_, hp) = foes.iter(&world).next().unwrap();
        assert_eq!(hp.hp, 14);
        let mut players = world.query::<(&Player, &Health)>();
        let (_, php) = players.iter(&world).next().unwrap();
        assert_eq!(php.hp, 14);
        assert!(world.resource::<SecretTriggers>().damage_taken_this_floor);
        assert_eq!(
            world.resource::<LastDamageTaken>().hit_id,
            None,
            "sourceless boom books no hit id"
        );
        let _ = HitId::Explosion(WeaponId::NONE);
        assert!(world.query::<&Explosion>().iter(&world).count() == 0);
    }

    #[test]
    fn hazard_cloud_ticks_opposing_team() {
        use crate::data::HazardKind;
        let mut world = clocked_world();
        spawn_enemy_at(&mut world, glam::Vec2::new(5.0, 0.0), 20);
        world.spawn((
            Team::Player,
            HazardCloud {
                kind: HazardKind::Fire,
                radius: 30.0,
                damage: 2,
                timer: GTimer::from_seconds(5.0, TimerMode::Once),
                tick: GTimer::from_seconds(0.03, TimerMode::Repeating),
            },
            Pos(glam::Vec2::ZERO),
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_hazard_clouds);
        for _ in 0..3 {
            sched.run(&mut world);
        }
        let mut foes = world.query::<(&Enemy, &Health)>();
        let (_, hp) = foes.iter(&world).next().unwrap();
        assert!(hp.hp < 20, "cloud ticked, hp {}", hp.hp);
    }

    #[test]
    fn player_fire_hazard_clouds_untouched_by_combat_tick() {
        use crate::data::HazardKind;
        let mut world = clocked_world();
        world.spawn((
            crate::comps_a::AbilityHazard,
            HazardCloud {
                kind: HazardKind::Toxic,
                radius: 30.0,
                damage: 2,
                timer: GTimer::from_seconds(5.0, TimerMode::Once),
                tick: GTimer::from_seconds(0.03, TimerMode::Repeating),
            },
            Pos(glam::Vec2::ZERO),
        ));
        let mut sched = bevy_ecs::schedule::Schedule::default();
        sched.add_systems(tick_hazard_clouds);
        sched.run(&mut world);
        assert_eq!(
            world.query::<&HazardCloud>().iter(&world).count(),
            1,
            "ability clouds belong to player_fire's tick"
        );
    }
}
