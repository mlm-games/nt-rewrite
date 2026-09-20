//! Enemy AI. Ported from the bevy reference `game/enemies.rs` (`enemy_ai`
/// and its spawn/fire/tick helpers) with positions as [`Pos`] (`Vec2`)
/// instead of `Transform.translation` (`Vec3`).
///
/// Render split: `Sprite`/`Anchor`/`Transform.rotation` writes, hurt/fire
/// strip swaps, `Juice::pop_in`, and `VfxSpawner` bursts stay out — the
/// render phase resolves visuals from sim state. Gameplay effects are
/// kept: movement impulses, projectile spawns with full combat traits
/// (`typ`/fade/friction/bounce/split/homing/fuse), pending-spawn queues,
/// trauma, hitstop, toasts, and audio stems routed as [`AudioCue`]s
/// through the [`Queue`] (the bevy build played them via direct asset
/// loads inside the 16-param system; here they queue like every other
/// sim system).
///
/// Timer adaptation: bevy `Timer` -> [`GTimer`]; `tick()` returns `()`,
/// then `just_finished()`/`finished()` are queried. The bevy
/// `ready_timer()` (finished from birth, silent until re-armed) has no
/// direct `GTimer` equivalent — `GTimer::disarmed()` reports
/// `just_finished()` on *every* tick — so the local [`ready_timer`]
/// double-ticks a 10 ms `Once` timer into the same observable state
/// (finished, not just-finished).
///
/// `enemy_ai` carries 15 params (under the bevy_ecs 16-param cap), so no
/// record-split is needed; the BigMaggot/Inspector tail already lives in
/// its own system in bevy ([`tick_bigmaggot_inspector`]) and is ported
/// as such.

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bevy_ecs::system::EntityCommands;
use rand::RngExt;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::anim::{SpriteAnim, derive_hurt_path, derive_walk_path};
use crate::audio::AudioCue;
use crate::combat::PendingEnemySpawn;
use crate::comps_a::{
    ARENA_H, ARENA_W, BossIntro, BouncesLeft, DamageSource, Euphoria, FloorMask, GameCleanup,
    GrenadeFuse, Health, Hitbox, Homing, LevelCleanup, NextHurt, Player, Projectile,
    ProjectileFade, ProjectileFriction, ProjectileTyp, Run, ShellWallBounce, SplitOnDeath, Team,
    Toast, Velocity, apply_gml_friction, gml_motion_add_clamp,
};
use crate::comps_b::{
    BossBrain, Corpse, EliteBlocker, Enemy, EnemyBrain, FxAngle, HazardCloud, HitWarning,
    HurtAnim, IdpdShieldUnit, IdpdVanBrain, LilHunterDie, MomShot, PendingDelayedBoss,
    PickupLifetime, PopoNadeM, PortalClear, Prop, ProtoGuardian, ShieldFollower, StaticFx,
    ThroneBall, TrapFire,
};
use crate::data::{AreaId, EnemyKind, HazardKind, SplitDef};
use crate::effects::{HitStop, spawn_burst};
use crate::enemy_data::{EnemyDef, enemy_def};
use crate::msg::Queue;
use crate::spatial::{Pos, clamp_to_arena, resolve_prop_collision};
use crate::time::{GTimer, TimerMode};

/// Floor-scaled HP multiplier (bevy `world::difficulty_multiplier`
/// parity: +5% per loop, +1.5% per route floor). Kept for the speed law
/// only; HP follows the GML spawn law below (no within-loop scaling).
pub fn difficulty_multiplier(floor: u32) -> f32 {
    let loop_n = ((floor.max(1) - 1) / 15) as f32;
    let rf = ((floor.max(1) - 1) % 15) as f32;
    1.0 + loop_n * 0.05 + rf * 0.015
}

/// GML spawn-HP law verbatim (`enemy/Create_0`: every enemy scales
/// `max_hp *= 1 + loops/20`; bosses override with their own formulas,
/// single-player values). `base_hp` is the table (loop-0) value.
///
/// Boss laws (all `1 + loops/3` except ScrapBoss `/1.2` and
/// ProtoStatue flat 120): BigBandit, Throne, ThroneII, Hyper+Technomancer
/// (coop `((pc/2)+0.5)` factor is 1.0 solo — GML `/` is float division,
/// so `(1/2)+0.5 = 1.0`), LilHunter, FrogQueen, Last (`Last`: base 1100).
/// Single-player-only bosses with no loop term stay flat: YV (700 +
/// coop-only scaling). `Mom`/`Captain`/`OldGuardian`/`PalaceGuardian`
/// have no GML object (spawn-table-only kinds); they ride the default
/// `/20` law like every other non-boss.
pub fn spawn_hp(kind: EnemyKind, base_hp: i32, loops: u32) -> i32 {
    let l = loops as f32;
    let hp = match kind {
        EnemyKind::BigBandit | EnemyKind::BigBanditLoop => (100.0 * (1.0 + l / 3.0)).ceil(),
        EnemyKind::BigDog | EnemyKind::BigDogLoop => (300.0 * (1.0 + l / 1.2)).ceil(),
        EnemyKind::Throne => 1500.0 * (1.0 + l / 3.0),
        EnemyKind::ThroneII => 600.0 * (1.0 + l / 3.0),
        EnemyKind::Hyper => 550.0 * (1.0 + l / 3.0),
        EnemyKind::Technomancer => 350.0 * (1.0 + l / 3.0),
        EnemyKind::LilHunter | EnemyKind::LilHunterLoop => 140.0 * (1.0 + l / 3.0),
        EnemyKind::FrogQueen => (490.0 * (1.0 + l / 3.0)).ceil(),
        EnemyKind::Captain => (1100.0 * (1.0 + l / 3.0)).ceil(),
        EnemyKind::ProtoStatue => 120.0,
        _ => (base_hp as f32 * (1.0 + l / 20.0)).ceil(),
    };
    hp.round().max(1.0) as i32
}

/// Full enemy spawn: base bundle from [`crate::setup::spawn_enemy`]
/// (cleanup markers, `Team`, `Pos`, `Velocity`, `Hitbox`, table `Enemy`)
/// plus difficulty/face/heart scaling, the randomized [`EnemyBrain`]
/// attack/strafe/gunangle state, boss/IDPD brains, and a [`SpriteAnim`]
/// seed when the catalog carries the idle strip (so `tick_fire_anims`
/// can resolve [`crate::comps_b::FireAnim`] markers from
/// [`show_enemy_fire`]; visual strips themselves stay render-side).
pub fn spawn_enemy(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: EnemyKind,
    pos: glam::Vec2,
    difficulty: f32,
    scarier_face: bool,
    heavy_heart: bool,
    loops: u32,
) -> Entity {
    let def = enemy_def(kind);
    let e = crate::setup::spawn_enemy(commands, kind, pos);

    // GML HP law (loop scaling); scarier-face 0.8 floor kept.
    let hp = spawn_hp(kind, def.hp, loops);
    let hp = if scarier_face {
        (hp as f32 * 0.8).floor() as i32
    } else {
        hp
    };
    let speed = def.speed * (0.9 + 0.02 * difficulty);
    let weapon_chance = if heavy_heart {
        def.weapon_chance + 9
    } else {
        def.weapon_chance
    };

    let mut ec = commands.entity(e);
    ec.insert(Health {
        hp,
        max: hp,
        invuln: ready_timer(),
    });
    ec.insert(Enemy {
        kind,
        score: def.score,
        touch_damage: def.touch_damage,
        rad_drop: def.rad_drop,
        drop_chance: def.drop_chance,
        weapon_chance,
    });
    ec.insert(NextHurt::default());
    ec.insert(Hitbox { radius: def.radius });
    ec.insert(EnemyBrain {
        speed,
        accel: def.accel,
        preferred_range: def.preferred_range,
        shoot_range: def.shoot_range,
        attack: GTimer::from_seconds(
            match kind {
                EnemyKind::Turret => (60.0 + rand::rng().random_range(0.0..60.0)) / 30.0,
                EnemyKind::SnowTank => (30.0 + rand::rng().random_range(0.0..10.0)) / 30.0,
                EnemyKind::GoldSnowtank => (120.0 + rand::rng().random_range(0.0..10.0)) / 30.0,
                EnemyKind::LaserCrystal
                | EnemyKind::LightningCrystal
                | EnemyKind::InvLaserCrystal => {
                    (50.0 + rand::rng().random_range(0.0..90.0)) / 30.0
                }
                EnemyKind::Guardian => (40.0 + rand::rng().random_range(0.0..10.0)) / 30.0,
                _ => def.attack_cooldown * rand::rng().random_range(0.5..1.5),
            },
            TimerMode::Once,
        ),
        burst_left: 0,
        burst_timer: ready_timer(),
        dash: 0.0,
        strafe_dir: if rand::rng().random_bool(0.5) {
            1.0
        } else {
            -1.0
        },
        strafe_timer: GTimer::from_seconds(rand::rng().random_range(0.8..1.6), TimerMode::Once),
        melee: ready_timer(),
        walk: 0.0,
        slash_delay: 0.0,
        ammo: match kind {
            EnemyKind::Scorpion | EnemyKind::GoldScorpion => 10,
            EnemyKind::IdpdGrunt => 2,
            EnemyKind::IdpdInspector => 4,
            EnemyKind::IdpdElite => 3,
            EnemyKind::Jock => 5,
            _ => 0,
        },
        gunangle: rand::rng().random_range(0.0..std::f32::consts::TAU),
    });
    if def.boss {
        ec.insert(BossBrain::new(kind, pos));
    }
    match kind {
        EnemyKind::IdpdVan => {
            ec.insert((IdpdVanBrain::default(), IdpdShieldUnit));
        }
        EnemyKind::IdpdShield => {
            ec.insert(IdpdShieldUnit);
        }
        EnemyKind::ProtoStatue => {
            ec.insert(ProtoGuardian::default());
        }
        _ => {}
    }
    // Bevy parity: every enemy carries its strip table (`EnemySprites`)
    // plus the seeded idle `SpriteAnim`; the switch/hurt/fire systems
    // resolve walk/hurt/fire strips from the table each tick.
    ec.insert(crate::comps_b::EnemySprites {
        idle: def.sprite,
        walk: derive_walk_path(def.sprite),
        hurt: derive_hurt_path(def.sprite),
    });
    // Seed the idle strip only (walk/hurt/fire strips resolve
    // renderer-side); without a catalog entry no anim rides along and
    // `show_enemy_fire` becomes a silent no-op for this enemy.
    if let Some(anim_def) = catalog.def(def.sprite) {
        ec.insert(SpriteAnim::new(def.sprite, anim_def));
    }
    e
}

/// Thin wrapper matching the bevy call-site order.
pub fn spawn_enemy_at(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: EnemyKind,
    pos: glam::Vec2,
    difficulty: f32,
    scarier_face: bool,
    heavy_heart: bool,
    loops: u32,
) -> Entity {
    spawn_enemy(
        commands,
        catalog,
        kind,
        pos,
        difficulty,
        scarier_face,
        heavy_heart,
        loops,
    )
}

/// Drain deferred spawns (bevy parity: face/heart modifiers are NOT
/// re-applied here — the bevy flush passes `false, false`).
pub fn flush_pending_enemy_spawns(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    pending: Query<(Entity, &PendingEnemySpawn)>,
) {
    for (entity, spawn) in pending.iter() {
        spawn_enemy_at(
            &mut commands,
            &catalog,
            spawn.kind,
            spawn.pos,
            spawn.difficulty,
            false,
            false,
            spawn.loops,
        );
        commands.entity(entity).despawn();
    }
}

/// Random arena point at least `min_from_center` from the origin
/// (bevy parity, including the corner fallback).
pub fn random_spawn_pos(rng: &mut impl RngExt, min_from_center: f32) -> glam::Vec2 {
    for _ in 0..64 {
        let x = rng.random_range(-ARENA_W / 2.0 + 80.0..ARENA_W / 2.0 - 80.0);
        let y = rng.random_range(-ARENA_H / 2.0 + 80.0..ARENA_H / 2.0 - 80.0);
        let p = glam::Vec2::new(x, y);
        if p.length() >= min_from_center {
            return p;
        }
    }
    glam::Vec2::new(ARENA_W / 2.0 - 80.0, 0.0)
}

/// Finished-from-birth, silent-until-re-armed timer (bevy `ready_timer`
/// parity — see module docs for why `GTimer::disarmed()` is wrong here).
fn ready_timer() -> GTimer {
    let mut t = GTimer::from_seconds(0.01, TimerMode::Once);
    t.tick(0.01);
    t.tick(0.0);
    t
}

/// Enemy telegraph cue through the sim audio queue (bevy
/// `play_enemy_cue` parity: 0.6 vol, 0.05 variance; the bevy build
/// loaded `audio/{stem}.wav` directly to dodge the 16-param cap — the
/// queue carries the stem name instead and the audio layer resolves it).
fn enemy_cue(cues: &mut Queue<AudioCue>, stem: &'static str) {
    cues.push(AudioCue {
        name: stem,
        volume: 0.6,
        variance: 0.05,
    });
}

/// Wall-aware sight check (bevy parity: 8 px samples, 16 px tile-center
/// recheck, arena-exterior samples ignored).
fn has_line_of_sight(from: glam::Vec2, to: glam::Vec2, mask: &FloorMask) -> bool {
    let dir = to - from;
    let dist = dir.length();
    if dist < 1.0 {
        return true;
    }
    let steps = (dist / 8.0).ceil().max(4.0) as usize;
    for i in 1..steps {
        let t = i as f32 / steps as f32;
        let p = from + dir * t;
        let tile_check = glam::Vec2::new(
            (p.x / 16.0).floor() * 16.0 + 8.0,
            (p.y / 16.0).floor() * 16.0 + 8.0,
        );
        if !mask.is_walkable(p) && !mask.is_walkable(tile_check) {
            if p.x.abs() < ARENA_W / 2.0 && p.y.abs() < ARENA_H / 2.0 {
                return false;
            }
        }
    }
    true
}

/// Non-boss enemy AI: walk impulses, dashes, charges, telegraphs, and
/// table-driven fire, per bevy `enemy_ai` top to bottom (bosses `continue`
/// before their first timer tick — boss brains live elsewhere).
#[allow(clippy::too_many_arguments)]
pub fn enemy_ai(
    time: Res<SimTime>,
    mut commands: Commands,
    mut trauma: ResMut<Trauma>,
    euphoria: Res<Euphoria>,
    mask: Res<FloorMask>,
    run: Res<Run>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut ratking_cd: Local<HashMap<Entity, GTimer>>,
    mut sniper_state: Local<HashMap<Entity, (GTimer, bool)>>,
    mut wolf_roll: Local<HashMap<Entity, GTimer>>,
    mut charge_state: Local<HashMap<Entity, GTimer>>,
    player_q: Query<(&Pos, &Player), (With<Player>, Without<Enemy>)>,
    mut enemies: Query<
        (
            Entity,
            &Enemy,
            &mut EnemyBrain,
            &mut Velocity,
            &mut Pos,
            Option<&BossBrain>,
            Option<&mut SpriteAnim>,
            Option<&HurtAnim>,
        ),
        (With<Enemy>, Without<Prop>),
    >,
    props: Query<(Entity, &Prop, &Pos), With<Prop>>,
    corpses: Query<(Entity, &Corpse, &Pos), (With<Corpse>, Without<Enemy>)>,
    catalog: Res<repame_anim::AnimCatalog>,
) {
    let Ok((player_pos, player)) = player_q.single() else {
        return;
    };
    let player_pos = player_pos.0;
    let dt = time.delta_secs;
    let mut rng = rand::rng();

    let euphoria = euphoria.0 || player.euphoria;

    // Pre-move snapshot for separation (bevy parity: pushes use the
    // snapshot, applied to the live position).
    let positions: Vec<glam::Vec2> =
        enemies.iter().map(|(_, _, _, _, pos, _, _, _)| pos.0).collect();

    for (entity, enemy, mut brain, mut vel, mut pos, boss, mut anim, hurt) in &mut enemies {
        let epos = pos.0;
        let to_player = player_pos - epos;
        let dist = to_player.length();
        let dir = to_player.normalize_or_zero();

        let def = enemy_def(enemy.kind);

        if boss.is_some() {
            continue;
        }

        if enemy.kind == EnemyKind::IdpdVan {
            vel.0 = glam::Vec2::ZERO;
            continue;
        }

        // Kinds with dedicated verbatim ticks below own their motion and
        // fire law; the generic chase must not double-drive them.
        if matches!(
            enemy.kind,
            EnemyKind::EliteInspector
                | EnemyKind::EliteShielder
                | EnemyKind::ScrapBossMissile
                | EnemyKind::ProtoStatue
        ) {
            continue;
        }

        let emplacement = matches!(
            enemy.kind,
            EnemyKind::Turret
                | EnemyKind::Crystal
                | EnemyKind::LaserCrystal
                | EnemyKind::LightningCrystal
                | EnemyKind::InvLaserCrystal
                | EnemyKind::MaggotSpawn
        );

        brain.melee.tick(dt);

        if brain.walk > 0.0 {
            let (impulse_f, cap_f) = match enemy.kind {
                EnemyKind::Scorpion | EnemyKind::GoldScorpion => (2.0, 4.0),
                EnemyKind::Bandit | EnemyKind::SnowBandit | EnemyKind::JungleBandit => (0.8, 3.0),
                EnemyKind::Maggot => (0.6, 2.0),
                EnemyKind::Rat | EnemyKind::Ratking => (0.8, 4.0),
                EnemyKind::Gator
                | EnemyKind::BuffGator
                | EnemyKind::Jock
                | EnemyKind::Molefish
                | EnemyKind::Molesarge
                | EnemyKind::BoneFish => (0.8, 3.0),
                EnemyKind::Raven => (0.8, 3.5),
                EnemyKind::Salamander => (2.0, 2.5),
                EnemyKind::Freak | EnemyKind::ExploFreak => (0.55, 4.0),
                EnemyKind::RhinoFreak => (0.8, 1.0),
                EnemyKind::Crab => (1.5, 4.5),
                EnemyKind::Turtle => (1.0, 5.0),
                EnemyKind::FireBaller => (0.6, 2.0),
                EnemyKind::SuperFireBaller => (0.6, 1.5),
                EnemyKind::SnowTank | EnemyKind::GoldSnowtank => (0.6, 1.5),
                EnemyKind::DogGuardian => (0.4, 2.0),
                EnemyKind::JungleFly => (0.8, 3.5),
                EnemyKind::Spider | EnemyKind::InvSpider => (2.0, 4.0),
                EnemyKind::Sniper => (0.8, 1.5),
                _ => (0.4, 4.0),
            };
            let walk_dir = if vel.0.length_squared() > 1.0 {
                vel.0.normalize_or_zero()
            } else {
                dir
            };
            gml_motion_add_clamp(&mut vel.0, walk_dir, impulse_f, cap_f, dt);
            brain.walk -= dt * 30.0;
            if brain.walk < 0.0 {
                brain.walk = 0.0;
            }
        }

        // Turtle fire-strip swap: visual-only, omitted (renderer resolves
        // strips from state; no gameplay effect).

        apply_gml_friction(&mut vel.0, 0.4, dt);

        if matches!(enemy.kind, EnemyKind::Bandit | EnemyKind::SnowBandit) {
            brain.attack.tick(dt);
            if brain.attack.just_finished() {
                let los = has_line_of_sight(epos, player_pos, &mask);
                if los {
                    if dist > 48.0 {
                        if rng.random::<f32>() < 0.25 {
                            let spread = rng.random_range(-10.0_f32..10.0).to_radians();
                            let base_ang = dir.y.atan2(dir.x);
                            let ang = base_ang + spread;
                            let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                            fire_enemy_bullet(
                                &mut commands,
                                &mut rng,
                                entity,
                                enemy,
                                def,
                                epos,
                                sdir,
                                euphoria,
                            );
                            show_enemy_fire(
                                &mut commands,
                                &catalog,
                                entity,
                                def.sprite,
                                anim.as_deref_mut(),
                                hurt.is_some(),
                            );
                            brain.gunangle = base_ang;
                            brain.attack = GTimer::from_seconds(
                                (20.0 + rng.random_range(0.0..5.0)) / 30.0,
                                TimerMode::Once,
                            );
                        } else {
                            let ang =
                                dir.y.atan2(dir.x) + rng.random_range(-90_f32..90.0).to_radians();
                            let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                            vel.0 = wdir * (0.4 * 30.0);
                            brain.walk = 10.0 + rng.random_range(0.0..10.0);
                            brain.gunangle = dir.y.atan2(dir.x);
                            brain.attack = GTimer::from_seconds(
                                (20.0 + rng.random_range(0.0..5.0)) / 30.0,
                                TimerMode::Once,
                            );
                        }
                    } else {
                        let away = -dir;
                        let ang =
                            away.y.atan2(away.x) + rng.random_range(-10_f32..10.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (0.4 * 30.0);
                        brain.walk = 40.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = dir.y.atan2(dir.x);
                        brain.attack = GTimer::from_seconds(
                            (20.0 + rng.random_range(0.0..5.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                } else if rng.random::<f32>() < 0.25 {
                    let ang = rng.random_range(0.0..std::f32::consts::TAU);
                    let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                    vel.0 = wdir * (0.4 * 30.0);
                    brain.walk = 20.0 + rng.random_range(0.0..10.0);
                    brain.attack = GTimer::from_seconds(
                        (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                        TimerMode::Once,
                    );
                    brain.gunangle = ang;
                } else {
                    // GML `Bandit/Alarm_1` refires unconditionally
                    // (`alarm[1] = 20 + random(10)` tops the event); without
                    // this the finished timer never re-arms and the bandit
                    // freezes at spawn whenever its first cycle has no
                    // line of sight (75% of no-LOS cycles).
                    brain.attack = GTimer::from_seconds(
                        (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                        TimerMode::Once,
                    );
                }
            }
            {
                if vel.0.length() > 90.0 {
                    vel.0 = vel.0.normalize() * 90.0;
                }
                pos.0 += vel.0 * dt;
                collide_enemy(&props, &mask, &mut pos, def.radius);
                separate(&positions, epos, &mut pos, def.radius);
            }
            continue;
        }

        if matches!(enemy.kind, EnemyKind::Scorpion | EnemyKind::GoldScorpion) {
            brain.attack.tick(dt);
            brain.burst_timer.tick(dt);

            if brain.ammo > 0 && brain.burst_left > 0 {
                if brain.burst_timer.just_finished() {
                    let gold = enemy.kind == EnemyKind::GoldScorpion;
                    if gold {
                        for (spd_lo, spd_hi, spread_deg) in
                            [(5.0_f32, 6.0_f32, 5.0_f32), (1.5_f32, 2.0_f32, 40.0_f32)]
                        {
                            let spread = rng.random_range(-spread_deg..spread_deg).to_radians();
                            let ang = brain.gunangle + spread;
                            let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                            let speed = rng.random_range(spd_lo..spd_hi) * 30.0;
                            let bullet = spawn_enemy_projectile(
                                &mut commands,
                                entity,
                                enemy.kind,
                                epos + sdir * 20.0,
                                sdir * speed,
                                def.projectile_damage,
                                def.projectile_lifetime,
                                def.projectile_radius,
                                150.0,
                                explosive_kind(enemy.kind),
                            );
                            // Gold-scorpion bullets are slashable (typ 2).
                            commands.entity(bullet).insert(ProjectileTyp(2));
                        }
                    } else {
                        let spread = rng.random_range(-20_f32..20.0).to_radians();
                        let base_ang = brain.gunangle;
                        let ang = base_ang + spread;
                        let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                        let speed = rng.random_range(90.0..120.0);
                        spawn_enemy_projectile(
                            &mut commands,
                            entity,
                            enemy.kind,
                            epos + sdir * 20.0,
                            sdir * speed,
                            def.projectile_damage,
                            def.projectile_lifetime,
                            def.projectile_radius,
                            150.0,
                            explosive_kind(enemy.kind),
                        );
                    }
                    brain.ammo = brain.ammo.saturating_sub(1);
                    brain.burst_left -= 1;
                    if brain.ammo == 0 || brain.burst_left == 0 {
                        brain.attack = GTimer::from_seconds(
                            (40.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                        brain.ammo = 10;
                    } else {
                        let frames = if enemy.kind == EnemyKind::GoldScorpion {
                            1.0
                        } else {
                            2.0
                        };
                        brain.burst_timer = GTimer::from_seconds(frames / 30.0, TimerMode::Once);
                    }
                }
            } else if brain.attack.just_finished() {
                let target_dir = dir.y.atan2(dir.x);
                let walk_ang = target_dir
                    + rng.random_range(-60_f32..60.0).to_radians()
                    + std::f32::consts::PI;
                let wdir = glam::Vec2::new(walk_ang.cos(), walk_ang.sin());
                vel.0 = wdir * (0.4 * 30.0);
                brain.walk = 10.0 + rng.random_range(0.0..10.0);

                let los = has_line_of_sight(epos, player_pos, &mask);
                // GML range gate is Scorpion-only (`scrTargetIsVisible`
                // 210); GoldScorpion fires at any visible range.
                let in_range =
                    enemy.kind == EnemyKind::GoldScorpion || dist < 210.0;
                if los && in_range && rng.random::<f32>() < 0.5 {
                    brain.attack = GTimer::from_seconds(
                        (30.0 + rng.random_range(0.0..5.0)) / 30.0,
                        TimerMode::Once,
                    );
                    brain.burst_timer = GTimer::from_seconds(1.0 / 30.0, TimerMode::Once);
                    if enemy.kind == EnemyKind::GoldScorpion {
                        brain.burst_left = 20;
                        brain.ammo = 20;
                    } else {
                        brain.burst_left = 10;
                        brain.ammo = 10;
                    }
                    brain.gunangle = target_dir;
                } else {
                    brain.attack = GTimer::from_seconds(
                        (30.0 + rng.random_range(0.0..10.0)) / 30.0,
                        TimerMode::Once,
                    );
                }
                if dist < 64.0 {
                    let away = -dir;
                    let ang = away.y.atan2(away.x) + rng.random_range(-10_f32..10.0).to_radians();
                    if dist > 32.0 {
                        let ang2 = ang + std::f32::consts::PI;
                        vel.0 = glam::Vec2::new(ang2.cos(), ang2.sin()) * (0.4 * 30.0);
                    } else {
                        vel.0 = glam::Vec2::new(ang.cos(), ang.sin()) * (0.4 * 30.0);
                    }
                    brain.walk = 40.0;
                }
            }

            if vel.0.length() > 120.0 {
                vel.0 = vel.0.normalize() * 120.0;
            }
            pos.0 += vel.0 * dt;
            collide_enemy(&props, &mask, &mut pos, def.radius);
            separate(&positions, epos, &mut pos, def.radius);
            continue;
        }

        let was_dashing = brain.dash > 0.0;

        if matches!(enemy.kind, EnemyKind::DogGuardian)
            && !was_dashing
            && dist < 160.0
            && dist > 40.0
            && brain.melee.is_finished()
            && rng.random::<f32>() < 0.67
        {
            brain.dash = 0.42;
            brain.melee = GTimer::from_seconds(10.0, TimerMode::Once);
            vel.0 = dir * 700.0;
        }

        if enemy.kind == EnemyKind::Raven && !was_dashing && brain.melee.is_finished() {
            brain.dash = 0.2;
            brain.melee = GTimer::from_seconds(rng.random_range(0.9..1.8), TimerMode::Once);
            let side = glam::Vec2::new(-dir.y, dir.x) * brain.strafe_dir;
            vel.0 = (dir * -0.35 + side).normalize() * 420.0;
        }

        if enemy.kind == EnemyKind::Guardian
            && brain.melee.is_finished()
            && dist < 480.0
            && rng.random_bool(0.012)
        {
            let want = def.preferred_range.max(140.0);
            let jump_dir = if dist > want { dir } else { -dir };
            let cand = epos + jump_dir * rng.random_range(70.0..130.0);
            if mask.is_walkable(cand) {
                pos.0 = cand;
                vel.0 = glam::Vec2::ZERO;
                spawn_burst(
                    &mut commands,
                    &mut rng,
                    epos,
                    8,
                    [0.3, 1.0, 0.45, 1.0],
                    (40.0, 120.0),
                );
            }
            brain.melee = GTimer::from_seconds(2.2, TimerMode::Once);
        }

        if enemy.kind == EnemyKind::PalaceGuardian
            && !was_dashing
            && dist < 80.0
            && dist > 24.0
            && brain.melee.is_finished()
        {
            brain.dash = 0.18;
            brain.melee = GTimer::from_seconds(0.9, TimerMode::Once);
            vel.0 = dir * 540.0;
        }
        if brain.dash > 0.0 {
            brain.dash = (brain.dash - dt).max(0.0);
        }

        let dashing = brain.dash > 0.0;

        if emplacement {
            vel.0 = glam::Vec2::ZERO;
        } else if dashing {
            pos.0 += vel.0 * dt;
        } else if brain.speed > 0.0 {
            brain.attack.tick(dt);
            if brain.attack.just_finished() {
                let los = has_line_of_sight(epos, player_pos, &mask);
                let base_ang = dir.y.atan2(dir.x);

                let (impulse, _cap, far_walk, close_walk, wander_walk) = match enemy.kind {
                    EnemyKind::Maggot => (0.6, 2.0, 8.0..14.0, 12.0..18.0, 10.0..20.0),
                    EnemyKind::Gator | EnemyKind::BuffGator => {
                        (0.8, 3.0, 10.0..14.0, 40.0..50.0, 20.0..30.0)
                    }
                    EnemyKind::Freak | EnemyKind::ExploFreak => {
                        (0.55, 4.0, 18.0..22.0, 12.0..18.0, 10.0..16.0)
                    }
                    EnemyKind::RhinoFreak => (0.8, 1.0, 18.0..22.0, 12.0..18.0, 10.0..16.0),
                    EnemyKind::Spider | EnemyKind::InvSpider => {
                        (2.0, 5.0, 15.0..20.0, 10.0..14.0, 10.0..20.0)
                    }
                    EnemyKind::Crab => (1.5, 4.5, 8.0..14.0, 50.0..60.0, 20.0..30.0),
                    EnemyKind::Turtle => (1.0, 5.0, 40.0..60.0, 40.0..60.0, 40.0..60.0),
                    EnemyKind::Salamander => (2.0, 2.5, 40.0..50.0, 20.0..30.0, 10.0..20.0),
                    EnemyKind::Sniper => (0.8, 1.5, 10.0..14.0, 40.0..50.0, 20.0..30.0),
                    EnemyKind::FireBaller | EnemyKind::SuperFireBaller => {
                        (0.6, 2.0, 8.0..12.0, 10.0..14.0, 10.0..16.0)
                    }
                    EnemyKind::Jock => (0.8, 3.0, 10.0..14.0, 40.0..50.0, 20.0..30.0),
                    EnemyKind::Molefish | EnemyKind::Molesarge => {
                        (0.8, 3.5, 10.0..14.0, 20.0..30.0, 20.0..30.0)
                    }
                    EnemyKind::Raven => (0.8, 3.5, 20.0..30.0, 40.0..50.0, 20.0..30.0),
                    EnemyKind::Rat
                    | EnemyKind::Ratking
                    | EnemyKind::FastRat
                    | EnemyKind::BigRat => (0.8, 4.0, 10.0..16.0, 40.0..50.0, 10.0..25.0),
                    EnemyKind::Wolf => (0.8, 4.0, 10.0..16.0, 20.0..30.0, 12.0..20.0),
                    EnemyKind::Assassin | EnemyKind::MeleeBandit => {
                        (0.8, 4.0, 10.0..14.0, 20.0..28.0, 16.0..24.0)
                    }
                    EnemyKind::Ballguy => (0.6, 3.0, 12.0..18.0, 12.0..18.0, 12.0..18.0),
                    EnemyKind::LightningCrystal => (0.5, 1.5, 10.0..14.0, 10.0..14.0, 10.0..20.0),
                    _ => (0.4, 4.0, 6.0..14.0, 18.0..28.0, 10.0..18.0),
                };

                if los {
                    if dist > 80.0 {
                        if rng.random::<f32>() < 0.35 {
                            brain.walk = 0.0;
                            vel.0 *= 0.5;
                        } else {
                            let ang = base_ang + rng.random_range(-45_f32..45.0).to_radians();
                            let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                            vel.0 = wdir * (impulse * 30.0);
                            brain.walk = rng.random_range(far_walk);
                            brain.gunangle = base_ang;
                        }
                    } else if dist < 44.0 {
                        let away = -dir;
                        let ang =
                            away.y.atan2(away.x) + rng.random_range(-15_f32..15.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (impulse * 30.0);
                        brain.walk = rng.random_range(close_walk);
                        brain.gunangle = base_ang;
                    } else {
                        let ang = base_ang + rng.random_range(-90_f32..90.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (impulse * 30.0);
                        brain.walk = rng.random_range(6.0..10.0);
                    }

                    let attack_secs = match enemy.kind {
                        EnemyKind::Maggot => rng.random_range(30.0..50.0) / 30.0,
                        EnemyKind::Rat
                        | EnemyKind::FastRat
                        | EnemyKind::BigRat
                        | EnemyKind::Ratking => rng.random_range(10.0..40.0) / 30.0,
                        EnemyKind::Freak | EnemyKind::ExploFreak => {
                            rng.random_range(6.0..11.0) / 30.0
                        }
                        EnemyKind::RhinoFreak | EnemyKind::DogGuardian | EnemyKind::Turtle => {
                            rng.random_range(6.0..11.0) / 30.0
                        }
                        EnemyKind::Spider | EnemyKind::InvSpider => {
                            rng.random_range(20.0..30.0) / 30.0
                        }
                        EnemyKind::Crab => rng.random_range(10.0..20.0) / 30.0,
                        EnemyKind::Salamander => rng.random_range(10.0..60.0) / 30.0,
                        EnemyKind::Sniper => rng.random_range(20.0..30.0) / 30.0,
                        EnemyKind::Assassin | EnemyKind::MeleeBandit | EnemyKind::Wolf => {
                            rng.random_range(6.0..11.0) / 30.0
                        }
                        _ => rng.random_range(0.35..0.75),
                    };
                    brain.attack = GTimer::from_seconds(attack_secs, TimerMode::Once);
                } else if rng.random::<f32>() < 0.4 {
                    let ang = rng.random_range(0.0..std::f32::consts::TAU);
                    let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                    vel.0 = wdir * (impulse * 30.0);
                    brain.walk = rng.random_range(wander_walk);
                    brain.attack = GTimer::from_seconds(
                        (brain.walk + 10.0 + rng.random_range(0.0..18.0)) / 30.0,
                        TimerMode::Once,
                    );
                } else {
                    brain.attack = GTimer::from_seconds(rng.random_range(0.3..0.6), TimerMode::Once);
                }
            }

            if vel.0.length() > brain.speed {
                vel.0 = vel.0.normalize() * brain.speed;
            }
            pos.0 += vel.0 * dt;
        } else {
            vel.0 = glam::Vec2::ZERO;
        }

        collide_enemy(&props, &mask, &mut pos, def.radius);

        // InvSpider/InvLaserCrystal fade: visual-only, omitted.

        separate(&positions, epos, &mut pos, def.radius);

        if enemy.kind == EnemyKind::Necromancer {
            brain.attack.tick(dt);
            if brain.attack.just_finished() {
                brain.attack = GTimer::from_seconds(def.attack_cooldown, TimerMode::Once);
                let mut best: Option<(Entity, glam::Vec2)> = None;
                let mut best_d = 160.0;
                for (ce, _, cpos) in &corpses {
                    let d = cpos.0.distance(epos);
                    if d < best_d {
                        best_d = d;
                        best = Some((ce, cpos.0));
                    }
                }
                if let Some((ce, cpos)) = best {
                    commands.entity(ce).despawn();
                    let revived = if run.loop_count >= 1
                        && matches!(run.area, AreaId::Labs | AreaId::Palace)
                    {
                        EnemyKind::PopoFreak
                    } else {
                        EnemyKind::Freak
                    };
                    commands.spawn(PendingEnemySpawn {
                        kind: revived,
                        pos: cpos,
                        difficulty: 1.0,
                        loops: run.loop_count,
                    });
                } else if (positions.len() as u32) < 40 {
                    let ang = rng.random_range(0.0..std::f32::consts::TAU);
                    let p = epos + glam::Vec2::new(ang.cos(), ang.sin()) * 40.0;
                    commands.spawn(PendingEnemySpawn {
                        kind: EnemyKind::Freak,
                        pos: p,
                        difficulty: 1.0,
                        loops: run.loop_count,
                    });
                }
            }
        }

        if enemy.kind == EnemyKind::MaggotSpawn {
            brain.attack.tick(dt);
            if brain.attack.just_finished() {
                brain.attack = GTimer::from_seconds(def.attack_cooldown, TimerMode::Once);
                let ang = rng.random_range(0.0..std::f32::consts::TAU);
                commands.spawn(PendingEnemySpawn {
                    kind: EnemyKind::Maggot,
                    pos: epos + glam::Vec2::new(ang.cos(), ang.sin()) * 24.0,
                    difficulty: 1.0,
                    loops: run.loop_count,
                });
            }
        }

        if enemy.kind == EnemyKind::Ratking {
            let cd = ratking_cd
                .entry(entity)
                .or_insert_with(|| GTimer::from_seconds(1.0, TimerMode::Once));
            if brain.burst_left > 0 {
                brain.burst_timer.tick(dt);
                if brain.burst_timer.just_finished() {
                    let spread = rng.random_range(-20_f32..20.0).to_radians();
                    let base = dir.y.atan2(dir.x);
                    let ang = base + spread;
                    let off = glam::Vec2::new(ang.cos(), ang.sin()) * 12.0;
                    commands.spawn(PendingEnemySpawn {
                        kind: EnemyKind::FastRat,
                        pos: epos + off,
                        difficulty: 1.0,
                        loops: run.loop_count,
                    });
                    brain.burst_left = brain.burst_left.saturating_sub(1);
                    if brain.burst_left > 0 {
                        brain.burst_timer = GTimer::from_seconds(6.0 / 30.0, TimerMode::Once);
                    }
                }
            } else {
                cd.tick(dt);
                if cd.just_finished() {
                    let los = has_line_of_sight(epos, player_pos, &mask);
                    if los && rng.random::<f32>() < 0.34 {
                        brain.burst_left = rng.random_range(3..=5);
                        brain.burst_timer = GTimer::from_seconds(6.0 / 30.0, TimerMode::Once);
                        *cd = GTimer::from_seconds(
                            (30.0 + rng.random_range(0.0..5.0)) / 30.0,
                            TimerMode::Once,
                        );
                    } else {
                        *cd = GTimer::from_seconds(
                            (30.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                }
            }
        }

        if enemy.kind == EnemyKind::Sniper {
            let (cd, aiming) = sniper_state.entry(entity).or_insert_with(|| {
                (
                    GTimer::from_seconds(
                        (60.0 + rand::rng().random_range(0.0..90.0)) / 30.0,
                        TimerMode::Once,
                    ),
                    false,
                )
            });
            cd.tick(dt);
            if *aiming && cd.remaining_secs() > 5.0 / 30.0 {
                brain.gunangle = dir.y.atan2(dir.x);
            }
            if cd.just_finished() {
                let los = has_line_of_sight(epos, player_pos, &mask);
                if !*aiming {
                    if los {
                        if dist > 96.0 {
                            if rng.random::<f32>() < 0.67 {
                                enemy_cue(&mut cues, "sndSniperTarget");
                                *aiming = true;
                                brain.gunangle = dir.y.atan2(dir.x);
                                brain.walk = 0.0;
                                vel.0 = glam::Vec2::ZERO;
                                *cd = GTimer::from_seconds(1.0 /* 30 ticks */, TimerMode::Once);
                            } else {
                                let ang = dir.y.atan2(dir.x)
                                    + rng.random_range(-80_f32..80.0).to_radians();
                                let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                                vel.0 = wdir * (0.4 * 30.0);
                                brain.walk = 10.0 + rng.random_range(0.0..10.0);
                                brain.gunangle = dir.y.atan2(dir.x);
                                *cd = GTimer::from_seconds(
                                    (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                                    TimerMode::Once,
                                );
                            }
                        } else {
                            let away = -dir;
                            let ang =
                                away.y.atan2(away.x) + rng.random_range(-10_f32..10.0).to_radians();
                            let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                            vel.0 = wdir * (0.4 * 30.0);
                            brain.walk = 40.0 + rng.random_range(0.0..10.0);
                            brain.gunangle = dir.y.atan2(dir.x);
                            *cd = GTimer::from_seconds(
                                (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                                TimerMode::Once,
                            );
                        }
                    } else if rng.random::<f32>() < 0.25 {
                        let ang = rng.random_range(0.0..std::f32::consts::TAU);
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (0.4 * 30.0);
                        brain.walk = 20.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = ang;
                        *cd = GTimer::from_seconds(
                            (brain.walk + 2.0 + rng.random_range(0.0..5.0)) / 30.0,
                            TimerMode::Once,
                        );
                    } else {
                        *cd = GTimer::from_seconds(
                            (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                } else {
                    *aiming = false;
                    enemy_cue(&mut cues, "sndSniperFire");
                    let base = brain.gunangle;
                    for off in [4.0_f32, -4.0, 0.0] {
                        let ang = base + off.to_radians();
                        let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                        spawn_enemy_projectile(
                            &mut commands,
                            entity,
                            enemy.kind,
                            epos + sdir * 20.0,
                            sdir * def.projectile_speed,
                            def.projectile_damage,
                            def.projectile_lifetime,
                            def.projectile_radius,
                            150.0,
                            explosive_kind(enemy.kind),
                        );
                    }
                    show_enemy_fire(
                        &mut commands,
                        &catalog,
                        entity,
                        def.sprite,
                        anim.as_deref_mut(),
                        hurt.is_some(),
                    );
                    *cd = GTimer::from_seconds(
                        (40.0 + rng.random_range(0.0..5.0)) / 30.0,
                        TimerMode::Once,
                    );
                }
            }
        }

        if matches!(
            enemy.kind,
            EnemyKind::LaserCrystal
                | EnemyKind::LightningCrystal
                | EnemyKind::InvLaserCrystal
                | EnemyKind::SnowTank
                | EnemyKind::GoldSnowtank
                | EnemyKind::Guardian
                | EnemyKind::ExploGuardian
        ) {
            let (charge_frames, min_range, max_range, aim_chance): (f32, f32, f32, f32) =
                match enemy.kind {
                    EnemyKind::LaserCrystal | EnemyKind::InvLaserCrystal => {
                        (30.0, 64.0, 160.0, 1.0)
                    }
                    EnemyKind::LightningCrystal => (20.0, 0.0, 96.0, 1.0),
                    EnemyKind::SnowTank => (40.0, 64.0, 240.0, 1.0 / 6.0),
                    EnemyKind::GoldSnowtank => (10.0, 64.0, 160.0, 0.5),
                    EnemyKind::Guardian => (12.0, 0.0, 999.0, 1.0),
                    EnemyKind::ExploGuardian => (60.0, 0.0, 90.0, 1.0),
                    _ => (30.0, 0.0, 999.0, 1.0),
                };
            if let Some(chg) = charge_state.get_mut(&entity) {
                chg.tick(dt);
                if chg.just_finished() {
                    charge_state.remove(&entity);
                    let gdir = glam::Vec2::new(brain.gunangle.cos(), brain.gunangle.sin());
                    if matches!(enemy.kind, EnemyKind::SnowTank | EnemyKind::GoldSnowtank) {
                        brain.burst_left = 16;
                        brain.burst_timer = GTimer::from_seconds(2.0 / 30.0, TimerMode::Once);
                        brain.strafe_dir = 0.0;
                    } else {
                        brain.burst_left = def.bullets_per_shot;
                        brain.burst_timer = GTimer::from_seconds(def.burst_interval, TimerMode::Once);
                        fire_enemy_bullet(
                            &mut commands,
                            &mut rng,
                            entity,
                            enemy,
                            def,
                            epos,
                            gdir,
                            euphoria,
                        );
                        brain.burst_left = brain.burst_left.saturating_sub(1);
                    }
                    show_enemy_fire(
                        &mut commands,
                        &catalog,
                        entity,
                        def.sprite,
                        anim.as_deref_mut(),
                        hurt.is_some(),
                    );
                    if enemy.kind == EnemyKind::GoldSnowtank {
                        let gdir = glam::Vec2::new(brain.gunangle.cos(), brain.gunangle.sin());
                        spawn_enemy_projectile(
                            &mut commands,
                            entity,
                            enemy.kind,
                            epos + gdir * 20.0,
                            gdir * 60.0,
                            5,
                            3.0,
                            5.0,
                            150.0,
                            true,
                        );
                    }
                }
            } else if brain.burst_left == 0 {
                brain.attack.tick(dt);
                if brain.attack.just_finished() {
                    let los = has_line_of_sight(epos, player_pos, &mask);
                    let in_range = dist >= min_range && dist <= max_range;
                    let guardian_ok = if enemy.kind == EnemyKind::Guardian {
                        los && ((dist > 96.0 && rng.random::<f32>() < 0.67)
                            || rng.random::<f32>() < 0.33)
                    } else {
                        los && in_range && rng.random::<f32>() < aim_chance
                    };
                    let explo_ok = if enemy.kind == EnemyKind::ExploGuardian {
                        los && dist <= 90.0
                    } else {
                        guardian_ok
                    };
                    if explo_ok {
                        brain.gunangle = dir.y.atan2(dir.x);
                        show_enemy_fire(
                            &mut commands,
                            &catalog,
                            entity,
                            def.sprite,
                            anim.as_deref_mut(),
                            hurt.is_some(),
                        );
                        match enemy.kind {
                            EnemyKind::LaserCrystal | EnemyKind::InvLaserCrystal => {
                                enemy_cue(&mut cues, "sndLaserCrystalCharge");
                            }
                            EnemyKind::LightningCrystal => {
                                enemy_cue(&mut cues, "sndLightningCrystalCharge");
                            }
                            EnemyKind::SnowTank => {
                                enemy_cue(&mut cues, "sndSnowTankAim");
                            }
                            EnemyKind::GoldSnowtank => {
                                enemy_cue(&mut cues, "sndGoldTankAim");
                            }
                            EnemyKind::ExploGuardian => {
                                enemy_cue(&mut cues, "sndExploGuardianCharge");
                            }
                            _ => {}
                        }
                        charge_state.insert(
                            entity,
                            GTimer::from_seconds(charge_frames / 30.0, TimerMode::Once),
                        );
                        brain.attack = GTimer::from_seconds(
                            def.attack_cooldown + charge_frames / 30.0,
                            TimerMode::Once,
                        );
                    } else {
                        let cd_secs = match enemy.kind {
                            EnemyKind::SnowTank => (40.0 + rng.random_range(0.0..30.0)) / 30.0,
                            EnemyKind::GoldSnowtank => (15.0 + rng.random_range(0.0..5.0)) / 30.0,
                            EnemyKind::Guardian => (10.0 + rng.random_range(0.0..40.0)) / 30.0,
                            EnemyKind::ExploGuardian => (6.0 + rng.random_range(0.0..5.0)) / 30.0,
                            EnemyKind::LaserCrystal | EnemyKind::InvLaserCrystal => {
                                (30.0 + rng.random_range(0.0..10.0)) / 30.0
                            }
                            _ => def.attack_cooldown,
                        };
                        brain.attack = GTimer::from_seconds(cd_secs, TimerMode::Once);
                        if matches!(enemy.kind, EnemyKind::SnowTank | EnemyKind::GoldSnowtank) {
                            let base =
                                dir.y.atan2(dir.x) + std::f32::consts::FRAC_PI_2 * brain.strafe_dir;
                            let wdir = glam::Vec2::new(base.cos(), base.sin());
                            vel.0 = wdir * (0.6 * 30.0);
                            brain.walk = 20.0 + rng.random_range(0.0..10.0);
                        }
                    }
                }
            } else if brain.burst_left > 0 {
                brain.burst_timer.tick(dt);
                if brain.burst_timer.just_finished() {
                    if matches!(enemy.kind, EnemyKind::SnowTank | EnemyKind::GoldSnowtank) {
                        let wave = brain.strafe_dir;
                        let amp = if enemy.kind == EnemyKind::SnowTank {
                            20.0_f32.to_radians()
                        } else {
                            15.0_f32.to_radians()
                        };
                        for i in [-1.0_f32, 1.0] {
                            let ang = brain.gunangle + (wave.sin() * amp) * i;
                            let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                            spawn_enemy_projectile(
                                &mut commands,
                                entity,
                                enemy.kind,
                                epos + sdir * 20.0,
                                sdir * def.projectile_speed,
                                def.projectile_damage,
                                def.projectile_lifetime,
                                def.projectile_radius,
                                150.0,
                                false,
                            );
                        }
                        brain.strafe_dir = wave + 0.1;
                        brain.burst_left = brain.burst_left.saturating_sub(1);
                        brain.burst_timer = GTimer::from_seconds(2.0 / 30.0, TimerMode::Once);
                        if brain.burst_left == 0 {
                            brain.attack = GTimer::from_seconds(
                                match enemy.kind {
                                    EnemyKind::SnowTank => {
                                        (40.0 + rng.random_range(0.0..30.0)) / 30.0
                                    }
                                    _ => (15.0 + rng.random_range(0.0..5.0)) / 30.0,
                                },
                                TimerMode::Once,
                            );
                        }
                    } else {
                        let gdir = glam::Vec2::new(brain.gunangle.cos(), brain.gunangle.sin());
                        fire_enemy_bullet(
                            &mut commands,
                            &mut rng,
                            entity,
                            enemy,
                            def,
                            epos,
                            gdir,
                            euphoria,
                        );
                        brain.burst_left = brain.burst_left.saturating_sub(1);
                        if brain.burst_left == 0 {
                            brain.attack = GTimer::from_seconds(def.attack_cooldown, TimerMode::Once);
                        } else {
                            brain.burst_timer =
                                GTimer::from_seconds(def.burst_interval, TimerMode::Once);
                        }
                    }
                    show_enemy_fire(
                        &mut commands,
                        &catalog,
                        entity,
                        def.sprite,
                        anim.as_deref_mut(),
                        hurt.is_some(),
                    );
                }
            }
        }

        if enemy.kind == EnemyKind::Wolf {
            let cd = wolf_roll
                .entry(entity)
                .or_insert_with(|| GTimer::from_seconds(1.5, TimerMode::Once));
            cd.tick(dt);
            if cd.just_finished() {
                let los = has_line_of_sight(epos, player_pos, &mask);
                if los && rng.random::<f32>() < 0.5 {
                    let base = dir.y.atan2(dir.x);
                    for off in [0.0_f32, 20.0, -20.0] {
                        let ang = base + off.to_radians();
                        let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                        spawn_enemy_projectile(
                            &mut commands,
                            entity,
                            enemy.kind,
                            epos + sdir * 20.0,
                            sdir * 120.0,
                            2,
                            3.0,
                            4.0,
                            150.0,
                            false,
                        );
                    }
                    show_enemy_fire(
                        &mut commands,
                        &catalog,
                        entity,
                        def.sprite,
                        anim.as_deref_mut(),
                        hurt.is_some(),
                    );
                }
                *cd = GTimer::from_seconds(
                    (30.0 + rng.random_range(0.0..20.0)) / 30.0,
                    TimerMode::Once,
                );
            }
        }

        if matches!(enemy.kind, EnemyKind::MeleeBandit) {
            brain.attack.tick(dt);
            if brain.attack.just_finished() {
                let los = has_line_of_sight(epos, player_pos, &mask);
                if los {
                    if dist < 64.0 {
                        let gdir = glam::Vec2::new(brain.gunangle.cos(), brain.gunangle.sin());
                        vel.0 = gdir * (6.0 * 30.0);
                        brain.gunangle = dir.y.atan2(dir.x);
                        enemy_cue(&mut cues, "sndAssassinAttack");
                        spawn_hit_warning(&mut commands, epos);
                        brain.slash_delay = 10.0;
                        brain.attack = GTimer::from_seconds(
                            (43.0 + rng.random_range(0.0..6.0)) / 30.0,
                            TimerMode::Once,
                        );
                    } else {
                        let ang = dir.y.atan2(dir.x) + rng.random_range(-10_f32..10.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (0.4 * 30.0);
                        brain.walk = 40.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = dir.y.atan2(dir.x);
                        brain.attack = GTimer::from_seconds(
                            (10.0 + rng.random_range(0.0..5.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                } else if rng.random::<f32>() < 0.25 {
                    let ang = rng.random_range(0.0..std::f32::consts::TAU);
                    vel.0 = glam::Vec2::new(ang.cos(), ang.sin()) * (0.4 * 30.0);
                    brain.walk = 20.0 + rng.random_range(0.0..10.0);
                    brain.attack = GTimer::from_seconds(
                        (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                        TimerMode::Once,
                    );
                } else {
                    brain.attack = GTimer::from_seconds(
                        (10.0 + rng.random_range(0.0..5.0)) / 30.0,
                        TimerMode::Once,
                    );
                }
            }
        }

        if matches!(enemy.kind, EnemyKind::Gator | EnemyKind::BuffGator) {
            brain.attack.tick(dt);
            if brain.attack.just_finished() {
                let los = has_line_of_sight(epos, player_pos, &mask);
                if los {
                    if dist > 48.0 && dist < 128.0 && rng.random::<f32>() < 0.34 {
                        brain.gunangle = dir.y.atan2(dir.x);
                        spawn_hit_warning(&mut commands, epos);
                        brain.walk = if enemy.kind == EnemyKind::BuffGator {
                            -15.0
                        } else {
                            -10.0
                        };
                        brain.attack = GTimer::from_seconds(
                            (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                    } else if dist <= 48.0 || dist >= 128.0 {
                        let ang =
                            (-dir).y.atan2((-dir).x) + rng.random_range(-10_f32..10.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (0.4 * 30.0);
                        brain.walk = 40.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = dir.y.atan2(dir.x);
                        brain.attack = GTimer::from_seconds(
                            (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                    } else {
                        let ang = dir.y.atan2(dir.x) + rng.random_range(-90_f32..90.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (0.4 * 30.0);
                        brain.walk = 10.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = dir.y.atan2(dir.x);
                        brain.attack = GTimer::from_seconds(
                            (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                } else {
                    brain.attack = GTimer::from_seconds(
                        (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                        TimerMode::Once,
                    );
                }
            }
            if brain.walk < 0.0 {
                brain.walk += dt * 30.0;
                if brain.walk >= 0.0 {
                    brain.walk = 0.0;
                    brain.gunangle = dir.y.atan2(dir.x);
                    if enemy.kind == EnemyKind::BuffGator {
                        enemy_cue(&mut cues, "sndFlakCannon");
                        let ang = brain.gunangle + rng.random_range(-25_f32..25.0).to_radians();
                        let spd = rng.random_range(240.0..300.0);
                        fire_enemy_flak(&mut commands, entity, enemy.kind, epos, ang, spd);
                    } else {
                        enemy_cue(&mut cues, "sndShotgun");
                        for _ in 0..6 {
                            let ang =
                                brain.gunangle + rng.random_range(-25_f32..25.0).to_radians();
                            let spd = rng.random_range(300.0..420.0);
                            fire_enemy_shell(&mut commands, entity, enemy.kind, epos, ang, spd);
                        }
                    }
                    trauma.add(0.1);
                }
            }
        }

        if matches!(
            enemy.kind,
            EnemyKind::IdpdGrunt | EnemyKind::IdpdInspector | EnemyKind::IdpdElite
        ) && brain.ammo > 0
        {
            let los = has_line_of_sight(epos, player_pos, &mask);
            let should_throw = (!los && dist > 64.0) || (los && rng.random::<f32>() < 0.02);
            if should_throw {
                brain.ammo = brain.ammo.saturating_sub(1);
                let base = dir.y.atan2(dir.x);
                let ang = base + rng.random_range(-10_f32..10.0).to_radians();
                let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                commands
                    .spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Projectile {
                            damage: 5,
                            life: GTimer::from_seconds(2.0, TimerMode::Once),
                            radius: 6.0,
                            knockback: 150.0,
                            explosive: true,
                            source: Some(DamageSource::enemy(entity, enemy.kind)),
                        },
                        Velocity(sdir * 300.0),
                        Pos(epos + sdir * 16.0),
                        BouncesLeft(255),
                        ProjectileFriction(0.1),
                        GrenadeFuse {
                            smoke_armed: false,
                            friction_switched: false,
                            alarm1: GTimer::from_seconds(6.0 / 30.0, TimerMode::Once),
                        },
                    ))
                    .insert(ProjectileTyp(1));
            }
        }

        if matches!(
            enemy.kind,
            EnemyKind::Mimic | EnemyKind::SuperMimic | EnemyKind::WepMimic
        ) {
            brain.attack.tick(dt);
            if brain.attack.just_finished() {
                // Tell-strip swap: visual-only, omitted; the taunt cue stays.
                if enemy.kind == EnemyKind::SuperMimic || enemy.kind == EnemyKind::WepMimic {
                    enemy_cue(&mut cues, "sndHPMimicTaunt");
                } else {
                    enemy_cue(&mut cues, "sndMimicSlurp");
                }
                let cd_secs = if enemy.kind == EnemyKind::SuperMimic {
                    (150.0 + rng.random_range(0.0..180.0)) / 30.0
                } else {
                    (90.0 + rng.random_range(0.0..150.0)) / 30.0
                };
                brain.attack = GTimer::from_seconds(cd_secs, TimerMode::Once);
            }
            if dist < 200.0 {
                let chase = dir * 60.0;
                vel.0 = vel.0.lerp(chase, 0.1);
            }
        }

        if matches!(enemy.kind, EnemyKind::IdpdShield) {
            brain.melee.tick(dt);

            if brain.slash_delay > 0.0 {
                brain.slash_delay -= dt * 30.0;
                if brain.slash_delay <= 0.0 {
                    brain.slash_delay = 0.0;
                    // Melee slash visual: GML never damages with it, so the
                    // headless build keeps the timing state only.
                }
            }
            if brain.melee.just_finished() {
                // PopoShield follower: sim side keeps the owner-tracking
                // marker; the shield art resolves renderer-side.
                let base = brain.gunangle;
                let off = glam::Vec2::new(base.cos(), base.sin()) * 16.0;
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    ShieldFollower { owner: entity },
                    Pos(epos + off),
                ));
                brain.melee = GTimer::from_seconds(85.0 / 30.0, TimerMode::Once);
            }
        }

        if enemy.kind == EnemyKind::FrogQueen {
            if brain.burst_left > 0 {
                brain.burst_timer.tick(dt);
                if brain.burst_timer.just_finished() {
                    if (positions.len() as u32) < 30 {
                        commands.spawn(PendingEnemySpawn {
                            kind: EnemyKind::FrogEgg,
                            pos: epos,
                            difficulty: 1.0,
                            loops: run.loop_count,
                        });
                    }
                    brain.burst_left = brain.burst_left.saturating_sub(1);
                    if brain.burst_left > 0 {
                        brain.burst_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
                    } else {
                        let base =
                            dir.y.atan2(dir.x) + rng.random_range(-15_f32..15.0).to_radians();
                        let sdir = glam::Vec2::new(base.cos(), base.sin());
                        spawn_enemy_projectile(
                            &mut commands,
                            entity,
                            enemy.kind,
                            epos + sdir * 20.0,
                            sdir * 120.0,
                            5,
                            4.0,
                            6.0,
                            150.0,
                            false,
                        );
                        show_enemy_fire(
                            &mut commands,
                            &catalog,
                            entity,
                            def.sprite,
                            anim.as_deref_mut(),
                            hurt.is_some(),
                        );
                        brain.attack = GTimer::from_seconds(
                            (30.0 + rng.random_range(0.0..20.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                }
            } else {
                brain.attack.tick(dt);
                if brain.attack.just_finished() {
                    let los = has_line_of_sight(epos, player_pos, &mask);
                    if los && rng.random::<f32>() < 0.25 {
                        brain.burst_left = (2 + run.loop_count).min(6) as usize;
                        brain.burst_timer = GTimer::from_seconds(10.0 / 30.0, TimerMode::Once);
                    } else {
                        brain.walk = 50.0;
                        brain.attack = GTimer::from_seconds(
                            (30.0 + rng.random_range(0.0..20.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                }
            }
        }

        if enemy.kind == EnemyKind::Jock {
            brain.attack.tick(dt);
            if brain.attack.just_finished() {
                let los = has_line_of_sight(epos, player_pos, &mask);
                if los {
                    if dist > 96.0 {
                        brain.gunangle = dir.y.atan2(dir.x);
                        let chance = 8 - brain.ammo as i32;
                        if brain.ammo > 0 && rng.random_range(0..chance.max(1)) == 0 {
                            brain.ammo = brain.ammo.saturating_sub(1);
                            let base = brain.gunangle;
                            let ang = base + rng.random_range(-10_f32..10.0).to_radians();
                            let sdir = glam::Vec2::new(ang.cos(), ang.sin());
                            commands
                                .spawn((
                                    GameCleanup,
                                    LevelCleanup,
                                    Team::Enemy,
                                    Projectile {
                                        damage: def.projectile_damage,
                                        life: GTimer::from_seconds(
                                            def.projectile_lifetime,
                                            TimerMode::Once,
                                        ),
                                        radius: def.projectile_radius,
                                        knockback: 150.0,
                                        explosive: true,
                                        source: Some(DamageSource::enemy(entity, enemy.kind)),
                                    },
                                    Velocity(sdir * 60.0),
                                    Pos(epos + sdir * 20.0),
                                    Homing {
                                        turn_rate: 3.0,
                                        acquire_range: 600.0,
                                    },
                                ))
                                .insert(ProjectileTyp(2));
                            show_enemy_fire(
                                &mut commands,
                                &catalog,
                                entity,
                                def.sprite,
                                anim.as_deref_mut(),
                                hurt.is_some(),
                            );
                            brain.attack = GTimer::from_seconds(8.0 / 30.0, TimerMode::Once);
                        } else if rng.random::<f32>() < 0.67 {
                            let ang =
                                dir.y.atan2(dir.x) + rng.random_range(-40_f32..40.0).to_radians();
                            let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                            vel.0 = wdir * (0.4 * 30.0);
                            brain.walk = 10.0 + rng.random_range(0.0..10.0);
                            brain.gunangle = dir.y.atan2(dir.x);
                            brain.attack = GTimer::from_seconds(
                                (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                                TimerMode::Once,
                            );
                        } else {
                            brain.attack = GTimer::from_seconds(
                                (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                                TimerMode::Once,
                            );
                        }
                    } else {
                        let ang = dir.y.atan2(dir.x) + rng.random_range(-5_f32..5.0).to_radians();
                        let wdir = glam::Vec2::new(ang.cos(), ang.sin());
                        vel.0 = wdir * (0.4 * 30.0);
                        brain.walk = 40.0 + rng.random_range(0.0..10.0);
                        brain.gunangle = dir.y.atan2(dir.x);
                        brain.attack = GTimer::from_seconds(
                            (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                            TimerMode::Once,
                        );
                    }
                } else if rng.random::<f32>() < 0.25 {
                    let ang = rng.random_range(0.0..std::f32::consts::TAU);
                    vel.0 = glam::Vec2::new(ang.cos(), ang.sin()) * (0.4 * 30.0);
                    brain.walk = 20.0 + rng.random_range(0.0..10.0);
                    brain.attack = GTimer::from_seconds(
                        (brain.walk + 10.0 + rng.random_range(0.0..30.0)) / 30.0,
                        TimerMode::Once,
                    );
                } else {
                    brain.attack = GTimer::from_seconds(
                        (20.0 + rng.random_range(0.0..10.0)) / 30.0,
                        TimerMode::Once,
                    );
                }
            }
        }

        let uses_charge = matches!(
            enemy.kind,
            EnemyKind::LaserCrystal
                | EnemyKind::LightningCrystal
                | EnemyKind::InvLaserCrystal
                | EnemyKind::SnowTank
                | EnemyKind::GoldSnowtank
                | EnemyKind::Guardian
                | EnemyKind::ExploGuardian
                | EnemyKind::Jock
        );
        if def.bullets_per_shot > 0
            && dist < brain.shoot_range
            && !dashing
            && !uses_charge
            && !matches!(enemy.kind, EnemyKind::Gator | EnemyKind::BuffGator)
        {
            if def.burst {
                if brain.burst_left > 0 {
                    brain.burst_timer.tick(dt);
                    if brain.burst_timer.just_finished() {
                        fire_enemy_bullet(
                            &mut commands,
                            &mut rng,
                            entity,
                            enemy,
                            def,
                            epos,
                            dir,
                            euphoria,
                        );
                        show_enemy_fire(
                            &mut commands,
                            &catalog,
                            entity,
                            def.sprite,
                            anim.as_deref_mut(),
                            hurt.is_some(),
                        );
                        brain.burst_left -= 1;
                        if brain.burst_left == 0 {
                            brain.attack = GTimer::from_seconds(def.attack_cooldown, TimerMode::Once);
                        }
                    }
                } else {
                    brain.attack.tick(dt);
                    if brain.attack.just_finished() {
                        brain.burst_left = def.bullets_per_shot;
                        brain.burst_timer =
                            GTimer::from_seconds(def.burst_interval, TimerMode::Once);
                        fire_enemy_bullet(
                            &mut commands,
                            &mut rng,
                            entity,
                            enemy,
                            def,
                            epos,
                            dir,
                            euphoria,
                        );
                        show_enemy_fire(
                            &mut commands,
                            &catalog,
                            entity,
                            def.sprite,
                            anim.as_deref_mut(),
                            hurt.is_some(),
                        );
                        brain.burst_left -= 1;
                    }
                }
            } else {
                brain.attack.tick(dt);
                if brain.attack.just_finished() {
                    fire_enemy_shot(&mut commands, &mut rng, entity, enemy, def, epos, dir);
                    show_enemy_fire(
                        &mut commands,
                        &catalog,
                        entity,
                        def.sprite,
                        anim.as_deref_mut(),
                        hurt.is_some(),
                    );
                    brain.attack = GTimer::from_seconds(def.attack_cooldown, TimerMode::Once);
                }
            }
        }
    }
}

/// BigMaggot burrow/relocate + Inspector mind-control pull (bevy
/// `tick_bigmaggot_inspector` parity; burst VFX rides `spawn_burst`,
/// everything else is positions and timers).
pub fn tick_bigmaggot_inspector(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    mut burrow: Local<HashMap<Entity, (f32, bool)>>,
    mut ctrl: Local<HashMap<Entity, f32>>,
    player_q: Query<(&Pos, &Player), (With<Player>, Without<Enemy>)>,
    mut player_vel: Query<&mut Velocity, (With<Player>, Without<Enemy>)>,
    healths: Query<&Health, (With<Enemy>, Without<Player>)>,
    mut enemies: Query<(Entity, &Enemy, &mut Velocity, &mut Pos), With<Enemy>>,
) {
    let Ok((player_pos, _)) = player_q.single() else {
        return;
    };
    let player_pos = player_pos.0;
    let dt = time.delta_secs;
    let mut rng = rand::rng();

    for (entity, enemy, mut vel, mut pos) in &mut enemies {
        let epos = pos.0;
        let to_player = player_pos - epos;
        let dist = to_player.length();
        let dir = to_player.normalize_or_zero();

        if enemy.kind == EnemyKind::BigMaggot {
            let los = has_line_of_sight(epos, player_pos, &mask);
            let entry = burrow.entry(entity).or_insert((0.0, false));
            if los {
                entry.0 = 0.0;
                if !entry.1 {
                    entry.1 = true;
                    vel.0 = dir * 90.0;
                }
            } else {
                entry.0 += dt;
                let damaged = healths.get(entity).map(|h| h.hp < h.max).unwrap_or(false);
                if damaged && entry.0 > 1.0 && rng.random::<f32>() < dt * 0.5 {
                    let ang = rng.random_range(0.0..std::f32::consts::TAU);
                    let dest = player_pos + glam::Vec2::new(ang.cos(), ang.sin()) * 64.0;
                    if mask.is_walkable(dest) {
                        pos.0 = dest;
                        vel.0 = glam::Vec2::ZERO;
                        entry.0 = 0.0;
                        entry.1 = false;
                        spawn_burst(
                            &mut commands,
                            &mut rng,
                            dest,
                            10,
                            [0.6, 0.45, 0.3, 1.0],
                            (40.0, 140.0),
                        );
                    }
                }
            }
        }

        if enemy.kind == EnemyKind::IdpdInspector && dist < 240.0 && dist > 1.0 {
            let los = has_line_of_sight(epos, player_pos, &mask);
            if los {
                let t = ctrl.entry(entity).or_insert(0.0);
                *t += dt;
                if *t > 0.5 {
                    if let Ok(mut pv) = player_vel.single_mut() {
                        let pull = (epos - player_pos).normalize_or_zero() * 30.0;
                        pv.0 += pull * dt;
                        if pv.0.length() > 200.0 {
                            pv.0 = pv.0.normalize() * 200.0;
                        }
                    }
                }
            } else {
                ctrl.remove(&entity);
            }
        } else if enemy.kind == EnemyKind::IdpdInspector {
            ctrl.remove(&entity);
        }
    }
}

/// Fire-strip swap for the render phase (bevy `play_fire` parity:
/// repath the live anim to the `derive_fire_path` strip as a 0.25 s
/// oneshot; skipped for hurting enemies or missing art).
pub fn show_enemy_fire(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    entity: Entity,
    idle: &'static str,
    anim: Option<&mut SpriteAnim>,
    hurting: bool,
) {
    if hurting {
        return;
    }
    let Some(anim) = anim else {
        return;
    };
    let Some(fire_path) = derive_fire_path(idle) else {
        return;
    };
    let Some(def) = catalog.def(fire_path) else {
        return;
    };
    anim.set_path(fire_path, def, true);
    commands.entity(entity).try_insert(crate::comps_b::FireAnim {
        idle,
        walk: None,
        timer: GTimer::from_seconds(0.25, TimerMode::Once),
    });
}

/// Bevy `derive_fire_path` verbatim: idle strip -> fire strip.
pub fn derive_fire_path(idle: &'static str) -> Option<&'static str> {
    match idle {
        "images/sprBanditBossIdle.png" => Some("images/sprBanditBossFire.png"),
        "images/sprCrabIdle.png" => Some("images/sprCrabFire.png"),
        "images/sprExploGuardianIdle.png" => Some("images/sprExploGuardianFire.png"),
        "images/sprFireBallerIdle.png" => Some("images/sprFireBallerFire.png"),
        "images/sprSuperFireBallerIdle.png" => Some("images/sprSuperFireBallerFire.png"),
        "images/sprFrogQueenIdle.png" => Some("images/sprFrogQueenFire.png"),
        "images/sprGoldScorpionIdle.png" => Some("images/sprGoldScorpionFire.png"),
        "images/sprGuardianIdle.png" => Some("images/sprGuardianFire.png"),
        "images/sprInvLaserCrystalIdle.png" => Some("images/sprInvLaserCrystalFire.png"),
        "images/sprJockIdle.png" => Some("images/sprJockFire.png"),
        "images/sprLaserCrystalIdle.png" => Some("images/sprLaserCrystalFire.png"),
        "images/sprLightningCrystalIdle.png" => Some("images/sprLightningCrystalFire.png"),
        "images/sprRatkingIdle.png" => Some("images/sprRatkingFire.png"),
        "images/sprSalamanderIdle.png" => Some("images/sprSalamanderFire.png"),
        "images/sprScorpionIdle.png" => Some("images/sprScorpionFire.png"),
        "images/sprScrapBossIdle.png" => Some("images/sprScrapBossFire.png"),
        "images/sprSnowBotIdle.png" => Some("images/sprSnowBotFire.png"),
        "images/sprTurretIdle.png" => Some("images/sprTurretFire.png"),
        "images/sprTurtleIdle.png" => Some("images/sprTurtleFire.png"),
        "images/sprWolfIdle.png" => Some("images/sprWolfFire.png"),
        "images/sprMimicIdle.png" => Some("images/sprMimicFire.png"),
        "images/sprSuperMimicIdle.png" => Some("images/sprSuperMimicFire.png"),
        "images/sprWepMimicIdle.png" => Some("images/sprWepMimicFire.png"),
        _ => None,
    }
}

/// Shared single-projectile spawn for enemy fire (art handles omitted;
/// the renderer derives the strip from `Projectile` + owner kind).
fn spawn_enemy_projectile(
    commands: &mut Commands,
    owner: Entity,
    kind: EnemyKind,
    pos: glam::Vec2,
    vel: glam::Vec2,
    damage: i32,
    lifetime: f32,
    radius: f32,
    knockback: f32,
    explosive: bool,
) -> Entity {
    commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            Team::Enemy,
            Projectile {
                damage,
                life: GTimer::from_seconds(lifetime, TimerMode::Once),
                radius,
                knockback,
                explosive,
                source: Some(DamageSource::enemy(owner, kind)),
            },
            Velocity(vel),
            Pos(pos),
        ))
        .id()
}

/// Gator Alarm_2 pellet (friction 0.6, full bounce, deflectable shell;
/// JungleBandit fires the same object).
fn fire_enemy_shell(
    commands: &mut Commands,
    owner: Entity,
    kind: EnemyKind,
    pos: glam::Vec2,
    angle: f32,
    speed: f32,
) {
    let sdir = glam::Vec2::new(angle.cos(), angle.sin());
    let e = spawn_enemy_projectile(
        commands,
        owner,
        kind,
        pos + sdir * 16.0,
        sdir * speed,
        1,
        3.0,
        3.5,
        150.0,
        false,
    );
    commands.entity(e).insert((
        ProjectileFriction(0.6),
        BouncesLeft(255),
        // GML EnemyBullet3 wall: cap 18 px/step, decay 0.9, no re-arm.
        ShellWallBounce {
            add: 0.0,
            cap: 540.0,
            decay: 0.9,
            rearm: None,
        },
        ProjectileTyp(1),
        ProjectileFade("images/sprEBullet3Disappear.png"),
    ));
}

/// BuffGator Alarm_2 flak (friction 0.4, no direct damage; splits into
/// 16 shells on death).
fn fire_enemy_flak(
    commands: &mut Commands,
    owner: Entity,
    kind: EnemyKind,
    pos: glam::Vec2,
    angle: f32,
    speed: f32,
) {
    let sdir = glam::Vec2::new(angle.cos(), angle.sin());
    let e = spawn_enemy_projectile(
        commands,
        owner,
        kind,
        pos + sdir * 16.0,
        sdir * speed,
        0,
        3.0,
        6.0,
        150.0,
        false,
    );
    commands.entity(e).insert((
        ProjectileFriction(0.4),
        ProjectileTyp(1),
        ProjectileFade("images/sprEnemyBulletHit.png"),
        SplitOnDeath(SplitDef {
            pellets: 16,
            spread: std::f32::consts::PI,
            speed: 300.0,
            damage: 1,
            lifetime: 2.5,
            radius: 3.0,
            knockback: 150.0,
            color: [1.0, 0.7, 0.3, 1.0],
            size: glam::Vec2::new(8.0, 3.0),
        }),
    ));
}

/// Single aimed enemy bullet with table spread (euphoria slows enemy
/// shots to 80%, bevy parity).
#[allow(clippy::too_many_arguments)]
pub fn fire_enemy_bullet(
    commands: &mut Commands,
    rng: &mut impl RngExt,
    owner: Entity,
    enemy: &Enemy,
    def: EnemyDef,
    pos: glam::Vec2,
    dir: glam::Vec2,
    euphoria: bool,
) {
    let base = dir.y.atan2(dir.x);
    let angle = base + rng.random_range(-def.projectile_spread..def.projectile_spread);
    let shot_dir = glam::Vec2::new(angle.cos(), angle.sin());
    let speed = def.projectile_speed * if euphoria { 0.8 } else { 1.0 };
    let e = spawn_enemy_projectile(
        commands,
        owner,
        enemy.kind,
        pos + shot_dir * 20.0,
        shot_dir * speed,
        def.projectile_damage,
        def.projectile_lifetime,
        def.projectile_radius,
        150.0,
        explosive_kind(enemy.kind),
    );
    finish_enemy_bullet(&mut commands.entity(e), enemy.kind);
}

/// Only Jock rockets explode on contact (bevy parity).
pub fn explosive_kind(kind: EnemyKind) -> bool {
    matches!(kind, EnemyKind::Jock)
}

/// Per-kind projectile traits: slash `typ` (0 = ignores slashes,
/// 1 = deflectable, 2 = slash-destructible), contact fade, and the
/// gator-family friction+bounce shell profile.
pub fn finish_enemy_bullet(ec: &mut EntityCommands, kind: EnemyKind) {
    let typ = match kind {
        EnemyKind::Scorpion | EnemyKind::GoldScorpion => 2,
        EnemyKind::Guardian | EnemyKind::Turtle => 0,
        EnemyKind::ExploGuardian | EnemyKind::Jock => 2,
        _ => 1,
    };
    ec.insert(ProjectileTyp(typ));
    if !explosive_kind(kind) {
        let fade_path = if matches!(
            kind,
            EnemyKind::Scorpion | EnemyKind::GoldScorpion
        ) {
            "images/sprScorpionBulletHit.png"
        } else if matches!(
            kind,
            EnemyKind::IdpdGrunt | EnemyKind::IdpdInspector | EnemyKind::IdpdElite
        ) {
            "images/sprIDPDBulletHit.png"
        } else if matches!(
            kind,
            EnemyKind::Gator
                | EnemyKind::BuffGator
                | EnemyKind::JungleBandit
                | EnemyKind::Molesarge
        ) {
            "images/sprEBullet3Disappear.png"
        } else {
            "images/sprEnemyBulletHit.png"
        };
        ec.insert(ProjectileFade(fade_path));
    }
    if matches!(
        kind,
        EnemyKind::Gator | EnemyKind::BuffGator | EnemyKind::JungleBandit | EnemyKind::Molesarge
    ) {
        ec.insert(ProjectileFriction(0.6));
        ec.insert(BouncesLeft(255));
        ec.insert(ShellWallBounce {
            add: 0.0,
            cap: 540.0,
            decay: 0.9,
            rearm: None,
        });
    }
}

/// Fan volley: `bullets_per_shot` pellets around the aim with per-pellet
/// jitter (SuperFireBaller's per-pellet speeds preserved).
#[allow(clippy::too_many_arguments)]
pub fn fire_enemy_shot(
    commands: &mut Commands,
    rng: &mut impl RngExt,
    owner: Entity,
    enemy: &Enemy,
    def: EnemyDef,
    pos: glam::Vec2,
    dir: glam::Vec2,
) {
    let base = dir.y.atan2(dir.x);
    let total = def.bullets_per_shot;
    for i in 0..total {
        let offset = if total > 1 {
            (i as f32 - (total as f32 - 1.0) * 0.5) * def.fan_spread
        } else {
            0.0
        };
        let angle = base + offset + rng.random_range(-0.06..0.06);
        let shot_dir = glam::Vec2::new(angle.cos(), angle.sin());
        let speed = if enemy.kind == EnemyKind::SuperFireBaller {
            [90.0, 120.0, 150.0][(i as usize).min(2)]
        } else {
            def.projectile_speed
        };
        let e = spawn_enemy_projectile(
            commands,
            owner,
            enemy.kind,
            pos + shot_dir * 20.0,
            shot_dir * speed,
            def.projectile_damage,
            def.projectile_lifetime,
            def.projectile_radius,
            150.0,
            explosive_kind(enemy.kind),
        );
        finish_enemy_bullet(&mut commands.entity(e), enemy.kind);
    }
}

/// Flush a kill-gated boss spawn: once enough trash died, pop the boss
/// out of a wall (or open floor), shake, toast, and hitstop (bevy
/// parity, including the hardcoded "BIG BANDIT" toast).
#[allow(clippy::too_many_arguments)]
pub fn tick_delayed_boss_spawns(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    run: Res<Run>,
    mask: Res<FloorMask>,
    mut trauma: ResMut<Trauma>,
    mut hitstop: ResMut<HitStop>,
    mut toast: ResMut<Toast>,
    pending: Query<(Entity, &PendingDelayedBoss)>,
    enemies: Query<&Enemy, With<Enemy>>,
    player_q: Query<&Pos, With<Player>>,
    walls: Query<
        (
            Entity,
            &crate::comps_a::WallCell,
            &Pos,
            Option<&crate::comps_b::ScreenEnd>,
        ),
        With<crate::comps_a::WallTile>,
    >,
) {
    let Ok((marker_e, pending_boss)) = pending.single() else {
        return;
    };

    let living_trash = enemies
        .iter()
        .filter(|e| !enemy_def(e.kind).boss)
        .count() as u32;
    let killed = pending_boss.initial_trash.saturating_sub(living_trash);
    if killed < pending_boss.kills_needed() {
        return;
    }

    let Ok(player_pos) = player_q.single() else {
        return;
    };
    let player_pos = player_pos.0;

    let mut best_wall: Option<(glam::Vec2, (i32, i32))> = None;
    let mut best_score = f32::MAX;
    if pending_boss.from_wall {
        for (_, cell, pos, screen_end) in &walls {
            let p = pos.0;
            let d = p.distance(player_pos);
            if d < 120.0 || d > 260.0 {
                continue;
            }
            let mut score = (d - 180.0).abs() + (p.y - player_pos.y).abs() * 0.25;
            if screen_end.is_some() {
                score -= 20.0;
            }
            if score < best_score {
                best_score = score;
                best_wall = Some((p, (cell.0, cell.1)));
            }
        }
    }

    let spawn_pos = if let Some((p, _)) = best_wall {
        p
    } else {
        let mut rng = rand::rng();
        let mut best = mask.random_floor_pos(&mut rng, 120.0);
        for _ in 0..32 {
            let ang = rng.random_range(0.0..std::f32::consts::TAU);
            let cand =
                player_pos + glam::Vec2::new(ang.cos(), ang.sin()) * rng.random_range(140.0..240.0);
            if mask.is_walkable(cand) {
                best = cand;
                break;
            }
        }
        best
    };

    commands.entity(marker_e).despawn();
    trauma.add(0.3);

    if let Some((p, cell)) = best_wall {
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)] {
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                crate::comps_a::PendingWallBreak {
                    cell: (cell.0 + dx, cell.1 + dy),
                    pos: p,
                    spawn_floor: true,
                },
            ));
        }
    }

    spawn_enemy_at(
        &mut commands,
        &catalog,
        pending_boss.kind,
        spawn_pos,
        difficulty_multiplier(run.floor),
        false,
        false,
        run.loop_count,
    );

    commands.spawn((
        GameCleanup,
        BossIntro {
            timer: GTimer::from_seconds(1.1, TimerMode::Once),
        },
    ));
    toast.show("BIG BANDIT");
    hitstop.trigger(0.2, 0.15);
}

/// Frog egg hatch: the egg despawns into a pending Ballguy plus an
/// 8-way acid ring (bevy parity).
pub fn tick_frog_eggs(
    time: Res<SimTime>,
    mut commands: Commands,
    run: Res<Run>,
    mut q: Query<(Entity, &Enemy, &mut EnemyBrain, &Pos), With<Enemy>>,
) {
    for (e, enemy, mut brain, pos) in &mut q {
        if enemy.kind != EnemyKind::FrogEgg {
            continue;
        }
        brain.attack.tick(time.delta_secs);
        if !brain.attack.just_finished() {
            continue;
        }
        let hatch = pos.0;
        commands.entity(e).despawn();

        commands.spawn(PendingEnemySpawn {
            kind: EnemyKind::Ballguy,
            pos: hatch,
            difficulty: 1.0,
            loops: run.loop_count,
        });

        for i in 0..8 {
            let ang = (i as f32) * std::f32::consts::TAU / 8.0;
            let d = glam::Vec2::new(ang.cos(), ang.sin());
            spawn_enemy_projectile(
                &mut commands,
                e,
                enemy.kind,
                hatch,
                d * 240.0,
                3,
                1.1,
                4.0,
                100.0,
                false,
            );
        }
    }
}

/// GML `LilHunter/Destroy_0:13-21` head spawn (`LilHunterDie/Create_0`):
/// speed 2 px/step toward the nearest player (else random),
/// `sndLilHunterBreak` + `PortalClear`.
/// (`scrOnPopoKill` has no port equivalent — skipped with this note;
/// its music-cue side effects ride the audio layer.)
pub fn spawn_lil_hunter_die(
    commands: &mut Commands,
    cues: &mut Queue<AudioCue>,
    pos: glam::Vec2,
    team: Team,
    target: Option<Entity>,
    target_pos: Option<glam::Vec2>,
) -> Entity {
    let mut rng = rand::rng();
    let dir = target_pos
        .map(|t| (t - pos).normalize_or_zero())
        .filter(|d| d.length_squared() > 0.001)
        .unwrap_or_else(|| {
            glam::Vec2::from_angle(rng.random_range(0.0..std::f32::consts::TAU))
        });
    let e = commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            team,
            LilHunterDie {
                // GML `trn = ((random(5) + 5) * choose(1, -1))`.
                trn: (rng.random_range(0.0..5.0) + 5.0)
                    * if rng.random_bool(0.5) { 1.0 } else { -1.0 },
                bounces: 0,
                target,
            },
            Velocity(dir * 2.0 * 30.0),
            Pos(pos),
            FxAngle(dir.y.atan2(dir.x).to_degrees() - 90.0),
        ))
        .id();
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        PortalClear {
            timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                        scale: 1.0,
        },
        Pos(pos),
    ));
    cues.push(AudioCue {
        name: "sndLilHunterBreak",
        volume: 0.8,
        variance: 0.2,
    });
    e
}

/// GML `LilHunter/Destroy_0` 80-`TrapFire` ring (`sprFireLilHunter`,
/// speed 2+random(0.2) px/step stepping 4.5°, `move_contact_solid`
/// baked as a clamped ≤12 px slide, team inherited).
pub fn spawn_lil_hunter_trapfire(commands: &mut Commands, at: glam::Vec2, team: Team) {
    let mut rng = rand::rng();
    let mut ang = rng.random_range(0.0..std::f32::consts::TAU);
    for _ in 0..80 {
        ang += 4.5f32.to_radians();
        let d = glam::Vec2::from_angle(ang);
        let speed = (2.0 + rng.random_range(0.0..0.2)) * 30.0;
        let mut p = at + d * 12.0;
        p.x = p.x.clamp(-ARENA_W * 0.5 + 8.0, ARENA_W * 0.5 - 8.0);
        p.y = p.y.clamp(-ARENA_H * 0.5 + 8.0, ARENA_H * 0.5 - 8.0);
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            team,
            TrapFire,
            HazardCloud {
                kind: HazardKind::Fire,
                radius: 10.0,
                damage: 2,
                timer: GTimer::from_seconds(4.0, TimerMode::Once),
                tick: GTimer::from_seconds(0.5, TimerMode::Repeating),
            },
            Velocity(d * speed),
            Pos(p),
        ));
    }
}

/// GML `LilHunterDie/Step_0` + `Collision_Wall`: per-step Smoke,
/// `image_angle = direction-90`, accel while `bounces <= 3`
/// (`speed < 6 → +2`, `+0.05`; `direction += trn`;
/// `trn += random(1)-0.5`; past 3 `direction += 59`), wall bounces
/// counted (past 3 the head stalls; otherwise a `PortalClear` pops).
pub fn tick_lil_hunter_die(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(
        Entity,
        &mut Pos,
        &mut Velocity,
        &mut LilHunterDie,
        Option<&mut FxAngle>,
    )>,
) {
    let dt = time.delta_secs;
    let mut rng = rand::rng();
    for (e, mut pos, mut vel, mut die, angle) in &mut q {
        spawn_burst(
            &mut commands,
            &mut rng,
            pos.0,
            1,
            [0.7, 0.7, 0.7, 1.0],
            (20.0, 60.0),
        );
        // GML speeds are px/step; the sim runs px/s (×30).
        let mut speed = vel.0.length() / 30.0;
        let mut dir_ang = if speed > 0.01 {
            vel.0.y.atan2(vel.0.x)
        } else {
            0.0
        };
        if die.bounces <= 3 {
            if speed < 6.0 {
                speed += 2.0;
            }
            speed += 0.05;
            dir_ang += die.trn.to_radians();
            die.trn += rng.random_range(0.0..1.0) - 0.5;
        } else {
            dir_ang += 59.0f32.to_radians();
        }
        vel.0 = glam::Vec2::new(dir_ang.cos(), dir_ang.sin()) * speed * 30.0;
        pos.0 += vel.0 * dt;
        let r = 8.0;
        let mut bounced = false;
        if pos.0.x < -ARENA_W * 0.5 + r || pos.0.x > ARENA_W * 0.5 - r {
            vel.0.x = -vel.0.x;
            bounced = true;
        }
        if pos.0.y < -ARENA_H * 0.5 + r || pos.0.y > ARENA_H * 0.5 - r {
            vel.0.y = -vel.0.y;
            bounced = true;
        }
        pos.0.x = pos.0.x.clamp(-ARENA_W * 0.5 + r, ARENA_W * 0.5 - r);
        pos.0.y = pos.0.y.clamp(-ARENA_H * 0.5 + r, ARENA_H * 0.5 - r);
        if bounced {
            die.bounces += 1;
            if die.bounces > 3 {
                // GML `Collision_Wall`: `alarm[2] = 15; speed = 0`.
                vel.0 = glam::Vec2::ZERO;
            } else {
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    PortalClear {
                        timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                        scale: 1.0,
                    },
                    Pos(pos.0),
                ));
            }
        }
        if let Some(mut a) = angle {
            a.0 = vel.0.y.atan2(vel.0.x).to_degrees() - 90.0;
        }
        let _ = e;
    }
}

/// Verbatim `objects/ScrapBossMissile` law (`Other_10` + `Alarm_0`):
/// homing drift 0.1 px/tick toward the target at forced speed 2 px/tick,
/// and on loops > 0 a trail bullet (`EnemyBullet1` stats: damage 3 at own
/// speed + 2 px/tick) every `max(1, 12 - loops)` ticks.
/// (`brain.ammo` doubles as the spawn-kick flag; wall bounce + hurt flash
/// from `Collision_Wall` are out: port enemies clamp to the arena.)
pub fn tick_scrap_missiles(
    time: Res<SimTime>,
    mut commands: Commands,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut q: Query<(Entity, &Enemy, &mut EnemyBrain, &mut Velocity, &mut Pos), With<Enemy>>,
) {
    let Ok(player_pos) = player_q.single() else {
        return;
    };
    let dt = time.delta_secs;
    let period = (12u32.saturating_sub(run.loop_count).max(1) as f32) / 30.0;
    let mut rng = rand::rng();
    for (entity, enemy, mut brain, mut vel, mut pos) in &mut q {
        if enemy.kind != EnemyKind::ScrapBossMissile {
            continue;
        }
        if brain.ammo == 0 {
            // Spawn kick (`motion_add(random_angle, 2)`).
            let a = rng.random_range(0.0..std::f32::consts::TAU);
            vel.0 = glam::Vec2::from_angle(a) * 60.0;
            brain.ammo = 1;
            brain.strafe_timer = GTimer::from_seconds(period, TimerMode::Once);
        }
        let to_player = player_pos.0 - pos.0;
        gml_motion_add_clamp(&mut vel.0, to_player.normalize_or_zero(), 0.1, 2.0, dt);
        if vel.0.length_squared() > 0.001 {
            vel.0 = vel.0.normalize() * 60.0;
        } else {
            vel.0 = to_player.normalize_or_zero() * 60.0;
        }
        if run.loop_count > 0 {
            brain.strafe_timer.tick(dt);
            if brain.strafe_timer.just_finished() {
                brain.strafe_timer = GTimer::from_seconds(period, TimerMode::Once);
                let heading = vel.0.normalize_or_zero();
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Team::Enemy,
                    Projectile {
                        damage: 3,
                        life: GTimer::from_seconds(2.5, TimerMode::Once),
                        radius: 4.0,
                        knockback: 120.0,
                        explosive: false,
                        source: Some(DamageSource::enemy(entity, enemy.kind)),
                    },
                    Velocity(heading * 120.0),
                    Pos(pos.0 + heading * 10.0),
                ));
            }
        }
        pos.0 += vel.0 * dt;
        clamp_to_arena(&mut pos.0, 8.0);
    }
}

/// Verbatim `objects/Throne2Ball` law (`Step_0`): friction 0.25 bleeds
/// speed; once stalled, `timeout` accrues and past 15 ticks the ball
/// sprays an `EnemyBullet2` (Horror stats: damage 2 at 10 px/tick) along
/// the latched `angle` every tick, dying past `40 + loops * 10` ticks.
/// While stalled but young it aims at the nearest player (±30 degrees).
/// (Position integration rides `move_projectiles`; the aim-converge
/// particles are visual-only.)
pub fn tick_throne_balls(
    time: Res<SimTime>,
    mut commands: Commands,
    run: Res<Run>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut q: Query<(Entity, &mut Velocity, &Pos, &mut ThroneBall), With<Projectile>>,
) {
    let Ok(player_pos) = player_q.single() else {
        return;
    };
    let dt = time.delta_secs;
    let cap = 40.0 + run.loop_count as f32 * 10.0;
    let mut rng = rand::rng();
    for (entity, mut vel, pos, mut ball) in &mut q {
        crate::comps_a::apply_gml_friction(&mut vel.0, 0.25, dt);
        if vel.0.length() >= 1.0 {
            continue;
        }
        vel.0 = glam::Vec2::ZERO;
        ball.timeout += dt * 30.0;
        if ball.timeout > cap {
            commands.entity(entity).despawn();
            continue;
        }
        if ball.timeout > 15.0 {
            ball.sounded = false;
            let sdir = glam::Vec2::from_angle(ball.angle);
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                Team::Enemy,
                Projectile {
                    damage: 2,
                    life: GTimer::from_seconds(2.0, TimerMode::Once),
                    radius: 4.5,
                    knockback: 120.0,
                    explosive: false,
                    source: Some(DamageSource::enemy(entity, EnemyKind::ThroneII)),
                },
                Velocity(sdir * 300.0),
                Pos(pos.0 + sdir * 12.0),
            ));
        } else {
            let aim = (player_pos.0 - pos.0).y.atan2((player_pos.0 - pos.0).x);
            ball.angle = aim + rng.random_range(-30.0..=30.0_f32).to_radians();
        }
    }
}

/// Verbatim `objects/MomProjectile/Step_0`: every tick the gas mortar
/// leaves a `ToxicGas` cloud at its tail.
pub fn tick_mom_shots(
    mut commands: Commands,
    q: Query<(&Team, &Pos), (With<Projectile>, With<MomShot>)>,
) {
    let mut rng = rand::rng();
    for (team, pos) in &q {
        let off = glam::Vec2::new(
            rng.random_range(-2.0..=2.0),
            rng.random_range(0.0..=0.0),
        );
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            *team,
            HazardCloud {
                kind: crate::data::HazardKind::Toxic,
                radius: 16.0,
                damage: 1,
                timer: GTimer::from_seconds(6.0, TimerMode::Once),
                tick: GTimer::from_seconds(0.5, TimerMode::Repeating),
            },
            Pos(pos.0 + off),
        ));
    }
}

/// Verbatim `objects/SuperFrog/Alarm_2`: every 3 ticks a stray
/// `EnemyBullet2` (damage 2 at 2 px/tick) leaves at a random angle.
pub fn tick_super_frogs(
    mut commands: Commands,
    mut ticks: Local<HashMap<Entity, u8>>,
    q: Query<(Entity, &Enemy, &Pos), With<Enemy>>,
) {
    let mut rng = rand::rng();
    for (entity, enemy, pos) in &q {
        if enemy.kind != EnemyKind::SuperFrog {
            continue;
        }
        let t = ticks.entry(entity).or_insert(0);
        *t = t.wrapping_add(1);
        if *t % 3 != 0 {
            continue;
        }
        let a = rng.random_range(0.0..std::f32::consts::TAU);
        let d = glam::Vec2::from_angle(a);
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            Team::Enemy,
            Projectile {
                damage: 2,
                life: GTimer::from_seconds(2.5, TimerMode::Once),
                radius: 4.0,
                knockback: 120.0,
                explosive: false,
                source: Some(DamageSource::enemy(entity, enemy.kind)),
            },
            Velocity(d * 60.0),
            Pos(pos.0 + d * 10.0),
        ));
    }
}

/// Verbatim `objects/EliteInspector` law (`Alarm_1`/`Alarm_2`/`Other_10`):
/// freeze-gated control field that drags projectiles and pulls the
/// player, close-range baton dash-slash (`EnemySlash` damage 8), and
/// `PopoNade` lobs at the last-seen position (5-grenade budget).
///
/// Register map: `brain.attack` = `alarm[1]`, `brain.slash_delay` =
/// `alarm[2]` countdown (ticks), `brain.ammo` = `grenades`,
/// `brain.burst_left` = `freeze`, `brain.walk` = `walk`,
/// `brain.gunangle` = `gunangle` (radians), `brain.strafe_dir` = `control`
/// (0/1), `boss.target`-equivalent heading in `EnemyBrain` is unused so
/// the move heading rides `Velocity`, last-seen rides a local map.
/// (Baton art, `wepangle` flips, and enter/taunt sounds are out.)
#[allow(clippy::too_many_arguments)]
pub fn tick_elite_inspectors(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    mut player_vel: Query<&mut Velocity, (With<Player>, Without<Enemy>)>,
    mut enemies: Query<(Entity, &Enemy, &mut EnemyBrain, &mut Velocity, &mut Pos, &Health), With<Enemy>>,
    mut shots: Query<(&mut Pos, &Team, Option<&ProjectileTyp>, Option<&PopoNadeM>), (With<Projectile>, Without<Enemy>, Without<Player>)>,
    mut last_seen: Local<HashMap<Entity, glam::Vec2>>,
    mut headings: Local<HashMap<Entity, glam::Vec2>>,
    mut inited: Local<std::collections::HashSet<Entity>>,
) {
    let Ok(player_pos) = player_q.single() else {
        return;
    };
    let player_speed_sq = player_vel
        .single()
        .map(|v| v.0.length_squared())
        .unwrap_or(0.0);
    let dt = time.delta_secs;
    let mut rng = rand::rng();
    inited.retain(|e| enemies.contains(*e));
    last_seen.retain(|e, _| enemies.contains(*e));
    headings.retain(|e, _| enemies.contains(*e));
    let nades_live = shots.iter().filter(|(_, _, _, m)| m.is_some()).count();

    for (entity, enemy, mut brain, mut vel, mut pos, health) in &mut enemies {
        if enemy.kind != EnemyKind::EliteInspector {
            continue;
        }
        let epos = pos.0;
        if !inited.contains(&entity) {
            inited.insert(entity);
            // GML `Create_0`: `walk = 30`, `grenades = 5`,
            // `alarm[1] = 30 + random(15)`.
            brain.walk = 30.0;
            brain.ammo = 5;
            brain.burst_left = 0;
            brain.strafe_dir = 0.0;
            brain.slash_delay = 0.0;
            brain.gunangle = rng.random_range(0.0..std::f32::consts::TAU);
            brain.attack = GTimer::from_seconds(
                rng.random_range(30.0..=45.0) / 30.0,
                TimerMode::Once,
            );
            last_seen.insert(entity, epos);
        }

        let to_player = player_pos.0 - epos;
        let dist = to_player.length();
        let aim = to_player.y.atan2(to_player.x);
        let los = has_line_of_sight(epos, player_pos.0, &mask);
        let freeze = brain.burst_left;

        // GML `Other_10`: freeze accrues while the target moves (or the
        // inspector is damaged); +3 while the target can shoot (port:
        // any living player counts as armed).
        if player_speed_sq > 0.001 || health.hp < health.max {
            brain.burst_left += 1;
        }
        brain.burst_left += 3;

        // Walk locomotion along the latched heading, capped at
        // 3.5 px/tick.
        let head = headings.get(&entity).copied().unwrap_or(glam::Vec2::X);
        if brain.walk > 0.0 {
            gml_motion_add_clamp(&mut vel.0, head, 0.8, 3.5, dt);
            brain.walk -= dt * 30.0;
            if brain.walk < 0.0 {
                brain.walk = 0.0;
            }
        }
        if vel.0.length() > 105.0 {
            vel.0 = vel.0.normalize() * 105.0;
        }

        // Control field: repel foreign projectiles, pull the player.
        if brain.strafe_dir == 1.0 {
            for (mut spos, team, typ, _) in &mut shots {
                if *team == Team::Enemy {
                    continue;
                }
                if typ.is_some_and(|t| t.0 == 0) {
                    continue;
                }
                let push = (spos.0 - epos).normalize_or_zero() * 60.0 * dt;
                spos.0 += push;
            }
            if dist < 160.0 && dist > 0.001 {
                if let Ok(mut pv) = player_vel.single_mut() {
                    pv.0 += to_player.normalize_or_zero() * 60.0 * dt;
                }
            }
        }

        // Baton slash delay (`alarm[2]`).
        if brain.slash_delay > 0.0 {
            brain.slash_delay -= dt * 30.0;
            if brain.slash_delay <= 0.0 {
                brain.slash_delay = 0.0;
                let sdir = glam::Vec2::from_angle(brain.gunangle);
                vel.0 += sdir * 180.0;
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Team::Enemy,
                    Projectile {
                        damage: 8,
                        life: GTimer::from_seconds(0.4, TimerMode::Once),
                        radius: 10.0,
                        knockback: 150.0,
                        explosive: false,
                        source: Some(DamageSource::enemy(entity, enemy.kind)),
                    },
                    Velocity(sdir * 60.0),
                    Pos(epos + sdir * 12.0),
                ));
                brain.attack =
                    GTimer::from_seconds(rng.random_range(15.0..=20.0) / 30.0, TimerMode::Once);
            }
        }

        // GML `Alarm_1` (brain).
        brain.attack.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds(rng.random_range(20.0..=30.0) / 30.0, TimerMode::Once);
            if brain.strafe_dir == 1.0 && rng.random::<f32>() < 0.5 {
                brain.strafe_dir = 0.0;
            }
            if rng.random::<f32>() < 0.5 && freeze > 40 {
                brain.strafe_dir = 1.0;
                let d = brain.attack.duration() + 10.0 / 30.0;
                brain.attack = GTimer::from_seconds(d, TimerMode::Once);
            }
            if los {
                brain.gunangle = aim;
                last_seen.insert(entity, player_pos.0);
                if rng.random::<f32>() < 2.0 / 3.0 && freeze > 40 && dist < 64.0 {
                    // Telegraph the baton dash.
                    brain.slash_delay = 5.0;
                    let d = brain.attack.duration() + 5.0 / 30.0;
                    brain.attack = GTimer::from_seconds(d, TimerMode::Once);
                    spawn_hit_warning(&mut commands, epos);
                    brain.walk = 0.0;
                } else {
                    let head = glam::Vec2::from_angle(
                        aim + rng.random_range(-20.0..=20.0_f32).to_radians(),
                    );
                    headings.insert(entity, head);
                    vel.0 = head * 12.0;
                    brain.walk = rng.random_range(30.0..=40.0);
                    let d = brain.attack.duration() / 2.0;
                    brain.attack = GTimer::from_seconds(d, TimerMode::Once);
                }
            } else if rng.random::<f32>() < 2.0 / 3.0 {
                let head = glam::Vec2::from_angle(
                    rng.random_range(0.0..std::f32::consts::TAU),
                );
                headings.insert(entity, head);
                brain.gunangle = head.y.atan2(head.x);
                brain.walk = rng.random_range(20.0..=30.0);
                vel.0 = head * 12.0;
            } else {
                // `PopoNade` at the last-seen position (budget-gated).
                let seen = last_seen.get(&entity).copied().unwrap_or(player_pos.0);
                let seen_d = seen.distance(player_pos.0);
                let self_seen_d = epos.distance(seen);
                let gated = rng.random::<f32>() < 1.0 / (3.0 + nades_live as f32 * 3.0)
                    && brain.ammo > 0
                    && freeze > 40
                    && dist < 160.0
                    && seen_d < 160.0
                    && self_seen_d > 64.0;
                if gated || rng.random::<f32>() < 1.0 / 8.0 {
                    if brain.ammo > 0 {
                        brain.ammo -= 1;
                    }
                    let nang = (seen - epos).y.atan2((seen - epos).x)
                        + rng.random_range(-10.0..=10.0_f32).to_radians();
                    let ndir = glam::Vec2::from_angle(nang);
                    brain.gunangle = nang;
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Projectile {
                            damage: 0,
                            life: GTimer::from_seconds(3.0, TimerMode::Once),
                            radius: 6.0,
                            knockback: 0.0,
                            explosive: true,
                            source: Some(DamageSource::enemy(entity, enemy.kind)),
                        },
                        ProjectileTyp(1),
                        PopoNadeM,
                        crate::comps_b::CustomExplosion {
                            radius: 70.0,
                            count: 1,
                            spread: 0.0,
                        },
                        Velocity(ndir * 300.0),
                        Pos(epos + ndir * 12.0),
                    ));
                }
            }
        }

        pos.0 += vel.0 * dt;
        clamp_to_arena(&mut pos.0, 10.0);
    }
}

/// Verbatim `objects/EliteShielder` law (`Alarm_1`/`Alarm_2`/`Other_10`):
/// freeze-gated 6-round `PopoPlasma` bursts (damage 8 at 1.5 px/tick),
/// and the `EliteShield` carry that teleports the shielder to a floor
/// 120..300 px away. Register map mirrors the Inspector
/// (`brain.attack` = `alarm[1]`, `brain.slash_delay` = `alarm[2]`
/// countdown, `brain.ammo` = burst rounds, `brain.burst_left` = `freeze`).
/// (The shield's projectile-block field has no port equivalent and is
/// out; the teleport + disappear poof are kept.)
#[allow(clippy::too_many_arguments)]
pub fn tick_elite_shielders(
    time: Res<SimTime>,
    mut commands: Commands,
    mask: Res<FloorMask>,
    player_q: Query<(&Pos, &Velocity), (With<Player>, Without<Enemy>)>,
    mut enemies: Query<(Entity, &Enemy, &mut EnemyBrain, &mut Velocity, &mut Pos, &Health), With<Enemy>>,
    mut headings: Local<HashMap<Entity, glam::Vec2>>,
    mut inited: Local<std::collections::HashSet<Entity>>,
) {
    let Ok((player_pos, player_v)) = player_q.single() else {
        return;
    };
    let dt = time.delta_secs;
    let mut rng = rand::rng();
    inited.retain(|e| enemies.contains(*e));
    headings.retain(|e, _| enemies.contains(*e));

    for (entity, enemy, mut brain, mut vel, mut pos, health) in &mut enemies {
        if enemy.kind != EnemyKind::EliteShielder {
            continue;
        }
        let epos = pos.0;
        if !inited.contains(&entity) {
            inited.insert(entity);
            // GML `Create_0`: `walk = 30`, `freeze = 20`,
            // `alarm[1] = 30 + random(15)`.
            brain.walk = 30.0;
            brain.ammo = 0;
            brain.burst_left = 20;
            brain.slash_delay = 0.0;
            brain.gunangle = rng.random_range(0.0..std::f32::consts::TAU);
            brain.attack = GTimer::from_seconds(
                rng.random_range(30.0..=45.0) / 30.0,
                TimerMode::Once,
            );
            headings.insert(entity, glam::Vec2::X);
        }

        let to_player = player_pos.0 - epos;
        let dist = to_player.length();
        let aim = to_player.y.atan2(to_player.x);
        let los = has_line_of_sight(epos, player_pos.0, &mask);
        let freeze = brain.burst_left;

        // GML `Other_10` freeze (target `can_shoot` reads true on players).
        if player_v.0.length_squared() > 0.001 && health.hp < health.max {
            brain.burst_left += 1;
        }
        brain.burst_left += 3;

        let head = headings.get(&entity).copied().unwrap_or(glam::Vec2::X);
        if brain.walk > 0.0 {
            gml_motion_add_clamp(&mut vel.0, head, 0.8, 3.5, dt);
            brain.walk -= dt * 30.0;
            if brain.walk < 0.0 {
                brain.walk = 0.0;
            }
        }
        if vel.0.length() > 105.0 {
            vel.0 = vel.0.normalize() * 105.0;
        }

        // Plasma burst tick (`alarm[2]`).
        if brain.slash_delay > 0.0 {
            brain.slash_delay -= dt * 30.0;
            if brain.slash_delay <= 0.0 {
                if brain.ammo > 0 {
                    let jitter =
                        rng.random_range(-10.0..=10.0_f32).to_radians();
                    let sdir = glam::Vec2::from_angle(brain.gunangle + jitter);
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Projectile {
                            damage: 8,
                            life: GTimer::from_seconds(3.0, TimerMode::Once),
                            radius: 5.0,
                            knockback: 60.0,
                            explosive: false,
                            source: Some(DamageSource::enemy(entity, enemy.kind)),
                        },
                        ProjectileTyp(2),
                        Velocity(sdir * 45.0),
                        Pos(epos + sdir * 12.0),
                    ));
                    // Recoil 0.5 px/tick opposite the muzzle.
                    vel.0 += glam::Vec2::from_angle(brain.gunangle + std::f32::consts::PI) * 15.0;
                    brain.slash_delay = 4.0;
                    brain.ammo -= 1;
                } else {
                    brain.slash_delay = 0.0;
                }
            }
        }

        // GML `Alarm_1` (brain).
        brain.attack.tick(dt);
        if brain.attack.just_finished() {
            brain.attack =
                GTimer::from_seconds(rng.random_range(15.0..=20.0) / 30.0, TimerMode::Once);
            if los {
                brain.gunangle = aim;
                if rng.random::<f32>() < 3.0 / 4.0 && freeze > 40 && dist < 150.0 {
                    brain.ammo = 6;
                    brain.slash_delay = 5.0;
                    brain.attack = GTimer::from_seconds(20.0 / 30.0, TimerMode::Once);
                } else if rng.random::<f32>() < 1.0 / 3.0 {
                    // `EliteShield` carry: teleport to a floor 120..300
                    // away, poof at the destination.
                    let mut dest = epos;
                    for _ in 0..100 {
                        let a = rng.random_range(0.0..std::f32::consts::TAU);
                        let d = rng.random_range(120.0..=300.0);
                        let cand = epos + glam::Vec2::from_angle(a) * d;
                        if mask.is_walkable(cand) {
                            dest = cand;
                            break;
                        }
                    }
                    pos.0 = dest;
                    // `EliteShield` anchor: pins the creator 60 ticks,
                    // blocks incoming fire, then poofs (`Alarm_0`).
                    commands.spawn((
                        GameCleanup,
                        LevelCleanup,
                        Team::Enemy,
                        Pos(dest),
                        EliteBlocker {
                            owner: entity,
                            timer: GTimer::from_seconds(60.0 / 30.0, TimerMode::Once),
                        },
                    ));
                    brain.attack = GTimer::from_seconds(85.0 / 30.0, TimerMode::Once);
                    vel.0 = glam::Vec2::ZERO;
                    brain.walk = 0.0;
                } else {
                    // GML compares against the constant `target.y > 64`
                    // (a bool): reproduce the literal law.
                    let probe = glam::Vec2::new(player_pos.0.x, 1.0);
                    let head = if epos.distance(probe) > 64.0 {
                        glam::Vec2::from_angle(
                            aim + rng.random_range(-25.0..=25.0_f32).to_radians(),
                        )
                    } else {
                        glam::Vec2::from_angle(
                            aim + std::f32::consts::PI
                                + rng.random_range(-45.0..=45.0_f32).to_radians(),
                        )
                    };
                    headings.insert(entity, head);
                    vel.0 = head * 12.0;
                    brain.walk = rng.random_range(10.0..=20.0);
                    if freeze < 40 {
                        let d = brain.attack.duration() + rng.random_range(0.0..=30.0) / 30.0;
                        brain.attack = GTimer::from_seconds(d, TimerMode::Once);
                    }
                }
            } else if rng.random::<f32>() < 1.0 / 3.0 {
                let head = glam::Vec2::from_angle(
                    rng.random_range(0.0..std::f32::consts::TAU),
                );
                headings.insert(entity, head);
                brain.gunangle = head.y.atan2(head.x);
                brain.walk = rng.random_range(20.0..=30.0);
                vel.0 = head * 12.0;
            } else if freeze > 40 && rng.random::<f32>() < 0.25 {
                let mut dest = epos;
                for _ in 0..100 {
                    let a = rng.random_range(0.0..std::f32::consts::TAU);
                    let d = rng.random_range(120.0..=300.0);
                    let cand = epos + glam::Vec2::from_angle(a) * d;
                    if mask.is_walkable(cand) {
                        dest = cand;
                        break;
                    }
                }
                pos.0 = dest;
                // `EliteShield` anchor: pins the creator 60 ticks,
                // blocks incoming fire, then poofs (`Alarm_0`).
                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    Team::Enemy,
                    Pos(dest),
                    EliteBlocker {
                        owner: entity,
                        timer: GTimer::from_seconds(60.0 / 30.0, TimerMode::Once),
                    },
                ));
                brain.attack = GTimer::from_seconds(75.0 / 30.0, TimerMode::Once);
                vel.0 = glam::Vec2::ZERO;
                brain.walk = 0.0;
            }
        }

        pos.0 += vel.0 * dt;
        clamp_to_arena(&mut pos.0, 11.0);
    }
}

/// Verbatim `objects/EliteShield` law (`Step_0` + `Alarm_0` +
/// `Collision_projectile`): while armed the anchor pins its creator to
/// itself, deflects `typ` 1 shots back to the IDPD team, destroys `typ`
/// 2 shots, then poofs away.
pub fn tick_elite_blockers(
    time: Res<SimTime>,
    mut commands: Commands,
    mut blockers: Query<(Entity, &Pos, &mut EliteBlocker), Without<Enemy>>,
    mut owners: Query<&mut Pos, With<Enemy>>,
    mut shots: Query<
        (Entity, &Pos, &mut Team, &mut Velocity, Option<&ProjectileTyp>),
        (With<Projectile>, Without<Enemy>),
    >,
) {
    let dt = time.delta_secs;
    for (b, bpos, mut blocker) in &mut blockers {
        blocker.timer.tick(dt);
        if blocker.timer.just_finished() {
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                Pos(bpos.0),
                StaticFx {
                    path: "images/sprEliteShielderShieldDisappear.png",
                },
                PickupLifetime {
                    timer: GTimer::from_seconds(1.0, TimerMode::Once),
                },
            ));
            commands.entity(b).despawn();
            continue;
        }
        if let Ok(mut opos) = owners.get_mut(blocker.owner) {
            opos.0 = bpos.0;
        }
        for (s, spos, mut team, mut vel, typ) in &mut shots {
            if *team != Team::Player {
                continue;
            }
            if spos.0.distance(bpos.0) > 20.0 {
                continue;
            }
            match typ.map(|t| t.0).unwrap_or(0) {
                1 => {
                    // Deflect: re-team + fling away from the shield.
                    *team = Team::Enemy;
                    let away = (spos.0 - bpos.0).normalize_or_zero();
                    let speed = vel.0.length().max(60.0);
                    vel.0 = away * speed;
                }
                2 => {
                    commands.entity(s).despawn();
                }
                _ => {}
            }
        }
    }
}

/// Verbatim `objects/ProtoStatue` law (`Step_0`): rad snapshots the
/// player's rads on placement; past 24 rads the statue charges (halves
/// its own HP, +2 IDPD portals); past 30% damage it phases (+2 IDPD
/// portals, once). Portals roll the GML `IDPDSpawn` table.
pub fn tick_proto_statues(
    mut commands: Commands,
    mut run: ResMut<Run>,
    player_q: Query<(&Player, &Pos), (With<Player>, Without<Enemy>)>,
    mut q: Query<(Entity, &Enemy, &mut Health, &Pos, &mut ProtoGuardian), With<Enemy>>,
) {
    let Ok((player, player_pos)) = player_q.single() else {
        return;
    };
    for (_, enemy, mut health, pos, mut statue) in &mut q {
        if enemy.kind != EnemyKind::ProtoStatue {
            continue;
        }
        if !statue.init {
            statue.init = true;
            statue.rad = player.rads;
        }
        let mut waves = 0;
        if statue.rad > 24 && !statue.charged {
            statue.charged = true;
            health.hp = (health.hp / 2).max(1);
            waves += 2;
        }
        if !statue.phased
            && (health.hp as f32) < (health.max as f32) * 0.7
            && health.hp > 0
        {
            statue.phased = true;
            waves += 2;
        }
        for _ in 0..waves {
            run.popolevel += 1;
            for kind in crate::idpd::roll_idpd_table(
                run.loop_count,
                run.area,
                run.popolevel,
                false,
            ) {
                commands.spawn(crate::combat::PendingEnemySpawn {
                    kind,
                    pos: pos.0,
                    difficulty: 1.0,
                    loops: run.loop_count,
                });
            }
        }
        let _ = player_pos;
    }
}

/// Boss intro banner timing (bevy parity).
pub fn tick_boss_intro(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut BossIntro)>,
) {
    for (e, mut intro) in &mut q {
        intro.timer.tick(time.delta_secs);
        if intro.timer.just_finished() {
            commands.entity(e).despawn();
        }
    }
}

/// Melee telegraph marker (GML `sprAssassinNotice` at y-16): sim side
/// keeps the timed marker; the sprite resolves renderer-side.
fn spawn_hit_warning(commands: &mut Commands, pos: glam::Vec2) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        HitWarning {
            timer: GTimer::from_seconds(0.5, TimerMode::Once),
        },
        Pos(pos + glam::Vec2::new(0.0, -16.0)),
    ));
}

/// Expire telegraph markers (bonus port — bevy `tick_hit_warnings`
/// minus the anim-end branch, which has no headless equivalent).
pub fn tick_hit_warnings(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut HitWarning)>,
) {
    for (e, mut w) in &mut q {
        w.timer.tick(time.delta_secs);
        if w.timer.just_finished() {
            commands.entity(e).despawn();
        }
    }
}

/// PopoShield follower tracking (bonus port — bevy
/// `tick_shield_followers` minus the rotation write, which the renderer
/// derives from the owner's `gunangle`).
pub fn tick_shield_followers(
    mut commands: Commands,
    owners: Query<(Entity, &Pos, &EnemyBrain), With<Enemy>>,
    mut shields: Query<(Entity, &ShieldFollower, &mut Pos), Without<Enemy>>,
) {
    for (e, sh, mut pos) in shields.iter_mut() {
        if let Ok((_, owner_pos, brain)) = owners.get(sh.owner) {
            let base = brain.gunangle;
            let off = glam::Vec2::new(base.cos(), base.sin()) * 16.0;
            pos.0 = owner_pos.0 + off;
        } else {
            commands.entity(e).despawn();
        }
    }
}

/// Prop + floor-mask + arena push-outs shared by the movement tails.
fn collide_enemy(
    props: &Query<(Entity, &Prop, &Pos), With<Prop>>,
    mask: &FloorMask,
    pos: &mut Pos,
    radius: f32,
) {
    resolve_prop_collision(
        &mut pos.0,
        radius,
        props.iter().map(|(_, p, pp)| (pp.0, p.size)),
    );
    mask.resolve_circle(&mut pos.0, radius);
    clamp_to_arena(&mut pos.0, radius);
}

/// Flock separation against the pre-move snapshot (bevy parity).
fn separate(positions: &[glam::Vec2], epos: glam::Vec2, pos: &mut Pos, radius: f32) {
    for other in positions {
        let d = epos.distance(*other);
        if d < radius + 14.0 && d > 0.001 {
            let push = (epos - *other).normalize() * (radius + 14.0 - d) * 0.5;
            pos.0.x += push.x;
            pos.0.y += push.y;
        }
    }
}

/// Corpse slide + expiry (bevy `enemies.rs:2552` parity: `Corpse` life
/// ticks, corpses drift with GML 0.4 friction, expiry despawns).
/// Port adaptation: `Transform.translation` is [`Pos`] here; corpses
/// without [`Velocity`] (player-kill drops) only tick life.
pub fn tick_corpses(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut Corpse, Option<&mut Velocity>, Option<&mut Pos>)>,
) {
    let dt = time.delta_secs;
    for (e, mut c, vel, pos) in &mut q {
        c.life.tick(dt);
        if c.life.just_finished() {
            commands.entity(e).despawn();
            continue;
        }
        if let (Some(mut v), Some(mut p)) = (vel, pos) {
            apply_gml_friction(&mut v.0, 0.4, dt);
            p.0 += v.0 * dt;
        }
    }
}


#[cfg(test)]
mod spawn_hp_tests {
    use super::*;
    use crate::enemy_data::enemy_def;

    #[test]
    fn loop0_matches_table() {
        for kind in [
            EnemyKind::Bandit,
            EnemyKind::FrogQueen,
            EnemyKind::Captain,
            EnemyKind::YvBoss,
            EnemyKind::ProtoStatue,
            EnemyKind::Scorpion,
        ] {
            assert_eq!(spawn_hp(kind, enemy_def(kind).hp, 0), enemy_def(kind).hp);
        }
    }

    #[test]
    fn boss_third_law() {
        assert_eq!(spawn_hp(EnemyKind::BigBandit, 100, 3), 200);
        assert_eq!(spawn_hp(EnemyKind::FrogQueen, 490, 3), 980);
        assert_eq!(spawn_hp(EnemyKind::Captain, 1100, 3), 2200);
        assert_eq!(spawn_hp(EnemyKind::Throne, 1500, 3), 3000);
        assert_eq!(spawn_hp(EnemyKind::LilHunter, 140, 3), 280);
    }

    #[test]
    fn flat_and_default_laws() {
        assert_eq!(spawn_hp(EnemyKind::ProtoStatue, 120, 5), 120);
        assert_eq!(spawn_hp(EnemyKind::YvBoss, 700, 4), 841);
        assert_eq!(spawn_hp(EnemyKind::Scorpion, 16, 20), 32);
        assert_eq!(
            spawn_hp(EnemyKind::BigDog, 300, 6),
            (300.0_f32 * (1.0 + 6.0 / 1.2)).ceil() as i32
        );
    }
}
