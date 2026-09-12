//! IDPD raid director. Ported from the bevy reference `game/idpd.rs`
//! (`is_idpd_kind`, `may_queue_new_raid`, `should_trigger_idpd`,
//! `choose_wave`, `edge_spawn_points_away_from`, `tick_idpd_raids`,
//! `spawn_raid_wave`, `spawn_at`/`grunt`/`shield`/`elite`/`van`,
//! `tick_idpd_vans`, `hq_pressure`) with positions as [`Pos`] (`Vec2`)
//! instead of `Transform.translation`.
//!
//! Render split: portal bursts route through
//! [`crate::effects::spawn_burst`] and trauma through
//! `repame_fx::Trauma`, matching the bevy `VfxSpawner`/`ScreenEffects`
//! call sites. The warning sting uses the `sndVanWarning` cue at
//! 0.7/0.05 (bevy `play_van_warning` parity); the portal whoosh goes
//! through [`GameAudio::play_portal`].
//!
//! Spawn path: waves call [`crate::enemies::spawn_enemy_at`] (base
//! bundle + brains + table stats), so no second spawn pipeline.
//! Timing is [`GTimer`] driven off [`SimTime`]; the
//! [`LoopTransition`] gating matches `loop_transition.rs` exactly
//! (`blocks_new_idpd_raids`, `throne_ii_alive`, `loop_ready`,
//! `campfire_active`).

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::audio::{AudioCue, GameAudio, QueuedReactiveCue, ReactiveCue};
use crate::comps_a::{ARENA_H, ARENA_W, GameCleanup, Player, Run, Toast};
use crate::comps_b::{Enemy, IdpdRaidState, IdpdShieldUnit, IdpdVanBrain, LoopTransition, RaidWave};
use crate::data::{AreaId, EnemyKind};
use crate::effects::spawn_burst;
use crate::enemies::spawn_enemy_at;
use crate::msg::Queue;
use crate::spatial::Pos;
use crate::time::{GTimer, TimerMode};

/// Areas where raids never trigger (bevy parity).
fn is_raid_suppressed_area(area: AreaId) -> bool {
    matches!(
        area,
        AreaId::Vault
            | AreaId::CrownVault
            | AreaId::Oasis
            | AreaId::PizzaSewers
            | AreaId::Campfire
            | AreaId::HQ
    )
}

/// True for the four IDPD kinds (bevy parity).
pub fn is_idpd_kind(kind: EnemyKind) -> bool {
    matches!(
        kind,
        EnemyKind::IdpdGrunt | EnemyKind::IdpdShield | EnemyKind::IdpdElite | EnemyKind::IdpdVan
    )
}

/// Raids may queue unless the loop transition blocks them (bevy parity).
pub fn may_queue_new_raid(transition: &LoopTransition) -> bool {
    !transition.blocks_new_idpd_raids()
}

/// Raid trigger gate: looped runs only, unsuppressed areas, a quiet
/// arena (<= 4 alive), 10+ kills since the last wave, and no wave
/// already pending (bevy parity).
pub fn should_trigger_idpd(
    run: &Run,
    enemies_alive: usize,
    kills_since_checkpoint: u32,
    pending: bool,
) -> bool {
    if pending || run.loop_count == 0 || run.game_over {
        return false;
    }

    if is_raid_suppressed_area(run.area) {
        return false;
    }

    if enemies_alive > 4 {
        return false;
    }

    kills_since_checkpoint >= 10
}

/// GML `IDPDSpawn/Create_0` elite law: 1-in-5 once loops are deep
/// enough (`loops > 1` on area 0 = campfire, any loop past it).
pub fn idpd_elite_roll(loop_count: u32, area: AreaId) -> bool {
    let eligible = (loop_count > 1 && area == AreaId::Campfire)
        || (loop_count > 0 && area != AreaId::Campfire);
    eligible && rand::rng().random::<f32>() < 0.2
}

/// GML `IDPDSpawn/Alarm_1` spawn table: a lone `PopoFreak` on deep
/// loops (`loops - (area == 0) >= 3`), else the popolevel-gated dir roll
/// (`rng_choose(1, 1, 2, 3)`, dir 3 needs popolevel 3+, dir 2 needs 5+;
/// loop 0 with LilHunter alive forces dir 1). Elites swap the whole dir
/// pick for their elite kind.
pub fn roll_idpd_table(
    loop_count: u32,
    area: AreaId,
    popolevel: u32,
    lil_hunter_alive: bool,
) -> Vec<EnemyKind> {
    if loop_count.saturating_sub(if area == AreaId::Campfire { 1 } else { 0 }) >= 3 {
        return vec![EnemyKind::PopoFreak];
    }
    let mut rng = rand::rng();
    let dir = if loop_count == 0 && lil_hunter_alive {
        1
    } else {
        loop {
            let d = match rng.random::<f32>() {
                x if x < 0.25 => 1,
                x if x < 0.5 => 1,
                x if x < 0.75 => 2,
                _ => 3,
            };
            if !(d == 3 && popolevel < 3) && !(d == 2 && popolevel < 5) {
                break d;
            }
        }
    };
    let elite = idpd_elite_roll(loop_count, area);
    match dir {
        2 => vec![if elite {
            EnemyKind::EliteShielder
        } else {
            EnemyKind::IdpdShield
        }],
        3 => vec![if elite {
            EnemyKind::EliteInspector
        } else {
            EnemyKind::IdpdInspector
        }],
        _ => {
            if elite {
                vec![EnemyKind::IdpdElite]
            } else {
                vec![EnemyKind::IdpdGrunt, EnemyKind::IdpdGrunt]
            }
        }
    }
}

/// Wave picker from loop pressure + floor (bevy parity).
pub fn choose_wave(loop_count: u32, floor: u32, roll: u8) -> RaidWave {
    let pressure = loop_count * 10 + floor.min(30);
    if pressure >= 28 {
        if roll % 4 == 0 {
            RaidWave::VanDrop
        } else if roll % 2 == 0 {
            RaidWave::Heavy
        } else {
            RaidWave::Medium
        }
    } else if pressure >= 18 {
        if roll % 5 == 0 {
            RaidWave::VanDrop
        } else {
            RaidWave::Medium
        }
    } else {
        RaidWave::Light
    }
}

/// Four arena-edge points (56 px margin), furthest-from-player first
/// (bevy parity, including the stable-sort tie order).
pub fn edge_spawn_points_away_from(player_pos: glam::Vec2) -> [glam::Vec2; 4] {
    let margin = 56.0;

    let left = glam::Vec2::new(
        -ARENA_W * 0.5 + margin,
        player_pos.y.clamp(-ARENA_H * 0.4, ARENA_H * 0.4),
    );
    let right = glam::Vec2::new(
        ARENA_W * 0.5 - margin,
        player_pos.y.clamp(-ARENA_H * 0.4, ARENA_H * 0.4),
    );
    let top = glam::Vec2::new(
        player_pos.x.clamp(-ARENA_W * 0.4, ARENA_W * 0.4),
        ARENA_H * 0.5 - margin,
    );
    let bottom = glam::Vec2::new(
        player_pos.x.clamp(-ARENA_W * 0.4, ARENA_W * 0.4),
        -ARENA_H * 0.5 + margin,
    );

    let mut pts = [left, right, top, bottom];
    pts.sort_by(|a, b| {
        b.distance_squared(player_pos)
            .partial_cmp(&a.distance_squared(player_pos))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pts
}

/// Raid director: cooldown tick, trigger gate, 1.25 s warning, then the
/// wave drops and the cooldown re-arms at `(18 - loop * 1.5).max(8)` s
/// (bevy `tick_idpd_raids` top to bottom).
#[allow(clippy::too_many_arguments)]
pub fn tick_idpd_raids(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    audio: Res<GameAudio>,
    mut trauma: ResMut<Trauma>,
    mut raid: ResMut<IdpdRaidState>,
    mut run: ResMut<Run>,
    transition: Res<LoopTransition>,
    player_q: Query<&Pos, With<Player>>,
    enemies_q: Query<(), With<Enemy>>,
    mut toast: ResMut<Toast>,
    mut cues: ResMut<Queue<AudioCue>>,
) {
    let dt = time.delta_secs;
    raid.cooldown.tick(dt);

    let Ok(player) = player_q.single() else {
        return;
    };
    let player_pos = player.0;
    let enemies_alive = enemies_q.iter().count();
    let kills_since_checkpoint = run.total_kills.saturating_sub(raid.kills_checkpoint);

    if transition.throne_ii_alive || transition.loop_ready {
        raid.pending_wave = None;
        return;
    }

    let may_queue = may_queue_new_raid(&transition);

    if may_queue
        && should_trigger_idpd(
            &run,
            enemies_alive,
            kills_since_checkpoint,
            raid.pending_wave.is_some(),
        )
        && raid.cooldown.just_finished()
    {
        let roll = ((run.gen_seed ^ run.total_kills as u64 ^ run.floor as u64) & 0xFF) as u8;
        let wave = choose_wave(run.loop_count, run.floor, roll);
        raid.pending_wave = Some(wave);
        raid.warning = GTimer::from_seconds(1.25, TimerMode::Once);
        toast.show("IDPD INCOMING");
        commands.spawn((GameCleanup, QueuedReactiveCue(ReactiveCue::IdpdIncoming)));
        // Bevy `play_van_warning`: sndVanWarning at 0.7 vol, 0.05 var.
        cues.push(AudioCue {
            name: "sndVanWarning",
            volume: 0.7,
            variance: 0.05,
        });
        trauma.add(0.12);
        return;
    }

    if transition.campfire_active && raid.pending_wave.is_none() {
        return;
    }

    let Some(wave) = raid.pending_wave else {
        return;
    };

    raid.warning.tick(dt);
    if !raid.warning.just_finished() {
        return;
    }

    let portals = spawn_raid_wave(
        &mut commands,
        &catalog,
        player_pos,
        run.loop_count,
        run.area,
        wave,
    );
    run.popolevel += portals as u32;

    raid.pending_wave = None;
    raid.wave_index += 1;
    raid.kills_checkpoint = run.total_kills;
    raid.cooldown = GTimer::from_seconds(
        (18.0 - (run.loop_count as f32 * 1.5)).max(8.0),
        TimerMode::Once,
    );

    audio.play_portal(&mut cues);
    let mut rng = rand::rng();
    spawn_burst(
        &mut commands,
        &mut rng,
        player_pos,
        16,
        [0.45, 0.7, 1.0, 1.0],
        (120.0, 260.0),
    );
}

/// Wave composition at `1.0 + loop * 0.18` difficulty (bevy parity,
/// including the Heavy midpoint offsets and the VanDrop bumps).
/// Returns the portal count for `Run.popolevel` (GML counts one portal
/// per spawn site; bevy waves spawn directly, so each wave counts its
/// sites). Shield sites roll the GML elite upgrade.
pub fn spawn_raid_wave(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    player_pos: glam::Vec2,
    loop_count: u32,
    area: AreaId,
    wave: RaidWave,
) -> usize {
    let points = edge_spawn_points_away_from(player_pos);
    let difficulty = 1.0 + loop_count as f32 * 0.18;

    match wave {
        RaidWave::Light => {
            spawn_grunt(commands, catalog, points[0], difficulty, loop_count);
            spawn_grunt(commands, catalog, points[1], difficulty, loop_count);
            spawn_shield(commands, catalog, points[2], difficulty, loop_count, area);
            3
        }

        RaidWave::Medium => {
            spawn_grunt(commands, catalog, points[0], difficulty, loop_count);
            spawn_grunt(commands, catalog, points[1], difficulty, loop_count);
            spawn_shield(commands, catalog, points[2], difficulty, loop_count, area);
            spawn_elite(commands, catalog, points[3], difficulty, loop_count);
            4
        }

        RaidWave::Heavy => {
            for &p in &points {
                spawn_grunt(commands, catalog, p, difficulty, loop_count);
            }
            spawn_elite(
                commands,
                catalog,
                (points[0] + points[1]) * 0.5,
                difficulty + 0.15,
                loop_count,
            );
            spawn_shield(
                commands,
                catalog,
                (points[2] + points[3]) * 0.5,
                difficulty + 0.15,
                loop_count,
                area,
            );
            6
        }

        RaidWave::VanDrop => {
            spawn_van(commands, catalog, points[0], difficulty + 0.25, loop_count);
            spawn_shield(commands, catalog, points[1], difficulty, loop_count, area);
            spawn_elite(commands, catalog, points[2], difficulty, loop_count);
            3
        }
    }
}

fn spawn_at(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: EnemyKind,
    pos: glam::Vec2,
    difficulty: f32,
    loops: u32,
) {
    spawn_enemy_at(commands, catalog, kind, pos, difficulty, false, false, loops);
}

fn spawn_grunt(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    difficulty: f32,
    loops: u32,
) {
    spawn_at(commands, catalog, EnemyKind::IdpdGrunt, pos, difficulty, loops);
}

fn spawn_shield(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    difficulty: f32,
    loops: u32,
    area: AreaId,
) {
    // GML `IDPDSpawn` dir-2 elite swap.
    let kind = if idpd_elite_roll(loops, area) {
        EnemyKind::EliteShielder
    } else {
        EnemyKind::IdpdShield
    };
    spawn_at(commands, catalog, kind, pos, difficulty, loops);
}

fn spawn_elite(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    difficulty: f32,
    loops: u32,
) {
    spawn_at(commands, catalog, EnemyKind::IdpdElite, pos, difficulty, loops);
}

fn spawn_van(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    difficulty: f32,
    loops: u32,
) {
    spawn_at(commands, catalog, EnemyKind::IdpdVan, pos, difficulty, loops);
}

/// Van deploy tick: each finished 2.2 s timer spends a charge to drop
/// two grunts (1.15) flanking below plus a shield (1.2) above on every
/// other charge, then sheds the shield marker (bevy parity; the bevy
/// tail `enemy_def(IdpdVan)` lookup is a no-op and stays out).
pub fn tick_idpd_vans(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    run: Res<Run>,
    mut vans: Query<(Entity, &Pos, &mut IdpdVanBrain), With<Enemy>>,
) {
    let dt = time.delta_secs;
    for (entity, pos, mut van) in vans.iter_mut() {
        if van.charges_left == 0 {
            continue;
        }

        van.deploy_timer.tick(dt);
        if !van.deploy_timer.just_finished() {
            continue;
        }

        van.charges_left -= 1;
        let center = pos.0;

        spawn_grunt(
            &mut commands,
            &catalog,
            center + glam::Vec2::new(-18.0, -22.0),
            1.15,
            run.loop_count,
        );
        spawn_grunt(
            &mut commands,
            &catalog,
            center + glam::Vec2::new(18.0, -22.0),
            1.15,
            run.loop_count,
        );

        if van.charges_left % 2 == 0 {
            spawn_shield(
                &mut commands,
                &catalog,
                center + glam::Vec2::new(0.0, 26.0),
                1.2,
                run.loop_count,
                run.area,
            );
        }

        commands.entity(entity).remove::<IdpdShieldUnit>();
    }
}

/// HQ pressure spawner: off-HQ areas return early; otherwise, while
/// fewer than 8 enemies live and the cooldown just finished, drop a
/// grunt/shield/elite trio (1.3/1.35/1.4) plus a van (1.45) on even
/// waves, then re-arm at 9.5 s (bevy parity).
pub fn hq_pressure(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    run: Res<Run>,
    player_q: Query<&Pos, With<Player>>,
    enemies_q: Query<(), With<Enemy>>,
    mut raid: ResMut<IdpdRaidState>,
) {
    if run.area != AreaId::HQ {
        return;
    }

    let Ok(player) = player_q.single() else {
        return;
    };

    raid.cooldown.tick(time.delta_secs);

    let enemies_alive = enemies_q.iter().count();
    if enemies_alive >= 8 {
        return;
    }

    if !raid.cooldown.just_finished() {
        return;
    }

    let player_pos = player.0;
    let points = edge_spawn_points_away_from(player_pos);

    spawn_grunt(&mut commands, &catalog, points[0], 1.3, run.loop_count);
    spawn_shield(
        &mut commands,
        &catalog,
        points[1],
        1.35,
        run.loop_count,
        run.area,
    );
    spawn_elite(&mut commands, &catalog, points[2], 1.4, run.loop_count);

    if raid.wave_index % 2 == 0 {
        spawn_van(&mut commands, &catalog, points[3], 1.45, run.loop_count);
    }

    raid.wave_index += 1;
    raid.cooldown = GTimer::from_seconds(9.5, TimerMode::Once);
}

