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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comps_a::Health;
    use repame_anim::{AnimCatalog, AtlasDesc};

    const DT: f32 = 1.0 / 30.0;

    fn sim_time(dt: f32) -> SimTime {
        let mut t = SimTime::default();
        t.delta_secs = dt;
        t
    }

    fn empty_catalog() -> AnimCatalog {
        AnimCatalog::from_json("{}", AtlasDesc {
            size: 128,
            max_pages: 1,
        padding: 0,
        })
        .expect("empty catalog parses")
    }

    fn raid_world(dt: f32, run: Run) -> World {
        let mut world = World::new();
        world.insert_resource(sim_time(dt));
        world.insert_resource(empty_catalog());
        world.insert_resource(GameAudio);
        world.insert_resource(Trauma::default());
        world.insert_resource(IdpdRaidState::default());
        world.insert_resource(run);
        world.insert_resource(LoopTransition::default());
        world.insert_resource(Toast::default());
        world.insert_resource(Queue::<AudioCue>::default());
        world.spawn((Player::default(), Pos(glam::Vec2::ZERO)));
        world
    }

    fn idpd_kinds(world: &mut World) -> Vec<EnemyKind> {
        let mut q = world.query::<&Enemy>();
        q.iter(world).map(|e| e.kind).collect()
    }

    #[test]
    fn spawn_table_follows_gml_gates() {
        // Loop 0 with LilHunter alive forces dir 1 (no elites at loop 0).
        for _ in 0..8 {
            assert_eq!(
                roll_idpd_table(0, AreaId::Desert, 0, true),
                vec![EnemyKind::IdpdGrunt, EnemyKind::IdpdGrunt]
            );
        }
        // Deep loops hatch a lone PopoFreak.
        assert_eq!(
            roll_idpd_table(3, AreaId::Palace, 9, false),
            vec![EnemyKind::PopoFreak]
        );
        assert_eq!(
            roll_idpd_table(4, AreaId::Desert, 9, false),
            vec![EnemyKind::PopoFreak]
        );
        // Elite roll is impossible pre-loop (and on campfire at loop 1).
        assert!(!idpd_elite_roll(0, AreaId::Desert));
        assert!(!idpd_elite_roll(0, AreaId::Palace));
        assert!(!idpd_elite_roll(1, AreaId::Campfire));
        // Popolevel gates the dir pool (loop 1, no LilHunter force).
        for _ in 0..16 {
            let kinds = roll_idpd_table(1, AreaId::Palace, 0, false);
            assert!(
                kinds == vec![EnemyKind::IdpdGrunt, EnemyKind::IdpdGrunt]
                    || kinds == vec![EnemyKind::IdpdElite],
                "popolevel 0 admits only dir 1 (elite swap allowed): {kinds:?}"
            );
        }
    }

    #[test]
    fn trigger_gates_match_bevy() {
        let base = Run {
            loop_count: 1,
            area: AreaId::Palace,
            total_kills: 12,
            ..Default::default()
        };
        assert!(should_trigger_idpd(&base, 0, 12, false));
        assert!(!should_trigger_idpd(&base, 0, 12, true), "pending blocks");
        assert!(!should_trigger_idpd(&base, 5, 12, false), "crowded blocks");
        assert!(!should_trigger_idpd(&base, 0, 9, false), "needs 10 kills");
        assert!(
            !should_trigger_idpd(
                &Run {
                    loop_count: 0,
                    ..base
                },
                0,
                12,
                false
            ),
            "loop 0 blocks"
        );
        for area in [
            AreaId::Vault,
            AreaId::CrownVault,
            AreaId::Oasis,
            AreaId::PizzaSewers,
            AreaId::Campfire,
            AreaId::HQ,
        ] {
            assert!(
                !should_trigger_idpd(&Run { area, ..base }, 0, 12, false),
                "suppressed: {area:?}"
            );
        }
        assert!(may_queue_new_raid(&LoopTransition::default()));
        let mut blocked = LoopTransition::default();
        blocked.begin_campfire();
        assert!(!may_queue_new_raid(&blocked));
        for kind in [
            EnemyKind::IdpdGrunt,
            EnemyKind::IdpdShield,
            EnemyKind::IdpdElite,
            EnemyKind::IdpdVan,
        ] {
            assert!(is_idpd_kind(kind));
        }
        assert!(!is_idpd_kind(EnemyKind::Bandit));
    }

    #[test]
    fn wave_picker_pressure_bands() {
        assert_eq!(choose_wave(0, 1, 0), RaidWave::Light);
        assert_eq!(choose_wave(1, 8, 3), RaidWave::Medium);
        assert_eq!(choose_wave(1, 8, 5), RaidWave::VanDrop);
        assert_eq!(choose_wave(2, 10, 4), RaidWave::VanDrop);
        assert_eq!(choose_wave(2, 10, 2), RaidWave::Heavy);
        assert_eq!(choose_wave(2, 10, 3), RaidWave::Medium);
    }

    #[test]
    fn edge_points_keep_margin_and_sort_furthest_first() {
        let player = glam::Vec2::new(400.0, 100.0);
        let pts = edge_spawn_points_away_from(player);
        let mut prev = f32::MAX;
        for p in pts {
            let on_edge = (p.x.abs() - (ARENA_W * 0.5 - 56.0)).abs() < 1e-4
                || (p.y.abs() - (ARENA_H * 0.5 - 56.0)).abs() < 1e-4;
            assert!(on_edge, "edge with 56 px margin: {p:?}");
            let d = p.distance_squared(player);
            assert!(d <= prev + 1e-3, "sorted furthest first");
            prev = d;
        }
    }

    #[test]
    fn raid_triggers_warns_then_drops_light_wave() {
        let mut world = raid_world(
            DT,
            Run {
                loop_count: 1,
                floor: 1,
                area: AreaId::Palace,
                total_kills: 10,
                gen_seed: 0,
                ..Default::default()
            },
        );
        // Arm the cooldown to finish this tick.
        world.resource_mut::<IdpdRaidState>().cooldown =
            GTimer::from_seconds(0.01, TimerMode::Once);
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(tick_idpd_raids);
        schedule.run(&mut world);

        let raid = world.resource::<IdpdRaidState>();
        assert_eq!(raid.pending_wave, Some(RaidWave::Light));
        assert_eq!(world.resource::<Toast>().text, "IDPD INCOMING");
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert!(
            cues.iter().any(|c| c.name == "sndVanWarning"
                && (c.volume - 0.7).abs() < 1e-6
                && (c.variance - 0.05).abs() < 1e-6),
            "van warning sting queued"
        );
        let mut rq = world.query::<&QueuedReactiveCue>();
        assert_eq!(rq.iter(&world).count(), 1);

        // Warning elapses -> Light wave (roll 11, pressure 11): 2 grunts + shield.
        world.resource_mut::<SimTime>().delta_secs = 1.3;
        schedule.run(&mut world);
        // Flush the deferred spawns through the shared enemy flush.
        let mut flush = bevy_ecs::schedule::Schedule::default();
        flush.add_systems(crate::enemies::flush_pending_enemy_spawns);
        flush.run(&mut world);
        let kinds = idpd_kinds(&mut world);
        assert_eq!(kinds.len(), 3, "light wave composition: {kinds:?}");
        assert_eq!(
            kinds.iter().filter(|k| **k == EnemyKind::IdpdGrunt).count(),
            2
        );
        assert_eq!(
            kinds
                .iter()
                .filter(|k| {
                    **k == EnemyKind::IdpdShield || **k == EnemyKind::EliteShielder
                })
                .count(),
            1,
            "shield site (GML elite roll may upgrade it)"
        );
        let raid = world.resource::<IdpdRaidState>();
        assert_eq!(raid.pending_wave, None);
        assert_eq!(raid.wave_index, 1);
        assert_eq!(raid.kills_checkpoint, 10);
        assert!((raid.cooldown.duration() - 16.5).abs() < 1e-4);
        let cues = world.resource_mut::<Queue<AudioCue>>().drain();
        assert!(cues.iter().any(|c| c.name == "sndPortalOpen"), "portal whoosh");
    }

    #[test]
    fn raid_stays_quiet_while_cooldown_runs() {
        let mut world = raid_world(
            DT,
            Run {
                loop_count: 1,
                area: AreaId::Palace,
                total_kills: 10,
                ..Default::default()
            },
        );
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(tick_idpd_raids);
        schedule.run(&mut world);
        assert_eq!(world.resource::<IdpdRaidState>().pending_wave, None);
    }

    #[test]
    fn wave_compositions_spawn_expected_kinds() {
        for (wave, grunts, shields, elites, vans) in [
            (RaidWave::Light, 2, 1, 0, 0),
            (RaidWave::Medium, 2, 1, 1, 0),
            (RaidWave::Heavy, 4, 1, 1, 0),
            (RaidWave::VanDrop, 0, 1, 1, 1),
        ] {
            let mut world = World::new();
            let catalog = empty_catalog();
            {
                let mut cmds = world.commands();
                // Loop 0: the GML elite roll cannot trigger, so the
                // composition is deterministic.
                spawn_raid_wave(
                    &mut cmds,
                    &catalog,
                    glam::Vec2::ZERO,
                    0,
                    AreaId::Desert,
                    wave,
                );
            }
            world.flush();
            let mut flush = bevy_ecs::schedule::Schedule::default();
            flush.add_systems(crate::enemies::flush_pending_enemy_spawns);
            // flush_pending needs its resources; provide the catalog.
            world.insert_resource(empty_catalog());
            flush.run(&mut world);
            let kinds = idpd_kinds(&mut world);
            let count = |k: EnemyKind| kinds.iter().filter(|x| **x == k).count();
            assert_eq!(count(EnemyKind::IdpdGrunt), grunts, "{wave:?}");
            assert_eq!(count(EnemyKind::IdpdShield), shields, "{wave:?}");
            assert_eq!(count(EnemyKind::IdpdElite), elites, "{wave:?}");
            assert_eq!(count(EnemyKind::IdpdVan), vans, "{wave:?}");
        }
    }

    #[test]
    fn van_deploy_spends_charges_and_sheds_shield() {
        let mut world = World::new();
        world.insert_resource(sim_time(0.1));
        world.init_resource::<Run>();
        let catalog = empty_catalog();
        world.insert_resource(catalog);
        let van = {
            let mut cmds = world.commands();
            spawn_enemy_at(
                &mut cmds,
                &empty_catalog(),
                EnemyKind::IdpdVan,
                glam::Vec2::new(50.0, 50.0),
                1.0,
                false,
                false,
                0,
            )
        };
        world.flush();
        assert!(world.get::<IdpdVanBrain>(van).is_some());
        assert!(world.get::<IdpdShieldUnit>(van).is_some());
        // Fast-forward the deploy timer to finish on the next tick.
        world.get_mut::<IdpdVanBrain>(van).expect("van brain").deploy_timer =
            GTimer::from_seconds(0.01, TimerMode::Once);
        let van_pos = world.get::<Pos>(van).expect("van pos").0;
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(tick_idpd_vans);
        schedule.run(&mut world);
        world.flush();

        let brain = world.get::<IdpdVanBrain>(van).expect("van alive");
        assert_eq!(brain.charges_left, 3);
        // Van itself never moves in the deploy tick.
        assert_eq!(world.get::<Pos>(van).expect("van pos").0, van_pos);
        assert!(
            world.get::<IdpdShieldUnit>(van).is_none(),
            "shield marker shed"
        );
        // Two grunts flank below; no shield on odd charges (3 % 2 != 0).
        let mut q = world.query::<(&Enemy, &Pos, &Health)>();
        let mut grunts = 0;
        let mut shields = 0;
        for (e, p, _) in q.iter(&world) {
            if e.kind == EnemyKind::IdpdGrunt {
                grunts += 1;
                assert!(((p.0 - van_pos).y + 22.0).abs() < 1e-3);
            }
            if e.kind == EnemyKind::IdpdShield {
                shields += 1;
            }
        }
        assert_eq!(grunts, 2);
        assert_eq!(shields, 0);

        // Next deploy (charges 3 -> 2, even) adds the overhead shield.
        world.get_mut::<IdpdVanBrain>(van).expect("van brain").deploy_timer =
            GTimer::from_seconds(0.01, TimerMode::Once);
        schedule.run(&mut world);
        world.flush();
        assert_eq!(
            world
                .get::<IdpdVanBrain>(van)
                .expect("van alive")
                .charges_left,
            2
        );
        let mut q = world.query::<&Enemy>();
        assert_eq!(
            q.iter(&world)
                .filter(|e| e.kind == EnemyKind::IdpdShield)
                .count(),
            1
        );
    }

    #[test]
    fn hq_pressure_spawns_trio_plus_even_wave_van() {
        let mut world = raid_world(
            DT,
            Run {
                area: AreaId::HQ,
                ..Default::default()
            },
        );
        world.resource_mut::<IdpdRaidState>().cooldown =
            GTimer::from_seconds(0.01, TimerMode::Once);
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(hq_pressure);
        schedule.run(&mut world);
        world.flush();
        let mut flush = bevy_ecs::schedule::Schedule::default();
        flush.add_systems(crate::enemies::flush_pending_enemy_spawns);
        flush.run(&mut world);

        let kinds = idpd_kinds(&mut world);
        assert_eq!(kinds.len(), 4, "trio + van on wave 0: {kinds:?}");
        let raid = world.resource::<IdpdRaidState>();
        assert_eq!(raid.wave_index, 1);
        assert!((raid.cooldown.duration() - 9.5).abs() < 1e-4);

        // Odd wave: no van.
        world.resource_mut::<IdpdRaidState>().cooldown =
            GTimer::from_seconds(0.01, TimerMode::Once);
        schedule.run(&mut world);
        world.flush();
        flush.run(&mut world);
        assert_eq!(idpd_kinds(&mut world).len(), 4 + 3);

        // Off-HQ: silent.
        world.resource_mut::<Run>().area = AreaId::Palace;
        world.resource_mut::<IdpdRaidState>().cooldown =
            GTimer::from_seconds(0.01, TimerMode::Once);
        schedule.run(&mut world);
        world.flush();
        flush.run(&mut world);
        assert_eq!(idpd_kinds(&mut world).len(), 4 + 3);
    }
}
