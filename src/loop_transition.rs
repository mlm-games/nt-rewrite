//! Throne-room loop transitions. Small pure helpers plus the
//! campfire phase machine (bevy `loop_transition.rs` gameplay half:
//! timers, IDPD-gate/raid interplay, trauma, toasts, Throne II spawn).
//! Ember/burst `Vfx` visuals stay renderer-side (skipped); the sim
//! keeps every phase edge, toast, trauma hit, and spawn.
//!
//! `try_apply_loop_portal_transition` lives in `progression.rs`
//! (private floor-advance helper, not duplicated here).

use bevy_ecs::prelude::*;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::audio::{QueuedReactiveCue, ReactiveCue};
use crate::combat::PendingEnemySpawn;
use crate::comps_a::{GameCleanup, LevelCleanup, Run, Toast};
use crate::comps_b::{
    CampfirePhase, CampfireProp, CampfireState, Enemy, IdpdRaidState, LoopTransition, YvCouch,
};
use crate::data::EnemyKind;
use crate::idpd::is_idpd_kind;
use crate::spatial::Pos;

/// Begin the campfire rest (Throne dead while loop-eligible).
pub fn begin_throne_campfire(
    commands: &mut Commands,
    transition: &mut LoopTransition,
    toast: &mut Toast,
    trauma: &mut Trauma,
    player_pos: glam::Vec2,
) {
    if transition.campfire_active || transition.throne_ii_alive {
        return;
    }

    transition.begin_campfire();

    let pos = player_pos + glam::Vec2::new(0.0, -42.0);

    commands.spawn((
        GameCleanup,
        LevelCleanup,
        CampfireProp,
        CampfireState::new(),
        Pos(pos),
    ));
    // Yung Venuz watches the fire from his couch (GML `YungVenuzCouch`,
    // crib: idle `sprYVBossGamingIdle`). Offset is a port choice — GML
    // places it in the crib room layout, which has no fixed anchor to
    // the fire here — kept clear of the flames on +x.
    commands.spawn((GameCleanup, LevelCleanup, YvCouch::idle(), Pos(pos + glam::Vec2::new(72.0, 0.0))));

    toast.show("REST");
    trauma.add(0.10);
}

/// Throne II dead: the loop opens.
pub fn mark_throne_ii_defeated(toast: &mut Toast, trauma: &mut Trauma) {
    toast.show("THE LOOP OPENS");
    trauma.add(0.35);
}

/// The campfire waits while IDPD are alive or a raid warning is
/// pending (bevy `campfire_needs_idpd_clear` parity).
pub fn campfire_needs_idpd_clear(idpd_alive: usize, raid_pending: bool) -> bool {
    idpd_alive > 0 || raid_pending
}

fn start_campfire_rising(campfire: &mut CampfireState, toast: &mut Toast, trauma: &mut Trauma) {
    campfire.set_phase(CampfirePhase::Rising, 1.15);
    toast.show("SOMETHING STIRS...");
    trauma.add(0.18);
}

/// Advance campfire phases (bevy `tick_campfire` gameplay half).
/// Sitting waits out its timer, then Rising spawns Throne II via
/// [`PendingEnemySpawn`]; any living IDPD or pending raid parks the
/// fire in `WaitingForIdpd` until the room stays clear 0.35 s.
/// Ember/particle bursts are skipped (renderer-side); trauma, toasts,
/// the `ThroneRises` reactive cue, and the spawn are verbatim.
pub fn tick_campfire(
    time: Res<SimTime>,
    mut commands: Commands,
    mut transition: ResMut<LoopTransition>,
    raid: Res<IdpdRaidState>,
    run: Res<Run>,
    mut trauma: ResMut<Trauma>,
    mut toast: ResMut<Toast>,
    enemies: Query<&Enemy>,
    mut campfires: Query<(Entity, &Pos, &mut CampfireState), With<CampfireProp>>,
) {
    let dt = time.delta_secs;
    let idpd_alive = enemies
        .iter()
        .filter(|enemy| is_idpd_kind(enemy.kind))
        .count();

    let raid_pending = raid.pending_wave.is_some();
    let needs_idpd_clear = campfire_needs_idpd_clear(idpd_alive, raid_pending);

    for (entity, camp_pos, mut campfire) in campfires.iter_mut() {
        let anchor = camp_pos.0;

        match campfire.phase {
            CampfirePhase::Sitting => {
                if needs_idpd_clear {
                    if !campfire.idpd_gate_armed {
                        campfire.arm_idpd_gate();

                        if idpd_alive > 0 {
                            toast.show("CLEAR THE IDPD");
                        }

                        trauma.add(0.10);
                    }

                    continue;
                }

                campfire.timer.tick(dt);

                if campfire.timer.just_finished() {
                    start_campfire_rising(&mut campfire, &mut toast, &mut trauma);
                }
            }

            CampfirePhase::WaitingForIdpd => {
                if needs_idpd_clear {
                    campfire.reset_idpd_clear_confirmation();
                    continue;
                }

                campfire.idpd_clear_confirm.tick(dt);

                if campfire.idpd_clear_confirm.just_finished() {
                    start_campfire_rising(&mut campfire, &mut toast, &mut trauma);
                }
            }

            CampfirePhase::Rising => {
                campfire.timer.tick(dt);
                trauma.add(0.02);

                if campfire.timer.just_finished() {
                    campfire.set_phase(CampfirePhase::SpawnThroneII, 0.35);
                }
            }

            CampfirePhase::SpawnThroneII => {
                campfire.timer.tick(dt);

                if !campfire.timer.just_finished() || campfire.spawned_throne_ii {
                    continue;
                }

                campfire.spawned_throne_ii = true;
                transition.throne_ii_spawned();

                let spawn = anchor + glam::Vec2::new(0.0, 84.0);

                commands.spawn((
                    GameCleanup,
                    LevelCleanup,
                    PendingEnemySpawn {
                        kind: EnemyKind::ThroneII,
                        pos: spawn,
                        difficulty: 1.0 + transition.last_completed_loop as f32 * 0.45,
                        loops: run.loop_count,
                    },
                ));

                trauma.add(0.45);
                commands.spawn((GameCleanup, QueuedReactiveCue(ReactiveCue::ThroneRises)));
                toast.show("THE THRONE RISES");

                commands.entity(entity).despawn();
            }
        }
    }
}

/// Animate campfire YV couches (GML `YungVenuzCouch/Step_0`:
/// `image_speed = timescale * 0.4`, airhorn one-shot back to idle on
/// animation end). Strip fps/frames resolve from the catalog when the
/// art is packed, else the GML strip values (idle 12 fps / 24 frames,
/// airhorn 6 fps / 22 frames).
pub fn tick_yv_couch(
    time: Res<SimTime>,
    catalog: Option<Res<repame_anim::AnimCatalog>>,
    mut q: Query<&mut YvCouch>,
) {
    let steps = time.delta_secs * 30.0;
    for mut couch in q.iter_mut() {
        let path = couch.sprite_path();
        let (fps, frames) = catalog
            .as_deref()
            .and_then(|c| c.def(path))
            .map(|d| (d.fps, d.frames))
            .unwrap_or(if couch.airhorn { (6.0, 22) } else { (12.0, 24) });
        crate::comps_b::yv_couch_step(&mut couch, steps, fps, frames);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn campfire_guards_and_announces() {
        let mut world = bevy_ecs::prelude::World::new();
        world.init_resource::<LoopTransition>();
        world.init_resource::<Toast>();
        world.init_resource::<Trauma>();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(
            |mut commands: Commands,
             mut transition: ResMut<LoopTransition>,
             mut toast: ResMut<Toast>,
             mut trauma: ResMut<Trauma>| {
                begin_throne_campfire(
                    &mut commands,
                    &mut transition,
                    &mut toast,
                    &mut trauma,
                    glam::Vec2::ZERO,
                );
            },
        );
        schedule.run(&mut world);
        let transition = world.resource::<LoopTransition>();
        assert!(transition.campfire_active);
        assert_eq!(world.resource::<Toast>().text, "REST");
        assert!((world.resource::<Trauma>().amount - 0.10).abs() < 1e-6);
        let mut q = world.query::<(&CampfireProp, &Pos)>();
        let (_, pos) = q.iter(&world).next().expect("campfire spawned");
        assert_eq!(pos.0, glam::Vec2::new(0.0, -42.0));
        // YV watches from his couch beside the fire (idle gaming YV).
        let mut cq = world.query::<(&YvCouch, &Pos)>();
        let (couch, cpos) = cq.iter(&world).next().expect("couch spawned");
        assert!(!couch.airhorn && couch.frame == 0.0);
        assert_eq!(cpos.0, glam::Vec2::new(72.0, -42.0));
        // Second call while active: no-op.
        schedule.run(&mut world);
        assert_eq!(q.iter(&world).count(), 1);
    }

    #[test]
    fn yv_couch_ticks_idle_and_airhorn() {
        let mut world = bevy_ecs::prelude::World::new();
        let mut time = repame_sim::SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        world.insert_resource(repame_anim::AnimCatalog::from_json(
            "{}",
            repame_anim::AtlasDesc {
                size: 128,
                max_pages: 1,
            padding: 0,
            },
        ).expect("empty catalog parses"));
        let couch = world.spawn(YvCouch::idle()).id();
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(tick_yv_couch);
        schedule.run(&mut world);
        // Empty catalog: GML strip fallbacks (idle 12 fps / 24 frames).
        let frame = world.get::<YvCouch>(couch).unwrap().frame;
        assert!((frame - 0.16).abs() < 1e-6, "got {frame}");
        world.get_mut::<YvCouch>(couch).unwrap().play_airhorn();
        for _ in 0..1000 {
            schedule.run(&mut world);
            if !world.get::<YvCouch>(couch).unwrap().airhorn {
                break;
            }
        }
        assert!(!world.get::<YvCouch>(couch).unwrap().airhorn);
    }

    fn campfire_world() -> bevy_ecs::prelude::World {
        let mut world = bevy_ecs::prelude::World::new();
        world.init_resource::<LoopTransition>();
        world.init_resource::<Run>();
        world.init_resource::<crate::comps_b::IdpdRaidState>();
        world.init_resource::<Toast>();
        world.init_resource::<Trauma>();
        let mut time = repame_sim::SimTime::default();
        time.delta_secs = 1.0 / 30.0;
        world.insert_resource(time);
        world.spawn((CampfireProp, CampfireState::new(), Pos(glam::Vec2::ZERO)));
        world
    }

    fn campfire_schedule() -> bevy_ecs::schedule::Schedule {
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(tick_campfire);
        schedule
    }

    #[test]
    fn campfire_gate_predicate() {
        assert!(!campfire_needs_idpd_clear(0, false));
        assert!(campfire_needs_idpd_clear(1, false));
        assert!(campfire_needs_idpd_clear(0, true));
    }

    #[test]
    fn campfire_advances_sitting_to_throne_spawn() {
        let mut world = campfire_world();
        let fire = world
            .query::<(Entity, &CampfireProp)>()
            .iter(&world)
            .next()
            .unwrap()
            .0;
        let mut schedule = campfire_schedule();

        // Sitting (3.5 s) -> Rising with toast + trauma.
        for _ in 0..130 {
            schedule.run(&mut world);
            if world.get::<CampfireState>(fire).unwrap().phase == CampfirePhase::Rising {
                break;
            }
        }
        let state = world.get::<CampfireState>(fire).unwrap();
        assert_eq!(state.phase, CampfirePhase::Rising);
        assert_eq!(world.resource::<Toast>().text, "SOMETHING STIRS...");
        assert!(world.resource::<Trauma>().amount > 0.0);

        // Rising (1.15 s) -> SpawnThroneII.
        for _ in 0..60 {
            schedule.run(&mut world);
            if world.get::<CampfireState>(fire).is_none()
                || world.get::<CampfireState>(fire).unwrap().phase
                    == CampfirePhase::SpawnThroneII
            {
                break;
            }
        }
        assert_eq!(
            world.get::<CampfireState>(fire).unwrap().phase,
            CampfirePhase::SpawnThroneII
        );

        // SpawnThroneII (0.35 s): Throne II queued, fire consumed.
        for _ in 0..30 {
            schedule.run(&mut world);
            if world.get_entity(fire).is_err() {
                break;
            }
        }
        assert!(world.get_entity(fire).is_err(), "campfire consumed");
        let pending: Vec<_> = world
            .query::<&crate::combat::PendingEnemySpawn>()
            .iter(&world)
            .collect();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].kind, crate::data::EnemyKind::ThroneII);
        assert_eq!(pending[0].pos, glam::Vec2::new(0.0, 84.0));
        assert_eq!(world.resource::<Toast>().text, "THE THRONE RISES");
        assert!(world.resource::<LoopTransition>().throne_ii_alive);
        assert_eq!(
            world
                .query::<&crate::audio::QueuedReactiveCue>()
                .iter(&world)
                .count(),
            1,
            "ThroneRises cue queued"
        );
    }

    #[test]
    fn campfire_parks_on_idpd_then_resumes_when_clear() {
        let mut world = campfire_world();
        let grunt = world
            .spawn(crate::comps_b::Enemy {
                kind: crate::data::EnemyKind::IdpdGrunt,
                score: 0,
                touch_damage: 1,
                rad_drop: 0,
                drop_chance: 0,
                weapon_chance: 0,
            })
            .id();
        let mut schedule = campfire_schedule();

        schedule.run(&mut world);
        let fire = world
            .query::<(Entity, &CampfireProp)>()
            .iter(&world)
            .next()
            .unwrap()
            .0;
        assert_eq!(
            world.get::<CampfireState>(fire).unwrap().phase,
            CampfirePhase::WaitingForIdpd
        );
        assert_eq!(world.resource::<Toast>().text, "CLEAR THE IDPD");

        // Still parked while the grunt lives, even after long waits.
        for _ in 0..120 {
            schedule.run(&mut world);
        }
        assert_eq!(
            world.get::<CampfireState>(fire).unwrap().phase,
            CampfirePhase::WaitingForIdpd
        );

        // Room clear 0.35 s -> Rising.
        world.despawn(grunt);
        for _ in 0..30 {
            schedule.run(&mut world);
            if world.get::<CampfireState>(fire).unwrap().phase == CampfirePhase::Rising {
                break;
            }
        }
        assert_eq!(
            world.get::<CampfireState>(fire).unwrap().phase,
            CampfirePhase::Rising
        );
        assert_eq!(world.resource::<Toast>().text, "SOMETHING STIRS...");
    }

    #[test]
    fn campfire_parks_on_pending_raid_without_accusing_idpd() {
        let mut world = campfire_world();
        world.resource_mut::<crate::comps_b::IdpdRaidState>().pending_wave =
            Some(crate::comps_b::RaidWave::Light);
        let mut schedule = campfire_schedule();
        schedule.run(&mut world);
        let fire = world
            .query::<(Entity, &CampfireProp)>()
            .iter(&world)
            .next()
            .unwrap()
            .0;
        assert_eq!(
            world.get::<CampfireState>(fire).unwrap().phase,
            CampfirePhase::WaitingForIdpd
        );
        assert!(
            world.resource::<Toast>().text.is_empty(),
            "no IDPD alive: no CLEAR toast"
        );
    }
}
