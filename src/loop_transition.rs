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

