//! Headless sim schedule: bevy `FixedUpdate` order without the engine.
//!
//! Mirrors `nt-recreated-bevy/src/game/mod.rs` (`NtSimSet::Always` →
//! `Input` → `Combat` → `Progression` → `Cleanup`) as one chained
//! tuple, so bevy `.before()`/`.after()` edges hold by position
//! (`update_carpet_occupancy` before `boss_ai`, `handle_throne_room_props`
//! after `move_projectiles`). `gameplay_active` gates the same subsets
//! bevy gates; `Update`-set audio/UI/ambience systems stay out (shell
//! phase), as do the deferred render/UI systems listed in the audit
//! (sprite strips, toasts-as-text, HUD bridge).
//!
//! Coverage notes live on [`build_sim_schedule`]; the gap list is in
//! the module docs of the audit (see repo notes, not code).

use bevy_ecs::prelude::*;

use crate::comps_a::{PendingMutation, PendingUltra, Run};
use crate::comps_b::FloorTransition;
use crate::state::{AppState, Paused, TransitionBlock};

/// Sim schedule sets, bevy `NtSimSet` parity (documentation value: the
/// chain order below is what actually sequences systems).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NtSimSet {
    Always,
    Input,
    Combat,
    Progression,
    Cleanup,
}

/// Gameplay gate (bevy `gameplay_active` parity, including the
/// `TransitionBlock` conjunct — the port flips states instantly so the
/// resource stays false, but the gate keeps bevy's shape).
pub fn gameplay_active(
    state: Res<AppState>,
    paused: Res<Paused>,
    transition: Option<Res<TransitionBlock>>,
    run: Res<Run>,
    ft: Option<Res<FloorTransition>>,
    pending_mut: Option<Res<PendingMutation>>,
    pending_ultra: Option<Res<PendingUltra>>,
    overlay: Option<Res<crate::state::OverlayMenu>>,
    menu: Option<Res<crate::state::menus::MenuState>>,
) -> bool {
    let loading = ft.map(|f| f.active).unwrap_or(false);
    let picking = pending_mut.is_some() || pending_ultra.is_some();
    let blocked = transition.is_some_and(|t| t.0);
    // Overlay menus (pause/settings/credits) freeze gameplay even on
    // paths that open them without the `Paused` flag. GML
    // `UnlockScreen` panels likewise own the frame while queued (the
    // run sits behind the panel, same as any other overlay).
    let overlay_open = overlay.is_some_and(|o| *o != crate::state::OverlayMenu::None);
    let unlock_open = menu.is_some_and(|m| !m.unlock_queue.is_empty());
    *state == AppState::InGame
        && !paused.0
        && !blocked
        && !run.game_over
        && !loading
        && !picking
        && !overlay_open
        && !unlock_open
}

/// In-game gate for the cleanup tail (bevy `in_state(InGame)` parity).
pub fn in_game(state: Res<AppState>) -> bool {
    *state == AppState::InGame
}

/// Build the fixed-step sim schedule in bevy order.
///
/// Registered: every ported `FixedUpdate` system, including the
/// secret-area observers/detects (`secrets::observe_oasis_floor_start`,
/// `detect_oasis_eligibility`, `detect_cursed_caves`, `detect_hq`,
/// `secret_debug_toast` in `Always`; `tick_oasis_bandit_window` in
/// `Combat`), `loop_transition::tick_campfire` (`Always`, before
/// `flush_pending_enemy_spawns`), `loop_transition::tick_yv_couch`
/// (campfire couch anim, `Always`), `deaths::tick_revive` (coop downed
/// timers, `Always`), `environment::tick_fog` (area-fog scroll,
/// `Always`, self-gated on pause), and the environment sim half
/// (`apply_surface_effects` after `player_move`/`enemy_ai`,
/// `tick_proximity_mines` before `apply_explosions`,
/// `tick_environment_hazards` after `apply_explosions`, all `Combat`).
/// NOT registered (audited gaps): presentation-only systems with no
/// sim state (`face_aim` flip — resolved renderer-side from AimDir;
/// `blink_player` alpha — resolved renderer-side from invuln;
/// `animate_environment` alpha — resolved renderer-side from
/// `SurfacePulse` via the same wave law; `sprite_from_candidates`
/// records its pick in `PulseSprite` at spawn)
/// and `Update`-set audio / HUD / ambience systems (music/intensity
/// bridges). `sample_input` IS ported (keyboard/mouse/gamepad/touch
/// samplers feed `NtInput` through the `App` shell staging).
/// `hurt_on_damage` / `prop_hurt_on_damage` ARE registered (state half
/// only — image/rect/anchor/flip resolve renderer-side);
/// `ensure_weapon_visual` / `tick_weapon_visuals` ARE registered
/// (entity + wkick/wep state; pose/art resolve renderer-side);
/// `clear_input_pulses`
/// and `clear_input_when_inactive` ARE registered (see below).
/// `combat::tick_hit_flash` is sim-side only (no bevy counterpart) and rides in `Always` as the `HitFlash` marker drain,
/// with the transient-FX tail behind it (`effects::tick_fired_weapons`
/// muzzle expiry, `effects::step_fx` particle/number/trauma/flash
/// stepping).
pub fn build_sim_schedule() -> Schedule {
    use crate::anim;
    use crate::boss_ai;
    use crate::combat;
    use crate::crown;
    use crate::deaths;
    use crate::effects;
    use crate::enemies;
    use crate::environment;
    use crate::idpd;
    use crate::loop_transition;
    use crate::pickups;
    use crate::player;
    use crate::player_fire;
    use crate::progression;
    use crate::savedata_part;
    use crate::secrets;
    use crate::state;
    use crate::walls;

    let mut sched = Schedule::default();
    // bevy_ecs 0.19 caps config tuples at 20 nodes, so the bevy order
    // is split into chained groups; the outer chain keeps the total
    // order (Always → Input → Combat → Progression → Cleanup), which is
    // what satisfies bevy's `.before()`/`.after()` edges by position.
    sched.add_systems(
        (
            (
                anim::animate_sprites.in_set(NtSimSet::Always),
                pickups::tick_toast.in_set(NtSimSet::Always),
                secrets::observe_oasis_floor_start.in_set(NtSimSet::Always),
                secrets::detect_oasis_eligibility.in_set(NtSimSet::Always),
                secrets::detect_cursed_caves.in_set(NtSimSet::Always),
                secrets::detect_hq.in_set(NtSimSet::Always),
                secrets::secret_debug_toast.in_set(NtSimSet::Always),
                crown::tick_crown_life.in_set(NtSimSet::Always),
                crown::tick_crown_protection.in_set(NtSimSet::Always),
                crown::tick_crown_love.in_set(NtSimSet::Always),
                crown::tick_crown_curses.in_set(NtSimSet::Always),
                crown::tick_crown_love_convert.in_set(NtSimSet::Always),
                crown::tick_crown_life_convert.in_set(NtSimSet::Always),
                crown::tick_crown_luck.in_set(NtSimSet::Always),
                crown::crown_floor_start_bonus.in_set(NtSimSet::Always),
                crown::tick_crown_pedestal.in_set(NtSimSet::Always),
                crown::tick_crown_object.in_set(NtSimSet::Always),
            )
                .chain(),
            (
                loop_transition::tick_campfire.in_set(NtSimSet::Always),
                loop_transition::tick_yv_couch.in_set(NtSimSet::Always),
                deaths::tick_revive.in_set(NtSimSet::Always),
                environment::tick_fog.in_set(NtSimSet::Always),
                enemies::flush_pending_enemy_spawns.in_set(NtSimSet::Always),
                savedata_part::tick_area_skins.in_set(NtSimSet::Always),
                savedata_part::tick_global_skins.in_set(NtSimSet::Always),
                savedata_part::tick_crystal_damage.in_set(NtSimSet::Always),
                savedata_part::tick_robot_skins.in_set(NtSimSet::Always),
                progression::tick_run_clock.in_set(NtSimSet::Always),
            )
                .chain(),
            (
                anim::tick_hurt_anims.in_set(NtSimSet::Always),
                anim::hurt_on_damage.in_set(NtSimSet::Always),
                anim::tick_player_dying.in_set(NtSimSet::Always),
                player::ensure_weapon_visual.in_set(NtSimSet::Always),
                player::tick_weapon_visuals.in_set(NtSimSet::Always),
                state::tick_current_frame.in_set(NtSimSet::Always),
                state::tick_pending_unpause.in_set(NtSimSet::Always),
                state::force_death_overlay_state.in_set(NtSimSet::Always),
                state::sync_audio_channels.in_set(NtSimSet::Always),
                state::tick_sanitize_save.in_set(NtSimSet::Always),
                (
                    progression::tick_portal_suck.in_set(NtSimSet::Always),
                    progression::tick_throne_sit.in_set(NtSimSet::Always),
                ),
                progression::tick_floor_transition.in_set(NtSimSet::Always),
                anim::tick_fire_anims.in_set(NtSimSet::Always),
                enemies::tick_hit_warnings.in_set(NtSimSet::Always),
                enemies::tick_shield_followers.in_set(NtSimSet::Always),
                combat::tick_hit_flash.in_set(NtSimSet::Always),
                state::menus::tick_menus.in_set(NtSimSet::Always),
                progression::handle_mutation_choice.in_set(NtSimSet::Always),
                progression::apply_floor_reach_unlocks
                    .in_set(NtSimSet::Always)
                    .run_if(in_game),
                walls::reset_hammerhead_budget
                    .in_set(NtSimSet::Always)
                    .run_if(in_game),
            )
                .chain(),
            // Transient-FX tail (Always, ungated): muzzle expiry then
            // particle/number/trauma/flash stepping. Own group so the
            // group above stays under the 20-node tuple cap.
            (
                effects::tick_fired_weapons.in_set(NtSimSet::Always),
                effects::step_fx.in_set(NtSimSet::Always),
                effects::tick_hitstop_slowmo.in_set(NtSimSet::Always),
            )
                .chain(),
            (
                anim::player_anim_switch.in_set(NtSimSet::Input),
                anim::enemy_anim_switch.in_set(NtSimSet::Input),
                player::tick_player_timers
                    .in_set(NtSimSet::Input)
                    .run_if(gameplay_active),
                player::player_aim
                    .in_set(NtSimSet::Input)
                    .run_if(gameplay_active),
                player::player_move
                    .in_set(NtSimSet::Input)
                    .run_if(gameplay_active),
                player_fire::hammerhead_chew
                    .in_set(NtSimSet::Input)
                    .run_if(gameplay_active),
                player::weapon_switch
                    .in_set(NtSimSet::Input)
                    .run_if(gameplay_active),
                player_fire::player_ability
                    .in_set(NtSimSet::Input)
                    .run_if(gameplay_active),
                player::tick_hold_abilities
                    .in_set(NtSimSet::Input)
                    .run_if(gameplay_active),
                player::tick_loop_sfx
                    .in_set(NtSimSet::Input)
                    .run_if(gameplay_active),
            )
                .chain(),
            (
                (
                    player_fire::player_fire
                        .in_set(NtSimSet::Combat)
                        .run_if(gameplay_active),
                    player::player_post_fire_speed_cap
                        .in_set(NtSimSet::Combat)
                        .run_if(gameplay_active),
                ),
                player_fire::move_swing_fx
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                player_fire::tick_snare_zones
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                player_fire::tick_slowed
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                player_fire::tick_portal_strikes
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                player_fire::tick_hazard_clouds
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                player::ally_ai
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                enemies::enemy_ai
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                enemies::tick_bigmaggot_inspector
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                enemies::tick_frog_eggs
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                enemies::tick_scrap_missiles
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                enemies::tick_throne_balls
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                enemies::tick_mom_shots
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                enemies::tick_super_frogs
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                (
                    enemies::tick_elite_inspectors
                        .in_set(NtSimSet::Combat)
                        .run_if(gameplay_active),
                    enemies::tick_elite_shielders
                        .in_set(NtSimSet::Combat)
                        .run_if(gameplay_active),
                    enemies::tick_elite_blockers
                        .in_set(NtSimSet::Combat)
                        .run_if(gameplay_active),
                    enemies::tick_proto_statues
                        .in_set(NtSimSet::Combat)
                        .run_if(gameplay_active),
                    combat::tick_throne_victory
                        .in_set(NtSimSet::Combat)
                        .run_if(gameplay_active),
                ),
                enemies::tick_delayed_boss_spawns
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                enemies::tick_boss_intro
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                enemies::tick_corpses
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                walls::update_carpet_occupancy
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                boss_ai::boss_ai
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
            )
                .chain(),
            (
                boss_ai::tick_hyper_orbit_crystals
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                boss_ai::tick_boss_taunts
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                player_fire::tick_cry_anim
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                pickups::tick_flung_weapons
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                enemies::tick_lil_hunter_die
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                walls::apply_pending_wall_breaks
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                idpd::tick_idpd_raids
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                idpd::tick_idpd_vans
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                idpd::hq_pressure
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                secrets::tick_oasis_bandit_window
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
            )
                .chain(),
            (
                combat::tick_homing_projectiles
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_sticky_projectiles
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_beams
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_sentry_turrets
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_spawn_grace
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_flame_trails
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_lightning_arcs
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_hit_effects
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_slash_projectiles
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_bullet2_fade
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_projectile_friction
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_grenade_fuse
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_shell_bonus
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
            )
                .chain(),
            (
                combat::move_projectiles
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                environment::apply_surface_effects
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                environment::tick_proximity_mines
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::tick_hazard_clouds
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::apply_explosions
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                environment::tick_environment_hazards
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::projectile_hits
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::contact_damage
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                combat::gamma_guts_aura
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                anim::prop_hurt_on_damage
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
                walls::handle_throne_room_props
                    .in_set(NtSimSet::Combat)
                    .run_if(gameplay_active),
            )
                .chain(),
            (
                combat::resolve_enemy_deaths
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                combat::resolve_death_drops
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                deaths::resolve_player_revives
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                deaths::resolve_player_gameover
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                pickups::tick_pickup_drag
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                pickups::collect_pickups
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                pickups::sync_weapon_label
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                pickups::tick_rad_container_contact
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                progression::portal_check
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                progression::tick_portal_shock
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                progression::tick_portal_clear
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                progression::portal_attract
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                progression::portal_enter
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                progression::animate_portal
                    .in_set(NtSimSet::Progression)
                    .run_if(gameplay_active),
                progression::flush_dirty_save
                    .in_set(NtSimSet::Cleanup)
                    .run_if(in_game),
                // Bevy `Update clear_input_when_inactive`: drops sampled
                // pulses/axes when paused or out of game, after all
                // consumers ran (this also subsumes `OnExit(InGame)
                // clear_input_pulses` — the next tick outside InGame
                // clears anything left).
                crate::input::clear_input_when_inactive.in_set(NtSimSet::Cleanup),
            )
                .chain(),
        )
            .chain(),
    );
    sched
}

#[cfg(test)]
mod schedule_tests {
    use super::*;

    /// Regression: menu rooms must survive full schedule ticks. Batch 1
    /// once removed `LoopTransition` in `reset_menu_room_resources`,
    /// and ungated `Always`-set systems (`tick_campfire` et al take it
    /// as a bare `ResMut`) panicked the schedule every frame on menu
    /// rooms — the remap-screen panic loop + stutter. Drives a real
    /// `App` boot into the Title campfire, then ticks the sim schedule
    /// the way `advance` does.
    #[test]
    fn menu_rooms_survive_schedule_tick() {
        let mut app = crate::App::new_with_seed(4242);
        app.sim.world.insert_resource(crate::state::AppState::Title);
        crate::setup::setup_title_campfire(&mut app.sim.world);
        let mut sched = build_sim_schedule();
        // Would panic before the fix (missing LoopTransition).
        sched.run(&mut app.sim.world);
        sched.run(&mut app.sim.world);
    }
}

