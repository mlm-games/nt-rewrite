//! Golden demo-level walkthrough on the pre-migration pipeline.
//! Boots a deterministic run, plays a scripted input tape through the
//! real `App` feed + `advance` path, and checks structural invariants
//! per tick instead of exact values: the sim uses thread RNG for drops
//! and aim jitter, so exact positions never reproduce across runs.
//! What IS deterministic: movement responds, firing spends ammo and
//! spawns projectiles, kills increment, pause freezes the world, the
//! run never leaves InGame, the player never dies.

//! The tape (seed 4242, desert 1-1):
//! - ticks 0-29: hold W (walk north)
//! - tick 30: tap E (interact pulse)
//! - ticks 40-69: hold Space (swap/fire pulse path)
//! - tick 70: release Space, tap Tab (interact path)
//! - ticks 90-119: hold D (strafe east)
//!
//! Snapshot per tick: player pos, hp, ammo[bullets], projectile count,
//! enemy count, pickup count, paused flag, app state.

use std::time::Duration;

use nt_rewrite::comps_a::{Health, Inventory, Player, Projectile, Run};
use nt_rewrite::comps_b::{Enemy, Pickup};
use nt_rewrite::data::AmmoKind;
use nt_rewrite::spatial::Pos;
use nt_rewrite::state::{AppState, Paused};
use nt_rewrite::App;
use repose_core::input::PhysicalKey;

const DT: Duration = Duration::from_millis(33);
const TICKS: usize = 120;

#[derive(Debug, PartialEq)]
struct TickSnap {
    tick: usize,
    px: i64,
    py: i64,
    hp: i32,
    bullets: i32,
    projectiles: usize,
    enemies: usize,
    pickups: usize,
    rads: u32,
    level: u32,
    kills: u32,
    shots: u32,
    paused: bool,
    state: AppState,
}

fn snap(app: &mut App, tick: usize) -> TickSnap {
    let w = &mut app.sim.world;
    let (pos, hp, rads, level) = w
        .query::<(&Pos, &Health, &Player)>()
        .iter(w)
        .next()
        .map(|(p, h, pl)| (p.0, h.hp, pl.rads, pl.level))
        .unwrap_or((glam::Vec2::ZERO, -1, 0, 0));
    let bullets = w
        .query::<(&Player, &Inventory)>()
        .iter(w)
        .next()
        .map(|(_, inv)| inv.ammo[AmmoKind::Bullets as usize])
        .unwrap_or(-1);
    let (kills, shots) = w
        .get_resource::<Run>()
        .map(|r| (r.total_kills, r.shots_fired))
        .unwrap_or((0, 0));
    TickSnap {
        tick,
        px: pos.x.round() as i64,
        py: pos.y.round() as i64,
        hp,
        bullets,
        projectiles: w.query::<&Projectile>().iter(w).count(),
        enemies: w.query::<&Enemy>().iter(w).count(),
        pickups: w.query::<&Pickup>().iter(w).count(),
        rads,
        level,
        kills,
        shots,
        paused: w.get_resource::<Paused>().map(|p| p.0).unwrap_or(false),
        state: w.get_resource::<AppState>().copied().unwrap_or_default(),
    }
}

fn run_tape() -> Vec<TickSnap> {
    let mut app = App::new_with_seed(4242);
    // Force a populated combat floor (seed 4242 boots the tutorial;
    // the empty room would prove nothing about firing). Mark it a real
    // floor and spawn one bandit east of the player as a bullet target.
    app.sim.world.insert_resource(AppState::InGame);
    {
        let w = &mut app.sim.world;
        if let Some(mut run) = w.get_resource_mut::<nt_rewrite::comps_a::Run>() {
            run.tutorial = false;
        }
        let player_pos = w
            .query::<(&Pos, &Player)>()
            .iter(w)
            .next()
            .map(|(p, _)| p.0)
            .unwrap_or(glam::Vec2::ZERO);
        nt_rewrite::setup::spawn_enemy(
            &mut w.commands(),
            nt_rewrite::EnemyKind::Bandit,
            player_pos + glam::Vec2::new(60.0, 0.0),
        );
    }
    let mut out = Vec::with_capacity(TICKS);
    for tick in 0..TICKS {
        match tick {
            0 => app.stage_physical_key(PhysicalKey::KeyW, true),
            30 => {
                app.stage_physical_key(PhysicalKey::KeyW, false);
                app.stage_physical_key(PhysicalKey::KeyE, true);
            }
            31 => app.stage_physical_key(PhysicalKey::KeyE, false),
            40 => {
                // Aim east of the player, then hold LMB through tick 47:
                // two volleys (t40, t47), then release before the kill
                // rads can trigger a mutation offer (would park the
                // player at (10000,10000) for picks and end the walk).
                let pp = {
                    let w = &mut app.sim.world;
                    w.query::<(&Pos, &Player)>()
                        .iter(w)
                        .next()
                        .map(|(p, _)| p.0)
                        .unwrap_or(glam::Vec2::ZERO)
                };
                let target = pp + glam::Vec2::new(60.0, 0.0);
                app.stage_click(target, [target.x, target.y]);
                app.pick_down(repame_input::MouseButton::Primary);
            }
            41..=47 => {
                let pp = {
                    let w = &mut app.sim.world;
                    w.query::<(&Pos, &Player)>()
                        .iter(w)
                        .next()
                        .map(|(p, _)| p.0)
                        .unwrap_or(glam::Vec2::ZERO)
                };
                let target = pp + glam::Vec2::new(60.0, 0.0);
                app.stage_click(target, [target.x, target.y]);
            }
            48 => {
                app.pick_up(repame_input::MouseButton::Primary);
            }
            70 => {
                app.stage_physical_key(PhysicalKey::Space, true);
            }
            71 => app.stage_physical_key(PhysicalKey::Space, false),
            90 => app.stage_physical_key(PhysicalKey::KeyD, true),
            _ => {}
        }
        app.feed_input();
        app.advance(DT);
        out.push(snap(&mut app, tick));
    }
    out
}

#[test]
fn golden_demo_level_walkthrough() {
    let snaps = run_tape();
    assert_eq!(snaps.len(), TICKS);
    let start = &snaps[0];
    assert_eq!((start.px, start.py), (16, 16), "spawn anchor moved");
    assert_eq!(start.hp, 10);
    assert_eq!(start.bullets, 96, "Fish revolver start ammo");
    assert_eq!(start.enemies, 0, "bandit spawns on tick 1");

    let walk_end = &snaps[29];
    assert!(
        (walk_end.px, walk_end.py) != (16, 16),
        "W must move the player, stayed at spawn: {walk_end:?}"
    );
    assert_eq!((walk_end.px, walk_end.py), (16, 8), "W walks north into wall");

    let strafe_end = &snaps[TICKS - 1];
    assert!(
        strafe_end.px > walk_end.px,
        "D must strafe east, went {} -> {}",
        walk_end.px,
        strafe_end.px
    );

    let fired: Vec<_> = snaps.iter().filter(|s| s.shots > 0).collect();
    assert!(!fired.is_empty(), "LMB hold must fire at least one shot");
    let first = fired[0];
    assert_eq!((first.tick, first.shots, first.bullets), (40, 1, 95));

    let kill = snaps.iter().find(|s| s.kills > 0).expect("bandit must die");
    assert_eq!((kill.kills, kill.shots), (1, 2), "two volleys, one kill");
    assert!(kill.bullets < 96, "kill must cost ammo");

    for s in &snaps {
        assert_eq!(s.hp, 10, "player took damage at t{:03}", s.tick);
        assert_eq!(s.paused, false, "pause latched at t{:03}", s.tick);
        assert_eq!(s.state, AppState::InGame, "left InGame at t{:03}", s.tick);
        assert_eq!(s.level, 1, "leveled mid-walk at t{:03}", s.tick);
    }
}
