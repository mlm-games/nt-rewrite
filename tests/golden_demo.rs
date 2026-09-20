//! Golden demo-level walkthrough on the pre-migration pipeline.
//! Boots a deterministic run, plays a scripted input tape through the
//! LIVE per-frame path (`feed_polled` with a synthetic `Scheduler`
//! snapshot → `feed_input` → `advance`), and checks structural
//! invariants per tick. The sim uses thread RNG for drops and aim
//! jitter, so exact positions never reproduce across runs. What IS
//! deterministic: movement responds, firing spends ammo and spawns
//! projectiles, kills increment, pause freezes the world, the run
//! never leaves InGame, the player never dies.
//!
//! Live-path coverage (deliberately NOT bypassed):
//! - every frame builds a synthetic `Scheduler` (polled `held_keys` +
//!   mouse levels mirror the hardware truth the tape asserts) and runs
//!   `feed_polled` before `feed_input`, exactly like `view` does. If
//!   `reconcile_held` drops a still-down key, movement stalls here.
//! - Esc travels the real shortcut pipeline: `install` (map + handler)
//!   once, then `resolve_action` + `handle` per press like the runtime
//!   dispatch does. `drain` runs inside `feed_input` as live.
//! - the fire leg stages TWO `stage_click`s before one `feed_input`
//!   (press + motion event in the same frame). A click pileup that
//!   fires twice fails the exact first-shot tick below.
//! - the release leg drops the key from the synthetic snapshot AND
//!   stages the key-up: levels must clear, not latch.
//!
//! The tape (seed 4242, desert 1-1):
//! - ticks 0-29: hold W (walk north)
//! - tick 30: tap E (interact pulse)
//! - ticks 92-119: hold D (strafe east; BEFORE the fire leg so the
//!   only other actor can't touch the player mid-walk)
//! - ticks 140-147: hold LMB (fire revolver at the bandit)
//! - tick 148: release LMB
//! - ticks 170-171: tap Space (swap pulse)
//! - tick 180: Escape (pause), tick 190: Escape (resume)

use std::collections::HashSet;
use std::time::Duration;

use nt_rewrite::comps_a::{Health, Inventory, Player, Projectile, Run};
use nt_rewrite::comps_b::{Enemy, Pickup};
use nt_rewrite::data::AmmoKind;
use nt_rewrite::spatial::Pos;
use nt_rewrite::state::{AppState, Paused};
use nt_rewrite::App;
use repose_core::input::{Key, KeyEvent, KeyEventType, Modifiers, PhysicalKey};
use repose_core::shortcuts::{Action, KeyChord, handle, resolve_action};
use repose_core::Scheduler;

const DT: Duration = Duration::from_millis(33);
const TICKS: usize = 220;
/// Sub-steps per tape tick: `advance` only runs the sim schedule when
/// the accumulator crosses the 30 Hz step, so one 33 ms advance may run
/// zero or one ticks. Four sub-steps guarantee progress per tick, like
/// live frames at variable dt do.
const SUBSTEPS: usize = 4;

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
    ammo_spent: i32,
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
        ammo_spent: 96 - bullets,
        paused: w.get_resource::<Paused>().map(|p| p.0).unwrap_or(false),
        state: w.get_resource::<AppState>().copied().unwrap_or_default(),
    }
}

fn key_event(key: Key, physical: PhysicalKey) -> KeyEvent {
    KeyEvent {
        key,
        modifiers: Modifiers::default(),
        is_repeat: false,
        event_type: KeyEventType::Down,
        utf16_code_point: 0,
        physical: Some(physical),
    }
}

fn key_up(key: Key, physical: PhysicalKey) -> KeyEvent {
    KeyEvent {
        key,
        modifiers: Modifiers::default(),
        is_repeat: false,
        event_type: KeyEventType::Up,
        utf16_code_point: 0,
        physical: Some(physical),
    }
}

/// Hardware truth the tape asserts: which physical keys + mouse buttons
/// are down. `feed_polled` reconciles against exactly this, like the
/// platform runner's snapshot does live.
#[derive(Default)]
struct Hardware {
    keys: HashSet<PhysicalKey>,
    mouse_primary: bool,
}

fn scheduler(hw: &Hardware) -> Scheduler {
    let mut sched = Scheduler::default();
    sched.window_focused = true;
    sched.held_keys = hw.keys.clone();
    sched.mouse_primary = hw.mouse_primary;
    sched
}

fn run_tape() -> Vec<TickSnap> {
    let mut app = App::new_with_seed(4242);
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
    nt_rewrite::nt_shortcuts::install(&nt_rewrite::nt_shortcuts::edges_handle(&app));
    let mut hw = Hardware::default();
    let mut out = Vec::with_capacity(TICKS);
    // Release bookkeeping: key-up must hit BOTH the App level set
    // (the sampler reads `App.held`) and the snapshot (so the
    // reconcile keeps it cleared). Both go through the same focus
    // route as live (`stage_key`); the direct level writer only runs
    // for keys whose Up event carries no physical position.
    let release_key = |app: &mut App, hw: &mut Hardware, key: PhysicalKey, glyph: Key| {
        hw.keys.remove(&key);
        app.stage_key(&key_up(glyph, key));
        app.stage_physical_key(key, false);
    };
    for tick in 0..TICKS {
        match tick {
            0 => {
                hw.keys.insert(PhysicalKey::KeyW);
                app.stage_physical_key(PhysicalKey::KeyW, true);
            }
            30 => {
                release_key(&mut app, &mut hw, PhysicalKey::KeyW, Key::Character('w'));
                app.stage_physical_key(PhysicalKey::KeyW, false);
                hw.keys.insert(PhysicalKey::KeyE);
                app.stage_physical_key(PhysicalKey::KeyE, true);
            }
            31 => {
                release_key(&mut app, &mut hw, PhysicalKey::KeyE, Key::Character('e'));
                app.stage_physical_key(PhysicalKey::KeyE, false);
            }
            40 => {
                hw.keys.insert(PhysicalKey::KeyD);
                app.stage_physical_key(PhysicalKey::KeyD, true);
            }
            140 => {
                let ep = {
                    let w = &mut app.sim.world;
                    w.query::<(&Pos, &Enemy)>()
                        .iter(w)
                        .next()
                        .map(|(p, _)| p.0)
                        .unwrap_or(glam::Vec2::ZERO)
                };
                app.stage_click(ep, [ep.x, ep.y]);
                hw.mouse_primary = true;
                app.pick_down(repame_input::MouseButton::Primary);
            }
            141..=147 => {
                let ep = {
                    let w = &mut app.sim.world;
                    w.query::<(&Pos, &Enemy)>()
                        .iter(w)
                        .next()
                        .map(|(p, _)| p.0)
                        .unwrap_or(glam::Vec2::ZERO)
                };
                app.stage_click(ep, [ep.x, ep.y]);
            }
            148 => {
                hw.mouse_primary = false;
                app.pick_up(repame_input::MouseButton::Primary);
            }
            170 => {
                hw.keys.insert(PhysicalKey::Space);
                app.stage_physical_key(PhysicalKey::Space, true);
                // Release the D strafe so the walk ends parked: Space now
                // fires the live fire path (fire_held OR swap edge), and a
                // held D under a live unlock offer would keep walking.
                hw.keys.remove(&PhysicalKey::KeyD);
            }
            171 => {
                release_key(&mut app, &mut hw, PhysicalKey::Space, Key::Space);
                app.stage_physical_key(PhysicalKey::Space, false);
            }
            180 => {
                // Live Esc path: focus-routed key event into staging AND
                // the runtime shortcut dispatch into the installed
                // handler; `drain` inside `feed_input` moves it to the
                // pause edge like `view` does. Release the key the next
                // tick: live, winit delivers the Up event (a held Esc
                // must not re-toggle while down).
                app.stage_key(&key_event(Key::Escape, PhysicalKey::Escape));
                let chord = KeyChord::new(Key::Escape, Modifiers::default());
                let action = resolve_action(chord).expect("Esc must resolve");
                assert!(handle(action), "pause shortcut must dispatch");
            }
            181 => {
                app.stage_key(&key_up(Key::Escape, PhysicalKey::Escape));
            }
            190 => {
                app.stage_key(&key_event(Key::Escape, PhysicalKey::Escape));
                let chord = KeyChord::new(Key::Escape, Modifiers::default());
                let action = resolve_action(chord).expect("Esc must resolve");
                assert!(handle(action), "resume shortcut must dispatch");
                let _ = Action::Custom("unused".into());
            }
            191 => {
                app.stage_key(&key_up(Key::Escape, PhysicalKey::Escape));
            }
            _ => {}
        }
        // Live per-frame order: polled snapshot first, then drain.
        // (Key-up events above already landed in staging this frame;
        // the snapshot carries the matching level clear.)
        let mut sched = scheduler(&hw);
        app.feed_polled(&mut sched);
        app.feed_input();
        // Key-up dispatch parity: live, winit delivers the Up event
        // through the same focus route. The tape asserts Down via
        // `stage_physical_key`; mirror the Up half here so staging
        // levels clear exactly like hardware release does.
        match tick {
            30 => app.stage_key(&key_up(Key::Character('w'), PhysicalKey::KeyW)),
            31 => app.stage_key(&key_up(Key::Character('e'), PhysicalKey::KeyE)),
            171 => app.stage_key(&key_up(Key::Space, PhysicalKey::Space)),
            _ => {}
        }
        for _ in 0..SUBSTEPS {
            app.advance(DT);
        }
        out.push(snap(&mut app, tick));
    }
    let _ = hw;
    out
}

#[test]
fn golden_demo_level_walkthrough() {
    let snaps = run_tape();
    assert_eq!(snaps.len(), TICKS);
    let start = &snaps[0];
    assert_eq!(start.hp, 10);
    assert_eq!(start.bullets, 96, "Fish revolver start ammo");

    let walk_end = &snaps[29];
    assert_eq!((walk_end.px, walk_end.py), (16, 8), "W walks north into wall");

    let strafe_end = &snaps[119];
    assert!(
        strafe_end.px > walk_end.px,
        "D must strafe east, went {} -> {}",
        walk_end.px,
        strafe_end.px
    );

    let fired: Vec<_> = snaps.iter().filter(|s| s.shots > 0).collect();
    assert!(!fired.is_empty(), "LMB hold must fire at least one shot");
    let first = fired[0];
    assert_eq!((first.tick, first.shots), (140, 1));
    assert_eq!(first.ammo_spent, 1, "first volley costs exactly 1 bullet");

    let kill = snaps.iter().find(|s| s.kills > 0).expect("bandit must die");
    assert_eq!(kill.kills, 1, "exactly one kill");
    assert!(kill.shots >= 1, "kill must come from firing, got {:?}", kill);
    assert!(
        kill.ammo_spent >= kill.shots as i32,
        "every shot costs at least a bullet, got {:?}",
        kill
    );

    // Esc pauses at 180; the resume Esc at 190 starts the delayed
    // Resume path (overlay None, pending timer armed, paused follows
    // when the 0.2 s timer drains — same as the Resume button).
    assert!(!snaps[179].paused, "must be live before Esc");
    assert!(snaps[180].paused, "Esc must pause");
    assert!(snaps[189].paused, "must stay paused until resume");
    assert!(
        !snaps[195].paused,
        "resume Esc must unpause after the delay"
    );
    let frozen = &snaps[180];
    for s in &snaps[181..190] {
        assert_eq!((s.px, s.py), (frozen.px, frozen.py), "world moved while paused");
    }

    for s in &snaps {
        assert_eq!(s.hp, 10, "player took damage at t{:03}", s.tick);
        assert_eq!(s.state, AppState::InGame, "left InGame at t{:03}", s.tick);
        assert_eq!(s.level, 1, "leveled mid-walk at t{:03}", s.tick);
    }
}
