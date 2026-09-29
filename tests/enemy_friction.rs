//! GML `enemy/Create_0.gml` sets `friction = 0.4` on every enemy, and GMS2
//! applies it to `hspeed`/`vspeed` once per step, between the object's step
//! code and the move. The port carried the value on `EnemyBrain` but only ever
//! read it in the `enemy/Collision_Wall` axis slide, so no enemy ever
//! decelerated: the walk law's `motion_add` ratcheted every one of them to its
//! `Other_10` cap, and once `walk` expired they glided at that cap until
//! something stopped them. That is the "enemies move far more often and far
//! faster" report.
//!
//! This drives the live schedule and pins the decay, so the law cannot be
//! dropped from the pipeline again without this failing.

use std::time::Duration;

use nt_rewrite::App;
use nt_rewrite::comps_a::{FloorMask, Velocity};
use nt_rewrite::comps_b::{Enemy, EnemyBrain};
use nt_rewrite::state::AppState;
use nt_rewrite::spatial::Pos;
use nt_rewrite::time::{GTimer, TimerMode};
use repose_core::Scheduler;

/// One 30 Hz sim step per advance. `34 ms > 1000/30 ms` always crosses the
/// fixed-step accumulator once, and `34 ms < 2 * 1000/30 ms` can never cross
/// it twice, so the step count is exact.
const STEP: Duration = Duration::from_millis(34);

/// The walkable cell with the most open floor around it, so the enemy has
/// room to travel its ~13 px and `move_bounce_solid` cannot inject a bounce
/// mid-decay. The tutorial level is too small to promise a full 5x5 block, so
/// this scores every cell instead of demanding one.
fn open_cell(app: &mut App) -> glam::Vec2 {
    let w = &mut app.sim.world;
    let mask = w.resource::<FloorMask>();
    let mut cells: Vec<(i32, i32)> = mask.cells.iter().copied().collect();
    cells.sort_unstable();
    let mut best: Option<(usize, glam::Vec2)> = None;
    for &(cx, cy) in &cells {
        let open = (-2..=2)
            .flat_map(|dx| (-2..=2).map(move |dy| (dx, dy)))
            .filter(|(dx, dy)| mask.is_walkable(mask.cell_center((cx + dx, cy + dy))))
            .count();
        if best.is_none_or(|(b, _)| open > b) {
            best = Some((open, mask.cell_center((cx, cy))));
        }
    }
    best.expect("generator must offer walkable floor").1
}

#[test]
fn enemy_speed_decays_by_gml_friction_each_step() {
    let mut app = App::new_with_seed(4242);
    app.sim.world.insert_resource(AppState::InGame);
    let start = open_cell(&mut app);
    {
        let w = &mut app.sim.world;
        // `AnimCatalog` is not `Clone` and `w.commands()` needs `&mut`, so the
        // catalog borrow is taken through a raw pointer to split the two.
        let catalog: *const repame_anim::AnimCatalog = w.resource::<repame_anim::AnimCatalog>();
        let mut cmds = w.commands();
        nt_rewrite::enemies::spawn_enemy(
            &mut cmds,
            unsafe { &*catalog },
            // `Turtle` runs the shared walk-law path: `Other_10` impulse 1.0,
            // unconditional cap 5, `PlainBounce` wall law, no dedicated ticker.
            nt_rewrite::EnemyKind::Turtle,
            start,
            1.0,
            false,
            false,
            0,
        );
    }
    {
        // The spawn went through `Commands`, so the entity does not exist in
        // the world yet. Flush before reaching for its components.
        app.sim.world.flush();
    }
    {
        // Silence the two inputs that add velocity of their own: the `walk`
        // impulse (`if walk > 0`) and the `Alarm_1` decide.
        let w = &mut app.sim.world;
        let mut q = w.query::<(&mut EnemyBrain, &mut Velocity)>();
        let mut hit = 0;
        for (mut brain, mut vel) in q.iter_mut(w) {
            hit += 1;
            brain.walk = 0.0;
            brain.attack = GTimer::from_seconds(999.0, TimerMode::Once);
            vel.0 = glam::Vec2::new(3.0 * 30.0, 0.0);
        }
        assert_eq!(hit, 1, "exactly one enemy must be under test");
    }

    let mut sched = Scheduler::default();
    sched.window_focused = true;
    // GML: `friction = 0.4` px/frame, applied after the step code and before the
    // move, so these are post-step speeds from a 3.0 px/frame start. `Turtle`'s
    // cap of 5 is above 3, so `if (speed > 5)` never fires and the decay is
    // pure friction: 3.0 decays to 0 in 8 frames, 9.8 px total.
    let expect: [f32; 8] = [2.6, 2.2, 1.8, 1.4, 1.0, 0.6, 0.2, 0.0];
    for (frame, want) in expect.iter().enumerate() {
        app.feed_polled(&mut sched);
        app.feed_input();
        app.advance(STEP);
        let w = &mut app.sim.world;
        let (speed, vx) = w
            .query::<(&Velocity, &Enemy)>()
            .iter(w)
            .next()
            .map(|(v, _)| (v.0.length() / 30.0, v.0.x / 30.0))
            .expect("the enemy must survive the decay");
        assert!(
            (speed - want).abs() < 0.02,
            "frame {frame}: speed {speed:.3} != {want:.3} — GML friction is not being applied"
        );
        if speed > 0.0 {
            assert!(
                vx > 0.0,
                "frame {frame}: decay must not reverse direction (vx {vx:.3})"
            );
        }
    }
    // Then it must come to rest and STAY there: no walk, no decide, so GML's
    // friction alone drives the last 0.2 px/frame to zero and holds it.
    for frame in 8..12 {
        app.feed_polled(&mut sched);
        app.feed_input();
        app.advance(STEP);
        let w = &mut app.sim.world;
        let speed = w
            .query::<(&Velocity, &Enemy)>()
            .iter(w)
            .next()
            .map(|(v, _)| v.0.length() / 30.0)
            .expect("the enemy must survive the decay");
        assert!(
            speed < 0.001,
            "frame {frame}: an enemy with no walk must be at rest, got {speed:.3} px/frame"
        );
    }
    // Total travel is what GML produces: the friction-weighted sum of the
    // profile, not `3.0 * frames`. Without the law this is more than double.
    let w = &mut app.sim.world;
    let travelled = w
        .query::<(&Pos, &Enemy)>()
        .iter(w)
        .next()
        .map(|(p, _)| (p.0 - start).x)
        .expect("enemy must exist");
    let gml_travel: f32 = expect.iter().sum();
    assert!(
        (travelled - gml_travel).abs() < 0.8,
        "travelled {travelled:.2} px != GML {gml_travel:.2} px"
    );
}
