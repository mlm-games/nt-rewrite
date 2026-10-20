//! The port's simulation streams: one engine [`SimRng`] per subsystem.
//!
//! The draws these replace are the original's built-in `random()`/`irandom()`
//! calls. GameMaker does not expose that generator's law, and it is not the
//! law of the game's own `global.rng_state[]` scripts: the installed runner's
//! `random_set_seed` advances `state = state * 214013 + 2531011` and takes
//! `state >> 16`, the MSVC `rand()` LCG, where the game's scripts advance
//! `state = (1103515245 * state + 12345) % 2147483647`. So the built-in draws
//! cannot be reproduced from the game source, and the port does not pretend
//! to: they use the engine's stream, seeded from the run.
//!
//! One stream per subsystem, not one shared stream: the schedule runs systems
//! in a fixed order, but separate streams keep a subsystem's draws from moving
//! when an unrelated one draws, which is what makes a run replayable.
//! Presentation draws live on `fx` so they can never move a simulation stream.
//!
//! For the draws the original makes from its own `rng_state[]`, see
//! [`crate::gml_rng`] instead; that is a different law and a different type.

use bevy_ecs::prelude::*;
use repame_sim::rng::{SimRng, WorldSeed};

/// Every simulation stream the port draws from.
#[derive(Resource, Clone, Copy, Debug)]
pub struct SimStreams {
    /// Enemy movement, targeting and spawn rolls.
    pub enemies: SimRng,
    /// The hit pipeline's rolls.
    pub combat: SimRng,
    /// Fog, rooms and the rest of the environment.
    pub environment: SimRng,
    /// Pickup and chest rolls.
    pub pickups: SimRng,
    /// The player's shots.
    pub player_fire: SimRng,
    /// The bosses' decisions.
    pub boss_ai: SimRng,
    /// Spawn tables.
    pub spawns: SimRng,
    /// The player's own rolls.
    pub player: SimRng,
    /// Run progression.
    pub progression: SimRng,
    /// The run's setup.
    pub setup: SimRng,
    /// The IDPD portals.
    pub idpd: SimRng,
    /// Death and revive.
    pub deaths: SimRng,
    /// The enemy data tables.
    pub enemy_data: SimRng,
    /// The crown.
    pub crown: SimRng,
    /// Run state.
    pub state: SimRng,
    /// Walls and the hammerhead budget.
    pub walls: SimRng,
    /// Worldgen phases.
    pub worldgen: SimRng,
    /// Presentation only: animation and audio jitter.
    pub fx: SimRng,
}

impl SimStreams {
    /// Every stream, derived from the run's seed.
    pub fn from_run_seed(gen_seed: u64) -> Self {
        let seed = WorldSeed::new(gen_seed);
        Self {
            enemies: seed.named("nt.enemies"),
            combat: seed.named("nt.combat"),
            environment: seed.named("nt.environment"),
            pickups: seed.named("nt.pickups"),
            player_fire: seed.named("nt.player_fire"),
            boss_ai: seed.named("nt.boss_ai"),
            spawns: seed.named("nt.spawns"),
            player: seed.named("nt.player"),
            progression: seed.named("nt.progression"),
            setup: seed.named("nt.setup"),
            idpd: seed.named("nt.idpd"),
            deaths: seed.named("nt.deaths"),
            enemy_data: seed.named("nt.enemy_data"),
            crown: seed.named("nt.crown"),
            state: seed.named("nt.state"),
            walls: seed.named("nt.walls"),
            worldgen: seed.named("nt.worldgen"),
            fx: seed.named("nt.fx"),
        }
    }
}

impl Default for SimStreams {
    fn default() -> Self {
        Self::from_run_seed(0)
    }
}
