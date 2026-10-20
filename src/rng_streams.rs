//! The port's simulation streams: one [`RunnerRng`] per subsystem.
//!
//! The draws these replace are the original's built-in `random()`/`irandom()`
//! calls, whose law is the runtime's WELL512 (see [`crate::runner_rng`]). The
//! original keeps a single built-in generator, and `scrRngStatesReset` re-seeds
//! it once per floor (`random_set_seed(_number)`), which is why these streams
//! are rebuilt on every floor from that same `_number`.
//!
//! The port does not reproduce the original's *draw order*: the systems that
//! draw are not the original's instances in the original's order, and several
//! of them (fog, decals, effects) do not exist there. One stream per subsystem,
//! every one of them derived from the floor's `_number`, keeps a subsystem's
//! draws from moving when an unrelated one draws, which is what makes a run
//! replayable. Presentation draws live on `fx` so they can never move a
//! simulation stream.
//!
//! Where the original re-seeds immediately before drawing (`scrPopEnemies`,
//! `scrDecideWep`, `scrPopulate`'s phases), the port uses [`RunnerRng::seeded`]
//! with the original's own seed at the call site and does reproduce the
//! sequence exactly; those sites do not go through this resource. `scrDrop`
//! still draws from the ambient stream here.

use bevy_ecs::prelude::*;

use crate::gml_rng::RngStates;
use crate::runner_rng::RunnerRng;

/// Per-stream tags. Every simulation stream is the floor's `_number` under one
/// of these, so no two subsystems share a sequence.
const TAG_ENEMIES: u32 = 0x9E37_79B9;
const TAG_COMBAT: u32 = 0x85EB_CA6B;
const TAG_ENVIRONMENT: u32 = 0xC2B2_AE35;
const TAG_PICKUPS: u32 = 0x27D4_EB2F;
const TAG_PLAYER_FIRE: u32 = 0x1656_67B1;
const TAG_BOSS_AI: u32 = 0xD1B5_4A35;
const TAG_SPAWNS: u32 = 0x94D0_49BB;
const TAG_PLAYER: u32 = 0x2545_F491;
const TAG_PROGRESSION: u32 = 0x9E37_79B7;
const TAG_SETUP: u32 = 0xBF58_476D;
const TAG_IDPD: u32 = 0x1CE4_E5B9;
const TAG_DEATHS: u32 = 0x1331_11EB;
const TAG_ENEMY_DATA: u32 = 0xA076_1D64;
const TAG_CROWN: u32 = 0x78BD_642F;
const TAG_STATE: u32 = 0xE703_7ED1;
const TAG_WALLS: u32 = 0xA0B4_28DB;
const TAG_WORLDGEN: u32 = 0x8EBC_6AF0;
const TAG_FX: u32 = 0x9C88_C6E3;

/// Every simulation stream the port draws from.
#[derive(Resource, Clone, Copy, Debug)]
pub struct SimStreams {
    /// Enemy movement, targeting and spawn rolls.
    pub enemies: RunnerRng,
    /// The hit pipeline's rolls.
    pub combat: RunnerRng,
    /// Fog, rooms and the rest of the environment.
    pub environment: RunnerRng,
    /// Pickup and chest rolls.
    pub pickups: RunnerRng,
    /// The player's shots.
    pub player_fire: RunnerRng,
    /// The bosses' decisions.
    pub boss_ai: RunnerRng,
    /// Spawn tables.
    pub spawns: RunnerRng,
    /// The player's own rolls.
    pub player: RunnerRng,
    /// Run progression.
    pub progression: RunnerRng,
    /// The run's setup.
    pub setup: RunnerRng,
    /// The IDPD portals.
    pub idpd: RunnerRng,
    /// Death and revive.
    pub deaths: RunnerRng,
    /// The enemy data tables.
    pub enemy_data: RunnerRng,
    /// The crown.
    pub crown: RunnerRng,
    /// Run state.
    pub state: RunnerRng,
    /// Walls and the hammerhead budget.
    pub walls: RunnerRng,
    /// Worldgen phases.
    pub worldgen: RunnerRng,
    /// Presentation only: animation and audio jitter.
    pub fx: RunnerRng,
}

impl SimStreams {
    /// Every stream, derived from the floor's built-in seed (`scrRngStatesReset`'s
    /// `_number`) and, for `fx`, from the run seed so that no floor reset can
    /// move presentation.
    pub fn from_floor(builtin_seed: u32, gen_seed: u64) -> Self {
        let sim = |tag: u32| RunnerRng::seeded(builtin_seed ^ tag);
        Self {
            enemies: sim(TAG_ENEMIES),
            combat: sim(TAG_COMBAT),
            environment: sim(TAG_ENVIRONMENT),
            pickups: sim(TAG_PICKUPS),
            player_fire: sim(TAG_PLAYER_FIRE),
            boss_ai: sim(TAG_BOSS_AI),
            spawns: sim(TAG_SPAWNS),
            player: sim(TAG_PLAYER),
            progression: sim(TAG_PROGRESSION),
            setup: sim(TAG_SETUP),
            idpd: sim(TAG_IDPD),
            deaths: sim(TAG_DEATHS),
            enemy_data: sim(TAG_ENEMY_DATA),
            crown: sim(TAG_CROWN),
            state: sim(TAG_STATE),
            walls: sim(TAG_WALLS),
            worldgen: sim(TAG_WORLDGEN),
            fx: RunnerRng::seeded((gen_seed as u32) ^ TAG_FX),
        }
    }
}

impl Default for SimStreams {
    fn default() -> Self {
        Self::from_floor(RngStates::default().builtin_seed(), 0)
    }
}

/// `global.rng_state[]` for the floor being built, from GML
/// `scrRngStatesReset`. `scrRngStatesReset` runs in `Vlambeer/Create_0`, the
/// room controller, so this is rebuilt at every floor entry with that room's
/// `area`/`subarea`/`loops`.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct FloorStates(pub RngStates);

#[cfg(test)]
mod tests {
    use super::*;
    use repame_sim::rng::Rng;

    #[test]
    fn a_floor_reseeds_every_stream() {
        let mut one = SimStreams::from_floor(1_234, 99);
        let mut two = SimStreams::from_floor(1_235, 99);
        assert_ne!(one.enemies.next_u64(), two.enemies.next_u64());
        assert_ne!(one.worldgen.next_u64(), two.worldgen.next_u64());
        // `fx` rides the run seed, not the floor's.
        assert_eq!(one.fx.next_u64(), two.fx.next_u64());
        let mut other_run = SimStreams::from_floor(1_234, 100);
        assert_ne!(one.fx.next_u64(), other_run.fx.next_u64());

        // Every simulation stream differs from its siblings on one floor.
        let mut seen = std::collections::HashSet::new();
        let streams = SimStreams::from_floor(7, 1);
        for stream in [
            streams.enemies,
            streams.combat,
            streams.environment,
            streams.pickups,
            streams.player_fire,
            streams.boss_ai,
            streams.spawns,
            streams.player,
            streams.progression,
            streams.setup,
            streams.idpd,
            streams.deaths,
            streams.enemy_data,
            streams.crown,
            streams.state,
            streams.walls,
            streams.worldgen,
        ] {
            assert!(seen.insert(stream.seed()));
        }
    }
}
