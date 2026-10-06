//! Mid-run save (GML `scrSavegame`, `savegame.dat`).
//!
//! GML keeps the live run in a second file beside the profile save. `scrSavegameSave`
//! writes three `buffer_u8`-tagged JSON parts - `SessionData` (the whole `GameCont`
//! struct plus `UberCont` run flags, the portal-carried guns and the pending
//! `UnlockScreen` popups), `GlobalVars` (the nine globals in
//! `scrSavegameGlobals`), and `PlayerData` (every `Player` instance struct).
//! `scrSavegameLoad` reads them back; `_only_first_part` peeks the session part alone.
//!
//! The port keeps the same three groups and the same captured content, in the same RON
//! store the profile save uses, at `run_save_file_path()`. The part framing is a GML
//! buffer detail with no port-side reader, so the three groups are plain fields.
//!
//! What GML does NOT store, and neither does this: the player's position. `scrPlayerCreate`
//! and `GenCont/Alarm_1:17-18` park the player at the `(10016, 10016)` sentinel while the
//! floor builds, and `PortalClear` drops them on the spawn, so a resumed run always
//! restarts at the floor's spawn point.

use std::path::{Path, PathBuf};

use bevy_ecs::prelude::*;
use game_utils::save_store::SaveStore;
use game_utils::storage::FsStorage;
use serde::{Deserialize, Serialize};

use crate::comps_a::{CrownState, Health, Inventory, Player, RaceState, Run};
use crate::comps_b::PortalCarriedWeapons;
use crate::data::{RaceId, SkinLetter, WeaponId};
use crate::state::menus::UnlockPopup;

pub const RUN_SAVE_VERSION: u32 = 1;

/// GML `MakeGame/Other_10:4` `file_rename(savegame_file, "m_gamestate.dat")`: a
/// save that failed to load is quarantined under this name, never deleted, so
/// the next boot takes the clean path (`MakeGame/Other_10:2` drops any older one).
pub const RUN_SAVE_QUARANTINE: &str = "m_gamestate.dat";

/// GML `scrSavegame.gml:169-173` `with UnlockScreen` inserts each popup at
/// index 0, so the stored list is newest-first; the port's queue is oldest-first
/// and the reversal happens here to match.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnlockScreenSave {
    pub race: RaceId,
    pub bskin: u8,
}

/// GML `scrSavegameSession` `_session_info`. `run` is the `GameCont` struct;
/// `unlockscreens` is the pending popup list. GML's `hardmode`, `daily_run`,
/// `weekly_run` and `custom` are `UberCont` flags with no port-side mode
/// behind them, and `recontinued_times` lives in [`GlobalsSave`] where
/// `scrSavegameGlobals` also writes it.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionSave {
    pub run: Run,
    pub unlockscreens: Vec<UnlockScreenSave>,
}

/// GML `scrSavegameGlobals` `_global_keys`. Only `current_frame` and
/// `recontinued_times` have port-side state; the rest (`rng_state`, `index`,
/// `seed`, `is_server`, `custom_seed`, `party_gun_special_drop`, `crownpick`)
/// are netcode or per-room values the port re-derives from `Run.gen_seed`.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct GlobalsSave {
    pub current_frame: u64,
    pub recontinued_times: u32,
}

/// GML `scrSavegamePlayers`: the full `Player` instance struct. The port splits
/// that struct across components, so all of them are stored here and written
/// back onto the spawned player. `portal_carried_weapons` is the port's
/// `persistentweps` (`WepPickup/Collision_Portal:4` marks a gun persistent when
/// the portal vacuums it; `GenCont/Create_0:52-61` re-spawns the set next floor).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerDataSave {
    pub race: RaceId,
    pub skin: SkinLetter,
    pub player: Player,
    pub health: Health,
    pub inventory: Inventory,
    pub crown: CrownState,
    pub portal_carried_weapons: Vec<WeaponId>,
}

/// The save the intro prompt is showing. GML loads the run up front
/// (`MakeGame/Alarm_0:85-87`) and draws the prompt over the live `GameCont`,
/// so the port parks the parsed save here instead of building a half-world:
/// the roadmap, the HUD rows and the YES/NO commit all read it, and picking
/// YES builds the real world through [`crate::setup::setup_continued_run`].
#[derive(Resource, Clone, Debug)]
pub struct PendingRunSave(pub RunSave);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunSave {
    pub version: u32,
    pub session: SessionSave,
    pub globals: GlobalsSave,
    pub player: PlayerDataSave,
}

impl RunSave {
    /// GML `MakeGame/Draw_0:103` reads `GameCont.waypoints` for the roadmap
    /// reveal, so the prompt needs the log length without touching a `Run`.
    pub fn waypoint_count(&self) -> usize {
        self.session.run.waypoints.len()
    }
}

pub fn run_save_file_name() -> &'static str {
    "nt-run.ron"
}

/// Sibling of [`crate::savedata_part::save_file_path`]: the mid-run save lives
/// beside the profile save so a corrupt run blob cannot take the profile with it.
pub fn run_save_file_path() -> PathBuf {
    profile_dir().join(run_save_file_name())
}

fn profile_dir() -> PathBuf {
    crate::savedata_part::save_file_path()
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn store_in(dir: &Path) -> SaveStore<FsStorage> {
    SaveStore::new(dir, run_save_file_name())
        .with_validator(SaveStore::<FsStorage>::is_intact_ron)
}

fn store() -> SaveStore<FsStorage> {
    store_in(&profile_dir())
}

/// GML `MakeGame/Create_0:37` `file_exists(savegame_file)`.
pub fn run_save_exists() -> bool {
    run_save_file_path().is_file()
}

/// GML `file_delete(savegame_file)` - every delete site funnels here
/// (`GameOver/Create_0:55`, `Credits/Create_0:36`, `Cinematic/Create_0:30`,
/// `TutCont/Other_5:3`, `scrGameRestart:23`, `Vlambeer/Create_0:24,28,82`,
/// `MakeGame/Draw_0:166`).
pub fn delete_run_save() {
    delete_run_save_in(&profile_dir());
}

fn delete_run_save_in(dir: &Path) {
    let _ = std::fs::remove_file(dir.join(run_save_file_name()));
}

/// GML `MakeGame/Other_10:1-6`: a save that throws on load is renamed to
/// `m_gamestate.dat` (an older one is dropped first) and the boot restarts.
pub fn quarantine_run_save() {
    quarantine_run_save_in(&profile_dir());
}

fn quarantine_run_save_in(dir: &Path) {
    let _ = std::fs::remove_file(dir.join(RUN_SAVE_QUARANTINE));
    let _ = std::fs::rename(
        dir.join(run_save_file_name()),
        dir.join(RUN_SAVE_QUARANTINE),
    );
}

/// GML `scrSavegamePlayers:234-256` writes the `Player` instances, then the
/// `Revive` instances. The port has neither `Revive` nor co-op, so this is the
/// single local player.
fn capture_player(world: &mut World) -> Option<PlayerDataSave> {
    let carried = world
        .get_resource::<PortalCarriedWeapons>()
        .map(|c| c.0.clone())
        .unwrap_or_default();
    let mut q = world.query::<(
        &crate::comps_a::Player,
        &Health,
        &Inventory,
        &RaceState,
        &CrownState,
    )>();
    let (player, health, inventory, race, crown) = q.iter(world).next()?;
    Some(PlayerDataSave {
        race: race.race,
        skin: race.skin,
        player: player.clone(),
        health: *health,
        inventory: inventory.clone(),
        crown: *crown,
        portal_carried_weapons: carried,
    })
}

/// GML `scrSavegameSession:140-174` + `scrSavegameGlobals:195-204` +
/// `scrSavegamePlayers:233-259`, gathered from the live world.
pub fn capture(world: &mut World) -> Option<RunSave> {
    world.init_resource::<Run>();
    world.init_resource::<crate::state::BootFlags>();
    world.init_resource::<crate::state::menus::MenuState>();
    world.init_resource::<crate::state::CurrentFrame>();
    let player = capture_player(world)?;
    let recontinued_times = world
        .resource::<crate::state::BootFlags>()
        .recontinued_times;
    let run = world.resource::<Run>().clone();
    let mut queue: Vec<UnlockScreenSave> = world
        .resource::<crate::state::menus::MenuState>()
        .unlock_queue
        .iter()
        .map(|popup| match popup {
            UnlockPopup::Race(race) => UnlockScreenSave {
                race: *race,
                bskin: 0,
            },
            UnlockPopup::Skin(race, skin) => UnlockScreenSave {
                race: *race,
                bskin: *skin,
            },
        })
        .collect();
    queue.reverse();
    Some(RunSave {
        version: RUN_SAVE_VERSION,
        session: SessionSave {
            run,
            unlockscreens: queue,
        },
        globals: GlobalsSave {
            current_frame: world.resource::<crate::state::CurrentFrame>().0,
            recontinued_times,
        },
        player,
    })
}

/// GML `scrSavegameSave`: write all three parts. A capture failure (no player,
/// i.e. the death frame already tore it down) is a silent no-op, matching GML's
/// `with Player` writing nothing.
pub fn save_run(world: &mut World) -> Result<(), String> {
    let Some(save) = capture(world) else {
        return Ok(());
    };
    let text = ron::ser::to_string_pretty(&save, Default::default()).map_err(|e| e.to_string())?;
    store().write(text.as_bytes())
}

#[cfg(test)]
fn save_run_in(world: &mut World, dir: &Path) -> Result<(), String> {
    let Some(save) = capture(world) else {
        return Ok(());
    };
    let text = ron::ser::to_string_pretty(&save, Default::default()).map_err(|e| e.to_string())?;
    store_in(dir).write(text.as_bytes())
}

/// GML `scrSavegameLoad`. GML returns `false` on a bad buffer and the caller
/// deletes the save and restarts (`Vlambeer/Create_0:27-31`); here that is
/// `Err`, which [`crate::run_save::quarantine_run_save`] handles.
pub fn load_run() -> Result<RunSave, String> {
    load_run_in(&profile_dir())
}

fn load_run_in(dir: &Path) -> Result<RunSave, String> {
    let result = store_in(dir).load(&SaveStore::<FsStorage>::is_intact_ron, &[]);
    let Some(bytes) = result.data else {
        return Err(format!("run save unavailable: {:?}", result.status));
    };
    let text = std::str::from_utf8(&bytes).map_err(|e| format!("run save is not utf8: {e}"))?;
    ron::from_str(text).map_err(|e| format!("run save is corrupt: {e}"))
}

/// GML `scrSavegamePlayers:219-230` + the `GenCont/Create_0:52-61` carried-gun
/// re-spawn. The `GameCont` half is NOT applied here: `setup_run_inner` already
/// installed it before generation so worldgen reads the restored run (and may
/// write back to it, exactly like GML), so only the player instance and the
/// globals land here.
///
/// The popup list is pushed back in stored order, which is GML's newest-first
/// order: `array_insert(..., 0, ...)` makes the newest unlock
/// `instance_find(UnlockScreen, 0)`, the one `TopCont/Step_2` arms and shows.
pub fn apply_run(world: &mut World, save: &RunSave) {
    world.init_resource::<PortalCarriedWeapons>();
    world.resource_mut::<PortalCarriedWeapons>().0 = save.player.portal_carried_weapons.clone();
    world.insert_resource(crate::state::CurrentFrame(save.globals.current_frame));
    {
        world.init_resource::<crate::state::BootFlags>();
        let mut flags = world.resource_mut::<crate::state::BootFlags>();
        flags.recontinued_times = save.globals.recontinued_times;
        flags.continued_run = true;
    }
    let mut q = world.query_filtered::<Entity, With<crate::comps_a::Player>>();
    let entities: Vec<Entity> = q.iter(world).collect();
    for e in entities {
        if let Some(mut p) = world.get_mut::<Player>(e) {
            *p = save.player.player.clone();
        }
        if let Some(mut h) = world.get_mut::<Health>(e) {
            *h = save.player.health;
        }
        if let Some(mut i) = world.get_mut::<Inventory>(e) {
            *i = save.player.inventory.clone();
        }
        if let Some(mut r) = world.get_mut::<RaceState>(e) {
            r.race = save.player.race;
            r.skin = save.player.skin;
        }
        if let Some(mut c) = world.get_mut::<CrownState>(e) {
            *c = save.player.crown;
        }
    }
    world.init_resource::<crate::state::menus::MenuState>();
    {
        let mut menu = world.resource_mut::<crate::state::menus::MenuState>();
        menu.unlock_queue.clear();
        for entry in &save.session.unlockscreens {
            let popup = if entry.bskin == 0 {
                UnlockPopup::Race(entry.race)
            } else {
                UnlockPopup::Skin(entry.race, entry.bskin)
            };
            menu.unlock_queue.push(popup);
        }
        menu.unlock = crate::state::menus::UnlockPopupState::default();
    }
}
/// GML `MakeGame/Alarm_0:85-97` loads the run before the prompt draws, so the
/// intro screen has a live `Player` for `scrDrawPlayerHUD` (`Draw_0:95`) and a
/// live `GameCont` for `scrDrawMiscHUD` (`Draw_0:141`). The port spawns the
/// same instance with the saved components; `setup_run_inner` tears it down
/// when the prompt commits.
pub fn spawn_saved_player(world: &mut World, save: &RunSave) {
    world.spawn((
        save.player.player.clone(),
        save.player.health,
        save.player.inventory.clone(),
        RaceState {
            race: save.player.race,
            skin: save.player.skin,
        },
        save.player.crown,
        crate::Pos(glam::Vec2::splat(crate::comps_a::TILE * 0.5)),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::MutationId;
    use crate::savedata_part::SaveData;
    use crate::setup::{build_player_bundle, resolve_run_loadout};

    /// A real player bundle so the captured `Health`/`Inventory`/`CrownState`
    /// carry the same shape the run writes.
    fn bundle(race: RaceId) -> impl Bundle {
        let save = SaveData::default();
        let b = build_player_bundle(race, &resolve_run_loadout(&save, race));
        (
            crate::comps_a::Player {
                rads: 33,
                ..b.player
            },
            b.health,
            b.inv,
            b.race_state,
            b.crown_state,
        )
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("nt-run-save-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn loaded_player(world: &mut World) -> PlayerDataSave {
        capture(world).expect("capture").player
    }

    /// GML `scrSavegameSession:169-173` inserts each `UnlockScreen` at index
    /// 0, so the stored list is newest-first; the port's queue is oldest-first
    /// and the reversal must survive the round trip.
    #[test]
    fn capture_reverses_the_unlock_queue() {
        let mut world = World::new();
        world.init_resource::<Run>();
        world.init_resource::<crate::state::menus::MenuState>();
        world.spawn(bundle(RaceId::Chicken));
        {
            let mut menu = world.resource_mut::<crate::state::menus::MenuState>();
            menu.unlock_queue.push(UnlockPopup::Race(RaceId::Fish));
            menu.unlock_queue
                .push(UnlockPopup::Skin(RaceId::Chicken, 1));
        }
        let save = capture(&mut world).expect("capture");
        assert_eq!(
            save.session.unlockscreens,
            vec![
                UnlockScreenSave {
                    race: RaceId::Chicken,
                    bskin: 1
                },
                UnlockScreenSave {
                    race: RaceId::Fish,
                    bskin: 0
                },
            ]
        );
    }

    /// The three GML groups must survive serialization intact: the `GameCont`
    /// struct, the globals and the full `Player` instance.
    #[test]
    fn run_save_round_trips_through_disk() {
        let dir = scratch("round-trip");

        let mut world = World::new();
        world.init_resource::<Run>();
        world.init_resource::<crate::state::menus::MenuState>();
        world.insert_resource(crate::state::CurrentFrame(1234));
        {
            let mut run = world.resource_mut::<Run>();
            run.floor = 7;
            run.area = crate::data::AreaId::City;
            run.loop_count = 2;
            run.gen_seed = 0xDEAD_BEEF;
            run.total_kills = 41;
            run.waypoints.push(crate::comps_a::Waypoint {
                area: 3,
                sub: 2,
                lp: 1,
            });
        }
        world.spawn(bundle(RaceId::Rogue));
        {
            let mut player = world
                .query_filtered::<&mut crate::comps_a::Player, With<crate::comps_a::Player>>()
                .iter_mut(&mut world)
                .next()
                .expect("player");
            player.rads = 33;
            player.mutations.push(MutationId::StrongSpirit);
        }
        let source = loaded_player(&mut world);

        save_run_in(&mut world, &dir).expect("write");
        assert!(dir.join(run_save_file_name()).is_file());

        // Every later save must overwrite in place: a write that quarantines
        // its own target renames it to `corrupted_*`, which on wasm panics
        // inside `std::process::id()`.
        world.resource_mut::<Run>().floor = 8;
        save_run_in(&mut world, &dir).expect("rewrite");
        let rewritten = load_run_in(&dir).expect("reload");
        assert_eq!(rewritten.session.run.floor, 8);
        let leftovers: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("corrupted_"))
            .collect();
        assert!(leftovers.is_empty(), "quarantined: {leftovers:?}");

        let loaded = load_run_in(&dir).expect("load");
        assert_eq!(loaded.session.run.floor, 7);
        assert_eq!(loaded.session.run.area, crate::data::AreaId::City);
        assert_eq!(loaded.session.run.loop_count, 2);
        assert_eq!(loaded.session.run.gen_seed, 0xDEAD_BEEF);
        assert_eq!(loaded.session.run.total_kills, 41);
        assert_eq!(loaded.session.run.waypoints.len(), 1);
        assert_eq!(loaded.globals.current_frame, 1234);
        assert_eq!(loaded.player.race, source.race);
        assert_eq!(loaded.player.skin, source.skin);
        assert_eq!(loaded.player.health.hp, source.health.hp);
        assert_eq!(loaded.player.health.max, source.health.max);
        assert_eq!(loaded.player.inventory.weapons, source.inventory.weapons);
        assert_eq!(loaded.player.inventory.cursed, source.inventory.cursed);
        assert_eq!(
            loaded.player.inventory.weapon_slots,
            source.inventory.weapon_slots
        );
        assert_eq!(loaded.player.inventory.current, source.inventory.current);
        assert_eq!(loaded.player.inventory.ammo, source.inventory.ammo);
        assert_eq!(loaded.player.crown.crown, source.crown.crown);
        assert_eq!(loaded.player.player.rads, 33);
        assert!(
            loaded
                .player
                .player
                .mutations
                .contains(&MutationId::StrongSpirit)
        );

        delete_run_save_in(&dir);
        assert!(!dir.join(run_save_file_name()).exists());
    }

    /// `apply_run` writes the restored instance back over the spawned player,
    /// so a resumed run runs the saved HP/guns/mutations, not the loadout's.
    #[test]
    fn apply_run_overwrites_the_spawned_player() {
        let dir = scratch("apply");
        let mut world = World::new();
        world.init_resource::<Run>();
        world.init_resource::<crate::state::menus::MenuState>();
        world.spawn(bundle(RaceId::Chicken));
        {
            let mut run = world.resource_mut::<Run>();
            run.floor = 9;
            run.gen_seed = 4242;
        }
        {
            let mut player = world
                .query_filtered::<&mut crate::comps_a::Player, With<crate::comps_a::Player>>()
                .iter_mut(&mut world)
                .next()
                .expect("player");
            player.rads = 77;
        }
        if let Some(mut health) = world
            .query_filtered::<&mut Health, With<crate::comps_a::Player>>()
            .iter_mut(&mut world)
            .next()
        {
            health.hp = 3;
        }
        if let Some(mut inv) = world
            .query_filtered::<&mut Inventory, With<crate::comps_a::Player>>()
            .iter_mut(&mut world)
            .next()
        {
            inv.current = 1;
        }
        if let Some(mut race) = world
            .query_filtered::<&mut RaceState, With<crate::comps_a::Player>>()
            .iter_mut(&mut world)
            .next()
        {
            race.race = RaceId::Rogue;
        }
        world
            .resource_mut::<crate::state::menus::MenuState>()
            .unlock_queue
            .push(UnlockPopup::Race(RaceId::Fish));
        save_run_in(&mut world, &dir).expect("write");
        let on_disk = load_run_in(&dir).expect("load");

        // A fresh world standing in for the loadout-built player.
        let mut fresh = World::new();
        fresh.init_resource::<Run>();
        fresh.init_resource::<crate::state::menus::MenuState>();
        fresh.spawn(bundle(RaceId::Chicken));
        apply_run(&mut fresh, &on_disk);

        let mut q = fresh.query::<(&crate::comps_a::Player, &Health, &RaceState)>();
        let (player, health, race) = q.iter(&fresh).next().expect("player");
        assert_eq!(player.rads, 77);
        assert_eq!(health.hp, 3);
        assert_eq!(race.race, RaceId::Rogue);
        assert_eq!(
            fresh
                .resource::<crate::state::menus::MenuState>()
                .unlock_queue
                .len(),
            1
        );
        // `Run` is owned by `setup_run_inner` (worldgen reads and writes it),
        // so `apply_run` must not clobber a run the generator already touched.
        assert_eq!(fresh.resource::<Run>().floor, 1);
    }

    /// GML `MakeGame/Other_10:2-4`: a bad save is renamed out of the way, not
    /// deleted, so the next boot takes the clean path.
    #[test]
    fn quarantine_renames_and_clears() {
        let dir = scratch("quarantine");
        store_in(&dir).write(b"not ron at all").expect("write");
        assert!(dir.join(run_save_file_name()).is_file());
        quarantine_run_save_in(&dir);
        assert!(
            !dir.join(run_save_file_name()).exists(),
            "the bad save leaves the load path"
        );
        assert!(dir.join(RUN_SAVE_QUARANTINE).is_file());
    }
}
