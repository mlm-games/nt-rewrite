//! IDPD raid director: the popo portal, its wave roll and the van lift.
//! GML splits those across `objects/IDPDSpawn/*` (`Create_0.gml:1-36`
//! places the portal and stamps `elite`, `Alarm_1.gml:3-48` rolls the
//! wave), `objects/Van/*` and `objects/VanSpawn/Create_0.gml`; portals
//! also come from the chest (`IDPDChest/Destroy_0.gml:11-20`) and from
//! worldgen. A body is a [`Pos`] (`Vec2`).
//!
//! Render split: portal bursts route through [`crate::effects::spawn_burst`]
//! and trauma through `repame_fx::Trauma`. The warning sting is the
//! `VanSpawn/Create_0.gml:41` cue (`sndOasisPopo` under water, else
//! `sndVanWarning`) at 1.0/0.0; the portal whoosh goes through
//! [`GameAudio::play_portal`].
//!
//! Spawn path: waves call [`crate::enemies::spawn_enemy_at`] (base bundle +
//! brains + table stats), so no second spawn pipeline. Timing is [`GTimer`]
//! driven off [`SimTime`]; the [`LoopTransition`] gating matches
//! `loop_transition.rs` exactly (`blocks_new_idpd_raids`, `throne_ii_alive`,
//! `loop_ready`, `campfire_active`).

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::audio::AudioCue;
use crate::combat::{queue_enemy_spawn, queue_enemy_spawn_birth};
use crate::comps_a::{
    FloorMask, GameCleanup, Health, HeavyHeart, Inventory, LevelCleanup, Player, Run, ScarierFace,
};
use crate::comps_b::{
    Enemy, GmlImage, IdpdShieldUnit, NativeMotion, NativeWallMotion, PickupLifetime, PortalClear,
};
use crate::data::{AreaId, EnemyKind};
use crate::decide_wep::WeaponDropsRng;
use crate::enemies::{EnemySpawnContext, spawn_enemy_at};
use crate::msg::Queue;
use crate::spatial::Pos;
use crate::time::{GTimer, TimerMode};

/// GML `IDPDSpawn/Create_0` elite law: 1-in-5 once loops are deep
/// enough (`loops > 1` on area 0 = campfire, any loop past it).
pub fn idpd_elite_roll(loop_count: u32, area: AreaId) -> bool {
    let eligible = (loop_count > 1 && area == AreaId::Campfire)
        || (loop_count > 0 && area != AreaId::Campfire);
    eligible && rand::rng().random::<f32>() < 0.2
}

/// GML `IDPDSpawn/Alarm_1` spawn table: a lone `PopoFreak` on deep loops
/// (`loops - (area == 0) >= 3`), else the popolevel-gated dir roll
/// (`rng_choose(1, 1, 2, 3)`, dir 3 needs popolevel 3+, dir 2 needs 5+; loop 0
/// with LilHunter alive forces dir 1). Elites swap the whole dir pick for their
/// elite kind.
pub fn roll_idpd_table(
    loop_count: u32,
    area: AreaId,
    popolevel: f32,
    lil_hunter_alive: bool,
) -> Vec<EnemyKind> {
    if loop_count.saturating_sub(if area == AreaId::Campfire { 1 } else { 0 }) >= 3 {
        return vec![EnemyKind::PopoFreak];
    }
    let mut rng = rand::rng();
    let dir = roll_idpd_dir(
        &mut || rng.random::<f32>(),
        popolevel,
        loop_count == 0 && lil_hunter_alive,
    );
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

/// GML `IDPDSpawn/Alarm_1.gml:10-14`: `rng_choose(1, 1, 2, 3)` re-rolled
/// until the `popolevel` gates pass (dir 3 needs 3+, dir 2 needs 5+), then
/// a live `LilHunter` on loop 0 forces dir 1. The float source is a
/// parameter so the global stream and the `RNGStates` LCG share this law.
pub fn roll_idpd_dir(
    next_float: &mut dyn FnMut() -> f32,
    popolevel: f32,
    force_grunts: bool,
) -> u8 {
    if force_grunts {
        return 1;
    }
    loop {
        let d = match next_float() {
            x if x < 0.25 => 1,
            x if x < 0.5 => 1,
            x if x < 0.75 => 2,
            _ => 3,
        };
        if !(d == 3 && popolevel < 3.0) && !(d == 2 && popolevel < 5.0) {
            return d;
        }
    }
}

/// GML `objects/IDPDSpawn/Create_0.gml` - the popo portal, distinct from the
/// `Portal` object. `alarm[0]` fires at `40 + instance_number * 3` frames, arms
/// `alarm[1]` 12 frames later, and `alarm[1]` raises the wave; `Other_7` destroys
/// the instance when the close strip ends. The elite flag is stamped once in
/// `Create_0`, so it is fixed before the `dir` roll in `Alarm_1`.
#[derive(Component, Clone, Copy, Debug)]
pub struct IdpdSpawnPortal {
    /// `Create_0.gml:30-34`: 1-in-5 once the loops/area gate passes.
    pub elite: bool,
    /// `Create_0.gml:5-24` relocates on the first sim tick, not at
    /// spawn: the raise sites (chest / portal / worldgen) have no
    /// player query.
    pub placed: bool,
    /// `Create_0.gml:28`: `40 + instance_number(IDPDSpawn) * 3` frames.
    pub alarm0: f32,
    /// `Alarm_0.gml:1` sets `alarm[1] = 12` frames; 0 means unarmed.
    pub alarm1: f32,
    /// Frames left on `sprIDPDPortalClose` (14 frames at
    /// `image_speed = 0.4`) before `Other_7.gml:3` despawns.
    pub close: f32,
}

/// GML `IDPDSpawn/Create_0.gml:28` and `Alarm_0.gml:1`: the open window
/// is `40 + instance_number(IDPDSpawn) * 3 + 12` frames, so N live
/// portals raised together open at 55, 58, 61, 64, 67, 70.
pub const IDPD_SPAWN_OPEN_BASE: f32 = 52.0;
pub const IDPD_SPAWN_OPEN_PER_LIVE: f32 = 3.0;

/// GML `objects/IDPDSpawn/Create_0.gml`. `instance_number(IDPDSpawn)`
/// counts every live portal, so the caller passes the live count.
/// Returns the portal and the `elite` flag `Create_0.gml:30-34` stamped
/// (the caller plays the matching spawn sting).
pub fn spawn_idpd_spawn(
    commands: &mut Commands,
    run: &mut Run,
    live_portals: u32,
    pos: glam::Vec2,
) -> (Entity, bool) {
    let elite = idpd_elite_roll(run.loop_count, run.area);
    // `Create_0.gml:1`
    run.popolevel += 1.0;
    let alarm0 = IDPD_SPAWN_OPEN_BASE + (live_portals as f32 + 1.0) * IDPD_SPAWN_OPEN_PER_LIVE;
    let entity = commands
        .spawn((
            GameCleanup,
            LevelCleanup,
            IdpdSpawnPortal {
                elite,
                placed: false,
                alarm0: alarm0 - 12.0,
                alarm1: 0.0,
                close: 0.0,
            },
            Pos(pos),
        ))
        .id();
    (entity, elite)
}

/// GML `objects/Van/Alarm_1.gml` deploy law: the one-shot that empties
/// the van. `drive = 0` is the drive-side half (see `BLOCKED`); this
/// marker is the firing bit, consumed once and never re-armed.
#[derive(Component, Clone, Copy, Debug)]
pub struct IdpdVanDeployed;

/// GML `objects/Van` deploy bookkeeping. `right` is stamped at spawn
/// (`Van/Create_0.gml:14-23`) because `Van/Alarm_1` places its payload at
/// `x
/// - 55 * right` / `x
/// - 50 * right`; `frames` counts `alarm[0] = 40` (`Van/Create_0.gml:28`)
///   plus `Van/Alarm_0.gml:3`'s `alarm[1] = 10` down to the one-shot
///   deploy, and `inert` re-arms once `Alarm_2` (15) + `Alarm_3` (20) have
///   parked the van for good.
#[derive(Component, Clone, Copy, Debug)]
pub struct IdpdVanDeploy {
    pub right: f32,
    pub frames: f32,
    /// GML `Van/Create_0.gml:32`:
    /// `(loops > 2) && ((area != 0) || (loops > 3))`.
    pub freak: bool,
    pub inert: f32,
}

/// GML `Van/Create_0.gml:28` + `Van/Alarm_0.gml:3`: 40 + 10 frames.
pub const VAN_DEPLOY_FRAMES: f32 = 50.0;
/// GML `Van/Alarm_2.gml:2` + `Van/Alarm_3.gml:2`: 15 + 20 frames.
pub const VAN_INERT_FRAMES: f32 = 35.0;

/// GML `objects/IDPDChest/Destroy_0.gml` verbatim:
///
/// ```gml
/// if instance_exists(GenCont) exit
/// with instance_create(x, y, ChestOpen) sprite_index = sprIDPDChestOpen
/// instance_create(x, y, FXChestOpen)
/// repeat 6 { with instance_create(x, y, IDPDSpawn) { ... } }
/// ```
///
/// `Destroy_0.gml:11-20`: `repeat 6 instance_create(x, y, IDPDSpawn)`, so this
/// raises six PORTALS, never six grunts. Each runs its own `IDPDSpawn/Create_0`
/// and bumps `GameCont.popolevel` (which gates the Shielders / Inspectors /
/// Elites / PopoFreak table in `Alarm_1`), schedules its wave 52+ frames out,
/// stamps its own elite flag and plays its own spawn sting.
/// `instance_number(IDPDSpawn)` is 1-based and counts every live portal, so a
/// batch staggers 55, 58, 61, 64, 67, 70; `live_portals` is the current
/// `IdpdSpawnPortal` count, keeping a raid raised alongside existing portals in
/// the same stagger GML's `instance_number` produces. Returns the elite count.
fn idpd_portals(
    commands: &mut Commands,
    run: &mut Run,
    cues: &mut Queue<AudioCue>,
    live_portals: u32,
    pos: glam::Vec2,
) -> u32 {
    let mut elites = 0;
    for live in 0..6u32 {
        let (_, elite) = spawn_idpd_spawn(commands, run, live_portals + live, pos);
        if elite {
            elites += 1;
        }
        // GML `IDPDSpawn/Create_0.gml:35-36`
        cues.push(AudioCue {
            name: if elite {
                "sndEliteIDPDPortalSpawn"
            } else {
                "sndIDPDPortalSpawn"
            },
            volume: 1.0,
            variance: 0.0,
        });
    }
    elites
}

/// The `Destroy_0` portal half alone. GML's
/// `IDPDChest/Collision_Player` runs `instance_destroy()` at the end, so
/// a chest opened by touch has already swapped itself to `OpenedChest`
/// and popped `FXChestOpen` before reaching this.
pub fn raise_idpd_portals(
    commands: &mut Commands,
    run: &mut Run,
    cues: &mut Queue<AudioCue>,
    live_portals: u32,
    pos: glam::Vec2,
) {
    idpd_portals(commands, run, cues, live_portals, pos);
}

/// GML `IDPDSpawn/Create_0.gml:6-22` (the `do..until` relocation)
/// verbatim: draw an angle and a `96 + rng(96)` radius off
/// `RNGStates.Popo`, snap the result onto the nearest `Floor` tile, and
/// retry until the site is both `place_free` and further than 64 px.
/// The retry loop is unbounded in GML; 64 draws is the port's hang guard.
pub fn idpd_spawn_site(
    rng: &mut WeaponDropsRng,
    player_pos: glam::Vec2,
    mask: &FloorMask,
) -> glam::Vec2 {
    let mut p = player_pos;
    for _ in 0..64 {
        let ang = rng.float(360.0);
        let dist = 96.0 + rng.float(96.0);
        p = player_pos + glam::Vec2::from_angle(ang.to_radians()) * dist;
        // GML `var dir = instance_nearest(x, y, Floor); if dir { x = dir.x
        // + 16; y = dir.y + 16 }` - snap to the nearest floor tile centre.
        if let Some(tile) = nearest_floor_tile(p, mask) {
            p = tile;
        }
        if p.distance(player_pos) > 64.0 && mask.is_walkable(p) {
            break;
        }
    }
    p
}

/// GML `instance_nearest(x, y, Floor)` plus the `+ 16` centre offset.
fn nearest_floor_tile(p: glam::Vec2, mask: &FloorMask) -> Option<glam::Vec2> {
    mask.cells
        .iter()
        .map(|c| mask.cell_center(*c))
        .min_by(|a, b| {
            a.distance_squared(p)
                .partial_cmp(&b.distance_squared(p))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// GML `IDPDSpawn/Alarm_1.gml:18-46`. Every child is born at
/// `x + random(4) - 2, y + random(4) - 2`; the non-elite ones only then
/// get `motion_add(point_direction(x, y, player.x, player.y) + random(90) - 45, 4)`
/// - a 4 px/frame charge at the player, +/-45 degrees of jitter. The
/// elite branch (`:27`, `:37`, `:47`) spawns with no `motion_add` at all.
fn spawn_idpd_child(
    commands: &mut Commands,
    kind: EnemyKind,
    at: glam::Vec2,
    player_pos: glam::Vec2,
    loops: u32,
    charge: bool,
) {
    let mut rng = rand::rng();
    let at = at
        + glam::Vec2::new(
            rng.random_range(0.0..4.0) - 2.0,
            rng.random_range(0.0..4.0) - 2.0,
        );
    let velocity = if charge {
        let base = (player_pos - at).y.atan2((player_pos - at).x);
        let ang = base + (rng.random_range(0.0f32..90.0) - 45.0).to_radians();
        Some(glam::Vec2::from_angle(ang) * (4.0 * crate::SIM_HZ as f32))
    } else {
        None
    };
    queue_enemy_spawn_birth(commands, kind, at, 1.0, loops, true, velocity, false);
}

/// GML `IDPDSpawn` alarm chain. `alarm[0]` carries the
/// `40 + instance_number(IDPDSpawn) * 3` delay, `Alarm_0` arms
/// `alarm[1] = 12`, and `Alarm_1` raises a `PortalClear` plus the
/// `popolevel`-gated wave. `Other_7` despawns the instance once the
/// close strip has run, so a portal outlives its own wave.
pub fn tick_idpd_spawns(
    commands: &mut Commands,
    dt: f32,
    run: &Run,
    mask: &FloorMask,
    player_pos: glam::Vec2,
    lil_hunter_alive: bool,
    portals: &mut Query<
        (Entity, &mut Pos, &mut IdpdSpawnPortal),
        (With<IdpdSpawnPortal>, Without<Player>, Without<Enemy>),
    >,
) {
    let steps = (dt * crate::SIM_HZ as f32).round();
    // GML `Create_0.gml:8,13` draw the relocation off `RNGStates.Popo`;
    // `Alarm_1.gml:11` draws the dir off the same stream. The port keeps
    // one local `Popo` stream per tick, seeded off the run.
    let mut popo = WeaponDropsRng::new((run.gen_seed as i32) ^ 0x50_50);
    let mut fired: Vec<(glam::Vec2, bool)> = Vec::new();
    for (entity, mut pos, mut portal) in portals.iter_mut() {
        if !portal.placed {
            // `Create_0.gml:5-24`
            pos.0 = idpd_spawn_site(&mut popo, player_pos, mask);
            portal.placed = true;
        }
        // `IDPDSpawn/Alarm_0.gml:1-3` and `IDPDSpawn/Alarm_1` run back to
        // back: the close strip is 14 frames at `image_speed = 0.4` (35
        // steps) while the wave lands 12 steps in, so both counters advance
        // together.
        if portal.close > 0.0 {
            portal.close -= steps;
            if portal.close <= 0.0 {
                // GML `IDPDSpawn/Other_7.gml:3`
                commands.entity(entity).despawn();
                continue;
            }
        }
        if portal.alarm1 > 0.0 {
            portal.alarm1 -= steps;
            if portal.alarm1 <= 0.0 {
                fired.push((pos.0, portal.elite));
            }
            continue;
        }
        portal.alarm0 -= steps;
        // GML `IDPDSpawn/Step_0.gml:4-9`: while the portal wears the charge
        // strip it sheds motes that converge on the player, each living for
        // its own travel time. The strip covers exactly the `alarm0` window
        // (`IDPDSpawn/Create_0.gml:28` arms it and
        // `IDPDSpawn/Alarm_0.gml:2` swaps the art on the same frame it
        // opens the close strip).
        if portal.alarm0 > 0.0 && portal.close <= 0.0 && portal.alarm1 <= 0.0 {
            let at = pos.0 + glam::Vec2::new(popo.float(96.0) - 48.0, popo.float(96.0) - 48.0);
            let to_player = player_pos - at;
            let speed = 2.0 + popo.next_float();
            let mut image = GmlImage::new("images/sprIDPDPortalCharge.png", 4, 0.0);
            // GML `IDPDPortalCharge/Create_0.gml:1-2`: `image_index = random(4)`,
            // `image_speed = 0` (a frozen frame, not a loop).
            image.phase = popo.float(4.0);
            commands.spawn((
                GameCleanup,
                LevelCleanup,
                Pos(at),
                NativeMotion {
                    velocity: to_player.normalize_or_zero() * speed * 30.0,
                    friction: 0.0,
                    radius: 6.0,
                    wall: NativeWallMotion::Stop,
                    tick: 0,
                },
                // GML `Step_0.gml:7`: `alarm[0] = point_distance / speed + 1`.
                PickupLifetime {
                    timer: GTimer::from_seconds(
                        to_player.length() / speed / 30.0 + 1.0 / 30.0,
                        TimerMode::Once,
                    ),
                },
                image,
            ));
        }
        if portal.alarm0 <= 0.0 {
            portal.alarm1 = 12.0;
            portal.close = 35.0;
        }
    }
    if fired.is_empty() {
        return;
    }
    let mut rng = rand::rng();
    for (at, elite) in fired {
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            PortalClear {
                timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
                scale: 1.0,
            },
            Pos(at),
        ));
        // GML `IDPDSpawn/Alarm_1.gml:3-6`: deep loops skip the dir table
        // entirely.
        if run
            .loop_count
            .saturating_sub(u32::from(run.area == AreaId::Campfire))
            >= 3
        {
            let jitter = glam::Vec2::new(
                rng.random_range(0.0..2.0) - 1.0,
                rng.random_range(0.0..2.0) - 1.0,
            );
            queue_enemy_spawn_birth(
                commands,
                EnemyKind::PopoFreak,
                at + jitter,
                1.0,
                run.loop_count,
                true,
                None,
                false,
            );
            continue;
        }
        let dir = roll_idpd_dir(
            &mut || popo.next_float(),
            run.popolevel,
            run.loop_count == 0 && lil_hunter_alive,
        );
        match dir {
            2 => spawn_idpd_child(
                commands,
                if elite {
                    EnemyKind::EliteShielder
                } else {
                    EnemyKind::IdpdShield
                },
                at,
                player_pos,
                run.loop_count,
                !elite,
            ),
            3 => spawn_idpd_child(
                commands,
                if elite {
                    EnemyKind::EliteInspector
                } else {
                    EnemyKind::IdpdInspector
                },
                at,
                player_pos,
                run.loop_count,
                !elite,
            ),
            _ => {
                if elite {
                    spawn_idpd_child(
                        commands,
                        EnemyKind::IdpdElite,
                        at,
                        player_pos,
                        run.loop_count,
                        false,
                    );
                } else {
                    for _ in 0..2 {
                        spawn_idpd_child(
                            commands,
                            EnemyKind::IdpdGrunt,
                            at,
                            player_pos,
                            run.loop_count,
                            true,
                        );
                    }
                }
            }
        }
    }
}

fn enemy_spawn_context(run: &Run, scarier_face: bool, heavy_heart: bool) -> EnemySpawnContext {
    EnemySpawnContext {
        subarea: run.floor_in_area,
        blood_crown: run.blood_crown,
        scarier_face,
        heavy_heart,
    }
}

fn spawn_at(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: EnemyKind,
    pos: glam::Vec2,
    difficulty: f32,
    loops: u32,
    context: EnemySpawnContext,
) -> Entity {
    spawn_enemy_at(
        commands, catalog, kind, pos, difficulty, false, false, loops, context,
    )
}

/// The `tick_idpd_spawns` wrapper the schedule uses. GML runs every
/// `IDPDSpawn`'s step inside the room's own step order; the port keeps it as
/// its own system next to the van tick.
pub fn tick_idpd_spawn_world(
    time: Res<SimTime>,
    mut commands: Commands,
    run: Res<Run>,
    mask: Res<FloorMask>,
    player_q: Query<&Pos, (With<Player>, Without<Enemy>)>,
    enemies_q: Query<&Enemy, With<Enemy>>,
    mut spawn_q: Query<
        (Entity, &mut Pos, &mut IdpdSpawnPortal),
        (With<IdpdSpawnPortal>, Without<Player>, Without<Enemy>),
    >,
) {
    let Ok(player_pos) = player_q.single().map(|p| p.0) else {
        return;
    };
    let lil_hunter_alive = enemies_q
        .iter()
        .any(|e| matches!(e.kind, EnemyKind::LilHunter));
    tick_idpd_spawns(
        &mut commands,
        time.delta_secs,
        &run,
        &mask,
        player_pos,
        lil_hunter_alive,
        &mut spawn_q,
    );
}

/// GML `objects/VanSpawn/Create_0.gml:8-34`: a floor tile `96 + random(24)`
/// to one side of a random player (sign per attempt), +-60 vertically,
/// rejected while it is off floor or within 8 px of another spawn site.
pub fn van_spawn_site(
    rng: &mut rand::rngs::ThreadRng,
    player_pos: glam::Vec2,
    mask: &FloorMask,
) -> glam::Vec2 {
    let mut site = player_pos;
    for _ in 0..250 {
        let flip = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        let spot = player_pos
            + glam::Vec2::new(
                rng.random_range(96.0..120.0) * flip,
                rng.random_range(-60.0..60.0),
            );
        site = mask.cell_center(mask.world_to_cell(spot));
        if !mask.is_walkable(site) {
            continue;
        }
        if site.distance(player_pos) > 96.0 {
            break;
        }
    }
    site
}

pub(crate) fn spawn_van(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    player_pos: glam::Vec2,
    loop_count: u32,
    area: AreaId,
    difficulty: f32,
    loops: u32,
    context: EnemySpawnContext,
) -> Entity {
    // GML `Van/Create_0.gml:14-23`: `right = choose(1, -1)`, overridden to
    // face the player when one exists. `Alarm_1` places its payload at
    // `x - 55 * right`, so the facing has to be latched at spawn.
    let right = if player_pos.x < pos.x { -1.0 } else { 1.0 };
    let freak = loop_count > 2 && (area != AreaId::Campfire || loop_count > 3);
    let e = spawn_at(
        commands,
        catalog,
        EnemyKind::IdpdVan,
        pos,
        difficulty,
        loops,
        context,
    );
    commands.entity(e).insert(IdpdVanDeploy {
        right,
        frames: VAN_DEPLOY_FRAMES,
        freak,
        inert: 0.0,
    });
    e
}

/// GML `objects/Portal/Create_0.gml:16-20`: a Rogue in the run makes every
/// exit portal raise two `IDPDSpawn`s and spend 1.5 `popolevel`, so a Rogue
/// pays for the reinforcements it just triggered.
pub fn rogue_portal_paidown(world: &mut World, pos: glam::Vec2) {
    let mut race_q =
        world.query_filtered::<&crate::comps_a::RaceState, With<crate::comps_a::Player>>();
    if !race_q
        .iter(world)
        .any(|r| r.race == crate::data::RaceId::Rogue)
    {
        return;
    }
    world.resource_mut::<Run>().popolevel -= 1.5;
    let mut portal_q = world.query_filtered::<Entity, With<IdpdSpawnPortal>>();
    let mut live = portal_q.iter(world).count() as u32;
    for _ in 0..2 {
        let stagger = live;
        live += 1;
        world.spawn((
            GameCleanup,
            LevelCleanup,
            PendingRoguePortal { at: pos, stagger },
        ));
    }
}

/// One `Portal/Create_0:18` `IDPDSpawn`, deferred a frame so the two Rogue
/// portals share a single `&mut Run` borrow.
#[derive(Component, Clone, Copy, Debug)]
pub struct PendingRoguePortal {
    pub at: glam::Vec2,
    pub stagger: u32,
}

/// Settles the deferred Rogue portal halves from `Portal/Create_0`.
pub fn tick_pending_rogue_portals(
    mut commands: Commands,
    mut run: ResMut<Run>,
    mut q: Query<(Entity, &PendingRoguePortal)>,
) {
    for (entity, pending) in &mut q {
        spawn_idpd_spawn(&mut commands, &mut run, pending.stagger, pending.at);
        commands.entity(entity).despawn();
    }
}

/// GML `scripts/scrOnPopoKill/scrOnPopoKill.gml`: killing any popo unit (or
/// the Captain / Lil Hunter) freezes every popo unit on the floor for 100
/// steps, which is what stops a whole wave from firing during a boss kill.
/// The `WantVan.canspawn` and `UberCont.ctot_uniq` halves have no counterpart
/// in the port (vans are raised directly, and `ctot_uniq` is a daily-run stat).
pub fn freeze_idpd_wave(world: &mut World) {
    let mut q = world.query::<(&Enemy, &mut crate::comps_b::EnemyBrain)>();
    for (enemy, mut brain) in q.iter_mut(world) {
        if matches!(
            enemy.kind,
            EnemyKind::IdpdGrunt
                | EnemyKind::IdpdInspector
                | EnemyKind::IdpdShield
                | EnemyKind::IdpdElite
                | EnemyKind::EliteInspector
                | EnemyKind::EliteShielder
        ) {
            brain.freeze += 100.0;
        }
    }
    // GML `scrOnPopoKill.gml:14` `with (WantVan) canspawn = true`.
    let mut vans = world.query::<&mut crate::comps_b::WantVan>();
    for mut van in vans.iter_mut(world) {
        van.canspawn = true;
    }
}

/// GML `objects/Van/Alarm_1.gml` in full: `drive = 0`, the freak self-destruct,
/// then `repeat 3 + GameCont.loops` grunts at `x - 55 * right, y + orandom(5)`
/// and a 50/50 second wave - `1 + loops` of one `{Inspector, Shielder}` or
/// `loops` of one `{EliteGrunt, EliteInspector, EliteShielder}`, all at
/// `x - 50 * right, y + orandom(5)`. Fires ONCE, 50 frames after spawn
/// (`Create_0.gml:28` `alarm[0] = 40` + `Alarm_0.gml:3` `alarm[1] = 10`), and the
/// van goes inert 35 frames later (`Alarm_2` 15 + `Alarm_3` 20). The `drive` half
/// (`drivespeed` / `wallbreak` / `x += right * drivespeed`) lives in `enemies.rs`.
pub fn tick_idpd_vans(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    run: Res<Run>,
    scarier: Res<ScarierFace>,
    heavy_heart: Res<HeavyHeart>,
    mut trauma: ResMut<Trauma>,
    mut cues: ResMut<Queue<AudioCue>>,
    player_q: Query<(&Pos, &Player, &Inventory, &Health), (With<Player>, Without<Enemy>)>,
    mut vans: Query<(Entity, &Pos, &mut IdpdVanDeploy), (With<Enemy>, Without<IdpdVanDeployed>)>,
) {
    let steps = (time.delta_secs * crate::SIM_HZ as f32).round();
    let context = enemy_spawn_context(&run, scarier.0, heavy_heart.0);
    for (entity, pos, mut van) in vans.iter_mut() {
        if van.inert > 0.0 {
            // GML `Van/Alarm_2.gml:2` then `Van/Alarm_3.gml:2`: `can_hq =
            // 0`, parked on `sprVanDeactivated` for good.
            van.inert -= steps;
            if van.inert <= 0.0 {
                commands.entity(entity).remove::<IdpdShieldUnit>();
                commands.entity(entity).insert(IdpdVanDeployed);
            }
            continue;
        }
        if van.frames > 0.0 {
            van.frames -= steps;
            if van.frames > 0.0 {
                continue;
            }
        }

        // GML `Van/Alarm_1.gml:1-5`
        if van.freak {
            commands.entity(entity).despawn();
            van_destroy(
                &mut commands,
                &catalog,
                &mut trauma,
                &mut cues,
                &run,
                pos.0,
                &player_q,
            );
            continue;
        }

        // GML `Van/Alarm_1.gml:14-15`
        let mut rng = rand::rng();
        let back = pos.0 + glam::Vec2::new(-55.0 * van.right, 0.0);
        for _ in 0..3 + run.loop_count {
            let y = rng.random_range(-5.0..5.0);
            spawn_at(
                &mut commands,
                &catalog,
                EnemyKind::IdpdGrunt,
                back + glam::Vec2::new(0.0, y),
                1.0,
                run.loop_count,
                context,
            );
        }
        // GML `Alarm_1.gml:17-32`
        let back = pos.0 + glam::Vec2::new(-50.0 * van.right, 0.0);
        if rng.random_bool(0.5) {
            let spwn = if rng.random_bool(0.5) {
                EnemyKind::IdpdInspector
            } else {
                EnemyKind::IdpdShield
            };
            for _ in 0..1 + run.loop_count {
                let y = rng.random_range(-5.0..5.0);
                spawn_at(
                    &mut commands,
                    &catalog,
                    spwn,
                    back + glam::Vec2::new(0.0, y),
                    1.0,
                    run.loop_count,
                    context,
                );
            }
        } else {
            let spwn = match rng.random_range(0..3) {
                0 => EnemyKind::IdpdElite,
                1 => EnemyKind::EliteInspector,
                _ => EnemyKind::EliteShielder,
            };
            for _ in 0..run.loop_count {
                let y = rng.random_range(-5.0..5.0);
                spawn_at(
                    &mut commands,
                    &catalog,
                    spwn,
                    back + glam::Vec2::new(0.0, y),
                    1.0,
                    run.loop_count,
                    context,
                );
            }
        }
        van.inert = VAN_INERT_FRAMES;
    }
}

/// GML `objects/Van/Destroy_0.gml`, the freak-van self-destruct:
/// `scrDrop(100, 0)` x3, three `PopoExplosion`s at
/// `(x + random(40) - 20, y + random(20) - 10)` (8 damage each), seven
/// `BlueFlame`s, and three `PopoFreak`s.
fn van_destroy(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    trauma: &mut Trauma,
    cues: &mut Queue<AudioCue>,
    run: &Run,
    pos: glam::Vec2,
    player_q: &Query<(&Pos, &Player, &Inventory, &Health), (With<Player>, Without<Enemy>)>,
) {
    let Ok((_, player, inv, health)) = player_q.single() else {
        return;
    };
    let mut rng = rand::rng();
    for _ in 0..3 {
        crate::pickups::maybe_spawn_drop_ctx(
            commands,
            catalog,
            pos,
            100,
            0,
            player,
            inv,
            health,
            run.loop_count,
            None,
            &crate::pickups::ChestCtx::default(),
        );
    }
    for _ in 0..3 {
        let at = pos
            + glam::Vec2::new(
                rng.random_range(0.0..40.0) - 20.0,
                rng.random_range(0.0..20.0) - 10.0,
            );
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            crate::combat::Explosion {
                timer: GTimer::from_seconds(0.05, TimerMode::Once),
                radius: 32.0,
                damage: 8,
                team: crate::comps_a::Team::Enemy,
                hits_player: true,
                source: None,
            },
            Pos(at),
        ));
    }
    for _ in 0..3 {
        let at = pos
            + glam::Vec2::new(
                rng.random_range(0.0..16.0) - 8.0,
                rng.random_range(0.0..16.0) - 8.0,
            );
        queue_enemy_spawn(commands, EnemyKind::PopoFreak, at, 1.0, run.loop_count);
    }
    trauma.add(0.3);
    cues.push(AudioCue {
        name: "sndIDPDNadeExplo",
        volume: 1.0,
        variance: 0.1,
    });
}
