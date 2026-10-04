//! Pickup / drop spawning. Ported from nt's `game/pickups.rs` spawn helpers plus
//! the combat drop fns (`spawn_rad_burst`, `maybe_spawn_drop`, `spawn_chest`, …).
//! Render split: only rads carry `SpriteAnim` (bevy parity); static kinds resolve
//! art renderer-side from the same kind -> path table. Juice pop-ins skipped
//! (no sim effect).

use bevy_ecs::prelude::*;
use rand::RngExt;
use repame_fx::Trauma;
use repame_sim::SimTime;

use crate::anim::SpriteAnim;
use crate::audio::{AudioCue, GameAudio};
use crate::combat::queue_enemy_spawn_birth;
use crate::comps_a::{
    FloorMask, GameCleanup, Health, Inventory, LevelCleanup, MAX_WEAPON_SLOTS, Player, RaceState,
    Run, Team, Toast,
};
use crate::comps_b::{
    ChestArt, ChestKind, CursedAmmoBlink, DropSeed, Enemy, FlungWeapon, GmlImage, GroundPhysics,
    NativeDepth, NativeMotion, NativeWallMotion, OpenedChest, Pickup, PickupCurse, PickupKind,
    PickupLifetime, Portal, PortalCarriedWeapons, PortalClear, Prop, PropSprites,
    RadChestContainer, Telekinesis, WepPickupAmmo,
};
use crate::data::{
    AmmoKind, CrownKind, EnemyKind, MutationId, RaceId, UltraMutationId, WeaponId,
    ammo_pickup_amount_for,
};
use crate::effects::{ChromaticAberration, FlashWhite, chromatic_pulse};
use crate::idpd::{IdpdVanDeploy, IdpdVanDeployed, VAN_DEPLOY_FRAMES};
use crate::input::NtInput;
use crate::msg::Queue;
use crate::progression::{check_level_up, try_recharge_strong_spirit};
use crate::spatial::Pos;
use crate::time::{GTimer, TimerMode};
use crate::weapon_runtime::{weapon_ammo, weapon_id_name, weapon_runtime_def};

#[derive(Component, Clone, Copy)]
pub enum ProtoChestState {
    Pending,
    Armed { weapon: WeaponId, cursed: bool },
    Carried { weapon: WeaponId, cursed: bool },
}

impl ProtoChestState {
    fn pending() -> Self {
        Self::Pending
    }
}

fn arm_proto_chests(
    commands: &mut Commands,
    proto_q: &mut Query<(Entity, &mut ProtoChestState)>,
    run: &Run,
) {
    let carried = proto_q
        .iter()
        .filter_map(|(_, state)| match &*state {
            ProtoChestState::Carried { weapon, cursed } => Some((*weapon, *cursed)),
            _ => None,
        })
        .last();
    let mut carriers = Vec::new();
    let mut armed = false;
    for (entity, mut state) in proto_q.iter_mut() {
        match &*state {
            ProtoChestState::Pending => {
                let (weapon, cursed) = carried.unwrap_or((run.protowep, run.protocurse));
                *state = ProtoChestState::Armed { weapon, cursed };
                armed = true;
            }
            ProtoChestState::Carried { .. } => carriers.push(entity),
            ProtoChestState::Armed { .. } => {}
        }
    }
    if armed {
        for entity in carriers {
            commands.entity(entity).despawn();
        }
    }
}

pub(crate) fn spawn_proto_weapon(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    weapon: WeaponId,
    cursed: bool,
    pos: glam::Vec2,
) -> Entity {
    let entity = spawn_pickup(commands, catalog, PickupKind::Weapon(weapon), pos, 0, false);
    // GML `ProtoChest/Collision_Player.gml:7` is
    // `scrWeaponPickupCreate(x, y, wep)` with `_has_ammo` defaulting to
    // false, so the vault prototype comes out empty.
    commands.entity(entity).insert(WepPickupAmmo(false));
    if cursed {
        commands.entity(entity).insert(PickupCurse);
    }
    entity
}

fn pickup_sprite(kind: PickupKind) -> Option<(&'static str, f32)> {
    Some(match kind {
        PickupKind::Rad(_) => ("images/sprRad.png", 12.0),
        PickupKind::Medkit(_) => ("images/sprHP.png", 16.0),
        PickupKind::Ammo(..) => ("images/sprAmmo.png", 12.0),
        PickupKind::CursedAmmo => ("images/sprCursedAmmo.png", 12.0),
        PickupKind::Curse => ("images/sprCurse.png", 10.0),
        PickupKind::Weapon(_) => return None,
        PickupKind::Chest(kind) => match kind {
            ChestKind::Weapon => ("images/sprWeaponChest.png", 32.0),
            ChestKind::Ammo => ("images/sprAmmoChest.png", 32.0),
            ChestKind::Mystery => ("images/sprAmmoChestMystery.png", 32.0),
            ChestKind::Gold => ("images/sprGoldChest.png", 32.0),
            ChestKind::Rad => ("images/sprRadChest.png", 32.0),
            ChestKind::Health => ("images/sprHealthChest.png", 32.0),
            ChestKind::CursedBig => ("images/sprCursedChestBig.png", 32.0),
            ChestKind::Rogue => ("images/sprRogueAmmoChest.png", 32.0),
            ChestKind::Proto => ("images/sprProtoChest.png", 32.0),
            ChestKind::BigWeapon => ("images/sprWeaponChestBig.png", 32.0),
            ChestKind::RadBig => ("images/sprRadChestBig.png", 32.0),
            ChestKind::RadMaggot => ("images/sprRadChestMaggot.png", 32.0),
            ChestKind::Idpd => ("images/sprIDPDChest.png", 32.0),
        },
    })
}

// GML mask half-extents (each object's `spriteMaskId` bbox; the reference
// resolves pickups as per-axis box overlap, never as a distance): `mskWepPickup`
// 28x28 origin 14, `mskPickup` 10x10 origin 5, `mskRad` 8x8 origin 4; the
// chestprop sprites carry no mask, so their 16x16 origin-8 bbox is the box.
// `PLAYER_MASK_HALF` takes the `mskPlayer` 16x16 FRAME (origin 8), not its
// `bbox_*` 4..11 x 4..13.
// The prop prompts instead keep GML `place_meeting(x, y, _player)` verbatim as
// a closed per-axis window of `player - prop`, from the real `bbox_*` + origin
// (both edge pixels inclusive): the player's box is `mskPlayer` bbox 4..11 x
// 4..13 at origin (8,8) - 8x10 inside the 16x16 frame, never mirrored (`Player`
// draws through `draw_sprite_ext(.., right, ..)`, it never writes
// `image_xscale`) - against `CarVenusFixed` 0..31 x 3..30 at origin (16,16)
// (both car sprites and both hurt strips share that box; `image_xscale =
// choose(1, -1)` in `Create_0` mirrors it, carried by `PropSprites.flip_x`),
// `IceFlower` `sprIceFlowerIdle` 1..30 x 2..29 at origin (16,16) (pinned
// `image_xscale = 1`), and `Van` `mskVan` 27..100 x 42..85 at origin (64,64)
// (`Van` never writes `image_xscale`).
const PLAYER_MASK_HALF: f32 = 8.0;
const WEP_PICKUP_MASK_HALF: f32 = 14.0;
const PICKUP_MASK_HALF: f32 = 5.0;
const RAD_MASK_HALF: f32 = 4.0;
const CHEST_SPRITE_HALF: f32 = 8.0;

pub struct MaskSpan {
    pub x: (f32, f32),
    pub y: (f32, f32),
}

const PLAYER_BOX: (f32, f32, f32, f32) = (-4.0, 3.0, -4.0, 5.0);

const fn mask_span(bbox: (f32, f32, f32, f32)) -> MaskSpan {
    MaskSpan {
        x: (bbox.0 - PLAYER_BOX.1, bbox.1 - PLAYER_BOX.0),
        y: (bbox.2 - PLAYER_BOX.3, bbox.3 - PLAYER_BOX.2),
    }
}

const CAR_PROMPT_SPAN: MaskSpan = mask_span((-16.0, 15.0, -13.0, 14.0));
const CAR_PROMPT_SPAN_FLIPPED: MaskSpan = mask_span((-15.0, 16.0, -13.0, 14.0));
const ICE_FLOWER_PROMPT_SPAN: MaskSpan = mask_span((-15.0, 14.0, -14.0, 13.0));
const VAN_PROMPT_SPAN: MaskSpan = mask_span((-37.0, 36.0, -22.0, 21.0));

/// GML `place_meeting`: a `Collision_Player` event fires when the two
/// masks overlap on BOTH axes. `half_sum` is the two half-extents
/// added, so the test is `|dx| < half_sum && |dy| < half_sum`.
pub fn mask_overlap(a: glam::Vec2, b: glam::Vec2, half_sum: f32) -> bool {
    (a.x - b.x).abs() < half_sum && (a.y - b.y).abs() < half_sum
}

/// Half-extent sum of the player's mask and a pickup's own box, `None` when
/// the pickup cannot react at all.
/// GML `Player/Collision_WepPickup.gml:6` gates the equip on
/// `other.id == instance_nearest(x, y, WepPickup)`, so only the NEAREST ground
/// weapon reacts to a press - but the ammo payout at `:90-104` sits OUTSIDE
/// that `if` and needs only mask overlap (see [`weapon_pickup_ammo_pays`]).
pub fn pickup_mask_half(kind: &PickupKind, is_nearest_weapon: bool) -> Option<f32> {
    Some(match kind {
        PickupKind::Weapon(_) if !is_nearest_weapon => return None,
        PickupKind::Weapon(_) => PLAYER_MASK_HALF + WEP_PICKUP_MASK_HALF,
        PickupKind::Ammo(..)
        | PickupKind::CursedAmmo
        | PickupKind::Medkit(_)
        | PickupKind::Curse => PLAYER_MASK_HALF + PICKUP_MASK_HALF,
        PickupKind::Rad(_) => PLAYER_MASK_HALF + RAD_MASK_HALF,
        PickupKind::Chest(_) => PLAYER_MASK_HALF + CHEST_SPRITE_HALF,
    })
}

/// Half-extent sum for the weapon-pickup ammo payout, which GML lets
/// fire on any overlapping `WepPickup` (nearest or not).
pub const WEP_AMMO_REACH: f32 = PLAYER_MASK_HALF + WEP_PICKUP_MASK_HALF;

/// GML `scrDrawInteractionHUD:342` verbatim: the prompt lights on the NEAREST
/// `WepPickup` whose mask overlaps the player. Rads, ammo, medkits, chests and
/// curse motes are not in `[ WepPickup, CarVenusFixed, IceFlower, Van ]`, so they
/// must never light the act button ("the pickup indicator showing over a rad").
/// The prop half (`scrDrawPlayerHUD.gml:371-376,402`: the `Prompt{Object}` line
/// plus the `active = true` act raise) rides [`WeaponLabel::prop_prompts`].
pub fn nearest_ground_weapon(
    player: glam::Vec2,
    pickups: impl Iterator<Item = (Entity, glam::Vec2, PickupKind)>,
) -> Option<Entity> {
    let mut best: Option<(Entity, f32)> = None;
    for (entity, pos, kind) in pickups {
        if !matches!(kind, PickupKind::Weapon(_)) {
            continue;
        }
        if !mask_overlap(player, pos, PLAYER_MASK_HALF + WEP_PICKUP_MASK_HALF) {
            continue;
        }
        let d = player.distance(pos);
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((entity, d));
        }
    }
    best.map(|(entity, _)| entity)
}

fn nearest_prompt_hit(
    player: glam::Vec2,
    candidates: impl Iterator<Item = (Entity, glam::Vec2, MaskSpan)>,
) -> Option<Entity> {
    let mut best: Option<(Entity, f32)> = None;
    for (entity, pos, span) in candidates {
        let dx = player.x - pos.x;
        let dy = player.y - pos.y;
        if dx < span.x.0 || dx > span.x.1 || dy < span.y.0 || dy > span.y.1 {
            continue;
        }
        let d = player.distance(pos);
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((entity, d));
        }
    }
    best.map(|(entity, _)| entity)
}

/// GML `ButtonAct` fade: `ButtonAct/Other_10` runs the 30 Hz step, and
/// `scrDrawPlayerHUD.gml:378-403` raises `active` while the player stands on a
/// promptable pickup - only inside its `is_touch` block. The port raises it on
/// every device because the sole reader is the touch chrome (`render.rs
/// touch_sprites`), which never draws on keyboard/gamepad, so the raise is
/// unobservable there. The prompt needs no interact press - it shows whether or
/// not the press landed.
pub fn tick_act_button(
    mut act: ResMut<crate::state::ActButton>,
    player_q: Query<&Pos, (With<Player>, Without<Pickup>)>,
    pickups: Query<(Entity, &Pos, &Pickup), Without<Player>>,
) {
    act.step();
    let Ok(player_pos) = player_q.single() else {
        return;
    };
    if nearest_ground_weapon(
        player_pos.0,
        pickups
            .iter()
            .map(|(entity, pos, pickup)| (entity, pos.0, pickup.kind)),
    )
    .is_some()
    {
        act.active = true;
    }
}

pub fn spawn_pickup(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: PickupKind,
    pos: glam::Vec2,
    loops: u32,
    hasted: bool,
) -> Entity {
    let path = pickup_sprite(kind);

    let mut rng = rand::rng();
    let mut ec = commands.spawn((GameCleanup, LevelCleanup, Pickup { kind }, Pos(pos)));
    // Only rads animate (bevy parity). Static kinds carry no render
    // handle here; the render phase maps kind -> art path itself
    // (same table as `pickup_sprite`).
    if matches!(kind, PickupKind::Rad(_))
        && let Some((path, _)) = path
        && let Some(def) = catalog.def(path)
    {
        let mut anim = SpriteAnim::new(path, def);
        anim.timer = GTimer::from_seconds(1.0 / 12.0, TimerMode::Repeating);
        anim.frame = rng.random_range(0..def.frames.max(1));
        ec.insert(anim);
    }
    match kind {
        PickupKind::Rad(_) => {
            // GML `Rad/Create_0.gml:5-9` verbatim:
            // `alarm[0] = ceil((150 + random(30)) / ((4 + loops) / 4))`,
            // then the Haste crown's INTEGER `/= 3`. `Rad/Alarm_0.gml`
            // blinks `blink = 30` times at 2 ticks each before the
            // despawn, so the pickup lives `alarm + 60` steps.
            let mut alarm = ((150.0 + rng.random_range(0.0..30.0)) / ((4.0 + loops as f32) / 4.0))
                .ceil() as u32;
            if hasted {
                alarm /= 3;
            }
            ec.insert(PickupLifetime {
                timer: GTimer::from_seconds((alarm + 60) as f32 / 30.0, TimerMode::Once),
            });
        }
        PickupKind::Medkit(_) | PickupKind::Ammo(..) => {
            let init =
                ((200.0 + rng.random_range(0.0..30.0)) / ((5.0 + loops as f32) / 5.0)).ceil();
            let total_steps = if hasted { init / 3.0 } else { init } + 62.0;
            ec.insert(PickupLifetime {
                timer: GTimer::from_seconds(total_steps / 30.0, TimerMode::Once),
            });
        }
        PickupKind::CursedAmmo => {
            // GML `CursedPickup/Create_0.gml` verbatim: `blink = 30`,
            // `alarm[0] = ceil((200 + random(30)) * mult)` (the Rush
            // crown's INTEGER `/= 3`), `image_speed = 0`. The alarm
            // never fires first - `Alarm_0` re-arms every 2 steps and
            // the detonation is keyed on `blink < 0`.
            let alarm = (200.0_f32 + rng.random_range(0.0_f32..30.0)).ceil();
            let alarm = if hasted { alarm / 3.0 } else { alarm };
            let frames = catalog
                .def("images/sprCursedAmmo.png")
                .map(|d| d.frames.max(1))
                .unwrap_or(1);
            ec.insert((
                GmlImage::new("images/sprCursedAmmo.png", frames, 0.0),
                CursedAmmoBlink {
                    blink: 30,
                    alarm,
                    sounded: false,
                },
            ));
        }
        PickupKind::Weapon(_) => {
            // GML `scrWeaponPickupCreate.gml:6-10`: `_has_ammo` defaults
            // to FALSE; only chests and `scrDrop:85` pass true.
            ec.insert(WepPickupAmmo(false));

            let ang = rng.random_range(0.0..std::f32::consts::TAU);
            ec.insert(GroundPhysics {
                vel: glam::Vec2::new(ang.cos(), ang.sin()) * rng.random_range(15.0..45.0),
                rotspeed: rng.random_range(0.7..1.0)
                    * if rng.random_bool(0.5) { 1.0 } else { -1.0 },
            });
        }
        PickupKind::Chest(_) => {}
        // GML `Curse` motes drift without a lifetime/physics setup.
        PickupKind::Curse => {}
    }
    if matches!(kind, PickupKind::Chest(ChestKind::Proto)) {
        ec.insert(ProtoChestState::pending());
    }
    ec.id()
}

/// GML `scrPlayerCountCursed`: the per-slot `curse` / `bcurse` flags plus
/// every `extra_weps_curse[i]`. The port folds all of them into
/// `Inventory::cursed`.
pub fn count_cursed(inv: &Inventory) -> u32 {
    inv.cursed.iter().filter(|c| **c).count() as u32
}

/// GML `AmmoPickup/Create_0.gml:14-17`: with 2+ cursed weapons carried,
/// `random(2) < 1` replaces the pickup with a `CursedPickup` in place
/// (`instance_destroy(id, false)` - the ammo never exists). `instance_is`/
/// `instance_exists(Player)` gates it, so only roll when a player is
/// present, which every caller already is.
pub fn spawn_ammo_pickup(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    cursed_count: u32,
    pos: glam::Vec2,
    loops: u32,
    hasted: bool,
) -> Entity {
    let kind = if cursed_count >= 2 && rand::rng().random::<f32>() < 0.5 {
        PickupKind::CursedAmmo
    } else {
        PickupKind::Ammo(AmmoKind::None, 0)
    };
    spawn_pickup(commands, catalog, kind, pos, loops, hasted)
}

/// [`spawn_ammo_pickup`] for callers that already resolved the ammo type.
/// GML decides the type in `AmmoPickup/Collision_Player` via
/// `scrAmmoDecideType`, so a `CursedPickup` conversion discards the
/// caller's choice and re-rolls on touch - which is why the converted
/// kind carries no pre-decided type.
pub fn maybe_cursed_ammo(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    cursed_count: u32,
    resolved: PickupKind,
    pos: glam::Vec2,
    loops: u32,
    hasted: bool,
) -> Entity {
    let kind = if cursed_count >= 2 && rand::rng().random::<f32>() < 0.5 {
        PickupKind::CursedAmmo
    } else {
        resolved
    };
    spawn_pickup(commands, catalog, kind, pos, loops, hasted)
}

/// GML `WeaponChest/Create_0.gml:4-11`: the Crown-of-Curses roll, made
/// ONCE at spawn and read back when the chest opens. GML gates it on
/// `GameCont.crown > 1 && instance_exists(GenCont)` - any real crown
/// (the port's `CrownKind::None` is GML `crwn_none`, so the same test)
/// AND worldgen only.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct ChestCurse(pub bool);

/// The rolls a chest makes exactly once, in `Create_0`. Everything GML
/// resolves at spawn (art, the Curses roll, `dropseed`) is decided from
/// this and then frozen on the entity; `Default` is the "not worldgen,
/// no crown" case, so a non-generation caller never curses a chest.
#[derive(Clone, Copy, Debug)]
pub struct ChestCtx {
    /// `instance_exists(GenCont)`.
    pub worldgen: bool,
    pub area: crate::data::AreaId,
    pub crown: CrownKind,
    /// `scr_ultra_get(Race.Steroids, UltraSkill.Ambidextrous)`.
    pub ambidextrous: bool,
    /// `scr_ultra_get(Race.Steroids, UltraSkill.GetLoaded)`.
    pub get_loaded: bool,
    /// `Run.gen_seed`, mixed into the per-chest `dropseed` draw.
    pub gen_seed: u64,
    /// Chest creation index this floor (GML's per-run
    /// `rng_state[WeaponDrops]` walk).
    pub order: u32,
}

impl Default for ChestCtx {
    fn default() -> Self {
        Self {
            worldgen: false,
            area: crate::data::AreaId::Desert,
            crown: CrownKind::None,
            ambidextrous: false,
            get_loaded: false,
            gen_seed: 0,
            order: 0,
        }
    }
}

impl ChestCtx {
    /// GML `GameCont.underwater` is `area == area_oasis`.
    pub fn underwater(&self) -> bool {
        self.area == crate::data::AreaId::Oasis
    }
}

/// GML `rng_next_int` (`scripts/rng_next_int/rng_next_int.gml:5`) with
/// the `scrRngStatesInit` constants: `rng_m 2147483647`, `rng_a
/// 1103515245`, `rng_c 12345`.
pub fn gml_rng_next(state: &mut u32) -> u32 {
    const RNG_M: u64 = 2_147_483_647;
    *state = ((1_103_515_245u64 * *state as u64 + 12_345) % RNG_M) as u32;
    *state
}

/// GML `chestprop/Create_0.gml:10` `dropseed = rng_next_int(RNGStates.
/// WeaponDrops)`. The reference walks one global stream; the port draws
/// from the run's own seed plus a per-floor creation counter so a
/// chest's contents stay fixed for the whole run.
pub fn chest_drop_seed(gen_seed: u64, order: u32) -> u32 {
    let mut state = (gen_seed as u32) ^ 0x5DEE_CE66 ^ order.wrapping_mul(2_654_435_761);
    gml_rng_next(&mut state)
}

/// GML `scrDecideWep.gml:11,57`: `random_set_seed(_seed)` before the roll
/// and `dropseed = irandom(0x7fffffff)` after it, so a chest's three
/// weapons come off a fixed chain instead of the live RNG.
pub struct DropSeedChain {
    seed: u32,
}

impl DropSeedChain {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    /// One `scrDecideWep` call: seed a generator from `dropseed`, roll,
    /// then advance `dropseed` for the next drop.
    pub fn roll<T>(&mut self, roll: impl FnOnce(&mut rand::rngs::StdRng) -> T) -> T {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed as u64);
        let out = roll(&mut rng);
        let mut state = self.seed;
        self.seed = gml_rng_next(&mut state);
        out
    }
}

/// GML per-chest `sprite_index` / `spr_dead`, resolved once in
/// `Create_0` and frozen on the entity as [`ChestArt`].
pub fn chest_art(kind: ChestKind, ctx: &ChestCtx, cursed: bool) -> ChestArt {
    let art = |idle, open| ChestArt { idle, open };
    match kind {
        // GML `WeaponChest/Create_0.gml:6-24`, gated on `object_index ==
        // WeaponChest` so BigWeaponChest keeps its own art.
        ChestKind::Weapon => {
            if ctx.underwater() {
                art("images/sprClamChest.png", "images/sprClamChestOpen.png")
            } else if cursed {
                art("images/sprCursedChest.png", "images/sprCursedChestOpen.png")
            } else if ctx.ambidextrous {
                art(
                    "images/sprWeaponChestSteroidsUltra.png",
                    "images/sprWeaponChestSteroidsUltraOpen.png",
                )
            } else {
                art("images/sprWeaponChest.png", "images/sprWeaponChestOpen.png")
            }
        }
        // GML `AmmoChest/Create_0.gml:16-19`.
        ChestKind::Ammo => {
            if ctx.get_loaded {
                art(
                    "images/sprAmmoChestSteroids.png",
                    "images/sprAmmoChestSteroidsOpen.png",
                )
            } else {
                art("images/sprAmmoChest.png", "images/sprAmmoChestOpen.png")
            }
        }
        // GML `AmmoChestMystery` keeps its own sprites: the open strip is
        // the only chest art that actually animates (5 frames).
        ChestKind::Mystery => art(
            "images/sprAmmoChestMystery.png",
            "images/sprAmmoChestMysteryOpen.png",
        ),
        // GML `GoldChest/Create_0.gml` inherits chestprop's
        // `sprGoldChest` / `spr_dead = sprGoldChestOpen`.
        ChestKind::Gold => art("images/sprGoldChest.png", "images/sprGoldChestOpen.png"),
        // GML `HealthChest/Create_0.gml:12-15`.
        ChestKind::Health => {
            if ctx.area == crate::data::AreaId::PizzaSewers {
                let pizza = if rand::rng().random_bool(0.5) {
                    "images/sprPizzaChest1.png"
                } else {
                    "images/sprPizzaChest2.png"
                };
                art(pizza, "images/sprPizzaChestOpen.png")
            } else {
                art("images/sprHealthChest.png", "images/sprHealthChestOpen.png")
            }
        }
        // GML `CursedBigChest/Create_0.gml:5` sets `sprite_index =
        // sprCursedChestBig`; `Destroy_0.gml:2` opens as
        // `sprWeaponChestBigOpen`.
        ChestKind::CursedBig => art(
            "images/sprCursedChestBig.png",
            "images/sprWeaponChestBigOpen.png",
        ),
        ChestKind::Rogue => art(
            "images/sprRogueAmmoChest.png",
            "images/sprRogueAmmoChestOpen.png",
        ),
        ChestKind::Proto => art("images/sprProtoChest.png", "images/sprProtoChestOpen.png"),
        ChestKind::BigWeapon => art(
            "images/sprWeaponChestBig.png",
            "images/sprWeaponChestBigOpen.png",
        ),
        ChestKind::Rad => art("images/sprRadChest.png", "images/sprRadChestCorpse.png"),
        ChestKind::RadBig => art("images/sprRadChestBig.png", "images/sprRadChestBigDead.png"),
        ChestKind::RadMaggot => art(
            "images/sprRadChestMaggot.png",
            "images/sprRadChestMaggotDead.png",
        ),
        ChestKind::Idpd => art("images/sprIDPDChest.png", "images/sprIDPDChestOpen.png"),
    }
}

pub fn spawn_chest(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: ChestKind,
    pos: glam::Vec2,
) {
    spawn_chest_with(commands, catalog, kind, pos, &ChestCtx::default());
}

/// GML `chestprop/Create_0` plus each subclass's `Create_0`: the built-in
/// advance is OFF (`image_speed = 0`) and the index is driven by hand -
/// `random(0.04)` while `image_index < 1`, then `+0.4` (`chestprop/Step_0.gml:4-7`,
/// the identical `RadChest/Step_1.gml` ramp, and `RogueChest/Step_1.gml`'s
/// `scrFirstFrameAnim(0.4)` whose first-frame jitter is `0.4 * 0.05 = 0.02`).
/// `sprite_index`/`spr_dead`, the Curses roll and `dropseed` are frozen here.
pub fn spawn_chest_with(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    kind: ChestKind,
    pos: glam::Vec2,
    ctx: &ChestCtx,
) {
    let mut rng = rand::rng();
    // GML `WeaponChest/Create_0.gml:7-11` verbatim, with GML
    // `crown > 1` (any real crown) and the `instance_exists(GenCont)`
    // worldgen gate.
    let cursed = ctx.worldgen
        && ctx.crown != CrownKind::None
        && rng.random_range(0.0..7.0)
            <= if ctx.crown == CrownKind::Curses {
                4.0
            } else {
                1.0
            };
    let art = chest_art(kind, ctx, cursed);
    // GML `RadChest/Create_0.gml:15`, `RadChestBig/Create_0.gml:7` and
    // `RadMaggotChest/Create_0.gml:4` set `image_speed = 0`, which only
    // switches off the automatic advance: `RadChest/Step_1.gml:3-8` drives
    // the index by hand on the same ramp as `chestprop/Step_0.gml:4-7`
    // (`random(0.04)` inside frame 0, then `+0.4` per step).
    let first_jitter = if kind == ChestKind::Rogue { 0.02 } else { 0.04 };
    let frames = catalog.def(art.idle).map(|def| def.frames).unwrap_or(1);

    let mut ec = commands.spawn((
        GameCleanup,
        LevelCleanup,
        Pickup {
            kind: PickupKind::Chest(kind),
        },
        ChestCurse(cursed),
        DropSeed(chest_drop_seed(ctx.gen_seed, ctx.order)),
        ChestArt {
            idle: art.idle,
            open: art.open,
        },
        GmlImage::ramped(art.idle, frames, 0.4, first_jitter),
        Pos(pos),
    ));
    if kind == ChestKind::Proto {
        ec.insert(ProtoChestState::pending());
    }
}

/// GML `FXChestOpen`: `image_speed = 0.4` on an 8-frame strip, so `Other_7`
/// (Animation End) fires after `8 / 0.4 = 20` steps. Underwater it also throws
/// `irandom_range(12, 20)` `Bubble`s with `motion_add(random_angle, random(3))`
/// (`Bubble/Create_0.gml` adds its own `motion_add(random_angle, random(2))`,
/// friction 0.02, `random(0.2) + 0.1` image speed).
pub fn spawn_fx_chest_open(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    underwater: bool,
) {
    let flash = "images/sprFXChestOpen.png";
    let frames = catalog.def(flash).map(|def| def.frames).unwrap_or(8);
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        Pos(pos),
        NativeDepth(0.0),
        GmlImage::animated(flash, frames, 0.4, true),
    ));
    if !underwater {
        return;
    }
    let mut rng = rand::rng();
    for _ in 0..rng.random_range(12..=20) {
        let ang = rng.random_range(0.0..std::f32::consts::TAU);
        let speed = rng.random_range(0.0..3.0);
        let image_speed = rng.random_range(0.1..0.3);
        commands.spawn((
            GameCleanup,
            LevelCleanup,
            Pos(pos),
            NativeDepth(0.0),
            NativeMotion {
                velocity: glam::Vec2::from_angle(ang) * speed * 30.0,
                friction: 0.02,
                radius: 4.0,
                wall: NativeWallMotion::Bounce,
                tick: 0,
            },
            GmlImage::animated("images/sprBubble.png", 7, image_speed, true),
        ));
    }
}

pub fn spawn_rad(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    amount: u32,
) {
    spawn_pickup(commands, catalog, PickupKind::Rad(amount), pos, 0, false);
}

pub fn spawn_rad_burst(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    amount: u32,
) {
    // `spawn_prop_death_effect` is the `prop/Destroy_0.gml:12` caller, so
    // `scrRadDrop.gml:10-13` takes its `instance_is(self, prop)` branch:
    // `_direction = random_angle`, `_speed = 16`.
    scr_rad_drop(commands, catalog, pos, amount, 0, false, false, true);
}

/// GML `scrRadDrop` (`scripts/scrRadDrop/scrRadDrop.gml:7-43`) verbatim:
/// `_high = instance_is(self, RadChest) ? 26 : 15`, overspill past `_high`
/// becomes 10-rad `BigRad` lumps, and every rad gets `motion_add(_direction,
/// _speed)` PLUS `motion_add(random_angle, random(_amount * 0.5) + 5)` then
/// `repeat (speed) speed *= 0.9` to settle.
/// `from_prop` picks the caller's own kick: a `prop` caller (every `RadChest`
/// descendant, via `prop/Destroy_0.gml:12`) uses `random_angle` / `16`, any
/// other caller (the Horror death drop) uses its own `direction` / `speed`,
/// which is 0 for a standing Player - so those rads only get the random kick.
pub fn scr_rad_drop(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    mut amount: u32,
    loops: u32,
    hasted: bool,
    from_rad_chest: bool,
    from_prop: bool,
) {
    let mut rng = rand::rng();
    let high = if from_rad_chest { 26 } else { 15 };
    while amount > high {
        amount -= 10;
        spawn_rad_motion(
            commands, catalog, pos, 10, amount, from_prop, loops, hasted, &mut rng,
        );
    }
    for _ in 0..amount {
        spawn_rad_motion(
            commands, catalog, pos, 1, amount, from_prop, loops, hasted, &mut rng,
        );
    }
}

/// One `instance_create(x, y, Rad)` from [`scr_rad_drop`], with the
/// double `motion_add` and the `repeat (speed) speed *= 0.9` settle.
#[allow(clippy::too_many_arguments)]
fn spawn_rad_motion(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    amount: u32,
    pool_left: u32,
    from_prop: bool,
    loops: u32,
    hasted: bool,
    rng: &mut impl rand::RngExt,
) {
    let (dir, base) = if from_prop {
        (rng.random_range(0.0..std::f32::consts::TAU), 16.0)
    } else {
        (0.0, 0.0)
    };
    let kick = rng.random_range(0.0..pool_left as f32 * 0.5) + 5.0;
    let ang = rng.random_range(0.0..std::f32::consts::TAU);
    let mut vel = glam::Vec2::from_angle(dir) * base + glam::Vec2::from_angle(ang) * kick;
    // GML `repeat (speed)` counts the ROUNDED speed, so a 7.3 px/step
    // kick settles to 7.3 * 0.9^7.
    let settle = vel.length().round() as i32;
    vel *= 0.9f32.powi(settle);
    let e = spawn_pickup(
        commands,
        catalog,
        PickupKind::Rad(amount),
        pos,
        loops,
        hasted,
    );
    commands.entity(e).insert(GroundPhysics {
        vel: vel * 30.0,
        rotspeed: rng.random_range(0.0..std::f32::consts::TAU),
    });
}

/// Random unit-ish offset for scatter drops (bevy parity).
pub fn random_offset() -> glam::Vec2 {
    let mut rng = rand::rng();
    let a = rng.random_range(0.0..std::f32::consts::TAU);
    let d = rng.random_range(0.0..22.0);
    glam::Vec2::new(a.cos(), a.sin()) * d
}

/// GML Chicken throw (`scripts/scrPowers/scrPowers.gml:218-242`): drop
/// the held weapon as a flung pickup - speed 16 px/step toward
/// `angle_rad` (gunangle±2), team/creator set, Determination ultra arming
/// the 60-tick `alarm[1]` return.
pub fn spawn_flung_weapon_pickup(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    weapon: WeaponId,
    pos: glam::Vec2,
    angle_rad: f32,
    team: Team,
    creator: Entity,
    return_ticks: u8,
) -> Entity {
    let e = spawn_pickup(commands, catalog, PickupKind::Weapon(weapon), pos, 0, false);
    let dir = glam::Vec2::new(angle_rad.cos(), angle_rad.sin());
    commands.entity(e).insert(GroundPhysics {
        vel: dir * 16.0 * 30.0,
        rotspeed: 0.0,
    });
    commands.entity(e).insert(FlungWeapon {
        team,
        creator,
        return_ticks,
    });
    e
}

/// GML `WepPickup/Alarm_1`: when the Determination return alarm fires,
/// fling the pickup back at its creator (`sndChickenReturn`).
pub fn tick_flung_weapons(
    mut commands: Commands,
    mut cues: ResMut<Queue<AudioCue>>,
    mut q: Query<(Entity, &Pos, &mut FlungWeapon, Option<&mut GroundPhysics>)>,
    creators: Query<&Pos, Without<FlungWeapon>>,
) {
    for (e, pos, mut flung, ground) in &mut q {
        if flung.return_ticks == 0 {
            continue;
        }
        flung.return_ticks -= 1;
        if flung.return_ticks > 0 {
            continue;
        }
        let Ok(cpos) = creators.get(flung.creator) else {
            continue;
        };
        let dir = (cpos.0 - pos.0).normalize_or_zero();
        let vel = dir * 16.0 * 30.0;
        if let Some(mut g) = ground {
            g.vel = vel;
        } else {
            commands
                .entity(e)
                .insert(GroundPhysics { vel, rotspeed: 0.0 });
        }
        cues.push(AudioCue {
            name: "sndChickenReturn",
            volume: 1.0,
            variance: 0.0,
        });
    }
}

/// GML `RadChest/Destroy_0.gml:4-9`: 4 x `Smoke` with
/// `motion_add(random_angle, random(3))`, then one `ExploderExplo`
/// (6 more `Smoke` plus `BackCont.shake += 6`; no `damage`, no
/// `Collision_Player`, so opening a rad cache never hurts the player).
fn rad_chest_burst(commands: &mut Commands, pos: glam::Vec2) {
    let mut rng = rand::rng();
    for _ in 0..4 {
        let ang = rng.random_range(0.0..std::f32::consts::TAU);
        let speed = rng.random_range(0.0..3.0);
        crate::environment::spawn_native_smoke_mote(
            commands,
            true,
            pos,
            glam::Vec2::from_angle(ang),
            speed,
        );
    }
    crate::environment::spawn_exploder_explo(commands, true, pos, glam::Vec2::ZERO, 0.0);
}

/// Cached per-tick gun-decide context (built once in a PreUpdate-ish
/// system so `move_projectiles` stays under bevy's 16-system-param
/// limit; GML `instance_nearest(x, y, Player)` + `GameCont.hard`).
#[derive(Resource, Clone)]
pub struct GunDecideCache {
    pub ctx: Option<crate::decide_wep::DecideCtx>,
    pub particles: bool,
}

impl Default for GunDecideCache {
    fn default() -> Self {
        Self {
            ctx: None,
            particles: true,
        }
    }
}

pub fn refresh_gun_decide_cache(
    run: Res<Run>,
    save: Res<crate::savedata_part::SaveData>,
    player_q: Query<(&Player, &Inventory, &RaceState), (With<Player>, Without<Prop>)>,
    mut cache: ResMut<GunDecideCache>,
) {
    cache.ctx = player_q.single().ok().map(|(p, inv, race)| {
        decide_ctx_for(
            &run,
            p,
            race.race,
            inv,
            u32::from(race.race == RaceId::Steroids),
        )
    });
    cache.particles = save.settings.particles;
}

/// Build the `DecideCtx` for drop/chest rolls around one player
/// (GML `instance_nearest(x, y, Player)` + `GameCont.hard`).
pub fn decide_ctx_for(
    run: &Run,
    player: &Player,
    race: RaceId,
    inv: &Inventory,
    robots: u32,
) -> crate::decide_wep::DecideCtx {
    crate::decide_wep::DecideCtx {
        hard: crate::decide_wep::game_hard(&run),
        hardmode: run.hardmode,
        robots,
        refined_taste: player.ultra == Some(UltraMutationId::RobotRefinedTaste),
        crown_guns: player.crown == CrownKind::Guns,
        tutorial: run.tutorial,
        target_race: race,
        owned: inv.weapons.iter().copied().collect(),
    }
}

pub fn maybe_spawn_drop(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    chance: usize,
    weapon_chance: usize,
    player: &Player,
    inv: &Inventory,
    health: &Health,
    loops: u32,
    decide: Option<&crate::decide_wep::DecideCtx>,
) {
    maybe_spawn_drop_ctx(
        commands,
        catalog,
        pos,
        chance,
        weapon_chance,
        player,
        inv,
        health,
        loops,
        decide,
        &ChestCtx {
            crown: player.crown,
            ambidextrous: steroids_ambidextrous(player),
            get_loaded: steroids_get_loaded(player),
            ..Default::default()
        },
    );
}

/// GML `scripts/scrDrop/scrDrop.gml` verbatim. `chest_ctx` carries the
/// spawn-time rolls a Confiscate `HealthChest` / `AmmoChest` /
/// `WeaponChest` makes in `Create_0` (GML resolves them there, outside
/// worldgen for the Confiscate case, so `worldgen` stays false).
#[allow(clippy::too_many_arguments)]
pub fn maybe_spawn_drop_ctx(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    pos: glam::Vec2,
    chance: usize,
    weapon_chance: usize,
    player: &Player,
    inv: &Inventory,
    health: &Health,
    loops: u32,
    decide: Option<&crate::decide_wep::DecideCtx>,
    chest_ctx: &ChestCtx,
) {
    let mut rng = rand::rng();

    // GML `scrDrop.gml:32-34`: Fish Confiscate rolls `_confiscate` at
    // 20%, which swaps the HPPickup / AmmoPickup / weapon pickup for a
    // real HealthChest / AmmoChest / WeaponChest. There is no per-pickup
    // ammo bonus anywhere in GML for it.
    let confiscate =
        matches!(player.ultra, Some(UltraMutationId::FishConfiscate)) && rng.random::<f32>() < 0.2;

    let need = scrub_need(inv, player);
    // GML `scrDrop.gml:27-29` keys `_paw_chance` off the Rabbit Paw SKILL
    // alone. The port also folds crowns and several ultimates into
    // `Player.drop_mult`, and none of those belong in the pickup roll, so
    // the paw is recomputed from the mutation list.
    let paw = if player.mutations.contains(&MutationId::RabbitPaw) {
        0.4
    } else {
        0.0
    };

    let mut weapon_chance = weapon_chance;
    if player.crown == CrownKind::Guns {
        weapon_chance += 9;
    }
    let mut chance = chance as f32;
    if player.crown == CrownKind::Risk {
        chance *= if health.hp >= health.max { 1.5 } else { 0.5 };
    }
    // GML `scrDrop.gml:58` short-circuits on `_pickup_chance > 0`, so a
    // zero-chance call consumes NO `random(100)`. The `else if` arm draws
    // its own, which is what keeps one `random(100)` per death roll.
    let pickup_hit = chance > 0.0 && rng.random_range(0.0..100.0) < (chance * (need + paw));

    let hasted = player.crown == CrownKind::Haste;

    if pickup_hit {
        // GML `scrDrop.gml:59,62`:
        //   `_advantage = scrGameIsHardmode() ? 1.5 : 2`
        //   `if random(_max_hp) > _hp && random(3) < _advantage && !crwn_life`
        // `random(3)` is INTEGER (0/1/2), so `< 2` and `< 1.5` both accept
        // 0 and 1: the medkit arm wins 2 of 3 times in BOTH modes, and
        // the two rolls must happen in GML's order (hp gate first).
        let advantage = if decide.is_some_and(|c| c.hardmode) {
            1.5_f32
        } else {
            2.0_f32
        };
        let life_blocks = player.crown == CrownKind::Life;
        let guns_blocks_ammo = player.crown == CrownKind::Guns;
        let at = pos + glam::Vec2::new(rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0));
        if rng.random_range(0..health.max.max(1)) as i32 > health.hp
            && (rng.random_range(0..3) as f32) < advantage
            && !life_blocks
        {
            if confiscate {
                spawn_chest_with(commands, catalog, ChestKind::Health, at, chest_ctx);
            } else {
                spawn_pickup(
                    commands,
                    catalog,
                    PickupKind::Medkit(hppickup_num(player)),
                    at,
                    loops,
                    hasted,
                );
            }
        } else {
            if !guns_blocks_ammo {
                if confiscate {
                    spawn_chest_with(commands, catalog, ChestKind::Ammo, at, chest_ctx);
                } else {
                    spawn_ammo_pickup(commands, catalog, count_cursed(inv), at, loops, hasted);
                }
            }
        }
    } else if weapon_chance > 0 && rng.random_range(0.0..100.0) < weapon_chance as f32 {
        let at = pos + glam::Vec2::new(rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0));
        if confiscate {
            spawn_chest_with(commands, catalog, ChestKind::Weapon, at, chest_ctx);
        } else {
            let weapon = match decide {
                Some(ctx) => crate::decide_wep::decide_wep(&mut rng, ctx, 0, false),
                None => random_weapon(&mut rng),
            };
            let e = spawn_pickup(commands, catalog, PickupKind::Weapon(weapon), at, 0, false);
            // GML `scrDrop.gml:85`: `scrWeaponPickupCreate(..., true)`.
            commands.entity(e).insert(WepPickupAmmo(true));
        }
    }
}

/// GML `HPPickup/Create_0.gml:8-15` verbatim: `num = 2`, `4` with Second
/// Stomach, `+1` for the Haste crown. `HPPickup/Collision_Player.gml:11-14`
/// heals exactly `num` - no multipliers.
pub fn hppickup_num(player: &Player) -> i32 {
    let mut num = if player.mutations.contains(&MutationId::SecondStomach) {
        4
    } else {
        2
    };
    if player.crown == CrownKind::Haste {
        num += 1;
    }
    num
}

fn random_ammo_kind(rng: &mut impl rand::RngExt) -> AmmoKind {
    match rng.random_range(0..5) {
        0 => AmmoKind::Bullets,
        1 => AmmoKind::Shells,
        2 => AmmoKind::Bolts,
        3 => AmmoKind::Explosives,
        _ => AmmoKind::Energy,
    }
}

pub fn random_weapon(rng: &mut impl rand::RngExt) -> WeaponId {
    match rng.random_range(0..8) {
        0 => WeaponId::MACHINEGUN,
        1 => WeaponId(5),
        2 => WeaponId::CROSSBOW,
        3 => WeaponId::GRENADE_LAUNCHER,
        4 => WeaponId::SMG,
        5 => WeaponId::ASSAULT_RIFLE,
        6 => WeaponId::WRENCH,
        _ => WeaponId::SLEDGEHAMMER,
    }
}

/// Gold-weapon roll (bevy `random_gold_weapon` parity: uniform over
/// the weapons table's gold flag, plain roll when empty).
pub fn random_gold_weapon_fallback(rng: &mut impl rand::RngExt) -> WeaponId {
    let gold: Vec<WeaponId> = crate::weapons_data::WEAPONS
        .iter()
        .filter(|w| w.wep_gold)
        .map(|w| WeaponId(w.id))
        .collect();
    if gold.is_empty() {
        return random_weapon(rng);
    }
    gold[rng.random_range(0..gold.len())]
}

/// GML `scrDrop.gml:38-51` `_need`: two slots, `wep` then `bwep`.
/// A valid slot scores 0.75 / 0.1 / 0.5 by ammo fill against
/// `scrAmmoGetTypeCapacity`; a MELEE gun has type `Ammo.None`, whose
/// capacity is 0 while `ammo[None]` is 999, so it always lands in the
/// `> cap * 0.6` arm and scores 0.1. An invalid FIRST slot scores
/// nothing (`_first` is still true) while an invalid `bwep` scores 0.5.
fn scrub_need(inv: &Inventory, player: &Player) -> f32 {
    let mut need = 0.0;
    for (i, w) in inv.weapons.iter().take(2).enumerate() {
        if *w == WeaponId::NONE {
            if i > 0 {
                need += 0.5;
            }
            continue;
        }
        let def = weapon_runtime_def(*w);
        let cap = if def.melee.is_some() {
            0.0
        } else {
            player.ammo_cap(def.ammo) as f32
        };
        let am = if def.melee.is_some() {
            999.0
        } else {
            inv_ammo(inv, def.ammo) as f32
        };
        if am < cap * 0.2 {
            need += 0.75;
        } else if am > cap * 0.6 {
            need += 0.1;
        } else {
            need += 0.5;
        }
    }
    need
}

/// Ammo count for a kind (used by need/give calculations).
pub fn inv_ammo(inv: &Inventory, kind: AmmoKind) -> i32 {
    inv.ammo_of(kind)
}

pub fn give_ammo(inv: &mut Inventory, player: &Player) {
    let id = inv.weapons[inv.current];
    let haste = haste_crown(player);
    if id == WeaponId::NONE {
        let mut rng = rand::rng();
        let kind = random_ammo_kind(&mut rng);
        let slot = inv.ammo_mut(kind);
        let add = ammo_pickup_amount_for(kind, 0, haste);
        *slot = (*slot + add).min(player.ammo_cap(kind));
        return;
    }
    let def = weapon_runtime_def(id);
    if def.melee.is_some() {
        return;
    }
    let slot = inv.ammo_mut(def.ammo);
    let add = ammo_pickup_amount_for(def.ammo, 0, haste);
    *slot = (*slot + add).min(player.ammo_cap(def.ammo));
}

/// GML `scrPlayerCountRace(Race.Fish)` in single-player: 1 for a Fish,
/// 0 otherwise. `scrAmmoUpdateTypeStats` (`scrAmmoInit.gml:121-129`) is
/// the only consumer of the Fish count in the pickup path.
pub fn fish_player_count(race: RaceId) -> u32 {
    u32::from(race == RaceId::Fish)
}

/// GML `scrCrownCheck(crwn_haste)` as the 0/1 the `typ_ammo` fold wants.
pub fn haste_crown(player: &Player) -> u32 {
    u32::from(player.crown == CrownKind::Haste)
}

/// GML `scr_ultra_get(Race.Steroids, UltraSkill.GetLoaded)`: the Steroids
/// ULTRA B, not the race's R ability (`AbilityKind::GetLoaded`).
pub fn steroids_get_loaded(player: &Player) -> bool {
    matches!(player.ultra, Some(UltraMutationId::SteroidsGetArmed))
}

/// GML `scr_ultra_get(Race.Steroids, UltraSkill.Ambidextrous)`.
pub fn steroids_ambidextrous(player: &Player) -> bool {
    matches!(player.ultra, Some(UltraMutationId::SteroidsAmbidextrous))
}

/// Toast expiry (bevy `pickups.rs:999` parity: duration-zero timers are
/// inert, otherwise the text clears when the 2.2 s timer lapses).
/// Text-only effect; kept so run-setup crown toasts fade headless.
pub fn tick_toast(time: Res<repame_sim::SimTime>, mut toast: ResMut<Toast>) {
    if toast.timer.duration() <= 0.0 {
        return;
    }
    toast.timer.tick(time.delta_secs);
    if toast.timer.is_finished() {
        toast.text.clear();
    }
}

// Pickup tick battery. Ported from nt's `game/pickups.rs` Progression-set
// systems. Render split: `Visibility` blink-out and sprite alpha fades are
// renderer-owned (skipped); the sim keeps lifetimes, motion, grants.

/// GML `CursedPickup` (`Step_0.gml` + `Alarm_0.gml`) verbatim:
/// `image_index` dwells on frame 0 advancing by `random(0.04)` then
/// runs at `0.4`; on every frame turn (`current_frame_active`) there is
/// a `random(4) < 1` chance to shed a `Curse`. `Alarm_0` re-arms every
/// 2 steps, decrements `blink`, and on `blink < 0` plays the
/// `SmallExplosion` and the disappear stings.
pub fn tick_cursed_ammo(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut q: Query<(Entity, &Pos, &mut CursedAmmoBlink, &mut GmlImage)>,
    audio: Res<crate::audio::GameAudio>,
    mut cues: ResMut<crate::msg::Queue<crate::audio::AudioCue>>,
) {
    let steps = time.delta_secs * crate::SIM_HZ as f32;
    let mut rng = rand::rng();
    for (entity, pos, mut blink, mut image) in &mut q {
        if !blink.sounded {
            blink.sounded = true;
            audio.play_cursed_pickup(&mut cues);
        }
        // GML `CursedPickup/Step_0.gml:3-6` verbatim: frame 0 dwells
        // while `image_index` creeps by `random(0.04)` a step, then the
        // whole strip runs at a flat `0.4`.
        if image.phase < 1.0 {
            image.phase += rng.random_range(0.0..0.04) * steps;
        } else {
            image.phase += 0.4 * steps;
        }
        if image.phase >= image.frames.max(1) as f32 {
            image.phase %= image.frames.max(1) as f32;
        }
        // GML `CursedPickup/Step_0.gml:9` gates on `current_frame_active`, a
        // `#macro` of `((current_frame % 1) < timescale)`. Nothing in the GML
        // ever increments `current_frame` (only `UberCont/Create_0.gml:140`
        // resets it) and `MainMenuButton/Step_0.gml:7` compares it against an
        // integer, so the macro is `0 < 1` - unconditionally true: the roll runs
        // every step, not once per image frame.
        if rng.random::<f32>() < 0.25 {
            spawn_pickup(&mut commands, &catalog, PickupKind::Curse, pos.0, 0, false);
        }
        blink.alarm -= steps;
        if blink.alarm > 0.0 {
            continue;
        }
        blink.alarm += 2.0;
        // GML `Alarm_0.gml` tests `blink < 0` BEFORE decrementing. `blink`
        // starts at 30, so the check sees 30-(k-1) on firing k and first
        // goes negative at k=32 - step `A + 2*31` = `A + 62`.
        if blink.blink < 0 {
            crate::spawns::spawn_explosion_with_source_radius_kind(
                &mut commands,
                pos.0,
                5,
                None,
                32.0,
                Team::Enemy,
                true,
                Some(crate::comps_b::NativeExplosionKind::Small),
            );
            audio.play_cursed_pickup_disappear(&mut cues);
            commands.entity(entity).despawn();
            continue;
        }
        blink.blink -= 1;
    }
}

/// Loose-pickup drift: weapons near the portal get carried through,
/// rads slide toward the player once a portal exists, ammo/medkits
/// drift at the GML rate inside pickup range (mask-gated per axis).
pub fn tick_pickup_drag(
    time: Res<SimTime>,
    mut commands: Commands,
    portal_q: Query<&Pos, (With<Portal>, Without<Pickup>)>,
    mut carried: ResMut<PortalCarriedWeapons>,
    player_q: Query<(&Pos, &Player), With<Player>>,
    mask: Res<FloorMask>,
    mut pickups: Query<(Entity, &mut Pos, &Pickup), (Without<Player>, Without<Portal>)>,
) {
    let Ok((player_pos, player)) = player_q.single() else {
        return;
    };
    let player_pos = player_pos.0;
    let portal_pos = portal_q.single().ok().map(|p| p.0);
    let dt = time.delta_secs;
    let hunger = player.mutations.contains(&MutationId::PlutoniumHunger);

    let loose_range = 32.0 + if hunger { 64.0 } else { 0.0 };

    for (e, mut pos, pickup) in &mut pickups {
        let ppos = pos.0;
        match pickup.kind {
            PickupKind::Weapon(w) => {
                if portal_pos.is_some_and(|pp| ppos.distance(pp) < 20.0) {
                    carried.0.push(w);
                    commands.entity(e).try_despawn();
                }
            }
            PickupKind::Rad(_) => {
                if portal_pos.is_none() {
                    continue;
                }

                let dir = (player_pos - ppos).normalize_or_zero();
                pos.0 += dir * 360.0 * dt;

                if ppos.distance(portal_pos.unwrap_or(player_pos)) < 20.0 {
                    pos.0 = player_pos;
                }
            }
            PickupKind::Ammo(..) | PickupKind::Medkit(_) => {
                let in_range = ppos.distance(player_pos) < loose_range;
                if !in_range && portal_pos.is_none() {
                    continue;
                }

                let dir = (player_pos - ppos).normalize_or_zero();
                let delta = dir * 6.0 * 30.0 * dt;
                let nx = glam::Vec2::new(ppos.x + delta.x, ppos.y);
                if mask.is_walkable(nx) {
                    pos.0.x = nx.x;
                }
                let ny = glam::Vec2::new(pos.0.x, ppos.y + delta.y);
                if mask.is_walkable(ny) {
                    pos.0.y = ny.y;
                }

                if portal_pos.is_some_and(|pp| ppos.distance(pp) < 14.0) {
                    pos.0 = player_pos;
                }
            }
            _ => {}
        }
    }
}

/// Pickup collection: ground-physics slide, lifetime expiry, telekinesis
/// magnet, per-kind ranges, chest loot tables, rad/ammo/medkit/weapon
/// grants. Bevy `collect_pickups` parity minus render blinks.
#[allow(clippy::too_many_arguments)]
pub fn collect_pickups(
    time: Res<SimTime>,
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    mut trauma: ResMut<Trauma>,
    mut flash: ResMut<FlashWhite>,
    mut chroma: ResMut<ChromaticAberration>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    input: ResMut<NtInput>,
    mut run: ResMut<Run>,
    mask: Res<FloorMask>,
    mut player_q: Query<
        (
            Entity,
            &Pos,
            &mut Player,
            &mut Health,
            &mut Inventory,
            Option<&Telekinesis>,
            Option<&RaceState>,
            Option<&mut crate::comps_a::FireCooldown>,
        ),
        (With<Player>, Without<Pickup>),
    >,
    mut pickups: Query<
        (
            Entity,
            &mut Pos,
            &Pickup,
            Option<&mut GroundPhysics>,
            Option<&mut PickupLifetime>,
            Option<&mut WepPickupAmmo>,
            Option<&PickupCurse>,
            Option<&ChestCurse>,
            Option<&DropSeed>,
        ),
        Without<Player>,
    >,
    mut proto_q: Query<(Entity, &mut ProtoChestState)>,
    mut toast: ResMut<Toast>,
    mut tut: Option<ResMut<crate::state::TutorialState>>,
) {
    let Ok((_player_e, player_pos, mut player, mut health, mut inv, telek, race_opt, mut fire_cd)) =
        player_q.single_mut()
    else {
        return;
    };

    arm_proto_chests(&mut commands, &mut proto_q, &run);
    let player_pos = player_pos.0;
    let dt = time.delta_secs;
    // The weapon arm below peeks the shared pulse (see its note);
    // nothing here consumes it, so `tick_throne_sit` can peek it too.
    let _ = input.peek_interact_pressed();

    let telek_active = telek.is_some_and(|t| !t.timer.is_finished());
    // GML `scrEyesTelekinesis.gml:2-3`: the attract box is the SCREEN box,
    // `game_screen_width div 2 x game_screen_height div 2`, and
    // `game_screen_width/height` are compile-time macros (320/240), so the
    // half-extents are 160 x 120. `:14` strength is
    // `1 + scr_skill_get(mut_throne_butt)` px/step (30/60 px per second here),
    // moved per axis behind `place_free`.
    let telek_step = if telek_active {
        30.0 * if player.throne_butt { 2.0 } else { 1.0 }
    } else {
        0.0
    };
    // GML `scrEyesTelekinesis:35-41` drags `chestprop`, `AmmoPickup`,
    // `HPPickup`, `WepPickup`, `RadChest` and `Rad` - NOT `Curse` motes -
    // through this direct-position arm, so chests DO drag and must stay out of
    // the ordinary pickup drift below.
    let telek_drag = |pos: &mut Pos, at: glam::Vec2| {
        if telek_step <= 0.0 {
            return;
        }
        if (at.x - player_pos.x).abs() >= crate::vortex::GUI_W / 2.0
            || (at.y - player_pos.y).abs() >= crate::vortex::GUI_H / 2.0
        {
            return;
        }
        let step = (player_pos - at).normalize_or_zero() * telek_step * dt;
        if mask.is_walkable(glam::Vec2::new(at.x + step.x, at.y)) {
            pos.0.x += step.x;
        }
        let moved = pos.0;
        if mask.is_walkable(glam::Vec2::new(moved.x, moved.y + step.y)) {
            pos.0.y += step.y;
        }
    };

    let race = race_opt.map(|r| r.race).unwrap_or(RaceId::Fish);
    let fish = fish_player_count(race);
    let haste = haste_crown(&player);
    let get_loaded = steroids_get_loaded(&player);
    let ambidextrous = steroids_ambidextrous(&player);
    let underwater = run.area == crate::data::AreaId::Oasis;
    let loops = run.loop_count;

    let nearest_weapon: Option<Entity> = nearest_ground_weapon(
        player_pos,
        pickups.iter().map(|(e, pos, p, ..)| (e, pos.0, p.kind)),
    );

    for (
        pickup_e,
        mut pickup_pos,
        pickup,
        ground,
        lifetime,
        mut wep_ammo,
        pickup_curse,
        chest_curse,
        drop_seed,
    ) in &mut pickups
    {
        let pickup_pos_value = pickup_pos.0;
        let dist = player_pos.distance(pickup_pos_value);
        // GML `Rad/Step_0.gml:6`: `if (speed > 0) exit` - a rad still
        // coasting from its drop kick does not home yet. Read before the
        // ground-physics slide below consumes `ground`.
        let rad_coasting = ground
            .as_ref()
            .is_some_and(|g| g.vel.length() > crate::comps_a::PLAYER_RADIUS * 0.0625);

        if let Some(mut gp) = ground {
            let speed = gp.vel.length();
            if speed > 0.5 {
                pickup_pos.0 += gp.vel * dt;
                gp.vel *= 0.4_f32.powf(dt * crate::SIM_HZ as f32);
            } else {
                gp.vel = glam::Vec2::ZERO;
            }
        }
        // Render split: bevy spun the sprite (`rotate_z`) while sliding;
        // headless pickups carry no angle channel, velocity decay above
        // is the sim effect.

        if let Some(mut lt) = lifetime {
            lt.timer.tick(time.delta_secs);
            if lt.timer.just_finished() {
                audio.play_pickup_disappear(&mut cues);
                commands.entity(pickup_e).try_despawn();
                continue;
            }
            // Render split: ammo/hp blink-out and sub-second alpha fade
            // are renderer-owned (skipped); expiry above is the sim law.
        }

        let is_chest = matches!(pickup.kind, PickupKind::Chest(_));
        let is_weapon = matches!(pickup.kind, PickupKind::Weapon(_));
        let is_rad = matches!(pickup.kind, PickupKind::Rad(_));
        if is_weapon || is_chest || is_rad {
            telek_drag(&mut pickup_pos, pickup_pos_value);
        }
        if is_rad {
            // GML `Rad/Step_0.gml:18-22`: `d = 80 + 60 *
            // scr_skill_get(mut_plutonium_hunger)`, then
            // `mp_potential_step(target, 12, 0)` - 12 px/step.
            let has_hunger = player.mutations.contains(&MutationId::PlutoniumHunger);
            let rad_range = 80.0 + if has_hunger { 60.0 } else { 0.0 };
            if !rad_coasting && dist < rad_range {
                let dir = (player_pos - pickup_pos_value).normalize_or_zero();
                pickup_pos.0 += dir * 12.0 * 30.0 * dt;
            }
        }

        // GML `Player/Collision_WepPickup:90-104` verbatim: the weapon
        // pickup's ammo payout sits OUTSIDE the pick if/else, so plain
        // mask overlap pays it whether or not the press landed, and
        // whether or not this is the nearest gun. The flag is consumed.
        if let Some(ammo_flag) = wep_ammo.as_deref_mut()
            && ammo_flag.0
            && is_weapon
            && mask_overlap(player_pos, pickup_pos_value, WEP_AMMO_REACH)
        {
            let PickupKind::Weapon(gun) = pickup.kind else {
                unreachable!()
            };
            // GML `Player/Collision_WepPickup.gml:103` is the unconditional
            // trailing `other.ammo = 0`: the payout fires once per pickup.
            // Leaving the flag set re-pays it every step the masks overlap.
            ammo_flag.0 = false;
            pay_weapon_pickup_ammo(
                &mut commands,
                &mut inv,
                gun,
                player_pos,
                &player,
                &mut health,
            );
        }

        // GML reach: per-axis mask overlap (`place_meeting`).
        let Some(half) = pickup_mask_half(&pickup.kind, nearest_weapon == Some(pickup_e)) else {
            continue;
        };
        if !mask_overlap(player_pos, pickup_pos_value, half) {
            continue;
        }
        if is_weapon {
            // GML `Player/Collision_WepPickup:6` verbatim: `press_pick` OR `autopick`.
            // `WepPickup/Create_0:11` starts it false and the only assignment to
            // true (`scrPowers:581`) sits in `scrCuzThrowAllAbility`, reached
            // solely behind `#macro cuz_fun false` (`scrPowers:2,426`), so every
            // ground gun needs the interact press; the pulse is peeked, not
            // taken, so the earlier `tick_throne_sit` peek of the same pulse
            // never starves it.
            if !input.peek_interact_pressed() {
                continue;
            }
        }

        if let PickupKind::Chest(chest) = pickup.kind {
            open_chest(&mut commands, pickup_e, chest);
            let ctx = decide_ctx_for(&run, &player, race, &inv, u32::from(race == RaceId::Robot));
            let seed = drop_seed.map_or(0, |s| s.0);

            // GML `scrChestOpened` (`scripts/scrChestOpened.gml:11-29`) is called
            // at the TOP of every chest's `Collision_Player`, before the loot. The
            // Crown of Hatred burns 1 HP through the normal i-frame check
            // (`scrPlayerProcTakeDamage`, so 1 HP survives) and drops 16 rads at
            // the PLAYER's position - the event runs `with (p)`. The `_amount =
            // 24` line is dead code (`other` in a `Collision_Player` event is the
            // Player, never a `RadChest`), deliberately NOT implemented.
            // `ProtoChest` never calls the script (it runs its own block below),
            // so it is excluded here to avoid the double trigger.
            if chest != ChestKind::Proto && player.crown == CrownKind::Hatred && health.hp > 0 {
                if health.invuln.is_finished() {
                    health.hp -= 1;
                    health.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                }
                spawn_hatred_rads(&mut commands, &catalog, player_pos, 16, loops);
            }

            // GML `Destroy_0` spawns `FXChestOpen` for every chest kind
            // except `RogueChest` (the only chest with no FX).
            if chest != ChestKind::Rogue {
                spawn_fx_chest_open(&mut commands, &catalog, pickup_pos_value, underwater);
            }

            match chest {
                ChestKind::Weapon => {
                    // GML `WeaponChest/Collision_Player.gml:11-24`
                    // verbatim: `scrDecideWep(1 + curse * 2, curse)` is
                    // rolled ONCE, then `repeat (_count)` spawns that
                    // SAME weapon. `_count` is 2 with Ambidextrous.
                    let cursed = chest_curse.is_some_and(|c| c.0);
                    let count = if ambidextrous { 2 } else { 1 };
                    let extra = 1 + if cursed { 2 } else { 0 };
                    let weapon = DropSeedChain::new(seed).roll(|rng| {
                        crate::decide_wep::decide_wep_at(
                            &mut commands,
                            rng,
                            &ctx,
                            pickup_pos_value,
                            extra,
                            cursed,
                        )
                    });
                    let mut rng = rand::rng();
                    for _ in 0..count {
                        let e = spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Weapon(weapon),
                            pickup_pos_value
                                + glam::Vec2::new(
                                    rng.random_range(-2.0..2.0),
                                    rng.random_range(-2.0..2.0),
                                ),
                            0,
                            false,
                        );
                        // GML `scrWeaponPickupCreate(..., true)`.
                        commands.entity(e).insert(WepPickupAmmo(true));
                        if cursed {
                            commands.entity(e).insert(PickupCurse);
                        }
                    }
                    toast.show(weapon_id_name(weapon));
                    audio.play_weapon_chest_open(&mut cues, underwater, cursed);
                }
                ChestKind::Gold => {
                    // GML `GoldChest/Collision_Player.gml`: one
                    // `scrDecideWepGold` pickup.
                    let mut rng = rand::rng();
                    let weapon = crate::decide_wep::decide_wep_gold(
                        &mut rng,
                        loops,
                        &inv.weapons,
                        race == RaceId::Steroids,
                    );
                    let e = spawn_pickup(
                        &mut commands,
                        &catalog,
                        PickupKind::Weapon(weapon),
                        pickup_pos_value,
                        0,
                        false,
                    );
                    commands.entity(e).insert(WepPickupAmmo(true));
                    audio.play_gold_chest(&mut cues);
                }
                ChestKind::Ammo | ChestKind::Mystery => {
                    // GML `AmmoChest/Collision_Player.gml:11-21` and
                    // `AmmoChestMystery/Collision_Player.gml:14-30`:
                    // `scrAmmoGetPickupAmount(_type) * 2` (or `* 3` for
                    // the mystery chest), Get Loaded filling every type.
                    open_ammo_chest(
                        &mut commands,
                        &mut inv,
                        &player,
                        race,
                        player_pos,
                        fish,
                        haste,
                        get_loaded,
                        chest == ChestKind::Mystery,
                        &mut toast,
                    );
                    audio.play_ammo_chest_open(&mut cues, underwater);
                }
                ChestKind::Rad => {
                    // GML `RadChest/Collision_Player.gml`:
                    // `if !scrChestOpened() { GameCont.noradch = 0; hp = 0 }`, then
                    // `RadChest/Destroy_0.gml:4-12` throws 4 x `Smoke`
                    // (`motion_add(random_angle, random(3))`), an `ExploderExplo`
                    // and `sndEXPChest`, and its own `event_inherited()` lands on
                    // `prop/Destroy_0.gml:12` to turn the inherited `raddrop =
                    // 25` (`RadChest/Create_0.gml:20`) into 25 rads.
                    run.noradch = 0;
                    rad_chest_burst(&mut commands, pickup_pos_value);
                    scr_rad_drop(
                        &mut commands,
                        &catalog,
                        pickup_pos_value,
                        25,
                        loops,
                        haste > 0,
                        true,
                        true,
                    );
                    audio.play_exp_chest(&mut cues);
                }
                ChestKind::RadBig => {
                    // GML `RadChestBig` overrides only `Create_0`
                    // (`raddrop = 45`, `max_hp = 20`), so it runs the
                    // inherited `RadChest/Destroy_0` body verbatim.
                    rad_chest_burst(&mut commands, pickup_pos_value);
                    scr_rad_drop(
                        &mut commands,
                        &catalog,
                        pickup_pos_value,
                        45,
                        loops,
                        haste > 0,
                        true,
                        true,
                    );
                    audio.play_exp_chest(&mut cues);
                }
                ChestKind::RadMaggot => {
                    // GML `RadMaggotChest/Destroy_0.gml:1-13`, in order:
                    //   `:1-3`  `if hp <= 0 instance_create(x, y, RadMaggotExplosion)`
                    //   `:5-8`  4 x `Smoke`, `motion_add(random_angle, random(3))`
                    //   `:10`   `instance_create(x, y, ExploderExplo)`
                    //   `:11`   `snd_play(sndEXPChest)`
                    //   `:13`   `event_inherited()` -> `RadChest/Destroy_0.gml`, which
                    //           repeats the same 4 x `Smoke` + `ExploderExplo` +
                    //           `sndEXPChest` before its own `event_inherited()`
                    //           reaches `prop/Destroy_0.gml:12`
                    //           `if (raddrop > 0) scrRadDrop(x, y, raddrop)`.
                    // `RadMaggotChest.yy`'s `parentObjectId` is `RadChest`, so
                    // `Create_0`'s `event_inherited()` runs `RadChest/Create_0.gml:20`
                    // and the maggot cache INHERITS `raddrop = 25` (it never overrides
                    // it) - 25 rads, scattered by `scrRadDrop`.
                    // `ExploderExplo/Create_0.gml` is 6 x `Smoke` plus
                    // `BackCont.shake += 6` - no `damage`, no `Collision_Player`, so
                    // opening a maggot cache deals 0.
                    // `RadMaggotExplosion/Create_0.gml:1-14` fires the ring: 6
                    // `Smoke` at `motion_add(dir, 4 + random(1))` stepping
                    // `dir += 360 / 6`, then 3 `AcidStreak` at `motion_add(dir, 8)`
                    // stepping `dir += 120`.
                    let mut rng = rand::rng();
                    let mut dir = rng.random_range(0.0..std::f32::consts::TAU);
                    for _ in 0..6 {
                        let speed = 4.0 + rng.random_range(0.0..1.0);
                        crate::environment::spawn_native_smoke_mote(
                            &mut commands,
                            true,
                            pickup_pos_value,
                            glam::Vec2::from_angle(dir),
                            speed,
                        );
                        dir += 360.0_f32.to_radians() / 6.0;
                    }
                    dir = rng.random_range(0.0..std::f32::consts::TAU);
                    for _ in 0..3 {
                        crate::environment::spawn_native_streak(
                            &mut commands,
                            true,
                            pickup_pos_value,
                            dir,
                            8.0,
                        );
                        dir += 120.0_f32.to_radians();
                    }
                    // `RadMaggotExplosion/Alarm_0.gml:1-4`:
                    //   `repeat(20) { with instance_create(x + random(8) - 4,
                    //     y + random(8) - 4, RadMaggot) motion_add(random_angle,
                    //     random(5)) }`
                    // A VELOCITY of 0-5 px/step, not a spawn offset.
                    // `alarm[0] = 8` (`Create_0.gml:3`) is the eight-frame wind-up.
                    for _ in 0..20 {
                        let at = pickup_pos_value
                            + glam::Vec2::new(
                                rng.random_range(0.0..8.0) - 4.0,
                                rng.random_range(0.0..8.0) - 4.0,
                            );
                        let ang = rng.random_range(0.0..std::f32::consts::TAU);
                        let speed = rng.random_range(0.0..5.0);
                        queue_enemy_spawn_birth(
                            &mut commands,
                            EnemyKind::RadMaggot,
                            at,
                            1.0,
                            loops,
                            true,
                            Some(glam::Vec2::from_angle(ang) * speed * 30.0),
                            false,
                        );
                    }
                    // `Destroy_0.gml:13`'s `event_inherited()` is
                    // `RadChest/Destroy_0.gml:4-12`, which repeats the same 4 x `Smoke`
                    // + `ExploderExplo` + `sndEXPChest` before its own
                    // `event_inherited()` reaches `prop/Destroy_0.gml:12`.
                    rad_chest_burst(&mut commands, pickup_pos_value);
                    audio.play_exp_chest(&mut cues);
                    rad_chest_burst(&mut commands, pickup_pos_value);
                    scr_rad_drop(
                        &mut commands,
                        &catalog,
                        pickup_pos_value,
                        25,
                        loops,
                        haste > 0,
                        true,
                        true,
                    );
                    audio.play_exp_chest(&mut cues);
                }
                ChestKind::Health => {
                    // GML `HealthChest/Collision_Player.gml`: banked
                    // head-loss pays back one max-HP first, then heals
                    // `num` (4, 8 with Second Stomach).
                    if player.headloses > 0 {
                        player.headloses -= 1;
                        health.max += 1;
                    }
                    let big = player.mutations.contains(&MutationId::SecondStomach);
                    let num = if big { 8 } else { 4 };
                    health.hp = (health.hp + num).min(health.max);
                    repame_fx::spawn_number(
                        &mut commands,
                        player_pos.x,
                        player_pos.y,
                        num.to_string(),
                        [0.3, 1.0, 0.3, 1.0],
                    );
                    audio.play_health_chest(&mut cues, big);
                    toast.show("Healed");
                }
                ChestKind::CursedBig => {
                    // GML `CursedBigChest/Collision_Player.gml:20` verbatim:
                    // `scrDecideWep(1 + curse * 2, false)` with `curse = true`
                    // from `Create_0:3`, so `extra = 3`, and
                    // `random_set_seed(dropseed)` runs ONCE before the `repeat` -
                    // `scrDecideWep` itself reseeds from `dropseed` and advances
                    // it, so all three guns come off that fixed chain. `:33`
                    // resets `nochest`, `:17` spawns a `PortalClear` and
                    // `Destroy_0:6` a SECOND one.
                    let count = if ambidextrous { 4 } else { 3 };
                    let mut chain = DropSeedChain::new(seed);
                    let mut rng = rand::rng();
                    for _ in 0..count {
                        let weapon = chain.roll(|r| {
                            crate::decide_wep::decide_wep_at(
                                &mut commands,
                                r,
                                &ctx,
                                pickup_pos_value,
                                3,
                                false,
                            )
                        });
                        let e = spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Weapon(weapon),
                            pickup_pos_value
                                + glam::Vec2::new(
                                    rng.random_range(-2.0..2.0),
                                    rng.random_range(-2.0..2.0),
                                ),
                            0,
                            false,
                        );
                        commands.entity(e).insert(WepPickupAmmo(true));
                        commands.entity(e).insert(PickupCurse);
                        toast.show(weapon_id_name(weapon));
                    }
                    spawn_portal_clear(&mut commands, pickup_pos_value);
                    run.nochest = 0;
                    audio.play_big_chest_open(&mut cues, true, chst_stem(race));
                }
                ChestKind::BigWeapon => {
                    // GML `BigWeaponChest/Collision_Player.gml:14-30`:
                    // `random_set_seed(dropseed)` ONCE, then `scrDecideWep(1,
                    // false)` per drop - each gun a fresh roll off the same
                    // chain. `:16` a `PortalClear`, `:30` resets `nochest`.
                    let count = if ambidextrous { 4 } else { 3 };
                    let mut chain = DropSeedChain::new(seed);
                    let mut rng = rand::rng();
                    for _ in 0..count {
                        let weapon = chain.roll(|r| {
                            crate::decide_wep::decide_wep_at(
                                &mut commands,
                                r,
                                &ctx,
                                pickup_pos_value,
                                1,
                                false,
                            )
                        });
                        let e = spawn_pickup(
                            &mut commands,
                            &catalog,
                            PickupKind::Weapon(weapon),
                            pickup_pos_value
                                + glam::Vec2::new(
                                    rng.random_range(-2.0..2.0),
                                    rng.random_range(-2.0..2.0),
                                ),
                            0,
                            false,
                        );
                        commands.entity(e).insert(WepPickupAmmo(true));
                        toast.show(weapon_id_name(weapon));
                    }
                    spawn_portal_clear(&mut commands, pickup_pos_value);
                    run.nochest = 0;
                    audio.play_big_chest_open(&mut cues, false, chst_stem(race));
                }
                ChestKind::Rogue => {
                    // GML `RogueChest/Collision_Player.gml:6-15`:
                    // a Rogue gets `RogueAmmo` (+1, +2 with Super Portal
                    // Strike, clamped to `rogue_ammo_max`); anyone else
                    // gets `scrRadDrop(other.x, other.y, 25)`.
                    if race == RaceId::Rogue {
                        let amount = 1 + i32::from(matches!(
                            player.ultra,
                            Some(UltraMutationId::RoguePortalStrike)
                        ));
                        player.rogue_ammo = (player.rogue_ammo as i32 + amount)
                            .min(player.rogue_ammo_max as i32)
                            as u8;
                        toast.show(if player.rogue_ammo >= player.rogue_ammo_max {
                            "MAX PORTAL STRIKES"
                        } else if amount > 1 {
                            "+2 PORTAL STRIKES"
                        } else {
                            "+1 PORTAL STRIKE"
                        });
                    } else {
                        scr_rad_drop(
                            &mut commands,
                            &catalog,
                            player_pos,
                            25,
                            loops,
                            haste > 0,
                            false,
                            false,
                        );
                    }
                    audio.play_rogue_canister(&mut cues);
                }
                ChestKind::Proto => {
                    // GML `ProtoChest/Collision_Player.gml:14-22` runs its
                    // OWN Crown-of-Hatred block (1 HP + 16 rads at the
                    // player's position) and never calls
                    // `scrChestOpened()`.
                    if player.crown == CrownKind::Hatred && health.hp > 0 {
                        if health.invuln.is_finished() {
                            health.hp -= 1;
                            health.invuln = GTimer::from_seconds(5.0 / 30.0, TimerMode::Once);
                        }
                        spawn_hatred_rads(&mut commands, &catalog, player_pos, 16, loops);
                    }
                    let (weapon, cursed) = match proto_q.get(pickup_e) {
                        Ok((_, &ProtoChestState::Armed { weapon, cursed })) => (weapon, cursed),
                        _ => (run.protowep, run.protocurse),
                    };
                    spawn_proto_weapon(&mut commands, &catalog, weapon, cursed, pickup_pos_value);
                    audio.play_weapon_chest(&mut cues);
                    toast.show(weapon_id_name(weapon));
                }
                ChestKind::Idpd => {
                    // GML `IDPDChest/Collision_Player.gml:11-13` verbatim:
                    // `repeat (8) instance_create(_player.x, _player.y, AmmoPickup)` -
                    // all eight land ON the player, are collected next tick, and
                    // each independently rolls `scrAmmoDecideType(id, false)`. The
                    // chest then `instance_destroy()`s, and `Destroy_0:11-20` raises
                    // the six `IDPDSpawn` portals.
                    for _ in 0..8 {
                        spawn_ammo_pickup(
                            &mut commands,
                            &catalog,
                            count_cursed(&inv),
                            player_pos,
                            loops,
                            false,
                        );
                    }
                    crate::idpd::raise_idpd_portals(
                        &mut commands,
                        &mut run,
                        &mut cues,
                        0,
                        pickup_pos_value,
                    );
                    audio.play_ammo_chest_open(&mut cues, underwater);
                }
            }
            continue;
        }

        commands.entity(pickup_e).try_despawn();

        match pickup.kind {
            PickupKind::Rad(amount) => {
                player.rads += amount;
                chromatic_pulse(&mut chroma, 0.04);
                audio.play_rad_pickup(&mut cues);
                check_level_up(
                    &mut commands,
                    &mut trauma,
                    &mut flash,
                    &mut player,
                    &mut toast,
                    &mut cues,
                    player_pos,
                );
            }
            PickupKind::Medkit(amount) => {
                // GML `HPPickup/Collision_Player.gml:11-14` heals exactly
                // `num` with no multiplier, so the spawn-time amount is
                // authoritative.
                health.hp = (health.hp + amount).min(health.max);
                try_recharge_strong_spirit(&mut player, &health);
                repame_fx::spawn_number(
                    &mut commands,
                    player_pos.x,
                    player_pos.y,
                    amount.to_string(),
                    [0.3, 1.0, 0.3, 1.0],
                );
                audio.play_hp_pickup(
                    &mut cues,
                    player.mutations.contains(&MutationId::SecondStomach),
                );
            }
            PickupKind::Ammo(..) | PickupKind::CursedAmmo => {
                // GML `AmmoPickup/Collision_Player.gml:9-21`:
                // `scrAmmoDecideType(id, false)`, then `_give_amount =
                // typ_ammo[_type]` plus the Haste `++` (the Haste bonus is
                // applied TWICE in GML: once folded into `typ_ammo` by
                // `scrAmmoUpdateTypeStats`, once here). The `instance_is(other,
                // CursedPickup)` 1.5x is DEAD (`other` is the Player), so a
                // CursedPickup pays the same amount as a normal AmmoPickup.
                let ammo = decide_ammo_type(&inv, &player, race, false);
                let amount = ammo_pickup_amount_for(ammo, fish, haste) + i32::from(haste > 0);
                let cap = player.ammo_cap(ammo);
                let slot = inv.ammo_mut(ammo);
                let gained = amount.min(cap - *slot).max(0);
                *slot += gained;

                if player.free_ammo && gained > 0 {
                    let heal = match player.ultra {
                        Some(
                            UltraMutationId::RobotRefinedTaste | UltraMutationId::RobotRegurgitate,
                        ) => 2,
                        _ => 1,
                    };
                    health.hp = (health.hp + heal).min(health.max);
                    try_recharge_strong_spirit(&mut player, &health);
                    repame_fx::spawn_number(
                        &mut commands,
                        player_pos.x,
                        player_pos.y,
                        heal.to_string(),
                        [0.55, 0.85, 0.95, 1.0],
                    );
                }

                repame_fx::spawn_number(
                    &mut commands,
                    player_pos.x,
                    player_pos.y,
                    gained.to_string(),
                    [0.35, 0.7, 1.0, 1.0],
                );

                let type_name = ammo_type_name(ammo);
                if *slot >= cap {
                    toast.show(&format!("MAX {type_name}"));
                } else {
                    toast.show(&format!("+{gained} {type_name}"));
                }
                audio.play_ammo_pickup(&mut cues);
            }
            PickupKind::Weapon(weapon) => {
                // GML `Player/Collision_WepPickup`: a cursed held gun
                // cannot be swapped for an uncursed one unless a free
                // slot (or invalid bwep) takes it.
                let held_cursed = inv.cursed[inv.current.min(MAX_WEAPON_SLOTS - 1)];
                let pickup_cursed = pickup_curse.is_some();
                let free_slot = first_empty_weapon_slot(&inv).is_some();
                if held_cursed && !pickup_cursed && !free_slot {
                    toast.show("Cursed");
                    // Re-drop: the blanket despawn above already consumed
                    // the entity, but GML leaves the gun on the ground.
                    let e2 = spawn_pickup(
                        &mut commands,
                        &catalog,
                        PickupKind::Weapon(weapon),
                        pickup_pos_value,
                        0,
                        false,
                    );
                    commands
                        .entity(e2)
                        .insert(WepPickupAmmo(wep_ammo.as_deref().is_some_and(|f| f.0)));
                    continue;
                }

                equip_weapon(
                    &mut commands,
                    &catalog,
                    &mut inv,
                    weapon,
                    pickup_pos_value,
                    pickup_cursed,
                );
                // GML `Player/Collision_WepPickup.gml:57-58`:
                // `can_shoot = true; reload = 0` - a gun picked up
                // mid-reload can fire immediately. The port carries the
                // reload on `FireCooldown.timer`.
                if let Some(cd) = fire_cd.as_deref_mut() {
                    cd.timer.reset();
                }
                if matches!(player.ultra, Some(UltraMutationId::RobotRefinedTaste)) {
                    health.hp = (health.hp + 1).min(health.max);
                }
                // GML `Player/Collision_WepPickup.gml:60,66-67`: the
                // cursed sting, then `snd_play(wep_swap[wep])`.
                if pickup_cursed {
                    audio.play_cursed_pickup(&mut cues);
                }
                audio.play_weapon_swap(&mut cues, weapon);
                audio.play_weapon_pickup(&mut cues, weapon);

                // GML `scrPlayerUpdateSameWeaponsFor`: a new weapon resets
                // the mimic clock.
                run.same_weapons_for = 0;
                // GML `GameCont.haspickedweps` (Steroids-C gate).
                run.weapons_picked += 1;

                // GML `Player/Collision_WepPickup:37` verbatim: a
                // successful pickup latches the tutorial PickingUp step.
                if let Some(tut) = tut.as_deref_mut() {
                    tut.complete_step(crate::state::TutorialStep::PickingUp);
                }

                toast.show(&format!("Picked up {}", weapon_id_name(weapon)));
            }
            PickupKind::Chest(_) => {}
            // GML `Curse` motes are ambient (no `Collision_Player`):
            // touching one just clears it.
            PickupKind::Curse => {}
        }
    }
}

/// GML `BigWeaponChest/Collision_Player.gml:16` and
/// `CursedBigChest:17` + `Destroy_0:6` (which spawns a SECOND one).
fn spawn_portal_clear(commands: &mut Commands, pos: glam::Vec2) {
    commands.spawn((
        GameCleanup,
        LevelCleanup,
        PortalClear {
            timer: GTimer::from_seconds(5.0 / 30.0, TimerMode::Once),
            scale: 1.0,
        },
        Pos(pos),
    ));
}

/// GML `scrChestOpened.gml:23-27` / `ProtoChest/Collision_Player.gml:17-21`
/// verbatim: `repeat _amount { with (instance_create(x, y, Rad))
/// motion_add(random_angle, 2 + random(4)) }` at the PLAYER's position.
fn spawn_hatred_rads(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    player_pos: glam::Vec2,
    amount: u32,
    loops: u32,
) {
    let mut rng = rand::rng();
    for _ in 0..amount {
        let ang = rng.random_range(0.0..std::f32::consts::TAU);
        let speed = 2.0 + rng.random_range(0.0..4.0);
        let e = spawn_pickup(
            commands,
            catalog,
            PickupKind::Rad(1),
            player_pos,
            loops,
            false,
        );
        commands.entity(e).insert(GroundPhysics {
            vel: glam::Vec2::from_angle(ang) * speed * 30.0,
            rotspeed: 0.0,
        });
    }
}

/// GML `AmmoChest/Collision_Player.gml:11-21` and
/// `AmmoChestMystery/Collision_Player.gml:14-30` verbatim.
/// `mystery` picks the x3 table and the `scrAmmoDecideTypeMystery`
/// roll (which re-rolls until the type is neither held gun's).
#[allow(clippy::too_many_arguments)]
fn open_ammo_chest(
    commands: &mut Commands,
    inv: &mut Inventory,
    player: &Player,
    race: RaceId,
    player_pos: glam::Vec2,
    fish: u32,
    haste: u32,
    get_loaded: bool,
    mystery: bool,
    toast: &mut Toast,
) {
    let multiplier = if mystery { 3 } else { 2 };
    let mut rng = rand::rng();
    let mut give = |inv: &mut Inventory, kind: AmmoKind, toast: &mut Toast| {
        let amount = ammo_pickup_amount_for(kind, fish, haste) * multiplier;
        let cap = player.ammo_cap(kind);
        let slot = inv.ammo_mut(kind);
        let gained = amount.min(cap - *slot).max(0);
        *slot += gained;
        repame_fx::spawn_number(
            commands,
            player_pos.x,
            player_pos.y,
            gained.to_string(),
            [0.35, 0.7, 1.0, 1.0],
        );
        let type_name = ammo_type_name(kind);
        toast.show(&if *slot >= cap {
            format!("MAX {type_name}")
        } else {
            format!("+{gained} {type_name}")
        });
    };

    if get_loaded {
        let a = decide_ammo_type(inv, player, race, true);
        let b = decide_ammo_type(inv, player, race, true);
        for kind in ALL_AMMO_KINDS {
            // GML `AmmoChestMystery:19` skips the two rolled types.
            if mystery && (kind == a || kind == b) {
                continue;
            }
            give(inv, kind, toast);
        }
    } else {
        let kind = if mystery {
            decide_ammo_type_mystery(inv, player, &mut rng)
        } else {
            decide_ammo_type(inv, player, race, true)
        };
        give(inv, kind, toast);
    }
}

/// Chest-open state flip (sim half): the pickup becomes an opened chest
/// carrying its kind, so the renderer can swap to the kind-specific open
/// art (`open_chest` sprite swap, renderer-owned). The idle
/// `GmlImage` ramp is dropped so the closed strip cannot draw over the
/// open art.
pub fn open_chest(commands: &mut Commands, e: Entity, kind: ChestKind) {
    commands.entity(e).remove::<Pickup>();
    commands.entity(e).remove::<GmlImage>();
    commands.entity(e).insert(OpenedChest(kind));
}

/// Chest-open entry used by gameplay callers (mines, scripted opens).
/// Gameplay half: currently the same state flip as [`open_chest`].
pub fn open_chest_shock(commands: &mut Commands, e: Entity, kind: ChestKind) {
    open_chest(commands, e, kind);
}

/// Headless weapon-label state: the nearest in-range weapon name for the HUD
/// bridge to poll, plus the interact-prompt half of `scrDrawInteractionHUD`:
/// the `sprEPickup` icon, the per-type ammo gauge (`scrDrawTypeAmmo`,
/// `_player.ammo[type] / capacity`) and `prop_prompts`, the per-type
/// `Prompt{Object}` hits over `CarVenusFixed` / `IceFlower` / `Van`
/// (`scrDrawPlayerHUD.gml:374`), whose overlap also raises the act button
/// (`:402`).
#[derive(Resource, Default, Debug)]
pub struct WeaponLabel {
    pub text: String,
    pub target: Option<Entity>,
    pub prop_prompts: Vec<(Entity, &'static str)>,
    /// GML `draw_pickup_button` (`draw_gamepad_button.gml:43-101`): the
    /// keyboard "E" pill is `sprEPickup` frame 0, the gamepad/other-key
    /// plate is frame 1. Subimage 0 in single-player (no gamepad
    /// binding), so the port draws frame 0.
    pub button_frame: u32,
    /// GML `scrDrawTypeAmmo` art pair for the prompt gauge, `None` for a
    /// melee gun (`if (type == Ammo.None) _text_offset = 0`).
    pub ammo_gauge: Option<(&'static str, &'static str)>,
    /// `_percentage = _player.ammo[type] / _type_capacity`.
    pub ammo_fill: f32,
}

pub fn sync_weapon_label(
    player_q: Query<(&Pos, &Player), With<Player>>,
    inv_q: Query<&Inventory>,
    weapon_q: Query<(Entity, &Pos, &Pickup), (Without<Player>, Without<Portal>)>,
    prop_q: Query<(Entity, &Pos, &Prop, &PropSprites)>,
    enemy_q: Query<(Entity, &Pos, &Enemy)>,
    van_q: Query<(Entity, &Pos, &IdpdVanDeploy), Without<IdpdVanDeployed>>,
    mut label: ResMut<WeaponLabel>,
    mut act: ResMut<crate::state::ActButton>,
) {
    let Some((player_pos, player)) = player_q.single().ok() else {
        return;
    };
    let player_pos = player_pos.0;
    let inv = inv_q.single().ok();

    let mut prop_prompts = Vec::new();
    if let Some(hit) = nearest_prompt_hit(
        player_pos,
        prop_q.iter().filter_map(|(entity, pos, _, sprites)| {
            matches!(
                sprites.idle,
                "images/sprVenusCarFixed.png" | "images/sprVenuzCar2.png"
            )
            .then_some((
                entity,
                pos.0,
                if sprites.flip_x {
                    CAR_PROMPT_SPAN_FLIPPED
                } else {
                    CAR_PROMPT_SPAN
                },
            ))
        }),
    ) {
        prop_prompts.push((hit, "CAR"));
    }
    if let Some(hit) = nearest_prompt_hit(
        player_pos,
        enemy_q
            .iter()
            .filter(|(_, _, enemy)| enemy.kind == EnemyKind::IceFlower)
            .map(|(entity, pos, _)| (entity, pos.0, ICE_FLOWER_PROMPT_SPAN)),
    ) {
        prop_prompts.push((hit, "FEED"));
    }
    if let Some(hit) = nearest_prompt_hit(
        player_pos,
        van_q
            .iter()
            .filter(|(_, _, deploy)| {
                !deploy.freak
                    && ((deploy.frames > 0.0 && deploy.frames <= VAN_DEPLOY_FRAMES - 40.0)
                        || deploy.inert > 0.0)
            })
            .map(|(entity, pos, _)| (entity, pos.0, VAN_PROMPT_SPAN)),
    ) {
        prop_prompts.push((hit, "VAN"));
    }
    if !prop_prompts.is_empty() {
        act.active = true;
    }
    label.prop_prompts = prop_prompts;

    let target = nearest_ground_weapon(
        player_pos,
        weapon_q
            .iter()
            .map(|(e, pos, pickup)| (e, pos.0, pickup.kind)),
    );

    if target == label.target {
        return;
    }
    label.target = target;
    let Some(weapon) = target.and_then(|e| {
        let (_, _, pickup) = weapon_q.get(e).ok()?;
        let PickupKind::Weapon(w) = pickup.kind else {
            return None;
        };
        Some(w)
    }) else {
        label.text.clear();
        label.ammo_gauge = None;
        label.ammo_fill = 0.0;
        return;
    };
    label.text = weapon_id_name(weapon).to_string();
    label.button_frame = 0;
    let kind = weapon_ammo(weapon);
    label.ammo_gauge = ammo_gauge_art(kind);
    label.ammo_fill = match (inv, kind) {
        (Some(_), AmmoKind::None) => 0.0,
        (Some(inv), kind) => {
            let cap = player.ammo_cap(kind).max(1);
            (inv.ammo_of(kind) as f32 / cap as f32).clamp(0.0, 1.0)
        }
        _ => 0.0,
    };
}

/// GML `scrDrawTypeAmmo` (`scrDrawPlayerHUD.gml:304-336`) art pair.
pub fn ammo_gauge_art(kind: AmmoKind) -> Option<(&'static str, &'static str)> {
    Some(match kind {
        AmmoKind::Bullets => ("images/sprBulletIconBG.png", "images/sprBulletIcon.png"),
        AmmoKind::Shells => ("images/sprShotIconBG.png", "images/sprShotIcon.png"),
        AmmoKind::Bolts => ("images/sprBoltIconBG.png", "images/sprBoltIcon.png"),
        AmmoKind::Explosives => ("images/sprExploIconBG.png", "images/sprExploIcon.png"),
        AmmoKind::Energy => ("images/sprEnergyIconBG.png", "images/sprEnergyIcon.png"),
        AmmoKind::None => return None,
    })
}

pub fn ammo_type_name(kind: AmmoKind) -> &'static str {
    match kind {
        AmmoKind::None => "NONE",
        AmmoKind::Bullets => "BULLETS",
        AmmoKind::Shells => "SHELLS",
        AmmoKind::Bolts => "BOLTS",
        AmmoKind::Explosives => "EXPLOSIVES",
        AmmoKind::Energy => "ENERGY",
    }
}

/// GML `enum Ammo` order (`scrAmmoInit.gml:1-9`): the `for` loops walk
/// `Ammo.Bullets .. < NUM_AMMO_TYPES`.
pub const ALL_AMMO_KINDS: [AmmoKind; 5] = [
    AmmoKind::Bullets,
    AmmoKind::Shells,
    AmmoKind::Bolts,
    AmmoKind::Explosives,
    AmmoKind::Energy,
];

/// GML `scrAmmoDecideType(_player, _prioritize_primary)` verbatim
/// (`scrAmmoInit.gml:34-67`):
///
/// * BigDog or no held gun at all -> `irandom_range(Bullets, Energy)`.
/// * the `extra_weps` re-roll replaces the secondary's type at a
///   `1 - 1/(_extra_count + 1)` rate (GML indexes 1-based with
///   `irandom(_extra_count - 1)`, so index 0 is an empty slot and the LAST extra
///   weapon is unreachable - kept verbatim);
/// * `_prioritize_primary || !bwep` walks primary then secondary; when BOTH are
///   live and the primary is full it splits 50/50 with `choose(_atype, _btype)`;
/// * otherwise (two live guns, no priority) it rolls `choose(_atype, _btype)`
///   FIRST and falls through to a flat random type if that pick is full.
fn decide_ammo_type(
    inv: &Inventory,
    player: &Player,
    race: RaceId,
    prioritize_primary: bool,
) -> AmmoKind {
    if race == RaceId::BigDog
        || (weapon_ammo(inv.weapons[inv.current]) == AmmoKind::None
            && weapon_ammo(inv.weapons[1.min(inv.weapon_slots.saturating_sub(1))])
                == AmmoKind::None)
    {
        return random_ammo_kind(&mut rand::rng());
    }

    let mut rng = rand::rng();
    let primary = weapon_ammo(inv.weapons[inv.current]);
    let mut secondary = weapon_ammo(inv.weapons[1.min(inv.weapon_slots.saturating_sub(1))]);
    let extras: Vec<WeaponId> = (2..inv.weapon_slots)
        .map(|i| inv.weapons[i])
        .filter(|w| *w != WeaponId::NONE)
        .collect();
    if !extras.is_empty() && rng.random::<f32>() > 1.0 / (extras.len() as f32 + 1.0) {
        // GML `extra_weps[irandom(_extra_count - 1)]`: index 0 reads past
        // the 1-based array and yields weapon 0 (`Ammo.None`).
        secondary = match rng.random_range(0..extras.len()) {
            0 => AmmoKind::None,
            i => weapon_ammo(extras[i - 1]),
        };
    }

    let has_secondary =
        weapon_ammo(inv.weapons[1.min(inv.weapon_slots.saturating_sub(1))]) != AmmoKind::None;
    let has_room =
        |kind: AmmoKind| kind != AmmoKind::None && inv.ammo_of(kind) < player.ammo_cap(kind);

    if prioritize_primary || !has_secondary {
        if has_room(primary) {
            return primary;
        }
        if has_room(secondary) {
            return if primary != AmmoKind::None && rng.random_bool(0.5) {
                primary
            } else {
                secondary
            };
        }
    } else if has_room(primary) || has_room(secondary) {
        let pick = if rng.random_bool(0.5) {
            primary
        } else {
            secondary
        };
        if has_room(pick) {
            return pick;
        }
    }
    random_ammo_kind(&mut rng)
}

/// GML `scrAmmoDecideTypeMystery` (`scrAmmoInit.gml:71-82`): re-roll a
/// random type until it is not the type of `wep` or `bwep`.
fn decide_ammo_type_mystery(
    inv: &Inventory,
    _player: &Player,
    rng: &mut impl rand::RngExt,
) -> AmmoKind {
    let primary = weapon_ammo(inv.weapons[inv.current]);
    let secondary = weapon_ammo(inv.weapons[1.min(inv.weapon_slots.saturating_sub(1))]);
    loop {
        let kind = random_ammo_kind(rng);
        if kind != primary && kind != secondary {
            return kind;
        }
    }
}

fn first_empty_weapon_slot(inv: &Inventory) -> Option<usize> {
    (0..inv.weapon_slots).find(|&i| inv.weapons[i] == WeaponId::NONE)
}

/// GML `Player/Collision_WepPickup.gml:44-47` verbatim: the swapped-out
/// gun re-drops at `other.x, other.y` (the GUN's position, not the
/// player's) with `_has_ammo` defaulting to false, keeping its curse.
fn spawn_dropped_weapon(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    weapon: WeaponId,
    at: glam::Vec2,
    curse: bool,
) {
    let e = spawn_pickup(commands, catalog, PickupKind::Weapon(weapon), at, 0, false);
    commands.entity(e).insert(WepPickupAmmo(false));
    if curse {
        commands.entity(e).insert(PickupCurse);
    }
}

/// GML `snd_chst` = `scr_race_get_sound(race, "Chst", sndMutant1Chst)`
/// (`scripts/scrPlayerCreate/scrPlayerCreate.gml:53`): the race's
/// `sndMutant<n>Chst` stem, falling back to `sndMutant1Chst`.
pub fn chst_stem(race: RaceId) -> &'static str {
    const STEMS: [&'static str; 16] = [
        "sndMutant1Chst",
        "sndMutant2Chst",
        "sndMutant3Chst",
        "sndMutant4Chst",
        "sndMutant5Chst",
        "sndMutant6Chst",
        "sndMutant7Chst",
        "sndMutant8Chst",
        "sndMutant9Chst",
        "sndMutant10Chst",
        "sndMutant11Chst",
        "sndMutant12Chst",
        "sndMutant13Chst",
        "sndMutant14Chst",
        "sndMutant15Chst",
        "sndMutant16Chst",
    ];
    STEMS.get(race as usize).copied().unwrap_or(STEMS[0])
}

fn equip_weapon(
    commands: &mut Commands,
    catalog: &repame_anim::AnimCatalog,
    inv: &mut Inventory,
    weapon: WeaponId,
    at: glam::Vec2,
    curse: bool,
) {
    if let Some(empty) = first_empty_weapon_slot(inv) {
        inv.weapons[empty] = weapon;
        inv.cursed[empty] = curse;
        inv.current = empty;
        return;
    }

    let slot = inv.current;
    let dropped = inv.weapons[slot];
    let dropped_curse = inv.cursed[slot];
    if dropped != WeaponId::NONE {
        spawn_dropped_weapon(commands, catalog, dropped, at, dropped_curse);
    }
    inv.weapons[slot] = weapon;
    inv.cursed[slot] = curse;
}

/// GML `Player/Collision_WepPickup.gml:90-104` verbatim - the block sits
/// OUTSIDE the pick `if/else`, so mask overlap alone pays it (press or not,
/// nearest gun or not) and `other.ammo` is always consumed. The Protection
/// crown converts the payout into a `1 + scr_skill_get(mut_second_stomach)`
/// heal plus a `HealFX`; otherwise it is `scrAmmoGetPickupAmount(_type) * 2`.
fn pay_weapon_pickup_ammo(
    commands: &mut Commands,
    inv: &mut Inventory,
    weapon: WeaponId,
    player_pos: glam::Vec2,
    player: &Player,
    health: &mut Health,
) {
    let kind = weapon_ammo(weapon);
    if kind == AmmoKind::None {
        return;
    }
    if player.crown == CrownKind::Protection {
        let amount = 1 + i32::from(player.mutations.contains(&MutationId::SecondStomach));
        health.hp = (health.hp + amount).min(health.max);
        repame_fx::spawn_number(
            commands,
            player_pos.x,
            player_pos.y,
            amount.to_string(),
            [0.3, 1.0, 0.3, 1.0],
        );
        return;
    }
    let add = ammo_pickup_amount_for(kind, 0, haste_crown(player)) * 2;
    let cap = player.ammo_cap(kind);
    let slot = inv.ammo_mut(kind);
    let gained = add.min(cap - *slot).max(0);
    *slot += gained;
    repame_fx::spawn_number(
        commands,
        player_pos.x,
        player_pos.y,
        gained.to_string(),
        [0.35, 0.7, 1.0, 1.0],
    );
}

/// Rad-container contact. GML `RadChest/Collision_Player.gml` verbatim
/// is `if !scrChestOpened() { GameCont.noradch = 0; hp = 0 }`, and
/// `prop/Destroy_0.gml:12` turns the inherited `raddrop = 25`
/// (`RadChest/Create_0.gml:20`) into `scrRadDrop(x, y, 25)`.
pub fn tick_rad_container_contact(
    mut commands: Commands,
    catalog: Res<repame_anim::AnimCatalog>,
    audio: Res<GameAudio>,
    mut cues: ResMut<Queue<AudioCue>>,
    mut run: ResMut<Run>,
    player_q: Query<(&Pos, &Player), With<Player>>,
    mut rad_q: Query<(Entity, &Pos, &Prop), With<RadChestContainer>>,
) {
    let Ok((player_pos, player)) = player_q.single() else {
        return;
    };
    let player_pos = player_pos.0;
    let loops = run.loop_count;
    let hasted = haste_crown(player) > 0;
    for (e, pos, prop) in &mut rad_q {
        let center = pos.0;
        let half = prop.size * 0.5;

        let closest = glam::Vec2::new(
            player_pos.x.clamp(center.x - half.x, center.x + half.x),
            player_pos.y.clamp(center.y - half.y, center.y + half.y),
        );
        if player_pos.distance(closest) > crate::comps_a::PLAYER_RADIUS + 2.0 {
            if player_pos.distance(center) > half.x + crate::comps_a::PLAYER_RADIUS + 4.0 {
                continue;
            }
        }

        commands.entity(e).try_despawn();
        run.noradch = 0;
        scr_rad_drop(
            &mut commands,
            &catalog,
            center,
            25,
            loops,
            hasted,
            true,
            true,
        );
        // GML `RadChest/Destroy_0.gml:11-12`: `sndEXPChest`, never the
        // generic pickup blip.
        audio.play_exp_chest(&mut cues);
    }
}
