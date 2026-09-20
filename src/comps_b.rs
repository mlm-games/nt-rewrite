use super::comps_a::{DamageSource, Team};
use crate::data::*;
use crate::time::{GTimer as Timer, TimerMode};
use bevy_ecs::prelude::*;
use glam::Vec2;

#[derive(Component, Clone, Copy, Debug)]
pub struct DeploysSentry {
    pub life: f32,
    pub fire_interval: f32,
    pub range: f32,
    pub projectile_speed: f32,
    pub projectile_damage: i32,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct CustomExplosion {
    pub radius: f32,
    /// GML multi-spawn pattern: `count` Explosion instances at `spread` px offsets
    /// (e.g. Nuke 8x @12px, Sticky-stuck 3x @16px). Single-circle when count <= 1.
    pub count: u8,
    pub spread: f32,
}

impl Default for CustomExplosion {
    fn default() -> Self {
        Self {
            radius: 32.0,
            count: 1,
            spread: 0.0,
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct BloodAmmo {
    pub hp_cost: i32,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct SpawnsWeaponPickup {
    pub weapon: Option<WeaponId>,
    /// GML `scrDecideWep` tier extra when `weapon` is `None` (GUN GUN
    /// fires `scrDecideWep(10)`); ignored for fixed weapons.
    pub decide_extra: i32,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct PlasmaBurst {
    pub pellets: u8,
    pub speed: f32,
    pub damage: i32,
    pub lifetime: f32,
    pub radius: f32,
    pub knockback: f32,
    pub color: [f32; 4],
    pub size: Vec2,
}

#[derive(Component, Clone, Debug)]
pub struct Beam {
    pub team: Team,
    pub dir: Vec2,
    pub length: f32,
    pub width: f32,
    pub damage: i32,
    pub knockback: f32,
    /// Sprite tint (bevy `BeamSpec.color` parity: ion cyan, laser red,
    /// boss orange/green; alpha rides along).
    pub color: [f32; 4],
    pub timer: Timer,
    pub tick: Timer,
    pub source: Option<DamageSource>,
}

#[derive(Component)]
pub struct IdpdVanBrain {
    pub deploy_timer: Timer,
    pub charges_left: u8,
}

impl Default for IdpdVanBrain {
    fn default() -> Self {
        Self {
            deploy_timer: Timer::from_seconds(2.2, TimerMode::Repeating),
            charges_left: 4,
        }
    }
}

#[derive(Component)]
pub struct IdpdShieldUnit;

#[derive(Resource)]
pub struct IdpdRaidState {
    pub cooldown: Timer,
    pub warning: Timer,
    pub pending_wave: Option<RaidWave>,
    pub wave_index: u32,
    pub kills_checkpoint: u32,
}

impl Default for IdpdRaidState {
    fn default() -> Self {
        Self {
            cooldown: Timer::from_seconds(20.0, TimerMode::Repeating),
            warning: Timer::from_seconds(1.25, TimerMode::Once),
            pending_wave: None,
            wave_index: 0,
            kills_checkpoint: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaidWave {
    Light,
    Medium,
    Heavy,
    VanDrop,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CampfirePhase {
    Sitting,

    WaitingForIdpd,

    Rising,

    SpawnThroneII,
}

#[derive(Component)]
pub struct CampfireState {
    pub phase: CampfirePhase,

    pub timer: Timer,

    pub idpd_clear_confirm: Timer,

    pub idpd_gate_armed: bool,

    pub spawned_throne_ii: bool,
}

impl CampfireState {
    pub fn new() -> Self {
        Self {
            phase: CampfirePhase::Sitting,
            timer: Timer::from_seconds(3.5, TimerMode::Once),
            idpd_clear_confirm: Timer::from_seconds(0.35, TimerMode::Once),
            idpd_gate_armed: false,
            spawned_throne_ii: false,
        }
    }

    pub fn set_phase(&mut self, phase: CampfirePhase, seconds: f32) {
        self.phase = phase;
        self.timer = Timer::from_seconds(seconds.max(0.01), TimerMode::Once);
        self.timer.reset();
    }

    pub fn arm_idpd_gate(&mut self) {
        self.phase = CampfirePhase::WaitingForIdpd;
        self.idpd_gate_armed = true;
        self.idpd_clear_confirm.reset();
    }

    pub fn reset_idpd_clear_confirmation(&mut self) {
        self.idpd_clear_confirm.reset();
    }
}

#[derive(Component)]
pub struct CampfireProp;

/// Coop downed marker (GML `objects/Revive`: `Create_0` + `Step_0` +
/// `Alarm_4/5`). `alarm4` counts the 300-step grace (`Step_0` re-pins it
/// to 300 while level generation runs, i.e. `GenCont`/`LevCont`
/// exist); when it fires the 30-step hurt pulse starts (`Alarm_4`
/// performs `Alarm_5`: every player takes 1, `alarm[5] = 30`).
/// The port has no coop downing (single-player only), so nothing
/// spawns or damages through this yet — the comp carries the timer
/// law for the HUD draw below.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Revive {
    /// GML `alarm[4]`: grace steps left (300 at spawn).
    pub alarm4: f32,
    /// GML `alarm[5]`: hurt-pulse steps left (30 once armed, else 0).
    pub alarm5: f32,
}

impl Revive {
    pub fn new() -> Self {
        Self {
            alarm4: 300.0,
            alarm5: 0.0,
        }
    }
}

impl Default for Revive {
    fn default() -> Self {
        Self::new()
    }
}

/// One downed-timer step (GML `Revive/Step_0` + `Alarm_4/5` verbatim,
/// damage application excluded): generation holds `alarm4` at 300;
/// otherwise `alarm4` drains, arming `alarm5 = 30` at zero; `alarm5`
/// then re-arms at 30 every time it drains (each re-arm is one GML
/// hurt pulse). `steps` is timescale steps (`delta * 30`).
/// Returns true on the tick a hurt pulse fires.
pub fn revive_step(revive: &mut Revive, steps: f32, generating: bool) -> bool {
    if generating {
        revive.alarm4 = 300.0;
        return false;
    }
    if revive.alarm4 > 0.0 {
        revive.alarm4 = (revive.alarm4 - steps).max(0.0);
        if revive.alarm4 <= 0.0 {
            revive.alarm5 = 30.0;
        }
        return false;
    }
    if revive.alarm5 > 0.0 {
        revive.alarm5 -= steps;
        if revive.alarm5 <= 0.0 {
            revive.alarm5 = 30.0;
            return true;
        }
    }
    false
}

/// Yung Venuz couch sitter (GML `objects/YungVenuzCouch`, crib: the
/// campfire YV visual; sprite `sprYVBossGamingIdle`, one-shot
/// `sprYVBossGamingAirhorn`). `frame` is the fractional `image_index`.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct YvCouch {
    /// True while the airhorn one-shot plays.
    pub airhorn: bool,
    /// Fractional frame (GML `image_index`).
    pub frame: f32,
}

impl YvCouch {
    pub fn idle() -> Self {
        Self {
            airhorn: false,
            frame: 0.0,
        }
    }

    /// Start the airhorn one-shot (GML `sprite_index =
    /// sprYVBossGamingAirhorn` from the interact; the Step law below
    /// returns to idle on animation end).
    pub fn play_airhorn(&mut self) {
        self.airhorn = true;
        self.frame = 0.0;
    }

    pub fn sprite_path(&self) -> &'static str {
        if self.airhorn {
            "images/sprYVBossGamingAirhorn.png"
        } else {
            "images/sprYVBossGamingIdle.png"
        }
    }
}

/// One couch animation step (GML `YungVenuzCouch/Step_0` verbatim:
/// `image_speed = timescale * 0.4`, airhorn one-shot returns to idle
/// on animation end; idle loops like every GML sprite).
/// `steps` is timescale steps (`delta * 30`); `fps`/`frames` are the
/// live strip's catalog values (GML normalizes `image_speed` by
/// `sprite_fps / room_speed`, room 30 Hz).
pub fn yv_couch_step(couch: &mut YvCouch, steps: f32, fps: f32, frames: u32) {
    couch.frame += steps * 0.4 * fps.max(0.0) / 30.0;
    let frames = frames.max(1) as f32;
    if couch.airhorn {
        if couch.frame >= frames {
            couch.airhorn = false;
            couch.frame = 0.0;
        }
    } else if couch.frame >= frames {
        couch.frame -= frames * (couch.frame / frames).floor();
    }
}

#[derive(Resource, Clone, Debug, Default)]
pub struct LoopTransition {
    pub campfire_active: bool,
    pub throne_ii_alive: bool,
    pub loop_ready: bool,
    pub last_completed_loop: u32,
}

impl LoopTransition {
    pub fn blocks_portal(&self) -> bool {
        self.campfire_active || self.throne_ii_alive
    }

    pub fn blocks_new_idpd_raids(&self) -> bool {
        self.campfire_active || self.throne_ii_alive || self.loop_ready
    }

    pub fn begin_campfire(&mut self) {
        self.campfire_active = true;
        self.throne_ii_alive = false;
        self.loop_ready = false;
    }

    pub fn throne_ii_spawned(&mut self) {
        self.campfire_active = false;
        self.throne_ii_alive = true;
        self.loop_ready = false;
    }

    pub fn throne_ii_defeated(&mut self) {
        self.campfire_active = false;
        self.throne_ii_alive = false;
        self.loop_ready = true;
    }

    pub fn consume_loop_ready(&mut self) -> bool {
        let ready = self.loop_ready;
        self.loop_ready = false;
        ready
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct PendingDelayedBoss {
    pub kind: EnemyKind,
    pub initial_trash: u32,
    pub kill_fraction: f32,

    pub from_wall: bool,
}

impl PendingDelayedBoss {
    pub fn kills_needed(&self) -> u32 {
        (((self.initial_trash as f32) * self.kill_fraction).ceil() as u32).max(1)
    }
}

#[derive(Component)]
pub struct HyperOrbitCrystal {
    pub owner: Entity,
    pub angle: f32,
    pub radius: f32,
    pub angular_speed: f32,
    pub fire_timer: Timer,
}

#[derive(Component, Clone, Debug)]
pub struct SentryTurret {
    pub life: Timer,
    pub fire: Timer,
    pub range: f32,
    pub projectile_speed: f32,
    pub projectile_damage: i32,
}

#[derive(Component, Clone, Copy)]
pub struct Enemy {
    pub kind: EnemyKind,
    pub score: u32,
    pub touch_damage: i32,
    pub rad_drop: usize,
    pub drop_chance: usize,
    pub weapon_chance: usize,
}

#[derive(Component)]
pub struct EnemyBrain {
    pub speed: f32,
    pub accel: f32,
    pub preferred_range: f32,
    pub shoot_range: f32,
    pub attack: Timer,
    pub burst_left: usize,
    pub burst_timer: Timer,
    pub dash: f32,
    pub strafe_dir: f32,
    pub strafe_timer: Timer,
    pub melee: Timer,

    pub walk: f32,

    pub ammo: u8,

    pub slash_delay: f32,

    pub gunangle: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BossPhase {
    Idle,
    Telegraph,
    Charging,
    Cooldown,
    Jumping,
    Landing,
    Radial,
    Beam,
    Enraged,

    Spawning,
    Teleport,
    CarpetBeam,
}

#[derive(Component)]
pub struct BossBrain {
    pub phase: BossPhase,
    pub phase_timer: Timer,
    pub attack_timer: Timer,
    pub special_timer: Timer,
    pub pattern_index: usize,
    pub home: Vec2,
    pub target: Vec2,
    pub enraged: bool,
    /// GML scratch register (YV `minigun_side`, Throne `mode`, ...).
    /// Zero means "uninitialized" for bosses that self-init.
    pub aux: f32,
    /// GML `sndtaunt` (fired-once latch for the no-player taunt line).
    pub taunt: bool,
    /// GML `tauntdelay` (ticks without a live Player before taunting;
    /// YV resets it when the player exists).
    pub tauntdelay: u32,
}

impl BossBrain {
    pub fn new(kind: EnemyKind, spawn: Vec2) -> Self {
        let (attack, special) = match kind {
            EnemyKind::BigBandit | EnemyKind::BigBanditLoop => {
                if kind == EnemyKind::BigBanditLoop {
                    (0.95, 2.25)
                } else {
                    (1.15, 2.8)
                }
            }
            EnemyKind::BigDog | EnemyKind::BigDogLoop => {
                if kind == EnemyKind::BigDogLoop {
                    (0.62, 1.8)
                } else {
                    (0.8, 2.2)
                }
            }
            EnemyKind::LilHunter | EnemyKind::LilHunterLoop => {
                if kind == EnemyKind::LilHunterLoop {
                    (0.42, 1.35)
                } else {
                    (0.55, 1.7)
                }
            }
            EnemyKind::Throne => (0.7, 2.5),
            EnemyKind::ThroneII => (0.85, 3.4),
            EnemyKind::Hyper => (1.1, 4.0),
            EnemyKind::Mom => (1.0, 2.4),
            EnemyKind::Technomancer => (2.0, 3.5),
            EnemyKind::Captain => (0.7, 2.0),
            EnemyKind::OldGuardian => (0.9, 2.2),
            // GML `YVBoss/Create_0`: `alarm[1] = 20`, `alarm[2] = 8`.
            EnemyKind::YvBoss => (20.0 / 30.0, 8.0 / 30.0),
            _ => (1.2, 3.0),
        };

        Self {
            phase: BossPhase::Idle,
            phase_timer: Timer::from_seconds(0.1, TimerMode::Once),
            attack_timer: Timer::from_seconds(attack, TimerMode::Repeating),
            special_timer: Timer::from_seconds(special, TimerMode::Repeating),
            pattern_index: 0,
            home: spawn,
            target: spawn,
            enraged: false,
            aux: 0.0,
            taunt: false,
            tauntdelay: 0,
        }
    }

    pub fn set_phase(&mut self, phase: BossPhase, seconds: f32) {
        self.phase = phase;
        self.phase_timer = Timer::from_seconds(seconds.max(0.01), TimerMode::Once);
        self.phase_timer.reset();
    }
}

#[derive(Component)]
pub struct Pickup {
    pub kind: PickupKind,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct WepPickupAmmo(pub bool);

#[derive(Clone, Copy)]
pub enum PickupKind {
    Rad(u32),
    Medkit(i32),
    Ammo(AmmoKind, i32),
    Weapon(WeaponId),
    Chest(ChestKind),
    /// GML `Curse` mote (Robot's cursed-weapon eat spills 10; ambient,
    /// no pickup effect — collected by despawn).
    Curse,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChestKind {
    Weapon,
    Ammo,
    Rad,
    /// GML `HealthChest`: heals 4 (8 with Second Stomach).
    Health,
    /// GML `CursedBigChest`: 3 (4 Ambidextrous) cursed weapons +
    /// `PortalClear` (cursed caves).
    CursedBig,
    /// GML `RogueChest`: 25 rads for non-Rogue, `RogueAmmo` for Rogue.
    Rogue,
    /// GML `ProtoChest`: the vault prototype weapon (port: random
    /// weapon; Hatred burns 1 HP + 16 rads like GML).
    Proto,
    /// GML `BigWeaponChest`: 3 (4 Ambidextrous) weapons + `PortalClear`,
    /// resets `nochest`.
    BigWeapon,
    /// GML `RadChestBig`: big rad cache (port E-open: 45 rads).
    RadBig,
    /// GML `RadMaggotChest`: trapped cache (port E-open: detonates).
    RadMaggot,
    /// GML `IDPDChest`: IDPD cache (port E-open: 8 ammo pickups).
    Idpd,
}

/// Marker for cursed weapon pickups (GML `WepPickup.curse`): skipped by
/// Robot's portal-entry eat; full curse mechanics (no-drop, reminders)
/// ride the curse-system phase.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PickupCurse;

impl From<WeaponKind> for PickupKind {
    fn from(k: WeaponKind) -> Self {
        PickupKind::Weapon(k.into())
    }
}

#[derive(Component)]
pub struct Portal;

/// GML Portal state machine (objects/Portal/*):
/// Spawn (sprPortalSpawn) → Idle (sprPortal/Popo/Proto by type) → Disappear.
/// `kind`: 1 normal, 2 popo (HQ), 3 proto (Vault).
/// `anim` stands in for bevy's oneshot sprite anim: Spawn lasts the
/// `sprPortalSpawn` strip (2 frames @ 8 fps = 0.25 s), Disappear lasts
/// `sprPortalDisappear` (9 frames @ 12 fps = 0.75 s).
#[derive(Component, Clone, Copy, Debug)]
pub struct PortalState {
    pub kind: u8,
    pub phase: PortalPhase,
    pub endgame: f32,
    pub close: bool,
    pub anim: Timer,
    /// Edge state for the `sndPortalLoop` cue (GML `Portal/Other_7`
    /// starts it, `Alarm_1`/`CleanUp_0`/`Other_5` stop it).
    pub loop_on: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PortalPhase {
    Spawn,
    Idle,
    Disappear,
}

impl PortalState {
    pub fn idle_sprite(kind: u8) -> &'static str {
        match kind {
            2 => "images/sprPopoPortal.png",
            3 => "images/sprProtoPortal.png",
            _ => "images/sprPortal.png",
        }
    }

    pub fn disappear_sprite(kind: u8) -> &'static str {
        match kind {
            2 => "images/sprPopoPortalDisappear.png",
            3 => "images/sprProtoPortalDisappear.png",
            _ => "images/sprPortalDisappear.png",
        }
    }

    pub fn spawn_sprite() -> &'static str {
        "images/sprPortalSpawn.png"
    }
}

/// GML `CrownObject` (spawned by `scrCrownSetCurrent` while a crown is
/// held: follows the player, `refresh` is the Alarm-2 heartbeat marker).
#[derive(Component, Clone, Copy, Debug)]
pub struct CrownObject {
    pub refresh: u32,
}

/// GML Chicken-throw flung weapon (`scrWeaponPickupCreate` + `motion_set`
/// gunangle±2 speed 16, `team`/`creator` set; Determination ultra arms
/// the 60-tick `alarm[1]` return).
#[derive(Component, Clone, Copy, Debug)]
pub struct FlungWeapon {
    pub team: Team,
    pub creator: Entity,
    pub return_ticks: u8,
}

/// GML Cuz cry-sprite swap (`sprite_index = spr_cry` on fire): headless
/// marker with the swap lifetime so the render phase can show it.
#[derive(Component, Clone, Copy, Debug)]
pub struct CryAnim {
    pub timer: Timer,
}

/// GML `LilHunterDie` (`LilHunterDie/Create_0` + `Step_0`): skittering
/// hunter head. `trn` is the per-step turn drift, `bounces` the wall-hit
/// count (past 3 it spins out), `target` the chased player if any.
#[derive(Component, Clone, Copy, Debug)]
pub struct LilHunterDie {
    pub trn: f32,
    pub bounces: u8,
    pub target: Option<Entity>,
}

/// GML `TrapFire` marker (`sprFireLilHunter` step-0 ring / death ring):
/// short-lived fire left by LilHunter.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct TrapFire;

#[derive(Component)]
pub struct PortalShock {
    pub timer: Timer,
    pub radius: f32,
}

#[derive(Component)]
pub struct PortalClear {
    pub timer: Timer,
    /// GML `image_xscale/yscale` (1.0 everywhere except the chicken-TV
    /// arm's half-scale pair in `scrCampfireMenuCreate`).
    pub scale: f32,
}

#[derive(Component)]
pub struct PortalClosing {
    pub timer: Timer,
}

#[derive(Resource, Default)]
pub struct PortalCarriedWeapons(pub Vec<WeaponId>);

#[derive(Component)]
pub struct HurtAnim {
    pub idle: &'static str,
    pub walk: Option<&'static str>,
    pub hurt: &'static str,
    pub timer: Timer,
    pub was_moving: bool,
}

#[derive(Component)]
pub struct FireAnim {
    pub idle: &'static str,
    pub walk: Option<&'static str>,
    pub timer: Timer,
}

/// Opened-chest marker carrying its kind (bevy swaps the chest sprite
/// to the kind-specific open art in `open_chest`; the renderer maps
/// kind -> open strip here).
#[derive(Component)]
pub struct OpenedChest(pub ChestKind);

#[derive(Component)]
pub struct PickupLifetime {
    pub timer: Timer,
}

#[derive(Component)]
pub struct GroundPhysics {
    pub vel: Vec2,
    pub rotspeed: f32,
}

#[derive(Component)]
pub struct PortalSucking {
    pub portal: Entity,
    pub timer: Timer,
    pub start_pos: Vec2,
    pub target_pos: Vec2,
}

#[derive(Component, Clone, Copy)]
pub struct WeaponVisual {
    pub owner: Entity,
    pub wkick: f32,
    pub wep_id: WeaponId,
    pub wep_angle: f32,

    pub slot: u8,
}

#[derive(Component, Clone, Copy)]
pub struct WeaponVisualOwner;

#[derive(Component, Clone, Copy)]
pub struct EnemySprites {
    pub idle: &'static str,
    pub walk: Option<&'static str>,
    pub hurt: &'static str,
}

#[derive(Component)]
pub struct Prop {
    pub size: Vec2,
    pub hp: i32,
    pub destructible: bool,
    pub explosive: bool,
}

#[derive(Component, Clone, Copy)]
pub struct PropSprites {
    pub idle: &'static str,
    pub hurt: &'static str,
    pub dead: &'static str,
    pub flip_x: bool,
}

#[derive(Component, Clone, Copy)]
pub struct PropHpTracker {
    pub last_hp: i32,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct SecretEntrance {
    pub target: crate::data::SecretTarget,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct ManholeCover;

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct ProtoStatue;

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct GoldCar;

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct BloodFlower;

/// Ground-decal tint marker (bevy draws the area top-decal strip at
/// gray 0.5 alpha; the renderer applies it).
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct GroundDecalTint;

#[derive(Component)]
pub struct SwingFx {
    pub timer: Timer,
    /// Facing rotation radians (bevy `Transform` rotation: slash swing
    /// angle or wall-hit `wang`; the renderer orients the strip).
    pub angle: f32,
}

/// Static-FX fallback marker (bevy `catalog.has` arm without an anim
/// strip: art present as a bare PNG, drawn one frame for the marker
/// lifetime). The renderer draws `path` frame 0 when resolvable.
#[derive(Component, Clone, Copy, Debug)]
pub struct StaticFx {
    pub path: &'static str,
}

/// Fade-FX orientation radians (bevy rotates the fade sprite by the
/// projectile angle; the renderer applies it).
#[derive(Component, Clone, Copy, Debug)]
pub struct FxAngle(pub f32);

#[derive(Component)]
pub struct Dash {
    pub timer: Timer,
    pub dir: Vec2,
}

#[derive(Component)]
pub struct Shield {
    pub timer: Timer,
}

/// GML HitWarning (sprAssassinNotice): melee lunge telegraph spawned at
/// (x, y-16), destroyed on anim end (Other_7). Used by MeleeBandit, Gator,
/// BuffGator, EliteInspector, YVBoss.
#[derive(Component)]
pub struct HitWarning {
    pub timer: Timer,
}

/// GML PopoShield/EliteShield: frontal shield follower spawned by Shielder.
/// Follows owner at gunangle offset, blocks incoming fire positionally.
#[derive(Component)]
pub struct ShieldFollower {
    pub owner: Entity,
}

/// GML `EliteShield`: teleport anchor that pins its creator for 60
/// ticks while deflecting `typ` 1 shots (re-teamed) and destroying
/// `typ` 2 shots, then poofs away (`Alarm_0`).
#[derive(Component)]
pub struct EliteBlocker {
    pub owner: Entity,
    pub timer: Timer,
}

#[derive(Component)]
pub struct Telekinesis {
    pub timer: Timer,
}

/// Horror beam hold state (GML horrortime). While spec held and rads remain,
/// time builds at 0.9/s and each tick spawns round(time+1) bullets.
#[derive(Component, Default)]
pub struct HorrorCharge {
    pub time: f32,
}

/// Frog toxic charge (GML froggas 0..30). Hold roots player and charges,
/// release spawns N gas clouds.
#[derive(Component, Default)]
pub struct FrogCharge {
    pub gas: f32,
}

#[derive(Component)]
pub struct PopPopCharges(pub u8);

#[derive(Component)]
pub struct SnareZone {
    pub timer: Timer,
    pub radius: f32,
    pub slow: f32,
}

#[derive(Component)]
pub struct Slowed {
    pub timer: Timer,
    pub factor: f32,
}

#[derive(Component)]
pub struct Ally {
    pub life: Timer,
    pub shoot: Timer,
}

#[derive(Component)]
pub struct PortalStrike {
    pub timer: Timer,
    pub radius: f32,
    pub damage: i32,
}

#[derive(Component)]
pub struct HazardCloud {
    pub kind: HazardKind,
    pub radius: f32,
    pub damage: i32,
    pub timer: Timer,
    pub tick: Timer,
}

#[derive(Component, Default)]
pub struct HeadlessReady(pub bool);

#[derive(Component, Clone, Copy, Debug)]
pub struct CrownPedestal {
    pub kind: CrownKind,
}

#[derive(Resource, Debug, Clone)]
pub struct ThroneRoomState {
    pub generators_total: u8,
    pub generators_destroyed: u8,
    pub all_generators_down: bool,

    pub loop_eligible: bool,

    pub player_on_carpet: bool,
    pub halved_throne: bool,
}

impl Default for ThroneRoomState {
    fn default() -> Self {
        Self {
            generators_total: 4,
            generators_destroyed: 0,
            all_generators_down: false,
            loop_eligible: false,
            player_on_carpet: false,
            halved_throne: false,
        }
    }
}

impl ThroneRoomState {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn note_generator_destroyed(&mut self) {
        self.generators_destroyed = self.generators_destroyed.saturating_add(1);
        if self.generators_destroyed >= self.generators_total {
            self.all_generators_down = true;
            self.loop_eligible = true;
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct BigGenerator {
    pub index: u8,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct ThroneStatueProp {
    pub guardian_count: u8,
}

/// GML `Throne2Ball`: stalled-ball state. `timeout` counts stalled ticks,
/// `angle` is the latched spray heading, `sounded` latches the beam cue.
#[derive(Component, Clone, Copy, Debug)]
pub struct ThroneBall {
    pub timeout: f32,
    pub angle: f32,
    pub sounded: bool,
}

/// GML `Nothing2Death`: 80-tick victory pageant (30 scattered
/// `GreenExplosion`s, then 10 flung `BigRad`s). The `BigPortal-1` exit
/// rides the existing `loop_ready` flow.
#[derive(Component)]
pub struct ThroneVictory {
    pub timer: Timer,
    pub pos: Vec2,
    pub bursts: u8,
}

/// GML `SitDown`: throne-sit zone left where Throne II fell.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct SitZone;

/// GML sitting player (`spr_gosit` 345-tick beat into the win screen).
#[derive(Component)]
pub struct ThroneSit {
    pub timer: Timer,
}

/// GML `InvisiWall`: throne-II-converted wall (invisible but solid;
/// cleared on victory).
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct InvisiWall;

/// GML `ProtoStatue` foe: vault guardian state. `rad` snapshots the
/// run's crown rads on placement; `charged`/`phased` latch the IDPD
/// waves. (The prop-side `ProtoStatue` marker below flags secret
/// entrance statues and is unrelated.)
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct ProtoGuardian {
    pub rad: u32,
    pub charged: bool,
    pub phased: bool,
    pub init: bool,
}

/// GML `MomProjectile` marker: leaves a `ToxicGas` trail every tick and
/// bursts 25 gas clouds plus a `PortalClear` on removal.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct MomShot;

/// GML `PopoNade` marker: thrown IDPD grenade (counts toward the
/// Inspector's lob budget gate like `instance_number(PopoNade)`).
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PopoNadeM;

#[derive(Component, Clone, Copy, Debug)]
pub struct SnowmanAmbush;

#[derive(Component, Clone, Copy, Debug)]
pub struct GoldBarrelDrop;

#[derive(Component, Clone, Copy, Debug)]
pub struct RadChestContainer;

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PropNestMarkers {
    pub snowman: bool,
    pub cocoon: bool,
    pub mutant_tube: bool,
    pub soda_machine: bool,
    pub pizza_box: bool,
    pub small_gen: bool,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct ThroneCarpet {
    pub half_extents: Vec2,
}

#[derive(Component, Debug)]
pub struct Corpse {
    pub kind: EnemyKind,
    pub life: Timer,
    pub pos: Vec2,
    /// Facing recorded at death (bevy `corpse_sprite.flip_x` parity;
    /// only the player husk records a live flip).
    pub flip_x: bool,
}

// Flag Dying before deferred despawn.
#[derive(Component, Debug, Default)]
pub struct Dying;

#[derive(Component, Debug)]
pub struct PlayerDying {
    pub timer: Timer,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct ScreenEnd;

#[derive(Component, Clone, Copy, Debug)]
/// GML `Campfire` title actor verbatim (`objects/Campfire/Create_0`:
/// 1M hp, size 1, campfire idle/hurt/dead strips). Sim keeps the marker
/// + position; art/health-detail stays renderer-owned.
pub struct TitleCampfire;

#[derive(Component, Clone, Copy, Debug)]
/// GML `LogMenu` seat marker (spawned at campfire `y-32`).
pub struct TitleLogMenu;

#[derive(Component, Clone, Copy, Debug)]
/// GML `CampChar` title actor verbatim: wandering mutant around the
/// campfire. `race_gml` is the GML race id (0..16), `fixed` marks the
/// four hand-placed starters (Fish/Crystal/Eyes/Melting) that skip the
/// scatter pass.
pub struct TitleCampChar {
    pub race_gml: usize,
    pub fixed: bool,
}

#[derive(Component, Clone, Copy, Debug)]
/// GML `TV` beside Chicken (`scrCampfireMenuCreate` chicken arm).
pub struct TitleTv;

#[derive(Resource, Debug, Clone)]
pub struct FloorTransition {
    pub active: bool,

    pub stage: u8,
    pub timer: Timer,
    pub progress: f32,
    pub tip: String,
}

impl Default for FloorTransition {
    fn default() -> Self {
        Self {
            active: false,
            stage: 0,
            timer: Timer::from_seconds(0.05, TimerMode::Repeating),
            progress: 0.0,
            tip: String::new(),
        }
    }
}


