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
    pub visual: Option<NativeExplosionKind>,
}

impl Default for CustomExplosion {
    fn default() -> Self {
        Self {
            radius: 32.0,
            count: 1,
            spread: 0.0,
            visual: None,
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
/// `Alarm_4/5`). `alarm4` counts the 300-step grace (`Step_0` re-pins it to
/// 300 while `GenCont`/`LevCont` exist); firing it starts the 30-step hurt
/// pulse (`Alarm_4` performs `Alarm_5`: every player takes 1, `alarm[5] = 30`).
/// The port has no coop downing -- the comp exists for the HUD timer law.
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
/// damage application excluded): `alarm4` is the 300-step grace, `alarm5` the
/// 30-step hurt pulse, one re-arm per `alarm5` drain = one pulse. `steps` is
/// timescale steps (`delta * 30`). True on the tick a hurt pulse fires.
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
/// `image_speed = timescale * 0.4`, airhorn one-shot). `steps` is timescale
/// steps (`delta * 30`); `fps`/`frames` are the live strip's catalog values
/// (GML normalizes `image_speed` by `sprite_fps / room_speed`, room 30 Hz).
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CuzStrip {
    Idle,
    InteractTo,
    InteractFrom,
    Heya,
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct YungCuz {
    pub frame: f32,
    pub strip: CuzStrip,
    pub flipped: bool,
    pub crying: bool,
    pub alarm1: f32,
    pub last_cry: u8,
}

impl YungCuz {
    pub fn flipped() -> Self {
        Self {
            frame: 0.0,
            strip: CuzStrip::Idle,
            flipped: true,
            crying: false,
            alarm1: -1.0,
            last_cry: 0,
        }
    }

    pub fn sprite_path(&self) -> &'static str {
        if self.crying {
            "images/sprCuzCry.png"
        } else {
            match self.strip {
                CuzStrip::Idle => "images/sprCuzIdle.png",
                CuzStrip::InteractTo => "images/sprCuzInteractTo.png",
                CuzStrip::InteractFrom => "images/sprCuzInteractFrom.png",
                CuzStrip::Heya => "images/sprCuzInteract.png",
            }
        }
    }
}

pub fn yung_cuz_step(cuz: &mut YungCuz, steps: f32, fps: f32, frames: u32) -> bool {
    cuz.frame += steps * 0.4 * fps.max(0.0) / 30.0;
    let total = frames.max(1) as f32;
    if cuz.frame >= total {
        cuz.frame -= total * (cuz.frame / total).floor();
        true
    } else {
        false
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

/// GML `WantBoss` - the marker that arms the Big Bandit. `WantBoss/Step_0`
/// compares the **surviving** enemy count against `enemies` (captured in
/// `WantBoss/Create_0` at generation time) scaled by `treshhold`, which is
/// `0.98` normally and `0.9` on the area's last subarea, so the bandit arms
/// once `0.02` / `0.10` of the floor's trash is dead.
#[derive(Component, Clone, Copy, Debug)]
pub struct PendingDelayedBoss {
    pub kind: EnemyKind,
    pub initial_trash: u32,
    /// Fraction of `initial_trash` that must be dead.
    pub kill_fraction: f32,
    /// GML `WantBoss/Step_0:20-23`: off the last subarea the bandit only arms
    /// once every chest is open (`!instance_exists(chestprop)` and a
    /// `ChestOpen` exists), which is what turns 1-1/1-2 into the CanOasis
    /// secret instead of a routine boss.
    pub require_open_chests: bool,
    /// GML `WantBoss/Step_0:16-17`: `alarm[0] = 120` (4 s) once the threshold
    /// clears on the last subarea. Zero everywhere else - there the chest
    /// condition sets `alarm[0] = 1` instead.
    pub arm_delay: f32,

    pub from_wall: bool,
}

impl PendingDelayedBoss {
    pub fn kills_needed(&self) -> u32 {
        (self.initial_trash as f32 * self.kill_fraction).ceil() as u32
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
    /// GML has no generic lifetime - `spawn_sentry_turret` is the port's
    /// weapon deployable and keeps one; the GML `SentryGun` body has
    /// `None` and lives on `ammo` / `hp` alone.
    pub life: Option<Timer>,
    pub fire: Timer,
    /// GML `SentryGun/Create_0.gml:14` `alarm[0] = 30`: the opening
    /// delay in steps, counted down before `fire` starts running.
    pub first_shot: f32,
    /// GML `SentryGun/Create_0.gml:7` `ammo = 24`, spent one per alarm
    /// and re-checked at the end of `Alarm_0.gml:56`.
    pub ammo: i32,
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
    /// GML `givekill`: false for the kinds that clear it at creation
    /// (`FastRat`, `BigDogMissile`) and for maggot conversions.
    pub give_kill: bool,
}

#[derive(Component)]
pub struct EnemyBrain {
    pub attack: Timer,
    /// Generic ranged-fire alarm. GML runs decide and the shot from one
    /// `alarm[1]` (`Molefish/Alarm_1.gml:1`) with `alarm[2]` for the burst
    /// cadence (`Crab/Alarm_2.gml:1`); the port keeps the shot on its own
    /// timer so the decide re-arm cannot starve every shot.
    pub fire_alarm: Timer,
    pub burst_left: usize,
    pub burst_timer: Timer,
    pub dash: f32,
    pub strafe_dir: f32,
    pub strafe_timer: Timer,
    pub melee: Timer,

    pub wkick: f32,

    pub walk: f32,

    pub ammo: u8,

    pub slash_delay: f32,

    pub gunangle: f32,

    pub heading: f32,

    pub rage: f32,

    pub fire: u8,

    pub friction: f32,

    pub close: bool,

    pub wepangle: f32,

    pub wepflip: f32,

    pub weapon_alarm: f32,

    pub burrow_state: u8,

    pub burrow_alarm0: f32,

    pub burrow_alarm1: f32,

    pub burrow_angle: f32,

    pub sniper_aiming: bool,

    pub maggot_spawn_charging: bool,

    pub maggot_spawn_charge_ticks: f32,

    pub maggot_spawn_facing: f32,

    /// GML `freeze` (`Grunt`, `EliteGrunt`, `Shielder`, `Inspector` `Create_0`).
    /// Gates the roll/grenade/burst arms behind `freeze > 40`; the `+ 3` term
    /// is dead against a player -- `Player/Create_0:99` sets `can_shoot = true`
    /// and nothing clears it, so `!target.can_shoot` is never true.
    pub freeze: f32,

    /// GML `roll`. `true` is mid-roll; `Grunt/Other_10` and
    /// `EliteGrunt/Step_0` run the roll physics in that state.
    pub roll: bool,

    /// GML `angle`, the roll spin accumulator (`Grunt/Other_10`) or the
    /// live roll heading (`EliteGrunt/Step_0`, in degrees).
    pub roll_angle: f32,

    /// GML `fuel` (`EliteGrunt`): roll duration, 100 frames, rearmed each
    /// step while `!roll`.
    pub fuel: f32,

    /// GML `grenades`: remaining `PopoNade` lobs (Grunt 2, Inspector 4,
    /// EliteGrunt 4).
    pub grenades: u8,

    /// GML `lastx, lasty` - the target's last seen position, which the
    /// `PopoNade` lob is aimed at.
    pub last_seen: glam::Vec2,

    /// GML `right`, +/-1, via `scrRight(0)` (`hspeed > 0 ? 1 : -1`) or
    /// `scrRight(1)` (gunangle east/west).
    pub right: f32,

    /// GML `control` (`Inspector`): the mind-control drag on the player.
    pub control: bool,

    /// GML `Ratking` `spawns`: rats vomited so far. Past 24 the king rolls
    /// `random(4) < 3` to `instance_change(RatkingRage)`.
    pub ratking_spawns: f32,

    /// GML `instance_change(RatkingRage, false)` swaps the king's behaviour
    /// without replacing the instance, so it is state rather than a kind.
    /// The rage form charges (touch 4), breaks walls, and bursts five
    /// `FastRat` plus five `AcidStreak` on death.
    pub ratking_rage: bool,

    /// GML `maxspeed` (`Spider/Create_0:19` and `Spider/Alarm_1`), the
    /// spider's per-tick speed ceiling in px/frame: 3 at rest, 5 chasing.
    pub maxspeed: f32,

    /// GML `gunoffset` (`HostileHorror/Alarm_1:13`): a +-10 degree bias
    /// added to `gunangle` on every step of a running spray.
    pub gunoffset: f32,

    /// GML `walkdir` (`PopoFreak/Alarm_1`): the heading held while hurt,
    /// so the impulse keeps pushing even with the walk sprite swapped out.
    pub walkdir: f32,

    /// GML `wave` (`Salamander/Create_0` `random(6.2)`, `Alarm_2:5`
    /// `wave += 0.03`): the spray phase. Each `Alarm_2` launches one
    /// `TrapFire` at `gunangle + sin(wave) * 70`, so the jets sweep.
    pub wave: f32,

    /// GML `charge` (`HostileHorror/Create_0` 0, `Other_10:20-31`
    /// `charge += 0.1`): the radial spray's per-tick bullet count is
    /// `round(charge + 1)`, paid for out of the enemy's `raddrop`.
    pub charge: f32,
}

pub const SCRAP_BOSS_MISSILE_HP: i32 = 22;
pub const SCRAP_BOSS_MISSILE_RADIUS: f32 = 6.0;
pub const BIG_DOG_MISSILE_HP: i32 = 22;
pub const BIG_DOG_MISSILE_RADIUS: f32 = 6.0;
pub const BIG_DOG_MISSILE_DAMAGE: i32 = 5;

#[derive(Component, Clone, Copy, Debug)]
pub struct ScrapBossMissileState {
    pub creator: Option<Entity>,
    pub fuse: Timer,
    pub hurt_timer: Timer,
    pub trail_timer: Timer,
    pub hurt: bool,
}

impl ScrapBossMissileState {
    pub fn new(loops: u32) -> Self {
        let trail = if loops > 0 {
            Timer::from_seconds(
                (12u32.saturating_sub(loops).max(1)) as f32 / 30.0,
                TimerMode::Once,
            )
        } else {
            Timer::disarmed()
        };
        Self {
            creator: None,
            fuse: Timer::disarmed(),
            hurt_timer: Timer::disarmed(),
            trail_timer: trail,
            hurt: false,
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct BigDogMissileState {
    pub creator: Entity,
    pub throne_butt: bool,
    pub fuse: Timer,
    pub hurt_timer: Timer,
    pub trail_timer: u8,
}

impl BigDogMissileState {
    pub fn new(creator: Entity) -> Self {
        Self {
            creator,
            throne_butt: false,
            fuse: Timer::disarmed(),
            hurt_timer: Timer::disarmed(),
            trail_timer: 0,
        }
    }

    pub fn hurt_from_projectile(&mut self) {
        self.hurt_timer = Timer::from_seconds(50.0 / 30.0, TimerMode::Once);
        self.fuse = Timer::from_seconds(50.0 / 30.0, TimerMode::Once);
    }

    pub fn hit_enemy(&mut self) {
        self.fuse = Timer::from_seconds(1.0, TimerMode::Once);
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct ToxicGasState {
    pub age: f32,
    pub typ: u8,
    pub friction: f32,
    pub radius: f32,
    pub scale: f32,
    pub grow_speed: f32,
    pub rot: f32,
    pub speed_bonus: f32,
}

impl ToxicGasState {
    pub fn new() -> Self {
        Self {
            age: 0.0,
            typ: 0,
            friction: 0.01,
            radius: 16.0,
            scale: 0.6,
            grow_speed: 0.003,
            rot: 1.0,
            speed_bonus: 0.0,
        }
    }
}

impl Default for ToxicGasState {
    fn default() -> Self {
        Self::new()
    }
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
            EnemyKind::Technomancer => (2.0, 3.5),
            EnemyKind::Captain => (0.7, 2.0),
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
    /// GML `CursedPickup` (parent `AmmoPickup`): pays out ammo exactly
    /// like an `AmmoPickup` on touch, but blinks for ~2 s spraying
    /// `Curse` motes and then detonates.
    CursedAmmo,
    Weapon(WeaponId),
    Chest(ChestKind),
    /// GML `Curse` mote (Robot's cursed-weapon eat spills 10; ambient,
    /// no pickup effect - collected by despawn).
    Curse,
}

/// GML `BigGenerator/Destroy_0.gml:33-35` fires
/// `with (Nothing) hp = round(hp / 2)`. Routed through a marker so the
/// drop system never holds `&mut Health` alongside the player's `&Health`.
#[derive(Component, Clone, Copy, Debug)]
pub struct ThroneWeaken;

/// GML `CursedPickup`: `blink = 30` with `Alarm_0` re-arming itself
/// every 2 steps, so the detonation lands on `blink < 0` after 62 steps.
/// `alarm` counts the re-arm; `sounded` gates the one-shot
/// `Create_0` sting, which plays on the first tick instead of at spawn
/// because the spawn helpers carry no audio queue.
#[derive(Component, Clone, Copy, Debug)]
pub struct CursedAmmoBlink {
    pub blink: i32,
    pub alarm: f32,
    pub sounded: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChestKind {
    Weapon,
    Ammo,
    /// GML `AmmoChestMystery`: 25% of ammo chests in Crystal Caves
    /// and later (`AmmoChest/Create_0.gml:10-13`). Pays `x3` of one ammo
    /// type the player is not holding, or under Get Loaded `x3` of all
    /// five except the two `scrAmmoDecideType(_player, true)` returns.
    Mystery,
    /// GML `GoldChest`: Y.V. Mansion weapon chests
    /// (`GenCont/Alarm_1.gml:89-94`), `scrDecideWepGold` loot.
    Gold,
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

/// GML per-chest `sprite_index` / `spr_dead` variants, fixed in `Create_0`:
/// Oasis `sprClamChest`/`Open` (`WeaponChest/Create_0.gml:12-15`), Crown of
/// Curses `sprCursedChest` (`WeaponChest/Create_0.gml:16-18`), Ambidextrous
/// `sprWeaponChestSteroidsUltra` (`:19-21`), Steroids Get Loaded
/// `sprAmmoChestSteroids` (`AmmoChest/Create_0.gml:16-19`), Pizza Sewers
/// `choose(sprPizzaChest1, sprPizzaChest2)` / `sprPizzaChestOpen`
/// (`HealthChest/Create_0.gml:12-15`).
#[derive(Component, Clone, Copy, Debug)]
pub struct ChestArt {
    pub idle: &'static str,
    pub open: &'static str,
}

impl From<WeaponKind> for PickupKind {
    fn from(k: WeaponKind) -> Self {
        PickupKind::Weapon(k.into())
    }
}

#[derive(Component)]
pub struct Portal;

/// GML Portal state machine (objects/Portal/*): Spawn -> Idle -> Disappear.
/// `kind`: 1 normal, 2 popo (HQ), 3 proto (Vault), which pick the Idle strip.
/// `anim` stands in for bevy's oneshot sprite anim: Spawn lasts
/// `sprPortalSpawn` (2 frames @ 8 fps = 0.25 s), Disappear
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

/// GML Chicken-thrown weapon. `scrWeaponPickupCreate` + `motion_set`
/// gunangle±2 at speed 16 with `friction = 0` and `mask_index = mskPlasma`,
/// plus the Determination ultra's `alarm[1]` return.
///
/// Self-integrating rather than using `GroundPhysics`, because the thrown gun
/// has no friction at all while it flies (GML sets `friction = 0`) and the
/// resting pickups' fixed 0.4 decay would stop it after ~26px.
#[derive(Component, Clone, Copy, Debug)]
pub struct FlungWeapon {
    pub team: Team,
    pub creator: Entity,
    /// px/step, mirroring GML's `speed` unit (16 on throw).
    pub vel: Vec2,
    /// GML `mask_index == mskPlasma` while `speed > 0`, flipping to
    /// `mskWepPickup` once it stops (`WepPickup/Step_0.gml:17-23`). Only the
    /// plasma mask damages things.
    pub airborne: bool,
    /// GML `friction`: 0 while flying, 0.5 once it has hit something.
    pub friction: f32,
    /// GML `scr_skill_get(mut_throne_butt)` on a Chicken: the throw pierces
    /// (`speed *= 0.8`) instead of sticking.
    pub pierce: bool,
    pub return_ticks: u8,
}

/// GML `objects/NecroReviveArea`: a 15-frame marker dropped on a corpse; when
/// it expires the corpse is re-created as a `Necromancer` if the spot is
/// still clear.
#[derive(Component, Clone, Copy, Debug)]
pub struct NecroReviveArea {
    pub target: Vec2,
    pub timer: Timer,
}

/// GML `TechnoMancer` registers. The boss runs a six-alarm state machine over
/// `main`/`intro`/`drawspr`, which does not fit the shared [`BossBrain`] pair of
/// timers, so it carries its own.
#[derive(Component, Clone, Copy, Debug)]
pub struct TechnomancerState {
    /// `alarm[1]`: the 90-tick decide cadence, armed 300 at create.
    pub alarm1: Timer,
    /// `alarm[2]`: fires 55 ticks after a corpse revive is armed.
    pub alarm2: Timer,
    /// `alarm[4]`: 9 ticks to finish appearing, 22 to finish disappearing.
    pub alarm4: Timer,
    /// `alarm[5]`: 70 after arming a revive, 52 after arming turrets,
    /// 17 for the first appearance.
    pub alarm5: Timer,
    /// `alarm[6]`: 35 ticks before armed turrets actually appear.
    pub alarm6: Timer,
    /// `main`: this instance is the live one; the rest are dormant.
    pub main: bool,
    /// `intro`: the boss intro has already played.
    pub intro: bool,
    /// `drawspr`, which also gates whether the instance can act.
    pub visual: TechnoVisual,
}

#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum TechnoVisual {
    /// `sprTechnoMancerInactive`
    Inactive,
    /// `sprTechnoMancerAppear`
    Appear,
    /// `sprTechnoMancer`
    Active,
    /// `sprTechnoMancerDisappear`
    Disappear,
}

impl Default for TechnomancerState {
    fn default() -> Self {
        Self {
            // GML `TechnoMancer/Create_0.gml:21` `alarm[1] = 300`.
            alarm1: Timer::from_seconds(300.0 / 30.0, TimerMode::Once),
            alarm2: Timer::disarmed(),
            alarm4: Timer::disarmed(),
            alarm5: Timer::disarmed(),
            alarm6: Timer::disarmed(),
            main: true,
            intro: false,
            visual: TechnoVisual::Inactive,
        }
    }
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
    pub ticks: u32,
}

/// GML `objects/TrapFire`: the short-lived fire jet. `lil_hunter` is the
/// `sprite_index != sprFireLilHunter` gate on `Collision_hitme.gml:4` -
/// only the LilHunterDie death ring flies through a body it cannot hurt,
/// every other jet reverts in place.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct TrapFire {
    pub lil_hunter: bool,
}

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

#[derive(Resource, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct PortalCarriedWeapons(pub Vec<WeaponId>);

#[derive(Component)]
pub struct HurtAnim {
    pub idle: &'static str,
    pub walk: Option<&'static str>,
    pub hurt: &'static str,
    pub timer: Timer,
    pub was_moving: bool,
    /// GML never changes `image_speed` on a hurt flip, so the return
    /// to `spr_idle` must resume the rate the entity was built with
    /// (props: `prop/Create_0.gml:3` `image_speed = 0.4`).
    pub rate: f32,
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

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeWallMotion {
    Stop,
    Bounce,
    BounceEveryThird,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct GmlImage {
    pub path: &'static str,
    pub frames: u32,
    pub phase: f32,
    pub image_speed: f32,
    pub looping: bool,
    pub finished: bool,
    pub destroy_on_end: bool,
    /// GML hand-driven first-frame ramp (`chestprop/Step_0.gml:4-7`):
    /// while `image_index < 1` the index advances by
    /// `random(first_jitter)` instead of `image_speed`, so the entity
    /// dwells on frame 0 for a random run before the strip plays.
    /// `0.0` disables the ramp (every other GML object).
    pub first_jitter: f32,
}

impl GmlImage {
    pub fn new(path: &'static str, frames: u32, image_speed: f32) -> Self {
        Self {
            path,
            frames: frames.max(1),
            phase: 0.0,
            image_speed,
            looping: true,
            finished: false,
            destroy_on_end: false,
            first_jitter: 0.0,
        }
    }

    /// GML `chestprop/Step_0.gml:4-7` verbatim ramp: `image_speed = 0`
    /// (the built-in advance is off) and the index is driven by hand -
    /// `random(0.04)` while `image_index < 1`, else `+0.4`.
    pub fn ramped(path: &'static str, frames: u32, image_speed: f32, first_jitter: f32) -> Self {
        Self {
            first_jitter,
            ..Self::new(path, frames, image_speed)
        }
    }

    pub fn animated(
        path: &'static str,
        frames: u32,
        image_speed: f32,
        destroy_on_end: bool,
    ) -> Self {
        Self {
            looping: !destroy_on_end,
            destroy_on_end,
            ..Self::new(path, frames, image_speed)
        }
    }

    pub fn set_path(&mut self, path: &'static str, frames: u32) {
        self.path = path;
        self.frames = frames.max(1);
        self.phase = 0.0;
        self.finished = false;
    }

    pub fn frame(&self) -> i32 {
        let frame = if self.looping {
            self.phase.rem_euclid(self.frames.max(1) as f32)
        } else {
            self.phase
        };
        (frame.floor() as i32).clamp(0, self.frames.max(1) as i32 - 1)
    }

    /// `ramp_draw` is the GML `random(first_jitter)` sample for the
    /// first-frame dwell; the caller passes `1.0` when the ramp is off.
    pub fn advance_ramped(&mut self, steps: f32, ramp_draw: f32) -> bool {
        if self.finished {
            return false;
        }
        if self.first_jitter > 0.0 && self.phase < 1.0 {
            self.phase += self.first_jitter * ramp_draw;
            if self.phase < 1.0 {
                return false;
            }
        }
        self.advance(steps)
    }

    pub fn advance(&mut self, steps: f32) -> bool {
        if self.finished || self.image_speed == 0.0 {
            return false;
        }
        self.phase += self.image_speed * steps;
        if self.phase < self.frames.max(1) as f32 {
            return false;
        }
        if self.looping {
            self.phase %= self.frames.max(1) as f32;
            false
        } else {
            self.phase = self.frames.max(1) as f32;
            self.finished = true;
            true
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct NativeLifetime {
    pub ticks: f32,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct NativeMotion {
    pub velocity: Vec2,
    pub friction: f32,
    pub radius: f32,
    pub wall: NativeWallMotion,
    pub tick: u32,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct NativeScale(pub Vec2);

#[derive(Component, Clone, Copy, Debug)]
pub struct NativeAngle(pub f32);

#[derive(Component, Clone, Copy, Debug)]
pub struct NativeFlip(pub bool);

#[derive(Component, Clone, Copy, Debug)]
pub struct NativeDepth(pub f32);

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeExplosionKind {
    Standard,
    Small,
    Green,
    Meat,
    Popo,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct ExplosionVisual(pub NativeExplosionKind);

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct TopSmall;

#[derive(Component, Clone, Copy, Debug)]
pub struct MaggotSpawnCharge {
    pub image: GmlImage,
    pub facing: f32,
    pub ticks_left: f32,
    pub duration: f32,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct MaggotSpawnInternalDrain;

impl MaggotSpawnCharge {
    pub fn new() -> Self {
        Self {
            image: GmlImage::new("images/sprMSpawnChrg.png", 4, 0.4),
            facing: 1.0,
            ticks_left: 19.0,
            duration: 19.0,
        }
    }
}

/// GML `Dust`/`Smoke`/`Feather`/`Curse` motes. `strip` selects the art
/// (`sprDust`, `sprSmoke`, `sprRavenFeather`/`sprLeaf`/`sprMoney` per-spawn,
/// `sprCurse`); `friction` is the flat GML value (0.3 dust, 0.1 smoke, 0.005
/// curse; feathers use fall-sway); `spin`/`grow`/`grow_decay` drive `FxAngle`
/// + `MoteScale` as in the `Step_0` handlers; `sway` is the feather fall law
/// (`speed *= 0.9` over 0.2).
#[derive(Component, Clone, Copy)]
pub struct Mote {
    pub friction: f32,
    pub spin: f32,
    pub grow: f32,
    pub grow_decay: f32,
    pub sway: bool,
    pub lifetime: Option<f32>,
    pub fall: Option<f32>,
    pub bounce_period: u8,
    pub kill_on_spiral: bool,
    pub tick: u32,
}

/// Mote visual scale (`image_xscale`/`image_yscale` verbatim): dust
/// 0.7, smoke 0.8, feathers 1.0. Renderer-owned draw scale.
#[derive(Component, Clone, Copy)]
pub struct MoteScale(pub f32);

/// Mote strip selector (renderer maps to the catalog strip; feathers
/// carry their per-spawn sprite: leaf/money/raven).
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoteStrip {
    Dust,
    Smoke,
    Leaf,
    Money,
    Raven,
    Curse,
    PortalL,
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
    pub charge: Option<&'static str>,
}

#[derive(Component)]
pub struct Prop {
    pub size: Vec2,
    pub hp: i32,
    pub destructible: bool,
    pub explosive: bool,
}

/// Props whose death runs a bespoke cascade instead of the generic
/// corpse-and-rads path: GML `VaultStatue/Destroy_0` raises a
/// `CrownGuardian` and zeroes its siblings, and `VenuzTV/Destroy_0`
/// raises the YV boss. [`crate::enemies::tick_special_props`] owns both, so
/// the generic prop-damage path leaves them alone.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialPropDeath {
    VaultStatue,
    VenuzTv,
}

#[derive(Component, Clone, Copy)]
pub struct PropSprites {
    pub idle: &'static str,
    pub hurt: &'static str,
    pub dead: &'static str,
    pub flip_x: bool,
}

/// GML `prop.size` is a knockdown *tier* (1..5), not a pixel box, and is
/// the only gate on `enemy/Collision_prop.gml:20`
/// (`if (size > other.size && meleedamage > 0 ...)`). The Rust `Prop.size`
/// is the pixel AABB, so the tier rides alongside it.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PropTier(pub u8);

/// GML `canbreak = 0` (`ThroneStatue/Create_0.gml:2`,
/// `BigGeneratorInactive/Create_0.gml:2`). `ThroneStatue/Step_1.gml:4`
/// re-pins `hp = 1000` every step, so a statue can never die from damage.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct UnbreakableProp;

/// GML `chestprop/Create_0.gml:10` `dropseed =
/// rng_next_int(RNGStates.WeaponDrops)`, drawn in `Create_0` before any
/// subclass roll. `BigWeaponChest/Collision_Player.gml:14` and
/// `CursedBigChest/Collision_Player.gml:14` `random_set_seed(dropseed)` before
/// rolling their three weapons, so contents are fixed at spawn.
#[derive(Component, Clone, Copy, Debug)]
pub struct DropSeed(pub u32);

/// GML `chestprop` ground physics: `friction = 0.4`
/// (`chestprop/Create_0.gml:5`), `if speed > 4 speed = 4`
/// (`chestprop/Step_0.gml:9-10`), `move_bounce_solid(true)` on walls
/// (`Collision_Wall.gml:4`), `motion_add(..., 1)` from an overlapping chest
/// (`Collision_chestprop.gml:5`) and `motion_add(..., 0.5)` from a walking
/// enemy (`enemy/Collision_chestprop.gml:4`). Velocities are px per second.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct ChestPropMotion {
    pub vel: Vec2,
}

/// GML `Detail` ground-decal scatter (`scrPopulate.gml:26-30`, one
/// `random(6) < 1` per room tile; `Detail/Create_0.gml` picks
/// `sprDetail<area>`, bails when the tile is `styleb` outside City, and
/// lands on a random frame with a random mirror).
#[derive(Component, Clone, Copy, Debug)]
pub struct GroundDetail {
    pub path: &'static str,
    pub frame: u32,
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
    /// GML `PortalStrike/Create_0.gml:3-7`: `size = 28` spacing between the
    /// five blasts, and `ammo = 5` shots left.
    pub size: f32,
    pub ammo_left: i32,
    /// Running offset along `direction` (`explo_x`/`explo_y` in GML).
    pub explo: Vec2,
    /// Unit vector the five blasts march along, fixed when the key went down.
    pub heading: Vec2,
    /// GML `PortalStrike/Step_0.gml:41-49`: the strike is an armed trajectory
    /// preview that only detonates once the key comes back up.
    pub armed: bool,
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

/// GML `PopoShield` (`Create_0`: `alarm[0] = 60`, `team = team_popo`,
/// `creator`): a 60-frame bubble pinned to its `Shielder` (`Step_2`) that
/// turns hostile `typ == 1` projectiles and eats `typ == 2` ones
/// (`Collision_projectile`). `Other_7` charges the owner's `alarm[1] += 20`
/// as it pops; `Shielder/Alarm_2` refuses to fire while one is up.
#[derive(Component, Clone, Copy, Debug)]
pub struct PopoShieldM {
    pub creator: Entity,
    pub frames: f32,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct SnowmanAmbush;

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
    /// GML corpse `size` (`Corpse/Create_0.gml:1` defaults to 1;
    /// `enemy/Destroy_0.gml:11` copies the dying enemy's).
    pub size: i32,
    pub life: Timer,
    pub pos: Vec2,
    /// Facing recorded at death (bevy `corpse_sprite.flip_x` parity;
    /// only the player husk records a live flip).
    pub flip_x: bool,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct CorpseCollision {
    pub source_size: i32,
    pub radius: f32,
    pub settled: bool,
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

/// GML `Campfire` title actor verbatim (`objects/Campfire/Create_0`:
/// 1M hp, size 1, campfire idle/hurt/dead strips). Sim keeps the marker
/// + position; art/health-detail stays renderer-owned.
#[derive(Component, Clone, Copy, Debug)]
pub struct TitleCampfire;

/// GML camper strip table verbatim (`CampChar` instance vars as set by
/// `scrCampfireMenuCreateCharacter` + the BigDog inline override + the Frog
/// `Step_0` far/near rewrite): `slct` = deselected end (`spr_slct`),
/// `to`/`menu` = selected transition/end (`spr_to`/`spr_menu`), `from` =
/// deselect transition (`spr_from`). Frog's `menu` doubles as the sit end,
/// `from` unused. Skeleton/Frog ship no `Select`/`Selected` strips, so their
/// selected half falls back to `sprMutant<gml>Idle` -- the `_default` arg of
/// `scr_race_get_sprite` (`scrRaces.gml:74`). Ends loop, transitions oneshot.
pub struct CamperStrips {
    pub slct: &'static str,
    pub to: &'static str,
    pub menu: &'static str,
    pub from: &'static str,
}

pub fn camper_strips(gml: usize, frog_far: bool) -> CamperStrips {
    if gml == 13 {
        return CamperStrips {
            slct: "images/sprScrapBossSleep.png",
            to: "images/sprScrapBossIntro.png",
            menu: "images/sprScrapBossIdle.png",
            from: "images/sprScrapBossSleepHurt.png",
        };
    }
    if gml == 15 {
        if frog_far {
            return CamperStrips {
                slct: "images/sprMutant15Sit.png",
                to: "images/sprMutant15Sit.png",
                menu: "images/sprMutant15Sit.png",
                from: "images/sprMutant15Sit.png",
            };
        }
        return CamperStrips {
            slct: "images/sprFrogMenu.png",
            to: "images/sprMutant15Idle.png",
            menu: "images/sprMutant15Idle.png",
            from: "images/sprFrogMenuDeselect.png",
        };
    }
    let (slct, to, menu, from) = match gml {
        1 => (
            "images/sprFishMenu.png",
            "images/sprFishMenuSelect.png",
            "images/sprFishMenuSelected.png",
            "images/sprFishMenuDeselect.png",
        ),
        2 => (
            "images/sprCrystalMenu.png",
            "images/sprCrystalMenuSelect.png",
            "images/sprCrystalMenuSelected.png",
            "images/sprCrystalMenuDeselect.png",
        ),
        3 => (
            "images/sprEyesMenu.png",
            "images/sprEyesMenuSelect.png",
            "images/sprEyesMenuSelected.png",
            "images/sprEyesMenuDeselect.png",
        ),
        4 => (
            "images/sprMeltingMenu.png",
            "images/sprMeltingMenuSelect.png",
            "images/sprMeltingMenuSelected.png",
            "images/sprMeltingMenuDeselect.png",
        ),
        5 => (
            "images/sprPlantMenu.png",
            "images/sprPlantMenuSelect.png",
            "images/sprPlantMenuSelected.png",
            "images/sprPlantMenuDeselect.png",
        ),
        6 => (
            "images/sprVenuzMenu.png",
            "images/sprVenuzMenuSelect.png",
            "images/sprVenuzMenuSelected.png",
            "images/sprVenuzMenuDeselect.png",
        ),
        7 => (
            "images/sprSteroidsMenu.png",
            "images/sprSteroidsMenuSelect.png",
            "images/sprSteroidsMenuSelected.png",
            "images/sprSteroidsMenuDeselect.png",
        ),
        8 => (
            "images/sprRobotMenu.png",
            "images/sprRobotMenuSelect.png",
            "images/sprRobotMenuSelected.png",
            "images/sprRobotMenuDeselect.png",
        ),
        9 => (
            "images/sprChickenMenu.png",
            "images/sprChickenMenuSelect.png",
            "images/sprChickenMenuSelected.png",
            "images/sprChickenMenuDeselect.png",
        ),
        10 => (
            "images/sprRebelMenu.png",
            "images/sprRebelMenuSelect.png",
            "images/sprRebelMenuSelected.png",
            "images/sprRebelMenuDeselect.png",
        ),
        11 => (
            "images/sprHorrorMenu.png",
            "images/sprHorrorMenuSelect.png",
            "images/sprHorrorMenuSelected.png",
            "images/sprHorrorMenuDeselect.png",
        ),
        12 => (
            "images/sprRogueMenu.png",
            "images/sprRogueMenuSelect.png",
            "images/sprRogueMenuSelected.png",
            "images/sprRogueMenuDeselect.png",
        ),
        14 => (
            "images/sprSkeletonMenu.png",
            "images/sprMutant14Idle.png",
            "images/sprMutant14Idle.png",
            "images/sprSkeletonMenuDeselect.png",
        ),
        16 => (
            "images/sprCuzMenu.png",
            "images/sprCuzMenuSelect.png",
            "images/sprCuzMenuSelected.png",
            "images/sprCuzMenuDeselect.png",
        ),
        _ => (
            "images/sprDefault.png",
            "images/sprDefault.png",
            "images/sprDefault.png",
            "images/sprDefault.png",
        ),
    };
    CamperStrips {
        slct,
        to,
        menu,
        from,
    }
}

/// GML camper menu-strip name verbatim (`scrCampfireMenuCreate`:
/// `spr<Name>Menu`, `_name` from `scrRaceGetStringID(race, true)` -
/// capitalized race name + `Menu`). Falls back to `sprMutant<gml>Menu`
/// then `sprDefault` per `scr_race_get_sprite`. BigDog campers sleep
/// (`sprScrapBossSleep`, set inline in `scrCampfireMenuCreate`).
pub fn camper_menu_strip(gml: usize) -> &'static str {
    match gml {
        1 => "images/sprFishMenu.png",
        2 => "images/sprCrystalMenu.png",
        3 => "images/sprEyesMenu.png",
        4 => "images/sprMeltingMenu.png",
        5 => "images/sprPlantMenu.png",
        6 => "images/sprVenuzMenu.png",
        7 => "images/sprSteroidsMenu.png",
        8 => "images/sprRobotMenu.png",
        9 => "images/sprChickenMenu.png",
        10 => "images/sprRebelMenu.png",
        11 => "images/sprHorrorMenu.png",
        12 => "images/sprRogueMenu.png",
        13 => "images/sprScrapBossSleep.png",
        14 => "images/sprSkeletonMenu.png",
        15 => "images/sprFrogMenu.png",
        16 => "images/sprCuzMenu.png",
        _ => "images/sprDefault.png",
    }
}

#[derive(Component, Clone, Copy, Debug)]
/// GML `LogMenu` seat marker (spawned at campfire `y-32`).
pub struct TitleLogMenu;

#[derive(Component, Clone, Copy, Debug)]
/// GML `CampChar` title actor verbatim. `fixed` marks the four hand-placed
/// starters (Fish/Crystal/Eyes/Melting) that skip the scatter pass. `swap` is
/// the `Other_7` select/deselect two-step: `None` = parked on the end strip,
/// `Some(true)` = playing `spr_to` toward `spr_menu`, `Some(false)` = playing
/// `spr_from` toward `spr_slct`; the render arm flips `swap` on selection
/// change and holds the strip until the oneshot finishes.
pub struct TitleCampChar {
    pub race_gml: usize,
    pub fixed: bool,
    pub swap: Option<bool>,
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
