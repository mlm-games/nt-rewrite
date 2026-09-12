use crate::data::*;
use crate::enemy_data::enemy_def;
use crate::time::{GTimer as Timer, TimerMode};
use bevy_ecs::prelude::*;
use glam::Vec2;
use serde::{Deserialize, Serialize};

pub const ARENA_W: f32 = 2560.0;
pub const ARENA_H: f32 = 1664.0;
pub const WALL_THICK: f32 = 60.0;
pub const PLAYER_RADIUS: f32 = 8.0;

// Base speed 4 px/frame = 120 px/s.
pub const PLAYER_BASE_SPEED: f32 = 120.0;

pub const PLAYER_ACCEL: f32 = 2700.0;

pub const PLAYER_FRICTION: f32 = 0.45;
/// GML reference view scale, no zoom: at a 1280x720 window the view
/// shows 426x240 world px (GML `macros_general`: base 320x240 GUI/view,
/// `scrSetViewSize` widens to `view_width_max = 240 * aspect` when
/// `opt_resolution`, which defaults on — so the visible height is always
/// 240 world px). The live frame derives the exact scale per window via
/// [`crate::render::gml_view_scale`] (`max(240/h, 320/w)` in dp); this
/// const is the 720p reference (`240/720 = 1/3`) for boot/tests.
pub const NT_CAM_SCALE: f32 = 1.0 / 3.0;

// Friction is subtractive per tick.
#[inline]
pub fn apply_gml_friction(vel: &mut Vec2, friction_f: f32, dt: f32) {
    let frames = dt * crate::SIM_HZ as f32;
    let sp = vel.length();
    if sp > 0.0 {
        let nsp = (sp - friction_f * 30.0 * frames).max(0.0);
        *vel = if nsp == 0.0 {
            Vec2::ZERO
        } else {
            *vel * (nsp / sp)
        };
    }
}

// Scale impulse by 30, not dt.
#[inline]
pub fn gml_motion_add_clamp(vel: &mut Vec2, dir: Vec2, impulse_f: f32, cap_f: f32, dt: f32) {
    let frames = dt * crate::SIM_HZ as f32;

    *vel += dir.normalize_or_zero() * (impulse_f * 30.0) * frames;
    let cap = cap_f * 30.0;
    if vel.length() > cap {
        *vel = vel.normalize() * cap;
    }
}

// Floor grid is 32 px tiles.
pub const TILE: f32 = 32.0;

#[derive(Resource, Default, Clone)]
pub struct FloorMask {
    pub cells: std::collections::HashSet<(i32, i32)>,
    pub cols: i32,
    pub rows: i32,
}

impl FloorMask {
    pub fn world_to_cell(&self, p: Vec2) -> (i32, i32) {
        ((p.x / TILE).floor() as i32, (p.y / TILE).floor() as i32)
    }

    pub fn cell_center(&self, c: (i32, i32)) -> Vec2 {
        Vec2::new(
            c.0 as f32 * TILE + TILE * 0.5,
            c.1 as f32 * TILE + TILE * 0.5,
        )
    }

    pub fn is_walkable(&self, p: Vec2) -> bool {
        self.cells.contains(&self.world_to_cell(p))
    }

    /// Push-out from unwalkable cells. Port adaptation: the bevy build
    /// carried 2D positions in `Vec3` (`Transform.translation`); here
    /// positions are `Vec2` throughout, so the dead `z` channel is gone.
    pub fn resolve_circle(&self, pos: &mut Vec2, radius: f32) {
        let p = *pos;
        if self.is_walkable(p) {
            return;
        }

        let mut best = None::<(f32, Vec2)>;
        let (cx, cy) = self.world_to_cell(p);
        for dy in -3..=3 {
            for dx in -3..=3 {
                let c = (cx + dx, cy + dy);
                if !self.cells.contains(&c) {
                    continue;
                }
                let center = self.cell_center(c);
                let d = center.distance_squared(p);
                if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                    best = Some((d, center));
                }
            }
        }
        if let Some((_, center)) = best {
            let dir = (center - p).normalize_or_zero();
            let dist = center.distance(p);
            let push = (dist - (TILE * 0.35 - radius)).max(0.0);
            pos.x += dir.x * push;
            pos.y += dir.y * push;
        }
    }

    pub fn random_floor_pos(&self, rng: &mut impl rand::RngExt, min_from_origin: f32) -> Vec2 {
        if self.cells.is_empty() {
            return Vec2::ZERO;
        }
        for _ in 0..80 {
            let idx = rng.random_range(0..self.cells.len());
            let c = *self.cells.iter().nth(idx).unwrap();
            let p = self.cell_center(c);
            if p.length() >= min_from_origin {
                return p;
            }
        }
        self.cell_center(*self.cells.iter().next().unwrap())
    }
}

#[derive(Component)]
pub struct WallTile;

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct WallCell(pub i32, pub i32);

#[derive(Component, Clone, Debug, Default)]
pub struct WallVisuals {
    pub parts: Vec<Entity>,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct PendingWallBreak {
    pub cell: (i32, i32),
    pub pos: Vec2,

    pub spawn_floor: bool,
}

#[derive(Resource, Debug)]
pub struct HammerheadBudget {
    pub remaining: u32,
}

impl Default for HammerheadBudget {
    fn default() -> Self {
        Self { remaining: 20 }
    }
}

#[derive(Resource, Default, Debug, Clone)]
pub struct LastDamageTaken {
    pub hit_id: Option<HitId>,
    pub enemy_kind: Option<EnemyKind>,
    pub source_name: String,
}

impl LastDamageTaken {
    pub fn note(&mut self, hit_id: Option<HitId>, enemy_kind: Option<EnemyKind>) {
        self.hit_id = hit_id;
        self.enemy_kind = enemy_kind;
        self.source_name = match (hit_id, enemy_kind) {
            (_, Some(kind)) => enemy_def(kind).name.to_ascii_uppercase(),
            (Some(HitId::Enemy(id)), None) => EnemyKind::from_u16(id)
                .map(|k| enemy_def(k).name.to_ascii_uppercase())
                .unwrap_or_else(|| "ENEMY".into()),
            (Some(HitId::Contact), _) => "CONTACT".into(),
            (Some(HitId::Toxic), _) => "TOXIC".into(),
            (Some(HitId::Fire), _) => "FIRE".into(),
            (Some(HitId::Trap), _) => "TRAP".into(),
            (Some(HitId::Explosion(_)), _) => "EXPLOSION".into(),
            (Some(HitId::Weapon(_)), _) => "BULLET".into(),
            (Some(HitId::Crown), _) => "CROWN".into(),
            (Some(HitId::Other(_)), _) => "???".into(),
            _ => "???".into(),
        };
    }

    pub fn note_from_source(&mut self, source: Option<&DamageSource>) {
        match source {
            Some(s) => self.note(Some(s.hit_id), s.enemy_kind),
            None => self.note(None, None),
        }
    }
}

#[derive(Component)]
pub struct BossIntro {
    pub timer: Timer,
}

#[derive(Resource, Default)]
pub struct Score(pub u32);

#[derive(Resource, Default)]
pub struct SaveDirty(pub bool);

#[derive(Resource)]
pub struct Run {
    pub floor: u32,
    pub world: u32,
    pub area: crate::data::AreaId,
    pub loop_count: u32,
    pub floor_in_area: u32,
    pub gen_seed: u64,
    pub portal_open: bool,
    pub game_over: bool,
    pub total_kills: u32,

    pub blackswords: u32,
    /// Run-time in sim steps (GML `GameCont.tottimer`: HUD clock and
    /// Plant/B throne-skin gate).
    pub tottimer: u32,
    /// GML `GameCont.popolevel`: counts IDPD portals raised this run and
    /// gates the `IDPDSpawn` dir roll (dir 3 at 3+, dir 2 at 5+).
    pub popolevel: u32,
    /// GML chest counters (`GameCont/Other_5` + `scrPopChests`): unopened
    /// weapon chests, unopened rad chests, and levels since a new weapon.
    pub nochest: u32,
    pub noradch: u32,
    pub same_weapons_for: u32,
    /// GML `GameCont.horror`: a `HostileHorror` already hatched.
    pub horror: bool,
    /// GML `GameCont.hasfiredshots` / `haspickedweps` (Eyes-C and
    /// Steroids-C throne-skin gates).
    pub shots_fired: u32,
    pub weapons_picked: u32,
    /// GML `GameCont.win`: the run was won on the throne.
    pub won: bool,
    /// GML `UberCont.hardmode`: +13 hard and +1 loop from the start.
    pub hardmode: bool,
}

impl Default for Run {
    fn default() -> Self {
        Self {
            floor: 1,
            world: 1,
            area: crate::data::AreaId::Desert,
            loop_count: 0,
            floor_in_area: 1,
            gen_seed: 0,
            portal_open: false,
            game_over: false,
            total_kills: 0,
            blackswords: 0,
            tottimer: 0,
            popolevel: 0,
            nochest: 0,
            noradch: 0,
            same_weapons_for: 0,
            horror: false,
            shots_fired: 0,
            weapons_picked: 0,
            won: false,
            hardmode: false,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectedCharacter(pub RaceId);

impl Default for SelectedCharacter {
    fn default() -> Self {
        Self(RaceId::Fish)
    }
}

#[derive(Resource)]
pub struct PendingMutation {
    pub choices: Vec<MutationId>,
}

#[derive(Resource)]
pub struct PendingUltra {
    pub choices: Vec<UltraMutationId>,
}

#[derive(Resource, Default)]
pub struct MutationChoice(pub Option<usize>);

#[derive(Resource)]
pub struct Toast {
    pub text: String,
    pub timer: Timer,
}

impl Default for Toast {
    fn default() -> Self {
        Self {
            text: String::new(),
            timer: Timer::from_seconds(0.0, TimerMode::Once),
        }
    }
}

impl Toast {
    pub fn show(&mut self, text: &str) {
        self.text = text.to_string();
        self.timer = Timer::from_seconds(2.2, TimerMode::Once);
    }
}

#[derive(Resource, Default)]
pub struct ScarierFace(pub bool);

#[derive(Resource, Default)]
pub struct Euphoria(pub bool);

#[derive(Resource, Default)]
pub struct OpenMind(pub bool);

#[derive(Resource, Default)]
pub struct HeavyHeart(pub bool);

#[derive(Component)]
pub struct GameCleanup;

#[derive(Component)]
pub struct LevelCleanup;

#[derive(Component)]
pub struct Player {
    pub speed: f32,
    pub accel: f32,
    pub friction: f32,
    pub speed_mult: f32,
    pub rads: u32,
    pub level: u32,
    pub next_level_rads: u32,
    pub pickup_range: f32,
    pub fire_rate_mult: f32,
    pub spread_mult: f32,
    pub accuracy: f32,
    pub knockback_mult: f32,
    pub melee_range_mult: f32,
    pub drop_mult: f32,
    pub medkit_mult: f32,
    pub boiling_veins: bool,
    pub veins_threshold: i32,
    pub bloodlust: bool,
    pub lucky_shot: bool,
    pub gamma_guts: bool,
    pub back_muscle: u32,
    pub stress: bool,
    pub sharp_teeth: bool,
    pub strong_spirit_ready: bool,
    pub strong_spirit_spent: bool,
    /// GML `headloses`: banked max-HP from head loss (Chicken), paid
    /// back one per HealthChest.
    pub headloses: u32,
    pub strong_spirit_area_cleared: bool,
    pub last_wish_used: bool,
    pub mutation_picks_owed: u32,
    pub ultra_pick_owed: bool,
    pub chain_explosions: bool,
    pub shield_on_hit: bool,
    pub ability: AbilityKind,
    pub ability_cooldown: Timer,
    pub rogue_ammo: u8,
    pub rogue_ammo_max: u8,
    pub cuz_ammo: u8,
    pub cuz_ammo_max: u8,
    pub headless_ready: bool,
    pub free_ammo: bool,
    /// GML `infammo` (Fish Gun Warrant): free-ammo ticks left after
    /// exiting a portal (7 seconds).
    pub warrant: f32,
    pub crown: CrownKind,
    pub bolt_marrow: bool,
    pub hammerhead: bool,
    pub laser_brain: bool,
    pub recycle_gland: bool,
    pub shotgun_shoulders: bool,
    pub throne_butt: bool,

    pub euphoria: bool,

    pub patience_bonus: bool,
    pub patience_used: bool,

    pub ultra: Option<UltraMutationId>,

    pub ultra_damage_mult: f32,
    /// Generic ability scaling used by Throne Butt / ultras.
    pub ultra_ability_mult: f32,
    /// Edge state for the looping cues (`tick_loop_sfx` in player.rs):
    /// GML `sndEyesLoop`, `sndHorrorLoop`, `sndFrogLoop`,
    /// `sndChickenHeadlessLoop` start/stop around the matching hold.
    pub eyes_loop_on: bool,
    pub horror_loop_on: bool,
    pub frog_loop_on: bool,
    pub chicken_headless_loop_on: bool,
    pub mutations: Vec<MutationId>,
    /// GML wepangle flip: multiplied by -1 on every melee swing, alternating
    /// the gun/slash angle offset.
    pub melee_flip: bool,
    /// Skeleton Blood Gamble streak (GML skeletongamble).
    pub skeleton_gamble: u32,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            speed: PLAYER_BASE_SPEED,
            accel: PLAYER_ACCEL,
            friction: PLAYER_FRICTION,
            speed_mult: 1.0,
            rads: 0,
            level: 1,
            next_level_rads: 60,
            pickup_range: 95.0,
            fire_rate_mult: 1.0,
            spread_mult: 1.0,
            accuracy: 1.0,
            knockback_mult: 1.0,
            melee_range_mult: 1.0,
            drop_mult: 0.0,
            medkit_mult: 1.0,
            boiling_veins: false,
            veins_threshold: 4,
            bloodlust: false,
            lucky_shot: false,
            gamma_guts: false,
            back_muscle: 0,
            stress: false,
            sharp_teeth: false,
            strong_spirit_ready: false,
            strong_spirit_spent: false,
            headloses: 0,
            strong_spirit_area_cleared: false,
            last_wish_used: false,
            mutation_picks_owed: 0,
            ultra_pick_owed: false,
            chain_explosions: false,
            shield_on_hit: false,
            ability: AbilityKind::Flip,
            ability_cooldown: Timer::from_seconds(0.0, TimerMode::Once),
            rogue_ammo: 1,
            rogue_ammo_max: 3,
            cuz_ammo: 1,
            cuz_ammo_max: 3,
            headless_ready: false,
            free_ammo: false,
            warrant: 0.0,
            crown: CrownKind::None,
            bolt_marrow: false,
            hammerhead: false,
            laser_brain: false,
            recycle_gland: false,
            shotgun_shoulders: false,
            throne_butt: false,
            euphoria: false,
            patience_bonus: false,
            patience_used: false,
            ultra: None,
            ultra_damage_mult: 1.0,
            ultra_ability_mult: 1.0,
            eyes_loop_on: false,
            horror_loop_on: false,
            frog_loop_on: false,
            chicken_headless_loop_on: false,
            mutations: Vec::new(),
            melee_flip: false,
            skeleton_gamble: 0,
        }
    }
}

impl Player {
    pub fn ammo_cap(&self, kind: AmmoKind) -> i32 {
        ammo_cap_with(self.back_muscle, kind)
    }

    pub fn try_recharge_strong_spirit(&mut self, health: &Health) {
        if self.strong_spirit_ready {
            return;
        }
        let has_ss = self.mutations.contains(&MutationId::StrongSpirit)
            || matches!(
                self.ultra,
                Some(UltraMutationId::MeltingDetachment | UltraMutationId::BigDogGuardian)
            );
        if !has_ss {
            return;
        }
        if self.strong_spirit_spent
            && self.strong_spirit_area_cleared
            && health.hp >= health.max
            && health.max > 1
        {
            self.strong_spirit_ready = true;
            self.strong_spirit_spent = false;
            self.strong_spirit_area_cleared = false;
        }
    }
}

pub fn ammo_cap_with(back_muscle: u32, kind: AmmoKind) -> i32 {
    let base = crate::data::ammo_max(kind);
    if back_muscle == 0 || kind == AmmoKind::None {
        return base;
    }
    base + match kind {
        AmmoKind::Bullets => 300 * back_muscle as i32,
        _ => 44 * back_muscle as i32,
    }
}

#[derive(Component)]
pub struct AimDir(pub Vec2);

#[derive(Component)]
pub struct Velocity(pub Vec2);

#[derive(Component)]
pub struct Health {
    pub hp: i32,
    pub max: i32,
    pub invuln: Timer,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct NextHurt(pub u64);

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct CurrentFrame(pub u64);

#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Team {
    Player,
    Enemy,
}

#[derive(Component)]
pub struct Hitbox {
    pub radius: f32,
}

#[derive(Component)]
pub struct FireCooldown {
    pub timer: Timer,
    pub burst_left: usize,
    pub burst_timer: Timer,

    pub timer_b: Timer,
    pub burst_left_b: usize,
    pub burst_timer_b: Timer,
}

pub const MAX_WEAPON_SLOTS: usize = 3;
pub const MAX_AMMO_TYPES: usize = 6;

#[derive(Component, Clone, Debug)]
pub struct Inventory {
    pub weapons: [WeaponId; MAX_WEAPON_SLOTS],
    /// GML per-slot `curse` (cursed guns cannot be swapped away).
    pub cursed: [bool; MAX_WEAPON_SLOTS],
    pub weapon_slots: usize,
    pub current: usize,
    pub ammo: [i32; MAX_AMMO_TYPES],
    /// GML `swapanim` (set on swap, decays 1/tick): back-gun offsets.
    pub swapanim: f32,
    /// GML `trigger_fingers_shine` (gun frame index, decays 0.4/tick).
    pub shine: f32,
    /// GML `wepflip`/`bwepflip` (melee mirror, toggled per reload).
    pub wepflip: f32,
    pub bwepflip: f32,
}

impl Inventory {
    pub fn ammo_mut(&mut self, kind: AmmoKind) -> &mut i32 {
        let idx = match kind {
            AmmoKind::None => 0,
            AmmoKind::Bullets => 1,
            AmmoKind::Shells => 2,
            AmmoKind::Bolts => 3,
            AmmoKind::Explosives => 4,
            AmmoKind::Energy => 5,
        };
        &mut self.ammo[idx]
    }

    pub fn ammo_of(&self, kind: AmmoKind) -> i32 {
        match kind {
            AmmoKind::None => self.ammo[0],
            AmmoKind::Bullets => self.ammo[1],
            AmmoKind::Shells => self.ammo[2],
            AmmoKind::Bolts => self.ammo[3],
            AmmoKind::Explosives => self.ammo[4],
            AmmoKind::Energy => self.ammo[5],
        }
    }
}

#[derive(Component, Clone, Debug)]
pub struct RaceState {
    pub race: RaceId,
    pub skin: SkinLetter,
}

#[derive(Component)]
pub struct CrownState {
    pub crown: CrownKind,
    pub life_timer: Timer,
    pub love_timer: Timer,
    pub protection_ready: bool,
    pub destiny_ready: bool,
    pub curses_timer: Timer,
}

impl CrownState {
    pub fn new(crown: CrownKind) -> Self {
        let mut life_timer = Timer::from_seconds(2.0, TimerMode::Repeating);
        life_timer.reset();

        let mut love_timer = Timer::from_seconds(35.0, TimerMode::Repeating);
        love_timer.reset();

        let mut curses_timer = Timer::from_seconds(14.0, TimerMode::Repeating);
        curses_timer.reset();

        Self {
            crown,
            life_timer,
            love_timer,
            protection_ready: true,
            destiny_ready: true,
            curses_timer,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FloorStarted {
    pub floor: u32,
    pub area: crate::data::AreaId,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RaceLoadout {
    pub unlocked: bool,
    pub unlocked_skins: [bool; 4],

    pub preferred_skin: u8,
    pub stored_weapon: WeaponId,
    pub start_weapon: WeaponId,
    pub start_crown: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HitId {
    Weapon(WeaponId),
    Explosion(WeaponId),

    Enemy(u16),
    Contact,
    Fire,
    Toxic,
    Trap,
    Crown,
    Other(u16),
}

impl HitId {
    #[inline]
    pub fn from_enemy_kind(kind: EnemyKind) -> Self {
        HitId::Enemy(kind as u16)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DamageSource {
    pub owner: Entity,
    pub team: Team,
    pub hit_id: HitId,

    pub enemy_kind: Option<EnemyKind>,
}

impl DamageSource {
    pub fn enemy(owner: Entity, kind: EnemyKind) -> Self {
        Self {
            owner,
            team: Team::Enemy,
            hit_id: HitId::from_enemy_kind(kind),
            enemy_kind: Some(kind),
        }
    }

    pub fn player_weapon(owner: Entity, wep: WeaponId) -> Self {
        Self {
            owner,
            team: Team::Player,
            hit_id: HitId::Weapon(wep),
            enemy_kind: None,
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct ProjectileFriction(pub f32);

#[derive(Component, Debug)]
pub struct GrenadeFuse {
    pub smoke_armed: bool,
    pub friction_switched: bool,
    pub alarm1: Timer,
}

#[derive(Component, Debug)]
pub struct ShellBonus {
    pub timer: Timer,
    pub bonus: i32,
}

/// GML FlameShell/Step_0: unlike the rest of the shell family it has no
/// fade anim and is destroyed outright once speed drops under 5 px/step
/// (Destroy spawns the Flame). Marker for the slow-death branch of
/// `tick_bullet2_fade`.
#[derive(Component, Clone, Copy, Debug)]
pub struct FlameShellSlowDeath;

#[derive(Component, Clone, Copy, Debug)]
pub struct ShellWallBounce {
    /// GML `wallbounce` in px/step: re-added to speed after the 0.8 cut.
    pub add: f32,
    /// GML speed cap in px/s (Bullet2 16, Slug/EBullet3 18).
    pub cap: f32,
    /// GML per-bounce decay (Bullet2 0.95, Slug/EBullet3 0.9).
    pub decay: f32,
    /// GML wall bonus re-arm: Some((threshold, amount)). Bullet2-class
    /// re-arms while wallbounce > 0, HeavySlug only while > 2, Slug and
    /// enemy shells never.
    pub rearm: Option<(f32, i32)>,
}

/// GML Slash/Shank projectile state (melee weapons fire real projectiles,
/// not hitscan). `typ`: 0 = nothing, 1 = deflectable, 2 = destructible.
/// `shank` passes through walls (screwdriver). `walled` latches after first
/// wall hit so MeleeHitWall + shake + sound fire once. `reach`/`back`/
/// `half_width` describe the oriented hitbox: GML sprites carry their origin
/// (e.g. sprSlash xorigin 0 = arc extends 48px forward), so hits must be
/// tested against the forward segment, not a circle at the entity.
#[derive(Component, Clone, Copy, Debug)]
pub struct SlashProjectile {
    pub typ: u8,
    pub shank: bool,
    pub walled: bool,
    pub hit: bool,
    pub guitar: bool,
    pub electric_guitar: bool,
    pub blood: bool,
    pub lightning: bool,
    pub hammer_wallbreak: bool,
    pub reach: f32,
    pub back: f32,
    pub half_width: f32,
    /// Last nonzero move direction (bevy `Transform.rotation`
    /// equivalent: bevy freezes rotation when the slash stops, so
    /// post-wall ticks keep testing the frozen direction).
    pub dir: Vec2,
}

/// GML Disc revert: team becomes neutral after leaving creator radius,
/// wall dies past travel distance.
#[derive(Component, Clone, Copy, Debug)]
pub struct DiscFlight {
    pub dist: f32,
    pub home: Vec2,
}

/// GML PlasmaBall shrink on wall hit (destroy at <= 0.5).
#[derive(Component, Clone, Copy, Debug)]
pub struct PlasmaSize(pub f32);

/// GML Bolt wall stick disables further damage (alarm[1]).
#[derive(Component, Clone, Copy, Debug)]
pub struct BoltWallDisable(pub bool);

/// GML `spr_fade`: smooth disappear anim spawned on projectile destroy
/// (projectile/Destroy -> scrBulletHitFX). None = destroyed silently
/// (Bolt, Disc, Plasma, Flak, Grenade, Slash have no spr_fade).
#[derive(Component, Clone, Copy, Debug)]
pub struct ProjectileFade(pub &'static str);

/// GML projectile `typ`: 0 = nothing (ignores slashes), 1 = deflectable
/// (bullets/shells/disc/grenades/flak/EBullet1/EBullet3), 2 = destructible
/// (bolts/plasma/rockets/EBullet2/horror). Slashes destroy typ 2 and
/// deflect typ 1 (shank destroys everything).
#[derive(Component, Clone, Copy, Debug)]
pub struct ProjectileTyp(pub u8);

#[derive(Component)]
pub struct Projectile {
    pub damage: i32,
    pub life: Timer,
    pub radius: f32,
    pub knockback: f32,
    pub explosive: bool,
    pub source: Option<DamageSource>,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct BouncesLeft(pub u8);

#[derive(Component, Clone, Copy, Debug)]
pub struct PiercesLeft(pub u8);

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct HitsAllTeams;

#[derive(Component, Debug)]
pub struct SpawnGrace(pub Timer);

#[derive(Component, Default, Debug, Clone)]
pub struct ProjectileHitSet(pub Vec<Entity>);

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct AbilityHazard;

#[derive(Component, Clone, Copy, Debug)]
pub struct SpawnHazardOnDeath(pub HazardDef);

#[derive(Component, Clone, Copy, Debug)]
pub struct SplitOnDeath(pub SplitDef);

#[derive(Component, Clone, Copy, Debug)]
pub struct Homing {
    pub turn_rate: f32,
    pub acquire_range: f32,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct Sticky {
    pub armed: bool,
    pub stuck_to: Option<Entity>,
    pub offset: Vec2,
}

impl Default for Sticky {
    fn default() -> Self {
        Self {
            armed: false,
            stuck_to: None,
            offset: Vec2::ZERO,
        }
    }
}

#[derive(Component, Debug)]
pub struct FlameTrail {
    pub timer: Timer,
    pub spec: HazardDef,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct ChainLightning {
    pub jumps_left: u8,
    pub range: f32,
    pub falloff: f32,
}

#[derive(Component, Debug)]
pub struct LightningArc {
    pub timer: Timer,
    /// Port adaptation: bevy carried these in the `Sprite` (custom
    /// size) and `Transform` rotation; the renderer needs them
    /// explicitly, so they live on the marker.
    pub len: f32,
    pub angle: f32,
}
