//! Sprite animation state + switching. Sim-side only: systems advance
//! `frame` and select `path`; the render phase maps `(path, frame)` to
//! atlas uvs through `repame-anim` (no sprite handles here — the bevy
//! build wrote `Sprite.image/rect` inline, which belongs to rendering).
//!
//! Ported from nt's `game/anim.rs` (frame laws byte-identical).

use bevy_ecs::prelude::*;
use repame_anim::AnimCatalog;
use repame_sim::SimTime;

use crate::comps_a::{GmlHurtSprite, Health, Player, Velocity};
use crate::comps_b::{
    Enemy, EnemySprites, FireAnim, HurtAnim, HyperOrbitCrystal, PlayerDying, Prop, PropHpTracker,
    PropSprites,
};
use crate::time::{GTimer, TimerMode};

/// Frame-animated sprite state. `frames`/`fps` come from the catalog
/// def at (re)path time; the renderer resolves pixels from there.
#[derive(Component, Clone, Debug)]
pub struct SpriteAnim {
    pub path: String,
    pub frames: u32,
    pub fps: f32,
    pub frame: u32,
    pub timer: GTimer,
    pub oneshot: bool,
    pub finished: bool,
}

impl SpriteAnim {
    pub fn new(path: impl Into<String>, def: &repame_anim::AnimDef) -> Self {
        Self {
            path: path.into(),
            frames: def.frames.max(1),
            fps: def.fps,
            frame: 0,
            timer: GTimer::from_seconds(1.0 / def.fps.max(0.1), TimerMode::Repeating),
            oneshot: false,
            finished: false,
        }
    }

    pub fn oneshot(path: impl Into<String>, def: &repame_anim::AnimDef) -> Self {
        let mut a = Self::new(path, def);
        a.oneshot = true;
        a
    }

    pub fn set_path(&mut self, path: impl Into<String>, def: &repame_anim::AnimDef, oneshot: bool) {
        self.path = path.into();
        self.frames = def.frames.max(1);
        self.fps = def.fps;
        self.frame = 0;
        self.oneshot = oneshot;
        self.finished = false;
        self.timer = GTimer::from_seconds(1.0 / def.fps.max(0.1), TimerMode::Repeating);
    }
}

/// Advance every live animation one fixed step (bevy `animate_sprites`
/// parity: loop wraps, oneshot clamps on the last frame and parks).
pub fn animate_sprites(time: Res<SimTime>, mut q: Query<&mut SpriteAnim>) {
    for mut anim in &mut q {
        if anim.finished {
            continue;
        }
        anim.timer.tick(time.delta_secs);
        if anim.timer.just_finished() {
            if anim.oneshot {
                if anim.frame + 1 >= anim.frames.max(1) {
                    anim.frame = anim.frames.saturating_sub(1);
                    anim.finished = true;
                } else {
                    anim.frame += 1;
                }
            } else {
                anim.frame = (anim.frame + 1) % anim.frames.max(1);
            }
        }
    }
}

#[derive(Component)]
pub struct PlayerAnim {
    pub idle: &'static str,
    pub walk: &'static str,
    pub hurt: &'static str,
    pub moving: bool,
}

/// Swap player idle/walk strips on the movement threshold (render
/// writes — image handle, rect, anchor — happen renderer-side from the
/// new path).
pub fn player_anim_switch(
    catalog: Res<AnimCatalog>,
    mut q: Query<
        (&Velocity, &mut PlayerAnim, &mut SpriteAnim),
        (Without<HurtAnim>, Without<PlayerDying>),
    >,
) {
    for (vel, mut pa, mut anim) in &mut q {
        if anim.oneshot && !anim.finished {
            continue;
        }
        // GML `Player/Step_0.gml:192-197` verbatim: `if (!speed)` idle
        // else walk (hurt handled by the `Without<HurtAnim>` filter, cry
        // by `animation_end`). Any nonzero speed counts as moving.
        let moving = vel.0.length_squared() > 1e-6;
        if moving == pa.moving && !anim.oneshot {
            continue;
        }
        pa.moving = moving;
        let path = if moving { pa.walk } else { pa.idle };
        let Some(def) = catalog.def(path) else {
            continue;
        };
        anim.set_path(path, def, false);
    }
}

/// Swap enemy idle/walk strips (same render split as above).
pub fn enemy_anim_switch(
    catalog: Res<AnimCatalog>,
    mut q: Query<
        (&Velocity, &EnemySprites, &mut SpriteAnim),
        (With<Enemy>, Without<HurtAnim>, Without<FireAnim>),
    >,
) {
    for (vel, sprites, mut anim) in &mut q {
        if anim.oneshot && !anim.finished {
            continue;
        }

        let moving = vel.0.length_squared() > 1e-6;
        let idle = sprites.idle;
        let walk = sprites.walk.unwrap_or(idle);
        let desired = if moving { walk } else { idle };
        if anim.path == desired {
            continue;
        }

        if moving && sprites.walk.is_none() {
            if anim.path != idle {
                if let Some(def) = catalog.def(idle) {
                    anim.set_path(idle, def, false);
                }
            }
            continue;
        }
        let Some(def) = catalog.def(desired) else {
            continue;
        };
        anim.set_path(desired, def, false);
    }
}

/// Interrupt with the hurt strip (oneshot). Image/rect/anchor writes
/// happen renderer-side from the new path; the `HurtAnim` timer below
/// restores idle/walk when it lapses.
pub fn play_hurt(
    commands: &mut Commands,
    entity: Entity,
    catalog: &AnimCatalog,
    anim: &mut SpriteAnim,
    hurt_path: &'static str,
    idle: &'static str,
    walk: Option<&'static str>,
) {
    let Some(def) = catalog.def(hurt_path).or_else(|| catalog.def(idle)) else {
        return;
    };

    let path = if catalog.def(hurt_path).is_some() {
        hurt_path
    } else {
        idle
    };
    let def = catalog.def(path).unwrap_or(def);
    anim.set_path(path, def, true);

    let secs = (3.0 / def.fps.max(1.0)).max(0.12).min(0.35);

    commands.entity(entity).try_insert(HurtAnim {
        idle,
        walk,
        hurt: path,
        timer: GTimer::from_seconds(secs, TimerMode::Once),
        was_moving: false,
    });
}

/// HP-drop edge detection for enemies + the player (bevy
/// `hurt_on_damage` state half): on a fresh HP loss, switch to the
/// hurt strip via [`play_hurt`]. Image/rect/anchor writes and the
/// `FireAnim` removal happen renderer-side from the new path.
pub fn hurt_on_damage(
    mut commands: Commands,
    catalog: Res<AnimCatalog>,
    mut last_enemy_hp: Local<std::collections::HashMap<Entity, i32>>,
    mut last_player_hp: Local<std::collections::HashMap<Entity, i32>>,
    mut damaged: Query<
        (Entity, &Health, &EnemySprites, &mut SpriteAnim),
        (With<Enemy>, Without<HurtAnim>, Without<Player>),
    >,
    mut player_damaged: Query<
        (Entity, &Health, &PlayerAnim, &mut SpriteAnim),
        (With<Player>, Without<HurtAnim>, Without<Enemy>),
    >,
) {
    for (e, health, sprites, mut anim) in &mut damaged {
        let last = last_enemy_hp.get(&e).copied().unwrap_or(health.max);
        if health.hp >= health.max || health.hp <= 0 || health.hp == last {
            last_enemy_hp.insert(e, health.hp);
            continue;
        }
        last_enemy_hp.insert(e, health.hp);
        play_hurt(
            &mut commands,
            e,
            &catalog,
            &mut anim,
            sprites.hurt,
            sprites.idle,
            sprites.walk,
        );
        commands.entity(e).try_remove::<FireAnim>();
    }
    for (e, health, pa, mut anim) in &mut player_damaged {
        let last = last_player_hp.get(&e).copied().unwrap_or(health.max);
        if health.hp >= health.max || health.hp <= 0 || health.hp == last {
            last_player_hp.insert(e, health.hp);
            continue;
        }
        last_player_hp.insert(e, health.hp);
        play_hurt(
            &mut commands,
            e,
            &catalog,
            &mut anim,
            pa.hurt,
            pa.idle,
            Some(pa.walk),
        );
    }
}

/// HP-drop edge detection for destructible props (bevy
/// `prop_hurt_on_damage` state half). The `flip_x` force and anchor
/// write happen renderer-side.
pub fn prop_hurt_on_damage(
    mut commands: Commands,
    catalog: Res<AnimCatalog>,
    mut q: Query<
        (
            Entity,
            &Prop,
            &mut PropHpTracker,
            &PropSprites,
            Option<&mut SpriteAnim>,
            Option<&GmlHurtSprite>,
        ),
        (With<Prop>, Without<HurtAnim>),
    >,
) {
    for (e, prop, mut tracker, sprites, anim_opt, gml_opt) in &mut q {
        if !prop.destructible {
            tracker.last_hp = prop.hp;
            continue;
        }
        if prop.hp <= 0 {
            tracker.last_hp = prop.hp;
            continue;
        }
        if prop.hp >= tracker.last_hp {
            tracker.last_hp = prop.hp;
            continue;
        }
        tracker.last_hp = prop.hp;

        let Some(mut anim) = anim_opt else {
            continue;
        };

        // GML object identity wins when present: props with `hurt: None`
        // keep the existing generic flash (no path swap), others swap to
        // the extracted hurt strip.
        if let Some(gml) = gml_opt {
            match gml.hurt {
                Some(hurt) => {
                    play_hurt(
                        &mut commands,
                        e,
                        &catalog,
                        &mut anim,
                        hurt,
                        gml.normal,
                        None,
                    );
                }
                None => {
                    // No hit anim: keep existing behavior (no path swap).
                }
            }
            continue;
        }

        play_hurt(
            &mut commands,
            e,
            &catalog,
            &mut anim,
            sprites.hurt,
            sprites.idle,
            None,
        );
    }
}

/// Restore idle when the hurt strip lapses (GML verbatim:
/// `Player/Step_0.gml:199-202`, `enemy/Step_0.gml:27-29,39-41`,
/// `prop/Step_1.gml:8-10`: `if (sprite_index == spr_hurt &&
/// image_index > 2) sprite_index = spr_idle` — always idle, never walk,
/// no timer; GML's only `+5` is i-frames in `scr_hit`, not a visual
/// timer). `anim.frame > 2` is the `image_index > 2` equivalent; the
/// finished-oneshot arm is a safety net for strips shorter than 3
/// frames, which GML would loop past `> 2` but our oneshot clamps.
/// `PropSprites.flip_x` is intentionally NOT written here: the render
/// phase resolves prop facing straight from `PropSprites`.
pub fn tick_hurt_anims(
    time: Res<SimTime>,
    catalog: Res<AnimCatalog>,
    mut commands: Commands,
    mut q: Query<(
        Entity,
        &mut HurtAnim,
        &mut SpriteAnim,
        Option<&Velocity>,
        Option<&mut PlayerAnim>,
        Option<&PropSprites>,
    )>,
) {
    for (e, mut hurt, mut anim, _vel, mut pa, prop_sprites) in &mut q {
        hurt.timer.tick(time.delta_secs);
        let frame_done = anim.frame > 2;
        if !(frame_done || (anim.oneshot && anim.finished)) {
            continue;
        }
        // GML restores `spr_idle` unconditionally (never walk).
        let path = hurt.idle;
        if let Some(def) = catalog.def(path) {
            anim.set_path(path, def, false);
        }
        if let Some(ref mut pa) = pa {
            pa.moving = false;
        }
        let _ = prop_sprites;

        commands.entity(e).try_remove::<HurtAnim>();
    }
}

/// Interrupt with the muzzle-flash strip (oneshot, 0.25 s approximation:
/// GML uses per-enemy alarm periods — Guardian 12 steps, Wolf ~30+rand,
/// Crab 1-frame re-fire loop — not a global duration; 0.25 s keeps the
/// bevy parity until per-enemy alarm data is modeled).
pub fn play_fire(
    commands: &mut Commands,
    entity: Entity,
    catalog: &AnimCatalog,
    anim: &mut SpriteAnim,
    fire_path: &'static str,
    idle: &'static str,
    walk: Option<&'static str>,
) {
    let Some(def) = catalog.def(fire_path) else {
        return;
    };
    anim.set_path(fire_path, def, true);

    commands.entity(entity).try_insert(FireAnim {
        idle,
        walk,
        timer: GTimer::from_seconds(0.25, TimerMode::Once),
    });
}

/// Restore idle when the flash strip lapses (GML verbatim: Wolf/Guardian
/// alarms restore `spr_idle`; the walk strip re-engages next step via the
/// `speed != 0` switch when the actor is still moving).
pub fn tick_fire_anims(
    time: Res<SimTime>,
    catalog: Res<AnimCatalog>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut FireAnim, &mut SpriteAnim, Option<&Velocity>), Without<HurtAnim>>,
) {
    for (e, mut fire, mut anim, _vel) in &mut q {
        fire.timer.tick(time.delta_secs);
        if !fire.timer.just_finished() {
            continue;
        }
        let path = fire.idle;
        if let Some(def) = catalog.def(path) {
            anim.set_path(path, def, false);
        }
        commands.entity(e).try_remove::<FireAnim>();
    }
}

/// Death-knell drain (bevy `anim.rs:1059` parity: the `PlayerDying`
/// timer ticks and the husk despawns on lapse — pure logic, no strips).
pub fn tick_player_dying(
    time: Res<SimTime>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut PlayerDying)>,
) {
    for (e, mut dying) in &mut q {
        dying.timer.tick(time.delta_secs);
        if dying.timer.just_finished() {
            commands.entity(e).try_despawn();
        }
    }
}

/// Backfill strip state for pre-asset spawns (headless `setup_run` runs
/// against the empty catalog, so actors spawn with no `SpriteAnim` and —
/// for enemies — no `EnemySprites` table; without them the switch/hurt
/// queries never match and actors stick on the render fallback frame 0).
/// Runs once from `App::load_assets_from` after the full catalog lands;
/// post-asset spawns already carry both and are skipped.
pub fn backfill_spawn_anims(world: &mut World) {
    let idle_of = |kind: crate::data::EnemyKind| -> &'static str {
        crate::enemy_data::enemy_def(kind).sprite
    };
    let enemy_rows: Vec<(Entity, crate::data::EnemyKind, bool, bool)> = {
        let mut q = world.query::<(
            Entity,
            &Enemy,
            Option<&EnemySprites>,
            Option<&SpriteAnim>,
            Option<&HyperOrbitCrystal>,
        )>();
        q.iter(world)
            .filter_map(|(e, enemy, sprites, anim, orbit)| {
                // Hyper orbit crystals are bevy-parity visual markers (tinted
                // sprite, no strip table or anim); giving them `EnemySprites`
                // would drag them into the enemy anim/hurt queries.
                orbit.is_none().then(|| {
                    (e, enemy.kind, sprites.is_some(), anim.is_some())
                })
            })
            .collect()
    };
    let player_rows: Vec<(Entity, String, bool)> = {
        let mut q = world.query::<(Entity, &PlayerAnim, Option<&SpriteAnim>)>();
        q.iter(world)
            .map(|(e, pa, anim)| (e, pa.idle.to_string(), anim.is_some()))
            .collect()
    };
    let catalog = world.resource::<AnimCatalog>();
    let mut enemy_inserts: Vec<(Entity, Option<EnemySprites>, Option<SpriteAnim>)> =
        Vec::new();
    for (e, kind, has_sprites, has_anim) in &enemy_rows {
        let idle = idle_of(*kind);
        let sprites = if *has_sprites {
            None
        } else {
            Some(EnemySprites {
                idle,
                walk: derive_walk_path(idle),
                hurt: derive_hurt_path(idle),
            })
        };
        let anim = if *has_anim {
            None
        } else {
            catalog
                .def(idle)
                .map(|def| SpriteAnim::new(idle, def))
        };
        if sprites.is_some() || anim.is_some() {
            enemy_inserts.push((*e, sprites, anim));
        }
    }
    let mut player_inserts: Vec<(Entity, SpriteAnim)> = Vec::new();
    for (e, idle, has_anim) in &player_rows {
        if *has_anim {
            continue;
        }
        if let Some(def) = catalog.def(idle.as_str()) {
            player_inserts.push((*e, SpriteAnim::new(idle.clone(), def)));
        }
    }
    for (e, sprites, anim) in enemy_inserts {
        if let Some(sprites) = sprites {
            world.entity_mut(e).insert(sprites);
        }
        if let Some(anim) = anim {
            world.entity_mut(e).insert(anim);
        }
    }
    for (e, anim) in player_inserts {
        world.entity_mut(e).insert(anim);
    }
}
pub fn derive_hurt_path(idle: &'static str) -> &'static str {
    let base = if idle.contains("sprMutant") {
        if idle.contains("sprMutant1BIdle") {
            return "images/sprMutant1BHurt.png";
        }
        if idle.contains("sprMutant1CIdle") {
            return "images/sprMutant1CHurt.png";
        }
        if idle.contains("sprMutant2BIdle") {
            return "images/sprMutant2BHurt.png";
        }
        if idle.contains("sprMutant2CIdle") {
            return "images/sprMutant2CHurt.png";
        }
        if idle.contains("sprMutant3BIdle") {
            return "images/sprMutant3BHurt.png";
        }
        if idle.contains("sprMutant3CIdle") {
            return "images/sprMutant3CHurt.png";
        }
        if idle.contains("sprMutant4BIdle") {
            return "images/sprMutant4BHurt.png";
        }
        if idle.contains("sprMutant4CIdle") {
            return "images/sprMutant4CHurt.png";
        }
        if idle.contains("sprMutant5BIdle") {
            return "images/sprMutant5BHurt.png";
        }
        if idle.contains("sprMutant5CIdle") {
            return "images/sprMutant5CHurt.png";
        }
        if idle.contains("sprMutant6BIdle") {
            return "images/sprMutant6BHurt.png";
        }
        if idle.contains("sprMutant6CIdle") {
            return "images/sprMutant6CHurt.png";
        }
        if idle.contains("sprMutant7BIdle") {
            return "images/sprMutant7BHurt.png";
        }
        if idle.contains("sprMutant7CIdle") {
            return "images/sprMutant7CHurt.png";
        }
        if idle.contains("sprMutant8BIdle") {
            return "images/sprMutant8BHurt.png";
        }
        if idle.contains("sprMutant8CIdle") {
            return "images/sprMutant8CHurt.png";
        }
        if idle.contains("sprMutant9BIdle") {
            return "images/sprMutant9BHurt.png";
        }
        if idle.contains("sprMutant9CIdle") {
            return "images/sprMutant9CHurt.png";
        }
        if idle.contains("sprMutant10BIdle") {
            return "images/sprMutant10BHurt.png";
        }
        if idle.contains("sprMutant10CIdle") {
            return "images/sprMutant10CHurt.png";
        }
        if idle.contains("sprMutant11BIdle") {
            return "images/sprMutant11BHurt.png";
        }
        if idle.contains("sprMutant11CIdle") {
            return "images/sprMutant11CHurt.png";
        }
        if idle.contains("sprMutant12BIdle") {
            return "images/sprMutant12BHurt.png";
        }
        if idle.contains("sprMutant12CIdle") {
            return "images/sprMutant12CHurt.png";
        }
        if idle.contains("sprMutant13BIdle") {
            return "images/sprMutant13BHurt.png";
        }
        if idle.contains("sprMutant13CIdle") {
            return "images/sprMutant13CHurt.png";
        }
        if idle.contains("sprMutant14BIdle") {
            return "images/sprMutant14BHurt.png";
        }
        if idle.contains("sprMutant14CIdle") {
            return "images/sprMutant14CHurt.png";
        }
        if idle.contains("sprMutant15BIdle") {
            return "images/sprMutant15BHurt.png";
        }
        if idle.contains("sprMutant15CIdle") {
            return "images/sprMutant15CHurt.png";
        }
        if idle.contains("sprMutant16BIdle") {
            return "images/sprMutant16BHurt.png";
        }
        if idle.contains("sprMutant16CIdle") {
            return "images/sprMutant16CHurt.png";
        }

        idle
    } else {
        idle
    };
    match base {
        "images/sprBanditIdle.png" => "images/sprBanditHurt.png",
        "images/sprMaggotIdle.png" => "images/sprMaggotHurt.png",
        "images/sprScorpionIdle.png" => "images/sprScorpionHurt.png",
        "images/sprRatIdle.png" => "images/sprRatHurt.png",
        "images/sprRatkingIdle.png" => "images/sprRatkingHurt.png",
        "images/sprFreak1Idle.png" => "images/sprFreak1Hurt.png",
        "images/sprJungleAssassinIdle.png" => "images/sprJungleAssassinHurt.png",
        "images/sprSnowBotIdle.png" => "images/sprSnowBotHurt.png",
        "images/sprTurretIdle.png" => "images/sprTurretHurt.png",
        "images/sprSnowBanditIdle.png" => "images/sprSnowBanditHurt.png",
        "images/sprWolfIdle.png" => "images/sprWolfHurt.png",
        "images/sprBanditBossIdle.png" => "images/sprBanditBossHurt.png",

        "images/sprGatorIdle.png" => "images/sprGatorHurt.png",
        "images/sprBuffGatorIdle.png" => "images/sprBuffGatorHurt.png",
        "images/sprRavenIdle.png" => "images/sprRavenHurt.png",
        "images/sprSalamanderIdle.png" => "images/sprSalamanderHurt.png",
        "images/sprMeleeIdle.png" => "images/sprMeleeHurt.png",
        "images/sprJungleBanditIdle.png" => "images/sprJungleBanditHurt.png",
        "images/sprBigMaggotIdle.png" => "images/sprBigMaggotHurt.png",
        "images/sprFastRatIdle.png" => "images/sprFastRatHurt.png",
        "images/sprGoldScorpionIdle.png" => "images/sprGoldScorpionHurt.png",
        "images/sprLightningCrystalIdle.png" => "images/sprLightningCrystalHurt.png",
        "images/sprExploFreakIdle.png" => "images/sprExploFreakHurt.png",
        "images/sprRhinoFreakIdle.png" => "images/sprRhinoFreakHurt.png",
        "images/sprSnowTankIdle.png" => "images/sprSnowTankHurt.png",
        "images/sprGoldTankIdle.png" => "images/sprGoldTankHurt.png",
        "images/sprGuardianIdle.png" => "images/sprGuardianHurt.png",
        "images/sprExploGuardianIdle.png" => "images/sprExploGuardianHurt.png",
        "images/sprDogGuardianWalk.png" => "images/sprDogGuardianHurt.png",

        "images/sprBoneFish1Idle.png" => "images/sprBoneFish1Hurt.png",
        "images/sprTurtleIdle.png" => "images/sprTurtleHurt.png",
        "images/sprMolefishIdle.png" => "images/sprMolefishHurt.png",
        "images/sprMolesargeIdle.png" => "images/sprMolesargeHurt.png",
        "images/sprFireBallerIdle.png" => "images/sprFireBallerHurt.png",
        "images/sprSuperFireBallerIdle.png" => "images/sprSuperFireBallerHurt.png",
        "images/sprJockIdle.png" => "images/sprJockHurt.png",
        "images/sprJungleFlyIdle.png" => "images/sprJungleFlyHurt.png",
        "images/sprInvSpiderIdle.png" => "images/sprInvSpiderHurt.png",
        "images/sprInvLaserCrystalIdle.png" => "images/sprInvLaserCrystalHurt.png",
        "images/sprPopoFreakIdle.png" => "images/sprPopoFreakHurt.png",
        "images/sprMSpawnIdle.png" => "images/sprMSpawnHurt.png",
        "images/sprSniperIdle.png" => "images/sprSniperHurt.png",
        "images/sprCrabIdle.png" => "images/sprCrabHurt.png",
        "images/sprSpiderIdle.png" => "images/sprSpiderHurt.png",
        "images/sprNecromancerIdle.png" => "images/sprNecromancerHurt.png",
        "images/sprExploderIdle.png" => "images/sprExploderHurt.png",
        "images/sprLaserCrystalIdle.png" => "images/sprLaserCrystalHurt.png",

        "images/sprFrogQueenIdle.png" => "images/sprFrogQueenHurt.png",

        "images/sprMutant1Idle.png" => "images/sprMutant1Hurt.png",
        "images/sprMutant2Idle.png" => "images/sprMutant2Hurt.png",
        "images/sprMutant3Idle.png" => "images/sprMutant3Hurt.png",
        "images/sprMutant4Idle.png" => "images/sprMutant4Hurt.png",
        "images/sprMutant5Idle.png" => "images/sprMutant5Hurt.png",
        "images/sprMutant6Idle.png" => "images/sprMutant6Hurt.png",
        "images/sprMutant7Idle.png" => "images/sprMutant7Hurt.png",
        "images/sprMutant8Idle.png" => "images/sprMutant8Hurt.png",
        "images/sprMutant9Idle.png" => "images/sprMutant9Hurt.png",
        "images/sprMutant10Idle.png" => "images/sprMutant10Hurt.png",
        "images/sprMutant11Idle.png" => "images/sprMutant11Hurt.png",
        "images/sprMutant12Idle.png" => "images/sprMutant12Hurt.png",
        "images/sprMutant13Idle.png" => "images/sprMutant13Hurt.png",
        "images/sprMutant14Idle.png" => "images/sprMutant14Hurt.png",
        "images/sprMutant15Idle.png" => "images/sprMutant15Hurt.png",
        "images/sprMutant16Idle.png" => "images/sprMutant16Hurt.png",
        "images/sprMimicIdle.png" => "images/sprMimicHurt.png",
        "images/sprSuperMimicIdle.png" => "images/sprSuperMimicHurt.png",
        "images/sprWepMimicIdle.png" => "images/sprWepMimicHurt.png",
        "images/sprScrapBossIdle.png" => "images/sprScrapBossHurt.png",
        "images/sprLilHunter.png" => "images/sprLilHunterHurt.png",
        "images/sprLilHunterIdle.png" => "images/sprLilHunterHurt.png",
        "images/sprHyperCrystalIdle.png" => "images/sprHyperCrystalHurt.png",
        "images/sprGruntIdle.png" => "images/sprGruntHurt.png",
        "images/sprShielderIdle.png" => "images/sprShielderHurt.png",
        "images/sprEliteGruntIdle.png" => "images/sprEliteGruntHurt.png",
        "images/sprEliteShielderIdle.png" => "images/sprEliteShielderHurt.png",
        "images/sprEliteInspectorIdle.png" => "images/sprEliteInspectorHurt.png",
        "images/sprInspectorIdle.png" => "images/sprInspectorHurt.png",
        "images/sprFrogEgg.png" => "images/sprFrogEggHurt.png",
        "images/sprTechnoMancer.png" => "images/sprTechnoMancerHurt.png",
        "images/sprVanDrive.png" => "images/sprVanHurt.png",
        "images/sprSpookyBanditIdle.png" => "images/sprSpookyBanditHurt.png",
        "images/sprYVBossIdle.png" => "images/sprYVBossHurt.png",
        "images/sprCrownGuardianIdle.png" => "images/sprCrownGuardianHurt.png",
        "images/sprIceFlowerIdle.png" => "images/sprIceFlowerHurt.png",
        "images/sprSuperFrogIdle.png" => "images/sprSuperFrogHurt.png",
        "images/sprRadMaggotIdle.png" => "images/sprRadMaggotHurt.png",
        "images/sprLastIdle.png" => "images/sprLastHurt.png",
        "images/sprNothing2Idle.png" => "images/sprNothing2Hurt.png",
        "images/sprEnemyHorrorIdle.png" => "images/sprEnemyHorrorHurt.png",
        "images/sprFiredMaggot.png" => "images/sprFiredMaggot.png",
        "images/sprRadMaggot.png" => "images/sprRadMaggotHurt.png",
        "images/sprBarrel.png" => "images/sprBarrelHurt.png",
        "images/sprToxicBarrel.png" => "images/sprToxicBarrelHurt.png",
        "images/sprGoldBarrel.png" => "images/sprGoldBarrelHurt.png",
        "images/sprOasisBarrel.png" => "images/sprOasisBarrelHurt.png",
        "images/sprCactus.png" => "images/sprCactusHurt.png",
        "images/sprCactus2.png" => "images/sprCactus2Hurt.png",
        "images/sprCactus3.png" => "images/sprCactus3Hurt.png",
        "images/sprCactusB.png" => "images/sprCactusBHurt.png",
        "images/sprCactusB2.png" => "images/sprCactusB2Hurt.png",
        "images/sprCactusB3.png" => "images/sprCactusB3Hurt.png",
        "images/sprNightCactus.png" => "images/sprNightCactusHurt.png",
        "images/sprNightCactus2.png" => "images/sprNightCactus2Hurt.png",
        "images/sprNightCactus3.png" => "images/sprNightCactus3Hurt.png",
        "images/sprSewerPipe.png" => "images/sprSewerPipeHurt.png",
        "images/sprPipe.png" => "images/sprSewerPipeHurt.png",
        "images/sprTires.png" => "images/sprTiresHurt.png",
        "images/sprCocoon.png" => "images/sprCocoonHurt.png",
        "images/sprSnowMan.png" => "images/sprSnowManHurt.png",
        "images/sprSnowManIdle.png" => "images/sprSnowManHurt.png",
        "images/sprTorch.png" => "images/sprTorchHurt.png",
        "images/sprBonePileIdle.png" => "images/sprBonePileHurt.png",
        "images/sprBonePile.png" => "images/sprBonePileHurt.png",
        "images/sprNightBonePileIdle.png" => "images/sprNightBonePileHurt.png",
        "images/sprBushIdle.png" => "images/sprBushHurt.png",
        "images/sprBush.png" => "images/sprBushHurt.png",
        "images/sprBigFlowerIdle.png" => "images/sprBigFlowerHurt.png",
        "images/sprPlantPotIdle.png" => "images/sprPlantPotHurt.png",
        "images/sprCrystalProp.png" => "images/sprCrystalPropHurt.png",
        "images/sprHydrant.png" => "images/sprHydrantHurt.png",
        "images/sprIcicle.png" => "images/sprIcicleHurt.png",
        "images/sprStreetLight.png" => "images/sprStreetLightHurt.png",
        "images/sprSodaMachine.png" => "images/sprSodaMachineHurt.png",
        "images/sprNewsStand.png" => "images/sprNewsStandHurt.png",
        "images/sprTube.png" => "images/sprTubeHurt.png",
        "images/sprMutantTube.png" => "images/sprMutantTubeHurt.png",
        "images/sprNuclearPillar.png" => "images/sprNuclearPillarHurt.png",
        "images/sprPillar.png" => "images/sprNuclearPillarHurt.png",
        "images/sprSmallGenerator.png" => "images/sprSmallGeneratorHurt.png",
        "images/sprBigGenerator.png" => "images/sprBigGeneratorHurt.png",
        "images/sprGenerator.png" => "images/sprBigGeneratorHurt.png",
        "images/sprAnchor.png" => "images/sprAnchorHurt.png",
        "images/sprWaterPlant.png" => "images/sprWaterPlantHurt.png",
        "images/sprWaterPlant2.png" => "images/sprWaterPlantHurt.png",
        "images/sprWaterMine.png" => "images/sprWaterMineHurt.png",
        "images/sprMine.png" => "images/sprMineHurt.png",
        "images/sprMineIdle.png" => "images/sprMineHurt.png",
        "images/sprMoneyPile.png" => "images/sprMoneyPileHurt.png",
        "images/sprYVStatue.png" => "images/sprYVStatueHurt.png",
        "images/sprPizzaBox.png" => "images/sprPizzaBoxHurt.png",
        "images/sprCarIdle.png" => "images/sprCarHurt.png",
        "images/sprFrozenCar.png" => "images/sprFrozenCarHurt.png",
        "images/sprBigSkullOpen.png" => "images/sprBigSkullOpenHurt.png",
        "images/sprBigSkull.png" => "images/sprBigSkullOpenHurt.png",
        "images/sprRadChest.png" => "images/sprRadChestHurt.png",
        "images/sprRadChestIdle.png" => "images/sprRadChestHurt.png",
        "images/sprThroneStatue.png" => "images/sprThroneStatue.png",
        _ => idle,
    }
}


pub fn derive_dead_path(idle: &'static str) -> &'static str {
    let base = if idle.contains("sprMutant") {
        if idle.contains("sprMutant1BIdle") {
            return "images/sprMutant1BDead.png";
        }
        if idle.contains("sprMutant1CIdle") {
            return "images/sprMutant1CDead.png";
        }
        if idle.contains("sprMutant2BIdle") {
            return "images/sprMutant2BDead.png";
        }
        if idle.contains("sprMutant2CIdle") {
            return "images/sprMutant2CDead.png";
        }
        if idle.contains("sprMutant3BIdle") {
            return "images/sprMutant3BDead.png";
        }
        if idle.contains("sprMutant3CIdle") {
            return "images/sprMutant3CDead.png";
        }
        if idle.contains("sprMutant4BIdle") {
            return "images/sprMutant4BDead.png";
        }
        if idle.contains("sprMutant4CIdle") {
            return "images/sprMutant4CDead.png";
        }
        if idle.contains("sprMutant5BIdle") {
            return "images/sprMutant5BDead.png";
        }
        if idle.contains("sprMutant5CIdle") {
            return "images/sprMutant5CDead.png";
        }
        if idle.contains("sprMutant6BIdle") {
            return "images/sprMutant6BDead.png";
        }
        if idle.contains("sprMutant6CIdle") {
            return "images/sprMutant6CDead.png";
        }
        if idle.contains("sprMutant7BIdle") {
            return "images/sprMutant7BDead.png";
        }
        if idle.contains("sprMutant7CIdle") {
            return "images/sprMutant7CDead.png";
        }
        if idle.contains("sprMutant8BIdle") {
            return "images/sprMutant8BDead.png";
        }
        if idle.contains("sprMutant8CIdle") {
            return "images/sprMutant8CDead.png";
        }
        if idle.contains("sprMutant9BIdle") {
            return "images/sprMutant9BDead.png";
        }
        if idle.contains("sprMutant9CIdle") {
            return "images/sprMutant9CDead.png";
        }
        if idle.contains("sprMutant10BIdle") {
            return "images/sprMutant10BDead.png";
        }
        if idle.contains("sprMutant10CIdle") {
            return "images/sprMutant10CDead.png";
        }
        if idle.contains("sprMutant11BIdle") {
            return "images/sprMutant11BDead.png";
        }
        if idle.contains("sprMutant11CIdle") {
            return "images/sprMutant11CDead.png";
        }
        if idle.contains("sprMutant12BIdle") {
            return "images/sprMutant12BDead.png";
        }
        if idle.contains("sprMutant12CIdle") {
            return "images/sprMutant12CDead.png";
        }
        if idle.contains("sprMutant13BIdle") {
            return "images/sprMutant13BDead.png";
        }
        if idle.contains("sprMutant13CIdle") {
            return "images/sprMutant13CDead.png";
        }
        if idle.contains("sprMutant14BIdle") {
            return "images/sprMutant14BDead.png";
        }
        if idle.contains("sprMutant14CIdle") {
            return "images/sprMutant14CDead.png";
        }
        if idle.contains("sprMutant15BIdle") {
            return "images/sprMutant15BDead.png";
        }
        if idle.contains("sprMutant15CIdle") {
            return "images/sprMutant15CDead.png";
        }
        if idle.contains("sprMutant16BIdle") {
            return "images/sprMutant16BDead.png";
        }
        if idle.contains("sprMutant16CIdle") {
            return "images/sprMutant16CDead.png";
        }
        idle
    } else {
        idle
    };
    match base {
        "images/sprMutant1Idle.png" => "images/sprMutant1Dead.png",
        "images/sprMutant2Idle.png" => "images/sprMutant2Dead.png",
        "images/sprMutant3Idle.png" => "images/sprMutant3Dead.png",
        "images/sprMutant4Idle.png" => "images/sprMutant4Dead.png",
        "images/sprMutant5Idle.png" => "images/sprMutant5Dead.png",
        "images/sprMutant6Idle.png" => "images/sprMutant6Dead.png",
        "images/sprMutant7Idle.png" => "images/sprMutant7Dead.png",
        "images/sprMutant8Idle.png" => "images/sprMutant8Dead.png",
        "images/sprMutant9Idle.png" => "images/sprMutant9Dead.png",
        "images/sprMutant10Idle.png" => "images/sprMutant10Dead.png",
        "images/sprMutant11Idle.png" => "images/sprMutant11Dead.png",
        "images/sprMutant12Idle.png" => "images/sprMutant12Dead.png",
        "images/sprMutant13Idle.png" => "images/sprMutant13Dead.png",
        "images/sprMutant14Idle.png" => "images/sprMutant14Dead.png",
        "images/sprMutant15Idle.png" => "images/sprMutant15Dead.png",
        "images/sprMutant16Idle.png" => "images/sprMutant16Dead.png",
        "images/sprBanditIdle.png" => "images/sprBanditDead.png",
        "images/sprMaggotIdle.png" => "images/sprMaggotDead.png",
        "images/sprScorpionIdle.png" => "images/sprScorpionDead.png",
        "images/sprRatIdle.png" => "images/sprRatDead.png",
        "images/sprRatkingIdle.png" => "images/sprRatkingDead.png",
        "images/sprFreak1Idle.png" => "images/sprFreak1Dead.png",
        "images/sprJungleAssassinIdle.png" => "images/sprJungleAssassinDead.png",
        "images/sprSnowBotIdle.png" => "images/sprSnowBotDead.png",
        "images/sprTurretIdle.png" => "images/sprTurretDead.png",
        "images/sprSnowBanditIdle.png" => "images/sprSnowBanditDead.png",
        "images/sprWolfIdle.png" => "images/sprWolfDead.png",
        "images/sprBanditBossIdle.png" => "images/sprBanditBossDead.png",
        "images/sprGatorIdle.png" => "images/sprGatorDead.png",
        "images/sprBuffGatorIdle.png" => "images/sprBuffGatorDead.png",
        "images/sprRavenIdle.png" => "images/sprRavenDead.png",
        "images/sprSalamanderIdle.png" => "images/sprSalamanderDead.png",
        "images/sprMeleeIdle.png" => "images/sprMeleeDead.png",
        "images/sprJungleBanditIdle.png" => "images/sprJungleBanditDead.png",
        "images/sprBigMaggotIdle.png" => "images/sprBigMaggotDead.png",
        "images/sprFastRatIdle.png" => "images/sprFastRatDead.png",
        "images/sprGoldScorpionIdle.png" => "images/sprGoldScorpionDead.png",
        "images/sprLightningCrystalIdle.png" => "images/sprLightningCrystalDead.png",
        "images/sprExploFreakIdle.png" => "images/sprExploFreakDead.png",
        "images/sprRhinoFreakIdle.png" => "images/sprRhinoFreakDead.png",
        "images/sprSnowTankIdle.png" => "images/sprSnowTankDead.png",
        "images/sprGoldTankIdle.png" => "images/sprGoldTankDead.png",
        "images/sprGuardianIdle.png" => "images/sprGuardianDead.png",
        "images/sprExploGuardianIdle.png" => "images/sprExploGuardianDead.png",
        "images/sprDogGuardianWalk.png" => "images/sprDogGuardianDead.png",
        "images/sprBoneFish1Idle.png" => "images/sprBoneFish1Dead.png",
        "images/sprTurtleIdle.png" => "images/sprTurtleDead.png",
        "images/sprMolefishIdle.png" => "images/sprMolefishDead.png",
        "images/sprMolesargeIdle.png" => "images/sprMolesargeDead.png",
        "images/sprFireBallerIdle.png" => "images/sprFireBallerDead.png",
        "images/sprSuperFireBallerIdle.png" => "images/sprSuperFireBallerDead.png",
        "images/sprJockIdle.png" => "images/sprJockDead.png",
        "images/sprJungleFlyIdle.png" => "images/sprJungleFlyDead.png",
        "images/sprInvSpiderIdle.png" => "images/sprInvSpiderDead.png",
        "images/sprInvLaserCrystalIdle.png" => "images/sprInvLaserCrystalDead.png",
        "images/sprPopoFreakIdle.png" => "images/sprPopoFreakDead.png",
        "images/sprMSpawnIdle.png" => "images/sprMSpawnDead.png",
        "images/sprFrogQueenIdle.png" => "images/sprFrogQueenDead.png",
        "images/sprSniperIdle.png" => "images/sprSniperDead.png",
        "images/sprCrabIdle.png" => "images/sprCrabDead.png",
        "images/sprSpiderIdle.png" => "images/sprSpiderDead.png",
        "images/sprNecromancerIdle.png" => "images/sprNecromancerDead.png",
        "images/sprExploderIdle.png" => "images/sprExploderDead.png",
        "images/sprLaserCrystalIdle.png" => "images/sprLaserCrystalDead.png",
        "images/sprMimicIdle.png" => "images/sprMimicDead.png",
        "images/sprSuperMimicIdle.png" => "images/sprSuperMimicDead.png",
        "images/sprWepMimicIdle.png" => "images/sprWepMimicDead.png",
        "images/sprScrapBossIdle.png" => "images/sprScrapBossDead.png",
        "images/sprLilHunter.png" => "images/sprLilHunterDead.png",
        "images/sprLilHunterIdle.png" => "images/sprLilHunterDead.png",
        "images/sprHyperCrystalIdle.png" => "images/sprHyperCrystalDead.png",
        "images/sprGruntIdle.png" => "images/sprGruntDead.png",
        "images/sprShielderIdle.png" => "images/sprShielderDead.png",
        "images/sprEliteGruntIdle.png" => "images/sprEliteGruntDead.png",
        "images/sprEliteShielderIdle.png" => "images/sprEliteShielderDead.png",
        "images/sprEliteInspectorIdle.png" => "images/sprEliteInspectorDead.png",
        "images/sprInspectorIdle.png" => "images/sprInspectorDead.png",
        "images/sprFrogEgg.png" => "images/sprFrogEggDead.png",
        "images/sprCrystalProp.png" => "images/sprCrystalPropDead.png",
        "images/sprTechnoMancer.png" => "images/sprTechnoMancerDead.png",
        "images/sprVanDrive.png" => "images/sprVanDead.png",
        "images/sprSpookyBanditIdle.png" => "images/sprSpookyBanditDead.png",
        "images/sprYVBossIdle.png" => "images/sprYVBossDead.png",
        "images/sprCrownGuardianIdle.png" => "images/sprCrownGuardianDead.png",
        "images/sprIceFlowerIdle.png" => "images/sprIceFlowerDead.png",
        "images/sprSuperFrogIdle.png" => "images/sprSuperFrogDead.png",
        "images/sprRadMaggotIdle.png" => "images/sprRadMaggotDead.png",
        "images/sprLastIdle.png" => "images/sprLastDeath.png",
        "images/sprNothing2Idle.png" => "images/sprNothing2Death.png",
        "images/sprEnemyHorrorIdle.png" => "images/sprEnemyHorrorDead.png",
        "images/sprFiredMaggot.png" => "images/sprMaggotDead.png",
        "images/sprRadMaggot.png" => "images/sprRadMaggotDead.png",
        _ => idle,
    }
}


pub fn derive_walk_path(idle: &'static str) -> Option<&'static str> {
    match idle {
        "images/sprBanditIdle.png" => Some("images/sprBanditWalk.png"),
        "images/sprRatIdle.png" => Some("images/sprRatWalk.png"),
        "images/sprFreak1Idle.png" => Some("images/sprFreak1Walk.png"),
        "images/sprJungleAssassinIdle.png" => Some("images/sprJungleAssassinWalk.png"),
        "images/sprSnowBotIdle.png" => Some("images/sprSnowBotWalk.png"),
        "images/sprSnowBanditIdle.png" => Some("images/sprSnowBanditWalk.png"),
        "images/sprWolfIdle.png" => Some("images/sprWolfWalk.png"),
        "images/sprScorpionIdle.png" => Some("images/sprScorpionWalk.png"),
        "images/sprSpiderIdle.png" => Some("images/sprSpiderWalk.png"),
        "images/sprCrabIdle.png" => Some("images/sprCrabWalk.png"),
        "images/sprSniperIdle.png" => Some("images/sprSniperWalk.png"),
        "images/sprNecromancerIdle.png" => Some("images/sprNecromancerWalk.png"),
        "images/sprExploderIdle.png" => Some("images/sprExploderWalk.png"),
        "images/sprGatorIdle.png" => Some("images/sprGatorWalk.png"),
        "images/sprBuffGatorIdle.png" => Some("images/sprBuffGatorWalk.png"),
        "images/sprRavenIdle.png" => Some("images/sprRavenWalk.png"),
        "images/sprSalamanderIdle.png" => Some("images/sprSalamanderWalk.png"),
        "images/sprMeleeIdle.png" => Some("images/sprMeleeWalk.png"),
        "images/sprJungleBanditIdle.png" => Some("images/sprJungleBanditWalk.png"),
        "images/sprFastRatIdle.png" => Some("images/sprFastRatWalk.png"),
        "images/sprRatkingIdle.png" => Some("images/sprRatkingWalk.png"),
        "images/sprGoldScorpionIdle.png" => Some("images/sprGoldScorpionWalk.png"),
        "images/sprExploFreakIdle.png" => Some("images/sprExploFreakWalk.png"),
        "images/sprRhinoFreakIdle.png" => Some("images/sprRhinoFreakWalk.png"),
        "images/sprSnowTankIdle.png" => Some("images/sprSnowTankWalk.png"),
        "images/sprGoldTankIdle.png" => Some("images/sprGoldTankWalk.png"),
        "images/sprExploGuardianIdle.png" => Some("images/sprExploGuardianWalk.png"),
        "images/sprGuardianIdle.png" => Some("images/sprGuardianWalk.png"),
        "images/sprBanditBossIdle.png" => Some("images/sprBanditBossWalk.png"),
        "images/sprFireBallerIdle.png" => Some("images/sprFireBallerWalk.png"),
        "images/sprSuperFireBallerIdle.png" => Some("images/sprSuperFireBallerWalk.png"),
        "images/sprBoneFish1Idle.png" => Some("images/sprBoneFish1Walk.png"),
        "images/sprMolefishIdle.png" => Some("images/sprMolefishWalk.png"),
        "images/sprMolesargeIdle.png" => Some("images/sprMolesargeWalk.png"),
        "images/sprJockIdle.png" => Some("images/sprJockWalk.png"),
        "images/sprJungleFlyIdle.png" => Some("images/sprJungleFlyWalk.png"),
        "images/sprInvSpiderIdle.png" => Some("images/sprInvSpiderWalk.png"),
        "images/sprPopoFreakIdle.png" => Some("images/sprPopoFreakWalk.png"),
        "images/sprFrogQueenIdle.png" => Some("images/sprFrogQueenWalk.png"),
        "images/sprMimicIdle.png" => Some("images/sprMimicFire.png"),
        "images/sprSuperMimicIdle.png" => Some("images/sprSuperMimicFire.png"),
        "images/sprWepMimicIdle.png" => Some("images/sprWepMimicFire.png"),
        _ => None,
    }
}

