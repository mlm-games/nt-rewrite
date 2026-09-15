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
    Enemy, EnemySprites, FireAnim, HurtAnim, PlayerDying, Prop, PropHpTracker, PropSprites,
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
