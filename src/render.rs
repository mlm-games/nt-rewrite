//! World → GPU sprite layer: sim [`World`] snapshots become
//! [`SpriteInstance`]s through the repame engine crates.
//!
//! Render split (renderer resolves art, sim owns truth):
//! - [`RenderAssets`] packs `assets/images/anims.json` strips into a
//!   [`repame_anim::AnimCatalog`] once and decodes the strip PNGs (via
//   the `image` crate — repame ships no image dep; per
//   `repame-atlas` docs games decode their own pixels).
// - [`world_instances`] maps floor cells, walls, props, pickups/chests,
//   player, enemies, projectiles and held-gun visuals to atlas uvs.
//   Entity→strip NAME tables are ported from the bevy build
//   (`world.rs` area sprites, `projectile_art.rs`, `pickups.rs`,
//   `weapon_id_sprite`); no pixels live here.
// - Portal/vortex is intentionally NOT sprite-mapped: it belongs to
//   the fullscreen background pass (`crate::vortex_pass::VortexPass`),
//   driven per frame from the [`crate::vortex::SpiralCtl`] snapshot.
//   This layer exposes [`background_color`] (per-area dark fill) as the
//   fallback where no vortex layer is mounted, and skips portal entities.
//   See fidelity notes at the bottom.
// - [`gml_camera_step`] is the GML `BackCont` camera law as a pure
//   function; [`world_camera`] turns the look point into a
//   [`Camera2d`] for the GPU view.
// - [`hud_texts`] exposes [`HudState`] labels as world-anchored
//   `(text, pos)` pairs; [`hud_texts_dp`] maps them to dp through the
//   live [`effective_fit`]/[`world_to_dp`] (same fit the viewports
//   paint with, so floaters sit on their sprites).

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use bevy_ecs::prelude::*;
use glam::Vec2;
use repame_anim::{AnimCatalog, AnimDef, AtlasDesc, UvRect, frame_key};
use repame_atlas::AtlasId;
use repame_fx::{DamageNumber, Particle, particle_sprites};
use repame_sprite::{
    AtlasUpload, BatchDesc, Camera2d, SpriteBlend, SpriteInstance, WorldText, dp_to_world,
    effective_fit, world_to_dp,
};

use crate::anim::{PlayerAnim, SpriteAnim};
use crate::audio::UiAction;
use crate::combat::HitFlash;
use crate::comps_a::{
    ARENA_H, ARENA_W, AimDir, FloorMask, GrenadeFuse, Health, HitId, Inventory, LightningArc,
    PendingMutation, PendingUltra, Player, Projectile, RaceState, Run, SelectedCharacter,
    SlashProjectile, TILE, Team, Velocity, WallCell, WallTile,
};
use crate::comps_b::{
    Beam, BossBrain, BossPhase, ChestKind, Corpse, Enemy, EnemyBrain, FxAngle, GroundDecalTint,
    HazardCloud, OpenedChest, Pickup, PickupKind, PickupLifetime, Portal, PortalClear, PortalShock,
    PortalStrike, Prop, PropSprites, StaticFx, SwingFx, Telekinesis, ThroneCarpet, ThroneSit,
    WeaponVisual, YvCouch,
};
use crate::data::{
    AreaId, CrownKind, EnemyKind, HazardKind, MutationId, RaceId, UltraMutationId, WeaponId,
};
use crate::environment::{EnvironmentHazard, PulseSprite, SurfacePulse};
use crate::hud::{HudState, run_area_string, run_timer_string, sync_hud_state};
use crate::savedata_part::{character_def, race_active_text, race_passive_text};
use crate::spatial::Pos;
use crate::state::menus::{CHAR_SELECT_ORDER, MenuState};
use crate::state::{OverlayMenu, SPLASH_GUN_STEPS, SplashState};
use crate::weapon_runtime::{sanitize_weapon_id, weapon_meta};
use crate::weapons_data::AmmoType;

/// Atlas page edge for full-catalog loads (matches the engine default).
pub const ATLAS_SIZE: u32 = 2048;
/// Page cap: the real catalog (2066 anims, ~11.8k frames) fits in 16
/// pages @2048 (see `repame-anim`'s `packs_full_nt_catalog` proof).
pub const ATLAS_PAGES: u32 = 16;

/// Unused historical constant (bevy lookahead); kept for the public
/// API while the GML law drives the camera.
pub const MAX_LOOK: f32 = 48.0;

/// Packed atlas + decoded strip pixels + one-shot GPU uploads.
pub struct RenderAssets {
    catalog: AnimCatalog,
    uploads: Vec<AtlasUpload>,
    atlas_size: u32,
    layers: u32,
}

impl RenderAssets {
    fn build(json: &str, assets_dir: &Path, desc: AtlasDesc) -> anyhow::Result<Self> {
        let mut catalog = AnimCatalog::from_json(json, desc).map_err(|e| anyhow::anyhow!("{e}"))?;
        // Decode every strip PNG the catalog references. Missing art is a
        // magenta placeholder (visible bug, never a panic or a hole).
        let mut strips = HashMap::new();
        for name in catalog.names() {
            let path = assets_dir.join("images").join(format!("{name}.png"));
            let (w, h, rgba) = match decode_png(&path) {
                Ok(px) => px,
                Err(e) => {
                    log::warn!("render assets: {path:?}: {e}; magenta placeholder");
                    let def = catalog.def(name).expect("just listed");
                    (def.w.max(1), def.h.max(1), vec![255, 0, 255, 255])
                }
            };
            // Placeholder is 1px; expand to the def size.
            let rgba = if rgba.len() == 4 {
                rgba.repeat((w * h) as usize)
            } else {
                rgba
            };
            strips.insert(name.to_string(), (w, h, rgba));
        }
        // Reverse-map drain keys → (stem, frame) for strip blits.
        let mut key_to_frame: HashMap<AtlasId, (String, u32)> = HashMap::new();
        for name in catalog.names() {
            let def = catalog.def(name).expect("just listed");
            for frame in 0..def.frames {
                key_to_frame.insert(frame_key(name, frame), (name.to_string(), frame));
            }
        }
        let mut uploads = Vec::new();
        for write in catalog.atlas_mut().drain_writes() {
            let rgba = match key_to_frame.get(&write.key) {
                Some((stem, frame)) => {
                    let src = catalog
                        .src_rect(stem, *frame as i32)
                        .expect("reverse-mapped");
                    let (sw, _sh, px) = strips.get(stem.as_str()).expect("decoded above");
                    blit_cell(px, *sw, src)
                }
                None => {
                    log::warn!("render assets: orphan atlas write; magenta fill");
                    vec![255, 0, 255, 255].repeat((write.w * write.h) as usize)
                }
            };
            uploads.push(AtlasUpload::from_write(&write, rgba));
        }
        let layers = catalog.atlas().page_count().max(1);
        Ok(Self {
            catalog,
            uploads,
            atlas_size: desc.size,
            layers,
        })
    }

    /// Full production load: `assets_dir` holds `images/anims.json` +
    /// the strip PNGs (e.g. `../nt-recreated-bevy/assets`).
    pub fn load(assets_dir: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(assets_dir.join("images").join("anims.json"))?;
        Self::build(
            &json,
            assets_dir,
            AtlasDesc {
                size: ATLAS_SIZE,
                max_pages: ATLAS_PAGES,
                padding: 0,
            },
        )
    }

    /// Tiny load for headless tests: packs only `names` (stems or full
    /// `images/…` paths) into a small multi-page atlas.
    pub fn load_subset(assets_dir: &Path, names: &[&str]) -> anyhow::Result<Self> {
        Self::load_subset_sized(assets_dir, names, 512, 2)
    }

    /// Sized subset load for headless tests whose strips exceed the
    /// tiny page (fog tiles, multi-frame portraits).
    pub fn load_subset_sized(
        assets_dir: &Path,
        names: &[&str],
        size: u32,
        max_pages: u32,
    ) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(assets_dir.join("images").join("anims.json"))?;
        let raw: HashMap<String, serde_json::Value> = serde_json::from_str(&json)?;
        let want: std::collections::HashSet<String> = names
            .iter()
            .map(|n| repame_anim::stem(n).to_string())
            .collect();
        let filtered: HashMap<String, serde_json::Value> = raw
            .into_iter()
            .filter(|(k, _)| want.contains(repame_anim::stem(k)))
            .collect();
        anyhow::ensure!(
            filtered.len() == want.len(),
            "subset strips missing from anims.json: want {want:?}, kept {}",
            filtered.len()
        );
        Self::build(
            &serde_json::to_string(&filtered)?,
            assets_dir,
            AtlasDesc {
                size,
                max_pages,
                padding: 0,
            },
        )
    }

    /// Uv + def for one strip frame (accepts stems and full paths).
    pub fn uv(&self, path: &str, frame: i32) -> Option<(UvRect, &AnimDef)> {
        Some((self.catalog.uv(path, frame)?, self.catalog.def(path)?))
    }

    /// [`BatchDesc`] matching this pack (layer size + page count).
    pub fn batch_desc(&self) -> BatchDesc {
        BatchDesc {
            layer_size: self.atlas_size,
            layers: self.layers,
            ..Default::default()
        }
    }

    /// One-shot GPU uploads (atlas blits). Empties the queue.
    pub fn take_uploads(&mut self) -> Vec<AtlasUpload> {
        std::mem::take(&mut self.uploads)
    }

    pub(crate) fn sprite_for(
        &self,
        path: &str,
        frame: i32,
        center: Vec2,
        flip_x: bool,
        rotation: f32,
        tint: [f32; 4],
    ) -> Option<SpriteInstance> {
        self.sprite_for_full(path, frame, center, flip_x, false, rotation, tint)
    }

    /// Native strip-cell size (for GML `image_xscale`-style multiples).
    pub(crate) fn native_size(&self, path: &str) -> Option<Vec2> {
        let (_, def) = self.uv(path, 0)?;
        Some(Vec2::new(def.w as f32, def.h as f32))
    }

    /// Full sprite constructor including the engine `flip_y` channel
    /// (bevy `Sprite.flip_y` — held guns mirror when aiming left).
    #[allow(clippy::too_many_arguments)]
    fn sprite_for_full(
        &self,
        path: &str,
        frame: i32,
        center: Vec2,
        flip_x: bool,
        flip_y: bool,
        rotation: f32,
        tint: [f32; 4],
    ) -> Option<SpriteInstance> {
        let (uv, def) = self.uv(path, frame)?;
        let anchor = def.anchor();
        Some(SpriteInstance {
            center,
            rotation,
            size: Vec2::new(def.w as f32, def.h as f32),
            anchor: Vec2::new(anchor[0], anchor[1]),
            flip_x,
            flip_y,
            uv_min: Vec2::new(uv.min[0], uv.min[1]),
            uv_max: Vec2::new(uv.max[0], uv.max[1]),
            color: tint_to_linear(tint),
            page: uv.page,
            z: 0.0,
            blend: SpriteBlend::Alpha,
        })
    }

    /// Sized textured sprite (bevy `custom_size` parity for pulse
    /// decals: cobweb/ice/trap visuals drawn at their recorded size,
    /// not the native strip cell).
    fn sprite_sized(
        &self,
        path: &str,
        frame: i32,
        center: Vec2,
        size: Vec2,
        flip_x: bool,
        tint: [f32; 4],
    ) -> Option<SpriteInstance> {
        let (uv, def) = self.uv(path, frame)?;
        let anchor = def.anchor();
        Some(SpriteInstance {
            center,
            rotation: 0.0,
            size,
            anchor: Vec2::new(anchor[0], anchor[1]),
            flip_x,
            flip_y: false,
            uv_min: Vec2::new(uv.min[0], uv.min[1]),
            uv_max: Vec2::new(uv.max[0], uv.max[1]),
            color: tint_to_linear(tint),
            page: uv.page,
            z: 0.0,
            blend: SpriteBlend::Alpha,
        })
    }

    /// Scaled + rotated textured sprite (GML `draw_sprite_ext` parity
    /// for the spiral center figures: native strip cell times `scale`,
    /// rotation in radians).
    fn sprite_scaled_rotated(
        &self,
        path: &str,
        frame: i32,
        center: Vec2,
        scale: f32,
        rotation: f32,
        tint: [f32; 4],
    ) -> Option<SpriteInstance> {
        self.sprite_stretched(path, frame, center, Vec2::splat(scale), rotation, tint)
    }

    /// Non-uniformly scaled + rotated textured sprite (GML loadout
    /// panel parity).
    fn sprite_stretched(
        &self,
        path: &str,
        frame: i32,
        center: Vec2,
        scale: Vec2,
        rotation: f32,
        tint: [f32; 4],
    ) -> Option<SpriteInstance> {
        let (uv, def) = self.uv(path, frame)?;
        let anchor = def.anchor();
        Some(SpriteInstance {
            center,
            rotation,
            size: Vec2::new(def.w as f32 * scale.x, def.h as f32 * scale.y),
            anchor: Vec2::new(anchor[0], anchor[1]),
            flip_x: false,
            flip_y: false,
            uv_min: Vec2::new(uv.min[0], uv.min[1]),
            uv_max: Vec2::new(uv.max[0], uv.max[1]),
            color: tint_to_linear(tint),
            page: uv.page,
            z: 0.0,
            blend: SpriteBlend::Alpha,
        })
    }
}

pub(crate) fn decode_png(path: &Path) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let img = image::ImageReader::open(path)?.decode()?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Ok((w, h, rgba.into_raw()))
}

/// Crop one horizontal-strip cell `[x, y, w, h]` out of a full strip.
fn blit_cell(strip: &[u8], strip_w: u32, src: [u32; 4]) -> Vec<u8> {
    let [sx, sy, w, h] = src;
    let (w, h) = (w.max(1), h.max(1));
    // Degenerate strip (e.g. 1px placeholder): stretch to the cell.
    if strip.len() == 4 {
        return strip.repeat((w * h) as usize);
    }
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h {
        let y = sy + row;
        let start = ((y * strip_w + sx) * 4) as usize;
        let end = start + (w * 4) as usize;
        if end <= strip.len() {
            out.extend_from_slice(&strip[start..end]);
        } else {
            log::warn!("render assets: strip overrun at frame blit; magenta row");
            out.extend_from_slice(&vec![255, 0, 255, 255].repeat(w as usize));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Name tables (ported from the bevy build; art keys only, no pixels).
// ---------------------------------------------------------------------------

/// GML `scrAreaGetMaxSubarea` verbatim (non-custom): the 3-floor
/// areas plus HQ hold 3 subareas, everything else 1. The custom-mode
/// branch (`area_size`/`area_size_alt`) is deferred — the port has no
/// custom runs, so the static table always applies.
pub fn area_max_subarea(area: AreaId) -> u32 {
    match area {
        AreaId::Desert | AreaId::Scrapyards | AreaId::FrozenCity | AreaId::Palace | AreaId::HQ => 3,
        _ => 1,
    }
}

/// Floor/wall/decal strips for a route floor.
/// Bevy `world.rs::area_sprites` verbatim (floor index over the 15-floor
/// route; wall `Out`/`Trans` variants exist but the renderer only needs
/// floor + bot/top — see fidelity notes).
fn area_sprites(floor: u32) -> (&'static str, &'static str, &'static str) {
    let rf = ((floor.max(1) - 1) % 15) + 1;
    match rf {
        3 => (
            "images/sprFloor0.png",
            "images/sprWall0Bot.png",
            "images/sprWall0Top.png",
        ),
        4 => (
            "images/sprFloor2.png",
            "images/sprWall2Bot.png",
            "images/sprWall2Top.png",
        ),
        5..=7 => (
            "images/sprFloor3.png",
            "images/sprWall3Bot.png",
            "images/sprWall3Top.png",
        ),
        8 => (
            "images/sprFloor4.png",
            "images/sprWall4Bot.png",
            "images/sprWall4Top.png",
        ),
        9..=11 => (
            "images/sprFloor5.png",
            "images/sprWall5Bot.png",
            "images/sprWall5Top.png",
        ),
        12 => (
            "images/sprFloor6.png",
            "images/sprWall6Bot.png",
            "images/sprWall6Top.png",
        ),
        13..=15 => (
            "images/sprFloor7.png",
            "images/sprWall7Bot.png",
            "images/sprWall7Top.png",
        ),
        _ => (
            "images/sprFloor1.png",
            "images/sprWall1Bot.png",
            "images/sprWall1Top.png",
        ),
    }
}

/// Route floor/wall/out/trans strips for a route floor, mirroring bevy
/// `world.rs::area_sprites` (out/trans follow the same per-floor arms).
fn area_sprites_full(
    floor: u32,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    let (f, b, t) = area_sprites(floor);
    let rf = ((floor.max(1) - 1) % 15) + 1;
    let n: u8 = match rf {
        3 => 0,
        4 => 2,
        5..=7 => 3,
        8 => 4,
        9..=11 => 5,
        12 => 6,
        13..=15 => 7,
        _ => 1,
    };
    // Out/Trans follow the same per-floor arms bevy uses
    // (sprWall{N}Out / sprWall{N}Trans).
    let (o, tr): (&'static str, &'static str) = match n {
        0 => ("images/sprWall0Out.png", "images/sprWall0Trans.png"),
        2 => ("images/sprWall2Out.png", "images/sprWall2Trans.png"),
        3 => ("images/sprWall3Out.png", "images/sprWall3Trans.png"),
        4 => ("images/sprWall4Out.png", "images/sprWall4Trans.png"),
        5 => ("images/sprWall5Out.png", "images/sprWall5Trans.png"),
        6 => ("images/sprWall6Out.png", "images/sprWall6Trans.png"),
        7 => ("images/sprWall7Out.png", "images/sprWall7Trans.png"),
        _ => ("images/sprWall1Out.png", "images/sprWall1Trans.png"),
    };
    (f, b, t, o, tr)
}

/// Dark surround tile for the padded floor bounds (currently unused:
/// GML draws the room background colour plus live floor cells only — no
/// outside ring — so the viewport stays transparent over the vortex
/// layer. Kept for the public API while the GML law is re-verified).
#[allow(dead_code)]
fn outside_sprite_for_run(floor: u32, has: impl Fn(&str) -> bool) -> &'static str {
    if has("images/sprFloorEx1.png") {
        return "images/sprFloorEx1.png";
    }
    let rf = ((floor.max(1) - 1) % 15) + 1;
    match rf {
        4 => "images/sprFloor2.png",
        5..=7 => "images/sprFloor3.png",
        8 => "images/sprFloor4.png",
        9..=11 => "images/sprFloor5.png",
        12 => "images/sprFloor6.png",
        13..=15 => "images/sprFloor7.png",
        3 => "images/sprFloor0.png",
        _ => "images/sprFloor1.png",
    }
}

/// Secret-area floor/wall override.
/// Bevy `world.rs::area_sprites_for_run` verbatim (Oasis→101 … HQ→106),
/// with the bevy `catalog.has` fallback to the route strips when the
/// secret tile PNG is absent from this pack.
fn area_sprites_for_run(
    floor: u32,
    area: AreaId,
    has: impl Fn(&str) -> bool,
) -> (&'static str, &'static str, &'static str) {
    let route = area_sprites(floor);
    let secret: Option<(&'static str, &'static str, &'static str)> = match area {
        // GML `Floor/Create_0` verbatim: the campfire title (`MenuGen`
        // / `Menu` present) uses `sprFloor0` + area-0 walls — the dark
        // slate camp, never the desert route strips.
        AreaId::Campfire => Some((
            "images/sprFloor0.png",
            "images/sprWall0Bot.png",
            "images/sprWall0Top.png",
        )),
        AreaId::Oasis => Some((
            "images/sprFloor101.png",
            "images/sprWall101Bot.png",
            "images/sprWall101Top.png",
        )),
        AreaId::PizzaSewers => Some((
            "images/sprFloor102.png",
            "images/sprWall102Bot.png",
            "images/sprWall102Top.png",
        )),
        AreaId::City => Some((
            "images/sprFloor103.png",
            "images/sprWall103Bot.png",
            "images/sprWall103Top.png",
        )),
        AreaId::CursedCaves | AreaId::Vault | AreaId::CrownVault => Some((
            "images/sprFloor104.png",
            "images/sprWall104Bot.png",
            "images/sprWall104Top.png",
        )),
        AreaId::Jungle => Some((
            "images/sprFloor105.png",
            "images/sprWall105Bot.png",
            "images/sprWall105Top.png",
        )),
        AreaId::HQ => Some((
            "images/sprFloor106.png",
            "images/sprWall106Bot.png",
            "images/sprWall106Top.png",
        )),
        _ => None,
    };
    match secret {
        Some((f, b, t)) => (
            if has(f) { f } else { route.0 },
            if has(b) { b } else { route.1 },
            if has(t) { t } else { route.2 },
        ),
        None => route,
    }
}

/// Full route+secret strips for wall compositing (floor, bot, top, out,
/// trans). Bevy `area_sprites_for_run` verbatim: secret 101–106 families
/// with per-slot fallback to the route strips when the pack lacks them.
fn area_sprites_full_for_run(
    floor: u32,
    area: AreaId,
    has: impl Fn(&str) -> bool + Copy,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    let route = area_sprites_full(floor);
    // GML `Floor/Create_0`: campfire uses the area-0 family throughout
    // (floor, bot, top, out, trans) with the same per-slot catalog
    // fallback the secret areas use.
    if area == AreaId::Campfire {
        let (rf, rb, rt) = area_sprites_for_run(floor, area, has);
        let o = "images/sprWall0Out.png";
        let tr = "images/sprWall0Trans.png";
        return (
            rf,
            rb,
            rt,
            if has(o) { o } else { route.3 },
            if has(tr) { tr } else { route.4 },
        );
    }
    let num: u8 = match area {
        AreaId::Oasis => 101,
        AreaId::PizzaSewers => 102,
        AreaId::City => 103,
        AreaId::CursedCaves | AreaId::Vault | AreaId::CrownVault => 104,
        AreaId::Jungle => 105,
        AreaId::HQ => 106,
        _ => 0,
    };
    if num == 0 {
        return route;
    }
    let (o, tr): (&'static str, &'static str) = match num {
        101 => ("images/sprWall101Out.png", "images/sprWall101Trans.png"),
        102 => ("images/sprWall102Out.png", "images/sprWall102Trans.png"),
        103 => ("images/sprWall103Out.png", "images/sprWall103Trans.png"),
        104 => ("images/sprWall104Out.png", "images/sprWall104Trans.png"),
        105 => ("images/sprWall105Out.png", "images/sprWall105Trans.png"),
        _ => ("images/sprWall106Out.png", "images/sprWall106Trans.png"),
    };
    let (rf, rb, rt) = area_sprites_for_run(floor, area, has);
    (
        rf,
        rb,
        rt,
        if has(o) { o } else { route.3 },
        if has(tr) { tr } else { route.4 },
    )
}

/// Player projectile strip by weapon id.
/// Bevy `projectile_art.rs::player_projectile_path` verbatim (ids 1–128;
/// GOLDEN/ULTRA/CURSED prefixes sanitize to base via
/// [`sanitize_weapon_id`]; unknown ids fall back by ammo type).
pub fn player_projectile_path(id: WeaponId) -> &'static str {
    let id = sanitize_weapon_id(id);
    if id == WeaponId::NONE {
        return "images/sprBullet1.png";
    }
    match id.0 {
        1 => "images/sprBullet1.png",
        2 => "images/sprBullet1.png",
        3 => "images/sprSlash.png",
        4 => "images/sprBullet1.png",
        5 => "images/sprBullet2.png",
        6 => "images/sprBolt.png",
        7 => "images/sprGrenade.png",
        8 => "images/sprBullet2.png",
        9 => "images/sprBullet1.png",
        10 => "images/sprBullet2.png",
        11 => "images/sprBolt.png",
        12 => "images/sprBolt.png",
        13 => "images/sprSlash.png",
        14 => "images/sprRocket.png",
        15 => "images/sprStickyGrenade.png",
        16 => "images/sprBullet1.png",
        17 => "images/sprBullet1.png",
        18 => "images/sprDisc.png",
        19 => "images/sprLaser.png",
        20 => "images/sprLaser.png",
        21 => "images/sprSlugBullet.png",
        22 => "images/sprSlugBullet.png",
        23 => "images/sprSlugBullet.png",
        24 => "images/sprEnergySlash.png",
        25 => "images/sprSlugBullet.png",
        26 => "images/sprBullet1.png",
        27 => "images/sprShank.png",
        28 => "images/sprLaser.png",
        29 => "images/sprBloodGrenade.png",
        30 => "images/sprSplinter.png",
        31 => "images/sprToxicBolt.png",
        32 => "images/sprBullet1.png",
        33 => "images/sprBullet2.png",
        34 => "images/sprPlasmaBall.png",
        35 => "images/sprPlasmaBallBig.png",
        36 => "images/sprEnergyHammer.png",
        37 => "images/sprShank.png",
        38 => "images/sprFlakBullet.png",
        39 => "images/sprBullet1.png",
        40 => "images/sprSlash.png",
        41 => "images/sprBullet1.png",
        42 => "images/sprBullet2.png",
        43 => "images/sprBoltGold.png",
        44 => "images/sprGoldGrenade.png",
        45 => "images/sprLaser.png",
        46 => "images/sprSlash.png",
        47 => "images/sprNuke.png",
        48 => "images/sprPlasmaBall.png",
        49 => "images/sprBullet1.png",
        50 => "images/sprTrapFire.png",
        51 => "images/sprTrapFire.png",
        52 => "images/sprFlare.png",
        53 => "images/sprEnergySlash.png",
        54 => "images/sprPopoNade.png",
        55 => "images/sprLaser.png",
        56 => "images/sprBullet1.png",
        57 => "images/sprLightning.png",
        58 => "images/sprLightning.png",
        59 => "images/sprLightning.png",
        60 => "images/sprSuperFlakBullet.png",
        61 => "images/sprBullet2.png",
        62 => "images/sprSplinter.png",
        63 => "images/sprSplinter.png",
        64 => "images/sprLightning.png",
        65 => "images/sprBullet1.png",
        66 => "images/sprHeavyBolt.png",
        67 => "images/sprBloodSlash.png",
        68 => "images/sprLightningBall.png",
        69 => "images/sprBullet2.png",
        70 => "images/sprPlasmaBall.png",
        71 => "images/sprBullet2.png",
        72 => "images/sprToxicGrenade.png",
        73 => "images/sprFlameBall.png",
        74 => "images/sprLightningSlash.png",
        75 => "images/sprFireShell.png",
        76 => "images/sprFireShell.png",
        77 => "images/sprFireShell.png",
        78 => "images/sprClusterNade.png",
        79 => "images/sprMininade.png",
        80 => "images/sprMininade.png",
        81 => "images/sprBullet1.png",
        82 => "images/sprConfettiBall.png",
        83 => "images/sprBullet1.png",
        84 => "images/sprRocket.png",
        85 => "images/sprMininade.png",
        86 => "images/sprUltraBullet.png",
        87 => "images/sprLaser.png",
        88 => "images/sprSlash.png",
        89 => "images/sprHeavyBullet.png",
        90 => "images/sprHeavyBullet.png",
        91 => "images/sprHeavySlug.png",
        92 => "images/sprUltraSlash.png",
        93 => "images/sprUltraShell.png",
        94 => "images/sprUltraBolt.png",
        95 => "images/sprUltraGrenade.png",
        96 => "images/sprPlasmaBall.png",
        97 => "images/sprPlasmaBall.png",
        98 => "images/sprPlasmaBall.png",
        99 => "images/sprSlugBullet.png",
        100 => "images/sprSplinter.png",
        101 => "images/sprShank.png",
        102 => "images/sprGoldRocket.png",
        103 => "images/sprBullet1.png",
        104 => "images/sprDisc.png",
        105 => "images/sprHeavyBolt.png",
        106 => "images/sprHeavyBullet.png",
        107 => "images/sprBloodBall.png",
        108 => "images/sprSlash.png",
        109 => "images/sprRocket.png",
        110 => "images/sprFireShell.png",
        111 => "images/sprPlasmaBallHuge.png",
        112 => "images/sprSeeker.png",
        113 => "images/sprSeeker.png",
        114 => "images/sprBullet2.png",
        115 => "images/sprSlash.png",
        116 => "images/sprBouncerBullet.png",
        117 => "images/sprBouncerBullet.png",
        118 => "images/sprSlugBullet.png",
        119 => "images/sprRocket.png",
        120 => "images/sprScorpionBullet.png",
        121 => "images/sprSlash.png",
        122 => "images/sprGoldNuke.png",
        123 => "images/sprGoldDisc.png",
        124 => "images/sprHeavyNade.png",
        125 => "images/sprBullet1.png",
        126 => "images/sprBullet1.png",
        127 => "images/sprScorpionBullet.png",
        128 => "images/sprSlash.png",
        _ => match weapon_meta(id).wep_type {
            AmmoType::Shells => "images/sprBullet2.png",
            AmmoType::Bolts => "images/sprBolt.png",
            AmmoType::Explosives => "images/sprGrenade.png",
            AmmoType::Energy => "images/sprPlasmaBall.png",
            AmmoType::None => "images/sprSlash.png",
            AmmoType::Bullets => "images/sprBullet1.png",
        },
    }
}

/// Enemy projectile strip by owner kind.
/// Bevy `projectile_art.rs::enemy_projectile_path` verbatim; unlisted
/// kinds (including nt-rewrite-only additions) fall back to the generic
/// enemy bullet, exactly like bevy's `_` arm.
pub fn enemy_projectile_path(kind: EnemyKind) -> &'static str {
    match kind {
        EnemyKind::Scorpion | EnemyKind::GoldScorpion => "images/sprScorpionBullet.png",
        EnemyKind::Jock => "images/sprJockRocket.png",
        EnemyKind::SnowTank | EnemyKind::GoldSnowtank => "images/sprEnemyBullet4.png",
        EnemyKind::Guardian => "images/sprGuardianBullet.png",
        EnemyKind::ExploGuardian => "images/sprHorrorBullet.png",
        EnemyKind::DogGuardian => "images/sprHeavyBullet.png",
        EnemyKind::LaserCrystal | EnemyKind::InvLaserCrystal => "images/sprEnemyLaser.png",
        EnemyKind::LightningCrystal => "images/sprEnemyLightning.png",
        EnemyKind::Crystal => "images/sprEnemyBullet1.png",
        EnemyKind::FireBaller | EnemyKind::SuperFireBaller => "images/sprFlameBall.png",
        EnemyKind::Turtle => "images/sprGuardianBullet.png",
        EnemyKind::Sniper => "images/sprEnemyBullet4.png",
        EnemyKind::Bandit | EnemyKind::SnowBandit | EnemyKind::MeleeBandit => {
            "images/sprEnemyBullet1.png"
        }
        EnemyKind::JungleBandit | EnemyKind::Molesarge => "images/sprEBullet3.png",
        EnemyKind::Rat | EnemyKind::BigRat | EnemyKind::FastRat | EnemyKind::Ratking => {
            "images/sprEnemyBullet1.png"
        }
        EnemyKind::Gator | EnemyKind::BuffGator => "images/sprEBullet3.png",
        EnemyKind::Raven => "images/sprEnemyBullet1.png",
        EnemyKind::Spider | EnemyKind::InvSpider => "images/sprEnemyBullet1.png",
        EnemyKind::Salamander => "images/sprSalamanderBullet.png",
        EnemyKind::Freak | EnemyKind::RhinoFreak | EnemyKind::ExploFreak | EnemyKind::PopoFreak => {
            "images/sprEnemyBullet1.png"
        }
        EnemyKind::IdpdGrunt | EnemyKind::IdpdShield | EnemyKind::IdpdElite => {
            "images/sprIDPDBullet.png"
        }
        EnemyKind::IdpdInspector => "images/sprPopoSlug.png",
        EnemyKind::Maggot | EnemyKind::BigMaggot | EnemyKind::MaggotSpawn => {
            "images/sprMaggotBullet.png"
        }
        _ => "images/sprEnemyBullet1.png",
    }
}

/// Pickup strip path.
/// GML per-kind enemy gun strips (`Draw_0` gun layer; Jock's is
/// commented out in GML and draws none).
fn enemy_gun_art(kind: EnemyKind) -> Option<&'static str> {
    match kind {
        EnemyKind::Bandit | EnemyKind::SnowBandit => Some("images/sprBanditGun.png"),
        EnemyKind::BigBandit | EnemyKind::BigBanditLoop => Some("images/sprBanditBossGun.png"),
        EnemyKind::IdpdGrunt => Some("images/sprPopoGun.png"),
        EnemyKind::IdpdElite => Some("images/sprElitePopoGun.png"),
        EnemyKind::JungleBandit => Some("images/sprJungleBanditGun.png"),
        EnemyKind::LilHunter | EnemyKind::LilHunterLoop => Some("images/sprLilHunterGun.png"),
        EnemyKind::Molefish => Some("images/sprMolefishGun.png"),
        EnemyKind::Molesarge => Some("images/sprMolesargeGun.png"),
        EnemyKind::Raven => Some("images/sprRavenGun.png"),
        EnemyKind::PopoFreak => Some("images/sprPopoFreakGun.png"),
        _ => None,
    }
}

/// Bevy `pickups.rs::pickup_sprite` paths verbatim (its sizes are
/// vestigial — `spawn_pickup` ignores them and draws native strip
/// frames via `sprite_exact`; quads here size from the catalog cell
/// the same way). Weapon pickups use the `weapon_id_sprite` law
/// (`images/{wep_sprt}.png`, else revolver).
pub fn pickup_art(kind: &PickupKind) -> Cow<'static, str> {
    match *kind {
        PickupKind::Rad(_) => Cow::Borrowed("images/sprRad.png"),
        PickupKind::Medkit(_) => Cow::Borrowed("images/sprHP.png"),
        PickupKind::Ammo(..) => Cow::Borrowed("images/sprAmmo.png"),
        PickupKind::Curse => Cow::Borrowed("images/sprCurse.png"),
        PickupKind::Weapon(w) => {
            let meta = weapon_meta(w);
            if !meta.wep_sprt.is_empty() && meta.wep_sprt != "mskNone" {
                Cow::Owned(format!("images/{}.png", meta.wep_sprt))
            } else {
                Cow::Borrowed("images/sprRevolver.png")
            }
        }
        PickupKind::Chest(kind) => match kind {
            ChestKind::Weapon => Cow::Borrowed("images/sprWeaponChest.png"),
            ChestKind::Ammo => Cow::Borrowed("images/sprAmmoChest.png"),
            ChestKind::Rad => Cow::Borrowed("images/sprRadChest.png"),
            ChestKind::Health => Cow::Borrowed("images/sprHealthChest.png"),
            ChestKind::CursedBig => Cow::Borrowed("images/sprCursedChestBig.png"),
            ChestKind::Rogue => Cow::Borrowed("images/sprRogueAmmoChest.png"),
            ChestKind::Proto => Cow::Borrowed("images/sprProtoChest.png"),
            ChestKind::BigWeapon => Cow::Borrowed("images/sprWeaponChestBig.png"),
            ChestKind::RadBig => Cow::Borrowed("images/sprRadChestBig.png"),
            ChestKind::RadMaggot => Cow::Borrowed("images/sprRadChestMaggot.png"),
            ChestKind::Idpd => Cow::Borrowed("images/sprIDPDChest.png"),
        },
    }
}

/// Resolve one projectile entity to its strip.
/// Priority: explicit GML visual (Bullet1/Bullet2) → melee slash flags →
/// player weapon table → enemy-kind table → team fallback. (nt-rewrite
/// projectiles carry no art path; bevy attached `Sprite` + candidates at
/// spawn, so this inverts the `player_projectile_candidates` /
/// `enemy_projectile_sprite` choice.)
fn projectile_art(
    proj: &Projectile,
    team: &Team,
    slash: Option<&SlashProjectile>,
    visual: Option<&crate::comps_a::ProjectileVisual>,
) -> &'static str {
    if let Some(v) = visual {
        return v.sprite;
    }
    if let Some(s) = slash {
        if s.blood {
            return "images/sprBloodSlash.png";
        }
        if s.lightning {
            return "images/sprLightningSlash.png";
        }
        if s.shank {
            return "images/sprShank.png";
        }
        return "images/sprSlash.png";
    }
    if let Some(src) = &proj.source {
        match src.hit_id {
            HitId::Weapon(w) => return player_projectile_path(w),
            HitId::Enemy(id) => {
                if let Some(kind) = EnemyKind::from_u16(id) {
                    return enemy_projectile_path(kind);
                }
            }
            _ => {}
        }
        if let Some(kind) = src.enemy_kind {
            return enemy_projectile_path(kind);
        }
    }
    match team {
        Team::Player => "images/sprBullet1.png",
        Team::Enemy => "images/sprEnemyBullet1.png",
    }
}

/// Bevy `sprite_from_projectile_path` frame law: 2-frame strips pin to
/// the second cell; longer strips animate at 12 fps (`projectile_anim`).
/// `life.elapsed_secs` drives the phase so identical bullets stay in
/// sync like bevy's per-entity `SpriteAnim` timers started at spawn.
/// Bevy `world.rs::wall_hash` verbatim: deterministic per-cell salt hash
/// driving wall art variant frames from the run seed.
fn wall_hash(seed: u64, wx: i32, wy: i32, salt: u64) -> u64 {
    let mut x = seed
        ^ ((wx as i64 as u64) << 32)
        ^ (wy as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ salt;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Frame count for a strip (0 when absent).
fn strip_frames(assets: &RenderAssets, path: &str) -> u32 {
    assets.uv(path, 0).map(|(_, def)| def.frames).unwrap_or(0)
}

/// Bevy `wall_body_frame` verbatim (Bot variant).
fn wall_body_frame(seed: u64, wx: i32, wy: i32, frames: u32) -> i32 {
    let raw = if wall_hash(seed, wx, wy, 0x11) % 150 == 0 {
        3
    } else {
        [0usize, 0, 0, 0, 0, 0, 0, 1, 2][(wall_hash(seed, wx, wy, 0x12) % 9) as usize]
            + [0usize, 4][(wall_hash(seed, wx, wy, 0x13) % 2) as usize]
    };
    (raw % frames.max(1) as usize) as i32
}

/// Bevy `wall_top_frame` verbatim.
fn wall_top_frame(seed: u64, wx: i32, wy: i32, frames: u32) -> i32 {
    let raw = if wall_hash(seed, wx, wy, 0x21) % 200 == 0 {
        3
    } else {
        [0usize, 0, 0, 0, 0, 0, 0, 1, 2][(wall_hash(seed, wx, wy, 0x22) % 9) as usize]
            + [0usize, 4, 8][(wall_hash(seed, wx, wy, 0x23) % 3) as usize]
    };
    (raw % frames.max(1) as usize) as i32
}

/// Bevy `wall_out_frame` verbatim.
fn wall_out_frame(seed: u64, wx: i32, wy: i32, frames: u32) -> i32 {
    let raw = [0usize, 0, 0, 0, 1, 2, 3, 4][(wall_hash(seed, wx, wy, 0x31) % 8) as usize]
        + [0usize, 4][(wall_hash(seed, wx, wy, 0x32) % 2) as usize];
    (raw % frames.max(1) as usize) as i32
}

/// sRGB channel -> linear light (exact transfer function). GPU tints
/// are authored as sRGB display colors (bevy `Color::srgb` parity): the
/// sRGB atlas decodes on sample and the sRGB target re-encodes on write,
/// so tints must be linear — raw sRGB tints render washed out.
pub fn srgb_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB tint -> linear working space (alpha is already linear).
pub fn tint_to_linear(t: [f32; 4]) -> [f32; 4] {
    [
        srgb_to_linear(t[0]),
        srgb_to_linear(t[1]),
        srgb_to_linear(t[2]),
        t[3],
    ]
}

/// GML healthbar ghost tint verbatim (`scrDrawPlayerHUD:42-51`):
/// hue − 5 (GameMaker 0-255 hue wheel), same saturation, value halved.
pub fn healthcol_dark(c: [f32; 4]) -> [f32; 4] {
    fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let d = max - min;
        let h = if d <= 1e-6 {
            0.0
        } else if max == r {
            60.0 * (((g - b) / d) % 6.0)
        } else if max == g {
            60.0 * ((b - r) / d + 2.0)
        } else {
            60.0 * ((r - g) / d + 4.0)
        };
        let h = if h < 0.0 { h + 360.0 } else { h };
        let s = if max <= 1e-6 { 0.0 } else { d / max };
        (h, s, max)
    }
    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        let (r, g, b) = match (h / 60.0).floor() as i32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        (r + m, g + m, b + m)
    }
    let (h, s, v) = rgb_to_hsv(c[0], c[1], c[2]);
    // GameMaker hue wheel is 0-255; −5 there = −5/255*360 here.
    let h2 = (h - 5.0 / 255.0 * 360.0).rem_euclid(360.0);
    let (r, g, b) = hsv_to_rgb(h2, s, v * 0.5);
    [r, g, b, c[3]]
}

/// Grid-quad overlap: butt-jointed opaque tiles crack along shared
/// edges under fractional cameras (float rounding + MSAA samples fall
/// in the ulp gap and show the background). Growing the +x/+y edges by
/// a world unit keeps every pixel covered; draw order is static so the
/// overlap is deterministic. Edge texels stretch ~3%: invisible.
pub const GRID_OVERLAP: f32 = 1.0;

/// Sprite z-ladder (GML `__global_object_depths` draw order verbatim:
/// higher GM depth draws first = further back, so the port's `z` runs
/// the other way — larger `z` draws on top; the engine stable-sorts by
/// `(blend, z, page)`, keeping push order on ties).
///
/// GM order (back → front): Floor(10) → Detail(8) → BackCont-shadows(5)
/// → Corpse(1) → Wall/shots(0) → Player/Ally(-2) → Portal(-3) →
/// SubTopCont wall-tops/bloom(-6) → TopCont fog/crosshair/revive(-15) →
/// SpiralCont figures(-101) → Draw-GUI HUD text/menus → Menu(-1001).
/// The world batch keeps its internal push order at 0 (bevy parity —
/// untouched); every chrome layer above it stamps one rung so atlas
/// page can never lottery a HUD bar under a floor tile again.
pub const Z_SHADOW: f32 = -10.0;
pub const Z_WORLD: f32 = 0.0;
pub const Z_FX: f32 = 1.0;
pub const Z_BLOOM: f32 = 2.0;
pub const Z_FOG: f32 = 3.0;
pub const Z_CROSSHAIR: f32 = 4.0;
pub const Z_FAINTED: f32 = 5.0;
pub const Z_PORTAL_INDICATOR: f32 = 6.0;
pub const Z_SPIRAL_FIGURES: f32 = 7.0;
pub const Z_HUD: f32 = 10.0;
pub const Z_SPLASH: f32 = 15.0;
pub const Z_MENU: f32 = 20.0;
pub const Z_SIDEART: f32 = 30.0;

/// Stamp a layer rung over a finished push batch (keeps the producer's
/// internal push order: the engine sort is stable on `(blend, z, page)`
/// ties). `pub(crate)` so the view composer assigns rungs per layer
/// next to the push order (single place both are visible).
pub(crate) fn stamp_z(out: &mut [SpriteInstance], z: f32) {
    for s in out.iter_mut() {
        s.z = z;
    }
}

/// Place a strip quad by its art top-left (GM draw origin): the catalog
/// anchor lands `center` so origin-(0,0) floor/wall art sits on the grid
/// exactly like bevy's `sprite_at_gm_origin` (passing a cell center would
/// shift it by half a cell). `grow` extends the +x/+y edges (see
/// [`GRID_OVERLAP`]); UVs still span the cell.
fn place_top_left(
    assets: &RenderAssets,
    path: &str,
    frame: i32,
    top_left: Vec2,
    tint: [f32; 4],
    grow: f32,
) -> Option<SpriteInstance> {
    let (uv, def) = assets.uv(path, frame)?;
    let anchor = def.anchor();
    let size = Vec2::new(def.w as f32 + grow, def.h as f32 + grow);
    Some(SpriteInstance {
        center: top_left + Vec2::new(anchor[0] * size.x, anchor[1] * size.y),
        rotation: 0.0,
        size,
        anchor: Vec2::new(anchor[0], anchor[1]),
        flip_x: false,
        flip_y: false,
        uv_min: Vec2::new(uv.min[0], uv.min[1]),
        uv_max: Vec2::new(uv.max[0], uv.max[1]),
        color: tint_to_linear(tint),
        page: uv.page,
        z: 0.0,
        blend: SpriteBlend::Alpha,
    })
}

fn projectile_frame(assets: &RenderAssets, path: &str, life: &crate::time::GTimer) -> i32 {
    match assets.uv(path, 0) {
        Some((_, def)) if def.frames <= 1 => 0,
        Some((_, def)) => {
            // GML image_speed ~0.4 → 12 fps at 30Hz.
            ((life.elapsed_secs() * 12.0).floor() as u32 % def.frames.max(1)) as i32
        }
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Camera + background.
// ---------------------------------------------------------------------------

/// GML `objects/BackCont/Step_0.gml` camera law verbatim (replaces the
/// earlier bevy `CameraFollow` approximation, whose two-stage smoothing
/// never matched the original feel).
///
/// Per game step, with the local player at `player`:
/// - POI pull: nearest Portal (else BecomeNothing/NothingDeath/
///   Nothing2Death/SitDown) contributes `dist/6` along its heading,
///   capped at 72 px for Portals and WeaponChests (the GML chain's
///   second `BecomeNothing` check is dead code — the first branch
///   already took it — so it contributes nothing here either).
/// - Aim lean: `dis_fire / viewdist` along `dir_fire` (`viewdist` 4,
///   8 for melee, 3 for bolts).
/// - Shake: `orandom(shake * opt_shake)` added to the target.
/// - Snap (level start): target jump (`m = 1`), knock decay zeroed.
/// - `view = round(lerp(view, target, m))` with `m = 0.4` live.
/// - Knock decay `round(viewx2 - viewx2 * 0.4)`; shake decay
///   `*= power(0.8, timescale)` above 10 else `-= timescale` to 0;
///   `opt_shake <= 0` zeroes shake + knock.
#[derive(Clone, Copy, Debug, Default)]
pub struct GmlCamera {
    /// Top-left view corner, world px (`view_xview`/`view_yview`).
    pub x: f32,
    pub y: f32,
    /// Knock decay state (`viewx2`/`viewy2`).
    pub viewx2: f32,
    pub viewy2: f32,
    /// Scalar shake (`BackCont.shake`).
    pub shake: f32,
    /// Level-start snap (`force_snap_camera_position`).
    pub snap: bool,
}

/// GML POI divisor (`_dis = point_distance / 6`).
pub const CAM_POI_DIV: f32 = 6.0;
/// GML POI cap for Portals + WeaponChests (`min(_dis, 72)`).
pub const CAM_POI_CAP: f32 = 72.0;
/// GML aim lean divisor (`_viewdist = 4`).
pub const CAM_VIEWDIST: f32 = 4.0;
/// GML aim lean divisor for melee (`scr_weapon_is_melee(wep)`).
pub const CAM_VIEWDIST_MELEE: f32 = 8.0;
/// GML aim lean divisor for bolts (`scr_weapon_get_type == Ammo.Bolts`).
pub const CAM_VIEWDIST_BOLTS: f32 = 3.0;
/// Aim-lean cap in world px (bevy `player_aim` `MAX_LOOK` verbatim:
/// the lookahead never exceeds 48 px from the player, mouse or stick).
pub const CAM_MAX_LOOK: f32 = 48.0;
/// GML follow rate (`lerp(..., 0.4)`).
pub const CAM_LERP: f32 = 0.4;
/// GML knock decay rate (`viewx2 - viewx2 * 0.4`).
pub const CAM_KNOCK_DECAY: f32 = 0.4;
/// Unused zoom-ease weight (GML has no zoom; the view is always 240
/// world px tall). Kept for save-compat/harness use.
pub const CAM_ZOOM_SPEED: f32 = 0.08;

/// Frame-rate-independent lerp factor:
/// Godot-style per-physics-frame `lerp(a, b, weight)` at 60 tps.
pub fn framed_lerp(weight: f32, dt: f32) -> f32 {
    1.0 - (1.0 - weight).powf((dt * 60.0).max(0.0))
}

/// Rate-adjusted GML step factor: GML steps at 30 tps, so a per-step
/// `k` becomes `1 - (1-k)^(dt*30)` per rendered frame (exactly `k` at
/// 30 fps).
pub fn gml_rate(k: f32, dt: f32) -> f32 {
    1.0 - (1.0 - k).powf((dt * 30.0).max(0.0))
}

/// One POI candidate for [`gml_camera_step`]: world pos + whether the
/// 72 px cap applies (Portals and WeaponChests only).
#[derive(Clone, Copy, Debug)]
pub struct CamPoi {
    pub pos: Vec2,
    pub capped: bool,
}

/// One [`gml_camera_step`] input snapshot. `aim_dir`/`aim_dis` are the
/// `dir_fire`/`dis_fire` pair (cursor offset from the player, world
/// px); `jx`/`jy` carry the frame's `orandom` shake sample in [-1, 1].
#[derive(Clone, Copy, Debug)]
pub struct CamStepInput {
    pub player: Vec2,
    pub aim_dir: Vec2,
    pub aim_dis: f32,
    pub viewdist: f32,
    pub poi: Option<CamPoi>,
    pub shake_scale: f32,
    pub timescale: f32,
    pub jx: f32,
    pub jy: f32,
}

/// GML aim-lean divisor for a weapon: 8 melee, 3 bolts, else 4.
pub fn cam_viewdist_for(wep: crate::data::WeaponId) -> f32 {
    let meta = crate::weapon_runtime::weapon_meta(wep);
    if meta.wep_mele {
        return CAM_VIEWDIST_MELEE;
    }
    if meta.wep_type == crate::weapons_data::AmmoType::Bolts {
        return CAM_VIEWDIST_BOLTS;
    }
    CAM_VIEWDIST
}

/// One GML `BackCont` camera step. `vw`/`vh` are the view size in world
/// px. (The headless `bleed → ChickenHead` branch has no port
/// counterpart — single-player has no ChickenHead entity — so it is
/// skipped with this note.)
pub fn gml_camera_step(cam: &mut GmlCamera, vw: f32, vh: f32, s: &CamStepInput, dt: f32) {
    fn lerp_f(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    if cam.snap {
        cam.viewx2 = 0.0;
        cam.viewy2 = 0.0;
    } else {
        if let Some(poi) = &s.poi {
            let delta = poi.pos - s.player;
            let d = delta.length();
            if d > 1e-6 {
                let mut dis = d / CAM_POI_DIV;
                if poi.capped {
                    dis = dis.min(CAM_POI_CAP);
                }
                sx += delta.x / d * dis;
                sy += delta.y / d * dis;
            }
        }
        let dis2 = s.aim_dis / s.viewdist.max(1e-6);
        sx += s.aim_dir.x * dis2;
        sy += s.aim_dir.y * dis2;
        if s.shake_scale > 0.0 {
            sx += s.jx * cam.shake * s.shake_scale;
            sy += s.jy * cam.shake * s.shake_scale;
        }
    }
    let m = if cam.snap {
        1.0
    } else {
        gml_rate(CAM_LERP, dt)
    };
    cam.x = lerp_f(cam.x, s.player.x - vw * 0.5 + cam.viewx2 + sx, m).round();
    cam.y = lerp_f(cam.y, s.player.y - vh * 0.5 + cam.viewy2 + sy, m).round();
    cam.snap = false;
    let kd = gml_rate(CAM_KNOCK_DECAY, dt);
    cam.viewx2 = (cam.viewx2 - cam.viewx2 * kd).round();
    cam.viewy2 = (cam.viewy2 - cam.viewy2 * kd).round();
    if s.shake_scale <= 0.0 {
        cam.shake = 0.0;
        cam.viewx2 = 0.0;
        cam.viewy2 = 0.0;
    }
    if cam.shake > 10.0 {
        cam.shake *= 0.8f32.powf(s.timescale.max(0.0));
    } else if cam.shake > 0.0 {
        cam.shake = (cam.shake - s.timescale).max(0.0);
    }
}

/// GPU camera for a look point: `units_per_pixel` carries the GML
/// view scale (see [`gml_view_scale`] — GML has no zoom). Pair with the
/// dp viewport extent ([`camera_fit_extent`](crate::camera_fit_extent))
/// so the engine fit shows `viewport * scale` world units.
pub fn world_camera(center: Vec2, scale: f32) -> Camera2d {
    Camera2d {
        center,
        offset: Vec2::ZERO,
        units_per_pixel: scale,
        zoom: 1.0,
        // No camera roll in the port (GML has none; trauma shake stays
        // translational through `GmlCamera`).
        roll: 0.0,
    }
}

/// GML `scrSetViewSize` view law verbatim (`scripts/macros_general` +
/// `UberCont/Create_0`: base `game_screen_width/height` 320x240, widened
/// to `view_width_max = 240 * aspect` when `opt_resolution`, which
/// defaults on; odd widths bumped +1). Returns the live GUI/view size
/// in world px for a dp viewport (portrait floors the width at 320).
pub fn gml_view_size(viewport_dp: [f32; 2]) -> [f32; 2] {
    let (w, h) = (viewport_dp[0].max(1.0), viewport_dp[1].max(1.0));
    let mut vw = (240.0 * w / h).max(320.0).floor();
    // GML `scrSetViewSize` verbatim: odd widths bump +1 (the camera and
    // GUI both run on the even width).
    if vw % 2.0 != 0.0 {
        vw += 1.0;
    }
    [vw, 240.0]
}

/// `units_per_pixel` for [`world_camera`] so the framed view matches
/// [`gml_view_size`] exactly: `max(240/h, 320/w)` over the dp viewport
/// (1280x720 → 1/3, i.e. [`crate::comps_a::NT_CAM_SCALE`]).
pub fn gml_view_scale(viewport_dp: [f32; 2]) -> f32 {
    let (w, h) = (viewport_dp[0].max(1.0), viewport_dp[1].max(1.0));
    (240.0 / h).max(320.0 / w).max(1e-6)
}

/// Full-viewport fill under the sprite batch, per area. Fallback where
/// no vortex layer is mounted (pre-run menus, missing art); live frames
/// drive `crate::vortex_pass::VortexPass` on top of this.
pub fn background_color(area: AreaId) -> [f32; 4] {
    // GML `scrAreaGetBackroundColor` verbatim (GameMaker `#rrggbb` -> sRGB
    // 0..1; custom area colors are shell-side and fall through here).
    fn hex(hex: u32) -> [f32; 4] {
        [
            ((hex >> 16) & 0xFF) as f32 / 255.0,
            ((hex >> 8) & 0xFF) as f32 / 255.0,
            (hex & 0xFF) as f32 / 255.0,
            1.0,
        ]
    }
    match area {
        AreaId::Campfire => hex(0x6a7aaf),
        AreaId::Desert => hex(0xaf8f6a),
        AreaId::Sewers => hex(0x4c5946),
        AreaId::Scrapyards => hex(0x8a969e),
        AreaId::CrystalCaves => hex(0x8152bc),
        AreaId::FrozenCity => hex(0xb4bdc5),
        AreaId::Labs => hex(0x091c20),
        AreaId::Palace => hex(0x611d24),
        AreaId::Vault | AreaId::CrownVault => hex(0x433523),
        AreaId::Oasis => hex(0x51d1c8),
        AreaId::PizzaSewers => hex(0xa04b63),
        AreaId::CursedCaves => hex(0xff9c23),
        AreaId::Jungle => hex(0x2a900c),
        AreaId::HQ => hex(0xf5fafb),
        // GML mansion/crib (#eef0f2) have no port area; City reuses the
        // city fill (GML has no city-secret fill; mansion white is closest).
        AreaId::City => hex(0xeef0f2),
        AreaId::Loop => hex(0x6a7aaf),
    }
}

// ---------------------------------------------------------------------------
// View-space atmosphere: area fog + sideart chrome.
// ---------------------------------------------------------------------------

/// GML `#macro FOG_ALPHA 0.1` (`objects/TopCont/Create_0`).
pub const FOG_ALPHA: f32 = 0.1;
/// GML fog tile size (`draw 3x3 of 480x360`).
pub const FOG_TILE_W: f32 = 480.0;
/// GML fog tile size (vertical).
pub const FOG_TILE_H: f32 = 360.0;
/// GML sideart tile size (`sprSideArt` 64 px, `i * -64` tiling).
pub const SIDEART_TILE: f32 = 64.0;

/// World-space view rect `[x, y, w, h]` under the live camera fit
/// (top-left + extent in world units; degenerate fits yield a zero
/// rect so atmosphere draws park at the look point).
pub fn view_rect_world(canvas_dp: [f32; 2], world_size: [f32; 2], cam: &Camera2d) -> [f32; 4] {
    let center = cam.effective_center();
    let fit = effective_fit(canvas_dp, world_size, cam);
    if !fit.0.is_finite() || fit.0 <= 1e-6 {
        return [center[0], center[1], 0.0, 0.0];
    }
    let tl = dp_to_world([0.0, 0.0], world_size, center, fit);
    [tl[0], tl[1], canvas_dp[0] / fit.0, canvas_dp[1] / fit.0]
}

/// Fog strip for an area (GML `TopCont/Draw_0` verbatim: pizza sewers
/// first, then sewers or the halloween event flag; everywhere else
/// draws nothing). The port has no halloween event flag, so callers
/// pass `false` (documented at the call site).
pub fn fog_sprite_for_area(area: AreaId, halloween: bool) -> Option<&'static str> {
    if area == AreaId::PizzaSewers {
        Some("images/sprFog102.png")
    } else if area == AreaId::Sewers || halloween {
        Some("images/sprFog2.png")
    } else {
        None
    }
}

/// Fog tile top-lefts (GML `TopCont/Draw_0` verbatim): 3x3 of 480x360
/// at `floor(view/480)*480 + 480 - fogscroll`,
/// `floor(view/360)*360`. `view` is the room-space view origin.
pub fn fog_tiles(view_x: f32, view_y: f32, scroll: f32) -> Vec<[f32; 2]> {
    let fogx = (view_x / FOG_TILE_W).floor() * FOG_TILE_W + FOG_TILE_W - scroll;
    let fogy = (view_y / FOG_TILE_H).floor() * FOG_TILE_H;
    let mut out = Vec::with_capacity(9);
    for ix in -1..=1 {
        for iy in -1..=1 {
            out.push([fogx + ix as f32 * FOG_TILE_W, fogy + iy as f32 * FOG_TILE_H]);
        }
    }
    out
}

/// Area-fog sprites over the live view (GML `TopCont/Draw_0`): the
/// area strip tiled 3x3 at [`FOG_ALPHA`], scrolling with
/// [`crate::environment::FogState`]. Missing art degrades to no fog
/// (same graceful fallback as every other optional strip).
pub fn fog_sprites(
    world: &mut World,
    assets: &RenderAssets,
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam: &Camera2d,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    let area = world
        .get_resource::<Run>()
        .map(|r| r.area)
        .unwrap_or(AreaId::Desert);
    // No halloween event flag in the port (see `fog_sprite_for_area`).
    let Some(path) = fog_sprite_for_area(area, false) else {
        return out;
    };
    let scroll = world
        .get_resource::<crate::environment::FogState>()
        .map(|f| f.scroll)
        .unwrap_or(0.0);
    let view = view_rect_world(canvas_dp, world_size, cam);
    for [x, y] in fog_tiles(view[0], view[1], scroll) {
        if let Some(s) = place_top_left(
            assets,
            path,
            0,
            Vec2::new(x, y),
            [1.0, 1.0, 1.0, FOG_ALPHA],
            0.0,
        ) {
            out.push(s);
        }
    }
    out
}

/// Sideart tile positions in view px (GML `scrDrawSidearts` verbatim):
/// `n = view_width_max / 64` columns (`for i = 1; i <= n` → `floor`
/// count), horizontal strips at `i * -64` and
/// `view_width - 64 + i * 64` over `nh = ceil(view_height / 64) + 1`
/// rows, then vertical `repeat 10` runs at `i * 64` above/below for
/// `i in 0..n`.
pub fn sideart_tiles(view_w: f32, view_h: f32) -> Vec<[f32; 2]> {
    let n = (view_w / SIDEART_TILE).floor().max(1.0) as usize;
    let nh = (view_h / SIDEART_TILE).ceil() as usize + 1;
    let mut out = Vec::new();
    for i in 1..=n {
        let mut yy = 0.0;
        for _ in 0..nh {
            out.push([i as f32 * -SIDEART_TILE, yy]);
            out.push([view_w - SIDEART_TILE + i as f32 * SIDEART_TILE, yy]);
            yy += SIDEART_TILE;
        }
    }
    for i in 0..n {
        let mut yy = 0.0;
        for _ in 0..10 {
            out.push([i as f32 * SIDEART_TILE, -SIDEART_TILE - yy]);
            out.push([i as f32 * SIDEART_TILE, view_h + yy]);
            yy += SIDEART_TILE;
        }
    }
    out
}

/// Sideart chrome around the view (GML `scrDrawSidearts`, drawn from
/// `UberCont/Draw_74` over everything): `sprSideArt` frame
/// `opt_sideart` at every [`sideart_tiles`] position, mapped through
/// the view like the menu art (the port canvas IS the view).
pub fn sideart_sprites(
    world: &mut World,
    assets: &RenderAssets,
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam: &Camera2d,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    let opt = world
        .get_resource::<crate::savedata_part::SaveData>()
        .map(|s| s.settings.sideart as i32)
        .unwrap_or(0);
    let frames = strip_frames(assets, "images/sprSideArt.png").max(1) as i32;
    let frame = opt.clamp(0, frames - 1);
    // Tile law is in GML view px (426x240 at 16:9), not dp: tiling
    // `canvas_dp` (1280x720) spawns ~3x too many tiles at third-size
    // offsets. Tile the live GUI view and map 1:1 through the view rect.
    let vw = gml_view_size(canvas_dp)[0];
    let view = view_rect_world(canvas_dp, world_size, cam);
    let gm = hud_gui_map(view);
    for [x, y] in sideart_tiles(vw, 240.0) {
        if let Some(s) = assets.sprite_for(
            "images/sprSideArt.png",
            frame,
            hud_gui_to_world(gm, view, x, y),
            false,
            0.0,
            [1.0; 4],
        ) {
            out.push(s);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// World → instances.
// ---------------------------------------------------------------------------

fn flash_tint(flash: Option<&HitFlash>) -> [f32; 4] {
    flash.map(|f| f.color).unwrap_or([1.0, 1.0, 1.0, 1.0])
}

/// Snapshot the sim world into GPU sprites (draw order = push order,
/// bevy z-ladder: floor, walls, decals, props, corpses/portals,
/// hazards, pickups, opened chests, enemies, player, projectiles,
/// hit-FX, held guns, melee swings).
pub fn world_instances(world: &mut World, assets: &RenderAssets) -> Vec<SpriteInstance> {
    let mut out = Vec::new();

    // Floor: bevy law (`spawn_level`): lit area strip over mask cells,
    // darkened outside ring over the padded bounding rect (this is what
    // makes the map fill its rect instead of floating as bare cells).
    // Quads place by art top-left (`place_top_left`, bevy
    // `sprite_at_gm_origin` parity): origin-(0,0) art would sit half a
    // cell off if given cell centers.
    if let (Some(run), Some(mask)) = (
        world.get_resource::<Run>(),
        world.get_resource::<FloorMask>(),
    ) {
        let floor = run.floor;
        let area = run.area;
        let seed = run.gen_seed;
        let cells: HashSet<(i32, i32)> = mask.cells.iter().copied().collect();
        let has = |p: &str| assets.catalog.def(p).is_some();
        let (floor_png, wall_bot_png, wall_top_png, wall_out_png, wall_trans_png) =
            area_sprites_full_for_run(floor, area, has);
        let (mut minx, mut miny, mut maxx, mut maxy) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for &(cx, cy) in &cells {
            minx = minx.min(cx);
            miny = miny.min(cy);
            maxx = maxx.max(cx);
            maxy = maxy.max(cy);
        }
        // GML draws the room background colour first
        // (`background_set_colour`), then ONLY the live floor cells —
        // there is no padded outside ring of floor tiles. The old
        // ±6-cell darkened ring covered the whole viewport with opaque
        // quads and buried the transparent vortex layer underneath on
        // the campfire title (the "no vortex on the title screen"
        // bug). Lit strip over mask cells only; the room colour shows
        // everywhere else.
        if minx <= maxx {
            for cy in miny..=maxy {
                for cx in minx..=maxx {
                    if !cells.contains(&(cx, cy)) {
                        continue;
                    }
                    let top_left = Vec2::new(cx as f32 * TILE, cy as f32 * TILE);
                    if let Some(s) =
                        place_top_left(assets, floor_png, 0, top_left, [1.0; 4], GRID_OVERLAP)
                    {
                        out.push(s);
                    }
                }
            }
        }
        // Walls: bevy `spawn_level` composite per 16px cell — Out skirt
        // always, Bot iff the screen-south tile is floor, Top always
        // straddling the cell's top edge (+8y in bevy numbers = 8px
        // screen-up = top-left (wx*16, wy*16-8) here), then the Trans
        // skirting pass. Variant frames are the seeded
        // `wall_{body,top,out}_frame` laws. Push order = draw order.
        let mut walls: Vec<WallCell> = world
            .query_filtered::<&WallCell, With<WallTile>>()
            .iter(world)
            .copied()
            .collect();
        walls.sort_by_key(|c| (c.1, c.0));
        let wall_set: HashSet<(i32, i32)> = walls.iter().map(|c| (c.0, c.1)).collect();
        let out_frames = strip_frames(assets, wall_out_png);
        let bot_frames = strip_frames(assets, wall_bot_png);
        let top_frames = strip_frames(assets, wall_top_png);
        for cell in &walls {
            let (wx, wy) = (cell.0, cell.1);
            // Screen-south tile (y-down: +y), mirroring bevy's
            // `(c.x, c.y - WALL_PX)` probe in y-up numbers.
            let south_tile = (wx.div_euclid(2), wy.div_euclid(2) + 1);
            let floor_south = cells.contains(&south_tile);
            if has(wall_out_png) {
                let frame = wall_out_frame(seed, wx, wy, out_frames);
                if let Some(s) = place_top_left(
                    assets,
                    wall_out_png,
                    frame,
                    Vec2::new(wx as f32 * 16.0 - 4.0, wy as f32 * 16.0 - 12.0),
                    [1.0; 4],
                    0.0,
                ) {
                    out.push(s);
                }
            }
            if floor_south && has(wall_bot_png) {
                let frame = wall_body_frame(seed, wx, wy, bot_frames);
                if let Some(s) = place_top_left(
                    assets,
                    wall_bot_png,
                    frame,
                    Vec2::new(wx as f32 * 16.0, wy as f32 * 16.0),
                    [1.0; 4],
                    GRID_OVERLAP,
                ) {
                    out.push(s);
                }
            }
            if has(wall_top_png) {
                let frame = wall_top_frame(seed, wx, wy, top_frames);
                if let Some(s) = place_top_left(
                    assets,
                    wall_top_png,
                    frame,
                    Vec2::new(wx as f32 * 16.0, wy as f32 * 16.0 - 8.0),
                    [1.0; 4],
                    GRID_OVERLAP,
                ) {
                    out.push(s);
                }
            }
        }
        // Trans skirting: floor-cell bottom edge 2x2 sub-tiles that are
        // neither wall nor floor (bevy loop verbatim, y-down: south = +y).
        if has(wall_trans_png) {
            let trans_frames = strip_frames(assets, wall_trans_png);
            for &(cx, cy) in &cells {
                let ftl = Vec2::new(cx as f32 * TILE, (cy as f32 + 1.0) * TILE);
                for (ox, oy) in [(0.0, 0.0), (16.0, 0.0), (0.0, 16.0), (16.0, 16.0)] {
                    let p = ftl + Vec2::new(ox, oy);
                    let wx = (p.x / 16.0).floor() as i32;
                    let wy = (p.y / 16.0).floor() as i32;
                    if wall_set.contains(&(wx, wy)) {
                        continue;
                    }
                    let owner = (wx.div_euclid(2), wy.div_euclid(2));
                    if cells.contains(&owner) {
                        continue;
                    }
                    let frame = (wall_hash(seed, wx, wy, 0x41) as usize
                        % trans_frames.max(1) as usize) as i32;
                    if let Some(s) = place_top_left(
                        assets,
                        wall_trans_png,
                        frame,
                        Vec2::new(wx as f32 * 16.0, wy as f32 * 16.0),
                        [1.0; 4],
                        GRID_OVERLAP,
                    ) {
                        out.push(s);
                    }
                }
            }
        }
        // Wall drop shadows (GML `scrShadows` wall half: the Out strip
        // flipped under each wall with open floor to its screen-south,
        // drawn into the `shad` surface in `shadow_color` at 0.4 alpha
        // and composited under the actors by `BackCont/Draw_0`). The
        // port has no TopSmall adjacency, so the south-neighbor wall
        // test stands in (interior skirts hide under the next wall's
        // own art either way); tint is flat black 0.4 (the area shadow
        // colors are not ported yet).
        if has(wall_out_png) {
            if let Some(size) = assets.native_size(wall_out_png) {
                let (w, h) = (size.x, size.y);
                for cell in &walls {
                    let (wx, wy) = (cell.0, cell.1);
                    if wall_set.contains(&(wx, wy + 1)) {
                        continue;
                    }
                    let frame = wall_out_frame(seed, wx, wy, out_frames);
                    // Flipped skirt spans [y+18-h, y+18] (GML draws the
                    // Out strip at (x, y+18) with yscale -1).
                    if let Some((_, def)) = assets.uv(wall_out_png, frame) {
                        let a = def.anchor();
                        let center = Vec2::new(
                            wx as f32 * 16.0 - 4.0 + a[0] * w,
                            wy as f32 * 16.0 + 18.0 - h + a[1] * h,
                        );
                        if let Some(s) = assets.sprite_for_full(
                            wall_out_png,
                            frame,
                            center,
                            false,
                            true,
                            0.0,
                            [0.0, 0.0, 0.0, 0.4],
                        ) {
                            out.push(s);
                        }
                    }
                }
            }
        }
    }

    // Throne carpet (bevy z -48: 72x480 srgba(0.75,0.12,0.14,0.85)
    // rect under the walls).
    {
        let mut q = world.query::<(&Pos, &ThroneCarpet)>();
        for (pos, carpet) in q.iter(world) {
            let size = carpet.half_extents * 2.0;
            out.push(white_quad(pos.0, 0.0, size, [0.75, 0.12, 0.14, 0.85]));
        }
    }

    // Pulse decals (bevy z -41: cobweb / ice / fire-trap ground
    // visuals under the actors). Textured at the recorded size, or a
    // solid tint rect when no candidate art is cataloged (bevy
    // `sprite_from_candidates` fallback); alpha always follows bevy
    // `animate_environment` via the paired `SurfacePulse`.
    {
        let now = pulse_now(world);
        let mut q = world.query::<(&Pos, &PulseSprite, &SurfacePulse)>();
        for (pos, vis, pulse) in q.iter(world) {
            let mut tint = vis.tint;
            tint[3] = pulse.alpha_at(now);
            match vis.path {
                Some(path) => {
                    if let Some(s) =
                        assets.sprite_sized(path, 0, pos.0, Vec2::splat(vis.size), vis.flip_x, tint)
                    {
                        out.push(s);
                    }
                }
                None => out.push(white_quad(pos.0, 0.0, Vec2::splat(vis.size), tint)),
            }
        }
    }

    // Props (recorded art paths; hurt flash tints; mine/torch sprites
    // throb via their `SurfacePulse`, bevy `animate_environment` law;
    // ground decals draw gray 0.5).
    {
        let now = pulse_now(world);
        let mut q = world.query::<(
            &Pos,
            &PropSprites,
            Option<&SpriteAnim>,
            Option<&HitFlash>,
            Option<&SurfacePulse>,
            Option<&GroundDecalTint>,
        )>();
        for (pos, sprites, anim, flash, pulse, decal) in q.iter(world) {
            // Skip non-prop carriers of PropSprites (corpses): they ride
            // the generic anim fallback below without the `Prop` gate.
            let (path, frame) = match anim {
                Some(a) => (a.path.as_str(), a.frame as i32),
                None => (sprites.idle, 0),
            };
            let mut tint = flash_tint(flash);
            if decal.is_some() {
                tint = [0.5, 0.5, 0.5, 0.5];
            }
            if let Some(pulse) = pulse {
                tint[3] *= pulse.alpha_at(now);
            }
            if let Some(s) = assets.sprite_for(path, frame, pos.0, sprites.flip_x, 0.0, tint) {
                out.push(s);
            }
        }
    }

    // Campfire YV couch (GML `YungVenuzCouch`: idle
    // `sprYVBossGamingIdle`, airhorn one-shot
    // `sprYVBossGamingAirhorn`; frame is the sim `image_index`).
    // Missing art skips the couch, like other optional strips.
    {
        let mut q = world.query::<(&Pos, &YvCouch)>();
        for (pos, couch) in q.iter(world) {
            let path = couch.sprite_path();
            let frames = strip_frames(assets, path).max(1);
            let frame = (couch.frame.floor() as i32).clamp(0, frames as i32 - 1);
            if let Some(s) = assets.sprite_for(path, frame, pos.0, false, 0.0, [1.0; 4]) {
                out.push(s);
            }
        }
    }

    // Live hazard rects (bevy z 7: spill hazards are a solid kind-color
    // rect, `radius * 2`, with the hazard pulse alpha; textured
    // fire-trap visuals already drew in the decal block above).
    {
        let now = pulse_now(world);
        let mut q = world.query::<(
            &Pos,
            &EnvironmentHazard,
            &SurfacePulse,
            Option<&PulseSprite>,
        )>();
        for (pos, hazard, pulse, vis) in q.iter(world) {
            if vis.is_some() {
                continue;
            }
            let rgb = hazard.spec.kind.color();
            let d = hazard.spec.radius.max(4.0) * 2.0;
            out.push(white_quad(
                pos.0,
                0.0,
                Vec2::new(d, d),
                [rgb[0], rgb[1], rgb[2], pulse.alpha_at(now)],
            ));
        }
    }
    // Corpse pass (bevy z -6, under hazards/pickups): enemy husks
    // (`Corpse`), prop corpses (`PropSprites` without `Prop`), and
    // portal strips. Prop corpses keep their recorded `PropSprites`
    // facing and the player husk its `Corpse.flip_x` (bevy parity).
    // Hit-effect oneshots fade in their last 0.12 s (bevy
    // `tick_hit_effects` law).
    {
        let mut q = world.query::<(
            Entity,
            &Pos,
            &SpriteAnim,
            Option<&HitFlash>,
            Option<&FxAngle>,
            Option<&PickupLifetime>,
            Option<&Prop>,
            Option<&PropSprites>,
            Option<&Corpse>,
            Option<&Portal>,
            Option<&Enemy>,
            Option<&Player>,
            Option<&Projectile>,
            Option<&Pickup>,
            Option<&WeaponVisual>,
        )>();
        for (
            e,
            pos,
            anim,
            flash,
            angle,
            lifetime,
            prop,
            sprites,
            corpse,
            portal,
            enemy,
            player,
            proj,
            pickup,
            gun,
        ) in q.iter(world)
        {
            if prop.is_some()
                || enemy.is_some()
                || player.is_some()
                || proj.is_some()
                || pickup.is_some()
                || gun.is_some()
            {
                continue;
            }
            let is_corpse =
                corpse.is_some() || (sprites.is_some() && prop.is_none()) || portal.is_some();
            if !is_corpse {
                continue;
            }
            let _ = e;
            let rotation = angle.map(|a| a.0).unwrap_or(0.0);
            let mut tint = flash_tint(flash);
            if let Some(lt) = lifetime {
                tint[3] *= (lt.timer.remaining_secs() / 0.12).clamp(0.0, 1.0);
            }
            // Corpses keep the facing recorded at death.
            let flip = sprites.map(|s| s.flip_x).unwrap_or(false)
                || corpse.map(|c| c.flip_x).unwrap_or(false);
            if let Some(s) =
                assets.sprite_for(&anim.path, anim.frame as i32, pos.0, flip, rotation, tint)
            {
                out.push(s);
            }
        }
    }

    // Portal shock/clear/strike draw with the hit-FX pass below
    // (bevy z 14 band); see the FX section after projectiles.

    // Pickups + chests: native strip-frame cells, exactly like bevy
    // `spawn_pickup`/`spawn_chest` (which ignore `pickup_sprite`'s
    // vestigial sizes and draw via `sprite_exact`). `sprite_for` already
    // sizes to the catalog cell — no override. Animated kinds (rads at
    // 12fps from a random start, chest idle shimmer) ride their live
    // `SpriteAnim` frame; static kinds sit on frame 0.
    {
        let mut q = world.query::<(&Pos, &Pickup, Option<&SpriteAnim>)>();
        for (pos, pickup, anim) in q.iter(world) {
            let path = pickup_art(&pickup.kind);
            let frame = anim.map(|a| a.frame as i32).unwrap_or(0);
            if let Some(s) = assets.sprite_for(&path, frame, pos.0, false, 0.0, [1.0; 4]) {
                out.push(s);
            }
        }
    }

    // Opened chests: bevy `open_chest` swaps to the kind-specific open
    // art frozen on the last frame (weapon/ammo frame 0, rad corpse
    // frame 2 clamped to the strip, exactly like bevy's
    // `last_frame.min(def.frames - 1)`).
    {
        let mut q = world.query::<(&Pos, &OpenedChest)>();
        for (pos, opened) in q.iter(world) {
            let (path, last) = match opened.0 {
                ChestKind::Weapon => ("images/sprWeaponChestOpen.png", 0u32),
                ChestKind::Ammo => ("images/sprAmmoChestOpen.png", 0u32),
                ChestKind::Rad => ("images/sprRadChestCorpse.png", 2u32),
                ChestKind::Health => ("images/sprHealthChestOpen.png", 0u32),
                ChestKind::CursedBig => ("images/sprCursedChestBigOpen.png", 0u32),
                ChestKind::Rogue => ("images/sprRogueAmmoChestOpen.png", 0u32),
                ChestKind::Proto => ("images/sprProtoChestOpen.png", 0u32),
                ChestKind::BigWeapon => ("images/sprWeaponChestBigOpen.png", 0u32),
                ChestKind::RadBig => ("images/sprRadChestBigDead.png", 0u32),
                ChestKind::RadMaggot => ("images/sprRadChestMaggotDead.png", 0u32),
                ChestKind::Idpd => ("images/sprIDPDChestOpen.png", 0u32),
            };
            let frames = strip_frames(assets, path);
            let frame = last.min(frames.saturating_sub(1)) as i32;
            if let Some(s) = assets.sprite_for(path, frame, pos.0, false, 0.0, [1.0; 4]) {
                out.push(s);
            }
        }
    }

    // Enemies: live SpriteAnim wins (idle/walk/hurt/fire already switched
    // sim-side); headless spawns without one fall back to the def idle.
    // (nt-rewrite never attaches `EnemySprites`; bevy's walk/hurt strips
    // resolve here from the anim path instead.) Gun carriers draw their
    // gun behind the body when aiming down-ish (gunangle ≤ 180°) and in
    // front above it (GML per-kind `Draw_0` law; gun at the body center —
    // the wkick offset is rest-zero).
    {
        // Throne flames (`Nothing/Draw_0` verbatim): four flame quads,
        // Big while the beam charges.
        let blink_t = world
            .get_resource::<repame_sim::SimTime>()
            .map(|t| t.elapsed_secs)
            .unwrap_or(0.0);
        let mut fq = world.query::<(&Pos, &Enemy, Option<&BossBrain>)>();
        for (pos, enemy, brain) in fq.iter(world) {
            if enemy.kind != EnemyKind::Throne {
                continue;
            }
            let big = brain.is_some_and(|b| b.phase == BossPhase::Beam)
                || brain.is_some_and(|b| b.phase == BossPhase::Telegraph);
            let path = if big {
                "images/sprThroneFlameBig.png"
            } else {
                "images/sprThroneFlameIdle.png"
            };
            let frames = strip_frames(assets, path).max(1);
            let frame = ((blink_t * 12.0).floor() as i32).rem_euclid(frames as i32);
            for off in [
                Vec2::new(-89.0, -43.0),
                Vec2::new(91.0, -43.0),
                Vec2::new(-89.0, 13.0),
                Vec2::new(91.0, 13.0),
            ] {
                if let Some(s) = assets.sprite_for(path, frame, pos.0 + off, false, 0.0, [1.0; 4]) {
                    out.push(s);
                }
            }
        }
        let mut q = world.query::<(
            &Pos,
            &Enemy,
            Option<&Velocity>,
            Option<&AimDir>,
            Option<&SpriteAnim>,
            Option<&HitFlash>,
            Option<&EnemyBrain>,
        )>();
        // Guns behind.
        for (pos, enemy, vel, aim, _anim, _flash, brain) in q.iter(world) {
            let Some(gun_path) = enemy_gun_art(enemy.kind) else {
                continue;
            };
            let Some(gunangle) = brain.map(|b| b.gunangle) else {
                continue;
            };
            if gunangle.to_degrees().rem_euclid(360.0) > 180.0 {
                continue;
            }
            let flip = vel.map(|v| v.0.x < 0.0).unwrap_or(false)
                || aim.map(|a| a.0.x < 0.0).unwrap_or(false);
            if let Some(s) =
                assets.sprite_for_full(gun_path, 0, pos.0, false, flip, gunangle, [1.0; 4])
            {
                out.push(s);
            }
        }
        // Bodies.
        for (pos, enemy, vel, aim, anim, flash, _brain) in q.iter(world) {
            let (path, frame) = match anim {
                Some(a) => (a.path.as_str(), a.frame as i32),
                None => (crate::enemy_data::enemy_def(enemy.kind).sprite, 0),
            };
            let flip = vel.map(|v| v.0.x < 0.0).unwrap_or(false)
                || aim.map(|a| a.0.x < 0.0).unwrap_or(false);
            if let Some(s) = assets.sprite_for(path, frame, pos.0, flip, 0.0, flash_tint(flash)) {
                out.push(s);
            }
        }
        // Guns in front.
        for (pos, enemy, vel, aim, _anim, _flash, brain) in q.iter(world) {
            let Some(gun_path) = enemy_gun_art(enemy.kind) else {
                continue;
            };
            let Some(gunangle) = brain.map(|b| b.gunangle) else {
                continue;
            };
            if gunangle.to_degrees().rem_euclid(360.0) <= 180.0 {
                continue;
            }
            let flip = vel.map(|v| v.0.x < 0.0).unwrap_or(false)
                || aim.map(|a| a.0.x < 0.0).unwrap_or(false);
            if let Some(s) =
                assets.sprite_for_full(gun_path, 0, pos.0, false, flip, gunangle, [1.0; 4])
            {
                out.push(s);
            }
        }
    }

    /// Current-slot recoil for the behind-gun draw (GML `wkick` rides the
    /// weapon visual; the player block reads it back).
    fn weapon_visual_kick(
        kicks: &[(Entity, usize, f32, f32)],
        owner: Entity,
        slot: usize,
    ) -> (f32, f32) {
        kicks
            .iter()
            .find(|(o, s, _, _)| *o == owner && *s == slot)
            .map(|(_, _, k, a)| (*k, *a))
            .unwrap_or((0.0, 0.0))
    }

    // Player: GML `Player/Draw_0` order verbatim — Eyes underlay, back
    // guns (extra-wep fan + silver bwep), behind-gun or body, bubble.
    // The front gun rides the held-gun block below (skipped there while
    // `back`). Invuln blink rides the tint.
    {
        let blink_t = world
            .get_resource::<repame_sim::SimTime>()
            .map(|t| t.elapsed_secs)
            .unwrap_or(0.0);
        // Recoil snapshot for the behind-gun draw (ends before the
        // player query borrows).
        let kicks: Vec<(Entity, usize, f32, f32)> = world
            .query::<&WeaponVisual>()
            .iter(world)
            .map(|v| (v.owner, v.slot as usize, v.wkick, v.wep_angle))
            .collect();
        let mut q = world.query::<(
            Entity,
            &Pos,
            &Player,
            &crate::anim::PlayerAnim,
            Option<&AimDir>,
            Option<&Velocity>,
            Option<&SpriteAnim>,
            Option<&HitFlash>,
            Option<&Health>,
            Option<&RaceState>,
            Option<&Inventory>,
            Option<&Telekinesis>,
            Option<&ThroneSit>,
        )>();
        for (entity, pos, player, pa, aim, vel, anim, flash, health, race, inv_opt, telek, sit) in
            q.iter(world)
        {
            let (anim_path, anim_frame) = match anim {
                Some(a) => (a.path.as_str(), a.frame as i32),
                None => {
                    let moving = vel.map(|v| v.0.length_squared() > 100.0).unwrap_or(false);
                    (if moving { pa.walk } else { pa.idle }, 0)
                }
            };
            // GML seated body (`spr_gosit` beat into `spr_sit`): derive
            // from the idle strip name; missing strips fall back.
            let (sit_base, sit_fr): (Option<String>, i32) = match sit {
                Some(s) if anim_path.contains("Idle") => {
                    let going = s.timer.remaining_secs() > 0.5;
                    let base = anim_path.replace("Idle", if going { "GoSit" } else { "Sit" });
                    let frames = strip_frames(assets, &base).max(1);
                    let elapsed = 11.5 - s.timer.remaining_secs().min(11.5);
                    let fr = if going {
                        ((elapsed * 12.0).floor() as i32).min(frames as i32 - 1)
                    } else {
                        0
                    };
                    if assets.uv(&base, fr).is_some() {
                        (Some(base), fr)
                    } else {
                        (None, anim_frame)
                    }
                }
                _ => (None, anim_frame),
            };
            let (path, frame): (&str, i32) = match &sit_base {
                Some(base) => (base.as_str(), sit_fr),
                None => (anim_path, anim_frame),
            };
            // GML Player/Step_0 facing quadrant law (`right`/`back` from
            // `gunangle`): `right = -1` when 90 < gunangle < 270, else 1;
            // `back` when 0 < gunangle < 180. Falls back to velocity.x
            // when aim is dead.
            let (right, back): (f32, bool) =
                match aim.map(|a| a.0).filter(|a| a.length_squared() > 0.001) {
                    Some(a) => {
                        let (r, b) = crate::player::gml_player_right_back_from_aim(a);
                        (r, b > 0.0)
                    }
                    None => {
                        let x = vel.map(|v| v.0.x).unwrap_or(0.0);
                        (if x < 0.0 { -1.0 } else { 1.0 }, false)
                    }
                };
            let flip = right < 0.0;
            let gunangle = aim.map(|a| a.0.y.atan2(a.0.x)).unwrap_or(if flip {
                std::f32::consts::PI
            } else {
                0.0
            });
            let is_eyes = race.is_some_and(|rs| rs.race == RaceId::Eyes);
            let is_steroids = race.is_some_and(|rs| rs.race == RaceId::Steroids);
            let swapanim = inv_opt.map(|inv| inv.swapanim).unwrap_or(0.0);
            let shine = inv_opt
                .map(|inv| inv.shine.floor() as i32)
                .unwrap_or(0)
                .max(0);

            // Eyes underlay: held-spec mind power, else MonsterStyle body.
            if is_eyes {
                let img = ((blink_t * 12.0).floor() as i32).max(0);
                if telek.is_some() {
                    let tb = player.mutations.contains(&MutationId::ThroneButt);
                    let path = if tb {
                        "images/sprMindPowerTB.png"
                    } else {
                        "images/sprMindPower.png"
                    };
                    if let Some(s) = assets.sprite_for(path, img % 3, pos.0, flip, 0.0, [1.0; 4]) {
                        out.push(s);
                    }
                } else if player.ultra == Some(UltraMutationId::EyesMonsterStyle) {
                    if let Some(s) = assets.sprite_for(
                        "images/sprEyesB.png",
                        img % 6,
                        pos.0,
                        flip,
                        0.0,
                        [0.8, 0.984, 0.78, 1.0],
                    ) {
                        out.push(s);
                    }
                }
            }

            // Back guns: extra-wep fan + silver bwep (non-Steroids).
            // `bwepright` (melee `bwepflip`, else facing) mirrors the
            // fan and the silver draw vertically.
            if let Some(inv) = inv_opt {
                let bwep_slot = if inv.weapon_slots > 1 {
                    (inv.current + 1) % inv.weapon_slots
                } else {
                    inv.current
                };
                let bwep = inv.weapons[bwep_slot.min(inv.weapon_slots.max(1) - 1)];
                let bwep_melee = weapon_meta(bwep).wep_mele;
                let bwep_right: f32 = if bwep_melee && bwep != WeaponId::NONE {
                    inv.bwepflip
                } else {
                    right
                };
                let extra: Vec<WeaponId> = (2..inv.weapon_slots)
                    .map(|i| inv.weapons[i])
                    .filter(|w| *w != WeaponId::NONE)
                    .collect();
                let deg = std::f32::consts::PI / 180.0;
                if !extra.is_empty() {
                    let mut backwep_angle = 90.0 - 5.0 * right;
                    let count = extra.len() as f32;
                    for (i, w) in extra.iter().enumerate() {
                        let meta = weapon_meta(*w);
                        if meta.wep_sprt.is_empty() || meta.wep_sprt == "mskNone" {
                            continue;
                        }
                        let t = i as f32 / count;
                        // `merge_color(c_silver, c_black, i / count)`.
                        let silver = 0.75 * (1.0 - t);
                        let at = Vec2::new(pos.0.x - right * (2.0 + i as f32), pos.0.y + swapanim);
                        if let Some(s) = assets.sprite_for_full(
                            &format!("images/{}.png", meta.wep_sprt),
                            0,
                            at,
                            false,
                            bwep_right < 0.0,
                            backwep_angle * deg,
                            [silver, silver, silver, 1.0],
                        ) {
                            out.push(s);
                        }
                        backwep_angle += (15.0 + i as f32) * right;
                    }
                }
                if !is_steroids && inv.weapon_slots > 1 {
                    let meta = weapon_meta(bwep);
                    if bwep != WeaponId::NONE
                        && !meta.wep_sprt.is_empty()
                        && meta.wep_sprt != "mskNone"
                    {
                        let at = Vec2::new(pos.0.x - right * 2.0, pos.0.y + swapanim);
                        if let Some(s) = assets.sprite_for_full(
                            &format!("images/{}.png", meta.wep_sprt),
                            0,
                            at,
                            false,
                            bwep_right < 0.0,
                            (90.0 - 5.0 * right + 15.0 * right) * deg,
                            [0.75, 0.75, 0.78, 1.0],
                        ) {
                            out.push(s);
                        }
                    }
                }
            }

            // Behind-gun (the gun block skips the current slot while
            // `back`; frame rides `trigger_fingers_shine`).
            if back {
                if let Some(inv) = inv_opt {
                    let w = inv.weapons[inv.current.min(inv.weapon_slots.max(1) - 1)];
                    let meta = weapon_meta(w);
                    let path = if !meta.wep_sprt.is_empty() && meta.wep_sprt != "mskNone" {
                        std::borrow::Cow::Owned(format!("images/{}.png", meta.wep_sprt))
                    } else {
                        std::borrow::Cow::Borrowed("images/sprRevolver.png")
                    };
                    if w != WeaponId::NONE {
                        // GML Draw_0: gunangle + wepangle*(1 - wkick/20),
                        // positioned at player + lengthdir(-wkick, swing).
                        // Kick/wep_angle come from the slot-0 WeaponVisual.
                        let (wkick, wep_ang) = weapon_visual_kick(&kicks, entity, 0);
                        let ang = crate::player::held_weapon_angle(gunangle, wep_ang, wkick);
                        let at = Vec2::new(
                            pos.0.x + (-wkick) * ang.cos(),
                            pos.0.y + (-wkick) * ang.sin() - swapanim,
                        );
                        let mirror = if meta.wep_mele {
                            inv.wepflip < 0.0
                        } else {
                            flip
                        };
                        if let Some(s) =
                            assets.sprite_for_full(&path, shine, at, false, mirror, ang, [1.0; 4])
                        {
                            out.push(s);
                        }
                    }
                }
            }

            let mut tint = flash_tint(flash);
            let invuln_live = health.is_some_and(|h| !h.invuln.is_finished());
            if invuln_live {
                let wave = (0.5 + 0.5 * (blink_t * 24.0).sin()) as f32;
                tint[3] *= 0.25 + 0.55 * wave;
            }
            if let Some(s) = assets.sprite_for(path, frame, pos.0, flip, 0.0, tint) {
                out.push(s);
            }
            // GML underwater bubble (Oasis, non-Fish/Robot): animated
            // `sprPlayerBubble` over the body.
            let area = world
                .get_resource::<Run>()
                .map(|r| r.area)
                .unwrap_or(AreaId::Desert);
            let race_ok = race.is_none_or(|rs| {
                !matches!(
                    rs.race,
                    crate::data::RaceId::Fish | crate::data::RaceId::Robot
                )
            });
            if area == AreaId::Oasis && race_ok {
                let frames = strip_frames(assets, "images/sprPlayerBubble.png").max(1);
                let bframe = ((blink_t * 8.0).floor() as u32 % frames) as i32;
                if let Some(s) = assets.sprite_for(
                    "images/sprPlayerBubble.png",
                    bframe,
                    pos.0,
                    false,
                    0.0,
                    [1.0; 4],
                ) {
                    out.push(s);
                }
            }
            // GML Gun Warrant (`infammo`): warrant seal over the player.
            if player.warrant > 0.0 {
                let frames = strip_frames(assets, "images/sprGunWarrant.png").max(1);
                let wframe = ((player.warrant * 0.4).floor() as i32).rem_euclid(frames as i32);
                if let Some(s) = assets.sprite_for(
                    "images/sprGunWarrant.png",
                    wframe,
                    pos.0,
                    false,
                    0.0,
                    [1.0; 4],
                ) {
                    out.push(s);
                }
            }
        }
    }

    // Projectiles: strip from the name tables; spin to the velocity
    // heading (y-down atan2, matching bevy transform rotation).
    // Grenade pre-detonation telegraph (bevy `tick_grenade_fuse`
    // tint half): white/black strobe under 0.334 s, solid white once
    // the friction switch armed.
    {
        let mut q = world.query::<(
            &Pos,
            &Projectile,
            Option<&Velocity>,
            &Team,
            Option<&SlashProjectile>,
            Option<&GrenadeFuse>,
            Option<&crate::comps_a::ProjectileVisual>,
        )>();
        for (pos, proj, vel, team, slash, fuse, visual) in q.iter(world) {
            let path = projectile_art(proj, team, slash, visual);
            let frame = projectile_frame(assets, path, &proj.life);
            let rotation = vel
                .filter(|v| v.0.length_squared() > 1e-6)
                .map(|v| v.0.y.atan2(v.0.x))
                .unwrap_or(0.0);
            let tint = match fuse {
                Some(f) => {
                    let remaining = f.alarm1.remaining_secs();
                    if remaining <= 0.334 && remaining > 0.01 {
                        let phase = (remaining * 30.0) as i32 % 5;
                        if phase <= 2 {
                            [1.0, 1.0, 1.0, 1.0]
                        } else {
                            [0.0, 0.0, 0.0, 1.0]
                        }
                    } else if f.friction_switched {
                        [1.0, 1.0, 1.0, 1.0]
                    } else {
                        [1.0; 4]
                    }
                }
                None => [1.0; 4],
            };
            if let Some(s) = assets.sprite_for(path, frame, pos.0, false, rotation, tint) {
                out.push(s);
            }
        }
    }

    // Hit-FX pass (bevy z 14 band, over projectiles, under guns):
    // portal shock/clear/strike (GML 1-frame-per-step strips),
    // bullet-hit/dust/fade oneshots with `FxAngle` orientation, and
    // bare-PNG static fallbacks — all fading in the last 0.12 s.
    {
        let mut q = world.query::<(&Pos, &PortalShock)>();
        for (pos, shock) in q.iter(world) {
            // GML `image_xscale/yscale = 2.25`.
            let elapsed = shock.timer.duration() - shock.timer.remaining_secs();
            let frames = strip_frames(assets, "images/sprPortalClear.png").max(1);
            let frame = ((elapsed * 30.0).floor() as u32).min(frames - 1) as i32;
            if let Some(def_size) = assets.native_size("images/sprPortalClear.png") {
                let size = def_size * 2.25;
                if let Some(s) = assets.sprite_sized(
                    "images/sprPortalClear.png",
                    frame,
                    pos.0,
                    size,
                    false,
                    [1.0; 4],
                ) {
                    out.push(s);
                }
            }
        }
        let mut q = world.query::<(&Pos, &PortalClear)>();
        for (pos, clear) in q.iter(world) {
            let elapsed = clear.timer.duration() - clear.timer.remaining_secs();
            let frames = strip_frames(assets, "images/sprPortalClear.png").max(1);
            let frame = ((elapsed * 30.0).floor() as u32).min(frames - 1) as i32;
            if let Some(s) = assets.sprite_for(
                "images/sprPortalClear.png",
                frame,
                pos.0,
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
        }
        let mut q = world.query::<(&Pos, &PortalStrike)>();
        for (pos, strike) in q.iter(world) {
            let elapsed = strike.timer.duration() - strike.timer.remaining_secs();
            let frames = strip_frames(assets, "images/sprRogueStrike.png").max(1);
            let frame = ((elapsed * 30.0).floor() as u32).min(frames - 1) as i32;
            if let Some(s) = assets.sprite_for(
                "images/sprRogueStrike.png",
                frame,
                pos.0,
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
        }
        // Animated hit-effect leftovers (anything with a live strip
        // that isn't a corpse/portal/actor).
        let mut q = world.query::<(
            &Pos,
            &SpriteAnim,
            Option<&HitFlash>,
            Option<&FxAngle>,
            Option<&PickupLifetime>,
            Option<&Prop>,
            Option<&PropSprites>,
            Option<&Corpse>,
            Option<&Portal>,
            Option<&Enemy>,
            Option<&Player>,
            Option<&Projectile>,
            Option<&Pickup>,
            Option<&WeaponVisual>,
            Option<&SwingFx>,
        )>();
        for (
            pos,
            anim,
            flash,
            angle,
            lifetime,
            prop,
            sprites,
            corpse,
            portal,
            enemy,
            player,
            proj,
            pickup,
            gun,
            swing,
        ) in q.iter(world)
        {
            if prop.is_some()
                || enemy.is_some()
                || player.is_some()
                || proj.is_some()
                || pickup.is_some()
                || gun.is_some()
                || swing.is_some()
            {
                continue;
            }
            let is_corpse =
                corpse.is_some() || (sprites.is_some() && prop.is_none()) || portal.is_some();
            if is_corpse {
                continue;
            }
            let rotation = angle.map(|a| a.0).unwrap_or(0.0);
            let mut tint = flash_tint(flash);
            if let Some(lt) = lifetime {
                tint[3] *= (lt.timer.remaining_secs() / 0.12).clamp(0.0, 1.0);
            }
            if let Some(s) =
                assets.sprite_for(&anim.path, anim.frame as i32, pos.0, false, rotation, tint)
            {
                out.push(s);
            }
        }
        // Static FX fallbacks (bevy bare-PNG arm).
        let mut q = world.query::<(&Pos, &StaticFx, Option<&FxAngle>, Option<&PickupLifetime>)>();
        for (pos, fx, angle, lifetime) in q.iter(world) {
            let rotation = angle.map(|a| a.0).unwrap_or(0.0);
            let mut tint = [1.0; 4];
            if let Some(lt) = lifetime {
                tint[3] *= (lt.timer.remaining_secs() / 0.12).clamp(0.0, 1.0);
            }
            if let Some(s) = assets.sprite_for(fx.path, 0, pos.0, false, rotation, tint) {
                out.push(s);
            }
        }
    }

    // Held guns: GML Player/Draw_0 verbatim.
    // Position = player + lengthdir(-wkick, gunangle + wepangle*(1 - wkick/20))
    // NO +12 forward hold. Sprite origin is the grip.
    {
        // Wall centers for the bolt-weapon laser-sight march.
        let walls: Vec<Vec2> = world
            .query::<(&Pos, &WallCell)>()
            .iter(world)
            .map(|(p, _)| p.0)
            .collect();
        let mut q = world.query::<&WeaponVisual>();
        for gun in q.iter(world) {
            let Some(pos) = world.get::<Pos>(gun.owner) else {
                continue;
            };
            let aim = world
                .get::<AimDir>(gun.owner)
                .map(|a| a.0)
                .unwrap_or(Vec2::X);
            // GML `back`: the current gun draws behind the body (in the
            // player block above), never here. Frame rides
            // `trigger_fingers_shine`; y-mirror rides `wepflip` for
            // melee (`wepright`/`bwepright`, else facing).
            let (back_current, shine, mirror, is_steroids_slot1) = world
                .get::<Inventory>(gun.owner)
                .map(|inv| {
                    let (_, back_v) = if aim.length_squared() > 0.001 {
                        crate::player::gml_player_right_back_from_aim(aim)
                    } else {
                        (if aim.x < 0.0 { -1.0 } else { 1.0 }, -1.0)
                    };
                    let back = back_v > 0.0;
                    let melee = weapon_meta(gun.wep_id).wep_mele;
                    let facing = if aim.x < 0.0 { -1.0 } else { 1.0 };
                    // GML `Player/Draw_0` mirror law: primary yscale is
                    // `wepright` (`wepflip` for melee, else facing); the
                    // slot-1 secondary draws with yscale `-bwepright`
                    // (`-(bwepflip)` for melee, else `-facing`).
                    let upright = if gun.slot == 1 {
                        -(if melee { inv.bwepflip } else { facing })
                    } else if melee {
                        inv.wepflip
                    } else {
                        facing
                    };
                    (
                        back && gun.slot as usize == inv.current,
                        inv.shine.floor() as i32,
                        upright < 0.0,
                        gun.slot == 1,
                    )
                })
                .unwrap_or((false, 0, aim.x < 0.0, false));
            if back_current {
                continue;
            }
            let meta = weapon_meta(gun.wep_id);
            let path = if !meta.wep_sprt.is_empty() && meta.wep_sprt != "mskNone" {
                Cow::Owned(format!("images/{}.png", meta.wep_sprt))
            } else {
                Cow::Borrowed("images/sprRevolver.png")
            };
            let angle = aim.y.atan2(aim.x);
            // GML: gunangle + (wepangle * (1 - wkick/20))
            let swing = crate::player::held_weapon_angle(angle, gun.wep_angle, gun.wkick);
            let forward = Vec2::new(swing.cos(), swing.sin());

            // GML lengthdir(-wkick, swing). Steroids bwep: y - 4.
            let mut hold = pos.0 + forward * (-gun.wkick);
            if is_steroids_slot1 {
                hold.y -= 4.0;
            }
            if let Some(s) =
                assets.sprite_for_full(&path, shine.max(0), hold, false, mirror, swing, [1.0; 4])
            {
                out.push(s);
            }
            // GML bolt-weapon laser sight (not the disc gun): 2 px
            // march to the first wall (1000-step cap). Origin is the
            // player (Steroids second: y - 4), NOT the muzzle/hold.
            if meta.wep_type == AmmoType::Bolts && meta.wep_name != "DISC GUN" {
                let origin = if is_steroids_slot1 {
                    Vec2::new(pos.0.x, pos.0.y - 4.0)
                } else {
                    pos.0
                };
                let step = Vec2::new(angle.cos(), angle.sin()) * 2.0;
                let mut tip = origin;
                for _ in 0..1000 {
                    let next = tip + step;
                    let hit = walls.iter().any(|w| w.distance(next) < 8.0);
                    tip = next;
                    if hit {
                        break;
                    }
                }
                let dist = origin.distance(tip);
                if let Some(native) = assets.native_size("images/sprLaserSightPlayer.png") {
                    let size = Vec2::new(native.x * (dist / 2.0 + 2.0), native.y);
                    // Origin of laser strip is at player; with center-anchor mid-point:
                    let mid = origin + Vec2::new(angle.cos(), angle.sin()) * (size.x * 0.5);
                    let mut s = match assets.sprite_sized(
                        "images/sprLaserSightPlayer.png",
                        0,
                        mid,
                        size,
                        false,
                        [1.0; 4],
                    ) {
                        Some(s) => s,
                        None => continue,
                    };
                    s.rotation = angle;
                    s.anchor = Vec2::new(0.5, 0.5);
                    out.push(s);
                }
            }
        }
    }

    // Melee swings + wall-hits (bevy z 24, over guns): `SwingFx`
    // rotation with the live strip, falling back to the wall-hit art.
    {
        let mut q = world.query::<(&Pos, &SwingFx, Option<&SpriteAnim>)>();
        for (pos, fx, anim) in q.iter(world) {
            let (path, frame) = match anim {
                Some(a) => (a.path.as_str(), a.frame as i32),
                None => ("images/sprMeleeHitWall.png", 0),
            };
            if let Some(s) = assets.sprite_for(path, frame, pos.0, false, fx.angle, [1.0; 4]) {
                out.push(s);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Transient combat FX → instances.
// ---------------------------------------------------------------------------

/// Beam strip (320x32, 1 frame): stretched along the sim segment.
pub const BEAM_STRIP: &str = "images/sprLightBeam.png";
/// Lightning strip (2x2, 4 frames): stretched along the arc segment,
/// frame following the 0.09 s life.
pub const ARC_STRIP: &str = "images/sprLightning.png";
/// Muzzle tongue size (no muzzle strip exists in the catalog).
pub const MUZZLE_SIZE: Vec2 = Vec2::new(24.0, 14.0);
/// Damage-number font size in world units (bevy
/// `DamageNumberConfig::default().font_size` verbatim).
pub const NUMBER_TEXT_SIZE: f32 = 28.0;

/// Solid-color fallback quad (particle convention): [`SpriteInstance`]
/// has no fill flag, so — like [`particle_sprites`] — this samples the
/// full first atlas page (`uv 0..1`, page 0) and leans on the tint.
/// Sim-clock seconds driving bevy `animate_environment`-law alphas
/// (`SurfacePulse::alpha_at`); 0.0 headless without the resource.
fn pulse_now(world: &World) -> f32 {
    world
        .get_resource::<repame_sim::SimTime>()
        .map(|t| t.elapsed_secs as f32)
        .unwrap_or(0.0)
}

/// Used when the catalog lacks the strip (subset packs) or when no
/// strip exists at all (muzzle tongues, hazard discs).
fn white_quad(center: Vec2, rotation: f32, size: Vec2, tint: [f32; 4]) -> SpriteInstance {
    SpriteInstance {
        center,
        rotation,
        size,
        anchor: Vec2::new(0.5, 0.5),
        flip_x: false,
        flip_y: false,
        uv_min: Vec2::ZERO,
        uv_max: Vec2::ONE,
        color: tint_to_linear(tint),
        page: 0,
        z: 0.0,
        blend: SpriteBlend::Alpha,
    }
}

/// Snapshot transient sim FX into GPU sprites (drawn after
/// [`world_instances`], on top): [`Beam`] strips, [`LightningArc`]
/// segments, [`FiredWeapon`] muzzle tongues, [`HazardCloud`] pulses,
/// and live [`Particle`]s (via [`particle_sprites`]).
///
/// Snapshot transient sim FX into GPU sprites (drawn after
/// [`world_instances`], on top): [`Beam`] strips, [`LightningArc`]
/// segments, [`FiredWeapon`] muzzle tongues, [`HazardCloud`] pulses,
/// and live [`Particle`]s (via [`particle_sprites`]).
///
/// Legacy 320x240 GUI view map (bevy `menus::nt_view` law, kept for
/// reference): fill scale `s = min(w/320, h/240)` with centered
/// offsets. Superseded by the GML law ([`gml_view_size`]: the GUI is
/// the live view, 426x240 at 16:9); the text pipeline now maps through
/// [`gui_texts_dp`] instead.
#[derive(Clone, Copy, Debug)]
pub struct NtView {
    pub s: f32,
    pub ox: f32,
    pub oy: f32,
}

/// Bevy `nt_view` law (viewport falls back to 1280x720). Kept for
/// reference; the live pipeline uses [`gml_view_size`].
pub fn nt_view_for(viewport_w: f32, viewport_h: f32) -> NtView {
    let w = if viewport_w > 1.0 { viewport_w } else { 1280.0 };
    let h = if viewport_h > 1.0 { viewport_h } else { 720.0 };
    let s = (w / 320.0).min(h / 240.0);
    NtView {
        s,
        ox: (w - 320.0 * s) * 0.5,
        oy: (h - 240.0 * s) * 0.5,
    }
}

/// Fill-scale map from a world-space view rect. GML law verbatim:
/// the GUI *is* the view (`display_set_gui_size(view_width,
/// view_height)` in `scrSetViewSize`), so GUI px == view px 1:1 —
/// identity map, no letterbox. Callers pass the live
/// [`view_rect_world`] rect; `gx`/`gy` authored in GML view px land on
/// the view 1:1 (at 16:9 the view is 426x240, so right-anchored rows
/// use `view[2] - 2`, centered headers `view[2] / 2`).
#[derive(Clone, Copy, Debug)]
pub struct HudGuiMap {
    pub s: f32,
    pub ox: f32,
    pub oy: f32,
}

/// GML identity GUI map over a [`view_rect_world`] rect.
pub fn hud_gui_map(view: [f32; 4]) -> HudGuiMap {
    let _ = view;
    HudGuiMap {
        s: 1.0,
        ox: 0.0,
        oy: 0.0,
    }
}

/// GUI point → world point through [`hud_gui_map`].
pub fn hud_gui_to_world(map: HudGuiMap, view: [f32; 4], gx: f32, gy: f32) -> Vec2 {
    Vec2::new(view[0] + map.ox + gx * map.s, view[1] + map.oy + gy * map.s)
}

/// Place a strip by its art top-left in GUI px (bevy `gm_sprite`
/// origin law for zero-origin art): quad top-left lands on the mapped
/// GUI point, size = native * `mul` * map scale.
///
/// GML `draw_sprite` (NOT `_ext`) honors the strip origin: the DRAW POINT
/// is `pos - origin`, i.e. art top-left lands on `pos - origin`. Callers
/// pass the GML draw position verbatim for ALL strips — this helper reads
/// the catalog origin and offsets the quad top-left by `-origin * mul *
/// map.s`, so pixels land exactly where GML puts them regardless of the
/// strip's origin (`sprUltraLevel` origin (4,5), Rogue pips (1,1), etc.).
pub fn hud_gui_place(
    assets: &RenderAssets,
    path: &str,
    frame: i32,
    gx: f32,
    gy: f32,
    mul: f32,
    tint: [f32; 4],
    map: HudGuiMap,
    view: [f32; 4],
) -> Option<SpriteInstance> {
    let (uv, def) = assets.uv(path, frame)?;
    let anchor = def.anchor();
    let size = Vec2::new(def.w as f32 * mul * map.s, def.h as f32 * mul * map.s);
    // GML draw point minus the origin: `draw_sprite(spr, sub, x, y)` puts
    // the art top-left at `(x - xorigin, y - yorigin)`.
    let top_left = hud_gui_to_world(
        map,
        view,
        gx - def.xorigin * mul,
        gy - def.yorigin * mul,
    );
    Some(SpriteInstance {
        center: top_left + Vec2::new(anchor[0] * size.x, anchor[1] * size.y),
        rotation: 0.0,
        size,
        anchor: Vec2::new(anchor[0], anchor[1]),
        flip_x: false,
        flip_y: false,
        uv_min: Vec2::new(uv.min[0], uv.min[1]),
        uv_max: Vec2::new(uv.max[0], uv.max[1]),
        color: tint_to_linear(tint),
        page: uv.page,
        z: 0.0,
        blend: SpriteBlend::Alpha,
    })
}

/// One HUD text in 320x240 GUI space (GML `scrDrawPlayerHUD` regions +
/// `scrDrawMiscHUD` bottom-right rows: string, GUI pos, sRGB color,
/// alignment).
#[derive(Clone, Debug, PartialEq)]
pub struct HudGuiText {
    pub text: String,
    pub gx: f32,
    pub gy: f32,
    pub color: [u8; 4],
    pub centered: bool,
    pub middle_y: bool,
    /// Right-aligned row (misc-HUD clock/area at `view_width - 2`).
    pub right: bool,
}

fn hud_gui_left(
    text: String,
    gx: f32,
    gy: f32,
    color: [u8; 4],
    centered: bool,
    middle_y: bool,
) -> HudGuiText {
    HudGuiText {
        text,
        gx,
        gy,
        color,
        centered,
        middle_y,
        right: false,
    }
}

/// GML weapon-row draw order: `_wep` at position 0, `_bwep` at 1,
/// then `extra_weps` (`scrDrawPlayerHUD` iterates `_hud_weapon_index`
/// 0, 1, 2... — position 0 is ALWAYS the primary slot, never the
/// active one; swapping guns swaps `_wep`/`_bwep` sim-side instead).
/// Returns inventory slot indices in draw order
/// (unbounded — Steroids/Cuz extras draw past two slots).
fn hud_weapon_order(hud: &HudState) -> Vec<usize> {
    let n = hud.weapon_ids.len().max(1);
    (0..n).collect()
}

/// GML weapon-row x positions: 24, then +44, then +20 per extra
/// (`_dx += 44` for the first slot, `+20` afterwards once extras
/// exist — i.e. 24/68/88/108/...).
fn hud_weapon_dx(pos: usize) -> f32 {
    match pos {
        0 => 24.0,
        1 => 68.0,
        _ => 88.0 + (pos as f32 - 2.0) * 20.0,
    }
}

/// GML `scrDrawPlayerHUD` text laws verbatim:
/// - `"hp/max"` at (67,7) white centered (`23+44, 7`);
/// - level number at (11,16) white centered middle-anchored, only while
///   below the level cap (`PLAYER_LEVEL_MAX = 10`; max level draws the
///   `sprUltraLevel` sprite instead);
/// - per-slot ammo at (42+pos*44,21) left in active-first draw order:
///   white when the slot's ammo type is active else silver; `c_uidark`
///   when dry; red (active) or gray when at/below one pickup; skipped
///   for melee/absent slots;
/// - `"LOW HP"` at (110,7) red left when `hp <= 4 && hp != max` while
///   recently hurt (`drawlowhp`; `HitFlash` stands in here, and the
///   shell adds the `sin(wave)` blink);
/// - `scrDrawMiscHUD` bottom-right rows, right-aligned at `view-2`:
///   run clock then map name (gated on `show_timer`/`show_area`).
///
/// Deferred GML HUD rows (need sim state the port never tracks):
/// `FAINTED` pulse, bleed gray bar, hurt white flash, and the analog
/// `scrDrawClock` surface clock (only the digital `timer_string` is
/// drawn). Everything else in `scrDrawPlayerHUD`/`scrDrawMiscHUD` draws:
/// health/exp/ammo strips + ultra/skill rows (sprites), per-slot ammo +
/// LOW-HP + low-ammo block (above), clock/area (below), event icons
/// (sprites), interaction prompt (`hud_texts`).
pub fn hud_gui_texts(world: &mut World) -> Vec<HudGuiText> {
    use crate::data::{AmmoKind, ammo_pickup_amount};

    let hud: HudState = sync_hud_state(world);
    let mut out = Vec::new();
    out.push(hud_gui_left(
        format!("{}/{}", hud.hp.max(0), hud.max_hp.max(0)),
        67.0,
        7.0,
        [255, 255, 255, 255],
        true,
        false,
    ));
    if hud.level < 10 {
        out.push(hud_gui_left(
            hud.level.to_string(),
            11.0,
            16.0,
            [255, 255, 255, 255],
            true,
            true,
        ));
    }
    let steroids = world
        .get_resource::<SelectedCharacter>()
        .is_some_and(|s| s.0 == crate::data::RaceId::Steroids);
    let order = hud_weapon_order(&hud);
    let extras = order.len() > 2;
    let t1 = weapon_meta(
        hud.weapon_ids
            .get(order[0])
            .copied()
            .unwrap_or(WeaponId::NONE),
    )
    .wep_type;
    for (pos, slot) in order.iter().copied().enumerate() {
        // GML draws ammo text for both slots normally, but only the
        // first once extras exist.
        if extras && pos > 0 {
            continue;
        }
        let amount = hud.weapon_ammo.get(slot).copied().unwrap_or(-1);
        if amount < 0 {
            continue;
        }
        let kind = hud
            .weapon_ids
            .get(slot)
            .map(|id| weapon_meta(*id).wep_type)
            .map(|t| match t {
                AmmoType::Bullets => AmmoKind::Bullets,
                AmmoType::Shells => AmmoKind::Shells,
                AmmoType::Bolts => AmmoKind::Bolts,
                AmmoType::Explosives => AmmoKind::Explosives,
                AmmoType::Energy => AmmoKind::Energy,
                AmmoType::None => AmmoKind::None,
            })
            .unwrap_or(AmmoKind::None);
        // GML `_is_active_ammo`: draw position 0 (or Steroids dual)
        // or the type matches the primary weapon's type.
        let active = pos == 0 || steroids || kind as usize == t1 as usize;
        // GML `scrDrawPlayerHUD:152-157` verbatim: active white else
        // silver; dry `c_uidark` (#333333 = 51,51,51); at/below one
        // pickup red (active, `c_red` = 252,56,0) or gray (inactive,
        // `c_gray` = 128,128,128); healthy inactive `c_silver`
        // (192,192,192).
        let color = if amount <= 0 {
            [51, 51, 51, 255]
        } else if amount <= ammo_pickup_amount(kind).max(0) {
            if active {
                [252, 56, 0, 255]
            } else {
                [128, 128, 128, 255]
            }
        } else if active {
            [255, 255, 255, 255]
        } else {
            [192, 192, 192, 255]
        };
        out.push(hud_gui_left(
            amount.to_string(),
            // GML `scrDrawPlayerHUD:162` verbatim: `_dx + 18, _dy + 5`
            // with `_dy = 16` (NOT 21 — the ammo digits sit 5 px below
            // the gun row origin, inside the 14-tall part window).
            hud_weapon_dx(pos) + 18.0,
            16.0 + 5.0,
            color,
            false,
            false,
        ));
    }
    let recently_hurt = world
        .query::<(&Player, &HitFlash)>()
        .iter(world)
        .next()
        .is_some();
    if hud.hp <= 4 && hud.hp != hud.max_hp && recently_hurt {
        // GML `LOW HP`: `draw_set_color(c_red)` = (255,0,0).
        out.push(hud_gui_left(
            "LOW HP".to_string(),
            110.0,
            7.0,
            [255, 0, 0, 255],
            false,
            false,
        ));
    }
    // GML `scrDrawPlayerHUD:253-284` low-ammo block verbatim: the held
    // weapon (plus the second on Steroids when its type differs) shows
    // `LOW <type>` / `NOT ENOUGH <type>` / `EMPTY` / `NOT ENOUGH RADS`
    // red-left at `(55 + icons*12, 35)` while `drawempty > 0` and the
    // shell blinks (`sin(wave) > 0` gate lives in `hud_overlay_lines`).
    // `drawempty` = the dry-fire toast window: the port raises it when
    // the EMPTY / NOT ENOUGH RADS toast is live.
    {
        let dry = world
            .get_resource::<crate::comps_a::Toast>()
            .is_some_and(|t| t.text == "EMPTY" || t.text == "NOT ENOUGH RADS");
        if dry {
            let primary = hud.weapon_ids.first().copied().unwrap_or(WeaponId::NONE);
            let pmeta = weapon_meta(primary);
            let ptype = pmeta.wep_type as usize;
            let cost = pmeta.wep_cost as i32;
            let ammo = hud.ammo.get(ptype.min(5)).copied().unwrap_or(0);
            let rads = hud.rads;
            let rad_cost = u32::from(pmeta.wep_rads);
            let kind_name = match ptype {
                1 => "BULLETS",
                2 => "SHELLS",
                3 => "BOLTS",
                4 => "EXPLOSIVES",
                5 => "ENERGY",
                _ => "NONE",
            };
            let txt = if ptype != 0 && ammo <= 0 {
                Some("EMPTY".to_string())
            } else if ptype != 0 && ammo < cost {
                Some(format!("NOT ENOUGH {kind_name}"))
            } else if rads < rad_cost {
                Some("NOT ENOUGH RADS".to_string())
            } else if ptype != 0 {
                Some(format!("LOW {kind_name}"))
            } else {
                None
            };
            if let Some(txt) = txt {
                out.push(HudGuiText {
                    text: txt,
                    gx: 55.0,
                    gy: 35.0,
                    color: [255, 0, 0, 255],
                    centered: false,
                    middle_y: false,
                    right: false,
                });
            }
        }
    }
    // Misc-HUD clock + map name (`scrDrawMiscHUD`: right-aligned rows
    // stacking up from `view_height - (font + 10)`; row height 9 px).
    let show_timer = world
        .get_resource::<crate::savedata_part::SaveData>()
        .is_none_or(|s| s.settings.show_timer);
    let paused = world
        .get_resource::<crate::state::Paused>()
        .is_some_and(|p| p.0)
        || world
            .get_resource::<crate::state::OverlayMenu>()
            .is_some_and(|o| *o != crate::state::OverlayMenu::None);
    let transitioning = world
        .get_resource::<crate::comps_b::FloorTransition>()
        .is_some_and(|f| f.active);
    let boss_intro = world
        .query::<&crate::comps_a::BossIntro>()
        .iter(world)
        .next()
        .is_some();
    let show_area = world
        .get_resource::<crate::savedata_part::SaveData>()
        .is_none_or(|s| s.settings.show_area)
        && !transitioning
        && (!paused || boss_intro);
    let mut gy = 240.0 - (7.0 + 10.0);
    if show_timer && !hud.timer_string.is_empty() {
        out.push(HudGuiText {
            text: hud.timer_string.clone(),
            gx: 320.0 - 2.0,
            gy,
            color: [255, 255, 255, 255],
            centered: false,
            middle_y: false,
            right: true,
        });
        gy -= 9.0;
    }
    if show_area && !hud.area_string.is_empty() {
        out.push(HudGuiText {
            text: hud.area_string.clone(),
            gx: 320.0 - 2.0,
            gy,
            color: [255, 255, 255, 255],
            centered: false,
            middle_y: false,
            right: true,
        });
    }
    out
}

/// One positioned overlay text in 320x240 GUI space (bevy
/// `nt_text_at`/`bigname_button_at`/title/main-menu law verbatim, plus
/// GML right-aligned misc-HUD rows).
#[derive(Clone, Debug, PartialEq)]
pub struct MenuGuiText {
    pub text: String,
    pub gx: f32,
    pub gy: f32,
    pub color: [u8; 4],
    /// Silkscreen px at 1x GUI scale (7 body, 10 bigname, 14 title
    /// name, 16 main menu).
    pub px: f32,
    pub centered: bool,
    pub middle_y: bool,
    /// Right-aligned row: the box right edge lands on `gx`
    /// (`scrDrawMiscHUD` clock/area at `view_width - 2`).
    pub right: bool,
}

/// Overlay row: (segments, dp top-left, font px, centered, box width,
/// right-aligned). `segments` are `(text, sRGB color 0..1)` runs from the
/// GML `draw_text_nt` `@`-tag parser ([`nt_text_segments`]); the shell
/// draws each run in its color with the standard 1px black shadow.
pub type GuiRow = (Vec<(String, [f32; 4])>, [f32; 2], f32, bool, f32, bool);

/// GML `draw_text_nt` `@`-tag colors verbatim
/// (`scripts/draw_text_nt/draw_text_nt.gml:215-222`): `@s` silver
/// (125,131,141), `@b` blue (22,97,223), `@r` red (252,56,0), `@y`
/// yellow (250,171,0), `@d` dark gray (59,62,67), `@g` green
/// (68,198,22), `@p` purple (86,34,110), `@w` white. Unknown tags
/// (incl. `@q` shake, `@(`/sprite, `@[`/`@]` bold, `@.` reset) carry no
/// color and render in the row's base color.
pub fn nt_tag_color(tag: char) -> Option<[u8; 4]> {
    match tag {
        's' => Some([125, 131, 141, 255]),
        'b' => Some([22, 97, 223, 255]),
        'r' => Some([252, 56, 0, 255]),
        'y' => Some([250, 171, 0, 255]),
        'd' => Some([59, 62, 67, 255]),
        'g' => Some([68, 198, 22, 255]),
        'p' => Some([86, 34, 110, 255]),
        'w' => Some([255, 255, 255, 255]),
        _ => None,
    }
}

/// Split GML `draw_text_nt` text into color runs: `@x` opens the tagged
/// color, `\@` escapes a literal `@`. `#` stays inside the run as a
/// literal line break (the shell wraps on it; GML `#` writes a newline
/// via `string_hash_to_newline` before parsing). Raw `\n` never reaches
/// this backend: multi-line rows are pre-split into one [`MenuGuiText`]
/// per visual line at the producer (GML draws each line of the block at
/// its own y). Every other `@?` sequence keeps its chars in the base
/// color.
pub fn nt_text_segments(text: &str, base: [u8; 4]) -> Vec<(String, [u8; 4])> {
    fn push(seg: &mut Vec<(String, [u8; 4])>, buf: &mut String, color: [u8; 4]) {
        if buf.is_empty() {
            return;
        }
        seg.push((std::mem::take(buf), color));
    }
    let mut segs: Vec<(String, [u8; 4])> = Vec::new();
    let mut buf = String::new();
    let mut color = base;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&next) = chars.peek() {
                if next == '@' {
                    buf.push(chars.next().unwrap_or('@'));
                    continue;
                }
            }
            buf.push(c);
            continue;
        }
        if c == '@' {
            match chars.peek() {
                Some(&t) if t.is_ascii_alphabetic() => {
                    let t = chars.next().unwrap_or('w');
                    let tag = t.to_ascii_lowercase();
                    push(&mut segs, &mut buf, color);
                    if let Some(tagged) = nt_tag_color(tag) {
                        color = tagged;
                    }
                }
                Some(&'(') => {
                    // `@(sprite,...)` inline icon: skip to `)`, keep base.
                    push(&mut segs, &mut buf, color);
                    chars.next();
                    for sc in chars.by_ref() {
                        if sc == ')' {
                            break;
                        }
                    }
                }
                _ => buf.push('@'),
            }
            continue;
        }
        buf.push(c);
    }
    push(&mut segs, &mut buf, color);
    if segs.is_empty() {
        segs.push((String::new(), base));
    }
    segs
}

/// Generic GUI text → dp mapper (GML law: the GUI is the live view,
/// `scrSetViewSize` + `display_set_gui_size`; `vw` is the live GUI
/// width in px from [`gml_view_size`], always 240 tall). Left texts sit
/// at `gx * k`, centered texts center their `2*gx` box on `gx * k`,
/// right texts right-align their `2*(vw-gx)` box `vw-gx` px from the
/// canvas right edge (`scrDrawMiscHUD` clock/area at `view_width - 2`
/// verbatim); `middle_y` centers the line on `gy`. `k` is dp per GUI
/// px (`canvas_h / 240`).
pub fn gui_texts_dp(canvas_dp: [f32; 2], items: Vec<MenuGuiText>) -> Vec<GuiRow> {
    let vw = gml_view_size(canvas_dp)[0];
    let w = canvas_dp[0].max(1.0);
    let k = (canvas_dp[1].max(1.0) / 240.0).max(1e-6);
    items
        .into_iter()
        .map(|t| {
            let font_px = (t.px * k).round().clamp(8.0, 180.0);
            let (left, box_w, centered) = if t.centered {
                // GML centers `draw_text_nt` on `gx`: the box spans the
                // symmetric fit around `gx` (`2 * min(gx, vw - gx)`),
                // centered content inside. View-centered rows (gx = cx)
                // get the full width; column headers (stats TOTAL at
                // `statx`) center on their column. The old full-width
                // box dragged column headers to `vw/2` (the centered
                // part of the "left-shortened stats" bug); the older
                // symmetric fit without centering clipped wide rows.
                let half = t.gx.min(vw - t.gx).max(1.0);
                let bw = (2.0 * half * k).max(font_px);
                (t.gx * k - bw * 0.5, bw, true)
            } else if t.right {
                // GML right-aligns on `gx`: the row's box right edge
                // lands exactly on `gx * k`, content right-aligned
                // inside. The old fixed 120px box pushed short rows
                // left of their anchor (the "left-shortened stats"
                // bug); HUD clock/area rows need a wide-enough box to
                // reach `vw - 2`, stats names only need their column.
                let right = t.gx * k;
                let bw = if t.gx >= vw - 3.0 {
                    (120.0 * k).max(font_px)
                } else {
                    (60.0 * k).max(font_px)
                };
                (right - bw, bw, false)
            } else {
                let left = t.gx * k;
                let bw = (200.0 * k).max(font_px).min((w - left).max(font_px));
                (left, bw, false)
            };
            let top = if t.middle_y {
                t.gy * k - font_px * 0.5
            } else {
                t.gy * k
            };
            let segs = nt_text_segments(&t.text, t.color)
                .into_iter()
                .map(|(s, c)| {
                    (
                        s,
                        [
                            c[0] as f32 / 255.0,
                            c[1] as f32 / 255.0,
                            c[2] as f32 / 255.0,
                            c[3] as f32 / 255.0,
                        ],
                    )
                })
                .collect();
            (
                segs,
                [left.max(0.0), top.max(0.0)],
                font_px,
                centered,
                box_w,
                t.right,
            )
        })
        .collect()
}

/// [`hud_gui_texts`] mapped to dp canvas points through
/// [`gui_texts_dp`] (7px Silkscreen rows). Right-anchored misc-HUD rows
/// (clock/area) ride the live GUI width (`view_width - 2` verbatim).
pub fn hud_gui_texts_dp(world: &mut World, canvas_dp: [f32; 2]) -> Vec<GuiRow> {
    let show_hud = world
        .get_resource::<crate::savedata_part::SaveData>()
        .is_none_or(|s| s.settings.show_hud);
    if !show_hud {
        return Vec::new();
    }
    let player_alive = world.query::<&Player>().iter(world).next().is_some();
    if !player_alive {
        return Vec::new();
    }
    let vw = gml_view_size(canvas_dp)[0];
    let cx = vw * 0.5;
    let mut items: Vec<MenuGuiText> = hud_gui_texts(world)
        .into_iter()
        .map(|t| MenuGuiText {
            text: t.text,
            gx: if t.right { vw - 2.0 } else { t.gx },
            gy: t.gy,
            color: t.color,
            px: 7.0,
            centered: t.centered,
            middle_y: t.middle_y,
            right: t.right,
        })
        .collect();
    let hud: HudState = sync_hud_state(world);
    // GML `SkillText` verbatim (`scrLevelUpScreenSubmit` spawn at the
    // level-up spot + `TopCont/Draw_75` / `LevCont/Draw_64` draw
    // `"@d" + loc(txt)` AT THE SkillText INSTANCE POS — i.e. the
    // offer Carlson position (view center-ish, NOT a fixed y=200) —
    // centered-middle, blinking while `disappear % 2` (the blink gate
    // lives in `hud_overlay_lines`, like LOW HP). The port has no
    // SkillText entity, so the toast rides the offer textbox anchor
    // `(cx, 179)` while an offer is open, else the legacy top-center
    // fallback. (The old fixed `gy: 200.0` matched neither.)
    if !hud.toast.is_empty() {
        let offer_open = world.get_resource::<PendingMutation>().is_some()
            || world.get_resource::<PendingUltra>().is_some();
        items.push(MenuGuiText {
            text: format!("@d{}", hud.toast),
            gx: cx,
            gy: if offer_open { 179.0 } else { 120.0 },
            color: [255, 255, 255, 255],
            px: 7.0,
            centered: true,
            middle_y: true,
            right: false,
        });
    }
    if hud.boss_max > 0 {
        items.push(MenuGuiText {
            text: format!("{} {}/{}", hud.boss_name, hud.boss_hp, hud.boss_max),
            gx: cx,
            gy: 20.0,
            color: [255, 64, 64, 255],
            px: 7.0,
            centered: true,
            middle_y: false,
            right: false,
        });
    }
    if hud.idpd_warning {
        items.push(MenuGuiText {
            text: "!! IDPD INCOMING !!".to_string(),
            gx: cx,
            gy: 32.0,
            color: [255, 64, 64, 255],
            px: 7.0,
            centered: true,
            middle_y: false,
            right: false,
        });
    }
    gui_texts_dp(canvas_dp, items)
}

// ---------------------------------------------------------------------------
// Menu overlay texts (bevy `menus/mod.rs` + `title_screen.rs` verbatim:
// strings, 320x240 positions, colors, px sizes).
// ---------------------------------------------------------------------------

const GUI_CREAM: [u8; 4] = [238, 239, 225, 255];
const GUI_GRAY: [u8; 4] = [125, 131, 141, 255];
const GUI_MID: [u8; 4] = [153, 153, 153, 255];
/// GML `c_uidark` (#333333): unavailable menu entries, dry ammo text.
const GUI_UIDARK: [u8; 4] = [51, 51, 51, 255];
const GUI_WHITE: [u8; 4] = [255, 255, 255, 255];
const GUI_RED2: [u8; 4] = [221, 56, 45, 255];
const GUI_GREEN: [u8; 4] = [98, 220, 88, 255];
/// GML yellow `(250, 171, 0)` (`draw_text_nt` `@y` tag,
/// `scripts/draw_text_nt/draw_text_nt.gml:218`).
const GUI_GOLD: [u8; 4] = [250, 171, 0, 255];
/// GML `c_ultra` (#3dc616): LEVEL ULTRA rest tint.
const GUI_ULTRA: [u8; 4] = [61, 198, 22, 255];

fn gui_body(text: impl Into<String>, gx: f32, gy: f32, color: [u8; 4]) -> MenuGuiText {
    MenuGuiText {
        text: text.into(),
        gx,
        gy,
        color,
        px: 7.0,
        centered: false,
        middle_y: false,
        right: false,
    }
}

fn gui_center(text: impl Into<String>, gx: f32, gy: f32, color: [u8; 4]) -> MenuGuiText {
    MenuGuiText {
        text: text.into(),
        gx,
        gy,
        color,
        px: 7.0,
        centered: true,
        middle_y: false,
        right: false,
    }
}

fn gui_button(text: impl Into<String>, gx: f32, gy: f32, color: [u8; 4]) -> MenuGuiText {
    MenuGuiText {
        text: text.into(),
        gx,
        gy,
        color,
        px: 10.0,
        centered: true,
        middle_y: false,
        right: false,
    }
}

/// Bevy `mutation_choice_parts` verbatim: ULTRA prefix + name/desc
/// split on em-dash or hyphen. Returns the GML skill id too (the
/// `mutation_choice_ids` parallel row), so the Throne Butt special can
/// key off the picked skill, not the parsed name.
fn mutation_choice_parts(choice: &str) -> (Option<u8>, String, String) {
    // `sync_hud_state` writes `mutation_choices` and
    // `mutation_choice_ids` in the same order; resolve the id by index.
    // (The id row is passed separately at the call site; this helper
    // keeps the parse half pure.)
    let trimmed = choice.trim();
    let (is_ultra, trimmed) = if let Some(rest) = trimmed.strip_prefix("ULTRA:") {
        (true, rest.trim())
    } else {
        (false, trimmed)
    };
    let _ = is_ultra;
    if let Some((name, desc)) = trimmed.split_once(" \u{2014} ") {
        (None, name.trim().to_string(), desc.trim().to_string())
    } else if let Some((name, desc)) = trimmed.split_once(" - ") {
        (None, name.trim().to_string(), desc.trim().to_string())
    } else {
        (None, trimmed.to_string(), String::new())
    }
}
/// GML `Credits/Other_11` `credittext` verbatim (unlocalized
/// defaults): 14 titled sections. Rows carry their GML `@w`/`@s`/`@y`
/// color tags and `#` line breaks; the tag backend (`nt_text_segments`
/// + `AnnotatedText`) renders them. GML joins each section's array with
/// `"\n@s"` and draws ONE centered-middle `draw_text_nt` at
/// `(gui_w/2, gui_h/2)` — tall sections (`height > gui_h - 36`) pan via
/// `scroll` (`MenuState::credits_scroll`).
pub const CREDIT_SECTIONS: &[&[&str]] = &[
    &["@yVLAMBEER @wPRESENTS"],
    &["@wA GAME BY"],
    &[
        "@wProject Lead, Design & Development",
        "@sJan Willem Nijman",
        "@wProduction & Additional Development",
        "@sRami Ismail",
        "@wArt Direction, Lead Artist & Animation",
        "@sPaul Veer",
        "@wOriginal Soundtrack",
        "@sJukio Kallio",
        "@wSound Design",
        "@sJoonas Turner",
        "@wAdditional Artwork",
        "@sJustin Chan",
    ],
    &[
        "@wAdditional Music Credits",
        "",
        "@wOasis",
        "@sJUKIO KALLIO#Danny Baranowski",
        "",
        "@wPizza Sewers",
        "@sJUKIO KALLIO#Eirik Suhrke",
        "",
        "@wVenus",
        "@sJUKIO KALLIO#Adam 'Doseone' Drucker",
        "",
        "@wCursed Caves",
        "@sJUKIO KALLIO#Richard 'Disasterpeace' Vreeland",
        "",
        "@wJungle",
        "@sJUKIO KALLIO#Daniel Hagstrom",
        "",
        "@wMansion & Hyper Crystal",
        "@sJUKIO KALLIO#Joonas Turner",
        "",
        "@wTea Break",
        "Original Composition by Eirik Suhrke@w",
        "",
        "@wAdditional Music Consultation",
        "@sJoonas Turner@w",
    ],
    &[
        "@wThe voice of Fish, Crystal, Eyes,#Melting, Plant, Steroids,#Robot, Chicken,Horror,#Yung Cuz, Enemies & Bosses",
        "@sJoonas Turner@w",
        "",
        "@wRebel & Additional IDPD",
        "@sIsa And",
        "",
        "@wY.V. & Venus Enemies",
        "@sAdam Drucker",
        "",
        "@wFrog",
        "@sJukio Kallio",
        "",
        "@wCaptain",
        "@sMyy Lohi",
        "",
        "@wRogue",
        "@sDanielle McRae",
        "",
        "@wAdditional IDPD",
        "@sNiilo Takalainen",
    ],
    &[
        "@wPromotional Artwork",
        "@sJustin Chan",
        "",
        "@wLanguage Design",
        "@sJoonas Turner",
        "",
        "@wBackend Programming#Community Management#Release Management",
        "@sRami Ismail",
        "",
        "@wAdditional Programming#Porting",
        "VADYM 'YELLOWAFTERLIFE' DIACHENKO",
        "",
        "@wAddional Engineering",
        "@sJuju Adams",
    ],
    &[
        "@wTrailers",
        "@sBram Ruiter#Daniel Carneiro#Kert Gartner#Marlon Wiebe#",
        "",
        "@wEvent Logistics",
        "@sAdriel Wallick#Fred Wood#Jon Kay#Maya Kramer#Rami Ismail",
    ],
    &[
        "@wMarketing",
        "@sRami Ismail",
        "",
        "@wAdditional Marketing",
        "@sJan Willem Nijman#Joonas Turner#Jukio Kallio#Justin Chan#Paul Veer",
    ],
    &[
        "@wNUCLEAR THRONE MOBILE",
        "",
        "@wPROJECT CREATOR & MAINTAINER",
        "TONCHO_",
        "",
        "@wTESTERS",
        "@sEVILCAT   CZIMBALA   SKUHNUH   WINT#VOROB   DRAKIN   TIREDMETAL   #LOMEGAI   PLOWECH   SENJEY",
        "",
        "#@wLOCALIZATION CONTRIBUTORS",
        "@wLEGACY LOCALIZERS",
        "",
        "@wBRAZILIAN PORTUGUESE#@sMIGUEL#TITIO CARTOLA#GUIZARD#POTATO SALAD#",
        "@wSPANISH#@sFRI#BAYRON#WALTERLZ#",
        "@wPOLISH#@sLOSSTAROTT#",
        "@wUKRAINIAN#@sPRAWO#REPKON#",
        "@wPERSIAN#@sPLOOB",
    ],
    &[
        "@wValve#@sAnna Sweet#Augusta Butlin#John Bartkiw#Matt Nickerson##@wHumble#@sAlex Ting#Will Turnbull##@wTwitch#@sErnest Le#Jon 'Carnage' Joyce##@wDevstream Support#@sBenn Powell#Dominik Johann#Gieron#Lisa 'Wertle' Brown#Seef 'IgnoTV' Ismail#Sleepcycles#Vlambot#XSplit##@wYoYo Games#@sMike Dailly#Russell Kay#Peter Hall#Sandy Duncan#Stuart Poole##@wSONY Computer Entertainment#@sAdam Boyes#Andrew Wong#Ben Andac#Blanca Nunez Ibanez#Bob Jordan#Brian Silva#Dan 'Shoe' Hsu#Gio Corsi#Jericho Guerrero#John Drake#John Kopp#Julio Perez#Justin Massongill#Laura Casey#Lorenzo Grimaldi#Nathalie Closs#Nick Suttner#Richard Lee#Ryan Clements#Shahid Kamal Ahmad#Shane Bettenhausen#Shuhei Yoshida#Sid Shuman",
        "",
        "@wMerchandise",
        "Fangamer#Fred Wood#Gijs van Kooten#Jon 'Jonty' Hicks#Jon Kay#Level Up Studios#Shawn Handyside#",
        "",
        "@wWiki Master",
        "Gieron",
        "",
        "@wThronebutt",
        "Ivan Ostric",
        "Chitinlink",
        "",
        "Update Videos",
        "Tengu Drop",
        "",
        "@wBobbing & Weaving",
        "SleepCycles",
        "",
        "@wCommunity Challenges",
        "Solid",
        "",
        "@wWorld Tournament",
        "Smite",
        "",
        "@wSpecial Thanks",
        "17-Bit#Adriel Wallick#Alexa#Allan Keith#Anthony Carboni#Bart Jan Bultman#Beau Blyth#Ben Vance#Bisnap#Brandon Boyer#Brian van Bruggen#Bronson Zgeb#Burgeroise#Chris Charla#CodeMan38#Crystal Chan#Daniel 'Manna' Hagstrom#DED#DevilAzite#Derek Yu#Dutch Game Garden#Evan Balster#Glitch City#Greg Wohlwend#Hademar#Jack Oatley#James Eagler#Janette Silvasti#Jerry Holkins#Jerry van Kooten#Jord Coerse#JMickle#Kakujo#Kitty Calis#Kosti Kallio#Kristy Norindr#Mark Essen#Martin Kvale#Martin van der Wolf#Mike Feith#Mirva Kontio#Neri & Bali#Nigel Lowrie#Phil Tibitoski#Pissasedat#Poppenkast#Ross Turner#Roy Nathan de Groot#Richard Boeser#Seef Ismail#Sidonie Tise#Sirpa Kallio#Sirpa Niskala#Slackbot#Ster#Sven Ruthner#Ted Martens#Teddy Diefenbach#Tingle#Tuuka#Vincent Nijman#Wiley 'Willy Woggins' Wiggins#Zach Gage",
        "",
        "@wNuclear Throne would not exist without#Humble, MOJANG and the organizers#of MOJAM 2013",
        "",
        "@wNuclear Throne was created using#YoYoGames' GameMaker",
        "",
        "@wNuclear Throne extends its thanks to#COOL BIG MONY BIZNIZ INC.#for providing the image of Yung Venuz.",
    ],
    &[
        "@wNuclear Throne could not exist#without the Vlambeer community.#This game could have gone off-track#in so many ways,#but your patience, support, enthusiasm,#feedback and hard work kept us sharp,#motivated and eager to find#fun and new ways to end your runs.",
    ],
    &[
        "@wThanks to our family, our friends#and our partners#for their unyielding support,#love, care and patience.#@w<3",
    ],
    &["@wThank you for playing!"],
    &["@wNUCLEAR THRONE"],
];

/// Section count for the credits cycler ([`MenuState::credits_section`]).
pub fn credit_section_count() -> usize {
    CREDIT_SECTIONS.len().max(1)
}

/// Menu overlay texts for one [`crate::MenuOverlay`] (bevy
/// `compose_root` panels verbatim: splash is sprite-only (empty);
/// loading shows GENERATING % + roadmap; main menu the 5 labels;
/// title the race name/passive/active; mutation the offer + selection;
/// game over the DEAD stats; pause the buttons; settings the full
/// page stack; credits the static lines).
///
/// `vw` is the live GML GUI width in px ([`gml_view_size`], 426 at
/// 16:9): view-centered rows use `vw / 2`, right-anchored rows
/// `vw - N` (`scrMakePauseButtons`, `scrDrawMiscHUD` verbatim);
/// left-anchored rows keep literal x.
pub fn menu_gui_texts_vw(kind: crate::MenuOverlay, world: &mut World, vw: f32) -> Vec<MenuGuiText> {
    let cx = vw * 0.5;
    match kind {
        // Boot reel captions (GML `Vlambeer/Draw_0` verbatim, expanded to
        // one row per visual line: `gui_text_layer` renders each row
        // `.single_line()`, so embedded `\n`/`#` would never break —
        // GML's single `draw_text_nt` block must arrive pre-split).
        // Mode 0 save note: 2 white lines, block middle at `cy+24`
        // (140/150 middle-centers ≈ 144). Mode 1 Gamemaker line at
        // `(cx, cy)`, `@s`-silver per GML. Mode 3 team block at
        // `(cx, cy)`: `@yVLAMBEER`, `@s&`, four `@w` names,
        // `PRESENT`; ys 80..160 average exactly 120 (blanks shape the
        // rhythm). Modes 2/4 sprite-only.
        crate::MenuOverlay::Splash => {
            let mode = world
                .get_resource::<SplashState>()
                .map(|s| s.mode)
                .unwrap_or(0);
            match mode {
                0 => vec![
                    MenuGuiText {
                        text: "DO NOT TURN OFF NUCLEAR THRONE".to_string(),
                        gx: cx,
                        gy: 140.0,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                    },
                    MenuGuiText {
                        text: "WHILE THIS SAVING ICON IS DISPLAYED.".to_string(),
                        gx: cx,
                        gy: 150.0,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                    },
                ],
                1 => vec![MenuGuiText {
                    text: "@sMADE IN GAMEMAKER".to_string(),
                    gx: cx,
                    gy: 120.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: true,
                    middle_y: true,
                    right: false,
                }],
                3 => [
                    ("@yVLAMBEER", 80.0),
                    ("@s&", 100.0),
                    ("@wPAUL VEER", 110.0),
                    ("@wJUKIO KALLIO", 120.0),
                    ("@wJOONAS TURNER", 130.0),
                    ("@wJUSTIN CHAN", 140.0),
                    ("@wPRESENT", 160.0),
                ]
                .iter()
                .map(|(text, gy)| MenuGuiText {
                    text: text.to_string(),
                    gx: cx,
                    gy: *gy,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: true,
                    middle_y: true,
                    right: false,
                })
                .collect(),
                _ => Vec::new(),
            }
        }
        crate::MenuOverlay::Loading => {
            // GML `GenCont/Draw_0` text layer verbatim: `GENERATING... %`
            // at `(_cx, _cy - 54)` from `Floor/goal` (pad-2), the
            // `VERIFYING... %` Venuz branch (`level >= 10` + a Venuz
            // player), the `@s`-tip at `(_cx, _cy + 24)`, and the
            // roadmap area/kill strings at `(drawx-60/drawx+23,
            // drawy-14)` (the dots ride `menu_sprites`). `_cx/_cy` is
            // the view center (`vw/2`, 120 at 240 high).
            let progress = world
                .get_resource::<crate::state::LoadingState>()
                .map(|l| l.progress)
                .unwrap_or(1.0);
            let pct = (progress.clamp(0.0, 1.0) * 100.0).round() as u32;
            // GML `GenCont/Draw_0:13-18`: Venuz verifying branch.
            let is_venuz = world
                .query::<&crate::comps_a::RaceState>()
                .iter(world)
                .any(|rs| rs.race == crate::data::RaceId::Venuz);
            let deep_enough = world
                .get_resource::<crate::comps_a::Run>()
                .is_some_and(|r| r.floor >= 10);
            let verb = if deep_enough && is_venuz {
                "VERIFYING"
            } else {
                "GENERATING"
            };
            let mut out = vec![gui_center(
                format!("{verb}... {pct:02}%"),
                cx,
                66.0,
                GUI_GRAY,
            )];
            // Tip is picked once per load (stable across draws).
            let tip = {
                let fresh = world
                    .get_resource::<crate::state::LoadingState>()
                    .map(|l| l.tip.clone())
                    .unwrap_or_default();
                if fresh.is_empty() {
                    let picked = world
                        .get_resource::<crate::comps_a::Run>()
                        .map(crate::progression::pick_loading_tip)
                        .unwrap_or_else(|| "KILL ENEMIES TO LEVEL UP".to_string());
                    if let Some(mut loading) =
                        world.get_resource_mut::<crate::state::LoadingState>()
                    {
                        loading.tip = picked.clone();
                    }
                    picked
                } else {
                    fresh
                }
            };
            out.push(gui_center(format!("@s{tip}"), cx, 144.0, GUI_GRAY));
            if let Some(run) = world.get_resource::<crate::comps_a::Run>() {
                if run.world != 0 || run.floor != 0 {
                    // GML `GenCont/Draw_0` roadmap at `(_cx, _cy)` FULL
                    // width (unlike GameOver's `_x - 48`): the strings
                    // ride `(drawx-60, drawy-14)` / `(drawx+23, drawy-14)`
                    // with `drawx = cx`, i.e. `(cx-60, cy-14)` /
                    // `(cx+23, cy-14)` = `(cx-60, 106)` / `(cx+23, 106)`.
                    out.push(MenuGuiText {
                        text: crate::hud::run_area_string(run),
                        gx: cx - 60.0,
                        gy: 106.0,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: false,
                        middle_y: true,
                        right: false,
                    });
                    out.push(MenuGuiText {
                        text: run.total_kills.to_string(),
                        gx: cx + 23.0,
                        gy: 106.0,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: false,
                        middle_y: true,
                        right: false,
                    });
                }
            }
            out
        }
        crate::MenuOverlay::MainMenu => {
            const LABELS: [(&str, i32); 5] = [
                ("PLAY", 0),
                ("CO-OP", 1),
                ("SETTINGS", 2),
                ("STATS", 3),
                ("QUIT", 4),
            ];
            let menu = world.get_resource::<MenuState>().cloned();
            if menu.as_ref().is_some_and(|m| m.play_submenu) {
                use crate::savedata_part::SaveData;
                use crate::state::menus::{play_row_available, play_row_name, play_rows};
                let save = world
                    .get_resource::<SaveData>()
                    .cloned()
                    .unwrap_or_default();
                let rows = play_rows(&save);
                let cursor = menu.map(|m| m.play_cursor).unwrap_or(0);
                let n = rows.len();
                let mut out: Vec<MenuGuiText> = rows
                    .iter()
                    .enumerate()
                    .map(|(i, row)| {
                        // GML `MainMenuButton/Other_10` verbatim:
                        // DAILY/WEEKLY draw `c_uidark`-dimmed when
                        // `!can_daily/can_weekly` (offline here, so
                        // always dimmed) but stay listed in the same
                        // positions.
                        let color = if !play_row_available(&save, *row) {
                            GUI_UIDARK
                        } else if i == cursor {
                            GUI_WHITE
                        } else {
                            GUI_MID
                        };
                        MenuGuiText {
                            text: play_row_name(*row).to_string(),
                            gx: cx,
                            gy: 120.0 - n as f32 * 12.0 + i as f32 * 24.0,
                            color,
                            px: 12.0,
                            centered: true,
                            middle_y: true,
                            right: false,
                        }
                    })
                    .collect();
                out.push(MenuGuiText {
                    text: "BACK".to_string(),
                    gx: 20.0,
                    gy: 20.0,
                    color: GUI_GRAY,
                    px: 10.0,
                    centered: true,
                    middle_y: false,
                    right: false,
                });
                return out;
            }
            let cursor = menu.map(|m| m.main_menu_cursor).unwrap_or(0);
            LABELS
                .iter()
                .map(|(label, index)| {
                    let available = matches!(index, 0 | 2 | 3 | 4);
                    let color = if !available {
                        GUI_UIDARK
                    } else if *index as usize == cursor {
                        GUI_WHITE
                    } else {
                        GUI_MID
                    };
                    MenuGuiText {
                        text: label.to_string(),
                        gx: cx,
                        gy: 120.0 - 48.0 + *index as f32 * 24.0,
                        color,
                        px: 12.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                    }
                })
                .collect()
        }
        crate::MenuOverlay::Stats => {
            use crate::hud::{gml_area_map_name, scr_time, scr_time_speedrun};
            use crate::savedata_part::{SaveData, unlock_progress};
            use crate::state::menus::race_from_gml_id;
            let save = world
                .get_resource::<SaveData>()
                .cloned()
                .unwrap_or_default();
            let race_name = |gml: u8| {
                race_from_gml_id(gml as usize)
                    .map(|r| character_def(r).name.to_ascii_uppercase())
                    .unwrap_or_else(|| "?".to_string())
            };
            // GML `DrawStats/Draw_0` + `scrDrawStats` verbatim: title via
            // `draw_text_bigname` at `(view_center, view_top + 24)` in
            // `c_uigray`; left column `statx = view_left + 110`,
            // `staty = view_top + 36 + 4`; right column
            // `statx = view_left + view_width - 70`. Names right-aligned
            // uigray at `statx - 1`, values left white at `statx + 1`,
            // headers centered; blank headers advance the (fractional)
            // line.
            let stat_name = |out: &mut Vec<MenuGuiText>, col: f32, line: &mut f32, name: &str| {
                // GML `draw_stat` verbatim: the NAME right-aligns on
                // `statx - 1` (right edge = `col - 1`), so the row's box
                // right edge sits at `col - 1`, not `col`.
                out.push(MenuGuiText {
                    text: name.to_string(),
                    gx: col - 1.0,
                    gy: 40.0 + *line * 8.0,
                    color: GUI_MID,
                    px: 7.0,
                    centered: false,
                    middle_y: true,
                    right: true,
                });
            };
            let stat_val = |out: &mut Vec<MenuGuiText>, col: f32, line: &mut f32, val: String| {
                out.push(MenuGuiText {
                    text: val,
                    gx: col + 1.0,
                    gy: 40.0 + *line * 8.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: false,
                    middle_y: true,
                    right: false,
                });
                *line += 1.0;
            };
            let header = |out: &mut Vec<MenuGuiText>, col: f32, line: &mut f32, name: &str| {
                if name.is_empty() {
                    *line += 1.0;
                    return;
                }
                // GML `draw_stat_header` verbatim: centered on `statx`
                // (`fa_center` at `col`), NOT on the view center — the
                // old `gui_center(name, col, ...)` built a full-width
                // box that centered on `vw/2`, dragging "TOTAL" right
                // of its column (the centered part of the
                // "left-shortened stats" bug).
                out.push(MenuGuiText {
                    text: name.to_string(),
                    gx: col,
                    gy: 40.0 + *line * 8.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: true,
                    middle_y: true,
                    right: false,
                });
                *line += 1.0;
            };
            let mut out = vec![MenuGuiText {
                text: "STATS".to_string(),
                gx: cx,
                gy: 24.0,
                color: GUI_MID,
                px: 12.0,
                centered: true,
                middle_y: true,
                right: false,
            }];
            let (lx, rx) = (110.0, vw - 70.0);
            let mut l = 0.0;
            header(&mut out, lx, &mut l, "TOTAL");
            let (un, unmax) = unlock_progress(&save);
            // GML `scrDrawStats:82` verbatim: `string_pad_zeroes(round(
            // unlock / unlockmax * 100), 2) + "%"` — rounds (never
            // truncates) and can display 100%.
            let unpct = if unmax == 0 {
                0
            } else {
                ((un as f32 / unmax as f32) * 100.0).round() as u32
            };
            for (name, val) in [
                ("kills", save.total_kills.to_string()),
                ("loops", save.total_loops.to_string()),
                ("runs", save.total_runs.to_string()),
                ("deaths", save.total_deaths.to_string()),
                ("wins", save.total_wins.to_string()),
                ("time", scr_time(save.total_time_steps / 30)),
                ("unlocks", format!("{unpct:02}%")),
            ] {
                stat_name(&mut out, lx, &mut l, name);
                stat_val(&mut out, lx, &mut l, val);
            }
            header(&mut out, lx, &mut l, "");
            if save.total_runs > 0 {
                header(&mut out, lx, &mut l, "BEST RUN");
                stat_name(&mut out, lx, &mut l, &race_name(save.best_run_race));
                stat_val(
                    &mut out,
                    lx,
                    &mut l,
                    gml_area_map_name(
                        save.best_run_area,
                        save.best_run_sub,
                        save.best_run_loop,
                        false,
                    ),
                );
                stat_name(&mut out, lx, &mut l, "kills");
                stat_val(&mut out, lx, &mut l, save.best_run_kills.to_string());
                header(&mut out, lx, &mut l, "");
            }
            let mut r = 0.0;
            if save.total_runs > 0 && save.total_wins > 0 {
                header(&mut out, rx, &mut r, "BEST STREAK");
                stat_name(&mut out, rx, &mut r, &race_name(save.best_streak_race));
                stat_val(&mut out, rx, &mut r, save.win_streak_best.to_string());
                header(&mut out, rx, &mut r, "");
            }
            if save.total_wins > 0 {
                header(&mut out, rx, &mut r, "BEST TIME");
                stat_name(&mut out, rx, &mut r, &race_name(save.best_time_race));
                stat_val(
                    &mut out,
                    rx,
                    &mut r,
                    scr_time_speedrun(save.best_time_steps),
                );
                header(&mut out, rx, &mut r, "");
            }
            // Deferred GML `scrDrawStats:108-114` DAILY block: it needs
            // the daily-run systems the port lacks (`dbst_*` bests +
            // `ctot_days` producers; DAILY/WEEKLY PLAY rows deny with
            // `sndNoSelect`, so `dailies > 0` can never hold here).
            // Deferred GML `scrDrawCharStats` per-race page: it needs
            // per-race `ctot_*`/`cbst_*`/`hbst_*` tracking arrays plus a
            // select overlay; the sim only keeps the global aggregates
            // the TOTAL/BEST blocks above read.
            // GML `scrDrawStats` HARD block verbatim: gated on `hardgot`
            // + hard runs, shows the global hard-best race + hard map
            // (`scrAreaGetMapName(..., hard = true)`), kills, and runs.
            if save.hardmode_unlocked && save.hard_runs > 0 {
                header(&mut out, rx, &mut r, "HARD");
                stat_name(&mut out, rx, &mut r, &race_name(save.hard_best_race));
                stat_val(
                    &mut out,
                    rx,
                    &mut r,
                    gml_area_map_name(
                        save.hard_best_area,
                        save.hard_best_sub,
                        save.hard_best_loop,
                        true,
                    ),
                );
                stat_name(&mut out, rx, &mut r, "kills");
                stat_val(&mut out, rx, &mut r, save.hard_best_kills.to_string());
                stat_name(&mut out, rx, &mut r, "runs");
                stat_val(&mut out, rx, &mut r, save.hard_runs.to_string());
                header(&mut out, rx, &mut r, "");
            }
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
            out
        }
        crate::MenuOverlay::Title => {
            let selected = world
                .get_resource::<SelectedCharacter>()
                .map(|s| s.0 as usize)
                .unwrap_or(1);
            let race = crate::state::menus::race_from_gml_id(selected);
            let Some(race) = race.filter(|r| *r != crate::data::RaceId::Random) else {
                return Vec::new();
            };
            let def = character_def(race);
            let textappear = world
                .get_resource::<MenuState>()
                .map(|m| m.textappear[0])
                .unwrap_or(0.0);
            let mut rows = Vec::new();
            if textappear != 2.0 {
                rows.push(MenuGuiText {
                    text: def.name.to_ascii_uppercase().to_string(),
                    gx: 0.0,
                    gy: 204.0,
                    color: GUI_WHITE,
                    px: 12.0,
                    centered: false,
                    middle_y: false,
                    right: false,
                });
                rows.push(MenuGuiText {
                    text: race_passive_text(race).to_string(),
                    gx: 8.0,
                    gy: 212.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: false,
                    middle_y: false,
                    right: false,
                });
                rows.push(MenuGuiText {
                    text: race_active_text(race).to_string(),
                    gx: 8.0,
                    gy: 220.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: false,
                    middle_y: false,
                    right: false,
                });
            }
            {
                let hardmode = world
                    .get_resource::<MenuState>()
                    .is_some_and(|m| m.hardmode_selected);
                if hardmode {
                    rows.push(MenuGuiText {
                        text: "HARD".to_string(),
                        gx: vw * 0.5,
                        gy: 120.0 - 45.0,
                        color: GUI_WHITE,
                        px: 10.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                    });
                }
            }

            // pod tooltip (`Menu/Draw_74`: `CharSelect.tooltip`, set only
            // while the mouse points at the pod or the gamepad selects
            // it). Headless has no pointer, so the row only shows while
            // the gamepad/hover path marks the cursor pod pointed.
            {
                let menu = world.get_resource::<MenuState>().cloned();
                let save = world
                    .get_resource::<crate::savedata_part::SaveData>()
                    .cloned();
                let roster = crate::state::menus::visible_roster(save.as_ref());
                let cursor = menu.as_ref().map(|m| m.title_cursor).unwrap_or(0);
                let pointed = menu.as_ref().is_some_and(|m| m.title_pod_pointed);
                let slot_h = 20.0;
                if pointed {
                    if let Some(pod_race) = roster.get(cursor) {
                        let weekly = menu.as_ref().is_some_and(|m| m.weekly_run_menu);
                        let can =
                            save.as_ref().is_some_and(|s| s.race_unlocked(*pod_race)) || weekly;
                        let tip = if can {
                            character_def(*pod_race)
                                .name
                                .to_ascii_uppercase()
                                .to_string()
                        } else {
                            crate::state::menus::unlock_hint_for_race(*pod_race)
                        };
                        if let Some(pos) =
                            char_pod_layout([vw, 240.0], roster.len(), slot_h).get(cursor)
                        {
                            rows.push(MenuGuiText {
                                text: tip,
                                gx: pos[0] + TITLE_POD_W * 0.5,
                                gy: pos[1] - 12.0,
                                color: GUI_WHITE,
                                px: 7.0,
                                centered: true,
                                middle_y: true,
                                right: false,
                            });
                        }
                    }
                }
                // GML `Menu.unlock_hint` box verbatim (touch locked-pod
                // hint): white `draw_text_nt` at `(gui_w/2, gui_h-30)`
                // over the `c_tooltip` roundrect for 90 steps
                // (`alarm[11]`), `pop` easing to 0. The sprite layer
                // cannot draw rects, so the text row carries the
                // position; the shell draws the box behind it.
                if let Some(menu) = menu.as_ref() {
                    if !menu.unlock_hint.is_empty() {
                        rows.push(MenuGuiText {
                            text: menu.unlock_hint.to_ascii_uppercase(),
                            gx: vw * 0.5,
                            gy: 240.0 - 30.0 + menu.unlock_hint_pop,
                            color: GUI_WHITE,
                            px: 7.0,
                            centered: true,
                            middle_y: true,
                            right: false,
                        });
                    }
                }
            }
            rows
        }
        crate::MenuOverlay::Mutation => {
            let hud = sync_hud_state(world);
            // GML `LevCont/Draw_0` law verbatim: the offer CENTER column
            // shows `sprLevelUpText` / `sprLevelUltraText` at
            // `(cx + appear, 48)` with subimage `appear > H*0.7 ? 0 : 2`
            // (white bigname while sliding in, `c_ultra` green rest) and
            // the `@s`-gray subtitle at `(cx + 1, 75 - appear)`. The port
            // has no `appear` anim resource, so the steady state
            // (`appear = 0`: art subimage 2, rest tint) is drawn.
            // Subtitles: ultra offers `INSTALL ULTRA UPDATE` (Robot) /
            // `PICK YOUR ULTRA MUTATION`; skill offers Robot
            // `INSTALL n UPDATES # DO NOT TURN OFF ROBOT`, else
            // `SELECT n MUTATIONS`. `n` counts the pending offer.
            let pending_n = world
                .get_resource::<PendingMutation>()
                .map(|p| p.choices.len());
            let pending_ultra_n = world
                .get_resource::<PendingUltra>()
                .map(|u| u.choices.len());
            let is_ultra = pending_n.is_none() && pending_ultra_n.is_some();
            let n = pending_n.or(pending_ultra_n).unwrap_or(0).max(1);
            let is_robot = world
                .get_resource::<SelectedCharacter>()
                .is_some_and(|s| s.0 == crate::data::RaceId::Robot);
            let (title, subtitle, extra) = if is_ultra {
                if is_robot {
                    (
                        "LEVEL ULTRA",
                        "@sINSTALL @gULTRA@s UPDATE".to_string(),
                        None,
                    )
                } else {
                    (
                        "LEVEL ULTRA",
                        "@sPICK YOUR @gULTRA@s MUTATION".to_string(),
                        None,
                    )
                }
            } else if is_robot {
                (
                    "LEVEL UP",
                    format!("@sINSTALL {n} UPDATES@s"),
                    Some("@sDO NOT TURN OFF ROBOT".to_string()),
                )
            } else {
                ("LEVEL UP", format!("@sSELECT {n} MUTATIONS"), None)
            };
            let accent = if is_ultra { GUI_ULTRA } else { GUI_GREEN };
            let mut out = vec![
                gui_center(title, cx, 48.0, accent),
                gui_center(subtitle, cx, 75.0, GUI_CREAM),
            ];
            if let Some(extra) = extra {
                out.push(gui_center(extra, cx, 87.0, GUI_CREAM));
            }
            // GML `SkillIcon/Draw_0` law: the SELECTED card's box is ONE
            // centered-middle text at `(w/2, H-61-selected)` = (cx, 179):
            // `"@wName#@sDesc@s"`, with the Throne Butt special (per-race
            // `Races:<race>:TB` text, or `Name - TB` lines per race when
            // players hold mixed races). `mutation_choices` already
            // carries `Name - Desc`; only the first ` - ` splits (descs
            // contain `#` line breaks which the backend renders).
            let selected = world
                .get_resource::<MenuState>()
                .and_then(|m| m.mutation_selected);
            let sel_id = selected.and_then(|i| hud.mutation_choice_ids.get(i).copied());
            let sel_text = selected.and_then(|i| hud.mutation_choices.get(i));
            if let Some(sel) = sel_text {
                let (_, name, desc) = mutation_choice_parts(sel);
                let race_tb = |race: crate::data::RaceId| -> &'static str {
                    match race {
                        crate::data::RaceId::Fish => "WATER BOOST",
                        crate::data::RaceId::Crystal => "TELEPORTATION",
                        crate::data::RaceId::Eyes => "STRONGER TELEKINESIS",
                        crate::data::RaceId::Melting => "BIGGER CORPSE EXPLOSIONS",
                        crate::data::RaceId::Plant => "SNARE FINISHES ENEMIES#IN UNDER 33% @rHP",
                        crate::data::RaceId::Venuz => "BRRRAP",
                        crate::data::RaceId::Steroids => "DUAL FIRING MAY GIVE AMMO SOMETIMES",
                        crate::data::RaceId::Robot => "BETTER GUN NUTRITION",
                        crate::data::RaceId::Chicken => "THROWN WEAPONS CAN PIERCE ENEMIES",
                        crate::data::RaceId::Rebel => "HIGHER ALLY RATE OF FIRE",
                        crate::data::RaceId::Horror => {
                            "GAIN @rHP@s WHEN USING#@gBEAM@s FOR A LONG TIME"
                        }
                        crate::data::RaceId::Rogue => "BIGGER PORTAL STRIKES",
                        crate::data::RaceId::BigDog => "FASTER ROCKETS",
                        crate::data::RaceId::Skeleton => "BETTER ODDS",
                        crate::data::RaceId::Frog => "TOXIC SPREADS FASTER",
                        crate::data::RaceId::Cuz => "CRY REFILLS FULLY WHEN HIT",
                        crate::data::RaceId::Random => "???",
                    }
                };
                // Throne Butt special (`box_text`): multi-race parties
                // return early above with one row per race; single-race
                // falls through to the shared box split below.
                let box_text: String = if sel_id
                    == Some(crate::hud::mutation_skill_index(
                        crate::data::MutationId::ThroneButt,
                    )) {
                    // GML multirace check over live players.
                    let mut races: Vec<crate::data::RaceId> = world
                        .query::<&RaceState>()
                        .iter(world)
                        .map(|rs| rs.race)
                        .collect();
                    races.sort_by_key(|r| *r as u8);
                    races.dedup();
                    if races.len() > 1 {
                        // Multi-race Throne Butt: one line per race at
                        // the block-middle spacing (same law as the box
                        // split below, inlined for the per-line color
                        // tags).
                        let lines: Vec<String> = races
                            .iter()
                            .map(|r| {
                                format!(
                                    "@w{} - {}@s",
                                    character_def(*r).name.to_ascii_uppercase(),
                                    race_tb(*r)
                                )
                            })
                            .collect();
                        let n = lines.len().max(1) as f32;
                        for (i, line) in lines.iter().enumerate() {
                            out.push(MenuGuiText {
                                text: line.clone(),
                                gx: cx,
                                gy: 179.0 + (i as f32 - (n - 1.0) * 0.5) * 8.0,
                                color: GUI_WHITE,
                                px: 7.0,
                                centered: true,
                                middle_y: true,
                                right: false,
                            });
                        }
                        return out;
                    }
                    let tb = races.first().map(|r| race_tb(*r)).unwrap_or(race_tb(
                        world
                            .get_resource::<SelectedCharacter>()
                            .map(|s| s.0)
                            .unwrap_or(crate::data::RaceId::Fish),
                    ));
                    format!("@w{}@s", tb)
                } else if desc.is_empty() {
                    format!("@w{}", name.to_ascii_uppercase())
                } else {
                    format!("@w{}#@s{}@s", name.to_ascii_uppercase(), desc)
                };
                // GML draws the `"@wName#@sDesc@s"` box as ONE
                // centered-middle block: `middle_y` centers on the whole
                // block, so split per line around the block middle
                // (`gy + (i - (n-1)/2) * 8`).
                let box_lines: Vec<&str> = box_text
                    .split('#')
                    .flat_map(|s| s.split('\n'))
                    .collect();
                let n = box_lines.len().max(1) as f32;
                for (i, line) in box_lines.iter().enumerate() {
                    // `gui_multiline` cannot carry per-block middle
                    // centering, so the split stays inline here.
                    out.push(MenuGuiText {
                        text: line.to_string(),
                        gx: cx,
                        gy: 179.0 + (i as f32 - (n - 1.0) * 0.5) * 8.0,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                    });
                }
            }
            out
        }
        crate::MenuOverlay::GameOver => {
            // GML `GameOver/Draw_0` text layer verbatim: struggle line
            // top-anchored at (vw/2, view_top+48) (no offset); the
            // roadmap lives at (_x-48, _y-offsety) with the area/kill
            // strings at (drawy-14) riding it; `KILLED BY` (or win
            // `COMPLETION TIME` + clock) centered at (_x+86,
            // _y-offsety-25/-10); MENU/RETRY are the two `PauseButton`s
            // at ystart+offsety (178/210). Splat frames + roadmap prefix
            // ride the anim state (`go_splat`, `go_death_pos`).
            let run = world.get_resource::<crate::comps_a::Run>();
            let (offsety, _death_pos) = world
                .get_resource::<MenuState>()
                .map(|m| (m.go_offsety, m.go_death_pos))
                .unwrap_or((0.0, 0.0));
            let mut out = Vec::new();
            if let Some(run) = run {
                let max_sub = area_max_subarea(run.area);
                // GML `GameOver/Create_0` verbatim: base text from the
                // area/loop position, then a win only overrides to
                // `THE STRUGGLE IS OVER` on the HQ final. The
                // `Cinematic`-gated `YOU REACHED THE NUCLEAR THRONE`
                // has no port counterpart (no Cinematic entity), so a
                // non-HQ win keeps its base text like GML without one.
                let palace_final = run.area == AreaId::Palace && run.floor_in_area == max_sub;
                let hq_final = run.area == AreaId::HQ && run.floor_in_area == max_sub;
                let mut text = if palace_final {
                    "YOU ALMOST REACHED THE NUCLEAR THRONE"
                } else if run.loop_count > 0 {
                    "THE STRUGGLE CONTINUES"
                } else {
                    "YOU DID NOT REACH THE NUCLEAR THRONE"
                };
                if run.won && hq_final {
                    text = "THE STRUGGLE IS OVER";
                }
                out.push(MenuGuiText {
                    text: text.to_string(),
                    gx: cx,
                    gy: 48.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: true,
                    // GML draws the struggle `draw_set_valign(fa_top)` at
                    // `view_yview + 48` (top-anchored, not middle).
                    middle_y: false,
                    right: false,
                });
                // GML `scrDrawRoadmap:23-24` verbatim: the area/kill
                // strings ride the roadmap at `(drawx-60, drawy-14)` /
                // `(drawx+23, drawy-14)` with `drawx = cx-48`.
                out.push(MenuGuiText {
                    text: run_area_string(run),
                    gx: cx - 108.0,
                    gy: 106.0 - offsety,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: false,
                    middle_y: true,
                    right: false,
                });
                out.push(MenuGuiText {
                    text: run.total_kills.to_string(),
                    gx: cx - 25.0,
                    gy: 106.0 - offsety,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: false,
                    middle_y: true,
                    right: false,
                });
                if run.won {
                    out.push(MenuGuiText {
                        text: "COMPLETION TIME".to_string(),
                        gx: cx + 86.0,
                        gy: 95.0 - offsety,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                    });
                    out.push(MenuGuiText {
                        text: run_timer_string(run.tottimer),
                        gx: cx + 86.0,
                        gy: 110.0 - offsety,
                        color: GUI_GRAY,
                        px: 7.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                    });
                } else {
                    out.push(MenuGuiText {
                        text: "KILLED BY".to_string(),
                        gx: cx + 86.0,
                        gy: 95.0 - offsety,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                    });
                }
            } else {
                out.push(gui_center("GAME OVER", cx, 100.0, GUI_WHITE));
            }
            // GML `PauseButton`s ride `ystart + offsety` at
            // `(center, center+58)` and `(center, center+90)` with
            // `appear = 3 + image` (image 0 MENU appear 3, image 1 RETRY
            // appear 4). `PauseButton/Draw_0` draws at `_dy = y + appear`
            // (a few px low while appearing — never parked off-screen),
            // so the buttons are clickable from the first frame. Event-run
            // swaps (`sprGameOverResult`, weekly keeps button 0 as RETRY
            // else destroys it) need the event systems the port lacks —
            // both buttons draw at once.
            let appear = world
                .get_resource::<MenuState>()
                .map(|m| m.go_appear.max(0.0))
                .unwrap_or(0.0);
            out.push(gui_button(
                "MENU",
                cx,
                120.0 + 58.0 + offsety + appear,
                GUI_MID,
            ));
            out.push(gui_button(
                "RETRY",
                cx,
                120.0 + 90.0 + offsety + appear,
                GUI_MID,
            ));
            out
        }
        crate::MenuOverlay::Pause => {
            // GML pause layer verbatim (`UberCont/Draw_0` paused branch
            // + `scrMakePauseButtons` + `PauseButton/Draw_0/Other_10`):
            // frozen `pausespr` screenshot (shell-owned, not drawn here)
            // + 0.7 black scrim (shell-owned) + bigname `PAUSED`
            // centered-middle at `(view_center + 1, 52 + 1 - yoff)`
            // (`yoff = 4` on event/hardmode runs) + corner `sprCharSplat`
            // pair (sprite layer) + full roadmap (sprite layer) +
            // `PauseButton`s at MENU `(left+45, bottom-64, appear 1)`,
            // RETRY `(left+60, bottom-32, appear 2)`, SETTINGS
            // `(right-68, top, appear 3)`, CONTINUE `(right-78, bottom,
            // appear 3)`. Buttons draw the bigname label at scale 0.65
            // at `y + appear` while `appear < 2`, else the
            // `sprPauseButton` art (`appear` ticks -1/step, `wait` 3
            // steps gates clicks). The port draws the steady state
            // (`appear = 0`); the appear lift is documented here so the
            // shell can animate it. RETRY is suppressed on daily runs
            let hardmode = world
                .get_resource::<crate::comps_a::Run>()
                .is_some_and(|r| r.hardmode);
            let yoff = if hardmode { 4.0 } else { 0.0 };
            let confirm = world
                .get_resource::<MenuState>()
                .and_then(|m| m.pause_confirm);
            if let Some(confirm) = confirm {
                let right = if confirm == 0 { "QUIT" } else { "RETRY" };
                let right_color = if confirm == 0 { GUI_RED2 } else { GUI_GREEN };
                vec![
                    gui_button("BACK", 52.0, 192.0, GUI_MID),
                    gui_button(right, vw - 52.0, 192.0, right_color),
                ]
            } else {
                vec![
                    MenuGuiText {
                        text: "PAUSED".to_string(),
                        gx: cx + 1.0,
                        gy: 52.0 + 1.0 - yoff,
                        color: GUI_WHITE,
                        px: 10.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                    },
                    gui_button("MENU", 45.0, 176.0, GUI_MID),
                    gui_button("RETRY", 60.0, 208.0, GUI_MID),
                    gui_button("SETTINGS", vw - 68.0, 176.0, GUI_MID),
                    gui_button("CONTINUE", vw - 78.0, 208.0, GUI_MID),
                ]
            }
        }
        crate::MenuOverlay::Settings => settings_gui_texts(world, vw),
        crate::MenuOverlay::Credits => {
            // GML `Credits/Draw_64` verbatim: the current section body
            // is ONE centered-middle `draw_text_nt` at `(gui_w/2,
            // gui_h/2)` — GML joins the section with `"\n@s"`; the tag
            // backend splits `#` the same way. Tall sections pan via
            // `MenuState::credits_scroll` (`_py += scroll - height +
            // gui_h * 0.6`, top-anchored). When a `Logo` instance owns
            // the credits (end-of-credits handoff) nothing draws.
            let section = world
                .get_resource::<MenuState>()
                .map(|m| m.credits_section)
                .unwrap_or(0);
            let scroll = world
                .get_resource::<MenuState>()
                .map(|m| m.credits_scroll)
                .unwrap_or(0.0);
            let n = credit_section_count();
            // GML draws the section body as ONE centered-middle block
            // (lines at `gy + i * 12` around the block middle); split
            // per line here since the shell renders `.single_line()`.
            let body = CREDIT_SECTIONS[section % n].join("\n@s");
            // GML scroll law (`Credits/Step_0`): only tall sections
            // (`height > gui_h - 36`) set `largetext` and pan.
            let rows = CREDIT_SECTIONS[section % n].len() as f32;
            let tall = rows * 12.0 > 240.0 - 36.0;
            let gy = if tall {
                120.0 + scroll - rows * 12.0 + 240.0 * 0.6
            } else {
                120.0
            };
            let lines: Vec<&str> = body.split('\n').collect();
            let nlines = lines.len().max(1) as f32;
            lines
                .into_iter()
                .enumerate()
                .map(|(i, line)| MenuGuiText {
                    text: format!("@s{line}"),
                    gx: cx,
                    gy: gy + (i as f32 - (nlines - 1.0) * 0.5) * 12.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: true,
                    middle_y: true,
                    right: false,
                })
                .collect()
        }
    }
}

/// Settings toggle row: label left (cream) + ON/OFF value (gray).
fn push_toggle(out: &mut Vec<MenuGuiText>, label: &str, y: f32, on: bool) {
    out.push(gui_body(label, 80.0, y, GUI_CREAM));
    out.push(gui_body(if on { "ON" } else { "OFF" }, 200.0, y, GUI_GRAY));
}

// ---------------------------------------------------------------------------
// Settings hot rows: the single source of truth for what each settings
// row DOES, shared by mouse hit-testing (`settings_click_action`) and
// keyboard nav (`tick_settings_nav` in `menus.rs`). `gy` mirrors
// `settings_gui_texts` exactly (same literals); `cx`/`hw` is the mouse
// hit box in GUI px (centered buttons at `vw/2`, value cells at 200,
// toggle rows spanning 40..240).
// ---------------------------------------------------------------------------

/// Player color presets cycled by the COLOR page button (bevy verbatim).
pub const COLOR_PRESETS: [&str; 5] = ["FF0000", "00FF00", "0000FF", "", "FF00FF"];

/// Volume channel with absolute `Set*Vol` steppers (bevy ±0.1 buttons).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolumeChannel {
    Master,
    Music,
    Ambience,
    Sfx,
}

/// One actionable settings row.
#[derive(Clone, Copy, Debug)]
pub struct SettingHotRow {
    pub gy: f32,
    pub cx: f32,
    pub hw: f32,
    pub op: SettingHotOp,
}

/// What activating the row does (`dir`: click half / arrow key; ignored
/// by toggles, buttons and language rows).
#[derive(Clone, Copy, Debug)]
pub enum SettingHotOp {
    Category(u8),
    Toggle(&'static str),
    Slider(&'static str),
    Cycle(&'static str),
    Volume(VolumeChannel),
    Language(&'static str),
    Back,
    Credits,
    ColorCycle,
    ResetOptions,
    EraseProgress,
}

/// Actionable rows for a settings page in visual order (headers and
/// static display rows excluded). `vw` is the live GUI width (only the
/// centered-button `cx` depends on it; pass anything for keyboard nav).
pub fn settings_hot_rows(page: u8, vw: f32) -> Vec<SettingHotRow> {
    let cx = vw * 0.5;
    // (gy, cx, hw, op)
    let btn = |gy: f32, op: SettingHotOp| SettingHotRow {
        gy,
        cx,
        hw: 100.0,
        op,
    };
    let tog = |gy: f32, op: SettingHotOp| SettingHotRow {
        gy,
        cx: 140.0,
        hw: 100.0,
        op,
    };
    let val = |gy: f32, op: SettingHotOp| SettingHotRow {
        gy,
        cx: 200.0,
        hw: 60.0,
        op,
    };
    match page {
        0 => vec![
            btn(72.0, SettingHotOp::Category(1)),
            btn(96.0, SettingHotOp::Category(2)),
            btn(120.0, SettingHotOp::Category(3)),
            btn(144.0, SettingHotOp::Category(4)),
            btn(168.0, SettingHotOp::Category(5)),
            btn(220.0, SettingHotOp::Back),
        ],
        1 => vec![
            val(56.0, SettingHotOp::Volume(VolumeChannel::Master)),
            val(76.0, SettingHotOp::Volume(VolumeChannel::Music)),
            val(96.0, SettingHotOp::Volume(VolumeChannel::Ambience)),
            val(116.0, SettingHotOp::Volume(VolumeChannel::Sfx)),
            tog(140.0, SettingHotOp::Toggle("volume_3dsound")),
            btn(200.0, SettingHotOp::Back),
        ],
        2 => vec![
            val(48.0, SettingHotOp::Cycle("crosshair")),
            val(66.0, SettingHotOp::Cycle("sideart")),
            val(84.0, SettingHotOp::Slider("screenshake")),
            val(102.0, SettingHotOp::Slider("freezeframes")),
            tog(120.0, SettingHotOp::Toggle("bloom")),
            tog(138.0, SettingHotOp::Toggle("particles")),
            tog(156.0, SettingHotOp::Toggle("show_hud")),
            val(174.0, SettingHotOp::Cycle("pixel_mode")),
            btn(192.0, SettingHotOp::Category(9)),
            btn(220.0, SettingHotOp::Back),
        ],
        9 => vec![
            tog(56.0, SettingHotOp::Toggle("widescreen")),
            tog(76.0, SettingHotOp::Toggle("fullscreen")),
            tog(96.0, SettingHotOp::Toggle("vsync")),
            btn(200.0, SettingHotOp::Back),
        ],
        3 => vec![
            tog(48.0, SettingHotOp::Toggle("boss_intros")),
            tog(62.0, SettingHotOp::Toggle("show_tutorial")),
            tog(76.0, SettingHotOp::Toggle("show_timer")),
            tog(90.0, SettingHotOp::Toggle("show_area")),
            tog(104.0, SettingHotOp::Toggle("pause_button")),
            tog(118.0, SettingHotOp::Toggle("achievements_popup")),
            tog(132.0, SettingHotOp::Toggle("auto_pause")),
            btn(146.0, SettingHotOp::Credits),
            btn(162.0, SettingHotOp::Category(10)),
            btn(178.0, SettingHotOp::Category(11)),
            btn(194.0, SettingHotOp::Category(12)),
            btn(228.0, SettingHotOp::Back),
        ],
        10 => vec![
            btn(88.0, SettingHotOp::Category(11)),
            btn(200.0, SettingHotOp::Back),
        ],
        11 => vec![
            btn(90.0, SettingHotOp::ColorCycle),
            btn(200.0, SettingHotOp::Back),
        ],
        12 => vec![
            btn(80.0, SettingHotOp::ResetOptions),
            btn(110.0, SettingHotOp::EraseProgress),
            btn(200.0, SettingHotOp::Back),
        ],
        4 => vec![
            tog(48.0, SettingHotOp::Toggle("gamepad_enabled")),
            tog(62.0, SettingHotOp::Toggle("aim_assist")),
            tog(76.0, SettingHotOp::Toggle("auto_aim")),
            tog(90.0, SettingHotOp::Toggle("volume_controls")),
            tog(104.0, SettingHotOp::Toggle("split_fire")),
            tog(118.0, SettingHotOp::Toggle("fixed_sight")),
            val(132.0, SettingHotOp::Cycle("gamepad_type")),
            val(146.0, SettingHotOp::Slider("controls_scale")),
            btn(160.0, SettingHotOp::Category(13)),
            btn(176.0, SettingHotOp::Category(15)),
            btn(192.0, SettingHotOp::Category(16)),
            btn(228.0, SettingHotOp::Back),
        ],
        13 => vec![btn(200.0, SettingHotOp::Back)],
        15 => {
            let mut rows: Vec<SettingHotRow> = (0..8)
                .map(|i| {
                    let key: &'static str = match i {
                        0 => "cprefs_0",
                        1 => "cprefs_1",
                        2 => "cprefs_2",
                        3 => "cprefs_3",
                        4 => "cprefs_4",
                        5 => "cprefs_5",
                        6 => "cprefs_6",
                        _ => "cprefs_7",
                    };
                    tog(48.0 + i as f32 * 18.0, SettingHotOp::Toggle(key))
                })
                .collect();
            rows.push(btn(200.0, SettingHotOp::Back));
            rows
        }
        16 => vec![btn(200.0, SettingHotOp::Back)],
        5 => {
            let mut rows: Vec<SettingHotRow> = crate::state::menus::AVAILABLE_LANGUAGES
                .iter()
                .enumerate()
                .map(|(i, lang)| SettingHotRow {
                    gy: 60.0 + i as f32 * 20.0,
                    cx,
                    hw: 60.0,
                    op: SettingHotOp::Language(lang),
                })
                .collect();
            rows.push(btn(200.0, SettingHotOp::Back));
            rows
        }
        _ => vec![],
    }
}

/// Resolve one hot-row activation into a [`UiAction`]. `dir` is the
/// stepper direction (-1/+1; 0 = keyboard Enter, treated as +1).
/// Reads live values from `SaveData` for the ±0.1 steppers.
pub fn settings_hot_action(world: &mut World, page: u8, idx: usize, dir: i8) -> Option<UiAction> {
    use crate::savedata_part::SaveData;
    let rows = settings_hot_rows(page, 320.0);
    let row = rows.get(idx)?;
    let step = if dir >= 0 { 1.0 } else { -1.0 };
    match row.op {
        SettingHotOp::Category(c) => Some(UiAction::SettingsCategory(c)),
        SettingHotOp::Toggle(k) => Some(UiAction::SettingToggle(k.to_string())),
        SettingHotOp::Slider(k) => {
            let v = world.get_resource::<SaveData>().map(|s| match k {
                "screenshake" => s.settings.screenshake,
                "freezeframes" => s.settings.freezeframes,
                "controls_scale" => s.settings.controls_scale,
                _ => 0.0,
            });
            v.map(|v| UiAction::SettingSlider {
                key: k.to_string(),
                value: v + step * 0.1,
            })
        }
        SettingHotOp::Cycle(k) => Some(UiAction::SettingCycle {
            key: k.to_string(),
            dir: if dir == 0 { 1 } else { dir },
        }),
        SettingHotOp::Volume(ch) => {
            let v = world.get_resource::<SaveData>().map(|s| match ch {
                VolumeChannel::Master => s.settings.master_volume,
                VolumeChannel::Music => s.settings.music_volume,
                VolumeChannel::Ambience => s.settings.ambience_volume,
                VolumeChannel::Sfx => s.settings.sfx_volume,
            });
            v.map(|v| {
                let nv = (v + step * 0.1).clamp(0.0, 1.0);
                match ch {
                    VolumeChannel::Master => UiAction::SetMasterVol(nv),
                    VolumeChannel::Music => UiAction::SetMusicVol(nv),
                    VolumeChannel::Ambience => UiAction::SetAmbienceVol(nv),
                    VolumeChannel::Sfx => UiAction::SetSfxVol(nv),
                }
            })
        }
        SettingHotOp::Language(code) => Some(UiAction::SetLanguage(code.to_string())),
        SettingHotOp::Back => Some(UiAction::SettingsBack),
        SettingHotOp::Credits => Some(UiAction::SettingViewCredits),
        SettingHotOp::ColorCycle => {
            let cur = world
                .get_resource::<SaveData>()
                .map(|s| s.settings.player_color_hex.clone())
                .unwrap_or_default();
            let i = COLOR_PRESETS.iter().position(|p| *p == cur).unwrap_or(3);
            Some(UiAction::SettingInput {
                key: "player_color_hex".to_string(),
                value: COLOR_PRESETS[(i + 1) % COLOR_PRESETS.len()].to_string(),
            })
        }
        SettingHotOp::ResetOptions => Some(UiAction::SettingResetOptions),
        SettingHotOp::EraseProgress => Some(UiAction::SettingEraseProgress),
    }
}

/// Mouse hit-test for a settings page: GUI-px click → [`UiAction`].
/// Stepper direction comes from the click half (left = -1, right = +1).
pub fn settings_click_action(
    world: &mut World,
    page: u8,
    gx: f32,
    gy: f32,
    vw: f32,
) -> Option<UiAction> {
    let rows = settings_hot_rows(page, vw);
    // Tight vertical band (rows sit 14px apart on dense pages).
    const HH: f32 = 7.0;
    let (idx, row) = rows
        .iter()
        .enumerate()
        .find(|(_, r)| (gy - r.gy).abs() <= HH && (gx - r.cx).abs() <= r.hw)?;
    let dir = match row.op {
        SettingHotOp::Slider(_) | SettingHotOp::Cycle(_) | SettingHotOp::Volume(_) => {
            if gx >= row.cx { 1 } else { -1 }
        }
        _ => 0,
    };
    settings_hot_action(world, page, idx, dir)
}

/// Settings pages (bevy `settings_ui` arms verbatim: headers, rows,
/// buttons; dynamic values read from [`SaveData`](crate::savedata_part::SaveData)
/// + [`MenuState`](crate::state::menus::MenuState)).
fn settings_gui_texts(world: &mut World, vw: f32) -> Vec<MenuGuiText> {
    use crate::savedata_part::SaveData;
    let cx = vw * 0.5;
    let page = world
        .get_resource::<MenuState>()
        .map(|m| m.settings_page)
        .unwrap_or(0);
    let save = world
        .get_resource::<SaveData>()
        .cloned()
        .unwrap_or_default();
    let s = &save.settings;
    let mut out = Vec::new();
    match page {
        0 => {
            out.push(gui_center("SETTINGS", cx, 24.0, GUI_MID));
            for (i, (label, _)) in [
                ("AUDIO", 1u8),
                ("VIDEO", 2),
                ("GAME", 3),
                ("CONTROLS", 4),
                ("LANGUAGE", 5),
            ]
            .iter()
            .enumerate()
            {
                out.push(gui_button(*label, cx, 72.0 + i as f32 * 24.0, GUI_MID));
            }
            out.push(gui_button("BACK", cx, 220.0, GUI_GRAY));
        }
        1 => {
            out.push(gui_center("AUDIO", cx, 24.0, GUI_MID));
            for (gy, label, val) in [
                (56.0, "MASTER VOLUME", s.master_volume),
                (76.0, "MUSIC VOLUME", s.music_volume),
                (96.0, "AMBIENCE VOLUME", s.ambience_volume),
                (116.0, "EFFECTS VOLUME", s.sfx_volume),
            ] {
                out.push(gui_body(label, 80.0, gy, GUI_CREAM));
                out.push(gui_body(
                    format!("{:.0}%", val * 100.0),
                    200.0,
                    gy,
                    GUI_GRAY,
                ));
            }
            out.push(gui_body("3D SOUND", 80.0, 140.0, GUI_CREAM));
            out.push(gui_body(
                if s.volume_3dsound { "ON" } else { "OFF" },
                200.0,
                140.0,
                GUI_GRAY,
            ));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        2 => {
            out.push(gui_center("VIDEO", cx, 24.0, GUI_MID));
            let mut y = 48.0;
            out.push(gui_body("CROSSHAIR", 80.0, y, GUI_CREAM));
            out.push(gui_body(
                format!("< {} >", s.crosshair + 1),
                200.0,
                y,
                GUI_GRAY,
            ));
            y += 18.0;
            out.push(gui_body("SIDE ART", 80.0, y, GUI_CREAM));
            out.push(gui_body(format!("< {} >", s.sideart), 200.0, y, GUI_GRAY));
            y += 18.0;
            for (label, val) in [
                ("SCREENSHAKE", s.screenshake),
                ("FREEZE FRAMES", s.freezeframes),
            ] {
                out.push(gui_body(label, 80.0, y, GUI_CREAM));
                out.push(gui_body(
                    format!("{:.0}%", (val * 100.0).clamp(0.0, 200.0)),
                    200.0,
                    y,
                    GUI_GRAY,
                ));
                y += 18.0;
            }
            push_toggle(&mut out, "BLOOM", y, s.bloom);
            y += 18.0;
            push_toggle(&mut out, "PARTICLES", y, s.particles);
            y += 18.0;
            push_toggle(&mut out, "HIDE HUD", y, !s.show_hud);
            y += 18.0;
            out.push(gui_body("PIXEL MODE", 80.0, y, GUI_CREAM));
            out.push(gui_body(
                format!("< {} >", s.pixel_mode),
                200.0,
                y,
                GUI_GRAY,
            ));
            y += 18.0;
            out.push(gui_button("DISPLAY SETTINGS", cx, y, GUI_MID));
            // DISPLAY lands at y=192; BACK at 200 would overlap its
            // 10px button box, so sit BACK at 220 like the OPTIONS page.
            out.push(gui_button("BACK", cx, 220.0, GUI_GRAY));
        }
        9 => {
            out.push(gui_center("DISPLAY", cx, 24.0, GUI_MID));
            let mut y = 56.0;
            push_toggle(&mut out, "WIDESCREEN", y, s.widescreen);
            y += 20.0;
            push_toggle(&mut out, "FULLSCREEN", y, s.fullscreen);
            y += 20.0;
            push_toggle(&mut out, "VSYNC", y, s.vsync);
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        3 => {
            // GML `Game` category verbatim (`Other_20.hml:218`): boss
            // intros, tutorial, timer, area, pause-button (mobile-only
            // in GML; shown here — desktop shells ignore it), the
            // `ACHIEVEMENT#POPUPS` two-line switch, auto-pause
            // (desktop-only in GML), VIEW CREDITS, then PROFILE (the
            // COLOR/DATA leaves live under PROFILE in GML; the port
            // surfaces all three here so they stay reachable without
            // the text-entry pages).
            out.push(gui_center("GAME", cx, 24.0, GUI_MID));
            let mut y = 48.0;
            for (label, on) in [
                ("BOSS INTROS", s.boss_intros),
                ("PLAY TUTORIAL", s.show_tutorial),
                ("SHOW TIMER", s.show_timer),
                ("SHOW AREA", s.show_area),
                ("PAUSE BUTTON", s.pause_button),
                ("ACHIEVEMENT#POPUPS", s.achievements_popup),
                ("AUTO PAUSE", s.auto_pause),
            ] {
                push_toggle(&mut out, label, y, on);
                y += 14.0;
            }
            out.push(gui_button("VIEW CREDITS", cx, y, GUI_MID));
            y += 16.0;
            out.push(gui_button("PROFILE", cx, y, GUI_MID));
            y += 16.0;
            out.push(gui_button("COLOR", cx, y, GUI_MID));
            y += 16.0;
            out.push(gui_button("DATA", cx, y, GUI_MID));
            out.push(gui_button("BACK", cx, 228.0, GUI_GRAY));
        }
        4 => {
            // GML `Controls` category verbatim (`Other_20.gml:520`):
            // GAMEPAD, GAMEPAD STYLE (XBOX ONE for XBONE), the
            // mobile-only AIM ASSIST / FULL AUTOAIM / VOLUME CONTROLS /
            // SPLIT AIM & FIRE / FIXED SIGHT / SIZE SCALE rows (shown
            // here; desktop shells ignore them), REMAP CONTROLS (with
            // the `(GAMEPAD)`/`(KEYBOARD)` suffix the port cannot know),
            // CHARACTER PREFERENCES + EXPERIMENTAL OPTIONS (mobile-only
            // in GML). Names use the GML loc defaults.
            out.push(gui_center("CONTROLS", cx, 24.0, GUI_MID));
            let mut y = 48.0;
            push_toggle(&mut out, "GAMEPAD", y, s.gamepad_enabled);
            y += 14.0;
            out.push(gui_body("GAMEPAD STYLE", 80.0, y, GUI_CREAM));
            let names = ["XBOX ONE", "PS4", "Switch", "SteamDeck"];
            out.push(gui_body(
                format!("< {} >", names[(s.gamepad_type as usize) % names.len()]),
                200.0,
                y,
                GUI_GRAY,
            ));
            y += 14.0;
            for (label, on) in [
                ("AIM ASSIST", s.aim_assist),
                ("FULL AUTOAIM", s.auto_aim),
                ("VOLUME CONTROLS", s.volume_controls),
                ("SPLIT AIM & FIRE", s.split_fire),
                ("FIXED SIGHT", s.fixed_sight),
            ] {
                push_toggle(&mut out, label, y, on);
                y += 14.0;
            }
            out.push(gui_body("SIZE SCALE", 80.0, y, GUI_CREAM));
            out.push(gui_body(
                format!("{:.0}%", s.controls_scale * 100.0),
                200.0,
                y,
                GUI_GRAY,
            ));
            y += 14.0;
            out.push(gui_button("REMAP CONTROLS", cx, y, GUI_MID));
            y += 16.0;
            out.push(gui_button("CHARACTER PREFERENCES", cx, y, GUI_MID));
            y += 16.0;
            out.push(gui_button("EXPERIMENTAL OPTIONS", cx, y, GUI_MID));
            out.push(gui_button("BACK", cx, 228.0, GUI_GRAY));
        }
        10 => {
            out.push(gui_center("PROFILE", cx, 24.0, GUI_MID));
            out.push(gui_body("PROFILE NAME", 80.0, 48.0, GUI_CREAM));
            out.push(gui_body(
                if s.profile_name.is_empty() {
                    "NONE".to_string()
                } else {
                    s.profile_name.clone()
                },
                200.0,
                48.0,
                GUI_GRAY,
            ));
            out.push(gui_body("COLOR", 80.0, 68.0, GUI_CREAM));
            out.push(gui_body(
                if s.player_color_hex.is_empty() {
                    "DEFAULT".to_string()
                } else {
                    s.player_color_hex.clone()
                },
                200.0,
                68.0,
                GUI_GRAY,
            ));
            out.push(gui_button("COLOR", cx, 88.0, GUI_MID));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        11 => {
            out.push(gui_center("COLOR", cx, 24.0, GUI_MID));
            out.push(gui_center(
                format!(
                    "HEX: {}",
                    if s.player_color_hex.is_empty() {
                        "DEFAULT".to_string()
                    } else {
                        s.player_color_hex.clone()
                    }
                ),
                cx,
                60.0,
                GUI_CREAM,
            ));
            out.push(gui_button("CYCLE COLOR", cx, 90.0, GUI_MID));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        12 => {
            out.push(gui_center("DATA", cx, 24.0, GUI_MID));
            out.push(gui_button("RESET OPTIONS", cx, 80.0, GUI_CREAM));
            out.push(gui_button("ERASE PROGRESS", cx, 110.0, GUI_RED2));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        13 => {
            out.push(gui_center("REMAP", cx, 24.0, GUI_MID));
            let mut y = 60.0;
            for label in ["FIRE", "ACTIVE", "SWAP", "PICK"] {
                out.push(gui_center(label, cx, y, GUI_CREAM));
                y += 18.0;
            }
            out.push(gui_center("PRESS ANY KEY - WIP", cx, y, GUI_GRAY));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        15 => {
            out.push(gui_center("CHAR PREFS", cx, 24.0, GUI_MID));
            let prefs = [
                s.cprefs_eyes,
                s.cprefs_melting,
                s.cprefs_plant,
                s.cprefs_yv,
                s.cprefs_steroids,
                s.cprefs_horror,
                s.cprefs_rogue,
                s.cprefs_skeleton,
            ];
            let labels = [
                "EYES", "MELTING", "PLANT", "VENUZ", "STER", "HORROR", "ROGUE", "SKELETON",
            ];
            let mut y = 48.0;
            for (label, on) in labels.iter().zip(prefs) {
                push_toggle(&mut out, label, y, on);
                y += 18.0;
            }
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        16 => {
            out.push(gui_center("EXPERIMENTAL", cx, 24.0, GUI_MID));
            out.push(gui_center("KEYBOARD MODE - WIP", cx, 80.0, GUI_GRAY));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        5 => {
            out.push(gui_center("LANGUAGE", cx, 24.0, GUI_MID));
            let mut y = 60.0;
            for lang in crate::state::menus::AVAILABLE_LANGUAGES {
                let current = lang == s.language;
                out.push(MenuGuiText {
                    text: lang.to_ascii_uppercase().to_string(),
                    gx: cx,
                    gy: y,
                    color: if current { GUI_WHITE } else { GUI_MID },
                    px: 10.0,
                    centered: true,
                    middle_y: false,
                    right: false,
                });
                y += 20.0;
            }
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        _ => {}
    }
    // Keyboard cursor highlight (bevy hover parity): the cursor row
    // renders white so arrow-key nav is visible, not blind.
    let cursor = world
        .get_resource::<MenuState>()
        .map(|m| m.settings_cursor)
        .unwrap_or(usize::MAX);
    if cursor != usize::MAX {
        if let Some(row) = settings_hot_rows(page, vw).get(cursor) {
            for t in out.iter_mut() {
                if (t.gy - row.gy).abs() < 0.5 {
                    t.color = GUI_WHITE;
                }
            }
        }
    }
    out
}

/// 320-wide compatibility wrapper over [`menu_gui_texts_vw`] (kept
/// for headless tests: at `vw = 320` every anchor matches the legacy
/// layout exactly).
pub fn menu_gui_texts(kind: crate::MenuOverlay, world: &mut World) -> Vec<MenuGuiText> {
    menu_gui_texts_vw(kind, world, 320.0)
}

/// [`menu_gui_texts_vw`] mapped to dp through [`gui_texts_dp`], with
/// the live GML GUI width ([`gml_view_size`]) so centered headers land
/// on the view center and right-anchored buttons hug the live edge.
pub fn menu_gui_texts_dp(
    kind: crate::MenuOverlay,
    world: &mut World,
    canvas_dp: [f32; 2],
) -> Vec<GuiRow> {
    let vw = gml_view_size(canvas_dp)[0];
    gui_texts_dp(canvas_dp, menu_gui_texts_vw(kind, world, vw))
}

/// Sprite HUD bars (GML `scrDrawPlayerHUD` regions verbatim, bevy
/// `spawn_hud_art` GUI positions): health bar frame 2 at GUI (20,4)
/// with ghost/live fills at (22,7) width `84*frac`, rad bar at (4,4),
/// ammo row at y=32, weapon icons at y=16, offer icons bottom row at
/// y=219, held ultra/skills top-right from (`view_width-12`,13).
/// View-anchored: `view` is the world-space view rect; GUI offsets map
/// through [`hud_gui_map`] (identity: GUI px == view px). `dt_secs`
/// drives the `lsthealth` ghost decay
/// (GML `Player/Step_0`: lerp 0.2 past a 20 gap, else 0.5/step).
pub fn hud_sprites(
    world: &mut World,
    assets: &RenderAssets,
    view: [f32; 4],
    dt_secs: f32,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    if world
        .get_resource::<crate::savedata_part::SaveData>()
        .is_some_and(|s| !s.settings.show_hud)
    {
        return out;
    }
    let player_alive = world.query::<&Player>().iter(world).next().is_some();
    if !player_alive {
        return out;
    }
    let hud: HudState = sync_hud_state(world);
    let player = world
        .query::<(&Player, Option<&RaceState>)>()
        .iter(world)
        .next()
        .map(|(pl, rs)| {
            (
                pl.rogue_ammo,
                pl.rogue_ammo_max,
                pl.cuz_ammo,
                pl.cuz_ammo_max,
                pl.ultra,
                pl.mutations.clone(),
                pl.patience_used,
                pl.back_muscle,
                rs.map(|r| (r.race, r.skin)),
            )
        });
    let Some((
        rogue_ammo,
        rogue_max,
        cuz_ammo,
        cuz_max,
        ultra,
        mutations,
        patience_used,
        back_muscle,
        race_skin,
    )) = player
    else {
        return out;
    };
    let hurt = world
        .query::<(&Player, &HitFlash)>()
        .iter(world)
        .next()
        .is_some();

    // Ghost fill persistence (GML `Player/Step_0::lsthealth`
    // verbatim over float state: lerp 0.2 when the gap exceeds 20,
    // else approach 0.5 per 1/30 step).
    let last_hp = {
        let cur = hud.hp.max(0) as f32;
        let prev = world
            .get_resource::<crate::hud::HudBars>()
            .map(|b| b.last_hp)
            .unwrap_or(cur);
        let steps = (dt_secs.max(0.0) * 30.0).max(0.0);
        let last = if (cur - prev).abs() < 1e-6 {
            prev
        } else if (cur - prev).abs() > 20.0 {
            prev + (cur - prev) * (1.0 - 0.8f32.powf(steps))
        } else {
            let d = 0.5 * steps;
            if prev < cur {
                (prev + d).min(cur)
            } else {
                (prev - d).max(cur)
            }
        };
        world.insert_resource(crate::hud::HudBars { last_hp: last });
        last
    };

    // Health bar + fills (GML `scrDrawPlayerHUD:17-58`: bar frame 2 at
    // (20,4) via `draw_sprite` (origin (0,0): pixels at exactly
    // (20,4)); fills are `draw_sprite_ext` frame 0 xscale-stretched
    // from (22,7) — `sprHealthFill` is a (0,0)-origin 1x8 strip so the
    // left edge pins at x=22 while the width shrinks. Ghost in the
    // darkened HSV variant, live in `opt_healthcol`; hurt flash =
    // white frame 0 at the hp width. Desktop nudges both by +0.01.
    let gm = hud_gui_map(view);
    if let Some(s) = hud_gui_place(
        assets,
        "images/sprHealthBar.png",
        2,
        20.0,
        4.0,
        1.0,
        [1.0; 4],
        gm,
        view,
    ) {
        out.push(s);
    }
    let healthcol = world
        .get_resource::<crate::savedata_part::SaveData>()
        .map(|s| s.settings.healthcol_rgba())
        .unwrap_or([252.0 / 255.0, 56.0 / 255.0, 0.0, 1.0]);
    if let Some(h) = assets.native_size("images/sprHealthFill.png") {
        let frac = (hud.hp as f32 / hud.max_hp.max(1) as f32).clamp(0.0, 1.0);
        let gfrac = (last_hp / hud.max_hp.max(1) as f32).clamp(0.0, 1.0);
        // Fill quads keep their left edge at GUI x=22 while the width
        // shrinks (GML `draw_sprite_ext` xscale from (22,7) with a
        // (0,0)-origin 1x8 strip). `sprite_sized` anchors (0,0) at the
        // passed center, so the center IS the top-left: pass (22,7),
        // not the quad middle.
        let fill_at = |_w: f32| hud_gui_to_world(gm, view, 22.0, 7.0);
        let fill_size = |w: f32, h: f32| Vec2::new((w * gm.s).max(0.001), h * gm.s);
        let bg_tint = healthcol_dark(healthcol);
        if gfrac > 0.0 {
            let w = 84.0 * gfrac;
            if let Some(s) = assets.sprite_sized(
                "images/sprHealthFill.png",
                0,
                fill_at(w),
                fill_size(w, h.y),
                false,
                bg_tint,
            ) {
                out.push(s);
            }
        }
        if frac > 0.0 {
            let w = 84.0 * frac;
            let tint = if hurt { [1.0; 4] } else { healthcol };
            if let Some(s) = assets.sprite_sized(
                "images/sprHealthFill.png",
                0,
                fill_at(w),
                fill_size(w, h.y),
                false,
                tint,
            ) {
                out.push(s);
            }
        }
    }

    // Rad/exp bar + level badge (GML `scrDrawPlayerHUD:184-202`
    // verbatim: `sprExpBarLevel` first while an offer is pending
    // (`skillpoints/ultrapoints/wantdestinyskill`), then `sprExpBar` with
    // `frac*16` at GUI (4,4) on top; `sprNomutsLevel` at (11,16) when
    // the level cap is 0, the number while below cap, `sprUltraLevel`
    // at cap). The cap is `PLAYER_LEVEL_MAX` (10, 0 in no-muts custom).
    let offer_pending = world.get_resource::<PendingMutation>().is_some()
        || world.get_resource::<PendingUltra>().is_some();
    if offer_pending {
        if let Some(s) = hud_gui_place(
            assets,
            "images/sprExpBarLevel.png",
            0,
            4.0,
            4.0,
            1.0,
            [1.0; 4],
            gm,
            view,
        ) {
            out.push(s);
        }
    }
    if let Some(s) = hud_gui_place(
        assets,
        "images/sprExpBar.png",
        (hud.rads as f32 / hud.max_rads.max(1) as f32)
            .clamp(0.0, 1.0)
            .mul_add(16.0, 0.0)
            .floor()
            .min(16.0) as i32,
        4.0,
        4.0,
        1.0,
        [1.0; 4],
        gm,
        view,
    ) {
        out.push(s);
    }
    // GML `_level_max <= 0` (no-muts custom): the port never sets a 0
    // cap (level floor is 1), so this arm is vacuous — documented, not
    // drawn.
    // GML `sprUltraLevel` at (11,16) via `draw_sprite` (NOT `_ext`):
    // honors the strip origin (4,5 of 8x8), so pixels center near
    // (11,16) — the old `sprite_scaled_rotated` treated (11,16) as the
    // quad center, shifting it half a cell. `hud_gui_place` lands the
    // art top-left on the GUI point like `draw_sprite` does.
    if hud.level >= 10 {
        if let Some(s) = hud_gui_place(
            assets,
            "images/sprUltraLevel.png",
            0,
            11.0,
            16.0,
            1.0,
            [1.0; 4],
            gm,
            view,
        ) {
            out.push(s);
        }
    }

    // Ammo icon pairs (GML `scrDrawPlayerHUD:209-230` verbatim: GUI row
    // y=32, `dx = 2+(t-1)*10` minus 2 at Bolts and up; BG frame 2 when
    // the type matches the primary weapon (or the secondary on
    // Steroids), 1 for the secondary, else 0; icon drains against the
    // strip's own frame count over the Back-Muscle-adjusted cap).
    const AMMO_ICONS: [(&str, &str); 5] = [
        ("images/sprBulletIconBG.png", "images/sprBulletIcon.png"),
        ("images/sprShotIconBG.png", "images/sprShotIcon.png"),
        ("images/sprBoltIconBG.png", "images/sprBoltIcon.png"),
        ("images/sprExploIconBG.png", "images/sprExploIcon.png"),
        ("images/sprEnergyIconBG.png", "images/sprEnergyIcon.png"),
    ];
    let order = hud_weapon_order(&hud);
    let t1 = weapon_meta(
        hud.weapon_ids
            .get(order[0])
            .copied()
            .unwrap_or(WeaponId::NONE),
    )
    .wep_type as usize;
    let t2 = if order.len() > 1 {
        weapon_meta(hud.weapon_ids[order[1]]).wep_type as usize
    } else {
        0
    };
    let steroids_race = race_skin.is_some_and(|(r, _)| r == RaceId::Steroids);
    for (t, (bg, icon)) in AMMO_ICONS.iter().enumerate() {
        use crate::data::AmmoKind;
        let kind = match t + 1 {
            1 => AmmoKind::Bullets,
            2 => AmmoKind::Shells,
            3 => AmmoKind::Bolts,
            4 => AmmoKind::Explosives,
            _ => AmmoKind::Energy,
        };
        let cap = crate::comps_a::ammo_cap_with(back_muscle, kind).max(1);
        let fill = (hud.ammo.get(t + 1).copied().unwrap_or(0) as f32 / cap as f32).clamp(0.0, 1.0);
        let bg_frame = if t + 1 == t1 || (steroids_race && t + 1 == t2) {
            2
        } else if t + 1 == t2 {
            1
        } else {
            0
        };
        let dx = 2.0 + t as f32 * 10.0 - if t >= 2 { 2.0 } else { 0.0 };
        if let Some(s) = hud_gui_place(assets, bg, bg_frame, dx, 32.0, 1.0, [1.0; 4], gm, view) {
            out.push(s);
        }
        let frames = strip_frames(assets, icon).max(1) as f32 - 1.0;
        let icon_frame = (frames - (fill * frames).ceil()).clamp(0.0, frames) as i32;
        if let Some(s) = hud_gui_place(assets, icon, icon_frame, dx, 32.0, 1.0, [1.0; 4], gm, view)
        {
            out.push(s);
        }
    }
    // GML `scrDrawPlayerHUD:231-245` event/continued icons verbatim at
    // GUI y=33: daily/weekly icon at x=56 on event runs (weekly picks
    // the weekly sprite), custom icon +12 when custom, continued icon
    // +12 per flag. The port has no daily/weekly/custom/continued
    // runs (PLAY sub-rows deny; `Run` carries no continued flag), so
    // all three arms are vacuous — documented, not drawn.

    // Rogue/Cuz ammo pips (GML GUI `draw_sprite(sprite, sub, 110, 4)`:
    // top-left origin art drawn with its TOP-LEFT at (110,4) — GML
    // `draw_sprite` (not `_ext`) honors the strip origin, and these
    // strips carry origin (1,1), so pixels land at (109,3). Subimage
    // `ammo ? max(1, floor((frames-1) * progress)) : 0`; Cuz draws
    // `sprCuzAmmoHUDU` under the Emotional ultra. `hud_gui_place`
    // already lands the art top-left on the GUI point (same helper
    // as the health bar/ammo icons), so pass the GML point verbatim.
    if let Some((race, skin)) = race_skin {
        let pip = if race == RaceId::Rogue {
            let cskin = skin == crate::data::SkinLetter::C;
            let path = if ultra == Some(UltraMutationId::RoguePortalStrike) {
                if cskin {
                    "images/sprRogueAmmoHUDCTB.png"
                } else {
                    "images/sprRogueAmmoHUDTB.png"
                }
            } else if cskin {
                "images/sprRogueAmmoHUDC.png"
            } else {
                "images/sprRogueAmmoHUD.png"
            };
            Some((path, rogue_ammo as f32, rogue_max.max(1) as f32))
        } else if race == RaceId::Cuz {
            Some((
                if ultra == Some(UltraMutationId::CuzEmotional) {
                    "images/sprCuzAmmoHUDU.png"
                } else {
                    "images/sprCuzAmmoHUD.png"
                },
                cuz_ammo as f32,
                cuz_max.max(1) as f32,
            ))
        } else {
            None
        };
        if let Some((path, ammo, max)) = pip {
            let frames = strip_frames(assets, path).max(1);
            let progress = (ammo / max).clamp(0.0, 1.0);
            // GML verbatim: `_subimage = ammo ? max(1,
            // floor(_subimage_max * progress)) : 0` with
            // `_subimage_max = sprite_get_number - 1`.
            let sub = if ammo <= 0.0 {
                0
            } else {
                (((frames as f32 - 1.0) * progress).floor() as i32).max(1)
            };
            if let Some(s) = hud_gui_place(assets, path, sub, 110.0, 4.0, 1.0, [1.0; 4], gm, view)
            {
                out.push(s);
            }
        }
    }

    // Offer icons (GML `LevCont/Other_10` layout verbatim: the offer
    // row sits at `_yview = view_yview + view_height - 21` — GUI y=219
    // — with `step = min(32, floor(view_width/(n+1)))`, `half = step
    // div 2` (integer!), `scale = max(0.65, step/32)`, centered on the
    // live view center with a -12 shift at n>=10; SkillIcon cards draw
    // at `image_xscale/yscale = scale` (UltraIcon/CrownIcon unscaled).
    // Selected card lifts 1 px (`y - sign(selected)`) and draws white,
    // others gray. The selected card's textbox is ONE centered-middle
    // text at `(w/2, H-61-selected)` = (cx, 179) — see the Mutation
    // overlay arm (`menu_gui_texts_vw`). Normal offers ride
    // `sprSkillIcon` at the GML skill id (scaled); ultra offers ride
    // `sprEGSkillIcon` at `(race-1)*3+tier-1` (unscaled — GML only
    // scales SkillIcon).
    {
        // Ultra offers win over normal ones (same precedence as
        // `sync_hud_state`, `tick_mutation_mirror` and the click
        // hit-test; bevy `hud.rs`): when both resources coexist the
        // screen shows ultra cards.
        let n = world
            .get_resource::<PendingUltra>()
            .map(|u| u.choices.len())
            .or_else(|| {
                world
                    .get_resource::<PendingMutation>()
                    .map(|p| p.choices.len())
            })
            .unwrap_or(0);
        let is_ultra = world.get_resource::<PendingUltra>().is_some();
        if n > 0 {
            let selected = world
                .get_resource::<MenuState>()
                .and_then(|m| m.mutation_selected);
            let race = world
                .get_resource::<SelectedCharacter>()
                .map(|s| s.0)
                .unwrap_or(RaceId::Fish);
            let vw = view[2];
            let step = (vw / (n as f32 + 1.0)).floor().min(32.0);
            let scale = (step / 32.0).max(0.65);
            let half = (step as i32 / 2) as f32;
            let xview_shift = if n >= 10 { -12.0 } else { 0.0 };
            let start_x = vw * 0.5 + xview_shift - (n as f32 - 1.0) * half;
            let icon_y = 240.0 - 21.0;
            for i in 0..n {
                let (path, frame, mul) = if is_ultra {
                    let u = world
                        .get_resource::<PendingUltra>()
                        .and_then(|u| u.choices.get(i))
                        .copied();
                    let tier = u.map(ultra_tier).unwrap_or(2);
                    let frame = (race as i32 - 1) * 3 + tier - 1;
                    ("images/sprEGSkillIcon.png", frame, 1.0)
                } else {
                    let m = world
                        .get_resource::<PendingMutation>()
                        .and_then(|p| p.choices.get(i))
                        .copied();
                    let frame = m.map(crate::hud::mutation_skill_index).unwrap_or(0) as i32;
                    ("images/sprSkillIcon.png", frame, scale)
                };
                // Fall back to the HUD strip when the offer strip is
                // absent from the pack (same frame law; both strips are
                // 1-based for skills).
                let (path, frame) = if assets.uv(path, 0).is_some() {
                    (path, frame)
                } else {
                    ("images/sprSkillIconHUD.png", frame)
                };
                let is_selected = selected == Some(i);
                // GML `draw_sprite_ext(sprite, skill, x, y + appeary -
                // sign(selected), ..., selected ? c_white : c_gray)`:
                // unselected cards are full gray (not half-alpha), and
                // the lift is `sign(selected)` = 1 for any nonzero
                // selection value. Steady state here (`appeary = 0`).
                let lift = if is_selected { 1.0 } else { 0.0 };
                let tint = if is_selected {
                    [1.0; 4]
                } else {
                    [0.5, 0.5, 0.5, 1.0]
                };
                if let Some(s) = assets.sprite_scaled_rotated(
                    path,
                    frame,
                    hud_gui_to_world(gm, view, start_x + i as f32 * step, icon_y - lift),
                    mul * gm.s,
                    0.0,
                    tint,
                ) {
                    out.push(s);
                }
            }
        }
    }

    // Held ultra + skill icons (GML `scrDrawMiscHUD:68-113` verbatim):
    // ultras on `sprEGIconHUD` with the held ultra's own frame at y=13,
    // then skills on `sprSkillIconHUD` at the GML skill id at y=12
    // (the `_py--` runs once after the ultra loop, even with zero
    // ultras — so the skill row is ALWAYS y=12, never 13). Patience
    // (`mut_patience` with a stored `patienceskill`) draws
    // `sprSkillIconHUD` + `sprPatienceIconHUD` at the SAME `_px` with a
    // single advance. Top-right from GUI `view_width - 12` in 16 px
    // steps, wrapping at x<=120 (rows stack +16).
    {
        let vw = view[2];
        let held_race = race_skin.map(|(r, _)| r).unwrap_or(RaceId::Fish);
        let mut ultras: Vec<(&str, i32)> = Vec::new();
        if let Some(u) = ultra {
            ultras.push(("images/sprEGIconHUD.png", ultra_hud_frame(held_race, u)));
        }
        // Each skill slot is (base, optional patience overlay drawn at the
        // same cursor before the single advance).
        let mut skills: Vec<((&str, i32), Option<(&str, i32)>)> = Vec::new();
        for m in &mutations {
            let base = (
                "images/sprSkillIconHUD.png",
                crate::hud::mutation_skill_index(*m) as i32,
            );
            let overlay = if *m == MutationId::Patience && patience_used {
                Some(("images/sprPatienceIconHUD.png", 0))
            } else {
                None
            };
            skills.push((base, overlay));
        }
        let mut x = vw - 12.0;
        let mut y = 13.0;
        for (path, frame) in ultras {
            if let Some(s) = assets.sprite_scaled_rotated(
                path,
                frame,
                hud_gui_to_world(gm, view, x, y),
                gm.s,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            x -= 16.0;
            if x <= 120.0 {
                x = vw - 12.0;
                y += 16.0;
            }
        }
        // GML `_py--` runs unconditionally after the ultra loop: the
        // skill row is ALWAYS y=12 (the old `y == 13.0` guard wrongly
        // kept y=13 when no ultra was held).
        y -= 1.0;
        for ((path, frame), overlay) in skills {
            if let Some(s) = assets.sprite_scaled_rotated(
                path,
                frame,
                hud_gui_to_world(gm, view, x, y),
                gm.s,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            if let Some((opath, oframe)) = overlay {
                if let Some(s) = assets.sprite_scaled_rotated(
                    opath,
                    oframe,
                    hud_gui_to_world(gm, view, x, y),
                    gm.s,
                    0.0,
                    [1.0; 4],
                ) {
                    out.push(s);
                }
            }
            x -= 16.0;
            if x <= 120.0 {
                x = vw - 12.0;
                y += 16.0;
            }
        }
    }

    // Weapon strip (GML `scrDrawPlayerHUD:99-146` verbatim): `_wep`
    // at GUI x=24, `_bwep` at 68, then +20 for extras, y=16 (slot
    // order — position 0 is always the primary slot; swapping guns
    // swaps `_wep`/`_bwep` sim-side via `scrSwapWeps`, never the draw
    // order). Each gun is a `draw_sprite_part_ext` pixel window
    // `(xoffset, yoffset+swapanim-8, weapon_width, 14+swapanim)` with
    // `weapon_width` 16 (32 for a slot-0 melee): the window's top-left
    // lands exactly on `(dx, dy)`, 4-way outline (white when
    // `_is_active_weapon = (index==0 || Steroids)` else `#404040`,
    // only for active/broke/darkness/letterbox) at ±1 px, body in
    // `c_black`; `gpu_fog` tints for curse (`c_curse`) / ultra rad guns
    // (`c_ultra`) / golden (`c_gold`).
    // (`swapanim` y-offset + the white 0.2 reload wipe are sim-state the
    // port never tracks, so the steady frame draws.)
    for (pos, slot) in order.iter().copied().enumerate() {
        let Some(id) = hud.weapon_ids.get(slot).copied() else {
            continue;
        };
        let meta = weapon_meta(id);
        let path = if !meta.wep_sprt.is_empty() && meta.wep_sprt != "mskNone" {
            std::borrow::Cow::Owned(format!("images/{}.png", meta.wep_sprt))
        } else {
            std::borrow::Cow::Borrowed("images/sprRevolver.png")
        };
        let active = pos == 0 || steroids_race;
        let cursed = hud.weapon_cursed.get(slot).copied().unwrap_or(false);
        let ultra_gun = meta.wep_rads != 0;
        let golden = meta.wep_gold;
        // GML fog tints over the body draw.
        let fog: Option<[f32; 4]> = if cursed {
            Some([139.0 / 255.0, 68.0 / 255.0, 140.0 / 255.0, 1.0])
        } else if ultra_gun {
            Some([61.0 / 255.0, 198.0 / 255.0, 22.0 / 255.0, 1.0])
        } else if golden {
            Some([218.0 / 255.0, 208.0 / 255.0, 144.0 / 255.0, 1.0])
        } else {
            None
        };
        let dx = hud_weapon_dx(pos);
        let outline: [f32; 4] = if active {
            [1.0; 4]
        } else {
            [
                0x40 as f32 / 255.0,
                0x40 as f32 / 255.0,
                0x40 as f32 / 255.0,
                1.0,
            ]
        };
        // GML part window, steady frame (`swapanim = 0`): source rect
        // `(xoffset, yoffset-8, weapon_width, 14)` of the strip cell,
        // 32 px wide only for a slot-0 melee (`Ammo.None`).
        let melee = meta.wep_type == AmmoType::None;
        let order_len = hud.weapon_ids.len();
        let ww = if (pos == 0 || order_len <= 2) && melee {
            32.0
        } else {
            16.0
        };
        // 4-way outline quads (1 px GUI offsets around the body).
        for (ox, oy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            if let Some(s) = hud_weapon_part(
                assets, &path, dx + ox, 16.0 + oy, ww, outline, gm, view,
            ) {
                out.push(s);
            }
        }
        let body_tint = fog.unwrap_or([0.0, 0.0, 0.0, 1.0]);
        if let Some(s) = hud_weapon_part(assets, &path, dx, 16.0, ww, body_tint, gm, view) {
            out.push(s);
        }
    }
    out
}

/// One GML `draw_sprite_part_ext` weapon window: source rect
/// `(xoffset, yoffset-8, ww, 14)` of strip frame 1 (steady
/// `swapanim = 0`), drawn with its top-left at GUI `(dx, dy)`.
/// UVs lerp inside the frame cell (the atlas packs whole frames, so a
/// sub-rect is a straight sub-range); the quad size is the window in
/// GUI px with a centered anchor (GML draws with the sprite's own
/// origin, i.e. centered on the window middle here).
fn hud_weapon_part(
    assets: &RenderAssets,
    path: &str,
    dx: f32,
    dy: f32,
    ww: f32,
    tint: [f32; 4],
    map: HudGuiMap,
    view: [f32; 4],
) -> Option<SpriteInstance> {
    let (uv, def) = assets.uv(path, 1)?;
    let (sw, sh) = (def.w as f32, def.h as f32);
    if sw <= 0.0 || sh <= 0.0 {
        return None;
    }
    // Source window in strip px, clamped to the cell (short cells show
    // what exists; GML would sample edge texels past short art).
    let sx = (def.xorigin as f32).clamp(0.0, sw);
    let sy = (def.yorigin as f32 - 8.0).clamp(0.0, sh);
    let ww = ww.min((sw - sx).max(1.0));
    let wh = 14.0_f32.min((sh - sy).max(1.0));
    // Frame cell spans uv.min..uv.max; the window is the matching
    // fraction of it (atlas Y is down, same as the strip).
    let fx0 = sx / sw;
    let fy0 = sy / sh;
    let fx1 = (sx + ww) / sw;
    let fy1 = (sy + wh) / sh;
    let uv_min = Vec2::new(
        uv.min[0] + (uv.max[0] - uv.min[0]) * fx0,
        uv.min[1] + (uv.max[1] - uv.min[1]) * fy0,
    );
    let uv_max = Vec2::new(
        uv.min[0] + (uv.max[0] - uv.min[0]) * fx1,
        uv.min[1] + (uv.max[1] - uv.min[1]) * fy1,
    );
    let size = Vec2::new(ww * map.s, wh * map.s);
    let top_left = hud_gui_to_world(map, view, dx, dy);
    Some(SpriteInstance {
        center: top_left + size * 0.5,
        rotation: 0.0,
        size,
        anchor: Vec2::new(0.5, 0.5),
        flip_x: false,
        flip_y: false,
        uv_min: Vec2::new(uv_min[0], uv_min[1]),
        uv_max: Vec2::new(uv_max[0], uv_max[1]),
        color: tint_to_linear(tint),
        page: uv.page,
        z: 0.0,
        blend: SpriteBlend::Alpha,
    })
}

/// Coop fainted bars (GML `TopCont/Draw_0` `Revive` block verbatim):
/// `sprFaintedBar` at the view-clamped pos, then the grace bar
/// (`alarm[4] / 300 * 28`, red-black pulse from `sin(tottimer / 4)`)
/// or the bleed bar (`alarm[5] / 30 * 28`, red — including GML's
/// missing `+ 2` on the bleed right edge). Alarm values ride the
/// [`HudState`] snapshot; `view` is the world-space view rect.
pub fn fainted_bar_sprites(
    world: &mut World,
    assets: &RenderAssets,
    view: [f32; 4],
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    let hud: HudState = sync_hud_state(world);
    if hud.fainted_bars.is_empty() {
        return out;
    }
    let tottimer = world.get_resource::<Run>().map(|r| r.tottimer).unwrap_or(0);
    for bar in &hud.fainted_bars {
        let x = bar.x.clamp(view[0] + 30.0, view[0] + view[2] - 30.0);
        let y = (bar.y - 16.0).clamp(view[1] + 6.0, view[1] + view[3] - 6.0);
        if let Some(s) = assets.sprite_for(
            "images/sprFaintedBar.png",
            0,
            Vec2::new(x, y),
            false,
            0.0,
            [1.0; 4],
        ) {
            out.push(s);
        }
        let (bx, by) = (x - 16.0, y - 5.0);
        if bar.alarm4 > 0.0 {
            // GML merge_color(c_red, c_black, 0.5 + sin(tottimer/4)*0.25).
            let amt = 0.5 + ((tottimer as f32) / 4.0).sin() * 0.25;
            let w = bar.alarm4 / 300.0 * 28.0;
            if w > 0.0 {
                out.push(white_quad(
                    Vec2::new(bx + 2.0 + w / 2.0, by + 4.0),
                    0.0,
                    Vec2::new(w, 4.0),
                    [1.0 - amt, 0.0, 0.0, 1.0],
                ));
            }
        } else if bar.alarm5 > 0.0 {
            // Verbatim right edge: `_x + alarm[5] / 30 * 28` (no + 2).
            let w = bar.alarm5 / 30.0 * 28.0 - 2.0;
            if w > 0.0 {
                out.push(white_quad(
                    Vec2::new(bx + 2.0 + w / 2.0, by + 4.0),
                    0.0,
                    Vec2::new(w, 4.0),
                    [1.0, 0.0, 0.0, 1.0],
                ));
            }
        }
    }
    out
}

/// Cursor world position staged by the shell each frame (the viewport
/// `Hover` pick; `None` before the first hover). Feeds the GML
/// crosshair distance (`KeyCont.dis_fire` parity).
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct HoverWorld(pub Option<Vec2>);

/// GML crosshair filter state (`crosshair_x/y/alpha` on `Player`,
/// `objects/TopCont/Draw_0` verbatim): lerped aim point + fade.
#[derive(Resource, Clone, Copy, Debug)]
pub struct CrosshairState {
    pub x: f32,
    pub y: f32,
    pub alpha: f32,
    pub init: bool,
}

impl Default for CrosshairState {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            alpha: 0.0,
            init: false,
        }
    }
}

/// GML attack-button deadzone (`JoystickAttack/Create_0`: 0.4125):
/// the crosshair counts as active past `32 * 0.4125` px from the player.
pub const CROSSHAIR_DEADZONE: f32 = 32.0 * 0.4125;

/// World-space crosshair (GML `TopCont/Draw_0` player block verbatim):
/// `sprCrosshair[opt_crosshair]` lerped toward
/// `player + (16 + dis)` along the aim heading (0.8 active / 0.1 idle),
/// alpha lerped toward 5/0 at 0.4 (drawn as `min(1, alpha)`), active
/// past the attack deadzone. Skipped while paused (GML `PauseImage`
/// gate) and until the first hover stages a cursor.
pub fn crosshair_sprites(
    world: &mut World,
    assets: &RenderAssets,
    dt_secs: f32,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    // Live gameplay only: paused/menus/offers/game-over keep the last
    // aim but hide the cursor (previously it drew over game-over/title
    // from stale sim entities once `Paused` was forced false on death).
    let live = world
        .get_resource::<crate::state::AppState>()
        .is_some_and(|s| *s == crate::state::AppState::InGame)
        && !world
            .get_resource::<crate::state::Paused>()
            .is_some_and(|p| p.0)
        && world
            .get_resource::<crate::state::OverlayMenu>()
            .is_none_or(|o| *o == crate::state::OverlayMenu::None)
        && !world
            .get_resource::<crate::comps_a::Run>()
            .is_some_and(|r| r.game_over)
        && world
            .get_resource::<crate::comps_a::PendingMutation>()
            .is_none()
        && world
            .get_resource::<crate::comps_a::PendingUltra>()
            .is_none();
    if !live {
        return out;
    }
    let player = world
        .query::<(&Pos, &Player, &AimDir)>()
        .iter(world)
        .next()
        .map(|(p, _, a)| (p.0, a.0));
    let Some((pp, aim)) = player else {
        return out;
    };
    let hover = world.get_resource::<HoverWorld>().and_then(|h| h.0);
    let dir = if aim.length_squared() > 1e-6 {
        aim.normalize_or_zero()
    } else {
        Vec2::X
    };
    let dis = hover.map(|h| h.distance(pp)).unwrap_or(0.0);
    let active = dis > CROSSHAIR_DEADZONE;
    let tx = pp.x + dir.x * (16.0 + dis);
    let ty = pp.y + dir.y * (16.0 + dis);
    world.init_resource::<CrosshairState>();
    let mut st = world.resource_mut::<CrosshairState>();
    if !st.init {
        st.x = tx;
        st.y = ty;
        st.init = true;
    }
    let px = gml_rate(if active { 0.8 } else { 0.1 }, dt_secs);
    st.x += (tx - st.x) * px;
    st.y += (ty - st.y) * px;
    let pa = gml_rate(0.4, dt_secs);
    let target = if active { 5.0 } else { 0.0 };
    st.alpha += (target - st.alpha) * pa;
    let (cx, cy, alpha) = (st.x, st.y, st.alpha.min(1.0).max(0.0));
    drop(st);
    if alpha <= 0.01 {
        return out;
    }
    let frame = world
        .get_resource::<crate::savedata_part::SaveData>()
        .map(|s| s.settings.crosshair as i32)
        .unwrap_or(0);
    let frames = strip_frames(assets, "images/sprCrosshair.png").max(1) as i32;
    if let Some(s) = assets.sprite_for(
        "images/sprCrosshair.png",
        frame.clamp(0, frames - 1),
        Vec2::new(cx, cy),
        false,
        0.0,
        [1.0, 1.0, 1.0, alpha],
    ) {
        out.push(s);
    }
    out
}

/// Offscreen portal arrow (GML `TopCont/Draw_0` `Portal` block
/// verbatim): `sprPortalIndicator` at the 10 px-clamped view pos for
/// every portal outside the live view rect.
pub fn portal_indicator_sprites(
    world: &mut World,
    assets: &RenderAssets,
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam: &Camera2d,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    // Live runs only: stale portals from a previous run must not draw
    // arrows over title/menus.
    if world
        .get_resource::<crate::state::AppState>()
        .is_some_and(|s| *s != crate::state::AppState::InGame)
    {
        return out;
    }
    let view = view_rect_world(canvas_dp, world_size, cam);
    let (vx, vy, vw, vh) = (view[0], view[1], view[2], view[3]);
    let mut q = world.query::<(&Pos, &Portal)>();
    for (pos, _) in q.iter(world) {
        let (x, y) = (pos.0.x, pos.0.y);
        if x > vx && y > vy && x < vx + vw && y < vy + vh {
            continue;
        }
        let ix = x.clamp(vx + 10.0, vx + vw - 10.0);
        let iy = y.clamp(vy + 10.0, vy + vh - 10.0);
        if let Some(s) = assets.sprite_for(
            "images/sprPortalIndicator.png",
            0,
            Vec2::new(ix, iy),
            false,
            0.0,
            [1.0; 4],
        ) {
            out.push(s);
        }
    }
    out
}

/// Soft blob shadows under actors (GML `scrShadows` entity half,
/// verbatim strips, drawn into the `shad` surface at alpha 0.4 and
/// composited under the actors by `BackCont/Draw_0`):
/// enemies + the player on `shd64`, pickups on `shd16`, props on
/// `shd32`. Strips absent from the pack degrade to no shadow (same
/// graceful fallback as every other optional strip; per-kind sizes
/// like `shd96` for BigDog follow once the pack carries them).
/// GML per-object `spr_shadow_x/y` offsets are not recorded by the
/// sim, so blobs center on the actor (crowns keep the GML +3 y).
pub fn shadow_sprites(world: &mut World, assets: &RenderAssets) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    let mut q = world.query::<(&Pos, &Enemy)>();
    for (pos, _) in q.iter(world) {
        if assets.uv("images/shd64.png", 0).is_none() {
            break;
        }
        if let Some(s) = assets.sprite_for(
            "images/shd64.png",
            0,
            pos.0,
            false,
            0.0,
            [1.0, 1.0, 1.0, 0.4],
        ) {
            out.push(s);
        }
    }
    let mut q = world.query::<(&Pos, &Player)>();
    for (pos, _) in q.iter(world) {
        if assets.uv("images/shd64.png", 0).is_none() {
            break;
        }
        if let Some(s) = assets.sprite_for(
            "images/shd64.png",
            0,
            pos.0,
            false,
            0.0,
            [1.0, 1.0, 1.0, 0.4],
        ) {
            out.push(s);
        }
    }
    let mut q = world.query::<(&Pos, &Pickup)>();
    for (pos, _) in q.iter(world) {
        if assets.uv("images/shd16.png", 0).is_none() {
            break;
        }
        if let Some(s) = assets.sprite_for(
            "images/shd16.png",
            0,
            pos.0,
            false,
            0.0,
            [1.0, 1.0, 1.0, 0.4],
        ) {
            out.push(s);
        }
    }
    let mut q = world.query::<(&Pos, &Prop)>();
    for (pos, _) in q.iter(world) {
        if assets.uv("images/shd32.png", 0).is_none() {
            break;
        }
        if let Some(s) = assets.sprite_for(
            "images/shd32.png",
            0,
            pos.0,
            false,
            0.0,
            [1.0, 1.0, 1.0, 0.4],
        ) {
            out.push(s);
        }
    }
    out
}

/// Additive bloom glows (GML `scrDrawBloom`, called from
/// `SubTopCont/Draw_0` when `UberCont.opt_bloom`): the actor strip at
/// 2x scale, alpha 0.1 (`bm_add`). Ported for live projectiles (same
/// strip/frame/rotation as the body) and portals; gated on the
/// `bloom` setting exactly like GML.
pub fn bloom_sprites(world: &mut World, assets: &RenderAssets) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    if !world
        .get_resource::<crate::savedata_part::SaveData>()
        .is_none_or(|s| s.settings.bloom)
    {
        return out;
    }
    let mut q = world.query::<(
        &Pos,
        &Projectile,
        Option<&Velocity>,
        &Team,
        Option<&SlashProjectile>,
        Option<&crate::comps_a::ProjectileVisual>,
    )>();
    for (pos, proj, vel, team, slash, visual) in q.iter(world) {
        let path = projectile_art(proj, team, slash, visual);
        let frame = projectile_frame(assets, path, &proj.life);
        let rotation = vel
            .filter(|v| v.0.length_squared() > 1e-6)
            .map(|v| v.0.y.atan2(v.0.x))
            .unwrap_or(0.0);
        if let Some(mut s) =
            assets.sprite_scaled_rotated(path, frame, pos.0, 2.0, rotation, [1.0, 1.0, 1.0, 0.1])
        {
            s.blend = SpriteBlend::Additive;
            out.push(s);
        }
    }
    let mut q = world.query::<(&Pos, &Portal, &SpriteAnim)>();
    for (pos, _, anim) in q.iter(world) {
        if let Some(mut s) = assets.sprite_scaled_rotated(
            &anim.path,
            anim.frame as i32,
            pos.0,
            2.0,
            0.0,
            [1.0, 1.0, 1.0, 0.1],
        ) {
            s.blend = SpriteBlend::Additive;
            out.push(s);
        }
    }
    out
}
/// Char-pod layout (GML `Menu/Create_0` campfire law verbatim): `count`
/// pods of height `slot_h` on a 240-tall view. `step = min(20,
/// floor((320-40)/count))` over the fixed `game_screen_width` (320, NOT
/// the live view width), `xstart = 8`,
/// `ystart = H-slot_h-((36-slot_h) div 2)`. `wh[1]` is the view height
/// (always 240); `wh[0]` is accepted for call-site symmetry but ignored
/// for the step so widescreen keeps the GML left-clustered pods.
pub fn char_pod_layout(wh: [f32; 2], count: usize, slot_h: f32) -> Vec<[f32; 2]> {
    let step = 20.0_f32.min(((320.0 - 40.0) / count.max(1) as f32).floor());
    let y = wh[1] - slot_h - ((36.0 - slot_h) / 2.0).floor();
    (0..count).map(|i| [8.0 + i as f32 * step, y]).collect()
}

/// GO-button position (GML `Menu/Create_0`: past the last pod,
/// `y = H-36+bbox_h div 2-2`; x from the same 320-base step as the pods).
/// Returns the GML instance position (sprite origin). Note
/// `sprGoButtonSymbolic` has origin `(0,-2)`, so the drawn pixels/bbox sit
/// 2 px below this (see `title_click_action` / `menu_sprites`).
pub fn go_button_pos(wh: [f32; 2], count: usize, bbox_h: f32) -> [f32; 2] {
    let step = 20.0_f32.min(((320.0 - 40.0) / count.max(1) as f32).floor());
    let _ = wh[0];
    [
        8.0 + count as f32 * step + 2.0,
        wh[1] - 36.0 + (bbox_h / 2.0).floor() - 2.0,
    ]
}

// ---------------------------------------------------------------------------
// Title click routing (GML campfire pods + GO + loadout zones).
// ---------------------------------------------------------------------------

/// Character-pod hit size (bevy `title_screen` hover law: 16x24 at the
/// layout origin).
pub const TITLE_POD_W: f32 = 16.0;
/// Character-pod hit height.
pub const TITLE_POD_H: f32 = 24.0;
/// GO button hit size (bevy `GO_W`/`GO_H`).
pub const TITLE_GO_W: f32 = 31.0;
/// GO button hit height.
pub const TITLE_GO_H: f32 = 19.0;

/// Loadout availability (GML `scr_loadout_is_available_for_race` +
/// `scrLoadoutMenuInit`: no panel for Random and the trio).
pub fn loadout_available_for_race(race: RaceId) -> bool {
    !matches!(
        race,
        RaceId::Random | RaceId::BigDog | RaceId::Skeleton | RaceId::Frog
    )
}

/// Title click → [`UiAction`](crate::audio::UiAction). `slot_h`,
/// `crownsize` and `skinsize` are the same native-size values
/// `menu_sprites` draws with (callers pass the catalog sizes or the
/// 20px fallback); all other geometry mirrors the draw code so clicks
/// land on the sprites. Stray clicks route to nothing (the old blind
/// confirm-anywhere is gone).
pub fn title_click_action(
    world: &mut World,
    gx: f32,
    gy: f32,
    vw: f32,
    slot_h: f32,
    crownsize: f32,
    skinsize: f32,
) -> Option<UiAction> {
    let menu = world.get_resource::<MenuState>().cloned()?;
    // Gml id doubles as the `CHAR_SELECT_ORDER` index (order matches
    // discriminants, Random 0 .. Cuz 16).
    let selected = world
        .get_resource::<SelectedCharacter>()
        .map(|s| s.0 as usize)
        .unwrap_or(1);
    let race = CHAR_SELECT_ORDER[selected.min(CHAR_SELECT_ORDER.len() - 1)];
    // Loadout zones first (the panel floats over the pods' right end).
    if loadout_available_for_race(race) {
        if let Some(a) = loadout_click_action(
            world,
            race,
            selected,
            gx,
            gy,
            vw,
            crownsize,
            skinsize,
            menu.loadout_open,
        ) {
            return Some(a);
        }
    }
    // GO button (armed only). `go_button_pos` is the GML instance position
    // (sprite origin); `sprGoButtonSymbolic` origin is (0,-2) so the
    let roster =
        crate::state::menus::visible_roster(world.get_resource::<crate::savedata_part::SaveData>());
    if menu.title_go_visible {
        let go = go_button_pos([vw, 240.0], roster.len(), 19.0);
        let top = go[1] + 2.0;
        if gx >= go[0] && gx <= go[0] + TITLE_GO_W && gy >= top && gy <= top + TITLE_GO_H {
            return Some(UiAction::StartGame);
        }
    }
    for (i, pos) in char_pod_layout([vw, 240.0], roster.len(), slot_h)
        .iter()
        .enumerate()
    {
        if gx >= pos[0] && gx <= pos[0] + TITLE_POD_W && gy >= pos[1] && gy <= pos[1] + TITLE_POD_H
        {
            if let Some(race) = roster.get(i) {
                return Some(UiAction::SelectCharacter(*race as usize));
            }
        }
    }
    None
}

/// Mutation/ultra offer icon hit-test → [`UiAction`](crate::audio::UiAction).
/// Geometry mirrors the offer-icon draw in `hud_sprites` exactly (same
/// `step`/`half`/`start_x`/`icon_y` formulas over the same live `vw`,
/// including the `-12` shift at `n >= 10`); the hit box is the bevy
/// `mutation_panel` size (`24*scale` x `32*scale` centered on the icon,
/// top at `icon_y - 16*scale`). Two-step bevy law: an unhighlighted card
/// highlights (`SelectMutation`), the highlighted card commits
/// (`PickMutation`). Stray clicks and empty offers route to nothing, so a
/// misclick can never confirm a pick or leak into gameplay.
pub fn mutation_icon_hit_action(world: &mut World, gx: f32, gy: f32, vw: f32) -> Option<UiAction> {
    // Ultra offers win over normal ones (same precedence as `sync_hud_state`
    // and `tick_mutation_mirror`, bevy `hud.rs`): when both resources
    // coexist the screen shows ultra cards, so hit-testing must too.
    let n = world
        .get_resource::<PendingUltra>()
        .map(|u| u.choices.len())
        .or_else(|| {
            world
                .get_resource::<PendingMutation>()
                .map(|p| p.choices.len())
        })
        .unwrap_or(0);
    if n == 0 {
        return None;
    }
    let step = (vw / (n as f32 + 1.0)).floor().min(32.0);
    let scale = (step / 32.0).max(0.65);
    let half = (step as i32 / 2) as f32;
    let xview_shift = if n >= 10 { -12.0 } else { 0.0 };
    let start_x = vw * 0.5 + xview_shift - (n as f32 - 1.0) * half;
    let icon_y = 240.0 - 21.0;
    let hw = 24.0 * scale * 0.5;
    let top = icon_y - 16.0 * scale;
    let hh = 32.0 * scale;
    for i in 0..n {
        let cx = start_x + i as f32 * step;
        if (gx - cx).abs() <= hw && gy >= top && gy <= top + hh {
            let selected = world
                .get_resource::<MenuState>()
                .and_then(|m| m.mutation_selected);
            return Some(if selected == Some(i) {
                UiAction::PickMutation(i)
            } else {
                UiAction::SelectMutation(i)
            });
        }
    }
    None
}

/// Loadout panel hit rects. Open-frame geometry replicates
/// `menu_loadout_sprites` exactly (same formulas, same walk order —
/// weapons draw last so they win overlaps); the closed-frame splat zone
/// is the GML closed `_splat_pointed` rect (toggle only).
#[allow(clippy::too_many_arguments)]
fn loadout_click_action(
    world: &mut World,
    race: RaceId,
    selected: usize,
    gx: f32,
    gy: f32,
    vw: f32,
    crownsize: f32,
    skinsize: f32,
    open: bool,
) -> Option<UiAction> {
    use crate::savedata_part::SaveData;
    let (w, h) = (vw, 240.0);
    if open && selected != 0 {
        let save = world.get_resource::<SaveData>().cloned();
        let crown_row = save
            .as_ref()
            .and_then(|s| s.crown_got.get(&race))
            .copied()
            .unwrap_or({
                let mut r = [false; 14];
                r[0] = true;
                r[1] = true;
                r
            });
        let total_unlocked = crown_row.iter().filter(|b| **b).count();
        let crowntop = 72.0;
        let crownbottom = h - 72.0;
        let space = crownbottom - crowntop;
        let per_column = (space as i32 / crownsize.max(1.0) as i32).max(1);
        let per_row = 14 / per_column;
        let crownright = w + 12.0;
        let crownleft = crownright - per_row as f32 * crownsize;
        // GML `point_in_circle(mx, my, crown_x, crown_y, crownsize * 0.5)`.
        let crown_r = crownsize * 0.5;
        let mut cx = crownright - crownsize * 3.0;
        let mut cy = crowntop - 24.0;
        let mut crown_hit: Option<UiAction> = None;
        for id in 0..14u8 {
            if id == 0 && total_unlocked == 0 {
                cx += crownsize;
                continue;
            }
            if crown_hit.is_none() && (gx - cx).hypot(gy - cy) <= crown_r {
                crown_hit = Some(UiAction::SelectCrown(id));
            }
            cx += crownsize;
            if cx >= crownright || id == 0 {
                cx = crownleft;
                cy += crownsize;
            }
        }
        if crown_hit.is_some() {
            return crown_hit;
        }
        // Skin column.
        let skin_count = race_max_skin_count(race);
        let skins_x = crownleft - (crownsize / 2.0).floor() - 22.0;
        let skins_y = (h / 2.0).floor() - (skinsize * 0.5) * skin_count as f32 - 2.0;
        // GML `point_in_circle(mx, my, skins_x, skins_y, 10)`.
        for j in 0..skin_count {
            let sy = skins_y + j as f32 * skinsize;
            if (gx - skins_x).hypot(gy - sy) <= 10.0 {
                return Some(UiAction::SelectSkin(j as u8));
            }
        }
        // Weapon row (same slot list as the draw: default + stored).
        let weaponsize = 44.0;
        let weapons_x = ((crownright + crownleft) / 2.0).floor() - weaponsize * 0.5 * 2.0 + 20.0;
        let weapons_y = crownbottom + (crownsize / 2.0).floor() - 19.0;
        let loadout = save.as_ref().map(|s| s.race_loadout(race).clone());
        let default_weapon = race_default_weapon(race);
        let stored = loadout
            .as_ref()
            .map(|l| l.stored_weapon)
            .unwrap_or(WeaponId::NONE);
        let nslots = if stored != WeaponId::NONE && stored != default_weapon {
            2
        } else {
            1
        };
        for k in 0..nslots {
            let wx = weapons_x + k as f32 * weaponsize;
            // GML `point_in_circle(mx, my, weapons_x, weapons_y, 10)`.
            if (gx - wx).hypot(gy - weapons_y) <= 10.0 {
                return Some(UiAction::CycleStartWeapon(1));
            }
        }
        // Close arrow at the splat: GML open `_splat_pointed` rect
        // `[splat_x - 109 div 4, splat_x] x [splat_y - 69 div 2, splat_y]`.
        let splat = [w + 2.0, h - 36.0 + 2.0];
        if gx >= splat[0] - 27.0 && gx <= splat[0] && gy >= splat[1] - 34.0 && gy <= splat[1] {
            return Some(UiAction::ToggleLoadout);
        }
        return None;
    }
    if !open && selected != 0 {
        // Closed-frame splat zone: GML closed `_splat_pointed` rect
        // `[splat_x - 109 div 2, splat_x] x [splat_y - 69 div 2, splat_y]`
        // (toggles only — GML offers no crown/weapon picking closed, and
        // Random has no toggle at all).
        let splat = [w + 2.0, h - 36.0 + 2.0];
        if gx >= splat[0] - 54.0 && gx <= splat[0] && gy >= splat[1] - 34.0 && gy <= splat[1] {
            return Some(UiAction::ToggleLoadout);
        }
    }
    None
}

/// GML `scrRaceGetMaxSkinCount` verbatim (hidden-NTT access assumed,
// like the reference build): BigDog/Frog 1, Skeleton 2, Robot 4,
// otherwise 3.
pub fn race_max_skin_count(race: RaceId) -> usize {
    match race {
        RaceId::BigDog | RaceId::Frog => 1,
        RaceId::Skeleton => 2,
        RaceId::Robot => 4,
        _ => 3,
    }
}

/// Loadout grid (`scrMenuDrawLoadout` verbatim geometry at the open
/// frame: panel, crown grid, skin column, weapon row). `selected` is the
/// `CHAR_SELECT_ORDER` index. `vwvh` is the GML view size ([`gml_view_size`]);
/// `to_world` maps view px to world.
fn menu_loadout_sprites(
    world: &mut World,
    assets: &RenderAssets,
    vwvh: [f32; 2],
    selected: usize,
    to_world: &dyn Fn([f32; 2]) -> Vec2,
    out: &mut Vec<SpriteInstance>,
) {
    let (w, h) = (vwvh[0], vwvh[1]);
    let race = CHAR_SELECT_ORDER[selected.min(CHAR_SELECT_ORDER.len() - 1)];
    let save = world
        .get_resource::<crate::savedata_part::SaveData>()
        .cloned();
    let loadout = save.as_ref().map(|s| s.race_loadout(race).clone());
    let crown_row = save
        .as_ref()
        .and_then(|s| s.crown_got.get(&race))
        .copied()
        .unwrap_or({
            let mut r = [false; 14];
            r[0] = true;
            r[1] = true;
            r
        });

    let crownsize = assets
        .native_size("images/sprLoadoutCrown.png")
        .map(|s| s.y - 4.0)
        .unwrap_or(20.0);
    let skinsize = assets
        .native_size("images/sprLoadoutSkin.png")
        .map(|s| s.x - 4.0)
        .unwrap_or(20.0);
    let skin_count = race_max_skin_count(race);
    let crowntop = 72.0;
    let crownbottom = h - 72.0;
    let space = crownbottom - crowntop;
    let per_column = (space as i32 / crownsize.max(1.0) as i32).max(1);
    let per_row = 14 / per_column;
    let crownright = w + 12.0;
    let crownleft = crownright - per_row as f32 * crownsize;
    let skins_x = crownleft - (crownsize / 2.0).floor() - 22.0;
    let skins_y = (h / 2.0).floor() - (skinsize * 0.5) * skin_count as f32 - 2.0;
    let weaponsize = 44.0;
    let weapons_x = ((crownright + crownleft) / 2.0).floor() - weaponsize * 0.5 * 2.0 + 20.0;
    let weapons_y = crownbottom + (crownsize / 2.0).floor() - 19.0;
    let splat = [w + 2.0, h - 36.0 + 2.0];

    // Panel + splat + arrow.
    if let Some(open) = assets.native_size("images/sprLoadoutOpen.png") {
        let xs = ((w - skins_x) / (open.x - crownsize * 2.0)).max(1.0);
        let ys = (splat[1] - 36.0) / open.y + 0.05;
        if let Some(s) = assets.sprite_stretched(
            "images/sprLoadoutOpen.png",
            2,
            to_world([splat[0] - 2.0, splat[1]]),
            Vec2::new(xs, ys),
            0.0,
            [1.0; 4],
        ) {
            out.push(s);
        }
    }
    // Splat + arrow (GML frame-0 draw: `draw_sprite_ext(sprLoadoutSplat,
    // splatindex, splat_x, splat_y, 1, 1.05, ...)`; the y-stretch pins
    // the bottom-right origin, same as GML).
    if let Some(s) = assets.sprite_stretched(
        "images/sprLoadoutSplat.png",
        0,
        to_world(splat),
        Vec2::new(1.0, 1.05),
        0.0,
        [1.0; 4],
    ) {
        out.push(s);
    }
    if let Some(s) = assets.sprite_for(
        "images/sprLoadoutArrow.png",
        1,
        to_world([splat[0] - 16.0, splat[1] - 16.0]),
        false,
        0.0,
        [1.0; 4],
    ) {
        out.push(s);
    }

    // Crown grid (GML `crwn_random` (0) skipped when bare; wraps at
    // the right edge or the none crown).
    let total_unlocked = crown_row.iter().filter(|b| **b).count();
    let current = loadout.as_ref().map(|l| l.start_crown).unwrap_or(0);
    let mut cx = crownright - crownsize * 3.0;
    let mut cy = crowntop - 24.0;
    for id in 0..14u8 {
        if id == 0 && total_unlocked == 0 {
            cx += crownsize;
            continue;
        }
        let unlocked = crown_row.get(id as usize).copied().unwrap_or(false);
        let tint = if unlocked && id == current {
            [1.0; 4]
        } else {
            [0.5, 0.5, 0.5, 1.0]
        };
        if let Some(s) = assets.sprite_for(
            if unlocked {
                "images/sprLoadoutCrown.png"
            } else {
                "images/sprLockedLoadoutCrown.png"
            },
            id as i32,
            to_world([cx, cy]),
            false,
            0.0,
            tint,
        ) {
            out.push(s);
        }
        cx += crownsize;
        if cx >= crownright || id == 0 {
            cx = crownleft;
            cy += crownsize;
        }
    }

    // Skin column for the selected race (GML frames are skin
    // subimages, not column positions; the preferred skin draws white,
    // others uigray).
    let skins = loadout
        .as_ref()
        .map(|l| l.unlocked_skins)
        .unwrap_or([true, false, false, false]);
    let preferred = loadout.as_ref().map(|l| l.preferred_skin).unwrap_or(0);
    let mut sy = skins_y;
    for j in 0..skin_count {
        let open = skins.get(j).copied().unwrap_or(false);
        let tint = if j as u8 == preferred {
            [1.0; 4]
        } else {
            [0.5, 0.5, 0.5, 1.0]
        };
        if let Some(s) = assets.sprite_for(
            if open {
                "images/sprLoadoutSkin.png"
            } else {
                "images/sprLoadoutSkinLocked.png"
            },
            race_skin_subimage(race, j as u8),
            to_world([skins_x, sy]),
            false,
            0.0,
            tint,
        ) {
            out.push(s);
        }
        sy += skinsize;
    }

    // Weapon row (GML slots `[race_default, stored]`, second skipped
    // when it equals the default; the chosen start weapon draws white,
    // the other uigray; dedicated loadout art native, else the world
    // sprite at 2x rotated 30 degrees).
    let mut wx = weapons_x;
    let default_weapon = race_default_weapon(race);
    let stored = loadout
        .as_ref()
        .map(|l| l.stored_weapon)
        .unwrap_or(WeaponId::NONE);
    let chosen = loadout
        .as_ref()
        .map(|l| l.start_weapon)
        .unwrap_or(WeaponId::NONE);
    let mut slots = vec![default_weapon];
    if stored != WeaponId::NONE && stored != default_weapon {
        slots.push(stored);
    }
    for wid in slots {
        let tint = if chosen == wid {
            [1.0; 4]
        } else {
            [0.5, 0.5, 0.5, 1.0]
        };
        push_loadout_weapon(assets, wid, to_world([wx, weapons_y]), tint, out);
        wx += weaponsize;
    }
}

/// One loadout weapon icon at a view point (shared by the open weapon
/// row and the closed-frame minis): dedicated loadout art native, else
/// the world sprite at 2x rotated 30 degrees.
fn push_loadout_weapon(
    assets: &RenderAssets,
    wid: WeaponId,
    pos: Vec2,
    tint: [f32; 4],
    out: &mut Vec<SpriteInstance>,
) {
    let meta = weapon_meta(wid);
    if let Some(lout) = meta.wep_lout {
        let path = format!("images/{lout}.png");
        if let Some(s) = assets.sprite_for(&path, 0, pos, false, 0.0, tint) {
            out.push(s);
        }
    } else if !meta.wep_sprt.is_empty() && meta.wep_sprt != "mskNone" {
        let path = format!("images/{}.png", meta.wep_sprt);
        if let Some(s) =
            assets.sprite_scaled_rotated(&path, 0, pos, 2.0, 30.0f32.to_radians(), tint)
        {
            out.push(s);
        }
    }
}

/// Closed-frame loadout preview (GML `scrCampfireMenuCreate` "Current
/// loadout" region: mini crown/weapons ride the closed frame for any
/// non-Random race; splat + arrow need an available loadout, and trio
/// races get minis only — `!scr_loadout_is_available_for_race` kills
/// just the splat toggle). Mini positions are headless steady-state
/// (`_splat_pointed = 0`): crown at `(splat-60, splat-40)`, weapons at
/// `(splat-44/splat-68, splat-15)`. The haste-crown clock
/// (`scrDrawClock`) is dropped.
#[allow(clippy::too_many_arguments)]
fn menu_loadout_closed_sprites(
    world: &mut World,
    assets: &RenderAssets,
    vwvh: [f32; 2],
    selected: usize,
    available: bool,
    to_world: &dyn Fn([f32; 2]) -> Vec2,
    out: &mut Vec<SpriteInstance>,
) {
    let (w, h) = (vwvh[0], vwvh[1]);
    let race = CHAR_SELECT_ORDER[selected.min(CHAR_SELECT_ORDER.len() - 1)];
    let save = world
        .get_resource::<crate::savedata_part::SaveData>()
        .cloned();
    let loadout = save.as_ref().map(|s| s.race_loadout(race).clone());
    let splat = [w + 2.0, h - 36.0 + 2.0];
    if available {
        if let Some(s) = assets.sprite_stretched(
            "images/sprLoadoutSplat.png",
            0,
            to_world(splat),
            Vec2::new(1.0, 1.05),
            0.0,
            [1.0; 4],
        ) {
            out.push(s);
        }
        if let Some(s) = assets.sprite_for(
            "images/sprLoadoutArrow.png",
            0,
            to_world([splat[0] - 16.0, splat[1] - 16.0]),
            false,
            0.0,
            [1.0; 4],
        ) {
            out.push(s);
        }
    }
    let lx = splat[0] - 60.0;
    let ly = splat[1] - 15.0;
    let crown = loadout.as_ref().map(|l| l.start_crown).unwrap_or(0);
    if crown != 0 {
        let gml = crate::savedata_part::crown_port_to_gml(crown);
        if let Some(s) = assets.sprite_for(
            "images/sprLoadoutCrown.png",
            gml as i32,
            to_world([lx, ly - 25.0]),
            false,
            0.0,
            [1.0; 4],
        ) {
            out.push(s);
        }
    }
    let default_weapon = race_default_weapon(race);
    let stored = loadout
        .as_ref()
        .map(|l| l.stored_weapon)
        .unwrap_or(WeaponId::NONE);
    let primary = loadout
        .as_ref()
        .map(|l| l.start_weapon)
        .filter(|w| *w != WeaponId::NONE)
        .unwrap_or(default_weapon);
    if stored != WeaponId::NONE && stored != primary {
        push_loadout_weapon(
            assets,
            stored,
            to_world([lx + 16.0, ly]),
            [0.75, 0.75, 0.75, 1.0],
            out,
        );
        push_loadout_weapon(assets, primary, to_world([lx - 8.0, ly]), [1.0; 4], out);
    } else {
        push_loadout_weapon(assets, primary, to_world([lx, ly]), [1.0; 4], out);
    }
}
/// GML `scrRaceGetStarterWeapon` verbatim (GML weapon ids double as
/// port [`WeaponId`]s).
pub fn race_default_weapon(race: RaceId) -> WeaponId {
    match race {
        RaceId::Venuz | RaceId::Cuz => WeaponId(39),
        RaceId::Chicken => WeaponId(46),
        RaceId::Rogue => WeaponId(81),
        RaceId::BigDog => WeaponId(108),
        RaceId::Skeleton => WeaponId(56),
        RaceId::Frog => WeaponId(127),
        _ => WeaponId::REVOLVER,
    }
}

/// GML `mut_*` frame for `sprSkillIconHUD` (port `MutationId` order
/// differs, so this is an explicit name table, ids 1-29).
pub fn mutation_hud_frame(id: MutationId) -> i32 {
    match id {
        MutationId::RhinoSkin => 1,
        MutationId::ExtraFeet => 2,
        MutationId::PlutoniumHunger => 3,
        MutationId::RabbitPaw => 4,
        MutationId::ThroneButt => 5,
        MutationId::LuckyShot => 6,
        MutationId::Bloodlust => 7,
        MutationId::GammaGuts => 8,
        MutationId::SecondStomach => 9,
        MutationId::BackMuscle => 10,
        MutationId::ScarierFace => 11,
        MutationId::Euphoria => 12,
        MutationId::LongArms => 13,
        MutationId::BoilingVeins => 14,
        MutationId::ShotgunShoulders => 15,
        MutationId::RecycleGland => 16,
        MutationId::LaserBrain => 17,
        MutationId::LastWish => 18,
        MutationId::EagleEyes => 19,
        MutationId::ImpactWrists => 20,
        MutationId::BoltMarrow => 21,
        MutationId::Stress => 22,
        MutationId::TriggerFingers => 23,
        MutationId::SharpTeeth => 24,
        MutationId::Patience => 25,
        MutationId::Hammerhead => 26,
        MutationId::StrongSpirit => 27,
        MutationId::OpenMind => 28,
        MutationId::HeavyHeart => 29,
    }
}

/// GML ultra tier per race (`UltraSkill` A=1/B=2): the offer strip
/// (`sprEGSkillIcon`) indexes `(race-1)*3+tier-1`, the held strip
/// (`sprEGIconHUD`, from `ultra_hud`) `race*3+tier-1`. Verified against
/// the GML `UltraSkill` enum for races 1-12; 13-16 are best-effort
/// (port-custom ultra names: missiles/blood/gas/arsenal second, and
/// the port-only Cuz QuickSwap rides tier 2 with no GML frame).
pub fn ultra_tier(ultra: UltraMutationId) -> i32 {
    match ultra {
        UltraMutationId::FishConfiscate
        | UltraMutationId::CrystalFortress
        | UltraMutationId::EyesProjectileStyle
        | UltraMutationId::MeltingBrainCapacity
        | UltraMutationId::PlantTrapper
        | UltraMutationId::VenuzGunGod
        | UltraMutationId::SteroidsAmbidextrous
        | UltraMutationId::RobotRefinedTaste
        | UltraMutationId::ChickenHarderToKill
        | UltraMutationId::RebelPersonalGuard
        | UltraMutationId::HorrorStalker
        | UltraMutationId::RoguePortalStrike
        | UltraMutationId::BigDogGuardian
        | UltraMutationId::SkeletonNecromancy
        | UltraMutationId::FrogToxicLord
        | UltraMutationId::CuzHoarder => 1,
        _ => 2,
    }
}

/// GML ultra HUD frame (`race * 3 + tier - 1`, the `ultra_hud` array
/// law in `scr_ultra_set`) for the held strip `sprEGIconHUD`.
pub fn ultra_hud_frame(race: RaceId, ultra: UltraMutationId) -> i32 {
    (race as i32) * 3 + ultra_tier(ultra) - 1
}

/// GML `scr_race_get_skin_subimage` verbatim (race/skin to
/// `sprBigPortrait` frame; Random → -1 skips the draw; the Robot-D
/// NTT bug sits at 56, not 55). Port `RaceId` discriminants match the
/// GML `Race` enum exactly; `skin` is the `SkinLetter` index.
pub fn race_skin_subimage(race: RaceId, skin: u8) -> i32 {
    if race == RaceId::Random {
        return -1;
    }
    if race == RaceId::Robot && skin == 3 {
        return 56;
    }
    let r = race as i32;
    if skin < 2 {
        skin as i32 + (r - 1) * 2
    } else {
        skin as i32 * 16 + (r - 1)
    }
}

/// Campfire big-portrait resolution (GML
/// `scrCampfireMenuDrawRacePortrait` portrait region verbatim):
/// headless Chicken when hp <= 0 (frame = skin), hooded Rebel B-skin
/// in the city, else `sprBigPortrait` + [`race_skin_subimage`].
/// `hp` is `None` when no live player of that race exists (char
/// select before the run: treated as alive). Returns `None` for
/// Random (GML subimage -1 skips the draw). The port has no separate
/// rebel-hood flag — GML keys it purely on B skin + city area, which
/// is what this checks.
pub fn big_portrait_for(
    race: RaceId,
    skin: u8,
    hp: Option<i32>,
    area: AreaId,
) -> Option<(&'static str, i32)> {
    if race == RaceId::Random {
        return None;
    }
    if race == RaceId::Chicken && hp.is_some_and(|h| h <= 0) {
        return Some((
            "images/sprBigPortraitChickenHeadless.png",
            skin.min(2) as i32,
        ));
    }
    if race == RaceId::Rebel && skin == 1 && area == AreaId::City {
        return Some(("images/sprBigPortraitRebelBHooded.png", 0));
    }
    Some(("images/sprBigPortrait.png", race_skin_subimage(race, skin)))
}

/// Char-splat frame for a race (GML draws `Menu.splatindex`, a random
/// frame per menu; the port pins it per race so menus snapshot
/// deterministically).
pub fn char_splat_frame(race: RaceId, frames: u32) -> i32 {
    if frames == 0 {
        return 0;
    }
    (race as u32 % frames) as i32
}

/// Deterministic 0..1 hash for headless jitter (GML `orandom`/`random`
/// stand-in). Callers mix the quantized 30 Hz step into the seed so the
/// boot reel shimmer/shake jitters every step like GML `Draw` (and bevy
/// `boot_intro`, which re-rolls each tick) while staying deterministic
/// for a given `t`.
fn hash01(n: u32) -> f32 {
    let x = n.wrapping_mul(0x9E37_79B9).wrapping_add(0x85EB_CA6B);
    let mut h = x ^ (x >> 15);
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    ((h & 0xFFFF) as f32) / 65535.0
}

/// Boot-reel art (`Vlambeer/Draw_0` + `Logo/Draw_0` verbatim): saving
/// icon, Vlambeer card + additive shimmer, logo gunfire frames with
/// shake and the additive glow ring. Driven by [`SplashState`]
/// (missing state reads as mode 0). GUI 320x240 through
/// [`hud_gui_map`], so rows line up with the splash text overlay.
pub fn splash_sprites(
    world: &mut World,
    assets: &RenderAssets,
    view: [f32; 4],
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    let splash = world.get_resource::<SplashState>().cloned();
    let (mode, t, guns) = splash
        .as_ref()
        .map(|s| (s.mode, s.t, s.guns))
        .unwrap_or((0, 0.0, 0));
    let gm = hud_gui_map(view);
    // GML splash rooms center on the live view (`Logo`/`Vlambeer`
    // objects sit at the room center = view center): `cx` is the view
    // center x in GUI px, `cy` the middle (120).
    let cx = view[2] * 0.5;
    match mode {
        0 => {
            // Saving icon (`da += 0.5`/step == 15/s) at view center-16.
            let frames = strip_frames(assets, "images/sprSaving.png").max(1) as f32;
            let frame = ((t * 15.0).floor() % frames) as i32;
            if let Some(s) = assets.sprite_scaled_rotated(
                "images/sprSaving.png",
                frame,
                hud_gui_to_world(gm, view, cx, 104.0),
                gm.s,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
        }
        2 => {
            // GML `Vlambeer/Draw_0` verbatim: `draw_sprite(sprite_index,
            // 0, view_x + (view_w - sprite_w)/2, view_y + (view_h -
            // sprite_h))` — top-left pinned at `center_x - w/2`,
            // `bottom - h`, drawn CENTERED (`sprite_for`, not the
            // top-left `hud_gui_place`: `sprVlambeer` is a top-left
            // zero-origin strip). + 10 additive shimmer copies at
            // `orandom(4)` (±4 px), alpha 0.1. GML re-rolls every draw,
            // so offsets jitter each 30 Hz step (quantized from `t` to
            // stay deterministic), not frozen on a static hash.
            let (fw, fh) = assets
                .native_size("images/sprVlambeer.png")
                .map(|v| (v.x, v.y))
                .unwrap_or((320.0, 240.0));
            let top_left = hud_gui_to_world(gm, view, cx - fw * 0.5, 240.0 - fh);
            if let Some(s) =
                assets.sprite_for("images/sprVlambeer.png", 0, top_left, false, 0.0, [1.0; 4])
            {
                out.push(s);
            }
            let step = (t * 30.0).floor().max(0.0) as u32;
            for i in 0..10u32 {
                let jx =
                    (hash01(i.wrapping_mul(2).wrapping_add(step.wrapping_mul(7919))) - 0.5) * 8.0;
                let jy = (hash01(
                    i.wrapping_mul(2)
                        .wrapping_add(1)
                        .wrapping_add(step.wrapping_mul(104_729)),
                ) - 0.5)
                    * 8.0;
                if let Some(mut s) = assets.sprite_for(
                    "images/sprVlambeer.png",
                    0,
                    top_left + Vec2::new(jx, jy),
                    false,
                    0.0,
                    [1.0, 1.0, 1.0, 0.1],
                ) {
                    // Additive shimmer (`bm_add`).
                    s.blend = SpriteBlend::Additive;
                    out.push(s);
                }
            }
        }
        4 => {
            // Logo gunfire (`Logo/Alarm_0` steps) + shake + glow ring
            // at full charge (`image_index == 7`).
            let frame = (guns as i32).min(7);
            let last = if guns == 0 {
                0.0
            } else {
                SPLASH_GUN_STEPS[(guns as usize).min(SPLASH_GUN_STEPS.len()) - 1]
            };
            let amp = if guns as usize >= SPLASH_GUN_STEPS.len() {
                2.5
            } else if guns > 0 {
                0.5
            } else {
                0.0
            };
            let shake = (amp - (t - last) * 30.0).max(0.0);
            // GML `Logo/Draw_0` re-rolls `orandom(shake)` every draw;
            // jitter each 30 Hz step like the Vlambeer shimmer above.
            let step = (t * 30.0).floor().max(0.0) as u32;
            let jx = (hash01(
                1000u32
                    .wrapping_add(guns as u32 * 7)
                    .wrapping_add(step.wrapping_mul(7919)),
            ) - 0.5)
                * 2.0
                * shake;
            let jy = (hash01(
                2000u32
                    .wrapping_add(guns as u32 * 13)
                    .wrapping_add(step.wrapping_mul(104_729)),
            ) - 0.5)
                * 2.0
                * shake;
            if let Some(s) = assets.sprite_scaled_rotated(
                "images/sprLogo.png",
                frame,
                hud_gui_to_world(gm, view, cx + jx, 120.0 + jy),
                gm.s,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            if frame >= 7 {
                // 8-way glow ring (`bm_add`, alpha 0.05), breathing
                // with `wave` (GML `Logo/Draw_0`: ~0.13/draw == 3.9/s
                // at 30 Hz: 0.05 plus `random(0.02)` per each of the 8
                // copies; bevy matches with `dt * 3.9`). The per-copy
                // radius re-rolls every step like GML's `random(1)`.
                let wave = t * 3.9;
                for i in 0..8u32 {
                    let ang = i as f32 * 45.0 * std::f32::consts::PI / 180.0;
                    let r_extra = hash01(
                        3000u32
                            .wrapping_add(i)
                            .wrapping_add(step.wrapping_mul(12979)),
                    );
                    let radius = 4.0 + (wave + i as f32 * 0.02).sin() * (2.0 + r_extra);
                    if let Some(mut s) = assets.sprite_scaled_rotated(
                        "images/sprLogoGlow.png",
                        0,
                        hud_gui_to_world(
                            gm,
                            view,
                            cx + jx + radius * ang.cos(),
                            120.0 + jy + radius * ang.sin(),
                        ),
                        gm.s,
                        0.0,
                        [1.0, 1.0, 1.0, 0.05],
                    ) {
                        // Additive glow (`bm_add`).
                        s.blend = SpriteBlend::Additive;
                        out.push(s);
                    }
                }
            }
        }
        _ => {}
    }
    out
}

/// GML `make_color_hsv` parity (h/s/v 0..255): loop-tint colors for
/// the roadmap waypoint pass (`scrDrawRoadmap` colors each loop
/// `make_color_hsv((loop * 39) % 255, 200, 200)`).
fn roadmap_loop_tint(loop_count: u32) -> [f32; 4] {
    let h = ((loop_count * 39) % 255) as f32 / 255.0 * 360.0;
    let (s, v) = (200.0 / 255.0, 200.0 / 255.0);
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as u32 / 60 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m, 1.0]
}

/// GML `draw_line_pixelated` verbatim (`scrDrawRoadmap` local):
/// `dist = point_distance`, `dir = point_direction` (0 east, 90 south,
/// y-down degrees); y-bias +2 when `dir in [90,270)` else +1, then
/// `draw_sprite_ext(sprPixel, 0, x1, y1b, dist, 1, dir, color, alpha)`.
/// The sprite pipe has a rotation channel but no line primitive, so the
/// line rides a stretched `sprPixel` centered at the biased midpoint
/// with `rotation = atan2(dy, dx)` (beam law). Zero-length draws are
/// skipped (GML `xscale = 0` draws nothing).
fn pixel_line(
    assets: &RenderAssets,
    to_world: &dyn Fn(f32, f32) -> Vec2,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    tint: [f32; 4],
    out: &mut Vec<SpriteInstance>,
) {
    let dx0 = x2 - x1;
    let dy0 = y2 - y1;
    if dx0.hypot(dy0) < 1e-6 {
        return;
    }
    let dir_deg = dy0.atan2(dx0).to_degrees().rem_euclid(360.0);
    let bias = if (90.0..270.0).contains(&dir_deg) {
        2.0
    } else {
        1.0
    };
    let (y1b, y2b) = (y1 + bias, y2 + bias);
    let dx = x2 - x1;
    let dy = y2b - y1b;
    let dist = dx.hypot(dy);
    if dist < 1e-6 {
        return;
    }
    let rot = dy.atan2(dx);
    let center = to_world((x1 + x2) * 0.5, (y1b + y2b) * 0.5);
    if let Some(native) = assets.native_size("images/sprPixel.png") {
        let nw = native.x.max(1.0);
        let nh = native.y.max(1.0);
        if let Some(mut s) = assets.sprite_stretched(
            "images/sprPixel.png",
            0,
            center,
            Vec2::new(dist / nw, 1.0 / nh),
            rot,
            tint,
        ) {
            s.anchor = Vec2::new(0.5, 0.5);
            out.push(s);
            return;
        }
    }
    out.push(white_quad(center, rot, Vec2::new(dist.max(1.0), 1.0), tint));
}

/// `scrDrawRoadmap` sprite layer (`GameOver/Draw_0`,
/// `GenCont/Draw_0`): score splats + kills icon (the area/kill strings
/// ride the text overlay), the 7-area dot strip (`sprMapDot` 3px,
/// segment 9px, strip 135px wide centered on `drawx`), the palace
/// crown, and the waypoint dots + connector lines from
/// `Run.waypoints` (`pos` caps the drawn prefix; game-over passes the
/// full log). GML area ids: `waypnt % 100` with secrets on a second
/// row 10px down (`sprMapDotOut`); per-loop colors come from
/// [`roadmap_loop_tint`]. Connector lines ride [`pixel_line`] (GML
/// `draw_line_pixelated`: stretched `sprPixel` at `point_direction`
/// rotation with the +1/+2 y-bias); the background hatch is 3 black +
/// 1 white lines verbatim, the waypoint shadow 3 black + 1 color line.
pub fn roadmap_sprites(
    assets: &RenderAssets,
    to_world: &dyn Fn(f32, f32) -> Vec2,
    drawx: f32,
    drawy: f32,
    waypoints: &[crate::comps_a::Waypoint],
    pos: usize,
) -> Vec<SpriteInstance> {
    const SEG: f32 = 9.0;
    const MAXSUB: [i32; 8] = [0, 3, 1, 3, 1, 3, 1, 3];
    const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
    const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    let mut out = Vec::new();
    let push =
        |out: &mut Vec<SpriteInstance>, path: &str, frame: i32, x: f32, y: f32, tint: [f32; 4]| {
            if let Some(s) = assets.sprite_for(path, frame, to_world(x, y), false, 0.0, tint) {
                out.push(s);
            }
        };
    push(
        &mut out,
        "images/sprScoreSplat.png",
        2,
        drawx - 68.0,
        drawy - 15.0,
        WHITE,
    );
    push(
        &mut out,
        "images/sprScoreSplat.png",
        2,
        drawx + 8.0,
        drawy - 15.0,
        WHITE,
    );
    push(
        &mut out,
        "images/sprKillsIcon.png",
        0,
        drawx + 14.0,
        drawy - 15.0,
        WHITE,
    );
    let total: f32 = MAXSUB[1..].iter().map(|m| *m as f32 * SEG).sum();
    let x0 = drawx - (total as i32 / 2) as f32;
    let mut map_x = x0;
    for area in 1..=7 {
        let (px, py) = (map_x, drawy);
        push(&mut out, "images/sprMapDot.png", 0, px, py + 2.0, BLACK);
        push(
            &mut out,
            "images/sprMapDot.png",
            0,
            px + 1.0,
            py + 2.0,
            BLACK,
        );
        push(
            &mut out,
            "images/sprMapDot.png",
            0,
            px + 1.0,
            py + 1.0,
            BLACK,
        );
        map_x += MAXSUB[area as usize] as f32 * SEG;
        // GML background hatch verbatim: 3 black + 1 white.
        pixel_line(
            assets,
            &to_world,
            px,
            py + 1.0,
            map_x,
            drawy + 1.0,
            BLACK,
            &mut out,
        );
        pixel_line(
            assets,
            &to_world,
            px + 1.0,
            py,
            map_x + 1.0,
            drawy,
            BLACK,
            &mut out,
        );
        pixel_line(
            assets,
            &to_world,
            px + 1.0,
            py + 1.0,
            map_x + 1.0,
            drawy + 1.0,
            BLACK,
            &mut out,
        );
        pixel_line(
            assets,
            &to_world,
            px + 1.0,
            py,
            map_x,
            drawy,
            WHITE,
            &mut out,
        );
        if area == 7 {
            // GML draws the crown first, then the black pixel only when
            // the palace holds more than one subarea (always, 3 > 1).
            push(
                &mut out,
                "images/sprMapCrown.png",
                0,
                map_x - 3.0,
                drawy + 1.0,
                WHITE,
            );
            if MAXSUB[7] > 1 {
                push(
                    &mut out,
                    "images/sprPixel.png",
                    0,
                    map_x - 8.0,
                    drawy + 1.0,
                    BLACK,
                );
            }
        }
        push(&mut out, "images/sprMapDot.png", 0, px, py + 1.0, WHITE);
    }
    let odd_len = MAXSUB[1] as f32 * SEG;
    let even_len = MAXSUB[2] as f32 * SEG;
    let wpx = |area_mod: i32, sub: u32| {
        let even = area_mod.div_euclid(2);
        let odd = area_mod - even;
        x0 + ((odd - 1) as f32 * even_len + even as f32 * odd_len + (sub as i32 - 1) as f32 * SEG)
    };
    let count = pos.min(waypoints.len());
    for pass in 0..2 {
        let mut cur_loop: Option<u32> = None;
        let (mut mx, mut my) = (x0, drawy);
        for wp in waypoints.iter().take(count) {
            let secret = wp.area >= 100;
            let area_mod = wp.area % 100;
            if cur_loop != Some(wp.lp) {
                cur_loop = Some(wp.lp);
                mx = x0;
                my = drawy;
            }
            let (px, py) = (mx, my);
            if !secret {
                mx = wpx(area_mod, wp.sub);
            }
            my = drawy + (SEG + 1.0) * secret as u32 as f32;
            let tint = if pass == 0 {
                BLACK
            } else {
                roadmap_loop_tint(wp.lp)
            };
            if wp.sub == 1 {
                // GML shadow pass always draws `sprMapDotOut` black;
                // the color pass picks `Out` for secrets, `Dot` else.
                let dot = if pass == 0 || secret {
                    "images/sprMapDotOut.png"
                } else {
                    "images/sprMapDot.png"
                };
                push(&mut out, dot, 0, mx, my + 1.0, tint);
            }
            if pass == 0 {
                // GML waypoint shadow verbatim: 3 black lines.
                pixel_line(
                    assets,
                    &to_world,
                    px + 1.0,
                    py + 1.0,
                    mx + 1.0,
                    my + 1.0,
                    BLACK,
                    &mut out,
                );
                pixel_line(
                    assets,
                    &to_world,
                    px + 2.0,
                    py,
                    mx + 1.0,
                    my,
                    BLACK,
                    &mut out,
                );
                pixel_line(
                    assets,
                    &to_world,
                    px + 2.0,
                    py + 1.0,
                    mx + 1.0,
                    my + 1.0,
                    BLACK,
                    &mut out,
                );
            } else {
                pixel_line(
                    assets,
                    &to_world,
                    px + 1.0,
                    py,
                    mx + 1.0,
                    my,
                    tint,
                    &mut out,
                );
            }
        }
    }
    out
}

/// Final waypoint cursor after the counted prefix (GML `scrDrawRoadmap`
/// `_map_x/_map_y` verbatim): replays the shadow-pass position walk so
/// the player `sprMapIcon`s land where GML draws them — on the last
/// counted waypoint, `drawy + 10` on the secret row.
#[allow(unused_assignments)]
pub fn roadmap_cursor_pos(
    waypoints: &[crate::comps_a::Waypoint],
    pos: usize,
    drawx: f32,
    drawy: f32,
) -> (f32, f32) {
    const SEG: f32 = 9.0;
    const MAXSUB: [i32; 8] = [0, 3, 1, 3, 1, 3, 1, 3];
    let total: f32 = MAXSUB[1..].iter().map(|m| *m as f32 * SEG).sum();
    let x0 = drawx - (total as i32 / 2) as f32;
    let odd_len = MAXSUB[1] as f32 * SEG;
    let even_len = MAXSUB[2] as f32 * SEG;
    let (mut mx, mut my) = (x0, drawy);
    let mut cur_loop: Option<u32> = None;
    for wp in waypoints.iter().take(pos.min(waypoints.len())) {
        let secret = wp.area >= 100;
        let area_mod = wp.area % 100;
        if cur_loop != Some(wp.lp) {
            cur_loop = Some(wp.lp);
            mx = x0;
            my = drawy;
        }
        if !secret {
            let even = area_mod.div_euclid(2);
            let odd = area_mod - even;
            mx = x0
                + ((odd - 1) as f32 * even_len
                    + even as f32 * odd_len
                    + (wp.sub as i32 - 1) as f32 * SEG);
        }
        my = drawy + (SEG + 1.0) * secret as u32 as f32;
    }
    (mx, my)
}

/// Menu art sprites (`Menu/Draw_0`, portrait, loadout, logo, and
/// game-over splats verbatim where the UI framework cannot draw them):
/// char pods (`sprCharSelect` frame = race, locked gray), GO button,
/// selected-race portrait + splat, logo, and game-over splats.
/// Positions are GML view px (`Menu/Draw_0` snaps pods/buttons to
/// `view + start`): view-space points map to world through the live
/// view rect (GUI == view, [`hud_gui_map`] identity).
pub fn menu_sprites(
    kind: crate::MenuOverlay,
    world: &mut World,
    assets: &RenderAssets,
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam: &Camera2d,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    let view = view_rect_world(canvas_dp, world_size, cam);
    let gm = hud_gui_map(view);
    let vw = view[2];
    let gui_to_world = |x: f32, y: f32| hud_gui_to_world(gm, view, x, y);
    match kind {
        crate::MenuOverlay::Splash => {}
        crate::MenuOverlay::MainMenu => {
            let vw = view[2];
            let cx = vw * 0.5;
            let cy = 120.0;
            for i in 0..5u32 {
                let gy = cy - 48.0 + i as f32 * 24.0;
                if let Some(s) = assets.sprite_for(
                    "images/sprMainMenuSplat.png",
                    0,
                    gui_to_world(cx, gy),
                    false,
                    0.0,
                    [1.0; 4],
                ) {
                    out.push(s);
                }
            }
        }
        crate::MenuOverlay::Title => {
            let menu = world.get_resource::<MenuState>().cloned();
            let save = world
                .get_resource::<crate::savedata_part::SaveData>()
                .cloned();
            let selected = world
                .get_resource::<SelectedCharacter>()
                .map(|s| s.0 as usize)
                .unwrap_or(1);
            let (cursor, go_visible, loadout_open) = menu
                .as_ref()
                .map(|m| (m.title_cursor, m.title_go_visible, m.loadout_open))
                .unwrap_or((1, false, false));
            let slot_h = assets
                .native_size("images/sprCharSelect.png")
                .map(|s| s.y)
                .unwrap_or(20.0);
            let roster = crate::state::menus::visible_roster(save.as_ref());
            for (i, pos) in char_pod_layout([vw, 240.0], roster.len(), slot_h)
                .iter()
                .enumerate()
            {
                let race = roster[i];
                // GML `CharSelect/Draw_0` verbatim: `can =
                // scr_race_is_unlocked(race) || UberCont.weekly_run`,
                // `draw_sprite_ext(can ? sprite_index :
                // sprCharSelectLocked, race, x, y, 1, 1, 0, color, 1)`
                // with `color = (can && selected) ? c_white : c_gray`
                // (`c_gray` = 128,128,128 — NOT half-alpha). Pods are
                // view-snapped (`x = view_xview + xstart`).
                let weekly = menu.as_ref().is_some_and(|m| m.weekly_run_menu);
                let can = save.as_ref().is_some_and(|s| s.race_unlocked(race)) || weekly;
                let selected_race = CHAR_SELECT_ORDER[selected.min(CHAR_SELECT_ORDER.len() - 1)];
                let is_selected = i == cursor || race == selected_race;
                let (path, tint): (&str, [f32; 4]) = if !can {
                    ("images/sprCharSelectLocked.png", [0.5, 0.5, 0.5, 1.0])
                } else if is_selected {
                    ("images/sprCharSelect.png", [1.0; 4])
                } else {
                    ("images/sprCharSelect.png", [0.5, 0.5, 0.5, 1.0])
                };
                if let Some(s) = assets.sprite_for(
                    path,
                    race as i32,
                    gui_to_world(pos[0], pos[1]),
                    false,
                    0.0,
                    tint,
                ) {
                    out.push(s);
                }
                // GML `Menu/Draw_74` verbatim: unlocked non-Random pods
                // with no recorded death (`!UberCont.ctot_dead[race]`)
                // draw `sprNew` at the pod's right edge. The port tracks
                // deaths via `total_deaths > 0`; fresh saves (no deaths)
                // show the badge, exactly like a new GML profile.
                let fresh = save.as_ref().is_none_or(|s| s.total_deaths == 0);
                if can && race != crate::data::RaceId::Random && fresh {
                    if let Some(s) = assets.sprite_for(
                        "images/sprNew.png",
                        -1,
                        gui_to_world(pos[0] + TITLE_POD_W, pos[1]),
                        false,
                        0.0,
                        [1.0; 4],
                    ) {
                        out.push(s);
                    }
                }
            }
            // Selected-race portrait + splat + name plate (GML
            // scrCampfireMenuDrawRacePortrait/CharText: headless
            // chicken, hooded rebel, skin subimages, char splat,
            // big-name art).
            let race = CHAR_SELECT_ORDER[selected.min(CHAR_SELECT_ORDER.len() - 1)];
            let skin = save
                .as_ref()
                .map(|s| s.race_loadout(race).preferred_skin)
                .unwrap_or(0);
            let area = world
                .get_resource::<Run>()
                .map(|r| r.area)
                .unwrap_or(AreaId::Desert);
            let hp: Option<i32> = world
                .query::<(&RaceState, &Health)>()
                .iter(world)
                .find(|(rs, _)| rs.race == race)
                .map(|(_, h)| h.hp);
            // GML front-layer portrait origin: `_portrait_x = -2 + 18`,
            // drawn at `_x + 16 - 18` = view-relative -2
            // (`scrCampfireMenuCreate.gml:339,392`); y = H - 36 - 8 + 44.
            // GML slides it by `Menu.portrait_offsets[0]` on select.
            let slide = menu.as_ref().map(|m| m.portrait_offsets[0]).unwrap_or(0.0);
            let portrait_dp = [-2.0 - slide, 240.0 - 36.0 - 8.0 + 44.0];
            if let Some((path, frame)) = big_portrait_for(race, skin, hp, area) {
                if let Some(s) = assets.sprite_for(
                    path,
                    frame,
                    gui_to_world(portrait_dp[0], portrait_dp[1]),
                    false,
                    0.0,
                    [1.0; 4],
                ) {
                    out.push(s);
                }
            }
            // GML `Menu.splatindex` verbatim (0→3 at 0.4/step, ticked in
            // `tick_title_anim`); falls back to the deterministic per-race
            // pin headless (no MenuState yet).
            let splat_frames = strip_frames(assets, "images/sprCharSplat.png");
            let splat_frame = menu
                .as_ref()
                .map(|m| m.splatindex.floor().clamp(0.0, 3.0) as i32)
                .unwrap_or_else(|| char_splat_frame(race, splat_frames));
            let splat_frame = if splat_frames == 0 {
                splat_frame
            } else {
                splat_frame.clamp(0, splat_frames as i32 - 1)
            };
            if let Some(s) = assets.sprite_for(
                "images/sprCharSplat.png",
                splat_frame,
                gui_to_world(0.0, 240.0 - 36.0 + 1.0),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            // Big-name sprite: GML draws it ONLY when the name is
            // unlocalized AND multiplayer
            // (`scrCampfireMenuDrawCharText:444`); single-player always
            // uses the bigname text (the overlay lines), so no sprite
            if go_visible {
                let dp = go_button_pos([vw, 240.0], roster.len(), 19.0);
                if let Some(s) = assets.sprite_for(
                    "images/sprGoButtonSymbolic.png",
                    0,
                    gui_to_world(dp[0], dp[1]),
                    false,
                    0.0,
                    [1.0; 4],
                ) {
                    out.push(s);
                }
            }
            // Loadout grid (`scrMenuDrawLoadout` verbatim geometry, open
            // frame, no tooltips/animation) plus the closed-frame preview
            // (splat + arrow + minis ride the closed frame in GML).
            if loadout_open && selected != 0 {
                menu_loadout_sprites(
                    world,
                    assets,
                    [vw, 240.0],
                    selected,
                    &|p| gui_to_world(p[0], p[1]),
                    &mut out,
                );
            } else if !loadout_open && selected != 0 {
                menu_loadout_closed_sprites(
                    world,
                    assets,
                    [vw, 240.0],
                    selected,
                    loadout_available_for_race(race),
                    &|p| gui_to_world(p[0], p[1]),
                    &mut out,
                );
            }
        }
        crate::MenuOverlay::GameOver => {
            // GML `GameOver/Draw_0` sprite layer verbatim: roadmap at
            // `(_x - 48, _y - offsety)` with prefix `round(death_pos)`,
            // `sprKilledBySplat[splatimg]` at `(_x + 86,
            // _y - offsety - 32)`, `sprGameOverCenterSplat[splatimg]` at
            // `(_x, view_bottom - 32)`. `_x/_y` is the view center
            // (`vw/2`, 120 at 240 high).
            let (offsety, death_pos, splat) = world
                .get_resource::<MenuState>()
                .map(|m| (m.go_offsety, m.go_death_pos, m.go_splat))
                .unwrap_or((0.0, 0.0, 0.0));
            let splat_frame = splat.floor().clamp(0.0, 2.0) as i32;
            let cx = vw * 0.5;
            if let Some(s) = assets.sprite_for(
                "images/sprGameOverCenterSplat.png",
                splat_frame,
                gui_to_world(cx, 240.0 - 32.0),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            if let Some(s) = assets.sprite_for(
                "images/sprKilledBySplat.png",
                splat_frame,
                gui_to_world(cx + 86.0, 120.0 - offsety - 32.0),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            if let Some(run) = world.get_resource::<Run>() {
                let wps = run.waypoints.clone();
                let prefix = death_pos.round() as usize;
                out.extend(roadmap_sprites(
                    assets,
                    &gui_to_world,
                    cx - 48.0,
                    120.0 - offsety,
                    &wps,
                    prefix,
                ));
                // GML `GameOver/Draw_0` deathcause icon verbatim:
                // `draw_sprite(scrDeathCauseGetSprite(cause), -1,
                // _x+86, _y-offsety)` — `-1` rides the GameOver
                // `image_speed = 0.4`, i.e. `floor(death_pos * 0.4)`
                // over the strip frames here.
                if let Some(path) = world
                    .get_resource::<MenuState>()
                    .and_then(|m| m.game_over)
                    .and_then(|g| g.deathcause_sprite)
                {
                    let frames = strip_frames(assets, path).max(1) as f32;
                    let frame = ((death_pos * 0.4).floor() % frames) as i32;
                    if let Some(s) = assets.sprite_for(
                        path,
                        frame,
                        gui_to_world(cx + 86.0, 120.0 - offsety),
                        false,
                        0.0,
                        [1.0; 4],
                    ) {
                        out.push(s);
                    }
                }
                // GML `scrDrawRoadmap:158-197` player icons verbatim:
                // `sprMapIcon[skin]` at the final cursor (secret row
                // included), `sprMapIconChickenHeadless[skin]` for a
                // dead Chicken, `sprMapIconRebelBHooded` for a B-skin
                // Rebel in the city (GML `area_city` = port
                // `FrozenCity`). Single-player draws one icon with no
                // co-op offset.
                let (cursor_x, cursor_y) =
                    roadmap_cursor_pos(&wps, prefix, cx - 48.0, 120.0 - offsety);
                let players: Vec<(RaceId, u8, i32, AreaId)> = {
                    let area = run.area;
                    let save = world
                        .get_resource::<crate::savedata_part::SaveData>()
                        .cloned();
                    world
                        .query::<(&crate::comps_a::RaceState, &Health)>()
                        .iter(world)
                        .filter(|(rs, _)| rs.race != crate::data::RaceId::Random)
                        .map(|(rs, h)| {
                            let skin = save
                                .as_ref()
                                .map(|s| s.race_loadout(rs.race).preferred_skin)
                                .unwrap_or(0);
                            (rs.race, skin, h.hp, area)
                        })
                        .collect()
                };
                for (race, skin, hp, area) in players {
                    let (path, frame) = if race == RaceId::Chicken && hp <= 0 {
                        ("images/sprMapIconChickenHeadless.png", skin as i32)
                    } else if race == RaceId::Rebel && skin == 1 && area == AreaId::FrozenCity {
                        ("images/sprMapIconRebelBHooded.png", 0)
                    } else {
                        (
                            "images/sprMapIcon.png",
                            crate::state::menus::race_skin_subimage(race as usize, skin),
                        )
                    };
                    if frame < 0 {
                        continue;
                    }
                    if let Some(s) = assets.sprite_for(
                        path,
                        frame,
                        gui_to_world(cursor_x, cursor_y),
                        false,
                        0.0,
                        [1.0; 4],
                    ) {
                        out.push(s);
                    }
                }
            }
        }
        crate::MenuOverlay::Loading => {
            if let Some(run) = world.get_resource::<Run>() {
                let n = run.waypoints.len();
                let wps = run.waypoints.clone();
                out.extend(roadmap_sprites(
                    assets,
                    &gui_to_world,
                    vw * 0.5,
                    120.0,
                    &wps,
                    n,
                ));
            }
        }
        // GML `LevCont/Draw_0` calls `scrDrawSpiral()` (opaque clear)
        // then draws ONLY the offer chrome: title/subtitle text plus
        // the SkillIcon/UltraIcon/CrownIcon cards. No camp pods, no
        // roadmap, no PlayerHUD — those rooms never had them. The text
        // half lives in `menu_gui_texts(Mutation)`; the cards compose
        // here so they sit under the opaque cover pass with the rest of
        // the viewport.
        crate::MenuOverlay::Mutation => {
            let cx = vw * 0.5;
            let selected = world
                .get_resource::<MenuState>()
                .and_then(|m| m.mutation_selected);
            let ultra = world.get_resource::<PendingUltra>();
            let skill = world.get_resource::<PendingMutation>();
            let is_ultra = ultra.is_some() && skill.is_none();
            let title_path = if is_ultra {
                "images/sprLevelUltraText.png"
            } else {
                "images/sprLevelUpText.png"
            };
            if let Some(s) = assets.sprite_for(
                title_path,
                2,
                gui_to_world(cx, 48.0),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            if let Some(s) = assets.sprite_for(
                "images/sprMutationSplat.png",
                0,
                gui_to_world(cx, 240.0 - 31.0),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            let selected_race = world
                .get_resource::<SelectedCharacter>()
                .map(|s| s.0)
                .unwrap_or(crate::data::RaceId::Fish);
            let ids: Vec<i32> = if is_ultra {
                ultra
                    .map(|u| {
                        u.choices.iter().map(|c| ultra_hud_frame(selected_race, *c)).collect()
                    })
                    .unwrap_or_default()
            } else {
                skill
                    .map(|p| {
                        p.choices.iter().map(|c| mutation_hud_frame(*c)).collect()
                    })
                    .unwrap_or_default()
            };
            let icon_path = "images/sprSkillIcon.png";
            // GML `LevCont/Other_10` icon row: `y = view_yview +
            // view_height - 21`, `step = min(32, floor(view_w / (n+1)))`,
            // centered on the view (`-12` nudge at 10+). Scale holds at
            // `max(0.65, step/32)`; the port draws native (steady-state
            // rows never exceed ~6 cards, so step is 32).
            let n = ids.len();
            let step = if n == 0 {
                32.0
            } else {
                (32.0f32).min((vw / (n as f32 + 1.0)).floor())
            };
            for (i, frame) in ids.iter().enumerate() {
                let half = step / 2.0;
                let xoff = if n >= 10 { -12.0 } else { 0.0 };
                let x = cx + xoff - (n as f32 - 1.0) * half + i as f32 * step;
                let tint = if Some(i) == selected {
                    [1.0; 4]
                } else {
                    [0.5, 0.5, 0.5, 1.0]
                };
                if let Some(s) = assets.sprite_for(
                    icon_path,
                    *frame,
                    gui_to_world(x, 240.0 - 21.0),
                    false,
                    0.0,
                    tint,
                ) {
                    out.push(s);
                }
            }
        }
        crate::MenuOverlay::Pause => {
            // GML `UberCont/Draw_0` paused branch sprite layer verbatim:
            // frozen `pausespr` screenshot (shell-owned surface, not
            // drawn here) + corner `sprCharSplat` pair at `(view_left,
            // view_bottom-31)` / mirrored at `(view_right,
            // view_bottom-31)` + FULL roadmap at `(view_center,
            // view_center)` (`pos = 1000`, i.e. the whole run). The
            // `PAUSED` bigname + buttons ride the text layer. Ordered
            // campfire portraits need the room actors the port
            // despawns on pause, so only the splat pair + roadmap draw
            // here.
            let cx = vw * 0.5;
            if let Some(s) = assets.sprite_for(
                "images/sprCharSplat.png",
                0,
                gui_to_world(0.0, 240.0 - 31.0),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            if let Some(s) = assets.sprite_for(
                "images/sprCharSplat.png",
                0,
                gui_to_world(vw, 240.0 - 31.0),
                true,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            if let Some(run) = world.get_resource::<Run>() {
                let wps = run.waypoints.clone();
                out.extend(roadmap_sprites(
                    assets,
                    &gui_to_world,
                    cx,
                    120.0,
                    &wps,
                    1000,
                ));
            }
        }
        _ => {}
    }
    out
}

/// Spiral center-figure layout (`scripts/scrDrawSpiral` CPU layer
/// verbatim): `(sprite path, frame, center, scale, rotation radians)`.
/// `angle_deg` is the spiral angle in degrees. Player `n` staggers the
/// angle by `n * 50` (GML mutates `image_angle` around each draw).
pub fn spiral_figure_layout(
    center: Vec2,
    angle_deg: f32,
    crown: CrownKind,
    hurts: &[String],
) -> Vec<(String, i32, Vec2, f32, f32)> {
    let mut out = Vec::new();
    let deg = std::f32::consts::PI / 180.0;
    // GML degree trig (`sin`/`cos` take degrees); `deg_sin`/`deg_cos`
    // keep the conversion explicit so raw `.sin()` on degree values
    // never slips in.
    let deg_sin = |d: f32| (d * deg).sin();
    if crown != CrownKind::None {
        let gml = crate::savedata_part::crown_port_to_gml(crown as u8);
        // GML `lengthdir_x(r, a) = r*cos(a)`, `lengthdir_y(r, a) =
        // -r*sin(a)` (y-down, 90 = north): the y term negates.
        let r = 15.0 + deg_sin(angle_deg / 60.0) * 4.0;
        let a = -angle_deg / 5.3;
        out.push((
            format!("images/sprCrown{gml}Idle.png"),
            1,
            center + Vec2::new((a * deg).cos() * r, -(a * deg).sin() * r),
            0.6 + deg_sin(angle_deg / 200.0) / 4.0,
            -angle_deg * 2.2 * deg,
        ));
    }
    for (n, hurt) in hurts.iter().enumerate() {
        if hurt.is_empty() {
            continue;
        }
        let a = angle_deg + n as f32 * 50.0;
        out.push((
            hurt.clone(),
            1,
            center + Vec2::new(n as f32 * 8.0, n as f32 * 6.0),
            0.8 + deg_sin(a / 200.0) / 5.0,
            -a * 2.0 * deg,
        ));
    }
    out
}

/// Spiral center figures (`scripts/scrDrawSpiral` CPU layer verbatim):
/// the carried crown orbiting the spiral center plus one hurt-frame
/// player figure per player, over the vortex background. `center` is the
/// view center (GML `fishx/fishy`), `angle_deg` the spiral angle in
/// degrees (GML `image_angle`; port `SpiralCtl.angle` is degrees too).
/// Skipped while Throne II lives (GML `Nothing2` gate: Nothing2,
/// Nothing2Corpse and Nothing2Death all suppress the figures — the port
/// reads it off any live Throne-II enemy) or while `Credits` runs
/// without a crown carrier (GML `!instance_exists(Credits)` gate).
pub fn spiral_figures(
    world: &mut World,
    assets: &RenderAssets,
    center: Vec2,
    angle_deg: f32,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();

    // GML `scrDrawSpiral` figure gate verbatim: nothing draws while
    // Throne II runs (`Nothing2`/`Nothing2Corpse`/`Nothing2Death` — the
    // port reads it off any live Throne-II enemy), and the whole block
    // (crown + players) is inside `!instance_exists(Credits)`.
    let throne_ii_alive = world
        .query::<&Enemy>()
        .iter(world)
        .any(|e| e.kind == EnemyKind::ThroneII);
    if throne_ii_alive {
        return out;
    }
    // GML `with SpiralCont` figure block sits inside
    // `if !instance_exists(Credits)`: the credits spiral is bare.
    let in_credits = world
        .get_resource::<OverlayMenu>()
        .is_some_and(|o| *o == OverlayMenu::Credits);
    if in_credits {
        return out;
    }

    let mut crown = CrownKind::None;
    let mut hurts: Vec<String> = Vec::new();
    {
        let mut q = world.query::<(&Player, Option<&PlayerAnim>)>();
        for (player, anim) in q.iter(world) {
            crown = player.crown;
            hurts.push(anim.map(|pa| pa.hurt.to_string()).unwrap_or_default());
        }
    }
    if hurts.is_empty() {
        return out;
    }

    for (path, frame, at, scale, rot) in spiral_figure_layout(center, angle_deg, crown, &hurts) {
        if let Some(s) = assets.sprite_scaled_rotated(&path, frame, at, scale, rot, [1.0; 4]) {
            out.push(s);
        }
    }
    out
}

/// Damage numbers ride [`fx_texts`] instead: [`SpriteInstance`] cannot
/// carry text, so [`DamageNumber`] comps resolve to [`WorldText`] for
/// the repose text overlay.
pub fn fx_instances(world: &mut World, assets: &RenderAssets) -> Vec<SpriteInstance> {
    let mut out = Vec::new();

    // Beams: oriented strip quads spanning the sim segment. `Pos` is
    // the segment center (`spawn_beam_shot`/`boss_ai` parity), so the
    // quad stays center-anchored (the strip's own `[0, 0.5]` anchor is
    // overridden) and `size.x = length` spans the endpoints. Tint is
    // the bevy `BeamSpec`/sprite color (ion cyan, laser red, boss
    // orange/green), carried on the sim `Beam`.
    {
        let mut q = world.query::<(&Pos, &Beam)>();
        for (pos, beam) in q.iter(world) {
            let rotation = beam.dir.y.atan2(beam.dir.x);
            let size = Vec2::new(beam.length.max(1.0), beam.width.max(2.0));
            let tint = beam.color;
            let mut s = match assets.sprite_for(BEAM_STRIP, 0, pos.0, false, rotation, tint) {
                Some(s) => s,
                None => white_quad(pos.0, rotation, size, tint),
            };
            s.size = size;
            s.anchor = Vec2::new(0.5, 0.5);
            out.push(s);
        }
    }

    // Lightning arcs: one stretched quad per segment (`Pos` = midpoint,
    // `len`/`angle` on the marker). Strip frame follows the elapsed
    // fraction; alpha follows bevy `tick_lightning_arcs`
    // (`(1 - t) * 0.9`, no floor) on the arc color.
    {
        let mut q = world.query::<(&Pos, &LightningArc)>();
        for (pos, arc) in q.iter(world) {
            let frac = arc.timer.fraction();
            let frame = ((frac * 4.0).floor() as i32).clamp(0, 3);
            let tint = [0.75, 0.95, 1.0, (1.0 - frac).clamp(0.0, 1.0) * 0.9];
            let size = Vec2::new(arc.len.max(1.0), 3.0);
            let mut s = match assets.sprite_for(ARC_STRIP, frame, pos.0, false, arc.angle, tint) {
                Some(s) => s,
                None => white_quad(pos.0, arc.angle, size, tint),
            };
            s.size = size;
            s.anchor = Vec2::new(0.5, 0.5);
            out.push(s);
        }
    }

    // Muzzles: no muzzle visual exists in GML or bevy — fire feedback
    // is the yellow `muzzle_burst` particle spray (spawned sim-side)
    // plus gun wkick. `FiredWeapon` markers expire silently.

    // Hazard clouds: static translucent kind-tinted rects (bevy spawns
    // the cloud `Sprite` with the spec color, no pulse; the ring strip
    // has no catalog entry here, so ring degrades to a disc).
    {
        let mut q = world.query::<(&Pos, &HazardCloud)>();
        for (pos, cloud) in q.iter(world) {
            let rgb = match cloud.kind {
                HazardKind::Fire => [1.0, 0.43, 0.10],
                HazardKind::Toxic => [0.30, 0.88, 0.30],
            };
            let d = cloud.radius.max(4.0) * 2.0;
            out.push(white_quad(
                pos.0,
                0.0,
                Vec2::new(d, d),
                [rgb[0], rgb[1], rgb[2], 0.36],
            ));
        }
    }

    // Particles: gradient color at life fraction, shrinking along the
    // ease curve (repame-fx parity, same convention as the fallback).
    // Gradient stops are sRGB-authored like every other tint.
    {
        let mut q = world.query::<&Particle>();
        out.extend(particle_sprites(q.iter(world)).into_iter().map(|mut s| {
            s.color = tint_to_linear(s.color);
            s
        }));
    }

    out
}

/// Damage-number floaters as world-anchored text: every live
/// [`DamageNumber`] (spawned via `repame_fx::spawn_number` in combat /
/// pickups) resolves here for the repose `Text` overlay, through the
/// same world→dp law as [`hud_texts_dp`]. Already the render path for
/// numbers — no separate sprite mapping.
pub fn fx_texts(world: &mut World) -> Vec<WorldText> {
    let mut q = world.query::<&DamageNumber>();
    q.iter(world)
        .map(|n| WorldText {
            text: n.text.clone(),
            pos: Vec2::new(n.x, n.y),
            color: n.color,
            size: NUMBER_TEXT_SIZE,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// HUD labels.
// ---------------------------------------------------------------------------

/// World-anchored HUD labels for the repose `Text` overlay: run extras
/// with no bevy `nt_hud_overlay` equivalent (toast, floor/score/kills,
/// GML clock + map name, boss bar, IDPD warning, game-over line, weapon
/// pickup label). The core HP/level/ammo/LOW-HP rows live in
/// [`hud_gui_texts`] (bevy content + positions verbatim) and are NOT
/// duplicated here.
pub fn hud_texts(world: &mut World) -> Vec<(String, [f32; 2])> {
    let hud: HudState = sync_hud_state(world);
    let player_pos = world
        .query::<(&Pos, &Player)>()
        .iter(world)
        .next()
        .map(|(p, _)| p.0);
    let mut out = Vec::new();
    if let Some(pp) = player_pos {
        if !hud.toast.is_empty() {
            out.push((hud.toast.clone(), [pp.x, pp.y - 48.0]));
        }
    }
    out.push((
        format!(
            "FLOOR {}  SCORE {}  KILLS {}",
            hud.floor, hud.score, hud.kills
        ),
        [-ARENA_W / 2.0 + 16.0, -ARENA_H / 2.0 + 8.0],
    ));
    // GML `scrDrawMiscHUD` bottom-right rows: timer + map name.
    if !hud.timer_string.is_empty() {
        out.push((
            hud.timer_string.clone(),
            [ARENA_W / 2.0 - 16.0, ARENA_H / 2.0 - 8.0],
        ));
    }
    if !hud.area_string.is_empty() {
        out.push((
            hud.area_string.clone(),
            [ARENA_W / 2.0 - 16.0, ARENA_H / 2.0 - 24.0],
        ));
    }
    if hud.boss_max > 0 {
        out.push((
            format!("{} {}/{}", hud.boss_name, hud.boss_hp, hud.boss_max),
            [0.0, -ARENA_H / 2.0 + 8.0],
        ));
    }
    if hud.idpd_warning {
        out.push((
            "!! IDPD INCOMING !!".to_string(),
            [0.0, -ARENA_H / 2.0 + 28.0],
        ));
    }
    if hud.game_over {
        out.push((
            format!("GAME OVER  SCORE {}  BEST {}", hud.score, hud.high_score),
            [0.0, 0.0],
        ));
    }
    // GML `scrDrawInteractionHUD` verbatim (nearest-pickup half):
    // the nearest weapon pickup's name at the pickup `+(0,-31)` in
    // room px (world-anchored here; the view subtracts the camera).
    // GML draws at `(floor(x - view_xview), floor(y - view_yview) - 31)`
    // = room pos - 31 px, i.e. world `[gun.x, gun.y - 31]`. (The ammo
    // gauge + touch `ButtonAct` ring ride the deferred sprite pass.)
    if let Some(label) = world.get_resource::<crate::pickups::WeaponLabel>() {
        if let Some(target) = label.target {
            if let Some(gun) = world.get::<Pos>(target) {
                if !label.text.is_empty() {
                    out.push((label.text.clone(), [gun.0.x, gun.0.y - 31.0]));
                }
            }
        }
    }
    out
}

/// [`hud_texts`] mapped to dp canvas points through the live camera
/// fit ([`effective_fit`], same [`world_to_dp`] law the canvas/GPU
/// viewports paint with, so world-anchored floaters sit on their
/// sprites instead of drifting when the camera zooms).
pub fn hud_texts_dp(
    world: &mut World,
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam: &Camera2d,
) -> Vec<(String, [f32; 2])> {
    let center = cam.effective_center();
    let fit = effective_fit(canvas_dp, world_size, cam);
    hud_texts(world)
        .into_iter()
        .map(|(t, p)| (t, world_to_dp(p, world_size, center, fit)))
        .collect()
}

// ---------------------------------------------------------------------------
// Fidelity notes.
// ---------------------------------------------------------------------------
//
// Compromises vs the bevy build (all art-name level; sim truth kept):
// - Wall quads composite Out/Bot/Top + Trans with the seeded bevy
//   variant frames and GM-origin placement, but top decals
//   (`sprNightDesert…`) are not drawn. The darkened outside-floor ring
//   (`sprFloorEx1` + 0.45 tint) is drawn.
// - Ground decals draw the route floor's top-decal strip (bevy
//   `area_sprites` 6th column) at gray 0.5 via `GroundDecalTint`.
// - Projectile strips follow the bevy frame law (`projectile_frame`:
//   2-frame strips pin to the second cell, longer strips animate at
//   12 fps off the projectile's life clock).
// - Projectile custom sizes (plasma growth, bolt stretch) are skipped:
//   quads use catalog cell size (pickups too: bevy `spawn_pickup`
//   ignores `pickup_sprite`'s sizes and draws native strip frames).
// - Chest idle shimmer rides the live `SpriteAnim` frame (bevy
//   attaches the strip the same way); opened chests draw the
//   kind-specific open art frozen on bevy's last frame.
// - Transient FX ride `fx_instances` / `fx_texts`: beams stretch
//   `sprLightBeam` along the sim segment, arcs stretch `sprLightning`
//   (frame follows life), muzzles are warm tinted quads (no muzzle
//   strip exists in the catalog), hazards are kind-tinted translucent
//   discs (the bevy ring sprite has no strip here, so ring → disc),
//   particles map 1:1 through `particle_sprites`, and damage numbers
//   resolve as `WorldText` (sprites cannot carry text).
// - Beam tint rides the sim `Beam.color` (bevy `BeamSpec`/sprite
//   parity: ion cyan, laser red, boss orange/green).
// - Hazard discs ignore per-cloud alpha: alpha pulses on the sim clock
//   instead of the cloud's own timer fraction.
// - Portal bodies draw their live strip (spawn/idle/disappear swaps
//   ride `animate_portal`); shock/clear/strike draw 1-frame-per-step
//   strips in the hit-FX pass. The vortex background itself is not a
//   sprite (it is the fullscreen `crate::vortex_pass::VortexPass`).
// - Dynamic art paths bevy built at runtime (`weapon_id_sprite` falls
//   back to revolver here too when `wep_sprt` is `mskNone`/absent, and
//   secret tile families fall back to route strips when the pack lacks
//   `sprFloor10x` — same `has` law, smaller pack).
// - Camera follow is the GML `BackCont` law verbatim
//   (`gml_camera_step`: POI pull /6 cap 72, aim lean dis/viewdist,
//   orandom shake, snap, round, knock decay; view-layer state in
//   `App`); the only deviation is no per-run snap reset beyond boot
//   and floor starts, so transitions don't re-swoop from the origin.

#[cfg(test)]
mod verbatim_ui_layers {
    use super::*;
    use crate::state::menus::MenuState;

    /// Reported bug verbatim: HUD bars/icons rendered behind floor
    /// ground. Every sprite used to share z=0, so the engine's
    /// `(blend, z, page)` sort let atlas page decide — floor tiles won
    /// over the health bar whenever their page sorted later. The
    /// z-ladder (GML `__global_object_depths` order) keeps GUI chrome
    /// above world chrome on every backend.
    #[test]
    fn z_ladder_orders_chrome_above_world() {
        assert!(Z_SHADOW < Z_WORLD);
        assert!(Z_WORLD < Z_FX);
        assert!(Z_FX <= Z_BLOOM);
        assert!(Z_BLOOM < Z_FOG);
        assert!(Z_FOG < Z_CROSSHAIR);
        assert!(Z_CROSSHAIR < Z_FAINTED);
        assert!(Z_FAINTED < Z_PORTAL_INDICATOR);
        assert!(Z_PORTAL_INDICATOR < Z_SPIRAL_FIGURES);
        assert!(Z_SPIRAL_FIGURES < Z_HUD);
        assert!(Z_HUD < Z_SPLASH);
        assert!(Z_SPLASH < Z_MENU);
        assert!(Z_MENU < Z_SIDEART);
    }

    /// `stamp_z` assigns the rung without disturbing intra-layer push
    /// order (the engine sort is stable on ties).
    #[test]
    fn stamp_z_assigns_rung_only() {
        let mut v = vec![SpriteInstance::default(), SpriteInstance::default()];
        stamp_z(&mut v, Z_HUD);
        assert!(v.iter().all(|s| s.z == Z_HUD));
    }

    /// Reported bug verbatim: clicking MENU right after dying did
    /// nothing — the game-over buttons parked a full screen below the
    /// view (`+240`) while `appear` ticked down, so the first frames'
    /// clicks missed. GML `PauseButton/Draw_0` draws at `_dy = y +
    /// appear` from frame one, so the buttons sit at `ystart + offsety
    /// + appear` here too: on-screen and clickable immediately.
    #[test]
    fn gameover_buttons_ride_appear_not_offscreen() {
        let mut world = World::new();
        world.insert_resource(Run {
            area: AreaId::Desert,
            ..Default::default()
        });
        world.init_resource::<MenuState>();
        // Fresh capture: offsety=128, appear=4 (worst case).
        let texts = menu_gui_texts_vw(crate::MenuOverlay::GameOver, &mut world, 426.0);
        let menu = texts.iter().find(|t| t.text == "MENU").expect("MENU row");
        let retry = texts.iter().find(|t| t.text == "RETRY").expect("RETRY row");
        // ystart + offsety + appear: 178+128+4 / 210+128+4.
        assert!((menu.gy - 310.0).abs() < 1.0, "MENU gy {}", menu.gy);
        assert!((retry.gy - 342.0).abs() < 1.0, "RETRY gy {}", retry.gy);
        // Settle the anim: buttons land exactly on the GML ystarts.
        if let Some(mut menu_state) = world.get_resource_mut::<MenuState>() {
            menu_state.go_offsety = 0.0;
            menu_state.go_appear = 0.0;
        }
        let texts = menu_gui_texts_vw(crate::MenuOverlay::GameOver, &mut world, 426.0);
        let menu = texts.iter().find(|t| t.text == "MENU").expect("MENU row");
        let retry = texts.iter().find(|t| t.text == "RETRY").expect("RETRY row");
        assert!((menu.gy - 178.0).abs() < 1.0, "MENU gy {}", menu.gy);
        assert!((retry.gy - 210.0).abs() < 1.0, "RETRY gy {}", retry.gy);
    }
}

#[cfg(test)]
mod ui_parity_regression {
    use super::*;

    /// GML `draw_sprite` honors the strip origin: the draw point is
    /// `pos - origin`. `sprUltraLevel` (origin 4,5) drawn at GML (11,16)
    /// must land pixels at (7,11), not (11,16).
    #[test]
    fn hud_gui_place_uses_strip_origin() {
        let dir = crate::resolve_assets_dir().expect("assets for parity test");
        let assets = RenderAssets::load(&dir).expect("catalog loads");
        let view = [0.0, 0.0, 426.0, 240.0];
        let gm = hud_gui_map(view);
        let s = hud_gui_place(
            &assets,
            "images/sprUltraLevel.png",
            0,
            11.0,
            16.0,
            1.0,
            [1.0; 4],
            gm,
            view,
        )
        .expect("ultra level art present");
        // Top-left = pos - origin = (11-4, 16-5). (Recovered via the
        // anchor law `center - anchor * size` since the strip's anchor
        // is (4/8, 5/8), not the quad middle.)
        let top_left = s.center - Vec2::new(s.anchor.x * s.size.x, s.anchor.y * s.size.y);
        assert!((top_left.x - 7.0).abs() < 1e-4, "x {top_left:?}");
        assert!((top_left.y - 11.0).abs() < 1e-4, "y {top_left:?}");
        // The origin itself lands on the GML draw point (11,16).
        assert!((s.center.x - 11.0).abs() < 1e-4, "cx {:?}", s.center);
        assert!((s.center.y - 16.0).abs() < 1e-4, "cy {:?}", s.center);
        // Zero-origin art is unaffected: health bar still at (20,4).
        let b = hud_gui_place(
            &assets,
            "images/sprHealthBar.png",
            2,
            20.0,
            4.0,
            1.0,
            [1.0; 4],
            gm,
            view,
        )
        .expect("health bar art present");
        let btl = b.center - Vec2::new(b.anchor.x * b.size.x, b.anchor.y * b.size.y);
        assert!((btl.x - 20.0).abs() < 1e-4, "x {btl:?}");
        assert!((btl.y - 4.0).abs() < 1e-4, "y {btl:?}");
    }

    /// GML `scrSetViewSize` verbatim: odd view widths bump +1.
    #[test]
    fn gml_view_size_bumps_odd_widths() {
        // 426.666… floors to 426 (even, unchanged).
        assert_eq!(gml_view_size([1280.0, 720.0]), [426.0, 240.0]);
        // A width flooring to an odd value bumps +1 (e.g. 321 -> 322).
        // 240 * 321/240 = 321 exactly.
        assert_eq!(gml_view_size([321.0, 240.0]), [322.0, 240.0]);
        // Portrait still floors at 320.
        assert_eq!(gml_view_size([200.0, 400.0]), [320.0, 240.0]);
    }

    /// GML `draw_text_nt` block law: one row per visual line (the shell
    /// renders `.single_line()`), so multi-line producers pre-split.
    /// The Title passive/active pair arrives as two rows at gy/gy+8,
    /// never one `\n` row (the literal-`\n` overlap bug).
    #[test]
    fn title_skill_rows_are_two_rows() {
        let mut world = World::new();
        world.insert_resource(crate::comps_a::SelectedCharacter(crate::data::RaceId::Fish));
        let mut menu = crate::state::menus::MenuState::default();
        // Steady state: GML `textappear` approaches 0 after entry (2.0
        // hides the name/skill rows while the portrait slides in).
        menu.textappear = [0.0; 4];
        world.insert_resource(menu);
        let rows = menu_gui_texts_vw(crate::MenuOverlay::Title, &mut world, 426.0);
        let skills: Vec<&MenuGuiText> = rows
            .iter()
            .filter(|r| r.gx == 8.0 && (r.gy == 212.0 || r.gy == 220.0))
            .collect();
        assert_eq!(skills.len(), 2, "passive + active rows: {rows:?}");
        assert!(rows.iter().all(|r| !r.text.contains('\n')));
    }

    /// GML centered-`draw_text_nt` law: the row centers on its own `gx`
    /// (symmetric box around `gx`), so stats column headers sit on
    /// their column — not dragged to the view center — while
    /// view-centered rows still span the full width.
    #[test]
    fn centered_rows_center_on_gx() {
        let dp = gui_texts_dp(
            [1280.0, 720.0],
            vec![
                MenuGuiText {
                    text: "TOTAL".to_string(),
                    gx: 110.0,
                    gy: 40.0,
                    color: [255, 255, 255, 255],
                    px: 7.0,
                    centered: true,
                    middle_y: true,
                    right: false,
                },
                MenuGuiText {
                    text: "PLAY".to_string(),
                    gx: 213.0,
                    gy: 72.0,
                    color: [255, 255, 255, 255],
                    px: 12.0,
                    centered: true,
                    middle_y: true,
                    right: false,
                },
            ],
        );
        // TOTAL box centers on gx=110 (330 dp at k=3).
        let (total_left, total_w) = (dp[0].1[0], dp[0].4);
        assert!((total_left + total_w * 0.5 - 330.0).abs() < 1.0, "{dp:?}");
        // PLAY box centers on the view center (639 dp).
        let (play_left, play_w) = (dp[1].1[0], dp[1].4);
        assert!((play_left + play_w * 0.5 - 639.0).abs() < 1.0, "{dp:?}");
    }

    /// GML `draw_stat` law: the name right-aligns on `statx - 1`, so
    /// the row's box right edge lands on `gx` — short names sit
    /// against the value column instead of floating left.
    #[test]
    fn stat_names_right_align_on_gx() {
        let dp = gui_texts_dp(
            [1280.0, 720.0],
            vec![MenuGuiText {
                text: "kills".to_string(),
                gx: 109.0,
                gy: 40.0,
                color: [153, 153, 153, 255],
                px: 7.0,
                centered: false,
                middle_y: true,
                right: true,
            }],
        );
        assert!((dp[0].1[0] + dp[0].4 - 109.0 * 3.0).abs() < 1.0, "{dp:?}");
    }
}
