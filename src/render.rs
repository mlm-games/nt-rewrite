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
use crate::combat::HitFlash;
use crate::comps_a::{
    ARENA_H, ARENA_W, AimDir, FloorMask, GrenadeFuse, Health, HitId, Inventory, LightningArc,
    PendingMutation, PendingUltra, Player, Projectile, RaceState, Run, SelectedCharacter,
    SlashProjectile, TILE, Team, Velocity, WallCell, WallTile,
};
use crate::comps_b::{
    Beam, BossBrain, BossPhase, Corpse, Enemy, EnemyBrain, FxAngle, GroundDecalTint, HazardCloud, OpenedChest, Pickup,
    PickupKind, ChestKind, PickupLifetime, Portal, PortalClear, PortalShock, PortalStrike,
    PropSprites, Prop, StaticFx, SwingFx, Telekinesis, ThroneCarpet, ThroneSit, WeaponVisual,
    YvCouch,
};
use crate::environment::{EnvironmentHazard, PulseSprite, SurfacePulse};
use crate::data::{AreaId, CrownKind, EnemyKind, HazardKind, MutationId, RaceId, UltraMutationId, WeaponId};
use crate::weapons_data::AmmoType;
use crate::hud::{HudState, ability_name, run_area_string, run_timer_string, sync_hud_state};
use crate::savedata_part::{character_def, race_passive_text};
use crate::state::menus::{CHAR_SELECT_ORDER, MenuState};
use crate::state::{SPLASH_GUN_STEPS, SplashState};
use crate::spatial::Pos;
use crate::weapon_runtime::{sanitize_weapon_id, weapon_meta};

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
        let mut catalog =
            AnimCatalog::from_json(json, desc).map_err(|e| anyhow::anyhow!("{e}"))?;
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
        let want: std::collections::HashSet<String> =
            names.iter().map(|n| repame_anim::stem(n).to_string()).collect();
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
            AtlasDesc { size, max_pages, padding: 0 },
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

    fn sprite_for(
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
/// areas plus HQ hold 3 subareas, everything else 1.
pub fn area_max_subarea(area: AreaId) -> u32 {
    match area {
        AreaId::Desert
        | AreaId::Scrapyards
        | AreaId::FrozenCity
        | AreaId::Palace
        | AreaId::HQ => 3,
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
fn area_sprites_full(floor: u32) -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
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

/// Dark surround tile for the padded floor bounds.
/// Bevy `spawn_level` verbatim: route-based `sprFloorN` (same arms as
/// [`area_sprites`]), overridden by `sprFloorEx1` when the catalog has it.
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
) -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
    let route = area_sprites_full(floor);
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
/// Priority: melee slash flags → player weapon table → enemy-kind table
/// → team fallback. (nt-rewrite projectiles carry no art path; bevy
/// attached `Sprite` + candidates at spawn, so this inverts the
/// `player_projectile_candidates` / `enemy_projectile_sprite` choice.)
fn projectile_art(
    proj: &Projectile,
    team: &Team,
    slash: Option<&SlashProjectile>,
) -> &'static str {
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
        Some((_, def)) if def.frames == 2 => 1,
        Some((_, def)) if def.frames > 2 => {
            ((life.elapsed_secs() * 12.0).floor() as u32 % def.frames) as i32
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
    let m = if cam.snap { 1.0 } else { gml_rate(CAM_LERP, dt) };
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
    let vw = (240.0 * w / h).max(320.0).floor();
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
    match area {
        AreaId::Desert => [0.055, 0.045, 0.060, 1.0],
        AreaId::Sewers => [0.030, 0.055, 0.050, 1.0],
        AreaId::Scrapyards => [0.060, 0.055, 0.045, 1.0],
        AreaId::CrystalCaves => [0.045, 0.035, 0.075, 1.0],
        AreaId::FrozenCity => [0.050, 0.060, 0.075, 1.0],
        AreaId::Labs => [0.040, 0.050, 0.060, 1.0],
        AreaId::Palace => [0.065, 0.045, 0.055, 1.0],
        _ => [0.045, 0.045, 0.060, 1.0],
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
pub fn view_rect_world(
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam: &Camera2d,
) -> [f32; 4] {
    let center = cam.effective_center();
    let fit = effective_fit(canvas_dp, world_size, cam);
    if !fit.0.is_finite() || fit.0 <= 1e-6 {
        return [center[0], center[1], 0.0, 0.0];
    }
    let tl = dp_to_world([0.0, 0.0], world_size, center, fit);
    [
        tl[0],
        tl[1],
        canvas_dp[0] / fit.0,
        canvas_dp[1] / fit.0,
    ]
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
            out.push([
                fogx + ix as f32 * FOG_TILE_W,
                fogy + iy as f32 * FOG_TILE_H,
            ]);
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
    let fit = effective_fit(canvas_dp, world_size, cam);
    let center = cam.effective_center();
    let to_world = |dp: [f32; 2]| {
        let w = dp_to_world(dp, world_size, center, fit);
        Vec2::new(w[0], w[1])
    };
    for [x, y] in sideart_tiles(canvas_dp[0], canvas_dp[1]) {
        if let Some(s) = assets.sprite_for(
            "images/sprSideArt.png",
            frame,
            to_world([x, y]),
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
        let outside_png = outside_sprite_for_run(floor, has);
        let (mut minx, mut miny, mut maxx, mut maxy) =
            (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for &(cx, cy) in &cells {
            minx = minx.min(cx);
            miny = miny.min(cy);
            maxx = maxx.max(cx);
            maxy = maxy.max(cy);
        }
        if minx <= maxx {
            for cy in (miny - 6)..=(maxy + 6) {
                for cx in (minx - 6)..=(maxx + 6) {
                    let top_left =
                        Vec2::new(cx as f32 * TILE, cy as f32 * TILE);
                    if cells.contains(&(cx, cy)) {
                        if let Some(s) = place_top_left(
                            assets,
                            floor_png,
                            0,
                            top_left,
                            [1.0; 4],
                            GRID_OVERLAP,
                        ) {
                            out.push(s);
                        }
                    } else if let Some(s) = place_top_left(
                        assets,
                        outside_png,
                        0,
                        top_left,
                        [0.45, 0.45, 0.48, 1.0],
                        GRID_OVERLAP,
                    ) {
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
            out.push(white_quad(
                pos.0,
                0.0,
                size,
                [0.75, 0.12, 0.14, 0.85],
            ));
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
                    if let Some(s) = assets.sprite_sized(
                        path,
                        0,
                        pos.0,
                        Vec2::splat(vis.size),
                        vis.flip_x,
                        tint,
                    ) {
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
            if let Some(s) = assets.sprite_for(
                path,
                frame,
                pos.0,
                sprites.flip_x,
                0.0,
                tint,
            ) {
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
            let is_corpse = corpse.is_some()
                || (sprites.is_some() && prop.is_none())
                || portal.is_some();
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
            if let Some(s) = assets.sprite_for(
                &anim.path,
                anim.frame as i32,
                pos.0,
                flip,
                rotation,
                tint,
            ) {
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
                if let Some(s) =
                    assets.sprite_for(path, frame, pos.0 + off, false, 0.0, [1.0; 4])
                {
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
            let flip = vel
                .map(|v| v.0.x < 0.0)
                .unwrap_or(false)
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
            let flip = vel
                .map(|v| v.0.x < 0.0)
                .unwrap_or(false)
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
            let flip = vel
                .map(|v| v.0.x < 0.0)
                .unwrap_or(false)
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
fn weapon_visual_kick(kicks: &[(Entity, usize, f32)], owner: Entity, slot: usize) -> f32 {
    kicks
        .iter()
        .find(|(o, s, _)| *o == owner && *s == slot)
        .map(|(_, _, k)| *k)
        .unwrap_or(0.0)
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
        let kicks: Vec<(Entity, usize, f32)> = world
            .query::<&WeaponVisual>()
            .iter(world)
            .map(|v| (v.owner, v.slot as usize, v.wkick))
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
        for (entity, pos, player, pa, aim, vel, anim, flash, health, race, inv_opt, telek, sit)
        in q.iter(world)
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
                    let base =
                        anim_path.replace("Idle", if going { "GoSit" } else { "Sit" });
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
            // Bevy `face_aim`: aim.x when aim is live, else velocity.x.
            let right: f32 = {
                let x = aim
                    .map(|a| a.0)
                    .filter(|a| a.length_squared() > 0.001)
                    .map(|a| a.x)
                    .unwrap_or_else(|| vel.map(|v| v.0.x).unwrap_or(0.0));
                if x < 0.0 { -1.0 } else { 1.0 }
            };
            let flip = right < 0.0;
            let gunangle = aim
                .map(|a| a.0.y.atan2(a.0.x))
                .unwrap_or(if flip {
                    std::f32::consts::PI
                } else {
                    0.0
                });
            // GML `Step_0:441-450`: `back` while aiming up-screen
            // (gunangle 0..180 in y-down degrees).
            let back = gunangle > 0.0 && gunangle < std::f32::consts::PI;
            let is_eyes = race.is_some_and(|rs| rs.race == RaceId::Eyes);
            let is_steroids = race.is_some_and(|rs| rs.race == RaceId::Steroids);
            let swapanim = inv_opt.map(|inv| inv.swapanim).unwrap_or(0.0);
            let shine = inv_opt.map(|inv| inv.shine.floor() as i32).unwrap_or(0).max(0);

            // Eyes underlay: held-spec mind power, else MonsterStyle body.
            if is_eyes {
                let img = ((blink_t * 12.0).floor() as i32).max(0);
                if telek.is_some() {
                    let tb = player
                        .mutations
                        .contains(&MutationId::ThroneButt);
                    let path = if tb {
                        "images/sprMindPowerTB.png"
                    } else {
                        "images/sprMindPower.png"
                    };
                    if let Some(s) = assets.sprite_for(
                        path,
                        img % 3,
                        pos.0,
                        flip,
                        0.0,
                        [1.0; 4],
                    ) {
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
                        let at = Vec2::new(
                            pos.0.x - right * (2.0 + i as f32),
                            pos.0.y + swapanim,
                        );
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
                        let at =
                            Vec2::new(pos.0.x - right * 2.0, pos.0.y + swapanim);
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
                        let wkick = weapon_visual_kick(&kicks, entity, inv.current);
                        let ang = gunangle;
                        let at = Vec2::new(
                            pos.0.x + (-wkick) * ang.cos(),
                            pos.0.y + (-wkick) * ang.sin() - swapanim,
                        );
                        let mirror = if meta.wep_mele {
                            inv.wepflip < 0.0
                        } else {
                            flip
                        };
                        if let Some(s) = assets.sprite_for_full(
                            &path,
                            shine,
                            at,
                            false,
                            mirror,
                            ang,
                            [1.0; 4],
                        ) {
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
                let bframe =
                    ((blink_t * 8.0).floor() as u32 % frames) as i32;
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
        )>();
        for (pos, proj, vel, team, slash, fuse) in q.iter(world) {
            let path = projectile_art(proj, team, slash);
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
            if let Some(s) =
                assets.sprite_for("images/sprPortalClear.png", frame, pos.0, false, 0.0, [1.0; 4])
            {
                out.push(s);
            }
        }
        let mut q = world.query::<(&Pos, &PortalStrike)>();
        for (pos, strike) in q.iter(world) {
            let elapsed = strike.timer.duration() - strike.timer.remaining_secs();
            let frames = strip_frames(assets, "images/sprRogueStrike.png").max(1);
            let frame = ((elapsed * 30.0).floor() as u32).min(frames - 1) as i32;
            if let Some(s) =
                assets.sprite_for("images/sprRogueStrike.png", frame, pos.0, false, 0.0, [1.0; 4])
            {
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
            let is_corpse = corpse.is_some()
                || (sprites.is_some() && prop.is_none())
                || portal.is_some();
            if is_corpse {
                continue;
            }
            let rotation = angle.map(|a| a.0).unwrap_or(0.0);
            let mut tint = flash_tint(flash);
            if let Some(lt) = lifetime {
                tint[3] *= (lt.timer.remaining_secs() / 0.12).clamp(0.0, 1.0);
            }
            if let Some(s) = assets.sprite_for(
                &anim.path,
                anim.frame as i32,
                pos.0,
                false,
                rotation,
                tint,
            ) {
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
            if let Some(s) = assets.sprite_for(fx.path, 0, pos.0, false, rotation, tint)
            {
                out.push(s);
            }
        }
    }

    // Held guns: bevy `tick_weapon_visuals` pose parity — hold point
    // `owner + forward * (12 - wkick) + side`, rotation from
    // `held_weapon_angle`, `flip_y` when aiming left (engine channel,
    // same as bevy `sprite.flip_y = aim.0.x < 0.0`).
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
            let (back_current, shine, mirror) = world
                .get::<Inventory>(gun.owner)
                .map(|inv| {
                    let back = aim.y > 0.0 && aim.length_squared() > 0.001;
                    let melee = weapon_meta(gun.wep_id).wep_mele;
                    let facing = if aim.x < 0.0 { -1.0 } else { 1.0 };
                    let upright = if melee {
                        if gun.slot == 1 {
                            inv.bwepflip
                        } else {
                            inv.wepflip
                        }
                    } else {
                        facing
                    };
                    (
                        back && gun.slot as usize == inv.current,
                        inv.shine.floor() as i32,
                        upright < 0.0,
                    )
                })
                .unwrap_or((false, 0, aim.x < 0.0));
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
            let swing = crate::player::held_weapon_angle(angle, gun.wep_angle, gun.wkick);
            let forward = Vec2::new(swing.cos(), swing.sin());
            let perp = Vec2::new(-forward.y, forward.x);
            let side = if gun.slot == 1 {
                perp * 8.0
            } else {
                Vec2::ZERO
            };
            let hold = pos.0 + forward * (12.0 - gun.wkick) + side;
            if let Some(s) =
                assets.sprite_for_full(&path, shine.max(0), hold, false, mirror, swing, [1.0; 4])
            {
                out.push(s);
            }
            // GML bolt-weapon laser sight (not the disc gun): 2 px
            // march to the first wall (1000-step cap), strip drawn
            // `dist / 2 + 2` wide at the aim angle from the muzzle.
            if meta.wep_type == AmmoType::Bolts && meta.wep_name != "DISC GUN" {
                let step = Vec2::new(angle.cos(), angle.sin()) * 2.0;
                let mut tip = hold;
                for _ in 0..1000 {
                    let next = tip + step;
                    let hit = walls.iter().any(|w| w.distance(next) < 8.0);
                    tip = next;
                    if hit {
                        break;
                    }
                }
                let dist = hold.distance(tip);
                if let Some(native) = assets.native_size("images/sprLaserSightPlayer.png") {
                    let size = Vec2::new(native.x * (dist / 2.0 + 2.0), native.y);
                    let mid = hold + Vec2::new(angle.cos(), angle.sin()) * (size.x * 0.5);
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
            if let Some(s) =
                assets.sprite_for(path, frame, pos.0, false, fx.angle, [1.0; 4])
            {
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
    Vec2::new(
        view[0] + map.ox + gx * map.s,
        view[1] + map.oy + gy * map.s,
    )
}

/// Place a strip by its art top-left in GUI px (bevy `gm_sprite`
/// origin law for zero-origin art): quad top-left lands on the mapped
/// GUI point, size = native * `mul` * map scale.
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
    let top_left = hud_gui_to_world(map, view, gx, gy);
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

/// GML weapon-row draw order: the active weapon first, backup second,
/// then any extra (`scrDrawPlayerHUD` iterates `_wep`, `_bwep`,
/// `extra_weps`). Returns inventory slot indices in draw order (max 3).
fn hud_weapon_order(hud: &HudState) -> Vec<usize> {
    let n = hud.weapon_ids.len().min(3).max(1);
    let cur = hud.current_weapon.min(n - 1);
    let mut order = vec![cur];
    order.extend((0..n).filter(|&s| s != cur));
    order
}

/// GML weapon-row x positions: 24, then +44, then +20 per extra
/// (`_dx += 44` for the first slot, `+20` afterwards once extras
/// exist — i.e. 24/68/88).
fn hud_weapon_dx(pos: usize) -> f32 {
    match pos {
        0 => 24.0,
        1 => 68.0,
        _ => 88.0,
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
        let color = if amount <= 0 {
            [51, 51, 51, 255]
        } else if amount <= ammo_pickup_amount(kind).max(0) {
            if active {
                [255, 0, 0, 255]
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
            hud_weapon_dx(pos) + 18.0,
            21.0,
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
        out.push(hud_gui_left(
            "LOW HP".to_string(),
            110.0,
            7.0,
            [255, 0, 0, 255],
            false,
            false,
        ));
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

/// Overlay row: (text, dp top-left, sRGB color 0..1, font px,
/// centered, box width, right-aligned).
pub type GuiRow = (String, [f32; 2], [f32; 4], f32, bool, f32, bool);

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
                let bw = (2.0 * t.gx * k).max(font_px);
                (t.gx * k - bw * 0.5, bw, true)
            } else if t.right {
                let bw = (2.0 * (vw - t.gx) * k).max(font_px);
                (w - (vw - t.gx) * k - bw, bw, false)
            } else {
                (t.gx * k, 200.0 * k, false)
            };
            let top = if t.middle_y {
                t.gy * k - font_px * 0.5
            } else {
                t.gy * k
            };
            let c = [
                t.color[0] as f32 / 255.0,
                t.color[1] as f32 / 255.0,
                t.color[2] as f32 / 255.0,
                t.color[3] as f32 / 255.0,
            ];
            (
                t.text,
                [left.max(0.0), top.max(0.0)],
                c,
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
    let vw = gml_view_size(canvas_dp)[0];
    let items: Vec<MenuGuiText> = hud_gui_texts(world)
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
const GUI_GOLD: [u8; 4] = [255, 221, 0, 255];

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
/// split on em-dash or hyphen.
fn mutation_choice_parts(choice: &str) -> (bool, String, String) {
    let trimmed = choice.trim();
    let (is_ultra, trimmed) = if let Some(rest) = trimmed.strip_prefix("ULTRA:") {
        (true, rest.trim())
    } else {
        (false, trimmed)
    };
    if let Some((name, desc)) = trimmed.split_once(" \u{2014} ") {
        (is_ultra, name.trim().to_string(), desc.trim().to_string())
    } else if let Some((name, desc)) = trimmed.split_once(" - ") {
        (is_ultra, name.trim().to_string(), desc.trim().to_string())
    } else {
        (is_ultra, trimmed.to_string(), String::new())
    }
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
pub fn menu_gui_texts_vw(
    kind: crate::MenuOverlay,
    world: &mut World,
    vw: f32,
) -> Vec<MenuGuiText> {
    let cx = vw * 0.5;
    match kind {
        // Boot reel captions (`Vlambeer/Draw_0` verbatim): mode 0 save
        // note (middle block at cy+24), mode 1 Gamemaker line, mode 3
        // credits (middle block at cy, blanks shape the rhythm).
        // Modes 2/4 are sprite-only.
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
                    text: "MADE IN GAMEMAKER".to_string(),
                    gx: cx,
                    gy: 120.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: true,
                    middle_y: true,
                    right: false,
                }],
                3 => [
                    ("VLAMBEER", 80.0, GUI_GOLD),
                    ("PAUL VEER", 100.0, GUI_WHITE),
                    ("JUKIO KALLIO", 110.0, GUI_WHITE),
                    ("JOONAS TURNER", 120.0, GUI_WHITE),
                    ("JUSTIN CHAN", 130.0, GUI_WHITE),
                    ("YELLOWAFTERLIFE", 140.0, GUI_WHITE),
                    ("PRESENT", 160.0, GUI_WHITE),
                ]
                .iter()
                .map(|(text, gy, color)| MenuGuiText {
                    text: text.to_string(),
                    gx: cx,
                    gy: *gy,
                    color: *color,
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
            let progress = world
                .get_resource::<crate::state::LoadingState>()
                .map(|l| l.progress)
                .unwrap_or(1.0);
            let mut out = vec![gui_center(
                format!("GENERATING... {}%", (progress.clamp(0.0, 1.0) * 100.0).round() as u32),
                cx,
                66.0,
                GUI_GRAY,
            )];
            let roadmap = world
                .get_resource::<crate::comps_a::Run>()
                .map(|r| {
                    if r.world == 0 && r.floor == 0 {
                        String::new()
                    } else {
                        format!(
                            "{}-{}  LOOP {}",
                            r.world,
                            crate::worldgen::floor_in_world(r.floor),
                            r.loop_count
                        )
                    }
                })
                .unwrap_or_default();
            if !roadmap.is_empty() {
                out.push(MenuGuiText {
                    text: roadmap,
                    px: 5.0,
                    right: false,
                    ..gui_center("", cx, 168.0, GUI_GRAY)
                });
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
            LABELS
                .iter()
                .map(|(label, index)| {
                    // GML `MainMenuButton` Draw: white on hover, uigray
                    // when available, uidark when not. Only CO-OP is
                    // gated (no multiplayer shell); STATS stays
                    // available like GML.
                    let available = matches!(index, 0 | 2 | 3 | 4);
                    let color = if available { GUI_MID } else { GUI_UIDARK };
                    MenuGuiText {
                        text: label.to_string(),
                        gx: cx,
                        gy: 72.0 + *index as f32 * 24.0,
                        color,
                        px: 16.0,
                        centered: true,
                        middle_y: false,
                        right: false,
                    }
                })
                .collect()
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
            // GML `scrCampfireMenuDrawCharText`: bigname at GUI y=172
            // with the raw passive/active descriptions beneath (no
            // prefixes), x=8.
            vec![
                MenuGuiText {
                    text: def.name.to_ascii_uppercase().to_string(),
                    gx: 2.0,
                    gy: 172.0,
                    color: GUI_WHITE,
                    px: 14.0,
                    centered: false,
                    middle_y: false,
                    right: false,
                },
                MenuGuiText {
                    text: race_passive_text(race).to_string(),
                    gx: 8.0,
                    gy: 180.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: false,
                    middle_y: false,
                    right: false,
                },
                MenuGuiText {
                    text: ability_name(def.ability).to_string(),
                    gx: 8.0,
                    gy: 189.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: false,
                    middle_y: false,
                    right: false,
                },
            ]
        }
        crate::MenuOverlay::Mutation => {
            let hud = sync_hud_state(world);
            // Offer kind + pick count straight from the pending
            // resources (GML `GameCont.skillpoints/ultrapoints`).
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
                    ("ULTRA MUTATION", "INSTALL ULTRA UPDATE".to_string(), None)
                } else {
                    (
                        "ULTRA MUTATION",
                        "PICK YOUR ULTRA MUTATION".to_string(),
                        None,
                    )
                }
            } else if is_robot {
                (
                    "LEVEL UP",
                    format!("INSTALL {n} UPDATES"),
                    Some("DO NOT TURN OFF ROBOT"),
                )
            } else {
                ("LEVEL UP", format!("SELECT {n} MUTATIONS"), None)
            };
            let accent = if is_ultra { GUI_GOLD } else { GUI_GREEN };
            let mut out = vec![
                gui_center(title, cx, 48.0, accent),
                gui_center(subtitle, cx, 75.0, GUI_CREAM),
            ];
            if let Some(extra) = extra {
                out.push(gui_center(extra, cx, 87.0, GUI_CREAM));
            }
            let selected = world
                .get_resource::<MenuState>()
                .and_then(|m| m.mutation_selected)
                .and_then(|i| hud.mutation_choices.get(i));
            if let Some(sel) = selected {
                let (_, name, desc) = mutation_choice_parts(sel);
                out.push(gui_center(name.to_ascii_uppercase(), cx, 173.0, GUI_WHITE));
                if !desc.is_empty() {
                    out.push(gui_center(desc, cx, 185.0, GUI_CREAM));
                }
            }
            out
        }
        crate::MenuOverlay::GameOver => {
            // GML `GameOver/Draw_0` text layer verbatim (the roadmap dot
            // map and death-cause sprite need unported state, so only
            // the text rows land here; splats ride `menu_sprites`):
            // struggle line centered-middle at (vw/2,48); the area +
            // kills rows at y=106 are a port extra (GML draws the
            // roadmap sprites there); `KILLED BY` (or win `COMPLETION
            // TIME` + clock) centered at vw/2+86.
            let run = world.get_resource::<crate::comps_a::Run>();
            let mut out = Vec::new();
            if let Some(run) = run {
                let max_sub = area_max_subarea(run.area);
                let palace_final =
                    run.area == AreaId::Palace && run.floor_in_area >= max_sub;
                let hq_final = run.area == AreaId::HQ && run.floor_in_area >= max_sub;
                let text = if run.won && hq_final {
                    "THE STRUGGLE IS OVER"
                } else if run.won {
                    "YOU REACHED THE NUCLEAR THRONE"
                } else if palace_final {
                    "YOU ALMOST REACHED THE NUCLEAR THRONE"
                } else if run.loop_count > 0 {
                    "THE STRUGGLE CONTINUES"
                } else {
                    "YOU DID NOT REACH THE NUCLEAR THRONE"
                };
                out.push(MenuGuiText {
                    text: text.to_string(),
                    gx: cx,
                    gy: 48.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: true,
                    middle_y: true,
                    right: false,
                });
                out.push(MenuGuiText {
                    text: run_area_string(run),
                    gx: 52.0,
                    gy: 106.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: false,
                    middle_y: true,
                    right: false,
                });
                out.push(MenuGuiText {
                    text: run.total_kills.to_string(),
                    gx: 135.0,
                    gy: 106.0,
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
                        gy: 95.0,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                    });
                    out.push(MenuGuiText {
                        text: run_timer_string(run.tottimer),
                        gx: cx + 86.0,
                        gy: 110.0,
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
                        gy: 95.0,
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
            out
        }
        crate::MenuOverlay::Pause => {
            // GML pause layer (`UberCont/Draw_0` + `scrMakePauseButtons`
            // + `PauseButton/Other_10`): bigname PAUSED centered-middle
            // at (vw/2,52) over the scrim, buttons at MENU (45,176),
            // RETRY (60,208), SETTINGS (vw-68,176), CONTINUE (vw-78,208);
            // confirm swaps in ARE YOU SURE? (vw/2,120) + BACK (52,192)
            // + QUIT/RETRY (vw-52,192).
            let confirm = world
                .get_resource::<MenuState>()
                .and_then(|m| m.pause_confirm);
            if let Some(confirm) = confirm {
                let right = if confirm == 0 { "QUIT" } else { "RETRY" };
                let right_color = if confirm == 0 { GUI_RED2 } else { GUI_GREEN };
                vec![
                    gui_center("ARE YOU SURE?", cx, 120.0, GUI_WHITE),
                    gui_button("BACK", 52.0, 192.0, GUI_MID),
                    gui_button(right, vw - 52.0, 192.0, right_color),
                ]
            } else {
                vec![
                    MenuGuiText {
                        text: "PAUSED".to_string(),
                        gx: cx,
                        gy: 52.0,
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
        crate::MenuOverlay::Settings => {
            settings_gui_texts(world, vw)
        }
        crate::MenuOverlay::Credits => {
            vec![
                gui_center("CREDITS", cx, 40.0, GUI_CREAM),
                gui_center(
                    "A fan recreation of Nuclear Throne (Vlambeer)",
                    cx,
                    80.0,
                    GUI_CREAM,
                ),
                gui_center("Built with Repame + Repose", cx, 96.0, GUI_GRAY),
                gui_center("No original game assets included", cx, 112.0, GUI_GRAY),
                gui_button("BACK", cx, 180.0, GUI_GRAY),
            ]
        }
    }
}

/// Settings toggle row: label left (cream) + ON/OFF value (gray).
fn push_toggle(out: &mut Vec<MenuGuiText>, label: &str, y: f32, on: bool) {
    out.push(gui_body(label, 80.0, y, GUI_CREAM));
    out.push(gui_body(if on { "ON" } else { "OFF" }, 200.0, y, GUI_GRAY));
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
    let save = world.get_resource::<SaveData>().cloned().unwrap_or_default();
    let s = &save.settings;
    let mut out = Vec::new();
    match page {
        0 => {
            out.push(gui_center("OPTIONS", cx, 24.0, GUI_MID));
            for (i, (label, _)) in
                [("AUDIO", 1u8), ("VIDEO", 2), ("GAME", 4), ("CONTROLS", 8), ("LANGUAGE", 12)]
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
                out.push(gui_body(format!("{:.0}%", val * 100.0), 200.0, gy, GUI_GRAY));
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
            out.push(gui_body(format!("< {} >", s.crosshair + 1), 200.0, y, GUI_GRAY));
            y += 18.0;
            out.push(gui_body("SIDE ART", 80.0, y, GUI_CREAM));
            out.push(gui_body(format!("< {} >", s.sideart), 200.0, y, GUI_GRAY));
            y += 18.0;
            for (label, val) in [("SCREENSHAKE", s.screenshake), ("FREEZE FRAMES", s.freezeframes)] {
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
            out.push(gui_body(format!("< {} >", s.pixel_mode), 200.0, y, GUI_GRAY));
            y += 18.0;
            out.push(gui_button("DISPLAY SETTINGS", cx, y, GUI_MID));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        3 => {
            out.push(gui_center("DISPLAY", cx, 24.0, GUI_MID));
            let mut y = 56.0;
            push_toggle(&mut out, "WIDESCREEN", y, s.widescreen);
            y += 20.0;
            push_toggle(&mut out, "FULLSCREEN", y, s.fullscreen);
            y += 20.0;
            push_toggle(&mut out, "VSYNC", y, s.vsync);
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        4 => {
            out.push(gui_center("GAME", cx, 24.0, GUI_MID));
            let mut y = 48.0;
            for (label, on) in [
                ("BOSS INTROS", s.boss_intros),
                ("PLAY TUTORIAL", s.show_tutorial),
                ("SHOW TIMER", s.show_timer),
                ("SHOW AREA", s.show_area),
                ("PAUSE BUTTON", s.pause_button),
                ("ACHIEVEMENT POPUPS", s.achievements_popup),
                ("AUTO PAUSE", s.auto_pause),
            ] {
                push_toggle(&mut out, label, y, on);
                y += 18.0;
            }
            out.push(gui_button("VIEW CREDITS", cx, y, GUI_MID));
            y += 20.0;
            out.push(gui_button("PROFILE", cx, y, GUI_MID));
            y += 20.0;
            out.push(gui_button("COLOR", cx, y, GUI_MID));
            y += 20.0;
            out.push(gui_button("DATA", cx, y, GUI_MID));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        8 => {
            out.push(gui_center("CONTROLS", cx, 24.0, GUI_MID));
            let mut y = 48.0;
            for (label, on) in [
                ("GAMEPAD", s.gamepad_enabled),
                ("AIM ASSIST", s.aim_assist),
                ("AUTO AIM", s.auto_aim),
                ("VOLUME CONTROLS", s.volume_controls),
                ("SPLIT FIRE", s.split_fire),
                ("FIXED SIGHT", s.fixed_sight),
            ] {
                push_toggle(&mut out, label, y, on);
                y += 18.0;
            }
            out.push(gui_body("GAMEPAD STYLE", 80.0, y, GUI_CREAM));
            let names = ["XBONE", "PS4", "Switch", "SteamDeck"];
            out.push(gui_body(
                format!("< {} >", names[(s.gamepad_type as usize) % names.len()]),
                200.0,
                y,
                GUI_GRAY,
            ));
            y += 18.0;
            out.push(gui_body("SIZE SCALE", 80.0, y, GUI_CREAM));
            out.push(gui_body(
                format!("{:.0}%", s.controls_scale * 100.0),
                200.0,
                y,
                GUI_GRAY,
            ));
            y += 18.0;
            out.push(gui_button("REMAP", cx, y, GUI_MID));
            y += 20.0;
            out.push(gui_button("CHAR PREFS", cx, y, GUI_MID));
            y += 20.0;
            out.push(gui_button("EXPERIMENTAL", cx, y, GUI_MID));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        5 => {
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
        6 => {
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
        7 => {
            out.push(gui_center("DATA", cx, 24.0, GUI_MID));
            out.push(gui_button("RESET OPTIONS", cx, 80.0, GUI_CREAM));
            out.push(gui_button("ERASE PROGRESS", cx, 110.0, GUI_RED2));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        9 => {
            out.push(gui_center("REMAP", cx, 24.0, GUI_MID));
            let mut y = 60.0;
            for label in ["FIRE", "ACTIVE", "SWAP", "PICK"] {
                out.push(gui_center(label, cx, y, GUI_CREAM));
                y += 18.0;
            }
            out.push(gui_center("PRESS ANY KEY - WIP", cx, y, GUI_GRAY));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        10 => {
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
        11 => {
            out.push(gui_center("EXPERIMENTAL", cx, 24.0, GUI_MID));
            out.push(gui_center("KEYBOARD MODE - WIP", cx, 80.0, GUI_GRAY));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        12 => {
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
    let Some((rogue_ammo, rogue_max, cuz_ammo, cuz_max, ultra, mutations, patience_used, back_muscle, race_skin)) =
        player
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
    // (20,4); fills are frame 0 xscale-stretched from (22,7), ghost in
    // the darkened HSV variant, live in `opt_healthcol`; hurt flash =
    // white frame 0 at the hp width).
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
        // shrinks (GML `draw_sprite_ext` xscale from (22,7)).
        let fill_at = |w: f32| hud_gui_to_world(gm, view, 22.0 + w * 0.5, 7.0 + 4.0);
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

    // Rad/exp bar + level badge (GML verbatim: exp frame =
    // `frac*16` at GUI (4,4); `sprExpBarLevel` while an offer is
    // pending (`skillpoints/ultrapoints`); `sprUltraLevel` at (11,16)
    // past the level cap).
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
    if hud.level >= 10 {
        if let Some(s) = assets.sprite_scaled_rotated(
            "images/sprUltraLevel.png",
            0,
            hud_gui_to_world(gm, view, 11.0, 16.0),
            gm.s,
            0.0,
            [1.0; 4],
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
        if let Some(s) = hud_gui_place(assets, bg, bg_frame, dx, 32.0, 1.0, [1.0; 4], gm, view)
        {
            out.push(s);
        }
        let frames = strip_frames(assets, icon).max(1) as f32 - 1.0;
        let icon_frame = (frames - (fill * frames).ceil()).clamp(0.0, frames) as i32;
        if let Some(s) = hud_gui_place(assets, icon, icon_frame, dx, 32.0, 1.0, [1.0; 4], gm, view)
        {
            out.push(s);
        }
    }

    // Rogue/Cuz ammo pips (GML GUI (110,4), subimage by fill progress,
    // 0 when dry; Cuz draws `sprCuzAmmoHUDU` under the Emotional ultra).
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
            let sub = if ammo <= 0.0 {
                0
            } else {
                (((frames as f32 - 1.0) * progress).floor() as i32).max(1)
            };
            if let Some(s) = assets.sprite_scaled_rotated(
                path,
                sub,
                hud_gui_to_world(gm, view, 110.0, 4.0),
                gm.s,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
        }
    }

    // Offer icons (GML `LevCont/Other_10` layout verbatim: single
    // bottom row at GUI y=219, `step = min(32, floor(view_width/(n+1)))`,
    // `scale = max(0.65, step/32)`, centered on the live view center
    // with a -12 shift at n>=10; selected card lifts 1 px and draws
    // white, others gray).
    // Normal offers ride `sprSkillIcon` at the GML skill id (scaled);
    // ultra offers ride `sprEGSkillIcon` at `(race-1)*3+tier-1`
    // (unscaled — GML only scales SkillIcon).
    {
        let n = world
            .get_resource::<PendingMutation>()
            .map(|p| p.choices.len())
            .or_else(|| {
                world
                    .get_resource::<PendingUltra>()
                    .map(|u| u.choices.len())
            })
            .unwrap_or(0);
        let is_ultra = world.get_resource::<PendingMutation>().is_none()
            && world.get_resource::<PendingUltra>().is_some();
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
                    let frame = m
                        .map(crate::hud::mutation_skill_index)
                        .unwrap_or(0) as i32;
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
    // ultras on `sprEGIconHUD` straight from the held frame at y=13,
    // then skills on `sprSkillIconHUD` at the GML skill id at y=12
    // (patience draws its overlay when spent). Top-right from GUI
    // `view_width - 12` in 16 px steps, wrapping at x<=120 (rows stack
    // +16).
    {
        let vw = view[2];
        let held_race = race_skin
            .map(|(r, _)| r)
            .unwrap_or(RaceId::Fish);
        let mut ultras: Vec<(&str, i32)> = Vec::new();
        if let Some(u) = ultra {
            ultras.push(("images/sprEGIconHUD.png", ultra_hud_frame(held_race, u)));
        }
        let mut skills: Vec<(&str, i32)> = Vec::new();
        for m in &mutations {
            skills.push((
                "images/sprSkillIconHUD.png",
                crate::hud::mutation_skill_index(*m) as i32,
            ));
            if *m == MutationId::Patience && patience_used {
                skills.push(("images/sprPatienceIconHUD.png", 0));
            }
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
        // GML `_py--`: the skill row sits one px lower and continues
        // the cursor.
        if !skills.is_empty() && y == 13.0 {
            y = 12.0;
        }
        for (path, frame) in skills {
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
    }

    // Weapon strip (GML draw order verbatim: active weapon left at
    // GUI x=24, then 68, then 88 for extras, y=16; outline white at
    // draw position 0 — or both on Steroids — else #404040). Art stays
    // per-weapon `wep_sprt` and the cursed-slot purple fog is kept
    // (documented extras: GML parts out the weapon sprite with fog
    // tints).
    for (pos, slot) in order.into_iter().enumerate() {
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
        let tint = if cursed {
            [0.7, 0.4, 1.0, 1.0]
        } else if active {
            [1.0; 4]
        } else {
            [0.25, 0.25, 0.25, 1.0]
        };
        let dx = hud_weapon_dx(pos);
        if let Some(s) = hud_gui_place(assets, &path, 0, dx, 16.0, 1.0, tint, gm, view) {
            out.push(s);
        }
    }
    out
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
    let tottimer = world
        .get_resource::<Run>()
        .map(|r| r.tottimer)
        .unwrap_or(0);
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
    if world
        .get_resource::<crate::state::Paused>()
        .is_some_and(|p| p.0)
    {
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
    let hover = world
        .get_resource::<HoverWorld>()
        .and_then(|h| h.0);
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
    )>();
    for (pos, proj, vel, team, slash) in q.iter(world) {
        let path = projectile_art(proj, team, slash);
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
        if let Some(mut s) =
            assets.sprite_scaled_rotated(&anim.path, anim.frame as i32, pos.0, 2.0, 0.0, [1.0, 1.0, 1.0, 0.1])
        {
            s.blend = SpriteBlend::Additive;
            out.push(s);
        }
    }
    out
}
/// Char-pod layout (GML campfire pods in view px): `count`
/// pods of height `slot_h` on a `wh` view. `step = min(20,
/// floor((W-40)/count))`, `xstart = 8`,
/// `ystart = H-slot_h-((36-slot_h) div 2)`.
pub fn char_pod_layout(wh: [f32; 2], count: usize, slot_h: f32) -> Vec<[f32; 2]> {
    let step = 20.0_f32.min(((wh[0] - 40.0) / count.max(1) as f32).floor());
    let y = wh[1] - slot_h - ((36.0 - slot_h) / 2.0).floor();
    (0..count).map(|i| [8.0 + i as f32 * step, y]).collect()
}

/// GO-button position (GML: past the last pod, `y = H-36+bbox_h/2-2`
/// with `div`; the caller passes the sprite bbox height).
pub fn go_button_pos(wh: [f32; 2], count: usize, bbox_h: f32) -> [f32; 2] {
    let step = 20.0_f32.min(((wh[0] - 40.0) / count.max(1) as f32).floor());
    [
        8.0 + count as f32 * step + 2.0,
        wh[1] - 36.0 + (bbox_h / 2.0).floor() - 2.0,
    ]
}

/// GML `scrRaceGetMaxSkinCount` verbatim (hidden-NTT access assumed,
/// like the reference build): BigDog/Frog 1, Skeleton 2, Robot 4,
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
    if let Some(s) = assets.sprite_for(
        "images/sprLoadoutSplat.png",
        0,
        to_world(splat),
        false,
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
    let skins = loadout.as_ref().map(|l| l.unlocked_skins).unwrap_or([true, false, false, false]);
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
    let stored = loadout.as_ref().map(|l| l.stored_weapon).unwrap_or(WeaponId::NONE);
    let chosen = loadout.as_ref().map(|l| l.start_weapon).unwrap_or(WeaponId::NONE);
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
        let meta = weapon_meta(wid);
        if let Some(lout) = meta.wep_lout {
            let path = format!("images/{lout}.png");
            if let Some(s) = assets.sprite_for(&path, 0, to_world([wx, weapons_y]), false, 0.0, tint)
            {
                out.push(s);
            }
        } else if !meta.wep_sprt.is_empty() && meta.wep_sprt != "mskNone" {
            let path = format!("images/{}.png", meta.wep_sprt);
            if let Some(s) = assets.sprite_scaled_rotated(
                &path,
                0,
                to_world([wx, weapons_y]),
                2.0,
                30.0f32.to_radians(),
                tint,
            ) {
                out.push(s);
            }
        }
        wx += weaponsize;
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
            // Vlambeer card at the view bottom + 10 additive shimmer
            // copies at +/-4 px, alpha 0.1. GML `Vlambeer/Draw_0`
            // re-rolls `orandom(4)` every draw, so the offsets must
            // jitter each 30 Hz step (quantized from `t` to stay
            // deterministic), not freeze on a static hash.
            let (fw, fh) = assets
                .native_size("images/sprVlambeer.png")
                .map(|v| (v.x, v.y))
                .unwrap_or((320.0, 240.0));
            let px = cx - fw * 0.5;
            let py = 240.0 - fh;
            if let Some(s) = hud_gui_place(
                assets,
                "images/sprVlambeer.png",
                0,
                px,
                py,
                1.0,
                [1.0; 4],
                gm,
                view,
            ) {
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
                if let Some(mut s) = hud_gui_place(
                    assets,
                    "images/sprVlambeer.png",
                    0,
                    px + jx,
                    py + jy,
                    1.0,
                    [1.0, 1.0, 1.0, 0.1],
                    gm,
                    view,
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
                    let r_extra =
                        hash01(3000u32.wrapping_add(i).wrapping_add(step.wrapping_mul(12979)));
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
    let fit = effective_fit(canvas_dp, world_size, cam);
    let center = cam.effective_center();
    let to_world = |dp: [f32; 2]| {
        let w = dp_to_world(dp, world_size, center, fit);
        Vec2::new(w[0], w[1])
    };
    // GML view-space layer: the live view rect + identity GUI map, so
    // campfire pods/portrait/loadout land in view px verbatim.
    let view = view_rect_world(canvas_dp, world_size, cam);
    let gm = hud_gui_map(view);
    let vw = view[2];
    let gui_to_world = |x: f32, y: f32| hud_gui_to_world(gm, view, x, y);
    match kind {
        crate::MenuOverlay::Splash | crate::MenuOverlay::MainMenu => {
            // Splash art rides `splash_sprites` (per-mode boot reel);
            // the logo room shows no static logo (GML destroys `Logo`
            // once the menu buttons spawn).
        }
        crate::MenuOverlay::Title => {
            let menu = world.get_resource::<MenuState>().cloned();
            let save = world.get_resource::<crate::savedata_part::SaveData>().cloned();
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
            for (i, pos) in char_pod_layout([vw, 240.0], CHAR_SELECT_ORDER.len(), slot_h)
                .iter()
                .enumerate()
            {
                let race = CHAR_SELECT_ORDER[i];
                let locked =
                    save.as_ref().is_none_or(|s| !s.race_unlocked(race));
                let (path, tint): (&str, [f32; 4]) = if locked {
                    ("images/sprCharSelectLocked.png", [0.5, 0.5, 0.5, 1.0])
                } else if i == cursor {
                    ("images/sprCharSelect.png", [1.0; 4])
                } else {
                    ("images/sprCharSelect.png", [0.5, 0.5, 0.5, 1.0])
                };
                if let Some(s) =
                    assets.sprite_for(path, i as i32, gui_to_world(pos[0], pos[1]), false, 0.0, tint)
                {
                    out.push(s);
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
            let portrait_dp = [20.0, 240.0 - 36.0 - 8.0 + 44.0];
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
            let splat_frames = strip_frames(assets, "images/sprCharSplat.png");
            if let Some(s) = assets.sprite_for(
                "images/sprCharSplat.png",
                char_splat_frame(race, splat_frames),
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
            // here — it would double-draw under the text.
            if go_visible {
                // GML `sprGoButtonSymbolic` bbox height 19 (`div 2` = 9).
                let dp = go_button_pos([vw, 240.0], CHAR_SELECT_ORDER.len(), 19.0);
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
            // frame, no tooltips/animation).
            if loadout_open && selected != 0 {
                menu_loadout_sprites(
                    world,
                    assets,
                    [vw, 240.0],
                    selected,
                    &|p| gui_to_world(p[0], p[1]),
                    &mut out,
                );
            }
        }
        crate::MenuOverlay::GameOver => {
            // GML `GameOver/Draw_0` splats at rest (`offsety = 0`):
            // center splat along the bottom, killed-by splat right of
            // view center.
            let dp = [canvas_dp[0] / 2.0, canvas_dp[1] - 32.0];
            if let Some(s) = assets.sprite_for(
                "images/sprGameOverCenterSplat.png",
                0,
                to_world(dp),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            if let Some(s) = assets.sprite_for(
                "images/sprKilledBySplat.png",
                0,
                to_world([canvas_dp[0] / 2.0 + 86.0, canvas_dp[1] / 2.0 - 32.0]),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
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
    if crown != CrownKind::None {
        let gml = crate::savedata_part::crown_port_to_gml(crown as u8);
        let r = 15.0 + (angle_deg / 60.0).sin() * 4.0;
        let a = -angle_deg / 5.3;
        out.push((
            format!("images/sprCrown{gml}Idle.png"),
            1,
            center + Vec2::new((a * deg).cos() * r, (a * deg).sin() * r),
            0.6 + (angle_deg / 200.0).sin() / 4.0,
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
            0.8 + (a / 200.0).sin() / 5.0,
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
/// Skipped while Throne II lives (GML `Nothing2` gate).
pub fn spiral_figures(
    world: &mut World,
    assets: &RenderAssets,
    center: Vec2,
    angle_deg: f32,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();

    let throne_ii_alive = world
        .query::<&Enemy>()
        .iter(world)
        .any(|e| e.kind == EnemyKind::ThroneII);
    if throne_ii_alive {
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
        format!("FLOOR {}  SCORE {}  KILLS {}", hud.floor, hud.score, hud.kills),
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
            format!(
                "GAME OVER  SCORE {}  BEST {}",
                hud.score, hud.high_score
            ),
            [0.0, 0.0],
        ));
    }
    // Weapon label (bevy `sync_weapon_label` text half): gun name at
    // the pickup +(0,31), `E` prompt on the gun (ammo gauge sprites
    // ride the deferred sprite-HUD pass).
    if let Some(label) = world.get_resource::<crate::pickups::WeaponLabel>() {
        if let Some(target) = label.target {
            if let Some(gun) = world.get::<Pos>(target) {
                if !label.text.is_empty() {
                    out.push((
                        label.text.clone(),
                        [gun.0.x, gun.0.y + 31.0],
                    ));
                }
                out.push(("E".to_string(), [gun.0.x, gun.0.y]));
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
mod tests {
    use super::*;
    use crate::anim::PlayerAnim;
    use crate::comps_a::{Health, Inventory};
    use crate::comps_a::{AimDir, ProjectileFriction, Score, SelectedCharacter, Toast, Velocity};
    use crate::comps_a::NT_CAM_SCALE;
    use crate::comps_b::{IdpdRaidState, PropHpTracker};
    use crate::comps_a::NextHurt;
    use crate::data::WeaponId;
    use crate::effects::FiredWeapon;
    use crate::savedata_part::SaveData;
    use crate::time::{GTimer, TimerMode};

    fn assets_dir() -> std::path::PathBuf {
        let home = std::env::var("HOME").unwrap_or_default();
        std::path::PathBuf::from(format!(
            "{home}/Documents/nt-recreated-bevy/assets"
        ))
    }

    const SUBSET: &[&str] = &[
        "sprFloor1",
        "sprWall1Bot",
        "sprWall1Top",
        "sprMutant1Idle",
        "sprMutant1Walk",
        "sprBanditIdle",
        "sprBullet1",
        "sprRad",
        "sprBarrel",
        "sprWeaponChest",
        "sprRevolver",
        "sprHealthBar",
        "sprHealthFill",
        "sprExpBarLevel",
        "sprExpBar",
        "sprEyesB",
        "sprMindPower",
    ];

    fn subset() -> RenderAssets {
        RenderAssets::load_subset(&assets_dir(), SUBSET).expect("subset packs")
    }

    #[test]
    fn sprite_quad_paints_at_spec_position_offscreen() {
        // Pixel proof that instance geometry lands exactly where the
        // snapshot says (no systematic quad offset): a 24x24 body at
        // world (128,128) under the snapped GML camera must paint
        // inside x=[604,676], y=[324,396] at 1280x720 (s=3), with
        // content near the screen center. Far outside the padded
        // floor bounds stays clear.
        use repame_sprite::SpriteBatch;
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, offscreen::OffscreenRenderer};

        let mut renderer = match OffscreenRenderer::new_blocking(1280, 720, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP quad position probe (no GPU): {e}");
                return;
            }
        };
        let mut assets = subset();
        let mut world = scene_world(&assets);
        let pp = world
            .query::<(&Pos, &Player)>()
            .iter(&world)
            .next()
            .map(|(p, _)| p.0)
            .unwrap();
        let cam = world_camera(pp, gml_view_scale([1280.0, 720.0]));
        let instances = world_instances(&mut world, &assets);
        assert!(!instances.is_empty());
        let uploads = assets.take_uploads();
        let at = |px: &[u8], x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 1280 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        // Only the 24x24 player body at (128,128).
        let mut solo = SpriteBatch::new(assets.batch_desc());
        solo.set_camera(cam.fit_matrix([1280.0, 720.0], [1280.0, 720.0]));
        for s in &instances {
            if s.center == Vec2::new(128.0, 128.0) && s.size == Vec2::new(24.0, 24.0) {
                solo.push_sprite(s);
            }
        }
        solo.extend_uploads(uploads.iter().cloned());
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 1280.0,
                    h: 720.0,
                },
                payload: Callback::new(solo),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let (mut mnx, mut mny, mut mxx, mut mxy) = (u32::MAX, u32::MAX, 0u32, 0u32);
        let mut n = 0u64;
        for y in 0..720u32 {
            for x in 0..1280u32 {
                if at(&px, x, y) != [0, 0, 0, 255] {
                    n += 1;
                    mnx = mnx.min(x);
                    mny = mny.min(y);
                    mxx = mxx.max(x);
                    mxy = mxy.max(y);
                }
            }
        }
        assert!(n > 100, "body paints pixels, got {n}");
        // Painted pixels stay inside the specified quad bounds
        // (atlas UV inset + transparent art margins only ever shrink
        // the opaque box, never move it).
        assert!(
            mnx >= 604 && mxx <= 676 && mny >= 324 && mxy <= 396,
            "quad paints inside its specified bounds, got x=[{mnx},{mxx}] y=[{mny},{mxy}]"
        );
        // And the opaque art sits on the screen center (snapped cam).
        assert!(
            at(&px, 640, 360) != [0, 0, 0, 255]
                || at(&px, 639, 361) != [0, 0, 0, 255],
            "body art covers the screen center"
        );
    }

    #[test]
    fn char_pod_layout_matches_gml() {
        // 17 pods on a 320-wide canvas: step = min(20, floor(280/17)).
        let pods = char_pod_layout([320.0, 240.0], 17, 20.0);
        assert_eq!(pods.len(), 17);
        let step = (280.0f32 / 17.0).floor().min(20.0);
        assert!((pods[0][0] - 8.0).abs() < 1e-4);
        assert!((pods[1][0] - (8.0 + step)).abs() < 1e-4);
        assert!((pods[0][1] - (240.0 - 20.0 - 8.0)).abs() < 1e-4);
        // Wide canvas caps the step at 20.
        let wide = char_pod_layout([1920.0, 1080.0], 17, 20.0);
        assert!((wide[1][0] - 28.0).abs() < 1e-4);
        // GO button sits past the last pod (`sprGoButtonSymbolic`
        // bbox 19: y = 240-36+9-2).
        let go = go_button_pos([320.0, 240.0], 17, 19.0);
        assert!(go[0] > pods[16][0]);
        assert!((go[1] - (240.0 - 36.0 + 9.0 - 2.0)).abs() < 1e-4);
    }

    #[test]
    fn back_gun_draws_behind_body() {
        use crate::comps_a::AimDir;
        let assets = subset();
        let mut world = scene_world(&assets);
        // Aim down-screen (y-down gunangle in (0, PI)) => `back`.
        {
            let mut q = world.query::<&mut AimDir>();
            for mut a in q.iter_mut(&mut world) {
                a.0 = Vec2::Y;
            }
        }
        // Current-slot visual for the scene player (kicked, so the
        // behind gun separates from the body).
        let owner = world
            .query::<(Entity, &Player)>()
            .iter(&world)
            .next()
            .map(|(e, _)| e)
            .expect("player");
        {
            let mut q = world.query::<&mut WeaponVisual>();
            let mut found = false;
            for mut v in q.iter_mut(&mut world) {
                if v.owner == owner && v.slot == 0 {
                    v.wkick = 12.0;
                    found = true;
                }
            }
            assert!(found, "scene spawns a current-slot visual");
        }
        let instances = world_instances(&mut world, &assets);
        let body = instances
            .iter()
            .position(|s| {
                s.center == Vec2::new(128.0, 128.0) && s.size == Vec2::new(24.0, 24.0)
            })
            .expect("body drawn");
        // Kicked behind gun sits 12 px up-aim from the body; the gun
        // block must not emit a second (front) copy.
        let guns: Vec<usize> = instances
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                (s.center - Vec2::new(128.0, 128.0)).length() < 16.0
                    && s.center != Vec2::new(128.0, 128.0)
            })
            .map(|(i, _)| i)
            .collect();
        assert_eq!(guns.len(), 1, "exactly one gun instance");
        assert!(guns[0] < body, "back gun precedes the body");
    }

    #[test]
    fn eyes_mind_power_draws_underlay() {
        use crate::comps_a::RaceState;
        use crate::data::RaceId;
        let assets = subset();
        let mut world = scene_world(&assets);
        world.spawn((
            Player::default(),
            RaceState {
                race: RaceId::Eyes,
                skin: crate::data::SkinLetter::A,
            },
            PlayerAnim {
                idle: "images/sprMutant1Idle.png",
                walk: "images/sprMutant1Walk.png",
                hurt: "images/sprMutant1Hurt.png",
                moving: false,
            },
            Pos(Vec2::new(64.0, 64.0)),
            Velocity(Vec2::ZERO),
            Health {
                hp: 8,
                max: 8,
                invuln: GTimer::from_seconds(0.0, TimerMode::Once),
            },
            Inventory {
                weapons: [WeaponId::NONE, WeaponId::NONE, WeaponId::NONE],
                cursed: [false, false, false],
                weapon_slots: 2,
                current: 0,
                ammo: [0; crate::comps_a::MAX_AMMO_TYPES],
                swapanim: 0.0,
                shine: 0.0,
                wepflip: 1.0,
                bwepflip: 1.0,
            },
            Telekinesis {
                timer: GTimer::from_seconds(0.25, TimerMode::Once),
            },
        ));
        let instances = world_instances(&mut world, &assets);
        // Mind-power underlay + Eyes body both emit (subset packs the
        // strip, so the underlay resolves).
        assert!(
            instances.len() >= 2,
            "underlay plus body emit, got {}",
            instances.len()
        );
    }

    #[test]
    fn hud_icon_frames_follow_gml_tables() {
        use crate::data::{MutationId, RaceId, UltraMutationId};
        // Spot-check the name table (GML `mut_*` ids).
        assert_eq!(mutation_hud_frame(MutationId::RhinoSkin), 1);
        assert_eq!(mutation_hud_frame(MutationId::ThroneButt), 5);
        assert_eq!(mutation_hud_frame(MutationId::Patience), 25);
        assert_eq!(mutation_hud_frame(MutationId::HeavyHeart), 29);
        // Ultra frame = race * 3 + tier - 1.
        assert_eq!(
            ultra_hud_frame(RaceId::Fish, UltraMutationId::FishConfiscate),
            1 * 3 + 1 - 1
        );
        assert_eq!(
            ultra_hud_frame(RaceId::Eyes, UltraMutationId::EyesMonsterStyle),
            3 * 3 + 2 - 1
        );
        assert_eq!(
            ultra_hud_frame(RaceId::Rogue, UltraMutationId::RoguePortalStrike),
            12 * 3 + 1 - 1
        );
        // Tier per the GML `UltraSkill` enum (A=1/B=2).
        assert_eq!(ultra_tier(UltraMutationId::FishGunWarrant), 2);
        assert_eq!(ultra_tier(UltraMutationId::EyesProjectileStyle), 1);
        assert_eq!(ultra_tier(UltraMutationId::BigDogHeavyArtillery), 2);
        assert_eq!(ultra_tier(UltraMutationId::BigDogGuardian), 1);
        assert_eq!(ultra_tier(UltraMutationId::CuzHoarder), 1);
        assert_eq!(ultra_tier(UltraMutationId::CuzEmotional), 2);
        // Ghost tint: hue-shifted, half-valued healthcol.
        let dark = healthcol_dark([252.0 / 255.0, 56.0 / 255.0, 0.0, 1.0]);
        assert_eq!(dark[3], 1.0);
        assert!(dark[0] < 252.0 / 255.0 && dark[1] <= 56.0 / 255.0);
    }

    #[test]
    fn menu_texts_follow_gml_panels() {
        use crate::data::{AreaId, MutationId, RaceId};
        // Mutation subtitle carries the pending pick count
        // (GML `LevCont/Draw_0`: `SELECT % MUTATIONS`).
        let mut world = World::new();
        world.init_resource::<MenuState>();
        world.insert_resource(PendingMutation {
            choices: vec![MutationId::RhinoSkin, MutationId::StrongSpirit],
        });
        world.init_resource::<SelectedCharacter>();
        let texts: Vec<String> = menu_gui_texts(crate::MenuOverlay::Mutation, &mut world)
            .into_iter()
            .map(|t| t.text)
            .collect();
        assert!(texts.contains(&"LEVEL UP".to_string()));
        assert!(texts.contains(&"SELECT 2 MUTATIONS".to_string()));
        // Game-over struggle rule (GML `GameOver/Create_0`).
        let mut world = World::new();
        world.init_resource::<MenuState>();
        world.insert_resource(crate::comps_a::Run {
            area: AreaId::Palace,
            floor_in_area: 3,
            ..Default::default()
        });
        let texts: Vec<String> = menu_gui_texts(crate::MenuOverlay::GameOver, &mut world)
            .into_iter()
            .map(|t| t.text)
            .collect();
        assert!(texts.contains(&"YOU ALMOST REACHED THE NUCLEAR THRONE".to_string()));
        assert!(texts.contains(&"KILLED BY".to_string()));
        world.resource_mut::<crate::comps_a::Run>().loop_count = 2;
        world.resource_mut::<crate::comps_a::Run>().area = AreaId::Desert;
        let texts: Vec<String> = menu_gui_texts(crate::MenuOverlay::GameOver, &mut world)
            .into_iter()
            .map(|t| t.text)
            .collect();
        assert!(texts.contains(&"THE STRUGGLE CONTINUES".to_string()));
        // Title rows are the raw GML campfire lines (no prefixes).
        let mut world = World::new();
        world.init_resource::<MenuState>();
        world.insert_resource(SelectedCharacter(RaceId::Fish));
        let texts: Vec<String> = menu_gui_texts(crate::MenuOverlay::Title, &mut world)
            .into_iter()
            .map(|t| t.text)
            .collect();
        assert_eq!(texts[0], "FISH");
        assert!(texts.iter().any(|l| l == "Kills drop extra ammo"));
        assert!(!texts.iter().any(|l| l.starts_with("PASSIVE")));
    }

    #[test]
    fn splash_reel_follows_boot_mode() {
        // Full-pack art (the tiny subset lacks the boot strips).
        let assets = RenderAssets::load(&assets_dir()).expect("full pack loads");
        let view = [0.0, 0.0, 576.0, 324.0];
        let mut world = World::new();
        // Mode 0: saving icon only.
        world.insert_resource(SplashState { mode: 0, t: 0.5, guns: 0 });
        let reel = splash_sprites(&mut world, &assets, view);
        assert_eq!(reel.len(), 1, "saving icon, got {}", reel.len());
        // Mode 1: caption only, no sprites.
        world.resource_mut::<SplashState>().mode = 1;
        assert!(splash_sprites(&mut world, &assets, view).is_empty());
        // Mode 2: card + 10 shimmer copies.
        world.resource_mut::<SplashState>().mode = 2;
        let reel = splash_sprites(&mut world, &assets, view);
        assert_eq!(reel.len(), 11, "card + shimmer, got {}", reel.len());
        // Mode 4 at full charge: logo frame 7 + 8-way glow ring.
        world.insert_resource(SplashState { mode: 4, t: 2.0, guns: 7 });
        let reel = splash_sprites(&mut world, &assets, view);
        assert_eq!(reel.len(), 9, "logo + glow, got {}", reel.len());
        // Mode 4 early: logo only, no glow.
        world.insert_resource(SplashState { mode: 4, t: 1.0, guns: 2 });
        let reel = splash_sprites(&mut world, &assets, view);
        assert_eq!(reel.len(), 1, "logo pre-charge, got {}", reel.len());
    }

    #[test]
    fn hud_sprites_emit_health_and_rad_bars() {
        let assets = subset();
        let mut world = scene_world(&assets);
        // View-anchored HUD (GML view-space GUI, identity map): a
        // 426x240 live view at 16:9.
        let view = [0.0, 0.0, 426.0, 240.0];
        let dt = 1.0 / 30.0;
        let bars = hud_sprites(&mut world, &assets, view, dt);
        // Health bar + fill (+ghost at full HP overlaps) and the rad bar.
        assert!(
            bars.len() >= 3,
            "health + rad bars emit, got {}",
            bars.len()
        );
        // Ghost persistence decays toward damage (GML `lsthealth`:
        // approach 0.5/step inside a 20 gap).
        {
            let mut q = world.query::<&mut Health>();
            for mut h in q.iter_mut(&mut world) {
                if h.max == 8 {
                    h.hp = 4;
                }
            }
        }
        let bars = hud_sprites(&mut world, &assets, view, dt);
        let ghost = world.resource::<crate::hud::HudBars>().last_hp;
        assert!(
            (ghost - 7.5).abs() < 1e-4,
            "ghost trails damage 8 -> 7.5, got {ghost}"
        );
        assert!(!bars.is_empty());
        // Fill frames/tints follow GML (`scrDrawPlayerHUD:42-58`):
        // both fills are frame 0 (same uvs), the ghost in the
        // darkened HSV variant, the live fill in `opt_healthcol` red.
        let red = tint_to_linear([252.0 / 255.0, 56.0 / 255.0, 0.0, 1.0]);
        // Wide short quads are the health fills (weapon icons are
        // narrow).
        let fills: Vec<_> = bars
            .iter()
            .filter(|s| s.size.y < 12.0 && s.size.x > 20.0)
            .collect();
        assert_eq!(fills.len(), 2, "ghost + live fills, got {}", fills.len());
        assert_eq!(
            fills[0].uv_min, fills[1].uv_min,
            "both fills are frame 0"
        );
        assert!(
            (fills[0].size.x - fills[1].size.x).abs() > 1e-6,
            "ghost wider than live fill"
        );
        assert_eq!(fills[1].color, red, "live fill tinted healthcol");
        assert!(
            fills[0].color[0] < red[0],
            "ghost darkened, got {:?}",
            fills[0].color
        );
        // View-anchored (GML view-space GUI): the HUD rides the passed
        // view rect, not the player — even a view that does not
        // contain the player still gets its corner bars.
        let far_view = [1000.0, 1000.0, 576.0, 324.0];
        let far = hud_sprites(&mut world, &assets, far_view, dt);
        assert!(!far.is_empty(), "HUD draws on any live view");
        for s in &far {
            assert!(
                s.center.x >= far_view[0]
                    && s.center.x <= far_view[0] + far_view[2]
                    && s.center.y >= far_view[1]
                    && s.center.y <= far_view[1] + far_view[3],
                "bar inside the view, got {:?}",
                s.center
            );
        }
    }

    #[test]
    fn spiral_figure_layout_matches_gml() {
        // Angle 0: crown due east at r=15, scale 0.6, no rotation;
        // player 0 centered, scale 0.8.
        let figs = spiral_figure_layout(
            Vec2::ZERO,
            0.0,
            CrownKind::Death,
            &["images/sprMutant1Hurt.png".to_string()],
        );
        assert_eq!(figs.len(), 2);
        assert_eq!(figs[0].0, "images/sprCrown2Idle.png");
        assert_eq!(figs[0].1, 1);
        assert!((figs[0].2 - Vec2::new(15.0, 0.0)).length() < 1e-3);
        assert!((figs[0].3 - 0.6).abs() < 1e-4);
        assert!(figs[0].4.abs() < 1e-4);
        assert_eq!(figs[1].2, Vec2::ZERO);
        assert!((figs[1].3 - 0.8).abs() < 1e-4);
        // Player stagger: n=1 offsets (8, 6) and shifts the angle.
        let figs = spiral_figure_layout(
            Vec2::ZERO,
            0.0,
            CrownKind::None,
            &["a".to_string(), "b".to_string()],
        );
        assert_eq!(figs.len(), 2);
        assert_eq!(figs[1].2, Vec2::new(8.0, 6.0));
        let expect_rot = -(50.0f32) * 2.0 * std::f32::consts::PI / 180.0;
        assert!((figs[1].4 - expect_rot).abs() < 1e-4);
        // Crown orbit radius breathes ±4 px.
        let figs = spiral_figure_layout(
            Vec2::ZERO,
            90.0,
            CrownKind::Death,
            &[],
        );
        let r = figs[0].2.length();
        assert!((r - (15.0 + (1.5f32).sin() * 4.0)).abs() < 1e-3);
    }

    const ATMOS_SUBSET: &[&str] = &[
        "sprFog2",
        "sprFog102",
        "sprSideArt",
        "sprBigPortrait",
        "sprBigPortraitChickenHeadless",
        "sprBigPortraitRebelBHooded",
        "sprCharSplat",
        "sprBigName",
        "sprFaintedBar",
        "sprYVBossGamingIdle",
        "sprYVBossGamingAirhorn",
    ];

    fn atmos_assets() -> RenderAssets {
        // Fog tiles (480x360) and the 64-frame big portrait need full
        // pages, not the tiny test page.
        RenderAssets::load_subset_sized(&assets_dir(), ATMOS_SUBSET, 2048, 4)
            .expect("atmos packs")
    }

    #[test]
    fn fog_sprite_selection_matches_gml() {
        // GML TopCont/Draw_0: pizza sewers first, then sewers/event.
        assert_eq!(
            fog_sprite_for_area(AreaId::PizzaSewers, false),
            Some("images/sprFog102.png")
        );
        assert_eq!(
            fog_sprite_for_area(AreaId::Sewers, false),
            Some("images/sprFog2.png")
        );
        assert_eq!(
            fog_sprite_for_area(AreaId::Desert, true),
            Some("images/sprFog2.png")
        );
        assert_eq!(fog_sprite_for_area(AreaId::Desert, false), None);
        assert_eq!(fog_sprite_for_area(AreaId::Campfire, false), None);
        assert_eq!(FOG_ALPHA, 0.1, "GML #macro FOG_ALPHA");
    }

    #[test]
    fn fog_tiles_cover_3x3_around_view() {
        // Origin view, no scroll: fogx = 0 + 480, fogy = 0.
        let tiles = fog_tiles(0.0, 0.0, 0.0);
        assert_eq!(tiles.len(), 9);
        assert!(tiles.contains(&[0.0, -360.0]));
        assert!(tiles.contains(&[480.0, 0.0]));
        assert!(tiles.contains(&[960.0, 360.0]));
        // Scroll shifts x only; negative views floor correctly.
        let scrolled = fog_tiles(0.0, 0.0, 100.0);
        assert!(scrolled.contains(&[380.0, 0.0]));
        let neg = fog_tiles(-10.0, -10.0, 0.0);
        assert!(neg.contains(&[-480.0 + 480.0, -360.0]));
        assert!(neg.contains(&[0.0, -360.0]));
    }

    #[test]
    fn fog_sprites_draw_sewers_only() {
        let assets = atmos_assets();
        let cam = world_camera(Vec2::ZERO, NT_CAM_SCALE);
        let mut world = World::new();
        world.insert_resource(Run {
            floor: 4,
            area: AreaId::Sewers,
            ..Default::default()
        });
        world.insert_resource(crate::environment::FogState { scroll: 0.0 });
        let fog = fog_sprites(&mut world, &assets, [640.0, 480.0], [640.0, 480.0], &cam);
        assert_eq!(fog.len(), 9, "3x3 fog tiles over sewers");
        for s in &fog {
            assert!((s.color[3] - FOG_ALPHA).abs() < 1e-6);
        }
        world.resource_mut::<Run>().area = AreaId::Desert;
        let clear = fog_sprites(&mut world, &assets, [640.0, 480.0], [640.0, 480.0], &cam);
        assert!(clear.is_empty(), "desert draws no fog");
    }

    #[test]
    fn sideart_tiles_match_gml_counts() {
        // 320x240 view: n = 5, nh = ceil(240/64)+1 = 5.
        let tiles = sideart_tiles(320.0, 240.0);
        assert_eq!(tiles.len(), 5 * 5 * 2 + 5 * 10 * 2, "horiz + vert quads");
        // GML horizontal arms: i=-64 column and view-64+i*64 column.
        assert!(tiles.contains(&[-64.0, 0.0]));
        assert!(tiles.contains(&[320.0, 0.0]));
        assert!(tiles.contains(&[-320.0, 256.0]));
        // GML vertical runs: i*64 columns above and below.
        assert!(tiles.contains(&[0.0, -64.0]));
        assert!(tiles.contains(&[0.0, 240.0]));
        assert!(tiles.contains(&[256.0, 240.0 + 9.0 * 64.0]));
        // Widescreen view: n = floor(426/64) = 6 (GML `i <= n`).
        let wide = sideart_tiles(426.0, 240.0);
        assert_eq!(wide.len(), 6 * 5 * 2 + 6 * 10 * 2);
    }

    #[test]
    fn gml_view_law_matches_set_view_size() {
        // GML `scrSetViewSize` verbatim (`opt_resolution` defaults on):
        // always 240 tall, widened to 240*aspect, portrait floors the
        // width at 320.
        assert_eq!(gml_view_size([1280.0, 720.0]), [426.0, 240.0]);
        assert_eq!(gml_view_size([960.0, 720.0]), [320.0, 240.0]);
        assert_eq!(gml_view_size([720.0, 1280.0]), [320.0, 240.0]);
        assert!((gml_view_scale([1280.0, 720.0]) - 1.0 / 3.0).abs() < 1e-6);
        assert!(
            (NT_CAM_SCALE - 1.0 / 3.0).abs() < 1e-6,
            "720p reference scale"
        );
    }

    #[test]
    fn gui_map_is_view_identity() {
        // GML `display_set_gui_size(view)`: GUI px == view px, so the
        // health bar lands 1:1 with no letterbox.
        let view = [10.0, 20.0, 426.0, 240.0];
        let gm = hud_gui_map(view);
        assert_eq!((gm.s, gm.ox, gm.oy), (1.0, 0.0, 0.0));
        assert_eq!(
            hud_gui_to_world(gm, view, 20.0, 4.0),
            Vec2::new(30.0, 24.0)
        );
    }

    #[test]
    fn gui_texts_dp_hug_live_edges() {
        // 1280x720: k = 3, vw = 426. Left rows sit at gx*k, centered
        // rows center on gx*k, right rows end 2 GUI px from the edge.
        let rows = gui_texts_dp(
            [1280.0, 720.0],
            vec![
                MenuGuiText {
                    text: "L".to_string(),
                    gx: 8.0,
                    gy: 180.0,
                    color: [255, 255, 255, 255],
                    px: 7.0,
                    centered: false,
                    middle_y: false,
                    right: false,
                },
                MenuGuiText {
                    text: "C".to_string(),
                    gx: 213.0,
                    gy: 48.0,
                    color: [255, 255, 255, 255],
                    px: 7.0,
                    centered: true,
                    middle_y: false,
                    right: false,
                },
                MenuGuiText {
                    text: "R".to_string(),
                    gx: 424.0,
                    gy: 223.0,
                    color: [255, 255, 255, 255],
                    px: 7.0,
                    centered: false,
                    middle_y: false,
                    right: true,
                },
            ],
        );
        assert_eq!(rows[0].1, [24.0, 540.0]);
        assert_eq!(rows[1].1, [0.0, 144.0]);
        assert_eq!(rows[1].5, 213.0 * 2.0 * 3.0);
        assert_eq!(rows[2].1[0], 1280.0 - 2.0 * 3.0 - rows[2].5);
        assert_eq!(rows[2].1[0] + rows[2].5, 1280.0 - 6.0);
    }

    #[test]
    fn pause_buttons_anchor_to_live_edges() {
        // GML `scrMakePauseButtons`/`PauseButton/Other_10`: left column
        // absolute, right column at view_width - N.
        let mut world = World::new();
        world.init_resource::<MenuState>();
        let texts = menu_gui_texts_vw(crate::MenuOverlay::Pause, &mut world, 426.0);
        let at = |name: &str| {
            texts
                .iter()
                .find(|t| t.text == name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .gx
        };
        assert_eq!(at("MENU"), 45.0);
        assert_eq!(at("RETRY"), 60.0);
        assert_eq!(at("SETTINGS"), 426.0 - 68.0);
        assert_eq!(at("CONTINUE"), 426.0 - 78.0);
        assert_eq!(at("PAUSED"), 213.0);
    }

    #[test]
    fn crosshair_tracks_hover_past_deadzone() {
        use crate::comps_a::AimDir;
        let scene = subset();
        let assets =
            RenderAssets::load_subset(&assets_dir(), &["sprCrosshair"]).expect("crosshair packs");
        let mut world = scene_world(&scene);
        {
            let mut q = world.query::<(&mut AimDir, &Pos)>();
            for (mut a, _) in q.iter_mut(&mut world) {
                a.0 = Vec2::X;
            }
        }
        let pp = world
            .query::<(&Pos, &Player)>()
            .iter(&world)
            .next()
            .map(|(p, _)| p.0)
            .unwrap_or(Vec2::ZERO);
        // No hover staged: deadzone idle, alpha stays 0, nothing drawn.
        let none = crosshair_sprites(&mut world, &assets, 1.0 / 30.0);
        assert!(none.is_empty(), "no cursor, no crosshair");
        // Cursor 100 px out along aim: active, snapped on the idle
        // target by the first call then lerped 0.8 toward
        // player + (16 + dis) = pp + 116: 16 + 100 * 0.8 = 96.
        world.insert_resource(HoverWorld(Some(pp + Vec2::new(100.0, 0.0))));
        let drawn = crosshair_sprites(&mut world, &assets, 1.0 / 30.0);
        assert_eq!(drawn.len(), 1);
        assert!((drawn[0].center.x - (pp.x + 96.0)).abs() < 1e-3);
        assert!((drawn[0].center.y - pp.y).abs() < 1e-3);
    }

    #[test]
    fn portal_indicator_marks_offscreen_portals() {
        let assets = RenderAssets::load_subset(&assets_dir(), &["sprPortalIndicator"])
            .expect("indicator packs");
        let cam = world_camera(Vec2::ZERO, NT_CAM_SCALE);
        let mut world = World::new();
        // View at 1280x720/1-3 scale: [-213,-120] + [426,240].
        world.spawn((Pos(Vec2::new(0.0, 0.0)), Portal));
        world.spawn((Pos(Vec2::new(500.0, 0.0)), Portal));
        let marks = portal_indicator_sprites(
            &mut world,
            &assets,
            [1280.0, 720.0],
            [1280.0, 720.0],
            &cam,
        );
        assert_eq!(marks.len(), 1, "only the offscreen portal");
        assert!(
            (marks[0].center.x - 610.0 / 3.0).abs() < 1e-3,
            "clamped 10px in, got {}",
            marks[0].center.x
        );
        assert!((marks[0].center.y - 0.0).abs() < 1e-3);
    }

    #[test]
    fn bloom_gates_on_opt_bloom() {
        let assets = subset();
        let mut world = scene_world(&assets);
        let mut save = SaveData::default();
        save.settings.bloom = false;
        world.insert_resource(save);
        assert!(
            bloom_sprites(&mut world, &assets).is_empty(),
            "opt_bloom off draws nothing"
        );
        world.resource_mut::<SaveData>().settings.bloom = true;
        let glows = bloom_sprites(&mut world, &assets);
        assert!(!glows.is_empty(), "live projectile glows");
        assert!(glows.iter().all(|s| s.blend == SpriteBlend::Additive));
    }

    #[test]
    fn sideart_sprites_follow_opt_sideart() {
        let assets = atmos_assets();
        let cam = world_camera(Vec2::ZERO, NT_CAM_SCALE);
        let mut world = World::new();
        world.insert_resource(SaveData::default());
        let tiles = sideart_sprites(&mut world, &assets, [320.0, 240.0], [320.0, 240.0], &cam);
        assert_eq!(tiles.len(), sideart_tiles(320.0, 240.0).len());
        // Out-of-range options clamp to the packed strip.
        world.resource_mut::<SaveData>().settings.sideart = 200;
        let clamped = sideart_sprites(&mut world, &assets, [320.0, 240.0], [320.0, 240.0], &cam);
        assert_eq!(clamped.len(), tiles.len());
    }

    #[test]
    fn race_skin_subimage_matches_gml() {
        // ((_skin < 2) ? (_skin + (_race-1)*2) : (_skin*16 + (_race-1))).
        assert_eq!(race_skin_subimage(RaceId::Random, 0), -1);
        assert_eq!(race_skin_subimage(RaceId::Fish, 0), 0);
        assert_eq!(race_skin_subimage(RaceId::Fish, 1), 1);
        assert_eq!(race_skin_subimage(RaceId::Fish, 2), 32);
        assert_eq!(race_skin_subimage(RaceId::Chicken, 1), 1 + 8 * 2);
        assert_eq!(race_skin_subimage(RaceId::Cuz, 3), 3 * 16 + 15);
        // NTT bug: Robot-D sits at 56, not 55.
        assert_eq!(race_skin_subimage(RaceId::Robot, 3), 56);
    }

    #[test]
    fn big_portrait_resolution_matches_gml() {
        // Headless chicken (hp <= 0): headless strip, frame = skin.
        assert_eq!(
            big_portrait_for(RaceId::Chicken, 1, Some(0), AreaId::Desert),
            Some(("images/sprBigPortraitChickenHeadless.png", 1))
        );
        // Living chicken: normal strip + skin subimage.
        assert_eq!(
            big_portrait_for(RaceId::Chicken, 1, Some(5), AreaId::Desert),
            Some((
                "images/sprBigPortrait.png",
                race_skin_subimage(RaceId::Chicken, 1)
            ))
        );
        // Rebel B-skin in the city: hooded strip.
        assert_eq!(
            big_portrait_for(RaceId::Rebel, 1, Some(8), AreaId::City),
            Some(("images/sprBigPortraitRebelBHooded.png", 0))
        );
        // Same rebel outside the city: normal strip.
        assert_eq!(
            big_portrait_for(RaceId::Rebel, 1, Some(8), AreaId::Desert),
            Some((
                "images/sprBigPortrait.png",
                race_skin_subimage(RaceId::Rebel, 1)
            ))
        );
        assert_eq!(big_portrait_for(RaceId::Random, 0, None, AreaId::Desert), None);
        // Splat frames stay inside the packed strip.
        assert!((0..4).contains(&char_splat_frame(RaceId::Fish, 4)));
        assert_eq!(char_splat_frame(RaceId::Fish, 0), 0);
    }

    #[test]
    fn fainted_bars_draw_grace_then_bleed() {
        use crate::comps_b::Revive;
        let assets = atmos_assets();
        let mut world = World::new();
        world.insert_resource(Run::default());
        let view = [-1000.0, -1000.0, 2000.0, 2000.0];
        assert!(fainted_bar_sprites(&mut world, &assets, view).is_empty());
        let e = world
            .spawn((
                Pos(Vec2::new(100.0, 50.0)),
                Revive {
                    alarm4: 150.0,
                    alarm5: 0.0,
                },
            ))
            .id();
        let bars = fainted_bar_sprites(&mut world, &assets, view);
        assert_eq!(bars.len(), 2, "bar + grace pulse quad");
        // Fainted bar centers on the clamped pos (view is huge: raw).
        assert!((bars[0].center - Vec2::new(100.0, 34.0)).length() < 1e-3);
        // Grace width: alarm4/300*28 = 14 px.
        assert!((bars[1].size.x - 14.0).abs() < 1e-3);
        // Bleed phase: alarm4 spent, alarm5 half.
        world.get_mut::<Revive>(e).unwrap().alarm4 = 0.0;
        world.get_mut::<Revive>(e).unwrap().alarm5 = 15.0;
        let bars = fainted_bar_sprites(&mut world, &assets, view);
        assert_eq!(bars.len(), 2, "bar + red bleed quad");
        assert!((bars[1].size.x - (15.0 / 30.0 * 28.0 - 2.0)).abs() < 1e-3);
    }

    #[test]
    fn yv_couch_draws_state_strip_and_frame() {
        use crate::comps_b::YvCouch;
        let assets = atmos_assets();
        let mut world = World::new();
        world.spawn((Pos(Vec2::new(10.0, 20.0)), YvCouch::idle()));
        let sprites = world_instances(&mut world, &assets);
        assert_eq!(sprites.len(), 1, "couch draws exactly one quad");
        assert!((sprites[0].center - Vec2::new(10.0, 20.0)).length() < 1e-3);
    }

    #[test]
    fn subset_packs_without_panic() {
        let assets = subset();
        assert_eq!(assets.layers, 1);
        assert!(assets.batch_desc().layer_size == 512);
        for name in SUBSET {
            let (uv, def) = assets.uv(name, 0).expect("resolves");
            assert!(uv.max[0] <= 1.0 && uv.max[1] <= 1.0);
            assert!(uv.max[0] > uv.min[0] && uv.max[1] > uv.min[1]);
            assert!(def.frames >= 1);
        }
    }

    fn scene_world(assets: &RenderAssets) -> World {
        let mut world = World::new();
        world.insert_resource(Run {
            floor: 1,
            ..Default::default()
        });
        world.insert_resource(FloorMask {
            cells: [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().collect(),
            cols: 80,
            rows: 52,
        });
        // Player (PlayerAnim fallback path — headless spawns carry no
        // SpriteAnim, exactly like `spawn_player_loaded`).
        let player = world
            .spawn((
                Player::default(),
                PlayerAnim {
                    idle: "images/sprMutant1Idle.png",
                    walk: "images/sprMutant1Walk.png",
                    hurt: "images/sprMutant1Hurt.png",
                    moving: false,
                },
                AimDir(Vec2::X),
                Velocity(Vec2::ZERO),
                Health {
                    hp: 8,
                    max: 8,
                    invuln: GTimer::from_seconds(0.0, TimerMode::Once),
                },
                Inventory {
                    weapons: [WeaponId::REVOLVER, WeaponId::SHOTGUN, WeaponId::NONE],
                    cursed: [false, false, false],
                    swapanim: 0.0,
                    shine: 0.0,
                    wepflip: 1.0,
                    bwepflip: 1.0,
                    weapon_slots: 2,
                    current: 0,
                    ammo: [0, 40, 8, 0, 0, 0],
                },
                Pos(Vec2::new(128.0, 128.0)),
            ))
            .id();
        // Bandit with a live anim + hurt flash (tint path).
        let bandit_def = assets.catalog.def("images/sprBanditIdle.png").unwrap().clone();
        world.spawn((
            Enemy {
                kind: EnemyKind::Bandit,
                score: 10,
                touch_damage: 0,
                rad_drop: 2,
                drop_chance: 16,
                weapon_chance: 0,
            },
            Team::Enemy,
            Pos(Vec2::new(160.0, 128.0)),
            Velocity(Vec2::new(-10.0, 0.0)),
            Health {
                hp: 4,
                max: 4,
                invuln: GTimer::from_seconds(0.0, TimerMode::Once),
            },
            SpriteAnim::new("images/sprBanditIdle.png", &bandit_def),
            HitFlash::new([1.0, 0.3, 0.2, 1.0], 0.15),
        ));
        // Player bullet (team fallback strip, velocity heading).
        world.spawn((
            Team::Player,
            Projectile {
                damage: 3,
                life: GTimer::from_seconds(2.0, TimerMode::Once),
                radius: 4.0,
                knockback: 60.0,
                explosive: false,
                source: None,
            },
            Velocity(Vec2::new(300.0, 0.0)),
            ProjectileFriction(0.0),
            Pos(Vec2::new(144.0, 120.0)),
        ));
        // Wall with no floor below → top strip.
        world.spawn((WallTile, WallCell(0, 6), Pos(Vec2::new(16.0, 208.0))));
        // Prop barrel with recorded art + anim.
        let barrel_def = assets.catalog.def("images/sprBarrel.png").unwrap().clone();
        world.spawn((
            Prop {
                size: Vec2::splat(24.0),
                hp: 1,
                destructible: true,
                explosive: false,
            },
            PropHpTracker { last_hp: 1 },
            NextHurt::default(),
            PropSprites {
                idle: "images/sprBarrel.png",
                hurt: "images/sprBarrelHurt.png",
                dead: "images/sprBarrelDead.png",
                flip_x: false,
            },
            SpriteAnim::new("images/sprBarrel.png", &barrel_def),
            Pos(Vec2::new(48.0, 96.0)),
        ));
        // Rad pickup + weapon chest.
        world.spawn((
            Pickup {
                kind: PickupKind::Rad(5),
            },
            Pos(Vec2::new(200.0, 200.0)),
        ));
        world.spawn((
            Pickup {
                kind: PickupKind::Chest(ChestKind::Weapon),
            },
            Pos(Vec2::new(64.0, 200.0)),
        ));
        // Held gun on the player (revolver sprite, rotated).
        world.spawn(WeaponVisual {
            owner: player,
            wkick: 0.0,
            wep_id: WeaponId::REVOLVER,
            wep_angle: 0.5,
            slot: 0,
        });
        world
    }

    #[test]
    fn every_mapped_kind_resolves_uv_inside_atlas() {
        let assets = subset();
        let mut world = scene_world(&assets);
        let instances = world_instances(&mut world, &assets);
        // 2x2 mask -> 14x14 padded rect (4 lit + 192 dark surround) +
        // 1 wall + 1 prop + 2 pickups + 1 enemy + 1 player +
        // 1 projectile + 1 gun = 204.
        assert_eq!(instances.len(), 204, "all kinds mapped");
        for s in &instances {
            assert!(s.uv_min.x >= 0.0 && s.uv_min.y >= 0.0);
            assert!(s.uv_max.x <= 1.0 && s.uv_max.y <= 1.0);
            assert!(s.uv_max.x > s.uv_min.x && s.uv_max.y > s.uv_min.y);
            assert!(s.page < assets.layers, "page inside the pack");
            assert!(s.size.x > 0.0 && s.size.y > 0.0);
        }
        // Hurt flash tints the bandit; the player keeps white.
        // (Instance colors linearize; sim tints stay sRGB.)
        let tints: Vec<[f32; 4]> = instances.iter().map(|s| s.color).collect();
        assert!(
            tints.contains(&tint_to_linear([1.0, 0.3, 0.2, 1.0])),
            "flash tint rides"
        );
        assert!(tints.contains(&[1.0, 1.0, 1.0, 1.0]), "plain white rides");
        // Enemy faces left (vel.x < 0) → mirrored.
        let flipped = instances.iter().filter(|s| s.flip_x).count();
        assert_eq!(flipped, 1, "exactly the bandit flips");
    }

    #[test]
    fn projectile_tables_match_bevy() {
        assert_eq!(
            player_projectile_path(WeaponId::REVOLVER),
            "images/sprBullet1.png"
        );
        assert_eq!(
            player_projectile_path(WeaponId::NONE),
            "images/sprBullet1.png"
        );
        assert_eq!(
            enemy_projectile_path(EnemyKind::Scorpion),
            "images/sprScorpionBullet.png"
        );
        assert_eq!(
            enemy_projectile_path(EnemyKind::FiredMaggot),
            "images/sprEnemyBullet1.png",
            "rewrite-only kinds take bevy's wildcard arm"
        );
        assert_eq!(pickup_art(&PickupKind::Rad(1)).as_ref(), "images/sprRad.png");
        assert_eq!(
            pickup_art(&PickupKind::Chest(ChestKind::Weapon)).as_ref(),
            "images/sprWeaponChest.png"
        );
    }

    #[test]
    fn pickup_quads_use_native_frame_cells() {
        // Bevy parity: `spawn_pickup` ignores `pickup_sprite` sizes and
        // draws native strip frames (`sprite_exact`). A weapon chest is
        // one 16x16 cell, a rad one cell of `sprRad` — never the old
        // 32/12-unit stretches.
        let assets = subset();
        let mut world = scene_world(&assets);
        let instances = world_instances(&mut world, &assets);
        let at = |p: Vec2| {
            instances
                .iter()
                .find(|s| (s.center - p).length() < 1e-3)
                .unwrap_or_else(|| panic!("no quad at {p:?}"))
                .size
        };
        let (_, chest_def) = assets.uv("images/sprWeaponChest.png", 0).unwrap();
        assert_eq!(
            at(Vec2::new(64.0, 200.0)),
            Vec2::new(chest_def.w as f32, chest_def.h as f32)
        );
        assert_eq!(chest_def.w, 16, "chest frame is 16px, not 32");
        let (_, rad_def) = assets.uv("images/sprRad.png", 0).unwrap();
        assert_eq!(
            at(Vec2::new(200.0, 200.0)),
            Vec2::new(rad_def.w as f32, rad_def.h as f32)
        );
    }

    #[test]
    fn camera_follow_matches_gml_backcont_law() {
        // `framed_lerp` survives for the zoom ease only.
        assert!((framed_lerp(0.10, 1.0 / 60.0) - 0.10).abs() < 1e-6);
        assert_eq!(framed_lerp(0.5, 0.0), 0.0);
        let step = |cam: &mut GmlCamera, vw: f32, vh: f32, s: &CamStepInput| {
            gml_camera_step(cam, vw, vh, s, 1.0 / 30.0)
        };
        let base = CamStepInput {
            player: Vec2::ZERO,
            aim_dir: Vec2::X,
            aim_dis: 0.0,
            viewdist: CAM_VIEWDIST,
            poi: None,
            shake_scale: 1.0,
            timescale: 1.0,
            jx: 0.0,
            jy: 0.0,
        };
        // Snap jumps straight to the framed player (m = 1).
        let mut cam = GmlCamera { snap: true, ..Default::default() };
        step(&mut cam, 480.0, 270.0, &base);
        assert_eq!((cam.x, cam.y), (-240.0, -135.0));
        assert!(!cam.snap, "snap clears after one step");
        // Steady state: no aim/POI/shake → framed player exactly.
        for _ in 0..60 {
            step(&mut cam, 480.0, 270.0, &base);
        }
        assert_eq!((cam.x, cam.y), (-240.0, -135.0));
        // Aim lean: dis_fire 120 / viewdist 4 = 30 px right. (The snap
        // frame itself zeroes aim/POI like GML, so step twice.)
        let mut cam2 = GmlCamera { snap: true, ..Default::default() };
        let aimed = CamStepInput { aim_dis: 120.0, ..base };
        step(&mut cam2, 480.0, 270.0, &aimed);
        assert_eq!((cam2.x, cam2.y), (-240.0, -135.0));
        step(&mut cam2, 480.0, 270.0, &aimed);
        assert_eq!((cam2.x, cam2.y), (-228.0, -135.0));
        // Melee/bolts divisors change the lean, not the law.
        assert_eq!(cam_viewdist_for(crate::data::WeaponId::NONE), CAM_VIEWDIST);
        // POI pull: portal 120 px away leans dist/6 = 20 px.
        let mut cam3 = GmlCamera { snap: true, ..Default::default() };
        let poi = CamStepInput {
            poi: Some(CamPoi { pos: Vec2::new(120.0, 0.0), capped: true }),
            ..base
        };
        step(&mut cam3, 480.0, 270.0, &poi);
        step(&mut cam3, 480.0, 270.0, &poi);
        assert_eq!((cam3.x, cam3.y), (-232.0, -135.0));
        // POI cap: 720 px portal clamps to 72 px lean.
        let mut cam4 = GmlCamera { snap: true, ..Default::default() };
        let far = CamStepInput {
            poi: Some(CamPoi { pos: Vec2::new(720.0, 0.0), capped: true }),
            ..base
        };
        step(&mut cam4, 480.0, 270.0, &far);
        step(&mut cam4, 480.0, 270.0, &far);
        assert_eq!((cam4.x, cam4.y), (-211.0, -135.0));
        // Uncapped POI does not clamp.
        let mut cam5 = GmlCamera { snap: true, ..Default::default() };
        let far2 = CamStepInput {
            poi: Some(CamPoi { pos: Vec2::new(720.0, 0.0), capped: false }),
            ..base
        };
        step(&mut cam5, 480.0, 270.0, &far2);
        step(&mut cam5, 480.0, 270.0, &far2);
        assert_eq!((cam5.x, cam5.y), (-192.0, -135.0));
        // Knock decays round(viewx2 - viewx2*0.4); shake linear-decays.
        let mut cam6 = GmlCamera { viewx2: 10.0, shake: 5.0, ..Default::default() };
        step(&mut cam6, 480.0, 270.0, &base);
        assert_eq!(cam6.viewx2, 6.0);
        assert_eq!(cam6.shake, 4.0);
        // Heavy shake uses the power branch; opt_shake 0 zeroes all.
        let mut cam7 = GmlCamera { shake: 12.0, ..Default::default() };
        step(&mut cam7, 480.0, 270.0, &base);
        assert!((cam7.shake - 12.0 * 0.8).abs() < 1e-5);
        let mut cam8 = GmlCamera { shake: 5.0, viewx2: 3.0, ..Default::default() };
        let mute = CamStepInput { shake_scale: 0.0, ..base };
        step(&mut cam8, 480.0, 270.0, &mute);
        assert_eq!((cam8.shake, cam8.viewx2), (0.0, 0.0));
        // Positions round to whole px.
        let mut cam9 = GmlCamera { x: 0.4, y: 0.4, ..Default::default() };
        step(&mut cam9, 480.0, 270.0, &base);
        assert_eq!(cam9.x.fract(), 0.0);
        // The GPU matrix centers the look point (NDC origin).
        let cam = world_camera(Vec2::new(48.0, 0.0), NT_CAM_SCALE);
        let ndc = cam
            .fit_matrix([800.0, 600.0], [800.0, 600.0])
            .project_point3(glam::Vec3::new(48.0, 0.0, 0.0));
        assert!(ndc.x.abs() < 1e-5 && ndc.y.abs() < 1e-5);
        assert!((cam.units_per_pixel - NT_CAM_SCALE).abs() < 1e-6);
        // Framing parity: raw-viewport extent + units-per-px shows
        // viewport * scale world units (1280x720 -> 426x240 at the
        // 1/3 reference scale).
        let fit = effective_fit([1280.0, 720.0], [1280.0, 720.0], &cam);
        assert!((fit.0 - 1.0 / NT_CAM_SCALE).abs() < 1e-5);
        assert!((1280.0 / fit.0 - 1280.0 * NT_CAM_SCALE).abs() < 1e-3);
        assert!((720.0 / fit.0 - 720.0 * NT_CAM_SCALE).abs() < 1e-3);
    }

    #[test]
    fn hud_texts_anchor_to_world() {
        let assets = subset();
        let mut world = scene_world(&assets);
        world.init_resource::<Score>();
        world.init_resource::<SaveData>();
        world.init_resource::<Toast>();
        world.init_resource::<SelectedCharacter>();
        world.init_resource::<IdpdRaidState>();
        // Core HP/level/ammo rows are GUI-space (bevy `nt_hud_overlay`
        // verbatim); world texts carry only the run extras.
        let gui = hud_gui_texts(&mut world);
        let hp = gui.iter().find(|t| t.text == "8/8").expect("hp row");
        assert_eq!((hp.gx, hp.gy), (67.0, 7.0));
        assert!(hp.centered);
        let level = gui.iter().find(|t| t.text == "1").expect("level row");
        assert_eq!((level.gx, level.gy), (11.0, 16.0));
        assert!(level.middle_y);
        // Revolver bullets (40) + shotgun shells (8), melee skipped.
        let ammo: Vec<_> = gui
            .iter()
            .filter(|t| t.text == "40" || t.text == "8")
            .collect();
        assert_eq!(ammo.len(), 2, "both weapon ammo rows, got {gui:?}");
        assert_eq!((ammo[0].gx, ammo[0].gy), (42.0, 21.0));
        assert_eq!((ammo[1].gx, ammo[1].gy), (86.0, 21.0));
        assert!(!gui.iter().any(|t| t.text == "LOW HP"), "full hp");
        // Active-first draw order (GML `_wep` then `_bwep`): swapping
        // to the shotgun moves shells left and bullets right.
        {
            let mut inv = world.query::<&mut Inventory>();
            for mut inv in inv.iter_mut(&mut world) {
                if inv.weapon_slots == 2 {
                    inv.current = 1;
                }
            }
        }
        let gui = hud_gui_texts(&mut world);
        let at42 = gui
            .iter()
            .find(|t| t.gx == 42.0 && t.gy == 21.0)
            .expect("left ammo row");
        let at86 = gui
            .iter()
            .find(|t| t.gx == 86.0 && t.gy == 21.0)
            .expect("right ammo row");
        assert_eq!(at42.text, "8", "shells lead when shotgun active");
        assert_eq!(at86.text, "40", "bullets follow");
        // Misc-HUD rows (GML `scrDrawMiscHUD`): map name right-aligned
        // (show_area defaults on), clock hidden (show_timer defaults
        // off).
        let area = gui.iter().find(|t| t.right).expect("area row");
        assert_eq!((area.gx, area.gy), (318.0, 223.0));
        assert!(!gui.iter().any(|t| t.text.contains(':')), "no clock");
        world.resource_mut::<SaveData>().settings.show_timer = true;
        let gui = hud_gui_texts(&mut world);
        let clock = gui
            .iter()
            .find(|t| t.right && t.text.contains(':'))
            .expect("clock row");
        assert_eq!((clock.gx, clock.gy), (318.0, 223.0));
        let area = gui
            .iter()
            .find(|t| t.right && !t.text.contains(':'))
            .expect("area row");
        assert_eq!((area.gx, area.gy), (318.0, 214.0));
        // GUI → dp mapping through the live GML law (1280x720:
        // k=3, GUI 426 wide; HP centers its 134-wide box on gx=67).
        let dps = hud_gui_texts_dp(&mut world, [1280.0, 720.0]);
        let hp_dp = dps
            .iter()
            .find(|(t, _, _, _, _, _, _)| t == "8/8")
            .expect("hp dp");
        assert_eq!(hp_dp.1, [0.0, 21.0]);
        assert_eq!(hp_dp.3, 21.0);
        // World texts: floor/score row + clock + map name (extras with
        // no bevy HUD equivalent).
        let texts = hud_texts(&mut world);
        let get = |prefix: &str| {
            texts
                .iter()
                .find(|(t, _)| t.starts_with(prefix))
                .cloned()
                .unwrap_or_else(|| panic!("missing {prefix}"))
        };
        let (floors, _) = get("FLOOR ");
        assert!(floors.contains("SCORE"));
        // dp mapping: world point through the live camera fit.
        let cam = Camera2d {
            center: Vec2::new(640.0, 416.0),
            ..Default::default()
        };
        let dps = hud_texts_dp(&mut world, [1280.0, 832.0], [1280.0, 832.0], &cam);
        assert_eq!(dps.len(), texts.len());
    }

    /// Transient FX slice: beams, arcs, muzzles, hazards, particles,
    /// and damage numbers each resolve to instances/texts.
    #[test]
    fn beam_quad_spans_endpoints() {
        let assets = subset();        let mut world = World::new();
        let center = Vec2::new(100.0, 100.0);
        let dir = Vec2::new(1.0, 1.0).normalize_or_zero();
        world.spawn((
            Pos(center),
            Beam {
                team: Team::Player,
                dir,
                length: 200.0,
                width: 12.0,
                damage: 5,
                knockback: 0.0,
                color: [1.0, 1.0, 1.0, 1.0],
                timer: GTimer::from_seconds(0.5, TimerMode::Once),
                tick: GTimer::from_seconds(0.1, TimerMode::Repeating),
                source: None,
            },
        ));
        let instances = fx_instances(&mut world, &assets);
        assert_eq!(instances.len(), 1);
        let s = &instances[0];
        assert_eq!(s.center, center);
        assert_eq!(s.size, Vec2::new(200.0, 12.0));
        assert!((s.rotation - dir.y.atan2(dir.x)).abs() < 1e-6);
        // Quad spans the sim segment: center ± axis * (length / 2).
        let axis = Vec2::new(s.rotation.cos(), s.rotation.sin());
        let (a, b) = (
            center - dir * 100.0,
            center + dir * 100.0,
        );
        assert!((s.center - axis * 100.0 - a).length() < 1e-4);
        assert!((s.center + axis * 100.0 - b).length() < 1e-4);
        // Subset lacks the beam strip: tinted fallback quad.
        assert_eq!(s.uv_min, Vec2::ZERO);
        assert_eq!(s.uv_max, Vec2::ONE);
        assert_eq!(s.page, 0);
    }

    #[test]
    fn walls_composite_out_bot_top_with_bot_only_on_floor_south() {
        // Bevy `spawn_level` parity: every wall cell draws Out + Top,
        // Bot only when the screen-south tile is floor; Top straddles
        // the cell's top edge (center 8px above the cell top-left row).
        let mut names = SUBSET.to_vec();
        names.push("sprWall1Out");
        names.push("sprWall1Trans");
        let assets =
            RenderAssets::load_subset(&assets_dir(), &names).expect("wall subset packs");
        let mut world = World::new();
        world.insert_resource(Run {
            floor: 1,
            gen_seed: 123,
            ..Default::default()
        });
        world.insert_resource(FloorMask {
            cells: [(0, 1)].into_iter().collect(),
            cols: 80,
            rows: 52,
        });
        // 16px wall units: (0,0) has floor tile (0,1) due south -> Bot;
        // (2,0) sits over tile (1,0)/(1,1) (no floor) -> Top only.
        world.spawn((WallTile, WallCell(0, 0)));
        world.spawn((WallTile, WallCell(2, 0)));
        let instances = world_instances(&mut world, &assets);
        // Grid quads grow by GRID_OVERLAP (17x17); Out keeps its 24x32.
        let small: Vec<_> = instances
            .iter()
            .filter(|s| s.size == Vec2::new(17.0, 17.0))
            .collect();
        let out: Vec<_> = instances
            .iter()
            .filter(|s| s.size == Vec2::new(24.0, 32.0) && !s.flip_y)
            .collect();
        // Wall drop shadows (GML `scrShadows`): same Out strip flipped
        // under each wall with open floor south (both cells here),
        // black at 0.4 alpha.
        let shadow: Vec<_> = instances
            .iter()
            .filter(|s| s.size == Vec2::new(24.0, 32.0) && s.flip_y)
            .collect();
        // Bot + Top for (0,0), Top for (2,0), Trans ring around the lone
        // floor cell: (0,4),(1,4),(0,5),(1,5) -> 4 Trans quads.
        assert_eq!(small.len(), 2 + 1 + 4, "bot+2x top+4x trans");
        assert_eq!(out.len(), 2, "one Out skirt per wall cell");
        assert_eq!(shadow.len(), 2, "one flipped shadow per wall cell");
        assert!(
            shadow.iter().all(|s| s.color[3] == 0.4f32 || (s.color[3] - 0.4).abs() < 1e-3),
            "shadows at 0.4 alpha"
        );
        // Top straddles the top edge: center 8px above the cell row.
        assert!(
            small.iter().any(|s| s.center == Vec2::new(0.0, -8.0)),
            "top of (0,0) at (0,-8)"
        );
        assert!(
            small.iter().any(|s| s.center == Vec2::new(32.0, -8.0)),
            "top of (2,0) at (32,-8)"
        );
        // Bot fills its cell only where floor sits south.
        assert!(
            small.iter().any(|s| s.center == Vec2::new(0.0, 0.0)),
            "bot of (0,0) fills (0,0)"
        );
        assert!(
            !small.iter().any(|s| s.center == Vec2::new(32.0, 0.0)),
            "no bot for (2,0)"
        );
        // Out skirt overhangs the cell top-left by its origin (4,12).
        assert!(
            out.iter().any(|s| s.center == Vec2::new(0.0, 0.0)),
            "out of (0,0) anchored at cell corner"
        );
    }

    #[test]
    fn adjacent_floor_quads_leave_no_background_seam() {
        // Screenshot parity (grid lines across the viewport): two
        // butt-jointed floor quads under a fractional camera must cover
        // every pixel — no clear-colored cracks at the shared edge.
        use repame_sprite::{Camera2d, SpriteBatch};
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, offscreen::OffscreenRenderer};

        let mut assets = subset();
        let mut batch = SpriteBatch::new(assets.batch_desc());
        // Fractional center + game-like scale: quad edges land on
        // fractional screen pixels, as in-game.
        batch.set_camera(
            Camera2d {
                center: Vec2::new(64.37, 64.0),
                offset: Vec2::ZERO,
                units_per_pixel: 0.45,
                zoom: 1.0,
                roll: 0.0,
            }
            .fit_matrix([256.0, 256.0], [256.0, 256.0]),
        );
        batch.extend_uploads(assets.take_uploads());
        for tl in [Vec2::new(0.0, 0.0), Vec2::new(32.0, 0.0)] {
            let s = place_top_left(&assets, "images/sprFloor1.png", 0, tl, [1.0; 4], GRID_OVERLAP)
                .expect("floor resolves");
            batch.push_sprite(&s);
        }
        let mut renderer = match OffscreenRenderer::new_blocking(256, 256, 4) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP offscreen render test (no GPU): {e}");
                return;
            }
        };
        let scene = Scene {
            clear_color: Color::from_rgba(255, 0, 255, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect { x: 0.0, y: 0.0, w: 256.0, h: 256.0 },
                payload: Callback::new(batch),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([1.0, 0.0, 1.0, 1.0]))
            .expect("offscreen render");
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 256 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        // Joint screen x: world 32 -> (32-64.37)/0.45+128 ~= 56;
        // quads span screen y -14..57, so scan the joint there.
        let mut magenta = 0;
        for y in 0..50 {
            for x in 50..62 {
                if at(x, y) == [255, 0, 255, 255] {
                    magenta += 1;
                }
            }
        }
        assert_eq!(magenta, 0, "background shows through the floor joint");
    }

    #[test]
    fn beam_tint_follows_sim_color() {        // Bevy parity: ion cyan / laser red / boss orange-green ride the
        // sim `Beam.color`, not a hardcoded white.
        let assets = subset();
        let mut world = World::new();
        let tint = [1.0, 0.18, 0.14, 1.0];
        world.spawn((
            Pos(Vec2::ZERO),
            Beam {
                team: Team::Player,
                dir: Vec2::X,
                length: 100.0,
                width: 10.0,
                damage: 1,
                knockback: 0.0,
                color: tint,
                timer: GTimer::from_seconds(0.5, TimerMode::Once),
                tick: GTimer::from_seconds(0.1, TimerMode::Repeating),
                source: None,
            },
        ));
        let instances = fx_instances(&mut world, &assets);
        assert_eq!(instances.len(), 1);
        // Instance colors are linearized for the GPU; the sim tint stays
        // sRGB (bevy `BeamSpec.color` parity).
        assert_eq!(instances[0].color, tint_to_linear(tint));
    }

    #[test]
    fn projectile_two_frame_strips_pin_to_second_cell() {
        // Bevy `sprite_from_projectile_path`: 2-frame strips render cell 1.
        let mut names = SUBSET.to_vec();
        names.push("sprEnemyBullet1");
        let assets =
            RenderAssets::load_subset(&assets_dir(), &names).expect("projectile subset packs");
        let life = GTimer::from_seconds(1.0, TimerMode::Once);
        assert_eq!(
            projectile_frame(&assets, "images/sprBullet1.png", &life),
            1
        );
        assert_eq!(
            projectile_frame(&assets, "images/sprEnemyBullet1.png", &life),
            1
        );
    }

        #[test]
    fn srgb_tints_linearize_with_alpha_untouched() {
        assert_eq!(tint_to_linear([1.0, 1.0, 1.0, 1.0]), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(tint_to_linear([0.0, 0.0, 0.0, 0.5]), [0.0, 0.0, 0.0, 0.5]);
        let mid = tint_to_linear([0.45, 0.45, 0.48, 1.0]);
        // srgb 0.45 -> linear ~0.172 (bevy outside-ring tint law).
        assert!((mid[0] - 0.172).abs() < 0.005, "{mid:?}");
        assert_eq!(mid[3], 1.0);
    }

    #[test]
    fn full_pack_wall_and_floor_sample_source_strips() {
        // Screenshot parity (green wall bands / pale floors in-game):
        // wall-top, wall-bot and floor quads from the FULL pack must
        // sample their own strips. Renders one quad per strip offscreen
        // and compares the center pixel against the strip average.
        use repame_sprite::{SpriteBatch, screen_camera};
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, offscreen::OffscreenRenderer};

        let mut assets = RenderAssets::load(&assets_dir()).expect("full pack loads");
        // Control: same quad from a 1-page subset pack (isolates
        // multi-page atlas layer indexing).
        let mut subset_assets =
            RenderAssets::load_subset(&assets_dir(), &["sprWall1Top"]).expect("subset packs");
        let cases = [
            ("images/sprWall1Top.png", [174.0, 142.0, 105.0]),
            ("images/sprWall1Bot.png", [98.0, 116.0, 34.0]),
            ("images/sprFloor1.png", [236.0, 188.0, 83.0]),
        ];
        let mut batch = SpriteBatch::new(assets.batch_desc());
        batch.set_camera(screen_camera([256.0, 256.0]));
        batch.extend_uploads(assets.take_uploads());
        for (i, (path, _)) in cases.iter().enumerate() {
            let center = Vec2::new(48.0 + i as f32 * 80.0, 128.0);
            let s = assets
                .sprite_for(path, 0, center, false, 0.0, [1.0; 4])
                .unwrap_or_else(|| panic!("resolves {path}"));
            batch.push_sprite(&s);
        }
        assert!(!batch.is_empty());
        let mut renderer = match OffscreenRenderer::new_blocking(256, 256, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP offscreen render test (no GPU): {e}");
                return;
            }
        };
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect { x: 0.0, y: 0.0, w: 256.0, h: 256.0 },
                payload: Callback::new(batch),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let at = |x: u32, y: u32| -> [f32; 3] {
            let i = ((y * 256 + x) * 4) as usize;
            [px[i] as f32, px[i + 1] as f32, px[i + 2] as f32]
        };
        for (i, (path, want)) in cases.iter().enumerate() {
            let got = at(48 + i as u32 * 80, 128);
            let dist = ((got[0] - want[0]).powi(2)
                + (got[1] - want[1]).powi(2)
                + (got[2] - want[2]).powi(2))
            .sqrt();
            assert!(
                dist < 60.0,
                "{path} center pixel {got:?} far from strip avg {want:?}"
            );
        }

        // Subset control: wall-top from a 1-page pack must hit the texel.
        {
            let mut batch = SpriteBatch::new(subset_assets.batch_desc());
            batch.set_camera(screen_camera([256.0, 256.0]));
            batch.extend_uploads(subset_assets.take_uploads());
            let s = subset_assets
                .sprite_for(
                    "images/sprWall1Top.png",
                    0,
                    Vec2::new(128.0, 128.0),
                    false,
                    0.0,
                    [1.0; 4],
                )
                .expect("subset resolves wall top");
            batch.push_sprite(&s);
            let scene = Scene {
                clear_color: Color::from_rgba(0, 0, 0, 255),
                nodes: vec![SceneNode::Callback {
                    rect: Rect { x: 0.0, y: 0.0, w: 256.0, h: 256.0 },
                    payload: Callback::new(batch),
                }],
            };
            let px = renderer
                .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
                .expect("offscreen render");
            let i = ((128 * 256 + 128) * 4) as usize;
            let got = [px[i] as f32, px[i + 1] as f32, px[i + 2] as f32];
            let dist = ((got[0] - 174.0).powi(2)
                + (got[1] - 142.0).powi(2)
                + (got[2] - 105.0).powi(2))
            .sqrt();
            assert!(
                dist < 30.0,
                "subset walltop center pixel {got:?} far from strip"
            );
        }
    }

    #[test]
    fn health_fill_renders_red_not_white() {
        // Pixel proof for the white-box bug: the live fill (frame 0,
        // `opt_healthcol` tint) over the bar must read red, not white.
        // Uses the subset pack: the full 2048x16 atlas exceeds what
        // software rasterizers allocate, which reads back black for
        // reasons unrelated to the tint law.
        use repame_sprite::{SpriteBatch, screen_camera};
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, offscreen::OffscreenRenderer};

        let mut assets = RenderAssets::load_subset(&assets_dir(), SUBSET).expect("subset packs");
        let mut batch = SpriteBatch::new(assets.batch_desc());
        batch.set_camera(screen_camera([256.0, 256.0]));
        batch.extend_uploads(assets.take_uploads());
        let center = Vec2::new(128.0, 128.0);
        let bar = assets
            .sprite_for("images/sprHealthBar.png", 2, center, false, 0.0, [1.0; 4])
            .expect("bar resolves");
        batch.push_sprite(&bar);
        let h = assets
            .native_size("images/sprHealthFill.png")
            .expect("fill resolves");
        let red = [252.0 / 255.0, 56.0 / 255.0, 0.0, 1.0];
        let fill = assets
            .sprite_sized(
                "images/sprHealthFill.png",
                0,
                center + Vec2::new(-44.0 + 2.0 + 84.0 / 2.0, 3.0),
                Vec2::new(84.0, h.y),
                false,
                red,
            )
            .expect("fill resolves");
        batch.push_sprite(&fill);
        let mut renderer = match OffscreenRenderer::new_blocking(256, 256, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP offscreen render test (no GPU): {e}");
                return;
            }
        };
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect { x: 0.0, y: 0.0, w: 256.0, h: 256.0 },
                payload: Callback::new(batch),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let i = ((135 * 256 + 160) * 4) as usize;
        let got = [px[i] as f32, px[i + 1] as f32, px[i + 2] as f32];
        assert!(
            got[0] > 150.0 && got[1] < 120.0 && got[2] < 100.0,
            "fill center pixel {got:?} must read red, not white"
        );
    }

    #[test]
    fn arc_polyline_covers_segment_on_strip() {
        let mut names = SUBSET.to_vec();
        names.push("sprLightning");
        let assets =
            RenderAssets::load_subset(&assets_dir(), &names).expect("fx subset packs");
        let mut world = World::new();
        let mid = Vec2::new(50.0, 60.0);
        let angle = 0.7f32;
        world.spawn((
            Pos(mid),
            LightningArc {
                timer: GTimer::from_seconds(0.09, TimerMode::Once),
                len: 120.0,
                angle,
            },
        ));
        let instances = fx_instances(&mut world, &assets);
        assert_eq!(instances.len(), 1);
        let s = &instances[0];
        // Single segment: quad center/size/rotation cover mid ± len/2
        // (bevy chain-arc sprite law: (dist, 3)).
        assert_eq!(s.center, mid);
        assert_eq!(s.size, Vec2::new(120.0, 3.0));
        assert!((s.rotation - angle).abs() < 1e-6);
        let axis = Vec2::new(angle.cos(), angle.sin());
        assert!((s.center + axis * 60.0 - (mid + axis * 60.0)).length() < 1e-4);
        // Strip path: fresh timer → frame 0 uvs from the catalog.
        let (uv, _) = assets.uv(ARC_STRIP, 0).expect("arc strip packs");
        assert_eq!(s.uv_min, Vec2::new(uv.min[0], uv.min[1]));
        assert_eq!(s.page, uv.page);
    }

    #[test]
    fn muzzle_hazard_and_particle_quads() {
        use repame_fx::{EaseKind, Gradient};

        let assets = subset();
        let mut world = World::new();
        // No muzzle visual exists in GML/bevy (fire feedback is the
        // particle spray): markers expire silently.
        world.spawn(FiredWeapon::new(Vec2::new(10.0, 10.0), Vec2::X));
        world.spawn((
            Pos(Vec2::new(300.0, 300.0)),
            HazardCloud {
                kind: HazardKind::Toxic,
                radius: 40.0,
                damage: 2,
                timer: GTimer::from_seconds(2.0, TimerMode::Once),
                tick: GTimer::from_seconds(0.5, TimerMode::Repeating),
            },
        ));
        world.spawn(Particle {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
            age_ticks: 0,
            life_ticks: 100,
            size_px: 4.0,
            gravity_pps2: 0.0,
            drag_per_sec: 0.0,
            gradient: Gradient::solid([1.0, 0.0, 0.0, 1.0]),
            ease: EaseKind::Linear,
            spawner: None,
            page: 0,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
        });
        let instances = fx_instances(&mut world, &assets);
        assert_eq!(instances.len(), 2);
        assert!(
            instances.iter().all(|s| s.size != MUZZLE_SIZE),
            "no invented muzzle quad"
        );
        let disc = instances
            .iter()
            .find(|s| s.size == Vec2::new(80.0, 80.0))
            .expect("hazard disc");
        assert_eq!(disc.center, Vec2::new(300.0, 300.0));
        assert_eq!(disc.color[0], tint_to_linear([0.30, 0.88, 0.30, 1.0])[0]);
        let dot = instances
            .iter()
            .find(|s| s.size == Vec2::new(4.0, 4.0))
            .expect("particle dot");
        assert_eq!(dot.color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn numbers_render_path_resolves() {
        use repame_fx::DamageNumber;

        let mut world = World::new();
        world.spawn(DamageNumber {
            text: "12".to_string(),
            x: 7.0,
            y: 9.0,
            age_ticks: 0,
            life_ticks: 80,
            rise_pps: 40.0,
            color: [0.7, 0.95, 1.0, 1.0],
        });
        // Numbers ride text, never sprites.
        assert!(fx_instances(
            &mut world,
            &RenderAssets::load_subset(&assets_dir(), SUBSET).expect("subset packs")
        )
        .is_empty());
        let texts = fx_texts(&mut world);
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0].text, "12");
        assert_eq!(texts[0].pos, Vec2::new(7.0, 9.0));
        assert_eq!(texts[0].color, [0.7, 0.95, 1.0, 1.0]);
        assert_eq!(texts[0].size, NUMBER_TEXT_SIZE);
    }

    /// End-to-end GPU proof (repame-sprite offscreen pattern): pack a
    /// tiny subset, push a scripted scene (floor + player + bandit +
    /// bullet), and assert non-background pixels at the player screen
    /// position with background clear elsewhere. Skips where no GPU.
    #[test]
    fn offscreen_scene_draws_player_over_background() {
        use repame_sprite::{SpriteBatch, screen_camera};
        use repose_core::{Color, Rect, Scene, SceneNode};
        use repose_render_wgpu::{Callback, offscreen::OffscreenRenderer};

        let mut assets = subset();
        let mut world = scene_world(&assets);
        let instances = world_instances(&mut world, &assets);
        assert!(!instances.is_empty());

        let mut renderer = match OffscreenRenderer::new_blocking(256, 256, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP offscreen render test (no GPU): {e}");
                return;
            }
        };
        let mut batch = SpriteBatch::new(assets.batch_desc());
        batch.set_camera(screen_camera([256.0, 256.0]));
        batch.extend_uploads(assets.take_uploads());
        for s in &instances {
            batch.push_sprite(s);
        }
        assert!(!batch.is_empty());
        let scene = Scene {
            clear_color: Color::from_rgba(0, 0, 0, 255),
            nodes: vec![SceneNode::Callback {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 256.0,
                    h: 256.0,
                },
                payload: Callback::new(batch),
            }],
        };
        let px = renderer
            .render_rgba(&scene, Some([0.0, 0.0, 0.0, 1.0]))
            .expect("offscreen render");
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 256 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        // Player rect (128,128 ± cell): at least one pixel differs from
        // clear (transparent texels blend to black, so scan the rect).
        let mut painted = 0;
        for y in 112..144 {
            for x in 112..144 {
                if at(x, y) != [0, 0, 0, 255] {
                    painted += 1;
                }
            }
        }
        assert!(painted > 8, "player paints over the background");
        // Padded rect reaches cell (7, 7) -> world (240, 240): the dark
        // surround tile covers the far corner, not clear.
        assert_ne!(
            at(250, 250),
            [0, 0, 0, 255],
            "surround covers the padded rect"
        );
    }
}
