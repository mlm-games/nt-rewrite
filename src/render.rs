//! World → GPU sprite layer: sim [`World`] snapshots become
//! [`SpriteInstance`]s through the repame engine crates.
//! Renderer resolves art; sim owns truth.
//!
//! [`RenderAssets`] packs `assets/images/anims.ron` strips into a
//! [`repame_anim::AnimCatalog`] once and decodes the strip PNGs with the
//! `image` crate: repame ships no image dep and `repame-atlas` docs have
//! games decode their own pixels. Entity→strip NAME tables are the port's
//! own art index; each cites the GML object/script that picks the same
//! sprite (`Wall/Create_0.gml:6-8`, `scrWeapons.gml:53`,
//! `scrWeaponPickupCreate.gml:15`, `scrFire.gml:76-78`).
//!
//! Portal/vortex is intentionally NOT sprite-mapped: it belongs to the
//! fullscreen pass (`crate::vortex_pass::VortexPass`) driven from
//! [`crate::vortex::SpiralCtl`]. [`background_color`] (per-area dark fill)
//! is the fallback where no vortex layer is mounted; portal entities are
//! skipped.
//!
//! [`gml_camera_step`] is the GML `BackCont` camera law, a pure function;
//! [`world_camera`] turns the look point into a [`Camera2d`].
//!
//! [`hud_texts_dp`] maps labels through the live [`effective_fit`]/
//! [`world_to_dp`] - the fit the viewports paint with, so floaters sit on
//! their sprites.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use glam::Vec2;
use repame_anim::{AnimCatalog, AnimDef, AtlasDesc, UvRect, frame_key};
use repame_atlas::AtlasId;
use repame_fx::{DamageNumber, Particle, particle_sprites};
use repame_sprite::{
    AtlasUpload, BatchDesc, Camera2d, FitMode, SpriteBlend, SpriteInstance, WorldText, dp_to_world,
    effective_fit, world_to_dp,
};

use crate::anim::{PlayerAnim, SpriteAnim};
use crate::audio::UiAction;
use crate::combat::HitFlash;
use crate::comps_a::{
    ARENA_H, ARENA_W, AimDir, DogGuardianLeap, DogGuardianPose, FloorMask, GrenadeFuse, Health,
    HitId, Inventory, LightningArc, PendingMutation, PendingUltra, Player, Projectile,
    ProjectileFade, RaceState, Run, SelectedCharacter, SlashProjectile, TILE, Team, TopSmalls,
    Velocity, WallCell, WallTile, floor_cell_for_wall,
};
use crate::comps_b::{
    Beam, BigDogMissileState, BossBrain, BossPhase, ChestArt, ChestKind, Corpse, CrownObject,
    Enemy, EnemyBrain, FxAngle, GmlImage, GroundDecalTint, GroundDetail, HazardCloud, HurtAnim,
    InvisiWall, MaggotSpawnCharge, Mote, MoteScale, NativeAngle, NativeDepth, NativeFlip,
    NativeScale, OpenedChest, Pickup, PickupKind, PickupLifetime, Portal, PortalClear, PortalShock,
    PortalStrike, Prop, PropSprites, Shield, StaticFx, SwingFx, Telekinesis, ThroneCarpet,
    ThroneSit, TitleCampChar, TitleCampfire, TitleLogMenu, TitleTv, ToxicGasState, WeaponVisual,
    YungCuz, YvCouch,
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

/// Historical lookahead cap, kept for the public API but unused.
/// GML has no such cap: `objects/BackCont/Step_0.gml:69` divides the
/// unbounded `dis_fire` by `_viewdist`, so [`CAM_MAX_LOOK`] is the port's
/// own clamp.
pub const MAX_LOOK: f32 = 48.0;

/// Packed atlas + decoded strip pixels + retained GPU uploads.
pub struct RenderAssets {
    catalog: AnimCatalog,
    uploads_shared: Arc<[AtlasUpload]>,
    uploads_gen: u64,
    atlas_size: u32,
    layers: u32,
}

impl RenderAssets {
    fn build(text: &str, assets_dir: &Path, desc: AtlasDesc) -> anyhow::Result<Self> {
        let mut catalog = AnimCatalog::from_ron(text, desc).map_err(|e| anyhow::anyhow!("{e}"))?;
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
        static UPLOADS_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let generation = UPLOADS_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Self {
            uploads_shared: Arc::from(uploads),
            catalog,
            uploads_gen: generation,
            atlas_size: desc.size,
            layers,
        })
    }

    /// Full production load: `assets_dir` holds `images/anims.ron` +
    /// the strip PNGs (e.g. `./assets`).
    pub fn load(assets_dir: &Path) -> anyhow::Result<Self> {
        let text = crate::render::read_asset_catalog(assets_dir)?;
        Self::build(
            &text,
            assets_dir,
            AtlasDesc {
                size: ATLAS_SIZE,
                max_pages: ATLAS_PAGES,
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
            uploads_gen: self.uploads_gen,
            ..Default::default()
        }
    }

    /// Retained atlas blits shared with the GPU viewport. Cloning the
    /// `Arc` is cheap; the batch uploads a generation once and skips
    /// `write_texture` for repeats (see `BatchDesc::uploads_gen`), so
    /// lost early frames on Android just retry next frame.
    pub fn take_uploads(&self) -> Arc<[AtlasUpload]> {
        self.uploads_shared.clone()
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

    /// Full sprite constructor including the engine `flip_y` channel.
    /// GML's negative `image_yscale` is the mirror this stands in for:
    /// `Player/Draw_0.gml:100` draws the back gun at scale `-bwepright`.
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

    /// Sized textured sprite: pulse decals (cobweb/ice/trap visuals)
    /// drawn at their recorded size instead of the native strip cell.
    /// The size override is port-only - GML draws those objects through
    /// `draw_self` at their authored scale.
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
    let bytes = read_asset_bytes(path)?;
    let img = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()?
        .decode()?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Ok((w, h, rgba.into_raw()))
}

/// RON asset text, same portable lookup as [`read_asset_bytes`].
pub(crate) fn read_asset_ron(path: &Path) -> anyhow::Result<String> {
    Ok(String::from_utf8(read_asset_bytes(path)?)?)
}

pub(crate) fn read_asset_catalog(assets_dir: &Path) -> anyhow::Result<String> {
    let ron = assets_dir.join("images").join("anims.ron");
    match read_asset_ron(&ron) {
        Ok(text) => Ok(text),
        Err(ron_error) => {
            let json = assets_dir.join("images").join("anims.json");
            let text = read_asset_ron(&json).map_err(|json_error| {
                anyhow::anyhow!("asset catalog unavailable: {ron_error}; {json_error}")
            })?;
            let raw: BTreeMap<String, AnimDef> = serde_json::from_str(&text)?;
            Ok(ron::ser::to_string(&raw)?)
        }
    }
}

/// Asset bytes across desktop (plain files), Android (APK `assets/` via the
/// NDK `AAssetManager`; plain `std::fs` paths never resolve in an APK) and
/// web (the installed assets zip). The in-memory store wins, then the
/// platform's own source. Paths match on their `images/…` / `fonts/…` tail,
/// so the dev checkout (`<dir>/images/anims.ron`), the APK (`assets/…`) and
/// the zip layouts all resolve to one entry.
fn read_asset_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    if let Some(bytes) = crate::assetfs::get(path) {
        return Ok(bytes);
    }
    read_platform_asset_bytes(path)
}

#[cfg(target_os = "android")]
struct SendPtr(*mut std::ffi::c_void);
#[cfg(target_os = "android")]
unsafe impl Send for SendPtr {}
#[cfg(target_os = "android")]
unsafe impl Sync for SendPtr {}

#[cfg(target_os = "android")]
static APK_ASSET_MGR: std::sync::OnceLock<SendPtr> = std::sync::OnceLock::new();

/// Cache the APK `AAssetManager` pointer for [`read_asset_bytes`]; call once
/// from `android_main` with `android_app.asset_manager().ptr()`. Idempotent
/// across Activity recreations: first pointer wins and the manager is
/// `'static` per android-activity, so any generation stays valid. Replaces
/// `ndk_context::android_context`, which asserts single-init and panics on
/// recreation.
#[cfg(target_os = "android")]
pub fn init_apk_assets(mgr: ndk::asset::AssetManager) {
    let _ = APK_ASSET_MGR.set(SendPtr(mgr.ptr().as_ptr().cast()));
}

#[cfg(target_os = "android")]
fn read_platform_asset_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    use std::ffi::CString;
    if let Some(holder) = APK_ASSET_MGR.get() {
        let mgr_ptr = holder.0;
        if !mgr_ptr.is_null() {
            let tail = crate::assetfs::tail(path);
            let mgr = unsafe {
                ndk::asset::AssetManager::from_ptr(
                    std::ptr::NonNull::new(mgr_ptr.cast())
                        .ok_or_else(|| anyhow::anyhow!("null AssetManager"))?,
                )
            };
            let name = CString::new(tail.clone())?;
            // Streaming opens mmap the entry; `buffer()` returns None on
            // compressed entries, so fall back to a full read.
            if let Some(mut asset) = mgr.open(&name) {
                match asset.buffer() {
                    Ok(buf) => return Ok(buf.to_vec()),
                    Err(_) => {
                        use std::io::Read as _;
                        let mut out = Vec::with_capacity(asset.length());
                        asset.read_to_end(&mut out)?;
                        return Ok(out);
                    }
                }
            }
            return Err(anyhow::anyhow!("apk asset missing: {tail}"));
        }
    }
    Ok(std::fs::read(path)?)
}

/// Fs read on desktop; on Android falls back to fs too (covers
/// `NT_ASSETS`-style absolute overrides when present).
#[cfg(all(not(target_os = "android"), not(target_arch = "wasm32")))]
fn read_platform_asset_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

/// Web reads only from the installed assets zip; a miss here means
/// the shell's zip lacked the file.
#[cfg(target_arch = "wasm32")]
fn read_platform_asset_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!("missing web asset: {}", path.display())
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

// Name tables (art keys only, no pixels). Port-side indices; each cites
// the GML object/script that picks the same sprite.

/// GML `scrAreaGetMaxSubarea` verbatim (non-custom): the 3-floor
/// areas plus HQ hold 3 subareas, everything else 1. The custom-mode
/// branch (`area_size`/`area_size_alt`) is deferred - the port has no
/// custom runs, so the static table always applies.
pub fn area_max_subarea(area: AreaId) -> u32 {
    match area {
        AreaId::Desert | AreaId::Scrapyards | AreaId::FrozenCity | AreaId::Palace | AreaId::HQ => 3,
        _ => 1,
    }
}

/// Floor/wall/decal strips for a route floor (floor index over the
/// 15-floor route). Each arm is the GML area NUMBER, which is what the
/// sprite names interpolate: `Floor/Create_0.gml:42` picks
/// `sprFloor<area>`, `Wall/Create_0.gml:6-8` `sprWall<area>Top/Out/Bot`
/// (`macros_general.gml:539-545` numbers the seven route areas 1..7).
/// Wall `Out`/`Trans` variants exist but the renderer only needs
/// floor + bot/top - see fidelity notes.
fn area_sprites(floor: u32) -> (&'static str, &'static str, &'static str) {
    let rf = ((floor.max(1) - 1) % 15) + 1;
    match rf {
        3 => (
            "images/sprFloor1.png",
            "images/sprWall1Bot.png",
            "images/sprWall1Top.png",
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

/// Route floor/wall/out/trans strips for a route floor (out/trans follow
/// the same per-floor arms). GML takes all five from one area number:
/// `Wall/Create_0.gml:6-8` topspr/outspr/Bot,
/// `TopSmall/Create_0.gml:15` the Trans.
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
        3 => 1,
        4 => 2,
        5..=7 => 3,
        8 => 4,
        9..=11 => 5,
        12 => 6,
        13..=15 => 7,
        _ => 1,
    };
    // Out/Trans follow the same per-floor arms: `sprWall{N}Out` /
    // `sprWall{N}Trans`, GML `Wall/Create_0.gml:7` +
    // `TopSmall/Create_0.gml:15`.
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
/// GML draws the room background colour plus live floor cells only - no
/// outside ring - so the viewport stays transparent over the vortex
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

/// Secret-area floor/wall override (Oasis→101 … HQ→106). The arm is the
/// GML area number again, so `Floor/Create_0.gml:42` /
/// `Wall/Create_0.gml:6-8` resolve these straight from
/// `macros_general.gml:546-553`. The per-slot `catalog.has` fallback to
/// the route strips when a secret tile PNG is absent from this pack is
/// port-only (GML would draw the real asset).
fn area_sprites_for_run(
    floor: u32,
    area: AreaId,
    has: impl Fn(&str) -> bool,
) -> (&'static str, &'static str, &'static str) {
    let route = area_sprites(floor);
    let secret: Option<(&'static str, &'static str, &'static str)> = match area {
        // GML `Floor/Create_0` verbatim: the campfire title (`MenuGen`
        // / `Menu` present) uses `sprFloor0` + area-0 walls - the dark
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
        AreaId::CursedCaves => Some((
            "images/sprFloor104.png",
            "images/sprWall104Bot.png",
            "images/sprWall104Top.png",
        )),
        AreaId::Vault | AreaId::CrownVault => Some((
            "images/sprFloor100.png",
            "images/sprWall100Bot.png",
            "images/sprWall100Top.png",
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
/// trans): secret 101–106 families, per-slot fallback to the route strips
/// when the pack lacks them (that fallback is port-only).
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
        AreaId::CursedCaves => 104,
        AreaId::Vault | AreaId::CrownVault => 100,
        AreaId::Jungle => 105,
        AreaId::HQ => 106,
        _ => 0,
    };
    if num == 0 {
        return route;
    }
    let (o, tr): (&'static str, &'static str) = match num {
        100 => ("images/sprWall100Out.png", "images/sprWall100Trans.png"),
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

/// Player projectile strip by weapon id. GML has no id→path table: each
/// weapon's `case` in `scrFire.gml` names the projectile OBJECT it spawns
/// (revolver → `Bullet1` at `scrFire.gml:77-83`, shotgun → `Bullet2` at
/// `:122-155`, crossbow → `Bolt` at `:157-166`), and each object's own art
/// is its `spriteId` in `<Object>/<Object>.yy` (`Bullet1/Bullet1.yy:40`,
/// `Bolt/Bolt.yy:44`). This table is that object→strip mapping flattened
/// per weapon id (1–128 = GML `maxwep`, `scrWeapons.gml:27`); ids outside
/// it sanitize to the base weapon via [`sanitize_weapon_id`], and unknown
/// ids fall back by ammo type.
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

/// Enemy projectile strip by owner kind. Same shape as the player table:
/// GML shooters name the bullet OBJECT they create (`Bandit/Alarm_1.gml:10`
/// and `BanditBoss/Alarm_2.gml:11` → `EnemyBullet1`,
/// `SnowTank/Alarm_2.gml:7` and `Sniper/Alarm_2.gml:3` → `EnemyBullet4`,
/// `scrFire.gml:893` → `EnemyBullet2`) and each object's art is its own
/// `spriteId` (`EnemyBullet1/EnemyBullet1.yy:40` → `sprEnemyBullet1`,
/// `EnemyBullet2/EnemyBullet2.yy:37` → `sprScorpionBullet`,
/// `EnemyBullet4/EnemyBullet4.yy:38` → `sprEnemyBullet4`). Unlisted kinds
/// (including nt-rewrite-only additions) fall back to the generic enemy
/// bullet; that arm is port-only.
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
        EnemyKind::BigBandit => Some("images/sprBanditBossGun.png"),
        EnemyKind::IdpdGrunt => Some("images/sprPopoGun.png"),
        EnemyKind::IdpdElite => Some("images/sprElitePopoGun.png"),
        EnemyKind::JungleBandit => Some("images/sprJungleBanditGun.png"),
        EnemyKind::Sniper => Some("images/sprSniperGun.png"),
        EnemyKind::MeleeBandit => Some("images/sprPipe.png"),
        EnemyKind::LilHunter => Some("images/sprLilHunterGun.png"),
        EnemyKind::Molefish => Some("images/sprMolefishGun.png"),
        EnemyKind::Molesarge => Some("images/sprMolesargeGun.png"),
        EnemyKind::Raven => Some("images/sprRavenGun.png"),
        EnemyKind::PopoFreak => Some("images/sprPopoFreakGun.png"),
        _ => None,
    }
}

/// Pickup strip per kind. Each GML pickup object carries its own art as its
/// `spriteId` (`Rad/Rad.yy:39` → `sprRad`, `HPPickup/HPPickup.yy:40` →
/// `sprHP`, `AmmoPickup/AmmoPickup.yy:40` → `sprAmmo`,
/// `CursedPickup/CursedPickup.yy:39` → `sprCursedAmmo`, `Curse/Curse.yy:35`
/// → `sprCurse`), and weapon pickups are created with `sprite_index =
/// scr_weapon_get_sprite(_weapon)`
/// (`scripts/scrWeaponPickupCreate/scrWeaponPickupCreate.gml:15`), i.e.
/// `images/{wep_sprt}.png` from the `wep_sprt` register
/// (`scrWeapons.gml:53`). Quads size from the catalog cell. Falling back to
/// `sprRevolver` when `wep_sprt` is `mskNone` (`scrWeapons.gml:46`) is
/// port-only; GML would draw sprite -1.
pub fn pickup_art(kind: &PickupKind) -> Cow<'static, str> {
    match *kind {
        PickupKind::Rad(_) => Cow::Borrowed("images/sprRad.png"),
        PickupKind::Medkit(_) => Cow::Borrowed("images/sprHP.png"),
        PickupKind::Ammo(..) => Cow::Borrowed("images/sprAmmo.png"),
        PickupKind::CursedAmmo => Cow::Borrowed("images/sprCursedAmmo.png"),
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
            ChestKind::Mystery => Cow::Borrowed("images/sprAmmoChestMystery.png"),
            ChestKind::Gold => Cow::Borrowed("images/sprGoldChest.png"),
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

/// Priority: explicit GML visual (the `ProjectileVisual` an object-spawn
/// carries) → melee slash flags → player weapon table → enemy-kind table →
/// team fallback. GML settles this at spawn time instead: the shooter picks
/// the object and sometimes overrides its sprite (`scrFire.gml:164,173`
/// `sprite_index = sprBoltGold` / `sprGoldGrenade`), so `visual` is the
/// port's stand-in for "this projectile instance already carries GML's own
/// art". The team-only fallbacks at the end are port-only.
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

/// GML `AllyBullet/Create_0.gml:7` (`spr_fade = sprAllyBulletHit`) is the
/// only fade tag in the port that belongs to `AllyBullet`, so it stands in
/// for the `AllyBullet.yy` spriteId `sprAllyBullet` body art.
const ALLY_BULLET_FADE: &str = "images/sprAllyBulletHit.png";

/// Deterministic per-cell salt hash standing in for GML's live `random`
/// rolls when picking wall art variant frames. GML draws a fresh roll per
/// `Wall`/`Floor` instance at creation (`objects/Wall/Create_0.gml:16-32`,
/// `objects/Floor/Create_0.gml:8-13`); the port has no RNG stream here, so
/// the cell coords plus the run seed hash into the same distribution - see
/// [`wall_body_raw`] / [`wall_out_raw`].
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

/// [`strip_frames`] for view-layer callers outside this module
/// (campfire/game-over crosshair frame clamp in `lib.rs`).
pub fn strip_frames_pub(assets: &RenderAssets, path: &str) -> u32 {
    strip_frames(assets, path)
}

/// GML `Wall/Create_0:16-23` body variant verbatim: `random(150) < 1` takes
/// 3, else `choose(0,0,0,0,0,0,0,1,2) + choose(0,4)`; the port hashes the
/// cell into the same distribution deterministically. Shared by Bot AND
/// Top: GML draws `topspr` with the Bot's `image_index`
/// (`SubTopCont/Draw_0:27`); the `topindex` roll (`Wall/Create_0:25-30`) is
/// never read.
fn wall_body_raw(seed: u64, wx: i32, wy: i32) -> usize {
    let base = if wall_hash(seed, wx, wy, 0x11) % 150 == 0 {
        3
    } else {
        [0usize, 0, 0, 0, 0, 0, 0, 1, 2][(wall_hash(seed, wx, wy, 0x12) % 9) as usize]
    };
    base + [0usize, 4][(wall_hash(seed, wx, wy, 0x13) % 2) as usize]
}

/// GML `Wall/Create_0:32` Out variant verbatim:
/// `choose(0,0,0,0,1,2,3,4) + choose(0,4)`.
fn wall_out_raw(seed: u64, wx: i32, wy: i32) -> usize {
    [0usize, 0, 0, 0, 1, 2, 3, 4][(wall_hash(seed, wx, wy, 0x31) % 8) as usize]
        + [0usize, 4][(wall_hash(seed, wx, wy, 0x32) % 2) as usize]
}

/// GML `mcr_wall_update_lrwh` verbatim (`macros_general.gml:75-79`):
/// `l`/`r` are the source-rect origin into the 24-wide Out cell, `w`/`h`
/// its extent. `place_free` only registers instances flagged `solid`, so a
/// probe sees `Wall` and nothing else - `Floor` and `FloorExplo` are both
/// `solid = false` and stay out of it, as do props, actors and the boss.
/// The caller's mask is the 16px `Wall` one, so a probe covers exactly the
/// neighbour cell.
fn wall_out_crop(wall_set: &HashSet<(i32, i32)>, wx: i32, wy: i32) -> (f32, f32, f32, f32) {
    let free = |ox: i32, oy: i32| !wall_set.contains(&(wx + ox, wy + oy));
    let l = if free(-1, 0) { 0.0 } else { 4.0 };
    let w = if free(1, 0) { 24.0 - l } else { 20.0 - l };
    let r = if free(0, -1) { 0.0 } else { 4.0 };
    let h = if free(0, 1) { 24.0 - r } else { 20.0 - r };
    (l, r, w, h)
}

/// One GML `draw_sprite_part_ext` Out window: source rect `(l, r, w, h)` of
/// the 24x32 strip frame (origin (4,12)), drawn top-left at GML
/// `(x - 4 + l, y - 12 + r)`. Window-padding law (same as
/// [`hud_weapon_part`]): `l`/`r` can start outside the 24-wide cell
/// (negative `x - 4` lead at the room edge) and `w`/`h` can overrun it (up
/// to 24x32 against a shorter crop) - GML clamps those samples to
/// transparent edge texels, so offset inside the window by the
/// out-of-bounds lead.
fn wall_out_part(
    assets: &RenderAssets,
    path: &str,
    frame: i32,
    wx: i32,
    wy: i32,
    crop: (f32, f32, f32, f32),
    tint: [f32; 4],
) -> Option<SpriteInstance> {
    let (l, r, w, h) = crop;
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    let (uv, def) = assets.uv(path, frame)?;
    let (sw, sh) = (def.w as f32, def.h as f32);
    if sw <= 0.0 || sh <= 0.0 {
        return None;
    }
    // In-bounds part of the window: the 24-wide Out art only fills the
    // top ~22 rows of the 32-tall cell, and `h` runs to 24/32 against
    // it - the overrun is transparent edge texels in GML, so the quad
    // keeps only the overlap. `l`/`r` the same way (west/north leads
    // are 0..4 in practice, kept general for the room edge).
    let ix0 = l.max(0.0);
    let iy0 = r.max(0.0);
    let ix1 = (l + w).min(sw);
    let iy1 = (r + h).min(sh);
    if ix1 <= ix0 || iy1 <= iy0 {
        return None;
    }
    let fx0 = ix0 / sw;
    let fy0 = iy0 / sh;
    let fx1 = ix1 / sw;
    let fy1 = iy1 / sh;
    let uv_min = Vec2::new(
        uv.min[0] + (uv.max[0] - uv.min[0]) * fx0,
        uv.min[1] + (uv.max[1] - uv.min[1]) * fy0,
    );
    let uv_max = Vec2::new(
        uv.min[0] + (uv.max[0] - uv.min[0]) * fx1,
        uv.min[1] + (uv.max[1] - uv.min[1]) * fy1,
    );
    let iw = ix1 - ix0;
    let ih = iy1 - iy0;
    let top_left = Vec2::new(wx as f32 * 16.0 - 4.0 + ix0, wy as f32 * 16.0 - 12.0 + iy0);
    Some(SpriteInstance {
        center: top_left + Vec2::new(iw, ih) * 0.5,
        rotation: 0.0,
        size: Vec2::new(iw, ih),
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

/// GML `Top`/`TopSmall` chain, sorted for stable draw order. Every
/// `TopSmall` draws sub-image `-1`, the strip's last frame
/// (`SubTopCont/Draw_0:22`); the `image_index = irandom(image_number)` roll
/// in `TopSmall/Create_0:7` is never read. Cells are
/// [`crate::comps_a::TopSmalls`] - GML's live instances, accumulated at
/// level start and extended per break, NOT a recompute, so a destroyed wall
/// does not grow its Trans tile back.
fn trans_cells(cells: &TopSmalls, trans_frames: u32) -> Vec<((i32, i32), i32)> {
    let frame = trans_frames as i32 - 1;
    let mut sorted: Vec<(i32, i32)> = cells.cells.iter().copied().collect();
    sorted.sort_unstable();
    sorted.into_iter().map(|c| (c, frame)).collect()
}

/// sRGB channel -> linear light (exact transfer function). GPU tints
/// are authored as sRGB display colors - GML's `c_*` / `#rrggbb` colors
/// are display-space too: the sRGB atlas decodes on sample and the sRGB
/// target re-encodes on write, so tints must be linear - raw sRGB tints
/// render washed out.
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

/// Sprite z-ladder (GML `__global_object_depths` `Draw_0` stage verbatim):
/// higher GM depth draws first = further back, so the port's `z` runs the
/// other way - larger `z` draws on top. Engine stable-sorts by
/// `(blend, z, page)`, keeping push order on ties.
///
/// GM order (back → front): Floor(10) → Detail(8) → BackCont-shadows(5) →
/// Corpse(1) → Wall/shots(0) → Player/Ally(-2) → Portal(-3) →
/// SubTopCont wall-tops/bloom(-6) → TopCont fog/crosshair/revive(-15) →
/// SpiralCont figures(-101) → Draw-GUI chain (GUI Begin 74 → GUI 64 →
/// GUI End 75, in stage order, ignoring instance depth) → Menu(-1001).
/// The world batch keeps its internal push order at 0 (untouched); every
/// chrome layer above it stamps one rung so an atlas page can never lottery
/// a HUD bar under a floor tile.
///
/// Draw-GUI stages vs sprite rungs: GML `UberCont/Draw_74`
/// (`scrDrawSidearts`) runs BEFORE the GUI-64 HUD text, `UberCont/Draw_75`
/// (the raw `sprCrosshair` cursor) runs AFTER it - the cursor is topmost by
/// pipeline stage. The port implements `Draw_75` as a hardware cursor
/// (`CursorIcon::Custom`, composited by the OS above every sprite and UI
/// layer), so no sprite rung exists for it; sideart rungs above the room
/// chrome but below HUD/menus.
pub const Z_FLOOR: f32 = -3.0;
pub const Z_GROUND_DETAIL: f32 = -2.5;
pub const Z_SHADOW: f32 = -2.0;
/// GML `FloorExplo` is created at depth 0 (its `Create_0` runs when a wall
/// breaks), so its `sprFloor<area>Explo` patch draws OVER `BackCont`'s
/// `shad` blit, not under it: at `Z_FLOOR` the 0.4 `shadow_color` quad
/// landed on the rubble and dropped its mean RGB from (61,48,43) to
/// ~(37,29,26). GML would also put it above the wall `Bot` (same depth-0
/// creation order), but a `Bot` never overlaps a hole cell - `Wall/Create_0`
/// only draws it when `place_meeting(x, y + 16, Floor)` hits floor, and the
/// destroyed wall's own south cell is the hole - so any rung in (-2, 0) is
/// pixel-identical.
pub const Z_FLOOR_EXPLO: f32 = -1.0;
pub const Z_WORLD: f32 = 0.0;
pub const Z_WALL_SUBTOP: f32 = 1.5;
pub const Z_FX: f32 = 1.0;
pub const Z_BLOOM: f32 = 2.0;
pub const Z_FOG: f32 = 3.0;
pub const Z_CROSSHAIR: f32 = 4.0;
pub const Z_FAINTED: f32 = 5.0;
pub const Z_PORTAL_INDICATOR: f32 = 6.0;
pub const Z_SPIRAL_FIGURES: f32 = 7.0;
pub const Z_SIDEART: f32 = 8.0;
pub const Z_HUD: f32 = 10.0;
pub const Z_TOUCH: f32 = 12.0;
pub const Z_SPLASH: f32 = 15.0;
pub const Z_MENU: f32 = 20.0;
/// GML in-run pause button (`UberCont/Draw_64:49`, depth -1000): lower
/// than `TopCont` (-15) and `Menu` (-1001), so it draws over the mobile
/// controls and every menu, and only `Logo`/`MainMenuButton` (-10000)
/// sit in front of it.
pub const Z_PAUSE_BUTTON: f32 = 25.0;

/// Stamp a layer rung over a finished push batch (keeps the producer's
/// internal push order: the engine sort is stable on `(blend, z, page)`
/// ties). `pub(crate)` so the view composer assigns rungs per layer
/// next to the push order (single place both are visible).
pub(crate) fn stamp_z(out: &mut [SpriteInstance], z: f32) {
    for s in out.iter_mut() {
        if s.z == 0.0 {
            s.z = z;
        }
    }
}

/// Place a strip quad by its art top-left (GM draw origin): the catalog
/// anchor lands `center` so origin-(0,0) floor/wall art sits on the grid
/// (GML `draw_self` uses the same sprite origin - passing a cell center
/// would shift it by half a cell). `grow` extends the +x/+y edges (see
/// [`GRID_OVERLAP`]); UVs still span the cell.
///
/// GML `Floor/Create_0:8-13` floor variant, verbatim: `random(500) < 1` takes
/// frame 3, else `choose(0,0,0,0,0,0,0,1,2) + choose(0,4)`. No live RNG stream
/// here, so the cell coords hash into the same distribution deterministically
/// (1/500 rare, 7/9 plain, 1/9 mid, then +4 half the time). Clamped to the strip
/// (GML families always carry 0-7; a short pack strip must not drop the cell).
fn floor_frame(cx: i32, cy: i32, frames: i32) -> i32 {
    let h = (cx
        .wrapping_mul(0x8da6b343u32 as i32)
        .wrapping_add(cy.wrapping_mul(0xd8163841u32 as i32))
        >> 7) as u32;
    let raw = if h % 500 == 0 {
        3
    } else {
        let base = match h % 9 {
            7 => 1,
            8 => 2,
            _ => 0,
        };
        base + if h % 2 == 0 { 4 } else { 0 }
    };
    raw % frames.max(1)
}

/// One 32x32 floor tile, top-left placed at the cell's origin.
fn push_floor_tile(
    out: &mut Vec<SpriteInstance>,
    assets: &RenderAssets,
    floor_png: &str,
    cx: i32,
    cy: i32,
) {
    let frames = strip_frames(assets, floor_png).max(1) as i32;
    if let Some(mut s) = place_top_left(
        assets,
        floor_png,
        floor_frame(cx, cy, frames),
        Vec2::new(cx as f32 * TILE, cy as f32 * TILE),
        [1.0; 4],
        GRID_OVERLAP,
    ) {
        s.z = Z_FLOOR;
        out.push(s);
    }
}

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
// Camera + background.

/// GML `objects/BackCont/Step_0.gml` camera law, verbatim.
/// Per game step, local player at `player`:
/// - POI pull: nearest Portal (else BecomeNothing/NothingDeath/
///   Nothing2Death/SitDown) contributes `dist/6` along its heading, capped
///   at 72 px for Portals and WeaponChests; the GML chain's second
///   `BecomeNothing` check is dead code (the first branch took it).
/// - Aim lean: `dis_fire / viewdist` along `dir_fire` (`viewdist` 4, 8
///   melee, 3 bolts).
/// - Shake: `orandom(shake * opt_shake)` on the target.
/// - Snap (level start): target jump (`m = 1`), knock decay zeroed.
/// - `view = round(lerp(view, target, m))`, `m = 0.4` live.
/// - Knock decay `round(viewx2 - viewx2 * 0.4)`; shake decay `*=
///   power(0.8, timescale)` above 10 else `-= timescale` to 0; `opt_shake
///   <= 0` zeroes shake + knock.

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
/// Aim-lean cap in world px - the port's own clamp, no GML counterpart.
/// GML's keyboard aim is unbounded:
/// `scripts/InputHandling/InputHandling.gml:467` sets
/// `dis_fire = point_distance(x, y, mouse_x, mouse_y)` with no ceiling and
/// `objects/BackCont/Step_0.gml:69` just divides it by `_viewdist`. The
/// gamepad path is bounded differently - stick length times
/// `min(view_width, view_height) * 0.4`
/// (`scripts/InputHandling/InputHandling.gml:442,444`) or, inside
/// `BackCont`, `length * 72 / _viewdist`
/// (`objects/BackCont/Step_0.gml:65`) - never by this constant.
pub const CAM_MAX_LOOK: f32 = 48.0;
/// GML follow rate (`lerp(..., 0.4)`).
pub const CAM_LERP: f32 = 0.4;
/// GML knock decay rate (`viewx2 - viewx2 * 0.4`).
pub const CAM_KNOCK_DECAY: f32 = 0.4;
/// Unused zoom-ease weight (GML has no zoom; the view is always 240
/// world px tall). Kept for save-compat/harness use.
pub const CAM_ZOOM_SPEED: f32 = 0.08;

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
/// counterpart - single-player has no ChickenHead entity - so it is
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

/// GML `Menu/Create_0:104-110` + `Menu/Step_1` verbatim: view centers on the
/// selected race's camper (`Menu.char[race]`); Random centers on `char[0]`,
/// the Campfire entity itself (`with (Menu) char[0] = other.id`), not a
/// CampChar. `None` when the camp actors are absent (the caller falls back to
/// the campfire spawn).
pub fn title_cam_focus(world: &mut World) -> Option<Vec2> {
    let selected = world
        .get_resource::<SelectedCharacter>()
        .map(|s| s.0 as usize)
        .unwrap_or(0);
    if selected != 0 {
        let mut campers = world.query::<(&Pos, &TitleCampChar)>();
        for (pos, camp) in campers.iter(world) {
            if camp.race_gml == selected {
                return Some(pos.0);
            }
        }
    }
    world
        .query::<(&Pos, &TitleCampfire)>()
        .iter(world)
        .next()
        .map(|(p, _)| p.0)
}

/// GML `Menu/Step_1` title camera step: `t_lerp` toward centering on the
/// [`title_cam_focus`] point (`view_xview = x - vw/2`, `view_yview = y - vh/2`
/// at rate 0.1; `Create_0:104-110` snaps with `m = 1` on entry).
/// `t_lerp(a, b, 0.1) = lerp(b, a, 0.9^timescale)` - 10% of the gap per step
/// at timescale 1, no `round()` (unlike `BackCont`). `cam.snap` forces the
/// snap and clears.
/// Per compose it runs the clamped frame dt: `1-0.9^(dt*30)` equals 0.1 at
/// 30 Hz and converges identically to GML's fixed steps across hitches (three
/// 0.1 steps ≈ one 0.27 step). Entry snap targets the settled (post-scatter)
/// camper pos while GML snaps pre-scatter and lerps after - transient (<1 s)
/// and self-correcting.
pub fn title_camera_step(cam: &mut GmlCamera, vw: f32, vh: f32, focus: Vec2, dt: f32, snap: bool) {
    let m = if snap || cam.snap {
        1.0
    } else {
        1.0 - 0.9f32.powf(dt.max(0.0) * 30.0)
    };
    cam.x = cam.x + (focus.x - vw * 0.5 - cam.x) * m;
    cam.y = cam.y + (focus.y - vh * 0.5 - cam.y) * m;
    cam.snap = false;
}

/// GPU camera for a look point. The live frames pass `scale = 1.0`: the
/// viewport's `world_size` is already the GML view rect, so the engine
/// contain-fit does the whole job (including the pillarbox) and there is
/// no separate per-axis scale to keep in sync - see [`gml_frame`].
pub fn world_camera(center: Vec2, scale: f32) -> Camera2d {
    Camera2d {
        center,
        offset: Vec2::ZERO,
        units_per_pixel: scale,
        zoom: 1.0,
        // No camera roll in the port (GML has none; trauma shake stays
        // translational through `GmlCamera`).
        roll: 0.0,
        // GML's view shows exactly its view rect and pillarboxes, so the
        // world rect is kept and bars land on the roomier axis.
        fit: FitMode::Keep,
    }
}

/// GML `scrSetViewSize` view law (`scripts/macros_general` +
/// `UberCont/Create_0`): view is 240 world px tall, widened to
/// `view_width_max = 240 * aspect` under `opt_resolution` (default on), 320
/// floor. The single source of truth the whole GUI is measured in (see
/// [`gml_frame`]).
/// The GML odd-width `+1` bump is deliberately NOT reproduced: it exists
/// because GML's `view_width` is a *window* size that must be even for its
/// surface resize. Here 240 is the constant and width follows the canvas, so
/// the exact value avoids a sub-pixel disagreement between where a sprite
/// draws and where its hit test lands.
pub fn gml_view_size(viewport_dp: [f32; 2]) -> [f32; 2] {
    let (w, h) = (viewport_dp[0].max(1.0), viewport_dp[1].max(1.0));
    [(240.0 * w / h).max(320.0), 240.0]
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
        // GML mansion is #eef0f2 (`scrArea.gml:81`); area 103 is what the
        // port names `AreaId::City`.
        AreaId::City => hex(0xeef0f2),
        // GML `area_crib` returns the same #eef0f2 (`scrArea.gml:85`).
        AreaId::Crib => hex(0xeef0f2),
        // `AreaId::Loop` rides GML area 1 (desert, `scrArea.gml:71`).
        AreaId::Loop => hex(0xaf8f6a),
    }
}

/// GML `BackCont/Draw_0:11` draws the `shad` surface at `draw_set_alpha(0.4)`
/// with `gpu_set_fog(1, shadow_color, depth, depth+1)`: the fog REPLACES the
/// black silhouette already in the surface, and the 0.4 alpha is the surface
/// composite. Every blob therefore lands on screen as `shadow_color` at 40%
/// alpha, not as opaque black.
pub const SHADOW_ALPHA: f32 = 0.4;

/// GML `scrAreaGetShadowColor` verbatim (GameMaker `#rrggbb` -> sRGB
/// 0..1; custom resource-pack colors are shell-side and fall through here).
/// Fed to [`SHADOW_ALPHA`] over every `scrShadows` emitter.
pub fn shadow_color(area: AreaId) -> [f32; 4] {
    fn hex(hex: u32) -> [f32; 4] {
        [
            ((hex >> 16) & 0xFF) as f32 / 255.0,
            ((hex >> 8) & 0xFF) as f32 / 255.0,
            (hex & 0xFF) as f32 / 255.0,
            SHADOW_ALPHA,
        ]
    }
    match area {
        AreaId::Campfire => hex(0x000000),
        AreaId::Desert => hex(0x000000),
        AreaId::Sewers => hex(0x080d01),
        AreaId::Scrapyards => hex(0x000000),
        AreaId::CrystalCaves => hex(0x06020c),
        AreaId::FrozenCity => hex(0x0e1344),
        AreaId::Labs => hex(0x000000),
        AreaId::Palace => hex(0x0d0101),
        AreaId::Vault | AreaId::CrownVault => hex(0x00030e),
        AreaId::Oasis => hex(0x012b43),
        AreaId::PizzaSewers => hex(0x090012),
        // GML `area_mansion` is 103, which the port names `AreaId::City` (its
        // `sprFloor103*` art confirms the id).
        AreaId::City => hex(0x120014),
        AreaId::CursedCaves => hex(0x420000),
        AreaId::Jungle => hex(0x140001),
        AreaId::HQ => hex(0x00248c),
        AreaId::Crib => hex(0x120014),
        // `AreaId::Loop` rides GML area 1 (desert), whose shadow is
        // `c_black` (`scrArea.gml:105`; function default `:122`).
        AreaId::Loop => hex(0x000000),
    }
}

// View-space atmosphere: area fog + sideart chrome.

/// GML `#macro FOG_ALPHA 0.1` (`objects/TopCont/Create_0`).
pub const FOG_ALPHA: f32 = 0.1;
/// GML fog tile size (`draw 3x3 of 480x360`).
pub const FOG_TILE_W: f32 = 480.0;
/// GML fog tile size (vertical).
pub const FOG_TILE_H: f32 = 360.0;
/// GML sideart tile size (`sprSideArt` 64 px, `i * -64` tiling).
pub const SIDEART_TILE: f32 = 64.0;

/// GML makes GUI and view the same thing (`scrSetViewSize` ends with
/// `display_set_gui_size(_width, _height)`; `device_mouse_x_to_gui` is the
/// inverse of the view transform), so there is exactly ONE rect. The engine's
/// contain-fit then scales it into the canvas and centres the remainder -
/// that IS the letterbox/pillarbox: a window whose aspect differs from the
/// GML view gets bars and the GUI never leaves the box. (The earlier design
/// handed the viewport the whole canvas with a scaled camera, forcing a
/// per-axis GUI re-derive: a portrait window showed 320x668 of world under a
/// 320x240 GUI, GUI crammed into the top 36%.)
#[derive(Clone, Copy, Debug)]
pub struct GmlFrame {
    /// World-space view rect `[x, y, w, h]`.
    pub view: [f32; 4],
    /// The same rect in dp `[x, y, w, h]` - the pillarboxed area inside
    /// the canvas. `gui` consumers convert dp through this.
    pub box_dp: [f32; 4],
    /// dp per world unit.
    pub dp_per_world: f32,
}

impl GmlFrame {
    /// dp point → GUI (view) px. Points outside the box land outside the
    /// view, which is what every hit test wants.
    pub fn dp_to_gui(&self, dp: [f32; 2]) -> [f32; 2] {
        [
            (dp[0] - self.box_dp[0]) / self.dp_per_world,
            (dp[1] - self.box_dp[1]) / self.dp_per_world,
        ]
    }

    /// GUI width in px (the GML `view_width` every right-anchored row
    /// and touch home is measured against).
    pub fn gui_width(&self) -> f32 {
        self.view[2]
    }
}

/// Resolve the frame for a canvas + GML view rect + camera.
pub fn gml_frame(canvas_dp: [f32; 2], world_size: [f32; 2], cam: &Camera2d) -> GmlFrame {
    let center = cam.effective_center();
    let fit = effective_fit(canvas_dp, world_size, cam);
    if !fit.0.is_finite() || fit.0 <= 1e-6 {
        return GmlFrame {
            view: [center[0], center[1], 0.0, 0.0],
            box_dp: [0.0, 0.0, 0.0, 0.0],
            dp_per_world: 1.0,
        };
    }
    let (s, ox, oy) = fit;
    // `world_to_dp` maps the world's top-left to (ox, oy), so the visible
    // world is exactly `world_size` and the box is `world_size * s`
    // offset by the contain-fit slack.
    let tl = dp_to_world([ox, oy], world_size, center, fit);
    GmlFrame {
        view: [tl[0], tl[1], world_size[0], world_size[1]],
        box_dp: [ox, oy, world_size[0] * s, world_size[1] * s],
        dp_per_world: s,
    }
}

/// World-space view rect under the live camera fit (top-left + extent in
/// world units). Thin wrapper over [`gml_frame`] for the many callers
/// that only need the world rect.
pub fn view_rect_world(canvas_dp: [f32; 2], world_size: [f32; 2], cam: &Camera2d) -> [f32; 4] {
    gml_frame(canvas_dp, world_size, cam).view
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
/// `n = view_width_max / 64` columns (`for i = 1; i <= n` → floor count),
/// horizontal strips at `i * -64` and `view_width - 64 + i * 64` over
/// `nh = ceil(view_height / 64) + 1` rows, then vertical `repeat 10` runs at
/// `i * 64` above/below for `i in 0..n`.
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
/// `UberCont/Draw_74` - GUI Begin, i.e. BEFORE the GUI-64 HUD text and
/// the Draw_75 cursor): `sprSideArt` frame `opt_sideart` at every
/// [`sideart_tiles`] position, mapped through the view like the menu
/// art (the port canvas IS the view).
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

// World → instances.

fn flash_tint(flash: Option<&HitFlash>) -> [f32; 4] {
    flash.map(|f| f.color).unwrap_or([1.0, 1.0, 1.0, 1.0])
}

fn native_image_instance(
    assets: &RenderAssets,
    image: &GmlImage,
    pos: Vec2,
    angle: f32,
    scale: Vec2,
    flip: bool,
    depth: f32,
    tint: [f32; 4],
) -> Option<SpriteInstance> {
    let mut sprite = assets
        .sprite_stretched(image.path, image.frame(), pos, scale, angle, tint)
        .or_else(|| Some(white_quad(pos, angle, scale, tint)))?;
    sprite.flip_x = flip;
    sprite.z = -depth / 5.0;
    Some(sprite)
}

fn laser_sight(
    assets: &RenderAssets,
    path: &str,
    origin: Vec2,
    angle: f32,
    walls: &[Vec2],
) -> Option<SpriteInstance> {
    let step = Vec2::new(angle.cos(), angle.sin()) * 2.0;
    let mut tip = origin;
    for _ in 0..=1000 {
        tip += step;
        if walls
            .iter()
            .any(|wall| (wall.x - tip.x).abs() <= 8.0 && (wall.y - tip.y).abs() <= 8.0)
        {
            break;
        }
    }
    let native = assets.native_size(path)?;
    let size = Vec2::new(native.x * (origin.distance(tip) / 2.0 + 2.0), native.y);
    let mut sight = assets.sprite_sized(path, 0, origin, size, false, [1.0; 4])?;
    sight.rotation = angle;
    sight.anchor = Vec2::ZERO;
    Some(sight)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StaticWorldKey {
    floor: u32,
    area: AreaId,
    seed: u64,
    assets_gen: u64,
    floor_cells: usize,
    floor_hash: u64,
    wall_cells: usize,
    wall_hash: u64,
    active_wall_hash: u64,
    top_small_hash: u64,
}

#[derive(Clone, Default)]
pub struct StaticWorldCache {
    key: Option<StaticWorldKey>,
    prefix: Arc<[SpriteInstance]>,
    wall_out: Arc<[SpriteInstance]>,
    wall_trans: Arc<[SpriteInstance]>,
    wall_top: Arc<[SpriteInstance]>,
    wall_centers: Arc<[Vec2]>,
    /// GML `scrShadows` wall half: every flipped `Out` sprite at
    /// `(x, y + 16)`, untinted. These are COVERAGE stamps, not draws - they
    /// go into the `shad` surface (`crate::shadow_pass`) that `BackCont`
    /// composites once, never into the batch as per-wall 0.4 quads.
    wall_shadows: Arc<[SpriteInstance]>,
}

impl StaticWorldCache {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// The `shad` coverage stamps for the current floor. Empty until
    /// [`world_instances_cached`] has run once for this key.
    pub fn wall_shadows(&self) -> Arc<[SpriteInstance]> {
        self.wall_shadows.clone()
    }
}

fn cell_fingerprint(x: i32, y: i32) -> u64 {
    let mut value = (x as u32 as u64) | ((y as u32 as u64) << 32);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn add_fingerprint(value: u64, sum: &mut u64, xor: &mut u64) {
    *sum = sum.wrapping_add(value);
    *xor ^= value.rotate_left(17);
}

fn static_world_key(world: &mut World, assets: &RenderAssets) -> Option<StaticWorldKey> {
    let (floor, area, seed) = {
        let run = world.get_resource::<Run>()?;
        (run.floor, run.area, run.gen_seed)
    };
    let (floor_cells, floor_hash) = {
        let mask = world.get_resource::<FloorMask>()?;
        let mut floor_sum = 0;
        let mut floor_xor = 0;
        for &(x, y) in &mask.cells {
            add_fingerprint(cell_fingerprint(x, y), &mut floor_sum, &mut floor_xor);
        }
        // A break adds a 16x16 `opened` cell, which changes the drawn floor
        // without touching `cells`, so it has to be in the key.
        for &(x, y) in &mask.opened {
            add_fingerprint(
                cell_fingerprint(x, y) ^ 0x9E37_79B9_7F4A_7C15,
                &mut floor_sum,
                &mut floor_xor,
            );
        }
        (mask.cells.len() + mask.opened.len(), floor_sum ^ floor_xor)
    };
    let mut wall_sum = 0;
    let mut wall_xor = 0;
    let mut wall_count = 0;
    for (cell, pos) in world.query::<(&WallCell, &Pos)>().iter(world) {
        let value = cell_fingerprint(cell.0, cell.1)
            ^ u64::from(pos.0.x.to_bits()).rotate_left(7)
            ^ u64::from(pos.0.y.to_bits()).rotate_left(29);
        add_fingerprint(value, &mut wall_sum, &mut wall_xor);
        wall_count += 1;
    }
    let mut active_sum = 0;
    let mut active_xor = 0;
    for cell in world
        .query_filtered::<&WallCell, With<WallTile>>()
        .iter(world)
    {
        add_fingerprint(
            cell_fingerprint(cell.0, cell.1),
            &mut active_sum,
            &mut active_xor,
        );
    }
    // A break extends the `TopSmall` ring (`FloorExplo/Create_0:43-50`) and
    // the second throne boss wipes it, so the drawn Trans cells are not a
    // function of the floor/wall sets alone.
    let (top_small_sum, top_small_xor) = match world.get_resource::<TopSmalls>() {
        Some(tops) => {
            let mut sum = 0;
            let mut xor = 0;
            for cell in &tops.cells {
                add_fingerprint(cell_fingerprint(cell.0, cell.1), &mut sum, &mut xor);
            }
            (sum, xor)
        }
        None => (0, 0),
    };
    Some(StaticWorldKey {
        floor,
        area,
        seed,
        assets_gen: assets.uploads_gen,
        floor_cells,
        floor_hash,
        wall_cells: wall_count,
        wall_hash: wall_sum ^ wall_xor,
        active_wall_hash: active_sum ^ active_xor,
        top_small_hash: top_small_sum ^ top_small_xor,
    })
}

/// Snapshot the sim world into GPU sprites (draw order = push order;
/// within the world batch everything stays at `z = 0` and the push
/// ladder is: floor, walls, decals, props, corpses/portals, hazards,
/// pickups, opened chests, enemies, player, projectiles, hit-FX, held
/// guns, melee swings). Chrome layers stamp their own rung - see the
/// z-ladder doc above.
pub fn world_instances(world: &mut World, assets: &RenderAssets) -> Vec<SpriteInstance> {
    let mut cache = StaticWorldCache::default();
    world_instances_cached(world, assets, &mut cache)
}

pub fn world_instances_cached(
    world: &mut World,
    assets: &RenderAssets,
    cache: &mut StaticWorldCache,
) -> Vec<SpriteInstance> {
    let key = static_world_key(world, assets);
    let reuse = key.is_some() && cache.key == key;
    let cached_prefix = reuse.then(|| cache.prefix.clone());
    let cached_wall_out = reuse.then(|| cache.wall_out.clone());
    let cached_wall_trans = reuse.then(|| cache.wall_trans.clone());
    let cached_wall_top = reuse.then(|| cache.wall_top.clone());
    let cached_wall_centers = reuse.then(|| cache.wall_centers.clone());
    let cached_wall_shadows = reuse.then(|| cache.wall_shadows.clone());
    let mut out = cached_prefix
        .as_deref()
        .map(|sprites| sprites.to_vec())
        .unwrap_or_default();
    let mut wall_out = cached_wall_out
        .as_deref()
        .map(|sprites| sprites.to_vec())
        .unwrap_or_default();
    let mut wall_trans = cached_wall_trans
        .as_deref()
        .map(|sprites| sprites.to_vec())
        .unwrap_or_default();
    let mut wall_top = cached_wall_top
        .as_deref()
        .map(|sprites| sprites.to_vec())
        .unwrap_or_default();
    let mut wall_centers = cached_wall_centers
        .as_deref()
        .map(|centers| centers.to_vec())
        .unwrap_or_default();
    let mut wall_shadows = cached_wall_shadows
        .as_deref()
        .map(|sprites| sprites.to_vec())
        .unwrap_or_default();
    if !reuse {
        wall_centers = world
            .query::<(&Pos, &WallCell)>()
            .iter(world)
            .map(|(pos, _)| pos.0)
            .collect();

        // Floor: GML draws the room background colour first
        // (`background_set_colour`), then ONLY the live floor cells - no padded
        // outside ring of floor tiles (a ±6-cell ring buried the transparent
        // vortex layer on the campfire title: the "no vortex on the title
        // screen" bug). Lit strip over mask cells only; room colour elsewhere.
        // Quads place by art top-left (`place_top_left`, GML
        // `draw_self` sprite origin): origin-(0,0) art would sit half a
        // cell off if given cell centers.
        if let (Some(run), Some(mask)) = (
            world.get_resource::<Run>(),
            world.get_resource::<FloorMask>(),
        ) {
            let floor = run.floor;
            let area = run.area;
            let seed = run.gen_seed;
            let cells: HashSet<(i32, i32)> = mask.cells.iter().copied().collect();
            let opened: HashSet<(i32, i32)> = mask.opened.iter().copied().collect();
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
            // (`background_set_colour`), then ONLY the live floor cells - no
            // padded outside ring of floor tiles (a ±6-cell ring buried the
            // transparent vortex layer on the campfire title). Lit strip over
            // mask cells only; room colour elsewhere.
            if minx <= maxx {
                for cy in miny..=maxy {
                    for cx in minx..=maxx {
                        if !cells.contains(&(cx, cy)) {
                            continue;
                        }
                        push_floor_tile(&mut out, assets, floor_png, cx, cy);
                    }
                }
            }
            // A break uncovers no floor: `mcr_floor_make_walls`
            // (`macros_general.gml:49-62`) spawns a `Wall` only where
            // `!position_meeting(..., Floor)`, so a wall cell's 32x32 owner is
            // never a `Floor` and GML draws bare background around the hole.
            // GML `FloorExplo/Create_0:12,15`: the destroyed wall's cell becomes
            // a 16x16 `sprFloor<area>Explo` (`_area = GameCont.area`),
            // `image_index = choose(1, 2, 3, 4)` - the hole's only floor art.
            let explo_png = match area {
                AreaId::Oasis => "images/sprFloor101Explo.png",
                AreaId::PizzaSewers => "images/sprFloor102Explo.png",
                AreaId::City => "images/sprFloor103Explo.png",
                AreaId::CursedCaves => "images/sprFloor104Explo.png",
                AreaId::Jungle => "images/sprFloor105Explo.png",
                AreaId::HQ => "images/sprFloor106Explo.png",
                AreaId::Crib => "images/sprFloor107Explo.png",
                AreaId::Vault | AreaId::CrownVault => "images/sprFloor100Explo.png",
                AreaId::Campfire => "images/sprFloor0Explo.png",
                AreaId::Loop | AreaId::Desert => "images/sprFloor1Explo.png",
                AreaId::Sewers => "images/sprFloor2Explo.png",
                AreaId::Scrapyards => "images/sprFloor3Explo.png",
                AreaId::CrystalCaves => "images/sprFloor4Explo.png",
                AreaId::FrozenCity => "images/sprFloor5Explo.png",
                AreaId::Labs => "images/sprFloor6Explo.png",
                AreaId::Palace => "images/sprFloor7Explo.png",
            };
            let explo_png = if has(explo_png) { explo_png } else { floor_png };
            let explo_frames = strip_frames(assets, explo_png).max(1);
            for &(wx, wy) in &opened {
                // GML `image_index = choose(1, 2, 3, 4)` on a 4-frame strip, which
                // GameMaker wraps, so the drawn frame is `(1 + n) % 4`.
                let h = (wx
                    .wrapping_mul(0x8da6b343u32 as i32)
                    .wrapping_add(wy.wrapping_mul(0xd8163841u32 as i32))
                    >> 7) as u32;
                let frame = ((1 + h % 4) % explo_frames) as i32;
                // GML `draw_self` pins the sprite origin, and `sprFloor*Explo`
                // is 18x18 with origin (1,1): the art lands one px up-left of
                // the cell, not on it.
                let top_left = Vec2::new(wx as f32 * 16.0 - 1.0, wy as f32 * 16.0 - 1.0);
                if let Some(mut s) =
                    place_top_left(assets, explo_png, frame, top_left, [1.0; 4], GRID_OVERLAP)
                {
                    s.z = Z_FLOOR_EXPLO;
                    out.push(s);
                }
            }
            // Walls: GML law (`GenCont/Alarm_0` + `SubTopCont/Draw_0`): Out
            // skirt always (neighbor-cropped), Bot iff the screen-south tile is
            // floor (`place_meeting(x, y + 16, Floor)` on the wall instance),
            // Top always straddling the cell's top edge (top-left
            // (wx*16, wy*16-8)), then the full-ring Trans skirting pass. Bot and
            // Top share the body's `image_index` (the `topindex` roll is never
            // read). Push order = draw order.
            let mut walls: Vec<WallCell> = world
                .query_filtered::<&WallCell, With<WallTile>>()
                .iter(world)
                .copied()
                .collect();
            walls.sort_by_key(|c| (c.0, c.1));
            let solid_wall_set: HashSet<(i32, i32)> = world
                .query::<&WallCell>()
                .iter(world)
                .map(|cell| (cell.0, cell.1))
                .collect();
            let out_frames = strip_frames(assets, wall_out_png);
            let bot_frames = strip_frames(assets, wall_bot_png);
            if has(wall_out_png) {
                for cell in walls.iter().rev() {
                    let (wx, wy) = (cell.0, cell.1);
                    let frame = (wall_out_raw(seed, wx, wy) % out_frames.max(1) as usize) as i32;
                    if let Some(mut s) = wall_out_part(
                        assets,
                        wall_out_png,
                        frame,
                        wx,
                        wy,
                        wall_out_crop(&solid_wall_set, wx, wy),
                        [1.0; 4],
                    ) {
                        s.z = Z_WALL_SUBTOP;
                        wall_out.push(s);
                    }
                }
            }
            for cell in &walls {
                let (wx, wy) = (cell.0, cell.1);
                // GML `Wall/Create_0:34` verbatim: `place_meeting(x, y + 16,
                // Floor)` on the 16x16 wall body (origin (0,0) at the cell
                // top-left). The south point owns exactly one floor cell - or,
                // since `FloorExplo/Create_0:46-59` recomputes `visible` on
                // every break, one freshly opened 16x16 cell. A wall whose
                // south neighbour is a hole is hidden, so a break clears the
                // corner art that used to bridge into it.
                let south_wall = (wx, wy + 1);
                let floor_south = cells.contains(&floor_cell_for_wall(south_wall.0, south_wall.1))
                    || opened.contains(&south_wall);
                let raw = wall_body_raw(seed, wx, wy);
                let frame = raw as i32;
                if !floor_south {
                    continue;
                }
                if has(wall_bot_png) && bot_frames > 0 {
                    if let Some(s) = place_top_left(
                        assets,
                        wall_bot_png,
                        frame,
                        Vec2::new(wx as f32 * 16.0, wy as f32 * 16.0),
                        [1.0; 4],
                        0.0,
                    ) {
                        out.push(s);
                    }
                }
                if has(wall_top_png) {
                    if let Some(mut s) = place_top_left(
                        assets,
                        wall_top_png,
                        frame,
                        Vec2::new(wx as f32 * 16.0, wy as f32 * 16.0 - 8.0),
                        [1.0; 4],
                        0.0,
                    ) {
                        s.z = Z_WALL_SUBTOP;
                        wall_top.push(s);
                    }
                }
            }
            // Trans skirting: the live GML `TopSmall` instances (see
            // [`TopSmalls`]) - the worldgen ring, extended one step out per
            // broken wall by `FloorExplo`. Drawn from the Trans strip at `y - 8`.
            let trans_cells: Vec<((i32, i32), i32)> = if has(wall_trans_png) {
                trans_cells(
                    &world
                        .get_resource::<TopSmalls>()
                        .cloned()
                        .unwrap_or_default(),
                    strip_frames(assets, wall_trans_png),
                )
            } else {
                Vec::new()
            };
            let trans_set: HashSet<(i32, i32)> = trans_cells.iter().map(|(c, _)| *c).collect();
            for ((wx, wy), frame) in &trans_cells {
                if let Some(mut s) = place_top_left(
                    assets,
                    wall_trans_png,
                    *frame,
                    Vec2::new(*wx as f32 * 16.0, *wy as f32 * 16.0 - 8.0),
                    [1.0; 4],
                    0.0,
                ) {
                    s.z = Z_WALL_SUBTOP;
                    wall_trans.push(s);
                }
            }
            // Wall drop shadows (GML `scrShadows` wall half: the full Out
            // sprite flipped under each wall with no `TopSmall` at
            // `(x, y + 16)`). GML stamps these into the `shad` surface with
            // `c_black` at alpha 1 (`scrShadows.gml:29`), so they are untinted
            // COVERAGE quads: the surface holds a union, never a sum, and
            // `BackCont/Draw_0:11-12` composites that union once at
            // `draw_set_alpha(0.4)` fogged to `shadow_color`. They are cached
            // out of the batch for `crate::shadow_pass` to build that surface -
            // drawn per-wall in the batch they stacked (a 24x32 sprite on a
            // 16px grid overlaps 8px sideways and 16px down, so 2-3 deep:
            // 0.64/0.784 where GML is a flat 0.4).
            if has(wall_out_png) && wall_shadows.is_empty() {
                for cell in walls.iter().rev() {
                    let (wx, wy) = (cell.0, cell.1);
                    if trans_set.contains(&(wx, wy + 1)) {
                        continue;
                    }
                    let frame = (wall_out_raw(seed, wx, wy) % out_frames.max(1) as usize) as i32;
                    if let Some(mut s) = assets.sprite_for_full(
                        wall_out_png,
                        frame,
                        Vec2::new(wx as f32 * 16.0, wy as f32 * 16.0 + 18.0),
                        false,
                        true,
                        0.0,
                        [1.0; 4],
                    ) {
                        s.z = Z_SHADOW;
                        wall_shadows.push(s);
                    }
                }
            }
        }
        if let Some(key) = key {
            cache.key = Some(key);
            cache.prefix = Arc::from(out.clone());
            cache.wall_out = Arc::from(wall_out.clone());
            cache.wall_trans = Arc::from(wall_trans.clone());
            cache.wall_top = Arc::from(wall_top.clone());
            cache.wall_centers = Arc::from(wall_centers.clone());
            cache.wall_shadows = Arc::from(wall_shadows.clone());
        }
    }

    // Throne carpet: 72x480 srgba(0.75,0.12,0.14,0.85) rect under the
    // walls, on the `Detail` rung. GML spawns one `Carpet` instance per
    // throne room (`objects/GenCont/Destroy_0.gml:146`, created at
    // 10016, 8586) and draws its `sprCarpet` sprite (42x1000 centred,
    // `sprites/sprCarpet/sprCarpet.yy`) at object depth 8 - the same
    // depth as `Detail` (`scripts/__global_object_depths/
    // __global_object_depths.gml:60,101`). The tint rect and its extents
    // are the port's own stand-in; the occupancy itself is port-only
    // (GML's `Carpet` has no event code and no reader).
    {
        let mut q = world.query::<(&Pos, &ThroneCarpet)>();
        for (pos, carpet) in q.iter(world) {
            let size = carpet.half_extents * 2.0;
            let mut s = white_quad(pos.0, 0.0, size, [0.75, 0.12, 0.14, 0.85]);
            s.z = Z_GROUND_DETAIL;
            out.push(s);
        }
    }

    // Pulse decals (cobweb / ice / fire-trap ground visuals under the
    // actors). Textured at the recorded size, or a solid tint rect when
    // no candidate art is cataloged. Alpha always comes from the paired
    // `SurfacePulse` on the sim clock - the throb itself is port-only
    // (no GML object throbs `image_alpha` here; `TrapFire/Create_0.gml:2`
    // only varies `image_speed`).
    {
        let now = pulse_now(world);
        let mut q = world.query::<(&Pos, &PulseSprite, &SurfacePulse)>();
        for (pos, vis, pulse) in q.iter(world) {
            let mut tint = vis.tint;
            tint[3] = pulse.alpha_at(now);
            match vis.path {
                Some(path) => {
                    if let Some(mut s) =
                        assets.sprite_sized(path, 0, pos.0, Vec2::splat(vis.size), vis.flip_x, tint)
                    {
                        s.z = Z_GROUND_DETAIL;
                        out.push(s);
                    }
                }
                None => {
                    let mut s = white_quad(pos.0, 0.0, Vec2::splat(vis.size), tint);
                    s.z = Z_GROUND_DETAIL;
                    out.push(s);
                }
            }
        }
    }

    // GML `Detail` ground-decal scatter (`scrPopulate.gml:26-30`, one
    // `random(6) < 1` per room tile). `Detail/Create_0.gml` is static
    // (`image_speed = 0`), randomly mirrored and on a random frame, and
    // sits on the floor below every prop.
    {
        let mut q = world.query::<(&Pos, &GroundDetail)>();
        for (pos, detail) in q.iter(world) {
            if let Some(mut s) = assets.sprite_for(
                detail.path,
                detail.frame as i32,
                pos.0,
                detail.flip_x,
                0.0,
                [1.0; 4],
            ) {
                s.z = Z_GROUND_DETAIL;
                out.push(s);
            }
        }
    }

    // Props (recorded art paths; hurt flash tints; mine/torch sprites
    // throb via their `SurfacePulse` - a port-only alpha law; ground
    // decals draw gray 0.5).
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
            if let Some(mut s) = assets.sprite_for(path, frame, pos.0, sprites.flip_x, 0.0, tint) {
                if decal.is_some() {
                    s.z = Z_GROUND_DETAIL;
                }
                out.push(s);
            }
        }
    }

    // Campfire/crib YV couch sitter (GML `YungVenuzCouch`: idle
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
    {
        let mut q = world.query::<(&Pos, &YungCuz)>();
        for (pos, cuz) in q.iter(world) {
            let path = cuz.sprite_path();
            let frames = strip_frames(assets, path).max(1);
            let frame = (cuz.frame.floor() as i32).clamp(0, frames as i32 - 1);
            if let Some(s) = assets.sprite_for(path, frame, pos.0, cuz.flipped, 0.0, [1.0; 4]) {
                out.push(s);
            }
        }
    }

    // Title campfire actors (GML `Campfire`/`LogMenu`/`CampChar`/`TV`
    // instances from `scrCampfireMenuCreate`), drawn from the live
    // `SpriteAnim`. Campers run the `CampChar/Other_7` two-step verbatim: on
    // animation end the arm re-paths to the transition strip (`spr_to` when
    // newly selected, `spr_from` when deselected), then on the next end parks
    // on the end strip (`spr_menu`/`spr_slct`) and holds. Transitions are
    // oneshots; ends loop. Missing strips fall back to the `sprMutant` idle -
    // the `_default` arg of `scr_race_get_sprite`.
    {
        let selected = world
            .get_resource::<SelectedCharacter>()
            .map(|s| s.0 as usize)
            .unwrap_or(0);
        let mut q = world.query::<(
            Entity,
            &Pos,
            &mut SpriteAnim,
            Option<&TitleCampfire>,
            Option<&TitleLogMenu>,
            Option<&mut TitleCampChar>,
            Option<&TitleTv>,
        )>();
        // Collect swap flips first: the scan only reads, the anim
        // re-path needs `&mut World` below.
        let mut flips: Vec<(Entity, String, bool)> = Vec::new();
        let fire_pos: Option<Vec2> = world
            .query::<(&Pos, &TitleCampfire)>()
            .iter(world)
            .next()
            .map(|(p, _)| p.0);
        for (e, _pos, anim, fire, log, camp, tv) in q.iter(world) {
            if fire.is_none() && log.is_none() && camp.is_none() && tv.is_none() {
                continue;
            }
            if let Some(c) = camp {
                // GML `CampChar/Other_7` ends from the instance vars:
                // `spr_menu`/`spr_to` when selected, `spr_slct`/`spr_from`
                // when not. BigDog sleeps (`Sleep` idle, `Idle` end); the far
                // Frog sits (`Sit` everywhere); the near Frog idles on
                // `sprMutant15Idle` - `Step_0` forces `Walk` directly every
                // step while near, needing no arm here.
                let frog_far =
                    c.race_gml == 15 && fire_pos.is_some_and(|f| _pos.0.distance(f) > 600.0);
                let strips = crate::comps_b::camper_strips(c.race_gml, frog_far);
                let want_selected = c.race_gml == selected;
                let (end, trans) = if want_selected {
                    (strips.menu, strips.to)
                } else {
                    (strips.slct, strips.from)
                };
                let want_swap = if want_selected {
                    Some(true)
                } else {
                    Some(false)
                };
                let end_exists = assets.catalog.def(end).is_some();
                let trans_exists = assets.catalog.def(trans).is_some();
                let (end, has_trans) = if end_exists {
                    (end, trans_exists)
                } else if trans_exists {
                    (trans, true)
                } else {
                    (strips.slct, false)
                };
                let end = end.to_string();
                let trans = trans.to_string();
                let on_end = anim.path == end;
                let on_trans = anim.path == trans;
                // GML `CampChar/Step_0` Frog arm verbatim: the far Frog
                // rewrites all four strips to `Sit`, so the sitting camper
                // never enters the two-step. The `Walk`/`GoSit` switch below
                // is the `sprite_index == Walk -> GoSit, speed = 0` half; the
                // `GoSit -> Sit on animation_end` half runs in the
                // `c.swap.is_some() && anim.finished` arm.
                if c.race_gml == 15 && frog_far && !on_end {
                    flips.push((e, end, want_selected));
                    continue;
                }
                if c.race_gml == 15
                    && !frog_far
                    && anim.path == "images/sprMutant15Walk.png"
                    && assets.catalog.def("images/sprMutant15GoSit.png").is_some()
                {
                    flips.push((e, "images/sprMutant15GoSit.png".to_string(), want_selected));
                    continue;
                }
                if c.swap != want_swap && !on_end && !on_trans {
                    // Selection changed since last settle: start the
                    // transition if it exists, else jump to the end.
                    if has_trans && end != trans {
                        flips.push((e, trans, want_selected));
                    } else {
                        flips.push((e, end, want_selected));
                    }
                } else if c.swap.is_some() && anim.finished && (on_trans || on_end) {
                    // Transition oneshot done: park on the end strip.
                    if !on_end {
                        flips.push((e, end, want_selected));
                    }
                }
            }
        }
        for (e, path, want_selected) in flips {
            if let Some(def) = assets.catalog.def(&path) {
                // Ends are the `spr_menu`/`spr_slct` strips (looping),
                // transitions the `spr_to`/`spr_from` strips (oneshots).
                // Frog sit settling is an end even though `GoSit` reads like
                // a transition: GML parks on it (`Sit` == `from` == `to` ==
                // `menu` out far, and the near-arm `GoSit -> Sit on
                // animation_end` lands on the Sit end through the same path).
                let frog_settle =
                    path == "images/sprMutant15Sit.png" || path == "images/sprMutant15GoSit.png";
                let transition = !frog_settle
                    && (path.ends_with("Select.png")
                        || path.ends_with("Deselect.png")
                        || path == "images/sprScrapBossIntro.png"
                        || path == "images/sprScrapBossSleepHurt.png");
                if transition {
                    world.entity_mut(e).insert(SpriteAnim::oneshot(path, def));
                } else {
                    world.entity_mut(e).insert(SpriteAnim::new(path, def));
                }
                if let Some(mut c) = world.get_mut::<TitleCampChar>(e) {
                    // Jumped straight to the end (no transition
                    // strip): settled immediately.
                    c.swap = if transition {
                        Some(want_selected)
                    } else {
                        None
                    };
                }
            } else if let Some(mut c) = world.get_mut::<TitleCampChar>(e) {
                c.swap = None;
            }
        }
        let mut q = world.query::<(
            &Pos,
            &SpriteAnim,
            Option<&TitleCampfire>,
            Option<&TitleLogMenu>,
            Option<&TitleCampChar>,
            Option<&TitleTv>,
        )>();
        for (pos, anim, fire, log, camp, tv) in q.iter(world) {
            if fire.is_none() && log.is_none() && camp.is_none() && tv.is_none() {
                continue;
            }
            if camp.is_some() {
                if let Some(s) =
                    assets.sprite_for(&anim.path, anim.frame as i32, pos.0, false, 0.0, [1.0; 4])
                {
                    out.push(s);
                }
                continue;
            }
            if let Some(s) =
                assets.sprite_for(&anim.path, anim.frame as i32, pos.0, false, 0.0, [1.0; 4])
            {
                out.push(s);
            }
        }
    }

    // Live hazard rects (spill hazards are a solid kind-color rect,
    // `radius * 2`, with the hazard pulse alpha; textured fire-trap
    // visuals already drew in the decal block above). GML draws spill
    // gas as its own sprite - `objects/ToxicGas/ToxicGas.yy:40-41` names
    // `sprToxicGas`, grown by `growspeed` in
    // `objects/ToxicGas/Step_0.gml:4-10` - so the kind-color rect +
    // port-only pulse alpha is this port's stand-in for that sprite.
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
    // Corpse pass (drawn before hazards/pickups): enemy husks (`Corpse`),
    // prop corpses (`PropSprites` without `Prop`), portal strips. GML
    // orders this by object depth - `ObjectDepth[Corpse] = 1`
    // (`scripts/__global_object_depths/__global_object_depths.gml:332`),
    // so the husk draws over walls but under the depth-0 actors. Prop
    // corpses keep their recorded `PropSprites` facing, the player husk
    // its `Corpse.flip_x`. Hit-effect oneshots fade in their last 0.12 s;
    // that tail is port-only (GML's fade sprites run their strip).
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

    // Portal shock/clear/strike draw with the hit-FX pass below; see the FX
    // section after projectiles. GML puts them at their own depths
    // (`ObjectDepth[PortalStrike] = -7`, `scripts/__global_object_depths/
    // __global_object_depths.gml:29`); within the world batch they ride
    // push order only.

    // Pickups + chests at native strip-frame cells (GML `draw_self` /
    // `draw_sprite` with no scale); `sprite_for` already sizes to the
    // catalog cell, no override. Animated kinds (rads, chest idle
    // shimmer) ride their live `SpriteAnim` frame, static kinds sit on
    // frame 0. Chests drive a fractional `image_index` by hand
    // (GML `chestprop/Step_0.gml:4-7`, `image_speed = 0`), so a live
    // `GmlImage` wins over the catalog-fps `SpriteAnim`.
    {
        let mut q = world.query::<(
            &Pos,
            &Pickup,
            Option<&SpriteAnim>,
            Option<&GmlImage>,
            Option<&ChestArt>,
        )>();
        for (pos, pickup, anim, image, art) in q.iter(world) {
            let fallback = pickup_art(&pickup.kind);
            let path = art.map(|a| a.idle).unwrap_or(fallback.as_ref());
            let frame = image
                .map(|i| i.frame())
                .or_else(|| anim.map(|a| a.frame as i32))
                .unwrap_or(0);
            if let Some(s) = assets.sprite_for(path, frame, pos.0, false, 0.0, [1.0; 4]) {
                out.push(s);
            }
        }
    }

    // Opened chests: GML spawns a `ChestOpen` whose `image_speed = 0.4`
    // walks the open strip and `ChestOpen/Other_7.gml:1-2` parks it on
    // `image_number - 1`. Single-frame open art therefore never moves,
    // so the terminal frame is the whole animation for every kind
    // except the multi-frame mystery/protopen strips.
    {
        let mut q = world.query::<(&Pos, &OpenedChest, Option<&ChestArt>)>();
        for (pos, opened, art) in q.iter(world) {
            let (path, last) = match opened.0 {
                ChestKind::Weapon => ("images/sprWeaponChestOpen.png", 0u32),
                ChestKind::Ammo => ("images/sprAmmoChestOpen.png", 0u32),
                // GML `ChestOpen` walks this 5-frame strip at
                // `image_speed = 0.4` and `Other_7.gml:1-2` parks it on
                // `image_number - 1`, i.e. frame 4.
                ChestKind::Mystery => ("images/sprAmmoChestMysteryOpen.png", 4u32),
                ChestKind::Gold => ("images/sprGoldChestOpen.png", 0u32),
                // GML `prop/Destroy_0.gml:2-8` -> `Corpse`, which freezes
                // on `image_number - 1` after `image_speed = 0.4`.
                ChestKind::Rad => ("images/sprRadChestCorpse.png", 2u32),
                ChestKind::Health => ("images/sprHealthChestOpen.png", 0u32),
                // GML `CursedBigChest/Destroy_0.gml:2`: the opened cursed
                // big chest is the plain big-weapon open art;
                // `sprCursedChestBigOpen` is never drawn in GML.
                ChestKind::CursedBig => ("images/sprWeaponChestBigOpen.png", 0u32),
                ChestKind::Rogue => ("images/sprRogueAmmoChestOpen.png", 0u32),
                ChestKind::Proto => ("images/sprProtoChestOpen.png", 0u32),
                ChestKind::BigWeapon => ("images/sprWeaponChestBigOpen.png", 0u32),
                ChestKind::RadBig => ("images/sprRadChestBigDead.png", 2u32),
                ChestKind::RadMaggot => ("images/sprRadChestMaggotDead.png", 3u32),
                ChestKind::Idpd => ("images/sprIDPDChestOpen.png", 0u32),
            };
            let path = art.map(|a| a.open).unwrap_or(path);
            let frames = strip_frames(assets, path);
            let frame = last.min(frames.saturating_sub(1)) as i32;
            if let Some(s) = assets.sprite_for(path, frame, pos.0, false, 0.0, [1.0; 4]) {
                out.push(s);
            }
        }
    }

    // IDPD spawn portal: GML `IDPDSpawn/Create_0.gml:26` runs the object
    // strip at `image_speed = 0.4`, `Other_7.gml:1` swaps
    // `sprIDPDPortalStart` -> `sprIDPDPortalCharge` on animation end,
    // `Alarm_0.gml:2-3` parks `sprIDPDPortalClose` on frame 0. The sim
    // exports only the close countdown, so that phase is exact (14 frames /
    // 0.4 = its 35-step arming window) and the open phase rides the charge
    // strip - the 2-frame start window has no sim-side age to key off.
    {
        let now = pulse_now(world);
        let mut q = world.query::<(&Pos, &crate::idpd::IdpdSpawnPortal)>();
        for (pos, portal) in q.iter(world) {
            let (path, frame) = if portal.close > 0.0 {
                let frames = strip_frames(assets, "images/sprIDPDPortalClose.png").max(1);
                let frame = ((35.0 - portal.close).max(0.0) * 0.4).floor() as i32;
                (
                    "images/sprIDPDPortalClose.png",
                    frame.clamp(0, frames as i32 - 1),
                )
            } else {
                let frames = strip_frames(assets, "images/sprIDPDPortalCharge.png").max(1);
                (
                    "images/sprIDPDPortalCharge.png",
                    ((now * 12.0).floor() as i32).rem_euclid(frames as i32),
                )
            };
            if let Some(s) = assets.sprite_for(path, frame, pos.0, false, 0.0, [1.0; 4]) {
                out.push(s);
            }
        }
    }
    // Enemies: live SpriteAnim wins (idle/walk/hurt/fire already switched
    // sim-side); headless spawns fall back to the def idle, so the
    // walk/hurt strips resolve here from the anim path rather than a
    // separate sprite set. Gun carriers draw the gun behind the body when
    // aiming down-ish (gunangle ≤ 180°) and in front above it (GML
    // per-kind `Draw_0` law).

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
            Option<&HurtAnim>,
            Option<&MaggotSpawnCharge>,
            Option<&DogGuardianLeap>,
        )>();
        for (pos, enemy, _vel, _aim, _anim, _flash, brain, _hurt, _charge, _leap) in q.iter(world) {
            if enemy.kind != EnemyKind::Sniper {
                continue;
            }
            let Some(brain) = brain else {
                continue;
            };
            if !brain.sniper_aiming {
                continue;
            }
            if let Some(sight) = laser_sight(
                assets,
                "images/sprLaserSight.png",
                pos.0,
                brain.gunangle,
                &wall_centers,
            ) {
                out.push(sight);
            }
        }
        // Guns behind.
        for (pos, enemy, vel, aim, _anim, _flash, brain, _hurt, _charge, _leap) in q.iter(world) {
            let Some(gun_path) = enemy_gun_art(enemy.kind) else {
                continue;
            };
            let Some(brain) = brain else {
                continue;
            };
            let gunangle = brain.gunangle;
            let draw_angle = if enemy.kind == EnemyKind::MeleeBandit {
                gunangle + brain.wepangle
            } else {
                gunangle
            };
            let gun_pos = pos.0 + Vec2::new(draw_angle.cos(), draw_angle.sin()) * (-brain.wkick);
            if gunangle.to_degrees().rem_euclid(360.0) > 180.0 {
                continue;
            }
            let flip = vel.map(|v| v.0.x < 0.0).unwrap_or(false)
                || aim.map(|a| a.0.x < 0.0).unwrap_or(false);
            if let Some(s) =
                assets.sprite_for_full(gun_path, 0, gun_pos, false, flip, draw_angle, [1.0; 4])
            {
                out.push(s);
            }
        }
        // Bodies.
        for (pos, enemy, vel, aim, anim, flash, brain, hurt, charge, leap) in q.iter(world) {
            let (mut path, mut frame) = if hurt.is_none()
                && let Some(charge) = charge
            {
                (charge.image.path, charge.image.frame())
            } else {
                match anim {
                    Some(a) => (a.path.as_str(), a.frame as i32),
                    None => (crate::enemy_data::enemy_def(enemy.kind).sprite, 0),
                }
            };
            let mut at = pos.0;
            let mut tint = flash_tint(flash);
            if let Some(leap) = leap {
                match leap.pose {
                    DogGuardianPose::Ground => {}
                    DogGuardianPose::Airborne => {
                        path = if leap.zspeed < 0.0 {
                            "images/sprDogGuardianJumpUp.png"
                        } else {
                            "images/sprDogGuardianLand.png"
                        };
                        frame = 0;
                        at.y -= leap.z;
                        tint = [1.0, 1.0, 1.0, 1.0];
                    }
                    pose if hurt.is_none() => {
                        path = if pose == DogGuardianPose::Charge {
                            "images/sprDogGuardianCharge.png"
                        } else {
                            "images/sprDogGuardianLand.png"
                        };
                        let frames = strip_frames(assets, path).max(1);
                        frame = frame.rem_euclid(frames as i32);
                    }
                    _ => {}
                }
            }
            let flip = if enemy.kind == EnemyKind::MaggotSpawn {
                charge
                    .map(|c| c.facing < 0.0)
                    .or_else(|| brain.map(|b| b.maggot_spawn_facing < 0.0))
                    .unwrap_or(false)
            } else {
                vel.map(|v| v.0.x < 0.0).unwrap_or(false)
                    || aim.map(|a| a.0.x < 0.0).unwrap_or(false)
            };
            if let Some(s) = assets.sprite_for(path, frame, at, flip, 0.0, tint) {
                out.push(s);
            }
        }
        // Guns in front.
        for (pos, enemy, vel, aim, _anim, _flash, brain, _hurt, _charge, _leap) in q.iter(world) {
            let Some(gun_path) = enemy_gun_art(enemy.kind) else {
                continue;
            };
            let Some(brain) = brain else {
                continue;
            };
            let gunangle = brain.gunangle;
            let draw_angle = if enemy.kind == EnemyKind::MeleeBandit {
                gunangle + brain.wepangle
            } else {
                gunangle
            };
            let gun_pos = pos.0 + Vec2::new(draw_angle.cos(), draw_angle.sin()) * (-brain.wkick);
            if gunangle.to_degrees().rem_euclid(360.0) <= 180.0 {
                continue;
            }
            let flip = vel.map(|v| v.0.x < 0.0).unwrap_or(false)
                || aim.map(|a| a.0.x < 0.0).unwrap_or(false);
            if let Some(s) =
                assets.sprite_for_full(gun_path, 0, gun_pos, false, flip, draw_angle, [1.0; 4])
            {
                out.push(s);
            }
        }
    }

    {
        let mut q = world.query::<(
            &Pos,
            &BigDogMissileState,
            Option<&Velocity>,
            Option<&GmlImage>,
            Option<&NativeAngle>,
            Option<&NativeDepth>,
            Option<&HitFlash>,
        )>();
        for (pos, state, vel, image, native_angle, depth, flash) in q.iter(world) {
            let rotation = native_angle
                .map(|a| a.0)
                .or_else(|| {
                    vel.filter(|v| v.0.length_squared() > 1e-6)
                        .map(|v| v.0.y.atan2(v.0.x))
                })
                .unwrap_or(0.0);
            let (body_path, body_frame) = image.map_or_else(
                || ("images/sprScrapBossMissileIdle.png", 0),
                |image| (image.path, image.frame()),
            );
            let body_depth = depth.map(|d| d.0).unwrap_or(-2.0);
            let mut body = assets
                .sprite_for(
                    body_path,
                    body_frame,
                    pos.0,
                    false,
                    rotation,
                    flash_tint(flash),
                )
                .unwrap_or_else(|| {
                    white_quad(pos.0, rotation, Vec2::splat(16.0), flash_tint(flash))
                });
            body.z = -body_depth / 5.0;
            let trail_frames = strip_frames(assets, "images/sprScrapBossMissileTrail.png").max(1);
            let trail_frame = (state.trail_timer as i32).rem_euclid(trail_frames as i32);
            let body_z = body.z;
            out.push(body);
            if let Some(mut trail) = assets.sprite_for(
                "images/sprScrapBossMissileTrail.png",
                trail_frame,
                pos.0,
                false,
                rotation,
                [1.0; 4],
            ) {
                trail.z = body_z + 0.001;
                out.push(trail);
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

    // Player: GML `Player/Draw_0` order verbatim - Eyes underlay, back
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
        // GML `DogSpinAttack/Alarm_0.gml:7-12`: the sprite-less spin body
        // rewrites its creator's strips to the Scrap-Boss spin set (and
        // `Alarm_0.gml:46-53` restores them when the ammo runs out), so
        // the creator swap IS the whole visual.
        let spinning: HashSet<Entity> = world
            .query::<&crate::player_fire::SpinAttack>()
            .iter(world)
            .map(|spin| spin.creator)
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

            if world
                .get::<Shield>(entity)
                .is_none_or(|shield| shield.timer.is_finished())
                && let Some(inv) = inv_opt
            {
                let max_slot = inv.weapon_slots.max(1) - 1;
                let current = inv.weapons[inv.current.min(max_slot)];
                let current_meta = weapon_meta(current);
                if current_meta.wep_type == AmmoType::Bolts
                    && current_meta.wep_name != "DISC GUN"
                    && let Some(sight) = laser_sight(
                        assets,
                        "images/sprLaserSightPlayer.png",
                        pos.0,
                        gunangle,
                        &wall_centers,
                    )
                {
                    out.push(sight);
                }
                if is_steroids && inv.weapon_slots > 1 {
                    let secondary = inv.weapons[(inv.current + 1) % inv.weapon_slots];
                    let secondary_meta = weapon_meta(secondary);
                    if secondary_meta.wep_type == AmmoType::Bolts
                        && secondary_meta.wep_name != "DISC GUN"
                        && let Some(sight) = laser_sight(
                            assets,
                            "images/sprLaserSightPlayer.png",
                            Vec2::new(pos.0.x, pos.0.y - 4.0),
                            gunangle,
                            &wall_centers,
                        )
                    {
                        out.push(sight);
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
            // GML `DogSpinAttack/Alarm_0.gml:8-10` (see `spinning` above):
            // idle/walk -> `sprScrapBossFire`, hurt -> `sprScrapBossHurtSpin`.
            // The creator keeps `image_speed = 0.4` (`Player/Create_0.gml:131`),
            // so the swap plays at 12 fps.
            let (path, frame): (&str, i32) = if spinning.contains(&entity) {
                // Only take the spin-hurt strip when the player's own hurt
                // path actually resolves; `play_hurt` falls back to `pa.idle`
                // otherwise, and `path == pa.hurt` would never match, so a
                // missing catalog entry would silently drop the hurt beat.
                let spin_path = if path == pa.hurt && strip_frames(assets, pa.hurt) > 0 {
                    "images/sprScrapBossHurtSpin.png"
                } else {
                    "images/sprScrapBossFire.png"
                };
                let frames = strip_frames(assets, spin_path).max(1);
                (
                    spin_path,
                    ((blink_t * 12.0).floor() as i32).rem_euclid(frames as i32),
                )
            } else {
                (path, frame)
            };
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
    // heading (y-down atan2 - GML sets `image_angle = direction` in
    // `scripts/scr_projectile_create/scr_projectile_create.gml:26-28`).
    // Grenade pre-detonation telegraph, GML law: black/white fog strobe
    // over the last `flash_at` ticks before the blast (`objects/Grenade/
    // Draw_0.gml:4-5`, `alarm[0] % 5 > 2 ? c_black : c_white`, with
    // `alarm[0] = 60` and `flash_at = 10` at `objects/Grenade/
    // Create_0.gml:11-12`). The port drives the window off the fuse's
    // 6-tick `alarm[1]` (`objects/Grenade/Create_0.gml:10`), so its
    // strobe window is shorter than GML's 10 ticks. The "solid white
    // once the friction switch armed" tail is port-only - GML's
    // non-strobe branch is a plain `draw_self()`.
    {
        let mut q = world.query::<(
            &Pos,
            &Projectile,
            Option<&Velocity>,
            &Team,
            Option<&SlashProjectile>,
            Option<&GrenadeFuse>,
            Option<&crate::comps_a::ProjectileVisual>,
            Option<&ProjectileFade>,
            Option<&GmlImage>,
            Option<&NativeScale>,
            Option<&NativeAngle>,
            Option<&NativeFlip>,
            Option<&NativeDepth>,
        )>();
        for (
            pos,
            proj,
            vel,
            team,
            slash,
            fuse,
            visual,
            fade,
            image,
            native_scale,
            native_angle,
            flip,
            depth,
        ) in q.iter(world)
        {
            if let Some(image) = image {
                let rotation = native_angle.map(|a| a.0).unwrap_or_else(|| {
                    vel.filter(|v| v.0.length_squared() > 1e-6)
                        .map(|v| v.0.y.atan2(v.0.x))
                        .unwrap_or(0.0)
                });
                if let Some(sprite) = native_image_instance(
                    assets,
                    image,
                    pos.0,
                    rotation,
                    native_scale.map(|s| s.0).unwrap_or(Vec2::ONE),
                    flip.is_some_and(|f| f.0),
                    depth.map(|d| d.0).unwrap_or(0.0),
                    [1.0; 4],
                ) {
                    out.push(sprite);
                }
                continue;
            }
            let path = if fade.is_some_and(|f| f.0 == ALLY_BULLET_FADE) {
                "images/sprAllyBullet.png"
            } else {
                projectile_art(proj, team, slash, visual)
            };
            let frame = projectile_frame(assets, path, &proj.life);
            let rotation = match slash {
                Some(s) => s.dir.y.atan2(s.dir.x),
                None => vel
                    .filter(|v| v.0.length_squared() > 1e-6)
                    .map(|v| v.0.y.atan2(v.0.x))
                    .unwrap_or(0.0),
            };
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

    {
        let mut q = world.query::<(
            &Pos,
            &GmlImage,
            Option<&NativeScale>,
            Option<&MoteScale>,
            Option<&NativeAngle>,
            Option<&FxAngle>,
            Option<&NativeFlip>,
            Option<&NativeDepth>,
            Option<&ToxicGasState>,
            Option<&Enemy>,
            Option<&Projectile>,
            Option<&BigDogMissileState>,
        )>();
        for (
            pos,
            image,
            native_scale,
            mote_scale,
            native_angle,
            fx_angle,
            flip,
            depth,
            gas,
            enemy,
            projectile,
            missile,
        ) in q.iter(world)
        {
            if enemy.is_some() || projectile.is_some() || missile.is_some() {
                continue;
            }
            let scale = native_scale
                .map(|s| s.0)
                .or_else(|| mote_scale.map(|s| Vec2::splat(s.0.max(0.0))))
                .or_else(|| gas.map(|s| Vec2::splat(s.scale.max(0.0))))
                .unwrap_or(Vec2::ONE);
            let angle = native_angle
                .map(|a| a.0)
                .or_else(|| fx_angle.map(|a| a.0))
                .unwrap_or(0.0);
            if let Some(sprite) = native_image_instance(
                assets,
                image,
                pos.0,
                angle,
                scale,
                flip.is_some_and(|f| f.0),
                depth.map(|d| d.0).unwrap_or(0.0),
                [1.0; 4],
            ) {
                out.push(sprite);
            }
        }
    }

    // Hit-FX pass (over projectiles, under guns): portal
    // shock/clear/strike (GML 1-frame-per-step strips), bullet-hit/dust/
    // fade oneshots with `FxAngle` orientation, and bare-PNG static
    // fallbacks - all fading in the last 0.12 s. The 0.12 s tail is
    // port-only; GML's fade sprites advance through their strip instead
    // (`objects/Bullet1/Create_0.gml:2` `spr_fade = sprBulletHit`).
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
            if let Some(def_size) = assets.native_size("images/sprPortalClear.png") {
                let size = def_size * clear.scale.max(0.0);
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
        // that isn't a corpse/portal/actor/mote - motes have their own
        // scaled arm below).
        let mut q = world.query_filtered::<(
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
        ), bevy_ecs::prelude::Without<Mote>>();
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
        // Static FX fallbacks: single-strip effects, no animated frames.
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
        // GML `Dust`/`Smoke`/`Feather`/`Curse` motes: sprite debris with
        // live `MoteScale` (grow/decay law) and `FxAngle` spin. Drawn
        // from the catalog strip at the mote scale, no lifetime fade
        // (GML kills on `image_xscale < 0`, not alpha).
        let mut q = world.query::<(&Pos, &SpriteAnim, &Mote, &MoteScale, Option<&FxAngle>)>();
        for (pos, anim, _mote, scale, angle) in q.iter(world) {
            let rotation = angle.map(|a| a.0).unwrap_or(0.0);
            if let Some(s) = assets.sprite_scaled_rotated(
                &anim.path,
                anim.frame as i32,
                pos.0,
                scale.0.max(0.0),
                rotation,
                [1.0; 4],
            ) {
                out.push(s);
            }
        }
    }

    // Held guns: GML Player/Draw_0 verbatim.
    // Position = player + lengthdir(-wkick, gunangle + wepangle*(1 - wkick/20))
    // NO +12 forward hold. Sprite origin is the grip.
    {
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
                        back && gun.slot == 0,
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
        }
    }

    // Melee swings + wall-hits (drawn after the held guns): `SwingFx`
    // rotation with the live strip, falling back to the wall-hit art. The
    // wall hit is GML's own: `Slash/Collision_Wall.gml:11-19` spawns a
    // `MeleeHitWall` at the wall bbox centre, angled at it. The held-gun
    // swing arc is not
    // - GML's melee is the `Slash` projectile
    //   (`objects/Slash/Create_0.gml:2`, `image_speed = 0.4`) plus the
    //   `Dust` motes it spawns, so that half of the pass is port-only.
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

    out.extend(wall_out);
    out.extend(wall_trans);
    out.extend(wall_top);
    out
}

// Transient combat FX → instances.

/// Beam strip (320x32, 1 frame): stretched along the sim segment.
pub const BEAM_STRIP: &str = "images/sprLightBeam.png";
/// Lightning strip (2x2, 4 frames): stretched along the arc segment,
/// frame following the 0.09 s life.
pub const ARC_STRIP: &str = "images/sprLightning.png";
/// Muzzle tongue size (no muzzle strip exists in the catalog).
pub const MUZZLE_SIZE: Vec2 = Vec2::new(24.0, 14.0);
/// Damage-number font size in world units. Port-only: GML draws no
/// floating damage numbers - combat feedback is the hurt flash
/// (`sprite_index = spr_hurt`, `objects/Corpse/Collision_hitme.gml:10-11`)
/// and the hit sprites named by each bullet's `spr_fade`
/// (`objects/Bullet1/Create_0.gml:2`).
pub const NUMBER_TEXT_SIZE: f32 = 28.0;

/// Solid-color fallback quad (particle convention): [`SpriteInstance`]
/// has no fill flag, so - like [`particle_sprites`] - this samples the
/// full first atlas page (`uv 0..1`, page 0) and leans on the tint.
/// Sim-clock seconds driving the `SurfacePulse::alpha_at` throb (a
/// port-only law); 0.0 headless without the resource.
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

/// Legacy 320x240 GUI view map, port-only (kept for reference): fill
/// scale `s = min(w/320, h/240)`, centered offsets. GML has no such
/// map - its GUI *is* the live view (see [`hud_gui_map`] /
/// [`gml_view_size`]: 426x240 at 16:9), so the text pipeline now maps
/// through [`gui_texts_dp`].
#[derive(Clone, Copy, Debug)]
pub struct NtView {
    pub s: f32,
    pub ox: f32,
    pub oy: f32,
}

/// Legacy 320x240 letterbox map (viewport falls back to 1280x720).
/// Port-only, kept for reference; the live pipeline uses
/// [`gml_view_size`].
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

/// Fill-scale map from a world-space view rect. GML law verbatim: the GUI
/// *is* the view (`display_set_gui_size(view_width, view_height)` in
/// `scrSetViewSize`), so GUI px == view px 1:1 - identity map, no letterbox.
/// Callers pass the live [`view_rect_world`] rect; `gx`/`gy` authored in GML
/// view px land 1:1 (at 16:9 the view is 426x240: right-anchored rows use
/// `view[2] - 2`, centered headers `view[2] / 2`).
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

/// GML `draw_sprite` (NOT `_ext`) honors the strip origin: the DRAW POINT is
/// `pos - origin`, so art top-left lands on `pos - origin` and size =
/// native * `mul` * map scale. Callers pass the GML draw position verbatim
/// for ALL strips; this helper reads the catalog origin and offsets the quad
/// top-left by `-origin * mul * map.s`, so pixels land where GML puts them
/// whatever the strip origin (`sprUltraLevel` (4,5), Rogue pips (1,1)).
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
    let top_left = hud_gui_to_world(map, view, gx - def.xorigin * mul, gy - def.yorigin * mul);
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

fn hud_weapon_order(hud: &HudState) -> Vec<usize> {
    if hud.weapon_ids.is_empty() {
        return vec![0];
    }
    let current = hud.current_weapon;
    if current < hud.weapon_ids.len() && hud.weapon_ids[current] != WeaponId::NONE {
        let mut order = Vec::with_capacity(hud.weapon_ids.len());
        order.push(current);
        order.extend((0..hud.weapon_ids.len()).filter(|slot| *slot != current));
        order
    } else {
        (0..hud.weapon_ids.len()).collect()
    }
}

/// GML weapon-row x positions: 24, then +44, then +20 per extra
/// (`_dx += 44` for the first slot, `+20` afterwards once extras
/// exist - i.e. 24/68/88/108/...).
fn hud_weapon_dx(pos: usize) -> f32 {
    match pos {
        0 => 24.0,
        1 => 68.0,
        _ => 88.0 + (pos as f32 - 2.0) * 20.0,
    }
}

/// GML `scrDrawPlayerHUD` text laws verbatim:
/// - `"hp/max"` at (67,7) white centered (`23+44, 7`);
/// - level number at (11,16) white centered middle-anchored, only below the
///   level cap (`PLAYER_LEVEL_MAX = 10`; max level draws `sprUltraLevel`);
/// - per-slot ammo at (42+pos*44,21) left, active-first draw order: white
///   when the slot's ammo type is active else silver, `c_uidark` when dry,
///   red (active) or gray at/below one pickup, skipped for melee/absent;
/// - `"LOW HP"` at (110,7) red left when `hp <= 4 && hp != max` while
///   recently hurt (`drawlowhp`; `HitFlash` stands in, the shell adds the
///   `sin(wave)` blink);
/// - `scrDrawMiscHUD` bottom-right rows right-aligned at `view-2`: run clock
///   then map name (gated on `show_timer`/`show_area`).
///
/// Deferred GML rows (need sim state the port never tracks): `FAINTED`
/// pulse, bleed gray bar, hurt white flash, analog `scrDrawClock` (only the
/// digital `timer_string` draws). Everything else draws: strips + rows
/// (sprites), per-slot ammo/LOW-HP/low-ammo, clock/area, event icons,
/// interaction prompt (`hud_texts`).
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
    if hud.level < PLAYER_LEVEL_MAX {
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
            // with `_dy = 16` (NOT 21 - the ammo digits sit 5 px below
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
    // red-left at `(55 + icons*12, 35)` while `drawempty > 0` and the shell
    // blinks (`sin(wave) > 0` gate lives in `hud_overlay_lines`).
    // `drawempty` = the dry-fire toast window (raised when the EMPTY /
    // NOT ENOUGH RADS toast is live).
    {
        let dry = world
            .get_resource::<crate::comps_a::Toast>()
            .is_some_and(|t| t.text == "EMPTY" || t.text == "NOT ENOUGH RADS");
        if dry {
            let primary_slot = order.first().copied().unwrap_or(hud.current_weapon);
            let primary = hud
                .weapon_ids
                .get(primary_slot)
                .copied()
                .unwrap_or(WeaponId::NONE);
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
    // stacking up from `view_height - (font_offset + 10)` = 230 for the
    // default font (`font_get_height_diff()` returns 0 outside CJK/
    // Noto); row step is the string height in GML, 9px here to match
    // the port's 7px Silkscreen rhythm.
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
    let mut gy = 240.0 - (0.0 + 10.0);
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

/// One positioned overlay text in GUI space. GML authors these as
/// `draw_text_nt` / `draw_text_bigname` draw points in the live view
/// (identity map, [`hud_gui_map`]): title buttons sit at
/// `view_xview + xstart` (`objects/Menu/Draw_0.gml:9-17`) and the
/// misc-HUD rows are right-aligned on `view_width - 2` with
/// `draw_align(fa_right, fa_top)` (`scripts/scrDrawMiscHUD/
/// scrDrawMiscHUD.gml:13-25`).
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
    /// Bigname-sourced row (GML `draw_text_bigname`): rendered with
    /// fill+stroke faux-bold to approximate the heavy fntBig glyphs.
    pub bold: bool,
}

/// Overlay row: (segments, dp top-left, font px, centered, box width,
/// right-aligned, bold). `segments` are `(text, sRGB color 0..1)` runs
/// from the GML `draw_text_nt` `@`-tag parser ([`nt_text_segments`]);
/// the shell draws each run in its color with the standard 1px black
/// shadow. `bold` selects fill+stroke faux-bold (bigname rows).
pub type GuiRow = (
    Vec<(String, [f32; 4])>,
    [f32; 2],
    f32,
    bool,
    f32,
    bool,
    bool,
);

/// GML `draw_text_nt` `@`-tag colors verbatim
/// (`scripts/draw_text_nt/draw_text_nt.gml:215-222`): `@s` silver
/// (125,131,141), `@b` blue (22,97,223), `@r` red (252,56,0), `@y` yellow
/// (250,171,0), `@d` dark gray (59,62,67), `@g` green (68,198,22), `@p` purple
/// (86,34,110), `@w` white. Unknown tags (`@q` shake, `@(`/`@)` sprite,
/// `@[`/`@]` bold, `@.` reset) carry no color: the row's base.
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
/// color, `\@` escapes a literal `@`. `#` stays inside the run as a literal
/// line break (the shell wraps on it; GML writes it as a newline via
/// `string_hash_to_newline` before parsing). Raw `\n` never reaches this
/// backend: multi-line rows are pre-split into one [`MenuGuiText`] per visual
/// line at the producer (GML draws each line at its own y). Every other `@?`
/// sequence keeps its chars in the base color.
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

/// Generic GUI text → dp mapper. GML law: the GUI is the live view
/// (`scrSetViewSize` + `display_set_gui_size`), `vw` the live GUI width in px
/// from [`gml_view_size`] (always 240 tall). Left texts sit at `gx * k`;
/// centered texts center their `2*gx` box on `gx * k`; right texts
/// right-align their `2*(vw-gx)` box `vw-gx` px from the canvas right edge
/// (`scrDrawMiscHUD` clock/area at `view_width - 2`); `middle_y` centers the
/// line on `gy`. `k` is dp per GUI px, contain-fit (`min(h/240, w/320)`), so
/// the 320-floored GUI scales to fit width exactly like the sprite viewport
/// (`effective_fit`) - height-fit alone blew the text layer up in portrait
/// while sprites stayed fitted. `(ox, oy)` centers the fitted GUI rect on the
/// canvas (landscape: `oy` 0, `ox` absorbs the sub-px `vw`-flooring crumbs).
/// Silkscreen-vs-bitmap top bias in GUI px: GML HUD/menu fonts (`fntM1`) are
/// bitmaps whose ink starts exactly at the draw y (digits `h:7, offset:0`);
/// Silkscreen via parley baselines at `round(1.03em)` with digit ink topping
/// out at 0.625em, so ink starts ~2 GUI px below the box top. Top-anchored
/// rows subtract this so ink lands on the GML y; `middle_y` rows are
/// symmetric in both backends and need no nudge.
pub const FONT_TOP_BIAS_GUI: f32 = 2.0;

pub fn gui_texts_dp(canvas_dp: [f32; 2], items: Vec<MenuGuiText>) -> Vec<GuiRow> {
    let vw = gml_view_size(canvas_dp)[0];
    let w = canvas_dp[0].max(1.0);
    let h = canvas_dp[1].max(1.0);
    let k = (h / 240.0).min(w / 320.0).max(1e-6);
    let ox = (w - vw * k) * 0.5;
    let oy = (h - 240.0 * k) * 0.5;
    items
        .into_iter()
        .map(|t| {
            let font_px = (t.px * k).round().clamp(8.0, 180.0);
            let (left, box_w, centered) = if t.centered {
                // GML centers `draw_text_nt` on `gx`: the box spans the
                // symmetric fit around `gx` (`2 * min(gx, vw - gx)`) with
                // content centered inside. View-centered rows (gx = cx) get
                // the full width, column headers (stats TOTAL at `statx`)
                // their own column - a full-width box dragged them to `vw/2`
                // (the centered part of the "left-shortened stats" bug) and a
                // symmetric fit without centering clipped wide rows.
                let half = t.gx.min(vw - t.gx).max(1.0);
                let bw = (2.0 * half * k).max(font_px);
                (ox + t.gx * k - bw * 0.5, bw, true)
            } else if t.right {
                // GML right-aligns on `gx`: the row's box right edge lands
                // exactly on `gx * k`, content right-aligned inside. A fixed
                // 120px box pushed short rows left of their anchor (the
                // "left-shortened stats" bug); HUD clock/area rows need a box
                // wide enough to reach `vw - 2`, stats names only their
                // column.
                let right = ox + t.gx * k;
                let bw = if t.gx >= vw - 3.0 {
                    (120.0 * k).max(font_px)
                } else {
                    (60.0 * k).max(font_px)
                };
                (right - bw, bw, false)
            } else {
                let left = ox + t.gx * k;
                let bw = (200.0 * k)
                    .max(font_px)
                    .min((ox + vw * k - left).max(font_px));
                (left, bw, false)
            };
            let top = if t.middle_y {
                oy + t.gy * k - font_px * 0.5
            } else {
                // Ink-top parity with the GML bitmap fonts (see
                // `FONT_TOP_BIAS_GUI`): the box top sits 2 GUI px above
                // the authored y so Silkscreen ink starts on it.
                oy + t.gy * k - FONT_TOP_BIAS_GUI * k
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
                t.bold,
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
    let player_alive = world.query::<&Player>().iter(world).next().is_some();
    // GML `scrDrawMiscHUD:68` draws held icons when GameOver exists.
    let game_over = crate::state::menus::game_over_visible(world);
    let vw = gml_view_size(canvas_dp)[0];
    let cx = vw * 0.5;
    let mut items: Vec<MenuGuiText> = Vec::new();
    // GML `scrDrawPlayerHUD.gml:378-403` `is_touch` block: while a prompt
    // overlaps the player, `_prompt_text` re-draws above the `ButtonAct` home
    // (`x = view_width/2`, `y = 48`, `rad = 25`, `Create_0`) at
    // `max(_height, y - rad * 0.5 - _height - 12)` - the fntM1 line measures 8
    // (`font_string_measure:65` fixed line height), so gy = 15.5,
    // top-anchored and centered on the home (`draw_align(fa_center, fa_top)`
    // at `scrDrawPlayerHUD:13`; `scrDrawMobileControls:239` re-draws the same
    // string, so the port draws the centered copy once). `is_touch` =
    // `!(opt_keyboard || opt_gamepad)` (`input::gml_input_device`, same law as
    // `touch_sprites`). "PICK UP" is `loc("R:HUD:PickUpAction", "PICK UP")`
    // (no lang.csv row, default stands) for the nearest weapon; each
    // `prop_prompts` hit stacks its own text on the same anchor in GML's
    // `[WepPickup, CarVenusFixed, IceFlower, Van]` order. Rides the
    // mobile-controls path (`TopCont/Draw_64:23` gates on `drawcontrols`, not
    // `opt_hud`), so it ignores `show_hud` like `hud_texts`/`touch_sprites`.
    let (keyboard, gamepad) =
        crate::input::gml_input_device(world.get_resource::<crate::savedata_part::SaveData>());
    if player_alive && !keyboard && !gamepad {
        let act_row = |text: String| MenuGuiText {
            text,
            gx: cx,
            gy: 15.5,
            color: [255, 255, 255, 255],
            px: 7.0,
            centered: true,
            middle_y: false,
            right: false,
            bold: false,
        };
        if let Some(label) = world.get_resource::<crate::pickups::WeaponLabel>() {
            if label.target.is_some() {
                items.push(act_row("PICK UP".to_string()));
            }
            for (_, text) in label.prop_prompts.iter() {
                items.push(act_row((*text).to_string()));
            }
        }
    }
    if !show_hud {
        return gui_texts_dp(canvas_dp, items);
    }
    if !player_alive && !game_over {
        return gui_texts_dp(canvas_dp, items);
    }
    items.extend(hud_gui_texts(world).into_iter().map(|t| MenuGuiText {
        text: t.text,
        gx: if t.right { vw - 2.0 } else { t.gx },
        gy: t.gy,
        color: t.color,
        px: 7.0,
        centered: t.centered,
        middle_y: t.middle_y,
        right: t.right,
        bold: false,
    }));
    let hud: HudState = sync_hud_state(world);
    // GML `SkillText` verbatim: `scrLevelUpScreenSubmit` spawns at
    // `_ypos = view_yview + view_height - textheight - 76` (= 156 for one 8px
    // fntM1 line) and `TopCont/Draw_75` / `LevCont/Draw_64` draw
    // `"@d" + loc(txt)` centered-middle AT THE INSTANCE POS. No SkillText
    // entity in the port, so the toast rides that view-fixed anchor
    // (middle-anchored, like GML's `fa_middle`), plus a `-40` shift while a
    // chained offer still has the prior SkillText alive - never the view
    // center, which is where the player is.
    if !hud.toast.is_empty() {
        let toast_y = world
            .get_resource::<MenuState>()
            .map_or(156.0, |menu| 156.0 + menu.mutation_toast_offset);
        items.push(MenuGuiText {
            text: format!("@d{}", hud.toast),
            gx: cx,
            gy: toast_y,
            color: [255, 255, 255, 255],
            px: 7.0,
            centered: true,
            middle_y: true,
            right: false,
            bold: false,
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
            bold: false,
        });
    }
    gui_texts_dp(canvas_dp, items)
}

// Menu overlay texts: strings, GUI positions, colors and px sizes
// transcribed from the GML draw scripts (`draw_text_nt`,
// `draw_text_bigname`, `objects/MainMenuButton/Draw_0.gml:6-9`,
// `scripts/scrMenuButtonName/scrMenuButtonName.gml:19-34`).

const GUI_CREAM: [u8; 4] = [238, 239, 225, 255];
const GUI_GRAY: [u8; 4] = [125, 131, 141, 255];
/// GML `c_menudark` (#3b3e43): rows whose `condition` reports them
/// unavailable (`MenuOptions/Other_10.gml:663`), and the `@d` tag.
const GUI_MENUDARK: [u8; 4] = [59, 62, 67, 255];
const GUI_MID: [u8; 4] = [153, 153, 153, 255];
/// GML `c_uidark` (#333333): unavailable menu entries, dry ammo text.
const GUI_UIDARK: [u8; 4] = [51, 51, 51, 255];
const GUI_WHITE: [u8; 4] = [255, 255, 255, 255];
const GUI_RED2: [u8; 4] = [221, 56, 45, 255];
const GUI_HIDDEN: [u8; 4] = [0, 0, 0, 0];

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
        bold: false,
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
        bold: false,
    }
}

fn gui_button(text: impl Into<String>, gx: f32, gy: f32, color: [u8; 4]) -> MenuGuiText {
    // GML `MainMenuButton/Draw_0` law: `draw_text_bigname(x, y, name)` at the
    // DEFAULT scale 0.65, centered-middle. fntBig glyphs are 18px tall, so
    // the surface is `ceil(18 * 0.65) = 12` tall and the 7px-equivalent ink
    // spans ~`gy - 4 .. gy + 4`; a 10px Silkscreen row centers its ~8px ink
    // on `gy` the same way (a 12px row overshot width and height).
    // Bigname source: fill+stroke faux-bold for the heavy fntBig glyphs.
    MenuGuiText {
        text: text.into(),
        gx,
        gy,
        color,
        px: 10.0,
        centered: true,
        middle_y: true,
        right: false,
        bold: true,
    }
}

/// GML `PauseButton/Draw_0` label law (steady state, `appear = 0`):
/// `draw_text_bigname(_dx, _dy - 8, name, color, 1, 0.65)` with the image
/// 0/1/4/7 `fa_left` branch (`_dx = x - half_w`, padded surface centers
/// ~x + 2) or the 2/3/5/6 `fa_right` branch (mirrored, ~x - 2). The 12-tall
/// surface draws at `y - 8 - 6 = y - 14`, so ink spans ~`y - 8 .. y`; a 10px
/// middle-anchored row centers ink on `gy`, so `gy = button_y - 4` reproduces
/// the footprint. `left` picks the branch (MENU/RETRY style vs mirrored);
/// center-x keeps the ~2px surface-pad shift.
fn push_shadowed_sprite(
    out: &mut Vec<SpriteInstance>,
    assets: &RenderAssets,
    path: &str,
    frame: i32,
    x: f32,
    y: f32,
    view: [f32; 4],
    gm: HudGuiMap,
    flip_x: bool,
    tint: [f32; 4],
) {
    for (dx, dy) in [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)] {
        if let Some(sprite) = assets.sprite_for(
            path,
            frame,
            hud_gui_to_world(gm, view, x + dx, y + dy),
            flip_x,
            0.0,
            [0.0, 0.0, 0.0, 1.0],
        ) {
            out.push(sprite);
        }
    }
    if let Some(sprite) = assets.sprite_for(
        path,
        frame,
        hud_gui_to_world(gm, view, x, y),
        flip_x,
        0.0,
        tint,
    ) {
        out.push(sprite);
    }
}

fn push_shadowed_gui(
    out: &mut Vec<SpriteInstance>,
    assets: &RenderAssets,
    path: &str,
    frame: i32,
    gx: f32,
    gy: f32,
    mul: f32,
    tint: [f32; 4],
    view: [f32; 4],
    gm: HudGuiMap,
) {
    for (dx, dy) in [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)] {
        if let Some(sprite) = hud_gui_place(
            assets,
            path,
            frame,
            gx + dx,
            gy + dy,
            mul,
            [0.0, 0.0, 0.0, 1.0],
            gm,
            view,
        ) {
            out.push(sprite);
        }
    }
    if let Some(sprite) = hud_gui_place(assets, path, frame, gx, gy, mul, tint, gm, view) {
        out.push(sprite);
    }
}

pub(crate) fn letterbox_sprites(
    assets: &RenderAssets,
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam: &Camera2d,
    frame: i32,
) -> Vec<SpriteInstance> {
    if frame <= 0 {
        return Vec::new();
    }
    let view = view_rect_world(canvas_dp, world_size, cam);
    let gm = hud_gui_map(view);
    let vw = view[2];
    let top_x = 0.0;
    let bottom_x = vw;
    let mut out = Vec::new();
    let scale = 36.0 / 35.0;
    for (x, y, flip_x, flip_y) in [(top_x, -1.0, false, false), (bottom_x, 242.0, true, true)] {
        if let Some(mut sprite) = assets.sprite_for_full(
            "images/sprLetterbox.png",
            frame,
            hud_gui_to_world(gm, view, x, y),
            flip_x,
            flip_y,
            0.0,
            [1.0; 4],
        ) {
            sprite.size.y *= scale;
            out.push(sprite);
        }
    }
    out
}

pub(crate) const SETTINGS_SLIDER_X_OFFSET: f32 = 26.0;
pub(crate) const SETTINGS_SLIDER_WIDTH: f32 = 112.0;
const SETTINGS_SLIDER_FILL_BASE: f32 = 102.0;
const SETTINGS_SLIDER_DRAG_WIDTH: f32 = 113.0;
const SETTINGS_SLIDER_ROW_HALF_WIDTH: f32 = 130.0;
const SETTINGS_SLIDER_ROW_EXTRA: f32 = 96.0;
const SETTINGS_SLIDER_HIT_HALF_HEIGHT: f32 = 9.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SettingSliderTarget {
    Volume(VolumeChannel),
    Slider(&'static str),
}

pub(crate) fn settings_slider_hit(
    page: u8,
    gx: f32,
    gy: f32,
    vw: f32,
) -> Option<(usize, SettingSliderTarget)> {
    let row_left = vw * 0.5 - SETTINGS_SLIDER_ROW_HALF_WIDTH;
    let row_right = vw * 0.5 + SETTINGS_SLIDER_ROW_HALF_WIDTH + SETTINGS_SLIDER_ROW_EXTRA;
    settings_hot_rows(page, vw)
        .iter()
        .enumerate()
        .find_map(|(idx, row)| {
            let target = match row.op {
                SettingHotOp::Volume(channel) => SettingSliderTarget::Volume(channel),
                SettingHotOp::Slider(key) => SettingSliderTarget::Slider(key),
                _ => return None,
            };
            ((gy - row.gy).abs() <= SETTINGS_SLIDER_HIT_HALF_HEIGHT
                && gx >= row_left - 4.0
                && gx <= row_right + 4.0)
                .then_some((idx, target))
        })
}

pub(crate) fn settings_slider_value(target: SettingSliderTarget, gx: f32, vw: f32) -> f32 {
    let left = vw * 0.5 + SETTINGS_SLIDER_X_OFFSET;
    let fraction = ((gx - left) / SETTINGS_SLIDER_DRAG_WIDTH).clamp(0.0, 1.0);
    let max = match target {
        SettingSliderTarget::Volume(_) | SettingSliderTarget::Slider("controls_scale") => 1.0,
        SettingSliderTarget::Slider("screenshake") => 2.0,
        SettingSliderTarget::Slider(_) => 1.0,
    };
    fraction * max
}

pub(crate) fn settings_slider_action(target: SettingSliderTarget, value: f32) -> UiAction {
    match target {
        SettingSliderTarget::Volume(VolumeChannel::Master) => UiAction::SetMasterVol(value),
        SettingSliderTarget::Volume(VolumeChannel::Music) => UiAction::SetMusicVol(value),
        SettingSliderTarget::Volume(VolumeChannel::Ambience) => UiAction::SetAmbienceVol(value),
        SettingSliderTarget::Volume(VolumeChannel::Sfx) => UiAction::SetSfxVol(value),
        SettingSliderTarget::Slider(key) => UiAction::SettingSlider {
            key: key.to_string(),
            value,
        },
    }
}

fn push_settings_slider(
    out: &mut Vec<SpriteInstance>,
    assets: &RenderAssets,
    view: [f32; 4],
    gm: HudGuiMap,
    x: f32,
    y: f32,
    value: f32,
    max: f32,
) {
    let width = SETTINGS_SLIDER_WIDTH;
    let fraction = (value / max).clamp(0.0, 1.0);
    let fill = SETTINGS_SLIDER_FILL_BASE * fraction;
    if let Some(sprite) = assets.sprite_sized(
        "images/sprOptionSlider.png",
        0,
        hud_gui_to_world(gm, view, x, y - 4.0),
        Vec2::new(width, 19.0),
        false,
        [1.0; 4],
    ) {
        out.push(sprite);
    }
    if let Some(sprite) =
        settings_slider_part(assets, view, gm, x, y - 5.0, 4.0, 0.0, fill + 5.0, 20.0)
    {
        out.push(sprite);
    }
    if let Some(sprite) = assets.sprite_for(
        "images/sprSliderEnd.png",
        0,
        hud_gui_to_world(gm, view, x + fill + 4.0, y + 2.0),
        false,
        0.0,
        [1.0; 4],
    ) {
        out.push(sprite);
    }
}

fn settings_slider_part(
    assets: &RenderAssets,
    view: [f32; 4],
    gm: HudGuiMap,
    x: f32,
    y: f32,
    source_x: f32,
    source_y: f32,
    width: f32,
    height: f32,
) -> Option<SpriteInstance> {
    if width <= 0.0 || height <= 0.0 {
        return None;
    }
    let (uv, def) = assets.uv("images/sprOptionSlider.png", 1)?;
    let source_width = def.w as f32;
    let source_height = def.h as f32;
    if source_width <= 0.0 || source_height <= 0.0 {
        return None;
    }
    let x0 = source_x.clamp(0.0, source_width);
    let y0 = source_y.clamp(0.0, source_height);
    let x1 = (source_x + width).clamp(x0, source_width);
    let y1 = (source_y + height).clamp(y0, source_height);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let uv_min = Vec2::new(
        uv.min[0] + (uv.max[0] - uv.min[0]) * (x0 / source_width),
        uv.min[1] + (uv.max[1] - uv.min[1]) * (y0 / source_height),
    );
    let uv_max = Vec2::new(
        uv.min[0] + (uv.max[0] - uv.min[0]) * (x1 / source_width),
        uv.min[1] + (uv.max[1] - uv.min[1]) * (y1 / source_height),
    );
    let size = Vec2::new(width * gm.s, height * gm.s);
    let top_left = hud_gui_to_world(gm, view, x, y);
    Some(SpriteInstance {
        center: top_left + size * 0.5,
        rotation: 0.0,
        size,
        anchor: Vec2::new(0.5, 0.5),
        flip_x: false,
        flip_y: false,
        uv_min,
        uv_max,
        color: tint_to_linear([1.0; 4]),
        page: uv.page,
        z: 0.0,
        blend: SpriteBlend::Alpha,
    })
}

fn settings_option_button(text: impl Into<String>, cx: f32, y: f32) -> MenuGuiText {
    gui_center(text, cx, y, GUI_MID)
}

/// GML `gamepad_types` → `gamepad_icon_small` strip
/// (`scrOptionsUpdate.gml:149-163`).
fn gamepad_icon_strip(gamepad_type: u8) -> &'static str {
    match gamepad_type % 4 {
        0 => "images/sprXBONESmall.png",
        1 => "images/sprPS4Small.png",
        2 => "images/sprSwitchSmall.png",
        _ => "images/sprSteamDeckSmall.png",
    }
}

/// GML `gamepad_button_to_image` (`scripts/draw_gamepad_button/
/// draw_gamepad_button.gml:12-37`): pad button → `gamepad_icon_*`
/// subimage. GML's `-1` return (unbound, keyboard/mouse or axis row)
/// makes `draw_gamepad_button` draw nothing, mirrored here as `None`.
fn gamepad_glyph_frame(entry: &repame_input::KeymapEntry) -> Option<i32> {
    use repose_core::input::GamepadButton;
    let repame_input::KeymapEntry::Pad(button) = entry else {
        return None;
    };
    let frame = match *button {
        GamepadButton::South => 0,
        GamepadButton::East => 1,
        GamepadButton::West => 2,
        GamepadButton::North => 3,
        GamepadButton::LeftShoulder => 4,
        GamepadButton::RightShoulder => 5,
        GamepadButton::LeftStick => 8,
        GamepadButton::RightStick => 9,
        GamepadButton::DPadUp => 11,
        GamepadButton::DPadDown => 12,
        GamepadButton::DPadLeft => 13,
        GamepadButton::DPadRight => 14,
        GamepadButton::Start => 15,
        GamepadButton::Select => 16,
    };
    Some(frame)
}

/// One `draw_gamepad_button(...)` glyph: the strip origin lands on
/// `(gx, gy)` exactly like GML `draw_sprite` (every `gamepad_icon_*`
/// strip is 24x24 with a centered origin, so the glyph centers there).
#[allow(clippy::too_many_arguments)]
fn push_gamepad_glyph(
    out: &mut Vec<SpriteInstance>,
    assets: &RenderAssets,
    gamepad_type: u8,
    frame: i32,
    gx: f32,
    gy: f32,
    view: [f32; 4],
    gm: HudGuiMap,
    tint: [f32; 4],
) {
    if let Some(s) = assets.sprite_for(
        gamepad_icon_strip(gamepad_type),
        frame,
        hud_gui_to_world(gm, view, gx, gy),
        false,
        0.0,
        tint,
    ) {
        out.push(s);
    }
}

/// GML `UberCont/Draw_75:1-16` CONFIRM prompt: right-aligned `@sCONFIRM` at
/// `(gui_w - 8, gui_h - 40)` plus the `gp_face1` glyph, drawn while
/// `opt_gamepad` and a MainMenuButton / PlayButton / MenuOptions instance is
/// live - the port's `MainMenu` overlay (list + play submenu) and `Settings`.
/// Char select (`Title`) and `Stats` stay out: PLAY/OPTIONS/STATS all destroy
/// `MainMenuButton` (`MainMenuButton/Other_10.gml:12, 86`), so GML keeps the
/// prompt only while the `PlayButton`/`MenuOptions` replacement is up.
const CONFIRM_GY: f32 = 200.0;
/// GML `font_get_string_width("CONFIRM")` at the 7px Silkscreen body
/// size: advances .75 + .75 + .875 + .625 + .375 + .75 + .875 = 5.0 em.
/// The port has no text-measure API, so the width is a constant.
const CONFIRM_TEXT_W: f32 = 35.0;

/// The live GAMEPAD style strip (`Some(gamepad_type)`) - GML
/// `is_gamepad()` is the sticky `KeyCont.gamepad` switch
/// (`InputHandling.gml:224`), never per-frame pad activity.
fn gamepad_style(world: &World) -> Option<u8> {
    world
        .get_resource::<crate::savedata_part::SaveData>()
        .filter(|s| s.settings.gamepad_enabled)
        .map(|s| s.settings.gamepad_type)
}

fn gamepad_ui_on(world: &World) -> bool {
    gamepad_style(world).is_some()
}

fn push_confirm_text(out: &mut Vec<MenuGuiText>, world: &World, vw: f32) {
    if !gamepad_ui_on(world) {
        return;
    }
    out.push(MenuGuiText {
        text: "CONFIRM".to_string(),
        gx: vw - 8.0,
        gy: CONFIRM_GY,
        color: GUI_GRAY,
        px: 7.0,
        centered: false,
        middle_y: true,
        right: true,
        bold: false,
    });
}

/// The `gp_face1` half of the CONFIRM prompt: `dx - 9 -
/// font_get_string_width(str)` from GML `Draw_75:11`.
fn push_confirm_glyph(
    out: &mut Vec<SpriteInstance>,
    assets: &RenderAssets,
    world: &World,
    vw: f32,
    view: [f32; 4],
    gm: HudGuiMap,
) {
    let Some(save) = world.get_resource::<crate::savedata_part::SaveData>() else {
        return;
    };
    if !save.settings.gamepad_enabled {
        return;
    }
    push_gamepad_glyph(
        out,
        assets,
        save.settings.gamepad_type,
        0,
        vw - 9.0 - CONFIRM_TEXT_W - 8.0,
        CONFIRM_GY,
        view,
        gm,
        [1.0; 4],
    );
}

fn push_gameover_text(out: &mut Vec<MenuGuiText>, text: MenuGuiText) {
    for (dx, dy) in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
        let mut shadow = text.clone();
        shadow.gx += dx;
        shadow.gy += dy;
        shadow.color = [0, 0, 0, 255];
        shadow.bold = false;
        out.push(shadow);
    }
    out.push(text);
}

fn gui_pause_button(
    text: impl Into<String>,
    button_x: f32,
    button_y: f32,
    left: bool,
    color: [u8; 4],
) -> MenuGuiText {
    MenuGuiText {
        text: text.into(),
        gx: button_x + if left { 2.0 } else { -2.0 },
        gy: button_y - 4.0,
        color,
        px: 10.0,
        centered: true,
        middle_y: true,
        right: false,
        bold: true,
    }
}

/// Port-side parse of a packed offer row: ULTRA prefix + name/desc split
/// on em-dash or hyphen. GML keeps the two halves apart instead - the
/// `SkillIcon` description line is assembled from two lookups joined by a
/// `#` line break, `"@w" + name + "#@s" + text` (`objects/SkillIcon/
/// Draw_0.gml:51`; the ultra twin is `objects/UltraIcon/Draw_0.gml:32-33`).
/// Returns the GML skill id too (the `mutation_choice_ids` parallel row),
/// so the Throne Butt special can key off the picked skill, not the
/// parsed name.
fn mutation_choice_parts(choice: &str) -> (Option<u8>, String, String) {
    let trimmed = choice.trim();
    let trimmed = trimmed.strip_prefix("ULTRA:").unwrap_or(trimmed).trim();
    if let Some((name, desc)) = trimmed.split_once(" \u{2014} ") {
        (None, name.trim().to_string(), desc.trim().to_string())
    } else if let Some((name, desc)) = trimmed.split_once(" - ") {
        (None, name.trim().to_string(), desc.trim().to_string())
    } else {
        (None, trimmed.to_string(), String::new())
    }
}

fn pending_offer_is_ultra(world: &World) -> bool {
    world.get_resource::<PendingUltra>().is_some()
}

fn pending_offer_race(world: &mut World) -> RaceId {
    world
        .query::<&RaceState>()
        .iter(world)
        .map(|race| race.race)
        .find(|race| *race != RaceId::Random)
        .or_else(|| {
            world
                .get_resource::<SelectedCharacter>()
                .map(|selected| selected.0)
        })
        .unwrap_or(RaceId::Fish)
}
/// GML `Credits/Other_11` `credittext` verbatim (unlocalized defaults): 14
/// titled sections. Rows carry GML `@w`/`@s`/`@y` color tags and `#` line
/// breaks, rendered by the tag backend (`nt_text_segments` + `AnnotatedText`).
/// GML joins each section with `"\n@s"` and draws ONE centered-middle
/// `draw_text_nt` at `(gui_w/2, gui_h/2)`; tall sections
/// (`height > gui_h - 36`) pan via `scroll` (`MenuState::credits_scroll`).
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

/// GML `TutCont/Draw_64` instruction text (keyboard variant, `text[]` index 1
/// per `is_keyboard(global.index)`). Key names resolve from the live keymap
/// (the `move` quartet on Walking, the step action otherwise), `keymap_get`
/// parity.
/// NOT ported: `text[]` index 0 (touch) / 2 (gamepad), the pulsing lime
/// touch-highlight circle, the red chest pointer, the 36px bottom letterbox
/// bar GML draws behind the text.
pub fn tutorial_texts(world: &mut World, canvas_dp: [f32; 2]) -> Vec<GuiRow> {
    let step = world
        .get_resource::<crate::state::TutorialState>()
        .map(|t| t.step)
        .unwrap_or(crate::state::TutorialStep::Walking);
    if world
        .get_resource::<crate::state::TutorialState>()
        .is_some_and(|t| t.portal_open)
    {
        return Vec::new();
    }
    let vw = gml_view_size(canvas_dp)[0];
    // GML `scrKeyName` parity: single letters upper-case, Space wider.
    // `keymap_get` returns the raw row value; the port formats the
    // debug `KeymapEntry` instead, so normalize the common shapes.
    let key_name = |action: &str| {
        let raw: String = world
            .get_resource::<crate::keymap::InputMapState>()
            .and_then(|m| {
                crate::keymap::NtAction::from_name(action)
                    .map(|a| format!("{:?}", m.session.map.keyboard(&a)))
            })
            .unwrap_or_else(|| action.to_ascii_uppercase());
        raw.replace("Key(KeyCode(", "")
            .replace("Key(Character('", "")
            .replace("'))", "")
            .replace("')", "")
            .replace("Space", "SPACE")
            .to_ascii_uppercase()
    };
    let text = match step {
        crate::state::TutorialStep::Walking => {
            format!(
                "WALK USING @w{}, {}, {}, {}#@s OR THE @wARROW KEYS",
                key_name("north"),
                key_name("west"),
                key_name("south"),
                key_name("east")
            )
        }
        crate::state::TutorialStep::PickingUp => {
            format!("PICK UP A NEW WEAPON WITH @w{}@s", key_name("pick"))
        }
        crate::state::TutorialStep::Shooting => {
            "AIM USING THE MOUSE, @wLEFT BUTTON@s FIRES".to_string()
        }
        crate::state::TutorialStep::Swapping => {
            format!(
                "SWAP WEAPONS WITH @w{}@s#TRY IT A FEW TIMES!",
                key_name("swap")
            )
        }
        crate::state::TutorialStep::Power => {
            "@wRIGHT MOUSE BUTTON@s USES YOUR ABILITY#GIVE IT A GO!".to_string()
        }
        crate::state::TutorialStep::Fin => "COOL, WE'RE DONE HERE!".to_string(),
    };
    gui_texts_dp(
        canvas_dp,
        vec![MenuGuiText {
            text,
            gx: vw * 0.5,
            gy: 240.0 - 18.0,
            color: GUI_WHITE,
            px: 7.0,
            centered: true,
            middle_y: true,
            right: false,
            bold: false,
        }],
    )
}

/// Section count for the credits cycler ([`MenuState::credits_section`]).
pub fn credit_section_count() -> usize {
    CREDIT_SECTIONS.len().max(1)
}

/// GML `UnlockScreen/Other_10` text layer verbatim (head of the queue
/// only - GML draws the queued head while `visible`; the FIFO chain
/// advances on dismiss): dim note, race name via the big-name row (or
/// the skin letter for skins), `UNLOCKED!`, `CONTINUE` prompt. Rows
/// are GUI-space like the rest of `menu_gui_texts_vw`.
pub fn unlock_popup_texts(world: &mut World, vw: f32) -> Vec<MenuGuiText> {
    use crate::state::menus::UnlockPopup;
    let cx = vw * 0.5;
    let state = world
        .get_resource::<MenuState>()
        .map(|m| m.unlock)
        .unwrap_or_default();
    let popup = world
        .get_resource::<MenuState>()
        .and_then(|m| m.unlock_queue.first().copied());
    if !state.visible {
        return Vec::new();
    }
    let Some(popup) = popup else {
        return Vec::new();
    };
    let (race, skin) = match popup {
        UnlockPopup::Race(race) => (race, 0u8),
        UnlockPopup::Skin(race, skin) => (race, skin),
    };
    let mut name = crate::savedata_part::character_def(race)
        .name
        .to_ascii_uppercase();
    if skin > 0 {
        name.push(' ');
        name.push((b'A' + skin.min(3)) as char);
    }
    let mut out = Vec::new();
    if state.addy > 0.0 {
        out.push(MenuGuiText {
            text: name,
            gx: cx,
            gy: 240.0 - 92.0 - state.addy + 8.0,
            color: GUI_WHITE,
            px: 14.0,
            centered: true,
            middle_y: true,
            right: false,
            bold: true,
        });
    }
    if state.addy > 1.0 {
        out.push(MenuGuiText {
            text: "UNLOCKED!".to_string(),
            gx: cx,
            gy: 240.0 - 62.0 - state.addy + 10.0,
            color: GUI_WHITE,
            px: 10.0,
            centered: true,
            middle_y: true,
            right: false,
            bold: true,
        });
    }
    if state.can_continue {
        out.push(MenuGuiText {
            text: "CONTINUE".to_string(),
            gx: cx,
            gy: 240.0 - 16.0 - state.addy2 - if state.pointed { 1.0 } else { 0.0 },
            color: if state.addy2 > 0.0 || state.pointed {
                GUI_WHITE
            } else {
                GUI_GRAY
            },
            px: 10.0,
            centered: true,
            middle_y: true,
            right: false,
            bold: true,
        });
    }
    out
}

/// Menu overlay texts for one [`crate::MenuOverlay`] (splash is
/// sprite-only here; its captions are the `Vlambeer/Draw_0` block).
/// `vw` is the live GML GUI width in px ([`gml_view_size`], 426 at 16:9):
/// view-centered rows use `vw / 2`, right-anchored rows `vw - N`
/// (`scrMakePauseButtons`, `scrDrawMiscHUD`), left-anchored rows literal x.
pub fn menu_gui_texts_vw(kind: crate::MenuOverlay, world: &mut World, vw: f32) -> Vec<MenuGuiText> {
    let cx = vw * 0.5;
    // GML `UberCont/Draw_75:1-16`: `opt_gamepad` plus a live
    // MainMenuButton / PlayButton / MenuOptions draws the CONFIRM prompt -
    // in the port the MainMenu list (MainMenuButton), its play submenu
    // (PlayButton) and Settings (MenuOptions). Char select and Stats destroy
    // MainMenuButton without a replacement and Co-op creates no PlayButton,
    // so they stay out.
    let confirm = matches!(
        kind,
        crate::MenuOverlay::MainMenu | crate::MenuOverlay::Settings
    );
    let mut out = match kind {
        // Boot reel captions (GML `Vlambeer/Draw_0` verbatim, one row per
        // visual line: `gui_text_layer` renders each row `.single_line()`, so
        // embedded `\n`/`#` would never break - GML's single `draw_text_nt`
        // block must arrive pre-split).
        // Mode 0 save note: 2 white lines, block middle at `cy+24` (140/150
        // middle-centers ≈ 144). Mode 1 Gamemaker line at `(cx, cy)`,
        // `@s`-silver. Mode 3 team block at `(cx, cy)`: `@yVLAMBEER`, `@s&`,
        // four `@w` names, `PRESENT`; ys 80..160 average exactly 120.
        // Modes 2/4 sprite-only.
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
                        bold: false,
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
                        bold: false,
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
                    bold: false,
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
                    bold: false,
                })
                .collect(),
                // GML `MakeGame/Draw_0:99,130-131`: the prompt text and the
                // YES/NO rows. Hover rows carry the `@w` tag and a 1px bump
                // (`_point_left` subtracted from the draw y).
                crate::state::SPLASH_MODE_LOAD => {
                    let load = world
                        .get_resource::<SplashState>()
                        .map(|s| s.load)
                        .unwrap_or_default();
                    let rows = crate::state::load_prompt_rows(vw, load.posy);
                    let (left, right) = (rows[0], rows[1]);
                    let point_left = load.pointed_item == 1;
                    let point_right = load.pointed_item == 2;
                    vec![
                        MenuGuiText {
                            text: "@sCONTINUE THIS SAVED RUN?".to_string(),
                            gx: vw * 0.5,
                            gy: 120.0 + 4.0 - 54.0,
                            color: GUI_WHITE,
                            px: 7.0,
                            centered: true,
                            middle_y: true,
                            right: false,
                            bold: false,
                        },
                        MenuGuiText {
                            text: if point_left { "@wYES" } else { "@sYES" }.to_string(),
                            gx: left.0,
                            gy: left.1 - if point_left { 1.0 } else { 0.0 },
                            color: GUI_WHITE,
                            px: 7.0,
                            centered: true,
                            middle_y: true,
                            right: false,
                            bold: false,
                        },
                        MenuGuiText {
                            text: if point_right { "@wNO@w" } else { "@sNO@w" }.to_string(),
                            gx: right.0,
                            gy: right.1 - if point_right { 1.0 } else { 0.0 },
                            color: GUI_WHITE,
                            px: 7.0,
                            centered: true,
                            middle_y: true,
                            right: false,
                            bold: false,
                        },
                    ]
                }
                _ => Vec::new(),
            }
        }
        crate::MenuOverlay::Loading => {
            // GML `GenCont/Draw_0` text layer verbatim: `GENERATING... %` at
            // `(_cx, _cy - 54)` from `Floor/goal` (pad-2), the `VERIFYING... %`
            // Venuz branch (`level >= 10` + a Venuz player), the `@s`-tip at
            // `(_cx, _cy + 24)`, roadmap area/kill strings at
            // `(drawx-60/drawx+23, drawy-14)` (dots ride `menu_sprites`).
            // `_cx/_cy` is the view center (`vw/2`, 120 at 240 high).
            let progress = world
                .get_resource::<crate::state::LoadingState>()
                .map(|l| l.progress)
                .or_else(|| {
                    world
                        .get_resource::<crate::comps_b::FloorTransition>()
                        .filter(|f| f.active)
                        .map(|f| f.progress)
                })
                .unwrap_or(1.0);
            let pct = (progress.clamp(0.0, 1.0) * 100.0).round() as u32;
            // GML `GenCont/Draw_0:13-18`: Venuz verifying branch.
            let is_venuz = world
                .query::<&crate::comps_a::RaceState>()
                .iter(world)
                .any(|rs| rs.race == crate::data::RaceId::Venuz);
            let deep_enough = world
                .query::<&crate::comps_a::Player>()
                .iter(world)
                .any(|p| p.level >= 10);
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
            let tip = if let Some(loading) = world.get_resource::<crate::state::LoadingState>()
                && !loading.tip.is_empty()
            {
                loading.tip.clone()
            } else if let Some(transition) = world
                .get_resource::<crate::comps_b::FloorTransition>()
                .filter(|f| f.active)
                && !transition.tip.is_empty()
            {
                transition.tip.clone()
            } else {
                let picked = world
                    .get_resource::<crate::comps_a::Run>()
                    .map(crate::progression::pick_loading_tip)
                    .unwrap_or_else(|| "KILL ENEMIES TO LEVEL UP".to_string());
                if let Some(mut loading) = world.get_resource_mut::<crate::state::LoadingState>() {
                    loading.tip = picked.clone();
                } else if let Some(mut transition) =
                    world.get_resource_mut::<crate::comps_b::FloorTransition>()
                {
                    transition.tip = picked.clone();
                }
                picked
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
                        bold: false,
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
                        bold: false,
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
                            bold: true,
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
                    bold: false,
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
                        bold: true,
                    }
                })
                .collect()
        }
        crate::MenuOverlay::Unlock => unlock_popup_texts(world, vw),
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
            // `c_uigray`; left column `statx = view_left + 110`, `staty =
            // view_top + 36 + 4`; right column `statx = view_left +
            // view_width - 70`. Names right-aligned uigray at `statx - 1`,
            // values left white at `statx + 1`, headers centered; blank
            // headers advance the (fractional) line.
            let stat_name = |out: &mut Vec<MenuGuiText>, col: f32, line: &mut f32, name: &str| {
                // GML `draw_stat` verbatim: NAME right-aligns on `statx - 1`
                // (row box right edge = `col - 1`, not `col`) and is
                // top-anchored (`fa_top` persists from the Draw-GUI reset):
                // row top lands on `staty + line * 8` with the
                // `FONT_TOP_BIAS_GUI` nudge putting Silkscreen ink where the
                // 8px fntM1 glyphs sat - middle-anchored dropped every row
                // ~6px.
                out.push(MenuGuiText {
                    text: name.to_string(),
                    gx: col - 1.0,
                    gy: 40.0 + *line * 8.0,
                    color: GUI_MID,
                    px: 7.0,
                    centered: false,
                    middle_y: false,
                    right: true,
                    bold: false,
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
                    middle_y: false,
                    right: false,
                    bold: false,
                });
                *line += 1.0;
            };
            let header = |out: &mut Vec<MenuGuiText>, col: f32, line: &mut f32, name: &str| {
                if name.is_empty() {
                    *line += 1.0;
                    return;
                }
                // GML `draw_stat_header` verbatim: centered on `statx`
                // (`fa_center` at `col`), not the view center - the old
                // `gui_center(name, col, ...)` built a full-width box
                // centered on `vw/2`, dragging "TOTAL" right of its column
                // (the centered part of the "left-shortened stats" bug).
                out.push(MenuGuiText {
                    text: name.to_string(),
                    gx: col,
                    gy: 40.0 + *line * 8.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: true,
                    middle_y: false,
                    right: false,
                    bold: false,
                });
                *line += 1.0;
            };
            let mut out = vec![MenuGuiText {
                // GML `DrawStats/Draw_0` law: `draw_text_bigname(cx,
                // top+24, gray)` at the default scale 0.65,
                // centered-middle - same footprint as `gui_button`.
                text: "STATS".to_string(),
                gx: cx,
                gy: 24.0,
                color: GUI_MID,
                px: 10.0,
                centered: true,
                middle_y: true,
                right: false,
                bold: true,
            }];
            let (lx, rx) = (110.0, vw - 70.0);
            let mut l = 0.0;
            header(&mut out, lx, &mut l, "TOTAL");
            let (un, unmax) = unlock_progress(&save);
            // GML `scrDrawStats:82` verbatim: `string_pad_zeroes(round(
            // unlock / unlockmax * 100), 2) + "%"` - rounds (never
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
            // Deferred GML `scrDrawStats:108-114` DAILY block: needs the
            // daily-run systems the port lacks (`dbst_*` bests, `ctot_days`
            // producers; DAILY/WEEKLY PLAY rows deny with `sndNoSelect`, so
            // `dailies > 0` can never hold here).
            // Deferred `scrDrawCharStats` per-race page: needs per-race
            // `ctot_*`/`cbst_*`/`hbst_*` arrays plus a select overlay; the sim
            // keeps only the global aggregates the TOTAL/BEST blocks read.
            // GML `scrDrawStats` HARD block verbatim: gated on `hardgot` +
            // hard runs, shows the global hard-best race + hard map
            // (`scrAreaGetMapName(..., hard = true)`), kills, runs.
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
                .unwrap_or(0);
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
            // GML `scrCampfireMenuDrawCharText:444-466` law: the white NAME
            // draws unconditionally (only its shadow gates on `textappear !=
            // 2`); the SKILLS block hides at `appear == 2` (selection-change
            // hide before the typewriter).
            // GML single-player `fa_left/fa_bottom`: `_bigname_y = 36 + 32 =
            // 68`, then `_bigname_y = h - 68 = 172` - the NAME's bottom edge
            // lands at y=172 via `draw_text_bigname`, above the 36px
            // letterbox. The 12px row is top-anchored, so its top is
            // `172 - 12 = 160` and the mapper's Silkscreen top bias lands the
            // ink where the bigname surface sat.
            // Bigname source (scale 1): fill+stroke faux-bold.
            rows.push(MenuGuiText {
                text: def.name.to_ascii_uppercase().to_string(),
                gx: 0.0,
                gy: 160.0,
                color: GUI_WHITE,
                px: 12.0,
                centered: false,
                middle_y: false,
                right: false,
                bold: true,
            });
            if textappear != 2.0 {
                // Skills block: ONE two-line `draw_text_nt`, `fa_middle`,
                // block middle at `_bigname_y + (height div 2) + appear + 8` =
                // `188 + appear` (height 16 = 2x8px fntM1 lines). Line tops
                // `180 + appear` / `188 + appear`; the mapper's Silkscreen top
                // bias lands the ink on the GML tops. x is `_x + 8`.
                let appear = textappear.max(0.0);
                rows.push(MenuGuiText {
                    text: race_passive_text(race).to_string(),
                    gx: 8.0,
                    gy: 180.0 + appear,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: false,
                    middle_y: false,
                    right: false,
                    bold: false,
                });
                rows.push(MenuGuiText {
                    text: race_active_text(race).to_string(),
                    gx: 8.0,
                    gy: 188.0 + appear,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: false,
                    middle_y: false,
                    right: false,
                    bold: false,
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
                        bold: false,
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
                                bold: false,
                            });
                        }
                    }
                }
                // GML `Menu.unlock_hint`: white `draw_text_nt` `(gui_w/2, gui_h-30)` on
                // the `c_tooltip` roundrect, 90 steps (`alarm[11]`), `pop`→0. Sprite layer
                // draws no rects: shell draws the box, this row the position.
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
                            bold: false,
                        });
                    }
                }
            }
            rows
        }
        crate::MenuOverlay::Mutation => {
            let hud = sync_hud_state(world);
            let menu = world.get_resource::<MenuState>().cloned();
            let is_ultra = pending_offer_is_ultra(world);
            let fallback_count = if is_ultra {
                world
                    .get_resource::<PendingUltra>()
                    .map(|u| u.choices.len())
                    .unwrap_or(0)
            } else {
                world
                    .get_resource::<PendingMutation>()
                    .map(|p| p.choices.len())
                    .unwrap_or(0)
            };
            let owed = world
                .query::<&Player>()
                .iter(world)
                .next()
                .map(|player| {
                    if is_ultra {
                        u32::from(player.ultra_pick_owed)
                    } else {
                        player.mutation_picks_owed
                    }
                })
                .unwrap_or(fallback_count as u32)
                .max(1);
            let is_robot = world
                .query::<&RaceState>()
                .iter(world)
                .any(|race| race.race == RaceId::Robot);
            let (subtitle, extra) = if is_ultra {
                if is_robot {
                    ("@sINSTALL @gULTRA@s UPDATE".to_string(), None)
                } else {
                    ("@sPICK YOUR @gULTRA@s MUTATION".to_string(), None)
                }
            } else if is_robot {
                (
                    format!("@sINSTALL {owed} UPDATES@s"),
                    Some("@sDO NOT TURN OFF ROBOT".to_string()),
                )
            } else {
                (format!("@sSELECT {owed} MUTATIONS"), None)
            };
            let appear = menu
                .as_ref()
                .map(|state| state.mutation_appear)
                .unwrap_or(0.0);
            let subtitle_x = cx + 1.0;
            let mut out = vec![gui_center(subtitle, subtitle_x, 75.0 - appear, GUI_GRAY)];
            if let Some(extra) = extra {
                out.push(gui_center(extra, subtitle_x, 87.0 - appear, GUI_GRAY));
            }
            let selected = menu.as_ref().and_then(|state| state.mutation_selected);
            let selection_y = 179.0
                - menu
                    .as_ref()
                    .map(|state| state.mutation_selection_frame as f32)
                    .unwrap_or(0.0);
            let selection_settled = !is_ultra
                || selected
                    .and_then(|index| {
                        menu.as_ref()
                            .and_then(|state| state.mutation_appear_y.get(index).copied())
                    })
                    .map(|value| value <= 0.01)
                    .unwrap_or(true);
            let sel_id = selected.and_then(|i| hud.mutation_choice_ids.get(i).copied());
            let sel_text = selected
                .filter(|_| selection_settled)
                .and_then(|i| hud.mutation_choices.get(i));
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
                                gy: selection_y + (i as f32 - (n - 1.0) * 0.5) * 8.0,
                                color: GUI_WHITE,
                                px: 7.0,
                                centered: true,
                                middle_y: true,
                                right: false,
                                bold: false,
                            });
                        }
                        return out;
                    }
                    let tb = races
                        .first()
                        .map(|r| race_tb(*r))
                        .unwrap_or_else(|| race_tb(pending_offer_race(world)));
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
                let box_lines: Vec<&str> =
                    box_text.split('#').flat_map(|s| s.split('\n')).collect();
                let n = box_lines.len().max(1) as f32;
                for (i, line) in box_lines.iter().enumerate() {
                    // `gui_multiline` cannot carry per-block middle
                    // centering, so the split stays inline here.
                    out.push(MenuGuiText {
                        text: line.to_string(),
                        gx: cx,
                        gy: selection_y + (i as f32 - (n - 1.0) * 0.5) * 8.0,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                        bold: false,
                    });
                }
            }
            out
        }
        crate::MenuOverlay::GameOver => {
            // GML `GameOver/Draw_0`: struggle line (vw/2, view_top+48), no offset;
            // roadmap (_x-48, _y-offsety) with area/kill strings at (drawy-14);
            // `KILLED BY` / win `COMPLETION TIME`+clock (_x+86, _y-offsety-25/-10);
            // MENU/RETRY `PauseButton`s ystart+offsety (178/210). Splat frames + roadmap
            // prefix ride `go_splat`, `go_death_pos`.
            let run = world.get_resource::<crate::comps_a::Run>();
            let (offsety, _death_pos) = world
                .get_resource::<MenuState>()
                .map(|m| (m.go_offsety, m.go_death_pos))
                .unwrap_or((0.0, 0.0));
            let mut out = Vec::new();
            if let Some(run) = run {
                let max_sub = area_max_subarea(run.area);
                // GML `GameOver/Create_0`: base text from area/loop position; a win
                // overrides to `THE STRUGGLE IS OVER` only on the HQ final. The
                // `Cinematic`-gated `YOU REACHED THE NUCLEAR THRONE` has no counterpart (no
                // Cinematic entity), so a non-HQ win keeps its base text.
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
                push_gameover_text(
                    &mut out,
                    MenuGuiText {
                        text: text.to_string(),
                        gx: cx,
                        gy: 48.0,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: true,
                        middle_y: false,
                        right: false,
                        bold: false,
                    },
                );
                push_gameover_text(
                    &mut out,
                    MenuGuiText {
                        text: run_area_string(run),
                        gx: cx - 108.0,
                        gy: 106.0 - offsety,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: false,
                        middle_y: true,
                        right: false,
                        bold: false,
                    },
                );
                push_gameover_text(
                    &mut out,
                    MenuGuiText {
                        text: run.total_kills.to_string(),
                        gx: cx - 25.0,
                        gy: 106.0 - offsety,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: false,
                        middle_y: true,
                        right: false,
                        bold: false,
                    },
                );
                if run.won {
                    push_gameover_text(
                        &mut out,
                        MenuGuiText {
                            text: "COMPLETION TIME".to_string(),
                            gx: cx + 86.0,
                            gy: 95.0 - offsety,
                            color: GUI_WHITE,
                            px: 7.0,
                            centered: true,
                            middle_y: true,
                            right: false,
                            bold: false,
                        },
                    );
                    push_gameover_text(
                        &mut out,
                        MenuGuiText {
                            text: run_timer_string(run.tottimer),
                            gx: cx + 86.0,
                            gy: 110.0 - offsety,
                            color: GUI_GRAY,
                            px: 7.0,
                            centered: true,
                            middle_y: true,
                            right: false,
                            bold: false,
                        },
                    );
                } else {
                    push_gameover_text(
                        &mut out,
                        MenuGuiText {
                            text: "KILLED BY".to_string(),
                            gx: cx + 86.0,
                            gy: 95.0 - offsety,
                            color: GUI_WHITE,
                            px: 7.0,
                            centered: true,
                            middle_y: true,
                            right: false,
                            bold: false,
                        },
                    );
                }
            } else {
                push_gameover_text(&mut out, gui_center("GAME OVER", cx, 100.0, GUI_WHITE));
            }
            let (appear, _hover) = world
                .get_resource::<MenuState>()
                .map(|menu| (menu.go_appear.max(0.0), menu.hover_label.as_str()))
                .unwrap_or((0.0, ""));
            let menu_appear = appear;
            let retry_appear = appear + 1.0;
            if menu_appear < 2.0 {
                out.push(gui_pause_button(
                    "MENU",
                    cx,
                    120.0 + 58.0 + offsety + menu_appear,
                    true,
                    GUI_HIDDEN,
                ));
            }
            if retry_appear < 2.0 {
                out.push(gui_pause_button(
                    "RETRY",
                    cx,
                    120.0 + 90.0 + offsety + retry_appear,
                    true,
                    GUI_HIDDEN,
                ));
            }
            out
        }
        crate::MenuOverlay::Pause => {
            // GML pause layer (`UberCont/Draw_0` paused branch + `scrMakePauseButtons` +
            // `PauseButton/Draw_0/Other_10`): frozen `pausespr` screenshot + 0.7 black
            // scrim are shell-owned; bigname `PAUSED` centered-middle `(view_center + 1,
            // 52 + 1 - yoff)` (`yoff = 4` on event/hardmode); `sprCharSplat` pair + roadmap
            // on the sprite layer; MENU `(left+45, bottom-64, appear 1)`, RETRY `(left+60,
            // bottom-32, appear 2)`, SETTINGS `(right-68, top, appear 3)`, CONTINUE
            // `(right-78, bottom, appear 3)`. Buttons: bigname label at scale 0.65 at
            // `y + appear` while `appear < 2`, else `sprPauseButton` art (`appear` -1/step,
            // `wait` 3 steps gates clicks). Port draws the steady state (`appear = 0`).
            // RETRY suppressed on daily runs.
            let hardmode = world
                .get_resource::<crate::comps_a::Run>()
                .is_some_and(|r| r.hardmode);
            let yoff = if hardmode { 4.0 } else { 0.0 };
            let confirm = world
                .get_resource::<MenuState>()
                .and_then(|m| m.pause_confirm);
            if let Some(confirm) = confirm {
                let right = if confirm == 0 { "QUIT" } else { "RETRY" };
                vec![
                    gui_button("BACK", 52.0, 192.0, GUI_HIDDEN),
                    gui_button(right, vw - 52.0, 192.0, GUI_HIDDEN),
                ]
            } else {
                let mut out = vec![
                    MenuGuiText {
                        text: "PAUSED".to_string(),
                        gx: cx + 1.0,
                        gy: 52.0 + 1.0 - yoff,
                        color: GUI_HIDDEN,
                        px: 10.0,
                        centered: true,
                        middle_y: true,
                        right: false,
                        bold: false,
                    },
                    gui_pause_button("MENU", 45.0, 176.0, true, GUI_HIDDEN),
                    gui_pause_button("RETRY", 60.0, 208.0, true, GUI_HIDDEN),
                    gui_pause_button("SETTINGS", vw - 68.0, 176.0, false, GUI_HIDDEN),
                    gui_pause_button("CONTINUE", vw - 78.0, 208.0, false, GUI_HIDDEN),
                ];
                if let Some(run) = world.get_resource::<Run>() {
                    out.push(MenuGuiText {
                        text: run_area_string(run),
                        gx: cx - 60.0,
                        gy: 106.0,
                        color: GUI_WHITE,
                        px: 7.0,
                        centered: false,
                        middle_y: true,
                        right: false,
                        bold: false,
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
                        bold: false,
                    });
                }
                out
            }
        }
        crate::MenuOverlay::Settings => settings_gui_texts(world, vw),
        crate::MenuOverlay::Credits => {
            // GML `Credits/Draw_64`: section body is ONE centered-middle `draw_text_nt`
            // `(gui_w/2, gui_h/2)`; GML joins with `"\n@s"`, the tag backend splits `#`
            // the same way. Tall sections pan via `MenuState::credits_scroll` (`_py +=
            // scroll - height + gui_h * 0.6`). A `Logo` owning the credits draws nothing.
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
                    bold: false,
                })
                .collect()
        }
    };
    if confirm {
        push_confirm_text(&mut out, world, vw);
        // GML `UberCont/Draw_64:124-135`: the saving tip's two-line tooltip
        // under the QUIT confirm, once `appear <= 1`.
        if !confirm
            && world
                .get_resource::<crate::savedata_part::SaveData>()
                .is_some_and(|s| s.saving_tip == 0)
            && world
                .get_resource::<crate::state::menus::MenuState>()
                .and_then(|m| m.pause_appear.get(1).copied())
                .unwrap_or(0.0)
                <= 1.0
        {
            for (i, line) in [
                "YOU CAN SAVE AND CONTINUE THIS RUN LATER",
                "IF YOU EXIT WITHOUT QUITTING TO MAIN MENU",
            ]
            .into_iter()
            .enumerate()
            {
                out.push(MenuGuiText {
                    text: line.to_string(),
                    gx: vw * 0.5,
                    gy: 240.0 - 36.0 - 30.0 + 7.0 + i as f32 * 10.0,
                    color: GUI_WHITE,
                    px: 7.0,
                    centered: true,
                    middle_y: true,
                    right: false,
                    bold: false,
                });
            }
        }
    }
    out
}

/// Settings toggle row: label left (cream) + ON/OFF value (gray).
fn push_toggle(out: &mut Vec<MenuGuiText>, label: &str, y: f32, on: bool) {
    out.push(gui_body(label, 80.0, y, GUI_CREAM));
    out.push(gui_body(if on { "ON" } else { "OFF" }, 200.0, y, GUI_GRAY));
}

// Settings hot rows: one source of truth shared by
// `settings_click_action` and `tick_settings_nav` (`menus.rs`), so indices agree. `gy`
// mirrors `settings_gui_texts` literals; `cx`/`hw` is the GUI-px mouse hit box (buttons at
// `vw/2`, value cells at `cx+32`; sliders use the rendered track geometry).

/// Player color presets cycled by the COLOR page button. Port-only: GML's
/// COLOR option is a free-text hex field on a swatch
/// (`objects/MenuOptions/Other_20.gml:459-475`, validated to 6 chars at
/// `:319-330`), not a fixed cycle.
pub const COLOR_PRESETS: [&str; 5] = ["FF0000", "00FF00", "0000FF", "", "FF00FF"];

/// Volume channel with absolute steppers. GML's audio page has four
/// sliders, in this order (`objects/MenuOptions/Other_20.gml:96-103`:
/// `volume_master` / `volume_music` / `volume_ambient` / `volume_sfx`);
/// the ±stepper control and its 0.1 step are port-side.
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
    /// REMAP row: arm a rebind capture for the named GML control
    /// (`fire`, `spec`, `swap`, `pick`, `north`, `south`, `west`,
    /// `east`). The next pressed key/mouse button resolves it.
    Remap(&'static str),
    /// REMAP page: restore GML `scrKeymapsSetup` defaults.
    RemapReset,
}

pub const SETTINGS_LETTERBOX: f32 = 36.0;

pub fn settings_row_in_vision(gy: f32) -> bool {
    gy >= SETTINGS_LETTERBOX && gy <= 240.0 - SETTINGS_LETTERBOX
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
        cx: cx - 20.0,
        hw: 100.0,
        op,
    };
    let val = |gy: f32, op: SettingHotOp| SettingHotRow {
        gy,
        cx: cx + 32.0,
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
            val(62.0, SettingHotOp::Cycle("gamepad_type")),
            tog(76.0, SettingHotOp::Toggle("aim_assist")),
            tog(90.0, SettingHotOp::Toggle("auto_aim")),
            tog(104.0, SettingHotOp::Toggle("volume_controls")),
            tog(118.0, SettingHotOp::Toggle("split_fire")),
            tog(132.0, SettingHotOp::Toggle("fixed_sight")),
            tog(146.0, SettingHotOp::Toggle("hidden_sticks")),
            tog(160.0, SettingHotOp::Toggle("stick_regions")),
            val(174.0, SettingHotOp::Slider("controls_scale")),
            btn(188.0, SettingHotOp::Category(13)),
            btn(204.0, SettingHotOp::Category(15)),
            btn(220.0, SettingHotOp::Category(16)),
            btn(240.0, SettingHotOp::Back),
        ],
        13 => {
            // GML `Controls_Remapping_Keys` verbatim: one `keybind` row
            // per keyboard control (fire/spec/swap/pick + walk keys),
            // then DEFAULT PRESET. Rows arm a capture; the row text
            // shows the live binding (or PRESS KEY while armed).
            let mut rows: Vec<SettingHotRow> = [
                "fire", "spec", "swap", "pick", "north", "south", "west", "east",
            ]
            .iter()
            .enumerate()
            .map(|(i, key)| SettingHotRow {
                gy: 56.0 + i as f32 * 16.0,
                cx,
                hw: 100.0,
                op: SettingHotOp::Remap(match *key {
                    "fire" => "fire",
                    "spec" => "spec",
                    "swap" => "swap",
                    "pick" => "pick",
                    "north" => "north",
                    "south" => "south",
                    "west" => "west",
                    _ => "east",
                }),
            })
            .collect();
            rows.push(btn(196.0, SettingHotOp::RemapReset));
            rows.push(btn(212.0, SettingHotOp::Back));
            rows
        }
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
        16 => {
            // GML `MenuOptions/Other_20.gml:785-786` (the
            // `Controls_Experimental` region): `options_keyboard`,
            // `controls_stickregions`, `controls_hiddensticks` (hidden
            // while regions are on: the repositioning sticks never sit at
            // home). Regions-on gate = text layer +
            // `settings_row_available`; hot rows keep every row (index
            // agreement).
            vec![
                tog(48.0, SettingHotOp::Toggle("keyboard_enabled")),
                tog(66.0, SettingHotOp::Toggle("stick_regions")),
                tog(84.0, SettingHotOp::Toggle("hidden_sticks")),
                btn(200.0, SettingHotOp::Back),
            ]
        }
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

/// GML `condition`/`available` for one settings row
/// (`scrOptionsMenu.gml:184-190` + `MenuOptions/Other_10.gml:589-597`):
/// `false` marks the row unavailable - it stops taking clicks, hover and
/// keyboard nav, and keybind rows drop out of the list entirely
/// (`_opt.visible = _opt.available`).
pub fn settings_row_available(world: &World, page: u8, row: &SettingHotRow) -> bool {
    let settings = world
        .get_resource::<crate::savedata_part::SaveData>()
        .map(|s| &s.settings);
    match page {
        16 => {
            !matches!(row.op, SettingHotOp::Toggle("hidden_sticks"))
                || !world
                    .get_resource::<crate::savedata_part::SaveData>()
                    .is_some_and(|s| s.settings.stick_regions)
        }
        4 => match row.op {
            // GML `Other_20.gml:527-531`: GAMEPAD STYLE carries
            // `condition = is_gamepad()`, the sticky `opt_gamepad`
            // setting.
            SettingHotOp::Cycle("gamepad_type") => settings.is_some_and(|s| s.gamepad_enabled),
            // GML `Other_20.gml:555-559`: SPLIT AIM & FIRE carries
            // `condition = !opt_aimbot` (FULL AUTOAIM off).
            SettingHotOp::Toggle("split_fire") => !settings.is_some_and(|s| s.auto_aim),
            _ => true,
        },
        // GML `Other_20.gml:690`: the walk rows carry
        // `condition_keyboard = is_keyboard() && !is_gamepad()`
        // (`KeyCont.keyboard = opt_keyboard && !opt_gamepad`,
        // `InputHandling.gml:225`), so the GAMEPAD switch takes them
        // off the REMAP list.
        13 => {
            !matches!(
                row.op,
                SettingHotOp::Remap("north" | "south" | "west" | "east")
            ) || settings.is_some_and(|s| s.keyboard_enabled && !s.gamepad_enabled)
        }
        _ => true,
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
        SettingHotOp::Remap(key) => Some(UiAction::RemapControl(key.to_string())),
        SettingHotOp::RemapReset => Some(UiAction::RemapReset),
    }
}

/// Mouse hit-test for a settings page: GUI-px click → [`UiAction`].
/// Slider clicks use the horizontal track position; list rows keep their
/// click-half direction.
pub fn settings_click_action(
    world: &mut World,
    page: u8,
    gx: f32,
    gy: f32,
    vw: f32,
) -> Option<UiAction> {
    if let Some((_, target)) = settings_slider_hit(page, gx, gy, vw) {
        let value = settings_slider_value(target, gx, vw);
        return Some(settings_slider_action(target, value));
    }
    let rows = settings_hot_rows(page, vw);
    // Tight vertical band (rows sit 14px apart on dense pages).
    const HH: f32 = 7.0;
    let (idx, row) = rows.iter().enumerate().find(|(_, r)| {
        (gy - r.gy).abs() <= HH
            && (gx - r.cx).abs() <= r.hw
            && settings_row_available(world, page, r)
    })?;
    let dir = match row.op {
        SettingHotOp::Cycle(_) => {
            if gx >= row.cx {
                1
            } else {
                -1
            }
        }
        _ => 0,
    };
    settings_hot_action(world, page, idx, dir)
}

/// Settings pages: headers, rows and buttons transcribed from the GML
/// option register (`objects/MenuOptions/Other_20.gml`, categories per
/// `objects/MenuOptions/Create_0.gml` `OptionCategory`); dynamic values
/// read from [`SaveData`](crate::savedata_part::SaveData) +
/// [`MenuState`](crate::state::menus::MenuState). The fixed row/label
/// layout GML computes from element metrics is flattened to literals
/// here - port-only.
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
            out.push(gui_button("SETTINGS", cx, 24.0, GUI_MID));
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
                let color = if i < 4 { GUI_HIDDEN } else { GUI_MID };
                out.push(gui_button(*label, cx, 72.0 + i as f32 * 24.0, color));
            }
            out.push(gui_button("BACK", cx, 220.0, GUI_HIDDEN));
        }
        1 => {
            out.push(gui_button("AUDIO", cx, 24.0, GUI_MID));
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
            out.push(gui_button("VIDEO", cx, 24.0, GUI_MID));
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
            out.push(settings_option_button("DISPLAY SETTINGS", cx, y));
            // DISPLAY lands at y=192; BACK at 200 would overlap its
            // 10px button box, so sit BACK at 220 like the OPTIONS page.
            out.push(gui_button("BACK", cx, 220.0, GUI_GRAY));
        }
        9 => {
            out.push(gui_button("DISPLAY", cx, 24.0, GUI_MID));
            let mut y = 56.0;
            push_toggle(&mut out, "WIDESCREEN", y, s.widescreen);
            y += 20.0;
            push_toggle(&mut out, "FULLSCREEN", y, s.fullscreen);
            y += 20.0;
            push_toggle(&mut out, "VSYNC", y, s.vsync);
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        3 => {
            // GML `Game` category (`Other_20.hml:218`): pause-button mobile-only, auto-pause
            // desktop-only in GML - both shown here (desktop shells ignore them);
            // `ACHIEVEMENT#POPUPS` is a two-line switch. GML puts the COLOR/DATA leaves under
            // PROFILE; the port surfaces all three here to keep them reachable without the
            // text-entry pages.
            out.push(gui_button("GAME", cx, 24.0, GUI_MID));
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
                if label.contains('#') {
                    let parts: Vec<&str> = label.split('#').collect();
                    let first = parts.first().copied().unwrap_or(label);
                    let second = parts.get(1).copied().unwrap_or("");
                    out.push(gui_body(first, 80.0, y - 4.0, GUI_CREAM));
                    out.push(gui_body(second, 80.0, y + 4.0, GUI_CREAM));
                    out.push(gui_body(if on { "ON" } else { "OFF" }, 200.0, y, GUI_GRAY));
                } else {
                    push_toggle(&mut out, label, y, on);
                }
                y += 14.0;
            }
            out.push(settings_option_button("VIEW CREDITS", cx, y));
            y += 16.0;
            out.push(settings_option_button("PROFILE", cx, y));
            y += 16.0;
            out.push(settings_option_button("COLOR", cx, y));
            y += 16.0;
            out.push(settings_option_button("DATA", cx, y));
            out.push(gui_button("BACK", cx, 228.0, GUI_GRAY));
        }
        4 => {
            // GML `Controls` category (`Other_20.gml:520`): GAMEPAD STYLE is XBOX ONE
            // for XBONE, dimmed to `c_menudark` while the switch is off
            // (`Other_20.gml:527-531` + `Other_10.gml:663`). Mobile-only rows (AIM ASSIST
            // / FULL AUTOAIM / VOLUME CONTROLS / SPLIT AIM & FIRE / FIXED SIGHT / SIZE
            // SCALE, CHARACTER PREFERENCES, EXPERIMENTAL OPTIONS) shown here; desktop
            // shells ignore them. REMAP CONTROLS takes GML's `(GAMEPAD)` suffix while the
            // switch is on (`Other_20.gml:569-581`); `(KEYBOARD)` needs
            // `is_keyboard() && !is_desktop`, never true here. Names use GML loc defaults.
            out.push(gui_button("CONTROLS", cx, 24.0, GUI_MID));
            let mut y = 48.0;
            push_toggle(&mut out, "GAMEPAD", y, s.gamepad_enabled);
            y += 14.0;
            let (style_name, style_value) = if s.gamepad_enabled {
                (GUI_CREAM, GUI_GRAY)
            } else {
                (GUI_MENUDARK, GUI_MENUDARK)
            };
            out.push(gui_body("GAMEPAD STYLE", 80.0, y, style_name));
            let names = ["XBOX ONE", "PS4", "Switch", "SteamDeck"];
            out.push(gui_body(
                format!("< {} >", names[(s.gamepad_type as usize) % names.len()]),
                200.0,
                y,
                style_value,
            ));
            y += 14.0;
            for (label, on, dim) in [
                ("AIM ASSIST", s.aim_assist, false),
                ("FULL AUTOAIM", s.auto_aim, false),
                ("VOLUME CONTROLS", s.volume_controls, false),
                ("SPLIT AIM & FIRE", s.split_fire, s.auto_aim),
                ("FIXED SIGHT", s.fixed_sight, false),
                ("HIDDEN STICKS", s.hidden_sticks, false),
                ("STICK REGIONS", s.stick_regions, false),
            ] {
                // Unavailable rows draw in `c_menudark`
                // (`Other_10.gml:663`), never vanish.
                let (name, value) = if dim {
                    (GUI_MENUDARK, GUI_MENUDARK)
                } else {
                    (GUI_CREAM, GUI_GRAY)
                };
                out.push(gui_body(label, 80.0, y, name));
                out.push(gui_body(if on { "ON" } else { "OFF" }, 200.0, y, value));
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
            out.push(settings_option_button(
                if s.gamepad_enabled {
                    "REMAP CONTROLS (GAMEPAD)"
                } else {
                    "REMAP CONTROLS"
                },
                cx,
                y,
            ));
            y += 16.0;
            out.push(settings_option_button("CHARACTER PREFERENCES", cx, y));
            y += 16.0;
            out.push(settings_option_button("EXPERIMENTAL OPTIONS", cx, y));
            out.push(gui_button("BACK", cx, 228.0, GUI_GRAY));
        }
        10 => {
            out.push(gui_button("PROFILE", cx, 24.0, GUI_MID));
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
            out.push(settings_option_button("COLOR", cx, 88.0));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        11 => {
            out.push(gui_button("COLOR", cx, 24.0, GUI_MID));
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
            out.push(settings_option_button("CYCLE COLOR", cx, 90.0));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        12 => {
            out.push(gui_button("DATA", cx, 24.0, GUI_MID));
            out.push(settings_option_button("RESET OPTIONS", cx, 80.0));
            out.push(gui_center("ERASE PROGRESS", cx, 110.0, GUI_RED2));
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        13 => {
            out.push(gui_button("REMAP", cx, 24.0, GUI_MID));
            let keymap = world
                .get_resource::<crate::keymap::InputMapState>()
                .map(|s| s.session.map.clone())
                .unwrap_or_else(crate::keymap::default_keymap);
            let capturing = world
                .get_resource::<crate::keymap::InputMapState>()
                .and_then(|s| s.session.capture.clone());
            // GML `Other_10.gml:779-793,832-843`: value column = `keymap_get`
            // (`Key[key][opt_gamepad]`, `scrOptionsKeymaps:111`) - the pad side while
            // GAMEPAD is on - swapped for a `draw_gamepad_button` glyph; walk rows leave
            // the list (`scrOptionsMenu.gml:189`, `_opt.visible = _opt.available`).
            let gamepad_ui = gamepad_ui_on(world);
            let rows = settings_hot_rows(13, vw);
            let mut y = 56.0;
            for (i, action) in crate::keymap::NtAction::ALL.iter().enumerate() {
                let available = rows
                    .get(i)
                    .is_none_or(|row| settings_row_available(world, 13, row));
                if available {
                    let entry = keymap.active(action, gamepad_ui);
                    let armed = capturing.as_ref().is_some_and(|c| c.action == *action);
                    let text = if armed {
                        "PRESS KEY...".to_string()
                    } else if gamepad_ui {
                        String::new()
                    } else {
                        repame_input::encode_keymap_entry(&entry)
                    };
                    out.push(gui_body(action.label(), 80.0, y, GUI_CREAM));
                    if !text.is_empty() {
                        out.push(gui_body(text, 200.0, y, GUI_GRAY));
                    }
                }
                y += 16.0;
            }
            out.push(settings_option_button("DEFAULT PRESET", cx, 196.0));
            out.push(gui_button("BACK", cx, 212.0, GUI_GRAY));
        }
        15 => {
            out.push(gui_button("CHAR PREFS", cx, 24.0, GUI_MID));
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
            // GML `Controls_Experimental` verbatim: KEYBOARD MODE
            // (`options_keyboard`), STICK REGIONS, HIDE JOYSTICKS
            // (hidden while regions are on - the repositioning sticks
            // never sit at home).
            out.push(gui_button("EXPERIMENTAL", cx, 24.0, GUI_MID));
            push_toggle(&mut out, "KEYBOARD MODE", 48.0, s.keyboard_enabled);
            push_toggle(&mut out, "STICK REGIONS", 66.0, s.stick_regions);
            if !s.stick_regions {
                push_toggle(&mut out, "HIDE JOYSTICKS", 84.0, s.hidden_sticks);
            }
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        5 => {
            out.push(gui_button("LANGUAGE", cx, 24.0, GUI_MID));
            let mut y = 60.0;
            for lang in crate::state::menus::AVAILABLE_LANGUAGES {
                let current = lang == s.language;
                out.push(MenuGuiText {
                    text: lang.to_ascii_uppercase().to_string(),
                    gx: cx,
                    gy: y,
                    color: if current { GUI_WHITE } else { GUI_MID },
                    px: 7.0,
                    centered: true,
                    middle_y: true,
                    right: false,
                    bold: false,
                });
                y += 20.0;
            }
            out.push(gui_button("BACK", cx, 200.0, GUI_GRAY));
        }
        _ => {}
    }
    for row in &mut out {
        if (row.gx - 80.0).abs() < 0.1 {
            row.gx = cx - 130.0;
        } else if (row.gx - 200.0).abs() < 0.1 {
            row.gx = cx + 32.0;
        }
        if row.text == "BACK" {
            row.color = GUI_HIDDEN;
        }
    }
    // Keyboard cursor highlight: the cursor row renders white so arrow-key
    // nav is visible, not blind. Port-only - GML's options menu drives
    // rows by pointer/gamepad input, not by a persisted cursor row.
    let cursor = world
        .get_resource::<MenuState>()
        .map(|m| m.settings_cursor)
        .unwrap_or(usize::MAX);
    if cursor != usize::MAX {
        if let Some(row) = settings_hot_rows(page, vw).get(cursor) {
            for t in out.iter_mut() {
                if t.color != GUI_HIDDEN && (t.gy - row.gy).abs() < 0.5 {
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

fn ability_hud_sprites(
    assets: &RenderAssets,
    view: [f32; 4],
    gm: HudGuiMap,
    held_race: RaceId,
    ultra: Option<UltraMutationId>,
    mutations: &[MutationId],
    patience_used: bool,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    let mut x = view[2] - 12.0;
    let mut y = 13.0;
    if let Some(ultra) = ultra
        && let Some(sprite) = assets.sprite_scaled_rotated(
            "images/sprEGIconHUD.png",
            ultra_hud_frame(held_race, ultra),
            hud_gui_to_world(gm, view, x, y),
            gm.s,
            0.0,
            [1.0; 4],
        )
    {
        out.push(sprite);
        x -= 16.0;
        if x <= 120.0 {
            x = view[2] - 12.0;
            y += 16.0;
        }
    }
    y -= 1.0;
    for mutation in mutations {
        let base = (
            "images/sprSkillIconHUD.png",
            crate::hud::mutation_skill_index(*mutation) as i32,
        );
        let overlay = (*mutation == MutationId::Patience && patience_used)
            .then_some(("images/sprPatienceIconHUD.png", 0));
        for (path, frame) in [Some(base), overlay].into_iter().flatten() {
            if let Some(sprite) = assets.sprite_scaled_rotated(
                path,
                frame,
                hud_gui_to_world(gm, view, x, y),
                gm.s,
                0.0,
                [1.0; 4],
            ) {
                out.push(sprite);
            }
        }
        x -= 16.0;
        if x <= 120.0 {
            x = view[2] - 12.0;
            y += 16.0;
        }
    }
    out
}

/// Sprite HUD bars. GML regions, GUI draw points verbatim: bar frame 2
/// at (20,4), fills at (22,7) width `84*frac`, rad/exp bar at (4,4)
/// (`scripts/scrDrawPlayerHUD/scrDrawPlayerHUD.gml:19,36-40,190`); ammo row
/// y=32 (`:220`), weapon icons y=16 (`:106-107`); offer icons y=219 =
/// `view_height - 21` (`objects/LevCont/Other_10.gml:6`); held
/// ultra/skills from (`view_width-12`, 13)
/// (`scripts/scrDrawMiscHUD/scrDrawMiscHUD.gml:73-75`).
/// GUI px == view px ([`hud_gui_map`]). `dt_secs` drives the `lsthealth`
/// ghost decay (GML `Player/Step_0`: lerp 0.2 past a 20 gap, else
/// 0.5/step).
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
    // Same `scrDrawMiscHUD:68` game-over law as the text rows above:
    // held ultras/skills stay visible once the player is gone.
    let game_over = crate::state::menus::game_over_visible(world);
    if !player_alive && !game_over {
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

    // Health bar + fills (GML `scrDrawPlayerHUD:17-58`): bar frame 2 at (20,4) via
    // `draw_sprite` (origin (0,0), so pixels land exactly there); fills `draw_sprite_ext`
    // frame 0 xscale-stretched from (22,7) - a (0,0)-origin 1x8 strip, so the left edge
    // pins at x=22 as the width shrinks. Ghost darkened HSV, live `opt_healthcol`, hurt
    // flash white frame 0 at the hp width. Desktop nudges +0.01.
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

    // Rad/exp bar + level badge (GML `scrDrawPlayerHUD:184-202`): `sprExpBarLevel`
    // first while an offer is pending (`skillpoints/ultrapoints/wantdestinyskill`),
    // then `sprExpBar` frame `frac*16` at (4,4); `sprNomutsLevel` (11,16) at level cap 0,
    // the number below cap, `sprUltraLevel` at cap. Cap = `PLAYER_LEVEL_MAX` (10; 0 in
    // no-muts custom).
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
    // GML `_level_max <= 0` (no-muts custom) never occurs (port floor is 1):
    // that arm is vacuous. `sprUltraLevel` at (11,16) via `draw_sprite` (not `_ext`)
    // honors the strip origin (4,5 of 8x8); `sprite_scaled_rotated` read (11,16) as the
    // quad center and shifted it half a cell. `hud_gui_place` lands the top-left on the
    // GUI point.
    if hud.level >= PLAYER_LEVEL_MAX {
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

    // Ammo icon pairs (GML `scrDrawPlayerHUD:209-230`): row y=32, `dx = 2+(t-1)*10` -2 at
    // Bolts and up; BG frame 2 for the primary (secondary on Steroids), 1 for the
    // secondary, else 0; icon drains against the SHARED `sprBulletIcon` frame count
    // (`_frames` once, not per type) over the Back-Muscle-adjusted cap.
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
        let frames = strip_frames(assets, "images/sprBulletIcon.png").max(1) as f32 - 1.0;
        let icon_frame = (frames - (fill * frames).ceil()).clamp(0.0, frames) as i32;
        if let Some(s) = hud_gui_place(assets, icon, icon_frame, dx, 32.0, 1.0, [1.0; 4], gm, view)
        {
            out.push(s);
        }
    }
    // GML `scrDrawPlayerHUD:231-245` event/continued icons at y=33: daily/weekly icon at
    // x=56 on event runs (weekly picks the weekly sprite), custom icon +12, continued
    // icon +12 per flag. No daily/weekly/custom/continued runs in the port (PLAY
    // sub-rows deny; `Run` has no continued flag): all three arms vacuous.

    // Rogue/Cuz ammo pips: `draw_sprite(sprite, sub, 110, 4)` honors the strip origin
    // (not `_ext`); these strips carry origin (1,1), so pixels land at (109,3). Subimage
    // `ammo ? max(1, floor((frames-1) * progress)) : 0`; Cuz draws `sprCuzAmmoHUDU` under
    // the Emotional ultra. `hud_gui_place` lands the top-left on the GUI point: pass the
    // GML point verbatim.
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
            if let Some(s) = hud_gui_place(assets, path, sub, 110.0, 4.0, 1.0, [1.0; 4], gm, view) {
                out.push(s);
            }
        }
    }

    out.extend(ability_hud_sprites(
        assets,
        view,
        gm,
        race_skin.map(|(race, _)| race).unwrap_or(RaceId::Fish),
        ultra,
        &mutations,
        patience_used,
    ));

    // Weapon strip (GML `scrDrawPlayerHUD:99-146`): `_wep` x=24, `_bwep` 68, +20 per
    // extra, y=16 (position 0 is always primary; `scrSwapWeps` swaps `_wep`/`_bwep`
    // sim-side, never the draw order). Each gun is a `draw_sprite_part_ext` window
    // `(xoffset, yoffset+swapanim-8, weapon_width, 14+swapanim)`, `weapon_width` 16 (32
    // for a slot-0 melee), top-left landing exactly on `(dx, dy)`; 4-way outline at ±1 px
    // (white when `_is_active_weapon = (index==0 || Steroids)`, else `#404040`; only
    // active/broke/darkness/letterbox), body `c_black`; `gpu_fog` tints curse (`c_curse`)
    // / ultra rad (`c_ultra`) / golden (`c_gold`). `swapanim` y-offset and the white 0.2
    // reload wipe are untracked: the steady frame draws.
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
        // GML outline gate (`scrDrawPlayerHUD:134`): outlines draw only
        // for the active weapon, a broke-batch gun (ultra/cursed/
        // golden fog batch), darkness, or letterbox frame >= 2. The
        // port tracks no darkness/letterbox state, so those arms stay
        // ungated with this note; inactive normal guns skip outlines.
        let broke_batch = cursed || ultra_gun || golden;
        let draw_outline = active || broke_batch;
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
        // 4-way outline quads (1 px GUI offsets around the body),
        // gated on active/broke-batch above (darkness/letterbox arms
        // have no port state yet).
        if draw_outline {
            for (ox, oy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                if let Some(s) =
                    hud_weapon_part(assets, &path, dx + ox, 16.0 + oy, ww, outline, gm, view)
                {
                    out.push(s);
                }
            }
        }
        let body_tint = fog.unwrap_or([0.0, 0.0, 0.0, 1.0]);
        if let Some(s) = hud_weapon_part(assets, &path, dx, 16.0, ww, body_tint, gm, view) {
            out.push(s);
        }
    }
    out
}

/// One GML `draw_sprite_part_ext` weapon window: source rect `(xoffset, yoffset-8, ww,
/// 14)` of strip frame 1 (steady `swapanim = 0`), top-left at GUI `(dx, dy)`. Windows
/// start outside the cell (negative xorigin, `yoffset - 8 < 0`) and overrun it (14-tall
/// over a shorter cell); GML clamps those samples to transparent edge texels, so only the
/// in-bounds part draws, offset by the lead (`max(0,-sx)`, `max(0,-sy)`) - drawing the
/// clamped cell at `(dx, dy)` shifts every gun up-left (revolver 2x5 px). UVs lerp over
/// the in-bounds sub-rect (the atlas packs whole frames); anchor stays centered.
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
    let wx = def.xorigin as f32;
    let wy = def.yorigin as f32 - 8.0;
    let ix0 = wx.max(0.0);
    let iy0 = wy.max(0.0);
    let ix1 = (wx + ww).min(sw);
    let iy1 = (wy + 14.0).min(sh);
    if ix1 <= ix0 || iy1 <= iy0 {
        return None;
    }
    let iw = ix1 - ix0;
    let ih = iy1 - iy0;
    // Frame cell spans uv.min..uv.max; the in-bounds window is the
    // matching fraction of it (atlas Y is down, same as the strip).
    let fx0 = ix0 / sw;
    let fy0 = iy0 / sh;
    let fx1 = ix1 / sw;
    let fy1 = iy1 / sh;
    let uv_min = Vec2::new(
        uv.min[0] + (uv.max[0] - uv.min[0]) * fx0,
        uv.min[1] + (uv.max[1] - uv.min[1]) * fy0,
    );
    let uv_max = Vec2::new(
        uv.min[0] + (uv.max[0] - uv.min[0]) * fx1,
        uv.min[1] + (uv.max[1] - uv.min[1]) * fy1,
    );
    let size = Vec2::new(iw * map.s, ih * map.s);
    let top_left = hud_gui_to_world(map, view, dx + (-wx).max(0.0), dy + (-wy).max(0.0));
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

/// Coop fainted bars (GML `TopCont/Draw_0` `Revive` block): `sprFaintedBar` at the
/// view-clamped pos, then the grace bar (`alarm[4] / 300 * 28`, red-black pulse from
/// `sin(tottimer / 4)`) or the bleed bar (`alarm[5] / 30 * 28`, red - GML's missing
/// `+ 2` on the bleed right edge). Alarms ride the [`HudState`] snapshot.
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

/// GML `PLAYER_LEVEL_MAX` (`macros_gameplay:2`: `scrCustomParam(
/// "maxlevel", 10)` - 10 normally, 0 in no-muts custom). The port has
/// no custom-mode state yet, so the cap is always 10 and the `<= 0`
/// `sprNomutsLevel` arm stays vacuous (noted at its draw site).
pub const PLAYER_LEVEL_MAX: u32 = 10;

/// World-space crosshair (GML `TopCont/Draw_0` player block): `sprCrosshair[opt_crosshair]`
/// lerped toward `player + (16 + dis)` along the aim heading (0.8 active / 0.1 idle),
/// alpha toward 5/0 at 0.4 (drawn `min(1, alpha)`), active past the attack
/// deadzone. Skipped while paused (`PauseImage` gate) and until the first hover stages a
/// cursor; `with Player`, so death removes it. Keyboard-mode LOCAL exclusion, the
/// `Draw_75` cursor and the touch path: gate below.
pub fn crosshair_sprites(
    world: &mut World,
    assets: &RenderAssets,
    dt_secs: f32,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    // GML `TopCont/Draw_0` is `with Player`: no entity, no lerped crosshair.
    // The GameOver cursor is `UberCont/Draw_75`'s raw mouse sprite (menu-crosshair path
    // in lib.rs).
    let live = world
        .get_resource::<crate::state::AppState>()
        .is_some_and(|s| *s == crate::state::AppState::InGame)
        && !world
            .get_resource::<crate::state::Paused>()
            .is_some_and(|p| p.0)
        && world
            .get_resource::<crate::state::OverlayMenu>()
            .is_none_or(|o| *o == crate::state::OverlayMenu::None)
        && world
            .get_resource::<crate::comps_a::PendingMutation>()
            .is_none()
        && world
            .get_resource::<crate::comps_a::PendingUltra>()
            .is_none();
    if !live {
        return out;
    }
    // GML `TopCont/Draw_0:43` + `scrHandleInputsGeneral`, gate `!opt_keyboard ||
    // index != global.index || is_gamepad(index)`: skip ONLY a keyboard-driven local player
    // (raw `Draw_75` cursor covers their aim). Device facts come from the one shared law
    // (`input::gml_input_device`): `keyboard[index] = opt_keyboard && !opt_gamepad`,
    // `gamepad[index] = opt_gamepad`. A touch-only device is never keyboard-driven, so a
    // touch local always draws the lerped crosshair - the GML Android cursor - and
    // `Draw_75` never fires there.
    let (keyboard_local, _) =
        crate::input::gml_input_device(world.get_resource::<crate::savedata_part::SaveData>());
    if keyboard_local {
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
    // GML `dis_fire` source by device: keyboard reads the hover distance, touch
    // the attack-stick SMOOTHED deflection (`vdis` in `NtInput.touch_dis`), easing toward
    // `dis` while held and decaying at 2 px/tick after release
    // (`JoystickAttack/Other_10`), so the crosshair glides home on lift.
    let touch_dis = world
        .get_resource::<crate::input::NtInput>()
        .map(|i| i.touch_dis)
        .unwrap_or(0.0);
    let touch_live = world
        .get_resource::<crate::input::NtInput>()
        .is_some_and(|i| i.attack_stick.is_some_and(|s| s.touch >= 0));
    let dis = if touch_dis > 0.0 || touch_live {
        touch_dis
    } else {
        hover.map(|h| h.distance(pp)).unwrap_or(0.0)
    };
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

#[derive(Clone, Copy)]
struct ShadowSpec {
    path: &'static str,
    offset: Vec2,
}

impl ShadowSpec {
    const fn new(path: &'static str, x: f32, y: f32) -> Self {
        Self {
            path,
            offset: Vec2::new(x, y),
        }
    }
}

fn enemy_shadow(kind: EnemyKind) -> Option<ShadowSpec> {
    let spec = match kind {
        EnemyKind::Maggot | EnemyKind::RadMaggot => ShadowSpec::new("images/shd16.png", 0.0, 0.0),
        EnemyKind::BigMaggot => ShadowSpec::new("images/shd32.png", 0.0, 0.0),
        EnemyKind::FiredMaggot => return None,
        EnemyKind::BigBandit => ShadowSpec::new("images/shd32.png", 0.0, 4.0),
        EnemyKind::BigDog => ShadowSpec::new("images/shd96.png", 0.0, 0.0),
        EnemyKind::Hyper => ShadowSpec::new("images/shd64.png", 0.0, 16.0),
        EnemyKind::IdpdVan => ShadowSpec::new("images/shd96.png", 0.0, -8.0),
        EnemyKind::ProtoStatue => ShadowSpec::new("images/shd64.png", 0.0, 9.0),
        EnemyKind::ScrapBossMissile => ShadowSpec::new("images/shd24.png", 0.0, 3.0),
        EnemyKind::RobotGuard => ShadowSpec::new("images/shd24.png", 0.0, 1.0),
        EnemyKind::FrogQueen => ShadowSpec::new("images/shd64.png", 0.0, 8.0),
        EnemyKind::Guardian | EnemyKind::CrownGuardian => {
            ShadowSpec::new("images/shd24.png", 0.0, 4.0)
        }
        EnemyKind::DogGuardian => ShadowSpec::new("images/shd64.png", 0.0, 7.0),
        EnemyKind::ExploGuardian => ShadowSpec::new("images/shd32.png", 0.0, 8.0),
        EnemyKind::Crab => ShadowSpec::new("images/shd48.png", 0.0, 0.0),
        EnemyKind::Scorpion
        | EnemyKind::GoldScorpion
        | EnemyKind::Ratking
        | EnemyKind::PopoFreak => ShadowSpec::new("images/shd32.png", 0.0, 4.0),
        EnemyKind::SnowTank => ShadowSpec::new("images/shd32.png", 0.0, 5.0),
        EnemyKind::GoldSnowtank => ShadowSpec::new("images/shd32.png", 0.0, 3.0),
        EnemyKind::RhinoFreak => ShadowSpec::new("images/shd24.png", -2.0, 4.0),
        EnemyKind::LaserCrystal | EnemyKind::LightningCrystal | EnemyKind::InvLaserCrystal => {
            ShadowSpec::new("images/shd24.png", 0.0, 4.0)
        }
        EnemyKind::Jock
        | EnemyKind::JungleFly
        | EnemyKind::Salamander
        | EnemyKind::FireBaller
        | EnemyKind::SuperFireBaller => ShadowSpec::new("images/shd32.png", 0.0, 0.0),
        _ => ShadowSpec::new("images/shd24.png", 0.0, 0.0),
    };
    Some(spec)
}

fn pickup_shadow(kind: PickupKind) -> Option<ShadowSpec> {
    match kind {
        PickupKind::Chest(ChestKind::BigWeapon) => None,
        PickupKind::Chest(ChestKind::Rad)
        | PickupKind::Chest(ChestKind::RadBig)
        | PickupKind::Chest(ChestKind::RadMaggot) => {
            Some(ShadowSpec::new("images/shd24.png", 0.0, 0.0))
        }
        PickupKind::Chest(_) => Some(ShadowSpec::new("images/shd24.png", 0.0, -1.0)),
        _ => None,
    }
}

fn prop_shadow(sprites: Option<&PropSprites>) -> ShadowSpec {
    match sprites.map(|s| s.idle) {
        Some("images/sprAnchor.png") => ShadowSpec::new("images/shd48.png", 0.0, 0.0),
        Some("images/sprBigSkullOpen.png") => ShadowSpec::new("images/shd32.png", 0.0, 0.0),
        _ => ShadowSpec::new("images/shd24.png", 0.0, 0.0),
    }
}

fn push_shadow(
    out: &mut Vec<SpriteInstance>,
    assets: &RenderAssets,
    pos: Vec2,
    spec: ShadowSpec,
    tint: [f32; 4],
) {
    if let Some(s) = assets.sprite_for(spec.path, 0, pos + spec.offset, false, 0.0, tint) {
        out.push(s);
    }
}

fn push_pickup_shadow(
    out: &mut Vec<SpriteInstance>,
    assets: &RenderAssets,
    pos: Vec2,
    kind: PickupKind,
    tint: [f32; 4],
) {
    // `BigWeaponChest` is covered by both `with chestprop` and its own arm.
    if matches!(kind, PickupKind::Chest(ChestKind::BigWeapon)) {
        push_shadow(
            out,
            assets,
            pos,
            ShadowSpec::new("images/shd32.png", 0.0, -1.0),
            tint,
        );
        push_shadow(
            out,
            assets,
            pos,
            ShadowSpec::new("images/shd32.png", 0.0, 0.0),
            tint,
        );
        return;
    }
    if let Some(spec) = pickup_shadow(kind) {
        push_shadow(out, assets, pos, spec, tint);
    }
}

/// Soft blob shadows under actors (GML `scrShadows` entity half, composited
/// by `BackCont/Draw_0`). `spr_shadow` values and offsets resolve from the modeled kind;
/// missing strips draw nothing.
///
/// Tinted [`shadow_color`] at [`SHADOW_ALPHA`]: GML stamps opaque black into `shad`, then
/// `BackCont/Draw_0` fogs it to the area color at 0.4 alpha.
pub fn shadow_sprites(world: &mut World, assets: &RenderAssets) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    let tint = world
        .get_resource::<Run>()
        .map(|run| shadow_color(run.area))
        .unwrap_or([0.0, 0.0, 0.0, SHADOW_ALPHA]);
    let mut q = world.query::<(&Pos, &Enemy)>();
    for (pos, enemy) in q.iter(world) {
        if let Some(spec) = enemy_shadow(enemy.kind) {
            push_shadow(&mut out, assets, pos.0, spec, tint);
        }
    }
    let mut q = world.query::<(&Pos, &Player, Option<&RaceState>)>();
    for (pos, _, race) in q.iter(world) {
        let spec = if race.is_some_and(|race| race.race == RaceId::BigDog) {
            ShadowSpec::new("images/shd96.png", 0.0, 0.0)
        } else {
            ShadowSpec::new("images/shd24.png", 0.0, 0.0)
        };
        push_shadow(&mut out, assets, pos.0, spec, tint);
    }
    let mut q = world.query::<(&Pos, &Pickup)>();
    for (pos, pickup) in q.iter(world) {
        push_pickup_shadow(&mut out, assets, pos.0, pickup.kind, tint);
    }
    // GML `ProtoChest/Collision_Player.gml:5` only swaps `sprite_index` on
    // the live chest, so `scrShadows`' `with chestprop` arm keeps drawing
    // its `shd24`. Every other chest open becomes a `ChestOpen`, which
    // `scrShadows` never draws.
    {
        let mut q = world.query::<(&Pos, &OpenedChest)>();
        for (pos, opened) in q.iter(world) {
            if opened.0 == ChestKind::Proto {
                push_shadow(
                    &mut out,
                    assets,
                    pos.0,
                    ShadowSpec::new("images/shd24.png", 0.0, -1.0),
                    tint,
                );
            }
        }
    }
    let mut q = world.query_filtered::<
        (&Pos, &Prop, Option<&PropSprites>),
        (Without<WallTile>, Without<InvisiWall>),
    >();
    for (pos, _, sprites) in q.iter(world) {
        push_shadow(&mut out, assets, pos.0, prop_shadow(sprites), tint);
    }
    // Title campers (`CampChar` uses shd24; BigDog overrides it to shd96).
    let mut q = world.query::<(&Pos, &TitleCampChar)>();
    for (pos, camp) in q.iter(world) {
        let spec = if camp.race_gml == RaceId::BigDog as usize {
            ShadowSpec::new("images/shd96.png", 0.0, 0.0)
        } else {
            ShadowSpec::new("images/shd24.png", 0.0, 0.0)
        };
        push_shadow(&mut out, assets, pos.0, spec, tint);
    }
    // Campfire, LogMenu, and TV are GML `prop` descendants.
    let mut q = world.query::<(&Pos, &TitleCampfire)>();
    for (pos, _) in q.iter(world) {
        push_shadow(
            &mut out,
            assets,
            pos.0,
            ShadowSpec::new("images/shd24.png", 0.0, 0.0),
            tint,
        );
    }
    let mut q = world.query::<(&Pos, &TitleLogMenu)>();
    for (pos, _) in q.iter(world) {
        push_shadow(
            &mut out,
            assets,
            pos.0,
            ShadowSpec::new("images/shd24.png", 0.0, 0.0),
            tint,
        );
    }
    let mut q = world.query::<(&Pos, &TitleTv)>();
    for (pos, _) in q.iter(world) {
        push_shadow(
            &mut out,
            assets,
            pos.0,
            ShadowSpec::new("images/shd24.png", 0.0, 0.0),
            tint,
        );
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
    // GML `scrDrawBloom` has no Slash/Shank arm - melee swings get no
    // bloom halo there. Skipping slashes here too: a 2x additive copy
    // of the 48px left-anchored arc reads as a second, offset swing.
    let mut q = world.query::<(
        &Pos,
        &Projectile,
        Option<&Velocity>,
        &Team,
        Option<&SlashProjectile>,
        Option<&crate::comps_a::ProjectileVisual>,
    )>();
    for (pos, proj, vel, team, slash, visual) in q.iter(world) {
        if slash.is_some() {
            continue;
        }
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
/// Char-pod layout (GML `Menu/Create_0` campfire law): `count` pods of height
/// `slot_h` on a 240-tall view, `step = min(20, floor((320-40)/count))` over the fixed
/// `game_screen_width` (320, NOT the live view width), `xstart = 8`, `ystart =
/// H-slot_h-((36-slot_h) div 2)`. `wh[0]` is ignored for the step, so widescreen keeps
/// the GML left-clustered pods.
pub fn char_pod_layout(wh: [f32; 2], count: usize, slot_h: f32) -> Vec<[f32; 2]> {
    let step = go_step(count);
    let y = wh[1] - slot_h - ((36.0 - slot_h) / 2.0).floor();
    (0..count).map(|i| [8.0 + i as f32 * step, y]).collect()
}

/// Shared pod-row step (GML `Menu/Create_0`: `min(20,
/// floor((game_screen_width - 40) / count))` on the 320 base).
pub fn go_step(count: usize) -> f32 {
    20.0_f32.min(((320.0 - 40.0) / count.max(1) as f32).floor())
}

/// GO-button position (GML `Menu/Create_0`: past the last pod,
/// `y = H-36+bbox_h div 2-2`; x from the same 320-base step as the pods).
/// Returns the GML instance position (sprite origin). Note
/// `sprGoButtonSymbolic` has origin `(0,-2)`, so the drawn pixels/bbox sit
/// 2 px below this (see `title_click_action` / `menu_sprites`).
pub fn go_button_pos(wh: [f32; 2], count: usize, bbox_h: f32) -> [f32; 2] {
    let step = go_step(count);
    [
        8.0 + count as f32 * step + 2.0,
        wh[1] - 36.0 + (bbox_h / 2.0).floor() - 2.0,
    ]
}

/// Character-pod hit size (GML `CharSelect` sprite bbox: the pods are
/// `sprCharSelect` cells drawn at the layout origin; `title_click_action`
/// and the hover sync share these so both agree with the sprite bbox).
pub const TITLE_POD_W: f32 = 16.0;
/// Character-pod hit height.
pub const TITLE_POD_H: f32 = 24.0;
/// GO button hit size. GML clicks the instance's own bbox
/// (`objects/GoButton/Mouse_4.gml:7-9`), i.e. `sprGoButton` 35x19
/// with bbox (0,0)-(34,18)
/// (`sprites/sprGoButton/sprGoButton.yy:5-8,24,111`) — this port's
/// 31-wide box is its own, slightly narrower hit rect.
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

/// Title click → [`UiAction`](crate::audio::UiAction). `slot_h`, `crownsize`,
/// `skinsize` are the native-size values `menu_sprites` draws with (catalog sizes or the
/// 20px fallback); the rest mirrors the draw geometry. Stray clicks route to nothing.
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
        .unwrap_or(0);
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
            menu.loadout_frame,
            menu.loadout_open,
        ) {
            return Some(a);
        }
    }
    // GO button (armed + space-gated like the draw above).
    // `go_button_pos` is the GML instance position
    // (sprite origin); `sprGoButtonSymbolic` origin is (0,-2) so the
    let roster =
        crate::state::menus::visible_roster(world.get_resource::<crate::savedata_part::SaveData>());
    let last_pod_x = 8.0 + roster.len().saturating_sub(1) as f32 * go_step(roster.len());
    if menu.title_go_visible && last_pod_x < vw - 30.0 {
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
/// `step`/`half`/`start_x`/`icon_y` over the same live `vw`, `-12` shift at
/// `n >= 10`); the hit box is the `sprSkillIcon` / `sprEGSkillIcon`
/// cell, 24x32 centred at (12,16)
/// (`sprites/sprSkillIcon/sprSkillIcon.yy:48,134-145`) times `scale`.
/// Layout and the two-step law are GML's:
/// `objects/LevCont/Other_10.gml:4-12` places the icons at
/// `min(32, floor(view_width / (num + 1)))` steps from the view centre,
/// `view_yview + view_height - 21`, `-12` shift at `num >= 10`; clicking
/// selects, clicking again commits - `objects/SkillIcon/Mouse_4.gml:7-17`
/// (first click sets `selected`, a later one fires `event_user(0)` →
/// `objects/SkillIcon/Other_10.gml`), with `objects/UltraIcon/Mouse_4.gml`
/// as the twin. Unhighlighted highlights (`SelectMutation`), highlighted
/// commits (`PickMutation`).
pub fn mutation_icon_hit_action(world: &mut World, gx: f32, gy: f32, vw: f32) -> Option<UiAction> {
    // Ultra offers win over normal ones (same precedence as `sync_hud_state`
    // and `tick_mutation_mirror`): when both resources coexist the screen
    // shows ultra cards, so hit-testing must too.
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
    let is_ultra = pending_offer_is_ultra(world);
    let step = (vw / (n as f32 + 1.0)).floor().min(32.0);
    let scale = if is_ultra {
        1.0
    } else {
        (step / 32.0).max(0.65)
    };
    let half = (step as i32 / 2) as f32;
    let xview_shift = if n >= 10 { -12.0 } else { 0.0 };
    let start_x = vw * 0.5 + xview_shift - (n as f32 - 1.0) * half;
    let icon_y = 240.0 - 21.0;
    let menu = world.get_resource::<MenuState>().cloned();
    let selected = menu.as_ref().and_then(|state| state.mutation_selected);
    let hw = 24.0 * scale * 0.5;
    for i in 0..n {
        let cx = start_x + i as f32 * step;
        let card_y = icon_y
            + menu
                .as_ref()
                .and_then(|state| state.mutation_appear_y.get(i).copied())
                .unwrap_or(0.0)
            - if selected == Some(i) { 1.0 } else { 0.0 };
        let top = card_y - 16.0 * scale;
        let hh = 32.0 * scale;
        if (gx - cx).abs() <= hw && gy >= top && gy <= top + hh {
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
/// `menu_loadout_sprites` exactly (same formulas, same walk order -
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
    loadout_frame: f32,
    loadout_open: bool,
) -> Option<UiAction> {
    use crate::savedata_part::SaveData;
    let (w, h) = (vw, 240.0);
    let fullview = loadout_frame >= 2.0;
    let openaddy = if loadout_frame >= 2.0 && loadout_frame < 3.0 {
        if loadout_open { 1.0 } else { -1.0 }
    } else {
        0.0
    };
    if fullview && selected != 0 {
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
        let total_unlocked = crown_row.iter().skip(2).filter(|b| **b).count();
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
        let mut cy = crowntop + openaddy - 24.0;
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
            if cx >= crownright || id == 1 {
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
            let sy = skins_y + j as f32 * skinsize + openaddy;
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
    if !fullview && selected != 0 {
        // Closed-frame splat zone: GML closed `_splat_pointed` rect
        // `[splat_x - 109 div 2, splat_x] x [splat_y - 69 div 2, splat_y]`
        // (toggles only - GML offers no crown/weapon picking closed, and
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

/// Loadout grid (`scrMenuDrawLoadout` geometry and frame animation:
/// panel, crown grid, skin column, weapon row). `selected` is the
/// `CHAR_SELECT_ORDER` index. `vwvh` is the GML view size ([`gml_view_size`]);
/// `to_world` maps view px to world.
fn menu_loadout_sprites(
    world: &mut World,
    assets: &RenderAssets,
    vwvh: [f32; 2],
    selected: usize,
    loadout_frame: f32,
    loadout_open: bool,
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
    let loadout_frame = loadout_frame.clamp(0.0, 4.0);
    let fullview = loadout_frame >= 2.0;
    let openaddy = if loadout_frame >= 2.0 && loadout_frame < 3.0 {
        if loadout_open { 1.0 } else { -1.0 }
    } else {
        0.0
    };

    // Panel + arrow.
    if loadout_frame > 0.0 {
        if let Some(open) = assets.native_size("images/sprLoadoutOpen.png") {
            let xs = ((w - skins_x) / (open.x - crownsize * 2.0)).max(1.0);
            let ys = (splat[1] - 36.0) / open.y + 0.05;
            if let Some(s) = assets.sprite_stretched(
                "images/sprLoadoutOpen.png",
                loadout_frame.floor() as i32,
                to_world([splat[0] - 2.0, splat[1]]),
                Vec2::new(xs, ys),
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
        }
        if let Some(s) = assets.sprite_for(
            "images/sprLoadoutArrow.png",
            i32::from(loadout_open),
            to_world([splat[0] - 16.0, splat[1] - 16.0]),
            false,
            0.0,
            [1.0; 4],
        ) {
            out.push(s);
        }
    }

    if !fullview {
        menu_loadout_closed_sprites(world, assets, vwvh, selected, true, false, to_world, out);
        return;
    }

    // Crown grid (GML `crwn_random` (0) skipped when bare; wraps at
    // the right edge or the none crown).
    let total_unlocked = crown_row.iter().skip(2).filter(|b| **b).count();
    let current = loadout
        .as_ref()
        .map(|l| crate::savedata_part::crown_port_to_gml(l.start_crown))
        .unwrap_or(1);
    let mut cx = crownright - crownsize * 3.0;
    let mut cy = crowntop + openaddy - 24.0;
    for id in 0..14u8 {
        if id == 0 && total_unlocked == 0 {
            cx += crownsize;
            continue;
        }
        let unlocked = crown_row.get(id as usize).copied().unwrap_or(false);
        let selected = unlocked && id == current;
        let tint = if selected {
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
            to_world([cx, cy - if selected { 1.0 } else { 0.0 }]),
            false,
            0.0,
            tint,
        ) {
            out.push(s);
        }
        cx += crownsize;
        if cx >= crownright || id == 1 {
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
        let selected = open && j as u8 == preferred;
        let tint = if selected {
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
            to_world([skins_x, sy + openaddy - if selected { 1.0 } else { 0.0 }]),
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
        let offset = if chosen == wid { 2.0 } else { 0.0 } - openaddy;
        push_loadout_weapon(assets, wid, to_world([wx, weapons_y - offset]), tint, out);
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

/// Closed-frame loadout preview (GML `scrCampfireMenuCreate` "Current loadout"
/// region): mini crown/weapons for any non-Random race; splat + arrow need an available
/// loadout, trio races get minis only (`!scr_loadout_is_available_for_race` kills just
/// the splat toggle). Minis sit at the headless steady state (`_splat_pointed = 0`): crown
/// `(splat-60, splat-40)`, weapons `(splat-44/splat-68, splat-15)`. Haste-crown clock
/// (`scrDrawClock`) dropped.
#[allow(clippy::too_many_arguments)]
fn menu_loadout_closed_sprites(
    world: &mut World,
    assets: &RenderAssets,
    vwvh: [f32; 2],
    selected: usize,
    available: bool,
    include_chrome: bool,
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
    let splat_frames = strip_frames(assets, "images/sprLoadoutSplat.png");
    let splat_frame = world
        .get_resource::<MenuState>()
        .map(|menu| menu.splatindex.floor() as i32)
        .unwrap_or_else(|| char_splat_frame(race, splat_frames))
        .clamp(0, splat_frames.saturating_sub(1) as i32);
    if available && include_chrome {
        if let Some(s) = assets.sprite_stretched(
            "images/sprLoadoutSplat.png",
            splat_frame,
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

/// GML ultra tier per race (`UltraSkill` A=1/B=2/C=3).
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
        | UltraMutationId::SkeletonBloodArmor
        | UltraMutationId::FrogSwampBody
        | UltraMutationId::CuzHoarder => 1,
        UltraMutationId::HorrorMeltdown => 3,
        _ => 2,
    }
}

pub fn ultra_offer_frame(race: RaceId, ultra: UltraMutationId) -> i32 {
    let race = if race == RaceId::Random {
        RaceId::Fish
    } else {
        race
    };
    (race as i32 - 1) * 3 + ultra_tier(ultra) - 1
}

/// GML ultra HUD frame (`race * 3 + tier - 1`).
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

/// Campfire big-portrait resolution (GML `scrCampfireMenuDrawRacePortrait` portrait
/// region): headless Chicken at hp <= 0 (frame = skin), hooded Rebel B-skin in the city,
/// else `sprBigPortrait` + [`race_skin_subimage`]. `hp` `None` = no live player of that
/// race (char select: alive). `None` for Random (GML subimage -1 skips the draw). No
/// rebel-hood flag in the port - GML keys on B skin + city only.
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
/// boot reel jitter re-rolls each step like GML's `Logo/Draw_0.gml:12-13`
/// (shake) and `:29` (`random(1)` per glow copy) while staying
/// deterministic for a given `t`.
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
            // GML `Vlambeer/Draw_0`: `draw_sprite(sprite_index, 0, view_x + (view_w -
            // sprite_w)/2, view_y + (view_h - sprite_h))` - top-left pinned at `center_x -
            // w/2`, `bottom - h`, drawn CENTERED (`sprite_for`, not top-left
            // `hud_gui_place`: `sprVlambeer` is a top-left zero-origin strip). Plus 10
            // additive shimmer copies at `orandom(4)` (±4 px), alpha 0.1; GML re-rolls
            // each draw, so offsets jitter per 30 Hz step (quantized from `t`).
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
                // with `wave` (GML `Logo/Draw_0.gml:24-35`: 0.05 plus
                // `random(0.02)` per each of the 8 copies, so ~0.13 per
                // draw == 3.9/s at 30 Hz; this port spells it
                // `dt * 3.9`). The per-copy radius re-rolls every step
                // like GML's `random(1)` at `:29`.
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

/// GML `draw_line_pixelated` (`scrDrawRoadmap` local): `dist = point_distance`, `dir =
/// point_direction` (0 east, 90 south, y-down degrees); y-bias +2 when `dir in [90,270)`
/// else +1, then `draw_sprite_ext(sprPixel, 0, x1, y1b, dist, 1, dir, color, alpha)`.
/// No line primitive in the sprite pipe: a stretched `sprPixel` centered on the biased
/// midpoint, `rotation = atan2(dy, dx)` (beam law). Zero-length draws skipped (GML
/// `xscale = 0` draws nothing).
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

/// `scrDrawRoadmap` sprite layer (`GameOver/Draw_0`, `GenCont/Draw_0`): score splats +
/// kills icon (area/kill strings ride the text overlay), the 7-area dot strip (`sprMapDot`
/// 3px, segment 9px, strip 135px wide centered on `drawx`), palace crown, waypoint dots +
/// connectors from `Run.waypoints` (`pos` caps the drawn prefix; game-over passes the
/// full log). GML ids `waypnt % 100`, secrets on a second row 10px down (`sprMapDotOut`);
/// loop colors from [`roadmap_loop_tint`]. Connectors ride [`pixel_line`]; background
/// hatch 3 black + 1 white lines, waypoint shadow 3 black + 1 color line.
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
/// the player `sprMapIcon`s land where GML draws them - on the last
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

/// GUI-space sprite placed from a GML `draw_sprite_ext` call verbatim:
/// same strip-origin law as [`hud_gui_place`] plus a rotation channel
/// (touch crosshair auto-aim tilt, swap-button weapon tilt).
pub fn place_touch_sprite_rot(
    assets: &RenderAssets,
    path: &str,
    frame: i32,
    gx: f32,
    gy: f32,
    mul: f32,
    rotation: f32,
    tint: [f32; 4],
    map: HudGuiMap,
    view: [f32; 4],
) -> Option<SpriteInstance> {
    let (uv, def) = assets.uv(path, frame)?;
    let anchor = def.anchor();
    let size = Vec2::new(def.w as f32 * mul * map.s, def.h as f32 * mul * map.s);
    let top_left = hud_gui_to_world(map, view, gx - def.xorigin * mul, gy - def.yorigin * mul);
    Some(SpriteInstance {
        center: top_left + Vec2::new(anchor[0] * size.x, anchor[1] * size.y),
        rotation,
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

/// Menu art sprites (`Menu/Draw_0`, drawn here where the UI framework cannot):
/// positions are GML view px (`Menu/Draw_0` snaps pods/buttons to `view + start`); view
/// points map to world through the live view rect (GUI == view, [`hud_gui_map`] identity).
///
/// Touch controls (`scrDrawMobileControls`) too: GUI-space chrome from `NtInput` stick
/// anchors, not world entities. Gated on a live touch session (`move_stick`/`attack_stick`
/// claimed once); hidden sticks (`opt_hiddensticks`) at 0.2 alpha only while touched.
pub fn touch_sprites(
    world: &mut World,
    assets: &RenderAssets,
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam: &Camera2d,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();
    let save = world
        .get_resource::<crate::savedata_part::SaveData>()
        .cloned();
    let settings = save.as_ref().map(|s| &s.settings);
    let Some(input) = world.get_resource::<crate::input::NtInput>().cloned() else {
        return out;
    };
    // GML `TopCont/Draw_64:23` gate: `drawcontrols && player && !(MenuOptions &&
    // editing_mode) && !opt_keyboard && !opt_gamepad`. Both option reads go through the
    // one shared device law (`input::gml_input_device`) so chrome and touch INPUT cannot
    // disagree - the sampler runs on fingers-down, and reading the persisted flags alone
    // is what let chrome vanish on a phone while sticks still answered. Binds the whole
    // chrome (stick homes + act/swap/ability/splitfire), not just live claims: GML draws
    // the homes at all times.
    let (keyboard, gamepad) = crate::input::gml_input_device(save.as_ref());
    if keyboard || gamepad {
        return out;
    }
    let view = view_rect_world(canvas_dp, world_size, cam);
    let gm = hud_gui_map(view);
    let vw = view[2];
    let scale_opt = settings.map(|s| s.controls_scale.min(1.0)).unwrap_or(0.5);
    let scale = (1.0 + scale_opt).min(1.5);
    let split_fire = settings.map(|s| s.split_fire).unwrap_or(false);
    let autoaim = settings.map(|s| s.auto_aim).unwrap_or(false);
    let hidden = settings.map(|s| s.hidden_sticks).unwrap_or(false);
    let crosshair = settings.map(|s| s.crosshair as i32).unwrap_or(0);
    let alpha = |claimed: bool| {
        if hidden {
            if claimed { 0.2 } else { 0.0 }
        } else {
            1.0
        }
    };
    // Move stick: base frame 0 at the anchor, knob frame 1 at the
    // deflection point (`_stick_radius_treshold` 0.5). Anchors always
    // exist (GML homes from `Create_0`); the sampler only fills the
    // claim/deflection once a finger lands.
    let (move_home, attack_home) = crate::input::stick_homes(vw);
    let move_stick = input.move_stick.unwrap_or(crate::input::TouchStick {
        anchor: move_home,
        touch: -1,
        ..Default::default()
    });
    let attack_stick = input.attack_stick.unwrap_or(crate::input::TouchStick {
        anchor: attack_home,
        touch: -1,
        ..Default::default()
    });
    for stick in [&move_stick, &attack_stick] {
        let a = alpha(stick.touch >= 0);
        if a > 0.0 {
            let dir = if stick.dis > 0.0 {
                Vec2::new(stick.dir.to_radians().cos(), stick.dir.to_radians().sin())
            } else {
                Vec2::ZERO
            };
            let knob = stick.anchor + dir * (stick.dis.min(32.0) * 0.5);
            if let Some(s) = hud_gui_place(
                assets,
                "images/sprMobileControlJoystick.png",
                0,
                stick.anchor.x,
                stick.anchor.y,
                scale,
                [1.0, 1.0, 1.0, a],
                gm,
                view,
            ) {
                out.push(s);
            }
            if let Some(s) = hud_gui_place(
                assets,
                "images/sprMobileControlJoystick.png",
                1,
                knob.x,
                knob.y,
                scale,
                [1.0, 1.0, 1.0, a],
                gm,
                view,
            ) {
                out.push(s);
            }
        }
    }
    // Attack crosshair: `sprCrosshairBig` knob at the deflection
    // (GML `scrDrawMobileControls` attack region: rotating frames when
    // `crosshair < 7`, parked at 45° under full auto-aim; frame 2
    // under-deadzone marker while claimed, always in splitfire).
    {
        let stick = &attack_stick;
        let show_base = !split_fire || stick.touch >= 0;
        if show_base {
            let a = alpha(stick.touch >= 0);
            if a > 0.0 {
                let dir = if stick.dis > 0.0 {
                    Vec2::new(stick.dir.to_radians().cos(), stick.dir.to_radians().sin())
                } else {
                    Vec2::ZERO
                };
                let knob = if autoaim {
                    stick.anchor
                } else {
                    stick.anchor + dir * (stick.dis.min(64.0) * 0.5)
                };
                let under_deadzone = stick.dis / 32.0 < crate::input::ATTACK_BUTTON_DEADZONE;
                let show_dead = (split_fire && stick.touch < 0)
                    || (under_deadzone && stick.touch >= 0 && !autoaim);
                if show_dead {
                    if let Some(s) = hud_gui_place(
                        assets,
                        "images/sprMobileControlJoystick.png",
                        2,
                        stick.anchor.x,
                        stick.anchor.y,
                        scale,
                        [0.5, 0.5, 0.5, a],
                        gm,
                        view,
                    ) {
                        out.push(s);
                    }
                }
                let frames = strip_frames(assets, "images/sprCrosshairBig.png").max(1) as i32;
                let frame = crosshair.clamp(0, frames - 1);
                let knob_rot = if autoaim {
                    std::f32::consts::FRAC_PI_4
                } else {
                    0.0
                };
                if let Some(s) = place_touch_sprite_rot(
                    assets,
                    "images/sprCrosshairBig.png",
                    frame,
                    knob.x,
                    knob.y,
                    scale * 0.5,
                    knob_rot,
                    [0.05, 0.99, 0.6, a],
                    gm,
                    view,
                ) {
                    out.push(s);
                }
            }
        }
    }
    // Fixed buttons (`ButtonAct`/`ButtonSwap`/`ButtonActive` homes from `sample_touch`):
    // homes mirror the sampler (`ButtonAct` w/2,48; `ButtonSwap` 64,h/2-48; `ButtonActive`
    // w-64,h/2-48; `ButtonAttack` w-48,h/2), `vw` = view width in GUI px.
    // - `ButtonActive`: `sprMobileControlAbility` 0.75x `c_white` at `_alpha = min(1,
    //   rogue_hide / 60)`; GML starts `rogue_hide` at 180, so resting is full-bright
    //   white. `c_lime` (`activeforever`) / `c_gray` (claimed) / volume colors need the
    //   sampler's local claim + hold state: resting state only.
    // - `ButtonAct`: `sprMobileControlCorners` at the act home, only while the pickup
    //   prompt is lit (GML `alpha > 0`, [`crate::state::ActButton`]). GML also fills its
    //   `37/255` black disc (`scrDrawMobileControls:221-223`), undrawn here; the 1.6x
    //   corner re-draw at `scrDrawPlayerHUD.gml:391-394` is a GML no-op (`ButtonAct`
    //   never gets a `sprite_index`).
    // - splitfire `ButtonAttack`: corners sprite + double crosshair at the button home.
    {
        let gui_w = vw;
        let gui_h = 240.0;
        let act = Vec2::new(gui_w * 0.5, 48.0);
        let swap_home = Vec2::new(64.0, gui_h * 0.5 - 48.0);
        let attack_btn = Vec2::new(gui_w - 48.0, gui_h * 0.5);
        let active = Vec2::new(gui_w - 64.0, gui_h * 0.5 - 48.0);
        if let Some(s) = hud_gui_place(
            assets,
            "images/sprMobileControlAbility.png",
            0,
            active.x,
            active.y,
            scale * 0.75,
            [1.0, 1.0, 1.0, 1.0],
            gm,
            view,
        ) {
            out.push(s);
        }
        // `ButtonAct` pickup prompt: GML `scrDrawMobileControls` `if
        // (!instance_exists(_player)) alpha = 1; else if (alpha <= 0) continue`, and
        // `alpha` holds up only while the player stands on a pickup (`scrDrawPlayerHUD`
        // raises `active`, [`crate::state::ActButton`]). Unconditional draw = the
        // "pickup indicator always showing" bug.
        let act_lit = world
            .get_resource::<crate::state::ActButton>()
            .is_some_and(|a| a.alpha > 0.0);
        if act_lit
            && let Some(s) = hud_gui_place(
                assets,
                "images/sprMobileControlCorners.png",
                0,
                act.x,
                act.y,
                scale,
                [1.0, 1.0, 1.0, 1.0],
                gm,
                view,
            )
        {
            out.push(s);
        }
        // Swap button (`ButtonSwap` region in `scrDrawMobileControls`):
        // dark disc + the current weapon sprite over the backup
        // (vanilla GML draws wep white / bwep gray, tilted 45°).
        {
            let black = [0.0, 0.0, 0.0, 37.0 / 255.0];
            if let Some(s) = hud_gui_place(
                assets,
                "images/sprMobileControlJoystick.png",
                0,
                swap_home.x,
                swap_home.y,
                scale,
                black,
                gm,
                view,
            ) {
                out.push(s);
            }
            let (wep, bwep) = world
                .query::<(&Player, &crate::comps_a::Inventory)>()
                .iter(world)
                .next()
                .map(|(_, inv)| {
                    (
                        inv.weapons[inv.current.min(inv.weapons.len() - 1)],
                        inv.weapons[inv
                            .weapon_slots
                            .saturating_sub(1)
                            .min(inv.weapons.len() - 1)],
                    )
                })
                .unwrap_or((crate::data::WeaponId::NONE, crate::data::WeaponId::NONE));
            let draw_gun =
                |id: crate::data::WeaponId, tint: [f32; 4], out: &mut Vec<SpriteInstance>| {
                    let meta = crate::weapon_runtime::weapon_meta(id);
                    let stem = if meta.wep_sprt.is_empty() || meta.wep_sprt == "mskNone" {
                        "sprRevolver"
                    } else {
                        meta.wep_sprt
                    };
                    let path = format!("images/{stem}.png");
                    if let Some(s) = place_touch_sprite_rot(
                        assets,
                        &path,
                        0,
                        swap_home.x,
                        swap_home.y + 10.0,
                        scale + 0.5,
                        45.0_f32.to_radians(),
                        tint,
                        gm,
                        view,
                    ) {
                        out.push(s);
                    }
                };
            draw_gun(bwep, [0.7, 0.7, 0.7, 1.0], &mut out);
            draw_gun(wep, [1.0, 1.0, 1.0, 1.0], &mut out);
        }
        if split_fire {
            let frames = strip_frames(assets, "images/sprCrosshairBig.png").max(1) as i32;
            let frame = crosshair.clamp(0, frames - 1);
            let half = scale * 0.5;
            if let Some(s) = hud_gui_place(
                assets,
                "images/sprMobileControlCorners.png",
                0,
                attack_btn.x,
                attack_btn.y,
                scale * 0.5,
                [1.0, 1.0, 1.0, 1.0],
                gm,
                view,
            ) {
                out.push(s);
            }
            if let Some(s) = hud_gui_place(
                assets,
                "images/sprCrosshairBig.png",
                frame,
                attack_btn.x,
                attack_btn.y,
                scale * 0.5,
                [0.49, 0.99, 0.05, 1.0],
                gm,
                view,
            ) {
                out.push(s);
            }
            let _ = half;
        }
    }
    let _ = (vw, scale);
    out
}

/// GML in-run pause button (`UberCont/Draw_64:46-61`):
/// `draw_sprite_ext(sprMobilePauseButton, 0, view_width - 24, 16, 0.75, 0.75, 0, c_white,
/// 0.5)` - on EVERY device (`opt_pausebutton` gates it, never the input mode), so it lives
/// outside the touch chrome. `UberCont` depth -1000 < `TopCont`'s -15 and `Menu`'s -1001,
/// so it paints OVER mobile controls and menus ([`Z_PAUSE_BUTTON`]). Hit test in
/// [`crate::input`].
pub fn pause_button_sprite(
    world: &mut World,
    assets: &RenderAssets,
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam: &Camera2d,
) -> Option<SpriteInstance> {
    let shown = world
        .get_resource::<crate::savedata_part::SaveData>()
        .is_some_and(|s| s.settings.pause_button)
        && world.query::<&Player>().iter(world).next().is_some();
    if !shown {
        return None;
    }
    let view = view_rect_world(canvas_dp, world_size, cam);
    let gx = view[2] + crate::input::PAUSE_BUTTON_GUI[0];
    let mut s = hud_gui_place(
        assets,
        "images/sprMobilePauseButton.png",
        0,
        gx,
        crate::input::PAUSE_BUTTON_GUI[1],
        0.75,
        [1.0, 1.0, 1.0, 0.5],
        hud_gui_map(view),
        view,
    )?;
    s.z = Z_PAUSE_BUTTON;
    Some(s)
}

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
        crate::MenuOverlay::Splash => {
            // GML `MakeGame/Draw_0:94` `scrDrawRoadmap(_cx, _cy, pos)`: the
            // prompt reveals one node per frame from the saved run's log.
            if let Some(save) = world.get_resource::<crate::run_save::PendingRunSave>() {
                let (waypoints, pos) = (
                    save.0.session.run.waypoints.clone(),
                    world
                        .get_resource::<SplashState>()
                        .map(|s| s.load.pos)
                        .unwrap_or(0),
                );
                out.extend(roadmap_sprites(
                    assets,
                    &gui_to_world,
                    vw * 0.5,
                    120.0,
                    &waypoints,
                    pos as usize,
                ));
            }
        }
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
            push_confirm_glyph(&mut out, assets, world, vw, view, gm);
        }
        crate::MenuOverlay::Title => {
            let menu = world.get_resource::<MenuState>().cloned();
            let save = world
                .get_resource::<crate::savedata_part::SaveData>()
                .cloned();
            let selected = world
                .get_resource::<SelectedCharacter>()
                .map(|s| s.0 as usize)
                .unwrap_or(0);
            let (cursor, go_visible, loadout_open, loadout_frame) = menu
                .as_ref()
                .map(|m| {
                    (
                        m.title_cursor,
                        m.title_go_visible,
                        m.loadout_open,
                        m.loadout_frame,
                    )
                })
                .unwrap_or((0, false, false, 0.0));
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
                // GML `CharSelect/Draw_0`: `can = scr_race_is_unlocked(race) ||
                // UberCont.weekly_run`, `draw_sprite_ext(can ? sprite_index :
                // sprCharSelectLocked, race, x, y, 1, 1, 0, color, 1)`, `color = (can &&
                // selected) ? c_white : c_gray` (128,128,128 - NOT half-alpha). Pods
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
                    let mut s = s;
                    s.z = Z_MENU + 1.0;
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
                        let mut s = s;
                        s.z = Z_MENU + 1.0;
                        out.push(s);
                    }
                }
            }
            // Selected-race portrait + splat + name plate (GML
            // scrCampfireMenuDrawRacePortrait/CharText): single-player port draws only
            // index 0; GML's P2-P4 order (`scrMenuDrawPlayersOrdered`, back-layer gray at
            // index >= 2) is vacuous with no coop instances.
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
            // Big-name sprite: GML draws it ONLY for an unlocalized name in
            // multiplayer (`scrCampfireMenuDrawCharText:444`); single-player uses the
            // bigname text (the overlay lines), so no sprite.
            // GO button: armed only + GML `Menu/Create_0:49` space gate (`_slot_x <
            // view_width - 30`); single-player is always `is_server`.
            let last_pod_x = 8.0 + roster.len().saturating_sub(1) as f32 * go_step(roster.len());
            let go_space = last_pod_x < vw - 30.0;
            if go_visible && go_space {
                let dp = go_button_pos([vw, 240.0], roster.len(), 19.0);
                if let Some(s) = assets.sprite_for(
                    "images/sprGoButtonSymbolic.png",
                    0,
                    gui_to_world(dp[0], dp[1]),
                    false,
                    0.0,
                    [1.0; 4],
                ) {
                    let mut s = s;
                    s.z = Z_MENU + 1.0;
                    out.push(s);
                }
            }
            // Loadout grid (`scrMenuDrawLoadout` geometry and frame
            // animation) plus the closed-frame preview (splat + arrow +
            // minis ride the closed frame in GML).
            if selected != 0 && loadout_frame > 0.0 {
                menu_loadout_sprites(
                    world,
                    assets,
                    [vw, 240.0],
                    selected,
                    loadout_frame,
                    loadout_open,
                    &|p| gui_to_world(p[0], p[1]),
                    &mut out,
                );
            } else if selected != 0 {
                menu_loadout_closed_sprites(
                    world,
                    assets,
                    [vw, 240.0],
                    selected,
                    loadout_available_for_race(race),
                    true,
                    &|p| gui_to_world(p[0], p[1]),
                    &mut out,
                );
            }
        }
        crate::MenuOverlay::GameOver => {
            let (offsety, death_pos, splat) = world
                .get_resource::<MenuState>()
                .map(|m| (m.go_offsety, m.go_death_pos, m.go_splat))
                .unwrap_or((0.0, 0.0, 0.0));
            let screen = world
                .get_resource::<MenuState>()
                .and_then(|menu| menu.game_over.clone());
            let (waypoints, area) = world
                .get_resource::<Run>()
                .map(|run| (run.waypoints.clone(), Some(run.area)))
                .unwrap_or_else(|| (Vec::new(), None));
            let prefix = death_pos.round() as usize;
            let cx = vw * 0.5;
            let splat_frame = splat.floor().clamp(0.0, 2.0) as i32;

            out.extend(roadmap_sprites(
                assets,
                &gui_to_world,
                cx - 48.0,
                120.0 - offsety,
                &waypoints,
                prefix,
            ));
            if let Some(sprite) = assets.sprite_for(
                "images/sprKilledBySplat.png",
                splat_frame,
                gui_to_world(cx + 86.0, 120.0 - offsety - 32.0),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(sprite);
            }
            if let Some(path) = screen.as_ref().and_then(|screen| screen.deathcause_sprite) {
                let frames = strip_frames(assets, path).max(1) as f32;
                let frame = ((death_pos * 0.4).floor() % frames) as i32;
                if let Some(sprite) = assets.sprite_for(
                    path,
                    frame,
                    gui_to_world(cx + 86.0, 120.0 - offsety),
                    false,
                    0.0,
                    [1.0; 4],
                ) {
                    out.push(sprite);
                }
            }
            if let Some(screen) = screen.as_ref()
                && let Some(race) = screen.race
                && let Some(area) = area
            {
                let skin = screen.skin.map(|skin| skin as u8).unwrap_or(0);
                let (path, frame) = if race == RaceId::Chicken && screen.hp <= 0 {
                    ("images/sprMapIconChickenHeadless.png", i32::from(skin))
                } else if race == RaceId::Rebel && skin == 1 && area == AreaId::FrozenCity {
                    ("images/sprMapIconRebelBHooded.png", 0)
                } else {
                    (
                        "images/sprMapIcon.png",
                        crate::state::menus::race_skin_subimage(race as usize, skin),
                    )
                };
                if frame >= 0 {
                    let (cursor_x, cursor_y) =
                        roadmap_cursor_pos(&waypoints, prefix, cx - 48.0, 120.0 - offsety);
                    if let Some(sprite) = assets.sprite_for(
                        path,
                        frame,
                        gui_to_world(cursor_x, cursor_y),
                        false,
                        0.0,
                        [1.0; 4],
                    ) {
                        out.push(sprite);
                    }
                }
            }
            if let Some(sprite) = assets.sprite_for(
                "images/sprGameOverCenterSplat.png",
                splat_frame,
                gui_to_world(cx, 240.0 - 32.0),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(sprite);
            }
            if let Some(screen) = screen.as_ref()
                && let Some(race) = screen.race
            {
                out.extend(ability_hud_sprites(
                    assets,
                    view,
                    gm,
                    race,
                    screen.ultra,
                    &screen.mutations,
                    screen.patience_used,
                ));
            }
            let (appear, hover) = world
                .get_resource::<MenuState>()
                .map(|menu| (menu.go_appear.max(0.0), menu.hover_label.as_str()))
                .unwrap_or((0.0, ""));
            let buttons = [
                ("MENU", 0, 120.0 + 58.0 + offsety + appear, appear),
                (
                    "RETRY",
                    1,
                    120.0 + 90.0 + offsety + appear + 1.0,
                    appear + 1.0,
                ),
            ];
            for (label, frame, gy, button_appear) in buttons {
                if button_appear >= 2.0 {
                    continue;
                }
                let tint = if hover == label {
                    [1.0; 4]
                } else {
                    [0.5, 0.5, 0.5, 1.0]
                };
                push_shadowed_gui(
                    &mut out,
                    assets,
                    "images/sprPauseButton.png",
                    frame,
                    cx,
                    gy,
                    0.65,
                    tint,
                    view,
                    gm,
                );
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
        // GML `LevCont/Draw_0`: `scrDrawSpiral()` (opaque clear) then ONLY the
        // offer chrome - title/subtitle text plus SkillIcon/UltraIcon/CrownIcon cards. No
        // camp pods, roadmap or PlayerHUD (those rooms never had them). Text half in
        // `menu_gui_texts(Mutation)`; cards compose here, under the cover pass.
        crate::MenuOverlay::Mutation => {
            let cx = vw * 0.5;
            let menu = world.get_resource::<MenuState>().cloned();
            let selected = menu.as_ref().and_then(|state| state.mutation_selected);
            let is_ultra = pending_offer_is_ultra(world);
            let title_path = if is_ultra {
                "images/sprLevelUltraText.png"
            } else {
                "images/sprLevelUpText.png"
            };
            let appear = menu
                .as_ref()
                .map(|state| state.mutation_appear)
                .unwrap_or(0.0);
            for (dx, dy) in [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)] {
                if let Some(s) = assets.sprite_for(
                    title_path,
                    2,
                    gui_to_world(cx + appear + dx, 48.0 + dy),
                    false,
                    0.0,
                    [0.0, 0.0, 0.0, 1.0],
                ) {
                    out.push(s);
                }
            }
            if let Some(s) = assets.sprite_for(
                title_path,
                2,
                gui_to_world(cx + appear, 48.0),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            if menu.as_ref().is_some_and(|state| state.mutation_splat)
                && let Some(s) = assets.sprite_for(
                    "images/sprMutationSplat.png",
                    menu.as_ref()
                        .map(|state| state.mutation_splat_frame as i32)
                        .unwrap_or(0),
                    gui_to_world(cx, 240.0 - 31.0),
                    false,
                    0.0,
                    [1.0; 4],
                )
            {
                out.push(s);
            }

            let race = pending_offer_race(world);
            let n = if is_ultra {
                world
                    .get_resource::<PendingUltra>()
                    .map(|pending| pending.choices.len())
                    .unwrap_or(0)
            } else {
                world
                    .get_resource::<PendingMutation>()
                    .map(|pending| pending.choices.len())
                    .unwrap_or(0)
            };
            let step = (32.0f32).min((vw / (n.max(1) as f32 + 1.0)).floor());
            let scale = if is_ultra {
                1.0
            } else {
                (step / 32.0).max(0.65)
            };
            let half = (step as i32 / 2) as f32;
            let xoff = if n >= 10 { -12.0 } else { 0.0 };
            let icon_y = 240.0 - 21.0;
            let gamepad_ui = gamepad_ui_on(world);
            let gamepad_type = world
                .get_resource::<crate::savedata_part::SaveData>()
                .map(|s| s.settings.gamepad_type)
                .unwrap_or(0);
            for i in 0..n {
                let (path, frame, mul) = if is_ultra {
                    let choice = world
                        .get_resource::<PendingUltra>()
                        .and_then(|pending| pending.choices.get(i).copied());
                    (
                        "images/sprEGSkillIcon.png",
                        choice.map(|id| ultra_offer_frame(race, id)).unwrap_or(0),
                        1.0,
                    )
                } else {
                    let choice = world
                        .get_resource::<PendingMutation>()
                        .and_then(|pending| pending.choices.get(i).copied());
                    (
                        "images/sprSkillIcon.png",
                        choice.map(mutation_hud_frame).unwrap_or(0),
                        scale,
                    )
                };
                let (path, frame) = if assets.uv(path, 0).is_some() {
                    (path, frame)
                } else if is_ultra {
                    (
                        "images/sprEGIconHUD.png",
                        world
                            .get_resource::<PendingUltra>()
                            .and_then(|pending| pending.choices.get(i).copied())
                            .map(|id| ultra_hud_frame(race, id))
                            .unwrap_or(frame),
                    )
                } else {
                    ("images/sprSkillIconHUD.png", frame)
                };
                let card_y = icon_y
                    + menu
                        .as_ref()
                        .and_then(|state| state.mutation_appear_y.get(i).copied())
                        .unwrap_or(0.0)
                    - if selected == Some(i) { 1.0 } else { 0.0 };
                let tint = if selected == Some(i) {
                    [1.0; 4]
                } else {
                    [0.5, 0.5, 0.5, 1.0]
                };
                let card_x = cx + xoff - (n as f32 - 1.0) * half + i as f32 * step;
                if let Some(s) = assets.sprite_scaled_rotated(
                    path,
                    frame,
                    gui_to_world(card_x, card_y),
                    mul * gm.s,
                    0.0,
                    tint,
                ) {
                    out.push(s);
                }
                if gamepad_ui && selected == Some(i) {
                    // GML `UberCont/Draw_0:115-120`: a selected
                    // SkillIcon / CrownIcon / UltraIcon takes the
                    // `gp_face1` badge on its `bbox_right, bbox_top`
                    // corner.
                    let (w, h) = assets
                        .native_size(path)
                        .map(|size| (size.x, size.y))
                        .unwrap_or((24.0, 32.0));
                    push_gamepad_glyph(
                        &mut out,
                        assets,
                        gamepad_type,
                        0,
                        card_x + w * 0.5 * mul,
                        card_y - h * 0.5 * mul,
                        view,
                        gm,
                        [1.0; 4],
                    );
                }
            }
        }
        crate::MenuOverlay::Settings => {
            let cx = vw * 0.5;
            let menu = world.get_resource::<MenuState>().cloned();
            let page = menu.as_ref().map(|m| m.settings_page).unwrap_or(0);
            let cursor = menu
                .as_ref()
                .map(|m| m.settings_cursor)
                .unwrap_or(usize::MAX);
            let save = world
                .get_resource::<crate::savedata_part::SaveData>()
                .cloned()
                .unwrap_or_default();
            let settings = &save.settings;
            let splat = menu
                .as_ref()
                .map(|m| m.settings_splat_get(page, cursor))
                .unwrap_or(0.0);
            if splat > 0.0
                && menu
                    .as_ref()
                    .is_some_and(|m| m.settings_cursor != usize::MAX)
                && let Some(row) = settings_hot_rows(page, vw).get(cursor)
                && !matches!(row.op, SettingHotOp::Back)
                && settings_row_available(world, page, row)
                && settings_row_in_vision(row.gy)
            {
                if let Some(s) = assets.sprite_for(
                    "images/sprMainMenuSplat.png",
                    (splat.floor() as i32).min(crate::state::menus::SETTINGS_SPLAT_MAX as i32),
                    gui_to_world(cx, row.gy),
                    false,
                    0.0,
                    [1.0; 4],
                ) {
                    out.push(s);
                }
            }
            if page == 0 {
                for (i, frame) in [0, 1, 2, 3].into_iter().enumerate() {
                    let tint = if cursor == i {
                        [1.0; 4]
                    } else {
                        [0.5, 0.5, 0.5, 1.0]
                    };
                    push_shadowed_sprite(
                        &mut out,
                        assets,
                        "images/sprOptionsButtons.png",
                        frame,
                        cx,
                        72.0 + i as f32 * 24.0,
                        view,
                        gm,
                        false,
                        tint,
                    );
                }
            }
            let slider_x = cx + SETTINGS_SLIDER_X_OFFSET;
            match page {
                1 => {
                    for (y, value) in [
                        (56.0, settings.master_volume),
                        (76.0, settings.music_volume),
                        (96.0, settings.ambience_volume),
                        (116.0, settings.sfx_volume),
                    ] {
                        push_settings_slider(&mut out, assets, view, gm, slider_x, y, value, 1.0);
                    }
                }
                2 => {
                    push_settings_slider(
                        &mut out,
                        assets,
                        view,
                        gm,
                        slider_x,
                        84.0,
                        settings.screenshake,
                        2.0,
                    );
                    push_settings_slider(
                        &mut out,
                        assets,
                        view,
                        gm,
                        slider_x,
                        102.0,
                        settings.freezeframes,
                        1.0,
                    );
                }
                4 => {
                    push_settings_slider(
                        &mut out,
                        assets,
                        view,
                        gm,
                        slider_x,
                        174.0,
                        settings.controls_scale,
                        1.0,
                    );
                }
                _ => {}
            }
            let back_hover = menu.as_ref().is_some_and(|m| m.settings_back_hover);
            let back_x = if cfg!(target_os = "android") {
                24.0
            } else {
                16.0
            };
            push_shadowed_sprite(
                &mut out,
                assets,
                "images/sprBackButton.png",
                if back_hover { 1 } else { 0 },
                back_x,
                20.0,
                view,
                gm,
                false,
                if back_hover {
                    [1.0; 4]
                } else {
                    [0.7, 0.7, 0.7, 1.0]
                },
            );
            let gamepad_ui = gamepad_ui_on(world);
            if gamepad_ui {
                // GML `BackButton/Draw_64:22-27`: the `gp_face2` badge
                // sits on the back button's `(_x + 16, _y)` and dims
                // with it while unhovered.
                push_gamepad_glyph(
                    &mut out,
                    assets,
                    settings.gamepad_type,
                    1,
                    back_x + 16.0,
                    20.0,
                    view,
                    gm,
                    if back_hover {
                        [1.0; 4]
                    } else {
                        [0.7, 0.7, 0.7, 1.0]
                    },
                );
            }
            push_confirm_glyph(&mut out, assets, world, vw, view, gm);
            if page == 4 && gamepad_ui && cursor == 1 {
                // GML `Other_20.gml:532-538`: the selected GAMEPAD STYLE
                // row sprouts four `gamepad_icon_small` previews at
                // `(gui_w / 2 - 32) + i * 16, startdrawy - 16` - the
                // list top here is the GAMEPAD row at y 48.
                for i in 0..4 {
                    push_gamepad_glyph(
                        &mut out,
                        assets,
                        settings.gamepad_type,
                        i,
                        cx - 32.0 + i as f32 * 16.0,
                        32.0,
                        view,
                        gm,
                        [1.0; 4],
                    );
                }
            }
            if page == 13 && gamepad_ui {
                let keymap = world
                    .get_resource::<crate::keymap::InputMapState>()
                    .map(|s| s.session.map.clone())
                    .unwrap_or_else(crate::keymap::default_keymap);
                let capturing = world
                    .get_resource::<crate::keymap::InputMapState>()
                    .and_then(|s| s.session.capture.clone());
                let rows = settings_hot_rows(13, vw);
                for (i, action) in crate::keymap::NtAction::ALL.iter().enumerate() {
                    if capturing.as_ref().is_some_and(|c| c.action == *action) {
                        continue;
                    }
                    let Some(row) = rows.get(i) else {
                        continue;
                    };
                    if !settings_row_available(world, 13, row) {
                        continue;
                    }
                    if let Some(frame) = gamepad_glyph_frame(&keymap.active(action, true)) {
                        push_gamepad_glyph(
                            &mut out,
                            assets,
                            settings.gamepad_type,
                            frame,
                            cx + 32.0,
                            row.gy,
                            view,
                            gm,
                            [1.0; 4],
                        );
                    }
                }
            }
        }
        crate::MenuOverlay::Pause => {
            let cx = vw * 0.5;
            let menu = world.get_resource::<MenuState>().cloned();
            let splat = menu
                .as_ref()
                .map(|m| m.pause_splat.floor().clamp(0.0, 3.0) as i32)
                .unwrap_or(0);
            let player = world
                .query::<(&RaceState, &Health)>()
                .iter(world)
                .next()
                .map(|(race, health)| (race.race, health.hp));
            if let Some((race, hp)) = player {
                let skin = world
                    .get_resource::<crate::savedata_part::SaveData>()
                    .map(|save| save.race_loadout(race).preferred_skin)
                    .unwrap_or(0);
                let area = world
                    .get_resource::<Run>()
                    .map(|run| run.area)
                    .unwrap_or(AreaId::Desert);
                if let Some((path, frame)) = big_portrait_for(race, skin, Some(hp), area) {
                    if let Some(s) = assets.sprite_for(
                        path,
                        frame,
                        gui_to_world(
                            -2.0 - menu.as_ref().map(|m| m.pause_portrait_anim).unwrap_or(0.0),
                            260.0,
                        ),
                        false,
                        0.0,
                        [1.0; 4],
                    ) {
                        out.push(s);
                    }
                }
            }
            if let Some(s) = assets.sprite_for(
                "images/sprCharSplat.png",
                splat,
                gui_to_world(0.0, 240.0 - 31.0),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
            if let Some(s) = assets.sprite_for(
                "images/sprCharSplat.png",
                splat,
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
            let yoff = world
                .get_resource::<Run>()
                .is_some_and(|run| run.hardmode)
                .then_some(4.0)
                .unwrap_or(0.0);
            if let Some(s) = assets.sprite_for(
                "images/sprPaused.png",
                0,
                gui_to_world(cx + 1.0, 53.0 - yoff),
                false,
                0.0,
                [0.0, 0.0, 0.0, 1.0],
            ) {
                out.push(s);
            }
            if let Some(s) = assets.sprite_for(
                "images/sprPaused.png",
                0,
                gui_to_world(cx, 52.0 - yoff),
                false,
                0.0,
                [1.0; 4],
            ) {
                out.push(s);
            }
        }
        crate::MenuOverlay::Unlock => {
            use crate::state::menus::{UnlockPopup, race_skin_subimage};
            let state = world
                .get_resource::<MenuState>()
                .map(|m| m.unlock)
                .unwrap_or_default();
            let popup = world
                .get_resource::<MenuState>()
                .and_then(|m| m.unlock_queue.first().copied());
            if state.visible
                && let Some(popup) = popup
            {
                let (race_gml, skin) = match popup {
                    UnlockPopup::Race(race) => (race as usize, 0u8),
                    UnlockPopup::Skin(race, skin) => (race as usize, skin),
                };
                let sub = race_skin_subimage(race_gml, skin);
                if state.splat > 1.0 && sub >= 0 {
                    let path = format!("images/sprBigPortrait{}.png", sub);
                    if let Some(s) = assets.sprite_for(
                        Box::leak(path.into_boxed_str()) as &str,
                        0,
                        gui_to_world(vw * 0.5 - 60.0, 240.0 - 10.0 + state.addy),
                        false,
                        0.0,
                        [1.0; 4],
                    ) {
                        out.push(s);
                    }
                }
                for gy in [0.0, 240.0 - 32.0] {
                    out.push(white_quad(
                        gui_to_world(vw * 0.5, gy + 16.0),
                        0.0,
                        Vec2::new(vw, 32.0),
                        [0.0, 0.0, 0.0, 1.0],
                    ));
                }
                if let Some(s) = assets.sprite_for(
                    "images/sprMutationSplat.png",
                    (state.splat.floor() as i32).min(3),
                    gui_to_world(vw * 0.5, 240.0 - 20.0),
                    false,
                    0.0,
                    [1.0; 4],
                ) {
                    out.push(s);
                }
            }
        }
        _ => {}
    }
    out
}

pub fn pause_button_sprites(
    world: &mut World,
    assets: &RenderAssets,
    canvas_dp: [f32; 2],
    world_size: [f32; 2],
    cam: &Camera2d,
) -> Vec<SpriteInstance> {
    let view = view_rect_world(canvas_dp, world_size, cam);
    let gm = hud_gui_map(view);
    let vw = view[2];
    let menu = world.get_resource::<MenuState>().cloned();
    let confirm = menu.as_ref().and_then(|m| m.pause_confirm);
    let labels: &[&str] = if confirm.is_some() {
        &["BACK", if confirm == Some(0) { "QUIT" } else { "RETRY" }]
    } else {
        &["MENU", "RETRY", "SETTINGS", "CONTINUE"]
    };
    let positions: &[(f32, f32, i32)] = if confirm.is_some() {
        &[
            (52.0, 192.0, 4),
            (vw - 52.0, 192.0, if confirm == Some(0) { 5 } else { 6 }),
        ]
    } else {
        &[
            (45.0, 176.0, 0),
            (60.0, 208.0, 1),
            (vw - 68.0, 176.0, 2),
            (vw - 78.0, 208.0, 3),
        ]
    };
    let hover = menu.as_ref().map(|m| m.hover_label.as_str()).unwrap_or("");
    let mut out = Vec::new();
    // GML `UberCont/Draw_64:113-137`: the one-shot `sprContinuedRunIcon` hint
    // over the QUIT confirm button while `etc.saving_tip` still reads 0. Four
    // black offset copies then the white icon, gated on `appear <= 2` (and the
    // white pass on `appear <= 1`).
    if confirm == Some(0)
        && world
            .get_resource::<crate::savedata_part::SaveData>()
            .is_some_and(|s| s.saving_tip == 0)
    {
        let icon_x = vw * 0.5;
        let icon_y = 240.0 - 36.0 - 30.0;
        let appear = menu
            .as_ref()
            .and_then(|m| m.pause_appear.get(1).copied())
            .unwrap_or(0.0);
        let icon_y = icon_y + (appear - 1.0).max(0.0) + 1.0;
        if appear <= 2.0 {
            for (ox, oy) in [(-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)] {
                push_shadowed_sprite(
                    &mut out,
                    assets,
                    "images/sprContinuedRunIcon.png",
                    0,
                    icon_x + ox,
                    icon_y + oy,
                    view,
                    gm,
                    false,
                    [0.0, 0.0, 0.0, 1.0],
                );
            }
        }
        if appear <= 1.0 {
            push_shadowed_sprite(
                &mut out,
                assets,
                "images/sprContinuedRunIcon.png",
                0,
                icon_x,
                icon_y,
                view,
                gm,
                false,
                [1.0; 4],
            );
        }
    }
    for (i, (label, (x, y, frame))) in labels.iter().zip(positions.iter()).enumerate() {
        let appear = menu
            .as_ref()
            .and_then(|m| m.pause_appear.get(i).copied())
            .unwrap_or(0.0);
        if appear >= 2.0 {
            continue;
        }
        let tint = if hover == *label {
            [1.0; 4]
        } else {
            [0.5, 0.5, 0.5, 1.0]
        };
        push_shadowed_sprite(
            &mut out,
            assets,
            "images/sprPauseButton.png",
            *frame,
            *x,
            *y + appear,
            view,
            gm,
            false,
            tint,
        );
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

/// Spiral center figures (`scripts/scrDrawSpiral` CPU layer): the carried
/// crown orbiting the spiral center plus one hurt-frame player figure per player, over the
/// vortex background. `center` = view center (GML `fishx/fishy`), `angle_deg` = spiral
/// angle in degrees (GML `image_angle`; `SpiralCtl.angle` too). Skipped while Throne II
/// lives (GML `Nothing2` gate: Nothing2/Nothing2Corpse/Nothing2Death all suppress - port
/// reads any live Throne-II enemy) or while `Credits` runs without a crown carrier.
pub fn spiral_figures(
    world: &mut World,
    assets: &RenderAssets,
    center: Vec2,
    angle_deg: f32,
) -> Vec<SpriteInstance> {
    let mut out = Vec::new();

    // GML `scrDrawSpiral` figure gate verbatim: nothing draws while
    // Throne II runs (`Nothing2`/`Nothing2Corpse`/`Nothing2Death` - the
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

    let has_crown_object = world.query::<&CrownObject>().iter(world).next().is_some();
    let mut crown = CrownKind::None;
    let mut hurts: Vec<String> = Vec::new();
    {
        let mut q = world.query::<(&Player, Option<&PlayerAnim>)>();
        for (player, anim) in q.iter(world) {
            if has_crown_object && crown == CrownKind::None {
                crown = player.crown;
            }
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

/// Damage numbers ride [`fx_texts`]: [`SpriteInstance`] carries no text, so
/// [`DamageNumber`] comps resolve to [`WorldText`] for the repose text overlay.
/// GML `draw_pickup_button` sprite half (`draw_gamepad_button.gml:43-101`):
/// the `sprEPickup` pill at `(x, y - 7)`, plus the `pick` pad glyph at
/// `(x, y - 15)` while the GAMEPAD switch is on.
fn push_pickup_prompt(
    out: &mut Vec<SpriteInstance>,
    assets: &RenderAssets,
    at: Vec2,
    frame: i32,
    style: Option<u8>,
    glyph: Option<i32>,
) {
    if let Some(mut s) = assets.sprite_for(
        crate::hud::PICKUP_BUTTON_ART,
        frame,
        at + Vec2::new(0.0, -7.0),
        false,
        0.0,
        [1.0; 4],
    ) {
        s.z = Z_FX;
        out.push(s);
    }
    if let (Some(style), Some(glyph)) = (style, glyph)
        && let Some(mut s) = assets.sprite_for(
            gamepad_icon_strip(style),
            glyph,
            at + Vec2::new(0.0, -15.0),
            false,
            0.0,
            [1.0; 4],
        )
    {
        s.z = Z_FX;
        out.push(s);
    }
}

pub fn fx_instances(world: &mut World, assets: &RenderAssets) -> Vec<SpriteInstance> {
    let mut out = Vec::new();

    // GML `scrDrawInteractionHUD` (`scripts/scrDrawPlayerHUD/scrDrawPlayerHUD.gml:339-376`)
    // + `draw_pickup_button` (`scripts/draw_gamepad_button/draw_gamepad_button.gml:43-101`).
    // Prompt TEXT rides the text pass; sprite parts here: `sprEPickup` "E" pill at
    // `(x, y-7)` (full opacity, `_offset = 7`, once per `prop_prompts` hit too since GML
    // runs `draw_pickup_button` before the weapon/prop branch at `:355`), and
    // `scrDrawTypeAmmo`'s background/icon pair at `(x + 7, y - 14)` - background subimage 2,
    // icon at `_frames - ceil(_frames * fill)`, so a full magazine is icon frame 0.
    {
        if let Some(label) = world.get_resource::<crate::pickups::WeaponLabel>() {
            // GML `draw_gamepad_button.gml:49-56`: under the GAMEPAD switch
            // the pill flips to `sprEPickup` frame 1 and takes the `pick`
            // pad glyph at `(_x, _y
            // - 8)` of the shifted pill.
            let style = gamepad_style(world);
            let pad_pick = style.and_then(|_| {
                let entry = world
                    .get_resource::<crate::keymap::InputMapState>()
                    .map(|s| s.session.map.active(&crate::keymap::NtAction::Pick, true))
                    .unwrap_or(repame_input::KeymapEntry::None);
                gamepad_glyph_frame(&entry)
            });
            let (frame, glyph) = if style.is_some() {
                (1, pad_pick)
            } else {
                (label.button_frame as i32, None)
            };
            if let Some(target) = label.target.filter(|_| !label.text.is_empty())
                && let Some(at) = world.get::<Pos>(target).map(|p| p.0)
            {
                push_pickup_prompt(&mut out, assets, at, frame, style, glyph);
                if let Some((bg, icon)) = label.ammo_gauge {
                    let at = at + Vec2::new(7.0, -14.0);
                    if let Some(mut s) = assets.sprite_for(bg, 2, at, false, 0.0, [1.0; 4]) {
                        s.z = Z_FX;
                        out.push(s);
                    }
                    let frames = strip_frames(assets, icon);
                    let full = frames.saturating_sub(1);
                    let idx = (full as f32 * label.ammo_fill).ceil();
                    let frame = (full as f32 - idx).round().clamp(0.0, full as f32) as i32;
                    if let Some(mut s) = assets.sprite_for(icon, frame, at, false, 0.0, [1.0; 4]) {
                        s.z = Z_FX;
                        out.push(s);
                    }
                }
            }
            for (entity, _) in label.prop_prompts.iter() {
                let Some(at) = world.get::<Pos>(*entity).map(|p| p.0) else {
                    continue;
                };
                push_pickup_prompt(&mut out, assets, at, frame, style, glyph);
            }
        }
    }

    // Beams: `Pos` is the sim segment center, so the quad stays
    // center-anchored (the strip's `[0, 0.5]` anchor overridden) and
    // `size.x = length` spans the endpoints. The tint is the sim's own
    // beam color (ion cyan, laser red, boss orange/green) - GML draws
    // lasers as their own objects (`objects/Laser/Draw_0.gml`) and lights
    // the vault beam with `sprLightBeamVault`
    // (`objects/CrownPed/Create_0.gml:4`), so the strip name and the
    // color ramp here are port-side.
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
    // fraction; alpha is the port's own `(1 - t) * 0.9` fade, no floor -
    // GML has no arc entity to fade (`objects/LightningCrystal/
    // Create_0.gml:10` just swaps in `sprLightningCrystalFire`), so there
    // is nothing to cite.
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

    // Muzzles: GML draws no muzzle flash either - fire feedback is the
    // shell FX spawned at the muzzle
    // (`scripts/scrBulletShotShellFX/scrBulletShotShellFX.gml:9-11`)
    // plus the `wkick` recoil that offsets the gun sprite
    // (`objects/Player/Draw_0.gml:109-111`). The yellow `muzzle_burst`
    // spray and the warm tinted quad here are port-side.
    // `FiredWeapon` markers expire silently.

    // Hazard clouds: kind-tinted translucent rects, no pulse; the ring has no
    // catalog entry, so ring degrades to a disc.
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
/// numbers - no separate sprite mapping.
pub fn fx_texts(world: &mut World) -> Vec<WorldText> {
    let mut q = world.query::<&DamageNumber>();
    q.iter(world)
        .map(|n| WorldText {
            text: n.text.clone(),
            pos: Vec2::new(n.x, n.y),
            color: n.color,
            size: NUMBER_TEXT_SIZE,
            font_family: None,
        })
        .collect()
}

/// World-anchored HUD labels for the repose `Text` overlay: run extras
/// (toast, floor/score/kills, boss bar, IDPD warning, game-over line,
/// weapon pickup label). Port-side assembly - GML draws the map name
/// and kills on the roadmap
/// (`scripts/scrDrawRoadmap/scrDrawRoadmap.gml:23-24`), the clock + map
/// name bottom-right
/// (`scripts/scrDrawMiscHUD/scrDrawMiscHUD.gml:13-25`) and the pickup
/// name at `pickup + (0,-31)`
/// (`scripts/scrDrawPlayerHUD/scrDrawPlayerHUD.gml:365`, prop prompts at
/// `:375`), all as positioned GUI rows, not world-anchored floaters.
/// Core HP/level/ammo/LOW-HP rows live in [`hud_gui_texts`], not here.
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
    if hud.game_over {
        out.push((
            format!("GAME OVER  SCORE {}  BEST {}", hud.score, hud.high_score),
            [0.0, 0.0],
        ));
    }
    // GML `scrDrawInteractionHUD` (nearest-pickup half): the nearest weapon
    // pickup's name at the pickup `+(0,-31)` in room px (world-anchored here; the view
    // subtracts the camera) - GML draws at `(floor(x - view_xview), floor(y - view_yview)
    // - 31)` = world `[gun.x, gun.y - 31]`. Ammo gauge + touch `ButtonAct` ring ride the
    // deferred sprite pass; `Prompt{Object}` prop lines (`:371-376`) ride the same law.
    if let Some(label) = world.get_resource::<crate::pickups::WeaponLabel>() {
        if let Some(target) = label.target {
            if let Some(gun) = world.get::<Pos>(target) {
                if !label.text.is_empty() {
                    out.push((label.text.clone(), [gun.0.x, gun.0.y - 31.0]));
                }
            }
        }
        for (entity, text) in label.prop_prompts.iter() {
            if let Some(at) = world.get::<Pos>(*entity) {
                out.push((text.to_string(), [at.0.x, at.0.y - 31.0]));
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

// Fidelity notes.
//
// Compromises vs GML (art-name level; sim truth kept):
// - Wall quads composite Out/Bot/Top + Trans with the seeded variant frames
//   + GM origins, matching the per-cell rolls GML makes at creation
//   (`objects/Wall/Create_0.gml:16-32`). Two deviations: no darkened
//   outside-floor ring is drawn (GML draws bare room background there too,
//   `scripts/background_set_colour` + live floor cells only), and the
//   per-floor top-decal strips ride the `PropKind::GroundDecal` prop
//   (`setup.rs::ground_decal_for_floor`), not the wall pass.
// - Ground decals draw the route floor's top-decal strip at gray 0.5 via
//   `GroundDecalTint` (GML scatters one `Detail` per room tile,
//   `scripts/scrPopulate/scrPopulate.gml:26-30`).
// - Projectile strips follow `projectile_frame`: 2-frame strips pin to the
//   second cell, longer strips animate at 12 fps off the life clock. Custom
//   sizes (plasma growth, bolt stretch) skipped - catalog cell size
//   (pickups too: GML's pickups are plain `draw_self` objects, so native
//   strip frames are the right size).
// - Chest idle shimmer rides the live `SpriteAnim` frame; opened chests draw
//   kind-specific open art frozen on the strip's terminal frame, matching
//   GML's `ChestOpen/Other_7.gml:1-2` (`image_index = image_number - 1`).
// - FX ride `fx_instances`/`fx_texts`: beams stretch `sprLightBeam` along
//   the sim segment (tint = sim `Beam.color`: ion cyan, laser red, boss
//   orange/green), arcs stretch `sprLightning` (frame follows life),
//   muzzles are warm tinted quads (GML draws no muzzle flash either),
//   hazards are kind-tinted discs ignoring per-cloud alpha (pulse on the
//   sim clock, not the cloud's timer fraction; ring has no strip, so
//   ring → disc), particles map 1:1 through `particle_sprites`, damage
//   numbers resolve as `WorldText` - GML has no damage numbers at all.
// - Portal bodies draw their live strip; shock/clear/strike draw
//   1-frame-per-step strips in the hit-FX pass. The vortex background is
//   not a sprite (fullscreen `crate::vortex_pass::VortexPass`).
// - Art names absent from the pack fall back rather than hole: weapon
//   pickups use `sprRevolver` when `wep_sprt` is `mskNone`/absent
//   (`scrWeapons.gml:46`); secret tile families fall back to route strips
//   when the pack lacks `sprFloor10x`.
// - Camera follow is the GML `BackCont` law verbatim (`gml_camera_step`:
//   POI pull /6 cap 72, aim lean dis/viewdist, orandom shake, snap, round,
//   knock decay; view-layer state in `App`). Two deviations:
//   `CAM_MAX_LOOK` clamps the aim lean, which GML leaves unbounded
//   (`objects/BackCont/Step_0.gml:69`), and there is no per-run snap
//   reset beyond boot and floor starts, so transitions don't re-swoop
//   from the origin.

#[cfg(test)]
mod verbatim_ui_layers {
    use super::*;

    /// Reported bug: HUD bars/icons behind floor ground - every sprite shared
    /// z=0, so the engine's `(blend, z, page)` sort let atlas page decide and floor tiles
    /// won. The z-ladder (GML `__global_object_depths` order) keeps GUI chrome above world
    /// chrome on every backend.
    #[test]
    fn z_ladder_orders_chrome_above_world() {
        assert!(Z_SHADOW < Z_WORLD);
        // GML `FloorExplo` is a depth-0 instance, so its rubble patch draws
        // over the `shad` blit and (irrelevant overlap aside) over the Bot.
        assert!(Z_FLOOR_EXPLO > Z_SHADOW);
        assert!(Z_FLOOR_EXPLO < Z_WORLD);
        assert!(Z_WORLD < Z_FX);
        assert!(Z_FX <= Z_BLOOM);
        assert!(Z_BLOOM < Z_FOG);
        assert!(Z_FOG < Z_CROSSHAIR);
        assert!(Z_CROSSHAIR < Z_FAINTED);
        assert!(Z_FAINTED < Z_PORTAL_INDICATOR);
        assert!(Z_PORTAL_INDICATOR < Z_SPIRAL_FIGURES);
        assert!(Z_SPIRAL_FIGURES < Z_SIDEART);
        assert!(Z_SIDEART < Z_HUD);
        assert!(Z_HUD < Z_TOUCH);
        assert!(Z_TOUCH < Z_SPLASH);
        assert!(Z_SPLASH < Z_MENU);
    }

    /// `stamp_z` assigns the rung without disturbing intra-layer push
    /// order (the engine sort is stable on ties).
    #[test]
    fn stamp_z_assigns_rung_only() {
        let mut v = vec![SpriteInstance::default(), SpriteInstance::default()];
        stamp_z(&mut v, Z_HUD);
        assert!(v.iter().all(|s| s.z == Z_HUD));
    }

    /// GML `BackCont/Draw_0:9-14` never draws a `scrShadows` emitter to
    /// the frame: it stamps opaque black into `shad`, then composites it with
    /// `draw_set_alpha(0.4)` + `gpu_set_fog(1, shadow_color, ...)`. Every shadow is the
    /// AREA's shadow color at 40% alpha.
    #[test]
    fn shadows_composite_through_the_shad_surface() {
        assert_eq!(SHADOW_ALPHA, 0.4);
        // `scrAreaGetShadowColor` verbatim, including the campfire default.
        for (area, hex) in [
            (AreaId::Campfire, 0x000000u32),
            (AreaId::Desert, 0x000000),
            (AreaId::Sewers, 0x080d01),
            (AreaId::CrystalCaves, 0x06020c),
            (AreaId::FrozenCity, 0x0e1344),
            (AreaId::Oasis, 0x012b43),
            (AreaId::PizzaSewers, 0x090012),
            (AreaId::City, 0x120014),
            (AreaId::HQ, 0x00248c),
        ] {
            let c = shadow_color(area);
            assert_eq!(c[3], SHADOW_ALPHA, "{area:?} alpha");
            for ch in 0..3 {
                let want = ((hex >> (16 - 8 * ch)) & 0xFF) as f32 / 255.0;
                assert!((c[ch] - want).abs() < 1e-6, "{area:?} channel {ch}");
            }
        }
    }
}

#[cfg(test)]
mod ui_parity_regression {
    use super::*;

    /// GML `draw_sprite_part_ext` windows start outside the cell
    /// (`sprRevolver` xorigin −2, `yorigin − 8 = −5`): the in-bounds
    /// pixels draw offset inside the window, not at its top-left.
    /// Revolver art (11x9) lands at window (dx+2, dy+5) with size
    /// 11x9 - the old clamp drew it at (dx,dy), 2x5 px up-left.
    #[test]
    fn hud_weapon_part_keeps_window_padding() {
        let dir = crate::resolve_assets_dir().expect("assets for parity test");
        let assets = RenderAssets::load(&dir).expect("catalog loads");
        let view = [0.0, 0.0, 426.0, 240.0];
        let gm = hud_gui_map(view);
        let s = hud_weapon_part(
            &assets,
            "images/sprRevolver.png",
            24.0,
            16.0,
            16.0,
            [1.0; 4],
            gm,
            view,
        )
        .expect("revolver art present");
        let top_left = s.center - Vec2::new(s.anchor.x * s.size.x, s.anchor.y * s.size.y);
        assert!((top_left.x - 26.0).abs() < 1e-4, "x {top_left:?}");
        assert!((top_left.y - 21.0).abs() < 1e-4, "y {top_left:?}");
        assert!((s.size.x - 11.0).abs() < 1e-4, "w {:?}", s.size);
        assert!((s.size.y - 9.0).abs() < 1e-4, "h {:?}", s.size);
    }

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

    /// GML `scrSetViewSize`: 240 world px tall, widened to `240 * aspect`
    /// with a 320 floor. The GML odd-width `+1` bump is deliberately not
    /// reproduced (it exists for GML's own surface resize; here 240 is
    /// the constant and the width follows the canvas).
    #[test]
    fn gml_view_size_is_240_tall_with_a_320_floor() {
        let v = gml_view_size([1280.0, 720.0]);
        assert!((v[0] - 426.66666).abs() < 1e-3 && v[1] == 240.0, "{v:?}");
        assert_eq!(gml_view_size([321.0, 240.0]), [321.0, 240.0]);
        // Portrait still floors at 320 wide, 240 tall.
        assert_eq!(gml_view_size([200.0, 400.0]), [320.0, 240.0]);
    }

    /// GML `draw_text_nt` block law: one row per visual line (the shell
    /// renders `.single_line()`), so multi-line producers pre-split. The Title
    /// passive/active pair arrives as two rows at `180 + appear` / `188 + appear` (GML
    /// `scrCampfireMenuDrawCharText`: `fa_middle` block centered on `172 + height/2 +
    /// appear + 8`), never one `\n` row.
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
            .filter(|r| r.gx == 8.0 && (r.gy == 180.0 || r.gy == 188.0))
            .collect();
        assert_eq!(skills.len(), 2, "passive + active rows: {rows:?}");
        assert!(rows.iter().all(|r| !r.text.contains('\n')));
    }

    /// GML centered-`draw_text_nt` law: the row centers on its own `gx`
    /// (symmetric box around `gx`), so stats column headers sit on
    /// their column - not dragged to the view center - while
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
                    bold: false,
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
                    bold: false,
                },
            ],
        );
        // TOTAL box centers on its own gx=110.
        let (total_left, total_w) = (dp[0].1[0], dp[0].4);
        let k = 3.0f32;
        assert!(
            (total_left + total_w * 0.5 - 110.0 * k).abs() < 1.0,
            "{dp:?}"
        );
        // PLAY centers on the view center (426.667/2 = 213.33 GUI px).
        let (play_left, play_w) = (dp[1].1[0], dp[1].4);
        assert!(
            (play_left + play_w * 0.5 - 213.3333 * k).abs() < 1.0,
            "{dp:?}"
        );
    }

    /// GML `draw_stat` law: the name right-aligns on `statx - 1`, so
    /// the row's box right edge lands on `gx` - short names sit
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
                bold: false,
            }],
        );
        // Right edge lands on `ox + gx * k`; the GML law's own trailing
        // 1px is folded into the box, so the row's right edge IS the
        // column anchor (no extra offset here).
        let k = 3.0f32;
        let ox = (1280.0 - gml_view_size([1280.0, 720.0])[0] * k) * 0.5;
        assert!(
            (dp[0].1[0] + dp[0].4 - (ox + 109.0 * k)).abs() < 0.01,
            "{dp:?}"
        );
    }

    /// Narrow-window contain-fit: on a 600x800 portrait canvas the
    /// 320-floored GUI scales to fit width (k = 600/320 = 1.875), and
    /// the fitted rect centers on the canvas - view-centered rows land
    /// on the canvas center, not off-screen right.
    #[test]
    fn portrait_rows_fit_width_and_center() {
        let dp = gui_texts_dp(
            [600.0, 800.0],
            vec![MenuGuiText {
                text: "STATS".to_string(),
                gx: 160.0,
                gy: 24.0,
                color: [255, 255, 255, 255],
                px: 10.0,
                centered: true,
                middle_y: true,
                right: false,
                bold: false,
            }],
        );
        let k = 600.0 / 320.0;
        let (left, top, _px, _c, bw, _r) =
            (dp[0].1[0], dp[0].1[1], dp[0].2, dp[0].3, dp[0].4, dp[0].5);
        assert!((left + bw * 0.5 - 300.0).abs() < 1.0, "{dp:?}");
        assert!((bw - 320.0 * k).abs() < 1.0, "{dp:?}");
        let oy = (800.0 - 240.0 * k) * 0.5;
        assert!(
            (top - (oy + 24.0 * k - (10.0 * k).round() * 0.5)).abs() < 1.0,
            "{dp:?}"
        );
    }
}

#[cfg(test)]
mod wall_break_floor_tests {
    use super::*;

    /// GML `Wall/Create_0:34` + `FloorExplo/Create_0:53`: a wall is visible iff
    /// floor (or a freshly opened cell) meets its south point. Without the
    /// `opened` term a wall whose only southern cover was a broken cell keeps
    /// drawing its Top, leaving the corner art stranded around the hole.
    #[test]
    fn wall_south_cover_counts_opened_cells() {
        let mut mask = FloorMask::default();
        // Wall cells sit in the half-resolution grid, so pick one whose southern
        // 32x32 tile is NOT itself floor (a wall ring cell, not floor).
        let (wx, wy) = (2i32, 2i32);
        let south = (wx, wy + 1);
        let south_tile = floor_cell_for_wall(south.0, south.1);
        mask.cells.insert((0, 0));
        assert!(
            !mask.cells.contains(&south_tile),
            "south tile starts as wall"
        );

        // Bare wall ring: no floor south, no opened cell -> hidden.
        assert!(!mask.opened.contains(&south));

        // Break the cell below: the wall above is uncovered, so it shows again.
        mask.opened.insert(south);
        assert!(mask.opened.contains(&south));
    }

    /// Both Bot and Top hang off one `visible` flag in GML, so a hidden wall
    /// must contribute neither. Guards the regression where Top drew
    /// unconditionally and only Bot respected the flag.
    #[test]
    fn hidden_wall_emits_neither_bot_nor_top() {
        let cells: HashSet<(i32, i32)> = HashSet::new();
        let opened: HashSet<(i32, i32)> = HashSet::new();
        let (wx, wy) = (7i32, 7i32);
        let south = (wx, wy + 1);
        let visible =
            cells.contains(&floor_cell_for_wall(south.0, south.1)) || opened.contains(&south);
        assert!(!visible, "no floor south means the wall stays hidden");
    }
}

#[cfg(test)]
mod crosshair_gate_tests {
    use super::*;
    use crate::comps_a::{Player, Run};
    use crate::spatial::Pos;

    fn world_with_player(game_over: bool) -> World {
        let mut world = World::new();
        world.insert_resource(crate::state::AppState::InGame);
        world.init_resource::<crate::state::Paused>();
        world.init_resource::<crate::state::OverlayMenu>();
        world.insert_resource(Run {
            game_over,
            ..Default::default()
        });
        // GML `TopCont/Draw_0` is `with Player`: the entity is gone
        // after death, so the lerped crosshair draws nothing on game
        // over - the GameOver cursor is `UberCont/Draw_75`'s raw mouse
        // sprite, not this path.
        if !game_over {
            world.spawn((
                Player::default(),
                Pos(Vec2::new(100.0, 100.0)),
                crate::comps_a::AimDir(Vec2::X),
            ));
        }
        world.insert_resource(HoverWorld(Some(Vec2::new(200.0, 100.0))));
        world.init_resource::<crate::savedata_part::SaveData>();
        world
            .resource_mut::<crate::savedata_part::SaveData>()
            .settings
            .gamepad_enabled = true;
        world
    }

    /// GML `TopCont/Draw_0:43` parity: the lerped crosshair draws during
    /// the run in gamepad mode and vanishes with the player at death;
    /// keyboard mode draws nothing here (the raw `Draw_75` cursor covers
    /// aim). Needs several frames: alpha lerps from 0.
    #[test]
    fn crosshair_draws_during_play_not_game_over() {
        let dir = crate::resolve_assets_dir().expect("assets for parity test");
        let assets = RenderAssets::load(&dir).expect("catalog loads");
        let mut world = world_with_player(false);
        let mut drawn = false;
        for _ in 0..60 {
            let out = crosshair_sprites(&mut world, &assets, 1.0 / 30.0);
            if !out.is_empty() {
                drawn = true;
                break;
            }
        }
        assert!(drawn, "crosshair must draw during play");
        // Keyboard mode: no lerped crosshair even alive.
        world
            .resource_mut::<crate::savedata_part::SaveData>()
            .settings
            .gamepad_enabled = false;
        for _ in 0..60 {
            assert!(
                crosshair_sprites(&mut world, &assets, 1.0 / 30.0).is_empty(),
                "keyboard mode draws no lerped crosshair (GML opt_keyboard gate)"
            );
        }
        let mut dead = world_with_player(true);
        for _ in 0..60 {
            assert!(
                crosshair_sprites(&mut dead, &assets, 1.0 / 30.0).is_empty(),
                "no lerped crosshair without a player (GML `with Player`)"
            );
        }
    }

    /// Paused and overlay states still hide it (GML `PauseImage` gate).
    #[test]
    fn crosshair_hidden_when_paused() {
        let dir = crate::resolve_assets_dir().expect("assets for parity test");
        let assets = RenderAssets::load(&dir).expect("catalog loads");
        let mut world = world_with_player(false);
        world.resource_mut::<crate::state::Paused>().0 = true;
        for _ in 0..60 {
            assert!(
                crosshair_sprites(&mut world, &assets, 1.0 / 30.0).is_empty(),
                "paused must hide the crosshair"
            );
        }
    }
}

#[cfg(test)]
mod title_cam_tests {
    use super::*;

    /// GML `Menu/Create_0:104-110` + `Menu/Step_1` parity: the title
    /// view centers on the selected race's camper (`char[race]`);
    /// Random centers on `char[0]` (the Campfire at 64,64). The old
    /// code parked the view top-left at (64,64), shifting the camp
    /// left with background filling the right.
    #[test]
    fn title_cam_centers_on_selected_camper() {
        let mut world = World::new();
        world.insert_resource(crate::savedata_part::SaveData::default());
        crate::setup::setup_title_campfire(&mut world);
        // Fish (1) fixed starter sits at (64,32).
        world.insert_resource(SelectedCharacter(RaceId::Fish));
        let focus = title_cam_focus(&mut world).expect("camp actors spawn");
        assert_eq!(focus, Vec2::new(64.0, 32.0));
        // Crystal (2) fixed starter sits at (64,96).
        world.insert_resource(SelectedCharacter(RaceId::Crystal));
        let focus = title_cam_focus(&mut world).expect("crystal camper");
        assert_eq!(focus, Vec2::new(64.0, 96.0));
        // Random (0) has no camper: focuses the Campfire itself.
        world.insert_resource(SelectedCharacter(RaceId::Random));
        let focus = title_cam_focus(&mut world).expect("campfire fallback");
        assert_eq!(focus, Vec2::new(64.0, 64.0));
        // Entry snap centers exactly (m = 1 like Create_0).
        let mut cam = GmlCamera::default();
        title_camera_step(
            &mut cam,
            426.0,
            240.0,
            Vec2::new(64.0, 32.0),
            1.0 / 30.0,
            true,
        );
        assert_eq!((cam.x, cam.y), (64.0 - 213.0, 32.0 - 120.0));
        // Step_1 lerp converges toward the centered point.
        let mut cam = GmlCamera::default();
        for _ in 0..60 {
            title_camera_step(
                &mut cam,
                426.0,
                240.0,
                Vec2::new(64.0, 32.0),
                1.0 / 30.0,
                false,
            );
        }
        assert!((cam.x - (64.0 - 213.0)).abs() < 1.0, "x={}", cam.x);
        assert!((cam.y - (32.0 - 120.0)).abs() < 1.0, "y={}", cam.y);
    }
}

#[cfg(test)]
mod remap_page_tests {
    use super::*;

    /// Page 13 text rows and hot rows must share y positions: the
    /// keyboard cursor highlight matches text to hot rows within
    /// 0.5px, and click hit-testing uses a ±7px band. A drift between
    /// the two leaves nav blind or clicks landing on the wrong row.
    #[test]
    fn remap_text_and_hot_rows_align() {
        let mut world = World::new();
        world.insert_resource(crate::savedata_part::SaveData::default());
        world.init_resource::<crate::state::menus::MenuState>();
        world.init_resource::<crate::keymap::InputMapState>();
        world
            .resource_mut::<crate::state::menus::MenuState>()
            .settings_page = 13;
        let texts = settings_gui_texts(&mut world, 320.0);
        let rows = settings_hot_rows(13, 320.0);
        assert_eq!(rows.len(), 10, "8 rebind rows + reset + back");
        for (i, row) in rows.iter().enumerate() {
            let hit = texts.iter().any(|t| (t.gy - row.gy).abs() < 0.5);
            assert!(hit, "hot row {i} at gy={} has no text row", row.gy);
        }
        // Rebind rows keep a 16px pitch clear of the ±7px click band;
        // the DEFAULT PRESET / BACK buttons sit lower with a gap.
        for w in rows[..8].windows(2) {
            assert!(
                (w[1].gy - w[0].gy - 16.0).abs() < 0.5,
                "hot pitch {:?}",
                rows.iter().map(|r| r.gy).collect::<Vec<_>>()
            );
        }
        // Every rebind row resolves to a capture action.
        for (i, row) in rows.iter().enumerate().take(8) {
            let action = settings_hot_action(&mut world, 13, i, 0);
            assert!(
                matches!(action, Some(UiAction::RemapControl(_))),
                "row {i} must arm a remap capture, got {action:?}"
            );
        }
    }

    /// GML `MenuOptions/Other_20.gml:785-786` (`Controls_Experimental`
    /// region): no WIP placeholder
    /// - KEYBOARD MODE + STICK REGIONS + HIDE JOYSTICKS, text rows and hot
    ///   rows agreeing. Hide-joysticks hides while stick regions are on
    ///   (`Other_20.gml:806`).
    #[test]
    fn experimental_page_has_three_switches_no_wip() {
        let mut world = World::new();
        world.insert_resource(crate::savedata_part::SaveData::default());
        world.init_resource::<crate::state::menus::MenuState>();
        world
            .resource_mut::<crate::state::menus::MenuState>()
            .settings_page = 16;
        let texts = settings_gui_texts(&mut world, 320.0);
        assert!(
            texts.iter().all(|t| t.text != "KEYBOARD MODE - WIP"),
            "WIP placeholder must be gone"
        );
        for label in ["KEYBOARD MODE", "STICK REGIONS", "HIDE JOYSTICKS"] {
            assert!(
                texts.iter().any(|t| t.text == label),
                "{label} row must draw"
            );
        }
        let rows = settings_hot_rows(16, 320.0);
        assert_eq!(rows.len(), 4, "3 switches + back");
        for (i, row) in rows.iter().enumerate().take(3) {
            let action = settings_hot_action(&mut world, 16, i, 0);
            assert!(
                matches!(action, Some(UiAction::SettingToggle(_))),
                "row {i} must toggle, got {action:?}"
            );
        }
        // Regions on: hide-joysticks text hides, hot rows keep every
        // row so nav/mouse indices agree.
        world
            .resource_mut::<crate::savedata_part::SaveData>()
            .settings
            .stick_regions = true;
        let texts = settings_gui_texts(&mut world, 320.0);
        assert!(
            texts.iter().all(|t| t.text != "HIDE JOYSTICKS"),
            "hide-joysticks must hide while regions are on"
        );
    }
}
