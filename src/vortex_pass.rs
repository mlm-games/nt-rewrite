//! nt portal vortex background (GML `SpiralCont`/`Spiral`), game content.
//!
//! Drawn as ordered instanced quads through [`SpriteBatch`]: the art lives in
//! this pass's own atlas layer (nearest filtered, one cell per strip frame) and
//! [`VortexBatch::push_snapshot`] emits the objects in GML `scrDrawSpiral`
//! order. World units are GUI px, so the batch camera frames the GML view and
//! the pillarbox clips the result without a screen-space quad.
//!
//! Layering: the generic batch mechanism lives in `repame-sprite`; everything
//! nt-specific - the GameMaker spiral laws, the atlas cut, the 30 Hz tick
//! shapes in [`crate::vortex`] - lives here next to that sim state. (The WGSL
//! that used to sit here, a fullscreen quad that walked every vortex object per
//! fragment, is gone: it cost `screen pixels x 352 objects` per frame.)

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use repame_sprite::{
    AtlasUpload, BatchDesc, SpriteBatch, SpriteBlend, TextureFilter, screen_camera,
};
use repose_render_wgpu::{CallbackRenderPass, CallbackResources, ScreenDescriptor, WgpuCallback};

pub const VORTEX_WISPS: usize = 128;
pub const VORTEX_DEBRIS: usize = 32;
pub const VORTEX_VARDS: usize = 64;
pub const VARD_VARIANTS: [&str; 7] = [
    "images/sprBanditHurt.png",
    "images/sprRatHurt.png",
    "images/sprCarIdle.png",
    "images/sprSpiderHurt.png",
    "images/sprFrozenCar.png",
    "images/sprFreak1Hurt.png",
    "images/sprSlice.png",
];
pub const VARD_CELL_SIZES: [(f32, f32); 7] = [
    (24.0, 24.0),
    (24.0, 24.0),
    (32.0, 32.0),
    (24.0, 24.0),
    (32.0, 32.0),
    (24.0, 24.0),
    (16.0, 8.0),
];
pub const VARD_FRAME_COUNTS: [usize; 7] = [3, 3, 1, 3, 1, 3, 7];

pub fn vard_slot(path: &str) -> Option<usize> {
    VARD_VARIANTS
        .iter()
        .position(|candidate| *candidate == path)
}

/// Atlas layer edge. One layer holds every strip cut into single frames; the
/// widest single cell is the 176px bolt frame, so 1024 covers the whole set
/// with room to spare (the strips unsplit would need 2048 for the 1056px bolt
/// alone).
const LAYER: u32 = 1024;
const SPIRAL_FRAMES: usize = 2;
const PROTO_FRAMES: usize = 2;
const STAR_FRAMES: usize = 2;
const DEBRIS_FRAMES: usize = 4;
const BOLT_FRAMES: usize = 6;

/// GML `scrDrawSpiral:32-33` scales the 64px wisp art by `image_xscale * 10`
/// (the 128px IDPD strip uses the same factor).
const WISP_ART_PX: f32 = 64.0 * 10.0;
const IDPD_ART_PX: f32 = 128.0 * 10.0;
/// GML `scrDrawSpiral:28-29` draws `sprPortalLightning` at plain
/// `image_xscale` on both axes.
const BOLT_ART_PX: f32 = 176.0;
/// GML `scrDrawSpiral:45-46` draws `sprSpiralStar` at plain `image_xscale`.
const STAR_ART_PX: f32 = 3.0;
/// GML `SpiralCont/Step_0` copies `image_angle` (degrees) onto every wisp;
/// `scrDrawSpiral:32` draws the art at `image_angle + 45` and the bolt at the
/// bare `image_angle`, so the art rotation the snapshot carries has the skew
/// removed again here.
const ART_SKEW_RAD: f32 = std::f32::consts::FRAC_PI_4;
/// `image_speed = 2` at 30 steps/s in the port's frame clock: the 2-frame wisp
/// strip advances every 15 ticks.
const SPIRAL_FRAME_TICKS: f32 = 15.0;

/// Process-wide atlas generation counter. `SpriteBatch` keys its atlas on the
/// generation and only replays uploads when it moves, so a generation must
/// never repeat for different pixels.
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

fn next_generation() -> u64 {
    NEXT_GENERATION.fetch_add(1, Ordering::Relaxed)
}

const CENTER: [f32; 2] = [0.5, 0.5];
const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// Plain-data snapshot, field-for-field the nt tick output: ring and debris
/// arrays with nt's paddings (`NEG_ONE` wisps, `-1000` debris), per-wisp
/// lightning streams (`[lanim, langle_rad, xscale]`, indexed exactly like
/// `wisps`), then `ticks, bg_rgb, bg_alpha, thresh, kindpacked`, the view rect
/// and the portal-bolt/detail flags. Spiral state ticks at 30 Hz (nt law).
pub struct VortexSnapshot {
    pub wisps: [[f32; 4]; VORTEX_WISPS],
    pub debris: [[f32; 4]; VORTEX_DEBRIS],
    pub streams: [[f32; 3]; VORTEX_WISPS],
    pub stars: [[f32; 4]; VORTEX_WISPS],
    pub vards: [[f32; 4]; VORTEX_VARDS],
    pub vard_meta: [[f32; 4]; VORTEX_VARDS],
    pub ticks: u32,
    pub bg_rgb: [f32; 3],
    pub bg_alpha: f32,
    pub thresh: f32,
    pub kindpacked: f32,
    pub draw_bolts: f32,
    pub draw_details: f32,
    pub view: [f32; 4],
}

/// One atlas cell: where a single strip frame lives in the layer.
#[derive(Clone, Copy)]
struct Cell {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// The vortex's own atlas: every strip cut into one cell per frame, the
/// decoded pixels cached per cell, and the pending upload set. An area switch
/// re-cuts only the debris strip (`sprDebris` + `GameCont.area`) and replays
/// the cached bytes; `SpriteBatch` only applies uploads when
/// `BatchDesc::uploads_gen` moves, hence [`Self::generation`].
pub struct VortexArt {
    cells: Vec<Cell>,
    pixels: Vec<Vec<u8>>,
    rects: Vec<[f32; 4]>,
    uploads: Vec<AtlasUpload>,
    generation: u64,
    pub spiral: usize,
    pub bolt: usize,
    pub debris: usize,
    pub proto: usize,
    pub idpd: usize,
    pub idpd2: usize,
    pub star: usize,
    pub vards: [usize; 7],
    pub white: usize,
    /// Frame width the debris cells were cut with (`sprDebris105` uses 16px
    /// frames, every other area 8px).
    pub debris_px: f32,
}

impl VortexArt {
    pub fn load(dir: &Path, gml_area: u8) -> Self {
        let mut cells: Vec<(u32, u32, Vec<u8>)> = Vec::new();
        let spiral = add(&mut cells, cut(dir, "images/sprSpiral.png", SPIRAL_FRAMES));
        let bolt = add(
            &mut cells,
            cut(dir, "images/sprPortalLightning.png", BOLT_FRAMES),
        );
        let debris = add(&mut cells, cut(dir, &debris_stem(gml_area), DEBRIS_FRAMES));
        let proto = add(
            &mut cells,
            cut(dir, "images/sprSpiralProto.png", PROTO_FRAMES),
        );
        let idpd = add(&mut cells, cut(dir, "images/sprSpiralIDPD.png", 1));
        let idpd2 = add(&mut cells, cut(dir, "images/sprSpiralIDPD2.png", 1));
        let star = add(
            &mut cells,
            cut(dir, "images/sprSpiralStar.png", STAR_FRAMES),
        );
        let mut vards = [0usize; VARD_VARIANTS.len()];
        for (slot, path) in VARD_VARIANTS.into_iter().enumerate() {
            vards[slot] = add(&mut cells, cut(dir, path, VARD_FRAME_COUNTS[slot]));
        }
        let white = add(&mut cells, vec![(1, 1, vec![255; 4])]);

        let debris_px = cells[debris].0 as f32;
        let pixels = cells.iter().map(|(_, _, px)| px.clone()).collect();
        let mut art = Self {
            cells: shelf(&mut cells),
            pixels,
            rects: Vec::new(),
            uploads: Vec::new(),
            generation: next_generation(),
            spiral,
            bolt,
            debris,
            proto,
            idpd,
            idpd2,
            star,
            vards,
            white,
            debris_px,
        };
        art.relayout();
        art
    }

    /// Re-cut the per-area debris strip (`sprDebris` + `GameCont.area`) into a
    /// fresh atlas. Only the strip is decoded; the other twelve cells replay
    /// from the pixel cache.
    pub fn with_area(self: &Arc<Self>, dir: &Path, gml_area: u8) -> Arc<Self> {
        let mut art = Self {
            cells: self.cells.clone(),
            pixels: self.pixels.clone(),
            rects: Vec::new(),
            uploads: Vec::new(),
            generation: next_generation(),
            spiral: self.spiral,
            bolt: self.bolt,
            debris: self.debris,
            proto: self.proto,
            idpd: self.idpd,
            idpd2: self.idpd2,
            star: self.star,
            vards: self.vards,
            white: self.white,
            debris_px: 0.0,
        };
        for (frame, (w, h, rgba)) in cut(dir, &debris_stem(gml_area), DEBRIS_FRAMES)
            .into_iter()
            .enumerate()
        {
            let slot = art.debris + frame;
            art.cells[slot].w = w;
            art.cells[slot].h = h;
            art.pixels[slot] = rgba;
        }
        art.debris_px = art.cells[art.debris].w as f32;
        art.relayout();
        Arc::new(art)
    }

    pub fn rect(&self, cell: usize) -> [f32; 4] {
        self.rects[cell]
    }

    fn relayout(&mut self) {
        let packed = shelf_cells(&mut self.cells);
        self.cells = packed;
        let inv = 1.0 / LAYER as f32;
        self.rects = self
            .cells
            .iter()
            .map(|c| {
                [
                    c.x as f32 * inv,
                    c.y as f32 * inv,
                    (c.x + c.w) as f32 * inv,
                    (c.y + c.h) as f32 * inv,
                ]
            })
            .collect();
        self.uploads = (0..self.cells.len())
            .map(|i| AtlasUpload {
                page: 0,
                x: self.cells[i].x,
                y: self.cells[i].y,
                w: self.cells[i].w,
                h: self.cells[i].h,
                rgba: self.pixels[i].clone(),
            })
            .collect();
    }
}

/// GML `SpiralDebris/Create_0:12`: the mote strip is `sprDebris` plus the FULL
/// GML area number, not the art variant. Campfire (area 0) reuses
/// `sprDebris0`; secret areas use their own (`sprDebris101` ...). Never
/// `sprDebris1` here.
fn debris_stem(gml_area: u8) -> String {
    format!("images/sprDebris{gml_area}.png")
}

/// Decode one strip and cut it into `frames` equal columns. A missing file
/// falls back to a transparent texel per frame and says so, so broken art
/// drops out instead of painting opaque white blocks over the frame.
fn cut(dir: &Path, path: &str, frames: usize) -> Vec<(u32, u32, Vec<u8>)> {
    let decoded = crate::render::decode_png(&dir.join(path));
    let (w, h, rgba) = match decoded {
        Ok(image) => image,
        Err(err) => {
            log::warn!("vortex art unavailable: {path}: {err}");
            return (0..frames).map(|_| (1, 1, vec![0, 0, 0, 0])).collect();
        }
    };
    let fw = (w / frames as u32).max(1);
    (0..frames)
        .map(|frame| {
            let row = (fw * 4) as usize;
            let mut px = Vec::with_capacity(row * h as usize);
            for y in 0..h {
                let start = ((y * w + frame as u32 * fw) * 4) as usize;
                let end = start + row;
                px.extend_from_slice(&rgba[start.min(rgba.len())..end.min(rgba.len())]);
            }
            (fw, h, px)
        })
        .collect()
}

fn add(cells: &mut Vec<(u32, u32, Vec<u8>)>, cut: Vec<(u32, u32, Vec<u8>)>) -> usize {
    let base = cells.len();
    cells.extend(cut);
    base
}

/// Shelf pack into one [`LAYER`] square, tallest cells first.
fn shelf(cells: &mut [(u32, u32, Vec<u8>)]) -> Vec<Cell> {
    let mut flat: Vec<Cell> = cells
        .iter()
        .map(|&(w, h, _)| Cell { x: 0, y: 0, w, h })
        .collect();
    flat = shelf_cells(&mut flat);
    flat
}

fn shelf_cells(cells: &mut [Cell]) -> Vec<Cell> {
    let mut order: Vec<usize> = (0..cells.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(cells[i].h));
    let (mut x, mut y, mut shelf) = (0u32, 0u32, 0u32);
    for i in order {
        if x + cells[i].w > LAYER {
            x = 0;
            y += shelf;
            shelf = 0;
        }
        cells[i].x = x;
        cells[i].y = y;
        x += cells[i].w;
        shelf = shelf.max(cells[i].h);
    }
    cells.to_vec()
}

/// Per-frame snapshot pass, rebuilt from the snapshot every rendered frame.
/// The atlas uploads ride along on the frames where `upload_pending` is set
/// (asset load and every area switch) and are dropped once `prepare` has
/// replayed them.
pub struct VortexBatch {
    batch: SpriteBatch,
    art: Arc<VortexArt>,
    upload_pending: Arc<AtomicBool>,
}

/// One drawn quad in `scrDrawSpiral` order: world centre, world size,
/// rotation radians, atlas cell, tint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VortexQuad {
    pub center: [f32; 2],
    pub size: [f32; 2],
    pub rot: f32,
    pub cell: usize,
    pub tint: [f32; 4],
}

impl VortexBatch {
    /// `view` is the GUI rect (`[cx, cy, w, h]`) the mounted node rect shows.
    /// The spiral's own world space is that rect, so the camera maps it 1:1
    /// onto the node - never the game camera, which during a floor transition
    /// frames the room instead and would throw the vortex off-centre.
    pub fn new(view: [f32; 2], art: Arc<VortexArt>, upload_pending: Arc<AtomicBool>) -> Self {
        let mut batch = SpriteBatch::with_id(
            "vortex",
            BatchDesc {
                layer_size: LAYER,
                layers: 1,
                filter: TextureFilter::Nearest,
                uploads_gen: art.generation,
            },
        );
        batch.set_camera(screen_camera([view[0], view[1]]));
        if upload_pending.load(Ordering::Relaxed) {
            batch.extend_uploads(art.uploads.iter().map(|u| AtlasUpload { ..u.clone() }));
        }
        Self {
            batch,
            art,
            upload_pending,
        }
    }

    /// The frame's quads in `scrDrawSpiral` order: background, then the wisps
    /// oldest to newest (GML `with` walks instances in creation order and the
    /// draw script ignores depth), then the detail passes on top. `z` rises
    /// with the list, and every quad is alpha-blended, so the batch's
    /// `(blend, z, insertion)` sort preserves this order.
    pub fn push_snapshot(&mut self, snap: &VortexSnapshot) {
        for (order, q) in self.quads(snap).into_iter().enumerate() {
            let uv = self.art.rect(q.cell);
            self.batch.push_blended(
                q.center,
                q.size,
                q.rot,
                CENTER,
                false,
                false,
                [uv[0], uv[1]],
                [uv[2], uv[3]],
                q.tint,
                0,
                order as f32,
                SpriteBlend::Alpha,
            );
        }
    }

    fn quads(&self, snap: &VortexSnapshot) -> Vec<VortexQuad> {
        let art = self.art.as_ref();
        let mut out: Vec<VortexQuad> = Vec::with_capacity(1024);
        let mut push = |center: [f32; 2], size: [f32; 2], rot: f32, cell: usize, tint: [f32; 4]| {
            out.push(VortexQuad {
                center,
                size,
                rot,
                cell,
                tint,
            });
        };
        // GML `scrDrawSpiral:4-14`: every caller opens with `draw_clear(c_black)`
        // except `Menu`, which the port spells as `bg_alpha == 0`.
        let view = snap.view;
        if snap.bg_alpha > 0.0 {
            push(
                [view[0], view[1]],
                [view[2], view[3]],
                0.0,
                art.white,
                [
                    snap.bg_rgb[0],
                    snap.bg_rgb[1],
                    snap.bg_rgb[2],
                    snap.bg_alpha,
                ],
            );
        }

        let kind = snap.kindpacked as u32 % 4;
        let bolts = snap.draw_bolts > 0.5;
        let base = snap.ticks;
        for k in 0..VORTEX_WISPS as u32 {
            let Some(birth) = base.checked_sub(VORTEX_WISPS as u32 - 1 - k) else {
                continue;
            };
            if birth < 1 {
                continue;
            }
            let slot = ((birth - 1) as usize) % VORTEX_WISPS;
            let wisp = snap.wisps[slot];
            if wisp[2] < 0.0 {
                continue;
            }
            let age = base as f32 - wisp[2];
            if age < 0.0 {
                continue;
            }
            let stream = snap.streams[slot];
            let (lanim, langle, xs) = (stream[0], stream[1], stream[2]);
            if xs <= 0.0 {
                continue;
            }
            let bolt_on = lanim > 0.0 && lanim < BOLT_FRAMES as f32;
            if xs > snap.thresh && !bolt_on {
                continue;
            }
            let at = [wisp[0], wisp[1]];
            let rot = wisp[3].abs();

            if bolts && bolt_on {
                let frame = (lanim.floor() as usize).min(BOLT_FRAMES - 1);
                let cell = art.bolt + frame;
                // GML `scrDrawSpiral:28-29`: plain `image_xscale` on both axes,
                // centred on the wisp, at `image_angle + langle` (the snapshot
                // carries `image_angle + 45`, so the skew comes off again).
                let bolt_rot = rot - ART_SKEW_RAD + langle;
                let size = [BOLT_ART_PX * xs, BOLT_ART_PX * xs];
                push(at, size, bolt_rot, cell, WHITE);
                shade(&mut push, at, size, bolt_rot, cell, 0.4 - xs * 0.5);
            }

            // GML `SpiralCont/Step_0` swaps `sprite_index` to `sprSpiralIDPD2`
            // without touching `image_angle`; the port carries that variant in
            // the rotation sign.
            let (cell, art_px) = if kind == 2 {
                let cell = if wisp[3] < 0.0 { art.idpd2 } else { art.idpd };
                (cell, IDPD_ART_PX)
            } else {
                let frame = ((age / SPIRAL_FRAME_TICKS).floor() as usize) % SPIRAL_FRAMES;
                let strip = if kind == 1 { art.proto } else { art.spiral };
                (strip + frame, WISP_ART_PX)
            };
            // GML `scrDrawSpiral:32-33`: `image_xscale * 10` on both axes.
            let size = [art_px * xs, art_px * xs];
            push(at, size, rot, cell, WHITE);
            shade(&mut push, at, size, rot, cell, 0.8 - xs);
        }

        if snap.draw_details <= 0.5 {
            return out;
        }

        for mote in &snap.debris {
            if mote[0] < -100.0 {
                continue;
            }
            let xs = mote[3].fract() * 32.0;
            if xs <= 0.0 {
                continue;
            }
            let frame = (mote[3].floor() as usize).min(DEBRIS_FRAMES - 1);
            let cell = art.debris + frame;
            let size = [art.debris_px * xs, art.debris_px * xs];
            push([mote[0], mote[1]], size, mote[2], cell, WHITE);
            shade(&mut push, [mote[0], mote[1]], size, mote[2], cell, 1.0 - xs);
        }
        for i in 0..VORTEX_VARDS {
            let mote = snap.vards[i];
            if mote[0] < -100.0 {
                continue;
            }
            let meta = snap.vard_meta[i];
            let xs = mote[3];
            if xs <= 0.001 || meta[1] <= 0.0 || meta[2] <= 0.0 {
                continue;
            }
            let slot = (meta[3] as usize).min(VARD_VARIANTS.len() - 1);
            let frame = (meta[0] as usize).min(VARD_FRAME_COUNTS[slot] - 1);
            let cell = art.vards[slot] + frame;
            let size = [meta[1] * xs, meta[2] * xs];
            push([mote[0], mote[1]], size, mote[2], cell, WHITE);
            shade(&mut push, [mote[0], mote[1]], size, mote[2], cell, 1.0 - xs);
        }
        for star in &snap.stars {
            if star[0] < -100.0 {
                continue;
            }
            let xs = star[2];
            if xs <= 0.001 {
                continue;
            }
            let frame = (star[3] as usize).min(STAR_FRAMES - 1);
            let cell = art.star + frame;
            let size = [STAR_ART_PX * xs, STAR_ART_PX * xs];
            // GML leaves the star's `image_angle` at 0 (its Step never turns it).
            push([star[0], star[1]], size, 0.0, cell, WHITE);
            shade(&mut push, [star[0], star[1]], size, 0.0, cell, 1.0 - xs);
        }
        out
    }
}

/// GML `scrDrawSpiral` draws every object twice: art in `c_white`, then the
/// same quad in `c_black` at a partial alpha.
fn shade(
    push: &mut impl FnMut([f32; 2], [f32; 2], f32, usize, [f32; 4]),
    center: [f32; 2],
    size: [f32; 2],
    rot: f32,
    cell: usize,
    alpha: f32,
) {
    let alpha = alpha.clamp(0.0, 1.0);
    if alpha <= 0.001 {
        return;
    }
    push(center, size, rot, cell, [0.0, 0.0, 0.0, alpha]);
}

impl WgpuCallback for VortexBatch {
    fn resource_key(&self) -> Option<&str> {
        self.batch.resource_key()
    }

    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        self.upload_pending.store(false, Ordering::Relaxed);
        self.batch
            .prepare(device, queue, encoder, screen, resources)
    }

    fn paint(
        &self,
        info: repose_core::PaintCallbackInfo,
        rpass: &mut CallbackRenderPass<'_, '_>,
        resources: &CallbackResources,
    ) {
        self.batch.paint(info, rpass, resources);
    }
}

#[cfg(test)]
mod vortex_art_tests {
    use super::*;

    fn assets() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
        dir.join("images/sprSpiral.png").is_file().then_some(dir)
    }

    /// Every configured strip must resolve at its documented frame geometry.
    /// This is the guard for the loader bug that appended a second `.png` to
    /// the variant paths (`sprBanditHurt.png.png`), which silently dropped all
    /// seven variant-debris cells to the transparent fallback.
    #[test]
    fn every_configured_strip_resolves() {
        let Some(dir) = assets() else { return };
        let art = VortexArt::load(&dir, 1);
        let expect = |base: usize, frames: usize, fw: u32, fh: u32| {
            for f in 0..frames {
                let cell = &art.cells[base + f];
                assert_eq!(
                    (cell.w, cell.h),
                    (fw, fh),
                    "cell {base}+{f} resolved at {}x{}, expected {fw}x{fh}",
                    cell.w,
                    cell.h
                );
                assert!(
                    art.pixels[base + f].iter().any(|b| *b != 0),
                    "cell {base}+{f} decoded to a fully transparent texel"
                );
            }
        };
        expect(art.spiral, SPIRAL_FRAMES, 64, 64);
        expect(art.bolt, BOLT_FRAMES, 176, 176);
        expect(art.debris, DEBRIS_FRAMES, 8, 8);
        expect(art.proto, PROTO_FRAMES, 64, 64);
        expect(art.idpd, 1, 128, 128);
        expect(art.idpd2, 1, 128, 128);
        expect(art.star, STAR_FRAMES, 3, 3);
        for slot in 0..VARD_VARIANTS.len() {
            let (w, h) = VARD_CELL_SIZES[slot];
            expect(art.vards[slot], VARD_FRAME_COUNTS[slot], w as u32, h as u32);
        }
        assert_eq!((art.cells[art.white].w, art.cells[art.white].h), (1, 1));
    }

    /// Jungle debris uses 16px frames, so an area switch must re-pack and the
    /// debris cells must land on the new strip rather than keep the old rects.
    #[test]
    fn area_switch_recuts_the_debris_strip() {
        let Some(dir) = assets() else { return };
        let desert = Arc::new(VortexArt::load(&dir, 1));
        assert_eq!(desert.debris_px, 8.0);
        let jungle = desert.with_area(&dir, 105);
        assert_eq!(jungle.debris_px, 16.0);
        assert_eq!(
            (jungle.cells[jungle.debris].w, jungle.cells[jungle.debris].h),
            (16, 16)
        );
        assert_ne!(jungle.rects, desert.rects, "relayout must republish rects");
        assert_ne!(
            jungle.generation, desert.generation,
            "a fresh generation is what replays the uploads"
        );
    }

    /// Shelf packing must keep every cell inside the layer without overlaps.
    #[test]
    fn packing_fits_one_layer_without_overlap() {
        let Some(dir) = assets() else { return };
        let art = VortexArt::load(&dir, 1);
        let mut claimed = vec![false; (LAYER * LAYER) as usize];
        for c in &art.cells {
            assert!(
                c.x + c.w <= LAYER && c.y + c.h <= LAYER,
                "cell {}x{}+{}+{} escapes the {LAYER} layer",
                c.w,
                c.h,
                c.x,
                c.y
            );
            for y in c.y..c.y + c.h {
                for x in c.x..c.x + c.w {
                    let i = (y * LAYER + x) as usize;
                    assert!(!claimed[i], "cell overlap at {x},{y}");
                    claimed[i] = true;
                }
            }
            let uv = art.rects[art
                .cells
                .iter()
                .position(|o| o.x == c.x && o.y == c.y && o.w == c.w && o.h == c.h)
                .expect("cell present")];
            assert_eq!(uv[0], c.x as f32 / LAYER as f32);
            assert_eq!(uv[2], (c.x + c.w) as f32 / LAYER as f32);
        }
    }
}
