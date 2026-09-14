//! nt portal vortex background pass (SpiralCont/Spiral), game content.
//!
//! Layering: the generic fullscreen-pass mechanism lives in
//! `repame-sprite` (`FullscreenPass`); everything nt-specific — the
//! WGSL (`shaders/vortex.wgsl`, ported from nt's `Material2d`), the
//! GameMaker spiral laws baked into it, the 30 Hz tick shapes below —
//! lives here in the main crate next to the [`crate::vortex`] sim state.
//! (It used to be a separate `repame-vortex` crate; nothing else ever
//! reused it, so it was folded in to cut workspace friction.)
//!
//! A future game background brings its own shader + snapshot in its own
//! module and never touches this one.

use repame_sprite::{FullscreenDesc, FullscreenPass, FullscreenTexture, TextureFilter};
use repose_render_wgpu::{CallbackResources, ScreenDescriptor, WgpuCallback};

pub const VORTEX_WISPS: usize = 128;
pub const VORTEX_DEBRIS: usize = 32;
/// Art texture slots: spiral, bolt, debris, proto, idpd, idpd2, star.
pub const VORTEX_TEXTURES: usize = 7;

/// Plain-data snapshot. Field-for-field the nt tick outputs: ring and
/// debris arrays with nt's paddings (`NEG_ONE` wisps, `-1000` debris),
/// per-wisp lightning streams (`[lanim, langle_rad]`, indexed exactly
/// like `wisps`; dead slots hold `lanim = -1`),
/// `glob_a = (ticks, drain_bias, bg_r, bg_g)`,
/// `glob_b = (bg_b, bg_alpha, thresh, kindpacked)`.
/// The game ticks spiral state at 30 Hz (nt law) and hands this over.
pub struct VortexSnapshot {
    pub wisps: [[f32; 4]; VORTEX_WISPS],
    pub debris: [[f32; 4]; VORTEX_DEBRIS],
    /// Per-wisp `Spiral` bolt clock (`lanim`, `langle` in radians).
    pub streams: [[f32; 2]; VORTEX_WISPS],
    /// Venuz `SpiralStar` motes (`[x, y, xscale, frame]`, GML draw pos
    /// already integrated; `x < -100` parks empty slots, same sentinel
    /// convention as debris). Drawn after debris (GML `with` order).
    pub stars: [[f32; 4]; VORTEX_WISPS],
    pub ticks: f32,
    pub drain_bias: f32,
    pub bg_rgb: [f32; 3],
    pub bg_alpha: f32,
    pub thresh: f32,
    pub kindpacked: f32,
    /// Look center + visible extent in wisp coord space. Bevy parity is
    /// `[160, 120, 1920, 1440]` (the 6x quad centered on the camera).
    pub view: [f32; 4],
}

impl VortexSnapshot {
    /// Empty sky: every slot parked, background transparent.
    pub fn empty(view: [f32; 4]) -> Self {
        Self {
            wisps: [[-1.0, -1.0, -1.0, -1.0]; VORTEX_WISPS],
            debris: [[-1000.0, 0.0, 0.0, 0.0]; VORTEX_DEBRIS],
            streams: [[-1.0, 0.0]; VORTEX_WISPS],
            stars: [[-1000.0, 0.0, 0.0, 0.0]; VORTEX_WISPS],
            ticks: 0.0,
            drain_bias: 0.0,
            bg_rgb: [0.0, 0.0, 0.0],
            bg_alpha: 0.0,
            thresh: 2.5,
            kindpacked: 0.0,
            view,
        }
    }

    fn uniform_words(&self) -> Vec<f32> {
        let mut raw = Vec::with_capacity(
            VORTEX_WISPS * 4 + VORTEX_DEBRIS * 4 + VORTEX_WISPS * 4 + VORTEX_WISPS * 4 + 12,
        );
        for w in &self.wisps {
            raw.extend_from_slice(w);
        }
        for d in &self.debris {
            raw.extend_from_slice(d);
        }
        // Padded to vec4: uniform arrays need a 16-byte stride.
        for s in &self.streams {
            raw.extend_from_slice(&[s[0], s[1], 0.0, 0.0]);
        }
        for s in &self.stars {
            raw.extend_from_slice(s);
        }
        raw.extend_from_slice(&[self.ticks, self.drain_bias, self.bg_rgb[0], self.bg_rgb[1]]);
        raw.extend_from_slice(&[self.bg_rgb[2], self.bg_alpha, self.thresh, self.kindpacked]);
        raw.extend_from_slice(&self.view);
        raw
    }
}

/// One art texture upload: tight `w`*`h`*4 RGBA8, row-major top first.
/// `slot` selects the art (0 spiral, 1 bolt, 2 debris, 3 proto,
/// 4 idpd, 5 idpd2, 6 star). Queued on load / area-switch frames only.
#[derive(Clone, Debug)]
pub struct VortexTexture {
    pub slot: u32,
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
}

/// Per-frame snapshot pass. `Send + Sync` for the compositor thread.
/// Rebuilt from the snapshot every frame; uniforms refresh each
/// prepare, texture uploads ride along only on change frames.
pub struct VortexPass {
    pass: FullscreenPass,
    snapshot: VortexSnapshot,
    textures: Vec<VortexTexture>,
}

impl VortexPass {
    pub fn new(snapshot: VortexSnapshot) -> Self {
        Self {
            pass: FullscreenPass::new(
                "vortex",
                include_str!("../shaders/vortex.wgsl"),
                FullscreenDesc {
                    texture_slots: VORTEX_TEXTURES as u32,
                    filter: TextureFilter::Nearest,
                },
            ),
            snapshot,
            textures: Vec::new(),
        }
    }

    /// Queue art uploads (load / area-switch frames only).
    pub fn extend_textures(&mut self, textures: impl IntoIterator<Item = VortexTexture>) {
        self.textures.extend(textures);
    }
}

impl WgpuCallback for VortexPass {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // translate+delegate: the engine owns upload mechanics, the game
        // owns what the bytes mean.
        let uploads: Vec<FullscreenTexture> = self
            .textures
            .iter()
            .map(|t| FullscreenTexture {
                slot: t.slot,
                w: t.w,
                h: t.h,
                rgba: t.rgba.clone(),
            })
            .collect();
        self.pass.prepare_with(
            device,
            queue,
            screen,
            resources,
            &f32_le_bytes(&self.snapshot.uniform_words()),
            &uploads,
        )
    }

    fn paint(
        &self,
        info: repose_core::PaintCallbackInfo,
        rpass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        self.pass.paint(info, rpass, resources);
    }
}

/// f32 words to little-endian bytes (no bytemuck dependency needed).
fn f32_le_bytes(words: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}
