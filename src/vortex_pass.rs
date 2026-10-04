//! nt portal vortex background pass (SpiralCont/Spiral), game content.
//! Layering: the generic fullscreen-pass mechanism lives in `repame-sprite`
//! (`FullscreenPass`); everything nt-specific - the WGSL
//! (`shaders/vortex.wgsl`, ported from nt's `Material2d`), the GameMaker spiral
//! laws baked into it, the 30 Hz tick shapes below - lives here in the main crate
//! next to the [`crate::vortex`] sim state. (It used to be a separate
//! `repame-vortex` crate; nothing else reused it, so it was folded in to cut
//! workspace friction.)

use std::sync::Arc;

use repame_sprite::{FullscreenDesc, FullscreenPass, FullscreenTextureUpload, TextureFilter};
use repose_render_wgpu::{CallbackRenderPass, CallbackResources, ScreenDescriptor, WgpuCallback};

pub const VORTEX_WISPS: usize = 128;
pub const VORTEX_DEBRIS: usize = 32;
pub const VORTEX_VARDS: usize = 64;
pub const VORTEX_TEXTURES: usize = 14;
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

/// Plain-data snapshot, field-for-field the nt tick output: ring and debris
/// arrays with nt's paddings (`NEG_ONE` wisps, `-1000` debris), per-wisp lightning
/// streams (`[lanim, langle_rad, xscale]`, indexed exactly like `wisps`; dead slots
/// hold `lanim = -1`), then `ticks, drain_bias, bg_r, bg_g, bg_b, bg_alpha,
/// thresh, kindpacked`, the view rect and the portal-bolt flag slots. Spiral
/// state ticks at 30 Hz (nt law).
pub struct VortexSnapshot {
    pub wisps: [[f32; 4]; VORTEX_WISPS],
    pub debris: [[f32; 4]; VORTEX_DEBRIS],
    pub streams: [[f32; 4]; VORTEX_WISPS],
    pub stars: [[f32; 4]; VORTEX_WISPS],
    pub vards: [[f32; 4]; VORTEX_VARDS],
    pub vard_meta: [[f32; 4]; VORTEX_VARDS],
    pub ticks: f32,
    pub drain_bias: f32,
    pub bg_rgb: [f32; 3],
    pub bg_alpha: f32,
    pub thresh: f32,
    pub kindpacked: f32,
    pub draw_bolts: f32,
    pub draw_details: f32,
    pub view: [f32; 4],
}

impl VortexSnapshot {
    fn uniform_words(&self) -> Vec<f32> {
        let mut raw = Vec::with_capacity(
            VORTEX_WISPS * 4
                + VORTEX_DEBRIS * 4
                + VORTEX_WISPS * 4
                + VORTEX_WISPS * 4
                + VORTEX_VARDS * 8
                + 16,
        );
        for w in &self.wisps {
            raw.extend_from_slice(w);
        }
        for d in &self.debris {
            raw.extend_from_slice(d);
        }
        for s in &self.streams {
            raw.extend_from_slice(&[s[0], s[1], s[2], 0.0]);
        }
        for s in &self.stars {
            raw.extend_from_slice(s);
        }
        for v in &self.vards {
            raw.extend_from_slice(v);
        }
        for v in &self.vard_meta {
            raw.extend_from_slice(v);
        }
        raw.extend_from_slice(&[self.ticks, self.drain_bias, self.bg_rgb[0], self.bg_rgb[1]]);
        raw.extend_from_slice(&[self.bg_rgb[2], self.bg_alpha, self.thresh, self.kindpacked]);
        raw.extend_from_slice(&self.view);
        raw.extend_from_slice(&[self.draw_bolts, self.draw_details, 0.0, 0.0]);
        raw
    }
}

/// One art texture upload: tight `w`*`h`*4 RGBA8, row-major top first.
/// `slot` selects the art. Queued on load / area-switch frames only.
#[derive(Clone, Debug)]
pub struct VortexTexture {
    pub slot: u32,
    pub w: u32,
    pub h: u32,
    pub rgba: Arc<[u8]>,
    pub generation: u64,
}

impl FullscreenTextureUpload for VortexTexture {
    fn slot(&self) -> u32 {
        self.slot
    }

    fn width(&self) -> u32 {
        self.w
    }

    fn height(&self) -> u32 {
        self.h
    }

    fn rgba(&self) -> &[u8] {
        self.rgba.as_ref()
    }

    fn generation(&self) -> Option<u64> {
        Some(self.generation)
    }
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

    /// Queue art uploads. The pass keeps no history.
    pub fn extend_textures(&mut self, textures: impl IntoIterator<Item = VortexTexture>) {
        self.textures.extend(textures);
    }
}

impl WgpuCallback for VortexPass {
    fn resource_key(&self) -> Option<&str> {
        self.pass.resource_key()
    }

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
        self.pass.prepare_with(
            device,
            queue,
            screen,
            resources,
            &f32_le_bytes(&self.snapshot.uniform_words()),
            self.textures.as_slice(),
        )
    }

    fn paint(
        &self,
        info: repose_core::PaintCallbackInfo,
        rpass: &mut CallbackRenderPass<'_, '_>,
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
