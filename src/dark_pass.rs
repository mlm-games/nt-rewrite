//! GML `TopCont` darkness (the "short view distance" in the sewers):
//! `scrDarkness` paints a mask of the run's lit areas, and `TopCont/Draw_0`
//! blits it over the world. The port reproduces the mask with an offscreen
//! surface exactly like [`crate::shadow_pass`] does for the `shad` surface,
//! but the composite is a MULTIPLY of the inverted mask: black = full dark,
//! gray = half dark, white = untouched. `scrDarkness` subtracts white/gray/
//! black instead, which is the same ramp in the other direction and cannot
//! be expressed with the sprite batch's Alpha/Add pipelines anyway.
//!
//! The pass runs after the world viewport and before the HUD, like
//! `TopCont`, and only for the even areas GML darkens (`TopCont/Create_0`:
//! sewers, caves, labs, cursed caves; not the HQ).

use std::sync::Arc;

use repame_actors::ActorDraw;
use repame_sprite::{AtlasUpload, BatchDesc, Camera2d, SpriteBatch, draw_batch_with_id};
use repose_render_wgpu::{CallbackRenderPass, CallbackResources, ScreenDescriptor, WgpuCallback};

/// Batch id for the darkness mask. Stable, so the engine reuses the atlas
/// bind group and instance buffer between frames.
const MASK_BATCH_ID: &str = "nt.dark.mask";

/// Single-sample for the same reason as the shadow mask: glow's WebGL2
/// backend cannot allocate a multisampled texture, and the composite maps
/// 1:1.
const MASK_SAMPLES: u32 = 1;

/// Luminance the gray discs and the Eyes clear multiply by. `c_gray` is
/// `#808080`; in the port's linear working space that is about `0.5^2.2`.
pub const DARK_DIM: f32 = 0.22;

#[derive(Clone)]
pub struct DarkSnapshot {
    /// Light stamps: gray discs first, then the white cores, already placed
    /// in world space.
    pub stamps: Arc<[ActorDraw]>,
    pub cam: Camera2d,
    pub world_size: [f32; 2],
    /// Cold-start viewport in dp; must match the world viewport's own.
    pub viewport_dp: [f32; 2],
    /// Mask clear colour: black on a normal run, [`DARK_DIM`] for Eyes.
    pub clear: [f64; 3],
}

struct MaskGpu {
    view: wgpu::TextureView,
    depth: wgpu::TextureView,
    size: [u32; 2],
    format: wgpu::TextureFormat,
    composite_samples: u32,
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
}

pub struct DarkPass {
    snapshot: DarkSnapshot,
    desc: BatchDesc,
    uploads: Arc<[AtlasUpload]>,
}

impl DarkPass {
    pub fn new(snapshot: DarkSnapshot, desc: BatchDesc, uploads: Arc<[AtlasUpload]>) -> Self {
        Self {
            snapshot,
            desc,
            uploads,
        }
    }
}

fn ensure_mask(
    device: &wgpu::Device,
    size: [u32; 2],
    format: wgpu::TextureFormat,
    composite_samples: u32,
    existing: Option<MaskGpu>,
) -> MaskGpu {
    if let Some(gpu) = existing
        && gpu.size == size
        && gpu.format == format
        && gpu.composite_samples == composite_samples
    {
        return gpu;
    }
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let extent = wgpu::Extent3d {
        width: size[0].max(1),
        height: size[1].max(1),
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("nt.dark.mask"),
        size: extent,
        mip_level_count: 1,
        sample_count: MASK_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("nt.dark.mask.depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: MASK_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("nt.dark.sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nt.dark.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nt.dark.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/dark.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("nt.dark.pl"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("nt.dark.composite"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                // `dst * mask`: the inverted subtract blit.
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Dst,
                        dst_factor: wgpu::BlendFactor::Zero,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: composite_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nt.dark.bg"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    MaskGpu {
        view,
        depth,
        size,
        format,
        composite_samples,
        pipeline,
        bind,
    }
}

impl WgpuCallback for DarkPass {
    fn resource_key(&self) -> Option<&str> {
        Some("nt.dark.resources")
    }

    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let snap = &self.snapshot;
        let mut batch = SpriteBatch::with_id(MASK_BATCH_ID, self.desc);
        batch.set_camera(snap.cam.fit_matrix(snap.viewport_dp, snap.world_size));
        for stamp in snap
            .stamps
            .iter()
            .filter_map(crate::render::to_sprite_instance)
        {
            batch.push_sprite(&stamp);
        }
        let mask_screen = ScreenDescriptor {
            sample_count: MASK_SAMPLES,
            ..*screen
        };
        let buffers = batch.prepare_with_uploads(
            device,
            queue,
            encoder,
            &mask_screen,
            resources,
            self.uploads.as_ref(),
        );
        let size = screen.size_in_pixels;
        let gpu = ensure_mask(
            device,
            size,
            screen.target_format,
            screen.sample_count,
            resources.remove::<MaskGpu>(),
        );
        if size[0] > 0 && size[1] > 0 {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("nt.dark.mask.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gpu.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: snap.clear[0],
                            g: snap.clear[1],
                            b: snap.clear[2],
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Discard,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            draw_batch_with_id(MASK_BATCH_ID, &mut rpass, resources);
        }
        resources.insert(gpu);
        buffers
    }

    fn paint(
        &self,
        _info: repose_core::PaintCallbackInfo,
        rpass: &mut CallbackRenderPass<'_, '_>,
        resources: &CallbackResources,
    ) {
        let Some(gpu) = resources.get::<MaskGpu>() else {
            return;
        };
        rpass.set_pipeline(&gpu.pipeline);
        rpass.set_bind_group(0, &gpu.bind, &[]);
        rpass.draw(0..3, 0..1);
    }
}
