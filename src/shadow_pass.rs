//! GML `shad` surface: the one alpha-1 coverage surface every `scrShadows`
//! emitter stamps, which `BackCont/Draw_0:11-12` then blits ONCE at
//! `draw_set_alpha(0.4)` with a fog that replaces the black silhouette by
//! `shadow_color`.
//!
//! Why the port cannot just draw the stamps: the wall `Out` sprite is 24x32 on
//! a 16px grid, so every stamp overlaps its neighbours 8px sideways and 16px
//! down. Drawn as per-wall 0.4 alpha quads in the world batch those bands
//! accumulated (`1-(1-0.4)^2 = 0.64`, `^3 = 0.784`) where GML shows a flat 0.4,
//! and `SpriteBlend` has no saturating mode to express the union in one pass
//! (Alpha, Additive and Multiply all stack). So the stamps go into an
//! offscreen surface of their own and ONE composite blends over it.
//!
//! Placement: mounted between the two world viewports, so it paints over the
//! floor rungs (`Z_FLOOR`, `Z_GROUND_DETAIL`) and under everything from
//! `Z_FLOOR_EXPLO` up - `Z_FLOOR_EXPLO` deliberately sits above
//! [`crate::render::Z_SHADOW`] so rubble lands on the composited band, as
//! GML's depth-0 `FloorExplo` does.
//!
//! Costs one batch and one fullscreen triangle per frame. `SpriteBatch` owns
//! the atlas bind group and the pipelines and the mask needs both, so this
//! pass drives it through `prepare_with_uploads` / `draw_batch_with_id` -
//! re-exposed from the engine crate for exactly this offscreen-surface case.
//! It shares the world viewport's atlas because the snapshot carries the same
//! [`BatchDesc`] (same `AtlasKey`).

use std::sync::Arc;

use repame_sprite::{
    AtlasUpload, BatchDesc, Camera2d, SpriteBatch, SpriteInstance, draw_batch_with_id,
};
use repose_render_wgpu::{CallbackRenderPass, CallbackResources, ScreenDescriptor, WgpuCallback};

/// Batch id for the mask surface. Stable, so the engine reuses the atlas bind
/// group and instance buffer instead of rebuilding pipelines every frame.
const MASK_BATCH_ID: &str = "nt.shad.mask";

/// Per-frame snapshot of the `shad` surface: the coverage stamps, the camera
/// that places them, and the area's shadow color. Plain data - the pass reads
/// no wall or floor state of its own.
#[derive(Clone)]
pub struct ShadowSnapshot {
    /// Coverage stamps (`scrShadows` wall half), untinted and already placed
    /// in world space by `world_instances_cached`.
    pub stamps: Arc<[SpriteInstance]>,
    pub cam: Camera2d,
    pub world_size: [f32; 2],
    /// Cold-start viewport in dp; must match the world viewport's own so both
    /// batches project the room through the same matrix.
    pub viewport_dp: [f32; 2],
    /// `shadow_color(area)` RGB, linearized for the working space. Alpha is
    /// [`SHADOW_ALPHA`](crate::render::SHADOW_ALPHA), applied once by the
    /// composite rather than baked into the stamps.
    pub color: [f32; 3],
}

/// Mask surface + composite pipeline, kept in the callback's scoped resources
/// so it survives across frames and is reallocated only on a size, format, or
/// sample-count change.
struct MaskGpu {
    /// Multisampled when the surface is; `resolve` is then the samplable copy.
    view: wgpu::TextureView,
    resolve: Option<wgpu::TextureView>,
    /// The sprite batch's pipelines all declare `Depth24PlusStencil8` with
    /// `depth_write_enabled: false` (batch.rs), so the mask pass has to
    /// attach one of a matching format or wgpu rejects the draw.
    depth: wgpu::TextureView,
    size: [u32; 2],
    format: wgpu::TextureFormat,
    samples: u32,
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// The `shad` surface pass.
pub struct ShadowPass {
    snapshot: ShadowSnapshot,
    desc: BatchDesc,
    /// Atlas uploads, held every frame on the same contract as the world
    /// viewport: the frame that introduces a generation can be built before
    /// the view is prepared and then dropped, so the batch must keep seeing
    /// them until it applies that generation itself.
    uploads: Arc<[AtlasUpload]>,
}

impl ShadowPass {
    pub fn new(snapshot: ShadowSnapshot, desc: BatchDesc, uploads: Arc<[AtlasUpload]>) -> Self {
        Self {
            snapshot,
            desc,
            uploads,
        }
    }
}

/// `[color.rgb linear, alpha, target_px.x, target_px.y]`, little-endian.
fn uniform_words(color: [f32; 3], alpha: f32, target_px: [u32; 2]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, w) in [
        color[0],
        color[1],
        color[2],
        alpha,
        target_px[0] as f32,
        target_px[1] as f32,
    ]
    .into_iter()
    .enumerate()
    {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

fn ensure_mask(
    device: &wgpu::Device,
    size: [u32; 2],
    format: wgpu::TextureFormat,
    samples: u32,
    existing: Option<MaskGpu>,
) -> MaskGpu {
    if let Some(gpu) = existing
        && gpu.size == size
        && gpu.format == format
        && gpu.samples == samples
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
        label: Some("nt.shad.mask"),
        size: extent,
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    // Never written (the batch runs `depth_compare: Always`, no write), but
    // the pipelines declare the state so the attachment must exist and match.
    let depth = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("nt.shad.mask.depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());
    // A multisampled attachment is not samplable, so the mask resolves into a
    // single-sample copy that the composite reads.
    let resolve = (samples > 1).then(|| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("nt.shad.mask.resolve"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("nt.shad.sampler"),
        // Nearest, like the atlas: the mask is canvas-sized and the composite
        // maps 1:1, so a pixel reads the texel that covers it.
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nt.shad.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nt.shad.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("nt.shad.pl"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("nt.shad.composite"),
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
                // `dst*(1-a) + shadow_color*a`: the GML surface blit, with the
                // surface's own coverage as `a`.
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        // ERR SRC: The pass we composite into always carries the surface's depth
        // attachment, and a pipeline may not omit the state when the pass
        // provides one - match it and never write, exactly like the sprite
        // batch's own pipelines.
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    });
    let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nt.shad.uniforms"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let sampled = resolve.as_ref().unwrap_or(&view);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nt.shad.bg"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(sampled),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    // The bind group keeps the mask view and sampler alive for the GPU; the
    // uniform buffer is written every frame, so it is held directly.
    MaskGpu {
        view,
        resolve,
        depth,
        size,
        format,
        samples,
        pipeline,
        uniforms,
        bind,
    }
}

impl WgpuCallback for ShadowPass {
    fn resource_key(&self) -> Option<&str> {
        Some("nt.shad.resources")
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
        // Coverage quads. The stamps are already untinted (GML's `c_black` at
        // alpha 1), so each quad's own alpha IS the coverage the surface keeps,
        // and ordinary alpha blending into the cleared surface unions them.
        let mut batch = SpriteBatch::with_id(MASK_BATCH_ID, self.desc);
        batch.set_camera(snap.cam.fit_matrix(snap.viewport_dp, snap.world_size));
        for stamp in snap.stamps.iter() {
            batch.push_sprite(stamp);
        }
        let buffers = batch.prepare_with_uploads(
            device,
            queue,
            encoder,
            screen,
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
        let words = uniform_words(snap.color, crate::render::SHADOW_ALPHA, size);
        queue.write_buffer(&gpu.uniforms, 0, &words);

        if !batch.is_empty() && size[0] > 0 && size[1] > 0 {
            // Written in `prepare`, which repose submits before the main pass,
            // so the composite below always samples THIS frame's stamps.
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("nt.shad.mask.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gpu.view,
                    resolve_target: gpu.resolve.as_ref(),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: Some(0),
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth,
                    // The batch runs `depth_compare: Always` with writes off,
                    // so both attachments are cleared once and thrown away.
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
