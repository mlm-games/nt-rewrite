// GML `shad` surface composite (`BackCont/Draw_0:11-12`).
//
// Every `scrShadows` emitter stamps its sprite into ONE offscreen surface at
// alpha 1, so a second stamp over the same pixel cannot darken it further -
// the surface holds coverage, not accumulated shade. `BackCont` then blits
// that whole surface once with `draw_set_alpha(0.4)` and a fog that replaces
// the black silhouette with `shadow_color`. Net per pixel:
//     dst' = dst * (1 - 0.4 * cov) + shadow_color * (0.4 * cov)
// which is one alpha blend of a constant colour scaled by coverage.
//
// `mask` is that surface. Its colour is the shadow art's own alpha, so only
// `.a` is read; the sprite batch writes the coverage quads untinted at alpha 1.

struct ShadowGlobals {
    // shadow_color, linear working space (the sRGB decode happens CPU-side
    // like every other tint in this port), plus SHADOW_ALPHA in `a`.
    color: vec4<f32>,
    // Target size in physical px. The mask is canvas-sized, so the fragment's
    // own target-pixel position over this is the mask's uv.
    target_px: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: ShadowGlobals;
@group(0) @binding(1) var mask: texture_2d<f32>;
@group(0) @binding(2) var nearest_smp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let x = f32(i / 2u) * 4.0 - 1.0;
    let y = f32(i % 2u) * 4.0 - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.pos.xy / g.target_px.xy;
    let cov = textureSampleLevel(mask, nearest_smp, uv, 0.0).a * g.color.a;
    // Straight alpha into ALPHA_BLENDING: dst*(1-a) + shadow_color*a, the GML
    // surface blit, for every coverage value including the soft edges the
    // shadow art's own alpha produces.
    return vec4<f32>(g.color.rgb, cov);
}