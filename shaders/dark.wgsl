// GML `TopCont` darkness composite (`scrDarkness` + `TopCont/Draw_0:79-83`).
//
// `scrDarkness` builds a white surface (full dark), stamps c_gray discs
// (half dark) and c_black cores (lit) around the player and the scene's
// light sources, then `TopCont` blits it with `bm_subtract`. Subtract white
// is black and subtract black is a no-op, so the same result is one
// MULTIPLY of the inverted surface: black where the mask is black, the
// scene untouched where it is white, and a half dim where it is gray.
//
// The mask is cleared to the base level (black for a normal run, gray for
// the Eyes race) and the discs are stamped in its inverted colours, so the
// composite only has to multiply.

@group(0) @binding(0) var mask: texture_2d<f32>;
@group(0) @binding(1) var nearest_smp: sampler;

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
    let dims = vec2<f32>(textureDimensions(mask, 0));
    let uv = in.pos.xy / dims;
    // The mask is canvas-sized, so the target-pixel position over its own
    // dimensions is the 1:1 texel lookup.
    return vec4<f32>(textureSampleLevel(mask, nearest_smp, uv, 0.0).rgb, 1.0);
}
