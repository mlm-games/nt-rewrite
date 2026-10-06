// Portal vortex (ported from the former bevy prototype's vortex shader,
// itself a direct transcription of `scrDrawSpiral.gml`).
// Rendered as ONE fullscreen quad: each wisp is a transformed sample of the
// real spiral art; growth/alpha/lightning follow the GameMaker laws.
// Bevy-isms removed: no `#import`, no material bind-group macro. Uniforms
// arrive in one globals block; the art textures share one Nearest sampler.
// `view` maps screen uv to world/view coordinates; its center is the
// live GUI view rect `(origin_x + view_w/2, origin_y + 120)`. GML
// `display_set_gui_size(view)` makes GUI px == view px 1:1, so wisps
// drawn at GUI coords land 1:1 on screen, filling the corners like
// `draw_sprite_ext(view_xview + x, ...)` does.

struct VortexGlobals {
    wisps: array<vec4<f32>, 128>,
    debris: array<vec4<f32>, 32>,
    // Per-wisp bolt clock (lanim, langle radians), indexed like wisps.
    // vec4 (not vec2): uniform arrays need a 16-byte stride.
    streams: array<vec4<f32>, 128>,
    // Venuz star motes ([x, y, xscale, frame]; x < -100 parks the slot).
    stars: array<vec4<f32>, 128>,
    vards: array<vec4<f32>, 64>,
    vard_meta: array<vec4<f32>, 64>,
    glob_a: vec4<f32>,
    glob_b: vec4<f32>,
    // Look center + visible extent in wisp coord space.
    view: vec4<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: VortexGlobals;
@group(1) @binding(0) var spiral_tex: texture_2d<f32>;
@group(1) @binding(1) var bolt_tex: texture_2d<f32>;
@group(1) @binding(2) var debris_tex: texture_2d<f32>;
@group(1) @binding(3) var spiral_proto_tex: texture_2d<f32>;
@group(1) @binding(4) var spiral_idpd_tex: texture_2d<f32>;
@group(1) @binding(5) var spiral_idpd2_tex: texture_2d<f32>;
@group(1) @binding(6) var star_tex: texture_2d<f32>;
@group(1) @binding(7) var vard_bandit_tex: texture_2d<f32>;
@group(1) @binding(8) var vard_rat_tex: texture_2d<f32>;
@group(1) @binding(9) var vard_car_tex: texture_2d<f32>;
@group(1) @binding(10) var vard_spider_tex: texture_2d<f32>;
@group(1) @binding(11) var vard_frozen_car_tex: texture_2d<f32>;
@group(1) @binding(12) var vard_freak_tex: texture_2d<f32>;
@group(1) @binding(13) var vard_slice_tex: texture_2d<f32>;
@group(1) @binding(14) var lin_smp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    // Screen uv, y-down (0,0) = top-left, like canvas space.
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let x = f32(i / 2u) * 4.0 - 1.0;
    let y = f32(i % 2u) * 4.0 - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

const N: u32 = 128u;
const BOLT_FRAMES: f32 = 6.0;

fn source_over(dst: vec4<f32>, src_rgb: vec3<f32>, src_a: f32) -> vec4<f32> {
    let a = src_a + dst.a * (1.0 - src_a);
    let rgb = src_rgb * src_a + dst.rgb * (1.0 - src_a);
    return vec4<f32>(rgb, a);
}

fn vard_texture(slot: u32, uv: vec2<f32>) -> vec4<f32> {
    if (slot == 0u) {
        return textureSampleLevel(vard_bandit_tex, lin_smp, uv, 0.0);
    }
    if (slot == 1u) {
        return textureSampleLevel(vard_rat_tex, lin_smp, uv, 0.0);
    }
    if (slot == 2u) {
        return textureSampleLevel(vard_car_tex, lin_smp, uv, 0.0);
    }
    if (slot == 3u) {
        return textureSampleLevel(vard_spider_tex, lin_smp, uv, 0.0);
    }
    if (slot == 4u) {
        return textureSampleLevel(vard_frozen_car_tex, lin_smp, uv, 0.0);
    }
    if (slot == 5u) {
        return textureSampleLevel(vard_freak_tex, lin_smp, uv, 0.0);
    }
    if (slot == 6u) {
        return textureSampleLevel(vard_slice_tex, lin_smp, uv, 0.0);
    }
    return vec4<f32>(0.0);
}

fn vard_dimensions(slot: u32) -> vec2<f32> {
    if (slot == 0u) {
        return vec2<f32>(textureDimensions(vard_bandit_tex));
    }
    if (slot == 1u) {
        return vec2<f32>(textureDimensions(vard_rat_tex));
    }
    if (slot == 2u) {
        return vec2<f32>(textureDimensions(vard_car_tex));
    }
    if (slot == 3u) {
        return vec2<f32>(textureDimensions(vard_spider_tex));
    }
    if (slot == 4u) {
        return vec2<f32>(textureDimensions(vard_frozen_car_tex));
    }
    if (slot == 5u) {
        return vec2<f32>(textureDimensions(vard_freak_tex));
    }
    return vec2<f32>(textureDimensions(vard_slice_tex));
}

// Per-wisp lightning clock comes from the sim (`Spiral` lanim/langle,
// advanced per tick in `SpiralCtl`); the slot matches the wisp slot.

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let gui = vec2<f32>(
        g.view.x + (in.uv.x - 0.5) * g.view.z,
        g.view.y + (in.uv.y - 0.5) * g.view.w,
    );
    let tick_now = g.glob_a.x;
    let bg_alpha = g.glob_b.y;
    let thresh = g.glob_b.z;
    // SpiralKind discriminant + jungle-debris flag (see Rust glob docs).
    let kind = g.glob_b.w - floor(g.glob_b.w / 4.0) * 4.0;
    let debris_big = g.glob_b.w >= 4.0;
    let draw_details = g.flags.y > 0.5;
    var acc = vec4<f32>(
        g.glob_a.z * bg_alpha,
        g.glob_a.w * bg_alpha,
        g.glob_b.x * bg_alpha,
        bg_alpha,
    );

    // Oldest drawn first, so the NEWEST wisp lands on top: GML `with`
    // iterates instances in creation order and script draws ignore depth.
    // Slot mapping matches CPU `(ticks-1) % N`.
    let base = u32(tick_now);
    for (var k: u32 = 0u; k < N; k = k + 1u) {
        let birth = base - (N - 1u - k);
        if (birth < 1u) { continue; }
        let slot = (birth - 1u) % N;
        let d = g.wisps[slot];
        if (d.z < 0.0) { continue; }
        let age = tick_now - d.z;
        if (age < 0.0) { continue; }
        let stream = g.streams[slot];
        let lanim = stream.x;
        let s = stream.z;
        if (s <= 0.0) { continue; }
        if (s > thresh && !(lanim > 0.0 && lanim < 6.0)) { continue; }

        if (g.flags.x > 0.5 && lanim > 0.0 && lanim < 6.0) {
            let frame = clamp(floor(lanim), 0.0, BOLT_FRAMES - 1.0);
            let rot = abs(d.w);
            let bolt_rot = rot - 0.7853982 + stream.y;
            let bc = cos(bolt_rot);
            let bs = sin(bolt_rot);
            var lrel = gui - d.xy;
            lrel = vec2<f32>(bc * lrel.x + bs * lrel.y, -bs * lrel.x + bc * lrel.y);
            let buv = (lrel / s + vec2<f32>(180.0, 88.0)) / vec2<f32>(176.0, 176.0);
            if (all(buv > vec2<f32>(0.0)) & all(buv < vec2<f32>(1.0))) {
                let bolt_uv_x = (frame * 176.0 + 0.5 + buv.x * 175.0) / 1056.0;
                let bolt_uv_y = (0.5 + buv.y * 175.0) / 176.0;
                let btex = textureSampleLevel(bolt_tex, lin_smp, vec2<f32>(bolt_uv_x, bolt_uv_y), 0.0);
                acc = source_over(acc, btex.rgb, btex.a);
                let bolt_black = clamp(0.4 - s * 0.5, 0.0, 1.0);
                if (bolt_black > 0.001) {
                    acc = source_over(acc, vec3<f32>(0.0), btex.a * bolt_black);
                }
            }
        }

        // IDPD2 variant rides in the rot sign (CPU); art is 128px single.
        let idpd2 = d.w < 0.0;
        let rot = abs(d.w);
        let c = cos(rot);
        let sn = sin(rot);
        var rel = gui - d.xy;
        rel = vec2<f32>(c * rel.x + sn * rel.y, -sn * rel.x + c * rel.y);

        if (kind > 1.5 && kind < 2.5) {
            // IDPD: single 128x128 frame, origin centre.
            let half_ext = 64.0 * s * 10.0;
            let suv = rel / half_ext * 0.5 + vec2<f32>(0.5, 0.5);
            if (all(suv > vec2<f32>(0.0)) & all(suv < vec2<f32>(1.0))) {
                let uv = (suv * 127.0 + 0.5) / 128.0;
                var tex = textureSampleLevel(spiral_idpd_tex, lin_smp, uv, 0.0);
                if (idpd2) {
                    tex = textureSampleLevel(spiral_idpd2_tex, lin_smp, uv, 0.0);
                }
                acc = source_over(acc, tex.rgb, tex.a);
                let black_a = clamp(0.8 - s, 0.0, 1.0);
                if (black_a > 0.001) {
                    acc = source_over(acc, vec3<f32>(0.0), tex.a * black_a);
                }
            }
            continue;
        }

        let suv = rel / (32.0 * s * 10.0) * 0.5 + vec2<f32>(0.5, 0.5);
        if (all(suv > vec2<f32>(0.0)) & all(suv < vec2<f32>(1.0))) {
        var sframe = f32(u32(floor(age * (2.0 / 30.0))) % 2u);
            if (stream.w > 0.5) {
                sframe = 1.0;
            }
            // Half-texel inset to avoid atlas bleeding (sprite is 64x64 in 128x64 strip)
            let uv_x = (sframe * 64.0 + 0.5 + suv.x * 63.0) / 128.0;
            let uv_y = (0.5 + suv.y * 63.0) / 64.0;
            var tex = textureSampleLevel(spiral_tex, lin_smp, vec2<f32>(uv_x, uv_y), 0.0);
            if (kind > 0.5 && kind < 1.5) {
                tex = textureSampleLevel(spiral_proto_tex, lin_smp, vec2<f32>(uv_x, uv_y), 0.0);
            }
            // GML two-pass: white (c_white, alpha tex.a) then black 0.8 - s
            acc = source_over(acc, tex.rgb, tex.a);
            let black_a = clamp(0.8 - s, 0.0, 1.0);
            if (black_a > 0.001) {
                acc = source_over(acc, vec3<f32>(0.0), tex.a * black_a);
            }
        }
    }

    if draw_details {
    // SpiralDebris pass - drawn AFTER all wisps (scrDrawSpiral order), on top.
    // CPU supplies [x, y, rot_rad, frame + xscale/32]; x < -100 = empty slot.
    // Jungle debris (sprDebris105) uses 16px frames instead of 8px.
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let d = g.debris[i];
        if (d.x < -100.0) { continue; }
        // Jungle debris uses 16px frames, all other areas 8px.
        var fpx = 8.0;
        if (debris_big) {
            fpx = 16.0;
        }
        let xs = fract(d.w) * 32.0;
        let frame = floor(d.w);
        let half_ext = (fpx * 0.5) * xs;
        var rel = gui - d.xy;
        let c = cos(d.z);
        let sn = sin(d.z);
        rel = vec2<f32>(c * rel.x + sn * rel.y, -sn * rel.x + c * rel.y);
        let duv = rel / half_ext * 0.5 + vec2<f32>(0.5, 0.5);
        if (all(duv > vec2<f32>(0.0)) & all(duv < vec2<f32>(1.0))) {
            // frames in a 4-wide horizontal strip, half-texel inset
            let uv_x = (frame * fpx + 0.5 + duv.x * (fpx - 1.0)) / (4.0 * fpx);
            let uv_y = (0.5 + duv.y * (fpx - 1.0)) / fpx;
            let tex = textureSampleLevel(debris_tex, lin_smp, vec2<f32>(uv_x, uv_y), 0.0);
            // scrDrawSpiral: white 1, then black (1 - xscale)
            acc = source_over(acc, tex.rgb, tex.a);
            let black_a = clamp(1.0 - xs, 0.0, 1.0);
            if (black_a > 0.001) {
                acc = source_over(acc, vec3<f32>(0.0), tex.a * black_a);
            }
        }
    }
    for (var vi: u32 = 0u; vi < 64u; vi = vi + 1u) {
        let vard = g.vards[vi];
        if (vard.x < -100.0) { continue; }
        let vmeta = g.vard_meta[vi];
        let xs = vard.w;
        if (xs <= 0.001 || vmeta.y <= 0.0 || vmeta.z <= 0.0) { continue; }
        let vc = cos(vard.z);
        let vs = sin(vard.z);
        var vrel = gui - vard.xy;
        vrel = vec2<f32>(vc * vrel.x + vs * vrel.y, -vs * vrel.x + vc * vrel.y);
        let half_ext = vec2<f32>(vmeta.y, vmeta.z) * (0.5 * xs);
        let duv = vrel / half_ext * 0.5 + vec2<f32>(0.5, 0.5);
        if (all(duv > vec2<f32>(0.0)) & all(duv < vec2<f32>(1.0))) {
            let dims = vard_dimensions(u32(vmeta.w));
            let uv = vec2<f32>(
                (vmeta.x * vmeta.y + 0.5 + duv.x * (vmeta.y - 1.0)) / dims.x,
                (0.5 + duv.y * (vmeta.z - 1.0)) / dims.y,
            );
            let tex = vard_texture(u32(vmeta.w), uv);
            acc = source_over(acc, tex.rgb, tex.a);
            let black_a = clamp(1.0 - xs, 0.0, 1.0);
            if (black_a > 0.001) {
                acc = source_over(acc, vec3<f32>(0.0), tex.a * black_a);
            }
        }
    }
    // SpiralStar pass - drawn AFTER debris (GML `with` order), on top.
    // CPU supplies [x, y, xscale, frame]; x < -100 = empty slot.
    // sprSpiralStar is a 2-frame 3x3 strip (frame 0/1, origin centre).
    for (var si: u32 = 0u; si < 128u; si = si + 1u) {
        let st = g.stars[si];
        if (st.x < -100.0) { continue; }
        let xs = st.z;
        if (xs <= 0.001) { continue; }
        var srel = gui - st.xy;
        // GML draws stars unrotated (image_angle stays 0: the Step
        // never turns it), scaled by xscale on both axes.
        let sduv = srel / (1.5 * xs) * 0.5 + vec2<f32>(0.5, 0.5);
        if (all(sduv > vec2<f32>(0.0)) & all(sduv < vec2<f32>(1.0))) {
            // two 3px frames side by side in the 6x3 strip
            let suv_x = (st.w * 3.0 + 0.5 + sduv.x * 2.0) / 6.0;
            let suv_y = (0.5 + sduv.y * 2.0) / 3.0;
            let stex = textureSampleLevel(star_tex, lin_smp, vec2<f32>(suv_x, suv_y), 0.0);
            // scrDrawSpiral: white 1, then black (1 - xscale)
            acc = source_over(acc, stex.rgb, stex.a);
            let sblack = clamp(1.0 - xs, 0.0, 1.0);
            if (sblack > 0.001) {
                acc = source_over(acc, vec3<f32>(0.0), stex.a * sblack);
            }
        }
    }
    }
    if (acc.a > 0.0001) {
        return vec4<f32>(acc.rgb / acc.a, acc.a);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
