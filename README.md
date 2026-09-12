# nt-rewrite

Nuclear Throne recreation running on repame via repose.

Migration target for `../nt-recreated-bevy` (bevy 0.20-dev): sim systems
port module-by-module into `repame-sim` schedules at the same 30 Hz fixed
 step; world rendering goes through `repame-sprite` (atlas + GPU batch)
 and the in-crate `vortex_pass` portal background; UI stays repose views.

Layout: `src/` grows `sim/` (components, resources, systems),
`render/` (snapshot producers), `ui/`, `audio.rs`, `save.rs`, `input.rs`
as strangler slices land. Nothing here depends on bevy, ever.
