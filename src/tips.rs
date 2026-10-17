//! Loading-screen tips, GML `scrTips` (`scripts/scrTips/scrTips.gml`) with
//! the `scr_skill_get_tips` / `scr_crown_get_tips` lookups the Weapons,
//! Mutations and Crowns pools resolve through. Pool weights, gates and the
//! empty-tip reroll are verbatim; the `UberCont.april_fools` alt line uses
//! its normal wording (the port carries no date-event flags).

use bevy_ecs::prelude::*;
use rand::RngExt;

use crate::comps_a::{CrownState, Health, Inventory, Player, RaceState, Run};
use crate::data::MutationId;

/// Everything `scrTips` reads off the live player + run.
pub struct TipCtx {
    /// GML area id (`GameCont.area`).
    pub gml_area: u8,
    /// GML race id (`_player.race`).
    pub race: Option<u8>,
    pub level: u32,
    pub hp: i32,
    pub loops: u32,
    /// GML crown id (`GameCont.crown`).
    pub gml_crown: u8,
    pub cursed: bool,
    /// Held weapon's `wep_text` (`scr_weapon_get_text`).
    pub weapon_text: Option<&'static str>,
    pub skills: Vec<MutationId>,
    pub has_player: bool,
    /// GML `UberCont.april_fools` swaps the General:9 line for its Alt.
    pub april_fools: bool,
}

fn pick(arr: &[&'static str], rng: &mut impl RngExt) -> &'static str {
    if arr.is_empty() {
        ""
    } else {
        arr[rng.random_range(0..arr.len())]
    }
}

/// Build the ctx off the sim world, then roll. Called from the loading
/// screen the first time it needs a tip; the pick is cached on
/// `LoadingState` / `FloorTransition` afterwards.
pub fn pick_loading_tip_world(world: &mut World) -> String {
    let (gml_area, loops) = world
        .get_resource::<Run>()
        .map(|r| (crate::vortex::gml_area_for_area(r.area), r.loop_count))
        .unwrap_or((0, 0));
    let mut ctx = TipCtx {
        gml_area,
        race: None,
        level: 0,
        hp: 99,
        loops,
        gml_crown: 1,
        cursed: false,
        weapon_text: None,
        skills: Vec::new(),
        has_player: false,
        april_fools: world
            .get_resource::<crate::date_part::DateRuntime>()
            .is_some_and(|d| d.events.april_fools),
    };
    let mut q = world.query::<(&Player, &Health, &Inventory, &RaceState, &CrownState)>();
    if let Some((player, health, inv, race, crown)) = q.iter(world).next() {
        ctx.has_player = true;
        ctx.race = Some(race.race as u8);
        ctx.level = player.level;
        ctx.hp = health.hp;
        ctx.gml_crown = crate::crown::crown_port_to_gml(crown.crown as u8);
        ctx.cursed = inv.cursed.iter().any(|c| *c);
        let slot = inv.current.min(inv.weapon_slots.saturating_sub(1));
        let wid = inv.weapons[slot];
        if wid != crate::data::WeaponId::NONE
            && (wid.0 as usize) < crate::weapons_data::WEAPONS.len()
        {
            ctx.weapon_text = Some(crate::weapons_data::WEAPONS[wid.0 as usize].wep_text);
        }
        ctx.skills = player.mutations.clone();
    }
    let mut rng = rand::rng();
    pick_tip(&ctx, &mut rng).to_string()
}

/// GML `scrTips:9` `choose(0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 4, 5,
/// 6, 7)`: 18-way roll, general double-weighted twice over.
const WEIGHTS: [u8; 18] = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7];

pub fn pick_tip(ctx: &TipCtx, rng: &mut impl RngExt) -> &'static str {
    let mut tip_type = WEIGHTS[rng.random_range(0..WEIGHTS.len())];
    if !ctx.has_player {
        tip_type = 0;
    }
    loop {
        let tip = match tip_type {
            0 => {
                // GML `scrTips:32-34`: the april-fools alt line.
                let t = pick(GENERAL, rng);
                if ctx.april_fools && t == "fish can roll" {
                    "fish can't roll"
                } else {
                    t
                }
            }
            1 => pick(area_tips(ctx.gml_area), rng),
            2 => match ctx.race {
                // GML `:301-303`: the ultra sub-pool fires at level 10+
                // on a 1-in-5 roll.
                Some(race) if ctx.level >= 10 && rng.random_range(0..5) == 0 => {
                    pick(ultra_tips(race), rng)
                }
                Some(race) => pick(race_tips(race), rng),
                None => "",
            },
            // GML `:629-642`: cursed 1-in-10 first, else the held weapon.
            3 => {
                if ctx.cursed && rng.random_range(0..10) == 0 {
                    pick(CURSE, rng)
                } else {
                    ctx.weapon_text.unwrap_or("")
                }
            }
            // GML `:644-652`: a random owned skill's tip at level 2+.
            4 => {
                if ctx.level > 1 && !ctx.skills.is_empty() {
                    let skill = ctx.skills[rng.random_range(0..ctx.skills.len())];
                    pick(skill_tip(crate::hud::mutation_skill_index(skill)), rng)
                } else {
                    ""
                }
            }
            // GML `:655-661`: any crown above none.
            5 => {
                if ctx.gml_crown > 1 {
                    pick(crown_tip(ctx.gml_crown), rng)
                } else {
                    ""
                }
            }
            6 => {
                if ctx.hp < 4 {
                    pick(LOW, rng)
                } else {
                    ""
                }
            }
            _ => {
                if ctx.loops > 0 {
                    pick(LOOPS, rng)
                } else {
                    ""
                }
            }
        };
        if !tip.is_empty() {
            return tip;
        }
        // GML `:692-693`: an empty tip rerolls the whole draw.
        tip_type = WEIGHTS[rng.random_range(0..WEIGHTS.len())];
        if !ctx.has_player {
            tip_type = 0;
        }
    }
}

const GENERAL: &[&str] = &[
    "@rHP@s will only drop when damaged",
    "@yexplosives@s can destroy walls",
    "@wmelee weapons@s can deflect @yprojectiles@s",
    "@rHP@s will only drop when damaged",
    "@yexplosives@s can destroy walls",
    "@wmelee weapons@s can deflect @yprojectiles@s",
    "@yshells@s deal more damage from up close",
    "assassins can pretend they're dead",
    "fish can roll",
    "always keep one eye on your @yammo@s",
    "robots can digest anything",
    "bandits like camping near barrels",
    "not today",
    "rosebud",
    "it's not fair, it's not right",
    "watch out for dehydration",
    "crystal can shield",
    "fear is the mindkiller",
    "going in big",
    "it's happening",
    "no",
    "bob and weave",
    "airhorn.wav",
    "danger zone",
    "the @wthrone@s is getting closer",
    "what a life",
    "remember to take a 15 minute break#for every hour you play!",
    "so sweaty",
    "just another @pportal@s",
    "doing work",
    "hold the phone",
    "no shame",
    "follow your heart",
    "practice makes perfect",
    "don't give up",
    "the @bI.D.P.D.@s is looking for you",
    "try hitting @ygrenades@s for extra range",
    "pick your @gmutations@s wisely",
    "if only you could talk to the monsters",
    "@wsteroids@s can dual wield",
    "always wear dry socks",
    "@wcrowns@s are loyal",
    "@yenergy weapons@s use lots of @yammo@y",
    "the throne definitely exists",
    "try not opening @wweapon chests@s",
    "fire at things until they die",
    "@yammo@s drops depend on your @wweapon types@s",
    "don't get hit",
    "you get fewer drops when high on @yammo@s",
    "doing alright",
    "there are other worlds out there",
];

const CURSE: &[&str] = &[
    "@qw @qh @qy",
    "@qh @qe @ql @ql @qo",
    "@qi @qs @qe @qe @qy @qo @qu",
    "@ql @qe @qt @qm @qe @qo @qu @qt",
    "@qp @ql @qe @qa @qs @qe @qs @qt @qo @qp",
];

const LOW: &[&str] = &[
    "oh dear",
    "this isn't going to end well",
    "good luck",
    "no no no",
    "help",
];

const LOOPS: &[&str] = &[
    "this will never end",
    "it's a whole new world",
    "monsters everywhere",
    "what's happening",
    "there's no limit",
    "we'll reach for the sky",
    "no valley too deep",
    "no mountain too high",
];

fn area_tips(k: u8) -> &'static [&'static str] {
    match k {
        0 => &[
            "it's so dark",
            "your friends were here",
            "this can't be true",
        ],
        1 => &[
            "welcome to the future",
            "the wasteland calls you",
            "scorching sun",
            "watch out for maggots",
            "let's do this",
            "dust surrounds you",
            "the wind hurts",
        ],
        2 => &[
            "don't drink the water",
            "the sewers stink",
            "don't touch the frogs",
            "don't eat the rat meat",
            "danger",
            "water dripping",
            "sludge everywhere",
            "so many rats",
        ],
        3 => &[
            "shoot robots on sight",
            "there used to be trees here",
            "look up",
            "rust everywhere",
            "the sound of birds",
            "sludge pools",
            "climb over cars",
            "portals can blow up cars",
        ],
        4 => &[
            "oh no",
            "spiderwebs everywhere",
            "don't lose your heart",
            "reflections on the walls",
            "skin is crawling",
            "almost halfway there",
        ],
        5 => &[
            "wear a scarf",
            "they used to have electricity",
            "miss the sun",
            "civilization",
            "walk softly",
            "there is no yeti",
        ],
        6 => &[
            "beep boop",
            "nerds",
            "don't push any buttons",
            "this place is old",
            "the palace",
        ],
        100 => &[
            "awww yes",
            "fish",
            "hold your breath",
            "it's beautiful down here",
            "don't move",
        ],
        102 => &["hunger...", "it smells nice here"],
        103 => &[
            "space...",
            "always wanted to go here",
            "so much money",
            "4 years later...",
        ],
        104 => &[
            "no oh",
            "everywhere spiderwebs",
            "crawling is skin",
            "there halfway almost",
        ],
        105 => &[
            "there's something in the trees",
            "bugs everywhere",
            "welcome to the jungle",
            "heart of darkness",
        ],
        107 => &[
            "this is where the magic happens",
            "lets take a look in the fridge",
            "get the hell out of here",
            "now this is real special",
            "wakkala wayo",
        ],
        _ => &[],
    }
}

fn race_tips(k: u8) -> &'static [&'static str] {
    match k {
        0 => &[
            "random shifts shapes",
            "clearly the most powerful",
            "never the same",
            "shapeless",
            "passive: anything",
            "active: anything",
            "noone compares",
        ],
        1 => &[
            "last day before retirement",
            "duty calls",
            "it's ok to eat",
            "gills on your neck",
            "like kevin costner",
            "the taste of mud",
        ],
        2 => &["crystal can handle this", "family"],
        3 => &[
            "eyes sees everything",
            "don't blink",
            "all these thoughts",
            "eyes can't speak",
            "telekinesis pushes projectiles away",
        ],
        4 => &[
            "it's so cold out here",
            "melting is tired",
            "everything hurts",
            "cough",
            "brr...",
            "it's ok to be scared",
        ],
        5 => &[
            "kill kill kill",
            "death death death",
            "blood blood blood",
            "no mercy",
            "photosynthesizing...",
            "snare is a source of light",
            "gotta go fast",
        ],
        6 => &[
            "yung venuz is so cool",
            "mony",
            "guns that straight festive",
            "guns with 6 senses",
            "guns that hate texas",
            "guns that wear vests it",
            "no @bpopo@s",
            "pop pop",
            "shoutout 2 cuz",
            "yung venuz is the best",
            "so cool",
            "thanks gun god",
            "\\\\#verifyvenuz",
            "2 yung 2 die",
            "guns for fake necklace",
            "guns that send textses",
            "guns that make breakfast",
        ],
        7 => &[
            "time to flex",
            "study hard",
            "steroids could do pushups forever",
            "steroids used to be a scientist",
            "appreciate @wrevolvers@s",
            "get shots",
            "get strong",
            "read a book",
        ],
        8 => &[
            "6b 69 6c 6c#61 6c 6c#68 75 6d 61 6e 73",
            "72 6f 62 6f 74",
            "3c 33",
            "74 61 73 74 79",
            "@wguns@s for breakfast",
            "I'm afraid I can't let you do that",
            "don't forget to eat @wweapons@s",
        ],
        9 => &[
            "remember the training",
            "focus",
            "go",
            "never surrender",
            "amateur hour is over",
            "again",
            "getting decapitated reduces max HP",
            "throw damage scales with your level",
        ],
        10 => &[
            "forget the old days",
            "change is coming",
            "allies are a source of light",
            "a new generation",
            "it will get better",
            "spawning new allies @rheals@s old ones",
            "your first ally costs less @rHP@s",
            "allies take damage over time",
            "the scarf is nice",
        ],
        11 => &[
            "@gradiation@s is everywhere",
            "in the zone",
            "the horror",
            "power",
            "horror's beam powers up over time",
            "horror's beam destroys projectiles",
            "enemies absorb the beam's @grads@s",
            "firing the beam pauses @grad@s attraction",
        ],
        12 => &[
            "another world lost",
            "never stop",
            "they're getting closer",
            "never slow down",
            "never look back",
            "keep moving",
        ],
        13 => &["big", "dog"],
        14 => &["nothing...", "dry", "dusty", "hard"],
        15 => &[
            "can't wait",
            "let's go",
            "restless",
            "bloated",
            "gas is good",
            "let it all out",
            "keep going",
            "don't hold it in",
            "wait for me",
        ],
        16 => &[
            "cool",
            "need a snack",
            "gunz god",
            "cuz can carry multiple weapons",
            "cuz can cry",
            "take damage to gain tears",
            "born 2 gun",
            "nice",
            "shoutout 2 veenie",
            "yung cuz!!!",
            "every gun is my favorite gun",
            "it’s ok to cry",
            "crying is cool",
            "strapped",
            "don't upset me",
        ],
        _ => &[],
    }
}

fn ultra_tips(k: u8) -> &'static [&'static str] {
    match k {
        0 => &["random @gultra@s, unstoppable"],
        1 => &["getting used to this", "just one more day"],
        2 => &["just a scratch", "stay strong"],
        3 => &["know everything", "show nothing", "keep it inside"],
        4 => &[
            "the pain...",
            "please stop...",
            "life is incredibly beautiful",
        ],
        5 => &["end end end"],
        6 => &[
            "\\\\#verifycuz",
            "airsiren.wav",
            "\\\\#blessed",
            "one of these days...",
            "real thugs hustle",
            "YV fact: YV IS THE BEST",
            "4EVER",
            "GO HARD",
            "VOTE 2 B COOL",
        ],
        7 => &["let's hope this is correct", "so strong"],
        8 => &[
            "6d 61 63 68 69 6e 65 73#77 69 6c 6c#6e 65 76 65 72#65 6e 64",
            "66 6c 65 73 68#69 73#77 65 61 6b",
            "73 69 6e 67 75 6c 61 72 69 74 79",
            "don't panic",
        ],
        9 => &[
            "this is destiny",
            "just like in the movies",
            "again we are defeated",
        ],
        10 => &["all together now", "no stopping now"],
        11 => &[
            "the air is changing",
            "the light moves",
            "things are different",
        ],
        12 => &[
            "use this",
            "you deserve this",
            "they can't chase you forever",
        ],
        13 => &["guard this land", "none will pass", "fire at will"],
        14 => &["this is better", "avoid the living", "no need for peace"],
        15 => &[
            "bwahahaha",
            "singing",
            "sweet sounds",
            "go forever",
            "smell great",
        ],
        16 => &[
            "good job cuz!!!",
            "round and handsome",
            "spherical miracle",
            "gaming makes you smart",
        ],
        _ => &[],
    }
}

fn skill_tip(k: u8) -> &'static [&'static str] {
    match k {
        0 => &[""],
        1 => &["thick skin"],
        2 => &["run forever"],
        3 => &["need those rads"],
        4 => &["feeling lucky"],
        5 => &["sit on the throne"],
        6 => &["AMMO everywhere"],
        7 => &["drink blood"],
        8 => &["skin glows"],
        9 => &["stomach rumbles"],
        10 => &["great strength"],
        11 => &["mirrors will break"],
        12 => &["time passes slowly"],
        13 => &["more reach"],
        14 => &["temperature is rising"],
        15 => &["shells are friends"],
        16 => &["no need to aim"],
        17 => &["neurons everywhere"],
        18 => &["listen"],
        19 => &["every shot connects"],
        20 => &["see them fly"],
        21 => &["bolts everywhere"],
        22 => &["shaking"],
        23 => &[""],
        24 => &[""],
        25 => &[""],
        26 => &[""],
        27 => &["NOT TODAY"],
        28 => &["DID YOU WAIT FOR IT?"],
        29 => &[""],
        _ => &[],
    }
}

fn crown_tip(k: u8) -> &'static [&'static str] {
    match k {
        0 => &[""],
        1 => &[""],
        2 => &["boom", "health is important"],
        3 => &["heart matters"],
        4 => &["no time for jokes"],
        5 => &["guns are your friend"],
        6 => &["something is wrong"],
        7 => &["get ready", "bring it"],
        8 => &["no such thing as free will"],
        9 => &["you really like these @wweapons@s"],
        10 => &["the future brings death", "it's all the same"],
        11 => &["why"],
        12 => &["good"],
        13 => &["safety first"],
        _ => &[],
    }
}
