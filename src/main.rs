#![feature(generic_const_exprs)]
use std::collections::BinaryHeap;
use std::io::stdin;
use std::sync::Mutex;
use std::time::SystemTime;
use std::{path::Path, time::Duration};

#[cfg(not(feature = "train"))]
use bevy::asset::AssetServer;
#[cfg(not(feature = "train"))]
use bevy::core_pipeline::core_2d::Camera2dBundle;
use bevy::ecs::query::Without;
use bevy::ecs::schedule::IntoSystemConfigs;
#[cfg(not(feature = "train"))]
use bevy::ecs::system::Res;
use bevy::input::keyboard::KeyCode;
use bevy::input::ButtonInput;
use bevy::reflect::GetField;
#[cfg(not(feature = "train"))]
use bevy::sprite::{Sprite, SpriteBundle};
#[cfg(not(feature = "train"))]
use bevy::DefaultPlugins;
#[cfg(feature = "train")]
use bevy::MinimalPlugins;
use bevy::{
    app::{App, MainScheduleOrder, PreUpdate, Startup, Update},
    ecs::{
        component::Component,
        entity::Entity,
        event::{Event, EventReader, EventWriter},
        query::With,
        schedule::ScheduleLabel,
        system::{Commands, NonSend, NonSendMut, Query, ResMut, Resource},
    },
    math::{NormedVectorSpace, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles},
    time::Timer,
    transform::components::Transform,
    utils::HashMap,
    window::{Window, WindowResolution},
};
#[allow(unused_imports)]
use dfdx::nn::PReLU1DConfig;
use dfdx::nn::{
    BuildModuleExt, BuildOnDevice, LinearConfig, LoadSafeTensors, Module, SaveSafeTensors,
    ZeroGrads,
};
// #[allow(unused_imports)]
use dfdx::nn::optim::Optimizer;
#[allow(unused_imports)]
use dfdx::nn::{Adam, AdamConfig, RMSprop, RMSpropConfig, Sgd, SgdConfig};
use dfdx::shapes::{Const, Rank1, Rank2};
#[allow(unused_imports)]
use dfdx::tensor::Cpu;
use dfdx::tensor::{AsArray, Cuda, Tape, Tensor, TensorFrom, TensorFromVec, Trace};
use dfdx::tensor_ops::{Backward, BroadcastTo, MaxTo, MeanTo, SumTo};
use once_cell::sync::Lazy;
use rand::seq::IteratorRandom;
#[allow(unused_imports)]
#[cfg(feature = "train")]
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Component, Debug)]
struct Player {
    ty: PlayerType,
    alive: f32,
}

#[allow(dead_code)]
#[derive(Debug)]
enum PlayerType {
    Keyboard,
    AI,
}

#[derive(Resource)]
struct ShootTimer(Timer);

#[derive(Component)]
struct PlayerBullet;

const HLS: usize = 200;
// type NonLin<const S: usize> = PReLU1DConfig<Const<S>>;
// type NonLin<const S: usize> = dfdx::nn::LeakyReLU;
type NonLin<const S: usize> = dfdx::nn::Tanh;
type ThingNet<const INPUTS: usize, const HLS: usize, const MOVES: usize> = (
    (
        LinearConfig<Const<INPUTS>, Const<HLS>>,
        NonLin<HLS>,
        // LinearConfig<Const<HLS>, Const<HLS>>,
        // NonLin<HLS>,
    ),
    (
        LinearConfig<Const<HLS>, Const<HLS>>,
        NonLin<HLS>,
        // LinearConfig<Const<HLS>, Const<HLS>>,
        // NonLin<HLS>,
    ),
    LinearConfig<Const<HLS>, Const<MOVES>>,
);
// type NetDevice = Cpu;
type NetDevice = Cuda;

// consts
// No, too unstable
// const LR: f64 = 0.01;
// const LR: f64 = 0.001;
// const LR: f64 = 0.0002;
// until about 0.01
// nvm lower rates dont seem to work
// nvm I think that was because I didnt tell it dying was bad
// This is good for initial training
// const LR: f64 = 0.00008;
// Try this for the last bit
static LR: Lazy<Mutex<f64>> = Lazy::new(|| {
    Mutex::new(
        std::fs::read_to_string("lr")
            .unwrap()
            .trim()
            .parse()
            .unwrap(),
    )
});
// const LR: f64 = 0.00001;
// const LR: f64 = 0.000001;
// const LR: f64 = 0.0000001;
// const LR: f64 = 0.00000001;
// const LR_DECAY: f64 = 0.995;
static LR_DECAY: Lazy<Mutex<f64>> = Lazy::new(|| {
    Mutex::new(
        std::fs::read_to_string("lr_decay")
            .unwrap()
            .trim()
            .parse()
            .unwrap(),
    )
});

// const GAMES_PER_CYCLE: usize = 50;
const FRAMES_PER_CYCLE: usize = 500;
const REPLAY_MEM: usize = 48_000;
const REPLAY_MEM_OVERFLOW: usize = 60_000;
const REPLAY_MEM_BEST: usize = 15_000;
const REPLAY_MEM_BEST_OVERFLOW: usize = 18_000;
const REPLAY_MIN: usize = REPLAY_MEM;
const REPLAY_INTERVAL: usize = 16;
const REPLAY_FRAME_RAND: usize = 6 * REPLAY_INTERVAL;
const REPLAY_FRAME_LAST: usize = 4 * REPLAY_INTERVAL;
const REPLAY_FRAME_BEST: usize = 3 * REPLAY_INTERVAL;
const TIMEDELTA: f32 = 0.02;
const MOVES_D: usize = 18;
// const MOVES: usize = 4;
const AWARES: usize = 5;
const INPUTS_D: usize = 3 * 2 * AWARES;
#[cfg(feature = "train")]
static EXPLORE: Lazy<Mutex<f32>> = Lazy::new(|| {
    Mutex::new(
        std::fs::read_to_string("explore")
            .unwrap()
            .trim()
            .parse()
            .unwrap(),
    )
});
#[cfg(not(feature = "train"))]
static EXPLORE: Lazy<Mutex<f32>> = Lazy::new(|| Mutex::new(0.0));
static EXPLORE_DECAY: Lazy<Mutex<f32>> = Lazy::new(|| {
    Mutex::new(
        std::fs::read_to_string("explore_decay")
            .unwrap()
            .trim()
            .parse()
            .unwrap(),
    )
});
const EXPLORE_LIMIT: f32 = 0.1;
const GAMMA: f32 = 0.98;
#[allow(dead_code)]
const RESET_CHANCE: f64 = 0.1;
#[allow(dead_code)]
const IGNORE_BAD_CHANCE: f64 = 0.1;
static MAX_AVG_LOSS: Lazy<Mutex<f32>> = Lazy::new(|| {
    Mutex::new(
        std::fs::read_to_string("max_avg_loss")
            .unwrap()
            .trim()
            .parse()
            .unwrap(),
    )
});

static UPDATE_CONSTS: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));
// const MAX_AVG_LOSS: f32 = 0.4;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
enum ConstName {
    LR,
    LR_DECAY,
    EXPLORE,
    EXPLORE_DECAY,
    MAX_AVG_LOSS,
}
impl TryFrom<&str> for ConstName {
    type Error = &'static str;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match &value.trim().to_uppercase() as &str {
            "LR" => Ok(Self::LR),
            "LR_DECAY" => Ok(Self::LR_DECAY),
            "EXPLORE" => Ok(Self::EXPLORE),
            "EXPLORE_DECAY" => Ok(Self::EXPLORE_DECAY),
            "MAX_AVG_LOSS" => Ok(Self::MAX_AVG_LOSS),
            _ => Err("Not valid"),
        }
    }
}
impl ConstName {
    fn modify(self, val: f64) {
        match self {
            ConstName::LR => {
                *LR.lock().unwrap() = val;
            }
            ConstName::LR_DECAY => {
                *LR_DECAY.lock().unwrap() = val;
            }
            ConstName::EXPLORE => {
                *EXPLORE.lock().unwrap() = val as f32;
            }
            ConstName::EXPLORE_DECAY => {
                *EXPLORE_DECAY.lock().unwrap() = val as f32;
            }
            ConstName::MAX_AVG_LOSS => {
                *MAX_AVG_LOSS.lock().unwrap() = val as f32;
            }
        }
        std::fs::write(
            match self {
                ConstName::LR => "lr",
                ConstName::LR_DECAY => "lr_decay",
                ConstName::EXPLORE => "explore",
                ConstName::EXPLORE_DECAY => "explore_decay",
                ConstName::MAX_AVG_LOSS => "max_avg_loss",
            },
            val.to_string(),
        )
        .unwrap();
        *UPDATE_CONSTS.lock().unwrap() = true;
    }
}

fn manage_consts() -> ! {
    let mut buf = String::new();
    println!("consts runing");
    // let mut stdin = stdin();
    loop {
        // println!("in loop");
        let stdin = stdin();
        buf.clear();
        // stdin.read_line()
        let Ok(_) = stdin.read_line(&mut buf) else {
            println!("Couldnt read");
            continue;
        };
        // println!("Got input");
        drop(stdin);
        // buf = buf.trim();
        let mut args = buf.trim().split(' ');
        let Some(constname) = args.next() else {
            println!("needs 2 arguments, 0 given");
            continue;
        };
        let Some(val) = args.next() else {
            println!("needs 2 arguments, 1 given");
            continue;
        };
        let constname: ConstName = match constname.try_into() {
            Ok(n) => n,
            Err(e) => {
                println!("{constname} {e:?}");
                continue;
            }
        };
        let val: f64 = match val.parse() {
            Ok(n) => n,
            Err(e) => {
                println!("{val} {e:?}");
                continue;
            }
        };
        constname.modify(val);
        println!("{constname:?} set to {val}");
    }
}

fn update_consts(bled: Option<NonSendMut<BleedingD>>, bled2: Option<NonSendMut<BleedingC>>) {
    if *UPDATE_CONSTS.lock().unwrap() {
        if let Some(mut bled) = bled {
            bled.net.optim.cfg.lr = *LR.lock().unwrap();
            *UPDATE_CONSTS.lock().unwrap() = false;
        }
        if let Some(mut bled) = bled2 {
            bled.net.optim.cfg.lr = *LR.lock().unwrap();
            *UPDATE_CONSTS.lock().unwrap() = false;
        }
    }
}

impl MoveOutcome {
    fn reward(&self) -> f32 {
        match self {
            MoveOutcome::EnemyHit => 0.1,
            MoveOutcome::EnemyDie => 1.0,
            MoveOutcome::Die => -1.0,
        }
    }
}
impl From<MoveOutcome> for f32 {
    fn from(value: MoveOutcome) -> Self {
        value.reward()
    }
}

type NetOptim<const INPUTS: usize, const HLS: usize, const MOVES: usize> =
    RMSprop<<ThingNet<INPUTS, HLS, MOVES> as BuildOnDevice<f32, NetDevice>>::Built, f32, NetDevice>;
// type NetOptim = Sgd<<ThingNet as BuildOnDevice<f32, NetDevice>>::Built, f32, NetDevice>;
// type NetOptim = Adam<<ThingNet as BuildOnDevice<f32, NetDevice>>::Built, f32, NetDevice>;

#[allow(non_upper_case_globals)]
const NetOptimConfig: Lazy<RMSpropConfig> = Lazy::new(|| RMSpropConfig {
    lr: *LR.lock().unwrap(),
    momentum: None,
    // momentum: Some(EXPLORE),
    alpha: 0.9,
    eps: 1e-8,
    centered: false,
    weight_decay: None,
});
// #[allow(non_upper_case_globals)]
// const NetOptimConfig: Lazy<SgdConfig> = Lazy::new(|| SgdConfig {
//     lr: *LR.lock().unwrap(),
//     momentum: None,
//     weight_decay: None,
// });
// #[allow(non_upper_case_globals)]
// const NetOptimConfig: Lazy<AdamConfig> = Lazy::new(|| AdamConfig {
//     lr: *LR.lock().unwrap(),
//     weight_decay: None,
//     betas: [0.9, 0.999],
//     eps: 1e-8,
// });

#[derive(Clone)]
struct AIControl<const INPUTS: usize, const HLS: usize, const MOVES: usize> {
    device: NetDevice,
    net: <ThingNet<INPUTS, HLS, MOVES> as BuildOnDevice<f32, NetDevice>>::Built,
    optim: NetOptim<INPUTS, HLS, MOVES>,
}

struct BleedingD {
    net: AIControl<INPUTS_D, HLS, MOVES_D>,
    last_loss: f32,
    last_loss_b: f32,
}
struct SageD {
    net: AIControl<INPUTS_D, HLS, MOVES_D>,
    frames_left: usize,
    alive_time: f32,
    games: f32,
    // last_alive_rat: f32,
    acc_loss: f32,
    acc_loss_b: f32,
    runtime: SystemTime,
    first_run: bool,
}
const INPUTS_C: usize = 4;
const MOVES_C: usize = 2;
struct BleedingC {
    net: AIControl<INPUTS_C, HLS, MOVES_C>,
    last_loss: f32,
    last_loss_b: f32,
}
struct SageC {
    net: AIControl<INPUTS_C, HLS, MOVES_C>,
    frames_left: usize,
    alive_time: f32,
    games: f32,
    // last_alive_rat: f32,
    acc_loss: f32,
    acc_loss_b: f32,
    runtime: SystemTime,
    first_run: bool,
}
impl SageD {
    fn get_net(&self) -> Option<&AIControl<INPUTS_D, HLS, MOVES_D>> {
        if self.first_run {
            None
        } else {
            Some(&self.net)
        }
    }
}
// Prioritized experience replay
#[derive(PartialEq, Clone, Copy, Debug)]
enum PER {
    Never,
    Last(f32),
}
impl PartialOrd for PER {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (PER::Never, PER::Never) => Some(std::cmp::Ordering::Equal),
            (PER::Never, PER::Last(_)) => Some(std::cmp::Ordering::Greater),
            (PER::Last(_), PER::Never) => Some(std::cmp::Ordering::Less),
            (PER::Last(s), PER::Last(o)) => s.partial_cmp(o),
        }
    }
}
impl Eq for PER {}
impl Ord for PER {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
#[derive(PartialEq, Debug, Clone)]
struct ReplayRecord<const INPUTS: usize, const MOVES: usize, M: Vectorable<MOVES> + PartialEq> {
    data: ([f32; INPUTS], M, f32, Option<[f32; INPUTS]>),
    key: PER,
}
impl<const INPUTS: usize, const MOVES: usize, M: Vectorable<MOVES> + PartialEq> Eq
    for ReplayRecord<INPUTS, MOVES, M>
{
}
impl<const INPUTS: usize, const MOVES: usize, M: Vectorable<MOVES> + PartialEq> Ord
    for ReplayRecord<INPUTS, MOVES, M>
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}
impl<const INPUTS: usize, const MOVES: usize, M: Vectorable<MOVES> + PartialEq> PartialOrd
    for ReplayRecord<INPUTS, MOVES, M>
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(&other))
    }
}
#[derive(Default, Resource)]
struct Replay<const INPUTS: usize, const MOVES: usize, M: Vectorable<MOVES> + PartialEq> {
    bests: BinaryHeap<ReplayRecord<INPUTS, MOVES, M>>,
    all: Vec<ReplayRecord<INPUTS, MOVES, M>>,
}
use std::fmt::Debug;
impl<const INPUTS: usize, const MOVES: usize, M: Vectorable<MOVES> + PartialEq>
    Replay<INPUTS, MOVES, M>
{
    fn get_rand<R: Rng, const N: usize>(
        &mut self,
        r: &mut R,
    ) -> Option<[([f32; INPUTS], M, f32, Option<[f32; INPUTS]>); N]>
    where
        M: Debug,
    {
        if self.all.len() < REPLAY_MIN {
            return None;
        }
        // let run = r.gen_range(0..self.0.len() - 1);
        let rem = self.all.iter().choose_multiple(r, N);
        // self.0.retain(|x| !rem.contains(&x));
        // let (sensor, movee, score, next_sensor) = rem.data.clone();
        // self.0;
        // Some((sensor, movee, score, next_sensor))
        let rem: [_; N] = rem.try_into().unwrap();
        Some(rem.map(|x| x.data))
    }
    fn get_last<const N: usize>(
        &self,
    ) -> Option<[([f32; INPUTS], M, f32, Option<[f32; INPUTS]>); N]>
    where
        M: Default,
    {
        if self.all.len() < REPLAY_MIN {
            return None;
        }
        if self.all.len() <= N {
            return None;
        }
        let mut ret: [([f32; INPUTS], M, f32, Option<[f32; INPUTS]>); N] = [0; N].map(|_| {
            (
                [0.; INPUTS],
                Default::default(),
                Default::default(),
                Default::default(),
            )
        });
        let mut c = 0;
        for i in self.all.len() - N - 1..self.all.len() - 1 {
            let (sensor, movee, score, next_sensor) = self.all[i].data.clone();
            ret[c] = (sensor, movee, score, next_sensor);
            c += 1;
        }
        Some(ret)
    }
    /// This removes from the container, so be sure to readd them
    fn get_best<const N: usize>(
        &mut self,
    ) -> Option<[([f32; INPUTS], M, f32, Option<[f32; INPUTS]>); N]>
    where
        M: Default,
    {
        if self.all.len() < REPLAY_MIN {
            return None;
        }
        if self.bests.len() <= N {
            return None;
        }
        // self.0.sort_unstable_by_key(|(_, _, _, _, per)| *per);
        let mut ret: [([f32; INPUTS], M, f32, Option<[f32; INPUTS]>); N] = [0; N].map(|_| {
            (
                [0.; INPUTS],
                Default::default(),
                Default::default(),
                Default::default(),
            )
        });
        let mut c = 0;
        for _ in self.bests.len() - N - 1..self.bests.len() - 1 {
            let p = self.bests.pop().unwrap();
            let (sensor, movee, score, next_sensor) = p.data;
            ret[c] = (sensor, movee, score, next_sensor);
            c += 1;
        }
        // dbg!(&ret);
        Some(ret)
    }
    fn insert(
        &mut self,
        sensor: [f32; INPUTS],
        movee: M,
        score: f32,
        next_sensor: Option<[f32; INPUTS]>,
        t: PER,
    ) {
        let rs = ReplayRecord {
            data: (sensor, movee, score, next_sensor),
            key: t,
        };
        self.all.push(rs.clone());
        if self.all.len() > REPLAY_MEM_OVERFLOW {
            let nv = self.all[self.all.len() - REPLAY_MEM..].to_vec();
            self.all = nv;
        }
        self.insert_best(rs);
    }
    fn insert_best(&mut self, rs: ReplayRecord<INPUTS, MOVES, M>) {
        self.bests.push(rs);
        if self.bests.len() > REPLAY_MEM_BEST_OVERFLOW {
            // replay.0.swap_remove(0);
            let mut s = BinaryHeap::default();
            std::mem::swap(&mut s, &mut self.bests);
            let mut tv = s.into_vec();
            tv.truncate(REPLAY_MEM_BEST);
            self.bests = tv.into();
        }
    }
    fn len(&self) -> usize {
        self.all.len()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum MoveDir {
    U,
    UR,
    R,
    DR,
    D,
    DL,
    L,
    UL,
    #[default]
    S,
}
#[derive(Event, Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Move {
    dir: MoveDir,
    shoot: bool,
}

trait Vectorable<const LEN: usize>: From<u8> + Into<u8> + Copy {
    fn vectorize(&self) -> [f32; LEN] {
        let mut r = [0.; LEN];
        r[<Self as Into<u8>>::into(*self) as usize] = 1.0;
        r
    }
}

#[derive(Event, PartialEq, Clone, Copy)]
enum MoveOutcome {
    EnemyHit,
    EnemyDie,
    Die,
}

#[derive(Event, PartialEq, Eq, Clone, Copy)]
enum MoveOutcomeCart {
    Dead,
}

impl From<MoveOutcomeCart> for f32 {
    fn from(val: MoveOutcomeCart) -> f32 {
        match val {
            MoveOutcomeCart::Dead => -1.,
        }
    }
}

impl From<Move> for u8 {
    fn from(value: Move) -> u8 {
        let x: u8 = match value.dir {
            MoveDir::U => 0,
            MoveDir::UR => 1,
            MoveDir::R => 2,
            MoveDir::DR => 3,
            MoveDir::D => 4,
            MoveDir::DL => 5,
            MoveDir::L => 6,
            MoveDir::UL => 7,
            MoveDir::S => 8,
        };
        let y = if value.shoot { 9 } else { 0 };
        x + y
    }
}
impl Vectorable<18> for Move {}
// impl From<Move> for [f32; MOVES] {
//     fn from(value: Move) -> Self {
//         let mut ret = [0.; MOVES];
//         ret[match value.dir {
//             // MoveDir::U => 0,
//             // MoveDir::UR => 1,
//             MoveDir::R => 0,
//             // MoveDir::DR => 3,
//             // MoveDir::D => 4,
//             // MoveDir::DL => 5,
//             MoveDir::L => 1,
//             // MoveDir::UL => 7,
//             // MoveDir::S => 8,
//             _ => unreachable!("Only left and right allowed"),
//         } + if value.shoot { MOVES / 2 } else { 0 }] = 1.0;
//         ret
//     }
// }
impl From<u8> for Move {
    fn from(value: u8) -> Self {
        Self {
            dir: match value % 9 {
                0 => MoveDir::U,
                1 => MoveDir::UR,
                2 => MoveDir::R,
                3 => MoveDir::DR,
                4 => MoveDir::D,
                5 => MoveDir::DL,
                6 => MoveDir::L,
                7 => MoveDir::UL,
                8 => MoveDir::S,
                _ => unreachable!("greter generated"),
            },
            shoot: value / 9 == 1,
        }
    }
}
// impl From<u8> for Move {
//     fn from(value: u8) -> Self {
//         Self {
//             dir: match value % 2 {
//                 0 => MoveDir::R,
//                 1 => MoveDir::L,
//                 _ => unreachable!("greater generated"),
//             },
//             shoot: value / 2 == 1,
//         }
//     }
// }
impl<const INPUTS: usize, const HLS: usize, const MOVES: usize> AIControl<INPUTS, HLS, MOVES> {
    fn load_or_new<P: AsRef<Path>>(path: P) -> (Self, bool) {
        let path = path.as_ref();
        let dev = NetDevice::default();
        let mut net = dev.build_module::<f32>(ThingNet::default());
        let mut saved = true;
        if let Err(e) = net.load_safetensors(path) {
            println!("Couldnt load {e:?}");
            saved = false;
            // net.fillc
            net.save_safetensors(path).unwrap();
        }
        let optim = NetOptim::new(&net, *NetOptimConfig);
        (
            Self {
                optim,
                net,
                device: dev,
            },
            saved,
        )
    }
    fn load_or_clone<P: AsRef<Path>>(path: P, net_ref: &Self) -> (Self, bool) {
        let path = path.as_ref();
        let dev = NetDevice::default();
        let mut net = dev.build_module::<f32>(ThingNet::default());
        let mut saved = true;
        if let Err(e) = net.load_safetensors(path) {
            // println!("Couldnt load {e:?}");
            println!("Loaded clone {e:?}");
            net = net_ref.net.clone();
            saved = false;
            // net.fillc
            net.save_safetensors(path).unwrap();
        }
        let optim = NetOptim::new(&net, *NetOptimConfig);
        (
            Self {
                optim,
                net,
                device: dev,
            },
            saved,
        )
    }
    fn save<P: AsRef<Path>>(&self, path: P) {
        self.net.save_safetensors(path).unwrap();
    }
    fn forward(&self, data: &[f32; INPUTS]) -> [f32; MOVES] {
        let input = self.device.tensor(data);
        self.forward_raw(input).array()
    }
    fn get_move<T: From<u8>>(&self, data: &[f32; INPUTS], explore: f32, r: &mut impl Rng) -> T {
        if explore > r.gen_range(0f32..1f32) {
            r.gen_range(0u8..(MOVES as u8)).into()
        } else {
            let out = self.forward(data);
            // #[cfg(not(feature = "train"))]
            let max = out.iter().fold(f32::NEG_INFINITY, |x, y| x.max(*y));
            // let min = out.iter().fold(f32::INFINITY, |x, y| x.min(*y));
            // #[cfg(feature = "train")]
            // let max = SliceRandom::choose_weighted(&out as &[f32], r, |s| {
            //     ((s * 5.) as f64).exp().min(10f64.powi(9))
            // })
            // .unwrap()
            // .clone();
            (out.into_iter().position(|x| x == max).unwrap() as u8).into()
        }
    }
    fn forward_raw<T: Tape<f32, NetDevice>>(
        &self,
        data: Tensor<Rank1<INPUTS>, f32, NetDevice, T>,
    ) -> Tensor<Rank1<MOVES>, f32, NetDevice, T> {
        self.net.forward(data)
    }
    fn forward_raw_batch<T: Tape<f32, NetDevice>, const N: usize>(
        &self,
        data: Tensor<Rank2<N, INPUTS>, f32, NetDevice, T>,
    ) -> Tensor<Rank2<N, MOVES>, f32, NetDevice, T> {
        self.net.forward(data)
    }
    fn train_one<const N: usize, M: Vectorable<MOVES>>(
        &mut self,
        stable: Option<&Self>,
        data: [[f32; INPUTS]; N],
        moved: [M; N],
        score: [f32; N],
        after_data: [Option<[f32; INPUTS]>; N],
    ) -> [f32; N] {
        self.train_m(
            stable,
            // data.map(|x| [x]),
            // moved.map(|x| [x]),
            // score.map(|x| [x]),
            [data],
            moved,
            [score],
            after_data,
        )
    }
    fn train_m<const M: usize, const N: usize, MV: Vectorable<MOVES>>(
        &mut self,
        stable: Option<&Self>,
        data: [[[f32; INPUTS]; N]; M],
        moved: [MV; N],
        score: [[f32; N]; M],
        after_data: [Option<[f32; INPUTS]>; N],
    ) -> [f32; N] {
        let grads = self.net.alloc_grads();
        let input = self.device.tensor(data[0]).traced(grads);
        let output = self.forward_raw_batch(input);

        let score = self.device.tensor(score);
        let moves_array: [[f32; MOVES]; N] = moved
            .into_iter()
            .map(|x| x.vectorize())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let movemask = self.device.tensor(moves_array);
        let output_masked: Tensor<Rank1<N>, f32, NetDevice, _> = (output * movemask.clone()).sum();

        let after_mask = self
            .device
            .tensor(after_data.map(|x| if x.is_some() { 1.0 } else { 0.0 }));
        let after_input = self
            .device
            .tensor(after_data.map(|x| x.unwrap_or([0.; INPUTS])));
        let after_output = stable.map(|stable| stable.forward_raw_batch(after_input));

        let after_masked: Tensor<Rank1<N>, f32, NetDevice, _> = after_output
            .map(|x| x.max())
            .unwrap_or(self.device.tensor(0.).broadcast());

        let score_weight: Tensor<Rank1<M>, f32, NetDevice, _> = self
            .device
            .tensor_from_vec((0..M).map(|x| GAMMA.powi(x as i32)).collect(), (Const,));
        let score_sum: Tensor<Rank1<N>, f32, NetDevice, _> =
            (score * score_weight.broadcast()).sum();

        let gamma = self.device.tensor(GAMMA.powi(M as i32));
        let error = output_masked - (after_masked * gamma.broadcast() * after_mask + score_sum);
        let mseloss = error.powi(2);
        let lossr = mseloss.array();
        let mseloss = mseloss.mean().backward();
        self.optim.update(&mut self.net, &mseloss).unwrap();
        lossr
    }
}

fn make_player(mut commands: Commands, #[cfg(not(feature = "train"))] assets: Res<AssetServer>) {
    let player = PlayerType::AI;

    commands.spawn((
        Player {
            ty: player,
            alive: 0.,
        },
        #[cfg(not(feature = "train"))]
        SpriteBundle {
            texture: assets.load("c.png"),
            sprite: Sprite {
                custom_size: Some(Vec2::splat(40.)),
                ..Default::default()
            },
            ..Default::default()
        },
        #[cfg(feature = "train")]
        Transform::from_xyz(0., 0., 0.),
    ));
    #[cfg(not(feature = "train"))]
    commands.spawn(Camera2dBundle::default());
    commands.insert_resource(ShootTimer(Timer::new(
        Duration::from_millis(500),
        bevy::time::TimerMode::Repeating,
    )));
}

fn window_size(mut windows: Query<&mut Window>) {
    let Ok(mut window) = windows.get_single_mut() else {
        return;
    };
    if window.resolution.width() != 640. {
        window.resolution = WindowResolution::new(640., 640.);
    } else {
        window.resizable = false;
    }
}

struct PrioQueue<C: PartialOrd, V>(Vec<(C, V)>, usize);

impl<C: PartialOrd, V> PrioQueue<C, V> {
    fn new(cap: usize) -> Self {
        Self(Vec::with_capacity(cap), cap)
    }
    fn insert(&mut self, prio: C, val: V) {
        let ipos = self.0.iter().position(|x| prio > x.0);
        // println!("{ipos:?}");
        match ipos {
            Some(n) => {
                // println!("Better");
                self.0.insert(n, (prio, val));
                if self.0.len() > self.1 {
                    self.0.remove(self.0.len() - 1);
                }
            }
            None => {
                // println!("Not full");
                if self.0.len() < self.1 {
                    self.0.push((prio, val));
                }
            }
        }
    }
    fn extend<I: Iterator<Item = (C, V)>>(&mut self, ext: I) {
        ext.for_each(|(c, v)| self.insert(c, v));
    }
}
#[derive(Event)]
struct Sensor<const INPUTS: usize>([f32; INPUTS]);
fn game_to_vec(
    player: Vec2,
    enemies: impl Iterator<Item = Vec2>,
    enemy_bullets: impl Iterator<Item = Vec2>,
    bullets: impl Iterator<Item = Vec2>,
) -> [f32; INPUTS_D] {
    let mut close_enemies = PrioQueue::<f32, Vec2>::new(AWARES);
    let mut close_ebullets = PrioQueue::<f32, Vec2>::new(AWARES);
    let mut close_bullets = PrioQueue::<f32, Vec2>::new(AWARES);
    let empty = [(f32::NEG_INFINITY, Vec2::splat(2.)); AWARES];
    close_ebullets.extend(empty.clone().into_iter());
    close_enemies.extend(empty.into_iter());
    close_bullets.extend(empty.into_iter());
    close_enemies.extend(enemies.map(|t| {
        let td = t - player;
        (-td.norm_squared(), td / 640.)
    }));
    close_ebullets.extend(enemy_bullets.map(|t| {
        let td = t - player;
        (-td.norm_squared(), td / 640.)
    }));
    close_bullets.extend(bullets.map(|t| {
        let dtop = t.y - 320.;
        let td = t - player;
        (dtop, td / 640.)
    }));

    let data: Vec<f32> = close_enemies
        .0
        .into_iter()
        .chain(close_ebullets.0.into_iter())
        .chain(close_bullets.0.into_iter())
        .flat_map(|(_, p)| [p.x, p.y])
        .collect();
    data.try_into().unwrap()
}
fn gen_move_ai(
    net: NonSend<BleedingD>,
    mut rand: ResMut<RandRes>,
    mut moves: EventWriter<Move>,
    mut inputs: EventWriter<Sensor<INPUTS_D>>,
    player: Query<&Transform, With<Player>>,
    enemies: Query<&Transform, With<Enemy>>,
    enemy_bullets: Query<&Transform, With<EnemyBullet>>,
    bullets: Query<&Transform, With<PlayerBullet>>,
) {
    let player = player.get_single().unwrap().translation.xy();
    let data = game_to_vec(
        player,
        enemies.iter().map(|x| x.translation.xy()),
        enemy_bullets.iter().map(|x| x.translation.xy()),
        bullets.iter().map(|x| x.translation.xy()),
    );
    moves.send(
        net.net
            .get_move(&data, *EXPLORE.lock().unwrap(), &mut rand.0),
    );
    inputs.send(Sensor(data));
}
fn put_after(
    mut afters: EventWriter<AfterSensor<INPUTS_D>>,
    mut outcomes: EventReader<MoveOutcome>,
    player: Query<&Transform, With<Player>>,
    enemies: Query<&Transform, With<Enemy>>,
    enemy_bullets: Query<&Transform, With<EnemyBullet>>,
    bullets: Query<&Transform, With<PlayerBullet>>,
) {
    let outs: Vec<_> = outcomes.read().collect();
    if !outs.contains(&&MoveOutcome::Die) {
        let player = player.get_single().unwrap().translation.xy();
        let data = game_to_vec(
            player,
            enemies.iter().map(|x| x.translation.xy()),
            enemy_bullets.iter().map(|x| x.translation.xy()),
            bullets.iter().map(|x| x.translation.xy()),
        );
        afters.send(AfterSensor(Some(data)));
    } else {
        afters.send(AfterSensor(None));
    }
}
#[derive(Event)]
struct AfterSensor<const INPUTS: usize>(Option<[f32; INPUTS]>);
fn update_ai(
    mut inputs: EventReader<Sensor<INPUTS_D>>,
    mut afters: EventReader<AfterSensor<INPUTS_D>>,
    mut moves: EventReader<Move>,
    mut outcomes: EventReader<MoveOutcome>,
    mut replay: ResMut<Replay<INPUTS_D, MOVES_D, Move>>,
    mut bled: NonSendMut<BleedingD>,
    mut sage: NonSendMut<SageD>,
    mut rand: ResMut<RandRes>,
) {
    let mut lossb = 0.;
    let mut loss = 0.;
    update_ai_gen(
        &mut inputs,
        &mut afters,
        &mut moves,
        &mut outcomes,
        &mut replay,
        &mut bled.net,
        &mut sage.net,
        &mut rand.0,
        |x| lossb += x,
        |x| loss += x,
    );
    sage.acc_loss_b += lossb;
    sage.acc_loss += loss;

    sage.frames_left -= 1;

    if sage.frames_left == 0 {
        // if replay.len() >= REPLAY_MIN {
        // println!(
        //     "{:?}",
        //     bled.net.forward(&[
        //         0.4212548, 0.90625, 0.67328036, 0.90625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        //         0.21659437, 0.31132317, 0.56151307, 0.755703, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        //     ])
        // );
        // }
        if replay.len() >= REPLAY_MIN {
            // let alive_rat = sage.alive_time / sage.games;
            // if alive_rat > sage.last_alive_rat || rand.0.gen::<f64>() < IGNORE_BAD_CHANCE {
            //     sage.net = bled.0.clone();
            //     bled.0.save("qnet");
            //     println!("Saved {}>{}", alive_rat, sage.last_alive_rat);
            //     sage.last_alive_rat = alive_rat;
            // } else {
            //     println!("Bad {}>{}", sage.last_alive_rat, alive_rat);
            //     if rand.0.gen::<f64>() < RESET_CHANCE {
            //         bled.0 = sage.net.clone();
            //         println!("Reset bad");
            //     }
            // }
            if sage.acc_loss < *MAX_AVG_LOSS.lock().unwrap() * (FRAMES_PER_CYCLE as f32) {
                // sage.net = bled.net.clone();
                bled.net.save("qnet");
                sage.net = AIControl::load_or_new("qnet").0;
                // println!("Saved {}", alive_rat);
                // sage.last_alive_rat = alive_rat;

                // bled.net.optim.cfg.lr = *LR.lock().unwrap();

                // bled.net.optim = NetOptim::new(&bled.net.net, bled.net.optim.cfg);
                // sage.net.optim = NetOptim::new(&sage.net.net, sage.net.optim.cfg);
                sage.first_run = false;

                // Maybe, idk
            }
            bled.net.save("qnet_bled");
            if sage.acc_loss > bled.last_loss {
                // bled.net.optim.cfg.lr *= *LR_DECAY.lock().unwrap();
            }
            bled.last_loss = sage.acc_loss;
            bled.last_loss_b = sage.acc_loss_b;
        }
        sage.frames_left = FRAMES_PER_CYCLE;
        // println!(
        //     "Generation lasted {} on avg, took {:?}",
        //     (sage.alive_time) / (sage.games as f32),
        //     SystemTime::now().duration_since(sage.runtime).unwrap()
        // );

        {
            let mut past_explore = *EXPLORE.lock().unwrap();
            past_explore = (past_explore.recip() + *EXPLORE_DECAY.lock().unwrap()).recip();
            past_explore = past_explore.max(EXPLORE_LIMIT);
            *EXPLORE.lock().unwrap() = past_explore;
            std::fs::write("explore", format!("{}", EXPLORE.lock().unwrap())).unwrap();

            let oldlr = bled.net.optim.cfg.lr;
            let newlr = (oldlr.recip() + *LR_DECAY.lock().unwrap()).recip();
            bled.net.optim.cfg.lr = newlr;
            std::fs::write("lr", format!("{newlr}")).unwrap();
        }

        if sage.games > 25. {
            // if let Some(pinput) = replay
            //     .bests
            //     .iter()
            //     .filter(|x| matches!(x.key, PER::Last(_)))
            //     .next()
            // {
            // println!("{:?} {:?}", bled.net.forward(&pinput.data.0), pinput);
            // println!("{:?}", bled.net.forward(&[0.; INPUTS]));
            // }
            println!(
                "Alive avg: {} in {}",
                sage.alive_time / sage.games,
                sage.games
            );
            let frames = sage.alive_time * 50.;
            sage.alive_time = 0.;
            sage.games = 0.;
            println!(
                "{} EXP:{} *{} avg loss:{}/{} bestloss:{} LR:{}",
                replay.len(),
                EXPLORE.lock().unwrap(),
                EXPLORE_DECAY.lock().unwrap(),
                sage.acc_loss / frames,
                MAX_AVG_LOSS.lock().unwrap(),
                sage.acc_loss_b / frames,
                bled.net.optim.cfg.lr,
            );
            sage.acc_loss = 0.;
            sage.acc_loss_b = 0.;
            // replay.bests.clear();
        }
        sage.runtime = SystemTime::now();
        // replay.0.push(vec![]);
    }
}
fn update_ai_cart(
    mut inputs: EventReader<Sensor<INPUTS_C>>,
    mut afters: EventReader<AfterSensor<INPUTS_C>>,
    mut moves: EventReader<CartMove>,
    mut outcomes: EventReader<MoveOutcomeCart>,
    mut replay: ResMut<Replay<INPUTS_C, MOVES_C, CartMove>>,
    mut bled: NonSendMut<BleedingC>,
    mut sage: NonSendMut<SageC>,
    mut rand: ResMut<RandRes>,
) {
    let mut lossb = 0.;
    let mut loss = 0.;
    update_ai_gen(
        &mut inputs,
        &mut afters,
        &mut moves,
        &mut outcomes,
        &mut replay,
        &mut bled.net,
        &mut sage.net,
        &mut rand.0,
        |x| lossb += x,
        |x| loss += x,
    );
    sage.acc_loss_b += lossb;
    sage.acc_loss += loss;

    sage.frames_left -= 1;

    if sage.frames_left == 0 {
        // if replay.len() >= REPLAY_MIN {
        // println!(
        //     "{:?}",
        //     bled.net.forward(&[
        //         0.4212548, 0.90625, 0.67328036, 0.90625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        //         0.21659437, 0.31132317, 0.56151307, 0.755703, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        //     ])
        // );
        // }
        if replay.len() >= REPLAY_MIN {
            // let alive_rat = sage.alive_time / sage.games;
            // if alive_rat > sage.last_alive_rat || rand.0.gen::<f64>() < IGNORE_BAD_CHANCE {
            //     sage.net = bled.0.clone();
            //     bled.0.save("qnet");
            //     println!("Saved {}>{}", alive_rat, sage.last_alive_rat);
            //     sage.last_alive_rat = alive_rat;
            // } else {
            //     println!("Bad {}>{}", sage.last_alive_rat, alive_rat);
            //     if rand.0.gen::<f64>() < RESET_CHANCE {
            //         bled.0 = sage.net.clone();
            //         println!("Reset bad");
            //     }
            // }
            if sage.acc_loss < *MAX_AVG_LOSS.lock().unwrap() * (FRAMES_PER_CYCLE as f32) {
                // sage.net = bled.net.clone();
                bled.net.save("qnet_cart");
                sage.net = AIControl::load_or_new("qnet_cart").0;
                // println!("Saved {}", alive_rat);
                // sage.last_alive_rat = alive_rat;

                // bled.net.optim.cfg.lr = *LR.lock().unwrap();

                // bled.net.optim = NetOptim::new(&bled.net.net, bled.net.optim.cfg);
                // sage.net.optim = NetOptim::new(&sage.net.net, sage.net.optim.cfg);
                sage.first_run = false;

                // Maybe, idk
            }
            bled.net.save("qnet_cart_bled");
            if sage.acc_loss > bled.last_loss {
                // bled.net.optim.cfg.lr *= *LR_DECAY.lock().unwrap();
            }
            bled.last_loss = sage.acc_loss;
            bled.last_loss_b = sage.acc_loss_b;
        }
        sage.frames_left = FRAMES_PER_CYCLE;
        // println!(
        //     "Generation lasted {} on avg, took {:?}",
        //     (sage.alive_time) / (sage.games as f32),
        //     SystemTime::now().duration_since(sage.runtime).unwrap()
        // );

        {
            let mut past_explore = *EXPLORE.lock().unwrap();
            past_explore = (past_explore.recip() + *EXPLORE_DECAY.lock().unwrap()).recip();
            past_explore = past_explore.max(EXPLORE_LIMIT);
            *EXPLORE.lock().unwrap() = past_explore;
            std::fs::write("explore", format!("{}", EXPLORE.lock().unwrap())).unwrap();

            let oldlr = bled.net.optim.cfg.lr;
            let newlr = (oldlr.recip() + *LR_DECAY.lock().unwrap()).recip();
            bled.net.optim.cfg.lr = newlr;
            std::fs::write("lr", format!("{newlr}")).unwrap();
        }

        if sage.games > 25. {
            // if let Some(pinput) = replay
            //     .bests
            //     .iter()
            //     .filter(|x| matches!(x.key, PER::Last(_)))
            //     .next()
            // {
            // println!("{:?} {:?}", bled.net.forward(&pinput.data.0), pinput);
            // println!("{:?}", bled.net.forward(&[0.; INPUTS]));
            // }
            println!(
                "Alive avg: {} in {}",
                sage.alive_time / sage.games,
                sage.games
            );
            let frames = sage.alive_time * 50.;
            sage.alive_time = 0.;
            sage.games = 0.;
            println!(
                "{} EXP:{} *{} avg loss:{}/{} bestloss:{} LR:{}",
                replay.len(),
                EXPLORE.lock().unwrap(),
                EXPLORE_DECAY.lock().unwrap(),
                sage.acc_loss / frames,
                MAX_AVG_LOSS.lock().unwrap(),
                sage.acc_loss_b / frames,
                bled.net.optim.cfg.lr,
            );
            sage.acc_loss = 0.;
            sage.acc_loss_b = 0.;
            // replay.bests.clear();
        }
        sage.runtime = SystemTime::now();
        // replay.0.push(vec![]);
    }
}

fn update_ai_gen<
    const INPUTS: usize,
    const MOVES: usize,
    M: Vectorable<MOVES> + Event + PartialEq + Debug + Default,
    O: Into<f32> + Event + Copy,
>(
    inputs: &mut EventReader<Sensor<INPUTS>>,
    afters: &mut EventReader<AfterSensor<INPUTS>>,
    moves: &mut EventReader<M>,
    outcomes: &mut EventReader<O>,
    replay: &mut Replay<INPUTS, MOVES, M>,
    bled: &mut AIControl<INPUTS, HLS, MOVES>,
    sage: &mut AIControl<INPUTS, HLS, MOVES>,
    rand: &mut impl Rng,
    mut lossb: impl FnMut(f32),
    mut loss: impl FnMut(f32),
) {
    let input = inputs.read().next().unwrap().0.clone();
    let after = afters.read().next().unwrap().0.as_ref().map(|x| x.clone());
    let outs: Vec<_> = outcomes.read().collect();
    let score = outs
        .iter()
        .map(|x| <O as Into<f32>>::into(**x))
        .sum::<f32>();
    let movec = moves.read().next().unwrap();
    replay.insert(input, *movec, score, after, PER::Never);
    // #[cfg(not(feature = "train"))]
    // {
    //     println!("{input:?}");
    // }

    if replay.len() % REPLAY_INTERVAL == 0 {
        if let Some(best) = replay.get_best::<REPLAY_FRAME_BEST>() {
            let ts = bled.train_one(
                Some(sage),
                best.map(|x| x.0),
                best.map(|x| x.1),
                best.map(|x| x.2),
                best.map(|x| x.3),
            );
            // sage.acc_loss_b += ts.iter().sum::<f32>();
            lossb(ts.iter().sum());
            for (b, t) in best.into_iter().zip(ts.into_iter()) {
                replay.insert_best(ReplayRecord {
                    data: b,
                    key: PER::Last(t),
                });
            }
        }
        if let Some(last) = replay.get_last::<REPLAY_FRAME_LAST>() {
            let ts = bled.train_one(
                Some(sage),
                last.map(|x| x.0),
                last.map(|x| x.1),
                last.map(|x| x.2),
                last.map(|x| x.3),
            );
            // sage.acc_loss += ts.iter().sum::<f32>();
            loss(ts.iter().sum());
            for (l, t) in last.into_iter().zip(ts.into_iter()) {
                replay.insert_best(ReplayRecord {
                    data: l,
                    key: PER::Last(t),
                });
            }
        }
        if let Some(rands) = replay.get_rand::<_, REPLAY_FRAME_RAND>(rand) {
            let ts = bled.train_one(
                Some(sage),
                rands.map(|x| x.0),
                rands.map(|x| x.1),
                rands.map(|x| x.2),
                rands.map(|x| x.3),
            );
            // sage.acc_loss += ts.iter().sum::<f32>();
            loss(ts.iter().sum());
            for (r, t) in rands.into_iter().zip(ts.into_iter()) {
                replay.insert_best(ReplayRecord {
                    data: r,
                    key: PER::Last(t),
                });
            }
        }
    }
}

fn move_player(
    mut player: Query<(&mut Transform, &mut Player)>,
    // #[cfg(not(feature = "train"))] keys: Res<ButtonInput<KeyCode>>,
    mut moves: EventReader<Move>,
) {
    let Ok((mut player_trans, mut player)) = player.get_single_mut() else {
        return;
    };
    // println!("pther move");
    let mut vel = Vec2::default();
    let delta = 200.;
    if matches!(player.ty, PlayerType::Keyboard) {
        // #[cfg(not(feature = "train"))]
        // {
        //     if keys.pressed(KeyCode::ArrowLeft) {
        //         vel.x -= 1.0;
        //     }
        //     if keys.pressed(KeyCode::ArrowRight) {
        //         vel.x += 1.0;
        //     }
        //     if keys.pressed(KeyCode::ArrowDown) {
        //         vel.y -= 1.0;
        //     }
        //     if keys.pressed(KeyCode::ArrowUp) {
        //         vel.y += 1.0;
        //     }
        // }
    } else if matches!(player.ty, PlayerType::AI) {
        let movec = moves.read().next().unwrap();
        vel = match movec.dir {
            MoveDir::U => Vec2::new(0., 1.),
            MoveDir::UR => Vec2::new(1., 1.),
            MoveDir::R => Vec2::new(1., 0.),
            MoveDir::DR => Vec2::new(1., -1.),
            MoveDir::D => Vec2::new(0., -1.),
            MoveDir::DL => Vec2::new(-1., -1.),
            MoveDir::L => Vec2::new(-1., 0.),
            MoveDir::UL => Vec2::new(-1., 1.),
            MoveDir::S => Vec2::new(0., 0.),
        };
    }
    player.alive += TIMEDELTA;
    let vel = vel.normalize_or_zero() * delta * TIMEDELTA;
    player_trans.translation += vel.xyy();
    player_trans.translation = player_trans
        .translation
        .clamp(Vec3::new(-300., -300., 0.), Vec3::new(300., 300., 0.));
}

fn shoot_player(
    player: Query<&Transform, With<Player>>,
    mut commands: Commands,
    // #[cfg(not(feature = "train"))] keys: Res<ButtonInput<KeyCode>>,
    #[cfg(not(feature = "train"))] assets: Res<AssetServer>,
    mut shoot_timer: ResMut<ShootTimer>,
    mut moves: EventReader<Move>,
) {
    let Ok(player) = player.get_single() else {
        return;
    };
    // println!("pther shoot");
    shoot_timer.0.tick(Duration::from_secs_f32(TIMEDELTA));
    if shoot_timer.0.finished() {
        // #[cfg(not(feature = "train"))]
        // if keys.pressed(KeyCode::KeyZ) {
        let movec = moves.read().next().unwrap();
        if movec.shoot {
            commands.spawn((
                PlayerBullet,
                #[cfg(not(feature = "train"))]
                SpriteBundle {
                    texture: assets.load("c.png"),
                    transform: player.clone(),
                    sprite: Sprite {
                        custom_size: Some(Vec2::splat(10.)),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                #[cfg(feature = "train")]
                player.clone(),
            ));
        }
    }
}

fn move_player_bullet(
    mut bullets: Query<(Entity, &mut Transform), With<PlayerBullet>>,
    mut commands: Commands,
) {
    for (e, mut t) in bullets.iter_mut() {
        t.translation.y += 400. * TIMEDELTA;
        if t.translation.y > 330. {
            // println!("Despawn bullet");
            commands.entity(e).despawn();
        }
    }
}

#[derive(Component)]
struct Enemy {
    health: usize,
    shoot: Timer,
}

#[derive(Component)]
struct EnemyBullet(Vec3);

#[derive(Resource)]
struct EnemyTimer(Timer);

fn make_enemy(mut commands: Commands) {
    commands.insert_resource(EnemyTimer(Timer::new(
        Duration::from_secs_f32(5.0),
        bevy::time::TimerMode::Repeating,
    )));
}

fn spawn_enemy(
    mut commands: Commands,
    mut enemy_timer: ResMut<EnemyTimer>,
    #[cfg(not(feature = "train"))] assets: Res<AssetServer>,
    mut rand: ResMut<RandRes>,
) {
    enemy_timer.0.tick(Duration::from_secs_f32(TIMEDELTA));
    // println!("tspawn");
    if enemy_timer.0.finished() {
        let mut timer = Timer::new(
            Duration::from_secs_f32(1.8),
            bevy::time::TimerMode::Repeating,
        );
        // println!("spawn");
        timer.set_elapsed(Duration::from_secs_f32(rand.0.gen_range(0f32..1f32)));
        commands.spawn((
            Enemy {
                health: 10,
                shoot: timer,
            },
            #[cfg(not(feature = "train"))]
            SpriteBundle {
                texture: assets.load("c.png"),
                transform: Transform::from_xyz(rand.0.gen_range(-290f32..290f32), 280., 0.),
                sprite: Sprite {
                    custom_size: Some(Vec2::splat(30.)),
                    ..Default::default()
                },
                ..Default::default()
            },
            #[cfg(feature = "train")]
            Transform::from_xyz(rand.0.gen_range(-290f32..290f32), 280., 0.),
        ));
    }
}

fn shoot_enemy(
    mut enemies: Query<(&Transform, &mut Enemy)>,
    mut commands: Commands,
    #[cfg(not(feature = "train"))] assets: Res<AssetServer>,
    player: Query<&Transform, With<Player>>,
) {
    // println!("try shoot1");
    // println!("{:?}", player.iter().collect::<Vec<_>>());
    let Some(ploc) = player.get_single().ok().map(|x| x.translation) else {
        return;
    };
    // println!("pthere shoot enemy");
    for (et, mut timer) in enemies.iter_mut() {
        // println!("try shoot");
        if timer
            .shoot
            .tick(Duration::from_secs_f32(TIMEDELTA))
            .finished()
        {
            // println!("shoot");
            commands.spawn((
                EnemyBullet(ploc - et.translation),
                #[cfg(not(feature = "train"))]
                SpriteBundle {
                    texture: assets.load("c.png"),
                    transform: et.clone(),
                    sprite: Sprite {
                        custom_size: Some(Vec2::splat(10.)),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                #[cfg(feature = "train")]
                et.clone(),
            ));
        }
    }
}

fn move_enemy_bullet(
    mut bullets: Query<(Entity, &mut Transform, &EnemyBullet)>,
    mut commands: Commands,
) {
    for (be, mut bt, bb) in bullets.iter_mut() {
        if let Some(tnorm) = bb.0.try_normalize() {
            bt.translation += tnorm * 200. * TIMEDELTA;
        } else {
            // println!("Despawn ebullet");
            commands.entity(be).despawn();
        }
        if bt.translation.norm_squared() > 320. * 320. * 2. {
            // println!("Despawn ebullet");
            commands.entity(be).despawn();
        }
    }
}

fn hit_enemy(
    player_bullets: Query<(Entity, &Transform), With<PlayerBullet>>,
    mut enemies: Query<(Entity, &Transform, &mut Enemy)>,
    mut commands: Commands,
    mut outcomes: EventWriter<MoveOutcome>,
) {
    let mut bmap: HashMap<_, _> = player_bullets.iter().collect();
    for (ee, et, mut een) in enemies.iter_mut() {
        let mut hitb = None;
        for (be, bt) in bmap.iter() {
            if et.translation.distance_squared(bt.translation) < 20. * 20. {
                hitb = Some(*be);
                break;
            }
        }
        if let Some(hitb) = hitb {
            een.health -= 1;
            if een.health == 0 {
                // println!("Despawn enemy");
                commands.entity(ee).despawn();
                outcomes.send(MoveOutcome::EnemyDie);
            } else {
                outcomes.send(MoveOutcome::EnemyHit);
            }
            // println!("Despawn bullet");
            commands.entity(hitb).despawn();
            bmap.remove(&hitb);
            // ReLU
        }
    }
}
fn hit_player(
    enemy_bullets: Query<(Entity, &Transform), With<EnemyBullet>>,
    enemies: Query<Entity, With<Enemy>>,
    mut players: Query<(Entity, &mut Transform, &mut Player), Without<EnemyBullet>>,
    mut commands: Commands,
    mut outcomes: EventWriter<MoveOutcome>,
    mut sage: NonSendMut<SageD>,
) {
    let Ok(mut player) = players.get_single_mut() else {
        return;
    };
    // println!("pthere hit");
    let mut hitb = false;
    for (_, bt) in enemy_bullets.iter() {
        if player.1.translation.distance_squared(bt.translation) < 5. * 5. {
            hitb = true;
            break;
        }
    }
    if hitb {
        // println!("Despawn enemies");
        enemies.iter().for_each(|e| commands.entity(e).despawn());

        // println!("Despawn ebullets");
        enemy_bullets
            .iter()
            .for_each(|e| commands.entity(e.0).despawn());
        outcomes.send(MoveOutcome::Die);
        // println!("Alive for {}", player.2.alive);
        sage.alive_time += player.2.alive;
        sage.games += 1.0;
        player.2.alive = 0.;
        player.1.translation = Vec3::splat(0.);
    }
}

#[derive(Resource)]
struct RandRes(StdRng);

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct GenMove;
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct UpdateAI;

fn danmaku() {
    // let stdin = stdin();
    std::thread::spawn(move || manage_consts());
    let mut a = App::new();
    a.insert_resource(RandRes(StdRng::from_entropy()))
        .add_systems(Startup, (make_player, make_enemy))
        .add_systems(
            Update,
            (
                move_player,
                window_size,
                shoot_player,
                move_player_bullet,
                spawn_enemy,
                shoot_enemy,
                move_enemy_bullet,
                hit_enemy,
                hit_player,
            ),
        );

    #[cfg(not(feature = "train"))]
    a.add_plugins(DefaultPlugins);
    #[cfg(feature = "train")]
    a.add_plugins(MinimalPlugins);

    let (net, loaded) = AIControl::load_or_new("qnet");
    let (netb, _) = AIControl::load_or_clone("qnet_bled", &net);
    a.insert_non_send_resource(SageD {
        net,
        frames_left: FRAMES_PER_CYCLE,
        alive_time: 0.,
        // last_alive_rat: 0.,
        acc_loss: 0.,
        games: 0.,
        runtime: SystemTime::now(),
        first_run: !loaded,
        acc_loss_b: 0.,
    })
    .insert_non_send_resource(BleedingD {
        net: netb,
        last_loss: f32::INFINITY,
        last_loss_b: f32::INFINITY,
    })
    .insert_resource(Replay::<INPUTS_D, MOVES_D, Move>::default())
    .add_event::<Move>()
    .add_event::<Sensor<INPUTS_D>>()
    .add_event::<AfterSensor<INPUTS_D>>()
    .add_event::<MoveOutcome>();
    // a.add_systems(Startup, add_ai_schedule);

    a.world_mut()
        .resource_mut::<MainScheduleOrder>()
        .insert_after(PreUpdate, GenMove);
    a.world_mut()
        .resource_mut::<MainScheduleOrder>()
        .insert_after(Update, UpdateAI);
    a.add_systems(GenMove, gen_move_ai);
    a.add_systems(
        UpdateAI,
        (put_after, update_consts, update_ai.after(put_after)),
    );

    a.run();
}

#[derive(Component, Debug)]
struct Cart {
    vel: f32,
    angvel: f32,
    alive: f32,
}

#[derive(Event, PartialEq, Eq, Clone, Copy, Debug)]
enum CartMove {
    Left,
    Right,
}
impl Default for CartMove {
    fn default() -> Self {
        Self::Left
    }
}

impl From<u8> for CartMove {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Left,
            1 => Self::Right,
            _ => unreachable!("Not possible"),
        }
    }
}
impl Into<u8> for CartMove {
    fn into(self) -> u8 {
        match self {
            CartMove::Left => 0,
            CartMove::Right => 1,
        }
    }
}
impl Vectorable<MOVES_C> for CartMove {}

fn make_cart(
    mut commands: Commands,
    #[cfg(not(feature = "train"))] assets: Res<AssetServer>,
    mut rand: ResMut<RandRes>,
) {
    commands.spawn((
        Cart {
            vel: rand.0.gen_range(-0.1..0.1),
            angvel: rand.0.gen_range(-0.1..0.1),
            alive: 0.,
        },
        #[cfg(not(feature = "train"))]
        SpriteBundle {
            texture: assets.load("bar.png"),
            sprite: Sprite {
                anchor: bevy::sprite::Anchor::BottomCenter,
                ..Default::default()
            },
            ..Default::default()
        },
        #[cfg(feature = "train")]
        Transform::from_xyz(rand.0.gen_range(-50.0..50.), 0., 0.),
    ));
    #[cfg(not(feature = "train"))]
    commands.spawn(Camera2dBundle::default());
    commands.insert_resource(ShootTimer(Timer::new(
        Duration::from_millis(500),
        bevy::time::TimerMode::Repeating,
    )));
    println!("Cart made");
}

fn move_cart(
    mut player: Query<(&mut Transform, &mut Cart)>,
    mut moves: EventReader<CartMove>,
    mut rand: ResMut<RandRes>,
    mut outcomes: EventWriter<MoveOutcomeCart>,
    mut sage: NonSendMut<SageC>,
    // keys: Res<ButtonInput<KeyCode>>,
) {
    let Ok((mut player_trans, mut cart)) = player.get_single_mut() else {
        return;
    };
    // println!("Moving cart");
    let mut newvel = cart.vel;
    match moves.read().next().unwrap() {
        CartMove::Left => newvel -= 200. * TIMEDELTA,
        CartMove::Right => newvel += 200. * TIMEDELTA,
    };
    // {
    // if keys.pressed(KeyCode::ArrowLeft) {
    // newvel -= 200. * TIMEDELTA;
    // }
    // if keys.pressed(KeyCode::ArrowRight) {
    // newvel += 200. * TIMEDELTA;
    // }
    // }
    let newpos = (player_trans.translation.x + newvel * TIMEDELTA).clamp(-200., 200.);
    let accel = if newpos == 200. && newvel > 0. || newpos == -200. && newvel < 0. {
        newvel = 0.;
        0.
    } else {
        newvel - cart.vel
    };
    let rot = player_trans.rotation.mul_vec3(Vec3::Y);
    let angle = rot.angle_between(Vec3::Y) * rot.x.signum();

    if angle.abs() > 3.14 / 2. {
        // println!("Bad");
        sage.alive_time += cart.alive;
        sage.games += 1.0;
        cart.alive = 0.;

        cart.vel = rand.0.gen_range(-0.1..0.1);
        cart.angvel = rand.0.gen_range(-0.1..0.1);
        *player_trans = Transform::from_xyz(rand.0.gen_range(-50.0..50.), 0., 0.);
        outcomes.send(MoveOutcomeCart::Dead);
    } else {
        // println!("acc:{accel} ang:{angle} v:{newvel}");
        cart.alive += TIMEDELTA;
        cart.angvel += accel * TIMEDELTA - angle * angle.abs() * TIMEDELTA;
        player_trans.rotate_z(cart.angvel * TIMEDELTA);
        cart.vel = newvel;
        player_trans.translation.x = newpos;
        // println!("{:?}, {:?}", player_trans, cart);
    }
}
fn gen_move_ai_cart(
    net: NonSend<BleedingC>,
    mut rand: ResMut<RandRes>,
    mut moves: EventWriter<CartMove>,
    mut inputs: EventWriter<Sensor<INPUTS_C>>,
    cart: Query<(&Transform, &Cart)>,
) {
    let Some((cart_t, cart)) = cart.get_single().ok() else {
        return;
    };
    let data = [
        cart_t.translation.x,
        cart_t.rotation.mul_vec3(Vec3::Y).angle_between(Vec3::Y),
        cart.vel,
        cart.angvel,
    ];
    moves.send(
        net.net
            .get_move(&data, *EXPLORE.lock().unwrap(), &mut rand.0),
    );
    inputs.send(Sensor(data));
}
fn put_after_cart(
    mut afters: EventWriter<AfterSensor<INPUTS_C>>,
    mut outcomes: EventReader<MoveOutcomeCart>,
    cart: Query<(&Transform, &Cart)>,
) {
    let outs: Vec<_> = outcomes.read().collect();
    if !outs.contains(&&MoveOutcomeCart::Dead) {
        let (cart_t, cart) = cart.get_single().unwrap();
        let data = [
            cart_t.translation.x,
            cart_t.rotation.mul_vec3(Vec3::Y).angle_between(Vec3::Y),
            cart.vel,
            cart.angvel,
        ];
        afters.send(AfterSensor(Some(data)));
    } else {
        afters.send(AfterSensor(None));
    }
}

fn cart() {
    std::thread::spawn(move || manage_consts());
    let mut a = App::new();
    a.insert_resource(RandRes(StdRng::from_entropy()))
        .add_systems(Update, move_cart)
        .add_systems(Startup, make_cart);

    #[cfg(not(feature = "train"))]
    a.add_plugins(DefaultPlugins);
    #[cfg(feature = "train")]
    a.add_plugins(MinimalPlugins);

    let (net, loaded) = AIControl::load_or_new("qnet_cart");
    let (netb, _) = AIControl::load_or_clone("qnet_cart_bled", &net);
    a.insert_non_send_resource(SageC {
        net,
        frames_left: FRAMES_PER_CYCLE,
        alive_time: 0.,
        // last_alive_rat: 0.,
        acc_loss: 0.,
        games: 0.,
        runtime: SystemTime::now(),
        first_run: !loaded,
        acc_loss_b: 0.,
    })
    .insert_non_send_resource(BleedingC {
        net: netb,
        last_loss: f32::INFINITY,
        last_loss_b: f32::INFINITY,
    })
    .insert_resource(Replay::<INPUTS_C, MOVES_C, CartMove>::default())
    .add_event::<CartMove>()
    .add_event::<Sensor<INPUTS_C>>()
    .add_event::<AfterSensor<INPUTS_C>>()
    .add_event::<MoveOutcomeCart>();
    // a.add_systems(Startup, add_ai_schedule);

    a.world_mut()
        .resource_mut::<MainScheduleOrder>()
        .insert_after(PreUpdate, GenMove);
    a.world_mut()
        .resource_mut::<MainScheduleOrder>()
        .insert_after(Update, UpdateAI);
    a.add_systems(GenMove, gen_move_ai_cart);
    a.add_systems(
        UpdateAI,
        (
            put_after_cart,
            update_consts,
            update_ai_cart.after(put_after_cart),
        ),
    );

    a.run();
}

fn main() {
    // danmaku()
    cart()
}
