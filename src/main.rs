mod def;
mod graph;
mod solver;
mod util;

use proconio::{input, marker::Usize1};

use crate::def::*;
use crate::graph::*;
use crate::solver::*;
use crate::util::*;

fn read_input() -> (Input, Graph) {
    // 入力の読み込み
    input! {
        n: usize,
        m: usize,
        d: usize,
        k: usize,
        edges: [(Usize1, Usize1, i64); m],
        points: [(i64, i64); n],
    };

    (Input { n, m, d, k }, Graph::new(n, edges, points))
}

fn main() {
    time::start_clock();
    const DEBUG: bool = false;
    const TIME_LIMIT: f64 = 50.8;

    let (input, graph) = read_input();

    // 初期解の生成
    let mut state = create_initial_state(&input, &graph, 1., DEBUG);

    // TODO: 連結にする
    // connect_all_state();

    // 最適化
    optimize_state(&mut state, &input, &graph, TIME_LIMIT, DEBUG);

    let output = state.output();
    println!("{}", output);

    eprintln!("{:?}", state.repair_counts);
    eprintln!("Time elapsed = {}", time::elapsed_seconds());
    eprintln!("Score = {}", calc_actual_score_slow(&input, &graph, &state));
}
