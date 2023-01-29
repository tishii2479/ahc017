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
    const TIME_LIMIT: f64 = 3.8;

    let (input, graph) = read_input();

    // 初期解の生成
    let state = if input.d <= 10 {
        let mut state = create_initial_state(&input, &graph, TIME_LIMIT);
        optimize_state(&mut state, &input, &graph, TIME_LIMIT);
        state
    } else {
        let mut state = create_initial_state3(&input, &graph, TIME_LIMIT);
        optimize_state(&mut state, &input, &graph, TIME_LIMIT);
        state
    };

    let output = state.output();
    println!("{}", output);

    eprintln!("Time elapsed = {}", time::elapsed_seconds());
    eprintln!("Score = {}", calc_actual_score_slow(&input, &graph, &state));
}
