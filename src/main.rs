mod def;
mod graph;
mod util;

use std::fs::File;
use std::io::Write;

use proconio::{input, marker::Usize1};
use util::rnd;

use crate::graph::*;
use crate::util::*;

#[derive(Debug)]
struct Input {
    n: usize,
    m: usize,
    d: usize,
    k: usize,
}

#[derive(Debug)]
struct State {
    when: Vec<usize>,
    score: i64,
}

impl State {
    fn output(&self) -> String {
        let mut ret = String::new();
        for e in &self.when {
            ret += &format!("{} ", e + 1);
        }
        ret += "\n";
        ret
    }
}

fn read_input() -> (Input, Graph) {
    // 入力の読み込み
    input! {
        n: usize,
        m: usize,
        d: usize,
        k: usize,
        edges: [(Usize1, Usize1, i64); m],
        _points: [(i64, i64); n],
    };

    (Input { n, m, d, k }, Graph::new(n, &edges))
}

fn calc_actual_score_slow(input: &Input, graph: &Graph, state: &State) -> i64 {
    let mut fk_sum = 0.;
    let mut base_dist_sum = 0;
    for v in 0..input.n {
        let dist: i64 = graph.dijkstra(v, &state.when, input.d).iter().sum();
        base_dist_sum += dist;
    }
    for day in 0..input.d {
        let mut dist_sum = 0;
        for v in 0..input.n {
            let dist: i64 = graph.dijkstra(v, &state.when, day).iter().sum();
            dist_sum += dist;
        }
        let fk = (dist_sum - base_dist_sum) as f64 / (input.n * (input.n - 1)) as f64;
        fk_sum += fk;
    }
    (1e3 * (fk_sum / input.d as f64)).round() as i64
}

fn main() {
    time::start_clock();
    const TIME_LIMIT: f64 = 300.8;

    let (input, graph) = read_input();

    // 初期解の生成
    let mut state = State {
        when: vec![0; input.m],
        score: 0,
    };
    for i in 0..input.m {
        state.when[i] = rnd::gen_range(0, input.d);
    }

    // 焼きなまし
    let mut ps = vec![];
    for _ in 0..10 {
        ps.push(rnd::gen_range(0, input.n));
    }

    for day in 0..input.d {
        for s in &ps {
            let dist_sum: i64 = graph.dijkstra(*s, &state.when, day).iter().sum();
            state.score += dist_sum;
        }
    }

    let mut score_progress_file = File::create("out/score_progress.csv").unwrap();

    const LOOP_INTERVAL: usize = 100;
    let start_temp: f64 = 100000.;
    let end_temp: f64 = 1000.;
    let mut iter_count = 0;
    let mut progress;
    let mut temp = 0.;

    while time::elapsed_seconds() < TIME_LIMIT {
        if iter_count % LOOP_INTERVAL == 0 {
            progress = time::elapsed_seconds() / TIME_LIMIT;
            temp = start_temp.powf(1. - progress) * end_temp.powf(progress);
        }

        let e = rnd::gen_range(0, input.m);
        let to = rnd::gen_range(0, input.d);

        let prev = state.when[e];
        state.when[e] = to;

        let mut new_score = 0;

        for day in 0..input.d {
            for s in &ps {
                let dist = graph.dijkstra(*s, &state.when, day);
                let dist_sum: i64 = dist.iter().sum();
                new_score += dist_sum;
            }
        }

        // let adopt = ((state.score - new_score) as f64 / temp).exp() > rnd::nextf();
        let adopt = new_score < state.score;
        if adopt {
            state.score = new_score;
        } else {
            state.when[e] = prev;
        }

        iter_count += 1;
        if iter_count % LOOP_INTERVAL == 0 {
            let actual_score = calc_actual_score_slow(&input, &graph, &state);
            writeln!(
                score_progress_file,
                "{},{},{:.2}",
                actual_score,
                state.score,
                time::elapsed_seconds()
            )
            .unwrap();
            eprintln!(
                "{}, {}, {:.2}",
                actual_score,
                state.score,
                time::elapsed_seconds()
            );
        }
    }

    let output = state.output();
    println!("{}", output);

    eprintln!("Time elapsed = {}", time::elapsed_seconds());
    eprintln!("Score = {}", calc_actual_score_slow(&input, &graph, &state));
}
