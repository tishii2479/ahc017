use crate::{
    def::*,
    graph::Graph,
    util::{rnd, time},
};

use std::fs::File;
use std::io::Write;

pub fn create_initial_state(input: &Input, graph: &Graph, time_limit: f64) -> State {
    let mut state = State {
        when: vec![0; input.m],
        score: 0,
    };
    for i in 0..input.m {
        state.when[i] = rnd::gen_range(0, input.d);
    }
    state
}

pub fn create_initial_state2(input: &Input, graph: &Graph, time_limit: f64) -> State {
    let mut state = State {
        when: vec![0; input.m],
        score: 0,
    };

    fn calc_cosine_similarity(to_pos: &Pos, from_pos: &Pos, to_pos2: &Pos, from_pos2: &Pos) -> f64 {
        let dy1 = to_pos.y - from_pos.y;
        let dx1 = to_pos.x - from_pos.x;

        let dy2 = to_pos2.y - from_pos2.y;
        let dx2 = to_pos2.x - from_pos2.x;

        let div = ((dy1 * dy1 + dx1 * dx1) as f64).sqrt() * ((dy2 * dy2 + dx2 * dx2) as f64).sqrt();
        let prod = (dy1 * dy2 + dx1 * dx2) as f64;

        prod / div
    }

    for day in 1..input.d {
        let max_count = input.m / input.d;
        let mut count = 0;

        while count < max_count && time::elapsed_seconds() < time_limit {
            let mut s = 0;
            while state.when[s] != 0 {
                s = rnd::gen_range(0, input.m);
            }

            let start_edge = graph.edges[s];
            state.when[s] = day;
            if *graph.dijkstra(0, &state.when, day).iter().max().unwrap() >= INF {
                state.when[s] = 0;
                continue;
            }
            count += 1;

            let mut v = start_edge.v;

            while count < max_count {
                let mut is_added = false;
                for e in &graph.adj[v] {
                    if state.when[e.index] != 0 {
                        continue;
                    }
                    let next_v = e.to;
                    let sim = calc_cosine_similarity(
                        &graph.pos[start_edge.v],
                        &graph.pos[start_edge.u],
                        &graph.pos[next_v],
                        &graph.pos[v],
                    );
                    if sim >= 0.6 {
                        state.when[e.index] = day;
                        if *graph.dijkstra(0, &state.when, day).iter().max().unwrap() >= INF {
                            state.when[e.index] = 0;
                            continue;
                        }
                        v = next_v;
                        count += 1;
                        is_added = true;
                        break;
                    }
                }
                if !is_added {
                    break;
                }
            }
        }
    }
    state
}

pub fn optimize_state(state: &mut State, input: &Input, graph: &Graph, time_limit: f64) {
    let mut ps = vec![];
    for _ in 0..5 {
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
    let mut iter_count = 0;

    while time::elapsed_seconds() < time_limit {
        let edge_index = rnd::gen_range(0, input.m);
        let next = rnd::gen_range(0, input.d);

        let prev = state.when[edge_index];
        state.when[edge_index] = next;

        let mut new_score = 0;
        // TODO: 差分計算
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
            state.when[edge_index] = prev;
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
}

pub fn calc_actual_score_slow(input: &Input, graph: &Graph, state: &State) -> i64 {
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
