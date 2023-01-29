use crate::{
    def::*,
    graph::Graph,
    util::{rnd, time},
};

use std::fs::File;
use std::io::Write;

#[allow(unused_variables, unused)]

pub fn create_random_initial_state(
    input: &Input,
    graph: &Graph,
    time_limit: f64,
    debug: bool,
) -> State {
    let mut state = State::new(input.d, vec![INF as usize; input.m], 0.);
    for i in 0..input.m {
        let mut day = rnd::gen_range(0, input.d);
        while state.repair_counts[day] >= input.k {
            day = rnd::gen_range(0, input.d);
        }
        state.update_when(i, day);
    }
    state
}

pub fn create_initial_state(input: &Input, graph: &Graph, time_limit: f64, debug: bool) -> State {
    fn calc_vertex_score(v: usize, graph: &Graph, state: &State) -> f64 {
        let mut score = 0.;
        for e1 in &graph.adj[v] {
            for e2 in &graph.adj[v] {
                if e1.index == e2.index {
                    continue;
                }
                if state.when[e1.index] != state.when[e2.index] {
                    continue;
                }
                let sim = calc_cosine_similarity(
                    &graph.pos[e1.to],
                    &graph.pos[v],
                    &graph.pos[e2.to],
                    &graph.pos[v],
                );
                // let e_score = sim;
                let mut e_score = sim + 0.7;
                if e_score < 0. {
                    e_score *= 3.;
                }
                score += e_score;
            }
        }
        score
    }

    let mut state = State::new(input.d, vec![INF as usize; input.m], 0.);
    for i in 0..input.m {
        let mut day = rnd::gen_range(0, input.d);
        while state.repair_counts[day] >= input.k {
            day = rnd::gen_range(0, input.d);
        }
        state.update_when(i, day);
    }

    for v in 0..input.n {
        state.score += calc_vertex_score(v, &graph, &state);
    }

    let mut score_progress_file =
        File::create("out/create_initial_state_score_progress.csv").unwrap();

    const LOOP_INTERVAL: usize = 100;
    let start_temp: f64 = 100.;
    let end_temp: f64 = 0.1;
    let mut iter_count = 0;
    let mut progress;
    let mut temp;
    let start_time = time::elapsed_seconds();

    while time::elapsed_seconds() < time_limit {
        progress = (time::elapsed_seconds() - start_time) / (time_limit - start_time);
        temp = start_temp.powf(1. - progress) * end_temp.powf(progress);

        let edge_index = rnd::gen_range(0, input.m);
        // TODO: 同じ頂点に繋がっている辺と同じものを高い確率で選ぶと良さそう
        let prev = state.when[edge_index];
        let next = rnd::gen_range(0, input.d);

        let mut new_score = state.score;

        new_score -= calc_vertex_score(graph.edges[edge_index].v, &graph, &state);
        new_score -= calc_vertex_score(graph.edges[edge_index].u, &graph, &state);

        state.update_when(edge_index, next);

        new_score += calc_vertex_score(graph.edges[edge_index].v, &graph, &state);
        new_score += calc_vertex_score(graph.edges[edge_index].u, &graph, &state);

        let is_valid = *state.repair_counts.iter().max().unwrap() <= input.k;
        let adopt = (-(new_score - state.score) / temp).exp() > rnd::nextf();
        if adopt && is_valid {
            state.score = new_score;
        } else {
            state.update_when(edge_index, prev);
        }

        iter_count += 1;
        if iter_count % LOOP_INTERVAL == 0 {
            if debug {
                writeln!(
                    score_progress_file,
                    "{},{:.2},{}",
                    state.score,
                    time::elapsed_seconds(),
                    calc_actual_score_slow(&input, &graph, &state),
                )
                .unwrap();
            }
            eprintln!("{}, {:.2}", state.score, time::elapsed_seconds());
        }
    }
    eprintln!("[create_initial_state] iter_count: {}", iter_count);

    state
}

pub fn optimize_state(
    state: &mut State,
    input: &Input,
    graph: &Graph,
    time_limit: f64,
    debug: bool,
) {
    let mut ps = vec![];
    for _ in 0..5 {
        ps.push(rnd::gen_range(0, input.n));
    }

    state.score = 0.;
    for day in 0..input.d {
        for s in &ps {
            state.score += graph.calc_dist_sum(*s, &state.when, day) as f64;
        }
    }

    let mut score_progress_file = File::create("out/optimize_state_score_progress.csv").unwrap();

    const LOOP_INTERVAL: usize = 100;
    let mut iter_count = 0;

    while time::elapsed_seconds() < time_limit {
        let edge_index = rnd::gen_range(0, input.m);
        // TODO: 同じ頂点に繋がっている辺と同じものを高い確率で選ぶと良さそう
        let prev = state.when[edge_index];
        let next = rnd::gen_range(0, input.d);

        let mut new_score = state.score;

        for s in &ps {
            new_score -= graph.calc_dist_sum(*s, &state.when, prev) as f64;
            new_score -= graph.calc_dist_sum(*s, &state.when, next) as f64;
        }
        state.update_when(edge_index, next);
        for s in &ps {
            new_score += graph.calc_dist_sum(*s, &state.when, prev) as f64;
            new_score += graph.calc_dist_sum(*s, &state.when, next) as f64;
        }

        // let adopt = ((state.score - new_score) as f64 / temp).exp() > rnd::nextf();
        let is_valid = *state.repair_counts.iter().max().unwrap() <= input.k;
        let adopt = new_score < state.score && is_valid;
        if adopt {
            state.score = new_score;
        } else {
            state.update_when(edge_index, prev);
        }

        iter_count += 1;
        if iter_count % LOOP_INTERVAL == 0 {
            if debug {
                writeln!(
                    score_progress_file,
                    "{},{:.2},{}",
                    state.score,
                    time::elapsed_seconds(),
                    calc_actual_score_slow(&input, &graph, &state),
                )
                .unwrap();
            }
            eprintln!("{} {:.2}", state.score, time::elapsed_seconds());
        }
    }

    eprintln!("[optimize_state] iter_count: {}", iter_count);
}

pub fn calc_actual_score_slow(input: &Input, graph: &Graph, state: &State) -> i64 {
    let mut fk_sum = 0.;
    let mut base_dist_sum = 0;
    for v in 0..input.n {
        base_dist_sum += graph.dist_sum[v];
    }
    for day in 0..input.d {
        let mut dist_sum = 0;
        for v in 0..input.n {
            dist_sum += graph.calc_dist_sum(v, &state.when, day);
        }
        let fk = (dist_sum - base_dist_sum) as f64 / (input.n * (input.n - 1)) as f64;
        fk_sum += fk;
    }
    (1e3 * (fk_sum / input.d as f64)).round() as i64
}

fn calc_cosine_similarity(to_pos: &Pos, from_pos: &Pos, to_pos2: &Pos, from_pos2: &Pos) -> f64 {
    let dy1 = to_pos.y - from_pos.y;
    let dx1 = to_pos.x - from_pos.x;

    let dy2 = to_pos2.y - from_pos2.y;
    let dx2 = to_pos2.x - from_pos2.x;

    let div = ((dy1 * dy1 + dx1 * dx1) as f64).sqrt() * ((dy2 * dy2 + dx2 * dx2) as f64).sqrt();
    let prod = (dy1 * dy2 + dx1 * dx2) as f64;

    prod / div
}
