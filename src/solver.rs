use crate::{
    def::*,
    graph::Graph,
    util::{min_index, rnd, time},
};

use std::fs::File;
use std::io::Write;

pub fn create_initial_state(input: &Input, graph: &Graph, time_limit: f64) -> State {
    let mut state = State::new(input.d, vec![0; input.m], 0);
    for i in 0..input.m {
        state.update_when(i, rnd::gen_range(0, input.d));
    }
    state
}

pub fn create_initial_state2(input: &Input, graph: &Graph, time_limit: f64) -> State {
    let mut state = State::new(input.d, vec![0; input.m], 0);

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
            state.update_when(s, day);
            if !graph.is_connected(&state.when, day) {
                state.update_when(s, 0);
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
                        state.update_when(e.index, day);
                        if !graph.is_connected(&state.when, day) {
                            state.update_when(e.index, 0);
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

pub fn create_initial_state3(input: &Input, graph: &Graph, time_limit: f64) -> State {
    let mut paths = vec![];
    let mut path_when = vec![];
    for i in 1..input.n {
        for j in 0..i {
            paths.push(graph.get_path(i, j));
            path_when.push(INF as usize);
        }
    }

    let mut ps = vec![];
    for _ in 0..5 {
        ps.push(rnd::gen_range(0, input.n));
    }

    let mut use_path_indices: Vec<usize>;
    let mut state;

    loop {
        // paths.sort_by(|a, b| b.len().partial_cmp(&a.len()).unwrap());
        rnd::shuffle(&mut paths);

        use_path_indices = vec![];

        state = State::new(input.d, vec![INF as usize; input.m], 0);
        for (path_index, path) in paths.iter().enumerate() {
            let mut is_occupied = false;
            for edge_index in path {
                if state.when[*edge_index] != INF as usize {
                    is_occupied = true;
                }
            }
            if is_occupied {
                continue;
            }

            let day = min_index(&state.repair_counts);

            let mut is_encased = false;
            for edge_index in path {
                state.update_when(*edge_index, day);
                if graph.is_encased(&state.when, graph.edges[*edge_index].u)
                    || graph.is_encased(&state.when, graph.edges[*edge_index].v)
                {
                    is_encased = true;
                }
            }
            if is_encased {
                for edge_index in path {
                    state.update_when(*edge_index, INF as usize);
                }
                continue;
            }
            use_path_indices.push(path_index);
            path_when[path_index] = day;
        }

        for (path_index, path) in paths.iter().enumerate() {
            if path.len() != 1 {
                continue;
            }
            let edge_index = path[0];
            if state.when[edge_index] == INF as usize {
                let mut day = rnd::gen_range(0, input.d);
                state.update_when(edge_index, day);
                while graph.is_encased(&state.when, graph.edges[edge_index].u)
                    || graph.is_encased(&state.when, graph.edges[edge_index].v)
                {
                    day = rnd::gen_range(0, input.d);
                    state.update_when(edge_index, day);
                }
                use_path_indices.push(path_index);
                path_when[path_index] = day;
            }
        }

        let mut is_connected = true;
        for day in 0..input.d {
            if !graph.is_connected(&state.when, day) {
                is_connected = false;
            }
        }
        state.score = 0;
        for day in 0..input.d {
            for s in &ps {
                state.score += graph.calc_dist_sum(*s, &state.when, day);
            }
        }
        eprintln!(
            "{}, {}, {}",
            use_path_indices.len(),
            is_connected,
            time::elapsed_seconds()
        );
        if is_connected || time::elapsed_seconds() >= 1. {
            break;
        }
    }

    state.score = 0;
    for day in 0..input.d {
        for s in &ps {
            state.score += graph.calc_dist_sum(*s, &state.when, day);
        }
    }

    let mut score_progress_file = File::create("out/score_progress.csv").unwrap();

    const LOOP_INTERVAL: usize = 100;
    let mut iter_count = 0;

    while time::elapsed_seconds() < time_limit {
        let path_index = use_path_indices[rnd::gen_range(0, use_path_indices.len())];
        let next = rnd::gen_range(0, input.d);
        let path = &paths[path_index];
        let prev = path_when[path_index];

        for edge_index in path {
            state.update_when(*edge_index, next);
        }

        let mut new_score = 0;
        // TODO: 差分計算
        for day in 0..input.d {
            for s in &ps {
                new_score += graph.calc_dist_sum(*s, &state.when, day);
            }
        }

        // let adopt = ((state.score - new_score) as f64 / temp).exp() > rnd::nextf();
        let adopt =
            new_score < state.score && *state.repair_counts.iter().max().unwrap() <= input.k;
        if adopt {
            // eprintln!("adopt: {}, {}", new_score, state.score);
            state.score = new_score;
            path_when[path_index] = next;
        } else {
            for edge_index in path {
                state.update_when(*edge_index, prev);
            }
        }

        iter_count += 1;
        if iter_count % LOOP_INTERVAL == 0 {
            let actual_score = calc_actual_score_slow(&input, &graph, &state);
            // let actual_score = -1;
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

    state
}

pub fn optimize_state(state: &mut State, input: &Input, graph: &Graph, time_limit: f64) {
    let mut ps = vec![];
    for _ in 0..5 {
        ps.push(rnd::gen_range(0, input.n));
    }

    for day in 0..input.d {
        for s in &ps {
            state.score += graph.calc_dist_sum(*s, &state.when, day);
        }
    }

    let mut score_progress_file = File::create("out/score_progress.csv").unwrap();

    const LOOP_INTERVAL: usize = 100;
    let mut iter_count = 0;

    while time::elapsed_seconds() < time_limit {
        let edge_index = rnd::gen_range(0, input.m);
        let next = rnd::gen_range(0, input.d);

        let prev = state.when[edge_index];
        state.update_when(edge_index, next);

        let mut new_score = 0;
        // TODO: 差分計算
        for day in 0..input.d {
            for s in &ps {
                new_score += graph.calc_dist_sum(*s, &state.when, day);
            }
        }

        // let adopt = ((state.score - new_score) as f64 / temp).exp() > rnd::nextf();
        let adopt = new_score < state.score;
        if adopt {
            state.score = new_score;
        } else {
            state.update_when(edge_index, prev);
        }

        iter_count += 1;
        if iter_count % LOOP_INTERVAL == 0 {
            // let actual_score = calc_actual_score_slow(&input, &graph, &state);
            let actual_score = -1;
            writeln!(
                score_progress_file,
                "{}, {}, {:.2}",
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
