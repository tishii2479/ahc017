use crate::{
    def::*,
    graph::{EdgeData, Graph},
    util::{rnd, time, VecSum},
};

use std::io::Write;
use std::{fs::File, iter::zip};

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

    let paths = {
        let mut paths = vec![];
        for i in 1..input.n {
            for j in 0..i {
                let p = graph.get_path(i, j);
                if p.0.len() <= 3 {
                    paths.push(p);
                }
            }
        }
        paths
    };

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

    const LOOP_INTERVAL: usize = 10000;
    let start_temp: f64 = 100.;
    let end_temp: f64 = 0.1;
    let mut iter_count = 0;
    let mut progress;
    let mut temp;
    let start_time = time::elapsed_seconds();

    while time::elapsed_seconds() < time_limit {
        progress = (time::elapsed_seconds() - start_time) / (time_limit - start_time);
        temp = start_temp.powf(1. - progress) * end_temp.powf(progress);

        let (path_edges, path_verticies) = &paths[rnd::gen_range(0, paths.len())];
        // eprintln!("{:?}", path);
        // TODO: 同じ頂点に繋がっている辺と同じものを高い確率で選ぶと良さそう
        let prev = {
            let mut ret = vec![];
            for edge_index in path_edges {
                ret.push(state.when[*edge_index]);
            }
            ret
        };
        let next = rnd::gen_range(0, input.d);

        let mut new_score = state.score;
        for v in path_verticies {
            new_score -= calc_vertex_score(*v, &graph, &state);
        }

        for edge_index in path_edges {
            state.update_when(*edge_index, next);
        }
        for v in path_verticies {
            new_score += calc_vertex_score(*v, &graph, &state);
        }

        let is_valid = *state.repair_counts.iter().max().unwrap() <= input.k;
        let adopt = (-(new_score - state.score) / temp).exp() > rnd::nextf();
        if adopt && is_valid {
            state.score = new_score;
        } else {
            for (edge_index, prev) in zip(path_edges, &prev) {
                state.update_when(*edge_index, *prev);
            }
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
    eprintln!("before: {}", calc_actual_score_slow(&input, &graph, &state));
    let ps = vec![
        graph.find_closest_point(&Pos { x: 250, y: 250 }),
        graph.find_closest_point(&Pos { x: 250, y: 750 }),
        // graph.find_closest_point(&Pos { x: 500, y: 500 }),
        graph.find_closest_point(&Pos { x: 750, y: 250 }),
        graph.find_closest_point(&Pos { x: 750, y: 750 }),
    ];
    // eprintln!("{:?}", ps);

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

        // TODO: キャッシュする
        for s in &ps {
            new_score -= graph.calc_dist_sum(*s, &state.when, prev) as f64;
            new_score -= graph.calc_dist_sum(*s, &state.when, next) as f64;
        }
        state.update_when(edge_index, next);
        for s in &ps {
            new_score += graph.calc_dist_sum(*s, &state.when, prev) as f64;
            new_score += graph.calc_dist_sum(*s, &state.when, next) as f64;
        }

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

struct S {
    s: usize,
    dist: VecSum,
    par: Vec<usize>,
}

impl S {
    fn new(s: usize, graph: &Graph, when: &Vec<usize>, day: usize) -> S {
        let mut dist = vec![INF; graph.adj.len()];
        let mut par = vec![INF as usize; graph.adj.len()];
        dist[s] = 0;
        let mut dist = VecSum::new(dist);
        graph.dijkstra(s, when, day, &mut dist, &mut par);
        S { s, dist, par }
    }

    fn remove_edge(&mut self, edge: &EdgeData, graph: &Graph, when: &Vec<usize>, day: usize) {
        let root = if self.par[edge.v] == edge.u {
            // edge.uの子孫のdistを再計算する
            edge.u
        } else if self.par[edge.u] == edge.v {
            // edge.vの子孫のdistを再計算する
            edge.v
        } else {
            // 最短路に含まれていないので何もしない
            return;
        };
        // 子孫のdistを全てINFに戻す
        let mut st = vec![root];
        while st.len() > 0 {
            let v = st.pop().unwrap();
            self.dist.set(v, INF);
            for e in &graph.adj[v] {
                if self.par[e.to] == v {
                    st.push(e.to);
                }
            }
        }
        graph.dijkstra(root, when, day, &mut self.dist, &mut self.par);
    }

    fn add_edge(&mut self, edge: &EdgeData, graph: &Graph, when: &Vec<usize>, day: usize) {
        let root = if self.dist.vec[edge.u] + edge.weight < self.dist.vec[edge.v] {
            self.par[edge.v] = edge.u;
            self.dist.set(edge.v, self.dist.vec[edge.u] + edge.weight);
            // edge.vに繋がっている頂点をdfsして更新し続ける
            edge.v
        } else if self.dist.vec[edge.v] + edge.weight < self.dist.vec[edge.u] {
            self.par[edge.u] = edge.v;
            self.dist.set(edge.u, self.dist.vec[edge.v] + edge.weight);
            // edge.uに繋がっている頂点をdfsして更新し続ける
            edge.u
        } else {
            // edgeの追加による最短路の更新がないので何もしない
            return;
        };
        let mut st = vec![root];
        while st.len() > 0 {
            let v = st.pop().unwrap();
            for e in &graph.adj[v] {
                // その辺が使えない場合
                if when[e.index] == day {
                    continue;
                }
                if self.dist.vec[v] + e.weight >= self.dist.vec[e.to] {
                    continue;
                }
                self.dist.set(e.to, self.dist.vec[v] + e.weight);
                self.par[e.to] = v;
                st.push(e.to);
            }
        }
    }
}

pub fn calc_actual_score_slow(input: &Input, graph: &Graph, state: &State) -> i64 {
    let mut fk_sum = 0.;
    let mut base_dist_sum = 0;
    for v in 0..input.n {
        base_dist_sum += graph.dist[v].sum;
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
