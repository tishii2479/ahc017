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
            // eprintln!("{}, {:.2}", state.score, time::elapsed_seconds());
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

    state.score = 0.;

    let mut agents = vec![];
    for day in 0..input.d {
        let mut a = vec![];
        for s in &ps {
            let agent = Agent::new(*s, &graph, &state.when, day);
            state.score += agent.dist.sum as f64;
            a.push(agent);
        }
        agents.push(a);
    }

    let mut score_progress_file = File::create("out/optimize_state_score_progress.csv").unwrap();

    const LOOP_INTERVAL: usize = 1000;
    let mut iter_count = 0;

    while time::elapsed_seconds() < time_limit {
        let edge_index = rnd::gen_range(0, input.m);
        let edge = &graph.edges[edge_index];

        let prev = state.when[edge_index];

        // TODO: nextの選択の工夫
        // 同じ頂点に繋がっている辺と同じものを高い確率で選ぶと良さそう
        let next = rnd::gen_range(0, input.d);
        if prev == next {
            continue;
        }

        let mut score_diff = 0.;

        for a in &agents[prev] {
            score_diff -= a.dist.sum as f64;
        }
        for a in &agents[next] {
            score_diff -= a.dist.sum as f64;
        }
        state.update_when(edge_index, next);
        for a in &mut agents[prev] {
            a.add_edge(&edge, &graph, &state.when);
            score_diff += a.dist.sum as f64;
        }
        for a in &mut agents[next] {
            a.remove_edge(&edge, &graph, &state.when);
            score_diff += a.dist.sum as f64;
        }

        let is_valid = *state.repair_counts.iter().max().unwrap() <= input.k;
        let new_score = state.score + score_diff;
        let adopt = new_score < state.score && is_valid;

        if adopt {
            eprintln!(
                "[{:.2}] improved score: {} -> {} ({})",
                time::elapsed_seconds(),
                state.score,
                new_score,
                score_diff,
            );
            state.score = new_score;
        } else {
            state.update_when(edge_index, prev);
            for a in &mut agents[prev] {
                a.remove_edge(&edge, &graph, &state.when);
            }
            for a in &mut agents[next] {
                a.add_edge(&edge, &graph, &state.when);
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
            eprintln!("[{:.2}] {}", time::elapsed_seconds(), state.score);
        }
    }

    eprintln!("[optimize_state] iter_count: {}", iter_count);
}

#[derive(Debug)]
struct Agent {
    start: usize,
    day: usize,
    dist: VecSum,
    par_edge: Vec<usize>,
}

impl Agent {
    fn new(start: usize, graph: &Graph, when: &Vec<usize>, day: usize) -> Agent {
        let mut dist = vec![INF; graph.adj.len()];
        let mut par_edge = vec![INF as usize; graph.adj.len()];
        dist[start] = 0;
        let mut dist = VecSum::new(dist);
        graph.dijkstra(start, when, day, &mut dist, &mut par_edge);
        Agent {
            start,
            day,
            dist,
            par_edge,
        }
    }

    fn remove_edge(&mut self, edge: &EdgeData, graph: &Graph, when: &Vec<usize>) {
        // eprintln!("remove_edge: {:?} {:?}", edge, self.par_edge);
        let _ = if self.par_edge[edge.v] != INF as usize
            && graph.edges[self.par_edge[edge.v]].has_vertex(edge.u)
        {
            // u -> v の最短炉が壊れた
            edge.u
        } else if self.par_edge[edge.u] != INF as usize
            && graph.edges[self.par_edge[edge.u]].has_vertex(edge.v)
        {
            // v -> u の最短炉が壊れた
            edge.v
        } else {
            // 最短路に含まれていないので何もしない
            return;
        };
        // eprintln!("root: {}", root);
        // 子孫のdistを全てINFに戻す
        // let mut st = vec![root];
        // while st.len() > 0 {
        //     let v = st.pop().unwrap();
        //     for e in &graph.adj[v] {
        //         if self.par_edge[e.to] != INF as usize
        //             && graph.edges[self.par_edge[e.to]].has_vertex(v)
        //         {
        //             self.dist.set(e.to, INF);
        //             self.par_edge[e.to] = INF as usize;
        //             st.push(e.to);
        //         }
        //     }
        // }
        // eprintln!("{:?}", self);

        // TODO: 工夫できそう
        // 一時的にペナルティを子孫に加える、一定間隔で再計算する、など
        // 再計算する
        for i in 0..graph.adj.len() {
            self.dist.set(i, INF);
            self.par_edge[i] = INF as usize;
        }
        self.dist.set(self.start, 0);
        graph.dijkstra(
            self.start,
            when,
            self.day,
            &mut self.dist,
            &mut self.par_edge,
        );
    }

    fn add_edge(&mut self, edge: &EdgeData, graph: &Graph, when: &Vec<usize>) {
        let root = if self.dist.vec[edge.u] + edge.weight < self.dist.vec[edge.v] {
            self.par_edge[edge.v] = edge.index;
            self.dist.set(edge.v, self.dist.vec[edge.u] + edge.weight);
            // edge.vに繋がっている頂点をdfsして更新し続ける
            edge.v
        } else if self.dist.vec[edge.v] + edge.weight < self.dist.vec[edge.u] {
            self.par_edge[edge.u] = edge.index;
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
                let weight = if when[e.index] == self.day {
                    PENALTY
                } else {
                    e.weight
                };
                if self.dist.vec[v] + weight >= self.dist.vec[e.to] {
                    continue;
                }
                self.dist.set(e.to, self.dist.vec[v] + weight);
                self.par_edge[e.to] = e.index;
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

#[test]
fn test_agent() {
    fn calc_expected(graph: &Graph, when: &Vec<usize>, n: usize, s: usize, day: usize) -> f64 {
        let mut ret = 0.;
        let mut dist = vec![INF; n];
        dist[s] = 0;
        let mut dist = VecSum::new(dist);
        graph.dijkstra(s, when, day, &mut dist, &mut vec![INF as usize; n]);
        ret += dist.sum as f64;
        eprintln!("{:?}", dist);
        ret
    }
    let n = 5;
    let s = 0;
    let graph = Graph::new(
        n,
        vec![
            (0, 1, 2),
            (0, 2, 3),
            (2, 3, 4),
            (2, 4, 2),
            (0, 4, 7),
            (1, 2, 5),
        ],
        vec![(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    );
    let mut when = vec![0; 6];
    let mut agents = vec![
        Agent::new(s, &graph, &when, 0),
        Agent::new(s, &graph, &when, 1),
    ];

    for _ in 0..100 {
        let e = rnd::gen_range(0, 6);
        when[e] = 1 - when[e];
        eprintln!("{}, {}", when[e], e);
        agents[when[e]].remove_edge(&graph.edges[e], &graph, &when);
        agents[1 - when[e]].add_edge(&graph.edges[e], &graph, &when);
        eprintln!("{:?}", agents);
        assert_eq!(
            calc_expected(&graph, &when, n, s, 0),
            agents[0].dist.sum as f64
        );
        assert_eq!(
            calc_expected(&graph, &when, n, s, 1),
            agents[1].dist.sum as f64
        );
    }
}
