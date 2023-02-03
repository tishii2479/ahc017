use crate::{
    def::*,
    graph::Graph,
    util::{rnd, time, VecSum},
};

use std::fs::File;
use std::io::Write;

pub fn create_random_initial_state(input: &Input) -> State {
    let mut state = State::new(input.d, vec![NA; input.m], 0.);
    for i in 0..input.m {
        let mut day = rnd::gen_range(0, input.d);
        while state.repair_counts[day] >= input.k {
            day = rnd::gen_range(0, input.d);
        }
        state.update_when(i, day);
    }
    state
}

pub fn optimize_state(state: &mut State, input: &Input, graph: &Graph, time_limit: f64) {
    let mut annealing_state = AnnealingState::new(&graph, &input, &state);
    let mut score_progress_file = File::create("out/optimize_state_score_progress.csv").unwrap();

    const LOOP_INTERVAL: usize = 1000;
    // TODO: 温度調整
    // input.nの大きさに従って決めた方が良さそう
    let start_temp: f64 = 100000.;
    let end_temp: f64 = 100.;
    let mut iter_count = 0;
    let mut progress;
    let mut temp;
    let start_time = time::elapsed_seconds();

    while time::elapsed_seconds() < time_limit {
        // TODO: 定期的にだけ更新する
        progress = (time::elapsed_seconds() - start_time) / (time_limit - start_time);
        temp = start_temp.powf(1. - progress) * end_temp.powf(progress);

        iter_count += 1;

        let edge_index = rnd::gen_range(0, input.m);
        let prev = state.when[edge_index];
        let next = select_next(edge_index, &graph, &state.when, input.d);
        if prev == next {
            continue;
        }

        let change = Change {
            prev,
            next,
            edge_index,
        };
        let score_diff = annealing_state.apply(&change, state, &graph);

        let is_valid = *state.repair_counts.iter().max().unwrap() <= input.k;
        let new_score = state.score + score_diff;
        let adopt = (-(new_score - state.score) / temp).exp() > rnd::nextf();

        if adopt && is_valid {
            // 採用
        } else {
            annealing_state.reverse(&change, state, &graph);
        }

        if iter_count % LOOP_INTERVAL == 0 {
            if true {
                // let actual_score = calc_actual_score_slow(&input, &graph, &state);
                let actual_score = -1;
                writeln!(
                    score_progress_file,
                    "{},{:.2},{}",
                    annealing_state.calc_score(),
                    time::elapsed_seconds(),
                    actual_score,
                )
                .unwrap();
                eprintln!(
                    "[{:.2}] {} {}",
                    time::elapsed_seconds(),
                    annealing_state.calc_score(),
                    actual_score
                );
            } else {
                eprintln!(
                    "[{:.2}] {}",
                    time::elapsed_seconds(),
                    annealing_state.calc_score()
                );
            }
        }
    }

    eprintln!("{:?}", annealing_state.agents[0][0].c);

    eprintln!("[optimize_state] iter_count: {}", iter_count);
}

struct Change {
    prev: usize,
    next: usize,
    edge_index: usize,
}

struct AnnealingState {
    agents: Vec<Vec<Agent>>,
}

impl AnnealingState {
    fn new(graph: &Graph, input: &Input, state: &State) -> AnnealingState {
        // TODO: 定期的に違う点を取り直す?
        let n = if (input.m as f64 / input.n as f64) < 2.5 {
            8
        } else {
            8
        };
        let mut ps = vec![graph.find_closest_point(&Pos { x: 500, y: 500 })];
        for i in 0..n {
            let d = i as f64 / n as f64 * 2. * std::f64::consts::PI;
            let p = Pos {
                x: (f64::cos(d) * 1000. + 500.).round() as i64,
                y: (f64::sin(d) * 1000. + 500.).round() as i64,
            };
            eprintln!("{:?}", p);
            ps.push(graph.find_closest_point(&p));
        }

        let mut agents = vec![];
        for day in 0..input.d {
            let mut a = vec![];
            for s in &ps {
                let agent = Agent::new(*s, &graph, &state.when, day);
                a.push(agent);
            }
            agents.push(a);
        }

        AnnealingState { agents }
    }

    fn apply(&mut self, change: &Change, state: &mut State, graph: &Graph) -> f64 {
        let mut score_diff = 0.;

        for a in &self.agents[change.prev] {
            score_diff -= a.dist.sum as f64;
        }
        for a in &self.agents[change.next] {
            score_diff -= a.dist.sum as f64;
        }
        state.update_when(change.edge_index, change.next);
        for a in &mut self.agents[change.prev] {
            a.add_edge(change.edge_index, &graph, &state.when);
            score_diff += a.dist.sum as f64;
        }
        for a in &mut self.agents[change.next] {
            a.remove_edge(change.edge_index, &graph, &state.when);
            score_diff += a.dist.sum as f64;
        }
        score_diff
    }

    fn reverse(&mut self, change: &Change, state: &mut State, graph: &Graph) {
        state.update_when(change.edge_index, change.prev);
        for a in &mut self.agents[change.prev] {
            a.remove_edge(change.edge_index, &graph, &state.when);
        }
        for a in &mut self.agents[change.next] {
            a.add_edge(change.edge_index, &graph, &state.when);
        }
    }

    fn calc_score(&self) -> f64 {
        let mut sum = 0;
        for a in &self.agents {
            for e in a {
                sum += e.dist.sum;
            }
        }
        sum as f64
    }
}

#[derive(Debug)]
struct Agent {
    start: usize,
    day: usize,
    dist: VecSum,
    par_edge: Vec<usize>,
    c: Vec<usize>,
}

impl Agent {
    fn new(start: usize, graph: &Graph, when: &Vec<usize>, day: usize) -> Agent {
        let mut dist = vec![INF; graph.adj.len()];
        let mut par_edge = vec![NA; graph.adj.len()];
        dist[start] = 0;
        let mut dist = VecSum::new(dist);
        graph.dijkstra(start, when, day, &mut dist, &mut par_edge);
        Agent {
            start,
            day,
            dist,
            par_edge,
            c: vec![0, 0],
        }
    }

    fn remove_edge(&mut self, edge_index: usize, graph: &Graph, when: &Vec<usize>) {
        let edge = &graph.edges[edge_index];
        let root = if self.par_edge[edge.v] != NA
            && graph.edges[self.par_edge[edge.v]].has_vertex(edge.u)
        {
            // u -> v の最短路が壊れた
            edge.v
        } else if self.par_edge[edge.u] != NA
            && graph.edges[self.par_edge[edge.u]].has_vertex(edge.v)
        {
            // v -> u の最短路が壊れた
            edge.u
        } else {
            // 最短路に含まれていないので何もしない
            return;
        };

        fn is_child_vertex(v: usize, par: usize, graph: &Graph, par_edge: &Vec<usize>) -> bool {
            let mut v = v;
            while par_edge[v] != NA && v != par {
                // 親の頂点を取得する
                v = graph.edges[par_edge[v]].u + graph.edges[par_edge[v]].v - v;
            }
            v == par
        }

        fn par_vertex(v: usize, graph: &Graph, par_edge: &Vec<usize>) -> usize {
            graph.edges[par_edge[v]].other_vertex(v)
        }

        // rootから距離が3以下の頂点を探す
        // 各頂点について、rootの子孫ではなく、繋がっていない頂点が隣接していて、工事されていない辺があれば、
        // それに繋げることができる
        // rootの距離の増分が最小の頂点を探す
        fn dfs(
            v: usize,
            root: usize,
            edge_path: &mut Vec<usize>,
            when: &Vec<usize>,
            graph: &Graph,
            agent: &Agent,
            best_path: &mut (i64, Vec<usize>, usize),
        ) {
            for e in &graph.adj[v] {
                // 工事されている辺は通らない
                if when[e.index] == agent.day {
                    continue;
                }
                // 親には遡らない
                if par_vertex(v, graph, &agent.par_edge) == e.to {
                    continue;
                }
                // rootの子孫ではなく、繋がっていない頂点が隣接していれば繋げることができる
                if !is_child_vertex(e.to, root, &graph, &agent.par_edge) {
                    // 繋げることができる
                    // 増える距離を計算する
                    let mut new_dist = agent.dist.vec[e.to] + e.weight;
                    for p in edge_path.into_iter() {
                        new_dist += graph.edges[*p].weight;
                    }
                    if new_dist < best_path.0 {
                        best_path.0 = new_dist;
                        best_path.1 = edge_path.clone();
                        best_path.2 = e.index;
                    }
                } else {
                    // 子孫の頂点に探索を広げる
                    // 深さ3以上は探索しない
                    if edge_path.len() == 3 {
                        continue;
                    }
                    edge_path.push(e.index);
                    dfs(e.to, root, edge_path, when, graph, agent, best_path);
                    edge_path.pop();
                }
            }
        }

        // rootの距離の増分、rootからnew_rootまでに通る辺、new_rootが新しく繋ぐ辺
        let mut best_path = (INF, vec![], NA);
        dfs(root, root, &mut vec![], when, graph, &self, &mut best_path);

        if best_path.2 != NA {
            let reconnect_path = best_path.1;
            let reconnect_edge = graph.edges[best_path.2];
            let mut cur = root;
            for e in &reconnect_path {
                self.par_edge[cur] = *e;
                let par = graph.edges[*e].other_vertex(cur);
                cur = par;
            }

            let new_root = cur;
            self.par_edge[new_root] = best_path.2;
            // これがあると閉路ができなくなる
            // 正しいけど、理屈がわからない
            self.dist.set(
                new_root,
                self.dist.vec[reconnect_edge.other_vertex(new_root)] + reconnect_edge.weight,
            );

            let mut st = vec![new_root];
            while st.len() > 0 {
                let v = st.pop().unwrap();
                for e in &graph.adj[v] {
                    if self.par_edge[e.to] != NA && graph.edges[self.par_edge[e.to]].has_vertex(v) {
                        self.dist.set(e.to, self.dist.vec[v] + e.weight);
                        st.push(e.to);
                    }
                }
            }
            self.c[0] += 1;
            return;
        }

        // 全て再計算する
        // TODO: たまに強制的に再計算する or 最後の方だけ常に再計算する
        for i in 0..graph.adj.len() {
            self.dist.set(i, INF);
            self.par_edge[i] = NA;
        }
        self.dist.set(self.start, 0);
        graph.dijkstra(
            self.start,
            when,
            self.day,
            &mut self.dist,
            &mut self.par_edge,
        );
        self.c[1] += 1;
    }

    fn add_edge(&mut self, edge_index: usize, graph: &Graph, when: &Vec<usize>) {
        let edge = &graph.edges[edge_index];
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

fn select_next(edge_index: usize, graph: &Graph, when: &Vec<usize>, d: usize) -> usize {
    if rnd::nextf() < 0.5 {
        // 頂点に繋がっている工事を伸ばす
        let edge = &graph.edges[edge_index];
        let s = rnd::gen_range(0, graph.adj[edge.u].len() + graph.adj[edge.v].len());
        let e = if s >= graph.adj[edge.u].len() {
            graph.adj[edge.v][s - graph.adj[edge.u].len()].index
        } else {
            graph.adj[edge.u][s].index
        };
        when[e]
    } else {
        // ランダムに選ぶ
        rnd::gen_range(0, d)
    }
}

#[test]
fn test_agent_random() {
    fn calc_expected(graph: &Graph, when: &Vec<usize>, n: usize, s: usize, day: usize) -> f64 {
        let mut ret = 0.;
        let mut dist = vec![INF; n];
        dist[s] = 0;
        let mut dist = VecSum::new(dist);
        graph.dijkstra(s, when, day, &mut dist, &mut vec![NA; n]);
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
        agents[when[e]].remove_edge(e, &graph, &when);
        agents[1 - when[e]].add_edge(e, &graph, &when);
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

#[test]
fn test_agent() {
    fn calc_expected(graph: &Graph, when: &Vec<usize>, n: usize, s: usize, day: usize) -> f64 {
        let mut ret = 0.;
        let mut dist = vec![INF; n];
        dist[s] = 0;
        let mut dist = VecSum::new(dist);
        graph.dijkstra(s, when, day, &mut dist, &mut vec![NA; n]);
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

    let e = 1;
    when[e] = 1;
    eprintln!("{}, {}", when[e], e);
    agents[when[e]].remove_edge(e, &graph, &when);
    agents[1 - when[e]].add_edge(e, &graph, &when);
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
