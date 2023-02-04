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
    let n = if (input.m as f64 / input.n as f64) < 2.5 {
        8
    } else {
        8
    };
    let mut annealing_state = AnnealingState::new(&graph, &input, &state, n);
    let mut score_progress_file = File::create("out/optimize_state_score_progress.csv").unwrap();

    const LOOP_INTERVAL: usize = 1000;
    // TODO: 温度調整
    // input.nの大きさに従って決めた方が良さそう
    let start_temp: f64 = 100000.;
    let end_temp: f64 = 100.;
    let mut iter_count = 0;
    let mut progress;
    let mut temp = 0.;
    let start_time = time::elapsed_seconds();

    loop {
        if iter_count % LOOP_INTERVAL == 0 {
            progress = (time::elapsed_seconds() - start_time) / (time_limit - start_time);
            temp = start_temp.powf(1. - progress) * end_temp.powf(progress);

            if progress >= 1. {
                break;
            }
        }
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
        state.update_when(change.edge_index, change.next);
        let (score_diff, reconnections) = annealing_state.estimate(&change, state, &graph);

        let is_valid = *state.repair_counts.iter().max().unwrap() <= input.k;
        let new_score = state.score + score_diff;
        let adopt = (-(new_score - state.score) / temp).exp() > rnd::nextf();

        if adopt && is_valid {
            // 採用
            annealing_state.apply(&reconnections, &graph);
        } else {
            state.update_when(change.edge_index, change.prev);
        }

        if iter_count % LOOP_INTERVAL == 0 {
            if false {
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
    fn new(graph: &Graph, input: &Input, state: &State, n: usize) -> AnnealingState {
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

    fn estimate(
        &self,
        change: &Change,
        state: &State,
        graph: &Graph,
    ) -> (f64, Vec<(usize, usize, Reconnection)>) {
        let mut score_diff = 0.;
        let mut reconnections = vec![];
        for (i, a) in self.agents[change.prev].iter().enumerate() {
            if let Some(reconnection) = a.estimate_add_edge(change.edge_index, &graph, &state.when)
            {
                score_diff += reconnection.score_diff as f64;
                reconnections.push((change.prev, i, reconnection));
            }
        }
        for (i, a) in self.agents[change.next].iter().enumerate() {
            if let Some(reconnection) =
                a.estimate_remove_edge(change.edge_index, &graph, &state.when)
            {
                score_diff += reconnection.score_diff as f64;
                reconnections.push((change.next, i, reconnection));
            }
        }
        (score_diff, reconnections)
    }

    fn apply(&mut self, reconnections: &Vec<(usize, usize, Reconnection)>, graph: &Graph) {
        for (day, agent_index, reconnection) in reconnections {
            self.agents[*day][*agent_index].apply_reconnection(reconnection, graph);
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
struct Reconnection {
    score_diff: i64,
    add_edge: usize,
    remove_edge: usize,
    edge_path: Vec<usize>,
}

#[derive(Debug)]
struct Agent {
    start: usize,
    day: usize,
    dist: VecSum,
    par_edge: Vec<usize>,
    sz: Vec<usize>,
    c: Vec<usize>,
}

impl Agent {
    fn new(start: usize, graph: &Graph, when: &Vec<usize>, day: usize) -> Agent {
        let mut dist = vec![INF; graph.n];
        let mut par_edge = vec![NA; graph.n];
        dist[start] = 0;
        let mut dist = VecSum::new(dist);
        graph.dijkstra(start, when, day, &mut dist, &mut par_edge);
        let sz = calc_subtree_size(start, graph, &par_edge);
        Agent {
            start,
            day,
            dist,
            par_edge,
            sz,
            c: vec![0, 0],
        }
    }

    fn estimate_remove_edge(
        &self,
        edge_index: usize,
        graph: &Graph,
        when: &Vec<usize>,
    ) -> Option<Reconnection> {
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
            return None;
        };

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
            best_reconnection: &mut Reconnection,
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
                    let mut score_diff = 0;
                    let mut last_size = 0;
                    let mut edge_path = vec![];
                    let mut cur = v;
                    let mut cur_dist = agent.dist.vec[e.to] + e.weight;

                    while cur != root {
                        score_diff +=
                            (agent.sz[cur] - last_size) as i64 * (cur_dist - agent.dist.vec[cur]);
                        last_size = agent.sz[cur];
                        let par_edge = &graph.edges[agent.par_edge[cur]];
                        cur_dist += par_edge.weight;
                        edge_path.push(par_edge.index);
                        cur = par_edge.other_vertex(cur);
                    }

                    edge_path.reverse();

                    if score_diff < best_reconnection.score_diff {
                        best_reconnection.score_diff = score_diff;
                        best_reconnection.add_edge = e.index;
                        best_reconnection.remove_edge = agent.par_edge[cur];
                        best_reconnection.edge_path = edge_path;
                    }
                } else {
                    // 子孫の頂点に探索を広げる
                    // 深さ3以上は探索しない
                    if edge_path.len() == 3 {
                        continue;
                    }
                    edge_path.push(e.index);
                    dfs(e.to, root, edge_path, when, graph, agent, best_reconnection);
                    edge_path.pop();
                }
            }
        }

        // rootの距離の増分、rootからnew_rootまでに通る辺、new_rootが新しく繋ぐ辺
        let mut best_reconnection = Reconnection {
            score_diff: INF,
            add_edge: 0,
            remove_edge: 0,
            edge_path: vec![],
        };
        dfs(
            root,
            root,
            &mut vec![],
            when,
            graph,
            &self,
            &mut best_reconnection,
        );

        Some(best_reconnection)
    }

    fn estimate_add_edge(
        &self,
        edge_index: usize,
        graph: &Graph,
        when: &Vec<usize>,
    ) -> Option<Reconnection> {
        let edge = &graph.edges[edge_index];
        let root = if self.dist.vec[edge.u] + edge.weight < self.dist.vec[edge.v] {
            // edge.vに繋がっている頂点をdfsして更新し続ける
            edge.v
        } else if self.dist.vec[edge.v] + edge.weight < self.dist.vec[edge.u] {
            // edge.uに繋がっている頂点をdfsして更新し続ける
            edge.u
        } else {
            // edgeの追加による最短路の更新がないので何もしない
            return None;
        };

        let dist_diff = self.dist.vec[edge.other_vertex(root)] + edge.weight - self.dist.vec[root];
        let mut score_diff = self.sz[root] as i64 * dist_diff;
        let mut edge_path = vec![];
        let mut cur = root;
        let mut cur_dist = self.dist.vec[cur];

        while cur != self.start {
            let par_edge = &graph.edges[self.par_edge[cur]];
            let weight = if when[par_edge.index] == self.day {
                PENALTY
            } else {
                par_edge.weight
            };
            let par = par_edge.other_vertex(cur);
            // 更新されなくなったら終了
            cur_dist += weight;
            let par_dist_diff = cur_dist - self.dist.vec[par];
            if par_dist_diff >= 0 {
                break;
            }
            score_diff += (self.sz[par] - self.sz[cur]) as i64 * par_dist_diff;
            edge_path.push(par_edge.index);
            cur = par;
        }

        let reconnection = Reconnection {
            score_diff,
            add_edge: edge_index,
            remove_edge: self.par_edge[cur],
            edge_path,
        };

        Some(reconnection)
    }

    fn apply_reconnection(&mut self, reconnection: &Reconnection, graph: &Graph) {
        // 削除する辺の親とその祖先のサイズを更新する
        let add_edge = &graph.edges[reconnection.add_edge];
        let remove_edge = &graph.edges[reconnection.remove_edge];
        let mut cur = if self.par_edge[remove_edge.u] == remove_edge.index {
            remove_edge.v
        } else {
            remove_edge.u
        };
        let old_root = remove_edge.other_vertex(cur);
        assert_eq!(self.par_edge[old_root], remove_edge.index);
        let subtree_size = self.sz[old_root];
        // dbg!(
        //     remove_edge,
        //     add_edge,
        //     cur,
        //     old_root,
        //     self.sz[cur],
        //     self.sz[old_root],
        //     self.par_edge[old_root]
        // );
        while cur != self.start {
            self.sz[cur] -= subtree_size;
            // dbg!(cur, par_vertex(cur, graph, &self.par_edge));
            cur = par_vertex(cur, graph, &self.par_edge);
        }
        self.sz[self.start] -= subtree_size;

        // 親の更新
        let mut cur = old_root;
        let mut add_size = 0;
        for e in &reconnection.edge_path {
            self.par_edge[cur] = *e;
            let par = graph.edges[*e].other_vertex(cur);
            // サイズの更新
            self.sz[cur] += add_size;
            self.sz[cur] -= self.sz[par];
            add_size = self.sz[cur];
            cur = par;
        }

        // 追加する辺の親を更新する
        let new_root = cur;
        self.sz[new_root] = subtree_size;
        self.par_edge[cur] = reconnection.add_edge;

        // 追加した辺の親とその祖先のサイズを更新する
        let mut cur = add_edge.other_vertex(cur);
        while cur != self.start {
            self.sz[cur] += subtree_size;
            cur = par_vertex(cur, graph, &self.par_edge);
        }
        self.sz[self.start] += subtree_size;

        // subtreeの距離の更新
        self.dist.set(
            new_root,
            self.dist.vec[par_vertex(new_root, graph, &self.par_edge)] + add_edge.weight,
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
        for i in 0..graph.n {
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

fn calc_subtree_size(root: usize, graph: &Graph, par_edge: &Vec<usize>) -> Vec<usize> {
    let mut sz = vec![0; graph.n];

    fn dfs(v: usize, sz: &mut Vec<usize>, graph: &Graph, par_edge: &Vec<usize>) {
        sz[v] += 1;
        assert!(sz[v] == 1);
        for e in &graph.adj[v] {
            // 子供にだけ動く
            if par_edge[e.to] != e.index {
                continue;
            }
            dfs(e.to, sz, graph, par_edge);
            sz[v] += sz[e.to];
        }
    }

    dfs(root, &mut sz, graph, par_edge);

    sz
}

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

#[test]
fn test_reconnection() {
    let n = 8;
    let s = 0;
    let graph = Graph::new(
        n,
        vec![
            (0, 1, 1),
            (0, 2, 1),
            (1, 3, 1),
            (1, 4, 1),
            (2, 5, 1),
            (4, 6, 1),
            (4, 7, 1),
            (5, 7, 1),
        ],
        vec![
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
    );
    let mut when = vec![0; 8];
    when[7] = 1;
    let mut agents = vec![
        Agent::new(s, &graph, &when, 0),
        Agent::new(s, &graph, &when, 1),
    ];
    let reconnection = Reconnection {
        score_diff: 0,
        add_edge: 7,
        remove_edge: 0,
        edge_path: vec![3, 6],
    };
    dbg!(&agents[1]);
    agents[1].apply_reconnection(&reconnection, &graph);
    dbg!(&agents[1]);
}
