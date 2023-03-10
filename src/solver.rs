use crate::{
    def::*,
    graph::Graph,
    util::{rnd, time, VecSum},
};

pub fn create_random_initial_state(input: &Input) -> State {
    let mut state = State::new(input.d, vec![NA; input.m]);
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
    const AGENT_N: usize = 8;
    const LOOP_INTERVAL: usize = 1000;
    const UPDATE_INTERVAL: f64 = 0.0401;

    let mut annealing_state = AnnealingState::new(&graph, &input, &state, AGENT_N);

    let rank = (input.m as f64 / input.n as f64) * (input.d as f64).powf(0.35);
    let start_temp: f64 = AGENT_N as f64 * 1e6;
    let end_temp: f64 = AGENT_N as f64 * (if rank <= 6. { 1e3 } else { 1e2 });
    let mut iter_count = 0;
    let mut progress;
    let mut temp = 0.;
    let mut last_update = UPDATE_INTERVAL;
    let start_time = time::elapsed_seconds();

    let mut adopted_count = 0;

    loop {
        if iter_count % LOOP_INTERVAL == 0 {
            progress = (time::elapsed_seconds() - start_time) / (time_limit - start_time);
            temp = start_temp.powf(1. - progress) * end_temp.powf(progress);

            if progress >= 1. {
                break;
            }
            if progress > last_update {
                // 定期的に基点を更新する
                last_update += UPDATE_INTERVAL;
                let agent_n = if progress >= 0.8 {
                    AGENT_N * 2
                } else {
                    AGENT_N
                };
                annealing_state = AnnealingState::new(&graph, &input, &state, agent_n);
            }
        }
        iter_count += 1;

        let change = annealing_state.suggest_change(&input, &graph, &state.when);
        if change.prev == change.next {
            continue;
        }
        state.update_when(change.edge_index, change.next);
        let (score_diff, reconnections) = annealing_state.estimate(&change, state, &graph);

        let is_valid = *state.repair_counts.iter().max().unwrap() <= input.k;
        let adopt = (-score_diff as f64 / temp).exp() > rnd::nextf();

        if adopt && is_valid {
            // 採用
            adopted_count += 1;
            annealing_state.apply(&reconnections, &graph);
        } else {
            state.update_when(change.edge_index, change.prev);
        }
    }

    eprintln!("[optimize_state] adopted_count: {}", adopted_count);
    eprintln!("[optimize_state] iter_count:    {}", iter_count);
}

#[derive(Debug)]
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
        let a = rnd::nextf() * 2. * std::f64::consts::PI;
        for i in 0..n {
            let r = rnd::gen_range(333, 500) as f64;
            let d = i as f64 / n as f64 * 2. * std::f64::consts::PI + a;
            let p = Pos {
                x: (f64::cos(d) * r + 500.).round() as i64,
                y: (f64::sin(d) * r + 500.).round() as i64,
            };
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

    fn suggest_change(&self, input: &Input, graph: &Graph, when: &Vec<usize>) -> Change {
        let edge_index = rnd::gen_range(0, input.m);
        let prev = when[edge_index];
        let next = select_next(edge_index, &graph, &when, input.d);
        return Change {
            prev,
            next,
            edge_index,
        };
    }

    fn estimate(
        &self,
        change: &Change,
        state: &State,
        graph: &Graph,
    ) -> (i64, Vec<(usize, usize, Reconnection)>) {
        let mut score_diff = 0;
        let mut reconnections = vec![];
        for (i, a) in self.agents[change.next].iter().enumerate() {
            if score_diff >= INF {
                break;
            }
            if let Some(reconnection) =
                a.estimate_remove_edge(change.edge_index, &graph, &state.when)
            {
                score_diff += reconnection.score_diff;
                reconnections.push((change.next, i, reconnection));
            }
        }
        for (i, a) in self.agents[change.prev].iter().enumerate() {
            if score_diff >= INF {
                break;
            }
            if let Some(reconnection) = a.estimate_add_edge(change.edge_index, &graph, &state.when)
            {
                score_diff += reconnection.score_diff;
                reconnections.push((change.prev, i, reconnection));
            }
        }
        (score_diff, reconnections)
    }

    fn apply(&mut self, reconnections: &Vec<(usize, usize, Reconnection)>, graph: &Graph) {
        for (day, agent_index, reconnection) in reconnections {
            self.agents[*day][*agent_index].apply_reconnection(reconnection, graph);
        }
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
}

impl Agent {
    fn new(start: usize, graph: &Graph, when: &Vec<usize>, day: usize) -> Agent {
        let (dist, par_edge) = graph.calc_dist(start, when, day);
        let sz = calc_subtree_size(start, graph, &par_edge);
        Agent {
            start,
            day,
            dist,
            par_edge,
            sz,
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
            depth: usize,
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
                if agent.par_edge[v] == e.index {
                    continue;
                }
                // rootの子孫ではなく、繋がっていない頂点が隣接していれば繋げることができる
                if !is_child_vertex(e.to, root, &graph, &agent.par_edge, &agent.dist.vec) {
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
                    score_diff +=
                        (agent.sz[cur] - last_size) as i64 * (cur_dist - agent.dist.vec[cur]);
                    edge_path.reverse();

                    if score_diff < best_reconnection.score_diff {
                        best_reconnection.score_diff = score_diff;
                        best_reconnection.add_edge = e.index;
                        best_reconnection.remove_edge = agent.par_edge[cur];
                        best_reconnection.edge_path = edge_path;
                    }
                } else if agent.par_edge[e.to] == e.index {
                    // 子孫の頂点に探索を広げる
                    // 深さ3以上は探索しない
                    if depth == 2 {
                        continue;
                    }
                    dfs(e.to, root, depth + 1, when, graph, agent, best_reconnection);
                }
            }
        }

        let mut best_reconnection = Reconnection {
            score_diff: INF,
            add_edge: 0,
            remove_edge: 0,
            edge_path: vec![],
        };
        dfs(root, root, 0, when, graph, &self, &mut best_reconnection);

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
        let mut cur_dist = self.dist.vec[edge.other_vertex(cur)] + edge.weight;

        while cur != self.start {
            let par_edge = &graph.edges[self.par_edge[cur]];
            let weight = if when[par_edge.index] == self.day {
                PENALTY
            } else {
                par_edge.weight
            };
            let par = par_edge.other_vertex(cur);
            cur_dist += weight;
            let par_dist_diff = cur_dist - self.dist.vec[par];
            // 更新されなくなったら終了
            if par_dist_diff >= 0 {
                break;
            }
            score_diff += (self.sz[par] - self.sz[cur]) as i64 * par_dist_diff;
            edge_path.push(par_edge.index);
            cur = par;
        }
        edge_path.reverse();

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
        // debug_assert_eq!(self.par_edge[old_root], remove_edge.index);
        let subtree_size = self.sz[old_root];
        while cur != self.start {
            self.sz[cur] -= subtree_size;
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
}

fn calc_subtree_size(root: usize, graph: &Graph, par_edge: &Vec<usize>) -> Vec<usize> {
    let mut sz = vec![0; graph.n];

    fn dfs(v: usize, sz: &mut Vec<usize>, graph: &Graph, par_edge: &Vec<usize>) {
        sz[v] += 1;
        debug_assert!(sz[v] == 1);
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

fn is_child_vertex(
    v: usize,
    par: usize,
    graph: &Graph,
    par_edge: &Vec<usize>,
    dist: &Vec<i64>,
) -> bool {
    let mut v = v;
    while par_edge[v] != NA && v != par && dist[v] > dist[par] {
        // 親の頂点を取得する
        v = par_vertex(v, graph, par_edge);
    }
    v == par
}

fn par_vertex(v: usize, graph: &Graph, par_edge: &Vec<usize>) -> usize {
    graph.edges[par_edge[v]].other_vertex(v)
}

#[allow(unused)]
pub fn calc_actual_score_slow(input: &Input, graph: &Graph, state: &State, n: usize) -> i64 {
    let mut fk_sum = 0.;
    let mut base_dist_sum = 0;
    for v in 0..n {
        // 全ての辺が使える日で計算する
        base_dist_sum += graph.calc_dist_sum_slow(v, &state.when, input.d);
    }
    for day in 0..input.d {
        let mut dist_sum = 0;
        for v in 0..n {
            dist_sum += graph.calc_dist_sum_slow(v, &state.when, day);
        }
        let fk = (dist_sum - base_dist_sum) as f64 / (n * (input.n - 1)) as f64;
        fk_sum += fk;
    }
    (1e3 * (fk_sum / input.d as f64)).round() as i64
}

fn select_next(edge_index: usize, graph: &Graph, when: &Vec<usize>, d: usize) -> usize {
    loop {
        let next = if rnd::nextf() < 0.8 {
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
        };
        if next == when[edge_index] {
            continue;
        }
        return next;
    }
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
    let when = vec![0; 8];
    let mut agents = vec![
        Agent::new(s, &graph, &when, 0),
        Agent::new(s, &graph, &when, 1),
    ];
    dbg!(&agents[1]);
    let reconnection = agents[1].estimate_remove_edge(0, &graph, &when).unwrap();
    dbg!(&reconnection);
    agents[1].apply_reconnection(&reconnection, &graph);
    dbg!(&agents[1]);
    let reconnection = agents[1].estimate_add_edge(0, &graph, &when).unwrap();
    dbg!(&reconnection);
    agents[1].apply_reconnection(&reconnection, &graph);
    dbg!(&agents[1]);
}
