pub mod def {
    pub const PENALTY: i64 = 1_000_000_000;
    pub const INF: i64 = 100_000_000_000_000;
    pub const NA: usize = 100_000_000_000_000;

    #[derive(Debug)]
    pub struct Input {
        pub n: usize,
        pub m: usize,
        pub d: usize,
        pub k: usize,
    }

    #[derive(Debug, PartialEq)]
    pub struct State {
        pub when: Vec<usize>,
        pub repair_counts: Vec<usize>,
        pub score: f64,
    }

    impl State {
        pub fn new(d: usize, when: Vec<usize>, score: f64) -> State {
            let mut repair_counts = vec![0; d];
            for i in &when {
                if *i == NA {
                    continue;
                }
                repair_counts[*i] += 1;
            }

            State {
                when,
                repair_counts,
                score,
            }
        }

        pub fn update_when(&mut self, edge_index: usize, day: usize) {
            if self.when[edge_index] != NA {
                self.repair_counts[self.when[edge_index]] -= 1;
            }
            self.when[edge_index] = day;
            if self.when[edge_index] != NA {
                self.repair_counts[self.when[edge_index]] += 1;
            }
        }

        pub fn output(&self) -> String {
            let mut ret = String::new();
            for e in &self.when {
                ret += &format!("{} ", e + 1);
            }
            ret
        }
    }

    #[derive(Debug)]
    pub struct Pos {
        pub x: i64,
        pub y: i64,
    }

    impl Pos {
        pub fn dist(&self, to: &Pos) -> i64 {
            let dy = to.y - self.y;
            let dx = to.x - self.x;
            dy * dy + dx * dx
        }
    }
}
pub mod graph {
    use crate::{def::*, util::VecSum};
    use std::{cmp::Reverse, collections::VecDeque};

    #[derive(Clone, Copy, Debug)]
    pub struct Edge {
        pub to: usize,
        pub weight: i64,
        pub index: usize,
    }

    #[derive(Debug)]
    pub struct EdgeData {
        pub v: usize,
        pub u: usize,
        pub weight: i64,
        pub index: usize,
    }

    impl EdgeData {
        pub fn other_vertex(&self, v: usize) -> usize {
            debug_assert!(self.u == v || self.v == v);
            self.u + self.v - v
        }

        pub fn has_vertex(&self, v: usize) -> bool {
            self.v == v || self.u == v
        }
    }

    #[derive(Debug)]
    pub struct Graph {
        pub n: usize,
        pub adj: Vec<Vec<Edge>>,
        pub pos: Vec<Pos>,
        pub edges: Vec<EdgeData>,
    }

    impl Graph {
        pub fn new(n: usize, edges: Vec<(usize, usize, i64)>, pos: Vec<(i64, i64)>) -> Graph {
            let mut adj = vec![vec![]; n];

            for (i, (u, v, w)) in edges.iter().enumerate() {
                adj[*u].push(Edge {
                    to: *v,
                    weight: *w,
                    index: i,
                });
                adj[*v].push(Edge {
                    to: *u,
                    weight: *w,
                    index: i,
                });
            }

            let pos = pos.iter().map(|(x, y)| Pos { x: *x, y: *y }).collect();
            let edges: Vec<EdgeData> = edges
                .iter()
                .enumerate()
                .map(|(i, (v, u, w))| EdgeData {
                    v: *v,
                    u: *u,
                    weight: *w,
                    index: i,
                })
                .collect();

            Graph { n, adj, pos, edges }
        }

        pub fn calc_dist(
            &self,
            start: usize,
            when: &Vec<usize>,
            day: usize,
        ) -> (VecSum, Vec<usize>) {
            let mut dist = VecSum::new(vec![INF; self.n]);
            let mut par_edge = vec![NA; self.n];
            let mut q = VecDeque::new();
            dist.set(start, 0);
            q.push_back((Reverse(0), start));

            while let Some((Reverse(d), v)) = q.pop_front() {
                if dist.vec[v] < d {
                    continue;
                }
                for &e in &self.adj[v] {
                    // その辺が使えない場合
                    let weight = if when[e.index] == day {
                        PENALTY
                    } else {
                        e.weight
                    };
                    if dist.vec[e.to] <= dist.vec[v] + weight {
                        continue;
                    }
                    par_edge[e.to] = e.index;
                    dist.set(e.to, dist.vec[v] + weight);
                    if weight < 25_000 {
                        q.push_front((Reverse(dist.vec[e.to]), e.to));
                    } else {
                        q.push_back((Reverse(dist.vec[e.to]), e.to));
                    }
                }
            }

            (dist, par_edge)
        }

        #[allow(unused)]
        pub fn calc_dist_sum_slow(&self, start: usize, when: &Vec<usize>, day: usize) -> i64 {
            let (dist, _) = self.calc_dist(start, &when, day);
            dist.sum
        }

        pub fn find_closest_point(&self, anchor: &Pos) -> usize {
            let mut min_dist = self.pos[0].dist(&anchor);
            let mut min_point = 0;
            for (i, p) in self.pos.iter().enumerate() {
                if p.dist(&anchor) < min_dist {
                    min_dist = p.dist(&anchor);
                    min_point = i;
                }
            }

            min_point
        }
    }
}
pub mod solver {
    use crate::{
        def::*,
        graph::Graph,
        util::{rnd, time, VecSum},
    };

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
        const AGENT_N: usize = 8;
        const LOOP_INTERVAL: usize = 1000;
        const UPDATE_INTERVAL: f64 = 0.0501;

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
            let new_score = state.score + score_diff;
            let adopt = (-(new_score - state.score) / temp).exp() > rnd::nextf();

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
                let r = rnd::gen_range(300, 500) as f64;
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
        ) -> (f64, Vec<(usize, usize, Reconnection)>) {
            let mut score_diff = 0.;
            let mut reconnections = vec![];
            for (i, a) in self.agents[change.next].iter().enumerate() {
                if score_diff >= INF as f64 {
                    break;
                }
                if let Some(reconnection) =
                    a.estimate_remove_edge(change.edge_index, &graph, &state.when)
                {
                    score_diff += reconnection.score_diff as f64;
                    reconnections.push((change.next, i, reconnection));
                }
            }
            for (i, a) in self.agents[change.prev].iter().enumerate() {
                if score_diff >= INF as f64 {
                    break;
                }
                if let Some(reconnection) =
                    a.estimate_add_edge(change.edge_index, &graph, &state.when)
                {
                    score_diff += reconnection.score_diff as f64;
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
                            score_diff += (agent.sz[cur] - last_size) as i64
                                * (cur_dist - agent.dist.vec[cur]);
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

            let dist_diff =
                self.dist.vec[edge.other_vertex(root)] + edge.weight - self.dist.vec[root];
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
}
pub mod util {
    #[allow(unused_features)]

    pub mod rnd {
        #[allow(unused)]
        static mut S: usize = 88172645463325252;

        #[allow(unused)]
        #[inline]
        pub fn next() -> usize {
            unsafe {
                S = S ^ S << 7;
                S = S ^ S >> 9;
                S
            }
        }

        #[allow(unused)]
        #[inline]
        pub fn nextf() -> f64 {
            (next() & 4294967295) as f64 / 4294967296.
        }

        #[allow(unused)]
        #[inline]
        pub fn gen_range(low: usize, high: usize) -> usize {
            (next() % (high - low)) + low
        }

        #[allow(unused)]
        pub fn shuffle<I>(vec: &mut Vec<I>) {
            for i in 0..vec.len() {
                let j = gen_range(0, vec.len());
                vec.swap(i, j);
            }
        }
    }

    pub mod time {
        static mut START: f64 = -1.;
        #[allow(unused)]
        pub fn start_clock() {
            let _ = elapsed_seconds();
        }

        #[allow(unused)]
        #[inline]
        pub fn elapsed_seconds() -> f64 {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            unsafe {
                if START < 0. {
                    START = t;
                }
                t - START
            }
        }
    }

    #[allow(unused)]
    pub fn min_index<I>(vec: &Vec<I>) -> usize
    where
        I: Ord,
    {
        let mut ret = 0;
        for i in 0..vec.len() {
            if vec[i] < vec[ret] {
                ret = i;
            }
        }
        return ret;
    }

    #[derive(Debug)]
    pub struct VecSum {
        pub vec: Vec<i64>,
        pub sum: i64,
    }

    impl VecSum {
        pub fn new(vec: Vec<i64>) -> VecSum {
            let sum = vec.iter().sum();
            VecSum { vec, sum }
        }

        pub fn set(&mut self, idx: usize, value: i64) {
            self.sum += value - self.vec[idx];
            self.vec[idx] = value;
        }
    }
}

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
    const TIME_LIMIT: f64 = 5.8;

    let (input, graph) = read_input();

    // 初期解の生成
    let mut state = create_random_initial_state(&input);

    // 最適化
    optimize_state(&mut state, &input, &graph, TIME_LIMIT);

    let output = state.output();
    println!("{}", output);

    eprintln!("Time elapsed = {}", time::elapsed_seconds());
}
