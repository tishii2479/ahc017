use crate::{def::*, util::time};
use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub to: usize,
    pub weight: i64,
    pub index: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeData {
    pub v: usize,
    pub u: usize,
    pub weight: i64,
}

#[derive(Debug)]
pub struct Graph {
    pub n: usize,
    pub adj: Vec<Vec<Edge>>,
    pub pos: Vec<Pos>,
    pub edges: Vec<EdgeData>,
    pub dist: Vec<Vec<i64>>,
    pub dist_sum: Vec<i64>,
    pub par_edge: Vec<Vec<usize>>,
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
            .map(|(v, u, w)| EdgeData {
                v: *v,
                u: *u,
                weight: *w,
            })
            .collect();

        let when = vec![1; edges.len()];
        let mut graph = Graph {
            n,
            adj,
            pos,
            edges,
            dist: vec![],
            dist_sum: vec![],
            par_edge: vec![],
        };

        // 前計算
        for v in 0..n {
            let (d, p) = graph.dijkstra(v, &when, 0);
            graph.dist_sum.push(d.iter().sum());
            graph.dist.push(d);
            graph.par_edge.push(p);
        }

        graph
    }

    pub fn dijkstra(&self, start: usize, when: &Vec<usize>, day: usize) -> (Vec<i64>, Vec<usize>) {
        let mut dist = vec![INF; self.n];
        let mut par = vec![INF as usize; self.n];

        let mut heap = BinaryHeap::new();
        dist[start] = 0;
        heap.push((Reverse(0), start));

        while let Some((Reverse(d), v)) = heap.pop() {
            if dist[v] < d {
                continue;
            }
            for &e in &self.adj[v] {
                // その辺が使えない場合
                if when[e.index] == day {
                    continue;
                }
                if dist[e.to] <= dist[v] + e.weight {
                    continue;
                }
                par[e.to] = e.index;
                dist[e.to] = dist[v] + e.weight;
                heap.push((Reverse(dist[e.to]), e.to));
            }
        }

        (dist, par)
    }

    pub fn is_connected(&self, when: &Vec<usize>, day: usize) -> bool {
        // TODO: O(n)のアルゴリズムに書き換える
        return *self.dijkstra(0, &when, day).0.iter().max().unwrap() >= INF;
    }

    pub fn calc_dist_sum(&self, start: usize, when: &Vec<usize>, day: usize) -> i64 {
        self.dijkstra(start, &when, day).0.iter().sum()
    }

    pub fn get_path(&self, v: usize, u: usize) -> Vec<usize> {
        // v -> u の最短路に通る辺のインデックスを返す
        let mut ret = vec![];
        let mut cur = u;
        while self.par_edge[v][cur] != INF as usize {
            ret.push(self.par_edge[v][cur]);
            cur = self.edges[self.par_edge[v][cur]].v + self.edges[self.par_edge[v][cur]].u - cur;
        }
        ret.reverse();
        ret
    }
}
