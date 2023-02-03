use crate::{def::*, util::VecSum};
use std::{cmp::Reverse, collections::VecDeque};

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
    pub index: usize,
}

impl EdgeData {
    pub fn other_vertex(&self, v: usize) -> usize {
        assert!(self.u == v || self.v == v);
        self.u + self.v - v
    }
}

impl EdgeData {
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
    pub dist: Vec<VecSum>,
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
            .enumerate()
            .map(|(i, (v, u, w))| EdgeData {
                v: *v,
                u: *u,
                weight: *w,
                index: i,
            })
            .collect();

        let when = vec![1; edges.len()];
        let mut graph = Graph {
            n,
            adj,
            pos,
            edges,
            dist: vec![],
            par_edge: vec![],
        };

        // TODO: いらない?
        // 前計算
        for v in 0..n {
            let mut dist = vec![INF; graph.adj.len()];
            let mut par_edge = vec![NA; graph.adj.len()];
            dist[v] = 0;
            let mut dist = VecSum::new(dist);
            graph.dijkstra(v, &when, 0, &mut dist, &mut par_edge);
            graph.dist.push(dist);
            graph.par_edge.push(par_edge);
        }

        graph
    }

    pub fn dijkstra(
        &self,
        start: usize,
        when: &Vec<usize>,
        day: usize,
        dist: &mut VecSum,
        par_edge: &mut Vec<usize>,
    ) {
        let mut q = VecDeque::new();
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
                q.push_back((Reverse(dist.vec[e.to]), e.to));
            }
        }
    }
    pub fn calc_dist_sum(&self, start: usize, when: &Vec<usize>, day: usize) -> i64 {
        let mut dist = vec![INF; self.n];
        dist[start] = 0;
        let mut dist = VecSum::new(dist);
        self.dijkstra(start, &when, day, &mut dist, &mut vec![NA; self.n]);
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
