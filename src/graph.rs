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

    pub fn dijkstra(&self, start: usize, when: &Vec<usize>, day: usize) -> (VecSum, Vec<usize>) {
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
        let (dist, _) = self.dijkstra(start, &when, day);
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
