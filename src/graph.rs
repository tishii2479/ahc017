use crate::def::*;
use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    to: usize,
    weight: i64,
    index: usize,
}

#[derive(Debug)]
pub struct Graph {
    adj: Vec<Vec<Edge>>,
}

impl Graph {
    pub fn new(n: usize, edges: &Vec<(usize, usize, i64)>) -> Graph {
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

        Graph { adj }
    }

    pub fn dijkstra(&self, start: usize, when: &Vec<usize>, day: usize) -> Vec<i64> {
        let mut dist = vec![INF; self.adj.len()];

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
                dist[e.to] = dist[v] + e.weight;
                heap.push((Reverse(dist[e.to]), e.to));
            }
        }

        dist
    }
}
