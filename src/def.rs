use crate::graph::Graph;

pub const INF: i64 = 1_000_000_000;

#[derive(Debug)]
pub struct Input {
    pub n: usize,
    pub m: usize,
    pub d: usize,
    pub k: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    pub when: Vec<usize>,
    pub repair_counts: Vec<usize>,
    pub cached_score: Vec<i64>,
    pub needs_update: Vec<bool>,
    pub score: i64,
}

impl State {
    pub fn new(d: usize, when: Vec<usize>, score: i64) -> State {
        let mut repair_counts = vec![0; d];
        for i in &when {
            if *i == INF as usize {
                continue;
            }
            repair_counts[*i] += 1;
        }

        State {
            when,
            repair_counts,
            cached_score: vec![0; d],
            needs_update: vec![true; d],
            score,
        }
    }

    pub fn update_when(&mut self, edge_index: usize, day: usize) {
        if self.when[edge_index] != INF as usize {
            self.repair_counts[self.when[edge_index]] -= 1;
            self.needs_update[self.when[edge_index]] = true;
        }
        self.when[edge_index] = day;
        if self.when[edge_index] != INF as usize {
            self.repair_counts[self.when[edge_index]] += 1;
            self.needs_update[self.when[edge_index]] = true;
        }
    }

    pub fn output(&self) -> String {
        let mut ret = String::new();
        for e in &self.when {
            ret += &format!("{} ", e + 1);
        }
        ret += "\n";
        ret
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pos {
    pub x: i64,
    pub y: i64,
}
