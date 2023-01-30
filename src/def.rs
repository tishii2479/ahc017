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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Clone, Copy, Debug)]
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
