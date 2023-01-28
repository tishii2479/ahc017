pub const INF: i64 = 1_000_000_000;

#[derive(Debug)]
pub struct Input {
    pub n: usize,
    pub m: usize,
    pub d: usize,
    pub k: usize,
}

#[derive(Debug)]
pub struct State {
    pub when: Vec<usize>,
    pub score: i64,
}

impl State {
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
