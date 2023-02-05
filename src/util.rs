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
        self.sum -= self.vec[idx];
        self.vec[idx] = value;
        self.sum += self.vec[idx];
    }
}
