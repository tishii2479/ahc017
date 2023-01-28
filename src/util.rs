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
