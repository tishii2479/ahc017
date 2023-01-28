mod util;

fn main() {
    for _ in 0..2277 {
        print!("{} ", util::rnd::gen_range(1, 12));
    }
    println!();
}
