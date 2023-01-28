FILE=$1

cargo build --release

cargo run --release < tools/in/$FILE.txt > tools/out/$FILE.txt

./tools/target/release/vis tools/in/$FILE.txt tools/out/$FILE.txt

pbcopy < tools/out/$FILE.txt
