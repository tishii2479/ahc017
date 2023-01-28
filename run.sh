FILE=$1

cargo build --release

./tools/target/release/tester ./target/release/sol < tools/in/$FILE.txt > tools/out/$FILE.txt

pbcopy < tools/out/$FILE.txt