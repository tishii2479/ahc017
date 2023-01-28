import multiprocessing
import subprocess

# mypy: ignore-errors

CASE = 100
TL = 7.0


def execute_case(seed):
    input_file_path = f"tools/in/{seed:04}.txt"
    output_file_path = f"tools/out/{seed:04}.txt"

    solver_cmd = "./target/release/ahc017"

    with open(input_file_path, "r") as f:
        N, M, D, K = map(int, f.readline().split())

    cmd = f"{solver_cmd} < {input_file_path} > {output_file_path}"
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, timeout=TL, shell=True)
    stderr = proc.stderr.decode("utf8")
    score = -1
    for line in stderr.splitlines():
        if len(line) >= 7 and line[:7].lower() == "score =":
            score = int(line.split()[-1])
    assert score != -1

    return seed, score, N, M, D, K


def main():
    subprocess.run("cargo build --release", shell=True)

    scores = []
    count = 0
    total = 0

    with multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 2)) as pool:
        for seed, score, N, M, D, K in pool.imap_unordered(execute_case, range(CASE)):
            count += 1

            try:
                scores.append((int(score), f"{seed:04}", N, M, D, K))
                total += scores[-1][0]
            except ValueError:
                print(seed, "ValueError", flush=True)
                print(score, flush=True)
                exit()
            except IndexError:
                print(seed, "IndexError", flush=True)
                print(f"error: {score}", flush=True)
                exit()

            print(
                f"case {seed:3}: (score: {scores[-1][0]:>13,}, current ave: "
                + f"{total / count:>15,.2f}, "
                + f"N = {N:4}, M = {M:4}, D = {D:2}, K = {K:3})",
                flush=True,
            )

    print("=" * 100)
    scores.sort()
    ave = total / count
    print(f"ave: {ave}")


if __name__ == "__main__":
    main()
