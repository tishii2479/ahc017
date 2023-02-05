import math
import multiprocessing
import subprocess

import pandas as pd

# mypy: ignore-errors

TL = 110.0


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


def run(case_num: int):
    subprocess.run("cargo build --release", shell=True)

    scores = []
    count = 0
    total = 0

    with multiprocessing.Pool() as pool:
        for seed, score, N, M, D, K in pool.imap_unordered(
            execute_case, range(case_num)
        ):
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
    print(f"ave: {ave:,.2f}")

    df = pd.DataFrame(scores, columns=["score", "case", "n", "m", "d", "k"])
    df["m/n"] = df.apply(lambda row: row["m"] / row["n"], axis=1)
    df["eval"] = df.apply(lambda row: row["m/n"] * math.pow(row["d"], 0.35), axis=1)
    df["rank"] = df["eval"] // 1
    df.groupby("rank").score.mean().map(int)

    return df


if __name__ == "__main__":
    df = run(1000)
    print(df.groupby("rank").score.mean().map(int))
