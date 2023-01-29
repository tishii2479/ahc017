import os
import subprocess
import uuid

import matplotlib.pyplot as plt
import pandas as pd
from pandarallel import pandarallel


def calc_actual_score(in_file: str, output: str) -> int:
    output_file = f"./tmp/out_{uuid.uuid4()}.txt"
    with open(output_file, "w") as f:
        f.write(output)
    cmd = f"./tools/target/release/vis {in_file} {output_file}"
    proc = subprocess.run(
        cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True
    )
    os.remove(output_file)
    # stderr = proc.stderr.decode("utf8")
    stdout = proc.stdout.decode("utf8")
    score = -1
    for line in stdout.splitlines():
        print("line:", line)
        if len(line) >= 7 and line[:7].lower() == "score =":
            score = int(line.split()[-1])
    assert score != -1
    return score


def visualize_score_progress(in_file: str, score_log_file: str) -> None:
    # score, time, actual_score
    # 実際のスコアと、評価値の推移をプロットする
    log_df = pd.read_csv(
        score_log_file, header=None, names=["score", "time", "actual_score"]
    )
    # log_df["actual_score"] = log_df.output.parallel_apply(
    #     lambda x: calc_actual_score(in_file, x)
    # )
    log_df.actual_score = log_df.actual_score.apply(lambda x: min(1e9, x), axis=1)

    fig, ax_log = plt.subplots()

    ax_log.plot(log_df.time, log_df.score, label="score")
    ax_actual = ax_log.twinx()

    ax_actual.plot(log_df.time, log_df.actual_score, c="red", label="actual")
    fig.legend()
    fig.savefig("out/create_initial_state_score_progress.png")
    plt.show()


if __name__ == "__main__":
    pandarallel.initialize()
    visualize_score_progress(
        "tools/in/0002.txt", "out/create_initial_state_score_progress.csv"
    )
