import matplotlib.pyplot as plt
import pandas as pd


def visualize_score_progress(in_file: str, score_log_file: str) -> None:
    # score, actual_score, time
    # 実際のスコアと、評価値の推移をプロットする
    log_df = pd.read_csv(
        score_log_file, header=None, names=["actual_score", "score", "time"]
    )
    log_df.actual_score = log_df.apply(lambda x: min(1e7, x.actual_score), axis=1)

    fig, ax_log = plt.subplots()

    ax_log.plot(log_df.time, log_df.score, label="score")
    ax_actual = ax_log.twinx()

    ax_actual.plot(log_df.time, log_df.actual_score, c="red", label="actual")
    fig.legend()
    fig.savefig("out/create_initial_state_score_progress.png")
    plt.show()


if __name__ == "__main__":
    visualize_score_progress(
        "tools/in/0002.txt", "out/create_initial_state_score_progress.csv"
    )
