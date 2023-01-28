import multiprocessing
import subprocess

# mypy: ignore-errors

CASE = 1000
TL = 40.0


def execute_case(seed):
    input_file_path = f"tools/in/{seed:04}.txt"
    output_file_path = f"tools/out/{seed:04}.txt"

    tester_path = "./tools/target/release/tester"
    solver_cmd = "./target/release/sol"

    with open(input_file_path, "r") as f:
        M, eps = f.readline().split()
        M = int(M)
        eps = float(eps)

    # if M % 2 == 0 and round(eps * 100) % 2 == 0:
    #     return 0, 0, M, eps, 0

    cmd = f"{tester_path} {solver_cmd} < {input_file_path} > {output_file_path}"
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, timeout=TL, shell=True)
    stderr = proc.stderr.decode("utf8")
    score = -1
    err = -1
    for line in stderr.splitlines():
        if len(line) >= 7 and line[:7].lower() == "score =":
            score = int(line.split()[-1])
        if len(line) >= 3 and line[:3].lower() == "e =":
            err = int(line.split()[-1])
    assert score != -1

    return seed, score, M, eps, err


def main():
    subprocess.run("cargo build --release", shell=True)

    scores = []
    count = 0
    total = 0
    err_count = 0

    with multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 2)) as pool:
        for seed, score, M, eps, err in pool.imap_unordered(execute_case, range(CASE)):
            eps = float(eps)
            M = int(M)
            # if M % 2 == 0 and round(eps * 100) % 2 == 0:
            #     continue

            err = int(err)
            count += 1

            try:
                scores.append((int(score), f"{seed:04}", M, eps, err))
                total += scores[-1][0]
            except ValueError:
                print(seed, "ValueError", flush=True)
                print(score, flush=True)
                exit()
            except IndexError:
                print(seed, "IndexError", flush=True)
                print(f"error: {score}", flush=True)
                exit()

            err_count += err

            print(
                f"case {seed:3}: (score: {scores[-1][0]:>13,}, current ave: "
                + f"{total / count:>15,.2f}, m: {M:3}, eps: {eps:4.2f}, err: {err:3})",
                flush=True,
            )

    print("=" * 100)
    scores.sort()
    ave = total / count
    print(f"ave: {ave}")

    print(f"err ave: {err_count / count}")

    M_div = list(range(10, 101, 10))
    eps_div = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    def is_in(v, low, high):
        d = 1e-5
        return low - d <= v <= high + d

    # score table
    print("|".rjust(10), end="")
    for j in range(len(eps_div) - 1):
        eps_low, eps_high = eps_div[j], eps_div[j + 1]
        print(f"{eps_low}~{eps_high}".rjust(10), end=" ")
    print()
    print("-" * 97)

    for i in range(len(M_div) - 1):
        M_low, M_high = M_div[i], M_div[i + 1]
        print(f"{M_low}~{M_high} |".rjust(10), end="")
        for j in range(len(eps_div) - 1):
            eps_low, eps_high = eps_div[j], eps_div[j + 1]

            score_sum = 0
            counter = 0
            for score, seed, M, eps, err in scores:
                if is_in(M, M_low, M_high) and is_in(eps, eps_low, eps_high):
                    score_sum += score
                    counter += 1
            if counter == 0:
                print(" " * 7 + "nan", end=" ")
            else:
                print(f"{int(score_sum / counter):10}", end=" ")
        print()
    print("-" * 97)
    print()

    # error table
    print("|".rjust(10), end="")
    for j in range(len(eps_div) - 1):
        eps_low, eps_high = eps_div[j], eps_div[j + 1]
        print(f"{eps_low}~{eps_high}".rjust(10), end=" ")
    print()
    print("-" * 97)

    for i in range(len(M_div) - 1):
        M_low, M_high = M_div[i], M_div[i + 1]
        print(f"{M_low}~{M_high} |".rjust(10), end="")
        for j in range(len(eps_div) - 1):
            eps_low, eps_high = eps_div[j], eps_div[j + 1]

            score_sum = 0
            counter = 0
            for score, seed, M, eps, err in scores:
                if is_in(M, M_low, M_high) and is_in(eps, eps_low, eps_high):
                    score_sum += err
                    counter += 1
            if counter == 0:
                print(" " * 7 + "nan", end=" ")
            else:
                print(f"{score_sum / counter:10.2f}", end=" ")
        print()
    print("-" * 97)
    print()


if __name__ == "__main__":
    main()
