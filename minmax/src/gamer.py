# from board import Board
# from minmax import MinMaxSolver
from minmax import playGame
import time


WAGES = [
    [3, 2, 3],
    [2, 4, 2],
    [3, 2, 3]
]

PRINT = False
SIZE = 3
MAX_DEPTH = 8
PRUNING = False


def get_time() -> str:
    current_time = ""
    time_struct = time.localtime()
    current_time += str(time_struct.tm_year)
    current_time += "-"
    current_time += str(time_struct.tm_mon)
    current_time += "-"
    current_time += str(time_struct.tm_mday)
    current_time += "_"
    current_time += str(time_struct.tm_hour)
    current_time += "-"
    current_time += str(time_struct.tm_min)
    current_time += "-"
    current_time += str(time_struct.tm_sec)
    return current_time


def name_file() -> str:
    file_name = "minmax-test_"
    file_name += get_time()
    return file_name


def make_test():
    content = name_file()
    content += '\n'
    for started in [True, False]:
        for d in range(MAX_DEPTH + 1):
            # print(f"Depth {d}")
            txt = f"\n\nDepth is {d},\n'x' is starting: {started}\n"
            content += txt
            # print(txt)
            for alphabeta in [False, True]:
                txt = f"\nAlpha-pruning: {alphabeta}\n"
                content += txt
                # print(txt)
                winner, d_hist, t_hist = playGame(
                    size=SIZE,
                    x_starts=started,
                    depth=d,
                    pruning=alphabeta,
                    printer=False,
                )
                txt = f"{d_hist}\n{t_hist}\n{winner} won the game!\n"
                content += txt
                # print(txt)
    return content


def save_results(content):
    file_name = "src/results/"
    file_name += name_file()
    file_name += ".txt"
    with open(file_name, "w+") as f:
        f.write(content)


if __name__ == "__main__":
    content = make_test()
    save_results(content)

    # f = "results/"
    # f += name_file()
    # f += ".txt"
    # with open(f, "w") as file:
    #     file.write(content)
