from board import Board
from minmax import MinMaxSolver
from minmax import playGame

import time
import copy
import numpy as np


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
    content += "\n"
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


def single_test(init_state, player, pruning, depth):
    state = copy.deepcopy(init_state)
    state = np.array(state)
    tictactoe = Board(len(state), x_starts=False, prints=False, init_state=state) # noqa
    minmax = MinMaxSolver(tictactoe, depth=depth, pruning=pruning)
    move = minmax.make_best_move(state, player)
    time = minmax.time_history[-1]
    node = minmax.nodes_history  # [-1]
    del state
    print(f'Move: {move} | Depth: {node} | Time: {time} ms')
    tictactoe.move(move, player)
    return time, node


def experiments(init_state, player, pruning, depth):
    timers = []
    nodes = []
    for _ in range(15):
        time, node = single_test(init_state, player, pruning, depth)
        timers.append(time)
        nodes.append(node)
    mean_time = np.mean(timers[3:13])
    deviatation_time = np.std(timers[3:13])
    mean_nodes = np.mean(nodes[3:13])
    deviatation_nodes = np.std(nodes[3:13])
    stats = {
        "time": {
            "mean": mean_time,
            "deviatation": deviatation_time
        },
        "nodes": {
            "mean": mean_nodes,
            "deviatation": deviatation_nodes
        },
    }
    return stats


def make_test_csv():
    content = "depth, alpha-pruning, x_starting, visited_nodes, time, moves, winner\n" # noqa
    for depth in range(MAX_DEPTH + 1):
        for alphabeta in [False, True]:
            for started in [True, False]:
                winner, d_hist, t_hist, m_hist = playGame(
                    size=SIZE,
                    x_starts=started,
                    depth=depth,
                    pruning=alphabeta,
                    printer=False,
                )
                content += f"{depth}, {alphabeta}, {started}, {d_hist}, {t_hist}, {m_hist}, {winner}\n" # noqa
    return content


def save_results(
    content: str, extension: str = "txt", location: str = "src/results/"
):
    file_name = "src/results/"
    file_name += name_file()
    file_name += f".{extension}"
    with open(file_name, "w+") as f:
        f.write(content)


# if __name__ == "__main__":
#     for _ in range(5):
#         print(f'Test #{_+1}:')
#         tic = time.perf_counter()
#         content = make_test_csv()
#         toc = time.perf_counter()
#         elapsed_time = toc - tic
#         print(f'took {elapsed_time} seconds.\n')
#         save_results(content, "csv")
#         del content
