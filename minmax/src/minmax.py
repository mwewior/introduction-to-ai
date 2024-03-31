import numpy as np
from board import Board


WAGES = [
    [3, 2, 3],
    [2, 4, 2],
    [3, 2, 3]
]
SIZE = 3
DEPTH = 3


def heurisic(state):
    score = 0
    size = state.shape[0]
    for row in range(size):
        for col in range(size):
            if state[row][col] == 'x':
                score += WAGES[row][col]
            if state[row][col] == 'o':
                score -= WAGES[row][col]
    return score


def check_win(state: np.ndarray = np.ndarray((3, 3), dtype='U7')):
    size = state.shape[0]
    for i in range(size):
        row_count = 0
        col_count = 0
        diag_up_count = 0
        diag_down_count = 0

        for j in range(size):
            if state[i][j] == state[i][0]:
                if state[i][j] != "":
                    row_count += 1
            if state[j][i] == state[0][i]:
                if state[j][i] != "":
                    col_count += 1
            if state[j][j] == state[0][0]:
                if state[j][j] != "":
                    diag_down_count += 1
            if state[-(j+1)][j] == state[0][-1]:
                if state[-(j+1)][j] != "":
                    diag_up_count += 1

        if row_count == size:
            winner = state[i][0]
            return True, winner
        if col_count == size:
            winner = state[0][i]
            return True, winner
        if diag_down_count == size:
            winner = state[0][0]
            return True, winner
        if diag_up_count == size:
            winner = state[-1][0]
            return True, winner
    return False, None


def level(state: np.ndarray):
    level = 0
    size = state.shape[0]
    for i in range(size):
        for j in range(size):
            if state[i][j] == 'o' or state[i][j] == 'x':
                level += 1
    return level


def is_board_full(state):
    return np.all(state != '')


def possible_moves(state):
    return [(i, j) for i in range(3) for j in range(3) if state[i, j] == '']


def evaluate_game(state: np.ndarray) -> int:
    anybody, who = check_win(state)
    if not anybody:
        return heurisic(state)
    if who == 'x':
        return 100
    if who == 'o':
        return -100


def minmax(
    state: np.ndarray,
    depth: int,
    # action: tuple, #[int, int]
    maximizing: bool,
) -> int:

    max_level = state.size
    if check_win(state)[0] or depth == 0 or level(state) == max_level:
        return evaluate_game(state)

    if maximizing:
        best_score = -1*np.inf
        for move in possible_moves(state):
            next_state = state.copy()
            next_state[move] = 'x'
            next_score = minmax(next_state, depth-1, not maximizing)
            best_score = max(best_score, next_score)
        return best_score
    if not maximizing:
        best_score = np.inf
        for move in possible_moves(state):
            next_state = state.copy()
            next_state[move] = 'o'
            next_score = minmax(next_state, depth-1, not maximizing)
            best_score = min(best_score, next_score)
        return best_score


def make_best_move(state):  # , maximizing):
    best_move = None
    best_score = -1*np.inf
    for move in possible_moves(state):
        next_state = state.copy()
        next_state[move] = 'x'
        score = minmax(next_state, DEPTH, maximizing=False)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


tictactoe = Board(SIZE, True)
tictactoe.print()
state = np.array(tictactoe.board)
size = state.size
while tictactoe.round() < size and not tictactoe.finished():
    if not tictactoe._x_player:
        tictactoe.move()
    else:
        move = make_best_move(state)
        tictactoe.move(move)
    state = np.array(tictactoe.board)
