try:
    from board import Board
except ModuleNotFoundError:
    from src.board import Board

import time
import numpy as np


WAGES = [
    [3, 2, 3],
    [2, 4, 2],
    [3, 2, 3]
]

PRINT = False
SIZE = 3
DEPTH = 3
PRUNING = False


class MinMaxSolver:
    def __init__(
            self,
            game: Board,
            depth: int = DEPTH,
            pruning: bool = PRUNING,
            ):
        self.game = game
        self.depth = depth
        self.pruning = pruning
        self.time_history = []
        self.current_depth = 0
        self.depths_history = []

    def heurisic(self, state: np.ndarray):
        score = 0
        size = state.shape[0]
        for row in range(size):
            for col in range(size):
                if state[row][col] == 'x':
                    score += WAGES[row][col]
                if state[row][col] == 'o':
                    score -= WAGES[row][col]
        return score

    def check_win(self, state: np.ndarray = np.ndarray((3, 3), dtype='U7')):
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

    def level(self, state: np.ndarray):
        level = 0
        size = state.shape[0]
        for i in range(size):
            for j in range(size):
                if state[i][j] == 'o' or state[i][j] == 'x':
                    level += 1
        return level

    def is_board_full(self, state):
        return np.all(state != '')

    def possible_moves(self, state):
        moves = []
        for i in range(state.shape[0]):
            for j in range(state.shape[0]):
                if state[i, j] == '':
                    moves.append((i, j))
        return moves

    def evaluate_game(self, state: np.ndarray) -> int:
        anybody, who = self.check_win(state)
        if not anybody:
            return self.heurisic(state)
        if who == 'x':
            return 100
        if who == 'o':
            return -100

    def minmax(
        self,
        state: np.ndarray,
        depth: int,
        # action: tuple, #[int, int]
        maximizing: bool,
    ) -> int:

        self.current_depth += 1
        max_level = state.size
        if (
            self.check_win(state)[0] or
            depth == 0 or
            self.level(state) == max_level
                ):
            return self.evaluate_game(state)

        if maximizing:
            best_score = -1*np.inf
            for move in self.possible_moves(state):
                next_state = state.copy()
                next_state[move] = 'x'
                next_score = self.minmax(next_state, depth-1, not maximizing)
                best_score = max(best_score, next_score)
            return best_score
        if not maximizing:
            best_score = np.inf
            for move in self.possible_moves(state):
                next_state = state.copy()
                next_state[move] = 'o'
                next_score = self.minmax(next_state, depth-1, not maximizing)
                best_score = min(best_score, next_score)
            return best_score

    def alpha_pruning(
        self,
        state: np.ndarray,
        depth: int,
        # action: tuple, # (int, int)
        maximizing: bool,
        alpha: int = -np.inf,
        beta: int = np.inf,
    ) -> int:
        self.current_depth += 1
        max_level = state.size
        if (
            self.check_win(state)[0] or
            depth == 0 or
            self.level(state) == max_level
                ):
            return self.evaluate_game(state)

        if maximizing:
            best_score = -1*np.inf
            for move in self.possible_moves(state):
                next_state = state.copy()
                next_state[move] = 'x'
                next_score = self.alpha_pruning(
                    next_state, depth-1, False, alpha, beta)
                best_score = max(best_score, next_score)
                if best_score > beta:
                    break
                alpha = max(alpha, best_score)
            return best_score
        if not maximizing:
            best_score = np.inf
            for move in self.possible_moves(state):
                next_state = state.copy()
                next_state[move] = 'o'
                next_score = self.alpha_pruning(
                    next_state, depth-1, True, alpha, beta)
                best_score = min(best_score, next_score)
                if best_score < alpha:
                    break
                beta = min(beta, best_score)
            return best_score

    def make_best_move(self, state, maximizing):
        best_move = None
        self.current_depth = 0
        if maximizing:
            best_score = -1*np.inf
            tic = time.time()
            for move in self.possible_moves(state):
                next_state = state.copy()
                next_state[move] = 'x'
                if self.pruning:
                    score = self.alpha_pruning(next_state, self.depth, False)
                else:
                    score = self.minmax(next_state, self.depth, False)
                if score > best_score:
                    best_score = score
                    best_move = move
            toc = time.time()
            elapsed_time = toc - tic
            self.time_history.append(round(1000*elapsed_time, 5))
            self.depths_history.append(self.current_depth)
            return best_move
        if not maximizing:
            best_score = np.inf
            tic = time.time()
            for move in self.possible_moves(state):
                next_state = state.copy()
                next_state[move] = 'o'
                if self.pruning:
                    score = self.alpha_pruning(next_state, self.depth, True)
                else:
                    score = self.minmax(next_state, self.depth, True)
                if score < best_score:
                    best_score = score
                    best_move = move
            toc = time.time()
            elapsed_time = toc - tic
            self.time_history.append(round(1000*elapsed_time, 5))
            self.depths_history.append(self.current_depth)
            return best_move


def playGame(size, x_starts, depth, pruning, printer=PRINT):
    tictactoe = Board(size, x_starts=x_starts, prints=printer)
    solver = MinMaxSolver(tictactoe, depth, pruning)
    if printer:
        tictactoe.print()
    state = np.array(solver.game.board)
    size = state.size
    while solver.game.round() < size and not solver.game.finished():
        if solver.game._o_player:
            # tictactoe.move()
            move = solver.make_best_move(state, False)
            solver.game.move(move)
        elif solver.game._x_player:
            # tictactoe.move()
            move = solver.make_best_move(state, True)
            solver.game.move(move)
        state = np.array(solver.game.board)
    d_history = solver.depths_history
    t_history = solver.time_history
    if printer:
        print(d_history)
        print(t_history)
    _, winner = tictactoe.check_win()
    # if printer:
    # tictactoe.print()
    del state, tictactoe, solver
    return winner, d_history, t_history


if __name__ == "__main__":
    playGame(size=SIZE, depth=3, pruning=PRUNING, printer=PRINT)
