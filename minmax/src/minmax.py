import numpy as np
from board import Board


WAGES = [
    [3, 2, 3],
    [2, 4, 2],
    [3, 2, 3]
]

SIZE = 3
DEPTH = 3
WITH_ALPHA = False


class MinMaxSolver:
    def __init__(self, game: Board, depth: int = DEPTH, with_alpha: bool = WITH_ALPHA) -> None:
        self.game = game
        self.current_depth = 0
        self.depths_history = []
        self.depth = depth
        self.with_alpha = with_alpha

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
        return [(i, j) for i in range(3) for j in range(3) if state[i, j] == '']

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
        if self.check_win(state)[0] or depth == 0 or level(state) == max_level:
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
        if self.check_win(state)[0] or depth == 0 or level(state) == max_level:
            return self.evaluate_game(state)

        if maximizing:
            best_score = -1*np.inf
            for move in self.possible_moves(state):
                next_state = state.copy()
                next_state[move] = 'x'
                next_score = self.alpha_pruning(next_state, depth-1, False, alpha, beta)
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
                next_score = self.alpha_pruning(next_state, depth-1, True, alpha, beta)
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
            for move in self.possible_moves(state):
                next_state = state.copy()
                next_state[move] = 'x'
                if self.with_alpha:
                    score = self.alpha_pruning(next_state, self.depth, False)
                else:
                    score = self.minmax(next_state, self.depth, False)
                # score = self.minmax(next_state, DEPTH, False)
                # score = self.alpha_pruning(next_state, DEPTH, False)
                if score > best_score:
                    best_score = score
                    best_move = move
            self.depths_history.append(self.current_depth)
            return best_move
        if not maximizing:
            best_score = np.inf
            for move in self.possible_moves(state):
                next_state = state.copy()
                next_state[move] = 'o'
                if self.with_alpha:
                    score = self.alpha_pruning(next_state, self.depth, True)
                else:
                    score = self.minmax(next_state, self.depth, True)
                # score = self.minmax(next_state, DEPTH, True)
                # score = self.alpha_pruning(next_state, DEPTH, True)
                if score < best_score:
                    best_score = score
                    best_move = move
            self.depths_history.append(self.current_depth)
            return best_move


if __name__ == "__main__":
    tictactoe = Board(SIZE, x_starts=False)
    mm = MinMaxSolver(tictactoe)
    tictactoe.print()
    state = np.array(mm.game.board)
    size = state.size
    while mm.game.round() < size and not mm.game.finished():
        if mm.game._o_player:
            # tictactoe.move()
            move = mm.make_best_move(state, False)
            mm.game.move(move)
        elif mm.game._x_player:
            # tictactoe.move()
            move = mm.make_best_move(state, True)
            mm.game.move(move)
        state = np.array(mm.game.board)
    print(mm.depths_history)
