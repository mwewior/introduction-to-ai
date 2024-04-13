import numpy as np


class SquareOccupiedError(Exception):
    pass


def printable_row_col(row, col):
    print_row = row * 3 + 1
    print_col = col * (5 + 1) + 2
    return print_row, print_col


class Board:
    def create_board(self, size: int) -> list:
        plain_board = []
        for r in range(size):
            plain_board.append(list(range(r * size + 1, (r + 1) * size + 1)))
        return plain_board

    def create_line(self, symbol: str) -> list:
        null_line = []
        size = self._size
        for box in range(size):
            for _ in range(5):
                null_line.append(symbol)
            null_line.append("|")
        null_line.pop(-1)
        return null_line

    def create_print_board(self, size: int) -> list:
        plain_board = []
        simple_board = self.create_board(size)
        for i in range(3 * size - 1):
            if (i + 1) % 3 == 0:
                plain_board.append(self.create_line("_"))
            else:
                plain_board.append(self.create_line(" "))
        plain_board.append(self.create_line(" "))
        for row in range(size):
            for col in range(size):
                num = simple_board[row][col]
                print_row, print_col = printable_row_col(row, col)
                if num < 10:
                    plain_board[print_row][print_col] = str(num)
                else:
                    plain_board[print_row][print_col] = str(num)[0]
                    plain_board[print_row][print_col + 1] = str(num)[1]
                # self._simple_board[row][col] = ''
        return plain_board

    def level(self, state):
        level = 0
        size = state.shape[0]
        for i in range(size):
            for j in range(size):
                if state[i][j] == 'o' or state[i][j] == 'x':
                    level += 1
        return level

    def __init__(
            self,
            size: int = 3,
            x_starts: bool = True,
            prints: bool = True,
            # init_state: np.ndarray = np.ndarray((3, 3), dtype='U7')
            init_state: np.ndarray = np.full((3, 3), '')
            ):
        # init_state.fill('')
        self._size = size
        self._max_round = size ** 2
        self._round_count = self.level(init_state)
        self.prints = prints
        self._players = ["x", "o"]
        self._x_player = x_starts
        self._o_player = not x_starts
        self._is_finished = False
        self.board = init_state
        self._print_board = self.create_print_board(self._size)

    def size(self) -> int:
        return self._size

    def get_print_board(self) -> list:
        return self._print_board

    def finished(self):
        return self._is_finished

    def round(self) -> int:
        return self._round_count

    def max_round(self) -> int:
        return self._max_round

    def next_player(self) -> bool:
        self._round_count += 1
        self._x_player = not self._x_player
        self._o_player = not self._o_player

    def current_player(self) -> str:
        if self._x_player:
            player = "x"
        if self._o_player:
            player = "o"
        return player

    def last_player(self) -> str:
        if not self._x_player:
            player = "x"
        if not self._o_player:
            player = "o"
        return player

    def convert_input(self, position: int):
        row = (int(position) - 1) // self._size     # całkowitoliczbowe
        col = int(position) % self._size - 1        # modulo
        if col == -1:
            col = self._size - 1
        return row, col

    def validate_input(self):
        position = input("Choose square:\n")
        try:
            position = int(position)
            row, col = self.convert_input(position)
            if self.board[row][col] != '':
                raise SquareOccupiedError
        except (ValueError, IndexError):
            print(f"\nValue must be a integer between 1 and {self._max_round}")
            position = self.validate_input()
        except SquareOccupiedError:
            print("Cannot put your mark on occupied square")
            position = self.validate_input()
        return position

    def read_input(self):
        player = self.current_player()
        print(f"\nCurrent player is '{player}'")
        position = self.validate_input()
        row, col = self.convert_input(position)
        return row, col

    def update_print_board(self, row, col):
        board = self.get_print_board()
        print_row, print_col = printable_row_col(row, col)
        board[print_row][print_col] = self.board[row][col]
        board[print_row][print_col + 1] = " "

    def print(self):
        print("\n")
        for row in range(self.size()):
            for col in range(self.size()):
                if self.board[row][col] != "":
                    self.update_print_board(row, col)
        board = self.get_print_board()
        for row in board:
            line = ""
            for char in row:
                line += str(char)
            print(line)
        print('\n-----------------------------')

    def check_win(self):
        size = self.size()
        state = self.board
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
                self._is_finished = True
                winner = state[i][0]
                return True, winner
            if col_count == size:
                self._is_finished = True
                winner = state[0][i]
                return True, winner
            if diag_down_count == size:
                self._is_finished = True
                winner = state[0][0]
                return True, winner
            if diag_up_count == size:
                self._is_finished = True
                winner = state[-1][0]
                return True, winner
        return False, None

    def who_wins(self, winner):
        if self.prints:
            if winner is not None:
                print(f"\n\n'{winner}' won the game!\n")
            if self.round() == self.max_round():
                print('\nDraw!\n')

    def move(self, move=None, player=None):
        winner = None
        if move is None:
            row, col = self.read_input()
        else:
            (row, col) = move
        if player is None:
            self.board[row][col] = self.current_player()
        else:
            self.board[row][col] = player
        if self.prints:
            self.print()
        # self._round_count += 1
        self.next_player()
        if self.round() >= 2*self.size()-1:
            _, winner = self.check_win()
        self.who_wins(winner)
