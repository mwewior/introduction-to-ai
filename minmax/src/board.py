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
        for i in range(3 * size - 1):
            if (i + 1) % 3 == 0:
                plain_board.append(self.create_line("_"))
            else:
                plain_board.append(self.create_line(" "))
        plain_board.append(self.create_line(" "))
        for row in range(size):
            for col in range(size):
                num = self._simple_board[row][col]
                print_row, print_col = printable_row_col(row, col)
                if num < 10:
                    plain_board[print_row][print_col] = str(num)
                else:
                    plain_board[print_row][print_col] = str(num)[0]
                    plain_board[print_row][print_col + 1] = str(num)[1]
                # self._simple_board[row][col] = ''
        return plain_board

    def __init__(self, size: int = 3):
        self._size = size
        self._max_round = size ** 2
        self._round_count = 0
        self._players = ["x", "o"]
        self._x_player = False
        self._is_finished = False
        self._simple_board = self.create_board(self._size)
        self._print_board = self.create_print_board(self._size)
        self.board = np.ndarray((3, 3), dtype='U7')

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
        self._x_player = not self._x_player

    def current_player(self) -> str:
        if self._x_player:
            player = self._players[0]
        else:
            player = self._players[1]
        return player

    def convert_input(self, position: int):
        row = (int(position) - 1) // self._size  # całkowitoliczbowe
        col = int(position) % self._size - 1  # modulo
        if col == -1:
            col = self._size - 1
        return row, col

    def validate_input(self):
        position = input("Choose square:\n")
        try:
            position = int(position)
            row, col = self.convert_input(position)
            if type(self._simple_board[row][col]) is str:
                raise SquareOccupiedError
        except (ValueError, IndexError):
            print(f"\nValue must be a integer between 1 and {self._max_round}")
            position = self.validate_input()
        except SquareOccupiedError:
            print("Cannot put your mark on occupied square")
            position = self.validate_input()

        return position

    def read_input(self, human_player=True):
        player = self.current_player()
        if human_player:
            print('-----------------------------')
            print(f"\nCurrent player is '{player}'")
        position = self.validate_input()
        row, col = self.convert_input(position)
        return row, col

    def print(self):
        print("\n")
        board = self.get_print_board()
        for row in board:
            line = ""
            for char in row:
                line += str(char)
            print(line)

    def update_print_board(self, row, col):
        board = self.get_print_board()
        print_row, print_col = printable_row_col(row, col)
        board[print_row][print_col] = self._simple_board[row][col]
        board[print_row][print_col + 1] = " "

    def check_win(self):
        for i in range(self._size):
            row_count = 0
            col_count = 0
            diag_up_count = 0
            diag_down_count = 0

            for j in range(self._size):
                if self._simple_board[i][j] == self._simple_board[i][0]:
                    row_count += 1
                if self._simple_board[j][i] == self._simple_board[0][i]:
                    col_count += 1
                if self._simple_board[j][j] == self._simple_board[0][0]:
                    diag_up_count += 1
                if self._simple_board[-(j+1)][j] == self._simple_board[0][-1]:
                    diag_down_count += 1

            if row_count == self._size:
                self._is_finished = True
            if col_count == self._size:
                self._is_finished = True
            if diag_up_count == self._size:
                self._is_finished = True
            if diag_down_count == self._size:
                self._is_finished = True

            if self._is_finished:
                self.next_player()

    def move(self, human_player=True):
        row, col = self.read_input(human_player)
        self._simple_board[row][col] = self.current_player()
        self.board[row][col] = self.current_player()
        self.update_print_board(row, col)
        if human_player:
            self.print()
        self._round_count += 1
        self.next_player()
        if self.round() >= 2*self.size()-1:
            self.check_win()
