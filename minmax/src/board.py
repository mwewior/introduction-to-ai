def printable_row_col(row, col):
    print_row = row*3 + 1
    print_col = col*(5+1) + 2
    return print_row, print_col


class Player:
    def __init__(self, char: str) -> None:
        self._char = char
        if char == 'x':
            self._logic_value = True
        else:
            self._logic_value = False


class Board:

    def create_board(self, size: int) -> list:
        plain_board = []
        for r in range(size):
            plain_board.append(list(range(r*size+1, (r+1)*size+1)))
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
        for i in range(3*size-1):
            if (i+1) % 3 == 0:
                plain_board.append(self.create_line('_'))
            else:
                plain_board.append(self.create_line(' '))
        plain_board.append(self.create_line(' '))
        for row in range(size):
            for col in range(size):
                num = self._simple_board[row][col]
                print_row, print_col = printable_row_col(row, col)
                if num < 10:
                    plain_board[print_row][print_col] = num
                else:
                    plain_board[print_row][print_col] = str(num)[0]
                    plain_board[print_row][print_col + 1] = str(num)[1]
        return plain_board

    def __init__(self, size: int = 3):
        self._size = size
        self._max_round = size**2
        self._round_count = 0
        self._players = ['x', 'o']
        self._x_player = True
        self._simple_board = self.create_board(self._size)
        self._print_board = self.create_print_board(self._size)

    def board(self) -> list:
        return self._simple_board

    def get_print_board(self) -> list:
        return self._print_board

    def round(self) -> int:
        return self._round_count

    def max_round(self) -> int:
        return self._max_round

    def next_player(self) -> bool:
        self._x_player = not self._x_player

    def get_char_player(self) -> str:
        if self._x_player:
            player = self._players[0]
        else:
            player = self._players[1]
        return player

    def read_input(self):
        self.next_player()
        player = self.get_char_player()
        print(f'\nCurrent player is {player}')
        position = input("Choose square:\n")
        row = (int(position)-1) // self._size  # całkowitoliczbowe
        col = int(position) % self._size - 1   # modulo
        if col == -1:
            col = self._size-1
        return row, col

    def print(self):
        print('\n')
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
        board[print_row][print_col + 1] = ' '

    def move(self):
        row, col = self.read_input()
        self._simple_board[row][col] = self.get_char_player()
        self.update_print_board(row, col)
        self.print()
        self._round_count += 1


board = Board(3)
board.print()
while board.round() < board.max_round():
    board.move()




# def input(self) -> int:
#     # przyjęcie inputu od człowieka lub komputera
#     # wyplucie jaką pozycję wybrał
#     # to co się pojawia poprzez current_player
#     # na początku albo na koncu tej funkcji powinno się to zmienić typu True/False albo po prostu jako 'o' / 'x'
#     pass

