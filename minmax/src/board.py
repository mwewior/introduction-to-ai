class Board:

    def create_board(self, s):
        plain_board = []
        for r in range(s):
            plain_board.append(list(range(r*s+1, (r+1)*s+1)))
        return plain_board

    def create_line(self, symbol) -> list:
        null_line = []
        size = self._size
        for boxes in range(size):
            for _ in range(5):
                null_line.append(symbol)
            null_line.append("|")
        null_line.pop(-1)
        return null_line

    def update_print_board(self, print_board):
        s = self._size
        for row in range(s):
            for col in range(s):
                num = self._board[row][col]
                if num < 10:
                    print_board[row*3 +1][col*(5+1) + 2] = num
                else:
                    print_board[row*3 +1][col*(5+1) + 2 ] = str(num)[0]
                    print_board[row*3 +1][col*(5+1) + 2 + 1] = str(num)[1]
        return print_board

    def create_print_board(self, s) -> list:
        plain_board = []
        for i in range(3*s-1):
            if (i+1)%3 == 0:
                plain_board.append(self.create_line('_'))
            else:
                plain_board.append(self.create_line(' '))
        plain_board.append(self.create_line(' '))
        plain_board = self.update_print_board(plain_board)
        return plain_board

    def __init__(self, size: int = 3) -> None:
        self._size = size
        self._board = self.create_board(self._size)
        self._players = ['x', 'o']
        self._x_player = True
        self._print_board = self.create_print_board(self._size)

    def board(self):
        return self._board

    def get_print_board(self):
        return self._print_board

    def next_player(self):
        self._x_player = not self._x_player

    def input(self) -> int:
        # przyjęcie inputu od człowieka lub komputera
        # wyplucie jaką pozycję wybrał
        # to co się pojawia poprzez current_player
        # na początku albo na koncu tej funkcji powinno się to zmienić typu True/False albo po prostu jako 'o' / 'x'
        pass

    def get_char_player(self):
        if self._x_player:
            player = self._players[0]
        else:
            player = self._players[1]
        return player

    def read_input(self):
        self.next_player()
        player = self.get_char_player()
        print(f'\n\n\nCurrent player is {player}')
        position = input("Choose square:\n")
        row = (int(position)-1) // self._size     # całkowitoliczbowe
        col = int(position) % self._size - 1   # modulo
        if col == -1:
            col = self._size-1
        return row, col

    def print(self):
        board = self.get_print_board()
        for row in board:
            line = ""
            for char in row:
                line += str(char)
            print(line)

    def print_board(self, row, col):
        # s = self._size
        board = self.get_print_board()
        # for row in range(s):
        #     for col in range(s):
        #         num = self._board[row][col]
        #         if num < 10:
        #             board[row*3 +1][col*(5+1) + 2] = str(num)
        #         else:
        #             board[row*3 +1][col*(5+1) + 2 - 1] = str(num)[0]
        #             board[row*3 +1][col*(5+1) + 2 ] = str(num)[1]
        # # return board
        board[row*3 +1][col*(5+1) + 2 ] = self._board[row][col]
        board[row*3 +1][col*(5+1) + 2 + 1] = ' '
        self.print()

    def move(self):
        row, col = self.read_input()
        self._board[row][col] = self.get_char_player()
        self.print_board(row, col)
        # for row in self.board():
        #     for _ in range(self._size):
        #         # row_str = ""
        #         row_str = "|"
        #         for col in row:
        #             row_str += str(col)
        #             row_str += "|"
        #         print(row_str)
        #         print("|-|-|-|")


board = Board(3)
board.print()
while True:
    board.move()
