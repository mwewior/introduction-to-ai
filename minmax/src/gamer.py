from board import Board

board = Board(3)
board.print()
while board.round() < board.max_round() and not board.finished():
    winner = None
    state = board.board
    row, col = board.read_input()
    state[row][col] = board.current_player()

    board.update_print_board(row, col)
    board.print()

    board._round_count += 1
    board.next_player()

    if board.round() >= 2*board.size()-1:
        _, winner = board.check_win()
    board.who_wins(winner)
    # if winner is not None:
    #     print(f"\n\n'{winner}' won the game!\n")
    # if board.round() == board.max_round():
    #     print('\n\nDraw!\n')
