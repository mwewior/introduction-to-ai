from board import Board

board = Board(3)
board.print()
while board.round() < board.max_round() and not board.finished():
    board.move()

# if board._x_player:
    # won_player = board._players[0]
# else:
    # won_player = board._players[1]
if board.finished():
    # board.next_player()
    print(f"\n\n'{board.current_player()}' won the game!\n")
else:
    print('\n\nDraw!\n')
