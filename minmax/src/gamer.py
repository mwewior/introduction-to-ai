# from board import Board
# from minmax import MinMaxSolver
from minmax import playGame


WAGES = [
    [3, 2, 3],
    [2, 4, 2],
    [3, 2, 3]
]

PRINT = False
SIZE = 3
MAX_DEPTH = 8
PRUNING = False

for started in [True, False]:
    for d in range(MAX_DEPTH+1):
        print(f'\n\nDepth is {d},\n\'x\' is starting: {started}\n')
        for alphabeta in [False, True]:
            print(f'Alpha-pruning: {alphabeta}')
            winner = playGame(size=SIZE, x_starts=started, depth=d, pruning=alphabeta, printer=False)
            print(f'{winner} won the game!\n')
