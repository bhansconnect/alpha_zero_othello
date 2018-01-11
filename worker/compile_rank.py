from config import RankingConfig
from lib import util
import numpy as np
import choix

def start():
    config = RankingConfig()
    print("Merging Staged Files")
    data = util.mergeStagedWTL(config)
    players = set()
    for elem in data:
        players.add(elem['player1'])
        players.add(elem['player2'])
    players = sorted(players)
    print(players)
    win_matrix = np.zeros((len(players),len(players)))
    for elem in data:
        win_matrix[players.index(elem["player1"]), players.index(elem["player2"])] += elem["wins"] + 0.5*elem["ties"]
        win_matrix[players.index(elem["player2"]), players.index(elem["player1"])] += elem["losses"] + 0.5*elem["ties"]
    print("\nWin Matrix(row beat column):")
    print(win_matrix)
    try:
        params = choix.ilsr_pairwise_dense(win_matrix)
        print("\nRankings:")
        for i, player in enumerate(np.argsort(params)[::-1]):
            print("%d. %s with %0.2f rating"% (i+1, players[player], params[player]))
        print("\n(Rating Diff, Winrate) -> (0.5, 62%), (1, 73%), (2, 88%), (3, 95%), (5, 99%)")
    except Exception:
        print("\nNot Enough data to calculate rankings")