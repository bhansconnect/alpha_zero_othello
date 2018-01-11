from config import RankingConfig
from lib import util
import numpy as np
import choix

def start():
    config = RankingConfig()
    print("Merging Staged Files")
    df = util.mergeStagedWTL(config)
    players = list(df)
    win_matrix = df.as_matrix()
    print("\nTotal Games: ", np.sum(win_matrix))
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