from config import RankingConfig
from lib import util
import numpy as np
import choix
import time
import os

def start():
    config = RankingConfig()
    print("Merging Staged Files")
    df = util.mergeStagedWTL(config)
    if df is None:
        print("Issue loading file, will keep trying")
        print("If this error doesn't go away soon, you may need to delete %s"%
              os.path.normpath(config.data.performance_location+"temp_win_matrix.csv"))
        print("You could also try manually merging it with %s"%
              os.path.normpath(config.data.performance_location+"win_matrix.csv"))
    while df is None:
        time.sleep(0.1)
        df = util.mergeStagedWTL(config)
    players = list(df)
    print("\n",players)
    win_matrix = df.as_matrix()
    print("\nTotal Games: ", np.sum(win_matrix))
    print("\nWin Matrix(row beat column):")
    print(win_matrix)
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            params = choix.ilsr_pairwise_dense(win_matrix)
        print("\nRankings:")
        for i, player in enumerate(np.argsort(params)[::-1]):
            print("%d. %s with %0.2f rating"% (i+1, players[player], params[player]))
        print("\n(Rating Diff, Winrate) -> (0.5, 62%), (1, 73%), (2, 88%), (3, 95%), (5, 99%)")
    except Exception:
        print("\nNot Enough data to calculate rankings")