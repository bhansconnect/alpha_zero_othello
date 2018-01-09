from config import RankingConfig
from lib import tf_util, util
from player.aiplayer import AIPlayer
from othello import Othello
from time import time
import numpy as np
import random
import choix
import glob
import os

def start():
    config = RankingConfig()
    tf_util.update_memory(config.gpu_mem_fraction)
    AIPlayer.create_if_nonexistant(config)
    calc_ranking(config)
    
def calc_ranking(config):
    models = sorted(glob.glob(config.data.model_location+"*.h5"))
    players = []
    for i, model in enumerate(models):
        if i % config.model_skip == 0 or i == len(models):
            players.append(model)
    
    wtl = np.zeros((len(players), 3))
    win_matrix = np.zeros((len(players),len(players)))
    game = Othello()
    
    king_index = len(players)-1
    king = AIPlayer(0, config.game.simulation_num_per_move, train=False, model=players[king_index], tau=config.game.tau_1)
    challenger = AIPlayer(0, config.game.simulation_num_per_move, train=False, model=players[0], tau=config.game.tau_1)
    total_games = config.game_num_per_model * len(players)
    
    start = time()
    print("Playing king of the hill with %d players and %d games per player" % (len(players), config.game_num_per_model))
    if config.game_num_per_model < len(players):
        print("We suggest that you increase games per player to be greater than players")
    for i in range(config.game_num_per_model):
        AIPlayer.clear()
        king_index = getKingIndex(win_matrix)
        if king_index == -1:
            king_index = (len(players)-1)-i%len(players)
            msg = "No King Yet"
        else:
            msg = "King is "+os.path.basename(players[king_index]).split(".")[0]
        king.load(players[king_index])
        if config.print_king:
            print(msg.ljust(90))
        for j in range(len(players)):
            util.print_progress_bar(i*len(players) + j, total_games, start=start)
            
            if j == king_index:
                continue
            
            challenger.load(players[j])
            
            if random.random() < 0.5:
                king_side = -1
                p1 = king
                p2 = challenger
            else:
                king_side = 1
                p1 = challenger
                p2 = king
            side = -1
            turn = 1
            while not game.game_over():
                tau = config.game.tau_1
                if config.game.tau_swap < turn:
                    tau = config.game.tau_2
                p1.tau = tau
                p2.tau = tau
                if side == -1:
                    t = p1.pick_move(game, side)
                else:
                    t = p2.pick_move(game, side)
                game.play_move(t[0], t[1], side)
                side *= -1
                turn += 1
            if game.get_winner() == king_side:
                win_matrix[king_index,j] += 1
                wtl[king_index,0] += 1
                wtl[j,2] += 1
            elif game.get_winner() == -1*king_side:
                win_matrix[j, king_index] += 1
                wtl[king_index,2] += 1
                wtl[j,0] += 1
            else:
                win_matrix[king_index,j] += 0.5
                win_matrix[j, king_index] += 0.5
                wtl[king_index,1] += 1
                wtl[j,1] += 1
            game.reset_board()
    util.print_progress_bar(total_games, total_games, start=start)
    try:
        params = choix.ilsr_pairwise_dense(win_matrix)
        print("\nRankings:")
        for i, player in enumerate(np.argsort(params)[::-1]):
            print("%d. %s (expected %d) with %0.2f rating and results of %d-%d-%d"% 
                  (i+1, os.path.basename(players[player]).split(".")[0], len(players)-player, params[player],
                    wtl[player,0], wtl[player,1], wtl[player,2]))
        print("\n(Rating Diff, Winrate) -> (0.5, 62%), (1, 73%), (2, 88%), (3, 95%), (5, 99%)")
    except Exception:
        print("\n Not Enough data to calculate rankings")
        print("\nWin Matrix:")
        print(win_matrix)
        print("\nResults:")
        for player in range(win_matrix.shape[0]):
            print("%s results of %d-%d-%d"% (os.path.basename(players[player]).split(".")[0], wtl[player,0], wtl[player,1],
                                             wtl[player,2]))
    
def getKingIndex(win_matrix):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            params = choix.ilsr_pairwise_dense(win_matrix)
        return np.argsort(params)[-1]
    except Exception:
        return -1