from config import RankingConfig
from lib import tf_util, util
from player.aiplayer import AIPlayer
from othello import Othello
from time import time
import numpy as np
import itertools
import random
import choix
import glob
import math
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
    
    challenger1 = AIPlayer(0, config.game.simulation_num_per_move, train=False, model=players[-1], tau=config.game.tau_1)
    challenger2 = AIPlayer(0, config.game.simulation_num_per_move, train=False, model=players[0], tau=config.game.tau_1)
    total_games = (config.game_num_per_model * (len(players)))//2
    played_games = 0
    finished = False
    start = time()
    print("Ranking with %d players and %d games per player" % (len(players), config.game_num_per_model))
    if config.game_num_per_model < len(players):
        print("We suggest that you increase games per player to be greater than players")
        
    for i in itertools.count():
        ranks = getRankings(win_matrix)

        if len(ranks) == 0 or played_games < config.roundrobin_percent*total_games:
            msg = "No Clear Best Yet"
        else:
            msg = "Current Best is "+util.getPlayerName(players[ranks[-1]])   
        if config.print_ranking_while_running:
            print(msg.ljust(90))
        for j in range(len(players)):
            util.print_progress_bar(played_games, total_games, start=start)
            
            #either challenge people around you with normal distribution or play least played
            if len(ranks) != 0 and played_games > config.roundrobin_percent*total_games:
                my_rank = ranks.index(j)
                challenger1_index = normalChoiceAroundPlayer(ranks, my_rank)
            else:
                challenger1_index = getLeastPlayed(win_matrix, j)
            
            AIPlayer.clear()
            challenger1.load(players[challenger1_index])
            challenger2.load(players[j])
            
            if random.random() < 0.5:
                challenger1_side = -1
                p1 = challenger1
                p2 = challenger2
            else:
                challenger1_side = 1
                p1 = challenger2
                p2 = challenger1
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
            if game.get_winner() == challenger1_side:
                win_matrix[challenger1_index,j] += 1
                wtl[challenger1_index,0] += 1
                wtl[j,2] += 1
            elif game.get_winner() == -1*challenger1_side:
                win_matrix[j, challenger1_index] += 1
                wtl[challenger1_index,2] += 1
                wtl[j,0] += 1
            else:
                win_matrix[challenger1_index,j] += 0.5
                win_matrix[j, challenger1_index] += 0.5
                wtl[challenger1_index,1] += 1
                wtl[j,1] += 1
            game.reset_board()
            played_games += 1
            if played_games >= total_games:
                finished = True
                break
        if finished:
            break
    util.print_progress_bar(total_games, total_games, start=start)   
    print("\nWin Matrix(row beat column):")
    print(win_matrix)
    try:
        params = choix.ilsr_pairwise_dense(win_matrix)
        print("\nRankings:")
        for i, player in enumerate(np.argsort(params)[::-1]):
            print("%d. %s (expected %d) with %0.2f rating and results of %d-%d-%d"% 
                  (i+1, util.getPlayerName(players[player]), len(players)-player, params[player],
                    wtl[player,0], wtl[player,1], wtl[player,2]))
        print("\n(Rating Diff, Winrate) -> (0.5, 62%), (1, 73%), (2, 88%), (3, 95%), (5, 99%)")
    except Exception:
        print("\n Not Enough data to calculate rankings")
        print("\nResults:")
        for player in range(win_matrix.shape[0]):
            print("%s results of %d-%d-%d"% (util.getPlayerName(players[player]), wtl[player,0], wtl[player,1],
                                             wtl[player,2]))
    
def getRankings(win_matrix):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            params = choix.ilsr_pairwise_dense(win_matrix)
        return np.argsort(params).tolist()
    except Exception:
        return []
    
def getLeastPlayed(win_matrix, player):
    min_plays = float("inf")
    least_played_opponent = -1
    for i in range(win_matrix.shape[0]):
        if i == player:
            continue
        plays = win_matrix[i,player] + win_matrix[player,i]
        if least_played_opponent == -1 or plays < min_plays:
            least_played_opponent = i
            min_plays = plays
    return least_played_opponent

def normalChoiceAroundPlayer(ranks, player_rank, reach=None):
    # if mean is the players location if removed from list
    mean = player_rank-0.5
    if mean < 0:
        mean = 0
    ranks_without_self = ranks[:player_rank] + ranks[player_rank+1 :]
    if reach is None:
        reach = len(ranks_without_self)/2
    
    stddev = len(ranks_without_self) * ((reach/len(ranks_without_self))/2)
    
    if mean > len(ranks_without_self)-1:
        mean = len(ranks_without_self)-1
    
    while True:
        index = math.floor(random.gauss(mean, stddev) + 0.5)
        if 0 <= index < len(ranks_without_self):
            return ranks_without_self[index]