from alpha_zero_othello.config import EloConfig
from alpha_zero_othello.lib import tf_util, util
from alpha_zero_othello.player.aiplayer import AIPlayer
from alpha_zero_othello.othello import Othello
from time import time
import numpy as np
import random
import glob
import os

def start():
    config = EloConfig()
    tf_util.update_memory(config.gpu_mem_fraction)

    
    models = glob.glob(config.data.model_location+"*.h5")
    if len(models) == 0:
        ai = AIPlayer(1, config.simulation_num_per_move)
        ai.save_weights(config.data.model_location+str(time())+".h5")
    calc_elo(config)
    
def calc_elo(config):
    models = glob.glob(config.data.model_location+"*.h5")
    players = []
    for i, model in enumerate(models):
        if i % config.model_skip == 0 or i == len(models):
            players.append(model)
    
    elo = np.multiply(np.ones((len(players))),1000)
    wtl = np.zeros((len(players), 3))
    game = Othello()
    
    ##give every player a random order to play games against opponents
    order = []
    for i in range(len(players)):
        nums = [x for x in range(len(players))]
        nums.remove(i)
        random.shuffle(nums)
        order.append(nums)
    
    p1 = AIPlayer(1, config.simulation_num_per_move, weights=players[0])
    p2 = AIPlayer(1, config.simulation_num_per_move, weights=players[order[0][0]])
    
    start = time()
    print("Playing random round robin with %d players and %d games per player" % (len(players), config.game_num_per_model))
    for i in range(config.game_num_per_model//2):
        k = config.k_val - (i / config.game_num_per_model//2) * (config.k_val - config.min_k_val)
        util.progress(i, config.game_num_per_model//2, start=start)
        ordering = [x for x in range(len(players))]
        random.shuffle(ordering)
        for j in ordering:
            x = i
            if x >= len(order[j]):
                x %= len(order[j])
                if x == 0:
                    random.shuffle(order[j])
            if x > len(order[j]):
                print(i, x, config.game_num_per_model//2)
            
            p1.load_weights(players[j])
            p2.load_weights(players[order[j][x]])
            
            side = -1
            while not game.game_over():
                if side == -1:
                    t = p1.pick_move(game, side)
                else:
                    t = p2.pick_move(game, side)
                game.play_move(t[0], t[1], side)
                side *= -1
            if game.get_winner() == -1:
                wtl[j,0] += 1
                wtl[order[j][x],2] += 1
                elo[j] += config.k_val*(1-expected(elo[j], elo[order[j][x]]))
                elo[order[j][x]] += config.k_val*(0-expected(elo[order[j][x]], elo[j]))
            elif game.get_winner() == 1:
                wtl[j,2] += 1
                wtl[order[j][x],0] += 1
                elo[j] += config.k_val*(0-expected(elo[j], elo[order[j][x]]))
                elo[order[j][x]] += config.k_val*(1-expected(elo[order[j][x]], elo[j]))
            else:
                wtl[j,1] += 1
                wtl[order[j][x],1] += 1
                elo[j] += config.k_val*(0.5-expected(elo[j], elo[order[j][x]]))
                elo[order[j][x]] += config.k_val*(0.5-expected(elo[order[j][x]], elo[j]))
            if(elo[j] < 0):
                elo[j] = 0
            if(elo[order[j][x]] < 0):
                elo[order[j][x]] = 0
            game.reset_board()
    util.progress(config.game_num_per_model//2, config.game_num_per_model//2, start=start)
    for i, player in enumerate(np.argsort(elo)[::-1]):
        print("%d. %s (expected %d) with %d elo and results of %d-%d-%d"% (i+1, os.path.basename(players[player]), 
            len(players)-player, elo[player], wtl[player,0], wtl[player,1], wtl[player,2]))
            
def expected(ra, rb):
    dif = ra - rb
    denom = 1 + 10**(dif/400)
    return 1/denom