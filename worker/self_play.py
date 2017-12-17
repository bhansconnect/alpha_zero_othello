from alpha_zero_othello.config import SelfPlayConfig, data_dir
from alpha_zero_othello.lib import tf_util, util
from alpha_zero_othello.player.aiplayer import AIPlayer
from alpha_zero_othello.othello import Othello
from threading import Thread
from time import time
import psutil
import sys
import glob
import os

def start():
    config = SelfPlayConfig()
    tf_util.update_memory(config.gpu_mem_fraction)
    p = psutil.Process(os.getpid())
    if sys.platform == "win32":
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        p.nice(5)

    
    models = glob.glob(config.data.model_location+"*.h5")
    if len(models) == 0:
        ai = AIPlayer(config.buffer_size, config.simulation_num_per_move, config)
        ai.save_weights(config.data.model_location+str(time())+".h5")
    start = time()
    run_games(config)
    print("Total Time: %0.2f seconds" % (time()-start))
             
    
def run_games(config):
    game = Othello()
    model = ""
    x = config.iterations
    i = 0
    while(x != 0):
        x -= 1
        print("Iteration %04d"%i)
        i += 1
        models = sorted(glob.glob(config.data.model_location+"*.h5"))
        if i == 1:
            model = models[-1]
            print("Loading new model: %s" % model)
            ai = AIPlayer(config.buffer_size, config.simulation_num_per_move, weights=model)
        elif models[-1] != model:
            model = models[-1]
            print("Loading new model: %s" % model)
            ai.load_weights(model)
		
        start=time()
        for j in range(config.nb_game_in_file):
            util.progress(j, config.nb_game_in_file, start=start)
            side = -1
            while not game.game_over():
                t = ai.pick_move(game, side)
                game.play_move(t[0], t[1], side)
                side *= -1
            ai.update_buffer(game.get_winner())
            game.reset_board()
        #print("Average Game Time: ", (time()-start)/(config.nb_game_in_file))
        util.progress(config.nb_game_in_file, config.nb_game_in_file, start=start)
        t = Thread(target=save_games, args=(config, ai.buffer))
        t.start()
    t.join()
    
                
def save_games(config, buffer):
    buffer.save(config.data.game_location+str(time())+".pickle")
    games = glob.glob(config.data.game_location+"*.pickle")
    if len(games) > config.max_file_num:
        for file in sorted(games)[:-1*config.max_file_num]:
            os.remove(file)