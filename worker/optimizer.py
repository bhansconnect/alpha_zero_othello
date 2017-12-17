from alpha_zero_othello.config import OptimizerConfig, data_dir
from alpha_zero_othello.lib import tf_util
from alpha_zero_othello.player.aiplayer import AIPlayer
from time import time, sleep
try:
   import cPickle as pickle
except:
   import pickle
import psutil
import os
import sys
import glob


def start():
    config = OptimizerConfig()
    tf_util.update_memory(config.gpu_mem_fraction)
    p = psutil.Process(os.getpid())
    if sys.platform == "win32":
        p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    else:
        p.nice(-5)
    
    models = glob.glob(config.data.model_location+"*.h5")
    if len(models) == 0:
        ai = AIPlayer(config.buffer_size, 1, config)
        ai.save_weights(config.data.model_location+str(time())+".h5")
    else:
        ai = AIPlayer(config.buffer_size, 1, config, weights=sorted(models)[-1])
        
    start = time()
    train(ai, config)
    print("Total Time: %0.2f seconds" % (time()-start))
    
def train(ai, config):
    loaded_files = []
    file_dif = 0
    x = config.iterations
    i = 0
    while(x != 0):
        x -= 1
        i += 1
        temp = load_games(ai, loaded_files, config)
        file_dif += temp[0]
        last_dif = file_dif
        loaded_files = temp[1]
        while(len(loaded_files) < config.min_game_files or file_dif < config.min_new_game_files):
            if last_dif != file_dif:
                last_dif = file_dif
                print("Waiting on %d more files" %
                      (max(config.min_game_files-len(loaded_files), config.min_new_game_files-file_dif)))
            sleep(60)
            temp = load_games(ai, loaded_files, config)
            file_dif += temp[0]
            loaded_files = temp[1]
        file_dif = 0
        print("Iteration %04d"%i)
        print("Training for %d epochs on %d samples" % (config.epochs_per_cycle, len(ai.buffer.buffer)))
        start = time()
        history = ai.train_epoch(config.batch_size, 1+2000//(len(ai.buffer.buffer)//config.batch_size), config.verbose)
        for val in history.history.keys():
            print("%s: %0.4f" % (val, history.history[val][-1]))
        if i % config.save_model_cycles == 0:
            ai.save_weights(config.data.model_location+str(time())+".h5")
			
        file = open(config.data.history_location+str(time())+".pickle", 'wb') 
        pickle.dump(pickle.dumps(history.history), file)
        file.close() 
        print("Iteration Time: %0.2f" % (time()-start))

def load_games(ai, loaded_files, config):
    games = glob.glob(config.data.game_location+"*.pickle")
    new = [game for game in games if game not in loaded_files]
    dif = len(new)
    for game in sorted(new):
        ai.buffer.load(game)
    return dif, games