from config import OptimizerConfig
from lib import tf_util, util
from player.aiplayer import AIPlayer
from time import time, sleep
try:
   import cPickle as pickle
except:
   import pickle
import os
import glob

def start():
    config = OptimizerConfig()
    tf_util.update_memory(config.gpu_mem_fraction)
    util.set_high_process_priority()
    AIPlayer.create_if_nonexistant(config)
    models = glob.glob(config.data.model_location+"*.h5")
    ai = AIPlayer(config.buffer_size, 1, model=sorted(models)[-1], compile=True)
    train(ai, config)
    
def train(ai, config):
    loaded_files = []
    file_dif = 0
    x = config.iterations
    i = len(glob.glob(config.data.model_location+"*.h5"))
    while(x != 0):
        x -= 1
        i += 1
        if i > config.iter3:
            ai.update_lr(config.learning_rate3)
        elif i > config.iter2:
            ai.update_lr(config.learning_rate2)
        else:
            ai.update_lr(config.learning_rate1)
        temp = load_games(ai, loaded_files, config)
        file_dif += temp[0]
        last_dif = file_dif
        loaded_files = temp[1]
        start = time()
        util.progress(0, config.min_new_game_files, start=start)
        while(len(loaded_files) < config.min_game_files or file_dif < config.min_new_game_files):
            if last_dif != file_dif:
                last_dif = file_dif
                if config.min_game_files-len(loaded_files) > 0:
                    util.progress(len(loaded_files), config.min_game_files, start=start)
                else:
                    util.progress(file_dif, config.min_new_game_files, start=start)
            sleep(60)
            temp = load_games(ai, loaded_files, config)
            file_dif += temp[0]
            loaded_files = temp[1]
        util.progress(config.min_new_game_files, config.min_new_game_files, start=start)
        file_dif = 0
        print("Iteration %04d"%i)
        print("Training for %d batches on %d samples" % (config.batches_per_iter, len(ai.buffer.buffer)))
        start = time()
        history = ai.train_batches(config.batch_size, config.batches_per_iter, config.verbose)
        for val in history.history.keys():
            print("%s: %0.4f" % (val, history.history[val][-1]))
        if i % config.save_model_cycles == 0:
            ai.save(config.data.model_location+str(time())+".h5")
			
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