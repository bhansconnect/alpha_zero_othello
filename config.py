import os

def project_dir():
    return os.path.dirname(os.path.abspath(__file__))

def data_dir():
    return os.path.join(project_dir(), "data")

class DataConfig:
    game_location = data_dir()+'/games/'
    model_location = data_dir()+'/models/'
    history_location = data_dir()+'/history/'
    performance_location = data_dir()+'/performance/'
    data_location = data_dir()+'/'
    def __init__(self):
        if not os.path.exists(DataConfig.game_location):
            os.makedirs(DataConfig.game_location)
        if not os.path.exists(DataConfig.model_location):
            os.makedirs(DataConfig.model_location)
        if not os.path.exists(DataConfig.history_location):
            os.makedirs(DataConfig.history_location)
        if not os.path.exists(DataConfig.performance_location):
            os.makedirs(DataConfig.performance_location)

class GameConfig:
    simulation_num_per_move = 100
    tau_1 = 1
    tau_2 = 1e-2
    tau_swap = 10 # change to tau_2 after this many moves

class SelfPlayConfig:
    nb_game_in_file = 10
    buffer_size = 64 * nb_game_in_file
    max_file_num = 2000  # 50000
    iterations = -1 #-1 for infinite
    gpu_mem_fraction = 0.249
    data = DataConfig()
    game = GameConfig()


class OptimizerConfig:
    batches_per_iter = 2000
    batch_size = 64
    buffer_size = 1000000
    save_model_cycles = 1
    min_game_files = 100
    min_new_game_files = 50
    iterations = -1 #-1 for infinite
    gpu_mem_fraction = 0.5
    verbose = 0 #0,1,2 like keras model.fit
    learning_rate1 = 1e-3
    iter2 = 50
    learning_rate2 = 3e-4
    iter3 = 100
    learning_rate3 = 1e-4
    data = DataConfig()
    game = GameConfig()

class EvaluateConfig:
    repeat_with_new_model = True
    gpu_mem_fraction = 0.249
    model_1 = "newest" # options: "newest", "random" or model name(no .h5)
    model_2 = "random" # options: "newest", "random" or model name(no .h5)
    data = DataConfig()
    game = GameConfig()
    game_num = 100
    rolling_avg_amount = 10
    
class RankingConfig:
    game_num_per_model = 100 #If you want accurate ratings, you may want this way higher
    gpu_mem_fraction = 0.249
    model_skip = 1 # aka grab every xth model...1 being all models
    print_best = False
    data = DataConfig()
    game = GameConfig()