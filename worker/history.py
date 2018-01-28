from config import DataConfig
import matplotlib.pyplot as plt
try:
   import _pickle as pickle
except:
   import pickle
import glob

def start():
    config = DataConfig()
    histories = sorted(glob.glob(config.history_location+"*.pickle"))
    data = {}
    for hist in histories:
        file = open(hist, 'rb') 
        h = pickle.loads(pickle.load(file))
        for k, v in h.items():
            if k not in data.keys():
                data[k] = []
            for item in v:
                data[k].append(item)
    legend = []
    plt.subplot(2, 1, 1)
    for kv in data.items():
        legend.append(kv[0])
        plt.plot(kv[1])
    plt.legend(legend)
    for i, kv in enumerate(data.items()):
        plt.subplot(2, 3, i+4)
        plt.title(kv[0])
        plt.plot(kv[1])
    plt.tight_layout()
    plt.show()
        