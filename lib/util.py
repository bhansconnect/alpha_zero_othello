import sys
from time import time
import pandas as pd
import psutil
import glob
import os
try:
   import cPickle as pickle
except:
   import pickle

def print_progress_bar(count, total, start=0):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    
    if percents == 100:
        if start == 0:
            msg = '[%s] %d%%\n' % (bar, percents)
        else:
            elapsed = time() - start
            if elapsed > 3600:
                elapsed /= 3600
                msg = '[%s] %d%% total: %0.2f hours\n' % (bar, percents, elapsed)
            elif elapsed > 60:
                elapsed /= 60
                msg = '[%s] %d%% total: %0.2f minutes\n' % (bar, percents, elapsed)
            else:
                msg = '[%s] %d%% total: %0.2f seconds\n' % (bar, percents, elapsed)
    elif start == 0 or percents == 0:
        msg = '[%s] %d%%\r' % (bar, percents)
    else:
        elapsed = time() - start
        eta = elapsed / (percents/100) - elapsed
        if eta > 3600:
            eta /= 3600
            msg = '[%s] %d%% eta: %0.2f hours   \r' % (bar, percents, eta)
        elif eta > 60:
            eta /= 60
            msg = '[%s] %d%% eta: %0.2f minutes \r' % (bar, percents, eta)
        else:
            msg = '[%s] %d%% eta: %0.2f seconds \r' % (bar, percents, eta)
    sys.stdout.write(msg[:-1].ljust(89)+msg[-1])
    sys.stdout.flush()
    
def set_low_process_priority():
    p = psutil.Process(os.getpid())
    if sys.platform == "win32":
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        p.nice(5)

def set_high_process_priority():
    p = psutil.Process(os.getpid())
    if sys.platform == "win32":
        p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    else:
        p.nice(-5)
        
def getPlayerName(player):
    return os.path.basename(player).split(".")[0]

def saveWTL(config, p1, p2, w, t, l):
    if w > 0 or t > 0 or l > 0:
        data = {
            "player1": p1,
            "player2": p2,
            "wins": w,
            "ties": t,
            "losses": l}
        pickle.dump(data, open(config.data.performance_location+"staged_"+str(time())+".pickle","wb"))

def mergeStagedWTL(config):
    merged_data = []
    files = glob.glob(config.data.performance_location+"staged_*.pickle")
    for file in files:
        try:
            data = pickle.load(open(file, "rb"))
            found = False
            for i in range(len(merged_data)):
                if merged_data[i]['player1'] == data['player1'] and merged_data[i]['player2'] == data['player2']:
                    found = True
                    merged_data[i]['wins'] += data['wins']
                    merged_data[i]['ties'] += data['ties']
                    merged_data[i]['losses'] += data['losses']
                    break
                elif merged_data[i]['player1'] == data['player2'] and merged_data[i]['player2'] == data['player1']:
                    found = True
                    merged_data[i]['wins'] += data['losses']
                    merged_data[i]['ties'] += data['ties']
                    merged_data[i]['losses'] += data['wins']
                    break
            if not found:
                merged_data.append(data)
            os.remove(file)
        except Exception as e:
            continue
    
    if os.path.isfile(config.data.performance_location+"win_matrix.csv"):
        df = pd.read_csv(config.data.performance_location+"win_matrix.csv", index_col=0)
    else:
        df = pd.DataFrame()
    for elem in merged_data:
        if not elem["player1"] in list(df):
            df[elem["player1"]] = 0
            df.loc[elem["player1"]] = 0
            df = df.sort_index(axis=0).sort_index(axis=1)
        if not elem["player2"] in list(df):
            df[elem["player2"]] = 0
            df.loc[elem["player2"]] = 0
            df = df.sort_index(axis=0).sort_index(axis=1)
        df.at[elem["player1"], elem["player2"]] = df.at[elem["player1"], elem["player2"]] + elem["wins"] + 0.5*elem["ties"]
        df.at[elem["player2"], elem["player1"]] = df.at[elem["player2"], elem["player1"]] + elem["losses"] + 0.5*elem["ties"]
    df.to_csv(config.data.performance_location+"win_matrix.csv")
    return df