import sys
from time import time

def progress(count, total, start=0):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    
    if percents == 100:
        if start == 0:
            sys.stdout.write('[%s] %d%%\n' % (bar, percents))
        else:
            elapsed = time() - start
            if elapsed > 3600:
                elapsed /= 3600
                sys.stdout.write('[%s] %d%% total: %0.2f hours\n' % (bar, percents, elapsed))
            elif elapsed > 60:
                elapsed /= 60
                sys.stdout.write('[%s] %d%% total: %0.2f minutes\n' % (bar, percents, elapsed))
            else:
                sys.stdout.write('[%s] %d%% total: %0.2f seconds\n' % (bar, percents, elapsed))
        sys.stdout.flush()
        return
    
    if start == 0 or percents == 0:
        sys.stdout.write('[%s] %d%%\r' % (bar, percents))
    else:
        elapsed = time() - start
        eta = elapsed / (percents/100) - elapsed
        if eta > 3600:
            eta /= 3600
            sys.stdout.write('[%s] %d%% eta: %0.2f hours\r' % (bar, percents, eta))
        elif eta > 60:
            eta /= 60
            sys.stdout.write('[%s] %d%% eta: %0.2f minutes\r' % (bar, percents, eta))
        else:
            sys.stdout.write('[%s] %d%% eta: %0.2f seconds\r' % (bar, percents, eta))
    sys.stdout.flush()