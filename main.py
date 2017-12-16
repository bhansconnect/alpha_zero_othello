from othello import Othello
from player.player import RandomPlayer
from player.aiplayer import AIPlayer
from time import time

if __name__ == "__main__":
    game = Othello()
    ai = AIPlayer()
    r = RandomPlayer()
    x = 100
    for e in range(0,100):
        print("Run %04d"%e)
        start=time()
        for i in range(x):
            side = -1
            while not game.game_over():
                t = ai.pick_move(game, side)
                game.play_move(t[0], t[1], side)
                side *= -1
            ai.update_buffer(game.get_winner())
            game.reset_board()
        print("Average Game Time: ", (time()-start)/(x))
        ai.train_epoch(epochs=100)
        ai.network.save("models\model_%04d.h5"%e)
        
        #test on random
        wins = 0
        ai.set_training(False)
        for i in range(100):
            side = -1
            while not game.game_over():
                if i % 2 == 0:
                    if side == -1:
                        t = ai.pick_move(game, side)
                    else:
                        t = r.pick_move(game, side)
                    game.play_move(t[0], t[1], side)
                else:
                    if side == 1:
                        t = ai.pick_move(game, side)
                    else:
                        t = r.pick_move(game, side)
                    game.play_move(t[0], t[1], side)
                side *= -1
            if i % 2 == 0:
                if game.get_winner() == -1:
                    wins += 1
            else:
                if game.get_winner() == 1:
                    wins += 1
            game.reset_board()
        ai.set_training(True)
        print("Wins vs Random out of 100: %d"%wins)
        print("Run Time: %0.4f"%(time()-start))
    