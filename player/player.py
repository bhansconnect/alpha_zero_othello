from abc import ABC, abstractmethod
from othello import Othello
import random

class Player(ABC):
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def pick_move(self, game):
        pass
    
class RandomPlayer(Player):
    
    def pick_move(self, game, side):
        t = game.possible_moves(side)
        if len(t) == 0:
            return (-1, -1)
        r = random.randint(0, len(t)-1)
        return (t[r][0], t[r][1])
    
class HumanPlayer(Player): 
    def pick_move(self, game, side):
        print("You are playing", Othello.piece_map(side))
        t = game.possible_moves(side)
        if len(t) == 0:
            game.print_board()
            print("No moves availible. Turn skipped.")
            return (-1, -1)
        move = (-1, -1)
        while move not in t:
            try:
                row = int(input("Please input row: "))
                col = int(input("Please input col: "))
                move = (row, col)
                if move not in t:
                    game.print_board()
                    print("Please input a valid move")
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                game.print_board()
                print("Please input a valid move")
        print()
        return move
            