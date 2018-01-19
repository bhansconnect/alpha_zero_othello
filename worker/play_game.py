from config import EvaluateConfig
from lib import tf_util
from player.player import RandomPlayer
from player.aiplayer import AIPlayer
from othello import Othello
from random import random
from tkinter import Tk, Canvas
import threading
import time
import glob

def start():
    root = Tk()
    AppLogic(root)
    root.mainloop()

class AppLogic(threading.Thread):

    def __init__(self, tk_root):
        self.root = tk_root
        threading.Thread.__init__(self)
        self.turn = 0
        self.update = False
        self.x = -1
        self.y = -1
        self.start()
    
    def run(self):
        self.game_gui = Canvas(self.root, width=600, height=600, background='green')
        self.game_gui.bind("<Button-1>", self.click)
        self.game_gui.focus_set()
        self.game_gui.bind("<Key>", self.key)
        self.game_gui.pack()
        for i in range(1, 8):
            self.game_gui.create_line(0, i*75, 600, i*75)
            self.game_gui.create_line(i*75, 0, i*75, 600)
        
        self.pieces = []
        for i in range(8):
            self.pieces.append([])
            for j in range(8):
                self.pieces[i].append(self.game_gui.create_oval(i*75+5, j*75+5, (i+1)*75-5, (j+1)*75-5, fill="green", outline="green"))
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.resizable(0,0)
        self.running = True
        config = EvaluateConfig()
        tf_util.update_memory(config.gpu_mem_fraction)
        AIPlayer.create_if_nonexistant(config)
        self.game = Othello()
        if(random() > 0.5):
            self.human = 1
        else:
            self.human = -1
        
        ai = create_player(config.model_1, config)
        #print("You are playing against", config.model_1)
        #print("Playing games with %d simulations per move" % config.game.simulation_num_per_move)
        self.side = -1
        self.draw_board()
        self.value = ai.evaluate(self.game, self.side)
        while self.running and not self.game.game_over():
            #play move
            if self.side != self.human:
                self.value = ai.evaluate(self.game, self.side)
                self.root.title("Othello (Thinking of Move) Current Value: %0.2f (1 white wins, -1 black wins)" % self.value)
                self.root.config(cursor="wait")
                t = ai.pick_move(self.game, self.side)
                self.game.play_move(t[0], t[1], self.side)
                self.draw_board()
                self.side *= -1
                self.value = ai.evaluate(self.game, self.side)
            else:
                if len(self.game.possible_moves(self.side)) == 0:
                    self.side *= -1
                    continue
                if self.side == -1:
                    color = "black"
                else:
                    color = "white"
                self.root.title("Othello (Play as %s) Current Value: %0.2f (1 white wins, -1 black wins)" % (color, self.value))
                self.root.config(cursor="")
                if self.update:
                    self.update = False
                    if (self.x, self.y) in self.game.possible_moves(self.side):
                        self.game.play_move(self.x, self.y, self.side)
                        self.draw_board()
                        self.side *= -1
            time.sleep(0.01)
        self.root.config(cursor="")
        if self.human == self.game.get_winner():
            self.root.title("Othello (You Win!)")
        elif self.game.get_winner() == 0:
            self.root.title("Othello (Its a draw!)")
        else:
            self.root.title("Othello (You Lose!)")

    def key(self, event):
        if event.char == "z":
            self.human *= -1

    def click(self, event):
        self.game_gui.focus_set()
        if self.human == self.side and not self.update:
            if self.x != event.x//75 or self.y != event.y//75:
                self.update = True
                self.x = event.x//75
                self.y = event.y//75
    
    def on_closing(self):
        self.running = False
        self.root.destroy()

    def draw_board(self):    
        for i in range(8):
            for j in range(8):
                if self.game.board[i, j] == 1:
                    self.game_gui.itemconfig(self.pieces[i][j], fill="white")
                if self.game.board[i, j] == -1:
                    self.game_gui.itemconfig(self.pieces[i][j], fill="black")

def create_player(player_name, config):
    if player_name == "random":
        model = "random"
        player = RandomPlayer()
    elif player_name == "newest":
        model = sorted(glob.glob(config.data.model_location+"*.h5"))[-1]
        #print("Loading model: %s" % model)
        player = AIPlayer(0, config.game.simulation_num_per_move, train=False, model=model, tau=config.game.tau_1)
    else:
        model = config.data.model_location+player_name
        player = AIPlayer(0, config.game.simulation_num_per_move, train=False, model=model, tau=config.game.tau_1)
    return player