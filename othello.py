import numpy as np

class Othello(object):
    
    def __init__(self):
        self.reset_board();
        
    def reset_board(self):
        self.board = np.zeros((8, 8), dtype=np.int)
        self.board[3, 3] = 1
        self.board[3, 4] = -1
        self.board[4, 3] = -1
        self.board[4, 4] = 1
      
    def play_move(self, x, y, side):
        if x == -1 and y == -1:
            return
        self.board[x,y] = side
        self.flip(x, y, side)
        
    def game_over(self):
        for i in range(8):
            for j in range(8):
                if self.board[i,j] == 0 and (self.valid_flip(i,j, -1) or self.valid_flip(i,j, 1)):
                    return False
        return True
    
    def get_winner(self):
        t = np.sum(self.board)
        if t > 0:
            return 1
        if t < 0:
            return -1
        return 0
    
    def possible_moves(self, side):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i,j] == 0 and self.valid_flip(i,j, side):
                    moves.append((i, j))
        return moves
    
    def valid_flip(self, x, y, side):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                if(self.valid_ray(x, y, side, dx, dy)):
                    return True
        return False
    
    def valid_ray(self, x, y, side, dx, dy):
        tx = x + 2*dx
        if tx < 0 or tx > 7:
            return False
        ty = y + 2*dy
        if ty < 0 or ty > 7:
            return False
        if self.board[x+dx, y+dy] != -1*side:
            return False
        while self.board[tx, ty] != side:
            if self.board[tx, ty] == 0:
                return False
            tx += dx
            ty += dy
            if tx < 0 or tx > 7:
                return False
            if ty < 0 or ty > 7:
                return False
        return True
    
    def flip(self, x, y, side):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                if(self.valid_ray(x, y, side, dx, dy)):
                    self.flip_ray(x, y, side, dx, dy)
    
    def flip_ray(self, x, y, side, dx, dy):
        tx = x + dx
        ty = y + dy
        while self.board[tx, ty] != side:
            self.board[tx, ty] = side
            tx += dx
            ty += dy
        
    def print_board(self):
        print("   ", end="")
        for i in range(8):
            print("%2d" % (i) , end="")
        print("\n   ", end="")
        for _ in range(16):
            print("-", end="")
        print("-")
        for i in range(8):
            print("%2d " % (i) , end="")
            for j in range(8):
                print("|" + Othello.piece_map(self.board[i,j]), end="")
            print("|")
            print("   ", end="")
            for _ in range(16):
                print("-", end="")
            print("-")
    
    @staticmethod
    def piece_map(x):
        return {
            1: 'W',
            -1: 'B',
            0: ' ',
        }[x]
        
    @staticmethod
    def move_id(move):
        if move == (-1,-1):
                return 64
        return move[0]+move[1]*8
    
    move_count = 65
    
    def get_move(mid):
        if mid == 64:
            return (-1, -1)
        x = mid%8
        y = mid//8
        return (x, y)
           
    @staticmethod
    def state_id(board):
        x = np.add(board, 1).flatten()
        id = 0
        mult = 1
        for t in x:
            id += mult*int(t)
            mult *= 3
        return id
        