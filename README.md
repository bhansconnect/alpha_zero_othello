# Alpha Zero Othello

A Python program for learning to play Othello/Reversi from zero. The best part is that it is easy
to plug this same algorithm into many other games. If you have any questions. feel free to contact me:
brendan.hansknecht@gmail.com

The program is currently working. I am trying to train a decent network. This means that you would need
to train your own network or you would be playing against random initialization. I also will work on buffing out
this README and the instructions.

To install required libraries run: `pip install -r requirements.txt`


This version currently uses a smaller neural network for testing purposes as things are confirmed
to be working, I will upgrade to full size.

Currently run by calling: `python run.py {self, opt, eval, play, rank, compile_rank}`

* self for it to generate self play games.
* opt for it to update the network.
* eval to compare different versions.
* play to play against it yourself.
* rank to compare multiple models
* compile_rank to load all ranking/eval history into one comparison

I generally run 2 instances self playing games and 1 optimizing while training it.
Config.py has all of the options. 

As a side note, it is possible to train on multiple servers by setting up a shared drive between multiple computers
from that shared drive, you can run the program on multiple computers and they will all generate self play games.
That being said, you would still be limited to one optimizer. Luckily, it takes much less time to optimize than
self play games.
