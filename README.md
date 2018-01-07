# Alpha Zero Othello

A Python program for learning to play Othello/Reversi from zero. The best part is that it is easy
to plug this same algorithm into many other games. If you have any questions. feel free to contact me:
brendan.hansknecht@gmail.com

The program is currently working. I am trying to train and decent network. Then I will work on upgrading
the interfaces such that self play games are not through the command line. I also will work on buffing out
this README and the instructions.

Requires Tensorflow, Keras, and Pickle. After that, all other libraries should be installed by them.
Will eventually add a requirements.txt

This version currently uses a smaller neural network for testing purposes as things are confirmed
to be working, I will upgrade to full size.

Currently run by calling "python alpha_zero_othello\run.py {self, opt, eval, play, rank}"

Self for it to generate self play games.
Opt for it to update the network.
eval to compare different versions.
play to play against it yourself.
rank to compare multiple models

I generally run 2 instances self playing games and 1 optimizing while training it.
Config.py has all of the options. 