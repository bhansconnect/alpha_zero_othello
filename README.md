# Alpha Zero Othello

Requires Tensorflow, Keras, and Pickle. After that, all other libraries should be installed by them.
Will eventually add a requirements.txt

This verion currently uses a smaller neural newtwork for testing purposes as things are confirmed
to be working, I will upgrade to full size.

Currently run by calling "python alpha_zero_othello\run.py {self, opt, eval, play, rank}"

Self for it to generate selfplay games.
Opt for it to update the network.
eval to compare different versions.
play to play against it yourself.
rank to compare multiple models

I generally run 2 instances self playing games and 1 optimizing while training it.
Config.py has all of the options. 