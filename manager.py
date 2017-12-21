import argparse

def start():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="which mode to run in: \"opt\" to train the network," + 
                        " \"self\" to generate selfplay games," + 
                        " \"play\" to play the best network yourself," + 
                        " \"eval\" to evaluate the network against random or a different version" + 
                        " \"rank\" to calculate the ranking of current models." +
                        " or \"hist\" to view the history of training.",
                        choices = ["opt", "self", "play", "eval", "rank", "hist"])
    
    args = parser.parse_args()
    
    if args.mode == "opt":
        from .worker import optimizer
        optimizer.start()
    elif args.mode == "self":
        from .worker import self_play
        self_play.start()
    elif args.mode == "play":
        from .worker import play_game
        play_game.start()
    elif args.mode == "eval":
        from .worker import evaluate
        evaluate.start()
    elif args.mode == "rank":
        from .worker import ranking
        ranking.start()
    elif args.mode == "hist":
        from .worker import history
        history.start()