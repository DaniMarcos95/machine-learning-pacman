Command to train an agent in a given layout:

	python pacman.py -p PacmanDQN -q -n <Number of games> -x <Number of training games> -l <layout>

Commands for testing over 100 games the trained agent in each layout:

	python pacman.py -p smallGridAgent -n 101 -x 1 -l smallGrid

	python pacman.py -p mediumGridAgent -n 101 -x 1 -l mediumGrid

	python pacman.py -p smallClassicAgent -n 101 -x 1 -l smallClassic

To hide the visual display of the games add "-q" to the command.
