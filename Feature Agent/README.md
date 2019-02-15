# PACMAN Feature Agent
The PACMAN feature agents uses a vector of 16 handcrafted features to play different versions of PACMAN, as described in our paper. This readme will describe how to use the code for training and testing agents.

We also recommend just using `python pacman.py --help` once to see all the possible parameters.

# Train an agent
To train an agent you can use the `FeatureAgent`:
```
python pacman.py -p FeatureAgent -n 2100 -x 2000 -l smallClassic -q -s FeatureSmallClassic
```
Here you can vary `-n` to determine how many games are played, `-x` to determine how many of those games are training games (the rest automatically become testing games) and `-s` to name the folder in which the snapshots of the agent's network are stored. If you do not use the `-s` flag the agent will not be saved. The `-q` flag indicates 'quiet graphics'; if you remove this flag the testing games will be displayed live. `-l` determines the layout on which is played, the list of options can be found in layouts folder.

If you are saving an agent, it will create a folder in the 'saves' directory with the indicated name. Inside will be a snapshot of the agent for each 1000 epochs. This repository comes with examples for each of the used layouts. These are the final agents used to generate the results in our paper.

# Test a Trained Agent
To test a trained agent, you can use the same command but instead use the `FeatureTestAgent`. The `-s` flag now determines the snapshot to load, but note that it now also needs a number for which snapshot to use
```
python pacman.py -p FeatureTestAgent -n 101 -x 1 -l smallClassic -q -s saves/FeatureSmallClassicFinal/163000
```
The implementation does allow us to set `-x` to 0, so there's always a leading 'training' game.

You can also designate a csv file to print the scores and winrate over the series of games to. Note that this file must already exist, as the program tries to append the results to the end of the file (this was used to generate results, running different snapshots of the agents making a csv file for each agent type).