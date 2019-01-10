# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util
import Queue

class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)

def scoreEvaluation(state):
    return state.getScore()
    
#------------ JOOSTCODE --------------#

#OBJECTIVES/Features index:

# 1) number of pills to remaining to eat
# 2) number of pills already eaten
# 3) number of ghost
# 4) distance from ghost
# 5) distance to nearest pill
# 6) safe paths
# 7) distance to big pill/fruit

class TestAgent(game.Agent):
    """ KeyboardAgent that tells you a lot of shit about the game state """
    # NOTE: Arrow keys also work.
    WEST_KEY  = 'a'
    EAST_KEY  = 'd'
    NORTH_KEY = 'w'
    SOUTH_KEY = 's'
    STOP_KEY = 'q'

    def __init__( self, index = 0 ):
        self.lastMove = Directions.STOP
        self.index = index
        self.keys = []
        self.startfood = None #in the very first state, log how much food existed to be able to track how much you ate
        
    def __mazeCrawler(self, state):
        #Apply BFS to the maze to find the nearest food pill
        search = Queue.Queue()
        search.put((int(state.getPacmanPosition()[0]),int(state.getPacmanPosition()[1])))#start state cannot contain pill; checking first is redundant but easier
        hit = []
        explored = []
        while not hit:
            (x,y) = search.get()
            explored.append((x,y))
            print(x,y)
            if state.hasFood(x,y):
                hit.append((x,y))
                print("hit!")
            else:
                for newx,newy in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                    if not state.hasWall(newx,newy) and (newx,newy) not in explored:
                        search.put((newx,newy))
        while not search.empty():
            (x,y) = search.get()
            if state.hasFood(x,y): hit.append((x,y))
        return hit
        
    def __getFeatures(self, state):
        print(state)
        print(vars(state.data))
        features = {}
        features['foodRemaining'] = state.getNumFood()
        if self.startfood == None: self.startfood = state.getNumFood()
        features['foodEaten'] = self.startfood - state.getNumFood()
        features['numGhosts'] = len(state.getGhostStates())
        pacpos = state.getPacmanPosition()
        features['distanceToGhosts'] = [(x-pacpos[0],y-pacpos[1]) for (x,y) in state.getGhostPositions()]
        #TODO: nearest pill requires a breath first search maze crawler, preferably finishing a certain distance
        features['nearestPill'] = self.__mazeCrawler(state)
        #TODO: if maze crawler is done, apply that to ghosts as well, because now it sees through walls
        features['distanceToCapsules'] = [(x-pacpos[0],y-pacpos[1]) for (x,y) in state.getCapsules()]
        
        return features
        

    def getAction( self, state):
        features = self.__getFeatures(state)
        print(features)
        
        from graphicsUtils import keys_waiting
        from graphicsUtils import keys_pressed
        keys = keys_waiting() + keys_pressed()
        if keys != []:
            self.keys = keys

        legal = state.getLegalActions(self.index)
        move = self.getMove(legal)

        if move == Directions.STOP:
            # Try to move in the same direction as before
            if self.lastMove in legal:
                move = self.lastMove

        if (self.STOP_KEY in self.keys) and Directions.STOP in legal: move = Directions.STOP

        if move not in legal:
            move = random.choice(legal)

        self.lastMove = move
        return move

    def getMove(self, legal):
        move = Directions.STOP
        if   (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:  move = Directions.WEST
        if   (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal: move = Directions.EAST
        if   (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:   move = Directions.NORTH
        if   (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
