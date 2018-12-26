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
from PIL import Image, ImageColor
import random
import numpy as np
  
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

class TestAgent(game.Agent, game.GameStateData):

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction

        im = Image.new('RGB', (11,20)) # create the Image of size 1 pixel 
        for i in range(0,11):
            for j in range(0,20):
                im.putpixel((i,j), ImageColor.getrgb("rgb(0, 0, 0)")) # or whatever color you wish

        currentFood = state.getFood()
        for i in range(0,currentFood.width):
            for j in range(0,currentFood.height):
                if currentFood[i][j] == True:
                    im.putpixel((j,i), ImageColor.getrgb("rgb(255, 255, 255)"))

        walls = state.getWalls()
        for i in range(0,walls.width):
            for j in range(0,walls.height):
                if walls[i][j] == True:
                    im.putpixel((j,i), ImageColor.getrgb("rgb(0, 128, 255)"))

        im.putpixel((state.getPacmanPosition()[1], state.getPacmanPosition()[0]), ImageColor.getrgb("rgb(255, 255, 0)"))

        if state.getGhostState(1).scaredTimer != 0:
            im.putpixel((int(state.getGhostPosition(1)[1]), int(state.getGhostPosition(1)[0])), ImageColor.getrgb("rgb(160, 160, 160)"))
        else:
            im.putpixel((int(state.getGhostPosition(1)[1]), int(state.getGhostPosition(1)[0])), ImageColor.getrgb("rgb(255, 0, 255)"))

        if state.getGhostState(2).scaredTimer != 0:
            im.putpixel((int(state.getGhostPosition(2)[1]), int(state.getGhostPosition(2)[0])), ImageColor.getrgb("rgb(160, 160, 160)"))
        else:
            im.putpixel((int(state.getGhostPosition(2)[1]), int(state.getGhostPosition(2)[0])), ImageColor.getrgb("rgb(255, 0, 255)"))

        if len(state.getCapsules()) == 2:
            im.putpixel((int(state.getCapsules()[0][1]), int(state.getCapsules()[0][0])), ImageColor.getrgb("rgb(0, 255, 0)"))
            im.putpixel((int(state.getCapsules()[1][1]), int(state.getCapsules()[1][0])), ImageColor.getrgb("rgb(0, 255, 0)"))
        elif len(state.getCapsules()) == 1:
            im.putpixel((int(state.getCapsules()[0][1]), int(state.getCapsules()[0][0])), ImageColor.getrgb("rgb(0, 255, 0)"))

        print(state.getGhostState(1).scaredTimer)

        fileNumber = random.randint(0,3000)
        fileNumber = 3;
        im.save('images/image' + str(fileNumber) + '.png')

        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

    def drawState():

        im = Image.new('1', (1,1)) # create the Image of size 1 pixel 
        im.putpixel((0,0), ImageColor.getcolor('black', '1')) # or whatever color you wish
        print('Im here')
        im.show()
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
