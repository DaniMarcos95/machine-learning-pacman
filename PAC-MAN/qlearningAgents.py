# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from pacman import Directions
from game import Agent
import game
import tensorflow as tf
from featureExtractors import *
from DeepQNetwork import *
import random,util,math
from PIL import Image, ImageColor
import numpy as np
from game import GameStateData
import util              
from collections import deque


params = {
    # Model backups
    'load_file': None,
    'save_file': None,
    'save_interval' : 10000, 

    # Training parameters
    'train_start': 5000,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 100000,     # Replay memory size

    'discount': 0.95,       # Discount rate (gamma value)
    'lr': .0002,            # Learning reate
    # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
    # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.1,       # Epsilon end value
    'eps_step': 10000       # Epsilon steps between start and end (linear)
}   

class DeepQLearningAgent(game.Agent, game.GameStateData):

  def __init__(self, args):

        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']

        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        self.qnet = DeepQNetwork()

        # Q and cost
        self.Q_global = []
        self.cost_disp = 0     


        self.numeps = 0
        self.last_score = 0
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

        self.eps = 1
        self.eps_final = 0.1
        self.eps_step = 1000000
        self.cnt = 0

  def getImageFromState(self, state):

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

        return im
      
  def getMove(self, state):
        # Exploit / Explore
        image = self.getImageFromState(state)
        keepLooking = True

        while(keepLooking):
          if np.random.rand() > self.eps:
              # Exploit action
              self.Q_pred = self.qnet.sess.run(
                  self.qnet.y,
                  feed_dict = {self.qnet.x: np.reshape(image,
                                                       (1, 11, 20, 3)), 
                               self.qnet.q_t: np.zeros(1),
                               self.qnet.actions: np.zeros((1, 4)),
                               self.qnet.terminals: np.zeros(1),
                               self.qnet.rewards: np.zeros(1)})[0]

              self.Q_global.append(max(self.Q_pred))
              a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

              if len(a_winner) > 1:
                  move = self.get_direction(
                      a_winner[np.random.randint(0, len(a_winner))][0])
              else:
                  move = self.get_direction(
                      a_winner[0][0])
          else:
              # Random:
              move = self.get_direction(np.random.randint(0, 4))

          # Save last_action
          self.last_action = self.get_value(move)
          self.eps = max(self.eps_final,
                                 1.00 - float(self.cnt)/ float(self.eps_step))
          self.cnt += 1
          
          legal = state.getLegalActions(0)
          keepLooking = False

          if move not in legal:
              keepLooking = True

        return move

  def get_value(self, direction):
      if direction == Directions.NORTH:
          return 0.
      elif direction == Directions.EAST:
          return 1.
      elif direction == Directions.SOUTH:
          return 2.
      else:
          return 3.

  def get_direction(self, value):
      if value == 0.:
          return Directions.NORTH
      elif value == 1.:
          return Directions.EAST
      elif value == 2.:
          return Directions.SOUTH
      else:
          return Directions.WEST

  def observation_step(self, state):
      if self.last_action is not None:
          # Process current experience state
          self.last_state = np.copy(self.current_state)

          # Process current experience reward
          self.current_score = state.getScore()
          reward = self.current_score - self.last_score
          self.last_score = self.current_score

          if reward > 20:
              self.last_reward = 50.    # Eat ghost   (Yum! Yum!)
          elif reward > 0:
              self.last_reward = 10.    # Eat food    (Yum!)
          elif reward < -10:
              self.last_reward = -500.  # Get eaten   (Ouch!) -500
              self.won = False
          elif reward < 0:
              self.last_reward = -1.    # Punish time (Pff..)

          
          if(self.terminal and self.won):
              self.last_reward = 100.
          self.ep_rew += self.last_reward

          # Store last experience into memory 
          experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
          self.replay_mem.append(experience)
          if len(self.replay_mem) > self.params['mem_size']:
              self.replay_mem.popleft()

          # Save model
          if(params['save_file']):
              if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                  self.qnet.save_ckpt('saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                  print('Model saved')

          # Train
          self.train()

      # Next
      self.local_cnt += 1
      self.frame += 1
      self.params['eps'] = max(self.params['eps_final'],
                               1.00 - float(self.cnt)/ float(self.params['eps_step']))

  def train(self, replay_mem):
      # Train
      #if (self.local_cnt > self.params['train_start']):
      batch = random.sample(replay_mem, 32)
      batch_s = [] # States (s)
      batch_r = [] # Rewards (r)
      batch_a = [] # Actions (a)
      batch_n = [] # Next states (s')
      batch_t = [] # Terminal state (t)

      for i in batch:
          batch_s.append(np.array(self.getImageFromState(i[0])).reshape(11,20,3))
          batch_r.append(i[1])
          batch_a.append(i[2])
          batch_n.append(np.array(self.getImageFromState(i[3])).reshape(11,20,3))
          batch_t.append(i[4])
      batch_s = np.array(batch_s)
      batch_r = np.array(batch_r)
      batch_a = self.get_onehot(np.array(batch_a))
      batch_n = np.array(batch_n)
      batch_t = np.array(batch_t)

      self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)

  def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(self.get_value(actions[i]))] = 1      
        return actions_onehot 

  def getAction(self, state):
      move = self.getMove(state)
      return move

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
