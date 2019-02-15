# Used code from
# DQN implementation by Tycho van der Ouderaa found at
# https://github.com/tychovdo/PacmanDQN.git

# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html


import numpy as np
import random
import util
import time
import sys

from pacman import Directions
from game import Agent
import game

from collections import deque
from PIL import Image, ImageColor

import tensorflow as tf
from DeepQNetwork import *

params = {

    'load_file': None,
    'save_file': None,
    'save_interval' : 1000, 
    'train_start': 5000,    
    'batch_size': 32,      
    'mem_size': 100000,     
    'discount': 0.9,       
    'lr': .00025,            
    'eps': 1.0,             
    'eps_final': 0.1,       
    'eps_step': 10000      
}                     



class PacmanDQN(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        self.qnet = DeepQNetwork(self.params)

        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        self.Q_global = []
        self.cost_disp = 0     

        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

    def getImageFromState(self, state):

        im = Image.new('RGB', (params['height'], params['width'])) 
        for i in range(0,params['height']):
            for j in range(0,params['width']):
                im.putpixel((i,j), ImageColor.getrgb("rgb(0, 0, 0)")) 

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

        if state.getNumAgents()==3:
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

        image = self.getImageFromState(state)

        keepLooking = True

        while keepLooking == True:
            if np.random.rand() > self.params['eps']:
                self.Q_pred = self.qnet.sess.run(
                    self.qnet.y,
                    feed_dict = {self.qnet.x: np.reshape(image,
                                                       (1, params['width'], params['height'], 3)), 
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
                move = self.get_direction(np.random.randint(0, 4))

            self.last_action = self.get_value(move)

            legal = state.getLegalActions(0)
            if move in legal:
                keepLooking = False

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
            self.last_state = self.current_state
            self.current_state = self.getImageFromState(state)

            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 25.    
            elif reward > 0:
                self.last_reward = 5.    
            elif reward < -10:
                self.last_reward = -250.  
                self.won = False
            elif reward < 0:
                self.last_reward = -0.5    

            
            if(self.terminal and self.won):
                self.last_reward = 50.
            self.ep_rew += self.last_reward

            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            if(params['save_file']):
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    self.qnet.save_ckpt('saves/model-' + params['save_file'] + str(self.cnt) + '_' + str(self.numeps) + '/' + params['save_file'])
                    print('Model saved')

            self.train()

        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt)/ float(self.params['eps_step']))


    def observationFunction(self, state):
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        self.ep_rew += self.last_reward

        self.terminal = True
        self.observation_step(state)

        log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(self.params['height'])+'-x-'+str(self.params['num_training'])+'.log','a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()

    def train(self):
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = [] 
            batch_r = [] 
            batch_a = [] 
            batch_n = [] 
            batch_t = [] 

            for i in batch:
                batch_s.append(np.array(i[0]))
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(np.array(i[3]))
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)


    def get_onehot(self, actions):
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def registerInitialState(self, state): 

        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        self.last_state = None
        self.current_state = self.getImageFromState(state)
        self.last_action = None

        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        return move

class smallGridAgent(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']

        self.sess = tf.Session()
        self.new_saver = tf.train.import_meta_graph('saves/smallGridModel/model-smallGridModel16981_1454/smallGridModel.meta')
        self.new_saver.restore(self.sess, tf.train.latest_checkpoint('saves/smallGridModel/model-smallGridModel16981_1454/./'))
        
        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name("input_image:0")
        self.q_t = self.graph.get_tensor_by_name("target_q:0")
        self.actions = self.graph.get_tensor_by_name("actions:0")
        self.rewards = self.graph.get_tensor_by_name("rewards:0")
        self.terminals = self.graph.get_tensor_by_name("terminals:0")
        self.logits = self.graph.get_tensor_by_name("logits/BiasAdd:0")

        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        self.Q_global = []
        self.cost_disp = 0     

        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

    def getImageFromState(self, state):

        im = Image.new('RGB', (params['height'], params['width'])) 
        for i in range(0,params['height']):
            for j in range(0,params['width']):
                im.putpixel((i,j), ImageColor.getrgb("rgb(0, 0, 0)")) 

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

        if len(state.getCapsules()) == 2:
            im.putpixel((int(state.getCapsules()[0][1]), int(state.getCapsules()[0][0])), ImageColor.getrgb("rgb(0, 255, 0)"))
            im.putpixel((int(state.getCapsules()[1][1]), int(state.getCapsules()[1][0])), ImageColor.getrgb("rgb(0, 255, 0)"))
        elif len(state.getCapsules()) == 1:
            im.putpixel((int(state.getCapsules()[0][1]), int(state.getCapsules()[0][0])), ImageColor.getrgb("rgb(0, 255, 0)"))

        return im

    def getMove(self, state):

        image = self.getImageFromState(state)

        self.Q_pred = self.sess.run(
            self.logits,
            feed_dict = {self.x: np.reshape(image,(1, params['width'], params['height'], 3)), 
                         self.q_t: np.zeros(1),
                         self.actions: np.zeros((1, 4)),
                         self.terminals: np.zeros(1),
                            self.rewards: np.zeros(1)})[0]

        self.Q_global.append(max(self.Q_pred))

        a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))
        
        if len(a_winner) > 1:
            move = self.get_direction(
                a_winner[np.random.randint(0, len(a_winner))][0])
        else:
            move = self.get_direction(
                a_winner)

        if move not in state.getLegalActions(0):
            move = state.getLegalActions(0)[np.random.randint(0, len(state.getLegalActions(0)))]


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

    def get_onehot(self, actions):
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def registerInitialState(self, state): 

        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        self.last_state = None
        self.current_state = self.getImageFromState(state)
        self.last_action = None

        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        self.frame = 0
        self.numeps += 1

    def final(self, state):
        self.ep_rew += self.last_reward

        self.terminal = True

    def getAction(self, state):
        move = self.getMove(state)

        return move

class mediumGridAgent(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']

        self.sess = tf.Session()
        self.new_saver = tf.train.import_meta_graph('saves/mediumGridModel/model-mediumGridModel30224_1965/mediumGridModel.meta')
        self.new_saver.restore(self.sess, tf.train.latest_checkpoint('saves/mediumGridModel/model-mediumGridModel30224_1965/./'))
        
        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name("input_image:0")
        self.q_t = self.graph.get_tensor_by_name("target_q:0")
        self.actions = self.graph.get_tensor_by_name("actions:0")
        self.rewards = self.graph.get_tensor_by_name("rewards:0")
        self.terminals = self.graph.get_tensor_by_name("terminals:0")
        self.logits = self.graph.get_tensor_by_name("logits/BiasAdd:0")

        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        self.Q_global = []
        self.cost_disp = 0     

        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

    def getImageFromState(self, state):

        im = Image.new('RGB', (params['height'], params['width'])) 
        for i in range(0,params['height']):
            for j in range(0,params['width']):
                im.putpixel((i,j), ImageColor.getrgb("rgb(0, 0, 0)")) 

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

        if len(state.getCapsules()) == 2:
            im.putpixel((int(state.getCapsules()[0][1]), int(state.getCapsules()[0][0])), ImageColor.getrgb("rgb(0, 255, 0)"))
            im.putpixel((int(state.getCapsules()[1][1]), int(state.getCapsules()[1][0])), ImageColor.getrgb("rgb(0, 255, 0)"))
        elif len(state.getCapsules()) == 1:
            im.putpixel((int(state.getCapsules()[0][1]), int(state.getCapsules()[0][0])), ImageColor.getrgb("rgb(0, 255, 0)"))

        return im

    def getMove(self, state):

        image = self.getImageFromState(state)

        self.Q_pred = self.sess.run(
            self.logits,
            feed_dict = {self.x: np.reshape(image,(1, params['width'], params['height'], 3)), 
                         self.q_t: np.zeros(1),
                         self.actions: np.zeros((1, 4)),
                         self.terminals: np.zeros(1),
                            self.rewards: np.zeros(1)})[0]

        self.Q_global.append(max(self.Q_pred))

        a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))
        
        if len(a_winner) > 1:
            move = self.get_direction(
                a_winner[np.random.randint(0, len(a_winner))][0])
        else:
            move = self.get_direction(
                a_winner)

        if move not in state.getLegalActions(0):
            move = state.getLegalActions(0)[np.random.randint(0, len(state.getLegalActions(0)))]

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

    def final(self, state):
        self.ep_rew += self.last_reward

        self.terminal = True


    def get_onehot(self, actions):
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def registerInitialState(self, state): 
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        self.last_state = None
        self.current_state = self.getImageFromState(state)
        self.last_action = None

        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        return move

class smallClassicAgent(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']

        self.modelNumber = 200000

        self.sess = tf.Session()
        self.new_saver = tf.train.import_meta_graph('saves/smallClassicModel/' + str(self.modelNumber) + '/model.meta')
        self.new_saver.restore(self.sess, tf.train.latest_checkpoint('saves/smallClassicModel/' + str(self.modelNumber) + '/./'))
        
        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name("input_image:0")
        self.q_t = self.graph.get_tensor_by_name("target_q:0")
        self.actions = self.graph.get_tensor_by_name("actions:0")
        self.rewards = self.graph.get_tensor_by_name("rewards:0")
        self.terminals = self.graph.get_tensor_by_name("terminals:0")
        self.logits = self.graph.get_tensor_by_name("logits/BiasAdd:0")

        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        self.Q_global = []
        self.cost_disp = 0     

        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

    def getImageFromState(self, state):

        im = Image.new('RGB', (params['height'], params['width'])) 
        for i in range(0,params['height']):
            for j in range(0,params['width']):
                im.putpixel((i,j), ImageColor.getrgb("rgb(0, 0, 0)")) 

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

        image = self.getImageFromState(state)

        self.Q_pred = self.sess.run(
            self.logits,
            feed_dict = {self.x: np.reshape(image,(1, params['width'], params['height'], 3)), 
                         self.q_t: np.zeros(1),
                         self.actions: np.zeros((1, 4)),
                         self.terminals: np.zeros(1),
                            self.rewards: np.zeros(1)})[0]

        self.Q_global.append(max(self.Q_pred))

        a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))
        
        if len(a_winner) > 1:
            move = self.get_direction(
                a_winner[np.random.randint(0, len(a_winner))][0])
        else:
            move = self.get_direction(
                a_winner)

        if move not in state.getLegalActions(0):
            move = state.getLegalActions(0)[np.random.randint(0, len(state.getLegalActions(0)))]

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

    def final(self, state):
        self.ep_rew += self.last_reward

        self.terminal = True


    def get_onehot(self, actions):
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def registerInitialState(self, state): 

        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        self.last_state = None
        self.current_state = self.getImageFromState(state)
        self.last_action = None

        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        return move