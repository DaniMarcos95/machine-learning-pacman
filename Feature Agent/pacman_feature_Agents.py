from pacman import Directions
from game import Agent
import random, time, util, sys
import game
import queue
from collections import deque

import numpy as np
import tensorflow as tf

import os
 
params = {
    # Model backups
    'load_file': None,
    # 'save_file': 'featureModelX',
    'save_interval' : 1000, #train steps between saves

    # Training parameters
    'train_start': 1000,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 100000,     # Replay memory size

    'discount': 0.8,       # Discount rate (gamma value)
    'lr': .0005,            # Learning reate
    
    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.01,       # Epsilon end value
    'eps_step': 10000        # Epsilon steps between start and end (linear) (for smallGrid ~10000, for smallClassic ~20000)
}    
 
class DeepQNetwork:

    def __init__(self, options):

        print("Initialising MLP")
        
        self.params = params
        self.network_name = 'DeepQNetwork'
        self.sess = tf.Session()
        self.x = tf.placeholder('float', [None, params['num_features']], name='x')
        self.q_t = tf.placeholder('float', [None], name='target_q')
        self.actions = tf.placeholder("float", [None, 4], name='actions')
        self.rewards = tf.placeholder("float", [None], name='rewards')
        self.terminals = tf.placeholder("float", [None], name='terminals')    

        dense = tf.layers.dense(inputs=self.x, units=32, activation=tf.nn.sigmoid)
        dense2 = tf.layers.dense(inputs=dense, units=16, activation=tf.nn.sigmoid)
        logits = tf.layers.dense(inputs=dense2, units=4, name='logits')

        self.y = logits
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.discount = tf.constant(self.params['discount'])

        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1)
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))
        self.optim = tf.train.GradientDescentOptimizer(params['lr']).minimize(self.cost, global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=0)

        self.sess.run(tf.global_variables_initializer())

    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):
        feed_dict={self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        q_t = self.sess.run(self.y,feed_dict=feed_dict)
        q_t = np.amax(q_t, axis=1)
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        _,cnt,cost = self.sess.run([self.optim, self.global_step,self.cost],feed_dict=feed_dict)
        return cnt, cost
            
    def save_ckpt(self,filename):
        self.saver.save(self.sess, filename)

#------------ JOOSTCODE --------------#

class FeatureAgent(game.Agent):
    """ Agent that uses the game state to create a feature vector and an MLP for simple learning"""
        
    def __init__(self, args):

        print("Initialise feature Agent")

        # Load parameters from user-given arguments
        self.params = params
        self.params['num_training'] = args['numTraining']
        self.params['num_features'] = 16

        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        self.qnet = DeepQNetwork(self.params)

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        
        # Q and cost
        self.cost_disp = 0     

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()
        
        #feature helpers
        self.startfood = None #in the very first state, log how much food existed to be able to track how much you ate
        self.lastMove = Directions.EAST
        
    def hasGhost(self, state, x, y):
        return (x,y) in state.getGhostPositions()
        
    def hasCapsule(self, state, x, y):
        return (x,y) in state.getCapsules()
        
    def mazeCrawler(self, state, pacx, pacy):
        #Apply BFS to the maze to find the items of interest
        search = queue.Queue()
        for newx,newy,dir,dist in [(pacx,pacy+1,self.get_value(Directions.NORTH),1),#for each direction keep track of what starting action gets you there
                            (pacx+1,pacy,self.get_value(Directions.EAST),1), #because of this implementation, the agent is biased to got north, then east, etc.
                            (pacx,pacy-1,self.get_value(Directions.SOUTH),1),
                            (pacx-1,pacy,self.get_value(Directions.WEST),1)]:
            if not state.hasWall(newx,newy):
                    search.put((newx,newy,dir,dist))
        ghosts = []
        pills = []
        capsules = []
        explored = []
        while not search.empty():
            (x,y,dir,dist) = search.get(block=False)
            explored.append((x,y))
            if state.hasFood(x,y):
                pills.append((x,y,dir,dist))
                # print("Foodhit: %s"%str((x,y,d)))
            if FeatureAgent.hasGhost(self,state,x,y):
                ghosts.append((x,y,dir,dist))
                # print("Ghosthit: %s"%str((x,y,d)))
            if FeatureAgent.hasCapsule(self,state,x,y):
                capsules.append((x,y,dir,dist))
            for newx,newy,dir in [(x-1,y,dir),(x+1,y,dir),(x,y-1,dir),(x,y+1,dir)]:
                # print(newx,newy)
                if not state.hasWall(newx,newy) and (newx,newy) not in explored:
                    search.put((newx,newy,dir,dist+1))
                    
        #some states exist where no pills exist (if won) or no ghosts (???), workaround to prevent index out of range
        if ghosts == []:
            ghosts = [(0,0,0,0)]
        if pills == []:
            pills = [(0,0,0,0)]
        if capsules == []:
            capsules = [(0,0,0,0)]
        return (ghosts,pills,capsules)
        
    def getFeatures(self, state):
        # print(vars(state.data))
        features = {}
        features['foodRemaining'] = state.getNumFood()
        if self.startfood == None: self.startfood = state.getNumFood()
        features['foodEaten'] = self.startfood - state.getNumFood()
        features['numGhosts'] = len(state.getGhostStates())
        pacx, pacy = state.getPacmanPosition()
        features['pacmanPosition'] = (pacx, pacy)
        features['distanceToGhosts'] = [(x-pacx,y-pacy) for (x,y) in state.getGhostPositions()]
        ghosts, pills, capsules = FeatureAgent.mazeCrawler(self, state, pacx, pacy)
        features['nearestGhostDirection'] = ghosts[0][2]
        features['nearestPill'] = (pills[0][0]-pacx,pills[0][1]-pacy)
        features['nearestPillDirection'] = pills[0][2]
        features['capsulesRemaining'] = len(state.getCapsules())
        features['distanceToCapsules'] = [(x-pacx,y-pacy) for (x,y) in state.getCapsules()]
        features['canEat'] = [g.scaredTimer for g in state.getGhostStates()]
        features['northWall'] = state.hasWall(pacx,pacy+1)
        features['eastWall'] = state.hasWall(pacx+1,pacy)
        features['southWall'] = state.hasWall(pacx,pacy-1)
        features['westWall'] = state.hasWall(pacx-1,pacy)
        features['northGhost'] = (pacx,pacy+1) in state.getGhostPositions()
        features['eastGhost'] = (pacx+1,pacy) in state.getGhostPositions()
        features['southGhost'] = (pacx,pacy-1) in state.getGhostPositions()
        features['westGhost'] = (pacx-1,pacy) in state.getGhostPositions()
        features['lastMove'] = self.get_value(self.lastMove)
        
        #Conform features to a list
        features['distanceToGhosts'].extend([(0,0)] * (4-features['numGhosts']))#ASSUMPTION: max 4 ghosts
        features['canEat'].extend([0] * (4-features['numGhosts']))#ASSUMPTION: max 4 ghosts
        features['distanceToCapsules'].extend([(0,0)] * (2-features['capsulesRemaining']))#ASSUMPTION: max 2 capsules
        features = [
            # features['foodRemaining'],
            # features['foodEaten'],
            # features['numGhosts'],
            # pacx,                               #Pacman's X
            # pacy,                               #Pacman's Y
            # features['distanceToGhosts'][0][0], #X to ghost 1
            # features['distanceToGhosts'][0][1], #Y to ghost 1
            # features['distanceToGhosts'][1][0], #X to ghost 2
            # features['distanceToGhosts'][1][1], #Y to ghost 2
            # features['distanceToGhosts'][2][0], #X to ghost 3
            # features['distanceToGhosts'][2][1], #Y to ghost 3
            # features['distanceToGhosts'][3][0], #X to ghost 4
            # features['distanceToGhosts'][3][1], #Y to ghost 4
            features['nearestGhostDirection'],  #path direction of nearest ghost
            -ghosts[0][3],                       #path distance to nearest ghost, inverted
            features['canEat'][0],              #can eat ghost 1
            features['canEat'][1],              #can eat ghost 2
            # features['canEat'][2],              #can eat ghost 3
            # features['canEat'][3],              #can eat ghost 4
            # features['nearestPill'][0],         #X to nearest pill
            # features['nearestPill'][1],         #Y to nearest pill
            features['nearestPillDirection'],   #direction of path to nearest pill
            -pills[0][3],                        #path distance to nearest pill, inverted
            capsules[0][2],                       #direction of nearest capsule
            -capsules[0][3],                      #distance to nearest capsule, inverted
            # features['capsulesRemaining'],      #remaining capsules
            # features['distanceToCapsules'][0][0],#X to capsule 1
            # features['distanceToCapsules'][0][1],#Y to capsule 1
            # features['distanceToCapsules'][1][0],#X to capsule 2
            # features['distanceToCapsules'][1][1],#Y to capsule 2
            features['northWall'],              #is there a wall above pacman    
            features['eastWall'],               #is there a wall right of pacman    
            features['southWall'],              #is there a wall below pacman    
            features['westWall'],               #is there a wall left of pacman  
            features['northGhost'],             #is there a ghost above pacman
            features['eastGhost'],              #is there a ghost right of pacman
            features['southGhost'],             #is there a ghost below pacman
            features['westGhost'],              #is there a ghost left of pacman
            # features['lastMove'],               #last action
        ]
        # print(state)
        # print(features)
        # raise(Exception)
        return features
        
    def getMove(self, state):

        features = self.getFeatures(state)
        
        # Exploit / Explore
        if np.random.rand() > self.params['eps']:
            # Exploit action
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(features, (1,self.params['num_features'])),
                             self.qnet.q_t: np.zeros(1),
                             self.qnet.actions: np.zeros((1, 4)),
                             self.qnet.terminals: np.zeros(1),
                             self.qnet.rewards: np.zeros(1)})[0]

            self.Q_global.append(np.max(self.Q_pred))
            sorted_actions = [self.get_direction(a) for a in np.flip(np.argsort(self.Q_pred),axis=0)]
            # print(sorted_actions)
            # print(self.Q_pred)
            for a_winner in sorted_actions:
                if a_winner in state.getLegalActions(0):
                    move = a_winner
                    break
        else:
            # Random:
            keepLooking = True
            while keepLooking == True:
                move = self.get_direction(np.random.randint(0, 4))
                if move in state.getLegalActions(0):
                    keepLooking = False
        
        # print(move)
        # print("done")
        # Save last_action
        self.last_action = self.get_value(move)
        
        self.lastMove = move
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
            self.last_state = self.current_state
            self.current_state = self.getFeatures(state)

            # Process current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.    # Eat ghost   (Yum! Yum!)
            elif reward > 0:
                self.last_reward = 10.    # Eat food    (Yum!)
            elif reward < -10:
                self.last_reward = -50.   # Get eaten   (Ouch!) -500
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
            if(self.savename):
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    dir = 'saves/%s/%d'%(self.savename,self.local_cnt)
                    if not os.path.exists(dir): os.makedirs(dir)
                    self.qnet.save_ckpt('%s/model'%(dir))
                    print('Model saved')

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt)/ float(self.params['eps_step']))


    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Print stats
        log_file = open('./logs/'+str(self.general_record_time)+'-x-'+str(self.params['num_training'])+'.log','a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()

    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = [] # States (s)
            batch_r = [] # Rewards (r)
            batch_a = [] # Actions (a)
            batch_n = [] # Next states (s')
            batch_t = [] # Terminal state (t)

            for i in batch:
                # print(i)
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
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getFeatures(state)
        # Reset actions
        self.last_action = None
        
        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        return move
        
class FeatureTestAgent(game.Agent):
    def __init__(self, args):

        loadname = args['savename']
        print("Initialise Feature Test Agent on model: %s"%loadname)
        
        # Load parameters from user-given arguments
        self.params = params
        self.params['num_training'] = args['numTraining']
        self.params['num_features'] = 16

        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']
        
        self.sess = tf.Session()
        self.new_saver = tf.train.import_meta_graph(loadname + '/model.meta')#designate model here
        self.new_saver.restore(self.sess, tf.train.latest_checkpoint(loadname + './'))
        
        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name("x:0")
        self.q_t = self.graph.get_tensor_by_name("target_q:0")
        self.actions = self.graph.get_tensor_by_name("actions:0")
        self.rewards = self.graph.get_tensor_by_name("rewards:0")
        self.terminals = self.graph.get_tensor_by_name("terminals:0")
        self.logits = self.graph.get_tensor_by_name("logits/BiasAdd:0")

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = []
        self.cost_disp = 0     

        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()
        
        #feature helpers
        self.startfood = None #in the very first state, log how much food existed to be able to track how much you ate
        self.lastMove = Directions.EAST
            
    def getMove(self, state):

        features = FeatureAgent.getFeatures(self, state)
        
        self.Q_pred = self.sess.run(
            self.logits,
            feed_dict = {self.x: np.reshape(features, (1,self.params['num_features'])),
                         self.q_t: np.zeros(1),
                         self.actions: np.zeros((1, 4)),
                         self.terminals: np.zeros(1),
                         self.rewards: np.zeros(1)})[0]

        self.Q_global.append(np.max(self.Q_pred))
        sorted_actions = [self.get_direction(a) for a in np.flip(np.argsort(self.Q_pred),axis=0)]
        # print(sorted_actions)
        # print(self.Q_pred)
        for a_winner in sorted_actions:
            if a_winner in state.getLegalActions(0):
                move = a_winner
                break
        
        # print(move)
        # print("done")
        # Save last_action
        self.last_action = self.get_value(move)
        
        self.lastMove = move
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
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True

    def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = FeatureAgent.getFeatures(self, state)
        # Reset actions
        self.last_action = None
        
        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        return move