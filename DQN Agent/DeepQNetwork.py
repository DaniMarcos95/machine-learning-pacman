import numpy as np
import tensorflow as tf
    
class DeepQNetwork:

    def __init__(self, params):

        print("Initialise CNN")
        
        self.params = params
        self.network_name = 'DeepQNetwork'
        self.sess = tf.Session()
        self.x = tf.placeholder('float', [None, params['width'], params['height'], 3], name='input_image')
        self.q_t = tf.placeholder('float', [None], name='target_q')
        self.actions = tf.placeholder("float", [None, 4], name='actions')
        self.rewards = tf.placeholder("float", [None], name='rewards')
        self.terminals = tf.placeholder("float", [None], name='terminals')    

        conv1 = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(inputs=conv2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        conv3_flat = tf.reshape(conv3, [-1, params['width'] * params['height'] * 256])
        dense = tf.layers.dense(inputs=conv3_flat, units=256, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=4, name='logits')

        self.y = logits

        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.discount = tf.constant(self.params['discount'])

        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1)
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))
        self.optim = tf.train.AdamOptimizer(0.00025).minimize(self.cost, global_step=self.global_step)
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