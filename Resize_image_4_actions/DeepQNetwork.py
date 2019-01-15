import numpy as np
import tensorflow as tf
    
class DeepQNetwork:

    def __init__(self, params):

        print("Initialise CNN")
        
        self.params = params
        self.network_name = 'DeepQNetwork'
        self.sess = tf.Session()
        self.x = tf.placeholder('float', [None, params['augmentation']*params['width'], params['augmentation']*params['height'], 3])
        self.tar_yj = tf.placeholder("float", [])
        self.actions = tf.placeholder("float", [None, 4])
        self.rewards = tf.placeholder("float", [None])
        self.terminals = tf.placeholder("float", [None])   

        conv1 = tf.layers.conv2d(inputs=self.x, filters=8, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        conv3_flat = tf.reshape(conv3, [-1, params['augmentation']*params['width']*params['augmentation']*params['height']  * 64])
        dense = tf.layers.dense(inputs=conv3_flat, units=64, activation=tf.nn.relu)
        print(params['augmentation']*params['width'], params['augmentation']*params['height'])
        # lstm_cell = tf.nn.rnn_cell.LSTMCell(64, forget_bias=1.0)
        # dense = tf.expand_dims(dense, axis = 2)
        # mem_cell, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=dense, dtype=tf.float32)
        
        # logits = tf.layers.dense(inputs=mem_cell[:,-1], units=4)
        logits = tf.layers.dense(inputs=dense, units=4)
        self.y = logits

        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.discount = tf.constant(self.params['discount'])

        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions))

        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.Q_pred)))

        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.tar_yj, self.Q_pred), 2))


        if self.params['load_file'] is not None:
            self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]),name='global_step', trainable=False)
        else:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.optim = tf.train.AdamOptimizer(self.params['lr']).minimize(self.cost, global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=0)

        self.sess.run(tf.global_variables_initializer())

        if self.params['load_file'] is not None:
            print('Loading checkpoint...')
            self.saver.restore(self.sess,self.params['load_file'])


    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r, tar_yj):

            feed_dict={self.x: bat_s, self.tar_yj: tar_yj, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
            _,cnt,cost = self.sess.run([self.optim, self.global_step,self.cost],feed_dict=feed_dict)

            return cnt, cost

    def save_ckpt(self,filename):
        self.saver.save(self.sess, filename)
            