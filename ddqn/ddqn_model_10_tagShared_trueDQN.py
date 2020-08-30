'''
created by zhenmaoli
2018-3-24
the action is defined as 10 news once
the state is defined as (index news,user information)

follow DQN
'''

import tensorflow as tf
import tensorflow.contrib.layers as layers
from utility.hype_parameters import HypeParameters as hypes
from models import model_nfm_batch as nfm
from tensorflow.python.client import timeline
import math

import numpy as np


class ActorPickNewsQ():
    '''
    pick a number of news form candidates
    '''
    def __init__(self,tau,gamma,
                 batch_size,
                 num_pre_vars,  #trainable variables offset
                 optimizer,
                 max_userTags_num,
                 max_candidates_num,
                 learning_rate=0.0001,
                 cliped_norm=500,
                 num_history=50,
                 weight_decay_rate=0.0001,
                 scope="Actor"):
        with tf.variable_scope(scope):
            self.tau=tau
            self.gamma = gamma
            self.batch_size=batch_size
            self.optimizer=optimizer
            self.max_candidates_num=max_candidates_num
            self.max_userTags_num=max_userTags_num
            self.num_history=50
            self.weight_decay_rate = weight_decay_rate

            self.action_feed_for_value = tf.placeholder(dtype=tf.int32, shape=[batch_size,hypes.num_news_action], name='action_feed_for_value')
            self.action_feed_for_history = tf.placeholder(dtype=tf.int32, shape=[batch_size, hypes.num_news_action],
                                              name='action_feed_for_history')

            #candidates_info
            self.candidates_tags = tf.placeholder(tf.int32,[batch_size,max_candidates_num, hypes.max_newsTag_num],'candidates_input')
            self.candidates_mask = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_candidates_num],
                                                  name='candidates_mask')#define mask
            #user_info
            self.user_tags = tf.placeholder(tf.int32, [batch_size,1, max_userTags_num], "user_input")
            self.user_tags_w = tf.placeholder(tf.float32, [batch_size,1, max_userTags_num], "user_input_w")
            self.user_tags_k = tf.placeholder(tf.int32, [batch_size, 1, max_userTags_num], "user_input_k")
            self.user_info=(self.user_tags,self.user_tags_w,self.user_tags_k)

            #feature extra
            self.feature_extra=tf.placeholder(tf.int32,[batch_size,1,2],"feature_extra_input")

            #clip_norm
            self.cliped_norm = cliped_norm * math.sqrt(hypes.num_news_action)

            #history_info
            self.history_newsTags=tf.placeholder(tf.int32,shape=[batch_size,num_history,hypes.max_newsTag_num],name='history_newsTags')
            self.history_clicks = tf.placeholder(tf.int32, shape=[batch_size, num_history, 1],
                                                   name='history_clicks')
            self.history_positions = tf.placeholder(tf.int32, shape=[batch_size, num_history, 1],
                                                   name='history_positions')
            self.history=(self.history_newsTags,self.history_clicks,self.history_positions)

            # define mask
            mask_ones = []
            for i in range(max_candidates_num):
                mask_list_row = [0.0] * max_candidates_num
                mask_list_row[i] = 1.
                mask_ones.append(mask_list_row)
            self.indices_ones_mask = tf.constant(mask_ones, dtype=tf.float32)

            #Actor Network
            with tf.variable_scope('actor_Q_net') as actor_scope:
                self.nfm_model = nfm.NeuralFM(hypes.max_tags_num, hypes.tags_emb_size)

                self.candi_user_nfm = \
                    self.nfm_model.build_nfm_candidates_user_outerHistory_batch(self.candidates_tags,
                                                                                      self.user_info,
                                                                                      self.history,
                                                                                      batch_size)

                # self.candi_user_nfm = \
                #     self.nfm_model.build_nfm_candidates_user_outerHistory_mean_conbined(self.candidates_tags,
                #                                                                 self.user_info,
                #                                                                 self.history,
                #                                                                 batch_size)

                action_feedTrue_tmp,step_reward_feedTrue_tmp  =self.pick_news_drop_batch(candi_user_nfm=self.candi_user_nfm,
                                                                                         candidates_tags=self.candidates_tags,
                                                                                         feed_for_value=True,
                                                                                         nfm_model=self.nfm_model,
                                                                                         feed_for_history=True)
                self.action_feedTrue=tf.transpose(action_feedTrue_tmp,[1,0])
                self.step_reward_feedTrue=tf.transpose(step_reward_feedTrue_tmp,[1,0])
                self.value_estimate_feedTrue=self.q_value_net(self.step_reward_feedTrue)

                self.network_params = tf.trainable_variables()[num_pre_vars:]
                actor_scope.reuse_variables()
                action_feedFalse_tmp, step_reward_feedFalse_tmp = self.pick_news_drop_batch(candi_user_nfm=self.candi_user_nfm,
                                                                                            candidates_tags=self.candidates_tags,
                                                                                            feed_for_value=False,
                                                                                            nfm_model=self.nfm_model,
                                                                                            feed_for_history=True)

                self.action_feedFalse = tf.transpose(action_feedFalse_tmp, [1, 0])
                self.step_reward_feedFalse = tf.transpose(step_reward_feedFalse_tmp, [1, 0])
                self.value_estimate_feedFalse = self.q_value_net(self.step_reward_feedFalse)

                action_tmp, step_reward_tmp = self.pick_news_drop_batch(
                    candi_user_nfm=self.candi_user_nfm,
                    candidates_tags=self.candidates_tags,
                    feed_for_value=False,
                    nfm_model=self.nfm_model,
                    feed_for_history=False)

                self.best_action_train = tf.transpose(action_tmp, [1, 0])


            # Target Network
            with tf.variable_scope('target_Q_net') as target_scope:
                self.target_nfm_model = nfm.NeuralFM(hypes.max_tags_num, hypes.tags_emb_size)

                self.target_candi_user_nfm = self.target_nfm_model. \
                    build_nfm_candidates_user_outerHistory_batch(self.candidates_tags,
                                                                       self.user_info,
                                                                       self.history,
                                                                       batch_size)

                # self.target_candi_user_nfm = self.target_nfm_model. \
                #     build_nfm_candidates_user_outerHistory_mean_conbined(self.candidates_tags,
                #                                                  self.user_info,
                #                                                  self.history,
                #                                                  batch_size)

                target_action_feedTrue_tmp, \
                target_step_reward_feedTrue_tmp= self.pick_news_drop_batch(candi_user_nfm=self.target_candi_user_nfm,
                                                                           candidates_tags=self.candidates_tags,
                                                                           feed_for_value=True,
                                                                           nfm_model=self.target_nfm_model,
                                                                           feed_for_history=True)

                self.target_action_feedTrue = tf.transpose(target_action_feedTrue_tmp, [1, 0])
                self.target_step_reward_feedTrue = tf.transpose(target_step_reward_feedTrue_tmp, [1, 0])
                self.target_value_estimate_feedTrue = self.q_value_net(self.target_step_reward_feedTrue)

                target_scope.reuse_variables()

                target_action_feedFalse_tmp, \
                target_step_reward_feedFalse_tmp = self.pick_news_drop_batch(candi_user_nfm=self.target_candi_user_nfm,
                                                                             candidates_tags=self.candidates_tags,
                                                                             feed_for_value=False,
                                                                             nfm_model=self.target_nfm_model,
                                                                             feed_for_history=False)

                self.target_action_feedFalse = tf.transpose(target_action_feedFalse_tmp, [1, 0])
                self.target_step_reward_feedFalse = tf.transpose(target_step_reward_feedFalse_tmp, [1, 0])
                self.target_value_estimate_feedFalse = self.q_value_net(self.target_step_reward_feedFalse)

                self.target_action_feedFalse_tensor=tf.convert_to_tensor(self.target_action_feedFalse,name='news_recommended')
                self.target_step_reward_feedFalse_tensor=tf.convert_to_tensor(self.target_step_reward_feedFalse,
                                                                              name='target_action_reward')

            self.target_network_params = tf.trainable_variables()[num_pre_vars+len(self.network_params):]

            # Op for periodically updating target network with online network
            self.update_target_network_params = \
                [self.target_network_params[i].
                     assign(tf.multiply(self.network_params[i], self.tau) +
                            tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

            # Network target (y_i)
            self.y = tf.placeholder(tf.float32,shape=[batch_size,hypes.num_news_action],name='y')
            list_summary = []
            # Define loss and optimization Op
            self.loss = tf.losses.mean_squared_error(self.y, self.step_reward_feedTrue)
            self.td_error=tf.reduce_mean(tf.abs(tf.subtract(self.y,self.step_reward_feedTrue)),axis=-1)
            with tf.name_scope("weights_decay"):
                # embedding 和 后面的参数分别处理
                embedding_weights = self.network_params[0:1]
                others = self.network_params[1:]
                self.weight_decay = tf.add_n([self.weight_decay_rate * tf.nn.l2_loss(var) for var in others])

            with tf.variable_scope('adam',reuse=tf.AUTO_REUSE):
                self.train_op = self.train_choice(total_loss=self.loss+self.weight_decay,
                                                  optimizer=self.optimizer,
                                                  learning_rate=learning_rate,
                                                  log_histograms=False,
                                                  list_summary=list_summary)

            self.num_trainable_vars = len(self.network_params) + \
                                      len(self.target_network_params)
            #tf.summmary

            # sv=tf.summary.scalar('value_estimate',tf.reduce_sum(self.value_estimate)/batch_size)
            # list_summary.append(sv)
            sl = tf.summary.scalar('loss/value_loss', self.loss)
            list_summary.append(sl)

            s_weight_d = tf.summary.scalar('loss/weight_decay', self.weight_decay)
            list_summary.append(s_weight_d)

            # for i in range(hypes.num_news_action):
            #     s = tf.summary.scalar('step_reward/train_action_probs_' + str(i),
            #                           tf.reduce_sum(self.step_reward_feedTrue[:, i]) / batch_size)
            #     list_summary.append(s)

            # for i in range(hypes.num_news_action):
            #     s = tf.summary.scalar('target_step_reward/target_action_probs_' + str(i),
            #                           tf.reduce_sum(self.target_step_reward_feedFalse[:, i]) / batch_size)
            #     list_summary.append(s)

            # for i in range(hypes.num_news_action):
            #     s = tf.summary.scalar('y_step/y_' + str(i),
            #                           tf.reduce_sum(self.y[:, i]) / batch_size)
            #     list_summary.append(s)

            # userTag_position_embedding
            userTag_position_embedding = self.network_params[1]
            for i in range(6):
                sc = tf.summary.histogram('position_embedding/userTag_position_' + str(i), userTag_position_embedding[i])
                list_summary.append(sc)
            # weights about history
            hist_click_embedding = self.network_params[2]
            for i in range(3):
                sc = tf.summary.histogram('position_embedding/hist_click_' + str(i), hist_click_embedding[i])
                list_summary.append(sc)
            hist_position_embedding = self.network_params[3]
            for i in range(5):
                sp = tf.summary.histogram('position_embedding/hist_position_' + str(i), hist_position_embedding[i])
                list_summary.append(sp)

            # for i in range(len(self.step_reward)):
            #     s = tf.summary.scalar('train_action_probs_'+str(i),self.step_reward[i])
            #     list_summary.append(s)

            # weights
            # for var in self.network_params:
            #     if 'tag_lookup' not in var.op.name:
            #         sw = tf.summary.histogram(var.op.name, var)
            #         list_summary.append(sw)

            self.merge=tf.summary.merge(list_summary)

    def train_choice(self, total_loss, optimizer, learning_rate,list_summary,log_histograms=True):

        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'LAZY_ADAM':
            opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
        # Compute gradients.
        # with tf.control_dependencies([total_loss]):
        grads = opt.compute_gradients(total_loss)

        # clip
        with tf.name_scope("gradients_clips"):
            for i, (g, v) in enumerate(grads):
                if g is not None:
                    grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients

        # Add histograms for gradients.
        if log_histograms:
            for grad, var in grads:
                if grad is not None:
                    sg = tf.summary.histogram(var.op.name + '/gradients', grad)
                    sv = tf.summary.histogram(name=var.op.name, values=var)
                    list_summary.append(sg)
                    list_summary.append(sv)
        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads)


        # # Add histograms for trainable variables.
        # if log_histograms:
        #     for var in tf.trainable_variables():
        #         tf.summary.histogram(var.op.name, var)
        #


        # # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     moving_average_decay, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())
        #
        # with tf.control_dependencies([apply_gradient_op]):
        # train_op = tf.no_op(name='train')

        return apply_gradient_op


    def train_choice_gradient(self, total_loss, optimizer, learning_rate,list_summary,log_histograms=True):

        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
        # Compute gradients.
        # with tf.control_dependencies([total_loss]):
        grads = opt.compute_gradients(total_loss)
        res_grads=[]

        # clip
        # with tf.name_scope("gradients_clips"):
        #     for i, (g, v) in enumerate(grads):
        #         if g is not None:
        #             grad_tmp = (tf.clip_by_norm(g, 5), v)  # clip gradients
        #             res_grads.append(grad_tmp)

                # Add histograms for gradients.
        if log_histograms:
            for grad, var in grads:
                if grad is not None:
                    sg = tf.summary.histogram(var.op.name + '/gradients', grad)
                    list_summary.append(sg)
        # Apply gradients.
        # apply_gradient_op = opt.apply_gradients(grads)


        # # Add histograms for trainable variables.
        # if log_histograms:
        #     for var in tf.trainable_variables():
        #         tf.summary.histogram(var.op.name, var)
        #


        # # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     moving_average_decay, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())
        #
        # with tf.control_dependencies([apply_gradient_op]):
        # train_op = tf.no_op(name='train')

        return opt,res_grads

    def pick_news_drop_batch(self, candi_user_nfm, candidates_tags, nfm_model,feed_for_value=False,feed_for_history=False):
        '''
        :param candidates_tags: candidates news tags idx
        :param candi_user_nfm: nfm of candidate news and user information
        :return: a number of news idx
        '''
        res_positions = []
        res_rewards = []

        indices_mask_current=tf.constant(1.0,dtype=tf.float32,shape=[self.batch_size,self.max_candidates_num])
        batch_range=tf.reshape(tf.range(0,self.batch_size,dtype=tf.int32),[self.batch_size,1],name='batch_range')

        candi_user_nfm_trans = self.transNFMVector(candi_user_nfm)

        history_newsTags = tf.zeros(shape=[self.batch_size,hypes.num_news_action - 1, hypes.max_newsTag_num], dtype=tf.int32)
        history_newsTags_current=history_newsTags

        # define history position weights
        history_position_weights = tf.get_variable('history_position_weights',
                                                   shape=[1, hypes.num_news_action - 1, 1, hypes.tags_emb_size],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        def step(candi_user_nfm_trans, candidates_tags, history_newsTags_current,indices_mask_current,i):

            candi_hist_nfm = self.build_candi_hist_nfm(candidates_tags, history_newsTags_current,nfm_model,
                                                       history_position_weights)
            #news_probs_tmp = self.build_attention_all(candi_user_nfm_trans,candi_hist_nfm)
            news_probs_tmp=self.predict_out_in_conbined(candi_user_nfm_trans, candi_hist_nfm,i,nfm_model)

            #set picked position of news_probs to 0
            news_probs_mask = indices_mask_current * self.candidates_mask
            # news_probs=tf.multiply(news_probs_tmp,indices_mask_current*self.candidates_mask,name='news_probs')
            paddings = tf.ones_like(news_probs_tmp) * (-2 ** 32 + 1)
            news_probs = tf.where(news_probs_mask <= 1e-7, paddings, news_probs_tmp)
            tmp_position = tf.argmax(input=news_probs, axis=1,output_type=tf.int32,name="max_position")

            picked_position_batch = tmp_position  # 每一步预测的最优action
            tmp_indices = tf.expand_dims(tmp_position, 1)
            indices_2d = tf.concat([batch_range, tmp_indices], 1, name='indices_2d')
            picked_prob_batch = tf.gather_nd(news_probs, indices_2d, name='picked_prob_batch')

            if feed_for_value == True: #根据历史中真实的action进行替换
                tmp_position = tf.gather(self.action_feed_for_value, i, axis=1, name='position_feed_for_value')
                tmp_indices = tf.expand_dims(tmp_position, 1)
                indices_2d = tf.concat([batch_range, tmp_indices], 1, name='indices_2d_feed_for_value')

                picked_position_batch = tmp_position
                picked_prob_batch = tf.gather_nd(news_probs, indices_2d, name='picked_prob_batch')

            if feed_for_history == True:
                tmp_position = tf.gather(self.action_feed_for_history, i, axis=1, name='position_feed_for_history')
                tmp_indices = tf.expand_dims(tmp_position, 1)
                indices_2d = tf.concat([batch_range, tmp_indices], 1, name='indices_2d_feed_for_history')

            #update indices_mask
            mask_ones_tmp = tf.gather(self.indices_ones_mask, tmp_position, axis=0)
            indices_mask_next=tf.subtract(indices_mask_current,mask_ones_tmp,name='indices_mask_next')

            history_newsTags_next=history_newsTags_current
            if i < hypes.num_news_action-1:
                picked = tf.gather_nd(candidates_tags,indices_2d,name='picked_candidates_tags')
                picked_e=tf.expand_dims(picked,axis=1)
                #picked = tf.reshape(picked, [-1, hypes.max_newsTag_num])
                history_newsTags_next = tf.concat([history_newsTags_current[:,0:i],
                                              picked_e,
                                              history_newsTags_current[:,i + 1:]], 1,name='history_newsTags_next')

            return indices_mask_next, history_newsTags_next,picked_position_batch, picked_prob_batch

        with tf.variable_scope('for_steps') as for_scope:
            for_scope.reuse_variables()
            for i in range(hypes.num_news_action):
                with tf.name_scope('step'):
                    # run n times, if the number is not known, then we just need while_loop
                    indices_mask_current, \
                    history_newsTags_current, \
                    picked_position, \
                    picked_prob = step(candi_user_nfm_trans=candi_user_nfm_trans,
                                       candidates_tags=candidates_tags,
                                       history_newsTags_current=history_newsTags_current,
                                       indices_mask_current=indices_mask_current,
                                       i=i)

                    res_positions.append(picked_position)
                    res_rewards.append(picked_prob)

        return res_positions, res_rewards


    def predict_best_action_train(self,states,sess=None):
        candidates_tags_batch=[]
        user_tags_batch=[]
        user_tags_w_batch=[]
        user_tags_k_batch = []
        #history
        h_newsTags_batch=[]
        h_clicks_batch=[]
        h_positions_batch=[]

        #mask
        candidates_mask_batch=[]

        #feature_extra
        feature_extra=[]
        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])
            user_tags_k_batch.append([s.user_tags_k])
            # print('c', len(s.candidates_tags))
            # print('u', len(s.user_tags))
            # print('uw', len(s.user_tags_w))
            h_newsTags=[]
            h_clicks=[]
            h_positions=[]
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

            feature_extra.append([s.feature_extra])
            # print('0', len(s.history[0]))
            # print('1', len(s.history[1]))
            # print('2', len(s.history[2]))
        # candidates_tags_batch=np.array(candidates_tags_batch,dtype=np.int32)
        # user_tags_batch=np.array(user_tags_batch,dtype=np.int32)
        # user_tags_w_batch = np.array(user_tags_w_batch, dtype=np.float32)

        # h_newsTags_batch = np.array(h_newsTags_batch, dtype=np.int32)
        # h_clicks_batch = np.array(h_clicks_batch, dtype=np.int32)
        # h_positions_batch = np.array(h_positions_batch, dtype=np.float32)

        sess=sess or tf.get_default_session()
        feed_in={self.candidates_tags:candidates_tags_batch,
                 self.user_tags:user_tags_batch,
                 self.user_tags_w:user_tags_w_batch,
                 self.user_tags_k: user_tags_k_batch,
                 self.history_newsTags:h_newsTags_batch,
                 self.history_clicks:h_clicks_batch,
                 self.history_positions:h_positions_batch,
                 self.candidates_mask:candidates_mask_batch,
                 self.feature_extra:feature_extra}
        return sess.run(self.best_action_train,feed_in)

    def predict_action(self,states,action_from_record,sess=None):
        candidates_tags_batch=[]
        user_tags_batch=[]
        user_tags_w_batch=[]
        user_tags_k_batch = []
        #history
        h_newsTags_batch=[]
        h_clicks_batch=[]
        h_positions_batch=[]

        #mask
        candidates_mask_batch=[]

        #feature_extra
        feature_extra=[]
        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])
            user_tags_k_batch.append([s.user_tags_k])
            # print('c', len(s.candidates_tags))
            # print('u', len(s.user_tags))
            # print('uw', len(s.user_tags_w))
            h_newsTags=[]
            h_clicks=[]
            h_positions=[]
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

            feature_extra.append([s.feature_extra])
            # print('0', len(s.history[0]))
            # print('1', len(s.history[1]))
            # print('2', len(s.history[2]))
        # candidates_tags_batch=np.array(candidates_tags_batch,dtype=np.int32)
        # user_tags_batch=np.array(user_tags_batch,dtype=np.int32)
        # user_tags_w_batch = np.array(user_tags_w_batch, dtype=np.float32)

        # h_newsTags_batch = np.array(h_newsTags_batch, dtype=np.int32)
        # h_clicks_batch = np.array(h_clicks_batch, dtype=np.int32)
        # h_positions_batch = np.array(h_positions_batch, dtype=np.float32)

        sess=sess or tf.get_default_session()
        feed_in={self.candidates_tags:candidates_tags_batch,
                 self.user_tags:user_tags_batch,
                 self.user_tags_w:user_tags_w_batch,
                 self.user_tags_k: user_tags_k_batch,
                 self.action_feed_for_history: action_from_record,
                 self.history_newsTags:h_newsTags_batch,
                 self.history_clicks:h_clicks_batch,
                 self.history_positions:h_positions_batch,
                 self.candidates_mask:candidates_mask_batch,
                 self.feature_extra:feature_extra}
        return sess.run([self.action_feedFalse,self.step_reward_feedFalse,self.value_estimate_feedFalse],feed_in)

    def predict_target_value(self, states, action_from_record,action_from_training,sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        user_tags_k_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []
        # mask
        candidates_mask_batch = []
        # feature_extra
        feature_extra = []

        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])
            user_tags_k_batch.append([s.user_tags_k])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

            feature_extra.append([s.feature_extra])

        sess = sess or tf.get_default_session()
        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.user_tags_k: user_tags_k_batch,
                   self.action_feed_for_history: action_from_record,
                   self.action_feed_for_value:action_from_training,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch,
                   self.feature_extra:feature_extra}
        return sess.run(self.target_step_reward_feedTrue, feed_in)

    def predict_target_action(self, states,sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        user_tags_k_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []
        # mask
        candidates_mask_batch = []
        # feature_extra
        feature_extra = []

        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])
            user_tags_k_batch.append([s.user_tags_k])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

            feature_extra.append([s.feature_extra])

        sess = sess or tf.get_default_session()
        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.user_tags_k: user_tags_k_batch,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch,
                   self.feature_extra:feature_extra}
        return sess.run(self.target_action_feedFalse,feed_in)

    def predict_target(self, states,sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        user_tags_k_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []
        # mask
        candidates_mask_batch = []
        # feature_extra
        feature_extra = []

        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])
            user_tags_k_batch.append([s.user_tags_k])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

            feature_extra.append([s.feature_extra])

        sess = sess or tf.get_default_session()
        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.user_tags_k: user_tags_k_batch,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch,
                   self.feature_extra: feature_extra}

        return sess.run([self.target_action_feedFalse,
                         self.target_step_reward_feedFalse,
                         self.target_value_estimate_feedFalse],feed_in)

    def predict_target_time(self, states, sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        user_tags_k_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []
        # mask
        candidates_mask_batch = []
        # feature_extra
        feature_extra = []

        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])
            user_tags_k_batch.append([s.user_tags_k])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

            feature_extra.append([s.feature_extra])

        sess = sess or tf.get_default_session()
        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.user_tags_k: user_tags_k_batch,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch,
                   self.feature_extra: feature_extra}

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        target_action,\
        target_action_reward,\
        tarrget_action_value=sess.run([self.target_action_feedFalse,
                                       self.target_step_reward_feedFalse,
                                       self.target_value_estimate_feedFalse], feed_in,
                                       options=run_options, run_metadata=run_metadata)
        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

        return target_action, target_action_reward, tarrget_action_value

    def train(self,states,action_from_history,y,sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        user_tags_k_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []
        # mask
        candidates_mask_batch = []
        # feature_extra
        feature_extra = []

        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])
            user_tags_k_batch.append([s.user_tags_k])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

            feature_extra.append([s.feature_extra])

        sess=sess or tf.get_default_session()

        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.user_tags_k: user_tags_k_batch,
                   self.action_feed_for_history: action_from_history,
                   self.action_feed_for_value:action_from_history,
                   self.y:y,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch,
                   self.feature_extra: feature_extra}

        return sess.run([self.train_op,
                         self.loss,
                         self.value_estimate_feedTrue,
                         self.step_reward_feedTrue,
                         self.action_feedTrue,
                         self.merge],feed_in)

    # def get_TD_error(self,states,action,y,sess=None):
    #     candidates_tags_batch = []
    #     user_tags_batch = []
    #     user_tags_w_batch = []
    #     user_tags_k_batch = []
    #     # history
    #     h_newsTags_batch = []
    #     h_clicks_batch = []
    #     h_positions_batch = []
    #     # mask
    #     candidates_mask_batch = []
    #     # feature_extra
    #     feature_extra = []
    #
    #     for s in states:
    #         candidates_tags_batch.append(s.candidates_tags)
    #         user_tags_batch.append([s.user_tags])
    #         user_tags_w_batch.append([s.user_tags_w])
    #         user_tags_k_batch.append([s.user_tags_k])
    #
    #         h_newsTags = []
    #         h_clicks = []
    #         h_positions = []
    #         for h in s.history:
    #             h_newsTags.append(h[0])
    #             h_clicks.append([h[1]])
    #             h_positions.append([h[2]])
    #         h_newsTags_batch.append(h_newsTags)
    #         h_clicks_batch.append(h_clicks)
    #         h_positions_batch.append(h_positions)
    #
    #         candidates_mask_batch.append(s.candidates_mask)
    #
    #         feature_extra.append([s.feature_extra])
    #
    #     sess=sess or tf.get_default_session()
    #
    #     feed_in = {self.candidates_tags: candidates_tags_batch,
    #                self.user_tags: user_tags_batch,
    #                self.user_tags_w: user_tags_w_batch,
    #                self.user_tags_k: user_tags_k_batch,
    #                self.action_recommended: action,
    #                self.y:y,
    #                self.history_newsTags: h_newsTags_batch,
    #                self.history_clicks: h_clicks_batch,
    #                self.history_positions: h_positions_batch,
    #                self.candidates_mask: candidates_mask_batch,
    #                self.feature_extra: feature_extra}
    #
    #     return sess.run([self.td_error],feed_in)
    #
    # def train_time(self,states,action,y,sess=None):
    #     candidates_tags_batch = []
    #     user_tags_batch = []
    #     user_tags_w_batch = []
    #     user_tags_k_batch = []
    #     # history
    #     h_newsTags_batch = []
    #     h_clicks_batch = []
    #     h_positions_batch = []
    #     # mask
    #     candidates_mask_batch = []
    #     # feature_extra
    #     feature_extra = []
    #
    #     for s in states:
    #         candidates_tags_batch.append(s.candidates_tags)
    #         user_tags_batch.append([s.user_tags])
    #         user_tags_w_batch.append([s.user_tags_w])
    #         user_tags_k_batch.append([s.user_tags_k])
    #
    #         h_newsTags = []
    #         h_clicks = []
    #         h_positions = []
    #         for h in s.history:
    #             h_newsTags.append(h[0])
    #             h_clicks.append([h[1]])
    #             h_positions.append([h[2]])
    #         h_newsTags_batch.append(h_newsTags)
    #         h_clicks_batch.append(h_clicks)
    #         h_positions_batch.append(h_positions)
    #
    #         candidates_mask_batch.append(s.candidates_mask)
    #
    #         feature_extra.append([s.feature_extra])
    #
    #     sess=sess or tf.get_default_session()
    #
    #     feed_in = {self.candidates_tags: candidates_tags_batch,
    #                self.user_tags: user_tags_batch,
    #                self.user_tags_w: user_tags_w_batch,
    #                self.user_tags_k: user_tags_k_batch,
    #                self.action_recommended: action,
    #                self.y: y,
    #                self.history_newsTags: h_newsTags_batch,
    #                self.history_clicks: h_clicks_batch,
    #                self.history_positions: h_positions_batch,
    #                self.candidates_mask: candidates_mask_batch,
    #                self.feature_extra: feature_extra}
    #
    #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     run_metadata = tf.RunMetadata()
    #
    #     train_op,loss,value_estimate,step_reward,action_record,merge=sess.run([self.train_op,
    #                      self.loss,
    #                      self.value_estimate,
    #                      self.step_reward,
    #                      self.action_record,
    #                      self.merge], feed_in,
    #               options=run_options, run_metadata=run_metadata)
    #     # Create the Timeline object, and write it to a json
    #     tl = timeline.Timeline(run_metadata.step_stats)
    #     ctf = tl.generate_chrome_trace_format()
    #     with open('train_timeline.json', 'w') as f:
    #         f.write(ctf)
    #
    #     return train_op,loss,value_estimate,step_reward,action_record,merge

    def update_target_network(self,sess=None):
        sess = sess or tf.get_default_session()
        sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def transNFMVector(self,candi_user_nfm):
        with tf.variable_scope('fc_for_candidates'):
            weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
            candi_user_nfm=tf.expand_dims(candi_user_nfm,1) #[b,1,200,128]
            candidate_inter_1 = layers.conv2d(candi_user_nfm, 256,
                                              kernel_size=[1,1],
                                              activation_fn=tf.nn.relu,
                                              weights_initializer=weights_initializer,
                                              weights_regularizer=weights_regularizer)

            candidate_inter_2 = layers.conv2d(candidate_inter_1, 128,
                                              kernel_size=[1, 1],
                                              activation_fn=tf.nn.relu,
                                              weights_initializer=weights_initializer,
                                              weights_regularizer=weights_regularizer)

        return tf.squeeze(candidate_inter_2,axis=1) #[b,200,128]

    def build_candi_hist_nfm(self,candi_tags,hist_tags,nfm_model,history_position_weights):

        candi_hist_nfm = nfm_model.build_nfm_candidates_innerHistory_cross_batch(
            candi_tags, hist_tags, history_position_weights, self.batch_size)

        candi_hist_nfm=tf.expand_dims(candi_hist_nfm,1)
        with tf.variable_scope('fc_for_history',reuse=tf.AUTO_REUSE):
            weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
            inter_1 = layers.conv2d(candi_hist_nfm, 256,
                                    kernel_size=[1,1],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=weights_initializer,
                                    weights_regularizer=weights_regularizer)

            inter_2 = layers.conv2d(inter_1, 128,
                                    kernel_size=[1,1],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=weights_initializer,
                                    weights_regularizer=weights_regularizer)
        return tf.squeeze(inter_2,axis=1) #[b,200,128]


    def build_attention_all(self,candi_user_current,newsTags_history):

        with tf.variable_scope(name_or_scope='attention_all',reuse=tf.AUTO_REUSE):

            in_overall=tf.concat([candi_user_current,newsTags_history],2) #[b,200,128+128]
            #in_overall = tf.nn.l2_normalize(in_overall,axis=2)
            #in_overall=tf.reshape(in_overall,[self.batch_size,1,-1,hypes.tags_emb_size*2])
            in_overall=tf.expand_dims(in_overall,2)


            weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
            tmp=layers.conv2d(in_overall, 128,
                              kernel_size=[1,1],
                              activation_fn=tf.nn.relu,
                              weights_initializer=weights_initializer,
                              weights_regularizer=weights_regularizer)
            tmp1 = layers.conv2d(tmp, 1,
                                 kernel_size=[1, 1],
                                 activation_fn=tf.identity,
                                 weights_initializer=weights_initializer,
                                 weights_regularizer=weights_regularizer)
            tmp1=tf.reshape(tmp1,[self.batch_size,-1])
        return tmp1
        #return tf.nn.softmax(tmp1)

    def predict_out_in_conbined(self, candi_user_current, newsTags_history, step, nfm_model):
        with tf.variable_scope(name_or_scope='attention_all', reuse=tf.AUTO_REUSE):
            step_embeddings = nfm_model.position_embedding(step,
                                                           10,
                                                           hypes.tags_emb_size*0.25,
                                                           scope="inner_step_embedding")

            # feature_extra
            # feature_extra_embedding = nfm_model.position_embedding(self.feature_extra,
            #                                                        60,
            #                                                        hypes.tags_emb_size*0.25,
            #                                                        scope="feature_extra_embedding")

            candi_shape=candi_user_current.get_shape().as_list()
            #position
            step_tile = tf.tile(tf.reshape(step_embeddings, [1, 1, -1]), [candi_shape[0],
                                                             candi_shape[1], 1], name='step_tile')

            # feature_extra_tile=tf.tile(tf.reshape(feature_extra_embedding,[candi_shape[0],1,-1]), [1,
            #                                                  candi_shape[1], 1], name='feature_extra_tile')
            in_overall = tf.concat([candi_user_current, newsTags_history,step_tile], -1)  # [b,200,128+128]

            # in_overall = tf.concat([candi_user_current, newsTags_history],-1)  # [b,200,128+128]
            # in_overall = tf.add(in_overall, tf.reshape(step_embeddings, [1,1,-1]))
            # in_overall = tf.add(in_overall, feature_extra_embedding[:,:,0])
            # in_overall = tf.add(in_overall, feature_extra_embedding[:,:,1])

            in_overall = tf.expand_dims(in_overall, 1)

            weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
            tmp = layers.conv2d(in_overall, 128,
                                kernel_size=[1, 1],
                                activation_fn=tf.nn.relu,
                                weights_initializer=weights_initializer,
                                weights_regularizer=weights_regularizer)
            tmp1 = layers.conv2d(tmp, 1,
                                 kernel_size=[1, 1],
                                 activation_fn=tf.identity,
                                 weights_initializer=weights_initializer,
                                 weights_regularizer=weights_regularizer)
            tmp1 = tf.reshape(tmp1, [self.batch_size, -1])
        return tmp1

    def q_value_net(self,action_rewards):
        with tf.variable_scope('q_out'):
            #in_overall = tf.concat([position,action_rewards], 1)
            action_rewards_cliped = tf.clip_by_norm(action_rewards, self.cliped_norm, axes=1)
            in_overall=action_rewards_cliped

            out = tf.reduce_sum(in_overall,1)

            # weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
            # tmp_1=layers.fully_connected(in_overall, 128, activation_fn=tf.nn.relu,
            #                              weights_initializer=weights_initializer,
            #                              weights_regularizer=weights_regularizer)
            #                              #biases_initializer=None)
            #
            # out = layers.fully_connected(tmp_1, 1, activation_fn=tf.identity,
            #                              weights_initializer=weights_initializer,
            #                              weights_regularizer=weights_regularizer)
            #                              #biases_initializer=None)
            #out=tf.squeeze(out,axis=1)

        return out