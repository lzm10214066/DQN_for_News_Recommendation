import argparse
import collections
import itertools
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.framework import graph_util

import Console
from ddqn import ddqn_model_10_tagShared_trueDQN as ddqn
from environment import environment_10_newsTags_tagShared_trueDQN as env
from utility import replay
from utility.hype_parameters import HypeParameters as hypes
import shutil
import numpy as np
import random

Transition = collections.namedtuple("Transition",
                                    ["state", "action", "reward", "next_state", "done"])
MAX_DATA_FILE = 336
MAX_LOG_FILE=  96


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # learning rate
    parser.add_argument('--policy_lr_rate', type=float,
                        help='policy net learning rate.',
                        default=0.0001)
    parser.add_argument('--optimizer', type=str,
                        help='optimizer.',
                        default='ADAM')

    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',
                        default='saved_models/ckpt/online')

    parser.add_argument('--saved_model_dir', type=str,
                        help='dir for saving model.',
                        default='saved_models')

    parser.add_argument('--logs_dir', type=str,
                        help='dir for log.',
                        default='logs')

    parser.add_argument('--buffer_size', type=int,
                        help='max buffer_size for start of training',
                        default=200)

    parser.add_argument('--start_rate', type=float,
                        help='max buffer_size for start of training',
                        default=0.5)

    parser.add_argument('--batch_size', type=int,
                        help='batch_size for training',
                        default=4)

    parser.add_argument('--average_size', type=int,
                        help='size for time estimation of test',
                        default=100)

    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='gpu',
                        default=0.4)

    parser.add_argument('--cliped_norm', type=float,
                        help='',
                        default=500)

    parser.add_argument('--tau', type=float,
                        help='',
                        default=0.001)

    parser.add_argument('--gpuID', type=str,
                        help='gpu ID',
                        default='0')

    parser.add_argument('--number_epoch', type=int,
                        help='',
                        default=1)

    parser.add_argument('--train_data', type=str,
                        help='newsID 2 newsTags ',
                        default='data')
    parser.add_argument('--weight_decay_rate', type=float,
                        help='',
                        default=0.0001)

    parser.add_argument('--task', type=str,
                        help='label of the task',
                        default='')

    return parser.parse_args(argv)


def get_model_info(path):
    file = open(path, 'r')
    result = dict()
    lines = file.read().strip().split("\n")
    for line in lines:
        tmp = line.split("=")
        result[tmp[0]] = tmp[1]
    file.close()
    return result


def write_model_infor(dic, path):
    file = open(path, "w")
    for key in dic.keys():
        line = key + "=" + dic[key] + "\n"
        file.write(line)
    file.close()

def write_episode_reward_infor(dic, path):
    file = open(path, "a")
    for key in dic.keys():
        line = key + "=" + dic[key] + "\n"
        file.write(line)
    file.close()


def get_model_filename(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    # elif len(meta_files) > 1:
    #     raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[-1]
    p = meta_file.rfind('.')
    model_file = model_dir + '/' + meta_file[0:p]
    return model_file


def train(sess, userEnv, actor_Q, saver, summary_writer, checkpoint_path, args,replay_buffer):
    # Initialize target network weights
    # actor_Q.update_target_network()

    # Initialize replay memory
    # replay_buffer = replay.ReplayBuffer(args.buffer_size,
    #                                     hypes.random_seed)

    epoch = 0
    train_time = 0
    global_step = -1
    uid_count = 0
    batch_step = 0
    add_step = 0

    episode_reward_total=0.
    episode_count=0

    vc = 0
    nvc = 0

    count_state=0
    count_no_click=0

    while epoch < args.number_epoch:
        state = userEnv.reset()
        uid_count += 1
        if userEnv.one_epoch_done:
            epoch += 1
            uid_count = 0

        if len(userEnv.episode_record) == 0:
            #print('the episode is empty')
            continue  # number of valid data is 0
        if(userEnv.episode_record[0].current_window == "false"):
            print('episode_record[0].current_window == "false')
            continue

        # if(uid_count<=3):
        #     continue
        '''
        while uid_count < 100:
            state=userEnv.reset()
            uid_count+=1
        '''
        # one step
        episode_reward = 0

        for t in itertools.count():
            start_time = time.time()
            # read action from user record
            picked_newsIDs = userEnv.episode_record[t].recommend_newsIDs
            valid = True
            action = []
            for id in picked_newsIDs:
                if id not in userEnv.episode_record[t].indexSeq:
                    valid = False
                    print('action newsID is not in indexSeq')
                    break
                else:
                    if id not in userEnv.news_dict:
                        valid = False
                        print('action newsID has no newsTag')
                        break
                    a = userEnv.episode_record[t].indexSeq.index(id)
                    action.append(a)

            next_state, rewards, done = userEnv.step(action, t)

            rewards_sum=np.sum(rewards)
            episode_reward += pow(hypes.gamma, t) * rewards_sum

            count_state+=1
            if(rewards_sum <= 1):
                count_no_click+=1

            # Keep track of the transition
            if valid == True:
                vc += 1
                replay_buffer.add(
                    Transition(state=state, action=action, reward=rewards, next_state=next_state, done=done))
                add_step += 1
            else:
                nvc += 1
            # Keep adding experience to the memory until
            # there are at least minibatch size samples

            if (replay_buffer.size() >= args.buffer_size*args.start_rate and add_step % (args.batch_size) == 0):
                s_batch, a_batch, r_batch, s2_batch, t_batch = replay_buffer.sample_batch(args.batch_size)

                y_batch = np.copy(r_batch)
                # t_s = time.time()
                a,_,_ = actor_Q.predict_action(s_batch,action_from_record=a_batch)
                target_step_reward_batch = actor_Q.predict_target_value(s_batch, action_from_record=a_batch,action_from_training=a)
                for i in range(hypes.num_news_action-1):
                    y_batch[:,i] +=actor_Q.gamma*target_step_reward_batch[:,i+1]

                # a_2=actor_Q.predict_best_action_train(s2_batch)
                # target_step_reward_batch_2=actor_Q.predict_target_value(s2_batch, action_from_record=a_batch,action_from_training=a_2)
                # for i in range(args.batch_size):
                #     if len(s2_batch[i].candidates_news) > 10:
                #         y_batch[i,9]+=actor_Q.gamma*target_step_reward_batch_2[i,0]

                _, loss, predicted_q_value, a_rewards, action_train, actor_summary = actor_Q.train(s_batch, a_batch,
                                                                                                   y_batch)
                actor_Q.update_target_network()
                summary_writer.add_summary(actor_summary, global_step)
                global_step += 1

                duration = time.time() - start_time

                a_str = ''
                if(global_step % 50 ==0):
                    # evaluate
                    target_action, target_probs, value_target_estimate = actor_Q.predict_target(s_batch)
                    _, step_choose_probs,value_choose_estimate = actor_Q.predict_action(s_batch,a_batch)

                    build_summary(s_batch, y_batch,
                                  target_probs,step_choose_probs,
                                  summary_writer, global_step)
                    a_str = ''
                    for b in target_action[0]:
                        a_str += str(b) + ','
                print('epoch:', epoch,
                      '\tstep:', global_step,
                      '\tadd_step', add_step,
                      '\tloss_critic:%.3f' % loss,
                      '\ta_predicted:', a_str,
                      '\ttime:%.3f s' % duration,
                      '\tbuffer_size:',replay_buffer.size())

                # if (global_step % 100 == 0):
                #     saver.save(sess, checkpoint_path)

            if done:
                summary = tf.Summary()
                summary.value.add(tag='episode_reward', simple_value=episode_reward)
                episode_reward_total+=episode_reward
                episode_count+=1
                summary_writer.add_summary(summary, global_step)
                break
            state = next_state

    print('vc:', vc,
          '\tnvc',nvc)

    print('count_state: ',count_state,
          '\tcount_no_click',count_no_click,
          '\trate:',count_no_click/(count_state+1))


    if (episode_count > 10):
        episode_reward_ave= episode_reward_total/episode_count
    else:
        episode_reward_ave=0
        print("episode_count<10")
    return episode_reward_ave

def build_summary(s_batch, y_batch,
                  target_probs,step_choose_probs,
                  summary_writer, global_step):

    y_ave=[]
    target_probs_ave=[]
    step_choose_probs_ave=[]
    tq_c = 0

    for i, s in enumerate(s_batch):
        if s_batch[i].position == 1:
            target_probs_ave.append(target_probs[i])
            step_choose_probs_ave.append(step_choose_probs[i])
            y_ave.append(y_batch[i])
            tq_c += 1
    if tq_c >= 1:
        target_probs_ave=np.mean(np.array(target_probs_ave),axis=0)
        step_choose_probs_ave = np.mean(np.array(step_choose_probs_ave), axis=0)
        y_ave=np.mean(np.array(y_batch),axis=0)

        summary_tq = tf.Summary()
        for i in range(hypes.num_news_action):
            summary_tq.value.add(tag='target_probs_ave/target_probs_ave_'+str(i), simple_value=target_probs_ave[i])
            summary_tq.value.add(tag='step_choose_probs_ave/step_choose_probs_ave_' + str(i), simple_value=step_choose_probs_ave[i])
            summary_tq.value.add(tag='y/y_ave_' + str(i),
                                 simple_value=y_ave[i])

        summary_writer.add_summary(summary_tq, global_step)

def evaluation(sess, userEnv, actor, time_size):
    epoch = 0
    test_time = 0
    global_step = 0
    uid_count = 0

    while epoch < hypes.number_epoch and global_step < time_size:
        state = userEnv.reset()
        uid_count += 1
        print('epoch: ', epoch, '\tuid_count: ', uid_count)
        '''
        while uid_count < 100:
            state=userEnv.reset()
            uid_count+=1
        '''
        if len(userEnv.episode_record) == 0:
            print("empty episode!")
            continue  # number of valid data is 0

        # one step
        for t in itertools.count():
            # read action from user record
            picked_newsIDs = userEnv.episode_record[t].recommend_newsIDs
            valid = True
            action = []
            for id in picked_newsIDs:
                if (id not in userEnv.episode_record[t].indexSeq):
                    valid = False
                    break
                else:
                    a = userEnv.episode_record[t].indexSeq.index(id)
                    action.append(a)

            if valid == False:
                print('the episode is not valid,')
                break

            next_state, reward, done = userEnv.step(action, t)

            t_s = time.time()
            target_action, target_probs, value_estimate = actor.predict_target(state)
            t_e = time.time() - t_s
            if global_step >= 5:
                test_time += t_e
            global_step += 1

            probs_str = ''
            for p in target_probs:
                probs_str += '{:.4f}'.format(p) + ','

            print('action_time:%.3f' % t_e,
                  '\tvalue_estimate:%.3f:' % value_estimate,
                  '\ttarget_action:', target_action)

            if (global_step == time_size):
                print('global_step:', global_step,
                      '\ttime average:', test_time / (global_step - 5))
                break
            if done:
                break
            state = next_state
        if userEnv.one_epoch_done:
            epoch += 1
            uid_count = 0


def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=None)
    return output_graph_def


def saveAsPb(sess, path):
    gd = sess.graph.as_graph_def()
    for node in gd.node:

        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, gd,
        output_node_names=["train/Actor_1/target_Q_net/transpose_2","train/Actor_1/target_Q_net/transpose_3"],
        variable_names_whitelist=None)
    tf.train.write_graph(output_graph_def, path, "graph.pb", as_text=False)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuID  # GPU

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction, allow_growth=False)
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=False))
    #print("allocate gpu")
    with tf.variable_scope("train") as scope:
        actor_Q = ddqn.ActorPickNewsQ(learning_rate=args.policy_lr_rate,
                                      tau=args.tau,
                                      gamma=hypes.gamma,
                                      batch_size=args.batch_size,
                                      num_pre_vars=0,
                                      optimizer=args.optimizer,
                                      max_candidates_num=hypes.max_candidates_num,
                                      max_userTags_num=hypes.max_userTags_num,
                                      cliped_norm=args.cliped_norm,
                                      weight_decay_rate=args.weight_decay_rate)

        scope.reuse_variables()
        actor_Q_test = ddqn.ActorPickNewsQ(learning_rate=args.policy_lr_rate,
                                           tau=args.tau,
                                           gamma=hypes.gamma,
                                           batch_size=1,
                                           num_pre_vars=0,
                                           optimizer=args.optimizer,
                                           max_candidates_num=hypes.max_candidates_num,
                                           max_userTags_num=hypes.max_userTags_num,
                                           cliped_norm=args.cliped_norm)

    # Create a saver
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=2)

    model_save_path = args.saved_model_dir
    current_ckpt_path = model_save_path + "/ckpt/online/model.ckpt"
    print("current_ckpt_path:", current_ckpt_path)
    current_pb_path = model_save_path + "/pb/online/"
    model_property_path = model_save_path + "/properties"
    model_property = get_model_info(model_property_path)

    train_data = args.train_data

    logs_dir = args.logs_dir
    #summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)

    #replay_buffer定义在循环外部，可以融合多个时间段的数据
    replay_buffer = replay.ReplayBuffer(args.buffer_size,hypes.random_seed)

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        if args.pretrained_model != '' and os.path.exists(args.pretrained_model + '/checkpoint') == True:
            print('restore form:', current_ckpt_path)
            saver.restore(sess, current_ckpt_path)

        while True:
            old_model = model_property["model_time"]
            files = Console.getDirFiles(train_data)
            train_files = Console.getTrainFiles(files, old_model)
            #random.shuffle(train_files)
            print("train_files:",train_files)

            if len(train_files) < 1:
                print("no data,waiting.......")
                time.sleep(10 * 60)
            else:
                for file_num in range(len(train_files)):
                    start_time = time.time()

                    train_file_tmp = train_files[file_num]
                    #old_model = train_file_tmp.split("/")[-1]
                    _,old_model=os.path.split(train_file_tmp)
                    print("#### line 401\t", train_file_tmp, "\t", old_model)
                    model_timeline_ckpt = model_save_path + "/ckpt/timeline/" + old_model + "/model.ckpt"
                    model_timeline_pb = model_save_path + "/pb/timeline/" + old_model

                    newsTag_dict_path = train_file_tmp + "/newsOnehot"
                    user_records_path = train_file_tmp + "/userOnehot"

                    if os.path.exists(newsTag_dict_path) == False or os.path.exists(user_records_path) == False:
                        print('There is no data file')
                        shutil.rmtree(train_file_tmp)
                        continue

                    data_path = env.DataPaths(newsTag_dict_path, user_records_path)
                    userEnv = env.Environment(data_path, max_newsTag_num=hypes.max_newsTag_num,
                                              max_userTags_num=hypes.max_userTags_num,
                                              max_candidates_num=hypes.max_candidates_num)

                    print('traing....')
                    # now = time.strftime('%Y-%m-%d_%H_%M')
                    now_log_dir=os.path.join(logs_dir,old_model)
                    summary_writer = tf.summary.FileWriter(now_log_dir, sess.graph)

                    episode_reward_ave=train(sess, userEnv, actor_Q, saver, summary_writer, current_ckpt_path, args,replay_buffer)

                    print('write_episode_reward_infor:', old_model)
                    episode_reward_info = {}
                    episode_reward_info[old_model] = "{0:.4f}".format(episode_reward_ave)
                    write_episode_reward_infor(episode_reward_info, model_save_path + "/episode_rewards.txt")

                    if(episode_reward_ave < 2): #数据有问题，直接删除，不更新模型,关闭训练
                        print("there're serious problems in the data")
                        shutil.rmtree(train_file_tmp)
                        exit(-1)

                    print('save model:', current_ckpt_path)
                    saver.save(sess, current_ckpt_path)
                    print('save model:', model_timeline_ckpt)
                    saver.save(sess, model_timeline_ckpt)

                    print('save pb:', current_pb_path)
                    saveAsPb(sess, current_pb_path)
                    print('save pb:', model_timeline_pb)
                    saveAsPb(sess, model_timeline_pb)

                    print('updateVideoChannelModel....')

                    #online
                    # shell_path = "/data/bd-recommend/lizhenmao/transfer_log_newOldLongTagMix_tagShared/updateVideoChannelModel.sh"
                    # os.system(shell_path)

                    #dev
                    shell_path = "/data/bd-recommend/lizhenmao/dev_transfer_log/updateVideoChannelModel.sh"
                    os.system(shell_path)

                    #print('testing....')
                    # evaluation(sess, userEnv, actor_Q, args.average_size)

                    model_property["model_time"] = train_file_tmp.split("/")[-1]
                    print('write_model_property:', model_property["model_time"])
                    write_model_infor(model_property, model_property_path)

                    duration = time.time() - start_time
                    print("training time:%.3f s" % duration)

                # online 注释去掉
                Console.delete_datas(files, MAX_DATA_FILE)

                timeline_ckpt_files = Console.getDirFiles(model_save_path + "/ckpt/timeline/")
                timeline_pb_files = Console.getDirFiles(model_save_path + "/pb/timeline/")
                Console.delete_datas(timeline_ckpt_files, MAX_DATA_FILE)
                Console.delete_datas(timeline_pb_files, MAX_DATA_FILE)

                timeline_log_files=Console.getDirFiles(logs_dir) #删除log文件
                Console.delete_datas(timeline_log_files, MAX_LOG_FILE)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))