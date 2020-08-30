import argparse
import collections
import itertools
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.framework import graph_util

import Console
from ddqn import ddqn_model_10_tagShared_trueDQN_debug as ddqn
from environment import environment_10_newsTags_tagShared_trueDQN as env
from utility import replay
from utility.hype_parameters import HypeParameters as hypes
import shutil
import numpy as np
import random

Transition = collections.namedtuple("Transition",
                                    ["state", "action", "reward", "next_state", "done"])
MAX_DATA_FILE = 336
MAX_LOG_FILE=  336


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

    parser.add_argument('--batch_size', type=int,
                        help='batch_size for training',
                        default=64)

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

def build_summary(s_batch, s2_batch,
                  target_probs,step_choose_probs,
                  summary_writer, global_step):

    target_probs_ave=[]
    step_choose_probs_ave=[]
    tq_c = 0

    for i, s in enumerate(s_batch):
        if s_batch[i].position == 1:
            target_probs_ave.append(target_probs[i])
            step_choose_probs_ave.append(step_choose_probs[i])
            tq_c += 1
    if tq_c > 1:
        target_probs_ave=np.mean(np.array(target_probs_ave),axis=0)
        step_choose_probs_ave = np.mean(np.array(step_choose_probs_ave), axis=0)


    summary_tq = tf.Summary()
    for i in range(hypes.num_news_action):
        summary_tq.value.add(tag='target_probs_ave/target_probs_ave_'+str(i), simple_value=target_probs_ave[i])
        summary_tq.value.add(tag='step_choose_probs_ave/step_choose_probs_ave_' + str(i), simple_value=step_choose_probs_ave[i])

    summary_writer.add_summary(summary_tq, global_step)

def evaluation(sess, userEnv, actor, time_size,args):
    epoch = 0
    test_time = 0
    global_step = 0
    uid_count = 0

    while epoch < hypes.number_epoch and global_step < time_size:
        state = userEnv.reset()
        uid_count += 1
        #print('epoch: ', epoch, '\tuid_count: ', uid_count)

        if len(userEnv.episode_record) == 0:
            #print("empty episode!")
            continue  # number of valid data is 0
        if (userEnv.episode_record[0].current_window == "false"):
            print('episode_record[0].current_window == "false')
            continue
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
            target_action, target_probs, value_estimate ,target_action_probs_mat= actor.predict_target([state])
            t_e = time.time() - t_s
            if global_step >= 5:
                test_time += t_e
            global_step += 1

            a_set=set(target_action[0])
            if len(a_set) < 10:
                print('Duplicated actions:')
                print(target_action_probs_mat[9])

            probs_str = ''
            for p in target_probs[0]:
                probs_str += '{:.4f}'.format(p) + ','

            print('action_time:%.3f' % t_e,
                  '\tprobs_str:',probs_str,
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
                                      cliped_norm=args.cliped_norm)

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
                    train_file_tmp = train_files[file_num]
                    #old_model = train_file_tmp.split("/")[-1]
                    _,old_model=os.path.split(train_file_tmp)


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


                    print('test....')

                    evaluation(sess, userEnv, actor_Q_test, 10000000,args)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))