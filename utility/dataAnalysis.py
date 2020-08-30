from utility.hype_parameters import HypeParameters as hypes
import argparse
import sys
from environment import environment_10_newsTags_tagShared_trueDQN as env
import os

def trainStrToDict(line):
    result = dict()
    if "".__eq__(line):
        return result

    tmp = line.strip().split("\t")
    for v in tmp:
        if "".__eq__(v):
            continue
        tmp = v.strip().split("#")
        result[tmp[0]] = tmp[1]

    return result


def userTags_statistic(userData_path):
    userData_file = open(userData_path, 'r')
    f = open("userTags.txt", 'w')
    num_user = 0
    max_num_userTags = 0
    num_userTags_total = 0
    while (True):
        line = userData_file.readline().strip('\n').strip()
        # pre_point = self.user_file.tell()
        if not "".__eq__(line):
            try:
                elem = trainStrToDict(line)
                uid = elem["uid"]
                state = elem["state"]
                feature = elem["feature"]
                features = feature.split(" ")
                instant_tag = []
                short_tag = []
                long_tag_old = []
                long_tag_new = []
                for t in features:
                    tag, tag_w, tag_type = t.split(":")
                    if tag_type == '1':
                        instant_tag.append(t)
                    elif tag_type == '2':
                        short_tag.append(t)
                    elif tag_type == '3':
                        long_tag_old.append(t)
                    elif tag_type == '5':
                        long_tag_new.append(t)

                num_userTags = len(features)
                max_num_userTags = max(max_num_userTags, num_userTags)
                num_userTags_total += num_userTags
                num_user += 1
                norm=len(instant_tag)+len(short_tag)+len(long_tag_old)+len(long_tag_new)
                print('processing ', num_user)
                if (num_userTags >= 0):
                    f.write(str(num_user) + '\t' +
                            uid + '\t' + str(num_userTags) + '\t' +
                            str(len(instant_tag)/norm) + '\t' +
                            str(len(short_tag)/norm) + '\t' +
                            str(len(long_tag_old)/norm) + '\t' +
                            str(len(long_tag_new)/norm) + '\t' +
                            "\n")
            except:
                print('read data error: ', line)
        else:
            print("one epoch done")
            break

    print("max: ", max_num_userTags)
    print("ave: ", num_userTags_total / num_user)
    f.close()


def indexSeq_statistic(userEnv):
    num_indexSeq_total = 0
    max_num_indexSeq = -10

    num_request_total = 0
    max_num_request = -10

    num_user = 0.
    epoch = 0
    uid_count = 0

    count_indseq = 0

    f = open("userIndexSeqs.txt", 'w')
    while epoch < 1:
        # t_read_s=time.time()
        _ = userEnv.reset()
        # t_read_e=time.time()-t_read_s
        # print('read time:',t_read_e)

        if userEnv.one_epoch_done:
            epoch += 1
            uid_count = 0

        if len(userEnv.episode_record) == 0:
            #print('the episode is not valid')
            continue  # number of valid data is 0
        else:
            uid_count += 1
            print('epoch: ', epoch, '\tuid_count: ', uid_count,'\tlen:',len(userEnv.episode_record))

        '''
        while uid_count < 100:
            state=userEnv.reset()
            uid_count+=1
        '''

        uid = userEnv.episode_record[0].userID
        num_request = len(userEnv.episode_record)
        num_request_total += num_request
        max_num_request = max(max_num_request, num_request)

        for t in range(len(userEnv.episode_record)):
            num_indexSeq = len(userEnv.episode_record[t].indexSeq)
            max_num_indexSeq = max(max_num_indexSeq, num_indexSeq)
            num_indexSeq_total += num_indexSeq
            num_click=len(userEnv.episode_record[t].click_newsIDs)
            log_time=userEnv.episode_record[t].log_time
            f.write(uid + '\t' + str(t) + '\t' + str(num_indexSeq) +
                    '\t'+str(len(userEnv.episode_record))+
                    '\t'+str(num_click)+
                    '\t'+log_time+"\n")
            count_indseq += 1

        num_user += 1

    print("max: ", max_num_indexSeq)
    print("ave: ", num_indexSeq_total / count_indseq)

    print("max: ", max_num_request)
    print("ave: ", num_request_total / num_user)

    f.close()


def showValidStates(userEnv, args):
    # Initialize target network weights
    # actor_Q.update_target_network()

    epoch = 0
    uid_count = 0

    while epoch < 1:
        # t_read_s=time.time()
        state = userEnv.reset()
        # t_read_e=time.time()-t_read_s
        # print('read time:',t_read_e)
        uid_count += 1
        # print('epoch: ', epoch, '\tuid_count: ', uid_count)

        if userEnv.one_epoch_done:
            epoch += 1
            uid_count = 0

        if len(userEnv.episode_record) == 0:
            # print('the episode is not valid')
            continue  # number of valid data is 0

        # if(uid_count<=3):
        #     continue
        '''
        while uid_count < 100:
            state=userEnv.reset()
            uid_count+=1
        '''

        print('valid states: ', len(userEnv.episode_record))
        episode_reward = 0
        for t in range(len(userEnv.episode_record)):
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

            episode_reward += pow(hypes.gamma, t) * reward
        print('episode_reward: ', episode_reward)


def main(args):
    newsTag_dict_path=args.data_news
    user_records_path=args.data_user
    data_path = env.DataPaths(newsTag_dict_path, user_records_path)
    userEnv = env.Environment(data_path, max_newsTag_num=hypes.max_newsTag_num,
                              max_userTags_num=hypes.max_userTags_num,
                              max_candidates_num=hypes.max_candidates_num)
    # showValidStates(userEnv,args)
    #userTags_statistic(args.user_records_path)
    indexSeq_statistic(userEnv)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_user', type=str,
                        help='',
                        default='D:download/data/2019-01-21_13-00/userOnehot')
    parser.add_argument('--data_news', type=str,
                        help='',
                        default='D:download/data/2019-01-21_13-00/newsOnehot')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
