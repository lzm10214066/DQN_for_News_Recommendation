'''
created by zhenmaoli
2018.2.26
a standalone file for hype parameters
'''

class HypeParameters:
    #data paths,two files for now, for news with content and title
    news_dic_path = '../data/news_content_version/news_contents_dict_20180228'
    user_features_path = '../data/news_content_version/user_records_test'

    #data path,for news with newsTags
    newsTag_dict_path='../data/news_tags_version/20180228/newsTags_idx_20180228'
    user_records_path='../data/news_tags_version/20180228/user_records_20180228'

    #training
    optimizer='ADAM'
    number_epoch=200
    policy_lr_rate=0.01
    value_lr_rate=0.1

    actor_lr_rate=0.01
    critic_lr_rate=0.1
    #save
    model_dir='saved_models'
    logs_dir='logs'

    #history
    history_num=40
    history_dim = 128
    history_number_layers = 2

    #news content
    word_max_len=200
    lstm_size=128
    number_of_layers=1
    vocab_size=480568+2
    word_vector_dim=64

    # user features
    number_tags=286696+4
    tag_dim=128

    #GPU
    gpu_memory_fraction=0.6

    #for tags
    max_tags_num=1000000 #
    tags_emb_size=64
    max_newsTag_num=10
    max_userTags_num=200
    max_candidates_num=500
    #for pre_train
    pretrained_model=''

    #for DDPG
    tau=0.001
    batch_size=32
    gamma=0.9
    buffer_size=50
    random_seed=1234

    #train stratagy
    record_multiply=True #if True ,the action in records will multiply the probs predicted
    weight_decay_rate=0.0001

    #action
    num_news_action=10

    #for reward
    weight_for_pv=0.1
    weight_for_continue=0.5
    time_base=2000.0