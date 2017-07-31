import sys
from env import Env, get_view, get_reward
from Model import Model_DNN
import argparse
import tensorflow as tf
import os
import shutil
import time

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    # Environment
    argparser.add_argument('--add_pig_number', type=int, default=500)
    argparser.add_argument('--add_rabbit_number', type=int, default=500)
    argparser.add_argument('--add_every', type=int, default=500)

    argparser.add_argument('--random_seed', type=int, default=10,
                           help='the random seed to generate the wall in the map')
    argparser.add_argument('--width', type=int, default=1000)
    argparser.add_argument('--height', type=int, default=1000)
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--view_args', type=str, default='2500-5-5-0,2500-5-5-1,2500-5-5-2,2500-5-5-3',
                           help="num-leftView-frontView-orientation, separated by space")
    argparser.add_argument('--pig_max_number', type=int, default=3000)
    argparser.add_argument('--pig_min_number', type=int, default=1500)
    argparser.add_argument('--pig_increase_every', type=int, default=5)
    argparser.add_argument('--pig_increase_rate', type=float, default=0.001)
    argparser.add_argument('--rabbit_increase_rate', type=float, default=0.001)
    argparser.add_argument('--rabbit_max_number', type=int, default=3000)
    argparser.add_argument('--agent_increase_rate', type=float, default=0.001)
    argparser.add_argument('--pig_increase_policy', type=int, default=1,
                           help='0: max_min; 1: increase every n timestep')
    argparser.add_argument('--reward_radius_rabbit', type=int, default=7)
    argparser.add_argument('--reward_radius_pig', type=int, default=7)
    argparser.add_argument('--reward_threshold_pig', type=int, default=5)
    argparser.add_argument('--img_length', type=int, default=5)
    argparser.add_argument('--images_dir', type=str, default='images')
    argparser.add_argument('--agent_mortal', type=int, default=0,
                           help='0: immortal, 1: mortal')
    argparser.add_argument('--agent_emb_dim', type=int, default=5)
    argparser.add_argument('--agent_id', type=int, default=0,
                           help='0: no id, 1: has id')
    argparser.add_argument('--damage_per_step', type=float, default=0.)
    # model
    argparser.add_argument('--model_name', type=str, default='DNN')
    argparser.add_argument('--model_hidden_size', type=str, default='32,32')
    argparser.add_argument('--activations', type=str, default='sigmoid,sigmoid')
    argparser.add_argument('--view_flat_size', type=int, default=66 * 5 + 5)
    argparser.add_argument('--num_actions', type=int, default=9)
    argparser.add_argument('--reward_decay', type=float, default=0.9)
    argparser.add_argument('--save_every_round', type=int, default=10)
    argparser.add_argument('--save_dir', type=str, default='models')
    argparser.add_argument('--load_dir', type=str, default=None,
                           help='e.g. models/round_0/model.ckpt')
    # Train
    argparser.add_argument('--video_dir', type=str, default='videos')
    argparser.add_argument('--video_per_round', type=int, default=None)
    argparser.add_argument('--round', type=int, default=100)
    argparser.add_argument('--time_step', type=int, default=10)
    argparser.add_argument('--policy', type=str, default='e_greedy')
    argparser.add_argument('--epsilon', type=float, default=0.1)
    argparser.add_argument('--agent_number', type=int, default=100)
    argparser.add_argument('--learning_rate', type=float, default=0.001)
    argparser.add_argument('--log_file', type=str, default='log.txt')
    argv = argparser.parse_args()

    argv.model_hidden_size = [int(x) for x in argv.model_hidden_size.split(',')]
    argv.view_args = [x for x in argv.view_args.split(',')]
    argv.activations = [x for x in argv.activations.split(',')]
    if argv.load_dir == 'None':
        argv.load_dir = None

    env = Env(argv)
    model = Model_DNN(argv)

    # Environment Initialization
    env.gen_wall(0.02, seed=argv.random_seed)
    env.gen_agent(argv.agent_number)

    env.gen_pig(argv.pig_max_number)
    # env.gen_rabbit(argv.rabbit_max_number)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    if argv.load_dir:
        model.load(sess, argv.load_dir)
        print 'Load model from ' + argv.load_dir

    if not os.path.exists(argv.images_dir):
        os.mkdir(argv.images_dir)
    if not os.path.exists(argv.video_dir):
        os.mkdir(argv.video_dir)
    if not os.path.exists(argv.save_dir):
        os.mkdir(argv.save_dir)

    flip = 0

    log = open(argv.log_file, 'w')
    log_largest_group = open('log_largest_group.txt', 'w')
    for r in xrange(argv.round):
        video_flag = False
        if argv.video_per_round > 0 and r % argv.video_per_round == 0:
            video_flag = True
            img_dir = os.path.join(argv.images_dir, str(r))
            try:
                os.makedirs(img_dir)
            except:
                shutil.rmtree(img_dir)
                os.makedirs(img_dir)
        for t in xrange(argv.time_step):
            if t == 0 and video_flag:
                env.dump_image(os.path.join(img_dir, '%d.png' % t))

            view_batches = get_view(env)  # s
            actions, actions_batches = model.infer_actions(sess, view_batches, policy=argv.policy,
                                                           epsilon=argv.epsilon)  # a

            env.take_action(actions)
            env.decrease_health()
            env.update_pig_pos()
            # env.update_rabbit_pos()

            if video_flag:
                env.dump_image(os.path.join(img_dir, '%d.png' % (t + 1)))

            rewards = get_reward(env)  # r, a dictionary
            env.increase_health(rewards)
            total_reward = 0
            for k, v in rewards.iteritems():
                total_reward += v

            new_view_batches = get_view(env)  # s'
            maxQ_batches = model.infer_max_action_values(sess, new_view_batches)

            model.train(sess=sess,
                        view_batches=view_batches,
                        actions_batches=actions_batches,
                        rewards=rewards,
                        maxQ_batches=maxQ_batches,
                        learning_rate=argv.learning_rate)

            dead_list = env.remove_dead_people(r * argv.time_step + t)
            model.remove_dead_agent_emb(dead_list)  # remove agent embeddings

            cur_pig_num = env.get_pig_num()
            cur_rabbit_num = env.get_rabbit_num()
            group_num, mean_size, variance_size, max_size, group_view_num = env.group_monitor()
            info = 'Round\t%d\ttimestep\t%d\tPigNum\t%d\tgroup_num\t%d\tmean_size\t%f\tvariance_size\t%f\tmax_group_size\t%d\trabbitNum\t%d' % \
                   (r, t, cur_pig_num, group_num, mean_size, variance_size, max_size, cur_rabbit_num)
            if group_view_num is not None:
                for k in group_view_num:
                    x = map(int, k[1:-1].split(','))
                    group_view_info = '\tView\t%d\tnumber\t%d' % (
                        (2 * x[0] + 1) * (x[1] + 1), group_view_num[k])
                    info += group_view_info
                    print group_view_info
            info += '\tagent_num\t%d' % env.get_agent_num()

            join_num = 0
            leave_num = 0
            for item in actions:
                if item[1] == 7:
                    join_num += 1
                elif item[1] == 8:
                    leave_num += 1
            join_num = 1.0 * join_num / len(actions)
            leave_num = 1.0 * leave_num / len(actions)
            info += '\tjoin_ratio\t%f\tleave_ratio\t%f' % (join_num, leave_num)
            ret_loss = model.get_loss()
            info += '\tloss\t%f' % (ret_loss)
            info += '\treward\t%f' % (total_reward)
            # divided by the number of pig number to normalize
            normalize_total_reward = 1. * total_reward / env.get_pig_num()
            info += '\tnorm_reward\t%f' % (normalize_total_reward)
            avg_life, dead_people = env.get_avg_life()
            info += '\tavg_life\t%f' % (avg_life)
            info += '\ttotal_dead_people\t%f' % (dead_people)

            print info

            # print 'average degroup number:\t', avg_degroup
            log.write(info + '\n')
            log.flush()

            # largest_group_pos = env.track_largest_group(time_step=r * argv.round + t, update_largest_every=200)
            # pos_info = []
            # for item in largest_group_pos:
            #     pos_info.append(str(item[0]) + ',' + str(item[1]))
            # log_largest_group.write('\t'.join(pos_info) + '\n')
            # log_largest_group.flush()

            if argv.pig_increase_policy == 0:
                if cur_pig_num < argv.pig_min_number:
                    env.gen_pig(argv.pig_max_number - cur_pig_num)
            elif argv.pig_increase_policy == 1:
                if t % argv.pig_increase_every == 0:
                    env.gen_pig(max(1, int(env.get_pig_num() * argv.pig_increase_rate)))
            elif argv.pig_increase_policy == 2:
                env.gen_pig(10)

            # env.gen_rabbit(max(10, int(env.get_rabbit_num() * argv.rabbit_increase_rate)))
            env.grow_agent(max(1, int(env.get_agent_num() * argv.agent_increase_rate)), r * argv.time_step + t)

            # if (r * argv.time_step + t) % argv.add_every == 0:
            #     if flip:
            #         env.gen_pig(argv.add_pig_number)
            #     else:
            #         env.gen_rabbit(argv.add_rabbit_number)
            #     flip ^= 1

            #if flip:
            #    if env.get_rabbit_num() < 1000:
            #        env.gen_pig(argv.pig_max_number - env.get_pig_num())
            #        flip ^= 1
            #else:
            #    if env.get_pig_num() < 2000:
            #        env.gen_rabbit(argv.rabbit_max_number - env.get_rabbit_num())
            #        flip ^= 1

        if argv.save_every_round and r % argv.save_every_round == 0:
            if not os.path.exists(os.path.join(argv.save_dir, "round_%d" % r)):
                os.mkdir(os.path.join(argv.save_dir, "round_%d" % r))
            model_path = os.path.join(argv.save_dir, "round_%d" % r, "model.ckpt")
            model.save(sess, model_path)
            print 'model saved into ' + model_path
        if video_flag:
            images = [os.path.join(img_dir, ("%d.png" % i)) for i in range(argv.time_step + 1)]
            env.make_video(images=images, outvid=os.path.join(argv.video_dir, "%d.avi" % r))
    log.close()
