import os
import random
import multiprocessing
from PIL import Image
import numpy as np
from cv2 import VideoWriter, imread, resize
from copy import deepcopy
import cv2


# from model import inference, train

class Env(object):
    def __init__(self, args):
        self.args = args
        self.h = args.height
        self.w = args.width
        self.batch_size = args.batch_size
        self.view_args = args.view_args
        self.agent_num = args.agent_number
        self.pig_num = 0
        self.rabbit_num = 0
        self.action_num = args.num_actions

        # Initialization
        self.view = []
        self.map = np.zeros((self.h, self.w), dtype=np.int32)
        self.id_pos = {}
        self.pig_pos = set()
        self.property = {}
        self.birth_year = {}
        self.rabbit_pos = set()

        # For the view size modify
        self.property_copy = {}
        self.max_group = 0
        self.id_group = {}
        self.group_ids = {}
        self.batch_views = {}
        self.ally = {}

        # For reason of degroup
        self.id_ally_number = {}
        self.actions = None

        # For health
        self.health = {}
        self.max_id = 0

        # For mortal
        self.dead_id = []
        # Record the avg_life of the dead people in current time step
        self.avg_life = None
        self.dead_people = None
        # A map: year -> (avg_life, live_year)
        self.avg_life = {}

        # For track largest group
        self.largest_group = 0

        self.rewards = None
        self.reward_radius_pig = args.reward_radius_pig
        self.reward_threshold_pig = args.reward_threshold_pig
        self.reward_radius_rabbit = args.reward_radius_rabbit

        self.groups_view_size = {}
        self.max_view_size = None
        self.min_view_size = None

        self._init_property()
        self._init_group()

    def _init_property(self):
        self.property[-3] = [1, [0, 1, 0]]
        self.property[-2] = [1, [1, 0, 0]]
        self.property[-1] = [1, [0.411, 0.411, 0.411]]
        self.property[0] = [1, [0, 0, 0]]

    def _init_group(self):
        for i in xrange(self.agent_num):
            self.id_group[i + 1] = 0

    def _gen_power(self, cnt):

        def max_view_size(view_size1, view_size2):
            view_size_area1 = (2 * view_size1[0] + 1) * (view_size1[1] + 1)
            view_size_area2 = (2 * view_size2[0] + 1) * (view_size2[1] + 1)

            return view_size1 if view_size_area1 > view_size_area2 else view_size2

        def min_view_size(view_size1, view_size2):
            view_size_area1 = (2 * view_size1[0] + 1) * (view_size1[1] + 1)
            view_size_area2 = (2 * view_size2[0] + 1) * (view_size2[1] + 1)

            return view_size1 if view_size_area1 < view_size_area2 else view_size2

        cur = 0
        for k in self.view_args:
            k = [int(x) for x in k.split('-')]
            assert len(k) == 4

            num, power_list = k[0], k[1:]
            # Maintain the max_view_size
            if self.max_view_size is None:
                self.max_view_size = power_list
            else:
                self.max_view_size = max_view_size(self.max_view_size, power_list)

            if self.min_view_size is None:
                self.min_view_size = power_list
            else:
                self.min_view_size = min_view_size(self.min_view_size, power_list)

            cur += num

            if cnt <= cur:
                return power_list

    def gen_wall(self, prob=0, seed=10):
        if prob == 0:
            return
        np.random.seed(seed)
        # Generate wall according to the prob
        for i in xrange(self.h):
            for j in xrange(self.w):
                if i == 0 or i == self.h - 1 or j == 0 or j == self.w - 1:
                    self.map[i][j] = -1
                    continue
                wall_prob = np.random.rand()
                if wall_prob < prob:
                    self.map[i][j] = -1

    def gen_agent(self, agent_num=None):
        if agent_num == None:
            agent_num = self.args.agent_number

        for i in xrange(agent_num):
            while True:
                x = np.random.randint(0, self.h)
                y = np.random.randint(0, self.w)
                if self.map[x][y] == 0:
                    self.map[x][y] = i + 1
                    self.id_pos[i + 1] = (x, y)
                    self.property[i + 1] = [self._gen_power(i + 1), [0, 0, 1]]
                    self.health[i + 1] = 1.0
                    # Record the birthday of any agent
                    self.birth_year[i + 1] = 0
                    break
        assert (2 * self.max_view_size[0] + 1) * (self.max_view_size[1] + 1) * 5 + self.args.agent_emb_dim == \
               self.args.view_flat_size

        self.agent_num = self.args.agent_number
        self.max_id = self.args.agent_number
        # self.property_copy = self.property[:]
        for k in self.property:
            self.property_copy[k] = self.property[k][:]
            # self.property_copy = deepcopy(self.property)

    def _grow_power(self):

        candidate_view = []
        for k in self.view_args:
            k = [int(x) for x in k.split('-')]
            assert len(k) == 4
            candidate_view.append(k)

        num = len(candidate_view)
        random_power = np.random.randint(0, num)

        return candidate_view[random_power][1:]

    def grow_agent(self, agent_num=0, cur_step=-1):
        if agent_num == 0:
            return

        for i in xrange(agent_num):
            while True:
                x = np.random.randint(0, self.h)
                y = np.random.randint(0, self.w)
                if self.map[x][y] == 0:
                    self.max_id += 1
                    self.map[x][y] = self.max_id
                    self.id_pos[self.max_id] = (x, y)
                    self.property[self.max_id] = [self._grow_power(), [0, 0, 1]]
                    self.property_copy[self.max_id] = self.property[self.max_id][:]
                    self.health[self.max_id] = 1.0
                    self.id_group[self.max_id] = 0
                    # Record the birthday of the new agent
                    self.birth_year[self.max_id] = cur_step
                    break

        self.agent_num += agent_num

    def gen_pig(self, pig_nums=None):
        if pig_nums == None:
            pig_nums = self.args.pig_max_number

        for i in xrange(pig_nums):
            while True:
                x = np.random.randint(0, self.h)
                y = np.random.randint(0, self.w)
                if self.map[x][y] == 0:
                    self.map[x][y] = -2
                    self.pig_pos.add((x, y))
                    break

        self.pig_num = self.pig_num + pig_nums

    def gen_rabbit(self, rabbit_num=None):
        if rabbit_num is None:
            rabbit_num = self.args.rabbit_max_number

        for i in xrange(rabbit_num):
            while True:
                x = np.random.randint(0, self.h)
                y = np.random.randint(0, self.w)
                if self.map[x][y] == 0:
                    self.map[x][y] = -3
                    self.rabbit_pos.add((x, y))
                    break

        self.rabbit_num = self.rabbit_num + rabbit_num

    def get_pig_num(self):
        return self.pig_num

    def get_rabbit_num(self):
        return self.rabbit_num

    def get_agent_num(self):
        return self.agent_num

    def _agent_act(self, x, y, face, action, id):

        def move_forward(x, y, face):
            if face == 0:
                return x - 1, y
            elif face == 1:
                return x, y + 1
            elif face == 2:
                return x + 1, y
            elif face == 3:
                return x, y - 1

        def move_backward(x, y, face):
            if face == 0:
                return x + 1, y
            elif face == 1:
                return x, y - 1
            elif face == 2:
                return x - 1, y
            elif face == 3:
                return x, y + 1

        def move_left(x, y, face):
            if face == 0:
                return x, y - 1
            elif face == 1:
                return x - 1, y
            elif face == 2:
                return x, y + 1
            elif face == 3:
                return x + 1, y

        def move_right(x, y, face):
            if face == 0:
                return x, y + 1
            elif face == 1:
                return x + 1, y
            elif face == 2:
                return x, y - 1
            elif face == 3:
                return x - 1, y

        def in_board(x, y):
            return self.map[x][y] == 0

        # return the max view size(the area of the view) of the two view sizes
        def max_view_size(view_size1, view_size2):
            view_size_area1 = (2 * view_size1[0] + 1) * (view_size1[1] + 1)
            view_size_area2 = (2 * view_size2[0] + 1) * (view_size2[1] + 1)

            return view_size1 if view_size_area1 > view_size_area2 else view_size2

        if action == 0:
            pass
        elif action == 1:
            new_x, new_y = move_forward(x, y, face)
            if in_board(new_x, new_y):
                self.map[x][y] = 0
                self.map[new_x][new_y] = id
                self.id_pos[id] = (new_x, new_y)
        elif action == 2:
            new_x, new_y = move_backward(x, y, face)
            if in_board(new_x, new_y):
                self.map[x][y] = 0
                self.map[new_x][new_y] = id
                self.id_pos[id] = (new_x, new_y)
        elif action == 3:
            new_x, new_y = move_left(x, y, face)
            if in_board(new_x, new_y):
                self.map[x][y] = 0
                self.map[new_x][new_y] = id
                self.id_pos[id] = (new_x, new_y)
        elif action == 4:
            new_x, new_y = move_right(x, y, face)
            if in_board(new_x, new_y):
                self.map[x][y] = 0
                self.map[new_x][new_y] = id
                self.id_pos[id] = (new_x, new_y)
        elif action == 5:
            self.property[id][0][2] = (face + 4 - 1) % 4
        elif action == 6:
            self.property[id][0][2] = (face + 1) % 4
        elif action == 7:
            if self.id_group[id] == 0:
                if id in self.ally:
                    ally_id = self.ally[id]
                    if self.id_group[ally_id] == 0:
                        self.max_group += 1
                        self.id_group[id] = self.max_group
                        self.id_group[ally_id] = self.max_group

                        self.group_ids[self.max_group] = []
                        self.group_ids[self.max_group].append(id)
                        self.group_ids[self.max_group].append(ally_id)

                        # For view size
                        assert self.property[id][0] == self.property_copy[id][0]
                        assert self.property[ally_id][0] == self.property_copy[ally_id][0]
                        self.groups_view_size[self.max_group] = max_view_size(self.property[id][0],
                                                                              self.property[ally_id][0])
                        self.property[id][0] = self.groups_view_size[self.max_group]
                        self.property[ally_id][0] = self.groups_view_size[self.max_group]
                    else:
                        assert self.property[id][0] == self.property_copy[id][0]
                        self.id_group[id] = self.id_group[ally_id]
                        self.group_ids[self.id_group[ally_id]].append(id)

                        group_id = self.id_group[ally_id]

                        cur_max_view_size = max_view_size(self.property[id][0], self.groups_view_size[group_id])
                        if cur_max_view_size == self.property[id][0] and self.property[id][0] != self.groups_view_size[
                            group_id]:
                            # A powerful man join in a group, need to change all the members' view size in that group
                            for people in self.group_ids[group_id]:
                                self.property[people][0] = cur_max_view_size
                            self.groups_view_size[group_id] = cur_max_view_size
                        else:
                            self.property[id][0] = cur_max_view_size

        elif action == 8:
            group_id = self.id_group[id]

            if group_id != 0:
                another_id = None
                if len(self.group_ids[group_id]) == 2:
                    for item in self.group_ids[group_id]:
                        if item != id:
                            another_id = item
                    self.id_group[id], self.id_group[another_id] = 0, 0
                    self.group_ids[group_id] = None

                    # Restore the origin view size
                    self.property[id] = self.property_copy[id][:]
                    self.property[another_id] = self.property_copy[another_id][:]
                    self.groups_view_size[group_id] = None
                else:
                    self.id_group[id] = 0
                    self.group_ids[group_id].remove(id)

                    # Restore the origin view size
                    self.property[id] = self.property_copy[id][:]
                    cur_max_view_size = None

                    for people in self.group_ids[group_id]:
                        if cur_max_view_size is None:
                            cur_max_view_size = self.property_copy[people][0][:]
                        else:
                            cur_max_view_size = max_view_size(cur_max_view_size, self.property_copy[people][0][:])

                    for people in self.group_ids[group_id]:
                        self.property[people][0] = cur_max_view_size

                    self.groups_view_size[group_id] = cur_max_view_size

            else:
                pass

        else:
            print action
            print "Wrong Action ID!!!!"

    def take_action(self, actions):

        # Move Agent
        self.actions = actions
        # for i in xrange(self.agent_num):
        for id, action in actions:
            x, y = self.id_pos[id]
            face = self.property[id][0][2]
            self._agent_act(x, y, face, action, id)

    def increase_health(self, rewards):
        for id in rewards:
            self.health[id] += 12. * rewards[id]

            # if rewards[id] > 0.2:
            #     self.health[id] = 1.
            # elif rewards > 0:
            #     self.health[id] += rewards[id]

            # self.health[id] += rewards[id]
            # if self.health[id] > 1.0:
            #     self.health[id] = 1.0

    def group_monitor(self):
        """
        :return: group_num, mean_size, variance_size, max_size 
        """
        group_sizes = []
        group_view_num = {}
        group_view_avg_size = {}
        for k in self.group_ids:
            ids = self.group_ids[k]
            if ids:
                group_size = len(ids)
                assert group_size >= 2
                group_sizes.append(group_size)

                # count group view size and group number
                group_view = self.groups_view_size[k]
                group_view = group_view[:2]
                if str(group_view) not in group_view_num:
                    group_view_num[str(group_view)] = 1
                else:
                    group_view_num[str(group_view)] += 1
                if str(group_view) not in group_view_avg_size:
                    group_view_avg_size[str(group_view)] = group_size
                else:
                    group_view_avg_size[str(group_view)] += group_size

        group_sizes = np.array(group_sizes)
        for k in group_view_avg_size:
            group_view_avg_size[k] = 1. * group_view_avg_size[k] / group_view_num[k]

        # For reason of degroup
        # cnt = 0
        # cnt_degroup = 0
        #
        # for i, action in enumerate(self.actions):
        #     id = i + 1
        #     if action == 8 and self.id_group[id] > 0:
        #         cnt += 1
        #         if id in self.id_ally_number:
        #             cnt_degroup += self.id_ally_number[id]
        #
        # avg_degroup = 0 if cnt == 0.0 else 1. * cnt_degroup / (1. * cnt)

        if len(group_sizes) > 0:
            return len(group_sizes), group_sizes.mean(), group_sizes.var(), np.max(
                group_sizes), group_view_num
        else:
            return 0, 0, 0, 0, None

    def track_largest_group(self, time_step, update_largest_every):
        if time_step % update_largest_every == 0 or (self.group_ids[self.largest_group] is None):
            self.largest_group_size = 0
            self.largest_group = 0
            for k in self.group_ids:
                ids = self.group_ids[k]
                if ids:
                    if len(ids) > self.largest_group_size:
                        self.largest_group_size = len(ids)
                        self.largest_group = k
        return [self.id_pos[i] for i in self.group_ids[self.largest_group]]

    def update_pig_pos(self):

        def in_board(x, y):
            return not (x < 0 or x >= self.h or y < 0 or y >= self.w)

        # Move Pigs
        for i, item in enumerate(self.pig_pos):
            x, y = item
            direction = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
            np.random.shuffle(direction)
            for pos_x, pos_y in direction:
                if (pos_x, pos_y) == (0, 0):
                    break
                new_x = x + pos_x
                new_y = y + pos_y

                if in_board(new_x, new_y) and self.map[new_x][new_y] == 0:
                    self.pig_pos.remove((x, y))
                    self.pig_pos.add((new_x, new_y))
                    self.map[new_x][new_y] = -2
                    self.map[x][y] = 0
                    break

    def update_rabbit_pos(self):

        def in_board(x, y):
            return not (x < 0 or x >= self.h or y < 0 or y >= self.w)

        # Move rabbits
        for i, item in enumerate(self.rabbit_pos):
            x, y = item
            direction = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
            np.random.shuffle(direction)
            for pos_x, pos_y in direction:
                if (pos_x, pos_y) == (0, 0):
                    break
                new_x = x + pos_x
                new_y = y + pos_y

                if in_board(new_x, new_y) and self.map[new_x][new_y] == 0:
                    self.rabbit_pos.remove((x, y))
                    self.rabbit_pos.add((new_x, new_y))
                    self.map[new_x][new_y] = -3
                    self.map[x][y] = 0
                    break

    def decrease_health(self):
        for id, _ in self.id_pos.iteritems():
            self.health[id] -= self.args.damage_per_step

    def get_avg_life(self):
        assert self.avg_life != None
        assert self.dead_people != None
        return self.avg_life, self.dead_people

    def remove_dead_people(self, cur_step):

        def max_view_size(view_size1, view_size2):
            view_size_area1 = (2 * view_size1[0] + 1) * (view_size1[1] + 1)
            view_size_area2 = (2 * view_size2[0] + 1) * (view_size2[1] + 1)

            return view_size1 if view_size_area1 > view_size_area2 else view_size2

        self.dead_id = []
        for id, pos in self.id_pos.iteritems():
            assert id > 0
            if self.health[id] <= 0.:
                x, y = pos
                self.map[x][y] = 0

                self.dead_id.append(id)
                self.agent_num -= 1

                group_id = self.id_group[id]
                if group_id > 0:
                    group_num = len(self.group_ids[group_id])

                    assert group_num >= 2

                    if group_num > 2:
                        del self.id_group[id]
                        self.group_ids[group_id].remove(id)

                        cur_max_view_size = None
                        for people in self.group_ids[group_id]:
                            if cur_max_view_size is None:
                                cur_max_view_size = self.property_copy[people][0][:]
                            else:
                                cur_max_view_size = max_view_size(cur_max_view_size, self.property_copy[people][0][:])
                        for people in self.group_ids[group_id]:
                            self.property[people][0] = cur_max_view_size

                        self.groups_view_size[group_id] = cur_max_view_size
                    else:
                        another_id = None
                        for item in self.group_ids[group_id]:
                            if item != id:
                                another_id = item
                        self.id_group[another_id] = 0
                        del self.id_group[id]
                        self.group_ids[group_id] = None

                        self.property[another_id] = self.property_copy[another_id][:]
                        self.groups_view_size[group_id] = None

        total_life = 0

        for id in self.dead_id:
            total_life += cur_step - self.birth_year[id]
            del self.id_pos[id]
            del self.property[id]
            del self.property_copy[id]
            del self.birth_year[id]

        if len(self.dead_id) == 0:
            self.avg_life = 0
        else:
            self.avg_life = 1. * total_life / (1. * len(self.dead_id))
        self.dead_people = len(self.dead_id)

        return self.dead_id

    def make_video(self, images, outvid=None, fps=5, size=None, is_color=True, format="XVID"):
        """
        Create a video from a list of images.
        @param      outvid      output video
        @param      images      list of images to use in the video
        @param      fps         frame per second
        @param      size        size of each frame
        @param      is_color    color
        @param      format      see http://www.fourcc.org/codecs.php
        """
        # fourcc = VideoWriter_fourcc(*format)
        # For opencv2 and opencv3:
        if int(cv2.__version__[0]) > 2:
            fourcc = cv2.VideoWriter_fourcc(*format)
        else:
            fourcc = cv2.cv.CV_FOURCC(*format)
        vid = None
        for image in images:
            assert os.path.exists(image)
            img = imread(image)
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        vid.release()

    def dump_image(self, img_name):
        new_w, new_h = self.w * 5, self.h * 5
        img = np.zeros((new_w, new_h, 3), dtype=np.uint8)
        length = self.args.img_length
        for i in xrange(self.w):
            for j in xrange(self.h):
                id = self.map[i][j]
                if id != 0:
                    for m in xrange(length):
                        for n in xrange(length):
                            img[i * length + m][j * length + n] = 255 * np.array(self.property[id][1])
        output_img = Image.fromarray(img, 'RGB')
        output_img.save(img_name)


def _get_reward_pig(pos):
    def in_bound(x, y):
        return not (x < 0 or x >= env_h or y < 0 or y >= env_w)

    x, y = pos
    groups_num = {}
    for i in xrange(-env_reward_radius_pig, env_reward_radius_pig + 1):
        for j in xrange(-env_reward_radius_pig, env_reward_radius_pig + 1):
            new_x, new_y = x + i, y + j
            if in_bound(new_x, new_y):
                id = env_map[new_x][new_y]
                if id > 0 and env_id_group[id] > 0:
                    if env_id_group[id] in groups_num:
                        groups_num[env_id_group[id]] += 1
                    else:
                        groups_num[env_id_group[id]] = 1
    if len(groups_num):
        groups_num = [(k, groups_num[k]) for k in groups_num if groups_num[k] >= env_reward_threshold_pig]
        if len(groups_num) > 0:
            groups_num = sorted(groups_num, key=lambda x: x[1])
            return env_group_ids[groups_num[-1][0]], pos
        else:
            return [], pos
    else:
        return [], pos


def _get_reward_rabbit_both(pos):
    # both groups and individuals can catch rabbits
    def in_bound(x, y):
        return not (x < 0 or x >= env_h or y < 0 or y >= env_w)

    x, y = pos
    candidates = []
    for i in xrange(-env_reward_radius_rabbit, env_reward_radius_rabbit + 1):
        for j in xrange(-env_reward_radius_rabbit, env_reward_radius_rabbit + 1):
            new_x, new_y = x + i, y + j
            if in_bound(new_x, new_y):
                id = env_map[new_x][new_y]
                if id > 0:
                    candidates.append(id)
    if len(candidates) > 0:
        winner = np.random.choice(candidates)
        if env_id_group[winner] == 0:
            return [winner], pos
        else:
            return env_group_ids[env_id_group[winner]], pos
    else:
        return [], pos


def _get_reward_rabbit_individual(pos):
    # only individuals can catch rabbits
    def in_bound(x, y):
        return not (x < 0 or x >= env_h or y < 0 or y >= env_w)

    x, y = pos
    candidates = []
    for i in xrange(-env_reward_radius_rabbit, env_reward_radius_rabbit + 1):
        for j in xrange(-env_reward_radius_rabbit, env_reward_radius_rabbit + 1):
            new_x, new_y = x + i, y + j
            if in_bound(new_x, new_y):
                id = env_map[new_x][new_y]
                if id > 0 and env_id_group[id] == 0:
                    candidates.append(id)
    if len(candidates) > 0:
        return [np.random.choice(candidates)], pos
    else:
        return [], pos


def get_reward(env):
    global env_pig_pos
    global env_agent_num
    global env_batch_size
    global env_map
    global env_reward_radius_pig
    global env_reward_radius_rabbit
    global env_reward_threshold_pig
    global env_w
    global env_h
    global env_id_group
    global env_group_ids

    env_pig_pos = env.pig_pos
    env_rabbit_pos = env.rabbit_pos
    env_agent_num = env.agent_num
    env_map = env.map
    env_batch_size = env.batch_size
    env_reward_radius_pig = env.reward_radius_pig
    env_reward_threshold_pig = env.reward_threshold_pig
    env_reward_radius_rabbit = env.reward_radius_rabbit
    env_w = env.w
    env_h = env.h
    env_id_group = env.id_group
    env_group_ids = env.group_ids

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    reward_ids_pig = pool.map(_get_reward_pig, env_pig_pos)
    reward_ids_rabbit = pool.map(_get_reward_rabbit_individual, env_rabbit_pos)
    pool.close()

    killed_pigs = set()
    killed_rabbits = set()
    rewards = {}

    for item in reward_ids_pig:
        if len(item[0]) > 0:
            reward_per_agent = 1. / len(item[0])
            for id in item[0]:
                if id not in rewards:
                    rewards[id] = reward_per_agent
                else:
                    rewards[id] += reward_per_agent
            killed_pigs.add(item[1])

    for item in reward_ids_rabbit:
        if len(item[0]) > 0:
            reward_per_agent = 0.05 / len(item[0])
            for id in item[0]:
                if id not in rewards:
                    rewards[id] = reward_per_agent
                else:
                    rewards[id] += reward_per_agent
            killed_rabbits.add(item[1])

    env_pig_pos = env_pig_pos - killed_pigs
    env.pig_pos = env_pig_pos
    env.pig_num -= len(killed_pigs)

    env_rabbit_pos = env_rabbit_pos - killed_rabbits
    env.rabbit_pos = env_rabbit_pos
    env.rabbit_num -= len(killed_rabbits)

    for item in killed_pigs:
        x, y = item
        env.map[x][y] = 0
    for item in killed_rabbits:
        x, y = item
        env.map[x][y] = 0

    return rewards


def get_view(env):
    global env_property
    global env_map
    global env_h
    global env_w
    global env_id_group
    global env_id_pos
    global batch_size
    global env_agent_num
    global env_max_view_size
    global env_min_view_size
    global env_id_ally_number
    global env_health

    env_property = env.property
    env_map = env.map
    env_h = env.h
    env_w = env.w
    env_id_group = env.id_group
    env_id_pos = env.id_pos
    env_batch_size = env.batch_size
    env_agent_num = env.agent_num
    env_max_view_size = env.max_view_size
    env_min_view_size = env.min_view_size
    env_id_ally_number = {}
    env_health = env.health

    allies = []

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    env_id_pos_keys = env_id_pos.keys()
    env_id_pos_keys.sort()
    pos = [env_id_pos[k] for k in env_id_pos_keys]
    view = pool.map(_get_view, pos)
    pool.close()

    env.id_ally_number = env_id_ally_number

    views = []
    ids = []
    for item in view:
        views.append(item[0])
        ids.append(item[2])
        if item[1]:
            allies.append(item[1])

    env.ally.clear()
    # Candidate ally
    for item in allies:
        env.ally[item[0]] = item[1]

    view = np.array(views)

    batch_views = []

    for i in xrange(int(np.ceil(1. * env_agent_num / env_batch_size))):
        st = env_batch_size * i
        ed = st + env_batch_size
        if ed > env_agent_num:
            ed = env_agent_num

        # batch_view_tmp = view[st:ed]
        # batch_ids = ids[st:ed]
        batch_view = []
        for j in xrange(st, ed):
            batch_view.append((ids[j], view[j]))

        batch_views.append(batch_view)

    return batch_views


def _get_view(pos):
    x, y = pos
    range_l, range_f, face = env_property[env_map[x][y]][0]
    max_range_l, max_range_f, _ = env_max_view_size
    min_range_l, min_range_f, _ = env_min_view_size
    # single_view = np.zeros(((2 * max_range_l + 1) * (max_range_f + 1), 4), dtype=np.float32)
    single_view = np.zeros(((2 * max_range_l + 1) * (max_range_f + 1), 5), dtype=np.float32)
    env_ally = None

    def in_bound(x, y):
        return not (x < 0 or x >= env_h or y < 0 or y >= env_w)

    def in_group(id_1, id_2):
        if env_id_group[id_1] == env_id_group[id_2]:
            return True
        else:
            return False

    cur_pos = 0
    allies = set()
    face = 0
    if face == 0:
        # for i in xrange(-range_f, 1):
        #     for j in xrange(-range_l, range_l + 1):
        for i in xrange(-max_range_f, 1):
            for j in xrange(-max_range_l, max_range_l + 1):
                new_x, new_y = x + i, y + j

                if not in_bound(new_x, new_y) or i < -range_f or j < -range_l or j > range_l:
                    single_view[cur_pos] = [1, 1, 0, 0, 0]
                else:
                    if env_id_group[env_map[x][y]] == 0 and env_map[new_x][new_y] > 0 and i in xrange(-min_range_f,
                                                                                                      1) and j in xrange(
                        -min_range_l, min_range_l + 1):
                        allies.add(env_map[new_x][new_y])
                    single_view[cur_pos][0], single_view[cur_pos][1], single_view[cur_pos][2] = \
                        env_property[env_map[x][y]][1]
                    if env_map[new_x][new_y] > 0 and in_group(env_map[x][y], env_map[new_x][new_y]):
                        single_view[cur_pos][3] = 1
                        # For exploring the reason of degroup
                        if env_map[x][y] in env_id_ally_number:
                            env_id_ally_number[env_map[x][y]] += 1
                        else:
                            env_id_ally_number[env_map[x][y]] = 1
                    else:
                        single_view[cur_pos][3] = 0

                    # For health
                    single_view[cur_pos][4] = env_health[env_map[x][y]]

                cur_pos = cur_pos + 1

        # TODO: the logic of join a group
        if len(allies) > 0:
            ally_id = random.sample(allies, 1)[0]
            id = env_map[x][y]
            if id != ally_id:
                env_ally = (id, ally_id)

    elif face == 1:
        # for i in xrange(-range_l, range_l + 1):
        #     for j in xrange(0, range_f + 1):
        for i in xrange(-max_range_l, max_range_l + 1):
            for j in xrange(0, max_range_f + 1):
                new_x, new_y = x + i, y + j
                if not in_bound(new_x, new_y) or i < -range_l or i > range_l or j > range_f:
                    single_view[cur_pos] = [1, 1, 0, 0, 0]
                else:
                    if env_id_group[env_map[x][y]] == 0 and env_map[new_x][new_y] > 0 and i in xrange(-min_range_l,
                                                                                                      min_range_l + 1) and j in xrange(
                        0, min_range_f + 1):
                        allies.add(env_map[new_x][new_y])
                    single_view[cur_pos][0], single_view[cur_pos][1], single_view[cur_pos][2] = \
                        env_property[env_map[x][y]][1]
                    if env_map[new_x][new_y] > 0 and in_group(env_map[x][y], env_map[new_x][new_y]):
                        single_view[cur_pos][3] = 1
                        if env_map[x][y] in env_id_ally_number:
                            env_id_ally_number[env_map[x][y]] += 1
                        else:
                            env_id_ally_number[env_map[x][y]] = 1
                    else:
                        single_view[cur_pos][3] = 0

                    # For health
                    single_view[cur_pos][4] = env_health[env_map[x][y]]

                cur_pos = cur_pos + 1
        if len(allies) > 0:
            ally_id = random.sample(allies, 1)[0]
            id = env_map[x][y]
            if id != ally_id:
                env_ally = (id, ally_id)

    elif face == 2:
        # range_i_st, range_i_ed = -range_f, 0
        # range_j_st, range_j_ed = -range_l, range_l
        # for i in xrange(range_f, -1):
        #     for j in xrange(range_l, -range_l - 1):
        for i in xrange(max_range_f, -1, -1):
            for j in xrange(max_range_l, -max_range_l - 1, -1):
                new_x, new_y = x + i, y + j
                if not in_bound(new_x, new_y) or i > range_f or j > range_l or j < -range_l:
                    single_view[cur_pos] = [1, 1, 0, 0, 0]
                else:
                    if env_id_group[env_map[x][y]] == 0 and env_map[new_x][new_y] > 0 and i in xrange(min_range_f, -1,
                                                                                                      -1) and j in xrange(
                        min_range_l, -min_range_l - 1, -1):
                        allies.add(env_map[new_x][new_y])
                    single_view[cur_pos][0], single_view[cur_pos][1], single_view[cur_pos][2] = \
                        env_property[env_map[x][y]][1]
                    if env_map[new_x][new_y] > 0 and in_group(env_map[x][y], env_map[new_x][new_y]):
                        single_view[cur_pos][3] = 1
                        if env_map[x][y] in env_id_ally_number:
                            env_id_ally_number[env_map[x][y]] += 1
                        else:
                            env_id_ally_number[env_map[x][y]] = 1
                    else:
                        single_view[cur_pos][3] = 0
                    # For health
                    single_view[cur_pos][4] = env_health[env_map[x][y]]

                cur_pos = cur_pos + 1
        if len(allies) > 0:
            ally_id = random.sample(allies, 1)[0]
            id = env_map[x][y]
            if id != ally_id:
                env_ally = (id, ally_id)


    elif face == 3:
        # for i in xrange(range_l, -range_l - 1):
        #    for j in xrange(-range_f, 1):
        for i in xrange(max_range_l, -max_range_l - 1, -1):
            for j in xrange(-max_range_f, 1):
                print "miaomiaomiao"
                new_x, new_y = x + i, y + j
                if not in_bound(new_x, new_y) or i > range_l or i < -range_l or j < -range_f:
                    single_view[cur_pos] = [1, 1, 0, 0, 0]
                else:
                    if env_id_group[env_map[x][y]] == 0 and env_map[new_x][new_y] > 0 and i in xrange(min_range_l,
                                                                                                      -min_range_l - 1,
                                                                                                      -1) and j in xrange(
                        -min_range_f, 1):
                        allies.add(env_map[new_x][new_y])
                    single_view[cur_pos][0], single_view[cur_pos][1], single_view[cur_pos][2] = \
                        env_property[env_map[x][y]][1]
                    if env_map[new_x][new_y] > 0 and in_group(env_map[x][y], env_map[new_x][new_y]):
                        single_view[cur_pos][3] = 1
                        if env_map[x][y] in env_id_ally_number:
                            env_id_ally_number[env_map[x][y]] += 1
                        else:
                            env_id_ally_number[env_map[x][y]] = 1
                    else:
                        single_view[cur_pos][3] = 0

                    # For health
                    single_view[cur_pos][4] = env_health[env_map[x][y]]

                cur_pos = cur_pos + 1
        if len(allies) > 0:
            ally_id = random.sample(allies, 1)[0]
            id = env_map[x][y]
            if id != ally_id:
                env_ally = (id, ally_id)

    else:
        print "Error Face!!!"
    assert cur_pos == (2 * max_range_l + 1) * (max_range_f + 1)
    return single_view.reshape(-1), env_ally, env_map[x][y]
