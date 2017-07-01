import matplotlib.pyplot as plt
import numpy as np
from cv2 import VideoWriter, imread, resize
import cv2
import os
import numpy as np


def make_video(images, outvid=None, fps=5, size=None, is_color=True, format="XVID"):
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


t = 0
start_step = 20000
target_step = 26000
width = 1000
height = 1000

# with open('log_largest_group.txt') as fin:
#     for line in fin:
#         t += 1
#         if t > target_step:
#             exit(0)
#         if t >= start_step:
#             x = []
#             y = []
#             line = line.split()
#             for item in line:
#                 item = item.split(',')
#                 x.append(int(item[0]))
#                 y.append(int(item[1]))
#             plt.figure(figsize=(8, 6))
#             plt.scatter(x, y, marker='o', color='r')
#
#             x_min_bound = (np.min(x) / 100) * 100
#             x_max_bound = (np.max(x) / 100 + 1) * 100 if np.max(x) % 100 != 0 else (np.max(x) / 100) * 100
#             plt.xlim(x_min_bound, x_max_bound)
#
#             y_min_bound = (np.min(y) / 100) * 100
#             y_max_bound = (np.max(y) / 100 + 1) * 100 if np.max(y) % 100 != 0 else (np.max(y) / 100) * 100
#             plt.ylim(y_min_bound, y_max_bound)
#
#             plt.title('largest group size: %d' % len(x))
#             plt.grid()
#             plt.savefig('largest_group/%d.png' % t)

images = ['largest_group/%d.png' % i for i in xrange(start_step, target_step + 1)]
make_video(images=images, outvid='largest_group_%d_%d.avi' % (start_step, target_step), fps=10)
