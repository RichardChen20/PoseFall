import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_poses(img, skeletons, save_path=None):
    # img BGR , skeletons [N, 17, 3],最后一个维度[x,y, score]
    # skeletons = skeletons.numpy()
    # 鼻子-0, 右眼 左眼 右耳 左耳 右肩 左肩 右肘 左肘 右手 左手 右臀 左臀 右膝 左膝 右脚 左脚
    #skeletons = np.array(skeletons)

    EDGES = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13],
             [13,15], [6, 12], [12, 14], [14, 16], [0, 17]]
    # [0, 5], [0, 6],

    # EDGES = [[0, 14], [0, 15], [14, 16], [15, 17], [0, 1], [1, 2], [1, 5], [1, 8], [1, 11], [8, 9], [9, 10], [11, 12],
    #         [12, 13], [2, 3], [3, 4], [5, 6], [6, 7]]

    # 森哥的
    # EDGES = [[0, 14], [0, 13], [0, 4], [0, 1], [14, 16], [13, 15], [4, 10],
    #         [1, 7], [10, 11], [7, 8], [11, 12], [8, 9], [4, 5], [1, 2], [5, 6], [2, 3]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]
    cmap = matplotlib.cm.get_cmap('hsv')
    canvas = img.copy()

    for i in range(18):
        rgba = np.array(cmap(1 - i / 17. - 1. / 34))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            cv2.circle(canvas, tuple(skeletons[j][i, 0:2].astype('int32')), 2, colors[i], thickness=-1)

    to_plot = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)
    stickwidth = 2

    for i in range(len(EDGES)):
        for j in range(len(skeletons)):
            edge = EDGES[i]
            if skeletons[j][edge[0], 2] == 0 or skeletons[j][edge[1], 2] == 0:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    #fig2 = plt.figure(figsize=(8, 8))
    #plt.imshow(canvas)
    #plt.show()
    return canvas