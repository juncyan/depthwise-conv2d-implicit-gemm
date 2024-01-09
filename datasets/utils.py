import math
import numpy as np

def rbox2poly(rrect):
    '''
    :param rrect: [x_ctr, y_ctr, w, h, angle]
    :return:  [x0, y0, x1, y1, x2, y2, x3, y3]
    '''

    x_ctr, y_ctr, width, height, angle = rrect[:5]
    angle = np.pi * angle / 180.
    t1_x, t1_y, br_x, br_y = -width/2, -height/2, width/2, height/2
    rect = np.array([[t1_x, br_x, br_x, t1_x], [t1_y, t1_y, br_y, br_y]])
    r = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    poly = r.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    return [x0, y0, x1, y1, x2, y2, x3, y3]

def poly2rbox(bbox):
    '''
    :param bbox: [x0, y0, x1, y1, x2, y2, x3, y3]
    :return: [x_ctr, y_ctr, w, h, angle]
    '''

    bbox = np.array(bbox, np.float32)
    bbox = np.reshape(bbox, newshape=(2,4), order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    angle = math.atan2(bbox[1,2] - bbox[1,1], bbox[0,2] - bbox[0,1])
    if angle > -math.pi /2 and angle < 0:
        angle += math.pi / 2
    elif angle > math.pi / 2 and angle < math.pi:
        angle -= math.pi / 2

    center = [[0], [0]]
    for i in range(4):
        center[0] += bbox[0, i]
        center[1] += bbox[1, i]
    center = np.array(center, np.float32) / 4.0

    rotate = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]], np.float32)
    normalized = np.matmul(rotate.transpose(), bbox - center)
    xmin = np.min(normalized[0, :])
    xmax = np.max(normalized[0, :])
    ymin = np.min(normalized[1, :])
    ymax = np.max(normalized[1, :])
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    return [float(center[0]), float(center[1]), w, h, angle]

def one_hot_it(label, label_info):
    semantic_map = []
    for info in label_info:
        color = label_info[info].values
        # print("label:\n", label.shape,label)
        # print("color:\n", color)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    return np.stack(semantic_map, axis=-1)