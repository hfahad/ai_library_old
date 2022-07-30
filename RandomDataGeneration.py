import cv2
import numpy as np
import os
import pandas as pd
import random

# Shape Generater
shapes = ['tri', 'cir', 'rec']  # 0,1,2


def generateShape(n_, seed=1000):
    # param: total number of instances
    random.seed(seed)
    shapeData = []
    labels = []
    for i in range(n_):
        shape = random.choices(shapes)[0]
        if shape == 'tri':
            image = np.zeros((64, 64), np.uint8)
            # random.seed(i)
            p = [random.randint(0, 64) for j in range(6)]
            vertices = np.array(
                [[p[0], p[1]], [p[2], p[3]], [p[4], p[5]]], np.int32)
            pts = vertices.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=1, thickness=1)
            # fill it
            img = cv2.fillPoly(image, [pts], color=1)
        elif shape == 'cir':
            image = np.zeros((64, 64), np.uint8)
            # random.seed(i)
            p = [random.randint(0, 64) for j in range(2)]
            img = cv2.circle(image, p, random.randint(
                0, 32), color=1, thickness=-1)
        elif shape == 'rec':
            image = np.zeros((64, 64), np.uint8)
            # random.seed(i)
            p = [random.randint(0, 64) for j in range(4)]
            img = cv2.rectangle(image, p[:2], p[2:], 1, thickness=-1)
        else:
            raise Exception("Sorry, input correct parameter")
        # print(shape)
        shapeData.append(img.reshape(-1))
        labels.append(shapes.index(shape))
    return shapeData, labels


def saveData(n_instance, outname, seed):
    data, labels = generateShape(n_instance, seed=seed)
    df = pd.DataFrame(data)
    df['Label'] = labels
    print('Data distribution %', 'Rectangles =', np.sum(df['Label'] == 2)/n_instance*100, ', Circles =', np.sum(
        df['Label'] == 1)/n_instance*100, 'Triangles =', sum(df['Label'] == 0)/n_instance*100)
    df.to_csv(outname, index=False)


def create_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)


create_dir('/output/shapes')
saveData(n_instance=10000, outname='./data/shapes/train.csv', seed=123)
saveData(n_instance=2000, outname='./data/shapes/test.csv', seed=456)
