from __future__ import division
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import os

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22

# Lines
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

# Model
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
traindata_path = 'images/'


def relative_coord(initial, target):
    """
    Helper method for getting coordinates relative to (0) (x_n - x_0, y_n - y_0)
    """
    return target - initial


def normalize(vec):
    """
    Normalize x to [-1,1], y to [-1,0]. y to [-1,0] because img y axis is upside down
    Assumes all y are positive - y will also have codomain [-1,1] if there are negative values
    """
    max_x = 0
    max_y = 0

    for i in vec:
        max_x = max(max_x, abs(i[0]))
        max_y = max(max_y, abs(i[1]))

    for i in range(len(vec)):
        vec[i][0] /= max_x
        vec[i][1] /= max_y

    return vec


def rotate(vec, move_og0=False, og0=None, down=True):
    """
    Rotates all inputs to one direction.
    if down: rotate to y negative (positive vertical axis in actual image)
    if up: rotate to y positive (negative vertical axis in actual image)
    """
    # Difference between point (9) and point (0)
    diff_x = vec[9][0] - vec[0][0]
    diff_y = vec[9][1] - vec[0][1]

    # Avoid division by 0
    if diff_x == 0:
        diff_x = 1e-9

    quotient = diff_y/diff_x
    ac = abs(np.arctan(quotient))

    if not down:
        if diff_x >= 0 and diff_y >= 0:
            # ++
            theta = np.pi / 2 - ac
        elif diff_x < 0 and diff_y >= 0:
            # -+
            theta = ac - np.pi / 2
        elif diff_x < 0 and diff_y < 0:
            # --
            theta = -1 * (np.pi / 2 + ac)
        else:
            # +-
            theta = np.pi / 2 + ac
    else:
        if diff_x >= 0 and diff_y >= 0:
            # ++
            theta = -1 * (np.pi/2 + ac)
            # theta = -1 * (np.pi/2 - ac)
            print("Q1")
        elif diff_x < 0 and diff_y >= 0:
            # -+
            theta = np.pi/2 + ac
            # theta = np.pi/2 - ac
            print("Q2")
        elif diff_x < 0 and diff_y < 0:
            # --
            theta = np.pi/2 - ac
            # theta = np.pi/2 + ac
            print("Q3")
        else:
            # +-
            theta = ac - np.pi/2
            # theta = -1 * (np.pi/2 + ac)
            print("Q4")
    print(f"Arctan:\t{np.rad2deg(ac)}")
    print(f"Counter rotation angle:\t{np.rad2deg(theta)} degrees")

    # Rotation matrix
    rot = np.array(
        [[np.cos(theta), -1*np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )

    for i in range(len(vec)):
        vec[i] = np.dot(rot, vec[i])
        if move_og0:
            vec[i] += og0
    return vec


def flip(vec_rotated):
    """
    Flips all points along the y axis if (1) is to the right of (0)
    """
    if vec_rotated[1][0] > vec_rotated[0][0]:
        for j in range(len(vec_rotated)):
            vec_rotated[j][0] *= -1
    return vec_rotated


with open("train.csv", "w") as fin:
    fin.write("fname")
    for i in range(21):
        fin.write(f",p{i}_x,p{i}_y")
    fin.write("\n")

# Vectorize relative coordinates function
relative_coord = np.vectorize(relative_coord)

# Main
fin = open("train.csv", "a+")
for img in os.listdir(traindata_path):
    if ".py" in img or "." not in img:
        continue
    print("\n" + img)
    frame = cv2.imread(traindata_path + img)
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight
    threshold = 0.1

    t = time.time()
    # input image dimensions for the network
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    # Not all points detected
    if None in points[:21]:
        print(f"Low confidence, passing {img}")
        continue

    # ---------------------Preprocess---------------------
    # Relative
    newpoints = np.array(points[:21], dtype=float)
    origin = newpoints[0]
    relcoords = relative_coord(origin, newpoints)

    # Rotate
    relcoords = rotate(relcoords, move_og0=False, og0=np.array((points[0])), down=True)

    # Display
    dispcoords = relcoords.astype("int32") + points[0]

    # Flip if (1) is on the right of (0)
    relcoords = flip(relcoords)

    # Flatten
    normcoords = normalize(relcoords).flatten()

    # --------------------- End Preprocess ---------------------
    # print(f"id:{img} \n{normcoords.flatten()}")
    fin.write(f"{img},"+",".join(map(str, normcoords.tolist()))+"\n")

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        # if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (255, 0, 255), 1)
        cv2.circle(frame, points[partA], 4, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, points[partB], 4, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

        # if relcoords[partA] and relcoords[partB]:
        cv2.line(frame, tuple(dispcoords[partA]), tuple(dispcoords[partB]), (0, 255, 255), 2)
        cv2.circle(frame, tuple(dispcoords[partA]), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, tuple(dispcoords[partB]), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.imwrite('results/' + img, frame)
    print("Total time taken : {:.3f}".format(time.time() - t))
fin.close()
