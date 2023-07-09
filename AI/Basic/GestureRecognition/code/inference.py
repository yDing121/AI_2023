from __future__ import division
import tensorflow as tf
import cv2
import time
import numpy as np
from makeTraining import relative_coord, rotate, flip, normalize

ANSDICT = {
    0: "Rock",
    1: "Paper",
    2: "Scissors"
}
def preprocess(
        imgname,
        thresh=0.1,
        net=cv2.dnn.readNetFromCaffe("hand/pose_deploy.prototxt","hand/pose_iter_102000.caffemodel")):
    """
    Preprocesses the input image (str) and feeds it into the first stage network, returning a (42,1) np.array
        or -1 if low confidence
    """

    frame = cv2.imread(imgname)
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight
    threshold = thresh

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

    for i in range(22):
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
        print(f"Low confidence, passing {imgname}")
        return -1

    # ---------------------Preprocess---------------------
    # Relative
    newpoints = np.array(points[:21], dtype=float)
    origin = newpoints[0]
    relcoords = relative_coord(origin, newpoints)

    # Rotate
    relcoords = rotate(relcoords, move_og0=False, og0=np.array((points[0])), down=True)

    # Flip if (1) is on the right of (0)
    relcoords = flip(relcoords)

    # Flatten
    normcoords = normalize(relcoords).flatten()

    # --------------------- End Preprocess ---------------------
    return np.asarray([normcoords])


def classify(points,
          model):
    return model.predict(points)


def infer(img, thresh=0.1):
    global ANSDICT
    points = preprocess(img,
                        thresh=thresh,
                        net=cv2.dnn.readNetFromCaffe("hand/pose_deploy.prototxt","hand/pose_iter_102000.caffemodel"))
    if type(points) == "<class 'int'>":
        print("Low confidence, cannot identify hand landmarks")
        return -1
    else:

        c = np.argmax(classify(points, model=m)).item()
        # print(f"{type(c)}:\t{c}")
        return ANSDICT.get(c)


if __name__ == "__main__":
    m = tf.keras.models.load_model("./stable_model")

    print(infer("./images/1.jpg"))
