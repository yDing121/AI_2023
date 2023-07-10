from __future__ import division
import tensorflow as tf
import cv2
import time
import numpy as np
from makeTraining import POSE_PAIRS, relative_coord, rotate, flip, normalize
import tkinter as tk
from PIL import ImageTk, Image
import os

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

    # Display
    dispcoords = relcoords.astype("int32") + points[0]

    # Flip if (1) is on the right of (0)
    relcoords = flip(relcoords)

    # Flatten
    normcoords = normalize(relcoords).flatten()

    # --------------------- End Preprocess ---------------------
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
    cv2.imwrite(f'{infPath}tmp.jpg', frame)
    return np.asarray([normcoords])


def classify(points,
          model):
    return model.predict(points)


def infer(img, thresh=0.1):
    global ANSDICT
    points = preprocess(img,
                        thresh=thresh,
                        net=cv2.dnn.readNetFromCaffe("hand/pose_deploy.prototxt", "hand/pose_iter_102000.caffemodel"))
    if type(points) == "<class 'int'>":
        print("Low confidence, cannot identify hand landmarks")
        return -1
    else:
        infer_change_pic()
        c = np.argmax(classify(points, model=m)).item()
        return ANSDICT.get(c)


if __name__ == "__main__":
    m = tf.keras.models.load_model("./stable_model")
    # print(infer("./images/1.jpg"))

    # List of pictures
    infPath = "./demo_images/"
    pictures = os.listdir(infPath)

    # --------------- tkinter app for inference with wraparound ----------------
    # Picture counter
    current_picture = 0


    def change_picture():
        global current_picture
        current_picture = (current_picture + 1) % len(pictures)
        img = Image.open(f"{infPath}{pictures[current_picture]}")
        img.thumbnail((400, 400), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)
        picture_label.configure(image=img_tk)
        picture_label.image = img_tk
        text_label.configure(text=f"File: {pictures[current_picture]}")


    def infer_change_pic():
        try:
            img = Image.open(f"{infPath}tmp.jpg")
            img.thumbnail((400, 400), Image.ANTIALIAS)
            img_tk = ImageTk.PhotoImage(img)
            picture_label.configure(image=img_tk)
            picture_label.image = img_tk
        except FileNotFoundError:
            return -1


    def infer_pressed():
        res = infer(f"{infPath}{pictures[current_picture]}")
        result_label.configure(text=f"Result: {res}")


    window = tk.Tk()

    # Picture
    img = Image.open(f"{infPath}{pictures[current_picture]}")
    img.thumbnail((400, 400), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(img)
    picture_label = tk.Label(window, image=img_tk)
    picture_label.pack()

    # Picture number label
    text_label = tk.Label(window, text=f"File: {pictures[current_picture]}")
    text_label.pack()

    # Next picture button
    next_button = tk.Button(window, text="Next Picture", command=change_picture)
    next_button.pack(side=tk.RIGHT)

    # Infer button
    infer_button = tk.Button(window, text="Infer", command=infer_pressed)
    infer_button.pack(side=tk.LEFT)

    # Inference results label
    result_label = tk.Label(window, text="Inference Result:")
    result_label.pack()

    window.mainloop()
    try:
        os.remove(f"{infPath}tmp.jpg")
    except FileNotFoundError:
        print("No tmp to clear")
