import tkinter as tk
import time
import os
from PIL import Image, ImageTk

files = os.listdir("./results")
for f in files:
    if f.find(".py") is not -1:
        files.remove(f)
iterator = iter(files)


def rock_click():
    writelabel(imgname, 0)
    print("rock clicked")
    next_picture()


def paper_click():
    writelabel(imgname, 1)
    print("paper clicked")
    next_picture()


def scissors_click():
    writelabel(imgname, 2)
    print("scissors clicked")
    next_picture()


def leave():
    fout.close()
    window.destroy()


def writelabel(fname, label):
    tmp = [0,0,0]
    tmp[label] = 1
    # print(f"{fname},label:{','.join(map(str, tmp))}")

    # # One hot
    # fout.write(f"{fname},{','.join(map(str, tmp))}\n")

    # Class
    fout.write(f"{fname},{label}\n")


def next_picture():
    global image, resized_image, imgname
    try:
        imgname = next(iterator)
        new_image = Image.open("./results/" + imgname)

        # Resize
        canvas_width = 400
        canvas_height = 400
        new_image.thumbnail((canvas_width, canvas_height), Image.ANTIALIAS)

        image = ImageTk.PhotoImage(new_image)
        resized_image = new_image
        canvas.itemconfig(image_id, image=image)
    except StopIteration or IOError:
        print("All files have been passed, shutting down in 3 seconds")
        time.sleep(3)
        leave()
    except PermissionError:
        next(iterator)
        next_picture()


# Main window
window = tk.Tk()

# Canvas
canvas = tk.Canvas(window, width=400, height=400)
canvas.pack()

# Initial pic
imgname = next(iterator)
initial_image = Image.open("./results/" + imgname)

# Resize
canvas_width = 400
canvas_height = 400
initial_image.thumbnail((canvas_width, canvas_height), Image.ANTIALIAS)

image = ImageTk.PhotoImage(initial_image)
resized_image = initial_image
image_id = canvas.create_image(0, 0, anchor=tk.NW, image=image)


# Buttons
rock = tk.Button(window, text="Rock 石头", command=rock_click)
rock.pack()

paper = tk.Button(window, text="Paper 布", command=paper_click)
paper.pack()

scissors = tk.Button(window, text="Scissors 剪刀", command=scissors_click)
scissors.pack()

exit = tk.Button(window, text="Exit 退出", command=leave)
exit.pack()


# Run the main loop
# fin = open("train.csv", "r+")
fout = open("targets.csv", "w")

# # One hot
# fout.write("fname,rock,paper,scissors\n")

# Class
fout.write("fname,cat\n")
window.mainloop()
