import os
os.chdir("./images")

for n, img in enumerate(os.listdir(".")):
    period_idx = img.find(".")
    ftype = img[period_idx + 1:]
    if ftype == "py" or period_idx < 0:
        continue

    os.rename(img, f"{n}.{ftype.lower()}")
    # print(f"{n}.{ftype}")
