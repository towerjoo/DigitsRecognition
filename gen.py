import random
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image


def gen_data(N=1):
    # N: the number of images to generate
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    for _ in range(N):
        length = random.randint(5, 21) # 5<=L<=20
        ims = []
        name = ""
        for i in range(length):
            index = random.choice(range(x_train.shape[0]))
            im = Image.fromarray(x_train[index])
            name += str(y_train[index])

            # apply resizing
            resize_ratio = random.random() * .5 + .5 # .5 <= r <= 1
            newsize = (int(im.size[0] * resize_ratio), int(im.size[1] * resize_ratio)) # PIL requires size to be int
            im = im.resize(newsize)

            # apply rotation
            rotate_degree = random.randint(0, 91) # 0<=d<=90
            im = im.rotate(rotate_degree)
            ims.append(im)
        w, h = sum(im.size[0] for im in ims), max(im.size[1] for im in ims)
        im = Image.new("1", (w, h))
        x_offset = 0
        for _im in ims:
            im.paste(_im, (x_offset, 0))
            x_offset += _im.size[0]
        im.save("{}.jpg".format(name))


if __name__ == "__main__":
    gen_data()
