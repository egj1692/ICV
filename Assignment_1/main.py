import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def ex1():
    img = cv.imread('assignment_1/Lena.png')
    img_r = img[:, :, 0]
    print(img.shape)

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.axis("off")
    plt.imshow(img_r, cmap='gray')

    hist, bins = np.histogram(img_r, 256, [0, 255])
    plt.subplot(222)
    plt.axis("off")
    plt.fill(hist)
    plt.xlabel("pixel value")

    img_eq = cv.equalizeHist(img_r)
    plt.subplot(223)
    plt.axis("off")
    plt.imshow(img_eq, cmap='gray')

    hist, bins = np.histogram(img_eq, 256, [0, 255])
    plt.subplot(224)
    plt.axis("off")
    plt.fill_between(range(256), hist, 0)
    plt.xlabel("pixel value")

    plt.show()
def ex2():
    image = cv.imread('assignment_1/image_House256rgb.png').astype(np.float32) / 255
    print(image.shape)

    noised = (image + 50 * np.random.rand(*image.shape).astype(np.float32) / 255)
    noised = noised.clip(0, 1)

    # diameter = int(input('diameter : '))
    # SigmaColor = float(input('SigmaColor : '))
    # SigmaSpace = int(input('SigmaSpace : '))
    diameter = 0
    SigmaColor = 0.3
    SigmaSpace = 10
    bilats = []
    for i in range(0, 30, 5):
        bilat = cv.bilateralFilter(noised, d=i, sigmaColor=SigmaColor, sigmaSpace=SigmaSpace)
        bilats.append(bilat)
    for i in range(0, 30, 5):
        bilat = cv.bilateralFilter(noised, d=diameter, sigmaColor=i, sigmaSpace=SigmaSpace)
        bilats.append(bilat)

    # img = np.hstack((image,noised, bilat))
    # cv.imshow('img',img)
    # cv.waitKey()
    plt.figure(figsize=(10, 11))
    plt.subplot(5,3,1)
    plt.title("image")
    plt.axis("off")
    plt.imshow(image[:, :, [2, 1, 0]])

    plt.subplot(5,3,2)
    plt.title("noised")
    plt.axis("off")
    plt.imshow(noised[:, :, [2, 1, 0]])

    for i in range(len(bilats)):
        plt.subplot(5, 3, i+3)
        plt.title("bilat")
        plt.axis("off")
        plt.imshow(bilats[i][:, :, [2, 1, 0]])

    plt.show()
def ex3():
    image = cv.imread('assignment_1/image_Peppers512rgb.png', 0).astype(np.float32) / 255
    fft = cv.dft(image, flags=cv.DFT_COMPLEX_OUTPUT)

    shifted = np.fft.fftshift(fft, axes=[0, 1])

    # 반지름 입력
    sz = int(input("circle size(정수): "))
    idx = np.zeros(image.shape[:2], np.uint8)
    idx_mask = cv.circle(idx, (int(image.shape[0] / 2), int(image.shape[1] / 2)), sz, 1, -1)
    mask = np.zeros(fft.shape, np.uint8)
    key = input("low(l or L) or high(h or H): ")
    # low, high 입력
    if key == 'l' or key == 'L':
        mask[np.where(idx_mask != 1)] = 1
    elif key == 'h' or key == 'H':
        mask[np.where(idx_mask == 1)] = 1
    else :
        key = 'l'
    shifted *= mask
    fft = np.fft.ifftshift(shifted, axes=[0, 1])

    filtered = cv.idft(fft, flags=cv.DFT_SCALE + cv.DFT_REAL_OUTPUT)
    mask_new = np.dstack((mask, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)))

    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.axis("off")
    plt.imshow(image, cmap='gray')
    plt.subplot(132)
    plt.axis("off")
    plt.imshow(filtered, cmap='gray')
    plt.subplot(133)
    plt.axis("off")
    plt.imshow(mask_new * 255, cmap='gray')
    plt.show()

def ex4():
    image = cv.imread('assignment_1/image_House256rgb.png', 0)

    _, binary = cv.threshold(image, 150, 1, cv.THRESH_BINARY)
    plt.subplot(121)
    plt.imshow(binary)

    # Erosion, Dilation, Opening, Closing에 대한 선택 입력
    kind = input('e(rosion) or d(ilation) or o(pening) or c(losing): ')
    n = int(input('iterations: '))

    morphologyed = None
    if kind == 'e' or kind == 'E':
        morphologyed = cv.morphologyEx(binary, cv.MORPH_ERODE, (3, 3), iterations=n)
    elif kind == 'd' or kind == 'D':
        morphologyed = cv.morphologyEx(binary, cv.MORPH_DILATE, (3, 3), iterations=n)
    elif kind == 'o' or kind == 'O':
        morphologyed = cv.morphologyEx(binary, cv.MORPH_OPEN, (3, 3), iterations=n)
    elif kind == 'c' or kind == 'C':
        morphologyed = cv.morphologyEx(binary, cv.MORPH_CLOSE, (3, 3), iterations=n)
    else:
        print("wrong key")

    plt.subplot(122)
    plt.imshow(morphologyed)
    plt.show()

if __name__ == "__main__":
    ex4()