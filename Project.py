import cv2
import numpy as np


def fill_holes(mask):
    maskFloodfill = mask.copy()
    h, w = maskFloodfill.shape[:2]
    maskTemp = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
    mask2 = cv2.bitwise_not(maskFloodfill)
    return mask2 | mask


def find_max_contour(contours):
    max_area = 0
    max_cont = None
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > max_area:
            max_area = area
            max_cont = cont
    return max_cont


img = cv2.imread("Test.jpg")

outImage = img.copy()

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

scaleFactor = 1.3  # Parameter specifying how much the image size is reduced at each image scale.
minNeighbors = 4
flag = 0
minSize = (25, 25)

eyes = eye_cascade.detectMultiScale(img, scaleFactor, minNeighbors, flag, minSize)

thickness = 3
focusEye = img.copy()
for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(focusEye, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), thickness)

scale_percent = 30  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

for (ex, ey, ew, eh) in eyes:
    eyeImage = img[ey:ey + eh, ex:ex + ew]

    # cv2.imshow('Just the eye', eyeImage)
    # cv2.waitKey()

    b, g, r = cv2.split(eyeImage)

    bg = cv2.add(b, g)

    # print(bg)
    # print("\n")
    #
    # print(r)
    # print("\n")

    # mask = (r > 150) & (r > bg) detecteaza doar rosul intens
    mask = ((r > (bg - 20)) & (r > 80))  # pentru asta am nevoie de o metoda care sa gaseasca zona cea mai mare

    mask = mask.astype(np.uint8) * 255
    # cv2.imshow('Mask', mask)

    # mask = fill_holes(mask)
    # mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    maxCont = find_max_contour(contours)
    mask = mask * 0
    cv2.drawContours(mask, [maxCont], 0, 255, 1)  # ultimul param e thickness-ul

    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5)))
    # mask = cv2.dilate(mask, (3, 3), iterations=3)

    mask = fill_holes(mask)
    # mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)

    # cv2.imshow('ImproveMask', mask)
    # cv2.waitKey()

    mean = bg / 2

    mask = mask.astype(bool)[:, :, np.newaxis]

    mean = mean[:, :, np.newaxis]
    eyeCorrection = eyeImage.copy()

    np.copyto(eyeCorrection, mean, casting='unsafe', where=mask)

    outImage[ey:ey + eh, ex:ex + ew, :] = eyeCorrection

cv2.imshow('Correct image', outImage)
cv2.waitKey()



