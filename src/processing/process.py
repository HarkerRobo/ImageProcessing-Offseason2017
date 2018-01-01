import itertools
import cv2
import numpy as np
from .contour import Contour
from . import scoring

KERNEL = np.ones((6, 1), np.uint8)
KERNEL2 = np.ones((3, 3), np.uint8)

DEBUG = False

def mask(img):
    """Return a mask where tape pixels are white."""
    imgf = img.astype(int) # Because we're going to use values > 255 (a uint8 will wrap around)

    # Custom filtering
    # STEP 1: Subtract the red and blue channels from the green channel
    filtered = np.sqrt(np.maximum(np.square(imgf[:,:,1]) - np.square(imgf[:,:,2] + imgf[:,:,0]), 0)).astype(np.uint8, copy=False)
    if DEBUG:
        cv2.imshow('f', filtered)
    # STEP 2: Compute the average of the 40 brightest pixels
    ind = np.unravel_index(np.argpartition(filtered, -40, None)[-40:], filtered.shape)
    maxi = np.average(filtered[ind])

    # STEP 3: Use this average the threshold the image
    pivot = maxi * np.square(maxi/255)
    _, th = cv2.threshold(filtered, pivot, 255, cv2.THRESH_BINARY)

    # Close some gaps and remove noise
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, KERNEL2, iterations=2)

    return opening

def width(cnt):
    """Return the smallest dimension of a contour's minimum bounding rect."""
    return min(cnt.rotrect[1])

def averagewidth(c1, c2):
    """Return the average width of two contours."""
    return (width(c1) + width(c2)) / 2

def goodtogether(c1, c2):
    """Return True if the two pieces of tape can make a whole."""
    div = averagewidth(c1, c2)
    offx = abs(c1.cx -c2.cx) / div
    offy = abs(c1.cy -c2.cy) / div
    return offx < 4 and offy < 0.3

def contours(ori, img):
    """Return checked contours and the bounding box of the tape in the image."""
    _, cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = [Contour(c) for c in cnt]
    if DEBUG:
        cv2.drawContours(ori, [c.array for c in cnt], -1, (0, 0, 255), 2)

    # Run some filters to discard contours that defintely aren't part of the target
    valid_short = list(filter(scoring.isvalidshort, cnt))
    valid_long = list(filter(scoring.isvalidlong, valid_short))
    if DEBUG:
        cv2.drawContours(ori, [c.array for c in valid_long], -1, (0, 100, 255), 2)

    thecnts = None
    if len(valid_long) >= 2:
        scores = np.array([scoring.weighted_score_long(c) for c in valid_long])
        maxes = np.argpartition(scores, -2, None)[-2:]
        best = [valid_long[i] for i in maxes]
        if DEBUG:
            cv2.drawContours(ori, [c.array for c in best], -1, (0, 255, 255), 2)
        if goodtogether(*best):
            # If there are two long contours that can make up a target then we're good
            thecnts = best
            if DEBUG:
                cv2.drawContours(ori, [c.array for c in best], -1, (0, 255, 0), 2)

    # Assuming there aren't two nice contours making up the target, see if we can find 3, of which 2
    # can be hulled together
    if thecnts is None and len(valid_short) >= 3:
        # if DEBUG:
        #     cv2.drawContours(ori, [c.array for c in valid_short], -1, (0, 100, 255), 2)
        for combination in itertools.permutations(valid_short, 2):
            # For some optimization first find the two to combine then loop looking for the third
            i, j = combination
            if abs(i.cx - j.cx) / averagewidth(i, j) > 1:
                continue
            c1 = Contour(cv2.convexHull(np.concatenate((i.array, j.array))))
            if DEBUG:
                cv2.drawContours(ori, [c1.array], -1, (255, 100, 0), 2)
            # Find the other contour
            for c2 in valid_short:
                if c2 == i or c2 == j:
                    continue
                if goodtogether(c1, c2):
                    thecnts = [c1, c2]
                    if DEBUG:
                        cv2.drawContours(ori, [c.array for c in thecnts], -1, (0, 255, 0), 2)
                    break
            # Break out of the outer for if need be
            if thecnts is not None:
                break

    corns = None
    if thecnts is not None:
        bbx, bby, bbw, bbh = corns = cv2.boundingRect(np.concatenate((thecnts[0].array, thecnts[1].array)))
        if DEBUG:
            cv2.rectangle(ori, (bbx, bby), (bbx+bbw, bby+bbh), (255, 255, 255), 8)
            cv2.rectangle(ori, (bbx, bby), (bbx+bbw, bby+bbh), (255, 0, 0), 4)

    if DEBUG:
        ori = cv2.resize(ori, None, None, 2, 2)
        for cnt in valid_long:
            M = cnt.moments
            cx = int(M['m10'] / M['m00'])*2
            cy = int(M['m01'] / M['m00'])*2
            y = np.random.randint(-100, 100)
            off = np.random.randint(10, 100)
            for k, v in scoring.scorelong(cnt).items():
                text = '{}: {:.2f}'.format(k, v)
                cv2.putText(ori, text, (cx+off, cy+y-25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                y += 15

        cv2.imshow('i', ori)

    return valid_long, corns

# Combine the two functions
process = lambda x: contours(x, mask(x))
