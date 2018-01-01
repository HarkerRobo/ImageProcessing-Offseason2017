"""
Various functions to score and validate contours.
"""

import numpy as np

def isvalidshort(cnt):
    """Return True if the contour is possibly a valid target piece."""
    if not cnt.area > 30:
        return False

    if not cnt.area / cnt.hull.area > 0.5:
        return False

    w, h = cnt.rotrect[1]
    rotrectarea = w * h
    if not cnt.area/rotrectarea > 0.4:
        return False

    return True

def isvalidlong(cnt):
    """Return True if the contour could be a full half of the target."""
    if not cnt.area > 100:
        return False

    w, h = cnt.rotrect[1]
    if not 0.5 < min(w, h)*2.5 / max(w, h) < 2:
        return False
    return cnt


def scorelong(cnt):
    """Return the score of several properties of the given contours, on a scale of 0 to 1."""
    hullscore = cnt.area/cnt.hull.area

    w, h = cnt.rotrect[1]
    arscore = np.sqrt(min(w, h)*3 / max(w, h))
    if arscore > 1:
        arscore = 1 / arscore

    rotrectarea = w * h
    rotrectscore = cnt.area / rotrectarea

    areascore = 2 / (1 + 1.1**(-cnt.area/100)) - 1

    return {
        'hull': hullscore,
        'ar': arscore,
        'rotrect': rotrectscore,
        'area': areascore
    }

def scoreshort(cnt):
    """Return the score of a contour that represents only a part of the target."""
    hullscore = cnt.area/cnt.hull.area

    w, h = cnt.rotrect[1]
    rotrectarea = w * h
    rotrectscore = cnt.area / rotrectarea

    areascore = 2 / (1 + 1.1**(-cnt.area/100)) - 1

    return {
        'hull': hullscore,
        'rotrect': rotrectscore,
        'area': areascore
    }


def weighted_score_long(cnt):
    """Return a total score from 0 to 1 on how likely a contour is to be a target."""
    s = scorelong(cnt)
    return 0.1*s['hull'] + 0.6*s['ar'] + 0.1*s['rotrect'] + 0.2*s['area']


def weighted_score_short(cnt):
    """Return a total score from 0 to 1 on how likely a contour is to be a target."""
    s = scoreshort(cnt)
    return (s['hull'] + s['rotrect'] + s['area']) / 3
