
import cv2

class Contour:
    def __init__(self, array):
        self.array = array
        self._area = None
        self._perimeter = None
        self._hull = None
        self._rotrect = None
        self._moments = None

    @property
    def area(self):
        if self._area is None:
            self._area = cv2.contourArea(self.array)
        return self._area

    @property
    def perimeter(self):
        if self._perimeter is None:
            self._perimeter = cv2.arcLength(self.array, True)
        return self._perimeter

    @property
    def hull(self):
        if self._hull is None:
            self._hull = Contour(cv2.convexHull(self.array))
        return self._hull

    @property
    def rotrect(self):
        if self._rotrect is None:
            self._rotrect = cv2.minAreaRect(self.array)
        return self._rotrect

    @property
    def moments(self):
        if self._moments is None:
            self._moments = cv2.moments(self.array)
        return self._moments

    @property
    def squaredness(self):
        return 1 - abs(16*self.hull.area / self.hull.perimeter**2 - 1)**1.5

    @property
    def cx(self):
        return self.moments['m10'] / self.moments['m00']

    @property
    def cy(self):
        return self.moments['m01'] / self.moments['m00']
