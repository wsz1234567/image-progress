import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
from pylsd.lsd import lsd


def get_corners(img):

    lines = lsd(img)

    # massages the output from LSD
    # LSD operates on edges. One "line" has 2 edges, and so we need to combine the edges back into lines.
    # 1. separate out the lines into horizontal and vertical lines.
    # 2. Draw the horizontal lines back onto a canvas, but slightly thicker and longer.
    # 3. Run connected-components on the new canvas
    # 4. Get the bounding box for each component, and the bounding box is final line.
    # 5. The ends of each line is a corner
    # 6. Repeat for vertical lines
    # 7. Draw all the final lines onto another canvas. Where the lines overlap are also corners

    corners = []
    if lines is not None:
        # separate out the horizontal and vertical lines, and draw them back onto separate canvases
        lines = lines.squeeze().astype(np.int32).tolist()
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2, _ = line
            if abs(x2 - x1) > abs(y2 - y1):
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
            else:
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

        lines = []

        # find the horizontal lines (connected-components -> bounding boxes -> final lines)
        (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_x = np.amin(contour[:, 0], axis=0) + 2
            max_x = np.amax(contour[:, 0], axis=0) - 2
            left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
            right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
            lines.append((min_x, left_y, max_x, right_y))
            cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 255, 5)
            cv2.imshow('canvas1',horizontal_lines_canvas)
            corners.append((min_x, left_y))
            corners.append((max_x, right_y))

        # find the vertical lines (connected-components -> bounding boxes -> final lines)
        (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_y = np.amin(contour[:, 1], axis=0) + 2
            max_y = np.amax(contour[:, 1], axis=0) - 2
            top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
            bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
            lines.append((top_x, min_y, bottom_x, max_y))
            cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 255, 5)
            cv2.imshow('canvas2', vertical_lines_canvas)
            corners.append((top_x, min_y))
            corners.append((bottom_x, max_y))

        # find the corners
        corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
        corners += zip(corners_x, corners_y)

    # remove corners in close proximity

    return corners


image = cv2.imread('result6.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
corners = get_corners(gray)

for corner in corners:
    x, y = corner
    cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

cv2.imshow('Intersections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

