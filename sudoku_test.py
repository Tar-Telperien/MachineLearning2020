from sudoku_corner_finder import find_corners as corners
import cv2

img = cv2.imread("sudoku_square/sudoku72.png", cv2.IMREAD_GRAYSCALE)

corners(img)