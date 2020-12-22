#Goal: Correctly detect all 4 corners of all 50 sudoku boards which will eventually be in the sudoku_midterm folder, which will be pulled from sudoku_square images for the mean time

"""
Erode image with 5x5 kernel
Run connected components (stats is a 5-tuple: x and y of top right of bounding rect of component; width and height of bounding rect, number of pixels in component)
Take biggest
Find corners
Done!
"""

import numpy as np
import cv2

#Globals
proc_img = cv2.imread("sudoku_square/sudoku5.png", cv2.IMREAD_GRAYSCALE)
#proc_img = cv2.imread("sudoku_square/sudoku72.png", cv2.IMREAD_GRAYSCALE)

#Preprocessing step to darken black lines and make grey areas white
def enhance_writing(im):
    img_orig = im

    #lightens white, darkens dark
    img = cv2.adaptiveThreshold(img_orig, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
    #expands dark areas
    kernel = np.ones((3, 3), np.uint8)
    img2 = cv2.erode(img, kernel, iterations=1)

    return img2

#run connected-components on processed image to find largest component, and return image of said component
def get_board(img):
    largest_area = 0
    board_img = img
    # Labels is an "image" where each pixel is the label of that pixel's connected component
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(255-img)

    for i in range(1, np.max(labels)):
        img3 = img.copy()
        img3[labels != i] = 255
        if stats[i][4] > largest_area:
            largest_area = stats[i][4]
            board_img = img3

    return board_img

'''
def sudoku_bounds(im):
    img_col = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) # color
    harris = cv2.cornerHarris(im,4,5,0.04)

    img_col[harris > 0.01*harris.max()]=[255, 0, 0]
    cv2.imshow("Harris corner", img_col)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img_col

#get_corners takes an image where the corners have been marked in blue
#get_corners is also my problem because it's not doing its job correctly
def get_corners(img):
    x = []
    y = []
    i_counter = len(img)
    j_counter = len(img[0])
    for i in range(i_counter):
        for j in range(j_counter):
            if img[i][j][0] == 255:
                y.append(i)
                x.append(j)
    upper = min(y)
    lower = max(y)
    left = min(x)
    right = max(x)
    return_list = ([upper, left], [upper, right], [lower, left], [lower, right])
    return return_list

def find_corners(img):
    return get_corners(sudoku_bounds(get_board(enhance_writing(img))))
'''

def get_corners(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #original line

    if cv2.__version__.startswith("3"): # findContours varies by CV version!
        _, im2, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        im2, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_corners(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #original line
    im2, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_corners(img):
    return get_corners(get_board(enhance_writing(img)))

find_corners(proc_img)