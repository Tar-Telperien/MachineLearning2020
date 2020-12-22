#Goal: Correctly detect all 4 corners of all 50 sudoku boards which will eventually be in the sudoku_midterm folder, which will be pulled from sudoku_square images for the mean time
#Recommended starter code is from sudoku_grid_find.py

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
#proc_img = cv2.imread("sudoku_square/sudoku5.png", cv2.IMREAD_GRAYSCALE)
proc_img = cv2.imread("sudoku_square/sudoku72.png", cv2.IMREAD_GRAYSCALE)
final_img = proc_img
harris_threshold = 0

#Preprocessing step to darken black lines and make grey areas white
def enhance_writing(im):
    img_orig = im
    #lightens white, darkens dark
    img = cv2.adaptiveThreshold(img_orig, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)


    '''
    cv2.imshow("Sawyer", img)
    cv2.imshow("Orig", img_orig)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    #expands dark areas
    kernel = np.ones((5, 5), np.uint8)
    img2 = cv2.erode(img, kernel, iterations=1)

    # Labels is an "image" where each pixel is the label of that pixel's connected component
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(255-img2)
    print(np.max(labels))
    for i in range(np.max(labels)):
        img3 = img2.copy()
        img3[labels != i] = 255

        '''
        cv2.imshow("One Component", img3)
        if i == 33: cv2.waitKey()
        cv2.waitKey(20)
        print("component", i, stats[i])
        '''

    return img

#get_corners takes an image where the corners have been marked in blue
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




# Finds the four segments bounding a sudoku board img using Harris corner
def sudoku_bounds(im):
    cv2.imshow("Original", im)
    img = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)
    img_col = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) # color
    harris = cv2.cornerHarris(im,4,5,0.04)

    img_col[harris > 0.01*harris.max()]=[255, 0, 0]
    #return img_col

    corners = get_corners(img_col)
    return corners

    '''
    cv2.imshow("Harris", img_col)
    cv2.waitKey(1000)
    return [(0, 0), (100, 100), (0, 100), (100, 0)] #this line is not necessary for Harris corner detection and was added to have something to return
    '''

def find_corners(im):
    #final_img = sudoku_bounds(enhance_writing(proc_img))
    corners = sudoku_bounds(enhance_writing(proc_img))
    return corners

find_corners(proc_img)
#cv2.imshow("Original", proc_img)
#cv2.imshow("Processed", final_img)
cv2.waitKey()
cv2.destroyAllWindows()