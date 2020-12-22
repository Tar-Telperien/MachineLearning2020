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
#proc_img = cv2.imread("sudoku_square/sudoku5.png", cv2.IMREAD_GRAYSCALE)
proc_img = cv2.imread("sudoku_square/sudoku72.png", cv2.IMREAD_GRAYSCALE)

#Preprocessing step to darken black lines and make grey areas white
def enhance_writing(im):
    img_orig = im

    #lightens white, darkens dark
    img = cv2.adaptiveThreshold(img_orig, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
    #expands dark areas
    kernel = np.ones((5, 5), np.uint8)
    img2 = cv2.erode(img, kernel, iterations=1)
    return img2

#run connected-components on processed image to find largest component, which is the sudoku board, then return the corners of said component
def get_corners(img):
    largest_area = 0
    board = 0
    # Labels is an "image" where each pixel is the label of that pixel's connected component
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(255-img)
    #print(np.max(labels))
    '''
    for i in range(1, np.max(labels)):
        print(i)
        img2 = img.copy()
        img2[labels != i] = 255
        cv2.imshow("Img2", img2)
        cv2.waitKey()
    '''
    for i in range(1, np.max(labels)):
        img3 = img.copy()
        img3[labels != i] = 255
        if stats[i][4] > largest_area:
            largest_area = stats[i][4]
            board = i
            cv2.imshow("Component", img3)
            cv2.waitKey(200)
            print("component", i, stats[i])
            #cv2.destroyAllWindows()
    board_comp = stats[board]
    top_x = board_comp[0]
    top_y = board_comp[1]
    width = board_comp[2]
    height = board_comp[3]
    corners = [[top_x, top_y], [top_x+width, top_y], [top_x, top_y+height], [top_x+width, top_y+height]]
    return corners

def find_corners(im):
    corners = get_corners(enhance_writing(im))
    return corners

find_corners(proc_img)
#cv2.imshow("Original", proc_img)
#cv2.imshow("Processed", final_img)
'''cv2.waitKey()
cv2.destroyAllWindows()'''