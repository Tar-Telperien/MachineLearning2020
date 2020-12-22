#Goal: Correctly detect all 4 corners of all 50 sudoku boards which will eventually be in the sudoku_midterm folder, which will be pulled from sudoku_square images for the mean time
#Recommended starter code is from sudoku_grid_find.py
import numpy as np
import cv2

#Globals
#proc_img = cv2.imread("sudoku_square/sudoku5.png", cv2.IMREAD_GRAYSCALE)
proc_img = cv2.imread("sudoku_square/sudoku72.png", cv2.IMREAD_GRAYSCALE)

#Preprocessing step to darken black lines and make grey areas white
def enhance_writing(im):
    img_orig = im
    img = cv2.adaptiveThreshold(img_orig, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3) #lightens white, darkens dark
    '''
    cv2.imshow("Sawyer", img)
    cv2.imshow("Orig", img_orig)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    return img

# Finds the four segments bounding a sudoku board img using Harris corner
def sudoku_bounds(im):
    cv2.imshow("Original", im)
    img = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)
    img_col = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) # color
    harris = cv2.cornerHarris(im,4,5,0.04)

    img_col[harris > 0.01*harris.max()]=[255, 0, 0]
    return img_col
    '''
    cv2.imshow("Harris", img_col)
    cv2.waitKey(1000)
    return [(0, 0), (100, 100), (0, 100), (100, 0)] #this line is not necessary for Harris corner detection and was added to have something to return
    '''

final_img = sudoku_bounds(enhance_writing(proc_img))
cv2.imshow("Original", proc_img)
cv2.imshow("Processed", final_img)