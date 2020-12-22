import numpy as np
import cv2

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

def sudoku_bounds(im):
    img_col = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) # color
    harris = cv2.cornerHarris(im,4,5,0.04)

    img_col[harris > 0.01*harris.max()]=[255, 0, 0]
    cv2.imshow("Harris corner", img_col)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img_col

#get_corners takes an image where the corners have been marked in blue
def get_corners(img):
    xy = []
    i_counter = len(img)
    j_counter = len(img[0])
    for i in range(i_counter):
        for j in range(j_counter):
            if img[i][j][0] == 255 and img[i][j][1] == 0 and img[i][j][2] == 0:
                xy.append([j, i])
    return_list = cv2.boxPoints(cv2.minAreaRect(np.array(xy)))
    return return_list

def find_corners(img):
    return get_corners(sudoku_bounds(get_board(enhance_writing(img))))
