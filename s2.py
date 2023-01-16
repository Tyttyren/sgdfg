import cv2

img = cv2.imread("D:\soft\python\python_project\lenet\data\WN6CW29.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

adaptive_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 333, 1)
cv2.imshow("g",gray_img)
cv2.waitKey(0)
cv2.imshow("1",adaptive_thresh)
cv2.waitKey(0)

def split_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 333, 1)
