import numpy as np
import cv2

def main():

    im = cv2.imread("plate.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image", im)
    f = None
    #Check BLUR kernel with different kernel size
    for i in range(3, 28, 4): 
        kernel = np.ones((i, i), np.float32) / (i*i)
        f = cv2.filter2D(im, cv2.CV_32F, kernel)  
        f = cv2.convertScaleAbs(f, alpha = 255/f.max())
        cv2.imshow("blur " + str(i), f)
        cv2.waitKey(0)

    retval1, b1 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
    retval, b2 = cv2.threshold(im-0.9*f, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary", b1)
    cv2.imshow("binary2", b2)
    cv2.waitKey(0)

    im = cv2.imread("code.png", cv2.IMREAD_GRAYSCALE)

    v_edge = np.matrix([[1, 1], [-1, -1]], np.float32)
    h_edge = np.matrix([[1, -1], [1, -1]], np.float32)
    v_sobel = np.matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    h_sobel = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    roberts = np.matrix([[1, 0], [0, -1]])

    kernels = (v_edge, h_edge, v_sobel, h_sobel, roberts);

    print(kernels)     

    for k in range(len(kernels)):
        kernel = kernels[k]
        f = cv2.filter2D(im, cv2.CV_32F, kernel)  
        f = cv2.convertScaleAbs(f, alpha = 255/f.max())
        cv2.imshow("filtered " + str(k), f)
        cv2.waitKey(0)


main()