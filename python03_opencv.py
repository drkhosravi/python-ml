import cv2
#this must be installed using "pip install opencv-python"
#in vs code use this: python -m pip install opencv-python  
#or
#& "C:/Program Files (x86)/Microsoft Visual Studio/Shared/Python37_64/python.exe" -m pip install -U pylint --user
    # -m (module: to allow modules to be located using the Python module namespace for execution as scripts.)

#check version:
print ("cv2 version = ", cv2.__version__)

im = cv2.imread("code.png")
cv2.imshow("image", im)

print(im.shape)
print(im.dtype)

print(im[100, 100])

subimage = im[10:100, 20:320]
cv2.imshow("subimage", subimage)

#clone a new image
# im2 = im.copy();
# print(im[10, 10])
# im2[10,10] = [100, 120, 130]
# print(im[10, 10])
# print(im2[10, 10])
im_org = im.copy()

#convert to grayscale
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", im_gray)

#convert to binary (input should be grayscale)
retval, im_bin = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY)
retval, im_bin2 = cv2.threshold(im_gray, 30, 255, cv2.THRESH_BINARY)
retval, im_bin3 = cv2.threshold(im_gray, 220, 255, cv2.THRESH_BINARY)
retval, im_bin4 = cv2.threshold(im_gray, 220, 255, cv2.THRESH_OTSU)
cv2.imshow("bin", im_bin)
cv2.imshow("bin2", im_bin2)
cv2.imshow("bin3", im_bin3)
cv2.imshow("bin4", im_bin4)
#better approach
im_bin = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 1)
cv2.imshow("adaptive bin", im_bin)
#working with RGB channels
im[:,:,2] = 0
#cv2.namedWindow("no red", 2)
cv2.imshow("no red", im)
#cv2.waitKey(0)


cap = cv2.VideoCapture(0)#'D:\\Dataset\\ANPR\\1.mov')
cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.namedWindow("edge", cv2.WINDOW_NORMAL)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 70, 150)
    # Display the resulting frame
    #cv2.imshow('video', frame)
    cv2.imshow('edge', edge)
    if cv2.waitKey(33) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
