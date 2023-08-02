import cv2
import numpy as np

# function for zncc score
def znccCalculation(leftImage, rightImage, point):
    average1 = 0
    average2 = 0
    for i in range(-point, point+1):
        for j in range(-point, point+1):
            try:
                x, y = leftImage[point+i][point+j], rightImage[point+i][point+j]
            except IndexError:
                continue
            average1 += leftImage[point+i][point+j]
            average2 += rightImage[point+i][point+j]
    average1 = float(average1)/(2*point+1)**2
    average2 = float(average2)/(2*point+1)**2

    standardDeviation1 = 0
    standardDeviation2 = 0
    for i in range(-point, point+1):
        for j in range(-point, point+1):
            try:
                x, y = (leftImage[point+i][point+j] - average1)**2, (rightImage[point+i][point+j] - average2)**2 
            except IndexError:
                continue
            standardDeviation1 += (leftImage[point+i][point+j] - average1)**2
            standardDeviation2 += (rightImage[point+i][point+j] - average2)**2
    standardDeviation1 = (standardDeviation1**0.5)/(2*point+1)
    standardDeviation2 = (standardDeviation2**0.5)/(2*point+1)

    total = 0
    for i in range(-point, point+1):
        for j in range(-point, point+1):
            try:
                x = (leftImage[point+i][point+j] - average1)*(rightImage[point+i][point+j] - average2)
            except IndexError:
                continue
            total += (leftImage[point+i][point+j] - average1)*(rightImage[point+i][point+j] - average2)
    if standardDeviation1 == 0:
        standardDeviation1 = 0.01
    if standardDeviation2 == 0:
        standardDeviation2 = 0.01

    return float(total)/((2*point+1)**2 * standardDeviation1 * standardDeviation2)

# Mouse call back function for checking click on the left image
def MouseCallback(event, x, y, flags, param):
    global checkClicked, clickedPoint
    if event == cv2.EVENT_LBUTTONUP:
        checkClicked = True
        clickedPoint = (x, y)

# Function for creating and passing the fundamental matrix
def CalculateFundamentalMatrix(grayImageLeft, grayImageRight):
    orb = cv2.ORB_create()
    keypointsLeft, descriptorsLeft = orb.detectAndCompute(grayImageLeft, None)
    keypointsRight, descriptorsRight = orb.detectAndCompute(grayImageRight, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptorsLeft, descriptorsRight)

    pointsLeft = np.float32([keypointsLeft[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pointsRight = np.float32([keypointsRight[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return cv2.findFundamentalMat(pointsLeft, pointsRight, cv2.FM_RANSAC, ransacReprojThreshold=1.0)
    
# Read the images
imgLeft = cv2.imread('givenLeftImage.jpeg')
imgRight = cv2.imread('givenRightImage.jpeg')

# Convert the images to grayscale
grayLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)

FundamentalMatrix, mask = CalculateFundamentalMatrix(grayLeft, grayRight)
print(FundamentalMatrix)
# Create a window to display the images
cv2.namedWindow('Left Image')

# Variables to store user interaction
checkClicked = False
clickedPoint = (-1, -1)

# Set the mouse callback
cv2.setMouseCallback('Left Image', MouseCallback)

print('Fundamental Matrix:')
print(FundamentalMatrix)

while True:
    # Left image to be displayed
    cv2.imshow('Left Image', imgLeft)

    if checkClicked:
        checkClicked = False

        markerType = cv2.MARKER_CROSS
        markerSize = 15
        thickness = 2
        color = (0, 255, 0)
        # Drwaing the marker on the left image
        cv2.drawMarker(imgLeft, clickedPoint, color, markerType, markerSize, thickness)

        # Epipolar line drawing on the right image
        line = np.dot(FundamentalMatrix, np.array([clickedPoint[0], clickedPoint[1], 1]))
        x0, y0 = 0, int(-line[2] / line[1])
        x1, y1 = imgRight.shape[1], int((-line[2] - line[0] * imgRight.shape[1]) / line[1])
        cv2.line(imgRight, (x0, y0), (x1, y1), (255, 0, 255), 2)

        perfectPoint = (-1e8, -1e8)
        perfectScore = -1e8
        sizeofWindow = 9
        leftWindow = grayLeft[clickedPoint[1]-sizeofWindow:clickedPoint[1]+sizeofWindow, clickedPoint[0]-sizeofWindow:clickedPoint[0]+sizeofWindow]
        for i in range(0, imgRight.shape[1]):
            newY = int(((i-x0)*(y0-y1))/(x0-x1) + y0);
            rightWindow = grayRight[newY-sizeofWindow:newY+sizeofWindow, i-sizeofWindow:i+sizeofWindow]
            znccScore=-1e8
            if rightWindow.size == leftWindow.size:
                znccScore = znccCalculation(rightWindow, leftWindow, sizeofWindow)

            if znccScore > perfectScore and newY > 0:
                perfectScore = znccScore
                perfectPoint = (int(i), int(newY))
              #  cv2.drawMarker(imgRight, perfectPoint, (0, 255, 255), markerType, markerSize, thickness)
                
        # Draw marker on the right image depending on the click of the left image
        print('Left Image point:', clickedPoint);
        print('Right Image point:', perfectPoint);
        cv2.drawMarker(imgRight, perfectPoint, (255, 255, 255), markerType, markerSize, thickness)
        FundamentalMatrixTranspose=FundamentalMatrix.transpose()
        InverseFundamentalMatrix=np.linalg.inv(FundamentalMatrixTranspose)
        print('Inverse: ',InverseFundamentalMatrix)
        rightPoint=clickedPoint@InverseFundamentalMatrix
        lineAnother = np.dot(FundamentalMatrixTranspose, np.array([perfectPoint[0], perfectPoint[1], 1]))
        x2, y2 = 0, int(-lineAnother[2] / lineAnother[1])
        x3, y3 = imgLeft.shape[1], int((-lineAnother[2] - lineAnother[0] * imgLeft.shape[1]) / lineAnother[1])
        cv2.line(imgLeft, (x2, y2), (x3, y3), (255, 0, 255), 2)

    # Right image to be displayed
    cv2.imshow('Right Image', imgRight)

    # Check for key press
    key = cv2.waitKey(33)
    if key==27:    # Esc key to stop
        break

# Cleanup
cv2.destroyAllWindows()
