import cv2
import numpy as np

def calculate_zncc(first_patch, second_patch):
    
    ### in some cases, there are empty patches
    if first_patch.size == 0 or second_patch.size == 0:
        return 0.0
    
    min_height = min(first_patch.shape[0], second_patch.shape[0])
    min_width = min(first_patch.shape[1], second_patch.shape[1])

    
    first_patch = first_patch[:min_height, :min_width]
    second_patch = second_patch[:min_height, :min_width]
    
    first_patch_mean = np.mean(first_patch)
    second_patch_mean = np.mean(second_patch)
    
    first_patch_std = np.std(first_patch)
    second_patch_std = np.std(second_patch)
    
    correlation = np.mean((first_patch - first_patch_mean) * (second_patch - second_patch_mean)) / (first_patch_std * second_patch_std)
    
    zncc_score = (correlation + 1) / 2
    
    return zncc_score

# Mouse call back function for checking click on the left image
def MouseCallback(event, x, y, flags, param):
    global checkClicked, clickedPoint
    if event == cv2.EVENT_LBUTTONUP:
        checkClicked = True
        clickedPoint = (x, y)

# Function for creating and passing the fundamental matrix
def calculate_fundamental_matrix(grayImageLeft, grayImageRight):
    orb = cv2.ORB_create()
    keypointsLeft, descriptorsLeft = orb.detectAndCompute(grayImageLeft, None)
    keypointsRight, descriptorsRight = orb.detectAndCompute(grayImageRight, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptorsLeft, descriptorsRight)

    pointsLeft = np.float32([keypointsLeft[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pointsRight = np.float32([keypointsRight[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return cv2.findFundamentalMat(pointsLeft, pointsRight, cv2.FM_RANSAC, ransacReprojThreshold=1.0)

def calculate_projection_matrix_image(object_points, image_points):

    no_points = len(image_points)

    A = np.zeros((2*no_points, 12))
    b = np.zeros((2*no_points, 1))

    for i in range(no_points):
        x_axis, y_axis, z_axis = object_points[i]
        u_axis, v_axis = image_points[i]

        A[2*i] = [x_axis, y_axis, z_axis, 1, 0, 0, 0, 0, -u_axis*x_axis, -u_axis*y_axis, -u_axis*z_axis, -u_axis]
        A[2*i+1] = [0, 0, 0, 0, x_axis, y_axis, z_axis, 1, -v_axis*x_axis, -v_axis*y_axis, -v_axis*z_axis, -v_axis]
        b[2*i] = u_axis
        b[2*i+1] = v_axis

    ##calcaute svd using numpy linear alg module
    _, _, prp_comp = np.linalg.svd(A)
    final_m = prp_comp[-1, :]

    calibration_matrix = final_m.reshape((3, 4))

    return calibration_matrix

def Final3DPoints(point1, point2, M1, M2):
    # Convert the 2D pixel coordinates to homogeneous coordinates
    p1_homogeneous = np.array([[point1[0]], [point1[1]], [1]])
    p2_homogeneous = np.array([[point2[0]], [point2[1]], [1]])

    # Create the A matrix for DLT triangulation
    A = np.vstack([
        p1_homogeneous[0, 0] * M1[2, :] - M1[0, :],
        p1_homogeneous[1, 0] * M1[2, :] - M1[1, :],
        p2_homogeneous[0, 0] * M2[2, :] - M2[0, :],
        p2_homogeneous[1, 0] * M2[2, :] - M2[1, :]
    ])
    print("A:")
    print(A)

    # Use Singular Value Decomposition (SVD) to solve for the 3D coordinates
    _, _, V = np.linalg.svd(A)
    X_homogeneous = V[-1, :]  # Last row of V

    # Normalize the homogeneous 3D coordinates to get the real 3D coordinates (X, Y, Z)
    X_homogeneous /= X_homogeneous[-1]
    X, Y, Z = X_homogeneous[:-1]
    print('Corresponding 3D point is: ',X,' ',Y,' ',Z)
    
# Read the images
imgLeft = cv2.imread('givenLeftImage.jpeg')
imgRight = cv2.imread('givenRightImage.jpeg')

# Convert the images to grayscale
grayLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)

FundamentalMatrix, mask = calculate_fundamental_matrix(grayLeft, grayRight)
# Create a window to display the images
cv2.namedWindow('Left Image')

# Variables to store user interaction
checkClicked = False
clickedPoint = (-1, -1)

# Set the mouse callback
cv2.setMouseCallback('Left Image', MouseCallback)

#print('Fundamental Matrix:')
#print(FundamentalMatrix)

# Example usage:
world_points = np.array([[0, 1.5, 3.8], [0, 22.5, 3.8], [0, 22.5, 18.8], [0, 1.5, 18.8]])  # Replace with 3D world points (6 x 3)
left_image_points = np.array([[255, 381], [84, 352], [93, 199], [260, 190]])  # Replace with 2D points in the left image (6 x 2)

pli = calculate_projection_matrix_image(world_points, left_image_points)
print('Left Image Projection Matrix:')
print(pli)
world_points1 = np.array([[24.3, 0, 3.8], [3.3, 0, 3.8], [3.3, 0, 18.8], [24.3, 0, 18.8]])  # Replace with 3D world points (6 x 3)
right_image_points = np.array([[259, 403], [104, 363], [114, 197], [268, 176]])  # Replace with 2D points in the right image (6 x 2)

pri = calculate_projection_matrix_image(world_points1, right_image_points)
print('Right Image Projection Matrix:')
print(pri)
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
        sizeofWindow = 21
        leftWindow = grayLeft[clickedPoint[1]-sizeofWindow:clickedPoint[1]+sizeofWindow, clickedPoint[0]-sizeofWindow:clickedPoint[0]+sizeofWindow]
        for i in range(0, imgRight.shape[1]):
            newY = int(((i-x0)*(y0-y1))/(x0-x1) + y0);
            rightWindow = grayRight[newY-sizeofWindow:newY+sizeofWindow, i-sizeofWindow:i+sizeofWindow]
            znccScore=-1e8
            if rightWindow.size == leftWindow.size:
                znccScore = calculate_zncc(rightWindow, leftWindow)
            
         #   averagePoint=(int(i), int(newY))
          #  differencePoint=(avera)
            if znccScore > perfectScore and newY > 0:
                perfectScore = znccScore
                perfectPoint = (int(i), int(newY))
               # cv2.drawMarker(imgRight, perfectPoint, (0, 255, 255), markerType, markerSize, thickness)
                
        # Draw marker on the right image depending on the click of the left image
        print('Left Image point:', clickedPoint);
        print('Right Image point:', perfectPoint);
        cv2.drawMarker(imgRight, perfectPoint, (255, 255, 255), markerType, markerSize, thickness)
        FundamentalMatrixTranspose=FundamentalMatrix.transpose()
        InverseFundamentalMatrix=np.linalg.inv(FundamentalMatrixTranspose)
        newClickedPoint=np.array([clickedPoint[0], clickedPoint[1], 1])
        newClickedPoint.reshape(3,1)
        rightpoint=newClickedPoint@InverseFundamentalMatrix
        rightpoint=rightpoint/rightpoint[2]
        rightpoint=rightpoint[:2]
        newRightPoint=(int(rightpoint[0]), int(rightpoint[1]))
        cv2.drawMarker(imgRight, newRightPoint, (255, 255, 0), markerType, markerSize, thickness)
        lineAnother = np.dot(FundamentalMatrixTranspose, np.array([perfectPoint[0], perfectPoint[1], 1]))
        x2, y2 = 0, int(-lineAnother[2] / lineAnother[1])
        x3, y3 = imgLeft.shape[1], int((-lineAnother[2] - lineAnother[0] * imgLeft.shape[1]) / lineAnother[1])
        cv2.line(imgLeft, (x2, y2), (x3, y3), (255, 0, 255), 2)
        Final3DPoints(clickedPoint, perfectPoint, pli, pri)

    # Right image to be displayed
    cv2.imshow('Right Image', imgRight)

    # Check for key press
    key = cv2.waitKey(33)
    if key==27:    # Esc key to stop
        break

# Cleanup
cv2.destroyAllWindows()
