import numpy as np

# 3D points are being read from the text file
f = open("C:/Users/Rumman Ali/Downloads/3D.txt", "r")
data = f.read()
data1 = np.array(data.split())
data2 = data1.astype(np.float)
data3 = np.delete(data2, 0)
threeDtPoints = data3.reshape(int(data2[0]), 3)
coordinates3D = []
i=0
for points in threeDtPoints:
    list=[]
    for points1 in points:
        list.append(points1)
    coordinates3D.append(list)

# 2D points are being read from the text file
f = open("C:/Users/Rumman Ali/Downloads/2D.txt", "r")
data = f.read()
data1 = np.array(data.split())
data2 = data1.astype(np.float)
data3 = np.delete(data2, 0)
twoDPoints = data3.reshape(int(data2[0]), 2)
coordinates2D =[]
for points in twoDPoints:
    list=[]
    for points1 in points:
        list.append(points1)
    coordinates2D.append(list)

numberOfPoints = len(coordinates3D)

# Construction of A matrix
A = np.zeros((2 * numberOfPoints, 12))
for i in range(numberOfPoints):
    u, v = coordinates2D[i]
    X, Y, Z = coordinates3D[i]
    A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
    A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]

# Apply SVD on matrix A
s, d, vT = np.linalg.svd(A)

# The last column of the V is the projection matrix
projectionMatrix = vT[-1]/vT[-1,-1]
projectionMatrix = projectionMatrix.reshape((3, 4))
print("Projection matrix:")
print(projectionMatrix)

newPixels = []
totalErrors = []

for i in range(numberOfPoints):
    newPoint = np.matmul(projectionMatrix, np.append(coordinates3D[i], 1))
    newPixel = newPoint[:2] / newPoint[2]
    newPixels.append(newPixel)

    error = np.linalg.norm(newPixel - coordinates2D[i])
    totalErrors.append(error)

errorAverage = np.mean(totalErrors)

print("\nError Average for newly projected pixels:", errorAverage)

print("\nNew pixels will be:")
point=1
for pixel in newPixels:
    print(f"Point {point}: {pixel}")
    print(f"Existing 2D Point {point}: {coordinates2D[point-1]}")
    point=point+1
