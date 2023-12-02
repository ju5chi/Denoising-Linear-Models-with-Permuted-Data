import cv2
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
import numpy as np
import matplotlib.pyplot as plt

def detectCorners(imageName, parametersHarris):
	'''
		Copied from: https://machinelearningmastery.com/opencv_edges_and_corners/

		IN: imageName <-> the file name as '____.jpg'
			showImage <-> True/False; whether the image should be dispalyed or not
				          (hangs in jupyter notebook)
		OUT: numpy.ndarray of the corner coordinates in (x, y) order
	'''
	# Load the image
	img = cv2.imread(imageName)

	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# Detect corners using the Harris method
	dst = cv2.cornerHarris(gray, *parametersHarris)

	# Create a boolean bitmap of corner positions
	corners = dst > 0.05 * dst.max()

	# Find the coordinates from the boolean bitmap
	coord = np.argwhere(corners)

	swappedCoords = coord.copy()
	swappedCoords[:, 0] = coord[:, 1]
	swappedCoords[:, 1] = coord[:, 0]

	return swappedCoords


def plotImage(imageName, cornerPoints=np.array([])):
	'''
		Does all the plotting work for an image
		
		IN: imageName 	 <-> the file name as '____.jpg'
			cornerPoints <-> numpy.ndarray where the i-th row contains [x, y] coordinates of the i-th corner
	'''
	img = mpimg.imread(imageName)
	fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
	imgplot = ax.imshow(img)
	if cornerPoints.size > 0:
		ax.scatter(cornerPoints[:, 0], cornerPoints[:, 1], color='tab:blue', alpha=0.5)
    
    
def transformImage(imageName, T, parametersHarris, returnLengthTransformedEdge=False):
	# Load the image
	img = cv2.imread(imageName)
	yLen, xLen = img.shape[:2]
	cornersImage = np.array([[0, 0],
				 			 [0, yLen],
				 			 [xLen, yLen],
			         		 [xLen, 0]])

	tCI = cornersImage @ T
	xLimits = (np.min(tCI[:, 0]), np.max(tCI[:, 0]))
	yLimits = (np.min(tCI[:, 1]), np.max(tCI[:, 1]))
	
	affineMatrix = np.eye(3)
	affineMatrix[:2, :2] = T.T
	# shift image, such that it is bündig with the upper left corner
	affineMatrix[:2, 2] = np.array([-xLimits[0], -yLimits[0]])
	
	dst = cv2.warpAffine(img, affineMatrix[:2], (int(xLimits[1]-xLimits[0]), int(yLimits[1]-yLimits[0])))
	
	# Convert to grayscale
	gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
	
	# Detect corners using the Harris method
	dst2 = cv2.cornerHarris(gray, *parametersHarris)
	
	# Create a boolean bitmap of corner positions
	corners = dst2 > 0.05 * dst2.max()
	
	# Find the coordinates from the boolean bitmap
	coord = np.argwhere(corners)

	swappedCoords = coord.copy()
	swappedCoords[:, 0] = coord[:, 1]
	swappedCoords[:, 1] = coord[:, 0]
	
	swappedCoords = swappedCoords + np.array([[xLimits[0], yLimits[0]] for i in range(len(swappedCoords))])
	
	if returnLengthTransformedEdge:
		return swappedCoords, abs(xLen)

	return swappedCoords
	

def plotTransformedImage(imageName, T, cornerPointsTransformed, Y, drawLines=False):
	'''
		Plots the transformed version of an image
		
		IN: imageName 	 <-> the file name as '____.jpg'
			T			 <-> 2x2 numpy.ndarray; the linear transformation
			cornerPoints <-> numpy.ndarray where the i-th row contains [x, y] coordinates of the i-th corner
	'''
	img = mpimg.imread(imageName)

	yLen, xLen = img.shape[:2]
	cornersImage = np.array([[0, 0],
							 [0, yLen],
							 [xLen, yLen],
							 [xLen, 0]])

	tCI = cornersImage @ T
	xLimits = (np.min(tCI[:, 0]), np.max(tCI[:, 0]))
	yLimits = (np.min(tCI[:, 1]), np.max(tCI[:, 1]))

	fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
	imgTransformed = ax.imshow(img)

	affineMatrix = np.eye(3)
	affineMatrix[:2, :2] = T.T

	trans_data = mtransforms.Affine2D(matrix=affineMatrix) + ax.transData
	imgTransformed.set_transform(trans_data)
	
	if drawLines:
		for i in range(len(Y)):
			ax.plot([cornerPointsTransformed[i, 0], Y[i, 0]], [cornerPointsTransformed[i, 1], Y[i, 1]], color='white', alpha=0.5)
	
	ax.scatter(Y[:, 0], Y[:, 1], color='tab:blue', alpha = 0.5)
	if cornerPointsTransformed != []:
	    ax.scatter(cornerPointsTransformed[:, 0], cornerPointsTransformed[:, 1], marker='x', color='tab:orange', alpha = 0.5)

	# display intended extent of the image
	ax.set_xlim(xLimits[0], xLimits[1])
	ax.set_ylim(yLimits[1], yLimits[0])
	
    
    
