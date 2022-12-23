#!/usr/bin/python
import cv2 as cv
import numpy as np
import multiprocessing
import roi
import matplotlib.pyplot as plt

def findInlier(img):
	# Initiate SIFT detector

	img1 = img[0]		# Image in question
	img2 = img[1][0] 	# Gold standard image
	sift = cv.SIFT_create()

	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)

	flann = cv.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1, des2, k=2)

	# Store all good matches as per Lowe's ratio test.
	good = [m for m, n in matches if m.distance < 0.7*n.distance]

	MIN_MATCH_COUNT = 4
	if len(good) > MIN_MATCH_COUNT:
		src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()

	else:
		#print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
		matchesMask = []

	# Print Inliers
	return (len(matchesMask), img[1][1])


#Gets the nearest neighbor for each tile in
#a list of ROI tiles to the testData
def NN(tiles, images):
	"""param tiles: The regions of interest, in the form of images.
	param testData: Array of 30 or forty images.
	"""
	#store the max inliers for each tile
	maxInliers = []
	for tile in tiles:

		pairs = [(tile.copy(), img2) for img2 in images]
		if False:
			p = multiprocessing.Pool(processes = 4)
			inliers = p.map(findInlier, pairs)
		else:
			inliers = [findInlier(pair) for pair in pairs]

		theMax = max(inliers)

		#Show the tile to identify
		# cv.imshow("Tile to identify", tile)
		plt.figure(figsize=(1,1))
		plt.title('ROI')
		plt.imshow(tile)
		#cv.waitKey(1)

		#Show the best guess tile
		guess = cv.imread(theMax[1])
		guess = cv.resize(guess, (0,0), fx = 0.2, fy = 0.2)
		# cv.imshow("The guess", guess)
		plt.figure(figsize=(1,1))
		plt.title('guess')
		plt.imshow(guess)
		#cv.waitKey(0)

		maxInliers.append(theMax)
	return(maxInliers)


#Converts a list of image names into a list of
#tuple's where the first value of the tuple is 
#the OpenCV image and the second value of the 
#tuple is the file name
def getImage(files):
	return [(cv.resize(cv.imread(f,0),(0,0), fx = 0.2, fy = 0.2), f) for f in files]

def main():
	#Get images

	#get list of images in golden data set
	# Obtains the array of images to the golden standards
	testData = glob.glob(DATAPATH)
	images = getImage(testData)
	
	#REMOVE ONCE HENRY'S DONE
	# tiles contains an array of images
	tiles = roi.findRoi(FILE)
	inliers = NN(tiles, images)

# if __name__ == "__main__":
# 	main()


# Plot a list of image arrays in a grid
def plot_arrays(images, cols=4):
    rows = round(len(images) / cols)

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    outer_grid = fig.add_gridspec(rows, cols, wspace=0, hspace=0)

    axs = outer_grid.subplots()  # Create all subplots for the inner grid.
    for (a, b), ax in np.ndenumerate(axs):
        n = a*cols + b
        if n<len(images):
            ax.imshow(images[n])
        ax.set(xticks=[], yticks=[])
    plt.show()


# Plot a list of image arrays
def plotImages(images, titles=None, cols=1, figsize=None, cmap=None, norm=None, interpolation=None, fontsize=None):
    """Plot a list of images.
    Parameters
    ----------
    images: List of np.arrays compatible with plt.imshow.
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).
    cmap: Colormap for the images. Only required if images have a
          single color channel.
    norm: Normalize instance for the images. Only required if images
          have a single color channel.
    interpolation: Interpolation type for the images. Only required
                   if images have a single color channel.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    if figsize is None: figsize = (1,1)
    fig = plt.figure(figsize=figsize)
    if fontsize is None:
        fontsize = min(fig.get_size_inches()) * n_images
    plt.rcParams.update({'font.size': fontsize})
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(round(n_images/cols), cols,  n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image, cmap=cmap, norm=norm, interpolation=interpolation)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


