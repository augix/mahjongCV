#!/usr/bin/python
import cv2 as cv
import numpy as np
import multiprocessing
import roi
import matplotlib.pyplot as plt
import math
import glob

def remove_empty_tiles(tiles):
	"""Removes empty tiles from a list of tiles.
	"""
	new_tiles = []
	for tile in tiles:
		if len(tile) > 0:
			new_tiles.append(tile)
	return new_tiles

def findInlier(img):
	# Initiate SIFT detector

	img1 = img[0]		# Tile found as ROI, image in question (array)
	img2 = img[1][0] 	# Gold standard image (array)
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
		#Find the best match for each tile
		pairs = [(tile.copy(), img2) for img2 in images]
		if False:
			p = multiprocessing.Pool(processes = 4)
			inliers = p.map(findInlier, pairs)
		else:
			inliers = [findInlier(pair) for pair in pairs]
		theMax = max(inliers)
		maxInliers.append(theMax)
	return(maxInliers)

def temp():
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

#Converts a list of image names into a list of
#tuple's where the first value of the tuple is 
#the OpenCV image and the second value of the 
#tuple is the file name
def getImage(files):
	return [(cv.resize(cv.imread(f,0),(0,0), fx = 0.2, fy = 0.2), f) for f in files]



# Plot a list of image arrays in a grid
def plot_arrays(images, cols=4):
	rows = math.ceil(len(images) / cols)
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
def plotImages(images, titles=None, cols=1, figsize=None, cmap=None, norm=None, interpolation=None, fontsize=10):
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
	if figsize is None: figsize = (6,6)
	fig = plt.figure(figsize=figsize, constrained_layout=True)
	# fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
	if fontsize is None:
		fontsize = min(fig.get_size_inches()) * n_images
	# plt.rcParams.update({'font.size': fontsize})
	for n, (image, title) in enumerate(zip(images, titles)):
		ax = fig.add_subplot(math.ceil(n_images/cols), cols,  n + 1)
		if len(image) == 0:
			ax.axis('off')
		else:
			ax.imshow(image, cmap=cmap, norm=norm, interpolation=interpolation)
		ax.set_title(title, fontsize=fontsize)
		ax.set(xticks=[], yticks=[])
	# plt.show()

def plot_matches(tiles, inliers, ncol=4, figsize=(8,8), fontsize=10):
    assert len(inliers) == len(tiles)
    n = len(inliers)
    nrow = math.ceil(n/ncol)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    outer_grid = fig.add_gridspec(nrow, ncol, wspace=0, hspace=0)
    for a in range(nrow):
        for b in range(ncol):
            # gridspec inside gridspec
            inner_grid = outer_grid[a, b].subgridspec(1, 2, wspace=0, hspace=0)
            axs = inner_grid.subplots()  # Create all subplots for the inner grid.
            i = a*ncol + b
            if i < n:
                ax = axs[0]
                ax.imshow(tiles[i])
                ax.set(xticks=[], yticks=[])
                ax.set_title(f"Tile {i+1}", fontsize=fontsize)
                ax = axs[1]
                ax.imshow(cv.imread(inliers[i][1]))
                ax.set(xticks=[], yticks=[])
                ax.set_title(f"Score {inliers[i][0]}", fontsize=fontsize)
            else:
                ax = axs[0]
                ax.axis('off')
                ax = axs[1]
                ax.axis('off')
    # show only the outside spines
    for ax in fig.get_axes():
        ss = ax.get_subplotspec()
        ax.spines.top.set_visible(ss.is_first_row())
        ax.spines.bottom.set_visible(ss.is_last_row())
        ax.spines.left.set_visible(ss.is_first_col())
        ax.spines.right.set_visible(ss.is_last_col())
    plt.show()

def plot_image_paris_side_by_side(images1,images2,titles1=None,titles2=None,figsize=(8,8), fontsize=10):
	assert len(images1) == len(images2)
	n_images = len(images1)
	if titles1 is None: titles1 = ['Image a(%d)' % i for i in range(1,n_images + 1)]
	if titles2 is None: titles2 = ['Image b(%d)' % i for i in range(1,n_images + 1)]
	n = len(images1)
	fig = plt.figure(figsize=figsize, constrained_layout=False)
	outer_grid = fig.add_gridspec(1, n, wspace=0, hspace=0)
	for b in range(n):
		# gridspec inside gridspec
		inner_grid = outer_grid[0, b].subgridspec(1, 2, wspace=0, hspace=0)
		axs = inner_grid.subplots()  # Create all subplots for the inner grid.
		ax = axs[0]
		ax.imshow(images1[b])
		ax.set_title(titles1[b], fontsize=fontsize)
		ax.set(xticks=[], yticks=[])
		ax = axs[1]
		ax.imshow(images2[b])
		ax.set_title(titles2[b], fontsize=fontsize)
		ax.set(xticks=[], yticks=[])
	plt.show()

def plot_images_up_and_down(images1,images2,titles1=None,titles2=None,figsize=(8,8), fontsize=10):
	assert len(images1) == len(images2)
	n_images = len(images1)
	if titles1 is None: titles1 = ['Image a(%d)' % i for i in range(1,n_images + 1)]
	if titles2 is None: titles2 = ['Image b(%d)' % i for i in range(1,n_images + 1)]
	n = len(images1)
	fig = plt.figure(figsize=figsize, constrained_layout=False)
	outer_grid = fig.add_gridspec(1, n, wspace=0, hspace=0)
	for a in range(n):
		# gridspec inside gridspec
		inner_grid = outer_grid[0, a].subgridspec(2, 1, wspace=0, hspace=0)
		axs = inner_grid.subplots()  # Create all subplots for the inner grid.
		ax = axs[0]
		ax.imshow(images1[a])
		ax.set_title(titles1[a], fontsize=fontsize)
		ax.set(xticks=[], yticks=[])
		ax = axs[1]
		ax.imshow(images2[a])
		ax.set_title(titles2[a], fontsize=fontsize)
		ax.set(xticks=[], yticks=[])
	plt.show()

def plotMatches(tiles, inliers,figsize=(8,8)):
	assert len(inliers) == len(tiles)
	images1 = tiles
	images2 = [ cv.imread(i[1]) for i in inliers]
	titles1 = [f"Tile {i+1}" for i in range(len(tiles))]
	titles2 = [f"Score {i[0]}" for i in inliers]
	plot_images_up_and_down(images1,images2,titles1=titles1,titles2=titles2,figsize=figsize, fontsize=10)

def main(DATAPATH, FILE):
	#Get images
	plt.imshow(plt.imread(FILE))
	# tiles contains an array of images
	tiles = roi.findRoi(FILE)
	tiles = remove_empty_tiles(tiles)
	plotImages(tiles,cols=10,figsize=(10,3),fontsize=10)

	#get list of images in golden standard data set
	# Obtains the array of images to the golden standards
	testData = glob.glob(DATAPATH)
	images = getImage(testData)
	inliers = NN(tiles, images)
	#plot_matches(tiles, inliers, ncol=5, figsize=(8,8))
	plotMatches(tiles, inliers, figsize=(20,5))

	res= [ (i+1, inliers[i]) for i in range(len(inliers)) ] 
	print(res)
