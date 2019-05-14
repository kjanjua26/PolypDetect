import numpy as np
import cv2

def cropBlack(img,tol=0):
    # img is image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def clustering(img):
	from sklearn.cluster import KMeans

	img_flat = img.reshape(-1,1)
	kmeans = KMeans(n_clusters=5, random_state=0).fit(img_flat)

	out = kmeans.cluster_centers_[kmeans.labels_]
	return out.reshape(img.shape[0],img.shape[1])


def region_growing(img, seed):
    #Parameters for region growing
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    region_threshold = 0.2
    region_size = 1
    intensity_difference = 0
    neighbor_points_list = []
    neighbor_intensity_list = []

    #Mean of the segmented region
    region_mean = img[seed]

    #Input image parameters
    height, width = img.shape
    image_size = height * width

    #Initialize segmented output image
    segmented_img = np.zeros((height, width, 1), np.uint8)

    #Region growing until intensity difference becomes greater than certain threshold
    while (intensity_difference < region_threshold) & (region_size < image_size):
        #Loop through neighbor pixels
        for i in range(4):
            #Compute the neighbor pixel position
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]

            #Boundary Condition - check if the coordinates are inside the image
            check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)

            #Add neighbor if inside and not already in segmented_img
            if check_inside:
                if segmented_img[x_new, y_new] == 0:
                    neighbor_points_list.append([x_new, y_new])
                    neighbor_intensity_list.append(img[x_new, y_new])
                    segmented_img[x_new, y_new] = 255

        #Add pixel with intensity nearest to the mean to the region
        distance = abs(neighbor_intensity_list-region_mean)
        pixel_distance = min(distance)
        index = np.where(distance == pixel_distance)[0][0]
        segmented_img[seed[0], seed[1]] = 255
        region_size += 1

        #New region mean
        region_mean = (region_mean*region_size + neighbor_intensity_list[index])/(region_size+1)

        #Update the seed value
        seed = neighbor_points_list[index]
        #Remove the value from the neighborhood lists
        neighbor_intensity_list[index] = neighbor_intensity_list[-1]
        neighbor_points_list[index] = neighbor_points_list[-1]

    return segmented_img


def adaptiveThresholding(img):
	img = cv2.medianBlur(img,3)
	mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2.15)

	edge_detected_image = cv2.Canny(mask, 100, 200)

	_, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	contour_list = []
	for contour in contours:
	    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
	    area = cv2.contourArea(contour)
	    if ((len(approx) > 8) & (len(approx) < 23) & (area > 75) ):
	        contour_list.append(contour)

	cv2.drawContours(mask, contour_list,  -1, (0,0,0), 12)

	mask = cv2.medianBlur(mask,7)
	#mask = region_growing(mask, (mask.shape[0]//2, mask.shape[1]//2))



	return mask