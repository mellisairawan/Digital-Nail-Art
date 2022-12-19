# Import Library
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(img_path):
  img = cv2.imread(img_path)

  return img

def import_nail_image(img_path):
  img = cv2.imread(img_path)

  # Resize the image into square
  target_size = np.min(img[:,:,0].shape)
  img = cv2.resize(img, (target_size, target_size))

  return img


def import_color_image(img_path):
  img = cv2.imread(img_path)
  # Resize the image into square
  target_size = np.min(img[:,:,0].shape)
  img = cv2.resize(img, (target_size, target_size))

  return img

def resize_to_img1(img1, img2):
  # Resize the image into square based on img1 size
  target_size = np.min(img1[:,:,0].shape)
  img1 = cv2.resize(img1, (target_size, target_size))
  img2 = cv2.resize(img2, (target_size, target_size))

  return img1, img2

def disp_img(img):
  fig = plt.figure(figsize=(12,10))
  ax = fig.add_subplot(111)
  plt.axis('off')
  plt.imshow(img, cmap='gray')


def bwareaopen(img, min_size, connectivity=8):
	"""Remove small objects from binary image (approximation of 
	bwareaopen in Matlab for 2D images).

	Args:
	    img: a binary image (dtype=uint8) to remove small objects from
	    min_size: minimum size (in pixels) for an object to remain in the image
	    connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).

	Returns:
	    the binary image with small objects removed
	"""

	# Find all connected components (called here "labels")
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
	    img, connectivity=connectivity)

	# check size of all connected components (area in pixels)
	for i in range(num_labels):
	    label_size = stats[i, cv2.CC_STAT_AREA]
	    
	    # remove connected components smaller than min_size
	    if label_size < min_size:
	        img[labels == i] = 0
	        
	return img


def create_nail_mask(img):

  # Turn the image into gray
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Image smoothing to eliminate background edges
  blur = cv2.blur(gray,ksize=(1,1))

  # Detect image using Laplacian operator
  gradient = cv2.Laplacian(blur, cv2.CV_64F)
  gradient = cv2.convertScaleAbs(gradient)

  # Detect edge using canny operator
  med = np.median(gray)
  lower = int(max(0,0.7*med))
  upper = int(min(255,1.3*med))
  edges = cv2.Canny(blur,lower,upper)
  edges = cv2.GaussianBlur(edges, (11, 11), 0)  # smoothing before applying  threshold

  # Find contours in binary image
  contours, hierarchy = cv2.findContours(edges, 
                                        cv2.RETR_CCOMP, 
                                        cv2.CHAIN_APPROX_SIMPLE)

  # Create empty array to hold internal contours
  image_internal = np.zeros(img.shape)

  # Iterate through list of contour arrays
  for i in range(len(contours)):
      # If third column value is NOT equal to -1 than its internal
      if hierarchy[0][i][3] != -1:
          
          # Draw the Contour
          cv2.drawContours(image_internal, contours, i, 255, -1)

  mask = image_internal[:,:,0].astype(np.uint8) #first layer only as the second and third is all black pixel


  # Resize the image into square
  target_size = np.min(mask.shape)
  mask = cv2.resize(mask, (target_size, target_size))

  # Delete small white pixels
  mask = bwareaopen(mask, min_size=115, connectivity=4)

  # Slightly increase the area of detected area
  kernel = np.ones((10, 10), 'uint8')
  mask = cv2.dilate(mask, kernel, iterations=1)

  return mask


def create_nail_color(img, color_img, mask):
  # Resize the overlay image to match the bg image dimensions
  target_size = np.min(img[:,:,0].shape)
  overlay_img = cv2.resize(color_img, (target_size, target_size))

  masked = cv2.bitwise_and(overlay_img, overlay_img, mask = mask)

  alpha = 0.7
  # Create a copy of the image to work with
  result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Create a new np array
  shapes = np.zeros_like(result, np.uint8)
  
  # Put the overlay at the bottom-right corner
  shapes = masked
  
  # Change this into bool to use it as mask
  mask = shapes.astype(bool)

  # Create the overlay
  result[mask] = cv2.addWeighted(result, 1 - alpha, masked, alpha, 0)[mask]

  return result