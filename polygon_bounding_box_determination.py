# Author: Henning Otto Brinkhaus
# Friedrich-Schiller Universität Jena

import sys
import os
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageStat
import random
import json
from copy import deepcopy
from multiprocessing import Pool
from skimage import morphology


def define_bounding_box_center(bounding_box: np.array):
	"""
	This function returns the center np.array([x,y]) of a given bounding box polygon 
	np.array([[x1, y1], [x2, y2],...,[x_n, y_n]])
	
	Args:
		bounding_box (np.array): [description]

	Returns:
		[np.array]: [description]
	"""	
	x_values = np.array([])
	y_values = np.array([])
	for node in bounding_box:
		x_values = np.append(x_values, node[0])
		y_values = np.append(y_values, node[1])
	return np.array([np.mean(x_values), np.mean(y_values)])


def define_local_center(bounding_box: np.array, node_index: int, n: int=5):
	"""
	This function return the center np.array([x,y]) of the n previous and the n following nodes 
	of bounding box np.array([[x1, y1], [x2, y2],...,[x_n, y_n]])

	Args:
		bounding_box (np.array): [description]
		node_index (int): [description]
		n (int, optional): Amount of nodes in both directions to define local center. Defaults to 5.

	Returns:
		np.array containing x and y value of local center
	"""	
	x_values = np.array([])
	y_values = np.array([])
	#for node in np.roll(bounding_box, -node_index+n)[:2*n]:
	for node in np.roll(bounding_box, -node_index+n)[:2*n]:
		x_values = np.append(x_values, bounding_box[node_index][0])
		y_values = np.append(y_values, bounding_box[node_index][1])
	return np.array([np.mean(x_values), np.mean(y_values)])


def define_edge_line(node_1: Tuple[int, int], node_2: Tuple[int, int]) -> Tuple[float, float]:
	"""
	This function returns a linear function between two given points in a 2-dimensional space.

	Args:
		node_1 (Tuple[int, int])
		node_2 (Tuple[int, int])

	Returns:
		Tuple[float, float]: slope, intercept
	"""
	if node_1[0]-node_2[0] != 0:
		slope = (node_2[1]-node_1[1])/(node_2[0]-node_1[0])
	else:
		slope = (node_2[1]-node_1[1])/(node_2[0]-node_1[0]+0.0000000001) # avoid dividing by zero, this line should not be necessary anymore.
	#intercept = y - m*x
	intercept = node_1[1] - slope * node_1[0]
	return slope, intercept


def euklidian_distance(node_1: Tuple[int, int], node_2: Tuple[int, int]) -> float:
	"""
	This function returns the euklidian distance between two given points in a 2-dimensional space.

	Args:
		node_1 (Tuple[int, int])
		node_2 (Tuple[int, int]) 

	Returns:
		float: distance
	"""
	return np.sqrt((node_2[0]-node_1[0]) ** 2 + (node_2[1]-node_1[1])**2)


def define_stepsize(slope: float, shape: Tuple[int, int, int], step_factor: float) -> float:
	"""
	This function takes the slope of the line along which the node is pushed out of the bounding box center
	and the shape of the image. Depending on the resolution and the slope, the step size in x-direction is defined.
    The step_factor can be varied to alter the step_size (The bigger it is, the smaller are the steps.)

	Args:
		slope (float)
		shape (Tuple[int, int, int]): shape of the image
		step_factor (float)

	Returns:
		step_size (float): stepsize
	"""
	diagonal_slope=shape[0]/shape[1]
	#diagonal_slope = 1
	if slope < diagonal_slope and slope > -diagonal_slope:
		return shape[1]/step_factor
	else:
		return (shape[1]/step_factor)/slope


def set_x_diff_range(x_diff, euklidian_distance, image_array):
	"""
	This function takes the amount of steps on an edge and returns the corresponding list of steps in random order

	Args:
		x_diff (int)
		euklidian_distance (float)
		image_array (np.array)

	Returns:
		x_diff_range (np.array):  
	"""
	#blur_factor = int(image_array.shape[1]/400) if image_array.shape[1]/400 >= 1 else 1
	blur_factor = 1
	if x_diff > 0:
		x_diff_range = np.arange(0, x_diff, blur_factor/euklidian_distance) 
	else:
		x_diff_range = np.arange(0, x_diff, -blur_factor/euklidian_distance)
	np.random.shuffle(x_diff_range)
	return x_diff_range


def define_next_pixel_to_check(bounding_box, node_index, step, image_shape):
	"""
	This function returns the next pixel to check in the image (along the edge of the bounding box). 
	In the case that it tries to check a pixel outside of the image, this is corrected.

	Args:
		bounding_box ([type]): [description]
		node_index (int): [description]
		step ([type]): [description]
		image_shape ([type]): [description]

	Returns:
		(tuple[int, int]): (x,y)
	"""
	# Define the edges of the bounding box
	slope, intercept = define_edge_line(bounding_box[node_index], bounding_box[node_index-1])
	x = int(bounding_box[node_index-1][0] + step)
	y = int((slope * (bounding_box[node_index-1][0] + step)) + intercept)
	if y >= image_shape[1]:
		y = image_shape[1]-1
	if y < 0:
		y = 0
	if x >= image_shape[0]:
		x = image_shape[0]-1
	if x < 0:
		x = 0
	return x,y


def adapt_x_values(bounding_box, node_index, image_shape):
	'''If two nodes form a vertical edge the function that descripes that edge will in- or decrease infinitely with dx so we need to alter those nodes. 
	This function returns a bounding box where the nodes are altered depending on their relative position to the center of the bounding box.
	If a node is at the border of the image, then it is not changed.'''
	if bounding_box[node_index][0] != image_shape[1]:
		bounding_box_center = define_bounding_box_center(bounding_box)
		if bounding_box[node_index][0] < bounding_box_center[0]:
			bounding_box[node_index][0] = bounding_box[node_index][0] - 1
		else: 
			bounding_box[node_index][0] = bounding_box[node_index][0] + 1
	return bounding_box


def factor_fold_nodes(bounding_box, factor):	
	'''A bounding box which is defined by n points is turned into a bounding box where each edge between two nodes has $factor more equidistant nodes.'''	
	#new_bounding_box = np.zeros((bounding_box.shape[0] * factor, bounding_box.shape[1]))
	new_bounding_box = np.zeros((len(bounding_box) * factor, 2))	
	for node_index in range(len(bounding_box)):	
        #These if/else blocks avoid steps of zero in the arange.
		if bounding_box[node_index][0]-bounding_box[node_index-1][0]:
			x_range = np.arange(bounding_box[node_index-1][0], bounding_box[node_index][0], (bounding_box[node_index][0]-bounding_box[node_index-1][0])/factor)	
		else:
			x_range = np.full((1, factor), bounding_box[node_index][0]).flatten()
		if (bounding_box[node_index][1]-bounding_box[node_index-1][1]) != 0:
			y_range = np.arange(bounding_box[node_index-1][1], bounding_box[node_index][1], (bounding_box[node_index][1]-bounding_box[node_index-1][1])/factor)	
		else:
			y_range = np.full((1, factor), bounding_box[node_index][1]).flatten()
		for index in range(len(x_range)):	
			new_bounding_box[node_index*factor+index] = [x_range[index], y_range[index]]	
	return new_bounding_box


def expand_bounding_box(bounding_box, nodes_to_be_changed, step_factor, shape, iterations, local_center_ratio = False):
	'''This function takes a list of nodes to be changed and modifies them by moving them away from the 
	bounding box center or a local center along a line between the node and the center point.'''

	# Every n_th time: use local center instead of global center, otherwise use global bounding box center
	if not local_center_ratio or iterations % local_center_ratio != 0:
		center = define_bounding_box_center(bounding_box) # Define bounding box center

	for node_index in nodes_to_be_changed:
		if local_center_ratio and iterations % local_center_ratio == 0:
			center = define_local_center(bounding_box, node_index, n = 10) # Define local center
		else:
			center = define_bounding_box_center(bounding_box)
            
		#Define the axis along which we want to move the node
		slope, intercept = define_edge_line(center, bounding_box[node_index])
		step_size = define_stepsize(slope, shape, step_factor)
		changed_node_1 = [bounding_box[node_index][0]+step_size, slope * (bounding_box[node_index][0]+step_size) + intercept]
		changed_node_2 = [bounding_box[node_index][0]-step_size, slope * (bounding_box[node_index][0]-step_size) + intercept]
		if euklidian_distance(changed_node_1, center) >= euklidian_distance(changed_node_2, center):
			bounding_box[node_index] = changed_node_1
		else:
			bounding_box[node_index] = changed_node_2
	return bounding_box


def bb_expansion_coordination(image_array: np.array, original_bounding_box: np.array, debug: bool = False):
	'''This function applies the bounding box expansion method to a given bounding box with a given image and returns the expanded bounding box.'''
	parameter_combinations = [(200, 160, False), (100, 80, False), (50, 40, False) ,(25, 20, False), (25, 20, 5)]
	for parameter_combination in parameter_combinations:
		bounding_box = original_bounding_box.copy()
		step_factor, step_limit, local_center_ratio = parameter_combination
		nothing_on_the_edges = False
		iterations = 0	#count iteration steps
		# Check edges for pixels that are not white and expand the bounding box until there is nothing on the edges.
		while nothing_on_the_edges == False:
			nodes_to_be_changed = []
			for node_index in range(len(bounding_box)):
				# Define amount of steps we have to go in x-direction to get from node 1 to node 2
				x_diff = bounding_box[node_index][0] - bounding_box[node_index-1][0]
				if x_diff == 0:
					bounding_box = adapt_x_values(bounding_box = bounding_box, node_index = node_index, image_shape = image_array.shape)
				x_diff = bounding_box[node_index][0] - bounding_box[node_index-1][0] # Define amount of steps we have to go in x-direction to get from node 1 to node 2
				dist = euklidian_distance(bounding_box[node_index], bounding_box[node_index-1])
				#if dist > 0:
				x_diff_range = set_x_diff_range(x_diff,dist, image_array)
				#Go down the edge and check if there is something that is not white. If something was found, the corresponding nodes are saved.
				for step in x_diff_range:
					x,y = define_next_pixel_to_check(bounding_box, node_index, step, image_shape = image_array.shape)
					# If there is something that is not white	
					if image_array[x, y] < 255: 
						nodes_to_be_changed.append(node_index)
						nodes_to_be_changed.append(node_index-1)
						break
			if iterations >= step_limit:
				break
			if nodes_to_be_changed == []:
				#print('Bounding box expansion complete. \n Step factor: ' + str(step_factor) + '\n Local center ratio: ' + str(local_center_ratio))
				nothing_on_the_edges = True
				# If nothing was detected after 0 iterations, the initial rectangle was probably a failure.
				if iterations == 0:
					#return bounding_box
					return False
			else: 
				iterations += 1
				if debug:
					print("The bounding box is modified. Iteration No. " + str(iterations))
					im = Image.fromarray(image_array)
					im = im.convert('RGB')
					# ImageDraw does not like np.arrays
					polygon = [(node[1], node[0]) for node in bounding_box]
					draw = ImageDraw.Draw(im, 'RGBA')
					draw.polygon(polygon, fill = (0,255,0,50))
					im.show()
				nodes_to_be_changed = set(nodes_to_be_changed)
				bounding_box = expand_bounding_box(bounding_box, nodes_to_be_changed, step_factor, shape = image_array.shape, iterations = iterations, local_center_ratio = local_center_ratio)
		if iterations < step_limit:
			return bounding_box
	print("Bounding box expansion was not successful. Return original bounding box.")
	return False


def initial_rectangle(image_array: np.array) -> List[Tuple[int, int]]:
	'''This function defines the initial rectangle for the polygon region definition.
	The output of this function is a small rectangle which does not cover the whole object.'''
	non_white_y, non_white_x = np.where(image_array < 254)
	try:
		min_y = min(non_white_y)
		max_y = max(non_white_y)
		min_x = min(non_white_x)
		max_x = max(non_white_x)
	except ValueError:
		return False
	y_diff = max_y - min_y
	x_diff = max_x - min_x
	mid_x = (min_x+max_x)/2
	mid_y = (min_y+max_y)/2
	initial_rectangle = [(mid_y, min_x + 1/5 * x_diff), (max_y - 1/5 * y_diff, mid_x), (mid_y, max_x - 1/5 * x_diff), (min_y + 1/5 * y_diff, mid_x)]
	#initial_rectangle = [(tup[1], tup[0]) for tup in initial_rectangle]
	return initial_rectangle

def polygon_coordinates(image_array: np.array, debug: bool = False) -> Tuple[List[int], List[int]]:
	'''Given an image path, this function loads the image and defines a polygon around the
	object in the image. This function assumes, that there only is one object in the (blurred)
	image. 
	The idea is to define a small rectangle in the center of the object, that does not enclose it.
	Then, the amount of nodes that defines the rectangle is multiplied and the 'rectangle' is 
	turned into a polygon that surrounds the object by pushing the nodes that define the polygon
	out of its center. The process stops when no non-white pixels are detected anymore on the
	polygon contours. Originally, we came up with this procedure for the mask expansion in 
	DECIMER Segmentation but there, it was replaced for a different procedure (which could not
	be applied here). But I am happy to be able to recycle my old code here.
	Returns (List[x_coordinates], List[y_coordinates])'''

	# Define initial small rectangle and tenfold amount of nodes that define it.
	polygon = initial_rectangle(image_array)
	if not polygon: 
		return False
	polygon = factor_fold_nodes(polygon, 6)

	# Expand the n-gon
	polygon = bb_expansion_coordination(image_array, polygon, debug = False)
	
	if type(polygon) == np.ndarray:
		if debug:
			im = Image.fromarray(image_array)
			im = im.convert('RGB')
			polygon = [(node[1], node[0]) for node in polygon]
			draw = ImageDraw.Draw(im, 'RGBA')
			draw.polygon(polygon, fill = (0,255,0,50))
			im.show()
	return polygon