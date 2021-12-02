# Author: Henning Otto Brinkhaus
# Friedrich-Schiller Universität Jena

# Some recycled code from an old version of the mask expansion mechanism 
# of DECIMER Segmentation found its way in here for the polygon annotation creation :)

import sys
import os
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageStat
import random
import json
from copy import deepcopy
from multiprocessing import Pool
from RanDepict import random_depictor


# To avoid confusion: 'bounding_box' usually refers to a polygon here and not necessarily a rectangle


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
				print('Bounding box expansion complete. \n Step factor: ' + str(step_factor) + '\n Local center ratio: ' + str(local_center_ratio))
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

def pad_image(pil_image: Image, factor: float):
	'''This function takes a Pillow Image and adds 10% padding on every side. It returns the padded Pillow Image'''
	original_size = pil_image.size

	new_size = (int(original_size[0]*factor), int(original_size[1]*factor))
	new_im = Image.new("L", new_size, color = 'white')
	new_im.paste(pil_image, (int((new_size[0]-original_size[0])/2),
                      int((new_size[1]-original_size[1])/2)))
	return new_im

def ID_label_text()->str:
	'''This function returns a string that resembles a typical chemica ID label'''
	label_num = range(1,50)
	label_letters = ["a", "b", "c", "d", "e", "f", "g", "i", "j", "k", "l", "m", "n", "o"]
	options = ["only_number", "num_letter_combination", "numtonum", "numcombtonumcomb"]
	option = random.choice(options)
	if option == "only_number":
		return str(random.choice(label_num))
	if option == "num_letter_combination":
		return str(random.choice(label_num)) + random.choice(label_letters)
	if option == "numtonum":
		return str(random.choice(label_num)) + "-" + str(random.choice(label_num))
	if option == "numcombtonumcomb":
		return str(random.choice(label_num)) + random.choice(label_letters) + "-" + random.choice(label_letters)

def get_random_label_position(width: int, height: int, x_coordinates: List[int], y_coordinates: List[int]) -> Tuple:
	'''Given the coordinates of the polygon around the chemical structure and the image shape, this function determines
	a random position for the label around the chemical struture.
	Returns: int x_coordinate, int y_coordinate or None, None if it cannot place the label without an overlap.'''
	x_y_limitation = random.choice([True, False])
	if x_y_limitation:
		y_range = list(range(50, int(min(y_coordinates)-60))) + list(range(int(max(y_coordinates)+20), height - 50))
		x_range = range(50, width - 50)
		if len(y_range) == 0 or len(x_range) == 0:
			x_y_limitation = False
	if not x_y_limitation:
		y_range = list(range(50, height - 50))
		x_range = list(range(50, int(min(x_coordinates)-60))) + list(range(int(max(x_coordinates)+20), width - 50))
		if len(y_range) == 0 or len(x_range) == 0:
			x_y_limitation = True
	if x_y_limitation:
		y_range = list(range(50, int(min(y_coordinates)-60))) + list(range(int(max(y_coordinates)+20), height - 50))
		x_range = range(50, width - 50)
	if len(y_range) > 0 and len(x_range) > 0:
		y_pos = random.choice(y_range)
		x_pos = random.choice(x_range)
		return x_pos, y_pos
	else: 
		return None, None



def add_chemical_ID(image: Image, chemical_structure_polygon: np.ndarray, debug = False)->Tuple:
	'''This function takes a PIL Image and the coordinates of the region that contains a chemical structure diagram
	and adds random text that looks like a chemical ID label around the structure.
	It returns the modified image and the coordinates of the bounding box of the added text.
	If it is not possible to place the label around the structure without an overlap, it returns None'''	
	im = image.convert('RGB')
	y_coordinates = [node[0] for node in chemical_structure_polygon]
	x_coordinates = [node[1] for node in chemical_structure_polygon]

	# Choose random font
	font_dir = os.path.abspath("./fonts/")
	fonts = os.listdir(font_dir)
	font_sizes = range(12, 24)

	# Define random position for text element around the chemical structure (if it can be placed around it)
	width, height = im.size
	x_pos, y_pos = get_random_label_position(width, height, x_coordinates, y_coordinates)
	if x_pos == None:
		return im, None

	# Choose random font size
	size = random.choice(font_sizes)
	label_text = ID_label_text()
	
	try:
		font = ImageFont.truetype(str(os.path.join(font_dir, random.choice(fonts))), size = size)
	except OSError:
		font = ImageFont.load_default()


	
	draw = ImageDraw.Draw(im, 'RGBA')
	bb = draw.textbbox((x_pos, y_pos), label_text, font = font) # left, up, right, low
	# upper left, lower left, lower right, upper right
	label_bounding_box = [[bb[0]-1, bb[1]-1], [bb[0]-1, bb[3]+1], [bb[2]+1, bb[3]+1], [bb[2]+1, bb[1]-1]]
	# Add text element to image
	draw = ImageDraw.Draw(im, 'RGBA')
	draw.text((x_pos,y_pos), label_text, font = font, fill=(0,0,0,255))
	
	# For illustrative reasons: Show the bounding boxes around structure and label
	if debug:
		polygon = [(node[1], node[0]) for node in chemical_structure_polygon]
		mod_label_bounding_box = [(node[0], node[1]) for node in label_bounding_box]
		draw.polygon(mod_label_bounding_box, fill = (255,0,0,50))
		draw.polygon(polygon, fill = (0,255,0,50))
		im.show()
	return im, label_bounding_box


def make_region_dict(type: str, polygon: np.ndarray) -> Dict:
	'''In order to avoid redundant code in make_img_metadata_dict, this function creates the dict that holds the information 
	for one single region. It expects a type (annotation string for the region) and an array containing tuples [(y0, x0), (y1, x1)...]'''
	region_dict = {}
	region_dict['region_attributes'] = {}
	region_dict['region_attributes']['type'] = type
	region_dict['shape_attributes'] = {}
	region_dict['shape_attributes']['name'] = 'polygon'

	y_coordinates = [int(node[0]) for node in polygon]
	x_coordinates = [int(node[1]) for node in polygon]
	region_dict['shape_attributes']['all_points_x'] = list(x_coordinates)
	region_dict['shape_attributes']['all_points_y'] = list(y_coordinates)
	return region_dict


def make_img_metadata_dict(
	image_name: str, 
	#output_dir:str, 
	chemical_structure_polygon: np.ndarray, 
	label_bounding_box = False) -> Dict:
	'''This function takes the name of an image, the coordinates of the chemical structure polygon and (if it exists) the 
	bounding box of the ID label and creates the _via_img_metadata subdictionary for the VIA input.'''
	metadata_dict = {}
	metadata_dict['filename'] = image_name

	#metadata_dict['size'] = int(os.stat(os.path.join(output_dir, image_name)).st_size)
	metadata_dict['regions'] = []
	# Add dict for first region which contains chemical structure
	metadata_dict['regions'].append(make_region_dict('chemical_structure', chemical_structure_polygon))
	# If there is an ID label: Add dict for the region that contains the label
	if label_bounding_box:
		label_bounding_box = [(node[1], node[0]) for node in label_bounding_box]
		metadata_dict['regions'].append(make_region_dict('chemical_ID', label_bounding_box))
	return metadata_dict


def make_VIA_dict(metadata_dicts: List[Dict]) -> Dict:
	'''This function takes a list of Dicts with the region information and returns a dict that can be opened
	using VIA when it is saved as a json file.'''
	VIA_dict = {}
	VIA_dict["_via_img_metadata"] = {}
	for region_dict in metadata_dicts:
		VIA_dict["_via_img_metadata"][region_dict["filename"]] = region_dict
	return VIA_dict["_via_img_metadata"]



def pick_random_colour(black = False) -> Tuple[int]:
	'''This funcion returns a random tuple with a colour for a four channel image.'''
	if not black:
		for _ in range(20):
			new_colour = (random.choice(range(255)), random.choice(range(255)), random.choice(range(255)), 255)
			if sum(new_colour[:3]) < 550:
				break
		else:
			new_colour = (0,0,0,255)
	else:
		for _ in range(20):
			num = random.choice(range(30))
			new_colour = (num, num, num, 255)
			if sum(new_colour[:3]) < 550:
				break
		else:
			new_colour = (0,0,0,255)
	return new_colour


def modify_colours(image: Image, blacken = False) -> Image:
	'''This function takes a Pillow Image, makes white pixels transparent, gives every other pixel a given new colour and returns the Image.
	If blacken is True, non-white pixels are going to be replaced with black-ish pixels instead of coloured pixels.'''
	image = image.convert('RGBA')
	datas = image.getdata()
	newData = []
	if blacken:
		new_colour = pick_random_colour(black = True)
	else:
		new_colour = pick_random_colour()
	for item in datas:
		if not blacken and item[0] > 230 and item[1] > 230 and item[2] > 230:
			newData.append((item[0], item[1], item[2], 0))
		elif item[0] < 230 and item[1] < 230 and item[2] < 230:
			newData.append(new_colour)
		else:
			newData.append(item)
	image.putdata(newData)
	return image


def add_arrows_to_structure(image: Image, arrow_dir: str, polygon: np.ndarray) -> Image:
	'''This function takes an image of a chemical structure and adds between 4 and 20 curved arrows in random positions in the chemical structure.
	It needs the polygon coordinates around the chemical structure to make sure that the arrows are in the structure.'''
	
	orig_image = deepcopy(image)
	# Determine area where arrows are pasted.
	x_coordinates = [int(node[1]) for node in polygon]
	y_coordinates = [int(node[0]) for node in polygon]
	x_min = min(x_coordinates)
	y_min = min(y_coordinates)
	x_max = max(x_coordinates)
	y_max = max(y_coordinates)
	# Open random amount of arrow images, resize and rotate them randomly and paste them randomly in the chemical structure depiction
	if random.choice(range(2)) == 0:
		random_colour = pick_random_colour()
	else:
		random_colour = (0,0,0,255)
	for _ in range(random.choice(range(4, 15))):
		arrow_image = Image.open(os.path.join(arrow_dir, random.choice(os.listdir(arrow_dir))))
		arrow_image = modify_colours(arrow_image)
		new_arrow_image_shape = int((x_max - x_min) / random.choice(range(3,6))), int((y_max - y_min) / random.choice(range(3,6)))
		arrow_image = arrow_image.resize(new_arrow_image_shape, resample=Image.BICUBIC)
		arrow_image = arrow_image.rotate(random.choice(range(360)), resample=Image.BICUBIC, expand=True)

		# Try different positions with the condition that the arrows are overlapping with non-white pixels (the structure)
		for _ in range(50):
			x_position = random.choice(range(x_min, x_max - new_arrow_image_shape[0]))
			y_position = random.choice(range(y_min, y_max - new_arrow_image_shape[1]))
			paste_region = orig_image.crop((x_position, y_position, x_position + new_arrow_image_shape[0], y_position + new_arrow_image_shape[1]))
			mean = ImageStat.Stat(paste_region).mean
			if sum(mean)/len(mean) < 250:
				image.paste(arrow_image, (x_position, y_position), arrow_image)
				break
	return image

def adapt_coordinates(polygon: np.ndarray, label_bounding_box: np.ndarray, crop_indices: Tuple[int,int,int,int]) -> Tuple[np.ndarray, np.ndarray]:
	'''This function adapts the annotation coordinates after white space on the edges has been deleted.
	It also corrects coordinates outside of the image'''
	first_relevant_row, last_relevant_row, first_relevant_col, last_relevant_col = crop_indices
	# Adapt chemical structure polygon coordinates
	for node in polygon: 
		node[0] = node[0] - first_relevant_row
		if node[0] < 0:
			node[0] = 0
		if node[0] > last_relevant_row - first_relevant_row:
			node[0] = last_relevant_row - first_relevant_row
		node[1] = node[1] - first_relevant_col
		if node[1] < 0:
			node[1] = 0
		if node[1] > last_relevant_col - first_relevant_col:
			node[1] = last_relevant_col - first_relevant_col
	# Do the same for the chemical ID label
	if label_bounding_box:
		for node in label_bounding_box:
			node[0] = node[0] - first_relevant_col
			if node[0] < 0:
				node[0] = 0
			if node[0] > last_relevant_col - first_relevant_col:
				node[0] = last_relevant_col - first_relevant_col
			node[1] = node[1] - first_relevant_row
			if node[1] < 0:
				node[1] = 0
			if node[1] > last_relevant_row - first_relevant_row:
				node[1] = last_relevant_row - first_relevant_row
	return polygon, label_bounding_box


def delete_whitespace(image: Image, polygon: np.ndarray, label_bounding_box: np.ndarray) -> Tuple:
	'''This function takes an image, the polygon coordinates around the chemical structure and the label_bounding_box and
	makes the image smaller if there is unnecessary whitespace at the edges. The polygon/bounding box coordinates are changed
	accordingly.'''	
	image = image.convert('RGB')
	image = np.asarray(image)	
	y, x, z = image.shape
	first_relevant_row = 0
	last_relevant_row = y-1
	first_relevant_col = 0
	last_relevant_col = x-1
	# Check for non-white rows
	for n in np.arange(0, y, 1):
		if image[n].sum() != x * z * 255:
			first_relevant_row = n -1
			break
	for n in np.arange(y-1, 0, -1):
		if image[n].sum() != x * z * 255:
			last_relevant_row = n +1
			break
	transposed_image = np.swapaxes(image, 0, 1)

	# Check for non-white columns
	for n in np.arange(0, x, 1):
		if transposed_image[n].sum() != y * z * 255:
			first_relevant_col = n -1
			break
	# Check for non-white columns
	for n in np.arange(x-1, 0, -1):
		if transposed_image[n].sum() != y * z * 255:
			last_relevant_col = n + 1
			break

	image = image[first_relevant_row:last_relevant_row, first_relevant_col:last_relevant_col, :]
	crop_indices = (first_relevant_row, last_relevant_row, first_relevant_col, last_relevant_col)
	polygon, label_bounding_box = adapt_coordinates(polygon, label_bounding_box, crop_indices)
	return Image.fromarray(image), polygon, label_bounding_box


def generate_structure_and_annotation(smiles: str, shape = (200, 200), label: bool = False, arrows : bool = False):
	"""
	Generate a chemical structure depiction and the metadata dict with the polygon bounding box.

	Args:
		smiles (str): SMILES representation of chemical compound
		label (bool, optional): Set to True if additional labels around the structure are desired. Defaults to False.
		arrows (bool, optional): Set to True if curved arrows in the structure are desired. Defaults to False.

	Returns:
		output_image (PIL.Image)
		metadata_dict (Dict): dictionary containing the coordinates that are necessary to define the polygon bounding box
							  around the chemical structure depiction.
	"""
	# Depict chemical structure
	depictor = random_depictor(seed = random.choice(range(10000000)))
	image = depictor.random_depiction(smiles, shape)
	image = Image.fromarray(image)
	# Get coordinates of polygon around chemical structure
	blurred_image = image.filter(ImageFilter.GaussianBlur(radius = 2.5))
	blurred_image = pad_image(blurred_image, factor = 1.8)
	polygon = polygon_coordinates(image_array = np.asarray(blurred_image), debug = False)
	# Pad the non-blurred image
	image = pad_image(image, factor = 1.8)
	if type(polygon) == np.ndarray:
		# Add a chemical ID label to the image
		if label:
			image, label_bounding_box = add_chemical_ID(image=image, chemical_structure_polygon=polygon, debug=False)
		else:
			label_bounding_box = None
		# The image should not be bigger than necessary.
		image, polygon, label_bounding_box = delete_whitespace(image, polygon, label_bounding_box)
		
		# In 20 percent of cases: Make structure image coloured to get more diversity in the training set (colours should be irrelevant for predictions)
		#if random.choice(range(5)) == 0:
		#	image = modify_colours(image, blacken = False)
		#elif random.choice(range(5)) in [1,2]:
		#	image = modify_colours(image, blacken = True)

		# Add curved arrows in the structure
		if arrows:
			arrow_dir = os.path.abspath('./arrows/arrow_images')
			image = add_arrows_to_structure(image, arrow_dir, polygon)

		metadata_dict = make_img_metadata_dict(smiles, polygon, label_bounding_box)
		return image, metadata_dict