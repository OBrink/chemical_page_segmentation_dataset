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
from itertools import cycle
from multiprocessing import Pool
from imantics import Polygons
from skimage import morphology
from polygon_bounding_box_determination import *
from RanDepict import random_depictor

#from generate_structures_with_annotations import generate_structure_and_annotation


class ChemPageSegmentationDatasetCreator:
    """
    The ChemPageSegmentationDatasetCreator class contains everything needed to generate a chemical page segmentation
    dataset based on PubLayNet.
    ___
    Some recycled code from an old version of the mask expansion mechanism 
    of DECIMER Segmentation found its way in here for the polygon annotation creation :)
    ___
    To avoid confusion: 'bounding_box' usually refers to a polygon here and not necessarily to a rectangle
    """
    def __init__(self, smiles_list):
        self.depictor = random_depictor(seed = random.choice(range(10000000)))
        self.depictor.ID_label_text
        self.smiles_iterator = cycle(smiles_list)
     

    def pad_image(self, pil_image: Image, factor: float):
        '''This function takes a Pillow Image and adds 10% padding on every side. It returns the padded Pillow Image'''
        original_size = pil_image.size

        new_size = (int(original_size[0]*factor), int(original_size[1]*factor))
        new_im = Image.new("L", new_size, color = 'white')
        new_im.paste(pil_image, (int((new_size[0]-original_size[0])/2),
                        int((new_size[1]-original_size[1])/2)))
        return new_im


    def get_random_label_position(self, width: int, height: int, x_coordinates: List[int], y_coordinates: List[int]) -> Tuple:
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


    def add_chemical_ID(self, image: Image, chemical_structure_polygon: np.ndarray, debug = False)->Tuple:
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
        x_pos, y_pos = self.get_random_label_position(width, height, x_coordinates, y_coordinates)
        if x_pos == None:
            return im, None

        # Choose random font size
        size = random.choice(font_sizes)
        label_text = self.depictor.ID_label_text()
        
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


    def make_region_dict(self, type: str, polygon: np.ndarray) -> Dict:
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
        self, 
        image_name: str, 
        chemical_structure_polygon: np.ndarray, 
        label_bounding_box = False
        ) -> Dict:
        '''This function takes the name of an image, the coordinates of the chemical structure polygon and (if it exists) the 
        bounding box of the ID label and creates the _via_img_metadata subdictionary for the VIA input.'''
        metadata_dict = {}
        metadata_dict['filename'] = image_name
        #metadata_dict['size'] = int(os.stat(os.path.join(output_dir, image_name)).st_size)
        metadata_dict['regions'] = []
        # Add dict for first region which contains chemical structure
        metadata_dict['regions'].append(self.make_region_dict('chemical_structure', chemical_structure_polygon))
        # If there is an ID label: Add dict for the region that contains the label
        if label_bounding_box:
            label_bounding_box = [(node[1], node[0]) for node in label_bounding_box]
            metadata_dict['regions'].append(self.make_region_dict('chemical_ID', label_bounding_box))
        return metadata_dict


    def make_VIA_dict(
        self, 
        metadata_dicts: List[Dict]
        ) -> Dict:
        '''This function takes a list of Dicts with the region information and returns a dict that can be opened
        using VIA when it is saved as a json file.'''
        VIA_dict = {}
        VIA_dict["_via_img_metadata"] = {}
        for region_dict in metadata_dicts:
            VIA_dict["_via_img_metadata"][region_dict["filename"]] = region_dict
        return VIA_dict["_via_img_metadata"]



    def pick_random_colour(
        self, 
        black = False
        ) -> Tuple[int]:
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


    def modify_colours(
        self, 
        image: Image, 
        blacken = False
        ) -> Image:
        '''This function takes a Pillow Image, makes white pixels transparent, gives every other pixel a given new colour and returns the Image.
        If blacken is True, non-white pixels are going to be replaced with black-ish pixels instead of coloured pixels.'''
        image = image.convert('RGBA')
        datas = image.getdata()
        newData = []
        if blacken:
            new_colour = self.pick_random_colour(black = True)
        else:
            new_colour = self.pick_random_colour()
        for item in datas:
            if not blacken and item[0] > 230 and item[1] > 230 and item[2] > 230:
                newData.append((item[0], item[1], item[2], 0))
            elif item[0] < 230 and item[1] < 230 and item[2] < 230:
                newData.append(new_colour)
            else:
                newData.append(item)
        image.putdata(newData)
        return image


    def add_arrows_to_structure(
        self, 
        image: Image, 
        arrow_dir: str, 
        polygon: np.ndarray
        ) -> Image:
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
            random_colour = self.pick_random_colour()
        else:
            random_colour = (0,0,0,255)
        for _ in range(random.choice(range(4, 15))):
            arrow_image = Image.open(os.path.join(arrow_dir, random.choice(os.listdir(arrow_dir))))
            arrow_image = self.modify_colours(arrow_image)
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

    def adapt_coordinates(
        self, 
        polygon: np.ndarray, 
        label_bounding_box: np.ndarray, 
        crop_indices: Tuple[int,int,int,int]
        ) -> Tuple[np.ndarray, np.ndarray]:
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


    def delete_whitespace(
        self, 
        image: Image, 
        polygon: np.ndarray, 
        label_bounding_box: np.ndarray
        ) -> Tuple:
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
        polygon, label_bounding_box = self.adapt_coordinates(polygon, label_bounding_box, crop_indices)
        return Image.fromarray(image), polygon, label_bounding_box


    def generate_structure_and_annotation(
        self, 
        smiles: str, 
        shape = (200, 200), 
        label: bool = False, 
        arrows : bool = False):
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
        image = self.depictor.random_depiction(smiles, shape)
        image = Image.fromarray(image)
        # Get coordinates of polygon around chemical structure
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius = 2.5))
        blurred_image = self.pad_image(blurred_image, factor = 1.8)
        polygon = polygon_coordinates(image_array = np.asarray(blurred_image), debug = False)
        # Pad the non-blurred image
        image = self.pad_image(image, factor = 1.8)
        if type(polygon) == np.ndarray:
            # Add a chemical ID label to the image
            if label:
                image, label_bounding_box = self.add_chemical_ID(image=image, chemical_structure_polygon=polygon, debug=False)
            else:
                label_bounding_box = None
            # The image should not be bigger than necessary.
            image, polygon, label_bounding_box = self.delete_whitespace(image, polygon, label_bounding_box)
            
            # In 20 percent of cases: Make structure image coloured to get more diversity in the training set (colours should be irrelevant for predictions)
            #if random.choice(range(5)) == 0:
            #	image = modify_colours(image, blacken = False)
            #elif random.choice(range(5)) in [1,2]:
            #	image = modify_colours(image, blacken = True)

            # Add curved arrows in the structure
            if arrows:
                arrow_dir = os.path.abspath('./arrows/arrow_images')
                image = self.add_arrows_to_structure(image, arrow_dir, polygon)

            metadata_dict = self.make_img_metadata_dict(smiles, polygon, label_bounding_box)
            return image, metadata_dict
    



    def make_img_metadata_dicts(
        self, 
        output_names: List[str], 
        output_dir: str, 
        region_dicts: List[Dict]
        ) -> Dict:
        """This function takes a list of output image names the image directory and a list of list of region 
        dictionaries. It creates the _via_img_metadata subdictionary for the VIA input.

        Args:
            output_names (List[str]): [description]
            output_dir (str): [description]
            region_dicts (List[Dict]): [description]

        Returns:
            Dict: _via_img_metadata subdictionary for the VIA input
        """
        metadata_dicts = []
        for index in range(len(output_names)):
            metadata_dict = {}
            metadata_dict['filename'] = output_names[index] + ".png"
            metadata_dict['size'] = int(os.stat(os.path.join(output_dir, output_names[index] + ".png")).st_size)
            metadata_dict['regions'] = region_dicts[index]
            metadata_dicts.append(metadata_dict)
        return metadata_dicts


    def is_valid_position(
        self, 
        structure_paste_position: Tuple[int,int], 
        image_shape: Tuple[int,int], 
        structure_image_shape: Image
        ) -> bool:
        """
        When a chemical structure depiction is pasted into the reaction scheme, it is necessary to make sure that the pasted structure
        is completely inside of the bigger image. This function determines whether or not this condition is fulfilled.

        Args:
            structure_paste_position (Tuple[int,int]): (x,y)
            image_shape (Tuple[int,int]): width, height
            structure_image_shape (Image): (width, height)

        Returns:
            bool: [description]
        """
        x,y = structure_paste_position
        meta_x, meta_y = image_shape
        struc_x, struc_y = structure_image_shape
        if x < 0:
            return
        if y < 0:
            return
        if x + struc_x > meta_x:
            return
        if y + struc_y > meta_y:
            return
        return True


    def paste_it(
        self, 
        image: Image, 
        paste_positions: List[Tuple[int, int]], 
        paste_images: List, 
        annotations: List[Dict]
        ) -> Image:
        """
        This function takes an empty image (PIL.Image), a list of paste positions (Tuple[int,int] -> x,y of upper left corner), 
        a list of images (PIL.Image) to paste into the empty image and a list of annotation Dicts. It pastes the images at the specified
        positions and adapts the annotation coordinates accordingly. 
        It returns the modified image and the modified annotations.

        Args:
            image (Image): PIL.Image
            paste_positions (List[Tuple[int, int]]): [(x,y)]
            paste_images (List): List of images
            annotations (List[Dict]): [description]

        Returns:
            Image: The input image with all pasted elements in it at the given positions.
        """
        for n in range(len(paste_positions)):
            if self.is_valid_position(paste_positions[n], image.size, paste_images[n].size):
                image.paste(paste_images[n], paste_positions[n])
            else:
                # Delete previously pasted arrow that points to a structure outside of the image
                image.paste(Image.new('RGBA', paste_images[n-1].size), paste_positions[n-1])
        return image


    def modify_annotations(
        self, 
        original_regions: List[List[Dict]], 
        paste_anchors: List[Tuple[int,int]]
        ) -> List[Dict]:
        """
        This function takes the original annotated regions (VIA dict format) and the paste position (x, y) [upper left corner].
        The coordinates of the regions are modified according to the position in the new image.
        It returns the modified annotated regions (same format as input but the list is flattened).

        Args:
            original_regions (List[List[Dict]]): region dicts
            paste_anchors (List[Tuple[int,int]]): [(x1, y1), ...]

        Returns:
            List[Dict]: modified regions
        """
        modified_regions = []
        for i in range(len(original_regions)):
            for n in range(len(original_regions[i])):
                x_coords = original_regions[i][n]['shape_attributes']['all_points_x']
                original_regions[i][n]['shape_attributes']['all_points_x'] = [x_coord + paste_anchors[i][0] for x_coord in x_coords]
                y_coords = original_regions[i][n]['shape_attributes']['all_points_y']
                original_regions[i][n]['shape_attributes']['all_points_y'] = [y_coord + paste_anchors[i][1] for y_coord in y_coords]
                modified_regions.append(original_regions[i][n])
        return modified_regions


    def determine_arrow_annotations(
        self, 
        images: List
        ) -> List[Dict]:
        """
        This function takes a list of images, binarizes them, applies a dilation and determines the coordinates of the contours of the
        dilated object in the image. It returns a list of VIA-compatible region dicts.

        Args:
            images (List): List of PIL.Image

        Returns:
            List[Dict]: List of VIA-compatible region dicts that describe the arrow positions
        """
        region_dicts = []
        for image in images:			
            # Binarize and apply dilation to image
            image = np.asarray(image)
            row, col, ch = image.shape
            r, g, b, _ = image[:,:,0], image[:,:,1], image[:,:,2], image[:,:,3]
            image = (r + g + b) / 3

            image = image > 80
            image = np.invert(image)
            image = morphology.binary_dilation(image, morphology.square(10))
            # Determine contour coordinates
            annotation = Polygons.from_mask(image).points
            # If there is more than one connected object, apply dilation again
            if len(annotation) != 1:
                image = morphology.binary_dilation(image, morphology.square(10))
                annotation = Polygons.from_mask(image).points
            #swap x and y around (numpy and Pillow don't agree here, this way, it is consistent)
            annotation = [(x[1], x[0]) for x in annotation[0]]
            region_dicts.append([self.make_region_dict('arrow', annotation)])
        return region_dicts


    def is_diagonal_arrow(
        self, 
        arrow_bbox: List[Tuple[int,int]]
        ) -> bool:
        """
        This function takes the bounding box of and arrow and checks whether or not the arrow is diagonal.

        Args:
            arrow_bbox (List[Tuple[int,int]])

        Returns:
            bool
        """
        # Check x/y ratio
        if (arrow_bbox[2][0] - arrow_bbox[0][0]) / (arrow_bbox[0][1] - arrow_bbox[1][1]) > 0.2:
            # Check y/x ratio
            if (arrow_bbox[2][0] - arrow_bbox[0][0]) / (arrow_bbox[0][1] - arrow_bbox[1][1]) < 5:
                return True
    

    
    def insert_labels(self, image: Image, max_labels: int, label_type: str, arrow_bboxes: bool = False):
        """
        This function takes a PIL Image, a maximal amount of labels to insert and a label type ('r_group' or 'reaction_condition').
        If label type is 'reaction_condition', arrow_bboxes [(x0,y0), [x1,y1)...] must also be given
        It generates random labels that look like the labels and inserts them at random positions (under the condition that there is 
        no overlap with other objects in the image.). It returns the modified image and the region dicts for the labels.

        Args:
            image (Image): PIL.Image
            max_labels (int): Maximal amount of labels
            label_type (str): 'r_group' or 'reaction_condition
            arrow_bboxes (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        n = 0
        region_dicts = []
        paste_positions = []
        for _ in range(500):
            invalid = False
            if n < max_labels:
                # Load random label
                if label_type == 'r_group':
                    label_str = self.depictor.make_R_group_str()
                elif label_type == 'reaction_condition':
                    label_str = self.depictor.reaction_condition_label_text()
                # Determine font properties and type
                fontsize = random.choice(range(12,24))
                try:
                    font_dir = './fonts/'
                    font = ImageFont.truetype(str(os.path.join(font_dir, random.choice(os.listdir(font_dir)))), size = fontsize)
                except OSError:
                    font = ImageFont.load_default()
                # Determine size of created text
                draw = ImageDraw.Draw(image, 'RGBA')
                bbox = draw.multiline_textbbox((0,0), label_str, font) # left, up, right, low
                # Check the resulting region in the reaction scheme if there already is anything else
                try:
                    if label_type == 'r_group':
                        paste_position = (random.choice(range(image.size[0] - bbox[2])), random.choice(range(image.size[1] - bbox[3])))
                        paste_region = image.crop((paste_position[0] - 30, paste_position[1] - 30, paste_position[0] + bbox[2] + 30, paste_position[1] + bbox[3] + 30)) # left, up, right, low
                    elif label_type == 'reaction_condition':
                        # take random arrow positions and place reaction condition label around it
                        arrow_bbox = random.choice(arrow_bboxes)
                        if self.is_diagonal_arrow(arrow_bbox):
                            paste_position = (random.choice(range(arrow_bbox[1][0] + 20, arrow_bbox[3][0] - 20)), random.choice(range(arrow_bbox[1][1] + 20, arrow_bbox[3][1] - 20)))
                        else:
                            paste_position = (random.choice(range(arrow_bbox[1][0] + 1, arrow_bbox[3][0] - 30)), random.choice(range(arrow_bbox[1][1] + 1, arrow_bbox[3][1] + 20)))
                        paste_region = image.crop((paste_position[0] - 10, paste_position[1] - 10, paste_position[0] + bbox[2] + 10, paste_position[1] + bbox[3] + 10)) # left, up, right, low
                except IndexError:
                    invalid = True
                try:
                    if not invalid:
                        # Check that background in paste region is completely white
                        mean = ImageStat.Stat(paste_region).mean
                        if sum(mean)/len(mean) == 0:
                            # Add text element to image if region in image is completely white
                            draw = ImageDraw.Draw(image, 'RGBA')
                            draw.multiline_text(paste_position, label_str, font = font, fill=(0,0,0,255))
                            paste_positions.append(paste_position)
                            n += 1
                            text_bbox = [(bbox[0], bbox[3]), (bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3])] # lower left, upper left, upper right, lower right [(x0, y0), (x1, y1), ...]
                            if label_type == 'r_group':
                                region_dicts.append([self.make_region_dict('R_group_label', text_bbox)])
                            if label_type == 'reaction_condition':
                                region_dicts.append([self.make_region_dict('reaction_condition_label', text_bbox)])
                except ZeroDivisionError:
                    pass
            else: 
                break
        region_dicts = self.modify_annotations(original_regions = region_dicts, paste_anchors = paste_positions)
        return image, region_dicts

    def get_bboxes(
        self, 
        image_list: List, 
        paste_position_list: List[Tuple[int,int]]
        ) -> List[Tuple]:
        """
        This function takes a list of PIL.Image objects and a list of their paste positions (x,y) [upper left corner] and
        returns a list of bounding boxes [(x0, y0), (x1, 1), ...](lower left, upper left, upper right, lower right)

        Args:
            image_list (List): [description]
            paste_position_list (List[Tuple[int,int]]): [description]

        Returns:
            List[Tuple]: [description]
        """
        bboxes = []
        for i in range(len(image_list)):
            lower_left = (paste_position_list[i][0], paste_position_list[i][1] + image_list[i].size[1])
            upper_left = paste_position_list[i]
            upper_right = (paste_position_list[i][0] + image_list[i].size[0], paste_position_list[i][1])
            lower_right = (paste_position_list[i][0] + image_list[i].size[0], paste_position_list[i][1] + image_list[i].size[1])
            bboxes.append((lower_left, upper_left, upper_right, lower_right))
        return bboxes

    def create_reaction_scheme(
        self, 
        ) -> Image:   
        """
        This function creates an artificial (nonsensical) reaction scheme and returns it ()
        Additionally, it creates a json file containing the coordinates elements.
        The json output file is compatible with VIA and is saved in the output directory.

        Returns:
            Image: artificial reaction scheme
        """

        #TODO: Split this giant mess into multiple functions
        # Load chemical structure depictions and annotations
        structure_images = []
        #image_names = []
        #with open(os.path.join(image_dir, 'annotations.json')) as annotation_file:
        #	chem_annotations = json.load(annotation_file)
        annotated_regions = []
        for n in range(random.choice([2,3,5,7,9])):
            smiles = next(self.smiles_iterator)
            side_len = random.choice(range(200,400))
            structure_image, annotation = self.generate_structure_and_annotation(smiles, shape=(side_len, side_len))
            structure_images.append(structure_image)
            annotated_regions.append(annotation['regions'])
            #image_name = random.choice([img for img in os.listdir(image_dir) if img[-3:].lower() == 'png'])
            #image_names.append(image_name)
            #annotated_regions.append(chem_annotations[image_name]['regions'])
            #structure_image = Image.open(os.path.join(image_dir, image_name))
            #structure_images.append(structure_image)
            

        # Load random arrow
        arrow_dir = os.path.abspath('./arrows/horizontal_arrows/')
        arrow_image_name = random.choice(os.listdir(arrow_dir))
        arrow_im = Image.open(os.path.join(arrow_dir, arrow_image_name))
        arrow_image = Image.new('RGBA', arrow_im.size, (255,255,255,255))
        arrow_image = Image.alpha_composite(arrow_image, arrow_im)
        
        # Overview of image sizes
        structure_image_sizes = [image.size for image in structure_images]
        structure_image_x = [size[0] for size in structure_image_sizes]
        structure_image_y = [size[1] for size in structure_image_sizes]
        arrow_image = arrow_image.resize((structure_image_x[0], int(structure_image_y[1]/8)))

        # TODO: Improve the mess below
        # I hope I never have to touch this again but I think this is the easiest way to do this.
        # I cannot get around defining the coordinates for putting everything together differently.
        # Depending on the number of structures, the positions for pasting structures and
        # arrows are defined so that we end up with artificial reaction schemes

        # Two structures, one arrow inbetween
        if len(structure_images) == 2: 
            size = (sum(structure_image_x) + arrow_image.size[0], max(structure_image_y))
            image = Image.new('RGBA', size, (255, 255, 255, 255))
            paste_positions = [(0, int((max(structure_image_y) - structure_images[0].size[1])/2)), # left structure
                (structure_image_x[0] + arrow_image.size[0], int((max(structure_image_y) - structure_images[1].size[1])/2))] #right structure
            arrow_paste_positions = [(structure_image_x[0], int(structure_image_y[0]/2))] # reaction arrow]
            paste_positions += arrow_paste_positions
            paste_images = [structure_images[0], structure_images[1]]
            arrow_image_list = [arrow_image]
            paste_images += arrow_image_list
            image = self.paste_it(image, paste_positions, paste_images, False)
            annotated_regions += self.determine_arrow_annotations([arrow_image])
            annotated_regions = self.modify_annotations(annotated_regions, paste_positions)

        # Three structures, two arrows inbetween
        if len(structure_images) == 3:
            if random.choice([True, False]):
                # Horizontal reaction scheme with three stuctures
                size = (sum(structure_image_x) + arrow_image.size[0] * 2, max(structure_image_y))
                image = Image.new('RGBA', size, (255, 255, 255, 255))
                paste_positions = [(0, int((max(structure_image_y) - structure_images[0].size[1])/2)), # left structure
                    (structure_image_x[0] + arrow_image.size[0], int((max(structure_image_y) - structure_images[1].size[1])/2)), # middle structure
                    (sum(structure_image_x[:2]) + arrow_image.size[0] * 2, int((max(structure_image_y) - structure_images[2].size[1])/2))] # right structure
                arrow_paste_positions = [(structure_image_x[0], int(max(structure_image_y)/2)), # left reaction arrow
                    (sum(structure_image_x[:2]) + arrow_image.size[0], int(max(structure_image_y)/2))] # right reaction arrow
                paste_positions += arrow_paste_positions
                paste_images = [structure_images[0], structure_images[1], structure_images[2]]
                arrow_image_list = [arrow_image.rotate(180, expand=True, fillcolor=(255,255,255,255)), arrow_image]
                paste_images += arrow_image_list
                image = self.paste_it(image, paste_positions, paste_images, False)
                annotated_regions += self.determine_arrow_annotations([arrow_image.rotate(180, expand=True, fillcolor=(255,255,255,255)), arrow_image])
                annotated_regions = self.modify_annotations(annotated_regions, paste_positions)
            else:
                # Vertical reaction scheme with three stuctures
                size = (max(structure_image_x), sum(structure_image_y) + arrow_image.size[0] * 2)
                image = Image.new('RGBA', size, (255, 255, 255, 255))
                paste_positions = [(int((max(structure_image_x) - structure_images[0].size[0])/2), 0), # upper structure
                    (int((max(structure_image_x) - structure_images[1].size[0])/2), sum(structure_image_y[:1]) + arrow_image.size[0]), # middle structure
                    (int((max(structure_image_x) - structure_images[2].size[0])/2), sum(structure_image_y[:2]) + arrow_image.size[0] * 2)] # lower structure
                arrow_paste_positions = [(int(max(structure_image_x)/2), structure_image_y[0]), # upper reaction arrow
                    (int(max(structure_image_x)/2), sum(structure_image_y[:2]) + arrow_image.size[0])] # lower arrow
                paste_positions += arrow_paste_positions
                paste_images = [structure_images[0], structure_images[1], structure_images[2]] 
                arrow_image_list = [arrow_image.rotate(90, expand=True, fillcolor=(255,255,255,255)), arrow_image.rotate(270, expand=True, fillcolor=(255,255,255,255))]
                paste_images += arrow_image_list
                image = self.paste_it(image, paste_positions, paste_images, False)
                annotated_regions += self.determine_arrow_annotations([arrow_image.rotate(90, expand=True, fillcolor=(255,255,255,255)), arrow_image.rotate(270, expand=True, fillcolor=(255,255,255,255))])
                annotated_regions = self.modify_annotations(annotated_regions, paste_positions)

        if len(structure_images) in [5, 9]:
            size = (sum([structure_image_x[n] for n in [0,1,2]]) + arrow_image.size[0] * 2, sum([structure_image_y[n] for n in [1,3,4]]) + arrow_image.size[0] * 2)
            image = Image.new('RGBA', size, (255, 255, 255, 255))
            paste_positions = [(0, int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1] - 0.5 * structure_image_y[0])), # left structure
                (sum(structure_image_x[:1]) + arrow_image.size[0] * 1, int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1] - 0.5 * structure_image_y[1])), # middle_structure
                (sum(structure_image_x[:2]) + arrow_image.size[0] * 2, int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1] - 0.5 * structure_image_y[2])), # right structure
                (int(structure_image_x[0] + arrow_image.size[0] + 0.5 * structure_image_x[1] - 0.5 * structure_image_x[3]), 0), # upper structure
                (int(structure_image_x[0] + arrow_image.size[0] + 0.5 * structure_image_x[1] - 0.5 * structure_image_x[4]), sum([structure_image_y[n] for n in [1,3]]) + arrow_image.size[0] * 2)] # lower structure
            arrow_paste_positions = [(sum(structure_image_x[:1]), int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1])), # left arrow
                (sum(structure_image_x[:2]) + arrow_image.size[0], int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1])), # right arrow
                (int(structure_image_x[0] + arrow_image.size[0] + 0.5 * structure_image_x[1]), structure_image_y[3]), # upper arrow
                (int(structure_image_x[0] + arrow_image.size[0] + 0.5 * structure_image_x[1]), sum([structure_image_y[n] for n in [1,3]]) + arrow_image.size[0])] # lower arrow
            paste_images = structure_images[:4] + [structure_images[4]]
            arrow_image_list = [arrow_image.rotate(180, expand=True, fillcolor=(255,255,255,255)), arrow_image, arrow_image.rotate(90, expand=True, fillcolor=(255,255,255,255)), arrow_image.rotate(270, expand=True, fillcolor=(255,255,255,255))]
            if len(structure_images) == 5:	
                annotated_regions += self.determine_arrow_annotations(arrow_image_list)
                paste_positions += arrow_paste_positions
                paste_images += arrow_image_list
                annotated_regions = self.modify_annotations(annotated_regions, paste_positions)
            if len(structure_images) == 9:
                # TODO: Shorten this. Trying to define the positions in a shorter way gives me a headache right now but it works the way it is.
                # I am aware that this is a mess.
                # Add diagonal and structures
                middle_structure = (sum(structure_image_x[:1]) + arrow_image.size[0] * 1, int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1] - 0.5 * structure_image_y[1]))
                # upper left 
                arrow_image_1 = arrow_image.rotate(135, expand=True, fillcolor=(255,255,255,255))
                arrow_paste_position_1 = (middle_structure[0]-arrow_image_1.size[0], middle_structure[1]-arrow_image_1.size[1])
                structure_paste_position_1 = (arrow_paste_position_1[0]-structure_image_x[5], arrow_paste_position_1[1]-int(0.75 * structure_image_y[5]))
                # upper right 
                arrow_image_2 = arrow_image_1.rotate(270, expand=True, fillcolor=(255,255,255,255))
                arrow_paste_position_2 = (middle_structure[0] + structure_image_x[1], middle_structure[1]-arrow_image_2.size[1])
                structure_paste_position_2 = (arrow_paste_position_2[0] + arrow_image_2.size[0], arrow_paste_position_2[1]-int(0.75 * structure_image_y[6]))
                # lower right 
                arrow_image_3 = arrow_image_2.rotate(270, expand=True, fillcolor=(255,255,255,255))
                arrow_paste_position_3 = (middle_structure[0] + structure_image_x[1], middle_structure[1] + structure_image_y[1])
                structure_paste_position_3 = (arrow_paste_position_3[0] + arrow_image_3.size[0], arrow_paste_position_3[1] + arrow_image_3.size[1] - int(0.25 * structure_image_y[7]))
                # lower left
                arrow_image_4 = arrow_image_3.rotate(270, expand=True, fillcolor=(255,255,255,255))
                arrow_paste_position_4 =(middle_structure[0] - arrow_image_4.size[0], middle_structure[1] + structure_image_y[1])
                structure_paste_position_4 = (arrow_paste_position_4[0]-structure_image_x[8], arrow_paste_position_4[1] + arrow_image_4.size[1] - int(0.25 * structure_image_y[8]))
                
                # Only add structures and arrows that point at them if the structures fit in the image
                annotated_regions_copy = deepcopy(annotated_regions)
                for n in range(1,5):
                    if self.is_valid_position(eval("structure_paste_position_" + str(n)), image.size, structure_images[4+n].size):
                        arrow_paste_positions.append(eval("arrow_paste_position_" + str(n)))
                        arrow_image_list.append(eval("arrow_image_" + str(n)))
                        paste_images.append(structure_images[4+n])
                        paste_positions.append(eval("structure_paste_position_" + str(n)))
                    else:
                        annotated_regions.remove(annotated_regions_copy[4+n])
                
                
                annotated_regions += self.determine_arrow_annotations(arrow_image_list)
                paste_positions += arrow_paste_positions
                paste_images += arrow_image_list

                annotated_regions = self.modify_annotations(annotated_regions, paste_positions)
            image = self.paste_it(image, paste_positions, paste_images, False)

        if len(structure_images) == 7:
            size = (sum([structure_image_x[n] for n in [0,1,2]]) + arrow_image.size[0] * 2, sum([structure_image_y[n] for n in [1,3,4]]) + arrow_image.size[0] * 2)
            image = Image.new('RGBA', size, (255, 255, 255, 255))
            # Horizontal reaction flow
            paste_positions = [(0, int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1] - 0.5 * structure_image_y[0])), # left structure
                (sum(structure_image_x[:1]) + arrow_image.size[0] * 1, int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1] - 0.5 * structure_image_y[1])), # middle_structure
                (sum(structure_image_x[:2]) + arrow_image.size[0] * 2, int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1] - 0.5 * structure_image_y[2]))] # right structure
            arrow_paste_positions = [(sum(structure_image_x[:1]), int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1])), # left arrow
                (sum(structure_image_x[:2]) + arrow_image.size[0], int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1]))] # right arrow
                
            paste_images = structure_images[:3]
            arrow_image_list = [arrow_image.rotate(180, expand=True, fillcolor=(255,255,255,255)), arrow_image]
                
            # TODO: Shorten this. Trying to define the positions in a shorter way gives me a headache right now but it works the way it is.
            # I am aware that this is a mess.
            # Add diagonal and structures
            middle_structure = (sum(structure_image_x[:1]) + arrow_image.size[0] * 1, int(structure_image_y[3] + arrow_image.size[0] + 0.5 * structure_image_y[1] - 0.5 * structure_image_y[1]))
            # upper left 
            arrow_image_1 = arrow_image.rotate(135, expand=True, fillcolor=(255,255,255,255))
            arrow_paste_position_1 = (middle_structure[0]-arrow_image_1.size[0], middle_structure[1]-arrow_image_1.size[1])
            structure_paste_position_1 = (arrow_paste_position_1[0]-structure_image_x[3], arrow_paste_position_1[1]-int(0.75 * structure_image_y[3]))
            # upper right 
            arrow_image_2 = arrow_image_1.rotate(270, expand=True, fillcolor=(255,255,255,255))
            arrow_paste_position_2 = (middle_structure[0] + structure_image_x[1], middle_structure[1]-arrow_image_2.size[1])
            structure_paste_position_2 = (arrow_paste_position_2[0] + arrow_image_2.size[0], arrow_paste_position_2[1]-int(0.75 * structure_image_y[4]))
            # lower right 
            arrow_image_3 = arrow_image_2.rotate(270, expand=True, fillcolor=(255,255,255,255))
            arrow_paste_position_3 = (middle_structure[0] + structure_image_x[1], middle_structure[1] + structure_image_y[1])
            structure_paste_position_3 = (arrow_paste_position_3[0] + arrow_image_3.size[0], arrow_paste_position_3[1] + arrow_image_3.size[1] - int(0.25 * structure_image_y[5]))
            # lower left
            arrow_image_4 = arrow_image_3.rotate(270, expand=True, fillcolor=(255,255,255,255))
            arrow_paste_position_4 =(middle_structure[0] - arrow_image_4.size[0], middle_structure[1] + structure_image_y[1])
            structure_paste_position_4 = (arrow_paste_position_4[0]-structure_image_x[6], arrow_paste_position_4[1] + arrow_image_4.size[1] - int(0.25 * structure_image_y[6]))
            
            # Only add structures and arrows that point at them if the structures fit in the image
            annotated_regions_copy = deepcopy(annotated_regions)
            for n in range(1,5):
                if self.is_valid_position(eval("structure_paste_position_" + str(n)), image.size, structure_images[2+n].size):
                    arrow_paste_positions.append(eval("arrow_paste_position_" + str(n)))
                    arrow_image_list.append(eval("arrow_image_" + str(n)))
                    paste_images.append(structure_images[2+n])
                    paste_positions.append(eval("structure_paste_position_" + str(n)))
                else:
                    annotated_regions.remove(annotated_regions_copy[2+n])
                
            annotated_regions += self.determine_arrow_annotations(arrow_image_list)
            paste_positions += arrow_paste_positions
            paste_images += arrow_image_list

            annotated_regions = self.modify_annotations(annotated_regions, paste_positions)
            image = self.paste_it(image, paste_positions, paste_images, False)

        # Insert labels and update annotations
        image, R_group_regions = self.insert_labels(image, len(structure_images)-1, 'r_group')
        annotated_regions += R_group_regions
        arrow_bboxes = self.get_bboxes(arrow_image_list, arrow_paste_positions)
        image, reaction_condition_regions = self.insert_labels(image, len(structure_images)-1, 'reaction_condition', arrow_bboxes)
        annotated_regions += reaction_condition_regions
        #image.show()
        #image.save(os.path.join(output_dir, str(number) + ".png"))
        #for img in structure_images:
        #	img.close()
        #for image_name in image_names:
        #	try:
        #		os.rename(str((os.path.join(image_dir, image_name))), str(os.path.join(image_dir, "used", image_name)))
        #	# Happens if image is accidently accessed twice. This might lead to some structures being represented twice in the dataset but that should not happen often.
        #	except FileNotFoundError:
        #		with open('reaction_scheme_creation_error_log.txt', 'a') as error_log:
        #			error_log.write('FileNotFoundError when copying file ' + image_name + '\n')
        return image.convert('RGB'), annotated_regions

# def main():
# 	'''This script takes an input directory with images of chemical structure depictions, a directory
# 	that contains images of horizontal arrows and an output directory.
# 	It creates artificial (nonsensical) reaction schemes and saves them in the
# 	output directory. Additionally, it creates a json file containing the coordinates elements.
# 	The json output file is compatible with VIA and is saved in the output directory.'''
# 	if len(sys.argv) != 4:
# 		print('Usage: ' + sys.argv[0] + 'input_dir arrow_dir output_dir')
# 	else:
# 		# Define relevant paths and get everything into the right shape to feed it to Pool.starmap
# 		input_dir = os.path.abspath(sys.argv[1])
# 		arrow_dir = os.path.abspath(sys.argv[2])
# 		output_dir = os.path.abspath(sys.argv[3])
# 		dirlist = os.listdir(input_dir)

# 		for _ in range(10):
# 			make_R_group_str()

# 		l = 100000 # number of reaction schemes
# 		starmap_iterable = [(input_dir, output_dir, arrow_dir, number) for number in range(l)]
# 		with Pool() as p:
# 			region_dicts = p.starmap(coordination, starmap_iterable)
# 		metadata_dicts = make_img_metadata_dicts(list([str(n) for n in range(l)]), output_dir, region_dicts)
            

# 		via_json = make_VIA_dict(metadata_dicts)
# 		with open(os.path.join(output_dir, 'annotations.json'), 'w') as output:
# 			json.dump(via_json, output)

# if __name__ == '__main__':
# 	main()

#_____
# Methods that are class methods of random_depictor

# def ID_label_text()->str:
# 	'''This function returns a string that resembles a typical chemica ID label'''
# 	label_num = range(1,50)
# 	label_letters = ["a", "b", "c", "d", "e", "f", "g", "i", "j", "k", "l", "m", "n", "o"]
# 	options = ["only_number", "num_letter_combination", "numtonum", "numcombtonumcomb"]
# 	option = random.choice(options)
# 	if option == "only_number":
# 		return str(random.choice(label_num))
# 	if option == "num_letter_combination":
# 		return str(random.choice(label_num)) + random.choice(label_letters)
# 	if option == "numtonum":
# 		return str(random.choice(label_num)) + "-" + str(random.choice(label_num))
# 	if option == "numcombtonumcomb":
# 		return str(random.choice(label_num)) + random.choice(label_letters) + "-" + random.choice(label_letters)

# def make_reaction_condition_str():
# 	'''This function returns a random string that looks like a reaction condition label.'''
# 	def new_elements():
# 		'''Randomly redefine reaction_time, solvent and other_reactand.'''
# 		reaction_time = random.choice([str(num) for num in range(30)]) + random.choice([' h', ' min'])
# 		solvent = random.choice(['MeOH', 'EtOH', 'CHCl3', 'DCM', 'iPrOH', 'MeCN', 'DMSO', 'pentane', 'hexane', 'benzene', 'Et2O', 'THF', 'DMF'])
# 		other_reactand = random.choice(['HF', 'HCl', 'HBr', 'NaOH', 'Et3N', 'TEA', 'Ac2O', 'DIBAL', 'DIBAL-H', 'DIPEA', 'DMAP', 'EDTA', 'HOBT', 'HOAt', 'TMEDA', 'p-TsOH', 'Tf2O'])
# 		return reaction_time, solvent, other_reactand

# 	reaction_condition_label = ''
# 	label_type = random.choice(['A', 'B', 'C', 'D'])
# 	if label_type in ['A', 'B']:
# 		for n in range(random.choice(range(1, 5))):
# 			reaction_time, solvent, other_reactand = new_elements()
# 			if label_type == 'A':
# 				reaction_condition_label += str(n+1) + ' ' + other_reactand + ', ' + solvent + ', ' + reaction_time + '\n'
# 			elif label_type == 'B':
# 				reaction_condition_label += str(n+1) + ' ' + other_reactand + ', ' + solvent + ' (' + reaction_time + ')\n'
# 	elif label_type == 'C':
# 		reaction_time, solvent, other_reactand = new_elements()
# 		reaction_condition_label += other_reactand + '\n' + solvent + '\n' + reaction_time
# 	elif label_type == 'D':
# 		reaction_condition_label += random.choice(new_elements())
# 	return reaction_condition_label



# def make_R_group_str():
# 	'''This function returns a random string that looks like an R group label'''
# 	rest_variables = ['X','Y','Z','R','R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','Y2','D']
# 	# Load list of superatoms (from OSRA)
# 	with open('./superatom.txt') as superatoms:
# 		superatoms = superatoms.readlines()
# 		superatoms = [s[:-2] for s in superatoms]
# 	label_type = random.choice(['A', 'B', 'C', 'D', 'E'])
# 	R_group_label = ''
# 	if label_type == 'A':
# 		for _ in range(1, random.choice(range(2,6))):
# 			R_group_label += random.choice(rest_variables) + ' = ' + random.choice(superatoms) + '\n'
# 	elif label_type == 'B':
# 		R_group_label += '      ' + random.choice(rest_variables)  + '\n'
# 		for n in range(1, random.choice(range(2,6))):
# 			R_group_label += str(n) + '    ' + random.choice(superatoms)  + '\n'
# 	elif label_type == 'C':
# 		R_group_label += '      ' + random.choice(rest_variables)  + '      ' + random.choice(rest_variables) + '\n'
# 		for n in range(1, random.choice(range(2,6))):
# 			R_group_label += str(n) + '  ' + random.choice(superatoms) + '  ' + random.choice(superatoms) + '\n'
# 	elif label_type == 'D':
# 		R_group_label += '      ' + random.choice(rest_variables)  + '      ' + random.choice(rest_variables) + '      ' + random.choice(rest_variables) + '\n'
# 		for n in range(1, random.choice(range(2,6))):
# 			R_group_label += str(n) + '  ' + random.choice(superatoms) + '  ' + random.choice(superatoms) + '  ' + random.choice(superatoms) + '\n'
# 	if label_type == 'E':
# 		for n in range(1, random.choice(range(2,6))):
# 			R_group_label += str(n) + '  ' + random.choice(rest_variables) + ' = ' + random.choice(superatoms) + '\n'
# 	return R_group_label


# _____
# DUPLICATES
# def add_chemical_ID(image: Image, chemical_structure_polygon: np.ndarray, debug = False) -> Image:
# 	'''This function takes a PIL Image and the coordinates of the region that contains a chemical structure diagram
# 	and adds random text that looks like a chemical ID label around the structure.
# 	It returns the modified image and the coordinates of the bounding box of the added text.
# 	If it is not possible to place the label around the structure without an overlap, it returns None'''	
# 	im = image.convert('RGB')
# 	y_coordinates = [node[0] for node in chemical_structure_polygon]
# 	x_coordinates = [node[1] for node in chemical_structure_polygon]

# 	# Choose random font
# 	font_dir = os.path.abspath("./fonts/")
# 	fonts = os.listdir(font_dir)
# 	font_sizes = range(7, 18)

# 	# Define random position for text element around the chemical structure (if it can be placed around it)
# 	width, height = im.size
# 	x_pos, y_pos = get_random_label_position(width, height, x_coordinates, y_coordinates)
# 	if x_pos == None:
# 		return im, None

# 	# Choose random font size
# 	size = random.choice(font_sizes)
# 	label_text = ID_label_text()
# 	label_bounding_box = [(x_pos - 0.5* size, y_pos + 1.3*size), (x_pos - 0.5*size, y_pos - 0.5*size), (x_pos + 1.5*size + 12*len(label_text), y_pos - 0.5*size), (x_pos + 1.5 * size + 12*len(label_text), y_pos + 1.3*size),]
# 	try:
# 		font = ImageFont.truetype(str(os.path.join(font_dir, random.choice(fonts))), size = size)
# 	except OSError:
# 		font = ImageFont.load_default()

# 	# Add text element to image
# 	draw = ImageDraw.Draw(im, 'RGBA')
# 	draw.text((x_pos,y_pos), label_text, font = font, fill=(0,0,0,255))
    
# 	# For illustrative reasons: Show the bounding boxes around structure and label
# 	if debug:
# 		polygon = [(node[1], node[0]) for node in chemical_structure_polygon]
# 		draw.polygon(label_bounding_box, fill = (255,0,0,50))
# 		draw.polygon(polygon, fill = (0,255,0,50))
# 		im.show()
# 	return im, label_bounding_box


# def make_VIA_dict(metadata_dicts: List[Dict]) -> Dict:
# 	'''This function takes a list of Dicts with the region information and returns a dict that can be opened
# 	using VIA when it is saved as a json file.'''
# 	VIA_dict = {}
# 	VIA_dict["_via_img_metadata"] = {}
# 	for metadata_dict in metadata_dicts:
# 		VIA_dict["_via_img_metadata"][metadata_dict["filename"]] = metadata_dict
# 	return VIA_dict["_via_img_metadata"]





# def make_region_dict(type: str, polygon: np.ndarray) -> Dict:
# 	'''In order to avoid redundant code in make_img_metadata_dict, this function creates the dict that holds the information 
# 	for one single region. It expects a type (annotation string for the region) and an array containing tuples [(x0, y0), (x1, y1)...]'''
# 	region_dict = {}
# 	region_dict['region_attributes'] = {}
# 	region_dict['region_attributes']['type'] = type
# 	region_dict['shape_attributes'] = {}
# 	region_dict['shape_attributes']['name'] = 'polygon'
# 	y_coordinates = [int(node[1]) for node in polygon[0]]
# 	x_coordinates = [int(node[0]) for node in polygon[0]]
# 	region_dict['shape_attributes']['all_points_x'] = list(x_coordinates)
# 	region_dict['shape_attributes']['all_points_y'] = list(y_coordinates)
# 	return region_dict