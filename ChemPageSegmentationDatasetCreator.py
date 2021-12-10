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
    
    """
    def __init__(self, smiles_list, load_PLN_annotations=True):
        self.depictor = random_depictor(seed = random.choice(range(10000000)))
        self.depictor.ID_label_text
        self.smiles_iterator = cycle(smiles_list)
        # Random images to be pasted (eg. COCO images) for diversification of elements on pages
        self.random_image_dir = os.path.normpath('./random_images/')
        self.random_image_iterator = cycle([os.path.join(self.random_image_dir, im) for im in  os.listdir(self.random_image_dir)])
        # PubLayNet images
        self.PLN_dir = os.path.normpath('./publaynet/')
        #self.PLN_image_iterator = cycle([os.path.join(self.PLN_dir, 'train', im) 
        #                                 for im in os.listdir(os.path.join(self.PLN_dir, 'train'))])
        # Load PLN annotations; this may take 1-2 minutes
        if load_PLN_annotations:
            self.PLN_annotations = self.load_PLN_annotations()
            self.PLN_annotations['categories'].append({'supercategory': '', 'id': 6, 'name': 'chemical_structure'})
            self.PLN_annotations['categories'].append({'supercategory': '', 'id': 7, 'name': 'chemical_ID'})
            self.PLN_annotations['categories'].append({'supercategory': '', 'id': 8, 'name': 'arrow'})
            self.PLN_annotations['categories'].append({'supercategory': '', 'id': 9, 'name': 'R_group_label'})
            self.PLN_annotations['categories'].append({'supercategory': '', 'id': 10, 'name': 'reaction_condition_label'})
            self.PLN_annotations['categories'].append({'supercategory': '', 'id': 11, 'name': 'chemical_structure_with_curved_arrows'})
            self.PLN_page_annotation_iterator = cycle([self.PLN_annotations[page]
                                                for page 
                                                in self.PLN_annotations.keys() 
                                                if page != 'categories'])
        else:
            self.PLN_annotations, self.PLN_page_annotation_iterator = False, False


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
        font_sizes = range(16, 40)

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


    def make_region_dict(self, type: str, polygon: np.ndarray, smiles: str=False) -> Dict:
        '''In order to avoid redundant code in make_img_metadata_dict, this function creates the dict that holds the information 
        for one single region. It expects a type (annotation string for the region) and an array containing tuples [(y0, x0), (y1, x1)...]'''
        region_dict = {}
        region_dict['region_attributes'] = {}
        region_dict['region_attributes']['type'] = type
        if smiles:
            region_dict['smiles'] = smiles
        region_dict['shape_attributes'] = {}
        region_dict['shape_attributes']['name'] = 'polygon'
        y_coordinates = [int(node[0]) for node in polygon]
        x_coordinates = [int(node[1]) for node in polygon]
        region_dict['shape_attributes']['all_points_x'] = list(x_coordinates)
        region_dict['shape_attributes']['all_points_y'] = list(y_coordinates)
        return region_dict


    # def make_img_metadata_dict(
    #     self, 
    #     image_name: str, 
    #     chemical_structure_polygon: np.ndarray, 
    #     label_bounding_box = False
    #     ) -> Dict:
    #     '''This function takes the name of an image, the coordinates of the chemical structure polygon and (if it exists) the 
    #     bounding box of the ID label and creates the _via_img_metadata subdictionary for the VIA input.'''
    #     metadata_dict = {}
    #     metadata_dict['filename'] = image_name
    #     #metadata_dict['size'] = int(os.stat(os.path.join(output_dir, image_name)).st_size)
    #     metadata_dict['regions'] = []
    #     # Add dict for first region which contains chemical structure
    #     metadata_dict['regions'].append(self.make_region_dict('chemical_structure', chemical_structure_polygon))
    #     # If there is an ID label: Add dict for the region that contains the label
    #     if label_bounding_box:
    #         label_bounding_box = [(node[1], node[0]) for node in label_bounding_box]
    #         metadata_dict['regions'].append(self.make_region_dict('chemical_ID', label_bounding_box))
    #     return metadata_dict


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
        return image.convert('RGB')


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
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius = 5))
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
            
            #In 20 percent of cases: Make structure image coloured to get more diversity in the training set (colours should be irrelevant for predictions)
            if random.choice(range(5)) == 0:
                image = self.modify_colours(image, blacken = False)
            elif random.choice(range(5)) in [1,2]:
                image = self.modify_colours(image, blacken = True)

            # Add curved arrows in the structure
            if arrows:
                arrow_dir = os.path.abspath('./arrows/arrow_images')
                image = self.add_arrows_to_structure(image, arrow_dir, polygon)

            region_annotations = []
            region_annotations.append(self.make_region_dict('chemical_structure', polygon, smiles))
            if label_bounding_box:
                label_bounding_box = [(node[1], node[0]) for node in label_bounding_box]
                region_annotations.append(self.make_region_dict('chemical_ID', label_bounding_box))
            #make_region_dict(self, type: str, polygon: np.ndarray, smiles: str=False) -> Dict:

            #metadata_dict = self.make_img_metadata_dict(smiles, polygon, label_bounding_box)
            return image, region_annotations
    



    # def make_img_metadata_dicts(
    #     self, 
    #     output_names: List[str], 
    #     output_dir: str, 
    #     region_dicts: List[Dict]
    #     ) -> Dict:
    #     """This function takes a list of output image names the image directory and a list of list of region 
    #     dictionaries. It creates the _via_img_metadata subdictionary for the VIA input.

    #     Args:
    #         output_names (List[str]): [description]
    #         output_dir (str): [description]
    #         region_dicts (List[Dict]): [description]

    #     Returns:
    #         Dict: _via_img_metadata subdictionary for the VIA input
    #     """
    #     metadata_dicts = []
    #     for index in range(len(output_names)):
    #         metadata_dict = {}
    #         metadata_dict['filename'] = output_names[index] + ".png"
    #         metadata_dict['size'] = int(os.stat(os.path.join(output_dir, output_names[index] + ".png")).st_size)
    #         metadata_dict['regions'] = region_dicts[index]
    #         metadata_dicts.append(metadata_dict)
    #     return metadata_dicts


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
                fontsize = random.choice(range(16,36))
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
                        buffer = 30
                        paste_region = image.crop((paste_position[0] - buffer, paste_position[1] - buffer, paste_position[0] + bbox[2] + buffer, paste_position[1] + bbox[3] + buffer)) # left, up, right, low
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
                        if sum(mean)/len(mean) == 255:
                            # Add text element to image if region in image is completely white
                            draw = ImageDraw.Draw(image, 'RGBA')
                            draw.multiline_text(paste_position, label_str, font = font, fill=(0,0,0,255))
                            paste_positions.append(paste_position)
                            n += 1
                            text_bbox = [(bbox[3], bbox[0]), (bbox[1], bbox[0]), (bbox[1], bbox[2]), (bbox[3], bbox[2])] # lower left, upper left, upper right, lower right [(x0, y0), (x1, y1), ...]
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
        annotated_regions = []
        for n in range(random.choice([2,3,5,7,9])):
            smiles = next(self.smiles_iterator)
            side_len = random.choice(range(200,400))
            structure_image, annotation = self.generate_structure_and_annotation(smiles, shape=(side_len, side_len), label=random.choice([True, False]))
            structure_images.append(structure_image)
            annotated_regions.append(annotation)
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
        if random.choice([True, False]):
            image, R_group_regions = self.insert_labels(image, len(structure_images)-1, 'r_group')
            annotated_regions += R_group_regions
        if random.choice([True, False]):
            arrow_bboxes = self.get_bboxes(arrow_image_list, arrow_paste_positions)
            image, reaction_condition_regions = self.insert_labels(image, len(structure_images)-1, 'reaction_condition', arrow_bboxes)
            annotated_regions += reaction_condition_regions
        return image.convert('RGB'), annotated_regions


    def load_PLN_annotations(
        self,
        ) -> Dict:
        """
        This function loads the PubLayNet annotation dictionary and returns them in a clear format.
        The returned dictionary only contain entries where the images actually exist locally in PLN_image_directory.
        (--> No problems if only a part of PubLayNet was downloaded.)

        Args:
            PLN_json_path (str): Path of PubLayNet annotation file
            PLN_image_dir ([type]): Path of directory with PubLayNet images

        Returns:
            Dict
        """
        #PLN_json_path = os.path.join(self.PLN_dir, 'train.json')
        #PLN_image_dir = os.path.join(self.PLN_dir, 'train')
        PLN_json_path = os.path.join(self.PLN_dir, 'val.json')
        PLN_image_dir = os.path.join(self.PLN_dir, 'val')
        with open(PLN_json_path) as annotation_file:
            PLN_annotations = json.load(annotation_file)
        PLN_dict = {}
        PLN_dict['categories'] = PLN_annotations['categories']
        for image in PLN_annotations['images']:
            if os.path.exists(os.path.join(PLN_image_dir,  image['file_name'])):
                PLN_dict[image['id']] = {'file_name': os.path.join(PLN_image_dir,  image['file_name']), 'annotations': []}
        for ann in PLN_annotations['annotations']:
            if ann['image_id'] in PLN_dict.keys():
                PLN_dict[ann['image_id']]['annotations'].append(ann)
        return PLN_dict


    def fix_polygon_coordinates(
        self,
        x_coords: List[int], 
        y_coords: List[int], 
        shape: Tuple[int]
        ) -> Tuple[List[int], List[int]]:
        """
        If the coordinates are placed outside of the image, this function takes the lists of coordinates and the 
        image shape and adapts coordinates that are placed outside of the image to be placed at its borders.

        Args:
            x_coords (List[int]): x coordinates
            y_coords (List[int]): y coordinates
            shape (Tuple[int]): image shape

        Returns:
            Tuple[List[int], List[int]]: x coordinates, y coordinates
        """
        for n in  range(len(x_coords)):
            if x_coords[n] < 0:
                x_coords[n] = 0
            if y_coords[n] < 0:
                y_coords[n] = 0
            if x_coords[n] > shape[0]:
                x_coords[n] = shape[0] - 1
            if y_coords[n] > shape[1]:
                y_coords[n] = shape[1] -1
        return x_coords, y_coords


    def modify_annotations_PLN(
        self, 
        #annotation_dir: str, 
        #image_name: str, 
        regions_dicts: List[Dict],
        old_image_shape: Tuple[int], 
        new_image_shape: Tuple[int], 
        paste_anchor: Tuple[int]
        ) -> Dict:
        """
        This function takes information about the region where an image (structure, scheme ) has been
        inserted. 
        The coordinates of the regions are modified according to the resizing of the image and the position on the
        page where it has been pasted.

        Args:
            regions_dict (List[Dict])
            old_image_shape (Tuple[int])
            new_image_shape (Tuple[int])
            paste_anchor (Tuple[int])

        Returns:
            Dict
        """
        modified_annotations = []
        if regions_dicts:
            for region in regions_dicts:
                categories = {'chemical_structure': 6, 
                            'chemical_ID': 7, 
                            'arrow': 8, 
                            'R_group_label': 9, 
                            'reaction_condition_label': 10, 
                            'chemical_structure_with_curved_arrows': 11}
                category = region['region_attributes']['type']
                category_ID = categories[category]

                # Load coordinates and alter them according to the resizing
                x_coords = region['shape_attributes']['all_points_x']
                y_coords = region['shape_attributes']['all_points_y']
                #x_coords, y_coords = self.fix_polygon_coordinates(x_coords, y_coords, old_image_shape)
                x_ratio = new_image_shape[0]/old_image_shape[0]
                y_ratio = new_image_shape[1]/old_image_shape[1]
                x_coords = [x_ratio*x_coord + paste_anchor[0] for x_coord in x_coords]
                y_coords = [y_ratio*y_coord + paste_anchor[1] for y_coord in y_coords]
                # Get the coordinates into the PLN annotation format ([x0, y0, x1, y1, ..., xn, yn])
                modified_annotation = {'segmentation': [[]], 'category_id': category_ID}
                for n in range(len(x_coords)):
                    modified_annotation['segmentation'][0].append(x_coords[n])
                    modified_annotation['segmentation'][0].append(y_coords[n])
                modified_annotations.append(modified_annotation)
        return modified_annotations


    def determine_images_per_region(
        self, 
        region: Tuple[int]
        ) -> Tuple[int, int]:
        """
        This function takes the bounding box coordinates of a region and returns two integers which indicate how many 
        chemical structure depictions should be added (int1: vertical, int2: horizontal). The returned values depend on 
        the region size and a random influence.

        Args:
            region (Tuple[int]): paste region bounding box

        Returns:
            Tuple[int, int]: "rows", "columns"
        """
        min_x, max_y, max_x, min_y = region
        x_diff = max_x - min_x
        y_diff = max_y - min_y
        n = random.choice([100, 100, 100, 100, 150, 150, 200, 200])
        horizontal_int = round(x_diff/n)
        vertical_int = round(y_diff/n)
        if horizontal_int < 1:
            horizontal_int = 1
        if vertical_int < 1:
            vertical_int = 1
        return vertical_int, horizontal_int


    def paste_images(
        self, 
        image: Image, 
        region: Tuple[int], 
        images_vertical: int, 
        images_horizontal: int,
        paste_im_type: str):
        """
        This function takes a page image (PIL.Image), the region where the structure depictions are supposed to be pasted into and the amount of images per row 
        (horizontal_image) and per column (vertical_images).
        It pastes the given amount of depictions into the region and returns the image and list of tuples that 
        contains the path(s), names, original shapes, modified shapes and the paste coordinates (min_y, min_x)
        of the pasted images for the annotation creation. If binarise_half is set True, 50% of the pasted images 
        are binarised

        Args:
            image (Image): Page image where the structure depictions etc are supposed to be pasted.
            region (Tuple[int]): bounding box of region where images are supposed to be pasted
            images_vertical (int): number of images in vertical direction
            images_horizontal (int): number of images in horizontal direction
            paste_im_tpye (str): If true, half of the pasted images are binarised. Defaults to True.

        Returns:
            Image, Dict: Page image with pasted elements, annotation information
        """
        min_x, max_y, max_x, min_y = region
        #Define positions for pasting images
        x_diff = (max_x - min_x) / images_horizontal
        x_steps = [min_x + x_diff * x for x in range(images_horizontal)]
        y_diff = (max_y - min_y) / images_vertical
        y_steps = [min_y + y_diff * y for y in range(images_vertical)]
        # Define paste region coordinates
        pasted_element_annotation = []
        paste_regions = []
        for n in range(len(x_steps)):
            for m in range(len(y_steps)):
                paste_regions.append((int(x_steps[n]), int(x_steps[n] + x_diff), int(y_steps[m]), int(y_steps[m]  + y_diff)))
                
        for paste_region in paste_regions:
            min_x, max_x, min_y, max_y = paste_region
            # Make sure that the type of pasted image is treated adequately
            binarise_half = False
            if paste_im_type == 'structure':
                smiles = next(self.smiles_iterator)
                paste_im, paste_im_annotation = self.generate_structure_and_annotation(smiles, label=random.choice([True, False]))    
            elif paste_im_type == 'scheme':
                paste_im, paste_im_annotation = self.create_reaction_scheme()
            elif paste_im_type == 'random':
                paste_im = Image.open(next(self.random_image_iterator))
                paste_im = paste_im.rotate(random.choice([0, 90, 180, 270]))
                binarise_half = True
                paste_im_annotation = False

            paste_im_shape = paste_im.size
            if max_x - min_x > 0 and max_y - min_y > 0:
                # Keep image aspect while making sure that it can be pasted in the paste region
                for n in [1.0, 0.9, 0.8]:
                    #if paste_im.size[0] > max_x-min_x:
                    x_factor = n * (max_x-min_x)/paste_im.size[0]
                    if x_factor * paste_im.size[1] <= (max_y-min_y):
                        modified_im_shape = (int(n * max_x - n * min_x), int(x_factor * paste_im.size[1]))
                        break
                    # Try fitting pasted image to paste region height while keeping aspect
                    #elif paste_im.size[0] > max_y-min_y:
                    y_factor = n * (max_y - min_y) / paste_im.size[1]
                    if y_factor * paste_im.size[0] <= (max_x-min_x):
                        modified_im_shape = (int(y_factor * paste_im.size[0]), int(n * max_y - n * min_y))
                        break
                            
                else:
                    # TODO: This distorts the image
                    modified_im_shape = (max_x-min_x, max_y-min_y)
                paste_im = paste_im.resize(modified_im_shape)
                # Binarize half of the images if desired
                if binarise_half:
                    if random.choice([True, False]):
                        fn = lambda x : 255 if x > 150 else 0
                        paste_im = paste_im.convert('L').point(fn, mode='1')
                image.paste(paste_im, (min_x,min_y))
            # Modify annotations according to resized pasted image
            modified_chem_annotations = self.modify_annotations_PLN(paste_im_annotation, paste_im_shape, modified_im_shape, (min_x,min_y))
            pasted_element_annotation += modified_chem_annotations
        return image, pasted_element_annotation


    def CreateChemPage(
        self,
        ):
        place_to_paste = False
        while not place_to_paste:
            page_annotation = next(self.PLN_page_annotation_iterator)
            # Open PubLayNet Page Image
            image = Image.open(page_annotation['file_name'])
            image = deepcopy(np.asarray(image))
            modified_annotations = []
            figure_regions = []
            
            # Make sure only pages that contain a figure or a table are processed.
            category_IDs =  [annotation['category_id'] - 1 for annotation in page_annotation['annotations']]
            found_categories = [self.PLN_annotations['categories'][category_ID]['name'] 
                                for category_ID in category_IDs]
            if 'figure' in found_categories:
                place_to_paste = True
            #if 'table' in found_categories:
            #    place_to_paste = True
        # Replace figures with white space
        for annotation in page_annotation['annotations']:
            category_ID = annotation['category_id'] - 1
            category = self.PLN_annotations['categories'][category_ID]['name']
            # Leave every element that is not a figure untouched
            if category not in ['figure', 'list']:
                modified_annotations.append(annotation)
            else:
                # Delete Figures in images
                polygon = annotation['segmentation'][0]
                polygon_y = [int(polygon[n]) for n in range(len(polygon)) if n%2!=0]
                polygon_x = [int(polygon[n]) for n in range(len(polygon)) if n%2==0]
                figure_regions.append((min(polygon_x),max(polygon_y),max(polygon_x),min(polygon_y)))
                #figure_regions.append(annotation['bbox'])
                for x in range(min(polygon_x), max(polygon_x)):
                    for y in range(min(polygon_y), max(polygon_y)):
                        image[y,x] = [255,255,255]

        #  Paste new elements
        image = Image.fromarray(image)
        for region in figure_regions:
            
            
            # Region boundaries
            region = [round(coord) for coord in region]
            min_x, max_y, max_x, min_y = region
            # Don't paste reaction schemes into tiny regions
            if max_x-min_x > 200 and max_y-min_y > 200:
                paste_im_type = random.choice(['structure', 'scheme', 'random'])
            else:
                paste_im_type = random.choice(['structure', 'random'])
            # Determine how many chemical structures should be placed in the region.
            if paste_im_type == 'structure':
                images_vertical, images_horizontal = self.determine_images_per_region(region)
            else:
                # We don't want a grid of reaction schemes or random images.
                images_vertical, images_horizontal = (1, 1)


                #image, modified_annotations = self.paste_images(image, region, images_vertical, images_horizontal)

            image, pasted_structure_info = self.paste_images(image, region, images_vertical, images_horizontal, paste_im_type=paste_im_type)
            modified_annotations += pasted_structure_info
        modified_annotations = self.make_img_metadata_dict_from_PLN_annotations(page_annotation['file_name'], modified_annotations, self.PLN_annotations['categories'])
        return image, modified_annotations

    def make_img_metadata_dict_from_PLN_annotations(
        self,
        image_name: str,
        PLN_annotation_subdicts: List[Dict], 
        categories: List[Dict]) -> Dict:
        '''This function takes the name of an image, the directory, the coordinates of annotated polygon region and the list
        of category dicts as given in the PLN annotations and returns the VIA _img_metadata_subdict for this image.'''
        metadata_dict = {}
        metadata_dict['regions'] = []
        #image_name = os.path.split(image_path)[-1]
        metadata_dict['filename'] = image_name
        #metadata_dict['size'] = int(os.stat(image_path).st_size)
        #metadata_dict['shape'] = image.shape[:2]
        for annotation in PLN_annotation_subdicts:
            polygon = annotation['segmentation'][0]
            polygon = [(polygon[n], polygon[n-1]) for n in range(len(polygon)) if n%2!=0]
            category_ID = annotation['category_id'] - 1
            category = categories[category_ID]['name']
            # Add dict for region which contains annotated entity
            metadata_dict['regions'].append(self.make_region_dict(category, polygon))
        return metadata_dict

# def coordination(filename: str, image_path: str, output_path: str, annotations: List[Dict], structure_dir: str, structure_with_curved_arrows_dir: str, reaction_scheme_dir: str, random_image_dir: str, categories: Dict):
# 	'''This function just wraps up the replacement of figure and the creation of the metadata_dicts per figure to enable multiprocessing.'''
# 	ann = replace_elements(filename, image_path, output_path, annotations, structure_dir, structure_with_curved_arrows_dir, reaction_scheme_dir, random_image_dir, categories)
# 	if not ann:
# 		return False
# 	else:
# 		metadata_dict = make_img_metadata_dict(filename, image_path, ann, categories)
# 		return metadata_dict

# def main(
# 	reaction_scheme_dir: str = sys.argv[1], 
# 	structure_dir: str = sys.argv[2], 
# 	curved_arrow_structure_dir: str = sys.argv[3], 
# 	random_image_dir: str = sys.argv[4], 
# 	PubLayNet_image_dir: str = sys.argv[5], 
# 	output_dir: str = sys.argv[6], 
# 	PubLayNet_json: str = sys.argv[7]):
# 	'''This script takes a directory with images of chemical structure depictions with 
# 	added chemical label, a directory containing PubLayNet images, an output directory and
# 	a json file that contains the PubLayNet annotations for the images.
# 	It deletes all regions that are annotated as 'Figure' in the original dataset and replaces
# 	them with images of chemical structure depictions with ID labels. 
# 	The modified images as well as a json file containing the annotations (VIA compatible) are saved
# 	in the output directory.'''

# 	# Define relevant paths, load annotations and print usage note if the input is invalid.
# 	for directory in sys.argv[1:]:
# 		if os.path.exists(directory) and len(sys.argv) == 8:
# 			pass
# 		else:
# 			print('INVALID INPUT \n Problem with: ' + directory + '\nUsage: ' + sys.argv[0] + ' reaction_scheme_dir structure_dir curved_arrow_structure_dir random_image_dir PubLayNet_image_dir output_dir PubLayNet_json')
# 			return

# 	reaction_scheme_dir = os.path.abspath(reaction_scheme_dir)
# 	structure_dir = os.path.abspath(structure_dir)
# 	curved_arrow_structure_dir = os.path.abspath(curved_arrow_structure_dir)
# 	random_image_dir = os.path.abspath(random_image_dir)
# 	output_dir = os.path.abspath(output_dir)
# 	PubLayNet_image_dir = os.path.abspath(PubLayNet_image_dir)
# 	PLN_annotations = load_PLN_annotations(PubLayNet_json, PubLayNet_image_dir)

# 	starmap_iterable = []
# 	for page in PLN_annotations.keys():
# 		if page != 'categories':
# 			path, filename = os.path.split(os.path.normpath(PLN_annotations[page]['file_name']))
# 			categories = PLN_annotations['categories']
# 			categories.append({'supercategory': '', 'id': 6, 'name': 'chemical_structure'})
# 			categories.append({'supercategory': '', 'id': 7, 'name': 'chemical_ID'})
# 			categories.append({'supercategory': '', 'id': 8, 'name': 'arrow'})
# 			categories.append({'supercategory': '', 'id': 9, 'name': 'R_group_label'})
# 			categories.append({'supercategory': '', 'id': 10, 'name': 'reaction_condition_label'})
# 			categories.append({'supercategory': '', 'id': 11, 'name': 'chemical_structure_with_curved_arrows'})
# 			starmap_iterable.append((filename, path, output_dir, PLN_annotations[page]['annotations'], structure_dir, curved_arrow_structure_dir, reaction_scheme_dir, random_image_dir, categories))

# 	with Pool(20) as p:
# 		metadata_dicts = p.starmap(coordination, starmap_iterable)
    

# 	metadata_dicts = [metadata_dict for metadata_dict in metadata_dicts if metadata_dict]
# 	via_json = make_VIA_dict(metadata_dicts)
# 	with open(os.path.join(output_dir, 'annotations.json'), 'w') as output:
# 		json.dump(via_json, output)


#DUPLICATES

