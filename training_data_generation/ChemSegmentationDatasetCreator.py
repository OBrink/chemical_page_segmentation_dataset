# Author: Henning Otto Brinkhaus
# Friedrich-Schiller Universität Jena


import os
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageStat
import random
from copy import deepcopy
from itertools import cycle
import logging
from imantics import Polygons
from skimage import morphology
from polygon_bounding_box_determination import get_polygon_coordinates
import RanDepict

from annotation_utils import (load_PLN_annotations,
                              make_region_dict,
                              make_img_metadata_dict_from_PLN_annotations,
                              modify_annotations_PLN)


class ChemSegmentationDatasetCreator:
    """
    The ChemSegmentationDatasetCreator class contains everything needed
    to generate a chemical page segmentation dataset based on PubLayNet.
    """

    def __init__(
        self, smiles_list, load_PLN=True, PLN_annotation_number: int = False
    ):
        self.depictor = RanDepict.RandomDepictor()
        self.smiles_iterator = cycle(smiles_list)
        # Random images to be pasted (eg. COCO images) for diversification
        self.random_image_dir = os.path.join(
            os.path.split(__file__)[0],
            "./random_images/")
        self.random_images = cycle(
            [
                os.path.join(self.random_image_dir, im)
                for im in os.listdir(self.random_image_dir)
            ]
        )
        # PubLayNet images
        self.PLN_dir = os.path.join(
            os.path.split(__file__)[0], "publaynet/")
        # Load PLN annotations and add custom categories; may take 1-2 min
        if load_PLN:
            self.PLN_annotations = load_PLN_annotations(PLN_dir=self.PLN_dir)
            categories = self.PLN_annotations["categories"]
            categories.append(
                {"supercategory": "", "id": 6, "name": "chemical_structure"}
            )
            categories.append({"supercategory": "", "id": 7, "name": "chemical_label"})
            categories.append({"supercategory": "", "id": 8, "name": "arrow"})

            if PLN_annotation_number:
                keys = list(self.PLN_annotations.keys())
                keys = keys[:PLN_annotation_number]
                self.annotation_iterator = cycle(
                    [
                        self.PLN_annotations[page]
                        for page in keys
                        if page != "categories"
                    ]
                )
            else:
                keys = list(self.PLN_annotations.keys())
                self.annotation_iterator = cycle(
                    [
                        self.PLN_annotations[page]
                        for page in keys
                        if page != "categories"
                    ]
                )
            # Just keep categories saved
            categories = self.PLN_annotations["categories"]
            self.PLN_annotations = {"categories": categories}
        else:
            self.PLN_annotations = None
            self.annotation_iterator = None

    def create_training_batch(self, batch_size: int) -> Tuple[List, List]:
        """
        Given a batch size, this function generates batch_size training images
        with the corresponding annotations.

        Args:
            batch_size (int): number of images and annotations

        Returns:
            Tuple[List, List]: List of images, list of annotation dicts
        """
        content_generation_functions = [
            self.create_chemical_page,
            self.create_reaction_scheme,
            self.create_grid_of_chemical_structures,
            self.get_random_COCO_image
        ]
        images = []
        annotations = []
        for _ in range(batch_size):
            content_function = random.choice(content_generation_functions)
            image, annotation = content_function()
            images.append(image)
            annotations.append(annotation)
        return images, annotations

    def create_chemical_page(self,):
        """
        This function returns a PubLayNet page with inserted chemical elements and the
        regions dict which includes the positions/annotations.

        Returns:
            PIL.Image: Modified PubLayNet image
            Dict: Annotations of elements in the image
        """
        place_to_paste = False
        while not place_to_paste:
            page_annotation = next(self.annotation_iterator)
            # Open PubLayNet Page Image
            image = Image.open(page_annotation["file_name"])
            image = deepcopy(np.asarray(image))
            modified_annotations = []
            figure_regions = []

            # Make sure only pages that contain a figure or a table are processed.
            category_IDs = [
                annotation["category_id"] - 1
                for annotation in page_annotation["annotations"]
            ]
            found_categories = [
                self.PLN_annotations["categories"][category_ID]["name"]
                for category_ID in category_IDs
            ]
            if "figure" in found_categories:
                place_to_paste = True
        # Replace figures with white space
        for annotation in page_annotation["annotations"]:
            category_ID = annotation["category_id"] - 1
            category = self.PLN_annotations["categories"][category_ID]["name"]
            # Leave every element that is not a figure/list untouched
            if category not in ["figure", "list"]:
                modified_annotations.append(annotation)
            else:
                # Delete Figures in images
                polygon = annotation["segmentation"][0]
                polygon_y = [int(polygon[n]) for n in range(len(polygon)) if n % 2 != 0]
                polygon_x = [int(polygon[n]) for n in range(len(polygon)) if n % 2 == 0]
                figure_regions.append(
                    (min(polygon_x), max(polygon_y), max(polygon_x), min(polygon_y))
                )
                for x in range(min(polygon_x) - 2, max(polygon_x) + 2):
                    for y in range(min(polygon_y) - 2, max(polygon_y) + 2):
                        image[y, x] = [255, 255, 255]

        #  Paste new elements
        image = Image.fromarray(image)
        for region in figure_regions:

            # Region boundaries
            region = [round(coord) for coord in region]
            min_x, max_y, max_x, min_y = region
            # Don't paste reaction schemes into tiny regions
            if max_x - min_x > 200 and max_y - min_y > 200:
                paste_im_type = random.choice(["structure", "scheme", "random"])
            else:
                paste_im_type = random.choice(["structure", "random"])
            # Determine how many chemical structures should be placed in the region.
            if paste_im_type == "structure":
                images_vertical, images_horizontal = self.determine_images_per_region(
                    region
                )
            else:
                # We don't want a grid of reaction schemes or random images.
                images_vertical, images_horizontal = (1, 1)

            image, pasted_structure_info = self.paste_chemical_contents(
                image,
                region,
                images_vertical,
                images_horizontal,
                paste_im_type=paste_im_type,
            )
            modified_annotations += pasted_structure_info

        modified_annotations = make_img_metadata_dict_from_PLN_annotations(
            page_annotation["file_name"],
            modified_annotations,
            self.PLN_annotations["categories"],
        )
        return image, modified_annotations['regions']

    def create_grid_of_chemical_structures(self):
        """
        This function returns a grid of chemical structures of random size with
        the corresponding annotated region dictionaries.

        Returns:
            Image, Annotations
        """
        shape = (random.choice([200, 650]),
                 random.choice([200, 650]))
        region = (0, shape[1], shape[0], 0)
        images_vertical, images_horizontal = self.determine_images_per_region(region)
        image = Image.new("RGBA", shape, (255, 255, 255, 255))
        image, pasted_structure_info = self.paste_chemical_contents(
                image,
                region,
                images_vertical,
                images_horizontal,
                paste_im_type="structure",
            )
        annotations = make_img_metadata_dict_from_PLN_annotations(
            "test",
            pasted_structure_info,
            self.PLN_annotations["categories"],
        )
        return image.convert("RGB"), annotations["regions"]

    def get_random_COCO_image(self):
        """
        This function returns a random non-chemical image. In half of the cases,
        it is returned as a binarised image (but still as an RGB image).

        ___
        Returns:

        PIL.Image (mode: "RGB")
        Empty list (where all other functions would return a list of annotation dicts)
        """
        im = Image.open(next(self.random_images))
        im = im.rotate(random.choice([0, 90, 180, 270]))
        if random.choice([True, False]):
            im = im.convert("L")
            im = im.point(lambda x: 255 if x > 150 else 0,
                          mode="1").convert("RGB")
        im = Image.fromarray(np.asarray(im))
        return im, []

    def generate_multiple_structures_with_annotation(
            self,
            number: int,
    ) -> Tuple[List, List]:
        """
        Given a number of desired structures, this function returns a list of
        PIL.Image objects that hold the chemical structure depictions and a list of
        annotations that hold information about the annotated region

        Args:
            number (int): desired number of structure depictions

        Returns:
            Tuple[List, List]: structures, annotations
        """
        structure_images: List = []
        annotated_regions: List = []
        for _ in range(number):
            smiles = next(self.smiles_iterator)
            side_len = random.choice(range(200, 400))
            structure_image, annotation = self.generate_structure_with_annotation(
                smiles, shape=(side_len, side_len),
                label=random.choice([True, False])
            )
            structure_images.append(structure_image)
            annotated_regions.append(annotation)
        return structure_images, annotated_regions

    def generate_structure_with_annotation(
        self,
        smiles: str,
        shape: Tuple[int] = (200, 200),
        label: bool = False,
        arrows: bool = False,
    ):
        """
        Generate a chemical structure depiction and the metadata dict with
        the polygon bounding box.

        Args:
            smiles (str): SMILES representation of chemical compound
            label (bool, optional): Set to True if additional labels around
                                    the structure are desired.
                                    Defaults to False.
            arrows (bool, optional): Set to True if curved arrows in the
                                     structure are desired. Defaults to False.

        Returns:
            output_image (PIL.Image)
            metadata_dict (Dict): dictionary containing the coordinates that
                                  are necessary to define the polygon bounding
                                  box around the chemical structure depiction.
        """
        # Depict chemical structure
        image = self.depictor.random_depiction(smiles, shape)
        image = Image.fromarray(image)
        # Add some padding
        image = self.pad_image(image, factor=1.8)
        # Get coordinates of polygon around chemical structure
        polygon = get_polygon_coordinates(
            image_array=np.asarray(image), debug=False
        )
        if type(polygon) == np.ndarray:
            # Add a chemical ID label to the image
            if label:
                if random.choice([True, False]):
                    image, label_bounding_box = self.add_chemical_ID(image,
                                                                     polygon,
                                                                     False)
                else:
                    image, R_group_regions = self.insert_labels(
                        image.convert("RGB"), 1, "r_group")
                    if len(R_group_regions) > 0:
                        xcoords = R_group_regions[0]["shape_attributes"]["all_points_x"]
                        ycoords = R_group_regions[0]["shape_attributes"]["all_points_y"]
                        label_bounding_box = [[xcoords[n], ycoords[n]]
                                              for n in range(len(xcoords))]
                    else:
                        label_bounding_box = False
            else:
                label_bounding_box = None
            # The image should not be bigger than necessary.
            (image, polygon, label_bounding_box) = self.delete_whitespace(
                image, polygon, label_bounding_box
            )
            # In 20 percent of cases: Make structure image coloured to get
            # more diversity in the training set (colours should be
            # irrelevant for predictions)
            if random.choice(range(5)) == 0:
                image = self.modify_colours(image, blacken=False)
            elif random.choice(range(5)) in [1, 2]:
                image = self.modify_colours(image, blacken=True)

            # Add curved arrows in the structure
            if arrows:
                arrow_dir = os.path.abspath("./arrows/arrow_images")
                image = self.add_arrows_to_structure(image, arrow_dir, polygon)

            region_annotations = []
            region_dict = make_region_dict("chemical_structure", polygon, smiles)
            region_annotations.append(region_dict)
            if label_bounding_box:
                label_bounding_box = [(node[1], node[0]) for node in label_bounding_box]
                region_dict = make_region_dict("chemical_label", label_bounding_box)
                region_annotations.append(region_dict)

            return image, region_annotations
        else:
            image, region_annotations = self.generate_structure_with_annotation(
                smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                shape=shape,
                label=label,
                arrows=arrows
            )
            return image, region_annotations

    def pad_image(self, pil_image: Image, factor: float):
        """
        This function takes a Pillow Image and adds n% padding on every
        side. It returns the padded Pillow Image

        Args:
            pil_image (Image): Input image
            factor (float): factor for increasing output image size;
                            if factor==1.5, a 100x100 input image will
                            be returned as a 150x150 padded image

        Returns:
            Image: padded image
        """
        original_size = pil_image.size
        new_size = (int(original_size[0] * factor), int(original_size[1] * factor))
        new_im = Image.new("L", new_size, color="white")
        new_im.paste(
            pil_image,
            (
                int((new_size[0] - original_size[0]) / 2),
                int((new_size[1] - original_size[1]) / 2),
            ),
        )
        return new_im

    def get_random_label_position(
        self,
        width: int,
        height: int,
        x_coordinates: List[int],
        y_coordinates: List[int],
    ) -> Tuple:
        """
        Given the coordinates of the polygon around the chemical structure and
        the image shape, this function determines a random position for the
        label around the chemical struture.
        ___

        Args:
            width (int): image width
            height (int): image height
            x_coordinates (List[int]): polygon x coodinates (structure region)
            y_coordinates (List[int]): polygon y coodinates (structure region)

        Returns:
            Tuple: (int x_coordinate, int y_coordinate) or (None, None)
                   if it cannot place the label without an overlap.
        """
        x_y_limitation = random.choice([True, False])
        if x_y_limitation:
            y_range = list(range(50, int(min(y_coordinates) - 60)))
            y_range += list(range(int(max(y_coordinates) + 20), height - 50))
            x_range = range(50, width - 50)
            if len(y_range) == 0 or len(x_range) == 0:
                x_y_limitation = False
        if not x_y_limitation:
            y_range = list(range(50, height - 50))
            x_range = list(range(50, int(min(x_coordinates) - 60)))
            x_range += list(range(int(max(x_coordinates) + 20), width - 50))
            if len(y_range) == 0 or len(x_range) == 0:
                x_y_limitation = True
        if x_y_limitation:
            y_range = list(range(50, int(min(y_coordinates) - 60)))
            y_range += list(range(int(max(y_coordinates) + 20), height - 50))
            x_range = range(50, width - 50)
        if len(y_range) > 0 and len(x_range) > 0:
            y_pos = random.choice(y_range)
            x_pos = random.choice(x_range)
            return x_pos, y_pos
        else:
            return None, None

    def add_chemical_ID(
        self, image: Image, chemical_structure_polygon: np.ndarray, debug: bool = False
    ) -> Tuple:
        """
        This function takes a PIL Image and the coordinates of the region
        that contains a chemical structure diagram and adds random text that
        looks like a chemical ID label around the structure.
        It returns the modified image and the coordinates of the bounding box
        of the added text. If it is not possible to place the label around the
        structure without an overlap, it returns None

        Args:
            image (Image): Input image
            chemical_structure_polygon (np.ndarray): [[x0, y0], [x1, y1],...]
            debug (bool, optional): Activates visualisation. Defaults to False.

        Returns:
            Tuple: image, label_bounding_box
        """
        im = image.convert("RGB")
        y_coordinates = [node[0] for node in chemical_structure_polygon]
        x_coordinates = [node[1] for node in chemical_structure_polygon]

        # Choose random font
        font_dir = os.path.join(
            os.path.split(__file__)[0],
            "fonts/")
        fonts = os.listdir(font_dir)
        font_sizes = range(16, 40)

        # Define random position for text element around the chemical
        # structure (if it can be placed around it)
        width, height = im.size
        x_pos, y_pos = self.get_random_label_position(
            width, height, x_coordinates, y_coordinates
        )
        if x_pos is None:
            return im, None
        # Choose random font size
        size = random.choice(font_sizes)
        label_text = self.depictor.ID_label_text()
        try:
            font = ImageFont.truetype(
                str(os.path.join(font_dir, random.choice(fonts))), size=size
            )
        except OSError:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(im, "RGBA")
        # left, up, right, low (x0, y0, x1, y1)
        bb = draw.textbbox((x_pos, y_pos), label_text, font=font)
        # upper left, lower left, lower right, upper right
        label_bounding_box = [
            [bb[0] - 1, bb[1] - 1],
            [bb[0] - 1, bb[3] + 1],
            [bb[2] + 1, bb[3] + 1],
            [bb[2] + 1, bb[1] - 1],
        ]
        # Add text element to image
        draw = ImageDraw.Draw(im, "RGBA")
        draw.text((x_pos, y_pos), label_text, font=font, fill=(0, 0, 0, 255))
        # For debugging: Show the bounding boxes around structure and label
        if debug:
            polygon = [(node[1], node[0]) for node in chemical_structure_polygon]
            mod_label_bounding_box = [(node[0], node[1]) for node in label_bounding_box]
            draw.polygon(mod_label_bounding_box, fill=(255, 0, 0, 50))
            draw.polygon(polygon, fill=(0, 255, 0, 50))
            im.show()
        return im, label_bounding_box

    def pick_random_colour(self, black=False) -> Tuple[int]:
        """
        This function returns a random tuple with a colour for an RGBA image
        (eg. (123, 234, 322, 255))

        Args:
            black (bool, optional): If True, the colour will be black-ish
                                    Defaults to False.

        Returns:
            Tuple[int]: (R, G, B, A) where A=255
        """
        if not black:
            for _ in range(20):
                new_colour = (
                    random.choice(range(255)),
                    random.choice(range(255)),
                    random.choice(range(255)),
                    255,
                )
                if sum(new_colour[:3]) < 550:
                    break
            else:
                new_colour = (0, 0, 0, 255)
        else:
            num = random.choice(range(30))
            new_colour = (num, num, num, 255)
        return new_colour

    def modify_colours(self, image: Image, blacken=False) -> Image:
        """
        This function takes a Pillow Image, makes white pixels transparent,
        gives every other pixel a given new colour and returns the Image.

        Args:
            image (Image): input image
            blacken (bool, optional): If True, non-white pixels are going
                                      to be replaced with black-ish pixels
                                      instead of coloured pixels.
                                      Defaults to False.

        Returns:
            Image: Image with modified colours
        """
        image = image.convert("RGBA")
        datas = image.getdata()
        newData = []
        if blacken:
            new_colour = self.pick_random_colour(black=True)
        else:
            new_colour = self.pick_random_colour()
        for ch in datas:
            if not blacken and ch[0] > 230 and ch[1] > 230 and ch[2] > 230:
                newData.append((ch[0], ch[1], ch[2], 0))
            elif ch[0] < 230 and ch[1] < 230 and ch[2] < 230:
                newData.append(new_colour)
            else:
                newData.append(ch)
        image.putdata(newData)
        return image.convert("RGB")

    def adapt_coordinates(
        self,
        polygon: np.ndarray,
        label_bounding_box: np.ndarray,
        crop_indices: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function adapts the annotation coordinates after white space
        on the edges has been deleted.
        It also corrects coordinates outside of the image

        Args:
            polygon (np.ndarray): annotated region
                                  [(y0, x0), (y1, x1)...]
            label_bounding_box (np.ndarray): label bounding box
            crop_indices (Tuple[int, int, int, int])

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        first_rel_row, last_rel_row, first_rel_col, last_rel_col = crop_indices
        # Adapt chemical structure polygon coordinates
        for node in polygon:
            node[0] = node[0] - first_rel_row
            if node[0] < 0:
                node[0] = 0
            if node[0] > last_rel_row - first_rel_row:
                node[0] = last_rel_row - first_rel_row
            node[1] = node[1] - first_rel_col
            if node[1] < 0:
                node[1] = 0
            if node[1] > last_rel_col - first_rel_col:
                node[1] = last_rel_col - first_rel_col
        # Do the same for the chemical ID label
        if label_bounding_box:
            for node in label_bounding_box:
                node[0] = node[0] - first_rel_col
                if node[0] < 0:
                    node[0] = 0
                if node[0] > last_rel_col - first_rel_col:
                    node[0] = last_rel_col - first_rel_col
                node[1] = node[1] - first_rel_row
                if node[1] < 0:
                    node[1] = 0
                if node[1] > last_rel_row - first_rel_row:
                    node[1] = last_rel_row - first_rel_row
        return polygon, label_bounding_box

    def delete_whitespace(
        self, image: Image, polygon: np.ndarray, label_bbox: np.ndarray
    ) -> Image:
        """
        This function takes an image, the polygon coordinates around the
        chemical structure and the label_bounding_box and makes the image
        smaller if there is unnecessary whitespace at the edges. The
        polygon/bounding box coordinates are changed accordingly.

        Args:
            image (Image): Input image
            polygon (np.ndarray): region annotation
                                  [(y0, x0), (y1, x1)...]
            label_bounding_box (np.ndarray): bounding box of label

        Returns:
            Image
        """
        image = image.convert("RGB")
        image = np.asarray(image)
        y, x, z = image.shape
        first_rel_row = 0
        last_rel_row = y - 1
        first_rel_col = 0
        last_rel_col = x - 1
        # Check for non-white rows
        for n in np.arange(0, y, 1):
            if image[n].sum() != x * z * 255:
                first_rel_row = n - 1
                break
        for n in np.arange(y - 1, 0, -1):
            if image[n].sum() != x * z * 255:
                last_rel_row = n + 1
                break
        transposed_image = np.swapaxes(image, 0, 1)

        # Check for non-white columns
        for n in np.arange(0, x, 1):
            if transposed_image[n].sum() != y * z * 255:
                first_rel_col = n - 1
                break
        # Check for non-white columns
        for n in np.arange(x - 1, 0, -1):
            if transposed_image[n].sum() != y * z * 255:
                last_rel_col = n + 1
                break

        image = image[first_rel_row:last_rel_row, first_rel_col:last_rel_col, :]
        crop_indices = (first_rel_row, last_rel_row, first_rel_col, last_rel_col)
        polygon, label_bbox = self.adapt_coordinates(polygon, label_bbox, crop_indices)
        return Image.fromarray(image), polygon, label_bbox

    def is_valid_position(
        self,
        structure_paste_position: Tuple[int, int],
        image_shape: Tuple[int, int],
        structure_image_shape: Image,
    ) -> bool:
        """
        When a chemical structure depiction is pasted into the reaction scheme,
        it is necessary to make sure that the pasted structure
        is completely inside of the bigger image. This function determines whether or
        not this condition is fulfilled.

        Args:
            structure_paste_position (Tuple[int,int]): (x,y)
            image_shape (Tuple[int,int]): width, height
            structure_image_shape (Image): (width, height)

        Returns:
            bool: [description]
        """
        x, y = structure_paste_position
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

    def paste_images(
        self,
        image: Image,
        paste_positions: List[Tuple[int, int]],
        paste_images: List,
    ) -> Image:
        """
        This function takes an empty image (PIL.Image), a list of paste positions
        (Tuple[int,int] -> x,y of upper left corner), a list of images (PIL.Image) to
        paste into the empty image and a list of annotation Dicts. It pastes the images
        at the specified positions and adapts the annotation coordinates accordingly.
        It returns the modified image and the modified annotations.

        Args:
            image (Image): PIL.Image
            paste_positions (List[Tuple[int, int]]): [(x,y)]
            paste_images (List): List of images
            annotations (List[Dict]): [description]

        Returns:
            Image: The input image with all pasted elements in it at the given positions
        """
        for n in range(len(paste_positions)):
            if self.is_valid_position(
                paste_positions[n], image.size, paste_images[n].size
            ):
                image.paste(paste_images[n], paste_positions[n])
            else:
                # Delete previously pasted arrow that points to a structure outside
                # of the image
                image.paste(
                    Image.new("RGBA", paste_images[n - 1].size), paste_positions[n - 1]
                )
        return image

    def modify_annotations(
        self,
        original_regions: List[List[Dict]],
        paste_anchors: List[Tuple[int, int]]
    ) -> List[Dict]:
        """
        This function takes the original annotated regions (VIA dict format) and the
        paste position (x, y) [upper left corner]. The coordinates of the regions are
        modified according to the position in the new image. It returns the modified
        annotated regions (same format as input but the list is flattened).

        Args:
            original_regions (List[List[Dict]]): region dicts
            paste_anchors (List[Tuple[int,int]]): [(x1, y1), ...]

        Returns:
            List[Dict]: modified regions
        """
        modified_regions = []
        for i in range(len(original_regions)):
            for n in range(len(original_regions[i])):
                x_coords = original_regions[i][n]["shape_attributes"]["all_points_x"]
                original_regions[i][n]["shape_attributes"]["all_points_x"] = [
                    x_coord + paste_anchors[i][0] for x_coord in x_coords
                ]
                y_coords = original_regions[i][n]["shape_attributes"]["all_points_y"]
                original_regions[i][n]["shape_attributes"]["all_points_y"] = [
                    y_coord + paste_anchors[i][1] for y_coord in y_coords
                ]
                modified_regions.append(original_regions[i][n])
        return modified_regions

    def determine_arrow_annotations(self, images: List) -> List[Dict]:
        """
        This function takes a list of images, binarizes them, applies a dilation and
        determines the coordinates of the contours of the
        dilated object in the image. It returns a list of VIA-compatible region dicts.

        Args:
            images (List): List of PIL.Image

        Returns:
            List[Dict]: List of VIA-compatible region dicts that describe the arrow
            positions
        """
        region_dicts = []
        for image in images:
            # Binarize and apply dilation to image
            image = np.asarray(image)
            r, g, b, _ = image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]
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
            # swap x and y (numpy and PIL don't agree here. This way, it is consistent)
            annotation = [(x[1], x[0]) for x in annotation[0]]
            region_dicts.append([make_region_dict("arrow", annotation)])
        return region_dicts

    def is_diagonal_arrow(self, arrow_bbox: List[Tuple[int, int]]) -> bool:
        """
        This function takes an arrow bounding box of and arrow and checks whether or
        not the arrow is diagonal based on the aspect of the image.

        Args:
            arrow_bbox (List[Tuple[int,int]])

        Returns:
            bool
        """
        # Check x/y ratio
        if (arrow_bbox[2][0] - arrow_bbox[0][0]) / (
            arrow_bbox[0][1] - arrow_bbox[1][1]
        ) > 0.2:
            # Check y/x ratio
            if (arrow_bbox[2][0] - arrow_bbox[0][0]) / (
                arrow_bbox[0][1] - arrow_bbox[1][1]
            ) < 5:
                return True

    def insert_labels(
        self, image: Image, max_labels: int, label_type: str, arrow_bboxes: bool = False
    ):
        """
        This function takes a PIL Image, a maximal amount of labels to insert and a
        label type ('r_group' or 'reaction_condition'). If label type is
        'reaction_condition', arrow_bboxes [(x0,y0), [x1,y1)...] must also be given
        It generates random labels that look like the labels and inserts them at random
        positions (under the condition that there is
        no overlap with other objects in the image.). It returns the modified image
        and the region dicts for the labels.

        Args:
            image (Image): PIL.Image
            max_labels (int): Maximal amount of labels
            label_type (str): 'r_group' or 'reaction_condition
            arrow_bboxes (bool, optional): Defaults to False.

        Returns:
            Image: Modified Image
            List[Dict]: Annotated region dicts
        """
        n = 0
        region_dicts = []
        paste_positions = []
        for _ in range(500):
            invalid = False
            if n < max_labels:
                # Load random label
                if label_type == "r_group":
                    label_str = self.depictor.make_R_group_str()
                elif label_type == "reaction_condition":
                    label_str = self.depictor.reaction_condition_label_text()
                # Determine font properties and type
                fontsize = random.choice(range(16, 36))
                try:
                    font_dir = "./fonts/"
                    font = ImageFont.truetype(
                        str(
                            os.path.join(font_dir, random.choice(os.listdir(font_dir)))
                        ),
                        size=fontsize,
                    )
                except OSError:
                    font = ImageFont.load_default()
                # Determine size of created text
                draw = ImageDraw.Draw(image, "RGBA")
                bbox = draw.multiline_textbbox(
                    (0, 0), label_str, font
                )  # left, up, right, low
                # Check the resulting region in the reaction scheme if there already
                # is anything else
                try:
                    if label_type == "r_group":
                        paste_position = (
                            random.choice(range(image.size[0] - bbox[2])),
                            random.choice(range(image.size[1] - bbox[3])),
                        )
                        buffer = 30
                        paste_region = image.crop(
                            (
                                paste_position[0] - buffer,
                                paste_position[1] - buffer,
                                paste_position[0] + bbox[2] + buffer,
                                paste_position[1] + bbox[3] + buffer,
                            )
                        )  # left, up, right, low
                    elif label_type == "reaction_condition":
                        # take random arrow positions and place reaction condition
                        # label around it
                        arrow_bbox = random.choice(arrow_bboxes)
                        if self.is_diagonal_arrow(arrow_bbox):
                            paste_position = (
                                random.choice(
                                    range(arrow_bbox[1][0] + 20, arrow_bbox[3][0] - 20)
                                ),
                                random.choice(
                                    range(arrow_bbox[1][1] + 20, arrow_bbox[3][1] - 20)
                                ),
                            )
                        else:
                            paste_position = (
                                random.choice(
                                    range(arrow_bbox[1][0] + 1, arrow_bbox[3][0] - 30)
                                ),
                                random.choice(
                                    range(arrow_bbox[1][1] + 1, arrow_bbox[3][1] + 20)
                                ),
                            )
                        paste_region = image.crop(
                            (
                                paste_position[0] - 10,
                                paste_position[1] - 10,
                                paste_position[0] + bbox[2] + 10,
                                paste_position[1] + bbox[3] + 10,
                            )
                        )  # left, up, right, low
                except IndexError:
                    invalid = True
                try:
                    if not invalid:
                        # Check that background in paste region is completely white
                        mean = ImageStat.Stat(paste_region).mean
                        if sum(mean) / len(mean) == 255:
                            # Add text element to image if region in image is
                            # completely white
                            draw = ImageDraw.Draw(image, "RGBA")
                            draw.multiline_text(
                                paste_position,
                                label_str,
                                font=font,
                                fill=(0, 0, 0, 255),
                            )
                            paste_positions.append(paste_position)
                            n += 1
                            # lower left, upper left, upper right, lower right
                            # [(x0, y0), (x1, y1), ...]
                            text_bbox = [
                                (bbox[3], bbox[0]),
                                (bbox[1], bbox[0]),
                                (bbox[1], bbox[2]),
                                (bbox[3], bbox[2]),
                            ]
                            if label_type == "r_group":
                                region_dicts.append(
                                    [make_region_dict("chemical_label", text_bbox)]
                                )
                            if label_type == "reaction_condition":
                                region_dicts.append(
                                    [
                                        make_region_dict(
                                            "chemical_label", text_bbox
                                        )
                                    ]
                                )
                except ZeroDivisionError:
                    pass
            else:
                break

        region_dicts = self.modify_annotations(
            original_regions=region_dicts, paste_anchors=paste_positions
        )
        return image, region_dicts

    def get_bboxes(
        self, image_list: List, paste_position_list: List[Tuple[int, int]]
    ) -> List[Tuple]:
        """
        This function takes a list of PIL.Image objects and a list of their paste
        positions (x,y) [upper left corner] and
        returns a list of bounding boxes [(x0, y0), (x1, 1), ...]
        (lower left, upper left, upper right, lower right)

        Args:
            image_list (List): [description]
            paste_position_list (List[Tuple[int,int]]): [description]

        Returns:
            List[Tuple]: [description]
        """
        bboxes = []
        for i in range(len(image_list)):
            lower_left = (
                paste_position_list[i][0],
                paste_position_list[i][1] + image_list[i].size[1],
            )
            upper_left = paste_position_list[i]
            upper_right = (
                paste_position_list[i][0] + image_list[i].size[0],
                paste_position_list[i][1],
            )
            lower_right = (
                paste_position_list[i][0] + image_list[i].size[0],
                paste_position_list[i][1] + image_list[i].size[1],
            )
            bboxes.append((lower_left, upper_left, upper_right, lower_right))
        return bboxes

    def create_reaction_scheme(self,) -> Image:
        """
        This function creates an artificial reaction scheme (PIL.Image) and an
        annotation dictionary that contains information about the regions of all
        included elements
        

        Returns:
            Image: artificial reaction scheme
            Dict: annotated region information
        """
        scheme_generation_functions = [
            self.create_reaction_scheme_two_structures_horizontal,
            self.create_reaction_scheme_three_structures_horizontal,
            self.create_reaction_scheme_three_structures_vertical,
            self.create_reaction_scheme_five_structures,
            self.create_reaction_scheme_seven_structures,
        ]
        scheme_generation_function = random.choice(scheme_generation_functions)

        reaction_scheme, annotations = scheme_generation_function()
        return reaction_scheme, annotations

    def get_random_arrow_image(self, x: int, y: int) -> Image:
        """
        This function loads a random arrow image from the arrow image directory
        and returns it as an RGBA image (PIL.Image object).

        Args:
            x (int): desired arrow image width
            y (int): desired arrow image height

        Returns:
            Image: arrow image
        """
        arrow_dir = os.path.join(
            os.path.split(__file__)[0],
            "arrows/horizontal_arrows/")
        arrow_image_name = random.choice(os.listdir(arrow_dir))
        arrow_im = Image.open(os.path.join(arrow_dir, arrow_image_name))
        arrow_image = Image.new("RGBA", arrow_im.size, (255, 255, 255, 255))
        arrow_image = Image.alpha_composite(arrow_image, arrow_im)
        resize_method = random.choice([3, 4, 5])
        arrow_image = arrow_image.resize((x, y), resample=resize_method,)
        return arrow_image

    def get_image_sizes(
        self,
        images: List
    ) -> Tuple[List[int], List[int]]:
        """
        Given a list of PIL.Image objects, this function returns a list of image
        widths and a list of image heights.

        Args:
            images (List): List of PIL.Image objects

        Returns:
            Tuple[List[int], List[int]]: Image widths, image heights
        """
        image_sizes = [image.size for image in images]
        image_sizes_x = [size[0] for size in image_sizes]
        image_sizes_y = [size[1] for size in image_sizes]
        return image_sizes_x, image_sizes_y

    def create_reaction_scheme_two_structures_horizontal(
        self,
    ) -> Tuple:
        """
        This function generates a horizontal reaction scheme with two chemical
        structures and a reaction arrow between them.

        Returns:
            Tuple[Image, List]: Reaction scheme, annotation dicts
        """
        structures, annotations = self.generate_multiple_structures_with_annotation(2)
        structure_image_x, structure_image_y = self.get_image_sizes(structures)
        arrow_image = self.get_random_arrow_image(structure_image_x[0],
                                                  int(structure_image_y[1] / 8))
        size = (sum(structure_image_x) + arrow_image.size[0],
                max(structure_image_y),)
        image = Image.new("RGBA", size, (255, 255, 255, 255))
        paste_positions = [
            (0, int((max(structure_image_y) - structures[0].size[1]) / 2),),
            (structure_image_x[0] + arrow_image.size[0],
             int((max(structure_image_y) - structures[1].size[1]) / 2),),
        ]
        arrow_paste_positions = [
            (structure_image_x[0], int(structure_image_y[0] / 2))
        ]
        paste_positions += arrow_paste_positions
        paste_images = structures
        arrow_image_list = [arrow_image]
        paste_images += arrow_image_list
        image = self.paste_images(image, paste_positions, paste_images)
        annotations += self.determine_arrow_annotations(arrow_image_list)
        annotations = self.modify_annotations(annotations, paste_positions)
        if random.choice([True, False]):
            arrow_bboxes = self.get_bboxes(arrow_image_list, arrow_paste_positions)
            image, reaction_condition_regions = self.insert_labels(
                image, len(structures) - 1, "reaction_condition", arrow_bboxes
            )
            annotations += reaction_condition_regions
        return image.convert("RGB"), annotations

    def create_reaction_scheme_three_structures_horizontal(
        self,
    ) -> Tuple:
        """
        This function generates a horizontal reaction scheme with three chemical
        structures and two reaction arrow between them.

        Returns:
            Tuple[Image, List]: Reaction scheme, annotation dicts
        """
        structures, annotations = self.generate_multiple_structures_with_annotation(3)
        structure_image_x, structure_image_y = self.get_image_sizes(structures)
        arrow_image = self.get_random_arrow_image(structure_image_x[0],
                                                  int(structure_image_y[1] / 8))
        size = (sum(structure_image_x) + arrow_image.size[0] * 2,
                max(structure_image_y))
        image = Image.new("RGBA", size, (255, 255, 255, 255))
        paste_positions = [
            # left structure
            (0, int((max(structure_image_y) - structures[0].size[1]) / 2),),
            # middle structure
            (structure_image_x[0] + arrow_image.size[0],
             int((max(structure_image_y) - structures[1].size[1]) / 2),),
            # right structure
            (sum(structure_image_x[:2]) + arrow_image.size[0] * 2,
             int((max(structure_image_y) - structures[2].size[1]) / 2),),]
        arrow_paste_positions = [
            # left reaction arrow
            (structure_image_x[0],
             int(max(structure_image_y) / 2),),
            # right reaction arrow
            (sum(structure_image_x[:2]) + arrow_image.size[0],
             int(max(structure_image_y) / 2),), ]
        paste_positions += arrow_paste_positions
        paste_images = structures
        arrow_image_list = [
            arrow_image.rotate(random.choice([180, 360]),
                               expand=True,
                               fillcolor=(255, 255, 255, 255)),
            arrow_image,
        ]
        paste_images += arrow_image_list
        image = self.paste_images(image, paste_positions, paste_images)
        annotations += self.determine_arrow_annotations(arrow_image_list)
        annotations = self.modify_annotations(annotations, paste_positions)
        if random.choice([True, False]):
            arrow_bboxes = self.get_bboxes(arrow_image_list, arrow_paste_positions)
            image, reaction_condition_regions = self.insert_labels(
                image, len(structures) - 1, "reaction_condition", arrow_bboxes
            )
            annotations += reaction_condition_regions
        return image.convert("RGB"), annotations

    def create_reaction_scheme_three_structures_vertical(
        self,
    ) -> Tuple:
        """
        This function generates a vertical reaction scheme with three chemical
        structures and two reaction arrow between them.

        Returns:
            Tuple[Image, List]: Reaction scheme, annotation dicts
        """
        structures, annotations = self.generate_multiple_structures_with_annotation(3)
        structure_image_x, structure_image_y = self.get_image_sizes(structures)
        arrow_image = self.get_random_arrow_image(structure_image_x[0],
                                                  int(structure_image_y[1] / 8))
        size = (max(structure_image_x),
                sum(structure_image_y) + arrow_image.size[0] * 2,)
        image = Image.new("RGBA", size, (255, 255, 255, 255))
        paste_positions = [
            # upper structure
            (int((max(structure_image_x) - structures[0].size[0]) / 2), 0,),
            # middle structure
            (int((max(structure_image_x) - structures[1].size[0]) / 2),
                sum(structure_image_y[:1]) + arrow_image.size[0],),
            # lower structure
            (int((max(structure_image_x) - structures[2].size[0]) / 2),
                sum(structure_image_y[:2]) + arrow_image.size[0] * 2,), ]
        arrow_paste_positions = [
            (
                int(max(structure_image_x) / 2),
                structure_image_y[0],
            ),  # upper reaction arrow
            (
                int(max(structure_image_x) / 2),
                sum(structure_image_y[:2]) + arrow_image.size[0],
            ),
        ]  # lower arrow
        paste_positions += arrow_paste_positions
        paste_images = structures
        arrow_image_list = [
            arrow_image.rotate(random.choice([90, 270]),
                               expand=True,
                               fillcolor=(255, 255, 255, 255)),
            arrow_image.rotate(random.choice([90, 270]),
                               expand=True,
                               fillcolor=(255, 255, 255, 255)),
        ]
        paste_images += arrow_image_list
        image = self.paste_images(image, paste_positions, paste_images)
        annotations += self.determine_arrow_annotations(arrow_image_list)
        annotations = self.modify_annotations(annotations, paste_positions)
        if random.choice([True, False]):
            arrow_bboxes = self.get_bboxes(arrow_image_list, arrow_paste_positions)
            image, reaction_condition_regions = self.insert_labels(
                image, len(structures) - 1, "reaction_condition", arrow_bboxes
            )
            annotations += reaction_condition_regions
        return image.convert("RGB"), annotations

    def create_reaction_scheme_five_structures(
        self,
    ) -> Tuple:
        """
        This function generates a reaction scheme with five chemical
        structures and four reaction arrow between them.

        Returns:
            Tuple[Image, List]: Reaction scheme, annotation dicts
        """
        structures, annotations = self.generate_multiple_structures_with_annotation(2)
        structure_image_x, structure_image_y = self.get_image_sizes(structures)
        arrow_image = self.get_random_arrow_image(structure_image_x[0],
                                                  int(structure_image_y[1] / 8))
        init_scheme, init_annotations = self.create_reaction_scheme_three_structures_horizontal()
        size = (init_scheme.size[0],
                sum(structure_image_y) + arrow_image.size[0] * 2 + init_scheme.size[1],)
        image = Image.new("RGBA", size, (255, 255, 255, 255))
        image.paste(init_scheme, (0, arrow_image.size[0] + structure_image_y[0]))
        init_annotations = self.modify_annotations(
            [[reg_dict] for reg_dict in init_annotations],
            [(0, arrow_image.size[0] + structure_image_y[0])] * len(init_annotations))

        central_ann = self.get_central_structure_shape_annotation(init_annotations)
        central_x_coords = central_ann["all_points_x"]
        central_x = int(sum(central_x_coords) / len(central_x_coords))
        central_y_coords = central_ann["all_points_y"]
        cen = int(min(central_x_coords)+(max(central_x_coords)-min(central_x_coords))/2)
        paste_positions = [
            # upper structure
            (cen - int(structures[0].size[0] / 2),
             0,),
            # lower structure
            (cen - int(structures[1].size[0] / 2),
             max(central_y_coords) + arrow_image.size[0])]
        arrow_paste_positions = [
            (central_x, min(central_y_coords) - arrow_image.size[0]),  # upper arrow
            (central_x, max(central_y_coords))  # lower arrow
        ]
        paste_images = structures
        arrow_image_list = [
            arrow_image.rotate(random.choice([90, 270]),
                               expand=True,
                               fillcolor=(255, 255, 255, 255)),
            arrow_image.rotate(random.choice([90, 270]),
                               expand=True,
                               fillcolor=(255, 255, 255, 255)),
        ]
        annotations += self.determine_arrow_annotations(arrow_image_list)
        paste_positions += arrow_paste_positions
        paste_images += arrow_image_list
        image = self.paste_images(image, paste_positions, paste_images)
        annotations = self.modify_annotations(annotations, paste_positions)
        if random.choice([True, False]):
            arrow_bboxes = self.get_bboxes(arrow_image_list, arrow_paste_positions)
            image, reaction_condition_regions = self.insert_labels(
                image, len(structures) - 1, "reaction_condition", arrow_bboxes
            )
            annotations += reaction_condition_regions
        return image.convert("RGB"), annotations + init_annotations

    def create_reaction_scheme_seven_structures(
        self,
    ) -> Tuple:
        """
        This function generates a reaction scheme with seven chemical
        structures and six reaction arrow between them.

        Returns:
            Tuple[Image, List]: Reaction scheme, annotation dicts
        """
        structures, annotations = self.generate_multiple_structures_with_annotation(7)
        structure_image_x, structure_image_y = self.get_image_sizes(structures)
        arrow_image = self.get_random_arrow_image(structure_image_x[0],
                                                  int(structure_image_y[1] / 8))
        size = (sum([structure_image_x[n] for n in [0, 1, 2]])
                + arrow_image.size[0] * 2,
                sum([structure_image_y[n] for n in [1, 3, 4]])
                + arrow_image.size[0] * 2,)
        image = Image.new("RGBA", size, (255, 255, 255, 255))

        # Horizontal reaction flow
        paste_positions = [
            (0,
             int(structure_image_y[3] + arrow_image.size[0]
                 + 0.5 * structure_image_y[1] - 0.5 * structure_image_y[0]),
             ),  # left structure
            (sum(structure_image_x[:1]) + arrow_image.size[0] * 1,
             int(structure_image_y[3] + arrow_image.size[0]
                 + 0.5 * structure_image_y[1] - 0.5 * structure_image_y[1]),
             ),  # middle_structure
            (sum(structure_image_x[:2]) + arrow_image.size[0] * 2,
             int(structure_image_y[3] + arrow_image.size[0]
                 + 0.5 * structure_image_y[1] - 0.5 * structure_image_y[2]),
             ),  # right structure
        ]
        arrow_paste_positions = [
            (sum(structure_image_x[:1]),
             int(structure_image_y[3] + arrow_image.size[0]
                 + 0.5 * structure_image_y[1]),),  # left arrow
            (sum(structure_image_x[:2]) + arrow_image.size[0],
             int(structure_image_y[3] + arrow_image.size[0]
                 + 0.5 * structure_image_y[1]), ), ]  # right arrow

        paste_images = structures[:3]
        arrow_image_list = [
            arrow_image.rotate(random.choice([0, 180]),
                               expand=True, fillcolor=(255, 255, 255, 255)),
            arrow_image.rotate(random.choice([0, 180]),
                               expand=True, fillcolor=(255, 255, 255, 255)),
        ]

        # Add diagonal arrows and structures

        # upper left
        arrow_image_1 = arrow_image.rotate(random.choice([135, 315]),
                                           expand=True, fillcolor=(255, 255, 255, 255))
        arrow_paste_position_1 = (
            paste_positions[1][0] - arrow_image_1.size[0],
            paste_positions[1][1] - arrow_image_1.size[1],
        )
        structure_paste_position_1 = (
            arrow_paste_position_1[0] - structure_image_x[3],
            arrow_paste_position_1[1] - int(0.75 * structure_image_y[3]),
        )
        # upper right
        arrow_image_2 = arrow_image.rotate(random.choice([45, 225]),
                                           expand=True, fillcolor=(255, 255, 255, 255))
        arrow_paste_position_2 = (
            paste_positions[1][0] + structure_image_x[1],
            paste_positions[1][1] - arrow_image_2.size[1],
        )
        structure_paste_position_2 = (
            arrow_paste_position_2[0] + arrow_image_2.size[0],
            arrow_paste_position_2[1] - int(0.75 * structure_image_y[4]),
        )
        # lower right
        arrow_image_3 = arrow_image.rotate(random.choice([135, 315]),
                                           expand=True, fillcolor=(255, 255, 255, 255))

        arrow_paste_position_3 = (
            paste_positions[1][0] + structure_image_x[1],
            paste_positions[1][1] + structure_image_y[1],
        )
        structure_paste_position_3 = (
            arrow_paste_position_3[0] + arrow_image_3.size[0],
            arrow_paste_position_3[1]
            + arrow_image_3.size[1]
            - int(0.25 * structure_image_y[5]),
        )
        # lower left
        arrow_image_4 = arrow_image.rotate(random.choice([45, 225]),
                                           expand=True, fillcolor=(255, 255, 255, 255))

        arrow_paste_position_4 = (
            paste_positions[1][0] - arrow_image_4.size[0],
            paste_positions[1][1] + structure_image_y[1],
        )
        structure_paste_position_4 = (
            arrow_paste_position_4[0] - structure_image_x[6],
            arrow_paste_position_4[1]
            + arrow_image_4.size[1]
            - int(0.25 * structure_image_y[6]),
        )

        # Only add structures and arrows that point at them if the structures fit
        # in the image
        annotated_regions_copy = deepcopy(annotations)
        for n in range(1, 5):
            if self.is_valid_position(
                eval("structure_paste_position_" + str(n)),
                image.size,
                structures[2 + n].size, ):
                arrow_paste_positions.append(eval("arrow_paste_position_" + str(n)))
                arrow_image_list.append(eval("arrow_image_" + str(n)))
                paste_images.append(structures[2 + n])
                paste_positions.append(eval("structure_paste_position_" + str(n)))
            else:
                annotations.remove(annotated_regions_copy[2 + n])

        annotations += self.determine_arrow_annotations(arrow_image_list)
        paste_positions += arrow_paste_positions
        paste_images += arrow_image_list

        annotations = self.modify_annotations(
            annotations, paste_positions
        )
        image = self.paste_images(image, paste_positions, paste_images)
        if random.choice([True, False]):
            arrow_bboxes = self.get_bboxes(arrow_image_list, arrow_paste_positions)
            image, reaction_condition_regions = self.insert_labels(
                image, len(structures) - 1, "reaction_condition", arrow_bboxes
            )
            annotations += reaction_condition_regions
        return image.convert("RGB"), annotations

    def get_central_structure_shape_annotation(self, annotations: List[Dict]) -> Dict:
        """
        This function takes the annotations returned from a scheme with three structures
        and returns the shape attribute dict of the central structure.

        Args:
            annotations (List[Dict]): List of annotation dicts from scheme with three
                                      structures

        Returns:
            Dict: shape attribute dict of central structure
        """
        
        count = 0
        for ann in annotations:
            if ann["region_attributes"]["type"] == "chemical_structure":
                if count == 1:
                    return ann["shape_attributes"]
                else:
                    count += 1

    def determine_images_per_region(self, region: Tuple[int]) -> Tuple[int, int]:
        """
        This function takes the bounding box coordinates of a region and returns
        two integers which indicate how many chemical structure depictions should be
        added (int1: vertical, int2: horizontal) to a grid of chemical structures in
        that region.
        The returned values depend on the region size and a random influence.

        Args:
            region (Tuple[int]): paste region bounding box

        Returns:
            Tuple[int, int]: "rows", "columns"
        """
        min_x, max_y, max_x, min_y = region
        x_diff = max_x - min_x
        y_diff = max_y - min_y
        n = random.choice([50, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200])
        horizontal_int = round(x_diff / n)
        vertical_int = round(y_diff / n)
        if horizontal_int < 1:
            horizontal_int = 1
        if vertical_int < 1:
            vertical_int = 1
        return vertical_int, horizontal_int

    def paste_chemical_contents(
        self,
        image: Image,
        region: Tuple[int],
        images_vertical: int,
        images_horizontal: int,
        paste_im_type: str,
    ):
        """
        This function takes a page image (PIL.Image), the region where the structure
        depictions are supposed to be pasted into and the amount of images per row
        (horizontal_image) and per column (vertical_images).
        It pastes the given amount of depictions into the region and returns the image
        and list of tuples that contains the path(s), names, original shapes, modified
        shapes and the paste coordinates (min_y, min_x) of the pasted images for the
        annotation creation.

        Args:
            image (Image): Page image where the structure depictions etc are supposed
                           to be pasted.
            region (Tuple[int]): bounding box of region where images are supposed to be
                                 pasted
            images_vertical (int): number of images in vertical direction
            images_horizontal (int): number of images in horizontal direction
            paste_im_type (str): scheme, structure, random

        Returns:
            Image, Dict: Page image with pasted elements, annotation information
        """
        min_x, max_y, max_x, min_y = region
        # Define positions for pasting images
        x_diff = (max_x - min_x) / images_horizontal
        x_steps = [min_x + x_diff * x for x in range(images_horizontal)]
        y_diff = (max_y - min_y) / images_vertical
        y_steps = [min_y + y_diff * y for y in range(images_vertical)]
        # Define paste region coordinates
        pasted_element_annotation = []
        paste_regions = []
        for n in range(len(x_steps)):
            for m in range(len(y_steps)):
                # Leave 5 px buffer
                paste_regions.append(
                    (
                        int(x_steps[n]),
                        int(x_steps[n] + x_diff - 5),
                        int(y_steps[m]),
                        int(y_steps[m] + y_diff - 5),
                    )
                )

        for paste_region in paste_regions:
            min_x, max_x, min_y, max_y = paste_region
            # Make sure that the type of pasted image is treated adequately
            binarise_half = False
            if paste_im_type == "structure":
                smiles = next(self.smiles_iterator)
                paste_im, paste_im_annotation = self.generate_structure_with_annotation(
                    smiles, label=random.choice([True, False])
                )
            elif paste_im_type == "scheme":
                paste_im, paste_im_annotation = self.create_reaction_scheme()
            elif paste_im_type == "random":
                paste_im = Image.open(next(self.random_images))
                paste_im = paste_im.rotate(random.choice([0, 90, 180, 270]))
                binarise_half = True
                paste_im_annotation = False

            paste_im_shape = paste_im.size
            if max_x - min_x > 0 and max_y - min_y > 0:
                # Keep image aspect while making sure that it can be pasted in the
                # paste region
                for n in [1.0, 0.9, 0.8]:
                    # if paste_im.size[0] > max_x-min_x:
                    x_factor = n * (max_x - min_x) / paste_im.size[0]
                    if x_factor * paste_im.size[1] <= (max_y - min_y):
                        modified_im_shape = (
                            int(n * max_x - n * min_x),
                            int(x_factor * paste_im.size[1]),
                        )
                        break
                    # Try fitting pasted image to paste region height and keep aspect
                    # elif paste_im.size[0] > max_y-min_y:
                    y_factor = n * (max_y - min_y) / paste_im.size[1]
                    if y_factor * paste_im.size[0] <= (max_x - min_x):
                        modified_im_shape = (
                            int(y_factor * paste_im.size[0]),
                            int(n * max_y - n * min_y),
                        )
                        break

                else:
                    # TODO: This distorts the image
                    modified_im_shape = (max_x - min_x, max_y - min_y)
                resize_method = random.choice([3, 4, 5])
                paste_im = paste_im.resize(modified_im_shape, resample=resize_method)
                # Binarize half of the images if desired
                if binarise_half:
                    if random.choice([True, False]):
                        paste_im = paste_im.convert("L")
                        paste_im = paste_im.point(lambda x: 255 if x > 150 else 0,
                                                  mode="1")
                image.paste(paste_im, (min_x, min_y))
            # Modify annotations according to resized pasted image
            modified_chem_annotations = modify_annotations_PLN(
                paste_im_annotation, paste_im_shape, modified_im_shape, (min_x, min_y)
            )
            pasted_element_annotation += modified_chem_annotations
        return image, pasted_element_annotation
