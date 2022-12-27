import json
import os
import numpy as np

from PIL import Image, ImageDraw
from typing import Dict, List, Tuple


def make_region_dict(
    type: str, polygon: np.ndarray, smiles: str = False
) -> Dict:
    """
    In order to avoid redundant code in make_img_metadata_dict,
    this function creates the dict that holds the information
    for one single region. It expects a type (annotation string
    for the region) and an array containing tuples [(y0, x0), (y1, x1)...]

    Args:
        type (str): type description (annotated class)
        polygon (np.ndarray): region annotation ([(y0, x0), (y1, x1)...])_
        smiles (str, optional): SMILES representation of chemical structure
                                Defaults to False.

    Returns:
        Dict: VIA-compatible region dict
    """
    region_dict = {}
    region_dict["region_attributes"] = {}
    region_dict["region_attributes"]["type"] = type
    if smiles:
        region_dict["smiles"] = smiles
    region_dict["shape_attributes"] = {}
    region_dict["shape_attributes"]["name"] = "polygon"
    y_coordinates = [int(node[0]) for node in polygon]
    x_coordinates = [int(node[1]) for node in polygon]
    region_dict["shape_attributes"]["all_points_x"] = list(x_coordinates)
    region_dict["shape_attributes"]["all_points_y"] = list(y_coordinates)
    return region_dict


def make_img_metadata_dict(
    image_name: str,
    chemical_structure_polygon: np.ndarray,
    label_bounding_box=False,
) -> Dict:
    """
    This function takes the name of an image, the coordinates of the
    chemical structure polygon and (if it exists) the bounding box of
    the ID label and creates the _via_img_metadata subdictionary for
    the VIA input.

    Args:
        image_name (str)
        chemical_structure_polygon (np.ndarray): region annotation
                                                    [(y0, x0), (y1, x1)...]
        label_bounding_box (bool): Label region. Defaults to False

    Returns:
        Dict: VIA-compatible metadata dict
    """
    metadata_dict = {}
    metadata_dict["filename"] = image_name
    # metadata_dict['size'] = int(os.stat(os.path.join(output_dir,
    #                                                  image_name)).st_size)
    metadata_dict["regions"] = []
    # Add dict for first region which contains chemical structure
    struc_region_dict = make_region_dict(
        "chemical_structure", chemical_structure_polygon
    )
    metadata_dict["regions"].append(struc_region_dict)
    # If there is an ID label: Add dict for the label region
    if label_bounding_box:
        label_bounding_box = [(node[1], node[0]) for node in label_bounding_box]
        ID_region_dict = make_region_dict("chemical_ID", label_bounding_box)
        metadata_dict["regions"].append(ID_region_dict)
    return metadata_dict


def make_VIA_dict(metadata_dicts: List[Dict],) -> Dict:
    """
    This function takes a list of Dicts with the region information
    and returns a dict that can be opened using VIA when it is saved
    as a json file.

    Args:
        metadata_dicts (List[Dict]): List of dictionaries (as returned
                                        by make_img_metadata_dict())

    Returns:
        Dict: VIA dict (can be dumped in a file and be read by VIA)
    """
    VIA_dict = {}
    for metadata_dict in metadata_dicts:
        VIA_dict[metadata_dict["filename"]] = metadata_dict
    return VIA_dict


def make_img_metadata_dict_from_PLN_annotations(
    image_name: str,
    PLN_annotation_subdicts: List[Dict],
    categories: List[Dict],
) -> Dict:
    """
    This function takes the name of an image, the coordinates of
    annotated polygon region and the list
    of category dicts as given in the PLN annotations and returns the VIA
    _img_metadata subdictionary for this image.

    Args:
        image_name (str): image name
        PLN_annotation_subdicts (List[Dict]): annotated regions
        categories (List[Dict]): annotated classes

    Returns:
        Dict: _description_
    """
    metadata_dict = {}
    metadata_dict["regions"] = []
    metadata_dict["filename"] = image_name
    for annotation in PLN_annotation_subdicts:
        polygon = annotation["segmentation"][0]
        polygon = [
            (polygon[n], polygon[n - 1]) for n in range(len(polygon)) if n % 2 != 0
        ]
        category_ID = annotation["category_id"] - 1
        category = categories[category_ID]["name"]
        # Add dict for region which contains annotated entity
        metadata_dict["regions"].append(make_region_dict(category, polygon))
    return metadata_dict


def load_PLN_annotations(PLN_dir: str, subset: str = "val") -> Dict:
    """
    This function loads the PubLayNet annotation dictionary and returns them in a
    clear format.The returned dictionary only contain entries where the images
    actually exist locally in PLN_image_directory.
    (--> No problems if only a part of PubLayNet was downloaded.)

    Args:
        PLN_dir (str): PubLayNet directory
        subset (str): subset name (eg. "train", "val")

    Returns:
        Dict
    """
    PLN_json_path = os.path.join(PLN_dir, f"{subset}.json")
    PLN_image_dir = os.path.join(PLN_dir, subset)
    with open(PLN_json_path) as annotation_file:
        PLN_annotations = json.load(annotation_file)
    PLN_dict = {}
    PLN_dict["categories"] = PLN_annotations["categories"]
    for image in PLN_annotations["images"]:
        if os.path.exists(os.path.join(PLN_image_dir, image["file_name"])):
            PLN_dict[image["id"]] = {
                "file_name": os.path.join(PLN_image_dir, image["file_name"]),
                "annotations": [],
            }
    for ann in PLN_annotations["annotations"]:
        if ann["image_id"] in PLN_dict.keys():
            PLN_dict[ann["image_id"]]["annotations"].append(ann)
    return PLN_dict


def modify_annotations_PLN(
    regions_dicts: List[Dict],
    old_image_shape: Tuple[int],
    new_image_shape: Tuple[int],
    paste_anchor: Tuple[int],
) -> Dict:
    """
    This function takes information about the region where an image (structure,
    scheme ) has been inserted.
    The coordinates of the regions are modified according to the resizing of the
    image and the position on the page where it has been pasted.

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
            categories = {
                "chemical_structure": 6,
                "chemical_label": 7,
                "arrow": 8,
            }
            category = region["region_attributes"]["type"]
            category_ID = categories[category]

            # Load coordinates and alter them according to the resizing
            x_coords = region["shape_attributes"]["all_points_x"]
            y_coords = region["shape_attributes"]["all_points_y"]
            x_coords, y_coords = fix_polygon_coordinates(x_coords,
                                                         y_coords,
                                                         old_image_shape)
            x_ratio = new_image_shape[0] / old_image_shape[0]
            y_ratio = new_image_shape[1] / old_image_shape[1]
            x_coords = [x_ratio * x_coord + paste_anchor[0] for x_coord in x_coords]
            y_coords = [y_ratio * y_coord + paste_anchor[1] for y_coord in y_coords]
            # Get the coordinates into the PLN annotation format
            # ([x0, y0, x1, y1, ..., xn, yn])
            modified_annotation = {"segmentation": [[]], "category_id": category_ID}
            for n in range(len(x_coords)):
                modified_annotation["segmentation"][0].append(x_coords[n])
                modified_annotation["segmentation"][0].append(y_coords[n])
            modified_annotations.append(modified_annotation)
    return modified_annotations


def fix_polygon_coordinates(
    x_coords: List[int], y_coords: List[int], shape: Tuple[int]
) -> Tuple[List[int], List[int]]:
    """
    If the coordinates are placed outside of the image, this function takes the
    lists of coordinates and the image shape and adapts coordinates that are placed
    outside of the image to be placed at its borders.

    Args:
        x_coords (List[int]): x coordinates
        y_coords (List[int]): y coordinates
        shape (Tuple[int]): image shape

    Returns:
        Tuple[List[int], List[int]]: x coordinates, y coordinates
    """
    for n in range(len(x_coords)):
        if x_coords[n] < 0:
            x_coords[n] = 0
        if y_coords[n] < 0:
            y_coords[n] = 0
        if x_coords[n] > shape[0]:
            x_coords[n] = shape[0] - 1
        if y_coords[n] > shape[1]:
            y_coords[n] = shape[1] - 1
    return x_coords, y_coords


def illustrate_annotations(image: Image, annotations: List[Dict]):
    """
    This function takes a PIL.Image object and an annotation dict and returns the
    Image where the annotated regions are depicted as coloured boxes.

    Args:
        image (Image)
        annotation (List[Dict]): List of region dictionaries

    Returns:
        PIL.Image: image with illustrated annotations
    """
    colour_dict = {
        'chemical_structure': (0, 255, 0, 50),
        'chemical_ID': (255, 0, 0, 50),
        'arrow': (0, 0, 255, 50),
        'R_group_label': (255, 255, 0, 50),
        'reaction_condition_label': (0, 255, 255, 50),
        'table': (255, 0, 255, 50),
        'text': (0, 100, 0, 50),
        'title': (100, 0, 0, 50)
    }

    for region in annotations:
        x_coords = region['shape_attributes']['all_points_x']
        y_coords = region['shape_attributes']['all_points_y']
        polygon = [(x_coords[n], y_coords[n]) for n in range(len(x_coords)) if len]
        draw = ImageDraw.Draw(image, 'RGBA')
        colour = colour_dict[region['region_attributes']['type']]
        draw.polygon(polygon, fill=colour)

    return image
