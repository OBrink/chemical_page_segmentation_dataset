"""
Mask R-CNN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla


"""
import os
import tensorflow as tf

import imgaug.augmenters as iaa
import sys
import numpy as np
from typing import List, Dict, Tuple
import skimage.draw



# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# Root directory of the project
ROOT_DIR = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
data_gen_path = os.path.join(dir_path, "../training_data_generation")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
sys.path.append(dir_path)
sys.path.append(data_gen_path)
from config import Config
import utils
import model as modellib

import warnings
warnings.filterwarnings("ignore")



from ChemSegmentationDatasetCreator import ChemSegmentationDatasetCreator


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class ChemSegmentConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ChemSegment"
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4
    GPU_COUNT = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + chemical structures
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 250
    # VALIDATION_STEPS = 1
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class ChemPageDataset(utils.Dataset):

    def __init__(
        self,
        smiles_path: str = None,
        pregenerated_depiction_path: str = None,
        test_mode=False
    ):
        self.categories = ['BG', 'chemical_structure']
        super().__init__()
        # List of annotated classes
        if test_mode:
            smiles_list = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
            self.data_creator = ChemSegmentationDatasetCreator(smiles_list,)
        elif smiles_path:
            smiles_path = os.path.join(os.path.split(__file__)[0], "smiles.txt")
            with open(smiles_path, 'r') as smiles_file:
                smiles_list = [line[:-1].split('\t')[1].replace(" ", "")
                               for line in smiles_file.readlines()]
            self.data_creator = ChemSegmentationDatasetCreator(smiles_list,)
        elif pregenerated_depiction_path:
            self.data_creator = ChemSegmentationDatasetCreator(
                precomputed_depiction_path=pregenerated_depiction_path)
        else:
            raise AttributeError(
                "No smiles file path or pregenerated depiction path given")

        # Dictionary that maps every class name (string) to an integer
        self.category_dict = {category: index for index, category
                              in enumerate(self.categories)}

    def load_mask(self, annotations: List[Dict], shape: Tuple[int, int]):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        # Create mask array and class ID array and fill them as given in the annotations
        IDs = []
        category_dict = self.category_dict
        annotations = [ann for ann in annotations
                       if ann["region_attributes"]["type"] in category_dict.keys()]
        mask = np.zeros([shape[0], shape[1], len(annotations)],
                        dtype=np.uint8)

        for i, region_dict in enumerate(annotations):
            if region_dict['region_attributes']['type'] in category_dict.keys():
                shape_attributes = region_dict['shape_attributes']

                # Correct annotated pixels outside of the image
                for index in range(len(shape_attributes['all_points_y'])):
                    if shape_attributes['all_points_y'][index] >= shape[0]:
                        shape_attributes['all_points_y'][index] = shape[0] - 1
                    if shape_attributes['all_points_y'][index] < 0:
                        shape_attributes['all_points_y'][index] = 0
                    if shape_attributes['all_points_x'][index] >= shape[1]:
                        shape_attributes['all_points_x'][index] = shape[1] - 1
                    if shape_attributes['all_points_x'][index] < 0:
                        shape_attributes['all_points_x'][index] = 0
                # Get indíces of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(shape_attributes['all_points_y'],
                                              shape_attributes['all_points_x'])
                mask[rr, cc, i] = 1
                IDs.append(category_dict[region_dict['region_attributes']['type']])
        # Return mask, and array of class IDs of each instance.
        return mask, np.array(IDs, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Molecule":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(
    model,
    smiles_path: str = None,
    pregenerated_depiction_path: str = None,
    test_mode: bool = False
):
    """Train the model."""
    # Training dataset.
    dataset = ChemPageDataset(smiles_path, pregenerated_depiction_path, test_mode)
    dataset.prepare()

    augmentation = iaa.Sometimes(
        0.2,
        iaa.SomeOf((1, 4), [
            iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}),
            iaa.Flipud(1),
            iaa.Fliplr(1),
            iaa.OneOf([iaa.GaussianBlur(sigma=(0.0, 2.0)),
                       iaa.imgcorruptlike.JpegCompression(severity=(1, 2)),
                       iaa.imgcorruptlike.Pixelate(severity=(1, 2))]),
            iaa.GammaContrast((2.0, 5.0)),
            iaa.ChangeColorTemperature((1100, 10000))
        ]))
    model.train(dataset,
                learning_rate=config.LEARNING_RATE,
                epochs=500,
                layers='all',
                augmentation=augmentation)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect elements in the chemical literature.')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--smiles_file',
                        required=False,
                        help='Smiles file to use for generation of depictions')
    parser.add_argument('--pregenerated_structure_path',
                        required=False,
                        help='Path with pregenerated of depictions for training')
    
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: Automatically generated")
    print("Logs: ", args.logs)

    # Configurations
    config = ChemSegmentConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=args.logs)

    # Select weights file to load
    if args.weights:
        if args.weights.lower() == "last":
            # Find last trained weights
            weights_path = model.find_last()[1]
        else:
            weights_path = args.weights
        # Load weights
        print("Loading weights ", weights_path)
        # Added anchors here as for some reason they cannot be loaded from DECIMER
        # Segmentation model It was also added to the 'heads' to train as they are
        # not pre-trained.
        # model.load_weights(weights_path, by_name=True, exclude=[
        #    #"mrcnn_class_logits",
        #    #"mrcnn_bbox_fc",
        #    #"mrcnn_bbox",
        #    #"mrcnn_mask",
        #    "anchors"
        #    ])
        # model.load_weights(weights_path, by_name=True, exclude=["mrcnn_mask"])
        model.load_weights(weights_path, by_name=True, exclude=["anchors"])
        print("Weights loaded")

    train(model, args.smiles_file, args.pregenerated_structure_path)
