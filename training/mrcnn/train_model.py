"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla


"""
import os
import tensorflow as tf

import imgaug.augmenters as iaa
import sys
import json
import numpy as np
import skimage.draw

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Root directory of the project
ROOT_DIR = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))

# Import Mask RCNN
sys.path.append(ROOT_DIR)
sys.path.append(dir_path)
from config import Config
import utils
import model as modellib
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
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    #NUM_CLASSES = 1 + 6  # Background + custom categories
    #NUM_CLASSES = 1 + 9
    #NUM_CLASSES = 1 + 21
    NUM_CLASSES = 1 + 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1700

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class ChemPageDataset(utils.Dataset):

    def __init__(self):
        super().__init__()
        # List of annotated classes
        self.categories = ['chemical_structure']
        
        #self.categories = ['chemical_structure', 'chemical_ID', 'text',
        #                   'title', 'table', 'list']
        #self.categories = ['chemical_structure', 'text_element', 'R_group_label', 'chemical_ID', 'reaction_arrow', 'table', '2D_plot', 'x_ray_structure', '2D_NMR_chemical_structure']
        #self.categories  = ['chemical_structure', 'ID', 'paragraph_text', 'R_group_label', 'chemical_ID', 'caption', 'reaction_arrow',
        #     'page_number', 'chemical_formula', 'paragraph_title', 'section_title', 'table', 'bar_plot', '2D_plot',
        #     'x_ray_structure', '2D_NMR_chemical_structure', 'publication_title', 'abstract_text', 'author_information',
        #     'references', 'equation']
        
        #self.categories = ['chemical_structure', 'segmented_text_element', 'table', 'arrow', 'chemical_structure_with_curved_arrows']


        # Dictionary that maps every class name (string) to an integer
        self.category_dict = {category: index + 1 for index, category in enumerate(self.categories)}

    def load_ChemPageData(self, dataset_dir, subset):
        """
        Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: "train" or "val"
        """
        # Add classes.
        categories = self.categories
        for category_index in range(len(categories)):
            self.add_class('type', category_index + 1, categories[category_index])

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations from json file
        with open(os.path.join(dataset_dir, "ChemTextOCR_annotations.json")) as annotation_file:
            annotation_dict = json.load(annotation_file)
        # Add images
        for page_name in annotation_dict['_via_img_metadata'].keys():
            page_dict = annotation_dict['_via_img_metadata'][page_name]
            image_path = os.path.join(dataset_dir, page_dict['filename'])
            region_dicts = page_dict['regions']
            # Make sure the image shape is actually saved there, VIA does not do this automatically.
            height, width = page_dict['shape']

            # to avoid confusion when looking at this:
            # kwargs are added to self.image_info
            self.add_image(
                "type",
                image_id=page_dict['filename'],
                path=image_path,
                width=width, height=height,
                region_dicts=region_dicts)
            #print(self.image_info)
            #self.image_info_dict = {info_dict['id']: info_dict for info_dict in self.image_info}


    

    def load_mask(self, image_id: int):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # Create mask array and class ID array and fill them as given in the annotation file
        IDs = []
        category_dict = self.category_dict
        #relevant_region_number = len([reg for reg in info["region_dicts"]
        #    if ])
        info["region_dicts"] = [reg for reg in info["region_dicts"] if reg["region_attributes"]["type"] in category_dict.keys()]
        mask = np.zeros([info["height"], info["width"], len(info["region_dicts"])],
                        dtype=np.uint8)
        #mask = np.zeros([info["height"], info["width"], relevant_region_number],
        #                        dtype=np.uint8)

        for i, region_dict in enumerate(info["region_dicts"]):
            if region_dict['region_attributes']['type'] in category_dict.keys():
                shape_attributes = region_dict['shape_attributes']
                # Get indíces of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(shape_attributes['all_points_y'], shape_attributes['all_points_x'])
                mask[rr, cc, i] = 1
                IDs.append(category_dict[region_dict['region_attributes']['type']])
        # Return mask, and array of class IDs of each instance.
        return mask, np.array(IDs, dtype=np.int32)

    #def load_mask(self, image_id: int):
    #    # LOAD FROM FILE
    #    """Generate instance masks for an image.
    #   Returns:
    #    masks: A bool array of shape [height, width, instance count] with
    #        one mask per instance.
    #    class_ids: a 1D array of class IDs of the instance masks.
    #    """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
    #    info = self.image_info[image_id]
        # Create mask array and class ID array and fill them as given in the annotation file
    #    IDs = []
    #    category_dict = self.category_dict

    #    IDs = []
    #    category_dict = self.category_dict

    #    image_dir = os.path.split(info['path'])[0]
    #    file_name = info['id'] # don't confuse this with the integer image_id
        # Load pre-computed mask from file and save classes of regions
    #    mask = np.load(os.path.join(image_dir, 'masks', file_name + '_mask.npz'))['arr_0']
    #    for i, region_dict in enumerate(info["region_dicts"]):
    #        IDs.append(category_dict[region_dict['region_attributes']['type']])

        # Return mask, and array of class IDs of each instance.
    #    return mask, np.array(IDs, dtype=np.int32)

    '''def load_mask(self, image_id: int):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # Create mask array and class ID array and fill them as given in the annotation file
        IDs = []
        category_dict = self.category_dict

        image_dir = os.path.split(info['path'])[0]
        file_name = info['id'] # don't confuse this with the integer image_id
        # Load pre-computed mask from file and save classes of regions
        mask = np.load(os.path.join(image_dir, 'masks', file_name + '_mask.npz'))['arr_0']
        for i, region_dict in enumerate(info["region_dicts"]):
            IDs.append(1)

        # Return mask, and array of class IDs of each instance.
        return mask, np.array(IDs, dtype=np.int32)'''


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Molecule":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ChemPageDataset()
    dataset_train.load_ChemPageData(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ChemPageDataset()
    dataset_val.load_ChemPageData(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    print("Training network")

    # Flip over and blur in 50% of cases
    #augmentation = iaa.Sometimes(0.5, [
    #    iaa.Fliplr(0.5),
    #    iaa.GaussianBlur(sigma=(0.0, 3.0))])
    # Random strong augmentation in 80% of cases
    #augmentation = iaa.Sometimes(0.8, iaa.RandAugment(m=30))
    # Resize to 50%-150% on both axes in 100% of cases
    #augmentation = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})
    #augmentation = None
    augmentation = iaa.Sometimes(0.9, 
    iaa.SomeOf((1,4), [
    iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}),
    iaa.Flipud(1),
    iaa.Fliplr(1),
    iaa.OneOf([iaa.GaussianBlur(sigma=(0.0, 2.0)),
               iaa.imgcorruptlike.JpegCompression(severity=(1,2)), 
               iaa.imgcorruptlike.Pixelate(severity=(1,2))]),
    iaa.GammaContrast((2.0, 5.0)),
    iaa.ChangeColorTemperature((1100, 10000))
    ]))
    


    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads',
                augmentation = augmentation)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect elements in the chemical literature.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ChemSegmentConfig()
    else:
        class InferenceConfig(ChemSegmentConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
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
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            # Added anchors here as for some reason they cannot be loaded from DECIMER Segmentation model
            # It was also added to the 'heads' to train as they are not pre-trained.
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask", "anchors"])
            #model.load_weights(weights_path, by_name=True, exclude=["mrcnn_mask"])
            #model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)

