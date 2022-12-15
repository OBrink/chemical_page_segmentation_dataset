import os
import json
from typing import Dict, Tuple
from PIL import Image


class FigureExtractor():

    def __init__(self):
        # Set paths according to your needs
        self.figure_save_path = os.path.normpath('./PLN_figures')
        self.PLN_dir = os.path.normpath('./publaynet/')
        self.PLN_json_path = os.path.join(self.PLN_dir, 'train.json')
        self.PLN_image_dir = os.path.join(self.PLN_dir, 'train')
        # self.PLN_json_path = os.path.join(self.PLN_dir, 'val.json')
        # self.PLN_image_dir = os.path.join(self.PLN_dir, 'val')
        self.PLN_annotations = self.load_PLN_annotations()
        self.PLN_annotation_iterator = (self.PLN_annotations[page]
                                        for page
                                        in self.PLN_annotations.keys()
                                        if page != 'categories')

    def load_PLN_annotations(
        self,
    ) -> Dict:
        """
        This function loads the PubLayNet annotation dictionary and
        returns them in a clear format.
        The returned dictionary only contain entries where the images
        actually exist locally in PLN_image_directory.
        (--> No problems if only a part of PubLayNet was downloaded.)

        Args:
            PLN_json_path (str): Path of PubLayNet annotation file
            PLN_image_dir ([type]): Path of directory with PubLayNet images

        Returns:
            Dict
        """
        with open(self.PLN_json_path) as annotation_file:
            PLN_annotations = json.load(annotation_file)
        PLN_dict = {}
        PLN_dict['categories'] = PLN_annotations['categories']
        for image in PLN_annotations['images']:
            file_path = os.path.join(self.PLN_image_dir,  image['file_name'])
            if os.path.exists(file_path):
                PLN_dict[image['id']] = {'file_path': file_path,
                                         'annotations': []}
        for ann in PLN_annotations['annotations']:
            if ann['image_id'] in PLN_dict.keys():
                PLN_dict[ann['image_id']]['annotations'].append(ann)
        return PLN_dict

    def save_figures_in_separate_images(
        self,
    ) -> None:
        """
        This function returns detects figures in PubLayNet and saves them in
        separate image files.

        Returns:
            None
        """
        for page_annotation in self.PLN_annotation_iterator:
            # Make sure only pages that contain figures are processed.
            category_IDs = [annotation['category_id'] - 1
                            for annotation in page_annotation['annotations']]
            found_categories = [self.PLN_annotations['categories'][category_ID]['name']
                                for category_ID in category_IDs]
            if 'figure' in found_categories:
                # Go through all annotated regions
                fig_count = 0
                for annotation in page_annotation['annotations']:
                    category_ID = annotation['category_id'] - 1
                    category = self.PLN_annotations['categories'][category_ID]['name']
                    # Determine figure bounding boxes
                    if category == 'figure':
                        polygon = annotation['segmentation'][0]
                        polygon_y = [int(polygon[n]) for n
                                     in range(len(polygon))
                                     if n % 2 != 0]
                        polygon_x = [int(polygon[n]) for n
                                     in range(len(polygon))
                                     if n % 2 == 0]
                        figure_bbox = (min(polygon_x),
                                       min(polygon_y),
                                       max(polygon_x),
                                       max(polygon_y))
                        page_image = Image.open(page_annotation['file_path'])
                        im_size = page_image.size
                        figure_bbox = self.correct_bbox(figure_bbox, im_size)
                        # Crop image and save it
                        figure_image = page_image.crop(figure_bbox)
                        page_name = os.path.split(page_annotation['file_path'])[1]
                        fig_name = f'{page_name}_fig_{fig_count}.png'
                        fig_save_path = os.path.join(self.figure_save_path,
                                                     fig_name)
                        figure_image.save(fig_save_path)
                        fig_count += 1

    def correct_bbox(self,
                     bbox: Tuple[int, int, int, int],
                     im_size: Tuple[int, int]
                     ) -> Tuple[int, int, int, int]:
        """
        This function takes bounding box coordinates (left, up, right, bottom)
        and an image size (width, height) and corrects the bounding box
        coordinates if they are outside of the image

        Args:
            bbox (Tuple[int, int, int, int]): (left, up, right, bottom)
            im_size (Tuple): x,y

        Returns:
            Tuple[int, int, int, int]: corrected bounding box
        """
        left, up, right, bottom = bbox
        x_max, y_max = im_size
        if left < 0:
            left = 0
        if right >= x_max:
            right = x_max - 1
        if up < 0:
            up = 0
        if bottom >= y_max:
            bottom = y_max - 1
        return left, up, right, bottom


def main():
    figure_extractor = FigureExtractor()
    figure_extractor.save_figures_in_separate_images()


if __name__ == '__main__':
    main()
