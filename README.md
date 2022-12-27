# Chemical Segmentation Training Data Generation

## Set up the training data generation

1) Download the PubLayNet dataset

    `wget -O training_data_generation/PubLayNet_PDF.tar.gz https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/PubLayNet_PDF.tar.gz
`

2) Unpack the PubLayNet dataset. The dataset should be located at training_data_generation/publaynet/

3) Download the COCO 2017 dataset to use its content as random (non-chemical) images using the Kaggle API (https://github.com/Kaggle/kaggle-api).

    `kaggle dataset download awsaf49/coco-2017-dataset`

4) Unpack the COCO dataset. The images should be located at `training_data_generation/random_images/`. We used the images from the `train` subset.

5) Download SMILES list from https://zenodo.org/record/5155037#.Y6r-9HbMK38 and save it as ´smiles.txt´ in training_data_generation. 