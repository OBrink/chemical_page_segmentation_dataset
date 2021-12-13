import sys
import os
from ChemPageSegmentationDatasetCreator import ChemPageSegmentationDatasetCreator as CPSD_creator
import json
from itertools import cycle
from typing import List, Dict
import numpy as np
from multiprocessing import Pool


def create_subset(smiles_list: List[str], PLN_annotations:List, subset_size:int) -> List[Dict]:
    dataset_creator = CPSD_creator(smiles_list, load_PLN_annotations=True)
    dataset_creator.PLN_page_annotation_iterator = cycle(PLN_annotations)
    annotations = []
    for n in range(subset_size):
        annotation = dataset_creator.create_and_save_chemical_page()
        annotations.append(annotation)
        print(n)
    return annotation
    


def create_dataset(smiles_file_path: str, dataset_size:int, n_workers: int = 40):
    with open(smiles_file_path, "r") as smiles_input:
        smiles_input = smiles_input.readlines()
        smiles_input = [smi.split(',')[-1][:-1] for smi in smiles_input]
        # Get n_workers lists of smiles strings
        smiles_input = [list(l) for l in np.array_split(smiles_input, n_workers)]
        dataset_creator = CPSD_creator(smiles_input)
        # Get n_workers list of PLN_annotation dicts
        PLN_annotations = [dataset_creator.PLN_annotations[page]
                            for page 
                            in dataset_creator.PLN_annotations.keys() 
                            if page != 'categories']
        
        PLN_annotations = [list(l) for l in np.array_split(PLN_annotations, n_workers)]
        starmap_tuples = [(smiles_input[i], PLN_annotations[i], int(dataset_size/n_workers))
                           for i in range(n_workers)
                          ]
        with Pool(n_workers) as p:
            annotations = p.starmap(create_subset, starmap_tuples)
        # Flatten list 
        annotations = [l for s in annotations for l in s]
        
        
        VIA_output = dataset_creator.make_VIA_dict(annotations)
        with open("ChemPageDatasetAnnotation.json", "w") as output:
            json.dump(VIA_output, output)
    return
        
                
if __name__ == "__main__":
    create_dataset(sys.argv[1], int(sys.argv[2]))