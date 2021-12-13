import sys
from ChemPageSegmentationDatasetCreator import ChemPageSegmentationDatasetCreator as CPSD_creator
import json


def create_dataset(smiles_file_path: str, dataset_size:int):
    with open(smiles_file_path, "r") as smiles_input:
        smiles_input = smiles_input.readlines()
        smiles_input = [smi.split(',')[-1][:-1] for smi in smiles_input]
        dataset_creator = CPSD_creator(smiles_input)
        VIA_output = dataset_creator.create_and_save_chemical_pages(dataset_size)
        VIA_output = dataset_creator.make_VIA_dict(VIA_output)
        with open("ChemPageDatasetAnnotation.json", "w") as output:
            json.dump(VIA_output, output)
    return


if __name__ == "__main__":
    if len(sys.argv) == 3:
        create_dataset(sys.argv[1], int(sys.argv[2]))
    else:
        print('{} smiles_file_path (str) dataset_size (int)')