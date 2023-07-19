from ChemSegmentationDatasetCreator import ChemSegmentationDatasetCreator
import json
from multiprocessing import Process
import time
import os
import sys
from typing import List, Tuple


def generate_and_save_depiction_and_annotation_batch(
    smiles_list: List[str],
    output_dir: str,
    ID_list: List[str],
    shape: Tuple[int, int] = (200, 200),
    seed: int = 42
) -> None:
    """
    This function generates one batch of images and annotations and saves the files
    in a given output_directory.

    Args:
        smiles_list (List[str]): list of SMILES str
        output_dir (str): path to output directory
        ID_list (List[str]): list of IDs for naming the files
        shape (Tuple[int, int], optional): output image shape. Defaults to (200, 200).
        seed (int, optional): _description_. Defaults to 42.
    """
    data_generator = ChemSegmentationDatasetCreator(smiles_list=["CCC"],
                                                    load_PLN=False,
                                                    seed=seed)
    for index in range(len(smiles_list)):
        try:
            smiles = smiles_list[index]
            ID = ID_list[index]
            im, annotation = data_generator.generate_structure_with_annotation(
                smiles,
                shape,
                data_generator.depictor.random_choice([True, False]),
                False)
            im.save(os.path.join(output_dir, f"{ID}.png"))
            with open(os.path.join(output_dir, f"{ID}.json"), "w") as outfile:
                outfile.write(json.dumps(annotation))
        except Exception as e:
            print(Exception, e)


def large_batch_generation(
    smiles_list: List[str],
    output_dir: str,
    shape: Tuple[int, int] = (200, 200),
    num_processes: int = 40,
    seed: int = 42,
    timeout_limit: int = 1800
) -> None:
    """
    This function generates a large batch of images and annotations and saves the files
    """

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    chunksize = 1500
    counter = (n for n in range(int(len(smiles_list)/chunksize)))
    ID_chunks = chunks(range(len(smiles_list)), chunksize)
    smiles_chunks = chunks(smiles_list, chunksize)
    async_proc_args = (
        (
            next(smiles_chunks),
            output_dir,
            next(ID_chunks),
            shape,
            (seed * n + 1),  # individual seed
        )
        for n in counter)

    process_list = []
    while True:
        for proc, init_time in process_list:
            # Remove finished processes
            if not proc.is_alive():
                process_list.remove((proc, init_time))
            # Remove timed out processes
            elif time.time() - init_time >= timeout_limit:
                process_list.remove((proc, init_time))
                proc.terminate()
                proc.join()
        if len(process_list) < num_processes:
            # Start new processes
            for _ in range(num_processes-len(process_list)):
                try:
                    p = Process(target=generate_and_save_depiction_and_annotation_batch,
                                args=next(async_proc_args))
                    process_list.append((p, time.time()))
                    p.start()
                except StopIteration:
                    break
        if len(process_list) == 0:
            break

def main():
    smiles_path = sys.argv[1]
    output_dir = sys.argv[2]
    with open(smiles_path, 'r') as smiles_file:
        smiles_list = [line[:-1].split('\t')[1].replace(" ", "")
                       for line in smiles_file.readlines()]
    large_batch_generation(smiles_list, output_dir)

if __name__ == "__main__":
    main()
