import argparse
import os

import preprocess
from finetune import utils

parser = argparse.ArgumentParser(description='Data preparation script')
parser.add_argument('--data_dir', type=str, help='location of data root folder')
parser.add_argument('--metadata', type=str, help='location of the metadata csv file')
parser.add_argument('--tissue_coverage', type=float, default=0.3, help='min. tissue percentage for a valid tile')

args = parser.parse_args()
num_workers = utils.get_cpu()



def prepare_dataset(data_dir, tile_size, tissue_coverage, mag, metadata):
    preprocess.make_segmentations(data_path=data_dir,
                                     num_workers=num_workers,
                                     metadata=metadata)
    preprocess.make_grid(ROOT_DIR=data_dir,
                            tile_sz=tile_size,
                            tissue_coverage=tissue_coverage,
                            desired_magnification=mag,
                            num_workers=num_workers,
                            metadata=metadata)



if __name__ == '__main__':
    prepare_dataset(data_dir = args.data_dir, 
                    tile_size=256,
                    tissue_coverage=args.tissue_coverage,
                    mag=10,
                    metadata = args.metadata)
