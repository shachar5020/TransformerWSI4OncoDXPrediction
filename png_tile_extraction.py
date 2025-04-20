import os
import argparse
import numpy as np
import pandas as pd
import openslide
import pickle
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
import cv2

import timm
import gigapath.slide_encoder as slide_encoder
from gigapath.pipeline import load_tile_slide_encoder, run_inference_with_tile_encoder, run_inference_with_slide_encoder

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Feature extraction script')
parser.add_argument('--data_dir', type=str, default='./TCGA', help='location of data root folder')
parser.add_argument('--metadata_file', type=str, default='./TCGA/meta.csv', help='location of metadata file')
parser.add_argument('--target_mpp', type=float, default=0.5, help='MPP at which the tile features will be extracted')
parser.add_argument('--hf_token', type=str, help='Hugging face token, needed to load the prov-gigapath model')

args = parser.parse_args()


# Process args
base_path = args.data_dir
meta_path = args.metadata_file
target_mpp = args.target_mpp
segmentation_dir = os.path.join(base_path, "Grids_10")
progress_file = os.path.join(base_path, f"progress.txt")

# Set the environment variable for the Hugging Face token
os.environ["HF_TOKEN"] = args.hf_token
# load the tile encoder
tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
# load the slide encoder
slide_encoder_model = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
tile_encoder, slide_encoder_model = load_tile_slide_encoder()

# Function to encode a slide and save its embeddings in the tile and slide level
def encode_slide(slide_patches_path, slide_name, slide_encoder_model, tile_encoder, output_dir):
    # tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)
    # outputs a dict with keys 'tile_embeds' and 'coords'. 'tile_embeds' is a tensor of shape (N, 1536) where N is the number of tiles
    # 'coords' is a tensor of shape (N, 2) containing the coordinates of the tiles. The i-th row of 'coords' corresponds to the i-th row of 'tile_embeds'
    # slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)
    # outputs a dict with keys 'layer_<1, 2, ..., 12, last>_embed' where each value is a tensor containing the embeddings of the slide at the corresponding layer
    # 'final' is the final layer of the slide encoder.
    # We want to lave for each slide all of its tile embeddings and the 6-th and final layer slide embeddings.
    # We will save the embeddings in seperate files, in the same folder, with the same name as the slide.
    # The embeddings will be saved in the following format:
    # tile_embeds_<slide_name>.npy
    # layer_6_embed_<slide_name>.npy
    # final_embed_<slide_name>.npy
    # The embeddings will be saved in the output_dir
    # The slide patches are saved in the slide_patches_path
    # The slide name is the name of the slide
    # The slide encoder model is the slide encoder model
    # The tile encoder is the tile encoder
    # The output_dir is the directory where the embeddings will be saved
    # The function does not return anything

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the slide patches
    image_paths = [os.path.join(slide_patches_path, img) for img in os.listdir(slide_patches_path) if img.endswith('.png')]
    # Run inference with the tile encoder
    tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)
    # Run inference with the slide encoder
    slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)
    # Save the tile embeddings
    np.save(os.path.join(output_dir, f"tile_embeds_{slide_name}.npy"), tile_encoder_outputs['tile_embeds'])
    # Save the 6-th layer embeddings
    np.save(os.path.join(output_dir, f"layer_6_embed_{slide_name}.npy"), slide_embeds['layer_6_embed'])
    # Save the final layer embeddings
    np.save(os.path.join(output_dir, f"final_embed_{slide_name}.npy"), slide_embeds['last_layer_embed'])


# Function to round mpp values to the nearest 1/(2n) or 1
def round_mpp(mpp_value):
    try:
        if abs(mpp_value - 1) < 0.1:
            return 1
        else:
            n = int(round(1 / (2 * mpp_value)))
            return 1 / (2 * n)
    except:
        return None

# Read slide metadata
try:
    slides_df = pd.read_csv(meta_path)
except Exception as e:
    raise FileNotFoundError(f"Error reading the metadata file: {e}")

# Check if 'mpp' column exists
if 'mpp' not in slides_df.columns:
    raise KeyError(f"Column 'mpp' not found in the metadata file. Available columns: {slides_df.columns.tolist()}")

# Drop rows with null or string 'MPP' values
slides_df = slides_df.dropna(subset=['mpp'])
slides_df = slides_df[slides_df['mpp'].apply(lambda x: isinstance(x, (int, float)))]

# Round the 'mpp' column values
slides_df['mpp'] = slides_df['mpp'].apply(round_mpp)

slides_df = slides_df.dropna(subset=['mpp'])
slides_df = slides_df[slides_df['mpp'].apply(lambda x: isinstance(x, (int, float)))]

# Function to extract and save patches
def extract_patches(slide_path, segmentation_path, output_dir, mpp, target_mpp = 0.5):
    slide = openslide.open_slide(slide_path)
    mpp_seg = 1
    output_tile_size = 256

    # Calculate the closest level for target MPP
    downsample_factors = [float(level_downsample) for level_downsample in slide.level_downsamples]
    closest_level = find_best_level(mpp, target_mpp, downsample_factors)
    scale_factor_image = target_mpp / (mpp * downsample_factors[closest_level])
    best_level_output_tile_size = int(round(output_tile_size * scale_factor_image))
    scale_factor_level_0_to_half_mpp = target_mpp / mpp
    scale_factor_seg = mpp_seg / mpp
    level_0_seg_tile_size = output_tile_size * scale_factor_seg
    #level_0_output_tile_size = int(round(level_0_seg_tile_size // 2.))
    level_0_output_tile_size = int(round(level_0_seg_tile_size // (mpp_seg / target_mpp)))

    # Load segmentation data
    with open(segmentation_path, 'rb') as f:
        tiles_coords = pickle.load(f)
        
    if len(tiles_coords) == 0:
        return False
    
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    for (x, y) in tiles_coords:
        for i in range(min(int((mpp_seg / target_mpp)),2)):
            for j in range(min(int((mpp_seg / target_mpp)),2)):
                coord_x = x + i * level_0_output_tile_size
                coord_y = y + j * level_0_output_tile_size
                try:
                    patch = slide.read_region((coord_y, coord_x), closest_level, (best_level_output_tile_size, best_level_output_tile_size)).convert('RGB')
                except:
                    print(f"couldn't read the coords {coord_y}, {coord_x} at level {closest_level} with size {best_level_output_tile_size}")
                    raise
                
                # Convert patch to numpy array for cv2.resize
                patch_np = np.array(patch)
                resized_patch = cv2.resize(patch_np, (output_tile_size, output_tile_size))
                
                # Convert back to PIL Image
                resized_patch_img = Image.fromarray(resized_patch)
                patch_name = f"{int(round(coord_x / scale_factor_level_0_to_half_mpp))}x_{int(round(coord_y / scale_factor_level_0_to_half_mpp))}y.png"
                resized_patch_img.save(os.path.join(output_dir, patch_name))
    return True

def find_best_level(mpp, target_mpp, downsample_factors):
    best_level = 0
    while mpp * downsample_factors[best_level] <= target_mpp:
        if len(downsample_factors)==best_level+1 or mpp * downsample_factors[best_level+1] > target_mpp:
            return best_level
        else:
            best_level += 1
    raise "slide level 0 mpp is smaller than target"

# Load progress if it exists
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        start_index = int(f.read().strip())
else:
    start_index = 0

slides_df = slides_df[start_index:]

# Process each slide with progress tracking and saving
for idx, row in tqdm(slides_df.iterrows(), total=len(slides_df), initial=start_index):
    slide_file = row['file']
    slide_path = os.path.join(base_path, slide_file)
    if not os.path.exists(slide_path):
        print(f"slide: {slide_path} missing, skipping")
        with open(progress_file, 'w') as f:
            f.write(str(idx + 1))
        continue
    suffix = f'_mpp{target_mpp}' if target_mpp != 0.5 else ''
    slide_name = os.path.splitext(slide_file)[0]
    segmentation_path = os.path.join(segmentation_dir, f"{slide_name}--tlsz256.data")
    output_dir_patches = os.path.join(base_path, f'png_tiles{suffix}', slide_name)
    output_dir_features = os.path.join(base_path, f'gigapath_features{suffix}', slide_name)
    try:
        if not extract_patches(slide_path, segmentation_path, output_dir_patches, row['mpp'], target_mpp):
            print(f"slide: {slide_path} has no legitimate tiles, skipping")
            with open(progress_file, 'w') as f:
                f.write(str(idx + 1))
            continue
    except:
        print(f"couldn't extract patches from slide: {slide_path}")
        raise
    encode_slide(output_dir_patches, slide_name, slide_encoder_model, tile_encoder, output_dir_features)
    
    # Save progress
    with open(progress_file, 'w') as f:
        f.write(str(idx + 1))

# Delete the progress file after completion
if os.path.exists(progress_file):
    os.remove(progress_file)
