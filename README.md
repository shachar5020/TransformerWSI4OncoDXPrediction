TransformerWSI4OncoDXPrediction
-----
A deep learning pipeline for the prediction of breast cancer recurrence risk (OncotypeDX) and chemotherapy benefit from H&E‑stained whole slide images.

The feature‑extraction backbone and slide‑level transformer architecture are adapted from https://github.com/prov-gigapath/prov-gigapath © the Prov‑GigaPath authors, licensed under Apache‑2.0, and adjusted to fit our requirements.

The software has been tested with Ubuntu 22.04, an NVIDIA RTX A6000 gpu and the following python libraries (also specified in environment.yml):

--------------------
        python=3.9.18
        torch=2.0.0
        numpy=1.26.4
        pandas=2.2.2
        scipy=1.13.1
        openslide-python=1.3.1


In the tested configuration, the environment installation time averages 20 minutes. Typical fine-tuning time for the public TCGA data with 500 training slides averages 15 minutes. Inference averages 3 minutes per slide.



Usage
-----

1) Run preprocessing:
-------------------
        run_preprocess.py [arguments]
    
    Arguments:
        --data_dir            Path to the folder containing the slides.
        --metadata            Path to the metadata csv file.
        --tissue_coverage     Minimum tissue percentage for a valid tile (Default 0.3).
        
Example
-------
Run with default parameters:

    $ python3 run_preprocess.py --data_dir ./TCGA --metadata ./TCGA/meta.csv
    
The script will create two folders in the data directory named Grids_10 and SegData which are necessary for the later steps.
The slides must be whole slide image (WSI) files in a format supported by the openslide library (for example .svs). One public source for WSIs is The Cancer Genome Atlas (TCGA) dataset, and the slides from TCGA can be obtained from the Genomic Data Commons (GDC) website. Explanation on how to download TCGA slides can be found at: https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/


2) Extract slide tile features:
--------------------------
        png_tile_extraction.py [arguments]
    
    Arguments:
        --data_dir            Location of data root folder.
        --metadata_file       Location of metadata file.
        --target_mpp          MPP at which the tile features will be extracted (Default 0.5).
        --hf_token            Hugging Face token, needed to load the prov-gigapath model.
        
Example
-------
Run with default parameters:

    $ python3 png_tile_extraction.py --data_dir ./TCGA --metadata_file ./TCGA/meta.csv --target_mpp 0.5 --hf_token <your_token>
    
The script will create two folders in the data directory: gigapath_features containing the per‑tile feature for each slide and png_tiles containing the corresponding image tiles and their coordinates.

To obtain the Hugging Face token used to access the model you need to agree to the terms set by Prov-Gigapath, this can be done at https://huggingface.co/prov-gigapath/prov-gigapath.


3) Train the transformer:
----------------------
        finetune/main.py [arguments]
    
    Arguments:
        --dataset_csv         Dataset csv file.
        --root_path           Path ending with "gigapath_features".
        --epochs              Number of training epochs (Default 5).
        --warmup_epochs       Number of warmup epochs (Default 1).
        --gc                  Gradient accumulation (Default 32).
        --model_select        Criteria for choosing the model checkpoint ("last_epoch" or "val").
        --lr_scheduler        Learning rate scheduler ("cosine" or "fixed").
        --save_dir            Save directory for outputs.
        --exp_name            Experiment name, a folder with that name will be created.
        --train_dataset       List of dataset names to train on.
        --train_fold          List of folds to train on.
        --val_dataset         List of dataset names to validate on.
        --val_fold            List of folds to validate on.
        --test_dataset        List of dataset names to test on.
        --test_fold           List of folds to test on.
        --label               Column name of the label.
        --loss_fn             Loss function (Default "mse").
        --hf_token            Hugging Face token, needed to load the prov-gigapath model.
        
Run python3 finetune/main.py -h for more arguments
        
Example
-------
Run with default parameters:

    $ python3 finetune/main.py --dataset_csv TCGA/meta.csv --root_path TCGA/gigapath_features/ --epochs 5 --warmup_epochs 1 --gc 32 --model_select 'last_epoch' --lr_scheduler 'cosine' --save_dir ./ --exp_name 'example_train' --train_dataset '["TCGA"]' --train_fold '[2,3,4,5]' --val_dataset '["TCGA"]' --val_fold '[1]' --test_dataset '["TCGA"]' --test_fold '[6]' --label 'RS' --loss_fn 'mse' --hf_token <your_token>

In order to train the transformer, the root_path must end with "gigapath_features" and a folder named "png_tiles" must be present at the same location.


4) Run inference with the transformer:
-----------------------------------
        finetune/main.py --run_inference [arguments]
    
    Arguments:
        --dataset_csv         Dataset csv file.
        --root_path           Path ending with "gigapath_features".
        --save_dir            Save directory for outputs.
        --exp_name            Experiment name.
        --test_dataset        List of dataset names to test on.
        --test_fold           List of folds to test on.
        --label               Column name of the label.
        --loss_fn             Loss function.
        --model_ckpt          Model checkpoint path.
        --hf_token            Hugging Face token
        
Example
-------
Run with default parameters:

    $ python3 finetune/main.py --dataset_csv TCGA/meta.csv --root_path TCGA/gigapath_features/ --save_dir ./ --exp_name 'example_test' --test_dataset '["TCGA"]' --test_fold '[6]' --label 'RS' --loss_fn 'mse' --model_ckpt finetune_gigapath/example_train/eval_pretrained_finetune_gigapath/checkpoint.pt --run_inference --hf_token <your_token>

Run inference without training with the Prov-Gigapath pretrained model:

    $ python3 finetune/main.py --dataset_csv TCGA/meta.csv --root_path TCGA/gigapath_features/ --save_dir ./ --exp_name 'example_test' --test_dataset '["TCGA"]' --test_fold '[6]' --label 'RS' --loss_fn 'mse' --run_inference --hf_token <your_token>


In order to train or run inference, the metadata csv file must have a row for each slide and the following columns with filled values:

 - "file": filename of the slide (with extension)
 - "id": dataset identifier used by the *dataset* arguments
 - "fold": fold identifier used by the *fold* arguments
 - "mpp": magnification in microns per pixel
 - label columns: any label you wish to train or test on (for example "RS")
 
For the *dataset* arguments (train_dataset, val_dataset, test_dataset) pass a list of names that will be matched to the values in the "id" column. For the *fold* arguments pass a list of identifiers that will be matched to the values in the "fold" column.


Model Uses
----------
The model intended use is derived from the intended use as set by Prov-Gigapath. The model is intended to support AI research on pathology and the reproduction of the reported results. Any deployed use of the model is unintended and is out of scope.


Requirements
------- 

To install the requirements, use:
    
    $ conda env create -f environment.yml

The code to reproduce the figures from the paper "Deep Learning on Histopathological Images to Predict Breast Cancer Recurrence Risk and Chemotherapy Benefit" in matlab is also provided:

Requirements
------- 

        MATLAB R2024a (tested version)
        
        Required data tables (for selected analyses, see below)

Usage
-----

Clone or download the repository and run the provided scripts together with the included metadata and functions folders. These reproduce the main figures presented in the paper.

Main Analysis Scripts
--------------

        main_analyses_UCMC.m and main_analyses_TCGA.m
        
        Reproduce the figures corresponding to the UCMC and TCGA cohorts.


Chemotherapy Benefit Analyses
--------------

        ChemoBenefit_baseline.m
        
        ChemoBenefit_postMeno_adjusted.m
        
        ChemoBenefit_preMeno_adjusted.m

        These scripts generate the Kaplan–Meier plots shown in Figure 3.
        ⚠️ Note: They require access to the TAILORx clinical data table, which is not distributed with this repository.



Citation
--------
If you use this code, please cite Prov-Gigapath:

```bibtex
@article{xu2024gigapath,
  title={A whole-slide foundation model for digital pathology from real-world data},
  author={Xu, Hanwen and Usuyama, Naoto and Bagga, Jaspreet and Zhang, Sheng and Rao, Rajesh and Naumann, Tristan and Wong, Cliff and Gero, Zelalem and González, Javier and Gu, Yu and Xu, Yanbo and Wei, Mu and Wang, Wenhui and Ma, Shuming and Wei, Furu and Yang, Jianwei and Li, Chunyuan and Gao, Jianfeng and Rosemon, Jaylen and Bower, Tucker and Lee, Soohee and Weerasinghe, Roshanthi and Wright, Bill J. and Robicsek, Ari and Piening, Brian and Bifulco, Carlo and Wang, Sheng and Poon, Hoifung},
  journal={Nature},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
