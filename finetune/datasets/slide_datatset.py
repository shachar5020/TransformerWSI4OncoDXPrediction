import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class SlideDatasetForTasks(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 root_path: str,
                 task_config: dict, 
                 slide_key: str='file',
                 label: str='RS',
                 dataset_name: list= ['TCGA'],
                 folds: list=[2,3,4,5],
                 test_on_all: bool=False,
                 **kwargs
                 ):
        '''
        This class is used to set up the slide dataset for different tasks

        Arguments:
        ----------
        data_df: pd.DataFrame
            The dataframe that contains the slide data
        root_path: str
            The root path of the tile embeddings
        task_config: dict
            The task configuration dictionary
        slide_key: str
            The key that contains the slide id
        '''
        self.root_path = root_path
        self.slide_key = slide_key
        self.task_cfg = task_config
        self.label = label
        self.test_on_all = test_on_all
        
        try:
            data_df[slide_key] = data_df[slide_key].str.removesuffix('.svs')
            data_df[slide_key] = data_df[slide_key].str.removesuffix('.mrxs')
            data_df[slide_key] = data_df[slide_key].str.removesuffix('.tiff')
            data_df[slide_key] = data_df[slide_key].str.removesuffix('.tif')            
        except:
            pass
        folds_list = data_df['fold']
        folds_list = pd.to_numeric(folds_list, errors='coerce').fillna(folds_list)
        data_df['fold'] = folds_list
        label_list = data_df[label]
        label_list = pd.to_numeric(label_list, errors='coerce').fillna(label_list)
        data_df[label] = label_list

        #filter based on dataset
        data_df = data_df[data_df['id'].isin(dataset_name)]
        #filter based on fold
        data_df = data_df[data_df['fold'].isin(folds)]
        if not test_on_all:
            data_df = data_df[pd.to_numeric(data_df[label], errors='coerce').notnull()]
        # get slides that have tile encodings
        valid_slides = self.get_valid_slides(root_path, data_df[slide_key].values)
        # filter out slides that do not have tile encodings
        data_df = data_df[data_df[slide_key].isin(valid_slides)]            
            
        # set up the task
        self.setup_data(data_df, task_config.get('setting', 'multi_class'))
        
        self.max_tiles = task_config.get('max_tiles', 1000)
        self.shuffle_tiles = task_config.get('shuffle_tiles', False)
        print('Dataset has been initialized!')
        
    def get_valid_slides(self, root_path: str, slides: list) -> list:
        '''This function is used to get the slides that have tile encodings stored in the tile directory'''
        valid_slides = []
        for i in range(len(slides)):
            sld = os.path.join(slides[i], "tile_embeds_"+slides[i]+".npy")
            sld_path = os.path.join(root_path, sld)
            if not os.path.exists(sld_path):
                print('Missing: ', sld_path)
            else:
                valid_slides.append(slides[i])
        return valid_slides
    
    def setup_data(self, df: pd.DataFrame, task: str='multi_class'):
        '''Prepare the data for multi-class setting or multi-label setting'''
        # Prepare slide data
        if task == 'multi_class' or task == 'binary':
            prepare_data_func = self.prepare_multi_class_or_binary_data
        elif task == 'multi_label':
            prepare_data_func = self.prepare_multi_label_data
        elif task == 'continuous':
            prepare_data_func = self.prepare_continuous_data
        else:
            raise ValueError('Invalid task: {}'.format(task))
        self.slide_data, self.images, self.labels, self.n_classes = prepare_data_func(df)
    
    def prepare_continuous_data(self, df: pd.DataFrame):
        '''Prepare the data for regression'''
        n_classes = 1
        
        images = df[self.slide_key].to_list()
        labels = df[[self.label]].to_numpy().astype(int)
        
        return df, images, labels, n_classes
    
    def prepare_multi_class_or_binary_data(self, df: pd.DataFrame):
        '''Prepare the data for multi-class classification'''
        # set up the label_dict
        label_dict = self.task_cfg.get('label_dict', {})
        assert  label_dict, 'No label_dict found in the task configuration'
        # set up the mappings
        assert self.label in df.columns, 'No label column found in the dataframe'
        df[self.label] = df[self.label].map(label_dict)
        n_classes = len(label_dict)
        
        images = df[self.slide_key].to_list()
        labels = df[[self.label]].to_numpy().astype(int)
        
        return df, images, labels, n_classes
        
    def prepare_multi_label_data(self, df: pd.DataFrame):
        '''Prepare the data for multi-label classification'''
        # set up the label_dict
        label_dict = self.task_cfg.get('label_dict', {})
        assert label_dict, 'No label_dict found in the task configuration'
        # Prepare mutation data
        label_keys = label_dict.keys()
        # sort key using values
        label_keys = sorted(label_keys, key=lambda x: label_dict[x])
        n_classes = len(label_dict)

        images = df[self.slide_key].to_list()
        labels = df[label_keys].to_numpy().astype(int)
            
        return df, images, labels, n_classes
    
    
class SlideDataset(SlideDatasetForTasks):
    def __init__(self,
                 data_df: pd.DataFrame,
                 root_path: str,
                 task_config: dict,
                 slide_key: str='file',
                 label: str='RS',
                 dataset_name: list= ['TCGA'],
                 folds: list=[1,2,3,4,5],
                 test_on_all: bool=False,
                 **kwargs
                 ):
        '''
        The slide dataset class for retrieving the slide data for different tasks

        Arguments:
        ----------
        data_df: pd.DataFrame
            The dataframe that contains the slide data
        root_path: str
            The root path of the tile embeddings
        task_config_path: dict
            The task configuration dictionary
        slide_key: str
            The key that contains the slide id
        '''
        super(SlideDataset, self).__init__(data_df, root_path, task_config, slide_key, label, dataset_name, folds, test_on_all, **kwargs)

    def shuffle_data(self, images: torch.Tensor, coords: torch.Tensor) -> tuple:
        '''Shuffle the serialized images and coordinates'''
        indices = torch.randperm(len(images))
        images_ = images[indices]
        coords_ = coords[indices]
        return images_, coords_
    
    def get_images_from_path(self, img_path: str) -> dict:
        '''Get the images from the path'''
        images = np.load(img_path) # load the numpy file
        images = torch.from_numpy(images) # convert to tensor  

        # We get the coordinates from file names in a sister dir
        # First we remove the .npy file name, like doing .. in cmd
        path = os.path.dirname(img_path).rsplit('gigapath_features', 1)
        try:
            coords_path = 'jpg_tiles'.join(path)

            # The file names contain the patch coordinates, so we list the coors_path dir
            coords_files = os.listdir(coords_path)
        except:
            #we want to support tiles being saved in png and in jpg
            coords_path = 'png_tiles'.join(path)
            coords_files = os.listdir(coords_path)
        # The coords files are named in the format <x coord>x_<y coord>y.jpg, so we split by x_ and y and convert to int
        coords = torch.tensor([[int(coord.split('x_')[0]), int(coord.split('x_')[1].split('y')[0])] for coord in coords_files])
        

        
        # set the input dict
        data_dict = {'imgs': images,
                'img_lens': images.size(0),
                'pad_mask': 0,
                'coords': coords}
        return data_dict
    
    def get_one_sample(self, idx: int) -> dict:
        '''Get one sample from the dataset'''
        # get the slide id
        slide_id = self.images[idx]
        # get the slide path
        slide_path = os.path.join(self.root_path, slide_id, "tile_embeds_"+slide_id+".npy")

        # get the slide images
        data_dict = self.get_images_from_path(slide_path)
            
        # get the slide label
        label = torch.from_numpy(self.labels[idx])
        # set the sample dict
        sample = {'imgs': data_dict['imgs'],
                  'img_lens': data_dict['img_lens'],
                  'pad_mask': data_dict['pad_mask'],
                  'coords': data_dict['coords'],
                  'slide_id': slide_id,
                  'labels': label}
        return sample
    
    def get_sample_with_try(self, idx, n_try=3):
        '''Get the sample with n_try'''
        for _ in range(n_try):
            try:
                sample = self.get_one_sample(idx)
                return sample
            except:
                print('Error in getting the sample, try another index')
                idx = np.random.randint(0, len(self.slide_data))
        print('Error in getting the sample, skip the sample')
        return None
        
        
    def __len__(self):
        return len(self.slide_data)
    
    def __getitem__(self, idx):
        sample = self.get_sample_with_try(idx)
        return sample
