import os
import torch
import pandas as pd
import numpy as np

from training import train, test
from params import get_finetune_params
from task_configs.utils import load_task_config
from utils import seed_torch, get_exp_code, get_loader, save_obj, get_test_loader
from datasets.slide_datatset import SlideDataset


if __name__ == '__main__':
    args = get_finetune_params()
    print(args)
    # Set the hf token
    os.environ["HF_TOKEN"] = args.hf_token

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # set the random seed
    seed_torch(device, args.seed)

    # load the task configuration
    print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config = load_task_config(args.task_cfg_path)
    print(args.task_config)
    args.task = args.task_config.get('name', 'task')

    args.save_dir = os.path.join(args.save_dir, args.task, args.exp_name)
    args.model_code, args.task_code, args.exp_code = get_exp_code(args) # get the experiment code
    args.save_dir = os.path.join(args.save_dir, args.exp_code)
    os.makedirs(args.save_dir, exist_ok = True)
    print('Experiment code: {}'.format(args.exp_code))
    print('Setting save directory: {}'.format(args.save_dir))

    # set the learning rate
    if not args.run_inference:
        eff_batch_size = args.batch_size * args.gc
        if args.lr is None or args.lr < 0:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.gc)
        print("effective batch size: %d" % eff_batch_size)

    args.slide_key = 'file'

    # set up the dataset
    dataset = pd.read_csv(args.dataset_csv) # read the dataset csv file

    # use the slide dataset
    DatasetClass = SlideDataset

    # set up the results dictionary
    results = {}

    # start cross validation
    if not args.run_inference:
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        # instantiate the dataset
        train_data, val_data, test_data = DatasetClass(dataset, args.root_path, args.task_config, slide_key=args.slide_key, label = args.label, \
                                                       dataset_name = args.train_dataset, folds = args.train_fold) \
                                        , DatasetClass(dataset, args.root_path, args.task_config, slide_key=args.slide_key, label = args.label, \
                                                       dataset_name = args.val_dataset, folds = args.val_fold, test_on_all = args.test_on_all) \
                                        if len(args.val_dataset) > 0 else None \
                                        , DatasetClass(dataset, args.root_path, args.task_config, slide_key=args.slide_key, label = args.label, \
                                                       dataset_name = args.test_dataset, folds = args.test_fold, test_on_all = args.test_on_all) \
                                        if len(args.test_dataset) > 0 else None
        #scale the lr in case of regression
        if args.task_config.get('setting', 'multi_class') == 'continuous':
            args.mean = train_data.labels.mean()
            args.std = train_data.labels.std()
        args.n_classes = train_data.n_classes # get the number of classes
        # get the dataloader
        train_loader, val_loader, test_loader = get_loader(train_data, val_data, test_data, **vars(args))
        # start training
        val_records, test_records = train((train_loader, val_loader, test_loader), args)

        # update the results
        records = {}
        if val_records is not None:
            records['val'] = val_records
        if test_records is not None:
            records['test'] = test_records
        for record_ in records:
            for key in records[record_]:
                if 'prob' in key or 'label' in key:
                    continue
                key_ = record_ + '_' + key
                if key_ not in results:
                    results[key_] = []
                results[key_].append(records[record_][key])
    else:
        save_dir = os.path.join(args.save_dir, f'inference_results')
        os.makedirs(save_dir, exist_ok=True)
        args.save_dir = save_dir
        test_data = DatasetClass(dataset, args.root_path, args.task_config, slide_key=args.slide_key, label = args.label, \
                                 dataset_name = args.test_dataset, folds = args.test_fold, \
                                 test_on_all = args.test_on_all)
        args.n_classes = test_data.n_classes
        test_loader = get_test_loader(test_data, **vars(args))

        test_records = test(test_loader, args)
        records = {'test': test_records}
        for record_ in records:
            for key in records[record_]:
                if 'prob' in key or 'label' in key:
                    continue
                key_ = record_ + '_' + key
                if key_ not in results:
                    results[key_] = []
                results[key_].append(records[record_][key])


    # save the results into a csv file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.save_dir, 'summary.csv'), index=False)

    # print the results, mean and std
    for key in results_df.columns:
        print('{}: {:.4f} +- {:.4f}'.format(key, np.mean(results_df[key]), np.std(results_df[key])))
    print('Results saved in: {}'.format(os.path.join(args.save_dir, 'summary.csv')))
    print('Done!')
