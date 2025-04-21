import argparse
import ast


def get_finetune_params():
    def parse_list(input_string):
        return ast.literal_eval(input_string)

    parser = argparse.ArgumentParser(description='Finetune on downstream tasks')

    # task settings
    parser.add_argument('--task_cfg_path',  type=str, default='finetune/task_configs/continuous.yaml', help='Path to the task configuration file')
    parser.add_argument('--exp_name',       type=str, default='', help='Experiment name')
    parser.add_argument('--hf_token',       type=str, default='', help='Hugging face token, needed to load the prov-gigapath model')

    # input data settings
    parser.add_argument('--dataset_csv',    type=str, default='./data/meta.csv', help='Dataset csv file')
    parser.add_argument('--root_path',      type=str, default='', help='The tile encodings path')
    parser.add_argument('--tile_size',      type=int, default=256, help='Tile size in pixels')
    parser.add_argument('--max_wsi_size',   type=int, default=250000, help='Maximum WSI size in pixels for the longer side (width or height).')
    parser.add_argument('--label',          type=str, default='RS', help='Column name of the label')
    parser.add_argument('--train_dataset',  type=parse_list, default=['TCGA'], help='Name of datasets to train on')
    parser.add_argument('--val_dataset',    type=parse_list, default=['TCGA'], help='Name of datasets to validate on')
    parser.add_argument('--test_dataset',   type=parse_list, default=['TCGA'], help='Name of datasets to test on')
    parser.add_argument('--train_fold',     type=parse_list, default=[2,3,4,5], help='Folds to train on')
    parser.add_argument('--val_fold',       type=parse_list, default=[1], help='Folds to validate on')
    parser.add_argument('--test_fold',      type=parse_list, default=[6], help='Folds to test on')

    # model settings
    parser.add_argument('--model_arch',     type=str, default='gigapath_slide_enc12l768d')
    parser.add_argument('--input_dim',      type=int, default=1536, help='Dimension of input tile embeddings')
    parser.add_argument('--latent_dim',     type=int, default=768, help='Hidden dimension of the slide encoder')
    parser.add_argument('--feat_layer',     type=str, default='11', help='The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers')
    parser.add_argument('--pretrained',     type=str, default='hf_hub:prov-gigapath/prov-gigapath', help='Pretrained GigaPath slide encoder')
    parser.add_argument('--freeze',         action='store_true', default=False, help='Freeze pretrained model')
    parser.add_argument('--global_pool',    action='store_true', default=False, help='Use global pooling, will use [CLS] token if False')
    parser.add_argument('--model_ckpt',     type=str, default='', help='Model checkpoint path')

    # training settings
    parser.add_argument('--seed',           type=int, default=0, help='Random seed')
    parser.add_argument('--epochs',         type=int, default=5, help='Number of training epochs')
    parser.add_argument('--warmup_epochs',  type=int, default=1, help='Number of warmup epochs')
    parser.add_argument('--batch_size',     type=int, default=1, help='Current version only supports batch size of 1')
    parser.add_argument('--lr',             type=float, default=None, help='Learning rate')
    parser.add_argument('--blr',            type=float, default=0.002, help='Base learning rate, will caculate the learning rate based on batch size')
    parser.add_argument('--min_lr',         type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--lr_scheduler',   type=str, default='cosine', help='Learning rate scheduler', choices=['cosine', 'fixed'])
    parser.add_argument('--gc',             type=int, default=32, help='Gradient accumulation')
    parser.add_argument('--optim',          type=str, default='adamw', help='Optimizer', choices=['adam', 'adamw'])
    parser.add_argument('--optim_wd',       type=float, default=0.05, help='Weight decay')
    parser.add_argument('--layer_decay',    type=float, default=0.95, help='Layer-wise learning rate decay')
    parser.add_argument('--dropout',        type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0, help='Drop path rate')
    parser.add_argument('--model_select',   type=str, default='last_epoch', help='Criteria for choosing the model checkpoint', choices=['val', 'last_epoch'])
    parser.add_argument('--save_dir',       type=str, default='', help='Save directory')
    parser.add_argument('--num_workers',    type=int, default=20, help='Number of workers')
    parser.add_argument('--report_to',      type=str, default='tensorboard', help='Logger used for recording', choices=['wandb', 'tensorboard'])
    parser.add_argument('--loss_fn',        type=str, default='mse', help='Which loss to use in continuous setting', choices=['mse', 'mae', 'huber', 'logcosh'])
    parser.add_argument('--fp16',           action='store_true', default=True, help='Fp16 training')
    parser.add_argument('--test_on_all',    action='store_true', default=False, help='Test also on slides without label')
    parser.add_argument('--run_inference',  action='store_true', default=False, help='Only run inference')

    return parser.parse_args()
