import os
import sys
from pathlib import Path

# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))

import time
import wandb
import torch
import numpy as np
import torch.utils.tensorboard as tensorboard
import pandas as pd
import gc
from PIL import Image

from gigapath.classification_head import get_model
from metrics import calculate_metrics_with_task_cfg
from utils import (get_optimizer, get_loss_function, \
                  Monitor_Score, get_records_array,
                  log_writer, adjust_learning_rate, 
                  get_regression_losses, LogCoshLoss, get_test_loader)
        

def test(test_loader, args):
    writer_dir = os.path.join(args.save_dir, 'tensorboard')
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir, exist_ok=True)

    # set up the writer
    writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)
    # set up writer
    if "wandb" in args.report_to:
        wandb.init(
            project='Finetune_Gigapath',
            name=args.exp_name,
            tags=[],
            config=vars(args),
        )
        writer = wandb
    elif "tensorboard" in args.report_to:
        writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)

    # set up the model
    model = get_model(**vars(args))
    model = model.to(args.device)
    # set up the loss function
    loss_fn = get_loss_function(args)
    # set up the fp16 scaler
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 training')
        
    
    print('Testing on {} samples'.format(len(test_loader.dataset)))
    
    test_records = None
    model.load_state_dict(torch.load(args.model_ckpt), strict = False)
    # test the model
    test_records, embeds = evaluate(test_loader, model, fp16_scaler, loss_fn, 'test', args, save_embed=True)
    results_df = pd.DataFrame({'slide_name': [], 'score': [], 'label': []})
    # save each embedding asa seperate filewith the name being the slide_id
    for idx, slide_id in enumerate(test_loader.dataset.slide_data[args.slide_key]):
        np.save(os.path.join(args.save_dir, f"{slide_id}.npy"), embeds[idx])
        if not args.task_config.get('setting', 'multi_class') == 'continuous':
            results = {'slide_name': slide_id, 'score': test_records['prob'][idx][1], 'label': 0 if test_records['label'][idx][0] == 1 else 1}
        else:
            results = {'slide_name': slide_id, 'score': test_records['prob'][idx][0], 'label': test_records['label'][idx][0]}
        results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
        
    results_df.to_csv(os.path.join(args.save_dir,'slide_scores.csv'), index=False)
    # update the writer for test
    log_dict = {'test_' + k: v for k, v in test_records.items() if 'prob' not in k and 'label' not in k}
    log_writer(log_dict, 0, args.report_to, writer)
    wandb.finish() if "wandb" in args.report_to else None
    return test_records

def train(dataloader, args):
    train_loader, val_loader, test_loader = dataloader
    # set up the writer
    writer_dir = os.path.join(args.save_dir, 'tensorboard')
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir, exist_ok=True)

    # set up the writer
    writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)
    # set up writer
    if "wandb" in args.report_to:
        wandb.init(
            project='Finetune_Gigapath',
            name=args.exp_name,
            tags=[],
            config=vars(args),
        )
        writer = wandb
    elif "tensorboard" in args.report_to:
        writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)

    # set up the model
    model = get_model(**vars(args))
    model = model.to(args.device)
    # set up the optimizer
    optimizer = get_optimizer(args, model)
    # set up the loss function
    loss_fn = get_loss_function(args)
    # set up the monitor
    monitor = Monitor_Score()
    # set up the fp16 scaler
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 training')

    print('Training on {} samples'.format(len(train_loader.dataset)))
    print('Validating on {} samples'.format(len(val_loader.dataset))) if val_loader is not None else None
    print('Testing on {} samples'.format(len(test_loader.dataset))) if test_loader is not None else None
    print('Training starts!')

    # test evaluate function
    # val_records = evaluate(val_loader, model, fp16_scaler, loss_fn, 0, args)

    val_records, test_records = None, None

    for i in range(args.epochs):
        print('Epoch: {}'.format(i))
        train_records = train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, i, args)

        if val_loader is not None:
            val_records = evaluate(val_loader, model, fp16_scaler, loss_fn, i, args)

            # update the writer for train and val
            try:
                log_dict = {'train_' + k: v for k, v in train_records.items() if 'prob' not in k and 'label' not in k}
                log_dict.update({'val_' + k: v for k, v in val_records.items() if 'prob' not in k and 'label' not in k})
                log_writer(log_dict, i, args.report_to, writer)
            except:
                for key in records:
                    print(key)
                    print(records[key])
                    try:
                        print(records[key].shape)
                    except:
                        pass
                raise
            # update the monitor scores
            if not args.task_config.get('setting', 'multi_class') == 'continuous':
                scores = val_records['macro_auroc']
            else:
                #the better loss is the lower one but we want to choose the epoch with the highest score
                scores = -val_records['loss']

        if args.model_select == 'val' and val_loader is not None:
            monitor(scores, model, ckpt_name=os.path.join(args.save_dir, "checkpoint.pt"))
        elif args.model_select == 'last_epoch' and i == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "checkpoint.pt"))
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_epoch_'+str(i)+ '_checkpoint.pt'))

    # load model for test
    if test_loader is not None:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "checkpoint.pt")), strict = False)
        if args.task_config.get('setting', 'multi_class') == 'continuous':
            print("in test time labels mean: %.2e" % model.mean.item())
            print("in test time labels std: %.2e" % model.std.item())
        # test the model
        test_records, embeds = evaluate(test_loader, model, fp16_scaler, loss_fn, i, args, save_embed=True)
        # save each embedding as a seperate file with the name being the slide_id
        for idx, slide_id in enumerate(test_loader.dataset.slide_data[args.slide_key]):
            np.save(os.path.join(args.save_dir, f"{slide_id}.npy"), embeds[idx])
        # update the writer for test
        log_dict = {'test_' + k: v for k, v in test_records.items() if 'prob' not in k and 'label' not in k}
        log_writer(log_dict, 0, args.report_to, writer)
    wandb.finish() if "wandb" in args.report_to else None

    return val_records, test_records


def train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, epoch, args):
    model.train()
    # set the start time
    start_time = time.time()

    # monitoring sequence length
    seq_len = 0
    max_num_samples = 40000

    # setup the records
    records = get_records_array(len(train_loader), args.n_classes, args)
    for batch_idx, batch in enumerate(train_loader):
        # we use a per iteration lr scheduler
        if batch_idx % args.gc == 0 and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch, args)
        num_tiles = batch['imgs'].shape[1]
        if num_tiles > max_num_samples:
            print(f"slide {batch['slide_id']} has too many tiles, number of tiles is {num_tiles}. Trying with {max_num_samples} tiles.")
            indices = torch.randint(low=0, high=num_tiles, size=(max_num_samples,))
            batch['imgs'] = batch['imgs'][:,indices,:]
            batch['coords'] = batch['coords'][:,indices,:]
            
        # load the batch and transform this batch
        images, img_coords, label = batch['imgs'], batch['coords'], batch['labels']
        images = images.to(args.device, non_blocking=True)
        img_coords = img_coords.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True).long()

        # add the sequence length
        seq_len += images.shape[1]

        with torch.cuda.amp.autocast(dtype=torch.float16 if args.fp16 else torch.float32):

            # get the logits
            logits = model(images, img_coords)
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                label = label.squeeze(-1).float()
            elif args.task_config.get('setting', 'multi_class') == 'continuous':
                label = label.squeeze(-1).float()
                logits = logits.squeeze(-1)
            else:
                label = label.squeeze(-1).long()
            if args.task_config.get('setting', 'multi_class') == 'continuous':
                label = (label - model.mean.item())/model.std.item()
            loss = loss_fn(logits, label)
            loss /= args.gc
            if fp16_scaler is None:
                loss.backward()
                # update the parameters with gradient accumulation
                if (batch_idx + 1) % args.gc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                fp16_scaler.scale(loss).backward()
                # update the parameters with gradient accumulation
                if (batch_idx + 1) % args.gc == 0:
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                    optimizer.zero_grad()

        # update the records
        records['loss'] += loss.item() * args.gc
            
            
        if (batch_idx + 1) % 20 == 0:
            time_per_it = (time.time() - start_time) / (batch_idx + 1)
            print('Epoch: {}, Batch: {}, Loss: {:.4f}, LR: {:.4f}, Time: {:.4f} sec/it, Seq len: {:.1f}, Slide ID: {}' \
                  .format(epoch, batch_idx, records['loss']/batch_idx, optimizer.param_groups[0]['lr'], time_per_it, \
                          seq_len/(batch_idx+1), batch['slide_id'][-1] if 'slide_id' in batch else 'None'))

    records['loss'] = records['loss'] / len(train_loader)
    print('Epoch: {}, Train Loss: {:.4f}'.format(epoch, loss))
    return records


def evaluate(loader, model, fp16_scaler, loss_fn, epoch, args, save_embed=False):
    model.eval()

    # set the evaluation records
    records = get_records_array(len(loader), args.n_classes, args)
    embeds = np.zeros((len(loader), 768))
    # get the task setting
    task_setting = args.task_config.get('setting', 'multi_class')
    if task_setting == 'continuous':
        regression_losses = get_regression_losses()
        for key in regression_losses:
            records[key] = 0
    with torch.no_grad():
        no_loss = False
        for batch_idx, batch in enumerate(loader):
            # load the batch and transform this batch
            num_tiles = batch['imgs'].shape[1]
            images, img_coords, label = batch['imgs'], batch['coords'], batch['labels']
            images = images.to(args.device, non_blocking=True)
            img_coords = img_coords.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True).long()

            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                # get the logits
                if save_embed:
                    logits, embed = model(images, img_coords, return_embed=True)
                    embeds[batch_idx] = embed.cpu().numpy()
                else:
                    logits = model(images, img_coords)
                # get the loss
                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    label = label.squeeze(-1).float()
                elif task_setting == 'continuous':
                    label = label.squeeze(-1).float()
                    logits = logits.squeeze(-1)
                else:
                    label = label.squeeze(-1).long()
                if task_setting == 'continuous':
                    label = (label - model.mean.item())/model.std.item()
                if not no_loss:
                    try:
                        loss = loss_fn(logits, label)
                    except:
                        if not loader.dataset.test_on_all:
                            raise
                        no_loss = True
                        print("Evaluating model on slides with no label so loss will be meaningless")

            # update the records
            if not no_loss:
                records['loss'] += loss.item()
            if task_setting == 'multi_label':
                Y_prob = torch.sigmoid(logits)
                records['prob'][batch_idx] = Y_prob.cpu().numpy()
                records['label'][batch_idx] = label.cpu().numpy()
            elif task_setting == 'multi_class' or task_setting == 'binary':
                Y_prob = torch.softmax(logits, dim=1).cpu()
                records['prob'][batch_idx] = Y_prob.numpy()
                # convert label to one-hot
                label_ = torch.zeros_like(Y_prob).scatter_(1, label.cpu().unsqueeze(1), 1)
                records['label'][batch_idx] = label_.numpy()
            elif task_setting == 'continuous':
                records['prob'][batch_idx] = ((logits*model.std.item())+model.mean.item()).cpu().numpy()
                records['label'][batch_idx] = ((label*model.std.item())+model.mean.item()).cpu().numpy()
                if not no_loss:
                    for key in regression_losses:
                        records[key] += regression_losses[key](logits, label).item()
                    
    
    
    if not no_loss:
        records['loss'] = records['loss'] / len(loader)
        if task_setting == 'continuous':
            for key in regression_losses:
                records[key] = records[key] / len(loader)
    else:
        records['loss'] = 0
        for key in regression_losses:
            records[key] = 0
    try:
        if not task_setting == 'continuous':
            records.update(calculate_metrics_with_task_cfg(records['prob'], records['label'], args.task_config))

        if task_setting == 'multi_label':
            info = 'Epoch: {}, Loss: {:.4f}, Micro AUROC: {:.4f}, Macro AUROC: {:.4f}, Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}'.format(epoch, records['loss'], records['micro_auroc'], records['macro_auroc'], records['micro_auprc'], records['macro_auprc'])
        elif task_setting == 'multi_class' or task_setting == 'binary':
            info = 'Epoch: {}, Loss: {:.4f}, AUROC: {:.4f}, ACC: {:.4f}, BACC: {:.4f}'.format(epoch, records['loss'], records['macro_auroc'], records['acc'], records['bacc'])
            for metric in args.task_config.get('add_metrics', []):
                info += ', {}: {:.4f}'.format(metric, records[metric])
        else:
            info = 'Epoch: {},'.format(epoch)
            for key in regression_losses:
                info += ' Eval {} Loss: {:.4f}'.format(key, records[key])
        print(info)
    except:
        print("Failed to get metrics, this should only happen on test set.")

    # return the embeddings
    if save_embed:
        return records, embeds
    return records