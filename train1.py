
import models
import torch
import torch.nn as nn
import util
from torch.autograd import Variable

from args import TrainArgParser
from evaluator import ModelEvaluator1
from logger import TrainLogger
from saver import ModelSaver
from pkl_read import CTPE

###
import pandas as pd
import numpy as np
import sys

#w
import data_loader

from PIL import Image
def save_2d_slice(data, count):
    data = data.detach().cpu().numpy()
    output_path = str(count) + '.png'
    slice_index = data.shape[2] // 2
    slice_2d = data[0, 0, slice_index, :, :]
    slice_2d = slice_2d.astype(np.float32)
    slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255
    slice_2d = slice_2d.astype(np.uint8)
    #w
    img = Image.fromarray(slice_2d, mode = 'L')
    img.save(output_path)



###
def train(args,table):
    #w
    count = 0
    much = 50

    if args.ckpt_path and not args.use_pretrained:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        ###
        model = model_fn(**vars(args))
        if args.use_pretrained:
            model.load_pretrained(args.ckpt_path, args.gpu_ids)
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Get optimizer and scheduler
    if args.use_pretrained or args.fine_tune:
        parameters = model.module.fine_tuning_parameters(args.fine_tuning_boundary, args.fine_tuning_lr)
    else:
        parameters = model.parameters()
    optimizer = util.get_optimizer(parameters, args)
    lr_scheduler = util.get_scheduler(optimizer, args)

    if args.ckpt_path and not args.use_pretrained and not args.fine_tune:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)
        # ModelSaver.load_optimizer(args.ckpt_path, optimizer, D_lr_scheduler)

    # Get logger, evaluator, saver
    cls_loss_fn = util.get_loss_fn(is_classification=True, dataset=args.dataset, size_average=False)

    #w
    data_loader_fn = data_loader.__dict__[args.data_loader]
    train_loader = data_loader_fn(args, phase = 'train', is_training = True)




    logger = TrainLogger(args, len(train_loader.dataset), train_loader.dataset.pixel_dict)
    #w
    eval_loader = [data_loader_fn(args, phase = 'val', is_training = False)]


    evaluator = ModelEvaluator1(args.dataset, eval_loader, args.agg_method, args.epochs_per_eval)
    saver = ModelSaver(args.save_dir, args.epochs_per_save, args.max_ckpts, args.best_ckpt_metric, args.maximize_metric)

    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()

        for img, target_dict in train_loader:
            logger.start_iter()
            
            ###
            ids = [item for item in target_dict['study_num']]
            tab=[]
            for i in range(len(target_dict['study_num'])):
                data = table[table['NewPatientID'] == ids[i]].iloc[0, 1:8].astype(np.float32)
                tab.append(torch.tensor(data, dtype=torch.float32))
            tab = torch.stack(tab).squeeze(1)
 
            with torch.set_grad_enabled(True):
                #w
                img = img.to(args.device)
                tab = tab.to(args.device)
                label = target_dict['is_abnormal'].to(args.device)
                #w
   
                
                #out = model.forward(img, tab)
                #tab=torch.randn(3,7).to(args.device)
                #img=torch.randn(3,1,12,192,192).to(args.device)
                #label=torch.tensor([1,1,1]).to(args.device)
                #out,img_loss,tab_loss = model.forward(img,tab,target_dict['bbox'],label)
                out = model.forward(img,tab)
                #print(target_dict['bbox'],type(target_dict['bbox']))
                #raise ValueError("self.preds是{}".format(self.preds))
                #f,out = model.forward(img)
                label = label.unsqueeze(1)
                cls_loss = cls_loss_fn(out, label).mean()
                loss = cls_loss

                if count != 0 and count % much == 0:
                    pass
                count += 1

                #w 三步走
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        
                ###Log
                logger.log_iter(img,None,None,loss,optimizer)

            logger.end_iter()
            util.step_scheduler(lr_scheduler,global_step=logger.global_step)
            # util.step_scheduler(D_lr_scheduler,global_step=logger.global_step)

        ###
        metrics,curves,avg_loss=evaluator.evaluate(model,args.device,logger.epoch)
        saver.save(logger.epoch,model,optimizer,lr_scheduler,args.device,
                    metric_val=avg_loss)#W metrics.get(args.best_ckpt_metric,None)  avg_loss
        print(metrics)
        logger.end_epoch(metrics,curves)
        util.step_scheduler(lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)
        #util.step_scheduler(D_lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)

###
if __name__ == '__main__':
    ###
    tab = pd.read_csv('/home/vesselseg3/publicdata/Lung-pet-ct/G_first_last_nor.csv')

    util.set_spawn_enabled()
    parser = TrainArgParser()
    args_ = parser.parse_args()
    ###
    train(args_,tab)
