import argparse
import datetime
import json
import numpy as np
import os
import random
import torch
import torch.backends.cudnn as cudnn
import util


class BaseArgParser(object):
    def __init__(self):
        self.parser=argparse.ArgumentParser(description='')
        #w
        self.parser.add_argument('--model',type=str,default='PENet',
                                    choices=('PENet','Img_new','MMCAF_Net'))
        #W
        self.parser.add_argument('--batch_size',type=int,default=4)
        self.parser.add_argument('--model_depth',type=int,default=50)
        self.parser.add_argument('--num_classes',type=int,default=1)
        self.parser.add_argument('--num_slices',type=int,default=24)
        self.parser.add_argument('--num_visuals',type=int,default=8)
        self.parser.add_argument('--num_workers',type=int,default=4)
        self.parser.add_argument('--resize_shape',type=str,default='208,208')
        self.parser.add_argument('--crop_shape',type=str,default='192,192')
        #W
        self.parser.add_argument('--phase',type=str,default='val',choices=('train','val','test'))
        self.parser.add_argument('--ckpt_path',type=str,default='')
        self.parser.add_argument('--data_dir',type=str,required=True)
        self.parser.add_argument('--pkl_path',type=str,default='/home/vesselseg3/publicdata/Lung-pet-ct/series_list_last_AG.pkl')
        self.parser.add_argument('--gpu_ids',type=str, default='0')
        self.parser.add_argument('--name',type=str,required=True)
        self.parser.add_argument('--save_dir',type=str,default='../ckpts/')
        self.parser.add_argument('--dataset',type=str,required=True,choices=('kinetics','pe'))
        self.parser.add_argument('--cudnn_benchmark',type=util.str_to_bool,default=False)
        self.parser.add_argument('--do_classify',type=util.str_to_bool,default=True)
        #w 2024.5.13
        self.parser.add_argument('--pe_types', type=eval, default='["central","segmental"]')
        self.is_training=None

        #W 原选择序列为('max','mean','logreg','')
        self.parser.add_argument('--agg_method',type=str,default='max')
        self.parser.add_argument('--deterministic',type=util.str_to_bool,default=True)
        #W 原选择序列为('window','series')和('window','image')
        self.parser.add_argument('--eval_mode',type=str,default='series')
        self.parser.add_argument('--hide_level',type=str,default='window')
        self.parser.add_argument('--hide_probability',type=float,default=0.0)
        #W 原选择序列为('raw','png')
        self.parser.add_argument('--img_format',type=str,default='raw')
        #W 原选择序列为('kaiming','normal','xavier')
        self.parser.add_argument('--init_method',type=str,default='kaiming')
        self.parser.add_argument('--vstep_size',type=int,default=1)
        self.parser.add_argument('--toy',type=util.str_to_bool,default=False)
        self.parser.add_argument('--toy_size',type=int,default=5)

        #W 原选择序列为('sagittal','axial','coronal')
        self.parser.add_argument('--series',type=str,default='sagittal')
        self.parser.add_argument('--only_topmost_window',type=util.str_to_bool,default=False)
        self.parser.add_argument('--num_channels',type=int,default=3)
        self.parser.add_argument('--min_abnormal_slices',type=int,default=4)

    def parse_args(self):
        args =self.parser.parse_args()

        # Save args to a JSON file
        date_string =datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir =os.path.join(args.save_dir, '{}_{}'.format(args.name, date_string))
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')
        args.save_dir =save_dir

        # Add configuration flags outside of the CLI
        args.is_training =self.is_training
        args.start_epoch =1  # Gets updated if we load a checkpoint
        if not args.is_training and not args.ckpt_path and not (hasattr(args, 'test_2d') and args.test_2d):
            raise ValueError('Must specify --ckpt_path in test mode.')
        if args.is_training and args.epochs_per_save % args.epochs_per_eval !=0:
            raise ValueError('epochs_per_save must be divisible by epochs_per_eval.')
        if args.is_training:
            args.maximize_metric =not args.best_ckpt_metric.endswith('loss')
            if args.lr_scheduler =='multi_step':
                args.lr_milestones =util.args_to_list(args.lr_milestones, allow_empty=False)
        if not args.pkl_path:
            args.pkl_path =os.path.join(args.data_dir, 'series_list_last_AG.pkl')

        # Set up resize and crop
        args.resize_shape =util.args_to_list(args.resize_shape, allow_empty=False, arg_type=int, allow_negative=False)
        args.crop_shape =util.args_to_list(args.crop_shape, allow_empty=False, arg_type=int, allow_negative=False)

        # Set up available GPUs
        args.gpu_ids =util.args_to_list(args.gpu_ids, allow_empty=True, arg_type=int, allow_negative=False)
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            args.device ='cuda'
            cudnn.benchmark =args.cudnn_benchmark
        else:
            args.device ='cpu'

        #W 输出以确定是否是固定运行，默认是
        if args.deterministic:
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            cudnn.deterministic=True
            print('固定种子')

        # Map dataset name to a class
        if args.dataset =='kinetics':
            args.dataset ='KineticsDataset'
        elif args.dataset =='pe':
            args.dataset ='CTPEDataset3d'

        if self.is_training and args.use_pretrained:
            if args.model !='PENet' and args.model !='PENetClassifier':
                raise ValueError('Pre-training only supported for PENet/PENetClassifier loading PENetClassifier.')
            if not args.ckpt_path:
                raise ValueError('Must specify a checkpoint path for pre-trained model.')

        args.data_loader ='CTDataLoader'
        if args.model =='PENet':
            if args.model_depth !=50:
                raise ValueError('Invalid model depth for PENet: {}'.format(args.model_depth))
            args.loader = 'window'


        #w 加上
        if args.model_depth !=50:
            raise ValueError('Invalid model depth for PENet: {}'.format(args.model_depth))
        args.loader = 'window'
        if args.dataset =='KineticsDataset':
            args.data_loader ='KineticsDataLoader'

        # Set up output dir (test mode only)
        if not self.is_training:
            args.results_dir =os.path.join(args.results_dir, '{}_{}'.format(args.name, date_string))
            os.makedirs(args.results_dir, exist_ok=True)

        return args
