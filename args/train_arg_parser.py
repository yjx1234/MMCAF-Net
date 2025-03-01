import util

from .base_arg_parser import BaseArgParser


class TrainArgParser(BaseArgParser):
    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True
        self.parser.add_argument('--epochs_per_save',type=int,default=1)
        self.parser.add_argument('--epochs_per_eval',type=int,default=1)
        self.parser.add_argument('--iters_per_print',type=int,required=True)
        self.parser.add_argument('--iters_per_visual',type=int,default=8000)
        self.parser.add_argument('--learning_rate',type=float,default=1e-2)
        self.parser.add_argument('--lr_scheduler',type=str,default='cosine_warmup',
                                    choices=('step','multi_step','plateau','cosine_warmup'))
        self.parser.add_argument('--lr_decay_step',type=int,default=600000)
        self.parser.add_argument('--lr_warmup_steps',type=int,default=10000)
        self.parser.add_argument('--num_epochs',type=int,default=100)
        self.parser.add_argument('--best_ckpt_metric',type=str,default='val_loss',choices=('val_loss','val_AUROC'))
        self.parser.add_argument('--optimizer',type=str,default='sgd',choices=('sgd','adam'))
        self.parser.add_argument('--weight_decay',type=float,default=1e-3)
        self.parser.add_argument('--use_pretrained',type=util.str_to_bool,default=False)
        self.parser.add_argument('--include_normals',type=util.str_to_bool,default=True)
        self.parser.add_argument('--fine_tune',type=util.str_to_bool,default=False)
        self.parser.add_argument('--fine_tuning_lr',type=float,default=0.01)
        self.parser.add_argument('--fine_tuning_boundary',type=str,default='classifier')
        #W 
        self.parser.add_argument('--adam_beta_1',type=float,default=0.9)
        self.parser.add_argument('--adam_beta_2',type=float,default=0.999)
        self.parser.add_argument('--do_center_pe',type=util.str_to_bool,default=True)
        self.parser.add_argument('--do_hflip',type=util.str_to_bool,default=True)
        self.parser.add_argument('--do_jitter',type=util.str_to_bool,default=True)
        self.parser.add_argument('--do_rotate',type=util.str_to_bool,default=True)
        self.parser.add_argument('--do_vflip',type=util.str_to_bool,default=False)
        self.parser.add_argument('--dropout_prob',type=float,default=0.0)
        self.parser.add_argument('--elastic_transform',type=util.str_to_bool,default=False)
        self.parser.add_argument('--hidden_dim',type=float,default=32)
        self.parser.add_argument('--sgd_dampening',type=float,default=0.9)
        self.parser.add_argument('--sgd_momentum',type=float,default=0.9)
        self.parser.add_argument('--patience',type=int,default=10)
        self.parser.add_argument('--max_eval',type=int,default=-1)
        self.parser.add_argument('--max_ckpts',type=int,default=2)
        self.parser.add_argument('--lr_milestones',type=str,default='50,125,250')
        self.parser.add_argument('--lr_decay_gamma',type=float,default=0.1)
        

        
