import numpy as np
import random
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
import util

from tqdm import tqdm
from .output_aggregator import OutputAggregator

###
import pandas as pd
import numpy as np
import torch.nn as nn


class ModelEvaluator1(object):
    def __init__(self, 
                 dataset_name, 
                 data_loaders, 
                 agg_method = None, 
                 epochs_per_eval = 1):


        self.aggregator=OutputAggregator(agg_method, num_bins=10, num_epochs=5)
        
        self.data_loaders=data_loaders
        self.dataset_name=dataset_name
        self.epochs_per_eval=epochs_per_eval #w
        self.cls_loss_fn= util.optim_util.get_loss_fn(is_classification=True, dataset=dataset_name)
        self.max_eval=None 



    def evaluate(self, model, device, epoch=None):

        #w
        tab = pd.read_csv('/home/vesselseg3/publicdata/Lung-pet-ct/G_first_last_nor.csv')
        metrics, curves={}, {}



        #w 还不确定self.data_loaders是不是有多个元素
        sum_loss = []

        model.eval()
        for data_loader in self.data_loaders:
            print('data_loader1次')
            phase_metrics, phase_curves, sum_every_loss = self._eval_phase(model, data_loader, data_loader.phase, device,tab, epoch)
            metrics.update(phase_metrics)
            curves.update(phase_curves)
            #w
            sum_loss.append(sum_every_loss)
            
        model.train()
        #w
        eval_loss = sum(sum_loss) / len(sum_loss)
        # raise ValueError("eval_loss是{}".format(eval_loss))
        print('eval_loss:', eval_loss)
        ###
        return metrics,curves, eval_loss

    ###
    def _eval_phase(self, model, data_loader, phase, device,table, epoch):
        #w
        out = None





        """Evaluate a model for a single phase.

        Args:
            model: Model to evaluate.
            data_loader: Torch DataLoader to sample from.
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            device: Device on which to evaluate the model.
        """
        batch_size=data_loader.batch_size

        # Keep track of task-specific records needed for computing overall metrics
    
        records={'keys': [], 'probs': []}
        


     
        num_examples=len(data_loader.dataset)
      

        # Sample from the data loader and record model outputs
        num_evaluated=0

        with tqdm(total=num_examples, unit=' ' + phase) as progress_bar:
            #w
            sum_every_loss = 0

            for img, targets_dict in data_loader:
                if num_evaluated >=num_examples:
                    break

                ###
                ids = [item for item in targets_dict['study_num']]

                tab=[]
                for i in range(len(targets_dict['study_num'])):
                    data = table[table['NewPatientID'] == ids[i]].iloc[0, 1:8].astype(np.float32)
                    tab.append(torch.tensor(data, dtype=torch.float32))
                tab=torch.stack(tab).squeeze(1)

                with torch.no_grad():


                    #w process data
                    img = img.to(device)
                    tab = tab.to(device)
                    label = targets_dict['is_abnormal'].to(device)

                    
                    #w forward
                    #f,out = model.forward(img)
                    out = model.forward(img,tab)
                    label = label.unsqueeze(1)
                    cls_loss = self.cls_loss_fn(out, label).mean()
                    loss = cls_loss
                    #w
                    sum_every_loss += loss.item()
                    cls_logits = out if out is not None else torch.randn([4, 1])


                    



                #w
                self._record_batch(cls_logits,targets_dict['series_idx'],loss,**records)



                progress_bar.update(min(batch_size, num_examples - num_evaluated))
                num_evaluated +=batch_size



        #Map to summary dictionaries
        metrics, curves=self._get_summary_dicts(data_loader, phase, device, **records)

        ###
        return metrics, curves, sum_every_loss

    @staticmethod
    def _record_batch(logits, targets, loss, probs=None, keys=None, loss_meter=None):
        """Record results from a batch to keep track of metrics during evaluation.

        Args:
            logits: Batch of logits output by the model.
            targets: Batch of ground-truth targets corresponding to the logits.
            probs: List of probs from all evaluations.
            keys: List of keys to map window-level logits back to their series-level predictions.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
        """
        if probs is not None:
            assert keys is not None, 'Must keep probs and keys lists in parallel'
            with torch.no_grad():
                batch_probs=F.sigmoid(logits)
            probs.append(batch_probs.detach().cpu())

            #Note: `targets` is assumed to hold the keys for these examples
            keys.append(targets.detach().cpu())
        

    def _get_summary_dicts(self, data_loader, phase, device, probs=None, keys=None, loss_meter=None):
        """Get summary dictionaries given dictionary of records kept during evaluation.

        Args:
            data_loader: Torch DataLoader to sample from.
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            device: Device on which to evaluate the model.
            probs: List of probs from all evaluations.
            keys: List of keys to map window-level logits back to their series-level predictions.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
        Returns:
            metrics: Dictionary of metrics for the current model.
            curves: Dictionary of curves for the current model. E.g. ROC.
        """
        metrics, curves={}, {}

        if probs is not None:
            # If records kept track of individual probs and keys, implied that we need to aggregate them
            assert keys is not None, 'Must keep probs and keys lists in parallel.'
            assert self.aggregator is not None, 'Must specify an aggregator to aggregate probs and keys.'

            # Convert to flat numpy array
            probs=np.concatenate(probs).ravel().tolist()
            keys=np.concatenate(keys).ravel().tolist()

            # Aggregate predictions across each series
            idx2prob=self.aggregator.aggregate(keys, probs, data_loader, phase, device)
            probs, labels=[], []
            for idx, prob in idx2prob.items():
                probs.append(prob)
                labels.append(data_loader.get_series_label(idx))
            probs, labels=np.array(probs), np.array(labels)

            # Update summary dicts
            metrics.update({
                phase + '_' + 'loss': sk_metrics.log_loss(labels, probs, labels=[0, 1])
            })

            # Update summary dicts
            try:
                metrics.update({
                    phase + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, probs),
                    phase + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, probs),
                })
                curves.update({
                    phase + '_' + 'PRC': sk_metrics.precision_recall_curve(labels, probs),
                    phase + '_' + 'ROC': sk_metrics.roc_curve(labels, probs)
                })
            except ValueError:
                pass



        return metrics, curves
