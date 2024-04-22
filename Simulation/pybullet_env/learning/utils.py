import time

import torch as th
import torch.nn as nn
import torch.optim as Optim
from typing import List, Dict, Callable, NamedTuple, Tuple
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Trainer():
    """
    Generic trainer for a pytorch nn.Module.
    Intended to be flexible, modify as needed.
    """
    def __init__(self,
                 config,
                 data_loader: th.utils.data.DataLoader,
                 model: th.nn.Module,
                 optimizer: th.optim.Optimizer,
                 loss_fn: Callable[[Dict[str, th.Tensor], Dict[str, th.Tensor]], Dict[str, th.Tensor]],
                 eval_fn: Callable = None,
                 scheduler: th.optim.lr_scheduler = None
                 ):
        """
        Args:
            config: Trainer options.
            model: The model to train.
            optimizer: Optimizer, e.g. `Adam`.
            loss_fn: The function that maps (model, next(iter(loader))) -> cost.
            loader: Iterable data loader.
        """
        self.config = config
        self.data_loader = data_loader
        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn

    def train(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses_elbo = AverageMeter('ELBO', ':4e')
        losses_recon = AverageMeter('Reconstruction Error', ':4e')
        losses_kld = AverageMeter('KL-divergence', ':.4e')
        vals_elbo = AverageMeter('ELBO', ':4e')
        vals_recon = AverageMeter('Reconstruction Error', ':4e')
        vals_kld = AverageMeter('KL-divergence', ':.4e')

        progress = ProgressMeter(len(self.data_loader),
                                    [batch_time, losses_elbo, vals_elbo],
                                    prefix="Epoch: [{}]".format(epoch))

        self.model.train()

        end = time.time()
        for i, data in enumerate(self.data_loader):

            losses = {}
            recon_x, mean, log_var, z = self.model(data['belief'], data['goal'], data['target_action'])
            loss = self.loss_fn(recon_x, data['target_action'], mean, log_var)

            losses_elbo.update(loss['total'].item(), data['target_action'].size(0))
            losses_recon.update(loss['Recon'].item(), data['target_action'].size(0))
            losses_kld.update(loss['KL_div'].item(), data['target_action'].size(0))
        
            losses['total'] = losses_elbo.avg
            losses['Recon'] = losses_recon.avg
            losses['KL_div'] = losses_kld.avg

            # Backprop + Optimize ...
            self.optim.zero_grad()
            loss['total'].backward()
            # th.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optim.step()

            if self.scheduler is not None:
                self.scheduler.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config.train_eval_freq == 0:
                self.model.eval()

                with th.no_grad():
                    vals = {}
                    val = self.eval_fn(recon_x, data['target_action'], mean, log_var)
                    
                    vals_elbo.update(val['total'].item(), data['target_action'].size(0))
                    vals_recon.update(val['Recon'].item(), data['target_action'].size(0))
                    vals_kld.update(val['KL_div'].item(), data['target_action'].size(0))
            
                    vals['total'] = vals_elbo.avg
                    vals['Recon'] = vals_recon.avg
                    vals['KL_div'] = vals_kld.avg
                    
                self.model.train()

            if i % self.config.print_freq == 0:
                progress.display(i)
 
        return losses, vals


class Evaluator():
    """
    Generic trainer for a pytorch nn.Module.
    Intended to be flexible, modify as needed.
    """
    def __init__(self,
                 config,
                 data_loader: th.utils.data.DataLoader,
                 model: th.nn.Module,
                 eval_fn: Callable[[Dict[str, th.Tensor], Dict[str, th.Tensor]], Dict[str, th.Tensor]]
                 ):
        self.config = config
        self.data_loader = data_loader
        self.model = model
        self.eval_fn = eval_fn
    
    def eval(self, epoch):        
        batch_time = AverageMeter('Time', ':6.3f')
        vals_elbo = AverageMeter('ELBO', ':4e')
        vals_recon = AverageMeter('Reconstruction Error', ':4e')
        vals_kld = AverageMeter('KL-divergence', ':.4e')

        progress = ProgressMeter(len(self.data_loader),
                                    [batch_time, vals_elbo],
                                    prefix="Epoch: [{}]".format(epoch))
        
        self.model.eval()
        if str(self.config.device) == 'cuda':
            th.cuda.empty_cache()

        with th.no_grad():
            end = time.time()
            for i, data in enumerate(self.data_loader):

                vals = {}
                recon_x, mean, log_var, z = self.model(data['belief'], data['goal'], data['target_action'])
                val = self.eval_fn(recon_x, data['target_action'], mean, log_var)

                vals_elbo.update(val['total'].item(), data['target_action'].size(0))
                vals_recon.update(val['Recon'].item(), data['target_action'].size(0))
                vals_kld.update(val['KL_div'].item(), data['target_action'].size(0))
            
                vals['total'] = vals_elbo.avg
                vals['Recon'] = vals_recon.avg
                vals['KL_div'] = vals_kld.avg
            
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    progress.display(i)

        return vals


def save_checkpoint(msg      : str,
                    filename : str, 
                    epoch    : int,
                    val      : float,
                    model    : nn.Module,
                    optimizer: Optim.Optimizer,
                    scheduler: nn.Module):
    """Save training checkpoint

    Args:
        filename (str): Full path to the save file
        epoch (int): epoch
        model (nn.Module): model 
        optimizer (Optim.Optimizer): optimizer
        scheduler (nn.Module): scheduler
    """

    state = {
        'epoch'     : epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'top_eval'  : val
        }
    th.save(state, filename)
    print(msg)


def load_checkpoint(config,
                    filename : str, 
                    model    : nn.Module,
                    optimizer: Optim.Optimizer,
                    scheduler: nn.Module) \
                        -> Tuple[int, nn.Module, Optim.Optimizer, nn.Module]: 
    """Load training checkpoint

    Args:
        filename (str): Full path to the save file
        model (nn.Module): model
        optimizer (Optim.Optimizer): optimizer
        scheduler (nn.Module): scheduler

    Returns:
        start_epoch (int): start_epoch
        model (nn.Module): model
        optimizer (Optim.Optimizer): optimizer
        scheduler (nn.Module): scheduler
    """

    checkpoint = th.load(filename, map_location=config.device)
    start_epoch = checkpoint['epoch']
    val = checkpoint['top_eval']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return start_epoch, model, optimizer, scheduler


def load_checkpoint_inference(device   : str,
                              filename : str, 
                              model    : nn.Module):
    """Load training checkpoint

    Args:
        filename (str): Full path to the save file
        model (nn.Module): model
    """
    checkpoint = th.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp