import time
import torch as th
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)

from trainer import AverageMeter, ProgressMeter

class Evaluator():
    """
    Generic trainer for a pytorch nn.Module.
    Intended to be flexible, modify as needed.
    """
    def __init__(self,
                 config,
                 loader: th.utils.data.DataLoader,
                 model: th.nn.Module,
                 eval_fn: Callable[[Dict[str, th.Tensor], Dict[str, th.Tensor]], Dict[str, th.Tensor]],
                 neg_loader: th.utils.data.DataLoader = None,
                 ):
        self.config = config
        self.loader = loader
        if neg_loader is not None:
            self.neg_loader = neg_loader
        else:
            self.neg_loader = None
        self.model = model
        self.eval_fn = eval_fn
    
    def eval(self, epoch):        
        batch_time = AverageMeter('Time', ':6.3f')
        if self.config.model == 'CVAE':
            vals_elbo = AverageMeter('ELBO', ':4e')
            vals_recon = AverageMeter('Reconstruction Error', ':4e')
            vals_kld = AverageMeter('KL-divergence', ':.4e')

            progress = ProgressMeter(len(self.loader),
                                     [batch_time, vals_elbo],
                                     prefix="Epoch: [{}]".format(epoch))

        elif self.config.model == 'ValueNet':
            if self.config.preference_loss or self.config.rank:
                vals_total = AverageMeter('BCELoss', ':.4e')
                
                progress = ProgressMeter(len(self.loader),
                                        [batch_time, vals_total, vals_total],
                                        prefix="Epoch: [{}]".format(epoch))
                
            elif self.config.alpha_go:
                vals_total = AverageMeter('MSELoss', ':.4e')
                
                progress = ProgressMeter(len(self.loader),
                                        [batch_time, vals_total, vals_total],
                                        prefix="Epoch: [{}]".format(epoch))
                
            elif self.config.value_distribution and self.config.custom_loss:
                vals_total = AverageMeter('NLLoss', ':.4e')
                if self.config.loss_1:
                    vals_loss_1 = AverageMeter('NLLoss', ':.4e')
                if self.config.loss_2:                
                    vals_loss_2 = AverageMeter('NLLoss', ':.4e')
                if self.config.loss_3:                
                    vals_loss_3 = AverageMeter('NLLoss', ':.4e')
                if self.config.loss_4:                
                    vals_loss_4 = AverageMeter('NLLoss', ':.4e')
                if self.config.loss_5:                
                    vals_loss_5 = AverageMeter('NLLoss', ':.4e')
                if self.config.loss_6:                
                    vals_loss_6 = AverageMeter('NLLoss', ':.4e')
                vals_total = AverageMeter('NLLoss', ':.4e')

                progress = ProgressMeter(len(self.loader),
                                        [batch_time, vals_total, vals_total],
                                        prefix="Epoch: [{}]".format(epoch))
                
            elif self.config.cql_loss == 'mse2':
                # if self.config.value_distribution:
                #     vals_total = AverageMeter('NLLoss', ':.4e')
                #     vals_pos = AverageMeter('NLLoss', ':.4e')
                #     vals_neg = AverageMeter('NLLoss', ':.4e')
                #     vals_pos_avg = AverageMeter('MSELoss', ':.4e')
                #     vals_neg_avg = AverageMeter('MSELoss', ':.4e')
                vals_total = AverageMeter('MSELoss', ':.4e')
                vals_pos = AverageMeter('MSELoss', ':.4e')
                vals_neg = AverageMeter('MSELoss', ':.4e')
                
                progress = ProgressMeter(len(self.loader),
                                        [batch_time, vals_total, vals_total],
                                        prefix="Epoch: [{}]".format(epoch))
            else:
                vals_total = AverageMeter('MSELoss', ':.4e')

                progress = ProgressMeter(len(self.loader),
                                        [batch_time, vals_total],
                                        prefix="Epoch: [{}]".format(epoch))

        elif self.config.model == 'PolicyValueNet':
            # |TODO| implement Chamfer distance
            vals_total = AverageMeter('Total Eval', ':.4e')
            vals_elbo = AverageMeter('Action ELBOLoss', ':.4e')
            vals_recon = AverageMeter('Action Reconstruction Error', ':.4e')
            vals_kld = AverageMeter('Action KL-divergence', ':.4e')
            vals_mse = AverageMeter('Accumulated Reward MSELoss', ':.4e')

            progress = ProgressMeter(len(self.loader),
                                     [batch_time, vals_total],
                                     prefix="Epoch: [{}]".format(epoch))

        elif self.config.model == 'ValueNetDiscreteRepresentation':
            if self.config.cql_loss == 'cql':
                vals_total = AverageMeter('CQLLoss', ':.4e')
                vals_accumulated_reward = AverageMeter('MSELoss', ':.4e')
                vals_conservative = AverageMeter('CQLLoss', ':.4e')
                
                progress = ProgressMeter(len(self.loader),
                                        [batch_time, vals_total, vals_total],
                                        prefix="Epoch: [{}]".format(epoch))           
            elif self.config.cql_loss == 'mse2':
                vals_total = AverageMeter('MSELoss', ':.4e')
                vals_pos = AverageMeter('MSELoss', ':.4e')
                vals_neg = AverageMeter('MSELoss', ':.4e')
                
                progress = ProgressMeter(len(self.loader),
                                        [batch_time, vals_total, vals_total],
                                        prefix="Epoch: [{}]".format(epoch))                     
            else:
                vals_total = AverageMeter('MSELoss', ':.4e')
                
                progress = ProgressMeter(len(self.loader),
                                        [batch_time, vals_total, vals_total],
                                        prefix="Epoch: [{}]".format(epoch))
        else:
            vals_action = AverageMeter('Action SmoothL1Loss', ':.4e')

            progress = ProgressMeter(len(self.loader),
                                     [batch_time, vals_action],
                                     prefix="Epoch: [{}]".format(epoch))
        
        self.model.eval()
        if str(self.config.device) == 'cuda':
            th.cuda.empty_cache()

        with th.no_grad():
            end = time.time()
            if self.neg_loader is not None:
                neg_iter = iter(self.neg_loader)
            for i, data in enumerate(self.loader):
                # target = {}
                # target_action = th.squeeze(data['next_action'])
                # target_value = th.squeeze(data['accumulated_reward'])
                # target['action'] = target_action
                # target['accumulated_reward'] = target_value
                # target['value'] = data['value']     # |TODO(jiyong)|: change the variable name
                # if self.config.use_reward:
                #     target_reward = th.squeeze(data['next_reward'])
                #     target['reward'] = target_reward

                vals = {}
                if self.config.model == 'CVAE':
                    recon_x, mean, log_var, z = self.model(data)
                    target = {}
                    target_action = th.squeeze(data['next_action'])
                    target['action'] = target_action
                    if self.config.data_type == 'q_policy':
                        val = self.eval_fn(recon_x, target['action'], mean, log_var, data['importance_weight'].unsqueeze(1))
                    else:
                        val = self.eval_fn(recon_x, target['action'], mean, log_var)

                    vals_elbo.update(val['total'].item(), data['observation'].size(0))
                    vals_recon.update(val['Recon'].item(), data['observation'].size(0))
                    vals_kld.update(val['KL_div'].item(), data['observation'].size(0))
                
                    vals['total'] = vals_elbo.avg
                    vals['Recon'] = vals_recon.avg
                    vals['KL_div'] = vals_kld.avg

                elif self.config.model == 'ValueNet':
                    if self.config.preference_loss:
                        success_score = self.model(data['success_node'])
                        comparison_score = self.model(data['comparison_node'])
                        target = data['preference']
                        
                        loss = self.eval_fn(success_score=success_score, comparison_score=comparison_score, target=target)
                        vals_total.update(loss['total'].item(), data['success_node']['observation'].size(0))
                        vals['total'] = vals_total.avg
                    
                    elif self.config.alpha_go:
                        pred = self.model(data)
                        val = self.eval_fn(pred=pred, target=th.squeeze(data['success_or_fail']))
                        
                        vals_total.update(val['total'].item(), data['observation'].size(0))
                        vals['total'] = vals_total.avg
                    
                    elif self.config.value_distribution and self.config.custom_loss:
                        pred = self.model(data)
                        
                        if not (self.config.loss_6 or self.config.loss_2):    
                            val = self.eval_fn(pred=pred, target=target['accumulated_reward'])
                        else:
                            try:
                                neg_data = next(neg_iter)
                            except StopIteration:
                                print("Restarting negative training epoch")
                                neg_iter = iter(self.neg_loader)
                                neg_data = next(neg_iter)
                            target_neg = {}
                            target_neg_action = th.squeeze(neg_data['next_action'])
                            target_neg_value = th.squeeze(neg_data['accumulated_reward'])
                            target_neg['action'] = target_neg_action
                            target_neg['accumulated_reward'] = target_neg_value

                            pred_neg = self.model(neg_data)

                            val = self.eval_fn(pred=pred, target=target['accumulated_reward'], pred_neg=pred_neg, target_neg=target_neg['accumulated_reward'])
                            
                        vals_total.update(val['total'].item(), data['observation'].size(0))
                        vals['total'] = vals_total.avg
                        if self.config.loss_1:
                            vals_loss_1.update(val['loss_1'].item(), data['observation'].size(0))
                            vals['loss_1'] = vals_loss_1.avg
                        if self.config.loss_2:                
                            vals_loss_2.update(val['loss_2'].item(), neg_data['observation'].size(0))
                            vals['loss_2'] = vals_loss_2.avg
                        if self.config.loss_3:                
                            vals_loss_3.update(val['loss_3'].item(), data['observation'].size(0))
                            vals['loss_3'] = vals_loss_3.avg
                        if self.config.loss_4:                
                            vals_loss_4.update(val['loss_4'].item(), data['observation'].size(0))
                            vals['loss_4'] = vals_loss_4.avg
                        if self.config.loss_5:                
                            vals_loss_5.update(val['loss_5'].item(), neg_data['observation'].size(0))
                            vals['loss_5'] = vals_loss_5.avg
                        if self.config.loss_6:                
                            vals_loss_5.update(val['loss_6'].item(), neg_data['observation'].size(0))
                            vals['loss_6'] = vals_loss_6.avg
                    
                    # elif self.config.cql_loss == 'mse2':
                    elif self.config.rank:
                        # if self.config.value_distribution:
                        #     pred = self.model.inference(data)
                        # else:
                        pred = self.model(data)
                        
                        try:
                            neg_data = next(neg_iter)
                        except StopIteration:
                            print("Restarting negative training epoch")
                            neg_iter = iter(self.neg_loader)
                            neg_data = next(neg_iter)
                        target_neg = {}
                        target_neg_action = th.squeeze(neg_data['next_action'])
                        target_neg_value = th.squeeze(neg_data['accumulated_reward'])
                        target_neg['action'] = target_neg_action
                        target_neg['accumulated_reward'] = target_neg_value
                        target_neg['value'] = neg_data['value']     # |TODO(jiyong)|: change the variable name
                        
                        # if self.config.value_distribution:
                        #     pred_neg = self.model.inference(neg_data)
                        # else:
                        pred_neg = self.model(neg_data)
                        
                        # if self.config.value_distribution:
                        #     val = self.eval_fn(pred=pred, target=target['accumulated_reward'], pred_neg=pred_neg, target_neg=target_neg['accumulated_reward'], value=target['value'], value_neg=target_neg['value'])
                        # else:
                        # val = self.eval_fn(pred=pred, target=target['accumulated_reward'], pred_neg=pred_neg, target_neg=target_neg['accumulated_reward'])
                        val = self.eval_fn(pred_success=pred, pred_fail=pred_neg)
                        
                        
                        vals_total.update(val['total'].item(), data['observation'].size(0))
                        vals['total'] = vals_total.avg
                        
                        # vals_pos.update(val['pos'].item(), data['observation'].size(0))
                        # vals_neg.update(val['neg'].item(), data['observation'].size(0))
                        # vals['pos'] = vals_pos.avg
                        # vals['neg'] = vals_neg.avg
                        
                        # if self.config.value_distribution:
                        #     vals_pos.update(val['pos_mse'].item(), data['observation'].size(0))
                        #     vals_neg.update(val['neg_mse'].item(), data['observation'].size(0))
                        #     vals['pos_mse'] = vals_pos.avg
                        #     vals['neg_mse'] = vals_neg.avg
                        
                    else:
                        pred = self.model(data)
                        val = self.eval_fn(pred=pred, target=target['accumulated_reward'])
                        
                        vals_total.update(val['total'].item(), data['observation'].size(0))                
                        vals['total'] = vals_total.avg

                elif self.config.model == 'PolicyValueNet':
                    value, recon_x, mean, log_var, z = self.model(data)
                    val = self.eval_fn(value, recon_x, target, mean, log_var)

                    vals_total.update(val['total'].item(), data['observation'].size(0))
                    vals_elbo.update(val['ELBO'].item(), data['observation'].size(0))
                    vals_recon.update(val['Recon'].item(), data['observation'].size(0))
                    vals_kld.update(val['KL_div'].item(), data['observation'].size(0))
                    vals_mse.update(val['MSE'].item(), data['observation'].size(0))
                
                    vals['total'] = vals_total.avg
                    vals['ELBO'] = vals_elbo.avg
                    vals['Recon'] = vals_recon.avg
                    vals['KL_div'] = vals_kld.avg
                    vals['MSE'] = vals_mse.avg

                elif self.config.model == 'ValueNetDiscreteRepresentation':
                    if self.config.cql_loss == 'cql':
                        pred, pred_rand, entropy = self.model(data)
                        val = self.eval_fn(pred=pred, target=target['accumulated_reward'], pred_rand=pred_rand, entropy=entropy)
                    
                        vals_total.update(val['total'].item(), data['observation'].size(0))
                        vals_accumulated_reward.update(val['accumulated_reward'].item(), data['observation'].size(0))
                        vals_conservative.update(val['conservative'].item(), data['observation'].size(0))
                        vals['total'] = vals_total.avg
                        vals['accumulated_reward'] = vals_accumulated_reward.avg
                        vals['conservative'] = vals_conservative.avg
                    
                    elif self.config.cql_loss == 'mse2':
                        pred, entropy = self.model(data)
                        try:
                            neg_data = next(neg_iter)
                        except StopIteration:
                            print("Restarting negative training epoch")
                            neg_iter = iter(self.neg_loader)
                            neg_data = next(neg_iter)
                        target_neg = {}
                        target_neg_action = th.squeeze(data['next_action'])
                        target_neg_value = th.squeeze(data['accumulated_reward'])
                        target_neg['action'] = target_neg_action
                        target_neg['accumulated_reward'] = target_neg_value
                        pred_neg, entropy_neg = self.model(neg_data)
                        
                        val = self.eval_fn(pred=pred, target=target['accumulated_reward'], pred_neg=pred_neg, target_neg=target_neg['accumulated_reward'], entropy=entropy, entropy_neg=entropy_neg)
                        
                        vals_total.update(val['total'].item(), data['observation'].size(0))
                        vals_pos.update(val['pos'].item(), data['observation'].size(0))
                        vals_neg.update(val['neg'].item(), data['observation'].size(0))
                        vals['total'] = vals_total.avg
                        vals['pos'] = vals_pos.avg
                        vals['neg'] = vals_neg.avg
                    
                    else:
                        pred, entropy = self.model(data)
                        val = self.eval_fn(pred=pred, target=target['accumulated_reward'])
                        
                        vals_total.update(val['total'].item(), data['observation'].size(0))                
                        vals['total'] = vals_total.avg

                else:
                    pred = self.model(data)
                    val = self.eval_fn(pred=pred, target=target)

                    vals_action.update(val['action'].item(), data['observation'].size(0))
                    # if self.config.use_reward:
                    #     vals_reward.update(val['reward'].item(), data['observation'].size(0))
                    
                    vals['action'] = vals_action.avg
                    # if self.config.use_reward:
                    #     vals['reward'] = vals_reward.avg
            
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    progress.display(i)

        return vals
