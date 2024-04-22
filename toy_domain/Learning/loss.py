import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict


class RegressionLossPolicy(nn.Module):
    def __init__(self, config):
        super(RegressionLossPolicy, self).__init__()
        self.config = config
        self.loss_action = nn.MSELoss()
        # if self.config.use_reward:
        #     self.coefficient_loss = config.coefficient_loss
        #     self.loss_reward = nn.MSELoss()
    
    def forward(self, pred, target):
        loss = {}
        loss_action = self.loss_action(pred['action'], target['action'])
        loss['action'] = loss_action
        
        # if self.config.use_reward:
        #     loss_reward = self.loss_reward(pred['reward'], target['reward'])
        #     loss['reward'] = loss_reward
        #     loss_total = loss_action + self.coefficient_loss * loss_reward
        # else:
        #     loss_total = loss_action

        # loss['total'] = loss_total
        loss['total'] = loss_action

        return loss


class RegressionLossValue(nn.Module):
    def __init__(self, config):
        super(RegressionLossValue, self).__init__()
        self.config = config
        self.mse = nn.MSELoss()
        # if self.config.use_reward:
        #     self.coefficient_loss = config.coefficient_loss
        #     self.loss_reward = nn.MSELoss()
    
    def forward(self, pred, target):
        loss = {}
        loss['total'] = self.mse(pred, target)
        
        # if self.config.use_reward:
        #     loss_reward = self.loss_reward(pred['reward'], target['reward'])
        #     loss['reward'] = loss_reward
        #     loss_total = loss_action + self.coefficient_loss * loss_reward
        # else:
        #     loss_total = loss_action

        # loss['total'] = loss_total

        return loss


class RegressionLossValueWithNegativeData(nn.Module):
    def __init__(self, config):
        super(RegressionLossValueWithNegativeData, self).__init__()
        self.config = config
        if config.value_distribution:
            self.pos = nn.CrossEntropyLoss()
            self.neg = nn.CrossEntropyLoss()
        else:
            self.pos = nn.MSELoss()
            self.neg = nn.MSELoss()
    
    def forward(self, pred, pred_neg, target, target_neg, entropy=0., entropy_neg=0.):
        loss = {}
        loss_pos = self.pos(pred, target)
        loss_neg = self.neg(pred_neg, target_neg)
        
        loss['pos'] = loss_pos
        loss['neg'] = loss_neg

        if self.config.cql_reg:
            loss['total'] = loss_pos + loss_neg + entropy.mean() - self.config.cql_reg * entropy_neg.mean()
        else:
            loss['total'] = loss_pos + loss_neg
            
        return loss


class RegressionLossValueWithNegativeDataEval(nn.Module):
    def __init__(self, config):
        super(RegressionLossValueWithNegativeDataEval, self).__init__()
        self.config = config
        if config.value_distribution:
            self.pos = nn.CrossEntropyLoss()
            self.neg = nn.CrossEntropyLoss()
            self.pos_avg = nn.MSELoss()
            self.neg_avg = nn.MSELoss()
        else:
            self.pos = nn.MSELoss()
            self.neg = nn.MSELoss()
    
    def forward(self, pred, pred_neg, target, target_neg, value=0., value_neg=0.):
        loss = {}
        if self.config.value_distribution:
            loss_pos = self.pos(pred[0], target)
            loss_neg = self.neg(pred_neg[0], target_neg)
            loss_pos_avg = self.pos_avg(pred[1], value)
            loss_neg_avg = self.pos_avg(pred_neg[1], value_neg)
        else:
            loss_pos = self.pos(pred, target)
            loss_neg = self.neg(pred_neg, target_neg)
        
        loss['pos'] = loss_pos
        loss['neg'] = loss_neg
        
        if self.config.value_distribution:
            loss['pos_mse'] = loss_pos_avg
            loss['neg_mse'] = loss_neg_avg
            loss['total'] = loss_pos_avg + loss_neg_avg
        else:
            loss['total'] = loss_pos + loss_neg
        
        return loss


class ELBOLoss(nn.Module):
    def __init__(self, config):
        super(ELBOLoss, self).__init__()
        self.config = config

    def forward(self, recon_x, x, mean, log_var):
        loss = {}
        recon_loss = F.mse_loss(recon_x, x)
        kld = -0.5 * th.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0) # Using reparameterization
        loss['Recon'] = recon_loss
        loss['KL_div'] = self.config.vae_beta * kld
        loss['total'] = recon_loss + kld

        return loss


class ELBOLossImportanceSampling(nn.Module):
    def __init__(self, config):
        super(ELBOLossImportanceSampling, self).__init__()
        self.config = config

    def forward(self, recon_x, x, mean, log_var, weight):
        loss = {}
        recon_loss = F.mse_loss(recon_x, x, reduction='none')
        recon_loss = recon_loss * weight
        recon_loss = th.mean(recon_loss)
        kld = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        kld = kld * weight
        kld = self.config.vae_beta * th.mean(kld)
        elbo_loss = recon_loss + kld
        loss['Recon'] = recon_loss
        loss['KL_div'] = kld
        loss['total'] = elbo_loss

        return loss


class CQLLoss(nn.Module):
    def __init__(self, config):
        super(CQLLoss, self).__init__()
        self.config = config
        self.reg_scale = config.cql_reg
        self.alpha = config.cql_alpha
        self.mse = nn.MSELoss()
    
    def forward(self, pred, pred_rand, target, entropy):
        loss = {}
        loss_accumulated_reward = self.mse(pred, target)
        loss['accumulated_reward'] = loss_accumulated_reward
        
        # if self.reg:
        #     min_v = th.logsumexp(pred_rand, dim=1).mean()
        # else:
        #     min_v = pred_rand.mean()
            
        loss_conservative = pred_rand.mean() - pred.mean()
        loss['conservative'] = loss_conservative
    
        loss['total'] = loss_accumulated_reward + self.alpha * loss_conservative - self.reg_scale * entropy.mean()

        return loss
    
    
class CustomLossValueDistribution(nn.Module):
    def __init__(self, config):
        super(CustomLossValueDistribution, self).__init__()
        self.config = config
        self.ce1 = nn.CrossEntropyLoss()
        self.ce2 = nn.CrossEntropyLoss()
        self.ce3 = nn.CrossEntropyLoss()
        self.ce4 = nn.CrossEntropyLoss()
        self.ce5 = nn.CrossEntropyLoss()
        self.ce6 = nn.CrossEntropyLoss()
    
    def forward(self, pred, target, pred_neg=None, target_neg=None, pred_ood=None, entropy=None, entropy_neg=None):
        loss = {'total': 0.}
        if self.config.loss_1:
            loss_1 = self.ce1(pred, target)
            loss['loss_1'] = loss_1
            loss['total'] += loss_1
        if self.config.loss_2:
            loss_2 = self.ce2(pred_neg, target_neg)
            loss['loss_2'] = loss_2
            loss['total'] += loss_2
        if self.config.loss_3:
            pass
        if self.config.loss_4:
            bin_v = th.where(target>=self.config.bin_boundary, self.config.num_bin-1, 0)
            loss_4 = self.ce4(pred, bin_v)
            loss['loss_4'] = loss_4
            loss['total'] += loss_4
        if self.config.loss_5:
            bin_v_max = th.full(size=(pred.size(0),), fill_value=self.config.num_bin-1, dtype=th.long, device=th.device(self.config.device))
            loss_5 = self.ce5(pred, bin_v_max)
            loss['loss_5'] = loss_5
            loss['total'] += loss_5
        if self.config.loss_6:
            bin_v_min = th.full(size=(pred_neg.size(0),), fill_value=0., dtype=th.long, device=th.device(self.config.device))
            loss_6 = self.ce6(pred_neg, bin_v_min)
            loss['loss_6'] = loss_6
            loss['total'] += loss_6
            
        return loss
    

class RankLoss(nn.Module):
    def __init__(self, config):
        super(RankLoss, self).__init__()
        self.config = config
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred_success, pred_fail):
        target_success = th.full(size=(pred_success.size(0),), fill_value=1., dtype=th.float32, device=th.device(self.config.device)).unsqueeze(1)
        target_fail = th.full(size=(pred_success.size(0),), fill_value=0., dtype=th.float32, device=th.device(self.config.device)).unsqueeze(1)
        pred = th.cat([pred_success, pred_fail], dim=1)
        target = th.cat([target_success, target_fail], dim=1)
        rank_loss = self.bce(pred, target)
        loss = {'total': rank_loss}
            
        return loss


class PreferenceLoss(nn.Module):
    def __init__(self, config):
        super(PreferenceLoss, self).__init__()
        self.config = config
        if self.config.preference_softmax:
            self.bce = nn.BCELoss()
        else:
            self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, success_score, comparison_score, target):
        if self.config.preference_softmax:
            pred = th.cat([success_score, comparison_score], dim=1).softmax(dim=1)
        else:
            pred = th.cat([success_score, comparison_score], dim=1)
        preference_loss = self.bce(pred, target)
        loss = {'total': preference_loss}
            
        return loss