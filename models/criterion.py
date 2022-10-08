import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

import utils.distributed as du
from utils.box_ops import box_cxcywh_to_xyxy
from utils.track_ops import generalized_track_iou, track_distance


class SetCriterion(nn.Module):
    """ This class computes the loss for MTTR.
    The process happens in two steps:
        1) we compute the hungarian assignment between the ground-truth and predicted sequences.
        2) we supervise each pair of matched ground-truth / prediction sequences (reference prediction)
    """
    def __init__(self, matcher, weight_dict, eos_coef=None, alpha=None, gamma=None, 
                 referred_loss_type='softmax'):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the un-referred category.
            alpha: Weighting factor of focal loss.
            gamma: Exponent of the modulating factor of focal loss.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.alpha = alpha
        self.gamma = gamma
        # make sure that only loss functions with non-zero weights are computed:
        losses_to_compute = []
        if weight_dict['loss_boxes'] > 0 or weight_dict['loss_giou'] > 0:
            losses_to_compute.append('boxes')
        if weight_dict['loss_is_referred'] > 0:
            losses_to_compute.append('cardinality')
            if referred_loss_type == 'softmax':
                losses_to_compute.append('is_referred_softmax')
            elif referred_loss_type == 'softmax_focal_loss':
                losses_to_compute.append('is_referred_softmax_focal_loss')
            elif referred_loss_type == 'sigmoid_focal_loss':
                losses_to_compute.append('is_referred_sigmoid_focal_loss')
            else:
                raise ValueError(f"Unknown referred_loss_type {referred_loss_type}")
        self.losses = losses_to_compute
        
    def forward(self, outputs, targets):
        aux_outputs_list = outputs.pop('aux_outputs', None)
        # compute the losses for the output of the last decoder layer:
        losses = self.compute_criterion(outputs, targets, losses_to_compute=self.losses)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate decoder layer.
        if aux_outputs_list is not None:
            aux_losses_to_compute = self.losses.copy()
            for i, aux_outputs in enumerate(aux_outputs_list):
                losses_dict = self.compute_criterion(aux_outputs, targets, aux_losses_to_compute)
                losses_dict = {k + f'_{i}': v for k, v in losses_dict.items()}
                losses.update(losses_dict)

        return losses

    def compute_criterion(self, outputs, targets, losses_to_compute):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_tracks = sum(len(t["boxes"]) for t in targets)
        num_tracks = torch.as_tensor([num_tracks], dtype=torch.float, device=indices[0][0].device)
        if du.get_rank() > 0:
            torch.distributed.all_reduce(num_tracks)
        num_tracks = torch.clamp(num_tracks / du.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in losses_to_compute:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_tracks=num_tracks))
        return losses
    
    def loss_is_referred_softmax(self, outputs, targets, indices, num_tracks, **kwargs):
        pred_is_referred = outputs['pred_is_referred'] # [T, B, N, 2]
        T, B, N = pred_is_referred.shape[:3]
        target_is_referred = torch.zeros((B, N, T), dtype=torch.int64,
                                         device=pred_is_referred.device)
        idx = self._get_src_permutation_idx(indices)
        target_is_referred_o = torch.cat([t["referred"][J] for t, (_, J) in zip(targets, indices)]) # [batch_obj_num, T]
        target_is_referred[idx] = target_is_referred_o.long() # [B, N, T]
        empty_weight = torch.ones(2, device=pred_is_referred.device)
        empty_weight[0] = self.eos_coef
        loss = F.cross_entropy(
            pred_is_referred.permute(1, 3, 2, 0), # [B, 2, N, T]
            target_is_referred, 
            weight=empty_weight,
            reduction='none') # [B, N, T]
        loss = loss.mean(-1).sum() / (B)
        
        losses = {'loss_is_referred': loss}
        return losses

    def loss_is_referred_softmax_focal_loss(self, outputs, targets, indices, num_tracks, **kwargs):
        pred_is_referred = outputs['pred_is_referred'] # [T, B, N, 2]
        T, B, N = pred_is_referred.shape[:3]
        target_is_referred = torch.zeros((B, N, T), dtype=torch.int64,
                                         device=pred_is_referred.device)
        
        idx = self._get_src_permutation_idx(indices)
        target_is_referred_o = torch.cat([t["referred"][J] for t, (_, J) in zip(targets, indices)]) # [batch_obj_num, T]
        target_is_referred[idx] = target_is_referred_o.long() # [B, N, T]
        
        target_onehot = torch.zeros((B, N, T, 2), dtype=torch.int64,
                                    device=pred_is_referred.device)
        target_onehot.scatter_(3, target_is_referred.unsqueeze(-1), 1)
        
        prob = pred_is_referred.softmax(-1).permute(1, 2, 0, 3) # [B, N, T, 2]
        ce_loss = -prob.log() * target_onehot + (-(1 - prob).log()) * (1-target_onehot)
        
        p_t = prob*target_onehot + (1 - prob) * (1-target_onehot)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        alpha_t = self.alpha * target_onehot + (1 - self.alpha) * (1-target_onehot)
        loss = alpha_t * loss
        
        loss = loss.mean(2).sum() / (B)
        
        losses = {'loss_is_referred': loss}
        return losses
    
    def loss_is_referred_sigmoid_focal_loss(self, outputs, targets, indices, num_tracks, **kwargs):
        pred_is_referred = outputs['pred_is_referred'] # [T, B, N, 2]
        device = pred_is_referred.device
        T, B, N = pred_is_referred.shape[:3]
        target_is_referred = torch.zeros((B, N, T), dtype=torch.int64,
                                         device=pred_is_referred.device)
        
        idx = self._get_src_permutation_idx(indices)
        target_is_referred_o = torch.cat([t["referred"][J] for t, (_, J) in zip(targets, indices)]) # [batch_obj_num, T]
        target_is_referred[idx] = target_is_referred_o.long() # [B, N, T]
        
        target_onehot = torch.zeros((B, N, T, 2), dtype=pred_is_referred.dtype,
                                    layout=pred_is_referred.layout, 
                                    device=pred_is_referred.device)
        target_onehot.scatter_(3, target_is_referred.unsqueeze(-1), 1)
        
        loss = sigmoid_focal_loss(
            pred_is_referred.permute(1, 2, 0, 3), # [B, N, T, 2]
            target_onehot,
            self.alpha, self.gamma)
        
        loss = loss.mean(2).sum() / (B)
        
        losses = {'loss_is_referred': loss}
        return losses
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_tracks):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty tracks
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_is_referred = rearrange(outputs['pred_is_referred'], 't b n c -> b n t c').softmax(dim=-1)
        pred_pos = (pred_is_referred.argmax(-1) == 1).sum((1,2)) # b
        pred_neg = (pred_is_referred.argmax(-1) == 0).sum((1,2))
        tgt_lengths = torch.as_tensor([v['referred'].view(-1).shape[0] for v in targets],
                                      device=pred_is_referred.device)
        card_err = F.l1_loss(pred_pos.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_tracks):
        """Compute the losses related to the bounding boxes, the distance regression loss and the GIoU loss targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, num_frame, 4], and the key "vis" containing a tensor of dim [nb_target_boxes, num_frame]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = rearrange(outputs["pred_boxes"], 't b n c -> b n t c')[idx] # [pred_obj_num, T, 4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)]) #[batch_obj_num, T, 4]
        target_is_referred_o = torch.cat([t["referred"][J] for t, (_, J) in zip(targets, indices)]) # [batch_obj_num, T]
        
        loss_boxes = torch.diag(track_distance(src_boxes, target_boxes, target_is_referred_o))
        
        losses = {}
        losses['loss_boxes'] = loss_boxes.sum() / num_tracks
        
        loss_giou = 1 - torch.diag(generalized_track_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes),
            target_is_referred_o))
        losses['loss_giou'] = loss_giou.sum() / num_tracks
        return losses
    
    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'is_referred_softmax': self.loss_is_referred_softmax,
            'is_referred_softmax_focal_loss': self.loss_is_referred_softmax_focal_loss,
            'is_referred_sigmoid_focal_loss': self.loss_is_referred_sigmoid_focal_loss,
            'cardinality': self.loss_cardinality,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the precision@k for specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2.):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss
