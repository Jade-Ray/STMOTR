import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy
from utils.track_ops import generalized_track_iou, track_distance


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_is_referred: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_is_referred: This is the relative weight of the reference cost in the total matching cost
            cost_bbox (float, optional): This is the relative weight of the L1 error of the bounding box coordinates in the matching cost. Defaults to 1.
            cost_giou (float, optional): This is the relative weight of the giou loss of the bounding box in the matching cost. Defaults to 1.
        """
        super().__init__()
        self.cost_is_referred = cost_is_referred
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_is_referred != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    @torch.inference_mode()
    def forward(self, outputs, targets):
        """ Performs the matching

        Args:
            outputs (dict): This is a dict that contains at least these entries:
            
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                "pred_vis": Tensor of dim [batch_size, num_queries, num_frames] with the visibility of each box of tracker, 0~1
                "pred_boxes": Tensor of dim [batch_size, num_queries, num_frames, 4] with the predicted box coordinates
            
            targets (list): This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
            
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of groud-truth objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, num_frames, 4] containing the target box coordinates
                "vis": Tensor of dim [num_target_boxes, num_frames] containing the target box visibility in each frame with 0~1
                
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        t, bs, num_queries = outputs["pred_boxes"].shape[:3]
        
        # We flatten to compute the cost matrices in a batch
        out_bbox = outputs["pred_boxes"].flatten(1, 2).transpose(0, 1) # [batch_size * num_queries, t, 4]
        
        # Also concat the target labels and boxes
        tgt_referred = torch.cat([v['referred'] for v in targets]).long() # [all_batch_num_target_boxes, t]
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # [all_batch_num_target_boxes, t, 4]
        
        # Compute the soft-tokens cost:
        cost_is_referred = compute_is_referred_cost(outputs, targets) # [batch_size * num_queries, all_batch_num_target_boxes]

        # Compute the distance cost between boxes of tracks
        cost_bbox = track_distance(out_bbox, tgt_bbox, tgt_referred, mode='mean') # [batch_size * num_queries, all_batch_num_target_boxes]
        
        # Compute the giou cost between tracks
        cost_giou = -generalized_track_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox), tgt_referred, mode='mean') # [batch_size * num_queries, all_batch_num_target_boxes]
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_is_referred * cost_is_referred + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        assert ~torch.isnan(C).any()
        
        sizes = [len(v["boxes"]) for v in targets] # each batch num target
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] # split cost with batch and compute Humgarian match
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            

def compute_is_referred_cost(outputs, targets):
    pred_is_referred = outputs['pred_is_referred'].flatten(1, 2).softmax(dim=-1)  # [t, b*nq, 2]
    device = pred_is_referred.device
    # note that ref_indices are shared across time steps
    target_is_referred_o = torch.cat([v['referred'] for v in targets]).long().transpose(0, 1) # [t, num_tra]
    t, num_traj = target_is_referred_o.shape
    
    tgt_referred = torch.zeros((t, num_traj, 2), device=device)
    # 'no object' class by default
    tgt_referred[:, :, :] = torch.tensor([1.0, 0.0], device=device)
    tgt_referred[target_is_referred_o == 1] = torch.tensor([0.0, 1.0], device=device)
    
    cost_is_referred = -(pred_is_referred.unsqueeze(2) * tgt_referred.unsqueeze(1)).sum(dim=-1).mean(dim=0)
    
    # tgt_referred = torch.cat([v['referred'] for v in targets]).long().transpose(0, 1) # [t, num_tra]
    # tgt_referred = tgt_referred[:, None].repeat(1, pred_is_referred.shape[1], 1) # [t, b*nq, num_tra]
    # cost_is_referred = torch.gather(pred_is_referred, 2, tgt_referred) # [t, b*nq, num_tra]
    # cost_is_referred = -cost_is_referred.mean(dim=0) # [b*nq, num_tra]
    
    return cost_is_referred


def build_matcher(args):
    return HungarianMatcher(cost_is_referred=args.set_cost_is_referred, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)