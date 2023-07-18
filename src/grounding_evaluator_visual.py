# ------------------------------------------------------------------------
# Modification: EDA
# Created: 05/21/2022
# Author: Yanmin Wu
# E-mail: wuyanminmax@gmail.com
# https://github.com/yanmin-wu/EDA 
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""A class to collect and evaluate language grounding results."""

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
import utils.misc as misc
import numpy as np
import wandb

def softmax(x):
    """Numpy function for softmax."""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs

# BRIEF Evaluator
class GroundingEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): IoU thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    def __init__(self, only_root=True, thresholds=[0.25, 0.5],
                 topks=[1, 5, 10], prefixes=[], filter_non_gt_boxes=False, logger=None):
        """Initialize accumulators."""
        self.only_root = only_root
        self.thresholds = thresholds
        self.topks = topks
        self.prefixes = prefixes
        self.filter_non_gt_boxes = filter_non_gt_boxes
        self.reset()
        self.logger = logger
        self.bad_case_threshold = 0.15
        self.bad_case_visualization = False
        self.kps_points_visualization = False

    def reset(self):
        """Reset accumulators to empty."""
        self.dets = {
            (prefix, t, k, mode): 0
            for prefix in self.prefixes
            for t in self.thresholds
            for k in self.topks
            for mode in ['bbs', 'bbf']
        }
        self.gts = dict(self.dets)

        self.dets.update({'vd': 0, 'vid': 0})
        self.dets.update({'hard': 0, 'easy': 0})
        self.dets.update({'multi': 0, 'unique': 0})
        self.gts.update({'vd': 1e-14, 'vid': 1e-14})
        self.gts.update({'hard': 1e-14, 'easy': 1e-14})
        self.gts.update({'multi': 1e-14, 'unique': 1e-14})
        self.dets.update({'vd50': 0, 'vid50': 0})
        self.dets.update({'hard50': 0, 'easy50': 0})
        self.dets.update({'multi50': 0, 'unique50': 0})
        self.gts.update({'vd50': 1e-14, 'vid50': 1e-14})
        self.gts.update({'hard50': 1e-14, 'easy50': 1e-14})
        self.gts.update({'multi50': 1e-14, 'unique50': 1e-14})

    def print_stats(self):
        """Print accumulated accuracies."""
        mode_str = {
            'bbs': 'position alignment',
            'bbf': 'semantic alignment'
        }
        for prefix in self.prefixes:
            for mode in ['bbs', 'bbf']:
                for t in self.thresholds:
                    self.logger.info(
                    # print(
                        prefix + ' ' + mode_str[mode] + ' ' +  'Acc%.2f:' % t + ' ' + 
                        ', '.join([
                            'Top-%d: %.5f' % (
                                k,
                                self.dets[(prefix, t, k, mode)]
                                / max(self.gts[(prefix, t, k, mode)], 1)
                            )
                            for k in self.topks
                        ])
                    )
        self.logger.info('\nAnalysis')
        self.logger.info('iou@0.25')
        for field in ['easy', 'hard', 'vd', 'vid', 'unique', 'multi']:
            self.logger.info(field + ' ' +  str(self.dets[field] / self.gts[field]))
        self.logger.info('iou@0.50')
        for field in ['easy50', 'hard50', 'vd50', 'vid50', 'unique50', 'multi50']:
            self.logger.info(field + ' ' +  str(self.dets[field] / self.gts[field]))

    def synchronize_between_processes(self):
        all_dets = misc.all_gather(self.dets)
        all_gts = misc.all_gather(self.gts)

        if misc.is_main_process():
            merged_predictions = {}
            for key in all_dets[0].keys():
                merged_predictions[key] = 0
                for p in all_dets:
                    merged_predictions[key] += p[key]
            self.dets = merged_predictions

            merged_predictions = {}
            for key in all_gts[0].keys():
                merged_predictions[key] = 0
                for p in all_gts:
                    merged_predictions[key] += p[key]
            self.gts = merged_predictions

    # BRIEF Evaluation
    def evaluate(self, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # NOTE Two Evaluation Ways: position alignment, semantic alignment
        self.evaluate_bbox_by_pos_align(end_points, prefix)
        self.evaluate_bbox_by_sem_align(end_points, prefix)
    
    # BRIEF position alignment
    def evaluate_bbox_by_pos_align(self, end_points, prefix):
        """
        Evaluate bounding box IoU by position alignment

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_bboxes = self._parse_gt(end_points)    
        
        # Parse predictions
        sem_scores = end_points[f'{prefix}sem_cls_scores'].softmax(-1)

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(
                sem_scores.shape[0], sem_scores.shape[1],
                positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q=256, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1) # ([B, 256, 6])

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            if self.filter_non_gt_boxes:  # this works only for the target box
                ious, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(
                        end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ]
                    ),  # (gt, 6)
                    box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
                )  # (gt, Q)
                is_correct = (ious.max(0)[0] > 0.25) * 1.0
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)    
                * pmap.unsqueeze(1)             
            ).sum(-1)

            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]    # num_obj
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other

            if is_correct is not None:
                scores = scores * is_correct[None]

            top = scores.argsort(1, True)[:, :10]
            pbox = pred_bbox[bid, top.reshape(-1)]

            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]   # ([1, 10])

            # step Measure IoU>threshold, ious are (obj, 10)
            topks = self.topks
            for t in self.thresholds:
                thresholded = ious > t
                for k in topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbs')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbs')] += len(thresholded)

    # BRIEF semantic alignment
    def evaluate_bbox_by_sem_align(self, end_points, prefix):
        """
        Evaluate bounding box IoU by semantic alignment.

        Args:
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # step get the position label and GT box 
        positive_map, modify_positive_map, pron_positive_map, other_entity_map, \
            auxi_entity_positive_map, rel_positive_map, gt_bboxes = self._parse_gt(end_points)    
        
        # Parse predictions
        pred_center = end_points[f'{prefix}center']  # B, Q, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,Q,3) (l,w,h)

        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        
        # step compute similarity between vision and text
        proj_tokens = end_points['proj_tokens']             # text feature   (B, 256, 64)
        proj_queries = end_points[f'{prefix}proj_queries']  # vision feature (B, 256, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))  # similarity ([B, 256, L]) 
        sem_scores_ = (sem_scores / 0.07).softmax(-1)                           # softmax ([B, 256, L])
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256) # ([B, 256, 256])
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_ # ([B, P=256, L=256])

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            is_correct = None
            if self.filter_non_gt_boxes:  # this works only for the target box
                ious, _ = _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(
                        end_points['all_detected_boxes'][bid][
                            end_points['all_detected_bbox_label_mask'][bid]
                        ]
                    ),  # (gt, 6)
                    box_cxcyczwhd_to_xyzxyz(pred_bbox[bid])  # (Q, 6)
                )  # (gt, Q)
                is_correct = (ious.max(0)[0] > 0.25) * 1.0
            
            # Keep scores for annotated objects only
            num_obj = int(end_points['box_label_mask'][bid].sum())
            pmap = positive_map[bid, :num_obj]
            scores_main = (
                sem_scores[bid].unsqueeze(0)  # (1, Q, 256)
                * pmap.unsqueeze(1)  # (obj, 1, 256)
            ).sum(-1)  # (obj, Q)
            
            # score
            pmap_modi = modify_positive_map[bid, :1]
            pmap_pron = pron_positive_map[bid, :1]
            pmap_other = other_entity_map[bid, :1]
            pmap_auxi = auxi_entity_positive_map[bid, :1]
            pmap_rel = rel_positive_map[bid, :1]
            scores_modi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_modi.unsqueeze(1)             
            ).sum(-1)
            scores_pron = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_pron.unsqueeze(1)             
            ).sum(-1)
            scores_other = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_other.unsqueeze(1)             
            ).sum(-1)
            scores_auxi = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_auxi.unsqueeze(1)             
            ).sum(-1)
            scores_rel = (
                sem_scores[bid].unsqueeze(0)    
                * pmap_rel.unsqueeze(1)             
            ).sum(-1)

            # total score
            scores = scores_main + scores_modi + scores_pron + scores_rel - scores_other

            if is_correct is not None:
                scores = scores * is_correct[None]

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, :10]  # (obj, 10)
            pbox = pred_bbox[bid, top.reshape(-1)]

            # IoU
            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (obj, 6)
                box_cxcyczwhd_to_xyzxyz(pbox)  # (obj*10, 6)
            )  # (obj, obj*10)
            ious = ious.reshape(top.size(0), top.size(0), top.size(1))
            ious = ious[torch.arange(len(ious)), torch.arange(len(ious))]

            # Check for bad cases
            if self.bad_case_visualization:
                wandb.init(project="vis", name="badcase")
                bad_cases = ious < self.bad_case_threshold  # Here you set your bad_case_threshold
                if bad_cases.any():
                    # Get point cloud and original color
                    point_cloud = end_points['point_clouds'][bid]
                    og_color = end_points['og_color'][bid]
                    point_cloud[:, 3:] = (og_color + torch.tensor([109.8, 97.2, 83.8]).cuda() / 256) * 256
                    target_name = end_points['target_name'][bid]
                    utterances = end_points['utterances'][bid]

                    # Get all boxes and predicted boxes
                    topk_boxes = 0
                    all_bboxes = end_points['all_bboxes'][bid].cpu()
                    pbox_bad_cases = pbox[bad_cases[0]].cpu()[topk_boxes].unsqueeze(0)  # top 1
                    gt_box = gt_bboxes[bid].cpu()

                    # Convert boxes to points for visualization
                    all_boxes_points = box2points(all_bboxes[..., :6])  # all boxes
                    gt_box = box2points(gt_box[..., :6])  # gt boxes
                    pbox_bad_cases_points = box2points(pbox_bad_cases[..., :6])

                    # Log bad case visualization to wandb
                    wandb.log({
                        "bad_case_point_scene": wandb.Object3D({
                            "type": "lidar/beta",
                            "points": point_cloud,
                            "boxes": np.array(
                                [  # actual boxes
                                    {
                                        "corners": c.tolist(),
                                        "label": "actual",
                                        "color": [0, 255, 0]
                                    }
                                    for c in gt_box
                                ]
                                + [  # predicted boxes
                                    {
                                        "corners": c.tolist(),
                                        "label": "predicted",
                                        "color": [255, 0, 0]
                                    }
                                    for c in pbox_bad_cases_points
                                ]
                            )
                        }),
                        "target_name": wandb.Html(target_name),
                        "utterance": wandb.Html(utterances),
                    })

            # Check for kps points
            if self.kps_points_visualization:
                wandb.init(project="vis", name="kps_points")
                point_cloud = end_points['point_clouds'][bid]
                og_color = end_points['og_color'][bid]
                point_cloud[:, 3:] = (og_color + torch.tensor([109.8, 97.2, 83.8]).cuda() / 256) * 256
                kps_points = end_points['query_points_xyz'][bid]
                red = torch.zeros((256, 3)).cuda()
                red[:, 0] = 255.0
                kps_points = torch.cat([kps_points, red], dim=1)
                total_point = torch.cat([point_cloud, kps_points], dim=0)
                utterances = end_points['utterances'][bid]
                gt_box = gt_bboxes[bid].cpu()
                gt_box = box2points(gt_box[..., :6])

                wandb.log({
                        "kps_point_scene": wandb.Object3D({
                            "type": "lidar/beta",
                            "points": total_point,
                            "boxes": np.array(
                                [
                                    {
                                        "corners": c.tolist(),
                                        "label": "target",
                                        "color": [0, 255, 0]
                                    }
                                    for c in gt_box
                                ]
                            )
                        }),
                        "utterance": wandb.Html(utterances),
                    })

            # step Measure IoU>threshold, ious are (obj, 10)
            for t in self.thresholds:
                thresholded = ious > t
                for k in self.topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, 'bbf')] += found.sum().item()
                    self.gts[(prefix, t, k, 'bbf')] += len(thresholded)
                    if prefix == 'last_':
                        found = found[0].item()
                        if k == 1 and t == self.thresholds[0]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd'] += 1
                                self.dets['vd'] += found
                            else:
                                self.gts['vid'] += 1
                                self.dets['vid'] += found
                            if end_points['is_hard'][bid]:
                                self.gts['hard'] += 1
                                self.dets['hard'] += found
                            else:
                                self.gts['easy'] += 1
                                self.dets['easy'] += found
                            if end_points['is_unique'][bid]:
                                self.gts['unique'] += 1
                                self.dets['unique'] += found
                            else:
                                self.gts['multi'] += 1
                                self.dets['multi'] += found
                        if k == 1 and t == self.thresholds[1]:
                            if end_points['is_view_dep'][bid]:
                                self.gts['vd50'] += 1
                                self.dets['vd50'] += found
                            else:
                                self.gts['vid50'] += 1
                                self.dets['vid50'] += found
                            if end_points['is_hard'][bid]:
                                self.gts['hard50'] += 1
                                self.dets['hard50'] += found
                            else:
                                self.gts['easy50'] += 1
                                self.dets['easy50'] += found
                            if end_points['is_unique'][bid]:
                                self.gts['unique50'] += 1
                                self.dets['unique50'] += found
                            else:
                                self.gts['multi50'] += 1
                                self.dets['multi50'] += found


    # BRIEF Get the postion label of the decoupled text component.
    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points['positive_map'])                  # main
        modify_positive_map = torch.clone(end_points['modify_positive_map'])    # attribute
        pron_positive_map = torch.clone(end_points['pron_positive_map'])        # pron
        other_entity_map = torch.clone(end_points['other_entity_map'])          # other(including auxi)
        auxi_entity_positive_map = torch.clone(end_points['auxi_entity_positive_map'])  # auxi
        rel_positive_map = torch.clone(end_points['rel_positive_map'])

        positive_map[positive_map > 0] = 1                      
        gt_center = end_points['center_label'][:, :, 0:3]       
        gt_size = end_points['size_gts']                        
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)     # GT box cxcyczwhd
        
        if self.only_root:
            positive_map = positive_map[:, :1]  # (B, 1, 256)
            gt_bboxes = gt_bboxes[:, :1]        # (B, 1, 6)
        
        return positive_map, modify_positive_map, pron_positive_map, other_entity_map, auxi_entity_positive_map, \
            rel_positive_map, gt_bboxes


def box2points(box):
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:, :3] - (box[:, 3:] / 2)).transpose(1, 0)
    x_max, y_max, z_max = (box[:, :3] + (box[:, 3:] / 2)).transpose(1, 0)
    return np.stack((
        np.concatenate((x_min[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_max[:, None]), 1)
    ), axis=1)


def points2box(box):
    """Convert vertices (Nx8x3) to box center/hwd coordinates (Nx6)."""
    return np.concatenate((
        (box.min(1) + box.max(1)) / 2,
        box.max(1) - box.min(1)
    ), axis=1)