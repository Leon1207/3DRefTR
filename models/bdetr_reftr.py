# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast

from .backbone_module import Pointnet2Backbone
from .modules import (
    PointsObjClsModule, GeneralSamplingModule,
    ClsAgnosticPredictHead, PositionEmbeddingLearned
)
from .encoder_decoder_layers import (
    BiEncoder, BiEncoderLayer, BiDecoderLayer
)

from pointnet2_modules import PointnetFPModule
import pytorch_utils as pt_utils

class BeaUTyDETR_reftr(nn.Module):
    """
    3D language grounder.

    Args:
        num_class (int): number of semantics classes to predict
        num_obj_class (int): number of object classes
        input_feature_dim (int): feat_dim of pointcloud (without xyz)
        num_queries (int): Number of queries generated
        num_decoder_layers (int): number of decoder layers
        self_position_embedding (str or None): how to compute pos embeddings
        contrastive_align_loss (bool): contrast queries and token features
        d_model (int): dimension of features
        butd (bool): use detected box stream
        pointnet_ckpt (str or None): path to pre-trained pp++ checkpoint
        self_attend (bool): add self-attention in encoder
    """

    def __init__(self, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=288, butd=True, pointnet_ckpt=None, data_path=None,
                 self_attend=True):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd

        # Visual encoder
        self.backbone_net = Pointnet2Backbone(
            input_feature_dim=input_feature_dim,
            width=1
        )
        if input_feature_dim == 3 and pointnet_ckpt is not None:
            self.backbone_net.load_state_dict(torch.load(
                pointnet_ckpt
            ), strict=False)

        # Text Encoder
        # # (1) online
        # t_type = "roberta-base"
        # NOTE (2) offline: load from the local folder.
        t_type = f'{data_path}roberta-base/'
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type, local_files_only=True)
        self.text_encoder = RobertaModel.from_pretrained(t_type, local_files_only=True)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )

        # Box encoder
        if self.butd:
            self.butd_class_embeddings = nn.Embedding(num_obj_class, 768)
            saved_embeddings = torch.from_numpy(np.load(
                'data/class_embeddings3d.npy', allow_pickle=True
            ))
            self.butd_class_embeddings.weight.data.copy_(saved_embeddings)
            self.butd_class_embeddings.requires_grad = False
            self.class_embeddings = nn.Linear(768, d_model - 128)
            self.box_embeddings = PositionEmbeddingLearned(6, 128)

        # Cross-encoder
        self.pos_embed = PositionEmbeddingLearned(3, d_model)
        bi_layer = BiEncoderLayer(
            d_model, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=256,
            self_attend_lang=self_attend, self_attend_vis=self_attend,
            use_butd_enc_attn=butd
        )
        self.cross_encoder = BiEncoder(bi_layer, 3)

        # Query initialization
        self.points_obj_cls = PointsObjClsModule(d_model)
        self.gsample_module = GeneralSamplingModule()
        self.decoder_query_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Proposal (layer for size and center)
        self.proposal_head = ClsAgnosticPredictHead(
            num_class, 1, num_queries, d_model,
            objectness=False, heading=False,
            compute_sem_scores=True
        )

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.decoder.append(BiDecoderLayer(
                d_model, n_heads=8, dim_feedforward=256,
                dropout=0.1, activation="relu",
                self_position_embedding=self_position_embedding, butd=self.butd
            ))

        # Prediction heads
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.prediction_heads.append(ClsAgnosticPredictHead(
                num_class, 1, num_queries, d_model,
                objectness=False, heading=False,
                compute_sem_scores=True
            ))

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )
            self.contrastive_align_projection_text = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )
        
        # Segmentation Head
        self.bbox_attention = MHAttentionMap(query_dim=d_model, hidden_dim=d_model, num_heads=8, dropout=0)
        self.src_project = nn.Linear(d_model, d_model)
        self.mask_head = MaskHeadSmallConv(dim=d_model * 2 + 8, context_dim=d_model)

        # Init
        self.init_bn_momentum()
    # BRIEF visual and text backbones.
    def _run_backbones(self, inputs):
        """Run visual and text backbones."""
        # step 1. Visual encoder
        end_points = self.backbone_net(inputs['point_clouds'], end_points={})
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = end_points['fp2_xyz']
        end_points['seed_features'] = end_points['fp2_features']

        # for segmentation head
        end_points['sa0_xyz'] = inputs['point_clouds'][..., 0:3].contiguous()
        end_points['sa0_features'] = inputs['point_clouds'][..., 3:].transpose(1, 2).contiguous()

        # step 2. Text encoder
        tokenized = self.tokenizer.batch_encode_plus(
            inputs['text'], padding="longest", return_tensors="pt"
        ).to(inputs['point_clouds'].device)

        encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_projector(encoded_text.last_hidden_state)

        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        end_points['text_feats'] = text_feats
        end_points['text_attention_mask'] = text_attention_mask
        end_points['tokenized'] = tokenized
        return end_points

    # BRIEF generate query.
    def _generate_queries(self, xyz, features, end_points):
        # kps sampling
        points_obj_cls_logits = self.points_obj_cls(features)
        end_points['seeds_obj_cls_logits'] = points_obj_cls_logits

        # top-k
        sample_inds = torch.topk(   
            torch.sigmoid(points_obj_cls_logits).squeeze(1),
            self.num_queries
        )[1].int()

        xyz, features, sample_inds = self.gsample_module(   
            xyz, features, sample_inds
        )

        end_points['query_points_xyz'] = xyz  # (B, V, 3)
        end_points['query_points_feature'] = features  # (B, F, V)
        end_points['query_points_sample_inds'] = sample_inds  # (B, V)
        return end_points

    # BRIEF forward.
    def forward(self, inputs):
        """
        Forward pass.
        Args:
            inputs: dict
                {point_clouds, text}
                point_clouds (tensor): (B, Npoint, 3 + input_channels)
                text (list): ['text0', 'text1', ...], len(text) = B

                more keys if butd is enabled:
                    det_bbox_label_mask
                    det_boxes
                    det_class_ids
        Returns:
            end_points: dict
        """
        # STEP 1. vision and text encoding
        end_points = self._run_backbones(inputs)
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        text_feats = end_points['text_feats']
        text_padding_mask = end_points['text_attention_mask']
        
        # STEP 2. Box encoding
        if self.butd:
            # attend on those features
            detected_mask = ~inputs['det_bbox_label_mask']

            # step box position.    det_boxes ([B, 132, 6]) -->  ([B, 128, 132])
            box_embeddings = self.box_embeddings(inputs['det_boxes'])
            # step box class        det_class_ids ([B, 132])  -->  ([B, 132, 160])
            class_embeddings = self.class_embeddings(self.butd_class_embeddings(inputs['det_class_ids']))
            # step box feature     ([B, 132, 288])
            detected_feats = torch.cat([box_embeddings, class_embeddings.transpose(1, 2)]
                                       , 1).transpose(1, 2).contiguous()
        else:
            detected_mask = None
            detected_feats = None

        # STEP 3. Cross-modality encoding
        points_features, text_feats = self.cross_encoder(
            vis_feats=points_features.transpose(1, 2).contiguous(),
            pos_feats=self.pos_embed(points_xyz).transpose(1, 2).contiguous(),
            padding_mask=torch.zeros(
                len(points_xyz), points_xyz.size(1)
            ).to(points_xyz.device).bool(),
            text_feats=text_feats,
            text_padding_mask=text_padding_mask,
            end_points=end_points,
            detected_feats=detected_feats,
            detected_mask=detected_mask
        )

        points_features = points_features.transpose(1, 2)
        points_features = points_features.contiguous()
        end_points["text_memory"] = text_feats
        end_points['seed_features'] = points_features  # V', [B, 288, 1024]

        # STEP 4. text projection --> 64
        if self.contrastive_align_loss:
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(text_feats), p=2, dim=-1
            )
            end_points['proj_tokens'] = proj_tokens  # ([B, L, 64])

        # STEP 5. Query Points Generation
        end_points = self._generate_queries(
            points_xyz, points_features, end_points
        )
        cluster_feature = end_points['query_points_feature']  # (B, F=288, V=256)
        cluster_xyz = end_points['query_points_xyz']  # (B, V=256, 3)
        query = self.decoder_query_proj(cluster_feature)
        query = query.transpose(1, 2).contiguous()  # (B, V=256, F=288)
        # projection 288 --> 64
        if self.contrastive_align_loss:
            end_points['proposal_proj_queries'] = F.normalize(
                self.contrastive_align_projection_image(query), p=2, dim=-1
            )

        # STEP 6.Proposals
        proposal_center, proposal_size = self.proposal_head(
            cluster_feature,
            base_xyz=cluster_xyz,
            end_points=end_points,
            prefix='proposal_'
        )
        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()
        query_mask = None

        # STEP 7. Decoder
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if i == self.num_decoder_layers - 1 else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError

            # step Transformer Decoder Layer
            query = self.decoder[i](
                query, points_features.transpose(1, 2).contiguous(),
                text_feats, query_pos,
                query_mask,
                text_padding_mask,
                detected_feats=(
                    detected_feats if self.butd
                    else None
                ),
                detected_mask=detected_mask if self.butd else None
            )  # (B, V, F)

            if i == self.num_decoder_layers - 1:
                end_points['query_last_layer'] = query  # O' [B, 256, 288]

            # step project
            if self.contrastive_align_loss:
                end_points[f'{prefix}proj_queries'] = F.normalize(
                    self.contrastive_align_projection_image(query), p=2, dim=-1
                )

            # step box Prediction head
            base_xyz, base_size = self.prediction_heads[i](
                query.transpose(1, 2).contiguous(),  # ([B, F=288, V=256])
                base_xyz=cluster_xyz,  # ([B, 256, 3])
                end_points=end_points,  #
                prefix=prefix
            )
            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

        # segmentation head
        decoder_hs = end_points['query_last_layer']  # [B, 256, 288]
        memory_visual = end_points['seed_features']  # [B, 288, 1024]
        bbox_mask = self.bbox_attention(decoder_hs, memory_visual)  # [B, 256, head=8, 1024]
        point_src = self.src_project(end_points['fp2_features'].transpose(1, 2)).transpose(1, 2)  # [B, 288, 1024]
        point_src = torch.cat([point_src, memory_visual], dim=1)  # [B, 576, 1024]
        seg_masks = self.mask_head(point_src, bbox_mask, end_points)
        end_points['pred_masks'] = seg_masks  # [B * 256, 1, 50000]
        # end_points['res_feat'] = res_feat  # [B * 256, 36, 50000]

        return end_points

    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1


class MHAttentionMap(nn.Module):

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k):
        q = self.q_linear(q)  # [B, 256, 288]
        k = self.k_linear(k.transpose(1, 2)).transpose(1, 2)  # [B, 288, 1024]
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)  # [B, 256, head, 288/head]
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[2])  # [B, head, 288/head, 1024]
        weights = torch.einsum("bqnc,bncp->bqnp", qh * self.normalize_fact, kh)  # [B, 256, head, 1024]
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)  # [B, 256, head, 1024]
        weights = self.dropout(weights)
        return weights
    

class MaskHeadSmallConv(nn.Module):

    def __init__(self, dim, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8]
        self.mlp1 = pt_utils.SharedMLP(args=[dim, dim], bn=True)
        self.mlp2 = pt_utils.SharedMLP(args=[dim, inter_dims[1]], bn=True)
        self.fp1 = PointnetFPModule(mlp=[inter_dims[1] * 2, inter_dims[1], inter_dims[2]])
        self.fp2 = PointnetFPModule(mlp=[inter_dims[2] * 2, inter_dims[2], inter_dims[3]])
        self.out_lay = nn.Linear(inter_dims[3], 1)

        self.adapter1 = nn.Linear(3, inter_dims[2])
        self.adapter2 = nn.Linear(128, inter_dims[1])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, end_points):

        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1).flatten(0, 1)
        num_query = bbox_mask.shape[1]

        x = expand(x, num_query)  # [B * 256, 576, 1024]
        x = torch.cat([x, bbox_mask.flatten(0, 1)], dim=1)  # [B * 256, 8 + 576, 1024]

        x = self.mlp1(x.unsqueeze(-1))  # [B * 256, 8 + 576, 1024, 1]
        x = self.mlp2(x).squeeze(-1)   # [B * 256, 144, 1024]

        sa0_features = end_points['sa0_features']  # [B, 3, 50000]
        sa0_features = self.adapter1(sa0_features.transpose(1, 2)).transpose(1, 2)  # [B, 72, 50000]
        sa0_features = expand(sa0_features, num_query)  # [B * 256, 72, 50000]
        sa1_features = end_points['sa1_features']  # [B, 128, 2048]
        sa1_features = self.adapter2(sa1_features.transpose(1, 2)).transpose(1, 2)  # [B, 144, 2048]
        sa1_features = expand(sa1_features, num_query)  # [B * 256, 144, 2048]

        sa0_xyz = end_points['sa0_xyz']  # [B, 50000, 3]
        sa0_xyz = expand(sa0_xyz, num_query)  # [B * 256, 50000, 3]
        sa1_xyz = end_points['sa1_xyz']  # [B, 2048, 3]
        sa1_xyz = expand(sa1_xyz, num_query)  # [B * 256, 2048, 3]
        sa2_xyz = end_points['sa2_xyz']  # [B, 1024, 3]
        sa2_xyz = expand(sa2_xyz, num_query)  # [B * 256, 1024, 3]

        x = self.fp1(sa1_xyz, sa2_xyz, sa1_features, x)  # [B * 256, 72, 2048]
        x = self.fp2(sa0_xyz, sa1_xyz, sa0_features, x)  # [B * 256, 36, 50000]
        out = self.out_lay(x.transpose(1, 2)).transpose(1, 2)  # [B * 256, 1, 50000]
       
        return out#, x