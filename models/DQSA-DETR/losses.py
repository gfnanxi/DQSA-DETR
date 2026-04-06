import torch
import torch.nn.functional as F
from util import box_ops

def SA_IA_BCE_loss(
    src_logits,
    pos_idx_c,
    src_boxes,
    target_boxes,
    avg_factor,
    alpha=0.5,
    gamma=2.0
):
    prob = src_logits.sigmoid()
    pos_weights = torch.zeros_like(src_logits)
    neg_weights = prob ** gamma

    # IoU via box_ops (返回 tuple)
    iou_tuple = box_ops.box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(target_boxes)
    )
    iou_matrix = iou_tuple[0]
    iou_scores = torch.diag(iou_matrix)

    # -----------------------------
    # 面积感知 IoU（关键改动）
    # -----------------------------
    areas = src_boxes[:, 2] * src_boxes[:, 3] + 1e-6  # w * h
    mean_area = areas.mean()

    # 小目标 area_factor < 1 → IoU 惩罚减弱
    area_factor = (areas / mean_area).clamp(min=1, max=2.0)

    # 调整 IoU：小目标提升 IoU（指数 < 1），大目标降低 IoU（指数 > 1）
    iou_scores_adj = iou_scores ** (1.0 / area_factor)
  
    # soft target t
    t = prob[pos_idx_c] ** alpha * iou_scores_adj ** (1 - alpha)
    t = torch.clamp(t, min=0.01).detach()

    pos_weights[pos_idx_c] = t
    neg_weights[pos_idx_c] = 1 - t

    eps = 1e-6
    loss = (
        -pos_weights * prob.clamp(min=eps).log()
        -neg_weights * (1 - prob).clamp(min=eps).log()
    )
    return loss.sum() / avg_factor
# import torch
# import torch.nn.functional as F
# from util import box_ops

# def IA_BCE_loss(
#     src_logits,
#     pos_idx_c,
#     src_boxes,
#     target_boxes,
#     avg_factor,
#     alpha=0.5,
#     gamma=2.0
# ):
#     prob = src_logits.sigmoid()
#     pos_weights = torch.zeros_like(src_logits)
#     neg_weights = prob ** gamma

#     # 计算 IoU
#     iou_scores = torch.diag(
#         box_iou(
#             box_cxcywh_to_xyxy(src_boxes),
#             box_cxcywh_to_xyxy(target_boxes)
#         )[0]
#     )

#     # soft target t
#     t = prob[pos_idx_c] ** alpha * iou_scores ** (1 - alpha)
#     t = torch.clamp(t, min=0.01).detach()

#     # 直接使用 t 作为 soft label（不再乘任何 w_prime）
#     pos_weights[pos_idx_c] = t
#     neg_weights[pos_idx_c] = 1 - t

#     eps = 1e-6
#     loss = (
#         -pos_weights * prob.clamp(min=eps).log()
#         -neg_weights * (1 - prob).clamp(min=eps).log()
#     )
#     return loss.sum() / avg_factor

# def SA_IA_BCE_loss(
#     src_logits,
#     pos_idx_c,
#     src_boxes,
#     target_boxes,
#     avg_factor,
#     alpha=0.5,
#     gamma=2.0
# ):
#     prob = src_logits.sigmoid()
#     pos_weights = torch.zeros_like(src_logits)
#     neg_weights = prob ** gamma

#     # IoU
#     iou_scores = torch.diag(
#         box_iou(
#             box_cxcywh_to_xyxy(src_boxes),
#             box_cxcywh_to_xyxy(target_boxes)
#         )[0]
#     )

#     # -----------------------------
#     # 面积感知 IoU（关键改动）
#     # -----------------------------
#     # src_boxes: (cx, cy, w, h)
#     areas = src_boxes[:, 2] * src_boxes[:, 3] + 1e-6
#     mean_area = areas.mean()

#     # 小目标 area_factor < 1 → IoU 惩罚减弱
#     area_factor = (areas / mean_area).clamp(min=0.3, max=3.0)

#     # 仅对正样本使用
#     iou_scores_adj = iou_scores ** (1.0 / area_factor)
#     # iou_scores_adj = iou_scores ** area_factor

#     # soft target t
#     t = prob[pos_idx_c] ** alpha * iou_scores_adj ** (1 - alpha)
#     t = torch.clamp(t, min=0.01).detach()

#     # soft label
#     pos_weights[pos_idx_c] = t
#     neg_weights[pos_idx_c] = 1 - t

#     eps = 1e-6
#     loss = (
#         -pos_weights * prob.clamp(min=eps).log()
#         -neg_weights * (1 - prob).clamp(min=eps).log()
#     )
#     return loss.sum() / avg_factor
