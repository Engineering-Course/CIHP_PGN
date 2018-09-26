from __future__ import division
import os
from PIL import Image
import numpy as np

PREDICT_DIR = '../output/cihp_instance_part_maps'
INST_PART_GT_DIR = './Instance_part_val'

CLASSES = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
               'dress', 'coat', 'socks', 'pants', 'tosor-skin', 'scarf', 'skirt',
               'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe', 'rightShoe']

IOU_THRE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#IOU_THRE = [0.1]

# compute mask overlap
def compute_mask_iou(mask_gt, masks_pre, mask_gt_area, masks_pre_area):
    """Calculates IoU of the given box with the array of the given boxes.
    mask_gt: [H,W] # a mask of gt
    masks_pre: [num_instances, height, width] predict Instance masks
    mask_gt_area: the gt_mask_area , int
    masks_pre_area: array of length masks_count. including all predicted mask, sum

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    intersection = np.logical_and(mask_gt, masks_pre)
    intersection = np.where(intersection == True, 1, 0).astype(np.uint8)
    intersection = NonZero(intersection)

    # print('intersection:', intersection)

    mask_gt_areas = np.full(len(masks_pre_area), mask_gt_area)

    union = mask_gt_areas + masks_pre_area[:] - intersection[:]

    iou = intersection / union

    return iou


# compute the number of nonzero in mask
def NonZero(masks):
    """
    :param masks: [N,h,w] a three-dimension array, includes N two-dimension mask arrays
    :return: (N) return a tuple with length N. N is the number of non-zero elements in the two-dimension mask
    """
    area = []  
    # print('NonZero masks',masks.shape)
    for i in masks:
        _, a = np.nonzero(i)
        area.append(a.shape[0])
    area = tuple(area)
    return area



def compute_mask_overlaps(masks_pre, masks_gt):
    """Computes IoU overlaps between two sets of boxes.
    masks_pre, masks_gt:
    masks_pre  [num_instances,height, width] Instance masks
    masks_gt ground truth
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of masks_pre and masks_gt , get the nubmer of the non-zero element
    area1 = NonZero(masks_pre)
    area2 = NonZero(masks_gt)

    # print(masks_pre.shape, masks_gt.shape)(1, 375, 1) (500, 375, 1)

    # Compute overlaps to generate matrix [masks count, masks_gt count]
    # Each cell contains the IoU value.
    # print('area1',len(area1))
    # print('area1',len(area2))
    overlaps = np.zeros((masks_pre.shape[0], masks_gt.shape[0]))
    # print('overlaps:',overlaps.shape)
    for i in range(overlaps.shape[1]):
        mask_gt = masks_gt[i]
        # print('overlaps:',overlaps)
        overlaps[:, i] = compute_mask_iou(mask_gt, masks_pre, area2[i], area1)

    return overlaps

def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall. If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    Args:
        rec: recall
        prec: precision
        use_07_metric:
    Returns:
        ap: average precision
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        # arange([start, ]stop, [step, ]dtype=None)
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def convert2evalformat(inst_id_map, id_to_convert=None):
    """
    param: 
        inst_id_map:[h, w]
        id_to_convert: a set
    return: 
        masks:[instances,h, w]
    """
    masks = []

    inst_ids = np.unique(inst_id_map)
    # print("inst_ids:", inst_ids)
    background_ind = np.where(inst_ids == 0)[0]
    inst_ids = np.delete(inst_ids, background_ind)

    if id_to_convert == None:
        for i in inst_ids:
            im_mask = (inst_id_map == i).astype(np.uint8)
            masks.append(im_mask)
    else:
        for i in inst_ids:
            if i not in id_to_convert:
                continue
            im_mask = (inst_id_map == i).astype(np.uint8)
            masks.append(im_mask)

    return masks, len(masks)

def compute_class_ap(image_id_list, class_id, iou_threshold):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Input:
    image_id_list : all pictures id list
    gt_masks:all mask  [N_pictures,num_inst,H,W]
    pre_masks:all predict masks [N_pictures,num_inst,H,W]
    pred_scores:scores for every predicted mask [N_pre_mask]
    pred_class_ids: the indices of all predicted masks

    Returns:
    AP: Average Precision of specific class
    """

    iou_thre_num = len(iou_threshold)
    ap = np.zeros((iou_thre_num,))

    gt_mask_num = 0
    pre_mask_num = 0
    tp = []
    fp = []
    scores = []
    for i in range(iou_thre_num):
        tp.append([])
        fp.append([])
    
    print("process class", CLASSES[class_id], class_id)

    for image_id in image_id_list:

        # print (image_id)
        inst_part_gt = Image.open(os.path.join(INST_PART_GT_DIR, '%s.png' % image_id))
        inst_part_gt = np.array(inst_part_gt)
        rfp = open(os.path.join(INST_PART_GT_DIR, '%s.txt' % image_id), 'r')
        gt_part_id = []
        for line in rfp.readlines():
            line = line.strip().split(' ')
            gt_part_id.append([int(line[0]), int(line[1])])
        rfp.close()


        pre_img = Image.open(os.path.join(PREDICT_DIR, '%s.png' % image_id))
        pre_img = np.array(pre_img)
        rfp = open(os.path.join(PREDICT_DIR, '%s.txt' % image_id), 'r')
        items = [x.strip().split(' ') for x in rfp.readlines()]
        rfp.close()

        pre_id = []
        pre_scores = []
        for i in range(len(items)):
            if int(items[i][0]) == class_id:
                pre_id.append(i+1)
                pre_scores.append(float(items[i][1]))

        gt_id = []
        for i in range(len(gt_part_id)):
            if gt_part_id[i][1] == class_id:
                gt_id.append(gt_part_id[i][0])


        gt_mask, n_gt_inst = convert2evalformat(inst_part_gt, set(gt_id))
        pre_mask, n_pre_inst = convert2evalformat(pre_img, set(pre_id))


        gt_mask_num += n_gt_inst
        pre_mask_num += n_pre_inst

        if n_pre_inst == 0:
            continue

        scores += pre_scores
        
        if n_gt_inst == 0:
            for i in range(n_pre_inst):
                for k in range(iou_thre_num):
                    fp[k].append(1)
                    tp[k].append(0)
            continue

        gt_mask = np.stack(gt_mask)
        pre_mask = np.stack(pre_mask)
        # Compute IoU overlaps [pred_masks, gt_makss]
        overlaps = compute_mask_overlaps(pre_mask, gt_mask)

        # print('overlaps.shape',overlaps.shape)

        max_overlap_ind = np.argmax(overlaps, axis=1)

         # l = len(overlaps[:,max_overlap_ind])
        for i in np.arange(len(max_overlap_ind)):
            max_iou = overlaps[i][max_overlap_ind[i]]
            # print('max_iou :', max_iou)
            for k in range(iou_thre_num):
                if max_iou > iou_threshold[k]:
                    tp[k].append(1)
                    fp[k].append(0)
                else:
                    tp[k].append(0)
                    fp[k].append(1)

    ind = np.argsort(scores)[::-1]


    for k in range(iou_thre_num):
        m_tp = tp[k]
        m_fp = fp[k]
        m_tp = np.array(m_tp)
        m_fp = np.array(m_fp)

        m_tp = m_tp[ind]
        m_fp = m_fp[ind]

        m_tp = np.cumsum(m_tp)
        m_fp = np.cumsum(m_fp)
        # print('m_tp : ',m_tp)
        # print('m_fp : ', m_fp)
        recall = m_tp / float(gt_mask_num)
        precition = m_tp / np.maximum(m_fp+m_tp, np.finfo(np.float64).eps)

        # Compute mean AP over recall range
        ap[k] = voc_ap(recall, precition, False)
        
    return ap



if __name__ == '__main__':
    print("result of", PREDICT_DIR)

    image_id_list = [x[:-4] for x in os.listdir(PREDICT_DIR) if x[-3:] == 'txt']

    AP = np.zeros((len(CLASSES)-1, len(IOU_THRE)))

    for ind in range(1, len(CLASSES)):
        AP[ind - 1, :] = compute_class_ap(image_id_list, ind, IOU_THRE)
    print("-----------------AP-----------------")
    print(AP)
    print("-------------------------------------")
    mAP = np.mean(AP, axis=0)
    print("-----------------mAP-----------------")
    print(mAP)
    print(np.mean(mAP))
    print("-------------------------------------")
