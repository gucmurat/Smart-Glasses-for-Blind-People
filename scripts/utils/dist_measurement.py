import copy
import math
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import torch
import torchvision
import torchvision.transforms.functional as tvtf
from pathlib import Path
import time

COLOURS = [
    tuple(int(colour_hex.strip('#')[i:i+2], 16) for i in (0, 2, 4))
    for colour_hex in plt.rcParams['axes.prop_cycle'].by_key()['color']
]

def tlbr_to_center1(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        cx = (tlx+brx)/2
        cy = (tly+bry)/2
        points.append([cx, cy])
    return points

def tlbr_to_corner(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        cx = (tlx+tlx)/2
        cy = (tly+tly)/2
        points.append((cx, cy))
    return points

def tlbr_to_corner_br(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        cx = (brx+brx)/2
        cy = (bry+bry)/2
        points.append((cx, cy))
    return points

def tlbr_to_area(boxes):
    areas = []
    for tlx, tly, brx, bry in boxes:
        cx = (brx-tlx)
        cy = (bry-tly)
        areas.append(abs(cx*cy))
    return areas

def get_horiz_dist_centre(boxes):
    pnts1 = np.array(tlbr_to_center1(boxes[0]))[:,0]
    pnts2 = np.array(tlbr_to_center1(boxes[1]))[:,0]
    return pnts1[:,None] - pnts2[None]

def get_horiz_dist_corner_tl(boxes):
    pnts1 = np.array(tlbr_to_corner(boxes[0]))[:,0]
    pnts2 = np.array(tlbr_to_corner(boxes[1]))[:,0]
    return pnts1[:,None] - pnts2[None]

def get_horiz_dist_corner_br(boxes):
    pnts1 = np.array(tlbr_to_corner_br(boxes[0]))[:,0]
    pnts2 = np.array(tlbr_to_corner_br(boxes[1]))[:,0]
    return pnts1[:,None] - pnts2[None]

def get_vertic_dist_centre(boxes):
    pnts1 = np.array(tlbr_to_center1(boxes[0]))[:,1]
    pnts2 = np.array(tlbr_to_center1(boxes[1]))[:,1]
    return pnts1[:,None] - pnts2[None]

def get_area_diffs(boxes):
    pnts1 = np.array(tlbr_to_area(boxes[0]))
    pnts2 = np.array(tlbr_to_area(boxes[1]))
    return abs(pnts1[:,None] - pnts2[None])

def get_cost(boxes, lbls = None, sz1 = 400):
    alpha = sz1; beta  = 10; gamma = 5
    
    vert_dist = gamma*abs(get_vertic_dist_centre(boxes))
    
    horiz_dist = get_horiz_dist_centre(boxes)
    
    horiz_dist[horiz_dist<0] = beta*abs(horiz_dist[horiz_dist<0])
    
    area_diffs = get_area_diffs(boxes)/alpha
    
    cost = np.array([vert_dist,horiz_dist,area_diffs])
    
    cost=cost.sum(axis=0)
    
    if lbls is not None:
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                if (lbls[0][i]!=lbls[1][j]):
                    cost[i,j]+=150
    return cost

def get_horiz_dist(masks, prob_thresh = 0.7):
    
    mask_bool = masks[0] > prob_thresh
    mask_bool = mask_bool.squeeze(1)
    
    mask_bool2 = masks[1] > prob_thresh
    mask_bool2 = mask_bool2.squeeze(1)
    
    
    mask_size = (mask_bool).sum(dim=[1,2])
    mask_com_matrix_1 = torch.tensor(range(mask_bool.shape[1]))
    com1 = ((mask_com_matrix_1.unsqueeze(1))*mask_bool).sum(dim=[1,2])/mask_size
    mask_com_matrix_2 = torch.tensor(range(mask_bool.shape[2]))
    com2 = ((mask_com_matrix_2.unsqueeze(0))*mask_bool).sum(dim=[1,2])/mask_size

    left_params = torch.stack((com1, com2, mask_size)).transpose(1,0)
    
    
    mask_size2 = (mask_bool2).sum(dim=[1,2])
    mask_com_matrix_12 = torch.tensor(range(mask_bool2.shape[1]))
    com12 = ((mask_com_matrix_12.unsqueeze(1))*mask_bool2).sum(dim=[1,2])/mask_size2
    mask_com_matrix_22 = torch.tensor(range(mask_bool2.shape[2]))
    com22 = ((mask_com_matrix_22.unsqueeze(0))*mask_bool2).sum(dim=[1,2])/mask_size2

    right_params = torch.stack((com12, com22, mask_size2)).transpose(1,0)
    
    
    cost = (left_params[:,None] - right_params[None])
    return cost[:,:,1]

def get_tracks(cost):
    return scipy.optimize.linear_sum_assignment(cost)
    

def get_tracks_ij(cost):
    tracks = scipy.optimize.linear_sum_assignment(cost)
    return [[i,j] for i, j in zip(*tracks)]


def measure_dist(image_left, image_right, labels_boxes_json_left, labels_boxes_json_right):
    """fl = -37.1140644247583
    tantheta = 0.230487718076677
    """
    fl = -11.7232268698777
    tantheta = 0.338647631699787
    sz1 = image_right.shape[1]
    sz2 = image_right.shape[0]
    det = [labels_boxes_json_left["boxes"], labels_boxes_json_right["boxes"]]
    lbls = [labels_boxes_json_left["classes"], labels_boxes_json_right["classes"]]
    centre = sz1/2
    def get_dist_to_centre_tl(box, cntr = centre):
        pnts = np.array(tlbr_to_corner(box))[:,0]
        return abs(pnts - centre)
    
    def get_dist_to_centre_br(box, cntr = centre):
        pnts = np.array(tlbr_to_corner_br(box))[:,0]
        return abs(pnts - centre)
    cost = get_cost(det, lbls = lbls)
    tracks = scipy.optimize.linear_sum_assignment(cost)
    
    dists_tl =  get_horiz_dist_corner_tl(det)
    dists_br =  get_horiz_dist_corner_br(det)
    final_dists = []
    dctl = get_dist_to_centre_tl(det[0])
    dcbr = get_dist_to_centre_br(det[0])
    for i, j in zip(*tracks):
        if len(lbls) <= i:
            continue
        if dctl[i] < dcbr[i]:
            final_dists.append((dists_tl[i][j],lbls[i]))
        else:
            final_dists.append((dists_br[i][j],lbls[i]))
    fd = [i for (i,j) in final_dists]
    dists_away = (4.8/2)*sz1*(1/tantheta)/np.array(fd)+fl
    return dists_away, det


