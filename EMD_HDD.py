import sys
sys.path.append('../utils/')


import numpy as np
import torch
import consts
from HDD_HDE import *
from PaviaClassifier import *
import torch.multiprocessing as mp
import DistancesHandler
from MetaLearner import HDDOnBands,HDD_HDE



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


def wasser_hdd(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='most_common', random_seed=None, M='', precomputed_pack=None):
    if precomputed_pack is None:
            if M!='hdd' and M!='euclidean':
                print("invalid M")
                exit(1)
            
            if M=='hdd':
                distances_bands = HDDOnBands.run(X, consts.METRIC_BANDS, None)
                distances_bands = distances_bands.to(device)
            elif M=='euclidean':
                tmp = torch.reshape(X, (X.shape[-1], -1)).float()
                distances_bands = torch.cdist(tmp, tmp)
            
            if is_normalize_each_band:
                X_tmp = HDD_HDE.normalize_each_band(X)
            else:
                X_tmp = X

            X_patches, _, _= HDD_HDE.patch_data_class(X_tmp, rows_factor, cols_factor, y, method_label_patch)
            distance_handler = DistancesHandler.DistanceHandler(consts.WASSERSTEIN,distances_bands)
            precomputed_distances = distance_handler.calc_distances(X_patches)
        
    else:
        precomputed_distances,_, _ = precomputed_pack
    

    return whole_pipeline_all(X,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch, random_seed=random_seed, method_type = consts.REGULAR_METHOD, distances_bands=None, precomputed_distances = precomputed_distances)
