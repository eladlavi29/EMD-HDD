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


def whole_pipeline_all(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='most_common', random_seed=None, method_type = consts.REGULAR_METHOD, distances_bands=None, precomputed_distances = None):
        print("XXXXXXX IN METHOD XXXXXXXXX")
        st = time.time()

        my_HDD_HDE = HDD_HDE(X,y, rows_factor, cols_factor, is_normalize_each_band, method_label_patch, method_type, distances_bands, precomputed_distances=precomputed_distances)
        d_HDD, labels_padded, num_patches_in_row,y_patches = my_HDD_HDE.calc_hdd()
        
        print("WHOLE METHOD TIME: ", time.time()-st, flush=True)
        st = time.time()

        print("XXXXXXX IN CLASSIFICATION XXXXXXXXX")

        y_patches = y_patches.int()
        
        if torch.cuda.is_available():
            d_HDD = d_HDD.cpu()
            y_patches = y_patches.cpu()
            labels_padded = labels_padded.cpu()

        clf = PaviaClassifier(d_HDD.numpy(), y_patches.numpy(), consts.N_NEIGHBORS, labels_padded.numpy(), rows_factor, cols_factor, num_patches_in_row, is_divided=False, random_seed = random_seed)

        return clf.classify()

        # print("WHOLE CLASSIFICATION TIME: ", time.time()-st)
        

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
