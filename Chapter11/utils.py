# Function for when you want to prepare DNA sequence feature for ML applications
import numpy as np

# Function for when you want to prepare DNA sequence feature for ML applications
def dnaseq_features(seq):
    start=0
    n_segs=101
    seq_name = 'seq'
    remaind = len(seq)%n_segs
    if(remaind != 0):
        last_id = len(seq) - remaind
    upd_seq = seq[start:last_id]
    dic_seq = {}
    for i in range(0,3):
        a = int(i*n_segs) ; b = int(i*n_segs)+n_segs 
        identifier = f"{seq_name}_{a}:{b}"
        dic_seq[identifier] = upd_seq[a:b]
    lst_seq = dic_seq.values()
    index = list(dic_seq.keys())
    values = list(dic_seq.values())

    # One hot encode    
    ii=-1
    for data in lst_seq:
        ii+=1
        abc = 'ACGT'
        char_to_int = dict((c, i) for i, c in enumerate(abc))
        int_enc = [char_to_int[char] for char in data]
        ohe = []
        for value in int_enc:
            base = [0 for _ in range(len(abc))]
            base[value] = 1
            ohe.append(base)
        np_mat = np.array(ohe)
        np_mat = np.expand_dims(np_mat,axis=0)

        if(ii != 0):
            matrix = np.concatenate([np_mat,matrix],axis=0)
        else:
            matrix = np_mat
        
    return matrix,index,values