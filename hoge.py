import numpy as np


peds_in_curr_seq = [i for i in range(3)]
print(peds_in_curr_seq)
seq_len = 20
curr_seq_rel = np.zeros((len(peds_in_curr_seq), 4, seq_len))

print(curr_seq_rel)