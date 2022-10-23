# covid19_meta.py
from Bio import SeqIO
import matplotlib.pyplot as plt

nucl = ['A', 'T', 'C', 'G']
di_nucl_dict = {}

# with open("covid19.fasta") as fh_in:
#     for record in SeqIO.parse(fh_in, "fasta"):
#         for n1 in nucl:
#             for n2 in nucl:
#                 di = str(n1) + str(n2)
#                 di_nucl_dict[di] = record.seq.count(di)

with open("covid19.fasta") as fh_in:
    for record in SeqIO.parse(fh_in, "fasta"):
        for n1 in nucl:
            for n2 in nucl:
                di = str(n1) + str(n2)
                di_nucl_dict[di] = record.seq.count(di)

di = [k for k, v in di_nucl_dict.items()]
counts = [v for k, v in di_nucl_dict.items()]

print(di_nucl_dict)
plt.bar(di,counts)
plt.ylabel("Counts")
plt.show()