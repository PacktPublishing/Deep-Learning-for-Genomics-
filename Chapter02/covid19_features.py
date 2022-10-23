# covid19_features.py

from Bio import SeqIO
from Bio.SeqUtils import GC
import pandas as pd

nucl = ['A', 'T', 'C', 'G']
final_dict = {}

# with open('covid19.fasta') as fh_in:
#     with open("test.csv", 'w') as fh_out:
#         for record in SeqIO.parse(fh_in, "fasta"):
#             for n1 in nucl:
#                 for n2 in nucl:
#                     di = str(n1) + str(n2)
#                     final_dict[di] = record.seq.count(di)
#             A_count = record.seq.count('A')
#             final_dict['A_count'] = round(A_count/len(record) * 100, 2)
#             C_count = record.seq.count('C')
#             final_dict['C_count'] = round(C_count/len(record) * 100, 2)
#             G_count = record.seq.count('G')
#             final_dict['G_count'] = round(G_count/len(record) * 100, 2)
#             T_count = record.seq.count('T')
#             final_dict['T_count'] = round(T_count/len(record) * 100, 2)
#             final_dict['GC_content'] = round(GC(record.seq), 2)
#             final_dict['Size'] = len(record)


with open('covid19.fasta') as fh_in:
    with open("test.csv", 'w') as fh_out:
        for record in SeqIO.parse(fh_in, "fasta"):
            for n1 in nucl:
                for n2 in nucl:
                    di = str(n1) + str(n2)
                    final_dict[di] = record.seq.count(di)
            A_count = record.seq.count('A')
            final_dict['A_count'] = round(A_count/len(record) * 100, 2)
            C_count = record.seq.count('C')
            final_dict['C_count'] = round(C_count/len(record) * 100, 2)
            G_count = record.seq.count('G')
            final_dict['G_count'] = round(G_count/len(record) * 100, 2)
            T_count = record.seq.count('T')
            final_dict['T_count'] = round(T_count/len(record) * 100, 2)
            final_dict['GC_content'] = round(GC(record.seq), 2)
            final_dict['Size'] = len(record)

final_df = pd.DataFrame.from_dict([final_dict])
final_df['virus'] = "Covid19"
final_df.to_csv("covid19_features.csv", index=None)