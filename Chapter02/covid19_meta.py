# covid19_meta.py
from Bio import SeqIO

with open("covid19.fasta") as fh_in:
    for record in SeqIO.parse(fh_in, "fasta"):
            print(f'sequence information: {record}')
            print(f'sequence length: {len(record)}')