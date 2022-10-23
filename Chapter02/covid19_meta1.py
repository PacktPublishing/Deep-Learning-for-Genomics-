# Covid19_meta1.py
from Bio import SeqIO
from Bio.SeqUtils import GC

# with open("covid19.fasta") as fh_in:
# 	for record in SeqIO.parse(fh_in, "fasta"):
# 		print(f'GC content: {GC(record.seq)}')

with open("covid19.fasta") as fh_in:
	for record in SeqIO.parse(fh_in, "fasta"):
		print(f'GC content: {GC(record.seq)}')