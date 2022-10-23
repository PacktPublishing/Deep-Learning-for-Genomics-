from Bio import SeqIO

with open("covid19.fasta") as fh_in:
    for record in SeqIO.parse(fh_in, "fasta"):
            seq_record = record.seq
            seq_length = len(record.seq)
            print(f'% of Ts: {round(seq_record.count("T")/seq_length*100, 2)}')
            print(f'% of As: {round(seq_record.count("A")/seq_length*100, 2)}')
            print(f'% of Cs: {round(seq_record.count("C")/seq_length*100, 2)}')
            print(f'% of Gs: {round(seq_record.count("G")/seq_length*100, 2)}')