#!/bin/sh

./grab_sequence.sh splice_table_Human.txt GRCh38.primary_assembly.genome.fa canonical_sequence_Human.txt
#./grab_sequence.sh splice_table_Human.test.txt ./GRCh38.primary_assembly.genome.fa canonical_sequence_Human.test.txt

# TRAINING
python create_datafile.py train all 1 Human
python create_datafile.py train all 2 Human
python create_datafile.py train all 3 Human
python create_datafile.py train all 4 Human
python create_datafile.py train all 5 Human
python create_datafile.py train all 6 Human

python create_dataset_multi.py train all Human 0 1
python create_dataset_multi.py train all Human 1 2
python create_dataset_multi.py train all Human 1 3
python create_dataset_multi.py train all Human 1 4
python create_dataset_multi.py train all Human 1 5
python create_dataset_multi.py train all Human 1 6

rm datafile*

# TESTING
python create_datafile.py test all 1 Human
python create_datafile.py test all 2 Human
python create_datafile.py test all 3 Human
python create_datafile.py test all 4 Human

python create_dataset_multi.py test 1 Human 0 1
python create_dataset_multi.py test 1 Human 1 2
python create_dataset_multi.py test 1 Human 1 3
python create_dataset_multi.py test 1 Human 1 4

rm datafile*

# Remove paralogs from datasets
python remove_paralogs_from_h5.py
