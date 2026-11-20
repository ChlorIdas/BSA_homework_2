# BSA Homework 2: Motif discovery

## Repository structure

```
motif_finder.py   # finds a motif in a multi-FASTA file with k-mer counting and Gibbs sampling
test_data/
  DosR.fasta        # 10 upstream regions of hypoxia-induced M. tuberculosis genes (Bioinformatics Algorithms, ch. 2)
  test.fasta        # 20 simulated sequences with the motif TTGACGCA planted (0-2 mutations each)
test_results/       # script output from runs on the test data
environment.yml     # Packages used in the conda enviorment
```

## Example run

```bash
python motif_finder.py ../test_data/DosR.fasta 20 10   # file, motif length k, number of top k-mers n
```
