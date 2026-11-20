"""
motif_finder.py - find a motif in a group of DNA sequences.

Method 1: k-mer counting  - count every k-mer and rank it by a Poisson p-value
Method 2: Gibbs sampling  - refine one motif site per sequence using a profile

Usage:   python motif_finder.py <sequences.fasta> <k> [n]
         k = motif length, n = number of top k-mers to print (default 10)
Example: python motif_finder.py DosR.fasta 20 10

Assumes the sequences contain only A, C, G and T.
"""
#Python is v.3.15.2

import sys
import math 
import numpy as np # v.2.5.4
import pandas as pd  # v.3.1.1
from Bio import SeqIO # v.1.88

BASES = ["A", "C", "G", "T"]
BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}

N_RESTARTS = 50      # Gibbs: number of runs from different random starts
N_ITERATIONS = 1000   # Gibbs: iterations per run
SEED = 67             # haha 67 


def read_fasta(path):
    names = []
    seqs = []
    for record in SeqIO.parse(path, "fasta"):
        names.append(record.description)
        seqs.append(str(record.seq).upper())
    return names, seqs


def background_freqs(seqs):
    # q(b): how often each base occurs in all sequences together.
    all_bases = "".join(seqs)
    return np.array([all_bases.count(b) for b in BASES]) / len(all_bases)


# ---------------------------------------------------------------------------
# Method 1: k-mer counting
# ---------------------------------------------------------------------------
def poisson_pvalue(z, lam):
    # P(X >= z) for X ~ Poisson(lam): sum the terms e^-lam * lam^i / i! for i >= z.
    # Summing the upper tail directly (instead of 1 - lower tail) keeps very
    # small p-values from rounding to 0.
    pvalue = 0.0
    for i in range(z, z + 50):
        pvalue = pvalue + math.exp(-lam) * lam**i / math.factorial(i)
    return pvalue


def kmer_counting(seqs, k, q):
    # Count all overlapping k-mers and compare each count with the expected one.
    # expected count = p(word) * number of k-mer positions,
    # where p(word) = q(x1) * q(x2) * ... * q(xk)
    # Returns a DataFrame with one row per k-mer seen at least twice.
    counts = {}
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            word = seq[i:i + k]
            counts[word] = counts.get(word, 0) + 1
    n_positions = sum(len(seq) - k + 1 for seq in seqs)

    rows = []
    for word, observed in counts.items():
        if observed < 2:
            continue
        p_word = 1.0
        for base in word:
            p_word = p_word * q[BASE_INDEX[base]]
        expected = p_word * n_positions
        rows.append({"kmer": word, "observed": observed, "expected": expected,
                     "p_value": poisson_pvalue(observed, expected)})
    return pd.DataFrame(rows, columns=["kmer", "observed", "expected", "p_value"])


# ---------------------------------------------------------------------------
# Method 2: Gibbs sampling
# ---------------------------------------------------------------------------
def make_profile(sites, k):
    # Profile matrix (4 x k): probability of each base at each motif position.
    # Starts every count at 1 (pseudocount), so no probability is ever 0.
    counts = np.ones((4, k))
    for site in sites:
        for i in range(k):
            counts[BASE_INDEX[site[i]], i] += 1
    return counts / counts.sum(axis=0)


def consensus(sites, k):
    # Most common base at each motif position.
    profile = make_profile(sites, k)
    return "".join(BASES[row] for row in profile.argmax(axis=0))


def motif_score(sites, k, q):
    # Log-odds score: sum over all sites and positions of log(p_i(b) / q(b)).
    # Higher = the sites look more like one shared motif and less like background.
    profile = make_profile(sites, k)
    score = 0.0
    for site in sites:
        for i in range(k):
            b = BASE_INDEX[site[i]]
            score = score + math.log(profile[b, i] / q[b])
    return score


def gibbs_run(seqs, k, q, rng):
    # One Gibbs sampling run. Returns the best start positions and their score.
    # Random start position in every sequence
    starts = [int(rng.integers(0, len(seq) - k + 1)) for seq in seqs]
    sites = [seq[s:s + k] for seq, s in zip(seqs, starts)]
    best_starts = starts.copy()
    best_score = motif_score(sites, k, q)

    for _ in range(N_ITERATIONS):
        # Peave one random sequence r out, build the profile from the rest
        r = int(rng.integers(0, len(seqs)))
        profile = make_profile(sites[:r] + sites[r + 1:], k)

        # Weight of every position a in sequence r:
        #W = product over motif positions i of p_i(base) / q(base)
        seq = seqs[r]
        weights = np.ones(len(seq) - k + 1)
        for a in range(len(seq) - k + 1):
            for i in range(k):
                b = BASE_INDEX[seq[a + i]]
                weights[a] = weights[a] * profile[b, i] / q[b]

        # Sample a new position, with probability proportional to its weight
        new_start = int(rng.choice(len(weights), p=weights / weights.sum()))
        starts[r] = new_start
        sites[r] = seq[new_start:new_start + k]

        # Remember the best set of sites seen so far
        score = motif_score(sites, k, q)
        if score > best_score:
            best_score = score
            best_starts = starts.copy()

    return best_starts, best_score


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python motif_finder.py <sequences.fasta> <k> [n]")
        sys.exit(1)
    path = sys.argv[1]
    k = int(sys.argv[2])
    n_top = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    names, seqs = read_fasta(path)

 # need a multi-FASTA: a motif is only "shared" if there are 2+ sequences
 # NOTE im not sure if the script SHOULD make a multifasta out of fastas, but i dont want to do it <3
    if len(seqs) < 2:
        print(f"Error: {path} has {len(seqs)} sequence(s); a multi-FASTA with at least 2 is needed.")
        sys.exit(1)

    q = background_freqs(seqs)
    print(f"Read {len(seqs)} sequences from {path}, k = {k}")
    print("Background q(b): " + ", ".join(f"{b}={x:.3f}" for b, x in zip(BASES, q)))

    # Method 1
    print(f"\n=== 1. K-mer counting: top {n_top} over-represented {k}-mers ===")
    kmers = kmer_counting(seqs, k, q)
    if len(kmers) == 0:
        print(f"No {k}-mer occurs more than once (exact repeats are too rare at this k).")
    else:
        top = kmers.sort_values(["p_value", "observed"], ascending=[True, False]).head(n_top)
        print(top.to_string(index=False))

    # Method 2: several runs, keep the best (Gibbs can get stuck in a local optimum)
    print(f"\n=== 2. Gibbs sampling ({N_RESTARTS} runs x {N_ITERATIONS} iterations) ===")
    rng = np.random.default_rng(seed=SEED)
    best_starts, best_score = None, None
    for run in range(N_RESTARTS):
        starts, score = gibbs_run(seqs, k, q, rng)
        if best_score is None or score > best_score:
            best_starts, best_score = starts, score

    sites = [seq[s:s + k] for seq, s in zip(seqs, best_starts)]
    print(f"Consensus motif: {consensus(sites, k)}")
    print(f"Log-odds score (higher is better): {best_score:.2f}")

    print("\nProfile matrix (rows = bases, columns = motif positions):")
    profile = pd.DataFrame(make_profile(sites, k), index=BASES, columns=range(1, k + 1))
    print(profile.round(2).to_string())

    print("\nMotif site found in each sequence:")
    for name, start, site in zip(names, best_starts, sites):
        print(f"  {site}  start={start:<4} {name}")


# run the program
main()