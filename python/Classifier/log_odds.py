import argparse
from collections import Counter, defaultdict
import sys
from ast import literal_eval
import pandas as pd
import math
import numpy as np
import csv
import re
import unicodedata


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub("\n", " ", s)
    s = re.sub(r"([.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def write_log_odds(counts1, counts2, prior, outfile_name = None):
    # COPIED FROM LOG_ODDS FILE
    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in prior.keys():
        prior[word] = int(prior[word] + 0.5)

    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    n1  = sum(counts1.values())
    n2  = sum(counts2.values())
    nprior = sum(prior.values())


    for word in prior.keys():
        if prior[word] > 0:
            l1 = float(counts1[word] + prior[word]) / (( n1 + nprior ) - (counts1[word] + prior[word]))
            l2 = float(counts2[word] + prior[word]) / (( n2 + nprior ) - (counts2[word] + prior[word]))
            sigmasquared[word] =  1/(float(counts1[word]) + float(prior[word])) + 1/(float(counts2[word]) + float(prior[word]))
            sigma[word] =  math.sqrt(sigmasquared[word])
            delta[word] = ( math.log(l1) - math.log(l2) ) / sigma[word]

    if outfile_name:
        outfile = open(outfile_name, 'w')
        for word in sorted(delta, key=delta.get):
            outfile.write(word)
            outfile.write(" %.3f\n" % delta[word])
            
        outfile.close()
    else:
        return delta

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def compute_word_scores(df, odds_column):
    header_to_count = Counter()

    header_to_counter = defaultdict(Counter)
    for i, row in df.iterrows():
        words = row['body']
        header = row[odds_column]
        header_to_counter[header].update(words)
        header_to_count[header] += 1

    print("Done loading raw counts")

    word_to_scores = defaultdict(list)
    header_to_py = []
    for h1,c1 in header_to_counter.items():
        alt_counter = Counter()
        prior = Counter()
        for h2,c2 in header_to_counter.items():
            prior.update(c2)
            if h2 != h1:
                alt_counter.update(c2)
        word_to_score = write_log_odds(c1, alt_counter, prior)
        for w,s in word_to_score.items():
            word_to_scores[w].append(sigmoid(s))
        header_to_py.append(header_to_count[h1] / len(df))

    # NOTE: we can get to here without memory issues
    print("Done computing word scores")
    word_to_scores = {w:np.array(s) for w,s in word_to_scores.items()}
    return word_to_scores, header_to_py



def score_and_write_rows(df, word_to_scores, header_to_py, filename):
    out_fp = open(filename, "w")
    csvwriter = csv.writer(out_fp, delimiter='\t')

    for i,row in df.iterrows():
        words = row["body"].split()
        # When we down-filter, we might have words that we haven't seen
        # just skip them for now
        scores = [word_to_scores[w] for w in words if w in word_to_scores]
        pw = np.prod(scores, axis=0) # multiply over words
        final_scores = np.multiply(pw, header_to_py) # multiple by prior (elementwise)

        log_odds = " ".join([str(s) for s in final_scores])
        csvwriter.writerow([row.index, row["body"], row["sex"], log_odds])
        # write_row(row, out_fp, log_odds)
    print("Done writing")
    out_fp.close()

df = pd.read_csv("train_set.csv")
odds_column = "NEL"
df.body = df.body.map(normalizeString)
word_to_scores, header_to_py = compute_word_scores(df,odds_column)
score_and_write_rows(df, word_to_scores, header_to_py, "subset_odds.tsv")
