"""
Provides metrics to evaluate the similarity and property of natural language texts.
"""
# Support forward lookup of time hints
from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import multiprocessing
from os import stat
from typing import List, Dict, Text, Tuple
from multiprocessing import Pool
import tempfile
from pathlib import Path
import subprocess
import glob
import json

import numpy as np
from numpy.random import default_rng
rng = default_rng()

from virtualdatalab.logging import getLogger
log = getLogger(__name__, stdout=True)

# If approximation is used for text analysis drop all word entries with less occurrences than
TOKEN_THRESHOLD = 1

# We select a subset of the vocab based on the word frequency as "representative words"

# Define how many words are used
REPR_WORDS = 0.1

# Word frequencies are calculated for 10 quantiles, index of the quantiles used
# to define what quartile is the upper bound for "representative words"
LOW_REPR_WORD_QUANTILE_FREQ = 7
HIGH_REPR_WORD_QUANTILE_FREQ = 9

@dataclass
class Properties:
    nWords : int
    nSentences : int
    sentence_length_distribution : List[int]
    word_frequency : Dict[str, int]    
    bigram : Dict[str, Dict[str, int]]
    word_frequency_distribution : List[int]
    sample : List[str]
    clean_sample : List[str]

    def __init__(self, nWords : int, nSentences : int, sentence_length_distribution : List[int], word_frequency : Dict[str, int], bigram : Dict[str, Dict[str, int]], sample : List[str], clean_sample : List[str]):
        self.nWords = nWords
        self.nSentences = nSentences
        self.sentence_length_distribution = sentence_length_distribution
        self.word_frequency = word_frequency
        self.bigram = bigram
        self.sample = sample
        self.clean_sample = clean_sample

        self.word_frequency_distribution = np.quantile(list(self.word_frequency.values()), np.linspace(0, 1.0, 11)).astype(int).tolist()

    def __str__(self):
        return f"{self.nSentences} Sentences , {self.nWords} Words, {len(self.word_frequency.keys())} Vocab size, avg. sentence length {self.sentence_length_distribution[5]}"
    
    def store(self, filename):
        with open(filename, 'w') as f:
            json.dump(dataclasses.asdict(self))
    
    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return Properties(**data)


@dataclass
class Perturbation:
    word_deletion : float
    word_repetition : float
    word_switch : float
    sentence_merging : float
    MIN_REPETITIONS = 2
    MAX_REPETITIONS = 4

    def apply(self, sentences : List[str]) -> List[str] :
        N = len(sentences)
        deletion = rng.choice([0, 1], size=N, p=[1.0 - self.word_deletion, self.word_deletion])
        repetition = rng.choice([0, 1], size=N, p=[1.0 - self.word_repetition, self.word_repetition])
        switch = rng.choice([0, 1], size=N, p=[1.0 - self.word_switch, self.word_switch])
        merging = rng.choice([0, 1], size=N, p=[1.0 - self.sentence_merging, self.sentence_merging])
        # Pre calculate some intergers that are used later
        R = rng.integers(9999, size=2*N)
        c = 0
        log.debug(f"Text Perturbation - sentences: {len(sentences)}, deletion: {deletion.sum()}, repetition: {repetition.sum()}, switch: {switch.sum()}, merging: {merging.sum()}")

        did_merge = 0
        did_repeat = 0
        did_switch = 0
        did_delete  = 0 

        perturbed = []
        for s, ns, delete, repeat, switch, merge in zip(sentences[0:-1], sentences[1:], deletion, repetition, switch, merging) :
            # s ... sentence
            # ns ... next sentence
            # ps ... perturbed sentence
            if len(s) == 0:
                continue

            ps = s.copy()
            if merge and len(ns) > 1:
                end = R[c] % len(ns) ; c+=1
                ps.extend(ns[0:end])
                did_merge+=1
            if repeat:
                pos = R[c] % len(ps) ; c+=1
                cnt = (R[c] % (self.MAX_REPETITIONS-self.MIN_REPETITIONS)) + self.MIN_REPETITIONS ; c+=1
                _ = [ps.insert(pos, ps[pos]) for i in range(cnt)]
                did_repeat += 1
            if switch and len(ps)>1:
                pos = R[c] % (len(ps)-1) ; c+=1                
                tmp = ps.pop(pos+1)
                ps.insert(pos, tmp)
                did_switch += 1
            if delete:
                pos = R[c] % len(ps) ; c+=1                
                ps.pop(pos)
                did_delete += 1
            
            perturbed.append(ps)
        # Just add the last sentence unperturbed in order to have the same number of sentences
        perturbed.append(ns)

        log.debug(f"After: merge {did_merge}, repeat {did_repeat}, switch {did_switch}, delete {did_delete}")
        return perturbed 
    
    def toDict(self):
        return dataclasses.asdict(self)
    
    @classmethod
    def fromDict(cls, params : Dict):
        return cls(**params)

def integerSplit(N : int, splits : int) -> List[int]:
    """
    Helper function to split an array of size N into `splits` parts without overlap
    """
    stepsize = N//splits
    idxrange = [0]
    i = 0
    while (i+stepsize) <= N:
        i+=stepsize
        idxrange.append(i)
    idxrange[-1] = max(idxrange[-1], N)

    return idxrange


def strip_symbols(lines : List[str], symbols=[',', '.', '!', '?', ':', '"', '\'', ';', '(', ')', '#', '/', '`', '“', '„', '”'], replacement='') -> List[str]:
    """
    Replace all symbols in a given list of strings
    """
    for s in symbols:
        lines = [l.replace(s, replacement) for l in lines]
    # Strip empty lines
    lines = [l for l in lines if l != '']
    return lines



def map_text_properties(text_file : str, line_sep : str, word_sep : str, lowpass : bool, perturbation : Perturbation) -> Properties:
    """
    Map step that reads a text file from disk, and extracts text properties.

    This will run in its own sub process!
    """
    try:
        with open(text_file, 'r') as f:
            lines = f.read().splitlines()
        nSentences = len(lines)

        # Remove all unwanted symbols from the input, leaving lines with words only
        clean_lines = strip_symbols(lines)

        split_sentences = [l.split(word_sep) for l in clean_lines]
        if perturbation:
            p = Perturbation.fromDict(perturbation)
            split_sentences = p.apply(split_sentences)

        sentence_length_distribution = [len(l) for l in split_sentences]
        # Extract one long list of words/tokens
        words = []
        _ = [words.extend(s) for s in split_sentences]
        nWords = len(words)

        # Calculate word coocurrences (bidirectional bigram)
        word_freq = {}
        word_cooc = {}
        for w1, w2 in zip(words[:-1], words[1:]):
            # Word freq
            word_freq[w1] = word_freq.get(w1, 0) + 1

            # Forward
            w1_neighbours = word_cooc.get(w1, {})
            w1_neighbours[w2] = w1_neighbours.get(w2, 0) + 1
            word_cooc[w1] = w1_neighbours

            # Backward
            w2_neighbours = word_cooc.get(w2, {})
            w2_neighbours[w1] = w2_neighbours.get(w1, 0) + 1
            word_cooc[w2] = w2_neighbours
        
        # If approxmation is set, remove words that are present less or equal to TOKEN_THRESHOLD
        if lowpass:
            drop_words = [k for k, v in word_freq.items() if v <= TOKEN_THRESHOLD]
            _ = [word_freq.pop(word) for word in drop_words]

            drop_center_words = []
            for center_word, neighborhood in word_cooc.items():
                drop_words = [k for k, v in neighborhood.items() if v <= TOKEN_THRESHOLD]
                # In case we would drop the complete neighborhood drop the whole key
                if len(drop_words) == len(neighborhood):
                    drop_words.append(center_word)
                else:
                    _ = [neighborhood.pop(word) for word in drop_words]
                    word_cooc[center_word] = neighborhood
            
            _ = [word_cooc.pop(word) for word in drop_center_words] 

    except Exception as e:
        log.error(f'Analysing text partitioned failed in map! {e}')


    return Properties(
        nWords=nWords,
        nSentences=nSentences,
        sentence_length_distribution=sentence_length_distribution,
        bigram=word_cooc,
        word_frequency=word_freq,
        sample = [word_sep.join(split_sentences[1]),],
        clean_sample=[clean_lines[1],]
    )

def _quantiles(frequencies : List[int]) -> List[int] :
    return np.quantile(frequencies, np.linspace(0, 1.0, 11)).astype(int).tolist()

def reduce_text_properties(properties : List[Properties]) -> Properties :
    """
    Reduce step that aggregates a list of text properties into a single set
    """
    nWords = 0
    nSentences = 0
    sentence_length_distribution = []
    word_frequency = {}
    bigram = {}
    samples = []
    clean_samples = []

    for p in properties:
        nWords += p.nWords
        nSentences += p.nSentences
        sentence_length_distribution.extend(p.sentence_length_distribution)

        # word frequency
        for word, cnt in p.word_frequency.items():
            word_frequency[word] = word_frequency.get(word, 0) + cnt

        # word coocurrence
        for word, neighbours in p.bigram.items():
            wn = bigram.get(word, {})
            for n, cnt in neighbours.items():
                wn[n] = wn.get(n, 0) + cnt
            bigram[word] = wn
        
        samples.extend(p.sample)
        clean_samples.extend(p.clean_sample)

    sentence_length_distribution = _quantiles(sentence_length_distribution) 

    return Properties(
        nWords = nWords,
        nSentences=nSentences,
        sentence_length_distribution=sentence_length_distribution,
        word_frequency=word_frequency,
        bigram=bigram,
        sample=samples,
        clean_sample = clean_samples
    )


class Corpus:
    """
    Load and hold information about a text corpus
    """
    name : str = ''
    properties : Properties
    perturbation : Perturbation
    _text_workspace = []

    def __init__(self, name : str, word_sep = ' ', sentence_sep = '\n'):
        self.name = name
        self.word_sep = word_sep
        self.sentence_sep = sentence_sep
        self.properties = None
    
    def __str__(self):
        return f"Text corpus [{self.name}] : {self.properties}"

    @classmethod
    def fromFile(cls, name : str, path : str, perturbation : Perturbation = None):
        text = cls(name)
        text.perturbation = perturbation
        text._load_file(path)
        return text
    

    def _load_file(self, path : str) :
        # Per default each partition is processed by its own sub-process ...
        nPartitions = multiprocessing.cpu_count()
        src_file = Path(path).absolute()
        self.properties = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            log.debug(f"Processing {src_file}, splitting it into {nPartitions} fragments in {tmpdirname}")
            cmd = f"split {src_file} -n {nPartitions} PREFIX"

            p = subprocess.run(cmd, shell=True, capture_output=True, cwd=tmpdirname)
            if p.returncode == 0:
                files = glob.glob(f'{tmpdirname}/PREFIX*')
                log.debug(f"Analyzing text making use of {nPartitions} subprocess")
                try:
                    self.properties = self._analyze(files, nPartitions)
                except Exception as e:
                    log.error(f"Analyzing text properties, failed! {e}")
            else:
                log.error("splitting text file for multi-thread processing failed!")

    def _analyze(self, files, nProcess, lowpass=True) -> Properties:
        if self.perturbation:
            args = [(file, self.sentence_sep, self.word_sep, lowpass, self.perturbation.toDict()) for file in files]
        else:
            args = [(file, self.sentence_sep, self.word_sep, lowpass, None) for file in files]

        nProcess = multiprocessing.cpu_count()
        with Pool(nProcess) as p:
            text_properties = p.starmap(map_text_properties, args) 

        properties = reduce_text_properties(text_properties)

        return properties

class Analysis:
    @staticmethod
    def _representative_quantiles(quantiles : List[int]) -> Tuple(int, int):
        # Select all words that frequency is between Q0.7 to Q0.9
        return quantiles[LOW_REPR_WORD_QUANTILE_FREQ], quantiles[HIGH_REPR_WORD_QUANTILE_FREQ]


    @staticmethod
    def _representative_frequencies(frequencies : List[int]) -> Tuple(int, int) :
        return Analysis._representative_quantiles(_quantiles(frequencies))


    @staticmethod
    def keywords(word_frequencies : Dict[str, int]) -> List[str]:
        freqs = np.array(list(word_frequencies.values()))
        words = np.array(list(word_frequencies.keys()))

        lt, ht = Analysis._representative_frequencies(freqs)
        selected = (freqs <= ht) & (freqs >= lt)
        repr_words_idx = np.argsort(freqs[selected])
        repr_words = words[selected][repr_words_idx]

        return repr_words.tolist()


    @staticmethod
    def corpus_keywords(T : Corpus) -> List[str]:
        freqs = np.array(list(T.properties.word_frequency.values()))
        words = np.array(list(T.properties.word_frequency.keys()))

        lt, ht = Analysis._representative_quantiles(T.properties.word_frequency_distribution)
        selected = (freqs <= ht) & (freqs >= lt)
        repr_words_idx = np.argsort(freqs[selected])
        repr_words = words[selected][repr_words_idx]

        return repr_words.tolist()


    @staticmethod    
    def _vocab_overlap(V1 : List[str], V2 : List[str]):
        """
        Return the overlapping vocab and the ratio of intersection/union
        """
        V1 = set(V1)
        V2 = set(V2)
        
        intersection = V1 & V2
        union = V1 | V2
        overlap = len(intersection) / len(union)

        return list(intersection), overlap


    def shared_vocab(T1 : Corpus, T2 : Corpus) -> List[str] :
        vocab, score = Analysis._vocab_overlap(T1.properties.word_frequency.keys(), T2.properties.word_frequency.keys())
        return vocab


    @staticmethod
    def corpus_sentence_length_similarity(T1 : Corpus, T2 : Corpus) -> float :
        """
        Calculate a similarity score [0.0, 1.0] (1.0 == identical) that is based
        on the sentence length distribution of two corpora.
        """
        F1 = np.array(T1.properties.sentence_length_distribution)
        F2 = np.array(T2.properties.sentence_length_distribution)
        fmax = np.max([F1, F2], axis=0)

        # Calculate the max normalized L1 distance in the range of Q0.1 to Q0.9
        MAGIC_NUMBER=2.0
        score = 1.0 - MAGIC_NUMBER*(np.divide(np.abs(F1-F2), fmax)[1:-1].mean())

        return score


    @staticmethod
    def word_frequency_similarity(W1 : Dict[str, int], W2 : Dict[str, int]) -> float:
        shared_vocab, shared_vocab_score = Analysis._vocab_overlap(W1, W2)
        if shared_vocab == []:
            return 0.0
        
        W1_freqs = np.array([W1[word] for word in shared_vocab])
        W2_freqs = np.array([W2[word] for word in shared_vocab])

        W1norm = sum(list(W1.values()))
        W2norm = sum(list(W2.values()))

        # Normalize freq by the total word usage (adjusting for different text lengths)
        W1_freqs = np.divide(W1_freqs, W1norm)
        W2_freqs = np.divide(W2_freqs, W2norm)

        # sum of the normalized L1 distance scaled by the shared vocab size
        #score = max(0.0, 1.0 - (np.abs(W1_freqs - W2_freqs).sum() / (2*len(shared_vocab)/(len(W1)+len(W2)))))
        #score = max(0.0, 1.0 - (np.abs(W1_freqs - W2_freqs).sum() / shared_vocab_score))
        score = max(0.0, 1.0 - (np.abs(W1_freqs - W2_freqs).sum()))
        return score
 

    @staticmethod
    def corpus_word_frequency_similarity(T1 : Corpus, T2 : Corpus) -> float :
        """
        Calculate a similarity score [0.0, 1.0] (1.0 == identical) that measues
        the similarity of two corpora based on the word frequencies distribution
        for a the shared vocab.
        """
        return Analysis.word_frequency_similarity(T1.properties.word_frequency, T2.properties.word_frequency)


    @staticmethod
    def corpus_bigram_similarity(T1 : Corpus, T2 : Corpus) -> float :
        """
        Calculate the average vocab_overlap score for each word in the shared vocab
        """
        shared_vocab = Analysis.shared_vocab(T1, T2)

        wfs = []
        for word in shared_vocab:
            _, score = Analysis._vocab_overlap(T1.properties.bigram[word], T2.properties.bigram[word])
            wfs.append(score)

        # calculate the word frequency similarity for each word in the shared_vocab
        #wfs = [Analysis._word_frequency_similarity(T1.properties.bigram[word], T2.properties.bigram[word]) for word in shared_vocab]
        #score = np.mean(wfs)

        return np.mean(wfs)


    @staticmethod
    def corpus_similarity(T1 : Corpus, T2 : Corpus) -> Dict[str, float]:
        sim = {}
        sim["sentence_length"] = Analysis.corpus_sentence_length_similarity(T1, T2)
        sim["word_frequency"] = Analysis.corpus_word_frequency_similarity(T1, T2)
        sim["bigram"] = Analysis.corpus_bigram_similarity(T1, T2)
        return sim
    
