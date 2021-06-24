"""
Provides metrics to evaluate the similarity and property of natural language texts.
"""
from dataclasses import dataclass
import dataclasses
import multiprocessing
from typing import List, Dict
from multiprocessing import Pool
import tempfile
from pathlib import Path
import subprocess
import glob
import json

import numpy as np
from virtualdatalab.logging import getLogger
log = getLogger(__name__)

# Support forward lookup of time hints
from __future__ import annotations

# If approximation is used for text analysis drop all word entries with less occurrences than
TOKEN_THRESHOLD = 1

# Word frequencies are calculated for 10 quantiles, index of the quantiles used
# to define lower and upper word frequencies defining "representative words"
MIN_REPR_WORD_QUANTILE_FREQ = 4
MAX_REPR_WORD_QUANTILE_FREQ = 6

@dataclass
class TextProperties():
    nWords : int
    nSentences : int
    sentence_length_distribution : List[int]
    word_frequency : Dict[str, int]    
    bigram : Dict[str, Dict[str, int]]
    word_frequency_distribution : List[int]

    def __init__(self, nWords : int, nSentences : int, sentence_length_distribution : List[int], word_frequency : Dict[str, int], bigram : Dict[str, Dict[str, int]]):
        self.nWords = nWords
        self.nSentences = nSentences
        self.sentence_length_distribution = sentence_length_distribution
        self.word_frequency = word_frequency
        self.bigram = bigram

        self.word_frequency_distribution = np.quantile(list(self.word_frequency.values()), np.linspace(0, 1.0, 11)).astype(int).tolist()

    def __str__(self):
        return f"{self.nSentences} Sentences , {self.nWords} Words, {len(self.word_frequency.keys())} Vocab size"
    
    def store(self, filename):
        with open(filename, 'w') as f:
            json.dump(dataclasses.asdict(self))
    
    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return TextProperties(**data)


 

def integerSplit(N, splits):
    stepsize = N//splits
    idxrange = [0]
    i = 0
    while (i+stepsize) <= N:
        i+=stepsize
        idxrange.append(i)
    idxrange[-1] = max(idxrange[-1], N)

    return idxrange


def strip_symbols(lines, symbols=[',', '.', '!', '?', ':', '"', '\'', ';', '(', ')', '#', '/', '`', '“', '„'], replacement=''):
    for s in symbols:
        lines = [l.replace(s, replacement) for l in lines]
    # Strip empty lines
    lines = [l for l in lines if l != '']
    return lines


def map_text_properties(text_file : str, line_sep : str, word_sep : str, approximate : bool) -> TextProperties:
    #with open(text_file, 'r', newline=line_sep) as f:
    with open(text_file, 'r') as f:
        lines = f.read().splitlines()
    nSentences = len(lines)

    # Remove all unwanted symbols from the input, leaving lines with words only
    clean_lines = strip_symbols(lines)
    sentence_length_distribution = [len(l) for l in clean_lines]

    # Extract one long list of words/tokens
    words = []
    _ = [words.extend(l.split(word_sep)) for l in clean_lines]
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
    if approximate:
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

    return TextProperties(
        nWords=nWords,
        nSentences=nSentences,
        sentence_length_distribution=sentence_length_distribution,
        bigram=word_cooc,
        word_frequency=word_freq
    )

def reduce_text_properties(properties : List[TextProperties]) -> TextProperties :
    nWords = 0
    nSentences = 0
    sentence_length_distribution = []
    word_frequency = {}
    bigram = {}

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

    sentence_length_distribution = np.quantile(sentence_length_distribution, np.linspace(0, 1.0, 11)).astype(int).tolist()

    return TextProperties(
        nWords = nWords,
        nSentences=nSentences,
        sentence_length_distribution=sentence_length_distribution,
        word_frequency=word_frequency,
        bigram=bigram
    )



class Text:
    name : str = ''
    properties : TextProperties
    _text_workspace = []

    def __init__(self, name : str, word_sep = ' ', sentence_sep = '\n'):
        self.name = name
        self.word_sep = word_sep
        self.sentence_sep = sentence_sep
    
    def __str__(self):
        return f"Text [{self.name}] : {self.properties}"


    @classmethod
    def fromFile(cls, name : str, path : str):
        text = cls(name)
        text._load_file(path)
        return text
    

    def _load_file(self, path : str) :
        # Per default each partition is processed by its own thread ...
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

    def _analyze(self, files, nProcess, approximate=True) -> TextProperties:
        args = [(file, self.sentence_sep, self.word_sep, approximate) for file in files]

        nProcess = multiprocessing.cpu_count()
        with Pool(nProcess) as p:
            text_properties = p.starmap(map_text_properties, args) 

        properties = reduce_text_properties(text_properties)

        return properties

    #@staticmethod
    #def compare(T1 : Text, T2 : Text) -> Dict[str: float] :
    #    pass