"""Utilities for tokenizing, vocabularies."""

import gzip
import os
import re
import sys
import tarfile
import urllib
import subprocess
from collections import defaultdict

from bs4 import BeautifulSoup
from tensorflow.python.platform import gfile
import nltk

basedir = os.path.abspath(os.path.dirname(__file__))

sent_tokenizers = {
  'en':nltk.data.load('tokenizers/punkt/english.pickle'),
  'de':nltk.data.load('tokenizers/punkt/german.pickle')
  }

path_to_moses_tokenizer = os.path.join(
  basedir, 'mosesdecoder', 'scripts', 'tokenizer', 'tokenizer.perl'
  )

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_START_VOCAB_IDS = [PAD_ID, GO_ID, EOS_ID, UNK_ID]

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

# URLs for IWSLT data.

def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print "Unpacking %s to %s" % (gz_path, new_path)
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "w") as new_file:
      for line in gz_file:
        new_file.write(line)


def paragraph_tokenizer(text, language):
  """
  The given text may have more than one sentence in it. It may also be in any
  language. Split it into its constituent sentences and return an array of them.
  """
  return sent_tokenizers[language].tokenize(text)


def moses_tokenizer(path, language):
  """Run the Moses tokenizer over all the lines in the file at this path."""
  process = path_to_moses_tokenizer + ' -l ' + language + ' < ' + path
  for line in subprocess.check_output(process, shell=True).split('\n'):
    yield line


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print "Creating vocabulary %s from data %s" % (vocabulary_path, data_path)

    vocab = {}
    counter = 0
    with open(data_path, 'rb') as fin:
      for line in fin.readlines():
        counter += 1
        tokens = line.strip().split()
        for w in tokens:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1

    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
      vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
      for w in vocab_list:
        vocab_file.write(w + "\n")
  print "Completed vocab!"


def initialize_vocabulary(vocabulary_path, start_index=None):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.
    start_index: number starting from start_index. None is effectively 0.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  start_index = start_index or 0
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y + start_index) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": "7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  words = sentence.split()
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]

def token_ids_to_sentence(tokens, rev_vocabulary):
  """Convert a list of integers to a sentence.

  For example, [1, 2, 8, 16] may return "I have a fire".

  Args:
    tokens: a list of token-ids to become a sentence.
    rev_vocabulary: the reverse dictionary of token-id to sentence.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  return [rev_vocabulary.get(w, UNK_ID) for w in tokens]

def data_to_token_ids(data_path, target_path, vocabulary_path,
                      normalize_digits=True, start_index=None):
  """Turn already tokenized data file into token-ids using vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
    @start_index: this language should number starting from
      start_index. None is effectively 0.
  """
  start_index = start_index or 0

  if not gfile.Exists(target_path):
    print "Data to token ids for %s saving to %s" % (data_path, target_path)
    vocab, _ = initialize_vocabulary(vocabulary_path, start_index)
    with gfile.GFile(target_path, mode="w") as tokens_file:
      counter = 0
      with open(data_path, 'rb') as fin:
        for line in fin.readlines():
          line = line.strip()
          counter += 1
          if counter % 100000 == 0:
            print "  tokenizing line %d" % counter
          token_ids = sentence_to_token_ids(line, vocab, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def get_monolingual_data(training_file, dev_file, vocab_size, data_dir,
                         start_index=None):

  """
  The data has already been tokenized, one sentence per line.
  Output the

  @data_file: where the stringed data is held.
  @vocab_size: size of the vocabulary to create and use
  @start_index: this language should number starting from
    vocab_size*start_index. None is effectively 0.
  """
  start_index = start_index or 0
  start_index = start_index * vocab_size
  training_name = training_file.split('/')[-1]
  dev_name      = dev_file.split('/')[-1]

  # Create vocab file
  vocab_name = '.'.join([training_name, 'vocab', str(vocab_size)])
  vocab_path = os.path.join(data_dir, vocab_name)
  create_vocabulary(vocab_path, training_file, vocab_size)

  # Create token ids
  training_ids_name = '.'.join([training_name, 'token', str(vocab_size)])
  training_ids_path = os.path.join(data_dir, training_ids_name)
  dev_ids_name = '.'.join([dev_name, 'token', str(vocab_size)])
  dev_ids_path = os.path.join(data_dir, dev_ids_name)
  data_to_token_ids(training_file, training_ids_path, vocab_path,
                    start_index=start_index)
  data_to_token_ids(dev_file, dev_ids_path, vocab_path)

  return training_ids_path, dev_ids_path, vocab_path


def get_bucket_info(bucket_sizes, data_set):
  train_bucket_sizes = [len(data_set[b]) for b in xrange(len(bucket_sizes))]
  train_total_size = float(sum(train_bucket_sizes))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in xrange(len(train_bucket_sizes))]
  return train_bucket_sizes, train_total_size, train_buckets_scale


def convert_iswlt_corpus(data_file, output_file):
  """
  The iswlt files have segmented and aligned data. Unfortunately, some of the
  lines consist of multiple sentences. We want to split those into multiple
  sentences.

  We *also* want to run moses tokenizer over the data, which takes file input.

  So we read in a <data_file>, parse each line into multiple sentences,
  and remember the order of the ids, e.g. 6.1, 10.5, etc.
  corresponding to sixth line, first sentence, or tenth line, fifth sentence.

  Then we write those sentences in order, one per line, to an output file.
  Moses that file. Take new output and prefix the numbered id to each line.

  Write that output to a file called <name>.<language>.numbered. It looks like:
  "
  ...
  6.1|I 'm the sentence you 're looking for .\n
  ...
  "

  In addition, write the same output to <name>.<language>.monolingual, except
  drop the prefixed numbers.
  """
  def moses_prefix(num_sent, text, mosesout, language):
    prefixes = []

    if not isinstance(text, unicode):
      text = text.decode('utf-8')

    sentences = paragraph_tokenizer(text, language)
    if len(sentences) == 0 or text == '':
      prefixes.append(num_sent)
      mosesout.write('\n')
    else:
      for num_sub, sub_sent in enumerate(sentences):
        prefixes.append(num_sent + '.' + str(num_sub))

        if not isinstance(sub_sent, unicode):
          sub_sent = sub_sent.decode('utf-8')
        mosesout.write(sub_sent.encode('utf-8') + '\n')
    return prefixes

  def write_files(prefixes, moses_file, language):
    if output_file.endswith('.' + language):
      monolingual = '.'.join([output_file, 'monolingual'])
      numbered    = '.'.join([output_file, 'numbered'])
    else:
      monolingual = '.'.join([output_file, language, 'monolingual'])
      numbered    = '.'.join([output_file, language, 'numbered'])

    with open(monolingual, 'wb') as monoout:
      with open(numbered, 'wb') as numout:
        for index, line in enumerate(moses_tokenizer(moses_file, language)):
          if index >= len(prefixes):
            # moses puts an extra \n on the end, which causes an off-by-1
            break

          line = line.strip().decode('utf-8')
          prefix = prefixes[index]
          number_output = unicode(prefix + '|') + line
          numout.write(number_output.encode('utf-8') + '\n')
          monoout.write(line.encode('utf-8') + '\n')

  def convert_iswlt_xml():
    soup = BeautifulSoup(open(data_file, 'rb'), features='xml')
    language = data_file.split('.')[-2]
    number_prefixes = []

    moses_file = output_file + '.moses'
    with open(moses_file, 'wb') as mosesout:
      for seg in soup.findAll('seg'):
        number_prefixes.extend(
          moses_prefix(seg.get('id'), seg.text.strip(), mosesout, language)
          )

    write_files(number_prefixes, moses_file, language)
    os.remove(moses_file)

  def convert_tagged():
    language = data_file.split('.')[-1]
    number_prefixes = []

    moses_file = output_file + '.moses'
    with open(data_file, 'rb') as fin:
      lines = fin.readlines()

      with open(moses_file, 'wb') as mosesout:
        for num_sent, sent in enumerate(lines):
          sent = sent.strip()
          if sent.startswith('<'):
            continue

          number_prefixes.extend(
            moses_prefix(str(num_sent), sent, mosesout, language)
            )

    write_files(number_prefixes, moses_file, language)
    os.remove(moses_file)

  if data_file.endswith('.xml'):
    output_file = output_file.rstrip('.xml')
    print 'Outputting XML type %s to %s...' % (data_file, output_file)
    convert_iswlt_xml()
  elif 'train.tags' in data_file:
    print 'Outputting train.tags type %s to %s...' % (data_file, output_file)
    convert_tagged()
  else:
    print 'File %s doesnt fit something we know how to decipher.' % data_file

def convert_paired_data(lang_one_file, lang_two_file,
                        output_dir, output_prefix):
  """
  The input files are the result of running convert above. ('<name>.numbered')
  After doing that, some of the data is no longer paired because the sentences
  were not aligned 100%.

  Here, we take the paired data in lang_one_file and in lang_two_file and figure
  out which are the successful common pairs. We then align those and write them
  out to a file named <name>.<language one>.<language two>.bilingual as:

  "
  ...
  The house is nt green .<SEP>Das Haus ist nicht grun .\n
  ...
  "

  Args:
    lang_one_file: first language strings. file: <name>.<language>.numbered
    lang_two_file: second " "
    output_dir: directory to output the file to
    output_prefix: the file name for the output. this will be appended with
    '.<language one>.<language two>.bilingual'
  """
  lang_one = lang_one_file.split('.')[-2]
  lang_two = lang_two_file.split('.')[-2]
  lang_one_dict = defaultdict(list)
  lang_two_dict = defaultdict(list)

  with open(lang_one_file, 'rb') as f:
    for line in f.readlines():
      line = line.strip()
      line_int = int(line.split('.')[0].strip('|'))
      lang_one_dict[line_int].append(line)

  with open(lang_two_file, 'rb') as f:
    for line in f.readlines():
      line = line.strip()
      line_int = int(line.split('.')[0].strip('|'))
      lang_two_dict[line_int].append(line)

  output = []
  for number in lang_one_dict:
    lang_one_lines = lang_one_dict[number]
    lang_two_lines = lang_two_dict[number]
    if len(lang_one_lines) == len(lang_two_lines):
      for index in xrange(len(lang_one_lines)):
        prefix = lang_one_lines[index].split('|')[0]
        line_one = '|'.join(lang_one_lines[index].split('|')[1:])
        line_two = '|'.join(lang_two_lines[index].split('|')[1:])
        output.append((prefix, line_one, line_two))

  output = sorted(output, key = lambda entry: float(entry[0]))
  output_name = '.'.join([output_prefix, lang_one, lang_two, 'bilingual'])
  output_file = os.path.join(output_dir, output_name)
  with open(output_file, 'wb') as fout:
    for _, lang_one_line, lang_two_line in output:
      fout.write(lang_one_line + '<SEP>' + lang_two_line + '\n')


def reverse_bilingual_corpus(f, outfile):
  """
  Takes in a file produced by convert_paired_data and reverses it so
  that if the file previously was 'l1_sent<SEP>l2_sent', it's now
  'l2_sent<SEP>l1_sent'.
  """
  with open(f, 'rb') as fin:
    with open(outfile, 'wb') as fout:
      line = fin.readline()
      while line:
        line = line.strip()
        l1, l2 = line.split('<SEP>')
        fout.write('<SEP>'.join([l2, l1]) + '\n')
        line = fin.readline()


def group_list_into_sublists(lst, num_per_sublist):
    return [lst[x:x+num_per_sublist] for x in
            xrange(0, len(lst), num_per_sublist)]

def read_data_from_monolingual_file(source_path, vocab_file, max_size=None,
                                    bucket_sizes=None, start_index=0):
  """Read data from source file and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(bucket_sizes); data_set[n] contains a list
      of (source, source) pairs read from the data file that fits into the n-th
      bucket, i.e., such that len(source) < bucket_sizes[n][0]; the sources
      are lists of token-ids.
  """
  print 'vocab file: ', vocab_file
  vocab, _ = initialize_vocabulary(vocab_file)
  data_set = [[] for _ in bucket_sizes]

  with gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline().strip()
      counter = 0
      while source and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
              print "  reading data line %d" % counter
              sys.stdout.flush()

          line_ids = []
          for id_ in sentence_to_token_ids(source, vocab):
            if id_ not in _START_VOCAB_IDS:
              id_ += start_index
            line_ids.append(id_)

          for bucket_id, (source_size, target_size) in enumerate(bucket_sizes):
            if len(line_ids) + 1 < target_size:
              data_set[bucket_id].append([line_ids, line_ids + [EOS_ID],
                                          source.split(), source.split()])
              break
          source = source_file.readline().strip()
  return data_set

def read_data_from_paired_file(path, model_one_vocab_file, model_two_vocab_file,
                               bucket_sizes, max_size=None,
                               start_index_one=0, start_index_two=0):
    """
    Reads the data from a paired file and uses the vocab files to convert
    it into token sentence pairs.

    Returns dict of {bucket_id:[
      [
        (word_id for word in line_one),
        (word_id for word in line_two)
      ]
    ]}

    Args:
      path: path to the paired file. Each line looks like:
      "this is a house .<SEP>das ist ein Haus .\n"
    """
    def is_bucket(line_one_ids, line_two_ids, size_one, size_two):
        return len(line_one_ids) < size_one and len(line_two_ids) < size_two

    vocab_one, _ = initialize_vocabulary(model_one_vocab_file)
    vocab_two, _ = initialize_vocabulary(model_two_vocab_file)

    data_set = [[] for _ in bucket_sizes]
    with gfile.GFile(path, mode="r") as source_file:
        line = source_file.readline().strip()
        counter = 0
        while line and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print "  reading data line %d" % counter
                sys.stdout.flush()

            line_one, line_two = line.split('<SEP>')

            line_one_ids = []
            for id_ in sentence_to_token_ids(line_one, vocab_one):
              if id_ not in _START_VOCAB_IDS:
                id_ += start_index_one
              line_one_ids.append(id_)

            line_two_ids = []
            for id_ in sentence_to_token_ids(line_two, vocab_two):
              if id_ not in _START_VOCAB_IDS:
                id_ += start_index_two
              line_two_ids.append(id_)
            line_two_ids.append(EOS_ID)

            for index, (size_one, size_two) in enumerate(bucket_sizes):
              if is_bucket(line_one_ids, line_two_ids, size_one, size_two):
                data_set[index].append([line_one_ids, line_two_ids,
                                        line_one, line_two])

            line = source_file.readline().strip()
    return data_set
