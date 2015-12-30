"""
Binary for the Multilingual model.

A Multilingual is a Seq2Seq that takes in any language input and can spit out
any language output. It's a seq2seq model with a single encoder and a single
decoder.

The vocabulary is formed by taking the top 30k words from each corpus. These
dicts are then appended to form a vocabulary of 30k * N words.

The encoder is given a batch from one language. After encoding, it then passes
off to a language-specific attention module. This module is chosen based off of
the language that we want to output into. That then feeds into the single
decoder which outputs the target language.

One question to consider is what to do with shared words. It's clear that 'will'
in English is different from 'will' in German, but does that mean that the
encoder / decoder should learn a different word for each of these? The above
description *does* learn a different word for each language.
"""

import os
import random
import itertools
import time
import math
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow.python.platform import gfile
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

import utils

layer_size = 128
num_layers = 1
batch_size = 32
max_gradient_norm = 5.0
learning_rate = 0.5
learning_rate_decay_factor = .99
vocab_size_per_language = 30000
max_train_data_size = 0

languages = ['de', 'en']
languages = sorted(languages)
language_pairs = [pair for pair in itertools.product(languages, languages)]

steps_per_checkpoint = 10
# how many steps to do for each pair before moving to the next one
steps_per_pair = steps_per_checkpoint * 2
steps_per_summary = steps_per_checkpoint

is_decode = False

def get_or_create_directory(prefix):
    ret = '%s/%s-Multilingual-%d-%d-%d-%d-%d' % (
        '-'.join(languages), prefix, vocab_size_per_language, layer_size,
        num_layers, batch_size, max_train_data_size)
    if not os.path.exists(ret):
        os.makedirs(ret)
    return ret

def get_or_create_model_directory():
    return get_or_create_directory('model')

def get_or_create_data_directory():
    return get_or_create_directory('data')

basedir  = os.path.abspath(os.path.dirname(__file__))
data_dir = get_or_create_data_directory()

monolingual_files = {}
bilingual_files = {}
vocab_paths = {}

print 'Preparing data from all of the different languages.'
for num, (source, target) in enumerate(language_pairs):
    print 'Preparing pair num %d (%s -> %s)' % (num, source, target)
    if source == target:
        train_file = os.path.join(basedir, 'training_files', 'multilingual',
                                  'train.%s.monolingual' % source)
        dev_file = os.path.join(basedir, 'training_files', 'multilingual',
                                'dev.%s.monolingual' % source)
        _, _, vocab_path = utils.get_monolingual_data(
            train_file, dev_file, vocab_size_per_language, data_dir,
            languages.index(source))
        monolingual_files[source] = (train_file, dev_file) # paths
        vocab_paths[source] = vocab_path
    else:
        train_file = os.path.join(
            basedir, 'training_files', 'multilingual',
            'train.%s-%s.bilingual' % (source, target))
        dev_file = os.path.join(
            basedir, 'training_files', 'multilingual',
            'dev.%s-%s.bilingual' % (source, target))
        bilingual_files[(source, target)] = (train_file, dev_file)
print 'Done preparing files!'

bucket_sizes = [(5, 10), (10, 15)]#, (20, 25), (40, 50)]

def create_model(session, forward_only):
  """Create model and initialize or load parameters in session."""
  model = MultilingualSeq2SeqModel(
      vocab_size_per_language * len(languages), bucket_sizes, layer_size,
      num_layers, max_gradient_norm, batch_size, learning_rate,
      learning_rate_decay_factor, forward_only=forward_only)

  model_dir = get_or_create_model_directory()
  merged = tf.merge_all_summaries()
  writer = tf.train.SummaryWriter(os.path.join(basedir, "multilingual-logs"),
                                  session.graph_def)

  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model, merged, writer

class MultilingualSeq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This is architecturally very similar to the Seq2Seq model in the TF package.
  The main difference is that it uses language-dependent attention functions.
  """

  def __init__(self, total_vocab_size, buckets, layer_size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=False,
               num_samples=512, forward_only=False):
    """Create the model.

    Args:
      total_vocab_size: size of the combined vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      layer_size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    self.total_vocab_size = total_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.total_vocab_size:
      with tf.device("/cpu:0"):
        w = tf.get_variable("proj_w", [layer_size, self.total_vocab_size])
        w_hist = tf.histogram_summary('proj_w_hist', w)
        w_t = tf.transpose(w)
        b = tf.get_variable("proj_b", [self.total_vocab_size])
        b_hist = tf.histogram_summary('proj_b_hist', b)
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          ssl = tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                           self.total_vocab_size)
          # ssl_hist = tf.histogram_summary('sampled_softmax_loss_hist', ssl)
          return ssl

      softmax_loss_function = sampled_loss

    # Create the internal multi-layer cell for our RNN.
    single_cell = rnn_cell.GRUCell(layer_size)
    if use_lstm:
      single_cell = rnn_cell.BasicLSTMCell(layer_size)
    cell = single_cell
    if num_layers > 1:
      cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)

    encoder_cell = rnn_cell.EmbeddingWrapper(cell, total_vocab_size)
    with vs.variable_scope('embedding_decoder_top_level'):
        with ops.device("/cpu:0"):
            embedding = vs.get_variable("embedding",
                                        [total_vocab_size, cell.input_size])

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, language, do_decode):
        return seq2seq.embedding_attention_seq2seq(
            encoder_inputs, decoder_inputs, cell, total_vocab_size,
            total_vocab_size, output_projection=output_projection,
            feed_previous=do_decode, module=language,
            encoder_cell=encoder_cell, embedding=embedding,
            modules=languages)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    self.language = tf.placeholder(tf.string, name="language")

    # Training outputs and losses.
    loss_summaries = {}
    clipped_gradient_summaries = {}

    self.all_losses = {}
    self.all_outputs = {}
    self.all_updates = {}
    self.all_gradient_norms = {}

    for num, language in enumerate(languages):
        with vs.variable_scope('multilingual', reuse=True if num > 0 else None):
            if forward_only:
                _outputs, _losses = seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, buckets, self.total_vocab_size,
                    lambda x, y: seq2seq_f(x, y, language, True),
                    softmax_loss_function=softmax_loss_function)

                if output_projection is not None:
                    # If we use output projection, project outputs for decoding.
                    for b in xrange(len(buckets)):
                        _outputs[b] = [
                            tf.nn.xw_plus_b(output, output_projection[0],
                                            output_projection[1])
                            for output in _outputs[b]]
            else:
                _outputs, _losses = seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, buckets, self.total_vocab_size,
                    lambda x, y: seq2seq_f(x, y, language, False),
                    softmax_loss_function=softmax_loss_function)

        for i in xrange(len(_losses)):
            name = 'loss_%s_%d' % (language, i)
            loss_summaries[name] = tf.scalar_summary(name, _losses[i])

        self.all_losses[language] = _losses
        self.all_outputs[language] = _outputs
        params = tf.trainable_variables()

        # Gradients and SGD update operation for training the model.
        if not forward_only:
            gradient_norms = []
            updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(_losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients, max_gradient_norm)
                gradient_norms.append(norm)
                updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self.global_step))

            self.all_updates[language] = updates
            self.all_gradient_norms[language] = gradient_norms

    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs, decoder_inputs, output_language,
           target_weights, bucket_id, forward_only, merged):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      output_language: two-letter string of the target output language.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of enconder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    print 'encoder_size: %d, decoder_size: %d' % (encoder_size, decoder_size)
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]
    input_feed[self.language.name] = output_language

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    updates = self.all_updates[output_language]
    gradient_norms = self.all_gradient_norms[output_language]
    losses = self.all_losses[output_language]
    outputs = self.all_outputs[output_language]

    output_feed = [merged]
    if not forward_only:
      output_feed.extend([updates[bucket_id],  # Update Op that does SGD.
                          gradient_norms[bucket_id],  # Gradient norm.
                          losses[bucket_id]])  # Loss for this batch.
    else:
      output_feed.extend([losses[bucket_id]])  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      # summary_str, Gradient norm, loss, no outputs.
      return outputs[0], outputs[1], outputs[2], None
    else:
      # summary_str, No gradient norm, loss, outputs.
      return outputs[0], None, outputs[1], outputs[2:]

  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input, _, _ = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([utils.GO_ID] + decoder_input +
                            [utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

def get_data_from_language_pair(pair):
    l1, l2 = pair
    if l1 == l2:
        train, dev = monolingual_files[l1]
        start_index = languages.index(l1) * vocab_size_per_language
        train_set = utils.read_data_from_monolingual_file(
            train, vocab_paths[l1], max_train_data_size, bucket_sizes,
            start_index=start_index)
        dev_set = utils.read_data_from_monolingual_file(
            dev, vocab_paths[l1], max_train_data_size, bucket_sizes,
            start_index=start_index)
    else:
        train, dev = bilingual_files[pair]
        start_index_one = languages.index(l1) * vocab_size_per_language
        start_index_two = languages.index(l2) * vocab_size_per_language
        train_set = utils.read_data_from_paired_file(
            train, vocab_paths[l1], vocab_paths[l2], bucket_sizes,
            start_index_one=start_index_one, start_index_two=start_index_two)
        dev_set = utils.read_data_from_paired_file(
            dev, vocab_paths[l1], vocab_paths[l2], bucket_sizes,
            start_index_one=start_index_one, start_index_two=start_index_two)
    return train_set, dev_set

# language_pairs = [p for p in language_pairs if p[0] != p[1]]

def train():
    _pairs = [p for p in language_pairs]

    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (num_layers, layer_size))
        model, merged, writer = create_model(sess, False)

        current_pair = random.choice(_pairs)
        print 'Choosing the first pair --> ', current_pair
        _pairs.remove(current_pair)
        print 'Reading data...'
        train_set, dev_set = get_data_from_language_pair(current_pair)
        print 'Got data!'

        info = utils.get_bucket_info(bucket_sizes, train_set)
        train_bucket_sizes, train_total_size, train_buckets_scale = info

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = model.global_step.eval()
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            print 'Language Pair: %s --> %s' % (current_pair[0], current_pair[1])
            summary_str, _, step_loss, _ = model.step(
                sess, encoder_inputs, decoder_inputs, current_pair[1],
                target_weights, bucket_id, False, merged)
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f "
                       "perplexity %.2f" % (
                           model.global_step.eval(), model.learning_rate.eval(),
                           step_time, perplexity)
                       )
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(get_or_create_model_directory(),
                                               "multilingual-training.ckpt")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(bucket_sizes)):
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, _, eval_loss, _ = model.step(
                        sess, encoder_inputs, decoder_inputs, current_pair[1],
                        target_weights, bucket_id, True, merged)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

            # After steps_per_pair, choose a new pair
            if current_step % steps_per_pair == 0:
                print 'Choosing a new pair (previously: ', current_pair, ' )'
                if not _pairs:
                    print 'We cycled through all the pairs. Starting at the top.'
                    _pairs = [k for k in language_pairs]

                current_pair = random.choice(_pairs)
                _pairs.remove(current_pair)
                print 'New Pair: ', current_pair
                print 'Getting the training and dev sets for it...'
                train_set, dev_set = get_data_from_language_pair(current_pair)

            if current_step % steps_per_summary == 0:
                print 'Recording summary data'
                writer.add_summary(summary_str, current_step)



def main(_):
  if is_decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
