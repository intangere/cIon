import tensorflow as tf
import numpy as np
from model import create_model, buildSentence, respond
from config.config import FLAGS, _buckets, name
import data_utils
import os.path

sess = tf.Session()
# Create model and load parameters.
model = create_model(sess, True)
model.batch_size = 1  # We decode one sentence at a time.

# Load vocabularies.
vocab_path = os.path.join(FLAGS.data_dir,
                             "vocab%d.in" % FLAGS.vocab_size)
vocab, vocab_rev = data_utils.initialize_vocabulary(vocab_path)

print '%s: %s' % (name, respond('hi.', sess, model, vocab, vocab_rev))

print '%s: %s' % (name, respond('hello.', sess, model, vocab, vocab_rev))

print '%s: %s' % (name, respond('hey.', sess, model, vocab, vocab_rev))

print '%s: %s' % (name, respond('how are you?', sess, model, vocab, vocab_rev))

print '%s: %s' % (name, respond('what is the meaning of life?', sess, model, vocab, vocab_rev))

print '%s: %s' % (name, respond('you are a machine.', sess, model, vocab, vocab_rev))

print '%s: %s' % (name, respond('you\'re a machine.', sess, model, vocab, vocab_rev))

print '%s: %s' % (name, respond('can you hold me?', sess, model, vocab, vocab_rev))

print '%s: %s' % (name, respond('i love you.', sess, model, vocab, vocab_rev))

print '%s: %s' % (name, respond('i just wanted some love.', sess, model, vocab, vocab_rev))