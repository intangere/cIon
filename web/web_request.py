from flask import Flask, redirect, render_template, request, send_file, Markup
from model import create_model, buildSentence, respond
from config.config import FLAGS, _buckets, name
import tensorflow as tf
import data_utils
import os.path

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__,template_folder=tmpl_dir)

sess = tf.Session()

# Create model and load parameters.
model = create_model(sess, True)
model.batch_size = 1  # We decode one sentence at a time.

# Load vocabularies.
vocab_path = os.path.join(FLAGS.data_dir,
                             "vocab%d.in" % FLAGS.vocab_size)
vocab, vocab_rev = data_utils.initialize_vocabulary(vocab_path)

@app.route('/')
def index():
	return 'Define your chat handling code here using model.respond(sentence)'


app.debug = False
app.config["DEBUG"] = False
app.config["SECRET_KEY"] = "2q39r0ajsdpfasidjfasidfsdsmcaoi"
app.run(port=8564)