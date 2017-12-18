"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time
import numpy as np

import tensorflow as tf
from tensorflow.python.client import timeline

from wavenet import WaveNetModel, optimizer_factory
from wavenet.midi_reader import MidiReader, load_all_audio
from wavenet.params import loadParams

BATCH_SIZE = 1
DATA_SET = 'JSB_Chorales'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 500
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-6
MAX_DILATION_POW  = 4;
EXPANSION_REPS = 1;
DIL_CHAN = 32;
RES_CHAN = 32;
SKIP_CHAN = 32;
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 5
METADATA = True


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--data_set', type=str, default=DATA_SET,
                        help='String id for Nottingham or JSB_Chorales.')
    parser.add_argument('--store_metadata', type=bool, default=METADATA,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--max_dilation_pow', type=int, default=MAX_DILATION_POW,
                        help='Maximum dilation of causal convolutional filter'
                        'max_dilation_pow. Default: ' + str(MAX_DILATION_POW) + '.')
    parser.add_argument('--dil_chan', type=int, default=DIL_CHAN,
                        help='Number of dilation channels'
                        'dil_chan. Default: ' + str(DIL_CHAN) + '.')
    parser.add_argument('--res_chan', type=int, default=RES_CHAN,
                        help='Number of residual channels'
                        'res_chan. Default: ' + str(RES_CHAN) + '.')
    parser.add_argument('--skip_chan', type=int, default=SKIP_CHAN,
                        help='Number of skip channels'
                        'skip_chan. Default: ' + str(SKIP_CHAN) + '.')
    parser.add_argument('--expansion_reps', type=int, default=EXPANSION_REPS,
                        help='How many times to repeat dilated causal convolutional expansion'
                        'expansion_reps. Default: ' + str(EXPANSION_REPS) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Default: False')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                        help='Whether to store histogram summaries. Default: False')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels. Default: None. Expecting: Int')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def main():
    args = get_arguments()
    data_dir = 'midi-Corpus/' + args.data_set + '/'
    logdir = data_dir + 'max_dilation=%d_reps=%d/' % (args.max_dilation_pow, args.expansion_reps);
    print('*************************************************');
    print(logdir);
    print('*************************************************');
    sys.stdout.flush()
    restore_from = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    wavenet_params = loadParams(args.max_dilation_pow, args.expansion_reps, args.dil_chan, args.res_chan, args.skip_chan);
        
    with open(logdir + 'wavenet_params.json', 'w') as outfile:
        json.dump(wavenet_params, outfile)

    # Create coordinator.
    coord = tf.train.Coordinator()

    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        gc_enabled = False
        # data queue for the training set
        train_dir = data_dir + 'train/';
        train_reader = MidiReader(
            train_dir,
            coord,
            sample_rate=wavenet_params['sample_rate'],
            gc_enabled=gc_enabled,
            receptive_field=WaveNetModel.calculate_receptive_field(wavenet_params["filter_width"],
                                                                   wavenet_params["dilations"],
                                                                   wavenet_params["scalar_input"],
                                                                   wavenet_params["initial_filter_width"]),
            sample_size=args.sample_size)
        train_batch = train_reader.dequeue(args.batch_size)
        # data queue for the validation set
        #valid_dir = data_dir + 'valid/';
        #valid_reader = MidiReader(
        #    valid_dir,
        #    coord,
        #    sample_rate=wavenet_params['sample_rate'],
        #    gc_enabled=gc_enabled,
        #    receptive_field=WaveNetModel.calculate_receptive_field(wavenet_params["filter_width"],
        #                                                           wavenet_params["dilations"],
        #                                                           wavenet_params["scalar_input"],
        #                                                           wavenet_params["initial_filter_width"]),
        #    sample_size=args.sample_size)
        #valid_batch = valid_reader.dequeue(args.batch_size)
        if gc_enabled:
            gc_id_batch = reader.dequeue_gc(args.batch_size)
        else:
            gc_id_batch = None
            
    # Create network.
    net = WaveNetModel(
        batch_size=BATCH_SIZE,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        use_biases=wavenet_params["use_biases"],
        scalar_input=wavenet_params["scalar_input"],
        initial_filter_width=wavenet_params["initial_filter_width"],
        histograms=False,
        global_condition_channels=None,
        global_condition_cardinality=train_reader.gc_category_cardinality)
    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    print('constructing training loss');
    sys.stdout.flush()
    train_loss, recon_loss, latent_loss, target_output, prediction, mu_enc, layers = net.loss(input_batch=train_batch,
                    global_condition_batch=gc_id_batch,
                    l2_regularization_strength=args.l2_regularization_strength)
    print('constructing validation loss');
    sys.stdout.flush()
    #valid_loss, target_output, prediction = net.loss(input_batch=valid_batch,
    #                global_condition_batch=gc_id_batch,
    #                l2_regularization_strength=args.l2_regularization_strength)

    print('making optimizer');
    sys.stdout.flush()
    optimizer = optimizer_factory['adam'](
                    learning_rate=args.learning_rate,
                    momentum=args.momentum)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(train_loss, var_list=trainable)

    print('setting up tensorboard');
    sys.stdout.flush()
    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    valid_input = tf.placeholder(dtype=tf.float32, shape=(1, None, 88));
    valid_loss, valid_recon_loss, valid_latent_loss, valid_target_output, valid_prediction, valid_mu, valid_enc_layers = net.loss(input_batch=valid_input,
                    global_condition_batch=gc_id_batch,
                    l2_regularization_strength=args.l2_regularization_strength)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    print('saver');
    sys.stdout.flush()
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=5)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    print('thread stuff');
    sys.stdout.flush()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    train_reader.start_threads(sess)

    step = None
    last_saved_step = saved_global_step
 
    # load validation data
    validation_audio = load_all_audio(data_dir + 'valid/');
    num_valid_files = len(validation_audio);
    valid_loss_values = np.zeros((int(np.ceil(args.num_steps/500)),));
    vl_ind = 0;
    print('figuring out wtf is going on time');
    sys.stdout.flush()

    valid_losses_step = np.zeros((num_valid_files,));
    audio_0 = np.expand_dims(validation_audio[2], 0);
    print('audio 0', audio_0.shape);
    print(audio_0);
    mu_enc_0, enc_layers_0 = sess.run([valid_mu, valid_enc_layers], {valid_input:audio_0});
    print('layer shapes');
    for layer in enc_layers_0:
        print(layer.shape);
    print('mu 0', mu_enc_0.shape);
    print(mu_enc_0);
    valid_loss_0 = sess.run(valid_loss, {valid_input:audio_0});
    print('valid_loss_0', valid_loss_0);
    #print('validation loss 0', valid_losses_step_0);

    print('optimization time');
    sys.stdout.flush()
    min_valid_loss = 1e10;
    try:
        for step in range(saved_global_step + 1, args.num_steps):
            print('step', step);
            sys.stdout.flush()
            start_time = time.time()
            if args.store_metadata and step % 500 == 0:
                # Slow run that stores extra information for debugging.
                print('Storing metadata')
                sys.stdout.flush()
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                print('mu comp')
                sys.stdout.flush()
                _mu_enc = sess.run(
                    mu_enc,
                    options=run_options,
                    run_metadata=run_metadata)
                print(_mu_enc.shape);
                summary, loss_value, _ = sess.run(
                    [summaries, train_loss, optim],
                    options=run_options,
                    run_metadata=run_metadata)
                print('writing summary')
                sys.stdout.flush()
                writer.add_summary(summary, step)
                writer.add_run_metadata(run_metadata,
                                        'step_{:04d}'.format(step))
                valid_losses_step = np.zeros((num_valid_files,));
                for i in range(num_valid_files):
                    audio_i = np.expand_dims(validation_audio[i], 0);
                    valid_losses_step[i] = sess.run(valid_loss, {valid_input:audio_i});
                valid_loss_value_step = np.mean(valid_losses_step);
                valid_loss_values[vl_ind] = valid_loss_value_step
                np.savez(logdir + 'validation.npz', validation_loss=valid_loss_values);
                vl_ind += 1;
                tl = timeline.Timeline(run_metadata.step_stats)
                timeline_path = os.path.join(logdir, 'timeline.trace')
                with open(timeline_path, 'w') as f:
                    f.write(tl.generate_chrome_trace_format(show_memory=True))

                if (valid_loss_value_step < min_valid_loss and (not np.isnan(valid_loss_value_step))):
                    min_valid_loss = valid_loss_value_step;
                    save(saver, sess, logdir, step)
                    last_saved_step = step
            else:
                _rec_ls, _lat_ls, _tot_ls, _pred  = sess.run([recon_loss, latent_loss, train_loss, prediction])
                print('recon', _rec_ls, 'latent', _lat_ls, 'total', _tot_ls, 'max pred', np.max(_pred), 'min pred', np.min(_pred));
                summary, loss_value, _ = sess.run([summaries, train_loss, optim])
                writer.add_summary(summary, step)

            duration = time.time() - start_time
            print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
                  .format(step, loss_value, duration))
            sys.stdout.flush()


    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
