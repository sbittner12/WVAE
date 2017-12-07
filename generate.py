from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import librosa
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, audio_reader
from wavenet.midi_reader.midi_utils import midiwrite

SAMPLES = 200
TEMPERATURE = 1.0
LOGDIR = './logdir'
SAVE_EVERY = None
SILENCE_THRESHOLD = 0.1


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'resdir', type=str, help='Which model directory to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many waveform samples to generate')
    parser.add_argument(
        '--gen_num',
        type=int,
        default=1,
        help='Index of complete song generated')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=True,
        help='Use fast generation')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    parser.add_argument(
        '--gc_channels',
        type=int,
        default=None,
        help='Number of global condition embedding channels. Omit if no '
             'global conditioning.')
    parser.add_argument(
        '--gc_cardinality',
        type=int,
        default=None,
        help='Number of categories upon which we globally condition.')
    parser.add_argument(
        '--gc_id',
        type=int,
        default=None,
        help='ID of category to generate, if globally conditioned.')
    arguments = parser.parse_args()
    if arguments.gc_channels is not None:
        if arguments.gc_cardinality is None:
            raise ValueError("Globally conditioning but gc_cardinality not "
                             "specified. Use --gc_cardinality=377 for full "
                             "VCTK corpus.")

        if arguments.gc_id is None:
            raise ValueError("Globally conditioning, but global condition was "
                              "not specified. Use --gc_id to specify global "
                              "condition.")

    return arguments


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def create_seed(filename,
                sample_rate,
                fzation_channels,
                window_size,
                silence_threshold=SILENCE_THRESHOLD):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio_reader.trim_silence(audio, silence_threshold)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))

    return quantized[:cut_index]


def main():
    midi_dims = 88;
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    checkpoint = tf.train.latest_checkpoint(args.resdir);
    print('checkpoint: ', checkpoint);
    wavenet_params_fname = args.resdir + 'wavenet_params.json';
    print('wavenet params fname', wavenet_params_fname);
    with open(wavenet_params_fname, 'r') as config_file:
        wavenet_params = json.load(config_file)
        wavenet_params['midi_dims'] = midi_dims;

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        midi_dims=wavenet_params['midi_dims'],
        initial_filter_width=wavenet_params['initial_filter_width'],
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=args.gc_cardinality)

    samples = tf.placeholder(tf.float32, shape=(None, midi_dims))

    if args.fast_generation:
        next_sample = net.predict_proba_incremental(samples, args.gc_id)
    else:
        next_sample = net.predict_proba(samples, args.gc_id)

    if args.fast_generation:
        sess.run(tf.global_variables_initializer())
        sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)
    print('vars to restore');
    for key in variables_to_restore.keys():
        print(key,variables_to_restore[key]);


    print('Restoring model from {}'.format(checkpoint))
    saver.restore(sess, checkpoint)

    if args.wav_seed:
        seed = create_seed(args.wav_seed,
                           wavenet_params['sample_rate'],
                           midi_dims,
                           net.receptive_field)
        waveform = sess.run(seed).tolist()
    else:
        # Silence with a single random sample at the end.
        #waveform = [quantization_channels / 2] * (net.receptive_field - 1)
        #waveform.append(np.random.randint(quantization_channels))
        random_note = np.zeros((1,midi_dims));
        random_note[0,np.random.randint(0, midi_dims-1)] = 1.0;
        waveform = np.concatenate((np.zeros((net.receptive_field - 1, midi_dims)), random_note), axis=0);


    if args.fast_generation and args.wav_seed:
        print('fast gen');
        # When using the incremental generation, we need to
        # feed in all priming samples one by one before starting the
        # actual generation.
        # TODO This could be done much more efficiently by passing the waveform
        # to the incremental generator as an optional argument, which would be
        # used to fill the queues initially.
        outputs = [next_sample]
        outputs.extend(net.push_ops)

        print('Priming generation...')
        for i, x in enumerate(waveform[-net.receptive_field: -1]):
            if i % 100 == 0:
                print('Priming sample {}'.format(i))
            sess.run(outputs, feed_dict={samples: x})
        print('Done.')

    print('receptive field is %d' % net.receptive_field);
    last_sample_timestamp = datetime.now()
    for step in range(args.samples):
        if args.fast_generation:
            outputs = [next_sample]
            outputs.extend(net.push_ops)
            window = np.expand_dims(waveform[-1, :], 0);
        else:
            if len(waveform) > net.receptive_field:
                window = waveform[-net.receptive_field:]
            else:
                window = waveform
            outputs = [next_sample]

        print(step, 'wave shape', waveform.shape, 'window shape', window.shape);
        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: window})[0]

        # Scale prediction distribution using temperature.
        #np.seterr(divide='ignore')
        #scaled_prediction = np.log(prediction) / args.temperature
        #scaled_prediction = (scaled_prediction -
        #                     np.logaddexp.reduce(scaled_prediction))
        #scaled_prediction = np.exp(scaled_prediction)
        #np.seterr(divide='warn')

        # Prediction distribution at temperature=1.0 should be unchanged after
        # scaling.
        #if args.temperature == 1.0:
        #    np.testing.assert_allclose(
        #            prediction, scaled_prediction, atol=1e-5,
        #           err_msg='Prediction scaling at temperature=1.0 '
        #                    'is not working as intended.')
        sample = 1*(prediction > 0.5);
        print('num notes', np.count_nonzero(sample));
        #sample = np.random.choice(
        #    np.arange(quantization_channels), p=scaled_prediction)
        waveform = np.concatenate((waveform, np.expand_dims(sample, 0)), axis=0);

        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

    # Introduce a newline to clear the carriage return from the progress.
    print()

    # Save the result as an audio summary.
    datestring = str(datetime.now()).replace(' ', 'T')
    #writer = tf.summary.FileWriter(logdir)
    #tf.summary.audio('generated', decode, wavenet_params['sample_rate'])
    #summaries = tf.summary.merge_all()
    #print('waveform', waveform);
    #summary_out = sess.run(summaries, feed_dict={samples: waveform})
    #writer.add_summary(summary_out)

    # Save the result as a wav file.
    if args.wav_out_path is None:
        args.wav_out_path = args.resdir;

    #out = sess.run(decode, feed_dict={samples: waveform})
    print(args.wav_out_path);
    filename = args.wav_out_path + ('sample_%d.mid' % int(args.gen_num));
    midiwrite(filename, waveform)

if __name__ == '__main__':
    main()
