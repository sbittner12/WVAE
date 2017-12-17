import numpy as np
import tensorflow as tf

from .ops import causal_conv, mu_law_encode


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        return tf.Variable(initial_val, name=name)
    else:
        return create_variable(name, shape)


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


class WaveNetModel(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 use_biases=False,
                 scalar_input=False,
                 midi_dims=88,
                 initial_filter_width=32,
                 histograms=False,
                 global_condition_channels=None,
                 global_condition_cardinality=None):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            scalar_input: Whether to use the quantized waveform directly as
                input to the network instead of one-hot encoding it.
                Default: False.
            initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input. This is only relevant
                if scalar_input=True.
            histograms: Whether to store histograms in the summary.
                Default: False.
            global_condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
            global_condition_cardinality: Number of mutually exclusive
                categories to be embedded in global condition embedding. If
                not None, then this implies that global_condition tensor
                specifies an integer selecting which of the N global condition
                categories, where N = global_condition_cardinality. If None,
                then the global_condition tensor is regarded as a vector which
                must have dimension global_condition_channels.

        '''
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.midi_dims = midi_dims
        self.initial_filter_width = initial_filter_width
        self.histograms = histograms
        self.global_condition_channels = global_condition_channels
        self.global_condition_cardinality = global_condition_cardinality

        self.receptive_field = WaveNetModel.calculate_receptive_field(
            self.filter_width, self.dilations, self.scalar_input,
            self.initial_filter_width)
        self.enc_vars, self.dec_vars = self._create_variables()

    @staticmethod
    def calculate_receptive_field(filter_width, dilations, scalar_input,
                                  initial_filter_width):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += filter_width - 1
        return receptive_field

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var_list = [dict(), dict()];
        var_scopes = ['encoder_wavenet', 'decoder_wavenet'];
        for i in range(len(var_list)):
            var = var_list[i];
            var_scope = var_scopes[i];
            with tf.variable_scope(var_scope):
                with tf.variable_scope('causal_layer'):
                    layer = dict()
                    if self.scalar_input:
                        initial_channels = 1
                        initial_filter_width = self.initial_filter_width
                    else:
                        if (var_scope == 'encoder_wavenet'):
                            initial_channels = self.midi_dims
                        else:
                            initial_channels = self.residual_channels
                        initial_filter_width = self.filter_width
                    layer['filter'] = create_variable(
                        'filter',
                        [initial_filter_width,
                         initial_channels,
                         self.residual_channels])
                    var['causal_layer'] = layer

                var['dilated_stack'] = list()
                with tf.variable_scope('dilated_stack'):
                    for i, dilation in enumerate(self.dilations):
                        with tf.variable_scope('layer{}'.format(i)):
                            current = dict()
                            current['filter'] = create_variable(
                                'filter',
                                [self.filter_width,
                                 self.residual_channels,
                                 self.dilation_channels])
                            current['gate'] = create_variable(
                                'gate',
                                [self.filter_width,
                                 self.residual_channels,
                                 self.dilation_channels])
                            current['dense'] = create_variable(
                                'dense',
                                [1,
                                 self.dilation_channels,
                                 self.residual_channels])
                            current['skip'] = create_variable(
                                'skip',
                                [1,
                                 self.dilation_channels,
                                 self.skip_channels])
                            if self.global_condition_channels is not None:
                                current['gc_gateweights'] = create_variable(
                                    'gc_gate',
                                    [1, self.global_condition_channels,
                                     self.dilation_channels])
                                current['gc_filtweights'] = create_variable(
                                    'gc_filter',
                                    [1, self.global_condition_channels,
                                     self.dilation_channels])

                            if self.use_biases:
                                current['filter_bias'] = create_bias_variable(
                                    'filter_bias',
                                    [self.dilation_channels])
                                current['gate_bias'] = create_bias_variable(
                                    'gate_bias',
                                    [self.dilation_channels])
                                current['dense_bias'] = create_bias_variable(
                                    'dense_bias',
                                    [self.residual_channels])
                                current['skip_bias'] = create_bias_variable(
                                   'skip_bias',
                                   [self.skip_channels])

                            var['dilated_stack'].append(current)

                with tf.variable_scope('postprocessing'):
                    if (var_scope == 'encoder_wavenet'):
                        output_channels = 2*self.residual_channels;
                    else:
                        output_channels = self.midi_dims;
                    current = dict()
                    current['postprocess1'] = create_variable(
                        'postprocess1',
                        [1, self.skip_channels, self.skip_channels])
                    current['postprocess2'] = create_variable(
                        'postprocess2',
                        [1, self.skip_channels, output_channels])
                    if self.use_biases:
                        current['postprocess1_bias'] = create_bias_variable(
                            'postprocess1_bias',
                            [self.skip_channels])
                        current['postprocess2_bias'] = create_bias_variable(
                            'postprocess2_bias',
                            [output_channels])
                    var['postprocessing'] = current

        return var_list

    def _create_causal_layer(self, input_batch, net_type='encoder'):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        print('create causal layer', net_type);
        if (net_type == 'encoder'):
            weights_filter = self.enc_vars['causal_layer']['filter']
            zeropad=False;
        elif (net_type == 'decoder'):
            weights_filter = self.dec_vars['causal_layer']['filter']
            zeropad=True;
        with tf.name_scope('causal_layer'):
            print('zeropad', zeropad)
            return causal_conv(input_batch, weights_filter, 1, zeropad)

    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               output_width, net_type='encoder'):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             global_conditioning_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        if (net_type == 'encoder'):
            variables = self.enc_vars['dilated_stack'][layer_index]
            zeropad=False;
        elif (net_type == 'decoder'):
            variables = self.dec_vars['dilated_stack'][layer_index]
            zeropad=True;


        weights_filter = variables['filter']
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation, zeropad=zeropad)
        conv_gate = causal_conv(input_batch, weights_gate, dilation, zeropad=zeropad)

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        out_dim1 = tf.shape(out)[1];

        # The 1x1 conv to produce the skip output
        if (net_type == 'encoder'):
            skip_cut = out_dim1 - output_width
            out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        elif (net_type == 'decoder'):
            skip_add = output_width - tf.shape(out)[1]
            delta = np.mod(skip_add, 2);
            out_dim0 = tf.shape(out)[0];
            out_dim2 = tf.shape(out)[2];
            out_skip = tf.concat((tf.zeros((out_dim0, skip_add//2+delta, out_dim2)), out), axis=1);
            out_skip = tf.concat((out_skip, tf.zeros((out_dim0, skip_add//2, out_dim2))), axis=1);
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        if self.histograms:
            layer = 'layer{}'.format(layer_index)
            tf.histogram_summary(layer + '_filter', weights_filter)
            tf.histogram_summary(layer + '_gate', weights_gate)
            tf.histogram_summary(layer + '_dense', weights_dense)
            tf.histogram_summary(layer + '_skip', weights_skip)
            if self.use_biases:
                tf.histogram_summary(layer + '_biases_filter', filter_bias)
                tf.histogram_summary(layer + '_biases_gate', gate_bias)
                tf.histogram_summary(layer + '_biases_dense', dense_bias)
                tf.histogram_summary(layer + '_biases_skip', skip_bias)

        if (net_type == 'encoder'):
            input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
            input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])
        elif (net_type == 'decoder'):
            input_dim0 = tf.shape(input_batch)[0];
            input_dim1 = tf.shape(input_batch)[1];
            input_dim2 = tf.shape(input_batch)[2];
            input_add = out_dim1-input_dim1;
            delta = np.mod(input_add, 2);
            input_batch = tf.concat((tf.zeros((input_dim0, input_add//2 + delta, input_dim2)), input_batch), axis=1);
            input_batch = tf.concat((input_batch, tf.zeros((input_dim0, input_add//2, input_dim2))), axis=1);
         
        residual_output = input_batch + transformed

        return skip_contribution, residual_output

    def _generator_conv(self, input_batch, state_batch, weights):
        '''Perform convolution for a single convolutional processing step.'''
        # TODO generalize to filter_width > 2
        past_weights = weights[0, :, :]
        curr_weights = weights[1, :, :]
        output = tf.matmul(state_batch, past_weights) + tf.matmul(
            input_batch, curr_weights)
        return output

    def _generator_causal_layer(self, input_batch, state_batch):
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            output = self._generator_conv(
                input_batch, state_batch, weights_filter)
        return output

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
                                  dilation, global_condition_batch):
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        output_filter = self._generator_conv(
            input_batch, state_batch, weights_filter)
        output_gate = self._generator_conv(
            input_batch, state_batch, weights_gate)

        if global_condition_batch is not None:
            global_condition_batch = tf.reshape(global_condition_batch,
                                                shape=(1, -1))
            weights_gc_filter = variables['gc_filtweights']
            weights_gc_filter = weights_gc_filter[0, :, :]
            output_filter += tf.matmul(global_condition_batch,
                                       weights_gc_filter)
            weights_gc_gate = variables['gc_gateweights']
            weights_gc_gate = weights_gc_gate[0, :, :]
            output_gate += tf.matmul(global_condition_batch,
                                     weights_gc_gate)

        if self.use_biases:
            output_filter = output_filter + variables['filter_bias']
            output_gate = output_gate + variables['gate_bias']

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = variables['dense']
        transformed = tf.matmul(out, weights_dense[0, :, :])
        if self.use_biases:
            transformed = transformed + variables['dense_bias']

        weights_skip = variables['skip']
        skip_contribution = tf.matmul(out, weights_skip[0, :, :])
        if self.use_biases:
            skip_contribution = skip_contribution + variables['skip_bias']

        return skip_contribution, input_batch + transformed

    def _create_encoder(self, input_batch, global_condition_batch):
        '''Construct the WaveNet encoder network.'''
        outputs = []
        layers = []
        current_layer = input_batch

        # Pre-process the input with a regular convolution
        if self.scalar_input:
            initial_channels = 1
        else:
            initial_channels = self.midi_dims

        print('create causal layer');
        current_layer = self._create_causal_layer(current_layer, net_type='encoder')
        layers.append(current_layer);
        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    print('create dilation layer', layer_index, dilation);
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                        output_width, net_type='encoder')
                    outputs.append(output)
                    layers.append(current_layer);

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.enc_vars['postprocessing']['postprocess1']
            w2 = self.enc_vars['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.enc_vars['postprocessing']['postprocess1_bias']
                b2 = self.enc_vars['postprocessing']['postprocess2_bias']

            if self.histograms:
                tf.histogram_summary('enc_postprocess1_weights', w1)
                tf.histogram_summary('enc_postprocess2_weights', w2)
                if self.use_biases:
                    tf.histogram_summary('enc_postprocess1_biases', b1)
                    tf.histogram_summary('enc_postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)

        mu = conv2[:,:,:self.residual_channels]
        sigma = conv2[:,:,self.residual_channels:]
        return mu, sigma, layers


    def _create_decoder(self, z, output_width):
        '''Construct the WaveNet decoder network.'''
        outputs = []
        current_layer = z

        # Pre-process the input with a regular convolution
        initial_channels = self.midi_dims

        current_layer = self._create_causal_layer(current_layer, net_type='decoder')

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    print('create dilation layer', layer_index, dilation);
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                        output_width, net_type='decoder')
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.dec_vars['postprocessing']['postprocess1']
            w2 = self.dec_vars['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.dec_vars['postprocessing']['postprocess1_bias']
                b2 = self.dec_vars['postprocessing']['postprocess2_bias']

            if self.histograms:
                tf.histogram_summary('dec_postprocess1_weights', w1)
                tf.histogram_summary('dec_postprocess2_weights', w2)
                if self.use_biases:
                    tf.histogram_summary('dec_postprocess1_biases', b1)
                    tf.histogram_summary('dec_postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)
        return tf.nn.sigmoid(conv2)

    def _create_generator(self, input_batch, global_condition_batch):
        '''Construct an efficient incremental generator.'''
        init_ops = []
        push_ops = []
        outputs = []
        current_layer = input_batch
        print('_create_generator input batch');
        print(input_batch);

        q = tf.FIFOQueue(
            1,
            dtypes=tf.float32,
            shapes=(self.batch_size, self.midi_dims))
        init = q.enqueue_many(
            tf.zeros((1, self.batch_size, self.midi_dims)))

        current_state = q.dequeue()
        push = q.enqueue([current_layer])
        init_ops.append(init)
        push_ops.append(push)

        current_layer = self._generator_causal_layer(
                            current_layer, current_state)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):

                    q = tf.FIFOQueue(
                        dilation,
                        dtypes=tf.float32,
                        shapes=(self.batch_size, self.residual_channels))
                    init = q.enqueue_many(
                        tf.zeros((dilation, self.batch_size,
                                  self.residual_channels)))

                    current_state = q.dequeue()
                    push = q.enqueue([current_layer])
                    init_ops.append(init)
                    push_ops.append(push)

                    output, current_layer = self._generator_dilation_layer(
                        current_layer, current_state, layer_index, dilation,
                        global_condition_batch)
                    outputs.append(output)
        self.init_ops = init_ops
        self.push_ops = push_ops

        with tf.name_scope('postprocessing'):
            variables = self.variables['postprocessing']
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = variables['postprocess1']
            w2 = variables['postprocess2']
            if self.use_biases:
                b1 = variables['postprocess1_bias']
                b2 = variables['postprocess2_bias']

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)

            conv1 = tf.matmul(transformed1, w1[0, :, :])
            if self.use_biases:
                conv1 = conv1 + b1
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.matmul(transformed2, w2[0, :, :])
            if self.use_biases:
                conv2 = conv2 + b2

        return conv2

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.quantization_channels,
                dtype=tf.float32)
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def _embed_gc(self, global_condition):
        '''Returns embedding for global condition.
        :param global_condition: Either ID of global condition for
               tf.nn.embedding_lookup or actual embedding. The latter is
               experimental.
        :return: Embedding or None
        '''
        embedding = None
        if self.global_condition_cardinality is not None:
            # Only lookup the embedding if the global condition is presented
            # as an integer of mutually-exclusive categories ...
            embedding_table = self.variables['embeddings']['gc_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               global_condition)
        elif global_condition is not None:
            # ... else the global_condition (if any) is already provided
            # as an embedding.

            # In this case, the number of global_embedding channels must be
            # equal to the the last dimension of the global_condition tensor.
            gc_batch_rank = len(global_condition.get_shape())
            dims_match = (global_condition.get_shape()[gc_batch_rank - 1] ==
                          self.global_condition_channels)
            if not dims_match:
                raise ValueError('Shape of global_condition {} does not'
                                 ' match global_condition_channels {}.'.
                                 format(global_condition.get_shape(),
                                        self.global_condition_channels))
            embedding = global_condition

        if embedding is not None:
            embedding = tf.reshape(
                embedding,
                [self.batch_size, 1, self.global_condition_channels])

        return embedding

    def predict_proba(self, waveform, global_condition=None, name='wavenet'):
        '''Computes the probability distribution of the next sample based on
        all samples in the input waveform.
        If you want to generate audio by feeding the output of the network back
        as an input, see predict_proba_incremental for a faster alternative.'''
        with tf.name_scope(name):
            if self.scalar_input:
                encoded = tf.cast(waveform, tf.float32)
                encoded = tf.reshape(encoded, [-1, 1])
            else:
                encoded = waveform

            gc_embedding = self._embed_gc(global_condition)
            raw_output = self._create_encoder(encoded, gc_embedding)
            out = tf.reshape(raw_output, [-1, self.midi_dims])
            # Cast to float64 to avoid bug in TensorFlow
            proba = tf.cast(
                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.midi_dims])
            return tf.reshape(last, [-1])

    def predict_proba_incremental(self, waveform, global_condition=None,
                                  name='wavenet'):
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''
        if self.filter_width > 2:
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")
        if self.scalar_input:
            raise NotImplementedError("Incremental generation does not "
                                      "support scalar input yet.")
        with tf.name_scope(name):
            #encoded = tf.one_hot(waveform, self.quantization_channels)
            #encoded = tf.reshape(encoded, [-1, self.quantization_channels])
            encoded = waveform
            gc_embedding = self._embed_gc(global_condition)
            raw_output = self._create_generator(encoded, gc_embedding)
            out = tf.reshape(raw_output, [-1, self.midi_dims])
            proba = tf.nn.sigmoid(out);
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.midi_dims])
            return tf.reshape(last, [-1])

    def loss(self,
             input_batch,
             global_condition_batch=None,
             l2_regularization_strength=None,
             name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.name_scope(name):
            # We mu-law encode and quantize the input audioform.
            gc_embedding = self._embed_gc(global_condition_batch)
            if self.scalar_input:
                network_input = tf.reshape(
                    tf.cast(input_batch, tf.float32),
                    [self.batch_size, -1, 1])
            else:
                network_input = input_batch

            # Cut off the last sample of network input to preserve causality.
            network_input_width = tf.shape(network_input)[1] - 1
            network_input = tf.slice(network_input, [0, 0, 0],
                                     [-1, network_input_width, -1])
            with tf.name_scope('encoder'):
                mu_enc, log_sigma_sq, enc_layers = self._create_encoder(network_input, gc_embedding)
                print('made encoder');

            

            #z = tf.placeholder(dtype=tf.float32, shape=mu.shape);
            eps = tf.random_normal(tf.shape(log_sigma_sq))
            z = mu_enc + tf.sqrt(tf.exp(log_sigma_sq)) * eps



            with tf.name_scope('decoder'):
                raw_output = self._create_decoder(z, network_input_width)


            with tf.name_scope('loss'):
                # Cut off the samples corresponding to the receptive field
                # for the first predicted sample.
                #target_output = tf.slice(
                #    tf.reshape(
                #        input_batch,
                #        [self.batch_size, -1, self.midi_dims]),
                #    [0, self.receptive_field, 0],
                #    [-1, -1, -1])
                target_output = tf.reshape(network_input,
                                           [-1, self.midi_dims])
                prediction = tf.reshape(raw_output, [-1, self.midi_dims])
                #loss = tf.reduce_sum(tf.square(target_output-prediction), axis=1);
                #reduced_loss = tf.reduce_mean(loss)

                # Prediction \in (0,1)
                recon_loss = tf.reduce_mean(target_output * tf.log(1e-10 + prediction) + (1 - target_output) * tf.log(1e-10+1- prediction))

                sigma_sq = tf.exp(log_sigma_sq)
                latent_loss = -.5 * tf.reduce_mean((1 + tf.log(1e-10 + sigma_sq)) - tf.square(mu_enc) - sigma_sq);
                
                total_loss = recon_loss + latent_loss


                print('made loss', total_loss);

                tf.summary.scalar('recon_loss', recon_loss)
                tf.summary.scalar('latent_loss', latent_loss)
                tf.summary.scalar('total_loss', total_loss)

                if l2_regularization_strength is None:
                    return total_loss, recon_loss, latent_loss, target_output, prediction, mu_enc, enc_layers
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (total_loss +
                                  l2_regularization_strength * l2_loss)

                    return total_loss, recon_loss, latent_loss, target_output, prediction, mu_enc, enc_layers
