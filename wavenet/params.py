def loadParams(max_dilation_pow, expansion_reps):
  
  wavenet_params = {'filter_width': 2, \
                    'sample_rate': 16000, \
                    'dilations': [2**i for i in range(max_dilation_pow)]*expansion_reps, \
                    'residual_channels': 32, \
                    'dilation_channels': 32, \
                    'skip_channels': 16, \
                    'use_biases': True, \
                    'scalar_input': False, \
                    'initial_filter_width': 32}
  return wavenet_params;
