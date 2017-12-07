def loadParams(max_dilation_pow, expansion_reps, res_chan, dil_chan, skip_chan):
  
  wavenet_params = {'filter_width': 2, \
                    'sample_rate': 16000, \
                    'dilations': [2**i for i in range(max_dilation_pow)]*expansion_reps, \
                    'residual_channels': res_chan, \
                    'dilation_channels': dil_chan, \
                    'skip_channels': skip_chan, \
                    'use_biases': True, \
                    'scalar_input': False, \
                    'initial_filter_width': res_chan}
  return wavenet_params;
