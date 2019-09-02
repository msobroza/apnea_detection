# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

import numpy as np
from scipy.io import wavfile
import six
import csv
import vggish_input
import vggish_params
import vggish_postprocess
import datetime
import numpy as np
import argparse
import csv
import sys
import json
import boto3
import numpy as np
import io
import glob
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Compares the inference in keras and in tf.')
parser.add_argument("--wav_file", default=None,
                    help='Path to a wav file. Should contain signed 16-bit PCM samples'
                         ' If none is provided, a synthetic sound is used.')
parser.add_argument("--pca_params", default='vggish_pca_params.npz',
                    help='Path to the VGGish PCA parameters file.')
parser.add_argument("--tfrecord_file", default=None,
                    help='Path to a TFRecord file where embeddings will be written.')

FLAGS = parser.parse_args()


def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128

def main():
    num_secs = 3
    # In this simple example, we run the examples from a single audio file through
    # the model. If none is provided, we generate a synthetic input.
    if FLAGS.wav_file:
        wav_file = FLAGS.wav_file
        print(wav_file)
    snore_path = "/home/grodri/bacelar/apnea_detection/data/audio/snore_3s/"
    bg_path = "/home/grodri/bacelar/apnea_detection/data/audio/bg_3s/"
    # Load datasets to dictionary
    snore_IDs= glob.glob(snore_path+'*.wav')
    bg_IDs  = glob.glob(bg_path + '*.wav')
    for c, class_file in enumerate(['bg','snore']):
        result = list()
        if class_file=='snore':
            continue
            file_ids = snore_IDs
        else:
            file_ids = bg_IDs
        for wav_file in tqdm(file_ids):
            filename = wav_file.split('/')[-1]
            examples_batch = vggish_input.wavfile_to_examples(wav_file)
            # Prepare a postprocessor to munge the model embeddings.
            pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
            client = boto3.client('runtime.sagemaker', region_name='eu-west-1')
            data = np.expand_dims(examples_batch, axis=-1).tolist()
            endpoint_feat_extract = 'vggish-features'
            response = client.invoke_endpoint(EndpointName=endpoint_feat_extract, Body=json.dumps(data))
            body = response['Body'].read().decode('utf-8')
            embedding_sound = np.array(json.loads(body)['outputs']['vgg_features']['floatVal']).reshape(-1, vggish_params.EMBEDDING_SIZE)
            if len(embedding_sound.shape) == 2:
                postprocessed_batch_keras = pproc.postprocess_single_sample(embedding_sound, num_secs)
                postprocessed_batch_keras = uint8_to_float32(postprocessed_batch_keras).reshape(num_secs, -1)
            else:
                postprocessed_batch_keras = pproc.postprocess(embedding_sound)
            result.append({'filename': filename, 'embedding':postprocessed_batch_keras, 'label':c})
        with open('./dataset/features_'+class_file+'.pickle', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
