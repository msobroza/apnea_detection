import numpy as np
import glob, os
from tqdm import tqdm


wav_path    = 'data/audio/snore_3s/'
np_path     = 'data/arrays/'

cmd         = 'python vggish/vggish_inference_demo.py --wav_file {} --numpy_file {} --checkpoint vggish/vggish_model.ckpt'
wavs        = glob.glob(wav_path + '*.wav')

for w in tqdm(wavs[0:5]):
    np_name = w.split('/')[-1].split('.')[0]
    _ = os.system(cmd.format(w, os.path.join(np_path, np_name)))