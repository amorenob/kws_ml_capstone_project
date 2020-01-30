
import urllib.request
import os
import re
import random
import math
import hashlib
import tarfile
import sys
from scipy import signal
from scipy.io import wavfile
import numpy as np
import tensorflow as tf
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

_BASE_DIR = 'data/'

_DEFAULT_META_CSV = os.path.join(_BASE_DIR, 'meta.csv')
OUTPUT_DIR = os.path.join(_BASE_DIR, 'tfrecords')

DATA_URL = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
RAW_DATA_DIR = os.path.join(_BASE_DIR, 'raw_data') 
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 42

SAMPLE_RATE = 16000

TEST_SIZE = 10
VAL_SIZE = 10
NUM_SHARDS_TRAIN = 16 
NUM_SHARDS_TEST = 2 
NUM_SHARDS_VAL = 2 


TRAIN_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']



def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result

def prepare_words_list(wanted_words):
    """Prepends common tokens to the custom word list.

    Args:
    wanted_words: List of strings containing the custom words.

    Returns:
    List with the standard silence and unknown tokens added.
    """
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def log_specgrama(audio, sample_rate, window_size=20,
                  step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def preprocess_mfcc(wave):

    spectrogram = librosa.feature.melspectrogram(wave, sr=SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    idx = [spectrogram > 0]
    spectrogram[idx] = np.log(spectrogram[idx])

    dct_filters = librosa.filters.dct(n_filters=40, n_input=40)
    mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]
    mfcc = np.hstack(mfcc)
    mfcc = mfcc.astype(np.float32)
    return mfcc


def preprocess_mel(data, n_mels=40, normalization=False):
    spectrogram = librosa.feature.melspectrogram(data, sr=SR, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    if normalization:
        spectrogram = spectrogram.spectrogram()
        spectrogram -= spectrogram
    return spectrogram


def download_data(source_url, dest_dir, extract=True):
    """Download files from a given url

    Code adapted from xxx example:
    http://xxxx 
    Source: https://github.com/xx_/input_data.py retrieved in January 2020.

    If the dest_dir does't exist it's created

    Args:
        source_url: Web location of the tar file containing the data.
        dest_dir: File path to save data to.
        extract: To extract the files or not
    """

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    file_name = source_url.split('/')[-1]
    file_path = os.path.join(dest_dir, file_name) 
    if not os.path.exists(file_path):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' %
                (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        try:
            file_path, _ = urllib.request.urlretrieve(source_url, file_path, _progress)

        except:
            print(f'Error downloading dataset')

        stat_info = os.stat(file_path)
        print(f'Successfully downloaded {file_name}, ({stat_info.st_size} bytes)')
        if extract:
            tarfile.open(file_path).extractall(dest_dir)


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class TFRecordsConverter:
    """Convert audio to TFRecords."""
    def __init__(self, data_dir, output_dir, silence_percentage, unknown_percentage,
                 wanted_words, validation_percentage, testing_percentage,  n_shards_val,
                 n_shards_test, n_shards_train):
        self.sample_rate=16000
        self.n_shards_train = n_shards_train
        self.n_shards_test = n_shards_test
        self.n_shards_val = n_shards_val

        self.output_dir = output_dir
        #create the directory to save the TFRecord files
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if data_dir:
            self.data_dir = data_dir
            self.prepare_data_index(silence_percentage, unknown_percentage,
                                    wanted_words, validation_percentage,
                                    testing_percentage)

    
    def prepare_data_index(self, silence_percentage, unknown_percentage,
                           wanted_words, validation_percentage,
                           testing_percentage):
        """Prepares a list of the samples organized by set and label.

        The training loop needs a list of all the available data, organized by
        which partition it should belong to, and with ground truth labels attached.
        This function analyzes the folders below the `data_dir`, figures out the
        right
        labels for each file based on the name of the subdirectory it belongs to,
        and uses a stable hash to assign it to a data set partition.

        Args:
        silence_percentage: How much of the resulting data should be background.
        unknown_percentage: How much should be audio outside the wanted classes.
        wanted_words: Labels of the classes we want to be able to recognize.
        validation_percentage: How much of the data set to use for validation.
        testing_percentage: How much of the data set to use for testing.

        Returns:
        Dictionary containing a list of file information for each set partition,
        and a lookup map for each class to determine its numeric index.

        Raises:
        Exception: If expected files are not found.
        """
        # Make sure the shuffling and picking of unknowns is deterministic.

        random.seed(RANDOM_SEED)
        wanted_words_index = {}
        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in gfile.Glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case, since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = which_set(wav_path, validation_percentage, testing_percentage)
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in wanted_words_index:
                self.data_index[set_index].append({'label': word, 'file': wav_path})
            else:
                unknown_index[set_index].append({'label': word, 'file': wav_path})
        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for index, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word +
                                ' in labels but only found ' +
                                ', '.join(all_words.keys()))
        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path
                })
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def _get_shard_path(self, split, shard_id, shard_size):
        return os.path.join(self.output_dir,
                            '{}-{:03d}-{}.tfrecord'.format(split, shard_id,
                                                           shard_size))

    def _write_tfrecord_file(self, split, shard_path, indices):
        """Write TFRecord file."""
        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            for index in indices:
                #file_path = self.df.file_path.iloc[index]
                #label = self.df.label.iloc[index]
                file_path = self.data_index[split][index]['file']
                label = self.word_to_index[self.data_index[split][index]['label']]

                raw_audio = tf.io.read_file(file_path)
                audio, sample_rate = tf.audio.decode_wav(
                    raw_audio,
                    desired_channels=1,  # mono
                    desired_samples=self.sample_rate )
                    #desired_samples=self.sample_rate * self.duration)

                # Example is a flexible message type that contains key-value
                # pairs, where each key maps to a Feature message. Here, each
                # Example contains two features: A FloatList for the decoded
                # audio data and an Int64List containing the corresponding
                # label's index.
                example = tf.train.Example(features=tf.train.Features(feature={
                    'audio': _float_feature(audio.numpy().flatten().tolist()),
                    'label': _int64_feature(label)}))

                out.write(example.SerializeToString())

    def convert(self):
            """Convert to TFRecords.
            Partition data into training, testing and validation sets. Then,
            divide each data set into the specified number of TFRecords shards.
            """
            splits = ('validation', 'testing', 'training')
            split_sizes = tuple(len(self.data_index[split]) for split in splits)
            #split_sizes = (self.n_train, self.n_test, self.n_val)
            self.n_val, self.n_test, self.n_train = split_sizes
            split_n_shards = (self.n_shards_val,  self.n_shards_test,
                              self.n_shards_train)

            #offset = 0
            for split, size, n_shards in zip(splits, split_sizes, split_n_shards):
                print('Converting {} set into TFRecord shards...'.format(split))
                shard_size = math.ceil(size / n_shards)
                offset = 0
                cumulative_size = offset + size
                for shard_id in range(1, n_shards + 1):
                    step_size = min(shard_size, cumulative_size - offset-1)
                    shard_path = self._get_shard_path(split, shard_id, step_size)
                    # Generate a subset of indices to select only a subset of
                    # audio-files/labels for the current shard.
                    file_indices = np.arange(offset, offset + step_size)
                    self._write_tfrecord_file(split, shard_path, file_indices)
                    offset += step_size

            print('Number of training examples: {}'.format(self.n_train))
            print('Number of testing examples: {}'.format(self.n_test))
            print('Number of validation examples: {}'.format(self.n_val))
            print('TFRecord files saved to {}'.format(self.output_dir))



def _parse_batch(record_batch, sample_rate, duration):
    n_samples = sample_rate * duration

    # Create a description of the features
    feature_description = {
        'audio': tf.io.FixedLenFeature([n_samples], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)

    return example['audio'], example['label']


def get_dataset_from_tfrecords(tfrecords_dir='tfrecords', split='train',
                               batch_size=64, sample_rate=22050, duration=4,
                               n_epochs=10):
    if split not in ('train', 'test', 'validate'):
        raise ValueError("split must be either 'train', 'test' or 'validate'")

    # List all *.tfrecord files for the selected split
    pattern = os.path.join(tfrecords_dir, '{}*.tfrecord'.format(split))
    files_ds = tf.data.Dataset.list_files(pattern)

    # Disregard data order in favor of reading speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    # Read TFRecord files in an interleaved order
    ds = tf.data.TFRecordDataset(files_ds,
                                 compression_type='ZLIB',
                                 num_parallel_reads=AUTOTUNE)
    # Prepare batches
    ds = ds.batch(batch_size)

    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration))

    # Repeat the training data for n_epochs. Don't repeat test/validate splits.
    if split == 'train':
        ds = ds.repeat(n_epochs)

    return ds.prefetch(buffer_size=AUTOTUNE)


if __name__ == '__main__':
    converter = TFRecordsConverter(RAW_DATA_DIR,
                       OUTPUT_DIR,
                       5,
                       11,
                       TRAIN_WORDS,
                       VAL_SIZE,
                       TEST_SIZE,
                       NUM_SHARDS_VAL,
                       NUM_SHARDS_TEST,
                       NUM_SHARDS_TRAIN
    )

    converter.convert()

print('ya')