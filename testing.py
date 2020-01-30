#from data_utils import *
import os
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
AUTOTUNE = tf.data.experimental.AUTOTUNE

AUTOTUNE = tf.data.experimental.AUTOTUNE


def audio_to_spectogram(audio, window_size=480, stride=160):
    spectrogram = audio_ops.audio_spectrogram(
          audio,
          window_size=window_size,
          stride=stride,
          magnitude_squared=True)
    expand_dims = tf.expand_dims(spectrogram, -1)
    resize = tf.image.resize(expand_dims, [224, 224])
    concate=tf.image.grayscale_to_rgb(resize)

    return concate


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


def get_dataset_from_tfrecords(tfrecords_dir='data/tfrecords', split='train',
                               batch_size=64, sample_rate=16000, duration=1,
                               n_epochs=2):
    if split not in ('train', 'test', 'validate'):
        raise ValueError("split must be either 'train', 'test' or 'validate'")

    # List all *.tfrecord files for the selected split
    split_match = {
        'train':'training', 
        'test':'testing', 
        'validate':'validation'
    }
    pattern = os.path.join(tfrecords_dir, '{}*.tfrecord'.format(split_match[split]))
    files_ds = tf.data.Dataset.list_files(pattern)

    # Disregard data order in favor of reading speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    # Read TFRecord files in an interleaved order
    ds = tf.data.TFRecordDataset(files_ds,
                                 compression_type='ZLIB',
                                 num_parallel_reads=2)
    

    # Prepare batches
    ds = ds.batch(batch_size)

    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration))
    ds = ds.map(lambda x,y: (audio_to_spectogram(x),y))

    # Repeat the training data for n_epochs. Don't repeat test/validate splits.
    if split == 'train':
        ds = ds.repeat(n_epochs)

    return ds.prefetch(buffer_size=AUTOTUNE)


def main():
    train_ds = get_dataset_from_tfrecords()
    
    model = tf.keras.applications.VGG16()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_ds, epochs=10)
#

if __name__ == '__main__':
    main()