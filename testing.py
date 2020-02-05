#from data_utils import *
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
from models import *
AUTOTUNE = tf.data.experimental.AUTOTUNE

def audio_to_melspectogram_3D(audio, label):
    sample_rate = 16000
    stfts = tf.signal.stft(
        audio,
        frame_length=480,
        frame_step=160,
        fft_length=None,
        window_fn=tf.signal.hann_window,
        pad_end=False,
        name=None
    )
    spectrograms = tf.abs(stfts)
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    print(num_spectrogram_bins)
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
    spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.

    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, -1)
    log_mel_spectrograms = tf.image.resize(log_mel_spectrograms, [98, 98])    # Resize the spectogram to match model input
    log_mel_spectrograms = tf.image.grayscale_to_rgb(log_mel_spectrograms)      # Transform image to 3 cahnels
    return log_mel_spectrograms, label

def audio_to_spectogram_1D(audio, label):
    stfts = tf.signal.stft(
        audio,
        frame_length=480,
        frame_step=160,
        fft_length=None,
        window_fn=tf.signal.hann_window,
        pad_end=False,
        name=None
    )
    spectrograms = tf.abs(stfts)
    #spectrograms = tf.expand_dims(spectrograms, -1)
    #spectrograms = tf.image.resize(spectrograms, [124, 124])    # Resize the spectogram to match model input
    return spectrograms, label

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
                                 num_parallel_reads=AUTOTUNE)
    
        # Shuffle during training
    if split == 'train':
        ds = ds.shuffle(2000)
    # Prepare batches
    ds = ds.batch(batch_size, drop_remainder=True)
    #print(ds)
    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration))
    #ds = ds.map(audio_to_spectogram)
    #print(ds)
    
    
    # Repeat the training data for n_epochs. Don't repeat test/validate splits.
    if split == 'train':
        ds = ds.repeat(n_epochs)

    return ds.prefetch(buffer_size=AUTOTUNE)

def preprocess_dataset(ds, preprocess_fc):

    ds = ds.map(ds, preprocess_fc)

def audio_to_2D(audio, label):
    audio_2d = tf.slice(audio,[0,0], [64, 15376])
    audio_2d = tf.reshape(audio_2d, [64, 124,124])
    audio_2d = tf.expand_dims(audio_2d, -1)
    audio_2d = tf.image.grayscale_to_rgb(audio_2d)
    return audio_2d, label        


def main():
    def get_datasets(batch_size, transformation_fc):
        """get train, validate and test datasets and perform the parsed transformation function"""
        datasets = {}
        splits = ('train', 'validate', 'test')
        for split in splits:
            ds = get_dataset_from_tfrecords(batch_size=batch_size, split=split)
            ds = ds.map(transformation_fc)    #Transform audio to spectogram
            datasets[split] = ds
        return datasets['train'], datasets['validate'], datasets['test']
    
    train_ds, validation_ds, test_ds = get_datasets(batch_size=64, 
                                                transformation_fc=audio_to_melspectogram_3D)
    print(train_ds,validation_ds, test_ds, sep='\n')

    model = simple_cnn(98, 98, 12)    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    #history = model.fit(train_ds, epochs=3, validation_data=validation_ds)
    #model.save_weights('data/checkpoints/my_checkpoint')

    # # Create a new model instance
    

    # # Restore the weights
    model.load_weights('data/checkpoints/my_checkpoint')
    #print(model.evaluate(test_ds, verbose=2))

    labels=[]
    audios = []
    for audio, label in train_ds.take(1):
        labels.append(label.numpy().flatten())
        audios.append(audio)
    labels=np.concatenate(labels)

    predictions = model.predict(audios)
    predictions = tf.argmax(predictions,1).numpy()
    print(tf.math.confusion_matrix(predictions, labels))
    pass

if __name__ == '__main__':
    #main()
    ds = get_dataset_from_tfrecords(batch_size=64, split='train')
    print(ds.take(1))
    ds = ds.map(audio_to_spectogram_1D) 
    print(ds.take(1))
    data_path = 'data/raw_data'
    # Choose one audio clip to work with
    bed_audio = os.path.join(data_path, 'bed', '00176480_nohash_0.wav')
    #readind the audio file...
    raw_audio = tf.io.read_file(bed_audio)
    # Let's check what we've got
    print(f'raw_audio {type(raw_audio)}')   
    print(f'raw_audio {raw_audio.numpy()[:20]}') # Print the head(20) of the raw audio
    # All audio files were recorded at 16000 samples / seconds
    sample_rate = 16000	
    audio, sample_rate = tf.audio.decode_wav(raw_audio,
                                            desired_channels=1,  # mono
                                            desired_samples=sample_rate )
    audio= tf.squeeze(audio)

    #audio = tf.expand_dims(audio, 0)
    print(f'audio {type(audio)}')   
    print(f'audio {audio.shape}') 
    print(f'audio {audio.numpy()[:10]}') # Print the head(10) of the decoded audio


    def audio_to_spectogram(audio):
        stfts = tf.signal.stft(
            audio,
            frame_length=480,
            frame_step=160,
            fft_length=None,
            window_fn=tf.signal.hann_window,
            pad_end=False,
            name=None
        )
        spectrogram = tf.abs(stfts)
        #spectrogram = tf.expand_dims(spectrogram, -1)             
        return spectrogram
    
    spectrogram = audio_to_spectogram(audio)
    print(f'audio {type(spectrogram)}')   
    print(f'audio {spectrogram.shape}') 
    print(f'audio {spectrogram.numpy()}') # Print the head(10) of the decoded audio