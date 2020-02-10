import os
#Define all the parameters for the input pipeline

#Base data directory
_BASE_DIR = 'data/'
DATA_URL = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'

RANDOM_SEED = 42
SAMPLE_RATE = 16000

#Where to store the TFRecord data
OUTPUT_DIR = os.path.join(_BASE_DIR, 'tfrecords')
#Raw audio files directory
RAW_DATA_DIR = os.path.join(_BASE_DIR, 'raw_data') 
#Silence category settings
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
#Percentage of the silent category
SILENT_SIZE = 5  

#Unknown category settings
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
#Percentage of the Unknown category
UNKNOW_SIZE = 11 

#where to find the background noise audios
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

#Percentage of the validation split
TEST_SIZE = 10
#Percentage of the test split
VAL_SIZE = 10
#How many TFRecords files will contain all the training set
NUM_SHARDS_TRAIN = 16 
#How many TFRecords files will contain all the testing set
NUM_SHARDS_TEST = 2   
#How many TFRecords files will contain all the validation set
NUM_SHARDS_VAL = 2    
# Words we want the system to recognize
TRAIN_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']