# Configuration file for music2latent using Hugging Face FMAR dataset

# Paths to training data


# Path to test data (using the same dataset but different split)



# MAIN PARAMETERS
batch_size = 32                                                             # increased batch size
lr = 0.0002                                                                 # slightly higher learning rate
total_iters = 400000                                                        # reduced total iterations
iters_per_epoch = 5000                                                      # reduced iterations per epoch
compile_model = True                                                        # keep compilation enabled
num_workers = 16                                                            # keep current workers
multi_gpu = True                                                            # enable multi-gpu training

data_paths = [
    "datasets/ryanleeme17/free-music-archive-retrieval"  # Hugging Face dataset path
]
data_fractions = None                                                       # list of sampling weights of each dataset (if None, equal sampling weights)
data_path_test = "datasets/ryanleeme17/free-music-archive-retrieval"  # Hugging Face dataset path 
data_extensions = ['.wav', '.flac']                                         # list of extensions of audio files to search for in the given paths

num_samples_fad = 500                                                       # number of samples that are encoded and decoded for FAD evaluation








# TRAINING
lr_decay = 'cosine'                                                         # learning rate schedule ['cosine', 'linear', 'inverse_sqrt]    
start_decay_iteration = 0                                                   # start decaying learning rate from this iteration
final_lr = 0.000001                                                         # if exponential_lr_decay=True, this is the learning rate after total_iters
warmup_steps = iters_per_epoch                                              # number of warmup steps of optimizer
accumulate_gradients = 1                                                    # will accumulate the gradients from this number of batches befire updating
checkpoint_path = 'checkpoints'                                             # path where to save config and checkpoints
torch_compile_cache_dir = 'tmp/torch_compile'                               # path where to save compiled kernels
mixed_precision = True                                                      # use mixed precision (float16)
seed = 42                                                                   # seed for Pytorch and Numpy

load_path = None                                                            # load checkpoint from this path 
load_iter = True                                                            # if False, reset the scheduler and start from iteration 0
load_ema = True                                                             # if False, do not load the EMA weights from checkpoint
load_optimizer = True                                                       # if False, do not load the optimizer parameters from checkpoint (helps in case of resuming collapsed run)
optimizer_beta1 = 0.9
optimizer_beta2 = 0.999


# EXPONENTIAL MOVING AVERAGE
enable_ema = True                                                           # track exponential moving averages for better inference model
ema_momentum = 0.9999                                                       # exponential moving average momentum parameter
warmup_ema = True                                                           # use warmup for exponential moving average


# DATA
rms_min = 0.001                                                             # minimum RMS value for audio samples used for training

data_channels = 2                                                           # channels of input data (real-imaginary STFT requires 2)
data_length = 64                                                            # sequence length of input spectrogram
data_length_test = 1024//4                                                  # sequence length of spectrograms used for testing
sample_rate = 44100                                                         # sampling rate used to render audio samples (does not matter for training)

hop = 128*4                                                                 # hop size of STFT

alpha_rescale = 0.65                                                        # alpha rescale parameter for STFT representation
beta_rescale = 0.34                                                         # beta rescale parameter for STFT representation
sigma_data = 0.5                                                            # sigma data for EDM framework                      


# EVALUATION
eval_samples_path = 'eval_samples'                                          # generated images for FAD evaluation during training will be saved here

inference_diffusion_steps = 1                                               # how many denoising steps to use for FAD calculation

fad_models = ['vggish', 'clap']                                             # list of FAD models to use
fad_workers = 16                                                            # number of workers for FAD evaluation
fad_background_embeddings = [f'fad_stats/{data_path_test.replace("/", "")}_{fm}.npy' for fm in fad_models]               # name of fad embeddings file. If does not exist, it will be created on the first run


# MODEL
base_channels = 48                                                          # reduced base channels
layers_list = [2,2,2,2]                                                    # reduced number of layers
multipliers_list = [1,2,4,4]                                               # reduced multipliers
attention_list = [0,0,1,1]                                                 # adjusted attention layers
freq_downsample_list = [1,0,0]                                             # adjusted downsampling

layers_list_encoder = [1,1,1,1]                                            # reduced encoder layers
attention_list_encoder = [0,0,1,1]                                         # adjusted encoder attention
bottleneck_base_channels = 384                                              # reduced bottleneck channels
num_bottleneck_layers = 4                                                   # number of blocks to use before/after bottleneck
frequency_scaling = True                                                    # use frequency scaling

heads = 4                                                                   # number of attention heads
cond_channels = 256                                                         # dimension of time embedding
use_fourier = False                                                         # if True, use random Fourier embedding, if False, use Positional
fourier_scale = 0.2                                                         # scale parameter for gaussian fourier layer (original is 0.02, but to me it appears too small)
normalization = True                                                        # use group normalization
dropout_rate = 0.                                                           # dropout rate
min_res_dropout = 16                                                        # dropout is applied on equal or smaller feature map resolutions
init_as_zero = True                                                         # initialize convolution kernels before skip connections with zeros

bottleneck_channels = 64                                                    # channels of encoder bottleneck

pre_normalize_2d_to_1d = True                                               # pre-normalize 2D to 1D connection in encoder
pre_normalize_downsampling_encoder = True                                   # pre-normalize downsampling layers in encoder


# DIFFUSION PARAMETERS
schedule = 'exponential'                                                    # step schedule to use ['constant', 'exponential']

start_exp = 1.                                                              # if schedule is 'exponential', the starting exponent
end_exp = 3.                                                                # the higher the exponent, the smaller the steps at the end of training or throughout training if schedule is 'constant'
base_step = 0.1                                                             # the base step on which the exponent is applied

sigma_min = 0.002                                                           # minimum sigma for EDM framework
sigma_max = 80.                                                             # maximum sigma for EDM framework

rho = 7.                                                                    # rho parameter for EDM framework

use_lognormal = True                                                        # use a lognormal noise schedule during training
p_mean = -1.1                                                               # mean of lognormal noise schedule
p_std = 2.                                                                  # standard deviation of lognormal noise schedule

# LSTM Configuration
use_lstm = True                                                             # whether to use LSTM for temporal aggregation
lstm_hidden_size = 256                                                     # hidden size of LSTM
lstm_num_layers = 2                                                        # number of LSTM layers
lstm_dropout = 0.1                                                         # dropout rate for LSTM
lstm_bidirectional = True                                                  # whether to use bidirectional LSTM

# Contrastive Learning Configuration
use_contrastive = True                                                     # whether to use contrastive learning
contrastive_loss_weight = 0.1                                             # weight for the contrastive loss term
contrastive_temperature = 0.07                                            # temperature parameter for contrastive loss
