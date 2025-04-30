import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import soundfile as sf
import random
import librosa
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset # Rename to avoid conflict
import numpy as np

from hparams import hparams
from contrastive_audio_dataset import ContrastiveAudioDataset # Import contrastive dataset

from torch.utils.data.distributed import DistributedSampler

class TestAudioDataset(Dataset):
    def __init__(self, wav_path, hop, fac, data_length, tot_samples=None, random_sampling=True):
        self.random_sampling = random_sampling
        self.paths = find_files_with_extensions(wav_path, extensions=['.wav', '.flac'])
        # sort paths
        self.paths = sorted(self.paths)
        seed_value = 42
        shuffling_random = random.Random(seed_value)
        shuffling_random.shuffle(self.paths)
        self.data_samples = len(self.paths)
        print(f'Found {self.data_samples} samples.')
        self.hop = hop
        if tot_samples is None:
            self.tot_samples = self.data_samples
        else:
            self.tot_samples = tot_samples
        self.num_repetitions = self.tot_samples//self.data_samples
        self.wv_length = hop * data_length + (fac-1)*hop

    def __len__(self):
        return int(self.tot_samples)

    def __getitem__(self, idx):
        if idx>(self.data_samples*self.num_repetitions):
            idx = torch.randint(self.data_samples, size=(1,)).item()
        else:
            idx = idx%self.data_samples
        path = self.paths[idx]
        try:
            wv,_ = sf.read(path, dtype='float32', always_2d=True)
            if wv.shape[0]<self.wv_length:
                idx = torch.randint(self.tot_samples, size=(1,)).item()
                return self.__getitem__(idx)
            wv = torch.from_numpy(wv)
            # convert to mono
            if wv.shape[-1]>1:
                wv = wv.mean(dim=-1, keepdim=True)
            if wv.shape[-1]==1:
                wv = torch.cat([wv,wv], dim=1)
            wv = wv[:,:2]
            wv = wv.permute(1,0)
            # if not stereo:
            wv = wv[torch.randint(wv.shape[0], size=(1,)).item(),:]
        except Exception as e:
            print(e)
            idx = torch.randint(self.tot_samples, size=(1,)).item()
            return self.__getitem__(idx)
        return wv
    

def find_files_with_extensions(path, extensions=['.wav', '.flac']):
    found_files = []
    # Recursively traverse the directory
    for foldername, subfolders, filenames in tqdm(os.walk(path)):
        for filename in filenames:
            # Check if the file has an extension from the specified list
            if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                # Build the full path to the file
                file_path = os.path.join(foldername, filename)
                found_files.append(file_path)

    return found_files


def get_train_val_datasets():
    """
    Loads the Hugging Face dataset specified in hparams and splits it into 
    training and validation sets based on hparams.validation_split_ratio.
    Returns two ContrastiveAudioDataset instances.
    """
    print(f"Loading dataset: {hparams.hf_dataset_name}, split: {hparams.hf_dataset_split}")
    # Load the full dataset
    full_dataset = load_dataset(hparams.hf_dataset_name, split=hparams.hf_dataset_split)
    
    # Split the dataset
    # Ensure reproducibility with a fixed seed if desired
    split_datasets = full_dataset.train_test_split(
        test_size=hparams.validation_split_ratio,
        shuffle=True, 
        seed=hparams.seed # Use the global seed for reproducibility
    )
    
    train_hf_dataset = split_datasets['train']
    val_hf_dataset = split_datasets['test']
    
    print(f"Train dataset size: {len(train_hf_dataset)}")
    print(f"Validation dataset size: {len(val_hf_dataset)}")
    
    # Create ContrastiveAudioDataset for train and validation
    # We assume ContrastiveAudioDataset handles the audio processing and augmentation
    train_dataset = ContrastiveAudioDataset(train_hf_dataset, sample_rate=hparams.sample_rate)
    val_dataset = ContrastiveAudioDataset(val_hf_dataset, sample_rate=hparams.sample_rate) # Note: usually no heavy augmentation for validation
    
    # Optional: Modify ContrastiveAudioDataset or create a simpler version for validation 
    # if you don't want augmentations during validation.
    # For now, we use the same class but the augmentations might not be ideal for eval.
    
    return train_dataset, val_dataset

def get_dataloader(dataset, batch_size_per_gpu, shuffle=True):
    """
    Generic function to create a DataLoader for a given dataset.
    Handles distributed training if hparams.multi_gpu is True.
    """
    if hparams.multi_gpu:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            sampler=sampler,
            num_workers=hparams.num_workers,
            pin_memory=True,
            drop_last=True # Often desired for training
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=shuffle,
            num_workers=hparams.num_workers,
            pin_memory=True,
            drop_last=True
        )

def get_test_dataset():
    # Check if using Hugging Face dataset for testing
    # This might need adjustment depending on how testing/evaluation is done.
    # For now, assume it uses a separate test path or a specific HF split.
    if 'datasets/' in hparams.data_path_test: 
        # This part might need review - is data_path_test set correctly?
        # Does it point to a dataset suitable for testing (e.g., original audio only)?
        # Let's assume for now it loads *some* dataset for testing.
        test_hf_dataset = load_dataset(hparams.data_path_test.replace('datasets/', ''), split='test') # Example split
        
        # We need a simple dataset for testing, not necessarily contrastive.
        # Let's create a simple wrapper if needed, or adapt TestAudioDataset.
        # For simplicity, maybe reuse TestAudioDataset logic but adapted for HF.
        # This part requires clarification on the *exact* format needed for testing.
        print(f"Warning: HuggingFace test dataset loading needs verification.")
        # Placeholder: return a basic dataset for now
        return BasicAudioDataset(test_hf_dataset, hparams.hop, 1, hparams.data_length_test) 

    else:
        # Original local file testing
        return TestAudioDataset(
            hparams.data_path_test,
            hparams.hop,
            1,
            hparams.data_length_test
        )

class BasicAudioDataset(Dataset):
    def __init__(self, hf_dataset, hop, fac, data_length):
        self.dataset = hf_dataset
        self.hop = hop
        self.wv_length = hop * data_length + (fac-1)*hop
        self.sample_rate = hparams.sample_rate # Assuming target sample rate from hparams
        print(f"BasicAudioDataset created with {len(hf_dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            sample = self.dataset[idx]
            audio_data = sample["audio"]["array"]
            orig_sr = sample["audio"]["sampling_rate"]

            # Resample if necessary
            if orig_sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=self.sample_rate)

            # Ensure correct length (pad or crop)
            if len(audio_data) < self.wv_length:
                audio_data = np.pad(audio_data, (0, self.wv_length - len(audio_data)))
            elif len(audio_data) > self.wv_length:
                start = random.randint(0, len(audio_data) - self.wv_length)
                audio_data = audio_data[start:start + self.wv_length]

            # Convert to torch tensor
            wv = torch.from_numpy(audio_data).float()
            
            # Ensure mono (take first channel if stereo)
            # Note: this differs from previous logic which averaged or duplicated.
            # Choose based on what the model expects.
            if wv.dim() > 1 and wv.shape[-1] > 1:
                wv = wv[..., 0] # Take first channel
            if wv.dim() == 1:
                 wv = wv.unsqueeze(0) # Add channel dim if mono: [length] -> [1, length]

            return wv.squeeze() # Return tensor, remove channel dim if added

        except Exception as e:
            print(f"Error processing sample {idx} in BasicAudioDataset: {e}")
            # Return zeros on error
            return torch.zeros(self.wv_length)