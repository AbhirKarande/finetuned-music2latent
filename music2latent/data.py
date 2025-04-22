import torch
from torch.utils.data import Dataset, DataLoader
import os
import soundfile as sf
import random
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

from .hparams import hparams

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


class AudioDataset(Dataset):
    def __init__(self, wav_paths, hop, fac, data_length, data_fractions, rms_min=0.001, data_extensions=['.wav', '.flac'], tot_samples=None, random_sampling=True):
        self.random_sampling = random_sampling
        if data_fractions is None:
            data_fractions = [1/len(wav_paths) for _ in wav_paths]
        tot_fractions = sum(data_fractions)
        data_fractions = [el/tot_fractions for el in data_fractions]
        self.tot_samples = tot_samples
        self.rms_min = rms_min
        self.paths = []
        self.num_samples = []
        self.num_tot_samples = []
        self.num_repetitions = []
        for path,fraction in zip(wav_paths,data_fractions):
            paths = find_files_with_extensions(path, extensions=data_extensions)
            seed_value = 42
            shuffling_random = random.Random(seed_value)
            shuffling_random.shuffle(paths)
            num_samples = len(paths)
            print(f'Found {num_samples} samples.')
            self.paths.append(paths)
            self.num_samples.append(num_samples)
            if tot_samples is None:
                self.num_tot_samples.append(int(num_samples))
            else:
                self.num_tot_samples.append(int(tot_samples))
            self.num_repetitions.append(self.num_tot_samples[-1]//num_samples)

        self.hop = hop
        self.data_length = data_length
        self.wv_length = hop * data_length + (fac-1)*hop
        self.data_fractions = torch.tensor(data_fractions)

    def __len__(self):
        return int(self.tot_samples)

    def __getitem__(self, idx):
        data_id = torch.multinomial(self.data_fractions, 1).item()
        if idx>(self.num_samples[data_id]*self.num_repetitions[data_id]):
            idx = torch.randint(self.num_samples[data_id], size=(1,)).item()
        else:
            idx = idx%self.num_samples[data_id]
        path = self.paths[data_id][idx]
        try:
            info = sf.info(path)
            samplerate = info.samplerate
            duration = info.duration
            length = int(samplerate*duration)
            rand_start = torch.randint(length-self.wv_length, size=(1,)).item()
            wv,_ = sf.read(path, frames=self.wv_length, start=rand_start, stop=None, dtype='float32', always_2d=True)
            wv = torch.from_numpy(wv)
            if wv.shape[-1]==1:
                wv = torch.cat([wv,wv], dim=1)
            wv = wv[:,:2]
            wv = wv.permute(1,0)

            # if not stereo:
            wv = wv[torch.randint(wv.shape[0], size=(1,)).item(),:]

            rms = torch.sqrt(torch.mean(wv**2))
            if rms < self.rms_min:
                idx = torch.randint(self.tot_samples, size=(1,)).item()
                return self.__getitem__(idx)

        except Exception as e:
            print(e)
            idx = torch.randint(self.tot_samples, size=(1,)).item()
            return self.__getitem__(idx)
        return wv


def get_contrastive_dataloader(batch_size_per_gpu):
    """
    Get a dataloader for contrastive learning with audio augmentations.
    """
    from datasets import load_dataset
    from .contrastive_audio_dataset import ContrastiveAudioDataset
    
    # Load the FMA dataset
    fma_dataset = load_dataset("ryanleeme17/free-music-archive-retrieval", split="train")
    
    # Create the contrastive dataset
    dataset = ContrastiveAudioDataset(fma_dataset, sample_rate=hparams.sample_rate)
    
    if hparams.multi_gpu:
        return DataLoader(dataset, batch_size=batch_size_per_gpu, drop_last=True, shuffle=False, 
                        sampler=DistributedSampler(dataset), num_workers=hparams.num_workers, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=batch_size_per_gpu, drop_last=True, shuffle=True, 
                        num_workers=hparams.num_workers, pin_memory=True)

def get_dataloader(batch_size_per_gpu):
    """
    Get the appropriate dataloader based on whether we're doing contrastive learning or not.
    """
    if hparams.use_contrastive:
        return get_contrastive_dataloader(batch_size_per_gpu)
    
    # Check if using Hugging Face dataset
    if any('datasets/' in path for path in hparams.data_paths):
        dataset = HuggingFaceAudioDataset(
            hparams.data_paths[0].replace('datasets/', ''),
            split='train',
            hop=hparams.hop,
            fac=1,
            data_length=hparams.data_length
        )
    else:
        dataset = AudioDataset(
            hparams.data_paths,
            hparams.hop,
            1,
            hparams.data_length,
            hparams.data_fractions,
            hparams.rms_min,
            hparams.data_extensions
        )
    
    if hparams.multi_gpu:
        sampler = DistributedSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            sampler=sampler,
            num_workers=hparams.num_workers,
            pin_memory=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=hparams.num_workers,
            pin_memory=True
        )


def get_test_dataset():
    # Check if using Hugging Face dataset
    if 'datasets/' in hparams.data_path_test:
        return HuggingFaceAudioDataset(
            hparams.data_path_test.replace('datasets/', ''),
            split='test',
            hop=hparams.hop,
            fac=1,
            data_length=hparams.data_length_test
        )
    else:
        return TestAudioDataset(
            hparams.data_path_test,
            hparams.hop,
            1,
            hparams.data_length_test
        )

class HuggingFaceAudioDataset(Dataset):
    def __init__(self, dataset_name, split='train', hop=None, fac=None, data_length=None, tot_samples=None, random_sampling=True):
        self.random_sampling = random_sampling
        self.dataset = load_dataset(dataset_name, split='train')  # Always load train split
        self.data_samples = len(self.dataset)
        print(f'Found {self.data_samples} samples.')
        
        # Split the data into train/test
        if split == 'test':
            # Use last 20% of data for testing
            test_size = int(0.2 * self.data_samples)
            self.start_idx = self.data_samples - test_size
            self.data_samples = test_size
        else:
            # Use first 80% of data for training
            train_size = int(0.8 * self.data_samples)
            self.start_idx = 0
            self.data_samples = train_size
            
        print(f'Using {self.data_samples} samples for {split} split.')
        
        self.hop = hop or hparams.hop
        if tot_samples is None:
            self.tot_samples = self.data_samples
        else:
            self.tot_samples = tot_samples
        self.num_repetitions = self.tot_samples//self.data_samples
        self.wv_length = self.hop * (data_length or hparams.data_length) + ((fac or 1)-1)*self.hop

    def __len__(self):
        return int(self.tot_samples)

    def __getitem__(self, idx):
        if idx > (self.data_samples*self.num_repetitions):
            idx = torch.randint(self.data_samples, size=(1,)).item()
        else:
            idx = idx % self.data_samples
            
        try:
            # Get audio from Hugging Face dataset with offset
            audio = self.dataset[self.start_idx + idx]['audio']
            wv = torch.from_numpy(audio['array']).float()
            
            # Ensure correct shape
            if len(wv.shape) == 1:
                wv = wv.unsqueeze(-1)
            
            # Convert to stereo if mono
            if wv.shape[-1] == 1:
                wv = torch.cat([wv, wv], dim=1)
            
            # Handle length mismatch
            if wv.shape[0] < self.wv_length:
                # Pad with zeros if too short
                pad_size = self.wv_length - wv.shape[0]
                wv = torch.nn.functional.pad(wv, (0, 0, 0, pad_size))
            elif wv.shape[0] > self.wv_length:
                # Randomly crop if too long
                start = torch.randint(0, wv.shape[0] - self.wv_length + 1, (1,)).item()
                wv = wv[start:start + self.wv_length]
                
            return wv
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            idx = torch.randint(self.tot_samples, size=(1,)).item()
            return self.__getitem__(idx)