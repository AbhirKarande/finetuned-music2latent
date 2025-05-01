import datetime
import os
import shutil
import glob
import torch
import torch.nn as nn # Added for parameter freezing
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import torch.nn.functional as F

import fad_utils
import misc
from ema import ExponentialMovingAverage
from hparams import hparams
from utils import *
from models import * # UNet, Encoder, Decoder are here
from data import get_train_val_datasets, get_dataloader, get_test_dataset # Updated imports
from audio import *
from config_loader import load_config

if hparams.torch_compile_cache_dir is not None:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = hparams.torch_compile_cache_dir

torch.backends.cudnn.benchmark = True

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group

# Helper function for InfoNCE Loss
def info_nce_loss(features, temperature):
    batch_size = features.shape[0] // 2 # features contains N originals + N transforms
    labels = torch.arange(batch_size).to(features.device)
    
    # Normalize features
    features = F.normalize(features, dim=1)
    
    # Compute similarity matrix (N originals x N transforms)
    similarity_matrix = torch.matmul(features[:batch_size], features[batch_size:].T) / temperature
    
    # Compute loss (original -> transformed and transformed -> original)
    loss_orig_to_transformed = F.cross_entropy(similarity_matrix, labels)
    loss_transformed_to_orig = F.cross_entropy(similarity_matrix.T, labels)
    
    return (loss_orig_to_transformed + loss_transformed_to_orig) / 2


class Trainer:
    def __init__(self):
        if hparams.multi_gpu:
            torch.multiprocessing.set_start_method('spawn')
            misc.init()
        np.random.seed((hparams.seed * misc.get_world_size() + misc.get_rank()) % (1 << 31))
        torch.manual_seed(np.random.randint(1 << 31))

        # Set device and dtype
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            # Force float32 for MPS
            torch.set_default_dtype(torch.float32)
            print("Using MPS device with float32")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")

        batch_size_per_gpu = hparams.batch_size // misc.get_world_size()

        # --- Load Data --- 
        if hparams.use_contrastive:
            print("Setting up contrastive dataloaders...")
            train_dataset, val_dataset = get_train_val_datasets()
            self.train_dl = get_dataloader(train_dataset, batch_size_per_gpu, shuffle=True)
            self.val_dl = get_dataloader(val_dataset, batch_size_per_gpu, shuffle=False)
            print(f"Train Dataloader size: {len(self.train_dl)}")
            print(f"Validation Dataloader size: {len(self.val_dl)}")
        else:
            # Fallback or alternative non-contrastive setup (if needed)
            # This part might need adjustment based on non-contrastive use cases
            print("Setting up standard dataloaders (Non-contrastive path needs review)")
            # placeholder: load some default dataset if not contrastive
            # self.train_dl = get_dataloader(...) 
            # self.val_dl = None # Or setup a standard validation set
            raise NotImplementedError("Non-contrastive training path is not fully implemented yet.")
        
        if misc.get_rank()==0:
            # Use the standard test dataset for evaluations like FAD
            self.ds_test = get_test_dataset()
        
        self.get_models() # Setup models, optimizer, EMA etc.
        
        self.switch_save_checkpoint = True # Controls saving best checkpoint
        self.step = self.it # Use loaded iteration count if available
        self.best_val_loss = float('inf') # Track best validation loss
        
        # INITIALIZE CHECKPOINT FOLDER
        if misc.get_rank()==0:
            if self.save_path is None: # Only create if not loading a checkpoint
                timestamp = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-").split(".")[0]
                self.save_path = f'{hparams.checkpoint_path}/{timestamp}_contrastive'
                os.makedirs(self.save_path, exist_ok=True)
                os.makedirs(os.path.join(self.save_path, 'code'), exist_ok=True)
                # Save code snapshot
                code_dir = os.path.dirname(__file__)
                for file in glob.glob(os.path.join(code_dir, '*.py')):
                    try:
                        shutil.copyfile(file, os.path.join(self.save_path, 'code', os.path.basename(file)))
                    except shutil.SameFileError:
                        pass # Ignore if source and destination are the same
                # Save hparams
                hparams_path = os.path.join(code_dir, 'hparams.py') # Assuming hparams is in the same dir
                if os.path.exists(hparams_path):
                     shutil.copyfile(hparams_path, os.path.join(self.save_path, 'code', 'hparams.py'))
            
            self.writer = SummaryWriter(log_dir=self.save_path)


    # @torch.compile(mode='max-autotune-no-cudagraphs', disable=not hparams.compile_model)
    # def forward_pass_consistency(self, data_encoder, noisy_samples, noisy_samples_plus_one, sigmas_step, sigmas):
    #     # THIS FUNCTION IS FOR THE ORIGINAL CONSISTENCY TRAINING - NOT USED IN CONTRASTIVE FINETUNING
    #     with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hparams.mixed_precision):
    #         if hparams.multi_gpu:
    #             fdata, fdata_plus_one = self.ddp(data_encoder, noisy_samples, noisy_samples_plus_one, sigmas_step, sigmas)
    #         else:
    #             fdata, fdata_plus_one = self.gen(data_encoder, noisy_samples, noisy_samples_plus_one, sigmas_step, sigmas)
            
    #         loss_weight = get_loss_weight(sigmas, sigmas_step)
    #         loss = huber(fdata,fdata_plus_one,loss_weight)
    #     return loss


    def train_it(self, batch):
        """
        Training iteration for contrastive learning.
        """
        if not hparams.use_contrastive:
            raise ValueError("train_it called but use_contrastive is False")
            
        # Ensure float32 for MPS compatibility
        original = batch["original"].to(torch.float32).to(self.device, non_blocking=True)
        transformed = batch["transformed"].to(torch.float32).to(self.device, non_blocking=True)
        
        if original.dim() == 2:
             original = original.unsqueeze(1)
             transformed = transformed.unsqueeze(1)
             
        inputs = torch.cat([original, transformed], dim=0)
        data_encoder_repr = to_representation_encoder(inputs)
        
        # Fix tensor dimensions if needed
        if data_encoder_repr.dim() == 5:
            # If we get [B, 1, 2, H, W], reshape to [B, 2, H, W]
            data_encoder_repr = data_encoder_repr.squeeze(1)

        # Only use autocast for CUDA, not for MPS
        if self.device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hparams.mixed_precision):
                embeddings = self.model(data_encoder_repr)
                pooled_embeddings = embeddings.mean(dim=2)
                contrastive_loss = info_nce_loss(pooled_embeddings, hparams.contrastive_temperature)
                loss = contrastive_loss
        else:
            # No autocast for MPS
            embeddings = self.model(data_encoder_repr)
            pooled_embeddings = embeddings.mean(dim=2)
            contrastive_loss = info_nce_loss(pooled_embeddings, hparams.contrastive_temperature)
            loss = contrastive_loss
        
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if hparams.enable_ema:
            self.ema.update()
        
        return loss.item()

    def validate_epoch(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        
        if misc.get_rank() == 0:
            pbar = tqdm(self.val_dl, desc=f'Validation Epoch {self.epoch}', leave=False)
        else:
            pbar = self.val_dl
            
        for batch in pbar:
            # Ensure float32 for MPS compatibility
            original = batch["original"].to(torch.float32).to(self.device, non_blocking=True)
            transformed = batch["transformed"].to(torch.float32).to(self.device, non_blocking=True)
            
            if original.dim() == 2:
                 original = original.unsqueeze(1)
                 transformed = transformed.unsqueeze(1)
                 
            inputs = torch.cat([original, transformed], dim=0)
            data_encoder_repr = to_representation_encoder(inputs)
            
            # Fix tensor dimensions if needed
            if data_encoder_repr.dim() == 5:
                data_encoder_repr = data_encoder_repr.squeeze(1)
            
            # Only use autocast for CUDA, not for MPS
            if self.device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hparams.mixed_precision):
                    embeddings = self.model(data_encoder_repr)
                    pooled_embeddings = embeddings.mean(dim=2)
                    contrastive_loss = info_nce_loss(pooled_embeddings, hparams.contrastive_temperature)
            else:
                embeddings = self.model(data_encoder_repr)
                pooled_embeddings = embeddings.mean(dim=2)
                contrastive_loss = info_nce_loss(pooled_embeddings, hparams.contrastive_temperature)
            
            total_val_loss += contrastive_loss.item()
            num_val_batches += 1
            
            if misc.get_rank() == 0:
                 pbar.set_postfix({'val_loss': contrastive_loss.item()})

        self.model.train()
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
        
        if hparams.multi_gpu:
            avg_val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
            torch.distributed.all_reduce(avg_val_loss_tensor, op=torch.distributed.ReduceOp.AVG)
            avg_val_loss = avg_val_loss_tensor.item()
            
        return avg_val_loss

    def train(self):
        self.model.train() # Ensure model is in training mode initially
        
        while self.it < hparams.total_iters:
            if hparams.multi_gpu:
                # Set epoch for distributed sampler shuffle
                self.train_dl.sampler.set_epoch(self.epoch)
                
            train_loss_list = []

            if misc.get_rank()==0:
                pbar = tqdm(self.train_dl, desc=f'Epoch {self.epoch} | Iter {self.it}/{hparams.total_iters}', leave=True)
            else:
                pbar = self.train_dl

            for batchi, batch in enumerate(pbar):
                if self.it >= hparams.total_iters:
                    break # Stop if max iterations reached mid-epoch
                    
                current_lr = self.update_learning_rate() # Update LR before training step
                
                loss_val = self.train_it(batch)
                train_loss_list.append(loss_val)

                self.it += 1 # Increment iteration counter
                self.step = self.it # Keep step aligned with it for contrastive setup

                # Logging (Rank 0 only)
                if batchi % 100 == 0 and misc.get_rank() == 0:
                    avg_loss = np.mean(train_loss_list[-100:]) # Avg over last 100 steps
                    pbar.set_postfix({'loss': avg_loss, 'lr': current_lr})
                    self.writer.add_scalar('Loss/train_step', loss_val, self.it)
                    self.writer.add_scalar('Loss/train_avg_100', avg_loss, self.it)
                    self.writer.add_scalar('Meta/learning_rate', current_lr, self.it)
            
            # --- End of Epoch --- 
            if misc.get_rank()==0:
                avg_epoch_train_loss = np.mean(train_loss_list) if train_loss_list else 0
                print(f"Epoch {self.epoch} finished. Average Training Loss: {avg_epoch_train_loss:.4f}")
                self.writer.add_scalar('Loss/train_epoch', avg_epoch_train_loss, self.epoch)
                self.writer.add_scalar('Meta/epoch', self.epoch, self.it) # Log epoch vs global step
                
                # --- Validation --- 
                avg_val_loss = self.validate_epoch()
                print(f"Epoch {self.epoch} Validation Loss: {avg_val_loss:.4f}")
                self.writer.add_scalar('Loss/validation', avg_val_loss, self.epoch)
                
                # --- Checkpointing --- 
                is_best = avg_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_val_loss
                    self.switch_save_checkpoint = True # Mark to save this as best
                    print(f"New best validation loss: {self.best_val_loss:.4f}. Saving checkpoint.")
                else:
                    self.switch_save_checkpoint = False
                    
                # Save checkpoint (potentially best or latest)
                # Note: FAD calculation is removed/commented as it uses the generator
                # self.calculate_fad(hparams.inference_diffusion_steps) # Requires UNet/generator
                self.save_checkpoint(avg_epoch_train_loss, avg_val_loss) # Pass losses for filename
                
                # --- Optional: Generate test samples (if Encoder can be used for this) ---
                # Requires adapting test_model or similar function for encoder output
                # if hparams.enable_ema:
                #     with self.ema.average_parameters():
                #         self.test_model_encoder() # Needs implementation
                # else:
                #     self.test_model_encoder() # Needs implementation

            self.epoch = self.epoch + 1
            # Ensure all processes sync before next epoch if using DDP
            if hparams.multi_gpu:
                torch.distributed.barrier()
                
        # --- End of Training --- 
        if misc.get_rank()==0:
            print("Training finished.")
            self.writer.close()
        # Cleanup DDP
        if hparams.multi_gpu:
            destroy_process_group()

    def update_learning_rate(self):
        if self.it < hparams.warmup_steps:
            lr = hparams.lr * (self.it / hparams.warmup_steps)
        else:
            if hparams.lr_decay == 'cosine':
                decay_iters = hparams.total_iters - hparams.warmup_steps
                current_iter = (self.it - hparams.warmup_steps) % decay_iters
                lr = hparams.final_lr + (0.5 * (hparams.lr - hparams.final_lr) * (1. + np.cos((current_iter / decay_iters) * np.pi)))
            elif hparams.lr_decay == 'linear':
                decay_iters = hparams.total_iters - hparams.warmup_steps
                current_iter = (self.it - hparams.warmup_steps) % decay_iters
                lr = hparams.lr - ((hparams.lr - hparams.final_lr) * (current_iter / decay_iters))
            elif hparams.lr_decay == 'inverse_sqrt':
                lr = hparams.lr * (hparams.warmup_steps ** 0.5) / max(self.it, hparams.warmup_steps) ** 0.5
            elif hparams.lr_decay is None:
                lr = hparams.lr
            else:
                raise ValueError('lr_decay must be None, "cosine", "linear", or "inverse_sqrt"')
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr # Return current LR for logging

    def test_model(self):
        # THIS FUNCTION IS FOR THE ORIGINAL CONSISTENCY MODEL (GENERATOR/UNET)
        # It needs to be adapted or replaced if we want to visualize/test the ENCODER output.
        # For now, it remains unchanged but likely won't work correctly in contrastive mode.
        print("Warning: test_model function is designed for the UNet/Generator, not the Encoder.")
        if not hasattr(self, 'gen') or not isinstance(self.gen, UNet):
             print("Skipping test_model as self.gen is not a UNet instance.")
             return
             
        self.gen.eval() # Assuming self.gen exists and is the UNet
        max_steps = hparams.inference_diffusion_steps
        # ... (rest of the original function)
        # Needs access to self.ds_test and potentially self.gen (the UNet)
        num_examples = 4
        try:
            original,reconstructed = encode_decode(self.gen, self.ds_test, num_examples)
            if len(original[0].shape)==2:
                original = [el[0,:] for el in original]
            if len(reconstructed[0].shape)==2:
                reconstructed = [el[0,:] for el in reconstructed]
            fig = plot_audio_compare(original,reconstructed)
            fig.suptitle(f'{max_steps} steps')
            if self.writer is not None:
                for ind in range(num_examples):
                    self.writer.add_audio(f"original_{ind}", original[ind].detach().cpu().squeeze().numpy(), global_step=self.it, sample_rate=hparams.sample_rate)
                    self.writer.add_audio(f"reconstructed_{ind}", reconstructed[ind].detach().cpu().squeeze().numpy(), global_step=self.it, sample_rate=hparams.sample_rate)
                self.writer.add_figure(f"figs/{max_steps}_steps", fig, global_step=self.it)
            plt.close()
        except Exception as e:
            print(f"Error during test_model execution: {e}")
        self.gen.train()

    def save_batch_to_wav(self, batch):
        # THIS FUNCTION IS FOR THE ORIGINAL CONSISTENCY MODEL (GENERATOR/UNET)
        # It saves the *output* of the generator. Not directly applicable to encoder embeddings.
        print("Warning: save_batch_to_wav is designed for UNet output, not Encoder embeddings.")
        # ... (rest of the original function - may need adaptation or removal)
        print('Saving audio samples...')
        self.final_fad_path = os.path.join(self.save_path, hparams.eval_samples_path)
        os.makedirs(self.final_fad_path, exist_ok=True)
        for i in range(len(batch)):
            # Assuming batch contains audio waveforms
            audio_data = batch[i].cpu().numpy() # Ensure on CPU
            if len(audio_data.shape)==2:
                audio_data = audio_data[0]
            audio_data = np.clip(audio_data, -1.0, 1.0) # Clip values
            audio_data = (audio_data * 32767.0).astype(np.int16)  # Scale to 16-bit PCM range
            audio_file_path = os.path.join(self.final_fad_path, f'audio_{i}.wav')
            # Save the audio file
            try:
                write(audio_file_path, hparams.sample_rate, audio_data)
            except Exception as e:
                 print(f"Error saving wav file {audio_file_path}: {e}")
            
    def calculate_fad(self, diffusion_steps=1, log=True):
        # THIS FUNCTION IS FOR THE ORIGINAL CONSISTENCY MODEL (GENERATOR/UNET)
        # It calculates FAD based on generated audio samples.
        # Not directly applicable when only training the encoder.
        print("Warning: calculate_fad requires the UNet/Generator. Skipping FAD calculation.")
        self.current_score = float('inf') # Set dummy score if only training encoder
        # ... (Original logic requires self.gen (UNet) and encode_decode_batch)
        # if hparams.enable_ema:
        #     with self.ema.average_parameters():
        #         self.gen.eval()
        #         samples = encode_decode_batch(self.gen, self.ds_test, hparams.num_samples_fad, diffusion_steps=diffusion_steps)
        #         self.gen.train()
        # else:
        #     self.gen.eval()
        #     samples = encode_decode_batch(self.gen, self.ds_test, hparams.num_samples_fad, diffusion_steps=diffusion_steps)
        #     self.gen.train()
        # self.save_batch_to_wav(samples)
        # score = fad_utils.compute_fad(self.final_fad_path)
        # print(f'FAD: {score}')
        # if log:
        #     for i in range(len(hparams.fad_models)):
        #         self.writer.add_scalar(f'fad_{hparams.fad_models[i]}', score[i], self.it)
        # score = score[0]
        # self.current_score = score
            
    def save_checkpoint(self, train_loss, val_loss):
        """Saves checkpoint, focusing on encoder state if training encoder only."""
        if misc.get_rank() != 0:
            return # Only save on rank 0
            
        print(f"Attempting to save checkpoint for epoch {self.epoch}, iter {self.it}...")
        
        # Define state dict
        save_dict = {
            'it': self.it,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'hparams': hparams.to_dict(), # Save hparams used for this run
        }
        
        # --- Decide what model state to save --- 
        if hparams.train_encoder_only:
            # Save only the encoder state
            model_to_save = self.model.module if hparams.multi_gpu else self.model
            save_dict['encoder_state_dict'] = model_to_save.state_dict()
            if hparams.enable_ema:
                # Create a temporary encoder instance to store EMA weights
                temp_encoder = Encoder().to(self.device)
                self.ema.copy_to(temp_encoder.parameters()) # Apply EMA to temp encoder
                save_dict['encoder_ema_state_dict'] = temp_encoder.state_dict()
                del temp_encoder
        else:
            # Save the full UNet state (as in original script)
            model_to_save = self.model.module if hparams.multi_gpu else self.model # self.model is the UNet here
            save_dict['gen_state_dict'] = model_to_save.state_dict()
            if hparams.enable_ema:
                 # EMA is applied to the full UNet parameters
                 save_dict['ema_state_dict'] = self.ema.state_dict()
                 
        # --- Add optimizer and scaler state (optional but recommended for resuming) --- 
        save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        save_dict['scaler_state_dict'] = self.scaler.state_dict()
        
        # --- Filename --- 
        # Use val_loss for checkpoint naming convention
        loss_str = f"{val_loss:.4f}".replace('.', '')
        filename_latest = f'checkpoint_latest_ep{self.epoch}_it{self.it}_valloss{loss_str}.pt'
        filepath_latest = os.path.join(self.save_path, filename_latest)
        
        # --- Save Latest Checkpoint --- 
        try:
            torch.save(save_dict, filepath_latest)
            print(f"Saved latest checkpoint to {filepath_latest}")
        except Exception as e:
            print(f"Error saving latest checkpoint: {e}")
            return # Don't proceed if save fails
            
        # --- Save Best Checkpoint (if current is best) --- 
        if self.switch_save_checkpoint:
            filename_best = f'checkpoint_best_ep{self.epoch}_it{self.it}_valloss{loss_str}.pt'
            filepath_best = os.path.join(self.save_path, filename_best)
            try:
                # Copy the latest save to be the best save
                shutil.copyfile(filepath_latest, filepath_best)
                print(f"Saved best checkpoint to {filepath_best}")
                # Update the path tracker for the best checkpoint
                if hasattr(self, 'best_checkpoint_path') and self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path) and self.best_checkpoint_path != filepath_best:
                    # Remove the previous best checkpoint file only if it exists and is different
                     print(f"Removing previous best checkpoint: {self.best_checkpoint_path}")
                     os.remove(self.best_checkpoint_path)
                     
                self.best_checkpoint_path = filepath_best # Update tracker
                self.switch_save_checkpoint = False # Reset flag
            except Exception as e:
                print(f"Error saving best checkpoint: {e}")
        
        # --- Clean up older latest checkpoints (keep only one latest) --- 
        latest_checkpoints = sorted(glob.glob(os.path.join(self.save_path, 'checkpoint_latest_*.pt')))
        for ckpt in latest_checkpoints:
            if ckpt != filepath_latest: # Don't remove the one just saved
                try:
                    print(f"Removing old latest checkpoint: {ckpt}")
                    os.remove(ckpt)
                except OSError as e:
                     print(f"Error removing old latest checkpoint {ckpt}: {e}")
        
    # save_checkpoint_clean seems specific to the generator output and FAD
    # It might need adaptation if a "clean" encoder save is desired.
    # def save_checkpoint_clean(self):
    #     # ... (Original function - requires UNet and EMA applied to UNet)

    @torch.no_grad()
    def load_matching_weights(self, model, state_dict, prefix=""):
        """
        Load weights from state_dict into model, only when module names match 
        (considering prefix) and parameter sizes match.
        Leaves model parameters untouched if no match is found.
        Args:
            model: The model (e.g., Encoder or UNet) to load weights into.
            state_dict: The state dictionary containing weights.
            prefix: A prefix to consider for the keys in state_dict (e.g., "encoder.").
        """
        model_dict = model.state_dict()
        loaded_count = 0
        ignored_count = 0
        total_state_dict_keys = len(state_dict.keys())
        
        print(f"Attempting to load weights into model {model.__class__.__name__} with prefix '{prefix}'")
        
        matched_keys = set()

        for name, param in state_dict.items():
            # Construct the potential key name in the target model's state_dict
            model_key_name = name[len(prefix):] if name.startswith(prefix) else None
            
            if model_key_name and model_key_name in model_dict:
                if param.shape == model_dict[model_key_name].shape:
                    model_dict[model_key_name].copy_(param)
                    loaded_count += 1
                    matched_keys.add(model_key_name)
                    # print(f"  Loaded: {model_key_name} (Matches: {name})") # Verbose logging
                else:
                    print(f"  Size mismatch: {model_key_name} ({model_dict[model_key_name].shape}) vs {name} ({param.shape}) - IGNORING")
                    ignored_count += 1
            # else: # Log keys from state_dict that were not found in the model
                # print(f"  Key not found in model or prefix mismatch: {name} - IGNORING")
                # ignored_count += 1
                
        missing_keys = set(model_dict.keys()) - matched_keys
        if missing_keys:
            print(f"  Warning: The following keys in the model were not found in the state_dict (or had prefix/size mismatch): {missing_keys}")
            
        # Load the modified state dict back into the model
        model.load_state_dict(model_dict)
        print(f"Loaded {loaded_count} / {len(model_dict)} layers from state_dict ({total_state_dict_keys} keys in state_dict, {ignored_count} ignored).")
        
        return loaded_count > 0 # Return True if any weights were loaded

    def get_models(self):
        """ Initialize models, optimizer, scaler, EMA, and load checkpoints. """
        
        # --- Initialize Model --- 
        if hparams.use_contrastive and hparams.train_encoder_only:
            print("Initializing Encoder model for contrastive finetuning.")
            model = Encoder().to(self.device)
            self.model_type = "Encoder"
        else:
            # Default to UNet for original consistency training or full model finetuning
            print("Initializing UNet model.")
            model = UNet().to(self.device)
            self.model_type = "UNet"
        
        # Store the raw model reference before DDP wrapping
        self.raw_model = model 
        
        # --- Parameters to Optimize --- 
        if hparams.train_encoder_only and self.model_type == "Encoder":
             print("Configuring optimizer for Encoder parameters only.")
             parameters_to_optimize = model.parameters()
        elif hparams.train_encoder_only and self.model_type == "UNet":
             print("Configuring optimizer for UNet's Encoder parameters only.")
             # Freeze all parameters first
             for param in model.parameters():
                 param.requires_grad = False
             # Unfreeze only the encoder parameters
             for param in model.encoder.parameters():
                 param.requires_grad = True
             parameters_to_optimize = [p for p in model.parameters() if p.requires_grad]
             # Verify some parameters are trainable
             num_trainable = sum(p.numel() for p in parameters_to_optimize)
             print(f"Number of trainable parameters (Encoder only): {num_trainable}")
             if num_trainable == 0:
                 print("Warning: No trainable parameters found for encoder-only training in UNet.")
        else:
             print("Configuring optimizer for all model parameters.")
             parameters_to_optimize = model.parameters()
             # Ensure all params are trainable if not freezing
             for param in model.parameters():
                 param.requires_grad = True
             num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
             print(f"Total number of trainable parameters: {num_trainable}")

        # --- Optimizer --- 
        self.optimizer = torch.optim.RAdam(
            parameters_to_optimize, 
            lr=hparams.lr, 
            betas=(hparams.optimizer_beta1, hparams.optimizer_beta2)
        )
        
        # --- Scaler --- 
        self.scaler = torch.amp.GradScaler(enabled=hparams.mixed_precision)
        
        # --- EMA --- 
        # EMA tracks the parameters passed to the optimizer
        self.ema = ExponentialMovingAverage(
            parameters_to_optimize, 
            decay=hparams.ema_momentum, 
            use_num_updates=hparams.warmup_ema
        )
        
        # --- Initialize Training State --- 
        self.it = 0
        self.epoch = 0
        self.score = 1e7 # Used for FAD in original script, maybe repurpose for val_loss?
        self.current_score = self.score # ditto
        self.best_checkpoint_path = '' # Path to the best checkpoint based on validation loss
        self.save_path = None # Will be set later or loaded from checkpoint
        
        # --- Load Checkpoint --- 
        load_path = hparams.load_path
        # If fine-tuning encoder, use pretrained_encoder_path if load_path is not set
        if hparams.train_encoder_only and not load_path and hparams.pretrained_encoder_path:
            load_path = hparams.pretrained_encoder_path
            print(f"Using pretrained encoder path for loading: {load_path}")
            # Ensure we load *only* encoder weights even if the file contains a full UNet
            load_optimizer_state = False 
            load_ema_state = False
            load_iter_state = False # Start fine-tuning from iter 0
        elif load_path: 
             # Resuming a previous training run (contrastive or otherwise)
             print(f"Resuming training from checkpoint: {load_path}")
             load_optimizer_state = hparams.load_optimizer
             load_ema_state = hparams.load_ema
             load_iter_state = hparams.load_iter
        else:
            print("No checkpoint specified, training from scratch.")
            load_path = None

        if load_path is not None:
            print(f"Loading checkpoint from: {load_path}")
            try:
                checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
                
                weights_loaded = False
                # --- Load Model Weights --- 
                if self.model_type == "Encoder":
                    if 'encoder_state_dict' in checkpoint:
                         weights_loaded = self.load_matching_weights(model, checkpoint['encoder_state_dict'])
                    elif 'gen_state_dict' in checkpoint:
                         # Try loading from UNet's encoder part
                         print("  Encoder state not found, attempting to load from UNet 'gen_state_dict' with prefix 'encoder.'...")
                         weights_loaded = self.load_matching_weights(model, checkpoint['gen_state_dict'], prefix="encoder.")
                    else:
                         print("  Warning: Checkpoint does not contain 'encoder_state_dict' or 'gen_state_dict'. Cannot load encoder weights.")
                
                elif self.model_type == "UNet":
                    if 'gen_state_dict' in checkpoint:
                        weights_loaded = self.load_matching_weights(model, checkpoint['gen_state_dict'])
                    elif 'encoder_state_dict' in checkpoint and hparams.train_encoder_only:
                        # Load only encoder weights into the UNet's encoder submodule
                        print("  Loading encoder weights into UNet's encoder submodule...")
                        weights_loaded = self.load_matching_weights(model.encoder, checkpoint['encoder_state_dict'])
                    else:
                         print("  Warning: Checkpoint does not contain 'gen_state_dict'. Cannot load UNet weights.")

                if not weights_loaded:
                     print("Warning: No model weights were loaded from the checkpoint.")
                     
                # --- Load Optimizer State --- 
                if load_optimizer_state and 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("  Loaded optimizer state.")
                        # Move optimizer state to the correct device
                        for state in self.optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(self.device)
                    except Exception as e:
                        print(f"  Warning: Could not load optimizer state: {e}. Training optimizer from scratch.")
                elif load_optimizer_state:
                     print("  Warning: load_optimizer=True but optimizer_state_dict not found in checkpoint.")
                     
                # --- Load Scaler State --- 
                # Always try to load scaler state if resuming, helpful for mixed precision
                if 'scaler_state_dict' in checkpoint:
                    try:
                        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                        print("  Loaded scaler state.")
                    except Exception as e:
                         print(f"  Warning: Could not load scaler state: {e}.")
                         
                # --- Load EMA State --- 
                if load_ema_state and hparams.enable_ema:
                     if self.model_type == "Encoder" and 'encoder_ema_state_dict' in checkpoint:
                         # Load EMA state into the separate EMA handler
                         # Need to load into a temporary encoder model first, then load EMA state dict
                         temp_encoder = Encoder().to(self.device)
                         self.load_matching_weights(temp_encoder, checkpoint['encoder_ema_state_dict'])
                         self.ema.load_state_dict(temp_encoder.state_dict()) # Load weights into EMA buffers
                         print("  Loaded encoder EMA state.")
                         del temp_encoder
                     elif self.model_type == "UNet" and 'ema_state_dict' in checkpoint:
                         self.ema.load_state_dict(checkpoint['ema_state_dict']) 
                         print("  Loaded UNet EMA state.")
                     else:
                          print(f"  Warning: load_ema=True but compatible EMA state ('{ 'encoder_ema_state_dict' if self.model_type == 'Encoder' else 'ema_state_dict'}') not found.")
                          
                # --- Load Training Progress --- 
                if load_iter_state:
                    self.it = checkpoint.get('it', 0)
                    self.epoch = checkpoint.get('epoch', 0)
                    self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    # Use the directory of the loaded checkpoint as the save path
                    self.save_path = os.path.dirname(load_path)
                    # Load the best checkpoint path if available
                    self.best_checkpoint_path = checkpoint.get('best_checkpoint_path', '') 
                    print(f"  Resuming from iteration {self.it}, epoch {self.epoch}, best_val_loss {self.best_val_loss:.4f}")
                    print(f"  Saving future checkpoints to: {self.save_path}")
                else:
                     print("  Starting training from iteration 0 (load_iter=False or not resuming).")

                del checkpoint # Free memory
                torch.cuda.empty_cache() # Clear cache after loading

            except FileNotFoundError:
                print(f"Checkpoint file not found at {load_path}. Training from scratch.")
            except Exception as e:
                 print(f"Error loading checkpoint from {load_path}: {e}. Training from scratch.")
                 # Reset state if loading failed badly
                 self.it = 0
                 self.epoch = 0
                 self.best_val_loss = float('inf')
                 self.best_checkpoint_path = ''
                 self.save_path = None # Let the init logic create a new path
                 # Re-initialize optimizer and EMA from scratch
                 self.optimizer = torch.optim.RAdam(parameters_to_optimize, lr=hparams.lr, betas=(hparams.optimizer_beta1, hparams.optimizer_beta2))
                 self.ema = ExponentialMovingAverage(parameters_to_optimize, decay=hparams.ema_momentum, use_num_updates=hparams.warmup_ema)

        # --- DDP Wrapping --- 
        if hparams.multi_gpu:
            print(f"Wrapping model with DDP on device {self.device}")
            self.model = DDP(model, device_ids=[self.device], broadcast_buffers=False, find_unused_parameters=False) # Set find_unused=False if sure no params are unused
        else:
            self.model = model # Use the raw model if not multi-gpu

        # Log parameter count (rank 0 only)
        if misc.get_rank()==0:
            total_params = sum(p.numel() for p in self.raw_model.parameters())
            trainable_params = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)
            print(f"--- Model Summary ({self.model_type}) ---")
            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")
            print(f"---------------------------")
            
        # Assign the potentially DDP-wrapped model to self.model for training
        # self.gen is kept only for compatibility with original test/FAD functions if needed
        if self.model_type == "UNet":
            self.gen = self.raw_model # Keep reference to raw UNet
        else:
            self.gen = None # No generator if only training encoder



def main():
    # Load configuration (remains the same)
    config_path = os.path.join(os.path.dirname(__file__), 'config.py')
    load_config(config_path)
    
    if not hparams.use_contrastive:
         print("Warning: Running main training loop but hparams.use_contrastive is False. The current setup is primarily for contrastive finetuning.")
         # Consider adding a check or separate entry point if non-contrastive training is intended
         
    trainer = Trainer()
    trainer.train()

# Keep main_fad and main_save_clean, but note they might not work if only encoder is trained
def main_fad():
    print("Warning: main_fad requires the full UNet model and is likely incompatible with encoder-only training.")
    # trainer = Trainer() # This will initialize based on hparams
    # trainer.calculate_fad() # This will likely fail or do nothing useful

def main_save_clean():
    print("Warning: main_save_clean requires the full UNet model and EMA setup. Incompatible with encoder-only training.")
    # trainer = Trainer() # This will initialize based on hparams
    # trainer.save_checkpoint_clean() # This will likely fail


if __name__ == "__main__":
    main()