from torch.utils.data import Dataset
import torchaudio
import torch
import glob

class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""
    
    def __init__(self, audio_dir):
        super().__init__()
        
        self.filenames = glob.glob(audio_dir+"/*.mp3")
        _, self.sr = torchaudio.load(self.filenames[0])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        waveform, sample_rate = torchaudio.load(self.filenames[index])
        # if sample_rate:
        #     print('Load OK')
        target_length = 160000
        if waveform.size(1) < target_length:
            # If waveform is shorter than target length, pad it
            padding = target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.size(1) > target_length:
            # If waveform is longer than target length, truncate it
            waveform = waveform[:, :target_length]
        return waveform
