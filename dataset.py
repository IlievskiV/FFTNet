"""
-*- coding: utf-8 -*-


"""

import os
import numpy as np
import collections
import librosa
import torch
from torch.utils.data import Dataset

from audio import AudioProcessor

class LJSpeechDataset(Dataset):
    """
    This class extends torch.utils.data.Dataset class and for this reason it overrides the two functions:
    __len__ and __get_item__ in order another wrapper around it to know how to load and handle the batches.
    """

    def __init__(self, csv_file, root_dir, sample_rate,
                 num_mels, num_freq, min_level_db, frame_shift_ms,
                 frame_length_ms, preemphasis, ref_level_db,
                 num_quant, min_wav_len=0, max_wav_len=-1, rand_offset=True):

        # reads the metadata
        with open(csv_file, "r") as f:
            self.frames = [line.split(', ') for line in f]

        self._parse_data()
        self.root_dir = root_dir

        self.sample_rate = sample_rate

        # the
        self.min_wav_len = min_wav_len
        self.max_wav_len = max_wav_len if max_wav_len > 0 else inf
        self.rand_offset = rand_offset

        self.receptive_field = 2 ** num_quant

        self.ap = AudioProcessor(sample_rate, num_mels, num_freq, min_level_db, frame_shift_ms,
                                 frame_length_ms, preemphasis, ref_level_db)
        print(" > Reading LJSpeech from - {}".format(root_dir))
        print(" | > Number of instances : {}".format(len(self.frames)))
        print(" | > Max wav length: {}".format(self.max_wav_len))
        print(" | > Min wav length: {}".format(self.min_wav_len))
        print(" | > Receptive field: {}".format(self.receptive_field))
        self._sort_frames()

    def load_wav(self, filename):
        try:
            audio = librosa.core.load(filename, sr=self.sample_rate)
            return audio
        except RuntimeError as e:
            print(" !! Cannot read file : {}".format(filename))

    def _parse_data(self):

        self.wav_files = [f[0] for f in self.frames]
        self.mel_files = [f[1] for f in self.frames]
        self.wav_lengths = [int(f[2]) for f in self.frames]
        self.mel_lengths = [int(f[3]) for f in self.frames]

    def _sort_frames(self):
        r"""Sort sequences in ascending order"""
        print(" | > Max wav length {}".format(np.max(self.wav_lengths)))
        print(" | > Min wav length {}".format(np.min(self.wav_lengths)))
        print(" | > Avg wav length {}".format(np.mean(self.wav_lengths)))

        idxs = np.argsort(self.wav_lengths)
        new_frames = []
        ignored = []
        for i, idx in enumerate(idxs):
            length = self.wav_lengths[idx]
            if length < self.min_wav_len:
                ignored.append(idx)
            else:
                new_frames.append(self.frames[idx])
        print(" | > {} instances are ignored by min_wav_len ({})".format(
            len(ignored), self.min_wav_len))
        self.frames = new_frames
        self._parse_data()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        """
        
        :param idx: 
        :return: 
        """

        wav_name = os.path.join(self.root_dir, self.wav_files[idx])
        mel_name = os.path.join(self.root_dir, self.mel_files[idx] + '.npy')
        mel = np.load(mel_name)
        mel = mel.transpose(1, 0)
        wav = np.asarray(self.load_wav(wav_name)[0], dtype=np.float32)
        sample = {'mel': mel, 'wav': wav, 'item_idx': self.wav_files[idx]}
        return sample

    def collate_fn(self, batch):
        """
            Perform pre-processing and create a final data batch. The 'batch' is coming from the __get_item__ function.
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.Mapping):
            B = len(batch)

            # determine the longest audio in the. In case the maximum defined length is less than the
            # actual maximum length in the data set
            pred_lens = [np.minimum(d['wav'].shape[0] - 1, self.max_wav_len - 1) for d in batch]
            max_len = np.max(pred_lens) + self.receptive_field - 1

            if max_len > self.max_wav_len:
                max_len = self.max_wav_len

            # initialize the batch of wavs and mels to all zeros in order to have the padding later on
            wavs = np.zeros([B, max_len + self.receptive_field - 1])
            mels = np.zeros([B, max_len + self.receptive_field - 1, self.ap.num_mels])

            # iterate the batches one by one
            for idx, d in enumerate(batch):
                wav = d['wav']
                mel = d['mel']

                # mu-law encoding
                wav = self.ap.mulaw_encode(wav, 2 ** 8)

                # align mel specs with wav by cloning frames such that wav and mel have the same length
                mel = self.ap.align_feats(wav, mel)

                # if wav len is long, sample a starting offset
                if wav.shape[0] > self.max_wav_len:
                    gap = wav.shape[0] - self.max_wav_len
                    if self.rand_offset:
                        offset = np.random.randint(0, gap)
                    else:
                        offset = 0
                    wav = wav[offset:offset + self.max_wav_len]
                    mel = mel[offset:offset + self.max_wav_len]

                # calculate the padding after the end of the actual content of the wav and mel
                pad_w = max_len - wav.shape[0]
                assert wav.shape[0] == mel.shape[0]
                assert wav.shape[0] <= self.max_wav_len

                # pad with zeros from the beginning until the receptive field
                # pad with zeros from the end of the actual content until the maximum possible length
                wav = np.pad(wav, [self.receptive_field - 1, pad_w], mode='constant', constant_values=0.0)
                mel = np.pad(mel, [[self.receptive_field - 1, pad_w], [0, 0]], mode='constant', constant_values=0.0)

                # update the batch with the actual values
                wavs[idx] += wav
                mels[idx] += mel

            # the mels are everything from the first element onwards
            mels = torch.FloatTensor(mels[:, 1:])

            # the target is everything from the receptive field onwards
            targets = torch.LongTensor(wavs[:, self.receptive_field:])

            # the inputs are everything but the last element, i.e. shifted by one
            inputs = torch.FloatTensor(wavs[:, :-1])
            pred_lens = torch.LongTensor(pred_lens)
            return inputs, mels, pred_lens, targets

        raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}".format(type(batch[0]))))
