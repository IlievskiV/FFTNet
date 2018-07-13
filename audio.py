import os
import librosa
import pickle
import copy
# import soundfile as sf
import numpy as np
# import pyworld as pw
from scipy import signal

_mel_basis = None


class WorldProcessor(object):
    def __init__(self, sample_rate, num_freq, num_mels, ref_level_db, min_level_db):
        self.sample_rate = sample_rate
        self.num_freq = num_freq
        self.n_fft = (num_freq - 1) * 2
        self.num_mels = num_mels
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.power = 1.4

    def load_wav(self, path):
        x, fs = librosa.load(path, dtype=np.float64)
        assert fs == self.sample_rate
        return x, fs

    def world_encode(self, x, fs):
        r"""Encode voice signal with WORLD features.It returns normalized
        spectral envelope and F0, ready for training."""
        _f0_h, t_h = pw.harvest(x, fs)
        f0_h = pw.stonemask(x, _f0_h, t_h, fs)
        sp_h = pw.cheaptrick(x, f0_h, t_h, fs, fft_size=self.n_fft)
        ap_h = pw.d4c(x, f0_h, t_h, fs, fft_size=self.n_fft)
        sp_hn = self.normalize_spectrogram(sp_h)
        f0_hn = self.normalize_f0(f0_h)
        return f0_hn, sp_hn, ap_h

    def world_decode(self, f0_hn, sp_hn, ap_h):
        r"""Decode WORLD features to voice signal. It denormalizes spectral
        envelope and F0 before decoding"""
        sp_h = self.denormalize_spectrogram(sp_hn)
        f0_h = self.denormalize_f0(f0_hn)
        sp_h = sp_h.astype(np.float64)
        y_h = pw.synthesize(f0_h, sp_h, ap_h, self.sample_rate, pw.default_frame_period)
        return y_h

    def normalize_f0(self, f0):
        """
        
        
        :param f0: 
        :return: 
        """
        return f0 / 800.0

    def denormalize_f0(self, f0):
        """
        
        :param f0: 
        :return: 
        """
        return f0 * 800.0

    def normalize_spectrogram(self, sp):
        """
         Converting the spectrogram to decibels and normalizing
        """

        sp = sp.astype(np.float32)
        sp[sp == 0] = 1e-5
        sp = np.log10(sp)
        # sp -= self.ref_level_db
        sp = (sp - self.min_level_db) / -self.min_level_db
        return sp

    def denormalize_spectrogram(self, sp):
        """
        
        :param sp: 
        :return: 
        """
        sp = sp.astype(np.float32)
        sp = (sp * -self.min_level_db) + self.min_level_db
        # sp += self.ref_level_db
        sp = np.power(10.0, sp)
        return sp

    def _build_mel_basis(self, ):
        """
        Calculates the Mel filter banks
        """
        return librosa.filters.mel(self.sample_rate, self.n_fft, n_mels=self.num_mels)

    def melspectrogram(self, sp):
        """Normalized Mel Spectrogram"""
        msp = self._linear_to_mel(sp)
        msp = self.normalize_spectrogram(msp)
        return msp

    def _linear_to_mel(self, spectrogram):
        """Mel Spectrogram in original scales"""
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)


class AudioProcessor(object):
    def __init__(self, sample_rate, num_mels, num_freq, min_level_db, frame_shift_ms,
                 frame_length_ms, preemphasis, ref_level_db):
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.num_freq = num_freq
        self.min_level_db = min_level_db
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.preemphasis = preemphasis
        self.ref_level_db = ref_level_db

    def save_wav(self, wav, path):
        """
        Write the samples to WAV file
        """
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        librosa.output.write_wav(path, wav.astype(np.float), self.sample_rate)

    def _linear_to_mel(self, spectrogram):
        """Calculates the Mel filter banks"""
        global _mel_basis
        if _mel_basis is None:
            _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _build_mel_basis(self, ):
        """Calculates the Mel filter banks"""
        n_fft = (self.num_freq - 1) * 2
        return librosa.filters.mel(self.sample_rate, n_fft, n_mels=self.num_mels)

    def _normalize(self, S):
        """Normalize the spectrogram if already converted to decibels"""
        return np.clip((S - self.min_level_db) / -self.min_level_db, 1e-8, 1)

    def _denormalize(self, S):
        """De-normalize the spectrogram if already converted to decibels"""
        return (np.clip(S, 0, 1) * -self.min_level_db) + self.min_level_db

    def _stft_parameters(self, ):
        """Calculate the number of samples in the audio corresponding to some time frame in miliseconds"""
        n_fft = (self.num_freq - 1) * 2
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(self.frame_length_ms / 1000.0 * self.sample_rate)
        return n_fft, hop_length, win_length

    def _amp_to_db(self, x):
        """Convert from amplitude to decibels"""
        return 20 * np.log10(np.maximum(1e-5, x))

    def _db_to_amp(self, x):
        """Convert from decibels to amplitude"""
        return np.power(10.0, x * 0.05)

    def apply_preemphasis(self, x):
        """The pre-emphasis pre-processing"""
        return signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x):
        """The inverse pre-emphasis pre-processing"""
        return signal.lfilter([1], [1, -self.preemphasis], x)

    def spectrogram(self, y):
        """Calculate linear-scale spectrogram expressed in decibels"""
        D = self._stft(self.apply_preemphasis(y))
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S)

    def inv_spectrogram(self, spectrogram):
        """Converts linear-scale spectrogram expressed in decibels to the original waveform"""
        S = self._denormalize(spectrogram)
        S = self._db_to_amp(S + self.ref_level_db)  # Convert back to linear
        # Reconstruct phase
        return self.apply_inv_preemphasis(self._griffin_lim(S ** self.power))

    def melspectrogram(self, y):
        """Calculates the mel-scale spectrogram expressed in decibels"""
        D = self._stft(self.apply_preemphasis(y))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S)

    def _stft(self, y):
        """Calculates the STFT of the input signal"""
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _istft(self, y):
        """Calculates the inverse STFT of the input signal"""
        _, hop_length, win_length = self._stft_parameters()
        return librosa.istft(y, hop_length=hop_length, win_length=win_length, window='hann')

    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        """
        This function slides a window over the audio waveform in order to detect silent parts.
        The sliding starts from the position 'hop_length' instead of zero. If all values in the
        window are below the minimal threshold value, it is considered as a silent part, the
        process is terminated and the starting position of the following window is returned.
        """

        # window size in terms of number of samples
        window_length = int(self.sample_rate * min_silence_sec)

        # hop length in terms of number of samples
        hop_length = int(window_length / 4)

        # threshold in linear-scale
        threshold = self._db_to_amp(threshold_db)

        # slides the window from the point of first hop to the point of last hop
        # no detection in the following fragment: [0, hop_length]
        for x in range(hop_length, len(wav) - window_length, hop_length):
            # if everything in the window is silent
            if np.max(wav[x:x + window_length]) < threshold:
                # return the starting index of the next window
                return x + hop_length

        # if no silence, return the length
        return len(wav)

    def mulaw_encode(self, wav, qc):
        """Mu-law quantization"""

        mu = qc - 1
        wav_abs = np.minimum(np.abs(wav), 1.0)
        magnitude = np.log(1 + mu * wav_abs) / np.log(1. + mu)
        signal = np.sign(wav) * magnitude
        # Quantize signal to the specified number of levels.
        signal = (signal + 1) / 2 * mu + 0.5
        return signal.astype(np.int32)

    def mulaw_decode(self, wav, qc):
        """ Recovers waveform from quantized values """
        mu = qc - 1
        # Map values back to [-1, 1].
        casted = wav.astype(np.float32)
        signal = 2 * (casted / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
        return np.sign(signal) * magnitude


    def align_feats(self, wav, feat):
        """Align audio signal and fetures. In this case the features are coming from the mel spectrogram."""

        assert len(wav.shape) == 1
        assert len(feat.shape) == 2

        # the shape of the feat matrix is (num_time_frames, num_mels)
        # by this we get the number of samples per one frame
        factor = wav.size // feat.shape[0]

        # with this each row (axis = 0) in the feature matrix is repeated 'factor'
        # number of times, such that feat.shape[0] (number od rows) = len(wav)
        feat = np.repeat(feat, factor, axis=0)

        # if any padding is needed if after the above division there is some reminder
        n_pad = wav.size - feat.shape[0]
        if n_pad != 0:
            assert n_pad > 0
            feat = np.pad(feat, [(0, n_pad), (0, 0)], mode="constant", constant_values=0)
        return feat
