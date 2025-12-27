import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torchaudio as ta
from loguru import logger
from numpy import ndarray
from torch import Tensor

# PATCH START: Handle missing AudioMetaData and backend in newer torchaudio
try:
    from torchaudio import AudioMetaData

    TA_RESAMPLE_SINC = "sinc_interp_hann"
    TA_RESAMPLE_KAISER = "sinc_interp_kaiser"
except ImportError:
    try:
        from torchaudio.backend.common import AudioMetaData

        TA_RESAMPLE_SINC = "sinc_interpolation"
        TA_RESAMPLE_KAISER = "kaiser_window"
    except ImportError:
        # Define dummy AudioMetaData for newer torchaudio versions (e.g. 2.9+)
        class AudioMetaData:
            def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample=0, encoding="UNKNOWN"):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

        TA_RESAMPLE_SINC = "sinc_interp_hann"
        TA_RESAMPLE_KAISER = "sinc_interp_kaiser"
# PATCH END

from df.logger import warn_once
from df.torchcodec_loader import check_torchcodec
from df.utils import download_file, get_cache_dir, get_git_root


def load_audio(
    file: str, sr: Optional[int] = None, verbose=True, **kwargs
) -> Tuple[Tensor, AudioMetaData]:
    """Loads an audio file using torchaudio.

    Args:
        file (str): Path to an audio file.
        sr (int): Optionally resample audio to specified target sampling rate.
        **kwargs: Passed to torchaudio.load(). Depends on the backend. The resample method
            may be set via `method` which is passed to `resample()`.

    Returns:
        audio (Tensor): Audio tensor of shape [C, T], if channels_first=True (default).
        info (AudioMetaData): Meta data of the original audio file. Contains the original sr.
    """
    require_torchcodec = os.getenv("DF_REQUIRE_TORCHCODEC", "0") == "1"
    check_torchcodec(require=require_torchcodec)

    ikwargs = {}
    if "format" in kwargs:
        ikwargs["format"] = kwargs["format"]
    rkwargs = {}
    if "method" in kwargs:
        rkwargs["method"] = kwargs.pop("method")
    
    # PATCH START: Handle missing ta.info
    if hasattr(ta, "info"):
        info: AudioMetaData = ta.info(file, **ikwargs)
    else:
        # Fallback for newer torchaudio where info is missing
        try:
            from torchcodec.decoders import AudioDecoder
            d = AudioDecoder(file)
            m = d.metadata
            # Estimate num_frames (approximate as strictly not available in simple metadata)
            # But accurate enough for sample rate scaling checks
            num_frames = int(m.duration_seconds * m.sample_rate)
            info = AudioMetaData(m.sample_rate, num_frames, m.num_channels, 0, m.codec)
        except Exception:
             # Fallback using soundfile if torchcodec missing
             try:
                 import soundfile as sf
                 sf_info = sf.info(file)
                 info = AudioMetaData(sf_info.samplerate, sf_info.frames, sf_info.channels, 0, sf_info.format)
             except ImportError:
                 # Last resort: load audio to get metadata (slow)
                 # We can't use kwargs that depend on info here yet
                 temp_audio, temp_sr = ta.load(file)
                 info = AudioMetaData(temp_sr, temp_audio.shape[1], temp_audio.shape[0], 0, "UNKNOWN")
    # PATCH END

    if "num_frames" in kwargs and sr is not None:
        kwargs["num_frames"] *= info.sample_rate // sr
    audio, orig_sr = ta.load(file, **kwargs)
    if sr is not None and orig_sr != sr:
        if verbose:
            warn_once(
                f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). "
                "Resampling..."
            )
        audio = resample(audio, orig_sr, sr, **rkwargs)
    return audio.contiguous(), info


def save_audio(
    file: str,
    audio: Union[Tensor, ndarray],
    sr: int,
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    log: bool = False,
    dtype=torch.int16,
):
    outpath = file
    if suffix is not None:
        file, ext = os.path.splitext(file)
        outpath = file + f"_{suffix}" + ext
    if output_dir is not None:
        outpath = os.path.join(output_dir, os.path.basename(outpath))
    if log:
        logger.info(f"Saving audio file '{outpath}'")
    audio = torch.as_tensor(audio)
    if audio.ndim == 1:
        audio.unsqueeze_(0)
    if dtype == torch.int16 and audio.dtype != torch.int16:
        audio = (audio * (1 << 15)).to(torch.int16)
    if dtype == torch.float32 and audio.dtype != torch.float32:
        audio = audio.to(torch.float32) / (1 << 15)
    ta.save(outpath, audio, sr)


try:
    from torchaudio.functional import resample as ta_resample
except ImportError:
    from torchaudio.compliance.kaldi import resample_waveform as ta_resample  # type: ignore


def get_resample_params(method: str) -> Dict[str, Any]:
    params = {
        "sinc_fast": {"resampling_method": TA_RESAMPLE_SINC, "lowpass_filter_width": 16},
        "sinc_best": {"resampling_method": TA_RESAMPLE_SINC, "lowpass_filter_width": 64},
        "kaiser_fast": {
            "resampling_method": TA_RESAMPLE_KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "beta": 8.555504641634386,
        },
        "kaiser_best": {
            "resampling_method": TA_RESAMPLE_KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.9475937167399596,
            "beta": 14.769656459379492,
        },
    }
    assert method in params.keys(), f"method must be one of {list(params.keys())}"
    return params[method]


def resample(audio: Tensor, orig_sr: int, new_sr: int, method="sinc_fast"):
    params = get_resample_params(method)
    return ta_resample(audio, orig_sr, new_sr, **params)


def get_test_sample(sr: int = 48000) -> Tensor:
    dir = get_git_root()
    file_path = os.path.join("assets", "clean_freesound_33711.wav")
    if dir is None:
        url = "https://github.com/Rikorose/DeepFilterNet/raw/main/" + file_path
        save_dir = get_cache_dir()
        path = download_file(url, save_dir)
    else:
        path = os.path.join(dir, file_path)
    sample, _ = load_audio(path, sr=sr)
    return sample
