import whisperx
import numpy as np
import ffmpeg
from typing import BinaryIO
import logging

SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


def transcribe(
        model: str,
        compute_type: str,
        language: str,
        align_model: str,
        initial_prompt: str,
        file: BinaryIO,
        ):
    # device = "cuda"
    device = "cpu"
    batch_size = 16  # reduce if low on GPU mem

    # 1. Transcribe with original whisper (batched)
    asr_options = {
        "initial_prompt": initial_prompt,
    }
    model = whisperx.load_model(model, device=device, compute_type=compute_type, language=language,
                                asr_options=asr_options)

    audio = load_audio(file)
    result = model.transcribe(audio, batch_size=batch_size)
    logger.info('before alignment:')
    logger.info(result["segments"])  # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"],
                                                  device=device, model_name=align_model)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    logger.info('after alignment:')
    logger.info(result["segments"])  # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token='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', device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    logger.info('diarize_segments:')
    logger.info(diarize_segments)
    logger.info('segments are now assigned speaker IDs:')
    logger.info(result["segments"])  # segments are now assigned speaker IDs
    return result


def load_audio(file: BinaryIO, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
