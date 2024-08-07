import whisperx
import gc
import numpy as np
import ffmpeg
from typing import BinaryIO, Union

SAMPLE_RATE = 16000


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
    #audio_file = "/Users/samlee/Documents/sample/asr/en/英语专业四级听力计划/1.mp3"
    batch_size = 16  # reduce if low on GPU mem
    # compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    # compute_type = "int8"

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model, device, compute_type=compute_type)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    audio = load_audio(file)
    result = model.transcribe(audio, batch_size=batch_size)
    print('before alignment:')
    print(result["segments"])  # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    print('after alignment:')
    print(result["segments"])  # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token='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', device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    print('diarize_segments:')
    print(diarize_segments)
    print('segments are now assigned speaker IDs:')
    print(result["segments"])  # segments are now assigned speaker IDs


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
