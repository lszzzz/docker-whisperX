import pytest
from api.service import transcribe
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(thread)d - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)

ch.setFormatter(formatter)
logger.addHandler(ch) #将日志输出至屏幕
logger = logging.getLogger(__name__)


def test_transcribe():
    file = open('/Users/samlee/Documents/sample/asr/en/cv-corpus-18.0-delta-2024-06-14/en/clips/common_voice_en_40493767.mp3', 'rb')

    transcribe(
        model_path='/Users/samlee/.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120',
        compute_type='int8',
        language=None,
        align_model=None,
        initial_prompt=None,
        file=file,
        device='cpu',
        identify_speaker=False
    )
