import io
import tempfile
import asyncio
import os
from Chains import models
import logging
logger = logging.getLogger("Chains")

async def voice_to_text(audio_buffer: io.BytesIO | None, max_seconds: int = 20) -> str:
    logger.debug("Я починаю слухати")

    if audio_buffer is None:
        raise ValueError("audio_buffer must be io.BytesIO, got None")

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(audio_buffer.read())
        tmp_name = tmp.name

    tmp_trimmed_name = tmp_name.replace(".ogg", "_trimmed.ogg")

    try:
        # Обрізаємо аудіо
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", tmp_name,
            "-t", str(max_seconds),
            "-y",
            tmp_trimmed_name,
            stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait()

        def _transcribe(path):
            with open(path, "rb") as audio_file:
                transcription = models.whisper_model.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    language="uk",
                )
            return transcription.text

        result = await asyncio.to_thread(_transcribe, tmp_trimmed_name)

    finally:
        os.remove(tmp_name)
        if os.path.exists(tmp_trimmed_name):
            os.remove(tmp_trimmed_name)

    logger.debug("Я закінчую слухати")
    return result