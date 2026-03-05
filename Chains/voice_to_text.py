import io
import whisper
import tempfile
import asyncio
import os
from Chains import models

async def voice_to_text(audio_buffer: io.BytesIO) -> str:
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(audio_buffer.read())
        tmp_name = tmp.name
    try:
        result = await asyncio.to_thread(models.whisper_model.transcribe, tmp_name, language="uk")
    finally:
        os.remove(tmp_name)

    return result["text"]