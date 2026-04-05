import io
import tempfile
import asyncio
import os
from Chains import models
import logging
import pydub

logger = logging.getLogger("Chains")


async def is_speech_loud_enough(audio_path: str, threshold_db: float = -35.0) -> bool:
    def _check_loudness_sync():
        try:
            audio = pydub.AudioSegment.from_file(audio_path)
            # dBFS — це середня гучність відносно максимально можливої (0 dB)
            return audio.dBFS > threshold_db
        except Exception as e:
            logger.error(f"Помилка при аналізі гучності: {e}")
            return False

    return await asyncio.to_thread(_check_loudness_sync)


async def voice_to_text(audio_buffer: io.BytesIO | None, max_seconds: int = 20) -> str:
    logger.debug("Я починаю слухати")

    if audio_buffer is None:
        raise ValueError("audio_buffer must be io.BytesIO, got None")

    audio_buffer.seek(0)

    # Створюємо тимчасову директорію, щоб не було конфліктів імен
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.ogg")
        output_path = os.path.join(tmpdir, "trimmed.ogg")

        with open(input_path, "wb") as f:
            f.write(audio_buffer.read())

        try:
            # Обрізаємо аудіо з таймаутом
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", input_path,
                "-t", str(max_seconds),
                "-y", output_path,
                stderr=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL
            )

            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)

                # Додаємо перевірку гучності
                if not await is_speech_loud_enough(output_path):
                    logger.info("Аудіо занадто тихе (ймовірно шум або порожнеча).")
                    return ""

            except asyncio.TimeoutError:
                proc.kill()
                logger.error("FFMPEG timeout")
                return ""

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                logger.warning("Файл після обрізки порожній")
                return ""

            def _transcribe(path):
                # Відкриваємо файл безпосередньо для API
                with open(path, "rb") as audio_file:
                    transcription = models.whisper_model.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio_file,
                        language="uk",
                        temperature=0,
                    )
                return transcription.text

            result = await asyncio.to_thread(_transcribe, output_path)

        except Exception as e:
            logger.error(f"Помилка при обробці голосу: {e}")
            return ""

    logger.debug(f"Я закінчую слухати. Результат: {result[:20]}...")
    return result.strip()