import edge_tts
import io
import logging
logger = logging.getLogger("Chains")

async def text_to_voice(text: str) -> io.BytesIO:

    logger.debug("Я починаю говорити")

    communicate = edge_tts.Communicate(text, voice="uk-UA-OstapNeural")
    audio = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.write(chunk["data"])
    audio.seek(0)

    logger.debug("Я закінчую говорити")

    return audio