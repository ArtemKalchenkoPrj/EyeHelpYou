from typing import Literal
import os

import edge_tts
import io
import logging

from aiogram.types import BufferedInputFile, Message

import re

from constants import DEFAULT_TTS_VOICE

logger = logging.getLogger("Chains")

def _extract_links(text: str) -> tuple[str, str]:
    """
    Повертає (текст без посилань, список посилань)
    """
    links = re.findall(r'https?://\S+', text)
    links = "\n".join(links)
    clean_text = re.sub(r'https?://\S+', '', text).strip()
    # Прибираємо подвійні пробіли
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text, links

def _filter_special_chars(text: str) -> str:
    allowed = r"a-zA-Zа-яА-ЯіІїЇєЄґҐ0-9\s.,!?"
    pattern = f"[^{allowed}]"
    return re.sub(pattern, "", text)


async def text_to_voice(text: str) -> io.BytesIO:

    logger.debug("Я починаю говорити")
    communicate = edge_tts.Communicate(text, voice=DEFAULT_TTS_VOICE)
    audio = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.write(chunk["data"])
    audio.seek(0)

    logger.debug("Я закінчую говорити")

    return audio

async def answer_to_user(message: Message, text: str, answer_type: Literal['voice','text'] | None = None):
    """
    Функція для відповіді на повідомлення за допомогою голосового повідомлення.
    :param message: Повідомлення на яке відповідає функція.
    :param text: Текст голосового.
    :param answer_type: Яким чином буде виконуватися відповідь на повідомлення. voice - перетворення тексту на голосове, text - відповідь звичайним текстом
    """
    if answer_type is None:
        answer_type = os.getenv("ANSWER_TYPE", "voice").lower().strip()

    match answer_type:
        case 'voice':
            clean_text, links = _extract_links(text)
            filtered_text = _filter_special_chars(clean_text)
            audio = await text_to_voice(filtered_text)
            await message.reply_voice(
                voice=BufferedInputFile(file=audio.read(), filename="voice.ogg")
            )
            if links:
                audio = await text_to_voice("Також посилання")
                await message.reply_voice(
                    voice=BufferedInputFile(file=audio.read(), filename="voice.ogg")
                )
                await message.answer(links)
        case 'text':
            await message.reply(text)
        case _:
            raise NotImplementedError(f"answer_type {answer_type} is not supported.")