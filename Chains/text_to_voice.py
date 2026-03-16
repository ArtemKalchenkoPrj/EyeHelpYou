from typing import Literal

import edge_tts
import io
import logging

from aiogram.types import BufferedInputFile, Message

from constants import *


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

async def answer_to_user(message: Message, text: str, answer_type: Literal['voice','text'] | None = None):
    if answer_type is None:
        answer_type = os.getenv("ANSWER_TYPE", "voice")

    """
    Функція для відповіді на повідомлення за допомогою голосового повідомлення.
    :param message: Повідомлення на яке відповідає функція.
    :param text: Текст голосового.
    :param answer_type: Яким чином буде виконуватися відповідь на повідомлення. voice - перетворення тексту на голосове, text - відповідь звичайним текстом
    """
    match answer_type:
        case 'voice':
            audio = await text_to_voice(text)
            await message.reply_voice(
                voice=BufferedInputFile(file=audio.read(), filename="voice.ogg")
            )
        case 'text':
            await message.reply(text)
        case _:
            raise NotImplementedError(f"answer_type {answer_type} is not supported.")