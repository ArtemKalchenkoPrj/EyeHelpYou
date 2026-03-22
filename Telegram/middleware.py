import os

from aiogram import BaseMiddleware
from aiogram.types import BufferedInputFile

from Chains import text_to_voice
import settings_manager as s

class ThrottlingMiddleware(BaseMiddleware):
    def __init__(self):
        self.processing = set()

    async def __call__(self, handler, event, data):
        user_id = event.from_user.id
        answer_to_user = "Я ще обробляю попереднє повідомлення. Будь ласка, зачекайте."
        if user_id in self.processing:
            bot = data["bot"]  # ← bot доступний через data
            answer_type = s.get('ANSWER_TYPE').lower().strip()
            if answer_type == "voice":
                audio = await text_to_voice(answer_to_user)
                await bot.answer_voice(
                    chat_id=event.chat.id,
                    voice=BufferedInputFile(file=audio.read(), filename="voice.ogg")
                )
            else:
                await event.answer(answer_to_user)
            return
        self.processing.add(user_id)
        try:
            return await handler(event, data)
        finally:
            self.processing.discard(user_id)