import os

from aiogram import BaseMiddleware
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, Message

from Chains import text_to_voice
import settings_manager as s
from utils import current_answer_type


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
                await bot.send_voice(
                    chat_id=event.chat.id,
                    voice=BufferedInputFile(file=audio.read(), filename="voice.ogg"),
                    reply_to_message_id=event.message_id
                )
            else:
                await event.answer(answer_to_user)
            return
        self.processing.add(user_id)
        try:
            return await handler(event, data)
        finally:
            self.processing.discard(user_id)


class AnswerTypeMiddleware(BaseMiddleware):
    async def __call__(self, handler, event: Message, data: dict):
        state: FSMContext = data.get("state")
        if state:
            state_data = await state.get_data()
            answer_type = state_data.get("answer_type", s.get('ANSWER_TYPE'))
            current_answer_type.set(answer_type)

        return await handler(event, data)