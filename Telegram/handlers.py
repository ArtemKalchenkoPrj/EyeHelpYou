from aiogram import Router, F
from aiogram.types import Message
from aiogram.filters.command import Command, CommandStart

user = Router()

@user.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Вітаю! Бот було створено для допомоги людям із вадами зору."
                         "Бот здатен аналізувати зображення і відповідати на задані питання."
                         "Для того щоб розпочати роботу, будь ласка, відправте фото і питання щодо нього."
                         "Питання може бути як в текстовому вигляді так і аудіо-повідомленням."
                         "Бот також допоможе Вам підібрати ракурс, якщо йому буде складно його аналізувати")

@user.message(F.photo)
async def cmd_photo(message: Message):
    await message.answer_photo(photo=message.photo[0].file_id)

@user.message(F.voice)
async def cmd_voice(message: Message):
    await message.answer_voice(voice=message.voice.file_id)

@user.message()
async def cmd_default_message(message: Message):
    await message.answer("Ви відправили текст")