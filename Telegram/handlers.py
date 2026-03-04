from aiogram import Router, F, Bot
from aiogram.types import Message, BufferedInputFile
from aiogram.filters.command import Command, CommandStart

from aiogram.fsm.context import FSMContext
from Telegram.state import UserState

from Chains import *

user = Router()

async def get_names(state_data: dict, message: Message) -> tuple[str, str]:
    user_name = state_data.get('user_name') or message.from_user.first_name
    bot_name = state_data.get('bot_name') or "Остап"
    return user_name, bot_name

async def reply_voice(message: Message, text: str):
    audio = await text_to_voice(text)
    await message.reply_voice(
        voice=BufferedInputFile(file=audio.read(), filename="voice.ogg")
    )

@user.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    hello_message = ("Вітаю! Бот було створено для допомоги людям із вадами зору. "
                     "Бот здатен аналізувати зображення і відповідати на задані питання. "
                     "Для того щоб розпочати роботу, будь ласка, відправте фото і питання щодо нього. "
                     "Питання може бути як в текстовому вигляді так і аудіо-повідомленням. "
                     "Бот також допоможе Вам підібрати ракурс, якщо йому буде складно його аналізувати")

    await reply_voice(message, hello_message)

    await state.set_state(UserState.user_photo)

@user.message(Command("name"))
async def cmd_name(message: Message, state: FSMContext):
    parts = message.text.split(maxsplit=1)

    if len(parts) > 1:
        name = parts[1]
        await reply_voice(message, f"Ваше ім'я збережено: {name}")
        await state.update_data(user_name=name)
    else:
        await reply_voice(message, "Будь ласка, вкажіть ім'я після команди. Наприклад: похила риска нейм англійськими літерами Іван")

@user.message(Command("name"), ~F.text)
async def cmd_name_not_text(message: Message):
    await reply_voice(message, "Після команди похила риска нейм англійськими літерами потрібно написати ім'я текстом")

@user.message(Command("botname"))
async def cmd_name(message: Message, state: FSMContext):
    parts = message.text.split(maxsplit=1)

    if len(parts) > 1:
        name = parts[1]
        await reply_voice(message, f"Ім'я бота збережено: {name}")
        await state.update_data(bot_name=name)
    else:
        await reply_voice(message, "Будь ласка, вкажіть ім'я після команди. Наприклад: похила риска бот нейм без пропусків англійськими літерами та ваше ім'я")

@user.message(Command("botname"), ~F.text)
async def cmd_name_not_text(message: Message):
    await reply_voice(message, "Після команди похила риска бот нейм без пропусків англійськими літерами потрібно написати ім'я текстом")


@user.message(UserState.user_photo, F.photo)
async def cmd_user_photo(message: Message, state: FSMContext, bot: Bot):
    photo = message.photo[-1].file_id

    buffer = await bot.download(photo)
    image_bytes = buffer.read()

    await state.update_data(user_photo=image_bytes)

    await reply_voice(message, "Фото отримано. Тепер, будь ласка, задайте питання")

    await state.set_state(UserState.user_question)


@user.message(UserState.user_question, F.text)
async def cmd_user_text_question(message: Message, state: FSMContext):
    await state.update_data(user_question=message.text)

    state_data = await state.get_data()

    user_name, bot_name = await get_names(state_data, message)
    await reply_voice(message, f"Помічник аналізує зображення. Будь ласка, почекайте його відповіді")

    llm_response = await run_llm(state_data['user_question'], state_data['user_photo'], user_name, bot_name)

    await reply_voice(message, llm_response)
    await state.set_state(UserState.user_question)

@user.message(UserState.user_question, F.voice)
async def cmd_user_voice_question(message: Message, state: FSMContext, bot: Bot):
    buffer = await bot.download(message.voice.file_id)
    text = await voice_to_text(buffer)

    await state.update_data(user_question=text)
    state_data = await state.get_data()

    user_name, bot_name = await get_names(state_data, message)
    await reply_voice(message, f"Помічник аналізує зображення. Будь ласка, почекайте його відповіді")

    llm_response = await run_llm(state_data['user_question'], state_data['user_photo'], user_name, bot_name)

    await reply_voice(message, llm_response)
    await state.set_state(UserState.user_question)

@user.message(UserState.user_question, F.photo)
async def cmd_new_photo(message: Message, state: FSMContext, bot: Bot):
    photo = message.photo[-1].file_id
    buffer = await bot.download(photo)
    image_bytes = buffer.read()

    await state.update_data(user_photo=image_bytes)

    audio = await text_to_voice("Нове фото отримано. Тепер, будь ласка, задайте питання")
    await message.reply_voice(
        voice=BufferedInputFile(file=audio.read(), filename="voice.ogg")
    )

@user.message(UserState.user_photo)
async def no_photo(message: Message):
    message_text = "Будь ласка, спершу надішліть фото"

    audio = await text_to_voice(message_text)
    await message.reply_voice(
        voice=BufferedInputFile(file=audio.read(), filename="voice.ogg")
    )

@user.message(UserState.user_question)
async def no_question(message: Message):
    message_text = "Будь ласка, надішліть питання текстом або голосовим повідомленням"

    audio = await text_to_voice(message_text)
    await message.reply_voice(
        voice=BufferedInputFile(file=audio.read(), filename="voice.ogg")
    )

@user.message()
async def default_handler(message: Message):
    message_text = ("Щось пішло не так. будь ласка, надрукуйте команду "
                    "похила риска старт англійськими літерами щоб перезапустити бота")

    audio = await text_to_voice(message_text)
    await message.reply_voice(
        voice=BufferedInputFile(file=audio.read(), filename="voice.ogg")
    )