from aiogram import Router, F, Bot
from aiogram.types import Message, BufferedInputFile
from aiogram.filters.command import Command, CommandStart
from aiogram.fsm.context import FSMContext
from Telegram.state import UserState
from functools import wraps
from Chains import *
from dotenv import load_dotenv
import os
from constants import *

load_dotenv()
admin_id = os.getenv('ADMIN_ID')
user = Router()

photo_quality = -1

def admin_only(func):
    @wraps(func)
    async def wrapper(message: Message, *args, **kwargs):
        if message.from_user.id != admin_id:
            return
        return await func(message, *args, **kwargs)
    return wrapper

async def reply_voice(message: Message, text: str):
    """
    Функція для відповіді на повідомлення за допомогою голосового повідомлення
    :param message: повідомлення на яке відповідає функція
    :param text: текст голосового
    """
    audio = await text_to_voice(text)
    await message.reply_voice(
        voice=BufferedInputFile(file=audio.read(), filename="voice.ogg")
    )

async def set_user_name(message: Message, state: FSMContext, name: str):
    """
    Функція для встановлення імені користувача
    :param message: повідомлення на яке відповідається підтвердженням
    :param state: стан телеграм чату
    :param name: ім'я яке треба записати
    """
    await state.update_data(user_name=name)
    await reply_voice(message, f"Ваше ім'я збережено: {name}")

async def set_bot_name(message: Message, state: FSMContext, name: str):
    """
    Функція для обробки імені бота
    :param message: повідомлення на яке відповідає бот
    :param state: стан чату
    :param name: ім'я яке треба записати
    """
    await state.update_data(bot_name=name)
    await reply_voice(message, f"Моє нове ім'я збережено: {name}")

async def handle_intent(message: Message, state: FSMContext):
    """
    Функція для відповіді на повідомлення за допомогою LLM.
    Використовує нейронну мережу для аналізу повідомлення
    на предмет того чи містить запит виконати якусь інструкцію. Наявні такі команди: змінити ім'я бота,
    змінити ім'я користувача в пам'яті бота, відповідь на запитання.
    Актуальне повідомлення для аналізу бере з актуального state. message потрібно тільки для відповіді на нього
    :param message: повідомлення на яке відповідає бот для підтвердження команди або відповіді на запитання
    :param state: пам'ять бота
    """
    state_data = await state.get_data()

    user_name, bot_name = state_data['user_name'], state_data['bot_name']

    await reply_voice(message, f"Я аналізую запит. Будь ласка, почекайте")

    history = state_data.get('history', [])

    history.append({"role": "human", "content": state_data['user_question']})

    if len(history) > MAX_MESSAGE_MEMORY:
        history = history[-MAX_MESSAGE_MEMORY:]

    llm_response = await run_llm(history,state_data['user_question'], state_data['user_photo'], user_name, bot_name)

    history.append({"role": "ai", "content": llm_response.value})

    await state.update_data(history=history)

    match llm_response.intent:
        case "set_name":
            await set_user_name(message, state, llm_response.value)

        case "set_bot_name":
            await set_bot_name(message, state, llm_response.value)

        case "question":
            await reply_voice(message, llm_response.value)
            await state.set_state(UserState.user_question)

        case "specification":
            await reply_voice(message, llm_response.value)
            await state.set_state(UserState.is_specification)
            return

@user.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.update_data(bot_name="Остап", user_name=message.from_user.first_name)
    state_data = await state.get_data()
    user_name, bot_name = state_data['user_name'], state_data['bot_name']

    hello_message = (f"Вітаю, {user_name}! Мене звуть {bot_name}!"
                     "Я створений для допомоги людям із вадами зору. "
                     "Я можу розпізнавати зображення і відповідати на питання стосовно них. "
                     "Для того щоб розпочати роботу, будь ласка, відправте фото і питання щодо нього. "
                     "Питання може бути як в текстовому вигляді так і аудіо-повідомленням. "
                     "Я також можу допомогти Вам підібрати ракурс, якщо мені буде складно його аналізувати. "
                     "Ви також можете попросити мене називати вас інакше, або змінити моє ім'я")

    await reply_voice(message, hello_message)

    await state.set_state(UserState.user_photo)


@user.message(Command("name"))
async def cmd_name(message: Message, state: FSMContext):
    parts = message.text.split(maxsplit=1)
    if len(parts) > 1:
        await set_user_name(message, state, parts[1])
    else:
        await reply_voice(message, "Будь ласка, вкажіть ім'я після команди. Наприклад: похила риска нейм англійськими літерами Іван")

@user.message(Command("name"), ~F.text)
async def cmd_name_not_text(message: Message):
    await reply_voice(message, "Після команди похила риска нейм англійськими літерами потрібно написати ім'я текстом")


@user.message(Command("botname"))
async def cmd_name(message: Message, state: FSMContext):
    parts = message.text.split(maxsplit=1)
    if len(parts) > 1:
        await set_bot_name(message, state, parts[1])
    else:
        await reply_voice(message,
                           "Будь ласка, вкажіть ім'я після команди. Наприклад: похила риска бот нейм без пропусків англійськими літерами та ваше ім'я")

@user.message(Command("botname"), ~F.text)
async def cmd_name_not_text(message: Message):
    await reply_voice(message, "Після команди похила риска бот нейм без пропусків англійськими літерами потрібно написати ім'я текстом")


@user.message(UserState.user_photo, F.photo)
async def cmd_user_photo(message: Message, state: FSMContext, bot: Bot):
    photo = message.photo[photo_quality].file_id

    buffer = await bot.download(photo)
    image_bytes = buffer.read()

    await state.update_data(user_photo=image_bytes)

    if message.caption:
        await state.update_data(user_question=message.caption)
        await handle_intent(message,state)
    else:
        await reply_voice(message, "Фото отримано. Тепер, будь ласка, задайте питання")
    await state.set_state(UserState.user_question)

@user.message(UserState.is_specification, F.photo)
async def cmd_specification_photo(message: Message, state: FSMContext, bot: Bot):
    photo = message.photo[photo_quality].file_id
    buffer = await bot.download(photo)
    image_bytes = buffer.read()

    await state.update_data(user_photo=image_bytes)

    await handle_intent(message, state)

@user.message(UserState.is_specification)
async def no_new_photo(message: Message):
    await reply_voice(message, "Будь ласка, надішліть нове фото")

@user.message(UserState.user_question, F.text)
async def cmd_user_text_question(message: Message, state: FSMContext):
    await state.update_data(user_question=message.text)

    await handle_intent(message, state)

@user.message(UserState.user_question, F.voice)
async def cmd_user_voice_question(message: Message, state: FSMContext, bot: Bot):
    buffer = await bot.download(message.voice.file_id)
    text = await voice_to_text(buffer)
    await state.update_data(user_question=text)

    await handle_intent(message, state)

@user.message(UserState.user_question, F.photo)
async def cmd_new_photo(message: Message, state: FSMContext, bot: Bot):
    photo = message.photo[photo_quality].file_id
    buffer = await bot.download(photo)
    image_bytes = buffer.read()

    await state.update_data(user_photo=image_bytes)

    if message.caption:
        await state.update_data(user_question=message.caption)
        await handle_intent(message, state)
    else:
        await reply_voice(message, "Нове фото отримано. Тепер, будь ласка, задайте питання")
    await state.set_state(UserState.user_question)


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