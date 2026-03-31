import base64
import os
import asyncio

from aiogram import Router, F, Bot
from aiogram.types import Message, ErrorEvent
from aiogram.filters.command import CommandStart
from aiogram.fsm.context import FSMContext
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langsmith import traceable

from Chains.text_to_voice import answer_to_user
from utils import search_web, trim_history, unpack_history, logger
from Telegram.state import UserState
from Chains import *
import settings_manager as s


user = Router()

async def _handle_with_thinking_message(message: Message, coro):
    """
    Запускає корутину і якщо через 2 секунди немає відповіді —
    надсилає повідомлення про очікування
    """
    thinking_message = None

    async def send_thinking():
        nonlocal thinking_message
        await asyncio.sleep(2)
        thinking_message = await answer_to_user(message, "Мені треба подумати, будь ласка, зачекайте")

    thinking_task = asyncio.create_task(send_thinking())

    try:
        result = await coro
    finally:
        thinking_task.cancel()
        if thinking_message:
            await thinking_message.delete()

    return result

async def is_valid_question_length(question, message,
                                   min_question_length: int = s.get("MIN_QUESTION_LENGTH"),
                                   max_question_length: int = s.get("MAX_QUESTION_LENGTH")) -> bool:
    if len(question.strip())<=min_question_length:
        await answer_to_user(message, "Вибачте, я не можу відповісти на таке коротке запитання. "
                                      "Будь ласка, надайте більш розгорнуте питання")
        return False
    if len(question.strip())>=max_question_length:
        await answer_to_user(message, "Вибачте, я не можу відповісти на таке велике питання. "
                                      "Будь ласка, сформулюйте питання більшстисло")
        return False
    return True

async def _get_image_from_message(message, bot):
    """Функція для отримання картинки з повідомлення і конвертація її у формат, який LLM може прочитати"""
    photo = message.photo[-1].file_id
    buffer = await bot.download(photo)
    buffer.seek(0)
    image_bytes = buffer.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64

@traceable(run_type="chain", name="Processor")
async def handle_processor(router_response: ChainRouter,
                           state: FSMContext,
                           message: Message):
    """
    Функція для обробки результатів, отриманих з processor
    :param router_response: інформація із router
    :param state: стан aiogram
    :param message: конкретне повідомлення, спіймане хендлером
    :return:
    """

    state_data = await state.get_data()
    history = state_data.get("history",[])
    user_name = state_data.get("user_name",f"{message.from_user.first_name}")
    bot_name = state_data.get("bot_name",s.get("DEFAULT_BOT_NAME"))
    is_vision_needed = router_response.is_vision_needed or False
    search_query = router_response.search_query or ""

    # Потібне зображення - дістати його
    user_image = state_data.get("wait_image", None)
    if is_vision_needed or user_image:
        # Є? Записати в історію
        if user_image:
            history.append(
                [AIMessage(
                    content=f"Маю отримати зображення, яке стосується запитання від користувача.",
                    tool_calls=[{"id": "vision_result", "name": "vision", "args": {}}]
                ),
                ToolMessage(
                    content=[
                        {
                            "type": "image",
                            "source_type": "base64",
                            "data": user_image,
                            "mime_type": "image/jpeg",
                        },
                    ],
                    tool_call_id="vision_result",
                )]
            )
            await state.update_data(history=trim_history(history), wait_image=None)

        # Нема? Зупинити процес. Попросити надати зображення
        else:
            await answer_to_user(message, "Здається, я не можу відповісти на це питання без зображення."
                                          "Чи не могли б ви його надати, будь ласка?")
            await state.set_state(UserState.wait_image)
            await state.update_data(history = trim_history(history), pending_router_response=router_response)
            return

    # Є пошуковий запит - виконати пошук - додати в історію
    if search_query:
        search_result = await search_web(search_query, 5)
        history.append(
            [AIMessage(
                content=f"Пошукаю інформацію в інтернеті за запитом: {search_query}",
                tool_calls=[{"id": "search_result", "name": "search", "args": {"search_query": search_query}}]
            ),
            ToolMessage(
                content=search_result,
                tool_call_id="search_result"
            )]
        )

    logger.debug(f"History types: {[type(m).__name__ for m in history]}")

    messages = unpack_history(history)

    processor_response = await run_processor(bot_name, user_name, messages)
    logger.debug(f"processor_response type: {type(processor_response)}, value: {repr(processor_response)}")

    history.append(AIMessage(processor_response))

    await state.set_state(UserState.wait_input)
    logger.debug("Стан змінено на wait_input")

    await answer_to_user(message, processor_response)
    logger.debug("answer_to_user виконано")

    await state.update_data(history=trim_history(history))
    logger.debug("Історію збережено")

@traceable(run_type="chain", name="Command")
async def handle_command(command_response: ChainCommand, state: FSMContext, message: Message):

    state_data = await state.get_data()

    history = state_data.get("history",[])

    match command_response.command:
        case "set_user_name":
            user_name = command_response.command_argument
            old_name = state_data.get("user_name", "невідомо")
            history.append(
                [AIMessage(
                content=f"Зміню ім'я користувача з {old_name} на {user_name}",
                tool_calls=[{"id": "set_user_name", "name": "set_user_name", "args": {"name": user_name}}]
                ),
                ToolMessage(
                    content=f"Тепер ім'я користувача - {user_name}",
                    tool_call_id="set_user_name"
                )]
            )
            history.append(AIMessage(f"Добре, тепер я буду називати вас {user_name}"))
            await state.update_data(user_name=user_name)
            await answer_to_user(message, f"Добре, тепер я буду називати вас {user_name}")
        case "set_bot_name":
            bot_name = command_response.command_argument
            old_bot_name = state_data.get("bot_name", s.get("DEFAULT_BOT_NAME"))
            history.append(
                [AIMessage(
                    content=f"Зміню своє ім'я з {old_bot_name} на {bot_name}",
                    tool_calls=[{"id": "set_bot_name", "name": "set_bot_name", "args": {"name": bot_name}}]
                ),
                ToolMessage(
                    content=f"Тепер твоє ім'я - {bot_name}",
                    tool_call_id="set_bot_name"
                )]
            )
            history.append(AIMessage(f"Добре, тепер мене звуть {bot_name}"))
            await state.update_data(bot_name=bot_name)
            await answer_to_user(message, f"Добре, тепер мене звуть {bot_name}")

    await state.update_data(history = trim_history(history))
    await state.set_state(UserState.wait_input)


def _mask_tools(history):
    _history = history.copy()
    for i, msg in enumerate(_history):
        if isinstance(msg, list):
            ai_msg, tool_msg = msg
            match tool_msg.tool_call_id:
                case 'vision_result':
                    _history[i] = [ai_msg, ToolMessage(content="Отримано фото", tool_call_id="vision_result")]
                case 'search_result':
                    _history[i] = [ai_msg, ToolMessage(content="Отримано дані пошуку", tool_call_id="search_result")]
                case 'set_user_name':
                    _history[i] = [ai_msg,
                                   ToolMessage(content="Змінено ім'я користувача", tool_call_id="set_user_name")]
                case 'set_bot_name':
                    _history[i] = [ai_msg,
                                   ToolMessage(content="Змінено твоє ім'я", tool_call_id="set_bot_name")]
                case _:
                    tool_call_id = ai_msg.tool_calls[0].get('id', "unknown")
                    _history[i] = [ai_msg,
                                   ToolMessage(content="НЕВІДОМИЙ ІНСТРУМЕНТ. ІГНОРУЙ ЦЕЙ ВИКЛИК",
                                               tool_call_id=tool_call_id)]
    return _history



@traceable(run_type="chain", name="Router")
async def handle_router(question: str, state: FSMContext, message: Message):
    """Обробка результату роутера: ставить відповідний стан"""
    state_data = await state.get_data()
    history = state_data.get("history",[])
    history.append(HumanMessage(question))

    # щоб не плутати модель маскуємо реальні виклики на просто підтвердження того що виклик був
    # плюс Router і Command моделі не обов'язково взагалі підтримують зображення
    _history = _mask_tools(history)
    messages = unpack_history(_history)

    # останнім повідомленням не може бути AIMessage
    while messages and isinstance(messages[-1], AIMessage):
        messages.pop()

    try:
        router_response = await run_router(messages)

        await state.update_data(history=history)

        match router_response.task:
            case "command":
                command_response = await run_command(question)
                await handle_command(command_response, state, message)
            case "answer":
                await handle_processor(router_response, state, message)

    except OutputParserException as e:
        error_message = "Вибачте, здається, виникла помилка на стороні серверу. Будь ласка, спробуйте ще раз"
        last_human_message = None
        for i, msg in reversed(list(enumerate(history))):
            if isinstance(msg,HumanMessage):
                    last_human_message = i
                    break

        # відкочуємо історію до поперереднього людського повідомлення.
        if last_human_message:
            history = history[:last_human_message]
        # якщо попереднє людське повідомлення - перше в історії, або його взагалі нема - історія стирається
        else:
            history = []

        await answer_to_user(message, error_message)
        await state.update_data(history=history)
        await state.set_state(UserState.wait_input)

        raise
    except Exception as e:
        error_message = ("Вибачте, я не можу обробити зображення. "
                        "Можливо, воно містить матеріали делікатного характеру")
        if "500" in str(e) or "Internal Server Error" in str(e):
            bad_image_index = None
            for i, msg in reversed(list(enumerate(history))):
                match msg:
                    case [_, ToolMessage(tool_call_id="vision_result")]:
                        bad_image_index = i
                        break

            if bad_image_index is not None:
                history[bad_image_index][1] = ToolMessage(content = error_message,
                                                      tool_call_id="vision_result")

                await answer_to_user(message, error_message)
                await state.update_data(history=history)
                await state.set_state(UserState.wait_input)
            else:
                logger.error("handle_router | vision_result not found in history")
                await state.set_state(UserState.wait_input)
        else:
            raise


@user.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    user_name = message.from_user.first_name
    default_bot_name = s.get("DEFAULT_BOT_NAME")
    await state.update_data(bot_name=default_bot_name, user_name=user_name)

    hello_message = (f"Вітаю {user_name} мене звуть {default_bot_name}. "
                     f"Я - ШІ-помічник для людей із вадами зору. "
                     f"Я можу подивитись на зображення і відповісти на ваше запитання щодо нього. "
                     f"Я також можу знайти якусь інформацію в інтернеті. "
                     f"Якщо вам не подобається моє ім'я, або те як я Вас називаю - попросіть змінити ім'я. "
                     f"Щоб розпочати роботу надішліть повідомлення, або фотографію."
                     f"Ви можете використовувати як текст так і аудіо - повідомлення.")

    await answer_to_user(message, hello_message)

    await state.set_state(UserState.wait_input)

@user.message(F.text.startswith("/"))
async def unknown_command(message: Message):
    await answer_to_user(message, "Невідома команда")

# Користувач вводить запитання текстом
@user.message(UserState.wait_input, F.text)
async def user_sent_text(message: Message, state: FSMContext):
    question = message.text
    if not is_valid_question_length(question, message):
        return

    await _handle_with_thinking_message(message,handle_router(question, state, message))


# Користувач відправив фото
@user.message(UserState.wait_input, F.photo)
async def user_sent_photo(message: Message, state: FSMContext, bot: Bot):
    image = await _get_image_from_message(message, bot)

    state_data = await state.get_data()
    history = state_data.get('history', [])
    human_msg = next((msg for msg in reversed(history) if isinstance(msg, HumanMessage)), None)
    question = message.caption

    if not question:
        if not human_msg:
            await answer_to_user(message, "Фото отримано. Тепер, будь ласка, задайте питання")
            await state.set_state(UserState.wait_input)
            await state.update_data(wait_image=image)
            return
        else:
            question = human_msg.content

    if not is_valid_question_length(question, message):
        return

    await state.update_data(wait_image=image)
    await _handle_with_thinking_message(message, handle_router(question, state, message))


# Користувач відправив питання у вигляді голосового повідомлення
@user.message(UserState.wait_input, F.voice)
async def user_sent_voice(message: Message, state: FSMContext, bot: Bot):
    buffer = await bot.download(message.voice.file_id)
    buffer.seek(0)
    question = await voice_to_text(buffer)

    if not is_valid_question_length(question, message):
        return

    await _handle_with_thinking_message(message, handle_router(question, state, message))


# Першим повідомленням користувач не відправив ні фото, ні текст, ні аудіо
@user.message(UserState.wait_input)
async def wait_input_default_handler(message: Message):
    await answer_to_user(message, "Вибачте, здається, я не можу це обробити, будь ласка, "
                               "спробуйте надіслати фото, текст, або аудіо")


# Роутер вирішив, що потрібне фото, а фото немає
@user.message(UserState.wait_image, F.photo)
async def cmd_wait_image(message: Message, state: FSMContext, bot: Bot):
    image = await _get_image_from_message(message, bot)

    state_data = await state.get_data()
    router_response = state_data["pending_router_response"]

    await state.update_data(wait_image=image, pending_router_response=None)
    await _handle_with_thinking_message(message, handle_processor(router_response, state, message))


# Очікувалося фото, а користувач відправив текст - обробити як текстове повідомлення
@user.message(UserState.wait_image, F.text)
async def cmd_wait_image_text(message: Message, state: FSMContext):
    await state.set_state(UserState.wait_input)
    await user_sent_text(message, state)


# Очікувалося фото, а користувач відправив голосове - обробити як голосове
@user.message(UserState.wait_image, F.voice)
async def cmd_wait_image_voice(message: Message, state: FSMContext):
    await state.set_state(UserState.wait_input)
    await user_sent_voice(message, state)


@user.message()
async def default_handler(message: Message):
    message_text = ("Щось пішло не так. будь ласка, надрукуйте команду "
                    "похила риска старт англійськими літерами щоб перезапустити бота")

    await answer_to_user(message, message_text)


@user.errors()
async def error_handler(event: ErrorEvent, bot):
    update = event.update
    # Витягуємо юзера і його повідомлення залежно від типу апдейту
    user = update.message.from_user
    user_input = update.message.text or update.message.caption or "[не текст]"

    user_info = (
        f"User ID: {user.id} | "
        f"Username: @{user.username} | "
        f"Ім'я: {user.first_name}"
    ) if user else "User: невідомо"

    logger.exception(
        f"Помилка: {event.exception}\n"
        f"{user_info}\n"
        f"Інпут: {user_input}\n",
        exc_info=event.exception
    )

    admin_id = int(os.getenv("ADMIN_ID"))
    await bot.send_message(admin_id, f"Помилка: {event.exception}\n{user_info}\nІнпут: {user_input}")