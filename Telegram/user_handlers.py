import base64
import os
import asyncio
import re
from datetime import datetime

from aiogram import Router, F, Bot
from aiogram.types import Message, ErrorEvent
from aiogram.filters.command import CommandStart
from aiogram.fsm.context import FSMContext
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langsmith import traceable

from Chains.processor_chain import search
from Chains.text_to_voice import answer_to_user
from utils import trim_history, unpack_history, logger
from Telegram.state import UserState
from Chains import *
import settings_manager as s
from Chains.command_chain import set_bot_name, set_user_name, set_answer_type

available_tools = {
    "set_user_name": set_user_name,
    "set_bot_name": set_bot_name,
    "set_answer_type": set_answer_type,
}

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
                                   min_question_length: int = None,
                                   max_question_length: int = None) -> bool:
    if min_question_length is None:
        min_question_length: int = s.get("MIN_QUESTION_LENGTH")
    if max_question_length is None:
        max_question_length: int = s.get("MAX_QUESTION_LENGTH")

    if len(question.strip())<=min_question_length:
        await answer_to_user(message, "Вибачте, я не можу відповісти на таке коротке запитання. "
                                      "Будь ласка, надайте більш розгорнуте питання")
        return False
    if len(question.strip())>=max_question_length:
        await answer_to_user(message, "Вибачте, я не можу відповісти на таке велике питання. "
                                      "Будь ласка, сформулюйте питання більш стисло")
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


def _keep_only_cyrillic_start(text: str) -> str:
    """Функція для чистки артефактів перед відповіддю"""
    # Шукаємо першу літеру кирилиці
    match = re.search(r'[а-яА-ЯёЁїЇіІєЄґҐ]', text)

    if match:
        # Повертаємо рядок, починаючи з позиції першої знайденої літери
        return text[match.start():].strip()

    # Якщо кирилиці взагалі немає (наприклад, тільки <0, back>)
    raise ValueError("Модель видала порожній рядок")


@traceable(run_type="chain", name="HandleProcessor")
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

    # Потрібне зображення - дістати його
    user_image = state_data.get("wait_image", None)
    if is_vision_needed or user_image:
        # Є? Записати в історію
        if user_image:
            unique_vision_id = f"vsn_{int(datetime.now().timestamp())}"

            # у випадку коли користувач надіслав декілька фото одним повідомленням
            if isinstance(user_image, list):
                image_content = []
                # для кожного повідомлення створюється результат
                for img in user_image:
                    image_content.append({
                        "type": "image",
                        "source_type": "base64",
                        "data": img,
                        "mime_type": "image/jpeg",
                    })

                history.append([
                    AIMessage(
                        content=f"Маю отримати зображення, яке стосується запитання від користувача.",
                        tool_calls=[{"id": unique_vision_id, "name": "vision", "args": {}}]
                    ),
                    ToolMessage(
                        content=image_content,
                        tool_call_id=unique_vision_id,
                    )
                ])
            else:
                history.append(
                    [AIMessage(
                        content=f"Маю отримати зображення, яке стосується запитання від користувача.",
                        tool_calls=[{"id": unique_vision_id, "name": "vision", "args": {}}]
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
                        tool_call_id=unique_vision_id,
                    )
                ])
            await state.update_data(history=trim_history(history), wait_image=None)

        # Нема? Зупинити процес. Попросити надати зображення
        else:
            await answer_to_user(message, "Здається, я не можу відповісти на це питання без зображення."
                                          "Чи не могли б ви його надати, будь ласка?")
            await state.set_state(UserState.wait_image)
            await state.update_data(history=trim_history(history), pending_router_response=router_response)
            return

    max_iterations = 3  # Максимальна кількість кроків "думок" бота
    current_step = 0

    while current_step < max_iterations:
        current_step += 1

        messages = unpack_history(history)

        response = await run_processor(bot_name, user_name, messages)

        if response.query:
            search_results = await search.ainvoke({"query": response.query, "max_results": 5})

            call_id = f"call_{int(datetime.now().timestamp())}_{current_step}"
            history.append(AIMessage(
                content=f"Шукаю інформацію: {response.query}",
                tool_calls=[{"name": "search", "args": {"query": response.query}, "id": call_id}]
            ))
            history.append(ToolMessage(content=search_results, tool_call_id=call_id))

            continue

        elif response.answer:
            final_text = _keep_only_cyrillic_start(response.answer)
            history.append(AIMessage(content=final_text))

            await answer_to_user(message, final_text)
            await state.update_data(history=trim_history(history))
            return

    # Якщо цикл завершився, а відповіді немає (перевищено ліміт)
    await answer_to_user(message, "На жаль, запит виявився занадто складним. Спробуйте уточнити питання.")


@traceable(run_type="chain", name="HandleCommand")
async def handle_command(ai_msg: AIMessage, state: FSMContext, message: Message):
    state_data = await state.get_data()

    history = state_data.get("history", [])

    logger.debug(f"ai_msg type: {type(ai_msg)}")
    logger.debug(f"ai_msg.content: {ai_msg.content}")
    logger.debug(f"ai_msg.tool_calls: {ai_msg.tool_calls}")
    logger.debug(f"ai_msg raw: {ai_msg}")

    if ai_msg.tool_calls:

        tool_call_pair: list[AIMessage|ToolMessage] = [ai_msg]

        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name in available_tools:
                tool_obj = available_tools[tool_name]
                try:
                    observation = await tool_obj.coroutine(**tool_args, state=state, message=message)
                    logger.info(f"Результат тула {tool_name}: {observation}")
                    tool_call_pair.append(ToolMessage(content=observation, tool_call_id=tool_id))

                except Exception as e:
                    logger.exception(f"Помилка в тулі {tool_name}: {e}")
            else:
                logger.warning(f"Команда {tool_name} не знайдена.")

        history.append(tool_call_pair)
    await state.update_data(history=trim_history(history))
    await state.set_state(UserState.wait_input)


@traceable(run_type="chain", name="HandleRouter")
async def handle_router(question: str, state: FSMContext, message: Message):
    """Обробка результату роутера: ставить відповідний стан"""
    state_data = await state.get_data()
    history = state_data.get("history", [])
    history.append(HumanMessage(question))

    try:
        router_response = await run_router(question)

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
            if isinstance(msg, HumanMessage):
                last_human_message = i
                break

        # відкочуємо історію до попереднього людського повідомлення.
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
                    case [AIMessage() as ai_msg, *other_tools]:
                        if any(tc.get('name') == 'vision' for tc in ai_msg.tool_calls):
                            bad_image_index = i
                            break

            if bad_image_index is not None:
                vision_id = history[bad_image_index][0].tool_calls[0].get('id')
                history[bad_image_index][1] = ToolMessage(
                    content=error_message,
                    tool_call_id=vision_id
                )

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

    await state.update_data(answer_type=s.get('ANSWER_TYPE'))
    user_name = message.from_user.first_name
    default_bot_name = s.get("DEFAULT_BOT_NAME")
    await state.update_data(bot_name=default_bot_name, user_name=user_name)

    hello_message = (f"Вітаю {user_name} мене звуть {default_bot_name}. "
                     f"Я - ШІ-помічник для людей із вадами зору. "
                     f"Я можу подивитись на зображення і відповісти на ваше запитання щодо нього. "
                     f"Я також можу знайти якусь інформацію в інтернеті. "
                     f"Якщо вам не подобається моє ім'я, або те як я Вас називаю - попросіть змінити ім'я. "
                     f"Щоб розпочати роботу надішліть повідомлення, або фотографію."
                     f"Ви можете використовувати як текст так і аудіо повідомлення."
                     f"Якщо хочете змінити тип моїх відповідей з голосових повідомлень на текст - просто попросіть.")

    await answer_to_user(message, hello_message)

    await state.set_state(UserState.wait_input)


@user.message(F.text.startswith("/"))
async def unknown_command(message: Message):
    await answer_to_user(message, "Невідома команда")


# Користувач вводить запитання текстом
@user.message(UserState.wait_input, F.text)
async def user_sent_text(message: Message, state: FSMContext):
    question = message.text
    if not await is_valid_question_length(question, message):
        return

    await _handle_with_thinking_message(message, handle_router(question, state, message))


# Користувач відправив фото
@user.message(UserState.wait_input, F.photo)
async def user_sent_photo(message: Message, state: FSMContext, bot: Bot, album:list[Message] = None):

    state_data = await state.get_data()
    history = state_data.get('history', [])
    human_msg = next((msg for msg in reversed(history) if isinstance(msg, HumanMessage)), None)
    question = message.caption

    if not album:
        image = await _get_image_from_message(message, bot)
    else :
        image = []
        for i, msg in enumerate(album):
            img = await _get_image_from_message(msg, bot)
            image.append(img)
            if not question and msg.caption:
                question = msg.caption

    if not question:
        if not human_msg:
            await answer_to_user(message, "Фото отримано. Тепер, будь ласка, задайте питання")
            await state.set_state(UserState.wait_input)
            await state.update_data(wait_image=image)
            return
        else:
            # Знаходимо індекс останнього human_msg
            human_msg_index = next((i for i, msg in reversed(list(enumerate(history)))
                                    if isinstance(msg, HumanMessage)), None)

            # Перевіряємо чи після нього є tool_call із виконанням команди
            commands = available_tools.keys()
            already_processed = False
            for msg_block in history[human_msg_index + 1:]:
                if isinstance(msg_block, list):
                    ai_msg = msg_block[0]
                    if isinstance(ai_msg, AIMessage) and ai_msg.tool_calls:
                        if any(tc['name'] in commands for tc in ai_msg.tool_calls):
                            already_processed = True
                            break

            if already_processed:
                await answer_to_user(message, "Будь ласка, задайте питання до фото")
                await state.set_state(UserState.wait_input)
                await state.update_data(wait_image=image)
                return

            question = human_msg.content

    if not await is_valid_question_length(question, message):
        return

    await state.update_data(wait_image=image)
    await _handle_with_thinking_message(message, handle_router(question, state, message))


# Користувач відправив питання у вигляді голосового повідомлення
@user.message(UserState.wait_input, F.voice)
async def user_sent_voice(message: Message, state: FSMContext, bot: Bot):
    buffer = await bot.download(message.voice.file_id)
    buffer.seek(0)
    question = await voice_to_text(buffer)

    if not await is_valid_question_length(question, message):
        return

    await _handle_with_thinking_message(message, handle_router(question, state, message))


# Першим повідомленням користувач не відправив ні фото, ні текст, ні аудіо
@user.message(UserState.wait_input)
async def wait_input_default_handler(message: Message):
    await answer_to_user(message, "Вибачте, здається, я не можу це обробити, будь ласка, "
                                  "спробуйте надіслати фото, текст, або аудіо")


# Роутер вирішив, що потрібне фото, а фото немає
@user.message(UserState.wait_image, F.photo)
async def cmd_wait_image(message: Message, state: FSMContext, bot: Bot, album:list[Message] = None):
    if not album:
        image = await _get_image_from_message(message, bot)
    else :
        image = []
        for msg in album:
            img = await _get_image_from_message(msg, bot)
            image.append(img)


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
    # Витягуємо користувача і його повідомлення
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
