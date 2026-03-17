from aiogram import Router, F, Bot
from aiogram.types import Message
from aiogram.filters.command import CommandStart
from aiogram.fsm.context import FSMContext

from Chains.text_to_voice import answer_to_user
from Chains.utils import *
from Telegram.state import UserState
from dotenv import load_dotenv

from Chains import *


load_dotenv()
user = Router()

async def _get_image_from_message(message, bot):
    """Функція для отримання картинки з повідомлення і конвертація її у формат, який LLM може прочитати"""
    photo = message.photo[-1].file_id
    buffer = await bot.download(photo)
    image_bytes = buffer.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64


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
    bot_name = state_data.get("bot_name",f"{message.from_user.first_name}")
    user_name = state_data.get("user_name","Остап")
    is_image_needed = router_response.is_vision_needed or False
    search_query = router_response.search_query or ""

    # Є пошуковий запит - виконати пошук - додати в історію
    if search_query:
        search_result = await search_web(search_query, 5, 'strong')
        history.append(AIMessage(
            content=f"Пошукаю інформацію в інтернеті за запитом: {search_query}",
            tool_calls=[{"id": "search_result", "name": "search", "args": {"search_query": search_query}}]
        ))
        history.append(ToolMessage(
            content=search_result,
            tool_call_id="search_result"
        ))

    # Потібне зображення - дістати його
    user_image = state_data.get("wait_image", None)
    if is_image_needed or user_image:
        # Є? Записати в історію
        if user_image:
            history.append(AIMessage(
                content=f"Маю отримати зображення, яке стосується запитання від користувача.",
                tool_calls=[{"id": "vision_result", "name": "vision", "args": {}}]
            ))
            history.append(ToolMessage(
                content=[
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": user_image,
                        "mime_type": "image/jpeg",
                    },
                ],
                tool_call_id="vision_result",
            ))
            await state.update_data(wait_image=None)
        # Нема? Зупинити процес. Попросити надати зображення
        else:
            await answer_to_user(message, "Здається, я не можу відповісти на це питання без зображення."
                                          "Чи не могли б ви його надати, будь ласка?")
            await state.set_state(UserState.wait_image)
            await state.update_data(pending_router_response=router_response)
            return

    await answer_to_user(message, "Мені треба подумати, це займе деякий час. Почекайте, будь ласка")
    history = await trim_history(history)

    logger.debug(f"History types: {[type(m).__name__ for m in history]}")
    for m in history:
        if isinstance(m, ToolMessage):
            logger.debug(f"ToolMessage content type: {type(m.content)}")
            if isinstance(m.content, list):
                logger.debug(f"ToolMessage content[0] keys: {m.content[0].keys()}")

    processor_response = await run_processor(bot_name, user_name, history)

    history.append(AIMessage(processor_response.value))

    match processor_response.task:
        case "question":
            await state.set_state(UserState.wait_input)
        case "specification":
            await state.set_state(UserState.specification)

    await answer_to_user(message, processor_response.value)
    await state.update_data(history = await trim_history(history))


async def handle_command(command_response: ChainCommand, state: FSMContext, message: Message):

    state_data = await state.get_data()

    history = state_data.get("history",[])

    match command_response.command:
        case "set_user_name":
            user_name = command_response.command_argument
            history.append(AIMessage(
                content=f"Зміню ім'я користувача з {state_data["user_name"]} на {user_name}",
                tool_calls=[{"id": "set_user_name", "name": "set_user_name", "args": {"name": user_name}}]
            ))
            history.append(ToolMessage(
                content=f"Тепер ім'я користувача - {user_name}",
                tool_call_id="set_user_name"
            ))
            await state.update_data(user_name=user_name)
            await answer_to_user(message, f"Добре, тепер я буду називати вас {user_name}")
        case "set_bot_name":
            bot_name = command_response.command_argument
            history.append(AIMessage(
                content=f"Зміню своє ім'я з {state_data["bot_name"]} на {bot_name}",
                tool_calls=[{"id": "set_bot_name", "name": "set_bot_name", "args": {"name": bot_name}}]
            ))
            history.append(ToolMessage(
                content=f"Тепер твоє ім'я - {bot_name}",
                tool_call_id="set_bot_name"
            ))
            await state.update_data(bot_name=bot_name)
            await answer_to_user(message, f"Добре, тепер мене звуть {bot_name}")

    await state.update_data(history = await trim_history(history))
    await state.set_state(UserState.wait_input)


async def handle_router(question: str, state: FSMContext, message: Message):
    """Обробка результату роутера: ставить відповідний стан"""
    state_data = await state.get_data()
    history = state_data.get("history",[])
    history.append(HumanMessage(question))
    router_response = await run_router(history)
    history = await trim_history(history)

    await state.update_data(history=history)

    match router_response.task:
        case "command":
            command_response = await run_command(question)
            await handle_command(command_response, state, message)
        case "answer":
            await handle_processor(router_response, state, message)


@user.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.update_data(bot_name="Остап", user_name=message.from_user.first_name)
    state_data = await state.get_data()
    user_name, bot_name = state_data['user_name'], state_data['bot_name']

    hello_message = (f"“Вітаю, {user_name} мене звуть {bot_name}. "
                     f"Я - ШІ-помічник для людей із вадами зору. "
                     f"Я можу подивитись на зображення і відповісти на ваше запитання щодо ного. "
                     f"Я також можу знайти якусь інформацію в інтернеті. "
                     f"Якщо вам не подобається моє ім’я, або те як я Вас називаю - попросіть змінити ім’я. "
                     f"Щоб розпочати роботу надішліть повідомлення, або фотографію.”")

    await answer_to_user(message, hello_message)

    await state.set_state(UserState.wait_input)


# Користувач вводить запитання текстом
@user.message(UserState.wait_input, F.text)
async def user_sent_text(message: Message, state: FSMContext):
    await handle_router(message.text, state, message)


# Користувач відправив фото
@user.message(UserState.wait_input, F.photo)
async def user_sent_photo(message: Message, state: FSMContext, bot: Bot):
    image = await _get_image_from_message(message, bot)
    await state.update_data(wait_image=image)

    question = message.caption

    if not question:
        await answer_to_user(message, "Фото отримано. Тепер, будь ласка, задайте питання")
        await state.set_state(UserState.wait_input)
        return

    await handle_processor(ChainRouter(
        task='answer',
        search_query=None,
        is_vision_needed=True
    ), state, message)


# Користувач відправив питання у вигляді голосового повідомлення
@user.message(UserState.wait_input, F.voice)
async def user_sent_voice(message: Message, state: FSMContext, bot: Bot):
    buffer = await bot.download(message.voice.file_id)
    question = await voice_to_text(buffer)
    await handle_router(question, state, message)


# Першим повідомленням користувач не відправив ні фото, ні текст, ні аудіо
@user.message(UserState.wait_input)
async def wait_input_default_handler(message: Message):
    await answer_to_user(message, "Вибачте, здається, я не можу це обробити, будь ласка, "
                               "спробуйте надіслати фото, текст, або аудіо")


# Роутер вирішив, що потрібне фото, а фото немає
@user.message(UserState.wait_image, F.photo)
async def cmd_wait_image(message: Message, state: FSMContext, bot: Bot):
    image = await _get_image_from_message(message, bot)

    await state.update_data(wait_image=image)

    state_data = await state.get_data()
    router_response = state_data["pending_router_response"]

    await handle_processor(router_response, state, message)


# Користувач додав питання до фото
@user.message(UserState.wait_question, F.text)
async def wait_question_text(message: Message, state: FSMContext, bot: Bot):
    raise NotImplementedError


# Користувач скинув нове фото замість того щоб поставити питання
@user.message(UserState.wait_question, F.photo)
async def wait_question_photo(message: Message, state: FSMContext, bot: Bot):
    raise NotImplementedError

# режим допомоги для зображення
@user.message(UserState.specification, F.photo)
async def cmd_specification(message: Message, state: FSMContext, bot: Bot):
    raise NotImplementedError



@user.message()
async def default_handler(message: Message):
    message_text = ("Щось пішло не так. будь ласка, надрукуйте команду "
                    "похила риска старт англійськими літерами щоб перезапустити бота")

    await answer_to_user(message, message_text)