import asyncio
import base64
import os
from typing import Literal
from functools import wraps

from aiogram import Router, F, Bot
from aiogram.types import Message, BufferedInputFile
from aiogram.filters.command import Command, CommandStart
from aiogram.fsm.context import FSMContext
from ddgs import DDGS
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from Chains.text_to_voice import answer_to_user
from Chains.utils import *
from Telegram.state import UserState
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

from Chains import *
from constants import *


load_dotenv()
user = Router()

async def handle_answer(router_response: ChainRouter,
                        state: FSMContext,
                        message: Message):
    """
    Функція для обробки результатів, отриманих з processor
    :param router_response: інформація із router
    :param state: стан aiogram
    :param message: конкретне повідомлення, спіймане хендлером
    :return:
    """
    search_query = router_response.search_query or ""

    state_data = await state.get_data()

    history = state_data.get("history",[])
    question = message.text or ""

    if not question:
        last_human = next((m for m in reversed(history) if isinstance(m, HumanMessage)), None)
        question = last_human.content if last_human else ""

    user_image = state_data.get("wait_image",None)
    bot_name = state_data.get("bot_name",f"{message.from_user.first_name}")
    user_name = state_data.get("user_name","Остап")
    is_image_needed = router_response.is_vision_needed or False

    if is_image_needed:
        if user_image:
            # імітація виконання інструменту для отримання зображення
            history.append(AIMessage(
                content=f"Маю отримати зображення, яке стосується запитання {question} від користувача.",
                tool_calls=[{"id": "vision_result", "name": "vision", "args": {"question": question}}]
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
        else:
            await answer_to_user(message, "Здається, я не можу відповісти на це питання без фото. "
                                       "Можете, будь ласка, надіслати його? "
                                       "Я допоможу якщо із фото будуть проблеми")
            await state.set_state(UserState.wait_image)
            await state.update_data(pending_router_response=router_response)
            return


    if search_query:
        await answer_to_user(message, "Мені потрібно пошукати в інтернеті. Відповідь може зайняти трішки більше часу ніж зазвичай")
        search_result = await search_web(search_query, 3, "strong")
        # імітація виконання пошуку
        history.append(AIMessage(
            content=f"Пошукаю інформацію в інтернеті за запитом: {search_query}",
            tool_calls=[{"id": "search_result", "name": "search", "args": {"search_query": search_query}}]
        ))
        history.append(ToolMessage(
            content=search_result,
            tool_call_id="search_result"
        ))

    await answer_to_user(message, "Мені треба подумати, це займе деякий час. Почекайте, будь ласка")
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


async def handle_router(router_response: ChainRouter, state: FSMContext, message: Message):
    question = message.text
    match router_response.task:
        case "command":
            command_response = await run_command(question)
            await handle_command(command_response, state, message)
        case "answer":
            await handle_answer(router_response, state, message)


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
    logger.debug("cmd_start: state: wait_input")


# користувач вводить перше запитання текстом
@user.message(UserState.wait_input, F.text)
async def user_text(message: Message, state: FSMContext):
    question = message.text
    state_data = await state.get_data()

    history = state_data.get("history",[])

    history.append(HumanMessage(question))
    history = await trim_history(history)
    await state.update_data(history=history)

    router_response = await run_router(history)

    await handle_router(router_response, state, message)


# першим повідомленням користувач надіслав фото
@user.message(UserState.wait_input, F.photo)
async def first_user_image(message: Message, state: FSMContext):
    raise NotImplementedError


# першим повідомленням користувач надіслав аудіо
@user.message(UserState.wait_input, F.voice)
async def first_user_voice(message: Message, state: FSMContext):
    raise NotImplementedError


# першим повідомленням користувач не відправив ні фото, ні текст, ні аудіо
@user.message(UserState.wait_input)
async def wait_input_default_handler(message: Message):
    await answer_to_user(message, "Вибачте, здається, я не можу це обробити, будь ласка, "
                               "спробуйте надіслати фото, текст, або аудіо")


# роутер вирішив, що потрібне фото, а фото немає
@user.message(UserState.wait_image, F.photo)
async def cmd_wait_image(message: Message, state: FSMContext, bot: Bot):
    image = await get_image_from_message(message, bot)

    await state.update_data(wait_image=image)

    state_data = await state.get_data()
    router_response = state_data["pending_router_response"]

    await handle_answer(router_response, state, message)


# режим допомоги для зображення
@user.message(UserState.specification, F.photo)
async def cmd_specification(message: Message, state: FSMContext, bot: Bot):
    """
    Хендлер для допомоги користувачу підібрати кадр. Історія повідомлень має обмеження. Тому необхідно її обрізати.
    Важливим є збереження питання людини (шукаємо останнє людське повідомлення) та пошук у мережі, якщо такий був для того,
    щоб модель могла дати відповідь на питання коли все ж таки потрібний кадр буде знайдено.
    :param message: Повідомлення захоплене хендлером
    :param state: Стан aiogram
    :param bot: Бот
    """

    state_data = await state.get_data()
    history = state_data['history']

    image = await get_image_from_message(message, bot)
    new_photo_call = AIMessage(
        content="Отримую нове фото від користувача",
        tool_calls=[{"id": "vision_result", "name": "vision", "args": {}}]
    )
    new_photo_tool = ToolMessage(
            content=[
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": image,
                    "mime_type": "image/jpeg",
                },
            ],
            tool_call_id="vision_result",
        )

    history = await trim_history(history) + [new_photo_call, new_photo_tool]

    await answer_to_user(message, "Мені треба подумати, це займе деякий час. Почекайте, будь ласка")
    processor_response = await run_processor(state_data["bot_name"], state_data["user_name"], history)

    history.append(AIMessage(content=processor_response.value))

    await state.update_data(history=await trim_history(history))

    match processor_response.task:
        case "answer":
            await answer_to_user(message, processor_response.value)
            await state.set_state(UserState.wait_input)
            logger.debug("cmd_specification: state: wait_input")
        case "specification":
            await answer_to_user(message, processor_response.value)
            await state.set_state(UserState.specification)
            logger.debug("cmd_specification: state: specification")


@user.message()
async def default_handler(message: Message):
    message_text = ("Щось пішло не так. будь ласка, надрукуйте команду "
                    "похила риска старт англійськими літерами щоб перезапустити бота")

    await answer_to_user(message, message_text)