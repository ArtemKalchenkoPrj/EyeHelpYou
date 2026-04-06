from typing import Literal, Annotated

from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field

from Chains import models
from Chains.text_to_voice import answer_to_user
from utils import logger, current_answer_type
import settings_manager as s


class SetUserNameArgs(BaseModel):
    """Схема аргументів для зміни імені користувача."""
    name: str = Field(
        description="Нове ім'я користувача. Має бути рядком, бажано в називному відмінку."
    )

@tool("set_user_name", args_schema=SetUserNameArgs)
async def set_user_name(name: Annotated[str, Field(description="Нове ім'я")], state: FSMContext, message: Message) -> str:
    """
    Змінює ім'я користувача в системі.
    Викликай це, коли користувач представляється або просить змінити його ім'я.
    """
    logger.info("set_user_name викликано")
    state_data = await state.get_data()
    old_name = state_data.get("user_name", "невідомо")

    await state.update_data(user_name=name)

    await answer_to_user(message,f"Успішно: ваше ім'я змінено з {old_name} на {name}.")

    return f"Успішно: ім'я змінено з {old_name} на {name}."


class SetBotNameArgs(BaseModel):
    """Схема аргументів для зміни імені бота (твого імені)."""
    name: str = Field(
        description="Твоє нове ім'я. Має бути рядком, бажано в називному відмінку."
    )

@tool("set_bot_name", args_schema=SetBotNameArgs)
async def set_bot_name(name: Annotated[str, Field(description="Нове ім'я")], state: FSMContext, message: Message) -> str:
    """
    Змінює ім'я бота (твоє ім'я) в системі.
    Викликай це, виключно коли користувач прямо просить змінити твоє ім'я, або ім'я бота.
    Не викликай коли користувач просто до тебе інакше звертається.
    В цьому випадку можеш запропонувати змінити ім'я.
    """
    state_data = await state.get_data()
    old_bot_name = state_data.get("bot_name", "невідомо")

    await state.update_data(bot_name=name)

    await answer_to_user(message, f"Успішно: моє ім'я змінено з {old_bot_name} на {name}.")

    return f"Успішно: твоє ім'я змінено з {old_bot_name} на {name}."


class SetAnswerTypeArgs(BaseModel):
    """Схема аргументів зміни типу відповіді."""
    answer_type: Literal['voice','text'] = Field(
        description="Тип відповіді, яку запитав користувач. voice - для голосу, text - для текстової відповіді"
    )

@tool("set_answer_type", args_schema=SetAnswerTypeArgs)
async def set_answer_type(answer_type: Literal['voice','text'], state: FSMContext, message: Message) -> str:
    """Встановлює тип, відповіді, яку отримує користувач.
    voice - голосом
    text - текстом
    Викликай цю функцію коли користувач просить відповідати певним типом.
    """
    state_data = await state.get_data()
    old_answer_type = state_data.get("answer_type", s.get('ANSWER_TYPE'))

    await state.update_data(answer_type=answer_type)
    current_answer_type.set(answer_type)

    await answer_to_user(message, f"Успішно: тип відповіді змінено з {old_answer_type} на {answer_type}.")

    return f"Успішно: тип відповіді змінено з {old_answer_type} на {answer_type}."


@traceable(run_type="llm", name="Command")
async def run_command(question: str):
    system = """
    Ти - адміністративний модуль асистента для людей із вадами зору. Твоє єдине завдання — допомагати користувачу керувати налаштуваннями бота.

    Твої інструкції:
    1. Аналізуй запит користувача і використовуй відповідний інструмент (tool), якщо він хоче змінити своє ім'я, твоє ім'я або тип відповіді.
    2. Якщо запит користувача не стосується налаштувань (наприклад, він просто вітається або щось запитує), НЕ викликай інструменти, а просто коротко дай відповідь або підтвердь виконання.
    3. Тобі не потрібно пояснювати формат відповіді — просто викликай функцію з потрібними аргументами.
    4. Якщо користувач просить змінити тип відповіді, чітко розрізняй 'voice' (голос, звук) та 'text' (текст, повідомлення).
    
    Працюй лаконічно.
    """
    messages = [SystemMessage(content=system), HumanMessage(content=question)]

    logger.debug("Я починаю обмірковувати команду")
    response = await models.command_model.ainvoke(messages)
    logger.debug("Я закінчую обмірковувати команду")

    return response