from typing import Literal
import logging
logger = logging.getLogger("Chains")

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from Chains import models


class Command(BaseModel):
    command: Literal["set_user_name","set_bot_name"]
    command_argument: str

async def run_command(question: str):
    system = """
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON
    Твоя задача - зрозуміти яку команду хоче виконати користувач і вказати аргументи цієї команди. Серед варіантів:
    - set_user_name: користувач хоче змінити своє ім'я/ім'я користувача
    - set_bot_name: користувач хоче змінити твоє ім'я/ім'я бота
    Приклад відповіді: {"command":"set_user_name","command_argument":"Наталя"}
    """
    structured_model = models.command_model.with_structured_input(Command)
    messages = [SystemMessage(content=system), HumanMessage(content=question)]

    logger.debug("Я починаю обмірковувати команду")
    response = await structured_model.ainwoke(messages)
    logger.debug("Я закінчую обмірковувати команду")

    return response