from typing import Literal
import logging
logger = logging.getLogger("Chains")

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from Chains import models


class Command(BaseModel):
    """
    command - команда
    command_argument - аргумент команди

    Достпні команди: set_user_name, set_bot_name
    """
    command: Literal["set_user_name","set_bot_name"]
    command_argument: str

async def run_command(question: str) -> Command:
    system = """
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON
    Твоя задача - зрозуміти яку команду хоче виконати користувач і вказати аргументи цієї команди. Серед варіантів:
    - set_user_name: користувач хоче змінити своє ім'я/ім'я користувача
    - set_bot_name: користувач хоче змінити твоє ім'я/ім'я бота
    Приклад відповіді: {"command":"set_user_name","command_argument":"Наталя"}
    """
    structured_model = models.command_model.with_structured_output(Command)
    messages = [SystemMessage(content=system), HumanMessage(content=question)]

    logger.debug("Я починаю обмірковувати команду")
    response = await structured_model.ainvoke(messages)
    logger.debug("Я закінчую обмірковувати команду")

    return response