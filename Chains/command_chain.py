from langchain_core.messages import HumanMessage, SystemMessage

from Chains import models
from utils import logger


async def run_command(question: str) -> models.Command:
    system = """
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON
    Твоя задача - зрозуміти яку команду хоче виконати користувач і вказати аргументи цієї команди. Серед варіантів:
    - set_user_name: користувач хоче змінити своє ім'я/ім'я користувача
    - set_bot_name: користувач хоче змінити твоє ім'я/ім'я бота
    Приклад відповіді: {"command":"set_user_name","command_argument":"Наталя"}
    """
    messages = [SystemMessage(content=system), HumanMessage(content=question)]

    logger.debug("Я починаю обмірковувати команду")
    response = await models.command_model.ainvoke(messages)
    logger.debug("Я закінчую обмірковувати команду")

    return response