from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from Chains import models
from utils import logger

@traceable(run_type="llm", name="Command")
async def run_command(question: str) -> models.Command:
    system = """
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON
    Твоя задача - зрозуміти яку команду хоче виконати користувач і вказати аргументи цієї команди. Серед варіантів:
    - set_user_name: користувач хоче змінити своє ім'я/ім'я користувача
    - set_bot_name: користувач хоче змінити твоє ім'я/ім'я бота
    Приклад відповіді: при запиті користувача "Називай мене Наталя" {"command":"set_user_name","command_argument":"Наталя"}
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON
    """
    messages = [SystemMessage(content=system), HumanMessage(content=question)]

    logger.debug("Я починаю обмірковувати команду")
    response = await models.command_model.ainvoke(messages)
    logger.debug("Я закінчую обмірковувати команду")

    return response