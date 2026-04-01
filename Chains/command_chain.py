from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from Chains import models
from utils import logger

@traceable(run_type="llm", name="Command")
async def run_command(question: str) -> models.Command:
    system = """
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON
    Твоя задача - зрозуміти яку команду хоче виконати користувач і вказати аргументи цієї команди. Серед варіантів:
    - set_user_name: користувач хоче змінити своє ім'я/ім'я користувача.
    - set_bot_name: користувач хоче змінити твоє ім'я/ім'я бота.
    - set_answer_type: користувач хоче змінити тип виводу повідомлень/твоєї відповіді йому. Варіанти: voice, text.
        - voice якщо користувач просить відповідати голосовими, голосом, звуком тощо.
        - text якщо користувач просить відповідати текстом.
        - інших варіантів для аргументів цієї команди нема.
    Приклад відповіді: при запиті користувача "Називай мене Наталя" {"command":"set_user_name","command_argument":"Наталя"}
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON
    """
    messages = [SystemMessage(content=system), HumanMessage(content=question)]

    logger.debug("Я починаю обмірковувати команду")
    response = await models.command_model.ainvoke(messages)
    logger.debug("Я закінчую обмірковувати команду")

    return response