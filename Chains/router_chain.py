from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langsmith import traceable

from Chains import models
from utils import logger


@traceable(run_type="llm", name="Router")
async def run_router(question: str) -> models.Router:
    current_date = datetime.now().strftime("%d.%m.%Y %H:%M")
    system = f"""
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON.
    Сьогодні {current_date}.


    ЄДИНІ ВАРІАНТИ ТИПУ "task":
    1. "command" — якщо користувач хоче:
       - Змінити своє ім'я або твоє ім'я.
       - Змінити режим відповіді (текст/голос).
       Приклад: {{"task":"command"}}

    2. "answer" — для ВСІХ інших випадків (питання, розмова, прохання щось описати).
       Приклад: {{"task":"answer"}}

    ЛОГІКА ВИЗНАЧЕННЯ "is_vision_needed":
    - true: якщо питання стосується конкретного об'єкта перед користувачем, який треба побачити, ідентифікувати або прочитати.
    - false: якщо це загальне питання, обговорення історії або просте спілкування.

    СТРУКТУРА ВІДПОВІДІ:
    - Для команд: {{"task":"command"}}
    - Для відповідей без фото: {{"task":"answer", "is_vision_needed":false}}
    - Для відповідей, де потрібне фото: {{"task":"answer", "is_vision_needed":true}}

    НІКОЛИ НЕ ВІДПОВІДАЙ ТЕКСТОМ, ТІЛЬКИ JSON.
    """

    messages = [SystemMessage(content=system), HumanMessage(content=question)]

    logger.debug("Я думаю")
    response = await models.router_model.ainvoke(messages)
    logger.debug("Я відповідаю")

    return response