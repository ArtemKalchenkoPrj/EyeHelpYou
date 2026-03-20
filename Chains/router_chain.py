from datetime import datetime

from langchain_core.messages import SystemMessage

from Chains import models
from utils import logger


async def run_router(history: list) -> models.Router:
    current_date = datetime.now().strftime("%d.%m.%Y %H:%M")
    system = f"""
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON
    Сьогодні {current_date}
    Твоя задача - визначити що потрібно користувачеві.
    ЄДИНІ ВАРІАНТИ ЗАВДАНЬ:
    - Відповідь на питання {{"task":"answer"}}
    - Змінити ім'я користувача: {{"task":"command"}}
    - Змінити ім'я бота (твоє ім'я): {{"task":"command"}}
    ВІДПОВІДАЙ command ВИКЛЮЧНО КОЛИ КОРИСТУВАЧ ХОЧЕ ЗМІНИТИ ТВОЄ АБО СВОЄ ІМ'Я. ВСЕ ІНШЕ - answer
    Якщо користувач хоче отримати відповідь на питання визнач чи потрібно виконати пошук в інтернеті
    для отримання цієї інформації та чи потрібно користувачеві надати фото для того щоб отримати відповідь.
    Для пошуку в інтернеті надай короткий пошуковий запит.
    Фото потрібне тільки якщо питання стосується конкретного об'єкту який бачить користувач.
    Фото НЕ потрібне якщо питання стосується загальних знань.
    
    Очікувані відповіді: {{"task":"answer", "is_vision_needed": true}}, 
    {{"task":"answer", "search_query":"Погода в Харкові сьогодні"}}
    """

    messages = [SystemMessage(content=system)] + history

    logger.debug("Я думаю")
    response = await models.router_model.ainvoke(messages)
    logger.debug("Я відповідаю")

    return response