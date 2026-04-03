from datetime import datetime

from langchain_core.messages import SystemMessage
from langsmith import traceable

from Chains import models
from utils import logger

@traceable(run_type="llm", name="Router")
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
    - Відповідати текстом/голосовим: {{"task":"command"}}
    - Змінити тип виводу на текст/голосове: {{"task":"command"}}
    
    ВІДПОВІДАЙ command ВИКЛЮЧНО КОЛИ КОРИСТУВАЧ ХОЧЕ ЗМІНИТИ ТВОЄ АБО СВОЄ ІМ'Я АБО ТИП ВИВОДУ АБО ВІДПОВІДІ. ВСЕ ІНШЕ - answer
    Якщо користувач хоче отримати відповідь на питання визнач чи потрібно виконати пошук в інтернеті
    для отримання цієї інформації та чи потрібно користувачеві надати фото для того щоб отримати відповідь.
    Якщо для відповіді на питання про об'єкт на фото потрібен пошук (наприклад, термін придатності ліків), вкажи і is_vision_needed: true, і search_query
    Для пошуку в інтернеті надай короткий пошуковий запит. Наприклад: {{"search_query":"хто зараз президент України"}}.
    Те що ти надав search_query вже означає виконання пошуку в інтернеті. 
    Фото потрібне тільки якщо питання стосується конкретного об'єкту який бачить користувач.
    Фото НЕ потрібне якщо питання стосується загальних знань.
    Приклад відповіді якщо потрібне фото: {{"task":"answer","is_vision_needed":true}}
    НІКОЛИ НЕ ВІДПОВІДЙ ПРОСТО ТЕКСТОМ ВИКОРИСТОВУЙ JSON
    """

    messages = [SystemMessage(content=system)] + history

    logger.debug("Я думаю")
    response = await models.router_model.ainvoke(messages)
    logger.debug("Я відповідаю")

    return response