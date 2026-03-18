from datetime import datetime
from typing import Literal, Optional
import logging
logger = logging.getLogger("Chains")

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel

from Chains import models

class Router(BaseModel):
    """
    task - тип завдання "answer" або "command"
    search_query - пошуковий запит для допомоги
    is_vision_needed - чи потрібно попросити користувача надати зображення
    """

    task: Literal["answer","command"]
    search_query: Optional[str] = None
    is_vision_needed: Optional[bool] = None

async def run_router(history: list) -> Router:
    current_date = datetime.now()
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

    text_only_history = [m for m in history if isinstance(m, HumanMessage) or
                         (isinstance(m, AIMessage) and not m.tool_calls)]
    messages = [SystemMessage(content=system)] + text_only_history
    structured_model = models.router_model.with_structured_output(Router)

    logger.debug("Я думаю")
    response = await structured_model.ainvoke(messages)
    logger.debug("Я відповідаю")

    return response