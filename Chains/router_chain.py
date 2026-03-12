from datetime import datetime
from typing import Literal, Optional
import logging
logger = logging.getLogger("Chains")

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from Chains import models

class Router(BaseModel):
    task: Literal["answer","command"]
    search_query: Optional[str] = None
    is_vision_needed: Optional[bool] = None

async def run_router(question: str):
    current_date = datetime.now()
    system = f"""
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON
    Сьогодні {current_date}
    Твоя задача - визначити що потрібно користувачеві.
    - Відповідь на питання
    - Змінити ім'я користувача: {{"task":"command"}}
    - Змінити ім'я бота (твоє ім'я): {{"task":"command"}}
    Якщо користувач хоче отримати відповідь на питання визнач чи потрібно виконати пошук в інтернеті
    для отримання цієї інформації та чи потрібно користувачеві надати фото для того щоб отримати відповідь.
    Для пошуку в інтернеті надай короткий пошуковий запит.
    Очікувані відповіді: {{"task":"answer", "is_vision_needed": true}}, 
    {{"task":"answer", "search_query":"Погода в Харкові сьогодні"}}
    """
    messages = [SystemMessage(content=system), HumanMessage(content=question)]
    structured_model = models.router_model.with_structured_output(Router)

    logger.debug("Я думаю")
    response = await structured_model.ainvoke(messages)
    logger.debug("Я відповідаю")

    return response