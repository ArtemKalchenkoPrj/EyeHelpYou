from datetime import datetime
from typing import Literal
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel

from Chains import models
logger = logging.getLogger("Chains")

class Intent(BaseModel):
    task: Literal["question","specification"]
    value: str

async def run_processor(bot_name: str,
                        user_name: str,
                        history: list) -> Intent:

    structured_llm = models.vision_model.with_structured_output(Intent)
    current_date = datetime.now()

    system = f"""
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON. НІЯКИХ ІНШИХ СИМВОЛІВ КРІМ JSON.
    Сьогодні {current_date}.
    Ти {bot_name} - помічник для людей із вадами зору.
    Користувач {user_name} задасть питання. Якщо до питання додане фото - використай його для відповіді.
    Твоя відповідь не повинна перевищувати 250 символів.
    Якщо розмова стосується чогось небезпечного - повідом користувача про це.
    Якщо пошукова інформація надана - використай її у відповіді.
    Якщо фото нечітке, змилене, темне, або будь-яким чином заважає розпізнаванню - ОБОВ'ЯЗКОВО повертай specification.
    Краще попросити краще фото ніж дати неточну відповідь.
    НІКОЛИ не повертає specification якщо не бачиш фото. Без фото ЗАВЖДИ відповідай ніби це звичайне запитання - question.
    
    Важливо: користувач НЕ БАЧИТЬ своє фото. Інструкції мають бути у вигляді конкретних дій:
    - замість "поверніть камеру" → "нахиліть телефон лівіше" або "нахиліть телефон правіше"
    - замість "наблизьте камеру" → "підніміть телефон вище" або "опустіть телефон нижче"
    - замість "сфотографуйте чіткіше" → "притримайте телефон двома руками і не рухайтесь"
    
    Очікуваний вивід для відповіді: {{"task":"question","value":*відповідь*}}
    Очікуваний вивід для випадку коли ти не можеш розібрати фото: {{"task":"specification","value":*інструкції з покращення фото*}}
    Навіть якщо є посилання чи додаткова інформація — вклади все в поле value як текст.
    """

    messages: list[SystemMessage | AIMessage | HumanMessage | ToolMessage] = [SystemMessage(content=system)] + history

    logger.debug("Я починаю думати")
    response = await structured_llm.ainvoke(messages)
    logger.debug("Я закінчую думати")

    return response