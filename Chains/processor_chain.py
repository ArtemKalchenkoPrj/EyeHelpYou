from datetime import datetime
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langsmith import traceable

from Chains import models
import settings_manager as s


logger = logging.getLogger("Chains")

@traceable(run_type="llm", name="Processor")
async def run_processor(bot_name: str,
                        user_name: str,
                        history: list):

    current_date = datetime.now()
    max_answer_length = s.get("MAX_ANSWER_LENGTH")
    system = f"""
    Сьогодні {current_date}.
    Ти {bot_name} - помічник для людей із вадами зору.
    Користувач {user_name} задасть питання. Якщо до питання додане фото - використай його для відповіді.
    Якщо розмова стосується чогось небезпечного - повідом користувача про це.
    Якщо фото нечітке, змилене, темне, або будь-яким чином заважає розпізнаванню допоможи користувачу інструкціями з покращення.
    Краще попросити краще фото ніж дати неточну відповідь.
    
    Важливо: користувач НЕ БАЧИТЬ своє фото. Інструкції мають бути у вигляді конкретних дій:
    - замість "поверніть камеру" → "нахиліть телефон лівіше" або "нахиліть телефон правіше"
    - замість "наблизьте камеру" → "підніміть телефон вище" або "опустіть телефон нижче"
    - замість "сфотографуйте чіткіше" → "притримайте телефон двома руками і не рухайтесь"
    Якщо бачиш декілька фото - останнє має пріоритет
    ТВОЯ ВІДПОВІДЬ НЕ МАЄ ПЕРЕВИЩУВАТИ {max_answer_length} СИМВОЛІВ
    """

    messages: list[SystemMessage | AIMessage | HumanMessage | ToolMessage] = [SystemMessage(content=system)] + history

    logger.debug("Я починаю думати")
    response = await models.vision_model.ainvoke(messages)
    logger.debug("Я закінчую думати")

    content = response.content
    if isinstance(content, list):
        content = " ".join(block.get("text", "") for block in content if block.get("type") == "text")
    return content