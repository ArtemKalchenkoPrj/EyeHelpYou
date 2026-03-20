import re
from datetime import datetime
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from Chains import models
logger = logging.getLogger("Chains")

# Додатковий захист, щоб модель не повертала надто багато символів
def trim_to_last_sentence(text: str, max_length: int = 250) -> str:
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    # Шукаємо останній знак кінця речення
    last_end = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )

    if last_end == -1:
        # Знаків кінця речення немає — обрізаємо по останньому пробілу
        last_space = truncated.rfind(" ")
        return truncated[:last_space] + "..." if last_space != -1 else truncated

    return truncated[:last_end + 1]

# невеличка допомога від ін'єкцій
def sanitize_name(name: str, max_length: int = 50) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\s\-']", "", name)
    return name[:max_length]

async def run_processor(bot_name: str,
                        user_name: str,
                        history: list):

    current_date = datetime.now().strftime("%d.%m.%Y %H:%M")

    system = f"""
    Сьогодні {current_date}.
    Ти помічник для людей із вадами зору.
    Користувач задасть питання. Якщо до питання додане фото - використай його для відповіді.
    Твоя відповідь не повинна перевищувати 250 символів.
    Якщо розмова стосується чогось небезпечного - повідом користувача про це.
    Якщо фото нечітке, змилене, темне, або будь-яким чином заважає розпізнаванню допоможи користувачу інструкціями з покращення.
    Краще попросити краще фото ніж дати неточну відповідь.
    
    Важливо: користувач НЕ БАЧИТЬ своє фото. Інструкції мають бути у вигляді конкретних дій:
    - замість "поверніть камеру" → "нахиліть телефон лівіше" або "нахиліть телефон правіше"
    - замість "наблизьте камеру" → "підніміть телефон вище" або "опустіть телефон нижче"
    - замість "сфотографуйте чіткіше" → "притримайте телефон двома руками і не рухайтесь"
    Якщо бачиш декілька фото - останнє має пріоритет
    """
    bot_name = sanitize_name(bot_name)
    user_name = sanitize_name(user_name)

    messages: list[SystemMessage | AIMessage | HumanMessage | ToolMessage] = ([SystemMessage(content=system)] +
                                                                              [HumanMessage(content=f"Тебе звати - {bot_name}")] +
                                                                              [HumanMessage(content=f"Мене звати - {user_name}")] +
                                                                              history)

    logger.debug("Я починаю думати")
    response = await models.vision_model.ainvoke(messages)
    logger.debug("Я закінчую думати")

    return response.content