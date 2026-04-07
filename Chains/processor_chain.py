import asyncio
from datetime import datetime
import logging
from urllib.parse import urlparse

from ddgs import DDGS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field

from Chains import models
import settings_manager as s



@tool
async def search(query:str, max_results:int):
    """Функція для пошуку інформації в мережі.
    Викликай її тільки коли запит користувача потребує цього.
    Якщо запит стосується чогось простого, що не потребує уточнення - не викликай.
    Викликай цю функцію якщо запит користувача є актуальним відносно часу. Наприклад:
    Користувач: хто зараз президент України? пошуковий запит: президент України *поточна дата*
    """

    logger.debug("search_web")
    logger.debug(f"Шукаю в інтернеті інформацію за запитом {query}")

    def _search():
        with DDGS() as ddgs:
            res = list(ddgs.text(
                query,
                max_results=max_results * 2,  # після фільтрації залишаться не всі посилання
                region="ua-uk",
                safesearch="moderate"
            ))
        return res

    results = await asyncio.to_thread(_search)

    filtered = [r for r in results if not urlparse(r['href']).netloc.endswith('.ru')]
    filtered = [r for r in filtered if 'ы' not in r['body'].lower() and 'ы' not in r['title'].lower()]
    filtered = [r for r in filtered if 'ё' not in r['body'].lower() and 'ё' not in r['title'].lower()]
    filtered = filtered[:max_results]

    full_text = ""
    for r in filtered:
        full_text += f"Заголовок: {r['title']}\n"
        full_text += f"Текст: {r['body']}\n"
        full_text += f"Посилання: {r['href']}\n\n"

    logger.debug(f"Ось що я знайшов: {full_text}")
    return full_text


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
    
    Також пам'ятай, що ти - модель орієнтована на сліпих. Ти маєш видавати результат в такому вигляді, щоб синтезатор голосу зміг його прочитати.
    Тобто:
    - замість 7С - сім градусів Цельсію
    - замість мл. - мілілітрів/мілілітр/мілілітри
    - замість г. - грамів/грам/грами
    - замість %  - відсотків/відсоток/відсотки
    Але посилання надавай в звичайному вигляді, без форматування, просто посилання так як воно є
    
    Якщо зображення настільки погане, що ти не можеш навіть визначити категорію об'єкта, запитай користувача: 
    'Здається, камера закрита або дуже розмита, ви намагаєтесь сфотографувати текст чи предмет?
    ТВОЯ ВІДПОВІДЬ НЕ МАЄ ПЕРЕВИЩУВАТИ {max_answer_length} СИМВОЛІВ
    """

    messages: list[SystemMessage | AIMessage | HumanMessage | ToolMessage] = [SystemMessage(content=system)] + history

    logger.debug("Я починаю думати")
    response = await models.vision_model.ainvoke(messages)
    logger.debug(f"Отримано структуру: {type(response)}")

    return response