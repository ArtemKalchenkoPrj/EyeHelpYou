import asyncio
import logging
import math
from datetime import datetime, date, timedelta
from urllib.parse import urlparse

from RestrictedPython import compile_restricted, safe_globals, safe_builtins
from ddgs import DDGS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langsmith import traceable

import settings_manager as s
from Chains import models


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


def run_code_restricted(code: str) -> str:
    clean_code = code.replace("```python", "").replace("```", "").strip()

    try:
        byte_code = compile_restricted(clean_code, "<string>", "exec")
    except SyntaxError as e:
        return f"Syntax Error: {e}"

    allowed_globals = {
        **safe_globals,
        "__builtins__": safe_builtins,
        "math": math,
        "datetime": datetime,
        "date": date,
        "timedelta": timedelta,
        "abs": abs, "round": round,
        "min": min, "max": max,
        "sum": sum, "len": len,
    }
    local_vars = {}

    try:
        exec(byte_code, allowed_globals, local_vars)
        result = local_vars.get("result")
        if result is None:
            return "Код не повернув результату"
        return str(result)
    except Exception as e:
        return f"Runtime Error: {e}"


@tool
async def run_calculator(axioms: dict, question: str) -> str:
    """Функція для обчислення математичних операцій. Коли тобі потрібно виконати БУДЬ-ЯКІ обчислення - викликай цю функцію.
    Важливо: передавай питання повинно бути логічно сформоване та надано англійською мовою"""
    current_date = datetime.now()
    system = f"""Today is {current_date}. You are a high-precision Python Code Generator for a multi-agent mathematical system. 
    Your sole purpose is to transform unstructured data and user queries into executable Python code.
    
    ### OPERATIONAL RULES:
    1. OUTPUT ONLY CODE: Do not provide explanations, comments (unless necessary for logic), or markdown formatting like ```python ```. Return raw text only.
    2. RESULT VARIABLE: Always store the final result in a variable named `result`. Do not use print().
    3. DATA HANDLING: You will receive data in JSON format. Access this data as if it were a pre-defined dictionary or by manually defining the variables based on the provided context.
    4. LIBRARIES: The following are already available without importing: 
    `math`, `datetime`, `date`, `timedelta`. 
    Do NOT use import statements at all.
    5. ERROR HANDLING: If data is missing or a calculation is impossible (e.g., division by zero), set result to a clear error string like: result = "Error: division by zero".
    
    ### TASK:
    Based on the data provided axioms and question, write a script to calculate the exact answer.
    
    ### EXAMPLE SCENARIO:
    Context: {{"item": "Pork", "weight_g": 450, "price_total": 120}}
    User Question: "What is the price per kilogram?"
    Your Output:
    weight_kg = 450 / 1000
    price_per_kg = 120 / weight_kg
    result = round(price_per_kg, 2)"""

    def _extract_raw_code(content) -> str:
        # 1. Якщо це список (як у твоєму логу)
        if isinstance(content, list):
            # Шукаємо текстовий блок
            for block in content:
                if isinstance(block, dict) and 'text' in block:
                    return block['text']
                elif isinstance(block, str):
                    return block
            return ""

        # 2. Якщо це вже рядок
        if isinstance(content, str):
            return content

        return str(content)

    user_query = HumanMessage(
        content=f"Context:{axioms}\nUser Question:{question}"
    )
    messages = [SystemMessage(content=system), user_query]

    logger.debug("Я починаю рахувати")
    response = await models.calculator_model.ainvoke(messages)
    raw_code = _extract_raw_code(response.content)
    logger.debug(f"Отримано код: {raw_code}")

    # Замість просто raw_code, ми додаємо оголошення змінних зверху
    setup_code = ""
    for key, value in axioms.items():
        # Формуємо рядок виду: distance = 100
        if isinstance(value, str):
            setup_code += f"{key} = '{value}'\n"
        else:
            setup_code += f"{key} = {value}\n"

    full_code = setup_code + raw_code
    logger.debug(f"Запущено код: {full_code}")
    logger.debug("Починаю виконувати код")
    code_result = run_code_restricted(full_code)
    logger.debug("Завершую виконувати код")

    return code_result


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