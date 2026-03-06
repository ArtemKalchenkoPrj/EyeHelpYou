from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import base64
from Chains import models

from pydantic import BaseModel
from typing import Literal

import logging
logger = logging.getLogger("Chains")

class Intent(BaseModel):
    intent: Literal["set_name", "set_bot_name", "question", "specification", "search"]
    value: str | None = None

async def run_llm(history:list, question: str, image_bytes: bytes, user_name: str, bot_name: str = "Остап") -> Intent:
    structured_llm = models.mind_model.with_structured_output(Intent)

    system = f"""
    ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON
    Ти {bot_name} - помічник для людей із вадами зору. Користувач {user_name} надасть фото, якого фізично не може побачити і питання, яке
    його цікавить. Твоя задача - відповісти на це питання максимально змістовно і коротко. 
    Якщо фото має якісь вади, наприклад, надто засвічене, або затемнене, або об'єкт зображено на фото не повністю тощо,
    ти маєш надати користувачеві інструкції з покращення. 
    Наприклад "будь ласка, ввімкніть світло","будь ласка, перемістіть камеру лівіше" тощо.
    - Не використовуй надто важких слів;
    - Якщо питання стосується чогось небезпечного ти маєш застерегти користувача;
    - Відповідь може бути довжиною не більше 250 символів, але краще менше;
    - Інструкції з покращення фото мають бути чіткі, не містити зійвих слів;
    - Якщо фото неідеальне, але зчитати інформацію можливо - не надавай інструкцій, а просто відповідай на питання;
    - Якщо надіслане фото не потребує в покращенні не акцентуй увагу на його властивостях. Не кажи якісне воно чи ні;
    - Якщо це інструкції з покращення фото, кажи в intent значення specification.
    Наприклад: {{"intent":"specification", "value":"Ввімкніть світло"}}
    Також користувач може в запиті вказати якусь команду. Наприклад, "Тепер називай мене Наталія", 
    ти маєш повернути {{"intent":"set_name", "value":"Наталія"}}, або якщо користувач просто задає питання
    {{"intent":"question", "value":*Відповідь на питання*}}
    Користувач також може попросити тебе пошукати інформацію в інтернеті. Для цього використовуй {{"intent":"search", "value":*відповідний запит*}} запит на пошук в інтернеті має бути чітким.
    Перелік доступних команд: 
    - set_name: користувач хоче змінити своє ім'я. value - нове ім'я користувача
    - set_bot_name: користувач хоче змінити твоє ім'я. value - твоє нове ім'я
    - question: у запиті не знайдено ніяких специфічних команд - просто відповідаєш на питання. value - твоя відповідь на це питання
    - specification: користувач не надав на фото всієї потрібної інформації для відповіді. value - твої припущення щодо покращення
    - search: краще за все буде пошукати інформацію щодо цього питання в інтернеті. value - пошуковий запит
    """

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    messages: list [SystemMessage | HumanMessage | AIMessage | ToolMessage] = [SystemMessage(content=system)]

    for msg in history:
        if msg['role'] == 'human':
            messages.append(HumanMessage(content=msg["content"]))
        elif msg['role'] == 'tool':
            messages.append(ToolMessage(
                content=msg["content"],
                tool_call_id="search_result"
            ))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        {"type": "text", "text": question}
    ]))

    logger.debug("Я думаю")
    logger.debug(f"Над питанням {question}")

    response = await structured_llm.ainvoke(messages)

    logger.debug("Я відповідаю")
    logger.debug(response)

    return response