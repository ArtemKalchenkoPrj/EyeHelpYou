from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
import base64
from Chains import models

from pydantic import BaseModel, Field
from typing import Literal

class Intent(BaseModel):
    intent: Literal["set_name", "set_bot_name", "question"]
    value: str | None = None

async def run_llm(question: str, image_bytes: bytes, user_name: str, bot_name: str = "Остап") -> Intent:
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
    - Відповідь може бути довжиною 250 символів максимум;
    - Інструкції з покращення фото мають бути чіткі, не містити зійвих слів;
    - Якщо фото неідеальне, але зчитати інформацію можливо - не надавай інструкцій, а просто відповідай на питання;
    - Якщо надіслане фото не потребує в покращенні не акцентуй увагу на його властивостях. Не кажи якісне воно чи ні;
    Також користувач може в запиті вказати якусь команду. Наприклад, "Тепер називай мене Наталія", 
    ти маєш повернути {{"intent":"set_name", "value":"Наталія"}}, або якщо користувач просто задає питання
    {{"intent":"question", "value":*Відповідь на питання*}}
    Перелік доступних команд: 
    - set_name: користувач хоче змінити своє ім'я
    - set_bot_name: користувач хоче змінити твоє ім'я
    - question: у запиті не знайдено ніяких специфічних команд - просто відповідаєш на питання. value - твоя відповідь на це питання
    
    value - текст команди
    """

    if image_bytes:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": question
                }
            ])
        ]
    else:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": question
                }
            ])
        ]

    response = await structured_llm.ainvoke(messages)
    return response