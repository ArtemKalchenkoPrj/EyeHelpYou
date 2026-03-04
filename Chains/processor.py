from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
import base64
from constants import *

async def run_llm(question: str, image_bytes: bytes, user_name: str, bot_name: "Остап") -> str:
    llm = ChatOllama(model="qwen3.5:397b-cloud", temperature=0)

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    system = f"""
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
    - Якщо це інструкція з уточнення ти обов'язково маєш на початку відповіді вказати ключове слово "УТОЧНЕННЯ", якщо відповідь - "ВІДПОВІДЬ"
    """

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
    response = await llm.ainvoke(messages)
    return response.content