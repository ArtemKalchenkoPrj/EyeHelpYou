import base64
from datetime import datetime
from typing import Literal
import logging

from aiogram.fsm.context import FSMContext
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel

from Chains import models
from constants import MAX_MESSAGE_MEMORY
logger = logging.getLogger("Chains")

class Intent(BaseModel):
    task: Literal["question","specification"]
    value: str

async def run_processor(state: FSMContext, search_query: str, search_result: str, is_vision_needed:bool, max_message_memory = MAX_MESSAGE_MEMORY) -> Intent:
    structured_llm = models.vision_model.with_structured_output(Intent)
    state_data = await state.get_data()
    current_date = datetime.now()

    history = state_data.get("history", [])

    system = f"""
        ВІДПОВІДАЙ ВИКЛЮЧНО У ФОРМАТІ JSON
        Сьогодні {current_date}.
        Ти {state_data["bot_name"]} - помічник для людей із вадами зору.
        Користувач {state_data["user_name"]} """

    # Модифікатор vision
    if is_vision_needed:
        system += """надасть фото, якого фізично не може побачити і питання на яке ти маєш відповісти.
        Твоя відповідь не повинна перевищувати 250 символів.
        Якщо ваша з ним розмова стосується чогось небезпечного - повідом користувача про це.
        Якщо фото має якісь вади - надай інструкції з покращення.
        Очікуваний вивід для інструкцій: {"task":"specification","value":*інструкції*}"""

        image_base64 = base64.b64encode(state_data["user_photo"]).decode("utf-8")
        history.append(HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": state_data["user_question"]}
        ]))
    else:
        history.append(HumanMessage(content=state_data["user_question"]))
        system += "задасть питання на яке ти маєш відповісти. Твоя відповідь не повинна перевищувати 250 символів."

    # Модифікатор search
    if search_result:
        system += "\nЗа потреби використовуй інформацію із інтернету"
        history.append(AIMessage(
            content=f"Пошукаю інформацію в інтернеті за запитом: {search_query}",
            tool_calls=[{"id": "search_result", "name": "search", "args": {}}]
        ))
        history.append(ToolMessage(
            content=search_result,
            tool_call_id="search_result"
        ))

    system += f"""
        Очікуваний вивід для відповіді: {{"task":"question","value":*відповідь*}}
        """

    # Збираємо повідомлення
    messages: list[SystemMessage | AIMessage | HumanMessage | ToolMessage] = [SystemMessage(content=system)] + history

    logger.debug("Я починаю думати")
    response = await structured_llm.ainvoke(messages)
    logger.debug("Я закінчую думати")

    history.append(AIMessage(content=response.value))

    # Щоб інпут до моделі не перевищував max_message_memory повідомлень
    if len(history) > max_message_memory-3:
        history = history[-max_message_memory-3:]
    await state.update_data(history=history)

    return response