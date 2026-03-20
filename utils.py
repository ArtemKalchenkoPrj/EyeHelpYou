import asyncio
import logging
from urllib.parse import urlparse

from ddgs import DDGS
from langchain_core.messages import HumanMessage, AIMessage

from constants import MAX_MESSAGE_MEMORY


logger = logging.getLogger("Chains")

async def search_web(query: str, max_results: int=5) -> str:
    """
    Функція для пошуку інформації в мережі.
    :param query: Пошуковий запит
    :param max_results: Максимальна кількість запитів
    :return: Результати запитів: Заголовок статті, текст статті, посилання на статтю
    """

    logger.debug("search_web")
    logger.debug(f"Шукаю в інтернеті інформацію за запитом {query}")
    def _search():
        with DDGS() as ddgs:
            res = list(ddgs.text(
                query,
                max_results=max_results*2, # після фільтрації залишаться не всі посилання
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


def trim_history(history: list, max_message_memory: int = MAX_MESSAGE_MEMORY) -> list:
    """
    Обрізає історію повідомлень до max_message_memory. Залишає важливі для контексту повідомлення: попереднє повідомлення від
    ШІ, попереднє повідомлення від користувача, попередній AiMessage + ToolMessage з результатами пошуку в мережі,
    попередні AiMessage + ToolMessage із зображенням.
    Залишає 1 місце для SystemMessage.
    :param history: список повідомлень
    :param max_message_memory: максимальна кількість повідомлень в історії
    :return: обрізана історія
    """

    # Шукаємо останнє людське повідомлення, щоб 100% залишити
    human_message = next((m for m in reversed(history) if isinstance(m, HumanMessage)), None)

    ai_message = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)

    # Знаходимо останню пару AIMessage з tool_call + ToolMessage для пошуку та зображення. В історю вони записані саме по-черзі
    vision_pair = []
    search_pair = []
    for i, msg in enumerate(history):
        if isinstance(msg, list):
            ai_msg, tool_msg = msg
            if any(tc["name"] == "search" for tc in ai_msg.tool_calls):
                search_pair = msg
            elif any(tc["name"] == "vision" for tc in ai_msg.tool_calls):
                vision_pair = msg

    # Обов'язкові повідомлення які завжди залишаються
    required = []
    if human_message:
        required.append(human_message)
    if ai_message:
        required.append(ai_message)

    if search_pair:
        required.append(search_pair)
    if vision_pair:
        required.append(vision_pair)


    optional = [m for m in history if m not in required]

    max_history = max_message_memory - len(required)
    if max_history <= 0:
        optional = []
    elif len(optional) > max_history:
        optional = optional[-max_history:]

    result = optional + required

    logger.debug(f"Історія: {[msg.content[:20] for msg in result if isinstance(msg, AIMessage)]}")

    # Сортуємо за оригінальним індексом в history
    return sorted(result, key=lambda m: history.index(m))


def unpack_history(history: list) -> list:
    result = []
    for msg in history:
        if isinstance(msg, list):
            result.extend(msg)
        else:
            result.append(msg)
    return result