import os
import warnings

from aiogram import Bot, Dispatcher

from Chains.models import load_models
from Telegram.user_handlers import user
from Telegram.middleware import ThrottlingMiddleware
import settings_manager as s
from Telegram.admin_handlers import admin

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")


async def main():


    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Chains")

    bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
    dp = Dispatcher()
    dp.include_router(admin)
    dp.include_router(user)
    dp.message.middleware(ThrottlingMiddleware())

    logger.debug("Я народився!")

    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        import asyncio
        from dotenv import load_dotenv

        load_dotenv()
        s.load_settings()
        load_models()
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
