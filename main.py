import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from Telegram.handlers import user

load_dotenv()

async def main():
    bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
    dp = Dispatcher()
    dp.include_router(user)
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
