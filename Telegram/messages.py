from telegram import Update
from telegram.ext import ContextTypes

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_photo("C:/Users/2004a/Downloads/іді.png")