from aiogram.fsm.state import State, StatesGroup

class UserState(StatesGroup):
    user_photo = State()
    user_question = State()
    user_name = State()
    bot_name = State()