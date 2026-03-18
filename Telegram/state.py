from aiogram.fsm.state import State, StatesGroup

class UserState(StatesGroup):
    # стейти для запису даних
    user_name = State()
    bot_name = State()
    history = State()
    pending_router_response = State()

    # стейти для обробки за допомогою хендлерів
    wait_input = State()
    specification = State()
    wait_image = State()
