from typing import Optional

from pydantic import BaseModel, Field, model_validator


class CalculatorInput(BaseModel):
    """Schema for input to the calculator"""
    question: str = Field(description="the math problem that you want to solve")
    axioms: dict = Field(description="the axioms of current math problem. keys - name of axiom, value - value of axiom",
                         examples=[
                             # Приклад 1: Простий чек/ціни
                             {"свинина_ціна_кг": 250, "вага_грам": 450, "знижка_відсоток": 10},

                             # Приклад 2: Лічильники (поточне та попереднє значення)
                             {"current_value": 124.68, "previous_value": 110.20, "tariff_per_m3": 4.32},

                             # Приклад 3: Геометрія або будівництво
                             {"wall_height_m": 2.7, "wall_width_m": 4.5, "paint_consumption_per_m2": 0.2}])


class ProcessorAnswer(BaseModel):
    """Якщо інформації достатньо - передай query - це буде запит в мережу за яким буде виконано пошук.
    Якщо інформації достатньо - надай answer - це буде фінальна відповідь користувачеві
    При БУДЬ-ЯКИХ підрахунках надай calculator_input.
    Якщо передаєш answer - ти не можеш передати ні query, ні calculator_input.
    Але можеш передати query та calculator_input разом.
    Маєш передати мінімум 1"""

    answer: Optional[str] = Field(default=None, description="Фінальна відповідь")
    query: Optional[str] = Field(default=None, description="Пошуковий запит в Google")
    calculator_input: Optional[CalculatorInput] = Field(default=None, description="Дані для розрахунку")

    @model_validator(mode='after')
    def check_logic_flow(self):
        tools = [bool(self.answer), bool(self.query),bool(self.calculator_input)]
        if sum(tools)==0:
            raise ValueError("Модель не надала відповіді, або надала не в тому форматі")
        return self