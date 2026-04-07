from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ProcessorAnswer(BaseModel):
    """Якщо інформації достатньо - передай query - це буде запит в мережу за яким буде виконано пошук.
    Якщо інформації достатньо - надай answer - це буде фінальна відповідь користувачеві.
    Ти можеш передати ЛИШЕ ОДИН з цих двох параметрів. Одразу 2 або жодного - не можеш"""

    answer: Optional[str] = Field(default=None, description="Фінальна відповідь")
    query: Optional[str] = Field(default=None, description="Пошуковий запит")

    @model_validator(mode='after')
    def check_only_one_field(self):
        # Перевіряємо, чи заповнено рівно одне поле
        if bool(self.answer) == bool(self.query):
            raise ValueError("Має бути заповнене рівно одне поле: або 'answer', або 'query'")
        return self