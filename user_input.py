from lib2to3.pytree import Base
from pydantic import BaseModel
class UserInput(BaseModel):
    user_input: str