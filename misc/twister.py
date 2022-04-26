from random import choice
import random

random.seed()

colors = ["желтый", "зеленый", "красный", "серый"]
sides = ["правую", "левую"]
ends = ["ногу", "руку"]

cont_signal = True

while True:
    input()
    print(f"Поставить {choice(sides)} {choice(ends)} на {choice(colors)} квадрат", end="")
