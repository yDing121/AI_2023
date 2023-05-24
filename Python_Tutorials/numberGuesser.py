import random

secret_number = random.randint(0, 10)
guess_number = int(input("Guess the number!\n"))
tries = 1
hint = True

while guess_number != secret_number:
    if not hint:
        pass
    elif guess_number > secret_number:
        print("Guess is too high!")
    else:
        print("Guess is too low!")
    guess_number = int(input("You've guessed incorrectly {} times, try again!\n".format(tries)))
    tries += 1

print("You guessed the number in {} tries!".format(tries))
