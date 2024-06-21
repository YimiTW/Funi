import funi

user = input("Please enter your name: ")
funi.mode = 'local'

if user == "/exit":
    quit()

while True:
    text_input = input(f"\n{user}: ")

    now = funi.datetime.datetime.now()
    response = funi.main_request(text_input, user)