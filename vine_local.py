import vine

user = input("Please enter your name: ")
vine.mode = 'local'

if user == "/exit":
    quit()

while True:
    text_input = input(f"\n{user}: ")

    now = vine.datetime.datetime.now()
    response = vine.main_request(text_input, user)