import vine

user = input("Please enter your name: ")
vine.mode = 'local'

if user == "/exit":
    quit()

while True:
    text_input = input(f"\n{user}: ")
    if text_input == "/cmd":
        while True:
            cmd = input("\n\n\n+" + "-"*35 + "+\n|    Please enter command number    |\n+" + "-"*35 + "|\n|1. Back to Chat" + " "*20 + "|\n|2. Exit Chat" + " "*23 + "+\n|3. Backup Chat" + " "*21 + "|\n|4. Load Chat Backup" + " "*16 + "|\n|5. Delete chat data" + " "*16 + "|\n+" + "-"*35 + "+\nCommand Number: ")
            vine.handle_command(cmd)
            if cmd == "1":
                break
    else:
        now = vine.datetime.datetime.now()
        response = vine.main_request(text_input, user)