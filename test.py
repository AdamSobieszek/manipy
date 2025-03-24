import re
import pyperclip


def remove_bracket_content(text):
    # Use regex to remove content inside square brackets, including the brackets
    return re.sub(r'\[.*?\]', '---', text)


def main():
    # Get the content from the clipboard
    clipboard_content = pyperclip.paste()

    # Remove the text inside [*]
    modified_content = remove_bracket_content(clipboard_content)

    # Set the modified content back to the clipboard
    pyperclip.copy(modified_content)


if __name__ == "__main__":
    main()


