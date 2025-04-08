"""Docstring"""
def main():
    """main"""
    txt = input()
    for i in txt:
        print(chr(ord(i)+3), end="")
main()
