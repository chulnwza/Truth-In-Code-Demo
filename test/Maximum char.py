"""Docstring"""
def main():
    """main"""
    txt = input()
    maxx = 0
    check = []
    for i in txt:
        if txt.count(i) > maxx:
            maxx = txt.count(i)
    for i in txt:
        if txt.count(i) == maxx and i not in check:
            print(i, end="")
            check.append(i)
main()
