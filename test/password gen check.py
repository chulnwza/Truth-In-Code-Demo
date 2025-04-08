"""password gen check"""
def main():
    """main"""
    txt = input()
    if len(txt) == 6 and txt.isalpha():
        if txt[:3].isupper() and txt[3:].islower():
            summ = 0
            for i in txt:
                summ += ord(i)
            print("Your password is : "+ txt + str(summ))
        else:
            print("Password is not secure or invalid")
    else:
        print("Password is not secure or invalid")
main()
