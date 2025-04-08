"""Docstring"""
def main():
    """main"""
    let = input()
    all_chr = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if let not in all_chr:
        print(let)
    else:
        if let.islower():
            num = ord(let) - 70
        else:
            num = ord(let) - 64
        space = num-1
        to_letter = 1
        for i in range(1, 2*num):
            char = all_chr[0:to_letter]
            print(" "*abs(space) + char + char[-2::-1])
            space -= 1
            if i >= num:
                to_letter -= 1
            else:
                to_letter += 1
main()
