"""Docstring"""
def main():
    """main"""
    import math
    num = int(input())
    ti4 = []
    ti3 = []
    if num < 3:
        print("Nope")
    else:
        for i in range((num//4)+1):
            for j in range((num//3)+1):
                if (4*i + 3*j) == num:
                    ti4.append(i)
                    ti3.append(j)
        if len(ti4) == 0:
            print("Nope")
        else:
            print("Yes")
            sol = 0
            for i in range(len(ti4)):
                sol += math.factorial(ti4[i] + ti3[i])/(math.factorial(ti4[i]) * \
                    math.factorial(ti3[i]))
            print(int(sol))
main()
