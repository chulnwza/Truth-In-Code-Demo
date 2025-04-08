"""Docstring"""
def main():
    """main"""
    way = input().upper()
    point = [[0, 0]]
    p_x, p_y = 0, 0
    for i in way:
        if i == "N":
            p_y += 1
        elif i == "S":
            p_y -= 1
        elif i == "E":
            p_x += 1
        elif i == "W":
            p_x -= 1
        point.append([p_y, p_x])
    tmpy = [i[0] for i in point]
    tmpx = [i[1] for i in point]
    for i in range(max(tmpy), min(tmpy)-1, -1):
        for j in range(min(tmpx), max(tmpx)+1):
            if [i, j] in point and [i, j] == point[0]:
                print("B", end=" ")
            elif [i, j] in point and [i, j] == point[-1]:
                print("E", end=" ")
            elif [i, j] in point:
                print("O", end=" ")
            else:
                print("-", end=" ")
        print()
main()
