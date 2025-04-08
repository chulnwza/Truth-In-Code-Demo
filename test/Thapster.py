"""Docstring"""
def main():
    """main"""
    rnd = int(input())
    for _ in range(rnd):
        num = int(input())
        note1 = input().split()
        note2 = input().split()
        score = [0 for _ in range(num)]
        for j in range(len(note1)):
            if note1[j] == note2[j]:
                score[j] += (score[j-1] + 1)
        print(sum(score))
main()
