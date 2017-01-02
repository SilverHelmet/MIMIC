import sys

if __name__ == "__main__":
    filepath = sys.argv[1]
    cnt = 0
    for line in file(filepath):
        cnt += 1
        print cnt
