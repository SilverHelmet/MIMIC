import sys


nums = []
for line in sys.stdin:
	parts = line.split("\t")
	nums.append(int(parts[1]))

add = lambda x , y: x + y
tot = reduce(add, nums)
print tot / (len(nums)+0.0)
