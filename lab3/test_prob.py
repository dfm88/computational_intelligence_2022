import random

N = 5

a = [random.random() for x in range(N)]
print(a)

ss = sum(a)

b = [x / ss for x in a]
print()
print(b)
print()

rr = random.random()
print(f"{rr}\n")

choos = 0
tot = 0
for x in b:
    tresh = tot + x
    tot += x
    if rr <= tresh:
        choos = x
        break

print(choos)

ff = b.copy()
import ipdb

ipdb.set_trace()
diff = ff[1] * 0.5
ff[1] = ff[1] - diff
ff[4] = ff[4] + diff

print(ff)
print(sum(b))
print(sum(ff))
print(sum(b) == sum(ff))
