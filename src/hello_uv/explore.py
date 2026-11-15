x = 3
y =2.5
name = "toy"

list = [2, 4, 6, 8]

dics = {"name": "toy", "layers":3}
print(type(x), type(y), type(name))

print(len(list), list[0], list[-1], list[1:3])

print(dics["name"], dics[ "layers"])

for l in list:
    if l > 4:
        print(l, "big")
    else:
        print(l, "small")

r = 0
for i in range(5):
    r += i

print(r)