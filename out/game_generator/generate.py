import random

f = open("Constraints.txt","r")
constraints = f.readlines()
f.close() 

f = open("Game Ideas.txt","r")
games = f.readlines()
f.close() 

for i in range(0, len(constraints)):
	constraints[i] = constraints[i].replace("\n", "")
	
for i in range(0, len(games)):
	games[i] = games[i].replace("\n", "")

x1 = random.randrange(len(constraints))
x2 = random.randrange(len(games))
print("Design a <mark class='blue'>{}, </mark><mark class='red'>{}</mark>".format(games[x2], constraints[x1]))
