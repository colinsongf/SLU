f = open('current.valid.txt')
from collections import defaultdict
conf = defaultdict(int)
tags = set()
for line in f :
	line = line.strip()
	if len(line) > 0 :
		line = line.split(' ')
		assert len(line) == 3
		conf[(line[1], line[2])] += 1
		tags.add(line[1])
		tags.add(line[2])

tags = list(tags)
for t in tags :
	for s in tags :
		print conf[(t, s)],
	print ""

#print conf
print tags