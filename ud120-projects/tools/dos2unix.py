"""
Convert dos linefeeds (crlf) to unix (lf)
usage:dos2unix.py
"""
original = "word_data.pkl"
destination = "word_data_unix.pkl"

content = ''
outsize = 0
with open(original, "rb") as infile:
	content = infile.read()

with open(destination, "wb") as output:
	for line in content.splitlines():
		outsize += len(line) + 1
		output.write(line + str.encode('\n'))

print("Done. Saved %s bytes" % (len(content) - outsize))