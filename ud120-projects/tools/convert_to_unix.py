def convert_pkl(original):
	destination = f"{(original.split('.pkl')[0])}_unix.pkl"

	outsize = 0
	content = ''
	with open(original, "rb") as infile:
		content = infile.read()

	with open(destination, "wb") as output:
		for line in content.splitlines():
			outsize += len(line) + 1
			output.write(line + str.encode('\n'))

	print("Done. Saved %s bytes" % (len(content) - outsize))