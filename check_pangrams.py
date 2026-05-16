import re

all_letters_pattern = re.compile(r"([a-z])(?!.*\1)")

pangrams_file = "pangrams.txt"
with open(pangrams_file) as f:
    i = 1
    while line := f.readline():
        # test for pangram
        res = all_letters_pattern.findall(line.lower())
        if len(res) != 26:
            print(f"{line} was not a pangram.")
            print(len(res), i)
            break

        i += 1

print("Done.")
