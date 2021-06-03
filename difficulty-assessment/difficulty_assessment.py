import json
import os
import re

N1_NUM_KANJI = 2141
N2_NUM_KANJI = 1140
N3_NUM_KANJI = 660
N4_NUM_KANJI = 300
N5_NUM_KANJI = 120
NUM_KANA = 92 

files = "/Users/isaacbevers/CS229_Final_Project/untranslated-texts/"

difficulty_dict = {}
for filename in os.listdir(files):
    input_text = ""
    with open(files + filename, "rb") as f:
        input_text = f.read().decode("UTF-8")
    s = set(input_text)
    unique_chars = len(s)
    difficulty = 0
    if unique_chars > NUM_KANA + N1_NUM_KANJI:
        difficulty = 1
    elif unique_chars > NUM_KANA + N2_NUM_KANJI:
        difficulty = 2
    elif unique_chars > NUM_KANA + N3_NUM_KANJI:
        difficulty = 3
    elif unique_chars > NUM_KANA + N4_NUM_KANJI:
        difficulty = 4
    else:
        difficulty = 5 
    difficulty_dict[filename] = difficulty
print(difficulty_dict)
f = open("difficulty_dict.txt","w")
f.write( str(difficulty_dict) )
f.close()
