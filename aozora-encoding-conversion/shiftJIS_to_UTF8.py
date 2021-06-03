import os


dir = "/Users/isaacbevers/CS229_Final_Project"
src = dir + "/aozora_text_corpus_no_apostr"
target = dir + "/aozora_unicode/"
i = 0
for filename in os.listdir(src):
    os.system("iconv -f SHIFT-JIS -t UTF-8 " + src + "/" + filename + " > " + target + filename)

