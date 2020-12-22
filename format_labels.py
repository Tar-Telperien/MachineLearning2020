#read all the stuff from labels.txt and make it into a presentable format
f = open("manyBoards2/labels.txt", "r")
ugly_mess = f.read()
pair_strings = ugly_mess.split("]")
#print("Length of the ugly mess split by ] ", len(pair_strings)) 

pair_dict = {}

for string in pair_strings:
    if string != "":
        print("The string is, ", string)
        two_strings = string.split("[")
        #print(two_strings)
        string_centre = two_strings[1].split(", ")
        centre = [float(string_centre[0]), float(string_centre[1])]
        pair_dict[two_strings[0]] = centre

f.close()

f2 = open("manyBoards2/labels_dict.txt", "w")
f2.write(str(pair_dict))
f2.close()
