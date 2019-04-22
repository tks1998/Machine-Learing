a = [eval(i) for i in open('data.json') ]
#print (a[0]['headline'])
s=""
output = {}
for i in range(0,len(a)):
    sentence = a[i]['headline']
    for word in sentence.split(" "):
        if word not in output.keys():
            output[word] = 0
        output[word] += 1   
maxx=0
for k,v in output.items():
    if (v>maxx):
        maxx=v
        kq=k
print(len(output),' ',kq ,' ',maxx)
#print(output)
#myList = output.sortNumericallyByKeys    
