import joblib

x = joblib.load("s_target.pkl")

l = []

for _ in x[:100] :
    l.append(_[0])
    l.append(_[1])

print(l) 


with open("rad.sh","w") as f :
    for x in l :
        string = "aws s3 cp s3://gen-scan/MachineCodex/"+x+" . --recursive"
        f.write("\n"+string)

