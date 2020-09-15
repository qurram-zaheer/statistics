import pandas as pd
import csv
from pprint import pprint

input_df = pd.read_csv("input.csv", delimiter=',')

fuzzy_set=[-1,-0.66,-0.33,0,0.15,0.33,0.45,0.75,1]

def normalise_fn(arr,ul,ll):
    normalise_arr=[]
    for i in arr:
        val=(2*i-(ul+ll))/(ul-ll)
        normalise_arr.append(val)
    return normalise_arr
        

def degree_of_belonging(val,a,b,c):
    if(val<a):
        return 0
    if(val>=a and val<=b):
        return (val-a)/(b-a)
    if(val>=b and val<=c):
        return (c-val)/(c-b)
    if(val>c):
         return 0
        

# input_df = input_df.drop(["Cases"], axis=1)
# print(input_df)
var1=input_df.iloc[:,0].tolist()
var2=input_df.iloc[:,1].tolist()

norm_var1=normalise_fn(var1,10,-10)
norm_var2=normalise_fn(var2,30,-30)

output_fuzzy_set={1:25,2:50,3:60,4:70,5:75,6:80,7:85,8:90,9:95}
pprint(output_fuzzy_set)

fuzzy_sets=[]
count=0

for i in range(1,len(fuzzy_set)-1):
    count=i-1
    temp=[]
    while(count<i+2):
        temp.append(fuzzy_set[count])
        count+=1
    fuzzy_sets.append(temp)
print(fuzzy_sets)

def mat_mul(x,y):
    resmat=[[],[],[],[],[],[],[]]
    for i in range(0,7):
        for j in range(0,7):
            resmat[i].append(x[i]*y[j])
    return resmat
        

def defuzzyfy(fam,mat):
    final_crisp=0
    denom=0
    for i in range(0,len(fam)):
        for j in range(0,len(fam[i])):
            final_crisp= final_crisp+fam[i][j]*mat[i][j]
            denom=denom+mat[i][j]
    final_crisp=final_crisp/denom
    return final_crisp


fam_subsets=[[1,1,1,1,1,1,1],[1,1,1,1,1,1,2],[1,1,1,1,1,2,3],[1,1,1,1,2,3,5],[1,1,1,2,2,4,6],[1,1,1,3,4,6,8],[1,2,3,5,6,8,9]]
fam=fam_subsets
for i in range(0,len(fam_subsets)):
    for j in range(0,len(fam_subsets[i])):
        fam[i][j]=output_fuzzy_set[fam_subsets[i][j]]
pprint(fam)

col = []
def final_val(x,y):
    fuzzy_score_x=[]
    fuzzy_score_y=[]
    for i in fuzzy_sets:
        fuzzy_score_x.append(degree_of_belonging(x,i[0],i[1],i[2]))
        fuzzy_score_y.append(degree_of_belonging(y,i[0],i[1],i[2]))
    #print(fuzzy_score_x)
    #print(fuzzy_score_y)
    mat=mat_mul(fuzzy_score_x,fuzzy_score_y)
    #pprint(mat)
    result=defuzzyfy(fam,mat)
    pprint(result)
    col.append(result)
    #print(output_fuzzy_set[result])


for i in range(0,len(norm_var1)):
    x=norm_var1[i]
    y=norm_var2[i]
    final_val(x,y)

input_df['Output'] = col

input_df.to_csv("17XJ1A0537.csv", index = False)