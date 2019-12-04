import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def convert(x,a,b,c,d):
    """converts values in the range [a,b] to values in the range [c,d]"""
    return c + float(x-a)*float(d-c)/(b-a)
df=pd.read_csv('sample.csv', sep=',')
W=np.zeros(6,dtype=float)
gk = df.groupby('Political')
#print(gk.first())
df1= gk.get_group('Liberal') 
#print(df1.shape[0])
count_row = df1.shape[0]
"""
x= df1[df1.Age == '18-24'].shape[0]
x1=x*26
y= df1[df1.Age == '25-34'].shape[0]
y1= y*26
z= df1[df1.Age == '35-44'].shape[0]
z1=z*40
u= df1[df1.Age == '45-54'].shape[0]
u1=u*50
v= df1[df1.Age == 'above 55'].shape[0]
v1=v*60
c1=x+y+z+u+v
c2=x1+y1+z1+u1+v1
print(c2/c1)

x= df1[df1.Political == 'Liberal'].shape[0]
y= df1[df1.Political == 'Moderate'].shape[0]
z= df1[df1.Political == 'Conservative'].shape[0]
c1=x+y+z
print((x/c1)*100)
print((y/c1)*100)
print((z/c1)*100)

"""
#W[0] = df1['Age'].sum()/count_row
#df2= gk.get_group(1) 
#count_row = df2.shape[0]
#W[1] = df2['Age'].sum()/count_row
#df3= gk.get_group(2) 
#count_row = df3.shape[0]
#W[2] = df3['Age'].sum()/count_row
#print(df1['Title-P'].sum()/count_row)
"""
W[0]= ((df1['Title-P'].sum())/(count_row))
W[1] = ((df1['Picture-P'].sum())/(count_row))
W[2] = ((df1['Content-P'].sum())/(count_row))
W[3] = ((df1['Source-P'].sum())/(count_row))
W[4] = ((df1['Authors-P'].sum())/(count_row))
W[5] = ((df1['Date-P'].sum())/(count_row))
v1= np.var(W)
print(v1)
"""

#W[0]= convert(W[0],1,5,1,10)
print(W)
array1=['Title-P','Picture-P','Content-P','Source-P','Authors-P','Date-P']
array1=[1,2,3,4,5,6]
#plt.plot(array1,W)
#plt.show()
#plt.hist(W,bins=6) 