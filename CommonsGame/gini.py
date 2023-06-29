x = 0.46
y = 1.0 - x
area1 = 0.5*x*0.5
area2 = 0.5*x
area3 = 0.5*y*0.5

B = area1 + area2 + area3
A = 1*1*0.5 - B

print(2*round(A/(A+B),2))

from scipy.special import comb

print(comb(19,1))

total = 0
for i in range(5+1):
    total += (40+1-i)*(40+2-i)
    print(total)

print(total*576)

c= 5
m=40
x = 576*( (c+1)*(m**2+3*m+2))

total = 0
for i in range(5+1):
    total += -(2*(m+1.5)*i)

total += (c+1)*(m**2+3*m+2) + c*(c+1)*(2*c+1)/6

y = 576*(c*(c+1)*(2*c+1)/6 - (2*m+3)*c*(c+1)/2)

print(x+y)
print("----")
print(96*(c+1)*(6*(m**2)))##+3*m+2)))#+c*(2*c-6*m-8)))
