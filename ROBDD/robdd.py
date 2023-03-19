import numpy as np

def evaluate_expression(x, exp):
    opr=exp.find('+')
    print(opr)
    if opr>-1:
        return evaluate_expression(x,exp[:opr]) or evaluate_expression(x,exp[opr+1:])
    else:
        out=1
        print(exp[0])
        if(np.size(exp)==1):
            return x[order.find(exp[0])]
        for i in range(np.size(exp)-1):
            if(exp[i]=='~'):
                out=out and not(x[order.find(exp[i+1])])
            elif(exp[i]!='.'):
                out=out and (x[order.find(exp[i+1])])
        print(out)
        return out 

def int_to_bin(x):
    b= []
    while(x>0):
        d=x%2
        b.append(d)
        x=x//2
    return b.reverse()

def main():
    tree=[]
    for i in range(np.size(order)):
        for j in range(2**(i)-1, 2**(i+1)-1):
            tree.append(order[i])
            height=i+1
    
    j=0
    for i in range(2**(height)-1, 2**(height+1)-1):
        x=int_to_bin(j)
        print(inp)
        tree.append(evaluate_expression(x,inp))
        j=j+1
    print(tree)

if __name__=="__main__":
    inp="a+b.c"
    order="abc"
    main()