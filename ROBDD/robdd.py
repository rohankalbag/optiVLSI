import numpy as np

def evaluate_expression(x, exp):
    opr=exp.find('+')
    if opr>-1:
        return (evaluate_expression(x,exp[:opr]) or evaluate_expression(x,exp[opr+1:]))
    else:
        out=1
        if(len(exp)==1):
            return x[order.find(exp[0])]
        if(exp[0]=='~'):
            out=out and not(x[order.find(exp[1])])
        else:
            out=out and (x[order.find(exp[0])])
        for i in range(1,len(exp)-1):
            if(exp[i]=='~'):
                out=out and not(x[order.find(exp[i+1])])
            elif(exp[i]=='.'):
                print('and')
                out=out and (x[order.find(exp[i+1])])
        print(out)
        return out 

def int_to_binary(integer):
    binary_string =np.zeros(shape=len(order),dtype=int)
    j=len(order)
    while(integer>0):
        j=j-1
        digit = integer % 2
        binary_string[j]=digit
        integer = integer // 2
    return binary_string

def main():
    tree=[]
    for i in range(len(order)):
        for j in range(2**(i)-1, 2**(i+1)-1):
            tree.append(order[i])
            height=i+1
    
    j=0
    for i in range(2**(height)-1, 2**(height+1)-1):
        x1=np.asarray(int_to_binary(j))
        print(x1)
        tree.append(evaluate_expression(x1,inp))
        j=j+1
    print(tree)

if __name__=="__main__":
    inp="a+b.c"
    order="abc"
    main()