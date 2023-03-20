import numpy as np
import numba
import time
from argparse import ArgumentParser

def command_line_fetcher():
    # function to fetch command line arguments
    parser = ArgumentParser(description="ROBDD Generation")
    parser.add_argument('--input', type=str, required=True, help="Input logical expression for generation of BDD")
    parser.add_argument('--order', type=str, required=True, help="The order of preference of variables, input without spaces in between")
    return parser.parse_args()


def bdd_to_robdd(t,h):
    dim=np.size(t)
    adj=np.zeros(shape=(dim,dim), dtype=int)
    for i in range(2**h-1):
        adj[i,2*i+1]=1
        adj[i,2*i+2]=1
        adj[2*i+1,i]=1
        adj[2*i+2,i]=1

    for i in range(2**h-2,2**(h-1),-2):
        if(t[i]==t[i-1]):
            print('x')
            adj[i,(int)((i-1)/2)]=0
            adj[(int)((i-1)/2),i]=0
    print(adj)


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
                out=out and (x[order.find(exp[i+1])])
        return out 

@numba.njit
def evaluate_expression2(x, exp):
    while(len(exp)>0):
        opr=exp.find('+')
        if opr>-1:
            out=1
            if(len(exp[:opr])==1):
                out= x[order.find(exp[0])]
                if out==1:
                    return 1
            if(exp[0]=='~'):
                out=out and not(x[order.find(exp[1])])
            else:
                out=out and (x[order.find(exp[0])])
            for i in range(1,len(exp)-1):
                if(exp[i]=='~'):
                    out=out and not(x[order.find(exp[i+1])])
                elif(exp[i]=='.'):
                    out=out and (x[order.find(exp[i+1])])
            if out==1:
                return 1
            else:
                exp=exp[opr+1:]
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
                    out=out and (x[order.find(exp[i+1])])
            return out

@numba.njit
def int_to_binary(integer):
    binary_string =np.zeros(shape=len(order))
    j=len(order)
    while(integer>0):
        j=j-1
        digit = integer % 2
        binary_string[j]=digit
        integer = integer // 2
    return binary_string

@numba.njit
def main():
    tree=[]
    for i in range(len(order)):
        for j in range(2**(i)-1, 2**(i+1)-1):
            tree.append(order[i])
            height=i+1
    
    j=0
    for i in range(2**(height)-1, 2**(height+1)-1):
        x1=np.asarray(int_to_binary(j))
        tree.append(evaluate_expression2(x1,inp))
        j=j+1
    print(tree)

    # bdd_to_robdd(tree, height)

if __name__=="__main__":
    inp="a+b.c"
    order="abc"
    s=time.perf_counter()
    main()
    print(time.perf_counter()-s)