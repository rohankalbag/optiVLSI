from argparse import ArgumentParser
import numpy as np

def command_line_fetcher():
    # function to fetch command line arguments
    parser = ArgumentParser(description="ccsim")
    parser.add_argument("-n", "--pow2", type=int)
    parser.add_argument("-c", '--circuit', help="save final circuit")
    return parser.parse_args()


if __name__ == "__main__":
    args = command_line_fetcher()
    n = args.pow2
    n = 2 ** int(np.log2(n))
    c = args.circuit
    nodes = list(range(n))
    node_count = n
    with open(f'{c}.txt', 'w') as t:
        s = "inp " + ' '.join([str(i) for i in nodes]) + '\n'
        t.write(s)
        t.write(f"outp {n*2 - 2}\n")
        while(len(nodes) > 0):
            op1 = nodes.pop(0)
            try:
                op2 = nodes.pop(0)
            except:
                break
            op3 = node_count
            t.write(f"and {op1} {op2} {op3}\n")
            nodes.append(op3)
            node_count += 1

