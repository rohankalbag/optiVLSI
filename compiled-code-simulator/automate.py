from automan.api import Problem, Simulation, Automator, mdict, opts2path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

names = {}

files = ['benchmarks/' + i[:-4] for i in os.listdir('benchmarks')]

def modify_path(s):
    k = s.index("benchmarks")
    return (s[:k] + s[k+11:])

class CCSimulator(Problem):
    def get_name(self):
        return 'CCSimulator'

    def setup(self):
        options = mdict(circuit=files)
        base_cmd = 'python3 compiled_code_sim.py -t $output_dir/normal --b'
        self.cases = [
            Simulation(
                root=self.input_path(opts2path(kw)),
                base_command=base_cmd,
                **kw
            )
            for kw in options
        ]

    def run(self):
        self.make_output_dir()
        self.store_data()
    
    def store_data(self):
        global names

        for case in self.cases:
            l = modify_path(case.input_path('stdout.txt'))
            key = l.split('/')[2]
            if key not in names.keys():
                names[key] = [-1, -1]
            with open(l) as file:
                m = file.readlines()
                names[key][0] = float(m[0])
        

class CCSimulatorNumba(Problem):
    def get_name(self):
        return 'CCSimulatorNumba'

    def setup(self):
        options = mdict(circuit=files)
        base_cmd = 'python3 compiled_code_sim_numba.py -t $output_dir/numba --b'
        self.cases = [
            Simulation(
                root=self.input_path(opts2path(kw)),
                base_command=base_cmd,
                **kw
            )
            for kw in options
        ]

    def run(self):
        self.make_output_dir()
        self.store_data()
    
    def store_data(self):
        global names

        for case in self.cases:
            l = modify_path(case.input_path('stdout.txt'))
            key = l.split('/')[2]
            if key not in names.keys():
                names[key] = [-1, -1]
            with open(l) as file:
                m = file.readlines()
                names[key][1] = float(m[0])
        

def plot():
    global names
    plt.figure()

    time1 = []
    time2 = []
    labels = []

    for i in names.keys():
        labels.append(i)
        time1.append(names[i][0])
        time2.append(names[i][1])

    plt.semilogy(labels, time1, label="regular-pythonic", marker='o')
    plt.semilogy(labels, time2, label="numba-accelerated", marker='s')

    plt.xlabel('circuit network benchmarks used')
    plt.ylabel('time taken')
    plt.legend()
    plt.savefig('manuscript/figures/timing.pdf')
    plt.close()

if __name__ == '__main__':
    automator = Automator(
        simulation_dir='outputs',
        output_dir='manuscript/figures',
        all_problems=[CCSimulator, CCSimulatorNumba]
    )
    automator.run()
    plot()
