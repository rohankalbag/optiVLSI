from automan.api import Problem, Simulation, Automator, mdict, opts2path
import matplotlib.pyplot as plt
import numpy as np
import os

time1 = []
time2 = []

files = [i[:-4] for i in os.listdir('benchmarks')]
sizes = [int(j[3:]) for j in [i[:-4] for i in os.listdir('benchmarks')]]

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
        global time1

        for case in self.cases:
            with open(case.input_path('stdout.txt')) as file:
                m = file.readlines()
                time1.append(float(m[0]))
        

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
        global time2

        for case in self.cases:
            with open(case.input_path('stdout.txt')) as file:
                m = file.readlines()
                time2.append(float(m[0]))
        

def plot():
    global time1, time2
    plt.figure()
    
    time1.sort()
    time2.sort()

    time1 = np.array(time1)
    time2 = np.array(time2)

    plt.semilogy(sizes, time1, '-s', label='pythonic-compiled-code-sim')
    plt.semilogy(sizes, time2, '-s', label='numba-compiled-code-sim')
    plt.grid()
    plt.xlabel('cascaded and network inputs')
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
