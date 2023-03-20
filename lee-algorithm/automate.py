from automan.api import Problem, Simulation, Automator, mdict, opts2path, filter_cases
import matplotlib.pyplot as plt
import numpy as np

sizes = [4,10,20,30,100,200,300,500,700, 1000]

class Benches(Problem):

    def get_name(self):
        return 'Benches'

    def setup(self):
        options = mdict(size = sizes)
        for x in range(len(options)):
            options[x]['file'] = sizes[x]
        base_cmd = 'python3 create-benchmark.py -p 0.1'
        self.cases = [
            Simulation(
                root=self.input_path(opts2path(kw)),
                base_command=base_cmd,
                **kw
            )
            for kw in options
        ]

    def run(self):
        pass

class Lee(Problem):
    def get_name(self):
        return 'Lee'

    def setup(self):
        options = mdict(maze = sizes)
        base_cmd = 'python3 lee-algorithm.py -sx 0 -sy 0 -ex 5000 -ey 5000 --b'
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
        self.make_plots()
    
    
    def make_plots(self):
        plt.figure()
        time1 = []
        time2 = []
        time3 = []
        for case in self.cases:
            with open(case.input_path('stdout.txt')) as file:
                m = file.readlines()
                time1.append(float(m[0]))
                time2.append(float(m[1]))
                time3.append(float(m[2]))
        plt.loglog(sizes,time1, label='accelerated-numba')
        plt.loglog(sizes,time2, label='regular-python')
        plt.loglog(sizes,time3, label='networkx-bfs')
        plt.grid()
        plt.xlabel('size')
        plt.ylabel('time taken')
        plt.legend()
        plt.savefig(self.output_path('solution.pdf'))
        plt.close()

if __name__ == '__main__':
    automator = Automator(
        simulation_dir='outputs',
        output_dir='manuscript/figures',
        all_problems=[Lee, Benches]
    )
    automator.run()