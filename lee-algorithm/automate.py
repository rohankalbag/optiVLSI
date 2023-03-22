from automan.api import Problem, Simulation, Automator, mdict, opts2path, filter_cases
import matplotlib.pyplot as plt
import numpy as np

sizes = [2,4,10,20,30,50,100,150,200,300,500]

class Lee(Problem):
    def get_name(self):
        return 'Lee'

    def setup(self):
        options = mdict(size = sizes)
        base_cmd = 'python3 lee-algorithm.py --c -sx 0 -sy 0 -ex 5000 -ey 5000 --b -p 0.1'
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
        
        time1 = np.array(time1)
        time2 = np.array(time2)
        time3 = np.array(time3)
        
        plt.loglog(sizes,time1,'-s', label='accelerated-numba')
        plt.loglog(sizes,time2,'-s', label='regular-python')
        plt.loglog(sizes,time3, '-s', label='networkx-bfs')
        plt.grid()
        plt.xlabel('size of grid')
        plt.ylabel('time taken')
        plt.legend()
        plt.savefig(self.output_path('timing.pdf'))
        plt.close()

        plt.figure()
        plt.loglog(sizes,time2/time1,'-s', label='speedup vs regular-python')
        plt.loglog(sizes,time3/time1, '-s', label='speedup vs networkx-bfs')
        plt.grid()
        plt.xlabel('size of grid')
        plt.ylabel('speedup')
        plt.legend()
        plt.savefig(self.output_path('speedup.pdf'))
        plt.close()

if __name__ == '__main__':
    automator = Automator(
        simulation_dir='outputs',
        output_dir='manuscript/figures',
        all_problems=[Lee]
    )
    automator.run()