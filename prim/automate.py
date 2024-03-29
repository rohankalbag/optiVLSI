from automan.api import Problem, Simulation, Automator, mdict, opts2path
import matplotlib.pyplot as plt
import numpy as np

sizes = [10, 20, 50, 100, 150, 200, 350, 400, 500]


class Prim(Problem):
    def get_name(self):
        return 'Prim'

    def setup(self):
        options = mdict(size=sizes)
        base_cmd = 'python3 prim.py --c -p 0.3 -w1 5 -w2 15 -m $output_dir/graph -t $output_dir/min_span_tree --b'
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

        plt.semilogy(sizes, time1, '-s', label='regular-pythonic-prim')
        plt.semilogy(sizes, time2, '-s', label='networkx-minimum_spanning_tree-prim')
        plt.semilogy(sizes, time3, '-s', label='numba-accelerated-prim')
        plt.grid()
        plt.xlabel('number of nodes in graph')
        plt.ylabel('time taken')
        plt.legend()
        plt.savefig(self.output_path('timing.pdf'))
        plt.close()

        plt.figure()
        plt.plot(sizes, time1/time3, '-s', label='speedup vs regular-pythonic-prim')
        plt.plot(sizes, time2/time3, '-s', label='speedup vs networkx-minimum_spanning_tree-prim')
        plt.grid()
        plt.xlabel('number of nodes in graph')
        plt.ylabel('speedup')
        plt.legend()
        plt.savefig(self.output_path('speedup.pdf'))
        plt.close()


if __name__ == '__main__':
    automator = Automator(
        simulation_dir='outputs',
        output_dir='manuscript/figures',
        all_problems=[Prim]
    )
    automator.run()
