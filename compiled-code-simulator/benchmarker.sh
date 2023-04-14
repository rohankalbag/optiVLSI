mkdir benchmarks
for n in 2 4 8 16; 
do
    python3 benchmark_create.py -n $n -c benchmarks/and$n
    python3 benchmark_create.py -n $n -c and$n
done
rm -rf outputs
rm -rf manuscript
python3 automate.py
rm a*.txt
