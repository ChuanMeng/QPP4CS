import matplotlib
matplotlib.use('Agg')
import os 
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
x=0
l = os.listdir('run/')
l.sort()


for run in ['cast-19.run-ConvDR-1000.txt','cast-19.run-QuReTeC-Q-bm25-1000.txt','cast-19.run-T5-Q-bm25-1000.txt','cast-19.run-manual-bm25-1000.txt']:
    font = {'family' : 'normal',
        'size'   : 18}


    if 'or' not in run:
        continue
    file=open('datasets/cast-19-20/runs/'+run,'r').readlines()
    scores=[]
    for line in file:
        try:
            _,_,_,_,score,_=line.split()
            scores.append(float(score))
        except:
            print(line)
            break
            scores.append(float(score))

    mi=min(scores)
    ma=max(scores)
    scores= [(float(i)-mi)/(ma-mi) for i in scores]

    x+=1
    plt.boxplot(scores,positions=[x],widths= 0.75)
    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({'font.size': 10})
    print(run)


ax.set_xticklabels( ['ConvDR','QuReTeC+BM25','T5+BM25','Human+BM25'], fontsize=14,rotation=25)
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rc('ytick', labelsize=10) 
plt.title('CAsT-19')
plt.tight_layout()

plt.savefig('plots/cast-19'+'.png')

plt.close()