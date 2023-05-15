import matplotlib
matplotlib.use('Agg')
import os 
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
x=0
l = os.listdir('run/')
l.sort()


for run in l:
    font = {'family' : 'normal',
        'size'   : 18}


    if 'or' not in run:
        continue
    file=open('run/'+run,'r').readlines()
    scores=[]
    for line in file:
        try:
            _,_,_,_,score,_=line.split()
            scores.append(float(score))
        except:
            print(line)
            break
            scores.append(float(score))
    #scores = [x for x in scores if x < 600 ]
    #scores= [(float(i)-min(scores))/(max(scores)-min(scores)) for i in scores]
    mi=min(scores)
    ma=max(scores)
    scores= [(float(i)-mi)/(ma-mi) for i in scores]
    print(len(scores))

 
    
    #ax.boxplot(scores)
    x+=1
    plt.boxplot(scores,positions=[x],widths= 0.75)
    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({'font.size': 10})
    

    print(run)
    #plt.boxplot(data1,0,'',positions=x-100,widths=150)
    #plt.boxplot(data2,0,'',positions=x+100,widths=150)'''
ax.set_xticklabels( ['ConvDR','QuReTeC+BM25','T5+BM25','Human+BM25'], fontsize=14,rotation=25)
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rc('ytick', labelsize=10) 
plt.title('OR-QuAC')
plt.tight_layout()

plt.savefig('QuAC'+'.png')

plt.close()