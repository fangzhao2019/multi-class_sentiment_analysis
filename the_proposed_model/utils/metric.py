import numpy as np

## input as sentence level labels
def get_ner_fmeasure(fact_results, predict_results, labelSet):
    #labelSet=[l for l in labelSet]
    #index=labelSet.index('3')

    results_count=np.zeros((len(labelSet),len(labelSet)))
    for idx in range(len(fact_results)):
        assert (len(fact_results[idx]) == len(predict_results[idx]))
        for idy in range(len(fact_results[idx])):
            fact= fact_results[idx][idy]
            predict= predict_results[idx][idy]

            index1=labelSet.index(fact)
            index2=labelSet.index(predict)

            results_count[index1][index2]+=1
    
    #results_count=np.delete(results_count,index,axis=0)
    #results_count=np.delete(results_count,index,axis=1)
    #labelSet.remove('3')

    fmeasure= {}
    total_TP= 0
    for idx in range(len(labelSet)):
        metric={}
        TP=results_count[idx,idx]
        total_TP += TP

        precision= TP/float(np.sum(results_count,axis=0)[idx]+0.5)
        recall= TP/float(np.sum(results_count,axis=1)[idx]+0.5)
        f_score=2*precision*recall/float(recall+precision)
        metric['p']=precision
        metric['r']=recall
        metric['f']=f_score
        fmeasure[labelSet[idx]]=metric

    accuracy=total_TP/np.sum(results_count)
    fmeasure['acc']=accuracy
    return fmeasure

