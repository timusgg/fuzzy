import skfuzzy as fuzz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


LinguisticTermSet = {'VL': [0, 0, 0.25], 'L': [0, 0.25, 0.50], 'D': [0.25, 0.50, 0.75], 'H': [0.50, 0.75, 1], 'VH': [0.75, 1, 1]}
Criteria = ['Price', 'Goals Scored', 'Assists', 'ICT']
Alternatives = ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5', 'Player 6']

UniverseOfDiscourse = np.linspace(0,1,num=5)


for term in LinguisticTermSet:
    MembershipValue = fuzz.trimf(UniverseOfDiscourse, LinguisticTermSet[term])
    plt.plot(UniverseOfDiscourse, MembershipValue, label = term)
    plt.axis([0,1,0,1])
    plt.xticks(UniverseOfDiscourse)
    plt.legend()

plt.tight_layout()
plt.show()

LinguisticResponses = pd.read_csv('responses.csv')

print(LinguisticResponses)

CollectivePerformanceVectors = []

for column in LinguisticResponses.columns[1:]:
    vector = []
    for i in range(0,3):
        LRvalues = [LinguisticTermSet[lr][i] for lr in LinguisticResponses[column].values]
        vector.append(sum(LRvalues)/4)
            
    CollectivePerformanceVectors.append(vector)



def extensionPrinciple():
    #Applying Linguistic Appproximation
    def Linguistic_Approximation(term, vec):
        p = [0.33, 0.33, 0.34]
        sum = 0
        for i in range(3):
            sum += p[i]*((term[i]-vec[i])**2)
        return math.sqrt(sum)
        
    Distances = []
    for vector in CollectivePerformanceVectors:
        d = []
        for term in LinguisticTermSet:    
            d.append(Linguistic_Approximation(LinguisticTermSet[term], vector))
        Distances.append(d)

    ranks = []
    for d in Distances:
        ranks.append(d.index(min(d)))
    
    indexes = [i for i,x in enumerate(ranks) if x == max(ranks)]

    BestAlternatives = [Alternatives[i] for i in indexes]

    print('Using Extension Principle based LDM Best Alternatives are', BestAlternatives)

extensionPrinciple()

def Twotuple():
    twotupleresponses = []
    for column in LinguisticResponses.columns[1:]:
        twotupleresponses.append([[i,0] for i in LinguisticResponses[column].values])

    lts = list(LinguisticTermSet)  

    AggregatedTwoTuples = []
    for r in twotupleresponses:
        arr = []  
        for tuples in r:
            arr.append(lts.index(tuples[0]))
        
        beta = sum(arr)/4
        alpha = beta - round(beta+0.01, None)

        AggregatedTwoTuples.append([round(beta+0.01, None), alpha])



    def ranking(att):
        sortedAtt = []
        original = att
        arr = sorted(att, key = lambda l: (-l[0],-l[1]))     
        
        for item in arr:
            sortedAtt.append([lts[item[0]], item[1]])
        
        indexes = [original.index(arr[i]) for i in range(0,2)]

        BestAlternatives = [Alternatives[i] for i in indexes]
    
        print('Using Two-tuple based LDM Best Alternatives are', BestAlternatives)
        
    ranking(AggregatedTwoTuples)


Twotuple()

def symbolicMethod():
    lts = list(LinguisticTermSet)
    #print(lts)
    sortedElements = []
    for column in LinguisticResponses.columns[1:]:
        lst = []
        for i in range(0,4):
            lst.append(lts.index(LinguisticResponses[column][i]))
        lst.sort(reverse=True)
        sortedElements.append(lst)
    #print(sortedElements)


    weights = [1/len(sortedElements) for _ in range(len(sortedElements))]
    m = len(sortedElements[0])

    combinations = []

    def convexCombination(w, b, m):
        if m == 2:
            l = b[1]
            q = b[0]
            w1 = w[0]
            r = min(len(lts),l + round(w1*(q-l)+0.01))
            return r
        
        if m>2:
            w1 = w[0]
            b1 = b[0]
            b.pop(0)
            newWeights = [1/len(b) for _ in range(len(b))]
            combination = ['C' + str(m), [w1, b1],[(1 - w1), ('C' + str(m-1))]]
            combinations.append(combination)
            return convexCombination(newWeights, b, m-1)


    linguisticResults = []
    for element in sortedElements:
        c2 = convexCombination(weights, element, m)
        c3 = 0
        c4 = 0
        for comb in combinations:
            q = comb[2][0]
            w1 = comb[1][0]
            if comb[2][1] == 'C2':
                l = c2
                c3 = min(len(lts),l + round(w1*(q-l)+0.01)) 
            else:
                l = c3
                c4 = min(len(lts),l + round(w1*(q-l)+0.01))
        linguisticResults.append([c2, c3, c4])

    maxlr = [max(lr) for lr in linguisticResults]
    #print(maxlr)
    indexes = [i for i,x in enumerate(maxlr) if x == max(maxlr)]
    BestAlternatives = [Alternatives[i] for i in indexes]

    print('Using Symbolic Method based LDM Best Alternatives are', BestAlternatives)


         
        
symbolicMethod()



'''

            

'''