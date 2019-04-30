import numpy as np

trivial_words = ['the','a']


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def similar_string(str1, str2):
    intersection = 0
    try:
        tokenA = set(str1.split())
        tokenB = set(str2.split())
    except:
        print('Exception handling %s and %s'%(str1, str2))
        return False
    tokenA = tokenA
    tokenB = tokenB
    similar = True
    for a in tokenA:
        for b in tokenB:
            if a == b:
                if a in trivial_words:
                    continue
                else:
                    intersection += 1
            elif levenshteinDistance(a, b)/float(max(len(a),len(b))) < 0.1:
                intersection += 1
            if intersection > 0:
                similar = True
                break
    return similar


# apply blocking rules
def blck_rule1(tp, dfa, dfb):
    t = tp[1]
    A_id = t.A_id
    B_id = t.B_id
    tuple_a = dfa.loc[A_id]
    tuple_b = dfb.loc[B_id]
    # check similarity of year, nan will always recognized as similar
    similar_year = not(abs(tuple_a['year'] - tuple_b['year']) > 1)
    similar_name = similar_string(tuple_a['name'], tuple_b['name'])
    if not similar_year:
        return False
    else:
        if not similar_name:
            return False
    return True

if __name__ == "__main__":
    import pandas as pd
    dfa = pd.read_csv(table_a, na_values=['nan', 'I', 'II'])
    dfb = pd.read_csv(table_b, na_values=['nan', 'I', 'II'])
    dfc = pd.read_csv(candidate_set)
    dfa = dfa.set_index('_id')
    dfb = dfb.set_index('_id')
    dfa['name'] = dfa['name'].apply(lambda x: x.lower())
    dfb['name'] = dfb['name'].apply(lambda x: x.lower())
    dfc2 = dfc[map(lambda x: blck_rule1(x, dfa, dfb), dfc.iterrows())]
    dfc2.to_csv('candidate_set2.csv', index=False)
