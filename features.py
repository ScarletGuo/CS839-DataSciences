import re


single_token_feature = ['tag', 'pos', 'dep', 'is_punct', 'is_space', 
                        'is_stop', 'is_alpha', 'ent_type','lemma', 
                        'is_quote', 'lex_id']

prefix = ['king', 'duke', 'lord', 'dowager', 'master']


def get_span_features(token):
    features = {}
    for attr in single_token_feature:
        features['span_'+attr] = getattr(token, attr)
    return features


def get_features(tokens, this_tokens, gram_idx, grams, sentence, span):
    #'upcase_ratio': find_ngram_upcase(grams),
    features = {
        'has_quote_s': u"'s" in " ".join(grams),
        'next_has_quote_s': u"'s" in get_next(tokens, gram_idx, 'text'),
        'pre_has_prefix': get_prev(tokens, gram_idx, 'text').lower() in prefix,
        'has_prefix': has_prefix(grams)
    }
    for attr in single_token_feature:
        features['prev_'+attr] = get_prev(tokens, gram_idx, attr)
        features['next_'+attr] = get_next(tokens, gram_idx, attr)
    return features


def filtering(items, doc_id):
    assert(len(items) > 0)
    for item in items:
        if not ('A' <= item[0] <= 'Z'): 
            return True
        if has_non_ascii(item):
            print("NON-ASCII in document {}".format(doc_id))
            return True
    return False


def has_prefix(grams):
    for t in grams:
        if t.lower() in prefix:
            return True
    return False
    
    
def has_non_ascii(string):
    return string is not None and any([ord(s) >= 128 for s in string])
    

def find_ngram_upcase(items):
    total = 0
    for item in items:
        if 'A' <= item[0] <= 'Z':
            total += 1
    return float(total) / len(items)


def get_prev(tokens, gram_idx, attr):
    first_idx = gram_idx[0]
    if first_idx-1 < 0:
        if attr == "text":
            return ""
        return 0
    return getattr(tokens[first_idx-1], attr)


def get_next(tokens, gram_idx, attr):
    last_idx = gram_idx[-1]
    if last_idx+1 >= len(tokens):
        if attr == "text":
            return ""
        return 0
    return getattr(tokens[last_idx+1], attr)


