import re


single_token_feature = ['tag', 'pos', 'dep', 'is_punct', 'is_space', 
                        'is_stop', 'is_alpha', 'ent_type','lemma', 
                        'is_quote', 'lex_id']


def get_span_features(token):
    features = {}
    for attr in single_token_feature:
        features['span_'+attr] = getattr(token, attr)
    return features


def get_features(tokens, this_tokens, gram_idx, grams, sentence, span):
    features = {
        'upcase_ratio': find_ngram_upcase(grams),
        'has_quote_s': "'s" in " ".join(grams),
        'next_has_quote_s': get_next(tokens, gram_idx, 'text') == "'s",
        'pre_has_prefix': get_prev(tokens, gram_idx, 'text').lower() in ['king', 'duke', 'dowager'],
    }
    for attr in single_token_feature:
        features['prev_'+attr] = get_prev(tokens, gram_idx, attr)
        features['next_'+attr] = get_next(tokens, gram_idx, attr)
    return features


def find_ngram_upcase(items):
    assert(len(items) > 0)
    total = 0
    for item in items:
        if 'A' <= item[0] <= 'Z':
            total += 1
    return float(total) / len(items)


def get_prev(tokens, gram_idx, attr):
    first_idx = gram_idx[0]
    if first_idx-1 > len(tokens):
        return 0
    return getattr(tokens[first_idx-1], attr)


def get_next(tokens, gram_idx, attr):
    last_idx = gram_idx[-1]
    if last_idx+1 > len(tokens):
        return 0
    return getattr(tokens[last_idx+1], attr)


