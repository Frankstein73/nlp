

def safe_divide(numerator, denominator):
    if denominator == 0 or denominator == 0.0:
        index = 0
    else:
        index = numerator / denominator
    return index


def indexer(header_list, index_list, name, index):
    header_list.append(name)
    index_list.append(index)


def resource_path(relative):
    import sys
    import os
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)


def tag_processor_spaCy(
    raw_text, adj_word_list_path, real_word_list_path
):  # uses default spaCy 2.016
    import spacy

    nlp = spacy.load("en_core_web_sm")
    adj_word_list = (
        open(resource_path(adj_word_list_path), "r", errors="ignore")
        .read()
        .split("\n")[:-1]
    )
    real_word_list = (
        open(resource_path(real_word_list_path), "r", errors="ignore")
        .read()
        .split("\n")[:-1]
    )
    noun_tags = ["NN", "NNS", "NNP", "NNPS"]  # consider whether to identify gerunds
    proper_n = ["NNP", "NNPS"]
    no_proper = ["NN", "NNS"]
    pronouns = ["PRP", "PRP$"]
    adjectives = ["JJ", "JJR", "JJS"]
    verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]
    adverbs = ["RB", "RBR", "RBS"]
    content = [
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "JJ",
        "JJR",
        "JJS",
    ]  # note that this is a preliminary list
    prelim_not_function = [
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "JJ",
        "JJR",
        "JJS",
        "RB",
        "RBR",
        "RBS",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "MD",
    ]
    pronoun_dict = {"me": "i", "him": "he", "her": "she"}
    punctuation = "`` '' ' . , ? ! ) ( % / - _ -LRB- -RRB- SYM : ;".split(" ")
    punctuation.append('"')
    lemma_list = []
    content_list = []
    function_list = []

    tagged_text = nlp(raw_text)

    for sent in tagged_text.sents:
        for token in sent:
            if token.tag_ in punctuation:
                continue
            if (
                token.text.lower() not in real_word_list
            ):  # lowered because real_word_list is lowered
                continue

            if token.tag_ in content:
                if token.tag_ in noun_tags:
                    content_list.append(token.lemma_ + "_cw_nn")
                    lemma_list.append(token.lemma_ + "_cw_nn")
                else:
                    content_list.append(token.lemma_ + "_cw")
                    lemma_list.append(token.lemma_ + "_cw")

            if token.tag_ not in prelim_not_function:
                if token.tag_ in pronouns:
                    if token.text.lower() in pronoun_dict:
                        function_list.append(pronoun_dict[token.text.lower()] + "_fw")
                        lemma_list.append(pronoun_dict[token.text.lower()] + "_fw")
                    else:
                        function_list.append(token.text.lower() + "_fw")
                        lemma_list.append(token.text.lower() + "_fw")
                else:
                    function_list.append(token.lemma_ + "_fw")
                    lemma_list.append(token.lemma_ + "_fw")

            if token.tag_ in verbs:
                if token.dep_ == "aux":
                    function_list.append(token.lemma_ + "_fw")
                    lemma_list.append(token.lemma_ + "_fw")

                elif token.lemma_ == "be":
                    function_list.append(token.lemma_ + "_fw")
                    lemma_list.append(token.lemma_ + "_fw")

                else:
                    content_list.append(token.lemma_ + "_cw_vb")
                    lemma_list.append(token.lemma_ + "_cw_vb")

            if token.tag_ in adverbs:
                if (
                    token.lemma_[-2:] == "ly" and token.lemma_[:-2] in adj_word_list
                ) or (
                    token.lemma_[-4:] == "ally" and token.lemma_[:-4] in adj_word_list
                ):
                    content_list.append(token.lemma_ + "_cw")
                    lemma_list.append(token.lemma_ + "_cw")
                else:
                    function_list.append(token.lemma_ + "_fw")
                    lemma_list.append(token.lemma_ + "_fw")
            # print(raw_token, lemma_list[-1])

    return {"lemma": lemma_list, "content": content_list, "function": function_list}


def lex_density(cw_text, fw_text):
    n_cw = len(cw_text)
    n_fw = len(fw_text)
    n_all = n_cw + n_fw

    n_type_cw = len(set(cw_text))
    n_type_fw = len(set(fw_text))
    n_all_type = n_type_cw + n_type_fw

    lex_dens_all = safe_divide(n_cw, n_all)  # percentage of content words
    lex_dens_cw_fw = safe_divide(n_cw, n_fw)  # ratio content words to function words

    lex_dens_all_type = safe_divide(
        n_type_cw, n_all_type
    )  # percentage of content words
    lex_dens_cw_fw_type = safe_divide(
        n_type_cw, n_type_fw
    )  # ratio content words to function words

    return [lex_dens_all, lex_dens_all_type]


def ttr(text):
    ntokens = len(text)
    ntypes = len(set(text))

    simple_ttr = safe_divide(ntypes, ntokens)
    root_ttr = safe_divide(ntypes, math.sqrt(ntokens))
    log_ttr = safe_divide(math.log10(ntypes), math.log10(ntokens))
    maas_ttr = safe_divide(
        (math.log10(ntokens) - math.log10(ntypes)), math.pow(math.log10(ntokens), 2)
    )

    return [simple_ttr, root_ttr, log_ttr, maas_ttr]


def simple_ttr(text):
    ntokens = len(text)
    ntypes = len(set(text))

    return safe_divide(ntypes, ntokens)


def root_ttr(text):
    ntokens = len(text)
    ntypes = len(set(text))

    return safe_divide(ntypes, math.sqrt(ntokens))


def log_ttr(text):
    ntokens = len(text)
    ntypes = len(set(text))

    return safe_divide(math.log10(ntypes), math.log10(ntokens))


def maas_ttr(text):
    ntokens = len(text)
    ntypes = len(set(text))

    return safe_divide(
        (math.log10(ntokens) - math.log10(ntypes)), math.pow(math.log10(ntokens), 2)
    )


def mattr(text, window_length=50):  # from TAACO 2.0.4

    if len(text) < (window_length + 1):
        ma_ttr = safe_divide(len(set(text)), len(text))

    else:
        sum_ttr = 0
        denom = 0
        for x in range(len(text)):
            small_text = text[x : (x + window_length)]
            if len(small_text) < window_length:
                break
            denom += 1
            sum_ttr += safe_divide(len(set(small_text)), float(window_length))
        ma_ttr = safe_divide(sum_ttr, denom)

    return ma_ttr


def msttr(text, window_length=50):

    if len(text) < (window_length + 1):
        ms_ttr = safe_divide(len(set(text)), len(text))

    else:
        sum_ttr = 0
        denom = 0

        n_segments = int(safe_divide(len(text), window_length))
        seed = 0
        for x in range(n_segments):
            sub_text = text[seed : seed + window_length]
            # print sub_text
            sum_ttr += safe_divide(len(set(sub_text)), len(sub_text))
            denom += 1
            seed += window_length

        ms_ttr = safe_divide(sum_ttr, denom)

    return ms_ttr


def hdd(text):
    # requires Counter import
    def choose(n, k):  # calculate binomial
        """
        A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
        """
        if 0 <= k <= n:
            ntok = 1
            ktok = 1
            for t in range(
                1, min(k, n - k) + 1
            ):  # this was changed to "range" from "xrange" for py3
                ntok *= n
                ktok *= t
                n -= 1
            return ntok // ktok
        else:
            return 0

    def hyper(
        successes, sample_size, population_size, freq
    ):  # calculate hypergeometric distribution
        # probability a word will occur at least once in a sample of a particular size
        try:
            prob_1 = 1.0 - (
                float(
                    (
                        choose(freq, successes)
                        * choose((population_size - freq), (sample_size - successes))
                    )
                )
                / float(choose(population_size, sample_size))
            )
            prob_1 = prob_1 * (1 / sample_size)
        except ZeroDivisionError:
            prob_1 = 0

        return prob_1

    prob_sum = 0.0
    ntokens = len(text)
    types_list = list(set(text))
    frequency_dict = Counter(text)

    for items in types_list:
        prob = hyper(
            0, 42, ntokens, frequency_dict[items]
        )  # random sample is 42 items in length
        prob_sum += prob

    return prob_sum


def mtld_original(input, min=10):
    def mtlder(text):
        factor = 0
        factor_lengths = 0
        start = 0
        for x in range(len(text)):
            factor_text = text[start : x + 1]
            if x + 1 == len(text):
                factor += safe_divide((1 - ttr(factor_text)[0]), (1 - 0.72))
                factor_lengths += len(factor_text)
            else:
                if ttr(factor_text)[0] < 0.720 and len(factor_text) >= min:
                    factor += 1
                    factor_lengths += len(factor_text)
                    start = x + 1
                else:
                    continue

        mtld = safe_divide(factor_lengths, factor)
        return mtld

    input_reversed = list(reversed(input))
    mtld_full = safe_divide((mtlder(input) + mtlder(input_reversed)), 2)
    return mtld_full


def mtld_bi_directional_ma(text, min=10):
    def mtld_ma(text, min=10):
        factor = 0
        factor_lengths = 0
        for x in range(len(text)):
            sub_text = text[x:]
            breaker = 0
            for y in range(len(sub_text)):
                if breaker == 0:
                    factor_text = sub_text[: y + 1]
                    if ttr(factor_text)[0] < 0.720 and len(factor_text) >= min:
                        factor += 1
                        factor_lengths += len(factor_text)
                        breaker = 1
                    else:
                        continue
        mtld = safe_divide(factor_lengths, factor)
        return mtld

    forward = mtld_ma(text)
    backward = mtld_ma(list(reversed(text)))

    mtld_bi = safe_divide(
        (forward + backward), 2
    )  # average of forward and backward mtld

    return mtld_bi


def mtld_ma_wrap(text, min=10):
    factor = 0
    factor_lengths = 0
    start = 0
    double_text = text + text  # allows wraparound
    for x in range(len(text)):
        breaker = 0
        sub_text = double_text[x:]
        for y in range(len(sub_text)):
            if breaker == 0:
                factor_text = sub_text[: y + 1]
                if ttr(factor_text)[0] < 0.720 and len(factor_text) >= min:
                    factor += 1
                    factor_lengths += len(factor_text)
                    breaker = 1
                else:
                    continue
    mtld = safe_divide(factor_lengths, factor)
    return mtld


if __name__ == "__main__":
    refined_lemma_dict = tag_processor_spaCy(raw_text)
    lemma_text_aw = refined_lemma_dict["lemma"]
    lemma_text_cw = refined_lemma_dict["content"]
    lemma_text_fw = refined_lemma_dict["function"]
