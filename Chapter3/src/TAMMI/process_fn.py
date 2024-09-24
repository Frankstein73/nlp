import sys
import spacy
import os
import csv


def analyze_text(text, morpho_lex_path):
    proc = spacy.load("en_core_web_sm")

    def safe_divide(a, b):
        if b != 0:
            return a / b
        else:
            return 0

    from csv import DictReader

    morph_dict = {}  # create blank dictionary

    with open(morpho_lex_path, mode="r") as inp:
        reader = csv.reader(inp)
        morph_dict = {rows[0]: rows[1:76] for rows in reader}

    # csv is read in as strings
    # Change all values to float if possible.
    # iterate through dictionary key value pairs
    for key, val in morph_dict.items():
        # create empty list that will replace current 'v' (=list)
        transformed_vals = []
        # iterate through current 'v' (=list)
        for v in val:
            # change element to float if possible
            try:
                newval = float(v)
            # if not (= if the value cannot be converted to type float), keep the current element
            except ValueError:
                newval = v
            # append the element (whether it has been changed or not) to the empty list
            transformed_vals.append(newval)
        # replace current 'v' with the new list
        morph_dict[key] = transformed_vals

    spacy_text = proc(text)  # process texts through spaCy
    stopwords = spacy.lang.en.stop_words.STOP_WORDS  # call in stop words


    cw = []  # list for the content words to be stored. Used for everything later
    final_result_list_cw = []  # list of results

    # bunch of list for storing each result
    tokens_w_inflections = []
    tokens_w_derivations = []
    tokens_w_prefixes = []
    tokens_w_suffixes = []
    compounds = []
    num_prefix = []
    num_root = []
    num_suffix = []
    num_affix = []
    num_root_affix = []
    num_root_affix_inflec = []
    prefix_PFMF = []
    prefix_fam_size = []
    prefix_freq = []
    prefix_log_freq = []
    prefix_len = []
    prefix_in_hapax = []
    hapax_in_prefix = []
    root_PFMF = []
    root_fam_size = []
    root_freq = []
    root_log_freq = []
    suffix_PFMF = []
    suffix_fam_size = []
    suffix_freq = []
    suffix_log_freq = []
    suffix_len = []
    suffix_in_hapax = []
    hapax_in_suffix = []
    affix_PFMF = []
    affix_fam_size = []
    affix_freq = []
    affix_log_freq = []
    affix_len = []
    affix_in_hapax = []
    hapax_in_affix = []

    # for MCI later
    inflections = []

    for token in spacy_text:
        if not token.is_stop:  # remove stop words
            if token.is_alpha:  # remove numbers
                cw.append(str(token))  # save it as string and not spacy token
                #      print(token.text) #stings
                #      print(token.lemma_) #strings
                if len(token.text) > len(
                    token.lemma_
                ):  # is the token is longer than the lemma
                    inflect = token.text.replace(
                        token.lemma_, ""
                    )  # replace token with unshared part from lemma. Works unless token and lemma different (see 'say' and 'said'). Anyway, most of the time this gives you inflections to be used in MCI
                    inflections.append(inflect)
                else:
                    inflections.append("")
                if (
                    token.text != token.lemma_
                ):  # see if lemma is different than token. If so, token has an inflectional morpheme
                    tokens_w_inflections.append(1)
                try:
                    val = morph_dict[token.text]
                    # this calculates the words that have derivational affixes
                    if val[3] or val[5] >= 1:
                        tokens_w_derivations.append(1)
                    # this calculates number of words that have prefixes
                    if val[3] >= 1:
                        tokens_w_prefixes.append(1)
                        # print("true")
                    # this calculates words that have suffixes
                    if val[5] >= 1:
                        tokens_w_suffixes.append(1)
                    # this calculates number of compounds (i.e., more than 1 root)
                    if val[4] > 1:
                        # print("compound")
                        compounds.append(1)

                    # number of affixes and root
                    # here calculate number of prefixes
                    num_prefix.append(val[3])
                    # here calculate number of roots
                    num_root.append(val[4])
                    # here calculate of suffixes
                    num_suffix.append(val[5])
                    # here calculate of affixes
                    num_affix.append(val[3])
                    num_affix.append(val[5])
                    # here calculate roots and affixes
                    num_root_affix.append(val[4])
                    num_root_affix.append(val[3])
                    num_root_affix.append(val[5])

                    # prefixes
                    # here calculate Percentage of more frequent words in the morphological family (prefix)
                    prefix_PFMF.append(val[7])
                    prefix_PFMF.append(val[14])
                    prefix_PFMF.append(val[21])
                    # here we calculate family size for prefixes (i.e., how many roots the prefix attaches to)
                    prefix_fam_size.append(val[8])
                    prefix_fam_size.append(val[15])
                    prefix_fam_size.append(val[22])
                    # here we calculate frequency of prefixes
                    prefix_freq.append(val[9])
                    prefix_freq.append(val[16])
                    prefix_freq.append(val[23])
                    # here we calculate log frequency of prefixes
                    prefix_log_freq.append(val[10])
                    prefix_log_freq.append(val[17])
                    prefix_log_freq.append(val[24])
                    # here we calculate prefix length
                    prefix_len.append(val[11])
                    prefix_len.append(val[18])
                    prefix_len.append(val[25])
                    # here we calculate prefix in hapax
                    prefix_in_hapax.append(val[12])
                    prefix_in_hapax.append(val[19])
                    prefix_in_hapax.append(val[26])
                    # here we calculate hapax in prefix
                    hapax_in_prefix.append(val[13])
                    hapax_in_prefix.append(val[20])
                    hapax_in_prefix.append(val[27])

                    # roots
                    # here calculate Percentage of more frequent words in the root
                    root_PFMF.append(val[28])
                    root_PFMF.append(val[32])
                    root_PFMF.append(val[36])
                    # here we calculate family size for roots
                    root_fam_size.append(val[29])
                    root_fam_size.append(val[33])
                    root_fam_size.append(val[37])
                    # here we calculate frequency of roots
                    root_freq.append(val[30])
                    root_freq.append(val[34])
                    root_freq.append(val[38])
                    # here we calculate log frequency of roots
                    root_log_freq.append(val[31])
                    root_log_freq.append(val[35])
                    root_log_freq.append(val[39])

                    # suffixes
                    # here calculate Percentage of more frequent words in the morphological family (suffix)
                    suffix_PFMF.append(val[40])
                    suffix_PFMF.append(val[47])
                    suffix_PFMF.append(val[54])
                    suffix_PFMF.append(val[61])
                    # here we calculate family size for suffix (i.e., how many roots the suffix attaches to)
                    suffix_fam_size.append(val[41])
                    suffix_fam_size.append(val[48])
                    suffix_fam_size.append(val[55])
                    suffix_fam_size.append(val[62])
                    # here we calculate frequency of suffixes
                    suffix_freq.append(val[42])
                    suffix_freq.append(val[49])
                    suffix_freq.append(val[56])
                    suffix_freq.append(val[63])
                    # here we calculate log frequency of suffix
                    suffix_log_freq.append(val[43])
                    suffix_log_freq.append(val[50])
                    suffix_log_freq.append(val[57])
                    suffix_log_freq.append(val[64])
                    # here we calculate suffix length
                    suffix_len.append(val[44])
                    suffix_len.append(val[51])
                    suffix_len.append(val[58])
                    suffix_len.append(val[65])
                    # here we calculate suffix in hapax
                    suffix_in_hapax.append(val[45])
                    suffix_in_hapax.append(val[52])
                    suffix_in_hapax.append(val[59])
                    suffix_in_hapax.append(val[66])
                    # here we calculate hapax in suffix
                    hapax_in_suffix.append(val[46])
                    hapax_in_suffix.append(val[53])
                    hapax_in_suffix.append(val[60])
                    hapax_in_suffix.append(val[67])

                    # affixes
                    # here calculate Percentage of more frequent words in the morphological family (affixes)
                    affix_PFMF.append(val[7])
                    affix_PFMF.append(val[14])
                    affix_PFMF.append(val[21])
                    affix_PFMF.append(val[40])
                    affix_PFMF.append(val[47])
                    affix_PFMF.append(val[54])
                    affix_PFMF.append(val[61])
                    # here we calculate family size for affixes (i.e., how many roots the affixes attaches to)
                    affix_fam_size.append(val[8])
                    affix_fam_size.append(val[15])
                    affix_fam_size.append(val[22])
                    affix_fam_size.append(val[41])
                    affix_fam_size.append(val[48])
                    affix_fam_size.append(val[55])
                    affix_fam_size.append(val[62])
                    # here we calculate frequency of affixes
                    affix_freq.append(val[9])
                    affix_freq.append(val[16])
                    affix_freq.append(val[23])
                    affix_freq.append(val[42])
                    affix_freq.append(val[49])
                    affix_freq.append(val[56])
                    affix_freq.append(val[63])
                    # here we calculate log frequency of affixes
                    affix_log_freq.append(val[10])
                    affix_log_freq.append(val[17])
                    affix_log_freq.append(val[24])
                    affix_log_freq.append(val[43])
                    affix_log_freq.append(val[50])
                    affix_log_freq.append(val[57])
                    affix_log_freq.append(val[64])
                    # here we calculate affix length
                    affix_len.append(val[11])
                    affix_len.append(val[18])
                    affix_len.append(val[25])
                    affix_len.append(val[44])
                    affix_len.append(val[51])
                    affix_len.append(val[58])
                    affix_len.append(val[65])
                    # here we calculate affix in hapax
                    affix_in_hapax.append(val[12])
                    affix_in_hapax.append(val[19])
                    affix_in_hapax.append(val[26])
                    affix_in_hapax.append(val[45])
                    affix_in_hapax.append(val[52])
                    affix_in_hapax.append(val[59])
                    affix_in_hapax.append(val[66])
                    # here we calculate hapax in affix
                    hapax_in_affix.append(val[13])
                    hapax_in_affix.append(val[20])
                    hapax_in_affix.append(val[27])
                    hapax_in_affix.append(val[46])
                    hapax_in_affix.append(val[53])
                    hapax_in_affix.append(val[60])
                    hapax_in_affix.append(val[67])
                except:
                    pass

    length = len(cw)

    # print(cw)
    # print(len(tokens_w_inflections))
    # print(len(tokens_w_derivations))
    # print(len(tokens_w_prefixes))
    # print(len(tokens_w_suffixes))
    # print(sum(num_prefix))
    # print(sum(num_root))
    # print(sum(num_suffix))
    # print(prefix_PFMF)
    # print(prefix_fam_size)
    # print(prefix_freq)
    # print(prefix_log_freq)
    # print(prefix_len)
    # print(prefix_in_hapax)
    # print(hapax_in_prefix)
    # print(root_PFMF)
    # print(root_fam_size)
    # print(root_freq)
    # print(root_log_freq)

    final_result_list_cw.append(safe_divide(len(tokens_w_inflections), length))
    final_result_list_cw.append(safe_divide(len(tokens_w_derivations), length))
    final_result_list_cw.append(safe_divide(len(tokens_w_prefixes), length))
    final_result_list_cw.append(safe_divide(len(tokens_w_suffixes), length))
    final_result_list_cw.append(safe_divide(len(compounds), length))

    final_result_list_cw.append(safe_divide(sum(num_prefix), length))
    final_result_list_cw.append(safe_divide(sum(num_root), length))
    final_result_list_cw.append(safe_divide(sum(num_suffix), length))
    final_result_list_cw.append(safe_divide(sum(num_affix), length))
    final_result_list_cw.append(safe_divide(sum(num_root_affix), length))

    # calculate number of roots, affixes, and inflectional morphemes
    num_root_affix_inflec = tokens_w_inflections + num_root_affix
    final_result_list_cw.append(safe_divide(len(num_root_affix_inflec), length))

    ##########function to remove all empty string for prefix, suffix, affix varaible and calculate norm variables

    def cal_fixes(fix_var):
        fix_var = [x for x in fix_var if x != ""]
        # print(fix_var)
        final_result_list_cw.append(sum(fix_var) / length)

    cal_fixes(prefix_PFMF)
    cal_fixes(prefix_fam_size)
    cal_fixes(prefix_freq)
    cal_fixes(prefix_log_freq)
    cal_fixes(prefix_len)
    cal_fixes(prefix_in_hapax)
    cal_fixes(hapax_in_prefix)
    cal_fixes(root_PFMF)
    cal_fixes(root_fam_size)
    cal_fixes(root_freq)
    cal_fixes(root_log_freq)
    cal_fixes(suffix_PFMF)
    cal_fixes(suffix_fam_size)
    cal_fixes(suffix_freq)
    cal_fixes(suffix_log_freq)
    cal_fixes(suffix_len)
    cal_fixes(suffix_in_hapax)
    cal_fixes(hapax_in_suffix)
    cal_fixes(affix_PFMF)
    cal_fixes(affix_fam_size)
    cal_fixes(affix_freq)
    cal_fixes(affix_log_freq)
    cal_fixes(affix_len)
    cal_fixes(affix_in_hapax)
    cal_fixes(hapax_in_affix)

    # ========================================================================
    # MW10 done in the same loop
    # ========================================================================

    ###########work on inflections first#############
    # create 10 word bins

    # print(inflections) #Note that inflections were calculated in spaCy earlier

    # function to break lists of morphemes into windows of n size
    def list_windows(list1, n):
        for i in range(0, len(list1), n):
            yield list1[i : i + n]

    # break inflections into lists of 10
    # inflections list was made available in the spaCy for loop above
    n = 10
    mw10 = list(
        list_windows(inflections, n)
    )  # call function above and returns it to list through list function
    # print(f'these are the lists of 10 word window inflections {mw10}')

    # this is for TTR, subsets, and mci
    inflec_10_types = []  # list for number of inflections per 10 word window
    inflec_10_tokens = (
        []
    )  # list for the length of the 10 word windows (because the last window is likely to be fewer than 10 words)
    inflec_10_subset = []

    for mw10_el in mw10:
        inflec_10_types.append(
            len(set(mw10_el)) - 1
        )  # this gets the length of unique elements per word window. -1 is to get rid of '' blank spaces (i.e., words without inflections)
        inflec_10_tokens.append(
            len(mw10_el)
        )  # this gets the length of the inflection morphemes
        inflec_10_subset.append(len(set(mw10_el)))  # this includes "no inflections"

    # print(f'this is the list of unique inflections {inflec_10_types}')
    # print(f'this is the length of each list {inflec_10_tokens}')
    # print(f'this is the list of unique inflections including no inflections {inflec_10_subset}')

    # this is to calculate TTR for inflections
    ttr_10_inflec = [
        a / b for a, b in zip(inflec_10_types, inflec_10_tokens)
    ]  # calculate TTR for each 10 word window
    inflec_10_ttr = safe_divide(
        sum(ttr_10_inflec), len(ttr_10_inflec)
    )  # average out scores for all windows

    # this is to calculate mean subset variety (MSV) for inflections. MSV = (number of unique inflections for each subset)/number of subsets
    msv_10_inflec = safe_divide(sum(inflec_10_subset), len(inflec_10_subset))

    # in practice, with longer texts, there will be no mean between subset diversity because it is unlikely that there will be a unique inflection
    # when there are 30 ten sample token windows. We should calculate anyway. Basically, need to calculate how many unique inflections there are across
    # all subsets of inflections. Smaller texts will have an advantage here. So... probably a shitty measure.

    # print(f'number of sublists {len(inflec_10_subset)}')

    # print(f'this is the unique inflections in the lists for 10 word tokens inflections {mw10}')

    sing_list_inflec = sum(mw10, [])  # put all the sublists into a single list

    # print(f'this is the list of all the inflection morphemes in the text {sing_list}')

    between_subset_div_inflec = []  # list for between subset diversity

    for i in sing_list_inflec:
        if (
            sing_list_inflec.count(i) == 1
        ):  # if the the item in the list only occurs once
            between_subset_div_inflec.append(i)  # append it

    # print(f'this is the mean subset variety {msv_10_inflec}')
    # print(f'these are the unique inflectional morphemes in the entire text {between_subset_div}')
    # print(f'this is the length of subset diversity {len(between_subset_div)}')

    mci_inflec = (msv_10_inflec + (len(between_subset_div_inflec))) / len(
        inflec_10_subset
    ) - 1  # this is the formula from Brezina and Pallotti

    # print(f'this is msv_10_inflec {msv_10_inflec}') #this is msv_10_inflec
    # print(f'this is inflection ttr {inflec_10_ttr}') #this is inflectional TTR
    # print(f'this is inflection mci {mci_inflec}') #this is MCI

    final_result_list_cw.append(msv_10_inflec)
    final_result_list_cw.append(inflec_10_ttr)
    final_result_list_cw.append(mci_inflec)

    #############now for derivational morphemes#####################

    deriv_10_types = (
        []
    )  # list for number of derivational affixes per 10 word window
    deriv_10_tokens = (
        []
    )  # list for the length of the 10 word windows (because the last window is likely to be fewer than 10 words)

    # grab up 10 word windows

    # print(type(cw[1]))
    # print(type(inflections[1]))

    cw10 = list(list_windows(cw, n))
    # print(f'these are the 10 token windows of words {cw10}')
    cw10_affixes = []

    for i in range(len(cw10)):
        affix_per_window = []
        for word in cw10[i]:
            #    print(word)
            try:
                morph_dict[word]
                #        print(f'this is the key {key}')
                #        print(f'this is the value {val}')
                #        print(f'this is the first prefix if one exists "{val[68]}"')
                affix_per_window.append(val[68])
                affix_per_window.append(val[69])
                affix_per_window.append(val[70])
                affix_per_window.append(val[71])
                affix_per_window.append(val[72])
                affix_per_window.append(val[73])
                affix_per_window.append(val[74])
            except:
                pass
        cw10_affixes.append(affix_per_window)

    # print(f'these are the lists of derivational morphemes: {cw10_affixes}')

    # remove all empty strings in lists. this is for list of lists. The join() method takes all items in an iterable and joins them into one string
    cw10_affixes = [" ".join(i).split() for i in cw10_affixes]

    # print(f'these are the lists of derivational morphemes: {cw10_affixes}')

    # print(f'these are the cleaned list of morphemes {cw10_affixes}')

    for cw10_aff_el in cw10_affixes:
        deriv_10_types.append(
            len(set(cw10_aff_el))
        )  # this get take the length of unique elements per word window.
        deriv_10_tokens.append(
            len(cw10_aff_el)
        )  # this gets the length of the inflection morphemes

    # print(f'this is the list of unique affixes {deriv_10_types}')
    # print(f'this is the length of each list {deriv_10_tokens}')

    # Added safe_divide
    ttr_10_deriv = [
        safe_divide(a, b) for a, b in zip(deriv_10_types, deriv_10_tokens)
    ]  # calculate TTR for each 10 word window
    deriv_10_ttr = safe_divide(
        sum(ttr_10_deriv), len(ttr_10_deriv)
    )  # average out scores for all windows

    #########now for MCI for derviational morphemes using cw10_affixes

    # this is to calculate mean subset variety (MSV) for derivational morphemes. MSV = (number of unique derivational morphemes for each subset)/number of subsets
    msv_10_deriv = safe_divide(sum(deriv_10_types), len(deriv_10_types))
    # print(f'msv derive = {msv_10_deriv}')

    sing_list_der = sum(cw10_affixes, [])  # put all the sublists into a single list

    # print(f'this is the list of all the derivational morphemes in the text {sing_list_der}')

    between_subset_div_der = []  # list for between subset diversity

    for i in sing_list_der:
        if (
            sing_list_der.count(i) == 1
        ):  # if the the item in the list only occurs once
            between_subset_div_der.append(i)  # append it

    # print(f'this is the list of individual morphemes not shared across list {between_subset_div_der}')

    mci_deriv = (msv_10_deriv + (len(between_subset_div_der))) / len(
        deriv_10_types
    ) - 1  # this is the formula from Brezina and Pallotti

    # print(ttr_10_deriv)
    # print(f'this is derivational TTR {deriv_10_ttr}')
    # print(f'this is derviational mean subset variety {msv_10_deriv}')
    # print(f'this is derivational mci {mci_deriv}')

    final_result_list_cw.append(msv_10_deriv)
    final_result_list_cw.append(deriv_10_ttr)
    final_result_list_cw.append(mci_deriv)

    # all_final_result_list_cw.append(final_result_list_cw)

    keys = [
        "Inflected_Tokens",
        "Derivational_Tokens",
        "Tokens_w_Prefixes",
        "Tokens_w_Affixes",
        "Compounds",
        "number_prefixes",
        "number_roots",
        "number_suffixes",
        "number_affixes",
        "num_roots_affixes",
        "num_root_affix_inflec",
        "%_more_freq_words_morpho-family_prefix",
        "prefix_family_size",
        "prefix_freq",
        "prefix_log_freq",
        "prefix_len",
        "prefix_in_hapax",
        "hapax_in_prefix",
        "%_more_freq_words_morpho-family_root",
        "root_family_size",
        "root_freq",
        "root_log_freq",
        "%_more_freq_words_morpho-family_suffix",
        "suffix_family_size",
        "suffix_freq",
        "suffix_log_freq",
        "suffix_len",
        "suffix_in_hapax",
        "hapax_in_suffix",
        "%_more_freq_words_morpho-family_affix",
        "affix_family_size",
        "affix_freq",
        "affix_log_freq",
        "affix_len",
        "affix_in_hapax",
        "hapax_in_affix",
        "mean subset inflectional variety (10)",
        "inflectional TTR (10)",
        "inflectional MCI (10)",
        "mean subset derivational variety (10)",
        "derivational TTR (10)",
        "derivational MCI (10)",
    ]

    result_dict = dict(zip(keys, final_result_list_cw))
    return result_dict

    