"""MFA English phoneme set

+ phoneme to numeric sequence conversion
"""

PAD = "_"
BOS = "^"
EOS = "$"

# phonemes = valid_symbols + 'spn' + 'sil' + 'sp'
phonemes = [
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "spn",
    "sil",
    "sp",
]

symbols = [PAD, BOS, EOS] + phonemes
symbol2id = {s: i for i, s in enumerate(symbols)}


def symbol_to_id(symbol):
    return symbol2id[symbol]


def id_to_symbol(idnum):
    return symbols[idnum]


def num_vocab():
    return len(symbols)


def text_to_sequence(text, add_special_token=True):
    """Phoeneme to numeric sequence conversion

    Args:
        text: Input phoneme sequence
        add_special_token: whether add BOS and EOS

    Returns:
        seq: Sequence of phoneme IDs
    """
    seq = []

    if add_special_token:
        # BOS
        seq.append(symbol_to_id(BOS))

    for ph in text.split():
        seq.append(symbol_to_id(ph))

    if add_special_token:
        # EOS token
        seq.append(symbol_to_id(EOS))

    return seq


def sequence_to_text(seq, remove_special_token=False):
    """Phoeneme to numeric sequence conversion

    Args:
        seq: Input phoneme id sequence
        remove_special_token: whether remove BOS and EOS

    Returns:
        text: Seq of phonemes
    """
    if remove_special_token:
        return [id_to_symbol(s) for s in seq[1:-1]]
    else:
        return [id_to_symbol(s) for s in seq]
