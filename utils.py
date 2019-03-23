# coding: utf-8

import numpy as np

from dataset import decode_map

def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0

    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    max_s_len = 0
    for n, seq in enumerate(sequences):
        s_len = len(seq)
        max_s_len = max(s_len, max_s_len)
        indices.extend(zip([n] * s_len, range(s_len)))
        values.extend(seq[0:s_len])

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), max_s_len], dtype=np.int64)

    return indices, values, shape

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)

    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))

    return result

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        s = decode_map[spars_tensor[1][m]]
        decoded.append(s)
    return decoded
