"""Streaming rANS (range Asymmetric Numeral Systems) implementation

NOTE: Detailed algorithm description and discussion is on the wiki page:
https://github.com/kedartatwawadi/stanford_compression_library/wiki/Asymmetric-Numeral-Systems

## Core idea
- the theoretical rANS Encoder maintains an integer `state` 
- For each symbol s, the state is updated by calling: 
    ```python
    # encode step
    state = rans_base_encode_step(s, state)
    ```
    the decoder does the reverse by decoding the s and retrieving the prev state
    ```python
    # decode step
    s, state = rans_base_decode_step(state)
    ```
- In the theoretical rANS version, the state keeps on increasing every time we call `rans_base_encode_step`
  To make this practical, the rANS encoder ensures that after each encode step, the `state` lies in the acceptable range
  `[L, H]`, where `L,H` are predefined interval values.

  ```
  state lies in [L, H], after every encode step
  ```
  To ensure this happens the encoder shrinks the `state` by streaming out its lower bits, *before* encoding each symbol. 
  This logic is implemented in the function `shrink_state`. Thus, the full encoding step for one symbol is as follows:

  ```python
    ## Encoding one symbol
    # output bits to the stream to bring the state in the range for the next encoding
        state, out_bits = self.shrink_state(state, s)
        encoded_bitarray = out_bits + encoded_bitarray

    # core encoding step
    state = self.rans_base_encode_step(s, state)
  ```

  The decoder does the reverse operation of `expand_state` where, after decoding a symbol, it reads in the a few bits to
  re-map and expand the state to lie within the acceptable range [L, H]
  Note that `shrink_state` and `expand_state` are inverses of each other

  Thus, the  full decoding step 
  ```python
    # base rANS decoding step
    s, state = self.rans_base_decode_step(state)

    # remap the state into the acceptable range
    state, num_bits_used_by_expand_state = self.expand_state(state, encoded_bitarray)
  ```
- For completeness: the acceptable range `[L, H]` are given by:
  L = RANGE_FACTOR*total_freq
  H = (2**NUM_BITS_OUT)*RANGE_FACTOR*total_freq - 1)
  Why specifically these values? Look at the *Streaming-rANS* section on https://kedartatwawadi.github.io/post--ANS/


## References
1. Original Asymmetric Numeral Systems paper:  https://arxiv.org/abs/0902.0271
2. https://github.com/kedartatwawadi/stanford_compression_library/wiki/Asymmetric-Numeral-Systems
More references in the wiki article
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple, Any, List
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.utils.bitarray_utils import (
    BitArray,
    get_bit_width,
    uint_to_bitarray,
    bitarray_to_uint,
)
from scl.core.data_block import DataBlock
from scl.core.prob_dist import Frequencies, get_avg_neg_log_prob
from scl.utils.test_utils import get_random_data_block, try_lossless_compression
from scl.utils.misc_utils import cache
import time
from tqdm import tqdm
from os import scandir
from random import randint, seed
# stats for graphs
ENCODING_TIMES = []
DECODING_TIMES = []
@dataclass
class rANSParams:
    """base parameters for the rANS encoder/decoder.
    More details in the overview
    """

    ## define global params
    freqs: Frequencies

    # num bits used to represent the data_block size
    DATA_BLOCK_SIZE_BITS: int = 32

    # the encoder can output NUM_BITS_OUT at a time when it performs the state shrinking operation
    NUM_BITS_OUT: int = 1  # number of bits

    # rANS state is limited to the range [RANGE_FACTOR*total_freq, (2**NUM_BITS_OUT)*RANGE_FACTOR*total_freq - 1)]
    # RANGE_FACTOR is a base parameter controlling this range
    RANGE_FACTOR: int = 1 << 16

    def __post_init__(self):
        ## define derived params
        # M -> sum of frequencies
        self.M = self.freqs.total_freq

        # the state always lies in the range [L,H]
        self.L = self.RANGE_FACTOR * self.M
        self.H = self.L * (1 << self.NUM_BITS_OUT) - 1

        # define min max range for shrunk_state (useful during encoding)
        self.min_shrunk_state = {}
        self.max_shrunk_state = {}
        for s in self.freqs.alphabet:
            f = self.freqs.frequency(s)
            self.min_shrunk_state[s] = self.RANGE_FACTOR * f
            self.max_shrunk_state[s] = self.RANGE_FACTOR * f * (1 << self.NUM_BITS_OUT) - 1

        ## define initial state, state bits etc.
        # NOTE: the choice of  this state is somewhat arbitrary, the only condition being, it should lie in the acceptable range [L, H]
        self.INITIAL_STATE = self.L

        # define num bits used to represent the final state
        self.NUM_STATE_BITS = get_bit_width(self.H)
        self.BITS_OUT_MASK = 1 << self.NUM_BITS_OUT


class rANSEncoder(DataEncoder):
    """rANS Encoder

    Detailed information in the overview
    """

    def __init__(self, rans_params: rANSParams):
        """init function

        Args:
            freqs (Frequencies): frequencies for which rANS encoder needs to be designed
            rans_params (rANSParams): global rANS hyperparameters
        """
        self.params = rans_params

    def rans_base_encode_step(self, s, state: int):
        """base rANS encode step

        updates the state based on the input symbols s, and returns the updated state
        """
        f = self.params.freqs.frequency(s)
        block_id = state // f
        slot = self.params.freqs.cumulative_freq_dict[s] + (state % f)
        next_state = block_id * self.params.M + slot
        return next_state

    def shrink_state(self, state: int, next_symbol) -> Tuple[int, BitArray]:
        """stream out the lower bits of the state, until the state is below params.max_shrunk_state[next_symbol]"""
        out_bits = BitArray("")

        # output bits to the stream to bring the state in the range for the next encoding
        while state > self.params.max_shrunk_state[next_symbol]:
            _bits = uint_to_bitarray(
                state % (1 << self.params.NUM_BITS_OUT), bit_width=self.params.NUM_BITS_OUT
            )
            out_bits = _bits + out_bits
            state = state >> self.params.NUM_BITS_OUT

        return state, out_bits

    def encode_symbol(self, s, state: int) -> Tuple[int, BitArray]:
        """Encodes the next symbol, returns some bits and  the updated state

        Args:
            s (Any): next symbol to be encoded
            state (int): the rANS state

        Returns:
            state (int), symbol_bitarray (BitArray):
        """
        # output bits to the stream so that the state is in the acceptable range
        # [L, H] *after*the `rans_base_encode_step`
        symbol_bitarray = BitArray("")
        state, out_bits = self.shrink_state(state, s)

        # NOTE: we are prepending bits for pedagogy. In practice, it might be faster to assign a larger memory chunk and then fill it from the back
        # see: https://github.com/rygorous/ryg_rans/blob/c9d162d996fd600315af9ae8eb89d832576cb32d/main.cpp#L176 for example
        symbol_bitarray = out_bits + symbol_bitarray

        # core encoding step
        state = self.rans_base_encode_step(s, state)
        return state, symbol_bitarray

    def encode_block(self, data_block: DataBlock):
        print("Start Encoding")
        t = time.time()
        # initialize the output
        encoded_bitarray = BitArray("")

        # initialize the state
        state = self.params.INITIAL_STATE

        # update the state
        for s in tqdm(data_block.data_list):
            state, symbol_bitarray = self.encode_symbol(s, state)
            encoded_bitarray = symbol_bitarray + encoded_bitarray

        # Finally, pre-pend binary representation of the final state
        encoded_bitarray = uint_to_bitarray(state, self.params.NUM_STATE_BITS) + encoded_bitarray

        # add the data_block size at the beginning
        # NOTE: rANS decoding needs a way to indicate where to stop the decoding
        # One way is to add a character at the end which signals EOF. This requires us to
        # change the probabilities of the other symbols. Another way is to just signal the size of the
        # block. These two approaches add a bit of overhead.. the approach we use is much more transparent
        encoded_bitarray = (
            uint_to_bitarray(data_block.size, self.params.DATA_BLOCK_SIZE_BITS) + encoded_bitarray
        )
        end_time = time.time()-t
        print(f"End Encoding: {end_time}")
        ENCODING_TIMES.append(end_time)
        return encoded_bitarray


class rANSDecoder(DataDecoder):
    def __init__(self, rans_params: rANSParams):
        self.params = rans_params

    @staticmethod
    def find_bin(cumulative_freqs_list: List, slot: int) -> int:
        """Performs binary search over cumulative_freqs_list to locate which bin
        the slot lies.

        Args:
            cumulative_freqs_list (List): the sorted list of cumulative frequencies
                For example: freqs_list = [2,7,3], cumulative_freqs_list [0,2,9]
            slot (int): the value to search in the sorted list

        Returns:
            bin: the bin in which the slot lies
        """
        # NOTE: side="right" corresponds to search of type a[i-1] <= t < a[i]
        bin = np.searchsorted(cumulative_freqs_list, slot, side="right") - 1
        return int(bin)

    def rans_base_decode_step(self, state: int):
        block_id = state // self.params.M
        slot = state % self.params.M

        # decode symbol
        cum_prob_list = list(self.params.freqs.cumulative_freq_dict.values())
        symbol_ind = self.find_bin(cum_prob_list, slot)
        s = self.params.freqs.alphabet[symbol_ind]

        # retrieve prev state
        prev_state = (
            block_id * self.params.freqs.frequency(s)
            + slot
            - self.params.freqs.cumulative_freq_dict[s]
        )
        return s, prev_state

    def expand_state(self, state: int, encoded_bitarray: BitArray) -> Tuple[int, int]:
        # remap the state into the acceptable range
        num_bits = 0
        while state < self.params.L:
            state_remainder = bitarray_to_uint(
                encoded_bitarray[num_bits : num_bits + self.params.NUM_BITS_OUT]
            )
            num_bits += self.params.NUM_BITS_OUT
            state = (state << self.params.NUM_BITS_OUT) + state_remainder
        return state, num_bits

    def decode_symbol(self, state: int, encoded_bitarray: BitArray):
        # base rANS decoding step
        s, state = self.rans_base_decode_step(state)

        # remap the state into the acceptable range
        state, num_bits_used_by_expand_state = self.expand_state(state, encoded_bitarray)
        return s, state, num_bits_used_by_expand_state

    def decode_block(self, encoded_bitarray: BitArray):
        print("Start Decoding")
        t = time.time()
        # get data block size
        data_block_size_bitarray = encoded_bitarray[: self.params.DATA_BLOCK_SIZE_BITS]
        input_data_block_size = bitarray_to_uint(data_block_size_bitarray)
        num_bits_consumed = self.params.DATA_BLOCK_SIZE_BITS

        # get the final state
        state = bitarray_to_uint(
            encoded_bitarray[num_bits_consumed : num_bits_consumed + self.params.NUM_STATE_BITS]
        )
        num_bits_consumed += self.params.NUM_STATE_BITS

        # perform the decoding
        decoded_data_list = []
        for _ in tqdm(range(input_data_block_size)):
            s, state, num_symbol_bits = self.decode_symbol(
                state, encoded_bitarray[num_bits_consumed:]
            )

            # rANS decoder decodes symbols in the reverse direction,
            # so we add newly decoded symbol at the beginning
            decoded_data_list = [s] + decoded_data_list
            num_bits_consumed += num_symbol_bits

        # Finally, as a sanity check, ensure that the end state should be equal to the initial state
        assert state == self.params.INITIAL_STATE
        end_time = time.time() - t
        print(f"End Decoding: {end_time}")
        DECODING_TIMES.append(end_time)
        return DataBlock(decoded_data_list), num_bits_consumed


######################################## TESTS ##########################################


def test_check_encoded_bitarray():
    # test a specific example to check if the bitstream is as expected
    freq = Frequencies({"A": 3, "B": 3, "C": 2})
    data = DataBlock(["A", "C", "B"])
    params = rANSParams(freq, DATA_BLOCK_SIZE_BITS=5, NUM_BITS_OUT=1, RANGE_FACTOR=1)

    # NOTE: the encoded_bitstream looks like = [<data_size_bits>, <final_state_bits>,<s_n-1_bits>, <s_n-2_bits>, ..., <s0_bits>]
    # as the bits for symbols are prepended.
    ## Lets manually encode to find intermediate state etc:
    M = 8  # freq.total_freq
    L = 8  # = Mt
    H = 15  # = 2Mt-1

    expected_encoded_bitarray = BitArray("")

    # lets start with defining the initial_state
    x = 8
    assert params.INITIAL_STATE == 8

    ## encode symbol 1 = A
    # step-1: shrink state x to be in [3, 5]
    x = 4  # x = x//2
    expected_encoded_bitarray = BitArray("0") + expected_encoded_bitarray

    # step-2: rANS base encoding step
    x = 9  # x = (x//3)*8 + 0 + (x%3)

    ## encode symbol 2 = C
    # step-1: shrink state x to be in [2, 3]
    x = 2  # x = x//4
    expected_encoded_bitarray = BitArray("01") + expected_encoded_bitarray

    # step-2: rANS base encoding step
    x = 14  # x = (x//2)*8 + 6 + (x%2)

    ## encode symbol 3 = B
    # step-1: shrink state x to be in [3, 5]
    x = 3  # x = x//4
    expected_encoded_bitarray = BitArray("10") + expected_encoded_bitarray

    # step-2: rANS base encoding step
    x = 11  # x = (x//3)*8 + 3 + (x%3)

    ## prepnd the final state to the bitarray
    num_state_bits = 4  # log2(15)
    assert params.NUM_STATE_BITS == num_state_bits
    expected_encoded_bitarray = BitArray("1011") + expected_encoded_bitarray

    # append number of symbols = 3 using params.DATA_BLOCK_SIZE_BITS
    expected_encoded_bitarray = BitArray("00011") + expected_encoded_bitarray

    ################################

    ## Now lets encode using the encode_block and see it the result matches
    encoder = rANSEncoder(params)
    encoded_bitarray = encoder.encode_block(data)

    assert expected_encoded_bitarray == encoded_bitarray


def test_rANS_coding():
    # List different distributions, rANS params to test
    # trying out some random frequencies
    freqs_list = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 12, "B": 34, "C": 1, "D": 45}),
        Frequencies({"A": 34, "B": 35, "C": 546, "D": 1, "E": 13, "F": 245}),
        Frequencies({"A": 5, "B": 5, "C": 5, "D": 5, "E": 5, "F": 5}),
        Frequencies({"A": 1, "B": 3}),
    ]
    params_list = [
        rANSParams(freqs_list[0]),
        rANSParams(freqs_list[1]),
        rANSParams(freqs_list[2], NUM_BITS_OUT=8),
        rANSParams(freqs_list[3], RANGE_FACTOR=1 << 12),
        rANSParams(freqs_list[4], RANGE_FACTOR=1 << 4),
    ]

    # generate random data and test if coding is lossless
    DATA_SIZE = 20000
    SEED = 0
    for freq, rans_params in zip(freqs_list, params_list):
        # generate random data
        prob_dist = freq.get_prob_dist()
        data_block = get_random_data_block(prob_dist, DATA_SIZE, seed=SEED)
        avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)

        # create encoder decoder
        encoder = rANSEncoder(rans_params)
        decoder = rANSDecoder(rans_params)

        # test lossless coding
        is_lossless, encode_len, _ = try_lossless_compression(
            data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
        )
        assert is_lossless
        # avg codelen ignoring the bits used to signal num data elements
        avg_codelen = encode_len / data_block.size
        print(f"rANS coding: avg_log_prob={avg_log_prob:.3f}, rANS codelen: {avg_codelen:.3f}")


def test_alias_rANS_coding():
    """
    Function to compare runtimes/compression performance against alias rANS implementation
    """
    files = []
    with scandir("project_data") as folder:
        for entry in folder:
            files.append(entry.name)
    files.sort()

    for i, f in enumerate(files):
        with open("project_data/" + f, "r") as file:
            print("Book:", f.replace("_", " ")[:-10].title())
            data = file.read()
            data_list = [i for i in data]
            data_block = DataBlock(data_list)
            freqs = Frequencies(dict(sorted(data_block.get_counts().items())))
            prob_dist = freqs.get_prob_dist()
            avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)
            # create encoder decoder
            encoder = rANSEncoder(rANSParams(freqs))
            decoder = rANSDecoder(rANSParams(freqs))

            # test lossless coding
            is_lossless, encode_len, _ = try_lossless_compression(
                data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
            )
            assert is_lossless
            # avg codelen ignoring the bits used to signal num data elements
            avg_codelen = encode_len / data_block.size
            print(f"rANS coding: avg_log_prob={avg_log_prob:.3f}, rANS codelen: {avg_codelen:.3f}\n")


def test_markov_rANS_coding():
    global ENCODING_TIMES
    global DECODING_TIMES
    data_sizes = [5000 * i for i in range(1, 11)]
    SEED = 0
    # set the seed for the tests
    seed(274)
    empirical_entropies, codelens = {data_size: [] for data_size in data_sizes}, {data_size: [] for data_size in data_sizes}
    encoding_times, decoding_times = {data_size: [] for data_size in data_sizes}, {data_size: [] for data_size in data_sizes}
    for DATA_SIZE in data_sizes:
        for _ in range(10):
            # random dependencies
            dct = {}
            alphabet = ["A", "B", "C", "D"]
            for key in alphabet:
                for key2 in alphabet:
                    dct[key+key2] = randint(50, 1000)
            freqs = Frequencies(dct)

            # get a random data block from these 1st order frequencies
            prob_dist = freqs.get_prob_dist()
            data_block = get_random_data_block(prob_dist, DATA_SIZE, seed=SEED)

            # make data_block have symbols of length 1, not 2
            tmp_list = data_block.data_list
            true_data_list = []
            for two_tuple in tmp_list:
                true_data_list.append(two_tuple[0])
                true_data_list.append(two_tuple[1])
            data_block = DataBlock(true_data_list)

            # re-create frequencies, to make them for the 1-symbol alphabet
            freqs = Frequencies({symbol: data_block.data_list.count(symbol) for symbol in alphabet})
            # # below is equivalent to just gettng avg_log_prob
            # empirical_entropy = 0
            # for key in freqs.freq_dict:
            #     prob = freqs.freq_dict[key] / (2 * DATA_SIZE)
            #     empirical_entropy += prob * np.log2(1/prob)

            prob_dist = freqs.get_prob_dist()
            avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)
            # start with complete knowledge of frequencies
            encoder = rANSEncoder(rANSParams(freqs))
            decoder = rANSDecoder(rANSParams(freqs))

            # test lossless coding
            is_lossless, encode_len, _ = try_lossless_compression(
                data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
            )
            assert is_lossless
            # avg codelen ignoring the bits used to signal num data elements
            avg_codelen = encode_len / data_block.size
            print(f"rANS coding: empirical entropy={avg_log_prob:.3f}, rANS codelen: {avg_codelen:.3f}")
            empirical_entropies[DATA_SIZE].append(avg_log_prob)
            codelens[DATA_SIZE].append(avg_codelen)
            encoding_times[DATA_SIZE] += ENCODING_TIMES
            ENCODING_TIMES = []
            decoding_times[DATA_SIZE] += DECODING_TIMES
            DECODING_TIMES = []

    encoding_averages, decoding_averages, empirical_entropies_averages, codelens_averages = [], [], [], []
    for data_size in data_sizes:
        encoding_averages.append(np.mean(encoding_times[data_size]))
        decoding_averages.append(np.mean(decoding_times[data_size]))
        empirical_entropies_averages.append(np.mean(empirical_entropies[data_size]))
        codelens_averages.append(np.mean(codelens[data_size]))

    with open("data_standard.txt", "w") as f:
        f.write(f"{encoding_averages}\n{decoding_averages}\n{empirical_entropies_averages}\n{codelens_averages}\n")
    return encoding_averages, decoding_averages, empirical_entropies_averages, codelens_averages
