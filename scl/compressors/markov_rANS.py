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
import copy
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
        # M -> power of 2
        self.M = 2**16

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
        # initialize this to just be the first symbol, which shouldn't matter in the long run
        self.prev_symbol = self.params.freqs.alphabet[0][0]
        self.true_freqs = copy.deepcopy(self.params.freqs)

    def update_internals(self):
        ratio = self.params.M / self.true_freqs.total_freq
        rounded_freqs = {}
        for s in self.params.freqs.alphabet:
            f = self.true_freqs.freq_dict[s]
            new_f = f * ratio
            rounded_freqs[s] = round(new_f)
        # make sure the rounded_freqs adds up to M, so we just add or subtract 1 randomly until they do
        i = 0
        while (sum(rounded_freqs.values())) > self.params.M:
            rounded_freqs[list(rounded_freqs.keys())[i]] -= 1
            i += 1
        while (sum(rounded_freqs.values())) < self.params.M:
            rounded_freqs[list(rounded_freqs.keys())[i % len(rounded_freqs)]] += 1
            i += 1
        # replace passed in freqs with our rounded_freqs
        self.params.freqs = Frequencies(rounded_freqs)
        # END MY CODE

        # the state always lies in the range [L,H]
        self.params.L = self.params.RANGE_FACTOR * self.params.M
        self.params.H = self.params.L * (1 << self.params.NUM_BITS_OUT) - 1

        # define min max range for shrunk_state (useful during encoding)
        self.params.min_shrunk_state = {}
        self.params.max_shrunk_state = {}
        for s in self.params.freqs.alphabet:
            f = self.params.freqs.frequency(s)
            self.params.min_shrunk_state[s] = self.params.RANGE_FACTOR * f
            self.params.max_shrunk_state[s] = self.params.RANGE_FACTOR * f * (1 << self.params.NUM_BITS_OUT) - 1

    def rans_base_encode_step(self, s, state: int):
        """base rANS encode step

        updates the state based on the input symbols s, and returns the updated state
        """
        # ADDED self.prev_symbol because our alphabet is 2 tuples not 1 symbol
        two_tuple = self.prev_symbol + s
        f = self.params.freqs.frequency(two_tuple)
        block_id = state // f
        slot = self.params.freqs.cumulative_freq_dict[two_tuple] + (state % f)
        next_state = block_id * self.params.M + slot
        return next_state

    def shrink_state(self, state: int, next_symbol) -> Tuple[int, BitArray]:
        """stream out the lower bits of the state, until the state is below params.max_shrunk_state[next_symbol]"""
        out_bits = BitArray("")

        # output bits to the stream to bring the state in the range for the next encoding
        # ADDED self.prev_symbol inside the subscript because our list is different
        while state > self.params.max_shrunk_state[self.prev_symbol + next_symbol]:
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
        # go forward, and build the freq_array
        for i, s in enumerate(data_block.data_list):
            curr_tuple = self.prev_symbol + s
            self.true_freqs.freq_dict[curr_tuple] += 1
            self.prev_symbol = s
        self.prev_symbol = data_block.data_list[-2]
        self.update_internals()
        # update the state, but going backwards
        backwards_data_block = DataBlock(list(reversed(data_block.data_list)))
        for i, s in tqdm(enumerate(backwards_data_block.data_list)):
            # get rid of the context because we are encoding backwards
            self.true_freqs.freq_dict[self.prev_symbol+s] -= 1
            self.update_internals()
            state, symbol_bitarray = self.encode_symbol(s, state)
            # tracking previous symbol
            if i < (len(backwards_data_block.data_list) - 2):
                self.prev_symbol = backwards_data_block.data_list[i+2]
            else:
                self.prev_symbol = self.params.freqs.alphabet[0][0]
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
        end_time = time.time() - t
        print(f"End Encoding: {end_time}")
        ENCODING_TIMES.append(end_time)
        return encoded_bitarray


class rANSDecoder(DataDecoder):
    def __init__(self, rans_params: rANSParams):
        self.params = rans_params
        self.prev_symbol = self.params.freqs.alphabet[0][0]
        self.true_freqs = copy.deepcopy(self.params.freqs)

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

    def update_internals(self):
        ratio = self.params.M / self.true_freqs.total_freq
        rounded_freqs = {}
        for s in self.params.freqs.alphabet:
            f = self.true_freqs.freq_dict[s]
            new_f = f * ratio
            rounded_freqs[s] = round(new_f)
        # make sure the rounded_freqs adds up to M, so we just add or subtract 1 randomly until they do
        i = 0
        while (sum(rounded_freqs.values())) > self.params.M:
            rounded_freqs[list(rounded_freqs.keys())[i]] -= 1
            i += 1
        while (sum(rounded_freqs.values())) < self.params.M:
            rounded_freqs[list(rounded_freqs.keys())[i % len(rounded_freqs)]] += 1
            i += 1
        # replace passed in freqs with our rounded_freqs
        self.params.freqs = Frequencies(rounded_freqs)
        # END MY CODE

        # the state always lies in the range [L,H]
        self.params.L = self.params.RANGE_FACTOR * self.params.M
        self.params.H = self.params.L * (1 << self.params.NUM_BITS_OUT) - 1

        # define min max range for shrunk_state (useful during encoding)
        self.params.min_shrunk_state = {}
        self.params.max_shrunk_state = {}
        for s in self.params.freqs.alphabet:
            f = self.params.freqs.frequency(s)
            self.params.min_shrunk_state[s] = self.params.RANGE_FACTOR * f
            self.params.max_shrunk_state[s] = self.params.RANGE_FACTOR * f * (1 << self.params.NUM_BITS_OUT) - 1

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
        # update the markov params
        self.true_freqs.freq_dict[s] += 1
        self.prev_symbol = s[-1]
        self.update_internals()
        return s[-1], prev_state

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
        self.update_internals()
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
            # markov_rANS decoder decodes from beginning to end, so add newly decoded symbol to end of list
            decoded_data_list.append(s)
            num_bits_consumed += num_symbol_bits

        # Finally, as a sanity check, ensure that the end state should be equal to the initial state
        assert state == self.params.INITIAL_STATE
        end_time = time.time() - t
        print(f"End Decoding: {end_time}")
        DECODING_TIMES.append(end_time)
        return DataBlock(decoded_data_list), num_bits_consumed


######################################## TESTS ##########################################

def test_markov_rANS_coding():
    global ENCODING_TIMES
    global DECODING_TIMES
    data_sizes = [5000 * i for i in range(1, 11)]
    SEED = 0
    # set the seed for the tests
    seed(274)
    # stats for graphs
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

            # measure empirical entropy
            all_2_tuples = []
            # skip last symbol
            for i in range(len(true_data_list) - 1):
                all_2_tuples.append(true_data_list[i]+true_data_list[i+1])

            all_counts = {key: 0 for key in dct.keys()}
            for two_tuple in all_2_tuples:
                all_counts[two_tuple] += 1
            sum_counts = sum(all_counts.values())

            # formula for empirical entropy from class
            empirical_entropy = 0
            for key in all_counts:
                prob = all_counts[key] / sum_counts
                empirical_entropy += prob * np.log2(1/prob)

            # start with no knowledge of the frequencies (uniform distribution)
            uniform_freqs = {}
            for symbol_1 in alphabet:
                for symbol_2 in alphabet:
                    uniform_freqs[symbol_1 + symbol_2] = 1
            uniform_freqs = Frequencies(uniform_freqs)
            encoder = rANSEncoder(rANSParams(uniform_freqs))
            decoder = rANSDecoder(rANSParams(uniform_freqs))

            # test lossless coding
            is_lossless, encode_len, _ = try_lossless_compression(
                data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
            )
            assert is_lossless
            # avg codelen ignoring the bits used to signal num data elements
            avg_codelen = encode_len / data_block.size
            print(f"markov_rANS coding: empirical entropy={empirical_entropy:.3f}, markov_rANS codelen: {avg_codelen:.3f}")
            # stats for graphs
            empirical_entropies[DATA_SIZE].append(empirical_entropy)
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

    with open("data_markov.txt", "w") as f:
        f.write(f"{encoding_averages}\n{decoding_averages}\n{empirical_entropies_averages}\n{codelens_averages}\n")
    return encoding_averages, decoding_averages, empirical_entropies_averages, codelens_averages


# test_markov_rANS_coding()
