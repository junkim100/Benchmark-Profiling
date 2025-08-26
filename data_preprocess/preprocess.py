"""
This script provides tools for preprocessing text data for machine learning tasks, particularly for tokenization,
sentence splitting, and dataset preparation. It supports multi-task inputs (e.g., English, code, math)
and handles special requirements like merging datasets or working with large-scale tokenized data.

Key Features:
-------------
1. **Tokenization**:
    - Supports tokenization using Hugging Face's `AutoTokenizer` for various models (e.g., LLaMA, Qwen).
    - Handles special tokens like Beginning-of-Sequence (BOS) and End-of-Sequence (EOS).
    - Provides flexibility for task-specific tokenization rules.

2. **Sentence Splitting**:
    - For English: Uses NLTK's Punkt tokenizer with optional customization (`FlexiblePunktLanguageVars`) to handle edge cases.

3. **Dataset Preparation**:
    - Converts tokenized data into binary format (`.bin`), along with index files (`.idx`) and distribution metadata (`.dis`).
    - Stores start positions and lengths of each sample in `.idx` files using `np.uint64` and `np.uint32`.

4. **Merging Datasets**:
    - Merges multiple datasets of the same or different types into a single dataset.
    - Supports `.dis` files to track the total number of samples for each dataset type.

5. **Statistics Collection**:
    - Tracks how sequences are structured after tokenization:
        - Starts with BOS.
        - Ends with EOS.
        - Neither starts nor ends with special tokens.
        - Complete sentences (both BOS and EOS).

6. **Parallel Processing**:
    - Uses Python's `multiprocessing` to parallelize encoding tasks across multiple workers.

File Structure:
---------------
1. **Tokenizer Class**:
    Provides a wrapper around Hugging Face's tokenizer to handle encoding/decoding and manage special tokens.

2. **DistributedTokenizer Class**:
    Handles distributed processing of text data, including sentence splitting, tokenization, and chunking long sequences.

3. **TokenizedDataset Class**:
    A PyTorch-compatible dataset class for loading preprocessed binary data (`.bin`, `.idx`, `.dis`) efficiently.

4. **Utility Functions**:
    - `count_lines`: Counts the number of lines in a file.
    - `collate_fn_from_json`: Extracts specific fields from JSON-formatted input lines.
    - `write`: Processes raw text into tokenized binary format.
    - `read`: Reads preprocessed binary data and decodes it back into text.
    - `merge`: Merges multiple datasets into a single dataset.

Data Storage Format:
--------------------
- **Tokenized Data (`.bin`)**:
  Stores tokenized sequences as binary data in `np.uint32` format (4 bytes per token ID).
- **Index File (`.idx`)**:
  Records the starting position (`np.uint64`, 8 bytes) and length (`np.uint32`, 4 bytes) of each sequence.
- **Distribution File (`.dis`)**:
  Tracks the number of samples per dataset type using PyTorch's `torch.save`.

Example Usage:
--------------
1. Preprocess a text file into binary format:
    python preprocess.py write --file_path=input.jsonl --key=text --save_prefix=output
    --save_path=./datasets --task=english --do_keep_newlines=True
    --do_split_sentences=True --seq_length=1024 --tokenizer_path=./tokenizer
    --num_per_doc=-1 --num_workers=4

2. Decode and print sample sentences from preprocessed data:
    python preprocess.py read --read_path_prefix=./datasets/output
    --seq_length=1024 --tokenizer_path=./tokenizer

3. Merge multiple datasets into one:
    python preprocess.py merge --merge_path_prefix="['dataset1', 'dataset2']"
    --merge_path_type="[0, 1]" --new_path_prefix=merged_dataset


Notes:
------
- The script assumes that each line in the input file is valid and appropriately formatted for tokenization.
- Special handling is included for long sequences that exceed the maximum length (`seq_length`):
 1. For English tasks: Split tokenized results directly into smaller chunks.
"""

import os
import sys
import numpy as np
import json
import shutil
import re
from typing import List, Tuple
from functools import partial
import multiprocessing
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import nltk
from tqdm import tqdm
import fire


class Tokenizer:
    def __init__(self, model_path: str):
        self.sp_model = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )

        # Handle model-specific tokenizer configurations
        self.n_words = self.sp_model.vocab_size

        # Safely get special tokens
        self.eos_id = getattr(self.sp_model, "eos_token_id", None)
        self.pad_id = getattr(self.sp_model, "pad_token_id", None)
        self.bos_id = getattr(self.sp_model, "bos_token_id", None)

        # Validate essential tokens exist
        if self.eos_id is None:
            print(f"Warning: Model {model_path} does not define an EOS token")

    def encode(self, s: str, bos: bool = False, eos: bool = True) -> List[int]:
        assert isinstance(s, str), f"Expected string input, got {type(s)}"

        # Handle different tokenization methods
        if hasattr(self.sp_model, "encode"):
            t = self.sp_model.encode(s)
        else:
            t = self.sp_model.convert_tokens_to_ids(self.sp_model.tokenize(s))

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t, skip_special_tokens=True)


class DistributedTokenizer:
    """
    A tokenizer class for distributed processing of text data. Supports tokenization, sentence splitting,
    and handling of multi-task inputs.

    Args:
        tokenizer_path (str): Path to the tokenizer model or configuration.
        task (str): Task of the input text.
        do_split_sentences (bool): Whether to split input text into sentences.
        do_keep_newlines (bool): Whether to retain newlines when splitting sentences.
        seq_length (int): Maximum sequence length for tokenized segments.
        eos (bool): Whether to add an End-of-Sequence (EOS) token during tokenization.
        bos (bool): Whether to add a Beginning-of-Sequence (BOS) token during tokenization.
        collate_fn (callable, optional): Function to preprocess input JSON lines into text format.

    Notes:
        - Sentence splitting uses NLTK for all tasks in English.
        - Tokenization is handled by the `Tokenizer` class.
    """

    def __init__(
        self,
        tokenizer_path,
        task,
        do_split_sentences,
        do_keep_newlines,
        seq_length,
        eos: bool,
        bos: bool,
        collate_fn=None,
    ):
        self.tokenizer_path = tokenizer_path
        self.task = task.lower()
        self.do_split_sentences = do_split_sentences
        self.do_keep_newlines = do_keep_newlines
        self.seq_length = seq_length
        self.eos = eos
        self.bos = bos
        self.collate_fn = collate_fn

    def _re_split(
        self, src: str, tokenized: List[int], start_part=False, end_part=False
    ) -> List[List[int]]:
        """
        Splits a tokenized sequence into smaller chunks if it exceeds the maximum sequence length.

        Args:
            src (str): Original sentence or text segment.
            tokenized (List[int]): Tokenized representation of the input text.
            start_part (bool): Whether this is the starting part of a sequence (adds BOS if True).
            end_part (bool): Whether this is the ending part of a sequence (adds EOS if True).

        Returns:
            List[List[int]]: List of smaller tokenized chunks.

        Raises:
            AssertionError: If an unsupported task is provided.

        Notes:
            - For all tasks English: Evenly splits the tokenized list into chunks.
              Handles special cases like BOS/EOS tokens specific to certain models.
              Example: LLaMA uses BOS/EOS tokens differently from Qwen models.
        """

        if len(tokenized) <= self.seq_length:
            return [tokenized]

        n_block = int(np.ceil(len(tokenized) / self.seq_length))

        # Evenly split the tokenized list into chunks
        return [
            tokenized[i * self.seq_length : (i + 1) * self.seq_length]
            for i in range(n_block)
        ]

    def _split(self, lst: List[int]) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Splits a list of token lengths into chunks that fit within the maximum sequence length.

        Args:
            lst (List[int]): List of token lengths.

        Returns:
            Tuple[List[int], List[Tuple[int, int]]]:
                - Merged list of token lengths for each chunk.
                - List of start and end indices for each chunk (left-closed, right-open).
        """
        maxlen = self.seq_length
        merged_lst = []
        answer_lst = []

        i = 0
        while i < len(lst):
            ans = [i, 0]
            sums = lst[i]
            j = i + 1
            while j < len(lst) and sums + lst[j] <= maxlen:
                sums += lst[j]
                j += 1

            ans[1] = j  # Record the endpoint (exclusive)
            merged_lst.append(sums)
            answer_lst.append(tuple(ans))
            i = j

        # If the merged list exceeds limits or a single chunk exceeds max length, reset
        if len(merged_lst) >= 2 or any(s > maxlen for s in merged_lst):
            merged_lst.clear()
            answer_lst.clear()

        return merged_lst, answer_lst

    def dsmt_encode(self, json_line, key):
        """
        Encodes a JSON line into tokenized format, handling cases where the input text
        contains multiple sentences or no valid content.
        """
        # Extract text based on collate function
        if self.collate_fn is None:
            text = json_line
        else:
            text = self.collate_fn(json_line, key)

        # Skip empty or invalid lines
        if text == "\n" or text.strip() == "" or text == r"\n":
            return []

        # Split input into sentences
        sentences = DistributedTokenizer.splitter.tokenize(text)

        # Handle cases with no sentences
        if len(sentences) == 0:
            return []

        # Tokenize and process each sentence
        _tokenized = []
        for idx, sentence in enumerate(sentences):
            cur_tokenized = DistributedTokenizer.tokenizer.encode(
                sentence,
                bos=(idx == 0 and self.bos),
                eos=(idx == len(sentences) - 1 and self.eos),
            )
            _tokenized.extend(
                self._re_split(
                    src=sentence,
                    tokenized=cur_tokenized,
                    start_part=(idx == 0 and self.bos),
                    end_part=(idx == len(sentences) - 1 and self.eos),
                )
            )

        # Record the number of tokens in each sentence after splitting
        length_tokenized = [len(_) for _ in _tokenized]

        # Get merged indices for tokenized chunks
        _, index = self._split(length_tokenized)
        ultra = []
        for pair in index:
            cur = []
            start, end = pair
            for i in range(start, end):
                cur.extend(_tokenized[i])
            ultra.append(cur)

        return ultra

    def dsmt_initializer(self):
        """
        Initializes the tokenizer and sentence splitter based on the specified task.

        Notes:
            - For all tasks in English: Uses NLTK PunktSentenceTokenizer.

        Raises:
            AssertionError: If an unsupported task is provided.
        """
        DistributedTokenizer.tokenizer = Tokenizer(self.tokenizer_path)

        class FlexiblePunktLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):
            _period_context_fmt = r"""
                \S*                          # some word material
                %(SentEndChars)s             # a potential sentence ending
                \s*                       #  <-- THIS is what I changed
                (?=(?P<after_tok>
                    %(NonWord)s              # either other punctuation
                    |
                    (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
                ))"""

        splitter = nltk.load("tokenizers/punkt/english.pickle")
        if self.do_keep_newlines:
            DistributedTokenizer.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                train_text=splitter._params,
                lang_vars=FlexiblePunktLanguageVars(),
            )
        else:
            DistributedTokenizer.splitter = splitter


class TokenizedDataset(Dataset):
    def __init__(self, data_prefix, seq_length, pad_id):
        super(TokenizedDataset, self).__init__()
        """Here data_prefix requires the complete path, but without the suffix"""
        """For example: /llama/our/data"""
        """The suffix will be automatically added later as needed: /llama/our/data.idx"""
        """The suffix will be automatically added later as needed: /llama/our/data.bin"""
        """The suffix will be automatically added later as needed: /llama/our/data.dis"""
        self.idx_file_path = f"{data_prefix}.idx"
        self.bin_file_path = f"{data_prefix}.bin"
        self.dis_file_path = f"{data_prefix}.dis"
        self.seq_length = seq_length
        self.pad_id = pad_id

        self.index_start_pos = None  # Starting position of each sample
        self.index_length = None  # Length of each sample
        self._load_index()
        self._load_bin()
        self._load_dis()

    def _load_index(self):
        file_size = os.stat(self.idx_file_path).st_size
        # Each entry is 12 bytes (8 for uint64 + 4 for uint32)
        assert file_size % 12 == 0
        self.total_sample = file_size // 12
        with open(self.idx_file_path, "rb") as f:
            self.index_start_pos = np.frombuffer(
                f.read(self.total_sample * 8), dtype=np.uint64
            ).tolist()
            self.index_length = np.frombuffer(
                f.read(self.total_sample * 4), dtype=np.uint32
            ).tolist()

    def _load_bin(self):
        """Load large files using memory mapping"""
        self.bin_buffer = np.memmap(self.bin_file_path, dtype=np.uint32, mode="r")

    def _load_dis(self):
        """Only valid when mixing multiple types of data"""
        self.distributed = torch.load(self.dis_file_path, weights_only=False)
        if len(self.distributed) != 0:
            assert sum(self.distributed) == self.total_sample

    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        start_idx = self.index_start_pos[idx]
        length = self.index_length[idx]
        if idx + 1 < self.total_sample:
            assert (
                start_idx + length == self.index_start_pos[idx + 1]
            ), f"{start_idx + length}!={self.index_start_pos[idx + 1]}, idx={idx}"
        if length > self.seq_length:
            length = self.seq_length
        return self.bin_buffer[start_idx : start_idx + length].tolist()


def count_lines(path):
    """Count the number of lines in the input file"""
    print(path)
    with open(path, "rb") as f:
        count = 0
        last_data = "\n"
        while True:
            data = f.read(1024 * 1024 * 1024)
            if not data:
                break
            count += data.count(b"\n")
            last_data = data
        if last_data[-1:] != b"\n":
            count += 1  # Remove this if a wc-like count is needed
    return count


def collate_fn_from_json(json_line: str, key):
    """
    Extracts text from a JSON line based on the provided key(s). If the key is a comma-separated string,
    this function splits it into multiple keys and concatenates their corresponding values.

    Args:
        json_line (str): A single line of JSON-formatted text.
        key (str): A key or comma-separated list of keys to extract text from the JSON object.

    Returns:
        str: Concatenated text from the specified key(s).
    """
    data = json.loads(json_line)

    if isinstance(key, tuple):
        missing_keys = [k for k in key if k not in data]
        if missing_keys:
            raise ValueError(f"Keys not found: {', '.join(missing_keys)}")
        for k in key:
            if isinstance(data[k], list):
                data[k] = " ".join(data[k])
        total_text = " ".join(data[k] for k in key)
    elif isinstance(key, str):
        if key not in data:
            raise ValueError(f"Key not found: {key}")
        if isinstance(data[key], list):
            data[key] = " ".join(data[key])
        total_text = data[key]
    else:
        raise ValueError(f"Invalid key type: {type(key)}")

    return total_text


def collate_fn_from_template(message: str, key):
    """
    Get chat template message as text and remove the system prompt.
    """
    # Cut the first part of the message from <|begin_of_text|><|start_header_id|>system<|end_header_id|> to the first appearing <|eot_id|> if they exist
    if (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" in message
        and "<|eot_id|>" in message
    ):
        parts = message.split("<|eot_id|>")
        # Keep everything except the first part (system message)
        message = "<|eot_id|>".join(parts[1:])
    return message


def process_line_with_template(line, tokenizer_path=None, apply_template=False):
    """Process a line by applying chat template if needed before encoding."""
    if apply_template and tokenizer_path:
        try:
            # Import here to avoid circular imports
            from transformers import AutoTokenizer

            # Load tokenizer if needed (in actual implementation, you'd want to cache this)
            template_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            # Parse JSON line
            import json

            data = json.loads(line)

            # Extract fields (adjust based on your JSON structure)
            context = data.get("context", "")
            question = data.get("question", "")
            answer = data.get("answer", "")

            # Format as chat messages
            if context == "":
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Question: {question}",
                    },
                    {"role": "assistant", "content": f"Answer: {answer}."},
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Context: {context}\nQuestion: {question}",
                    },
                    {"role": "assistant", "content": f"Answer: {answer}."},
                ]

            # Apply the chat template
            if "apply_chat_template" in dir(template_tokenizer):
                formatted_text = template_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                # Return the formatted JSON with the templated text
                return json.dumps({"content": formatted_text})

        except Exception as e:
            print(f"Warning: Could not apply template to line: {str(e)}")

    # Return original line if template not applied
    return line


def write(
    file_path,
    key,
    save_prefix,
    save_path,
    task,
    do_keep_newlines,
    do_split_sentences,
    seq_length,
    tokenizer_path,
    num_per_doc,
    num_workers,
    apply_chat_template=True,
):
    """
    Processes a text file by tokenizing its content, sampling segments, and saving the results in binary format.

    This function also collects statistics about the structure of tokenized sequences based on their starting
    and ending tokens. Specifically, it tracks whether sequences:
        - Start with a Beginning-of-Sequence (BOS) token (ID = 1).
        - End with an End-of-Sequence (EOS) token (ID = 2).
        - Neither start nor end with special tokens.
        - Both start with BOS and end with EOS (complete sentence).

    Args:
        file_path (str):
            Path to the input text file containing data to be processed. This is the source of raw text that will be tokenized.

        key (str):
            A key used for extracting specific fields from JSON-formatted lines in the input file.
            If the input is plain text, this argument may not be relevant.

        save_prefix (str):
            Prefix for naming the output files. The generated files will include:
                - <save_prefix>.bin: Tokenized data in binary format.
                - <save_prefix>.idx: Index file containing start positions and lengths of tokenized segments.
                - <save_prefix>.dis: Metadata about the number of samples.

        save_path (str):
            Directory where all output files will be saved. Ensure this directory exists or can be created.

        task (str):
            Task of the input text (e.g., "if" for Instruction Following). This is used by the tokenizer for task-specific processing rules.

        do_keep_newlines (bool):
            Whether to preserve newlines in the input text during processing. If `True`, newlines are retained; otherwise, they are removed.

        do_split_sentences (bool):
            Whether to split the input text into individual sentences before tokenization. Useful for sentence-level tokenization.

        seq_length (int):
            Maximum sequence length for tokenized segments. Longer sequences will be split into smaller chunks.

        tokenizer_path (str):
            Path to the tokenizer model or configuration directory. This is required to initialize and use the tokenizer, such as a Hugging Face tokenizer.

        num_per_doc (int):
            Number of samples to extract per document:
                - `-1`: Use all tokenized segments from each document.
                - `<= 2`: Randomly sample up to this number of segments per document.
                - `> 2`: Include specific segments (e.g., first and last) and sample additional ones if needed.

        num_workers (int):
            Number of worker processes for parallel processing using `multiprocessing.Pool`. Increasing this value can speed up processing but requires more system resources.

        apply_chat_template (bool):
            Whether to apply a chat template to the input text before tokenization. This is useful for formatting chat-like data for specific models.

    Returns:
        None: The function saves processed data into binary files and does not return any value.

    Output Files:
        - <save_prefix>.bin: Binary file containing tokenized data.
        - <save_prefix>.idx: Index file with start positions and lengths of tokenized segments.
        - <save_prefix>.dis: Distribution metadata about the number of samples processed.
        - <save_prefix>.tmp: Temporary statistics about sentence structure.

    Notes:
        - `target` represents a tokenized sequence, which is a list of token IDs generated by the tokenizer. Each ID corresponds to a word, subword, or character from the tokenizer's vocabulary.
        - Special tokens used in `target`:
            - **BOS** (`1`): Indicates the beginning of a sequence.
            - **EOS** (`2`): Indicates the end of a sequence.
        - The function collects statistics as follows:
            1. **Starts with BOS**: Sequences where `target[0] == 1` but do not end with EOS (`target[-1] != 2`).
            2. **Ends with EOS**: Sequences where `target[-1] == 2` but do not start with BOS (`target[0] != 1`).
            3. **Neither BOS nor EOS**: Sequences that neither start with BOS nor end with EOS.
            4. **Complete Sentence**: Sequences that both start with BOS (`target[0] == 1`) and end with EOS (`target[-1] == 2`).

        These statistics help analyze how sequences are structured after tokenization and ensure that special tokens are used correctly.
    """
    try:
        print(f"Start counting lines")
        count = count_lines(file_path)
        print(f"Number of lines: {count}")

        if apply_chat_template:
            from transformers import AutoTokenizer

            try:
                temp_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

                apply_chat_template = "apply_chat_template" in dir(temp_tokenizer)

                print(f"Chat template enabled: {apply_chat_template}")
            except Exception as e:
                print(f"Warning: Could not load tokenizer for chat template: {str(e)}")
                apply_chat_template = False

        # Open the text file and create a tokenizer
        with open(file_path, "r", encoding="utf-8") as fin:
            encoder = DistributedTokenizer(
                tokenizer_path,
                task,
                do_split_sentences,
                do_keep_newlines,
                seq_length,
                eos=True,
                bos=False,
                # collate_fn=collate_fn_from_json,
                collate_fn=collate_fn_from_template,
            )

            # Pre-process lines with template if needed
            if apply_chat_template:
                preprocessed_lines = []
                for line in tqdm(fin, total=count, desc="Applying templates"):
                    preprocessed_line = process_line_with_template(
                        line,
                        tokenizer_path=tokenizer_path,
                        apply_template=apply_chat_template,
                    )
                    preprocessed_lines.append(preprocessed_line)

                # Create a multiprocessing pool
                pool = multiprocessing.Pool(
                    num_workers, initializer=encoder.dsmt_initializer
                )

                dsmt_encode_with_key = partial(encoder.dsmt_encode, key=key)

                # Read and encode samples directly
                encoded_samples = list(
                    tqdm(
                        pool.imap(dsmt_encode_with_key, preprocessed_lines, 25),
                        total=count,
                        desc="Reading progress",
                    )
                )
            else:
                # Original approach without templates
                pool = multiprocessing.Pool(
                    num_workers, initializer=encoder.dsmt_initializer
                )

                # Wrap the dsmt_encode method with the additional argument 'key'
                dsmt_encode_with_key = partial(encoder.dsmt_encode, key=key)

                # Read and encode samples directly
                encoded_samples = list(
                    tqdm(
                        pool.imap(dsmt_encode_with_key, fin, 25),
                        total=count,
                        desc="Reading progress",
                    )
                )

        def get_sample_indices(num_per_doc, doc_length, generator):
            """Get indices for sampling from a document."""
            if num_per_doc <= 2:
                return torch.randint(
                    0, doc_length, [num_per_doc], generator=generator
                ).tolist()
            idx = [0, -1]
            if doc_length > 2:
                idx.extend(
                    (torch.randperm(doc_length - 2, generator=generator) + 1).tolist()
                )
            return idx[:num_per_doc]

        def update_statistics(statistic, target):
            """Update statistics based on the tokenized target."""
            if target[0] == 1 and target[-1] == 2:
                statistic[3] += 1  # Complete sentence
            elif target[0] == 1:
                statistic[0] += 1  # Starts with BOS but no EOS
            elif target[-1] == 2:
                statistic[1] += 1  # Ends with EOS but no BOS
            else:
                statistic[2] += 1  # Neither BOS nor EOS

        # Start writing output files
        with open(f"{save_path}/{save_prefix}.bin", "wb") as f_bin_out:
            encoded_samples = list(encoded_samples)
            pbar = tqdm(total=len(encoded_samples))
            start_pos = 0
            start = []
            length = []
            num_samples = 0
            g = torch.Generator()
            g.manual_seed(2023)
            statistic = [
                0,
                0,
                0,
                0,
            ]  # [starts with BOS, ends with EOS, neither, complete]

            for doc in encoded_samples:
                # Determine indices for sampling
                idx = (
                    list(range(len(doc)))
                    if num_per_doc == -1
                    else get_sample_indices(num_per_doc, len(doc), g)
                )

                for i in idx:
                    target = doc[i]
                    num_samples += 1
                    # Record statistics
                    update_statistics(statistic, target)
                    f_bin_out.write(
                        np.array(target, dtype=np.uint32).tobytes(order="C")
                    )
                    length.append(len(target))
                    start.append(start_pos)
                    start_pos += len(target)
                pbar.update(1)

        # Save index file
        with open(f"{save_path}/{save_prefix}.idx", "wb") as f_idx_out:
            f_idx_out.write(np.array(start, dtype=np.uint64).tobytes(order="C"))
            f_idx_out.write(np.array(length, dtype=np.uint32).tobytes(order="C"))

        # Save statistics
        torch.save([num_samples], f"{save_path}/{save_prefix}.dis")
        torch.save(statistic, f"{save_path}/{save_prefix}.tmp")

    except Exception as e:
        raise fire.core.FireError(f"Write operation failed: {str(e)}")


def read(read_path_prefix, seq_length, tokenizer_path):
    """
    Reads a dataset from binary files, decodes tokenized data, and prints sample sentences.

    Args:
        read_path_prefix (str):
            Prefix for the dataset files to be read. The function will look for files with extensions `.bin`, `.idx`, and `.dis`.

        seq_length (int):
            Maximum sequence length for padding or truncation when reading data.

        tokenizer_path (str):
            Path to the tokenizer model or configuration directory. Used to decode tokenized data back into text.

    Returns:
        None: This function prints sample sentences and dataset statistics to the console.
    """
    try:
        # Load the dataset
        ds = TokenizedDataset(read_path_prefix, seq_length=seq_length, pad_id=0)

        # Initialize the tokenizer
        tokenizer = Tokenizer(model_path=tokenizer_path)
        eos_token = tokenizer.eos_id  # End-of-sequence token ID
        print("EOS token:", eos_token)

        # Print dataset statistics
        print(f"Length: {len(ds)}")

        # Print the first 20 sentences in the dataset
        for i in range(len(ds)):
            if i == 20:
                break
            print(f"Sentence {i}: {tokenizer.decode(ds[i])}\n")

        # Print distribution metadata
        print(f"Distribution: {ds.distributed}")
    except Exception as e:
        raise fire.core.FireError(f"Read operation failed: {str(e)}")


def merge(merge_path_prefix, merge_path_type, new_path_prefix):
    """
    Merges multiple datasets into a single dataset. Supports merging datasets of the same type only.

    Args:
        merge_path_prefix (list[str]):
            List of prefixes for the datasets to be merged. Each prefix should correspond to `.bin`, `.idx`, and `.dis` files.

        merge_path_type (list[int] or None):
            Specifies the types of datasets being merged. If provided, it should be a list where each element corresponds to a dataset type (e.g., `[0, 1, 1]`).
            If `None`, assumes all datasets are of the same type.

        new_path_prefix (str):
            Prefix for the output merged dataset. The function will generate files with this prefix and extensions `.bin`, `.idx`, and `.dis`.

    Returns:
        None: This function writes merged binary data, index files, and distribution metadata to disk.

    Notes:
        - Mixing different types of data in one merged dataset is not supported.
        - Ensure that all input datasets are compatible in terms of structure and format.
        - The `.dis` file for each input dataset must contain only one entry.
    """

    try:
        if merge_path_prefix is None:
            raise ValueError("merge_path_prefix cannot be None.")

        # Convert string input to list if necessary
        merge_path_prefix = (
            eval(merge_path_prefix)
            if isinstance(merge_path_prefix, str)
            else merge_path_prefix
        )

        # Check if datasets are of the same type
        if merge_path_type is None:
            print("The datasets to be merged are assumed to be of the same type.")
        else:
            merge_path_type = (
                eval(merge_path_type)
                if isinstance(merge_path_type, str)
                else merge_path_type
            )

        def classify_by_type(merge_path_type, merge_path_prefix):
            """
            Classifies dataset prefixes by type.

            Args:
                merge_path_type (list[int]):
                    List specifying types of datasets.

                merge_path_prefix (list[str]):
                    List of prefixes corresponding to each dataset.

            Returns:
                dict: A dictionary mapping types to lists of prefixes.
            """
            classifier_prefix = {}

            for idx, types in enumerate(merge_path_type):
                if types not in classifier_prefix:
                    classifier_prefix[types] = [merge_path_prefix[idx]]
                else:
                    classifier_prefix[types].append(merge_path_prefix[idx])

            return classifier_prefix

        def merge_datasets(file_prefixes, new_file_bin, index_start_pos, index_length):
            """
            Merges binary and index files from multiple datasets into a single output.

            Args:
                file_prefixes (list[str]):
                    List of dataset prefixes to be merged.

                new_file_bin (file object):
                    File object for writing merged binary data.

                index_start_pos (list[int]):
                    List to store updated start positions for the index file.

                index_length (list[int]):
                    List to store lengths of samples for the index file.

            Returns:
                None
            """
            for file_prefix in file_prefixes:
                # Merge binary files
                with open(file_prefix + ".bin", "rb") as f:
                    shutil.copyfileobj(f, new_file_bin)

                # Merge index files
                file_size = os.stat(file_prefix + ".idx").st_size
                assert (
                    file_size % 12 == 0
                )  # Each entry is 12 bytes (8B start pos + 4B length)
                total_sample = file_size // 12

                with open(file_prefix + ".idx", "rb") as f:
                    _index_start_pos = np.frombuffer(
                        f.read(total_sample * 8), dtype=np.uint64
                    )
                    _index_length = np.frombuffer(
                        f.read(total_sample * 4), dtype=np.uint32
                    ).tolist()

                if len(index_start_pos) > 0:
                    offset = index_start_pos[-1] + index_length[-1]
                    index_start_pos.extend((_index_start_pos + offset).tolist())
                else:
                    index_start_pos.extend(_index_start_pos)

                index_length.extend(_index_length)

        # Initialize output files
        with open(new_path_prefix + ".bin", "wb") as new_file_bin:
            index_start_pos = []
            index_length = []

            if merge_path_type is not None:
                # Classify by type and merge
                classifier_prefix = classify_by_type(merge_path_type, merge_path_prefix)
                for _, file_prefixes in classifier_prefix.items():
                    merge_datasets(
                        file_prefixes, new_file_bin, index_start_pos, index_length
                    )

                # Merge distribution files by type
                new_dist = []
                for _, file_prefixes in classifier_prefix.items():
                    current_size = 0
                    for file_prefix in file_prefixes:
                        data = torch.load(file_prefix + ".dis", weights_only=False)
                        assert len(data) == 1
                        current_size += data[0]
                    new_dist.append(current_size)

                torch.save(new_dist, new_path_prefix + ".dis")

            else:
                # Merge all datasets without classification
                merge_datasets(
                    merge_path_prefix, new_file_bin, index_start_pos, index_length
                )

                # Merge distribution files without classification
                total_samples = sum(
                    torch.load(file + ".dis", weights_only=False)[0]
                    for file in merge_path_prefix
                )
                torch.save([total_samples], new_path_prefix + ".dis")

        # Write merged index file
        with open(new_path_prefix + ".idx", "wb") as new_file_idx:
            new_file_idx.write(
                np.array(index_start_pos, dtype=np.uint64).tobytes(order="C")
            )
            new_file_idx.write(
                np.array(index_length, dtype=np.uint32).tobytes(order="C")
            )

        print("Merge completed successfully.")
    except Exception as e:
        raise fire.core.FireError(f"Merge operation failed: {str(e)}")


if __name__ == "__main__":
    """There is no judgment on the length after tokenization, because it is assumed here that each line is a correct and appropriate"""
    """Whether to add EOS at the end of the text and BOS at the beginning of the text seems to be uncertain"""

    try:
        fire.Fire({"write": write, "read": read, "merge": merge})
    except fire.core.FireExit as e:
        sys.exit(1)
    except fire.core.FireError as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)
