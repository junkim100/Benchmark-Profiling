import json
import random
import re
from pathlib import Path

random.seed(0)

# ────────────────────────────────────────────────────────────────────
#  Generic helpers
# ────────────────────────────────────────────────────────────────────
NUM = re.compile(r"\b\d+(\.\d+)?\b")
CAP = re.compile(r"\b[A-Z][a-zA-Z\-]+\b")
WORD = re.compile(r"\w+")


def rev_sentences(txt: str) -> str:
    """Reverses the order of sentences in a text."""
    parts = re.split(r"(?<=[.!?])\s+", txt.strip())
    # Handle case where text might not end with punctuation
    if not parts[-1]:
        parts.pop()
    return " ".join(parts[::-1])


def rev_words(txt: str) -> str:
    """Reverses the order of words in a text."""
    return " ".join(txt.split()[::-1])


def scramble_letters(txt: str) -> str:
    """Reverses the letters within each word."""
    return WORD.sub(lambda m: m.group(0)[::-1], txt)


def mask_content(txt: str) -> str:
    """Masks every capitalised token and every numeral with [ENTx]."""
    out, idx = [], 1
    for w in txt.split():
        # Check if word starts with a capital letter or is a number
        if (w and w[0].isupper() and w.isalpha()) or NUM.match(w):
            out.append(f"[ENT{idx}]")
            idx += 1
        else:
            out.append(w)
    return " ".join(out)


def mutilate_options(opts, aggressive=True):
    """Scrambles each option deterministically so it loses semantics."""
    if aggressive:
        # Ensure options are strings before scrambling
        return [scramble_letters(str(o)) for o in opts]
    return opts


# ────────────────────────────────────────────────────────────────────
#  Transformation functions  (context, question, options) ➜ 3‑tuple
# ────────────────────────────────────────────────────────────────────
LOGIC = re.compile(r"\b(all|any|every|if|then|because|therefore|hence|thus)\b", re.I)
MATH = re.compile(
    r"\b(add|sum|total|discount|multiply|divide|minus|plus|more than|less than|area|perimeter|average)\b",
    re.I,
)
TEMP = re.compile(
    r"\b(before|after|earlier|later|first|last|since|until|during|while|when|next|previous)\b",
    re.I,
)
SPAT = re.compile(
    r"\b(left|right|above|below|north|south|east|west|behind|in front of|beside|near|far|inside|outside)\b",
    re.I,
)


# Deductive reasoning (logic_puzzle) - Modified for 4-choice options
def tr_deductive(ctx, q, opts):
    """Corrupts deductive reasoning questions."""
    ctx = rev_sentences(LOGIC.sub("", ctx))
    q = scramble_letters(rev_words(q))
    # Always mutilate options for logic puzzles which likely have 4 choices
    opts = mutilate_options(opts)
    return ctx, q, opts


# Commonsense cause/effect (everyday_cause_effect)
def tr_cause(ctx, q, opts):
    """Corrupts cause/effect questions."""
    # Attempt to remove causal links and add noise
    ctx = re.sub(r"\b(because|so|due to|as a result)\b.*?[.!?]", "", ctx, flags=re.I)
    ctx_parts = ctx.split(".")
    if ctx_parts:
        # Keep first part and add confusing sentence
        ctx = ctx_parts[0].strip() + ". Afterwards, a bird flew by."
    else:  # If no sentence structure, reverse words
        ctx = rev_words(ctx)
    q = scramble_letters(q)  # Scramble question for good measure
    return ctx, q, mutilate_options(opts)


# Contextual recall (single_fact_recall, multi_hop_recall) - Very aggressive
def tr_contextual(ctx, q, opts):
    """Aggressively corrupts contextual recall by replacing text."""
    f = lambda s: re.sub(r"[A-Za-z0-9]", "X", s)  # Replace all alphanumeric with X
    # Keep original options as replacing them might break index-based eval
    return f(ctx), f(q), opts


# Knowledge trivia (world_fact)
def tr_knowledge(ctx, q, opts):
    """Corrupts knowledge questions by masking and scrambling."""
    # Mask potential entities in context, scramble question
    return mask_content(ctx), scramble_letters(q), mutilate_options(opts)


# Semantic roles (roles_and_relations)
def tr_semantic(ctx, q, opts):
    """Corrupts semantic role questions by reversing context words."""
    # Reverse word order in context, keep question structure but mutilate options
    return rev_words(ctx), q, mutilate_options(opts)


# Analogy (analogy)
def tr_analogy(ctx, q, opts):
    """Corrupts analogy questions by scrambling the question."""
    # Keep context structure (A:B::C:?), scramble question text
    return ctx, scramble_letters(q), mutilate_options(opts)


# Pattern completion (pattern_completion)
def tr_pattern(ctx, q, opts):
    """Corrupts pattern completion by reversing context sentences."""
    # Reverse sentence order in context (if applicable), mutilate options
    return rev_sentences(ctx), q, mutilate_options(opts)


# Quantitative (arithmetic_word_problem) - Removes keywords
def tr_quant(ctx, q, opts):
    """Corrupts quantitative reasoning by removing numbers and math terms."""
    g = lambda s: MATH.sub("", NUM.sub("<NUM>", s))  # Remove math terms and numbers
    # Keep original options as numerical options might be needed for eval format
    return g(ctx), g(q), opts


# Temporal (temporal_order)
def tr_temporal(ctx, q, opts):
    """Corrupts temporal reasoning by reversing sentences and swapping terms."""
    swap = {
        "before": "after",
        "after": "before",
        "earlier": "later",
        "later": "earlier",
        "first": "last",
        "last": "first",
        "start": "end",
        "end": "start",
        "began": "finished",
        "finished": "began",
        "am": "pm",
        "pm": "am",  # Basic time swap
    }
    # Use regex for case-insensitive swapping
    swap_pattern = re.compile(r"\b(" + "|".join(swap.keys()) + r")\b", re.I)

    ctx2 = rev_sentences(
        swap_pattern.sub(lambda m: swap.get(m.group(1).lower(), m.group(1)), ctx)
    )
    q2 = swap_pattern.sub(lambda m: swap.get(m.group(1).lower(), m.group(1)), q)
    return ctx2, q2, mutilate_options(opts)


# Spatial (spatial_relation)
def tr_spatial(ctx, q, opts):
    """Corrupts spatial reasoning by reversing sentences and swapping terms."""
    swap = {
        "left": "right",
        "right": "left",
        "above": "below",
        "below": "above",
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
        "behind": "in front of",
        "in front of": "behind",
        "inside": "outside",
        "outside": "inside",
        "top": "bottom",
        "bottom": "top",
    }
    # Use regex for case-insensitive swapping
    swap_pattern = re.compile(r"\b(" + "|".join(swap.keys()) + r")\b", re.I)

    ctx2 = rev_sentences(
        swap_pattern.sub(lambda m: swap.get(m.group(1).lower(), m.group(1)), ctx)
    )
    # Don't modify question text as much, rely on context corruption and option mutilation
    # q2 = swap_pattern.sub(lambda m: swap.get(m.group(1).lower(), m.group(1)), q)
    return ctx2, q, mutilate_options(opts)


# Compositionality (not present in new list, kept for generic fallback possibility)
def tr_compo(ctx, q, opts):
    """Corrupts compositional questions by reversing strings."""
    # Keep original options
    return ctx[::-1], q[::-1], opts


# Generic fallback for unmapped or new question types
def generic(ctx, q, opts):
    """Generic corruption: reverse words and mutilate options."""
    print(f"Warning: Using generic transform for question.")
    return rev_words(ctx), rev_words(q), mutilate_options(opts)


# ────────────────────────────────────────────────────────────────────
#  Registry mapping NEW question_type names to transformation functions
# ────────────────────────────────────────────────────────────────────
MAP = {
    # Contextual_Recall
    "single_fact_recall": tr_contextual,
    "multi_hop_recall": tr_contextual,
    # Long_Term_Knowledge
    "world_fact": tr_knowledge,
    # Semantic_Relationship
    "roles_and_relations": tr_semantic,
    # Deductive_Reasoning
    "logic_puzzle": tr_deductive,
    # Inductive_Reasoning
    "pattern_completion": tr_pattern,
    # Analogical_Reasoning
    "analogy": tr_analogy,
    # Quantitative_Reasoning
    "arithmetic_word_problem": tr_quant,
    # Temporal_Reasoning
    "temporal_order": tr_temporal,
    # Spatial_Reasoning
    "spatial_relation": tr_spatial,
    # Commonsense_Causal
    "everyday_cause_effect": tr_cause,
}

# ────────────────────────────────────────────────────────────────────
#  File processing
# ────────────────────────────────────────────────────────────────────
SRC = Path("data_preprocess/datasets")  # Source directory of original datasets
DST = Path(
    "data_preprocess/transform"
)  # Destination directory for transformed datasets
DST.mkdir(parents=True, exist_ok=True)
unseen = set()
processed_files = 0
processed_lines = 0

print(f"Starting transformation from {SRC} to {DST}")

# Iterate through ability directories (e.g., Contextual_Recall, Long_Term_Knowledge)
for ability_dir in SRC.iterdir():
    if not ability_dir.is_dir():
        continue

    original_subdir = ability_dir / "original"
    if not original_subdir.exists():
        print(f"Skipping {ability_dir.name}: 'original' subdirectory not found.")
        continue

    # Define output directory structure
    out_base_dir = DST / ability_dir.name
    out_dir = (
        out_base_dir / "original"
    )  # Maintain the 'original' subfolder name in transform dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process splits (train, dev, test)
    for split_filename in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        inp_path = original_subdir / split_filename
        if not inp_path.exists():
            # print(f"Info: File not found, skipping: {inp_path}")
            continue

        outp_path = out_dir / split_filename
        print(f"Processing: {inp_path} -> {outp_path}")

        try:
            with inp_path.open("r", encoding="utf-8") as fi, outp_path.open(
                "w", encoding="utf-8"
            ) as fo:

                file_lines = 0
                for line_num, line in enumerate(fi, 1):
                    try:
                        ex = json.loads(line)
                        qt = ex.get("question_type")

                        if not qt:
                            print(
                                f"Warning: Missing 'question_type' in line {line_num} of {inp_path}. Skipping line."
                            )
                            continue

                        # Get context, question, options safely
                        ctx = ex.get("context", "")
                        q = ex.get("question", "")
                        opts = ex.get("options", [])

                        # Find the appropriate transformation function
                        transform_func = MAP.get(qt)

                        if transform_func:
                            ctx_t, q_t, opts_t = transform_func(ctx, q, opts)
                        else:
                            # Use generic fallback if question_type is not in MAP
                            if qt not in unseen:
                                print(
                                    f"Warning: No specific transform found for question_type '{qt}'. Using generic fallback."
                                )
                                unseen.add(qt)
                            ctx_t, q_t, opts_t = generic(ctx, q, opts)

                        # Update the example dictionary
                        ex["context"] = ctx_t
                        ex["question"] = q_t
                        ex["options"] = opts_t  # Options might be mutilated or original

                        # Write the transformed example to the output file
                        fo.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        file_lines += 1

                    except json.JSONDecodeError:
                        print(
                            f"Error: Invalid JSON on line {line_num} in {inp_path}. Skipping line."
                        )
                    except Exception as e:
                        print(f"Error processing line {line_num} in {inp_path}: {e}")

                processed_lines += file_lines
                processed_files += 1
                print(f"Finished processing {file_lines} lines from {inp_path.name}")

        except IOError as e:
            print(f"Error opening or writing file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred processing {inp_path}: {e}")


print(
    f"\n✓ Aggressive transforms complete. Processed {processed_files} files and {processed_lines} lines."
)
if unseen:
    print(
        "Generic transform used for the following unseen question types:",
        sorted(list(unseen)),
    )
else:
    print("All found question types had specific transforms mapped.")
