"""Prompts for the router trace run: varied, synthetic, no private data.  Categories mirror how the model is used (code, agent-style tool calls,
reasoning, prose, other languages, summarising a given passage).  Each gets ~450 tokens of its own continuation (thinking on, as served)."""
CODE = [
    "Write a Python function that parses an ISO-8601 date string without using datetime.fromisoformat, with tests.",
    "Implement an LRU cache class in Python with O(1) get and put, and explain the data structures.",
    "Write a C function that reverses a singly linked list in place, and a main that tests it.",
    "Explain what this shell pipeline does and fix any bugs: find . -name '*.log' | xargs grep -c ERROR | sort -n | tail",
    "Write a Rust function that returns the n-th Fibonacci number using iteration and handle overflow with checked arithmetic.",
    "Refactor this into smaller functions: def process(data): out=[]; for d in data: if d['ok']: v=d['x']*2; if v>10: out.append(v); return out",
    "Write a SQL query that returns the top 3 customers by total order value per month, using window functions.",
    "Write a TypeScript debounce function with correct typing and a short usage example.",
    "Find the bug: def avg(xs): return sum(xs)/len(xs) when xs can be empty. Propose two different fixes.",
    "Write a bash script that backs up a directory to a timestamped tar.gz and keeps only the 5 newest backups.",
    "Implement binary search in Go and explain why mid = lo + (hi-lo)/2 is used.",
    "Write a regular-expression-free tokenizer for arithmetic expressions in Python that yields numbers and operators.",
]
AGENT = [
    "You are a coding agent. The user says: 'the tests in tests/test_parser.py fail with KeyError: version'. Describe step by step which tools you would call (read_file, run_tests, edit_file) and what each call's arguments would be, as a JSON list.",
    "Produce the JSON arguments for a tool call named edit_file that replaces the function name parse_cfg with parse_config in src/loader.py, then explain the risk of a blind rename.",
    "Plan how to add a --verbose flag to a Python CLI built with argparse: list the files to read first, the edit, and the test to add.",
    "A build fails with 'undefined reference to foo::bar()'. List the five most likely causes in order and the command to check each.",
    "Write a commit message and a short PR description for a change that makes an HTTP client retry on 503 with exponential backoff.",
    "Given a stack trace ending in 'RecursionError: maximum recursion depth exceeded' in a function that walks a tree, explain how to find the cause and two ways to fix it.",
]
REASON = [
    "A train leaves at 9:15 and travels 180 km at 72 km/h, then waits 20 minutes and travels 90 km at 90 km/h. When does it arrive? Show the steps.",
    "Three friends split a bill of 2,340 rupees in the ratio 2:3:5. How much does each pay? Then recompute if the largest payer gets a 10% discount on their share.",
    "Is 1,001 prime? Prove your answer, then find the prime factorisation of 123,456.",
    "You have 8 identical-looking coins, one is heavier. What is the minimum number of weighings on a balance to find it, and how?",
    "Explain why the sum of the first n odd numbers is n squared, with a visual argument and an induction proof.",
    "A box contains 5 red and 7 blue balls. Two are drawn without replacement. What is the probability that they are the same colour?",
    "Estimate how many piano tuners work in a city of one million people, showing the assumptions you make.",
    "Compare quicksort and mergesort for sorting a linked list of one million items. Which would you choose and why?",
]
PROSE = [
    "Explain how a refrigerator works to a ten-year-old, in about 250 words.",
    "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
    "Summarise the causes and consequences of the 2008 financial crisis in four paragraphs.",
    "Give practical advice for someone starting to learn to play the guitar as an adult, as a numbered list with reasons.",
    "Describe the water cycle and then explain how climate change may alter it.",
    "Write a polite email asking a landlord to fix a leaking tap that has been dripping for two weeks.",
    "Explain the difference between weather and climate, with two everyday examples of each.",
    "Describe how vaccines train the immune system, and why boosters are sometimes needed.",
]
OTHER = [
    "भारत की तीन सबसे बड़ी नदियों के नाम बताइए और हर एक के बारे में दो वाक्य लिखिए।",
    "请用中文解释一下什么是哈希表，并给出一个简单的例子。",
    "Explique en français comment fonctionne un moteur à combustion interne, en un paragraphe.",
    "Escribe un párrafo sobre la importancia de dormir bien, en español.",
    "日本語で、電車の中で気をつけるべきマナーを三つ挙げて説明してください。",
    "Übersetze ins Deutsche und erkläre die Grammatik: 'The quick brown fox jumps over the lazy dog.'",
]
PASSAGE = (
    "Strata streams the experts of a large mixture-of-experts model from an NVMe drive. A small cache on the graphics card keeps the most used experts, "
    "the operating system keeps another part in RAM, and the rest waits on disk. For each token a router chooses ten of five hundred and twelve experts in "
    "every layer. When a chosen expert is not on the card, the best two are computed on the processor and the others may be replaced by a similar expert that is "
    "already resident. A small draft model guesses the next words and the big model checks them in one pass."
)
SUMMARISE = [
    "Summarise this in two sentences, then list three questions a reader might ask:\n\n" + PASSAGE,
    "Rewrite this for a non-technical reader in under 120 words:\n\n" + PASSAGE,
    "Extract the key terms from this passage and define each in one line:\n\n" + PASSAGE,
    "Find any unclear claims in this passage and suggest how to make them precise:\n\n" + PASSAGE,
]
ALL = CODE + AGENT + REASON + PROSE + OTHER + SUMMARISE
