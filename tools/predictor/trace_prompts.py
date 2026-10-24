"""Prompts for the router trace run: varied, synthetic, no private data.  Categories mirror how the model is used (code, agent-style tool calls,
reasoning, prose, other languages, summarising a given passage).  Each gets ~450 tokens of its own continuation (thinking on, as served).
PREFILL is a second set of long prompts (800-4000 tokens) for STRATA_ROUTER_TRACE_PREFILL: local source text and self-written texts, in conversations."""
import os

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

# ---- PREFILL set: long prompts (800-4000 tokens each) over distinct topics, for the prompt's own routing (STRATA_ROUTER_TRACE_PREFILL=1).
# Source text is read from the tree at run time (TRACE_ROOT, default the predictor-online worktree); the self-written texts are below.
# Each conversation is a list of (topic, text) turns, sent in order with the replies kept, so its follow-ups share the topic.  Conversation 3
# switches topic twice.  24 requests in all, in 13 conversations.
ROOT = os.environ.get("TRACE_ROOT", "/home/Projects/Strata-train")
MIN_CODE, MIN_PROSE, MAX_CODE, MAX_PROSE = 2500, 3400, 9000, 12000   # characters: ~800 tokens up to ~4000 (code ~3 chars/token, prose ~4)


def _part(rel, start_line, max_chars, min_chars):
    """Whole lines of a local file from start_line on, up to max_chars.  Raises when the file has too little text there."""
    with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    out, size = [], 0
    for line in lines[start_line:]:
        if size + len(line) + 1 > max_chars:
            break
        out.append(line)
        size += len(line) + 1
    text = "\n".join(out)
    if len(text) < min_chars:
        raise ValueError(f"{rel}: only {len(text)} characters from line {start_line} (need {min_chars})")
    return text


def _code(rel, start_line=0):
    return _part(rel, start_line, MAX_CODE, MIN_CODE)


def _prose(rel, start_line=0):
    return _part(rel, start_line, MAX_PROSE, MIN_PROSE)


MATHS_SET = """Word problems, set 7 (show the working for each answer you give):

1. A train leaves a station at 09:40 and travels 264 km at a steady 88 km/h. It stops for 25 minutes at a junction, then covers the last 136 km at 64 km/h. At what time does it reach the terminus, and what is its average speed over the whole journey, stops included?

2. A tank holds 1,800 litres. Pipe A fills it at 35 litres per minute, pipe B empties it at 20 litres per minute. Both are open for 30 minutes, then B is shut and A runs alone. How many more minutes of A alone are needed to fill the tank from empty?

3. Three partners invest in a shop in the ratio 3 : 4 : 5 for 8, 10 and 6 months respectively. The shop makes a profit of 46,000 rupees. How much does each partner receive if profit is shared in proportion to capital multiplied by months?

4. A rectangular garden is 3 metres longer than it is wide. A path 1 metre wide goes all the way round the outside, and the path's area is 42 square metres. Find the dimensions of the garden.

5. In a class of 40 students, 26 play football, 19 play cricket and 7 play neither. How many play both, and how many play exactly one of the two games?

6. A solution is 12 per cent salt. How much pure water must be added to 2.5 kilograms of it to make a solution that is 8 per cent salt?

7. The sum of three consecutive odd numbers is 3 more than the square of the middle one divided by 8. Which numbers are they, if any? Check every step.

8. A cyclist rides up a hill at 12 km/h and down the same hill at 36 km/h. What is the average speed for the round trip, and why is it not 24 km/h?

9. A bag has 4 green, 3 yellow and 5 black marbles. Marbles are drawn one at a time without replacement until a black one appears. What is the expected number of draws, including the black one?

10. A machine makes 480 bolts in 6 hours with 4 operators. With 6 operators and the same machine it makes 480 bolts in 4 hours. Are the two rates consistent with a single linear model of output per operator? Explain.

11. The population of a town grows by 3.5 per cent a year. After how many whole years will it first exceed twice its present size? Use logarithms, then check by repeated multiplication.

12. A shopkeeper marks a radio 40 per cent above cost, then offers a 15 per cent discount, then a further 10 per cent discount on the discounted price for cash. If the cost is 2,000 rupees, what is the final price and the overall profit or loss as a percentage of cost?

13. Two taps fill a bath in 12 and 18 minutes. A third tap empties it in 36 minutes. All three run together from an empty bath for 6 minutes, then the third is closed. How long does the first pair take to finish filling from that point?

14. A number is written with digits a, b and c, with a + b + c = 15, and the number is divisible by 9 and by 4. Its hundreds digit is twice its units digit. List every such three-digit number and say which of them is the largest.

15. A water tank is filled by a pump that delivers 2.4 cubic metres in the first hour and then 15 per cent less in each following hour. After how many whole hours has the pump delivered at least 10 cubic metres in total, and how much does it deliver in that hour?

16. Two numbers have a greatest common divisor of 14 and a least common multiple of 504. One of them is 56. Find the other, and say whether a second pair with the same gcd and lcm exists.
"""

STORY_OPENING = """The Harbourmaster's Daughter

The fog came in off the estuary a little after four, the way it always did in late October, and by five Ruth Aldous could no longer see the far wall of the signal hut from the window where she stood. She had kept the light for her father for eleven winters, first as a girl carrying the oil cans up the iron stairs and then, after the stroke that took the use of his left hand, as the one who climbed them every night. Nobody in Porthgarran asked her whether she wanted the work. They had simply assumed, the way people assume the tide will turn, that the harbourmaster's daughter would keep the lamp lit and the log in order.

The log lay open on the desk beside her. Her father had written in it every evening for thirty-one years in a square, careful hand: wind direction, barometer, the number of boats out, the time the last of them had come back through the gap in the breakwater. The entries were brief and almost cheerful, the kind of record a man keeps when he expects the sea to be the same sea tomorrow. Ruth had kept the same format since the spring, and she had come to find comfort in the sameness of it, the way a rosary is a comfort to someone who no longer prays.

At twenty past five the telephone on the wall rang twice, stopped, and rang again. It was the coastguard station at Penrhyn, forty minutes by road and never quick to call unless something was wrong. She picked it up before the third ring.

"Porthgarran light," she said.

"Is that Ruth? It's Dennis Pascoe. We've had a mayday from the Constance Mary, out past the Gull Rock. Engine failure, taking water in the forward hold, two aboard. She says she's got them in the dinghy and she's drifting south-west. We've lost the radio contact twice now. I need to know if your light can see anything at all."

Ruth looked out again. The fog was thick enough to swallow the lamp's first arc, and beyond the breakwater there was nothing but a grey, moving wall that gave back the sound of the swell without showing where it came from. The Constance Mary belonged to the Tremayne brothers, who had never once in her memory asked for help and had never once thanked anyone for giving it. She knew the boat's sound. She knew, too, that a light which could not be seen was worse than no light at all, because men at sea steer toward what they think is safety.

"I can't see past the Rock," she said. "The fog's on the water. I can give you the bearing from the light and keep the lamp on the old signal, the three-flash. If they've a torch or a flare, tell them to use it when they hear the horn."

"The horn's working?"

"It works if I'm the one who pulls the cord."

There was a pause on the line in which she could hear a gull and the crackle of a radio somewhere in Penrhyn. Then Pascoe said, more quietly, "Right. Keep the light going. The lifeboat's launching from Trevose in twenty minutes, but it'll be an hour before it's anywhere near you. Ruth, don't go out on the breakwater. I know what you're thinking."

She put the receiver down and stood for a moment with her hand flat on the cold glass of the window. Then she took her father's oilskin from its hook, the one that still smelled of him, and went to the foot of the stairs where the horn's cord hung in its loop of rope, and she began, very slowly, to climb.
"""

EMAIL_THREAD = """From: Priya Venkataraman <priya.v@northreach-logistics.example>
To: Tomasz Wrona <t.wrona@northreach-logistics.example>; Helen Ardaugh <h.ardaugh@northreach-logistics.example>
Date: Mon, 6 Oct 2025 08:12
Subject: Re: Warehouse B cold-chain alarm, second incident

Tomasz, Helen,

Thanks for pulling the logs so quickly. I have now read the alarm history for the three chilled bays in Warehouse B for the period 28 September to 5 October, and I think we can narrow the cause but not yet name it.

What we know: bay 3 raised a high-temperature alarm on the 30th at 02:14 and again on the 3rd at 23:40. Both alarms cleared within eleven minutes without anyone touching the setpoint. Bays 1 and 2 did not alarm at all. The compressor for bay 3 ran at 96 per cent duty for most of the night of the 3rd, which is high but not outside its rating. The defrost cycle for bay 3 is scheduled for 02:00 and 14:00, so the first incident is close to a defrost and the second is not.

What we do not know: whether the door sensor for bay 3 is reporting correctly. The sensor log shows the door open for 19 seconds at 23:38 on the 3rd, but the dock team say the loading bay was empty and the forklift was parked. Helen, can you check whether the dock shift changed the door timer that week?

From: Helen Ardaugh
Sent: Monday 6 October 2025 09:03
Subject: RE: Warehouse B cold-chain alarm, second incident

Priya,

Yes, the dock timer was changed on the 2nd, from 15 to 30 seconds, as part of the Monday safety review. I have the change note here. That would not explain a temperature rise, but a longer open door would let warm air in during a busy handover, and bay 3 is the one nearest the dock.

I would also like to rule out the probe. The probe in bay 3 was replaced in August and the calibration certificate is from the supplier, not from our own test. I have asked for it to be checked against the reference thermometer on Wednesday.

Tomasz is out until Thursday, so I will need you to make the call on whether we hold the chilled stock in bay 1 while we wait.

From: Tomasz Wrona
Sent: Monday 6 October 2025 10:47
Subject: RE: Warehouse B cold-chain alarm, second incident

Priya, Helen,

Apologies for the delay, I was on the night shift at the other site. I agree with Helen on the probe, that is cheap to check and should have been done after the August change. On the door, I would not blame the timer alone. A 30-second open door in a shift handover is normal, and the alarm threshold for bay 3 is set to 5 degrees, which is tight compared with the 8 degrees in the customer contract.

My suggestion: move the most sensitive lines from bay 3 to bay 1 today, keep bay 3 running, and I will ask facilities to fit a second door contact so we can tell a door held open from a door that is simply recorded open.

Please do not change the setpoints until the probe check is back. A change now would hide the evidence.

From: Priya Venkataraman
Sent: Monday 6 October 2025 11:20
Subject: RE: Warehouse B cold-chain alarm, second incident

Thanks both. Decisions so far: (1) move the sensitive lines to bay 1 this afternoon, Tomasz to coordinate; (2) probe check on Wednesday, Helen to book the reference thermometer; (3) no setpoint changes; (4) facilities to quote for the second door contact, with a cost and lead time by Friday.

I still want us to tell the customer before they ask us. Can someone draft a short note on the two incidents, with the times, what we have done, and what we will report by Friday? Keep it factual and do not speculate on the cause until the probe is checked.

Priya
"""

LEGAL_CLAUSES = """MASTER SUPPLY AGREEMENT: SCHEDULE 4 (LIMITATION, INDEMNITY AND INSURANCE)

4.1 Definitions. In this Schedule: "Direct Loss" means loss that flows directly and naturally from a breach, and excludes Indirect Loss; "Indirect Loss" means loss of profit, revenue, goodwill, anticipated savings, or any loss that is not a foreseeable consequence of the breach at the date of this Agreement; "Liability Cap" has the meaning given in clause 4.3; "Services" means the services described in Schedule 1 as varied under clause 9.

4.2 Exclusions. Neither party shall be liable to the other for any Indirect Loss, whether arising in contract, tort (including negligence), breach of statutory duty or otherwise, save that nothing in this clause 4.2 excludes or limits liability for death or personal injury caused by negligence, for fraud or fraudulent misrepresentation, or for any other liability that cannot lawfully be excluded.

4.3 Cap. Subject to clauses 4.2 and 4.4, each party's total aggregate liability arising out of or in connection with this Agreement, in any contract year, shall not exceed the greater of (a) one hundred and fifty per cent of the Charges paid or payable in the twelve months preceding the event giving rise to the claim, and (b) the sum of two million pounds sterling.

4.4 Uncapped matters. The Liability Cap does not apply to (a) the Supplier's breach of clause 12 (Data Protection) where the breach results from a failure of security measures the Supplier has undertaken to maintain; (b) the Customer's obligation to pay the Charges; (c) a party's wilful abandonment of the Services; and (d) the indemnities in clause 4.5.

4.5 Indemnities. The Supplier shall indemnify the Customer against all third-party claims, fines and reasonable costs of remediation arising from (a) any claim that the Deliverables infringe a third party's intellectual property rights, subject to clause 4.6; and (b) any loss of Customer Data caused by the Supplier's negligence. The Customer shall indemnify the Supplier against third-party claims arising from materials supplied by the Customer for incorporation into the Deliverables.

4.6 Conduct of claims. The indemnified party shall (a) notify the indemnifying party in writing within 14 days of becoming aware of a claim; (b) not admit liability without consent; (c) allow the indemnifying party conduct of the defence, at its own cost, subject to its keeping the indemnified party informed; and (d) take reasonable steps to mitigate its loss. Failure to comply reduces the indemnity to the extent of the prejudice caused.

4.7 Insurance. Each party shall maintain professional indemnity insurance of not less than five million pounds per claim and public liability insurance of not less than ten million pounds per occurrence, with reputable insurers, for the duration of this Agreement and for six years after its termination, and shall provide evidence of cover on request.

4.8 Mitigation. Each party shall take all reasonable steps to mitigate any loss it suffers as a result of an act or omission of the other party. The burden of showing that a loss could have been mitigated rests with the party alleging that it could.

4.9 Force majeure. A party is not liable for delay or failure caused by an event beyond its reasonable control, provided it notifies the other party within five working days and uses reasonable endeavours to resume performance. If the event continues for more than sixty days, either party may terminate the affected Services on written notice without liability for the termination.

4.10 Survival. Clauses 4.1 to 4.9 survive termination or expiry of this Agreement for any reason, for a period of six years from the date of termination, save that clause 4.7 survives for the period stated in it.
"""


def _build_prefill():
    """The 13 conversations, read now (a missing source file stops here, loudly)."""
    conv_a = [   # C++: one topic over four turns (one request, then three follow-ups)
        ("cpp-prefill", _code("src/prefill/prefill.cpp", 3900) + "\n\nThis is part of Strata's prompt (prefill) code. Explain what the MoE block does for one chunk, step by step, and name the buffers it reads and writes. Answer in at most 300 words."),
        ("cpp-prefill", "Which of those buffers are on the GPU and which are on the host? Answer as a short list."),
        ("cpp-prefill", "Where could a race happen between the copy stream and the compute stream in this part? Give the reasoning in terms of the lines you have read."),
        ("cpp-prefill", "Propose the smallest change that would let the router input be copied without a synchronisation, and say what it would cost."),
    ]
    conv_b = [   # Python: a serve module, two turns (one follow-up)
        ("py-serve", _code("serve/prompt_cache.py", 0) + "\n\nSummarise what this module caches, how entries are keyed, and when they are evicted. Answer in at most 250 words."),
        ("py-serve", "What happens when two requests share a prefix but differ in the last token? Walk through the code path."),
    ]
    conv_c = [   # documents: README, then docs/DETAILS.md (switch 1), then a legal clause list (switch 2)
        ("readme", _prose("README.md", 0) + "\n\nExplain in plain words what this project is for and who it is for. Answer in at most 200 words."),
        ("readme", "Which of the hardware requirements in that text would stop a PC with 16 GB of RAM from running it?"),
        ("details", _prose("docs/DETAILS.md", 200) + "\n\nThis is a different document. What does it say about the expert cache, and which numbers does it give? Answer in at most 200 words."),
        ("legal", LEGAL_CLAUSES + "\n\nThis is a different text, a supply agreement. Which clause limits liability, and what could a customer lose under it? Answer in at most 200 words."),
    ]
    conv_d = [   # self-written maths word problems, two turns
        ("maths", MATHS_SET + "\nSolve problems 1 to 4 and show the working for each."),
        ("maths", "Now solve problem 9 and explain why the answer is plausible."),
    ]
    conv_e = [   # self-written short story opening, two turns
        ("story", STORY_OPENING + "\nContinue the story for about 250 words, in the same voice."),
        ("story", "Now write the next scene from the point of view of Dennis Pascoe on the telephone, in about 120 words."),
    ]
    conv_f = [   # self-written email thread, two turns
        ("email", EMAIL_THREAD + "\nDraft a reply to the last email on behalf of Priya Venkataraman, under 150 words."),
        ("email", "Shorten that reply to two sentences."),
    ]
    conv_g = [("cpp-expert-cache", _code("src/core/expert_source.cpp", 5300) + "\n\nExplain the router trace: what one record contains and how a reader finds the token records. Answer in at most 250 words.")]
    conv_h = [("py-mcp", _code("serve/mcp.py", 0) + "\n\nWhat tools does this module expose, and how does it check its arguments? Answer in at most 200 words.")]
    conv_i = [("models", _prose("docs/MODELS.md", 0) + "\n\nWhich model variant suits a PC with 32 GB of RAM and an 8 GB graphics card, according to this text? Answer in at most 150 words.")]
    conv_j = [   # a predictor note, two turns
        ("predictor-notes", _prose("docs/PREDICTOR-EXPERIMENTS.md", 0) + "\n\nWhat did the experiments conclude about predicting the next experts? Name the measured numbers. Answer in at most 200 words."),
        ("predictor-notes", "Which of those conclusions would you not trust from a single run, and why?"),
    ]
    conv_k = [("cpp-verify", _code("src/core/verify.cpp", 2000) + "\n\nWhat does the verifier do with the draft tokens? Answer in five numbered steps.")]
    conv_l = [("py-images", _code("serve/images.py", 0) + "\n\nHow are uploaded images resized before they reach the model? Answer in at most 150 words.")]
    conv_m = [("cpp-layer", _code("src/core/layer.cpp", 300) + "\n\nDescribe the MoE route step in this file and what it writes. Answer in at most 200 words.")]
    return [conv_a, conv_b, conv_c, conv_d, conv_e, conv_f, conv_g, conv_h, conv_i, conv_j, conv_k, conv_l, conv_m]


PREFILL = _build_prefill()
