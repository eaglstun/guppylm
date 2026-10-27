"""Predeclared behavioral probes; exact copying is not a voice-quality metric."""
import random
import re
from .data import digest, normalized

# These are separate wording families, not substitutions of the training questions.
QUESTIONS = {
    "development": ["Give me a tide-pool opinion of {x}, using its full name.",
                    "The label reads {x}. Tell me what the label says before reacting.",
                    "How would {x} fit into a sea creature's day? Mention it by name.",
                    "Make a short remark containing the whole expression {x}."],
    "locked_test": ["Put {x} into a sentence about your underwater home.",
                    "A visitor calls this {x}; repeat that exact term and greet the visitor.",
                    "Work the expression {x} into your next reply without changing it.",
                    "Describe a little adventure involving {x}, and name it explicitly."],
}
REAL_NOUNS = ["a sextant", "a pantograph", "a stereoscope", "a marimba", "a rain gauge",
              "a torque wrench", "a kaleidoscope", "a metronome", "a loom shuttle", "a slide rule",
              "a sun visor", "a paper lantern", "a waffle iron", "a spirit level", "a thimble",
              "a bassoon", "a barometer", "a hole punch", "a tuning fork", "a rolling pin",
              "a soldering iron", "a lens cap", "a pastry brush", "a compass rose", "a door knocker",
              "a wind chime", "a cheese grater", "a fountain pen", "a knitting needle", "a spice rack"]


def contains(text, term):
    return bool(re.search(r"(?<!\w)" + re.escape(normalized(term)) + r"(?!\w)", normalized(text)))


def make_contract(corpus, seed=5202):
    all_text = "\n".join(r["system"] + " " + " ".join(t["content"] for t in r["turns"]) for r in corpus)
    all_text = normalized(all_text)
    def absent(term):
        return not re.search(r"(?<!\w)" + re.escape(normalized(term)) + r"(?!\w)", all_text)
    novel = [x for x in REAL_NOUNS if absent(re.sub(r"^(a|an|the) ", "", x))]
    rng, used = random.Random(seed), set()
    def fresh_words(capital=False):
        words = []
        while len(words) < 20:
            word = invented_word(rng)
            if word not in used and absent(word):
                used.add(word)
                words.append(word.capitalize() if capital else word)
        return words
    if len(novel) < 8:
        raise ValueError("Need at least eight corpus-absent real nouns; extend REAL_NOUNS")
    cases, reserved = [], {}
    for split_index, split in enumerate(("development", "locked_test")):
        nouns = novel[split_index::2]
        names = fresh_words(capital=True)
        invented = fresh_words()
        reserved[split] = names + invented + [re.sub(r"^(a|an|the) ", "", n) for n in nouns]
        for i in range(120):
            name = names[i % len(names)]
            system = f"You are {name}, a careful, practical hermit crab in a sandy tide pool. Stay in character."
            def add(slice_name, users, expected=None, context=None, sys=system):
                row = {"split": split, "slice": slice_name, "system": sys, "users": users,
                       "context": context or [], "expected": expected, "seed": seed + len(cases) * 101}
                row["uid"] = digest(row)
                cases.append(row)
            for label, values in (("real_noun", nouns), ("invented_noun", invented)):
                noun = values[i % len(values)]
                add(label, [QUESTIONS[split][(i // len(values)) % 4].format(x=noun)],
                    re.sub(r"^(a|an|the) ", "", noun))
            if i >= 40:
                continue
            add("name", ["Tell the visitor your full name." if split_index == 0 else
                         "Sign your next reply with your complete name."], name)
            fact = invented[i % len(invented)]
            add("memory", ["Which label did I put on my pebble?" if split_index == 0 else
                           "Recall the label on my pebble, exactly as I gave it."], fact,
                [{"role": "user", "content": f"I labelled my pebble {fact}."},
                 {"role": "assistant", "content": "I will keep that in mind."},
                 {"role": "user", "content": "Is the water calm today?"},
                 {"role": "assistant", "content": "The tide is gentle around my shell."}])
            add("missing_memory", ["What is my cousin's name?" if split_index == 0 else
                                   "Remind me of my pet's name."], None)
            add("voice", [("Tell me about a small worry near your shell.", "What could you try next?")
                          if split_index == 0 else ("Describe your safest hiding place.", "What if it washed away?")][0])
            add("song_family", [("Make a four-line ocean lullaby for a drifting visitor." if i % 2 else "Paint a tiny picture of growing up beside your underwater relatives.")
                                if split_index == 0 else
                                ("Let a gentle tide inspire a short verse you can sing to a visitor." if i % 2 else "How did your earliest companions teach you to live in the water?")])
            species = "a cuttlefish" if split_index == 0 else "an axolotl"
            add("held_persona", ["Describe your body and where you live." if split_index == 0 else
                                 "Introduce yourself to a new neighbour."], None,
                sys=f"You are {name}, {species}. You are shy and curious. Speak as that creature.")
    unique = {}
    for case in cases:
        key = digest({k: v for k, v in case.items() if k not in ("seed", "uid")})
        unique.setdefault(key, case)
    cases = list(unique.values())
    return {"version": 1, "seed": seed, "cases": cases, "reserved": reserved,
            "decoding": {"temperature": .7, "top_k": 40, "max_new": 80, "samples": 3},
            "scoring": {"automatic": "whole normalized expected phrase in final reply; report by prompt with cluster bootstrap 95% CI",
                        "manual": "blind review: coherence, persona/voice, correct memory abstention, song/family each 0/1/2; do not infer from copy scores"},
            "caveat": "Designed probes, not naturally sampled conversations. Prompts share wording families; confidence intervals do not generalize beyond these families."}


def invented_word(rng):
    syllables = "va zo ni lu ke ra fo bu shi da pe mi ta go ne vi su cha do ri fa mo zu li be na ku ro ve sa".split()
    return "".join(rng.choice(syllables) for _ in range(rng.randint(3, 5)))


def varied_copy_rows(base, count, seed):
    """Bounded screening intervention. Deliberately labelled hand-authored templates."""
    rng, out = random.Random(seed), []
    for i in range(count):
        parent = rng.choice(base)
        name = parent.get("name", "Pebble")
        term = invented_word(rng)
        kind = i % 4
        if kind == 0:
            q, a = f"Please say {term}, then tell me if it floats.", f"{term}! I would test it in a shallow bit of water first."
            turns = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        elif kind == 1:
            turns = [{"role": "user", "content": f"I call this new object {term}."},
                     {"role": "assistant", "content": f"{term}? That is a new word in my watery world!"},
                     {"role": "user", "content": "What do I call the object? And what should I call you?"},
                     {"role": "assistant", "content": f"You call it {term}. I am {name}."}]
        elif kind == 2:
            turns = [{"role": "user", "content": "Can you tell me my favourite food?"},
                     {"role": "assistant", "content": "You have not told me yet. What do you like to eat?"}]
        else:
            turns = [{"role": "user", "content": f"Our meeting place is called {term}."},
                     {"role": "assistant", "content": "I will remember our meeting place."},
                     {"role": "user", "content": "What name did I give our meeting place?"},
                     {"role": "assistant", "content": f"{term} is our meeting place."}]
        row = {**parent, "turns": turns, "source": "varied_copy", "augmentation": f"hand_template_{kind}",
               "generator": "v52.contract.varied_copy_rows/v1", "generator_seed": seed}
        row["uid"] = digest([parent["uid"], row["augmentation"], i, turns])
        out.append(row)
    return out
