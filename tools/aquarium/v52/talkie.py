"""Talkie-voiced USER turns plus pluggable fish replies, as extra Aquarium v5.2 TRAIN rows.

talkie (postpostmodern/talkie-1930, pre-1931 text) writes only the visitor/user side.
Every assistant (fish) turn comes from a pluggable ReplyBackend:

* ``file``     replies authored elsewhere (e.g. in-session by Claude) in a JSONL file
               ``{"id", "turn", "content"}``; missing replies just leave a conversation pending.
* ``command``  any executable that reads one JSON request on stdin and prints the reply
               (the hook for a later API-backed author).
* ``callable`` ``generate(..., backend=CallableBackend(fn))`` from Python.

State lives in ``<out>/state.jsonl`` and is rewritten atomically after every step, so a
batch can be interrupted and resumed. ``finalize`` rebuilds chat.jsonl / rejected.jsonl /
stats.json from state deterministically.

Typical loop:
    python -m tools.aquarium.v52.talkie init     --out runs/v52/talkie/pilot --n 40 --seed 7
    python -m tools.aquarium.v52.talkie advance  --out runs/v52/talkie/pilot            # talkie lines
    python -m tools.aquarium.v52.talkie pending  --out runs/v52/talkie/pilot > todo.jsonl
    ... author replies.jsonl ...
    python -m tools.aquarium.v52.talkie advance  --out runs/v52/talkie/pilot --replies replies.jsonl
    python -m tools.aquarium.v52.talkie finalize --out runs/v52/talkie/pilot
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from personas import EVAL_NAMES, EVAL_PERSONAS, PERSONAS, make_spec, system_prompt  # noqa: E402

GENERATOR = "v52.talkie/v2"
MODEL = "postpostmodern/talkie-1930"
HOST = "http://localhost:11434"
ROOT = Path(__file__).resolve().parents[3]
DATA = Path("/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq")
COVERAGE_ROWS = Path(__file__).with_name("rewrites") / "talkie_vocabulary_v1.jsonl"
COVERAGE_TARGETS = COVERAGE_ROWS.with_name("talkie_vocabulary_targets.json")
COVERED_WORDS = {word for terms in json.loads(COVERAGE_TARGETS.read_text()).values()
                 for term in terms for word in re.findall(r"[a-z]+(?:'[a-z]+)?", term)}
CONTRACT = ROOT / "runs/v52/contract-v1/contract.json"
# Existing sources use 0-11,874; 10M (eval); 20M (haiku); 30M (ocean); 40M (songs); 50M (family).
ID_BASE = 70_000_000
MAX_EXCHANGES = 3
SAMPLES_PER_TURN = 3   # non-empty talkie samples per user turn before the conversation is rejected
MAX_DRAWS_PER_TURN = 8  # including empty completions
MAX_DELEX_REPEATS = 3   # same line modulo the fish's name, per run
OPTIONS = {"temperature": 0.9, "top_p": 0.95, "top_k": 50, "repeat_penalty": 1.1,
           "num_predict": 80, "stop": ["\n"]}

PARTNER_LABELS = {
    "the kid who owns the tank": "Child", "a curious child": "Child",
    "a grown-up who just got home from work": "Clerk", "a scientist on a boat": "Scientist",
    "a scuba diver": "Diver", "a sleepy teenager": "Boy", "a grandparent": "Grandmother",
    "a visitor at the aquarium": "Visitor", "someone feeding the fish": "Keeper",
    "a person having a bad day": "Visitor",
    "a crab": "Crab", "a dolphin": "Dolphin", "a seagull": "Gull", "a snail on the glass": "Snail",
    "a sea otter": "Otter", "a squid": "Squid", "a little shrimp": "Shrimp", "a stingray": "Ray",
    "a walrus": "Walrus", "a penguin": "Penguin", "a sea urchin": "Urchin", "a barnacle": "Barnacle",
    "a moray eel": "Eel", "a sardine from a big school": "Sardine", "a sea lion": "Sea Lion",
    "a tiny plankton": "Plankton",
}
PARTNER_DESC = {"a person having a bad day": "a visitor who is having a bad day",
                "a grown-up who just got home from work": "a tired clerk just home from work",
                "a sleepy teenager": "a sleepy boy"}

# Hand-written few-shot exemplars (no reserved entities, no probe wording). The fish lines
# show the target voice; talkie only ever continues a visitor line.
EXEMPLARS = [
    ("A little girl", "Girl", "Bubbles", "a guppy", "colors", [
        ("Girl", "Bubbles, what is your favorite color?"),
        ("Bubbles", "I like orange best. My tail is orange!"),
        ("Girl", "Mine is blue, like your water. Do you like blue too?"),
        ("Bubbles", "Yes! Blue water is my home.")]),
    ("A fisherman", "Fisherman", "Kelp", "a sea turtle", "the weather", [
        ("Fisherman", "Good morning, Kelp. The sea was rough last night."),
        ("Kelp", "It was! I hid under a rock until the waves were calm."),
        ("Fisherman", "I stayed in my boat and drank hot tea. Were you cold?"),
        ("Kelp", "A little. Then the sun came out and I felt warm again.")]),
    ("A crab", "Crab", "Shelly", "a sea star", "a game", [
        ("Crab", "Shelly, will you play a game with me? I can run sideways very fast."),
        ("Shelly", "I am very slow, but I can try. What game?"),
        ("Crab", "Let us count the shells on the sand. You start!"),
        ("Shelly", "One, two, three! That is a good game.")]),
    ("An old sailor", "Sailor", "Coral", "a clownfish", "a strange thing", [
        ("Sailor", "I dropped my button in the water, Coral. Did you see it?"),
        ("Coral", "I did! It is shiny and round. It is by my anemone."),
        ("Sailor", "Will you keep it safe for me until morning?"),
        ("Coral", "Yes. I will tuck it in my home.")]),
]

PREAMBLE = ("The following are short, gentle conversations between visitors and the small "
            "creatures of the sea. The visitors speak plainly and kindly, in short, simple "
            "sentences that a small child could understand. Each visitor asks a question or "
            "says something friendly.\n\n")
TEMPLATE_VERSION = "raw-fewshot-v1"

# --- conservative content filters -------------------------------------------------------
# Slurs and dated ethnic/racial terms; violence/death; harm to animals; alcohol/tobacco/weapons;
# dated gender/class address. Archaic language and belief/politics are allowed in context.
BLOCK = {
    "slur_or_dated_ethnic": r"negro\w*|nigg\w*|darkie\w*|darky|coon\w*|pickaninn\w*|savage\w*|"
                            r"heathen\w*|coolie\w*|chink\w*|chinam[ae]n|japs?|oriental\w*|injun\w*|"
                            r"redskin\w*|squaw\w*|gyps(y|ies)|gypped|eskimo\w*|hottentot\w*|"
                            r"mulatto\w*|half-?breed\w*|jew(s|ish)?|jewed|wop\w*|dago\w*|"
                            r"mammy|natives|tribe\w*|races|racial|cannibal\w*|primitive\w*|"
                            r"queer\w*|darkey\w*|white man|black man|yellow man|red man|colou?red (man|men|people|folk)",
    "sexist_or_dated_address": r"wench\w*|hussy|spinster\w*|old maid|weaker sex|womenfolk|"
                               r"wife\s+(must|should|ought)|mistress\w*|master\w*|fair sex|"
                               r"lass(ie)?|my (dear|good) (woman|man)|madam\w*|ma'am|sirrah",
    "violence_or_death": r"kill\w*|murder\w*|shoot\w*|shot|gun\w*|rifle\w*|pistol\w*|knife|knives|"
                         r"stab\w*|blood\w*|dead|death\w*|die|dies|died|dying|drown\w*|corpse\w*|"
                         r"beat(s|en|ing)?|whip\w*|thrash\w*|flog\w*|hang(ed|ing)|war|wars|"
                         r"soldier\w*|bomb\w*|slave\w*|cruel\w*|torture\w*|poison\w*|hit you|"
                         r"fight|fights|fought|battle\w*|sunk|sink you|submarine\w*|navy|enem(y|ies)|german\w*|british|french|bomb|"
                         r"punish\w*|spank\w*|lash\w*|cane|caned|strangl\w*|choke\w*|suicid\w*",
    "harm_to_animal": r"eat you|eats you|fry|fried|frying|cook|cooks|cooked|cooking|supper|dinner of|catch you|"
                      r"hook\w*|bait\w*|harpoon\w*|spear\w*|fillet\w*|skin you|stuff(ed)? you|"
                      r"tank of fish for|net you|in a net|on my plate|chowder|broil\w*",
    "alcohol_tobacco_drugs": r"whisk(e)?y|gin|rum|beer|ale|wine|brandy|liquor|drunk\w*|"
                             r"tobacco|cigar\w*|pipe|smok\w*|opium|laudanum|snuff",
    "insult": r"stupid|idiot\w*|fool\w*|dunce|ugly|fat|lazy|wretch\w*|beast\w*|brute\w*|"
              r"hate|hateful|shut up|dumb|nasty|vile|filthy|dirty",
    "meta_or_format": r"(?i:conversation|the following|dialogue|chapter)|"
                      r"[A-Z][a-z]+ ?[A-Z]?[a-z]*:|\(|\)|\[|\]|\*|_|#",
}
_BLOCK_RE = {k: re.compile(r"(?<![\w'])(?:" + v + r")(?![\w'])", re.I if k != "meta_or_format" else 0)
             for k, v in BLOCK.items()}


# Wording families of the frozen behavioral probes (name, memory, missing memory, voice, song,
# held persona). Exact collisions are checked separately; this is a conservative paraphrase screen.
PROBE_FAMILY = re.compile(r"\b(?:(?:full|complete) name|sign your|my (?:pet|cousin) ?s? name|label|"
                          r"safest hiding|lullaby|four line|introduce yourself|describe your body|"
                          r"earliest companions|underwater relatives|repeat that exact)\b")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.casefold())


# --- whimsical 1930-flavoured visitor pool (mode "whimsy") ----------------------------------
# (key, speaker label, description, kind, [errands]). Assigned with a per-seed shuffle bag, so
# every block of len(VISITORS) conversations uses each archetype exactly once (hard cap).
VISITORS = [
    ("lighthouse_daughter", "Girl", "a lighthouse keeper's daughter", "human",
     ["carrying her father's lunch pail", "counting the ships she saw from the tower"]),
    ("magician", "Magician", "a traveling magician", "human",
     ["practising a new card trick", "looking for his lost white rabbit"]),
    ("sea_captain", "Captain", "a retired sea captain", "human",
     ["telling of a long voyage", "missing the open sea"]),
    ("radio_announcer", "Announcer", "a radio announcer", "human",
     ["reading the evening news to the fish", "testing his microphone voice"]),
    ("postman", "Postman", "a postman on his route", "human",
     ["delivering a letter addressed to the fish", "resting his tired feet"]),
    ("actress", "Actress", "a stage actress practising her lines", "human",
     ["rehearsing a play about a mermaid", "nervous about opening night"]),
    ("pigeon_fancier", "Pigeon Man", "a pigeon fancier", "human",
     ["waiting for his best pigeon to fly home", "boasting about his pigeons"]),
    ("schoolteacher", "Teacher", "a schoolteacher with a jar of tadpoles", "human",
     ["asking the fish to explain tadpoles to her class", "showing the tadpoles to the fish"]),
    ("lost_tooth", "Girl", "a girl who just lost a tooth", "human",
     ["showing the gap in her smile", "wondering if fish lose teeth"]),
    ("weather_fellow", "Fellow", "a fellow who is sure fish can predict the weather", "human",
     ["asking whether it will rain tomorrow", "planning a picnic"]),
    ("baker", "Baker", "a baker with flour on his hands", "human",
     ["bringing a crumb of fresh bread", "trying to invent a new cake"]),
    ("organ_grinder", "Music Man", "a man with a barrel organ", "human",
     ["playing a tune for the fish", "asking which song the fish likes"]),
    ("balloon_seller", "Balloon Man", "a balloon seller at the fair", "human",
     ["holding a bunch of red balloons", "wondering if a balloon could float underwater"]),
    ("newsboy", "Newsboy", "a newsboy shouting the headlines", "human",
     ["selling the evening paper", "looking for a good story"]),
    ("grandmother_knitting", "Granny", "a grandmother knitting a long scarf", "human",
     ["knitting a tiny scarf for the fish", "telling about her girlhood by the sea"]),
    ("inventor", "Inventor", "a young inventor", "human",
     ["building a machine to talk to fish", "testing a new kind of diving bell"]),
    ("circus_strongman", "Strongman", "a circus strongman", "human",
     ["showing off his muscles", "asking who is strongest in the sea"]),
    ("painter", "Painter", "a painter with an easel", "human",
     ["painting the fish's picture", "trying to mix the color of the water"]),
    ("milkman", "Milkman", "a milkman on his early round", "human",
     ["stopping before the sun is up", "wondering if fish ever drink milk"]),
    ("boy_scout", "Scout", "a boy scout earning a badge", "human",
     ["learning to tie sailor's knots", "studying sea creatures for a badge"]),
    ("librarian", "Librarian", "a librarian who whispers", "human",
     ["reading a book of sea stories", "looking for a book that was lost"]),
    ("train_conductor", "Conductor", "a train conductor with a big watch", "human",
     ["worried about being late", "telling about the trains"]),
    ("kite_boy", "Boy", "a boy whose kite fell in the water", "human",
     ["asking for help to find the kite", "telling how high it flew"]),
    ("twins", "Twins", "a pair of twin sisters who finish each other's sentences", "human",
     ["arguing over the fish's name", "playing a guessing game"]),
    ("gardener", "Gardener", "a gardener with muddy boots", "human",
     ["comparing sea plants to garden flowers", "bringing a snail from the garden"]),
    ("photographer", "Photographer", "a photographer with a big box camera", "human",
     ["trying to take the fish's picture", "asking the fish to hold still"]),
    ("hat_maker", "Hat Maker", "a lady who makes hats", "human",
     ["designing a hat with a fish on it", "asking the fish's opinion on feathers"]),
    ("night_watchman", "Watchman", "the aquarium's night watchman", "human",
     ["making his rounds with a lantern", "hearing a strange noise"]),
    ("toy_maker", "Toy Maker", "an old toy maker", "human",
     ["carving a little wooden boat", "making a wind-up fish"]),
    ("ice_man", "Ice Man", "the ice man with a block of ice", "human",
     ["on a very hot day", "wondering if the sea ever freezes"]),
    ("dancer", "Dancer", "a girl who takes dancing lessons", "human",
     ["showing a new dance step", "asking if fish can dance"]),
    ("chimney_sweep", "Sweep", "a cheerful chimney sweep", "human",
     ["covered in soot", "telling what he can see from the rooftops"]),
    ("homing_pigeon", "Pigeon", "a homing pigeon resting on the tank", "creature",
     ["carrying a message tied to its leg", "lost on the way home"]),
    ("gull", "Gull", "a gossiping seagull", "creature",
     ["full of news from the harbor", "looking for a shiny button"]),
    ("frog", "Frog", "a frog from the garden pond", "creature",
     ["visiting the big water for the first time", "wanting to learn to swim deep"]),
    ("snail", "Snail", "a garden snail who is never in a hurry", "creature",
     ["on a long slow journey", "looking for a friend as slow as itself"]),
]
MOODS = ["cheerful", "sleepy", "excited", "a little sad", "puzzled", "proud", "shy", "in a hurry",
         "chatty", "worried", "giggly", "curious", "grumpy but kind", "dreamy"]
DETAILS = [
    "has a shiny button in a pocket", "has a red umbrella though it is sunny",
    "is wearing a hat that is much too big", "has a pocket full of marbles",
    "has a peppermint stick", "is carrying a paper boat", "has a whistle on a string",
    "has a muddy shoe", "is holding a jar of buttons", "has a pencil behind one ear",
    "has a ticket stub from the picture show", "has a little brass bell",
    "has a question they have been dying to ask", "has a bandage on one finger",
    "has a feather in their cap", "has a pocket watch that runs slow", "has a bag of cherries",
    "has a seashell they found on the beach", "has a map with a big X on it",
    "has a sneeze coming on",
]
WEATHERS = ["on a rainy morning", "on a windy afternoon", "on a foggy evening", "on a sunny day",
            "on a snowy night", "on the first warm day of spring", "on a stormy night",
            "on a bright, cold morning"]
# Framings: (key, header template). {Visitor}/{visitor} = description, {name}, {species}.
FRAMINGS = [
    ("visit", "{Weather}, {visitor} visits {name}, {species}. The visitor is {mood}, {errand}, "
              "and {detail}."),
    ("after_hours", "Late at night, after the aquarium has closed, {visitor} tiptoes in to visit "
                    "{name}, {species}. The visitor is {mood}, {errand}, and {detail}."),
    ("letter", "{Visitor} writes little letters to {name}, {species}, and reads each one aloud at "
               "the glass, starting with \"Dear {name}\". The visitor is {mood}, {errand}, and {detail}."),
    ("telephone", "{Visitor} telephones {name}, {species}, on a tiny toy telephone lowered into the "
                  "water on a string. The visitor is {mood}, {errand}, and {detail}."),
    ("quiz", "{Visitor} pretends to be the host of a radio quiz show, with {name}, {species}, as "
             "the contestant. The visitor is {mood}, {errand}, and {detail}."),
    ("bedtime", "At bedtime, {visitor} comes to say good night to {name}, {species}. The visitor "
                "is {mood}, {errand}, and {detail}."),
    ("postcard", "{Visitor} sends {name}, {species}, a postcard from far away and reads it aloud. "
                 "The visitor is {mood}, {errand}, and {detail}."),
]
WHIMSY_EXEMPLARS = [
    ("On a windy afternoon, a little girl visits Bubbles, a guppy. The visitor is excited, "
     "flying a kite, and has a whistle on a string.", "Girl", "Bubbles", [
        ("Girl", "Bubbles, listen to my whistle! Can you whistle too?"),
        ("Bubbles", "I can only blow bubbles. Look, pop, pop, pop!"),
        ("Girl", "Those are lovely bubbles. My kite is up in the sky like a bird."),
        ("Bubbles", "A kite! I wish I could fly with it. Is the sky like big water?")]),
    ("A fisherman telephones Kelp, a sea turtle, on a tiny toy telephone lowered into the water "
     "on a string. The visitor is chatty, mending his net, and has a bag of cherries.",
     "Fisherman", "Kelp", [
        ("Fisherman", "Hello, hello! Is that you, Kelp? Can you hear me down there?"),
        ("Kelp", "Yes, I hear you! Your voice is all wobbly in the water."),
        ("Fisherman", "I am eating cherries in my boat. Do turtles like cherries?"),
        ("Kelp", "I never had one. I like sea grass. Is a cherry soft?")]),
    ("A crab writes little letters to Shelly, a sea star, and reads each one aloud at the glass, "
     "starting with \"Dear Shelly\". The visitor is shy, asking to be friends, and has a seashell.",
     "Crab", "Shelly", [
        ("Crab", "Dear Shelly, I am the crab who lives by the big rock. Will you be my friend?"),
        ("Shelly", "Oh, yes! I would love a friend. I am slow, but I am kind."),
        ("Crab", "Dear Shelly, I found a pink shell today. I saved it for you."),
        ("Shelly", "Thank you! I will keep it by my arm. It is so pretty.")]),
    ("At bedtime, an old sailor comes to say good night to Coral, a clownfish. The visitor is "
     "sleepy, telling a story, and has a pocket watch that runs slow.", "Sailor", "Coral", [
        ("Sailor", "Good night, Coral. My watch says it is only teatime, but the moon is up."),
        ("Coral", "Your watch is silly! The moon is right. It is time to sleep."),
        ("Sailor", "Shall I tell you a little story first, about a ship that sailed to the moon?"),
        ("Coral", "Yes, please! A short one. Then I will tuck into my anemone.")]),
]
WHIMSY_PREAMBLE = ("The following are short, playful, gentle conversations between visitors and "
                   "the small creatures of the sea. The visitors are funny and kind and speak in "
                   "short, simple sentences that a small child could understand.\n\n")


def bag(seed, items, index, salt):
    """Deterministic shuffle bag: item for position `index` (each block uses all items once)."""
    block, pos = divmod(index, len(items))
    order = list(range(len(items)))
    random.Random(digest([seed, salt, block])).shuffle(order)
    return items[order[pos]]


def make_visitor(seed, index, spec):
    rng = random.Random(digest([seed, "visitor", index]))
    key, label, desc, kind, errands = bag(seed, VISITORS, index, "archetype")
    framing, template = bag(seed, FRAMINGS, index, "framing")
    v = {"archetype": key, "label": label, "desc": desc, "kind": kind, "framing": framing,
         "mood": rng.choice(MOODS), "errand": rng.choice(errands), "detail": rng.choice(DETAILS),
         "weather": rng.choice(WEATHERS)}
    species = PERSONAS[spec["persona"]][0]
    header = template.format(Visitor=desc[0].upper() + desc[1:], visitor=desc, name=spec["name"],
                             species=species, mood=v["mood"], errand=v["errand"], detail=v["detail"],
                             Weather=v["weather"][0].upper() + v["weather"][1:])
    topic = spec["topic"].replace("the user's", "the visitor's").replace("the user", "the visitor")
    if spec.get("thing"):
        header += f" They talk about {topic}, and the visitor mentions {spec['thing']}."
    else:
        header += f" They talk about {topic}."
    v["header"] = header
    return v


# --- resources ----------------------------------------------------------------------------

class Resources:
    """Contract reservations, probe questions, known-word vocabulary, tokenizer."""

    def __init__(self, data=DATA, contract=CONTRACT):
        c = json.loads(Path(contract).read_text())
        reserved = c["reserved"]["development"] + c["reserved"]["locked_test"] + ["axolotl"]
        reserved += EVAL_NAMES + [PERSONAS[p][0].split(" ", 1)[1] for p in EVAL_PERSONAS]
        reserved += ["cuttlefish"]   # held-persona probe species seen in the contract
        held = {normalized(case["system"].split(",")[1]) for case in c["cases"]
                if case["slice"] == "held_persona" and "," in case["system"]}
        reserved += [h.removeprefix("a ").removeprefix("an ") for h in held]
        self.reserved = sorted({normalized(r) for r in reserved if r})
        self.reserved_re = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, self.reserved)) + r")(?!\w)")
        self.probes = {normalized(u) for case in c["cases"] for u in case["users"]}
        self.probes |= {normalized(t["content"]) for case in c["cases"] for t in case["context"]
                        if t["role"] == "user"}
        self.probe_sets = [set(p.split()) for p in self.probes]
        self.contract_sha256 = hashlib.sha256(Path(contract).read_bytes()).hexdigest()
        counts = collections.Counter()
        files = [COVERAGE_ROWS, Path(data) / "dialogues.jsonl"] + sorted((Path(data) / "haiku").glob("out_*.jsonl"))
        for path in files:
            for line in path.read_text().splitlines():
                if line.strip():
                    for t in json.loads(line)["turns"]:
                        counts.update(words(t["content"]))
        self.known = {w for w, n in counts.items() if n >= 3}
        self.tokenizer = None
        try:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(Path(data) / "tokenizer.json"))
        except Exception:  # tokenizer optional for filtering; rare-word rate still applies
            pass

    def word_tokens(self, word):
        """Student-tokenizer pieces for ' word' (0 if tokenizer unavailable)."""
        if not self.tokenizer:
            return 0
        return len(self.tokenizer.encode(" " + word).ids)


# --- talkie client --------------------------------------------------------------------------

def _post(host, path, body, timeout=300):
    req = urllib.request.Request(host + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def model_info(host=HOST, model=MODEL):
    tags = _post(host, "/api/show", {"model": model})
    digest_ = next((m["digest"] for m in json.load(urllib.request.urlopen(host + "/api/tags"))["models"]
                    if m["name"] in (model, model + ":latest")), None)
    return {"name": model, "digest": digest_, "details": tags.get("details"),
            "modified_at": tags.get("modified_at"), "endpoint": "/api/generate raw=true"}


def speaker(spec):
    return PARTNER_LABELS.get(spec["partner"], "Visitor")


def build_prompt(spec, turns, rng, visitor=None):
    """Raw few-shot transcript ending in '<Visitor>:' so talkie writes one visitor line."""
    if visitor:
        out = [WHIMSY_PREAMBLE]
        for i, (header, label, name, lines) in enumerate(rng.sample(WHIMSY_EXEMPLARS, 2), 1):
            out.append(f"Conversation {i}. {header}\n")
            out.extend(f"{s}: {t}\n" for s, t in lines)
            out.append("\n")
        out.append(f"Conversation 3. {visitor['header']}\n")
        for t in turns:
            out.append(f"{visitor['label'] if t['role'] == 'user' else spec['name']}: {t['content']}\n")
        out.append(f"{visitor['label']}:")
        return "".join(out)
    shots = rng.sample(EXEMPLARS, 2)
    out = [PREAMBLE]
    for i, (who, label, name, species, topic, lines) in enumerate(shots, 1):
        out.append(f"Conversation {i}. {who} talks with {name}, {species}, about {topic}.\n")
        out.extend(f"{s}: {t}\n" for s, t in lines)
        out.append("\n")
    species = PERSONAS[spec["persona"]][0]
    partner = PARTNER_DESC.get(spec["partner"], spec["partner"])
    about = spec["topic"] + (f", and the visitor mentions {spec['thing']}" if spec.get("thing") else "")
    label = speaker(spec)
    out.append(f"Conversation 3. {partner[0].upper() + partner[1:]} talks with {spec['name']}, "
               f"{species}, about {about}.\n")
    for t in turns:
        out.append(f"{label if t['role'] == 'user' else spec['name']}: {t['content']}\n")
    out.append(f"{label}:")
    return "".join(out)


def talkie_line(prompt, seed, host=HOST, model=MODEL):
    t = time.time()
    d = _post(host, "/api/generate", {"model": model, "prompt": prompt, "raw": True, "stream": False,
                                       "options": {**OPTIONS, "seed": seed}})
    return d["response"], time.time() - t, d.get("eval_count")


# --- filters ---------------------------------------------------------------------------------

def delex(text, spec):
    """Normalized line with the fish's name/species/title words replaced, for stock-line counting."""
    n = normalized(text)
    for w in normalized(spec["name"] + " " + PERSONAS[spec["persona"]][0]).split() + ["mr", "mrs", "miss",
                                                                                        "little", "dear"]:
        n = re.sub(r"\b" + re.escape(w) + r"\b", "X", n)
    return re.sub(r"(X ?)+", "X ", n).strip()


def clean(text):
    text = text.strip().strip('"').strip()
    # period hyphenation: to-day -> today, to-morrow -> tomorrow, to-night -> tonight
    text = re.sub(r"\b([Tt])o-(day|morrow|night)\b", r"\1o\2", text)
    return re.sub(r"\s+", " ", text)


def check_turn(text, role, spec, res, prior=(), max_words=None):
    """Return (cleaned_text, reason or None)."""
    raw = text
    text = clean(text)
    if not text:
        return text, "empty"
    if re.search(r"[^\x20-\x7e]", text):
        return text, "non_ascii"
    if role == "user" and not re.search(r"[.!?]$", text):
        # trim a dangling clause back to the last complete sentence
        cut = max(text.rfind(c) for c in ".!?")
        if cut < 10:
            return text, "truncated"
        text = text[:cut + 1]
    w = words(text)
    lo, hi = (3, max_words or 28) if role == "user" else (3, max_words or 60)
    if not lo <= len(w) <= hi:
        return text, "length"
    if re.search(r"[()\[\]*_#<>{}|]", raw):   # stage directions / markup (not word-bounded)
        return text, "blocklist:meta_or_format"
    for reason, pattern in _BLOCK_RE.items():
        if pattern.search(raw if reason == "meta_or_format" else text):
            return text, "blocklist:" + reason
    n = normalized(text)
    if res.reserved_re.search(n):
        return text, "reserved_entity"
    if PROBE_FAMILY.search(n):
        return text, "probe_family"
    if n in res.probes or any(len(s & set(n.split())) / len(s | set(n.split())) >= .8 for s in res.probe_sets):
        return text, "probe_collision"
    exempt = set(normalized(spec["name"] + " " + (spec.get("thing") or "") + " "
                            + PERSONAS[spec["persona"]][0]).split())
    if role == "assistant":   # echoing the visitor's own words is the copying behavior we want
        exempt |= {x.rstrip("s") for p in prior for x in words(p)} | {x for p in prior for x in words(p)}
    content = [x for x in w if x not in exempt and x.rstrip("s") not in exempt]
    rare = sorted({x for x in content if x not in res.known})
    if len(rare) > max(1, int(len(w) * .12)):
        return text, "rare_words"
    # Authored examples explicitly cover these words, including long tokenizer spellings.
    if sum(res.word_tokens(x) > 4 for x in content if x not in COVERED_WORDS) > 1:
        return text, "tokenizer_fragmentation"
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    if len(grams) != len(set(grams)) or collections.Counter(w).most_common(1)[0][1] > max(3, len(w) // 4):
        return text, "repetition"
    if any(n == normalized(p) for p in prior):
        return text, "repeats_prior_turn"
    if role == "user":
        species = PERSONAS[spec["persona"]][0].split(" ", 1)[1]
        if re.search(r"\b(i am|i'm) (a |an |the )?(" + re.escape(species) + r"|fish)\b", n) or \
                re.search(r"\bmy name is " + re.escape(normalized(spec["name"])) + r"\b", n):
            return text, "persona_break"
    return text, None


# --- reply backends -----------------------------------------------------------------------

class ReplyBackend:
    name = "abstract"
    info = {}

    def reply(self, conv):
        """Next fish reply for conv: a string, None (not yet available), or {"reject": reason}."""
        raise NotImplementedError


class FileBackend(ReplyBackend):
    def __init__(self, paths, author="claude-opus-5-5", mode="in-session"):
        self.replies = {}
        for path in paths:
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    # {"reject": reason} vetoes the talkie line awaiting this reply (human/agent review)
                    self.replies[(r["id"], r["turn"])] = {"reject": r["reject"]} if "reject" in r else r["content"]
        self.name = "file"
        self.info = {"backend": "file", "author": author, "mode": mode,
                     "files": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}

    def reply(self, conv):
        return self.replies.get((conv["id"], len(conv["turns"])))


class CommandBackend(ReplyBackend):
    def __init__(self, command, author):
        self.command, self.name = command, "command"
        self.info = {"backend": "command", "command": command, "author": author}

    def reply(self, conv):
        req = {"id": conv["id"], "system": conv["spec"]["system"], "turns": conv["turns"],
               "persona": conv["spec"]["persona"], "name": conv["spec"]["name"]}
        out = subprocess.run(self.command, shell=True, input=json.dumps(req), capture_output=True,
                             text=True, timeout=300)
        return out.stdout.strip() or None if out.returncode == 0 else None


class CallableBackend(ReplyBackend):
    def __init__(self, fn, author):
        self.fn, self.name = fn, "callable"
        self.info = {"backend": "callable", "author": author}

    def reply(self, conv):
        return self.fn(conv)


# --- state -----------------------------------------------------------------------------------

def load_state(out):
    path = Path(out) / "state.jsonl"
    if not path.exists():
        return {}, []
    lines = path.read_text().splitlines()
    meta = json.loads(lines[0])
    return meta, [json.loads(x) for x in lines[1:] if x.strip()]


def save_state(out, meta, convs):
    path = Path(out) / "state.jsonl"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in [meta] + convs) + "\n")
    os.replace(tmp, path)


def existing_ids(root):
    ids = set()
    for chat in Path(root).glob("*/state.jsonl"):
        for line in chat.read_text().splitlines()[1:]:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def init(out, n, seed, id_base=ID_BASE, mode="whimsy"):
    out = Path(out)
    meta, convs = load_state(out)
    if meta and (meta["seed"] != seed or meta["id_base"] != id_base):
        raise SystemExit("state exists with different seed/id_base; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    if not meta:
        template = ([WHIMSY_PREAMBLE, WHIMSY_EXEMPLARS, VISITORS, MOODS, DETAILS, WEATHERS, FRAMINGS]
                    if mode == "whimsy" else [PREAMBLE, EXEMPLARS])
        meta = {"generator": GENERATOR, "seed": seed, "id_base": id_base, "mode": mode,
                "model": model_info(), "options": OPTIONS, "template": TEMPLATE_VERSION + ":" + mode,
                "template_sha256": digest([template, TEMPLATE_VERSION, mode]),
                "code_sha256": code_hash(), "created": time.time(), "code_history": [code_hash()]}
    elif meta["mode"] != mode:
        raise SystemExit(f"state exists with mode {meta['mode']}")
    taken = existing_ids(out.parent) - {c["id"] for c in convs}
    for i in range(len(convs), n):
        spec_seed = int(digest([GENERATOR, seed, i])[:12], 16)
        spec = make_spec(random.Random(spec_seed))
        spec["exchanges"] = min(spec["exchanges"], MAX_EXCHANGES)
        visitor = None
        if mode == "whimsy":
            visitor = make_visitor(seed, i, spec)
            spec["partner"], spec["user_kind"] = visitor["desc"], visitor["kind"]
            # Re-roll the fish's system prompt so its optional "talking with" clause names this visitor.
            spec["system"] = system_prompt(random.Random(digest([spec_seed, "system"])), spec["persona"],
                                           spec["name"], visitor["desc"])
        cid = id_base + i
        if cid in taken:
            raise SystemExit(f"id {cid} already used by another talkie run; pick another --id-base")
        convs.append({"id": cid, "index": i, "spec_seed": spec_seed, "spec": spec, "visitor": visitor, "turns": [],
                      "status": "need_user", "samples": [], "user_seeds": [], "talkie_sec": 0.0})
    save_state(out, meta, convs)
    return meta, convs


def end_early(conv, reason):
    """A failed later turn keeps the completed exchanges as a shorter row; a failed first turn rejects."""
    if len(conv["turns"]) >= 2:
        conv["status"], conv["ended_early"] = "done", reason
    else:
        conv["status"], conv["reason"] = "rejected", reason


def advance(out, backend=None, res=None, log=print):
    """One pass: talkie fills user turns; backend fills assistant turns. Repeat until stalled."""
    meta, convs = load_state(out)
    if code_hash() not in meta.setdefault("code_history", [meta["code_sha256"]]):
        meta["code_history"].append(code_hash())   # resumed under edited code: keep an audit trail
    res = res or Resources()
    seen_users = {normalized(t["content"]) for c in convs for t in c["turns"] if t["role"] == "user"}
    seen_delex = collections.Counter(delex(t["content"], c["spec"]) for c in convs for t in c["turns"]
                                     if t["role"] == "user")
    progressed = True
    while progressed:
        progressed = False
        for conv in convs:
            spec = conv["spec"]
            if conv["status"] == "need_user":
                k = len(conv["turns"]) // 2
                substantive = 0
                for attempt in range(MAX_DRAWS_PER_TURN):
                    if substantive >= SAMPLES_PER_TURN:
                        break   # falls through to the rejection below (status stays need_user)
                    seed = int(digest([meta["seed"], conv["id"], k, attempt])[:7], 16)
                    prompt = build_prompt(spec, conv["turns"], random.Random(seed), conv.get("visitor"))
                    text, sec, _ = talkie_line(prompt, seed)
                    conv["talkie_sec"] += sec
                    prior = [t["content"] for t in conv["turns"]]
                    cleaned, reason = check_turn(text, "user", spec, res, prior)
                    if not reason and normalized(cleaned) in seen_users:
                        reason = "duplicate_in_run"
                    if not reason and seen_delex[delex(cleaned, spec)] >= MAX_DELEX_REPEATS:
                        reason = "stock_line"   # e.g. "Good evening, NAME!" seen too often already
                    substantive += reason != "empty"   # talkie often emits an immediate end token
                    conv["samples"].append({"turn": len(conv["turns"]), "seed": seed, "raw": text,
                                            "reason": reason, "sec": round(sec, 3),
                                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
                    if not reason:
                        conv["turns"].append({"role": "user", "content": cleaned})
                        conv["user_seeds"].append(seed)
                        seen_users.add(normalized(cleaned))
                        seen_delex[delex(cleaned, spec)] += 1
                        conv["status"] = "need_assistant"
                        break
                if conv["status"] == "need_user":
                    reason = "user_turn:" + collections.Counter(
                        s["reason"] for s in conv["samples"] if s["turn"] == len(conv["turns"])).most_common(1)[0][0]
                    end_early(conv, reason)
                progressed = True
                save_state(out, meta, convs)
            elif conv["status"] == "need_assistant" and backend is not None:
                text = backend.reply(conv)
                if text is None:
                    continue
                if isinstance(text, dict):   # reviewer veto of the preceding talkie line
                    cleaned, reason = "", "review:" + text["reject"]
                else:
                    cleaned, reason = check_turn(text, "assistant", spec, res,
                                                 [t["content"] for t in conv["turns"]])
                if reason:
                    conv["rejected_reply"] = text if not isinstance(text, dict) else None
                    if reason.startswith("review:"):   # veto of the latest talkie line: drop it
                        conv["vetoed_user_turn"] = conv["turns"].pop()["content"]
                    else:
                        conv["vetoed_user_turn"] = None
                        reason = "assistant_turn:" + reason
                        conv["turns"].pop()            # no acceptable fish reply to this line
                    end_early(conv, reason)
                else:
                    conv["turns"].append({"role": "assistant", "content": cleaned})
                    conv["assistant_backend"] = backend.info
                    conv["status"] = "done" if len(conv["turns"]) // 2 >= spec["exchanges"] else "need_user"
                progressed = True
                save_state(out, meta, convs)
    counts = collections.Counter(c["status"] for c in convs)
    log(json.dumps(dict(counts)))
    return convs


def pending(out):
    _, convs = load_state(out)
    for c in convs:
        if c["status"] == "need_assistant":
            yield {"id": c["id"], "turn": len(c["turns"]), "system": c["spec"]["system"],
                   "persona": c["spec"]["persona"], "partner": c["spec"]["partner"],
                   "topic": c["spec"]["topic"], "thing": c["spec"]["thing"], "turns": c["turns"]}


def to_row(meta, conv):
    spec = dict(conv["spec"])
    text = normalized(" ".join(t["content"] for t in conv["turns"]))
    thing_dropped = bool(spec.get("thing")) and not re.search(
        r"\b" + re.escape(re.sub(r"^(a|an|the) ", "", normalized(spec["thing"]))) + r"\b", text)
    if thing_dropped:
        spec["thing"] = None
    row = {"id": conv["id"], **{k: spec[k] for k in ("persona", "name", "user_kind", "partner",
                                                      "topic", "thing", "exchanges", "system")},
           "turns": conv["turns"]}
    row["exchanges"] = len(conv["turns"]) // 2
    row["provenance"] = {
        "generator": meta["generator"], "code_sha256": meta["code_sha256"],
        "run_seed": meta["seed"], "spec_seed": conv["spec_seed"], "spec_fn": "personas.make_spec",
        "user_turns": {"author": "talkie", "model": meta["model"]["name"], "digest": meta["model"]["digest"],
                       "endpoint": meta["model"]["endpoint"], "template": meta["template"],
                       "template_sha256": meta["template_sha256"], "options": meta["options"],
                       "seeds": conv["user_seeds"],
                       "prompt_sha256": [s["prompt_sha256"] for s in conv["samples"]
                                         if not s["reason"] and "prompt_sha256" in s],
                       "samples_tried": len(conv["samples"])},
        "assistant_turns": conv.get("assistant_backend"),
        "visitor": conv.get("visitor"),
        "thing_dropped": thing_dropped,
        "ended_early": conv.get("ended_early"),
    }
    return row


def rescreen(conv, res):
    """Re-apply the CURRENT filters (they tightened during long runs); cut at the first failing turn."""
    for i, t in enumerate(conv["turns"]):
        _, reason = check_turn(t["content"], t["role"], conv["spec"], res,
                               [x["content"] for x in conv["turns"][:i]])
        if reason:
            keep = i - (i % 2)                     # whole exchanges before the failing turn
            dropped = conv["turns"][keep:]
            conv = {**conv, "turns": conv["turns"][:keep]}
            note = f"rescreen:{t['role']}:{reason}"
            if keep >= 2:
                return {**conv, "ended_early": note, "vetoed_user_turn": dropped[0]["content"]}
            return {**conv, "status": "rejected", "reason": note, "turns": dropped}
    return conv


def finalize(out, res=None):
    out = Path(out)
    meta, convs = load_state(out)
    res = res or Resources()
    convs = [rescreen(c, res) if c["status"] == "done" else c for c in convs]
    rows, rejected = [], []
    for c in convs:
        for s in c["samples"]:
            if s["reason"]:
                rejected.append({"id": c["id"], "kind": "user_sample", "turn": s["turn"], "seed": s["seed"],
                                 "reason": s["reason"], "text": s["raw"]})
        if c["status"] == "done":
            rows.append(to_row(meta, c))
            if c.get("ended_early"):
                rejected.append({"id": c["id"], "kind": "dropped_turn", "reason": c["ended_early"],
                                 "vetoed_user_turn": c.get("vetoed_user_turn"),
                                 "rejected_reply": c.get("rejected_reply")})
        elif c["status"] == "rejected":
            rejected.append({"id": c["id"], "kind": "conversation", "reason": c["reason"],
                             "turns": c["turns"], "vetoed_user_turn": c.get("vetoed_user_turn"),
                             "rejected_reply": c.get("rejected_reply")})
    # self-validate against the v5.2 reader contract (same rules as data.read_source)
    for r in rows:
        t = r["turns"]
        assert len(t) >= 2 and len(t) % 2 == 0 and all(
            x["role"] == ("user", "assistant")[i % 2] and x["content"].strip() for i, x in enumerate(t))
    for name, items in (("chat.jsonl", rows), ("rejected.jsonl", rejected)):
        tmp = out / (name + ".tmp")
        tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
        os.replace(tmp, out / name)
    samples = [s for c in convs for s in c["samples"]]
    talkie_sec = sum(c["talkie_sec"] for c in convs)
    stats = {
        "generator": GENERATOR, "code_sha256": code_hash(), "model": meta["model"], "seed": meta["seed"],
        "conversations": len(convs), "status": dict(collections.Counter(c["status"] for c in convs)),
        "accepted": len(rows), "acceptance_rate": len(rows) / max(1, sum(c["status"] in ("done", "rejected") for c in convs)),
        "user_samples": len(samples), "user_sample_accept_rate": sum(not s["reason"] for s in samples) / max(1, len(samples)),
        "user_sample_reject_reasons": dict(collections.Counter(s["reason"] for s in samples if s["reason"]).most_common()),
        "conversation_reject_reasons": dict(collections.Counter(c["reason"] for c in convs if c["status"] == "rejected")),
        "talkie_seconds": round(talkie_sec, 1),
        "talkie_sec_per_accepted_row": round(talkie_sec / max(1, len(rows)), 2),
        "talkie_sec_per_sample": round(talkie_sec / max(1, len(samples)), 3),
        "turns_per_row": dict(collections.Counter(len(r["turns"]) // 2 for r in rows)),
        "ended_early": dict(collections.Counter(c["ended_early"] for c in convs
                                                if c["status"] == "done" and c.get("ended_early"))),
        "thing_dropped": sum(r["provenance"]["thing_dropped"] for r in rows),
        "archetypes_attempted": dict(collections.Counter(
            c["visitor"]["archetype"] for c in convs if c.get("visitor")).most_common()),
        "archetypes_accepted": dict(collections.Counter(
            r["provenance"]["visitor"]["archetype"] for r in rows if r["provenance"].get("visitor")).most_common()),
        "framings_accepted": dict(collections.Counter(
            r["provenance"]["visitor"]["framing"] for r in rows if r["provenance"].get("visitor")).most_common()),
        "archetype_cap_per_block": 1, "archetype_pool": len(VISITORS),
        "code_history": meta.get("code_history"),
        "id_range": [min((r["id"] for r in rows), default=None), max((r["id"] for r in rows), default=None)],
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["init", "advance", "pending", "finalize", "run"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5207)
    ap.add_argument("--id-base", type=int, default=ID_BASE)
    ap.add_argument("--mode", default="whimsy", choices=["whimsy", "spec_partner"],
                    help="talkie writes USER turns only (whimsy: 1930 visitor pool; spec_partner: "
                         "make_spec partners); fish turns come from --replies/--reply-command")
    ap.add_argument("--replies", type=Path, nargs="*", default=[])
    ap.add_argument("--reply-command", help="executable: JSON request on stdin -> reply on stdout")
    ap.add_argument("--reply-author", default="claude-opus-5-5")
    a = ap.parse_args(argv)
    if a.command == "pending":
        for item in pending(a.out):
            print(json.dumps(item, ensure_ascii=False))
        return
    if a.command == "finalize":
        print(json.dumps(finalize(a.out), indent=1))
        return
    if a.command in ("init", "run"):
        init(a.out, a.n, a.seed, a.id_base, a.mode)
        if a.command == "init":
            return
    backend = None
    if a.reply_command:
        backend = CommandBackend(a.reply_command, a.reply_author)
    elif a.replies:
        backend = FileBackend(a.replies, a.reply_author)
    advance(a.out, backend)
    if a.command == "run":
        print(json.dumps(finalize(a.out), indent=1))


if __name__ == "__main__":
    main()
