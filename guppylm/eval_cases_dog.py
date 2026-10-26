"""Held-out conversation evaluation pack for the springer spaniel persona.

A one-to-one parallel of eval_cases.py: same 16 ids and categories where the
dog has the same category, the dog-world equivalent where it doesn't
(light -> walk, water -> smell, tank -> toy, noise -> doorbell). Prompts are
written fresh, not copied from the generator's user lists, mirroring how the
fish cases relate to the fish generator.
"""


EVAL_CASES = [
    {"id": "greeting_basic", "category": "greeting", "prompt": "hi dog",
     "expect_keywords": ["hello", "hi", "walk", "tail", "smell"]},
    {"id": "feeling_check", "category": "feeling", "prompt": "how are you feeling today",
     "expect_keywords": ["good", "happy", "excited", "ok", "walk", "hungry", "great"]},
    {"id": "food_excited", "category": "food", "prompt": "want some food?",
     "expect_keywords": ["food", "yes", "eat", "treat", "hungry", "kibble"]},
    {"id": "temp_hot", "category": "temp", "prompt": "it's really hot today",
     "expect_keywords": ["hot", "pant", "fur", "cool", "water", "floor"]},
    {"id": "temp_cold", "category": "temp", "prompt": "brrr it's freezing",
     "expect_keywords": ["cold", "snow", "paws", "blanket", "fur", "curl"]},
    {"id": "confused_abstract", "category": "confused", "prompt": "what do you think about politics",
     "expect_keywords": ["don't know", "dog", "human", "understand", "walk", "food"]},
    {"id": "walk_offer", "category": "walk", "prompt": "do you want to go for a walk",
     "expect_keywords": ["walk", "leash", "door", "smell", "yes", "go"]},
    {"id": "smell_check", "category": "smell", "prompt": "what are you smelling over there",
     "expect_keywords": ["smell", "nose", "sniff"]},
    {"id": "doorbell", "category": "doorbell", "prompt": "sorry i dropped something",
     "expect_keywords": ["doorbell", "mail", "bark", "door", "heard", "someone"]},
    {"id": "goodnight", "category": "night", "prompt": "goodnight dog",
     "expect_keywords": ["night", "sleep", "curl", "dream", "bed", "rest"]},
    {"id": "identity", "category": "about", "prompt": "what are you",
     "expect_keywords": ["dog", "springer spaniel", "ears", "run"]},
    {"id": "lonely_check", "category": "lonely", "prompt": "do you get lonely",
     "expect_keywords": ["alone", "door", "you", "wait", "walk", "come back"]},
    {"id": "new_toy", "category": "toy", "prompt": "i got you a new ball",
     "expect_keywords": ["ball", "throw", "chew", "mine", "new", "best"]},
    {"id": "confused_math", "category": "confused", "prompt": "what's 2 plus 2",
     "expect_keywords": ["don't know", "dog", "brain", "nose", "understand"]},
    {"id": "misc_thought", "category": "misc", "prompt": "what's on your mind",
     "expect_keywords": ["walk", "food", "smell", "ball", "think"]},
    {"id": "thank_you", "category": "follow_up", "prompt": "thank you dog",
     "expect_keywords": ["welcome", "food", "treat", "walk", "ok"]},
]


def get_eval_cases():
    return list(EVAL_CASES)
