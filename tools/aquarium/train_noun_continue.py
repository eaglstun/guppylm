"""Small v5 continuation testing varied, explicitly copied human nouns.

This is an experiment. Probe nouns remain held out. Run from the project root:
  .venv/bin/python tools/aquarium/train_noun_continue.py --steps 400
"""
import argparse
import json
import os
import random
import torch
from tokenizers import Tokenizer

from train_aq import D, GuppyConfig, GuppyLM, encode_chat, load_dialogues, pad
from personas import make_spec

NOUNS = """a skateboard|a suitcase|a paintbrush|a paper clip|a fire extinguisher|
a telescope|a frying pan|a windmill|a raincoat|a seat belt|a snow globe|
a hiking boot|a coffee mug|a parking meter|a football helmet|a train station|
a washing machine|a camping stove|a sewing needle|a library card|a birthday invitation|
a flashlight|a water bottle|a traffic light|a fishing rod|a baseball glove|
a board game|a snow shovel|a door handle|a paper airplane|a lunch box|
a microphone|a washing line|a bottle opener|a paint bucket|a picnic blanket|
a skateboard ramp|a toy spaceship|a garden hose|a rolling pin|a stapler|
a walnut|a pumpkin|a snowflake|a pineapple|a seashell|a feather|a compass|
a bridge|a castle|a mountain|a thunderstorm|a rainbow|a sunflower|
the world cup|a science fair|a chess tournament|a dance contest|a magic trick|
a cooking lesson|a school play|a birthday party|a treasure hunt|a music festival|
sculpture|jazz music|astronomy|photography|a newspaper article|a recipe book""".replace("\n", "").split("|")

def examples(rng, variant):
    rows = []
    questions = ["Do you know what {thing} is?", "I just got {thing}!", "What do you think about {thing}?",
                 "Can you tell me about {thing}?", "Have you ever seen {thing}?"]
    replies = ["{thing}? I have not seen one here, but I would like to learn about it.",
               "You found {thing}? That sounds curious. Would it fit in my tide pool?",
               "I wonder if {thing} would float. Tell me more about it!",
               "I do not know much about {thing}, but I can listen while you explain.",
               "Maybe {thing} is something from your world. What do you use it for?"]
    if variant == "diverse":
        questions = [
            "My friend mentioned {thing} today. Have you heard of it?",
            "I found {thing} near the shore. Any ideas?",
            "Could {thing} survive under water?",
            "Someone left {thing} beside your tide pool.",
            "Imagine that {thing} drifted past your home.",
            "Please tell me a little story about {thing}.",
            "What would your friends think if they found {thing}?",
            "I am trying to understand {thing}. Can you help?",
            "How would you describe {thing} to another sea creature?",
            "Have the fish ever talked about {thing}?",
        ]
        replies = [
            "I have only heard of {thing} from humans. What is it like?",
            "If {thing} came ashore, I would approach slowly and look at it.",
            "I wonder whether {thing} could float beside my shell.",
            "That {thing} sounds curious. My friends might gather to see it.",
            "Once I found {thing} by the water, and I wondered where it came from.",
            "I would ask you to show me {thing} before guessing what it does.",
            "My fish friends might circle around {thing} to investigate.",
            "I do not understand {thing} yet. Can you show me how it works?",
            "I would call {thing} a strange visitor from your world.",
            "The fish have not told me about {thing}. Perhaps you could tell us.",
        ]
    for noun in NOUNS:
        for i in range(len(questions)):
            spec = make_spec(rng)
            rows.append({"system": spec["system"], "turns": [
                {"role": "user", "content": questions[i].format(thing=noun)},
                {"role": "assistant", "content": replies[(i + rng.randrange(len(replies))) % len(replies)].format(thing=noun)},
            ]})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="runs/aq-chat-v5-noun-cont")
    ap.add_argument("--init", default="runs/aq-chat-v5/best.pt")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--variant", choices=["standard", "diverse"], default="standard")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    cfg = GuppyConfig(**ck["config"])
    model = GuppyLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    tok = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
    base_rows = load_dialogues([os.path.join(D, "dialogues.jsonl")])
    copy_rows = examples(rng, a.variant)
    base = [encode_chat(tok, r, cfg.max_seq_len) for r in base_rows]
    copy = [encode_chat(tok, r, cfg.max_seq_len) for r in copy_rows]
    held_rows = load_dialogues([os.path.join(D, "dialogues_eval.jsonl")])
    held = [encode_chat(tok, r, cfg.max_seq_len) for r in held_rows]
    held_batches = [pad(held[i:i + 32], cfg.max_seq_len) for i in range(0, len(held), 32)]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    def eval_loss():
        model.eval()
        with torch.no_grad():
            val = sum(model(X.to(device), Y.to(device))[1].item() for X, Y in held_batches) / len(held_batches)
        model.train()
        return val
    print(json.dumps({"device": str(device), "base_rows": len(base), "new_rows": len(copy),
                      "held_rows": len(held), "step": 0, "held_loss": eval_loss()}), flush=True)
    for step in range(1, a.steps + 1):
        # One quarter focused copy examples, three quarters ordinary dialogue.
        X, Y = pad([rng.choice(copy) for _ in range(8)] + [rng.choice(base) for _ in range(24)], cfg.max_seq_len)
        loss = model(X.to(device), Y.to(device))[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == a.steps:
            print(json.dumps({"step": step, "train_loss": round(loss.item(), 4),
                              "held_loss": eval_loss()}), flush=True)
    path = os.path.join(a.out, "final.pt")
    torch.save({"model": model.cpu().state_dict(), "config": ck["config"], "step": a.steps}, path)
    print(path, flush=True)

if __name__ == "__main__":
    main()
