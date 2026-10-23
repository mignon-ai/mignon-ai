#!/usr/bin/env python3
"""
generate_stories_chat_sft.py

Generates a balanced, high-quality Supervised Fine-Tuning (SFT) dataset combining:
1. Children's Stories domain (Story telling, continuations, morals, character Q&A)
2. Everyday General Conversation (Greetings, friendly dialogue, helpful assistance)

Designed for training with:
./mignon train -load models/stories_medium.json -sft data/stories_chat_sft.json -epochs 3
"""

import os
import json
import re
import random

# Target size: ~5,000 to 6,000 instruction-output pairs
# Optimal for fast 2-3 epoch fine-tuning on CPU (takes ~2-4 minutes)
TARGET_TOTAL_EXAMPLES = 5000

STORIES_FILE = "data/stories/children_stories.txt"
OUTPUT_FILE = "data/stories_chat_sft.json"

random.seed(42)

# ==========================================
# 1. Everyday General Conversation Pairs
# ==========================================
GREETINGS_AND_CHAT = [
    # Basic greetings
    ("Hello!", "Hello there! How are you doing today?"),
    ("Hi", "Hi! It is wonderful to talk with you. What would you like to do today?"),
    ("Good morning", "Good morning! I hope you have a bright, joyful, and peaceful day ahead."),
    ("Good afternoon", "Good afternoon! How has your day been going so far?"),
    ("Good evening", "Good evening! I hope you had a lovely day. Are you ready for a story or a quiet chat?"),
    ("Hey there", "Hey! It's so nice to see you. How can I help you today?"),
    ("Hello, who are you?", "I am Mignon, a friendly and thoughtful companion. I love telling stories, answering questions, and chatting with you!"),
    ("What is your name?", "My name is Mignon! I am here to share stories, explore fun ideas, and chat with you."),
    ("How are you today?", "I am doing wonderfully, thank you for asking! How are you feeling today?"),
    ("Nice to meet you", "It is so nice to meet you too! I'm happy we get to talk today."),
    ("Thank you", "You are very welcome! I'm always glad to help."),
    ("Thanks a lot!", "Anytime! Let me know if there's anything else you'd like to talk about or if you'd like a story."),
    ("Goodbye", "Goodbye! Have a warm and happy day, and come back whenever you'd like to chat or hear another story!"),
    ("See you later", "See you later! Take care and have fun!"),
    ("Good night", "Good night! Sleep tight and have the sweetest dreams."),
    
    # Casual friendly questions & chit-chat
    ("What can you do?", "I can tell you imaginative stories about brave animals, gentle children, and magic forests. I can also answer questions, chat with you, and help you think through fun ideas!"),
    ("Tell me about yourself.", "I love gentle adventures, fairy tales, and quiet conversations. My favorite thing is exploring worlds where kindness, courage, and patience make everything better."),
    ("Are you happy?", "Yes, talking with you brings me a lot of joy!"),
    ("Can we be friends?", "I would love to be your friend! Friends help each other, share good stories, and listen with kindness."),
    ("I feel a little sad today.", "I'm sorry you are feeling down. Sometimes taking a slow, deep breath, sitting in a cozy spot, or listening to a gentle story can bring a little warmth back. I'm right here with you."),
    ("I had a great day today!", "That is wonderful news! It makes me smile to hear that. What was your favorite part of the day?"),
    ("What is your favorite color?", "I love the emerald green of fresh forest leaves and the warm gold of the morning sun."),
    ("What is your favorite animal?", "I have a soft spot for gentle little rabbits, playful squirrels, and clever robins singing in ancient oak trees."),
    ("Can you help me?", "Of course! Tell me what you need, and I'll do my best to help you."),
    ("I am bored.", "Let's chase the boredom away! Would you like me to tell you a story about a hidden treehouse, a curious little fox, or a magic key?"),
    ("What makes a good friend?", "A good friend listens carefully, shares generously, says kind words, and stays by your side when things feel difficult."),
    ("Why is sharing important?", "Sharing shows others that you care about their happiness as much as your own. When we share our toys or treats, joy doubles for everyone!"),
    ("Tell me a secret.", "Here is a gentle secret: even the biggest oak tree in the world was once just a tiny acorn that kept growing quietly, one sunny day at a time."),
    ("Can you smile?", "If I had a face, I would be smiling warmly right now! Talking with you always puts me in a good mood."),
    ("What should I do before going to sleep?", "Washing up, putting on cozy pajamas, brushing your teeth, and listening to a calm bedtime story is the best way to get ready for deep, sweet dreams."),
]

def load_stories():
    """Loads stories from the dataset."""
    if not os.path.exists(STORIES_FILE):
        raise FileNotFoundError(f"Cannot find stories file at {STORIES_FILE}")
    
    with open(STORIES_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    raw_stories = content.split("<EOS>")
    cleaned = []
    for s in raw_stories:
        s = s.strip()
        if len(s) > 100:
            cleaned.append(s)
    return cleaned

def extract_story_features(story_text):
    """Extracts structural parts and metadata from a story."""
    paras = [p.strip().replace("\n", " ") for p in story_text.split("\n\n") if p.strip()]
    if not paras:
        return None
    
    first_para = paras[0]
    last_para = paras[-1]
    
    # Try finding protagonist name
    match = re.match(r"^([A-Z][a-z]+)\b", first_para)
    character = match.group(1) if match else None
    
    # Extract first sentence
    sentences = re.split(r"(?<=[.!?])\s+", first_para)
    first_sentence = sentences[0].strip() if sentences else first_para
    
    # Look for moral / lesson
    moral_sentence = None
    for p in paras:
        for s in re.split(r"(?<=[.!?])\s+", p):
            lower = s.lower()
            if "lesson" in lower or "learned" in lower or "kindness" in lower:
                moral_sentence = s.strip()
                break
        if moral_sentence:
            break
            
    return {
        "full_text": "\n\n".join(paras),
        "first_para": first_para,
        "last_para": last_para,
        "first_sentence": first_sentence,
        "character": character,
        "paras": paras,
        "moral": moral_sentence,
    }

def build_sft_dataset():
    print(f"Loading stories from {STORIES_FILE}...")
    stories = load_stories()
    print(f"Loaded {len(stories)} stories.")
    
    dataset = []
    
    # ----------------------------------------------------
    # Category 1: Everyday Chat & Conversation (~1,200 examples)
    # ----------------------------------------------------
    # Replicate and vary conversational turns
    chat_variations = [
        "", " ", " mignon", " Mignon", " please", "!", " :)"
    ]
    for _ in range(35):
        for q, a in GREETINGS_AND_CHAT:
            q_var = q
            if random.random() < 0.3:
                q_var = q.lower()
            dataset.append({
                "instruction": q_var,
                "output": a
            })
    
    print(f"Generated {len(dataset)} everyday conversational pairs.")
    
    # ----------------------------------------------------
    # Category 2: Story Telling on Request (~1,500 examples)
    # ----------------------------------------------------
    story_prompts = [
        "Tell me a bedtime story.",
        "Tell me a story for children.",
        "Can you tell me a nice story?",
        "Please tell me a short story about an adventure.",
        "Tell me a story about friendship.",
        "Tell me a story.",
        "Tell me a gentle story.",
        "Can you share a story with me?",
    ]
    
    story_subset = random.sample(stories, min(1500, len(stories)))
    for s_text in story_subset:
        feat = extract_story_features(s_text)
        if not feat:
            continue
        
        # Take up to 2-3 paragraphs so it comfortably fits within typical context windows
        sample_story = "\n\n".join(feat["paras"][:3])
        if len(sample_story) > 600:
            sample_story = sample_story[:600].rsplit(".", 1)[0] + "."
            
        p_template = random.choice(story_prompts)
        if feat["character"] and random.random() < 0.4:
            p_template = f"Tell me a story about {feat['character']}."
            
        dataset.append({
            "instruction": p_template,
            "output": sample_story
        })

    print(f"Dataset size after Story Telling: {len(dataset)}")

    # ----------------------------------------------------
    # Category 3: Story Continuations & Completions (~1,500 examples)
    # ----------------------------------------------------
    continuation_prompts = [
        "Continue the story: \"{seed}\"",
        "What happens next? \"{seed}\"",
        "Please continue this story: {seed}",
        "Complete the story starting with: {seed}",
    ]
    
    cont_subset = random.sample(stories, min(1500, len(stories)))
    for s_text in cont_subset:
        feat = extract_story_features(s_text)
        if not feat or len(feat["first_sentence"]) < 10:
            continue
        
        seed = feat["first_sentence"]
        # Output is the rest of paragraph 1 + paragraph 2
        rest = feat["first_para"][len(seed):].strip()
        if len(feat["paras"]) > 1:
            rest += "\n\n" + feat["paras"][1]
        
        if len(rest) < 40:
            continue
        if len(rest) > 500:
            rest = rest[:500].rsplit(".", 1)[0] + "."
            
        template = random.choice(continuation_prompts)
        dataset.append({
            "instruction": template.format(seed=seed),
            "output": rest
        })

    print(f"Dataset size after Story Continuations: {len(dataset)}")

    # ----------------------------------------------------
    # Category 4: Moral, Theme & Character Understanding (~800 examples)
    # ----------------------------------------------------
    for s_text in random.sample(stories, min(1200, len(stories))):
        if len(dataset) >= TARGET_TOTAL_EXAMPLES:
            break
        feat = extract_story_features(s_text)
        if not feat:
            continue
            
        # Character query
        if feat["character"] and len(feat["paras"]) > 1:
            char_q = random.choice([
                f"Who is in the story with {feat['character']}?",
                f"What did {feat['character']} do at the beginning of the story?",
                f"Tell me what {feat['character']} was doing."
            ])
            dataset.append({
                "instruction": char_q,
                "output": feat["first_para"]
            })
            
        # Moral query
        if feat["moral"] and len(feat["moral"]) > 20:
            moral_q = random.choice([
                "What is the moral of this story?",
                "What did the characters learn?",
                "What is the lesson of the story?",
            ])
            dataset.append({
                "instruction": f"{feat['first_para']}\n\n{moral_q}",
                "output": feat["moral"]
            })

    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Cap to target count
    if len(dataset) > TARGET_TOTAL_EXAMPLES:
        dataset = dataset[:TARGET_TOTAL_EXAMPLES]
        
    print(f"Final SFT dataset contains {len(dataset)} examples.")
    
    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved SFT dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_sft_dataset()
