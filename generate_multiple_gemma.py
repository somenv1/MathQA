import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
import bitsandbytes as bnb
import pandas as pd
import random
import numpy as np
import transformers
import random
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from huggingface_hub import login

with open('hg.token','r') as ft:
	token = ft.read().strip()
login(token=token) #Log into huggingface

model_path = "google/gemma-2-2b-it"   # Specify the path to the model
adapter_path = "mathwell/checkpoint-4250"   # Specify the path to the adapter weights

tokenizer_default = AutoTokenizer.from_pretrained(model_path) # Load tokenizer
tokenizer_new = AutoTokenizer.from_pretrained(adapter_path) # Load tokenizer

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
) # Set up bitsandbytes config to load model in 4 bit

generated_questions = np.array([])

topics = ['Superman', "Batman", "Wonder Woman", "Barbie", "Power Rangers", "basketball", "soccer", "football", "volleyball", 'field hockey',\
'Fortnite', 'Spiderman', "Iron Man", "Captain America", "Captain Marvel", "Thor, the God of Thunder", "Black Panther", "Taylor Swift", "swimming",\
"Pokémon", "Super Mario", "Naruto", "unicorns", "Hello Kitty", "Minecraft", "lacrosse", "cheer leading", "LeBron James", "Steph Curry", "Patrick Mahomes",\
"Serena Williams", "dogs", "cats", "dinosaurs", "Harry Potter", "cars", "planes", "trains", "pizza", "cookies", "ice cream", 'candy',\
    "Frozen (Elsa and Anna)",
    "Minecraft",
    "Star Wars",
    "Paw Patrol",
    "My Little Pony",
    "Minions",
    "Jurassic Park",
    "SpongeBob SquarePants",
    "Disney Princesses",
    "Toy Story",
    "The Incredibles",
    "Scooby-Doo",
    "Peppa Pig",
    "Dora the Explorer",
    "Pikachu",
    "Thomas the Tank Engine",
    "Sonic the Hedgehog",
    "Transformers",
    "Minions",
    "Cinderella",
    "Moana",
    "Shrek",
    "Winnie the Pooh",
    "Tom and Jerry",
    "Sesame Street",
    "The Lion King",
    "Alice in Wonderland",
    "The Little Mermaid",
    "Peter Pan",
    "Aladdin",
    "The Jungle Book",
    "Pocahontas",
    "Beauty and the Beast",
    "Frozen",
    "Ratatouille",
    "Finding Nemo",
    "Cars",
    "Up",
    "The Simpsons",
    "Looney Tunes",
    "Teenage Mutant Ninja Turtles",
    "Scooby-Doo",
    "Mythical Creatures (dragons, unicorns)",
    "Dinosaurs",
    "Space and Astronauts",
    "Robots",
    "Aliens",
    "Exploring the Ocean",
    "Underwater Creatures",
    "Pirates",
    "Fairies",
    "Wizards",
    "Magic Tricks",
    "Time Travel",
    "Detectives and Mystery",
    "Inventions",
    "The Avengers",
    "The Justice League",
    "Dance and Ballet",
    "Music Instruments",
    "Art and Drawing",
    "Science Experiments",
    "Cooking and Baking",
    "DIY Crafts",
    "Board Games",
    "Puzzles",
    "Riddles",
    "Pets (cats, dogs, hamsters)",
    "Farm Animals",
    "Zoo Animals",
    "Wildlife Conservation",
    "Plants and Gardening",
    "Hiking and Nature",
    "Weather and Meteorology",
    "The Solar System",
    "Camping",
    "National Parks",
    "Trains and Railroads",
    "Planes and Aviation",
    "Cars and Racing",
    "Construction Vehicles",
    "Firefighters",
    "Police Officers",
    "Doctors and Nurses",
    "Astronauts and Space Exploration",
    "Animals and Wildlife",
    "Space and Astronomy",
    "Robots and Technology",
    "Underwater Life",
    "Fairy Tales and Folklore",
    "Science Experiments",
    "Outer Space",
    "Weather and Meteorology",
    "Art and Drawing",
    "Music and Instruments",
    "Cooking and Baking",
    "DIY Crafts",
    "Board Games",
    "Puzzles",
    "Riddles",
    "Pets (cats, dogs, hamsters)",
    "Farm Animals",
    "Zoo Animals",
    "Wildlife Conservation",
    "Plants and Gardening",
    "Hiking and Nature",
    "Weather and Meteorology",
    "The Solar System",
    "Camping",
    "National Parks",
    "Trains and Railroads",
    "Planes and Aviation",
    "Cars and Racing",
    "Construction Vehicles",
    "Firefighters",
    "Police Officers",
    "Doctors and Nurses",
    "Astronauts and Space Exploration",
    "Animals and Wildlife",
    "Space and Astronomy",
    "Robots and Technology",
    "Underwater Life",
    "Fairy Tales and Folklore",
    "Science Experiments",
    "Outer Space",
    "Weather and Meteorology",
    "Art and Drawing",
    "Music and Instruments",
    "Cooking and Baking",
    "Insects and Bugs",
    "Historical Figures",
    "Countries and Cultures",
    "Mythical Creatures",
    "Magic and Wizards",
    "Friendship and Relationships",
    "Ocean Life",
    "Cars and Vehicles",
    "Famous Inventors",
    "Famous Artists",
    "Ancient Civilizations",
    "Space Exploration",
    "DIY Crafts",
    "Gardening",
    "Environmental Conservation",
    "Time Travel",
    "Pirates and Treasure",
    "Famous Scientists",
    "Computer Programming",
    "Unexplained Mysteries",
    "Planets and the Solar System",
    "Cartoons and Animated Shows",
    "Photography",
    "National Parks",
    "Dance and Ballet",
    "Board Games",
    "Books and Reading",
    "Volcanoes",
    "Mythology",
    "Ancient Egypt",
    "Reptiles and Amphibians",
    "Recycling",
    "Fairy Gardens",
    "Indoor Games",
    "Marine Biology",
    "Virtual Reality",
    "Natural Disasters",
    "Construction and Building",
    "Inventions",
    "the Circus and Performing Arts",
    "Science Fiction",
    "Pottery and Ceramics",
    "Famous Explorers",
    "Birds and Bird Watching",
    "Famous Landmarks",
    "Health and Nutrition",
    "Myths and Legends",
    "Fashion and Clothing",
    "DIY Science Projects",
    "Cultural Festivals",
    "Construction Vehicles",
    "Forests and Trees",
    "Mummies",
    "Famous Composers",
    "Circus Animals",
    "Geology",
    "Farm Life",
    "Travel and Adventure",
    "Ballet and Dance",
    "Native American Culture",
    "Whales and Dolphins",
    "Mystery Stories",
    "Hiking and Camping",
    "Games and Puzzles",
    "Space Aliens and UFOs"
]


def generate(tokenizer,model):
	#prompt = " Concreteness is defined as a word's total amount of tangible words. High concreteness means a sentence has sufficient real objects and well-defined words that make a meaningful sentence with a high number of specifics. Low concreteness means a sentence has a large amount of abstract words and undefined objects that still make a meaningful sentence but lack a clear story. Use the given definition of concreteness and generate questions with high, medium and low concreteness based on the following information, 200 cats are in a cat park. 100 more cats show up at the cat park. 25% of the cats at the cat park are meowing. How many cats are meowing?  " #TODO
	topic = random.choice(topics) # Select a random topic
	prompt = f"Write a grade school math word problem about {topic} and Python function with a commented out step-by-step solution to solve the word problem."
	#Query the model 
	inputs = tokenizer.encode(prompt, return_tensors="pt")
	attention_mask = torch.ones_like(inputs)
	attention_mask = attention_mask.to('cuda') #Line added
	inputs = inputs.to('cuda')
	output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 512, do_sample = True)
	#output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 250, do_sample = False)
	generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
	return generated_text
	#with open("questions.txt",'a') as outfile:
	#with open(f"questions{index}.txt",'a') as outfile:
	#	outfile.write(generated_text+"\n")



model_default = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto",
    torch_dtype=torch.bfloat16, use_auth_token=True) # Load model in 4 bit

model_new = PeftModel.from_pretrained(model_default, adapter_path) # Create PEFT model 

print("###################TRAINED MODEL###################")
#print(model_new.peft_config)
#for name, param in model_new.named_parameters():
#	print(name)
for i in range (1000):
	generated_questions = np.append(generated_questions,generate(tokenizer_new,model_new))
#print(generated_questions)
np.save("questions_arr.npy",generated_questions)
