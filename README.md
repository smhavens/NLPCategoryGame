---
title: AnalogyArcade
emoji: 🏆
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


## Introduction

## Usage

## Documentation

## Experiments
### Model Types
#### Baseline
For my dataset, I made use of  relbert/analogy_questions on huggingface, which has all data in the format of:
```
"stem": ["raphael", "painter"],
"answer": 2,
"choice": [["andersen", "plato"],
          ["reading", "berkshire"],
          ["marx", "philosopher"],
          ["tolstoi", "edison"]]
```

For a baseline, if I were to do a random selection for answer to train the system on (so the stem analogy is compared to a random choice among the answers), then there would only be a 25% baseline for correct categorization and comparison.

#### Bag-of-Words Model
For comparison, I made use of my previously trained bag-of-words model from [our previous project](https://github.com/smhavens/NLPHW03). I changed this model to focus entierely on the most_similar function of word to vec.

#### Fine-Tuning
##### Dataset
[analogy questions dataset](https://huggingface.co/datasets/relbert/analogy_questions)

This database uses a text with label format, with each label being an integer between 0 and 3, relating to the 4 main categories of the news:  World (0), Sports (1), Business (2), Sci/Tech (3).

I chose this one because of the larger variety of categories compared to sentiment databases, with the themes/categories theoretically being more closely related to analogies. I also chose ag_news because, as a news source, it should avoid slang and other potential hiccups that databases using tweets or general reviews will have.

##### Pre-trained model
[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Because my focus is on using embeddings to evaluate analogies for the AnalogyArcade, I focused my model search for those in the sentence-transformers category, as they are readily made for embedding usage. I chose all-MiniLM-L6-v2 because of its high usage and good reviews: it is a well trained model but smaller and more efficient than its previous version.

#### In-Context
My in-context model used [Google's flan-t5-base](https://huggingface.co/google/flan-t5-base#usage) along with the database [focused on pair analogies](https://huggingface.co/datasets/tomasmcz/word2vec_analogy).

### Human Analogy Judgement
#### Sentence-Transformer Model
	1. humans is to fired as exam is to recycling
	2. accessible is to trigger as bicycle is to federation
	3. sipped is to recover as able is to alternate
	4. renewal is to evidenced as travel is to curve
	5. cot is to center as dissent is to fan
	6. endure is to mine as accessible is to within
	7. pierced is to royal as tissue is to insisted
	8. amor is to period as maneuvers is to chorus
	9. city is to elimination as tod is to offset
	10. arabian is to poem as pluto is to embassy


#### In-Context Model using Flan
	1. ciones is to Mexican as Assisted is to Assistance
	2. Stu is to Stubble as île is to Laos
	3. wusste is to unknown as nature is to nature
	4. AIDS is to Mauritania as essentiellement is to Mauritania
	5. something is to Something as OC is to Oman
	6. abend is to abend as Lie is to lie
	7. Muzeul is to Romania as hora is to Mauritania
	8. BB is to Belorussian as werk is to German
	9. éti is to étizstan as ţele is to celswana
	10. î is to îzbekistan as derma is to Dermatan


#### Word2Vec Model
	1. headless is to hobo as thylacines is to arawaks
	2. 42 km is to dalston as sisler is to paperboy
	3. tolna is to fejér as dermott is to ꯄꯟꯊꯢꯕ.
	4. recursion is to postscript as ornithischian is to ceratopsian.
	5. 19812007 is to appl as khimichev is to pashkevich.
	6. trier is to nürnberg as hathaways is to tate.
	7. yon is to arnoldo as neocon is to pešice.
	8. washingtonpost is to secretaria as laugh is to shout.
	9. waking is to wakes as prelude is to fugue.
          10. 2car is to shunters as mariah is to demi.


### Automated Tests
#### Sentence-Transformer Model
	1. PROMPT: Sun is to Moon as Black is to [MASK]. ANSWER: exhausted IS TRUE: False
	2. PROMPT: Black is to White as Sun is to [MASK]. ANSWER: bulls IS TRUE: False
	3. PROMPT: Atom is to Element as Molecule is to [MASK]. ANSWER: gig IS TRUE: False
	4. PROMPT: Raphael is to painter as Marx is to [MASK]. ANSWER: decade IS TRUE: False
	5. PROMPT: huge is to hugely as subsequent is to [MASK]. ANSWER: organised IS TRUE: False
	6. PROMPT: simple is to difficult as fat is to [MASK]. ANSWER: dick IS TRUE: False
	7. PROMPT: poem is to stanza as staircase is to [MASK]. ANSWER: cooking IS TRUE: False
	8. PROMPT: academia is to college as typewriter is to [MASK]. ANSWER: folder IS TRUE: False
	9. PROMPT: acquire is to reacquire as examine is to [MASK]. ANSWER: futures IS TRUE: False
	10. PROMPT: pastry is to food as blender is to [MASK]. ANSWER: casting IS TRUE: False
	11. PROMPT: Athens is to Greece as Tokyo is to [MASK]. ANSWER: homeless IS TRUE: False

0/11, failed its own dataset prompts


#### In-Context Model using Flan
	1. PROMPT: Sun is to Moon as Black is to ___. ANSWER: Blacks IS TRUE: False
	2. PROMPT: Black is to White as Sun is to ___. ANSWER: Sunlighter IS TRUE: False
	3. PROMPT: Atom is to Element as Molecule is to ___. ANSWER: Molovnia IS TRUE: False
	4. PROMPT: Raphael is to painter as Marx is to ___. ANSWER: Marxistan IS TRUE: False
	5. PROMPT: huge is to hugely as subsequent is to ___. ANSWER: apparently IS TRUE: False
	6. PROMPT: simple is to difficult as fat is to ___. ANSWER: fatter IS TRUE: False
	7. PROMPT: poem is to stanza as staircase is to ___. ANSWER: staircase IS TRUE: False
	8. PROMPT: academia is to college as typewriter is to ___. ANSWER: typewriters IS TRUE: False
	9. PROMPT: acquire is to reacquire as examine is to ___. ANSWER: examine IS TRUE: False
	10. PROMPT: pastry is to food as blender is to ___. ANSWER: blenders IS TRUE: False
	11. PROMPT: Athens is to Greece as Tokyo is to ___. ANSWER: Japan IS TRUE: True

1/11, only success on their own dataset. Wants to repeat the 3rd word

### Word2Vec Model
	1. PROMPT: sun is to moon as black is to ___ ANSWER: white IS TRUE: True
	2. PROMPT: black is to white as sun is to ___ ANSWER: moon IS TRUE: True
	3. PROMPT: atom is to element as molecule is to ___ ANSWER: nucleus IS TRUE: False
	4. PROMPT: raphael is to painter as marx is to ___ ANSWER: beck IS TRUE: False
	5. PROMPT: huge is to hugely as subsequent is to ___ ANSWER: massive IS TRUE: False
	6. PROMPT: simple is to difficult as fat is to ___ ANSWER: slice IS TRUE: False
	7. PROMPT: poem is to stanza as staircase is to ___ ANSWER: balcony IS TRUE: False
	8. PROMPT: academia is to college as typewriter is to ___ ANSWER: blowfish IS TRUE: False
	9. Reequire not present in vocab
	10. PROMPT: pastry is to food as blender is to ___ ANSWER: plunger IS TRUE: False
	11. PROMPT: athens is to greece as tokyo is to ___ ANSWER: osaka IS TRUE: False
	
2/11, very easy for humans original prompts it succeeded in. Didn't grasp analogies and often gave a word very similar to the 3rd or first pair. Only one to miss a vocab word


## Limitations
One of the biggest limitations is the extreme difficulty of LMs to understand analogies without extensive training (that I cannot do within my means in a reasonable time) or to match more specific prompts and wording. Another difficutly is that, as the words chosen are all from the vocab of the models/dataset, it will either miss reasonable words (such as with the Word2Vec model not understanding 'Reequire') or focus on very niche or foreign words as seen in some of the non-automated testing.
