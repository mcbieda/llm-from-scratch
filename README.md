# llm-from-scratch
Building an LLM from scratch with my own mods and experiments  

## IMPORTANT  
This was my learning project - I went through "Build an LLM from scratch" by Raschka page by page.   
The great majority of the code here is from Raschka's book "Build an LLM from Scratch". Note that this code is the actual code from the published book; it is not the code from Raschka's github site.
My modifications and some additional code were from my direct coding, along with chatGPT and Gemini for coding. I did most planning.

## HOW TO: QUICKLY SEE WHAT I HAVE DONE AND SOME RESULTS  
1. The most obvious way: go to /notebooks in github, click on a notebook, and look at the writeup.  
	a. the notebooks are fully run notebooks with the outputs  
2. To look at my refactoring (which is less obvious)  
	a. look at the notebooks in /notebooks  
	b. look at the code in /src  
	

## HOW IS THIS DIFFERENT FROM THE RASCHKA REPO?  
I refactored this myself into the modules in /src/llm_from_scratch. So these are my choices. Also, the notebook and the main function in the notebook (run_training()) is of my design. The setup of the configuration parameters is my design. This was/is a learning project for me, which explains the evolution from really hacky and chapter based code to the refactored version.  

## HOW TO: UNDERSTAND AND MODIFY THE MODEL/RUN CONFIGURATION  
OPTION 1:  
1. just make a new notebook cell in the notebooks (early in the notebook!) that looks at "cfg".
(cfg is a configuration dictionary. Note that cfg['model_config'] is a subdictionary with model parameters)
 
OPTION 2:  
1. look at /src/llm_from_scratch/configs/gpt2small_config.py  
2. RUN_CONFIG is loaded into the notebook and sets model configuration and run configuration  

It is very easy to modify the config, as it is this simple dictionary.  

## RUNNING AND INSTALLING
This is very easy and all basic data is included, aside from a larger file of pubmed abstracts and also the OpenAI parameters for gpt2-small are not included. Raschka supplies complete information on getting these in his book. I supply the convenience function get_rachka_downloader.py to get these. Note that they must be moved into /data (do NOT use /src/data) for them to be recognized.

To get a basic model going:  
1. clone the repo into some new folder  
2. setup a virtual environment in python  
3. pip install -r requirements.txt  
    a. note that this does not install the acceleration package for intel, which I only used in early portions of this project  
4. run the notebook from /notebooks  
    a. with a GPU, this can run fairly fast (~50 seconds for experiment 1 in colab with a T4) - but on my i7 system using CPU, it is much slower (approx 10 minutes)  
    b. strong recommend to upload the code and data to google colab (free!) and use there  

## GOOGLE COLAB/DRIVE INSTRUCTIONS  
1. clone the repo into some new local folder  
2. DO NOT run pip install or any of that  
3. In google drive, make a new folder for this (like llmfirst)  
4. copy the directories into your new google drive folder  
5. you will probably need to install the colaboratory extension  
6. go to notebooks, open the notebook in colaboratory  
7. Set the runtime to use T4 (as of this writing in late January 2026, this is the only good option)  
8. Run the notebook!  
    a. again, note that this is much faster than a local CPU (at least my local CPU)  

## WHAT IS NOT INCLUDED  
This is basically up to the end of chapter 5 in the Raschka book - building the LLM.  

It does not include:  
1. code for chapter 6, which makes the LLM into a classifier  
2. code for chapter 7, which does instruction tuning  
3. use of other gpt2 models/architectures, like the 355M model  
4. probabilistic next-token selection  
I have implemented and run code for chapter 6 and 7, but not adjusted for this refactored code. Not currently on my todo list.  
For #3, it is pretty simple to change the model to be like a larger model - just change a few parameters in the configuration file (specifically, parameters in cfg['model_config'] - and the differences are listed in the Raschka book). It is also quite easy to get the openai weights for the pretrained version of this model.
For #4, I am currently interested in deterministic models to allow easy reproducibility and comparisons. Hence, no implementation.  

## NEXT STEPS  
1. **Pubmed abstract download and selection code at scale**: I have this code and will put it in a directory soon with some instructions. This is easy but there can be annoyances, which the code gets past.  
2. **Notebook for continued pre-training of a openai-pretrained gpt2-small**. I have done this but want to refine the notebook a bit. A fun part of this is using weight-tying for some additional intepretability and a somewhat smaller model.
   


