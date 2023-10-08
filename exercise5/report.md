# Report
Name: Gerald Hirsch

Assignment: No. 5

# Workflow and Installation
1. create and verify HuggingFace account
2. Research for transformer models
3. Read through requirements
4. create personal token
5. add taken to device
6. request access to model weights (not necessary for the one sin the script)
7. specify model(s) of interest: GPT2, and Llama2
8. create main.py
9. Install "torch" and "transformers" module
10. Download GPT2 and Llama2 model weights and tokenizer
11. Write basic script to have GPT2 and Llama2 chatbot

# Deployment

The deployment is relatively straight forward and basic use is documented on the hugging face website. After the model weights are downloaded and chached locally, both the tokenizer and model can be initialized. Both models utilize the torch library for their underlying implementation, which allows the usage of cuda. This highly increase the parallel computing speed. The following three steps were necessary for both transformers:

1. initialize the model and tokenizer
2. fetch and encode the input via tokenizer
3. generate the output via the model

# Challenges

Besides the download time (Llama2 size 27GB), there were no major challenges to overcome. However, despite choosing the smallest Llama2 model with 7Bn parameters (max 70Bn). This already shows the shear complexity and size of state-of-the-art large language models. Any adaption, fine tuning will just not be possible on my personal hardware. For example, training the Llama2 7Bn modell requires 184k hours of GPU time (A100-8GB).