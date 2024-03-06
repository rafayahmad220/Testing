from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
from langchain import PromptTemplate,  LLMChain

model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})
template = """
You are an intelligent chatbot. Help the following question with answers.
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Explain what is Artificial Intellience "

print(llm_chain.run(question))

"""AI is like the sun that shines,
Bringing happiness and knowledge to everyone.
It helps us in many ways,
From finding answers to complex questions.
AI can do amazing things,
That we can't even imagine yet!
AI is like a powerful tool,
That can help us solve any riddle.
It can even predict the future,
Based on current situations and trends.
AI is here to stay,
And will be an essential element in our lives,
Providing a better tomorrow
and an even brighter future!
"""
