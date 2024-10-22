import os
import json
import langchain
import requests
from langchain_community.cache import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage



parser = StrOutputParser()

item = {
    'instruction': 'you are a AI assistant',
    'input':'hello, who are you'
}


model = ChatOpenAI(
    api_key='666',
    base_url="http://localhost:8001/v1",
    temperature=0.9,
    max_tokens=128,
    top_p=1,
    frequency_penalty=0.0,
    stop="<|eot_id|>"
)

messages = [
    SystemMessage(content=item['instruction']),
    HumanMessage(content=item['input']),
]

chain = model | parser
pred = chain.invoke(messages)
print(pred)
    # json_content = instruction+":"+input_content
    # # small model provide knowledge
    # response = requests.post(
    #     "http://localhost:8001/v1/chat/completions",
    #     # json={
    #     #     "model": "qwen",
    #     #     "messages": [
    #     #         {
    #     #         "role": "user",
    #     #         "content": input_content
    #     #         }
    #     #     ],
    #     #     "do_sample": "true",
    #     #     "temperature": 0,
    #     #     "stream": "false"
    #     #     }
    # )

    # data = json.loads(response.text)
    # # kit_output = data['choices'][0]['message']['content']
    # print(print("\nqwen输出：\n\n"))
    # print(data)
