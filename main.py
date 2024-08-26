from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler


chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    callbacks=[StreamingStdOutCallbackHandler()],
)

poetry_chain_for_language_propmt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a world-class poet, and you write poetry about programming languages.",
        ),
        ("human", "Write a poem about {language}"),
    ]
)

poetry_explain_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The user will type a poem. Recite the poem verbatim, then interpret and explain the poem. Enclose the poem in quotation marks.",
        ),
        ("human", "{poem}"),
    ]
)

poetry_chain = poetry_chain_for_language_propmt | chat

poetry_explain_chain = poetry_explain_prompt | chat

final_chain = {"poem": poetry_chain} | poetry_explain_chain  # type: ignore

result = final_chain.invoke(input={"language": "python"})

print(result)
