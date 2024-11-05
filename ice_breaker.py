from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()

    print("Hello LangChain")

    summary_template = """
    given the linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # llm = ChatOllama(
    #     model="llama3.2",
    #     temperature=0,
    # )

    # The | pipe is a LangChain Expression Language (LCEL) element and used to describe chains
    chain = summary_prompt_template | llm
    linkedin_data = scrape_linkedin_profile("https://www.linkedin.com/in/eden-marco/", mock=True)
    res = chain.invoke(input={"information": linkedin_data})

    print(res)
