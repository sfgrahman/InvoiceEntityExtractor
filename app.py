import streamlit as st
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

def main():
    st.title("PDF Entity Extractor:books:")

    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()

        st.write(f"Number of pages: {len(pages)}")

        # for page in pages:
        #     st.write(page.page_content)
        #     st.write("")
        st.write(pages[0].page_content)
        llm = OpenAI(temperature=0.1, max_tokens=1024)
        #llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",config={'max_new_tokens':128,'temperature':0.01})
        
        template = """Create a six-columns table with the Course Name, Number(beside course), Title, Attempted, Earned, Grade for each line of text below. No header.
          {pages}
        use this format:
        Course | Number | Title | Attempted | Earned | Grade
        """
        prompt_template = PromptTemplate(input_variables=["pages"], template=template)
        chain = LLMChain(llm=llm, prompt=prompt_template)

        result = chain.run(pages=pages[0].page_content)
        
        st.write("Extracted entities:")
        entities = result.strip().split("\n")
        table_data = [line.split("|") for line in entities]
        print(entities)
        st.table(table_data)

if __name__ == "__main__":
    main()


