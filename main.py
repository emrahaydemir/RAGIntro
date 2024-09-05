from dotenv import load_dotenv
from langchain import hub
import bs4
from langchain_chroma import Chroma  # vector veritabani yonetimi //vector store
from langchain_community.document_loaders import (
    WebBaseLoader,
)  # internetten verileri cekmek icin
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough  # promptlari genisletmek icin
from langchain_openai import OpenAIEmbeddings  # verileri vectorize etmek icin
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)  # internetten indirilen verilerin bolunmesi icin
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo"
)  # llm i chatgptyi kullanarak olustur. model gpt-3.5

loader = WebBaseLoader(  # internetten verileri cekmek icin loader kurulumu
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(  # internetten indirilen verileri parse ediyoruz
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # parse edilecek ilgili iceriklerin dahil edilmesi class tabanli selector
        )
    ),
)

docs = loader.load()  # dokumani indir


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)  # veri parcalama yapisi, chucksize kac karatere kadar parcalama limit daha buyuk verilerde chunk_size 1000 1500 civari uygun


splits = text_splitter.split_documents(docs)  # dokumanlarin bolunmus hali

vector_store = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)  # veritabanina verileri vectorize edip kaydet

retriever = (
    vector_store.as_retriever()
)  # vektorize edilmis verileri getir, baglam verileri

# rag prompt

prompt = hub.pull(
    "rlm/rag-prompt"
)  # baskalarinin olusturdugu promptlar, prompt kutuphanesine bak. elle yazmakla ayni sey, question ve context verilerini verecegiz. context retriever da parse ettigimiz baglamlar olacak

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    for chunk in rag_chain.stream("what is test decomposition?"):
        print(chunk, end="", flush=True)
