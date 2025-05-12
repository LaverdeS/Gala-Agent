import datasets

from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool


def load_guest_dataset():
    """Loads the guest dataset and converts it into Document objects."""
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

    # Convert dataset entries into Document objects
    docs = [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}"
            ]),
            metadata={"name": guest["name"]}
        )
        for guest in guest_dataset
    ]

    # Return the documents
    return docs


# Load the dataset
docs = load_guest_dataset()

# Initialize the retriever
bm25_retriever = BM25Retriever.from_documents(docs)


def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.invoke(query)
    if results:
        return results[0].page_content  # [doc.page_content for doc in results[:1]]), :3
    else:
        return "No matching guest information found."


guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)


if __name__ == "__main__":
    query = "Marie"
    print(f"query: {query}:\nretrieval: {extract_text(query)}")