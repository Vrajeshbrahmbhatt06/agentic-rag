# import chromadb
# client = chromadb.PersistentClient(path="chroma_db")

# # Get the collection (bookshelf)
# collection = client.get_collection("papers_chunks")

# query = "What are the main use of Agents?"
# results = collection.query(
#     query_texts=[query],
#     n_results=3  # top 3 results
# )

# for i, doc in enumerate(results["documents"][0]):
#     print(f"--- Result {i+1} ---")
#     print(doc[:400], "...\n")  # preview first 400 chars
#     print("Metadata:", results["metadatas"][0][i])
