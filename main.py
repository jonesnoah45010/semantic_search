import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import ssl
import certifi


ssl._create_default_https_context = ssl._create_unverified_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SemanticQuery:
    def __init__(self, directory_path="tmp/your_files_directory", collection_name="your_collection_name", model='all-MiniLM-L6-v2'):
        self.directory_path = directory_path
        self.model = SentenceTransformer(model)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name)
        contents = []
        metadatas = []
        i = 0
        ids = []
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.directory_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    contents.append(content)
                    metadatas.append({"source": filename})
                    ids.append("file" + str(i))
                    i += 1
        embeddings = self.model.encode(contents, show_progress_bar=False)
        self.collection.add(
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query_files(self, query_text, num_results=3):
        # finds which files are relevent to your query
        results = self.collection.query(
            query_texts=[query_text],
            n_results=num_results
        )
        results["matches"] = results["metadatas"][0]
        new_matches = []
        for item in results["matches"]:
            new_matches.append(item["source"])
        results["matches"] = new_matches
        results["file_contents"] = results["documents"][0]
        return results

    def query_single_file(self, file_path, query_text, num_results=3, sentence_chunking=5):
        file_path = os.path.join(self.directory_path, file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        sentences = sent_tokenize(content)
        sentence_groups = [' '.join(sentences[i:i+sentence_chunking]) for i in range(0, len(sentences), sentence_chunking)]
        sentence_embeddings = self.model.encode(sentence_groups, show_progress_bar=False)
        query_embedding = self.model.encode([query_text], show_progress_bar=False)[0]
        similarities = [np.dot(query_embedding, sentence_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(sentence_embedding)) for sentence_embedding in sentence_embeddings]
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:num_results]
        top_sentence_groups = [sentence_groups[i] for i in top_indices]
        top_similarities = [similarities[i] for i in top_indices]
        return list(zip(top_sentence_groups, top_similarities))

if __name__ == "__main__":

    searcher = SemanticQuery("my_docs", "my_collection")

    print("\n\n\n\n\n\n\n\n\n\n\n\n")

    q = "whales and the sea"
    results = searcher.query_files(q, 1)

    print(q)
    print("...")
    print(results["matches"])
    print("...")

    print("\n\n\n\n")

    q2 = "reanimating corpses"
    results2 = searcher.query_files(q2, 1)

    print(q2)
    print("...")
    print(results2["matches"])
    print("...")

    print("\n\n\n\n")

    q3 = "a girl getting lost down a rabbit hole"
    results3 = searcher.query_files(q3, 1)

    print(q3)
    print("...")
    print(results3["matches"])
    print("...")

    print("\n\n\n\n")



    file_path = "alice_in_wonderland.txt"
    q4 = 'the mad hatter'
    single_file_results = searcher.query_single_file(file_path, q4, num_results=3)

    print("Single file query results: \n")
    for sentence, similarity in single_file_results:
        print(f"Sentences: {sentence}")
        print(f"Similarity: {similarity}")
        print("...")








