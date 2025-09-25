# nltk related imports
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import download

# other packages needed
import numpy as np
import re
from typing import List, Dict, Tuple
from collections import defaultdict, Counter


class InformationRetrievalSystem:
    def __init__(self):
        # Download required NLTK data
        print("Downloading NLTK data...")
        download('wordnet', quiet=True)
        download('stopwords', quiet=True)
        print("NLTK data downloaded successfully.")

        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer(language='english')
        print("NLTK components initialized.")
        
        # Core data structures
        self.documents = []  # Store original parsed documents
        self.processed_docs_content = [] # Store only the processed content for matrix building
        self.dictionary = [] # The sorted vocabulary list
        self.word_to_index = {} # Word -> index map for fast lookups
        self.relevance_judgements = defaultdict(set)
        
        # TF-IDF matrices
        self.tf_matrix = None
        self.idf_matrix = None
        self.tf_idf_matrix = None
        
        # System state
        self.is_initialized = False

    # === PARSING & UTILITY FUNCTIONS ===
    
    def parse_documents(self, dataset_content: str) -> List[Dict]:
        """
        Parses dataset format and extracts documents.
        """
        parsed_docs = []
        doc_parts = re.split(r'\.I\s*(\d+)', dataset_content)[1:]

        for i in range(0, len(doc_parts), 2):
            if i + 1 < len(doc_parts):
                doc_index = int(doc_parts[i].strip())
                doc_content = doc_parts[i+1].strip()

                title_match = re.search(r'\.T(.*?)(?=\.[A-Z]|\Z)', doc_content, re.DOTALL)
                author_match = re.search(r'\.A(.*?)(?=\.[A-Z]|\Z)', doc_content, re.DOTALL)
                abstract_match = re.search(r'\.W(.*?)(?=\.[A-Z]|\Z)', doc_content, re.DOTALL)

                title = title_match.group(1).strip() if title_match else ""
                author = author_match.group(1).strip() if author_match else ""
                abstract = abstract_match.group(1).strip() if abstract_match else ""
                
                parsed_docs.append({
                    "id": doc_index, 
                    "content": f"{title}\n{author}\n{abstract}"
                })
        return parsed_docs

    def parse_relevance_judgement(self, relevance_content: str) -> Dict[int, List[int]]:
        """
        Parses relevance judgements content.
        """
        relevance = defaultdict(set)
        for line in relevance_content.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                query_id = int(parts[0])
                doc_id = int(parts[1])
                relevance[query_id].add(doc_id)
        return relevance

    def parse_queries(self, query_content: str) -> List[Dict]:
        """
        Parses the query file.
        """
        queries = []
        query_parts = re.split(r'\.I\s*(\d+)', query_content)[1:]

        for i in range(0, len(query_parts), 2):
            if i + 1 < len(query_parts):
                query_id = int(query_parts[i].strip())
                content = query_parts[i+1]
                text_match = re.search(r'\.W\s*(.*?)(?=\s*\.I|\Z)', content, re.DOTALL)
                if text_match:
                    query_text = text_match.group(1).strip().replace('\n', ' ')
                    queries.append({"id": query_id, "text": query_text})
        return queries

    # === TEXT PROCESSING ===

    def is_meaningful(self, word: str) -> bool:
        return len(wordnet.synsets(word)) > 0

    def is_valid_word(self, word: str) -> bool:
        return word.isalnum() and word not in self.stop_words and self.is_meaningful(word)

    def _process_text(self, text: str) -> List[str]:
        processed_tokens = []
        for word in text.lower().split():
            if self.is_valid_word(word):
                processed_tokens.append(self.stemmer.stem(word))
        return processed_tokens

    def process_query(self, query: str) -> List[str]:
        processed_tokens = self._process_text(query)
        return [token for token in processed_tokens if token in self.word_to_index]

    # === TF-IDF & RANKING ===

    def _build_vocabulary_and_process_docs(self):
        print("# (1/3) Processing documents and building vocabulary...")
        vocabulary = set()
        for doc in self.documents:
            processed_tokens = self._process_text(doc["content"])
            self.processed_docs_content.append(processed_tokens)
            vocabulary.update(processed_tokens)
        
        self.dictionary = sorted(list(vocabulary))
        self.word_to_index = {word: i for i, word in enumerate(self.dictionary)}
        print(f"# Vocabulary built with {len(self.dictionary)} unique terms.")

    def _calculate_tf_idf_matrices(self):
        print("# (2/3) Calculating TF-IDF matrices...")
        num_docs = len(self.documents)
        dict_size = len(self.dictionary)

        self.tf_matrix = np.zeros((dict_size, num_docs))
        df_vector = np.zeros((dict_size, 1))

        for doc_idx, processed_tokens in enumerate(self.processed_docs_content):
            word_counts = Counter(processed_tokens)
            for word, freq in word_counts.items():
                if word in self.word_to_index:
                    word_idx = self.word_to_index[word]
                    self.tf_matrix[word_idx, doc_idx] = freq
                    df_vector[word_idx] += 1
        
        self.tf_matrix = np.where(self.tf_matrix > 0, 1 + np.log10(self.tf_matrix), 0)
        self.idf_matrix = np.log10((num_docs + 1) / (df_vector + 1))
        self.tf_idf_matrix = self.tf_matrix * self.idf_matrix
        print("# TF-IDF matrices calculated successfully.")

    def calc_query_tfidf(self, tokenized_query: List[str]):
        query_tf = np.zeros((len(self.dictionary), 1))
        word_counts = Counter(tokenized_query)

        for word, freq in word_counts.items():
            if word in self.word_to_index:
                word_idx = self.word_to_index[word]
                query_tf[word_idx, 0] = 1 + np.log10(freq) if freq > 0 else 0
        
        return query_tf * self.idf_matrix

    def cosine_similarity(self, vec1, vec2):
        dot = np.dot(vec1.T, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot[0, 0] / (norm1 * norm2)

    def rank_documents(self, query_tfidf) -> List[Tuple[int, float]]:
        scores = []
        for i in range(self.tf_idf_matrix.shape[1]):
            doc_vector = self.tf_idf_matrix[:, i:i+1]
            sim = self.cosine_similarity(query_tfidf, doc_vector)
            scores.append((self.documents[i]["id"], sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # === EVALUATION METRICS ===

    def _calculate_metrics(self, query_id: int, retrieved_doc_ids: List[int]) -> Dict:
        relevant_docs = self.relevance_judgements.get(query_id, set())
        if not relevant_docs:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        retrieved_set = set(retrieved_doc_ids)
        relevant_retrieved = relevant_docs.intersection(retrieved_set)
        
        num_relevant_retrieved = len(relevant_retrieved)
        num_retrieved = len(retrieved_doc_ids)
        num_total_relevant = len(relevant_docs)

        precision = num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0.0
        recall = num_relevant_retrieved / num_total_relevant if num_total_relevant > 0 else 0.0
        
        f1_score = 0.0
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        return {"precision": precision, "recall": recall, "f1_score": f1_score}

    def evaluate_system(self, query_content: str, top_k: int = 10) -> Dict:
        """
        Runs all queries from the query file, calculates metrics for each,
        and returns the overall system performance.
        """
        if not self.is_initialized:
            return {"status": "error", "message": "System not initialized."}

        parsed_queries = self.parse_queries(query_content)
        if not parsed_queries:
            return {"status": "error", "message": "No queries found in the provided content."}

        all_metrics = []
        individual_results = {}
        
        print(f"Evaluating system with {len(parsed_queries)} queries...")
        for query in parsed_queries:
            query_id = query["id"]
            query_text = query["text"]
            
            if query_id in self.relevance_judgements:
                search_result = self.search(query_text, top_k=top_k, query_id=query_id)
                if search_result['status'] == 'success':
                    metrics = search_result['metrics']
                    all_metrics.append(metrics)
                    individual_results[query_id] = {
                        "query_text": query_text,
                        "metrics": metrics
                    }
        
        if not all_metrics:
            return {"status": "warning", "message": "No queries with relevance judgements were found to evaluate."}

        # Calculate mean scores
        mean_precision = np.mean([m['precision'] for m in all_metrics])
        mean_recall = np.mean([m['recall'] for m in all_metrics])
        mean_f1_score = np.mean([m['f1_score'] for m in all_metrics])
        
        print("Evaluation complete.")
        return {
            "status": "success",
            "overall_performance": {
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "mean_f1_score": mean_f1_score,
                "evaluated_query_count": len(all_metrics)
            },
            # "individual_query_results": individual_results
        }

    # === BACKEND API METHODS ===
    
    def initialize_system(self, dataset_content: str, relevance_content: str) -> Dict:
        try:
            print("# Starting system initialization...")
            self.documents = self.parse_documents(dataset_content)
            self.relevance_judgements = self.parse_relevance_judgement(relevance_content)
            
            if not self.documents:
                return {"status": "error", "message": "No documents found in dataset"}
            
            self._build_vocabulary_and_process_docs()
            self._calculate_tf_idf_matrices()
            
            print("# (3/3) System initialization complete.")
            self.is_initialized = True
            
            return {
                "status": "success",
                "message": "System initialized successfully",
                "statistics": {
                    "total_documents": len(self.documents),
                    "dictionary_size": len(self.dictionary),
                    "tf_idf_matrix_shape": self.tf_idf_matrix.shape
                }
            }
        except Exception as e:
            return {"status": "error", "message": f"Initialization failed: {str(e)}"}
    
    def search(self, query: str, top_k: int = 10, query_id: int = None) -> Dict:
        if not self.is_initialized:
            return {"status": "error", "message": "System not initialized."}
        
        try:
            tokenized_query = self.process_query(query)
            
            if not tokenized_query:
                return {"status": "warning", "message": "No valid terms found in query", "results": []}
            
            query_tfidf = self.calc_query_tfidf(tokenized_query)
            ranked_docs = self.rank_documents(query_tfidf)
            
            doc_map = {doc['id']: doc for doc in self.documents}
            results = []
            retrieved_doc_ids = []
            for doc_id, score in ranked_docs[:top_k]:
                if score > 0:
                    retrieved_doc_ids.append(doc_id)
                    results.append({
                        "rank": len(results) + 1,
                        "document_id": doc_id,
                        "similarity_score": float(score),
                        "content": doc_map[doc_id]["content"]
                    })

            metrics = {"precision": "N/A", "recall": "N/A", "f1_score": "N/A"}
            if query_id is not None:
                metrics = self._calculate_metrics(query_id, retrieved_doc_ids)
            
            return {
                "status": "success",
                "query": query,
                "processed_query": tokenized_query,
                "total_results": len(results),
                "metrics": metrics,
                "results": results,
            }
        except Exception as e:
            return {"status": "error", "message": f"Search failed: {str(e)}"}