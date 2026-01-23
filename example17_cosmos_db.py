# pyautogen[retrievechat]==0.2.35
# azure-cosmos==4.7.0
# openai==1.40.0
# PyPDF2==3.0.1
# sentence-transformers==2.2.2


"""
HR Q&A Bot using AutoGen with Cosmos DB Vector Search
This script creates a complete RAG-based chatbot for HR queries using:
- Azure Cosmos DB for NoSQL with vector search
- Azure OpenAI for embeddings and LLM
- AutoGen for multi-agent conversation
- PyPDF2 for PDF processing
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import hashlib
from typing import List, Dict, Tuple
import PyPDF2
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.vectordb.base import VectorDB


# ============================================================================
# CONFIGURATION
# ============================================================================
# Azure Cosmos DB Configuration
COSMOS_ENDPOINT = "https://your-account.documents.azure.com:443/"
COSMOS_KEY = "your-cosmos-primary-key"
COSMOS_DATABASE_NAME = "hr_knowledge_base"
COSMOS_CONTAINER_NAME = "hr_policies"

# Azure OpenAI Configuration (for embeddings and chat)
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_KEY = "your-azure-openai-key"
AZURE_OPENAI_API_VERSION = "2024-02-01"
EMBEDDING_MODEL = "text-embedding-ada-002"  # Deployment name in Azure
CHAT_MODEL = "gpt-4"  # Deployment name in Azure

# OpenAI Configuration (alternative to Azure OpenAI for chat)
OPENAI_API_KEY = "your-openai-key"  # If using OpenAI instead of Azure OpenAI

# Document Processing Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
EMBEDDING_DIMENSIONS = 1536  # For text-embedding-ada-002


# ============================================================================
# CUSTOM COSMOS DB VECTOR DATABASE CLASS
# ============================================================================
class CosmosDBVectorDB(VectorDB):
    """
    Custom Cosmos DB Vector Database implementation for AutoGen.
    Extends the base VectorDB class to provide vector similarity search
    capabilities using Azure Cosmos DB for NoSQL.
    """
    
    def __init__(
        self,
        cosmos_endpoint: str,
        cosmos_key: str,
        database_name: str,
        container_name: str,
        embedding_function=None,
        embedding_dimensions: int = 1536
    ):
        """
        Initialize Cosmos DB vector database.
        
        Args:
            cosmos_endpoint: Cosmos DB account endpoint URL
            cosmos_key: Cosmos DB account primary key
            database_name: Name of the database to use/create
            container_name: Name of the container to use/create
            embedding_function: Function to generate embeddings from text
            embedding_dimensions: Dimension of embedding vectors (default 1536 for Ada-002)
        """
        # Initialize Cosmos DB client
        self.client = CosmosClient(cosmos_endpoint, cosmos_key)
        self.embedding_function = embedding_function
        self.embedding_dimensions = embedding_dimensions
        
        # Create database if it doesn't exist
        self.database = self.client.create_database_if_not_exists(database_name)
        
        # Define vector embedding policy for the container
        # This tells Cosmos DB how to index and search vector data
        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/contentVector",  # Field containing embedding vectors
                    "dataType": "float32",     # Data type of vector elements
                    "distanceFunction": "cosine",  # Similarity metric (cosine, euclidean, dotproduct)
                    "dimensions": embedding_dimensions  # Vector dimensions
                }
            ]
        }
        
        # Define indexing policy
        # Exclude contentVector from default indexing for performance
        indexing_policy = {
            "includedPaths": [{"path": "/*"}],  # Index all paths by default
            "excludedPaths": [
                {"path": "/\"_etag\"/?"},  # Exclude system fields
                {"path": "/contentVector/*"}  # Exclude vectors from default indexing
            ],
            "vectorIndexes": [
                {
                    "path": "/contentVector",  # Path to vector field
                    "type": "quantizedFlat"  # Vector index type (quantizedFlat or diskANN)
                }
            ]
        }
        
        # Create container with vector support
        try:
            self.container = self.database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path='/id'),  # Partition by document ID
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embedding_policy
            )
            print(f"✓ Container '{container_name}' ready with vector search enabled")
        except exceptions.CosmosHttpResponseError as e:
            print(f"✗ Error creating container: {e}")
            raise
    
    def create_collection(self, collection_name: str, overwrite: bool = False, get_or_create: bool = True):
        """
        Create or get collection (required by VectorDB interface).
        For Cosmos DB, we use a single container, so just return the name.
        """
        return collection_name
    
    def get_collection(self, collection_name: str = None):
        """Get collection (required by VectorDB interface)."""
        return collection_name
    
    def delete_collection(self, collection_name: str):
        """Delete collection (not implemented for safety)."""
        print("Delete collection not implemented for Cosmos DB container")
        pass
    
    def insert_docs(self, docs: List[Dict], collection_name: str = None, upsert: bool = True):
        """
        Insert documents with embeddings into Cosmos DB.
        
        Args:
            docs: List of documents to insert. Each doc should have 'content' field
            collection_name: Collection name (unused for Cosmos DB)
            upsert: If True, update existing documents; if False, only insert new
        """
        inserted_count = 0
        
        for doc in docs:
            try:
                # Generate embedding if not already present
                if "contentVector" not in doc and self.embedding_function:
                    text = doc.get("content", "")
                    if text.strip():  # Only generate embedding if content exists
                        doc["contentVector"] = self.embedding_function(text)
                
                # Generate unique ID if not present
                if "id" not in doc:
                    # Use MD5 hash of content as ID
                    content_hash = hashlib.md5(doc.get("content", "").encode()).hexdigest()
                    doc["id"] = f"doc_{content_hash}"
                
                # Upsert document into Cosmos DB
                self.container.upsert_item(doc)
                inserted_count += 1
                
            except Exception as e:
                print(f"✗ Error inserting document: {e}")
                continue
        
        print(f"✓ Inserted/Updated {inserted_count} documents")
    
    def update_docs(self, docs: List[Dict], collection_name: str = None):
        """Update existing documents."""
        self.insert_docs(docs, collection_name, upsert=True)
    
    def delete_docs(self, ids: List[str], collection_name: str = None):
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            collection_name: Collection name (unused)
        """
        deleted_count = 0
        for doc_id in ids:
            try:
                self.container.delete_item(item=doc_id, partition_key=doc_id)
                deleted_count += 1
            except exceptions.CosmosResourceNotFoundError:
                print(f"Document {doc_id} not found")
                continue
        
        print(f"✓ Deleted {deleted_count} documents")
    
    def retrieve_docs(
        self, 
        queries: List[str], 
        collection_name: str = None, 
        n_results: int = 5,
        distance_threshold: float = -1
    ) -> List[List[Tuple]]:
        """
        Retrieve documents using vector similarity search.
        This is the core RAG retrieval method.
        
        Args:
            queries: List of query strings to search for
            collection_name: Collection name (unused)
            n_results: Number of top results to return per query
            distance_threshold: Maximum distance for results (-1 = no threshold)
            
        Returns:
            List of lists of tuples (content, metadata) for each query
        """
        all_results = []
        
        for query in queries:
            try:
                # Generate embedding for the query
                query_embedding = self.embedding_function(query) if self.embedding_function else []
                
                if not query_embedding:
                    print(f"✗ No embedding generated for query: {query}")
                    all_results.append([])
                    continue
                
                # Perform vector similarity search using VectorDistance
                # VectorDistance calculates similarity between query vector and stored vectors
                query_sql = f"""
                SELECT TOP {n_results} 
                    c.id, 
                    c.content, 
                    c.metadata,
                    VectorDistance(c.contentVector, @embedding) AS SimilarityScore
                FROM c
                ORDER BY VectorDistance(c.contentVector, @embedding)
                """
                
                # Execute query with embedding as parameter
                items = list(self.container.query_items(
                    query=query_sql,
                    parameters=[
                        {"name": "@embedding", "value": query_embedding}
                    ],
                    enable_cross_partition_query=True
                ))
                
                # Format results as list of tuples (content, metadata)
                # This is the format AutoGen expects
                query_results = []
                for item in items:
                    similarity = item.get("SimilarityScore", 1.0)
                    
                    # Filter by distance threshold if specified
                    if distance_threshold < 0 or similarity <= distance_threshold:
                        query_results.append((
                            item.get("content", ""),
                            {
                                **item.get("metadata", {}),
                                "similarity_score": similarity,
                                "doc_id": item.get("id", "")
                            }
                        ))
                
                all_results.append(query_results)
                print(f"✓ Found {len(query_results)} results for query: {query[:50]}...")
                
            except Exception as e:
                print(f"✗ Error retrieving documents for query '{query}': {e}")
                all_results.append([])
        
        return all_results
    
    def get_docs_by_ids(self, ids: List[str], collection_name: str = None) -> List[Dict]:
        """
        Retrieve documents by their IDs.
        
        Args:
            ids: List of document IDs
            collection_name: Collection name (unused)
            
        Returns:
            List of documents
        """
        docs = []
        for doc_id in ids:
            try:
                item = self.container.read_item(item=doc_id, partition_key=doc_id)
                docs.append(item)
            except exceptions.CosmosResourceNotFoundError:
                print(f"Document {doc_id} not found")
                continue
        
        return docs


# ============================================================================
# PDF PROCESSOR CLASS
# ============================================================================
class PDFProcessor:
    """
    Handles PDF processing, text extraction, chunking, and embedding generation.
    """
    
    def __init__(
        self, 
        azure_endpoint: str, 
        azure_api_key: str, 
        api_version: str = "2024-02-01",
        embedding_model: str = "text-embedding-ada-002"
    ):
        """
        Initialize PDF processor with Azure OpenAI client.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            api_version: API version to use
            embedding_model: Name of embedding model deployment
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=api_version
        )
        self.embedding_model = embedding_model
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"📄 Extracting text from {pdf_path} ({total_pages} pages)...")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                
                print(f"✓ Extracted {len(text)} characters from {total_pages} pages")
                
        except Exception as e:
            print(f"✗ Error extracting text from {pdf_path}: {e}")
            
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        # Create overlapping chunks
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end].strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start += (chunk_size - overlap)
        
        print(f"✓ Created {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})")
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using Azure OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Call Azure OpenAI embeddings API
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
            
        except Exception as e:
            print(f"✗ Error generating embedding: {e}")
            return []
    
    def process_pdf_to_docs(
        self, 
        pdf_path: str, 
        chunk_size: int = 1000,
        overlap: int = 200,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Complete pipeline: Extract text from PDF, chunk it, and generate embeddings.
        
        Args:
            pdf_path: Path to PDF file
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            metadata: Additional metadata to attach to each chunk
            
        Returns:
            List of documents ready for insertion into vector DB
        """
        print(f"\n{'='*60}")
        print(f"Processing PDF: {pdf_path}")
        print(f"{'='*60}")
        
        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print(f"✗ No text extracted from {pdf_path}")
            return []
        
        # Step 2: Chunk the text
        chunks = self.chunk_text(text, chunk_size, overlap)
        
        # Step 3: Create documents with embeddings
        docs = []
        for i, chunk in enumerate(chunks):
            print(f"Generating embedding for chunk {i+1}/{len(chunks)}...", end='\r')
            
            # Generate embedding for this chunk
            embedding = self.generate_embedding(chunk)
            
            # Create document structure
            doc = {
                "content": chunk,
                "contentVector": embedding,
                "metadata": {
                    "source": os.path.basename(pdf_path),
                    "full_path": pdf_path,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                }
            }
            docs.append(doc)
        
        print(f"\n✓ Created {len(docs)} documents with embeddings")
        return docs


# ============================================================================
# HR Q&A BOT CLASS
# ============================================================================
class HRQABot:
    """
    Complete HR Q&A Bot using AutoGen with Cosmos DB RAG.
    """
    
    def __init__(
        self,
        cosmos_db: CosmosDBVectorDB,
        chat_model: str = "gpt-4",
        api_key: str = None,
        use_azure_openai: bool = True,
        azure_endpoint: str = None
    ):
        """
        Initialize HR Q&A Bot.
        
        Args:
            cosmos_db: Initialized Cosmos DB vector database
            chat_model: Model name for chat/completion
            api_key: API key for OpenAI or Azure OpenAI
            use_azure_openai: If True, use Azure OpenAI; else use OpenAI
            azure_endpoint: Azure OpenAI endpoint (required if use_azure_openai=True)
        """
        self.cosmos_db = cosmos_db
        
        # Configure LLM for AutoGen agents
        if use_azure_openai:
            # Azure OpenAI configuration
            llm_config = {
                "config_list": [{
                    "model": chat_model,
                    "api_type": "azure",
                    "api_key": api_key,
                    "base_url": azure_endpoint,
                    "api_version": AZURE_OPENAI_API_VERSION
                }],
                "timeout": 120,
                "temperature": 0,  # Use 0 for deterministic responses
                "cache_seed": None  # Disable caching for fresh responses
            }
        else:
            # Standard OpenAI configuration
            llm_config = {
                "config_list": [{
                    "model": chat_model,
                    "api_key": api_key
                }],
                "timeout": 120,
                "temperature": 0
            }
        
        # Create HR Assistant Agent
        # This agent generates answers based on retrieved context
        self.hr_assistant = AssistantAgent(
            name="hr_assistant",
            system_message="""You are a helpful and professional HR assistant. 
            Your role is to answer employee questions about company policies, benefits, 
            leave policies, compensation, and other HR-related topics.
            
            IMPORTANT RULES:
            1. Always base your answers on the provided context from HR documents
            2. If information is not in the context, say "I don't have that information in the HR documents"
            3. Be concise but complete in your answers
            4. Use a friendly, professional tone
            5. If asked about personal employee data, remind them to contact HR directly
            6. Cite the source document when providing information
            """,
            llm_config=llm_config
        )
        
        # Create Retrieve User Proxy Agent
        # This agent handles document retrieval from vector DB
        self.hr_ragproxy = RetrieveUserProxyAgent(
            name="hr_ragproxy",
            human_input_mode="NEVER",  # Fully automated, no human input needed
            max_consecutive_auto_reply=5,  # Limit conversation length
            retrieve_config={
                "task": "qa",  # Question-answering task
                "vector_db": cosmos_db,  # Use our custom Cosmos DB
                "n_results": 5,  # Retrieve top 5 most relevant chunks
                "get_or_create": True,  # Create collection if doesn't exist
                "overwrite": False  # Don't overwrite existing data
            }
        )
        
        print("✓ HR Q&A Bot initialized successfully")
    
    def ask(self, question: str) -> str:
        """
        Ask the HR bot a question.
        
        Args:
            question: Employee's question
            
        Returns:
            Bot's answer
        """
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}\n")
        
        # Initiate chat between agents
        # The RAG proxy will retrieve relevant docs and pass to assistant
        self.hr_ragproxy.initiate_chat(
            self.hr_assistant,
            message=self.hr_ragproxy.message_generator,
            problem=question,
            n_results=5
        )
        
        return "Chat completed"


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main function to set up and run the HR Q&A bot.
    """
    
    print("\n" + "="*60)
    print("HR Q&A BOT - INITIALIZATION")
    print("="*60 + "\n")
    
    # Step 1: Initialize PDF Processor
    print("Step 1: Initializing PDF Processor...")
    pdf_processor = PDFProcessor(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        embedding_model=EMBEDDING_MODEL
    )
    print("✓ PDF Processor initialized\n")
    
    # Step 2: Initialize Cosmos DB Vector Database
    print("Step 2: Initializing Cosmos DB Vector Database...")
    cosmos_db = CosmosDBVectorDB(
        cosmos_endpoint=COSMOS_ENDPOINT,
        cosmos_key=COSMOS_KEY,
        database_name=COSMOS_DATABASE_NAME,
        container_name=COSMOS_CONTAINER_NAME,
        embedding_function=pdf_processor.generate_embedding,
        embedding_dimensions=EMBEDDING_DIMENSIONS
    )
    print("✓ Cosmos DB initialized\n")
    
    # Step 3: Process and Index HR PDF Documents
    print("Step 3: Processing and Indexing HR Documents...")
    
    # List of HR PDF files to process
    hr_pdfs = [
        "./hr_documents/employee_handbook.pdf",
        "./hr_documents/benefits_guide.pdf",
        "./hr_documents/leave_policy.pdf",
        "./hr_documents/code_of_conduct.pdf"
    ]
    
    # Process each PDF and index in Cosmos DB
    for pdf_path in hr_pdfs:
        if os.path.exists(pdf_path):
            # Process PDF into chunks with embeddings
            docs = pdf_processor.process_pdf_to_docs(
                pdf_path=pdf_path,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
                metadata={
                    "document_type": "hr_policy",
                    "indexed_date": "2025-10-15"
                }
            )
            
            # Insert documents into Cosmos DB
            if docs:
                cosmos_db.insert_docs(docs)
        else:
            print(f"⚠ Warning: File not found - {pdf_path}")
    
    print("\n✓ All documents indexed\n")
    
    # Step 4: Initialize HR Q&A Bot
    print("Step 4: Initializing HR Q&A Bot...")
    hr_bot = HRQABot(
        cosmos_db=cosmos_db,
        chat_model=CHAT_MODEL,
        api_key=AZURE_OPENAI_KEY,
        use_azure_openai=True,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    print("✓ HR Bot ready\n")
    
    # Step 5: Run Example Queries
    print("Step 5: Running Example Queries...")
    print("="*60 + "\n")
    
    # Example questions
    questions = [
        "What is the parental leave policy for new fathers?",
        "How many vacation days do employees get per year?",
        "What health insurance options are available?",
        "What is the process for requesting medical leave?",
        "What is the company's policy on remote work?"
    ]
    
    # Ask each question
    for question in questions:
        hr_bot.ask(question)
        print("\n" + "-"*60 + "\n")
    
    print("\n" + "="*60)
    print("HR Q&A BOT - SESSION COMPLETE")
    print("="*60 + "\n")


# ============================================================================
# ALTERNATIVE: INTERACTIVE MODE
# ============================================================================
def interactive_mode():
    """
    Run bot in interactive mode for live Q&A.
    """
    # Initialize components
    pdf_processor = PDFProcessor(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_api_key=AZURE_OPENAI_KEY,
        embedding_model=EMBEDDING_MODEL
    )
    
    cosmos_db = CosmosDBVectorDB(
        cosmos_endpoint=COSMOS_ENDPOINT,
        cosmos_key=COSMOS_KEY,
        database_name=COSMOS_DATABASE_NAME,
        container_name=COSMOS_CONTAINER_NAME,
        embedding_function=pdf_processor.generate_embedding
    )
    
    hr_bot = HRQABot(
        cosmos_db=cosmos_db,
        chat_model=CHAT_MODEL,
        api_key=AZURE_OPENAI_KEY,
        use_azure_openai=True,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    
    print("\n" + "="*60)
    print("HR Q&A BOT - INTERACTIVE MODE")
    print("="*60)
    print("Type your questions (or 'quit' to exit)")
    print("="*60 + "\n")
    
    # Interactive loop
    while True:
        question = input("\n🤔 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋\n")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        hr_bot.ask(question)


# ============================================================================
# RUN THE PROGRAM
# ============================================================================
if __name__ == "__main__":
    # Choose mode:
    # 1. Run main() for automated demo with predefined questions
    # 2. Run interactive_mode() for live Q&A session
    
    main()  # Run automated demo
    # interactive_mode()  # Or run interactive mode
