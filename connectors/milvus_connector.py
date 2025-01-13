import os
from dotenv import load_dotenv
import sys

from utils.files_handler import FileHandler
from pymilvus import MilvusClient, MilvusException, DataType
from sentence_transformers import SentenceTransformer


class MilvusConnector:

    def __init__(self, local=True, db_name="thames_water_procurement.db"):
        """
        Instantiate the MilvusConnector class. 
        Choose between connecting to local or remote hosted instance of Milvus.
        Methods use standard MilvusClient methods and attributes, with some modication and abstraction.
        """

        if local:
            try:
                self.client = MilvusClient(db_name)
                self_hosted_conn = self.client.is_self_hosted
                if self_hosted_conn:
                    print("Connection established to Milvus client!")
            except MilvusException:
                raise

        file_handler = FileHandler()
        file_handler.get_config('vector_db_config.yaml')
        self.params_config = file_handler.config['MILVUS']
        self.index_building_params = self.params_config['INDEX_BUILDING_PARAMS']
        self.search_params = self.params_config['SEARCH_PARAMS']
        
        file_handler.get_config('embeddings_config.yaml')
        embedding_model_config = file_handler.config

        self.model_provider = embedding_model_config['MODEL_PROVIDER']
        self.model_name = embedding_model_config[self.model_provider]['EMBEDDING_MODEL']

        # read model parameters if available
        self.model_params = embedding_model_config[self.model_provider]['MODEL_PARAMS']
        self.model_max_seq_len = self.model_params['max_input_tokens']
        self.embeddings_dim = self.model_params['dimension']

        # print("self.model_params is: ", self.model_params)
        # print("self.model_max_seq_len is: ", self.model_max_seq_len)

        if self.model_provider == 'HUGGING_FACE':
            # Load the SentenceTransformer model
            self.model = SentenceTransformer(self.model_name)
            self.model.max_seq_length = self.model_max_seq_len
            # print(f"Max Input Tokens: {self.model.max_seq_length}")

        elif self.model_provider == 'ELASTICSEARCH':
            self.model = self.model_name


        # store schema if created
        self.collection_schema = None

    def list_collections(self):
        return self.client.list_collections()

    def check_collection(self, collection_name):
        """Check of collection exists"""
        return self.client.has_collection(collection_name=collection_name)
    
    def drop_collection(self, collection_name):
        return self.client.drop_collection(collection_name=collection_name)
    
    def create_doc_url_schema(self):
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False
        )

        # Add the fields to the schema for storing embeddings, id and url.
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embeddings", datatype=DataType.FLOAT_VECTOR, dim=self.embeddings_dim)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1024, is_primary=False)
        schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length =128, is_primary=False)
        schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=64, is_primary=False)

        self.collection_schema = schema

    def create_doc_files_schema(self):
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False
        )

        # Add the fields to the schema for storing embeddings, id and url.
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embeddings", datatype=DataType.FLOAT_VECTOR, dim=self.embeddings_dim)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1024, is_primary=False)
        schema.add_field(field_name="document_title", datatype=DataType.VARCHAR, max_length=128, is_primary=False)
        schema.add_field(field_name="page_number", datatype=DataType.VARCHAR, max_length=32, is_primary=False)
        schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=128, is_primary=False)

        self.collection_schema = schema

    def create_collection(self, collection_name):
        """Create collection based on simple logic."""
        # Drop collection if it already exists.
        if self.client.has_collection(collection_name=collection_name):
            print(f"{collection_name} already exists. Dropping.")
            self.client.drop_collection(collection_name=collection_name)
            print(f"{collection_name} dropped. Creating new collection.")

        # Create collection.
        if self.collection_schema is not None:
            print("Building Index")
            # Prepare Index parameters
            index_params = self.client.prepare_index_params()
            # add indexes
            index_params.add_index(
                field_name="embeddings",
                metric_type="COSINE",
                params={"nlist": 128}
            )
            
            # create collection with schema and index
            self.client.create_collection(
                collection_name=collection_name,
                schema=self.collection_schema,
                index_params=index_params
            )

        else:
            self.client.create_collection(
                collection_name=collection_name,
                dimension=self.embeddings_dim
            )
        
        if self.client.has_collection(collection_name=collection_name):
            return f"Collection {collection_name} created!"
        else:
            return f"Unable to create collection {collection_name}"
        
    def describe_collection(self, collection_name):
        return self.client.describe_collection(collection_name)
        
    def insert(self, collection_name, data):
        """Insert data to collection based on simple logic."""
        
        result = self.client.insert(
            collection_name=collection_name,
            data=data
        )

        print(result)
        
        return result
    
    def get_embedding(self, text):
        """
        Encode a piece of text into a vector representation using the model.
        """
        return self.model.encode(text)
    
    def search(
            self, 
            collection_name, 
            query: str, 
            fields: list,
            n_results: int = 4,
            anns_field='vector'):
        """Conduct search on the knowledge base using vector similarity search."""
        # model = SentenceTransformer(self.embedding_model_name)
        query_vector = self.model.encode([query])

        response = self.client.search(
            collection_name=collection_name,
            data=query_vector,
            anns_field=anns_field,
            limit=n_results,
            output_fields=fields
        )

         # Prepare output
        result = {field: [] for field in fields}
        result['scores'] = []

        # Populate the output dictionary with results and scores
        for hits in response:
            for hit in hits:
                result['scores'].append(hit['distance'])
                for field in fields:
                    result[field].append(hit['entity'].get(field, None))

        return result
