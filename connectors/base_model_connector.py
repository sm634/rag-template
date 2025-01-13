"""
Docstring
---------

The base class to connect to models. There are currently supported for LLM foundation models from OpenAI and Watsonx.
The particular values assigned to parameters specified within this class will depend on the config file. Once the
values have been assigned to the relevant class attributes, these will be inherited in ModelsConnector class.
"""
import os

from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

from dotenv import load_dotenv
from utils.files_handler import FileHandler


class BaseModelConnector:

    def __init__(self):
        """
        BaseModelConnector reads in the models_config.yaml file and sets up the relevant variable values to instantiate the
        required model to be initialized using hte ModelConnector class, handling dependencies for model access and
        inference.
        """
        # we will need the environment variables for credentials.
        self.provider_task = None
        load_dotenv()

        # instantiate the credentials values
        self.api_key = ''
        self.project_id = ''
        self.model_endpoint = ''

        # get models configs
        file_handler = FileHandler()
        file_handler.get_config('models_config.yaml')
        self.config = file_handler.config

        # get the model provider and the task of interest.
        self.model_provider = self.config['MODEL_PROVIDER'].lower()
        self.task = self.config['TASK'].lower()

        if self.model_provider == 'openai':
            # get the credentials
            self.api_key = os.environ['OPENAI_API_KEY']

            if self.task == 'create_knowledge_base_webcrawl':
                self.provider_task = self.config['OPENAI']['CREATE_KNOWLEDGE_BASE_WEBCRAWL']
            elif self.task == 'create_knowledge_base_from_files':
                self.provider_task = self.config['OPENAI']['CREATE_KNOWLEDGE_BASE_FROM_FILES']
            elif self.task == 'generate_faq_answers':
                self.provider_task = self.config['OPENAI']['GENERATE_FAQ_ANSWERS']

        elif self.model_provider == 'watsonx':
            # get the watsonx credentials
            self.api_key = os.environ['WATSONX_APIKEY']
            self.project_id = os.environ['PROJECT_ID']
            self.model_endpoint = os.environ['WATSONX_MODEL_ENDPOINT']

            if self.task == 'create_knowledge_base':
                self.provider_task = self.config['WATSONX']['CREATE_KNOWLEDGE_BASE_WEBCRAWL']
            elif self.task == 'create_knowledge_base_from_files':
                self.provider_task = self.config['WATSONX']['CREATE_KNOWLEDGE_BASE_FROM_FILES']
            elif self.task == 'generate_faq_answers':
                self.provider_task = self.config['WATSONX']['GENERATE_FAQ_ANSWERS']
        else:
            raise

        model_type = self.provider_task['model_type']

        if self.model_provider == 'watsonx':
            try:
                self.model_type = getattr(ModelTypes, model_type)
                self.model_name = self.model_type.name
                self.model_id = self.model_type.value
            except AttributeError:
                self.model_id = model_type
                self.model_type = model_type
                self.model_name = model_type[model_type.index('/')+1:]
        else:
            self.model_type = model_type
            self.model_name = model_type

        self.decoding_method = self.provider_task['decoding_method']

        self.max_tokens = self.provider_task['max_tokens']
        self.min_tokens = self.provider_task['min_tokens']
        self.temperature = self.provider_task['temperature']
        self.top_p = self.provider_task['top_p']
        self.top_k = self.provider_task['top_k']
        self.repetition_penalty = self.provider_task['repetition_penalty']
        if not isinstance(self.provider_task['stop_sequences'], type(None)):
            if ',' in self.provider_task['stop_sequences']:
                self.stop_sequences = self.provider_task['stop_sequences'].split(',')  # needs to be a list.
            else:
                self.stop_sequences = list(self.provider_task['stop_sequences'])
        else:
            self.stop_sequences = None