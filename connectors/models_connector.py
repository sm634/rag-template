from connectors.base_model_connector import BaseModelConnector

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM

from langchain_openai import ChatOpenAI


class ModelConnector(BaseModelConnector):

    def __init__(self):
        """
        The ModelConnector class initializing the specified model (based on config) and initialises the hyperparameters,
        many can be used/is relevant for different model providers. These are all integrated with LangChain, hence the
        set of applications are highly dependent on Langchain model capabilities.
        """
        # The model provider will be from a list of model providers.
        super().__init__()

        self.task = self.task

        self.params = {
            GenParams.MAX_NEW_TOKENS: self.max_tokens,
            GenParams.MIN_NEW_TOKENS: self.min_tokens,
            GenParams.DECODING_METHOD: self.decoding_method,
            GenParams.TEMPERATURE: self.temperature,
            GenParams.TOP_P: self.top_p,
            GenParams.TOP_K: self.top_k,
            GenParams.REPETITION_PENALTY: self.repetition_penalty,
            GenParams.STOP_SEQUENCES: self.stop_sequences
        }

    def instantiate_model(self):

        if self.model_provider == 'watsonx':

            model = WatsonxLLM(
                model_id=self.model_id,
                url=self.model_endpoint,
                project_id=self.project_id,
                params=self.params,
            )

            return model

        elif self.model_provider == 'openai':

            model = ChatOpenAI(
                api_key=self.api_key,
                temperature=self.temperature,
                model=self.model_type
            )

            return model

        else:
            pass
