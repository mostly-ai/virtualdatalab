from virtualdatalab.synthesizers.utils import check_is_fitted, check_common_data_format
from virtualdatalab.logging import getLogger

LOGGER = getLogger(__name__)

class BaseSynthesizer:
    ''' Base class for Synthesizer'''

    def train(self, target_data, **kwargs):
        LOGGER.info(f"Training {self.__class__.__name__}")
        check_common_data_format(target_data)

    def generate(self, number_of_subjects, **kwargs):
        LOGGER.info(f"Generating {self.__class__.__name__}")
        check_is_fitted(self)