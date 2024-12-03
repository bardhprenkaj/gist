import random
from copy import deepcopy
import tqdm

from typing import List
from src.core.oracle_base import Oracle
from src.dataset.instances.graph import GraphInstance
from src.explainer.helpers.perturbers.base import GraphPerturber


class DCEPerturber(GraphPerturber):

    def __init__(self, oracle: Oracle, instances: List[GraphInstance]):
        super().__init__(instances)
        self.oracle: Oracle = oracle


    def perturb(self) -> List[GraphInstance]:
        return [self.perturb_instance(instance) for instance in self.instances]
    
    def perturb_instance(self, instance) -> GraphInstance:
        return self.__perturb(self.oracle.predict(instance))

    def __perturb(self, org_label) -> GraphInstance:
        print(f'org_label = {org_label}')
        shuffled = deepcopy(self.instances)
        random.shuffle(shuffled)
        for ctf_candidate in tqdm.tqdm(shuffled):
            if org_label != self.oracle.predict(ctf_candidate):
                return ctf_candidate
        return None
