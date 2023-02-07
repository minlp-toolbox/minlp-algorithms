# Adrian Buerger, 2022

import logging
import numpy as np
import pandas as pd

from system import System

logger = logging.getLogger(__name__)

class State(System):

    @property
    def x_hat(self):
        x_index = []
        for index in [self.x_index, self.x_aux_index]:
            for x in index:
                try:
                    for j, idx in enumerate(index[x]):
                        x_index.append(x.lower() + "_" + str(j))
                except TypeError:
                    x_index.append(x.lower())
        try:
            state = []
            for idx in x_index:
                state.append(self._data[idx])
            return state
        except AttributeError:
            msg = "System state not available yet, call initialize() first."
            logging.error(msg)
            raise RuntimeError(msg)


    def _get_initial_system_state(self):

        '''
        Generate some generic state data for demonstrational purposes.
        '''

        self._data = {

            't_hts_0': 70.0,
            't_hts_1': 65.0,
            't_hts_2': 63.0,
            't_hts_3': 60.0,
            't_lts': 14.0,
            't_fpsc': 20.0,
            't_fpsc_s': 20.0,
            't_vtsc': 22.0,
            't_vtsc_s': 22.0,
            't_pscf': 18.0,
            't_pscr': 20.0,
            't_shx_psc_0': 11.0,
            't_shx_psc_1': 11.0,
            't_shx_psc_2': 11.0,
            't_shx_psc_3': 11.0,
            't_shx_ssc_0': 32.0,
            't_shx_ssc_1': 32.0,
            't_shx_ssc_2': 32.0,
            't_shx_ssc_3': 32.0,

        }


    def initialize(self):

        self._get_initial_system_state()


if __name__ == "__main__":

    state = State()
    state.initialize()

    print(state.x_hat)

